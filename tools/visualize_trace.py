#!/usr/bin/env python3
"""Turns a CriticalityConfig::enableTraceLog log into an interactive
matplotlib window showing training-table and prediction-table behavior
over time -- a native plot window (TkAgg), not a browser page. The
x-axis starts at twice the samples seen so far. The toolbar zoom
and pan work; they hold until Home, which returns to that widest
view. The per-table radio buttons switch
which of the 8 tracked slots is shown.

Live: a background thread continuously reads new complete lines from
the log file (in bounded chunks, never one unbounded read) and pushes
their parsed events into a thread-safe queue. The GUI drains that
queue on the Tk event loop -- a few in-memory queue.get_nowait() calls
-- and never touches the filesystem itself, so the GUI can never block
on disk I/O or on how large the log has grown. As soon as new lines are
queued they are appended and painted; there is no redraw timer. Lines
are created once. A redraw appends only the samples
that arrived since the previous frame, and sets the x-axis to twice
the samples seen so far, so the trace fills the left half and the
right half stays empty for what arrives next. It does not clear the
axes, rebuild the legend, or walk the history again.
Zoom and pan hold until Home, which snaps back to twice the samples
seen at that moment and follows the stream again.
Switching slots points the same lines at that slot's series.
For this to actually show anything
live, each trace line is fflush()'d individually by the writer (see
PredictorConfig::traceLogStream in src/criticality_predictor.h) --
otherwise the OS wouldn't see new bytes until the writer's internal
buffer fills or it exits.

The log may be mixed with other, unrelated output on the same stream --
every line this predictor emits starts with the exact token
"@@ISQCRIT@@" (see kTraceLogPrefix in src/criticality_predictor.cpp),
so this script only ever looks at lines containing that token and
ignores everything else.

Usage:
    python3 tools/visualize_trace.py <logfile> [--interval SECONDS]

Generate a log with tools/trace_demo.cpp:
    g++ -std=c++17 -Isrc tools/trace_demo.cpp src/token_engine.cpp src/criticality_predictor.cpp src/token.cpp -o trace_demo
    ./trace_demo > /tmp/trace.log &
    python3 tools/visualize_trace.py /tmp/trace.log

Requires matplotlib with an interactive backend (e.g. `apt install
python3-matplotlib python3-tk` on Debian/Ubuntu).
"""
import argparse
import os
import queue
import re
import sys
import threading
import time

TRACE_PREFIX = "@@ISQCRIT@@"
# Queued when the log file is removed or replaced. The charts drop the
# previous run and read the new file from the start.
RUN_RESET = object()
FIELD_RE = re.compile(r"(\w+)=(\S+)")
HEX_FIELDS = {"pc", "new_pc", "old_pc"}
FLOAT_FIELDS = {
    "rate", "targetCriticalRate", "criticalRateDeadband",
    "criticalRatio", "nonCriticalRatio", "normalRatio",
}
STRING_FIELDS = {
    "table", "event", "path",
    "trainingTableEvictionPolicy", "predictionTableEvictionPolicy",
}
# docs/DESIGN.md §6.1. One control and one chart each.
LIVE_FIELDS = [
    ("enablePrediction", "bool"),
    ("criticalDistanceCycles", "int"),
    ("nonCriticalDistanceCycles", "int"),
    ("promoteCriticalAt", "int"),
    ("promoteNonCriticalAt", "int"),
    ("trainCounterMin", "int"),
    ("trainCounterMax", "int"),
    ("enableCriticalRateGovernor", "bool"),
    ("targetCriticalRate", "float"),
    ("criticalRateDeadband", "float"),
    ("criticalRateControlInterval", "int"),
    ("criticalRateRelaxDwellWindows", "int"),
    ("trainingTableEvictionPolicy", "policy"),
    ("predictionTableEvictionPolicy", "policy"),
    ("trainingTableDecayIdleUnit", "int"),
    ("predictionTableDecayIdleUnit", "int"),
]
POLICIES = ("kLru", "kDecayedHitCount", "kDecayedPriorityThenLru")
# Editor starting values. Same numbers as PredictorLiveConfig. The
# panel always opens here; it does not copy the last snapshot out of
# the log.
LIVE_DEFAULTS = {
    "enablePrediction": True,
    "criticalDistanceCycles": "2",
    "nonCriticalDistanceCycles": "8",
    "promoteCriticalAt": "5",
    "promoteNonCriticalAt": "-3",
    "trainCounterMin": "-10",
    "trainCounterMax": "40",
    "enableCriticalRateGovernor": True,
    "targetCriticalRate": "0.10",
    "criticalRateDeadband": "0.01",
    "criticalRateControlInterval": "1024",
    "criticalRateRelaxDwellWindows": "4",
    "trainingTableEvictionPolicy": "kLru",
    "predictionTableEvictionPolicy": "kDecayedPriorityThenLru",
    "trainingTableDecayIdleUnit": "128",
    "predictionTableDecayIdleUnit": "128",
}
NUM_TRACK_SLOTS = 8
# Matches SetAssociativeTable::kNoEvictionTag in src/criticality_predictor.cpp --
# "no real eviction happened" is never 0 (0 can be a real PC), it's
# always all-ones for the engine's 64-bit Addr type.
NO_EVICTION_TAG = (1 << 64) - 1


def parse_line(line):
    """Returns a dict of fields for one trace line, or None if this line
    isn't one of ours (the whole point of the prefix: ignore noise)."""
    idx = line.find(TRACE_PREFIX)
    if idx < 0:
        return None
    rest = line[idx + len(TRACE_PREFIX):]
    fields = {}
    for key, value in FIELD_RE.findall(rest):
        if key in HEX_FIELDS:
            fields[key] = int(value, 16)
        elif key in FLOAT_FIELDS:
            fields[key] = float(value)
        elif key in STRING_FIELDS:
            fields[key] = value
        else:
            fields[key] = int(value)
    return fields


def pc_hex(v):
    return "0x%x" % v


class LogTailThread(threading.Thread):
    """Background reader: tracks a byte offset into log_path and
    continuously pushes newly-completed lines' parsed events into a
    thread-safe queue, entirely off the GUI thread.

    Two things keep this from ever being the thing that blocks:
    - It reads in bounded CHUNK_SIZE pieces, not one unbounded read, so
      even catching up on a large backlog (e.g. after a slow poll)
      pushes events to the queue incrementally rather than only after
      reading the whole backlog.
    - It's the GUI that drains the queue, via non-blocking
      queue.get_nowait() calls (drain_available()) -- the GUI thread
      never calls open()/read() itself, so disk latency or log size can
      never stall rendering.

    A trailing line with no newline yet (the writer is mid-line) is
    simply left unconsumed and picked up whole, combined with whatever
    follows, once it's complete.
    """

    CHUNK_SIZE = 1 << 16  # 64 KiB per read -- bounds each individual read's cost
    POLL_INTERVAL = 0.001  # only used when the file has nothing new

    def __init__(self, log_path):
        super().__init__(daemon=True)
        self.log_path = log_path
        self.queue = queue.Queue()
        self.pos = 0
        self._stop = threading.Event()
        # inode of the file we are following, and its opening bytes. A
        # new run removes that file and creates another, or truncates
        # it; either one means the bytes already queued belong to the
        # previous run.
        self.identity = None
        self.prefix = None
        self._fast_lock = threading.Lock()
        self.live_pending = []
        self.control_path_fast = None

    def run(self):
        while not self._stop.is_set():
            # New bytes are queued immediately. Sleep only when the
            # writer has not appended anything yet.
            if self.drain_file_once() == 0:
                if self._stop.wait(self.POLL_INTERVAL):
                    break

    def stop(self):
        self._stop.set()

    def _note_fast(self, fields):
        # Live config must show up when it is written, not after the
        # chart has replayed every earlier training line.
        table = fields.get("table")
        if table == "live":
            with self._fast_lock:
                self.live_pending.append(fields)
        elif table == "control" and fields.get("path"):
            with self._fast_lock:
                self.control_path_fast = fields["path"]

    def take_live(self):
        with self._fast_lock:
            items = self.live_pending
            self.live_pending = []
            path = self.control_path_fast
        return items, path

    def clear_fast(self):
        with self._fast_lock:
            self.live_pending = []
            self.control_path_fast = None

    def _discard_queued(self):
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def drain_file_once(self):
        """Reads and queues every complete line currently available,
        bounded CHUNK_SIZE bytes at a time. Safe to call directly for an
        initial synchronous seed before the background loop starts.
        A removed, replaced, or shortened file queues RUN_RESET and is
        read from the start."""
        queued = 0
        try:
            with open(self.log_path, "rb") as f:
                st = os.fstat(f.fileno())
                identity = (st.st_dev, st.st_ino)
                # The first lines of a new run carry a new socket path.
                # Comparing them catches a replaced file whose inode was
                # reused and which is already longer than the old offset.
                head = f.read(256) if st.st_size else b""
                content_changed = bool(self.prefix) and head != self.prefix[:len(head)]
                replaced = self.identity is not None and (
                    identity != self.identity or st.st_size < self.pos or content_changed)
                self.identity = identity
                if head:
                    self.prefix = head
                if replaced:
                    self._discard_queued()
                    self.clear_fast()
                    self.pos = 0
                    self.queue.put(RUN_RESET)
                    queued += 1
                f.seek(self.pos)
                pending = b""
                while True:
                    chunk = f.read(self.CHUNK_SIZE)
                    if not chunk:
                        break
                    pending += chunk
                    last_newline = pending.rfind(b"\n")
                    if last_newline >= 0:
                        complete, pending = pending[:last_newline + 1], pending[last_newline + 1:]
                        self.pos += len(complete)
                        for line in complete.decode("utf-8", errors="replace").splitlines():
                            fields = parse_line(line)
                            if fields is not None:
                                self._note_fast(fields)
                                self.queue.put(fields)
                                queued += 1
                    if len(chunk) < self.CHUNK_SIZE:
                        break  # caught up to whatever's currently on disk
        except FileNotFoundError:
            # Removed, and the new file is not there yet. One reset,
            # then wait. Repeating this while the path is still missing
            # would clear the chart on every poll.
            if self.identity is not None:
                self.identity = None
                self.prefix = None
                self.pos = 0
                self.clear_fast()
                self._discard_queued()
                self.queue.put(RUN_RESET)
                return 1
        return queued

    def drain_available(self):
        """Non-blocking: returns every event currently queued, or []."""
        return self.drain_some(None)

    def drain_some(self, limit):
        """Non-blocking. limit is the most events to take; None takes all.
        A live frame paints one batch and leaves the rest queued, so a
        slow run advances on screen instead of appearing all at once."""
        events = []
        while limit is None or len(events) < limit:
            try:
                events.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return events


class SlotData:
    """Everything needed to draw one physical slot's panel: the thing
    being watched (per-table-type), the PC-change vertical markers, and
    cumulative promote/demote counters."""

    def __init__(self):
        # value series: list of (x, y) pairs
        self.count_series = []       # training table only
        # Prediction table: running tallies of samples whose critical bit
        # was 1 or 0. Both only increase. Replaces the old 0/1 step.
        self.critical_count_series = []
        self.noncritical_count_series = []
        # Prediction table: hitCount accumulated so a new allocation
        # (logged counter drops) adds onto the total instead of resetting it.
        self.hitcount_acc_series = []
        self.promote_cum = []        # cumulative count, step series
        self.demote_cum = []
        self.pc_changes = []         # [(decode, pc_str)] for each alloc event
        self._critical_count = 0
        self._noncritical_count = 0
        self._hit_acc = 0
        self._hit_last = 0


class DatasetBuilder:
    """Incremental dataset accumulator: holds all running state (slot
    data, cumulative counters, the pc->training-slot occupancy map)
    across calls, so feed() only ever processes the events handed to it
    THIS time -- the whole point of LogTailThread only ever returning
    what's new. feed([]) is a cheap no-op."""

    def __init__(self):
        self.train_slots = {i: SlotData() for i in range(NUM_TRACK_SLOTS)}
        self.pred_slots = {i: SlotData() for i in range(NUM_TRACK_SLOTS)}
        self.effective_series = []       # global: [(x, y)]
        self.demote_threshold_series = []  # global: signed level the counter crosses
        self.critical_ratio_series = []  # governor window rate, one point when that window closes
        # Distance-class window (table=train event=distance_class).
        # Global, like the effective threshold: one point per completed
        # window, class count over the window total.
        self.critical_total_series = []
        self.noncritical_total_series = []
        self.normal_total_series = []
        # HostTraceStats::ninst, logged as the field ninst= on table=host.
        self.ninst_series = []
        # Live-knob channel (PredictorLiveConfig). control_path is the
        # socket key. live_series is the numeric trace of each knob.
        # live_latest is the last complete snapshot, in wire form.
        self.control_path = None
        self.live_series = {name: [] for name, _kind in LIVE_FIELDS}
        self.live_lo = {name: None for name, _kind in LIVE_FIELDS}
        self.live_hi = {name: None for name, _kind in LIVE_FIELDS}
        self.live_latest = None

        self._train_promote_cum = {i: 0 for i in range(NUM_TRACK_SLOTS)}
        self._train_demote_cum = {i: 0 for i in range(NUM_TRACK_SLOTS)}

        # promote/demote events are logged under table=pred (that's
        # where the write lands), but we want them attributed to
        # whichever TRAINING slot currently holds that PC (promotion is
        # triggered by crossing a training-table threshold). The log
        # doesn't carry the training slot index on those lines, so we
        # track current occupancy of each training slot ourselves, from
        # that same table's own alloc events, processed in the same
        # chronological order as the log.
        self._pc_to_train_slot = {}
        self._live_seen = set()

    def reset(self):
        """Drop every series. The log file was removed or replaced."""
        self.__init__()

    def _note_pred_sample(self, sd, decode, critical, hit_count):
        """One prediction-table sample for this slot. Counts step by one
        and never fall. A rising hitCount adds only the increase; a drop
        (new allocation) adds the new logged value on top of the total."""
        if critical:
            sd._critical_count += 1
        else:
            sd._noncritical_count += 1
        sd.critical_count_series.append((decode, sd._critical_count))
        sd.noncritical_count_series.append((decode, sd._noncritical_count))
        if hit_count >= sd._hit_last:
            sd._hit_acc += hit_count - sd._hit_last
        else:
            sd._hit_acc += hit_count
        sd._hit_last = hit_count
        sd.hitcount_acc_series.append((decode, sd._hit_acc))

    def feed(self, events):
        for ev in events:
            table = ev.get("table")
            event = ev.get("event")
            decode = ev.get("decode")

            if table == "train" and event == "effective_change":
                self.effective_series.append((decode, ev["value"]))
                continue

            if table == "train" and event == "demote_threshold":
                # The counter demotes at a negative level. Older logs
                # stored that level as a positive magnitude.
                value = ev["value"]
                if value > 0:
                    value = -value
                self.demote_threshold_series.append((decode, value))
                continue

            if table == "train" and event == "distance_class":
                critical = ev.get("criticalRatio")
                noncritical = ev.get("nonCriticalRatio")
                normal = ev.get("normalRatio")
                if critical is not None and noncritical is not None and normal is not None:
                    self.critical_total_series.append((decode, critical))
                    self.noncritical_total_series.append((decode, noncritical))
                    self.normal_total_series.append((decode, normal))
                continue

            if table == "host" and "ninst" in ev:
                self.ninst_series.append((decode, ev["ninst"]))
                continue

            if table == "control" and "path" in ev:
                self.control_path = ev["path"]
                continue

            if table == "live":
                if decode in self._live_seen:
                    continue
                self._live_seen.add(decode)
                snap = {}
                complete = True
                for name, kind in LIVE_FIELDS:
                    if name not in ev:
                        complete = False
                        continue
                    raw = ev[name]
                    if kind == "policy":
                        y = POLICIES.index(raw) if raw in POLICIES else 0
                        snap[name] = raw
                    else:
                        y = raw
                        snap[name] = raw
                    self.live_series[name].append((decode, y))
                    lo = self.live_lo[name]
                    hi = self.live_hi[name]
                    self.live_lo[name] = y if lo is None else min(lo, y)
                    self.live_hi[name] = y if hi is None else max(hi, y)
                if complete:
                    self.live_latest = snap
                continue

            if table == "train":
                slot = ev.get("slot")
                if slot is None or slot < 0 or slot >= NUM_TRACK_SLOTS:
                    continue
                sd = self.train_slots[slot]
                if event == "alloc":
                    new_pc = ev["new_pc"]
                    sd.pc_changes.append((decode, pc_hex(new_pc)))
                    old_pc = ev["old_pc"]
                    if old_pc != NO_EVICTION_TAG:
                        self._pc_to_train_slot.pop(old_pc, None)
                    self._pc_to_train_slot[new_pc] = slot
                elif event == "update":
                    sd.count_series.append((decode, ev["count"]))

            elif table == "pred":
                if event == "criticality":
                    self.critical_ratio_series.append((decode, ev["rate"]))
                    continue
                if event == "lookup":
                    slot = ev.get("slot")
                    if slot is not None and 0 <= slot < NUM_TRACK_SLOTS:
                        sd = self.pred_slots[slot]
                        self._note_pred_sample(sd, decode, ev["critical"], ev["hitCount"])
                    continue

                if event == "alloc":
                    slot = ev.get("slot")
                    if slot is not None and 0 <= slot < NUM_TRACK_SLOTS:
                        self.pred_slots[slot].pc_changes.append(
                            (decode, pc_hex(ev["new_pc"]))
                        )
                elif event in ("promote", "demote"):
                    slot = ev.get("slot")
                    if slot is not None and 0 <= slot < NUM_TRACK_SLOTS:
                        sd = self.pred_slots[slot]
                        self._note_pred_sample(
                            sd, decode, 1 if event == "promote" else 0, ev["hitCount"])

                    train_slot = self._pc_to_train_slot.get(ev["pc"])
                    if train_slot is not None:
                        tsd = self.train_slots[train_slot]
                        if event == "promote":
                            self._train_promote_cum[train_slot] += 1
                            tsd.promote_cum.append((decode, self._train_promote_cum[train_slot]))
                        else:
                            self._train_demote_cum[train_slot] += 1
                            tsd.demote_cum.append((decode, self._train_demote_cum[train_slot]))


def attach_live_controls(fig, dataset):
    """Editors for the §6.1 knobs, packed on the right of the plot
    window. A change sends the whole set over the socket path the model
    wrote into the trace."""
    import socket as sockmod
    import tkinter as tk

    root = getattr(getattr(fig.canvas, "manager", None), "window", None)
    if root is None:
        return lambda: None
    canvas_widget = fig.canvas.get_tk_widget()
    toolbar = getattr(getattr(fig.canvas, "manager", None), "toolbar", None)
    if toolbar is not None:
        toolbar.pack_forget()
    canvas_widget.pack_forget()

    win = tk.Frame(root, bd=1, relief="groove")
    win.pack(side="right", fill="y")
    if toolbar is not None:
        toolbar.pack(side="bottom", fill="x")
    canvas_widget.pack(side="left", fill="both", expand=True)

    tk.Label(win, text="Change live knobs", anchor="w",
             font=("TkDefaultFont", 11, "bold")).pack(fill="x", padx=8, pady=(8, 0))
    status = tk.StringVar(value="waiting for the model key")
    tk.Label(win, textvariable=status, anchor="w").pack(fill="x", padx=8, pady=4)
    holder = tk.Frame(win)
    holder.pack(fill="both", expand=True)
    scroll = tk.Canvas(holder, highlightthickness=0, width=410)
    bar = tk.Scrollbar(holder, orient="vertical", command=scroll.yview)
    body = tk.Frame(scroll)
    body.bind("<Configure>", lambda _e: scroll.configure(scrollregion=scroll.bbox("all")))
    scroll.create_window((0, 0), window=body, anchor="nw")
    scroll.configure(yscrollcommand=bar.set)
    bar.pack(side="right", fill="y")
    scroll.pack(side="left", fill="both", expand=True)

    vars_ = {}
    loading = {"on": False}
    sock = {"fd": None, "path": None, "tried": 0.0}
    pending = {"id": None}

    def _same_snapshot(values, snap):
        for name, kind in LIVE_FIELDS:
            got = values[name]
            have = snap[name]
            if kind == "float":
                if abs(float(got) - float(have)) > 1e-9:
                    return False
            elif kind == "policy":
                if got != have:
                    return False
            elif int(got) != int(have):
                return False
        return True

    def current_values():
        out = {}
        for name, kind in LIVE_FIELDS:
            raw = vars_[name].get()
            if kind == "bool":
                out[name] = 1 if raw else 0
            elif kind == "int":
                out[name] = int(raw)
            elif kind == "float":
                out[name] = float(raw)
            else:
                out[name] = raw
        return out

    def format_line(values):
        parts = []
        for name, kind in LIVE_FIELDS:
            value = values[name]
            if kind == "float":
                parts.append(f"{name}={float(value):.6f}")
            else:
                parts.append(f"{name}={value}")
        return " ".join(parts) + "\n"

    def load_from(snap):
        loading["on"] = True
        try:
            for name, kind in LIVE_FIELDS:
                value = snap[name]
                if kind == "bool":
                    vars_[name].set(bool(int(value)))
                elif kind == "float":
                    vars_[name].set(f"{float(value):.6f}")
                else:
                    vars_[name].set(value if kind == "policy" else str(int(value)))
        finally:
            loading["on"] = False

    def send_now():
        pending["id"] = None
        if loading["on"] or sock["fd"] is None:
            return
        try:
            values = current_values()
            if dataset.live_latest and _same_snapshot(values, dataset.live_latest):
                return
            line = format_line(values)
        except ValueError:
            status.set("rejected")
            if dataset.live_latest:
                load_from(dataset.live_latest)
            return
        try:
            sock["fd"].sendall(line.encode())
            buf = b""
            while b"\n" not in buf:
                chunk = sock["fd"].recv(64)
                if not chunk:
                    raise ConnectionError("closed")
                buf += chunk
            reply = buf.split(b"\n", 1)[0].decode()
        except OSError:
            try:
                sock["fd"].close()
            except OSError:
                pass
            sock["fd"] = None
            status.set("disconnected")
            return
        status.set(reply)
        if reply == "rejected" and dataset.live_latest:
            load_from(dataset.live_latest)

    def schedule_send(*_args):
        if loading["on"]:
            return
        if pending["id"] is not None:
            win.after_cancel(pending["id"])
        pending["id"] = win.after(150, send_now)

    for name, kind in LIVE_FIELDS:
        row = tk.Frame(body)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=name, width=34, anchor="w").pack(side="left")
        if kind == "bool":
            var = tk.BooleanVar(value=LIVE_DEFAULTS[name])
            tk.Checkbutton(row, variable=var, command=schedule_send).pack(side="left")
        elif kind == "policy":
            var = tk.StringVar(value=LIVE_DEFAULTS[name])
            tk.OptionMenu(row, var, *POLICIES, command=lambda _v: schedule_send()).pack(
                side="left", fill="x", expand=True)
        else:
            var = tk.StringVar(value=LIVE_DEFAULTS[name])
            entry = tk.Entry(row, textvariable=var, width=18)
            entry.pack(side="left")
            entry.bind("<Return>", schedule_send)
            entry.bind("<FocusOut>", schedule_send)
        vars_[name] = var

    tk.Button(win, text="Send knobs", command=schedule_send).pack(
        side="bottom", fill="x", padx=8, pady=8)

    def poll_connection():
        path = dataset.control_path
        if sock["fd"] is not None and path == sock["path"]:
            return
        if sock["fd"] is not None:
            try:
                sock["fd"].close()
            except OSError:
                pass
            sock["fd"] = None
        if not path:
            return
        now = time.time()
        if now < sock["tried"]:
            return
        sock["tried"] = now + 0.5
        try:
            fd = sockmod.socket(sockmod.AF_UNIX, sockmod.SOCK_STREAM)
            fd.settimeout(2.0)
            fd.connect(path)
        except OSError:
            status.set("waiting for " + path)
            return
        sock["fd"] = fd
        sock["path"] = path
        status.set("connected")

    def close():
        if sock["fd"] is not None:
            try:
                sock["fd"].close()
            except OSError:
                pass
            sock["fd"] = None

    fig._close_live = close
    return poll_connection


def show(log_thread, interval_seconds):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons

    fig = plt.figure(figsize=(11, 12))
    fig.suptitle("Criticality predictor trace -- X axis: cumulative decoded instructions")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.96, bottom=0.03)

    # Charts share one x-axis, same left edge and width so a sample
    # lines up vertically. Zoom or pan on any chart moves all of them.
    # The signed counter has its own chart. The criticality table is
    # the 0..1 chart: distance-class shares and Criticality, labeled
    # every tenth. The bottom chart is every live knob: one row each.
    # A change is labeled with the new value, drawn on the step.
    # Room on the right so the outer scale (ninst) sits past the inner
    # one. A new PC in a slot is a tick on that chart's x-axis, not a line
    # through the trace.
    plot_left, plot_width = 0.28, 0.44
    ax_count = fig.add_axes([plot_left, 0.835, plot_width, 0.115])
    ax_count.grid(True, alpha=0.2)
    ax_train = fig.add_axes([plot_left, 0.700, plot_width, 0.115], sharex=ax_count)
    ax_train_r = ax_train.twinx()
    ax_class = fig.add_axes([plot_left, 0.540, plot_width, 0.145], sharex=ax_count)
    ax_class.set_ylabel("fraction")
    ax_class.set_title("Criticality table")
    ax_class.grid(True, alpha=0.2)
    ax_class.autoscale(enable=False)

    def lock_criticality_axis():
        ax_class.set_ylim(0.0, 1.0)
        ax_class.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax_class.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        ax_class.tick_params(axis="y", labelsize=7)

    lock_criticality_axis()
    ax_pred = fig.add_axes([plot_left, 0.395, plot_width, 0.135], sharex=ax_count)
    ax_pred_r = ax_pred.twinx()
    ax_pred_r2 = ax_pred.twinx()
    ax_pred_r2.spines["right"].set_position(("axes", 1.38))
    ax_live = fig.add_axes([plot_left, 0.04, plot_width, 0.32], sharex=ax_count)
    ax_live.set_title("Live knobs")
    ax_live.set_xlabel("decoded instructions")
    ax_live.grid(True, alpha=0.2)

    ax_train_radio = fig.add_axes([0.93, 0.80, 0.07, 0.15], frameon=False)
    ax_pred_radio = fig.add_axes([0.93, 0.395, 0.07, 0.135], frameon=False)
    labels = [f"Slot {i}" for i in range(NUM_TRACK_SLOTS)]
    train_radio = RadioButtons(ax_train_radio, labels)
    pred_radio = RadioButtons(ax_pred_radio, labels)

    dataset = DatasetBuilder()
    # The backlog stays in the queue. Each frame below takes a bounded
    # batch, so the window opens immediately and the trace grows.

    ui = {"train_slot": 0, "pred_slot": 0}
    # One x-axis for all three charts. True until the user pans or
    # zooms; Home turns it back on and snaps every chart to twice the
    # samples seen then.
    follow = {"x": True}
    seen_x = {"v": 0}
    applying = {"n": 0}

    def note_user_zoom(_event):
        if applying["n"] == 0:
            follow["x"] = False

    ax_count.callbacks.connect("xlim_changed", note_user_zoom)
    ax_train.callbacks.connect("xlim_changed", note_user_zoom)
    ax_class.callbacks.connect("xlim_changed", note_user_zoom)
    ax_pred.callbacks.connect("xlim_changed", note_user_zoom)
    ax_live.callbacks.connect("xlim_changed", note_user_zoom)

    def format_knob(kind, value):
        if kind == "policy":
            index = int(value)
            if 0 <= index < len(POLICIES):
                return POLICIES[index]
        if kind == "float":
            return f"{float(value):.4g}"
        if kind == "bool":
            return "1" if int(value) else "0"
        return str(int(value))

    def grow_ylim(ax, hi, bottom=0.0):
        # Expand only. Shrinking the axis on every frame is a jump.
        top = max(hi * 1.1, 1.0)
        lo, cur = ax.get_ylim()
        if cur < top or lo > bottom:
            ax.set_ylim(bottom, max(top, cur))

    def grow_ylim_signed(ax, lo, hi):
        # The training counter is signed, so the axis grows below zero
        # as well as above. Expand only.
        pad_hi = max(hi * 1.1, 1.0)
        pad_lo = min(lo * 1.1, -1.0) if lo < 0 else 0.0
        cur_lo, cur_hi = ax.get_ylim()
        new_lo = min(cur_lo, pad_lo)
        new_hi = max(cur_hi, pad_hi)
        if new_lo < cur_lo or new_hi > cur_hi:
            ax.set_ylim(new_lo, new_hi)

    def widen_shared(end_x):
        # Twice the samples so far, on the shared x-axis: the trace
        # occupies the left half and the right half is room for what
        # arrives next. Setting one axes moves the other two.
        if end_x > seen_x["v"]:
            seen_x["v"] = end_x
        if not follow["x"]:
            return
        span = max(seen_x["v"], 1)
        applying["n"] += 1
        try:
            ax_count.set_xlim(0, span * 2)
        finally:
            applying["n"] -= 1

    # Artists are created once. Each line remembers how much of its
    # source series is already on screen and only appends the new tail.
    made = {"ok": False}
    shown = {}

    def _keep_peaks(dx, dy):
        # Fold the line in place so it stays bounded, but keep the high
        # and low sample of each group. A plain stride dropped those
        # peaks, so the top of the line sank as the run grew.
        out_x = []
        out_y = []
        i = 0
        n = len(dx)
        while i < n:
            j = min(i + 4, n)
            xs = dx[i:j]
            ys = dy[i:j]
            if len(ys) <= 2:
                out_x.extend(xs)
                out_y.extend(ys)
            else:
                low = min(range(len(ys)), key=ys.__getitem__)
                high = max(range(len(ys)), key=ys.__getitem__)
                for k in sorted({low, high}):
                    out_x.append(xs[k])
                    out_y.append(ys[k])
            i = j
        if out_x and out_x[0] != dx[0]:
            out_x.insert(0, dx[0])
            out_y.insert(0, dy[0])
        if out_x and out_x[-1] != dx[-1]:
            out_x.append(dx[-1])
            out_y.append(dy[-1])
        return out_x, out_y

    def push(line, key, slot, series, ymap=None):
        st = shown.get(key)
        if st is None or st[0] != slot:
            dx, dy, n, ymin, ymax = [], [], 0, 0.0, 0.0
            changed = True
        else:
            _, n, dx, dy, ymin, ymax = st
            changed = False
        if n < len(series):
            changed = True
            for i in range(n, len(series)):
                x, y = series[i]
                if ymap is not None:
                    y = ymap(y)
                dx.append(x)
                dy.append(y)
                if y < ymin:
                    ymin = y
                if y > ymax:
                    ymax = y
            n = len(series)
            while len(dx) > 4000:
                before = len(dx)
                dx, dy = _keep_peaks(dx, dy)
                if len(dx) >= before:
                    break
        if changed:
            line.set_data(dx, dy)
        shown[key] = (slot, n, dx, dy, ymin, ymax)
        return ymin, ymax, (dx[-1] if dx else 0)

    def hold_to(line, key, end_x):
        # A threshold that is not changing still has to reach the current
        # sample, or it disappears off the left as the shared axis grows.
        # The extra point is only for this paint; the stored series stays
        # the real events.
        st = shown.get(key)
        if not st:
            return
        _slot, _n, dx, dy, _ymin, _ymax = st
        if not dx or dx[-1] >= end_x:
            return
        line.set_data(list(dx) + [end_x], list(dy) + [dy[-1]])
    marks = {"train_slot": None, "train": None, "pred_slot": None, "pred": None,
             "train_n": -1, "pred_n": -1}

    def sync_marks(ax, key, slot_idx, pc_changes):
        # One artist for every allocation on this slot: a tick on the
        # x-axis. A line through the plot hides the trace.
        if marks[key + "_slot"] != slot_idx or marks[key] is None:
            old = marks[key]
            if old is not None:
                old.remove()
            from matplotlib.transforms import blended_transform_factory
            line, = ax.plot([], [], linestyle="none", marker="|",
                            markersize=10, markeredgewidth=1.4,
                            color="#ffa726", clip_on=False, zorder=10)
            line.set_transform(blended_transform_factory(ax.transData, ax.transAxes))
            marks[key] = line
            marks[key + "_slot"] = slot_idx
            marks[key + "_n"] = -1
        n = len(pc_changes)
        if marks[key + "_n"] == n:
            return
        marks[key].set_data([x for x, _pc in pc_changes], [0.0] * n)
        marks[key + "_n"] = n

    def label_signed(ax, key, series, color):
        # Write the signed level on the line, once at the start and
        # again whenever it changes.
        labels = made.setdefault(key + "_labels", [])
        scanned = made.get(key + "_scanned", 0)

        def add(x, y):
            labels.append(ax.text(x, y, str(int(y)), color=color, fontsize=8,
                                  va="bottom", ha="left", clip_on=True))

        begin = scanned
        if begin == 0 and series:
            add(series[0][0], series[0][1])
            begin = 1
        for j in range(max(begin, 1), len(series)):
            if series[j][1] != series[j - 1][1]:
                add(series[j][0], series[j][1])
        made[key + "_scanned"] = len(series)

    def draw_train(slot_idx, preserve_view=False):
        if not made["ok"]:
            made["count"], = ax_count.plot(
                [], [], drawstyle="steps-post", color="#1f77b4", label="counter", linewidth=1.5)
            made["thr"], = ax_count.plot(
                [], [], drawstyle="steps-post", color="#444444", linestyle=":",
                label="effective promote threshold", linewidth=1.6)
            made["dthr"], = ax_count.plot(
                [], [], drawstyle="steps-post", color="#e67e22", linestyle=":",
                label="demote threshold", linewidth=1.6)
            made["cmin"], = ax_count.plot(
                [], [], drawstyle="steps-post", color="#7f8c8d", linestyle=":",
                label="counter minimum", linewidth=1.4)
            made["prom"], = ax_train.plot(
                [], [], drawstyle="steps-post", color="#2ca02c", label="cumulative promotes", linewidth=1.5)
            made["demc"], = ax_train.plot(
                [], [], drawstyle="steps-post", color="#d62728", label="cumulative demotes", linewidth=1.5)
            made["ninst_t"], = ax_train_r.plot(
                [], [], color="#9467bd", label="ninst (retired)", linewidth=1.2)
            made["crit_tot"], = ax_class.plot(
                [], [], drawstyle="steps-post", color="#e74c3c",
                label="Critical/Total", linewidth=1.5)
            made["noncrit_tot"], = ax_class.plot(
                [], [], drawstyle="steps-post", color="#1abc9c",
                label="NonCritical/Total", linewidth=1.5)
            made["normal_tot"], = ax_class.plot(
                [], [], drawstyle="steps-post", color="#34495e",
                label="Normal/Total", linewidth=1.5)
            made["ratio"], = ax_class.plot(
                [], [], color="#6c3483", linestyle="--",
                label="Criticality", linewidth=1.8)
            ax_class.legend(loc="upper right", fontsize=7)
            ax_count.set_ylabel("counter")
            h1, l1 = ax_count.get_legend_handles_labels()
            ax_count.legend(h1, l1, loc="upper left", fontsize=7)
            ax_train.set_ylabel("cumulative events")
            ax_train_r.set_ylabel("ninst (retired instructions)")
            h1, l1 = ax_train.get_legend_handles_labels()
            h2, l2 = ax_train_r.get_legend_handles_labels()
            ax_train.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
            ax_train.grid(True, alpha=0.2)
            made["crit"], = ax_pred.plot(
                [], [], drawstyle="steps-post", color="#1f77b4",
                label="cumulative critical count", linewidth=1.5)
            made["noncrit"], = ax_pred.plot(
                [], [], drawstyle="steps-post", color="#2ca02c",
                label="cumulative non-critical count", linewidth=1.5)
            made["hits"], = ax_pred_r.plot(
                [], [], drawstyle="steps-post", color="#ff7f0e", label="cumulative hit count", linewidth=1.5)
            made["ninst_p"], = ax_pred_r2.plot(
                [], [], color="#9467bd", label="ninst (retired)", linewidth=1.2)
            ax_pred.set_xlabel("decoded instructions")
            ax_pred.set_ylabel("cumulative critical / non-critical count")
            ax_pred_r.set_ylabel("cumulative hit count")
            ax_pred_r2.set_ylabel("ninst (retired instructions)")
            h1, l1 = ax_pred.get_legend_handles_labels()
            h2, l2 = ax_pred_r.get_legend_handles_labels()
            h3, l3 = ax_pred_r2.get_legend_handles_labels()
            ax_pred.legend(h1 + h2 + h3, l1 + l2 + l3, loc="upper left", fontsize=8)
            ax_pred.grid(True, alpha=0.2)
            made["ok"] = True

        sd = dataset.train_slots[slot_idx]
        sync_marks(ax_count, "train", slot_idx, sd.pc_changes)
        clo, chi, cx = push(made["count"], "count", slot_idx, sd.count_series)
        tlo, thi, tx = push(made["thr"], "thr", "g", dataset.effective_series)
        dlo, dhi, dtx = push(made["dthr"], "dthr", "g", dataset.demote_threshold_series)
        label_signed(ax_count, "dthr", dataset.demote_threshold_series, made["dthr"].get_color())
        mlo, mhi, mtx = push(made["cmin"], "cmin", "g", dataset.live_series["trainCounterMin"])
        label_signed(ax_count, "cmin", dataset.live_series["trainCounterMin"], made["cmin"].get_color())
        _plo, pmax, px = push(made["prom"], "prom", slot_idx, sd.promote_cum)
        _dlo, dmmax, dmx = push(made["demc"], "demc", slot_idx, sd.demote_cum)
        _nlo, nmax, nx = push(made["ninst_t"], "ninst_t", "g", dataset.ninst_series)
        end_x = max(cx, tx, dtx, mtx, px, dmx, nx, seen_x["v"])
        hold_to(made["thr"], "thr", end_x)
        hold_to(made["dthr"], "dthr", end_x)
        hold_to(made["cmin"], "cmin", end_x)
        ax_count.set_title(f"Training, selected slot, counter -- Slot {slot_idx}")
        ax_train.set_title(f"Training, selected slot -- Slot {slot_idx}")
        grow_ylim_signed(ax_count, min(clo, tlo, dlo, mlo), max(chi, thi, dhi, mhi))
        grow_ylim(ax_train, max(pmax, dmmax))
        if nmax:
            grow_ylim(ax_train_r, nmax)
        widen_shared(end_x)

    def draw_class():
        if not made["ok"]:
            draw_train(ui["train_slot"])
        _clo, _crmax, crx = push(made["crit_tot"], "crit_tot", "g", dataset.critical_total_series)
        _nclo, _nrmax, nrx = push(made["noncrit_tot"], "noncrit_tot", "g",
                                  dataset.noncritical_total_series)
        _mlo, _mrmax, mrx = push(made["normal_tot"], "normal_tot", "g", dataset.normal_total_series)
        _rlo, _rmax, rx = push(made["ratio"], "ratio", "g", dataset.critical_ratio_series)
        end_x = max(crx, nrx, mrx, rx, seen_x["v"])
        hold_to(made["crit_tot"], "crit_tot", end_x)
        hold_to(made["noncrit_tot"], "noncrit_tot", end_x)
        hold_to(made["normal_tot"], "normal_tot", end_x)
        hold_to(made["ratio"], "ratio", end_x)
        lock_criticality_axis()
        widen_shared(end_x)

    def draw_pred(slot_idx, preserve_view=False):
        if not made["ok"]:
            draw_train(ui["train_slot"])
        sd = dataset.pred_slots[slot_idx]
        sync_marks(ax_pred, "pred", slot_idx, sd.pc_changes)
        _clo, cmax, cx = push(made["crit"], "crit", slot_idx, sd.critical_count_series)
        _nclo, ncmax, ncx = push(made["noncrit"], "noncrit", slot_idx, sd.noncritical_count_series)
        _hlo, hmax, hx = push(made["hits"], "hits", slot_idx, sd.hitcount_acc_series)
        _nlo, nmax, nx = push(made["ninst_p"], "ninst_p", "g", dataset.ninst_series)
        ax_pred.set_title(f"Prediction table -- Slot {slot_idx}")
        grow_ylim(ax_pred, max(cmax, ncmax))
        if hmax:
            grow_ylim(ax_pred_r, hmax)
        if nmax:
            grow_ylim(ax_pred_r2, nmax)
        widen_shared(max(cx, ncx, hx, nx))

    def draw_knobs():
        if "knobs" not in made:
            made["knobs"] = []
            colors = plt.cm.tab20.colors
            for i, (name, _kind) in enumerate(LIVE_FIELDS):
                line, = ax_live.plot([], [], drawstyle="steps-post",
                                      color=colors[i % len(colors)], linewidth=1.4)
                made["knobs"].append((name, line))
            ax_live.set_ylim(-0.5, len(LIVE_FIELDS) - 0.5)
            made["knob_scanned"] = [0] * len(LIVE_FIELDS)
            made["knob_now"] = []
            for i, (_name, line) in enumerate(made["knobs"]):
                made["knob_now"].append(ax_live.text(
                    0, i + 0.08, "", color=line.get_color(), fontsize=8,
                    va="bottom", ha="right", clip_on=True))
        end_x = 0
        if "knob_ticks" not in made:
            ax_live.set_yticks(range(len(LIVE_FIELDS)))
            ax_live.set_yticklabels([name for name, _kind in LIVE_FIELDS], fontsize=7)
            made["knob_ticks"] = True
        for i, (name, line) in enumerate(made["knobs"]):
            series = dataset.live_series[name]
            lo = dataset.live_lo[name]
            hi = dataset.live_hi[name]
            xs = []
            ys = []
            for x, y in series:
                xs.append(x)
                end_x = max(end_x, x)
                if lo is None or hi is None or lo == hi:
                    ys.append(i)
                else:
                    ys.append(i - 0.35 + 0.7 * (y - lo) / (hi - lo))
            line.set_data(xs, ys)
            kind = LIVE_FIELDS[i][1]
            scanned = made["knob_scanned"][i]
            begin = scanned
            if begin == 0 and series:
                ax_live.text(series[0][0], i + 0.08, format_knob(kind, series[0][1]),
                             color=line.get_color(), fontsize=7, va="bottom", ha="left",
                             clip_on=True)
                begin = 1
            for j in range(max(begin, 1), len(series)):
                if series[j][1] != series[j - 1][1]:
                    ax_live.text(series[j][0], i + 0.08, format_knob(kind, series[j][1]),
                                 color=line.get_color(), fontsize=7, va="bottom", ha="left",
                                 clip_on=True)
            made["knob_scanned"][i] = len(series)
            # A knob that is not changing still has to reach the current
            # sample. One point at the start is invisible once the axis
            # has grown, so the chart looked dead.
            span = max(end_x, seen_x["v"])
            if xs and xs[-1] < span:
                xs.append(span)
                ys.append(ys[-1])
                line.set_data(xs, ys)
            if series:
                now = made["knob_now"][i]
                now.set_position((span if xs else series[-1][0], i + 0.08))
                now.set_text(format_knob(kind, series[-1][1]))
        for tick, (_name, line) in zip(ax_live.get_yticklabels(), made["knobs"]):
            tick.set_color(line.get_color())
        widen_shared(max(end_x, seen_x["v"]))

    def paint():
        # draw() puts the frame on screen before the next batch is read.
        # draw_idle() was getting starved: a 1 ms timer always had more
        # samples, so the window never painted.
        fig.canvas.draw()

    def on_train_radio(label):
        ui["train_slot"] = int(label.split()[-1])
        draw_train(ui["train_slot"], preserve_view=False)
        paint()

    def on_pred_radio(label):
        ui["pred_slot"] = int(label.split()[-1])
        draw_pred(ui["pred_slot"], preserve_view=False)
        paint()

    train_radio.on_clicked(on_train_radio)
    pred_radio.on_clicked(on_pred_radio)

    draw_train(ui["train_slot"], preserve_view=False)
    draw_class()
    draw_pred(ui["pred_slot"], preserve_view=False)
    draw_knobs()
    poll_live = attach_live_controls(fig, dataset)

    # One paint shows every line read since the previous paint, then
    # yields so the window can display it and take clicks. 40 ms is
    # about 25 frames a second; faster than that never reached the screen.
    gap_ms = 40 if interval_seconds <= 0 else max(40, int(interval_seconds * 1000))

    def clear_charts():
        dataset.reset()
        shown.clear()
        seen_x["v"] = 0
        follow["x"] = True
        for key in ("train", "pred"):
            if marks[key] is not None:
                marks[key].remove()
            marks[key] = None
            marks[key + "_slot"] = None
            marks[key + "_n"] = -1
        for txt in list(ax_live.texts):
            txt.remove()
        made["knob_now"] = []
        if "knobs" in made:
            for i, (_name, line) in enumerate(made["knobs"]):
                made["knob_now"].append(ax_live.text(
                    0, i + 0.08, "", color=line.get_color(), fontsize=8,
                    va="bottom", ha="right", clip_on=True))
        if "knob_scanned" in made:
            made["knob_scanned"] = [0] * len(LIVE_FIELDS)
        for txt in made.get("dthr_labels", []):
            txt.remove()
        made["dthr_labels"] = []
        made["dthr_scanned"] = 0
        for txt in made.get("cmin_labels", []):
            txt.remove()
        made["cmin_labels"] = []
        made["cmin_scanned"] = 0
        for ax in (ax_count, ax_train, ax_train_r, ax_pred, ax_pred_r, ax_pred_r2):
            ax.set_ylim(0.0, 1.0)
        lock_criticality_axis()

    def pump():
        if not plt.fignum_exists(fig.number):
            return
        poll_live()
        # Live config is taken from the reader before the backlog. A
        # change is drawn on this frame, not after every older line.
        lives, path = log_thread.take_live()
        if path and not dataset.control_path:
            dataset.control_path = path
        new_events = log_thread.drain_some(8000)
        if new_events and RUN_RESET in new_events:
            last = max(i for i, ev in enumerate(new_events) if ev is RUN_RESET)
            clear_charts()
            log_thread.clear_fast()
            new_events = new_events[last + 1:]
            lives, path = log_thread.take_live()
            if path:
                dataset.control_path = path
        if lives:
            dataset.feed(lives)
        if new_events:
            dataset.feed(new_events)
        if lives or new_events:
            if new_events:
                draw_train(ui["train_slot"], preserve_view=False)
                draw_class()
                draw_pred(ui["pred_slot"], preserve_view=False)
            draw_knobs()
            paint()
        window = getattr(getattr(fig.canvas, "manager", None), "window", None)
        if window is not None:
            window.after(gap_ms, pump)

    toolbar = getattr(getattr(fig.canvas, "manager", None), "toolbar", None)
    if toolbar is not None and hasattr(toolbar, "home"):
        def home(*args, **kwargs):
            # Back to the widest view: twice the samples that exist now.
            # Later frames keep following the stream until the next zoom.
            follow["x"] = True
            draw_train(ui["train_slot"], preserve_view=False)
            draw_class()
            draw_pred(ui["pred_slot"], preserve_view=False)
            draw_knobs()
            paint()

        toolbar.home = home
        buttons = getattr(toolbar, "_buttons", {})
        # The toolbar captured the original bound method when its
        # buttons were built, so replacing the attribute is not enough.
        home_button = buttons.get("Home")
        if home_button is not None:
            home_button.configure(command=home)

    window = getattr(getattr(fig.canvas, "manager", None), "window", None)
    if window is not None:
        window.after(gap_ms, pump)

    plt.show()
    closer = getattr(fig, "_close_live", None)
    if closer is not None:
        closer()
    log_thread.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Live/static viewer for an ISQCrit CriticalityConfig::enableTraceLog log.")
    parser.add_argument("logfile")
    parser.add_argument("--interval", type=float, default=0.0,
                         help="Minimum seconds between paints. 0 (default) paints as soon as "
                              "new lines have been read.")
    args = parser.parse_args()

    log_thread = LogTailThread(args.logfile)
    log_thread.drain_file_once()  # one synchronous pass to seed the initial view
    if log_thread.queue.qsize():
        print(f"Found {log_thread.queue.qsize()} trace lines already in {args.logfile}; "
              f"opening plot window (drawing each new line as it arrives)...")
    else:
        print(f"{args.logfile} has no '{TRACE_PREFIX}' lines yet -- opening plot window and "
              f"waiting for data (drawing each new line as it arrives)...")
    log_thread.start()

    show(log_thread, args.interval)


if __name__ == "__main__":
    main()
