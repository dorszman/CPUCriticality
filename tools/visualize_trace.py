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
import queue
import re
import sys
import threading
import time

TRACE_PREFIX = "@@ISQCRIT@@"
FIELD_RE = re.compile(r"(\w+)=(\S+)")
HEX_FIELDS = {"pc", "new_pc", "old_pc"}
FLOAT_FIELDS = {"rate"}
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
        elif key in ("table", "event"):
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

    def run(self):
        while not self._stop.is_set():
            # New bytes are queued immediately. Sleep only when the
            # writer has not appended anything yet.
            if self.drain_file_once() == 0:
                if self._stop.wait(self.POLL_INTERVAL):
                    break

    def stop(self):
        self._stop.set()

    def drain_file_once(self):
        """Reads and queues every complete line currently available,
        bounded CHUNK_SIZE bytes at a time. Safe to call directly for an
        initial synchronous seed before the background loop starts."""
        queued = 0
        try:
            with open(self.log_path, "rb") as f:
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
                                self.queue.put(fields)
                                queued += 1
                    if len(chunk) < self.CHUNK_SIZE:
                        break  # caught up to whatever's currently on disk
        except FileNotFoundError:
            pass  # writer hasn't created the file yet; next poll tries again
        return queued

    def drain_available(self):
        """Non-blocking: returns every event currently queued, or []."""
        events = []
        while True:
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
        self.critical_series = []    # prediction table only (0/1)
        self.hitcount_series = []    # prediction table only
        self.promote_cum = []        # cumulative count, step series
        self.demote_cum = []
        self.pc_changes = []         # [(decode, pc_str)] for each alloc event


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
        self.demote_threshold_series = []  # global: positive magnitude
        self.critical_ratio_series = []  # global: [(x, y)]
        # HostTraceStats::ninst, logged as the field ninst= on table=host.
        self.ninst_series = []

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

        self._total_lookups = 0
        self._total_critical = 0

    def feed(self, events):
        for ev in events:
            table = ev.get("table")
            event = ev.get("event")
            decode = ev.get("decode")

            if table == "train" and event == "effective_change":
                self.effective_series.append((decode, ev["value"]))
                continue

            if table == "train" and event == "demote_threshold":
                self.demote_threshold_series.append((decode, ev["value"]))
                continue

            if table == "host" and "ninst" in ev:
                self.ninst_series.append((decode, ev["ninst"]))
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
                if event == "lookup":
                    self._total_lookups += 1
                    if ev.get("critical"):
                        self._total_critical += 1
                    self.critical_ratio_series.append(
                        (decode, self._total_critical / self._total_lookups)
                    )
                    slot = ev.get("slot")
                    if slot is not None and 0 <= slot < NUM_TRACK_SLOTS:
                        sd = self.pred_slots[slot]
                        sd.critical_series.append((decode, ev["critical"]))
                        sd.hitcount_series.append((decode, ev["hitCount"]))
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
                        sd.critical_series.append((decode, 1 if event == "promote" else 0))
                        sd.hitcount_series.append((decode, ev["hitCount"]))

                    train_slot = self._pc_to_train_slot.get(ev["pc"])
                    if train_slot is not None:
                        tsd = self.train_slots[train_slot]
                        if event == "promote":
                            self._train_promote_cum[train_slot] += 1
                            tsd.promote_cum.append((decode, self._train_promote_cum[train_slot]))
                        else:
                            self._train_demote_cum[train_slot] += 1
                            tsd.demote_cum.append((decode, self._train_demote_cum[train_slot]))


def show(log_thread, interval_seconds):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons

    fig = plt.figure(figsize=(10.5, 7))
    fig.suptitle("Criticality predictor trace -- X axis: cumulative decoded instructions")
    fig.subplots_adjust(left=0.14, right=0.88, top=0.93, bottom=0.06, hspace=0.35)

    # Narrower than before (0.72 -> 0.62) to leave room for a second,
    # outward-offset right axis (ninst, a host stat with its own scale
    # unrelated to either existing right axis) without overlapping the
    # radio button column.
    ax_train = fig.add_axes([0.14, 0.55, 0.62, 0.38])
    ax_train_r = ax_train.twinx()
    ax_train_r2 = ax_train.twinx()
    ax_train_r2.spines["right"].set_position(("axes", 1.28))
    ax_pred = fig.add_axes([0.14, 0.07, 0.62, 0.38])
    ax_pred_r = ax_pred.twinx()
    ax_pred_r2 = ax_pred.twinx()
    ax_pred_r2.spines["right"].set_position(("axes", 1.28))

    ax_train_radio = fig.add_axes([0.89, 0.55, 0.1, 0.38], frameon=False)
    ax_pred_radio = fig.add_axes([0.89, 0.07, 0.1, 0.38], frameon=False)
    labels = [f"Slot {i}" for i in range(NUM_TRACK_SLOTS)]
    train_radio = RadioButtons(ax_train_radio, labels)
    pred_radio = RadioButtons(ax_pred_radio, labels)

    dataset = DatasetBuilder()
    dataset.feed(log_thread.drain_available())

    ui = {"train_slot": 0, "pred_slot": 0}
    # True until the user pans or zooms that chart. Home sets both back.
    follow = {"train": True, "pred": True}
    applying = {"n": 0}

    def note_user_zoom(which):
        def on_xlim(_event):
            if applying["n"] == 0:
                follow[which] = False
        return on_xlim

    ax_train.callbacks.connect("xlim_changed", note_user_zoom("train"))
    ax_pred.callbacks.connect("xlim_changed", note_user_zoom("pred"))

    def grow_ylim(ax, hi, bottom=0.0):
        # Expand only. Shrinking the axis on every frame is a jump.
        top = max(hi * 1.1, 1.0)
        lo, cur = ax.get_ylim()
        if cur < top or lo > bottom:
            ax.set_ylim(bottom, max(top, cur))

    def widen(ax, end_x):
        # Twice the samples so far: the trace occupies the left half
        # and the right half is room for the next live batches.
        span = max(end_x, 1)
        applying["n"] += 1
        try:
            ax.set_xlim(0, span * 2)
        finally:
            applying["n"] -= 1

    # Artists are created once. Each line remembers how much of its
    # source series is already on screen and only appends the new tail.
    made = {"ok": False}
    shown = {}

    def push(line, key, slot, series, ymap=None, max_points=2500):
        st = shown.get(key)
        if st is None or st[0] != slot:
            dx, dy, n, ymax = [], [], 0, 0.0
            changed = True
        else:
            _, n, dx, dy, ymax = st
            changed = False
        if n < len(series):
            changed = True
            for i in range(n, len(series)):
                x, y = series[i]
                if ymap is not None:
                    y = ymap(y)
                dx.append(x)
                dy.append(y)
                if y > ymax:
                    ymax = y
            n = len(series)
            if len(dx) > max_points:
                step = max(1, len(dx) // max_points)
                dx = dx[::step]
                dy = dy[::step]
                x, y = series[-1]
                if ymap is not None:
                    y = ymap(y)
                if not dx or dx[-1] != x:
                    dx.append(x)
                    dy.append(y)
        if changed:
            line.set_data(dx, dy)
        shown[key] = (slot, n, dx, dy, ymax)
        return ymax, (dx[-1] if dx else 0)
    marks = {"train_slot": None, "train": [], "pred_slot": None, "pred": []}

    def sync_marks(ax, key, slot_idx, pc_changes):
        if marks[key + "_slot"] != slot_idx:
            for ln in marks[key]:
                ln.remove()
            marks[key] = []
            marks[key + "_slot"] = slot_idx
        have = len(marks[key])
        for decode, _pc in pc_changes[have:]:
            marks[key].append(ax.axvline(decode, color="#ffa726", linestyle="--",
                                          linewidth=1, alpha=0.5))

    def draw_train(slot_idx, preserve_view=False):
        if not made["ok"]:
            made["count"], = ax_train.plot(
                [], [], drawstyle="steps-post", color="#1f77b4", label="count", linewidth=1.5)
            made["demote"], = ax_train.plot(
                [], [], drawstyle="steps-post", color="#e67e22", label="demotion", linewidth=1.5)
            made["thr"], = ax_train.plot(
                [], [], drawstyle="steps-post", color="#444444", linestyle=":",
                label="effective promote threshold", linewidth=1.6)
            made["dthr"], = ax_train.plot(
                [], [], drawstyle="steps-post", color="#e67e22", linestyle=":",
                label="demote threshold", linewidth=1.6)
            made["prom"], = ax_train_r.plot(
                [], [], drawstyle="steps-post", color="#2ca02c", label="cumulative promotes", linewidth=1.5)
            made["demc"], = ax_train_r.plot(
                [], [], drawstyle="steps-post", color="#d62728", label="cumulative demotes", linewidth=1.5)
            made["ninst_t"], = ax_train_r2.plot(
                [], [], color="#9467bd", label="ninst (retired)", linewidth=1.2)
            ax_train.set_xlabel("decoded instructions")
            ax_train.set_ylabel("count / threshold")
            ax_train_r.set_ylabel("cumulative events")
            ax_train_r2.set_ylabel("ninst (retired instructions)")
            h1, l1 = ax_train.get_legend_handles_labels()
            h2, l2 = ax_train_r.get_legend_handles_labels()
            h3, l3 = ax_train_r2.get_legend_handles_labels()
            ax_train.legend(h1 + h2 + h3, l1 + l2 + l3, loc="upper left", fontsize=8)
            ax_train.grid(True, alpha=0.2)
            made["crit"], = ax_pred.plot(
                [], [], drawstyle="steps-post", color="#1f77b4", label="critical bit", linewidth=1.5)
            made["ratio"], = ax_pred.plot(
                [], [], color="#888888", linestyle=":",
                label="overall critical/total ratio", linewidth=1.2)
            made["hits"], = ax_pred_r.plot(
                [], [], drawstyle="steps-post", color="#ff7f0e", label="hitCount", linewidth=1.5)
            made["ninst_p"], = ax_pred_r2.plot(
                [], [], color="#9467bd", label="ninst (retired)", linewidth=1.2)
            ax_pred.set_ylim(-0.05, 1.1)
            ax_pred.set_xlabel("decoded instructions")
            ax_pred.set_ylabel("critical bit / ratio")
            ax_pred_r.set_ylabel("hitCount")
            ax_pred_r2.set_ylabel("ninst (retired instructions)")
            h1, l1 = ax_pred.get_legend_handles_labels()
            h2, l2 = ax_pred_r.get_legend_handles_labels()
            h3, l3 = ax_pred_r2.get_legend_handles_labels()
            ax_pred.legend(h1 + h2 + h3, l1 + l2 + l3, loc="upper left", fontsize=8)
            ax_pred.grid(True, alpha=0.2)
            made["ok"] = True

        sd = dataset.train_slots[slot_idx]
        sync_marks(ax_train, "train", slot_idx, sd.pc_changes)
        cmax, cx = push(made["count"], "count", slot_idx, sd.count_series,
                        ymap=lambda y: y if y > 0 else 0)
        dmax, _dx = push(made["demote"], "demote", slot_idx, sd.count_series,
                         ymap=lambda y: -y if y < 0 else 0)
        tmax, tx = push(made["thr"], "thr", "g", dataset.effective_series)
        dtmax, dtx = push(made["dthr"], "dthr", "g", dataset.demote_threshold_series)
        pmax, px = push(made["prom"], "prom", slot_idx, sd.promote_cum)
        dmmax, dmx = push(made["demc"], "demc", slot_idx, sd.demote_cum)
        nmax, nx = push(made["ninst_t"], "ninst_t", "g", dataset.ninst_series)
        end_x = max(cx, tx, dtx, px, dmx, nx)
        ax_train.set_title(f"Training table -- Slot {slot_idx}")
        grow_ylim(ax_train, max(cmax, dmax, tmax, dtmax))
        grow_ylim(ax_train_r, max(pmax, dmmax))
        if nmax:
            grow_ylim(ax_train_r2, nmax)
        if follow["train"]:
            widen(ax_train, end_x)
        fig.canvas.draw_idle()

    def draw_pred(slot_idx, preserve_view=False):
        if not made["ok"]:
            draw_train(ui["train_slot"])
        sd = dataset.pred_slots[slot_idx]
        sync_marks(ax_pred, "pred", slot_idx, sd.pc_changes)
        _cmax, cx = push(made["crit"], "crit", slot_idx, sd.critical_series)
        _rmax, rx = push(made["ratio"], "ratio", "g", dataset.critical_ratio_series)
        hmax, hx = push(made["hits"], "hits", slot_idx, sd.hitcount_series)
        nmax, nx = push(made["ninst_p"], "ninst_p", "g", dataset.ninst_series)
        ax_pred.set_title(f"Prediction table -- Slot {slot_idx}")
        if hmax:
            grow_ylim(ax_pred_r, hmax)
        if nmax:
            grow_ylim(ax_pred_r2, nmax)
        if follow["pred"]:
            widen(ax_pred, max(cx, rx, hx, nx))
        fig.canvas.draw_idle()

    def on_train_radio(label):
        ui["train_slot"] = int(label.split()[-1])
        draw_train(ui["train_slot"], preserve_view=False)

    def on_pred_radio(label):
        ui["pred_slot"] = int(label.split()[-1])
        draw_pred(ui["pred_slot"], preserve_view=False)

    train_radio.on_clicked(on_train_radio)
    pred_radio.on_clicked(on_pred_radio)

    draw_train(ui["train_slot"], preserve_view=False)
    draw_pred(ui["pred_slot"], preserve_view=False)

    # Paint as soon as the reader has queued new lines. One paint
    # includes every line that arrived since the previous paint, then
    # the next check is scheduled. Nothing waits on a redraw timer.
    gap_ms = 1 if interval_seconds <= 0 else max(1, int(interval_seconds * 1000))

    def pump():
        if not plt.fignum_exists(fig.number):
            return
        new_events = log_thread.drain_available()
        if new_events:
            dataset.feed(new_events)
            draw_train(ui["train_slot"], preserve_view=False)
            draw_pred(ui["pred_slot"], preserve_view=False)
        window = getattr(getattr(fig.canvas, "manager", None), "window", None)
        if window is not None:
            window.after(gap_ms, pump)

    toolbar = getattr(getattr(fig.canvas, "manager", None), "toolbar", None)
    if toolbar is not None and hasattr(toolbar, "home"):
        def home(*args, **kwargs):
            # Back to the widest view: twice the samples that exist now.
            # Later frames keep following the stream until the next zoom.
            follow["train"] = True
            follow["pred"] = True
            draw_train(ui["train_slot"], preserve_view=False)
            draw_pred(ui["pred_slot"], preserve_view=False)

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
