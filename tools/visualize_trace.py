#!/usr/bin/env python3
"""Turns a CriticalityConfig::enableTraceLog log into an interactive
matplotlib window showing training-table and prediction-table behavior
over time -- a native plot window (TkAgg), not a browser page: use the
toolbar's pan/zoom-to-box/home buttons to navigate, and the per-table
radio buttons to switch which of the 8 tracked slots is shown.

Live: a background thread continuously reads new complete lines from
the log file (in bounded chunks, never one unbounded read) and pushes
their parsed events into a thread-safe queue. The GUI's periodic
redraw (default every 1s) only ever drains that queue -- a few
in-memory queue.get_nowait() calls -- and never touches the filesystem
itself, so the GUI can never block on disk I/O or on how large the log
has grown, no matter how slow the reader thread or how bursty the
writer is. Your pan/zoom position is preserved across redraws -- it
only resets when you switch slots. For this to actually show anything
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
        elif key in ("table", "event", "name"):
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
    POLL_INTERVAL = 0.05  # seconds between checks for newly-appended data

    def __init__(self, log_path):
        super().__init__(daemon=True)
        self.log_path = log_path
        self.queue = queue.Queue()
        self.pos = 0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            self.drain_file_once()
            time.sleep(self.POLL_INTERVAL)

    def stop(self):
        self._stop.set()

    def drain_file_once(self):
        """Reads and queues every complete line currently available,
        bounded CHUNK_SIZE bytes at a time. Safe to call directly for an
        initial synchronous seed before the background loop starts."""
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
                    if len(chunk) < self.CHUNK_SIZE:
                        break  # caught up to whatever's currently on disk
        except FileNotFoundError:
            pass  # writer hasn't created the file yet; next poll tries again

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
        # Arbitrary host-reported stats (TokenEngine::TraceLogStat), e.g.
        # "ninst" -- name -> [(x, y)]. Generic on purpose: whatever name
        # the host logs just shows up here, no new wiring needed per stat.
        self.host_stats = {}

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

            if table == "host" and event == "stat":
                self.host_stats.setdefault(ev["name"], []).append((decode, ev["value"]))
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


def _xy(series):
    if not series:
        return [], []
    xs, ys = zip(*series)
    return list(xs), list(ys)


def show(log_thread, interval_seconds):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
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

    def draw_pc_markers(ax, pc_changes):
        for decode, pc in pc_changes:
            ax.axvline(decode, color="#ffa726", linestyle="--", linewidth=1, alpha=0.5)

    def draw_train(slot_idx, preserve_view):
        view = None
        if preserve_view:
            view = (ax_train.get_xlim(), ax_train.get_ylim(), ax_train_r.get_ylim(),
                     ax_train_r2.get_ylim())
        ax_train.clear()
        ax_train_r.clear()
        ax_train_r2.clear()
        sd = dataset.train_slots[slot_idx]

        draw_pc_markers(ax_train, sd.pc_changes)

        xs, raw = _xy(sd.count_series)
        # The counter itself runs negative to reach a demotion. Draw that
        # depth upward so a demotion is a positive number on this axis.
        count_y = [y if y > 0 else 0 for y in raw]
        demote_y = [-y if y < 0 else 0 for y in raw]
        ax_train.step(xs, count_y, where="post", color="#1f77b4", label="count", linewidth=1.5)
        ax_train.step(xs, demote_y, where="post", color="#e67e22", label="demotion", linewidth=1.5)

        end_x = xs[-1] if xs else 0
        for series in (sd.count_series, sd.promote_cum, sd.demote_cum,
                       dataset.effective_series, dataset.demote_threshold_series,
                       dataset.host_stats.get("ninst", [])):
            if series:
                end_x = max(end_x, series[-1][0])

        def stretch(series):
            sx, sy = _xy(series)
            if sx and end_x > sx[-1]:
                sx = list(sx) + [end_x]
                sy = list(sy) + [sy[-1]]
            return sx, sy

        xs, ys = stretch(dataset.effective_series)
        ax_train.step(xs, ys, where="post", color="#444444", linestyle=":",
                       label="effective promote threshold", linewidth=1.6)
        xs, ys = stretch(dataset.demote_threshold_series)
        ax_train.step(xs, ys, where="post", color="#e67e22", linestyle=":",
                       label="demote threshold", linewidth=1.6)

        xs, ys = _xy(sd.promote_cum)
        ax_train_r.step(xs, ys, where="post", color="#2ca02c", label="cumulative promotes", linewidth=1.5)
        xs, ys = _xy(sd.demote_cum)
        ax_train_r.step(xs, ys, where="post", color="#d62728", label="cumulative demotes", linewidth=1.5)

        xs, ys = _xy(dataset.host_stats.get("ninst", []))
        ax_train_r2.plot(xs, ys, color="#9467bd", label="ninst (retired)", linewidth=1.2)

        ax_train.set_title(f"Training table -- Slot {slot_idx}")
        ax_train.set_xlabel("decoded instructions")
        ax_train.set_ylabel("count / threshold")
        ax_train_r.set_ylabel("cumulative events")
        ax_train_r2.set_ylabel("ninst (retired instructions)")
        lines1, labels1 = ax_train.get_legend_handles_labels()
        lines2, labels2 = ax_train_r.get_legend_handles_labels()
        lines3, labels3 = ax_train_r2.get_legend_handles_labels()
        ax_train.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
                         loc="upper left", fontsize=8)
        ax_train.grid(True, alpha=0.2)
        if view is not None:
            ax_train.set_xlim(view[0])
            ax_train.set_ylim(view[1])
            ax_train_r.set_ylim(view[2])
            ax_train_r2.set_ylim(view[3])
        fig.canvas.draw_idle()

    def draw_pred(slot_idx, preserve_view):
        view = None
        if preserve_view:
            view = (ax_pred.get_xlim(), ax_pred.get_ylim(), ax_pred_r.get_ylim(),
                     ax_pred_r2.get_ylim())
        ax_pred.clear()
        ax_pred_r.clear()
        ax_pred_r2.clear()
        sd = dataset.pred_slots[slot_idx]

        draw_pc_markers(ax_pred, sd.pc_changes)

        xs, ys = _xy(sd.critical_series)
        ax_pred.step(xs, ys, where="post", color="#1f77b4", label="critical bit", linewidth=1.5)
        xs, ys = _xy(dataset.critical_ratio_series)
        ax_pred.plot(xs, ys, color="#888888", linestyle=":",
                      label="overall critical/total ratio", linewidth=1.2)
        ax_pred.set_ylim(-0.05, 1.1)

        xs, ys = _xy(sd.hitcount_series)
        ax_pred_r.step(xs, ys, where="post", color="#ff7f0e", label="hitCount", linewidth=1.5)

        xs, ys = _xy(dataset.host_stats.get("ninst", []))
        ax_pred_r2.plot(xs, ys, color="#9467bd", label="ninst (retired)", linewidth=1.2)

        ax_pred.set_title(f"Prediction table -- Slot {slot_idx}")
        ax_pred.set_xlabel("decoded instructions")
        ax_pred.set_ylabel("critical bit / ratio")
        ax_pred_r.set_ylabel("hitCount")
        ax_pred_r2.set_ylabel("ninst (retired instructions)")
        lines1, labels1 = ax_pred.get_legend_handles_labels()
        lines2, labels2 = ax_pred_r.get_legend_handles_labels()
        lines3, labels3 = ax_pred_r2.get_legend_handles_labels()
        ax_pred.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
                        loc="upper left", fontsize=8)
        ax_pred.grid(True, alpha=0.2)
        if view is not None:
            ax_pred.set_xlim(view[0])
            ax_pred.set_ylim(view[1])
            ax_pred_r.set_ylim(view[2])
            ax_pred_r2.set_ylim(view[3])
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

    def refresh(_frame):
        new_events = log_thread.drain_available()  # non-blocking: in-memory queue only
        if not new_events:
            return ()
        dataset.feed(new_events)
        draw_train(ui["train_slot"], preserve_view=True)
        draw_pred(ui["pred_slot"], preserve_view=True)
        return ()

    # Keep a reference on fig itself -- FuncAnimation is only kept alive
    # by whoever holds a reference to it, and a bare local would be
    # garbage-collected (silently stopping the timer) the moment show()
    # returns control to plt.show()'s own event loop.
    fig._live_anim = FuncAnimation(fig, refresh, interval=int(interval_seconds * 1000),
                                    cache_frame_data=False)

    plt.show()
    log_thread.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Live/static viewer for an ISQCrit CriticalityConfig::enableTraceLog log.")
    parser.add_argument("logfile")
    parser.add_argument("--interval", type=float, default=1.0,
                         help="Seconds between GUI redraws while the window is open (default: 1.0). "
                              "The background reader polls the file independently, every 0.05s.")
    args = parser.parse_args()

    log_thread = LogTailThread(args.logfile)
    log_thread.drain_file_once()  # one synchronous pass to seed the initial view
    if log_thread.queue.qsize():
        print(f"Found {log_thread.queue.qsize()} trace lines already in {args.logfile}; "
              f"opening plot window (redrawing every {args.interval:g}s)...")
    else:
        print(f"{args.logfile} has no '{TRACE_PREFIX}' lines yet -- opening plot window and "
              f"waiting for data (redrawing every {args.interval:g}s)...")
    log_thread.start()

    show(log_thread, args.interval)


if __name__ == "__main__":
    main()
