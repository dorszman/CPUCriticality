#!/usr/bin/env python3
"""Turns a CriticalityConfig::enableTraceLog log into an interactive
matplotlib window showing training-table and prediction-table behavior
over time -- a native plot window (TkAgg), not a browser page: use the
toolbar's pan/zoom-to-box/home buttons to navigate, and the per-table
radio buttons to switch which of the 8 tracked slots is shown.

The log may be mixed with other, unrelated stdout output -- every line
this engine emits starts with the exact token "@@ISQCRIT@@" (see
kTraceLogPrefix in src/token_engine.cpp), so this script only ever looks
at lines containing that token and ignores everything else.

Usage:
    python3 tools/visualize_trace.py <logfile>

Generate a log with tools/trace_demo.cpp:
    g++ -std=c++17 -Isrc tools/trace_demo.cpp src/token_engine.cpp src/token.cpp -o trace_demo
    ./trace_demo > /tmp/trace.log
    python3 tools/visualize_trace.py /tmp/trace.log

Requires matplotlib with an interactive backend (e.g. `apt install
python3-matplotlib python3-tk` on Debian/Ubuntu).
"""
import re
import sys

TRACE_PREFIX = "@@ISQCRIT@@"
FIELD_RE = re.compile(r"(\w+)=(\S+)")
HEX_FIELDS = {"pc", "new_pc", "old_pc"}
FLOAT_FIELDS = {"rate"}
NUM_TRACK_SLOTS = 8
# Matches SetAssociativeTable::kNoEvictionTag in src/token_engine.cpp --
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


def build_dataset(events):
    train_slots = {i: SlotData() for i in range(NUM_TRACK_SLOTS)}
    pred_slots = {i: SlotData() for i in range(NUM_TRACK_SLOTS)}
    effective_series = []       # global: [(x, y)]
    critical_ratio_series = []  # global: [(x, y)]

    train_promote_cum = {i: 0 for i in range(NUM_TRACK_SLOTS)}
    train_demote_cum = {i: 0 for i in range(NUM_TRACK_SLOTS)}

    # promote/demote events are logged under table=pred (that's where the
    # write lands), but we want them attributed to whichever TRAINING
    # slot currently holds that PC (promotion is triggered by crossing a
    # training-table threshold). The log doesn't carry the training slot
    # index on those lines, so we track current occupancy of each
    # training slot ourselves, from that same table's own alloc
    # events, processed in the same chronological order as the log.
    pc_to_train_slot = {}

    total_lookups = 0
    total_critical = 0

    for ev in events:
        table = ev.get("table")
        event = ev.get("event")
        decode = ev.get("decode")

        if table == "train" and event == "effective_change":
            effective_series.append((decode, ev["value"]))
            continue

        if table == "train":
            slot = ev.get("slot")
            if slot is None or slot < 0 or slot >= NUM_TRACK_SLOTS:
                continue
            sd = train_slots[slot]
            if event == "alloc":
                new_pc = ev["new_pc"]
                sd.pc_changes.append((decode, pc_hex(new_pc)))
                old_pc = ev["old_pc"]
                if old_pc != NO_EVICTION_TAG:
                    pc_to_train_slot.pop(old_pc, None)
                pc_to_train_slot[new_pc] = slot
            elif event == "update":
                sd.count_series.append((decode, ev["count"]))

        elif table == "pred":
            if event == "lookup":
                total_lookups += 1
                if ev.get("critical"):
                    total_critical += 1
                critical_ratio_series.append((decode, total_critical / total_lookups))
                slot = ev.get("slot")
                if slot is not None and 0 <= slot < NUM_TRACK_SLOTS:
                    sd = pred_slots[slot]
                    sd.critical_series.append((decode, ev["critical"]))
                    sd.hitcount_series.append((decode, ev["hitCount"]))
                continue

            if event == "alloc":
                slot = ev.get("slot")
                if slot is not None and 0 <= slot < NUM_TRACK_SLOTS:
                    pred_slots[slot].pc_changes.append((decode, pc_hex(ev["new_pc"])))
            elif event in ("promote", "demote"):
                slot = ev.get("slot")
                if slot is not None and 0 <= slot < NUM_TRACK_SLOTS:
                    sd = pred_slots[slot]
                    sd.critical_series.append((decode, 1 if event == "promote" else 0))
                    sd.hitcount_series.append((decode, ev["hitCount"]))

                train_slot = pc_to_train_slot.get(ev["pc"])
                if train_slot is not None:
                    tsd = train_slots[train_slot]
                    if event == "promote":
                        train_promote_cum[train_slot] += 1
                        tsd.promote_cum.append((decode, train_promote_cum[train_slot]))
                    else:
                        train_demote_cum[train_slot] += 1
                        tsd.demote_cum.append((decode, train_demote_cum[train_slot]))

    return {
        "train_slots": train_slots,
        "pred_slots": pred_slots,
        "effective_series": effective_series,
        "critical_ratio_series": critical_ratio_series,
    }


def _xy(series):
    if not series:
        return [], []
    xs, ys = zip(*series)
    return list(xs), list(ys)


def show(dataset):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons

    fig = plt.figure(figsize=(13, 9))
    fig.suptitle("Criticality predictor trace -- X axis: cumulative decoded instructions")
    fig.subplots_adjust(left=0.14, right=0.88, top=0.93, bottom=0.06, hspace=0.35)

    ax_train = fig.add_axes([0.14, 0.55, 0.72, 0.38])
    ax_train_r = ax_train.twinx()
    ax_pred = fig.add_axes([0.14, 0.07, 0.72, 0.38])
    ax_pred_r = ax_pred.twinx()

    ax_train_radio = fig.add_axes([0.89, 0.55, 0.1, 0.38], frameon=False)
    ax_pred_radio = fig.add_axes([0.89, 0.07, 0.1, 0.38], frameon=False)
    labels = [f"Slot {i}" for i in range(NUM_TRACK_SLOTS)]
    train_radio = RadioButtons(ax_train_radio, labels)
    pred_radio = RadioButtons(ax_pred_radio, labels)

    def draw_pc_markers(ax, pc_changes):
        for decode, pc in pc_changes:
            ax.axvline(decode, color="#ffa726", linestyle="--", linewidth=1, alpha=0.5)

    def draw_train(slot_idx):
        ax_train.clear()
        ax_train_r.clear()
        sd = dataset["train_slots"][slot_idx]

        draw_pc_markers(ax_train, sd.pc_changes)

        xs, ys = _xy(sd.count_series)
        ax_train.step(xs, ys, where="post", color="#1f77b4", label="count", linewidth=1.5)
        xs, ys = _xy(dataset["effective_series"])
        ax_train.step(xs, ys, where="post", color="#888888", linestyle=":",
                       label="effective promote threshold", linewidth=1.2)

        xs, ys = _xy(sd.promote_cum)
        ax_train_r.step(xs, ys, where="post", color="#2ca02c", label="cumulative promotes", linewidth=1.5)
        xs, ys = _xy(sd.demote_cum)
        ax_train_r.step(xs, ys, where="post", color="#d62728", label="cumulative demotes", linewidth=1.5)

        ax_train.set_title(f"Training table -- Slot {slot_idx}")
        ax_train.set_xlabel("decoded instructions")
        ax_train.set_ylabel("count / threshold")
        ax_train_r.set_ylabel("cumulative events")
        lines1, labels1 = ax_train.get_legend_handles_labels()
        lines2, labels2 = ax_train_r.get_legend_handles_labels()
        ax_train.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
        ax_train.grid(True, alpha=0.2)
        fig.canvas.draw_idle()

    def draw_pred(slot_idx):
        ax_pred.clear()
        ax_pred_r.clear()
        sd = dataset["pred_slots"][slot_idx]

        draw_pc_markers(ax_pred, sd.pc_changes)

        xs, ys = _xy(sd.critical_series)
        ax_pred.step(xs, ys, where="post", color="#1f77b4", label="critical bit", linewidth=1.5)
        xs, ys = _xy(dataset["critical_ratio_series"])
        ax_pred.plot(xs, ys, color="#888888", linestyle=":",
                      label="overall critical/total ratio", linewidth=1.2)
        ax_pred.set_ylim(-0.05, 1.1)

        xs, ys = _xy(sd.hitcount_series)
        ax_pred_r.step(xs, ys, where="post", color="#ff7f0e", label="hitCount", linewidth=1.5)

        ax_pred.set_title(f"Prediction table -- Slot {slot_idx}")
        ax_pred.set_xlabel("decoded instructions")
        ax_pred.set_ylabel("critical bit / ratio")
        ax_pred_r.set_ylabel("hitCount")
        lines1, labels1 = ax_pred.get_legend_handles_labels()
        lines2, labels2 = ax_pred_r.get_legend_handles_labels()
        ax_pred.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
        ax_pred.grid(True, alpha=0.2)
        fig.canvas.draw_idle()

    train_radio.on_clicked(lambda label: draw_train(int(label.split()[-1])))
    pred_radio.on_clicked(lambda label: draw_pred(int(label.split()[-1])))

    draw_train(0)
    draw_pred(0)

    plt.show()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    log_path = sys.argv[1]

    events = []
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            fields = parse_line(line)
            if fields is not None:
                events.append(fields)

    if not events:
        print(f"No '{TRACE_PREFIX}' lines found in {log_path} -- "
              f"was CriticalityConfig::enableTraceLog set?", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(events)} trace lines, opening plot window...")
    dataset = build_dataset(events)
    show(dataset)


if __name__ == "__main__":
    main()
