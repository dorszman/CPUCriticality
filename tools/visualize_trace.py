#!/usr/bin/env python3
"""Turns a CriticalityConfig::enableTraceLog log into an interactive HTML
visualization of training-table and prediction-table behavior over time.

The log may be mixed with other, unrelated stdout output -- every line
this engine emits starts with the exact token "@@ISQCRIT@@" (see
kTraceLogPrefix in src/token_engine.cpp), so this script only ever looks
at lines containing that token and ignores everything else.

Usage:
    python3 tools/visualize_trace.py <logfile> [output.html]

Generate a log with tools/trace_demo.cpp:
    g++ -std=c++17 -Isrc tools/trace_demo.cpp src/token_engine.cpp src/token.cpp -o trace_demo
    ./trace_demo > /tmp/trace.log
    python3 tools/visualize_trace.py /tmp/trace.log trace.html
"""
import json
import re
import sys

TRACE_PREFIX = "@@ISQCRIT@@"
FIELD_RE = re.compile(r"(\w+)=(\S+)")
HEX_FIELDS = {"pc", "new_pc", "old_pc"}
FLOAT_FIELDS = {"rate"}
NUM_TRACK_SLOTS = 8


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
        # value series: list of {x: decode, y: value}
        self.count_series = []       # training table only
        self.critical_series = []    # prediction table only (0/1)
        self.hitcount_series = []    # prediction table only
        self.promote_cum = []        # cumulative count, step series
        self.demote_cum = []
        self.pc_changes = []         # [{decode, pc}] for alloc/evict


def build_dataset(events):
    train_slots = {i: SlotData() for i in range(NUM_TRACK_SLOTS)}
    pred_slots = {i: SlotData() for i in range(NUM_TRACK_SLOTS)}
    effective_series = []    # global: [{x: decode, y: value}]
    critical_ratio_series = []  # global: [{x: decode, y: ratio}]

    train_promote_cum = {i: 0 for i in range(NUM_TRACK_SLOTS)}
    train_demote_cum = {i: 0 for i in range(NUM_TRACK_SLOTS)}

    # promote/demote events are logged under table=pred (that's where the
    # write lands), but we want them attributed to whichever TRAINING
    # slot currently holds that PC (promotion is triggered by crossing a
    # training-table threshold). The log doesn't carry the training slot
    # index on those lines, so we track current occupancy of each
    # training slot ourselves, from that same table's own alloc/evict
    # events, processed in the same chronological order as the log.
    pc_to_train_slot = {}

    total_lookups = 0
    total_critical = 0

    for ev in events:
        table = ev.get("table")
        event = ev.get("event")
        decode = ev.get("decode")

        if table == "train" and event == "effective_change":
            effective_series.append({"x": decode, "y": ev["value"]})
            continue

        if table == "train":
            slot = ev.get("slot")
            if slot is None or slot < 0 or slot >= NUM_TRACK_SLOTS:
                continue
            sd = train_slots[slot]
            if event in ("alloc", "evict"):
                new_pc = ev["new_pc"]
                sd.pc_changes.append({"decode": decode, "pc": pc_hex(new_pc)})
                if event == "evict":
                    pc_to_train_slot.pop(ev["old_pc"], None)
                pc_to_train_slot[new_pc] = slot
            elif event == "update":
                sd.count_series.append({"x": decode, "y": ev["count"]})

        elif table == "pred":
            if event == "lookup":
                total_lookups += 1
                if ev.get("critical"):
                    total_critical += 1
                critical_ratio_series.append(
                    {"x": decode, "y": total_critical / total_lookups}
                )
                slot = ev.get("slot")
                if slot is not None and 0 <= slot < NUM_TRACK_SLOTS:
                    sd = pred_slots[slot]
                    sd.critical_series.append({"x": decode, "y": ev["critical"]})
                    sd.hitcount_series.append({"x": decode, "y": ev["hitCount"]})
                continue

            if event in ("alloc", "evict"):
                slot = ev.get("slot")
                if slot is not None and 0 <= slot < NUM_TRACK_SLOTS:
                    pred_slots[slot].pc_changes.append(
                        {"decode": decode, "pc": pc_hex(ev["new_pc"])}
                    )
            elif event in ("promote", "demote"):
                slot = ev.get("slot")
                if slot is not None and 0 <= slot < NUM_TRACK_SLOTS:
                    sd = pred_slots[slot]
                    sd.critical_series.append(
                        {"x": decode, "y": 1 if event == "promote" else 0}
                    )
                    sd.hitcount_series.append({"x": decode, "y": ev["hitCount"]})

                train_slot = pc_to_train_slot.get(ev["pc"])
                if train_slot is not None:
                    tsd = train_slots[train_slot]
                    if event == "promote":
                        train_promote_cum[train_slot] += 1
                        tsd.promote_cum.append({"x": decode, "y": train_promote_cum[train_slot]})
                    else:
                        train_demote_cum[train_slot] += 1
                        tsd.demote_cum.append({"x": decode, "y": train_demote_cum[train_slot]})

    return {
        "train_slots": train_slots,
        "pred_slots": pred_slots,
        "effective_series": effective_series,
        "critical_ratio_series": critical_ratio_series,
    }


PAGE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Criticality predictor trace</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/hammer.js/2.0.8/hammer.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-zoom/2.0.1/chartjs-plugin-zoom.min.js"></script>
<style>
  :root { --bg: #0f1115; --panel: #1a1d24; --text: #e6e6e6; --accent: #4fc3f7;
          --grid: #2a2e38; --promote: #66bb6a; --demote: #ef5350; --marker: #ffa726; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; padding: 16px; }
  h1 { font-size: 18px; margin: 0 0 4px 0; }
  .sub { color: #9aa0ac; font-size: 12px; margin-bottom: 16px; }
  .half { background: var(--panel); border-radius: 10px; padding: 12px; margin-bottom: 16px; }
  .half h2 { margin: 0 0 8px 0; font-size: 15px; }
  .tabs { display: flex; gap: 6px; margin-bottom: 10px; flex-wrap: wrap; }
  .tab-btn { background: #262b35; color: var(--text); border: 1px solid var(--grid);
             border-radius: 6px; padding: 6px 12px; cursor: pointer; font-size: 13px; }
  .tab-btn.active { background: var(--accent); color: #06232b; font-weight: 600; }
  .chart-wrap { position: relative; height: 320px; }
  .legend-note { color: #9aa0ac; font-size: 11px; margin-top: 6px; }
  .zoom-hint { color: #9aa0ac; font-size: 11px; }
  .reset-btn { background: #262b35; color: var(--text); border: 1px solid var(--grid);
               border-radius: 6px; padding: 4px 10px; cursor: pointer; font-size: 12px;
               margin-left: 10px; }
  .chart-toolbar { display: flex; align-items: center; margin-bottom: 6px; }
</style>
</head>
<body>
<h1>Criticality predictor trace</h1>
<div class="sub">X axis: cumulative decoded instructions. Dashed vertical lines mark a slot's occupant PC changing (alloc/evict).</div>

<div class="half">
  <h2>Training table (8 tracked slots)</h2>
  <div class="tabs" id="train-tabs"></div>
  <div class="chart-toolbar">
    <span class="zoom-hint">Scroll/wheel to zoom, click-drag to pan (X axis only).</span>
    <button class="reset-btn" id="train-reset">Reset zoom</button>
  </div>
  <div class="chart-wrap"><canvas id="train-canvas"></canvas></div>
  <div class="legend-note">count = this slot's training counter · cumulative promote/demote per slot · dynamic effective promotion threshold (governor) overlaid on every tab.</div>
</div>

<div class="half">
  <h2>Prediction table (8 tracked slots)</h2>
  <div class="tabs" id="pred-tabs"></div>
  <div class="chart-toolbar">
    <span class="zoom-hint">Scroll/wheel to zoom, click-drag to pan (X axis only).</span>
    <button class="reset-btn" id="pred-reset">Reset zoom</button>
  </div>
  <div class="chart-wrap"><canvas id="pred-canvas"></canvas></div>
  <div class="legend-note">critical bit (0/1, right axis) and hitCount (left axis) for this slot · overall critical/total lookup ratio overlaid on every tab.</div>
</div>

<script>
const DATA = __DATA_JSON__;

Chart.register(window['chartjs-plugin-zoom'] || window.ChartZoom);

const ZOOM_PAN_OPTIONS = {
  pan: { enabled: true, mode: 'x' },
  zoom: {
    wheel: { enabled: true },
    pinch: { enabled: true },
    drag: { enabled: false },
    mode: 'x',
  },
};

function pcMarkerPlugin(pcChanges) {
  return {
    id: 'pcMarkers',
    afterDraw(chart) {
      const xScale = chart.scales.x;
      const ctx = chart.ctx;
      const top = chart.chartArea.top;
      const bottom = chart.chartArea.bottom;
      ctx.save();
      ctx.strokeStyle = '#ffa726';
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      pcChanges.forEach(ch => {
        const xPix = xScale.getPixelForValue(ch.decode);
        if (xPix < chart.chartArea.left || xPix > chart.chartArea.right) return;
        ctx.beginPath();
        ctx.moveTo(xPix, top);
        ctx.lineTo(xPix, bottom);
        ctx.stroke();
      });
      ctx.restore();
    }
  };
}

function makeTrainChart(canvas, slotIdx) {
  const sd = DATA.train_slots[slotIdx];
  return new Chart(canvas, {
    type: 'line',
    data: {
      datasets: [
        { label: 'count', data: sd.count_series, borderColor: '#4fc3f7',
          stepped: true, pointRadius: 0, borderWidth: 2, yAxisID: 'y' },
        { label: 'cumulative promotes', data: sd.promote_cum, borderColor: '#66bb6a',
          stepped: true, pointRadius: 0, borderWidth: 2, yAxisID: 'y2' },
        { label: 'cumulative demotes', data: sd.demote_cum, borderColor: '#ef5350',
          stepped: true, pointRadius: 0, borderWidth: 2, yAxisID: 'y2' },
        { label: 'effective promote threshold', data: DATA.effective_series, borderColor: '#bdbdbd',
          stepped: true, pointRadius: 0, borderWidth: 1, borderDash: [2, 2], yAxisID: 'y' },
      ]
    },
    options: {
      animation: false,
      parsing: false,
      scales: {
        x: { type: 'linear', title: { display: true, text: 'decoded instructions' },
             grid: { color: '#2a2e38' } },
        y: { position: 'left', title: { display: true, text: 'count / threshold' },
             grid: { color: '#2a2e38' } },
        y2: { position: 'right', title: { display: true, text: 'cumulative events' },
              grid: { drawOnChartArea: false } },
      },
      plugins: { legend: { labels: { color: '#e6e6e6' } }, zoom: ZOOM_PAN_OPTIONS },
    },
    plugins: [pcMarkerPlugin(sd.pc_changes)],
  });
}

function makePredChart(canvas, slotIdx) {
  const sd = DATA.pred_slots[slotIdx];
  return new Chart(canvas, {
    type: 'line',
    data: {
      datasets: [
        { label: 'critical bit', data: sd.critical_series, borderColor: '#4fc3f7',
          stepped: true, pointRadius: 0, borderWidth: 2, yAxisID: 'ybit' },
        { label: 'hitCount', data: sd.hitcount_series, borderColor: '#ffb74d',
          stepped: true, pointRadius: 0, borderWidth: 2, yAxisID: 'y' },
        { label: 'overall critical/total ratio', data: DATA.critical_ratio_series, borderColor: '#bdbdbd',
          pointRadius: 0, borderWidth: 1, borderDash: [2, 2], yAxisID: 'ybit' },
      ]
    },
    options: {
      animation: false,
      parsing: false,
      scales: {
        x: { type: 'linear', title: { display: true, text: 'decoded instructions' },
             grid: { color: '#2a2e38' } },
        y: { position: 'right', title: { display: true, text: 'hitCount' },
             grid: { drawOnChartArea: false } },
        ybit: { position: 'left', min: 0, max: 1.05,
                title: { display: true, text: 'critical bit / ratio' },
                grid: { color: '#2a2e38' } },
      },
      plugins: { legend: { labels: { color: '#e6e6e6' } }, zoom: ZOOM_PAN_OPTIONS },
    },
    plugins: [pcMarkerPlugin(sd.pc_changes)],
  });
}

function setupTabs(tabsDiv, canvas, resetBtn, makeChart) {
  let chart = null;
  function render(idx) {
    [...tabsDiv.children].forEach((btn, i) => btn.classList.toggle('active', i === idx));
    if (chart) chart.destroy();
    chart = makeChart(canvas, idx);
  }
  for (let i = 0; i < 8; i++) {
    const btn = document.createElement('button');
    btn.className = 'tab-btn';
    btn.textContent = 'Slot ' + i;
    btn.onclick = () => render(i);
    tabsDiv.appendChild(btn);
  }
  resetBtn.onclick = () => { if (chart) chart.resetZoom(); };
  render(0);
}

setupTabs(document.getElementById('train-tabs'), document.getElementById('train-canvas'),
          document.getElementById('train-reset'), makeTrainChart);
setupTabs(document.getElementById('pred-tabs'), document.getElementById('pred-canvas'),
          document.getElementById('pred-reset'), makePredChart);
</script>
</body>
</html>
"""


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    log_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "trace_viz.html"

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

    dataset = build_dataset(events)

    json_data = {
        "train_slots": {
            str(i): dataset["train_slots"][i].__dict__ for i in range(NUM_TRACK_SLOTS)
        },
        "pred_slots": {
            str(i): dataset["pred_slots"][i].__dict__ for i in range(NUM_TRACK_SLOTS)
        },
        "effective_series": dataset["effective_series"],
        "critical_ratio_series": dataset["critical_ratio_series"],
    }
    # JS object keys from json.dumps come out as strings ("0","1",...);
    # index with DATA.train_slots[slotIdx] in JS still works since JS
    # coerces the numeric index to the same string key.

    page = PAGE_TEMPLATE.replace("__DATA_JSON__", json.dumps(json_data))
    with open(out_path, "w") as f:
        f.write(page)

    print(f"Parsed {len(events)} trace lines -> {out_path}")


if __name__ == "__main__":
    main()
