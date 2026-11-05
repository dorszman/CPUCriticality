// Synthetic workload driver for visualizing training/prediction table
// behavior over time (see tools/visualize_trace.py). Not part of the
// library's public API or test suite -- a debugging/visualization aid
// only, built around CriticalityConfig::enableTraceLog.
//
// Shape: both tables are configured as exactly 8 entries (ways=1,
// entriesLog2=3), so the 8 physical slots ARE the whole table -- every
// slot is "interesting" to watch. A sequence of PC "generations" cycles
// through, 8 distinct PCs per generation, all hashing into the SAME 8
// sets as the previous generation, so each new generation necessarily
// evicts the previous one once it arrives. Within a generation, each
// PC is first driven with several critical-distance edges (pushing it
// toward promotion), then several non-critical-distance edges (pushing
// it toward demotion) before the next generation evicts it -- so each
// tracked slot's lifetime shows alloc -> promote -> demote -> alloc (by
// the next generation, evicting this one).
//
// Build: g++ -std=c++17 -Isrc tools/trace_demo.cpp src/token_engine.cpp src/criticality_predictor.cpp src/token.cpp -o trace_demo
//
// Run, trace lines on the demo's own stdout (fine here since this demo
// prints nothing else):
//   ./trace_demo
//
// Run with the predictor opening a dedicated file. It truncates that
// file itself, so a new run does not keep the previous one:
//   ./trace_demo /tmp/trace.log
//
// Either way: python3 tools/visualize_trace.py /tmp/trace.log

#include "token_engine.h"

#include <cstdio>
#include <random>

using namespace ISQCrit;

namespace {

// ninst: a stand-in for the host model's own retired-instruction
// counter. The engine does not track retirement; the host writes the
// field by name (HostTraceStats::ninst). Advances by more than 1 per
// edge to stand in for the instructions around each measured
// producer-consumer edge that this synthetic workload does not model.
void DoEdge(TokenEngine& engine, Addr pc, Cycle& t, std::int64_t distance, std::uint64_t& ninst) {
    TokenRef producer = engine.DecodeGetToken(pc);
    engine.Tick(t);
    producer.NotifyExecuted(); // roots
    TokenRef consumer = producer;
    producer.Reset();
    t += distance;
    engine.Tick(t);
    consumer.NotifyExecuted(); // closes, trains at the given distance
    consumer.Reset();
    ++t;
    engine.IsPredictedCritical(pc); // fetch-time lookup, interleaved

    ninst += 3;
    HostTraceStats host;
    host.ninst = static_cast<std::int64_t>(ninst);
    engine.TraceLogHostStats(host);
}

} // namespace

int main(int argc, char** argv) {
    CriticalityConfig cfg;
    cfg.tokenAssignInterval = 1;
    cfg.predictor.criticalDistanceCycles = 2;
    cfg.predictor.nonCriticalDistanceCycles = 8;

    // Exactly 8 slots per table -- the whole table IS the 8 tracked
    // entries, guaranteeing every slot sees real activity.
    cfg.predictor.trainingTableEntriesLog2 = 3;
    cfg.predictor.trainingTableWays = 1;
    cfg.predictor.predictionTableEntriesLog2 = 3;
    cfg.predictor.predictionTableWays = 1;

    // Real headroom above the floor -- see DESIGN.md Sec 3.8: 1-2 steps
    // of headroom makes the governor overshoot badly regardless of
    // dwell tuning. repsDist below goes up to 30, so a PC's critical
    // phase (half its reps) can be up to 15 reps long -- trainCounterMax
    // needs to clear THAT, not just be "a few steps above the floor",
    // or the effective threshold hits its own ceiling before it's
    // actually strict enough to filter most PCs out.
    cfg.predictor.promoteCriticalAt = 10;
    cfg.predictor.promoteNonCriticalAt = -10;
    cfg.predictor.trainCounterMin = -40;
    cfg.predictor.trainCounterMax = 40;

    cfg.predictor.enableCriticalRateGovernor = true;
    cfg.predictor.targetCriticalRate = 0.10;
    cfg.predictor.criticalRateDeadband = 0.02;
    cfg.predictor.criticalRateControlInterval = 64;
    cfg.predictor.criticalRateRelaxDwellWindows = 4;

    cfg.predictor.enableTraceLog = true;
    if (argc > 1) {
        // The predictor opens this path with truncation when it starts.
        cfg.predictor.traceLogPath = argv[1];
    }

    TokenEngine engine(cfg);

    std::mt19937 rng(7);
    std::uniform_int_distribution<int> repsDist(15, 30);

    const int kGenerations = 40;
    Cycle t = 0;
    std::uint64_t ninst = 0;
    for (int gen = 0; gen < kGenerations; ++gen) {
        for (int slot = 0; slot < 8; ++slot) {
            // 4-byte-aligned PC whose low (log2(8)=3) set-index bits
            // equal 'slot', but whose upper bits are unique to this
            // generation -- same set every generation, new tag.
            Addr pc = (static_cast<Addr>(gen) << 16) | (static_cast<Addr>(slot) << 2);

            int reps = repsDist(rng);
            int criticalReps = reps / 2;
            for (int r = 0; r < reps; ++r) {
                std::int64_t distance = (r < criticalReps) ? 1 : 12;
                DoEdge(engine, pc, t, distance, ninst);
            }
        }
    }

    return 0;
}
