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
// Build: g++ -std=c++17 -Isrc tools/trace_demo.cpp src/token_engine.cpp src/token.cpp -o trace_demo
// Run:   ./trace_demo > /tmp/trace.log
//        python3 tools/visualize_trace.py /tmp/trace.log trace.html

#include "token_engine.h"

#include <cstdio>
#include <random>

using namespace ISQCrit;

namespace {

void DoEdge(TokenEngine& engine, Addr pc, Cycle& t, std::int64_t distance) {
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
}

} // namespace

int main() {
    CriticalityConfig cfg;
    cfg.tokenAssignInterval = 1;
    cfg.criticalDistanceCycles = 2;
    cfg.nonCriticalDistanceCycles = 8;

    // Exactly 8 slots per table -- the whole table IS the 8 tracked
    // entries, guaranteeing every slot sees real activity.
    cfg.trainingTableEntriesLog2 = 3;
    cfg.trainingTableWays = 1;
    cfg.predictionTableEntriesLog2 = 3;
    cfg.predictionTableWays = 1;

    // Real headroom above the floor -- see DESIGN.md Sec 3.8: 1-2 steps
    // of headroom makes the governor overshoot badly regardless of
    // dwell tuning. repsDist below goes up to 30, so a PC's critical
    // phase (half its reps) can be up to 15 reps long -- trainCounterMax
    // needs to clear THAT, not just be "a few steps above the floor",
    // or the effective threshold hits its own ceiling before it's
    // actually strict enough to filter most PCs out.
    cfg.promoteCriticalAt = 2;
    cfg.promoteNonCriticalAt = -2;
    cfg.trainCounterMin = -40;
    cfg.trainCounterMax = 40;

    cfg.enableCriticalRateGovernor = true;
    cfg.targetCriticalRate = 0.10;
    cfg.criticalRateDeadband = 0.02;
    cfg.criticalRateControlInterval = 64;
    cfg.criticalRateRelaxDwellWindows = 4;

    cfg.enableTraceLog = true;

    TokenEngine engine(cfg);

    std::mt19937 rng(7);
    std::uniform_int_distribution<int> repsDist(15, 30);

    const int kGenerations = 40;
    Cycle t = 0;
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
                DoEdge(engine, pc, t, distance);
            }
        }
    }

    return 0;
}
