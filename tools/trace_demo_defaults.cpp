// Same synthetic workload shape as tools/trace_demo.cpp, but
// PredictorConfig is left completely untouched (all defaults),
// including the rate governor. enableTraceLog is turned on here only
// so this program has output at all; tokenAssignInterval is a sampling
// knob, not a PredictorConfig field. A baseline for comparing against
// trace_demo.cpp's heavily-tuned config.
//
// Build: g++ -std=c++17 -Isrc tools/trace_demo_defaults.cpp src/token_engine.cpp src/criticality_predictor.cpp src/token.cpp -o trace_demo_defaults
// Run:   ./trace_demo_defaults /tmp/trace_defaults.log
//        python3 tools/visualize_trace.py /tmp/trace_defaults.log
//
// The default pace is one instruction per millisecond, so a live
// viewer advances in small steps. --fast removes that pause.
// This program does not open or delete the log. It only passes the
// path; the predictor removes the previous file and creates a new one.
//   ./trace_demo_defaults /tmp/trace_live.log
//   python3 tools/visualize_trace.py /tmp/trace_live.log

#include "token_engine.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>

using namespace ISQCrit;

namespace {

void DoEdge(TokenEngine& engine, Addr pc, Cycle& t, std::int64_t distance) {
    TokenRef producer = engine.DecodeGetToken(pc);
    engine.Tick(t);
    producer.NotifyExecuted();
    TokenRef consumer = producer;
    producer.Reset();
    t += distance;
    engine.Tick(t);
    consumer.NotifyExecuted();
    consumer.Reset();
    ++t;
    engine.IsPredictedCritical(pc);
}

// 1000 instructions per second, spread evenly so each viewer frame
// (default 0.1s) sees a small step rather than a 1000-instruction burst.
constexpr auto kSlowPace = std::chrono::milliseconds(1);

} // namespace

int main(int argc, char** argv) {
    bool slow = true;
    const char* logPath = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fast") == 0) slow = false;
        else if (std::strcmp(argv[i], "--slow") == 0) slow = true;
        else if (logPath == nullptr) logPath = argv[i];
        else {
            std::fprintf(stderr, "usage: %s [--fast] [logfile]\n", argv[0]);
            return 1;
        }
    }

    CriticalityConfig cfg;
    cfg.tokenAssignInterval = 1; // token-sampling knob, not a PredictorConfig default
    cfg.predictor.enableTraceLog = true; // only override: need output at all
    // Path only. The predictor deletes the previous file and opens a new one.
    if (logPath != nullptr) cfg.predictor.traceLogPath = logPath;

    TokenEngine engine(cfg);

    std::mt19937 rng(7);
    std::uniform_int_distribution<int> repsDist(15, 30);
    // Waits are drawn inside each class, including the boundary, so the
    // test covers critical, the ignored middle, and non-critical. The
    // middle does not move the training counter.
    const auto critMax = cfg.predictor.criticalDistanceCycles;
    const auto nonCritMin = cfg.predictor.nonCriticalDistanceCycles;
    std::uniform_int_distribution<std::int64_t> criticalWait(0, critMax);
    std::uniform_int_distribution<std::int64_t> nonCriticalWait(nonCritMin, nonCritMin + 8);
    const bool hasNormal = nonCritMin > critMax + 1;
    std::uniform_int_distribution<std::int64_t> normalWait(
        critMax + 1, hasNormal ? nonCritMin - 1 : critMax + 1);
    // Demotion starts at promoteNonCriticalAt. The floor is
    // trainCounterMin, and only some PCs are given enough
    // non-critical samples to land on it.
    const int demoteAt = cfg.predictor.promoteNonCriticalAt;
    const int counterMin = cfg.predictor.trainCounterMin;
    const bool floorBelowDemote = counterMin < demoteAt;
    std::uniform_int_distribution<int> hitMin(0, 4);
    std::uniform_int_distribution<int> stopShort(
        floorBelowDemote ? counterMin + 1 : demoteAt, demoteAt);

    // Long enough for the rate governor to finish tightening and, if the
    // critical fraction then falls under the cap, to start relaxing.
    const int kGenerations = 4000;
    Cycle t = 0;
    for (int gen = 0; gen < kGenerations; ++gen) {
        for (int slot = 0; slot < 8; ++slot) {
            Addr pc = (static_cast<Addr>(gen) << 16) | (static_cast<Addr>(slot) << 2);
            int reps = repsDist(rng);
            int criticalReps = reps / 2;
            // One PC in five ends on the floor. The rest stop somewhere
            // from the demote level down to just above it.
            int target = demoteAt;
            if (floorBelowDemote) {
                target = hitMin(rng) == 0 ? counterMin : stopShort(rng);
            }
            int nonCriticalReps = criticalReps - target;
            if (nonCriticalReps < 0) nonCriticalReps = 0;
            int normalReps = hasNormal ? criticalReps : 0;
            for (int r = 0; r < criticalReps; ++r) {
                DoEdge(engine, pc, t, criticalWait(rng));
                if (slow) std::this_thread::sleep_for(kSlowPace);
            }
            for (int r = 0; r < normalReps; ++r) {
                DoEdge(engine, pc, t, normalWait(rng));
                if (slow) std::this_thread::sleep_for(kSlowPace);
            }
            for (int r = 0; r < nonCriticalReps; ++r) {
                DoEdge(engine, pc, t, nonCriticalWait(rng));
                if (slow) std::this_thread::sleep_for(kSlowPace);
            }
        }
    }
    return 0;
}
