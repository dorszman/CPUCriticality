// Same synthetic workload shape as tools/trace_demo.cpp, but
// PredictorConfig is left completely untouched (all defaults),
// including the rate governor. enableTraceLog is turned on here only
// so this program has output at all; tokenAssignInterval is a sampling
// knob, not a PredictorConfig field. A baseline for comparing against
// trace_demo.cpp's heavily-tuned config.
//
// Build: g++ -std=c++17 -Isrc tools/trace_demo_defaults.cpp src/token_engine.cpp src/criticality_predictor.cpp src/token.cpp -o trace_demo_defaults
// Run:   ./trace_demo_defaults > /tmp/trace_defaults.log
//        python3 tools/visualize_trace.py /tmp/trace_defaults.log

#include "token_engine.h"

#include <cstdio>
#include <random>

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

} // namespace

int main() {
    CriticalityConfig cfg;
    cfg.tokenAssignInterval = 1; // token-sampling knob, not a PredictorConfig default
    cfg.predictor.enableTraceLog = true; // only override: need output at all

    TokenEngine engine(cfg);

    std::mt19937 rng(7);
    std::uniform_int_distribution<int> repsDist(15, 30);

    // Long enough for the rate governor to finish tightening and, if the
    // critical fraction then falls under the cap, to start relaxing.
    const int kGenerations = 4000;
    Cycle t = 0;
    for (int gen = 0; gen < kGenerations; ++gen) {
        for (int slot = 0; slot < 8; ++slot) {
            Addr pc = (static_cast<Addr>(gen) << 16) | (static_cast<Addr>(slot) << 2);
            int reps = repsDist(rng);
            int criticalReps = reps / 2;
            // A matched non-critical phase only walks the counter back
            // to ~0. Demotion fires at promoteNonCriticalAt (default
            // -3), so keep decrementing until the counter crosses it.
            int nonCriticalReps = criticalReps - cfg.predictor.promoteNonCriticalAt;
            for (int r = 0; r < criticalReps + nonCriticalReps; ++r) {
                std::int64_t distance = (r < criticalReps) ? 1 : 12;
                DoEdge(engine, pc, t, distance);
            }
        }
    }
    return 0;
}
