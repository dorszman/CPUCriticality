// Same synthetic workload shape as tools/trace_demo.cpp, but
// PredictorConfig is left completely untouched (all defaults) except
// enableTraceLog, which has to be on just to see any output at all --
// a baseline for comparing against trace_demo.cpp's heavily-tuned
// config (see docs/DESIGN.md Sec 3.8 for why the header's own defaults
// are a known-bad case for the rate governor specifically; the governor
// is left disabled here, also a default).
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

    const int kGenerations = 40;
    Cycle t = 0;
    for (int gen = 0; gen < kGenerations; ++gen) {
        for (int slot = 0; slot < 8; ++slot) {
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
