// Minimal synthetic harness exercising the CriticalityPredictor API end to
// end. This is a compile/smoke test, not a real simulator integration —
// see docs/DESIGN.md S5 for how to wire the hooks into an actual CA model.
//
// Build: g++ -std=c++17 -I../include example_usage.cpp ../src/criticality_predictor.cpp -o example_usage

#include "criticality_predictor.h"

#include <cstdio>
#include <initializer_list>

using namespace cacrit;

int main() {
    CriticalityConfig cfg;
    cfg.tokenPoolSize = 4;
    cfg.tokenAssignInterval = 1; // tag every fetched instruction, for a dense demo
    cfg.criticalDistanceCycles = 1;
    cfg.nonCriticalDistanceCycles = 4;
    cfg.promoteCriticalAt = 1;
    cfg.promoteNonCriticalAt = -1;

    CriticalityPredictor predictor(cfg);

    // Synthetic tight chain: PC 0x1000 -> 0x1004 -> 0x1008, each instruction
    // executing exactly one cycle after the previous one (no slack).
    const Addr pcA = 0x1000, pcB = 0x1004, pcC = 0x1008;
    const InstrId a = 1, b = 2, c = 3;

    predictor.onFetch(a, pcA);
    predictor.onFetch(b, pcB);
    predictor.registerDependency(/*consumer=*/b, /*producer=*/a);
    predictor.onFetch(c, pcC);
    predictor.registerDependency(/*consumer=*/c, /*producer=*/b);

    predictor.onExecute(a, pcA, /*now=*/10);
    predictor.onExecute(b, pcB, /*now=*/11); // distance(a->b) = 1 -> critical evidence
    predictor.onExecute(c, pcC, /*now=*/12); // distance(b->c) = 1 -> critical evidence

    // Run the same chain a few more times to push the training counters
    // past the promotion threshold (one sample per PC in cfg above is
    // already enough given promoteCriticalAt = 1, but repeat to show
    // steady-state behavior under a more realistic hysteresis setting).
    for (InstrId id : {InstrId{4}, InstrId{5}, InstrId{6}}) {
        predictor.onFetch(id, pcA);
    }

    std::printf("pcA predicted-critical: %d\n", predictor.isPredictedCritical(pcA));
    std::printf("pcB predicted-critical: %d\n", predictor.isPredictedCritical(pcB));
    std::printf("unknown pc predicted-critical (should be 0): %d\n",
                 predictor.isPredictedCritical(0xDEAD0));

    const CriticalityStats& s = predictor.stats();
    std::printf("tokensAssigned=%llu tokensRetired=%llu trainingEvents=%llu "
                "criticalPromotions=%llu nonCriticalPromotions=%llu\n",
                static_cast<unsigned long long>(s.tokensAssigned),
                static_cast<unsigned long long>(s.tokensRetired),
                static_cast<unsigned long long>(s.trainingEvents),
                static_cast<unsigned long long>(s.criticalPromotions),
                static_cast<unsigned long long>(s.nonCriticalPromotions));

    return 0;
}
