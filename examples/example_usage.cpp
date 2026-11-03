// Minimal synthetic harness exercising the CriticalityEngine / Token API
// end to end, in the shape a real host integration actually takes: a
// host-owned Uop struct embeds a TokenRef as a plain member, and the
// engine never tracks tokens by id internally -- the uop IS the owner.
//
// This is a compile/smoke test, not a real simulator integration -- see
// docs/DESIGN.md S5 for the exact hook contract.
//
// Build: g++ -std=c++17 -Iinclude examples/example_usage.cpp src/criticality_engine.cpp src/token.cpp -o example_usage

#include "criticality_engine.h"

#include <cstdio>
#include <initializer_list>
#include <unordered_map>

using namespace cacrit;

// A stand-in for a real host's uop/micro-op class. The only thing this
// module requires of it is the TokenRef member -- everything else here
// (pc, id) is just what this toy harness needs to drive the demo.
struct Uop {
    InstrId id = kInvalidInstrId;
    Addr pc = 0;
    TokenRef token; // <-- the integration point: a plain refcounted handle
};

int main() {
    CriticalityConfig cfg;
    cfg.tokenPoolSize = 4;
    cfg.tokenAssignInterval = 1; // tag every fetched instruction, for a dense demo
    cfg.criticalDistanceCycles = 1;
    cfg.nonCriticalDistanceCycles = 4;
    cfg.promoteCriticalAt = 1;
    cfg.promoteNonCriticalAt = -1;

    CriticalityEngine engine(cfg);

    // A toy "ROB": the host's own uop-by-id lookup, which this module
    // never needs to know about directly.
    std::unordered_map<InstrId, Uop> rob;

    auto fetch = [&](InstrId id, Addr pc) {
        Uop u;
        u.id = id;
        u.pc = pc;
        u.token = engine.maybeAssignToken(pc); // usually empty; may carry a token
        rob[id] = std::move(u);
    };

    auto execute = [&](InstrId id, Cycle now) {
        Uop& u = rob.at(id);
        if (u.token.valid()) {
            u.token->notifyExecuted(u.pc, now); // the token's own API, per docs/DESIGN.md
        }
        InstrId readyId = engine.resolveDependents(u.id);
        if (u.token.valid()) {
            if (readyId != kInvalidInstrId) {
                rob.at(readyId).token = u.token; // hand-off: refcount 1 -> 2
            }
            u.token.reset(); // producer's leg is done either way; refcount 2->1 (handed off) or 1->0 (dead end, slot reclaimed)
        }
    };

    // Synthetic tight chain: PC 0x1000 -> 0x1004 -> 0x1008, each
    // instruction executing exactly one cycle after the previous one.
    const Addr pcA = 0x1000, pcB = 0x1004, pcC = 0x1008;
    const InstrId a = 1, b = 2, c = 3;

    fetch(a, pcA);
    fetch(b, pcB);
    engine.registerDependency(/*consumer=*/b, /*producer=*/a);
    fetch(c, pcC);
    engine.registerDependency(/*consumer=*/c, /*producer=*/b);

    execute(a, 10);
    execute(b, 11); // distance(a->b) = 1 -> critical evidence
    execute(c, 12); // distance(b->c) = 1 -> critical evidence

    for (InstrId id : {InstrId{4}, InstrId{5}, InstrId{6}}) {
        fetch(id, pcA); // more samples to show steady-state behavior
    }

    std::printf("pcA predicted-critical: %d\n", engine.isPredictedCritical(pcA));
    std::printf("pcB predicted-critical: %d\n", engine.isPredictedCritical(pcB));
    std::printf("unknown pc predicted-critical (should be 0): %d\n",
                 engine.isPredictedCritical(0xDEAD0));

    const CriticalityStats& s = engine.stats();
    std::printf("tokensAssigned=%llu tokensRetired=%llu trainingEvents=%llu "
                "criticalPromotions=%llu nonCriticalPromotions=%llu\n",
                static_cast<unsigned long long>(s.tokensAssigned),
                static_cast<unsigned long long>(s.tokensRetired),
                static_cast<unsigned long long>(s.trainingEvents),
                static_cast<unsigned long long>(s.criticalPromotions),
                static_cast<unsigned long long>(s.nonCriticalPromotions));

    return 0;
}
