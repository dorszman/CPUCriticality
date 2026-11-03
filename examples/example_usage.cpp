// Minimal synthetic harness exercising the TokenEngine / Token API end to
// end, in the shape a real host integration actually takes: a token is
// held by a uop's TokenRef member. On that uop's execute, the host's OWN
// issue-queue wakeup/broadcast logic determines which consumer(s), if
// any, were woken by this result. Exactly one woken consumer gets the
// token handed directly to it (a plain move, no engine call). SEVERAL
// woken consumers (fan-out) each get their own COPY of the token (a plain
// TokenRef copy, refcount > 1 -- a normal, intended state here, not a
// bug) and race: whichever of them executes first necessarily has the
// shortest distance back to the producer, so it is automatically the
// strongest critical-path evidence. The moment it closes the measurement,
// the slot is recycled IMMEDIATELY (its epoch bumped) rather than waiting
// for every slower sibling to drop its copy -- so when a slower sibling
// (like H below) later calls NotifyExecuted() on its own copy, that copy
// has already gone stale (its captured epoch no longer matches the
// token's current one) and the call is a silent no-op, touching nothing,
// even though the underlying pool slot may by then belong to a completely
// different instruction. If the broadcast wakes nobody, nothing is gated
// by this producer right now, so the token is simply dropped. There is no
// Ptag/register-scoreboard table in this module at all -- wakeup matching
// is the host's own reservation-station logic, reused as-is, not
// duplicated here.
//
// Single-hop lifecycle: a token measures exactly one producer->consumer
// edge, then its job is done. NotifyExecuted() tells the host via its
// return value whether this was the first execute (look for wakeups) or
// a close attempt (always drop; training already happened, didn't, or
// was a stale no-op, internally, and the caller never needs to know
// which).
//
// This is a compile/smoke test, not a real simulator integration -- see
// docs/DESIGN.md S5 for the exact hook contract.
//
// Build: g++ -std=c++17 -Iinclude examples/example_usage.cpp src/token_engine.cpp src/token.cpp -o example_usage

#include "token_engine.h"

#include <cstdio>
#include <unordered_map>
#include <vector>

using namespace cacrit;

using Ptag = std::uint64_t;
constexpr Ptag kNoPtag = 0;

// A stand-in for a real host's uop/micro-op class.
struct Uop {
    InstrId id = kInvalidInstrId;
    Addr pc = 0;
    Ptag destTag = kNoPtag;
    TokenRef token; // <-- the integration point: a plain refcounted handle
};

int main() {
    CriticalityConfig cfg;
    cfg.tokenPoolSize = 4;
    cfg.tokenAssignInterval = 1; // tag every decoded uop, for a dense demo
    cfg.criticalDistanceCycles = 1;
    cfg.nonCriticalDistanceCycles = 4;
    cfg.promoteCriticalAt = 1;
    cfg.promoteNonCriticalAt = -1;

    TokenEngine engine(cfg);

    // Toy versions of the host's own structures -- none of this is known
    // to, or owned by, TokenEngine. A real host already has exactly this
    // in its reservation-station wakeup/matching logic; it is reproduced
    // minimally here only so the demo has something to wake up against.
    std::unordered_map<InstrId, Uop> rob;
    std::unordered_map<Ptag, std::vector<InstrId>> waitingOnTag; // producer tag -> not-yet-ready consumers
    std::unordered_map<InstrId, int> pendingOperandCount;        // consumer -> # not-yet-ready sources

    auto decodeAndRename = [&](InstrId id, Addr pc, Ptag destTag, Ptag srcTagIfNotReady) {
        Uop u;
        u.id = id;
        u.pc = pc;
        u.destTag = destTag;
        if (srcTagIfNotReady != kNoPtag) {
            waitingOnTag[srcTagIfNotReady].push_back(id);
            pendingOperandCount[id] += 1;
        }
        if (engine.TokenAvailable()) {
            u.token = engine.DecodeGetToken(pc); // fresh sample
        }
        rob[id] = std::move(u);
    };

    auto execute = [&](InstrId id, Cycle now) {
        Uop& u = rob.at(id);

        // The host's own wakeup broadcast: find ALL consumers waiting on
        // this uop's destination tag that become ready as a result (there
        // can be more than one -- shared condition code, a cache-line
        // fill waking several loads). A real host already computes this
        // as part of ordinary issue-queue select/wakeup logic.
        std::vector<InstrId> woken;
        auto it = waitingOnTag.find(u.destTag);
        if (u.destTag != kNoPtag && it != waitingOnTag.end()) {
            for (InstrId c : it->second) {
                auto cntIt = pendingOperandCount.find(c);
                if (cntIt == pendingOperandCount.end()) continue; // flushed; ignore
                if (--(cntIt->second) <= 0) {
                    pendingOperandCount.erase(cntIt);
                    woken.push_back(c);
                }
            }
            waitingOnTag.erase(it);
        }

        if (!u.token.Valid()) return;

        engine.Tick(now);
        bool firstExecute = u.token.NotifyExecuted(); // the TokenRef's own API (epoch-guarded)
        if (firstExecute && !woken.empty()) {
            // Hand every woken consumer its OWN copy of the token. If
            // there's only one, this is just a direct hand-off; if
            // several, they race -- see the file header comment.
            for (InstrId c : woken) {
                rob.at(c).token = u.token; // copy, not move: refcount++
            }
        }
        // Either way, the producer's own reference is done with it now:
        // if copies were handed out, this just drops the producer's share
        // (siblings still hold theirs); if nothing was woken, or this was
        // itself a close attempt, this is the token's last reference.
        u.token.Reset();
    };

    // Edge 1: PC 0x1000 (writes T1) -> PC 0x1004 (reads T1), one cycle
    // apart -- a single producer/consumer edge, trained as critical.
    const Addr pcA = 0x1000, pcB = 0x1004;
    const Ptag t1 = 101;

    decodeAndRename(2, pcB, kNoPtag, /*srcTagIfNotReady=*/t1); // consumer renamed first, waits on t1
    decodeAndRename(1, pcA, t1, kNoPtag);
    execute(1, 10); // first execute: wakes uop 2; token handed directly to it
    execute(2, 11); // closing execute: distance(A->B) = 1 -> trains pcA critical

    // Edge 2: an independent sample landing on pcB as the PRODUCER this
    // time (a different dynamic instance / different token), to show the
    // predictor trains per static PC across separate samples, not just
    // within one edge.
    const Addr pcD = 0x1008;
    const Ptag t2 = 102;

    decodeAndRename(11, pcD, kNoPtag, /*srcTagIfNotReady=*/t2);
    decodeAndRename(10, pcB, t2, kNoPtag);
    execute(10, 30); // wakes uop 11; token handed directly to it
    execute(11, 31); // closing execute: distance(B->D) = 1 -> trains pcB critical

    // Edge 3: a producer whose broadcast wakes NOBODY (no consumer is
    // waiting on its result). Per the rule this turn introduced: the
    // token does not propagate anywhere -- it is simply dropped, and
    // this producer accumulates no training signal (stays at the
    // engine's non-critical default).
    const Addr pcE = 0x100C;
    decodeAndRename(30, pcE, /*destTag=*/777, kNoPtag); // nobody ever registers interest in tag 777
    execute(30, 50);

    // Edge 4: fan-out race. Producer F wakes TWO consumers (G and H) in
    // the same broadcast. G executes 1 cycle later (tight -> critical
    // evidence); H executes 5 cycles later (would look non-critical on
    // its own) but must NOT get to train at all, because G -- the faster
    // sibling -- already closed the measurement. If this works, pcF ends
    // up critical, and exactly ONE training event comes out of this
    // whole edge, not two.
    const Addr pcF = 0x1010, pcG = 0x1014, pcH = 0x1018;
    const Ptag tF = 201;
    const std::uint64_t trainingEventsBeforeFanout = engine.Stats().trainingEvents;

    decodeAndRename(50, pcG, kNoPtag, /*srcTagIfNotReady=*/tF);
    decodeAndRename(51, pcH, kNoPtag, /*srcTagIfNotReady=*/tF);
    decodeAndRename(52, pcF, tF, kNoPtag);
    execute(52, 60);              // wakes both G and H; both get a copy of the token
    execute(50, 61);              // G first: distance 1 -> closes the measurement, trains pcF critical
    execute(51, 65);              // H second: distance would be 5, but closed_ makes this a no-op

    std::printf("pcF predicted-critical (should be 1 -- trained by faster sibling G): %d\n",
                 engine.IsPredictedCritical(pcF));
    std::printf("fan-out race produced %llu training event(s) (should be 1, not 2)\n",
                 static_cast<unsigned long long>(engine.Stats().trainingEvents - trainingEventsBeforeFanout));

    std::printf("pcA predicted-critical: %d\n", engine.IsPredictedCritical(pcA));
    std::printf("pcB predicted-critical: %d\n", engine.IsPredictedCritical(pcB));
    std::printf("pcE predicted-critical (should be 0 -- its broadcast woke nobody): %d\n",
                 engine.IsPredictedCritical(pcE));
    std::printf("unknown pc predicted-critical (should be 0): %d\n",
                 engine.IsPredictedCritical(0xDEAD0));

    // --- Flush demo: no engine call ---------------------------------------
    // A uop holding a token is flushed before executing; its TokenRef
    // releases via ordinary RAII, no engine call anywhere.
    decodeAndRename(40, 0x101C, kNoPtag, kNoPtag);
    rob.erase(40);

    const CriticalityStats& s = engine.Stats();
    std::printf("tokensAssigned=%llu tokensExpired=%llu trainingEvents=%llu "
                "criticalPromotions=%llu nonCriticalPromotions=%llu\n",
                static_cast<unsigned long long>(s.tokensAssigned),
                static_cast<unsigned long long>(s.tokensExpired),
                static_cast<unsigned long long>(s.trainingEvents),
                static_cast<unsigned long long>(s.criticalPromotions),
                static_cast<unsigned long long>(s.nonCriticalPromotions));

    return 0;
}
