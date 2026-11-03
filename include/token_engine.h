// TokenEngine: the token-passing dynamic criticality predictor itself.
// Owns the token pool and the PC-indexed training/prediction tables. See
// docs/DESIGN.md for the full mechanism and the exact host integration
// contract.
//
// This header is self-contained (C++17, standard library only) and has no
// dependency on any particular simulator's internal types. Notably, it
// has NO dependency-graph bookkeeping of its own: token hand-off rides
// directly on the host's own issue-queue wakeup/broadcast logic (see
// token.h) -- a producer delivers its token straight to whichever
// consumer its broadcast wakes, or drops it if the broadcast wakes
// nobody. This engine never needs to be told about flush/squash events
// either -- a flushed uop's TokenRef releases by ordinary object lifetime
// on the host side.
#pragma once

#include "token.h"

#include <cstddef>
#include <cstdint>

namespace cacrit {

struct CriticalityConfig {
    // Number of simultaneously in-flight tokens. Small by design: this
    // bounds both the sampling rate and the hardware cost of the token
    // pool. Must be >= 1.
    unsigned tokenPoolSize = 8;

    // log2 of the number of entries in the training table and prediction
    // table (both direct-mapped, PC-indexed).
    unsigned trainingTableEntriesLog2 = 10;
    unsigned predictionTableEntriesLog2 = 10;

    // Default sampling policy (matches the original mechanism): a token
    // is assigned on average once every tokenAssignInterval decoded
    // instructions, deliberately undersampling even when the pool has
    // free slots. Must be >= 1. Ignored when pureAvailabilityGating is
    // true.
    unsigned tokenAssignInterval = 32;

    // When true, TokenAvailable() ignores tokenAssignInterval and is
    // gated purely by pool occupancy: every decoded uop is eligible for a
    // token as long as a slot is free, and scarcity alone throttles the
    // sampling rate. When false (default), TokenAvailable() requires
    // BOTH a free slot AND the sparse-interval counter to hit, matching
    // the original mechanism's deliberate undersampling. This is a
    // sampling-policy knob, not a correctness one -- either setting
    // produces a valid, bounded-cost predictor; they differ in sampling
    // statistics (see docs/DESIGN.md).
    bool pureAvailabilityGating = false;

    // Distance (in cycles) at or below which a measured producer/consumer
    // edge counts as evidence toward "critical".
    std::int32_t criticalDistanceCycles = 2;

    // Distance (in cycles) at or above which a measured edge counts as
    // evidence toward "non-critical". Distances strictly between the two
    // thresholds are treated as ambiguous and do not update training.
    std::int32_t nonCriticalDistanceCycles = 8;

    // Saturating range for the per-PC training counter.
    std::int32_t trainCounterMin = -4;
    std::int32_t trainCounterMax = 3;

    // Training-counter thresholds at which a PC is (re-)promoted into the
    // prediction table as critical / non-critical.
    std::int32_t promoteCriticalAt = 2;
    std::int32_t promoteNonCriticalAt = -2;
};

struct CriticalityStats {
    std::uint64_t tokensAssigned = 0;
    std::uint64_t tokensExpired = 0;       // refcount reached zero, slot freed
    std::uint64_t trainingEvents = 0;      // a distance measurement was taken
    std::uint64_t criticalPromotions = 0;
    std::uint64_t nonCriticalPromotions = 0;
};

class TokenEngine {
public:
    explicit TokenEngine(const CriticalityConfig& cfg = CriticalityConfig{});
    ~TokenEngine();

    TokenEngine(const TokenEngine&) = delete;
    TokenEngine& operator=(const TokenEngine&) = delete;

    // Call once per cycle, before any decode/execute hooks for that cycle.
    // The engine tracks the current cycle internally so it does not need
    // to be passed to every other call (see Token::NotifyExecuted).
    void Tick(Cycle now);

    // --- Decode -----------------------------------------------------------
    // Call once per decoded uop, in this order:
    //
    //   if (engine.TokenAvailable()) {
    //       uop.token = engine.DecodeGetToken(pc);
    //   }
    //
    // TokenAvailable() is a pure query with one side effect: it always
    // advances the internal sparse-sampling counter (so the counter
    // tracks decoded uops, not granted tokens), then returns whether a
    // token may be taken right now -- pool-occupancy alone if
    // cfg.pureAvailabilityGating is set, or pool occupancy AND the
    // sampling-interval hit otherwise (the default, matching the
    // original mechanism's deliberate undersampling).
    //
    // DecodeGetToken(pc) assumes TokenAvailable() was just checked and
    // returned true; it pops a free slot, roots it at 'pc', and returns
    // the refcounted handle to store as a plain member of your own
    // uop/micro-op object. Calling it when no slot is free returns an
    // empty (invalid) TokenRef rather than asserting, as a defensive
    // fallback.
    bool TokenAvailable();
    TokenRef DecodeGetToken(Addr pc);

    // Read-only prediction lookup, usable independently of whether this
    // particular dynamic instance was sampled by a token. Call in
    // parallel with your instruction-cache fetch; false (non-critical) is
    // returned for any PC with no trained entry yet.
    bool IsPredictedCritical(Addr pc) const;

    // --- Execute, wakeup, and flush: no engine calls at all --------------
    // There is deliberately no execute/wakeup/flush hook here. On
    // execute, the host calls Token::NotifyExecuted() (see token.h) and
    // acts on its return value together with its OWN issue-queue wakeup
    // result: true plus a woken consumer means hand the TokenRef directly
    // to that consumer (a plain move, not an engine call); true with no
    // wakeup, or false, both mean simply drop the TokenRef. On flush (a
    // uop squashed before executing, or holding a token mid-measurement),
    // the host's own existing object-lifetime management destroys or
    // resets that TokenRef, which is sufficient -- see docs/DESIGN.md
    // S3.2/S3.3/S5/S8 for the exact sequence and why no separate flush
    // entry point is needed.

    static std::int32_t PriorityBoost(bool predictedCritical,
                                       std::int32_t boostMagnitude = 1);

    const CriticalityStats& Stats() const;

private:
    friend class Token;
    // API called by the token itself (Token::DoNotifyExecuted forwards
    // here, only once TokenRef has confirmed its epoch is current), not
    // by the host directly. Returns true if this was the first execute
    // (host should check its own wakeup result and hand the token to
    // whoever got woken, if anyone), false if the measurement just
    // closed -- in which case the slot has ALREADY been recycled
    // (epoch bumped, returned to the pool) before this call returns,
    // regardless of any other TokenRef copies still outstanding. The
    // host should drop it either way.
    bool UpdateCriticality(Token* token);
    // Called once a token is dead by refcount (Token::Release at 0). Not
    // called for the closed-measurement recycle path, which bumps the
    // epoch and frees the slot immediately inside UpdateCriticality
    // instead of waiting for refcount to reach zero.
    void TokenExpire(Token* token);

    struct Impl;
    Impl* impl_;
};

} // namespace cacrit
