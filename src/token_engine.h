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

namespace ISQCrit {

// How a set-associative table picks a victim once a set is full. Shared
// by the training and prediction tables, though only the training
// table's choice is exposed as a config knob (see
// CriticalityConfig::trainingTableEvictionPolicy below) -- the
// prediction table's policy is fixed, already validated, and not
// something this project currently has evidence to make configurable.
enum class EvictionPolicy {
    // Pure least-recently-used: whichever way in the set was touched
    // longest ago loses, full stop. Ignores any confidence/frequency
    // signal an entry may carry. The simplest, safest choice -- no
    // staleness math, nothing to calibrate.
    kLru,

    // Evict by lowest decayed access-frequency counter (see
    // DecayedPriority in token_engine.cpp), with NO recency tiebreak at
    // all -- ties are broken by fixed way order, not by lastTouched_.
    // Protects frequently-(re)trained entries from eviction by a
    // merely-more-recent one-off, but a long-idle entry's advantage
    // still erodes on its own as ITS set sees more traffic, so nothing
    // earns permanent protection.
    kDecayedHitCount,

    // Decayed priority first, least-recently-used only as a tiebreak
    // among exactly equal (decayed) priorities. The prediction table's
    // fixed policy; also selectable for the training table if wanted.
    kDecayedPriorityThenLru,
};

struct CriticalityConfig {
    // Number of simultaneously in-flight tokens. Small by design: this
    // bounds both the sampling rate and the hardware cost of the token
    // pool. Must be >= 1 -- TokenEngine's constructor throws
    // std::invalid_argument if it is 0.
    unsigned tokenPoolSize = 8;

    // log2 of the number of SETS in the training table and prediction
    // table (both PC-indexed). Total entries per table = (1 <<
    // ...Log2) * ...Ways.
    unsigned trainingTableEntriesLog2 = 10;
    unsigned predictionTableEntriesLog2 = 10;

    // Ways per set. 1 (the default) reproduces the original direct-mapped
    // design exactly: two PCs that hash to the same set unconditionally
    // evict each other. >1 adds associativity -- both PCs can coexist in
    // the same set (up to 'ways' of them) instead of aliasing, at the
    // cost of searching every way on each lookup and needing a
    // replacement policy (LRU within the set) once a set is full. Must
    // be >= 1 -- TokenEngine's constructor throws std::invalid_argument
    // if either is 0.
    unsigned trainingTableWays = 1;
    unsigned predictionTableWays = 1;

    // Which policy the training table uses to pick a victim once a set
    // is full (only matters when trainingTableWays > 1 and a set is
    // actually full). A deliberate either/or choice between two
    // well-understood, non-overlapping policies -- not a numeric
    // threshold to calibrate. Defaults to the simplest option.
    EvictionPolicy trainingTableEvictionPolicy = EvictionPolicy::kLru;

    // Same choice, for the prediction table (only matters when
    // predictionTableWays > 1). Defaults to kDecayedPriorityThenLru --
    // consulted heavily via IsPredictedCritical() on every fetch, so
    // unlike the training table, decayed usefulness (with LRU only as a
    // tiebreak) rather than plain LRU is the validated starting point
    // here.
    EvictionPolicy predictionTableEvictionPolicy = EvictionPolicy::kDecayedPriorityThenLru;

    // How many idle accesses to an entry's own set correspond to one
    // halving of its eviction priority, for each table's decay-based
    // policy (training table: only when trainingTableEvictionPolicy ==
    // kDecayedHitCount; prediction table: always, its policy is fixed
    // at kDecayedPriorityThenLru). Below this many idle accesses, an
    // entry hasn't decayed at all. Separate per table since the two
    // tables typically see very different access patterns/contention.
    // Both must be >= 1 -- TokenEngine's constructor throws
    // std::invalid_argument if either is 0 (used as a divisor).
    std::uint64_t trainingTableDecayIdleUnit = 128;
    std::uint64_t predictionTableDecayIdleUnit = 128;

    // Default sampling policy (matches the original mechanism): a token
    // is assigned on average once every tokenAssignInterval decoded
    // instructions, deliberately undersampling even when the pool has
    // free slots. Must be >= 1 -- it is used as a modulo divisor, and
    // TokenEngine's constructor throws std::invalid_argument if it is 0
    // rather than letting that become undefined behavior at runtime.
    // Ignored when pureAvailabilityGating is true, but still validated
    // regardless (simpler and safer than a conditionally-enforced rule).
    unsigned tokenAssignInterval = 32;

    // When true, DecodeGetToken() ignores tokenAssignInterval and gates
    // purely on pool occupancy: every decoded uop is eligible for a
    // token as long as a slot is free, and scarcity alone throttles the
    // sampling rate. When false (default), DecodeGetToken() requires
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

// The counter width/type lives in exactly one place. Change it here (e.g.
// to a narrower type, or to something atomic if a host ever drives this
// engine from more than one thread) and every counter follows.
using StatCounter = std::uint64_t;

// The counters themselves live in the HOST's own stats storage, not
// inside this engine -- it never allocates or owns any counter memory of
// its own. Call TokenEngine::SetupStats() once, typically right after
// construction, with each field pointing at wherever the host already
// keeps that counter (a field in its own stats struct, an element of a
// counter array, whatever it already uses for every other component's
// counters). The engine then writes straight through these pointers from
// then on. Any field left null is simply never incremented -- leave
// unset whichever counters you don't care about.
struct CriticalityStatPointers {
    StatCounter* tokensAssigned = nullptr;
    StatCounter* tokensExpired = nullptr;       // refcount reached zero, slot freed
    StatCounter* trainingEvents = nullptr;      // a distance measurement was taken
    StatCounter* criticalPromotions = nullptr;
    StatCounter* nonCriticalPromotions = nullptr;
};

class TokenEngine {
public:
    // Throws std::invalid_argument if cfg.tokenPoolSize == 0 or
    // cfg.tokenAssignInterval == 0 -- both are caller-supplied values
    // this engine cannot safely treat as zero (see their comments above),
    // and validating unconditionally is simpler and safer in a release
    // build than relying on assert(), which compiles away under NDEBUG.
    explicit TokenEngine(const CriticalityConfig& cfg = CriticalityConfig{});
    ~TokenEngine();

    TokenEngine(const TokenEngine&) = delete;
    TokenEngine& operator=(const TokenEngine&) = delete;

    // Call once per cycle, before any decode/execute hooks for that cycle.
    // The engine tracks the current cycle internally so it does not need
    // to be passed to every other call (see Token::NotifyExecuted).
    void Tick(Cycle now);

    // --- Decode -----------------------------------------------------------
    // Call once per decoded uop, unconditionally:
    //
    //   uop.token = engine.DecodeGetToken(pc);
    //
    // Always advances the internal sparse-sampling counter first (so it
    // tracks decoded uops, not granted tokens), then either pops a free
    // slot, roots it at 'pc', and returns the refcounted handle to store
    // as a plain member of your own uop/micro-op object -- or, most of
    // the time, returns an empty (invalid) TokenRef and touches nothing
    // else. A grant requires pool occupancy alone if
    // cfg.pureAvailabilityGating is set, or pool occupancy AND the
    // sampling-interval hit otherwise (the default, matching the
    // original mechanism's deliberate undersampling). There is no
    // separate availability pre-check to call first -- counting and
    // gating happen together, in this one call, every time.
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

    // Call once, typically right after construction, pointing each field
    // at counter storage the HOST already owns. See CriticalityStatPointers.
    // Safe to call with any subset of fields left null; safe to never
    // call at all (every counter is then simply never incremented).
    void SetupStats(const CriticalityStatPointers& counters);

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

} // namespace ISQCrit
