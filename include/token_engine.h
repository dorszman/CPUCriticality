// TokenEngine: the token-passing dynamic criticality predictor itself.
// Owns the token pool, the dependency-graph bookkeeping, and the
// PC-indexed training/prediction tables. See docs/DESIGN.md for the full
// mechanism and the exact host integration contract.
//
// This header is self-contained (C++17, standard library only) and has no
// dependency on any particular simulator's internal types.
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

    // Seed for the internal PRNG used only to fairly tie-break among
    // several consumers that become simultaneously ready from the same
    // producer (fan-out). A fixed default keeps simulation runs
    // reproducible; vary it to decorrelate multiple engine instances or
    // to sweep seeds for sensitivity analysis. Must be nonzero (0 is
    // remapped to a fixed internal default).
    std::uint64_t rngSeed = 0x9E3779B97F4A7C15ULL;
};

struct CriticalityStats {
    std::uint64_t tokensAssigned = 0;
    std::uint64_t tokensRetired = 0;      // refcount reached zero, slot recycled
    std::uint64_t trainingEvents = 0;     // a distance measurement was taken
    std::uint64_t criticalPromotions = 0;
    std::uint64_t nonCriticalPromotions = 0;
};

class TokenEngine {
public:
    explicit TokenEngine(const CriticalityConfig& cfg = CriticalityConfig{});
    ~TokenEngine();

    TokenEngine(const TokenEngine&) = delete;
    TokenEngine& operator=(const TokenEngine&) = delete;

    // --- Hook 1: decode ---------------------------------------------------
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
    bool isPredictedCritical(Addr pc) const;
    bool isPredictedCriticalForEviction(Addr pc) const;

    // --- Hook 2: dependency resolution (rename / scoreboard) ----------
    // Call once per source operand of 'consumer' whose producing
    // instruction 'producer' has NOT yet executed.
    void registerDependency(InstrId consumer, InstrId producer);

    // --- Hook 3: execute / writeback -----------------------------------
    // Call once per instruction at execute time, independent of token
    // state: resolves 'producerId' against its registered dependents and
    // returns the id of the ONE consumer (fairly tie-broken if several
    // became ready simultaneously) for which 'producerId' just supplied
    // the last-arriving operand, or kInvalidInstrId if none. The host is
    // responsible for looking that id up in its own structures and, if
    // the executing uop currently holds a token, copying that TokenRef
    // onto the consumer uop (see docs/DESIGN.md S5 for the exact
    // sequence). This call does not touch tokens itself.
    InstrId resolveDependents(InstrId producerId);

    // --- Hook 4: flush / squash -----------------------------------------
    // Call for any instruction removed from the pipeline without
    // executing. Only cleans up dependency-graph bookkeeping (pending
    // operand counts); any token the flushed uop held is released
    // automatically when its TokenRef is destroyed/reset by the host, not
    // by this call.
    void onFlush(InstrId id);

    static std::int32_t priorityBoost(bool predictedCritical,
                                       std::int32_t boostMagnitude = 1);

    const CriticalityStats& stats() const;

private:
    friend class Token;
    // API called by the token itself (Token::notifyExecuted forwards
    // here), not by the host directly.
    void UpdateCriticality(Token* token, Addr pc, Cycle now);
    // Called once a token is dead by refcount (Token::release at 0).
    void TokenRecycle(Token* token);

    struct Impl;
    Impl* impl_;
};

} // namespace cacrit
