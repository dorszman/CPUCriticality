// Token-passing dynamic criticality predictor for out-of-order issue-queue
// scheduling. See docs/DESIGN.md for the full mechanism description,
// pipeline integration points, and tuning guidance.
//
// This header is self-contained (C++17, standard library only) and has no
// dependency on any particular simulator's internal types. A host model
// integrates it by calling the five hooks declared below at the
// corresponding pipeline events (fetch/rename, dependency resolution,
// execute/writeback, flush/squash) and by consulting isPredictedCritical()
// from its own issue-queue select logic.
#pragma once

#include <cstddef>
#include <cstdint>

namespace cacrit {

using InstrId = std::uint64_t;
using Addr = std::uint64_t;
using Cycle = std::uint64_t;

// Sentinel for "no instruction" — real instruction ids must be nonzero.
constexpr InstrId kInvalidInstrId = 0;

struct CriticalityConfig {
    // Number of simultaneously in-flight tokens. Small by design: this
    // bounds both the sampling rate and the hardware cost of the token
    // pool. Must be >= 1.
    unsigned tokenPoolSize = 8;

    // log2 of the number of entries in the training table and prediction
    // table (both direct-mapped, PC-indexed).
    unsigned trainingTableEntriesLog2 = 10;
    unsigned predictionTableEntriesLog2 = 10;

    // A new token is assigned on average once every tokenAssignInterval
    // fetched instructions (sparse sampling). Must be >= 1.
    unsigned tokenAssignInterval = 32;

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

// Runtime counters, exposed for validation/tuning (see docs/DESIGN.md, S7).
struct CriticalityStats {
    std::uint64_t tokensAssigned = 0;
    std::uint64_t tokensRetired = 0;      // chain dead-ended, token freed
    std::uint64_t trainingEvents = 0;     // a distance measurement was taken
    std::uint64_t criticalPromotions = 0;
    std::uint64_t nonCriticalPromotions = 0;
};

class CriticalityPredictor {
public:
    explicit CriticalityPredictor(const CriticalityConfig& cfg = CriticalityConfig{});
    ~CriticalityPredictor();

    CriticalityPredictor(const CriticalityPredictor&) = delete;
    CriticalityPredictor& operator=(const CriticalityPredictor&) = delete;

    // --- Hook 1: fetch / rename ---------------------------------------
    // Call exactly once per instruction as it is fetched (or renamed, if
    // your model's front end tags instructions there instead). May assign
    // a sparse token to this instruction as a side effect. Returns the
    // current prediction-table verdict for 'pc' (false/non-critical if the
    // PC has no trained entry yet).
    bool onFetch(InstrId id, Addr pc);

    // --- Hook 2: dependency resolution (rename / scoreboard) ----------
    // Call once per source operand of 'consumer' whose producing
    // instruction 'producer' has NOT yet executed. Do not call this for
    // operands that are already ready at rename time (nothing to track).
    void registerDependency(InstrId consumer, InstrId producer);

    // --- Hook 3: execute / writeback -----------------------------------
    // Call exactly once per instruction, at the cycle it executes /
    // broadcasts its result. 'pc' must match the pc passed to onFetch for
    // the same id. Drives all token hand-off, distance measurement, and
    // training (see docs/DESIGN.md S3.3).
    void onExecute(InstrId id, Addr pc, Cycle now);

    // --- Hook 4: flush / squash -----------------------------------------
    // Call for any instruction removed from the pipeline without
    // executing (branch misprediction, exception, pipeline flush). Frees
    // any token waiting on 'id' and drops its pending dependency state.
    void onFlush(InstrId id);

    // --- Hook 5: issue-queue / cache-controller query -------------------
    // Read-only lookups into the prediction table. Both currently consult
    // the same table; isPredictedCriticalForEviction is named separately
    // so a cache-controller integration (docs/DESIGN.md S3.7) reads as a
    // distinct, intentional call site rather than reusing the scheduler's
    // query by accident.
    bool isPredictedCritical(Addr pc) const;
    bool isPredictedCriticalForEviction(Addr pc) const;

    // Convenience for host scheduling logic: returns a non-negative value
    // to add to an instruction's effective age/priority field so that a
    // predicted-critical entry wins ties against non-critical competitors
    // in an existing age-based picker (see docs/DESIGN.md S3.6). Does not
    // read or write predictor state.
    static std::int32_t priorityBoost(bool predictedCritical,
                                       std::int32_t boostMagnitude = 1);

    const CriticalityStats& stats() const;

private:
    struct Impl;
    Impl* impl_;
};

} // namespace cacrit
