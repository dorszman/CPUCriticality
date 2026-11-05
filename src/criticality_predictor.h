// CriticalityPredictor: the training/prediction mechanism itself, with
// no sampling-strategy opinions at all -- it only knows how to turn a
// stream of (pc, distance) samples into predictions. How those samples
// get chosen (TokenEngine's budget-constrained token-passing, an
// "ideal" driver that measures every single dynamic instruction, or
// anything else) is entirely the caller's decision; this class has no
// idea which one is driving it.
//
// Deliberately independent of token.h / TokenEngine: no Token, TokenRef,
// or epoch concepts anywhere in this header, so it can be driven
// directly -- one TrainOn() call per sample, however that sample was
// obtained -- with zero token-pool machinery pulled in. See
// docs/DESIGN.md.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace ISQCrit {

using Addr = std::uint64_t;

// How a set-associative table picks a victim once a set is full. Shared
// by the training and prediction tables, though only the training
// table's choice is exposed as a config knob (see
// PredictorConfig::trainingTableEvictionPolicy below) -- the prediction
// table's policy is fixed, already validated, and not something this
// project currently has evidence to make configurable.
enum class EvictionPolicy {
    // Pure least-recently-used: whichever way in the set was touched
    // longest ago loses, full stop. Ignores any confidence/frequency
    // signal an entry may carry. The simplest, safest choice -- no
    // staleness math, nothing to calibrate.
    kLru,

    // Evict by lowest decayed access-frequency counter (see
    // DecayedPriority in criticality_predictor.cpp), with NO recency
    // tiebreak at all -- ties are broken by fixed way order, not by
    // lastTouched_. Protects frequently-(re)trained entries from
    // eviction by a merely-more-recent one-off, but a long-idle entry's
    // advantage still erodes on its own as ITS set sees more traffic,
    // so nothing earns permanent protection.
    kDecayedHitCount,

    // Decayed priority first, least-recently-used only as a tiebreak
    // among exactly equal (decayed) priorities. The prediction table's
    // fixed policy; also selectable for the training table if wanted.
    kDecayedPriorityThenLru,
};

struct PredictorConfig {
    // Master on/off switch for this predictor's OUTPUT. When false,
    // IsPredictedCritical() unconditionally returns false, regardless of
    // trained state -- but training and all table bookkeeping (hitCount,
    // the rate governor's observed rate, etc.) still run exactly as
    // normal underneath. This masks the verdict, it does not pause
    // learning: flipping it back to true immediately benefits from
    // whatever was learned while it was off. Intended for clean A/B
    // comparison (is criticality-aware scheduling actually helping?) or
    // live on/off toggling without discarding accumulated state or
    // restarting the predictor.
    bool enablePrediction = true;

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
    // be >= 1 -- CriticalityPredictor's constructor throws
    // std::invalid_argument if either is 0.
    unsigned trainingTableWays = 4;
    unsigned predictionTableWays = 4;

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
    // Both must be >= 1 -- CriticalityPredictor's constructor throws
    // std::invalid_argument if either is 0 (used as a divisor).
    std::uint64_t trainingTableDecayIdleUnit = 128;
    std::uint64_t predictionTableDecayIdleUnit = 128;

    // Distance (in cycles) at or below which a measured producer/consumer
    // edge counts as evidence toward "critical".
    std::int32_t criticalDistanceCycles = 2;

    // Distance (in cycles) at or above which a measured edge counts as
    // evidence toward "non-critical". Distances strictly between the two
    // thresholds are treated as ambiguous and do not update training.
    std::int32_t nonCriticalDistanceCycles = 8;

    // Saturating range for the per-PC training counter.
    std::int32_t trainCounterMin = -10;
    std::int32_t trainCounterMax = 10;

    // Training-counter thresholds at which a PC is (re-)promoted into the
    // prediction table as critical / non-critical.
    std::int32_t promoteCriticalAt = 5;
    std::int32_t promoteNonCriticalAt = -3;

    // Criticality-rate governor: a one-sided CAP (never a target to hit
    // exactly) on the fraction of IsPredictedCritical() lookups that
    // return true. Disabled by default -- when disabled,
    // promoteCriticalAt above is used exactly as configured, as if this
    // whole mechanism didn't exist.
    //
    // When enabled, promoteCriticalAt becomes a FLOOR rather than a
    // fixed value: the predictor tracks the observed critical rate over
    // windows of criticalRateControlInterval lookups, and if a window's
    // rate exceeds targetCriticalRate + criticalRateDeadband, the
    // EFFECTIVE promotion threshold is nudged one step stricter
    // (requiring more evidence before promoting). If a window's rate
    // falls below targetCriticalRate - criticalRateDeadband, the
    // effective threshold is nudged one step back toward -- but never
    // past -- promoteCriticalAt itself. Being under the cap is always
    // fine and never relaxes the threshold any softer than the host's
    // own configured value; the governor only ever adds extra
    // strictness on top of it, and only when the cap would otherwise be
    // exceeded. promoteNonCriticalAt (demotion) is never touched by
    // this mechanism.
    //
    // If the organic (un-governed) critical rate for a workload is
    // already at or below the cap, the effective threshold simply
    // settles back at promoteCriticalAt and this is a no-op in
    // practice. If even the maximum possible strictness
    // (trainCounterMax) isn't enough to bring a workload under the cap,
    // the cap is approached as closely as a single threshold dimension
    // allows, never exceeded by construction, but not fully hit either
    // -- an honest limit, not a bug.
    //
    // IMPORTANT usage requirement, empirically confirmed (not just a
    // theoretical caveat): this governor needs real headroom between
    // promoteCriticalAt and trainCounterMax to work at all. Each
    // tightening step can only move the effective threshold to the next
    // integer value the training counter can actually reach, so with
    // only 1-2 such steps available, the governor swings between
    // "admit everyone" and "admit no one" and the ACHIEVED rate can
    // overshoot the cap by 5-7x on realistic workloads, no matter how
    // criticalRateRelaxDwellWindows is tuned. With >=5-6 steps of
    // headroom, the same mechanism converges cleanly near the cap. This
    // project's OWN baseline defaults (promoteCriticalAt=5,
    // trainCounterMax=10) give exactly 5 steps of headroom -- right at
    // the low edge of the validated-good range, not the bad case
    // anymore, but still worth raising trainCounterMax further above
    // promoteCriticalAt if your workload's own data shows the governor
    // still overshooting; the defaults are tuned for hysteresis depth
    // alone and were never validated end-to-end against this mechanism.
    bool enableCriticalRateGovernor = false;
    double targetCriticalRate = 0.10;     // the cap, e.g. 0.10 == 10%
    double criticalRateDeadband = 0.01;   // +/- band around the cap with no adjustment
    // How many IsPredictedCritical() lookups make up one control
    // decision. Must be >= 1 when enableCriticalRateGovernor is true --
    // CriticalityPredictor's constructor throws std::invalid_argument
    // otherwise (it is used as a divisor).
    std::uint32_t criticalRateControlInterval = 1024;

    // Asymmetric response, deliberately: a single over-cap window
    // tightens immediately (the cap must never be exceeded for long),
    // but relaxing back toward the floor requires this many CONSECUTIVE
    // under-cap windows first. Without this, a population that responds
    // to the threshold in one hard step (many PCs crossing it at once,
    // rather than a smooth spread) makes the governor oscillate: it
    // tightens, the rate crashes, the very next window looks
    // comfortably under cap and relaxes right back, re-admitting
    // another burst -- a clean bang-bang limit cycle that averages out
    // far above the cap despite every individual decision looking
    // correct in isolation. Must be >= 1 when
    // enableCriticalRateGovernor is true.
    std::uint32_t criticalRateRelaxDwellWindows = 4;

    // Debugging/visualization aid only -- never enable in anything
    // performance-sensitive. When true, every table allocation,
    // eviction, promotion, demotion, prediction lookup, and governor
    // threshold change is printed to traceLogStream, one line each,
    // every line starting with the exact token "@@ISQCRIT@@" so a log
    // consumer (see tools/visualize_trace.py) can pull these lines out
    // unambiguously even if something else writes to the same stream.
    // Disabled by default; zero cost when disabled beyond the branch
    // itself.
    bool enableTraceLog = false;

    // Where enableTraceLog's lines go. Defaults to stdout, but a host
    // that itself prints a lot of unrelated volume to stdout should
    // point this at a dedicated FILE* instead (e.g. fopen() a separate
    // trace file and hand it in here) -- this predictor never opens,
    // closes, or otherwise owns this stream; exactly like
    // PredictorStatPointers, the host owns the resource and this
    // predictor only ever writes through it. Each line is fflush()'d
    // individually (never a stream-wide buffering-mode change, which
    // would affect everything else written to the same stream too) so
    // a live consumer tailing it sees lines promptly; this flush cost
    // is scoped to exactly this stream and only while enableTraceLog
    // is on.
    std::FILE* traceLogStream = stdout;
};

// The counter width/type lives in exactly one place. Change it here (e.g.
// to a narrower type, or to something atomic if a host ever drives this
// predictor from more than one thread) and every counter follows.
using StatCounter = std::uint64_t;

// The counters themselves live in the HOST's own stats storage, not
// inside this predictor -- it never allocates or owns any counter memory
// of its own. Call CriticalityPredictor::SetupStats() once, typically
// right after construction, with each field pointing at wherever the
// host already keeps that counter. The predictor then writes straight
// through these pointers from then on. Any field left null is simply
// never incremented.
struct PredictorStatPointers {
    StatCounter* trainingEvents = nullptr;      // a distance measurement was taken
    StatCounter* criticalPromotions = nullptr;
    StatCounter* nonCriticalPromotions = nullptr;
};

class CriticalityPredictor {
public:
    // Throws std::invalid_argument for any of the structural/divisor
    // fields documented above as requiring >= 1.
    explicit CriticalityPredictor(const PredictorConfig& cfg = PredictorConfig{});
    ~CriticalityPredictor();

    CriticalityPredictor(const CriticalityPredictor&) = delete;
    CriticalityPredictor& operator=(const CriticalityPredictor&) = delete;

    // Feed one measured sample: a ready-to-issue uop waited
    // 'waitingTime' cycles, attributed to 'producerPc' -- the PC of the
    // producer whose result it was waiting on (e.g. its write-back cycle
    // to this uop's own issue/execute cycle, the "wakeup to issue"
    // distance). However the sample was obtained -- via TokenEngine's
    // sparse, budget-gated sampling, or directly from a host that
    // measures every dynamic instruction ("ideal mode") -- this is the
    // one entry point that trains 'producerPc'. Only ever compared
    // against criticalDistanceCycles/nonCriticalDistanceCycles; the
    // waiting-time convention itself is the caller's to define
    // consistently.
    void UpdateIssueqReadyWaitingTime(Addr producerPc, std::int64_t waitingTime);

    // Read-only prediction lookup. False (non-critical) for any PC with
    // no trained entry yet, or unconditionally when
    // PredictorConfig::enablePrediction is false (see its comment) --
    // training and table bookkeeping (hitCount, the rate governor, etc.)
    // still run normally underneath either way; only the value actually
    // returned here is masked.
    bool IsPredictedCritical(Addr pc) const;

    // Call once, typically right after construction, pointing each field
    // at counter storage the HOST already owns. See PredictorStatPointers.
    // Safe to call with any subset of fields left null; safe to never
    // call at all (every counter is then simply never incremented).
    void SetupStats(const PredictorStatPointers& counters);

    // Debugging/visualization aid only, same spirit as enableTraceLog --
    // a no-op unless it's enabled. Lets the host log an arbitrary named
    // stat it tracks itself (retired instruction count, IPC, whatever)
    // through the SAME trace stream, tagged with this predictor's own
    // internal call counter so it lines up on the same X axis as every
    // other trace line (see tools/visualize_trace.py). This predictor
    // has no notion of "retirement" or any other host-side concept
    // itself -- it never computes or interprets 'value', only logs it.
    // Safe to call at any rate the host finds convenient; each call is
    // one trace line.
    void TraceLogStat(const char* name, std::int64_t value);

private:
    struct Impl;
    Impl* impl_;
};

} // namespace ISQCrit
