#include "criticality_predictor.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace ISQCrit {

namespace {

constexpr std::int32_t ClampAdd(std::int32_t value, std::int32_t delta,
                                 std::int32_t lo, std::int32_t hi) {
    std::int32_t result = value + delta;
    if (result < lo) return lo;
    if (result > hi) return hi;
    return result;
}

// Counter storage lives in the host, not here -- a counter the host
// never wired up via SetupStats() is simply never incremented.
void IncrementIfSet(StatCounter* counter) {
    if (counter) ++(*counter);
}

// Every PredictorConfig::enableTraceLog line starts with this exact
// token, deliberately unusual so a consumer (e.g. tools/visualize_trace.py)
// can pull these lines out of the log file the predictor creates.
constexpr const char* kTraceLogPrefix = "@@ISQCRIT@@";

// Shared by UpdateIssueqReadyWaitingTime()/Promote(): logs an Acquire() that didn't hit an
// existing entry, i.e. a slot allocation -- always reported as
// event=alloc, whether or not a prior occupant had to be evicted to
// make room (true in steady state for essentially every allocation
// past a table's initial fill). old_pc is always printed -- it is
// SetAssociativeTable::kNoEvictionTag (all-ones) when the way was
// simply free (no real eviction), never 0, since 0 can be a real PC
// (see kNoEvictionTag's own comment) and a log consumer has no other
// way to tell the two apart.
void TraceLogAcquire(std::FILE* stream, const char* table, std::size_t index, bool wasHit,
                      Addr evictedTag, Addr newPc, std::uint64_t decode) {
    if (!stream || wasHit) return;
    std::fprintf(stream, "%s table=%s slot=%zu event=alloc old_pc=0x%llx new_pc=0x%llx decode=%llu\n",
                 kTraceLogPrefix, table, index,
                 static_cast<unsigned long long>(evictedTag),
                 static_cast<unsigned long long>(newPc),
                 static_cast<unsigned long long>(decode));
    // Flush this line on the predictor's own log file, so a reader
    // sees it without waiting for the file buffer to fill.
    std::fflush(stream);
}

// A PC-indexed, set-associative table shared by the training and
// prediction tables. EntryT must have 'bool valid' and 'Addr tag'
// members (TrainEntry and PredEntry both do); whatever else it carries
// is this table's payload.
//
// ways=1 reproduces the original direct-mapped design exactly: a single
// candidate entry per set, no search, no replacement decision. ways>1
// searches every way in the addressed set on each access and, when a
// miss needs to allocate, prefers any invalid way before evicting the
// least-recently-used valid one. This is what lets two PCs that happen
// to hash to the same set coexist instead of unconditionally evicting
// each other the way a direct-mapped table always would.
//
// Recency is tracked by a plain monotonic access counter (not wall
// time), bumped on every hit AND every allocation -- not cycle-based,
// since several lookups can legitimately happen within the same cycle
// and still need a strict, unambiguous order for LRU to pick a victim
// consistently.
template <typename EntryT>
class SetAssociativeTable {
public:
    // Assumes 4-byte-aligned (fixed-width RISC-style) instruction
    // encoding; a real design should size/hash this to its own
    // fetch-group granularity instead of this fixed default.
    static constexpr unsigned kPcAlignmentShiftBits = 2;

    // "No real eviction happened" sentinel for AcquireResult::evictedTag
    // -- deliberately NOT 0: a host's real PC/tag space can plausibly
    // include address 0 (a reset vector, a null-like sentinel address,
    // whatever), so 0 cannot double as "no value" without an ambiguity
    // a log consumer has no way to resolve. All-ones is not a
    // physically addressable PC in any realistic tag width, so it's
    // safe as an out-of-band marker.
    static constexpr Addr kNoEvictionTag = std::numeric_limits<Addr>::max();

    // Physical slot index alongside the entry itself -- needed by
    // Impl's trace logging (see PredictorConfig::enableTraceLog) to
    // identify which of a handful of tracked slots an event belongs to.
    // entry == nullptr on a miss; index is then meaningless.
    struct FindResult {
        const EntryT* entry;
        std::size_t index;
    };

    // Same idea for Acquire(): which physical slot was used, whether it
    // was already holding this same tag (a hit), and -- if not a hit --
    // the tag evicted to make room, or 0 if the way was simply free (no
    // real eviction).
    struct AcquireResult {
        EntryT& entry;
        std::size_t index;
        bool wasHit;
        Addr evictedTag;
    };

    void Configure(unsigned log2Sets, unsigned ways,
                   EvictionPolicy policy = EvictionPolicy::kDecayedPriorityThenLru,
                   std::uint64_t decayIdleUnit = 128) {
        numSets_ = std::size_t{1} << log2Sets;
        ways_ = ways;
        policy_ = policy;
        decayIdleUnit_ = decayIdleUnit;
        entries_.assign(numSets_ * ways_, EntryT{});
        lastTouched_.assign(numSets_ * ways_, 0);
        setCounters_.assign(numSets_, 0);
    }

    // Next victim choice uses these. Entries already in the table stay.
    void SetReplacement(EvictionPolicy policy, std::uint64_t decayIdleUnit) const {
        policy_ = policy;
        decayIdleUnit_ = decayIdleUnit;
    }

    // Read-only lookup: returns the matching entry, or nullptr on a
    // miss. Never allocates. Still counts as an access for LRU/decay
    // purposes on a hit (an ordinary cache semantic: reading an entry is
    // still a use of it) -- hence the mutable recency bookkeeping on an
    // otherwise-const operation, and the OnAccessed() callback (see
    // TrainEntry/PredEntry) that lets an entry type track its own notion
    // of "consulted," distinct from Acquire()'s "written."
    FindResult Find(Addr tag) const {
        std::size_t set = SetIndex(tag);
        std::size_t base = set * ways_;
        for (unsigned w = 0; w < ways_; ++w) {
            std::size_t i = base + w;
            if (entries_[i].valid && entries_[i].tag == tag) {
                lastTouched_[i] = ++setCounters_[set];
                entries_[i].OnAccessed();
                return FindResult{&entries_[i], i};
            }
        }
        return FindResult{nullptr, 0};
    }

    // Returns the entry to write for 'tag': an existing match if one is
    // present (payload left untouched, caller updates it), otherwise a
    // freshly allocated entry (valid=true, tag=tag, rest
    // default-constructed) -- picking a free way in the set if one
    // exists, else evicting the set's least-"useful" way. A HIT also
    // calls OnAccessed(), same as Find() -- for the training table, each
    // re-trained sample IS the "consulted" event (there is no separate
    // fetch-time lookup the way the prediction table has); for the
    // prediction table this additionally fires on a re-promotion, a
    // comparatively rare event next to ordinary fetch-time Find() calls.
    AcquireResult Acquire(Addr tag) {
        std::size_t set = SetIndex(tag);
        std::size_t base = set * ways_;
        for (unsigned w = 0; w < ways_; ++w) {
            std::size_t i = base + w;
            if (entries_[i].valid && entries_[i].tag == tag) {
                lastTouched_[i] = ++setCounters_[set];
                entries_[i].OnAccessed();
                return AcquireResult{entries_[i], i, /*wasHit=*/true, kNoEvictionTag};
            }
        }
        std::size_t victim = base;
        bool foundFree = false;
        for (unsigned w = 0; w < ways_; ++w) {
            std::size_t i = base + w;
            if (!entries_[i].valid) {
                victim = i;
                foundFree = true;
                break;
            }
        }
        Addr evictedTag = kNoEvictionTag; // stays the sentinel unless a real occupant is evicted below
        if (!foundFree) {
            // Victim choice depends on policy_ -- see LessEvictable.
            victim = base;
            for (unsigned w = 1; w < ways_; ++w) {
                std::size_t i = base + w;
                if (LessEvictable(i, victim)) victim = i;
            }
            evictedTag = entries_[victim].tag; // every way in this set is valid when !foundFree
        }
        entries_[victim] = EntryT{};
        entries_[victim].valid = true;
        entries_[victim].tag = tag;
        lastTouched_[victim] = ++setCounters_[set];
        return AcquireResult{entries_[victim], victim, /*wasHit=*/false, evictedTag};
    }

private:
    std::size_t SetIndex(Addr tag) const {
        return static_cast<std::size_t>((tag >> kPcAlignmentShiftBits) & (numSets_ - 1));
    }

    // EntryT::EvictionPriority() after applying staleness decay: halved
    // for each doubling of decayIdleUnit_-sized idle blocks THIS SET has
    // seen since entry 'i' was last touched (idle < decayIdleUnit_ -> no
    // decay at all; idle in [decayIdleUnit_, 2*decayIdleUnit_) -> one
    // halving; next doubling -> two; ...). Measuring idle time per-set
    // rather than table-wide is what keeps this self-scaling with a
    // single fixed rate (contention for a busy set decays faster, a
    // quiet set barely decays at all) -- decayIdleUnit_ is the one
    // knob the formula is built around (see
    // PredictorConfig::...DecayIdleUnit). Only called by the two
    // policies that use it; kLru never reaches this.
    std::int64_t DecayedPriority(std::size_t i, std::size_t set) const {
        std::int64_t raw = entries_[i].EvictionPriority();
        if (raw <= 0) return raw;
        std::uint64_t idle = setCounters_[set] - lastTouched_[i];
        std::uint64_t units = idle / decayIdleUnit_;
        if (units == 0) return raw;
        unsigned halvings = 0;
        for (std::uint64_t t = units; t > 0; t >>= 1) ++halvings;
        return raw >> halvings;
    }

    // True if way 'a' should be evicted before way 'b'. Both must be in
    // the same set (the only way this is ever called). Branches on
    // policy_ -- the three EvictionPolicy values are mutually exclusive,
    // non-overlapping victim-selection rules, not layered on each other.
    bool LessEvictable(std::size_t a, std::size_t b) const {
        std::size_t set = a / ways_;
        switch (policy_) {
            case EvictionPolicy::kLru:
                return lastTouched_[a] < lastTouched_[b];
            case EvictionPolicy::kDecayedHitCount:
                // No recency tiebreak at all, by design: a tie here
                // just keeps whichever way LessEvictable was called
                // with first (fixed way order), never lastTouched_.
                return DecayedPriority(a, set) < DecayedPriority(b, set);
            case EvictionPolicy::kDecayedPriorityThenLru:
            default: {
                std::int64_t pa = DecayedPriority(a, set);
                std::int64_t pb = DecayedPriority(b, set);
                if (pa != pb) return pa < pb;
                return lastTouched_[a] < lastTouched_[b];
            }
        }
    }

    std::vector<EntryT> entries_;
    mutable std::vector<std::uint64_t> lastTouched_;
    mutable std::vector<std::uint64_t> setCounters_; // one per set, not table-wide
    mutable EvictionPolicy policy_ = EvictionPolicy::kDecayedPriorityThenLru;
    mutable std::uint64_t decayIdleUnit_ = 128;
    std::size_t numSets_ = 1;
    unsigned ways_ = 1;
};

// The knobs the listener may change. The socket line carries these
// fields; nothing outside this file constructs the struct.
struct PredictorLiveConfig {
    bool enablePrediction = true;
    std::int32_t criticalDistanceCycles = 2;
    std::int32_t nonCriticalDistanceCycles = 8;
    std::int32_t promoteCriticalAt = 5;
    std::int32_t promoteNonCriticalAt = -3;
    std::int32_t trainCounterMin = -10;
    std::int32_t trainCounterMax = 40;
    bool enableCriticalRateGovernor = true;
    double targetCriticalRate = 0.10;
    double criticalRateDeadband = 0.01;
    std::uint32_t criticalRateControlInterval = 1024;
    std::uint32_t criticalRateRelaxDwellWindows = 4;
    EvictionPolicy trainingTableEvictionPolicy = EvictionPolicy::kLru;
    EvictionPolicy predictionTableEvictionPolicy = EvictionPolicy::kDecayedPriorityThenLru;
    std::uint64_t trainingTableDecayIdleUnit = 128;
    std::uint64_t predictionTableDecayIdleUnit = 128;
};

} // namespace

struct CriticalityPredictor::Impl {
    explicit Impl(const PredictorConfig& c) : cfg(c) {
        if (cfg.trainingTableWays == 0) {
            throw std::invalid_argument("PredictorConfig::trainingTableWays must be >= 1");
        }
        if (cfg.predictionTableWays == 0) {
            throw std::invalid_argument("PredictorConfig::predictionTableWays must be >= 1");
        }
        if (cfg.trainingTableDecayIdleUnit == 0) {
            throw std::invalid_argument("PredictorConfig::trainingTableDecayIdleUnit must be >= 1");
        }
        if (cfg.predictionTableDecayIdleUnit == 0) {
            throw std::invalid_argument("PredictorConfig::predictionTableDecayIdleUnit must be >= 1");
        }
        if (cfg.enableCriticalRateGovernor && cfg.criticalRateControlInterval == 0) {
            throw std::invalid_argument(
                "PredictorConfig::criticalRateControlInterval must be >= 1 when "
                "enableCriticalRateGovernor is true");
        }
        if (cfg.enableCriticalRateGovernor && cfg.criticalRateRelaxDwellWindows == 0) {
            throw std::invalid_argument(
                "PredictorConfig::criticalRateRelaxDwellWindows must be >= 1 when "
                "enableCriticalRateGovernor is true");
        }
        if (cfg.distanceClassInterval == 0) {
            throw std::invalid_argument("PredictorConfig::distanceClassInterval must be >= 1");
        }

        trainTable.Configure(cfg.trainingTableEntriesLog2, cfg.trainingTableWays,
                              cfg.trainingTableEvictionPolicy, cfg.trainingTableDecayIdleUnit);
        predTable.Configure(cfg.predictionTableEntriesLog2, cfg.predictionTableWays,
                             cfg.predictionTableEvictionPolicy, cfg.predictionTableDecayIdleUnit);

        // Floor == the host's own configured value; see
        // PredictorConfig::enableCriticalRateGovernor. Equal to
        // cfg.promoteCriticalAt and never updated when the governor is
        // disabled, so UpdateIssueqReadyWaitingTime() below can use this field unconditionally.
        promoteCriticalAtEffective = cfg.promoteCriticalAt;
        rateWindowLimit = cfg.criticalRateControlInterval;
        OpenTraceLog();

        // Logged once up front, including when the governor is off and
        // this value never moves. Later changes are logged from
        // UpdateRateGovernor(); without this initial line a constant
        // threshold never appears in the trace at all.
        if (cfg.enableTraceLog && traceLog_) {
            std::fprintf(traceLog_,
                         "%s table=train event=effective_change value=%d rate=%.4f decode=%llu\n",
                         kTraceLogPrefix, promoteCriticalAtEffective, 0.0,
                         static_cast<unsigned long long>(lookupCounter));
            // The signed level the counter compares against
            // (promoteNonCriticalAt). Negative in the default config.
            std::fprintf(traceLog_,
                         "%s table=train event=demote_threshold value=%d decode=%llu\n",
                         kTraceLogPrefix, cfg.promoteNonCriticalAt,
                         static_cast<unsigned long long>(lookupCounter));
            std::fflush(traceLog_);
        }
        StartLiveControl();
    }

    ~Impl() {
        StopLiveControl();
        if (traceLog_ != nullptr) {
            std::fclose(traceLog_);
            traceLog_ = nullptr;
        }
    }

    void OpenTraceLog() {
        if (!cfg.enableTraceLog) return;
        if (cfg.traceLogPath == nullptr || cfg.traceLogPath[0] == '\0') return;
        // Create the one log file at this path.
        ::unlink(cfg.traceLogPath);
        traceLog_ = std::fopen(cfg.traceLogPath, "w");
        if (traceLog_ == nullptr) {
            throw std::runtime_error(std::string("could not open trace log ") + cfg.traceLogPath);
        }
    }

    // --- Training / prediction tables -----------------------------------
    struct TrainEntry {
        bool valid = false;
        Addr tag = 0;
        std::int32_t count = 0; // directional confidence, unrelated to eviction

        // How many times this PC has actually been (re)trained (an
        // Acquire() hit) since this entry was last allocated -- the
        // training table's own "usefulness" signal, independent of
        // 'count' (which tracks DIRECTION -- critical vs not -- and
        // saturates quickly, so it can't distinguish a PC trained once
        // from one trained fifty times). Only consulted when
        // trainingTableEvictionPolicy == kDecayedHitCount; kLru ignores
        // this field entirely. mutable for the same reason
        // PredEntry::hitCount is: Acquire()'s hit path calls
        // OnAccessed() on what is otherwise a payload write, because for
        // this table a (re)training sample IS the "was this used" event.
        mutable std::uint32_t hitCount = 0;

        std::int64_t EvictionPriority() const { return static_cast<std::int64_t>(hitCount); }
        void OnAccessed() const {
            if (hitCount < std::numeric_limits<std::uint32_t>::max()) ++hitCount;
        }
    };
    struct PredEntry {
        bool valid = false;
        Addr tag = 0;
        bool critical = false;

        // How many times this prediction has actually been consulted
        // (via Find() -- i.e. IsPredictedCritical(), a real fetch-time
        // lookup) since it was last (re)promoted. This is the
        // "usefulness" signal the training table has no equivalent of:
        // a prediction leaned on heavily is worth protecting even if a
        // colliding PC was touched more recently -- SetAssociativeTable
        // additionally decays this by how idle the entry's own SET has
        // been (see DecayedPriority), so heavy-but-old usage doesn't
        // earn permanent protection once it's no longer relevant.
        //
        // mutable: Find() is logically read-only from the caller's
        // perspective (it's a lookup, not a write), but reading a
        // prediction is itself the event that makes it "used" -- the
        // same justification SetAssociativeTable already uses for its
        // own lastTouched_/accessCounter_ bookkeeping.
        mutable std::uint32_t hitCount = 0;

        std::int64_t EvictionPriority() const { return static_cast<std::int64_t>(hitCount); }
        void OnAccessed() const {
            if (hitCount < std::numeric_limits<std::uint32_t>::max()) ++hitCount;
        }
    };
    SetAssociativeTable<TrainEntry> trainTable;
    SetAssociativeTable<PredEntry> predTable;

    // Mutable so a lookup, which is const to the caller, can apply a
    // live-knob update that arrived on the control socket.
    mutable PredictorConfig cfg;
    PredictorStatPointers statPtrs; // all null until SetupStats() is called

    // Local, predictor-owned call counter, used only as the "decode="
    // timestamp field in trace log lines -- this predictor has no idea
    // whether it's being driven by TokenEngine's sampled decode stream
    // or an "ideal" driver feeding it every instruction directly, so it
    // keeps its own notion of "time passing," bumped on every
    // IsPredictedCritical() call (the one call BOTH kinds of driver make
    // once per dynamic instruction, unlike UpdateIssueqReadyWaitingTime()'s
    // frequency, which varies hugely by sampling strategy).
    mutable std::uint64_t lookupCounter = 0;

    // --- Criticality-rate governor ---------------------------------------
    // See PredictorConfig::enableCriticalRateGovernor. Set to
    // cfg.promoteCriticalAt in the constructor and never touched again
    // when the governor is disabled, so UpdateIssueqReadyWaitingTime() can
    // read this unconditionally either way. mutable: updated from inside
    // LookupPrediction(), which is logically read-only to its caller --
    // same justification as SetAssociativeTable's own recency/decay
    // bookkeeping.
    mutable std::int32_t promoteCriticalAtEffective = 0;
    mutable std::uint32_t rateWindowHits = 0;
    mutable std::uint32_t rateWindowTotal = 0;
    mutable std::uint32_t rateGoodWindowStreak = 0; // see criticalRateRelaxDwellWindows
    // Length of the window already open. A live change of
    // criticalRateControlInterval takes effect on the next window.
    mutable std::uint32_t rateWindowLimit = 1;

    // Distance-class window. See PredictorConfig::distanceClassInterval.
    // Counts every training sample, including the middle band that
    // returns before the counter is touched. Reset after each logged
    // window. Fixed length; a live distance-threshold update changes
    // how later samples are classified, not the window length.
    std::uint32_t classWindowCritical = 0;
    std::uint32_t classWindowNonCritical = 0;
    std::uint32_t classWindowNormal = 0;

    void NoteDistanceClass(std::int64_t waitingTime) {
        if (!cfg.enableTraceLog) return;
        if (waitingTime <= cfg.criticalDistanceCycles) {
            ++classWindowCritical;
        } else if (waitingTime >= cfg.nonCriticalDistanceCycles) {
            ++classWindowNonCritical;
        } else {
            ++classWindowNormal;
        }
        std::uint32_t total = classWindowCritical + classWindowNonCritical + classWindowNormal;
        if (total < cfg.distanceClassInterval) return;
        if (traceLog_ && total != 0) {
            double inv = 1.0 / static_cast<double>(total);
            std::fprintf(traceLog_,
                         "%s table=train event=distance_class criticalRatio=%.6f "
                         "nonCriticalRatio=%.6f normalRatio=%.6f decode=%llu\n",
                         kTraceLogPrefix,
                         static_cast<double>(classWindowCritical) * inv,
                         static_cast<double>(classWindowNonCritical) * inv,
                         static_cast<double>(classWindowNormal) * inv,
                         static_cast<unsigned long long>(lookupCounter));
            std::fflush(traceLog_);
        }
        classWindowCritical = 0;
        classWindowNonCritical = 0;
        classWindowNormal = 0;
    }

    void UpdateIssueqReadyWaitingTime(Addr producerPc, std::int64_t waitingTime) {
        DrainLiveUpdates();
        NoteDistanceClass(waitingTime);
        auto acq = trainTable.Acquire(producerPc);
        TrainEntry& e = acq.entry;
        TraceLogAcquire(cfg.enableTraceLog ? traceLog_ : nullptr, "train", acq.index,
                        acq.wasHit, acq.evictedTag, producerPc, lookupCounter);

        if (waitingTime <= cfg.criticalDistanceCycles) {
            e.count = ClampAdd(e.count, +1, cfg.trainCounterMin, cfg.trainCounterMax);
        } else if (waitingTime >= cfg.nonCriticalDistanceCycles) {
            e.count = ClampAdd(e.count, -1, cfg.trainCounterMin, cfg.trainCounterMax);
        } else {
            return; // ambiguous zone: no update, no promotion check
        }
        IncrementIfSet(statPtrs.trainingEvents);
        if (cfg.enableTraceLog) {
            std::fprintf(traceLog_,
                         "%s table=train slot=%zu event=update pc=0x%llx count=%d decode=%llu\n",
                         kTraceLogPrefix, acq.index, static_cast<unsigned long long>(producerPc), e.count,
                         static_cast<unsigned long long>(lookupCounter));
            std::fflush(traceLog_);
        }

        // promoteCriticalAtEffective == cfg.promoteCriticalAt whenever
        // the governor is disabled (never updated in that case) or the
        // observed rate is within its cap -- see
        // PredictorConfig::enableCriticalRateGovernor.
        if (e.count >= promoteCriticalAtEffective) {
            Promote(producerPc, true);
        } else if (e.count <= cfg.promoteNonCriticalAt) {
            Promote(producerPc, false);
        }
    }

    void Promote(Addr pc, bool critical) {
        auto acq = predTable.Acquire(pc); // valid/tag already correct either way
        PredEntry& e = acq.entry;
        TraceLogAcquire(cfg.enableTraceLog ? traceLog_ : nullptr, "pred", acq.index,
                        acq.wasHit, acq.evictedTag, pc, lookupCounter);
        e.critical = critical;
        if (critical) IncrementIfSet(statPtrs.criticalPromotions);
        else IncrementIfSet(statPtrs.nonCriticalPromotions);
        if (cfg.enableTraceLog) {
            std::fprintf(traceLog_,
                         "%s table=pred slot=%zu event=%s pc=0x%llx hitCount=%u decode=%llu\n",
                         kTraceLogPrefix, acq.index, critical ? "promote" : "demote",
                         static_cast<unsigned long long>(pc), e.hitCount,
                         static_cast<unsigned long long>(lookupCounter));
            std::fflush(traceLog_);
        }
    }

    bool LookupPrediction(Addr pc) const {
        DrainLiveUpdates();
        ++lookupCounter;
        auto found = predTable.Find(pc);
        // The REAL, unmasked verdict -- the governor calibrates and the
        // trace log reports against this, not the masked return value,
        // so disabling output (enablePrediction=false) never perturbs
        // the governor's learned threshold or hides what the predictor
        // is actually deciding from someone inspecting the trace log.
        bool critical = found.entry ? found.entry->critical : false; // unknown PC -> non-critical
        if (cfg.enableCriticalRateGovernor) {
            UpdateRateGovernor(critical);
        }
        if (cfg.enableTraceLog) {
            std::fprintf(traceLog_,
                         "%s table=pred slot=%lld event=lookup pc=0x%llx critical=%d hitCount=%u decode=%llu\n",
                         kTraceLogPrefix,
                         found.entry ? static_cast<long long>(found.index) : -1LL,
                         static_cast<unsigned long long>(pc), critical ? 1 : 0,
                         found.entry ? found.entry->hitCount : 0u,
                         static_cast<unsigned long long>(lookupCounter));
            std::fflush(traceLog_);
        }
        // Only the RETURNED value is masked -- see
        // PredictorConfig::enablePrediction.
        return cfg.enablePrediction && critical;
    }

    // Caps -- never targets -- the fraction of LookupPrediction() calls
    // that return true, by making promoteCriticalAtEffective
    // progressively stricter than its floor (cfg.promoteCriticalAt)
    // when over the cap, and relaxing it back toward -- never past --
    // that floor when comfortably under. This is a one-sided cap, not a
    // setpoint to converge on exactly: once at the floor, being under
    // cap is simply left alone.
    //
    // Asymmetric response: tightening is immediate (any single over-cap
    // window), but relaxing requires criticalRateRelaxDwellWindows
    // CONSECUTIVE under-cap windows first (see its comment in
    // criticality_predictor.h) -- without this, a sharp tighten-then-crash
    // produces a clean bang-bang limit cycle that averages out far
    // above the cap despite every individual decision looking correct
    // in isolation.
    void UpdateRateGovernor(bool critical) const {
        ++rateWindowTotal;
        if (critical) ++rateWindowHits;
        if (rateWindowTotal < rateWindowLimit) return;

        double rate = static_cast<double>(rateWindowHits) / rateWindowTotal;
        std::int32_t before = promoteCriticalAtEffective;
        if (rate > cfg.targetCriticalRate + cfg.criticalRateDeadband) {
            promoteCriticalAtEffective =
                std::min(promoteCriticalAtEffective + 1, cfg.trainCounterMax);
            rateGoodWindowStreak = 0;
        } else if (rate < cfg.targetCriticalRate - cfg.criticalRateDeadband &&
                   promoteCriticalAtEffective > cfg.promoteCriticalAt) {
            ++rateGoodWindowStreak;
            if (rateGoodWindowStreak >= cfg.criticalRateRelaxDwellWindows) {
                promoteCriticalAtEffective =
                    std::max(promoteCriticalAtEffective - 1, cfg.promoteCriticalAt);
                rateGoodWindowStreak = 0;
            }
        } else {
            rateGoodWindowStreak = 0; // in the deadband: neutral, not "good"
        }
        if (cfg.enableTraceLog && traceLog_) {
            // Same window the governor just closed. The chart plots this
            // rate and does not keep a second counter of its own.
            std::fprintf(traceLog_,
                         "%s table=pred event=criticality rate=%.6f decode=%llu\n",
                         kTraceLogPrefix, rate,
                         static_cast<unsigned long long>(lookupCounter));
            if (promoteCriticalAtEffective != before) {
                std::fprintf(traceLog_,
                             "%s table=train event=effective_change value=%d rate=%.4f decode=%llu\n",
                             kTraceLogPrefix, promoteCriticalAtEffective, rate,
                             static_cast<unsigned long long>(lookupCounter));
            }
            std::fflush(traceLog_);
        }
        rateWindowHits = 0;
        rateWindowTotal = 0;
        rateWindowLimit = cfg.criticalRateControlInterval;
    }

    void TraceLogHostStats(const HostTraceStats& stats) const {
        if (!cfg.enableTraceLog) return;
        std::fprintf(traceLog_,
                     "%s table=host ninst=%lld decode=%llu\n",
                     kTraceLogPrefix, static_cast<long long>(stats.ninst),
                     static_cast<unsigned long long>(lookupCounter));
        std::fflush(traceLog_);
    }

    // --- Live knobs ------------------------------------------------------
    struct LiveJob {
        bool apply = false;
        PredictorLiveConfig cfg;
    };

    mutable std::mutex liveMu_;
    mutable std::deque<LiveJob> liveQueue_;
    std::atomic<bool> liveStop_{false};
    std::atomic<int> liveClientFd_{-1};
    int liveListenFd_ = -1;
    std::FILE* traceLog_ = nullptr;
    std::string livePath_;
    std::thread liveThread_;

    static bool ParseI32(const std::string& s, std::int32_t& out) {
        char* end = nullptr;
        errno = 0;
        long v = std::strtol(s.c_str(), &end, 10);
        if (end == s.c_str() || *end != '\0' || errno != 0) return false;
        if (v < std::numeric_limits<std::int32_t>::min() ||
            v > std::numeric_limits<std::int32_t>::max()) {
            return false;
        }
        out = static_cast<std::int32_t>(v);
        return true;
    }

    static bool ParseU32(const std::string& s, std::uint32_t& out) {
        char* end = nullptr;
        errno = 0;
        unsigned long v = std::strtoul(s.c_str(), &end, 10);
        if (s.empty() || s[0] == '-' || end == s.c_str() || *end != '\0' || errno != 0) return false;
        if (v > std::numeric_limits<std::uint32_t>::max()) return false;
        out = static_cast<std::uint32_t>(v);
        return true;
    }

    static bool ParseU64(const std::string& s, std::uint64_t& out) {
        char* end = nullptr;
        errno = 0;
        unsigned long long v = std::strtoull(s.c_str(), &end, 10);
        if (s.empty() || s[0] == '-' || end == s.c_str() || *end != '\0' || errno != 0) return false;
        out = static_cast<std::uint64_t>(v);
        return true;
    }

    static bool ParseBool01(const std::string& s, bool& out) {
        if (s == "0") { out = false; return true; }
        if (s == "1") { out = true; return true; }
        return false;
    }

    static bool ParseDouble01(const std::string& s, double& out) {
        char* end = nullptr;
        errno = 0;
        double v = std::strtod(s.c_str(), &end);
        if (end == s.c_str() || *end != '\0' || errno != 0) return false;
        if (v < 0.0 || v > 1.0) return false;
        out = v;
        return true;
    }

    static bool ParsePolicy(const std::string& s, EvictionPolicy& out) {
        if (s == "kLru") { out = EvictionPolicy::kLru; return true; }
        if (s == "kDecayedHitCount") { out = EvictionPolicy::kDecayedHitCount; return true; }
        if (s == "kDecayedPriorityThenLru") {
            out = EvictionPolicy::kDecayedPriorityThenLru;
            return true;
        }
        return false;
    }

    static const char* PolicyName(EvictionPolicy policy) {
        switch (policy) {
            case EvictionPolicy::kLru: return "kLru";
            case EvictionPolicy::kDecayedHitCount: return "kDecayedHitCount";
            case EvictionPolicy::kDecayedPriorityThenLru: return "kDecayedPriorityThenLru";
        }
        return "kLru";
    }

    static bool ParseLiveLine(const std::string& line, PredictorLiveConfig& out) {
        std::istringstream in(line);
        std::string tok;
        std::map<std::string, std::string> kv;
        while (in >> tok) {
            if (!tok.empty() && tok.back() == '\r') tok.pop_back();
            if (tok.empty()) continue;
            auto eq = tok.find('=');
            if (eq == std::string::npos || eq == 0 || eq + 1 == tok.size()) return false;
            if (!kv.emplace(tok.substr(0, eq), tok.substr(eq + 1)).second) return false;
        }
        if (kv.size() != 16) return false;
        PredictorLiveConfig parsed;
        if (!ParseBool01(kv["enablePrediction"], parsed.enablePrediction)) return false;
        if (!ParseI32(kv["criticalDistanceCycles"], parsed.criticalDistanceCycles)) return false;
        if (!ParseI32(kv["nonCriticalDistanceCycles"], parsed.nonCriticalDistanceCycles)) return false;
        if (!ParseI32(kv["promoteCriticalAt"], parsed.promoteCriticalAt)) return false;
        if (!ParseI32(kv["promoteNonCriticalAt"], parsed.promoteNonCriticalAt)) return false;
        if (!ParseI32(kv["trainCounterMin"], parsed.trainCounterMin)) return false;
        if (!ParseI32(kv["trainCounterMax"], parsed.trainCounterMax)) return false;
        if (!ParseBool01(kv["enableCriticalRateGovernor"], parsed.enableCriticalRateGovernor)) return false;
        if (!ParseDouble01(kv["targetCriticalRate"], parsed.targetCriticalRate)) return false;
        if (!ParseDouble01(kv["criticalRateDeadband"], parsed.criticalRateDeadband)) return false;
        if (!ParseU32(kv["criticalRateControlInterval"], parsed.criticalRateControlInterval)) return false;
        if (!ParseU32(kv["criticalRateRelaxDwellWindows"], parsed.criticalRateRelaxDwellWindows)) return false;
        if (!ParsePolicy(kv["trainingTableEvictionPolicy"], parsed.trainingTableEvictionPolicy)) return false;
        if (!ParsePolicy(kv["predictionTableEvictionPolicy"], parsed.predictionTableEvictionPolicy)) return false;
        if (!ParseU64(kv["trainingTableDecayIdleUnit"], parsed.trainingTableDecayIdleUnit)) return false;
        if (!ParseU64(kv["predictionTableDecayIdleUnit"], parsed.predictionTableDecayIdleUnit)) return false;
        // operator[] inserts a missing key as "", which then fails to parse.
        // A missing key also makes size != 16 only when no extra key replaces it.
        // Reject a blank insertion by requiring every lookup to have been present:
        // size is 16 and each Parse saw a real token. An absent key yields "" and fails.
        out = parsed;
        return LiveConfigOk(out);
    }

    static bool LiveConfigOk(const PredictorLiveConfig& live) {
        if (live.trainCounterMin > live.trainCounterMax) return false;
        if (live.criticalDistanceCycles > live.nonCriticalDistanceCycles) return false;
        if (live.trainingTableDecayIdleUnit == 0 || live.predictionTableDecayIdleUnit == 0) return false;
        if (live.targetCriticalRate < 0.0 || live.targetCriticalRate > 1.0) return false;
        if (live.criticalRateDeadband < 0.0 || live.criticalRateDeadband > 1.0) return false;
        if (live.enableCriticalRateGovernor &&
            (live.criticalRateControlInterval == 0 || live.criticalRateRelaxDwellWindows == 0)) {
            return false;
        }
        return true;
    }

    PredictorLiveConfig SnapshotLive() const {
        PredictorLiveConfig s;
        s.enablePrediction = cfg.enablePrediction;
        s.criticalDistanceCycles = cfg.criticalDistanceCycles;
        s.nonCriticalDistanceCycles = cfg.nonCriticalDistanceCycles;
        s.promoteCriticalAt = cfg.promoteCriticalAt;
        s.promoteNonCriticalAt = cfg.promoteNonCriticalAt;
        s.trainCounterMin = cfg.trainCounterMin;
        s.trainCounterMax = cfg.trainCounterMax;
        s.enableCriticalRateGovernor = cfg.enableCriticalRateGovernor;
        s.targetCriticalRate = cfg.targetCriticalRate;
        s.criticalRateDeadband = cfg.criticalRateDeadband;
        s.criticalRateControlInterval = cfg.criticalRateControlInterval;
        s.criticalRateRelaxDwellWindows = cfg.criticalRateRelaxDwellWindows;
        s.trainingTableEvictionPolicy = cfg.trainingTableEvictionPolicy;
        s.predictionTableEvictionPolicy = cfg.predictionTableEvictionPolicy;
        s.trainingTableDecayIdleUnit = cfg.trainingTableDecayIdleUnit;
        s.predictionTableDecayIdleUnit = cfg.predictionTableDecayIdleUnit;
        return s;
    }

    void DumpLive() const {
        if (!cfg.enableTraceLog || !traceLog_) return;
        PredictorLiveConfig s = SnapshotLive();
        std::fprintf(traceLog_,
                     "%s table=live enablePrediction=%d criticalDistanceCycles=%d "
                     "nonCriticalDistanceCycles=%d promoteCriticalAt=%d promoteNonCriticalAt=%d "
                     "trainCounterMin=%d trainCounterMax=%d enableCriticalRateGovernor=%d "
                     "targetCriticalRate=%.6f criticalRateDeadband=%.6f "
                     "criticalRateControlInterval=%u criticalRateRelaxDwellWindows=%u "
                     "trainingTableEvictionPolicy=%s predictionTableEvictionPolicy=%s "
                     "trainingTableDecayIdleUnit=%llu predictionTableDecayIdleUnit=%llu "
                     "decode=%llu\n",
                     kTraceLogPrefix,
                     s.enablePrediction ? 1 : 0,
                     s.criticalDistanceCycles,
                     s.nonCriticalDistanceCycles,
                     s.promoteCriticalAt,
                     s.promoteNonCriticalAt,
                     s.trainCounterMin,
                     s.trainCounterMax,
                     s.enableCriticalRateGovernor ? 1 : 0,
                     s.targetCriticalRate,
                     s.criticalRateDeadband,
                     s.criticalRateControlInterval,
                     s.criticalRateRelaxDwellWindows,
                     PolicyName(s.trainingTableEvictionPolicy),
                     PolicyName(s.predictionTableEvictionPolicy),
                     static_cast<unsigned long long>(s.trainingTableDecayIdleUnit),
                     static_cast<unsigned long long>(s.predictionTableDecayIdleUnit),
                     static_cast<unsigned long long>(lookupCounter));
        std::fflush(traceLog_);
    }

    void ApplyLiveNow(const PredictorLiveConfig& live) const {
        bool governorWasOn = cfg.enableCriticalRateGovernor;
        std::int32_t effectiveBefore = promoteCriticalAtEffective;
        std::int32_t demoteBefore = cfg.promoteNonCriticalAt;
        cfg.enablePrediction = live.enablePrediction;
        cfg.criticalDistanceCycles = live.criticalDistanceCycles;
        cfg.nonCriticalDistanceCycles = live.nonCriticalDistanceCycles;
        cfg.promoteCriticalAt = live.promoteCriticalAt;
        cfg.promoteNonCriticalAt = live.promoteNonCriticalAt;
        cfg.trainCounterMin = live.trainCounterMin;
        cfg.trainCounterMax = live.trainCounterMax;
        cfg.enableCriticalRateGovernor = live.enableCriticalRateGovernor;
        cfg.targetCriticalRate = live.targetCriticalRate;
        cfg.criticalRateDeadband = live.criticalRateDeadband;
        cfg.criticalRateControlInterval = live.criticalRateControlInterval;
        cfg.criticalRateRelaxDwellWindows = live.criticalRateRelaxDwellWindows;
        cfg.trainingTableEvictionPolicy = live.trainingTableEvictionPolicy;
        cfg.predictionTableEvictionPolicy = live.predictionTableEvictionPolicy;
        cfg.trainingTableDecayIdleUnit = live.trainingTableDecayIdleUnit;
        cfg.predictionTableDecayIdleUnit = live.predictionTableDecayIdleUnit;
        trainTable.SetReplacement(live.trainingTableEvictionPolicy, live.trainingTableDecayIdleUnit);
        predTable.SetReplacement(live.predictionTableEvictionPolicy, live.predictionTableDecayIdleUnit);

        if (!cfg.enableCriticalRateGovernor || !governorWasOn) {
            // Off, or just turned on: drop the open window. The new
            // interval applies to the window that starts now.
            rateWindowHits = 0;
            rateWindowTotal = 0;
            rateGoodWindowStreak = 0;
            rateWindowLimit = cfg.criticalRateControlInterval;
            if (!cfg.enableCriticalRateGovernor) {
                promoteCriticalAtEffective = cfg.promoteCriticalAt;
            }
        }
        if (promoteCriticalAtEffective < cfg.promoteCriticalAt) {
            promoteCriticalAtEffective = cfg.promoteCriticalAt;
        }
        if (promoteCriticalAtEffective > cfg.trainCounterMax) {
            promoteCriticalAtEffective = cfg.trainCounterMax;
        }
        // A lower max (or a higher floor) moves the threshold even when
        // the governor's next window would not. Without this line the
        // chart keeps the old value and then looks frozen, because a
        // threshold already sitting on the new max does not move again
        // until the rate falls under the cap.
        if (cfg.enableTraceLog && traceLog_ &&
            promoteCriticalAtEffective != effectiveBefore) {
            double rate = rateWindowTotal == 0
                              ? 0.0
                              : static_cast<double>(rateWindowHits) / rateWindowTotal;
            std::fprintf(traceLog_,
                         "%s table=train event=effective_change value=%d rate=%.4f decode=%llu\n",
                         kTraceLogPrefix, promoteCriticalAtEffective, rate,
                         static_cast<unsigned long long>(lookupCounter));
            std::fflush(traceLog_);
        }
        if (cfg.enableTraceLog && traceLog_ &&
            cfg.promoteNonCriticalAt != demoteBefore) {
            std::fprintf(traceLog_,
                         "%s table=train event=demote_threshold value=%d decode=%llu\n",
                         kTraceLogPrefix, cfg.promoteNonCriticalAt,
                         static_cast<unsigned long long>(lookupCounter));
            std::fflush(traceLog_);
        }
        DumpLive();
    }

    void DrainLiveUpdates() const {
        std::deque<LiveJob> batch;
        {
            std::lock_guard<std::mutex> lock(liveMu_);
            batch.swap(liveQueue_);
        }
        for (const LiveJob& job : batch) {
            if (job.apply && LiveConfigOk(job.cfg)) ApplyLiveNow(job.cfg);
            else DumpLive();
        }
    }

    static void WriteReply(int fd, const char* text) {
        std::size_t left = std::strlen(text);
        const char* p = text;
        while (left > 0) {
            ssize_t n = ::send(fd, p, left, MSG_NOSIGNAL);
            if (n < 0) {
                if (errno == EINTR) continue;
                return;
            }
            p += n;
            left -= static_cast<std::size_t>(n);
        }
    }

    void EnqueueLive(LiveJob job) {
        std::lock_guard<std::mutex> lock(liveMu_);
        if (liveQueue_.size() >= 32) return;
        liveQueue_.push_back(std::move(job));
    }

    void HandleControlLine(int fd, const std::string& line) {
        PredictorLiveConfig parsed;
        if (!ParseLiveLine(line, parsed)) {
            WriteReply(fd, "rejected\n");
            LiveJob job;
            job.apply = false;
            EnqueueLive(std::move(job));
            return;
        }
        LiveJob job;
        job.apply = true;
        job.cfg = parsed;
        {
            std::lock_guard<std::mutex> lock(liveMu_);
            if (liveQueue_.size() >= 32) {
                WriteReply(fd, "rejected\n");
                return;
            }
            liveQueue_.push_back(std::move(job));
        }
        WriteReply(fd, "queued\n");
    }

    void ServeControlClient(int fd) {
        std::string buf;
        char tmp[256];
        while (!liveStop_.load()) {
            pollfd pfd;
            pfd.fd = fd;
            pfd.events = POLLIN;
            pfd.revents = 0;
            int pr = ::poll(&pfd, 1, 200);
            if (pr < 0) {
                if (errno == EINTR) continue;
                break;
            }
            if (pr == 0) continue;
            ssize_t n = ::read(fd, tmp, sizeof tmp);
            if (n <= 0) break;
            buf.append(tmp, tmp + n);
            for (;;) {
                auto nl = buf.find('\n');
                if (nl == std::string::npos) {
                    if (buf.size() > 4096) {
                        WriteReply(fd, "rejected\n");
                        buf.clear();
                    }
                    break;
                }
                std::string line = buf.substr(0, nl);
                buf.erase(0, nl + 1);
                HandleControlLine(fd, line);
            }
        }
    }

    void ControlLoop() {
        while (!liveStop_.load()) {
            int fd = ::accept(liveListenFd_, nullptr, nullptr);
            if (fd < 0) {
                if (liveStop_.load()) break;
                if (errno == EINTR) continue;
                break;
            }
            liveClientFd_.store(fd);
            ServeControlClient(fd);
            int cur = fd;
            if (liveClientFd_.compare_exchange_strong(cur, -1)) ::close(fd);
        }
    }

    void StartLiveControl() {
        if (!cfg.enableTraceLog || !traceLog_) return;
        int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) return;
        static std::atomic<std::uint32_t> seq{0};
        livePath_ = "/tmp/isqcrit-" + std::to_string(::getpid()) + "-" +
                    std::to_string(seq.fetch_add(1)) + ".sock";
        ::unlink(livePath_.c_str());
        sockaddr_un addr;
        std::memset(&addr, 0, sizeof addr);
        addr.sun_family = AF_UNIX;
        if (livePath_.size() >= sizeof addr.sun_path) {
            ::close(fd);
            livePath_.clear();
            return;
        }
        std::memcpy(addr.sun_path, livePath_.c_str(), livePath_.size() + 1);
        if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0 ||
            ::listen(fd, 1) != 0) {
            ::close(fd);
            ::unlink(livePath_.c_str());
            livePath_.clear();
            return;
        }
        ::chmod(livePath_.c_str(), 0600);
        liveListenFd_ = fd;
        std::fprintf(traceLog_,
                     "%s table=control path=%s decode=%llu\n",
                     kTraceLogPrefix, livePath_.c_str(),
                     static_cast<unsigned long long>(lookupCounter));
        std::fflush(traceLog_);
        DumpLive();
        liveThread_ = std::thread([this] { ControlLoop(); });
    }

    void StopLiveControl() {
        liveStop_.store(true);
        int client = liveClientFd_.exchange(-1);
        if (client >= 0) {
            ::shutdown(client, SHUT_RDWR);
            ::close(client);
        }
        if (liveListenFd_ >= 0) {
            ::shutdown(liveListenFd_, SHUT_RDWR);
            ::close(liveListenFd_);
            liveListenFd_ = -1;
        }
        if (liveThread_.joinable()) liveThread_.join();
        if (!livePath_.empty()) {
            ::unlink(livePath_.c_str());
            livePath_.clear();
        }
    }
};

CriticalityPredictor::CriticalityPredictor(const PredictorConfig& cfg)
    : impl_(new Impl(cfg)) {}

CriticalityPredictor::~CriticalityPredictor() { delete impl_; }

void CriticalityPredictor::UpdateIssueqReadyWaitingTime(Addr producerPc, std::int64_t waitingTime) {
    impl_->UpdateIssueqReadyWaitingTime(producerPc, waitingTime);
}

bool CriticalityPredictor::IsPredictedCritical(Addr pc) const {
    return impl_->LookupPrediction(pc);
}

void CriticalityPredictor::SetupStats(const PredictorStatPointers& counters) {
    impl_->statPtrs = counters;
}

void CriticalityPredictor::TraceLogHostStats(const HostTraceStats& stats) {
    impl_->TraceLogHostStats(stats);
}

} // namespace ISQCrit
