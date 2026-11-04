#include "token_engine.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

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

// Every CriticalityConfig::enableTraceLog line starts with this exact
// token, deliberately unusual so a consumer (e.g. tools/visualize_trace.py)
// can grep/filter it out of stdout unambiguously even when the host's
// own logging is interleaved on the same stream.
constexpr const char* kTraceLogPrefix = "@@ISQCRIT@@";

// Shared by TrainOn()/Promote(): logs an Acquire() that didn't hit an
// existing entry, i.e. a slot allocation -- always reported as
// event=alloc, whether or not a prior occupant had to be evicted to
// make room (true in steady state for essentially every allocation
// past a table's initial fill); old_pc is simply omitted on the rare
// case where the way was already free, rather than needing a separate
// event type.
void TraceLogAcquire(bool enabled, const char* table, std::size_t index, bool wasHit,
                      bool wasEviction, Addr evictedTag, Addr newPc, std::uint64_t decode) {
    if (!enabled || wasHit) return;
    if (wasEviction) {
        std::printf("%s table=%s slot=%zu event=alloc old_pc=0x%llx new_pc=0x%llx decode=%llu\n",
                    kTraceLogPrefix, table, index,
                    static_cast<unsigned long long>(evictedTag),
                    static_cast<unsigned long long>(newPc),
                    static_cast<unsigned long long>(decode));
    } else {
        std::printf("%s table=%s slot=%zu event=alloc new_pc=0x%llx decode=%llu\n",
                    kTraceLogPrefix, table, index, static_cast<unsigned long long>(newPc),
                    static_cast<unsigned long long>(decode));
    }
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

    // Physical slot index alongside the entry itself -- needed by
    // Impl's trace logging (see CriticalityConfig::enableTraceLog) to
    // identify which of a handful of tracked slots an event belongs to.
    // entry == nullptr on a miss; index is then meaningless.
    struct FindResult {
        const EntryT* entry;
        std::size_t index;
    };

    // Same idea for Acquire(): which physical slot was used, whether it
    // was already holding this same tag (a hit), and -- if not a hit --
    // whether a DIFFERENT, still-valid tag had to be evicted to make
    // room (as opposed to landing in a previously-free way), plus that
    // evicted tag. evictedTag is only meaningful when wasEviction.
    struct AcquireResult {
        EntryT& entry;
        std::size_t index;
        bool wasHit;
        bool wasEviction;
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
                return AcquireResult{entries_[i], i, /*wasHit=*/true, /*wasEviction=*/false, Addr{0}};
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
        bool wasEviction = false;
        Addr evictedTag = 0;
        if (!foundFree) {
            // Victim choice depends on policy_ -- see LessEvictable.
            victim = base;
            for (unsigned w = 1; w < ways_; ++w) {
                std::size_t i = base + w;
                if (LessEvictable(i, victim)) victim = i;
            }
            wasEviction = true; // every way in this set is valid when !foundFree
            evictedTag = entries_[victim].tag;
        }
        entries_[victim] = EntryT{};
        entries_[victim].valid = true;
        entries_[victim].tag = tag;
        lastTouched_[victim] = ++setCounters_[set];
        return AcquireResult{entries_[victim], victim, /*wasHit=*/false, wasEviction, evictedTag};
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
    // CriticalityConfig::decayIdleUnit). Only called by the two policies
    // that use it; kLru never reaches this.
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
    EvictionPolicy policy_ = EvictionPolicy::kDecayedPriorityThenLru;
    std::uint64_t decayIdleUnit_ = 128;
    std::size_t numSets_ = 1;
    unsigned ways_ = 1;
};

} // namespace

struct TokenEngine::Impl {
    Impl(const CriticalityConfig& c, TokenEngine* self_)
        : cfg(c), self(self_) {
        if (cfg.tokenPoolSize == 0) {
            throw std::invalid_argument("CriticalityConfig::tokenPoolSize must be >= 1");
        }
        if (cfg.tokenAssignInterval == 0) {
            throw std::invalid_argument("CriticalityConfig::tokenAssignInterval must be >= 1");
        }
        if (cfg.trainingTableWays == 0) {
            throw std::invalid_argument("CriticalityConfig::trainingTableWays must be >= 1");
        }
        if (cfg.predictionTableWays == 0) {
            throw std::invalid_argument("CriticalityConfig::predictionTableWays must be >= 1");
        }
        if (cfg.trainingTableDecayIdleUnit == 0) {
            throw std::invalid_argument("CriticalityConfig::trainingTableDecayIdleUnit must be >= 1");
        }
        if (cfg.predictionTableDecayIdleUnit == 0) {
            throw std::invalid_argument("CriticalityConfig::predictionTableDecayIdleUnit must be >= 1");
        }
        if (cfg.enableCriticalRateGovernor && cfg.criticalRateControlInterval == 0) {
            throw std::invalid_argument(
                "CriticalityConfig::criticalRateControlInterval must be >= 1 when "
                "enableCriticalRateGovernor is true");
        }
        if (cfg.enableCriticalRateGovernor && cfg.criticalRateRelaxDwellWindows == 0) {
            throw std::invalid_argument(
                "CriticalityConfig::criticalRateRelaxDwellWindows must be >= 1 when "
                "enableCriticalRateGovernor is true");
        }

        tokenPool.resize(cfg.tokenPoolSize);
        freeList.reserve(cfg.tokenPoolSize);
        for (std::size_t i = 0; i < tokenPool.size(); ++i) {
            tokenPool[i].engine_ = self;
            tokenPool[i].poolIndex_ = i;
            freeList.push_back(tokenPool.size() - 1 - i); // order doesn't matter
        }

        trainTable.Configure(cfg.trainingTableEntriesLog2, cfg.trainingTableWays,
                              cfg.trainingTableEvictionPolicy, cfg.trainingTableDecayIdleUnit);
        predTable.Configure(cfg.predictionTableEntriesLog2, cfg.predictionTableWays,
                             cfg.predictionTableEvictionPolicy, cfg.predictionTableDecayIdleUnit);

        // Floor == the host's own configured value; see
        // CriticalityConfig::enableCriticalRateGovernor. Equal to
        // cfg.promoteCriticalAt and never updated when the governor is
        // disabled, so TrainOn() below can use this field unconditionally.
        promoteCriticalAtEffective = cfg.promoteCriticalAt;
    }

    // --- Token pool ------------------------------------------------------
    std::vector<Token> tokenPool;
    std::vector<std::size_t> freeList;

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

    CriticalityConfig cfg;
    TokenEngine* self;
    CriticalityStatPointers statPtrs; // all null until SetupStats() is called
    std::uint64_t decodeCounter = 0;
    Cycle currentCycle = 0;

    // --- Criticality-rate governor ---------------------------------------
    // See CriticalityConfig::enableCriticalRateGovernor. Set to
    // cfg.promoteCriticalAt in the constructor and never touched again
    // when the governor is disabled, so TrainOn() can read this
    // unconditionally either way. mutable: updated from inside
    // LookupPrediction(), which is logically read-only to its caller --
    // same justification as SetAssociativeTable's own recency/decay
    // bookkeeping.
    mutable std::int32_t promoteCriticalAtEffective = 0;
    mutable std::uint32_t rateWindowHits = 0;
    mutable std::uint32_t rateWindowTotal = 0;
    mutable std::uint32_t rateGoodWindowStreak = 0; // see criticalRateRelaxDwellWindows

    // Called once per decoded uop, unconditionally. Always advances the
    // sampling counter first (so it tracks decoded uops, not granted
    // tokens), then decides in the SAME call whether this decode earns a
    // token -- counting and gating are not split across two calls, so
    // there is no way for the counter to advance only on grants (which
    // would stall it the moment the gate ever says no).
    TokenRef DecodeGetToken(Addr pc) {
        ++decodeCounter;
        if (freeList.empty()) {
            return TokenRef{};
        }
        if (!cfg.pureAvailabilityGating &&
            (decodeCounter % cfg.tokenAssignInterval) != 0) {
            return TokenRef{};
        }

        std::size_t idx = freeList.back();
        freeList.pop_back();

        Token& t = tokenPool[idx];
        t.pc_ = pc;
        t.state_ = Token::State::PendingProducerExe; // Invalid -> PendingProducerExe
        t.producerWbTime_ = 0;
        t.refcount_ = 0; // TokenRef ctor below brings this to 1
        // t.epoch_ is deliberately left untouched here: whichever path
        // most recently returned this slot to the pool (RecycleSlot, via
        // either a closed measurement or refcount reaching zero) already
        // bumped it, and the TokenRef constructed below captures that
        // current value as this new assignment's epoch.

        IncrementIfSet(statPtrs.tokensAssigned);
        return TokenRef(&t);
    }

    void TrainOn(Addr pc, std::int64_t distance) {
        auto acq = trainTable.Acquire(pc);
        TrainEntry& e = acq.entry;
        TraceLogAcquire(cfg.enableTraceLog, "train", acq.index, acq.wasHit, acq.wasEviction,
                        acq.evictedTag, pc, decodeCounter);

        if (distance <= cfg.criticalDistanceCycles) {
            e.count = ClampAdd(e.count, +1, cfg.trainCounterMin, cfg.trainCounterMax);
        } else if (distance >= cfg.nonCriticalDistanceCycles) {
            e.count = ClampAdd(e.count, -1, cfg.trainCounterMin, cfg.trainCounterMax);
        } else {
            return; // ambiguous zone: no update, no promotion check
        }
        IncrementIfSet(statPtrs.trainingEvents);
        if (cfg.enableTraceLog) {
            std::printf("%s table=train slot=%zu event=update pc=0x%llx count=%d decode=%llu\n",
                        kTraceLogPrefix, acq.index, static_cast<unsigned long long>(pc), e.count,
                        static_cast<unsigned long long>(decodeCounter));
        }

        // promoteCriticalAtEffective == cfg.promoteCriticalAt whenever
        // the governor is disabled (never updated in that case) or the
        // observed rate is within its cap -- see
        // CriticalityConfig::enableCriticalRateGovernor.
        if (e.count >= promoteCriticalAtEffective) {
            Promote(pc, true);
        } else if (e.count <= cfg.promoteNonCriticalAt) {
            Promote(pc, false);
        }
    }

    void Promote(Addr pc, bool critical) {
        auto acq = predTable.Acquire(pc); // valid/tag already correct either way
        PredEntry& e = acq.entry;
        TraceLogAcquire(cfg.enableTraceLog, "pred", acq.index, acq.wasHit, acq.wasEviction,
                        acq.evictedTag, pc, decodeCounter);
        e.critical = critical;
        if (critical) IncrementIfSet(statPtrs.criticalPromotions);
        else IncrementIfSet(statPtrs.nonCriticalPromotions);
        if (cfg.enableTraceLog) {
            std::printf("%s table=pred slot=%zu event=%s pc=0x%llx hitCount=%u decode=%llu\n",
                        kTraceLogPrefix, acq.index, critical ? "promote" : "demote",
                        static_cast<unsigned long long>(pc), e.hitCount,
                        static_cast<unsigned long long>(decodeCounter));
        }
    }

    bool LookupPrediction(Addr pc) const {
        auto found = predTable.Find(pc);
        bool critical = found.entry ? found.entry->critical : false; // unknown PC -> non-critical
        if (cfg.enableCriticalRateGovernor) {
            UpdateRateGovernor(critical);
        }
        if (cfg.enableTraceLog) {
            std::printf("%s table=pred slot=%lld event=lookup pc=0x%llx critical=%d hitCount=%u decode=%llu\n",
                        kTraceLogPrefix,
                        found.entry ? static_cast<long long>(found.index) : -1LL,
                        static_cast<unsigned long long>(pc), critical ? 1 : 0,
                        found.entry ? found.entry->hitCount : 0u,
                        static_cast<unsigned long long>(decodeCounter));
        }
        return critical;
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
    // token_engine.h) -- without this, a sharp tighten-then-crash
    // produces a clean bang-bang limit cycle that averages out far
    // above the cap despite every individual decision looking correct
    // in isolation.
    void UpdateRateGovernor(bool critical) const {
        ++rateWindowTotal;
        if (critical) ++rateWindowHits;
        if (rateWindowTotal < cfg.criticalRateControlInterval) return;

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
        if (cfg.enableTraceLog && promoteCriticalAtEffective != before) {
            std::printf("%s table=train event=effective_change value=%d rate=%.4f decode=%llu\n",
                        kTraceLogPrefix, promoteCriticalAtEffective, rate,
                        static_cast<unsigned long long>(decodeCounter));
        }
        rateWindowHits = 0;
        rateWindowTotal = 0;
    }

    // Returns a slot to the pool and bumps its epoch, regardless of WHY
    // (refcount hit zero, or the measurement closed and we're recycling
    // immediately without waiting for stragglers). Bumping the epoch here
    // is what makes any TokenRef that still thinks it owns this slot
    // (e.g. a slower fan-out sibling that hasn't executed yet) correctly
    // treat itself as stale from this point on -- see token.h.
    void RecycleSlot(Token* token) {
        token->state_ = Token::State::Invalid;
        token->producerWbTime_ = 0;
        ++token->epoch_;
        freeList.push_back(token->poolIndex_);
        IncrementIfSet(statPtrs.tokensExpired);
    }

    // --- Token callbacks (see TokenRef::NotifyExecuted / Token::Release) -
    // Single-hop lifecycle, matching the original mechanism: a token
    // measures exactly one producer->consumer edge, then its job is done
    // -- no recursion, no re-rooting, ever (pc_ is fixed at creation).
    // Only ever called by TokenRef after it has already confirmed its own
    // epoch is current, so this never needs to check staleness itself.
    //
    // Fan-out: a producer's broadcast may wake several consumers at once;
    // the host hands all of them a copy of the SAME token (refcount > 1
    // is a normal, intended state here, not just a transient during a
    // single hand-off). Whichever of those consumers executes first has,
    // by construction, the shortest distance back to the root -- the
    // strongest possible critical-path evidence -- so the FIRST call that
    // reaches the closing branch below trains and immediately recycles
    // the slot (not waiting for slower siblings to drop their copies);
    // any slower sibling's later call will, by then, fail the epoch check
    // in TokenRef::NotifyExecuted() before it ever reaches here.
    bool UpdateCriticality(Token* token) {
        if (token->state_ == Token::State::PendingProducerExe) {
            token->producerWbTime_ = currentCycle;
            token->state_ = Token::State::PendingConsumerExe;
            return true;
        }
        std::int64_t distance =
            static_cast<std::int64_t>(currentCycle) - static_cast<std::int64_t>(token->producerWbTime_);
        TrainOn(token->pc_, distance);
        RecycleSlot(token); // immediate reuse; do not wait for refcount to reach zero
        return false;
    }

    // Called when a TokenRef holding a still-current reference drops it
    // and refcount reaches zero (the non-racing, no-wakeup, or
    // flushed-before-execute paths -- nothing closed the measurement, so
    // RecycleSlot wasn't already triggered by UpdateCriticality above).
    void TokenExpire(Token* token) {
        RecycleSlot(token);
    }
};

TokenEngine::TokenEngine(const CriticalityConfig& cfg)
    : impl_(new Impl(cfg, this)) {}

TokenEngine::~TokenEngine() { delete impl_; }

void TokenEngine::Tick(Cycle now) {
    impl_->currentCycle = now;
}

TokenRef TokenEngine::DecodeGetToken(Addr pc) {
    return impl_->DecodeGetToken(pc);
}

bool TokenEngine::IsPredictedCritical(Addr pc) const {
    return impl_->LookupPrediction(pc);
}

std::int32_t TokenEngine::PriorityBoost(bool predictedCritical,
                                         std::int32_t boostMagnitude) {
    return predictedCritical ? boostMagnitude : 0;
}

void TokenEngine::SetupStats(const CriticalityStatPointers& counters) {
    impl_->statPtrs = counters;
}

bool TokenEngine::UpdateCriticality(Token* token) {
    return impl_->UpdateCriticality(token);
}

void TokenEngine::TokenExpire(Token* token) {
    impl_->TokenExpire(token);
}

} // namespace ISQCrit
