#include "token_engine.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace cacrit {

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

        tokenPool.resize(cfg.tokenPoolSize);
        freeList.reserve(cfg.tokenPoolSize);
        for (std::size_t i = 0; i < tokenPool.size(); ++i) {
            tokenPool[i].engine_ = self;
            tokenPool[i].poolIndex_ = i;
            freeList.push_back(tokenPool.size() - 1 - i); // order doesn't matter
        }

        trainTable.resize(std::size_t{1} << cfg.trainingTableEntriesLog2);
        predTable.resize(std::size_t{1} << cfg.predictionTableEntriesLog2);
    }

    // --- Token pool ------------------------------------------------------
    std::vector<Token> tokenPool;
    std::vector<std::size_t> freeList;

    // --- Training / prediction tables -----------------------------------
    struct TrainEntry {
        bool valid = false;
        Addr tag = 0;
        std::int32_t count = 0;
    };
    struct PredEntry {
        bool valid = false;
        Addr tag = 0;
        bool critical = false;
    };
    std::vector<TrainEntry> trainTable;
    std::vector<PredEntry> predTable;

    CriticalityConfig cfg;
    TokenEngine* self;
    CriticalityStatPointers statPtrs; // all null until SetupStats() is called
    std::uint64_t decodeCounter = 0;
    Cycle currentCycle = 0;

    // Assumes 4-byte-aligned (fixed-width RISC-style) instruction
    // encoding; a real design should size/hash this to its own
    // fetch-group granularity instead of this fixed default.
    static constexpr unsigned kPcAlignmentShiftBits = 2;

    std::size_t IndexFor(Addr pc, unsigned log2Entries) const {
        return static_cast<std::size_t>((pc >> kPcAlignmentShiftBits) &
                                         ((std::uint64_t{1} << log2Entries) - 1));
    }

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
        std::size_t idx = IndexFor(pc, cfg.trainingTableEntriesLog2);
        TrainEntry& e = trainTable[idx];
        if (!e.valid || e.tag != pc) {
            e.valid = true;
            e.tag = pc;
            e.count = 0;
        }
        if (distance <= cfg.criticalDistanceCycles) {
            e.count = ClampAdd(e.count, +1, cfg.trainCounterMin, cfg.trainCounterMax);
        } else if (distance >= cfg.nonCriticalDistanceCycles) {
            e.count = ClampAdd(e.count, -1, cfg.trainCounterMin, cfg.trainCounterMax);
        } else {
            return; // ambiguous zone: no update, no promotion check
        }
        IncrementIfSet(statPtrs.trainingEvents);

        if (e.count >= cfg.promoteCriticalAt) {
            Promote(pc, true);
        } else if (e.count <= cfg.promoteNonCriticalAt) {
            Promote(pc, false);
        }
    }

    void Promote(Addr pc, bool critical) {
        std::size_t idx = IndexFor(pc, cfg.predictionTableEntriesLog2);
        PredEntry& e = predTable[idx];
        e.valid = true;
        e.tag = pc;
        e.critical = critical;
        if (critical) IncrementIfSet(statPtrs.criticalPromotions);
        else IncrementIfSet(statPtrs.nonCriticalPromotions);
    }

    bool LookupPrediction(Addr pc) const {
        std::size_t idx = IndexFor(pc, cfg.predictionTableEntriesLog2);
        const PredEntry& e = predTable[idx];
        if (e.valid && e.tag == pc) return e.critical;
        return false; // unknown PC defaults to non-critical (baseline behavior)
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

} // namespace cacrit
