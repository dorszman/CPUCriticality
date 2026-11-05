#include "token_engine.h"

#include <stdexcept>
#include <vector>

namespace ISQCrit {

namespace {

// Counter storage lives in the host, not here -- a counter the host
// never wired up via SetupStats() is simply never incremented.
void IncrementIfSet(StatCounter* counter) {
    if (counter) ++(*counter);
}

} // namespace

struct TokenEngine::Impl {
    Impl(const CriticalityConfig& c, TokenEngine* self_)
        : cfg(c), self(self_), predictor(c.predictor) {
        if (cfg.tokenPoolSize == 0) {
            throw std::invalid_argument("CriticalityConfig::tokenPoolSize must be >= 1");
        }
        if (cfg.tokenAssignInterval == 0) {
            throw std::invalid_argument("CriticalityConfig::tokenAssignInterval must be >= 1");
        }
        // Everything else (table sizing, decay units, governor knobs)
        // is validated by CriticalityPredictor's own constructor above,
        // in the member-initializer list -- it throws before this body
        // even runs if cfg.predictor is invalid.

        tokenPool.resize(cfg.tokenPoolSize);
        freeList.reserve(cfg.tokenPoolSize);
        for (std::size_t i = 0; i < tokenPool.size(); ++i) {
            tokenPool[i].engine_ = self;
            tokenPool[i].poolIndex_ = i;
            freeList.push_back(tokenPool.size() - 1 - i); // order doesn't matter
        }
    }

    // --- Token pool ------------------------------------------------------
    std::vector<Token> tokenPool;
    std::vector<std::size_t> freeList;

    CriticalityConfig cfg;
    TokenEngine* self;
    CriticalityPredictor predictor; // all training/prediction logic lives here
    StatCounter* tokensAssignedPtr = nullptr;
    StatCounter* tokensExpiredPtr = nullptr;
    std::uint64_t decodeCounter = 0;
    Cycle currentCycle = 0;

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

        IncrementIfSet(tokensAssignedPtr);
        return TokenRef(&t);
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
        IncrementIfSet(tokensExpiredPtr);
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
        std::int64_t waitingTime =
            static_cast<std::int64_t>(currentCycle) - static_cast<std::int64_t>(token->producerWbTime_);
        predictor.UpdateIssueqReadyWaitingTime(token->pc_, waitingTime);
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
    return impl_->predictor.IsPredictedCritical(pc);
}

void TokenEngine::SetupStats(const CriticalityStatPointers& counters) {
    impl_->tokensAssignedPtr = counters.tokensAssigned;
    impl_->tokensExpiredPtr = counters.tokensExpired;
    impl_->predictor.SetupStats(counters.predictor);
}

bool TokenEngine::UpdateCriticality(Token* token) {
    return impl_->UpdateCriticality(token);
}

void TokenEngine::TokenExpire(Token* token) {
    impl_->TokenExpire(token);
}

} // namespace ISQCrit
