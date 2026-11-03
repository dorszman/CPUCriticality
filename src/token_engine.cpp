#include "token_engine.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace cacrit {

namespace {

constexpr std::int32_t clampAdd(std::int32_t value, std::int32_t delta,
                                 std::int32_t lo, std::int32_t hi) {
    std::int32_t result = value + delta;
    if (result < lo) return lo;
    if (result > hi) return hi;
    return result;
}

} // namespace

struct TokenEngine::Impl {
    Impl(const CriticalityConfig& c, TokenEngine* self_)
        : cfg(c), self(self_) {
        if (cfg.tokenPoolSize == 0) cfg.tokenPoolSize = 1;
        if (cfg.tokenAssignInterval == 0) cfg.tokenAssignInterval = 1;

        tokenPool.resize(cfg.tokenPoolSize);
        freeList.reserve(cfg.tokenPoolSize);
        for (std::size_t i = 0; i < tokenPool.size(); ++i) {
            tokenPool[i].engine_ = self;
            tokenPool[i].poolIndex_ = i;
            freeList.push_back(tokenPool.size() - 1 - i); // order doesn't matter
        }

        trainTable.resize(std::size_t{1} << cfg.trainingTableEntriesLog2);
        predTable.resize(std::size_t{1} << cfg.predictionTableEntriesLog2);

        rngState = cfg.rngSeed != 0 ? cfg.rngSeed : 0x9E3779B97F4A7C15ULL;
    }

    // --- PRNG (fan-out tie-break fairness only; see DESIGN.md S3.2) -----
    std::uint64_t rngState;
    std::uint64_t nextRandom() {
        std::uint64_t x = rngState;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        rngState = x;
        return x * 0x2545F4914F6CDD1DULL;
    }

    // --- Token pool ------------------------------------------------------
    std::vector<Token> tokenPool;
    std::vector<std::size_t> freeList;

    // --- Dependency graph bookkeeping -----------------------------------
    std::unordered_map<InstrId, std::vector<InstrId>> dependents;
    std::unordered_map<InstrId, std::int32_t> pendingCount;

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
    CriticalityStats stats;
    std::uint64_t decodeCounter = 0;
    bool lastAvailableCheckPassed = false; // guards DecodeGetToken misuse

    std::size_t indexFor(Addr pc, unsigned log2Entries) const {
        return static_cast<std::size_t>((pc >> 2) & ((std::uint64_t{1} << log2Entries) - 1));
    }

    // Called once per decoded uop. Always advances the sampling counter
    // (it tracks decoded uops, not granted tokens) regardless of outcome.
    bool TokenAvailable() {
        ++decodeCounter;
        if (freeList.empty()) {
            lastAvailableCheckPassed = false;
            return false;
        }
        if (!cfg.pureAvailabilityGating &&
            (decodeCounter % cfg.tokenAssignInterval) != 0) {
            lastAvailableCheckPassed = false;
            return false;
        }
        lastAvailableCheckPassed = true;
        return true;
    }

    TokenRef DecodeGetToken(Addr pc) {
        // Defensive fallback: a caller that skips the TokenAvailable()
        // check, or calls again without rechecking, gets an empty handle
        // instead of undefined behavior if the pool is actually empty.
        if (freeList.empty()) {
            return TokenRef{};
        }
        lastAvailableCheckPassed = false;

        std::size_t idx = freeList.back();
        freeList.pop_back();

        Token& t = tokenPool[idx];
        t.valid_ = true;
        t.state_ = Token::State::AwaitingRootExec;
        t.rootPc_ = pc;
        t.rootExecCycle_ = 0;
        t.refcount_ = 0; // TokenRef ctor below brings this to 1

        ++stats.tokensAssigned;
        return TokenRef(&t);
    }

    void registerDependency(InstrId consumer, InstrId producer) {
        dependents[producer].push_back(consumer);
        pendingCount[consumer] += 1;
    }

    void trainOn(Addr pc, std::int64_t distance) {
        std::size_t idx = indexFor(pc, cfg.trainingTableEntriesLog2);
        TrainEntry& e = trainTable[idx];
        if (!e.valid || e.tag != pc) {
            e.valid = true;
            e.tag = pc;
            e.count = 0;
        }
        if (distance <= cfg.criticalDistanceCycles) {
            e.count = clampAdd(e.count, +1, cfg.trainCounterMin, cfg.trainCounterMax);
        } else if (distance >= cfg.nonCriticalDistanceCycles) {
            e.count = clampAdd(e.count, -1, cfg.trainCounterMin, cfg.trainCounterMax);
        } else {
            return; // ambiguous zone: no update, no promotion check
        }
        ++stats.trainingEvents;

        if (e.count >= cfg.promoteCriticalAt) {
            promote(pc, true);
        } else if (e.count <= cfg.promoteNonCriticalAt) {
            promote(pc, false);
        }
    }

    void promote(Addr pc, bool critical) {
        std::size_t idx = indexFor(pc, cfg.predictionTableEntriesLog2);
        PredEntry& e = predTable[idx];
        e.valid = true;
        e.tag = pc;
        e.critical = critical;
        if (critical) ++stats.criticalPromotions;
        else ++stats.nonCriticalPromotions;
    }

    bool lookupPrediction(Addr pc) const {
        std::size_t idx = indexFor(pc, cfg.predictionTableEntriesLog2);
        const PredEntry& e = predTable[idx];
        if (e.valid && e.tag == pc) return e.critical;
        return false; // unknown PC defaults to non-critical (baseline behavior)
    }

    // Resolve 'id' as a producer. Fan-out fairness: see DESIGN.md S3.2.
    InstrId resolveDependents(InstrId id) {
        InstrId readyConsumer = kInvalidInstrId;
        unsigned readyCount = 0;
        auto depIt = dependents.find(id);
        if (depIt != dependents.end()) {
            for (InstrId c : depIt->second) {
                auto cntIt = pendingCount.find(c);
                if (cntIt == pendingCount.end()) continue; // c was flushed; ignore
                if (--(cntIt->second) <= 0) {
                    pendingCount.erase(cntIt);
                    ++readyCount;
                    if (nextRandom() % readyCount == 0) {
                        readyConsumer = c;
                    }
                }
            }
            dependents.erase(depIt);
        }
        return readyConsumer;
    }

    void onFlush(InstrId id) {
        pendingCount.erase(id);

        // 'id' may be a producer that other, still-in-flight consumers
        // are waiting on; since it will now never execute, erase (not
        // decrement) their pending counts so no future producer mistakes
        // a stale count for a genuine last-arriving-operand event.
        auto depIt = dependents.find(id);
        if (depIt != dependents.end()) {
            for (InstrId c : depIt->second) {
                pendingCount.erase(c);
            }
            dependents.erase(depIt);
        }
        // Note: any token 'id' held is released via its own TokenRef's
        // destruction on the host side, not by this call.
    }

    // --- Token callbacks (see Token::notifyExecuted / Token::release) --
    void UpdateCriticality(Token* token, Addr pc, Cycle now) {
        if (token->state_ == Token::State::AwaitingConsumerExec) {
            std::int64_t distance =
                static_cast<std::int64_t>(now) - static_cast<std::int64_t>(token->rootExecCycle_);
            trainOn(token->rootPc_, distance);
        }
        token->rootPc_ = pc;
        token->rootExecCycle_ = now;
        token->state_ = Token::State::AwaitingConsumerExec;
    }

    void TokenRecycle(Token* token) {
        token->valid_ = false;
        freeList.push_back(token->poolIndex_);
        ++stats.tokensRetired;
    }
};

TokenEngine::TokenEngine(const CriticalityConfig& cfg)
    : impl_(new Impl(cfg, this)) {}

TokenEngine::~TokenEngine() { delete impl_; }

bool TokenEngine::TokenAvailable() {
    return impl_->TokenAvailable();
}

TokenRef TokenEngine::DecodeGetToken(Addr pc) {
    return impl_->DecodeGetToken(pc);
}

bool TokenEngine::isPredictedCritical(Addr pc) const {
    return impl_->lookupPrediction(pc);
}

bool TokenEngine::isPredictedCriticalForEviction(Addr pc) const {
    return impl_->lookupPrediction(pc);
}

void TokenEngine::registerDependency(InstrId consumer, InstrId producer) {
    impl_->registerDependency(consumer, producer);
}

InstrId TokenEngine::resolveDependents(InstrId producerId) {
    return impl_->resolveDependents(producerId);
}

void TokenEngine::onFlush(InstrId id) {
    impl_->onFlush(id);
}

std::int32_t TokenEngine::priorityBoost(bool predictedCritical,
                                         std::int32_t boostMagnitude) {
    return predictedCritical ? boostMagnitude : 0;
}

const CriticalityStats& TokenEngine::stats() const {
    return impl_->stats;
}

void TokenEngine::UpdateCriticality(Token* token, Addr pc, Cycle now) {
    impl_->UpdateCriticality(token, pc, now);
}

void TokenEngine::TokenRecycle(Token* token) {
    impl_->TokenRecycle(token);
}

} // namespace cacrit
