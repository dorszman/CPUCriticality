#include "criticality_predictor.h"

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

struct CriticalityPredictor::Impl {
    explicit Impl(const CriticalityConfig& c) : cfg(c) {
        tokenPool.resize(std::max<unsigned>(cfg.tokenPoolSize, 1));
        trainTable.resize(std::size_t{1} << cfg.trainingTableEntriesLog2);
        predTable.resize(std::size_t{1} << cfg.predictionTableEntriesLog2);
        if (cfg.tokenAssignInterval == 0) cfg.tokenAssignInterval = 1;
    }

    // --- Token pool -----------------------------------------------------
    enum class TokState : std::uint8_t { AwaitingRootExec, AwaitingConsumerExec };

    struct TokenEntry {
        bool valid = false;
        TokState state = TokState::AwaitingRootExec;
        InstrId pendingId = kInvalidInstrId; // instruction whose execution closes this leg
        Addr rootPc = 0;                     // PC to train when this leg closes
        Cycle rootExecCycle = 0;              // cycle the current leg's producer executed
    };

    std::vector<TokenEntry> tokenPool;
    // Maps an awaited instruction id -> the token slot waiting on it.
    std::unordered_map<InstrId, std::size_t> holderIndex;

    // --- Dependency graph bookkeeping -----------------------------------
    // producer id -> consumers depending on it (entries still in flight)
    std::unordered_map<InstrId, std::vector<InstrId>> dependents;
    // consumer id -> number of not-yet-ready source operands
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
    CriticalityStats stats;
    std::uint64_t fetchCounter = 0;

    std::size_t indexFor(Addr pc, unsigned log2Entries) const {
        // Drop the low 2 bits assuming 4-byte instruction alignment; this
        // is a default suitable for a functional reference model. A real
        // design should size/hash this to its own fetch-group granularity.
        return static_cast<std::size_t>((pc >> 2) & ((std::uint64_t{1} << log2Entries) - 1));
    }

    // --- Fetch-time sparse token assignment -----------------------------
    bool tryAssignToken(InstrId id, Addr pc) {
        for (std::size_t i = 0; i < tokenPool.size(); ++i) {
            if (!tokenPool[i].valid) {
                TokenEntry& slot = tokenPool[i];
                slot.valid = true;
                slot.state = TokState::AwaitingRootExec;
                slot.pendingId = id;
                slot.rootPc = pc;
                slot.rootExecCycle = 0;
                holderIndex[id] = i;
                ++stats.tokensAssigned;
                return true;
            }
        }
        return false; // pool exhausted this cycle; just skip sampling
    }

    bool onFetch(InstrId id, Addr pc) {
        ++fetchCounter;
        if ((fetchCounter % cfg.tokenAssignInterval) == 0) {
            tryAssignToken(id, pc);
        }
        return lookupPrediction(pc);
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

    void onExecute(InstrId id, Addr pc, Cycle now) {
        bool hasTokenInProgress = false;
        std::size_t slotIdx = 0;

        auto it = holderIndex.find(id);
        if (it != holderIndex.end()) {
            hasTokenInProgress = true;
            slotIdx = it->second;
            holderIndex.erase(it);

            TokenEntry& slot = tokenPool[slotIdx];
            if (slot.state == TokState::AwaitingConsumerExec) {
                std::int64_t distance =
                    static_cast<std::int64_t>(now) - static_cast<std::int64_t>(slot.rootExecCycle);
                trainOn(slot.rootPc, distance);
            }
            // Re-root the leg at 'id': this instruction is now the head of
            // the next edge in the chain, if one is found below.
            slot.rootPc = pc;
            slot.rootExecCycle = now;
            slot.state = TokState::AwaitingConsumerExec;
        }

        // Resolve 'id' as a producer for its registered dependents. This
        // always runs, independent of token state, because the dependency
        // graph must stay correct even for edges no token is riding.
        InstrId readyConsumer = kInvalidInstrId;
        auto depIt = dependents.find(id);
        if (depIt != dependents.end()) {
            for (InstrId c : depIt->second) {
                auto cntIt = pendingCount.find(c);
                if (cntIt == pendingCount.end()) continue; // c was flushed; ignore
                if (--(cntIt->second) <= 0) {
                    pendingCount.erase(cntIt);
                    if (readyConsumer == kInvalidInstrId) readyConsumer = c;
                }
            }
            dependents.erase(depIt);
        }

        if (hasTokenInProgress) {
            TokenEntry& slot = tokenPool[slotIdx];
            if (readyConsumer != kInvalidInstrId) {
                slot.pendingId = readyConsumer;
                holderIndex[readyConsumer] = slotIdx;
            } else {
                slot.valid = false;
                ++stats.tokensRetired;
            }
        }
    }

    void onFlush(InstrId id) {
        pendingCount.erase(id);

        // 'id' may be a producer that other, still-in-flight consumers are
        // waiting on. Since 'id' will now never execute, their pending
        // count can never legitimately reach zero through this path —
        // erase it (rather than decrement it) so no future producer
        // mistakes a stale count for a genuine last-arriving-operand
        // event, and so the map entry does not linger forever. In a real
        // pipeline these consumers are almost always flushed in the same
        // batch anyway (flush squashes a contiguous younger suffix); this
        // just makes correctness independent of flush ordering/batching.
        auto depIt = dependents.find(id);
        if (depIt != dependents.end()) {
            for (InstrId c : depIt->second) {
                pendingCount.erase(c);
            }
            dependents.erase(depIt);
        }

        auto it = holderIndex.find(id);
        if (it != holderIndex.end()) {
            tokenPool[it->second].valid = false;
            holderIndex.erase(it);
            ++stats.tokensRetired;
        }
        // Note: 'id' may still appear in some OTHER producer's dependents
        // list (if id itself was a consumer awaiting resolution elsewhere,
        // and that producer has not executed yet). That is handled lazily
        // and safely in onExecute()'s pendingCount lookup, which skips
        // entries with no pendingCount record — so no cleanup is needed
        // there; this call has already removed 'id' as a pendingCount key.
    }
};

CriticalityPredictor::CriticalityPredictor(const CriticalityConfig& cfg)
    : impl_(new Impl(cfg)) {}

CriticalityPredictor::~CriticalityPredictor() { delete impl_; }

bool CriticalityPredictor::onFetch(InstrId id, Addr pc) {
    return impl_->onFetch(id, pc);
}

void CriticalityPredictor::registerDependency(InstrId consumer, InstrId producer) {
    impl_->registerDependency(consumer, producer);
}

void CriticalityPredictor::onExecute(InstrId id, Addr pc, Cycle now) {
    impl_->onExecute(id, pc, now);
}

void CriticalityPredictor::onFlush(InstrId id) {
    impl_->onFlush(id);
}

bool CriticalityPredictor::isPredictedCritical(Addr pc) const {
    return impl_->lookupPrediction(pc);
}

bool CriticalityPredictor::isPredictedCriticalForEviction(Addr pc) const {
    return impl_->lookupPrediction(pc);
}

std::int32_t CriticalityPredictor::priorityBoost(bool predictedCritical,
                                                  std::int32_t boostMagnitude) {
    return predictedCritical ? boostMagnitude : 0;
}

const CriticalityStats& CriticalityPredictor::stats() const {
    return impl_->stats;
}

} // namespace cacrit
