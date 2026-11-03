// Token: a small, pool-allocated, reference-counted handle representing
// one in-flight criticality measurement (see docs/DESIGN.md).
//
// Ownership model: a token is held by a TokenRef wherever the host happens
// to put it -- embedded in the uop that currently owns it. On the
// producer uop's execute, the host's own issue-queue wakeup/broadcast
// logic determines which consumer(s), if any, were woken by this result.
// If exactly one, the token is handed directly to its TokenRef. If the
// broadcast wakes SEVERAL consumers at once (fan-out), the token is
// handed to ALL of them -- an ordinary TokenRef copy to each, refcount
// rising accordingly, still a single pool slot.
//
// Whichever of those consumers executes FIRST necessarily has the
// shortest distance to the producer (distance is monotonic in execute
// cycle from a fixed root), so it is automatically the strongest
// critical-path evidence with no comparison needed. The moment that
// winning consumer closes the measurement, the slot is recycled
// IMMEDIATELY -- bumped to a new epoch and returned to the pool for a
// fresh, unrelated random sample -- rather than waiting for every slower
// sibling's TokenRef to drop naturally. A slower sibling's TokenRef is
// therefore not a reference to "the same measurement" anymore once that
// happens; it is a stale reference to a pool slot that may already belong
// to a completely different, later instruction. Every TokenRef operation
// (AddRef on copy, Release on drop, the criticality-update call) checks
// its own captured epoch against the token's current epoch first, and is
// a silent no-op if they don't match -- this is what makes "first to
// execute wins, and the others are inert when they later execute" safe
// even though the underlying slot has already moved on to a new owner by
// then, instead of relying on every straggler eventually dropping its
// reference before the slot can be reused.
//
// TokenRef is an intrusive-refcount RAII wrapper (copy/move/destroy all
// manage Token::refcount_ automatically, gated by the epoch check above).
// Because every holder is an ordinary host-owned object with its own
// normal lifetime (a uop being flushed, or retired normally), releasing a
// token is never a separate step this module must be told to perform --
// it falls out of whichever host-side container's destructor or Reset()
// already runs for unrelated reasons.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

namespace cacrit {

using InstrId = std::uint64_t;
using Addr = std::uint64_t;
using Cycle = std::uint64_t;
using Epoch = std::uint64_t;

constexpr InstrId kInvalidInstrId = 0;

class TokenEngine;
class TokenRef;

class Token {
private:
    friend class TokenEngine;
    friend class TokenRef;

    void AddRef() { ++refcount_; }
    void Release(); // defined in token.cpp; recycles via engine_ at refcount 0

    // The criticality-update call. Only ever invoked by TokenRef, and
    // only after it has confirmed its own captured epoch is still
    // current -- see TokenRef::NotifyExecuted(). No PC or cycle
    // parameter: the PC is fixed to whatever instruction this token was
    // originally assigned to at creation and never changes, and the
    // current cycle is tracked by the owning engine (see
    // TokenEngine::Tick), not threaded through every call.
    //
    // Returns true if this was the token's first execute (its producer's
    // execute cycle was just recorded; nothing trained yet) -- the host
    // should then consult its own issue-queue wakeup logic: hand this
    // token's TokenRef to every consumer the broadcast woke (a copy each,
    // if more than one), or drop it if it woke nobody.
    //
    // Returns false once the measurement closes (trains the original PC
    // against the distance to this execute) -- at which point the slot is
    // immediately recycled (epoch bumped) regardless of any other
    // TokenRef copies still outstanding. The host should simply drop the
    // token either way; it is never told which outcome happened, because
    // it never needs to know.
    bool DoNotifyExecuted();

    TokenEngine* engine_ = nullptr;
    unsigned refcount_ = 0;
    bool valid_ = false;
    Epoch epoch_ = 0; // bumped every time this slot returns to the pool
    Addr pc_ = 0; // fixed at creation; trained when the measurement closes
    // Cycle the producer (pc_) executed, i.e. its "production time" --
    // empty until the first successful call records it. Present once set
    // means a subsequent call is an attempt to close the measurement.
    std::optional<Cycle> rootExecCycle_;
    std::size_t poolIndex_ = 0;
};

// RAII ref-counted, epoch-guarded handle. Embed this directly as a plain
// member of a host uop/micro-op class — e.g. `TokenRef token;` — no
// pointer management, explicit cleanup call, or separate id-keyed lookup
// required. Every operation checks the epoch captured at construction/copy
// time against the token's current epoch, and is a no-op if they've
// diverged (the slot was recycled for a different instruction in the
// meantime) — this is what makes it safe for a slot to be reused before
// every outstanding TokenRef to its previous occupant has been dropped.
class TokenRef {
public:
    TokenRef() = default;
    explicit TokenRef(Token* t) : token_(t), epoch_(t ? t->epoch_ : 0) {
        if (token_) token_->AddRef();
    }

    TokenRef(const TokenRef& other) : token_(other.token_), epoch_(other.epoch_) {
        if (IsCurrent()) token_->AddRef();
    }
    TokenRef(TokenRef&& other) noexcept : token_(other.token_), epoch_(other.epoch_) {
        other.token_ = nullptr;
    }
    TokenRef& operator=(const TokenRef& other) {
        if (this != &other) {
            Reset();
            token_ = other.token_;
            epoch_ = other.epoch_;
            if (IsCurrent()) token_->AddRef();
        }
        return *this;
    }
    TokenRef& operator=(TokenRef&& other) noexcept {
        if (this != &other) {
            Reset();
            token_ = other.token_;
            epoch_ = other.epoch_;
            other.token_ = nullptr;
        }
        return *this;
    }
    ~TokenRef() { Reset(); }

    void Reset() {
        if (IsCurrent()) {
            token_->Release();
        }
        token_ = nullptr;
    }

    // The criticality-update API: call exactly once, when the uop holding
    // this TokenRef executes. A no-op (returns false, touches nothing)
    // if this reference's epoch is stale -- the slot it used to point at
    // has since been recycled for a different instruction. See the class
    // comment and Token::DoNotifyExecuted() for the full contract.
    bool NotifyExecuted() {
        if (!IsCurrent()) return false;
        return token_->DoNotifyExecuted();
    }

    bool Valid() const { return token_ != nullptr; }
    Token* Get() const { return token_; }

private:
    bool IsCurrent() const { return token_ && token_->epoch_ == epoch_; }

    Token* token_ = nullptr;
    Epoch epoch_ = 0;
};

} // namespace cacrit
