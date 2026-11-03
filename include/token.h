// Token: a small, pool-allocated, reference-counted handle representing
// one in-flight leg of a criticality-tracking chain (see docs/DESIGN.md).
//
// Ownership model: a host's own uop/micro-op class embeds a TokenRef as a
// plain member. TokenRef is an intrusive-refcount RAII wrapper (copy/move/
// destroy all manage Token::refcount_ automatically) — there is no
// separate "does this id hold a token" lookup anywhere. When the last
// TokenRef referencing a Token is destroyed or reset (including as a side
// effect of a uop being flushed and its storage reset/recycled), the
// token's refcount reaches zero and it calls back into the owning
// CriticalityEngine to return its pool slot — "token release triggers
// engine release" is exactly this call chain, not a manually-triggered
// cleanup step.
#pragma once

#include <cstddef>
#include <cstdint>

namespace cacrit {

using InstrId = std::uint64_t;
using Addr = std::uint64_t;
using Cycle = std::uint64_t;

constexpr InstrId kInvalidInstrId = 0;

class CriticalityEngine;

class Token {
public:
    Token() = default;

    // The criticality-update API: call exactly once, when the uop holding
    // this token executes / broadcasts its result. The token does not
    // compute anything itself — it forwards the event to its owning
    // engine, passing itself as the identity the engine updates state for.
    void notifyExecuted(Addr pc, Cycle now);

private:
    friend class CriticalityEngine;
    friend class TokenRef;

    void addRef() { ++refcount_; }
    void release(); // defined in token.cpp; reclaims via engine_ at refcount 0

    enum class State : std::uint8_t { AwaitingRootExec, AwaitingConsumerExec };

    CriticalityEngine* engine_ = nullptr;
    unsigned refcount_ = 0;
    bool valid_ = false;
    State state_ = State::AwaitingRootExec;
    Addr rootPc_ = 0;        // PC to train when this leg closes
    Cycle rootExecCycle_ = 0; // cycle this leg's producer executed
    std::size_t poolIndex_ = 0;
};

// RAII ref-counted handle. Embed this directly as a plain member of a host
// uop/micro-op class — e.g. `TokenRef token;` — no pointer management,
// explicit cleanup call, or separate id-keyed lookup required. Copying a
// TokenRef (e.g. handing a token from a producer uop to the consumer uop
// it was passed to) increments the refcount; destroying or reset()-ing one
// decrements it.
class TokenRef {
public:
    TokenRef() = default;
    explicit TokenRef(Token* t) : token_(t) { if (token_) token_->addRef(); }

    TokenRef(const TokenRef& other) : token_(other.token_) {
        if (token_) token_->addRef();
    }
    TokenRef(TokenRef&& other) noexcept : token_(other.token_) {
        other.token_ = nullptr;
    }
    TokenRef& operator=(const TokenRef& other) {
        if (this != &other) {
            reset();
            token_ = other.token_;
            if (token_) token_->addRef();
        }
        return *this;
    }
    TokenRef& operator=(TokenRef&& other) noexcept {
        if (this != &other) {
            reset();
            token_ = other.token_;
            other.token_ = nullptr;
        }
        return *this;
    }
    ~TokenRef() { reset(); }

    void reset() {
        if (token_) {
            token_->release();
            token_ = nullptr;
        }
    }

    bool valid() const { return token_ != nullptr; }
    Token* get() const { return token_; }
    Token* operator->() const { return token_; }

private:
    Token* token_ = nullptr;
};

} // namespace cacrit
