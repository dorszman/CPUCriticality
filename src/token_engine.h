// TokenEngine: the budget-constrained TOKEN SAMPLING layer sitting on
// top of CriticalityPredictor. Owns the token pool and the Token
// lifecycle (epoch/refcount/fan-out racing); delegates all actual
// training/prediction to an internal CriticalityPredictor instance,
// which has no idea tokens exist at all -- see criticality_predictor.h
// and docs/DESIGN.md.
//
// This header is self-contained (C++17, standard library only) and has no
// dependency on any particular simulator's internal types. Notably, it
// has NO dependency-graph bookkeeping of its own: token hand-off rides
// directly on the host's own issue-queue wakeup/broadcast logic (see
// token.h) -- a producer delivers its token straight to whichever
// consumer its broadcast wakes, or drops it if the broadcast wakes
// nobody. This engine never needs to be told about flush/squash events
// either -- a flushed uop's TokenRef releases by ordinary object lifetime
// on the host side.
#pragma once

#include "token.h"
#include "criticality_predictor.h"

#include <cstddef>
#include <cstdint>

namespace ISQCrit {

struct CriticalityConfig {
    // Number of simultaneously in-flight tokens. Small by design: this
    // bounds both the sampling rate and the hardware cost of the token
    // pool. Must be >= 1 -- TokenEngine's constructor throws
    // std::invalid_argument if it is 0.
    unsigned tokenPoolSize = 8;

    // Default sampling policy (matches the original mechanism): a token
    // is assigned on average once every tokenAssignInterval decoded
    // instructions, deliberately undersampling even when the pool has
    // free slots. Must be >= 1 -- it is used as a modulo divisor, and
    // TokenEngine's constructor throws std::invalid_argument if it is 0
    // rather than letting that become undefined behavior at runtime.
    // Ignored when pureAvailabilityGating is true, but still validated
    // regardless (simpler and safer than a conditionally-enforced rule).
    unsigned tokenAssignInterval = 32;

    // When true, DecodeGetToken() ignores tokenAssignInterval and gates
    // purely on pool occupancy: every decoded uop is eligible for a
    // token as long as a slot is free, and scarcity alone throttles the
    // sampling rate. When false (default), DecodeGetToken() requires
    // BOTH a free slot AND the sparse-interval counter to hit, matching
    // the original mechanism's deliberate undersampling. This is a
    // sampling-policy knob, not a correctness one -- either setting
    // produces a valid, bounded-cost predictor; they differ in sampling
    // statistics (see docs/DESIGN.md).
    bool pureAvailabilityGating = false;

    // Everything about HOW a sample gets turned into a prediction --
    // table sizing, eviction policy, distance thresholds, the rate
    // governor -- lives here, in the predictor's own config, completely
    // independent of the token-sampling strategy above. See
    // criticality_predictor.h. A host wanting "ideal mode" (no tokens,
    // every dynamic instruction measured) constructs a bare
    // CriticalityPredictor directly from this same struct instead of
    // going through TokenEngine at all.
    PredictorConfig predictor;
};

// The counters themselves live in the HOST's own stats storage, not
// inside this engine -- it never allocates or owns any counter memory of
// its own. Call TokenEngine::SetupStats() once, typically right after
// construction, with each field pointing at wherever the host already
// keeps that counter (a field in its own stats struct, an element of a
// counter array, whatever it already uses for every other component's
// counters). The engine then writes straight through these pointers from
// then on. Any field left null is simply never incremented -- leave
// unset whichever counters you don't care about.
struct CriticalityStatPointers {
    StatCounter* tokensAssigned = nullptr;
    StatCounter* tokensExpired = nullptr;       // refcount reached zero, slot freed

    // Training/promotion counters -- forwarded straight through to the
    // internal CriticalityPredictor's own SetupStats(). See
    // PredictorStatPointers.
    PredictorStatPointers predictor;
};

class TokenEngine {
public:
    // Throws std::invalid_argument if cfg.tokenPoolSize == 0 or
    // cfg.tokenAssignInterval == 0 -- both are caller-supplied values
    // this engine cannot safely treat as zero (see their comments above),
    // and validating unconditionally is simpler and safer in a release
    // build than relying on assert(), which compiles away under NDEBUG.
    // Also throws for any invalid field in cfg.predictor -- see
    // CriticalityPredictor's own constructor.
    explicit TokenEngine(const CriticalityConfig& cfg = CriticalityConfig{});
    ~TokenEngine();

    TokenEngine(const TokenEngine&) = delete;
    TokenEngine& operator=(const TokenEngine&) = delete;

    // Call once per cycle, before any decode/execute hooks for that cycle.
    // The engine tracks the current cycle internally so it does not need
    // to be passed to every other call (see Token::NotifyExecuted).
    void Tick(Cycle now);

    // --- Decode -----------------------------------------------------------
    // Call once per decoded uop, unconditionally:
    //
    //   uop.token = engine.DecodeGetToken(pc);
    //
    // Always advances the internal sparse-sampling counter first (so it
    // tracks decoded uops, not granted tokens), then either pops a free
    // slot, roots it at 'pc', and returns the refcounted handle to store
    // as a plain member of your own uop/micro-op object -- or, most of
    // the time, returns an empty (invalid) TokenRef and touches nothing
    // else. A grant requires pool occupancy alone if
    // cfg.pureAvailabilityGating is set, or pool occupancy AND the
    // sampling-interval hit otherwise (the default, matching the
    // original mechanism's deliberate undersampling). There is no
    // separate availability pre-check to call first -- counting and
    // gating happen together, in this one call, every time.
    TokenRef DecodeGetToken(Addr pc);

    // Read-only prediction lookup, usable independently of whether this
    // particular dynamic instance was sampled by a token. Call in
    // parallel with your instruction-cache fetch; false (non-critical) is
    // returned for any PC with no trained entry yet. A pure pass-through
    // to the internal CriticalityPredictor.
    bool IsPredictedCritical(Addr pc) const;

    // --- Execute, wakeup, and flush: no engine calls at all --------------
    // There is deliberately no execute/wakeup/flush hook here. On
    // execute, the host calls Token::NotifyExecuted() (see token.h) and
    // acts on its return value together with its OWN issue-queue wakeup
    // result: true plus a woken consumer means hand the TokenRef directly
    // to that consumer (a plain move, not an engine call); true with no
    // wakeup, or false, both mean simply drop the TokenRef. On flush (a
    // uop squashed before executing, or holding a token mid-measurement),
    // the host's own existing object-lifetime management destroys or
    // resets that TokenRef, which is sufficient -- see docs/DESIGN.md
    // S3.2/S3.3/S5/S8 for the exact sequence and why no separate flush
    // entry point is needed.

    // Call once, typically right after construction, pointing each field
    // at counter storage the HOST already owns. See CriticalityStatPointers.
    // Safe to call with any subset of fields left null; safe to never
    // call at all (every counter is then simply never incremented).
    void SetupStats(const CriticalityStatPointers& counters);

private:
    friend class Token;
    // API called by the token itself (Token::DoNotifyExecuted forwards
    // here, only once TokenRef has confirmed its epoch is current), not
    // by the host directly. Returns true if this was the first execute
    // (host should check its own wakeup result and hand the token to
    // whoever got woken, if anyone), false if the measurement just
    // closed -- in which case the slot has ALREADY been recycled
    // (epoch bumped, returned to the pool) before this call returns,
    // regardless of any other TokenRef copies still outstanding. The
    // host should drop it either way.
    bool UpdateCriticality(Token* token);
    // Called once a token is dead by refcount (Token::Release at 0). Not
    // called for the closed-measurement recycle path, which bumps the
    // epoch and frees the slot immediately inside UpdateCriticality
    // instead of waiting for refcount to reach zero.
    void TokenExpire(Token* token);

    struct Impl;
    Impl* impl_;
};

} // namespace ISQCrit
