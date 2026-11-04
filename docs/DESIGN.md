# Token-Passing Critical-Instruction Prediction for Issue-Queue Scheduling

**Audience:** this document is written to brief an engineer or AI coding agent
well enough to implement the mechanism below inside a cycle-accurate (CA)
out-of-order CPU performance model, and to extend/retarget the accompanying
reference implementation (`src/token_engine.h`,
`src/token.h`, `src/token_engine.cpp`, `src/token.cpp`) onto a
specific host simulator.

**Status:** this is an architectural technique description (not RTL), meant
for a performance/timing simulator. It describes an *algorithm*, not a
specific company's product. Treat it as you would any academic
critical-path-prediction scheme in the OoO-microarchitecture literature
(token-passing / last-arriving-operand dependency tracking is a well known
family of techniques for approximating a program's dynamic critical path in
hardware at low cost).

---

## 1. Problem statement

In an out-of-order core, not every in-flight instruction is equally
important to overall performance:

- An instruction on the **critical path** (the chain of true data
  dependencies that bounds how fast the program can finish) directly limits
  IPC if delayed.
- An instruction *off* the critical path can be delayed by several cycles
  with no measurable effect on program completion time, because something
  else is already gating forward progress.

A plain age-based issue-queue picker ("oldest ready wins") does not know
this distinction. It can issue a non-critical instruction ahead of a
critical one purely because it is older, needlessly delaying the real
bottleneck.

The challenge: criticality is **dynamic** (the same static instruction may
be critical in one dynamic instance and not in another) and **not known
until execution**. Tracking the true dependency chain of every in-flight
instruction exactly would require per-instruction bookkeeping that is too
expensive in area and power for a real issue queue. The technique below
produces a cheap, *statistical*, self-training predictor instead, with a
structure directly analogous to a branch predictor (PC-indexed, trained
online, consumed at fetch).

---

## 2. Terminology

| Term | Meaning |
|---|---|
| Producer | An instruction that generates an operand consumed by a later instruction |
| Consumer | An instruction waiting on one or more producer operands |
| Last-arriving operand | Among a consumer's source operands, the one that becomes ready *last* — i.e. the producer that actually determines the consumer's wakeup cycle |
| Token | A small, scarce, pool-allocated, reference-counted object that rides along one chain of last-arriving-operand edges, carrying just enough state to measure one edge's latency at a time |
| TokenRef | A plain RAII handle to a `Token`, embedded directly as a member of the host's own uop/micro-op object — this, not any id-keyed lookup, is what makes a uop "hold" a token |
| Distance | Cycles between a producer's execution and the execution of the consumer for which it supplied the last-arriving operand |
| Engine | The `TokenEngine`: owns the token pool, the dependency-graph bookkeeping, and the training/prediction tables. A `Token` never computes anything itself — it forwards events to its owning engine |
| Training table | PC-indexed table of saturating counters, incremented/decremented by observed distances |
| Prediction table | PC-indexed 1-bit table (critical / non-critical), promoted from the training table once a counter saturates |

---

## 3. Mechanism

### 3.1 Sparse token assignment (decode stage)

- Maintain a small fixed pool of tokens (reference default: 8 tokens, i.e.
  a 3-bit token ID — deliberately small to bound area/power).
- As instructions are decoded, assign a free token to an instruction only
  occasionally: call `DecodeGetToken(pc)` once per decoded uop,
  unconditionally — it always advances an internal sampling counter
  first, then either pops a free slot and roots it at `pc`, or (most of
  the time) returns an empty `TokenRef` and touches nothing else. There
  is no separate availability pre-check to call first; counting and
  gating happen together in this one call, every time — splitting them
  across two calls would let the counter stall the moment a decode is
  gated out, since nothing would be left to advance it. By default a
  grant fires on average 1 in N decoded uops (`tokenAssignInterval`),
  undersampling even when the pool has free slots, matching the original
  mechanism's deliberate sparsity; setting `pureAvailabilityGating`
  instead gates purely on pool occupancy (every decoded uop is eligible
  as long as a slot is free, letting scarcity alone throttle sampling —
  see the config comments for the trade-off). **Do not tag every
  instruction.** Sparse sampling is what keeps this cheap; the predictor
  is trained from a statistical sample of the dynamic instruction stream,
  not from exhaustive tracking.
- A freshly tagged instruction's PC is **fixed for the token's entire
  lifetime** — it is never reassigned. This single instruction's PC is
  the only thing ever trained by this token (see §3.3): there is no
  predecessor yet, so no training happens until the token's *second*
  execute event.

### 3.2 Hand-off via the host's own issue-queue wakeup logic (no Ptag table, no shadow dependency graph)

There is **no register-scoreboard/Ptag table and no producer→consumer
dependency graph maintained inside the engine** — two different
approaches were tried while building this design and both were dropped in
favor of the current one, for the same underlying reason: real OoO
hardware already knows exactly this information via its own issue-queue
wakeup/broadcast logic (which instruction produces which register, and
which reservation-station entries are waiting to read it), so duplicating
any part of that as a parallel structure inside this module is redundant,
costly, and a source of its own bugs. Token hand-off rides directly on
that existing wakeup logic instead:

- When the uop holding a token executes, its result broadcast wakes
  whichever reservation-station entries were waiting on it — this is the
  host's own ordinary wakeup/match logic, unmodified. The host checks the
  outcome of that broadcast at the same point it calls `notifyExecuted()`
  (§3.3).
- **If the broadcast woke exactly one consumer**, the token is handed off
  directly to that consumer's `TokenRef` — a plain host-side move, not a
  call into this module.
- **If the broadcast woke nobody** — no instruction currently in the issue
  queue was waiting on this result — the token simply does not propagate.
  It is dropped immediately, on the spot: nothing is gated by this
  producer right now, so there is nothing further to measure. This is a
  deliberate, explicit rule, not an inferred default: a producer whose
  result nobody was waiting for never gets a chance to train anything,
  and stays at the engine's baseline non-critical prediction for any PC
  with no trained entry (§3.5).
- **If several entries are woken simultaneously** by one broadcast (a
  shared condition code, a cache-line fill waking multiple loads), the
  token is handed to **all** of them — an ordinary `TokenRef` copy to
  each, refcount rising accordingly, still a single pool slot — and they
  race (§3.3). This replaced an earlier version of this design where the
  host had to pick one woken consumer itself (with reservoir sampling
  recommended for fairness, since picking the same sibling every dynamic
  instance would starve the others). That host-side arbitration is no
  longer necessary: racing resolves fan-out *correctly*, not just
  *fairly* — see §3.3.

### 3.3 Single-hop distance measurement, and resolving fan-out by racing (execute)

A token measures **exactly one** producer→consumer edge, then its job is
done — this matches the original mechanism's token lifecycle exactly (a
token trains the one PC it was originally assigned to, once, then is
flushed and made available for a fresh, unrelated reassignment — confirmed
directly against the primary patent specification text, not inferred).
The engine does not track *which* instruction currently holds a token by
any id; the token itself is a pool-allocated, reference-counted object,
and whichever uop currently "holds" it does so simply by owning a
`TokenRef` — an ordinary RAII handle.

Call `token.NotifyExecuted()` (on the `TokenRef`, not the token directly —
there is no public way to reach `Token` except through it) exactly once,
whenever the uop currently holding that `TokenRef` executes. It takes no
PC or cycle argument: the PC is fixed at token creation (§3.1) and never
changes, and the current cycle is tracked internally by the engine (see
`TokenEngine::Tick`, §5) rather than threaded through every call. Note
this also answers a question worth stating explicitly: the patent's own
distance definition is execution-to-execution, not wakeup-to-execution —
there is no need to separately record when a consumer became *ready*, only
when it actually *executed*. (A wakeup-to-retirement variant was discussed
as a possible improvement in §9 below; that is this project's own
extension, not something the original mechanism does.)

`NotifyExecuted()` returns a `bool`:

- **`true`** — this call found the token in its `PendingProducerExe`
  state: the producer's write-back cycle was just recorded into
  `producerWbTime_`, and the token advances to `PendingConsumerExe`.
  Nothing is trained yet. The host should consult its own wakeup result
  from this broadcast (§3.2): hand a copy of the token to every consumer
  that got woken (one copy if one woken, none if none), or drop it if
  nobody did.
- **`false`** — either this call closed the measurement, or it was a stale
  no-op (see below); the host cannot tell which, and does not need to. The
  host should simply **drop** the token now (`Reset()` its `TokenRef`)
  regardless — there is no third leg in this design.

**Resolving fan-out by racing, not arbitrating.** If the producer's
broadcast woke several consumers, they each hold their own copy of the
same token (§3.2) and all eventually call `NotifyExecuted()`. Whichever of
them executes *first* necessarily has the shortest distance back to the
root — distance is monotonic in execute cycle from a fixed root — so it is
automatically the strongest critical-path evidence, with **no comparison
across siblings needed**: the first call to reach the closing branch
trains the token's fixed PC against that shortest distance, **and
immediately recycles the slot** (bumps its epoch, returns it to the free
list) rather than waiting for the slower siblings to drop their copies
naturally. "First to execute wins" falls directly out of the ordering of
calls, not out of any explicit winner-selection step.

**Why this needs an epoch, not just a "closed" flag.** Because the slot is
reused *immediately* on close — not once every outstanding `TokenRef` has
been dropped — a slower sibling's copy is now a reference to a pool slot
that may, by the time it finally executes, belong to a completely
different, unrelated instruction's fresh sample. A simple "already closed"
boolean on the token can't distinguish "you lost the race for *this*
measurement" from "this slot has moved on to a *different* measurement
entirely" — both look identical to a flag that's just sitting there
`true`. Each `Token` instead carries an `epoch_` counter, bumped exactly
once every time its slot returns to the pool for any reason (closed
measurement, refcount reaching zero with no wakeup, a flush). Each
`TokenRef` captures the epoch current *at the time it was created or
copied*. Every operation a `TokenRef` performs — `AddRef` on copy,
`Release` on drop, `NotifyExecuted` — first checks whether its captured
epoch still matches the token's current one, and is a complete no-op if
not: it doesn't train, doesn't touch the refcount, doesn't do anything.
This is what makes immediate reuse safe: a stale sibling calling in later
can never corrupt whatever the slot has since been reassigned to measure,
and never double-decrements a refcount that's already been reset for a
new owner.

Token expiry/recycling is never a separate step the host must remember to
call: it falls out of ordinary reference-counted, epoch-guarded object
lifetime. Whether a slot was recycled because a measurement closed
(immediately, inside that call, independent of stragglers), because a
broadcast woke nobody, or because the uop holding it was flushed before
executing, the mechanism is the same underlying helper
(`TokenEngine::Impl::RecycleSlot`) reached either via
`Token::Release()` → `TokenEngine::TokenExpire` (refcount hit zero) or
directly from `TokenEngine::UpdateCriticality` (measurement closed) — see
§8.

### 3.4 Training table (PC-indexed, saturating counter, set-associative)

- Indexed by the *producer's* PC (same indexing scheme as an instruction
  cache lookup, so it can be probed in parallel with fetch with no added
  latency on the critical fetch path). `trainingTableWays` (default 1,
  i.e. direct-mapped) controls how many candidate entries share each
  set: with `ways = 1`, two PCs that hash to the same set unconditionally
  evict each other; `ways > 1` lets up to `ways` such PCs coexist,
  searching every way in the set on each access.
- When `ways > 1` and a set is full, which entry is evicted is governed
  by `trainingTableEvictionPolicy` — a deliberate either/or choice
  between two complete, non-overlapping policies (not a numeric
  threshold to calibrate):
  - `kLru` (the default): plain least-recently-used. Whichever entry in
    the set was touched longest ago is evicted, full stop, ignoring how
    much training evidence it has accumulated.
  - `kDecayedHitCount`: evict by lowest decayed hit-count instead, with
    **no** recency tiebreak at all. Each `TrainEntry` carries its own
    `hitCount` — bumped every time it is re-trained (independent of
    `count`, which tracks *direction*, critical vs. non-critical, and
    saturates quickly; `hitCount` tracks sheer *frequency* and does
    not). This protects a PC that's been trained many times from being
    evicted by a PC that merely happens to have been touched more
    recently. The protection isn't permanent, though: `hitCount` is
    discounted by how idle the entry's own *set* has been since it was
    last touched (see the decay formula under §3.5 — the exact same
    mechanism, just also offered here, governed by its own
    `trainingTableDecayIdleUnit` knob, independent of the prediction
    table's).
- On each distance measurement:
  - `distance <= criticalDistanceCycles` → increment the counter (evidence
    toward "critical").
  - `distance >= nonCriticalDistanceCycles` → decrement the counter
    (evidence toward "non-critical").
  - Otherwise (ambiguous middle zone) → leave the counter unchanged.
- The counter saturates within a small signed range (reference default:
  `[-4, +3]`). This hysteresis is deliberate and important: a single sample
  must not flip a prediction. The predictor needs *repeated* evidence
  across many dynamic instances of the same static instruction before
  committing — exactly like a 2-bit branch-history counter.

### 3.5 Prediction table (PC-indexed, 1 bit, set-associative)

- When a training counter crosses a promotion threshold in either
  direction, write a 1-bit critical/non-critical verdict into the
  prediction table, indexed the same way. `predictionTableWays` (default
  1) controls its associativity independently of the training table's.
- Same kind of choice as the training table, via its own
  `predictionTableEvictionPolicy` (only matters when
  `predictionTableWays > 1`), defaulting to `kDecayedPriorityThenLru`
  rather than `kLru` — this table is consulted on every single fetch via
  `IsPredictedCritical()`, so the validated starting point here leans on
  that signal instead of plain recency. Each `PredEntry` carries a
  `hitCount`, bumped every time it's actually consulted (the genuine
  "usefulness" signal here, since there is no per-entry confidence
  counter the way the training table has). Under the default policy, a
  miss that needs to allocate within a full set evicts the lowest
  **decayed** `hitCount` first, with recency only as the tiebreak among
  equal decayed values (`kLru` and `kDecayedHitCount` behave exactly as
  described in §3.4, just against `hitCount` instead of the training
  table's own counter). The decay: an entry's raw
  `hitCount` is right-shifted once for each doubling of
  `idle / predictionTableDecayIdleUnit` (integer division — below
  `predictionTableDecayIdleUnit` idle accesses, no decay at all), where
  `idle` is how many accesses *that entry's own set* has seen since it
  was last touched — not a table-wide counter, which would make the
  decay rate secretly depend on overall table traffic, and
  `predictionTableDecayIdleUnit` (`CriticalityConfig`, default `128`,
  must be >= 1) is the one rate knob this table's decay is built around:
  it sets how many idle accesses correspond to one halving. The training
  table has its own separate `trainingTableDecayIdleUnit` for the same
  formula, independently tunable since the two tables typically see very
  different access patterns. Measuring idle time per-set keeps the
  *onset* self-scaling with actual contention for that set even though
  the rate itself is configurable: a heavily-consulted-but-now-stale
  entry's advantage still erodes on its own as its set sees more
  activity, rather than earning permanent protection — the knob only
  tunes how fast that erosion happens.
- This table is what the front end actually consults: read in parallel with
  the instruction cache on every fetch, so by the time an instruction
  reaches the issue queue it already carries a predicted-critical tag with
  no extra lookup latency anywhere downstream.
- **Unknown PCs default to non-critical** (i.e. fall back to the baseline
  age-based policy). This is a deliberate safety property: a misprediction
  in this direction costs a missed optimization opportunity, not a
  regression below the baseline scheduler.

### 3.6 Issue-queue priority integration

Two common reservation-station styles, both achievable without a new
priority network:

**Shifting / compacting reservation stations** (entries physically
reordered toward one end as they issue): time-multiplex the issue picker —
in a cycle where at least one predicted-critical entry is ready, mask out
all predicted-non-critical entries and issue the oldest *ready and critical*
entry; only fall back to plain oldest-ready-first when no critical entry is
ready.

**Non-shifting / matrix-based reservation stations** (age tracked via an
age matrix, entries issue in place): rather than adding a second priority
dimension to the matrix, bias the *existing* fields —
  - Record a predicted-critical operation as artificially older than any
    predicted-non-critical operation in the age matrix (i.e. it always wins
    age-based arbitration against non-critical competitors), and/or
  - Suppress the ready bit of a non-critical entry in cycles where a
    critical entry is also contending, so the matrix's own existing
    select logic naturally favors the critical entry without new
    comparator hardware.

The reference C++ module intentionally does not implement either reservation
station style itself — reservation-station design is host-specific. It
exposes `isPredictedCritical(pc)` and a `priorityBoost()` helper; the host
model's own picker consumes that boolean/boost however its existing select
logic is structured.

### 3.7 Optional extension: cache-retention hint

The same prediction bit can also bias replacement-policy decisions
(not implemented in the reference module, described here for completeness
since it was part of the original idea): on a cache miss caused by a
*non-critical* load, mask cache lines belonging to predicted-critical
instructions from eviction, so non-critical traffic cannot evict data the
predicted-critical path depends on. Critical-load misses use the ordinary
replacement policy unmodified. This only needs the cache controller
querying `IsPredictedCritical(pc)` same as the issue-queue scheduler does;
the training/prediction tables are unchanged and shared — no separate
eviction-specific query is needed.

### 3.8 Optional extension: criticality-rate governor

If too large a fraction of dynamic uops get tagged predicted-critical,
the whole mechanism degrades toward a no-op (prioritizing "everyone" is
the same as prioritizing no one). `enableCriticalRateGovernor` caps —
never targets — the fraction of `IsPredictedCritical()` calls that
return true, against `targetCriticalRate` (e.g. `0.10`).

It is a **one-sided** cap: `promoteCriticalAt` becomes a floor rather
than a fixed value. Over a window of `criticalRateControlInterval`
lookups, if the observed rate exceeds `targetCriticalRate +
criticalRateDeadband`, the effective promotion threshold tightens by
one step (clamped at `trainCounterMax`); if a window is comfortably
under the cap, it relaxes one step back toward — never past —
`promoteCriticalAt` itself. Being under the cap is always fine; the
governor only ever adds strictness on top of the host's own configured
floor, never removes strictness the host asked for. `promoteNonCriticalAt`
(demotion) is never touched.

**Asymmetric response is required, not optional.** Tightening is
immediate (a single over-cap window), but relaxing requires
`criticalRateRelaxDwellWindows` *consecutive* under-cap windows first.
Without this, a population that crosses the threshold in one hard step
(many PCs promoting in lockstep) makes the governor oscillate cleanly
between "admit everyone" and "admit no one" — every individual decision
looks correct, but the time-averaged rate lands far above the cap
despite that.

**Headroom between `promoteCriticalAt` and `trainCounterMax` matters
more than the dwell-time tuning, empirically.** Each tightening step
can only move to the next integer the training counter can actually
reach, so with only 1-2 such steps available the governor still
overshoots the cap by 5-7x on realistic workloads no matter how dwell
is tuned; with 5-6+ steps of headroom it converges cleanly near the
cap. This project's own baseline defaults (`promoteCriticalAt=2`,
`trainCounterMax=3`) are exactly the bad case — raise `trainCounterMax`
well above `promoteCriticalAt` specifically when enabling this
governor.

---

## 4. Data structures (reference sizes — tune for your target)

| Structure | Fields | Default size |
|---|---|---|
| `Token` (pool entry) | `engine_` (owning engine pointer), `refcount_`, `epoch_` (bumped every time the slot returns to the pool, for any reason), `pc_` (fixed at creation), `state_` (`Invalid` / `PendingProducerExe` / `PendingConsumerExe`), `producerWbTime_` (valid once `state_ == PendingConsumerExe`), `poolIndex_` | 8 entries (3-bit token id) |
| `TokenRef` | a single `Token*`; held as a plain member of the host's uop class — never tracked by the engine itself | one per uop, usually null |
| Training table entry | `valid`, `tag (PC)`, `count` (signed, saturating), `hitCount` (unsigned, only meaningful under `kDecayedHitCount`) | 1024 sets × `trainingTableWays` (default 1) |
| Prediction table entry | `valid`, `tag (PC)`, `critical` (1 bit), `hitCount` (unsigned) | 1024 sets × `predictionTableWays` (default 1) |

Both tables are `SetAssociativeTable<EntryT>` (see §3.4/§3.5): PC-indexed
into a set, then every way within that set searched on access. Eviction
policy is an explicit, independently host-configurable `EvictionPolicy`
choice per table (`kLru` / `kDecayedHitCount` / `kDecayedPriorityThenLru`)
rather than one fixed mechanism — `trainingTableEvictionPolicy` (default
`kLru`) for the training table, `predictionTableEvictionPolicy` (default
`kDecayedPriorityThenLru`) for the prediction table. The default `ways = 1` for both
reproduces a plain direct-mapped table exactly regardless of policy (no
search, no eviction decision ever needed — the original design).
PC-truncated tags are the cheap realistic choice for a real
implementation; this reference implementation stores full PCs for
simplicity/correctness in a functional model and notes where a real
design would truncate tag bits to save area. There is no register
scoreboard table and no dependency bookkeeping structure at all (see
§3.2) — hand-off rides directly on the host's own issue-queue wakeup
logic instead of any shadow structure this module would otherwise have to
maintain.

---

## 5. Required integration hooks from the host CA model

The engine is deliberately decoupled from any specific simulator. The
host model's own uop/micro-op class must embed one plain member:

```cpp
#include "token.h"
struct Uop {
    /* ... host's own fields ... */
    ISQCrit::TokenRef token; // empty (invalid) on almost every uop
};
```

and must call the engine at exactly these points, which correspond to
pipeline events essentially every OoO performance model already has:

| Pipeline event (host side) | Call into the engine |
|---|---|
| Once per cycle, before any decode/execute hooks that cycle | `engine.Tick(now);` — the engine tracks the current cycle internally so it is never threaded through other calls |
| Instruction decoded | `uop.token = engine.DecodeGetToken(pc);` (unconditional — usually returns empty), and, separately, `bool crit = engine.IsPredictedCritical(pc);` to tag the instruction for scheduling use |
| Instruction executes / broadcasts result | see the sequence below — exactly once per instruction, alongside whatever the host's own wakeup/match logic already does on this broadcast |
| Instruction squashed (branch mispredict, exception, etc.) | **no engine call at all** — just let the uop's `TokenRef` member be destroyed/reset as part of normal teardown; that alone releases any token it held (§8) |
| Issue-queue select logic | `engine.IsPredictedCritical(pc)` (or the tag captured at fetch); combine with `TokenEngine::PriorityBoost(...)` per §3.6 |

The execute-time sequence, per §3.2/§3.3:

```cpp
// 'woken' is whatever the host's OWN issue-queue wakeup/match logic
// already determines for this broadcast -- the ids of every consumer it
// woke (zero, one, or several). This module has no part in computing it.
if (uop.token.Valid()) {
    bool firstExecute = uop.token.NotifyExecuted(); // on the TokenRef, no pc/cycle args
    if (firstExecute && !woken.empty()) {
        // First execute: hand every woken consumer its OWN copy. If
        // several, they race -- the fastest one to execute wins (§3.3).
        for (auto id : woken) {
            consumerUop(id).token = uop.token; // copy, not move
        }
    }
    // Either way the producer's own reference is done with it now: if
    // copies were handed out this just drops the producer's share
    // (siblings keep theirs); if nothing was woken, or this call itself
    // closed the measurement, this is the token's last live reference.
    uop.token.Reset();
}
```

No other coupling is required. The engine never needs to know about
physical register files, ROB indices, issue-queue internals, or the
host's own instruction representation beyond opaque `Addr`/`Cycle`
integers and the `TokenRef` member on the host's own uop type.

**Stats integration.** The engine owns no counter storage of its own —
counters live wherever the host already keeps them (a field in its own
stats struct, an element of a counter array, whatever every other
component's counters already go through). Call `engine.SetupStats(...)`
once, typically right after construction, with a `CriticalityStatPointers`
pointing each field at the host's own storage; the engine writes straight
through those pointers from then on, so the host reads updated values
directly, on its own schedule, with no call back into the engine ever
needed. Any field left null is simply never incremented — set up only the
counters you care about. Never calling `SetupStats()` at all is safe too;
every counter is then a no-op write-sink.

---

## 6. Configuration knobs

All exposed via `CriticalityConfig` (see header). Starting points to tune
against your workload mix:

- `tokenPoolSize`: trade-off between statistical sample rate and area/power.
- `tokenAssignInterval`: sparsity of sampling; smaller = more samples, more
  power, faster convergence; larger = cheaper, slower/noisier convergence.
  Ignored when `pureAvailabilityGating` is set.
- `pureAvailabilityGating`: `false` (default) samples on the interval above
  regardless of pool occupancy; `true` instead grants a token to every
  decoded uop as long as a slot is free, letting pool scarcity alone
  throttle the rate (see §3.1).
- `criticalDistanceCycles` / `nonCriticalDistanceCycles`: should bracket the
  expected "back-to-back, no slack" latency for your pipeline's bypass
  network (e.g. near 1–2 cycles for ALU-to-ALU forwarding) versus a distance
  that clearly indicates slack existed.
- `trainCounterMin/Max`, `promoteCriticalAt/NonCriticalAt`: hysteresis depth
  before a prediction flips; deeper hysteresis = more stable but slower to
  adapt to phase changes.
- `trainingTableWays` / `predictionTableWays`: associativity of each table
  (default 1 = direct-mapped, matching the original design). Raise either
  if your workload has PCs that collide at the same set and alias each
  other's trained state away — the cost is a linear search across `ways`
  entries on every table access, and `ways` recency/priority comparisons
  on an eviction.
- `trainingTableEvictionPolicy` / `predictionTableEvictionPolicy`:
  independent per-table choice of `kLru`, `kDecayedHitCount`, or
  `kDecayedPriorityThenLru` (see §3.4/§3.5). Training table defaults to
  `kLru` (simplest, nothing to calibrate); prediction table defaults to
  `kDecayedPriorityThenLru` (the validated starting point, since it's
  consulted on every fetch). The decayed options cost a per-entry
  counter and the decay formula below. Only matters when the
  corresponding table's `...Ways > 1`.
- `trainingTableDecayIdleUnit` / `predictionTableDecayIdleUnit` (default
  `128` each, independent of each other): how many idle accesses to an
  entry's own set correspond to one halving of its decayed eviction
  priority, for whichever decayed policy each table is actually using.
  Smaller = faster-decaying, more volatile protection; larger = slower
  decay, closer to no staleness consideration at all. Each one only
  matters when its own table's `...EvictionPolicy` is `kDecayedHitCount`
  or `kDecayedPriorityThenLru`.
- `enableCriticalRateGovernor` / `targetCriticalRate` /
  `criticalRateDeadband` / `criticalRateControlInterval` /
  `criticalRateRelaxDwellWindows`: see §3.8. Disabled by default. If
  enabled, make sure `trainCounterMax` has real headroom above
  `promoteCriticalAt` (5-6+ steps, empirically) — see the warning in
  §3.8; this project's own baseline defaults do not have enough.

---

## 7. Validation plan

1. **Unit-level correctness.** Build a synthetic producer/consumer edge
   (decode a producer, decode a consumer that registers interest in its
   result in your toy wakeup map, two `Tick` + `NotifyExecuted()` calls at
   known cycles, with the second call's return value used to hand the
   token to the woken consumer) and assert the predictor trains the
   expected PC critical/non-critical given deliberately short/long
   synthetic distances.
2. **Expiry / token reuse.** Verify a token is expired (reclaimed to the
   pool, via a dropped `TokenRef`) whenever its producer's broadcast wakes
   nobody, and that the token pool never exceeds its configured size (no
   leaks across any of: closed measurement, dropped on an unwoken
   broadcast, flushed uop).
3. **Flush correctness.** Destroy a uop's `TokenRef` mid-measurement
   (simulating a squash) and verify the token expires cleanly with no
   crash and no stale training from an instruction that never actually
   executed in the committed path — there is no explicit engine call to
   make for this; it is pure RAII (§8).
4. **Full-model integration.** Wire the hooks into your CA model's existing
   fetch/rename/execute/flush events (per §5) and measure, relative to a
   pure age-based baseline:
   - IPC delta across a representative workload suite.
   - Issue-queue occupancy / average wait time, split by predicted-critical
     vs. predicted-non-critical instructions, to confirm the mechanism is
     actually shifting latency away from the predicted-critical population.
   - Prediction accuracy proxy: correlate predicted-critical instructions
     against an oracle/offline critical-path analysis on a small trace, if
     available, to sanity-check the predictor isn't just noise.
5. **Sensitivity sweep.** Sweep `tokenAssignInterval` and the distance
   thresholds to find the area/power-vs-accuracy knee for your target
   design point.

---

## 8. Flush handling and token lifetime

Tokens use a real reference count (`Token::refcount_`) *and* an epoch
(`Token::epoch_`, §3.3), and together they make flush handling robust to
arbitrary call ordering — including a slot being recycled while a stale
reference to it still technically exists — with **zero engine calls
involved**: there is no `onFlush` entry point at all, because there is
nothing left for it to do.

A token may be referenced by one `TokenRef` (the common case) or by
several at once (a fan-out race in progress, §3.2/§3.3) — each living in
whichever uop currently holds it. Every uop is an ordinary object with its
own normal lifetime that the host already manages for unrelated reasons —
flushed on a branch misprediction, retired normally, whatever the host
already does. When a `TokenRef` is destroyed or `Reset()` — including as a
side effect of any of those ordinary teardown paths — it first checks
whether its captured epoch still matches the token's current one
(§3.3); if not, it does nothing at all (the slot has already moved on to
a different instruction). If current, `Token::Release()` decrements the
refcount, and at zero calls back into the engine
(`engine_->TokenExpire(this)` → the same `RecycleSlot` helper used when a
measurement closes, §3.3) to return the pool slot and bump its epoch.

This means token lifetime is simply correct, epoch-guarded C++ object
lifetime, full stop. There is no "does this flushed uop hold a token, and
if so tell the engine" step to remember anywhere — not on squash, not
when a broadcast wakes nobody and the host drops the token on purpose, not
when a measurement closes normally and the host calls `.Reset()` on
purpose (§3.3), not when a slower fan-out sibling finally gets around to
dropping a reference to a slot that's long since been reassigned.
Whatever the reason a `TokenRef` goes away, the outcome is identical and
automatic — and safe, specifically *because* of the epoch check, in the
one case (immediate slot reuse ahead of a straggler) where refcounting
alone would not have been.

This also means there is nothing analogous to the earlier
"instruction-id reuse" concern this document used to describe: that
applied to a shadow dependency graph keyed by host instruction ids, which
no longer exists (§3.2). `Token` objects are referenced purely by pointer
via `TokenRef`; there is no integer id anywhere in this module's state to
accidentally alias across a recycled tag.

## 9. Known limitation worth tracking as a design variant

The distance measured in §3.3 is "producer execution → consumer's actual
execution," which conflates true dependency latency with any scheduling/
structural delay the consumer incurred before winning issue arbitration
(e.g. port contention). A possible variant worth prototyping alongside this
baseline: measure from the consumer's *wakeup* (ready-bit set) to its
*retirement*, rather than producer-execute to consumer-execute. This shifts
the signal from "pure dataflow critical path" toward "is this instruction
actually gating ROB retirement / resource reclamation right now" — a
resource-constrained notion of criticality that may correlate better with
real throughput impact, at the cost of a longer-lived tracking entry (must
survive until commit, including through squash/replay) and conflating
"waiting behind older instructions" with "this instruction's own latency."
Treat this as a follow-on experiment, not a prerequisite for the baseline
implementation above.

---

## 10. Implementation checklist

- [x] `src/token.h` — `Token` (pool-allocated, refcounted) and
      `TokenRef` (RAII handle, meant to be embedded in the host's uop
      class) declarations.
- [x] `src/token.cpp` — `Token::Release()` / `Token::DoNotifyExecuted()`.
- [x] `src/token_engine.h` — public API, `CriticalityConfig`,
      `TokenEngine` class declaration, `CriticalityStatPointers`.
- [x] `src/token_engine.cpp` — implementation of §3.1–§3.5 exactly
      as described above.
- [x] `examples/example_usage.cpp` — minimal synthetic harness (with a
      toy `Uop` struct embedding a `TokenRef`) exercising the API end to
      end, usable as a compile-smoke-test.
- [ ] Host-specific adapter wiring the hooks in §5 into your actual CA
      model's pipeline stages and your real uop class (left to the
      integrating engineer — this is the only piece that is necessarily
      simulator-specific).
- [ ] Reservation-station priority integration per §3.6, in whichever style
      (shifting vs. matrix) your issue queue already uses.
- [ ] (Optional) cache-controller hook per §3.7.

---

## 11. Background reading (generic, non-proprietary)

Token-passing / last-arriving-operand critical-path approximation is a
known family of techniques in the computer-architecture literature on
dynamic critical-path prediction and criticality-aware scheduling
(e.g. work on "critical path prediction" and "slack-based" instruction
scheduling from the early-2000s ISCA/MICRO era). Consult that literature
for alternative formulations (e.g. dataflow-slack estimation, token-passing
over a DAG) if you want to explore variants beyond the baseline described
here.
