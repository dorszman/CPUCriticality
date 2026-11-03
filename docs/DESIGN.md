# Token-Passing Critical-Instruction Prediction for Issue-Queue Scheduling

**Audience:** this document is written to brief an engineer or AI coding agent
well enough to implement the mechanism below inside a cycle-accurate (CA)
out-of-order CPU performance model, and to extend/retarget the accompanying
reference implementation (`include/criticality_predictor.h`,
`src/criticality_predictor.cpp`) onto a specific host simulator.

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
| Token | A small, scarce tag that rides along one chain of last-arriving-operand edges, carrying just enough state to measure one edge's latency at a time |
| Distance | Cycles between a producer's execution and the execution of the consumer for which it supplied the last-arriving operand |
| Training table | PC-indexed table of saturating counters, incremented/decremented by observed distances |
| Prediction table | PC-indexed 1-bit table (critical / non-critical), promoted from the training table once a counter saturates |

---

## 3. Mechanism

### 3.1 Sparse token assignment (fetch stage)

- Maintain a small fixed pool of tokens (reference default: 8 tokens, i.e.
  a 3-bit token ID — deliberately small to bound area/power).
- As instructions are fetched/renamed, assign a free token to a instruction
  only occasionally — e.g. on average 1 in N fetched instructions
  (`tokenAssignInterval`), using a cheap pseudo-random or counter-based
  trigger. **Do not tag every instruction.** Sparse sampling is what keeps
  this cheap; the predictor is trained from a statistical sample of the
  dynamic instruction stream, not from exhaustive tracking.
- A freshly tagged instruction becomes the *root* of a new chain: it has no
  predecessor yet, so no training happens until it executes.

### 3.2 Last-arriving-operand detection (rename / dependency resolution)

- At rename, for every source operand of a newly-renamed instruction whose
  producer has **not yet executed**, register a dependency edge
  `(consumer, producer)`. This is bookkeeping any OoO model already needs in
  some form (register scoreboard / wakeup matrix), so this step mostly
  reuses existing dependency-tracking logic rather than adding new state.
- Maintain, per consumer, a count of not-yet-ready source operands.
- When a producer executes, decrement the pending-operand count of every
  consumer that depends on it. **Whichever consumer's count reaches zero as
  a direct result of this producer's completion has just had this producer
  supply its last-arriving operand** — by definition, this is the edge of
  interest.
- **Fan-out is common, and matters for how the tie-break is chosen.** A
  single producer can resolve several consumers' last-arriving operand in
  the same cycle — a shared condition code, a loop induction variable, a
  cache-line fill waking multiple waiting loads. Exactly one token
  continues forward regardless of fan-out width (forking per sibling would
  let one wide-fan-out producer exhaust the whole token pool in a single
  cycle, which defeats the bounded-cost point of this mechanism). Which
  sibling it continues to must be chosen **fairly**, not by a fixed rule
  such as "first in program order": a fixed rule picks the *same* sibling
  on every dynamic instance of the same static fan-out (since the fan-out
  shape itself is static code structure), which permanently starves every
  other sibling's edge of ever being sampled. The reference implementation
  uses reservoir sampling across whichever consumers become ready in that
  cycle, so every sibling gets an equal long-run chance of being the one a
  given token follows (verified empirically with a synthetic three-way
  fan-out harness: ~33/34/33% selection across siblings over 2000 trials).
- **This does not fully solve fan-out, and it's worth knowing why.**
  Training is indexed by the *producer's* PC, not by the specific edge —
  so a producer that is genuinely critical for one consumer but irrelevant
  for several others still has only one shared counter. Its trained
  verdict ends up as an implicit average over however its real uses get
  sampled, which can mask a critical corner-case use sitting among many
  unimportant ones. Fair tie-breaking makes that average *representative*
  of the producer's actual fan-out instead of arbitrarily biased toward
  whichever sibling happens to come first in a list, but it does not give
  per-edge fidelity. Getting that would require per-edge (not per-PC)
  prediction state, which is a materially larger table — a deliberate
  scope boundary of this baseline design, not an oversight.

### 3.3 Token hand-off and distance measurement (execute / writeback)

For an instruction `I` executing in cycle `now`:

1. **Close the previous leg, if `I` was the awaited consumer of an in-flight
   token.** Compute `distance = now - producerExecCycle` and train the
   training-table entry indexed by the *producer's* PC (not `I`'s PC) with
   this distance (see §3.4). This is the measurement the whole mechanism
   exists to produce: "when this static instruction supplies someone's
   last-arriving operand, how many cycles typically elapse before that
   consumer executes?" A short, consistent distance means the dependency
   chain through this instruction is tight — nothing else had a chance to
   interleave — which is the behavioral signature of a critical-path edge.
   A long distance means there was slack: the consumer's actual execution
   time was not determined by this edge alone.
2. **Re-root the token at `I`.** Regardless of whether step 1 fired, if `I`
   currently holds a token, record `producerPC = I`'s PC and
   `producerExecCycle = now` — this instruction is now the head of the
   next leg of the chain.
3. **Resolve `I`'s own dependents** (§3.2). If this identifies a consumer
   `C` for which `I` supplied the last-arriving operand, and `I` holds a
   token, hand the token to `C` (the token now waits for `C`'s own
   execution to close this new leg, per step 1, on a future call).
4. **Terminate the chain if there is nowhere to extend.** If `I` holds a
   token but no dependent's count reached zero this cycle, the chain has
   nothing to extend to; free the token back to the pool immediately. This
   keeps token occupancy bounded and self-cleaning — there is no scenario
   where a token sits idle indefinitely.

Steps 1–4 are the entire runtime algorithm. Note that dependency-count
bookkeeping (§3.2) runs on *every* instruction's execution regardless of
token state — tokens only gate whether a *measurement* is taken on a given
edge, not whether the dependency graph itself is tracked.

### 3.4 Training table (PC-indexed, saturating counter)

- Indexed by the *producer's* PC (same indexing scheme as an instruction
  cache lookup, so it can be probed in parallel with fetch with no added
  latency on the critical fetch path).
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

### 3.5 Prediction table (PC-indexed, 1 bit)

- When a training counter crosses a promotion threshold in either
  direction, write a 1-bit critical/non-critical verdict into the
  prediction table, indexed the same way.
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
replacement policy unmodified. This only needs one more
`isPredictedCriticalForEviction(pc)` query at the cache controller; the
training/prediction tables are unchanged and shared.

---

## 4. Data structures (reference sizes — tune for your target)

| Structure | Fields | Default size |
|---|---|---|
| Token pool entry | `valid`, `state {AwaitingRootExec, AwaitingConsumerExec}`, `rootPc`, `rootExecCycle`, `pendingId` | 8 entries (3-bit token id) |
| Training table entry | `valid`, `tag (PC)`, `count` (signed, saturating) | 1024 entries, direct-mapped |
| Prediction table entry | `valid`, `tag (PC)`, `critical` (1 bit) | 1024 entries, direct-mapped |
| Dependency bookkeeping | `producer -> [consumer...]`, `consumer -> pendingCount` | sized to in-flight window (ROB-sized upper bound) |

Direct-mapped, PC-truncated-tag tables are the cheap realistic choice for
both training and prediction tables; the reference implementation stores
full PCs for simplicity/correctness in a functional model and notes where a
real design would truncate tag bits to save area.

---

## 5. Required integration hooks from the host CA model

The predictor is deliberately decoupled from any specific simulator. The
host model must call it at exactly these points, which correspond to
pipeline events essentially every OoO performance model already has:

| Pipeline event (host side) | Call into predictor |
|---|---|
| Instruction fetched/renamed | `bool crit = predictor.onFetch(id, pc);` — use `crit` to tag the instruction for later scheduling use |
| Dependency discovered at rename, producer still in flight | `predictor.registerDependency(consumerId, producerId);` once per such operand |
| Instruction executes / broadcasts result | `predictor.onExecute(id, pc, currentCycle);` exactly once per instruction |
| Instruction squashed (branch mispredict, exception, etc.) | `predictor.onFlush(id);` |
| Issue-queue select logic | `predictor.isPredictedCritical(pc)` or the tag captured at fetch; combine with `CriticalityPredictor::priorityBoost(...)` per §3.6 |

No other coupling is required. The predictor never needs to know about
physical register files, ROB indices, or the host's own instruction
representation beyond opaque `InstrId`/`Addr`/`Cycle` integers.

---

## 6. Configuration knobs

All exposed via `CriticalityConfig` (see header). Starting points to tune
against your workload mix:

- `tokenPoolSize`: trade-off between statistical sample rate and area/power.
- `tokenAssignInterval`: sparsity of sampling; smaller = more samples, more
  power, faster convergence; larger = cheaper, slower/noisier convergence.
- `criticalDistanceCycles` / `nonCriticalDistanceCycles`: should bracket the
  expected "back-to-back, no slack" latency for your pipeline's bypass
  network (e.g. near 1–2 cycles for ALU-to-ALU forwarding) versus a distance
  that clearly indicates slack existed.
- `trainCounterMin/Max`, `promoteCriticalAt/NonCriticalAt`: hysteresis depth
  before a prediction flips; deeper hysteresis = more stable but slower to
  adapt to phase changes.

---

## 7. Validation plan

1. **Unit-level correctness.** Build a synthetic dependency chain (a fixed
   sequence of registered producer/consumer edges with known execute
   cycles) and assert the predictor trains the expected PCs critical/
   non-critical given deliberately short/long synthetic distances.
2. **Chain termination / token reuse.** Verify tokens are reclaimed when a
   chain dead-ends (no dependent ready), and that the token pool never
   exceeds its configured size (no leaks across flush events).
3. **Flush correctness.** Inject squash events mid-chain and verify no
   stale training occurs from instructions that never actually executed in
   the committed path.
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

## 8. Flush handling and instruction-id lifetime

A token never needs a reference count: by construction a token has at most
one pending awaiter at a time (the mechanism tracks a single chain, not a
fan-out DAG), so its liveness collapses to a single valid bit plus a
reverse lookup (`InstrId -> token slot`). On `onFlush(id)`, if `id` happens
to be a token's current awaitee, the token is freed immediately — the same
effect a refcount reaching zero would have, just without needing a counter
that can only ever be 0 or 1.

What does need explicit handling is the other direction: `id` flushed while
it is still an *unexecuted producer* that some other, surviving consumer is
depending on. That consumer's pending-operand count can now never
legitimately reach zero through this path, since the only place it
decrements is the producer's own `onExecute`, which will never come. The
reference implementation handles this by having `onFlush` walk the flushed
instruction's registered dependents and **erase** (not decrement) their
pending-operand counts — which stops any future producer from mistaking a
stale count for a genuine last-arriving-operand event, and avoids leaving
the bookkeeping map entry to linger indefinitely. In most real pipelines
this case is naturally rare (a flush squashes a contiguous younger suffix
of the window, so a flushed producer's consumers are almost always flushed
in the same event), but the module does not rely on that to stay correct.

**Instruction-id lifetime contract (read this before integrating):** the
module keys all of its internal state — `dependents`, `pendingCount`,
`holderIndex` — by the raw `InstrId` the host passes in. If the host reuses
id values (e.g. a small ROB-tag width recycled as instructions retire), a
new instruction could alias a still-referenced old id before that old id's
predictor-side state has been fully cleared (every id touched by
`onFetch`/`registerDependency` is guaranteed cleared by the matching
`onExecute` or `onFlush` call — the contract is simply: **do not reuse an
id until one of those two calls has been made for it**). Hosts whose tags
are recycled faster than that should widen the id passed to this module
(e.g. concatenate the ROB tag with a per-slot generation/epoch counter)
rather than passing the raw recycled tag directly.

## 10. Known limitation worth tracking as a design variant

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

## 11. Implementation checklist

- [x] `include/criticality_predictor.h` — public API, `CriticalityConfig`,
      `CriticalityPredictor` class declaration, `Stats`.
- [x] `src/criticality_predictor.cpp` — implementation of §3.1–§3.5 exactly
      as described above.
- [x] `examples/example_usage.cpp` — minimal synthetic harness exercising
      the API end to end, usable as a compile-smoke-test.
- [ ] Host-specific adapter wiring the five hooks in §5 into your actual CA
      model's pipeline stages (left to the integrating engineer — this is
      the only piece that is necessarily simulator-specific).
- [ ] Reservation-station priority integration per §3.6, in whichever style
      (shifting vs. matrix) your issue queue already uses.
- [ ] (Optional) cache-controller hook per §3.7.

---

## 12. Background reading (generic, non-proprietary)

Token-passing / last-arriving-operand critical-path approximation is a
known family of techniques in the computer-architecture literature on
dynamic critical-path prediction and criticality-aware scheduling
(e.g. work on "critical path prediction" and "slack-based" instruction
scheduling from the early-2000s ISCA/MICRO era). Consult that literature
for alternative formulations (e.g. dataflow-slack estimation, token-passing
over a DAG) if you want to explore variants beyond the baseline described
here.
