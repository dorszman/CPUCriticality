# CPU Criticality

A token-passing dynamic criticality predictor for out-of-order CPU issue
queues: a cheap, self-training mechanism that identifies which in-flight
instructions are likely on the program's critical path and biases issue
scheduling (and optionally cache retention) to prioritize them over
plain age-based ordering.

- Full mechanism, data structures, pipeline integration points, and
  validation plan: [`docs/DESIGN.md`](docs/DESIGN.md)
- Token API (pool-allocated, refcounted, embedded in your uop class):
  [`src/token.h`](src/token.h) / [`src/token.cpp`](src/token.cpp)
- Engine API: [`src/token_engine.h`](src/token_engine.h) /
  [`src/token_engine.cpp`](src/token_engine.cpp)
- Minimal end-to-end usage / compile smoke test: [`examples/example_usage.cpp`](examples/example_usage.cpp)

## Build the example

```bash
g++ -std=c++17 -Isrc examples/example_usage.cpp src/token_engine.cpp src/token.cpp -o example_usage
./example_usage
```

## Integrating into an existing performance model

The module is simulator-agnostic (C++17, standard library only). A host
model embeds a `cacrit::TokenRef` as a plain member of its own uop/
micro-op class, calls `TokenEngine::Tick` once per cycle, and wires in
`TokenEngine::TokenAvailable` / `DecodeGetToken` at decode and
`TokenRef::NotifyExecuted` at execute. There is no execute/rename/flush
hook on the engine at all: hand-off rides directly on the host's own
issue-queue wakeup logic (a producer whose broadcast wakes one consumer
hands its token straight to it; wakes several and they each get their own
copy and race — fastest execute wins, no arbitration needed; wakes nobody
and the token is simply dropped, treated as non-critical). Token lifetime
(including on flush, and on immediate slot reuse ahead of a slower
fan-out straggler) is handled entirely by ordinary, epoch-guarded C++
reference
counting on `TokenRef` — there is no id-keyed token lookup, register
scoreboard, or dependency graph to maintain on the host side. See
[`docs/DESIGN.md`](docs/DESIGN.md) section 5 for the exact hook contract.
