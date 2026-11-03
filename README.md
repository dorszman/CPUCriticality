# CPU Criticality

A token-passing dynamic criticality predictor for out-of-order CPU issue
queues: a cheap, self-training mechanism that identifies which in-flight
instructions are likely on the program's critical path and biases issue
scheduling (and optionally cache retention) to prioritize them over
plain age-based ordering.

- Full mechanism, data structures, pipeline integration points, and
  validation plan: [`docs/DESIGN.md`](docs/DESIGN.md)
- Token API (pool-allocated, refcounted, embedded in your uop class):
  [`include/token.h`](include/token.h) / [`src/token.cpp`](src/token.cpp)
- Engine API: [`include/token_engine.h`](include/token_engine.h) /
  [`src/token_engine.cpp`](src/token_engine.cpp)
- Minimal end-to-end usage / compile smoke test: [`examples/example_usage.cpp`](examples/example_usage.cpp)

## Build the example

```bash
g++ -std=c++17 -Iinclude examples/example_usage.cpp src/token_engine.cpp src/token.cpp -o example_usage
./example_usage
```

## Integrating into an existing performance model

The module is simulator-agnostic (C++17, standard library only). A host
model embeds a `cacrit::TokenRef` as a plain member of its own uop/
micro-op class, and wires in calls to `TokenEngine::TokenAvailable / TokenEngine::DecodeGetToken`,
`registerDependency`, `resolveDependents`, `onFlush`, and a read-only
`isPredictedCritical` query from the issue-queue picker, at its existing
fetch/rename/execute/squash pipeline events. Token lifetime (including on
flush) is handled by ordinary C++ reference counting on `TokenRef` — there
is no separate id-keyed token lookup to maintain on the host side. See
[`docs/DESIGN.md`](docs/DESIGN.md) section 5 for the exact hook contract.
