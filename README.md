# CPU Criticality

A token-passing dynamic criticality predictor for out-of-order CPU issue
queues: a cheap, self-training mechanism that identifies which in-flight
instructions are likely on the program's critical path and biases issue
scheduling (and optionally cache retention) to prioritize them over
plain age-based ordering.

- Full mechanism, data structures, pipeline integration points, and
  validation plan: [`docs/DESIGN.md`](docs/DESIGN.md)
- Public API: [`include/criticality_predictor.h`](include/criticality_predictor.h)
- Reference implementation: [`src/criticality_predictor.cpp`](src/criticality_predictor.cpp)
- Minimal end-to-end usage / compile smoke test: [`examples/example_usage.cpp`](examples/example_usage.cpp)

## Build the example

```bash
g++ -std=c++17 -Iinclude examples/example_usage.cpp src/criticality_predictor.cpp -o example_usage
./example_usage
```

## Integrating into an existing performance model

The module is simulator-agnostic (C++17, standard library only). A host
model wires in five calls — `onFetch`, `registerDependency`, `onExecute`,
`onFlush`, and a read-only `isPredictedCritical` query from the issue-queue
picker — at its existing fetch/rename/execute/squash pipeline events. See
[`docs/DESIGN.md`](docs/DESIGN.md) section 5 for the exact hook contract.
