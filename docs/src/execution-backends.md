# Execution backends

> Evidence status: historical results and revision labels below are inherited, not a current Joptuna hosted qualification. See the repository-root `QUALIFICATION.md`.

JoptunaLearners separates model and training semantics from execution policy:

```julia
TrainingSpec(backend=EagerCPU())
TrainingSpec(backend=ReactantCPU(synchronization=:epoch))
TrainingSpec(backend=MetalGPU(device=1))
```

`device=:cpu` remains a compatibility alias for `EagerCPU()`. Backend selection is fixed
application infrastructure and should not be suggested as a model hyperparameter. Compilation or
device failures raise an error; there is no silent eager fallback.

| Backend | Runtime | AD | Precision | Claim boundary |
|---|---|---|---|---|
| `EagerCPU` | ordinary Julia/Lux arrays | closed-form DLinear; Zygote otherwise | Float32 model state | portable reference |
| `ReactantCPU` | Reactant/XLA on CPU | Enzyme | Float32 model state | optional, model-specific qualification |
| `MetalGPU` | native Metal arrays | Zygote | Float32 only | optional, model- and hardware-specific qualification |

The supervised synthetic matrix covers all fourteen architectures, numerical comparisons,
deterministic replay, causal and entity contracts, early stopping, restoration, interruption,
checkpoint round trips, and bounded warm-shape memory. These tests establish compatibility, not
universal acceleration.

`prepare_windows(...; materialization=:lazy)` stores one normalized feature surface plus causal
start indices. Reusable staging buffers materialize only the current batch, and validation
inference is streamed. Parameters and optimizer state remain resident for a fit; scalar metrics
and checkpoint payloads cross to the host at explicit boundaries.

Reactant and Metal execution can be asynchronous. JoptunaLearners synchronizes at metric,
checkpoint, callback, prediction, and benchmark boundaries. Reports distinguish backend setup,
first synchronized update, steady-state rate, and total fit time. Checkpoints contain ordinary CPU
arrays, never compiled executables or device arrays.

Use `execution_backend(fitted)`, `backend_capabilities(backend, model)`, and
`execution_provenance(fitted)` for diagnostics. `require_performance_qualified` is a fail-closed
deployment gate. A performance record should always name the model, workload, hardware, package
lock, numerical tolerance, and timing protocol.
