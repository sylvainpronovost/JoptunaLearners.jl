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
deployment gate. Historical accelerator records now return `performance_qualified=false`,
`recommended=false` and `qualification_result=:historical_unverified`. No accelerator is
currently promoted by this API. Eager CPU remains the reference, not a measured speedup claim.
A future performance record must bind the current source, model, workload, hardware, package
lock, numerical tolerance, and timing protocol before this gate can be promoted.

## Attention and fit-local memory

PatchTSTLite and TFTLite use NNlib batched matrix products for multi-head attention.
Small independent reference tests compare forward values and full gradients, including
cross-attention with different query/source lengths. This replaces a query-by-query
reverse-mode allocation pattern without changing the attention normalization or architecture.

The Metal extension bounds native autoreleased command-object lifetimes around each training
batch and prediction call, outside the AD region, including exception exits. Julia GC alone
does not drain native autorelease pools. A small batched-matrix reproducer demonstrated the
retention independently of Lux training; explicit Lux-cache cleanup alone did not solve it.
No fitted-array copy or manual Lux-cache invalidation is needed. Checkpoint conversion remains
a separate host boundary. Accelerator tests track driver allocation as well as managed memory;
process RSS alone does not measure all Apple unified-memory consumption.

The benchmark supervisor defaults to a 6 GiB RSS ceiling and a smaller 3 GiB Julia heap hint,
with process-group cleanup. Neither number is a hard GPU-memory budget. See the execution
benchmark README and the repository qualification report for exact workload-specific results.
