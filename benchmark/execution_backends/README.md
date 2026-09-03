# Execution-backend qualification

This directory contains bounded, supervised evidence for JoptunaLearners execution backends.
It is qualification tooling, not a claim that every accelerator is faster for every model.

Run a synthetic DLinear lifecycle benchmark with:

```sh
benchmark/execution_backends/supervise.sh eager_cpu
benchmark/execution_backends/supervise.sh reactant_cpu
benchmark/execution_backends/supervise.sh metal_gpu
```

For a substantive Metal performance gate, use the compute-heavy PatchTSTLite profile rather than
DLinear, whose small analytic update is intentionally device-launch-bound. The environment
variables make the model and fixed workload explicit; a performance pass still requires a paired
eager/Metal numerical comparison and a predeclared speed threshold:

```sh
JOPTUNALEARNERS_BENCHMARK_MODEL=patchtst JOPTUNALEARNERS_BENCHMARK_FEATURES=16 \
JOPTUNALEARNERS_BENCHMARK_LOOKBACK=128 JOPTUNALEARNERS_BENCHMARK_OBSERVATIONS=4096 \
JOPTUNALEARNERS_BENCHMARK_BATCH_SIZE=2048 JOPTUNALEARNERS_BENCHMARK_EPOCHS=2 \
JOPTUNALEARNERS_BENCHMARK_REPETITIONS=3 \
benchmark/execution_backends/supervise.sh metal_gpu
```

Set `JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1` for the 2026-09-02 host profile.
Without an explicit Julia thread setting the supervisor defaults to one thread.

The authoritative PatchTSTLite profile is 4,096 observations, 16 features, lookback 128, batch
size 2,048, two epochs, and at least three warm repetitions. `check_metal_patchtst.jl` requires
the fixed configuration, exact fail-closed Metal policy, bounded warm-memory growth, prediction
agreement within `5e-4`, and Metal steady-state time no greater than half of eager CPU time.
This qualifies Metal for the **PatchTSTLite large-batch profile only**; it does not alter the
negative DLinear disposition or imply that every architecture benefits from Metal.

Run the identical environment with `eager_cpu` as well; set
`JOPTUNALEARNERS_BENCHMARK_PREDICTION_OUTPUT` separately for each backend to retain the
Float32 prediction payloads required by the checker. Use a dedicated results directory per
campaign and point `JOPTUNALEARNERS_METAL_PATCHTST_RESULTS` to it for the paired check.

The shell supervisor owns a separate process group, samples aggregate RSS every second,
and enforces a timeout. Cleanup escalates from TERM to KILL, including on interruption.
Its default RSS ceiling is 6 GiB; Julia receives a 3 GiB managed-heap hint so collection
starts before host/compiler allocations exhaust that budget. The hint is not a hard memory
limit. Override it with `JOPTUNALEARNERS_BENCHMARK_HEAP_SIZE_HINT` and record the value.
The resolved environment can be selected with `JOPTUNALEARNERS_BENCHMARK_PROJECT`.

RSS is **not total Apple unified-memory usage**. Metal driver allocations are recorded
separately after synchronized full collections for every fit; those samples are not device
peak measurements. Benchmark JSON also records exact data/model dimensions and timing scope.
The paired checker rejects more than 16 MiB of warm-fit driver-allocation growth, in addition
to its managed-memory gate. The runner stops further repetitions when a post-GC driver
sample exceeds 6 GiB or warm-fit growth exceeds 16 MiB. These are boundary checks, not a
continuous device-peak limiter. This catches native retention that RSS alone can miss.
`process_total_seconds` currently covers fit cycles, not Julia startup/data preparation;
the supervisor's elapsed time is the external end-to-end measurement. Generated results are
ignored because they are machine- and environment-specific.

Representative application replays belong outside this package. Their supervisors should refuse
dirty JoptunaLearners, Joptuna, and model-source checkouts and record commit identifiers before
starting, so even a memory-terminated run retains an auditable source-state record.

`bash test/benchmark_supervisor.sh` verifies heap-hint forwarding, timeout/RSS failures,
and TERM-resistant process-group cleanup without launching a model.
