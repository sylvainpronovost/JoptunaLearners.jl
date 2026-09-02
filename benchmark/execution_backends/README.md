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
JOPTUNALEARNERS_BENCHMARK_LOOKBACK=128 JOPTUNALEARNERS_BENCHMARK_OBSERVATIONS=2048 \
JOPTUNALEARNERS_BENCHMARK_BATCH_SIZE=256 JOPTUNALEARNERS_BENCHMARK_EPOCHS=3 \
benchmark/execution_backends/supervise.sh metal_gpu
```

The authoritative PatchTSTLite profile is 4,096 observations, 16 features, lookback 128, batch
size 2,048, two epochs, and at least three warm repetitions. `check_metal_patchtst.jl` requires
the fixed configuration, exact fail-closed Metal policy, bounded warm-memory growth, prediction
agreement within `5e-4`, and Metal steady-state time no greater than half of eager CPU time.
This qualifies Metal for the **PatchTSTLite large-batch profile only**; it does not alter the
negative DLinear disposition or imply that every architecture benefits from Metal.

The supervisor owns the complete process group, samples external RSS, enforces a timeout and
memory ceiling, and terminates the worker on violation. Generated results are intentionally
ignored because they are machine- and environment-specific.

Representative application replays belong outside this package. Their supervisors should refuse
dirty JoptunaLearners, Joptuna, and model-source checkouts and record commit identifiers before
starting, so even a memory-terminated run retains an auditable source-state record.
