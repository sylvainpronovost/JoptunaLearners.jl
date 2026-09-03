# JoptunaLearners.jl

Publication inputs and audit instructions: [content boundary](PUBLICATION.md).

> **Qualification status:** local reconstruction; inherited results are not a new hosted pass. See [qualification boundaries](QUALIFICATION.md).

`JoptunaLearners.jl` is a Julia-native learner and training library built around Lux. It provides
typed model, training, validation, event, checkpoint, execution-backend, and prediction contracts
without owning cross-validation or hyperparameter optimization.

The package is designed to compose with:

- `Joptuna.jl` and `JoptunaIntegrations.jl` for study lifecycle, reporting, and pruning;
- MLJ through `JoptunaRegressor` and `MLJModelInterface`;
- LearnAPI through native `fit` and `predict` methods;
- Lux as the primary model and training substrate;
- optional Reactant/Enzyme, Metal, and Flux extensions.

## Capabilities

- Fourteen native model contracts across tabular, window-linear, temporal, and advanced groups.
- Deterministic batching, weighted Huber loss, AdamW, schedules, clipping, early stopping, and
  best-state restoration.
- Dense or lazy causal-window data preparation and bounded validation inference.
- Eager CPU, optional Reactant CPU, and optional Apple Metal execution policies.
- Julia-native checkpoints with ordinary CPU arrays and execution provenance.
- Typed validation events suitable for Joptuna reporting and cross-trial pruning.
- MLJ deterministic-regression lifecycle and Joptuna-backed MLJTuning compatibility.
- Stateless rank blending, inner-OOF simplex stacking, and residual-correction primitives.
- Runtime-purity auditing that rejects Python runtimes, bindings, and subprocess dependencies.

## Quick start

```julia
using JoptunaLearners
using Random

X = randn(MersenneTwister(7), Float32, 4, 256)
y = vec(0.7f0 .* X[1, :] .- 0.2f0 .* X[2, :])
data = LearnerData(X, y)

validation = ValidationSpec(
    :mse,
    (prediction, heldout) -> sum(abs2, prediction .- heldout.target) / length(prediction);
    direction=:minimize,
    prediction=:prediction,
    target=:target,
    metric_id="example/mse", metric_version="1", grouping=(),
)

learner = LuxLearner(
    :mlp;
    training=TrainingSpec(epochs=10, batch_size=32, seed=7),
)
fitted = fit(learner, data[1:192]; validation, validation_data=data[193:256])
prediction = predict(fitted, data)
report = training_report(fitted)
```

Applications own dataset construction, split geometry, metric semantics, experiment artifacts,
and final evaluation. JoptunaLearners consumes an explicit `ValidationSpec` and carries its digest
through training so consumers can reject inconsistent declared contracts. The evaluator remains
caller-owned: the digest does not inspect a closure or prove correct target selection.

## Execution backends

`EagerCPU()` is the portable default. `ReactantCPU()` and `MetalGPU()` are strict optional
backends: unavailable operations raise an error instead of silently falling back. Performance
qualification is model- and workload-specific; compatibility alone is not an acceleration claim.

## Scope

JoptunaLearners does not implement its own HPO engine, cross-validation framework, dataset recipe
system, or Python compatibility layer. It never imports or launches Python. See
[`docs/src/index.md`](docs/src/index.md) for the ownership model, training lifecycle, MLJ adapter,
hybrid primitives, execution backends, and model ledger.

Licensing and registry publication remain pending owner review.
