# MLJ integration

> Evidence status: historical results and revision labels below are inherited, not a current Joptuna hosted qualification. See the repository-root `QUALIFICATION.md`.

`JoptunaRegressor` is a native `MLJModelInterface.Deterministic` model backed by a
`LuxLearner`. `MLJModelInterface` is a lightweight direct dependency; the larger MLJ stack is
not required to load JoptunaLearners.

```julia
using JoptunaLearners
using MLJBase

X = (feature_1=randn(Float32, 128), feature_2=randn(Float32, 128))
y = 0.7f0 .* X.feature_1 .- 0.2f0 .* X.feature_2

model = JoptunaRegressor(
    learner=LuxLearner(:mlp; training=TrainingSpec(epochs=20, batch_size=32, seed=7)),
)
mach = machine(model, X, y)
fit!(mach)
prediction = MLJBase.predict(mach, X)
```

The adapter accepts continuous Tables.jl inputs and continuous targets, validates missing and
non-finite values explicitly, supports non-negative sample weights, preserves feature names, and
reorders prediction tables to the fitted feature order. It implements MLJ reformatting and row
selection so MLJ resampling avoids repeated table conversion. Reports expose the native training
report, feature names, row counts, selected validation policy, fitted architecture/configuration,
and per-epoch training losses.

`JoptunaRegressor` is mutable because MLJTuning must apply each candidate value to an
independent model clone. Its `update` method deliberately starts a fresh statistical fit;
only backend compilation artifacts for an identical shape may be reused. This prevents a
candidate from inheriting neural parameters or optimizer state from the preceding candidate.

Backend selection is nested in the learner and remains fixed across an MLJ tuning campaign:

```julia
native = LuxLearner(:mlp; training=TrainingSpec(backend=ReactantCPU()))
model = JoptunaRegressor(learner=native)
```

Import `Reactant` plus `Enzyme`, or `Metal`, before constructing the corresponding backend.
The optional qualification matrix covers MLJ machines, repeated `fit!`, prediction, two-fold
resampling, and Joptuna-backed `JoptunaTuning` on eager CPU, Reactant CPU, and Metal.

## Validation and early stopping

By default, `validation_fraction=0.0`. JoptunaLearners then trains on every row passed by MLJ and
disables early stopping and best-state restoration. This keeps MLJ's resampling strategy as the
sole owner of out-of-sample evaluation.

For ordinary non-temporal MLJ work, set a positive fraction to create a deterministic shuffled
validation split inside each MLJ training fold:

```julia
model = JoptunaRegressor(
    learner=LuxLearner(:mlp; training=TrainingSpec(patience=5, restore_best=true, seed=7)),
    validation_fraction=0.15,
)
```

Do not use this generic shuffled split for ordered, grouped, or leakage-sensitive experiments.
Such applications should supply an explicit validation contract and split geometry.

## Scope

This integration provides the standard deterministic-regression MLJ model lifecycle: `fit`,
`update`, `predict`, `reformat`, `selectrows`, sample weights, fitted parameters, reports, and
training-loss access. It intentionally does not provide classification, probabilistic prediction,
categorical encoding, window/entity routing from arbitrary MLJ tables, or a second HPO engine.
For Joptuna-driven MLJ search, use the separate `JoptunaIntegrations.jl` tuning strategy; the
adapter supplies the learner model it needs.
