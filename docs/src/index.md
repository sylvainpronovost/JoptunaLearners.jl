# JoptunaLearners.jl

The MLJ entry point is `JoptunaRegressor` (or the `mlj_model` factory). Native
predictions expose a `prediction` vector, and `DataFrame(surface)` exports that
column alongside application-provided row keys, scale and provenance. Validation defaults use
`target` and no grouping; entity-conditioned models use training-fitted `entity_codes`.
Hybrid helpers require explicit `key_cols`; `group_cols=()` ranks the whole surface.
Residual scales describe units, not an application domain.

## Small synthetic training example

```@example neutral_schema
using JoptunaLearners, Statistics, DataFrames
x = reshape(Float32.(1:24) ./ 24, 2, 12)
data = LearnerData(x, vec(x[1, :] .- x[2, :]))
metric = ValidationSpec(:mse, (p, d) -> mean(abs2, p .- d.target);
                        direction=:minimize)
learner = LuxLearner(:mlp; hidden_dims=(4,),
    training=TrainingSpec(epochs=1, batch_size=6, patience=0, seed=7))
fitted = fit(learner, data; validation=metric, validation_data=data, verbosity=-1)
surface = predict(fitted, data)
@assert length(surface.prediction) == length(data)
@assert :prediction in propertynames(DataFrame(surface))
size(DataFrame(surface))
```

This is an API smoke example, not a held-out model-quality estimate. Publication
snapshots contain only public benchmark or synthetic data; see the repository-root
`PUBLICATION.md` for the content audit and checkpoint schema boundary.

> Evidence status: historical results and revision labels below are inherited, not a current Joptuna hosted qualification. See the repository-root `QUALIFICATION.md`.

JoptunaLearners provides Julia-native, callback-driven learner training. It owns model
construction, optimization, early stopping, best-state restoration, checkpoints, and execution
provenance. It deliberately does not own cross-validation, metric definitions, HPO, or
application artifacts.

## Validation flow

1. The application constructs training and validation data plus an explicit `ValidationSpec`.
2. JoptunaLearners trains and emits `TrainingEvent`s carrying that contract digest.
3. JoptunaIntegrations can report the same value and digest to an active Joptuna trial.
4. Promotion and final reporting can reject a digest mismatch.

This detects disagreement in declared contracts. It does not inspect evaluator code or prove
that an evaluator accesses the declared target or implements the declared grouping correctly.

See [architecture ledger](model-ledger.md), [ownership and contracts](ownership.md),
[training and pruning](training-and-pruning.md), [execution backends](execution-backends.md), and
[MLJ integration](mlj.md). The primary LearnAPI contract is exercised by LearnTestAPI; the direct
MLJModelInterface adapter and optional Flux adapter are qualified in isolated environments.

Until registration, add the package from its public repository URL and retain resolved revisions
in reproducible application manifests.
