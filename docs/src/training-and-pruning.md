# Training and pruning

> Evidence status: historical results and revision labels below are inherited, not a current Joptuna hosted qualification. See the repository-root `QUALIFICATION.md`.

The application supplies held-out validation data and a canonical evaluator:

```@example training_pruning
using JoptunaLearners, Random
rng = MersenneTwister(7)
windows = randn(rng, Float32, 2, 9, 24)
data = LearnerData(windows[:, end, :], vec(windows[1, end, :]); windows)
train_data, validation_data = data[1:16], data[17:24]
events = TrainingEvent[]
callback = event -> push!(events, event)
contract = ValidationSpec(
    :mse,
    (prediction, heldout) -> sum(abs2, prediction .- heldout.target) / length(prediction);
    direction=:minimize,
    prediction=:prediction,
    target=:target,
    grouping=(), metric_id="example/mse", metric_version="1",
)

learner = LuxLearner(
    :window_dlinear;
    validation=contract,
    kernel_size=5,
    training=TrainingSpec(
        epochs=2, minimum_epochs=1, patience=1, batch_size=8, restore_best=true,
    ),
)
fitted = fit(learner, train_data; validation_data=validation_data, callbacks=(callback,))
@assert !isempty(events)
@assert all(event -> event.contract_digest == contract.digest, events)
training_report(fitted).completed_epochs
```

Every `TrainingEvent` contains the validation-contract digest. The optional
JoptunaIntegrations extension constructs `JoptunaLearnersPruningCallback(trial;
expected_contract_digest=contract.digest)`. It reports completed epochs to Joptuna and throws
`TrialPruned` without intercepting unrelated training errors.

`minimum_epochs` prevents patience-based early stopping before the requested epoch. Setting
`patience=0` disables native early stopping. Applications should use early stopping and pruning
only on legitimate selection data. A promoted fixed refit should use a predetermined epoch count
with early stopping, restoration, and pruning disabled when no separate selection surface exists.

For a user-owned Lux architecture, provide a builder receiving a named tuple with `n_features`,
`lookback`, `n_entities`, and `config`. JoptunaLearners still owns the qualified training lifecycle;
the application owns the meaning and provenance of its data and validation contract.

## Reuse the training entity vocabulary

```@example entity_encoder
using JoptunaLearners, DataFrames
table = DataFrame(entity=repeat(["A", "B"], inner=3), time=repeat(1:3, 2),
                  feature=Float32.(1:6), target=ones(Float32, 6))
settings = (; feature_cols=[:feature], target_col=:target, entity_col=:entity,
             order_col=:time, lookback=2)
training = prepare_windows(table; settings...)
validation = prepare_windows(table[4:6, :]; settings...,
                             entity_encoder=training.entity_encoder)
@assert validation.data.entity_codes == [2, 2]
validation.entity_vocabulary
```

An omitted encoder fits a new vocabulary and is only appropriate for training preparation.
Persist `training.entity_encoder` in application preprocessing/checkpoint context and reuse it
for validation and inference. Unknown entities fail explicitly; no embedding index is guessed.

When `checkpoint_every > 0`, snapshots include parameters, Lux state, optimizer state, shuffle
RNG, batch ordering, event history, best state, and digests for the learner, data, and validation
contract. Resume fails closed when those identities disagree. The epoch budget may increase, but
optimization semantics may not change mid-run.
