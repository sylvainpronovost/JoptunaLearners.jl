# Training and pruning

> Evidence status: historical results and revision labels below are inherited, not a current Joptuna hosted qualification. See the repository-root `QUALIFICATION.md`.

The application supplies held-out validation data and a canonical evaluator:

```julia
contract = ValidationSpec(
    :mse,
    (prediction, heldout) -> sum(abs2, prediction .- heldout.y) / length(prediction);
    direction=:minimize,
    prediction=:prediction,
    target=:target,
    grouping=(:row,),
)

learner = LuxLearner(
    :window_dlinear;
    validation=contract,
    kernel_size=5,
    training=TrainingSpec(
        epochs=40, minimum_epochs=5, patience=5, restore_best=true,
    ),
)
fitted = fit(learner, train_data; validation_data=validation_data, callbacks=(callback,))
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

When `checkpoint_every > 0`, snapshots include parameters, Lux state, optimizer state, shuffle
RNG, batch ordering, event history, best state, and digests for the learner, data, and validation
contract. Resume fails closed when those identities disagree. The epoch budget may increase, but
optimization semantics may not change mid-run.
