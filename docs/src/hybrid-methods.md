# Hybrid methods and leakage boundaries

JoptunaLearners provides stateless calculations; the application owns how their input surfaces are
generated.

| Method | Required evidence | Output scale |
|---|---|---|
| Fixed rank blend | aligned component predictions; weights fixed independently of the evaluated holdout | `dimensionless_rank` |
| Simplex rank stack | aligned inner out-of-fold component predictions; canonical evaluator and contract digest | `dimensionless_rank` |
| Residual target | inner out-of-fold base prediction on the same target scale | `target_scale_residual` |
| Residual correction | aligned target-scale base and residual predictions | `target_scale` |

Every surface is reconciled on caller-declared keys. Duplicate, missing, or extra rows are
rejected. Learned stacking weights and residual targets must not be fitted from final holdout
predictions. In a nested experiment, every component used to learn a stack must be refitted on the
inner-fold geometry. A predeclared fixed blend may reuse accepted holdout predictions because it
learns no weight from them.

`key_cols` is required; no entity, date or fold schema is assumed. `group_cols=()` means
global ranks; specify grouping columns from the row keys for within-group ranks. Include
fold identity in grouping when repeated groups from different folds must remain separate.

`inner_oof=true` is an explicit caller assertion, not independent verification of CV provenance.
Present input contract digests must match; missing digests rely on the caller's declaration.
The helpers preserve per-row component provenance, but cannot verify how predictions were made.

Stack fitting evaluates a finite, deterministic candidate set: every simplex vertex, equal
weights, and optional `weight_candidates`. This avoids false convergence on rank-metric plateaus.
It is not exhaustive global optimization. Ties retain the first candidate; `selection_value`
and `candidate_count` record what was selected and how many candidates were tested.

```@example generic_hybrids
using JoptunaLearners, DataFrames
a = DataFrame(sample=1:4, prediction=[1., 2, 3, 4])
b = DataFrame(sample=1:4, prediction=[3., 1, 4, 2])
score(df) = sum(sign(df.prediction[j] - df.prediction[i]) for i=1:3 for j=i+1:4)
stack = fit_simplex_rank_stack([a, b], score; key_cols=(:sample,),
    inner_oof=true, contract_digest="synthetic-ordering/v1",
    weight_candidates=([0.75, 0.25], [0.25, 0.75]))
@assert stack.selection_value == 6
apply_rank_stack(stack, [a, b]; contract_digest="synthetic-ordering/v1")
```

This synthetic ordering score is only an API illustration, not a validation protocol.

Residual operations require an explicit scale. This prevents dimensionless ranks from being
silently added to target-scale predictions.
`make_residual_targets` returns row keys, a `residual_target` column (customizable with
`output`), scale, contract digest and component provenance. Base and target value columns may
have the same name: they are aligned independently and never subtracted from themselves.
