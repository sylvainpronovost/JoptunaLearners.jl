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

Residual operations require an explicit scale. This prevents dimensionless ranks from being
silently added to target-scale predictions.
