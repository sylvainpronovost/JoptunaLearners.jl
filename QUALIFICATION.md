# Qualification and distribution status

Start with the [current coordinated stack status](../Joptuna.jl/docs/src/current-status.md).
Earlier sections below remain dated evidence, not certification of a later dependency pin.

## Committed-stack consolidation — 2026-09-02

At `80ab60e`, the full core suite passed 319 assertions. Focused Reactant CPU controls
passed 10 assertions; Metal controls passed 22, including the 12 native-resource checks.
Strict executable docs passed. These focused reruns do not replace or pretend to repeat
the full model-zoo and large-attention campaigns below. The development branch has been
fast-forwarded into local `main`; its history and existing worktrees remain preserved.

This is a local, unregistered reconstruction; no hosted run for this repository is claimed.
Historical benchmark/campaign files and abbreviated revision identifiers are retained as
inherited evidence, not current-source certification or checkout instructions. Historical
assertion counts are not a coverage percentage. No license grant or publication is implied.

Current core remediation and results are tracked in the sibling
[audit tracker](../Joptuna.jl/docs/src/audit-remediation.md).

## Semantic portability increment — 2026-09-02

Fresh local validation after the contract/hybrid fixes: 246 learner assertions, 34 MLJ
assertions, 2 Flux assertions, LearnTestAPI, and strict executable docs passed on Julia 1.12.7.
This includes all 14 native architecture gradient tests, checkpoint recovery and the actual
README quick start. It is not fresh accelerator performance or hosted qualification.
See the [scoped results and API migration](../Joptuna.jl/docs/src/portability-remediation.md).

Hybrid row keys are now required, validation grouping defaults to empty, entity vocabularies
are reusable training-fitted encoders, residual output is `residual_target`, and checkpoint
layouts are 5/6. Historical accelerator measurements no longer grant runtime qualification.

## Intensive local qualification — 2026-09-02

Fresh real-device tests passed 192 assertions on Reactant CPU and 192 on Metal, including
the complete 14-model lifecycle/memory matrix. MLJ passed 34 assertions on each of eager CPU,
Reactant CPU and Metal, and integrated MLJ tuning passed seven assertions per backend.
These are current behavioral checks, not a universal performance qualification.
The initial fixed large-batch PatchTST performance rerun exceeded its 6 GiB eager-CPU RSS
ceiling and was stopped. That failed run remains a failed run; its subsequent correction is
recorded below. See the
[coordinated report](../Joptuna.jl/docs/src/intensive-qualification.md).

## Attention and native Metal resource correction — 2026-09-02

Batched attention replaces query-by-query reverse-mode intermediates, with 72 forward/full
gradient comparisons and a passing allocation regression (about 80.6% less in the focused
fixture). The full current core suite passed 319 assertions. Metal training and prediction
now use bounded native autorelease scopes, outside differentiation. A controlled matrix-call
probe demonstrated zero driver-memory growth with the scope versus about 40 MiB without it;
explicit Lux-cache cleanup alone was ruled out and is not part of the final patch.

The final 14-model Metal matrix plus eager/DLinear controls passed 206 assertions. Its initial
seven-assertion native-resource probe also passed, and the expanded success/exception resource
probe subsequently passed all 12 assertions. Metal MLJ passed 34 assertions and Joptuna-backed
MLJTuning passed seven. The affected Reactant models and controls passed 36 assertions.
Benchmark safety tests cover the managed-heap hint, timeout/RSS failures, and process-group
cleanup even for TERM-resistant workers. No hosted qualification or universal acceleration
claim follows from these local runs.

The unchanged full-size PatchTST paired run now passes: eager/Metal peak RSS was
5.27/1.81 GiB under the original 6 GiB guard; median warm fits were 21.797/1.202 seconds.
Metal post-fit driver allocation stayed at 4,931,584 bytes across all four fits, instead
of growing to 13,720,305,664 bytes in the rejected intermediate run. Prediction agreement,
deterministic replay, and the 16 MiB warm-memory-growth gates passed. The source/environment
digests and full measurements are in [the fresh record](qualification/metal-attention-20260902.json).
These are this host's synthetic-profile results, not a universal speed claim or an Optuna
comparison. Device snapshots are not peak unified-memory measurements.

Before publication: choose and approve licensing, configure real repository destinations,
pin the qualified core commit, resolve dependency availability, and run the new hosted
workflow. The reviewed upstream gRPC runtime and callback-enabled EvoTrees revision must
be actually obtainable by the intended audience. Registry registration is a separate gate.

Local installations use Julia 1.12 and sibling checkouts, not hypothetical registry entries.
Framework/backend support remains subject to its documented behavioral and hardware gates.
