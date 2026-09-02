# Publication content boundary

This repository is a standalone, domain-neutral library. Its publication inputs are
Git-tracked files only. Do not distribute a working-directory ZIP: local dependency
environments, generated caches, credentials, and recovery material are not release inputs.

Run before sharing:

```sh
julia --startup-file=no tools/publication_audit.jl --self-test --history --generated
git archive --format=tar --output=../publication-snapshot.tar HEAD
```

The audit checks tracked paths/content and reachable Git history for restricted identifier
fingerprints, obsolete package prefixes (except the public EvoTrees API), personal absolute
paths, credential-shaped strings, and unapproved binary/data/checkpoint archives. Policy
fingerprints prevent the guard itself from publishing a private-name dictionary. Synthetic
numeric TPE trace binaries have an explicit path, shape, and range allowlist. Rendered HTML,
search indexes, and related first-party text assets are checked with `--generated`.

The check is a regression barrier, not a legal ownership determination or a guarantee that
arbitrarily renamed confidential content can be detected. Review source/data provenance when
adding fixtures, model implementations, or evidence. Public datasets and synthetic fixtures
must be identified explicitly. Third-party installed environments are not audited or included
in the Git archive. Never add private recovery bundles to a publication repository.

The September 2026 sanitation establishes fresh publication history. Earlier local histories
are retained only in a private recovery location outside this repository. No legacy API alias
is promised. No remote publication, package-version change, registry submission, or license
grant is part of this increment.

## Neutral learner contracts

- MLJ model: `JoptunaRegressor`; `mlj_model` remains its factory.
- Prediction vector and exported table column: `prediction`.
- Validation defaults: `prediction`, `target`, and grouping by `row_id`.
- Entity conditioning: `uses_entity`, `entity_codes`, and `entity_vocabulary`.
- Hybrid row keys: `group_id`, `entity_id`, `fold_id`; rank groups use `group_id`.
- Residual scale labels: `target_scale` and `target_scale_residual`.

Applications may supply their own explicit column/grouping contracts. No business metric or
dataset is built into these defaults. All learner regression fixtures are synthetic.
Checkpoint schemas advance to fitted-model format 3 and resumable-training format 4;
older layouts are not silently treated as compatible. Retrain or explicitly migrate before
using a checkpoint created against a previous layout.
