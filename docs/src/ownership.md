# Ownership and validation contracts

JoptunaLearners does not define application-specific metrics or target names. It accepts an opaque
evaluator in `ValidationSpec`. The deterministic, length-framed digest includes metric ID,
version, configuration, name, direction, prediction/target labels and grouping. Configuration
maps are order-independent. Callers must change the version or configuration when changing
evaluator semantics. The library cannot infer the meaning of arbitrary Julia closures.

Grouping defaults to `()` (ungrouped). Target/grouping fields document the contract; the
evaluator receives the full data and owns target access and aggregation. Digest equality proves
agreement of declarations, not correctness of the evaluator or absence of data leakage.

The application remains responsible for:

- dataset preparation and split geometry;
- leakage and temporal-order controls;
- metric implementation and aggregation;
- promotion rules and final evaluation;
- durable experiment artifacts.

JoptunaLearners owns the training lifecycle only. Joptuna owns the optional study and trial
lifecycle. JoptunaIntegrations validates and transports scalar training events between them.
