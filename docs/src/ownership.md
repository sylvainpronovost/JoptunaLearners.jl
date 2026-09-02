# Ownership and validation contracts

JoptunaLearners does not define application-specific metrics or target names. It accepts an opaque
evaluator in `ValidationSpec` and treats the digest of metric name, direction, prediction, target,
and grouping as part of every training event and report.

The application remains responsible for:

- dataset preparation and split geometry;
- leakage and temporal-order controls;
- metric implementation and aggregation;
- promotion rules and final evaluation;
- durable experiment artifacts.

JoptunaLearners owns the training lifecycle only. Joptuna owns the optional study and trial
lifecycle. JoptunaIntegrations validates and transports scalar training events between them.
