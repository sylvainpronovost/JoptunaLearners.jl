const MMI = MLJModelInterface

"""A reformatted, feature-first continuous table used by the MLJ adapter."""
struct MLJTabularData
    values::Matrix{Float32}
    feature_names::Vector{Symbol}
end

Base.length(data::MLJTabularData) = size(data.values, 2)

"""
    JoptunaRegressor(; learner=LuxLearner(:mlp), validation=nothing,
                          validation_fraction=0.0)

An MLJ deterministic regressor backed by a native `LuxLearner`.

`validation_fraction=0` is the safe default: all rows presented by MLJ are used for
training and JoptunaLearners' early stopping and best-state restoration are disabled. Set a
positive fraction only when an additional deterministic validation split *inside the MLJ
training fold* is appropriate for the experiment.
"""
mutable struct JoptunaRegressor <: MMI.Deterministic
    learner::JoptunaLearners.LuxLearner
    validation::JoptunaLearners.ValidationSpec
    validation_fraction::Float64
end

function _default_validation()
    JoptunaLearners.ValidationSpec(:mse,
        (prediction, data) -> mean((prediction .- data.target) .^ 2);
        direction=:minimize, prediction=:prediction, target=:target, grouping=(:row_id,))
end

function JoptunaRegressor(; learner=JoptunaLearners.LuxLearner(:mlp), validation=nothing,
                             validation_fraction::Real=0.0)
    validation_spec = validation === nothing ? _default_validation() : validation
    validation_spec isa JoptunaLearners.ValidationSpec || throw(ArgumentError(
        "validation must be an JoptunaLearners.ValidationSpec or nothing"))
    JoptunaRegressor(learner, validation_spec, Float64(validation_fraction))
end

function JoptunaRegressor(learner::JoptunaLearners.LuxLearner;
                             validation=nothing, validation_fraction::Real=0.0)
    JoptunaRegressor(; learner, validation, validation_fraction)
end

JoptunaLearners.mlj_model(learner::JoptunaLearners.LuxLearner;
                       validation=nothing, validation_fraction::Real=0.0) =
    JoptunaRegressor(learner; validation, validation_fraction)

function MMI.clean!(model::JoptunaRegressor)
    model.learner isa JoptunaLearners.LuxLearner || return "learner must be an JoptunaLearners.LuxLearner"
    model.validation isa JoptunaLearners.ValidationSpec || return "validation must be an JoptunaLearners.ValidationSpec"
    0.0 <= model.validation_fraction < 1.0 || return "validation_fraction must satisfy 0 <= validation_fraction < 1"
    ""
end

MMI.metadata_model(JoptunaRegressor;
    input_scitype=MMI.Table(MMI.Continuous),
    target_scitype=AbstractVector{<:MMI.Continuous},
    output_scitype=AbstractVector{<:MMI.Continuous},
    supports_weights=true,
    supports_training_losses=true,
    load_path="JoptunaLearners.JoptunaRegressor",
    human_name="JoptunaLearners native Lux regressor")

function _tabular_data(X)
    X isa MLJTabularData && return X
    Tables.istable(X) || throw(ArgumentError("MLJ input must implement the Tables.jl interface"))
    columns = Tables.columns(X)
    names = collect(Symbol.(Tables.columnnames(columns)))
    isempty(names) && throw(ArgumentError("MLJ input table must have at least one feature column"))
    first_column = collect(Tables.getcolumn(columns, 1))
    n = length(first_column)
    values = Matrix{Float32}(undef, length(names), n)
    for (i, name) in enumerate(names)
        column = collect(Tables.getcolumn(columns, i))
        length(column) == n || throw(ArgumentError("MLJ feature columns have inconsistent lengths"))
        any(ismissing, column) && throw(ArgumentError("MLJ feature column $name contains missing values; impute or coerce before fitting"))
        all(value -> value isa Real, column) || throw(ArgumentError(
            "MLJ feature column $name is not continuous; encode categorical features before fitting"))
        converted = Float32.(column)
        all(isfinite, converted) || throw(ArgumentError("MLJ feature column $name contains nonfinite values"))
        values[i, :] = converted
    end
    MLJTabularData(values, names)
end

function _target_data(y)
    y isa AbstractVector || throw(ArgumentError("MLJ target must be an AbstractVector"))
    any(ismissing, y) && throw(ArgumentError("MLJ target contains missing values"))
    all(value -> value isa Real, y) || throw(ArgumentError("MLJ target must be continuous"))
    converted = Float32.(y)
    all(isfinite, converted) || throw(ArgumentError("MLJ target contains nonfinite values"))
    converted
end

function _weight_data(w, n)
    w === nothing && return ones(Float32, n)
    w isa AbstractVector || throw(ArgumentError("MLJ sample weights must be an AbstractVector"))
    length(w) == n || throw(DimensionMismatch("MLJ sample weights and rows differ"))
    any(ismissing, w) && throw(ArgumentError("MLJ sample weights contain missing values"))
    all(value -> value isa Real, w) || throw(ArgumentError("MLJ sample weights must be continuous"))
    converted = Float32.(w)
    all(value -> isfinite(value) && value >= 0, converted) || throw(ArgumentError(
        "MLJ sample weights must be finite and nonnegative"))
    converted
end

MMI.reformat(::JoptunaRegressor, X) = (_tabular_data(X),)

function MMI.reformat(::JoptunaRegressor, X, y)
    data = _tabular_data(X)
    target = _target_data(y)
    length(data) == length(target) || throw(DimensionMismatch("MLJ input rows and target differ"))
    (data, target)
end

function MMI.reformat(model::JoptunaRegressor, X, y, w)
    data, target = MMI.reformat(model, X, y)
    (data, target, _weight_data(w, length(data)))
end

function MMI.selectrows(::JoptunaRegressor, rows, data::MLJTabularData)
    (MLJTabularData(data.values[:, rows], copy(data.feature_names)),)
end

function MMI.selectrows(model::JoptunaRegressor, rows, data::MLJTabularData, rest...)
    (MMI.selectrows(model, rows, data)[1], map(item -> item[rows], rest)...)
end

function _learner_data(data::MLJTabularData, target, weights, rows)
    JoptunaLearners.LearnerData(data.values[:, rows], target[rows]; weights=weights[rows],
                            keys=DataFrame(row_id=collect(rows)))
end

function _configured_learner(model::JoptunaRegressor, selection_enabled::Bool)
    training = model.learner.training
    if !selection_enabled
        training = JoptunaLearners.TrainingSpec(
            epochs=training.epochs, batch_size=training.batch_size,
            learning_rate=training.learning_rate, weight_decay=training.weight_decay,
            warmup_epochs=training.warmup_epochs, gradient_clip=training.gradient_clip,
            huber_delta=training.huber_delta, seed=training.seed, backend=training.backend,
            minimum_epochs=training.minimum_epochs, patience=0,
            min_delta=training.min_delta, restore_best=false,
            checkpoint_every=training.checkpoint_every)
    end
    JoptunaLearners.LuxLearner(model.learner.spec, model.learner.model_config, training,
                           model.learner.validation, model.learner.builder)
end

function _split_rows(model::JoptunaRegressor, n::Int)
    fraction = model.validation_fraction
    fraction == 0 && return collect(1:n), Int[], false
    n >= 3 || throw(ArgumentError("validation_fraction requires at least three training rows"))
    validation_count = clamp(round(Int, n * fraction), 1, n - 1)
    shuffled = randperm(MersenneTwister(model.learner.training.seed), n)
    validation_rows = sort!(shuffled[1:validation_count])
    validation_set = Set(validation_rows)
    training_rows = [row for row in 1:n if !(row in validation_set)]
    training_rows, validation_rows, true
end

function _fit(model::JoptunaRegressor, verbosity, data::MLJTabularData, target, weights)
    message = MMI.clean!(model)
    isempty(message) || throw(ArgumentError(message))
    length(data) == length(target) || throw(DimensionMismatch("MLJ input rows and target differ"))
    training_rows, validation_rows, selection_enabled = _split_rows(model, length(data))
    training_data = _learner_data(data, target, weights, training_rows)
    validation_data = selection_enabled ?
        _learner_data(data, target, weights, validation_rows) : training_data
    learner = _configured_learner(model, selection_enabled)
    fitted = JoptunaLearners.fit(learner, training_data; validation=model.validation,
        validation_data, verbosity=verbosity - 1,
        context=(feature_names=copy(data.feature_names), mlj_training_rows=length(training_rows),
                 mlj_validation_rows=length(validation_rows),
                 mlj_selection_enabled=selection_enabled))
    report = (
        training_report=JoptunaLearners.training_report(fitted),
        feature_names=copy(data.feature_names),
        training_rows=length(training_rows),
        validation_rows=length(validation_rows),
        selection_enabled=selection_enabled,
        boundary=selection_enabled ?
            "deterministic validation split is contained inside the MLJ training fold" :
            "no internal validation split; early stopping and best-state restoration are disabled",
    )
    cache = (feature_names=copy(data.feature_names), rows=length(data),
             validation_fraction=model.validation_fraction)
    fitted, cache, report
end

function MMI.fit(model::JoptunaRegressor, verbosity, data::MLJTabularData,
                 target::AbstractVector, weights::AbstractVector=ones(Float32, length(target)))
    _fit(model, verbosity, data, _target_data(target), _weight_data(weights, length(data)))
end

function MMI.fit(model::JoptunaRegressor, verbosity, X, y, w=nothing)
    data, target, weights = MMI.reformat(model, X, y, w)
    MMI.fit(model, verbosity, data, target, weights)
end

function MMI.update(model::JoptunaRegressor, verbosity, old_fitresult, old_cache,
                    data::MLJTabularData, target::AbstractVector,
                    weights::AbstractVector=ones(Float32, length(target)))
    # A fresh fit is deliberate: continuing a neural optimiser after MLJ changes either the
    # data rows or a nested learner hyperparameter is not semantically safe.
    MMI.fit(model, verbosity, data, target, weights)
end

function MMI.update(model::JoptunaRegressor, verbosity, old_fitresult, old_cache, X, y, w=nothing)
    data, target, weights = MMI.reformat(model, X, y, w)
    MMI.update(model, verbosity, old_fitresult, old_cache, data, target, weights)
end

function _aligned(data::MLJTabularData, feature_names::AbstractVector{Symbol})
    data.feature_names == feature_names && return data
    Set(data.feature_names) == Set(feature_names) || throw(ArgumentError(
        "MLJ prediction features differ from the fitted feature set"))
    positions = [findfirst(==(name), data.feature_names) for name in feature_names]
    MLJTabularData(data.values[positions, :], collect(feature_names))
end

function MMI.predict(::JoptunaRegressor, fitted::JoptunaLearners.FittedLearner,
                     data::MLJTabularData)
    names = fitted.context.feature_names
    aligned = _aligned(data, names)
    n = length(aligned)
    prediction_data = JoptunaLearners.LearnerData(aligned.values, zeros(Float32, n);
        keys=DataFrame(row_id=collect(1:n)))
    JoptunaLearners.predict(fitted, prediction_data).prediction
end

function MMI.predict(model::JoptunaRegressor, fitted::JoptunaLearners.FittedLearner, X)
    MMI.predict(model, fitted, MMI.reformat(model, X)[1])
end

_parameter_count(x::AbstractArray) = length(x)
_parameter_count(x::NamedTuple) = sum(_parameter_count(value) for value in values(x); init=0)
_parameter_count(::Any) = 0

function MMI.fitted_params(::JoptunaRegressor, fitted::JoptunaLearners.FittedLearner)
    (architecture=fitted.learner.spec.public_name,
     model_config=fitted.learner.model_config,
     feature_names=copy(fitted.context.feature_names),
     parameter_count=_parameter_count(fitted.parameters))
end

MMI.training_losses(::JoptunaRegressor, report) =
    [event.training_loss for event in report.training_report.events]
