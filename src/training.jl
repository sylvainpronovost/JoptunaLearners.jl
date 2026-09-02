"""Primary native Lux learner. Validation semantics are supplied only at `fit` time."""
struct LuxLearner{C<:NamedTuple,V,B}
    spec::ModelSpec
    model_config::C
    training::TrainingSpec
    validation::V
    builder::B
end

function LuxLearner(name::Union{Symbol,AbstractString}; training=TrainingSpec(), validation=nothing, kwargs...)
    spec=model_spec(name)
    LuxLearner(spec, model_config(spec.name;kwargs...), training, validation, nothing)
end

LuxLearner(spec::ModelSpec, config::NamedTuple, training::TrainingSpec) =
    LuxLearner(spec, config, training, nothing, nothing)

"""
    LuxLearner(builder; name=:custom, uses_windows=false, uses_entity=false, ...)

Create a learner around a user-owned native Lux model builder. The builder receives one
named tuple containing `n_features`, `lookback`, `n_entities`, and `config`, and must
return a `Lux.AbstractLuxLayer`. The calling application still owns data preparation and validation.
"""
function LuxLearner(builder; name=:custom, uses_windows=false, uses_entity=false,
                    temporal=uses_windows, required_context=(), training=TrainingSpec(),
                    validation=nothing, kwargs...)
    spec=ModelSpec(Symbol(name),Symbol(name),!uses_windows,Bool(uses_windows),Bool(uses_entity),
        Bool(temporal),:point,Tuple(Symbol.(required_context)),NamedTuple(),NamedTuple())
    LuxLearner(spec,(;kwargs...),training,validation,builder)
end

"""Optional generic Flux learner; its methods are installed by the Flux extension."""
struct FluxLearner{B,T}
    builder::B
    training::T
end
FluxLearner(builder; training=TrainingSpec())=FluxLearner(builder,training)

struct FittedLearner{L,M,P,S,O,R,C}
    learner::L
    model::M
    parameters::P
    states::S
    optimizer_state::O
    report::R
    context::C
end

training_report(f::FittedLearner)=f.report
execution_backend(learner::LuxLearner) = learner.training.backend
execution_backend(fitted::FittedLearner) = execution_backend(fitted.learner)

function execution_provenance(fitted::FittedLearner)
    backend = execution_backend(fitted)
    merge(_backend_provenance(backend), (; report=fitted.report.provenance))
end

LearnAPI.constructor(::LuxLearner) =
    (spec, model_config, training, validation, builder) ->
        LuxLearner(spec, model_config, training, validation, builder)
LearnAPI.functions(::LuxLearner) = (
    :(LearnAPI.fit), :(LearnAPI.learner), :(LearnAPI.clone), :(LearnAPI.strip),
    :(LearnAPI.obs), :(LearnAPI.features), :(LearnAPI.target), :(LearnAPI.weights),
    :(LearnAPI.predict), :(JoptunaLearners.training_report),
)
LearnAPI.kinds_of_proxy(::LuxLearner) = (LearnAPI.Point(),)
LearnAPI.tags(::LuxLearner) = ("regression", "gradient descent", "time series forecasting")
LearnAPI.is_pure_julia(::LuxLearner) = true
LearnAPI.learner(f::FittedLearner)=f.learner
LearnAPI.features(::LuxLearner, data::LearnerData) = LearnerData(
    data.tabular, zeros(Float32, length(data)); windows=data.windows,
    entity_codes=data.entity_codes, weights=ones(Float32, length(data)), keys=data.keys,
)
LearnAPI.target(::LuxLearner, data::LearnerData) = data.target
LearnAPI.weights(::LuxLearner, data::LearnerData) = data.weights

function _weighted_huber(pred, y, weights, delta)
    r=pred.-y
    a=abs.(r)
    losses=ifelse.(a .<= delta, 0.5f0 .* r.^2, Float32(delta) .* (a .- Float32(0.5*delta)))
    denom=max(sum(weights),eps(Float32))
    sum(weights .* losses)/denom
end

function _weighted_huber_derivative(pred, y, weights, delta)
    residual = pred .- y
    threshold = Float32(delta)
    slope = clamp.(residual, -threshold, threshold)
    slope .* weights ./ max(sum(weights), eps(Float32))
end

function _objective(model, ps, st, batch)
    input,y,w,delta=batch
    pred,st2=model(input,ps,st)
    loss=_weighted_huber(pred,y,w,delta)
    return loss,st2,(prediction=pred,)
end


mutable struct _DLinearBatchWorkspace{T}
    trend::Array{T,3}
    prediction::Vector{T}
    derivative::Matrix{T}
    trend_weight::Matrix{T}
    remainder_weight::Matrix{T}
    trend_bias::Matrix{T}
    remainder_bias::Matrix{T}
end

function _DLinearBatchWorkspace(input::Array{T,3}) where {T}
    width = size(input, 1) * size(input, 2)
    observations = size(input, 3)
    _DLinearBatchWorkspace(
        similar(input), Vector{T}(undef, observations),
        Matrix{T}(undef, 1, observations), Matrix{T}(undef, 1, width),
        Matrix{T}(undef, 1, width), Matrix{T}(undef, 1, 1),
        Matrix{T}(undef, 1, 1),
    )
end

function _dlinear_eager_gradients(model::NativeArchitecture, ps, batch)
    input, target, weights, delta = batch[1:4]
    workspace = length(batch) == 5 && batch[5] isa _DLinearBatchWorkspace ?
        batch[5] : _DLinearBatchWorkspace(input)
    _dlinear_eager_prediction!(workspace, model, ps, input)
    trend = workspace.trend
    flat_input = reshape(input, model.n_features * model.lookback, size(input, 3))
    flat_trend = reshape(trend, model.n_features * model.lookback, size(input, 3))

    denominator = max(sum(weights), eps(eltype(weights)))
    threshold = eltype(input)(delta)
    loss = zero(eltype(input))
    @inbounds for index in eachindex(workspace.prediction, target, weights)
        residual = workspace.prediction[index] - target[index]
        absolute = abs(residual)
        loss += weights[index] * (absolute <= threshold ?
            eltype(input)(0.5) * residual * residual :
            threshold * (absolute - eltype(input)(0.5) * threshold))
        workspace.derivative[index] =
            clamp(residual, -threshold, threshold) * weights[index] / denominator
    end
    loss /= denominator

    mul!(workspace.trend_weight, workspace.derivative, transpose(flat_trend))
    mul!(workspace.remainder_weight, workspace.derivative, transpose(flat_input))
    workspace.remainder_weight .-= workspace.trend_weight
    bias_value = sum(workspace.derivative)
    fill!(workspace.trend_bias, bias_value)
    fill!(workspace.remainder_bias, bias_value)
    gradients = (
        trend=(weight=workspace.trend_weight, bias=workspace.trend_bias),
        remainder=(weight=workspace.remainder_weight, bias=workspace.remainder_bias),
    )
    gradients, loss
end

function _dlinear_eager_prediction!(workspace::_DLinearBatchWorkspace,
                                    model::NativeArchitecture, ps, input::Array)
    trend = _moving_average_cpu!(workspace.trend, input, model.config.kernel_size)
    flat_input = reshape(input, model.n_features * model.lookback, size(input, 3))
    flat_trend = reshape(trend, model.n_features * model.lookback, size(input, 3))
    # `remainder = input - trend`, rearranged to avoid allocating that complete
    # batch tensor. Reuse the prediction and trend buffers during validation too.
    prediction = reshape(workspace.prediction, 1, :)
    mul!(prediction, ps.remainder.weight, flat_input)
    mul!(prediction, ps.trend.weight, flat_trend, one(eltype(input)), one(eltype(input)))
    mul!(prediction, ps.remainder.weight, flat_trend, -one(eltype(input)), one(eltype(input)))
    prediction .+= ps.trend.bias .+ ps.remainder.bias
    workspace.prediction
end

# Accelerator-generic analytic DLinear gradient.  Reactant compiles this whole
# expression together with clipping and the optimizer update; Metal executes it
# natively.  Other architectures continue to use their qualified AD backends.
function _dlinear_accelerator_gradients(model::NativeArchitecture, ps, batch)
    input, target, weights, delta = batch[1:4]
    trend = _moving_average_functional(input, model.config.kernel_size)
    flat_input = reshape(input, model.n_features * model.lookback, size(input, 3))
    flat_trend = reshape(trend, model.n_features * model.lookback, size(input, 3))
    prediction = vec(
        ps.remainder.weight * flat_input .+
        (ps.trend.weight .- ps.remainder.weight) * flat_trend .+
        ps.trend.bias .+ ps.remainder.bias
    )
    loss = _weighted_huber(prediction, target, weights, delta)
    derivative = reshape(
        _weighted_huber_derivative(prediction, target, weights, delta), 1, :,
    )
    trend_weight = derivative * transpose(flat_trend)
    remainder_weight = derivative * transpose(flat_input) .- trend_weight
    bias_value = sum(derivative)
    trend_bias = zero(ps.trend.bias) .+ bias_value
    remainder_bias = zero(ps.remainder.bias) .+ bias_value
    gradients = (
        trend=(weight=trend_weight, bias=trend_bias),
        remainder=(weight=remainder_weight, bias=remainder_bias),
    )
    gradients, loss
end

function _dlinear_compiled_train_step(model, batch, ps, st, optimizer_state,
                                      gradient_clip)
    gradients, loss = _dlinear_accelerator_gradients(model, ps, batch)
    gradients = _clip_gradients_traceable(gradients, gradient_clip)
    optimizer_state, ps = Optimisers.update(optimizer_state, ps, gradients)
    loss, ps, st, optimizer_state
end

_tree_sqnorm(x::AbstractArray)=sum(abs2,x)
_tree_sqnorm(x::NamedTuple)=sum(_tree_sqnorm(v) for v in values(x);init=0.0)
_tree_sqnorm(x::Tuple)=sum(_tree_sqnorm(v) for v in x;init=0.0)
_tree_sqnorm(::Nothing)=0.0
_tree_scale(x::AbstractArray,s)=x .* convert(eltype(x), s)
_tree_scale(x::NamedTuple,s)=map(v->_tree_scale(v,s),x)
_tree_scale(x::Tuple,s)=map(v->_tree_scale(v,s),x)
_tree_scale(::Nothing,s)=nothing

function _clip_gradients(gs, ceiling)
    ceiling<=0 && return gs
    norm=sqrt(_tree_sqnorm(gs))
    isfinite(norm) || throw(ArgumentError("nonfinite gradient norm"))
    norm<=ceiling ? gs : _tree_scale(gs,ceiling/(norm+eps(Float64)))
end

# Branch-free form for compiled accelerator kernels. Host-side training retains
# `_clip_gradients` and its explicit nonfinite exception; accelerated callers
# validate finite losses, parameters, and predictions at synchronization gates.
function _clip_gradients_traceable(gs, ceiling)
    norm = sqrt(_tree_sqnorm(gs))
    scale = min(one(norm), ceiling / (norm + eps(Float64)))
    _tree_scale(gs, scale)
end

mutable struct _BatchStager{D,S}
    data::D
    spec::S
    inputs::Dict{Int,Any}
    targets::Dict{Int,Vector{Float32}}
    weights::Dict{Int,Vector{Float32}}
    workspaces::Dict{Tuple{Symbol,Int},Any}
end

_BatchStager(data, spec) = _BatchStager(data, spec, Dict{Int,Any}(),
    Dict{Int,Vector{Float32}}(), Dict{Int,Vector{Float32}}(),
    Dict{Tuple{Symbol,Int},Any}())

function _dlinear_batch_workspace!(stager::_BatchStager, input::Array{T,3}) where {T}
    get!(stager.workspaces, (:dlinear, size(input, 3))) do
        _DLinearBatchWorkspace(input)
    end
end

function _backend_predict(::EagerCPU, model::NativeArchitecture, ps, st,
                          input::Array, stager::_BatchStager)
    model.name === :window_dlinear || return _backend_predict(
        EagerCPU(), model, ps, st, input,
    )
    workspace = _dlinear_batch_workspace!(stager, input)
    Float64.(_dlinear_eager_prediction!(workspace, model, ps, input))
end

function _allocate_staged_input(data::LearnerData, spec::ModelSpec, n::Int)
    if spec.uses_windows
        windows = zeros(Float32, size(data.windows, 1), size(data.windows, 2), n)
        return spec.uses_entity ? (windows=windows, entity_codes=zeros(Int, n)) : windows
    end
    zeros(Float32, size(data.tabular, 1), n)
end

function _fill_staged_input!(destination::AbstractMatrix, data::LearnerData, indices)
    for (column, observation) in enumerate(indices)
        copyto!(@view(destination[:, column]), @view(data.tabular[:, observation]))
    end
    destination
end

function _fill_staged_windows!(destination, source, indices)
    for (column, observation) in enumerate(indices)
        copyto!(@view(destination[:, :, column]), @view(source[:, :, observation]))
    end
    destination
end

function _fill_staged_input!(destination::AbstractArray{<:Any,3}, data::LearnerData, indices)
    _fill_staged_windows!(destination, data.windows, indices)
end

function _fill_staged_input!(destination::NamedTuple, data::LearnerData, indices)
    _fill_staged_windows!(destination.windows, data.windows, indices)
    for (column, observation) in enumerate(indices)
        destination.entity_codes[column] = data.entity_codes[observation]
    end
    destination
end

function _stage_batch!(stager::_BatchStager, indices, delta)
    n = length(indices)
    input = get!(stager.inputs, n) do
        _allocate_staged_input(stager.data, stager.spec, n)
    end
    _fill_staged_input!(input, stager.data, indices)
    target = get!(stager.targets, n) do
        Vector{Float32}(undef, n)
    end
    weights = get!(stager.weights, n) do
        Vector{Float32}(undef, n)
    end
    for (column, observation) in enumerate(indices)
        target[column] = stager.data.target[observation]
        weights[column] = stager.data.weights[observation]
    end
    (input, target, weights, Float32(delta))
end

function _predict_values(backend::ExecutionBackend, model, ps, st, data, spec;
                         batch_size::Int=length(data))
    batch_size > 0 || throw(ArgumentError("prediction batch_size must be positive"))
    prediction = Vector{Float64}(undef, length(data))
    stager = _BatchStager(data, spec)
    for first_index in 1:batch_size:length(data)
        last_index = min(first_index + batch_size - 1, length(data))
        indices = first_index:last_index
        host_batch = _stage_batch!(stager, indices, 1.0)
        batch = _backend_batch(backend, host_batch, stager)
        values = _backend_predict(backend, model, ps, st, first(batch), stager)
        copyto!(@view(prediction[indices]), values)
    end
    _backend_synchronize(backend, prediction)
    prediction
end

function _validation_value(validation::ValidationSpec,pred,data)
    value=Float64(validation.evaluator(pred,data))
    isfinite(value) || throw(ArgumentError("validation evaluator returned nonfinite value"))
    value
end

function _provenance(learner::LuxLearner, validation::ValidationSpec, context,
                     execution_metrics)
    (; package="JoptunaLearners.jl", backend="Lux", julia=string(VERSION),
      execution=_backend_provenance(learner.training.backend),
      execution_metrics,
      model=String(learner.spec.public_name), contract_digest=validation.digest,
      seed=learner.training.seed, context=context === nothing ? NamedTuple() : context)
end

function _native_model(learner::LuxLearner; n_features, lookback, n_entities)
    isnothing(learner.builder) && return build_model(
        learner.spec; n_features, lookback, n_entities, pairs(learner.model_config)...,
    )
    build_context=(;n_features,lookback,n_entities,config=learner.model_config)
    model=learner.builder(build_context)
    model isa Lux.AbstractLuxLayer || throw(ArgumentError(
        "the custom Lux builder must return Lux.AbstractLuxLayer, got $(typeof(model))",
    ))
    model
end

function fit(learner::LuxLearner, data::LearnerData; validation=nothing,
             validation_data::LearnerData=data, context=nothing, callbacks=(),
             checkpoint_path=nothing, resume_from=nothing,
             validation_schedule::Symbol=:epoch,
             verbosity=LearnAPI.default_verbosity())
    validation = isnothing(validation) ? learner.validation : validation
    validation isa ValidationSpec || throw(ArgumentError(
        "a ValidationSpec must be stored on the learner or supplied to fit",
    ))
    learner.spec.uses_windows && data.windows === nothing && throw(ArgumentError("window model requires training windows"))
    learner.spec.uses_entity && data.entity_codes === nothing && throw(ArgumentError("FiLM model requires entity codes"))
    train=learner.training
    validation_schedule in (:epoch, :final, :none) || throw(ArgumentError(
        "validation_schedule must be :epoch, :final, or :none",
    ))
    if validation_schedule !== :epoch
        train.patience == 0 || throw(ArgumentError(
            "non-epoch validation requires patience=0",
        ))
        !train.restore_best || throw(ArgumentError(
            "non-epoch validation requires restore_best=false",
        ))
        isempty(callbacks) || throw(ArgumentError(
            "non-epoch validation cannot drive callbacks",
        ))
        checkpoint_path === nothing && resume_from === nothing || throw(ArgumentError(
            "non-epoch validation does not support interruption checkpoints",
        ))
    end
    backend=train.backend
    rng=MersenneTwister(train.seed)
    lookback=data.windows===nothing ? 1 : size(data.windows,2)
    entities=data.entity_codes===nothing ? 0 : maximum(data.entity_codes)
    model=_native_model(learner;n_features=size(data.tabular,1),lookback,n_entities=entities)
    optimizer=Optimisers.AdamW(Float32(train.learning_rate),(0.9f0,0.999f0),Float32(train.weight_decay);couple=false)
    indices=collect(1:length(data))
    learner_digest=_resume_learner_digest(learner)
    training_data_digest=_learner_data_digest(data)
    validation_data_digest=_learner_data_digest(validation_data)
    snapshot=resume_from===nothing ? nothing : _load_training_checkpoint(String(resume_from))
    setup_started = time_ns()
    if snapshot===nothing
        ps,st=Lux.setup(rng,model)
        ts=_backend_setup(backend,model,ps,st,optimizer,train.gradient_clip)
        events=TrainingEvent[]; best_value=_initial_best(validation.direction); best_epoch=0
        best_ps=deepcopy(ps); best_st=deepcopy(st); stale=0; update=0; first_epoch=1
    else
        assert_contract(validation.digest,snapshot.contract_digest)
        snapshot.model_name==learner.spec.name||throw(ArgumentError("checkpoint model does not match learner"))
        snapshot.seed==train.seed||throw(ArgumentError("checkpoint seed does not match learner"))
        snapshot.learner_digest==learner_digest||throw(ArgumentError("checkpoint learner configuration does not match"))
        snapshot.training_data_digest==training_data_digest||throw(ArgumentError("checkpoint training data does not match"))
        snapshot.validation_data_digest==validation_data_digest||throw(ArgumentError("checkpoint validation data does not match"))
        model=snapshot.model
        ts=_backend_restore(backend,model,snapshot.parameters,snapshot.states,optimizer,
            snapshot.optimizer_state,snapshot.update,train.gradient_clip)
        events=copy(snapshot.events); best_value=snapshot.best_value; best_epoch=snapshot.best_epoch
        best_ps=deepcopy(snapshot.best_parameters); best_st=deepcopy(snapshot.best_states)
        stale=snapshot.stale; update=snapshot.update; first_epoch=snapshot.epoch+1
        first_epoch<=train.epochs||throw(ArgumentError("checkpoint already reached the requested epoch budget"))
        rng=deepcopy(snapshot.rng); indices=copy(snapshot.indices)
    end
    setup_seconds = (time_ns() - setup_started) / 1e9
    started=time()
    stager=_BatchStager(data,learner.spec)
    stopped=false
    first_update_seconds = nothing
    steady_update_seconds = Float64[]
    updates_this_fit = 0
    completed_epochs = first_epoch - 1
    for epoch in first_epoch:train.epochs
        shuffle!(rng,indices)
        lr=train.warmup_epochs>0 && epoch<=train.warmup_epochs ?
            train.learning_rate*epoch/train.warmup_epochs : train.learning_rate
        # Adjust the complete training state rather than only its leaf state.  Lux's
        # Reactant-compatible optimiser keeps runtime-variable hyperparameters in
        # both fields, and returns an updated immutable TrainState.
        ts=Optimisers.adjust!(ts,Float32(lr))
        epoch_loss=nothing
        seen=0
        epoch_update_count=0
        epoch_started=time_ns()
        first_update_in_epoch=0.0
        for first in 1:train.batch_size:length(indices)
            idx=@view indices[first:min(first+train.batch_size-1,end)]
            host_batch=_stage_batch!(stager,idx,train.huber_delta)
            batch=_backend_batch(backend,host_batch,stager)
            if backend isa EagerCPU && learner.spec.name === :window_dlinear
                batch = (batch..., _dlinear_batch_workspace!(stager, batch[1]))
            end
            update_started = time_ns()
            loss,ts=_backend_train_batch(backend,_objective,batch,ts,train.gradient_clip)
            epoch_update_count += 1
            updates_this_fit += 1
            if _backend_times_each_update(backend)
                _backend_synchronize(backend, ts.parameters)
                update_seconds = (time_ns() - update_started) / 1e9
                first_update_seconds === nothing ?
                    (first_update_seconds = update_seconds) :
                    push!(steady_update_seconds, update_seconds)
            elseif first_update_seconds === nothing
                # Synchronize one update so the reported first-update value includes
                # compilation and execution. Later updates remain asynchronous until
                # the epoch metric boundary.
                _backend_synchronize(backend, ts.parameters)
                first_update_in_epoch = (time_ns() - update_started) / 1e9
                first_update_seconds = first_update_in_epoch
            end
            epoch_loss=_backend_accumulate_loss(
                backend,epoch_loss,loss,length(idx),
            )
            update+=1; seen+=length(idx)
        end
        epoch_loss=_backend_finalize_loss(backend,epoch_loss)
        if !_backend_times_each_update(backend)
            epoch_seconds=(time_ns()-epoch_started)/1e9
            remaining=epoch_update_count-(first_update_in_epoch>0 ? 1 : 0)
            if remaining>0
                push!(steady_update_seconds,
                    max(0.0,epoch_seconds-first_update_in_epoch)/remaining)
            end
        end
        completed_epochs = epoch
        evaluate_now = validation_schedule === :epoch ||
            (validation_schedule === :final && epoch == train.epochs)
        event = nothing
        if evaluate_now
            prediction=_predict_values(backend,model,ts.parameters,ts.states,validation_data,
                learner.spec;batch_size=train.batch_size)
            value=_validation_value(validation,prediction,validation_data)
            event=TrainingEvent(epoch,update,epoch_loss/seen,value,time()-started,validation.digest)
            push!(events,event)
            if _better(validation.direction,value,best_value,train.min_delta)
                best_value=value; best_epoch=epoch; best_ps=deepcopy(ts.parameters); best_st=deepcopy(ts.states); stale=0
            else
                stale+=1
            end
        end
        if checkpoint_path!==nothing && train.checkpoint_every>0 && epoch%train.checkpoint_every==0
            host_ps=_backend_to_host(backend,ts.parameters)
            host_st=_backend_to_host(backend,ts.states)
            host_optimizer=_backend_to_host(backend,ts.optimizer_state)
            host_best_ps=_backend_to_host(backend,best_ps)
            host_best_st=_backend_to_host(backend,best_st)
            _save_training_checkpoint(String(checkpoint_path),_TrainingCheckpoint(
                TRAINING_CHECKPOINT_FORMAT_VERSION,learner.spec.name,train.seed,validation.digest,
                learner_digest,training_data_digest,validation_data_digest,
                epoch,update,model,deepcopy(host_ps),deepcopy(host_st),
                deepcopy(host_optimizer),copy(events),best_epoch,best_value,
                deepcopy(host_best_ps),deepcopy(host_best_st),stale,deepcopy(rng),copy(indices)))
        end
        if event !== nothing
            for callback in callbacks; callback(event); end
        end
        if evaluate_now && train.patience>0 && epoch>=train.minimum_epochs && stale>=train.patience
            stopped=true
            break
        end
        if verbosity > 0
            @info "JoptunaLearners epoch" epoch validation_value=(
                event === nothing ? missing : event.validation_value
            ) training_loss=epoch_loss/seen
        end
    end
    if validation_schedule === :none
        best_epoch = completed_epochs
        best_value = NaN
    end
    final_ps=train.restore_best ? best_ps : ts.parameters
    final_st=train.restore_best ? best_st : ts.states
    first_update_seconds = something(first_update_seconds, 0.0)
    steady_state_update_seconds = isempty(steady_update_seconds) ? first_update_seconds :
        median(steady_update_seconds)
    execution_metrics = (; setup_seconds, first_update_seconds,
        steady_state_update_seconds, update_count=updates_this_fit,
        total_fit_seconds=time() - started)
    report=TrainingReport(learner.spec.name,:complete,best_epoch,best_value,completed_epochs,events,
                          validation.digest,train.seed,stopped,train.restore_best,
                          _provenance(learner,validation,context,execution_metrics))
    FittedLearner(learner,model,final_ps,final_st,ts.optimizer_state,report,
                  context===nothing ? NamedTuple() : context)
end

function predict(fitted::FittedLearner, data::LearnerData)
    report=fitted.report
    backend=execution_backend(fitted)
    values=_predict_values(backend,fitted.model,fitted.parameters,fitted.states,data,
        fitted.learner.spec;batch_size=fitted.learner.training.batch_size)
    PredictionSurface(copy(data.keys),values,fitted.learner.spec.name,report.contract_digest,:model_output,report.provenance)
end

LearnAPI.predict(fitted::FittedLearner, ::LearnAPI.Point, data::LearnerData)=
    predict(fitted,data).prediction
