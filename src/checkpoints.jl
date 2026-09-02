const CHECKPOINT_FORMAT_VERSION=3
const TRAINING_CHECKPOINT_FORMAT_VERSION=4

struct _TrainingCheckpoint{M,P,S,O,BP,BS,R,I}
    format_version::Int
    model_name::Symbol
    seed::Int
    contract_digest::String
    learner_digest::String
    training_data_digest::String
    validation_data_digest::String
    epoch::Int
    update::Int
    model::M
    parameters::P
    states::S
    optimizer_state::O
    events::Vector{TrainingEvent}
    best_epoch::Int
    best_value::Float64
    best_parameters::BP
    best_states::BS
    stale::Int
    rng::R
    indices::I
end

function _stable_digest(values...)
    io=IOBuffer()
    for value in values
        Serialization.serialize(io,value)
    end
    bytes2hex(sha256(take!(io)))
end

const _DIGEST_SEPARATOR = (0xff, 0x00, 0xff, 0x00)

function _digest_tag!(context, value)
    bytes = codeunits(string(value))
    length_bytes = ntuple(8) do index
        UInt8((UInt64(length(bytes)) >> (8 * (index - 1))) & 0xff)
    end
    SHA.update!(context, length_bytes)
    SHA.update!(context, bytes)
    SHA.update!(context, _DIGEST_SEPARATOR)
    context
end

function _digest_array!(context, values::AbstractArray)
    _digest_tag!(context, eltype(values))
    _digest_tag!(context, size(values))
    if isbitstype(eltype(values)) && values isa Array
        SHA.update!(context, reinterpret(UInt8, vec(values)))
    elseif eltype(values) <: AbstractString || eltype(values) <: Symbol
        for value in values
            _digest_tag!(context, value)
        end
    else
        # Nullable or wrapper-backed columns retain a canonical element stream
        # without constructing a serialized copy of the complete column.
        for value in values
            _digest_tag!(context, ismissing(value) ? "<missing>" : value)
        end
    end
    context
end

function _digest_windows!(context, source::CausalWindowSource)
    _digest_tag!(context, :causal_window_source)
    _digest_array!(context, source.values)
    _digest_array!(context, source.starts)
    _digest_tag!(context, source.lookback)
end

function _digest_windows!(context, windows::AbstractArray)
    _digest_tag!(context, :dense_windows)
    _digest_array!(context, windows)
end

_digest_windows!(context, ::Nothing) = _digest_tag!(context, :no_windows)

function _digest_frame!(context, frame::DataFrames.AbstractDataFrame)
    _digest_tag!(context, :data_frame)
    _digest_tag!(context, Tuple(propertynames(frame)))
    for column in eachcol(frame)
        _digest_array!(context, column)
    end
    context
end

function _resume_learner_digest(learner::LuxLearner)
    training=learner.training
    resumable_training=(
        batch_size=training.batch_size,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        warmup_epochs=training.warmup_epochs,
        gradient_clip=training.gradient_clip,
        huber_delta=training.huber_delta,
        seed=training.seed,
        backend=_backend_contract(training.backend),
        minimum_epochs=training.minimum_epochs,
        patience=training.patience,
        min_delta=training.min_delta,
        restore_best=training.restore_best,
    )
    _stable_digest(learner.spec.name,learner.model_config,resumable_training)
end

function _learner_data_digest(data::LearnerData)
    context = SHA.SHA256_CTX()
    _digest_tag!(context, :joptunalearners_learner_data_v2)
    _digest_array!(context, data.tabular)
    _digest_windows!(context, data.windows)
    data.entity_codes === nothing ? _digest_tag!(context, :no_entity_codes) :
        _digest_array!(context, data.entity_codes)
    _digest_array!(context, data.target)
    _digest_array!(context, data.weights)
    _digest_frame!(context, data.keys)
    bytes2hex(SHA.digest!(context))
end

function _save_training_checkpoint(path,snapshot::_TrainingCheckpoint)
    mkpath(dirname(abspath(path)))
    JLD2.jldsave(path;snapshot)
    path
end

function _load_training_checkpoint(path)
    snapshot=JLD2.load(path,"snapshot")
    snapshot.format_version==TRAINING_CHECKPOINT_FORMAT_VERSION||
        throw(ArgumentError("unsupported JoptunaLearners training checkpoint version $(snapshot.format_version)"))
    snapshot
end

function save_checkpoint(path::AbstractString,fitted::FittedLearner)
    mkpath(dirname(abspath(path)))
    backend=execution_backend(fitted)
    payload=(format_version=CHECKPOINT_FORMAT_VERSION,created_at=string(now(UTC)),
             learner=fitted.learner,model=fitted.model,
             parameters=_backend_to_host(backend,fitted.parameters),
             states=_backend_to_host(backend,fitted.states),
             optimizer_state=_backend_to_host(backend,fitted.optimizer_state),report=fitted.report,
             context=fitted.context)
    JLD2.jldsave(path;payload)
    path
end

function load_checkpoint(path::AbstractString)
    payload=JLD2.load(path,"payload")
    payload.format_version==CHECKPOINT_FORMAT_VERSION ||
        throw(ArgumentError("unsupported JoptunaLearners checkpoint version $(payload.format_version)"))
    FittedLearner(payload.learner,payload.model,payload.parameters,payload.states,
                  payload.optimizer_state,payload.report,payload.context)
end
