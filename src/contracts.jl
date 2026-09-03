const _VALID_DIRECTIONS = (:minimize, :maximize)

"""A framework-neutral architecture and input-routing contract."""
struct ModelSpec
    name::Symbol
    public_name::Symbol
    uses_tabular::Bool
    uses_windows::Bool
    uses_entity::Bool
    temporal::Bool
    output_kind::Symbol
    required_context::Tuple{Vararg{Symbol}}
    defaults::NamedTuple
    schema::NamedTuple
end

"""Training policy owned by JoptunaLearners, independent of data splitting and HPO."""
struct TrainingSpec
    epochs::Int
    batch_size::Int
    learning_rate::Float64
    weight_decay::Float64
    warmup_epochs::Int
    gradient_clip::Float64
    huber_delta::Float64
    seed::Int
    device::Symbol
    backend::ExecutionBackend
    minimum_epochs::Int
    patience::Int
    min_delta::Float64
    restore_best::Bool
    checkpoint_every::Int
end

function TrainingSpec(; kwargs...)
    s = Base.structdiff((; kwargs...), NamedTuple())
    supplied_device = haskey(s, :device)
    supplied_backend = haskey(s, :backend)
    backend = supplied_backend ? get(s, :backend, EagerCPU()) : EagerCPU()
    backend isa ExecutionBackend || throw(ArgumentError(
        "backend must be an JoptunaLearners.ExecutionBackend",
    ))
    device = supplied_device ? Symbol(get(s, :device, :cpu)) : _backend_device(backend)
    if supplied_backend && supplied_device && device != _backend_device(backend)
        throw(ArgumentError(
            "device=$device conflicts with backend=$(_backend_name(backend))",
        ))
    end
    obj = TrainingSpec(
        Int(get(s, :epochs, 40)), Int(get(s, :batch_size, 256)),
        Float64(get(s, :learning_rate, 1.0e-3)), Float64(get(s, :weight_decay, 1.0e-4)),
        Int(get(s, :warmup_epochs, 0)), Float64(get(s, :gradient_clip, 1.0)),
        Float64(get(s, :huber_delta, 1.0)), Int(get(s, :seed, 0)),
        device, backend, Int(get(s, :minimum_epochs, 1)),
        Int(get(s, :patience, 8)),
        Float64(get(s, :min_delta, 0.0)), Bool(get(s, :restore_best, true)),
        Int(get(s, :checkpoint_every, 0)))
    obj.epochs > 0 || throw(ArgumentError("epochs must be positive"))
    obj.batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    obj.learning_rate > 0 || throw(ArgumentError("learning_rate must be positive"))
    obj.gradient_clip >= 0 || throw(ArgumentError("gradient_clip must be nonnegative"))
    obj.minimum_epochs > 0 || throw(ArgumentError("minimum_epochs must be positive"))
    obj.minimum_epochs <= obj.epochs || throw(ArgumentError("minimum_epochs cannot exceed epochs"))
    obj.patience >= 0 || throw(ArgumentError("patience must be nonnegative"))
    obj.device == _backend_device(obj.backend) || throw(ArgumentError(
        "device=$(obj.device) conflicts with backend=$(_backend_name(obj.backend))",
    ))
    return obj
end

"""
Validation semantics supplied by the application or experiment authority.

`evaluator(predictions, data)` must return one finite scalar. The application owns
the metric definition; the library does not assume a business domain.
`metric_id`, `metric_version` and `metric_config` identify the caller's evaluator
contract; they do not hash or prove the meaning of an arbitrary Julia closure.
`grouping=()` denotes ungrouped evaluation. Grouping and target labels are metadata:
the evaluator receives the full data and owns grouping/target access.
"""
struct ValidationSpec{F}
    name::Symbol
    direction::Symbol
    prediction::Symbol
    target::Symbol
    grouping::Tuple{Vararg{Symbol}}
    metric_id::String
    metric_version::String
    metric_config::String
    evaluator::F
    digest::String
end

function _contract_encode(value)
    frame(tag,text) = tag * string(ncodeunits(text)) * ":" * text
    if value isa NamedTuple || value isa AbstractDict
        entries = [(String(key), item) for (key,item) in pairs(value)]
        allunique(first.(entries)) || throw(ArgumentError("metric config keys collide"))
        sort!(entries;by=first)
        return frame("map",join(_contract_encode(key)*_contract_encode(item) for (key,item) in entries))
    elseif value isa Tuple || value isa AbstractVector
        return frame("seq",join(_contract_encode(item) for item in value))
    elseif value isa AbstractString || value isa Symbol
        return frame(value isa Symbol ? "sym" : "str",String(value))
    elseif value isa Bool || value isa Integer || value isa AbstractFloat
        isfinite(value) || throw(ArgumentError("metric config must be finite"))
        return frame(string(typeof(value)),repr(value))
    elseif value === nothing
        return "nothing0:"
    end
    throw(ArgumentError("metric config must contain strings, symbols, finite numbers, sequences or maps"))
end

function contract_digest(name, direction, prediction, target, grouping;
                         metric_id=String(name), metric_version="1", metric_config=(;))
    bytes2hex(sha256(_contract_encode((;schema="validation/v2",name,direction,prediction,target,
                                      grouping=Tuple(grouping),metric_id=String(metric_id),
                                      metric_version=String(metric_version),metric_config))))
end

function ValidationSpec(name::Symbol, evaluator::F; direction::Symbol,
                        prediction::Symbol=:prediction, target::Symbol=:target,
                        grouping=(), metric_id=String(name), metric_version="1",
                        metric_config=(;)) where {F}
    direction in _VALID_DIRECTIONS || throw(ArgumentError("direction must be :minimize or :maximize"))
    groups = Tuple(Symbol.(grouping))
    allunique(groups) || throw(ArgumentError("grouping columns must be unique"))
    isempty(metric_id) && throw(ArgumentError("metric_id must not be empty"))
    isempty(metric_version) && throw(ArgumentError("metric_version must not be empty"))
    digest = contract_digest(name, direction, prediction, target, groups;metric_id,metric_version,metric_config)
    ValidationSpec{F}(name, direction, prediction, target, groups, String(metric_id),
                     String(metric_version),_contract_encode(metric_config),evaluator,digest)
end

contract_digest(spec::ValidationSpec) = spec.digest

function assert_contract(expected::AbstractString, actual::AbstractString)
    expected == actual || throw(ArgumentError("validation contract mismatch: expected $expected, got $actual"))
    nothing
end

"""Immutable callback payload emitted after a completed epoch."""
struct TrainingEvent
    epoch::Int
    update::Int
    training_loss::Float64
    validation_value::Float64
    elapsed_seconds::Float64
    contract_digest::String
end

"""Auditable summary of a completed or interrupted training lifecycle."""
struct TrainingReport
    model_name::Symbol
    status::Symbol
    best_epoch::Int
    best_value::Float64
    completed_epochs::Int
    events::Vector{TrainingEvent}
    contract_digest::String
    seed::Int
    stopped_early::Bool
    restored_best::Bool
    provenance::NamedTuple
end

function _better(direction::Symbol, value::Real, incumbent::Real, min_delta::Real)
    direction === :maximize ? value > incumbent + min_delta : value < incumbent - min_delta
end

_initial_best(direction::Symbol) = direction === :maximize ? -Inf : Inf
