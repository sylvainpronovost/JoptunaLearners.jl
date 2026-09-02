"""Execution policy for JoptunaLearners training and inference."""
abstract type ExecutionBackend end

"""Reference Lux/Zygote execution on ordinary CPU arrays."""
struct EagerCPU <: ExecutionBackend end

"""Reactant/XLA execution on CPU. Reactant failures never fall back silently."""
struct ReactantCPU <: ExecutionBackend
    synchronization::Symbol
    fallback::Symbol
    function ReactantCPU(; synchronization=:epoch, fallback=:error)
        synchronization = Symbol(synchronization)
        synchronization in (:epoch, :step) || throw(ArgumentError(
            "ReactantCPU synchronization must be :epoch or :step",
        ))
        fallback = Symbol(fallback)
        fallback == :error || throw(ArgumentError(
            "ReactantCPU fallback must be :error; silent eager fallback is unsupported",
        ))
        new(synchronization, fallback)
    end
end

"""Native Lux/Metal execution on an Apple GPU."""
struct MetalGPU <: ExecutionBackend
    device::Int
    fallback::Symbol
    function MetalGPU(; device::Integer=1, fallback=:error)
        device == 1 || throw(ArgumentError(
            "MetalGPU currently supports the single Apple GPU exposed as device=1",
        ))
        fallback = Symbol(fallback)
        fallback == :error || throw(ArgumentError(
            "MetalGPU fallback must be :error; silent CPU fallback is unsupported",
        ))
        new(Int(device), fallback)
    end
end

Base.:(==)(::EagerCPU, ::EagerCPU) = true
Base.:(==)(a::ReactantCPU, b::ReactantCPU) =
    a.synchronization == b.synchronization && a.fallback == b.fallback
Base.:(==)(a::MetalGPU, b::MetalGPU) = a.device == b.device && a.fallback == b.fallback

_backend_name(::EagerCPU) = :eager_cpu
_backend_name(::ReactantCPU) = :reactant_cpu
_backend_name(::MetalGPU) = :metal_gpu
_backend_device(::EagerCPU) = :cpu
_backend_device(::ReactantCPU) = :cpu
_backend_device(::MetalGPU) = :metal
_backend_contract(::EagerCPU) = (; name=:eager_cpu)
_backend_contract(backend::ReactantCPU) =
    (; name=:reactant_cpu, synchronization=backend.synchronization, fallback=backend.fallback)
_backend_contract(backend::MetalGPU) =
    (; name=:metal_gpu, device=backend.device, fallback=backend.fallback)

function backend_capabilities(backend::ExecutionBackend)
    name = _backend_name(backend)
    (; name, available=backend isa EagerCPU, device=_backend_device(backend),
       compiled=backend isa ReactantCPU,
       accelerator_backend=!(backend isa EagerCPU),
       performance_qualified=backend isa EagerCPU,
       recommended=backend isa EagerCPU,
       precision=backend isa MetalGPU ? :float32 : :float32_or_float64,
       fallback=:error)
end

"""
    backend_capabilities(backend, model)

Return model-scoped execution and performance qualification. A backend being
available and numerically compatible is deliberately distinct from it being a
useful accelerator for a particular workload.
"""
function backend_capabilities(backend::ExecutionBackend, model)
    (model isa ModelSpec || model isa Symbol || model isa AbstractString) ||
        throw(ArgumentError("model must be a ModelSpec, Symbol, or string"))
    spec = model isa ModelSpec ? model : model_spec(model)
    base = backend_capabilities(backend)
    if spec.name === :window_dlinear
        if backend isa ReactantCPU
            return merge(base, (;
                performance_qualified=true,
                recommended=true,
                qualification=:dlinear_reference_m5_max,
                qualification_result=:pass,
                performance_reason=:compiled_update_throughput,
            ))
        elseif backend isa MetalGPU
            return merge(base, (;
                performance_qualified=false,
                recommended=false,
                qualification=:dlinear_reference_m5_max,
                qualification_result=:correct_but_slower,
                performance_reason=:device_launch_bound,
            ))
        end
        return merge(base, (;
            qualification=:dlinear_reference_m5_max,
            qualification_result=:reference,
            performance_reason=:reference_backend,
        ))
    end
    if spec.name === :patchtst && backend isa MetalGPU
        return merge(base, (;
            performance_qualified=true,
            recommended=true,
            qualification=:patchtst_large_batch_m5_max,
            qualification_result=:pass,
            performance_reason=:large_batch_attention_throughput,
        ))
    end
    merge(base, (;
        qualification=:synthetic_model_zoo,
        qualification_result=:compatibility_only,
        performance_reason=:representative_workload_required,
    ))
end

"""Fail unless `backend` has passed a representative performance gate for `model`."""
function require_performance_qualified(backend::ExecutionBackend, model)
    capabilities = backend_capabilities(backend, model)
    capabilities.available || throw(ArgumentError(
        "backend $(capabilities.name) is unavailable in this process",
    ))
    capabilities.performance_qualified || throw(ArgumentError(
        "backend $(capabilities.name) is not performance-qualified for " *
        "$(model isa ModelSpec ? model.name : Symbol(model)); reason=" *
        "$(capabilities.performance_reason), result=$(capabilities.qualification_result)",
    ))
    capabilities
end

function _backend_unavailable(backend::ExecutionBackend)
    package = backend isa ReactantCPU ? "Reactant and Enzyme" : "Metal"
    throw(ArgumentError(
        "backend $(_backend_name(backend)) is unavailable; import $package to load " *
        "the JoptunaLearners package extension",
    ))
end

_backend_setup(::EagerCPU, model, ps, st, optimizer, gradient_clip) =
    Lux.Training.TrainState(model, ps, st, optimizer)
_backend_setup(backend::ExecutionBackend, model, ps, st, optimizer, gradient_clip) =
    _backend_unavailable(backend)

_backend_restore(::EagerCPU, model, ps, st, optimizer, optimizer_state, update,
                 gradient_clip) =
    Lux.Training.TrainState(
        nothing, nothing, nothing, model, ps, st, optimizer, optimizer_state, update,
    )
_backend_restore(backend::ExecutionBackend, model, ps, st, optimizer, optimizer_state, update,
                 gradient_clip) =
    _backend_unavailable(backend)

_backend_batch(::EagerCPU, batch) = batch
_backend_batch(backend::ExecutionBackend, batch) = _backend_unavailable(backend)
_backend_batch(backend::ExecutionBackend, batch, stager) =
    _backend_batch(backend, batch)

_backend_accumulate_loss(::EagerCPU, accumulated, loss, observations) =
    something(accumulated, 0.0) + Float64(loss) * observations
_backend_finalize_loss(::EagerCPU, accumulated) = Float64(accumulated)
_backend_times_each_update(::EagerCPU) = true
_backend_accumulate_loss(backend::ExecutionBackend, accumulated, loss, observations) =
    _backend_unavailable(backend)
_backend_finalize_loss(backend::ExecutionBackend, accumulated) =
    _backend_unavailable(backend)
_backend_times_each_update(backend::ExecutionBackend) = _backend_unavailable(backend)

function _backend_train_batch(::EagerCPU, objective, batch, ts, gradient_clip)
    if ts.model isa NativeArchitecture && ts.model.name === :window_dlinear
        gs, loss = _dlinear_eager_gradients(ts.model, ts.parameters, batch)
        gs = _clip_gradients(gs, gradient_clip)
        ts = Lux.Training.apply_gradients!(ts, gs)
        return Float64(loss), ts
    end
    gs, loss, _, ts = Lux.Training.compute_gradients(
        ADTypes.AutoZygote(), objective, batch, ts,
    )
    gs = _clip_gradients(gs, gradient_clip)
    ts = Lux.Training.apply_gradients!(ts, gs)
    Float64(loss), ts
end
_backend_train_batch(backend::ExecutionBackend, objective, batch, ts, gradient_clip) =
    _backend_unavailable(backend)

function _backend_predict(::EagerCPU, model, ps, st, input)
    prediction, _ = model(input, ps, st)
    Float64.(prediction)
end
_backend_predict(backend::ExecutionBackend, model, ps, st, input) =
    _backend_unavailable(backend)
function _backend_predict(backend::ExecutionBackend, model, ps, st, input, stager)
    _backend_predict(backend, model, ps, st, input)
end

_backend_to_host(::EagerCPU, value) = value
_backend_to_host(backend::ExecutionBackend, value) = _backend_unavailable(backend)
_backend_synchronize(::EagerCPU, value=nothing) = nothing
_backend_synchronize(backend::ExecutionBackend, value=nothing) = _backend_unavailable(backend)

function _backend_provenance(backend::ExecutionBackend)
    capabilities = backend_capabilities(backend)
    (; execution_backend=String(capabilities.name), device=String(capabilities.device),
       compiled=capabilities.compiled, fallback="error")
end
