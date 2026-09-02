module JoptunaLearnersReactantExt

using ADTypes
using Enzyme
using JoptunaLearners
using Lux
using Optimisers
using Reactant

const _COMPILATION_CACHE_LOCK = ReentrantLock()
const _COMPILATION_CACHE = Dict{Any,Any}()
const _GRADIENT_CLIP_CACHE = Dict{Any,Any}()
const _DLINEAR_STEP_CACHE = Dict{Any,Any}()
const _PREDICTION_CACHE = Dict{Any,Any}()
const _COMPILATION_CACHE_HITS = Threads.Atomic{Int}(0)
const _COMPILATION_CACHE_MISSES = Threads.Atomic{Int}(0)

_shape_signature(value::AbstractArray) = (typeof(value), size(value))
_shape_signature(value::NamedTuple) = map(_shape_signature, value)
_shape_signature(value::Tuple) = map(_shape_signature, value)
_shape_signature(value::Number) = typeof(value)
_shape_signature(::Nothing) = Nothing
_shape_signature(value) = typeof(value)

function _cache_key(backend, objective, ts, batch)
    (
        typeof(objective), typeof(ts.model), typeof(ts.optimizer),
        _shape_signature(ts.parameters), _shape_signature(ts.states),
        _shape_signature(ts.optimizer_state), _shape_signature(batch),
        backend.synchronization,
        string(pkgversion(Reactant)), string(pkgversion(Enzyme)), string(VERSION),
    )
end

function _with_cached_executable(ts, objective, key)
    cache = lock(_COMPILATION_CACHE_LOCK) do
        get(_COMPILATION_CACHE, key, nothing)
    end
    if cache === nothing
        Threads.atomic_add!(_COMPILATION_CACHE_MISSES, 1)
        # A TrainState may carry the executable for the preceding batch shape.
        # A genuine key miss must clear it or Lux will attempt to run (for
        # example) a full-batch executable on a smaller remainder batch.
        return Lux.Training.TrainState(
            nothing, nothing, ts.allocator_cache, ts.model, ts.parameters, ts.states,
            ts.optimizer, ts.optimizer_state, ts.step,
        )
    end
    Threads.atomic_add!(_COMPILATION_CACHE_HITS, 1)
    Lux.Training.TrainState(
        cache, objective, ts.allocator_cache, ts.model, ts.parameters, ts.states,
        ts.optimizer, ts.optimizer_state, ts.step,
    )
end

function _remember_executable!(key, ts)
    lock(_COMPILATION_CACHE_LOCK) do
        _COMPILATION_CACHE[key] = ts.cache
    end
    ts
end

function _compiled_clip_gradients(gs, ceiling, key)
    clip_key = (key, :global_gradient_clip, Float32(ceiling))
    compiled = lock(_COMPILATION_CACHE_LOCK) do
        get(_GRADIENT_CLIP_CACHE, clip_key, nothing)
    end
    if compiled === nothing
        threshold = Float32(ceiling)
        compiled = Reactant.@compile JoptunaLearners._clip_gradients_traceable(gs, threshold)
        lock(_COMPILATION_CACHE_LOCK) do
            _GRADIENT_CLIP_CACHE[clip_key] = compiled
        end
    end
    compiled(gs, Float32(ceiling))
end

function _device()
    Reactant.set_default_backend("cpu")
    Lux.reactant_device()
end

function JoptunaLearners.backend_capabilities(backend::JoptunaLearners.ReactantCPU)
    base = invoke(JoptunaLearners.backend_capabilities,
        Tuple{JoptunaLearners.ExecutionBackend}, backend)
    merge(base, (; available=true, runtime="Reactant", ad="Enzyme",
                  synchronization=String(backend.synchronization)))
end

function JoptunaLearners._backend_setup(::JoptunaLearners.ReactantCPU, model, ps, st, optimizer,
                                    gradient_clip)
    dev = _device()
    Lux.Training.TrainState(model, dev(ps), dev(st), optimizer)
end

function JoptunaLearners._backend_restore(::JoptunaLearners.ReactantCPU, model, ps, st,
                                      optimizer, optimizer_state, update, gradient_clip)
    dev = _device()
    fresh = Lux.Training.TrainState(model, dev(ps), dev(st), optimizer)
    restored_optimizer_state = _restore_optimizer_state(
        fresh.optimizer_state, optimizer_state, dev,
    )
    Lux.Training.TrainState(nothing, nothing, fresh.allocator_cache, model,
        fresh.parameters, fresh.states, fresh.optimizer, restored_optimizer_state, update)
end

# Checkpoints intentionally contain ordinary host arrays.  Their optimiser leaves
# may therefore also contain the host copy of Lux's Reactant wrapper.  Reuse the
# freshly constructed device-aware rules and restore only the numerical moments;
# otherwise a later learning-rate adjustment cannot discover a Reactant device.
_restore_optimizer_state(fresh::Optimisers.Leaf, saved::Optimisers.Leaf, dev) =
    Optimisers.Leaf(
        fresh.rule,
        Reactant.to_rarray(saved.state; track_numbers=true),
        fresh.frozen,
    )
_restore_optimizer_state(fresh::NamedTuple, saved::NamedTuple, dev) =
    map((f, s) -> _restore_optimizer_state(f, s, dev), fresh, saved)
_restore_optimizer_state(fresh::Tuple, saved::Tuple, dev) =
    map((f, s) -> _restore_optimizer_state(f, s, dev), fresh, saved)
_restore_optimizer_state(fresh, saved, dev) = dev(saved)

JoptunaLearners._backend_batch(::JoptunaLearners.ReactantCPU, batch) = _device()(batch)

_refresh_batch!(destination::Reactant.AbstractConcreteArray, source::Array) =
    copyto!(destination, source)
_refresh_batch!(destination::NamedTuple, source::NamedTuple) =
    map(_refresh_batch!, destination, source)
_refresh_batch!(destination::Tuple, source::Tuple) =
    map(_refresh_batch!, destination, source)
_refresh_batch!(destination::Number, source::Number) = source

function JoptunaLearners._backend_batch(::JoptunaLearners.ReactantCPU, batch,
                                    stager::JoptunaLearners._BatchStager)
    key = (:reactant_batch, length(batch[2]))
    device_batch = get!(stager.workspaces, key) do
        _device()(batch)
    end
    _refresh_batch!(device_batch, batch)
end

function _host_scalar(value)
    host = Lux.cpu_device()(value)
    Float64(host isa Number ? host : only(host))
end

function JoptunaLearners._backend_train_batch(backend::JoptunaLearners.ReactantCPU,
                                          objective, batch, ts, gradient_clip)
    key = _cache_key(backend, objective, ts, batch)
    if ts.model isa JoptunaLearners.NativeArchitecture &&
       ts.model.name === :window_dlinear
        compiled = lock(_COMPILATION_CACHE_LOCK) do
            get(_DLINEAR_STEP_CACHE, key, nothing)
        end
        if compiled === nothing
            Threads.atomic_add!(_COMPILATION_CACHE_MISSES, 1)
            threshold = Float32(gradient_clip)
            compiled = Reactant.@compile JoptunaLearners._dlinear_compiled_train_step(
                ts.model, batch, ts.parameters, ts.states, ts.optimizer_state, threshold,
            )
            lock(_COMPILATION_CACHE_LOCK) do
                _DLINEAR_STEP_CACHE[key] = compiled
            end
        else
            Threads.atomic_add!(_COMPILATION_CACHE_HITS, 1)
        end
        loss, parameters, states, optimizer_state = compiled(
            ts.model, batch, ts.parameters, ts.states, ts.optimizer_state,
            Float32(gradient_clip),
        )
        ts = Lux.Training.TrainState(
            ts.cache, ts.objective_function, ts.allocator_cache, ts.model,
            parameters, states, ts.optimizer, optimizer_state, ts.step + 1,
        )
        return loss, ts
    end
    ts = _with_cached_executable(ts, objective, key)
    gs, loss, _, ts = Lux.Training.compute_gradients(
        ADTypes.AutoEnzyme(), objective, batch, ts;
        sync=backend.synchronization == :step,
    )
    gs = gradient_clip <= 0 ? gs :
        _compiled_clip_gradients(gs, gradient_clip, key)
    ts = Lux.Training.apply_gradients!(ts, gs)
    _remember_executable!(key, ts)
    loss, ts
end

JoptunaLearners._backend_accumulate_loss(::JoptunaLearners.ReactantCPU, accumulated, loss,
                                     observations) =
    accumulated === nothing ? loss * Float32(observations) :
    accumulated + loss * Float32(observations)
JoptunaLearners._backend_finalize_loss(::JoptunaLearners.ReactantCPU, accumulated) =
    _host_scalar(accumulated)
JoptunaLearners._backend_times_each_update(::JoptunaLearners.ReactantCPU) = false

function JoptunaLearners._backend_predict(::JoptunaLearners.ReactantCPU, model, ps, st, input)
    # Loaded checkpoints intentionally contain ordinary CPU arrays.  Normalize
    # all prediction operands at this boundary; `_device()` is idempotent for
    # already-resident Reactant arrays used during training validation.
    dev = _device()
    input = dev(input)
    ps = dev(ps)
    st = dev(st)
    key = (
        :prediction, typeof(model), _shape_signature(input),
        _shape_signature(ps), _shape_signature(st), string(pkgversion(Reactant)),
    )
    compiled = lock(_COMPILATION_CACHE_LOCK) do
        get(_PREDICTION_CACHE, key, nothing)
    end
    if compiled === nothing
        Threads.atomic_add!(_COMPILATION_CACHE_MISSES, 1)
        compiled = Reactant.@compile Lux.apply(model, input, ps, st)
        lock(_COMPILATION_CACHE_LOCK) do
            _PREDICTION_CACHE[key] = compiled
        end
    else
        Threads.atomic_add!(_COMPILATION_CACHE_HITS, 1)
    end
    prediction, _ = compiled(model, input, ps, st)
    Float64.(Lux.cpu_device()(prediction))
end

JoptunaLearners._backend_to_host(::JoptunaLearners.ReactantCPU, value) = Lux.cpu_device()(value)
JoptunaLearners._backend_synchronize(::JoptunaLearners.ReactantCPU, value=nothing) =
    Reactant.synchronize(value)

function JoptunaLearners._backend_provenance(backend::JoptunaLearners.ReactantCPU)
    (; execution_backend="reactant_cpu", device="cpu", compiled=true,
       ad="Enzyme", reactant=string(pkgversion(Reactant)),
       enzyme=string(pkgversion(Enzyme)), synchronization=String(backend.synchronization),
       compilation_cache_hits=_COMPILATION_CACHE_HITS[],
       compilation_cache_misses=_COMPILATION_CACHE_MISSES[],
       fallback="error")
end

end
