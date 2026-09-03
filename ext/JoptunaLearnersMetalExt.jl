module JoptunaLearnersMetalExt

using ADTypes
using JoptunaLearners
using Lux
using Metal

_device() = Lux.gpu_device(Float32; force=true)

# NNlib's generic reverse rule for one-dimensional convolutions currently reaches
# an im2col implementation that performs scalar indexing on MtlArray.  Keep the
# mathematical operation on the GPU, but express the small causal kernels as a
# fixed collection of matrix multiplications.  The loop bounds are model-shape
# constants and Zygote differentiates the MtlArray GEMMs without host fallback.
function JoptunaLearners._causal_conv1d(x::Metal.MtlArray, weight, bias,
                                    dilation::Int;
                                    normalize_weight::Bool=false,
                                    time::Int=size(x, 2))
    out_channels, in_channels, kernel = size(weight)
    batch = size(x, 3)
    w = normalize_weight ?
        weight ./ sqrt.(sum(abs2, weight; dims=(2, 3)) .+ 1f-8) : weight
    slices = ntuple(time) do destination
        value = repeat(reshape(bias, out_channels, 1), 1, batch)
        for tap in 1:kernel
            source = destination - (kernel - tap) * dilation
            if source >= 1
                values = copy(reshape(@view(x[:, source:source, :]),
                                      in_channels, batch))
                kernel_matrix = copy(reshape(@view(w[:, :, tap:tap]),
                                             out_channels, in_channels))
                value = value .+ kernel_matrix * values
            end
        end
        reshape(value, out_channels, 1, batch)
    end
    cat(slices...; dims=2)
end

function JoptunaLearners._depthwise_causal_conv1d(x::Metal.MtlArray, weight, bias;
                                               time::Int=size(x, 2))
    channels = size(x, 1)
    kernel = size(weight, 2)
    batch = size(x, 3)
    slices = ntuple(time) do destination
        value = repeat(reshape(bias, channels, 1), 1, batch)
        for tap in 1:kernel
            source = destination - (kernel - tap)
            if source >= 1
                values = copy(reshape(@view(x[:, source:source, :]), channels, batch))
                coefficients = copy(reshape(@view(weight[:, tap:tap]), channels, 1))
                value = value .+ coefficients .* values
            end
        end
        reshape(value, channels, 1, batch)
    end
    cat(slices...; dims=2)
end

function JoptunaLearners.backend_capabilities(backend::JoptunaLearners.MetalGPU)
    base = invoke(JoptunaLearners.backend_capabilities,
        Tuple{JoptunaLearners.ExecutionBackend}, backend)
    merge(base, (; available=Metal.functional(), runtime="Metal", ad="Zygote"))
end

function JoptunaLearners._backend_setup(::JoptunaLearners.MetalGPU, model, ps, st, optimizer,
                                    gradient_clip)
    dev = _device()
    Lux.Training.TrainState(model, dev(ps), dev(st), optimizer)
end

function JoptunaLearners._backend_restore(::JoptunaLearners.MetalGPU, model, ps, st,
                                      optimizer, optimizer_state, update, gradient_clip)
    dev = _device()
    fresh = Lux.Training.TrainState(model, dev(ps), dev(st), optimizer)
    Lux.Training.TrainState(nothing, nothing, fresh.allocator_cache, model,
        fresh.parameters, fresh.states, fresh.optimizer, dev(optimizer_state), update)
end

JoptunaLearners._backend_batch(::JoptunaLearners.MetalGPU, batch) = _device()(batch)

_refresh_batch!(destination::Metal.MtlArray, source::Array) =
    copyto!(destination, source)
_refresh_batch!(destination::NamedTuple, source::NamedTuple) =
    map(_refresh_batch!, destination, source)
_refresh_batch!(destination::Tuple, source::Tuple) =
    map(_refresh_batch!, destination, source)
_refresh_batch!(destination::Number, source::Number) = source

function JoptunaLearners._backend_batch(::JoptunaLearners.MetalGPU, batch,
                                    stager::JoptunaLearners._BatchStager)
    key = (:metal_batch, length(batch[2]))
    device_batch = get!(stager.workspaces, key) do
        _device()(batch)
    end
    _refresh_batch!(device_batch, batch)
end

# NNlib's batched MPS path creates autoreleased native command objects. Bound
# their lifetime outside the AD region, including when training raises an error.
# Julia GC or Lux allocation-cache cleanup cannot drain native autorelease pools.
Metal.@autoreleasepool function JoptunaLearners._backend_train_batch(::JoptunaLearners.MetalGPU,
                                          objective, batch, ts, gradient_clip)
    if ts.model isa JoptunaLearners.NativeArchitecture &&
       ts.model.name === :window_dlinear
        gs, loss = JoptunaLearners._dlinear_accelerator_gradients(
            ts.model, ts.parameters, batch,
        )
        gs = JoptunaLearners._clip_gradients(gs, gradient_clip)
        ts = Lux.Training.apply_gradients!(ts, gs)
        return loss, ts
    end
    gs, loss, _, ts = Lux.Training.compute_gradients(
        ADTypes.AutoZygote(), objective, batch, ts,
    )
    gs = JoptunaLearners._clip_gradients(gs, gradient_clip)
    ts = Lux.Training.apply_gradients!(ts, gs)
    loss, ts
end

JoptunaLearners._backend_accumulate_loss(::JoptunaLearners.MetalGPU, accumulated, loss,
                                     observations) =
    accumulated === nothing ? loss .* Float32(observations) :
    accumulated .+ loss .* Float32(observations)

function JoptunaLearners._backend_finalize_loss(::JoptunaLearners.MetalGPU, accumulated)
    Metal.synchronize()
    host = Lux.cpu_device()(accumulated)
    Float64(host isa Number ? host : only(host))
end

JoptunaLearners._backend_times_each_update(::JoptunaLearners.MetalGPU) = false

Metal.@autoreleasepool function JoptunaLearners._backend_predict(::JoptunaLearners.MetalGPU, model, ps, st, input)
    dev = _device()
    prediction, _ = model(dev(input), dev(ps), dev(st))
    Metal.synchronize()
    Float64.(Lux.cpu_device()(prediction))
end

JoptunaLearners._backend_to_host(::JoptunaLearners.MetalGPU, value) = Lux.cpu_device()(value)
JoptunaLearners._backend_synchronize(::JoptunaLearners.MetalGPU, value=nothing) = Metal.synchronize()

function JoptunaLearners._backend_provenance(::JoptunaLearners.MetalGPU)
    (; execution_backend="metal_gpu", device="metal", compiled=false, ad="Zygote",
       metal=string(pkgversion(Metal)), precision="Float32", fallback="error")
end

end
