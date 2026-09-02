_randn32(rng::AbstractRNG, dims...) = Float32.(0.02 .* randn(rng, dims...))
_zeros32(dims...) = zeros(Float32, dims...)
_dense(x, p) = p.weight * x .+ p.bias
_head(rng, n) = (weight=_randn32(rng, 1, n), bias=_zeros32(1, 1))

function _mlp_params(rng, input::Int, widths)
    names = ntuple(i -> Symbol(:layer, i), length(widths))
    values = map(widths) do width
        p = (weight=_randn32(rng, width, input), bias=_zeros32(width, 1))
        input = width
        p
    end
    return NamedTuple{names}(Tuple(values)), input
end

function _mlp(x, ps)
    foldl(values(ps); init=x) do h, p
        NNlib.gelu.(_dense(h, p))
    end
end

function _layernorm(x; eps=1.0f-5)
    μ = mean(x; dims=1)
    σ² = mean((x .- μ) .^ 2; dims=1)
    (x .- μ) ./ sqrt.(σ² .+ eps)
end

function _revin(x; eps=1.0f-5)
    μ = mean(x; dims=2)
    σ² = mean((x .- μ) .^ 2; dims=2)
    (x .- μ) ./ sqrt.(σ² .+ eps)
end

function _moving_average_cpu!(trend::Array{T,3}, windows::Array{T,3},
                              kernel::Int) where {T}
    size(trend) == size(windows) || throw(DimensionMismatch(
        "moving-average scratch shape $(size(trend)) does not match $(size(windows))",
    ))
    radius = kernel ÷ 2
    features, width, observations = size(windows)
    @inbounds for observation in 1:observations, position in 1:width,
                  feature in 1:features
        total = zero(T)
        for offset in -radius:radius
            source = clamp(position + offset, 1, width)
            total += windows[feature, source, observation]
        end
        trend[feature, position, observation] = total / kernel
    end
    trend
end

function _moving_average_cpu(windows::Array{T,3}, kernel::Int) where {T}
    _moving_average_cpu!(similar(windows), windows, kernel)
end

Zygote.@adjoint function _moving_average_cpu(windows::Array{T,3}, kernel::Int) where {T}
    trend = _moving_average_cpu(windows, kernel)
    function pullback(tangent)
        radius = kernel ÷ 2
        features, width, observations = size(windows)
        gradient = zeros(T, size(windows))
        scale = inv(T(kernel))
        @inbounds for observation in 1:observations, position in 1:width,
                      feature in 1:features
            contribution = tangent[feature, position, observation] * scale
            for offset in -radius:radius
                source = clamp(position + offset, 1, width)
                gradient[feature, source, observation] += contribution
            end
        end
        return gradient, nothing
    end
    return trend, pullback
end

function _moving_average_functional(windows, kernel::Int)
    radius = kernel ÷ 2
    padded = cat(
        repeat(windows[:, 1:1, :], 1, radius, 1),
        windows,
        repeat(windows[:, end:end, :], 1, radius, 1);
        dims=2,
    )
    # `ntuple` fixes the reduction structure at model-construction time. Besides
    # avoiding one allocation per time position, this is traceable by Reactant:
    # no Julia iteration depends on a traced tensor dimension.
    width = size(windows, 2)
    reduce(+, ntuple(kernel) do offset
        @view padded[:, offset:(offset + width - 1), :]
    end) ./ kernel
end

_moving_average(windows::Array{<:Any,3}, kernel::Int) =
    _moving_average_cpu(windows, kernel)

_moving_average(windows, kernel::Int) =
    _moving_average_functional(windows, kernel)

# These primitives deliberately use feature-first tensors `(channels, time, batch)`, the
# native counterpart of common batch-first tensor models. Keeping them functional is
# important: Zygote can differentiate the same paths used by qualification and training.
function _affine_layernorm(x, gamma, beta; eps=1f-5)
    μ = mean(x; dims=1)
    σ² = mean((x .- μ) .^ 2; dims=1)
    shape = (length(gamma), ntuple(_ -> 1, ndims(x) - 1)...)
    ((x .- μ) ./ sqrt.(σ² .+ eps)) .* reshape(gamma, shape) .+ reshape(beta, shape)
end

function _causal_conv1d(x, weight, bias, dilation::Int;
                        normalize_weight::Bool=false, time::Int=size(x, 2))
    out_channels, _, kernel = size(weight)
    w = normalize_weight ? weight ./ sqrt.(sum(abs2, weight; dims=(2, 3)) .+ 1f-8) : weight
    native = NNlib.conv(
        permutedims(x, (2, 1, 3)),
        permutedims(w, (3, 2, 1));
        pad=((kernel - 1) * dilation, 0), dilation, flipped=true,
    )
    permutedims(native, (2, 1, 3)) .+ reshape(bias, out_channels, 1, 1)
end

function _depthwise_causal_conv1d(x, weight, bias; time::Int=size(x, 2))
    channels = size(x, 1)
    kernel = size(weight, 2)
    kernel_weights = reshape(permutedims(weight, (2, 1)), kernel, 1, channels)
    native = NNlib.depthwiseconv(
        permutedims(x, (2, 1, 3)), kernel_weights;
        pad=(kernel - 1, 0), flipped=true,
    )
    permutedims(native, (2, 1, 3)) .+ reshape(bias, channels, 1, 1)
end

_sequence_linear(x, p) = reshape(_dense(reshape(x, size(x, 1), :), p), size(p.weight, 1), size(x, 2), size(x, 3))

function _multihead_attention(query, source, p, n_heads::Int;
                              n_query::Int=size(query, 2),
                              n_source::Int=size(source, 2))
    hidden = size(query, 1)
    hidden % n_heads == 0 || throw(ArgumentError("hidden_dim must be divisible by n_heads"))
    width = hidden ÷ n_heads
    q, k, v = _sequence_linear(query, p.q), _sequence_linear(source, p.k), _sequence_linear(source, p.v)
    per_head = ntuple(n_heads) do head
        rows = (head - 1) * width + 1:head * width
        qh, kh, vh = q[rows, :, :], k[rows, :, :], v[rows, :, :]
        contexts = ntuple(n_query) do position
            score = dropdims(sum(reshape(qh[:, position, :], width, 1, :) .* kh; dims=1); dims=1) ./ sqrt(Float32(width))
            weights = NNlib.softmax(score; dims=1)
            reshape(dropdims(sum(vh .* reshape(weights, 1, n_source, :); dims=2); dims=2), width, 1, :)
        end
        cat(contexts...; dims=2)
    end
    _sequence_linear(cat(per_head...; dims=1), p.out)
end

function _transformer_block(x, p, n_heads, n_tokens)
    h = _affine_layernorm(x, p.norm1_gamma, p.norm1_beta)
    x = x .+ _multihead_attention(h, h, p.attention, n_heads;
                                  n_query=n_tokens, n_source=n_tokens)
    h = _affine_layernorm(x, p.norm2_gamma, p.norm2_beta)
    ff = _sequence_linear(NNlib.gelu.(_sequence_linear(h, p.ff1)), p.ff2)
    x .+ ff
end

function _grn(x, p)
    linear = ndims(x) == 2 ? _dense : _sequence_linear
    candidate = linear(NNlib.elu.(linear(x, p.w1)), p.w2)
    gated = NNlib.sigmoid.(linear(x, p.gate)) .* candidate
    _affine_layernorm(x .+ gated, p.norm_gamma, p.norm_beta)
end

function _lstm_layer(x, p; time::Int=size(x, 2))
    hidden = size(x, 1)
    initial_state = zero(@view x[:, 1, :])
    initial = (initial_state, initial_state, ())
    _, _, outputs = foldl(ntuple(identity, time); init=initial) do carry, position
        h, c, history = carry
        gates = _dense(vcat(@view(x[:, position, :]), h), p)
        i = NNlib.sigmoid.(@view gates[1:hidden, :])
        f = NNlib.sigmoid.(@view gates[hidden+1:2hidden, :])
        g = tanh.(@view gates[2hidden+1:3hidden, :])
        o = NNlib.sigmoid.(@view gates[3hidden+1:4hidden, :])
        c = f .* c .+ i .* g
        h = o .* tanh.(c)
        (h, c, (history..., reshape(h, hidden, 1, :)))
    end
    cat(outputs...; dims=2)
end

function _mamba_states(u, decay, time::Int)
    hidden = size(u, 1)
    initial = (zero(@view u[:, 1, :]), ())
    _, history = foldl(ntuple(identity, time); init=initial) do carry, position
        state, outputs = carry
        next = reshape(decay, hidden, 1) .* state .+
               (1f0 .- reshape(decay, hidden, 1)) .* @view(u[:, position, :])
        (next, (outputs..., reshape(next, hidden, 1, :)))
    end
    history
end

function _mamba_block(x, p; time::Int=size(x, 2))
    residual = x
    h = _affine_layernorm(x, p.norm_gamma, p.norm_beta)
    projected = _sequence_linear(h, p.in_proj)
    hidden = size(x, 1)
    u, gate = projected[1:hidden, :, :], projected[hidden+1:2hidden, :, :]
    u = _depthwise_causal_conv1d(u, p.conv_weight, p.conv_bias; time)
    decay = NNlib.sigmoid.(p.decay_logit)
    states = _mamba_states(u, decay, time)
    y = _sequence_linear(NNlib.sigmoid.(gate) .* cat(states...; dims=2), p.out_proj)
    residual .+ y
end

"""
One native Lux layer family implementing the published JoptunaLearners architecture contracts.
The layer type is immutable; all trainable state lives in the Lux parameter tree.
"""
struct NativeArchitecture{C<:NamedTuple} <: Lux.AbstractLuxLayer
    name::Symbol
    n_features::Int
    lookback::Int
    n_entities::Int
    config::C
end

Lux.initialstates(::AbstractRNG, ::NativeArchitecture) = NamedTuple()

function Lux.initialparameters(rng::AbstractRNG, m::NativeArchitecture)
    n, f, l, c = m.name, m.n_features, m.lookback, m.config
    if n in (:mlp, :window_mlp)
        widths = Tuple(c.hidden_dims)
        body, out = _mlp_params(rng, n === :mlp ? f : f*l, widths)
        return (; body, head=_head(rng, out))
    elseif n === :window_linear
        return (head=_head(rng, f*l),)
    elseif n === :window_nlinear
        return (delta=_head(rng, f*l), anchor=_head(rng, f))
    elseif n === :window_dlinear
        return (trend=_head(rng, f*l), remainder=_head(rng, f*l))
    elseif n === :tabular_resnet
        input = (weight=_randn32(rng, c.block_width, f), bias=_zeros32(c.block_width,1))
        blocks = NamedTuple{ntuple(i -> Symbol(:block, i), c.n_blocks)}(ntuple(c.n_blocks) do _
            inner = max(1, round(Int, c.block_width*c.hidden_factor))
            (w1=_randn32(rng, inner, c.block_width), b1=_zeros32(inner,1),
             w2=_randn32(rng, c.block_width, inner), b2=_zeros32(c.block_width,1))
        end)
        return (; input, blocks, head=_head(rng, c.block_width))
    elseif n in (:tabular_mixer, :tsmixer, :film)
        time = n === :tabular_mixer ? 1 : l
        blocks = NamedTuple{ntuple(i -> Symbol(:block, i), c.depth)}(ntuple(c.depth) do _
            (tw1=_randn32(rng,c.token_dim,time), tb1=_zeros32(c.token_dim,1),
             tw2=_randn32(rng,time,c.token_dim), tb2=_zeros32(time,1),
             cw1=_randn32(rng,c.channel_dim,f), cb1=_zeros32(c.channel_dim,1),
             cw2=_randn32(rng,f,c.channel_dim), cb2=_zeros32(f,1))
        end)
        proj=(weight=_randn32(rng,c.hidden_dim,f), bias=_zeros32(c.hidden_dim,1))
        if n === :film
            return (; blocks, proj, embedding=_randn32(rng,c.emb_dim,m.n_entities),
                    gamma=(weight=_randn32(rng,c.hidden_dim,c.emb_dim),bias=_zeros32(c.hidden_dim,1)),
                    beta=(weight=_randn32(rng,c.hidden_dim,c.emb_dim),bias=_zeros32(c.hidden_dim,1)),
                    head=_head(rng,c.hidden_dim))
        end
        return (; blocks, proj, head=_head(rng,c.hidden_dim))
    elseif n in (:tcn, :tcn_v2)
        input = n === :tcn ?
            (weight=_randn32(rng, c.hidden_dim, f), bias=_zeros32(c.hidden_dim, 1)) :
            (weight=_randn32(rng, c.hidden_dim, f), bias=_zeros32(c.hidden_dim, 1),
             norm_gamma=ones(Float32, f), norm_beta=zeros(Float32, f))
        blocks = NamedTuple{ntuple(i -> Symbol(:block, i), c.depth)}(ntuple(c.depth) do _
            (conv1_weight=_randn32(rng, c.hidden_dim, c.hidden_dim, c.kernel_size),
             conv1_bias=_zeros32(c.hidden_dim, 1),
             conv2_weight=_randn32(rng, c.hidden_dim, c.hidden_dim, c.kernel_size),
             conv2_bias=_zeros32(c.hidden_dim, 1))
        end)
        return (; input, blocks, head=_head(rng, c.hidden_dim))
    elseif n === :mamba
        blocks = NamedTuple{ntuple(i -> Symbol(:block, i), c.depth)}(ntuple(c.depth) do _
            (norm_gamma=ones(Float32, c.hidden_dim), norm_beta=zeros(Float32, c.hidden_dim),
             in_proj=(weight=_randn32(rng, 2 * c.hidden_dim, c.hidden_dim), bias=_zeros32(2 * c.hidden_dim, 1)),
             conv_weight=_randn32(rng, c.hidden_dim, c.kernel_size), conv_bias=_zeros32(c.hidden_dim, 1),
             decay_logit=zeros(Float32, c.hidden_dim),
             out_proj=(weight=_randn32(rng, c.hidden_dim, c.hidden_dim), bias=_zeros32(c.hidden_dim, 1)))
        end)
        return (input=(weight=_randn32(rng, c.hidden_dim, f), bias=_zeros32(c.hidden_dim, 1)),
                blocks=blocks, norm_gamma=ones(Float32, c.hidden_dim), norm_beta=zeros(Float32, c.hidden_dim),
                head=_head(rng, c.hidden_dim))
    elseif n === :patchtst
        patch_dim = f*c.patch_len
        blocks = NamedTuple{ntuple(i -> Symbol(:block, i), c.depth)}(ntuple(c.depth) do _
            (norm1_gamma=ones(Float32, c.hidden_dim), norm1_beta=zeros(Float32, c.hidden_dim),
             attention=(q=(weight=_randn32(rng, c.hidden_dim, c.hidden_dim), bias=_zeros32(c.hidden_dim, 1)),
                        k=(weight=_randn32(rng, c.hidden_dim, c.hidden_dim), bias=_zeros32(c.hidden_dim, 1)),
                        v=(weight=_randn32(rng, c.hidden_dim, c.hidden_dim), bias=_zeros32(c.hidden_dim, 1)),
                        out=(weight=_randn32(rng, c.hidden_dim, c.hidden_dim), bias=_zeros32(c.hidden_dim, 1))),
             norm2_gamma=ones(Float32, c.hidden_dim), norm2_beta=zeros(Float32, c.hidden_dim),
             ff1=(weight=_randn32(rng, 4 * c.hidden_dim, c.hidden_dim), bias=_zeros32(4 * c.hidden_dim, 1)),
             ff2=(weight=_randn32(rng, c.hidden_dim, 4 * c.hidden_dim), bias=_zeros32(c.hidden_dim, 1)))
        end)
        return (patch=(weight=_randn32(rng,c.hidden_dim,patch_dim),bias=_zeros32(c.hidden_dim,1)),
                positional=_randn32(rng, c.hidden_dim, max(1, 1 + max(0, (l - min(c.patch_len, l)) ÷ c.stride))),
                blocks=blocks, norm_gamma=ones(Float32, c.hidden_dim), norm_beta=zeros(Float32, c.hidden_dim),
                head=_head(rng,c.hidden_dim))
    elseif n === :tft
        grn = () -> (w1=(weight=_randn32(rng, 2 * c.hidden_dim, c.hidden_dim), bias=_zeros32(2 * c.hidden_dim, 1)),
                     w2=(weight=_randn32(rng, c.hidden_dim, 2 * c.hidden_dim), bias=_zeros32(c.hidden_dim, 1)),
                     gate=(weight=_randn32(rng, c.hidden_dim, c.hidden_dim), bias=_zeros32(c.hidden_dim, 1)),
                     norm_gamma=ones(Float32, c.hidden_dim), norm_beta=zeros(Float32, c.hidden_dim))
        layers = NamedTuple{ntuple(i -> Symbol(:layer, i), c.recurrent_layers)}(ntuple(c.recurrent_layers) do i
            in_dim = i == 1 ? c.hidden_dim : c.hidden_dim
            (weight=_randn32(rng, 4 * c.hidden_dim, in_dim + c.hidden_dim), bias=_zeros32(4 * c.hidden_dim, 1))
        end)
        return (input=(weight=_randn32(rng,c.hidden_dim,f),bias=_zeros32(c.hidden_dim,1)),
                feature_grn=grn(), lstm=layers,
                attention=(q=(weight=_randn32(rng,c.hidden_dim,c.hidden_dim),bias=_zeros32(c.hidden_dim,1)),
                           k=(weight=_randn32(rng,c.hidden_dim,c.hidden_dim),bias=_zeros32(c.hidden_dim,1)),
                           v=(weight=_randn32(rng,c.hidden_dim,c.hidden_dim),bias=_zeros32(c.hidden_dim,1)),
                           out=(weight=_randn32(rng,c.hidden_dim,c.hidden_dim),bias=_zeros32(c.hidden_dim,1))),
                post_grn=grn(), head=_head(rng,c.hidden_dim))
    end
    throw(ArgumentError("unimplemented architecture $(m.name)"))
end

function _mixer_encode(x, blocks, proj)
    f,t,b = size(x)
    for p in values(blocks)
        y = _layernorm(x)
        token = reshape(permutedims(y,(2,1,3)),t,f*b)
        token = p.tw2 * NNlib.gelu.(p.tw1*token .+ p.tb1) .+ p.tb2
        x = x .+ permutedims(reshape(token,t,f,b),(2,1,3))
        y = _layernorm(x)
        channel = reshape(y,f,t*b)
        channel = p.cw2 * NNlib.gelu.(p.cw1*channel .+ p.cb1) .+ p.cb2
        x = x .+ reshape(channel,f,t,b)
    end
    pooled = dropdims(mean(x; dims=2); dims=2)
    NNlib.gelu.(_dense(pooled,proj))
end

function _tcn_encode(m, windows, ps)
    x = get(m.config, :revin, false) ? _revin(windows) : windows
    if m.name === :tcn_v2
        x = _affine_layernorm(x, ps.input.norm_gamma, ps.input.norm_beta)
    end
    x = _sequence_linear(x, (weight=ps.input.weight, bias=ps.input.bias))
    for (index, block) in enumerate(values(ps.blocks))
        dilation = 2 ^ (index - 1)
        residual = x
        x = NNlib.gelu.(_causal_conv1d(x, block.conv1_weight, block.conv1_bias, dilation;
            normalize_weight=m.config.use_weight_norm, time=m.lookback))
        x = NNlib.gelu.(_causal_conv1d(x, block.conv2_weight, block.conv2_bias, dilation;
            normalize_weight=m.config.use_weight_norm, time=m.lookback))
        x = residual .+ x
    end
    x[:, end, :]
end

function _patches(w, patch_len, stride, lookback)
    starts = Tuple(1:stride:max(1, lookback - patch_len + 1))
    map(starts) do s
        block=w[:, s:(s + patch_len - 1), :]
        reshape(block,size(block,1)*patch_len,size(block,3))
    end
end

function (m::NativeArchitecture)(input, ps, st::NamedTuple)
    n = m.name
    if n === :mlp
        h=_mlp(input,ps.body); return vec(_dense(h,ps.head)),st
    elseif n === :window_mlp
        h=_mlp(reshape(input,m.n_features*m.lookback,size(input,3)),ps.body); return vec(_dense(h,ps.head)),st
    elseif n === :window_linear
        return vec(_dense(reshape(input,m.n_features*m.lookback,size(input,3)),ps.head)),st
    elseif n === :window_nlinear
        last=input[:,end,:]; centered=input .- reshape(last,m.n_features,1,size(input,3))
        y=_dense(reshape(centered,m.n_features*m.lookback,size(input,3)),ps.delta) .+ _dense(last,ps.anchor)
        return vec(y),st
    elseif n === :window_dlinear
        trend=_moving_average(input,m.config.kernel_size); remainder=input.-trend
        y=_dense(reshape(trend,m.n_features*m.lookback,size(input,3)),ps.trend) .+
          _dense(reshape(remainder,m.n_features*m.lookback,size(input,3)),ps.remainder)
        return vec(y),st
    elseif n === :tabular_resnet
        h=NNlib.gelu.(_dense(input,ps.input))
        for p in values(ps.blocks)
            z=_layernorm(h); z=NNlib.gelu.(p.w1*z .+ p.b1); z=p.w2*z .+ p.b2; h=h.+z
        end
        return vec(_dense(h,ps.head)),st
    elseif n in (:tabular_mixer,:tsmixer,:film)
        windows=n === :film ? input.windows : n === :tabular_mixer ? reshape(input,m.n_features,1,size(input,2)) : input
        if get(m.config,:revin,false) && size(windows,2)>1; windows=_revin(windows); end
        h=_mixer_encode(windows,ps.blocks,ps.proj)
        if n === :film
            e=ps.embedding[:,input.entity_codes]; h=(1 .+ _dense(e,ps.gamma)).*h .+ _dense(e,ps.beta)
        end
        return vec(_dense(h,ps.head)),st
    elseif n in (:tcn,:tcn_v2)
        h = _tcn_encode(m, input, ps)
        return vec(_dense(h, ps.head)), st
    elseif n === :patchtst
        embeds=map(_patches(input,m.config.patch_len,m.config.stride,m.lookback)) do p
            NNlib.gelu.(_dense(p,ps.patch))
        end
        reshaped=map(e->reshape(e,size(e,1),1,size(e,2)),embeds)
        tokens=cat(reshaped...;dims=2) .+ reshape(ps.positional[:, 1:length(reshaped)],
            size(ps.positional, 1), length(reshaped), 1)
        for block in values(ps.blocks)
            tokens = _transformer_block(tokens, block, m.config.n_heads, length(reshaped))
        end
        pooled = dropdims(mean(tokens; dims=2); dims=2)
        h = _affine_layernorm(reshape(pooled, size(pooled, 1), 1, size(pooled, 2)),
            ps.norm_gamma, ps.norm_beta)[:, 1, :]
        return vec(_dense(h, ps.head)), st
    elseif n === :tft
        h = _grn(_sequence_linear(input, ps.input), ps.feature_grn)
        for layer in values(ps.lstm)
            h = _lstm_layer(h, layer; time=m.lookback)
        end
        query = h[:, end:end, :]
        attended = _multihead_attention(query, h, ps.attention, m.config.n_heads;
                                        n_query=1, n_source=m.lookback)
        h = _grn(attended[:, 1, :], ps.post_grn)
        return vec(_dense(h, ps.head)), st
    elseif n === :mamba
        h = _sequence_linear(input, ps.input)
        for block in values(ps.blocks)
            h = _mamba_block(h, block; time=m.lookback)
        end
        h = _affine_layernorm(h[:, end:end, :], ps.norm_gamma, ps.norm_beta)[:, 1, :]
        return vec(_dense(h, ps.head)), st
    end
    throw(ArgumentError("unimplemented architecture $n"))
end

function build_model(spec::ModelSpec; n_features::Int, lookback::Int=1, n_entities::Int=0, kwargs...)
    n_features > 0 || throw(ArgumentError("n_features must be positive"))
    lookback > 0 || throw(ArgumentError("lookback must be positive"))
    spec.uses_entity && n_entities <= 0 && throw(ArgumentError("$(spec.public_name) requires n_entities > 0"))
    config=model_config(spec.name;kwargs...)
    if spec.name === :window_dlinear
        isodd(config.kernel_size) || throw(ArgumentError("DLinear kernel_size must be odd"))
        config.kernel_size <= lookback || throw(ArgumentError("DLinear kernel_size cannot exceed lookback"))
    end
    if spec.name === :patchtst
        config.hidden_dim % config.n_heads == 0 || throw(ArgumentError(
            "PatchTSTLite hidden_dim must be divisible by n_heads"))
        config.patch_len > 0 && config.stride > 0 || throw(ArgumentError(
            "PatchTSTLite patch_len and stride must be positive"))
        # Match the Python constructor's `min(patch_len, lookback)` rule before
        # parameter initialization, so a short input produces one padded patch.
        config = merge(config, (patch_len=min(config.patch_len, lookback),))
    elseif spec.name === :tft
        config.hidden_dim % config.n_heads == 0 || throw(ArgumentError(
            "TFTLite hidden_dim must be divisible by n_heads"))
    elseif spec.name === :mamba
        config.kernel_size > 0 || throw(ArgumentError("MambaLite kernel_size must be positive"))
    end
    if spec.name in (:tcn,:tcn_v2) && config.auto_depth
        rf=1+2*(config.kernel_size-1)*(2^config.depth-1)
        depth=config.depth
        while rf<lookback
            depth+=1; rf=1+2*(config.kernel_size-1)*(2^depth-1)
        end
        config=merge(config,(depth=depth,))
    elseif spec.name in (:tcn, :tcn_v2)
        rf = 1 + 2 * (config.kernel_size - 1) * (2 ^ config.depth - 1)
        rf >= lookback || throw(ArgumentError(
            "$(spec.public_name) receptive field $rf < lookback $lookback; raise depth or set auto_depth=true"))
    end
    NativeArchitecture(spec.name,n_features,lookback,n_entities,config)
end

build_model(name::Union{Symbol,AbstractString};kwargs...)=build_model(model_spec(name);kwargs...)
