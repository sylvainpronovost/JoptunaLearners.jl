"""Train-only feature normalization parameters."""
struct FeatureNormalizer{T<:AbstractFloat}
    mean::Vector{T}
    scale::Vector{T}
end

function fit_normalizer(x::AbstractMatrix)
    μ = vec(mean(x; dims=2))
    σ = vec(std(x; dims=2, corrected=false))
    σ = ifelse.(isfinite.(σ) .& (σ .> eps(eltype(σ))), σ, one(eltype(σ)))
    FeatureNormalizer(Float32.(μ), Float32.(σ))
end

function apply_normalizer(n::FeatureNormalizer, x::AbstractMatrix)
    size(x, 1) == length(n.mean) || throw(DimensionMismatch("feature count does not match normalizer"))
    Float32.((x .- n.mean) ./ n.scale)
end

"""Lazy causal windows backed by one feature matrix and per-observation start rows."""
struct CausalWindowSource <: AbstractArray{Float32,3}
    values::Matrix{Float32}
    starts::Vector{Int}
    lookback::Int
    function CausalWindowSource(values::AbstractMatrix, starts, lookback::Integer)
        lookback > 0 || throw(ArgumentError("lookback must be positive"))
        # Preserve an already canonical matrix. Lazy train/evaluation subsets
        # share this immutable feature surface and differ only in their start
        # indices; copying the whole surface per subset defeats lazy windows.
        converted = values isa Matrix{Float32} ? values : Matrix{Float32}(values)
        all(isfinite, converted) || throw(ArgumentError("window features must be finite"))
        locations = Int.(starts)
        all(start -> 1 <= start <= size(converted, 2) - lookback + 1, locations) ||
            throw(ArgumentError("window start lies outside the feature surface"))
        new(converted, locations, Int(lookback))
    end
end

Base.size(source::CausalWindowSource) =
    (size(source.values, 1), source.lookback, length(source.starts))
Base.IndexStyle(::Type{CausalWindowSource}) = IndexCartesian()
@inline Base.getindex(source::CausalWindowSource, feature::Int, offset::Int, observation::Int) =
    source.values[feature, source.starts[observation] + offset - 1]
Base.getindex(source::CausalWindowSource, ::Colon, ::Colon,
              observations::AbstractVector{<:Integer}) =
    CausalWindowSource(source.values, source.starts[observations], source.lookback)
Base.getindex(source::CausalWindowSource, ::Colon, ::Colon, ::Colon) = source

"""
Canonical learner data. Features use Lux's feature-first convention. Windows either
materialize `(features, lookback, observations)` or use `CausalWindowSource`.
"""
struct LearnerData{K<:AbstractDataFrame,W,S}
    tabular::Matrix{Float32}
    windows::W
    entity_codes::S
    target::Vector{Float32}
    weights::Vector{Float32}
    keys::K
end

function LearnerData(tabular::AbstractMatrix, target::AbstractVector;
                     windows=nothing, entity_codes=nothing, weights=ones(length(target)),
                     keys=DataFrame(row_id=collect(1:length(target))))
    n = length(target)
    size(tabular, 2) == n || throw(DimensionMismatch("tabular observations and target differ"))
    windows === nothing || size(windows, 3) == n || throw(DimensionMismatch("window observations and target differ"))
    entity_codes === nothing || length(entity_codes) == n || throw(DimensionMismatch("entity codes and target differ"))
    length(weights) == n || throw(DimensionMismatch("weights and target differ"))
    nrow(keys) == n || throw(DimensionMismatch("keys and target differ"))
    all(isfinite, tabular) || throw(ArgumentError("tabular features must be finite"))
    windows === nothing || windows isa CausalWindowSource || all(isfinite, windows) ||
        throw(ArgumentError("windows must be finite"))
    all(isfinite, target) || throw(ArgumentError("targets must be finite"))
    all(w -> isfinite(w) && w >= 0, weights) || throw(ArgumentError("weights must be finite and nonnegative"))
    stored_windows = windows === nothing || windows isa CausalWindowSource ? windows :
        Array{Float32,3}(windows)
    stored_tabular = tabular isa Matrix{Float32} ? tabular : Matrix{Float32}(tabular)
    stored_entity_codes = entity_codes === nothing ? nothing :
        (entity_codes isa Vector{Int} ? entity_codes : Int.(entity_codes))
    stored_target = target isa Vector{Float32} ? target : Float32.(target)
    stored_weights = weights isa Vector{Float32} ? weights : Float32.(weights)
    stored_keys = keys isa DataFrame ? keys : DataFrame(keys; copycols=false)
    LearnerData(stored_tabular, stored_windows, stored_entity_codes, stored_target,
                stored_weights, stored_keys)
end

Base.length(d::LearnerData) = length(d.target)

function Base.getindex(d::LearnerData, idx::AbstractVector{<:Integer})
    LearnerData(d.tabular[:, idx], d.target[idx];
        windows=d.windows === nothing ? nothing : d.windows[:, :, idx],
        entity_codes=d.entity_codes === nothing ? nothing : d.entity_codes[idx],
        weights=d.weights[idx], keys=d.keys[idx, :])
end

Base.getindex(d::LearnerData, idx::Integer) = d[[idx]]
Base.getindex(d::LearnerData, ::Colon) = d

"""Convert row-major tabular values to feature-first Float32 data."""
prepare_tabular(x::AbstractMatrix) = permutedims(Float32.(x))

"""Training-fitted, ordered entity vocabulary. Unseen entities fail explicitly."""
struct EntityEncoder{T<:Tuple}
    vocabulary::T
    function EntityEncoder(values)
        vocabulary = Tuple(values)
        isempty(vocabulary) && throw(ArgumentError("entity vocabulary must not be empty"))
        any(ismissing, vocabulary) && throw(ArgumentError("entity vocabulary contains missing values"))
        allunique(vocabulary) || throw(ArgumentError("entity vocabulary contains duplicates"))
        new{typeof(vocabulary)}(vocabulary)
    end
end

fit_entity_encoder(values) = EntityEncoder(unique(values))

function encode_entities(encoder::EntityEncoder, values)
    mapping = Dict(value => index for (index, value) in enumerate(encoder.vocabulary))
    map(values) do value
        haskey(mapping, value) || throw(ArgumentError("unseen entity; reuse the training vocabulary and define new-entity handling upstream"))
        mapping[value]
    end
end

"""
Construct causal, entity-local windows. A row is retained only after `lookback`
observations exist for that entity; no future row can enter a window.
Omit `entity_encoder` only when fitting the training vocabulary. Pass the returned
encoder when preparing validation/inference subsets. Unknown entities raise an error.
"""
function prepare_windows(table; feature_cols, target_col::Symbol, entity_col::Symbol,
                         order_col::Symbol, lookback::Int, weight_col=nothing,
                         key_cols=(order_col, entity_col), materialization=:dense,
                         copycols::Bool=true, entity_encoder::Union{Nothing,EntityEncoder}=nothing)
    lookback > 0 || throw(ArgumentError("lookback must be positive"))
    features = Symbol.(feature_cols)
    keys = collect(Symbol.(key_cols))
    required = unique(vcat(features, [target_col, entity_col, order_col], keys,
                           weight_col === nothing ? Symbol[] : [Symbol(weight_col)]))
    source = DataFrame(table; copycols=false)
    missing_cols = setdiff(required, propertynames(source))
    isempty(missing_cols) || throw(ArgumentError("missing columns: $(join(missing_cols, ", "))"))
    # Window construction only needs this narrow surface. In particular, do not
    # duplicate every unrelated column from a wide research panel.
    # `copycols=false` is reserved for orchestration code that has constructed a
    # disposable, private frame. Sorting then reuses those owned columns instead
    # of materializing the complete multi-million-row surface a second time.
    df = DataFrames.select(source, required; copycols)
    sort!(df, [entity_col, order_col])

    groups = groupby(df, entity_col)
    observations = sum(max(0, nrow(group) - lookback + 1) for group in groups)
    observations > 0 || throw(ArgumentError("no complete causal windows were produced"))
    materialization = Symbol(materialization)
    materialization in (:dense, :lazy) || throw(ArgumentError(
        "materialization must be :dense or :lazy",
    ))
    windows = materialization == :dense ?
        Array{Float32,3}(undef, length(features), lookback, observations) : nothing
    window_starts = materialization == :lazy ? Vector{Int}(undef, observations) : Int[]
    rows = Vector{Int}(undef, observations)
    cursor = 1
    for group in groups
        parent_rows = parentindices(group)[1]
        values = materialization == :dense ? Matrix{Float32}(group[:, features]) : nothing
        for stop in lookback:nrow(group)
            start = stop - lookback + 1
            if materialization == :dense
                for offset in 1:lookback, feature in eachindex(features)
                    windows[feature, offset, cursor] = values[start + offset - 1, feature]
                end
            else
                window_starts[cursor] = parent_rows[start]
            end
            rows[cursor] = parent_rows[stop]
            cursor += 1
        end
    end
    selected = df[rows, :]
    tabular = prepare_tabular(Matrix(selected[:, features]))
    y = Float32.(selected[!, target_col])
    weights = weight_col === nothing ? ones(Float32, length(rows)) : Float32.(selected[!, weight_col])
    encoder = entity_encoder === nothing ? fit_entity_encoder(df[!, entity_col]) : entity_encoder
    entity_codes = encode_entities(encoder, selected[!, entity_col])
    if materialization == :lazy
        windows = CausalWindowSource(
            prepare_tabular(Matrix(df[:, features])), window_starts, lookback,
        )
    end
    data = LearnerData(tabular, y; windows, entity_codes, weights, keys=selected[:, keys])
    return (; data, entity_encoder=encoder, entity_vocabulary=collect(encoder.vocabulary), retained_rows=rows)
end

function _model_input(data::LearnerData, spec::ModelSpec)
    if spec.uses_windows
        data.windows === nothing && throw(ArgumentError("$(spec.public_name) requires causal windows"))
        if spec.uses_entity
            data.entity_codes === nothing && throw(ArgumentError("$(spec.public_name) requires entity codes"))
            return (windows=data.windows, entity_codes=data.entity_codes)
        end
        return data.windows
    end
    data.tabular
end

struct PredictionSurface
    keys::DataFrame
    prediction::Vector{Float64}
    model_name::Symbol
    contract_digest::String
    scale::Symbol
    provenance::NamedTuple
end

_semantic_provenance(provenance::NamedTuple) =
    haskey(provenance, :execution_metrics) ?
        Base.structdiff(provenance, (; execution_metrics=nothing)) : provenance

Base.:(==)(a::PredictionSurface, b::PredictionSurface) =
    a.keys == b.keys && a.prediction == b.prediction && a.model_name == b.model_name &&
    a.contract_digest == b.contract_digest && a.scale == b.scale &&
    _semantic_provenance(a.provenance) == _semantic_provenance(b.provenance)

Base.isapprox(a::PredictionSurface, b::PredictionSurface; kwargs...) =
    a.keys == b.keys && isapprox(a.prediction, b.prediction; kwargs...) &&
    a.model_name == b.model_name && a.contract_digest == b.contract_digest &&
    a.scale == b.scale &&
    _semantic_provenance(a.provenance) == _semantic_provenance(b.provenance)

function DataFrames.DataFrame(p::PredictionSurface)
    out = copy(p.keys)
    out[!, :prediction] = p.prediction
    out[!, :model_name] = fill(String(p.model_name), nrow(out))
    out[!, :contract_digest] = fill(p.contract_digest, nrow(out))
    out[!, :prediction_scale] = fill(String(p.scale), nrow(out))
    out[!, :provenance] = fill(p.provenance, nrow(out))
    out
end
