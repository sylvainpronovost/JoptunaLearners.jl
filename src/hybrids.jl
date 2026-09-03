function _validate_keys(df::DataFrame,key_cols)
    DataFrames.nrow(df)>0 || throw(ArgumentError("prediction surfaces must not be empty"))
    isempty(key_cols) && throw(ArgumentError("explicit nonempty row keys are required"))
    allunique(key_cols) || throw(ArgumentError("row keys must be unique column names"))
    any(c->c in (:prediction,:prediction_scale,:contract_digest,:provenance),key_cols) &&
        throw(ArgumentError("row keys must not collide with output metadata or prediction columns"))
    all(c->c in propertynames(df),key_cols) || throw(ArgumentError("prediction surface lacks required key columns"))
    DataFrames.nonunique(df,collect(key_cols)) |> any && throw(ArgumentError("prediction surface has duplicate row keys"))
end

function _align_surfaces(surfaces,key_cols,prediction; predictions=fill(prediction,length(surfaces)))
    length(surfaces)>=2 || throw(ArgumentError("at least two component surfaces are required"))
    dfs=DataFrame.(surfaces)
    length(predictions)==length(dfs) || throw(DimensionMismatch("one prediction column per surface is required"))
    for (df, column) in zip(dfs,predictions)
        _validate_keys(df,key_cols)
        column in key_cols && throw(ArgumentError("prediction columns must be separate from row keys"))
        column in propertynames(df) || throw(ArgumentError("surface lacks $column"))
        all(x -> x isa Real && isfinite(x), df[!,column]) || throw(ArgumentError("predictions must be finite real values"))
    end
    reference=sort(DataFrames.select(dfs[1],collect(key_cols)),collect(key_cols))
    for df in dfs[2:end]
        candidate=sort(DataFrames.select(df,collect(key_cols)),collect(key_cols))
        isequal(reference,candidate) || throw(ArgumentError("component prediction surfaces are incomplete or misaligned"))
    end
    map(dfs,predictions) do df,column
        DataFrames.leftjoin(reference,DataFrames.select(df,vcat(collect(key_cols),[column]));on=collect(key_cols),validate=(true,true),order=:left)
    end
end

# Missing metadata is an explicit caller declaration, not proof of provenance.
# Present metadata must agree; a new output label must never override a conflict.
function _surface_contracts(surfaces, expected=nothing)
    observed = String[]
    for surface in surfaces
        df=DataFrame(surface)
        if :contract_digest in propertynames(df)
            all(x->x isa AbstractString && !isempty(x), df.contract_digest) ||
                throw(ArgumentError("invalid input contract digest"))
            append!(observed, df.contract_digest)
        end
    end
    if expected !== nothing
        isempty(expected) && throw(ArgumentError("contract digest must not be empty"))
        push!(observed, String(expected))
    end
    unique!(observed)
    length(observed)<=1 || throw(ArgumentError("input validation contracts disagree"))
    isempty(observed) ? nothing : only(observed)
end

function _retain_surface_metadata!(out, surfaces, key_cols; expected=nothing)
    digest = _surface_contracts(surfaces, expected)
    digest === nothing || (out[!,:contract_digest] = fill(digest,DataFrames.nrow(out)))
    columns = map(surfaces) do surface
        df=DataFrame(surface)
        if :provenance in propertynames(df)
            aligned=DataFrames.leftjoin(DataFrames.select(out,collect(key_cols)),
                DataFrames.select(df,vcat(collect(key_cols),[:provenance]));on=collect(key_cols),validate=(true,true),order=:left)
            aligned.provenance
        else
            fill(nothing,DataFrames.nrow(out))
        end
    end
    out[!,:provenance] = [(; components=Tuple(column[row] for column in columns)) for row in 1:DataFrames.nrow(out)]
    out
end

function _require_scale(df::DataFrame, label::AbstractString, supplied, expected::Symbol)
    observed = if :prediction_scale in propertynames(df)
        scales=unique(Symbol.(df.prediction_scale))
        length(scales)==1 || throw(ArgumentError("$label contains mixed prediction scales"))
        only(scales)
    elseif supplied === nothing
        throw(ArgumentError("$label must carry prediction_scale or an explicit scale keyword"))
    else
        Symbol(supplied)
    end
    observed==expected||throw(ArgumentError("$label must be $expected, got $observed"))
    supplied===nothing || Symbol(supplied)==observed || throw(ArgumentError("$label scale keyword contradicts surface metadata"))
    observed
end

function _group_ranks(df,group_cols,prediction)
    isempty(group_cols) && return StatsBase.tiedrank(Float64.(df[!,prediction]))
    out=similar(Float64[],DataFrames.nrow(df)); resize!(out,DataFrames.nrow(df))
    for group in DataFrames.groupby(df,collect(group_cols))
        out[group.__row__]=StatsBase.tiedrank(Float64.(group[!,prediction]))
    end
    out
end

"""Blend ranks on explicit row keys; `group_cols=()` ranks the entire surface."""
function rank_blend(surfaces;weights=fill(1/length(surfaces),length(surfaces)),
                    key_cols,group_cols=(),prediction=:prediction)
    allunique(group_cols) && all(c->c in key_cols,group_cols) ||
        throw(ArgumentError("ranking groups must be unique columns from the row keys"))
    length(weights)==length(surfaces) || throw(DimensionMismatch("one weight is required per surface"))
    all(w->isfinite(w)&&w>=0,weights) || throw(ArgumentError("weights must be finite and nonnegative"))
    isfinite(sum(weights)) && sum(weights)>0 || throw(ArgumentError("weights must have finite positive sum"))
    aligned=_align_surfaces(surfaces,key_cols,prediction)
    digest=_surface_contracts(surfaces)
    base=DataFrames.select(aligned[1],collect(key_cols))
    base[!,:prediction]=zeros(DataFrames.nrow(base))
    for (w,df) in zip(weights./sum(weights),aligned)
        df[!,:__row__]=1:DataFrames.nrow(df)
        base.prediction .+= w .* _group_ranks(df,group_cols,prediction)
    end
    base[!,:prediction_scale].="dimensionless_rank"
    _retain_surface_metadata!(base,surfaces,key_cols;expected=digest)
    base
end

struct SimplexRankStack
    weights::Vector{Float64}
    component_names::Vector{Symbol}
    key_cols::Tuple{Vararg{Symbol}}
    group_cols::Tuple{Vararg{Symbol}}
    contract_digest::String
    selection_value::Float64
    candidate_count::Int
end

"""
Select weights on caller-declared out-of-fold data using a finite candidate set.
Always includes simplex vertices and equal weights; `weight_candidates` adds
caller-declared candidates. Ties retain the first candidate. This is not a claim
of global optimization of a discontinuous rank objective. `inner_oof=true` is
the caller's provenance assertion, not an independent leakage check.
"""
function fit_simplex_rank_stack(surfaces,evaluator;component_names=nothing,
                                key_cols,group_cols=(),weight_candidates=(),
                                prediction=:prediction,direction=:maximize,
                                inner_oof::Bool=false,contract_digest::AbstractString)
    inner_oof || throw(ArgumentError("learned rank stacks require explicitly declared inner-OOF component surfaces"))
    _surface_contracts(surfaces,contract_digest)
    direction in (:minimize,:maximize) || throw(ArgumentError("direction must be :minimize or :maximize"))
    n=length(surfaces)
    n>=2 || throw(ArgumentError("at least two component surfaces are required"))
    names=component_names===nothing ? [Symbol(:component_,i) for i in eachindex(surfaces)] : Symbol.(component_names)
    length(names)==length(surfaces) || throw(DimensionMismatch("component names and surfaces differ"))
    allunique(names) || throw(ArgumentError("component names must be unique"))
    candidates = [[i==j ? 1.0 : 0.0 for i in 1:n] for j in 1:n]
    push!(candidates,fill(1/n,n))
    for candidate in weight_candidates
        length(candidate)==n || throw(DimensionMismatch("one weight per component is required"))
        weights=Float64.(collect(candidate))
        all(w->isfinite(w)&&w>=0,weights) && isfinite(sum(weights)) && sum(weights)>0 ||
            throw(ArgumentError("candidate weights must be finite, nonnegative and have positive sum"))
        push!(candidates,weights./sum(weights))
    end
    unique!(candidates)
    values=map(candidates) do weights
        blended=rank_blend(surfaces;weights,key_cols,group_cols,prediction)
        value=Float64(evaluator(blended))
        isfinite(value) || throw(ArgumentError("stack evaluator returned a nonfinite value"))
        value
    end
    index=direction===:maximize ? argmax(values) : argmin(values)
    SimplexRankStack(candidates[index],names,Tuple(key_cols),Tuple(group_cols),String(contract_digest),values[index],length(candidates))
end

function apply_rank_stack(stack::SimplexRankStack,surfaces;contract_digest::AbstractString,
                          prediction=:prediction)
    assert_contract(stack.contract_digest,contract_digest)
    _surface_contracts(surfaces,contract_digest)
    rank_blend(surfaces;weights=stack.weights,key_cols=stack.key_cols,
               group_cols=stack.group_cols,prediction)
end

function make_residual_targets(base_surface,target_surface;key_cols,
                               base_prediction=:prediction,target=:target,
                               base_scale=nothing,inner_oof::Bool=false,
                               contract_digest::AbstractString,output=:residual_target)
    inner_oof || throw(ArgumentError("residual targets require an explicitly declared inner-OOF base surface"))
    base=DataFrame(base_surface); target_df=DataFrame(target_surface)
    _surface_contracts([base,target_df],contract_digest)
    _require_scale(base,"base prediction surface",base_scale,:target_scale)
    _validate_keys(base,key_cols); _validate_keys(target_df,key_cols)
    for (df,column) in ((base,base_prediction),(target_df,target))
        column in propertynames(df) || throw(ArgumentError("surface lacks $column"))
        all(x->x isa Real && isfinite(x),df[!,column]) || throw(ArgumentError("residual inputs must be finite real values"))
    end
    (output in key_cols || output in (:prediction_scale,:contract_digest,:provenance)) &&
        throw(ArgumentError("residual output must not replace a row key or metadata"))
    aligned=_align_surfaces([base,target_df],key_cols,base_prediction;predictions=[base_prediction,target])
    joined=DataFrames.select(aligned[1],collect(key_cols))
    joined[!,output]=Float64.(aligned[2][!,target]).-Float64.(aligned[1][!,base_prediction])
    joined[!,:prediction_scale].="target_scale_residual"
    joined[!,:contract_digest].=String(contract_digest)
    _retain_surface_metadata!(joined,[base,target_df],key_cols;expected=contract_digest)
    joined
end

function apply_residual_correction(base_surface,residual_surface;key_cols,
                                   base_prediction=:prediction,residual_prediction=:prediction,
                                   base_scale=nothing,residual_scale=nothing,
                                   contract_digest::AbstractString)
    base=DataFrame(base_surface); residual=DataFrame(residual_surface)
    _surface_contracts([base,residual],contract_digest)
    _require_scale(base,"base prediction surface",base_scale,:target_scale)
    _require_scale(residual,"residual prediction surface",residual_scale,:target_scale_residual)
    aligned=_align_surfaces([base_surface,residual_surface],key_cols,base_prediction;
                            predictions=[base_prediction,residual_prediction])
    residual_prediction in propertynames(residual) || throw(ArgumentError("residual surface lacks $residual_prediction"))
    keys=sort(DataFrames.select(aligned[1],collect(key_cols)),collect(key_cols))
    r=DataFrames.leftjoin(keys,DataFrames.select(residual,vcat(collect(key_cols),[residual_prediction]));on=collect(key_cols),validate=(true,true),order=:left)
    out=copy(keys)
    out[!,:prediction]=Float64.(aligned[1][!,base_prediction]).+Float64.(r[!,residual_prediction])
    out[!,:prediction_scale].="target_scale"
    out[!,:contract_digest].=String(contract_digest)
    _retain_surface_metadata!(out,[base,residual],key_cols;expected=contract_digest)
    out
end
