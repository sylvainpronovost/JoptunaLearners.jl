function _validate_keys(df::DataFrame,key_cols)
    all(c->c in propertynames(df),key_cols) || throw(ArgumentError("prediction surface lacks required key columns"))
    DataFrames.nonunique(df,collect(key_cols)) |> any && throw(ArgumentError("prediction surface has duplicate row keys"))
end

function _align_surfaces(surfaces,key_cols,prediction)
    length(surfaces)>=2 || throw(ArgumentError("at least two component surfaces are required"))
    dfs=DataFrame.(surfaces)
    for df in dfs
        _validate_keys(df,key_cols)
        prediction in propertynames(df) || throw(ArgumentError("surface lacks $prediction"))
    end
    reference=sort(DataFrames.select(dfs[1],collect(key_cols)),collect(key_cols))
    for df in dfs[2:end]
        candidate=sort(DataFrames.select(df,collect(key_cols)),collect(key_cols))
        isequal(reference,candidate) || throw(ArgumentError("component prediction surfaces are incomplete or misaligned"))
    end
    map(dfs) do df
        DataFrames.leftjoin(reference,DataFrames.select(df,vcat(collect(key_cols),[prediction]));on=collect(key_cols),validate=(true,true))
    end
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
    observed
end

function _group_ranks(df,group_cols,prediction)
    out=similar(Float64[],DataFrames.nrow(df)); resize!(out,DataFrames.nrow(df))
    for group in DataFrames.groupby(df,collect(group_cols))
        out[group.__row__]=StatsBase.tiedrank(Float64.(group[!,prediction]))
    end
    out
end

function rank_blend(surfaces;weights=fill(1/length(surfaces),length(surfaces)),
                    key_cols=(:group_id,:entity_id,:fold_id),group_cols=(:group_id,),prediction=:prediction)
    length(weights)==length(surfaces) || throw(DimensionMismatch("one weight is required per surface"))
    all(w->isfinite(w)&&w>=0,weights) || throw(ArgumentError("weights must be finite and nonnegative"))
    sum(weights)>0 || throw(ArgumentError("weights must have positive sum"))
    aligned=_align_surfaces(surfaces,key_cols,prediction)
    base=DataFrames.select(aligned[1],collect(key_cols))
    base[!,:prediction]=zeros(DataFrames.nrow(base))
    for (w,df) in zip(weights./sum(weights),aligned)
        df[!,:__row__]=1:DataFrames.nrow(df)
        base.prediction .+= w .* _group_ranks(df,group_cols,prediction)
    end
    base[!,:prediction_scale].="dimensionless_rank"
    base
end

struct SimplexRankStack
    weights::Vector{Float64}
    component_names::Vector{Symbol}
    key_cols::Tuple{Vararg{Symbol}}
    group_cols::Tuple{Vararg{Symbol}}
    contract_digest::String
end

function _softmax_weights(z)
    e=exp.(z.-maximum(z)); e./sum(e)
end

function fit_simplex_rank_stack(surfaces,evaluator;component_names=nothing,
                                key_cols=(:group_id,:entity_id,:fold_id),group_cols=(:group_id,),
                                prediction=:prediction,direction=:maximize,
                                inner_oof::Bool=false,contract_digest::AbstractString)
    inner_oof || throw(ArgumentError("learned rank stacks require explicitly declared inner-OOF component surfaces"))
    names=component_names===nothing ? [Symbol(:component_,i) for i in eachindex(surfaces)] : Symbol.(component_names)
    length(names)==length(surfaces) || throw(DimensionMismatch("component names and surfaces differ"))
    objective=z->begin
        blended=rank_blend(surfaces;weights=_softmax_weights(z),key_cols,group_cols,prediction)
        value=Float64(evaluator(blended))
        direction===:maximize ? -value : value
    end
    result=Optim.optimize(objective,zeros(length(surfaces)),Optim.NelderMead(),
                          Optim.Options(iterations=500))
    SimplexRankStack(_softmax_weights(Optim.minimizer(result)),names,Tuple(key_cols),Tuple(group_cols),String(contract_digest))
end

function apply_rank_stack(stack::SimplexRankStack,surfaces;contract_digest::AbstractString,
                          prediction=:prediction)
    assert_contract(stack.contract_digest,contract_digest)
    rank_blend(surfaces;weights=stack.weights,key_cols=stack.key_cols,
               group_cols=stack.group_cols,prediction)
end

function make_residual_targets(base_surface,target_surface;key_cols=(:group_id,:entity_id,:fold_id),
                               base_prediction=:prediction,target=:target,
                               base_scale=nothing,inner_oof::Bool=false,
                               contract_digest::AbstractString)
    inner_oof || throw(ArgumentError("residual targets require an explicitly declared inner-OOF base surface"))
    base=DataFrame(base_surface); target_df=DataFrame(target_surface)
    _require_scale(base,"base prediction surface",base_scale,:target_scale)
    _validate_keys(base,key_cols); _validate_keys(target_df,key_cols)
    joined=DataFrames.innerjoin(DataFrames.select(base,vcat(collect(key_cols),[base_prediction])),
                     DataFrames.select(target_df,vcat(collect(key_cols),[target]));on=collect(key_cols),validate=(true,true),makeunique=true)
    DataFrames.nrow(joined)==DataFrames.nrow(base)==DataFrames.nrow(target_df) || throw(ArgumentError("base and target surfaces are incomplete"))
    joined[!,:inner_residual_target]=Float64.(joined[!,target]).-Float64.(joined[!,base_prediction])
    joined[!,:prediction_scale].="target_scale_residual"
    joined[!,:contract_digest].=String(contract_digest)
    joined
end

function apply_residual_correction(base_surface,residual_surface;key_cols=(:group_id,:entity_id,:fold_id),
                                   base_prediction=:prediction,residual_prediction=:prediction,
                                   base_scale=nothing,residual_scale=nothing,
                                   contract_digest::AbstractString)
    base=DataFrame(base_surface); residual=DataFrame(residual_surface)
    _require_scale(base,"base prediction surface",base_scale,:target_scale)
    _require_scale(residual,"residual prediction surface",residual_scale,:target_scale_residual)
    aligned=_align_surfaces([base_surface,residual_surface],key_cols,base_prediction)
    residual_prediction in propertynames(residual) || throw(ArgumentError("residual surface lacks $residual_prediction"))
    keys=sort(DataFrames.select(aligned[1],collect(key_cols)),collect(key_cols))
    r=DataFrames.leftjoin(keys,DataFrames.select(residual,vcat(collect(key_cols),[residual_prediction]));on=collect(key_cols),validate=(true,true))
    out=copy(keys)
    out[!,:prediction]=Float64.(aligned[1][!,base_prediction]).+Float64.(r[!,residual_prediction])
    out[!,:prediction_scale].="target_scale"
    out[!,:contract_digest].=String(contract_digest)
    out
end
