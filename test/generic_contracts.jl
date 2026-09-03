@testset "Training-fitted entity vocabulary" begin
    train=DataFrame(entity=repeat(["A","B"],inner=3),time=repeat(1:3,2),x=Float32.(1:6),y=ones(6))
    kwargs=(;feature_cols=[:x],target_col=:y,entity_col=:entity,order_col=:time,lookback=2)
    fitted=prepare_windows(train;kwargs...)
    validation=prepare_windows(train[4:6,:];kwargs...,entity_encoder=fitted.entity_encoder)
    @test validation.data.entity_codes == [2,2]
    @test validation.entity_vocabulary == ["A","B"]
    @test encode_entities(fitted.entity_encoder,["B","A"]) == [2,1]
    @test_throws ArgumentError encode_entities(fitted.entity_encoder,["C"])
    @test_throws ArgumentError EntityEncoder(["A","A"])
    @test_throws ArgumentError EntityEncoder([missing])
    lazy=prepare_windows(train[4:6,:];kwargs...,materialization=:lazy,entity_encoder=fitted.entity_encoder)
    @test lazy.data.entity_codes == validation.data.entity_codes
    @test Array(lazy.data.windows) == validation.data.windows
end

@testset "Explicit metric identity and ungrouped validation" begin
    metric=(p,d)->sum(p)
    a=ValidationSpec(:score,metric;direction=:maximize,metric_id="example/sum",metric_version="1",metric_config=(;scale=1.0))
    @test a.grouping == ()
    @test a.metric_id == "example/sum"
    @test a.digest == ValidationSpec(:score,metric;direction=:maximize,metric_id="example/sum",metric_version="1",metric_config=Dict("scale"=>1.0)).digest
    @test a.digest != ValidationSpec(:score,metric;direction=:maximize,metric_id="example/sum",metric_version="2",metric_config=(;scale=1.0)).digest
    @test a.digest != ValidationSpec(:score,metric;direction=:maximize,metric_id="example/sum",metric_config=(;scale=2.0)).digest
    @test_throws ArgumentError ValidationSpec(:score,metric;direction=:maximize,metric_version="")
    @test_throws ArgumentError ValidationSpec(:score,metric;direction=:maximize,grouping=(:row,:row))
    @test_throws ArgumentError ValidationSpec(:score,metric;direction=:maximize,metric_config=(;scale=NaN))
    # Length-framed encoding prevents delimiter collisions in labels/configuration.
    @test JoptunaLearners._contract_encode(("a;b","c")) != JoptunaLearners._contract_encode(("a","b;c"))
end

@testset "Rank-stack plateaus and arbitrary row keys" begin
    a=DataFrame(sample=[1,2,3,4],prediction=[1.,2,3,4])
    b=DataFrame(sample=[3,1,4,2],prediction=[4.,3,2,1])
    score(df)=sum(sign(df.prediction[j]-df.prediction[i]) for i=1:3 for j=i+1:4)
    key_cols=(:sample,)
    stack=fit_simplex_rank_stack([a,b],score;key_cols,inner_oof=true,contract_digest="synthetic")
    @test stack.weights == [1.,0.]
    @test stack.selection_value == 6
    @test stack.candidate_count == 3
    @test score(apply_rank_stack(stack,[a,b];contract_digest="synthetic")) == 6
    @test rank_blend([a,b];key_cols).prediction == [2.,1.5,3.5,3.]
    @test_throws UndefKeywordError rank_blend([a,b])
    @test_throws ArgumentError rank_blend([a,b];key_cols,group_cols=(:missing,))
    @test_throws ArgumentError fit_simplex_rank_stack([a,b],score;key_cols,inner_oof=true,contract_digest="x",direction=:invalid)
    @test_throws ArgumentError fit_simplex_rank_stack([a,b],df->NaN;key_cols,inner_oof=true,contract_digest="x")
    @test_throws ArgumentError fit_simplex_rank_stack([a,b],score;key_cols,inner_oof=true,contract_digest="x",weight_candidates=([-1.,2.],))
    selected=fit_simplex_rank_stack([a,b],score;key_cols,inner_oof=true,contract_digest="x",weight_candidates=([3.,1.],),direction=:minimize)
    @test selected.candidate_count == 4
    @test selected.selection_value <= score(rank_blend([a,b];key_cols))
    @test_throws ArgumentError rank_blend([a,vcat(b,b[1:1,:])];key_cols)
    @test_throws ArgumentError rank_blend([a,b[1:3,:]];key_cols)
    @test_throws ArgumentError rank_blend([a,b];key_cols=(:prediction,))
end

@testset "Portable prediction metadata and residual columns" begin
    a=PredictionSurface(DataFrame(row_id=[2,1]),[2.,1.],:mlp,"metric-v1",:target_scale,(;origin="synthetic"))
    df=DataFrame(a)
    @test df.prediction_scale == ["target_scale","target_scale"]
    @test df.provenance == fill((;origin="synthetic"),2)
    base=rename(df,:prediction=>:base)
    residual=DataFrame(row_id=[1,2],residual=[0.5,0.25],prediction_scale=fill("target_scale_residual",2),contract_digest=fill("metric-v1",2))
    out=apply_residual_correction(base,residual;key_cols=(:row_id,),base_prediction=:base,
        residual_prediction=:residual,contract_digest="metric-v1")
    @test out.row_id == [1,2]
    @test out.prediction == [1.5,2.25]
    @test out.provenance[1].components[1] == (;origin="synthetic")
    @test_throws ArgumentError apply_residual_correction(base,residual;key_cols=(:row_id,),
        base_prediction=:base,residual_prediction=:residual,contract_digest="different")
    residual.contract_digest .= "other"
    @test_throws ArgumentError apply_residual_correction(base,residual;key_cols=(:row_id,),
        base_prediction=:base,residual_prediction=:residual,contract_digest="metric-v1")
    target=DataFrame(row_id=[1,2],target=[2.,4.],contract_digest=fill("other",2))
    @test_throws ArgumentError make_residual_targets(df,target;key_cols=(:row_id,),inner_oof=true,contract_digest="metric-v1")
    target.contract_digest .= "metric-v1"
    made=make_residual_targets(df,target;key_cols=(:row_id,),inner_oof=true,contract_digest="metric-v1")
    @test made.row_id == [1,2]
    @test made.residual_target == [1.,2.]
    renamed_target=rename(target,:target=>:base)
    same_names=make_residual_targets(base,renamed_target;key_cols=(:row_id,),
        base_prediction=:base,target=:base,inner_oof=true,contract_digest="metric-v1")
    @test same_names.residual_target == [1.,2.]
    @test_throws ArgumentError make_residual_targets(base,renamed_target;key_cols=(:row_id,),
        base_prediction=:base,target=:base,output=:contract_digest,inner_oof=true,contract_digest="metric-v1")
    @test_throws ArgumentError make_residual_targets(base,renamed_target;key_cols=(:row_id,),
        base_prediction=:base,target=:base,base_scale=:dimensionless_rank,inner_oof=true,contract_digest="metric-v1")
end
