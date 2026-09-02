using Test
using ADTypes
using DataFrames
using Dates
using JoptunaLearners
using LearnAPI
using Lux
using Optimisers
using Random
using Statistics
using Zygote

@testset "Prediction-surface semantic equality" begin
    keys = DataFrame(row_id=1:2)
    stable = (; package="JoptunaLearners.jl", model="MLPRegressor", seed=7,
              execution_metrics=(; total_fit_seconds=1.0))
    later = merge(stable, (; execution_metrics=(; total_fit_seconds=2.0)))
    first = PredictionSurface(keys, [0.1, 0.2], :mlp, "digest", :model_output, stable)
    second = PredictionSurface(keys, [0.1, 0.2], :mlp, "digest", :model_output, later)
    changed = PredictionSurface(keys, [0.1, 0.2], :mlp, "digest", :model_output,
                                merge(later, (; seed=8)))
    @test first == second
    @test isapprox(first, second)
    @test first != changed
end

@testset "JoptunaLearners foundation" begin
    @test VERSION >= v"1.12"
    @test audit_runtime_purity()
    @test length(model_specs()) == 14
    @test Set(s.name for s in model_specs()) == Set((:mlp,:tabular_mixer,:tabular_resnet,
        :window_mlp,:window_linear,:window_nlinear,:window_dlinear,:tsmixer,:film,
        :tcn,:tcn_v2,:patchtst,:tft,:mamba))
    @test model_spec(:film).uses_entity
    @test model_spec(:window_dlinear).defaults.kernel_size == 5
    @test hyperparameter_schema(:patchtst).n_heads.values == [2,4,8]
    @test_throws ArgumentError model_spec(:not_a_model)
end

@testset "Typed execution backends" begin
    eager = EagerCPU()
    reactant = ReactantCPU()
    metal = MetalGPU()
    @test backend_capabilities(eager).available
    @test backend_capabilities(eager).name == :eager_cpu
    @test backend_capabilities(eager).performance_qualified
    @test backend_capabilities(eager).recommended
    @test backend_capabilities(reactant).name == :reactant_cpu
    @test !backend_capabilities(reactant).performance_qualified
    @test !backend_capabilities(reactant).recommended
    @test backend_capabilities(metal).name == :metal_gpu
    @test !backend_capabilities(metal).performance_qualified
    @test !backend_capabilities(metal).recommended
    @test backend_capabilities(eager, :window_dlinear).qualification_result == :reference
    @test backend_capabilities(reactant, :window_dlinear).performance_qualified
    @test backend_capabilities(metal, :patchtst).performance_qualified
    @test backend_capabilities(metal, :patchtst).performance_reason ==
          :large_batch_attention_throughput
    if backend_capabilities(metal).available
        @test require_performance_qualified(metal, :patchtst).name == :metal_gpu
    else
        @test_throws ArgumentError require_performance_qualified(metal, :patchtst)
    end
    @test backend_capabilities(reactant, :window_dlinear).recommended
    @test backend_capabilities(metal, :window_dlinear).qualification_result ==
          :correct_but_slower
    @test backend_capabilities(metal, :window_dlinear).performance_reason ==
          :device_launch_bound
    @test require_performance_qualified(eager, :window_dlinear).name == :eager_cpu
    @test_throws ArgumentError require_performance_qualified(metal, :window_dlinear)
    @test_throws ArgumentError backend_capabilities(eager, :not_a_model)
    @test TrainingSpec().backend == eager
    @test TrainingSpec(device=:cpu).backend == eager
    @test TrainingSpec(backend=reactant).device == :cpu
    @test TrainingSpec(backend=metal).device == :metal
    @test_throws ArgumentError ReactantCPU(fallback=:eager)
    @test_throws ArgumentError MetalGPU(fallback=:cpu)
    @test_throws ArgumentError MetalGPU(device=2)
    @test_throws ArgumentError TrainingSpec(device=:cpu, backend=metal)
end

@testset "Validation contracts" begin
    evaluator=(pred,data)->mean(pred)
    a=ValidationSpec(:example_metric,evaluator;direction=:maximize,prediction=:prediction,
                     target=:target,grouping=(:row_id,))
    b=ValidationSpec(:example_metric,evaluator;direction=:maximize,prediction=:prediction,
                     target=:alternative_target,grouping=(:row_id,))
    @test length(a.digest)==64
    @test a.digest != b.digest
    @test assert_contract(a.digest,a.digest) === nothing
    @test_throws ArgumentError assert_contract(a.digest,b.digest)
    defaults=ValidationSpec(:mse,evaluator;direction=:minimize)
    @test defaults.prediction == :prediction
    @test defaults.target == :target
    @test defaults.grouping == (:row_id,)
end

@testset "Causal data and train-only normalization" begin
    df=DataFrame(time_index=repeat(1:5,2),entity_id=repeat([:a,:b],inner=5),
                 x=Float32.(1:10),y=Float32.(11:20))
    prepared=prepare_windows(df;feature_cols=[:x],target_col=:y,entity_col=:entity_id,
                             order_col=:time_index,lookback=3)
    @test length(prepared.data)==6
    @test size(prepared.data.windows)==(1,3,6)
    @test vec(prepared.data.windows[1,:,1]) == Float32[1,2,3]
    @test vec(prepared.data.windows[1,:,4]) == Float32[6,7,8]
    @test prepared.retained_rows == [3,4,5,8,9,10]
    @test prepared.data.keys == df[[3,4,5,8,9,10],[:time_index,:entity_id]]
    normalizer=fit_normalizer(Float32[1 2 3;10 20 30])
    normalized=apply_normalizer(normalizer,Float32[1 2 3;10 20 30])
    @test maximum(abs.(mean(normalized;dims=2))) < 1f-6
end

@testset "Lazy causal windows preserve dense semantics" begin
    df = DataFrame(
        time_index=repeat(1:9, 3), entity_id=repeat([:a, :b, :c], inner=9),
        x1=Float32.(1:27), x2=Float32.(mod.(1:27, 5)), y=Float32.(31:57),
    )
    kwargs = (; feature_cols=[:x1, :x2], target_col=:y, entity_col=:entity_id,
              order_col=:time_index, lookback=4)
    dense = prepare_windows(df; kwargs..., materialization=:dense)
    lazy = prepare_windows(df; kwargs..., materialization=:lazy)
    @test lazy.data.windows isa CausalWindowSource
    @test dense.retained_rows == lazy.retained_rows
    @test dense.entity_vocabulary == lazy.entity_vocabulary
    @test dense.data.tabular == lazy.data.tabular
    @test dense.data.target == lazy.data.target
    @test dense.data.entity_codes == lazy.data.entity_codes
    @test dense.data.keys == lazy.data.keys
    @test dense.data.windows == Array(lazy.data.windows)
    lazy_subset = lazy.data[2:2:length(lazy.data)]
    @test lazy_subset.windows isa CausalWindowSource
    @test Array(lazy_subset.windows) == Array(lazy.data.windows)[:, :, 2:2:end]

    validation = ValidationSpec(:mse,
        (prediction, data) -> mean((prediction .- data.target) .^ 2);
        direction=:minimize)
    training = TrainingSpec(epochs=2, batch_size=5, seed=91, patience=0)
    learner = LuxLearner(:window_dlinear; training, kernel_size=3)
    fitted_dense = fit(learner, dense.data; validation, validation_data=dense.data,
                       verbosity=-1)
    fitted_lazy = fit(learner, lazy.data; validation, validation_data=lazy.data,
                      verbosity=-1)
    event_contract(event) = (event.epoch, event.update, event.training_loss,
                             event.validation_value, event.contract_digest)
    @test event_contract.(training_report(fitted_dense).events) ==
          event_contract.(training_report(fitted_lazy).events)
    @test predict(fitted_dense, dense.data).prediction ==
          predict(fitted_lazy, lazy.data).prediction
end

@testset "Lazy subsets share their canonical feature surface" begin
    values = reshape(Float32.(1:80), 2, 40)
    source = CausalWindowSource(values, collect(1:32), 9)
    data = LearnerData(values[:, 9:40], ones(Float32, 32); windows=source)
    subset = data[collect(1:2:31)]
    @test subset.windows.values === data.windows.values
    @test subset.windows.starts == data.windows.starts[1:2:31]
    @test subset.windows[1, 1, 2] == data.windows[1, 1, 3]
end

@testset "Window construction has bounded transient allocation" begin
    rows_per_sid=1_000
    frame=DataFrame(
        time_index=repeat(1:rows_per_sid,10),entity_id=repeat(1:10,inner=rows_per_sid),
        x1=Float32.(1:(10*rows_per_sid)),x2=fill(0.25f0,10*rows_per_sid),
        y=fill(0.0f0,10*rows_per_sid),weight=fill(1.0f0,10*rows_per_sid),
        irrelevant=fill("not copied into the window surface",10*rows_per_sid))
    kwargs=(;feature_cols=[:x1,:x2],target_col=:y,entity_col=:entity_id,
            order_col=:time_index,lookback=60,weight_col=:weight)
    prepare_windows(frame;kwargs...) # compile before measuring
    allocated=@allocated prepared=prepare_windows(frame;kwargs...)
    @test size(prepared.data.windows)==(2,60,9_410)
    @test allocated < 100_000_000

    prepare_windows(frame; kwargs..., materialization=:lazy)
    lazy_allocated = @allocated lazy = prepare_windows(
        frame; kwargs..., materialization=:lazy)
    @test lazy.data.windows isa CausalWindowSource
    @test lazy_allocated < allocated
    @test Base.summarysize(lazy.data.windows) <=
          Base.summarysize(prepared.data.windows) * 0.20
end

@testset "Owned window preparation avoids a redundant column copy" begin
    frame = DataFrame(
        time_index=repeat(Date(2020, 1, 1):Day(1):Date(2020, 1, 20), 2),
        entity_id=repeat(["A", "B"], inner=20),
        feature=Float32.(1:40),
        target=Float32.(1:40),
    )
    protected = copy(frame)
    prepare_windows(frame; feature_cols=[:feature], target_col=:target,
        entity_col=:entity_id, order_col=:time_index, lookback=5,
        materialization=:lazy)
    @test frame == protected

    owned = copy(frame)
    prepared = prepare_windows(owned; feature_cols=[:feature], target_col=:target,
        entity_col=:entity_id, order_col=:time_index, lookback=5,
        materialization=:lazy, copycols=false)
    @test length(prepared.data) == 32
    @test all(isfinite, prepared.data.tabular)
end

@testset "Allocation-light DLinear moving average preserves mathematics" begin
    rng = MersenneTwister(824)
    windows = randn(rng, Float32, 3, 11, 7)
    kernel = 5
    functional = JoptunaLearners._moving_average_functional(windows, kernel)
    optimized = JoptunaLearners._moving_average_cpu(windows, kernel)
    @test isapprox(optimized, functional; rtol=2f-6, atol=2f-6)

    objective(f, x) = sum(abs2, f(x, kernel))
    functional_gradient = only(Zygote.gradient(
        x -> objective(JoptunaLearners._moving_average_functional, x), windows))
    optimized_gradient = only(Zygote.gradient(
        x -> objective(JoptunaLearners._moving_average_cpu, x), windows))
    @test isapprox(optimized_gradient, functional_gradient;
                   rtol=5f-6, atol=5f-6)

    JoptunaLearners._moving_average_cpu(windows, kernel)
    JoptunaLearners._moving_average_functional(windows, kernel)
    optimized_allocation = @allocated JoptunaLearners._moving_average_cpu(windows, kernel)
    functional_allocation = @allocated JoptunaLearners._moving_average_functional(windows, kernel)
    @test optimized_allocation < functional_allocation

    scratch = similar(windows)
    JoptunaLearners._moving_average_cpu!(scratch, windows, kernel)
    scratch_allocation = @allocated JoptunaLearners._moving_average_cpu!(scratch, windows, kernel)
    @test scratch == optimized
    @test scratch_allocation == 0
end

@testset "Closed-form eager DLinear gradient agrees with Zygote" begin
    rng = MersenneTwister(825)
    model = build_model(:window_dlinear; n_features=2, lookback=9,
                        n_entities=0, kernel_size=5)
    parameters, states = Lux.setup(rng, model)
    batch = (
        randn(rng, Float32, 2, 9, 13),
        randn(rng, Float32, 13),
        rand(rng, Float32, 13) .+ 0.1f0,
        0.5f0,
    )
    optimizer = Optimisers.AdamW(1f-3, (0.9f0, 0.999f0), 1f-4; couple=false)
    train_state = Lux.Training.TrainState(model, parameters, states, optimizer)
    reference, reference_loss, _, _ = Lux.Training.compute_gradients(
        ADTypes.AutoZygote(), JoptunaLearners._objective, batch, train_state,
    )
    optimized, optimized_loss = JoptunaLearners._dlinear_eager_gradients(
        model, parameters, batch,
    )
    accelerator, accelerator_loss = JoptunaLearners._dlinear_accelerator_gradients(
        model, parameters, batch,
    )
    @test isapprox(optimized_loss, reference_loss; rtol=2f-6, atol=2f-6)
    @test isapprox(accelerator_loss, reference_loss; rtol=2f-6, atol=2f-6)
    for branch in (:trend, :remainder), field in (:weight, :bias)
        @test isapprox(getproperty(getproperty(optimized, branch), field),
                       getproperty(getproperty(reference, branch), field);
                       rtol=5f-5, atol=5f-6)
        @test isapprox(getproperty(getproperty(accelerator, branch), field),
                       getproperty(getproperty(reference, branch), field);
                       rtol=5f-5, atol=5f-6)
    end

    stager = JoptunaLearners._BatchStager(
        LearnerData(copy(batch[1][:, end, :]), batch[2]; windows=batch[1]),
        model_spec(:window_dlinear),
    )
    workspace = JoptunaLearners._dlinear_batch_workspace!(stager, batch[1])
    workspace_batch = (batch..., workspace)
    JoptunaLearners._dlinear_eager_gradients(model, parameters, workspace_batch)
    @test @allocated(
        JoptunaLearners._dlinear_eager_gradients(model, parameters, workspace_batch)
    ) <= 1024
    JoptunaLearners._dlinear_eager_prediction!(workspace, model, parameters, batch[1])
    @test @allocated(
        JoptunaLearners._dlinear_eager_prediction!(workspace, model, parameters, batch[1])
    ) <= 512
end

@testset "Learner-data digests stream without dataset-sized buffers" begin
    observations = 20_000
    lookback = 9
    values = reshape(Float32.(1:(2 * (observations + lookback - 1))),
                     2, observations + lookback - 1)
    windows = JoptunaLearners.CausalWindowSource(
        values, collect(1:observations), lookback,
    )
    keys = DataFrame(
        time_index=Date(2020, 1, 1) .+ Day.(mod.(0:(observations - 1), 100)),
        entity_id=string.(mod.(0:(observations - 1), 263)),
    )
    data = LearnerData(values[:, lookback:end], ones(Float32, observations);
        windows, keys)
    reference = JoptunaLearners._learner_data_digest(data)
    @test reference == JoptunaLearners._learner_data_digest(data)
    allocated = @allocated JoptunaLearners._learner_data_digest(data)
    @test allocated < 4 * Base.summarysize(data)
    altered_values = copy(values)
    altered_values[1, 1] += 1f0
    altered = LearnerData(altered_values[:, lookback:end], data.target;
        windows=JoptunaLearners.CausalWindowSource(
            altered_values, collect(1:observations), lookback,
        ), keys)
    @test JoptunaLearners._learner_data_digest(altered) != reference
end

@testset "All architecture contracts execute" begin
    rng=MersenneTwister(22)
    for spec in model_specs()
        model=build_model(spec;n_features=4,lookback=12,n_entities=5)
        ps,st=Lux.setup(rng,model)
        x=spec.uses_windows ? randn(rng,Float32,4,12,3) : randn(rng,Float32,4,3)
        input=spec.uses_entity ? (windows=x,entity_codes=[1,2,3]) : x
        y,_=model(input,ps,st)
        @test size(y)==(3,)
        @test all(isfinite,y)
    end
    @test_throws ArgumentError build_model(:window_dlinear;n_features=2,lookback=4,kernel_size=6)
end

@testset "Temporal architecture structural qualification" begin
    rng = MersenneTwister(71)
    x = randn(rng, Float32, 3, 17, 2)

    # This is the defining TCN invariant: observations strictly after a timestep
    # cannot change its encoded causal convolution result.
    w = randn(rng, Float32, 4, 3, 3)
    b = randn(rng, Float32, 4, 1)
    before = JoptunaLearners._causal_conv1d(x, w, b, 2)
    altered = copy(x); altered[:, 12:end, :] .= randn(rng, Float32, 3, 6, 2)
    after = JoptunaLearners._causal_conv1d(altered, w, b, 2)
    @test before[:, 1:11, :] == after[:, 1:11, :]

    tcn = build_model(:tcn; n_features=3, lookback=17, kernel_size=3, depth=2, auto_depth=true)
    @test tcn.config.depth >= 3
    @test 1 + 2 * (tcn.config.kernel_size - 1) * (2^tcn.config.depth - 1) >= 17
    @test_throws ArgumentError build_model(:tcn; n_features=3, lookback=17, kernel_size=3,
        depth=2, auto_depth=false)

    @test_throws ArgumentError build_model(:patchtst; n_features=3, lookback=12,
        hidden_dim=30, n_heads=4)
    @test_throws ArgumentError build_model(:tft; n_features=3, lookback=12,
        hidden_dim=30, n_heads=4)
    short_patch = build_model(:patchtst; n_features=3, lookback=3, patch_len=8, stride=2)
    @test short_patch.config.patch_len == 3

    for (name, config) in ((:tcn, (;)), (:tcn_v2, (;)), (:patchtst, (; depth=2)),
                           (:tft, (; recurrent_layers=2)), (:mamba, (; depth=2)))
        model = build_model(name; n_features=3, lookback=17, pairs(config)...)
        ps, st = Lux.setup(rng, model)
        output, _ = model(x, ps, st)
        @test size(output) == (2,)
        @test all(isfinite, output)
    end

    # A Mamba state at time t is causal for the same reason as the TCN output.
    mamba = build_model(:mamba; n_features=3, lookback=17)
    ps, st = Lux.setup(rng, mamba)
    encoded_before = JoptunaLearners._mamba_block(JoptunaLearners._sequence_linear(x, ps.input), ps.blocks.block1)
    encoded_after = JoptunaLearners._mamba_block(JoptunaLearners._sequence_linear(altered, ps.input), ps.blocks.block1)
    @test encoded_before[:, 1:11, :] == encoded_after[:, 1:11, :]
end

@testset "Every native architecture differentiates" begin
    rng=MersenneTwister(23); n=6
    x=randn(rng,Float32,3,n); windows=randn(rng,Float32,3,12,n)
    y=vec(0.5f0*x[1,:])
    data=LearnerData(x,y;windows,entity_codes=collect(1:n))
    validation=ValidationSpec(:mse,(pred,d)->mean((pred.-d.target).^2);direction=:minimize)
    for spec in model_specs()
        learner=LuxLearner(spec.name;training=TrainingSpec(epochs=1,batch_size=n,seed=3,patience=0))
        fitted=fit(learner,data;validation,validation_data=data,verbosity=-1)
        @test isfinite(training_report(fitted).best_value)
    end
end

@testset "Lux training, callbacks, early stopping, checkpoint" begin
    rng=MersenneTwister(4)
    x=randn(rng,Float32,3,48)
    y=vec(0.7f0*x[1,:].-0.2f0*x[2,:])
    data=LearnerData(x,y)
    validation=ValidationSpec(:mse,(pred,d)->mean((pred.-d.target).^2);direction=:minimize)
    events=TrainingEvent[]
    learner=LuxLearner(:mlp;training=TrainingSpec(epochs=3,batch_size=16,seed=8,patience=0))
    fitted=fit(learner,data;validation,validation_data=data,callbacks=(e->push!(events,e),),verbosity=-1)
    @test length(events)==3
    @test all(e->e.contract_digest==validation.digest,events)
    surface=predict(fitted,data)
    @test length(surface.prediction)==48
    @test LearnAPI.clone(learner)==learner
    path=tempname()*".jld2"
    save_checkpoint(path,fitted)
    restored=load_checkpoint(path)
    @test predict(restored,data).prediction == surface.prediction
    mktempdir() do directory
        invalid=joinpath(directory,"old-layout.jld2")
        JoptunaLearners.JLD2.jldsave(invalid; payload=(format_version=2,))
        @test_throws ArgumentError load_checkpoint(invalid)
        JoptunaLearners.JLD2.jldsave(invalid; snapshot=(format_version=3,))
        @test_throws ArgumentError JoptunaLearners._load_training_checkpoint(invalid)
    end
end

@testset "Minimum epoch policy" begin
    @test_throws ArgumentError TrainingSpec(epochs=4, minimum_epochs=5)
    @test_throws ArgumentError TrainingSpec(minimum_epochs=0)
    rng=MersenneTwister(40)
    data=LearnerData(randn(rng,Float32,2,24),randn(rng,Float32,24))
    validation=ValidationSpec(:constant,(pred,d)->1.0;direction=:minimize)
    learner=LuxLearner(:mlp;training=TrainingSpec(
        epochs=10,batch_size=8,seed=12,minimum_epochs=5,patience=1))
    fitted=fit(learner,data;validation,validation_data=data,verbosity=-1)
    report=training_report(fitted)
    @test report.stopped_early
    @test report.completed_epochs==5
    @test report.best_epoch==1
end

@testset "Fixed refits can omit selection-only validation" begin
    rng = MersenneTwister(918)
    windows = randn(rng, Float32, 2, 9, 32)
    data = LearnerData(copy(windows[:, end, :]), randn(rng, Float32, 32); windows)
    evaluations = Ref(0)
    validation = ValidationSpec(:counted, (_, _) -> begin
        evaluations[] += 1
        0.0
    end; direction=:minimize)
    learner = LuxLearner(:window_dlinear;
        training=TrainingSpec(epochs=3, batch_size=8, patience=0,
            restore_best=false, seed=918), kernel_size=5)
    fitted = fit(learner, data; validation, validation_data=data,
        validation_schedule=:none, verbosity=-1)
    report = training_report(fitted)
    @test evaluations[] == 0
    @test report.completed_epochs == 3
    @test report.best_epoch == 3
    @test isnan(report.best_value)
    @test isempty(report.events)
    @test_throws ArgumentError fit(learner, data; validation,
        validation_data=data, validation_schedule=:invalid, verbosity=-1)
end


@testset "Deterministic interruption and resume" begin
    rng=MersenneTwister(31); x=randn(rng,Float32,2,24); y=vec(x[1,:].-0.3f0*x[2,:])
    data=LearnerData(x,y)
    validation=ValidationSpec(:mse,(pred,d)->mean((pred.-d.target).^2);direction=:minimize)
    training=TrainingSpec(epochs=4,batch_size=8,seed=19,patience=0,checkpoint_every=1)
    learner=LuxLearner(:mlp;training)
    uninterrupted=fit(learner,data;validation,validation_data=data,verbosity=-1)
    checkpoint=tempname()*".jld2"
    interrupt=event->event.epoch==2 ? error("controlled interruption") : nothing
    @test_throws ErrorException fit(learner,data;validation,validation_data=data,
        checkpoint_path=checkpoint,callbacks=(interrupt,),verbosity=-1)
    resumed=fit(learner,data;validation,validation_data=data,resume_from=checkpoint,verbosity=-1)
    @test predict(resumed,data).prediction == predict(uninterrupted,data).prediction
    @test training_report(resumed).best_epoch == training_report(uninterrupted).best_epoch
    @test [e.validation_value for e in training_report(resumed).events] ==
          [e.validation_value for e in training_report(uninterrupted).events]
    changed_training=LuxLearner(:mlp;training=TrainingSpec(
        epochs=4,batch_size=6,seed=19,patience=0,checkpoint_every=1))
    @test_throws ArgumentError fit(changed_training,data;validation,validation_data=data,
        resume_from=checkpoint,verbosity=-1)
    changed_data=LearnerData(copy(x),copy(y)); changed_data.target[1]+=1
    @test_throws ArgumentError fit(learner,changed_data;validation,
        validation_data=changed_data,resume_from=checkpoint,verbosity=-1)
end

@testset "User-owned Lux builder" begin
    rng=MersenneTwister(37); x=randn(rng,Float32,3,18); y=vec(x[1,:])
    data=LearnerData(x,y)
    validation=ValidationSpec(:mse,(p,d)->mean((p.-d.target).^2);direction=:minimize)
    seen=Ref{Any}()
    builder=context->begin
        seen[]=context
        Lux.Chain(Lux.Dense(context.n_features=>5,tanh),Lux.Dense(5=>1),
            x->vec(x))
    end
    learner=LuxLearner(builder;name=:custom_tabular,training=TrainingSpec(
        epochs=1,batch_size=6,seed=7,patience=0),width=5)
    fitted=fit(learner,data;validation,validation_data=data,verbosity=-1)
    @test seen[].n_features==3
    @test seen[].config.width==5
    @test all(isfinite,predict(fitted,data).prediction)
end

@testset "Hybrid alignment and leakage guards" begin
    keys=DataFrame(group_id=repeat(1:2,inner=3),entity_id=repeat(1:3,2),fold_id=ones(Int,6))
    a=hcat(keys,DataFrame(prediction=[1.,2,3,3,2,1],prediction_scale=fill("target_scale",6)))
    b=hcat(keys,DataFrame(prediction=[3.,2,1,1,2,3]))
    blend=rank_blend([a,b];weights=[0.75,0.25])
    @test nrow(blend)==6
    @test all(blend.prediction_scale.=="dimensionless_rank")
    @test_throws ArgumentError fit_simplex_rank_stack([a,b],df->mean(df.prediction);
        contract_digest="x",inner_oof=false)
    stack=fit_simplex_rank_stack([a,b],df->mean(df.prediction);contract_digest="x",inner_oof=true)
    @test isapprox(sum(stack.weights),1;atol=1e-10)
    @test all(stack.weights .>= 0)
    @test_throws ArgumentError apply_rank_stack(stack,[a,b];contract_digest="wrong")
    target=hcat(keys,DataFrame(target=collect(1.0:6.0)))
    @test_throws ArgumentError make_residual_targets(a,target;contract_digest="x")
    residual=make_residual_targets(a,target;contract_digest="x",inner_oof=true)
    @test residual.inner_residual_target == target.target .- a.prediction
    residual_surface=hcat(keys,DataFrame(prediction=fill(0.1,6),
        prediction_scale=fill("target_scale_residual",6)))
    corrected=apply_residual_correction(a,residual_surface;contract_digest="x")
    @test corrected.prediction == a.prediction .+ 0.1
    @test_throws ArgumentError apply_residual_correction(
        select(a,Not(:prediction_scale)),residual_surface;contract_digest="x")
end
