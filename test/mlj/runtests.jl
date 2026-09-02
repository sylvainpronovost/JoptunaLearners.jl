using JoptunaLearners
using MLJBase
using MLJModelInterface
using MLJTestInterface
using StatisticalMeasures: rms
using Statistics
using Test

const MMI = MLJModelInterface
const ACCELERATOR_BACKEND = get(ENV, "JOPTUNALEARNERS_MLJ_BACKEND", "")
if ACCELERATOR_BACKEND == "reactant_cpu"
    import Enzyme
    import Reactant
elseif ACCELERATOR_BACKEND == "metal_gpu"
    import Metal
elseif !isempty(ACCELERATOR_BACKEND)
    error("JOPTUNALEARNERS_MLJ_BACKEND must be eager_cpu, reactant_cpu, or metal_gpu")
end

function selected_backend()
    ACCELERATOR_BACKEND == "reactant_cpu" && return ReactantCPU()
    ACCELERATOR_BACKEND == "metal_gpu" && return MetalGPU()
    EagerCPU()
end

@testset "MLJ deterministic-regressor adapter" begin
    X = (a=Float32.(1:32), b=Float32.(32:-1:1))
    y = 0.7f0 .* X.a .- 0.2f0 .* X.b
    validation = ValidationSpec(:mse, (p, d) -> mean((p .- d.target) .^ 2);
        direction=:minimize)
    native = LuxLearner(:mlp; training=TrainingSpec(
        epochs=2, batch_size=8, seed=5, patience=1, restore_best=true))

    model = mlj_model(native; validation)
    @test MMI.clean!(model) == ""
    @test MMI.supports_weights(typeof(model))
    @test MMI.supports_training_losses(typeof(model))

    reformatted = MMI.reformat(model, X, y, ones(Float32, length(y)))
    @test length(reformatted[1]) == length(y)
    selected = MMI.selectrows(model, 1:12, reformatted...)
    @test length(selected[1]) == 12

    fitted, cache, report = MMI.fit(model, 0, reformatted...)
    prediction = MMI.predict(model, fitted, reformatted[1])
    @test length(prediction) == 32
    @test all(isfinite, prediction)
    @test report.training_report.contract_digest == validation.digest
    @test !report.selection_enabled
    @test report.training_report.stopped_early == false
    @test length(MMI.training_losses(model, report)) == 2
    parameters = MMI.fitted_params(model, fitted)
    @test parameters.architecture == :MLPRegressor
    @test parameters.parameter_count > 0

    reordered = (b=X.b, a=X.a)
    @test MMI.predict(model, fitted, reordered) == prediction
    @test_throws ArgumentError MMI.predict(model, fitted, (a=X.a, c=X.b))
    @test_throws ArgumentError MMI.fit(model, 0,
        (a=Union{Missing,Float32}[missing; X.a[2:end]],), y)

    updated, _, updated_report = MMI.update(model, 0, fitted, cache, reformatted...)
    @test length(MMI.predict(model, updated, reformatted[1])) == 32
    @test updated_report.selection_enabled == false
end

@testset "MLJ lifecycle on fixed execution backend" begin
    backend = selected_backend()
    X = (a=Float32.(1:24) ./ 10, b=Float32.(24:-1:1) ./ 10)
    y = 0.4f0 .* X.a .- 0.15f0 .* X.b
    learner = LuxLearner(:mlp; training=TrainingSpec(
        backend=backend, epochs=2, batch_size=8, seed=79, patience=0,
        restore_best=false,
    ))
    model = JoptunaRegressor(learner=learner)
    mach = machine(model, X, y)
    fit!(mach; verbosity=-1)
    first_prediction = MLJBase.predict(mach, X)
    @test all(isfinite, first_prediction)
    @test execution_backend(mach.fitresult) == backend
    @test report(mach).training_report.provenance.execution.execution_backend ==
          String(backend_capabilities(backend).name)

    # MLJ update deliberately performs a fresh statistical fit while the backend
    # may safely reuse its process-level compilation cache for an identical shape.
    fit!(mach; verbosity=-1)
    second_prediction = MLJBase.predict(mach, X)
    tolerance = backend isa MetalGPU ? 5e-4 : backend isa ReactantCPU ? 1e-4 : 0.0
    @test isapprox(second_prediction, first_prediction; rtol=tolerance, atol=tolerance)

    evaluation = evaluate!(mach; resampling=CV(nfolds=2, shuffle=false),
                           measure=rms, verbosity=-1)
    @test length(evaluation.per_fold[1]) == 2
    @test all(isfinite, evaluation.per_fold[1])
end

@testset "MLJ internal validation is bounded inside the training fold" begin
    X = (a=randn(Float32, 30), b=randn(Float32, 30))
    y = X.a .- 0.1f0 .* X.b
    model = JoptunaLearners.JoptunaRegressor(
        learner=LuxLearner(:mlp; training=TrainingSpec(epochs=3, batch_size=10, seed=17,
            patience=1, restore_best=true)), validation_fraction=0.2)
    fitted, _, report = MMI.fit(model, 0, X, y)
    @test report.selection_enabled
    @test report.training_rows + report.validation_rows == length(y)
    @test report.validation_rows == 6
    @test fitted.context.mlj_selection_enabled
    @test !isempty(MMI.clean!(JoptunaLearners.JoptunaRegressor(validation_fraction=1.0)))
end

@testset "MLJBase machine lifecycle" begin
    X = (a=randn(Float32, 24), b=randn(Float32, 24))
    y = X.a .+ 0.3f0 .* X.b
    model = JoptunaLearners.JoptunaRegressor(learner=LuxLearner(:mlp; training=TrainingSpec(
        epochs=2, batch_size=8, seed=9, patience=0)))
    mach = machine(model, X, y)
    fit!(mach; verbosity=-1)
    @test length(MLJBase.predict(mach, X)) == length(y)
    @test report(mach).training_rows == length(y)
    @test fitted_params(mach).parameter_count > 0
    fit!(mach; verbosity=-1)
    @test length(MLJBase.predict(mach; rows=1:4)) == 4
end

@testset "MLJTestInterface level-one traits" begin
    X, y = MLJTestInterface.make_regression()
    failures, _ = MLJTestInterface.test([JoptunaLearners.JoptunaRegressor], X, y;
        mod=JoptunaLearners, level=1, throw=false, verbosity=-1)
    @test isempty(failures)
end
