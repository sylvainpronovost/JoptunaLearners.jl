const TEST_REACTANT = get(ENV, "JOPTUNALEARNERS_TEST_REACTANT", "false") == "true"
const TEST_METAL = get(ENV, "JOPTUNALEARNERS_TEST_METAL", "false") == "true"

TEST_REACTANT && (@eval using Enzyme; @eval using Reactant)
TEST_METAL && (@eval using Metal)

using JoptunaLearners
using Lux
using Statistics
using Test

function dlinear_fixture(backend; epochs=2, batch_size=8)
    observations = 24
    lookback = 9
    windows = reshape(Float32.(1:(2 * lookback * observations)),
                      2, lookback, observations) ./ 100
    tabular = copy(windows[:, end, :])
    target = vec(0.3f0 .* windows[1, end, :] .- 0.1f0 .* windows[2, 1, :])
    data = LearnerData(tabular, target; windows)
    validation = ValidationSpec(:mse,
        (prediction, validation_data) ->
            mean((prediction .- validation_data.target) .^ 2);
        direction=:minimize)
    learner = LuxLearner(:window_dlinear;
        training=TrainingSpec(; backend, epochs, batch_size, seed=31,
                              patience=0, restore_best=true),
        kernel_size=5)
    learner, data, validation
end

function architecture_fixture(name, backend; epochs=1)
    spec = model_spec(name)
    observations = 8
    lookback = 12
    windows = reshape(Float32.(1:(3 * lookback * observations)),
                      3, lookback, observations) ./ 200
    tabular = copy(windows[:, end, :])
    target = vec(0.2f0 .* windows[1, end, :] .- 0.05f0 .* windows[2, 1, :])
    data = LearnerData(tabular, target;
        windows=spec.uses_windows ? windows : nothing,
        entity_codes=spec.uses_entity ? collect(1:observations) : nothing)
    validation = ValidationSpec(:mse,
        (prediction, validation_data) ->
            mean((prediction .- validation_data.target) .^ 2);
        direction=:minimize)
    learner = LuxLearner(name;
        training=TrainingSpec(; backend, epochs, batch_size=observations, seed=37,
                              patience=0, restore_best=true,
                              learning_rate=parse(Float64,
                                  get(ENV, "JOPTUNALEARNERS_LEARNING_RATE", "0.001")),
                              gradient_clip=parse(Float64,
                                  get(ENV, "JOPTUNALEARNERS_GRADIENT_CLIP", "1.0"))))
    learner, data, validation
end

function fit_architecture(name, backend)
    learner, data, validation = architecture_fixture(name, backend)
    fitted = fit(learner, data; validation, validation_data=data, verbosity=-1)
    (; fitted, data, predictions=predict(fitted, data).prediction,
       report=training_report(fitted))
end

event_contract(events) = [
    (epoch=event.epoch, update=event.update,
     training_loss=event.training_loss, validation_value=event.validation_value,
     contract_digest=event.contract_digest)
    for event in events
]

function fit_backend(backend; batch_size=8)
    learner, data, validation = dlinear_fixture(backend; batch_size)
    events = TrainingEvent[]
    started = time_ns()
    fitted = fit(learner, data; validation, validation_data=data,
                 callbacks=(event -> push!(events, event),), verbosity=-1)
    elapsed = (time_ns() - started) / 1e9
    (; fitted, data, events, elapsed, predictions=predict(fitted, data).prediction)
end

function resume_backend(backend)
    learner, data, validation = dlinear_fixture(backend; epochs=2)
    checkpoint = tempname() * ".jld2"
    interrupted = LuxLearner(
        learner.spec, learner.model_config,
        TrainingSpec(
            backend=backend, epochs=2, batch_size=8, seed=31,
            patience=0, restore_best=true, checkpoint_every=1,
        ),
    )
    fit(interrupted, data; validation, validation_data=data,
        checkpoint_path=checkpoint, verbosity=-1)
    continued = LuxLearner(
        learner.spec, learner.model_config,
        TrainingSpec(
            backend=backend, epochs=3, batch_size=8, seed=31,
            patience=0, restore_best=true, checkpoint_every=1,
        ),
    )
    resumed = fit(continued, data; validation, validation_data=data,
                  resume_from=checkpoint, verbosity=-1)
    uninterrupted = fit(continued, data; validation, validation_data=data, verbosity=-1)
    (; resumed, uninterrupted, data)
end

if get(ENV, "JOPTUNALEARNERS_QUALIFY_MODEL_ZOO", "false") == "true"
    selected = let raw=get(ENV, "JOPTUNALEARNERS_MODELS", "")
        isempty(raw) ? [spec.name for spec in model_specs()] : Symbol.(split(raw, ','))
    end
    backend = TEST_REACTANT ? ReactantCPU() : TEST_METAL ? MetalGPU() : EagerCPU()
    tolerance = TEST_METAL ? 5e-4 : TEST_REACTANT ? 1e-4 : 1e-6
    @testset "Complete model-zoo qualification: $(backend_capabilities(backend).name)" begin
        for name in selected
            @testset "$name" begin
                eager = fit_architecture(name, EagerCPU())
                accelerated = fit_architecture(name, backend)
                @test all(isfinite, accelerated.predictions)
                @test isfinite(accelerated.report.best_value)
                @test accelerated.report.completed_epochs == 1
                @test isapprox(accelerated.predictions, eager.predictions;
                               rtol=tolerance, atol=tolerance)
                @test execution_backend(accelerated.fitted) == backend
                replay = fit_architecture(name, backend)
                @test isapprox(replay.predictions, accelerated.predictions;
                               rtol=tolerance, atol=tolerance)
                @test event_contract(replay.report.events) ==
                      event_contract(accelerated.report.events)
                checkpoint = tempname() * ".jld2"
                save_checkpoint(checkpoint, accelerated.fitted)
                restored = load_checkpoint(checkpoint)
                @test isapprox(predict(restored, accelerated.data).prediction,
                               accelerated.predictions; rtol=tolerance, atol=tolerance)
                if get(ENV, "JOPTUNALEARNERS_QUALIFY_LIFECYCLE", "false") == "true"
                    learner, data, _ = architecture_fixture(name, backend; epochs=3)
                    constant_validation = ValidationSpec(
                        :constant, (_, _) -> 1.0; direction=:minimize,
                    )
                    training = learner.training
                    lifecycle_learner = LuxLearner(
                        learner.spec, learner.model_config,
                        TrainingSpec(
                            backend=backend, epochs=3, batch_size=training.batch_size,
                            seed=training.seed, patience=1, minimum_epochs=1,
                            restore_best=true, learning_rate=training.learning_rate,
                            gradient_clip=training.gradient_clip,
                        ),
                    )
                    lifecycle = fit(
                        lifecycle_learner, data; validation=constant_validation,
                        validation_data=data, verbosity=-1,
                    )
                    lifecycle_report = training_report(lifecycle)
                    @test lifecycle_report.stopped_early
                    @test lifecycle_report.completed_epochs == 2
                    @test lifecycle_report.best_epoch == 1
                    @test lifecycle_report.restored_best
                end
                if get(ENV, "JOPTUNALEARNERS_QUALIFY_MEMORY", "false") == "true"
                    GC.gc(true)
                    live_before = Base.gc_live_bytes()
                    second_replay = fit_architecture(name, backend)
                    all(isfinite, second_replay.predictions) || error("nonfinite replay")
                    GC.gc(true)
                    live_after = Base.gc_live_bytes()
                    @test live_after <= live_before + 16 * 1024^2
                end
            end
        end
    end
end

@testset "DLinear eager reference" begin
    eager = fit_backend(EagerCPU())
    @test all(isfinite, eager.predictions)
    @test length(eager.events) == 2
    @test execution_backend(eager.fitted) == EagerCPU()
end

if TEST_REACTANT
    @testset "DLinear Reactant CPU" begin
        eager = fit_backend(EagerCPU())
        accelerated = fit_backend(ReactantCPU())
        @test backend_capabilities(ReactantCPU()).available
        @test isapprox(accelerated.predictions, eager.predictions; rtol=1e-4, atol=1e-4)
        @test [event.epoch for event in accelerated.events] ==
              [event.epoch for event in eager.events]
        @test [event.contract_digest for event in accelerated.events] ==
              [event.contract_digest for event in eager.events]
        remainder = fit_backend(ReactantCPU(); batch_size=10)
        @test all(isfinite, remainder.predictions)
        recovery = resume_backend(ReactantCPU())
        @test isapprox(predict(recovery.resumed, recovery.data).prediction,
                       predict(recovery.uninterrupted, recovery.data).prediction;
                       rtol=1e-4, atol=1e-4)
        @test event_contract(training_report(recovery.resumed).events) ==
              event_contract(training_report(recovery.uninterrupted).events)
    end
end

if TEST_METAL
    @testset "DLinear Metal" begin
        @test Metal.functional()
        eager = fit_backend(EagerCPU())
        accelerated = fit_backend(MetalGPU())
        @test backend_capabilities(MetalGPU()).available
        @test isapprox(accelerated.predictions, eager.predictions; rtol=5e-4, atol=5e-4)
        @test [event.epoch for event in accelerated.events] ==
              [event.epoch for event in eager.events]
        @test [event.contract_digest for event in accelerated.events] ==
              [event.contract_digest for event in eager.events]
        recovery = resume_backend(MetalGPU())
        @test isapprox(predict(recovery.resumed, recovery.data).prediction,
                       predict(recovery.uninterrupted, recovery.data).prediction;
                       rtol=5e-4, atol=5e-4)
        @test event_contract(training_report(recovery.resumed).events) ==
              event_contract(training_report(recovery.uninterrupted).events)
    end
end
