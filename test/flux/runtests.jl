using JoptunaLearners
using Flux
using Statistics
using Test

@testset "optional Flux adapter" begin
    x=randn(Float32,3,24); y=vec(x[1,:].-x[2,:])
    data=LearnerData(x,y)
    validation=ValidationSpec(:mse,(p,d)->mean((p.-d.target).^2);direction=:minimize)
    learner=FluxLearner(n->Flux.Chain(Flux.Dense(n=>6,Flux.relu),Flux.Dense(6=>1));
        training=TrainingSpec(epochs=2,batch_size=8,seed=4,patience=0))
    fitted=fit(learner,data;validation,validation_data=data,verbosity=-1)
    @test length(training_report(fitted).events)==2
    @test all(isfinite,predict(fitted,data).prediction)
end
