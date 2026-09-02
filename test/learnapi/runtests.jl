using JoptunaLearners
using LearnTestAPI
using Statistics

validation = ValidationSpec(
    :mse,
    (prediction, data) -> mean((prediction .- data.target) .^ 2);
    direction = :minimize,
    prediction = :prediction,
    target = :target,
    grouping = (:observation,),
)
learner = LuxLearner(
    :mlp;
    validation,
    training = TrainingSpec(epochs=2, batch_size=4, seed=17, patience=0),
    hidden_dims = (4,),
)
x = Float32[1 2 3 4 5 6 7 8; 2 1 0 -1 -2 -3 -4 -5]
y = vec(0.5f0 .* x[1, :] .- 0.2f0 .* x[2, :])
data = LearnerData(x, y)

@testapi learner data verbosity=0
