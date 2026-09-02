module JoptunaLearnersFluxExt

using JoptunaLearners
using Flux
using LearnAPI
using Optimisers
using Random

import JoptunaLearners: fit, predict

function fit(learner::JoptunaLearners.FluxLearner, data::JoptunaLearners.LearnerData;
             validation::JoptunaLearners.ValidationSpec,
             validation_data::JoptunaLearners.LearnerData=data,
             context=nothing, callbacks=(), verbosity=LearnAPI.default_verbosity())
    data.windows === nothing || throw(ArgumentError("the generic Flux adapter currently accepts tabular inputs; user-owned window routing belongs in the builder"))
    train=learner.training
    Random.seed!(train.seed)
    model=applicable(learner.builder,size(data.tabular,1),context) ?
        learner.builder(size(data.tabular,1),context) : learner.builder(size(data.tabular,1))
    opt=Optimisers.AdamW(Float32(train.learning_rate),(0.9f0,0.999f0),Float32(train.weight_decay);couple=false)
    opt_state=Flux.setup(opt,model)
    events=JoptunaLearners.TrainingEvent[]
    best_value=JoptunaLearners._initial_best(validation.direction)
    best_epoch=0
    best_model=deepcopy(model)
    stale=0; update=0; stopped=false; started=time()
    rng=MersenneTwister(train.seed)
    indices=collect(1:length(data))
    for epoch in 1:train.epochs
        shuffle!(rng,indices)
        epoch_loss=0.0; seen=0
        for first in 1:train.batch_size:length(indices)
            idx=indices[first:min(first+train.batch_size-1,end)]
            loss,grads=Flux.withgradient(model) do current
                pred=vec(current(data.tabular[:,idx]))
                JoptunaLearners._weighted_huber(pred,data.target[idx],data.weights[idx],train.huber_delta)
            end
            isfinite(loss)||throw(ArgumentError("nonfinite Flux training loss"))
            Flux.update!(opt_state,model,grads[1])
            update+=1; epoch_loss+=Float64(loss)*length(idx); seen+=length(idx)
        end
        pred=Float64.(vec(model(validation_data.tabular)))
        value=JoptunaLearners._validation_value(validation,pred,validation_data)
        event=JoptunaLearners.TrainingEvent(epoch,update,epoch_loss/seen,value,time()-started,validation.digest)
        push!(events,event); foreach(cb->cb(event),callbacks)
        if JoptunaLearners._better(validation.direction,value,best_value,train.min_delta)
            best_value=value; best_epoch=epoch; best_model=deepcopy(model); stale=0
        else
            stale+=1
        end
        if train.patience>0 && stale>=train.patience; stopped=true; break; end
        verbosity>0 && @info "JoptunaLearners Flux epoch" epoch value
    end
    final_model=train.restore_best ? best_model : model
    provenance=(package="JoptunaLearners.jl",backend="Flux",julia=string(VERSION),model="user_owned",
                contract_digest=validation.digest,seed=train.seed,
                context=context===nothing ? NamedTuple() : context)
    report=JoptunaLearners.TrainingReport(:flux_adapter,:complete,best_epoch,best_value,length(events),events,
        validation.digest,train.seed,stopped,train.restore_best,provenance)
    JoptunaLearners.FittedLearner(learner,final_model,nothing,nothing,opt_state,report,
        context===nothing ? NamedTuple() : context)
end

function predict(fitted::JoptunaLearners.FittedLearner{<:JoptunaLearners.FluxLearner}, data::JoptunaLearners.LearnerData)
    values=Float64.(vec(fitted.model(data.tabular)))
    JoptunaLearners.PredictionSurface(copy(data.keys),values,:flux_adapter,
        fitted.report.contract_digest,:model_output,fitted.report.provenance)
end

LearnAPI.predict(fitted::JoptunaLearners.FittedLearner{<:JoptunaLearners.FluxLearner}, ::LearnAPI.Point,
                 data::JoptunaLearners.LearnerData)=predict(fitted,data)

end
