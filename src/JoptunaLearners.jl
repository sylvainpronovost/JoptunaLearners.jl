module JoptunaLearners

using ADTypes
using Dates
using DataFrames
using JLD2
using LearnAPI
using LinearAlgebra
using Lux
using MLJModelInterface
using NNlib
using Optim
using Optimisers
using Random
using SHA
using Serialization
using Statistics
using StatsBase
using Tables
using Zygote

import LearnAPI: fit, predict

"""Return the MLJ-compatible wrapper for a native `LuxLearner`."""
function mlj_model end

include("backends.jl")
include("contracts.jl")
include("registry.jl")
include("data.jl")
include("architectures.jl")
include("training.jl")
include("mlj.jl")
include("checkpoints.jl")
include("hybrids.jl")
include("purity.jl")

export ModelSpec, TrainingSpec, ValidationSpec, TrainingEvent, TrainingReport
export ExecutionBackend, EagerCPU, ReactantCPU, MetalGPU
export execution_backend, backend_capabilities, require_performance_qualified
export execution_provenance
export LearnerData, CausalWindowSource, PredictionSurface, LuxLearner, FluxLearner, FittedLearner
export fit, predict, training_report, save_checkpoint, load_checkpoint
export model_spec, model_specs, model_config, hyperparameter_schema, build_model
export prepare_tabular, prepare_windows, fit_normalizer, apply_normalizer
export rank_blend, fit_simplex_rank_stack, apply_rank_stack
export make_residual_targets, apply_residual_correction
export assert_contract, contract_digest, audit_runtime_purity
export mlj_model, JoptunaRegressor

end
