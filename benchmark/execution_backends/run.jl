using JoptunaLearners
using JSON3
using SHA
using Statistics

const BACKEND_NAME = get(ENV, "JOPTUNALEARNERS_BENCHMARK_BACKEND", "eager_cpu")
if BACKEND_NAME == "reactant_cpu"
    import Enzyme
    import Reactant
elseif BACKEND_NAME == "metal_gpu"
    import Metal
elseif BACKEND_NAME != "eager_cpu"
    error("backend must be eager_cpu, reactant_cpu, or metal_gpu")
end

backend = BACKEND_NAME == "reactant_cpu" ? ReactantCPU() :
          BACKEND_NAME == "metal_gpu" ? MetalGPU() : EagerCPU()
model_name = Symbol(get(ENV, "JOPTUNALEARNERS_BENCHMARK_MODEL", "window_dlinear"))
observations = parse(Int, get(ENV, "JOPTUNALEARNERS_BENCHMARK_OBSERVATIONS", "4096"))
epochs = parse(Int, get(ENV, "JOPTUNALEARNERS_BENCHMARK_EPOCHS", "3"))
repetitions = parse(Int, get(ENV, "JOPTUNALEARNERS_BENCHMARK_REPETITIONS", "3"))
output = abspath(get(ENV, "JOPTUNALEARNERS_BENCHMARK_OUTPUT",
    joinpath(@__DIR__, "results", "$BACKEND_NAME.json")))
prediction_output = get(ENV, "JOPTUNALEARNERS_BENCHMARK_PREDICTION_OUTPUT", "")

lookback = parse(Int, get(ENV, "JOPTUNALEARNERS_BENCHMARK_LOOKBACK", "60"))
features = parse(Int, get(ENV, "JOPTUNALEARNERS_BENCHMARK_FEATURES", "4"))
windows = reshape(
    sin.(Float32.(1:(features * lookback * observations)) ./ 37),
    features, lookback, observations,
)
tabular = copy(windows[:, end, :])
target = vec(0.4f0 .* windows[1, end, :] .- 0.2f0 .* windows[2, 1, :])
spec = model_spec(model_name)
data = LearnerData(tabular, target;
    windows=spec.uses_windows ? windows : nothing,
    entity_codes=spec.uses_entity ? mod1.(collect(1:observations), 32) : nothing,
)
validation = ValidationSpec(
    :mse, (prediction, candidate) -> mean((prediction .- candidate.target) .^ 2);
    direction=:minimize,
)
batch_size = parse(Int, get(ENV, "JOPTUNALEARNERS_BENCHMARK_BATCH_SIZE", "512"))
training = TrainingSpec(
    backend=backend, epochs=epochs, batch_size=batch_size, seed=191,
    patience=0, restore_best=false,
)
# The default is the existing DLinear smoke.  The explicit PatchTST profile is
# deliberately compute-heavy enough to make a GPU performance decision meaningful.
model_kwargs = model_name === :patchtst ?
    (; patch_len=parse(Int, get(ENV, "JOPTUNALEARNERS_BENCHMARK_PATCH_LEN", "16")),
       stride=parse(Int, get(ENV, "JOPTUNALEARNERS_BENCHMARK_STRIDE", "8")),
       hidden_dim=parse(Int, get(ENV, "JOPTUNALEARNERS_BENCHMARK_HIDDEN_DIM", "128")),
       depth=parse(Int, get(ENV, "JOPTUNALEARNERS_BENCHMARK_DEPTH", "4")),
       n_heads=parse(Int, get(ENV, "JOPTUNALEARNERS_BENCHMARK_HEADS", "8")), dropout=0.0) :
    model_name === :tcn ?
    (; hidden_dim=parse(Int, get(ENV, "JOPTUNALEARNERS_BENCHMARK_HIDDEN_DIM", "128")),
       depth=parse(Int, get(ENV, "JOPTUNALEARNERS_BENCHMARK_DEPTH", "4")),
       kernel_size=parse(Int, get(ENV, "JOPTUNALEARNERS_BENCHMARK_KERNEL_SIZE", "3")),
       dropout=0.0, revin=false, use_weight_norm=false) :
    model_name === :window_dlinear ? (; kernel_size=9,) : NamedTuple()
learner = LuxLearner(model_name; training, model_kwargs...)

function measured_fit()
    measurement = @timed fit(
        learner, data; validation, validation_data=data, verbosity=-1,
    )
    prediction = predict(measurement.value, data).prediction
    all(isfinite, prediction) || error("benchmark produced nonfinite predictions")
    # Qualification needs scalar timing/allocation fields and one prediction surface, not a
    # retained fitted object per repetition. Releasing it here prevents the benchmark itself from
    # turning a bounded large-batch workload into cumulative live-memory pressure.
    (; seconds=measurement.time, allocated_bytes=measurement.bytes,
       gc_seconds=measurement.gctime, prediction,
       provenance=execution_provenance(measurement.value))
end

process_started = time_ns()
cold = measured_fit()
GC.gc(true)
live = Int[Base.gc_live_bytes()]
warm = NamedTuple[]
for _ in 1:repetitions
    push!(warm, measured_fit())
    GC.gc(true)
    push!(live, Base.gc_live_bytes())
end
tolerance = backend isa MetalGPU ? 5e-4 : backend isa ReactantCPU ? 1e-4 : 0.0
all(run -> isapprox(run.prediction, first(warm).prediction;
                    rtol=tolerance, atol=tolerance), warm) ||
    error("same-backend replay changed predictions")
warm_growth = maximum(live[2:end]) - minimum(live[2:end])
warm_growth <= 16 * 1024^2 || error(
    "warm-shape managed live memory grew by $(warm_growth) bytes",
)

mkpath(dirname(output))
if !isempty(prediction_output)
    mkpath(dirname(abspath(prediction_output)))
    open(prediction_output, "w") do io
        write(io, Float32.(first(warm).prediction))
    end
end
device_memory = BACKEND_NAME == "metal_gpu" ? Int(Metal.device().currentAllocatedSize) : 0
record = (
    schema_version=1,
    backend=BACKEND_NAME,
    model=String(model_name),
    model_config=learner.model_config,
    observations,
    epochs,
    repetitions,
    cold_seconds=cold.seconds,
    cold_allocated_bytes=cold.allocated_bytes,
    cold_gc_seconds=cold.gc_seconds,
    steady_state_seconds=median(getproperty.(warm, :seconds)),
    steady_state_allocated_bytes=median(getproperty.(warm, :allocated_bytes)),
    steady_state_gc_seconds=median(getproperty.(warm, :gc_seconds)),
    process_total_seconds=(time_ns() - process_started) / 1e9,
    managed_live_bytes=live,
    warm_shape_growth_bytes=warm_growth,
    process_peak_rss_bytes=Sys.maxrss(),
    device_allocated_bytes=device_memory,
    execution_provenance=first(warm).provenance,
    prediction_digest=bytes2hex(sha256(reinterpret(UInt8, first(warm).prediction))),
    julia_version=string(VERSION),
)
open(output, "w") do io
    JSON3.pretty(io, record)
    println(io)
end
println(JSON3.write(record))
