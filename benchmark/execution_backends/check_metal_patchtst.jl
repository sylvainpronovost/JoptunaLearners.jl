using JSON3

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const RESULTS = get(ENV, "JOPTUNALEARNERS_METAL_PATCHTST_RESULTS",
    joinpath(ROOT, "benchmark", "execution_backends", "results", "metal-patchtst"))

function read_record(name)
    JSON3.read(read(joinpath(RESULTS, "$name.json"), String))
end

function read_prediction(name)
    bytes = read(joinpath(RESULTS, "$name.prediction.f32"))
    length(bytes) % sizeof(Float32) == 0 || error("$name prediction payload is malformed")
    copy(reinterpret(Float32, bytes))
end

function require_condition(condition, message)
    condition || error(message)
end

function main()
    eager = read_record("eager_cpu")
    metal = read_record("metal_gpu")
    for (name, record) in (("eager_cpu", eager), ("metal_gpu", metal))
        require_condition(record.backend == name, "$name record has the wrong backend")
        require_condition(record.model == "patchtst", "$name did not run PatchTSTLite")
        require_condition(record.observations == 4096 && record.epochs == 2 &&
                          record.repetitions >= 3 && record.features == 16 &&
                          record.lookback == 128 && record.batch_size == 2048 &&
                          record.seed == 191,
            "$name did not run the fixed compute-heavy profile")
        config = record.model_config
        require_condition(config.patch_len == 16 && config.stride == 8 &&
                          config.hidden_dim == 128 && config.depth == 4 &&
                          config.n_heads == 8 && config.dropout == 0,
            "$name did not use the fixed PatchTSTLite configuration")
        require_condition(record.warm_shape_growth_bytes <= 16 * 1024^2,
            "$name exceeds the warm-shape memory-growth ceiling")
        require_condition(0 < record.process_peak_rss_bytes <= 6 * 1024^3,
            "$name exceeds the 6 GiB process RSS ceiling")
        require_condition(isfinite(record.steady_state_seconds) && record.steady_state_seconds > 0,
            "$name has an invalid steady-state duration")
    end
    require_condition(metal.execution_provenance.fallback == "error",
        "Metal run did not have a fail-closed execution policy")
    device_samples = metal.device_allocated_after_fit_bytes
    require_condition(length(device_samples) == metal.repetitions + 1,
        "missing per-fit Metal driver memory samples")
    require_condition(all(x -> 0 <= x <= 6 * 1024^3, device_samples),
        "Metal post-GC driver sample exceeds the 6 GiB boundary limit")
    require_condition(maximum(device_samples[2:end]) - minimum(device_samples[2:end]) <= 16 * 1024^2,
        "Metal driver allocations grow across repeated fits")
    eager_prediction = read_prediction("eager_cpu")
    metal_prediction = read_prediction("metal_gpu")
    length(eager_prediction) == length(metal_prediction) == 4096 ||
        error("eager and Metal prediction lengths differ")
    all(isfinite, eager_prediction) && all(isfinite, metal_prediction) ||
        error("nonfinite prediction payload")
    maximum(abs.(eager_prediction .- metal_prediction)) <= 5f-4 ||
        error("Metal prediction mismatch exceeds 5e-4")
    ratio = Float64(metal.steady_state_seconds) / Float64(eager.steady_state_seconds)
    require_condition(ratio <= 0.5,
        "Metal PatchTSTLite did not reduce steady-state time by at least 50%")
    println("Metal PatchTSTLite performance PASS: Metal/eager steady-state ratio=" *
            string(round(ratio; digits=3)) * ", time reduction=" *
            string(round(100 * (1 - ratio); digits=1)) * "%")
end

main()
