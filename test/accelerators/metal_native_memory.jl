@testset "Metal native batched-call resources are released" begin
    x = ones(Float32,64,64,256)
    model(input,ps,st) = (vec(sum(JoptunaLearners.NNlib.batched_mul(input,input);
                                 dims=(1,2))), st)
    samples = Int[]
    for _ in 1:6
        prediction = JoptunaLearners._backend_predict(MetalGPU(),model,nothing,nothing,x)
        @test all(==(64.0^3),prediction)
        GC.gc(true); Metal.synchronize(); GC.gc(true)
        push!(samples,Int(Metal.device().currentAllocatedSize))
    end
    # Without the native autorelease boundary, this fixture retained roughly
    # 8 MiB per call even after full Julia collections and device synchronization.
    @test maximum(samples[2:end])-minimum(samples[2:end]) <= 1024^2

    function failing_model(input,ps,st)
        JoptunaLearners.NNlib.batched_mul(input,input)
        error("intentional native-resource exception")
    end
    empty!(samples)
    for _ in 1:4
        @test_throws ErrorException JoptunaLearners._backend_predict(
            MetalGPU(),failing_model,nothing,nothing,x)
        GC.gc(true); Metal.synchronize(); GC.gc(true)
        push!(samples,Int(Metal.device().currentAllocatedSize))
    end
    @test maximum(samples)-minimum(samples) <= 1024^2
end
