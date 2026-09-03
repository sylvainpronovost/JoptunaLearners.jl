using Test, Random, Zygote, JoptunaLearners
using JoptunaLearners: NNlib

# Independent, query-at-a-time reference retained only for small regression cases.
function attention_reference(query, source, p, n_heads)
    hidden, n_query, batch = size(query)
    n_source = size(source, 2)
    width = hidden ÷ n_heads
    linear(x, p) = reshape(p.weight * reshape(x, size(x, 1), :) .+ p.bias,
                           size(p.weight, 1), size(x, 2), size(x, 3))
    q, k, v = linear(query, p.q), linear(source, p.k), linear(source, p.v)
    heads = ntuple(n_heads) do head
        rows = (head-1)*width+1:head*width
        qh, kh, vh = q[rows,:,:], k[rows,:,:], v[rows,:,:]
        contexts = ntuple(n_query) do position
            scores = dropdims(sum(reshape(qh[:,position,:], width, 1, batch) .* kh;
                                   dims=1); dims=1) ./ sqrt(Float32(width))
            weights = NNlib.softmax(scores; dims=1)
            reshape(dropdims(sum(vh .* reshape(weights, 1, n_source, batch);
                                dims=2); dims=2), width, 1, batch)
        end
        cat(contexts...; dims=2)
    end
    linear(cat(heads...; dims=1), p.out)
end

attention_leaves(x::AbstractArray) = [x]
attention_leaves(x::NamedTuple) = reduce(vcat, attention_leaves.(values(x)))

@testset "Batched attention preserves values and full gradients" begin
    rng = MersenneTwister(206)
    for T in (Float32, Float64), (hidden, heads, nq, ns, batch) in
            ((8, 1, 3, 3, 2), (8, 2, 4, 7, 3), (8, 4, 1, 5, 1))
        dense() = (weight=randn(rng,T,hidden,hidden) ./ T(4),
                   bias=randn(rng,T,hidden,1) ./ T(4))
        p = (q=dense(), k=dense(), v=dense(), out=dense())
        q, s = randn(rng,T,hidden,nq,batch), randn(rng,T,hidden,ns,batch)
        reference = attention_reference(q,s,p,heads)
        actual = JoptunaLearners._multihead_attention(q,s,p,heads)
        tol = T === Float32 ? 2e-5 : 1e-11
        @test size(actual) == (hidden,nq,batch)
        @test actual ≈ reference rtol=tol atol=tol
        g_ref = Zygote.gradient((q,s,p)->sum(abs2,attention_reference(q,s,p,heads)),q,s,p)
        g_new = Zygote.gradient((q,s,p)->sum(abs2,JoptunaLearners._multihead_attention(q,s,p,heads)),q,s,p)
        for (a,b) in zip(g_new,g_ref), (x,y) in zip(attention_leaves(a),attention_leaves(b))
            @test x ≈ y rtol=tol atol=tol
        end
    end
end

@testset "Batched attention bounds reverse-mode allocations" begin
    rng = MersenneTwister(207)
    hidden, heads, tokens, batch = 32, 4, 9, 16
    dense() = (weight=randn(rng,Float32,hidden,hidden) ./ 8,
               bias=zeros(Float32,hidden,1))
    p = (q=dense(),k=dense(),v=dense(),out=dense())
    x = randn(rng,Float32,hidden,tokens,batch)
    grad(f) = Zygote.gradient(p -> sum(abs2,f(x,x,p,heads)),p)
    grad(attention_reference)
    grad(JoptunaLearners._multihead_attention)
    old_bytes = @allocated grad(attention_reference)
    new_bytes = @allocated grad(JoptunaLearners._multihead_attention)
    @info "Attention allocation comparison" old_bytes new_bytes
    @test new_bytes < old_bytes ÷ 2
end
