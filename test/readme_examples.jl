@testset "Executable README quick start" begin
    source=read(joinpath(@__DIR__,"..","README.md"),String)
    blocks=collect(eachmatch(r"(?ms)^```julia\n(.*?)^```",source))
    @test !isempty(blocks)
    sandbox=Module(:ReadmeExample)
    for block in blocks
        Base.include_string(sandbox,block.captures[1],"README.md")
    end
    @test length(sandbox.prediction.prediction)==256
    @test all(isfinite,sandbox.prediction.prediction)
    @test sandbox.report.completed_epochs>0
end
