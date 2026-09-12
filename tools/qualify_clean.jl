using Pkg

const ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(joinpath(ROOT, "qualification"))
Pkg.develop(PackageSpec(path=ROOT))
Pkg.instantiate()
Pkg.test("JoptunaLearners")
