using Documenter
using JoptunaLearners

makedocs(
    modules = [JoptunaLearners],
    sitename = "JoptunaLearners.jl",
    format = Documenter.HTML(edit_link = nothing, repolink = nothing),
    remotes = nothing,
    checkdocs = :none,
    warnonly = false,
    pages = [
        "Overview" => "index.md",
        "Ownership and contracts" => "ownership.md",
        "Training and pruning" => "training-and-pruning.md",
        "Execution backends" => "execution-backends.md",
        "MLJ integration" => "mlj.md",
        "Hybrid methods" => "hybrid-methods.md",
        "Model inventory" => "model-ledger.md",
    ],
)
