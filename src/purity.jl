const _FORBIDDEN_RUNTIME_PATTERNS=(
    r"\bPythonCall\b",r"\bPyCall\b",r"\bCondaPkg\b",r"\bConda\b",
    r"\bMicroMamba\b",r"\bPython_jll\b",r"libpython",r"run\s*\([^\n]*(python|pip)",
    r"\.pt\b",r"\.pth\b",r"torch\.save",r"torch\.load",
)

function audit_runtime_purity(root::AbstractString=dirname(@__DIR__))
    violations=NamedTuple[]
    for dir in ("src","ext")
        path=joinpath(root,dir)
        isdir(path)||continue
        for (current,_,files) in walkdir(path), file in files
            endswith(file,".jl")||continue
            full=joinpath(current,file)
            basename(full)=="purity.jl" && continue
            for (line_no,line) in enumerate(eachline(full)), pattern in _FORBIDDEN_RUNTIME_PATTERNS
                occursin(pattern,line) && push!(violations,(file=relpath(full,root),line=line_no,pattern=string(pattern)))
            end
        end
    end
    isempty(violations)||throw(ArgumentError("forbidden Python/PyTorch runtime references: $violations"))
    true
end
