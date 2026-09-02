using JoptunaLearners
using Libdl

JoptunaLearners.audit_runtime_purity()

loaded = lowercase.(Libdl.dllist())
python_libraries = filter(path -> occursin("libpython", path), loaded)
isempty(python_libraries) || error("Python runtime libraries are loaded: $(join(python_libraries, ", "))")

println("JoptunaLearners runtime purity gate passed: no forbidden source references or loaded libpython.")
