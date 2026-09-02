module PublicationAudit
using SHA
using Test

# Fingerprints, not private identifiers, belong in a distributable policy.
# Normalize case and separators before matching. This is a regression guard,
# not a substitute for reviewing the origin and license of contributed code.
const BLOCKED = Set(split("""
a2aad5ac7c6057f508a76bd8b29e9ab8c5dcfdfbd4c55cade535c6bbb29d2e91
523c7c1840bf607b9a858a38960cbf5f693f72cd97dd1a65867f35e59bba9af5
e3528c2541074d2c2afe19fbf8dfe530463ec4266d455d24da2025f6bf579100
71432f6703149639e318825a082ab1364f862067f4e113113dc02d7783bbbfac
500cc515505e64f965a539b579dd66599169294172a494c328e6ae35c3921da1
d3c0d2873cb3fe8c618d463b4c7e67746f42ff017dbca927b2aa461d707bb217
154ff70f6c0c7e4d3402de240d102ebbb9672c801df80001242ad33d36407fb7
cc652e65b818cd43c45033ed899e76776f2f98365431c1fcfeff29ca24de9a2c
1a753feb5ef31c1240bca3d39272f1a8da34df0f5282a096405c92fd1181884b
4160847da0f1a3ab68c751f74b32d0cddfdb6e1c9bc9896491fe58e2d7427e4f
61bc8efdc014dffd7f7b384772f9aaebd49c1dff135475ae1e5cc5d58c543e01
397a821fa8bbd676a4a85ccfc0eac2f04bc07a4c38fd31d2530f5ea8aed6ebe8
f126383346171c87ecedda474b4ddf461956dbb551ebbe2a39863e566458c834
75c6c9ab0e481e3a8548190cb94ee1bd99f8edb6bd3cdb2fae22251da168ac00
e03d915ace47a8896cbf24449e1cd797bddb627233c535f352798003058c5335
9ce6d998c1c3b210ceda1f7c1da89c11a35188a007b5bd97249c9bcc4c7faf33
f5c7787b0d8587a465e271c123910fb01610991e302d01f4e7cb9772b1e3a77e
02adde942428e2f3b92a78e832d752eaf9918f7eee09ef227af5d30fe7e4f7a3
"""))
const HASH_CACHE = Dict{String,Bool}()
blocked(token) = get!(HASH_CACHE, token) do
    bytes2hex(sha256(token)) in BLOCKED
end
const BINARY_TRACES = Set([
    "benchmark/results/tpe-scaling-20260812/$(engine)-$(trials).json.trace.bin"
    for engine in ("julia", "python") for trials in (2000, 10000)
])
const TRACE_HASHES = Dict(
    "julia-10000" => "73504b719b819aaf53ffa8f9ffea2da27e0076ae75ac766397cc3af456d72486",
    "julia-2000" => "473038baf1453e012eda222a702f4fcfb41f147b05e91220302e1d727c07e8c5",
    "python-10000" => "b06889e509c05dc8f945ab0ae7221d816808741aa90788e64516a6a66bd82f0a",
    "python-2000" => "225548e0822e6eb53762bf7a94d9e06c2d6dca34e3c578e59cac659daf66de10",
)
const ORDINARY_WORDS = Set(["evolves", "evolve", "evolved", "evolving", "evolution", "evolutionary"])

function text_issues(text)
    issues = String[]
    for word in eachmatch(r"[A-Za-z][A-Za-z0-9_-]*", text)
        token = lowercase(replace(word.match, r"[_-]" => ""))
        blocked(token) && push!(issues, "restricted identifier fingerprint")
        if length(token) > 3 && startswith(token, "evo") && !startswith(token, "evotree") && !(token in ORDINARY_WORDS)
            push!(issues, "nonpublic legacy-prefixed identifier")
        end
        # Also inspect individual underscore/hyphen components.
        for part in split(lowercase(word.match), r"[_-]")
            blocked(part) && push!(issues, "restricted identifier component")
        end
    end
    occursin(r"/(?:Users|home)/(?!runner/|user/|username/)[A-Za-z0-9_.-]+/", text) &&
        push!(issues, "personal absolute path")
    occursin(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", text) &&
        push!(issues, "private key material")
    occursin(r"\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|AKIA[A-Z0-9]{16})\b", text) &&
        push!(issues, "credential-shaped token")
    unique(issues)
end

function inspect(path, data)
    issues = text_issues(path)
    # Raw data, checkpoints and opaque archives must be explicitly reviewed,
    # rather than slipping through because a text scanner cannot read them.
    if occursin(r"(?i)\.(arrow|parquet|jld2|bson|sqlite3?|db|pkl|pickle|pt|pth|zip|tar|gz|bundle)$", path)
        push!(issues, "unapproved dataset/checkpoint/archive artifact")
    end
    if path in BINARY_TRACES
        key = first(split(basename(path), "."))
        bytes2hex(sha256(data)) == TRACE_HASHES[key] || push!(issues, "unreviewed synthetic trace digest")
        count = occursin("10000", path) ? 10000 : 2000
        length(data) == count * 5 * 8 || push!(issues, "invalid synthetic trace size")
        if length(data) == count * 5 * 8
            values = reinterpret(Float64, data)
            all(v -> isfinite(v) && -5 <= v <= 5, values) ||
                push!(issues, "invalid synthetic trace value")
        end
    elseif 0x00 in data || !isvalid(String(copy(data)))
        push!(issues, "unreviewed binary content")
    else
        append!(issues, text_issues(String(copy(data))))
    end
    unique(issues)
end

git(root, args...) = read(`git -C $root $args`)
paths(bytes) = filter(!isempty, split(String(bytes), '\0'))

function audit(root; history=false, generated=false)
    failures = String[]
    files = paths(git(root, "ls-files", "-z"))
    count = 0
    for path in files
        file = joinpath(root, path)
        issues = islink(file) ? ["tracked symlink requires review"] : inspect(path, read(file))
        append!(failures, ["$path: $issue" for issue in issues])
        count += 1
    end
    blobs = 0
    if history
        seen = Set{String}()
        for commit in split(String(git(root, "rev-list", "--all")))
            for entry in paths(git(root, "ls-tree", "-r", "-z", commit))
                info, path = split(entry, '\t'; limit=2)
                mode, kind, oid = split(info)
                kind == "blob" || continue
                oid in seen && continue
                push!(seen, oid)
                issues = mode == "120000" ? ["historical symlink requires review"] :
                    inspect(path, git(root, "cat-file", "blob", oid))
                append!(failures, ["history $path: $issue" for issue in issues])
            end
        end
        blobs = length(seen)
        append!(failures, text_issues(String(git(root, "log", "--all", "--format=fuller"))))
    end
    generated_count = 0
    if generated
        # Audit first-party rendered documentation. Installed dependencies and
        # local environment manifests are not publication inputs.
        for directory in (joinpath(root, "docs", "build"), joinpath(root, "tutorial-comparisons", "_site"))
            isdir(directory) || continue
            for (base, _, entries) in walkdir(directory), entry in entries
                file = joinpath(base, entry)
                any(suffix -> endswith(lowercase(file), suffix), (".html", ".js", ".json", ".md", ".css")) || continue
                append!(failures, ["generated $(relpath(file, root)): $issue" for issue in text_issues(read(file, String))])
                generated_count += 1
            end
        end
    end
    foreach(println, unique(failures))
    isempty(failures) || error("publication audit failed with $(length(unique(failures))) findings")
    println("Publication audit passed: $count tracked files, $blobs history blobs, $generated_count generated files")
end

function selftest()
    @testset "Publication audit policy" begin
        @test isempty(text_issues("JoptunaRegressor EvoTreesPruningCallback evolves"))
        @test !isempty(text_issues(join(("internal", "fixture", "project"))))
        @test !isempty(text_issues(join(("INTERNAL", "FIXTURE", "PROJECT"), "_")))
        @test !isempty(text_issues(join(("Ev", "o", "PrivateFixture"))))
        @test !isempty(text_issues(joinpath("/", "Users", "testperson", "project")))
        @test !isempty(inspect("weights.jld2", UInt8[1, 2, 3]))
        @test !isempty(inspect("data.bin", UInt8[0, 1, 2]))
        @test isempty(inspect("example.csv", collect(codeunits("x,y\n1,2\n"))))
        mktempdir() do root
            git(root, "init", "-q")
            git(root, "config", "user.name", "Publication policy test")
            git(root, "config", "user.email", "policy@example.invalid")
            file = joinpath(root, "fixture.txt")
            write(file, join(("internal", "fixture", "project")))
            git(root, "add", "fixture.txt")
            git(root, "commit", "-qm", "Add policy fixture")
            @test_throws ErrorException audit(root)
            write(file, "public synthetic fixture")
            git(root, "add", "fixture.txt")
            git(root, "commit", "-qm", "Clear working tree fixture")
            @test audit(root) === nothing
            @test_throws ErrorException audit(root; history=true)
        end
    end
end

function main(args)
    "--self-test" in args && selftest()
    root = normpath(joinpath(@__DIR__, ".."))
    audit(root; history="--history" in args, generated="--generated" in args)
end
end
if abspath(PROGRAM_FILE) == @__FILE__
    PublicationAudit.main(ARGS)
end
