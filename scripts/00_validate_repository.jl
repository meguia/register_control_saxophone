#!/usr/bin/env julia

"""
Fast integrity check for the archived publication repository.

The check does not recompute analyses. It verifies the deposited inventory,
session index, WAV containers, reviewed tables, and compact Figure 6 product.
"""

import Pkg

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(REPOSITORY_ROOT)

using CSV
using DataFrames
using JLD2

const EXPECTED_SUBJECTS =
    ["13", "22", "27", "33", "34", "37", "49",
     "50", "64", "70", "80", "83", "88", "90"]
const EXPECTED_TASKS =
    ["LegatoAsc", "LegatoDesc", "NonlegatoAsc", "NonlegatoDesc", "Overtone"]
const EXPECTED_REVIEW_TABLES = [
    "model_choices.csv",
    "model_notes.csv",
    "model_overtone_onoff.csv",
    "model_report.csv",
    "real_choices.csv",
    "real_notes.csv",
    "real_overtone_onoff.csv",
    "real_report.csv",
]

function repository_files(directory::AbstractString, extension::AbstractString)
    isdir(directory) || return String[]
    suffix = lowercase(extension)
    return sort!([joinpath(root, file)
                  for (root, _, files) in walkdir(directory)
                  for file in files
                  if endswith(lowercase(file), suffix)])
end

function is_wave_file(path::AbstractString)
    filesize(path) >= 12 || return false
    return open(path, "r") do io
        String(read(io, 4)) == "RIFF" &&
            (skip(io, 4); String(read(io, 4)) == "WAVE")
    end
end

function validate_repository()
    failures = String[]
    check(condition, message) = condition || push!(failures, message)

    required = [
        "Project.toml",
        "Manifest.toml",
        "README.md",
        "CITATION.cff",
        "LICENSE",
        "LICENSES/MIT.txt",
        "LICENSES/CC-BY-4.0.txt",
        "data/derived/multistability_map_v1.jld2",
        "src/impedances/alto/Dx4.jld2",
        "src/sessions/rt_sax_experiment_sept_2025.csv",
        ["scripts/0$(i)_$(name).jl" for (i, name) in enumerate((
            "postprocess",
            "wasserstein_distances",
            "permutation_tests",
            "recompute_figure6_map",
            "generate_figures",
        ))]...,
    ]
    for relative in required
        check(isfile(joinpath(REPOSITORY_ROOT, relative)),
              "missing required file: $(relative)")
    end

    audio_root = joinpath(REPOSITORY_ROOT, "audiofiles")
    wav_files = repository_files(audio_root, ".wav")
    expected_wavs = sort!([
        joinpath(audio_root, "S$(subject)",
                 "S$(subject)_$(condition)_B$(block)_$(task).wav")
        for subject in EXPECTED_SUBJECTS
        for condition in ("Model", "Real")
        for block in 1:2
        for task in EXPECTED_TASKS
    ])
    check(wav_files == expected_wavs,
          "WAV inventory differs from the expected 14 × 2 × 2 × 5 design")
    bad_wavs = filter(path -> !is_wave_file(path), wav_files)
    check(isempty(bad_wavs),
          "invalid RIFF/WAVE containers: $(join(relpath.(bad_wavs, REPOSITORY_ROOT), ", "))")

    sessions = joinpath(REPOSITORY_ROOT, "src", "sessions")
    logs = repository_files(sessions, ".log")
    dat_files = repository_files(sessions, ".dat")
    check(length(logs) == 56, "expected 56 block logs; found $(length(logs))")
    check(length(dat_files) == 675,
          "expected 675 indexed raw sensor files; found $(length(dat_files))")
    check(all(path -> filesize(path) > 0, logs), "one or more block logs are empty")
    check(all(path -> filesize(path) > 0, dat_files),
          "one or more raw sensor files are empty")

    index_path = joinpath(sessions, "rt_sax_experiment_sept_2025.csv")
    if isfile(index_path)
        index = CSV.read(index_path, DataFrame)
        check(nrow(index) == 56, "session index must contain 56 rows")
        check(Set(string.(index.subject)) == Set(EXPECTED_SUBJECTS),
              "session-index subject set differs from the paper cohort")
        check(Set(string.(index.type)) == Set(["Model", "Real"]),
              "session index must contain Model and Real conditions")
        indexed_logs = Set(joinpath(sessions, string(name)) for name in index.logfile)
        check(indexed_logs == Set(logs),
              "session index and deposited block logs do not match exactly")
    end

    referenced_dat = Set{String}()
    dat_pattern = r"S[0-9]+_(?:Model|Real)_Dx4_[A-Za-z]+_[0-9]+\.dat"
    for path in logs, match in eachmatch(dat_pattern, read(path, String))
        push!(referenced_dat, joinpath(sessions, match.match))
    end
    check(Set(dat_files) == referenced_dat,
          "deposited sensor files and filenames recorded by the block logs do not match exactly")

    review_root = joinpath(sessions, "reviewed_data")
    review_names = sort!(basename.(repository_files(review_root, ".csv")))
    check(review_names == sort(EXPECTED_REVIEW_TABLES),
          "reviewed-table inventory differs from the expected eight files")
    for name in review_names
        table = CSV.read(joinpath(review_root, name), DataFrame)
        check(nrow(table) > 0, "reviewed table is empty: $(name)")
    end

    all_deposited = vcat(wav_files, logs, dat_files,
                         repository_files(review_root, ".csv"))
    check(!any(path -> occursin(r"(^|[/_])S?97([/_\.]|$)", path), all_deposited),
          "excluded participant 97 is present in the deposit")

    map_path = joinpath(REPOSITORY_ROOT, "data", "derived",
                        "multistability_map_v1.jld2")
    if isfile(map_path)
        map = JLD2.load(map_path, "map")
        check(map.schema_version == 1, "unexpected Figure 6 map schema")
        check(length(map.gamma) == 389 && length(map.zeta) == 72,
              "unexpected Figure 6 parameter grid")
        expected_size = (length(map.zeta), length(map.gamma))
        for field in (:known, :t1, :t2, :other, :t1_high_p2,
                      :low_p1_high_p2)
            check(size(getproperty(map, field)) == expected_size,
                  "Figure 6 field $(field) has an inconsistent size")
        end
        check(count(map.known) == 28_008, "Figure 6 known-cell count changed")
        check(count(map.t1) == 16_635, "Figure 6 T1-cell count changed")
        check(count(map.t2) == 9_095, "Figure 6 T2-cell count changed")
        check(length(map.mixed_points) == 1_181,
              "Figure 6 mixed-edge sample count changed")
    end

    if isempty(failures)
        println("Repository validation passed")
        println("  subjects:       ", length(EXPECTED_SUBJECTS))
        println("  WAV files:      ", length(wav_files))
        println("  block logs:     ", length(logs))
        println("  sensor files:   ", length(dat_files))
        println("  reviewed tables:", length(review_names))
        println("  Figure 6 cells: 28,008")
        return true
    end

    println(stderr, "Repository validation failed:")
    foreach(message -> println(stderr, "  - ", message), failures)
    return false
end

validate_repository() || exit(1)
