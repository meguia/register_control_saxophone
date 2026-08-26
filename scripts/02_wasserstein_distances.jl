#!/usr/bin/env julia

"""Compute the Wasserstein summaries used in paper Figure 5."""

import Pkg
const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(PROJECT_ROOT)

using CSV
using DataFrames
using JLD2

const SRC_DIR = joinpath(PROJECT_ROOT, "src")
const PROCESSED_DIR = joinpath(SRC_DIR, "sessions", "processed_data")
const MODEL_FILE = joinpath(
    PROCESSED_DIR, "all_model_trials_wamplitudes_onoff.jld2")
const REAL_FILE = joinpath(
    PROCESSED_DIR, "all_real_trials_wamplitudes_onoff.jld2")
const OUTPUT_FILE = joinpath(PROCESSED_DIR, "wasserstein_summary.jld2")
const OUTPUT_CSV = joinpath(PROCESSED_DIR, "wasserstein_figure5_summary.csv")

include(joinpath(SRC_DIR, "rt_sax_experiment_analysis.jl"))
include(joinpath(SRC_DIR, "rt_sax_experiment_statistics.jl"))

function _load_processed_trials()
    isfile(MODEL_FILE) || error(
        "Missing $MODEL_FILE. Run scripts/01_postprocess.jl first.")
    isfile(REAL_FILE) || error(
        "Missing $REAL_FILE. Run scripts/01_postprocess.jl first.")
    model_trials = JLD2.load(MODEL_FILE, "model_trials")
    real_trials = JLD2.load(REAL_FILE, "real_trials")

    # Reviewed correction retained from the paper analysis.
    matches = filter_trials(
        real_trials;
        subject_ids=["83"], blocks=[2], task=:Overtone, type=:Real)
    length(matches) == 1 || error(
        "Expected one S83 Real block-2 multiphonic trial")
    matches[1].success = false
    return (; model_trials, real_trials,
            trials=vcat(model_trials, real_trials))
end

function _figure5_table(summary)
    rows = NamedTuple[]
    append_rows!(panel, values) = foreach(values) do row
        push!(rows, (
            panel=panel,
            label=_paper_distance_label(row),
            mean=Float64(row.mean),
            sd=Float64(row.sd),
            n=hasproperty(row, :n) ? Int(row.n) : 1,
        ))
    end

    append_rows!("Nonlegato low-high (trial)",
                 summary.nonlegato_trial.summary)
    append_rows!("Nonlegato low-high (pooled)", [(
        task=row.task, typ=row.typ, mean=row.distance, sd=0.0, n=row.n_trials)
        for row in summary.nonlegato_pooled.rows])
    append_rows!("Legato low-high (trial)", summary.legato.summary)
    append_rows!("Legato low-high (pooled)", [(
        task=row.task, typ=row.typ, mean=row.distance, sd=0.0, n=row.n_trials)
        for row in summary.legato.pooled_rows])
    append_rows!("Multiphonic vs nonlegato low", [row
        for row in summary.multiphonic.summary if row.note == :low])
    append_rows!("Multiphonic vs nonlegato high", [row
        for row in summary.multiphonic.summary if row.note == :high])
    append_rows!("Same note asc-desc", summary.same_note.subject_summary)
    return DataFrame(rows)
end

function main()
    data = _load_processed_trials()
    @info "Computing Figure 5 Wasserstein summaries" trials=length(data.trials)
    summary = compute_paper_wasserstein_summary(data.trials)
    mkpath(PROCESSED_DIR)
    JLD2.jldsave(OUTPUT_FILE; summary)
    CSV.write(OUTPUT_CSV, _figure5_table(summary))
    @info "Wasserstein analysis finished" output=OUTPUT_FILE csv=OUTPUT_CSV failures=length(summary.failures)
    return summary
end

main()
