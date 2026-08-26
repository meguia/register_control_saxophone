#!/usr/bin/env julia

"""Generate paper Figures 3, 4, 5, and 6."""

import Pkg
const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(PROJECT_ROOT)
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using JLD2

const SRC_DIR = joinpath(PROJECT_ROOT, "src")
const PROCESSED_DIR = joinpath(SRC_DIR, "sessions", "processed_data")
const RESULTS_DIR = joinpath(PROJECT_ROOT, "results")

include(joinpath(SRC_DIR, "rt_sax_experiment_analysis.jl"))
include(joinpath(SRC_DIR, "rt_sax_experiment_statistics.jl"))

function _load_figure_inputs()
    model_file = joinpath(
        PROCESSED_DIR, "all_model_trials_wamplitudes_onoff.jld2")
    real_file = joinpath(
        PROCESSED_DIR, "all_real_trials_wamplitudes_onoff.jld2")
    isfile(model_file) || error(
        "Missing processed trials. Run scripts/01_postprocess.jl first.")
    isfile(real_file) || error(
        "Missing processed trials. Run scripts/01_postprocess.jl first.")
    model_trials = JLD2.load(model_file, "model_trials")
    real_trials = JLD2.load(real_file, "real_trials")

    matches = filter_trials(
        real_trials;
        subject_ids=["83"], blocks=[2], task=:Overtone, type=:Real)
    length(matches) == 1 || error(
        "Expected one S83 Real block-2 multiphonic trial")
    matches[1].success = false
    return (; model_trials, real_trials)
end

function _load_or_compute_wasserstein(model_trials, real_trials)
    path = joinpath(PROCESSED_DIR, "wasserstein_summary.jld2")
    if isfile(path)
        return JLD2.load(path, "summary")
    end
    @info "Wasserstein cache is absent; computing it now"
    summary = compute_paper_wasserstein_summary(
        vcat(model_trials, real_trials))
    JLD2.jldsave(path; summary)
    return summary
end

function _figure6_map_path()
    explicit = get(ENV, "RTSAX_FIGURE6_MAP", "")
    !isempty(explicit) && return abspath(explicit)
    recomputed = joinpath(
        PROCESSED_DIR, "multistability_map_recomputed.jld2")
    isfile(recomputed) && return recomputed
    return joinpath(
        PROJECT_ROOT, "data", "derived", "multistability_map_v1.jld2")
end

function main()
    mkpath(RESULTS_DIR)
    data = _load_figure_inputs()

    figure3 = plot_paper_trial_figure(
        data.model_trials; condition=:Model)
    figure4 = plot_paper_trial_figure(
        data.real_trials; condition=:Real)
    summary = _load_or_compute_wasserstein(
        data.model_trials, data.real_trials)
    figure5 = plot_paper_wasserstein_figure(summary)

    map_path = _figure6_map_path()
    map = load_portable_multistability_map(map_path)
    fields = paper_multistability_fields(
        map; smoothing_factor=0.7, threshold=0.7)
    trajectories = paper_figure6_gesture_points(data.model_trials)
    figure6 = plot_paper_multistability_map(
        map; fields, trajectories,
        gamma_limits=(0.02, 0.99), zeta_limits=extrema(map.zeta))

    outputs = Dict(
        "Figure3" => save_paper_figure(
            figure3, joinpath(RESULTS_DIR, "Figure3")),
        "Figure4" => save_paper_figure(
            figure4, joinpath(RESULTS_DIR, "Figure4")),
        "Figure5" => save_paper_figure(
            figure5, joinpath(RESULTS_DIR, "Figure5")),
        "Figure6" => save_paper_figure(
            figure6, joinpath(RESULTS_DIR, "Figure6")),
    )
    @info "Paper figures generated" output=RESULTS_DIR figure6_map=map_path
    return (; outputs, summary, map_counts=validate_multistability_map(map))
end

main()
