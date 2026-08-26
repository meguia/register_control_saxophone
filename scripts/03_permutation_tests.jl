#!/usr/bin/env julia

import Pkg
const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(PROJECT_ROOT)

using Dates
using JLD2
using Printf

const SRC_DIR = joinpath(PROJECT_ROOT, "src")
const PROCESSED_DIR = joinpath(SRC_DIR, "sessions", "processed_data")
const OUT_DIR = get(ENV, "RTSAX_PERM_OUT_DIR", joinpath(PROCESSED_DIR, "permutation_tests"))

include(joinpath(SRC_DIR, "rt_sax_experiment_analysis.jl"))
include(joinpath(SRC_DIR, "rt_sax_experiment_statistics.jl"))
statistics_validate_dependencies()

if !isdefined(Main, :Trial)
    @eval Main const Trial = $(Trial)
end

@load joinpath(PROCESSED_DIR, "all_model_trials_wamplitudes_onoff.jld2") model_trials
@load joinpath(PROCESSED_DIR, "all_real_trials_wamplitudes_onoff.jld2") real_trials
const TRIALS = vcat(model_trials, real_trials)

const ANALYSIS_TRIAL_OVERRIDES = (
    (subject_id = "83", block = 2, task = :Overtone, typ = :Real, success = false),
)

function apply_analysis_trial_overrides!(trials)
    for override in ANALYSIS_TRIAL_OVERRIDES
        matches = filter_trials(
            trials;
            subject_ids = [override.subject_id],
            blocks = [override.block],
            task = override.task,
            type = override.typ,
        )
        length(matches) == 1 || error("Expected exactly one trial for override $(override), got $(length(matches))")
        matches[1].success = override.success
    end
    return trials
end

apply_analysis_trial_overrides!(TRIALS)

const COND_SPECS = [
    (task = :NonlegatoAsc, typ = :Model, label = "Asc Model"),
    (task = :NonlegatoDesc, typ = :Model, label = "Desc Model"),
    (task = :NonlegatoAsc, typ = :Real, label = "Asc Instrument"),
    (task = :NonlegatoDesc, typ = :Real, label = "Desc Instrument"),
]

const LEGATO_SPECS = [
    (task = :LegatoAsc, typ = :Model, label = "Asc Model"),
    (task = :LegatoDesc, typ = :Model, label = "Desc Model"),
    (task = :LegatoAsc, typ = :Real, label = "Asc Instrument"),
    (task = :LegatoDesc, typ = :Real, label = "Desc Instrument"),
]

const PERM_FAMILIES = [
    :nonlegato_low_high,
    :same_note_asc_desc,
    :overtone_vs_nonlegato,
    :legato_low_high,
]

const PERM_SEED_OFFSETS = Dict(
    :nonlegato_low_high => 0,
    :same_note_asc_desc => 1000,
    :overtone_vs_nonlegato => 2000,
    :legato_low_high => 3000,
)

# Match current notebook 08 defaults unless you intentionally change them here.
const EXCLUDE_SUBJECT = "97"
const TRIM_MS = 100
const INNER_PAD = 0.05
const EDGE_EXCL = 0.15

const W_GLOBAL_SCALE_INFO = compute_global_wasserstein_scales(
    TRIALS;
    cond_specs = COND_SPECS,
    legato_specs = LEGATO_SPECS,
    exclude_subject_id = EXCLUDE_SUBJECT,
    trim_start_ms = TRIM_MS,
    trim_end_ms = TRIM_MS,
    space = :physical,
    pressure_calib = pressure_from_adc,
    force_calib = force_newton_from_adc,
    inner_pad = INNER_PAD,
    edge_exclusion_frac = EDGE_EXCL,
)
const W_SCALES = W_GLOBAL_SCALE_INFO.scales
const X_SCALE_KPA = W_SCALES[1]
const Y_SCALE_N = W_SCALES[2]
const W_BANDWIDTH = 0.03
const W_BUFFER = 0.03
const W_GRID = WassersteinGridSpec(nx = 30, ny = 30, pad_frac = 0.08)
const W_METHOD = :optimaltransport
const W_SINKHORN_REG = 0.10
const W_SINKHORN_MAXITER = 50_000
const W_SINKHORN_TOL = 1e-5
const PERM_N_PERMUTATIONS = parse(Int, get(ENV, "RTSAX_PERM_N_PERMUTATIONS", "1000"))
const PERM_RANDOM_SEED = 42
const PERM_MODE = :subject_stratified  # :subject_stratified (Mode B) or :pooled_global
const PERM_MAX_POINTS_PER_SET = 20   # set to nothing in code to disable subsampling
const PERM_SUBSAMPLE_SEED = 12345

const DISTANCE_KWARGS = (
    scales = W_SCALES,
    bandwidth = W_BANDWIDTH,
    buffer_size = W_BUFFER,
    grid_spec = W_GRID,
    method = W_METHOD,
    sinkhorn_reg = W_SINKHORN_REG,
    sinkhorn_maxiter = W_SINKHORN_MAXITER,
    sinkhorn_tol = W_SINKHORN_TOL,
)

function parse_requested_families(args)
    if isempty(args)
        return copy(PERM_FAMILIES)
    end

    requested = Symbol[]
    allowed = Set(PERM_FAMILIES)
    for a in args
        fam = Symbol(a)
        fam in allowed || error("Unsupported family '$a'. Allowed: $(PERM_FAMILIES)")
        push!(requested, fam)
    end
    return unique(requested)
end

function family_seed(family::Symbol)
    if PERM_RANDOM_SEED === nothing
        return nothing
    end
    return Int(PERM_RANDOM_SEED) + PERM_SEED_OFFSETS[family]
end

function family_outfile(family::Symbol)
    return joinpath(OUT_DIR, "permutation_rows_$(family).jld2")
end

function run_family(family::Symbol)
    start_dt = now()
    t0 = time()

    println("[$(Dates.format(start_dt, "HH:MM:SS"))] Starting family $(family)...")

    rows = if PERM_MODE == :subject_stratified
        permutation_test_wasserstein_rows_by_family_subject_stratified(
            TRIALS;
            family = family,
            cond_specs = COND_SPECS,
            legato_specs = LEGATO_SPECS,
            exclude_subject_id = EXCLUDE_SUBJECT,
            trim_start_ms = TRIM_MS,
            trim_end_ms = TRIM_MS,
            space = :physical,
            pressure_calib = pressure_from_adc,
            force_calib = force_newton_from_adc,
            inner_pad = INNER_PAD,
            edge_exclusion_frac = EDGE_EXCL,
            n_permutations = PERM_N_PERMUTATIONS,
            random_seed = family_seed(family),
            max_points_per_set = PERM_MAX_POINTS_PER_SET,
            subsample_seed = PERM_SUBSAMPLE_SEED,
            distance_kwargs = DISTANCE_KWARGS,
        )
    elseif PERM_MODE == :pooled_global
        permutation_test_wasserstein_rows_by_family(
            TRIALS;
            family = family,
            cond_specs = COND_SPECS,
            legato_specs = LEGATO_SPECS,
            exclude_subject_id = EXCLUDE_SUBJECT,
            trim_start_ms = TRIM_MS,
            trim_end_ms = TRIM_MS,
            space = :physical,
            pressure_calib = pressure_from_adc,
            force_calib = force_newton_from_adc,
            inner_pad = INNER_PAD,
            edge_exclusion_frac = EDGE_EXCL,
            n_permutations = PERM_N_PERMUTATIONS,
            random_seed = family_seed(family),
            distance_kwargs = DISTANCE_KWARGS,
        )
    else
        error("Unsupported PERM_MODE=$(PERM_MODE). Use :subject_stratified or :pooled_global")
    end

    elapsed_s = time() - t0
    ok_rows = filter(r -> r.status == "ok" && isfinite(r.p_value), rows)
    error_rows = filter(r -> startswith(r.status, "error:"), rows)
    insufficient_rows = filter(r -> r.status == "insufficient samples", rows)

    outfile = family_outfile(family)
    @save outfile rows family start_dt elapsed_s PERM_MODE PERM_N_PERMUTATIONS PERM_RANDOM_SEED PERM_MAX_POINTS_PER_SET PERM_SUBSAMPLE_SEED DISTANCE_KWARGS W_GLOBAL_SCALE_INFO EXCLUDE_SUBJECT TRIM_MS INNER_PAD EDGE_EXCL W_METHOD W_SINKHORN_REG W_SINKHORN_MAXITER W_SINKHORN_TOL ANALYSIS_TRIAL_OVERRIDES

    return (
        family = family,
        outfile = outfile,
        n_rows = length(rows),
        n_ok = length(ok_rows),
        n_error = length(error_rows),
        n_insufficient = length(insufficient_rows),
        elapsed_s = elapsed_s,
        min_p_ok = isempty(ok_rows) ? NaN : minimum(r.p_value for r in ok_rows),
    )
end

function main(args)
    mkpath(OUT_DIR)

    requested = parse_requested_families(args)
    println("Using $(Threads.nthreads()) Julia threads")
    println("Permutation mode: $(PERM_MODE)")
    println("Distance method: $(W_METHOD)")
    println("Sinkhorn reg: $(W_SINKHORN_REG), maxiter: $(W_SINKHORN_MAXITER), tol: $(W_SINKHORN_TOL)")
    println("Analysis trial overrides: $(ANALYSIS_TRIAL_OVERRIDES)")
    @printf("Global SD scales: pressure=%.6g force=%.6g from %d retained points\n", X_SCALE_KPA, Y_SCALE_N, W_GLOBAL_SCALE_INFO.n_points)
    println("Point cap per set: $(PERM_MAX_POINTS_PER_SET)")
    println("Families to run: $(requested)")
    println("Output folder: $(OUT_DIR)")

    tasks = Dict{Symbol,Task}()
    for family in requested
        tasks[family] = Threads.@spawn run_family(family)
    end

    summaries = NamedTuple[]
    for family in requested
        summary = fetch(tasks[family])
        push!(summaries, summary)
        @printf(
            "Completed %-24s rows=%d ok=%d err=%d insuff=%d min_p=%.4f time=%.2fs\n",
            String(summary.family),
            summary.n_rows,
            summary.n_ok,
            summary.n_error,
            summary.n_insufficient,
            summary.min_p_ok,
            summary.elapsed_s,
        )
        println("  Saved: $(summary.outfile)")
    end

    index_file = joinpath(OUT_DIR, "permutation_run_index.jld2")
    run_started_at = now()
    @save index_file summaries requested run_started_at
    println("Saved run index: $(index_file)")
end

main(ARGS)
