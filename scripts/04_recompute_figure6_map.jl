#!/usr/bin/env julia

"""
Restartable continuation-guided computation behind paper Figure 6.

The regularized model is used only to locate and continue finite-amplitude
periodic families.  The final map is then rebuilt and validated with the
original, non-regularized experiment equations.
"""

import Pkg
const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(PROJECT_ROOT)

using JLD2

include(joinpath(
    PROJECT_ROOT, "src", "nonregularized_multistability", "scripts",
    "run_nonregularized_multistability_pipeline.jl"))

const RECOMPUTED_MAP = joinpath(
    PROJECT_ROOT, "src", "sessions", "processed_data",
    "multistability_map_recomputed.jld2")

"""Export a completed exact-model product to the compact Figure 6 schema."""
function export_figure6_map(;
        profile::Symbol=:final,
        run_id::AbstractString=SAX_NONREGULARIZED_EXPANDED_RUN_ID,
        output::AbstractString=RECOMPUTED_MAP)
    paths = sax_nonregularized_multistability_paths(
        PROJECT_ROOT; run_id=String(run_id), profile)
    isfile(paths.product) || error(
        "Completed exact-model product not found: $(paths.product)")
    isfile(paths.model) || error("Exact model file not found: $(paths.model)")
    atlas = JLD2.load(paths.product, "cache").payload
    atlas.status == :complete || error(
        "Exact-model product is not complete: $(atlas.status)")
    raw_model = load_object(paths.model)
    masks = sax_nonregularized_period_masks(atlas, raw_model)
    p2_transition = _sax_nonregularized_low_p1_high_p2_mask(atlas)
    evidence = sax_nonregularized_mixed_edge_evidence(atlas, raw_model)
    t1_t2_coordinates = Set(
        (Float64(point.gamma), Float64(point.zeta))
        for point in evidence.t1_t2_points)
    mixed_points = [(
        gamma=Float64(point.gamma),
        zeta=Float64(point.zeta),
        right_censored=Bool(point.metrics.right_censored),
        escape_time=Float64(point.metrics.escape_time),
        mixed_fraction=Float64(point.metrics.mixed_fraction),
        source=(Float64(point.gamma), Float64(point.zeta)) in
               t1_t2_coordinates ? :t1_t2 : :low_p1_high_p2,
    ) for point in evidence.points]
    known = BitMatrix(atlas.sheets.cache_present)
    counts = (
        expected_points=length(atlas.sheets.gamma) * length(atlas.sheets.zeta),
        known=count(known),
        t1=count(masks.t1),
        t2=count(masks.t2),
        overlap=count(masks.coexistence),
        mixed_completed=length(mixed_points),
        mixed_at_cutoff=count(point -> point.right_censored, mixed_points),
    )
    map = (
        schema_version=1,
        analysis=:nonregularized_fixed_parameter_multistability_map,
        classification=:minimal_full_state_recurrence_period,
        gamma=Float64.(atlas.sheets.gamma),
        zeta=Float64.(atlas.sheets.zeta),
        known,
        t1=BitMatrix(masks.t1),
        t2=BitMatrix(masks.t2),
        other=BitMatrix(masks.other),
        t1_high_p2=BitMatrix(masks.t1_high_p2),
        low_p1_high_p2=BitMatrix(p2_transition),
        mixed_points,
        counts,
        source=(
            repository="register_control_saxophone",
            run_id=String(run_id),
            source_file=basename(paths.product),
        ),
        definitions=(
            t1="Attracting responses with minimal recurrence time T1; includes low-P1 and octave-related high-P2 families.",
            t2="Attracting high-P1 responses with minimal recurrence time T2.",
            overlap="Distinct attracting responses from both recurrence classes have local support.",
            mixed="Selected basin-edge integrations retaining both acoustic modes at a finite 3 s cutoff.",
        ),
        created_at=string(Dates.now()),
    )
    mkpath(dirname(output))
    JLD2.jldsave(output; map)
    @info "Exported compact Figure 6 map" output counts
    return (; map, output, paths)
end

"""
Run the regularized continuation guide and exact non-regularized map in order.

Compatible checkpoints are reused automatically.  `profile=:final` produces
the publication grid and writes `multistability_map_recomputed.jld2` after a
complete run.  `profile=:smoke` is for installation testing only.
"""
function run_figure6_computation(;
        profile::Symbol=:final,
        run_regularized::Bool=true,
        verbosity::Int=1)
    profile in (:smoke, :pilot, :final) ||
        error("profile must be :smoke, :pilot, or :final")
    regularized = if run_regularized
        run_sax_regularized_unattended(
            eta=1e-3,
            profile,
            coverage=:all,
            main_passes=profile == :final ? 3 : 1,
            verbosity,
        )
    else
        nothing
    end

    exact = if profile == :final
        run_sax_nonregularized_expanded_unattended(
            profile=:final,
            guide_source_profile=:final,
            guide_atlas_profile=:final,
            verbosity,
        )
    else
        run_sax_nonregularized_multistability_unattended(
            run_id="paper_$(profile)",
            profile,
            guide_source_profile=profile,
            guide_atlas_profile=profile,
            verbosity,
        )
    end
    exported = profile == :final ? export_figure6_map() : nothing
    return (; regularized, exact, exported)
end

function _main(args)
    profile = any(==("--profile=smoke"), args) ? :smoke :
              any(==("--profile=pilot"), args) ? :pilot : :final
    exact_only = "--exact-only" in args
    return run_figure6_computation(;
        profile, run_regularized=!exact_only)
end

if abspath(PROGRAM_FILE) == @__FILE__
    _main(ARGS)
end
