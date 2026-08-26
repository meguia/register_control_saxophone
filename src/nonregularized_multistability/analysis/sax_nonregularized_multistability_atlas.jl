# Direct attracting-response and multistability map for the historical model.
#
# This analysis intentionally does not infer stable parameter regions from a
# smooth continuation curve.  Every coloured raster cell is backed by a
# trajectory of the exact `saxRN!` equations and by a local perturbation test.
# The regularized map is used only to obtain good initial states. Failed
# seed routes remain explicit unresolved evidence and can never erase a
# successful route from another direction.

const SAX_NONREGULARIZED_MULTISTABILITY_SCHEMA_VERSION = 2
const SAX_NONREGULARIZED_MULTISTABILITY_RUN_ID = "exact_v1"
const SAX_NONREGULARIZED_REFINED_RUN_ID = "exact_v2_refined"
const SAX_NONREGULARIZED_EXPANDED_RUN_ID = "exact_v3_expanded"
const SAX_NONREGULARIZED_REFINED_GAMMA_POINTS = 277
const SAX_NONREGULARIZED_REFINED_ZETA_STEP = 0.0125
const SAX_NONREGULARIZED_EXPANDED_GAMMA_RANGE = (0.02, 0.99)
const SAX_NONREGULARIZED_EXPANDED_GAMMA_POINTS = 389
const SAX_NONREGULARIZED_EXPANDED_ZETA_RANGE = (0.0125, 0.90)
const SAX_NONREGULARIZED_EXPANDED_ZETA_STEP = 0.0125
const SAX_NONREGULARIZED_SEED_SNAPSHOT_SCHEMA_VERSION = 1
const SAX_NONREGULARIZED_MODAL_DOMINANCE_MARGIN = 0.10
# Kept at the value recorded by the completed exact_v1 caches.  The compact
# seed snapshot replaces the old regularized P1 map at runtime, but changing
# this provenance value would unnecessarily invalidate exact-model evidence
# that has already been recomputed and validated at eta=0.
const SAX_NONREGULARIZED_LEGACY_GUIDE_SCHEMA_VERSION = 3
const _SAX_NONREGULARIZED_MULTISTABILITY_SAVE_LOCK = ReentrantLock()
const _SAX_NONREGULARIZED_MULTISTABILITY_PQZ_LOCK = ReentrantLock()

Base.@kwdef struct SaxNonregularizedMultistabilitySettings
    schema_version::Int = SAX_NONREGULARIZED_MULTISTABILITY_SCHEMA_VERSION
    profile::Symbol = :final
    nmodes::Int = 8
    gamma_range::Tuple{Float64,Float64} = (0.30, 0.99)
    gamma_points::Int = 139
    zeta_range::Tuple{Float64,Float64} = (0.10, 0.90)
    zeta_step::Float64 = 0.025
    contact_stiffness::Float64 = 100.0

    # The guide supplies initial states only.  All accepted evidence is
    # recomputed with eta=0 and the historical piecewise vector field.
    guide_eta::Float64 = 1e-3
    guide_max_gamma_distance::Float64 = 0.045
    guide_max_zeta_distance::Float64 = 0.075
    guide_sources_per_register::Int = 5
    track_high_p2::Bool = true
    homotopy_etas::Tuple{Vararg{Float64}} = (3e-4, 1e-4, 3e-5)
    parameter_gamma_step::Float64 = 0.0125
    parameter_zeta_step::Float64 = 0.0125
    intermediate_settling_time::Float64 = 35.0
    homotopy_settling_time::Float64 = 45.0

    # Exact-model convergence is assessed over consecutive blocks so slow
    # relaxation near a fold or double-Hopf neighbourhood is not mistaken for
    # absence of an attractor.
    block_time::Float64 = 100.0
    maximum_blocks::Int = 10
    required_stationary_blocks::Int = 3
    tail_fraction::Float64 = 0.50
    saveat::Float64 = 0.10
    dtmax::Float64 = 0.05
    reltol::Float64 = 1e-8
    abstol::Float64 = 1e-10
    activation_threshold::Float64 = 2e-3
    within_block_drift_tolerance::Float64 = 0.06
    across_block_amplitude_tolerance::Float64 = 0.06
    across_block_period_tolerance::Float64 = 0.025
    recurrence_tolerance::Float64 = 0.045
    maximum_period_cv::Float64 = 0.015
    maximum_return_order::Int = 4
    fallback_excitation::Float64 = 0.03

    # Snapshot propagation is deliberately bidirectional.  A pass only reads
    # anchors committed before the pass began, making threaded execution
    # deterministic and each point cache independently restartable.
    propagation_passes::Int = 5
    p2_propagation_passes::Int = 14
    propagation_max_gamma_distance::Float64 = 0.035
    propagation_max_zeta_distance::Float64 = 0.055
    propagation_sources_per_register::Int = 4

    # Dedicated low-register repair around the double-Hopf strip.  It uses
    # sources above and below the strip and rectangular paths through several
    # bridge gamma values instead of relying on one fragile Hopf predictor.
    dh_zeta_range::Tuple{Float64,Float64} = (0.15, 0.325)
    dh_gamma_range::Tuple{Float64,Float64} = (0.30, 0.90)
    dh_source_gamma_distance::Float64 = 0.18
    dh_source_zeta_distance::Float64 = 0.15
    dh_bridge_gammas::Tuple{Vararg{Float64}} = (0.68, 0.705, 0.74)
    dh_time_factor::Float64 = 1.75

    # Finite-amplitude Poincare-return validation is applicable even when the
    # orbit crosses a nonsmooth surface.  Strict Periodic Schur is attempted
    # additionally only away from flow reversal.  A reed-contact crossing is
    # eligible when it is transverse (not grazing): the vector field is
    # continuous there and the two one-sided Jacobians are finite.
    validation_time::Float64 = 220.0
    perturbation_directions::Int = 4
    perturbation_scales::Tuple{Vararg{Float64}} = (1e-5, 1e-4)
    perturbation_success_fraction::Float64 = 0.75
    return_distance_tolerance::Float64 = 0.08
    maximum_return_contraction_ratio::Float64 = 0.70
    pqz_enabled::Bool = true
    pqz_surface_clearance::Float64 = 2e-4
    pqz_reed_grazing_velocity_tolerance::Float64 = 1e-4
    pqz_collocation_intervals::Int = 40
    pqz_collocation_degree::Int = 4
    pqz_growth_margin::Float64 = 2e-5
    pqz_newton_tolerance::Float64 = 1e-10
    pqz_newton_iterations::Int = 40
    # Dense PQZ is a strict audit, not the raster classifier.  The exact
    # finite-amplitude return test is performed at every candidate.  These
    # strides keep the Final audit reproducible and computationally finite.
    pqz_gamma_stride::Int = 12
    pqz_zeta_stride::Int = 4
    pqz_dh_zeta_stride::Int = 2

    # Edge experiments are sparse and never rasterized over the stable maps.
    edge_gamma_stride::Int = 4
    edge_zeta_stride::Int = 2
    edge_bisections::Int = 10
    edge_decision_time::Float64 = 300.0
    edge_maximum_time::Float64 = 3000.0
    edge_saveat::Float64 = 0.5
    mixedness_threshold::Float64 = 0.55
    mixed_hold_cycles::Float64 = 5.0
end

function _validate_sax_nonregularized_multistability_settings(
        settings::SaxNonregularizedMultistabilitySettings)
    settings.schema_version == SAX_NONREGULARIZED_MULTISTABILITY_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported non-regularized multistability schema"))
    settings.profile in (:smoke, :pilot, :final) ||
        throw(ArgumentError("profile must be smoke, pilot, or final"))
    settings.gamma_range[1] < settings.gamma_range[2] ||
        throw(ArgumentError("gamma_range must be increasing"))
    settings.zeta_range[1] <= settings.zeta_range[2] ||
        throw(ArgumentError("zeta_range must be ordered"))
    settings.gamma_points >= 3 || throw(ArgumentError(
        "gamma_points must be at least three"))
    settings.zeta_step > 0 || throw(ArgumentError("zeta_step must be positive"))
    settings.guide_eta > 0 || throw(ArgumentError("guide_eta must be positive"))
    all(eta -> 0 < eta < settings.guide_eta, settings.homotopy_etas) ||
        throw(ArgumentError("homotopy eta values must lie between zero and guide_eta"))
    issorted(collect(settings.homotopy_etas); rev=true) ||
        throw(ArgumentError("homotopy eta values must decrease toward zero"))
    settings.maximum_blocks >= settings.required_stationary_blocks >= 2 ||
        throw(ArgumentError("invalid convergence block counts"))
    0 < settings.tail_fraction < 1 ||
        throw(ArgumentError("tail_fraction must lie in (0,1)"))
    all(>(0), (settings.block_time, settings.saveat, settings.dtmax,
               settings.reltol, settings.abstol)) ||
        throw(ArgumentError("integration controls must be positive"))
    settings.maximum_return_order >= 1 || throw(ArgumentError(
        "maximum_return_order must be positive"))
    settings.propagation_passes >= 1 || throw(ArgumentError(
        "propagation_passes must be positive"))
    settings.p2_propagation_passes >= 1 || throw(ArgumentError(
        "p2_propagation_passes must be positive"))
    settings.dh_zeta_range[1] <= settings.dh_zeta_range[2] ||
        throw(ArgumentError("dh_zeta_range must be ordered"))
    settings.dh_gamma_range[1] <= settings.dh_gamma_range[2] ||
        throw(ArgumentError("dh_gamma_range must be ordered"))
    settings.perturbation_directions >= 1 || throw(ArgumentError(
        "perturbation_directions must be positive"))
    all(>(0), settings.perturbation_scales) || throw(ArgumentError(
        "perturbation scales must be positive"))
    0 < settings.perturbation_success_fraction <= 1 || throw(ArgumentError(
        "perturbation_success_fraction must lie in (0,1]"))
    settings.pqz_collocation_intervals >= 5 &&
        settings.pqz_collocation_degree >= 2 || throw(ArgumentError(
        "invalid Periodic-Schur mesh"))
    settings.pqz_reed_grazing_velocity_tolerance > 0 ||
        throw(ArgumentError("reed grazing velocity tolerance must be positive"))
    all(>(0), (settings.pqz_gamma_stride, settings.pqz_zeta_stride,
               settings.pqz_dh_zeta_stride)) || throw(ArgumentError(
        "Periodic-Schur audit strides must be positive"))
    return settings
end

"""Resolution presets for the exact attracting-response map."""
function sax_nonregularized_multistability_settings(
        profile::Symbol=:final; kwargs...)
    profile in (:smoke, :pilot, :final) || throw(ArgumentError(
        "profile must be smoke, pilot, or final"))
    preset = profile == :smoke ? (
        profile=:smoke,
        gamma_range=(0.40, 0.60), gamma_points=5,
        zeta_range=(0.60, 0.60), zeta_step=1.0,
        guide_sources_per_register=2,
        homotopy_etas=(1e-4,),
        intermediate_settling_time=4.0,
        homotopy_settling_time=5.0,
        block_time=18.0, maximum_blocks=3,
        required_stationary_blocks=2,
        saveat=0.15, dtmax=0.10,
        within_block_drift_tolerance=0.60,
        across_block_amplitude_tolerance=0.60,
        across_block_period_tolerance=0.25,
        recurrence_tolerance=0.30,
        maximum_period_cv=0.20,
        propagation_passes=1,
        p2_propagation_passes=1,
        perturbation_directions=1,
        perturbation_scales=(1e-4,),
        perturbation_success_fraction=1.0,
        validation_time=25.0,
        return_distance_tolerance=0.35,
        maximum_return_contraction_ratio=1.25,
        pqz_enabled=false,
        edge_gamma_stride=2, edge_zeta_stride=1,
        edge_bisections=2, edge_decision_time=20.0,
        edge_maximum_time=40.0,
    ) : profile == :pilot ? (
        profile=:pilot,
        gamma_range=(0.32, 0.92), gamma_points=61,
        zeta_range=(0.125, 0.50), zeta_step=0.05,
        guide_sources_per_register=3,
        homotopy_etas=(1e-4, 3e-5),
        intermediate_settling_time=18.0,
        homotopy_settling_time=25.0,
        block_time=60.0, maximum_blocks=7,
        required_stationary_blocks=3,
        saveat=0.15, dtmax=0.075,
        within_block_drift_tolerance=0.12,
        across_block_amplitude_tolerance=0.12,
        across_block_period_tolerance=0.06,
        recurrence_tolerance=0.08,
        maximum_period_cv=0.04,
        propagation_passes=3,
        p2_propagation_passes=8,
        perturbation_directions=3,
        perturbation_scales=(1e-4,),
        validation_time=100.0,
        pqz_collocation_intervals=25,
        pqz_collocation_degree=3,
        pqz_gamma_stride=8,
        pqz_zeta_stride=3,
        pqz_dh_zeta_stride=2,
        edge_bisections=6,
        edge_decision_time=120.0,
        edge_maximum_time=900.0,
    ) : (profile=:final,)
    settings = SaxNonregularizedMultistabilitySettings(;
        merge(preset, (; kwargs...))...)
    return _validate_sax_nonregularized_multistability_settings(settings)
end

_portable_sax_nonregularized_multistability_settings(settings) =
    NamedTuple{fieldnames(SaxNonregularizedMultistabilitySettings)}(
        Tuple(getfield(settings, name)
              for name in fieldnames(SaxNonregularizedMultistabilitySettings)))

function _sax_nonregularized_validate_run_id(run_id::AbstractString)
    value = String(run_id)
    occursin(r"^[A-Za-z0-9_-]+$", value) || throw(ArgumentError(
        "run_id accepts only letters, numbers, underscores, and hyphens"))
    isempty(value) && throw(ArgumentError("run_id cannot be empty"))
    return value
end

_sax_nonregularized_tag(value::Real; digits::Integer=6) = replace(
    @sprintf("%.*f", Int(digits), float(value)), "." => "p", "-" => "m")

"""Every exact-model output path; no regularized or historical cache is writable here."""
function sax_nonregularized_multistability_paths(
        project_root::AbstractString;
        fingering::AbstractString="Dx4",
        run_id::AbstractString=SAX_NONREGULARIZED_MULTISTABILITY_RUN_ID,
        profile::Symbol=:final)
    profile in (:smoke, :pilot, :final) || throw(ArgumentError(
        "profile must be smoke, pilot, or final"))
    selected_run = _sax_nonregularized_validate_run_id(run_id)
    root = joinpath(
        project_root, "src", "sessions", "processed_data",
        "multistability_nonregularized_$(fingering)", "runs",
        selected_run, String(profile))
    points = joinpath(root, "points")
    edges = joinpath(root, "edge_escape")
    return (
        root=root,
        model=joinpath(project_root, "src", "impedances", "alto",
                       "$(fingering).jld2"),
        points=points,
        point=(gamma, zeta) -> joinpath(
            points, "z$(_sax_nonregularized_tag(zeta))",
            "g$(_sax_nonregularized_tag(gamma)).jld2"),
        edges=edges,
        edge=(gamma, zeta) -> joinpath(
            edges, "z$(_sax_nonregularized_tag(zeta))_g$(_sax_nonregularized_tag(gamma)).jld2"),
        manifest=joinpath(root, "nonregularized_multistability_manifest.jld2"),
        product=joinpath(root, "nonregularized_multistability_atlas.jld2"),
        figures=joinpath(root, "figures"),
        log=joinpath(root, "unattended_nonregularized_multistability.log"),
    )
end

"""Path of the compact, exact-model P1 seed snapshot used by new/resumed runs."""
function sax_nonregularized_seed_snapshot_path(
        project_root::AbstractString;
        fingering::AbstractString="Dx4",
        run_id::AbstractString=SAX_NONREGULARIZED_MULTISTABILITY_RUN_ID,
        profile::Symbol=:final)
    source = sax_nonregularized_multistability_paths(
        project_root; fingering=fingering, run_id=run_id, profile=profile)
    return joinpath(source.root, "exact_p1_seed_snapshot.jld2")
end

function _sax_nonregularized_grid(settings)
    gamma = collect(range(settings.gamma_range[1], settings.gamma_range[2];
                          length=settings.gamma_points))
    lower, upper = settings.zeta_range
    zeta = if lower == upper
        [float(lower)]
    else
        count = floor(Int, (upper - lower) / settings.zeta_step + 1e-9)
        values = [float(lower + index * settings.zeta_step) for index in 0:count]
        if upper - last(values) > settings.zeta_step * 1e-6
            push!(values, float(upper))
        else
            values[end] = float(upper)
        end
        unique(values)
    end
    return (gamma=gamma, zeta=zeta)
end

function _sax_nonregularized_signature(settings, guide_signature)
    return (
        settings=_portable_sax_nonregularized_multistability_settings(settings),
        guide=guide_signature,
        exact_model=:historical_piecewise_saxRN,
    )
end

function _save_sax_nonregularized_cache(
        path::AbstractString, kind::Symbol, payload,
        model_p::NamedTuple, settings, guide_signature)
    signature = _sax_nonregularized_signature(settings, guide_signature)
    cache = (
        schema_version=SAX_NONREGULARIZED_MULTISTABILITY_SCHEMA_VERSION,
        cache_kind=kind,
        settings_signature=signature,
        model_signature=_sax_bifurcation_model_signature(
            merge(model_p, (contact_stiffness=settings.contact_stiffness,)),
            settings.nmodes),
        saved_at_unix=time(),
        payload=payload,
    )
    lock(_SAX_NONREGULARIZED_MULTISTABILITY_SAVE_LOCK) do
        _atomic_jld2_save(path; cache)
    end
    return payload
end

function _load_sax_nonregularized_cache(
        path::AbstractString, kind::Symbol,
        model_p::NamedTuple, settings, guide_signature)
    isfile(path) || return (
        status=:missing, payload=nothing, reason="cache is absent")
    stored = try
        Logging.with_logger(Logging.NullLogger()) do
            JLD2.load(path, "cache")
        end
    catch err
        return (status=:corrupt, payload=nothing, reason=sprint(showerror, err))
    end
    compatible = try
        stored.schema_version == SAX_NONREGULARIZED_MULTISTABILITY_SCHEMA_VERSION &&
        stored.cache_kind == kind &&
        isequal(stored.settings_signature,
                _sax_nonregularized_signature(settings, guide_signature)) &&
        isequal(stored.model_signature,
                _sax_bifurcation_model_signature(
                    merge(model_p, (contact_stiffness=settings.contact_stiffness,)),
                    settings.nmodes))
    catch
        false
    end
    compatible || return (
        status=:incompatible, payload=nothing,
        reason="schema, exact model, guide identity, or settings changed")
    return (status=:valid, payload=stored.payload, reason="compatible cache")
end

function _sax_nonregularized_empty_point(gamma::Real, zeta::Real)
    return (
        analysis=:nonregularized_multistability_point,
        gamma=float(gamma), zeta=float(zeta),
        attempts=Any[], attractors=Any[],
        updated_at_unix=time(),
    )
end

function _load_sax_nonregularized_point(
        paths, gamma, zeta, model_p, settings, guide_signature)
    loaded = _load_sax_nonregularized_cache(
        paths.point(gamma, zeta), :nonregularized_point,
        model_p, settings, guide_signature)
    return loaded.status == :valid ? loaded.payload :
        _sax_nonregularized_empty_point(gamma, zeta)
end

function _save_sax_nonregularized_point(
        paths, point, model_p, settings, guide_signature)
    updated = merge(point, (updated_at_unix=time(),))
    return _save_sax_nonregularized_cache(
        paths.point(point.gamma, point.zeta), :nonregularized_point,
        updated, model_p, settings, guide_signature)
end

function _sax_nonregularized_seed_anchor(point, attractor)
    return (
        continuation_step=0,
        source_mode=Int(attractor.register),
        direction=:exact_atlas,
        gamma=float(point.gamma),
        zeta=float(point.zeta),
        period=float(attractor.period),
        pressure_l2=NaN,
        dominant_growth=NaN,
        dominant_angle=NaN,
        stability=:stable,
        classified_register=Int8(attractor.register),
        register_contrast=float(attractor.register_contrast),
        modal_amplitudes=collect(float.(attractor.modal_amplitudes)),
        fixed_residual=NaN,
        phase_convention=:exact_poincare_section,
        eta=0.0,
        family=attractor.family,
        quality=float(attractor.quality),
        state=collect(float.(attractor.phase_state)),
    )
end

"""
    build_sax_nonregularized_seed_snapshot(raw_model, source_paths, destination;
                                           settings)

Extract one compact P1 seed per validated exact-model grid response. The
snapshot contains initial states only: it is not a stability map and cannot
colour a cell in the map. It replaces the much larger regularized P1 sheet
cache as the seed source for future exact-model runs.
"""
function build_sax_nonregularized_seed_snapshot(
        raw_model::NamedTuple, source_paths, destination::AbstractString;
        settings::SaxNonregularizedMultistabilitySettings=
            sax_nonregularized_multistability_settings(:final))
    isdir(source_paths.points) || error(
        "exact point-cache directory is absent: $(source_paths.points)")
    expected_model = _sax_bifurcation_model_signature(
        merge(raw_model, (contact_stiffness=settings.contact_stiffness,)),
        settings.nmodes)
    point_files = String[]
    for (directory, _, files) in walkdir(source_paths.points), file in files
        endswith(file, ".jld2") && push!(point_files, joinpath(directory, file))
    end
    sort!(point_files)
    selected = Dict{Tuple{Symbol,Float64,Float64},Any}()
    compatible_points = 0
    for path in point_files
        stored = try
            Logging.with_logger(Logging.NullLogger()) do
                JLD2.load(path, "cache")
            end
        catch
            continue
        end
        compatible = try
            stored.schema_version ==
                SAX_NONREGULARIZED_MULTISTABILITY_SCHEMA_VERSION &&
            stored.cache_kind == :nonregularized_point &&
            isequal(stored.model_signature, expected_model)
        catch
            false
        end
        compatible || continue
        point = stored.payload
        compatible_points += 1
        for attractor in point.attractors
            accepted = try
                Bool(attractor.accepted) &&
                    Bool(attractor.validation.attracting) &&
                    Bool(attractor.validation.exact_model) &&
                    attractor.family in (:low_p1, :high_p1)
            catch
                false
            end
            accepted || continue
            key = (attractor.family, float(point.gamma), float(point.zeta))
            anchor = _sax_nonregularized_seed_anchor(point, attractor)
            if !haskey(selected, key) || anchor.quality < selected[key].quality
                selected[key] = anchor
            end
        end
    end
    isempty(selected) && error(
        "no validated exact P1 attractors were found below $(source_paths.points)")
    anchors = sort!(collect(values(selected)); by=anchor ->
        (anchor.classified_register, anchor.zeta, anchor.gamma, anchor.quality))
    component = (
        analysis=:exact_validated_p1_seed_snapshot,
        mode=0,
        direction=:exact_atlas,
        zeta=NaN,
        anchors=Any[anchors...],
    )
    snapshot = (
        schema_version=SAX_NONREGULARIZED_SEED_SNAPSHOT_SCHEMA_VERSION,
        cache_kind=:nonregularized_p1_seed_snapshot,
        model_signature=expected_model,
        saved_at_unix=time(),
        source=(
            exact_model=:historical_piecewise_saxRN,
            eta=0.0,
            root=source_paths.root,
            compatible_point_caches=compatible_points,
        ),
        counts=(
            anchors=length(anchors),
            low=count(anchor -> anchor.classified_register == 1, anchors),
            high=count(anchor -> anchor.classified_register == 2, anchors),
        ),
        progress=(
            analysis=:exact_p1_seed_snapshot,
            status=:complete,
            sheets=(status=:complete, components=Any[component]),
        ),
    )
    _atomic_jld2_save(destination; snapshot)
    return snapshot
end

"""Load and validate a compact exact-model P1 seed snapshot."""
function load_sax_nonregularized_seed_snapshot(
        path::AbstractString, raw_model::NamedTuple;
        settings::SaxNonregularizedMultistabilitySettings=
            sax_nonregularized_multistability_settings(:final))
    isfile(path) || return (
        status=:missing, snapshot=nothing,
        reason="exact P1 seed snapshot is absent")
    snapshot = try
        Logging.with_logger(Logging.NullLogger()) do
            JLD2.load(path, "snapshot")
        end
    catch err
        return (status=:corrupt, snapshot=nothing,
                reason=sprint(showerror, err))
    end
    expected_model = _sax_bifurcation_model_signature(
        merge(raw_model, (contact_stiffness=settings.contact_stiffness,)),
        settings.nmodes)
    compatible = try
        snapshot.schema_version ==
            SAX_NONREGULARIZED_SEED_SNAPSHOT_SCHEMA_VERSION &&
        snapshot.cache_kind == :nonregularized_p1_seed_snapshot &&
        isequal(snapshot.model_signature, expected_model) &&
        snapshot.counts.low > 0 && snapshot.counts.high > 0
    catch
        false
    end
    compatible || return (
        status=:incompatible, snapshot=nothing,
        reason="snapshot schema or exact-model signature changed")
    return (status=:valid, snapshot=snapshot, reason="compatible exact P1 seeds")
end

# ---------------------------------------------------------------------------
# Guide extraction and deterministic source selection
# ---------------------------------------------------------------------------

function _sax_nonregularized_anchor_register(anchor, model_p::NamedTuple)
    if hasproperty(anchor, :classified_register)
        selected = Int(anchor.classified_register)
        selected in (1, 2) && return selected
    end
    source_mode = hasproperty(anchor, :source_mode) ?
        Int(anchor.source_mode) : 0
    isfinite(anchor.period) && anchor.period > 0 || return 0
    length(model_p.ω) >= 2 || return 0
    reference_periods = (2pi / float(model_p.ω[1]),
                         2pi / float(model_p.ω[2]))
    errors = abs.(log.(float(anchor.period) ./ reference_periods))
    separation = abs(errors[1] - errors[2])
    if separation <= 100eps(Float64)
        return source_mode in (1, 2) ? source_mode : 0
    end
    return Int(argmin(errors))
end

function sax_nonregularized_guide_anchors(
        guide_progress, guide_model::NamedTuple,
        settings::SaxNonregularizedMultistabilitySettings)
    anchors = Any[]
    hasproperty(guide_progress, :sheets) || return anchors
    for component in guide_progress.sheets.components
        for anchor in component.anchors
            hasproperty(anchor, :stability) && anchor.stability == :stable || continue
            register = _sax_nonregularized_anchor_register(anchor, guide_model)
            register in (1, 2) || continue
            key = join((
                "guide", "r$(register)",
                "g$(_sax_nonregularized_tag(anchor.gamma))",
                "z$(_sax_nonregularized_tag(anchor.zeta))",
                hasproperty(anchor, :direction) ? String(anchor.direction) : "unknown",
                hasproperty(component, :analysis) ? String(component.analysis) : "component",
            ), "_")
            push!(anchors, (
                key=key,
                kind=hasproperty(anchor, :eta) && anchor.eta == 0 ?
                    :exact_atlas_guide : :regularized_guide,
                register=Int(register),
                family=register == 1 ? :low_p1 : :high_p1,
                gamma=float(anchor.gamma),
                zeta=float(anchor.zeta),
                eta=hasproperty(anchor, :eta) ? float(anchor.eta) :
                    float(settings.guide_eta),
                period=float(anchor.period),
                state=collect(float.(anchor.state)),
                source_analysis=hasproperty(component, :analysis) ?
                    component.analysis : :unknown,
            ))
        end
    end
    sort!(anchors; by=anchor ->
        (anchor.register, anchor.zeta, anchor.gamma, anchor.key))
    unique_anchors = Any[]
    for anchor in anchors
        duplicate = any(existing ->
            existing.register == anchor.register &&
            hypot(existing.gamma - anchor.gamma,
                  existing.zeta - anchor.zeta) <= 1e-7,
            unique_anchors)
        duplicate || push!(unique_anchors, anchor)
    end
    return unique_anchors
end

"""
    sax_nonregularized_p2_guide_anchors(p2_root, guide_model, settings)

Read stable, branch-switched P2 checkpoints from the canonical regularized
run.  The collocation vector supplies only a phase state and branch provenance;
every accepted map response is subsequently transported to and reintegrated
at eta=0.  Component settings may belong to an older continuation schema, so
compatibility is established from the stored model signature and the strict
Periodic-Schur `stable` flag rather than from the current P2 settings object.
"""
function sax_nonregularized_p2_guide_anchors(
        p2_root::AbstractString, guide_model::NamedTuple,
        settings::SaxNonregularizedMultistabilitySettings)
    settings.track_high_p2 || return Any[]
    component_directory = joinpath(p2_root, "components")
    checkpoint_directory = joinpath(p2_root, "checkpoints")
    isdir(component_directory) && isdir(checkpoint_directory) || return Any[]
    expected_model = _sax_bifurcation_model_signature(
        guide_model, settings.nmodes)
    dimension = 2 + 2settings.nmodes
    anchors = Any[]
    for component_path in sort(readdir(component_directory; join=true))
        endswith(component_path, ".jld2") || continue
        component_cache = try
            Logging.with_logger(Logging.NullLogger()) do
                JLD2.load(component_path, "cache")
            end
        catch
            continue
        end
        compatible = try
            isequal(component_cache.model_signature, expected_model) &&
                component_cache.payload.status == :complete
        catch
            false
        end
        compatible || continue
        component = component_cache.payload
        samples = try
            component.analysis.pqz_samples
        catch
            Any[]
        end
        stable_samples = Dict(
            Int(sample.continuation_step) => sample
            for sample in samples if Bool(sample.stable))
        isempty(stable_samples) && continue
        component_key = splitext(basename(component_path))[1]
        checkpoint_path = joinpath(
            checkpoint_directory, "$(component_key)_checkpoints.jld2")
        isfile(checkpoint_path) || continue
        checkpoint_cache = try
            Logging.with_logger(Logging.NullLogger()) do
                JLD2.load(checkpoint_path, "checkpoint_cache")
            end
        catch
            continue
        end
        checkpoint_compatible = try
            isequal(checkpoint_cache.model_signature, expected_model)
        catch
            false
        end
        checkpoint_compatible || continue
        for checkpoint in checkpoint_cache.checkpoints
            step = Int(checkpoint.accepted_step)
            haskey(stable_samples, step) || continue
            solution = checkpoint.solution
            length(solution) >= dimension + 1 || continue
            sample = stable_samples[step]
            key = "p2_$(component_key)_step$(step)"
            push!(anchors, (
                key=key,
                kind=:regularized_p2_guide,
                register=2,
                family=:high_p2,
                gamma=float(checkpoint.gamma),
                zeta=float(checkpoint.zeta),
                eta=float(settings.guide_eta),
                period=float(solution[end]),
                state=collect(float.(solution[1:dimension])),
                continuation_step=step,
                dominant_growth=float(sample.dominant_growth),
                source_analysis=:regularized_p2_periodic_schur,
            ))
        end
    end
    sort!(anchors; by=anchor ->
        (anchor.zeta, anchor.gamma, anchor.dominant_growth, anchor.key))
    return anchors
end

function sax_nonregularized_guide_signature(;
        fingering::AbstractString,
        eta::Real,
        run_id::AbstractString,
        source_profile::Symbol,
        atlas_profile::Symbol,
        p2_anchors=Any[])
    p2_identity = Tuple((
        key=String(anchor.key),
        gamma=round(float(anchor.gamma); digits=10),
        zeta=round(float(anchor.zeta); digits=10),
        period=round(float(anchor.period); digits=10),
        continuation_step=Int(anchor.continuation_step),
    ) for anchor in p2_anchors)
    return (
        role=:initial_states_only,
        fingering=String(fingering),
        eta=float(eta),
        run_id=String(run_id),
        source_profile=source_profile,
        atlas_profile=atlas_profile,
        p2_checkpoint_schema=:canonical_periodic_schur,
        p2_checkpoint_identity=p2_identity,
        guide_schema=SAX_NONREGULARIZED_LEGACY_GUIDE_SCHEMA_VERSION,
    )
end

function _sax_nonregularized_diverse_sources(
        candidates, gamma::Real, zeta::Real, maximum::Integer)
    isempty(candidates) && return Any[]
    ordered = sort(collect(candidates); by=source -> (
        isapprox(source.zeta, zeta; atol=1e-10, rtol=0) ? 0 : 1,
        hypot(source.gamma - gamma, source.zeta - zeta),
        source.key,
    ))
    # Always try the nearest source first. Directional buckets then add
    # independent routes without making a farther below-DH seed precede a
    # close above-DH seed.
    selected = Any[first(ordered)]
    length(selected) >= maximum && return selected
    buckets = (
        source -> isapprox(source.zeta, zeta; atol=1e-10, rtol=0),
        source -> source.zeta < zeta,
        source -> source.zeta > zeta,
        source -> source.gamma < gamma,
        source -> source.gamma > gamma,
    )
    for predicate in buckets
        index = findfirst(source -> predicate(source) &&
            all(existing -> existing.key != source.key, selected), ordered)
        isnothing(index) || push!(selected, ordered[index])
        length(selected) >= maximum && return selected
    end
    for source in ordered
        all(existing -> existing.key != source.key, selected) || continue
        push!(selected, source)
        length(selected) >= maximum && break
    end
    return selected
end

function _sax_nonregularized_select_guide_sources(
        anchors, register::Integer, gamma::Real, zeta::Real,
        settings::SaxNonregularizedMultistabilitySettings;
        dh::Bool=false)
    maximum_gamma = dh ? settings.dh_source_gamma_distance :
        settings.guide_max_gamma_distance
    maximum_zeta = dh ? settings.dh_source_zeta_distance :
        settings.guide_max_zeta_distance
    candidates = [anchor for anchor in anchors
        if anchor.register == register &&
           abs(anchor.gamma - gamma) <= maximum_gamma &&
           abs(anchor.zeta - zeta) <= maximum_zeta]
    maximum = dh ? max(settings.guide_sources_per_register, 7) :
        settings.guide_sources_per_register
    return _sax_nonregularized_diverse_sources(
        candidates, gamma, zeta, maximum)
end

function _sax_nonregularized_select_guide_family_sources(
        anchors, family::Symbol, gamma::Real, zeta::Real,
        settings::SaxNonregularizedMultistabilitySettings)
    candidates = [anchor for anchor in anchors
        if anchor.family == family &&
           abs(anchor.gamma - gamma) <= settings.guide_max_gamma_distance &&
           abs(anchor.zeta - zeta) <= settings.guide_max_zeta_distance]
    return _sax_nonregularized_diverse_sources(
        candidates, gamma, zeta, settings.guide_sources_per_register)
end

# ---------------------------------------------------------------------------
# Exact and eta-homotopy integration
# ---------------------------------------------------------------------------

_sax_nonregularized_algorithm(kind::Symbol) =
    kind == :vern9 ? Vern9() : kind == :tsit5 ? Tsit5() :
    throw(ArgumentError("solver kind must be tsit5 or vern9"))

function _sax_nonregularized_solve(
        initial_state::AbstractVector{<:Real}, gamma::Real, zeta::Real,
        raw_model::NamedTuple,
        settings::SaxNonregularizedMultistabilitySettings,
        duration::Real;
        eta::Union{Nothing,Real}=nothing,
        solver_kind::Symbol=:tsit5,
        history::Bool=false,
        dense_output::Bool=false,
        saveat::Real=settings.saveat,
        reltol::Real=settings.reltol,
        abstol::Real=settings.abstol)
    duration > 0 || throw(ArgumentError("integration duration must be positive"))
    state = collect(float.(initial_state))
    problem = if isnothing(eta)
        parameters = set_parameters(
            float(gamma), float(zeta), raw_model, Int64(settings.nmodes))
        ODEProblem(saxRN!, state, (0.0, float(duration)), parameters)
    else
        regularized = regularize_sax_model(raw_model; eta=float(eta))
        parameters = sax_bifurcation_parameters(
            regularized; gamma=gamma, zeta=zeta, nmodes=settings.nmodes)
        ODEProblem(
            sax_regularized_dynamics!, state,
            (0.0, float(duration)), parameters)
    end
    solution = if history && dense_output
        solve(
            problem, _sax_nonregularized_algorithm(solver_kind);
            dense=true,
            dtmax=settings.dtmax, reltol=float(reltol), abstol=float(abstol),
        )
    elseif history
        solve(
            problem, _sax_nonregularized_algorithm(solver_kind);
            saveat=float(saveat), dense=false,
            dtmax=settings.dtmax, reltol=float(reltol), abstol=float(abstol),
        )
    else
        solve(
            problem, _sax_nonregularized_algorithm(solver_kind);
            save_everystep=false, save_start=false, dense=false,
            dtmax=settings.dtmax, reltol=float(reltol), abstol=float(abstol),
        )
    end
    DifferentialEquations.SciMLBase.successful_retcode(solution) || error(
        "integration failed at gamma=$(gamma), zeta=$(zeta), eta=$(eta): $(solution.retcode)")
    terminal = collect(float.(solution.u[end]))
    all(isfinite, terminal) || error("integration produced a non-finite state")
    return history ? (
        terminal_state=terminal,
        times=collect(float.(solution.t)),
        states=real.(Array(solution)),
        solution=solution,
    ) : (terminal_state=terminal,)
end

function _sax_nonregularized_parameter_points(
        start_gamma::Real, start_zeta::Real,
        target_gamma::Real, target_zeta::Real,
        settings::SaxNonregularizedMultistabilitySettings)
    steps = max(
        ceil(Int, abs(target_gamma - start_gamma) /
                  settings.parameter_gamma_step),
        ceil(Int, abs(target_zeta - start_zeta) /
                  settings.parameter_zeta_step),
        1,
    )
    return [(
        gamma=float(start_gamma + fraction * (target_gamma - start_gamma)),
        zeta=float(start_zeta + fraction * (target_zeta - start_zeta)),
    ) for fraction in range(0.0, 1.0; length=steps + 1)[2:end]]
end

function _sax_nonregularized_transport_regularized_source(
        source, target_gamma::Real, target_zeta::Real,
        raw_model::NamedTuple,
        settings::SaxNonregularizedMultistabilitySettings)
    state = copy(source.state)
    for point in _sax_nonregularized_parameter_points(
            source.gamma, source.zeta, target_gamma, target_zeta, settings)
        state = _sax_nonregularized_solve(
            state, point.gamma, point.zeta, raw_model, settings,
            settings.intermediate_settling_time;
            eta=settings.guide_eta).terminal_state
    end
    for eta in settings.homotopy_etas
        state = _sax_nonregularized_solve(
            state, target_gamma, target_zeta, raw_model, settings,
            settings.homotopy_settling_time; eta=eta).terminal_state
    end
    return state
end

function _sax_nonregularized_transport_exact_source(
        source, target_gamma::Real, target_zeta::Real,
        raw_model::NamedTuple,
        settings::SaxNonregularizedMultistabilitySettings;
        waypoints=Tuple{Float64,Float64}[])
    state = copy(source.state)
    current_gamma, current_zeta = float(source.gamma), float(source.zeta)
    targets = vcat(collect(waypoints),
                   [(float(target_gamma), float(target_zeta))])
    for (waypoint_gamma, waypoint_zeta) in targets
        for point in _sax_nonregularized_parameter_points(
                current_gamma, current_zeta,
                waypoint_gamma, waypoint_zeta, settings)
            state = _sax_nonregularized_solve(
                state, point.gamma, point.zeta, raw_model, settings,
                settings.intermediate_settling_time).terminal_state
        end
        current_gamma, current_zeta = waypoint_gamma, waypoint_zeta
    end
    return state
end

function _sax_nonregularized_fallback_source(
        gamma::Real, zeta::Real, register::Integer,
        raw_model::NamedTuple,
        settings::SaxNonregularizedMultistabilitySettings)
    state, residual = _estimate_fixed_point(
        gamma, zeta, raw_model; nmodes=settings.nmodes)
    state[2register + 1] += settings.fallback_excitation
    return (
        key="fallback_r$(register)_g$(_sax_nonregularized_tag(gamma))_z$(_sax_nonregularized_tag(zeta))",
        kind=:equilibrium_mode_excitation,
        register=Int(register),
        family=register == 1 ? :low_p1 : :high_p1,
        gamma=float(gamma), zeta=float(zeta), eta=0.0,
        period=2pi / float(raw_model.ω[register]),
        state=collect(float.(state)),
        fixed_residual=float(residual),
    )
end

# ---------------------------------------------------------------------------
# Recurrence, register, stationarity, and surface diagnostics
# ---------------------------------------------------------------------------

function _sax_nonregularized_section(
        times::AbstractVector{<:Real}, states::AbstractMatrix{<:Real},
        mode::Integer,
        settings::SaxNonregularizedMultistabilitySettings)
    index = 2Int(mode) + 1
    coordinate = collect(float.(@view states[index, :]))
    coordinate .-= mean(coordinate)
    crossing_indices = Int[]
    fractions = Float64[]
    for column in 1:(length(coordinate) - 1)
        coordinate[column] <= 0 < coordinate[column + 1] || continue
        denominator = coordinate[column + 1] - coordinate[column]
        fraction = abs(denominator) <= eps(Float64) ? 0.0 :
            -coordinate[column] / denominator
        push!(crossing_indices, column)
        push!(fractions, clamp(float(fraction), 0.0, 1.0))
    end
    if length(crossing_indices) < 4
        return (
            valid=false, mode=Int(mode), return_order=0,
            recurrence_error=Inf, period=NaN, carrier_period=NaN,
            period_cv=Inf, crossings=length(crossing_indices),
            phase_state=collect(float.(@view states[:, end])),
            section_points=zeros(Float64, size(states, 1), 0),
            crossing_times=Float64[], scales=ones(Float64, size(states, 1)),
        )
    end
    section = Matrix{Float64}(
        undef, size(states, 1), length(crossing_indices))
    crossing_times = Vector{Float64}(undef, length(crossing_indices))
    for (item, (column, fraction)) in enumerate(
            zip(crossing_indices, fractions))
        section[:, item] .=
            (1 - fraction) .* @view(states[:, column]) .+
            fraction .* @view(states[:, column + 1])
        crossing_times[item] =
            (1 - fraction) * times[column] + fraction * times[column + 1]
    end
    scales = vec(std(states; dims=2))
    scale_floor = max(maximum(scales) * 1e-6, 1e-10)
    scales .= max.(scales, scale_floor)
    # Retain at least three independent return pairs.  Without this guard a
    # short tail can spuriously prefer lag two merely because it has only one
    # or two comparisons.
    maximum_lag = min(settings.maximum_return_order,
                      max(1, size(section, 2) - 3))
    errors = Float64[]
    periods_by_lag = Vector{Vector{Float64}}()
    for lag in 1:maximum_lag
        push!(errors, median([
            norm((@view(section[:, item + lag]) .-
                  @view(section[:, item])) ./ scales) /
                sqrt(size(states, 1))
            for item in 1:(size(section, 2) - lag)
        ]))
        push!(periods_by_lag, [
            crossing_times[item + lag] - crossing_times[item]
            for item in 1:(length(crossing_times) - lag)
        ])
    end
    period_cvs = [std(periods) /
        max(abs(mean(periods)), eps(Float64)) for periods in periods_by_lag]
    valid_lags = findall(lag ->
        errors[lag] <= settings.recurrence_tolerance &&
        period_cvs[lag] <= settings.maximum_period_cv,
        eachindex(errors))
    # The first valid return is the minimal recurrence order.  This prevents a
    # P1 cycle from being relabelled P2 simply because its two-return error is
    # marginally smaller at machine precision.
    order = isempty(valid_lags) ? argmin(errors) : first(valid_lags)
    periods = periods_by_lag[order]
    period = median(periods)
    period_cv = period_cvs[order]
    valid = order in valid_lags
    return (
        valid=valid,
        mode=Int(mode),
        return_order=Int(order),
        recurrence_error=float(errors[order]),
        period=float(period),
        carrier_period=float(period / order),
        period_cv=float(period_cv),
        crossings=length(crossing_indices),
        phase_state=collect(float.(@view section[:, end])),
        section_points=section,
        crossing_times=crossing_times,
        scales=scales,
    )
end

function _sax_nonregularized_total_pressure_phase(states)
    pressure = vec(sum(@view states[3:2:end, :]; dims=1))
    centered = pressure .- mean(pressure)
    for index in (length(centered) - 1):-1:1
        centered[index] <= 0 < centered[index + 1] || continue
        denominator = centered[index + 1] - centered[index]
        fraction = abs(denominator) <= eps(Float64) ? 0.0 :
            -centered[index] / denominator
        return collect(float.((1 - fraction) .* @view(states[:, index]) .+
                              fraction .* @view(states[:, index + 1])))
    end
    return collect(float.(@view states[:, end]))
end

function _sax_nonregularized_surface_diagnostics(
        states::AbstractMatrix{<:Real}, gamma::Real)
    pressure = vec(sum(@view states[3:2:end, :]; dims=1))
    pressure_drop = float(gamma) .- pressure
    reed_opening = collect(float.(@view states[1, :])) .+ 1
    nearest_contact = argmin(abs.(reed_opening))
    contact_crossing_speed = abs(float(states[2, nearest_contact]))
    crosses_reed = minimum(reed_opening) <= 0 <= maximum(reed_opening)
    return (
        minimum_reed_opening=float(minimum(reed_opening)),
        maximum_reed_opening=float(maximum(reed_opening)),
        minimum_absolute_reed_opening=float(minimum(abs, reed_opening)),
        minimum_pressure_drop=float(minimum(pressure_drop)),
        maximum_pressure_drop=float(maximum(pressure_drop)),
        minimum_absolute_pressure_drop=float(minimum(abs, pressure_drop)),
        crosses_reed_contact=crosses_reed,
        contact_crossing_speed=contact_crossing_speed,
        possible_grazing_contact=crosses_reed &&
            contact_crossing_speed <= 1e-4,
        crosses_flow_reversal=minimum(pressure_drop) <= 0 <= maximum(pressure_drop),
    )
end

function _sax_nonregularized_block_summary(
        times, states, gamma, raw_model, settings;
        preferred_register::Integer=0,
        preferred_family::Symbol=:unknown)
    start = clamp(
        floor(Int, (1 - settings.tail_fraction) * size(states, 2)) + 1,
        1, size(states, 2))
    tail_states = @view states[:, start:end]
    tail_times = @view times[start:end]
    modal_amplitudes = [
        begin
            i = 2mode + 1
            center1 = mean(@view tail_states[i, :])
            center2 = mean(@view tail_states[i + 1, :])
            mean(sqrt.((@view(tail_states[i, :]) .- center1).^2 .+
                       (@view(tail_states[i + 1, :]) .- center2).^2))
        end for mode in 1:min(3, settings.nmodes)
    ]
    split = max(1, size(tail_states, 2) ÷ 2)
    half_amplitudes = [
        begin
            i = 2mode + 1
            function half_mean(columns)
                c1 = mean(@view tail_states[i, columns])
                c2 = mean(@view tail_states[i + 1, columns])
                mean(sqrt.((@view(tail_states[i, columns]) .- c1).^2 .+
                           (@view(tail_states[i + 1, columns]) .- c2).^2))
            end
            late_start = min(split + 1, size(tail_states, 2))
            (half_mean(1:split), half_mean(late_start:size(tail_states, 2)))
        end for mode in 1:min(3, settings.nmodes)
    ]
    within_drift = maximum(
        abs(pair[2] - pair[1]) /
        max(modal_amplitudes[index], settings.activation_threshold)
        for (index, pair) in enumerate(half_amplitudes))
    mode_order = sortperm(modal_amplitudes[1:min(2, length(modal_amplitudes))];
                          rev=true)
    if preferred_register in (1, 2)
        filter!(!=(preferred_register), mode_order)
        pushfirst!(mode_order, preferred_register)
    end
    sections = [_sax_nonregularized_section(
        tail_times, tail_states, mode, settings) for mode in mode_order]
    valid_sections = [section for section in sections if section.valid]
    selected = isempty(valid_sections) ?
        sections[argmin(section.recurrence_error for section in sections)] :
        valid_sections[argmin(section.recurrence_error for section in valid_sections)]
    # The full-state recurrence period, not the number of crossings of the
    # selected modal section, defines this internal temporal-family candidate.
    # It does not define the played register in the paper map; that independent
    # classification uses the validated mode-1/mode-2 amplitude contrast.  A
    # low-register orbit with a strong second harmonic can cross the mode-2
    # section twice per global period, and the near-octave relation can also
    # make low P1 and high P2 periods almost indistinguishable.
    mode_periods = [2pi / float(raw_model.ω[mode]) for mode in 1:2]
    family_references = (
        (family=:low_p1, register=1, generation=1,
         period=mode_periods[1]),
        (family=:high_p1, register=2, generation=1,
         period=mode_periods[2]),
        (family=:low_p2, register=1, generation=2,
         period=2mode_periods[1]),
        (family=:high_p2, register=2, generation=2,
         period=2mode_periods[2]),
    )
    errors = isfinite(selected.period) && selected.period > 0 ?
        [abs(log(selected.period / reference.period))
         for reference in family_references] : fill(Inf, 4)
    minimum_error = minimum(errors)
    tied = findall(error -> error <= minimum_error + 0.03, errors)
    preferred_index = findfirst(reference ->
        reference.family == preferred_family, family_references)
    selected_index = !isnothing(preferred_index) && preferred_index in tied ?
        preferred_index : first(tied)
    family_reference = family_references[selected_index]
    register = isfinite(minimum_error) ? family_reference.register :
        preferred_register in (1, 2) ? Int(preferred_register) : 0
    family = isfinite(minimum_error) ? family_reference.family : :unresolved
    period_generation = isfinite(minimum_error) ?
        family_reference.generation : 0
    order = selected.return_order
    active = sum(modal_amplitudes[1:min(2, end)]) >=
        settings.activation_threshold
    stationary = within_drift <= settings.within_block_drift_tolerance
    surface = _sax_nonregularized_surface_diagnostics(tail_states, gamma)
    return (
        active=active,
        stationary=stationary,
        within_block_drift=float(within_drift),
        modal_amplitudes=collect(float.(modal_amplitudes)),
        register_contrast=float((modal_amplitudes[1] - modal_amplitudes[2]) /
            (modal_amplitudes[1] + modal_amplitudes[2] + eps(Float64))),
        family=family,
        register=Int8(register),
        period_generation=Int8(period_generation),
        return_order=Int8(order),
        period=selected.period,
        carrier_period=selected.carrier_period,
        recurrence_error=selected.recurrence_error,
        period_cv=selected.period_cv,
        recurrence_valid=selected.valid,
        section_mode=selected.mode,
        crossings=selected.crossings,
        phase_state=selected.phase_state,
        total_pressure_phase_state=
            _sax_nonregularized_total_pressure_phase(tail_states),
        surface=surface,
        section=selected,
    )
end

function _sax_nonregularized_stationary_window(
        summaries, settings::SaxNonregularizedMultistabilitySettings)
    required = settings.required_stationary_blocks
    length(summaries) >= required || return (
        accepted=false, amplitude_drift=Inf, period_drift=Inf)
    window = summaries[(end - required + 1):end]
    all(summary -> summary.active && summary.stationary &&
                    summary.recurrence_valid, window) || return (
        accepted=false, amplitude_drift=Inf, period_drift=Inf)
    families = Set(summary.family for summary in window)
    length(families) == 1 && first(families) != :unresolved || return (
        accepted=false, amplitude_drift=Inf, period_drift=Inf)
    amplitude_matrix = hcat((summary.modal_amplitudes for summary in window)...)
    amplitude_mean = vec(mean(amplitude_matrix; dims=2))
    amplitude_drift = maximum(
        (maximum(@view amplitude_matrix[row, :]) -
         minimum(@view amplitude_matrix[row, :])) /
        max(amplitude_mean[row], settings.activation_threshold)
        for row in axes(amplitude_matrix, 1))
    periods = [summary.period for summary in window]
    period_drift = (maximum(periods) - minimum(periods)) /
        max(abs(mean(periods)), eps(Float64))
    return (
        accepted=amplitude_drift <=
            settings.across_block_amplitude_tolerance &&
            period_drift <= settings.across_block_period_tolerance,
        amplitude_drift=float(amplitude_drift),
        period_drift=float(period_drift),
    )
end

function _sax_nonregularized_converge(
        initial_state, gamma::Real, zeta::Real,
        raw_model::NamedTuple,
        settings::SaxNonregularizedMultistabilitySettings,
        source;
        preferred_register::Integer=0,
        block_time::Real=settings.block_time,
        maximum_blocks::Integer=settings.maximum_blocks,
        solver_kind::Symbol=:tsit5)
    state = collect(float.(initial_state))
    summaries = Any[]
    last_history = nothing
    window = (accepted=false, amplitude_drift=Inf, period_drift=Inf)
    for block in 1:Int(maximum_blocks)
        history = _sax_nonregularized_solve(
            state, gamma, zeta, raw_model, settings, block_time;
            solver_kind=solver_kind, history=true)
        state = history.terminal_state
        summary = _sax_nonregularized_block_summary(
            history.times, history.states, gamma, raw_model, settings;
            preferred_register=preferred_register,
            preferred_family=hasproperty(source, :family) ?
                source.family : :unknown)
        push!(summaries, merge(summary, (block=block,)))
        last_history = history
        window = _sax_nonregularized_stationary_window(summaries, settings)
        window.accepted && break
    end
    selected = last(summaries)
    if !window.accepted
        reason = !selected.active ? :inactive :
            !selected.recurrence_valid ? :nonperiodic_or_unresolved :
            !selected.stationary ? :within_block_drift :
            :across_block_drift
        return (
            accepted=false, reason=reason,
            family=selected.family, register=selected.register,
            period_generation=selected.period_generation,
            return_order=selected.return_order,
            period=selected.period,
            carrier_period=selected.carrier_period,
            recurrence_error=selected.recurrence_error,
            period_cv=selected.period_cv,
            terminal_state=state,
            blocks=length(summaries),
            block_summaries=[_sax_nonregularized_strip_summary(summary)
                             for summary in summaries],
        )
    end
    quality = selected.recurrence_error + selected.period_cv +
        window.amplitude_drift + window.period_drift
    return (
        accepted=true,
        reason=:converged_periodic_attractor,
        family=selected.family,
        register=selected.register,
        period_generation=selected.period_generation,
        return_order=selected.return_order,
        period=selected.period,
        carrier_period=selected.carrier_period,
        recurrence_error=selected.recurrence_error,
        period_cv=selected.period_cv,
        modal_amplitudes=selected.modal_amplitudes,
        register_contrast=selected.register_contrast,
        phase_state=selected.phase_state,
        total_pressure_phase_state=selected.total_pressure_phase_state,
        terminal_state=state,
        surface=selected.surface,
        blocks=length(summaries),
        across_block_amplitude_drift=window.amplitude_drift,
        across_block_period_drift=window.period_drift,
        quality=float(quality),
        source=source,
        validation=(
            status=:not_run,
            attracting=false,
            level=:unvalidated_candidate,
        ),
        block_summaries=[_sax_nonregularized_strip_summary(summary)
                         for summary in summaries],
    )
end

function _sax_nonregularized_strip_summary(summary)
    return (
        block=hasproperty(summary, :block) ? Int(summary.block) : 0,
        active=Bool(summary.active),
        stationary=Bool(summary.stationary),
        within_block_drift=float(summary.within_block_drift),
        modal_amplitudes=collect(float.(summary.modal_amplitudes)),
        register_contrast=float(summary.register_contrast),
        family=summary.family,
        register=Int8(summary.register),
        period_generation=Int8(summary.period_generation),
        return_order=Int8(summary.return_order),
        period=float(summary.period),
        carrier_period=float(summary.carrier_period),
        recurrence_error=float(summary.recurrence_error),
        period_cv=float(summary.period_cv),
        recurrence_valid=Bool(summary.recurrence_valid),
        section_mode=Int(summary.section_mode),
        crossings=Int(summary.crossings),
        surface=summary.surface,
    )
end

_sax_nonregularized_p1_family(register::Integer) =
    register == 1 ? :low_p1 : :high_p1

function _sax_nonregularized_attempt_summary(route_key, source, result)
    return (
        route_key=String(route_key),
        source_key=String(source.key),
        source_kind=source.kind,
        requested_register=Int(source.register),
        status=result.accepted ? :accepted : :rejected,
        reason=result.reason,
        family=result.family,
        register=Int8(result.register),
        return_order=Int8(result.return_order),
        period=float(result.period),
        recurrence_error=float(result.recurrence_error),
        blocks=Int(result.blocks),
        saved_at_unix=time(),
    )
end

function _sax_nonregularized_add_candidate(point, candidate)
    attractors = copy(point.attractors)
    index = findfirst(attractor -> attractor.family == candidate.family,
                      attractors)
    support = (
        source_key=String(candidate.source.key),
        source_kind=candidate.source.kind,
        route=candidate.source.route,
        quality=float(candidate.quality),
    )
    if isnothing(index)
        push!(attractors, merge(candidate, (supports=Any[support],)))
    else
        existing = attractors[index]
        supports = hasproperty(existing, :supports) ?
            copy(existing.supports) : Any[]
        any(item -> item.source_key == support.source_key &&
                    item.route == support.route, supports) || push!(supports, support)
        existing_validated = hasproperty(existing, :validation) &&
            hasproperty(existing.validation, :attracting) &&
            Bool(existing.validation.attracting)
        existing_rejected = hasproperty(existing, :validation) &&
            existing.validation.status == :complete && !existing_validated
        # A newly converged route must be allowed to replace a representative
        # that failed the finite-amplitude validation, even when its scalar
        # recurrence score is marginally larger. This is important near the
        # DH strip, where different routes can land at different phases or on
        # different connected pieces of a narrow basin.
        replacement = existing_rejected ? candidate :
            !existing_validated && candidate.quality < existing.quality ?
                candidate : existing
        attractors[index] = merge(replacement, (supports=supports,))
    end
    return merge(point, (attractors=attractors,))
end

function _sax_nonregularized_has_route(point, route_key::AbstractString)
    return any(attempt -> attempt.route_key == route_key, point.attempts)
end

function _sax_nonregularized_has_family(point, family::Symbol;
                                        validated::Bool=false)
    return any(point.attractors) do attractor
        attractor.family == family || return false
        !validated && return true
        hasproperty(attractor, :validation) &&
            hasproperty(attractor.validation, :attracting) &&
            Bool(attractor.validation.attracting)
    end
end

function _sax_nonregularized_record_attempt(
        point, route_key, source, result)
    attempts = copy(point.attempts)
    push!(attempts, _sax_nonregularized_attempt_summary(
        route_key, source, result))
    updated = merge(point, (attempts=attempts,))
    return result.accepted ?
        _sax_nonregularized_add_candidate(updated, result) : updated
end

# ---------------------------------------------------------------------------
# Guided exact conversion, bidirectional propagation, and DH-strip repair
# ---------------------------------------------------------------------------

function _sax_nonregularized_parallel_map!(
        evaluate!, count::Integer; parallel::Bool=true,
        verbosity::Integer=1, stage::AbstractString="stage")
    completed = Threads.Atomic{Int}(0)
    function wrapped(index)
        evaluate!(index)
        done = Threads.atomic_add!(completed, 1) + 1
        if verbosity > 0 &&
                (done == count || done % max(1, Int(count) ÷ 20) == 0)
            @info(
                "Non-regularized multistability progress",
                stage, completed=done, total=count,
                percent=round(100done / max(count, 1); digits=1),
                threads=Threads.nthreads(),
            )
        end
        return nothing
    end
    if parallel && Threads.nthreads() > 1
        Threads.@threads for index in 1:Int(count)
            wrapped(index)
        end
    else
        for index in 1:Int(count)
            wrapped(index)
        end
    end
    return Int(completed[])
end

function _sax_nonregularized_try_source(
        point, route_key::AbstractString, source,
        target_gamma::Real, target_zeta::Real,
        raw_model::NamedTuple,
        settings::SaxNonregularizedMultistabilitySettings;
        waypoints=Tuple{Float64,Float64}[],
        block_time::Real=settings.block_time,
        maximum_blocks::Integer=settings.maximum_blocks)
    _sax_nonregularized_has_route(point, route_key) && return point
    result = try
        state = source.kind in (:regularized_guide, :regularized_p2_guide) ?
            _sax_nonregularized_transport_regularized_source(
                source, target_gamma, target_zeta, raw_model, settings) :
            source.kind == :equilibrium_mode_excitation ? copy(source.state) :
            _sax_nonregularized_transport_exact_source(
                source, target_gamma, target_zeta, raw_model, settings;
                waypoints=waypoints)
        run_source = merge(source, (
            route=String(route_key),
            target_gamma=float(target_gamma),
            target_zeta=float(target_zeta),
        ))
        _sax_nonregularized_converge(
            state, target_gamma, target_zeta, raw_model, settings, run_source;
            preferred_register=source.register,
            block_time=block_time,
            maximum_blocks=maximum_blocks)
    catch err
        err isa InterruptException && rethrow()
        (
            accepted=false,
            reason=:integration_failure,
            family=:unresolved,
            register=Int8(0),
            return_order=Int8(0),
            period=NaN,
            carrier_period=NaN,
            recurrence_error=Inf,
            period_cv=Inf,
            terminal_state=copy(source.state),
            blocks=0,
            block_summaries=Any[],
            exception_type=Symbol(nameof(typeof(err))),
            error=sprint(showerror, err),
        )
    end
    return _sax_nonregularized_record_attempt(
        point, route_key, source, result)
end

"""
    compute_sax_nonregularized_guided_points(...)

Transport validated exact P1 snapshot states and branch-labelled high-P2
checkpoints to the target parameters. A regularized P2 source is converted
through the eta ladder. Each target also has a legacy finite-amplitude P1
modal seed, but that fallback is evidence only and is never interpreted as
proof of absence.
"""
function compute_sax_nonregularized_guided_points(
        raw_model::NamedTuple, guide_progress, guide_model::NamedTuple,
        paths, guide_signature;
        p2_guide_anchors=Any[],
        settings::SaxNonregularizedMultistabilitySettings=
            sax_nonregularized_multistability_settings(:final),
        resume::Bool=true,
        parallel::Bool=true,
        verbosity::Integer=1)
    _validate_sax_nonregularized_multistability_settings(settings)
    grid = _sax_nonregularized_grid(settings)
    targets = collect(Iterators.product(grid.gamma, grid.zeta))
    guide_anchors = sax_nonregularized_guide_anchors(
        guide_progress, guide_model, settings)
    isempty(guide_anchors) && @warn(
        "The exact P1 seed snapshot has no stable anchors; only explicitly labelled fallback attacks will run")
    started = time()
    function evaluate(index)
        gamma, zeta = targets[index]
        point = resume ? _sax_nonregularized_point_or_empty(
            paths, gamma, zeta, raw_model, settings, guide_signature) :
            _sax_nonregularized_empty_point(gamma, zeta)
        for register in (1, 2)
            family = _sax_nonregularized_p1_family(register)
            _sax_nonregularized_has_family(point, family) && continue
            sources = _sax_nonregularized_select_guide_sources(
                guide_anchors, register, gamma, zeta, settings)
            fallback = _sax_nonregularized_fallback_source(
                gamma, zeta, register, raw_model, settings)
            for source in vcat(sources, Any[fallback])
                route = source.kind == :exact_atlas_guide ?
                    "exact_seed_$(source.key)" :
                    source.kind == :regularized_guide ?
                    "guided_eta0_$(source.key)" : "fallback_$(source.key)"
                point = _sax_nonregularized_try_source(
                    point, route, source, gamma, zeta,
                    raw_model, settings)
                _sax_nonregularized_has_family(point, family) && break
            end
        end
        if settings.track_high_p2 &&
                !_sax_nonregularized_has_family(point, :high_p2)
            sources = _sax_nonregularized_select_guide_family_sources(
                p2_guide_anchors, :high_p2, gamma, zeta, settings)
            for source in sources
                route = "guided_eta0_$(source.key)"
                point = _sax_nonregularized_try_source(
                    point, route, source, gamma, zeta,
                    raw_model, settings)
                _sax_nonregularized_has_family(point, :high_p2) && break
            end
        end
        _save_sax_nonregularized_point(
            paths, point, raw_model, settings, guide_signature)
        return nothing
    end
    _sax_nonregularized_parallel_map!(
        evaluate, length(targets); parallel=parallel,
        verbosity=verbosity, stage="guided eta-to-exact conversion")
    return merge(
        load_sax_nonregularized_multistability_progress(
            raw_model, paths, guide_signature; settings=settings),
        (elapsed_seconds=round(time() - started; digits=1),
         guide_anchors=length(guide_anchors),
         p2_guide_anchors=length(p2_guide_anchors)))
end

function _sax_nonregularized_point_or_empty(
        paths, gamma, zeta, raw_model, settings, guide_signature)
    return _load_sax_nonregularized_point(
        paths, gamma, zeta, raw_model, settings, guide_signature)
end

function _sax_nonregularized_snapshot(
        raw_model::NamedTuple, paths, guide_signature,
        settings::SaxNonregularizedMultistabilitySettings)
    grid = _sax_nonregularized_grid(settings)
    points = Dict{Tuple{Float64,Float64},Any}()
    sources = Any[]
    for zeta in grid.zeta, gamma in grid.gamma
        point = _sax_nonregularized_point_or_empty(
            paths, gamma, zeta, raw_model, settings, guide_signature)
        points[(float(gamma), float(zeta))] = point
        for attractor in point.attractors
            attractor.family in (:low_p1, :high_p1, :low_p2, :high_p2) ||
                continue
            push!(sources, (
                key="exact_$(attractor.family)_g$(_sax_nonregularized_tag(gamma))_z$(_sax_nonregularized_tag(zeta))",
                kind=:exact_attractor,
                register=Int(attractor.register),
                family=attractor.family,
                gamma=float(gamma), zeta=float(zeta), eta=0.0,
                period=float(attractor.period),
                state=collect(float.(attractor.phase_state)),
                validated=hasproperty(attractor, :validation) &&
                    hasproperty(attractor.validation, :attracting) &&
                    Bool(attractor.validation.attracting),
            ))
        end
    end
    return (grid=grid, points=points, sources=sources)
end

function _sax_nonregularized_select_exact_sources(
        sources, register::Integer, gamma::Real, zeta::Real,
        settings::SaxNonregularizedMultistabilitySettings;
        dh::Bool=false,
        family::Symbol=_sax_nonregularized_p1_family(register))
    maximum_gamma = dh ? settings.dh_source_gamma_distance :
        settings.propagation_max_gamma_distance
    maximum_zeta = dh ? settings.dh_source_zeta_distance :
        settings.propagation_max_zeta_distance
    candidates = [source for source in sources
        if source.family == family &&
           abs(source.gamma - gamma) <= maximum_gamma &&
           abs(source.zeta - zeta) <= maximum_zeta &&
           !(isapprox(source.gamma, gamma; atol=1e-12, rtol=0) &&
             isapprox(source.zeta, zeta; atol=1e-12, rtol=0))]
    maximum = dh ? max(settings.propagation_sources_per_register, 8) :
        settings.propagation_sources_per_register
    return _sax_nonregularized_diverse_sources(
        candidates, gamma, zeta, maximum)
end

"""Expand exact low/high anchors through deterministic bidirectional snapshots."""
function compute_sax_nonregularized_propagation(
        raw_model::NamedTuple, paths, guide_signature;
        settings::SaxNonregularizedMultistabilitySettings=
            sax_nonregularized_multistability_settings(:final),
        resume::Bool=true,
        parallel::Bool=true,
        verbosity::Integer=1)
    started = time()
    pass_summaries = Any[]
    total_passes = max(
        settings.propagation_passes,
        settings.track_high_p2 ? settings.p2_propagation_passes : 0)
    for pass in 1:total_passes
        snapshot = _sax_nonregularized_snapshot(
            raw_model, paths, guide_signature, settings)
        tasks = Any[]
        for zeta in snapshot.grid.zeta, gamma in snapshot.grid.gamma
            point = snapshot.points[(float(gamma), float(zeta))]
            missing_families = Symbol[]
            if pass <= settings.propagation_passes
                for family in (:low_p1, :high_p1)
                    _sax_nonregularized_has_family(point, family) ||
                        push!(missing_families, family)
                end
            end
            settings.track_high_p2 &&
                pass <= settings.p2_propagation_passes &&
                !_sax_nonregularized_has_family(point, :high_p2) &&
                push!(missing_families, :high_p2)
            isempty(missing_families) || push!(tasks, (
                gamma=float(gamma), zeta=float(zeta),
                missing_families=missing_families))
        end
        isempty(tasks) && break
        new_attractors = Threads.Atomic{Int}(0)
        function evaluate(index)
            task = tasks[index]
            point = _sax_nonregularized_point_or_empty(
                paths, task.gamma, task.zeta, raw_model, settings,
                guide_signature)
            before = length(point.attractors)
            for family in task.missing_families
                register = family in (:low_p1, :low_p2) ? 1 : 2
                _sax_nonregularized_has_family(point, family) && continue
                sources = _sax_nonregularized_select_exact_sources(
                    snapshot.sources, register, task.gamma, task.zeta,
                    settings; family=family)
                for source in sources
                    route = "propagate_$(source.key)_to_g$(_sax_nonregularized_tag(task.gamma))_z$(_sax_nonregularized_tag(task.zeta))"
                    point = _sax_nonregularized_try_source(
                        point, route, source, task.gamma, task.zeta,
                        raw_model, settings)
                    _sax_nonregularized_has_family(point, family) && break
                end
            end
            length(point.attractors) > before &&
                Threads.atomic_add!(new_attractors,
                                    length(point.attractors) - before)
            _save_sax_nonregularized_point(
                paths, point, raw_model, settings, guide_signature)
            return nothing
        end
        _sax_nonregularized_parallel_map!(
            evaluate, length(tasks); parallel=parallel,
            verbosity=verbosity, stage="exact bidirectional propagation pass $(pass)")
        push!(pass_summaries, (
            pass=pass, targets=length(tasks),
            new_attractors=Int(new_attractors[])))
        new_attractors[] == 0 && break
    end
    return merge(
        load_sax_nonregularized_multistability_progress(
            raw_model, paths, guide_signature; settings=settings),
        (elapsed_seconds=round(time() - started; digits=1),
         passes=pass_summaries))
end

function _sax_nonregularized_dh_waypoints(source, target, bridge_gamma)
    return Tuple{Float64,Float64}[
        (float(bridge_gamma), float(source.zeta)),
        (float(bridge_gamma), float(target.zeta)),
    ]
end

"""
Repair missing low-register cells around the DH neighbourhood using independent
routes from both zeta sides and rectangular paths through safe bridge gamma
values.  It records every failed route and never paints between successes.
"""
function compute_sax_nonregularized_dh_repair(
        raw_model::NamedTuple, guide_progress, guide_model::NamedTuple,
        paths, guide_signature;
        settings::SaxNonregularizedMultistabilitySettings=
            sax_nonregularized_multistability_settings(:final),
        resume::Bool=true,
        parallel::Bool=true,
        verbosity::Integer=1)
    started = time()
    snapshot = _sax_nonregularized_snapshot(
        raw_model, paths, guide_signature, settings)
    guide_anchors = sax_nonregularized_guide_anchors(
        guide_progress, guide_model, settings)
    targets = Any[]
    for zeta in snapshot.grid.zeta, gamma in snapshot.grid.gamma
        settings.dh_zeta_range[1] <= zeta <= settings.dh_zeta_range[2] ||
            continue
        settings.dh_gamma_range[1] <= gamma <= settings.dh_gamma_range[2] ||
            continue
        point = snapshot.points[(float(gamma), float(zeta))]
        _sax_nonregularized_has_family(point, :low_p1) && continue
        push!(targets, (gamma=float(gamma), zeta=float(zeta)))
    end
    repaired = Threads.Atomic{Int}(0)
    function evaluate(index)
        target = targets[index]
        point = _sax_nonregularized_point_or_empty(
            paths, target.gamma, target.zeta,
            raw_model, settings, guide_signature)
        # First try eta-homotopy anchors selected from both sides of the strip.
        for source in _sax_nonregularized_select_guide_sources(
                guide_anchors, 1, target.gamma, target.zeta, settings; dh=true)
            route = "dh_guide_$(source.key)"
            point = _sax_nonregularized_try_source(
                point, route, source, target.gamma, target.zeta,
                raw_model, settings;
                block_time=settings.block_time * settings.dh_time_factor,
                maximum_blocks=ceil(Int,
                    settings.maximum_blocks * settings.dh_time_factor))
            _sax_nonregularized_has_family(point, :low_p1) && break
        end
        # If the direct eta route did not work, transport already accepted
        # eta=0 cycles through several independent rectangular paths.
        if !_sax_nonregularized_has_family(point, :low_p1)
            sources = _sax_nonregularized_select_exact_sources(
                snapshot.sources, 1, target.gamma, target.zeta,
                settings; dh=true)
            for source in sources
                routes = vcat(
                    [Tuple{Float64,Float64}[]],
                    [_sax_nonregularized_dh_waypoints(
                        source, target, bridge)
                     for bridge in settings.dh_bridge_gammas],
                )
                for (route_index, waypoints) in enumerate(routes)
                    route = "dh_exact_$(source.key)_route$(route_index)"
                    point = _sax_nonregularized_try_source(
                        point, route, source, target.gamma, target.zeta,
                        raw_model, settings;
                        waypoints=waypoints,
                        block_time=settings.block_time * settings.dh_time_factor,
                        maximum_blocks=ceil(Int,
                            settings.maximum_blocks * settings.dh_time_factor))
                    _sax_nonregularized_has_family(point, :low_p1) && break
                end
                _sax_nonregularized_has_family(point, :low_p1) && break
            end
        end
        _sax_nonregularized_has_family(point, :low_p1) &&
            Threads.atomic_add!(repaired, 1)
        _save_sax_nonregularized_point(
            paths, point, raw_model, settings, guide_signature)
        return nothing
    end
    _sax_nonregularized_parallel_map!(
        evaluate, length(targets); parallel=parallel,
        verbosity=verbosity, stage="DH low-register multi-route repair")
    return merge(
        load_sax_nonregularized_multistability_progress(
            raw_model, paths, guide_signature; settings=settings),
        (elapsed_seconds=round(time() - started; digits=1),
         targeted_points=length(targets), repaired_points=Int(repaired[])))
end

# ---------------------------------------------------------------------------
# Local attraction validation and optional smooth-cycle Periodic Schur
# ---------------------------------------------------------------------------

function _sax_nonregularized_section_distance(point, prototypes, scales)
    isempty(prototypes) && return Inf
    return minimum(
        norm((point .- prototype) ./ scales) / sqrt(length(point))
        for prototype in prototypes)
end

function _sax_nonregularized_return_test(
        candidate, raw_model::NamedTuple,
        settings::SaxNonregularizedMultistabilitySettings)
    reference_history = _sax_nonregularized_solve(
        candidate.phase_state,
        candidate.source.target_gamma,
        candidate.source.target_zeta,
        raw_model, settings, settings.validation_time;
        solver_kind=:vern9, history=true,
        reltol=min(settings.reltol, 2e-9),
        abstol=min(settings.abstol, 2e-11))
    reference = _sax_nonregularized_block_summary(
        reference_history.times, reference_history.states,
        candidate.source.target_gamma, raw_model, settings;
        preferred_register=Int(candidate.register),
        preferred_family=candidate.family)
    solver_agreement = reference.recurrence_valid &&
        reference.family == candidate.family &&
        abs(reference.period - candidate.period) /
            max(abs(candidate.period), eps(Float64)) <=
                2settings.across_block_period_tolerance
    order = max(1, Int(reference.return_order))
    section = reference.section
    prototype_count = min(order, size(section.section_points, 2))
    prototypes = [collect(float.(@view section.section_points[:, index]))
        for index in (size(section.section_points, 2) - prototype_count + 1):
                     size(section.section_points, 2)]
    directions = Any[]
    dimension = length(candidate.phase_state)
    amplitude_scale = max(norm(candidate.phase_state), 0.1)
    for direction_index in 1:settings.perturbation_directions,
            epsilon in settings.perturbation_scales
        vector = [
            sin((direction_index + 1) * state_index) +
            cos((2direction_index + 1) * state_index / 2)
            for state_index in 1:dimension
        ]
        vector ./= max(norm(vector), eps(Float64))
        direction_index % 2 == 0 && (vector .*= -1)
        initial = candidate.phase_state .+
            max(float(epsilon) * amplitude_scale, 1e-8) .* vector
        diagnostic = try
            history = _sax_nonregularized_solve(
                initial,
                candidate.source.target_gamma,
                candidate.source.target_zeta,
                raw_model, settings, settings.validation_time;
                solver_kind=:tsit5, history=true)
            summary = _sax_nonregularized_block_summary(
                history.times, history.states,
                candidate.source.target_gamma, raw_model, settings;
                preferred_register=Int(candidate.register),
                preferred_family=candidate.family)
            points = summary.section.section_points
            distances = [_sax_nonregularized_section_distance(
                @view(points[:, item]), prototypes, section.scales)
                for item in axes(points, 2)]
            usable = length(distances) >= max(4, 2order)
            first_count = usable ? min(max(2, order), length(distances) ÷ 2) : 0
            last_count = first_count
            initial_distance = usable ?
                median(@view distances[2:(first_count + 1)]) : Inf
            final_distance = usable ?
                median(@view distances[(end - last_count + 1):end]) : Inf
            ratio = final_distance /
                max(initial_distance, 0.1settings.return_distance_tolerance)
            same_family = summary.recurrence_valid &&
                summary.family == candidate.family
            contracting = same_family &&
                final_distance <= settings.return_distance_tolerance &&
                (initial_distance <= settings.return_distance_tolerance ||
                 ratio <= settings.maximum_return_contraction_ratio)
            (
                status=:complete,
                accepted=contracting,
                direction=direction_index,
                epsilon=float(epsilon),
                same_family=same_family,
                family=summary.family,
                initial_return_distance=float(initial_distance),
                final_return_distance=float(final_distance),
                contraction_ratio=float(ratio),
                recurrence_error=float(summary.recurrence_error),
            )
        catch err
            err isa InterruptException && rethrow()
            (
                status=:failed,
                accepted=false,
                direction=direction_index,
                epsilon=float(epsilon),
                exception_type=Symbol(nameof(typeof(err))),
                error=sprint(showerror, err),
            )
        end
        push!(directions, diagnostic)
    end
    successes = count(diagnostic -> diagnostic.accepted, directions)
    required = ceil(Int,
        settings.perturbation_success_fraction * length(directions))
    return (
        status=:complete,
        solver_agreement=solver_agreement,
        attracting=solver_agreement && successes >= required,
        successes=successes,
        required=required,
        total=length(directions),
        reference=(
            family=reference.family,
            period=float(reference.period),
            recurrence_error=float(reference.recurrence_error),
            period_cv=float(reference.period_cv),
            modal_amplitudes=collect(float.(reference.modal_amplitudes)),
            phase_state=collect(float.(reference.phase_state)),
            surface=reference.surface,
        ),
        perturbations=directions,
    )
end

function _sax_nonregularized_pqz_eligibility(
        surface, settings::SaxNonregularizedMultistabilitySettings)
    eligible = !surface.crosses_flow_reversal &&
        surface.minimum_absolute_pressure_drop > settings.pqz_surface_clearance &&
        (!surface.crosses_reed_contact ||
         surface.contact_crossing_speed >
            settings.pqz_reed_grazing_velocity_tolerance)
    reason = eligible ? :eligible :
        surface.crosses_flow_reversal ? :flow_reversal_crossing :
        surface.minimum_absolute_pressure_drop <=
            settings.pqz_surface_clearance ? :flow_reversal_clearance :
        :grazing_reed_contact
    return (eligible=eligible, reason=reason)
end

function _sax_nonregularized_pqz_scheduled(
        candidate, settings::SaxNonregularizedMultistabilitySettings)
    settings.pqz_enabled || return false
    grid = _sax_nonregularized_grid(settings)
    gamma = float(candidate.source.target_gamma)
    zeta = float(candidate.source.target_zeta)
    gamma_index = argmin(abs.(grid.gamma .- gamma))
    zeta_index = argmin(abs.(grid.zeta .- zeta))
    on_stride = ((gamma_index - 1) % settings.pqz_gamma_stride == 0 ||
                 gamma_index == length(grid.gamma)) &&
                ((zeta_index - 1) % settings.pqz_zeta_stride == 0 ||
                 zeta_index == length(grid.zeta))
    in_dh = settings.dh_gamma_range[1] <= gamma <=
                settings.dh_gamma_range[2] &&
            settings.dh_zeta_range[1] <= zeta <=
                settings.dh_zeta_range[2]
    gamma_spacing = length(grid.gamma) > 1 ?
        minimum(diff(grid.gamma)) : settings.parameter_gamma_step
    on_dh_bridge = in_dh &&
        minimum(abs(gamma - bridge) for bridge in settings.dh_bridge_gammas) <=
            gamma_spacing / 2 + 10eps(Float64) &&
        ((zeta_index - 1) % settings.pqz_dh_zeta_stride == 0 ||
         zeta_index == length(grid.zeta))
    return on_stride || on_dh_bridge
end

function _sax_nonregularized_pqz_validation(
        candidate, return_test, raw_model::NamedTuple,
        settings::SaxNonregularizedMultistabilitySettings)
    settings.pqz_enabled || return (
        status=:disabled, eligible=false, stable=false,
        reason="Periodic Schur disabled by the selected profile")
    surface = return_test.reference.surface
    eligibility = _sax_nonregularized_pqz_eligibility(surface, settings)
    eligibility.eligible || return (
        status=:ineligible_nonsmooth_crossing,
        eligible=false, stable=false,
        reason=eligibility.reason,
        surface=surface,
    )
    gamma = float(candidate.source.target_gamma)
    zeta = float(candidate.source.target_zeta)
    period = float(return_test.reference.period)
    try
        bifurcation_settings = sax_bifurcation_settings(
            settings.profile == :final ? :final : :pilot;
            nmodes=settings.nmodes,
            po_collocation_intervals=settings.pqz_collocation_intervals,
            po_collocation_degree=settings.pqz_collocation_degree,
            newton_tol=settings.pqz_newton_tolerance,
            smoothness_tol=min(
                settings.pqz_surface_clearance / 10, 1e-7),
            allow_transverse_reed_contact=true,
        )
        seed_state = return_test.reference.phase_state
        tail = _sax_nonregularized_solve(
            seed_state, gamma, zeta, raw_model, settings, 2.2period;
            solver_kind=:vern9, history=true, dense_output=true,
            reltol=min(settings.reltol, 1e-10),
            abstol=min(settings.abstol, 1e-12))
        problem, parameters = _sax_problem_from_checkpoint(
            (state=seed_state, gamma=gamma, zeta=zeta),
            raw_model, bifurcation_settings)
        intervals = settings.pqz_collocation_intervals
        degree = settings.pqz_collocation_degree
        state_dimension = 2 + 2settings.nmodes
        orbit_dimension = state_dimension * (1 + intervals * degree)
        condensed = bifurcation_settings.po_linear_solver == :condensed
        collocation = BK.Collocation(
            intervals, degree;
            N=state_dimension,
            prob_vf=problem,
            ϕ=zeros(orbit_dimension),
            xπ=zeros(orbit_dimension),
            ∂ϕ=zeros(state_dimension, intervals * degree),
            jacobian=condensed ? BK.DenseAnalyticalInplace() :
                BK.DenseAnalytical(),
            update_section_every_step=1,
        )
        guess = BK.generate_solution(
            collocation, time_value -> tail.solution(time_value), period)
        BK.updatesection!(collocation, guess, parameters)
        corrected = BK.newton(
            collocation,
            guess,
            BK.NewtonPar(
                tol=settings.pqz_newton_tolerance,
                max_iterations=settings.pqz_newton_iterations,
                linsolver=condensed ? BK.COPLS() : BK.DefaultLS(),
                verbose=false,
            );
            normN=BK.norminf,
        )
        BK.converged(corrected) || error(
            "exact periodic collocation correction did not converge")
        checkpoint = (
            key="nonregularized_pqz",
            type=:ns,
            mode=Int(candidate.register),
            source_hopf_key="direct_exact_attractor",
            gamma=gamma, zeta=zeta,
            floquet_angle=0.5,
            solution=collect(float.(corrected.u)),
        )
        wrapper, refined_parameters = _sax_periodic_wrapper(
            checkpoint, raw_model, bifurcation_settings)
        residual = float(norm(BK.residual(
            wrapper, checkpoint.solution, refined_parameters), Inf))
        jacobian = BK.jacobian(
            wrapper, checkpoint.solution, refined_parameters)
        discretization = BK.get_discretization(wrapper)
        exponents, _, converged, iterations = SaxFloquetPQZ(
            cyclic_retries=max(8, intervals - 1),
            fallback_to_floquet_coll=false,
        )(discretization, jacobian, state_dimension)
        Bool(converged) || error("Periodic-Schur spectrum did not converge")
        canonical = sax_canonical_floquet_pairs(exponents)
        isempty(canonical.pairs) && error(
            "Periodic-Schur spectrum has no nontrivial Floquet pair")
        dominant = canonical.pairs[
            argmax(pair.growth for pair in canonical.pairs)]
        stable = dominant.growth < -settings.pqz_growth_margin
        neutral = abs(dominant.growth) <= settings.pqz_growth_margin
        orbit = BK.get_periodic_orbit(
            wrapper, checkpoint.solution, refined_parameters)
        orbit_record = _record_sax_periodic_orbit(
            checkpoint.solution, (prob=wrapper, p=refined_parameters))
        return (
            status=:validated,
            eligible=true,
            stable=stable,
            neutral=neutral,
            residual_norm=residual,
            dominant_growth=float(dominant.growth),
            dominant_angle=float(dominant.angle),
            neutral_exponent=canonical.neutral_exponent,
            iterations=Int(iterations),
            mesh=(intervals, degree),
            orbit=(
                period=float(BK.getperiod(
                    wrapper, checkpoint.solution, refined_parameters)),
                minimum_absolute_pressure_drop=
                    float(orbit_record.minimum_absolute_pressure_drop),
                minimum_absolute_reed_opening=
                    float(orbit_record.minimum_absolute_reed_opening),
                crosses_reed_contact=Bool(orbit_record.crosses_reed_contact),
            ),
        )
    catch err
        err isa InterruptException && rethrow()
        return (
            status=:failed,
            eligible=true,
            stable=false,
            exception_type=Symbol(nameof(typeof(err))),
            error=sprint(showerror, err),
        )
    end
end

function _sax_nonregularized_validate_candidate(
        candidate, raw_model::NamedTuple,
        settings::SaxNonregularizedMultistabilitySettings)
    return_test = try
        _sax_nonregularized_return_test(candidate, raw_model, settings)
    catch err
        err isa InterruptException && rethrow()
        return (
            status=:failed,
            attracting=false,
            level=:unresolved,
            exception_type=Symbol(nameof(typeof(err))),
            error=sprint(showerror, err),
        )
    end
    pqz = if _sax_nonregularized_pqz_scheduled(candidate, settings)
        # PQZ uses a dense collocation workspace.  Return-map integrations may
        # run on all Julia threads, but only one factorization is resident at
        # a time so the Final run cannot exhaust RAM.
        lock(_SAX_NONREGULARIZED_MULTISTABILITY_PQZ_LOCK) do
            _sax_nonregularized_pqz_validation(
                candidate, return_test, raw_model, settings)
        end
    else
        (
            status=:not_sampled,
            eligible=false,
            stable=false,
            reason=:sparse_periodic_schur_audit,
        )
    end
    pqz_conflict = pqz.status == :validated && !pqz.stable
    attracting = return_test.attracting && !pqz_conflict
    level = attracting && pqz.status == :validated && pqz.stable ?
        :strict_periodic_schur_and_return_map :
        attracting ? :finite_amplitude_return_map :
        pqz_conflict ? :validation_conflict : :unresolved
    return (
        status=:complete,
        attracting=attracting,
        level=level,
        return_map=return_test,
        periodic_schur=pqz,
        exact_model=true,
        eta=0.0,
        validated_at_unix=time(),
    )
end

"""Validate every exact periodic candidate with perturbed Poincare returns."""
function compute_sax_nonregularized_attraction_validation(
        raw_model::NamedTuple, paths, guide_signature;
        settings::SaxNonregularizedMultistabilitySettings=
            sax_nonregularized_multistability_settings(:final),
        resume::Bool=true,
        parallel::Bool=true,
        verbosity::Integer=1)
    grid = _sax_nonregularized_grid(settings)
    targets = collect(Iterators.product(grid.gamma, grid.zeta))
    started = time()
    validated = Threads.Atomic{Int}(0)
    function evaluate(index)
        gamma, zeta = targets[index]
        point = _sax_nonregularized_point_or_empty(
            paths, gamma, zeta, raw_model, settings, guide_signature)
        attractors = copy(point.attractors)
        changed = false
        for item in eachindex(attractors)
            candidate = attractors[item]
            candidate.family in (
                :low_p1, :high_p1, :low_p2, :high_p2,
                :higher_periodic) || continue
            already = hasproperty(candidate, :validation) &&
                candidate.validation.status == :complete
            resume && already && continue
            validation = _sax_nonregularized_validate_candidate(
                candidate, raw_model, settings)
            attractors[item] = merge(candidate, (validation=validation,))
            changed = true
            validation.status == :complete &&
                Threads.atomic_add!(validated, 1)
        end
        if changed
            point = merge(point, (attractors=attractors,))
            _save_sax_nonregularized_point(
                paths, point, raw_model, settings, guide_signature)
        end
        return nothing
    end
    _sax_nonregularized_parallel_map!(
        evaluate, length(targets); parallel=parallel,
        verbosity=verbosity, stage="exact local-attraction validation")
    return merge(
        load_sax_nonregularized_multistability_progress(
            raw_model, paths, guide_signature; settings=settings),
        (elapsed_seconds=round(time() - started; digits=1),
         validations=Int(validated[])))
end

# ---------------------------------------------------------------------------
# Sparse edge tracking inside validated coexistence only
# ---------------------------------------------------------------------------

function _sax_nonregularized_attractor(point, family::Symbol;
                                       validated::Bool=true)
    index = findfirst(point.attractors) do attractor
        attractor.family == family || return false
        !validated && return true
        hasproperty(attractor, :validation) &&
            hasproperty(attractor.validation, :attracting) &&
            Bool(attractor.validation.attracting)
    end
    return isnothing(index) ? nothing : point.attractors[index]
end

_sax_nonregularized_register_families(register::Integer) =
    Int(register) == 1 ? (:low_p1, :low_p2) :
    Int(register) == 2 ? (:high_p1, :high_p2) : ()

"""
Map an accepted full-state recurrence period to the two principal period
classes displayed in the paper.

`T1` and `T2` denote the recurrence periods associated with the first and
second acoustic resonances. They are deliberately different from the
bifurcation labels P1 and P2: P1 is a period-one branch and P2 is its doubled
branch. Because `2T2 ≈ T1` for the octave-related resonances, a
high-family P2 response is `:t1`, while a low-family P2 response near `2T1`
is `:other`. The latter must not be forced into either principal class.
"""
# Angular frequency is stored under the Greek omega field in the model.
# separate makes synthetic tests and older portable cache records easier to
# support without changing the model file.
function _sax_nonregularized_period_class(
        period::Real, raw_model::NamedTuple{names}) where {names}
    omega_symbol = Symbol("ω")
    omega = hasproperty(raw_model, omega_symbol) ?
        getproperty(raw_model, omega_symbol) : raw_model.omega
    isfinite(period) && period > 0 || return :unresolved
    t1 = 2pi / float(omega[1])
    t2 = 2pi / float(omega[2])
    references = (t1, t2, 2t1, 2t2)
    nearest = argmin(abs(log(float(period) / reference))
                     for reference in references)
    return nearest == 1 || nearest == 4 ? :t1 :
        nearest == 2 ? :t2 : :other
end

_sax_nonregularized_family_period_class(family::Symbol) =
    family in (:low_p1, :high_p2) ? :t1 :
    family == :high_p1 ? :t2 :
    family == :low_p2 ? :other : :unresolved

_sax_nonregularized_attracting_level(level::Symbol) = level in (
    :finite_amplitude_return_map,
    :strict_periodic_schur_and_return_map,
)

function _sax_nonregularized_register_attractor(
        point, register::Integer; validated::Bool=true)
    for family in _sax_nonregularized_register_families(register)
        attractor = _sax_nonregularized_attractor(
            point, family; validated=validated)
        isnothing(attractor) || return attractor
    end
    return nothing
end

function _sax_nonregularized_edge_decision(
        state, gamma, zeta, raw_model, settings;
        low_attractor=nothing, high_attractor=nothing)
    history = _sax_nonregularized_solve(
        state, gamma, zeta, raw_model, settings,
        settings.edge_decision_time;
        solver_kind=:tsit5, history=true,
        saveat=max(settings.saveat, 0.2))
    summary = _sax_nonregularized_block_summary(
        history.times, history.states, gamma, raw_model, settings)
    settled = summary.recurrence_valid && summary.stationary
    low_distance, high_distance = Inf, Inf
    family = settled ? summary.family : :unresolved
    if settled && !isnothing(low_attractor) && !isnothing(high_attractor)
        state_at_phase = summary.total_pressure_phase_state
        low_state = low_attractor.total_pressure_phase_state
        high_state = high_attractor.total_pressure_phase_state
        scales = max.(abs.(low_state), abs.(high_state), 1e-3)
        low_distance = norm((state_at_phase .- low_state) ./ scales) /
            sqrt(length(scales))
        high_distance = norm((state_at_phase .- high_state) ./ scales) /
            sqrt(length(scales))
        family = low_distance <= high_distance ?
            low_attractor.family : high_attractor.family
    end
    return (
        family=family,
        recurrence_family=summary.family,
        low_attractor_distance=float(low_distance),
        high_attractor_distance=float(high_distance),
        state=summary.total_pressure_phase_state,
        terminal_state=history.terminal_state,
        recurrence_error=float(summary.recurrence_error),
        period=float(summary.period),
        mixedness=float(4summary.modal_amplitudes[1] *
            summary.modal_amplitudes[2] /
            (summary.modal_amplitudes[1] +
             summary.modal_amplitudes[2] + eps(Float64))^2),
    )
end

function _compute_sax_nonregularized_edge_point(
        point, raw_model::NamedTuple,
        settings::SaxNonregularizedMultistabilitySettings)
    # The paper edge now brackets the two displayed minimal periods. These
    # exact families have periods T1 and T2 respectively. A high_p2 endpoint
    # has period near T1 and therefore cannot serve as the T2 endpoint.
    low = something(_sax_nonregularized_attractor(point, :low_p1))
    high = something(_sax_nonregularized_attractor(point, :high_p1))
    low_state = collect(float.(low.total_pressure_phase_state))
    high_state = collect(float.(high.total_pressure_phase_state))
    lower, upper = 0.0, 1.0
    decisions = Any[]
    edge_state = 0.5 .* (low_state .+ high_state)
    for iteration in 1:settings.edge_bisections
        lambda = (lower + upper) / 2
        initial = (1 - lambda) .* low_state .+ lambda .* high_state
        decision = try
            _sax_nonregularized_edge_decision(
                initial, point.gamma, point.zeta, raw_model, settings;
                low_attractor=low, high_attractor=high)
        catch err
            err isa InterruptException && rethrow()
            (
                family=:failed,
                state=collect(float.(initial)),
                terminal_state=collect(float.(initial)),
                recurrence_error=Inf,
                period=NaN,
                mixedness=NaN,
                exception_type=Symbol(nameof(typeof(err))),
                error=sprint(showerror, err),
            )
        end
        push!(decisions, merge(decision, (
            iteration=iteration, lambda=float(lambda))))
        edge_state = collect(float.(decision.state))
        if decision.family in (:low_p1, :low_p2)
            lower = lambda
        elseif decision.family in (:high_p1, :high_p2)
            upper = lambda
        else
            break
        end
    end
    long_history = _sax_nonregularized_solve(
        edge_state, point.gamma, point.zeta, raw_model, settings,
        settings.edge_maximum_time;
        solver_kind=:tsit5, history=true,
        saveat=settings.edge_saveat)
    fixed_state, fixed_residual = _estimate_fixed_point(
        point.gamma, point.zeta, raw_model; nmodes=settings.nmodes)
    amplitude1 = _mode_amplitude_series(
        long_history.states, 1;
        center=(fixed_state[3], fixed_state[4]))
    amplitude2 = _mode_amplitude_series(
        long_history.states, 2;
        center=(fixed_state[5], fixed_state[6]))
    metrics = _sax_mixed_escape_metrics(
        long_history.times, amplitude1, amplitude2;
        activation_threshold=settings.activation_threshold,
        mixedness_threshold=settings.mixedness_threshold,
        hold_time=settings.mixed_hold_cycles *
            2pi / float(raw_model.ω[1]))
    return (
        analysis=:nonregularized_multistability_edge_point,
        status=:complete,
        gamma=float(point.gamma), zeta=float(point.zeta),
        eta=0.0,
        bracket=(lower=float(lower), upper=float(upper)),
        decisions=decisions,
        metrics=metrics,
        escape_cycles=float(metrics.escape_time /
            (2pi / float(raw_model.ω[1]))),
        fixed_residual=float(fixed_residual),
        initial_condition=:phase_aligned_low_high_edge_bisection,
        endpoint_families=(t1=low.family, t2=high.family),
        endpoint_periods=(t1=float(low.period), t2=float(high.period)),
    )
end

function _sax_nonregularized_stride_indices(count::Integer, stride::Integer)
    values = collect(1:Int(stride):Int(count))
    isempty(values) || last(values) == count || push!(values, Int(count))
    return unique(values)
end

"""
Compute sparse mixed residence between compatible low-P1 T1 and high-P1 T2
attractors. High-P2-only T1 support is audited as uncovered by this legacy
low/high edge construction rather than being interpreted as non-mixed.
"""
function compute_sax_nonregularized_edge_escape(
        raw_model::NamedTuple, paths, guide_signature;
        settings::SaxNonregularizedMultistabilitySettings=
            sax_nonregularized_multistability_settings(:final),
        resume::Bool=true,
        parallel::Bool=true,
        verbosity::Integer=1)
    snapshot = _sax_nonregularized_snapshot(
        raw_model, paths, guide_signature, settings)
    gamma_indices = _sax_nonregularized_stride_indices(
        length(snapshot.grid.gamma), settings.edge_gamma_stride)
    zeta_indices = _sax_nonregularized_stride_indices(
        length(snapshot.grid.zeta), settings.edge_zeta_stride)
    targets = Any[]
    for row in zeta_indices, column in gamma_indices
        gamma, zeta = snapshot.grid.gamma[column], snapshot.grid.zeta[row]
        point = snapshot.points[(float(gamma), float(zeta))]
        isnothing(_sax_nonregularized_attractor(point, :low_p1)) && continue
        isnothing(_sax_nonregularized_attractor(point, :high_p1)) && continue
        push!(targets, point)
    end
    started = time()
    computed = Threads.Atomic{Int}(0)
    function evaluate(index)
        point = targets[index]
        path = paths.edge(point.gamma, point.zeta)
        loaded = resume ? _load_sax_nonregularized_cache(
            path, :nonregularized_edge,
            raw_model, settings, guide_signature) :
            (status=:missing, payload=nothing, reason="resume disabled")
        if loaded.status != :valid
            result = try
                _compute_sax_nonregularized_edge_point(
                    point, raw_model, settings)
            catch err
                err isa InterruptException && rethrow()
                (
                    analysis=:nonregularized_multistability_edge_point,
                    status=:failed,
                    gamma=float(point.gamma), zeta=float(point.zeta),
                    eta=0.0,
                    exception_type=Symbol(nameof(typeof(err))),
                    error=sprint(showerror, err),
                )
            end
            _save_sax_nonregularized_cache(
                path, :nonregularized_edge, result,
                raw_model, settings, guide_signature)
        end
        Threads.atomic_add!(computed, 1)
        return nothing
    end
    _sax_nonregularized_parallel_map!(
        evaluate, length(targets); parallel=parallel,
        verbosity=verbosity, stage="exact coexistence edge tracking")
    return merge(
        load_sax_nonregularized_multistability_progress(
            raw_model, paths, guide_signature; settings=settings),
        (elapsed_seconds=round(time() - started; digits=1),
         edge_targets=length(targets), processed=Int(computed[])))
end

# ---------------------------------------------------------------------------
# Progress assembly, compact cache, and read-only loader
# ---------------------------------------------------------------------------

function _sax_nonregularized_point_matrices(
        raw_model::NamedTuple, paths, guide_signature,
        settings::SaxNonregularizedMultistabilitySettings)
    grid = _sax_nonregularized_grid(settings)
    dimensions = (length(grid.zeta), length(grid.gamma))
    cache_present = falses(dimensions)
    attempted_low = falses(dimensions)
    attempted_high = falses(dimensions)
    candidate_low = falses(dimensions)
    candidate_high = falses(dimensions)
    candidate_p2 = falses(dimensions)
    stable_low = falses(dimensions)
    stable_high = falses(dimensions)
    stable_p2 = falses(dimensions)
    stable_mode1 = falses(dimensions)
    stable_mode2 = falses(dimensions)
    stable_modal_mixed = falses(dimensions)
    stable_register_low = falses(dimensions)
    stable_register_high = falses(dimensions)
    stable_period_t1 = falses(dimensions)
    stable_period_t2 = falses(dimensions)
    stable_period_other = falses(dimensions)
    stable_period_t1_high_p2 = falses(dimensions)
    stable_period_edge_eligible = falses(dimensions)
    stable_p2_mode1 = falses(dimensions)
    stable_p2_mode2 = falses(dimensions)
    stable_p2_modal_mixed = falses(dimensions)
    strict_pqz_low = falses(dimensions)
    strict_pqz_high = falses(dimensions)
    pqz_audit_scheduled = falses(dimensions)
    pqz_audit_validated = falses(dimensions)
    pqz_audit_ineligible = falses(dimensions)
    pqz_audit_failed = falses(dimensions)
    validation_complete = falses(dimensions)
    validation_pending = falses(dimensions)
    unresolved_low = falses(dimensions)
    unresolved_high = falses(dimensions)
    point_status = Any[]
    compact_points = Any[]
    for (row, zeta) in enumerate(grid.zeta),
            (column, gamma) in enumerate(grid.gamma)
        loaded = _load_sax_nonregularized_cache(
            paths.point(gamma, zeta), :nonregularized_point,
            raw_model, settings, guide_signature)
        push!(point_status, (
            gamma=float(gamma), zeta=float(zeta),
            status=loaded.status, reason=loaded.reason,
            path=paths.point(gamma, zeta)))
        loaded.status == :valid || continue
        cache_present[row, column] = true
        point = loaded.payload
        attempted_low[row, column] = any(
            attempt -> Int(attempt.requested_register) == 1,
            point.attempts)
        attempted_high[row, column] = any(
            attempt -> Int(attempt.requested_register) == 2,
            point.attempts)
        low = _sax_nonregularized_register_attractor(
            point, 1; validated=false)
        high = _sax_nonregularized_register_attractor(
            point, 2; validated=false)
        p2 = [attractor for attractor in point.attractors
              if attractor.family in (:low_p2, :high_p2)]
        candidate_low[row, column] = !isnothing(low)
        candidate_high[row, column] = !isnothing(high)
        candidate_p2[row, column] = !isempty(p2)
        stable_low[row, column] = !isnothing(
            _sax_nonregularized_register_attractor(point, 1))
        stable_high[row, column] = !isnothing(
            _sax_nonregularized_register_attractor(point, 2))
        stable_p2[row, column] = any(attractor ->
            hasproperty(attractor, :validation) &&
            hasproperty(attractor.validation, :attracting) &&
            Bool(attractor.validation.attracting), p2)
        stable_attractors = [attractor for attractor in point.attractors
            if hasproperty(attractor, :validation) &&
               hasproperty(attractor.validation, :attracting) &&
               Bool(attractor.validation.attracting)]
        stable_mode1[row, column] = any(attractor ->
            attractor.register_contrast >=
                SAX_NONREGULARIZED_MODAL_DOMINANCE_MARGIN,
            stable_attractors)
        stable_mode2[row, column] = any(attractor ->
            attractor.register_contrast <=
                -SAX_NONREGULARIZED_MODAL_DOMINANCE_MARGIN,
            stable_attractors)
        stable_modal_mixed[row, column] = any(attractor ->
            abs(attractor.register_contrast) <
                SAX_NONREGULARIZED_MODAL_DOMINANCE_MARGIN,
            stable_attractors)
        # A balanced first/second-mode spectrum belongs to the low register:
        # mode 2 is then the octave harmonic of a response whose global period
        # is represented by mode 1. Only clear mode-2 dominance is classified
        # as high register. This prevents a low orbit with a strong octave
        # harmonic from becoming an artificial gray hole.
        stable_register_low[row, column] = any(attractor ->
            attractor.register_contrast >
                -SAX_NONREGULARIZED_MODAL_DOMINANCE_MARGIN,
            stable_attractors)
        stable_register_high[row, column] = stable_mode2[row, column]
        period_classes = [
            _sax_nonregularized_period_class(attractor.period, raw_model)
            for attractor in stable_attractors]
        stable_period_t1[row, column] = :t1 in period_classes
        stable_period_t2[row, column] = :t2 in period_classes
        stable_period_other[row, column] = :other in period_classes
        stable_period_t1_high_p2[row, column] = any(attractor ->
            attractor.family == :high_p2 &&
            _sax_nonregularized_period_class(
                attractor.period, raw_model) == :t1,
            stable_attractors)
        stable_period_edge_eligible[row, column] =
            any(attractor -> attractor.family == :low_p1,
                stable_attractors) &&
            any(attractor -> attractor.family == :high_p1,
                stable_attractors)
        stable_p2_attractors = [attractor for attractor in stable_attractors
            if hasproperty(attractor, :period_generation) &&
               Int(attractor.period_generation) == 2]
        stable_p2_mode1[row, column] = any(attractor ->
            attractor.register_contrast >=
                SAX_NONREGULARIZED_MODAL_DOMINANCE_MARGIN,
            stable_p2_attractors)
        stable_p2_mode2[row, column] = any(attractor ->
            attractor.register_contrast <=
                -SAX_NONREGULARIZED_MODAL_DOMINANCE_MARGIN,
            stable_p2_attractors)
        stable_p2_modal_mixed[row, column] = any(attractor ->
            abs(attractor.register_contrast) <
                SAX_NONREGULARIZED_MODAL_DOMINANCE_MARGIN,
            stable_p2_attractors)
        strict_pqz_low[row, column] = any(attractor ->
            attractor.family in _sax_nonregularized_register_families(1) &&
            hasproperty(attractor, :validation) &&
            attractor.validation.status == :complete &&
            attractor.validation.level ==
                :strict_periodic_schur_and_return_map,
            point.attractors)
        strict_pqz_high[row, column] = any(attractor ->
            attractor.family in _sax_nonregularized_register_families(2) &&
            hasproperty(attractor, :validation) &&
            attractor.validation.status == :complete &&
            attractor.validation.level ==
                :strict_periodic_schur_and_return_map,
            point.attractors)
        pqz_statuses = [attractor.validation.periodic_schur.status
            for attractor in point.attractors
            if hasproperty(attractor, :validation) &&
               attractor.validation.status == :complete &&
               hasproperty(attractor.validation, :periodic_schur)]
        pqz_audit_scheduled[row, column] = any(status -> status in (
            :validated, :ineligible_nonsmooth_crossing, :failed), pqz_statuses)
        pqz_audit_validated[row, column] = :validated in pqz_statuses
        pqz_audit_ineligible[row, column] =
            :ineligible_nonsmooth_crossing in pqz_statuses
        pqz_audit_failed[row, column] = :failed in pqz_statuses
        validations = [attractor.validation for attractor in point.attractors
            if hasproperty(attractor, :validation)]
        validation_complete[row, column] =
            !isempty(point.attractors) &&
            all(validation -> validation.status == :complete, validations) &&
            length(validations) == length(point.attractors)
        validation_pending[row, column] = any(attractor ->
            !hasproperty(attractor, :validation) ||
            attractor.validation.status != :complete,
            point.attractors)
        unresolved_low[row, column] =
            attempted_low[row, column] &&
            !stable_register_low[row, column]
        unresolved_high[row, column] =
            attempted_high[row, column] &&
            !stable_register_high[row, column]
        push!(compact_points, (
            gamma=float(gamma), zeta=float(zeta),
            attempts=length(point.attempts),
            families=[attractor.family for attractor in point.attractors],
            validation_levels=[
                hasproperty(attractor, :validation) ?
                    attractor.validation.level : :unvalidated_candidate
                for attractor in point.attractors],
            attracting=[
                hasproperty(attractor, :validation) &&
                    hasproperty(attractor.validation, :attracting) &&
                    Bool(attractor.validation.attracting)
                for attractor in point.attractors],
            periods=[hasproperty(attractor, :period) ?
                float(attractor.period) : NaN
                for attractor in point.attractors],
            period_generations=[
                hasproperty(attractor, :period_generation) ?
                    Int(attractor.period_generation) : 0
                for attractor in point.attractors],
            period_classes=[hasproperty(attractor, :period) ?
                _sax_nonregularized_period_class(
                    attractor.period, raw_model) : :unresolved
                for attractor in point.attractors],
            modal_classes=[
                attractor.register_contrast >=
                    SAX_NONREGULARIZED_MODAL_DOMINANCE_MARGIN ? :mode1 :
                attractor.register_contrast <=
                    -SAX_NONREGULARIZED_MODAL_DOMINANCE_MARGIN ? :mode2 :
                :balanced
                for attractor in stable_attractors],
        ))
    end
    return (
        gamma=grid.gamma, zeta=grid.zeta,
        cache_present=cache_present,
        attempted_low=attempted_low,
        attempted_high=attempted_high,
        candidate_low=candidate_low,
        candidate_high=candidate_high,
        candidate_p2=candidate_p2,
        stable_low=stable_low,
        stable_high=stable_high,
        stable_p2=stable_p2,
        stable_mode1=stable_mode1,
        stable_mode2=stable_mode2,
        stable_modal_mixed=stable_modal_mixed,
        stable_modal_bistable=stable_mode1 .& stable_mode2,
        stable_register_low=stable_register_low,
        stable_register_high=stable_register_high,
        stable_register_bistable=
            stable_register_low .& stable_register_high,
        stable_period_t1=stable_period_t1,
        stable_period_t2=stable_period_t2,
        stable_period_t1_t2=stable_period_t1 .& stable_period_t2,
        stable_period_other=stable_period_other,
        stable_period_t1_high_p2=stable_period_t1_high_p2,
        stable_period_edge_eligible=stable_period_edge_eligible,
        stable_p2_mode1=stable_p2_mode1,
        stable_p2_mode2=stable_p2_mode2,
        stable_p2_modal_mixed=stable_p2_modal_mixed,
        bistable=stable_low .& stable_high,
        strict_pqz_low=strict_pqz_low,
        strict_pqz_high=strict_pqz_high,
        pqz_audit_scheduled=pqz_audit_scheduled,
        pqz_audit_validated=pqz_audit_validated,
        pqz_audit_ineligible=pqz_audit_ineligible,
        pqz_audit_failed=pqz_audit_failed,
        validation_complete=validation_complete,
        validation_pending=validation_pending,
        unresolved_low=unresolved_low,
        unresolved_high=unresolved_high,
        point_status=point_status,
        compact_points=compact_points,
    )
end

function _sax_nonregularized_edge_progress(
        raw_model, paths, guide_signature, settings, sheets)
    points = Any[]
    statuses = Any[]
    for (row, zeta) in enumerate(sheets.zeta),
            (column, gamma) in enumerate(sheets.gamma)
        eligible = hasproperty(sheets, :stable_period_edge_eligible) ?
            sheets.stable_period_edge_eligible[row, column] :
            sheets.bistable[row, column]
        eligible || continue
        column in _sax_nonregularized_stride_indices(
            length(sheets.gamma), settings.edge_gamma_stride) || continue
        row in _sax_nonregularized_stride_indices(
            length(sheets.zeta), settings.edge_zeta_stride) || continue
        path = paths.edge(gamma, zeta)
        loaded = _load_sax_nonregularized_cache(
            path, :nonregularized_edge,
            raw_model, settings, guide_signature)
        push!(statuses, (
            gamma=float(gamma), zeta=float(zeta),
            status=loaded.status, reason=loaded.reason, path=path))
        loaded.status == :valid && push!(points, loaded.payload)
    end
    expected = length(statuses)
    complete = count(point -> point.status == :complete, points)
    failed = count(point -> point.status == :failed, points)
    return (
        status=expected == 0 ? :no_bistability :
            length(points) == expected ? :complete :
            isempty(points) ? :missing : :partial,
        points=points,
        statuses=statuses,
        counts=(expected=expected, cached=length(points),
                complete=complete, failed=failed,
                right_censored=count(point ->
                    point.status == :complete &&
                    point.metrics.right_censored, points)),
    )
end

"""Read every atomic exact-model cache without starting a computation."""
function load_sax_nonregularized_multistability_progress(
        raw_model::NamedTuple, paths, guide_signature;
        settings::SaxNonregularizedMultistabilitySettings=
            sax_nonregularized_multistability_settings(:final))
    matrices = _sax_nonregularized_point_matrices(
        raw_model, paths, guide_signature, settings)
    total = length(matrices.cache_present)
    cached = count(matrices.cache_present)
    sheets = (
        analysis=:nonregularized_attractor_sheets,
        status=cached == total ? :complete : cached == 0 ? :missing : :partial,
        gamma=matrices.gamma,
        zeta=matrices.zeta,
        cache_present=matrices.cache_present,
        attempted_low=matrices.attempted_low,
        attempted_high=matrices.attempted_high,
        candidate_low=matrices.candidate_low,
        candidate_high=matrices.candidate_high,
        candidate_p2=matrices.candidate_p2,
        stable_low=matrices.stable_low,
        stable_high=matrices.stable_high,
        stable_p2=matrices.stable_p2,
        stable_mode1=matrices.stable_mode1,
        stable_mode2=matrices.stable_mode2,
        stable_modal_mixed=matrices.stable_modal_mixed,
        stable_modal_bistable=matrices.stable_modal_bistable,
        stable_register_low=matrices.stable_register_low,
        stable_register_high=matrices.stable_register_high,
        stable_register_bistable=matrices.stable_register_bistable,
        stable_period_t1=matrices.stable_period_t1,
        stable_period_t2=matrices.stable_period_t2,
        stable_period_t1_t2=matrices.stable_period_t1_t2,
        stable_period_other=matrices.stable_period_other,
        stable_period_t1_high_p2=matrices.stable_period_t1_high_p2,
        stable_period_edge_eligible=matrices.stable_period_edge_eligible,
        stable_p2_mode1=matrices.stable_p2_mode1,
        stable_p2_mode2=matrices.stable_p2_mode2,
        stable_p2_modal_mixed=matrices.stable_p2_modal_mixed,
        modal_dominance_margin=
            SAX_NONREGULARIZED_MODAL_DOMINANCE_MARGIN,
        stable_register1=matrices.stable_register_low,
        stable_register2=matrices.stable_register_high,
        bistable=matrices.bistable,
        strict_pqz_low=matrices.strict_pqz_low,
        strict_pqz_high=matrices.strict_pqz_high,
        pqz_audit_scheduled=matrices.pqz_audit_scheduled,
        pqz_audit_validated=matrices.pqz_audit_validated,
        pqz_audit_ineligible=matrices.pqz_audit_ineligible,
        pqz_audit_failed=matrices.pqz_audit_failed,
        validation_complete=matrices.validation_complete,
        validation_pending=matrices.validation_pending,
        unresolved_low=matrices.unresolved_low,
        unresolved_high=matrices.unresolved_high,
        unknown_low=.!matrices.stable_register_low,
        unknown_high=.!matrices.stable_register_high,
        compact_points=matrices.compact_points,
        point_status=matrices.point_status,
        counts=(
            cached_points=cached,
            expected_points=total,
            candidate_low=count(matrices.candidate_low),
            candidate_high=count(matrices.candidate_high),
            candidate_p2=count(matrices.candidate_p2),
            stable_low=count(matrices.stable_low),
            stable_high=count(matrices.stable_high),
            stable_p2=count(matrices.stable_p2),
            stable_mode1=count(matrices.stable_mode1),
            stable_mode2=count(matrices.stable_mode2),
            stable_modal_mixed=count(matrices.stable_modal_mixed),
            stable_modal_bistable=count(matrices.stable_modal_bistable),
            stable_register_low=count(matrices.stable_register_low),
            stable_register_high=count(matrices.stable_register_high),
            stable_register_bistable=
                count(matrices.stable_register_bistable),
            stable_period_t1=count(matrices.stable_period_t1),
            stable_period_t2=count(matrices.stable_period_t2),
            stable_period_t1_t2=count(matrices.stable_period_t1_t2),
            stable_period_other=count(matrices.stable_period_other),
            stable_period_t1_high_p2=
                count(matrices.stable_period_t1_high_p2),
            stable_period_edge_eligible=
                count(matrices.stable_period_edge_eligible),
            stable_p2_mode1=count(matrices.stable_p2_mode1),
            stable_p2_mode2=count(matrices.stable_p2_mode2),
            bistable=count(matrices.bistable),
            strict_pqz_low=count(matrices.strict_pqz_low),
            strict_pqz_high=count(matrices.strict_pqz_high),
            pqz_audit_scheduled=count(matrices.pqz_audit_scheduled),
            pqz_audit_validated=count(matrices.pqz_audit_validated),
            pqz_audit_ineligible=count(matrices.pqz_audit_ineligible),
            pqz_audit_failed=count(matrices.pqz_audit_failed),
            unresolved_low=count(matrices.unresolved_low),
            unresolved_high=count(matrices.unresolved_high),
        ),
    )
    edge = _sax_nonregularized_edge_progress(
        raw_model, paths, guide_signature, settings, sheets)
    validations_pending = count(matrices.validation_pending)
    return (
        analysis=:nonregularized_multistability_progress,
        status=sheets.status == :complete && validations_pending == 0 &&
            edge.status in (:complete, :no_bistability) ? :complete :
            sheets.status == :missing ? :missing : :partial,
        eta=0.0,
        exact_model=:historical_piecewise_saxRN,
        paths=paths,
        settings=_portable_sax_nonregularized_multistability_settings(settings),
        guide=guide_signature,
        sheets=sheets,
        edge=edge,
        counts=merge(sheets.counts, (
            validations_pending=validations_pending,
            edge_points=edge.counts.cached,
            expected_edge_points=edge.counts.expected,
        )),
    )
end

"""Commit a compact exact-model product for the read-only Pluto notebook."""
function assemble_sax_nonregularized_multistability_atlas(
        raw_model::NamedTuple, paths, guide_signature;
        settings::SaxNonregularizedMultistabilitySettings=
            sax_nonregularized_multistability_settings(:final),
        source=NamedTuple())
    progress = load_sax_nonregularized_multistability_progress(
        raw_model, paths, guide_signature; settings=settings)
    compact_sheets = merge(progress.sheets, (point_status=Any[],))
    product = (
        schema_version=SAX_NONREGULARIZED_MULTISTABILITY_SCHEMA_VERSION,
        analysis=:nonregularized_multistability_atlas,
        status=progress.status,
        saved_at_unix=time(),
        eta=0.0,
        exact_model=:historical_piecewise_saxRN,
        settings=progress.settings,
        guide=guide_signature,
        sheets=compact_sheets,
        edge=progress.edge,
        counts=progress.counts,
        definitions=(
            stable_low="legacy eta=0 low-family label passed solver agreement and finite-amplitude Poincare-return attraction tests; not the paper modal-register mask",
            stable_high="legacy eta=0 high-family or branch-labelled P2 label passed solver agreement and finite-amplitude Poincare-return attraction tests; not the paper modal-register mask",
            stable_p2="eta=0 period-two response passed the same local-attraction tests",
            bistability="stable_low AND stable_high at the same sampled parameter point",
            strict_periodic_schur="additional validation only away from flow reversal and reed-contact grazing; transverse reed-contact crossings are permitted",
            unresolved="no validated response from the attempted routes; not synonymous with unstable",
            edge="exact sampled basin-edge trajectory; never a filled parameter-space region",
            regularized_guide="initial-state generator only; not evidence in this product",
            stable_mode1="validated eta=0 attracting response with mode 1 exceeding mode 2 by the declared modal-contrast margin",
            stable_mode2="validated eta=0 attracting response with mode 2 exceeding mode 1 by the declared modal-contrast margin",
            stable_modal_bistable="distinct mode-1-dominant and mode-2-dominant attracting responses at the same sampled point",
            stable_register_low="validated attracting response that is mode-1 dominant or modally balanced; mode 2 is interpreted as the octave harmonic in the balanced case",
            stable_register_high="validated attracting response with clear mode-2 dominance beyond the declared modal-contrast margin",
            stable_register_bistable="distinct low-register and clearly mode-2-dominant high-register attracting responses at the same sampled point",
            stable_period_t1="validated attracting response whose minimal full-state recurrence is compatible with T1; this also includes high-family P2 responses when 2T2 is compatible with T1",
            stable_period_t2="validated attracting response whose minimal full-state recurrence is compatible with T2",
            stable_period_t1_t2="distinct validated T1- and T2-periodic attracting responses accessible at the same sampled point; not one orbit containing both periods",
            stable_period_other="validated attracting response nearest 2T1 rather than either principal period; retained as an audit and excluded from the three paper hatches",
            stable_period_edge_eligible="validated low_p1 and high_p1 cycles coexist, so the saved basin-edge experiment brackets T1 and T2 rather than a P2 endpoint",
            p2="temporal period-two label; deliberately independent of modal register",
        ),
        source=source,
    )
    _save_sax_nonregularized_cache(
        paths.product, :nonregularized_product,
        product, raw_model, settings, guide_signature)
    return product
end

function load_sax_nonregularized_multistability_atlas(
        raw_model::NamedTuple, paths, guide_signature;
        settings::SaxNonregularizedMultistabilitySettings=
            sax_nonregularized_multistability_settings(:final),
        allow_progress::Bool=true)
    loaded = _load_sax_nonregularized_cache(
        paths.product, :nonregularized_product,
        raw_model, settings, guide_signature)
    if !allow_progress
        return loaded.status == :valid ? (
            status=:valid, source=:product, atlas=loaded.payload,
            paths=paths, reason=loaded.reason) : (
            status=loaded.status, source=:none, atlas=nothing,
            paths=paths, reason=loaded.reason)
    end
    progress = load_sax_nonregularized_multistability_progress(
        raw_model, paths, guide_signature; settings=settings)
    if loaded.status != :valid
        return (
            status=progress.status == :missing ? :missing : :valid,
            source=:progress, atlas=progress,
            paths=paths, reason=loaded.reason)
    end
    product = loaded.payload
    product_pending = hasproperty(product.counts, :validations_pending) ?
        product.counts.validations_pending : typemax(Int)
    product_pqz = hasproperty(product.counts, :pqz_audit_scheduled) ?
        product.counts.pqz_audit_scheduled : 0
    newer = progress.counts.cached_points > product.counts.cached_points ||
        progress.counts.stable_low > product.counts.stable_low ||
        progress.counts.stable_high > product.counts.stable_high ||
        progress.counts.stable_p2 > product.counts.stable_p2 ||
        progress.counts.validations_pending < product_pending ||
        progress.counts.pqz_audit_scheduled > product_pqz ||
        progress.counts.edge_points > product.counts.edge_points
    return newer ? (
        status=:valid, source=:progress, atlas=progress,
        paths=paths, reason="atomic point caches are newer than the product") : (
        status=:valid, source=:product, atlas=product,
        paths=paths, reason=loaded.reason)
end

# ---------------------------------------------------------------------------
# Paper-oriented visualizations
# ---------------------------------------------------------------------------

function _sax_nonregularized_compact_attracting_indices(point)
    families = hasproperty(point, :families) ? point.families : Symbol[]
    if hasproperty(point, :attracting) &&
            length(point.attracting) == length(families)
        return Int[index for index in eachindex(families)
                   if Bool(point.attracting[index])]
    end
    levels = hasproperty(point, :validation_levels) ?
        point.validation_levels : fill(:unvalidated_candidate, length(families))
    return Int[index for index in eachindex(families)
               if index <= length(levels) &&
                  _sax_nonregularized_attracting_level(
                      Symbol(levels[index]))]
end

function _sax_nonregularized_compact_attracting_families(point)
    families = hasproperty(point, :families) ? point.families : Symbol[]
    return Symbol[families[index]
                  for index in _sax_nonregularized_compact_attracting_indices(
                      point)]
end

"""
Recover the two principal minimal-period existence masks from a map product.

New products contain these sheets directly. Older products are reconstructed
without reintegration from the stored period-derived family and validation
records. `T1+T2` means that at least one distinct attracting response from
each period class exists at the same sampled parameter point. It never means
that a single trajectory simultaneously has two minimal periods.
"""
function sax_nonregularized_period_masks(atlas, raw_model::NamedTuple)
    sheets = atlas.sheets
    if hasproperty(sheets, :stable_period_t1) &&
            hasproperty(sheets, :stable_period_t2)
        t1 = BitMatrix(sheets.stable_period_t1)
        t2 = BitMatrix(sheets.stable_period_t2)
        other = hasproperty(sheets, :stable_period_other) ?
            BitMatrix(sheets.stable_period_other) : falses(size(t1))
        high_p2 = hasproperty(sheets, :stable_period_t1_high_p2) ?
            BitMatrix(sheets.stable_period_t1_high_p2) : falses(size(t1))
        return (
            t1=t1, t2=t2, coexistence=t1 .& t2,
            other=other, t1_high_p2=high_p2,
            source=:period_sheets,
        )
    end

    dimensions = (length(sheets.zeta), length(sheets.gamma))
    t1 = falses(dimensions)
    t2 = falses(dimensions)
    other = falses(dimensions)
    high_p2 = falses(dimensions)
    compact = hasproperty(sheets, :compact_points) ?
        sheets.compact_points : Any[]
    for point in compact
        row = argmin(abs.(sheets.zeta .- float(point.zeta)))
        column = argmin(abs.(sheets.gamma .- float(point.gamma)))
        indices = _sax_nonregularized_compact_attracting_indices(point)
        families = hasproperty(point, :families) ?
            Symbol[point.families[index] for index in indices] : Symbol[]
        classes = if hasproperty(point, :families) &&
                hasproperty(point, :periods) &&
                length(point.periods) == length(point.families)
            [_sax_nonregularized_period_class(
                point.periods[index], raw_model) for index in indices]
        else
            _sax_nonregularized_family_period_class.(families)
        end
        t1[row, column] = :t1 in classes
        t2[row, column] = :t2 in classes
        other[row, column] = :other in classes
        high_p2[row, column] = :high_p2 in families
    end
    return (
        t1=t1, t2=t2, coexistence=t1 .& t2,
        other=other, t1_high_p2=high_p2,
        source=:compact_period_families,
    )
end

function _sax_nonregularized_period_compact_lookup(atlas)
    sheets = atlas.sheets
    compact = hasproperty(sheets, :compact_points) ?
        sheets.compact_points : Any[]
    return Dict(
        (argmin(abs.(sheets.zeta .- float(point.zeta))),
         argmin(abs.(sheets.gamma .- float(point.gamma)))) => point
        for point in compact)
end

function _sax_nonregularized_selected_edge_families(point)
    families = _sax_nonregularized_compact_attracting_families(point)
    first_present(preferences) = begin
        index = findfirst(family -> family in families, preferences)
        isnothing(index) ? :missing : preferences[index]
    end
    return (
        low=first_present((:low_p1, :low_p2)),
        high=first_present((:high_p1, :high_p2)),
    )
end

"""
Audit whether the saved basin-edge experiments cover the redefined T1+T2 set.

The historical edge bisection selected one low-family and one high-family
endpoint. Such a cache is reusable for the period map only when those selected
endpoints were `low_p1` (period T1) and `high_p1` (period T2). Results involving
`low_p2` do not bracket the two displayed period classes. Points whose T1
response exists only as `high_p2` were outside the historical low/high edge
plan and are reported as uncovered rather than treated as negative evidence.
"""
function sax_nonregularized_period_edge_coverage(
        atlas, raw_model::NamedTuple;
        edge_gamma_stride::Union{Nothing,Integer}=nothing,
        edge_zeta_stride::Union{Nothing,Integer}=nothing)
    sheets = atlas.sheets
    masks = sax_nonregularized_period_masks(atlas, raw_model)
    gamma_stride = isnothing(edge_gamma_stride) ?
        Int(atlas.settings.edge_gamma_stride) : Int(edge_gamma_stride)
    zeta_stride = isnothing(edge_zeta_stride) ?
        Int(atlas.settings.edge_zeta_stride) : Int(edge_zeta_stride)
    gamma_indices = _sax_nonregularized_stride_indices(
        length(sheets.gamma), gamma_stride)
    zeta_indices = _sax_nonregularized_stride_indices(
        length(sheets.zeta), zeta_stride)
    planned = [(row, column) for row in zeta_indices,
               column in gamma_indices if masks.coexistence[row, column]]
    compact_lookup = _sax_nonregularized_period_compact_lookup(atlas)
    edge_lookup = Dict(
        (argmin(abs.(sheets.zeta .- float(point.zeta))),
         argmin(abs.(sheets.gamma .- float(point.gamma)))) => point
        for point in atlas.edge.points)
    compatible = Tuple{Int,Int}[]
    incompatible = Tuple{Int,Int}[]
    missing_endpoint_record = Tuple{Int,Int}[]
    completed_points = Any[]
    failed_points = Any[]
    missing_cache = Tuple{Int,Int}[]
    incompatible_cached = 0
    for index in planned
        compact_point = get(compact_lookup, index, nothing)
        if isnothing(compact_point)
            push!(missing_endpoint_record, index)
            continue
        end
        endpoints = _sax_nonregularized_selected_edge_families(compact_point)
        if endpoints != (low=:low_p1, high=:high_p1)
            push!(incompatible, index)
            haskey(edge_lookup, index) && (incompatible_cached += 1)
            continue
        end
        push!(compatible, index)
        edge_point = get(edge_lookup, index, nothing)
        if isnothing(edge_point)
            push!(missing_cache, index)
        elseif edge_point.status == :complete
            push!(completed_points, edge_point)
        else
            push!(failed_points, edge_point)
        end
    end
    return (
        status=isempty(planned) ? :no_t1_t2_coexistence :
            length(completed_points) == length(planned) ? :complete : :partial,
        planned_points=length(planned),
        endpoint_compatible_points=length(compatible),
        endpoint_incompatible_points=length(incompatible),
        legacy_plan_uncovered_points=length(incompatible),
        endpoint_incompatible_cached_points=incompatible_cached,
        missing_endpoint_records=length(missing_endpoint_record),
        completed_points=length(completed_points),
        failed_points=length(failed_points),
        missing_cache_points=length(missing_cache),
        full_overlap_coverage_fraction=isempty(planned) ? NaN :
            length(completed_points) / length(planned),
        right_censored_points=count(
            point -> point.metrics.right_censored, completed_points),
        eligible_points=completed_points,
        incompatible_indices=incompatible,
        missing_cache_indices=missing_cache,
        period_mask_source=masks.source,
    )
end

function _sax_nonregularized_low_p1_high_p2_mask(atlas)
    sheets = atlas.sheets
    dimensions = (length(sheets.zeta), length(sheets.gamma))
    mask = falses(dimensions)
    coexistence = hasproperty(sheets, :stable_register_bistable) ?
        sheets.stable_register_bistable :
        _sax_nonregularized_modal_masks(sheets).coexistence
    for (index, point) in _sax_nonregularized_period_compact_lookup(atlas)
        families = _sax_nonregularized_compact_attracting_families(point)
        mask[index...] = :low_p1 in families && :high_p2 in families &&
            coexistence[index...]
    end
    return mask
end

"""
Select the two defensible sources for the gray Mixed diagnostic.

The first source brackets low-P1 T1 and high-P1 T2 attractors inside the
period overlap. The second restores the physically important central
transition between low-P1 and high-P2 attractors. Those two attractors have
approximately the same global period T1 but different family provenance and
modal organization. Requiring the original validated low/high coexistence mask
prevents high-P2 edge caches across the rest of the plane from flooding the
paper figure.
"""
function sax_nonregularized_mixed_edge_evidence(
        atlas, raw_model::NamedTuple)
    sheets = atlas.sheets
    period_masks = sax_nonregularized_period_masks(atlas, raw_model)
    coexistence = hasproperty(sheets, :stable_register_bistable) ?
        sheets.stable_register_bistable :
        _sax_nonregularized_modal_masks(sheets).coexistence
    compact_lookup = _sax_nonregularized_period_compact_lookup(atlas)
    t1_t2 = Any[]
    t1_p2 = Any[]
    for point in atlas.edge.points
        point.status == :complete || continue
        index = (
            argmin(abs.(sheets.zeta .- float(point.zeta))),
            argmin(abs.(sheets.gamma .- float(point.gamma))),
        )
        compact_point = get(compact_lookup, index, nothing)
        isnothing(compact_point) && continue
        endpoints = _sax_nonregularized_selected_edge_families(compact_point)
        if period_masks.coexistence[index...] &&
                endpoints == (low=:low_p1, high=:high_p1)
            push!(t1_t2, point)
        elseif coexistence[index...] &&
                endpoints == (low=:low_p1, high=:high_p2)
            push!(t1_p2, point)
        end
    end
    points = vcat(t1_t2, t1_p2)
    return (
        points=points,
        t1_t2_points=t1_t2,
        t1_p2_points=t1_p2,
        counts=(
            total=length(points),
            right_censored=count(
                point -> point.metrics.right_censored, points),
            t1_t2=length(t1_t2),
            t1_t2_right_censored=count(
                point -> point.metrics.right_censored, t1_t2),
            t1_p2=length(t1_p2),
            t1_p2_right_censored=count(
                point -> point.metrics.right_censored, t1_p2),
        ),
    )
end

function _sax_nonregularized_modal_masks(sheets)
    if hasproperty(sheets, :stable_register_low) &&
            hasproperty(sheets, :stable_register_high)
        return (
            low=sheets.stable_register_low,
            high=sheets.stable_register_high,
            balanced=sheets.stable_modal_mixed,
            coexistence=sheets.stable_register_bistable,
        )
    end
    if hasproperty(sheets, :stable_mode1) &&
            hasproperty(sheets, :stable_mode2)
        return (
            low=sheets.stable_mode1,
            high=sheets.stable_mode2,
            balanced=sheets.stable_modal_mixed,
            coexistence=sheets.stable_modal_bistable,
        )
    end
    return (
        low=sheets.stable_low,
        high=sheets.stable_high,
        balanced=falses(Base.size(sheets.stable_low)),
        coexistence=sheets.stable_low .& sheets.stable_high,
    )
end

function _sax_nonregularized_categorical_values(sheets)
    masks = _sax_nonregularized_modal_masks(sheets)
    values = zeros(Int8, Base.size(masks.low))
    values[masks.low .& .!masks.high] .= 1
    values[masks.high .& .!masks.low] .= 2
    values[masks.coexistence] .= 3
    values[masks.balanced .& .!masks.low .& .!masks.high] .= 4
    return values
end

function _sax_nonregularized_binary_noise_counts(mask::AbstractMatrix{Bool})
    isolated_positive = 0
    surrounded_hole = 0
    for row in axes(mask, 1), column in axes(mask, 2)
        row_first, row_last = max(first(axes(mask, 1)), row - 1),
            min(last(axes(mask, 1)), row + 1)
        column_first, column_last =
            max(first(axes(mask, 2)), column - 1),
            min(last(axes(mask, 2)), column + 1)
        neighbourhood = @view mask[
            row_first:row_last, column_first:column_last]
        positives = count(neighbourhood)
        mask[row, column] && positives <= 2 &&
            (isolated_positive += 1)
        interior = row_first < row < row_last &&
            column_first < column < column_last
        !mask[row, column] && interior && positives == 8 &&
            (surrounded_hole += 1)
    end
    return (
        isolated_positive=isolated_positive,
        surrounded_hole=surrounded_hole)
end

"""Count raw one-cell islands and fully surrounded holes in each modal mask."""
function sax_nonregularized_raster_noise_audit(atlas)
    masks = _sax_nonregularized_modal_masks(atlas.sheets)
    return (
        low=_sax_nonregularized_binary_noise_counts(masks.low),
        high=_sax_nonregularized_binary_noise_counts(masks.high),
        coexistence=_sax_nonregularized_binary_noise_counts(
            masks.coexistence),
    )
end

"""Count raw one-cell islands and holes in the T1 and T2 period masks."""
function sax_nonregularized_period_raster_noise_audit(
        atlas, raw_model::NamedTuple)
    masks = sax_nonregularized_period_masks(atlas, raw_model)
    return (
        t1=_sax_nonregularized_binary_noise_counts(masks.t1),
        t2=_sax_nonregularized_binary_noise_counts(masks.t2),
        coexistence=_sax_nonregularized_binary_noise_counts(
            masks.coexistence),
    )
end

"""Plot validated eta=0 mode-1- and mode-2-dominant sheets independently."""
function plot_sax_nonregularized_stable_register_sheets(
        atlas; size=(1500, 620))
    sheets = atlas.sheets
    masks = _sax_nonregularized_modal_masks(sheets)
    function panel(mask, color_value, title_value)
        values = fill(NaN, Base.size(mask))
        values[mask] .= 1.0
        p = heatmap(
            sheets.gamma, sheets.zeta, values;
            color=cgrad([color_value, color_value]),
            background_color_inside=colorant"#E5E5E5",
            colorbar=false, clims=(0, 1),
            xlabel="γ", ylabel="ζ", title=title_value,
            framestyle=:box, legend=false)
        return p
    end
    low = panel(masks.low, colorant"#D55E00",
                "Validated low register (mode 1 fundamental)")
    high = panel(masks.high, colorant"#0072B2",
                 "Validated high register (clear mode 2 dominance)")
    return plot(low, high; layout=(1, 2), size=size)
end

"""Plot the raw validated T1- and T2-period existence sheets independently."""
function plot_sax_nonregularized_stable_period_sheets(
        atlas, raw_model::NamedTuple; size=(1500, 620))
    sheets = atlas.sheets
    masks = sax_nonregularized_period_masks(atlas, raw_model)
    function panel(mask, color_value, title_value)
        values = fill(NaN, Base.size(mask))
        values[mask] .= 1.0
        return heatmap(
            sheets.gamma, sheets.zeta, values;
            color=cgrad([color_value, color_value]),
            background_color_inside=colorant"#E5E5E5",
            colorbar=false, clims=(0, 1),
            xlabel="γ", ylabel="ζ", title=title_value,
            framestyle=:box, legend=false)
    end
    t1 = panel(masks.t1, colorant"#D55E00",
               "Validated minimal period T₁")
    t2 = panel(masks.t2, colorant"#0072B2",
               "Validated minimal period T₂")
    return plot(t1, t2; layout=(1, 2), size=size)
end

"""Plot exact modal register attraction independently of the P1/P2 label."""
function plot_sax_nonregularized_multistability_atlas(
        atlas; title::AbstractString="", size=(1050, 760),
        show_p2::Bool=false)
    sheets = atlas.sheets
    values = _sax_nonregularized_categorical_values(sheets)
    palette = cgrad([
        colorant"#E5E5E5", colorant"#D55E00",
        colorant"#0072B2", colorant"#CC79A7",
        colorant"#999999"], categorical=true)
    p = heatmap(
        sheets.gamma, sheets.zeta, values;
        color=palette, clims=(-0.5, 4.5), colorbar=false,
        xlabel="γ", ylabel="ζ", title=title,
        framestyle=:box, legend=:topright, size=size)
    for (color_value, label_value) in (
            (colorant"#D55E00", "Low mode"),
            (colorant"#0072B2", "High mode"),
            (colorant"#CC79A7", "Both attract"),
            (colorant"#999999", "Balanced only"),
            (colorant"#E5E5E5", "Unresolved"))
        scatter!(p, [NaN], [NaN]; marker=:square,
                 markercolor=color_value, markerstrokecolor=color_value,
                 markersize=8, label=label_value)
    end
    if show_p2 && any(sheets.stable_p2)
        indices = Tuple.(findall(sheets.stable_p2))
        rows, columns = first.(indices), last.(indices)
        scatter!(p, sheets.gamma[columns], sheets.zeta[rows];
                 marker=:circle, markercolor=colorant"#E69F00",
                 markerstrokecolor=:white, markersize=4,
                 label="P2-labelled")
    end
    return p
end

function _sax_nonregularized_left_edge(gamma, mask)
    return [begin
        column = findfirst(@view mask[row, :])
        isnothing(column) ? NaN : float(gamma[column])
    end for row in axes(mask, 1)]
end

"""
Compute a local Gaussian consensus score for a Boolean existence mask.

The kernel is specified in physical parameter units, so the same smoothing
has the same meaning on the ordinary and refined grids. Missing point caches
are excluded from the denominator rather than counted as negative evidence.
This is a presentation statistic, never a replacement for the raw mask.
"""
function sax_nonregularized_consensus_score(
        gamma::AbstractVector, zeta::AbstractVector,
        mask::AbstractMatrix{Bool};
        known::AbstractMatrix{Bool}=trues(size(mask)),
        sigma_gamma::Real=0.020,
        sigma_zeta::Real=0.040,
        radius_sigma::Real=2.5)
    size(mask) == (length(zeta), length(gamma)) ||
        throw(DimensionMismatch("mask dimensions must be (zeta, gamma)"))
    size(known) == size(mask) ||
        throw(DimensionMismatch("known mask dimensions must match mask"))
    sigma_gamma > 0 && sigma_zeta > 0 && radius_sigma > 0 ||
        throw(ArgumentError("consensus scales must be positive"))
    score = fill(NaN, size(mask))
    support = zeros(Int, size(mask))
    gamma_radius = radius_sigma * float(sigma_gamma)
    zeta_radius = radius_sigma * float(sigma_zeta)
    for row in axes(mask, 1), column in axes(mask, 2)
        numerator = 0.0
        denominator = 0.0
        samples = 0
        row_first = searchsortedfirst(zeta, zeta[row] - zeta_radius)
        row_last = searchsortedlast(zeta, zeta[row] + zeta_radius)
        column_first = searchsortedfirst(
            gamma, gamma[column] - gamma_radius)
        column_last = searchsortedlast(
            gamma, gamma[column] + gamma_radius)
        for neighbour_row in row_first:row_last
            dz = abs(float(zeta[neighbour_row] - zeta[row]))
            for neighbour_column in column_first:column_last
                known[neighbour_row, neighbour_column] || continue
                dg = abs(float(
                    gamma[neighbour_column] - gamma[column]))
                weight = exp(-0.5 * (
                    (dg / sigma_gamma)^2 +
                    (dz / sigma_zeta)^2))
                denominator += weight
                numerator += weight *
                    mask[neighbour_row, neighbour_column]
                samples += 1
            end
        end
        support[row, column] = samples
        denominator > 0 && (score[row, column] = numerator / denominator)
    end
    return (score=score, support=support)
end

function _sax_nonregularized_bilinear_field(
        gamma::AbstractVector, zeta::AbstractVector,
        values::AbstractMatrix{<:Real}; factor::Integer=5)
    factor >= 1 || throw(ArgumentError("display refinement must be positive"))
    size(values) == (length(zeta), length(gamma)) ||
        throw(DimensionMismatch("field dimensions must be (zeta, gamma)"))
    dense_gamma = collect(range(
        first(gamma), last(gamma);
        length=max(1, (length(gamma) - 1) * factor + 1)))
    dense_zeta = collect(range(
        first(zeta), last(zeta);
        length=max(1, (length(zeta) - 1) * factor + 1)))
    dense = Matrix{Float64}(
        undef, length(dense_zeta), length(dense_gamma))
    for (row, zeta_value) in enumerate(dense_zeta),
            (column, gamma_value) in enumerate(dense_gamma)
        source_column = length(gamma) == 1 ? 1 : clamp(
            searchsortedlast(gamma, gamma_value), 1, length(gamma) - 1)
        source_row = length(zeta) == 1 ? 1 : clamp(
            searchsortedlast(zeta, zeta_value), 1, length(zeta) - 1)
        next_column = min(source_column + 1, length(gamma))
        next_row = min(source_row + 1, length(zeta))
        gamma_fraction = next_column == source_column ? 0.0 :
            (gamma_value - gamma[source_column]) /
                (gamma[next_column] - gamma[source_column])
        zeta_fraction = next_row == source_row ? 0.0 :
            (zeta_value - zeta[source_row]) /
                (zeta[next_row] - zeta[source_row])
        corners = (
            float(values[source_row, source_column]),
            float(values[source_row, next_column]),
            float(values[next_row, source_column]),
            float(values[next_row, next_column]),
        )
        dense[row, column] = any(isnan, corners) ? NaN :
            (1 - zeta_fraction) * (
                (1 - gamma_fraction) * corners[1] +
                gamma_fraction * corners[2]) +
            zeta_fraction * (
                (1 - gamma_fraction) * corners[3] +
                gamma_fraction * corners[4])
    end
    return (gamma=dense_gamma, zeta=dense_zeta, values=dense)
end

"""
Derive paper-oriented local-support fields from two exact Boolean masks.

The raw existence masks remain authoritative. The returned fields estimate
whether a local parameter neighbourhood consistently supports each class;
they do not create new validated attractors at individual parameter points.
"""
function sax_nonregularized_consensus_fields(
        atlas; sigma_gamma::Real=0.020,
        sigma_zeta::Real=0.040,
        smoothing_factor::Real=1.0,
        threshold::Real=0.55,
        display_refinement::Integer=5,
        classification_masks=nothing)
    smoothing_factor >= 0 || throw(ArgumentError(
        "smoothing_factor must be non-negative"))
    0 < threshold < 1 ||
        throw(ArgumentError("consensus threshold must lie in (0,1)"))
    sheets = atlas.sheets
    masks = isnothing(classification_masks) ?
        _sax_nonregularized_modal_masks(sheets) : classification_masks
    low_mask = hasproperty(masks, :low) ? masks.low : masks.t1
    high_mask = hasproperty(masks, :high) ? masks.high : masks.t2
    known = hasproperty(sheets, :cache_present) ?
        sheets.cache_present : trues(size(low_mask))
    effective_sigma_gamma = float(smoothing_factor) * float(sigma_gamma)
    effective_sigma_zeta = float(smoothing_factor) * float(sigma_zeta)
    low, high = if iszero(smoothing_factor)
        # Factor zero is the unsmoothed classified raster. Unknown cache cells
        # remain NaN and therefore cannot turn into positive modal support.
        raw_score(mask) = (
            score=map((value, is_known) -> is_known ? float(value) : NaN,
                      mask, known),
            support=Int.(known),
        )
        raw_score(low_mask), raw_score(high_mask)
    else
        sax_nonregularized_consensus_score(
            sheets.gamma, sheets.zeta, low_mask;
            known=known, sigma_gamma=effective_sigma_gamma,
            sigma_zeta=effective_sigma_zeta),
        sax_nonregularized_consensus_score(
            sheets.gamma, sheets.zeta, high_mask;
            known=known, sigma_gamma=effective_sigma_gamma,
            sigma_zeta=effective_sigma_zeta)
    end
    effective_refinement = iszero(smoothing_factor) ? 1 :
        Int(display_refinement)
    dense_low = _sax_nonregularized_bilinear_field(
        sheets.gamma, sheets.zeta, low.score;
        factor=effective_refinement)
    dense_high = _sax_nonregularized_bilinear_field(
        sheets.gamma, sheets.zeta, high.score;
        factor=effective_refinement)
    values = zeros(Int8, size(dense_low.values))
    low_consensus = dense_low.values .>= threshold
    high_consensus = dense_high.values .>= threshold
    values[low_consensus .& .!high_consensus] .= 1
    values[high_consensus .& .!low_consensus] .= 2
    values[low_consensus .& high_consensus] .= 3
    return (
        gamma=dense_low.gamma,
        zeta=dense_low.zeta,
        low=dense_low.values,
        high=dense_high.values,
        values=values,
        threshold=float(threshold),
        smoothing_factor=float(smoothing_factor),
        sigma_gamma=effective_sigma_gamma,
        sigma_zeta=effective_sigma_zeta,
        base_sigma_gamma=float(sigma_gamma),
        base_sigma_zeta=float(sigma_zeta),
        display_refinement=effective_refinement,
        raw=(low=low, high=high),
    )
end

"""
Derive the paper fields from the two principal minimal-period masks.

The `values` codes are 1 for T1 only, 2 for T2 only, and 3 for simultaneous
local support of distinct T1 and T2 attractors. Smoothing changes only this
presentation field; the raw period masks remain the evidence.
"""
function sax_nonregularized_period_consensus_fields(
        atlas, raw_model::NamedTuple; kwargs...)
    masks = sax_nonregularized_period_masks(atlas, raw_model)
    return merge(
        sax_nonregularized_consensus_fields(
            atlas; classification_masks=masks, kwargs...),
        (classification=:minimal_recurrence_period,
         mask_source=masks.source,
         raw_period_masks=masks))
end

"""
Plot the modal map with sampled left edges and gesture-direction arrows.
The edges are descriptive limits of validated attraction on this raster, not
continued bifurcation curves.
"""
function plot_sax_nonregularized_directional_register_map(
        atlas; title::AbstractString="", size=(1120, 800),
        show_arrows::Bool=true)
    sheets = atlas.sheets
    masks = _sax_nonregularized_modal_masks(sheets)
    p = plot_sax_nonregularized_multistability_atlas(
        atlas; title=title, size=size, show_p2=false)
    high_edge = _sax_nonregularized_left_edge(sheets.gamma, masks.high)
    plot!(p, high_edge, sheets.zeta;
          color=colorant"#0072B2", linewidth=2.5, linestyle=:dash,
          label="Left edge high attraction",
          legend=:outertopright)
    if show_arrows
        plot!(p, [0.88, 0.50], [0.965, 0.965];
              color=:black, linewidth=2.5, arrow=true, label="",
              ylims=(minimum(sheets.zeta), 0.99))
        annotate!(p, 0.69, 0.978,
                  text("decreasing γ: high → low", 10))
        plot!(p, [0.50, 0.88], [0.93, 0.93];
              color=:black, linewidth=2.5, arrow=true, label="")
        annotate!(p, 0.69, 0.942,
                  text("increasing γ: basin-dependent low → high", 10))
    end
    return p
end

function _sax_nonregularized_figure6_pattern_grid(
        gamma, zeta, classes; spacing::Real=0.01)
    size(classes) == (length(zeta), length(gamma)) || throw(
        DimensionMismatch(
            "pattern classes must have dimensions (zeta, gamma)"))
    spacing > 0 || throw(ArgumentError("pattern spacing must be positive"))
    issorted(gamma) && issorted(zeta) || throw(ArgumentError(
        "pattern axes must be sorted in increasing order"))
    (!isempty(gamma) && !isempty(zeta)) || throw(ArgumentError(
        "pattern axes must not be empty"))

    # Figure 6 was drawn on a 0.01 x 0.01 parameter grid with motifs every
    # second cell.  The presentation fields used here are much denser and
    # anisotropic, so sample them on the original physical grid before calling
    # the original renderer.  Otherwise `pattern_decimation=2` would mean a
    # different physical spacing on each axis.
    gamma_count = floor(Int,
        (float(last(gamma)) - float(first(gamma))) / spacing + 1e-8) + 1
    zeta_count = floor(Int,
        (float(last(zeta)) - float(first(zeta))) / spacing + 1e-8) + 1
    target_gamma = float(first(gamma)) .+
        float(spacing) .* collect(0:(gamma_count - 1))
    target_zeta = float(first(zeta)) .+
        float(spacing) .* collect(0:(zeta_count - 1))
    if float(last(gamma)) - last(target_gamma) > spacing * 1e-6
        push!(target_gamma, float(last(gamma)))
    else
        target_gamma[end] = float(last(gamma))
    end
    if float(last(zeta)) - last(target_zeta) > spacing * 1e-6
        push!(target_zeta, float(last(zeta)))
    else
        target_zeta[end] = float(last(zeta))
    end

    nearest_index(axis, value) = begin
        upper = searchsortedfirst(axis, value)
        upper <= 1 && return 1
        upper > length(axis) && return length(axis)
        lower = upper - 1
        abs(float(axis[lower]) - value) <=
            abs(float(axis[upper]) - value) ? lower : upper
    end
    sampled = Matrix{Int8}(
        undef, length(target_zeta), length(target_gamma))
    for (row, zeta_value) in enumerate(target_zeta),
            (column, gamma_value) in enumerate(target_gamma)
        sampled[row, column] = Int8(classes[
            nearest_index(zeta, zeta_value),
            nearest_index(gamma, gamma_value)])
    end
    return (gamma=target_gamma, zeta=target_zeta, classes=sampled)
end

function _sax_nonregularized_figure6_pattern_plot(
        gamma, zeta, classes;
        title::AbstractString="",
        size=(1120, 800),
        spacing::Real=0.01,
        pattern_decimation::Integer=2,
        pattern_legend_labels=("Mode 1", "Mode 2", "Mode 1+2"),
        pattern_legend_classes=Int8[6, 3, 5],
        legend_fill_entries=(),
        legend_point_entries=(),
        legend_panel_fraction::Real=0.20,
        legend_fontsize::Real=12,
        legend_position::Symbol=:inside,
        inside_legend_position::Symbol=:topright,
        show_region_boundaries::Bool=true)
    pattern_decimation >= 1 || throw(ArgumentError(
        "pattern decimation must be positive"))
    sampled = _sax_nonregularized_figure6_pattern_grid(
        gamma, zeta, classes; spacing=spacing)
    dominant_mode1 = zeros(Int8, Base.size(sampled.classes))
    dominant_mode2 = zeros(Int8, Base.size(sampled.classes))
    for index in eachindex(sampled.classes)
        class = sampled.classes[index]
        if class == 1
            # Class 21: ascending diagonals (///) for Mode 1.
            dominant_mode1[index] = 2
            dominant_mode2[index] = 1
        elseif class == 2
            # Class 12: descending diagonals (\\\) for Mode 2.
            dominant_mode1[index] = 1
            dominant_mode2[index] = 2
        elseif class == 3
            # Class 11: crossed diagonals (XXX) for Mode 1+2.
            dominant_mode1[index] = 1
            dominant_mode2[index] = 1
        elseif class != 0
            throw(ArgumentError(
                "non-regularized pattern classes must lie between 0 and 3"))
        end
    end
    maps = (
        gamma_values=sampled.gamma,
        zeta_values=sampled.zeta,
        dominant_mode1=dominant_mode1,
        dominant_mode2=dominant_mode2,
        grow_mode1=zeros(Int8, Base.size(sampled.classes)),
        grow_mode2=zeros(Int8, Base.size(sampled.classes)),
    )
    axis = plot_sweep_mode_regions_pattern(
        maps;
        title=title,
        region_sep_color=:black,
        region_sep_lw=1,
        pattern_decimation=pattern_decimation,
        pattern_legend_labels=pattern_legend_labels,
        pattern_legend_classes=pattern_legend_classes,
        legend_fill_entries=legend_fill_entries,
        legend_point_entries=legend_point_entries,
        legend_fontsize=legend_fontsize,
        legend_label_x=1.42,
        legend_label_halign=:left,
        legend_panel_fraction=legend_panel_fraction,
        pattern_legend_position=legend_position,
        inside_legend_position=inside_legend_position,
        region_sep_alpha=show_region_boundaries ? 1.0 : 0.0,
    )
    plot!(axis; size=size)
    return axis
end

"""
Plot the locally smoothed exact-model register map with Figure-6-style hatching.

The displayed names refer to modal content: Mode 1 is the low-register
response, Mode 2 is the high-register response, and Mode 1+2 means that both
attracting responses have local support. The hatching changes presentation
only; `sax_nonregularized_consensus_fields` remains the numerical source.
"""
function plot_sax_nonregularized_patterned_mode_map(
        atlas; title::AbstractString="", size=(1120, 800),
        fields=nothing,
        sigma_gamma::Real=0.020,
        sigma_zeta::Real=0.040,
        smoothing_factor::Real=1.0,
        threshold::Real=0.55,
        display_refinement::Integer=5,
        pattern_spacing::Real=0.01,
        pattern_decimation::Integer=2,
        pattern_legend_labels=("Mode 1", "Mode 2", "Mode 1+2"),
        legend_point_entries=(),
        legend_panel_fraction::Real=0.20,
        legend_position::Symbol=:inside,
        inside_legend_position::Symbol=:topright,
        smooth_boundaries::Bool=true,
        boundary_linewidth::Real=1.25)
    selected_fields = isnothing(fields) ?
        sax_nonregularized_consensus_fields(
            atlas; sigma_gamma=sigma_gamma,
            sigma_zeta=sigma_zeta, smoothing_factor=smoothing_factor,
            threshold=threshold,
            display_refinement=display_refinement) : fields
    axis = _sax_nonregularized_figure6_pattern_plot(
        selected_fields.gamma, selected_fields.zeta, selected_fields.values;
        title=title,
        size=size,
        spacing=pattern_spacing,
        pattern_decimation=pattern_decimation,
        pattern_legend_labels=pattern_legend_labels,
        legend_point_entries=legend_point_entries,
        legend_panel_fraction=legend_panel_fraction,
        legend_position=legend_position,
        inside_legend_position=inside_legend_position,
        show_region_boundaries=!smooth_boundaries)
    if smooth_boundaries
        for score in (selected_fields.low, selected_fields.high)
            contour!(axis, selected_fields.gamma, selected_fields.zeta, score;
                     levels=[selected_fields.threshold], color=:black,
                     linewidth=boundary_linewidth, linestyle=:solid,
                     colorbar_entry=false, label="")
        end
    end
    return axis
end


"""Plot T1, T2, and T1+T2 period support with Figure-6-style hatching."""
function plot_sax_nonregularized_patterned_period_map(
        atlas, raw_model::NamedTuple; fields=nothing, kwargs...)
    selected_fields = isnothing(fields) ?
        sax_nonregularized_period_consensus_fields(
            atlas, raw_model) : fields
    return plot_sax_nonregularized_patterned_mode_map(
        atlas; fields=selected_fields,
        pattern_legend_labels=("T₁", "T₂", "T₁ + T₂"), kwargs...)
end

function _sax_nonregularized_modal_edge_points(atlas)
    sheets = atlas.sheets
    masks = _sax_nonregularized_modal_masks(sheets)
    modal_coexistence(point) = masks.coexistence[
        argmin(abs.(sheets.zeta .- point.zeta)),
        argmin(abs.(sheets.gamma .- point.gamma))]
    return [point for point in atlas.edge.points
            if point.status == :complete && modal_coexistence(point)]
end

"""
Smooth the binary observation "still mixed at the integration cutoff" over
the sparse exact edge experiments.

Only completed edge experiments inside validated modal coexistence contribute.
The score is a local Gaussian-weighted fraction, not evidence of an invariant
multiphonic attractor. `minimum_support` prevents extrapolation from an
isolated censored sample, and the reported region is clipped to local Mode 1+2
support from the register map.
"""
function sax_nonregularized_mixed_cutoff_fields(
        atlas; sigma_gamma::Real=0.030,
        sigma_zeta::Real=0.060,
        radius_sigma::Real=2.5,
        threshold::Real=0.55,
        minimum_support::Integer=3,
        modal_sigma_gamma::Real=0.020,
        modal_sigma_zeta::Real=0.040,
        modal_smoothing_factor::Real=1.0,
        modal_threshold::Real=0.55,
        modal_fields=nothing,
        classification_fields=nothing,
        edge_points=nothing,
        eligibility_region=nothing,
        display_refinement::Integer=5)
    sigma_gamma > 0 && sigma_zeta > 0 && radius_sigma > 0 ||
        throw(ArgumentError("mixed-cutoff smoothing scales must be positive"))
    0 < threshold < 1 || throw(ArgumentError(
        "mixed-cutoff threshold must lie in (0,1)"))
    minimum_support >= 1 || throw(ArgumentError(
        "minimum_support must be positive"))
    sheets = atlas.sheets
    points = isnothing(edge_points) ?
        _sax_nonregularized_modal_edge_points(atlas) : edge_points
    point_gamma = Float64[float(point.gamma) for point in points]
    point_zeta = Float64[float(point.zeta) for point in points]
    point_censored = Bool[Bool(point.metrics.right_censored)
                          for point in points]
    score = zeros(Float64, length(sheets.zeta), length(sheets.gamma))
    support = zeros(Float64, size(score))
    for (row, zeta_value) in enumerate(sheets.zeta),
            (column, gamma_value) in enumerate(sheets.gamma)
        numerator = 0.0
        denominator = 0.0
        neighbours = 0
        for index in eachindex(point_gamma)
            normalized_gamma =
                (gamma_value - point_gamma[index]) / sigma_gamma
            normalized_zeta =
                (zeta_value - point_zeta[index]) / sigma_zeta
            radius_squared = normalized_gamma^2 + normalized_zeta^2
            radius_squared <= radius_sigma^2 || continue
            weight = exp(-0.5radius_squared)
            denominator += weight
            numerator += weight * point_censored[index]
            neighbours += 1
        end
        support[row, column] = neighbours
        denominator > 0 && (score[row, column] = numerator / denominator)
    end
    dense_score = _sax_nonregularized_bilinear_field(
        sheets.gamma, sheets.zeta, score; factor=display_refinement)
    dense_support = _sax_nonregularized_bilinear_field(
        sheets.gamma, sheets.zeta, support; factor=display_refinement)
    base_fields = isnothing(classification_fields) ? modal_fields :
        classification_fields
    modal = isnothing(base_fields) ?
        sax_nonregularized_consensus_fields(
            atlas; sigma_gamma=modal_sigma_gamma,
            sigma_zeta=modal_sigma_zeta,
            smoothing_factor=modal_smoothing_factor,
            threshold=modal_threshold,
            display_refinement=display_refinement) : base_fields
    eligible = isnothing(eligibility_region) ?
        modal.values .== 3 : BitMatrix(eligibility_region)
    size(eligible) == size(dense_score.values) || throw(DimensionMismatch(
        "eligibility_region must match the refined mixed field"))
    region = (dense_score.values .>= threshold) .&
        (dense_support.values .>= minimum_support) .& eligible
    return (
        gamma=dense_score.gamma,
        zeta=dense_score.zeta,
        score=dense_score.values,
        support=dense_support.values,
        region=region,
        threshold=float(threshold),
        minimum_support=Int(minimum_support),
        sigma_gamma=float(sigma_gamma),
        sigma_zeta=float(sigma_zeta),
        completed_points=length(points),
        right_censored_points=count(
            point -> point.metrics.right_censored, points),
        points=points,
        modal=modal,
        eligibility_region=eligible,
    )
end

"""
Construct the gray finite-time mixed layer for the recurrence-period map.

Completed T1-to-T2 edges contribute inside T1+T2 support. The central
low-P1-to-high-P2 transition is also retained because it contains distinct
validated attractors and long mixed transients even though both have global
period near T1. This second source is restricted by the independently
validated low/high coexistence mask. The returned `coverage` and `evidence`
records expose both sources and all uncovered points.
"""
function sax_nonregularized_period_mixed_cutoff_fields(
        atlas, raw_model::NamedTuple;
        period_fields=nothing,
        period_sigma_gamma::Real=0.020,
        period_sigma_zeta::Real=0.040,
        period_smoothing_factor::Real=1.0,
        period_threshold::Real=0.55,
        display_refinement::Integer=5,
        kwargs...)
    fields = isnothing(period_fields) ?
        sax_nonregularized_period_consensus_fields(
            atlas, raw_model;
            sigma_gamma=period_sigma_gamma,
            sigma_zeta=period_sigma_zeta,
            smoothing_factor=period_smoothing_factor,
            threshold=period_threshold,
            display_refinement=display_refinement) : period_fields
    coverage = sax_nonregularized_period_edge_coverage(atlas, raw_model)
    evidence = sax_nonregularized_mixed_edge_evidence(atlas, raw_model)
    p2_transition_mask = _sax_nonregularized_low_p1_high_p2_mask(atlas)
    p2_transition_fields = sax_nonregularized_consensus_fields(
        atlas;
        sigma_gamma=period_sigma_gamma,
        sigma_zeta=period_sigma_zeta,
        smoothing_factor=period_smoothing_factor,
        threshold=period_threshold,
        display_refinement=display_refinement,
        classification_masks=(
            low=p2_transition_mask,
            high=falses(size(p2_transition_mask))))
    eligibility = (fields.values .== 3) .|
        (p2_transition_fields.low .>= period_threshold)
    mixed = sax_nonregularized_mixed_cutoff_fields(
        atlas;
        classification_fields=fields,
        edge_points=evidence.points,
        eligibility_region=eligibility,
        display_refinement=display_refinement,
        kwargs...)
    return merge(mixed, (
        periodic=fields,
        coverage=coverage,
        evidence=evidence,
        p2_transition=p2_transition_fields,
        classification=:t1_t2_and_low_p1_high_p2_edges,
    ))
end

"""
Plot three accessibility regions and the right-censored mixed-edge subset.

The first class, second class, and their accessibility overlap retain the
Figure-6 patterns. Parameter cells still mixed at the finite integration
cutoff form a fourth, solid-gray class; they do not erase the other regions.
"""
function plot_sax_nonregularized_mixed_cutoff_map(
        atlas; title::AbstractString="", size=(1120, 800),
        fields=nothing,
        pattern_spacing::Real=0.01,
        pattern_decimation::Integer=2,
        mixed_alpha::Real=1.0,
        legend_point_entries=(),
        legend_panel_fraction::Real=0.32,
        legend_position::Symbol=:inside,
        inside_legend_position::Symbol=:topright,
        smooth_boundaries::Bool=true,
        boundary_linewidth::Real=1.25,
        kwargs...)
    0 <= mixed_alpha <= 1 || throw(ArgumentError(
        "mixed_alpha must lie between zero and one"))
    selected_fields = isnothing(fields) ?
        sax_nonregularized_mixed_cutoff_fields(atlas; kwargs...) : fields
    modal = hasproperty(selected_fields, :periodic) ?
        selected_fields.periodic :
        hasproperty(selected_fields, :modal) ? selected_fields.modal :
        sax_nonregularized_consensus_fields(atlas)
    pattern_labels = hasproperty(selected_fields, :periodic) ?
        ("T₁", "T₂", "T₁ + T₂") :
        ("Mode 1", "Mode 2", "Mode 1+2")
    solid_gray = colorant"#8C8C8C"
    axis = _sax_nonregularized_figure6_pattern_plot(
        modal.gamma, modal.zeta, modal.values;
        title=title,
        size=size,
        spacing=pattern_spacing,
        pattern_decimation=pattern_decimation,
        legend_panel_fraction=legend_panel_fraction,
        legend_position=:none,
        show_region_boundaries=!smooth_boundaries)
    main_axis = axis
    if smooth_boundaries
        for score in (modal.low, modal.high)
            contour!(main_axis, modal.gamma, modal.zeta, score;
                     levels=[modal.threshold], color=:black,
                     linewidth=boundary_linewidth, linestyle=:solid,
                     colorbar_entry=false, label="")
        end
    end
    if any(selected_fields.region)
        # A filled contour, rather than a second heatmap, preserves drawing
        # order: hatch strokes and the custom inset key remain legible above
        # the translucent mixed-at-cutoff layer.
        contourf!(
            main_axis,
            selected_fields.gamma,
            selected_fields.zeta,
            Float64.(selected_fields.region);
            levels=[0.5, 1.5],
            color=cgrad([solid_gray, solid_gray]),
            linewidth=0,
            fillalpha=mixed_alpha,
            colorbar=false,
            colorbar_entry=false,
            label="",
        )
        any(.!selected_fields.region) && contour!(
            main_axis,
            selected_fields.gamma,
            selected_fields.zeta,
            Float64.(selected_fields.region);
            levels=[0.5], color=:black, linewidth=1.25,
            linestyle=:solid, label="", colorbar_entry=false)
    end
    legend_position in (:inside, :none) || throw(ArgumentError(
        "mixed-cutoff legend_position must be :inside or :none"))
    if legend_position == :inside
        _sax_pattern_legend_inside!(
            main_axis, modal.gamma, modal.zeta;
            pattern_labels=pattern_labels,
            pattern_classes=Int8[6, 3, 5],
            fill_entries=((
                label="Still mixed at cutoff",
                color=solid_gray,
                alpha=mixed_alpha),),
            point_entries=legend_point_entries,
            position=inside_legend_position,
            fontsize=12)
    end
    return axis
end

"""
Plot a smooth local-consensus summary while preserving the raw map elsewhere.

The dashed blue contour is the chosen high-register consensus level. It is a
descriptive switching locus with a kernel-scale uncertainty, not a continued
bifurcation curve.
"""
function plot_sax_nonregularized_consensus_directional_map(
        atlas; title::AbstractString="", size=(1120, 800),
        sigma_gamma::Real=0.020,
        sigma_zeta::Real=0.040,
        threshold::Real=0.55,
        display_refinement::Integer=5)
    fields = sax_nonregularized_consensus_fields(
        atlas; sigma_gamma=sigma_gamma,
        sigma_zeta=sigma_zeta, threshold=threshold,
        display_refinement=display_refinement)
    palette = cgrad([
        colorant"#E5E5E5", colorant"#D55E00",
        colorant"#0072B2", colorant"#CC79A7"],
        categorical=true)
    p = heatmap(
        fields.gamma, fields.zeta, fields.values;
        color=palette, clims=(-0.5, 3.5), colorbar=false,
        xlabel="γ", ylabel="ζ", title=title,
        framestyle=:box, legend=:outertopright, size=size)
    for (color_value, label_value) in (
            (colorant"#D55E00", "Low consensus"),
            (colorant"#0072B2", "High consensus"),
            (colorant"#CC79A7", "Both consensus"),
            (colorant"#E5E5E5", "Transition / insufficient consensus"))
        scatter!(p, [NaN], [NaN]; marker=:square,
                 markercolor=color_value, markerstrokecolor=color_value,
                 markersize=8, label=label_value)
    end
    if length(fields.gamma) > 1 && length(fields.zeta) > 1
        contour!(p, fields.gamma, fields.zeta, fields.high;
                 levels=[fields.threshold], color=colorant"#0072B2",
                 linewidth=2.5, linestyle=:dash, label="")
    end
    plot!(p, [NaN], [NaN]; color=colorant"#0072B2",
          linewidth=2.5, linestyle=:dash,
          label="High-consensus edge")
    plot!(p, [0.88, 0.50], [0.965, 0.965];
          color=:black, linewidth=2.5, arrow=true, label="",
          ylims=(minimum(fields.zeta), 0.99))
    annotate!(p, 0.69, 0.978,
              text("decreasing γ: high → low", 10))
    plot!(p, [0.50, 0.88], [0.93, 0.93];
          color=:black, linewidth=2.5, arrow=true, label="")
    annotate!(p, 0.69, 0.942,
              text("increasing γ: basin-dependent low → high", 10))
    return p
end

"""Plot P2 as a temporal label, coloured by its independent modal content."""
function plot_sax_nonregularized_p2_modal_diagnostic(
        atlas; title::AbstractString="P2-labelled responses by modal content",
        size=(1000, 700))
    sheets = atlas.sheets
    required = (:stable_p2_mode1, :stable_p2_mode2,
                :stable_p2_modal_mixed)
    all(name -> hasproperty(sheets, name), required) || return plot(
        title="Reassemble the exact product to derive the P2 modal diagnostic",
        framestyle=:box, legend=false, size=size)
    mode1 = sheets.stable_p2_mode1
    mode2 = sheets.stable_p2_mode2
    balanced = sheets.stable_p2_modal_mixed
    values = zeros(Int8, Base.size(mode1))
    values[mode1 .& .!mode2] .= 1
    values[mode2 .& .!mode1] .= 2
    values[mode1 .& mode2] .= 3
    values[balanced .& .!mode1 .& .!mode2] .= 4
    palette = cgrad([
        colorant"#E5E5E5", colorant"#D55E00",
        colorant"#0072B2", colorant"#CC79A7",
        colorant"#999999"], categorical=true)
    p = heatmap(
        sheets.gamma, sheets.zeta, values;
        color=palette, clims=(-0.5, 4.5), colorbar=false,
        xlabel="γ", ylabel="ζ", title=title,
        framestyle=:box, legend=:topright, size=size)
    for (color_value, label_value) in (
            (colorant"#D55E00", "P2, low-mode dominant"),
            (colorant"#0072B2", "P2, high-mode dominant"),
            (colorant"#CC79A7", "Both P2 modal responses"),
            (colorant"#999999", "P2, balanced"),
            (colorant"#E5E5E5", "No validated P2 label"))
        scatter!(p, [NaN], [NaN]; marker=:square,
                 markercolor=color_value, markerstrokecolor=color_value,
                 markersize=8, label=label_value)
    end
    return p
end

"""Plot exact sampled mixed-mode edge residence without interpolating it."""
function plot_sax_nonregularized_edge_diagnostic(
        atlas; title::AbstractString="", size=(950, 690))
    p = plot_sax_nonregularized_multistability_atlas(
        atlas; title=title, size=size, show_p2=false)
    sheets = atlas.sheets
    masks = _sax_nonregularized_modal_masks(sheets)
    function modal_coexistence(point)
        row = argmin(abs.(sheets.zeta .- point.zeta))
        column = argmin(abs.(sheets.gamma .- point.gamma))
        return masks.coexistence[row, column]
    end
    completed = [point for point in atlas.edge.points
                 if point.status == :complete && modal_coexistence(point)]
    escaped = [point for point in completed
               if !point.metrics.right_censored]
    censored = [point for point in completed
                if point.metrics.right_censored]
    !isempty(escaped) && scatter!(
        p, [point.gamma for point in escaped],
        [point.zeta for point in escaped];
        marker=:circle, marker_z=[point.escape_cycles for point in escaped],
        color=:viridis, markersize=5,
        label="Mixed transient ended")
    !isempty(censored) && scatter!(
        p, [point.gamma for point in censored],
        [point.zeta for point in censored];
        marker=:diamond, markercolor=colorant"#6A3D9A",
        markerstrokecolor=:white, markersize=6,
        label="Still mixed at cutoff")
    return p
end
