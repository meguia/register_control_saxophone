# Cache namespace and comparison helpers for the parallel regularized study.

const SAX_REGULARIZED_CANONICAL_RUN_ID = "canonical_v1"

function _sax_regularized_validate_run_id(run_id::AbstractString)
    value = String(run_id)
    isempty(value) && throw(ArgumentError("run_id cannot be empty"))
    occursin(r"^[A-Za-z0-9_-]+$", value) || throw(ArgumentError(
        "run_id accepts only letters, numbers, underscores, and hyphens"))
    return value
end

"""Filesystem-safe short name for a positive regularization value."""
function sax_regularization_tag(eta::Real)
    eta > 0 || throw(ArgumentError("eta must be positive"))
    value = replace(@sprintf("%.8g", float(eta)), "." => "p", "-" => "m", "+" => "")
    return "eta_$(value)"
end

"""
    sax_regularized_study_paths(project_root; fingering="Dx4", eta=1e-3,
                                profile=:final, run_id=nothing)

Return every input and output path for the parallel study. All generated files
live below `bifurcation_regularized_<fingering>` and cannot overwrite the
piecewise-model products below `bifurcation_<fingering>`.
"""
function sax_regularized_study_paths(
        project_root::AbstractString;
        fingering::AbstractString="Dx4",
        eta::Real=1e-3,
        profile::Symbol=:final,
        run_id::Union{Nothing,AbstractString}=nothing)
    profile in (:smoke, :pilot, :final) || throw(ArgumentError(
        "profile must be :smoke, :pilot, or :final"))
    tag = sax_regularization_tag(eta)
    processed = joinpath(
        project_root, "src", "sessions", "processed_data")
    family_root = joinpath(
        processed, "bifurcation_regularized_$(fingering)")
    eta_root = joinpath(family_root, "colinot_$(tag)")
    selected_run = isnothing(run_id) ? nothing :
        _sax_regularized_validate_run_id(run_id)
    run_root = isnothing(selected_run) ? eta_root :
        joinpath(eta_root, "runs", selected_run)
    root = joinpath(run_root, String(profile))
    dense_focused = joinpath(root, "periodic_schur_ns_dense_focused")
    dense_full = joinpath(root, "periodic_schur_ns_dense_full")
    dense_extended = joinpath(root, "periodic_schur_ns_dense_extended")
    return (
        fingering=String(fingering),
        eta=float(eta),
        eta_tag=tag,
        run_id=selected_run,
        profile=profile,
        model=joinpath(
            project_root, "src", "impedances", "alto", "$(fingering).jld2"),
        family_root=family_root,
        eta_root=eta_root,
        root=root,
        manifest=joinpath(root, "regularized_study_manifest.jld2"),
        complete_product=joinpath(root, "complete_bifurcation_product.jld2"),
        main=joinpath(root, "bifurcation_curves_regularized_$(fingering)_$(tag).jld2"),
        stages=joinpath(root, "stages"),
        dense_focused=dense_focused,
        dense_focused_slices=joinpath(dense_focused, "slices"),
        dense_focused_validation=joinpath(dense_focused, "seed_validation.jld2"),
        dense_focused_seed=joinpath(dense_focused, "refined_seed.jld2"),
        dense_focused_augmented=joinpath(dense_focused, "augmented"),
        dense_full=dense_full,
        dense_full_slices=joinpath(dense_full, "slices"),
        dense_full_validation=joinpath(dense_full, "seed_validation.jld2"),
        dense_full_seed=joinpath(dense_full, "refined_seed.jld2"),
        dense_full_augmented=joinpath(dense_full, "augmented"),
        dense_extended=dense_extended,
        dense_extended_slices=joinpath(dense_extended, "slices"),
        dense_extended_validation=joinpath(
            dense_extended, "seed_validation.jld2"),
        dense_extended_seed=joinpath(dense_extended, "refined_seed.jld2"),
        dense_extended_augmented=joinpath(dense_extended, "augmented"),
        p2_r2=joinpath(root, "p2_r2_periodic_schur"),
        plane_completion=joinpath(root, "fixed_zeta_plane_completion"),
        fold_completion=joinpath(root, "fixed_zeta_fold_completion"),
        figures=joinpath(root, "figures"),
    )
end

"""Settings used by the regularized main continuation for one profile."""
function sax_regularized_bifurcation_settings(profile::Symbol=:final)
    if profile == :smoke
        return sax_bifurcation_settings(
            :pilot;
            gamma_range=(0.30, 0.70),
            zeta_range=(0.10, 0.90),
            zeta_seeds=(0.50,),
            equilibrium_ds=5e-3,
            equilibrium_dsmax=2e-2,
            equilibrium_max_steps=120,
            hopf_ds=5e-3,
            hopf_dsmax=2e-2,
            hopf_max_steps=100,
            po_collocation_intervals=8,
            po_collocation_degree=2,
            po_max_steps=40,
            po_curve_max_steps=30,
        )
    elseif profile in (:pilot, :final)
        return sax_bifurcation_settings(profile)
    end
    throw(ArgumentError("profile must be :smoke, :pilot, or :final"))
end

"""
    sax_regularized_graphical_style(; register_modes=(1, 2, 3))

Return the shared presentation style for the marker-free regularized diagrams
in notebooks 07b and 10b. Completed local curves, including both targeted fold
components, and validated codimension-two points are shown. The mode-1 fold is
red and dashed, while NS is green and PD is orange. `register_modes=(1, 2)`
provides the register-only background used by notebook 10b: it removes H3, all
mode-3 NS curves, and the H3 generalized-Hopf marker. The PQZ-derived NS-like
guide remains dashed and is omitted from the legend.
"""
function sax_regularized_graphical_style(;
        register_modes::Tuple{Vararg{Int}}=(1, 2, 3))
    all(mode -> mode in (1, 2, 3), register_modes) || throw(ArgumentError(
        "register_modes accepts only modes 1, 2, and 3",
    ))
    colors = SAX_GESTURE_COLORBLIND_COLORS
    hopf_palette = (
        1 => (colors.red, "H1"),
        2 => (colors.blue, "H2"),
        3 => (colorant"#666666", "H3"),
    )
    hopf_styles = Tuple(
        (mode, color, label)
        for (mode, (color, label)) in hopf_palette
        if mode in register_modes
    )
    include_mode3 = 3 in register_modes
    return (
        show_markers=false,
        show_dh=true,
        show_dh_legend=true,
        dh_modes=register_modes,
        show_gh=include_mode3,
        show_gh_legend=include_mode3,
        hopf_styles=hopf_styles,
        periodic_order=(:fold, :pd, :ns),
        ns_selection=:all,
        ns_modes=register_modes,
        fold_color=colors.red,
        fold_linestyle=:dash,
        pd_color=colors.orange,
        ns_color=colors.green,
        ns_like_color=colors.green,
        r2_color=colors.lightgreen,
        show_ns_like_legend=false,
    )
end

function _sax_regularized_dense_settings(
        profile::Symbol,
        coverage::Symbol;
        zeta_step::Union{Nothing,Real}=nothing)
    dense_profile = profile == :smoke ? :smoke : profile
    if coverage != :extended
        return sax_dense_pqz_ns_settings(
            dense_profile;
            coverage=coverage,
            zeta_step=zeta_step,
        )
    end

    # The original dense survey intentionally stopped at gamma=0.72. Its
    # NS-like root reaches that boundary near zeta=0.60, although fixed-zeta
    # slices continue to zeta=0.99. Keep those expensive caches immutable and
    # extend only the missing high-zeta/high-gamma part in a separate namespace.
    selected_range = profile == :smoke ? (0.65, 0.65) : (0.55, 0.99)
    selected_step = isnothing(zeta_step) ?
        (profile == :final ? 0.01 : profile == :pilot ? 0.04 : 1.0) :
        float(zeta_step)
    base = sax_dense_pqz_ns_settings(
        dense_profile;
        coverage=:full,
        zeta_range=selected_range,
        zeta_step=selected_step,
    )
    names = fieldnames(SaxDensePQZNSSettings)
    portable = NamedTuple{names}(
        Tuple(getfield(base, name) for name in names))
    return SaxDensePQZNSSettings(; merge(portable, (
        gamma_hint=0.72,
        gamma_range=(0.30, 0.99),
        root_gamma_range=(0.65, 0.99),
        po_max_steps=max(base.po_max_steps, profile == :smoke ? 250 : 1000),
    ))...)
end

"""
    load_sax_regularized_study(raw_model, project_root; ...)

Load only atomically committed regularized caches and checkpoints. This is the
single read-only loader used by notebooks 07b and 10b while a runner is active.
"""
function load_sax_regularized_study(
        raw_model::NamedTuple,
        project_root::AbstractString;
        fingering::AbstractString="Dx4",
        eta::Real=1e-3,
        profile::Symbol=:final,
        run_id::Union{Nothing,AbstractString}=nothing)
    paths = sax_regularized_study_paths(
        project_root;
        fingering=fingering,
        eta=eta,
        profile=profile,
        run_id=run_id,
    )
    model = regularize_sax_model(raw_model; eta=eta)
    main_settings = sax_regularized_bifurcation_settings(profile)
    main = load_sax_bifurcation_cache(
        paths.main,
        model;
        fingering=fingering,
        settings=main_settings,
    )
    stages = sax_bifurcation_stage_cache_status(
        paths.stages, model; settings=main_settings)
    focused_settings = sax_dense_pqz_cached_settings(
        paths.dense_focused;
        fallback=_sax_regularized_dense_settings(profile, :focused),
    )
    full_settings = sax_dense_pqz_cached_settings(
        paths.dense_full;
        fallback=_sax_regularized_dense_settings(profile, :full),
    )
    extended_settings = sax_dense_pqz_cached_settings(
        paths.dense_extended;
        fallback=_sax_regularized_dense_settings(profile, :extended),
    )
    focused = load_sax_dense_pqz_ns_progress(
        model, paths.dense_focused; settings=focused_settings)
    full = load_sax_dense_pqz_ns_progress(
        model, paths.dense_full; settings=full_settings)
    extended = load_sax_dense_pqz_ns_progress(
        model, paths.dense_extended; settings=extended_settings)
    dense = merge_sax_dense_pqz_ns_progress(
        merge_sax_dense_pqz_ns_progress(focused, full),
        extended,
    )
    p2_settings = sax_regularized_p2_settings(
        profile; reference_eta=1e-3)
    p2_r2 = load_sax_regularized_p2_progress(
        model, paths.root; settings=p2_settings)
    fixed_zeta_amplitude_settings =
        sax_fixed_zeta_amplitude_settings(profile)
    fixed_zeta_amplitude = load_sax_fixed_zeta_amplitude_progress(
        model,
        paths.root;
        settings=fixed_zeta_amplitude_settings,
    )
    fixed_zeta_schur_settings = sax_fixed_zeta_schur_settings(profile)
    fixed_zeta_schur = load_sax_fixed_zeta_schur_progress(
        model,
        paths.root;
        settings=fixed_zeta_schur_settings,
    )
    plane_completion_settings =
        sax_regularized_plane_completion_settings(profile)
    plane_completion = load_sax_regularized_plane_completion_progress(
        model,
        paths.root;
        settings=plane_completion_settings,
        stability_settings=fixed_zeta_schur_settings,
    )
    fold_completion_settings =
        sax_regularized_fold_completion_settings(profile)
    fold_completion = load_sax_regularized_fold_completion_progress(
        model,
        paths.root;
        settings=fold_completion_settings,
    )
    complete_product = if isdefined(
            @__MODULE__, :load_sax_regularized_complete_product)
        load_sax_regularized_complete_product(
            model,
            paths;
            fingering=fingering,
            profile=profile,
            reference_eta=1e-3,
        )
    else
        (
            status=:unavailable,
            product=nothing,
            reason="canonical product module is not loaded",
            path=paths.complete_product,
        )
    end
    manifest = if isfile(paths.manifest)
        try
            JLD2.load(paths.manifest, "manifest")
        catch err
            (status=:corrupt, error=sprint(showerror, err))
        end
    else
        (status=:missing,)
    end
    return (
        paths=paths,
        model=model,
        regularization=model.sax_regularization,
        main_settings=main_settings,
        main=main,
        stages=stages,
        focused_settings=focused_settings,
        full_settings=full_settings,
        extended_settings=extended_settings,
        focused=focused,
        full=full,
        extended=extended,
        dense=dense,
        p2_settings=p2_settings,
        p2_r2=p2_r2,
        fixed_zeta_amplitude_settings=fixed_zeta_amplitude_settings,
        fixed_zeta_amplitude=fixed_zeta_amplitude,
        fixed_zeta_schur_settings=fixed_zeta_schur_settings,
        fixed_zeta_schur=fixed_zeta_schur,
        plane_completion_settings=plane_completion_settings,
        plane_completion=plane_completion,
        fold_completion_settings=fold_completion_settings,
        fold_completion=fold_completion,
        complete_product=complete_product,
        manifest=manifest,
    )
end

function _sax_regularized_curve_at_zeta(curve, zeta::Real)
    isempty(curve.gamma) && return nothing
    indices = [index for index in eachindex(curve.gamma, curve.zeta)
               if isfinite(float(curve.gamma[index])) &&
                  isfinite(float(curve.zeta[index]))]
    isempty(indices) && return nothing
    index = indices[argmin(abs(float(curve.zeta[index]) - float(zeta))
                           for index in indices)]
    return (gamma=float(curve.gamma[index]), zeta=float(curve.zeta[index]))
end

function _sax_merge_regularized_fold_curves(base, completion;
                                            seed_zeta::Real=0.6)
    merged = Any[base...]
    for candidate in completion
        candidate_point = _sax_regularized_curve_at_zeta(
            candidate, seed_zeta)
        duplicate = !isnothing(candidate_point) && any(merged) do stored
            same_kind = !hasproperty(candidate, :kind) ||
                !hasproperty(stored, :kind) || candidate.kind == stored.kind
            same_mode = !hasproperty(candidate, :mode) ||
                !hasproperty(stored, :mode) || candidate.mode == stored.mode
            stored_point = _sax_regularized_curve_at_zeta(stored, seed_zeta)
            same_kind && same_mode && !isnothing(stored_point) &&
                hypot(stored_point.gamma - candidate_point.gamma,
                      stored_point.zeta - candidate_point.zeta) <= 2e-3
        end
        duplicate || push!(merged, candidate)
    end
    return merged
end

function _sax_regularized_complete_product_overlay(study, assembled)
    hasproperty(study, :complete_product) || return assembled
    loaded = study.complete_product
    loaded.status == :valid || return assembled
    isnothing(assembled.result) && return assembled
    product = loaded.product
    # A committed product can lag behind a currently running component stage.
    # Do not let an older partial product hide newer checkpointed curves.
    if product.status != :complete && (
            (hasproperty(study, :plane_completion) &&
             !isempty(study.plane_completion.curves)) ||
            (hasproperty(study, :fold_completion) &&
             !isempty(study.fold_completion.curves)))
        return assembled
    end
    plane = product.plane
    base = assembled.result
    counts = hasproperty(base, :counts) ? merge(base.counts, (
        hopf_curves=length(plane.hopf_curves),
        double_hopf_points=length(plane.double_hopf_points),
        generalized_hopf_points=length(plane.generalized_hopf_points),
        fold_curves=length(plane.fold_curves),
        pd_curves=length(plane.pd_curves),
        ns_curves=length(plane.ns_curves),
    )) : product.counts
    metadata = hasproperty(base, :metadata) ? merge(base.metadata, (
        canonical_product=true,
        canonical_status=product.status,
        duplicate_aware=true,
    )) : (
        canonical_product=true,
        canonical_status=product.status,
        duplicate_aware=true,
    )
    result = merge(base, (
        settings=plane.settings,
        counts=counts,
        hopf_curves=plane.hopf_curves,
        double_hopf_points=plane.double_hopf_points,
        generalized_hopf_points=plane.generalized_hopf_points,
        fold_curves=plane.fold_curves,
        pd_curves=plane.pd_curves,
        ns_curves=plane.ns_curves,
        metadata=metadata,
    ))
    return (
        source=product.status == :complete ?
            :canonical_complete : :canonical_partial,
        result=result,
    )
end

"""
    sax_regularized_partial_bifurcation(study)

Return the completed portable main result when available, otherwise assemble a
read-only plotting result from compatible Hopf, periodic-orbit, and
two-parameter curve stage checkpoints. Localized fold, PD, and NS points from
the periodic stage are exposed as `periodic_bifurcation_checkpoints`; they are
evidence and continuation seeds, not completed two-parameter curves. The return
field `source` distinguishes `:complete`, `:checkpoint`, and `:missing` without
promoting partial results to completed validation.
"""
function sax_regularized_partial_bifurcation(study)
    fold_completion = hasproperty(study, :fold_completion) ?
        study.fold_completion : (
            curves=Any[],
            settings=(seed_zeta=0.6,),
        )
    plane_completion = hasproperty(study, :plane_completion) ?
        study.plane_completion : (
            curves=Any[],
            periodic_codim2_checkpoints=Any[],
            settings=(seed_zeta=0.6,),
        )
    plane_codim2 = hasproperty(
            plane_completion, :periodic_codim2_checkpoints) ?
        plane_completion.periodic_codim2_checkpoints : Any[]
    hopf = _load_sax_stage_cache(
        _sax_stage_cache_path(study.paths.stages, :hopf),
        :hopf,
        study.model,
        study.main_settings,
    )
    periodic = _load_sax_stage_cache(
        _sax_stage_cache_path(study.paths.stages, :periodic),
        :periodic,
        study.model,
        study.main_settings,
    )
    curves = _load_sax_stage_cache(
        _sax_stage_cache_path(study.paths.stages, :curves),
        :curves,
        study.model,
        study.main_settings,
    )
    hopf_payload = hopf.status == :valid ? hopf.payload : nothing
    periodic_payload = periodic.status == :valid ? periodic.payload : nothing
    curve_payload = curves.status == :valid ? curves.payload : nothing
    periodic_checkpoints = isnothing(periodic_payload) ? Any[] :
        Any[periodic_payload.periodic_checkpoints...]
    periodic_diagnostics = isnothing(periodic_payload) ? Any[] :
        Any[periodic_payload.periodic_branch_diagnostics...]
    if study.main.status == :valid
        base = study.main.result
        isnothing(periodic_payload) && isnothing(curve_payload) &&
            isempty(fold_completion.curves) &&
            isempty(plane_completion.curves) && return
                _sax_regularized_complete_product_overlay(study, (
                    source=:complete,
                    result=base,
                ))
        # An equilibrium-only portable main cache is already valid while the
        # restartable periodic stages are still running. Prefer each newly
        # committed curve-stage payload so Pluto can monitor PD/NS progress
        # without waiting for the final portable cache to be rewritten.
        fold_curves = isnothing(curve_payload) ? Any[base.fold_curves...] :
            Any[curve_payload.fold_curves...]
        fold_curves = _sax_merge_regularized_fold_curves(
            fold_curves, fold_completion.curves;
            seed_zeta=fold_completion.settings.seed_zeta)
        pd_curves = isnothing(curve_payload) ? Any[base.pd_curves...] :
            Any[curve_payload.pd_curves...]
        pd_curves = _sax_merge_regularized_fold_curves(
            pd_curves,
            [curve for curve in plane_completion.curves
             if curve.kind == :pd];
            seed_zeta=plane_completion.settings.seed_zeta)
        ns_curves = isnothing(curve_payload) ? Any[base.ns_curves...] :
            Any[curve_payload.ns_curves...]
        ns_curves = _sax_merge_regularized_fold_curves(
            ns_curves,
            [curve for curve in plane_completion.curves
            if curve.kind == :ns];
            seed_zeta=plane_completion.settings.seed_zeta)
        periodic_codim2 = isnothing(curve_payload) ||
                !hasproperty(curve_payload, :periodic_codim2_checkpoints) ?
            (hasproperty(base, :periodic_codim2_checkpoints) ?
                Any[base.periodic_codim2_checkpoints...] : Any[]) :
            Any[curve_payload.periodic_codim2_checkpoints...]
        for candidate in plane_codim2
            _push_unique_periodic_codim2_checkpoint!(
                periodic_codim2, candidate)
        end
        failures = Any[base.failures...]
        isnothing(periodic_payload) || append!(failures, periodic_payload.failures)
        isnothing(curve_payload) || append!(failures, curve_payload.failures)
        periodic_branch_count = isnothing(periodic_payload) ?
            (hasproperty(base.counts, :periodic_branches) ?
                base.counts.periodic_branches : 0) :
            Int(periodic_payload.periodic_branch_count)
        result = merge(base, (
            counts=merge(base.counts, (
                periodic_branches=periodic_branch_count,
                periodic_bifurcation_checkpoints=length(periodic_checkpoints),
                fold_curves=length(fold_curves),
                pd_curves=length(pd_curves),
                ns_curves=length(ns_curves),
                failures=length(failures),
            )),
            fold_curves=fold_curves,
            pd_curves=pd_curves,
            ns_curves=ns_curves,
            periodic_bifurcation_checkpoints=periodic_checkpoints,
            periodic_branch_diagnostics=isnothing(periodic_payload) ?
                (hasproperty(base, :periodic_branch_diagnostics) ?
                    base.periodic_branch_diagnostics : Any[]) :
                periodic_diagnostics,
            periodic_codim2_checkpoints=periodic_codim2,
            failures=failures,
            metadata=merge(base.metadata, (
                partial=true,
                periodic_checkpoint_overlay=!isnothing(periodic_payload),
                curve_checkpoint_overlay=!isnothing(curve_payload),
                fold_completion_overlay=
                    !isempty(fold_completion.curves),
                plane_completion_overlay=
                    !isempty(plane_completion.curves),
            )),
        ))
        return _sax_regularized_complete_product_overlay(
            study, (source=:checkpoint, result=result))
    end
    if isnothing(hopf_payload) && isnothing(periodic_payload) &&
            isnothing(curve_payload) && isempty(fold_completion.curves) &&
            isempty(plane_completion.curves)
        return (source=:missing, result=nothing)
    end
    failures = Any[]
    isnothing(hopf_payload) || append!(failures, hopf_payload.failures)
    isnothing(periodic_payload) || append!(failures, periodic_payload.failures)
    isnothing(curve_payload) || append!(failures, curve_payload.failures)
    fallback_fold_curves = _sax_merge_regularized_fold_curves(
        isnothing(curve_payload) ? Any[] :
            Any[curve_payload.fold_curves...],
        fold_completion.curves;
        seed_zeta=fold_completion.settings.seed_zeta)
    fallback_pd_curves = _sax_merge_regularized_fold_curves(
        isnothing(curve_payload) ? Any[] : Any[curve_payload.pd_curves...],
        [curve for curve in plane_completion.curves if curve.kind == :pd];
        seed_zeta=plane_completion.settings.seed_zeta)
    fallback_ns_curves = _sax_merge_regularized_fold_curves(
        isnothing(curve_payload) ? Any[] : Any[curve_payload.ns_curves...],
        [curve for curve in plane_completion.curves if curve.kind == :ns];
        seed_zeta=plane_completion.settings.seed_zeta)
    fallback_codim2 = isnothing(curve_payload) ||
            !hasproperty(curve_payload, :periodic_codim2_checkpoints) ?
        Any[] : Any[curve_payload.periodic_codim2_checkpoints...]
    for candidate in plane_codim2
        _push_unique_periodic_codim2_checkpoint!(fallback_codim2, candidate)
    end
    result = (
        settings=_portable_sax_bifurcation_settings(study.main_settings),
        counts=(
            equilibrium_runs=isnothing(hopf_payload) ? 0 :
                hopf_payload.equilibrium_run_count,
            hopf_curves=isnothing(hopf_payload) ? 0 :
                length(hopf_payload.hopf_curves),
            double_hopf_points=isnothing(hopf_payload) ? 0 :
                length(hopf_payload.double_hopf_points),
            generalized_hopf_points=isnothing(hopf_payload) ? 0 :
                length(hopf_payload.generalized_hopf_points),
            periodic_branches=isnothing(periodic_payload) ? 0 :
                Int(periodic_payload.periodic_branch_count),
            periodic_bifurcation_checkpoints=length(periodic_checkpoints),
            fold_curves=length(fallback_fold_curves),
            pd_curves=length(fallback_pd_curves),
            ns_curves=length(fallback_ns_curves),
            failures=length(failures),
        ),
        hopf_curves=isnothing(hopf_payload) ? Any[] :
            Any[hopf_payload.hopf_curves...],
        double_hopf_points=isnothing(hopf_payload) ? Any[] :
            Any[hopf_payload.double_hopf_points...],
        generalized_hopf_points=isnothing(hopf_payload) ? Any[] :
            Any[hopf_payload.generalized_hopf_points...],
        periodic_bifurcation_checkpoints=periodic_checkpoints,
        periodic_branch_diagnostics=periodic_diagnostics,
        periodic_codim2_checkpoints=fallback_codim2,
        fold_curves=fallback_fold_curves,
        pd_curves=fallback_pd_curves,
        ns_curves=fallback_ns_curves,
        failures=failures,
        metadata=(
            regularized=true,
            partial=true,
            jacobian=:analytic_regularized,
            fold_completion_overlay=!isempty(fold_completion.curves),
            plane_completion_overlay=!isempty(plane_completion.curves),
        ),
    )
    return _sax_regularized_complete_product_overlay(
        study, (source=:checkpoint, result=result))
end

function _sax_portable_curve_points(curves)
    points = Tuple{Float64,Float64}[]
    for curve in curves
        append!(points, zip(float.(curve.gamma), float.(curve.zeta)))
    end
    return points
end

function _sax_directed_curve_distance(first_points, second_points)
    (isempty(first_points) || isempty(second_points)) &&
        return (mean=NaN, maximum=NaN)
    distances = map(first_points) do point
        minimum(hypot(point[1] - other[1], point[2] - other[2])
                for other in second_points)
    end
    return (mean=mean(distances), maximum=maximum(distances))
end

"""
    sax_regularized_curve_robustness(piecewise, regularized)

Compute symmetric nearest-point distances between like-named portable curve
families. These are descriptive sensitivity measures, not certified branch
correspondences. Missing families return `available=false`.
"""
function sax_regularized_curve_robustness(piecewise, regularized)
    families = (
        hopf=:hopf_curves,
        fold=:fold_curves,
        pd=:pd_curves,
        ns=:ns_curves,
    )
    # `pairs(::NamedTuple)` is dictionary-like in Julia 1.12 and deliberately
    # does not implement `map`. A comprehension preserves the small ordered
    # family table without depending on dictionary mapping semantics.
    rows = [begin
        first_points = _sax_portable_curve_points(getproperty(piecewise, field))
        second_points = _sax_portable_curve_points(getproperty(regularized, field))
        if isempty(first_points) || isempty(second_points)
            (
                family=name,
                available=false,
                piecewise_points=length(first_points),
                regularized_points=length(second_points),
                mean_distance=NaN,
                hausdorff_distance=NaN,
            )
        else
            forward = _sax_directed_curve_distance(first_points, second_points)
            backward = _sax_directed_curve_distance(second_points, first_points)
            (
                family=name,
                available=true,
                piecewise_points=length(first_points),
                regularized_points=length(second_points),
                mean_distance=(forward.mean + backward.mean) / 2,
                hausdorff_distance=max(forward.maximum, backward.maximum),
            )
        end
    end for (name, field) in pairs(families)]
    return rows
end
