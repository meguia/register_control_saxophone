# Canonical, duplicate-aware assembly of the regularized bifurcation study.
#
# Numerical continuation remains in the focused analysis files.  This layer is
# deliberately small: it accepts only compatible caches, removes repeated
# continuations of the same geometric component, records the Periodic-Schur
# evidence behind every periodic-orbit claim, and writes one portable product
# for notebooks and reproducibility audits.

const SAX_REGULARIZED_COMPLETE_PRODUCT_SCHEMA_VERSION = 4

function sax_regularized_complete_product_path(root::AbstractString)
    return joinpath(root, "complete_bifurcation_product.jld2")
end

function _sax_complete_product_signature(
        model_p::NamedTuple;
        fingering::AbstractString,
        run_id,
        profile::Symbol,
        coverages::Tuple,
        zeta_step,
        reference_eta::Real)
    return (
        fingering=String(fingering),
        run_id=isnothing(run_id) ? nothing : String(run_id),
        profile=profile,
        coverages=Tuple(Symbol.(coverages)),
        zeta_step=isnothing(zeta_step) ? nothing : float(zeta_step),
        reference_eta=float(reference_eta),
        regularization=hasproperty(model_p, :sax_regularization) ?
            model_p.sax_regularization : nothing,
        model=_sax_bifurcation_model_signature(model_p, 8),
    )
end

function _sax_complete_curve_points(curve; maximum_points::Int=240)
    points = Tuple{Float64,Float64}[]
    for index in eachindex(curve.gamma, curve.zeta)
        gamma = float(curve.gamma[index])
        zeta = float(curve.zeta[index])
        isfinite(gamma) && isfinite(zeta) && push!(points, (gamma, zeta))
    end
    length(points) <= maximum_points && return points
    indices = unique(round.(Int, range(
        1, length(points); length=maximum_points)))
    return points[indices]
end

function _sax_complete_curve_score(curve)
    points = _sax_complete_curve_points(curve; maximum_points=10_000)
    isempty(points) && return -Inf
    gamma_span = maximum(first, points) - minimum(first, points)
    zeta_span = maximum(last, points) - minimum(last, points)
    return length(points) + 200gamma_span + 200zeta_span
end

function _sax_complete_curve_duplicate(
        first_curve,
        second_curve;
        tolerance::Real)
    if hasproperty(first_curve, :mode) && hasproperty(second_curve, :mode)
        Int(first_curve.mode) == Int(second_curve.mode) || return false
    end
    first_points = _sax_complete_curve_points(first_curve)
    second_points = _sax_complete_curve_points(second_curve)
    (isempty(first_points) || isempty(second_points)) && return false
    short, long = length(first_points) <= length(second_points) ?
        (first_points, second_points) : (second_points, first_points)
    short_zeta = extrema(last, short)
    long_zeta = extrema(last, long)
    short_span = short_zeta[2] - short_zeta[1]
    overlap = max(0.0,
        min(short_zeta[2], long_zeta[2]) -
        max(short_zeta[1], long_zeta[1]))
    (short_span <= 1e-4 || overlap >= 0.60short_span) || return false
    distances = [minimum(hypot(point[1] - other[1], point[2] - other[2])
                         for other in long) for point in short]
    return median(distances) <= float(tolerance) &&
        quantile(distances, 0.90) <= 2float(tolerance)
end

function _sax_complete_deduplicate_curves(
        curves;
        tolerance::Real)
    ordered = sort(collect(curves);
                   by=_sax_complete_curve_score, rev=true)
    retained = Any[]
    duplicates = Any[]
    for curve in ordered
        index = findfirst(stored -> _sax_complete_curve_duplicate(
            curve, stored; tolerance=tolerance), retained)
        if isnothing(index)
            push!(retained, curve)
        else
            push!(duplicates, (
                mode=hasproperty(curve, :mode) ? Int(curve.mode) : 0,
                discarded_source=hasproperty(curve, :source) ?
                    curve.source : nothing,
                retained_source=hasproperty(retained[index], :source) ?
                    retained[index].source : nothing,
            ))
        end
    end
    sort!(retained; by=curve -> (
        hasproperty(curve, :mode) ? Int(curve.mode) : 0,
        isempty(curve.zeta) ? Inf : minimum(float.(curve.zeta)),
    ))
    return (curves=retained, duplicates=duplicates)
end

function _sax_complete_hopf_curves(curves)
    retained = Any[]
    duplicates = Any[]
    modes = sort(unique(Int(curve.mode) for curve in curves
                        if hasproperty(curve, :mode) && curve.mode > 0))
    for mode in modes
        candidates = [curve for curve in curves if Int(curve.mode) == mode]
        ordered = sort(candidates; by=_sax_complete_curve_score, rev=true)
        isempty(ordered) && continue
        push!(retained, first(ordered))
        append!(duplicates, (
            mode=mode,
            discarded_source=hasproperty(curve, :source) ? curve.source : nothing,
            retained_source=hasproperty(first(ordered), :source) ?
                first(ordered).source : nothing,
        ) for curve in Iterators.drop(ordered, 1))
    end
    return (curves=retained, duplicates=duplicates)
end

function _sax_complete_deduplicate_points(
        points;
        tolerance::Real=2e-3,
        group=point -> nothing,
        coordinates=point -> (float(point.gamma), float(point.zeta)))
    retained = Any[]
    for point in points
        xy = coordinates(point)
        duplicate = any(retained) do stored
            group(stored) == group(point) &&
                hypot(coordinates(stored)[1] - xy[1],
                      coordinates(stored)[2] - xy[2]) <= tolerance
        end
        duplicate || push!(retained, point)
    end
    return retained
end

function _sax_complete_dense_evidence(progress, coverage::Symbol)
    hasproperty(progress, :slices) || return Any[]
    hasproperty(progress.slices, :roots) || return Any[]
    rows = Any[]
    for root in progress.slices.roots
        root.accepted || continue
        checkpoint = root.checkpoint
        bracket = root.source_bracket
        push!(rows, (
            family=root.near_r2 ? :near_r2 : :ns_like,
            gamma=float(checkpoint.gamma),
            zeta=float(checkpoint.zeta),
            mode=Int(checkpoint.mode),
            floquet_angle=float(checkpoint.floquet_angle),
            angle_to_pi=abs(pi - abs(float(checkpoint.floquet_angle))),
            gamma_error=hasproperty(bracket, :gamma_error) ?
                float(bracket.gamma_error) : NaN,
            periodic_schur=true,
            floquet_coll_crosscheck=root.validation.methods_agree,
            mesh_converged=root.mesh_converged,
            source=coverage,
        ))
    end
    return rows
end

function _sax_complete_r2_candidates(raw_plane)
    hasproperty(raw_plane, :periodic_codim2_checkpoints) || return Any[]
    candidates = Any[]
    for checkpoint in raw_plane.periodic_codim2_checkpoints
        hasproperty(checkpoint, :resonance_type) || continue
        checkpoint.resonance_type == :R2 || continue
        hasproperty(checkpoint, :r2_gamma) || continue
        hasproperty(checkpoint, :r2_zeta) || continue
        candidate = (
            type=:R2,
            mode=Int(checkpoint.mode),
            gamma=float(checkpoint.r2_gamma),
            zeta=float(checkpoint.r2_zeta),
            status=:augmented_pd_candidate,
            periodic_schur_validated=false,
            source=:pd_curve_special_point,
        )
        duplicate = any(stored ->
            stored.mode == candidate.mode &&
            hypot(stored.gamma - candidate.gamma,
                  stored.zeta - candidate.zeta) <= 5e-3,
            candidates,
        )
        duplicate || push!(candidates, candidate)
    end
    sort!(candidates; by=point -> (point.zeta, point.gamma))
    return candidates
end

function _sax_complete_fixed_slice_r2_evidence(plane_completion)
    hasproperty(plane_completion, :source_events) || return Any[]
    points = Any[]
    for event in plane_completion.source_events
        event.accepted && event.type == :r2 || continue
        push!(points, (
            family=:near_r2,
            mode=Int(event.mode),
            gamma=float(event.gamma),
            zeta=float(event.zeta),
            floquet_angle=float(event.floquet_angle),
            angle_to_pi=abs(pi - abs(float(event.floquet_angle))),
            gamma_error=NaN,
            periodic_schur=true,
            floquet_coll_crosscheck=!isnothing(event.validation) &&
                event.validation.methods_agree,
            mesh_converged=true,
            status=:dual_floquet_validated_slice_root,
            validation=event.validation,
            source=:supplemental_fixed_zeta_periodic_schur,
        ))
    end
    return _sax_complete_deduplicate_points(
        points; tolerance=2e-3, group=point -> point.mode)
end

function _sax_complete_fold_validations(fold_progress)
    return [(
        component=component.key,
        accepted=component.status == :complete &&
            !isnothing(component.validation) &&
            hasproperty(component.validation, :accepted) &&
            component.validation.accepted,
        validation_status=isnothing(component.validation) ? :missing :
            hasproperty(component.validation, :validation_status) ?
                component.validation.validation_status : :unknown,
    ) for component in fold_progress.components]
end

function _sax_complete_fold_domain_coverage(plane)
    covered = 0
    for curve in plane.fold_curves
        zeta = [float(value) for value in curve.zeta if isfinite(float(value))]
        isempty(zeta) && continue
        if minimum(zeta) <= 0.10 && maximum(zeta) >= 0.98
            covered += 1
        end
    end
    return (covered=covered, expected=2, accepted=covered >= 2)
end

function _sax_complete_ns_coverage(plane)
    finite_points(curve) = [
        (float(curve.gamma[index]), float(curve.zeta[index]))
        for index in eachindex(curve.gamma, curve.zeta)
        if isfinite(float(curve.gamma[index])) &&
           isfinite(float(curve.zeta[index]))
    ]
    dh_index = findfirst(plane.double_hopf_points) do point
        hasproperty(point, :modes) &&
            Set(Int.(point.modes)) == Set((1, 2))
    end
    dh = isnothing(dh_index) ? nothing : plane.double_hopf_points[dh_index]
    distance(curve) = isnothing(dh) ? Inf : begin
        points = finite_points(curve)
        isempty(points) ? Inf : minimum(
            hypot(point[1] - float(dh.gamma),
                  point[2] - float(dh.zeta)) for point in points)
    end
    mode_curves(mode) = [curve for curve in plane.ns_curves
                         if Int(curve.mode) == mode]
    zeta_limits(curve) = begin
        points = finite_points(curve)
        isempty(points) ? (Inf, -Inf) : extrema(last, points)
    end
    top = hasproperty(plane.settings, :zeta_range) ?
        float(plane.settings.zeta_range[2]) : 0.99
    mode1 = mode_curves(1)
    mode2 = mode_curves(2)
    return (
        dh_12_available=!isnothing(dh),
        dh_mode1_arm=!isnothing(dh) &&
            any(curve -> distance(curve) <= 1e-2, mode1),
        dh_mode2_arm=!isnothing(dh) &&
            any(curve -> distance(curve) <= 1e-2, mode2),
        mode1_reaches_top=any(curve ->
            zeta_limits(curve)[2] >= top - 2e-2, mode1),
        mode1_dh_to_top=any(curve ->
            distance(curve) <= 1e-2 &&
            zeta_limits(curve)[2] >= top - 2e-2, mode1),
        mode2_lower_arm=any(curve ->
            distance(curve) <= 1e-2 && zeta_limits(curve)[1] <= 0.22,
            mode2),
        mode2_high_component=any(curve ->
            zeta_limits(curve)[1] <= 0.20 &&
            zeta_limits(curve)[2] >= top - 2e-2,
            mode2),
    )
end

function _sax_complete_checks(
        plane,
        plane_completion,
        stability,
        amplitude,
        folds,
        p2,
        fixed_slice_r2,
        dense_evidence,
        dense_status)
    hopf_modes = sort(unique(Int(curve.mode) for curve in plane.hopf_curves))
    accepted_events = [event for event in stability.events if event.accepted]
    fold_validations = _sax_complete_fold_validations(folds)
    fold_coverage = _sax_complete_fold_domain_coverage(plane)
    ns_coverage = _sax_complete_ns_coverage(plane)
    checks = (
        hopf_modes=all(mode -> mode in hopf_modes, (1, 2, 3)),
        double_hopf=!isempty(plane.double_hopf_points),
        generalized_hopf=!isempty(plane.generalized_hopf_points),
        fold_curves=length(plane.fold_curves) >= 2,
        fold_domain_coverage=fold_coverage.accepted,
        fold_periodic_schur=!isempty(fold_validations) &&
            all(row -> row.accepted, fold_validations),
        period_doubling=!isempty(plane.pd_curves) &&
            any(event -> event.type in (:pd, :r2), accepted_events),
        neimark_sacker=all(requirement -> count(
                curve -> Int(curve.mode) == requirement[1],
                plane.ns_curves) >= requirement[2],
            ((1, 1), (2, 2), (3, 2))) &&
            any(event -> event.type == :ns, accepted_events),
        double_hopf_ns_arms=ns_coverage.dh_12_available &&
            ns_coverage.dh_mode1_arm && ns_coverage.dh_mode2_arm,
        ns_domain_coverage=ns_coverage.mode1_dh_to_top &&
            ns_coverage.mode2_lower_arm &&
            ns_coverage.mode2_high_component,
        plane_seed_audits=plane_completion.status == :complete,
        resonance_1_2=!isempty(fixed_slice_r2) ||
            any(row -> row.family == :near_r2, dense_evidence),
        dense_coverages=!isempty(dense_status) &&
            all(row -> row.status == :complete, dense_status),
        fixed_zeta_pqz=stability.status == :complete,
        fixed_zeta_amplitude=amplitude.status == :complete &&
            amplitude.settings.floquet_solver == :periodic_schur,
        p2_periodic_schur=p2.status == :complete &&
            any(component -> component.status == :complete &&
                component.analysis.total_samples > 0, p2.components),
    )
    return (
        passed=all(values(checks)),
        checks=checks,
        missing=Symbol[name for (name, valid) in pairs(checks) if !valid],
        fold_validations=fold_validations,
        fold_coverage=fold_coverage,
        ns_coverage=ns_coverage,
    )
end

"""
    compute_sax_regularized_complete_product(model, paths; ...)

Assemble one portable, duplicate-aware result from the canonical regularized
pipeline caches. The function performs no continuation. It records strict
Periodic-Schur evidence separately from BifurcationKit's independent
FloquetColl cross-check and never promotes an evidence marker by itself. A
near-R2 root appears as a solid NS curve only if the upstream plane stage
explicitly selected its narrow resonant window and completed the minimally
augmented two-parameter continuation.
"""
function compute_sax_regularized_complete_product(
        model_p::NamedTuple,
        paths;
        fingering::AbstractString="Dx4",
        profile::Symbol=:final,
        coverages::Tuple=(:focused, :full, :extended),
        zeta_step=nothing,
        reference_eta::Real=1e-3)
    main_settings = sax_regularized_bifurcation_settings(profile)
    main = load_sax_bifurcation_cache(
        paths.main, model_p;
        fingering=fingering,
        settings=main_settings,
    )
    folds = load_sax_regularized_fold_completion_progress(
        model_p, paths.root;
        settings=sax_regularized_fold_completion_settings(profile),
    )
    stability_settings = sax_fixed_zeta_schur_settings(profile)
    plane_completion = load_sax_regularized_plane_completion_progress(
        model_p, paths.root;
        settings=sax_regularized_plane_completion_settings(profile),
        stability_settings=stability_settings,
    )
    study = (
        paths=paths,
        model=model_p,
        main_settings=main_settings,
        main=main,
        fold_completion=folds,
        plane_completion=plane_completion,
    )
    partial = sax_regularized_partial_bifurcation(study)
    isnothing(partial.result) && error(
        "no compatible regularized main or stage cache is available",
    )
    raw_plane = partial.result
    hopf = _sax_complete_hopf_curves(raw_plane.hopf_curves)
    fold = _sax_complete_deduplicate_curves(
        raw_plane.fold_curves; tolerance=4e-3)
    pd = _sax_complete_deduplicate_curves(
        raw_plane.pd_curves; tolerance=3e-3)
    # Four-point NS objects are failed continuation fragments, not curves.
    # Keep their provenance in the audit but do not present them as completed
    # geometry in the canonical diagram.
    ns_fragments = [curve for curve in raw_plane.ns_curves
                    if length(curve.gamma) < 12]
    ns = _sax_complete_deduplicate_curves(
        [curve for curve in raw_plane.ns_curves
         if length(curve.gamma) >= 12];
        tolerance=7e-3,
    )
    double_hopf = _sax_complete_deduplicate_points(
        [point for point in raw_plane.double_hopf_points if point.valid];
        tolerance=2e-3,
        group=point -> Tuple(sort(collect(point.modes))),
    )
    generalized_hopf = _sax_complete_deduplicate_points(
        [point for point in raw_plane.generalized_hopf_points if point.valid];
        tolerance=2e-3,
    )
    plane = (
        settings=raw_plane.settings,
        hopf_curves=hopf.curves,
        fold_curves=fold.curves,
        pd_curves=pd.curves,
        ns_curves=ns.curves,
        double_hopf_points=double_hopf,
        generalized_hopf_points=generalized_hopf,
    )

    stability = load_sax_fixed_zeta_schur_progress(
        model_p, paths.root;
        settings=stability_settings,
    )
    amplitude = load_sax_fixed_zeta_amplitude_progress(
        model_p, paths.root;
        settings=sax_fixed_zeta_amplitude_settings(profile),
    )
    unique_schur = sax_fixed_zeta_unique_schur_components(stability)
    p2 = load_sax_regularized_p2_progress(
        model_p, paths.root;
        settings=sax_regularized_p2_settings(
            profile; reference_eta=reference_eta),
    )

    dense_rows = Any[]
    dense_status = Any[]
    for coverage in coverages
        selected_paths = _sax_regularized_dense_paths(paths, coverage)
        settings = sax_dense_pqz_cached_settings(
            selected_paths.directory;
            fallback=_sax_regularized_dense_settings(
                profile, coverage; zeta_step=zeta_step),
        )
        progress = load_sax_dense_pqz_ns_progress(
            model_p, selected_paths.directory; settings=settings)
        append!(dense_rows, _sax_complete_dense_evidence(progress, coverage))
        push!(dense_status, (
            coverage=coverage,
            status=progress.slices.status,
            completed=progress.slices.completed,
            expected=progress.slices.expected,
            accepted=progress.slices.accepted_roots,
        ))
    end
    dense_evidence = _sax_complete_deduplicate_points(
        sort(dense_rows; by=row -> isfinite(row.gamma_error) ?
            row.gamma_error : Inf);
        tolerance=2e-3,
        group=row -> (row.family, row.mode),
    )
    r2_candidates = _sax_complete_r2_candidates(raw_plane)
    fixed_slice_r2 =
        _sax_complete_fixed_slice_r2_evidence(plane_completion)
    compatibility = sax_fixed_zeta_plane_compatibility(
        stability, plane;
        gamma_limits=(0.30, 0.99),
    )
    validation = _sax_complete_checks(
        plane, plane_completion, stability, amplitude, folds, p2,
        fixed_slice_r2, dense_evidence, dense_status)
    duplicate_report = (
        hopf=hopf.duplicates,
        fold=fold.duplicates,
        pd=pd.duplicates,
        ns=ns.duplicates,
        ns_fragments=[(
            mode=Int(curve.mode),
            points=length(curve.gamma),
            source=hasproperty(curve, :source) ? curve.source : nothing,
        ) for curve in ns_fragments],
        fixed_zeta_components=unique_schur.duplicates,
    )
    signature = _sax_complete_product_signature(
        model_p;
        fingering=fingering,
        run_id=paths.run_id,
        profile=profile,
        coverages=coverages,
        zeta_step=zeta_step,
        reference_eta=reference_eta,
    )
    product = (
        schema_version=SAX_REGULARIZED_COMPLETE_PRODUCT_SCHEMA_VERSION,
        analysis=:regularized_complete_bifurcation,
        status=validation.passed ? :complete : :partial,
        saved_at_unix=time(),
        signature=signature,
        plane=plane,
        fixed_zeta=(
            zeta=float(stability.settings.zeta),
            p1_components=unique_schur.components,
            p2_branches=amplitude.p2_branches,
            hopf_events=_sax_fixed_zeta_hopf_events(stability),
            periodic_events=stability.events,
            diagnostic_events=stability.diagnostic_events,
        ),
        resonance_1_2=(
            status=!isempty(r2_candidates) &&
                    (!isempty(fixed_slice_r2) ||
                     any(row -> row.family == :near_r2, dense_evidence)) ?
                :candidate_with_periodic_schur_neighbourhood :
                !isempty(r2_candidates) ? :augmented_candidate_only :
                !isempty(fixed_slice_r2) ||
                    any(row -> row.family == :near_r2, dense_evidence) ?
                    :periodic_schur_neighbourhood_only : :missing,
            exact_points=Any[],
            candidates=r2_candidates,
            pqz_evidence=_sax_complete_deduplicate_points(
                vcat(
                    Any[fixed_slice_r2...],
                    Any[row for row in dense_evidence
                        if row.family == :near_r2],
                );
                tolerance=2e-3,
                group=row -> (row.family, row.mode),
            ),
        ),
        ns_like_evidence=[row for row in dense_evidence
                          if row.family == :ns_like],
        p2_periodic_schur=(
            status=p2.status,
            curves=p2.p2_curves,
            ns_roots=p2.ns_roots,
        ),
        fixed_plane_compatibility=compatibility,
        validation=validation,
        duplicates=duplicate_report,
        counts=(
            raw_hopf=length(raw_plane.hopf_curves),
            hopf=length(plane.hopf_curves),
            raw_fold=length(raw_plane.fold_curves),
            fold=length(plane.fold_curves),
            raw_pd=length(raw_plane.pd_curves),
            pd=length(plane.pd_curves),
            raw_ns=length(raw_plane.ns_curves),
            ns=length(plane.ns_curves),
            dense_evidence=length(dense_evidence),
        ),
        methods=(
            equilibrium=:analytic_jacobian,
            periodic_orbits=:orthogonal_collocation,
            periodic_stability=:generalized_periodic_schur,
            periodic_stability_implementation=:SaxFloquetPQZ,
            decomposition_package=:PeriodicSchurDecompositions,
            floquet_coll=:independent_crosscheck_only,
            curve_continuation=:BifurcationKit_minimally_augmented,
            resonant_ns_seed=:explicit_low_gamma_near_r2_window,
            fold_predictor_fallback=:FloquetColl_localization_then_strict_PQZ,
        ),
        source_status=(
            plane=partial.source,
            plane_completion=plane_completion.status,
            stability=stability.status,
            amplitude=amplitude.status,
            folds=folds.status,
            p2=p2.status,
            dense=dense_status,
        ),
    )
    _atomic_jld2_save(
        sax_regularized_complete_product_path(paths.root); product)
    return product
end

function load_sax_regularized_complete_product(
        model_p::NamedTuple,
        paths;
        fingering::AbstractString="Dx4",
        profile::Symbol=:final,
        coverages::Tuple=(:focused, :full, :extended),
        zeta_step=nothing,
        reference_eta::Real=1e-3)
    path = sax_regularized_complete_product_path(paths.root)
    isfile(path) || return (
        status=:missing, product=nothing, reason="cache is absent", path=path)
    product = try
        JLD2.load(path, "product")
    catch err
        return (status=:corrupt, product=nothing,
                reason=sprint(showerror, err), path=path)
    end
    expected = _sax_complete_product_signature(
        model_p;
        fingering=fingering,
        run_id=paths.run_id,
        profile=profile,
        coverages=coverages,
        zeta_step=zeta_step,
        reference_eta=reference_eta,
    )
    product.schema_version == SAX_REGULARIZED_COMPLETE_PRODUCT_SCHEMA_VERSION ||
        return (status=:incompatible, product=nothing,
                reason="schema changed", path=path)
    isequal(product.signature, expected) || return (
        status=:incompatible, product=nothing,
        reason="settings or model changed", path=path)
    return (status=:valid, product=product, reason="", path=path)
end
