# Targeted completion of the two fold-of-periodic-orbits curves for the
# Colinot-regularized model.

const SAX_REGULARIZED_FOLD_COMPLETION_SCHEMA_VERSION = 2

Base.@kwdef struct SaxRegularizedFoldCompletionSettings
    schema_version::Int = SAX_REGULARIZED_FOLD_COMPLETION_SCHEMA_VERSION
    nmodes::Int = 8
    mode::Int = 1
    seed_zeta::Float64 = 0.6
    hopf_gamma_hint::Float64 = 0.40
    gamma_range::Tuple{Float64,Float64} = (0.32, 0.45)
    zeta_range::Tuple{Float64,Float64} = (0.001, 0.99)
    collocation_intervals::Int = 40
    collocation_degree::Int = 4
    po_ds::Float64 = 2e-4
    po_dsmax::Float64 = 1e-3
    po_max_steps::Int = 270
    curve_ds::Float64 = 5e-4
    curve_dsmax::Float64 = 4e-3
    curve_max_steps::Int = 900
    checkpoint_every::Int = 10
    newton_tol::Float64 = 1e-10
    stability_tol::Float64 = 1e-8
    turn_gamma_tolerance::Float64 = 5e-3
    minimum_pressure_l2::Float64 = 5e-2
    minimum_curve_points::Int = 30
    minimum_curve_zeta_span::Float64 = 0.10
    required_zeta_limits::Union{Nothing,Tuple{Float64,Float64}} = (0.10, 0.98)
    maximum_plus_one_multiplier_distance::Float64 = 2e-2
end

"""Resolution presets for the targeted regularized fold-curve completion."""
function sax_regularized_fold_completion_settings(
        profile::Symbol=:final; kwargs...)
    profile in (:smoke, :pilot, :final) || throw(ArgumentError(
        "profile must be :smoke, :pilot, or :final"))
    resolution = if profile == :smoke
        (
            zeta_range=(0.55, 0.65),
            collocation_intervals=8,
            collocation_degree=2,
            po_ds=1e-3,
            po_dsmax=4e-3,
            po_max_steps=100,
            curve_ds=1e-3,
            curve_dsmax=4e-3,
            curve_max_steps=20,
            newton_tol=1e-8,
            turn_gamma_tolerance=2e-2,
            minimum_curve_points=3,
            minimum_curve_zeta_span=1e-4,
            required_zeta_limits=nothing,
        )
    elseif profile == :pilot
        (
            collocation_intervals=25,
            collocation_degree=3,
            po_ds=5e-4,
            po_dsmax=2e-3,
            po_max_steps=180,
            curve_ds=8e-4,
            curve_dsmax=5e-3,
            curve_max_steps=400,
            required_zeta_limits=nothing,
        )
    else
        NamedTuple()
    end
    return _validate_sax_regularized_fold_completion_settings(
        SaxRegularizedFoldCompletionSettings(;
            merge(resolution, (; kwargs...))...))
end

function _validate_sax_regularized_fold_completion_settings(
        settings::SaxRegularizedFoldCompletionSettings)
    settings.schema_version == SAX_REGULARIZED_FOLD_COMPLETION_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported regularized fold-completion schema"))
    1 <= settings.mode <= settings.nmodes ||
        throw(ArgumentError("mode must lie inside 1:nmodes"))
    settings.gamma_range[1] < settings.gamma_range[2] ||
        throw(ArgumentError("gamma_range must be increasing"))
    settings.zeta_range[1] < settings.seed_zeta < settings.zeta_range[2] ||
        throw(ArgumentError("seed_zeta must lie inside zeta_range"))
    settings.collocation_intervals >= 5 ||
        throw(ArgumentError("at least five collocation intervals are required"))
    settings.collocation_degree >= 2 ||
        throw(ArgumentError("collocation_degree must be at least two"))
    settings.po_max_steps > 0 && settings.curve_max_steps > 0 ||
        throw(ArgumentError("continuation step limits must be positive"))
    settings.checkpoint_every > 0 ||
        throw(ArgumentError("checkpoint_every must be positive"))
    settings.minimum_curve_points >= 3 || throw(ArgumentError(
        "minimum_curve_points must be at least three"))
    settings.minimum_curve_zeta_span > 0 || throw(ArgumentError(
        "minimum_curve_zeta_span must be positive"))
    if !isnothing(settings.required_zeta_limits)
        lower, upper = settings.required_zeta_limits
        settings.zeta_range[1] <= lower < upper <= settings.zeta_range[2] ||
            throw(ArgumentError(
                "required_zeta_limits must lie inside zeta_range"))
    end
    settings.maximum_plus_one_multiplier_distance > 0 || throw(ArgumentError(
        "maximum_plus_one_multiplier_distance must be positive"))
    return settings
end

"""Cache paths for the two independently committed fold components."""
function sax_regularized_fold_completion_paths(
        output_root::AbstractString;
        settings::SaxRegularizedFoldCompletionSettings=
            SaxRegularizedFoldCompletionSettings())
    directory = joinpath(output_root, "fixed_zeta_fold_completion")
    return (
        directory=directory,
        manifest=joinpath(directory, "manifest.jld2"),
        component=kind -> joinpath(directory, "fold_$(kind).jld2"),
    )
end

function _sax_regularized_fold_bifurcation_settings(
        settings::SaxRegularizedFoldCompletionSettings)
    half_width = max(1e-6, settings.seed_zeta * 1e-6)
    return sax_bifurcation_settings(
        :final;
        nmodes=settings.nmodes,
        gamma_range=settings.gamma_range,
        zeta_range=(settings.seed_zeta - half_width,
                    settings.seed_zeta + half_width),
        zeta_seeds=(settings.seed_zeta,),
        po_collocation_intervals=settings.collocation_intervals,
        po_collocation_degree=settings.collocation_degree,
        po_linear_solver=:condensed,
        po_ds=settings.po_ds,
        po_dsmax=settings.po_dsmax,
        po_max_steps=settings.po_max_steps,
        po_save_sol_every_step=1,
        newton_tol=settings.newton_tol,
        stability_tol=settings.stability_tol,
    )
end

function _sax_regularized_fold_hopf_settings(
        settings::SaxRegularizedFoldCompletionSettings)
    schur = sax_fixed_zeta_schur_settings(
        :final;
        nmodes=settings.nmodes,
        zeta=settings.seed_zeta,
        modes=(settings.mode,),
        hopf_gamma_hints=(settings.hopf_gamma_hint,),
        gamma_range=settings.gamma_range,
        collocation_intervals=settings.collocation_intervals,
        collocation_degree=settings.collocation_degree,
        po_ds=settings.po_ds,
        po_dsmax=settings.po_dsmax,
        po_max_steps=settings.po_max_steps,
        newton_tol=settings.newton_tol,
        stability_tol=settings.stability_tol,
    )
    return _sax_fixed_zeta_schur_hopf_settings(schur)
end

function _sax_regularized_fold_turns(branch,
        settings::SaxRegularizedFoldCompletionSettings)
    rows = collect(branch.branch)
    candidates = Any[]
    length(rows) < 5 && return candidates
    for index in 3:(length(rows) - 2)
        left, center, right = rows[index - 2], rows[index], rows[index + 2]
        before = float(center.gamma - left.gamma)
        after = float(right.gamma - center.gamma)
        before * after < 0 || continue
        float(center.pressure_l2) >= settings.minimum_pressure_l2 || continue
        kind = before > 0 && after < 0 ? :gamma_maximum : :gamma_minimum
        push!(candidates, (
            kind=kind,
            gamma=float(center.gamma),
            pressure_l2=float(center.pressure_l2),
            continuation_step=Int(center.step),
        ))
    end
    turns = Any[]
    for kind in (:gamma_minimum, :gamma_maximum)
        selected = [turn for turn in candidates if turn.kind == kind]
        isempty(selected) && continue
        index = kind == :gamma_minimum ?
            argmin(turn.gamma for turn in selected) :
            argmax(turn.gamma for turn in selected)
        push!(turns, selected[index])
    end
    sort!(turns; by=turn -> String(turn.kind))
    return turns
end

function _sax_regularized_fold_tasks(branch, turns,
        settings::SaxRegularizedFoldCompletionSettings;
        allowed_statuses::Tuple{Vararg{Symbol}}=(:converged, :guess),
        localizer::Symbol=:periodic_schur)
    # With Periodic-Schur eigenvalues BifurcationKit can leave a well-resolved
    # +1 crossing as `:guess` when the product is ill-conditioned at one scan
    # sample.  The geometric turn supplies an independent localization check;
    # the minimally augmented fold solve below performs the actual correction,
    # and strict PQZ plus FloquetColl subsequently validate the corrected root.
    branch_points = [(index=index, special=special)
                     for (index, special) in enumerate(branch.specialpoint)
                     if special.type in (:bp, :fold) &&
                        special.status in allowed_statuses &&
                        isfinite(float(special.param))]
    tasks = Any[]
    used = Set{Int}()
    for turn in turns
        available = [point for point in branch_points
                     if !(point.index in used)]
        isempty(available) && error(
            "no converged periodic branch point is available for $(turn.kind)")
        selected = available[argmin(
            abs(float(point.special.param) - turn.gamma)
            for point in available)]
        distance = abs(float(selected.special.param) - turn.gamma)
        distance <= settings.turn_gamma_tolerance || error(
            "nearest branch point is $(distance) from $(turn.kind)")
        push!(used, selected.index)
        push!(tasks, (
            key=turn.kind,
            mode=settings.mode,
            specialpoint_index=Int(selected.index),
            specialpoint_gamma=float(selected.special.param),
            specialpoint_precision=float(selected.special.precision),
            specialpoint_status=Symbol(selected.special.status),
            localizer=localizer,
            turn=turn,
        ))
    end
    sort!(tasks; by=task -> String(task.key))
    return tasks
end

function _sax_regularized_fold_curve_quality(
        curve,
        settings::SaxRegularizedFoldCompletionSettings)
    points = Tuple{Float64,Float64}[]
    for index in eachindex(curve.gamma, curve.zeta)
        gamma = float(curve.gamma[index])
        zeta = float(curve.zeta[index])
        isfinite(gamma) && isfinite(zeta) && push!(points, (gamma, zeta))
    end
    unique!(points)
    zeta_span = isempty(points) ? 0.0 :
        maximum(last, points) - minimum(last, points)
    zeta_limits = isempty(points) ? (Inf, -Inf) : extrema(last, points)
    required_coverage = isnothing(settings.required_zeta_limits) ||
        (zeta_limits[1] <= settings.required_zeta_limits[1] &&
         zeta_limits[2] >= settings.required_zeta_limits[2])
    return (
        accepted=length(points) >= settings.minimum_curve_points &&
            zeta_span >= settings.minimum_curve_zeta_span &&
            required_coverage,
        finite_points=length(points),
        zeta_span=zeta_span,
        zeta_limits=zeta_limits,
        required_coverage=required_coverage,
        required_zeta_limits=settings.required_zeta_limits,
        minimum_points=settings.minimum_curve_points,
        minimum_zeta_span=settings.minimum_curve_zeta_span,
    )
end

function _sax_regularized_fold_partial_curve(segments, task, status::Symbol)
    gamma = Float64[]
    zeta = Float64[]
    for segment in segments
        isempty(segment) && continue
        isempty(gamma) || (push!(gamma, NaN); push!(zeta, NaN))
        append!(gamma, (point.gamma for point in segment))
        append!(zeta, (point.zeta for point in segment))
    end
    isempty(gamma) && return Any[]
    return Any[(
        kind=:fold,
        gamma=gamma,
        zeta=zeta,
        mode=task.mode,
        frequency=NaN,
        source=(
            analysis=:regularized_fold_completion,
            status=status,
            component=task.key,
            provisional=status != :complete,
        ),
        diagnostics=nothing,
    )]
end

function _sax_regularized_fold_checkpoint_callback(
        path::AbstractString,
        task,
        model_p::NamedTuple,
        settings::SaxRegularizedFoldCompletionSettings,
        verbosity::Integer)
    segments = Vector{Vector{NamedTuple}}([NamedTuple[]])
    last_step = Ref(0)
    started_ns = time_ns()
    callback = function (z, tangent, step, branch; kwargs...)
        solver_state = get(kwargs, :state, nothing)
        !isnothing(solver_state) && BK.in_bisection(solver_state) && return true
        inner = BK.getvec(z)
        gamma = float(BK.getp(inner))
        zeta = float(BK.getp(z))
        if Int(step) <= last_step[]
            push!(segments, NamedTuple[])
        end
        last_step[] = Int(step)
        push!(segments[end], (gamma=gamma, zeta=zeta, step=Int(step)))
        if verbosity > 0 && (step == 1 || step % settings.checkpoint_every == 0)
            @info(
                "Regularized fold-curve progress",
                component=task.key,
                accepted_step=Int(step),
                step_limit=settings.curve_max_steps,
                gamma,
                zeta,
                elapsed_seconds=_sax_elapsed_seconds(started_ns),
            )
        end
        if step == 1 || step % settings.checkpoint_every == 0
            payload = (
                analysis=:regularized_fold_completion_component,
                status=:partial,
                key=task.key,
                seed=task,
                curves=_sax_regularized_fold_partial_curve(
                    segments, task, :partial),
                validation=nothing,
                saved_at_unix=time(),
            )
            _save_sax_transition_mechanism_cache(
                path, :regularized_fold_completion_component,
                payload, model_p, settings)
        end
        return true
    end
    return callback
end

function _sax_continue_regularized_fold(
        branch,
        task,
        model_p::NamedTuple,
        component_path::AbstractString,
        settings::SaxRegularizedFoldCompletionSettings;
        verbosity::Integer=1)
    options = BK.ContinuationPar(
        p_min=settings.zeta_range[1],
        p_max=settings.zeta_range[2],
        ds=settings.curve_ds,
        dsmin=min(1e-7, abs(settings.curve_ds) / 100),
        dsmax=settings.curve_dsmax,
        max_steps=settings.curve_max_steps,
        detect_bifurcation=0,
        nev=2 + 2settings.nmodes,
        save_eigenvectors=true,
        save_sol_every_step=1,
        tol_stability=settings.stability_tol,
        newton_options=BK.NewtonPar(
            tol=settings.newton_tol,
            max_iterations=45,
            verbose=false,
        ),
    )
    callback = _sax_regularized_fold_checkpoint_callback(
        component_path, task, model_p, settings, verbosity)
    return BK.continuation(
        branch,
        task.specialpoint_index,
        (BK.@optic _.zeta),
        options;
        detect_codim2_bifurcation=0,
        update_minaug_every_step=1,
        start_with_eigen=false,
        usehessian=true,
        jacobian_ma=BK.MinAug(),
        bothside=true,
        normC=BK.norminf,
        bdlinsolver=BK.BorderingBLS(
            solver=BK.DefaultLS(), check_precision=false),
        finalise_solution=callback,
        verbosity=max(0, Int(verbosity) - 1),
        plot=false,
    )
end

function _sax_regularized_fold_dual_acceptance(
        validation,
        settings::SaxRegularizedFoldCompletionSettings)
    pqz_distance = abs(validation.periodic_schur.multiplier - 1)
    coll_distance = abs(validation.floquet_coll.multiplier - 1)
    accepted = validation.periodic_schur.converged &&
        validation.floquet_coll.converged &&
        validation.orbit_residual <= 100settings.newton_tol &&
        validation.methods_agree && validation.fold_turn_agrees &&
        _sax_schur_expected_matches(
            :fold, validation.periodic_schur.classification) &&
        _sax_schur_expected_matches(
            :fold, validation.floquet_coll.classification) &&
        pqz_distance <= settings.maximum_plus_one_multiplier_distance &&
        coll_distance <= settings.maximum_plus_one_multiplier_distance
    return (
        accepted=accepted,
        periodic_schur=float(pqz_distance),
        floquet_coll=float(coll_distance),
        tolerance=settings.maximum_plus_one_multiplier_distance,
    )
end

function _sax_regularized_fold_validation(curve, task,
        model_p::NamedTuple,
        settings::SaxRegularizedFoldCompletionSettings)
    try
        isempty(curve.sol) && error("fold curve saved no augmented solution")
        entry = curve.sol[argmin(abs(float(solution.p) - settings.seed_zeta)
                                 for solution in curve.sol)]
        augmented = entry.x
        hasproperty(augmented, :p1) || error(
            "fold continuation did not save a minimally augmented solution",
        )
        hasproperty(augmented, :x) || error(
            "fold continuation did not save the underlying periodic orbit",
        )
        stored_orbit = augmented.x
        stored_orbit = stored_orbit isa BK.BorderedArray ?
            BK.getvec(stored_orbit) : stored_orbit
        stored_orbit = hasproperty(stored_orbit, :sol) ?
            stored_orbit.sol : stored_orbit
        checkpoint = (
            key="regularized_fold_$(task.key)",
            type=:fold,
            mode=settings.mode,
            source_hopf_key="fixed_zeta_fold_completion",
            specialpoint_index=task.specialpoint_index,
            localization_status=:minimally_augmented,
            localization_precision=task.specialpoint_precision,
            gamma=float(augmented.p1),
            zeta=float(entry.p),
            floquet_angle=0.0,
            solution=collect(float.(stored_orbit)),
        )
        bifurcation_settings = sax_bifurcation_settings(
            :final;
            nmodes=settings.nmodes,
            gamma_range=settings.gamma_range,
            zeta_range=settings.zeta_range,
            zeta_seeds=(settings.seed_zeta,),
            po_collocation_intervals=settings.collocation_intervals,
            po_collocation_degree=settings.collocation_degree,
            po_linear_solver=:condensed,
            newton_tol=settings.newton_tol,
            stability_tol=settings.stability_tol,
        )
        schur_settings = sax_fixed_zeta_schur_settings(
            :final;
            nmodes=settings.nmodes,
            zeta=settings.seed_zeta,
            modes=(settings.mode,),
            hopf_gamma_hints=(settings.hopf_gamma_hint,),
            collocation_intervals=settings.collocation_intervals,
            collocation_degree=settings.collocation_degree,
            newton_tol=settings.newton_tol,
            stability_tol=settings.stability_tol,
        )
        validation = _sax_schur_validate_checkpoint(
            checkpoint, :fold, model_p, bifurcation_settings,
            schur_settings, Any[], Any[task.turn])
        # At a periodic Fold the phase multiplier and the physical Fold
        # multiplier form a double +1 cluster. Finite collocation can split
        # this cluster either radially (small real growth) or angularly (a
        # small conjugate angle). The generic fixed-slice test bounds those
        # coordinates separately and treated the two splittings
        # inconsistently. Here the invariant metric is distance to +1 in the
        # multiplier plane, backed by the geometric turn and the minimally
        # augmented Fold equation.
        acceptance = _sax_regularized_fold_dual_acceptance(
            validation, settings)
        return merge(validation, (
            accepted=acceptance.accepted,
            minaug_converged=true,
            plus_one_multiplier_distance=(
                periodic_schur=acceptance.periodic_schur,
                floquet_coll=acceptance.floquet_coll,
                tolerance=acceptance.tolerance,
            ),
            validation_status=acceptance.accepted ?
                :dual_floquet_validated : :minaug_only,
        ))
    catch err
        err isa InterruptException && rethrow()
        return (
            accepted=false,
            minaug_converged=true,
            validation_status=:minaug_only,
            exception_type=Symbol(nameof(typeof(err))),
            error=sprint(showerror, err),
        )
    end
end

"""
    compute_sax_regularized_fold_completion(model, output_root; settings, ...)

Resolve the two geometric turns of the mode-1 P1 family at ζ=0.6. Periodic
Schur is the primary scan and final stability test. If it leaves a
geometrically matched branch point as `:guess`, an independent FloquetColl
scan is allowed to provide a converged predictor only. Each predictor is then
corrected with the minimally augmented fold equations and continued in
(γ, ζ), and the corrected seed must pass strict PQZ plus the independent
cross-check. Each component is committed independently and exposes plot-ready
partial curves while it is running.
"""
function compute_sax_regularized_fold_completion(
        model_p::NamedTuple,
        output_root::AbstractString;
        settings::SaxRegularizedFoldCompletionSettings=
            SaxRegularizedFoldCompletionSettings(),
        resume::Bool=true,
        verbosity::Integer=1)
    _validate_sax_regularized_fold_completion_settings(settings)
    paths = sax_regularized_fold_completion_paths(
        output_root; settings=settings)
    mkpath(paths.directory)
    expected = (:gamma_minimum, :gamma_maximum)
    pending = Symbol[]
    for key in expected
        loaded = resume ? _load_sax_transition_mechanism_cache(
            paths.component(key), :regularized_fold_completion_component,
            model_p, settings) :
            (status=:missing, payload=nothing, reason="resume disabled")
        reusable = loaded.status == :valid &&
            loaded.payload.status == :complete &&
            !isnothing(loaded.payload.validation) &&
            hasproperty(loaded.payload.validation, :accepted) &&
            loaded.payload.validation.accepted
        reusable || push!(pending, key)
    end
    if isempty(pending)
        return load_sax_regularized_fold_completion_progress(
            model_p, output_root; settings=settings)
    end

    hopf = refine_sax_hopf_checkpoint(
        model_p,
        settings.seed_zeta,
        settings.mode;
        settings=_sax_regularized_fold_hopf_settings(settings),
        gamma_hint=settings.hopf_gamma_hint,
    )
    bifurcation_settings =
        _sax_regularized_fold_bifurcation_settings(settings)
    verbosity > 0 && @info(
        "Regularized fold seed branch started",
        mode=settings.mode,
        zeta=settings.seed_zeta,
        collocation="$(settings.collocation_intervals)x$(settings.collocation_degree)",
    )
    # PQZ remains the primary stability solver. A single ill-conditioned
    # periodic product must not destroy the geometric fold seed branch, so all
    # cyclic orderings are tried before FloquetColl is used for that sample.
    # The final fold checkpoint is independently re-evaluated with strict PQZ.
    branch = continue_sax_periodic_orbits(
        hopf,
        model_p;
        settings=bifurcation_settings,
        verbosity=max(0, Int(verbosity) - 1),
        bothside=false,
        eigsolver=SaxFloquetPQZ(
            cyclic_retries=max(8, settings.collocation_intervals - 1),
            fallback_to_floquet_coll=true,
        ),
        detect_bifurcation=3,
        save_eigenvectors=true,
    )
    turns = _sax_regularized_fold_turns(branch, settings)
    tasks = _sax_regularized_fold_tasks(
        branch, turns, settings; localizer=:periodic_schur)
    found = Set(task.key for task in tasks)
    all(key -> key in found, expected) || error(
        "the fine P1 branch did not resolve both geometric turns")
    # A :guess close to a geometric turn is a useful predictor, but the upper
    # fold previously started from a 1e-5 predictor and its first positive-zeta
    # correction produced NaNs near zeta=0.6.  Re-localize both predictors on
    # an independent FloquetColl scan when strict PQZ did not return converged
    # special points.  This scan is not the stability authority: every final
    # minimally augmented fold seed still has to pass strict Periodic Schur and
    # the independent FloquetColl agreement test below.
    if any(task -> task.specialpoint_status != :converged, tasks)
        verbosity > 0 && @info(
            "Regularized fold predictors need converged localizers; starting independent seed scan",
            guessed_components=Tuple(task.key for task in tasks
                                     if task.specialpoint_status != :converged),
        )
        try
            localization_branch = continue_sax_periodic_orbits(
                hopf,
                model_p;
                settings=bifurcation_settings,
                verbosity=max(0, Int(verbosity) - 1),
                bothside=false,
                eigsolver=BK.FloquetColl(),
                detect_bifurcation=3,
                save_eigenvectors=true,
            )
            localization_turns = _sax_regularized_fold_turns(
                localization_branch, settings)
            localization_tasks = _sax_regularized_fold_tasks(
                localization_branch, localization_turns, settings;
                allowed_statuses=(:converged,),
                localizer=:floquet_coll_predictor,
            )
            localized = Set(task.key for task in localization_tasks)
            if all(key -> key in localized, expected)
                branch = localization_branch
                tasks = localization_tasks
                verbosity > 0 && @info(
                    "Converged fold predictors recovered; strict PQZ remains the acceptance test",
                    components=Tuple(task.key for task in tasks),
                    gammas=Tuple(task.specialpoint_gamma for task in tasks),
                )
            else
                @warn "Independent fold seed scan did not recover both converged predictors; retaining geometrically matched PQZ guesses" localized
            end
        catch err
            err isa InterruptException && rethrow()
            @warn "Independent fold seed scan failed; retaining geometrically matched PQZ guesses" exception=(err, catch_backtrace())
        end
    end
    failures = Any[]
    for task in tasks
        task.key in pending || continue
        component_path = paths.component(task.key)
        started_ns = time_ns()
        verbosity > 0 && @info(
            "Regularized fold-curve continuation started",
            component=task.key,
            seed_gamma=task.specialpoint_gamma,
            turn_gamma=task.turn.gamma,
            seed_zeta=settings.seed_zeta,
        )
        try
            curve = _sax_continue_regularized_fold(
                branch, task, model_p, component_path, settings;
                verbosity=verbosity)
            summary = _curve_summary(
                curve,
                :fold;
                mode=settings.mode,
                source=(
                    analysis=:regularized_fold_completion,
                    status=:complete,
                    component=task.key,
                    provisional=false,
                ),
                active_parameter=:zeta,
            )
            quality = _sax_regularized_fold_curve_quality(summary, settings)
            validation = _sax_regularized_fold_validation(
                curve, task, model_p, settings)
            complete = quality.accepted && validation.accepted
            stored_summary = complete ? summary : merge(summary, (
                source=merge(summary.source, (
                    status=:partial,
                    provisional=true,
                )),
            ))
            payload = (
                analysis=:regularized_fold_completion_component,
                status=complete ? :complete : :partial,
                key=task.key,
                seed=task,
                curves=Any[_portable_sax_curve(stored_summary)],
                validation=validation,
                quality=quality,
                saved_at_unix=time(),
            )
            _save_sax_transition_mechanism_cache(
                component_path, :regularized_fold_completion_component,
                payload, model_p, settings)
            completion_message = complete ?
                "Regularized fold-curve continuation completed" :
                "Regularized fold curve remains partial"
            verbosity > 0 && @info(
                completion_message,
                component=task.key,
                points=length(summary.gamma),
                zeta_limits=quality.zeta_limits,
                zeta_span=quality.zeta_span,
                required_coverage=quality.required_coverage,
                validation=validation.validation_status,
                elapsed_seconds=_sax_elapsed_seconds(started_ns),
            )
            if !complete
                push!(failures, (
                    component=task.key,
                    exception_type=:IncompleteCurve,
                    error="fold curve or its seed did not satisfy the completion criteria",
                    quality=quality,
                    validation_status=validation.validation_status,
                ))
            end
        catch err
            err isa InterruptException && rethrow()
            failure = (
                component=task.key,
                exception_type=Symbol(nameof(typeof(err))),
                error=sprint(showerror, err),
            )
            push!(failures, failure)
            @warn "Regularized fold component failed; its latest checkpoint remains plot-ready" failure
        end
    end
    progress = load_sax_regularized_fold_completion_progress(
        model_p, output_root; settings=settings)
    manifest = (
        analysis=:regularized_fold_completion,
        status=progress.status,
        counts=progress.counts,
        failures=failures,
        settings=_portable_sax_mechanism_settings(settings),
        saved_at_unix=time(),
    )
    _atomic_jld2_save(paths.manifest; manifest)
    return progress
end

"""Load completed or partial regularized fold components without mutating them."""
function load_sax_regularized_fold_completion_progress(
        model_p::NamedTuple,
        output_root::AbstractString;
        settings::SaxRegularizedFoldCompletionSettings=
            SaxRegularizedFoldCompletionSettings())
    _validate_sax_regularized_fold_completion_settings(settings)
    paths = sax_regularized_fold_completion_paths(
        output_root; settings=settings)
    components = Any[]
    statuses = Any[]
    curves = Any[]
    for key in (:gamma_minimum, :gamma_maximum)
        loaded = _load_sax_transition_mechanism_cache(
            paths.component(key), :regularized_fold_completion_component,
            model_p, settings)
        component_status = loaded.status == :valid ?
            loaded.payload.status : loaded.status
        push!(statuses, (
            key=key,
            status=component_status,
            reason=loaded.reason,
            path=paths.component(key),
        ))
        loaded.status == :valid || continue
        push!(components, loaded.payload)
        append!(curves, loaded.payload.curves)
    end
    complete = count(component -> component.status == :complete, components)
    partial = count(component -> component.status == :partial, components)
    status = complete == 2 ? :complete :
        isempty(components) ? :missing : :partial
    return (
        analysis=:regularized_fold_completion,
        status=status,
        settings=_portable_sax_mechanism_settings(settings),
        paths=paths,
        components=components,
        component_status=statuses,
        curves=curves,
        counts=(
            complete=complete,
            partial=partial,
            expected=2,
            curves=length(curves),
            dual_floquet_validated=count(component ->
                component.status == :complete &&
                hasproperty(component.validation, :accepted) &&
                component.validation.accepted, components),
        ),
    )
end
