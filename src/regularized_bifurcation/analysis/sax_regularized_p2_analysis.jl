# Restartable P2 continuation and Periodic-Schur stability analysis around the
# two R2 neighbourhoods of the regularized mode-2 PD component.

const SAX_REGULARIZED_P2_SCHEMA_VERSION = 4

Base.@kwdef struct SaxRegularizedP2Settings
    schema_version::Int = SAX_REGULARIZED_P2_SCHEMA_VERSION
    profile::Symbol = :final
    nmodes::Int = 8
    mode::Int = 2
    preferred_seed_zeta::Float64 = 0.25
    maximum_source_seeds::Int = 4
    reference_eta::Float64 = 1e-3
    minimum_ns_angle::Float64 = 0.08
    minimum_angle_to_pi::Float64 = 0.08
    maximum_pair_angle_jump::Float64 = 0.35
    maximum_pair_growth_jump::Float64 = 0.50
    maximum_gamma_bracket::Float64 = 0.02
    stability_tolerance::Float64 = 1e-7
    root_gamma_tolerance::Float64 = 2e-6
    root_max_iterations::Int = 18
    orbit_residual_tolerance::Float64 = 1e-8
    pqz_growth_tolerance::Float64 = 2e-5
    floquet_coll_growth_tolerance::Float64 = 3e-3
    angle_agreement_tolerance::Float64 = 3e-2
    parent::SaxPeriodTwoSettings = SaxPeriodTwoSettings(
        gamma_range=(0.30, 0.99),
        save_sol_every_step=1,
    )
end

function sax_regularized_p2_settings(
        profile::Symbol=:final;
        reference_eta::Real=1e-3)
    profile in (:smoke, :pilot, :final) || throw(ArgumentError(
        "regularized P2 profile must be :smoke, :pilot, or :final",
    ))
    parent = if profile == :smoke
        SaxPeriodTwoSettings(
            gamma_range=(0.30, 0.75),
            collocation_intervals=8,
            collocation_degree=2,
            delta_gamma_candidates=(-2e-3, 2e-3),
            amplitude_factors=(0.05,),
            max_steps=4,
            save_sol_every_step=1,
            checkpoint_every=1,
            progress_every=1,
            newton_max_iterations=12,
        )
    elseif profile == :pilot
        SaxPeriodTwoSettings(
            gamma_range=(0.30, 0.90),
            collocation_intervals=25,
            collocation_degree=3,
            max_steps=160,
            save_sol_every_step=1,
        )
    else
        SaxPeriodTwoSettings(
            gamma_range=(0.30, 0.99),
            collocation_intervals=40,
            collocation_degree=4,
            max_steps=500,
            save_sol_every_step=1,
        )
    end
    return SaxRegularizedP2Settings(
        profile=profile,
        reference_eta=float(reference_eta),
        parent=parent,
        root_max_iterations=profile == :smoke ? 2 :
            profile == :pilot ? 12 : 18,
    )
end

function _validate_sax_regularized_p2_settings(settings::SaxRegularizedP2Settings)
    settings.schema_version == SAX_REGULARIZED_P2_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported regularized P2 schema"))
    settings.profile in (:smoke, :pilot, :final) ||
        throw(ArgumentError("invalid regularized P2 profile"))
    settings.maximum_source_seeds >= 1 ||
        throw(ArgumentError("maximum_source_seeds must be positive"))
    settings.minimum_ns_angle > 0 ||
        throw(ArgumentError("minimum_ns_angle must be positive"))
    settings.minimum_angle_to_pi > 0 ||
        throw(ArgumentError("minimum_angle_to_pi must be positive"))
    settings.root_max_iterations >= 1 ||
        throw(ArgumentError("root_max_iterations must be positive"))
    _validate_sax_period_two_settings(settings.parent)
    return settings
end

function sax_regularized_p2_paths(root::AbstractString)
    directory = joinpath(root, "p2_r2_periodic_schur")
    return (
        directory=directory,
        components=joinpath(directory, "components"),
        checkpoints=joinpath(directory, "checkpoints"),
        manifest=joinpath(directory, "p2_r2_manifest.jld2"),
    )
end

_sax_regularized_p2_safe_key(value) =
    replace(String(value), r"[^A-Za-z0-9_-]+" => "_")

function _sax_regularized_p2_component_paths(paths, task)
    tag = _sax_regularized_p2_safe_key(task.key)
    return (
        result=joinpath(paths.components, "$(tag).jld2"),
        checkpoints=joinpath(paths.checkpoints, "$(tag)_checkpoints.jld2"),
    )
end

function _sax_regularized_p2_component_signature(settings, task)
    return (
        nmodes=settings.nmodes,
        settings=_portable_sax_mechanism_settings(settings),
        task=(key=String(task.key), direction=task.direction),
    )
end

function _sax_regularized_raw_pd_hints(stage_directory::AbstractString,
                                       mode::Integer,
                                       preferred_zeta::Real)
    path = _sax_stage_cache_path(stage_directory, :periodic)
    isfile(path) || return Any[]
    stored = try
        JLD2.load(path, "cache")
    catch
        return Any[]
    end
    hasproperty(stored, :payload) || return Any[]
    candidates = [checkpoint for checkpoint in stored.payload.periodic_checkpoints
                  if checkpoint.type == :pd && checkpoint.mode == Int(mode)]
    isempty(candidates) && return Any[]
    distance = minimum(abs(checkpoint.zeta - float(preferred_zeta))
                       for checkpoint in candidates)
    selected = [checkpoint for checkpoint in candidates
                if abs(abs(checkpoint.zeta - float(preferred_zeta)) - distance) <= 1e-8]
    sort!(selected; by=checkpoint -> checkpoint.gamma)
    return [(gamma=float(checkpoint.gamma), zeta=float(checkpoint.zeta))
            for checkpoint in selected]
end

function _sax_regularized_fallback_pd_seeds(periodic_payload,
                                             settings,
                                             reference_hints)
    candidates = [checkpoint for checkpoint in periodic_payload.periodic_checkpoints
                  if checkpoint.type == :pd &&
                     checkpoint.mode == settings.mode &&
                     checkpoint.localization_status == :converged]
    isempty(candidates) && return Any[]
    zeta_distance = minimum(abs(checkpoint.zeta - settings.preferred_seed_zeta)
                            for checkpoint in candidates)
    local_candidates = [checkpoint for checkpoint in candidates
                        if abs(abs(checkpoint.zeta - settings.preferred_seed_zeta) -
                               zeta_distance) <= 1e-8]
    sort!(local_candidates; by=checkpoint -> checkpoint.gamma)
    selected = Any[]
    if !isempty(reference_hints)
        available = collect(eachindex(local_candidates))
        for hint in reference_hints
            isempty(available) && break
            position = available[argmin(
                hypot(local_candidates[index].gamma - hint.gamma,
                      local_candidates[index].zeta - hint.zeta)
                for index in available)]
            push!(selected, local_candidates[position])
            filter!(!=(position), available)
        end
    end
    if isempty(selected)
        push!(selected, first(local_candidates))
        length(local_candidates) > 1 && push!(selected, last(local_candidates))
    end
    sort!(selected; by=checkpoint -> checkpoint.gamma)
    return [merge(checkpoint, (
        resonance_type=:R2_fallback,
        resonance_cluster=index,
        resonance_side=:arm,
    )) for (index, checkpoint) in enumerate(selected)]
end

function _sax_regularized_fixed_zeta_pd_seeds(
        progress,
        settings::SaxRegularizedP2Settings)
    isnothing(progress) && return Any[]
    hasproperty(progress, :status) && progress.status == :complete ||
        return Any[]
    raw = sax_fixed_zeta_continuation_seeds(
        progress;
        gamma_limits=settings.parent.gamma_range,
        types=(:pd,),
    )
    selected = [seed for seed in raw if seed.mode == settings.mode]
    sort!(selected; by=seed -> seed.gamma)
    return [merge(seed.checkpoint, (
        key="fixed_zeta_pqz_pd_$(index)_$(seed.checkpoint.key)",
        type=:pd,
        mode=seed.mode,
        gamma=seed.gamma,
        zeta=seed.zeta,
        floquet_angle=seed.floquet_angle,
        localization_status=:fixed_zeta_pqz_validated,
        resonance_type=:R2_fallback,
        resonance_cluster=index,
        resonance_side=:arm,
    )) for (index, seed) in enumerate(selected)]
end

function _sax_regularized_p2_r2_neighbours(checkpoints, settings)
    neighbours = [checkpoint for checkpoint in checkpoints
                  if checkpoint.type == :pd &&
                     checkpoint.mode == settings.mode &&
                     hasproperty(checkpoint, :resonance_type) &&
                     checkpoint.resonance_type == :R2]
    isempty(neighbours) && return Any[]
    # A second seed or direction on the same PD component can rediscover the
    # same R2 point. Cluster by its augmented location and side before P2 work.
    sort!(neighbours; by=checkpoint -> checkpoint.localization_precision)
    distinct = Any[]
    for checkpoint in neighbours
        duplicate = any(distinct) do stored
            stored.resonance_side == checkpoint.resonance_side &&
                hypot(stored.r2_gamma - checkpoint.r2_gamma,
                      stored.r2_zeta - checkpoint.r2_zeta) <= 5e-3
        end
        duplicate || push!(distinct, checkpoint)
    end
    sort!(distinct; by=checkpoint -> (
        checkpoint.r2_gamma,
        checkpoint.resonance_side == :before ? 0 : 1,
    ))
    return collect(Iterators.take(
        distinct, settings.maximum_source_seeds))
end

function _sax_regularized_p2_source_seeds(model_p,
                                          stage_directory,
                                          main_settings,
                                          settings;
                                          reference_stage_directory=nothing,
                                          fixed_zeta_progress=nothing,
                                          plane_progress=nothing)
    periodic = _load_sax_stage_cache(
        _sax_stage_cache_path(stage_directory, :periodic),
        :periodic, model_p, main_settings)
    fixed_zeta = _sax_regularized_fixed_zeta_pd_seeds(
        fixed_zeta_progress, settings)
    plane_checkpoints = !isnothing(plane_progress) &&
            hasproperty(plane_progress, :periodic_codim2_checkpoints) ?
        plane_progress.periodic_codim2_checkpoints : Any[]
    curves = _load_sax_stage_cache(
        _sax_stage_cache_path(stage_directory, :curves),
        :curves, model_p, main_settings)
    main_checkpoints = curves.status == :valid &&
            hasproperty(curves.payload, :periodic_codim2_checkpoints) ?
        curves.payload.periodic_codim2_checkpoints : Any[]
    neighbours = _sax_regularized_p2_r2_neighbours(
        Any[main_checkpoints..., plane_checkpoints...], settings)
    if !isempty(neighbours)
        return (
            status=:valid,
            source=:r2_neighbours,
            seeds=neighbours,
            reason="PD orbits immediately before and after localized R2 points",
        )
    end
    if periodic.status != :valid
        return isempty(fixed_zeta) ? (
            status=periodic.status,
            source=:missing,
            seeds=Any[],
            reason="periodic stage is unavailable and no fixed-zeta PQZ PD seed exists: $(periodic.reason)",
        ) : (
            status=:valid,
            source=:fixed_zeta_periodic_schur,
            seeds=collect(Iterators.take(
                fixed_zeta, settings.maximum_source_seeds)),
            reason="dual-validated fixed-zeta Periodic-Schur PD roots",
        )
    end
    reference_hints = isnothing(reference_stage_directory) ? Any[] :
        _sax_regularized_raw_pd_hints(
            reference_stage_directory,
            settings.mode,
            settings.preferred_seed_zeta,
        )
    fallback = _sax_regularized_fallback_pd_seeds(
        periodic.payload, settings, reference_hints)
    if isempty(fallback) && !isempty(fixed_zeta)
        return (
            status=:valid,
            source=:fixed_zeta_periodic_schur,
            seeds=collect(Iterators.take(
                fixed_zeta, settings.maximum_source_seeds)),
            reason="main continuation contained no PD checkpoint; using dual-validated fixed-zeta Periodic-Schur roots",
        )
    end
    return (
        status=isempty(fallback) ? :missing : :valid,
        source=:pd_arms,
        seeds=collect(Iterators.take(fallback, settings.maximum_source_seeds)),
        reason=isempty(fallback) ? "no converged mode-2 PD seed is available" :
            "R2-neighbour states are absent; using distinct PD-U arms",
    )
end

function _sax_regularized_p2_tasks(sources)
    tasks = Any[]
    for seed in sources.seeds
        cluster = hasproperty(seed, :resonance_cluster) ?
            Int(seed.resonance_cluster) : length(tasks) + 1
        side = hasproperty(seed, :resonance_side) ?
            Symbol(seed.resonance_side) : :arm
        key = "r2$(cluster)_$(side)_$(seed.key)"
        push!(tasks, (
            key=key,
            cluster=cluster,
            side=side,
            direction=:natural,
            seed=seed,
        ))
    end
    return tasks
end

function _sax_regularized_match_pairs(previous, current, settings)
    edges = [(
        cost=hypot(left.growth - right.growth, left.angle - right.angle),
        left=i,
        right=j,
    ) for (i, left) in enumerate(previous), (j, right) in enumerate(current)
      if abs(left.angle - right.angle) <= settings.maximum_pair_angle_jump &&
         abs(left.growth - right.growth) <= settings.maximum_pair_growth_jump]
    sort!(vec(edges); by=edge -> edge.cost)
    used_left = falses(length(previous))
    used_right = falses(length(current))
    matches = Dict{Int,Int}()
    for edge in vec(edges)
        (used_left[edge.left] || used_right[edge.right]) && continue
        used_left[edge.left] = true
        used_right[edge.right] = true
        matches[edge.right] = edge.left
    end
    return matches
end

function _sax_regularized_track_p2_floquet(samples, settings)
    tracked = Any[]
    previous = Any[]
    next_track = 1
    for raw in samples
        canonical = sax_canonical_floquet_pairs(raw.exponents)
        matches = isempty(previous) ? Dict{Int,Int}() :
            _sax_regularized_match_pairs(previous, canonical.pairs, settings)
        pairs = Any[]
        for (index, pair) in enumerate(canonical.pairs)
            track = haskey(matches, index) ?
                previous[matches[index]].track_id : next_track
            haskey(matches, index) || (next_track += 1)
            push!(pairs, merge(pair, (track_id=track,)))
        end
        dominant = isempty(pairs) ? nothing :
            pairs[argmax(pair.growth for pair in pairs)]
        push!(tracked, merge(raw, (
            neutral_exponent=canonical.neutral_exponent,
            pairs=pairs,
            dominant_growth=isnothing(dominant) ? NaN : dominant.growth,
            dominant_angle=isnothing(dominant) ? NaN : dominant.angle,
            stable=!isnothing(dominant) &&
                dominant.growth < -settings.stability_tolerance,
        )))
        previous = pairs
    end
    return tracked
end

function _sax_regularized_pair_by_track(sample, track_id)
    index = findfirst(pair -> pair.track_id == track_id, sample.pairs)
    return isnothing(index) ? nothing : sample.pairs[index]
end

function _sax_regularized_p2_ns_brackets(samples, settings)
    brackets = Any[]
    for index in 1:(length(samples) - 1)
        left, right = samples[index], samples[index + 1]
        abs(right.gamma - left.gamma) <= settings.maximum_gamma_bracket || continue
        for left_pair in left.pairs
            left_pair.multiplicity == 2 || continue
            right_pair = _sax_regularized_pair_by_track(right, left_pair.track_id)
            isnothing(right_pair) && continue
            right_pair.multiplicity == 2 || continue
            min(left_pair.angle, right_pair.angle) >= settings.minimum_ns_angle ||
                continue
            min(left_pair.angle_to_pi, right_pair.angle_to_pi) >=
                settings.minimum_angle_to_pi || continue
            crossed = left_pair.growth == 0 || right_pair.growth == 0 ||
                signbit(left_pair.growth) != signbit(right_pair.growth)
            crossed || continue
            push!(brackets, (
                left_index=index,
                right_index=index + 1,
                track_id=Int(left_pair.track_id),
                gamma_lower=min(left.gamma, right.gamma),
                gamma_upper=max(left.gamma, right.gamma),
                zeta=float(left.zeta),
                left_gamma=float(left.gamma),
                right_gamma=float(right.gamma),
                left_growth=float(left_pair.growth),
                right_growth=float(right_pair.growth),
                floquet_angle=float((left_pair.angle + right_pair.angle) / 2),
                status=:pqz_bracket,
            ))
        end
    end
    return brackets
end

function _sax_regularized_select_floquet_pair(exponents, target_angle, settings)
    canonical = sax_canonical_floquet_pairs(exponents)
    candidates = [pair for pair in canonical.pairs
                  if pair.multiplicity == 2 &&
                     pair.angle >= settings.minimum_ns_angle &&
                     pair.angle_to_pi >= settings.minimum_angle_to_pi]
    isempty(candidates) && error("no admissible complex Floquet pair")
    return candidates[argmin(abs(pair.angle - target_angle)
                             for pair in candidates)]
end

function _sax_regularized_p2_bifurcation_settings(settings)
    return _sax_mechanism_bifurcation_settings(
        settings.nmodes,
        2settings.parent.collocation_intervals,
        settings.parent.collocation_degree;
        gamma_range=settings.parent.gamma_range,
        newton_tol=settings.parent.newton_tol,
        stability_tol=settings.parent.stability_tol,
    )
end

function _sax_regularized_evaluate_p2_orbit(
        guess,
        gamma,
        zeta,
        target_angle,
        model_p,
        settings)
    bifurcation_settings = _sax_regularized_p2_bifurcation_settings(settings)
    checkpoint = (
        key="regularized_p2_ns_trial",
        type=:ns,
        mode=settings.mode,
        source_hopf_key="regularized_p2",
        gamma=float(gamma),
        zeta=float(zeta),
        floquet_angle=float(target_angle),
        solution=collect(float.(guess)),
    )
    wrapper, parameters = _sax_periodic_wrapper(
        checkpoint, model_p, bifurcation_settings)
    collocation = BK.get_discretization(wrapper)
    corrected_guess = copy(checkpoint.solution)
    BK.updatesection!(collocation, corrected_guess, parameters)
    corrected = BK.newton(
        collocation,
        corrected_guess,
        BK.NewtonPar(
            tol=settings.parent.newton_tol,
            max_iterations=settings.parent.newton_max_iterations,
            linsolver=BK.COPLS(),
            verbose=false,
        );
        normN=BK.norminf,
    )
    BK.converged(corrected) || error(
        "P2 orbit correction failed at gamma=$(float(gamma)), zeta=$(float(zeta))",
    )
    solution = collect(float.(corrected.u))
    refined = merge(checkpoint, (solution=solution,))
    refined_wrapper, refined_parameters = _sax_periodic_wrapper(
        refined, model_p, bifurcation_settings)
    residual = norm(BK.residual(
        refined_wrapper, solution, refined_parameters), Inf)
    jacobian = BK.jacobian(refined_wrapper, solution, refined_parameters)
    discretization = BK.get_discretization(refined_wrapper)
    pqz_values, _, pqz_converged, _ = SaxFloquetPQZ(
        cyclic_retries=max(8, 2settings.parent.collocation_intervals - 1),
        fallback_to_floquet_coll=false)(
            discretization, jacobian, 2 + 2settings.nmodes)
    coll_values, _, coll_converged, _ = BK.FloquetColl()(
        discretization, jacobian, 2 + 2settings.nmodes)
    pqz_pair = _sax_regularized_select_floquet_pair(
        pqz_values, target_angle, settings)
    coll_pair = _sax_regularized_select_floquet_pair(
        coll_values, target_angle, settings)
    return (
        gamma=float(gamma),
        zeta=float(zeta),
        solution=solution,
        residual_norm=float(residual),
        pqz_converged=Bool(pqz_converged),
        floquet_coll_converged=Bool(coll_converged),
        pqz_pair=pqz_pair,
        floquet_coll_pair=coll_pair,
    )
end

function _sax_regularized_refine_p2_ns_bracket(
        bracket,
        samples,
        model_p,
        settings)
    left = samples[bracket.left_index]
    right = samples[bracket.right_index]
    hasproperty(left, :solution) && hasproperty(right, :solution) || error(
        "P2 NS refinement requires solutions saved at every continuation step",
    )
    left_gamma = float(left.gamma)
    right_gamma = float(right.gamma)
    left_growth = float(bracket.left_growth)
    right_growth = float(bracket.right_growth)
    left_solution = copy(left.solution)
    right_solution = copy(right.solution)
    best = nothing
    for _ in 1:settings.root_max_iterations
        gamma = (left_gamma + right_gamma) / 2
        fraction = iszero(right_gamma - left_gamma) ? 0.5 :
            clamp((gamma - left_gamma) / (right_gamma - left_gamma), 0.0, 1.0)
        guess = (1 - fraction) .* left_solution .+ fraction .* right_solution
        evaluated = _sax_regularized_evaluate_p2_orbit(
            guess,
            gamma,
            bracket.zeta,
            bracket.floquet_angle,
            model_p,
            settings,
        )
        best = isnothing(best) ||
            abs(evaluated.pqz_pair.growth) < abs(best.pqz_pair.growth) ?
            evaluated : best
        if signbit(left_growth) == signbit(evaluated.pqz_pair.growth)
            left_gamma = gamma
            left_growth = evaluated.pqz_pair.growth
            left_solution = evaluated.solution
        else
            right_gamma = gamma
            right_growth = evaluated.pqz_pair.growth
            right_solution = evaluated.solution
        end
        abs(right_gamma - left_gamma) <= settings.root_gamma_tolerance && break
    end
    isnothing(best) && error("P2 NS bracket refinement produced no orbit")
    methods_agree =
        abs(best.pqz_pair.growth - best.floquet_coll_pair.growth) <=
            settings.floquet_coll_growth_tolerance &&
        abs(best.pqz_pair.angle - best.floquet_coll_pair.angle) <=
            settings.angle_agreement_tolerance
    accepted = best.pqz_converged &&
        best.floquet_coll_converged &&
        methods_agree &&
        best.residual_norm <= settings.orbit_residual_tolerance &&
        abs(best.pqz_pair.growth) <= settings.pqz_growth_tolerance
    checkpoint = (
        key="p2_ns_g$(round(best.gamma; digits=10))_z$(round(best.zeta; digits=10))",
        type=:ns,
        mode=settings.mode,
        source_hopf_key="period_two_from_mode2_pd",
        specialpoint_index=0,
        localization_status=accepted ? :pqz_refined : :pqz_candidate,
        localization_precision=abs(right_gamma - left_gamma),
        gamma=best.gamma,
        zeta=best.zeta,
        floquet_angle=best.pqz_pair.angle,
        solution=best.solution,
    )
    return (
        status=accepted ? :accepted : :rejected,
        accepted=accepted,
        methods_agree=methods_agree,
        gamma_interval=extrema((left_gamma, right_gamma)),
        checkpoint=checkpoint,
        residual_norm=best.residual_norm,
        pqz=(growth=best.pqz_pair.growth,
             angle=best.pqz_pair.angle,
             multiplier=best.pqz_pair.multiplier),
        floquet_coll=(growth=best.floquet_coll_pair.growth,
                      angle=best.floquet_coll_pair.angle,
                      multiplier=best.floquet_coll_pair.multiplier),
    )
end

function _sax_regularized_p2_stability_intervals(samples)
    isempty(samples) && return Any[]
    intervals = Any[]
    first_index = 1
    for index in 2:length(samples)
        samples[index].stable == samples[first_index].stable && continue
        group = samples[first_index:(index - 1)]
        push!(intervals, (
            stable=group[1].stable,
            gamma_range=extrema(float(sample.gamma) for sample in group),
            step_range=(first(group).continuation_step,
                        last(group).continuation_step),
            points=length(group),
        ))
        first_index = index
    end
    group = samples[first_index:end]
    push!(intervals, (
        stable=group[1].stable,
        gamma_range=extrema(float(sample.gamma) for sample in group),
        step_range=(first(group).continuation_step,
                    last(group).continuation_step),
        points=length(group),
    ))
    return intervals
end

function _sax_regularized_strip_p2_solution(sample)
    return (
        continuation_step=sample.continuation_step,
        gamma=sample.gamma,
        zeta=sample.zeta,
        period=sample.period,
        exponents=sample.exponents,
        neutral_exponent=sample.neutral_exponent,
        pairs=sample.pairs,
        dominant_growth=sample.dominant_growth,
        dominant_angle=sample.dominant_angle,
        stable=sample.stable,
    )
end

function _sax_regularized_analyze_p2_run(run, model_p, settings)
    tracked = _sax_regularized_track_p2_floquet(
        run.floquet_samples, settings)
    brackets = _sax_regularized_p2_ns_brackets(tracked, settings)
    roots = Any[]
    failures = Any[]
    for bracket in brackets
        try
            push!(roots, _sax_regularized_refine_p2_ns_bracket(
                bracket, tracked, model_p, settings))
        catch err
            err isa InterruptException && rethrow()
            push!(failures, (
                stage=:p2_ns_refinement,
                gamma_interval=(bracket.gamma_lower, bracket.gamma_upper),
                exception_type=Symbol(nameof(typeof(err))),
                error=sprint(showerror, err),
            ))
        end
    end
    portable_samples = _sax_regularized_strip_p2_solution.(tracked)
    portable_run = merge(run, (floquet_samples=portable_samples,))
    return (
        run=portable_run,
        stability_intervals=_sax_regularized_p2_stability_intervals(tracked),
        pqz_samples=portable_samples,
        ns_brackets=brackets,
        ns_roots=roots,
        accepted_ns_roots=count(root -> root.accepted, roots),
        stable_samples=count(sample -> sample.stable, tracked),
        total_samples=length(tracked),
        failures=failures,
    )
end

function _sax_regularized_save_p2_manifest(
        paths, model_p, settings, sources, tasks, components, failures)
    result = (
        analysis=:regularized_p2_r2_periodic_schur,
        status=isempty(components) ? :missing :
            length(components) == length(tasks) &&
            all(component -> component.status == :complete, components) ?
                :complete : :partial,
        source_status=sources.status,
        source_kind=sources.source,
        source_reason=sources.reason,
        task_count=length(tasks),
        components=copy(components),
        failures=copy(failures),
        p2_curves=[component.analysis.run.curve for component in components
                   if component.status == :complete],
        ns_roots=[root for component in components
                  for root in (component.status == :complete ?
                      component.analysis.ns_roots : Any[])],
        settings=_portable_sax_mechanism_settings(settings),
    )
    return _save_sax_transition_mechanism_cache(
        paths.manifest,
        :regularized_p2_r2_manifest,
        result,
        model_p,
        settings,
    )
end

"""
    compute_sax_regularized_p2_r2(model_p, stage_directory, output_root; ...)

Branch-switch to P2 from the PD orbits on both sides of each retained R2
neighbourhood.  Every P2 family uses generalized Periodic Schur Floquet
multipliers, is cached independently, and is scanned for stable intervals and
refined fixed-zeta P2 Neimark-Sacker roots.
"""
function compute_sax_regularized_p2_r2(
        model_p::NamedTuple,
        stage_directory::AbstractString,
        output_root::AbstractString;
        main_settings::SaxBifurcationSettings=sax_bifurcation_settings(:final),
        settings::SaxRegularizedP2Settings=sax_regularized_p2_settings(:final),
        reference_stage_directory::Union{Nothing,AbstractString}=nothing,
        fixed_zeta_progress=nothing,
        plane_progress=nothing,
        resume::Bool=true,
        verbosity::Integer=1)
    _validate_sax_regularized_p2_settings(settings)
    paths = sax_regularized_p2_paths(output_root)
    mkpath(paths.components)
    mkpath(paths.checkpoints)
    sources = _sax_regularized_p2_source_seeds(
        model_p,
        stage_directory,
        main_settings,
        settings;
        reference_stage_directory=reference_stage_directory,
        fixed_zeta_progress=fixed_zeta_progress,
        plane_progress=plane_progress,
    )
    sources.status == :valid || error(sources.reason)
    tasks = _sax_regularized_p2_tasks(sources)
    components = Any[]
    failures = Any[]
    for task in tasks
        component_paths = _sax_regularized_p2_component_paths(paths, task)
        signature = _sax_regularized_p2_component_signature(settings, task)
        cached = resume ? _load_sax_transition_mechanism_cache(
            component_paths.result,
            :regularized_p2_r2_component,
            model_p,
            signature,
        ) : (status=:missing, payload=nothing, reason="resume disabled")
        component = if cached.status == :valid &&
                cached.payload.status == :complete
            cached.payload
        else
            verbosity > 0 && @info(
                "Starting R2-neighbour P2 Periodic-Schur component",
                key=task.key,
                cluster=task.cluster,
                side=task.side,
                gamma=task.seed.gamma,
                zeta=task.seed.zeta,
            )
            computed = try
                run = continue_sax_period_two(
                    task.seed,
                    model_p;
                    settings=settings.parent,
                    checkpoint_path=component_paths.checkpoints,
                    eigsolver=SaxFloquetPQZ(
                        cyclic_retries=max(
                            8, 2settings.parent.collocation_intervals - 1),
                        fallback_to_floquet_coll=false,
                    ),
                    detect_bifurcation=1,
                    save_eigenvectors=false,
                    include_floquet_solutions=true,
                    verbosity=verbosity,
                )
                analysis = _sax_regularized_analyze_p2_run(
                    run, model_p, settings)
                (
                    status=:complete,
                    task=(key=task.key, cluster=task.cluster,
                          side=task.side, direction=task.direction),
                    seed=(key=task.seed.key, gamma=task.seed.gamma,
                          zeta=task.seed.zeta),
                    analysis=analysis,
                    failure=nothing,
                )
            catch err
                err isa InterruptException && rethrow()
                (
                    status=:failed,
                    task=(key=task.key, cluster=task.cluster,
                          side=task.side, direction=task.direction),
                    seed=(key=task.seed.key, gamma=task.seed.gamma,
                          zeta=task.seed.zeta),
                    analysis=nothing,
                    failure=(
                        exception_type=Symbol(nameof(typeof(err))),
                        error=sprint(showerror, err),
                    ),
                )
            end
            _save_sax_transition_mechanism_cache(
                component_paths.result,
                :regularized_p2_r2_component,
                computed,
                model_p,
                signature,
            )
        end
        push!(components, component)
        component.status == :complete || push!(failures, merge(
            (stage=:p2_r2_component, key=task.key), component.failure))
        _sax_regularized_save_p2_manifest(
            paths, model_p, settings, sources, tasks, components, failures)
    end
    return _sax_regularized_save_p2_manifest(
        paths, model_p, settings, sources, tasks, components, failures)
end

function load_sax_regularized_p2_progress(
        model_p::NamedTuple,
        output_root::AbstractString;
        settings::SaxRegularizedP2Settings=sax_regularized_p2_settings(:final))
    paths = sax_regularized_p2_paths(output_root)
    loaded = _load_sax_transition_mechanism_cache(
        paths.manifest,
        :regularized_p2_r2_manifest,
        model_p,
        settings,
    )
    return loaded.status == :valid ? loaded.payload : (
        analysis=:regularized_p2_r2_periodic_schur,
        status=loaded.status,
        source_status=:missing,
        source_kind=:missing,
        source_reason=loaded.reason,
        task_count=0,
        components=Any[],
        failures=Any[],
        p2_curves=Any[],
        ns_roots=Any[],
        settings=_portable_sax_mechanism_settings(settings),
    )
end
