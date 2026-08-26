# Complete fixed-zeta Floquet audit for the two P1 register families.
#
# This stage is intentionally separate from the amplitude continuation.  It
# continues the periodic families ending at H1 and H2 with generalized
# Periodic Schur (PQZ), tracks every nontrivial Floquet multiplier or conjugate
# pair, brackets every loss or recovery of stability, and validates localized
# fold, PD, and NS events against the independent FloquetColl calculation.

const SAX_FIXED_ZETA_SCHUR_SCHEMA_VERSION = 2

Base.@kwdef struct SaxFixedZetaSchurSettings
    schema_version::Int = SAX_FIXED_ZETA_SCHUR_SCHEMA_VERSION
    nmodes::Int = 8
    zeta::Float64 = 0.6
    modes::Tuple{Vararg{Int}} = (1, 2)
    hopf_gamma_hints::Tuple{Vararg{Float64}} = (0.40, 0.43)
    directions::Tuple{Vararg{Symbol}} = (:positive, :negative)
    gamma_range::Tuple{Float64,Float64} = (0.30, 2.00)
    contact_stiffness::Float64 = 100.0
    collocation_intervals::Int = 40
    collocation_degree::Int = 4
    po_ds::Float64 = 1e-3
    po_dsmax::Float64 = 6e-3
    po_max_steps::Int = 900
    hopf_scan_points::Int = 681
    newton_tol::Float64 = 1e-10
    stability_tol::Float64 = 1e-8
    root_growth_tolerance::Float64 = 2e-5
    root_gamma_tolerance::Float64 = 1e-7
    maximum_root_iterations::Int = 40
    method_growth_tolerance::Float64 = 2e-5
    method_angle_tolerance::Float64 = 2e-3
    plus_one_angle_tolerance::Float64 = 0.06
    minus_one_angle_tolerance::Float64 = 0.06
    resonance_angle_tolerance::Float64 = 0.04
    maximum_pair_angle_jump::Float64 = 0.30
    maximum_pair_growth_jump::Float64 = 0.40
    turn_min_gamma_excursion::Float64 = 5e-5
    event_gamma_tolerance::Float64 = 2e-3
    event_amplitude_tolerance::Float64 = 5e-3
    minimum_event_amplitude::Float64 = 1e-2
    flow_reversal_margin::Float64 = 1e-3
end

"""Resolution presets for the fixed-zeta Periodic-Schur stability audit."""
function sax_fixed_zeta_schur_settings(profile::Symbol=:final; kwargs...)
    profile in (:smoke, :pilot, :final) || throw(ArgumentError(
        "profile must be :smoke, :pilot, or :final"))
    resolution = if profile == :smoke
        (
            gamma_range=(0.34, 0.48),
            collocation_intervals=8,
            collocation_degree=2,
            po_ds=4e-3,
            po_dsmax=1e-2,
            po_max_steps=45,
            hopf_scan_points=81,
            root_growth_tolerance=2e-4,
            root_gamma_tolerance=1e-5,
            method_growth_tolerance=5e-4,
            method_angle_tolerance=2e-2,
        )
    elseif profile == :pilot
        (
            gamma_range=(0.30, 1.00),
            collocation_intervals=25,
            collocation_degree=3,
            po_ds=2e-3,
            po_dsmax=8e-3,
            po_max_steps=450,
            hopf_scan_points=281,
            root_growth_tolerance=5e-5,
            method_growth_tolerance=1e-4,
            method_angle_tolerance=5e-3,
        )
    else
        NamedTuple()
    end
    settings = SaxFixedZetaSchurSettings(; merge(resolution, (; kwargs...))...)
    return _validate_sax_fixed_zeta_schur_settings(settings)
end

function _validate_sax_fixed_zeta_schur_settings(
        settings::SaxFixedZetaSchurSettings)
    settings.schema_version == SAX_FIXED_ZETA_SCHUR_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported fixed-zeta Schur schema"))
    settings.nmodes > 0 || throw(ArgumentError("nmodes must be positive"))
    settings.zeta > 0 || throw(ArgumentError("zeta must be positive"))
    settings.gamma_range[1] < settings.gamma_range[2] ||
        throw(ArgumentError("gamma_range must be increasing"))
    !isempty(settings.modes) || throw(ArgumentError("modes cannot be empty"))
    length(settings.modes) == length(settings.hopf_gamma_hints) ||
        throw(ArgumentError("one Hopf gamma hint is required per mode"))
    all(mode -> 1 <= mode <= settings.nmodes, settings.modes) ||
        throw(ArgumentError("modes must lie in 1:nmodes"))
    all(direction -> direction in (:positive, :negative), settings.directions) ||
        throw(ArgumentError("directions must be :positive or :negative"))
    length(unique(settings.directions)) == length(settings.directions) ||
        throw(ArgumentError("directions must be unique"))
    settings.collocation_intervals >= 5 ||
        throw(ArgumentError("at least five collocation intervals are required"))
    settings.collocation_degree >= 2 ||
        throw(ArgumentError("collocation_degree must be at least two"))
    settings.po_max_steps > 0 || throw(ArgumentError("po_max_steps must be positive"))
    settings.maximum_root_iterations > 0 ||
        throw(ArgumentError("maximum_root_iterations must be positive"))
    settings.maximum_pair_angle_jump > 0 ||
        throw(ArgumentError("maximum_pair_angle_jump must be positive"))
    settings.maximum_pair_growth_jump > 0 ||
        throw(ArgumentError("maximum_pair_growth_jump must be positive"))
    settings.minimum_event_amplitude >= 0 ||
        throw(ArgumentError("minimum_event_amplitude must be nonnegative"))
    return settings
end

function _portable_sax_fixed_zeta_schur_settings(
        settings::SaxFixedZetaSchurSettings)
    names = fieldnames(SaxFixedZetaSchurSettings)
    return NamedTuple{names}(Tuple(getfield(settings, name) for name in names))
end

"""Paths for independently restartable H1/H2 directional audits."""
function sax_fixed_zeta_schur_paths(
        root::AbstractString;
        settings::SaxFixedZetaSchurSettings=SaxFixedZetaSchurSettings())
    _validate_sax_fixed_zeta_schur_settings(settings)
    zeta_tag = _sax_fixed_zeta_tag(settings.zeta)
    stiffness_tag = _sax_fixed_zeta_tag(settings.contact_stiffness; digits=1)
    directory = joinpath(
        root, "fixed_zeta_schur_z$(zeta_tag)_kc$(stiffness_tag)")
    return (
        directory=directory,
        component=(mode, direction) -> joinpath(
            directory, "p1_m$(Int(mode))_$(String(direction)).jld2"),
        manifest=joinpath(directory, "manifest.jld2"),
    )
end

function _save_sax_fixed_zeta_schur_cache(
        path::AbstractString,
        payload,
        model_p::NamedTuple,
        settings::SaxFixedZetaSchurSettings)
    cache = (
        schema_version=SAX_FIXED_ZETA_SCHUR_SCHEMA_VERSION,
        analysis=:fixed_zeta_periodic_schur,
        settings_signature=_portable_sax_fixed_zeta_schur_settings(settings),
        model_signature=_sax_bifurcation_model_signature(model_p, settings.nmodes),
        saved_at_unix=time(),
        payload=payload,
    )
    _atomic_jld2_save(path; cache)
    return payload
end

function _load_sax_fixed_zeta_schur_cache(
        path::AbstractString,
        model_p::NamedTuple,
        settings::SaxFixedZetaSchurSettings)
    isfile(path) || return (status=:missing, payload=nothing, reason="cache is absent")
    stored = try
        Logging.with_logger(Logging.NullLogger()) do
            JLD2.load(path, "cache")
        end
    catch err
        return (status=:corrupt, payload=nothing, reason=sprint(showerror, err))
    end
    checks = try
        (
            stored.schema_version == SAX_FIXED_ZETA_SCHUR_SCHEMA_VERSION =>
                "cache schema changed",
            stored.analysis == :fixed_zeta_periodic_schur =>
                "analysis kind changed",
            isequal(stored.settings_signature,
                    _portable_sax_fixed_zeta_schur_settings(settings)) =>
                "settings changed",
            isequal(stored.model_signature,
                    _sax_bifurcation_model_signature(model_p, settings.nmodes)) =>
                "model changed",
        )
    catch err
        return (status=:corrupt, payload=nothing, reason=sprint(showerror, err))
    end
    for (valid, reason) in checks
        valid || return (status=:incompatible, payload=nothing, reason=reason)
    end
    return (status=:valid, payload=stored.payload, reason="")
end

function _sax_fixed_zeta_schur_model(
        model_p::NamedTuple,
        settings::SaxFixedZetaSchurSettings)
    return merge(model_p, (contact_stiffness=settings.contact_stiffness,))
end

function _sax_fixed_zeta_schur_hopf_settings(
        settings::SaxFixedZetaSchurSettings)
    amplitude = SaxFixedZetaAmplitudeSettings(
        nmodes=settings.nmodes,
        zeta=settings.zeta,
        modes=settings.modes,
        hopf_gamma_hints=settings.hopf_gamma_hints,
        gamma_range=settings.gamma_range,
        contact_stiffness=settings.contact_stiffness,
        collocation_intervals=settings.collocation_intervals,
        collocation_degree=settings.collocation_degree,
        po_ds=settings.po_ds,
        po_dsmax=settings.po_dsmax,
        po_max_steps=settings.po_max_steps,
        po_save_sol_every_step=1,
        hopf_scan_points=settings.hopf_scan_points,
        include_period_two=false,
        newton_tol=settings.newton_tol,
        stability_tol=settings.stability_tol,
    )
    return _sax_fixed_zeta_hopf_settings(amplitude)
end

function _sax_fixed_zeta_schur_bifurcation_settings(
        settings::SaxFixedZetaSchurSettings,
        direction::Symbol)
    signed_ds = direction == :positive ? abs(settings.po_ds) : -abs(settings.po_ds)
    half_width = max(1e-6, settings.zeta * 1e-6)
    return sax_bifurcation_settings(
        :final;
        nmodes=settings.nmodes,
        gamma_range=settings.gamma_range,
        zeta_range=(settings.zeta - half_width, settings.zeta + half_width),
        zeta_seeds=(settings.zeta,),
        po_collocation_intervals=settings.collocation_intervals,
        po_collocation_degree=settings.collocation_degree,
        po_linear_solver=:condensed,
        po_ds=signed_ds,
        po_dsmax=settings.po_dsmax,
        po_max_steps=settings.po_max_steps,
        po_save_sol_every_step=1,
        newton_tol=settings.newton_tol,
        stability_tol=settings.stability_tol,
    )
end

function _sax_schur_match_pairs(previous, current,
                                settings::SaxFixedZetaSchurSettings)
    edges = Any[]
    for (left_index, left) in enumerate(previous)
        for (right_index, right) in enumerate(current)
            angle_jump = abs(left.angle - right.angle)
            growth_jump = abs(left.growth - right.growth)
            angle_jump <= settings.maximum_pair_angle_jump || continue
            growth_jump <= settings.maximum_pair_growth_jump || continue
            push!(edges, (
                cost=hypot(
                    angle_jump / settings.maximum_pair_angle_jump,
                    growth_jump / settings.maximum_pair_growth_jump,
                ),
                left=left_index,
                right=right_index,
            ))
        end
    end
    sort!(edges; by=edge -> edge.cost)
    used_left = Set{Int}()
    used_right = Set{Int}()
    matches = Dict{Int,Int}()
    for edge in edges
        edge.left in used_left && continue
        edge.right in used_right && continue
        matches[edge.right] = edge.left
        push!(used_left, edge.left)
        push!(used_right, edge.right)
    end
    return matches
end

function _sax_schur_track_samples(raw, settings::SaxFixedZetaSchurSettings)
    tracked = Any[]
    previous = Any[]
    next_track_id = 1
    for sample in raw
        canonical = sax_canonical_floquet_pairs(sample.exponents)
        matches = isempty(previous) ? Dict{Int,Int}() :
            _sax_schur_match_pairs(previous, canonical.pairs, settings)
        pairs = Any[]
        for (index, pair) in enumerate(canonical.pairs)
            track_id = if haskey(matches, index)
                previous[matches[index]].track_id
            else
                value = next_track_id
                next_track_id += 1
                value
            end
            push!(pairs, merge(pair, (track_id=track_id,)))
        end
        dominant = isempty(pairs) ? nothing :
            pairs[argmax(pair.growth for pair in pairs)]
        unstable = sum((pair.multiplicity for pair in pairs
                        if pair.growth > settings.stability_tol); init=0)
        push!(tracked, merge(sample, (
            neutral_exponent=canonical.neutral_exponent,
            pairs=pairs,
            stable=unstable == 0,
            n_unstable=Int(unstable),
            dominant_growth=isnothing(dominant) ? NaN : dominant.growth,
            dominant_angle=isnothing(dominant) ? NaN : dominant.angle,
        )))
        previous = pairs
    end
    return tracked
end

function _sax_fixed_zeta_schur_samples(
        branch,
        mode::Integer,
        direction::Symbol,
        settings::SaxFixedZetaSchurSettings)
    rows = Dict(Int(row.step) => row for row in branch.branch)
    solutions = Dict(Int(solution.step) => collect(float.(solution.x))
                     for solution in branch.sol)
    raw = Any[]
    for eigen_record in branch.eig
        eigen_record.converged || continue
        step = Int(eigen_record.step)
        haskey(rows, step) && haskey(solutions, step) || continue
        row = rows[step]
        exponents = ComplexF64.(eigen_record.eigenvals)
        isempty(exponents) && continue
        push!(raw, (
            sample_index=length(raw) + 1,
            continuation_step=step,
            mode=Int(mode),
            direction=direction,
            gamma=float(row.gamma),
            zeta=float(settings.zeta),
            pressure_l2=float(row.pressure_l2),
            pressure_amplitude=float(row.pressure_amplitude),
            period=float(row.period),
            minimum_reed_opening=float(row.minimum_reed_opening),
            minimum_absolute_pressure_drop=
                float(row.minimum_absolute_pressure_drop),
            crosses_reed_contact=Bool(row.crosses_reed_contact),
            possible_grazing_contact=Bool(row.possible_grazing_contact),
            exponents=exponents,
        ))
    end
    sort!(raw; by=sample -> sample.continuation_step)
    return (
        samples=_sax_schur_track_samples(raw, settings),
        solutions=solutions,
    )
end

function _sax_schur_pair_by_track(sample, track_id::Integer)
    index = findfirst(pair -> pair.track_id == track_id, sample.pairs)
    return isnothing(index) ? nothing : sample.pairs[index]
end

function _sax_schur_resonance(angle::Real,
                              settings::SaxFixedZetaSchurSettings)
    candidates = ((:r2, pi), (:r3, 2pi / 3), (:r4, pi / 2))
    index = argmin(abs(float(angle) - value) for (_, value) in candidates)
    name, value = candidates[index]
    distance = abs(float(angle) - value)
    return distance <= settings.resonance_angle_tolerance ? name : :none
end

function _sax_schur_pair_classification(
        pair,
        settings::SaxFixedZetaSchurSettings)
    if pair.angle <= settings.plus_one_angle_tolerance
        return pair.multiplicity > 1 ? :multiple_plus_one : :plus_one
    elseif pair.angle_to_pi <= settings.minus_one_angle_tolerance
        return pair.multiplicity > 1 ? :r2 : :pd
    end
    return :ns
end

function _sax_schur_turn_candidates(samples,
                                    settings::SaxFixedZetaSchurSettings)
    turns = Any[]
    length(samples) < 5 && return turns
    for index in 3:(length(samples) - 2)
        left = samples[index - 2]
        center = samples[index]
        right = samples[index + 2]
        left_slope = center.gamma - left.gamma
        right_slope = right.gamma - center.gamma
        left_slope * right_slope < 0 || continue
        min(abs(left_slope), abs(right_slope)) >=
            settings.turn_min_gamma_excursion || continue
        plus_one = [pair for pair in center.pairs
                    if pair.angle <= 2settings.plus_one_angle_tolerance]
        selected = isempty(plus_one) ? nothing :
            plus_one[argmin(abs(pair.growth) for pair in plus_one)]
        candidate = (
            mode=center.mode,
            direction=center.direction,
            sample_index=index,
            continuation_step=center.continuation_step,
            gamma=center.gamma,
            pressure_l2=center.pressure_l2,
            turn=center.gamma < left.gamma && center.gamma < right.gamma ?
                :gamma_minimum : :gamma_maximum,
            plus_one_growth=isnothing(selected) ? NaN : selected.growth,
            plus_one_angle=isnothing(selected) ? NaN : selected.angle,
            track_id=isnothing(selected) ? 0 : selected.track_id,
            floquet_supported=!isnothing(selected) &&
                abs(selected.growth) <= 10settings.root_growth_tolerance,
        )
        duplicate = any(turn ->
            abs(turn.gamma - candidate.gamma) <= settings.event_gamma_tolerance &&
            abs(turn.pressure_l2 - candidate.pressure_l2) <=
                settings.event_amplitude_tolerance,
            turns,
        )
        duplicate || push!(turns, candidate)
    end
    return turns
end

function _sax_schur_candidate_brackets(
        samples,
        settings::SaxFixedZetaSchurSettings)
    candidates = Any[]
    for index in 1:(length(samples) - 1)
        left, right = samples[index], samples[index + 1]
        for left_pair in left.pairs
            max(left.pressure_l2, right.pressure_l2) >=
                settings.minimum_event_amplitude || continue
            right_pair = _sax_schur_pair_by_track(right, left_pair.track_id)
            isnothing(right_pair) && continue
            crossed = signbit(left_pair.growth) != signbit(right_pair.growth) ||
                min(abs(left_pair.growth), abs(right_pair.growth)) <=
                    settings.root_growth_tolerance
            crossed || continue
            angle = (left_pair.angle + right_pair.angle) / 2
            representative = merge(left_pair, (
                angle=float(angle),
                angle_to_pi=float(abs(pi - angle)),
                multiplicity=max(left_pair.multiplicity,
                                 right_pair.multiplicity),
            ))
            classification = _sax_schur_pair_classification(
                representative, settings)
            push!(candidates, (
                mode=left.mode,
                direction=left.direction,
                track_id=left_pair.track_id,
                classification=classification,
                resonance=_sax_schur_resonance(angle, settings),
                pair_multiplicity=representative.multiplicity,
                left_index=index,
                right_index=index + 1,
                group_right_index=index + 1,
                left_step=left.continuation_step,
                right_step=right.continuation_step,
                left_gamma=left.gamma,
                right_gamma=right.gamma,
                left_amplitude=left.pressure_l2,
                right_amplitude=right.pressure_l2,
                gamma_interval=extrema((left.gamma, right.gamma)),
                amplitude_interval=extrema((left.pressure_l2,
                                            right.pressure_l2)),
                gamma=float((left.gamma + right.gamma) / 2),
                pressure_l2=float((left.pressure_l2 + right.pressure_l2) / 2),
                left_growth=left_pair.growth,
                right_growth=right_pair.growth,
                floquet_angle=float(angle),
                unstable_before=left.n_unstable,
                unstable_after=right.n_unstable,
            ))
        end
    end
    sort!(candidates; by=candidate ->
          (candidate.track_id, candidate.left_index))
    brackets = Any[]
    for candidate in candidates
        duplicate_index = findfirst(bracket ->
            bracket.track_id == candidate.track_id &&
            bracket.classification == candidate.classification &&
            candidate.left_index <= bracket.group_right_index + 1,
            brackets,
        )
        if isnothing(duplicate_index)
            push!(brackets, candidate)
        else
            stored = brackets[duplicate_index]
            keep = min(abs(candidate.left_growth), abs(candidate.right_growth)) <
                min(abs(stored.left_growth), abs(stored.right_growth)) ?
                candidate : stored
            brackets[duplicate_index] = merge(keep, (
                group_right_index=max(stored.group_right_index,
                                      candidate.group_right_index),
            ))
        end
    end
    sort!(brackets; by=bracket -> bracket.left_index)
    return brackets
end

function _sax_schur_diagnostic_events(samples,
                                      settings::SaxFixedZetaSchurSettings)
    events = Any[]
    for index in 1:(length(samples) - 1)
        left, right = samples[index], samples[index + 1]
        diagnostics = Symbol[]
        left.crosses_reed_contact != right.crosses_reed_contact &&
            push!(diagnostics, :reed_contact_onset)
        left.possible_grazing_contact != right.possible_grazing_contact &&
            push!(diagnostics, :possible_grazing)
        left_flow = left.minimum_absolute_pressure_drop <=
            settings.flow_reversal_margin
        right_flow = right.minimum_absolute_pressure_drop <=
            settings.flow_reversal_margin
        left_flow != right_flow && push!(diagnostics, :flow_reversal_neighbourhood)
        for diagnostic in diagnostics
            push!(events, (
                type=diagnostic,
                mode=left.mode,
                direction=left.direction,
                gamma=float((left.gamma + right.gamma) / 2),
                pressure_l2=float((left.pressure_l2 + right.pressure_l2) / 2),
                gamma_interval=extrema((left.gamma, right.gamma)),
                source=:orbit_geometry,
            ))
        end
    end
    return events
end

function _sax_schur_spectrum_summary(
        exponents,
        expected::Symbol,
        target_angle::Real,
        settings::SaxFixedZetaSchurSettings;
        method::Symbol,
        converged::Bool=true,
        iterations::Integer=0)
    canonical = sax_canonical_floquet_pairs(exponents)
    pairs = canonical.pairs
    filtered = if expected == :fold
        [pair for pair in pairs
         if pair.angle <= 2settings.plus_one_angle_tolerance]
    elseif expected == :pd
        [pair for pair in pairs
         if pair.angle_to_pi <= 2settings.minus_one_angle_tolerance]
    else
        [pair for pair in pairs
         if pair.angle > settings.plus_one_angle_tolerance &&
            pair.angle_to_pi > settings.minus_one_angle_tolerance]
    end
    candidates = isempty(filtered) ? pairs : filtered
    isempty(candidates) && error("Floquet spectrum has no nontrivial pair")
    target = expected == :fold ? 0.0 : expected == :pd ? pi : abs(target_angle)
    selected = candidates[argmin(
        abs(pair.growth) / max(settings.root_growth_tolerance, eps()) +
        abs(pair.angle - target) / max(settings.method_angle_tolerance, 1e-3)
        for pair in candidates)]
    classification = _sax_schur_pair_classification(selected, settings)
    return (
        method=method,
        converged=converged,
        iterations=Int(iterations),
        neutral_exponent=canonical.neutral_exponent,
        pair=selected,
        corrected_growth=selected.growth,
        floquet_angle=selected.angle,
        angle_to_pi=selected.angle_to_pi,
        multiplier=selected.multiplier,
        classification=classification,
        resonance=_sax_schur_resonance(selected.angle, settings),
    )
end

function _sax_schur_expected_matches(expected::Symbol, classification::Symbol)
    expected == :fold && return classification in (:plus_one, :multiple_plus_one)
    expected == :pd && return classification in (:pd, :r2)
    expected == :ns && return classification == :ns
    return false
end

function _sax_schur_validate_checkpoint(
        checkpoint,
        expected::Symbol,
        model_p::NamedTuple,
        bifurcation_settings::SaxBifurcationSettings,
        settings::SaxFixedZetaSchurSettings,
        samples,
        turns)
    wrapper, parameters = _sax_periodic_wrapper(
        checkpoint, model_p, bifurcation_settings)
    collocation = BK.get_discretization(wrapper)
    jacobian = BK.jacobian(wrapper, checkpoint.solution, parameters)
    nev = 2 + 2settings.nmodes
    pqz_values, _, pqz_converged, pqz_iterations = SaxFloquetPQZ(
        cyclic_retries=max(8, settings.collocation_intervals - 1),
        fallback_to_floquet_coll=false,
    )(collocation, jacobian, nev)
    coll_values, _, coll_converged, coll_iterations = BK.FloquetColl()(
        collocation, jacobian, nev)
    target_angle = expected == :ns ? checkpoint.floquet_angle :
        expected == :pd ? pi : 0.0
    pqz = _sax_schur_spectrum_summary(
        pqz_values, expected, target_angle, settings;
        method=:periodic_schur,
        converged=Bool(pqz_converged),
        iterations=Int(pqz_iterations),
    )
    coll = _sax_schur_spectrum_summary(
        coll_values, expected, target_angle, settings;
        method=:floquet_coll,
        converged=Bool(coll_converged),
        iterations=Int(coll_iterations),
    )
    residual = float(norm(
        BK.residual(wrapper, checkpoint.solution, parameters), Inf))
    nearest = isempty(samples) ? nothing : samples[argmin(
        abs(sample.gamma - checkpoint.gamma) for sample in samples)]
    nearest_turn = isempty(turns) ? nothing : turns[argmin(hypot(
        (turn.gamma - checkpoint.gamma) /
            max(settings.event_gamma_tolerance, eps()),
        (turn.pressure_l2 - (isnothing(nearest) ? 0.0 : nearest.pressure_l2)) /
            max(settings.event_amplitude_tolerance, eps()),
    ) for turn in turns)]
    fold_turn_agrees = expected != :fold || (!isnothing(nearest_turn) &&
        abs(nearest_turn.gamma - checkpoint.gamma) <=
            5settings.event_gamma_tolerance)
    methods_agree = abs(pqz.corrected_growth - coll.corrected_growth) <=
            settings.method_growth_tolerance &&
        abs(pqz.floquet_angle - coll.floquet_angle) <=
            settings.method_angle_tolerance
    accepted = Bool(pqz_converged) && Bool(coll_converged) &&
        residual <= 100settings.newton_tol &&
        abs(pqz.corrected_growth) <= settings.root_growth_tolerance &&
        methods_agree &&
        _sax_schur_expected_matches(expected, pqz.classification) &&
        _sax_schur_expected_matches(expected, coll.classification) &&
        fold_turn_agrees
    return (
        type=expected,
        mode=Int(checkpoint.mode),
        gamma=float(checkpoint.gamma),
        zeta=float(checkpoint.zeta),
        pressure_l2=isnothing(nearest) ? NaN : nearest.pressure_l2,
        period=isnothing(nearest) ? NaN : nearest.period,
        localization_status=checkpoint.localization_status,
        localization_precision=checkpoint.localization_precision,
        orbit_residual=residual,
        periodic_schur=pqz,
        floquet_coll=coll,
        methods_agree=methods_agree,
        fold_turn_agrees=fold_turn_agrees,
        accepted=accepted,
        checkpoint=checkpoint,
    )
end

function _sax_schur_correct_orbit(
        solution,
        gamma::Real,
        mode::Integer,
        target_angle::Real,
        expected::Symbol,
        model_p::NamedTuple,
        bifurcation_settings::SaxBifurcationSettings,
        settings::SaxFixedZetaSchurSettings)
    checkpoint = (
        key="fixed_zeta_schur_trial",
        type=expected,
        mode=Int(mode),
        source_hopf_key="fixed_zeta_schur_hopf",
        gamma=float(gamma),
        zeta=float(settings.zeta),
        floquet_angle=float(target_angle),
        solution=collect(float.(solution)),
    )
    wrapper, parameters = _sax_periodic_wrapper(
        checkpoint, model_p, bifurcation_settings)
    collocation = BK.get_discretization(wrapper)
    guess = copy(checkpoint.solution)
    BK.updatesection!(collocation, guess, parameters)
    corrected = BK.newton(
        collocation,
        guess,
        BK.NewtonPar(
            tol=settings.newton_tol,
            max_iterations=45,
            linsolver=BK.COPLS(),
            verbose=false,
        );
        normN=BK.norminf,
    )
    BK.converged(corrected) || error(
        "periodic correction failed at gamma=$(float(gamma))")
    refined = merge(checkpoint, (solution=collect(float.(corrected.u)),))
    refined_wrapper, refined_parameters = _sax_periodic_wrapper(
        refined, model_p, bifurcation_settings)
    jacobian = BK.jacobian(
        refined_wrapper, refined.solution, refined_parameters)
    discretization = BK.get_discretization(refined_wrapper)
    nev = 2 + 2settings.nmodes
    pqz_values, _, pqz_converged, pqz_iterations = SaxFloquetPQZ(
        cyclic_retries=max(8, settings.collocation_intervals - 1),
        fallback_to_floquet_coll=false,
    )(discretization, jacobian, nev)
    coll_values, _, coll_converged, coll_iterations = BK.FloquetColl()(
        discretization, jacobian, nev)
    pqz = _sax_schur_spectrum_summary(
        pqz_values, expected, target_angle, settings;
        method=:periodic_schur,
        converged=Bool(pqz_converged),
        iterations=Int(pqz_iterations),
    )
    coll = _sax_schur_spectrum_summary(
        coll_values, expected, target_angle, settings;
        method=:floquet_coll,
        converged=Bool(coll_converged),
        iterations=Int(coll_iterations),
    )
    residual = float(norm(BK.residual(
        refined_wrapper, refined.solution, refined_parameters), Inf))
    return (
        checkpoint=refined,
        periodic_schur=pqz,
        floquet_coll=coll,
        orbit_residual=residual,
    )
end

function _sax_schur_refine_bracket(
        bracket,
        left_solution,
        right_solution,
        model_p::NamedTuple,
        bifurcation_settings::SaxBifurcationSettings,
        settings::SaxFixedZetaSchurSettings)
    bracket.classification in (:pd, :r2, :ns) || throw(ArgumentError(
        "only PD, R2, and NS brackets can be refined at fixed gamma"))
    expected = bracket.classification in (:pd, :r2) ? :pd : :ns
    abs(bracket.right_gamma - bracket.left_gamma) <= eps() &&
        error("zero-width gamma bracket")
    if bracket.left_gamma <= bracket.right_gamma
        first_gamma, second_gamma = bracket.left_gamma, bracket.right_gamma
        lower_solution, upper_solution = left_solution, right_solution
        lower_amplitude, upper_amplitude =
            bracket.left_amplitude, bracket.right_amplitude
    else
        first_gamma, second_gamma = bracket.right_gamma, bracket.left_gamma
        lower_solution, upper_solution = right_solution, left_solution
        lower_amplitude, upper_amplitude =
            bracket.right_amplitude, bracket.left_amplitude
    end
    initial_lower_gamma, initial_upper_gamma = first_gamma, second_gamma
    lower = _sax_schur_correct_orbit(
        lower_solution, first_gamma, bracket.mode, bracket.floquet_angle,
        expected, model_p, bifurcation_settings, settings)
    upper = _sax_schur_correct_orbit(
        upper_solution, second_gamma, bracket.mode, bracket.floquet_angle,
        expected, model_p, bifurcation_settings, settings)
    lower_growth = lower.periodic_schur.corrected_growth
    upper_growth = upper.periodic_schur.corrected_growth
    (signbit(lower_growth) != signbit(upper_growth) ||
     min(abs(lower_growth), abs(upper_growth)) <=
        settings.root_growth_tolerance) || error(
        "corrected PQZ growth no longer brackets zero")
    chosen = abs(lower_growth) <= abs(upper_growth) ? lower : upper
    iteration = 0
    for current_iteration in 1:settings.maximum_root_iterations
        iteration = current_iteration
        abs(chosen.periodic_schur.corrected_growth) <=
            settings.root_growth_tolerance &&
            second_gamma - first_gamma <= 10settings.root_gamma_tolerance && break
        middle_gamma = (first_gamma + second_gamma) / 2
        fraction = (middle_gamma - first_gamma) /
            (second_gamma - first_gamma)
        middle_guess = (1 - fraction) .* lower.checkpoint.solution .+
            fraction .* upper.checkpoint.solution
        middle = _sax_schur_correct_orbit(
            middle_guess, middle_gamma, bracket.mode, bracket.floquet_angle,
            expected, model_p, bifurcation_settings, settings)
        middle_growth = middle.periodic_schur.corrected_growth
        abs(middle_growth) < abs(chosen.periodic_schur.corrected_growth) &&
            (chosen = middle)
        if middle_growth == 0
            first_gamma = second_gamma = middle_gamma
            chosen = middle
            break
        elseif signbit(lower_growth) == signbit(middle_growth)
            first_gamma = middle_gamma
            lower = middle
            lower_growth = middle_growth
        else
            second_gamma = middle_gamma
            upper = middle
            upper_growth = middle_growth
        end
        second_gamma - first_gamma <= settings.root_gamma_tolerance && break
    end
    pqz = chosen.periodic_schur
    coll = chosen.floquet_coll
    methods_agree = abs(pqz.corrected_growth - coll.corrected_growth) <=
            settings.method_growth_tolerance &&
        abs(pqz.floquet_angle - coll.floquet_angle) <=
            settings.method_angle_tolerance
    accepted = pqz.converged && coll.converged &&
        chosen.orbit_residual <= 100settings.newton_tol &&
        abs(pqz.corrected_growth) <= settings.root_growth_tolerance &&
        methods_agree &&
        _sax_schur_expected_matches(expected, pqz.classification) &&
        _sax_schur_expected_matches(expected, coll.classification)
    refined_type = pqz.classification == :r2 ? :r2 : expected
    amplitude_fraction = (chosen.checkpoint.gamma - initial_lower_gamma) /
        max(initial_upper_gamma - initial_lower_gamma, eps())
    pressure_l2 = (1 - amplitude_fraction) * lower_amplitude +
        amplitude_fraction * upper_amplitude
    checkpoint = merge(chosen.checkpoint, (
        key="fixed_zeta_schur_$(refined_type)_m$(bracket.mode)_" *
            "g$(round(chosen.checkpoint.gamma; digits=9))",
        type=refined_type,
        floquet_angle=pqz.floquet_angle,
        localization_status=:pqz_refined,
        localization_precision=float(second_gamma - first_gamma),
    ))
    return (
        type=refined_type,
        mode=Int(bracket.mode),
        gamma=float(chosen.checkpoint.gamma),
        zeta=float(settings.zeta),
        pressure_l2=float(pressure_l2),
        period=float(last(chosen.checkpoint.solution)),
        localization_status=:pqz_refined,
        localization_precision=float(second_gamma - first_gamma),
        orbit_residual=chosen.orbit_residual,
        periodic_schur=pqz,
        floquet_coll=coll,
        methods_agree=methods_agree,
        fold_turn_agrees=true,
        accepted=accepted,
        checkpoint=checkpoint,
        source_bracket=bracket,
        refinement_iterations=iteration,
    )
end

function _sax_schur_root_duplicate(left, right,
                                   settings::SaxFixedZetaSchurSettings)
    compatible = left.type == right.type ||
        (left.type in (:pd, :r2) && right.type in (:pd, :r2))
    compatible || return false
    return abs(left.gamma - right.gamma) <= settings.event_gamma_tolerance
end

function _sax_merge_schur_roots(roots,
                                settings::SaxFixedZetaSchurSettings)
    merged = Any[]
    for root in roots
        index = findfirst(stored ->
            _sax_schur_root_duplicate(stored, root, settings), merged)
        if isnothing(index)
            push!(merged, root)
        elseif root.accepted && !merged[index].accepted
            merged[index] = root
        elseif root.type == :r2 && merged[index].type == :pd
            merged[index] = root
        end
    end
    sort!(merged; by=root -> root.gamma)
    return merged
end

function _sax_schur_bracket_event(
        bracket,
        samples,
        turns,
        settings::SaxFixedZetaSchurSettings)
    left, right = samples[bracket.left_index], samples[bracket.right_index]
    denominator = bracket.right_growth - bracket.left_growth
    fraction = abs(denominator) <= eps() ? 0.5 :
        clamp(-bracket.left_growth / denominator, 0.0, 1.0)
    gamma = (1 - fraction) * left.gamma + fraction * right.gamma
    amplitude = (1 - fraction) * left.pressure_l2 +
        fraction * right.pressure_l2
    nearest_turn = isempty(turns) ? nothing : turns[argmin(hypot(
        (turn.gamma - gamma) / max(settings.event_gamma_tolerance, eps()),
        (turn.pressure_l2 - amplitude) /
            max(settings.event_amplitude_tolerance, eps()),
    ) for turn in turns)]
    fold_like = bracket.classification in (:plus_one, :multiple_plus_one) &&
        !isnothing(nearest_turn) &&
        abs(nearest_turn.gamma - gamma) <= 5settings.event_gamma_tolerance
    type = fold_like ? :fold : bracket.classification
    return (
        type=type,
        mode=bracket.mode,
        direction=bracket.direction,
        gamma=float(gamma),
        gamma_interval=bracket.gamma_interval,
        pressure_l2=float(amplitude),
        floquet_angle=bracket.floquet_angle,
        corrected_growth=min(abs(bracket.left_growth),
                             abs(bracket.right_growth)),
        unstable_change=bracket.unstable_after - bracket.unstable_before,
        resonance=bracket.resonance,
        pair_multiplicity=bracket.pair_multiplicity,
        status=:bracket_only,
        accepted=false,
        source=:tracked_periodic_schur,
        track_id=bracket.track_id,
    )
end

function _sax_schur_events(
        brackets,
        roots,
        samples,
        turns,
        direction::Symbol,
        settings::SaxFixedZetaSchurSettings)
    events = Any[]
    used_roots = Set{Int}()
    for bracket in brackets
        candidate = _sax_schur_bracket_event(
            bracket, samples, turns, settings)
        compatible(root) =
            (candidate.type == :fold && root.type == :fold) ||
            (candidate.type in (:pd, :r2) && root.type in (:pd, :r2)) ||
            (candidate.type == :ns && root.type == :ns)
        indices = [index for (index, root) in enumerate(roots)
                   if compatible(root) &&
                      abs(root.gamma - candidate.gamma) <=
                          5settings.event_gamma_tolerance]
        if isempty(indices)
            push!(events, candidate)
            continue
        end
        root_index = indices[argmin(abs(roots[index].gamma - candidate.gamma)
                                    for index in indices)]
        root = roots[root_index]
        push!(used_roots, root_index)
        push!(events, merge(candidate, (
            type=root.type,
            gamma=root.gamma,
            pressure_l2=root.pressure_l2,
            floquet_angle=root.periodic_schur.floquet_angle,
            corrected_growth=root.periodic_schur.corrected_growth,
            resonance=root.periodic_schur.resonance,
            pair_multiplicity=root.periodic_schur.pair.multiplicity,
            status=root.accepted ? :validated : :localized_unvalidated,
            accepted=root.accepted,
            source=:bifurcationkit_localized_dual_validated,
            validation=root,
        )))
    end
    for (index, root) in enumerate(roots)
        index in used_roots && continue
        push!(events, (
            type=root.type,
            mode=root.mode,
            direction=direction,
            gamma=root.gamma,
            gamma_interval=(root.gamma, root.gamma),
            pressure_l2=root.pressure_l2,
            floquet_angle=root.periodic_schur.floquet_angle,
            corrected_growth=root.periodic_schur.corrected_growth,
            unstable_change=0,
            resonance=root.periodic_schur.resonance,
            pair_multiplicity=root.periodic_schur.pair.multiplicity,
            status=root.accepted ? :validated : :localized_unvalidated,
            accepted=root.accepted,
            source=:bifurcationkit_localized_dual_validated,
            track_id=0,
            validation=root,
        ))
    end
    sort!(events; by=event -> event.gamma)
    return events
end

function _sax_fixed_zeta_schur_component(
        model_p::NamedTuple,
        hopf,
        mode::Integer,
        direction::Symbol,
        settings::SaxFixedZetaSchurSettings;
        verbosity::Integer=1)
    bifurcation_settings = _sax_fixed_zeta_schur_bifurcation_settings(
        settings, direction)
    branch = continue_sax_periodic_orbits(
        hopf,
        model_p;
        settings=bifurcation_settings,
        verbosity=Int(verbosity),
        bothside=false,
        eigsolver=SaxFloquetPQZ(
            cyclic_retries=max(8, settings.collocation_intervals - 1),
            fallback_to_floquet_coll=false,
        ),
        detect_bifurcation=3,
        save_eigenvectors=true,
    )
    extracted = _sax_fixed_zeta_schur_samples(
        branch, mode, direction, settings)
    turns = _sax_schur_turn_candidates(extracted.samples, settings)
    brackets = _sax_schur_candidate_brackets(extracted.samples, settings)
    checkpoints = _sax_periodic_bifurcation_checkpoints(branch, hopf, mode)
    roots = Any[]
    root_failures = Any[]
    for bracket in brackets
        bracket.classification in (:pd, :r2, :ns) || continue
        if !haskey(extracted.solutions, bracket.left_step) ||
                !haskey(extracted.solutions, bracket.right_step)
            push!(root_failures, (
                type=bracket.classification,
                gamma=bracket.gamma,
                exception_type=:MissingSolution,
                error="saved solutions are absent at a PQZ bracket endpoint",
            ))
            continue
        end
        try
            push!(roots, _sax_schur_refine_bracket(
                bracket,
                extracted.solutions[bracket.left_step],
                extracted.solutions[bracket.right_step],
                model_p,
                bifurcation_settings,
                settings,
            ))
        catch err
            err isa InterruptException && rethrow()
            push!(root_failures, (
                type=bracket.classification,
                gamma=bracket.gamma,
                exception_type=Symbol(nameof(typeof(err))),
                error=sprint(showerror, err),
            ))
        end
    end
    for checkpoint in checkpoints
        try
            push!(roots, _sax_schur_validate_checkpoint(
                checkpoint,
                checkpoint.type,
                model_p,
                bifurcation_settings,
                settings,
                extracted.samples,
                turns,
            ))
        catch err
            err isa InterruptException && rethrow()
            push!(root_failures, (
                type=checkpoint.type,
                gamma=checkpoint.gamma,
                exception_type=Symbol(nameof(typeof(err))),
                error=sprint(showerror, err),
            ))
        end
    end
    roots = _sax_merge_schur_roots(roots, settings)
    events = _sax_schur_events(
        brackets, roots, extracted.samples, turns, direction, settings)
    return (
        analysis=:fixed_zeta_periodic_schur_component,
        mode=Int(mode),
        register=mode == 1 ? :low : mode == 2 ? :high : Symbol("mode$(mode)"),
        direction=direction,
        hopf=(gamma=float(hopf.gamma), zeta=float(hopf.zeta),
              frequency=float(hopf.frequency), mode=Int(mode)),
        samples=extracted.samples,
        brackets=brackets,
        roots=roots,
        events=events,
        turn_candidates=turns,
        diagnostic_events=_sax_schur_diagnostic_events(
            extracted.samples, settings),
        root_failures=root_failures,
        diagnostics=_sax_branch_terminal_diagnostics(branch, :gamma),
        counts=(
            samples=length(extracted.samples),
            brackets=length(brackets),
            localized_roots=length(roots),
            validated_roots=count(root -> root.accepted, roots),
            turns=length(turns),
            failures=length(root_failures),
        ),
    )
end

function _sax_schur_event_duplicate(left, right,
                                    settings::SaxFixedZetaSchurSettings)
    left.mode == right.mode || return false
    left.type == right.type || return false
    abs(left.gamma - right.gamma) <= settings.event_gamma_tolerance ||
        return false
    amplitude_tolerance = left.direction == right.direction ?
        settings.event_amplitude_tolerance :
        3 * settings.event_amplitude_tolerance
    return abs(left.pressure_l2 - right.pressure_l2) <=
        amplitude_tolerance
end

function _sax_merge_schur_events(components,
                                 settings::SaxFixedZetaSchurSettings)
    events = Any[]
    for component in components, event in component.events
        event.pressure_l2 >= settings.minimum_event_amplitude || continue
        duplicate_index = findfirst(stored ->
            _sax_schur_event_duplicate(stored, event, settings), events)
        if isnothing(duplicate_index)
            push!(events, event)
        elseif event.accepted && !events[duplicate_index].accepted
            events[duplicate_index] = event
        end
    end
    sort!(events; by=event -> (event.gamma, event.mode, String(event.type)))
    return events
end

"""
    compute_sax_fixed_zeta_schur(model, root; settings, resume=true)

Run four independent directional P1 continuations ending at H1 and H2. Every
accepted continuation step uses generalized Periodic Schur, every neutral
Floquet sign change is retained, and each BifurcationKit-localized event is
independently checked with both Periodic Schur and FloquetColl.
"""
function compute_sax_fixed_zeta_schur(
        model_p::NamedTuple,
        root::AbstractString;
        settings::SaxFixedZetaSchurSettings=SaxFixedZetaSchurSettings(),
        resume::Bool=true,
        verbosity::Integer=1)
    _validate_sax_fixed_zeta_schur_settings(settings)
    model = _sax_fixed_zeta_schur_model(model_p, settings)
    paths = sax_fixed_zeta_schur_paths(root; settings=settings)
    mkpath(paths.directory)
    hopf_settings = _sax_fixed_zeta_schur_hopf_settings(settings)
    failures = Any[]
    for (mode, gamma_hint) in zip(settings.modes,
                                  settings.hopf_gamma_hints)
        pending = Symbol[]
        for direction in settings.directions
            cached = resume ? _load_sax_fixed_zeta_schur_cache(
                paths.component(mode, direction), model, settings) :
                (status=:missing, payload=nothing, reason="resume disabled")
            cached.status == :valid || push!(pending, direction)
        end
        isempty(pending) && continue
        hopf = refine_sax_hopf_checkpoint(
            model,
            settings.zeta,
            mode;
            settings=hopf_settings,
            gamma_hint=gamma_hint,
        )
        for direction in pending
            verbosity > 0 && @info(
                "Fixed-zeta Periodic-Schur component started",
                mode,
                register=mode == 1 ? :low : :high,
                direction,
                zeta=settings.zeta,
                gamma_range=settings.gamma_range,
                collocation="$(settings.collocation_intervals)x$(settings.collocation_degree)",
            )
            try
                component = _sax_fixed_zeta_schur_component(
                    model, hopf, mode, direction, settings;
                    verbosity=verbosity)
                _save_sax_fixed_zeta_schur_cache(
                    paths.component(mode, direction), component, model, settings)
                verbosity > 0 && @info(
                    "Fixed-zeta Periodic-Schur component completed",
                    mode,
                    direction,
                    component.counts,
                )
            catch err
                err isa InterruptException && rethrow()
                failure = (
                    mode=Int(mode),
                    direction=direction,
                    exception_type=Symbol(nameof(typeof(err))),
                    error=sprint(showerror, err),
                )
                push!(failures, failure)
                @warn "Fixed-zeta Periodic-Schur component failed" failure
            end
        end
    end
    progress = load_sax_fixed_zeta_schur_progress(
        model_p, root; settings=settings)
    manifest = (
        analysis=:fixed_zeta_periodic_schur,
        status=progress.status,
        counts=progress.counts,
        failures=failures,
        saved_at_unix=time(),
        settings=_portable_sax_fixed_zeta_schur_settings(settings),
    )
    _atomic_jld2_save(paths.manifest; manifest)
    return progress
end

"""Load completed directional components while another process is running."""
function load_sax_fixed_zeta_schur_progress(
        model_p::NamedTuple,
        root::AbstractString;
        settings::SaxFixedZetaSchurSettings=SaxFixedZetaSchurSettings())
    _validate_sax_fixed_zeta_schur_settings(settings)
    model = _sax_fixed_zeta_schur_model(model_p, settings)
    paths = sax_fixed_zeta_schur_paths(root; settings=settings)
    components = Any[]
    statuses = Any[]
    for mode in settings.modes, direction in settings.directions
        loaded = _load_sax_fixed_zeta_schur_cache(
            paths.component(mode, direction), model, settings)
        push!(statuses, (
            mode=Int(mode),
            register=mode == 1 ? :low : :high,
            direction=direction,
            status=loaded.status,
            reason=loaded.reason,
            path=paths.component(mode, direction),
        ))
        loaded.status == :valid && push!(components, loaded.payload)
    end
    expected = length(settings.modes) * length(settings.directions)
    status = length(components) == expected ? :complete :
        isempty(components) ? :missing : :partial
    events = _sax_merge_schur_events(components, settings)
    return (
        analysis=:fixed_zeta_periodic_schur,
        status=status,
        settings=_portable_sax_fixed_zeta_schur_settings(settings),
        regularization=hasproperty(model, :sax_regularization) ?
            model.sax_regularization : nothing,
        paths=paths,
        components=components,
        component_status=statuses,
        events=events,
        diagnostic_events=Any[event for component in components
                              for event in component.diagnostic_events],
        turn_candidates=Any[turn for component in components
                            for turn in component.turn_candidates],
        counts=(
            components=length(components),
            expected_components=expected,
            samples=sum((component.counts.samples for component in components);
                        init=0),
            events=length(events),
            validated_events=count(event -> event.accepted, events),
            bracket_only=count(event -> event.status == :bracket_only, events),
            turns=sum((component.counts.turns for component in components);
                      init=0),
            failures=sum((component.counts.failures for component in components);
                         init=0),
        ),
    )
end

"""Return a compact, gamma-ordered event table for notebook display."""
function sax_fixed_zeta_schur_events(
        progress;
        gamma_limits::Tuple{<:Real,<:Real}=(0.30, 1.00))
    lower, upper = float.(gamma_limits)
    rows = [event for event in progress.events
            if lower <= event.gamma <= upper]
    sort!(rows; by=event -> (event.gamma, event.mode, String(event.type)))
    return rows
end

"""
    sax_fixed_zeta_unique_schur_components(progress)

Return one authoritative P1 continuation per register mode. Positive and
negative Hopf predictors normally converge to the same geometric family; the
component with the denser saved mesh is retained and the discarded direction
is reported. This removes a plotting and product-assembly duplicate without
discarding either checkpoint on disk.
"""
function sax_fixed_zeta_unique_schur_components(progress)
    retained = Any[]
    duplicates = Any[]
    modes = sort(unique(Int(component.mode) for component in progress.components))
    for mode in modes
        candidates = [component for component in progress.components
                      if Int(component.mode) == mode]
        ordered = sort(candidates; by=component -> length(component.samples),
                       rev=true)
        isempty(ordered) && continue
        selected = first(ordered)
        push!(retained, selected)
        append!(duplicates, (
            mode=mode,
            discarded_direction=component.direction,
            retained_direction=selected.direction,
        ) for component in Iterators.drop(ordered, 1))
    end
    return (components=retained, duplicates=duplicates)
end

function _sax_schur_plot_segments!(axis, samples, register_color;
                                   show_labels::Bool=false)
    isempty(samples) && return axis
    start = 1
    labels = Dict(true => false, false => false)
    for stop in 2:(length(samples) + 1)
        split = stop == length(samples) + 1 ||
            samples[stop].stable != samples[stop - 1].stable
        split || continue
        indices = start:(stop - 1)
        stable = samples[start].stable
        label = !show_labels || labels[stable] ? "" :
            stable ? "PQZ stable" : "PQZ unstable"
        plot!(
            axis,
            getproperty.(samples[indices], :gamma),
            getproperty.(samples[indices], :pressure_l2);
            color=stable ? register_color : "#4D4D4D",
            linewidth=stable ? 3.0 : 2.4,
            linestyle=stable ? :solid : :dash,
            label=label,
        )
        labels[stable] = true
        start = stop
    end
    return axis
end

function _sax_schur_event_style(type::Symbol)
    type == :fold && return (marker=:rect, color="#CC79A7", label="Fold")
    type == :pd && return (marker=:utriangle, color="#E69F00", label="PD")
    type == :r2 && return (marker=:dtriangle, color="#66C2A5", label="R2")
    type == :ns && return (marker=:circle, color="#009E73", label="NS")
    type in (:plus_one, :multiple_plus_one) &&
        return (marker=:rect, color="#999999", label="+1 candidate")
    return (marker=:circle, color="#999999", label=String(type))
end

function _sax_schur_plot_events!(axis, events; yfield::Symbol=:pressure_l2)
    labels_used = Set{String}()
    order = (:fold, :pd, :r2, :ns, :plus_one, :multiple_plus_one)
    for type in order
        selected = [event for event in events if event.type == type]
        isempty(selected) && continue
        style = _sax_schur_event_style(type)
        label = style.label in labels_used ? "" : style.label
        push!(labels_used, style.label)
        scatter!(
            axis,
            getproperty.(selected, :gamma),
            yfield == :zero ? zeros(length(selected)) :
                getproperty.(selected, yfield);
            marker=style.marker,
            markersize=7,
            markercolor=[event.accepted ? style.color : :white
                         for event in selected],
            markerstrokecolor=style.color,
            markerstrokewidth=1.5,
            label=label,
        )
    end
    return axis
end

function _sax_schur_amplitude_panel(progress, mode::Integer, gamma_limits)
    color = mode == 1 ? "#D55E00" : "#0072B2"
    title = mode == 1 ? "Low register P1 ending at H1" :
        "High register P1 ending at H2"
    axis = plot(
        xlabel="gamma",
        ylabel="L2 pressure amplitude",
        xlims=gamma_limits,
        title=title,
        framestyle=:box,
        grid=false,
        legend=:topright,
    )
    components = [component for component in
                  sax_fixed_zeta_unique_schur_components(progress).components
                  if component.mode == mode]
    first_component = true
    for component in components
        _sax_schur_plot_segments!(
            axis, component.samples, color; show_labels=first_component)
        first_component = false
    end
    hopf = isempty(components) ? nothing : components[1].hopf
    if !isnothing(hopf) && gamma_limits[1] <= hopf.gamma <= gamma_limits[2]
        scatter!(axis, [hopf.gamma], [0.0]; marker=:diamond, markersize=7,
                 markercolor=color, markerstrokecolor=:black,
                 label="H$(mode)")
    end
    events = [event for event in progress.events
              if event.mode == mode && gamma_limits[1] <= event.gamma <=
                 gamma_limits[2]]
    _sax_schur_plot_events!(axis, events)
    return axis
end

function _sax_schur_growth_panel(progress, mode::Integer, gamma_limits,
                                 growth_limits)
    color = mode == 1 ? "#D55E00" : "#0072B2"
    axis = plot(
        xlabel="gamma",
        ylabel="corrected Floquet growth",
        xlims=gamma_limits,
        ylims=growth_limits,
        framestyle=:box,
        grid=false,
        legend=false,
    )
    hline!(axis, [0.0]; color=:black, linewidth=1.2, label="")
    for component in sax_fixed_zeta_unique_schur_components(progress).components
        component.mode == mode || continue
        track_ids = sort(unique(pair.track_id for sample in component.samples
                                for pair in sample.pairs))
        for track_id in track_ids
            samples = Any[]
            growth = Float64[]
            for sample in component.samples
                pair = _sax_schur_pair_by_track(sample, track_id)
                isnothing(pair) && continue
                push!(samples, sample)
                push!(growth, pair.growth)
            end
            length(samples) < 2 && continue
            plot!(axis, getproperty.(samples, :gamma), growth;
                  color="#999999", alpha=0.45, linewidth=1.0, label="")
        end
        plot!(axis,
              getproperty.(component.samples, :gamma),
              getproperty.(component.samples, :dominant_growth);
              color=color, linewidth=2.2, label="")
    end
    events = [event for event in progress.events
              if event.mode == mode && gamma_limits[1] <= event.gamma <=
                 gamma_limits[2]]
    _sax_schur_plot_events!(axis, events; yfield=:zero)
    return axis
end

"""
    plot_sax_fixed_zeta_schur_audit(progress; ...)

Plot low- and high-register P1 amplitude branches together with the complete
Periodic-Schur spectrum. Filled event markers passed the dual PQZ/FloquetColl
check; open markers are bracket-only or localized but unvalidated candidates.
"""
function plot_sax_fixed_zeta_schur_audit(
        progress;
        gamma_limits::Tuple{<:Real,<:Real}=(0.30, 1.00),
        growth_limits::Tuple{<:Real,<:Real}=(-0.08, 0.08),
        size=(1200, 780))
    limits = Tuple(float.(gamma_limits))
    growth = Tuple(float.(growth_limits))
    low_amplitude = _sax_schur_amplitude_panel(progress, 1, limits)
    high_amplitude = _sax_schur_amplitude_panel(progress, 2, limits)
    low_growth = _sax_schur_growth_panel(progress, 1, limits, growth)
    high_growth = _sax_schur_growth_panel(progress, 2, limits, growth)
    return plot(
        low_amplitude,
        high_amplitude,
        low_growth,
        high_growth;
        layout=(2, 2),
        size=size,
        link=:x,
    )
end

function _sax_fixed_zeta_hopf_events(progress)
    events = Any[]
    for component in progress.components
        any(event -> event.mode == component.mode, events) && continue
        push!(events, (
            type=:hopf,
            mode=Int(component.mode),
            gamma=float(component.hopf.gamma),
            zeta=float(component.hopf.zeta),
            pressure_l2=0.0,
            status=:validated,
            accepted=true,
            source=:equilibrium_hopf,
        ))
    end
    sort!(events; by=event -> event.mode)
    return events
end

function _sax_plot_schur_register_segments!(
        axis,
        samples,
        color;
        linewidth::Real=3.2,
        alpha::Real=0.95)
    isempty(samples) && return axis
    start = 1
    for stop in 2:(length(samples) + 1)
        split = stop == length(samples) + 1 ||
            samples[stop].stable != samples[stop - 1].stable
        split || continue
        indices = start:(stop - 1)
        stable = samples[start].stable
        plot!(
            axis,
            getproperty.(samples[indices], :gamma),
            getproperty.(samples[indices], :pressure_l2);
            color=color,
            linewidth=stable ? linewidth : 0.86linewidth,
            linestyle=stable ? :solid : :dash,
            alpha=alpha,
            label="",
        )
        start = stop
    end
    return axis
end

function _sax_plot_available_p2_segments!(
        axis,
        curve,
        color;
        linewidth::Real=2.2,
        alpha::Real=0.55)
    isempty(curve.gamma) && return axis
    start = 1
    for stop in 2:(length(curve.gamma) + 1)
        split = stop == length(curve.gamma) + 1 ||
            curve.stable[stop] != curve.stable[stop - 1]
        split || continue
        indices = start:(stop - 1)
        stable = curve.stable[start] === true
        plot!(
            axis,
            curve.gamma[indices],
            curve.pressure_l2[indices];
            color=color,
            linewidth=stable ? linewidth : 0.86linewidth,
            linestyle=stable ? :solid : :dot,
            alpha=alpha,
            label="",
        )
        start = stop
    end
    return axis
end

"""
    plot_sax_fixed_zeta_schur_diagram(progress; amplitude_progress=nothing, ...)

Build a single fixed-`zeta` amplitude diagram from the Periodic-Schur P1
branches. Register is encoded by color and Periodic-Schur stability by line
style. If the canonical amplitude cache is supplied, its P2 branches are also
included; that cache uses the same strict generalized Periodic-Schur solver for
P2 stability and event detection.
"""
function plot_sax_fixed_zeta_schur_diagram(
        progress;
        amplitude_progress=nothing,
        gamma_limits::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
        include_period_two::Bool=true,
        show_candidates::Bool=true,
        title::AbstractString="",
        legend=:topright,
        size=(980, 620))
    limits = isnothing(gamma_limits) ?
        Tuple(float.(progress.settings.gamma_range)) :
        Tuple(float.(gamma_limits))
    colors = (
        low="#D73027",
        high="#2166AC",
        p2="#762A83",
        neutral="#333333",
    )
    amplitudes = Float64[]
    unique_components =
        sax_fixed_zeta_unique_schur_components(progress).components
    for component in unique_components, sample in component.samples
        limits[1] <= sample.gamma <= limits[2] || continue
        push!(amplitudes, sample.pressure_l2)
    end
    p2_branches = isnothing(amplitude_progress) ||
        !include_period_two ? Any[] : amplitude_progress.p2_branches
    for branch in p2_branches, (gamma, amplitude) in
            zip(branch.gamma, branch.pressure_l2)
        limits[1] <= gamma <= limits[2] || continue
        push!(amplitudes, float(amplitude))
    end
    maximum_amplitude = maximum(amplitudes; init=1e-6)
    axis = plot(
        xlabel="γ",
        ylabel="L² pressure amplitude",
        xlims=limits,
        ylims=(-0.035maximum_amplitude, 1.06maximum_amplitude),
        title=title,
        legend=legend,
        size=size,
        framestyle=:box,
        grid=false,
    )
    hline!(axis, [0.0]; color=colors.neutral, linewidth=1.2,
           alpha=0.7, label="Equilibrium")
    plot!(axis, [NaN], [NaN]; color=colors.low, linewidth=3.2,
          label="Low-register P1")
    plot!(axis, [NaN], [NaN]; color=colors.high, linewidth=3.2,
          label="High-register P1")
    if !isempty(p2_branches)
        plot!(axis, [NaN], [NaN]; color=colors.p2, linewidth=2.2,
              alpha=0.6, label="Available P2")
    end
    plot!(axis, [NaN], [NaN]; color=colors.neutral, linewidth=2.6,
          linestyle=:solid, label="Stable")
    plot!(axis, [NaN], [NaN]; color=colors.neutral, linewidth=2.3,
          linestyle=:dash, label="Unstable")
    for component in unique_components
        color = component.mode == 1 ? colors.low : colors.high
        _sax_plot_schur_register_segments!(
            axis, component.samples, color)
    end
    for branch in p2_branches
        _sax_plot_available_p2_segments!(axis, branch, colors.p2)
    end
    for hopf in _sax_fixed_zeta_hopf_events(progress)
        limits[1] <= hopf.gamma <= limits[2] || continue
        color = hopf.mode == 1 ? colors.low : colors.high
        scatter!(axis, [hopf.gamma], [0.0]; marker=:diamond,
                 markersize=7, markercolor=color,
                 markerstrokecolor=:black, markerstrokewidth=0.8,
                 label="H$(hopf.mode)")
    end
    events = [event for event in progress.events
              if limits[1] <= event.gamma <= limits[2] &&
                 (show_candidates || event.accepted)]
    _sax_schur_plot_events!(axis, events)
    return axis
end

function _sax_curve_zeta_crossings(
        curves,
        target_zeta::Real,
        mode::Integer;
        duplicate_tolerance::Real=1e-4)
    crossings = Float64[]
    target = float(target_zeta)
    for curve in curves
        hasproperty(curve, :mode) && Int(curve.mode) != mode && continue
        gamma = curve.gamma
        zeta = curve.zeta
        for index in 1:(length(zeta) - 1)
            first_offset = float(zeta[index]) - target
            second_offset = float(zeta[index + 1]) - target
            if first_offset == 0
                candidate = float(gamma[index])
            elseif second_offset == 0
                candidate = float(gamma[index + 1])
            elseif signbit(first_offset) == signbit(second_offset)
                continue
            else
                fraction = first_offset / (first_offset - second_offset)
                candidate = float(gamma[index] +
                    fraction * (gamma[index + 1] - gamma[index]))
            end
            any(value -> abs(value - candidate) <= duplicate_tolerance,
                crossings) || push!(crossings, candidate)
        end
    end
    sort!(crossings)
    return crossings
end

function _sax_schur_plane_curves(result, type::Symbol)
    type == :hopf && return result.hopf_curves
    type == :fold && return result.fold_curves
    type in (:pd, :r2) && return result.pd_curves
    type == :ns && return result.ns_curves
    return Any[]
end

"""
    sax_fixed_zeta_plane_compatibility(progress, result; ...)

Compare every fixed-`zeta` H1/H2 and P1 Floquet event with intersections of
the corresponding two-parameter curve. `:matched` means that a curve of the
same family and mode crosses the slice within `match_tolerance`; it is a
numerical consistency test, not an independent proof of curve connectivity.
"""
function sax_fixed_zeta_plane_compatibility(
        progress,
        result;
        gamma_limits::Tuple{<:Real,<:Real}=(0.30, 0.99),
        match_tolerance::Real=2e-3)
    limits = Tuple(float.(gamma_limits))
    slice_events = Any[_sax_fixed_zeta_hopf_events(progress)...;
                       progress.events...]
    rows = Any[]
    for event in slice_events
        limits[1] <= event.gamma <= limits[2] || continue
        plane_type = event.type in (:plus_one, :multiple_plus_one) ?
            :fold : event.type
        curves = _sax_schur_plane_curves(result, plane_type)
        crossings = _sax_curve_zeta_crossings(
            curves, progress.settings.zeta, Int(event.mode))
        nearest = isempty(crossings) ? missing :
            crossings[argmin(abs(value - event.gamma) for value in crossings)]
        difference = ismissing(nearest) ? missing :
            abs(float(nearest) - float(event.gamma))
        plane_status = ismissing(nearest) ?
            (event.accepted ? :not_continued : :candidate_only) :
            difference <= match_tolerance ? :matched : :unmatched
        seed_available = event.type != :hopf && event.accepted &&
            hasproperty(event, :validation) &&
            hasproperty(event.validation, :checkpoint)
        push!(rows, (
            family=event.type == :hopf ? Symbol("H$(event.mode)") :
                event.type,
            mode=Int(event.mode),
            fixed_gamma=float(event.gamma),
            plane_gamma=nearest,
            absolute_difference=difference,
            fixed_status=event.status,
            plane_status=plane_status,
            continuation_seed=seed_available,
        ))
    end
    sort!(rows; by=row -> (row.fixed_gamma, row.mode, String(row.family)))
    return rows
end

"""
    sax_fixed_zeta_continuation_seeds(progress; ...)

Return dual-validated periodic checkpoints that can be passed directly to
`continue_sax_periodic_bifurcation_curve(checkpoint, model; settings=...)`.
Only validated fold, PD, and NS events are returned; bracket-only candidates
are never promoted to two-parameter seeds.
"""
function sax_fixed_zeta_continuation_seeds(
        progress;
        gamma_limits::Tuple{<:Real,<:Real}=(0.30, 0.99),
        types::Tuple{Vararg{Symbol}}=(:fold, :pd, :ns))
    limits = Tuple(float.(gamma_limits))
    seeds = Any[]
    for event in progress.events
        event.accepted || continue
        event.type in types || continue
        limits[1] <= event.gamma <= limits[2] || continue
        hasproperty(event, :validation) || continue
        hasproperty(event.validation, :checkpoint) || continue
        checkpoint = event.validation.checkpoint
        duplicate = any(seed ->
            seed.type == event.type && seed.mode == event.mode &&
            abs(seed.gamma - event.gamma) <= 1e-4,
            seeds,
        )
        duplicate && continue
        push!(seeds, (
            type=event.type,
            mode=Int(event.mode),
            gamma=float(event.gamma),
            zeta=float(progress.settings.zeta),
            floquet_angle=float(event.floquet_angle),
            checkpoint=checkpoint,
        ))
    end
    sort!(seeds; by=seed -> seed.gamma)
    return seeds
end

"""
    overlay_sax_fixed_zeta_schur_slice!(axis, progress; ...)

Overlay the fixed-`zeta` slice and its validated or candidate Floquet events on
an existing `(gamma,zeta)` bifurcation diagram. Filled black circles are
dual-validated slice events and open circles are unvalidated candidates.
"""
function overlay_sax_fixed_zeta_schur_slice!(
        axis,
        progress;
        gamma_limits::Tuple{<:Real,<:Real}=(0.30, 0.99),
        show_candidates::Bool=true,
        show_hopf::Bool=true)
    limits = Tuple(float.(gamma_limits))
    zeta = float(progress.settings.zeta)
    hline!(axis, [zeta]; color="#555555", linewidth=1.5,
           linestyle=:dash, alpha=0.65,
           label="fixed ζ=$(round(zeta; digits=3))")
    events = Any[]
    show_hopf && append!(events, _sax_fixed_zeta_hopf_events(progress))
    append!(events, progress.events)
    selected = [event for event in events
                if limits[1] <= event.gamma <= limits[2] &&
                   (event.accepted || show_candidates)]
    validated = [event for event in selected if event.accepted]
    candidates = [event for event in selected if !event.accepted]
    if !isempty(validated)
        scatter!(axis, getproperty.(validated, :gamma),
                 fill(zeta, length(validated)); marker=:circle,
                 markersize=6, markercolor="#222222",
                 markerstrokecolor=:white, markerstrokewidth=0.8,
                 label="validated slice event")
    end
    if !isempty(candidates)
        scatter!(axis, getproperty.(candidates, :gamma),
                 fill(zeta, length(candidates)); marker=:circle,
                 markersize=6, markercolor=:white,
                 markerstrokecolor="#222222", markerstrokewidth=1.3,
                 label="slice candidate")
    end
    return axis
end
