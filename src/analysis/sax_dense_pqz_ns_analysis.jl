# Dense fixed-zeta Periodic-Schur analysis of the high-gamma Floquet boundary.
#
# This is deliberately separate from the historical event-localization code in
# `sax_periodic_schur_ns_analysis.jl`.  BifurcationKit is used only to continue
# and correct periodic orbits here.  Its bifurcation bisection is disabled:
# every PQZ Floquet pair is saved and tracked, sign changes are bracketed from
# accepted continuation steps, and each bracket is refined independently.

const SAX_DENSE_PQZ_NS_SCHEMA_VERSION = 1

"""Numerical controls for dense, event-free fixed-zeta PQZ slicing."""
Base.@kwdef struct SaxDensePQZNSSettings
    schema_version::Int = SAX_DENSE_PQZ_NS_SCHEMA_VERSION
    profile::Symbol = :final
    coverage::Symbol = :focused
    nmodes::Int = 8
    mode::Int = 2
    gamma_hint::Float64 = 0.6043477221884095
    gamma_range::Tuple{Float64,Float64} = (0.30, 0.72)
    root_gamma_range::Tuple{Float64,Float64} = (0.52, 0.72)
    zeta_range::Tuple{Float64,Float64} = (0.18, 0.50)
    zeta_step::Float64 = 0.01
    adaptive_levels::Int = 0
    adaptive_min_step::Float64 = 0.005
    adaptive_gamma_jump::Float64 = 0.012
    collocation_intervals::Int = 40
    collocation_degree::Int = 4
    validation_meshes::Tuple{Vararg{Tuple{Int,Int}}} = ((40, 4), (60, 5))
    po_ds::Float64 = 5e-4
    po_dsmax::Float64 = 1e-3
    po_max_steps::Int = 850
    hopf_scan_points::Int = 241
    newton_tol::Float64 = 1e-10
    newton_max_iterations::Int = 45
    stability_tol::Float64 = 1e-8
    root_growth_tolerance::Float64 = 2e-5
    root_gamma_tolerance::Float64 = 2e-6
    mesh_gamma_tolerance::Float64 = 5e-4
    mesh_angle_tolerance::Float64 = 2e-2
    method_growth_tolerance::Float64 = 2e-7
    method_angle_tolerance::Float64 = 2e-7
    minimum_ns_angle::Float64 = 1e-3
    minimum_angle_to_pi::Float64 = 1e-3
    r2_warning_angle::Float64 = 0.08
    maximum_pair_angle_jump::Float64 = 0.35
    maximum_pair_growth_jump::Float64 = 0.10
    maximum_gamma_bracket::Float64 = 0.01
    maximum_root_iterations::Int = 32
end

function _sax_regular_grid(bounds::Tuple{<:Real,<:Real}, step::Real)
    lower, upper = float.(bounds)
    values = collect(lower:float(step):upper)
    isempty(values) && push!(values, lower)
    isapprox(values[end], upper; atol=64eps(max(abs(upper), 1.0))) ||
        push!(values, upper)
    return Tuple(unique(round.(values; digits=12)))
end

"""
    sax_dense_pqz_ns_settings(profile=:final; coverage=:focused, ...)

Create the settings used by the replacement runner.  `coverage=:focused`
uses zeta in `[0.18, 0.50]`.  `coverage=:full` follows the same mode-2
periodic family from its Hopf point over the model interval
zeta interval `[0.001, 0.99]`.  Explicit bounds or spacing can be supplied later
without modifying the analysis code.
"""
function sax_dense_pqz_ns_settings(
        profile::Symbol=:final;
        coverage::Symbol=:focused,
        zeta_range::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
        zeta_step::Union{Nothing,Real}=nothing)
    profile in (:smoke, :pilot, :final) || throw(ArgumentError(
        "dense PQZ profile must be :smoke, :pilot, or :final",
    ))
    coverage in (:focused, :full) || throw(ArgumentError(
        "dense PQZ coverage must be :focused or :full",
    ))
    default_range = coverage == :focused ? (0.18, 0.50) : (0.001, 0.99)
    default_step = if profile == :smoke
        1.0
    elseif profile == :pilot
        coverage == :focused ? 0.04 : 0.08
    else
        coverage == :focused ? 0.01 : 0.025
    end
    selected_range = isnothing(zeta_range) ? default_range :
        Tuple(float.(zeta_range))
    selected_step = isnothing(zeta_step) ? default_step : float(zeta_step)
    if profile == :smoke
        return SaxDensePQZNSSettings(
            profile=profile,
            coverage=coverage,
            gamma_range=(0.30, 0.64),
            root_gamma_range=(0.58, 0.63),
            zeta_range=isnothing(zeta_range) ? (0.25, 0.25) : selected_range,
            zeta_step=selected_step,
            collocation_intervals=25,
            collocation_degree=3,
            validation_meshes=((25, 3), (40, 4)),
            po_ds=2e-3,
            po_dsmax=5e-3,
            po_max_steps=120,
            hopf_scan_points=101,
            maximum_gamma_bracket=0.02,
            adaptive_levels=0,
        )
    elseif profile == :pilot
        return SaxDensePQZNSSettings(
            profile=profile,
            coverage=coverage,
            zeta_range=selected_range,
            zeta_step=selected_step,
            collocation_intervals=25,
            collocation_degree=3,
            validation_meshes=((25, 3), (40, 4)),
            po_ds=1e-3,
            po_dsmax=3e-3,
            po_max_steps=400,
            hopf_scan_points=161,
            adaptive_levels=coverage == :full ? 1 : 0,
            adaptive_min_step=selected_step / 2,
        )
    end
    return SaxDensePQZNSSettings(
        profile=profile,
        coverage=coverage,
        zeta_range=selected_range,
        zeta_step=selected_step,
        adaptive_levels=coverage == :full ? 2 : 0,
        adaptive_min_step=selected_step / 4,
    )
end

function _validate_sax_dense_pqz_ns_settings(settings::SaxDensePQZNSSettings)
    settings.schema_version == SAX_DENSE_PQZ_NS_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported dense PQZ schema version"))
    settings.profile in (:smoke, :pilot, :final) ||
        throw(ArgumentError("invalid dense PQZ profile"))
    settings.coverage in (:focused, :full) ||
        throw(ArgumentError("invalid dense PQZ coverage"))
    1 <= settings.mode <= settings.nmodes ||
        throw(ArgumentError("dense PQZ mode must lie in 1:nmodes"))
    settings.gamma_range[1] < settings.gamma_range[2] ||
        throw(ArgumentError("gamma_range must be increasing"))
    settings.gamma_range[1] <= settings.root_gamma_range[1] <
        settings.root_gamma_range[2] <= settings.gamma_range[2] ||
        throw(ArgumentError("root_gamma_range must lie inside gamma_range"))
    settings.zeta_range[1] <= settings.zeta_range[2] ||
        throw(ArgumentError("zeta_range must be nondecreasing"))
    0.001 <= settings.zeta_range[1] && settings.zeta_range[2] <= 0.99 ||
        throw(ArgumentError("zeta_range must lie inside [0.001, 0.99]"))
    settings.zeta_step > 0 || throw(ArgumentError("zeta_step must be positive"))
    settings.adaptive_levels >= 0 ||
        throw(ArgumentError("adaptive_levels must be nonnegative"))
    settings.adaptive_min_step > 0 ||
        throw(ArgumentError("adaptive_min_step must be positive"))
    settings.maximum_root_iterations > 0 ||
        throw(ArgumentError("maximum_root_iterations must be positive"))
    all(mesh -> mesh[1] >= 5 && mesh[2] >= 2,
        settings.validation_meshes) ||
        throw(ArgumentError("dense PQZ validation meshes are too small"))
    return settings
end

sax_dense_pqz_zeta_values(settings::SaxDensePQZNSSettings) =
    _sax_regular_grid(settings.zeta_range, settings.zeta_step)

function _sax_dense_legacy_settings(settings::SaxDensePQZNSSettings)
    return SaxPeriodicSchurNSSettings(
        nmodes=settings.nmodes,
        mode=settings.mode,
        gamma_hint=settings.gamma_hint,
        gamma_range=settings.gamma_range,
        root_gamma_range=settings.root_gamma_range,
        zeta_range=settings.zeta_range,
        zeta_values=sax_dense_pqz_zeta_values(settings),
        validation_meshes=settings.validation_meshes,
        collocation_intervals=settings.collocation_intervals,
        collocation_degree=settings.collocation_degree,
        po_ds=settings.po_ds,
        po_dsmax=settings.po_dsmax,
        po_max_steps=settings.po_max_steps,
        hopf_scan_points=settings.hopf_scan_points,
        newton_tol=settings.newton_tol,
        newton_max_iterations=settings.newton_max_iterations,
        stability_tol=settings.stability_tol,
        root_growth_tolerance=settings.root_growth_tolerance,
        method_growth_tolerance=settings.method_growth_tolerance,
        method_angle_tolerance=settings.method_angle_tolerance,
        minimum_ns_angle=settings.minimum_ns_angle,
        minimum_angle_to_pi=settings.minimum_angle_to_pi,
        r2_warning_angle=settings.r2_warning_angle,
    )
end

function _sax_dense_bifurcation_settings(
        settings::SaxDensePQZNSSettings;
        intervals::Integer=settings.collocation_intervals,
        degree::Integer=settings.collocation_degree,
        direction::Integer=1)
    return _sax_mechanism_bifurcation_settings(
        settings.nmodes,
        intervals,
        degree;
        gamma_range=settings.gamma_range,
        zeta_range=(0.001, 0.99),
        po_ds=sign(direction) * abs(settings.po_ds),
        po_dsmax=settings.po_dsmax,
        po_max_steps=settings.po_max_steps,
        po_save_sol_every_step=1,
        newton_tol=settings.newton_tol,
        stability_tol=settings.stability_tol,
    )
end

function _sax_dense_rescue_settings(settings::SaxDensePQZNSSettings,
                                    zeta::Real)
    return SaxPDRescueSettings(
        nmodes=settings.nmodes,
        gamma_range=settings.gamma_range,
        zeta_range=(0.001, 0.99),
        seed_zetas=(float(zeta),),
        po_collocation_intervals=settings.collocation_intervals,
        po_collocation_degree=settings.collocation_degree,
        po_linear_solver=:condensed,
        po_ds=settings.po_ds,
        po_dsmax=settings.po_dsmax,
        po_max_steps=settings.po_max_steps,
        po_save_sol_every_step=1,
        newton_tol=settings.newton_tol,
        stability_tol=settings.stability_tol,
        hopf_scan_points=settings.hopf_scan_points,
    )
end

function _sax_dense_periodic_flow_guard(
        verbosity::Integer;
        minimum_margin::Real=5e-6)
    return function (_z, _tangent, step, _branch; kwargs...)
        state = get(kwargs, :state, nothing)
        iterator = get(kwargs, :iter, nothing)
        if isnothing(state) || isnothing(iterator)
            return true
        end
        summary = try
            BK.get_state_summary(iterator, state)
        catch
            return true
        end
        hasproperty(summary, :minimum_absolute_pressure_drop) || return true
        margin = float(summary.minimum_absolute_pressure_drop)
        if isfinite(margin) && margin <= minimum_margin
            verbosity > 0 && @info(
                "Stopping dense periodic scan before flow reversal",
                accepted_step=Int(step),
                gamma=float(summary.gamma),
                zeta=float(summary.zeta),
                minimum_absolute_pressure_drop=margin,
                guard_margin=float(minimum_margin),
            )
            return false
        end
        return true
    end
end

# Return one record per real Floquet multiplier or complex-conjugate pair.  The
# neutral phase exponent is removed before pairing, which makes growth rates
# comparable across continuation steps even on a finite collocation mesh.
function sax_canonical_floquet_pairs(exponents; conjugacy_tolerance::Real=1e-6)
    values = ComplexF64.(exponents)
    isempty(values) && return (neutral_exponent=0.0 + 0.0im, pairs=Any[])
    neutral_index = argmin(abs.(values))
    neutral = values[neutral_index]
    corrected = values .- neutral
    available = [index for index in eachindex(values) if index != neutral_index]
    used = falses(length(values))
    pairs = Any[]
    for index in available
        used[index] && continue
        value = corrected[index]
        angle = abs(imag(value))
        members = ComplexF64[value]
        if angle > conjugacy_tolerance && abs(pi - angle) > conjugacy_tolerance
            candidates = [other for other in available
                          if other != index && !used[other]]
            if !isempty(candidates)
                other = candidates[argmin(abs(corrected[candidate] - conj(value))
                                           for candidate in candidates)]
                if abs(corrected[other] - conj(value)) <=
                        conjugacy_tolerance * max(1.0, abs(value))
                    push!(members, corrected[other])
                    used[other] = true
                end
            end
        end
        used[index] = true
        growth = sum(real, members) / length(members)
        canonical_angle = sum(abs ∘ imag, members) / length(members)
        exponent = ComplexF64(growth, canonical_angle)
        push!(pairs, (
            pair_index=0,
            track_id=0,
            exponent=exponent,
            growth=float(growth),
            angle=float(canonical_angle),
            angle_to_pi=float(abs(pi - canonical_angle)),
            multiplier=ComplexF64(exp(exponent)),
            members=Tuple(members),
            multiplicity=length(members),
        ))
    end
    sort!(pairs; by=pair -> (pair.angle, pair.growth))
    pairs = [merge(pair, (pair_index=index,)) for (index, pair) in enumerate(pairs)]
    return (neutral_exponent=neutral, pairs=pairs)
end

function _sax_match_floquet_pairs(previous, current,
                                  settings::SaxDensePQZNSSettings)
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
        if used_left[edge.left] || used_right[edge.right]
            continue
        end
        used_left[edge.left] = true
        used_right[edge.right] = true
        matches[edge.right] = edge.left
    end
    return matches
end

function _sax_track_floquet_samples(raw_samples,
                                    settings::SaxDensePQZNSSettings)
    isempty(raw_samples) && return Any[]
    tracked = Any[]
    next_track_id = 1
    previous_pairs = Any[]
    for raw in raw_samples
        canonical = sax_canonical_floquet_pairs(raw.exponents)
        pairs = canonical.pairs
        matches = isempty(previous_pairs) ? Dict{Int,Int}() :
            _sax_match_floquet_pairs(previous_pairs, pairs, settings)
        assigned = Any[]
        for (index, pair) in enumerate(pairs)
            track_id = if haskey(matches, index)
                previous_pairs[matches[index]].track_id
            else
                value = next_track_id
                next_track_id += 1
                value
            end
            push!(assigned, merge(pair, (track_id=track_id,)))
        end
        dominant = isempty(assigned) ? nothing :
            assigned[argmax(pair.growth for pair in assigned)]
        push!(tracked, merge(raw, (
            neutral_exponent=canonical.neutral_exponent,
            pairs=assigned,
            dominant_exponent=isnothing(dominant) ? 0.0 + 0.0im : dominant.exponent,
            dominant_growth=isnothing(dominant) ? NaN : dominant.growth,
            dominant_angle=isnothing(dominant) ? NaN : dominant.angle,
            dominant_angle_to_pi=isnothing(dominant) ? NaN : dominant.angle_to_pi,
        )))
        previous_pairs = assigned
    end
    return tracked
end

function _sax_dense_branch_samples(branch,
                                   zeta::Real,
                                   settings::SaxDensePQZNSSettings)
    rows = Dict(Int(row.step) => row for row in branch.branch)
    solutions = Dict(Int(solution.step) => collect(float.(solution.x))
                     for solution in branch.sol)
    raw = Any[]
    for eigen_record in branch.eig
        eigen_record.converged || continue
        step = Int(eigen_record.step)
        haskey(rows, step) && haskey(solutions, step) || continue
        row = rows[step]
        values = ComplexF64.(eigen_record.eigenvals)
        isempty(values) && continue
        push!(raw, (
            branch_index=length(raw) + 1,
            continuation_step=step,
            gamma=float(row.gamma),
            zeta=float(zeta),
            period=float(row.period),
            exponents=values,
        ))
    end
    sort!(raw; by=sample -> sample.continuation_step)
    return (
        samples=_sax_track_floquet_samples(raw, settings),
        solutions=solutions,
    )
end

function _sax_dense_pair_by_track(sample, track_id::Integer)
    index = findfirst(pair -> pair.track_id == track_id, sample.pairs)
    return isnothing(index) ? nothing : sample.pairs[index]
end

function _sax_dense_candidate_brackets(samples,
                                       settings::SaxDensePQZNSSettings)
    brackets = Any[]
    for index in 1:(length(samples) - 1)
        left, right = samples[index], samples[index + 1]
        gamma_lower, gamma_upper = extrema((left.gamma, right.gamma))
        gamma_upper < settings.root_gamma_range[1] && continue
        gamma_lower > settings.root_gamma_range[2] && continue
        gamma_upper - gamma_lower <= settings.maximum_gamma_bracket || continue
        for left_pair in left.pairs
            right_pair = _sax_dense_pair_by_track(right, left_pair.track_id)
            isnothing(right_pair) && continue
            left_pair.angle >= settings.minimum_ns_angle || continue
            right_pair.angle >= settings.minimum_ns_angle || continue
            left_pair.angle_to_pi >= settings.minimum_angle_to_pi || continue
            right_pair.angle_to_pi >= settings.minimum_angle_to_pi || continue
            crossed = left_pair.growth == 0 || right_pair.growth == 0 ||
                signbit(left_pair.growth) != signbit(right_pair.growth)
            crossed || continue
            angle = (left_pair.angle + right_pair.angle) / 2
            classification = abs(pi - angle) <= settings.r2_warning_angle ?
                :near_r2 : :ns_like
            push!(brackets, (
                zeta=float(left.zeta),
                track_id=Int(left_pair.track_id),
                classification=classification,
                gamma_lower=float(gamma_lower),
                gamma_upper=float(gamma_upper),
                gamma=float((gamma_lower + gamma_upper) / 2),
                gamma_error=float((gamma_upper - gamma_lower) / 2),
                left_step=Int(left.continuation_step),
                right_step=Int(right.continuation_step),
                left_gamma=float(left.gamma),
                right_gamma=float(right.gamma),
                left_exponent=left_pair.exponent,
                right_exponent=right_pair.exponent,
                floquet_angle=float(angle),
                angle_to_pi=float(abs(pi - angle)),
                detected_types=(classification == :near_r2 ? (:pd,) : (:ns,)),
                source=:tracked_pqz_sign_change,
                status=:tracked,
                validated=false,
            ))
        end
    end
    sort!(brackets; by=bracket -> (bracket.gamma, bracket.floquet_angle))
    unique_brackets = Any[]
    for bracket in brackets
        duplicate = any(unique_brackets) do stored
            abs(stored.gamma - bracket.gamma) <= settings.root_gamma_tolerance &&
                abs(stored.floquet_angle - bracket.floquet_angle) <= 1e-3
        end
        duplicate || push!(unique_brackets, bracket)
    end
    return unique_brackets
end

function _sax_dense_compatible_bracket(bracket)
    gamma_error = hasproperty(bracket, :gamma_error) ?
        float(bracket.gamma_error) :
        float((bracket.gamma_upper - bracket.gamma_lower) / 2)
    status = hasproperty(bracket, :status) ? bracket.status : :tracked
    validated = hasproperty(bracket, :validated) ?
        Bool(bracket.validated) : false
    return merge(bracket, (; gamma_error, status, validated))
end

"""
    sax_dense_pqz_bracket_components(brackets; ...)

Track provisional Floquet-neutral brackets between neighboring fixed-zeta
slices.  A component is joined only when gamma, Floquet angle, classification,
and zeta separation are all compatible.  Track identifiers are local to one
fixed-zeta continuation and are therefore deliberately not compared here.
"""
function sax_dense_pqz_bracket_components(
        brackets;
        classification::Symbol=:near_r2,
        maximum_gamma_jump::Real=0.01,
        maximum_angle_jump::Real=0.03,
        maximum_zeta_gap::Real=0.021,
        minimum_points::Integer=2)
    maximum_gamma_jump > 0 ||
        throw(ArgumentError("maximum_gamma_jump must be positive"))
    maximum_angle_jump > 0 ||
        throw(ArgumentError("maximum_angle_jump must be positive"))
    maximum_zeta_gap > 0 ||
        throw(ArgumentError("maximum_zeta_gap must be positive"))
    minimum_points >= 1 ||
        throw(ArgumentError("minimum_points must be positive"))
    points = [_sax_dense_compatible_bracket(point) for point in brackets
              if point.classification == classification]
    isempty(points) && return Any[]
    sort!(points; by=point -> (point.zeta, point.gamma, point.floquet_angle))
    zeta_values = sort(unique(point.zeta for point in points))
    components = Any[]
    active = Int[]
    for zeta in zeta_values
        group = [point for point in points if point.zeta == zeta]
        candidates = Any[]
        for component_index in active
            previous = last(components[component_index])
            delta_zeta = float(zeta - previous.zeta)
            0 < delta_zeta <= maximum_zeta_gap || continue
            for (point_index, point) in enumerate(group)
                delta_gamma = abs(point.gamma - previous.gamma)
                delta_angle = abs(point.floquet_angle - previous.floquet_angle)
                delta_gamma <= maximum_gamma_jump || continue
                delta_angle <= maximum_angle_jump || continue
                cost = delta_gamma / maximum_gamma_jump +
                    delta_angle / maximum_angle_jump +
                    0.1delta_zeta / maximum_zeta_gap
                push!(candidates, (
                    cost=float(cost),
                    component_index=component_index,
                    point_index=point_index,
                ))
            end
        end
        sort!(candidates; by=candidate -> candidate.cost)
        used_components = Set{Int}()
        used_points = Set{Int}()
        for candidate in candidates
            candidate.component_index in used_components && continue
            candidate.point_index in used_points && continue
            push!(components[candidate.component_index],
                  group[candidate.point_index])
            push!(used_components, candidate.component_index)
            push!(used_points, candidate.point_index)
        end
        next_active = collect(used_components)
        for (point_index, point) in enumerate(group)
            point_index in used_points && continue
            push!(components, Any[point])
            push!(next_active, length(components))
        end
        active = next_active
    end
    retained = [sort(component; by=point -> point.zeta)
                for component in components
                if length(component) >= minimum_points]
    sort!(retained; by=component -> (
        -length(component),
        first(component).zeta,
        sum(point.gamma for point in component) / length(component),
    ))
    return retained
end

"""Draw tracked provisional bracket components as lines and uncertainty bands."""
function overlay_sax_dense_pqz_boundary!(
        axis,
        brackets;
        classification::Symbol=:near_r2,
        color="#0072B2",
        linewidth::Real=2.4,
        fillalpha::Real=0.12,
        linestyle::Symbol=:dash,
        show_band::Bool=true,
        show_label::Bool=true,
        compact_label::Bool=false,
        maximum_gamma_jump::Real=0.01,
        maximum_angle_jump::Real=0.03,
        maximum_zeta_gap::Real=0.021)
    components = sax_dense_pqz_bracket_components(
        brackets;
        classification=classification,
        maximum_gamma_jump=maximum_gamma_jump,
        maximum_angle_jump=maximum_angle_jump,
        maximum_zeta_gap=maximum_zeta_gap,
    )
    labeled = false
    for component in components
        gamma_lower = [point.gamma_lower for point in component]
        gamma_upper = [point.gamma_upper for point in component]
        zeta = [point.zeta for point in component]
        if show_band && fillalpha > 0
            boundary_shape = Shape(
                vcat(gamma_lower, reverse(gamma_upper)),
                vcat(zeta, reverse(zeta)),
            )
            plot!(
                axis,
                boundary_shape;
                seriestype=:shape,
                fillcolor=color,
                fillalpha=float(fillalpha),
                linealpha=0,
                label="",
            )
        end
        long_label = classification == :near_r2 ?
            "PQZ near-R2 boundary*" : "PQZ NS-like boundary*"
        short_label = classification == :near_r2 ? "R2*" : "NS-like*"
        plot!(
            axis,
            [point.gamma for point in component],
            zeta;
            color=color,
            linewidth=float(linewidth),
            linestyle=linestyle,
            label=show_label && !labeled ?
                (compact_label ? short_label : long_label) : "",
        )
        labeled = true
    end
    return components
end

function _sax_dense_select_pair(exponents,
                                target::Complex,
                                settings::SaxDensePQZNSSettings)
    canonical = sax_canonical_floquet_pairs(exponents)
    isempty(canonical.pairs) && error("PQZ spectrum has no nontrivial pair")
    candidates = [pair for pair in canonical.pairs
                  if pair.angle >= settings.minimum_ns_angle &&
                     pair.angle_to_pi >= settings.minimum_angle_to_pi]
    isempty(candidates) && error("PQZ spectrum has no admissible complex pair")
    selected = candidates[argmin(hypot(
        pair.growth - real(target), pair.angle - abs(imag(target)))
        for pair in candidates)]
    abs(selected.angle - abs(imag(target))) <=
        settings.maximum_pair_angle_jump || error(
        "tracked Floquet pair jumped by more than the configured angle limit",
    )
    return selected
end

function _sax_dense_correct_orbit(solution,
                                  gamma::Real,
                                  zeta::Real,
                                  model_p::NamedTuple,
                                  bifurcation_settings::SaxBifurcationSettings,
                                  settings::SaxDensePQZNSSettings)
    checkpoint = (
        key="dense_pqz_trial",
        type=:ns,
        mode=settings.mode,
        source_hopf_key="dense_pqz_hopf",
        gamma=float(gamma),
        zeta=float(zeta),
        floquet_angle=NaN,
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
            max_iterations=settings.newton_max_iterations,
            linsolver=BK.COPLS(),
            verbose=false,
        );
        normN=BK.norminf,
    )
    BK.converged(corrected) || error(
        "periodic-orbit correction failed at gamma=$(float(gamma)), zeta=$(float(zeta))",
    )
    refined = merge(checkpoint, (solution=collect(float.(corrected.u)),))
    refined_wrapper, refined_parameters = _sax_periodic_wrapper(
        refined, model_p, bifurcation_settings)
    residual = norm(BK.residual(
        refined_wrapper, refined.solution, refined_parameters), Inf)
    jacobian = BK.jacobian(refined_wrapper, refined.solution, refined_parameters)
    pqz_values, _, converged, iterations = SaxFloquetPQZ(
        fallback_to_floquet_coll=false)(
        BK.get_discretization(refined_wrapper),
        jacobian,
        2 + 2settings.nmodes,
    )
    converged || error("PQZ Floquet solve did not converge")
    return (
        checkpoint=refined,
        exponents=ComplexF64.(pqz_values),
        orbit_residual=float(residual),
        pqz_iterations=Int(iterations),
    )
end

function _sax_dense_regrid_solution(checkpoint,
                                    source_settings::SaxBifurcationSettings,
                                    target_settings::SaxBifurcationSettings,
                                    model_p::NamedTuple,
                                    settings::SaxDensePQZNSSettings)
    source_wrapper, _ = _sax_periodic_wrapper(
        checkpoint, model_p, source_settings)
    interpolant = BK.POSolution(
        BK.get_discretization(source_wrapper), checkpoint.solution)
    state_dimension = 2 + 2settings.nmodes
    target_checkpoint = (
        state=collect(checkpoint.solution[1:state_dimension]),
        gamma=float(checkpoint.gamma),
        zeta=float(checkpoint.zeta),
    )
    problem, parameters = _sax_problem_from_checkpoint(
        target_checkpoint, model_p, target_settings)
    intervals = target_settings.po_collocation_intervals
    degree = target_settings.po_collocation_degree
    orbit_dimension = state_dimension * (1 + intervals * degree)
    collocation = BK.Collocation(
        intervals,
        degree;
        N=state_dimension,
        prob_vf=problem,
        ϕ=zeros(orbit_dimension),
        xπ=zeros(orbit_dimension),
        ∂ϕ=zeros(state_dimension, intervals * degree),
        jacobian=BK.DenseAnalyticalInplace(),
        update_section_every_step=1,
    )
    guess = BK.generate_solution(
        collocation, interpolant, float(last(checkpoint.solution)))
    BK.updatesection!(collocation, guess, parameters)
    corrected = BK.newton(
        collocation,
        guess,
        BK.NewtonPar(
            tol=settings.newton_tol,
            max_iterations=settings.newton_max_iterations,
            linsolver=BK.COPLS(),
            verbose=false,
        );
        normN=BK.norminf,
    )
    BK.converged(corrected) || error(
        "periodic-orbit regridding failed on mesh $(intervals) x $(degree)",
    )
    return merge(checkpoint, (solution=collect(float.(corrected.u)),))
end

function _sax_dense_refine_bracket(
        bracket,
        left_solution,
        right_solution,
        model_p::NamedTuple,
        bifurcation_settings::SaxBifurcationSettings,
        settings::SaxDensePQZNSSettings)
    if bracket.left_gamma <= bracket.right_gamma
        lower_gamma, upper_gamma = bracket.left_gamma, bracket.right_gamma
        lower_solution, upper_solution = left_solution, right_solution
        lower_target, upper_target = bracket.left_exponent, bracket.right_exponent
    else
        lower_gamma, upper_gamma = bracket.right_gamma, bracket.left_gamma
        lower_solution, upper_solution = right_solution, left_solution
        lower_target, upper_target = bracket.right_exponent, bracket.left_exponent
    end
    lower = _sax_dense_correct_orbit(
        lower_solution, lower_gamma, bracket.zeta,
        model_p, bifurcation_settings, settings)
    lower_pair = _sax_dense_select_pair(
        lower.exponents, lower_target, settings)
    upper = _sax_dense_correct_orbit(
        upper_solution, upper_gamma, bracket.zeta,
        model_p, bifurcation_settings, settings)
    upper_pair = _sax_dense_select_pair(
        upper.exponents, upper_target, settings)
    (lower_pair.growth == 0 || upper_pair.growth == 0 ||
     signbit(lower_pair.growth) != signbit(upper_pair.growth)) || error(
        "corrected PQZ growth no longer brackets zero",
    )

    chosen = abs(lower_pair.growth) <= abs(upper_pair.growth) ? lower : upper
    chosen_pair = abs(lower_pair.growth) <= abs(upper_pair.growth) ?
        lower_pair : upper_pair
    iteration = 0
    for current_iteration in 1:settings.maximum_root_iterations
        iteration = current_iteration
        if abs(chosen_pair.growth) <= settings.root_growth_tolerance &&
                upper_gamma - lower_gamma <= 10 * settings.root_gamma_tolerance
            break
        end
        middle_gamma = (lower_gamma + upper_gamma) / 2
        fraction = (middle_gamma - lower_gamma) / (upper_gamma - lower_gamma)
        middle_guess = (1 - fraction) .* lower.checkpoint.solution .+
            fraction .* upper.checkpoint.solution
        target = (1 - fraction) * lower_pair.exponent +
            fraction * upper_pair.exponent
        middle = _sax_dense_correct_orbit(
            middle_guess, middle_gamma, bracket.zeta,
            model_p, bifurcation_settings, settings)
        middle_pair = _sax_dense_select_pair(middle.exponents, target, settings)
        if abs(middle_pair.growth) < abs(chosen_pair.growth)
            chosen, chosen_pair = middle, middle_pair
        end
        if middle_pair.growth == 0
            lower_gamma = upper_gamma = middle_gamma
            chosen, chosen_pair = middle, middle_pair
            break
        elseif signbit(lower_pair.growth) == signbit(middle_pair.growth)
            lower_gamma = middle_gamma
            lower, lower_pair = middle, middle_pair
        else
            upper_gamma = middle_gamma
            upper, upper_pair = middle, middle_pair
        end
        upper_gamma - lower_gamma <= settings.root_gamma_tolerance && break
    end
    checkpoint = merge(chosen.checkpoint, (
        key="dense_pqz_ns_z$(_sax_periodic_schur_slice_tag(bracket.zeta))_" *
            "g$(round(chosen.checkpoint.gamma; digits=8))",
        type=:ns,
        mode=settings.mode,
        source_hopf_key="dense_pqz_mode$(settings.mode)_hopf",
        floquet_angle=float(chosen_pair.angle),
    ))
    return (
        checkpoint=checkpoint,
        pair=chosen_pair,
        orbit_residual=chosen.orbit_residual,
        iterations=iteration,
        gamma_interval=(float(lower_gamma), float(upper_gamma)),
        converged=abs(chosen_pair.growth) <= settings.root_growth_tolerance &&
                  upper_gamma - lower_gamma <= 10 * settings.root_gamma_tolerance,
    )
end

function _sax_dense_refine_mesh_bracket(
        bracket,
        left_solution,
        right_solution,
        model_p::NamedTuple,
        bifurcation_settings::SaxBifurcationSettings,
        settings::SaxDensePQZNSSettings)
    gamma_left = float(bracket.left_gamma)
    gamma_right = float(bracket.right_gamma)
    gamma_span = gamma_right - gamma_left
    abs(gamma_span) > eps(Float64) || error("zero-width PQZ root bracket")
    solution_at = gamma -> begin
        fraction = (float(gamma) - gamma_left) / gamma_span
        (1 - fraction) .* left_solution .+ fraction .* right_solution
    end
    center = (gamma_left + gamma_right) / 2
    half_width = abs(gamma_span) / 2
    lower_target, upper_target = gamma_left <= gamma_right ?
        (bracket.left_exponent, bracket.right_exponent) :
        (bracket.right_exponent, bracket.left_exponent)
    attempts = Any[]
    for expansion in (1.0, 2.0, 4.0, 8.0)
        lower = max(settings.root_gamma_range[1], center - expansion * half_width)
        upper = min(settings.root_gamma_range[2], center + expansion * half_width)
        lower < upper || continue
        trial = merge(bracket, (
            left_gamma=lower,
            right_gamma=upper,
            gamma_lower=lower,
            gamma_upper=upper,
            left_exponent=lower_target,
            right_exponent=upper_target,
        ))
        try
            refined = _sax_dense_refine_bracket(
                trial,
                solution_at(lower),
                solution_at(upper),
                model_p,
                bifurcation_settings,
                settings,
            )
            return merge(refined, (
                mesh_bracket_expansion=expansion,
                mesh_bracket_attempts=attempts,
            ))
        catch err
            err isa InterruptException && rethrow()
            push!(attempts, (
                expansion=expansion,
                gamma_interval=(lower, upper),
                exception_type=Symbol(nameof(typeof(err))),
                error=sprint(showerror, err),
            ))
        end
    end
    error("PQZ root refinement failed after bracket expansion: $(attempts)")
end

function _sax_dense_validate_bracket(
        bracket,
        left_solution,
        right_solution,
        model_p::NamedTuple,
        settings::SaxDensePQZNSSettings;
        verbosity::Integer=1)
    base_settings = _sax_dense_bifurcation_settings(settings)
    left_checkpoint = (
        key="dense_pqz_left",
        type=:ns,
        mode=settings.mode,
        source_hopf_key="dense_pqz_hopf",
        gamma=float(bracket.left_gamma),
        zeta=float(bracket.zeta),
        floquet_angle=float(abs(imag(bracket.left_exponent))),
        solution=collect(float.(left_solution)),
    )
    right_checkpoint = merge(left_checkpoint, (
        key="dense_pqz_right",
        gamma=float(bracket.right_gamma),
        floquet_angle=float(abs(imag(bracket.right_exponent))),
        solution=collect(float.(right_solution)),
    ))
    rows = Any[]
    for mesh in settings.validation_meshes
        verbosity > 0 && @info(
            "Dense PQZ root mesh validation",
            zeta=float(bracket.zeta),
            mesh=mesh,
            gamma_interval=(bracket.gamma_lower, bracket.gamma_upper),
        )
        target_settings = _sax_dense_bifurcation_settings(
            settings; intervals=mesh[1], degree=mesh[2])
        left = mesh == (settings.collocation_intervals,
                        settings.collocation_degree) ? left_checkpoint :
            _sax_dense_regrid_solution(
                left_checkpoint, base_settings, target_settings,
                model_p, settings)
        right = mesh == (settings.collocation_intervals,
                         settings.collocation_degree) ? right_checkpoint :
            _sax_dense_regrid_solution(
                right_checkpoint, base_settings, target_settings,
                model_p, settings)
        refined = _sax_dense_refine_mesh_bracket(
            bracket, left.solution, right.solution,
            model_p, target_settings, settings)
        dual = _sax_dual_floquet_spectrum(
            refined.checkpoint,
            model_p,
            target_settings,
            _sax_dense_legacy_settings(settings),
        )
        push!(rows, merge(dual, (
            checkpoint=refined.checkpoint,
            refinement=refined,
        )))
    end
    isempty(rows) && error("dense PQZ validation_meshes cannot be empty")
    mesh_converged = length(rows) == 1 || (
        abs(rows[end].checkpoint.gamma - rows[end - 1].checkpoint.gamma) <=
            settings.mesh_gamma_tolerance &&
        abs(rows[end].periodic_schur.floquet_angle -
            rows[end - 1].periodic_schur.floquet_angle) <=
            settings.mesh_angle_tolerance
    )
    accepted = mesh_converged && all(rows) do row
        row.refinement.converged &&
        row.periodic_schur.valid &&
        row.floquet_coll.valid &&
        row.methods_agree &&
        row.classification_agrees &&
        row.orbit_residual <= 100 * settings.newton_tol
    end
    finest = rows[end]
    return (
        checkpoint=finest.checkpoint,
        validation=finest,
        mesh_validation=rows,
        mesh_converged=mesh_converged,
        accepted=accepted,
        near_r2=finest.periodic_schur.near_r2,
        source_bracket=bracket,
    )
end


# ---------------------------------------------------------------------------
# Per-slice cache, adaptive coverage, seed selection, and progress loading
# ---------------------------------------------------------------------------

function _sax_dense_pqz_slice_path(directory::AbstractString, zeta::Real)
    return joinpath(
        directory,
        "dense_pqz_ns_slice_z$(_sax_periodic_schur_slice_tag(zeta)).jld2",
    )
end

function _sax_dense_boundary_point(slice,
                                   settings::SaxDensePQZNSSettings)
    accepted = [root for root in slice.roots if root.accepted]
    if !isempty(accepted)
        selected = accepted[argmin(
            abs(root.checkpoint.gamma - settings.gamma_hint) for root in accepted)]
        return (
            gamma=float(selected.checkpoint.gamma),
            classification=selected.near_r2 ? :near_r2 : :ns_like,
            validated=true,
        )
    end
    brackets = slice.candidate_brackets
    isempty(brackets) && return nothing
    selected = brackets[argmin(abs(bracket.gamma - settings.gamma_hint)
                               for bracket in brackets)]
    return (
        gamma=float(selected.gamma),
        classification=selected.classification,
        validated=false,
    )
end

function _sax_dense_adaptive_zeta_values(slices,
                                         settings::SaxDensePQZNSSettings)
    length(slices) >= 2 || return Float64[]
    ordered = sort(slices; by=slice -> slice.zeta)
    additions = Float64[]
    for index in 1:(length(ordered) - 1)
        left, right = ordered[index], ordered[index + 1]
        gap = right.zeta - left.zeta
        gap > 1.5 * settings.adaptive_min_step || continue
        left_boundary = _sax_dense_boundary_point(left, settings)
        right_boundary = _sax_dense_boundary_point(right, settings)
        refine = if isnothing(left_boundary) && isnothing(right_boundary)
            false
        elseif isnothing(left_boundary) || isnothing(right_boundary)
            true
        else
            abs(left_boundary.gamma - right_boundary.gamma) >=
                settings.adaptive_gamma_jump ||
                left_boundary.classification != right_boundary.classification
        end
        refine && push!(additions, (left.zeta + right.zeta) / 2)
    end
    return sort(unique(round.(additions; digits=12)))
end

function _sax_compute_dense_zeta_set!(
        slices,
        failures,
        zeta_values,
        model_p::NamedTuple,
        output_directory::AbstractString,
        settings::SaxDensePQZNSSettings;
        resume::Bool,
        verbosity::Integer)
    completed = Set(round(slice.zeta; digits=12) for slice in slices)
    for (index, zeta) in enumerate(zeta_values)
        rounded = round(float(zeta); digits=12)
        rounded in completed && continue
        path = _sax_dense_pqz_slice_path(output_directory, zeta)
        cached = resume ? _load_sax_transition_mechanism_cache(
            path, :dense_pqz_ns_slice, model_p, settings) :
            (status=:missing, payload=nothing, reason="resume disabled")
        if cached.status == :valid
            push!(slices, cached.payload)
            push!(completed, rounded)
            continue
        end
        verbosity > 0 && @info(
            "Starting event-free dense PQZ slice",
            slice=index,
            scheduled=length(zeta_values),
            zeta=float(zeta),
            coverage=settings.coverage,
        )
        try
            slice = compute_sax_dense_pqz_ns_slice(
                model_p, zeta; settings=settings, verbosity=verbosity)
            _save_sax_transition_mechanism_cache(
                path, :dense_pqz_ns_slice, slice, model_p, settings)
            push!(slices, slice)
            push!(completed, rounded)
        catch err
            err isa InterruptException && rethrow()
            if occursin("no mode-$(settings.mode) Hopf crossing was bracketed",
                        sprint(showerror, err))
                slice = (
                    analysis=:dense_pqz_ns_slice,
                    status=:no_hopf,
                    zeta=float(zeta),
                    hopf_checkpoint=nothing,
                    samples=Any[],
                    candidate_brackets=Any[],
                    bifurcations=Any[],
                    roots=Any[],
                    accepted_roots=0,
                    root_failures=Any[],
                    periodic_diagnostics=nothing,
                    continuation_method=:event_free_pqz,
                    reason="mode-$(settings.mode) Hopf is absent in gamma_range",
                    settings=_portable_sax_mechanism_settings(settings),
                )
                _save_sax_transition_mechanism_cache(
                    path, :dense_pqz_ns_slice, slice, model_p, settings)
                push!(slices, slice)
                push!(completed, rounded)
                verbosity > 0 && @info(
                    "Dense PQZ slice has no mode-$(settings.mode) Hopf",
                    zeta=float(zeta),
                )
                continue
            end
            failure = (
                zeta=float(zeta),
                exception_type=Symbol(nameof(typeof(err))),
                error=sprint(showerror, err),
            )
            push!(failures, failure)
            verbosity > 0 && @warn "Dense PQZ slice failed" failure
        end
    end
    return slices
end

"""Compute or resume all fixed-zeta dense PQZ slices."""
function compute_sax_dense_pqz_ns_slices(
        model_p::NamedTuple,
        output_directory::AbstractString;
        settings::SaxDensePQZNSSettings=SaxDensePQZNSSettings(),
        resume::Bool=true,
        verbosity::Integer=1)
    _validate_sax_dense_pqz_ns_settings(settings)
    mkpath(output_directory)
    slices = Any[]
    failures = Any[]
    base_values = collect(sax_dense_pqz_zeta_values(settings))
    _sax_compute_dense_zeta_set!(
        slices, failures, base_values, model_p, output_directory, settings;
        resume=resume, verbosity=verbosity)
    scheduled = copy(base_values)
    for level in 1:settings.adaptive_levels
        additions = _sax_dense_adaptive_zeta_values(slices, settings)
        isempty(additions) && break
        append!(scheduled, additions)
        verbosity > 0 && @info(
            "Dense PQZ adaptive zeta refinement",
            level=level,
            added_slices=length(additions),
        )
        _sax_compute_dense_zeta_set!(
            slices, failures, additions, model_p, output_directory, settings;
            resume=resume, verbosity=verbosity)
    end
    ordered = sort(slices; by=slice -> slice.zeta)
    result = (
        analysis=:dense_pqz_ns_slices,
        coverage=settings.coverage,
        slices=ordered,
        roots=Any[root for slice in ordered for root in slice.roots],
        candidate_brackets=Any[bracket for slice in ordered
                               for bracket in slice.candidate_brackets],
        failures=failures,
        completed_slices=length(ordered),
        expected_slices=length(unique(round.(scheduled; digits=12))),
        base_expected_slices=length(base_values),
        scheduled_zeta=sort(unique(round.(scheduled; digits=12))),
        settings=_portable_sax_mechanism_settings(settings),
    )
    return _save_sax_transition_mechanism_cache(
        joinpath(output_directory, "dense_pqz_ns_slice_manifest.jld2"),
        :dense_pqz_ns_slices,
        result,
        model_p,
        settings,
    )
end

function _sax_dense_root_score(root)
    return (
        root.accepted ? 0 : 1,
        root.near_r2 ? 1 : 0,
        abs(root.validation.periodic_schur.corrected_growth),
        abs(root.checkpoint.gamma - 0.604),
    )
end

function sax_dense_pqz_augmented_seed(
        seed_result,
        settings::SaxDensePQZNSSettings)
    seed_result.status == :valid || return seed_result
    target_mesh = (
        settings.collocation_intervals,
        settings.collocation_degree,
    )
    state_dimension = 2 + 2settings.nmodes
    expected_length = state_dimension * (
        1 + target_mesh[1] * target_mesh[2]) + 1
    if length(seed_result.seed.solution) == expected_length
        return seed_result
    end
    root = hasproperty(seed_result, :root) ? seed_result.root : nothing
    rows = isnothing(root) || !hasproperty(root, :mesh_validation) ? Any[] :
        root.mesh_validation
    index = findfirst(row -> row.mesh == target_mesh, rows)
    if isnothing(index)
        return (
            analysis=:dense_pqz_ns_seed,
            status=:missing,
            seed=nothing,
            reason="validated root has no checkpoint on augmented mesh $(target_mesh)",
        )
    end
    checkpoint = rows[index].checkpoint
    length(checkpoint.solution) == expected_length || return (
        analysis=:dense_pqz_ns_seed,
        status=:missing,
        seed=nothing,
        reason="stored augmented-mesh checkpoint has an unexpected dimension",
    )
    return merge(seed_result, (
        seed=checkpoint,
        reason="selected validated root on augmented mesh $(target_mesh)",
    ))
end

"""Select and cache the best mesh-validated dense PQZ NS root."""
function select_sax_dense_pqz_ns_seed(
        model_p::NamedTuple,
        slices,
        output_path::AbstractString;
        settings::SaxDensePQZNSSettings=SaxDensePQZNSSettings())
    accepted = [root for root in slices.roots if root.accepted]
    result = if isempty(accepted)
        (
            analysis=:dense_pqz_ns_seed,
            status=:missing,
            seed=nothing,
            reason="no mesh-converged dual-method NS root is available",
        )
    else
        selected = accepted[argmin(_sax_dense_root_score(root)
                                   for root in accepted)]
        (
            analysis=:dense_pqz_ns_seed,
            status=:valid,
            seed=selected.checkpoint,
            root=selected,
            reason="selected independently refined dense PQZ root",
        )
    end
    result = sax_dense_pqz_augmented_seed(result, settings)
    return _save_sax_transition_mechanism_cache(
        output_path, :dense_pqz_ns_seed, result, model_p, settings)
end

function _sax_dense_root_locus(slices)
    gamma = Float64[]
    zeta = Float64[]
    angles = Float64[]
    for slice in sort(slices; by=slice -> slice.zeta)
        accepted = [root for root in slice.roots if root.accepted]
        isempty(accepted) && continue
        root = accepted[argmin(_sax_dense_root_score(candidate)
                               for candidate in accepted)]
        push!(gamma, float(root.checkpoint.gamma))
        push!(zeta, float(root.checkpoint.zeta))
        push!(angles, float(root.checkpoint.floquet_angle))
    end
    return (
        kind=:ns,
        gamma=gamma,
        zeta=zeta,
        floquet_angle=angles,
        mode=2,
        frequency=NaN,
        source=(analysis=:dense_pqz_fixed_zeta_roots, provisional=false),
        diagnostics=nothing,
    )
end


"""
    load_sax_dense_pqz_ns_progress(model_p, directory; settings)

Read dense per-slice caches safely while the runner is active.  Historical
validation and augmented-continuation products in the same directory are read
through the legacy loader; only the brittle fixed-zeta localization stage is
replaced.
"""
function sax_dense_pqz_cached_settings(
        directory::AbstractString;
        fallback::SaxDensePQZNSSettings=SaxDensePQZNSSettings())
    slices_directory = joinpath(directory, "slices")
    candidates = String[
        joinpath(slices_directory, "dense_pqz_ns_slice_manifest.jld2"),
    ]
    if isdir(slices_directory)
        append!(candidates, sort([
            joinpath(slices_directory, name)
            for name in readdir(slices_directory)
            if startswith(name, "dense_pqz_ns_slice_z") &&
               endswith(name, ".jld2")
        ]))
    end
    for path in unique(candidates)
        isfile(path) || continue
        stored = try
            Logging.with_logger(Logging.NullLogger()) do
                JLD2.load(path, "cache")
            end
        catch
            continue
        end
        signature = try
            stored.settings_signature
        catch
            continue
        end
        settings = try
            SaxDensePQZNSSettings(; signature...)
        catch
            continue
        end
        try
            return _validate_sax_dense_pqz_ns_settings(settings)
        catch
            continue
        end
    end
    return fallback
end

function load_sax_dense_pqz_ns_progress(
        model_p::NamedTuple,
        directory::AbstractString;
        settings::SaxDensePQZNSSettings=SaxDensePQZNSSettings())
    _validate_sax_dense_pqz_ns_settings(settings)
    legacy = load_sax_periodic_schur_ns_progress(
        model_p,
        directory;
        profile=settings.profile,
        settings=_sax_dense_legacy_settings(settings),
    )
    slices_directory = joinpath(directory, "slices")
    manifest = _load_sax_transition_mechanism_cache(
        joinpath(slices_directory, "dense_pqz_ns_slice_manifest.jld2"),
        :dense_pqz_ns_slices,
        model_p,
        settings,
    )
    slices = Any[]
    failures = Any[]
    expected = length(sax_dense_pqz_zeta_values(settings))
    if manifest.status == :valid
        append!(slices, manifest.payload.slices)
        append!(failures, manifest.payload.failures)
        expected = manifest.payload.expected_slices
    else
        for zeta in sax_dense_pqz_zeta_values(settings)
            loaded = _load_sax_transition_mechanism_cache(
                _sax_dense_pqz_slice_path(slices_directory, zeta),
                :dense_pqz_ns_slice,
                model_p,
                settings,
            )
            loaded.status == :valid && push!(slices, loaded.payload)
        end
    end
    sort!(slices; by=slice -> slice.zeta)
    roots = Any[root for slice in slices for root in slice.roots]
    brackets = Any[_sax_dense_compatible_bracket(bracket)
                   for slice in slices for bracket in slice.candidate_brackets]
    completed = length(slices)
    status = if completed == expected && isempty(failures)
        :complete
    elseif completed > 0
        :partial
    elseif manifest.status == :valid && !isempty(failures)
        :failed
    else
        manifest.status
    end
    seed_cache = _load_sax_transition_mechanism_cache(
        joinpath(directory, "refined_seed.jld2"),
        :dense_pqz_ns_seed,
        model_p,
        settings,
    )
    return merge(legacy, (
        method=:dense_event_free_pqz,
        coverage=settings.coverage,
        settings=settings,
        slices=(
            status=status,
            completed=completed,
            expected=expected,
            rows=slices,
            failures=failures,
            roots=roots,
            accepted_roots=count(root -> root.accepted, roots),
            locus=_sax_dense_root_locus(slices),
            provisional_brackets=brackets,
        ),
        seed=seed_cache.status == :valid ? seed_cache.payload : nothing,
        seed_status=seed_cache.status == :valid ?
            seed_cache.payload.status : seed_cache.status,
    ))
end

"""Merge focused and full-coverage progress for one notebook overlay."""
function merge_sax_dense_pqz_ns_progress(focused, full)
    full_has_data = full.slices.completed > 0 ||
        !isnothing(full.validation) || !isempty(full.augmented.curves)
    full_has_data || return focused
    by_zeta = Dict{Float64,Any}()
    for slice in focused.slices.rows
        by_zeta[round(float(slice.zeta); digits=12)] = slice
    end
    # Prefer a full-run slice at a duplicate zeta because its cache signature
    # records the eventual coverage requested by the user.
    for slice in full.slices.rows
        by_zeta[round(float(slice.zeta); digits=12)] = slice
    end
    rows = sort(collect(values(by_zeta)); by=slice -> slice.zeta)
    roots = Any[root for slice in rows for root in slice.roots]
    brackets = Any[_sax_dense_compatible_bracket(bracket)
                   for slice in rows for bracket in slice.candidate_brackets]
    requested = unique(vcat(
        collect(sax_dense_pqz_zeta_values(focused.settings)),
        collect(sax_dense_pqz_zeta_values(full.settings)),
    ))
    completed = length(rows)
    expected = max(length(requested), full.slices.expected)
    failures = vcat(focused.slices.failures, full.slices.failures)
    status = completed == expected && isempty(failures) ? :complete :
        completed > 0 ? :partial : full.slices.status
    selected_seed = full.seed_status == :valid ? full : focused
    selected_augmented = !isempty(full.augmented.curves) ?
        full.augmented : focused.augmented
    return merge(full, (
        directory=(focused=focused.directory, full=full.directory),
        coverage=:focused_plus_full,
        validation=isnothing(full.validation) ? focused.validation : full.validation,
        validation_status=isnothing(full.validation) ?
            focused.validation_status : full.validation_status,
        slices=(
            status=status,
            completed=completed,
            expected=expected,
            rows=rows,
            failures=failures,
            roots=roots,
            accepted_roots=count(root -> root.accepted, roots),
            locus=_sax_dense_root_locus(rows),
            provisional_brackets=brackets,
        ),
        seed=selected_seed.seed,
        seed_status=selected_seed.seed_status,
        augmented=selected_augmented,
    ))
end



function _sax_dense_unique_roots(roots,
                                 settings::SaxDensePQZNSSettings)
    ordered = sort(roots; by=root -> (
        root.accepted ? 0 : 1,
        abs(root.validation.periodic_schur.corrected_growth),
    ))
    unique_roots = Any[]
    for root in ordered
        duplicate = any(unique_roots) do stored
            abs(stored.checkpoint.gamma - root.checkpoint.gamma) <=
                settings.mesh_gamma_tolerance &&
            abs(stored.checkpoint.floquet_angle - root.checkpoint.floquet_angle) <=
                settings.mesh_angle_tolerance
        end
        duplicate || push!(unique_roots, root)
    end
    sort!(unique_roots; by=root -> root.checkpoint.gamma)
    return unique_roots
end

"""
    compute_sax_dense_pqz_ns_slice(model_p, zeta; settings, verbosity=1)

Continue a mode-2 periodic family from its independently recomputed Hopf point.
The continuation stores a full PQZ spectrum at every accepted step but does
not ask BifurcationKit to localize events.  Floquet pairs are then tracked,
zero crossings are bracketed, and every candidate is independently refined
and checked with PQZ and FloquetColl on all validation meshes.
"""
function compute_sax_dense_pqz_ns_slice(
        model_p::NamedTuple,
        zeta::Real;
        settings::SaxDensePQZNSSettings=SaxDensePQZNSSettings(),
        verbosity::Integer=1)
    _validate_sax_dense_pqz_ns_settings(settings)
    hopf = refine_sax_hopf_checkpoint(
        model_p,
        zeta,
        settings.mode;
        settings=_sax_dense_rescue_settings(settings, zeta),
    )
    bifurcation_settings = _sax_dense_bifurcation_settings(settings)
    branch = continue_sax_periodic_orbits(
        hopf,
        model_p;
        settings=bifurcation_settings,
        verbosity=Int(verbosity),
        bothside=false,
        eigsolver=SaxFloquetPQZ(),
        detect_bifurcation=1,
        save_eigenvectors=false,
        step_callback=_sax_dense_periodic_flow_guard(Int(verbosity)),
    )
    extracted = _sax_dense_branch_samples(branch, zeta, settings)
    brackets = _sax_dense_candidate_brackets(extracted.samples, settings)
    roots = Any[]
    root_failures = Any[]
    for (index, bracket) in enumerate(brackets)
        verbosity > 0 && @info(
            "Refining tracked PQZ sign change",
            zeta=float(zeta),
            candidate=index,
            candidates=length(brackets),
            gamma_interval=(bracket.gamma_lower, bracket.gamma_upper),
            classification=bracket.classification,
        )
        try
            push!(roots, _sax_dense_validate_bracket(
                bracket,
                extracted.solutions[bracket.left_step],
                extracted.solutions[bracket.right_step],
                model_p,
                settings,
                verbosity=verbosity,
            ))
        catch err
            err isa InterruptException && rethrow()
            push!(root_failures, (
                zeta=float(zeta),
                gamma_interval=(bracket.gamma_lower, bracket.gamma_upper),
                track_id=bracket.track_id,
                exception_type=Symbol(nameof(typeof(err))),
                error=sprint(showerror, err),
            ))
        end
    end
    roots = _sax_dense_unique_roots(roots, settings)
    return (
        analysis=:dense_pqz_ns_slice,
        zeta=float(zeta),
        hopf_checkpoint=hopf,
        samples=extracted.samples,
        candidate_brackets=brackets,
        bifurcations=brackets,
        roots=roots,
        accepted_roots=count(root -> root.accepted, roots),
        root_failures=root_failures,
        periodic_diagnostics=_sax_branch_terminal_diagnostics(branch, :gamma),
        continuation_method=:event_free_pqz,
        settings=_portable_sax_mechanism_settings(settings),
    )
end
