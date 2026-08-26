# Adaptive, cacheable diagnostics for finite-amplitude register transitions.
#
# This layer deliberately starts from time integrations and attractor history.
# It complements local continuation: once a periodic attractor and a switching
# bracket are reliable, that orbit can be promoted to a BifurcationKit
# collocation seed for Floquet-based fold, PD, or NS identification.

if !isdefined(@__MODULE__, :_sax_bifurcation_model_signature)
    _sax_bifurcation_model_signature(model_p::NamedTuple, nmodes::Int) = (
        nmodes=nmodes,
        alpha=collect(float.(model_p.α[1:nmodes])),
        omega=collect(float.(model_p.ω[1:nmodes])),
        coupling=collect(float.(model_p.C[1:nmodes])),
    )
end

if !isdefined(@__MODULE__, :_atomic_jld2_save)
    function _atomic_jld2_save(path::AbstractString; kwargs...)
        directory = dirname(abspath(path))
        mkpath(directory)
        temporary_path = tempname(directory; cleanup=false)
        try
            JLD2.jldsave(temporary_path; kwargs...)
            mv(temporary_path, path; force=true)
        finally
            isfile(temporary_path) && rm(temporary_path; force=true)
        end
        return path
    end
end

Base.@kwdef struct SaxTransitionRefinementSettings
    nmodes::Int = 8
    zeta_candidates::Tuple = (0.30, 0.40, 0.50, 0.60)
    transition_gamma_range::Tuple{Float64,Float64} = (0.40, 0.70)
    transition_activity_half_width::Float64 = 0.12
    detailed_half_width::Float64 = 0.08
    detailed_gamma_points::Int = 121

    block_time::Float64 = 750.0
    max_blocks::Int = 8
    required_blocks::Int = 3
    block_relative_tolerance::Float64 = 0.05
    saveat::Float64 = 0.4
    tail_fraction::Float64 = 0.5
    sustain_threshold::Float64 = 1e-3
    endpoint_excitation::Float64 = 1e-3
    reltol::Float64 = 1e-7
    abstol::Float64 = 1e-9

    coexistence_contrast_gap::Float64 = 0.5
    poincare_recurrence_tolerance::Float64 = 0.05
    poincare_max_return_period::Int = 8

    compute_basin::Bool = true
    basin_points::Int = 25
    basin_transverse_fraction::Float64 = 0.20
end

function sax_transition_refinement_settings(
        profile::Symbol=:final;
        zeta_candidates=(0.30, 0.40, 0.50, 0.60))
    profile in (:smoke, :pilot, :final) || throw(ArgumentError(
        "transition-refinement profile must be :smoke, :pilot, or :final",
    ))
    common = (; zeta_candidates=Tuple(float.(zeta_candidates)))
    profile == :smoke && return SaxTransitionRefinementSettings(;
        common...,
        detailed_gamma_points=3,
        block_time=0.6,
        max_blocks=1,
        required_blocks=1,
        saveat=0.2,
        compute_basin=true,
        basin_points=3,
    )
    profile == :pilot && return SaxTransitionRefinementSettings(;
        common...,
        detailed_gamma_points=61,
        block_time=500.0,
        max_blocks=5,
        required_blocks=2,
        basin_points=17,
    )
    return SaxTransitionRefinementSettings(; common...)
end

function _validate_sax_transition_refinement_settings(
        settings::SaxTransitionRefinementSettings)
    settings.nmodes >= 2 || throw(ArgumentError("nmodes must be at least two"))
    isempty(settings.zeta_candidates) && throw(ArgumentError(
        "zeta_candidates cannot be empty",
    ))
    all(zeta -> 0 < zeta < 1, settings.zeta_candidates) || throw(ArgumentError(
        "all zeta candidates must lie inside (0,1)",
    ))
    settings.transition_gamma_range[1] < settings.transition_gamma_range[2] ||
        throw(ArgumentError("transition_gamma_range must be ordered"))
    settings.detailed_gamma_points >= 3 || throw(ArgumentError(
        "detailed_gamma_points must be at least three",
    ))
    settings.block_time > 0 || throw(ArgumentError("block_time must be positive"))
    1 <= settings.required_blocks <= settings.max_blocks || throw(ArgumentError(
        "require 1 <= required_blocks <= max_blocks",
    ))
    settings.block_relative_tolerance >= 0 || throw(ArgumentError(
        "block_relative_tolerance must be nonnegative",
    ))
    settings.basin_points >= 3 || throw(ArgumentError(
        "basin_points must be at least three",
    ))
    return settings
end

_portable_sax_transition_refinement_settings(settings) =
    NamedTuple{fieldnames(SaxTransitionRefinementSettings)}(
        Tuple(getfield(settings, name)
              for name in fieldnames(SaxTransitionRefinementSettings)),
    )

function _sax_persistence_class(grid, row::Integer, column::Integer)
    return (
        Int(grid.dominant_mode1[row, column]),
        Int(grid.dominant_mode2[row, column]),
    )
end

"""
    rank_sax_transition_zetas(grid, zeta_candidates; gamma_range)

Rank candidate zeta rows using the existing persistence map. The score favors
rows with a wide interval of repeated class changes before the outer `12 → 22`
transition. This is only a cheap target selector; the adaptive time integrations
still decide whether stationary coexistence actually exists.
"""
function rank_sax_transition_zetas(
        grid,
        zeta_candidates;
        gamma_range::Tuple{<:Real,<:Real}=(0.40, 0.70),
        activity_half_width::Real=0.12,
        source::Tuple{<:Integer,<:Integer}=(1, 2),
        target::Tuple{<:Integer,<:Integer}=(2, 2))
    gamma = collect(float.(grid.gamma_values))
    zeta = collect(float.(grid.zeta_values))
    rows = Any[]
    for requested in zeta_candidates
        row = argmin(abs.(zeta .- float(requested)))
        changes = Int[]
        targets = Int[]
        for index in 1:(length(gamma) - 1)
            gamma_range[1] <= gamma[index] <= gamma_range[2] || continue
            left = _sax_persistence_class(grid, row, index)
            right = _sax_persistence_class(grid, row, index + 1)
            left != right && push!(changes, index)
            left == source && right == target && push!(targets, index)
        end
        if isempty(targets)
            push!(rows, (
                requested_zeta=float(requested),
                sampled_zeta=zeta[row],
                found=false,
                transition_center=NaN,
                transition_bracket=(NaN, NaN),
                activity_count=0,
                activity_spread=0.0,
                score=-Inf,
            ))
            continue
        end
        selected = last(targets)
        center = (gamma[selected] + gamma[selected + 1]) / 2
        active = [index for index in changes
                  if center - float(activity_half_width) <= gamma[index] <= center]
        spread = length(active) >= 2 ?
            gamma[last(active)] - gamma[first(active)] : 0.0
        # Count dominates, while spread resolves rows with similar alternation.
        score = length(active) + 10spread
        push!(rows, (
            requested_zeta=float(requested),
            sampled_zeta=zeta[row],
            found=true,
            transition_center=center,
            transition_bracket=(gamma[selected], gamma[selected + 1]),
            activity_count=length(active),
            activity_spread=spread,
            score=score,
        ))
    end
    return sort(rows; by=row -> row.score, rev=true)
end

function _sax_refinement_block(
        initial_state::AbstractVector{<:Real},
        gamma::Real,
        zeta::Real,
        model_p::NamedTuple,
        settings::SaxTransitionRefinementSettings)
    parameters = set_parameters(
        float(gamma), float(zeta), model_p, Int64(settings.nmodes))
    problem = ODEProblem(
        saxRN!, collect(float.(initial_state)), (0.0, settings.block_time), parameters)
    solution = solve(
        problem,
        Tsit5();
        saveat=settings.saveat,
        reltol=settings.reltol,
        abstol=settings.abstol,
    )
    DifferentialEquations.SciMLBase.successful_retcode(solution) || error(
        "transition-refinement integration failed at gamma=$(gamma), zeta=$(zeta): $(solution.retcode)",
    )
    states = Array(solution)
    fixed_state, fixed_residual = _estimate_fixed_point(
        gamma, zeta, model_p; nmodes=settings.nmodes)
    summary = _sax_tail_attractor_summary(
        states,
        fixed_state;
        classified_modes=min(3, settings.nmodes),
        threshold=settings.sustain_threshold,
        tail_fraction=settings.tail_fraction,
        # Cross-block convergence below is the authoritative test. This value
        # remains diagnostic for modulation within an individual block.
        stationarity_tolerance=settings.block_relative_tolerance,
    )
    return (
        terminal_state=copy(states[:, end]),
        times=collect(float.(solution.t)),
        states=states,
        fixed_state=fixed_state,
        fixed_residual=fixed_residual,
        summary=summary,
    )
end

function _sax_recent_block_drift(blocks, required::Integer, threshold::Real)
    length(blocks) >= required || return Inf
    recent = blocks[(end - required + 1):end]
    means = reduce(hcat, (block.mean_amplitudes for block in recent))
    center = vec(mean(means; dims=2))
    scale = max.(center, float(threshold))
    return maximum((vec(maximum(means; dims=2)) .-
                    vec(minimum(means; dims=2))) ./ scale)
end

"""Integrate in blocks until attractor statistics agree across several blocks."""
function sax_adaptive_attractor(
        initial_state::AbstractVector{<:Real},
        gamma::Real,
        zeta::Real,
        model_p::NamedTuple;
        settings::SaxTransitionRefinementSettings=SaxTransitionRefinementSettings(),
        retain_history::Bool=false)
    _validate_sax_transition_refinement_settings(settings)
    state = collect(float.(initial_state))
    summaries = Any[]
    last_block = nothing
    converged = false
    cross_block_drift = Inf
    for block_index in 1:settings.max_blocks
        block = _sax_refinement_block(state, gamma, zeta, model_p, settings)
        state = block.terminal_state
        last_block = block
        push!(summaries, (
            block=block_index,
            label=block.summary.label,
            mean_amplitudes=copy(block.summary.mean_amplitudes),
            register_contrast=block.summary.register_contrast,
            confidence=block.summary.confidence,
            within_block_drift=block.summary.relative_drift,
        ))
        cross_block_drift = _sax_recent_block_drift(
            summaries, settings.required_blocks, settings.sustain_threshold)
        if length(summaries) >= settings.required_blocks
            recent = summaries[(end - settings.required_blocks + 1):end]
            same_label = first(recent).label != 0 &&
                all(item -> item.label == first(recent).label, recent)
            converged = same_label &&
                cross_block_drift <= settings.block_relative_tolerance
            converged && break
        end
    end
    last_summary = last(summaries)
    return (
        terminal_state=state,
        label=last_summary.label,
        mean_amplitudes=last_summary.mean_amplitudes,
        register_contrast=last_summary.register_contrast,
        confidence=last_summary.confidence,
        stationary=converged,
        cross_block_drift=cross_block_drift,
        within_block_drift=last_summary.within_block_drift,
        blocks_used=length(summaries),
        integrated_time=length(summaries) * settings.block_time,
        block_summaries=summaries,
        history=retain_history ? (
            times=last_block.times,
            states=last_block.states,
            fixed_state=last_block.fixed_state,
        ) : nothing,
    )
end

function _sax_adaptive_hysteresis_direction(
        gamma_path,
        zeta::Real,
        model_p::NamedTuple,
        settings::SaxTransitionRefinementSettings,
        excitation_mode::Integer;
        verbosity::Integer=0)
    state, _ = _estimate_fixed_point(
        first(gamma_path), zeta, model_p; nmodes=settings.nmodes)
    state[2excitation_mode + 1] += settings.endpoint_excitation
    count = length(gamma_path)
    labels = Vector{Int8}(undef, count)
    contrasts = Vector{Float64}(undef, count)
    confidences = Vector{Float64}(undef, count)
    drifts = Vector{Float64}(undef, count)
    stationary = BitVector(undef, count)
    blocks = Vector{Int}(undef, count)
    amplitudes = Matrix{Float64}(undef, min(3, settings.nmodes), count)
    terminal_states = Matrix{Float64}(undef, length(state), count)
    for (index, gamma) in enumerate(gamma_path)
        run = sax_adaptive_attractor(
            state, gamma, zeta, model_p; settings=settings)
        state = run.terminal_state
        labels[index] = run.label
        contrasts[index] = run.register_contrast
        confidences[index] = run.confidence
        drifts[index] = run.cross_block_drift
        stationary[index] = run.stationary
        blocks[index] = run.blocks_used
        amplitudes[:, index] .= run.mean_amplitudes
        terminal_states[:, index] .= state
        if verbosity > 0
            @info(
                "Adaptive hysteresis progress",
                direction=gamma_path[1] <= gamma_path[end] ? :increasing : :decreasing,
                step=index,
                steps=count,
                gamma=float(gamma),
                zeta=float(zeta),
                label=labels[index],
                converged=stationary[index],
                blocks=blocks[index],
            )
        end
    end
    return (
        gamma=collect(float.(gamma_path)),
        labels=labels,
        register_contrast=contrasts,
        confidence=confidences,
        relative_drift=drifts,
        stationary=stationary,
        blocks_used=blocks,
        mean_amplitudes=amplitudes,
        terminal_states=terminal_states,
    )
end

function _reverse_sax_adaptive_path(path)
    return (
        gamma=reverse(path.gamma),
        labels=reverse(path.labels),
        register_contrast=reverse(path.register_contrast),
        confidence=reverse(path.confidence),
        relative_drift=reverse(path.relative_drift),
        stationary=reverse(path.stationary),
        blocks_used=reverse(path.blocks_used),
        mean_amplitudes=reverse(path.mean_amplitudes; dims=2),
        terminal_states=reverse(path.terminal_states; dims=2),
    )
end

"""Warm-start increasing and decreasing scans with adaptive convergence."""
function sax_adaptive_hysteresis_scan(
        model_p::NamedTuple,
        zeta::Real,
        gamma_values;
        settings::SaxTransitionRefinementSettings=SaxTransitionRefinementSettings(),
        verbosity::Integer=0)
    _validate_sax_transition_refinement_settings(settings)
    gamma = sort(unique(collect(float.(gamma_values))))
    length(gamma) >= 3 || throw(ArgumentError("at least three gamma values are required"))
    increasing = _sax_adaptive_hysteresis_direction(
        gamma, zeta, model_p, settings, 1; verbosity=verbosity)
    decreasing = _reverse_sax_adaptive_path(
        _sax_adaptive_hysteresis_direction(
            reverse(gamma), zeta, model_p, settings, 2; verbosity=verbosity),
    )
    contrast_gap = abs.(increasing.register_contrast .-
                        decreasing.register_contrast)
    stationary_pair = increasing.stationary .& decreasing.stationary
    disagreement = increasing.labels .!= decreasing.labels
    coexistence = stationary_pair .&
        (disagreement .| (contrast_gap .> settings.coexistence_contrast_gap))
    return (
        analysis=:adaptive_hysteresis,
        zeta=float(zeta),
        gamma=gamma,
        increasing=increasing,
        decreasing=decreasing,
        label_disagreement=disagreement,
        contrast_gap=contrast_gap,
        stationary_pair=stationary_pair,
        coexistence_mask=coexistence,
        coexistence_gamma=gamma[coexistence],
        unresolved_gamma=gamma[.!stationary_pair],
    )
end

function sax_poincare_diagnostics(
        history;
        mode::Integer=1,
        max_return_period::Integer=8,
        recurrence_tolerance::Real=0.05)
    states = history.states
    times = history.times
    index = 2mode + 1
    coordinate = @view(states[index, :]) .- history.fixed_state[index]
    crossings = Int[]
    fractions = Float64[]
    for column in 1:(length(coordinate) - 1)
        coordinate[column] <= 0 < coordinate[column + 1] || continue
        denominator = coordinate[column + 1] - coordinate[column]
        fraction = abs(denominator) <= eps(Float64) ? 0.0 :
            -coordinate[column] / denominator
        push!(crossings, column)
        push!(fractions, clamp(float(fraction), 0.0, 1.0))
    end
    if length(crossings) < 4
        return (
            classification=:insufficient_crossings,
            return_period=0,
            recurrence_error=NaN,
            orbit_period=NaN,
            period_cv=NaN,
            crossings=length(crossings),
        )
    end
    section = Matrix{Float64}(undef, size(states, 1), length(crossings))
    crossing_times = Vector{Float64}(undef, length(crossings))
    for (item, (column, fraction)) in enumerate(zip(crossings, fractions))
        section[:, item] .= (1 - fraction) .* states[:, column] .+
                            fraction .* states[:, column + 1]
        crossing_times[item] = (1 - fraction) * times[column] +
                               fraction * times[column + 1]
    end
    scales = vec(std(section; dims=2))
    scales .= max.(scales, sqrt(eps(Float64)))
    normalized = section ./ scales
    maximum_lag = min(Int(max_return_period), size(section, 2) ÷ 2)
    errors = [median([
        norm(@view(normalized[:, i + lag]) .- @view(normalized[:, i])) /
            sqrt(size(normalized, 1))
        for i in 1:(size(normalized, 2) - lag)
    ]) for lag in 1:maximum_lag]
    best_lag = argmin(errors)
    best_error = errors[best_lag]
    periods = [crossing_times[i + best_lag] - crossing_times[i]
               for i in 1:(length(crossing_times) - best_lag)]
    orbit_period = median(periods)
    period_cv = std(periods) / max(abs(mean(periods)), eps(Float64))
    classification = best_error <= recurrence_tolerance ? :periodic : :nonperiodic
    return (
        classification=classification,
        return_period=best_lag,
        recurrence_error=best_error,
        orbit_period=orbit_period,
        period_cv=period_cv,
        crossings=length(crossings),
    )
end

function sax_adaptive_basin_section(
        low_state::AbstractVector{<:Real},
        high_state::AbstractVector{<:Real},
        gamma::Real,
        zeta::Real,
        model_p::NamedTuple;
        settings::SaxTransitionRefinementSettings=SaxTransitionRefinementSettings(),
        parallel::Bool=true,
        verbosity::Integer=0)
    length(low_state) == length(high_state) || throw(DimensionMismatch(
        "low_state and high_state must have the same length",
    ))
    mixture = collect(range(0.0, 1.0; length=settings.basin_points))
    transverse = collect(range(-1.0, 1.0; length=settings.basin_points))
    difference = collect(float.(high_state .- low_state))
    perpendicular = zeros(Float64, length(difference))
    perpendicular[3] = -difference[5]
    perpendicular[5] = difference[3]
    if norm(perpendicular) <= sqrt(eps(Float64))
        perpendicular[3] = 1.0
    end
    perpendicular ./= norm(perpendicular)
    transverse_scale = settings.basin_transverse_fraction * norm(difference)
    labels = Matrix{Int8}(undef, length(transverse), length(mixture))
    stationary = BitMatrix(undef, size(labels))
    confidence = Matrix{Float64}(undef, size(labels))
    drift = Matrix{Float64}(undef, size(labels))
    blocks = Matrix{Int}(undef, size(labels))
    completed = Threads.Atomic{Int}(0)
    total = length(labels)
    function evaluate!(linear_index)
        row, column = Tuple(CartesianIndices(labels)[linear_index])
        lambda = mixture[column]
        initial = (1 - lambda) .* low_state .+ lambda .* high_state .+
                  transverse[row] * transverse_scale .* perpendicular
        run = sax_adaptive_attractor(
            initial, gamma, zeta, model_p; settings=settings)
        labels[row, column] = run.label
        stationary[row, column] = run.stationary
        confidence[row, column] = run.confidence
        drift[row, column] = run.cross_block_drift
        blocks[row, column] = run.blocks_used
        done = Threads.atomic_add!(completed, 1) + 1
        verbosity > 0 && (done == total || done % max(1, total ÷ 20) == 0) &&
            @info(
                "Adaptive basin progress",
                completed=done,
                total,
                percent=round(100done / total; digits=1),
                gamma=float(gamma),
                zeta=float(zeta),
                threads=Threads.nthreads(),
            )
        return nothing
    end
    if parallel && Threads.nthreads() > 1
        Threads.@threads for linear_index in eachindex(labels)
            evaluate!(linear_index)
        end
    else
        for linear_index in eachindex(labels)
            evaluate!(linear_index)
        end
    end
    uncertainty_labels = copy(labels)
    uncertainty_labels[.!stationary] .= 0
    return (
        analysis=:adaptive_basin_section,
        gamma=float(gamma),
        zeta=float(zeta),
        mixture=mixture,
        transverse=transverse,
        transverse_scale=transverse_scale,
        labels=labels,
        stationary=stationary,
        confidence=confidence,
        relative_drift=drift,
        blocks_used=blocks,
        uncertainty=sax_basin_uncertainty(
            uncertainty_labels, mixture, transverse),
    )
end

function _sax_refinement_candidate_index(hysteresis)
    eligible = findall(hysteresis.stationary_pair)
    isempty(eligible) && return argmax(hysteresis.contrast_gap)
    coexistence = findall(hysteresis.coexistence_mask)
    candidates = isempty(coexistence) ? eligible : coexistence
    return candidates[argmax(hysteresis.contrast_gap[candidates])]
end

function _sax_refinement_interpretation(low_dynamics, high_dynamics, has_coexistence)
    !has_coexistence && return :no_stationary_coexistence
    low_dynamics.classification == :periodic ||
        high_dynamics.classification == :periodic ||
        return :nonperiodic_or_long_transient
    if low_dynamics.classification == :periodic &&
            high_dynamics.classification == :periodic
        low_dynamics.return_period != high_dynamics.return_period &&
            return :period_multiplication_candidate
        return :coexisting_periodic_orbits_requires_floquet_continuation
    end
    return :mixed_periodic_nonperiodic_requires_continuation
end

"""Run the selected-zeta adaptive hysteresis, dynamics, and basin workflow."""
function compute_sax_transition_refinement(
        model_p::NamedTuple,
        persistence_grid;
        settings::SaxTransitionRefinementSettings=SaxTransitionRefinementSettings(),
        selected_zeta::Union{Nothing,Real}=nothing,
        parallel::Bool=true,
        verbosity::Integer=1)
    _validate_sax_transition_refinement_settings(settings)
    ranking = rank_sax_transition_zetas(
        persistence_grid,
        settings.zeta_candidates;
        gamma_range=settings.transition_gamma_range,
        activity_half_width=settings.transition_activity_half_width,
    )
    available = [row for row in ranking if row.found]
    isempty(available) && error("no outer 12 to 22 persistence transition was found")
    target = if isnothing(selected_zeta)
        first(available)
    else
        available[argmin(abs.([row.requested_zeta for row in available] .-
                             float(selected_zeta)))]
    end
    center = target.transition_center
    lower = max(settings.transition_gamma_range[1],
                center - settings.detailed_half_width)
    upper = min(settings.transition_gamma_range[2],
                center + settings.detailed_half_width)
    gamma = collect(range(lower, upper; length=settings.detailed_gamma_points))
    hysteresis = sax_adaptive_hysteresis_scan(
        model_p,
        target.sampled_zeta,
        gamma;
        settings=settings,
        verbosity=verbosity,
    )
    candidate_index = _sax_refinement_candidate_index(hysteresis)
    candidate_gamma = gamma[candidate_index]
    low_initial = @view hysteresis.increasing.terminal_states[:, candidate_index]
    high_initial = @view hysteresis.decreasing.terminal_states[:, candidate_index]
    low = sax_adaptive_attractor(
        low_initial, candidate_gamma, target.sampled_zeta, model_p;
        settings=settings, retain_history=true)
    high = sax_adaptive_attractor(
        high_initial, candidate_gamma, target.sampled_zeta, model_p;
        settings=settings, retain_history=true)
    low_dynamics = sax_poincare_diagnostics(
        low.history;
        max_return_period=settings.poincare_max_return_period,
        recurrence_tolerance=settings.poincare_recurrence_tolerance,
    )
    high_dynamics = sax_poincare_diagnostics(
        high.history;
        max_return_period=settings.poincare_max_return_period,
        recurrence_tolerance=settings.poincare_recurrence_tolerance,
    )
    has_coexistence = hysteresis.coexistence_mask[candidate_index]
    basin = settings.compute_basin && has_coexistence ?
        sax_adaptive_basin_section(
            low.terminal_state,
            high.terminal_state,
            candidate_gamma,
            target.sampled_zeta,
            model_p;
            settings=settings,
            parallel=parallel,
            verbosity=verbosity,
        ) : nothing
    return (
        analysis=:transition_refinement,
        ranking=ranking,
        selected=target,
        settings=_portable_sax_transition_refinement_settings(settings),
        hysteresis=hysteresis,
        representative=(
            index=candidate_index,
            gamma=candidate_gamma,
            zeta=target.sampled_zeta,
            has_stationary_coexistence=has_coexistence,
            low=(
                label=low.label,
                contrast=low.register_contrast,
                stationary=low.stationary,
                dynamics=low_dynamics,
            ),
            high=(
                label=high.label,
                contrast=high.register_contrast,
                stationary=high.stationary,
                dynamics=high_dynamics,
            ),
        ),
        basin=basin,
        interpretation=_sax_refinement_interpretation(
            low_dynamics, high_dynamics, has_coexistence),
    )
end

function plot_sax_transition_zeta_ranking(
        result;
        title::AbstractString="Persistence-map zeta targeting")
    rows = sort(result.ranking; by=row -> row.requested_zeta)
    axis = bar(
        [row.requested_zeta for row in rows],
        [max(0.0, row.score) for row in rows];
        bar_width=0.055,
        color=colorant"#8073AC",
        alpha=0.8,
        xlabel="zeta",
        ylabel="transition-activity score",
        label="",
        title=title,
        framestyle=:box,
    )
    scatter!(
        axis,
        [result.selected.requested_zeta],
        [max(0.0, result.selected.score)];
        marker=:star5,
        markersize=9,
        color=:black,
        label="selected zeta",
    )
    return axis
end

function plot_sax_refined_hysteresis(
        result;
        title::AbstractString="Adaptive register hysteresis")
    hysteresis = result.hysteresis
    axis = plot(
        hysteresis.gamma,
        hysteresis.increasing.register_contrast;
        marker=:circle,
        markersize=3,
        linewidth=2,
        color=colorant"#D73027",
        label="increasing gamma",
        xlabel="gamma",
        ylabel="(A1 - A2) / (A1 + A2)",
        ylims=(-1.05, 1.05),
        title=title,
        framestyle=:box,
    )
    plot!(axis, hysteresis.gamma, hysteresis.decreasing.register_contrast;
          marker=:diamond, markersize=3, linewidth=2,
          color=colorant"#2166AC", label="decreasing gamma")
    hline!(axis, [0.0]; color=:gray45, linestyle=:dot, label="")
    bracket = result.selected.transition_bracket
    vspan!(axis, collect(bracket); color=:gray55, alpha=0.14,
           label="persistence bracket")
    if !isempty(hysteresis.coexistence_gamma)
        scatter!(axis, hysteresis.coexistence_gamma,
                 hysteresis.increasing.register_contrast[hysteresis.coexistence_mask];
                 marker=:star5, markersize=6, color=colorant"#1A9850",
                 label="stationary coexistence")
    end
    unresolved_label = true
    for path in (hysteresis.increasing, hysteresis.decreasing)
        unresolved = .!path.stationary
        any(unresolved) && scatter!(
            axis,
            hysteresis.gamma[unresolved],
            path.register_contrast[unresolved];
            marker=:xcross,
            markersize=4,
            color=:black,
            label=unresolved_label ? "unresolved" : "",
        )
        any(unresolved) && (unresolved_label = false)
    end
    return axis
end

function plot_sax_refined_basin(
        result;
        title::AbstractString="Adaptive attractor-to-attractor basin section")
    isnothing(result.basin) && return plot(
        ; title="Basin omitted: no stationary coexistence",
        framestyle=:none,
    )
    basin = result.basin
    display_labels = copy(basin.labels)
    display_labels[.!basin.stationary] .= 0
    palette = cgrad([
        colorant"#BDBDBD",
        colorant"#D73027",
        colorant"#2166AC",
        colorant"#1A9850",
    ], 4; categorical=true)
    axis = heatmap(
        basin.mixture,
        basin.transverse,
        display_labels;
        color=palette,
        clims=(-0.5, 3.5),
        colorbar_ticks=([0, 1, 2, 3],
                        ["unresolved", "mode 1", "mode 2", "mode 3"]),
        xlabel="mixture from low (0) to high (1) attractor",
        ylabel="transverse perturbation",
        title=title,
        framestyle=:box,
    )
    scatter!(axis, [0.0, 1.0], [0.0, 0.0];
             marker=[:circle, :diamond], color=:black,
             label="reference attractors")
    return axis
end

function plot_sax_transition_refinement(result; size=(1100, 760))
    ranking = plot_sax_transition_zeta_ranking(result)
    hysteresis = plot_sax_refined_hysteresis(
        result;
        title="Adaptive scan at zeta = $(round(result.selected.sampled_zeta; digits=4))",
    )
    basin = plot_sax_refined_basin(
        result;
        title="Basin section at gamma = $(round(result.representative.gamma; digits=5))",
    )
    return plot(hysteresis, ranking, basin;
                layout=@layout([a{0.62h}; b c]), size=size)
end

const SAX_TRANSITION_REFINEMENT_CACHE_SCHEMA_VERSION = 1

function save_sax_transition_refinement_cache(
        path::AbstractString,
        result,
        model_p::NamedTuple;
        fingering::AbstractString)
    cache = (
        schema_version=SAX_TRANSITION_REFINEMENT_CACHE_SCHEMA_VERSION,
        cache_kind=:transition_refinement,
        fingering=String(fingering),
        model_signature=_sax_bifurcation_model_signature(
            model_p, result.settings.nmodes),
        saved_at_unix=time(),
        result=result,
    )
    _atomic_jld2_save(path; cache)
    return result
end

function load_sax_transition_refinement_cache(
        path::AbstractString,
        model_p::NamedTuple;
        fingering::AbstractString="Dx4")
    isfile(path) || return (
        status=:missing, result=nothing, reason="cache file is absent",
    )
    stored = try
        Logging.with_logger(Logging.NullLogger()) do
            JLD2.load(path, "cache")
        end
    catch err
        return (status=:corrupt, result=nothing, reason=sprint(showerror, err))
    end
    compatible = try
        stored.schema_version == SAX_TRANSITION_REFINEMENT_CACHE_SCHEMA_VERSION &&
        stored.cache_kind == :transition_refinement &&
        stored.fingering == String(fingering) &&
        isequal(
            stored.model_signature,
            _sax_bifurcation_model_signature(
                model_p, stored.result.settings.nmodes),
        )
    catch
        false
    end
    compatible || return (
        status=:incompatible,
        result=nothing,
        reason="schema, fingering, or acoustic model changed",
    )
    return (status=:valid, result=stored.result, reason="compatible cache")
end
