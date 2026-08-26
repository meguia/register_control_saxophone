# Targeted finite-amplitude diagnostics for register transitions.
#
# Local continuation identifies changes of equilibria and periodic orbits. The
# functions in this file answer the complementary question raised by the mode-
# persistence map: do coexisting attractors or a complicated basin boundary
# explain a register switch that is not aligned with a local bifurcation curve?

Base.@kwdef struct SaxTransitionSettings
    nmodes::Int = 8
    gamma_window::Tuple{Float64,Float64} = (0.45, 0.62)
    gamma_points::Int = 25
    integration_time::Float64 = 1000.0
    saveat::Float64 = 0.4
    tail_fraction::Float64 = 0.25
    sustain_threshold::Float64 = 1e-3
    stationarity_tolerance::Float64 = 0.15
    endpoint_excitation::Float64 = 1e-3
    ic_span::Float64 = 2e-3
    ic_points::Int = 25
    reltol::Float64 = 1e-7
    abstol::Float64 = 1e-9
end

function _validate_sax_transition_settings(settings::SaxTransitionSettings)
    settings.nmodes >= 2 || throw(ArgumentError("nmodes must be at least two"))
    0 < settings.gamma_window[1] < settings.gamma_window[2] < 1 ||
        throw(ArgumentError("gamma_window must be ordered inside (0,1)"))
    settings.gamma_points >= 2 || throw(ArgumentError("gamma_points must be at least two"))
    settings.integration_time > 0 || throw(ArgumentError("integration_time must be positive"))
    settings.saveat > 0 || throw(ArgumentError("saveat must be positive"))
    0 < settings.tail_fraction <= 1 || throw(ArgumentError("tail_fraction must lie in (0,1]"))
    settings.sustain_threshold > 0 || throw(ArgumentError("sustain_threshold must be positive"))
    settings.stationarity_tolerance >= 0 ||
        throw(ArgumentError("stationarity_tolerance must be nonnegative"))
    settings.endpoint_excitation >= 0 ||
        throw(ArgumentError("endpoint_excitation must be nonnegative"))
    settings.ic_span > 0 || throw(ArgumentError("ic_span must be positive"))
    settings.ic_points >= 3 || throw(ArgumentError("ic_points must be at least three"))
    return settings
end

_portable_sax_transition_settings(settings::SaxTransitionSettings) =
    NamedTuple{fieldnames(SaxTransitionSettings)}(
        Tuple(getfield(settings, name) for name in fieldnames(SaxTransitionSettings)),
    )

"""
    sax_persistence_transition_boundary(grid, zeta; source=(1,2), target=(2,2))

Locate the finite-amplitude `source → target` register transition in the row of
an existing persistence grid nearest `zeta`. The returned bracket is bounded by
two actually simulated gamma values, so it is a target for refined hysteresis
and basin calculations rather than a claim of a bifurcation location.
"""
function sax_persistence_transition_boundary(
        grid,
        zeta::Real;
        source::Tuple{<:Integer,<:Integer}=(1, 2),
        target::Tuple{<:Integer,<:Integer}=(2, 2),
        gamma_range::Tuple{<:Real,<:Real}=(-Inf, Inf))
    gamma_values = collect(float.(grid.gamma_values))
    zeta_values = collect(float.(grid.zeta_values))
    row = argmin(abs.(zeta_values .- float(zeta)))
    first_map = grid.dominant_mode1
    second_map = grid.dominant_mode2
    candidates = NamedTuple[]
    for index in 1:(length(gamma_values) - 1)
        gamma_range[1] <= gamma_values[index] <= gamma_range[2] || continue
        left = (Int(first_map[row, index]), Int(second_map[row, index]))
        right = (Int(first_map[row, index + 1]), Int(second_map[row, index + 1]))
        left == source && right == target || continue
        push!(candidates, (
            left_index=index,
            right_index=index + 1,
            gamma_bracket=(gamma_values[index], gamma_values[index + 1]),
            gamma_center=(gamma_values[index] + gamma_values[index + 1]) / 2,
            left_class=left,
            right_class=right,
        ))
    end
    isempty(candidates) && return (
        found=false,
        requested_zeta=float(zeta),
        sampled_zeta=zeta_values[row],
        row=row,
        source=(Int(source[1]), Int(source[2])),
        target=(Int(target[1]), Int(target[2])),
        candidates=NamedTuple[],
        gamma_bracket=(NaN, NaN),
        gamma_center=NaN,
    )
    # The high-gamma transition is the relevant outer boundary when a row has
    # isolated one-cell alternations before entering the persistent 22 region.
    selected = candidates[end]
    return merge((
        found=true,
        requested_zeta=float(zeta),
        sampled_zeta=zeta_values[row],
        row=row,
        source=(Int(source[1]), Int(source[2])),
        target=(Int(target[1]), Int(target[2])),
        candidates=candidates,
    ), selected)
end

function _sax_tail_attractor_summary(
        states::AbstractMatrix{<:Real},
        fixed_state::AbstractVector{<:Real};
        classified_modes::Int=3,
        threshold::Real=1e-3,
        tail_fraction::Real=0.25,
        stationarity_tolerance::Real=0.15)
    classified_modes >= 2 || throw(ArgumentError("classified_modes must be at least two"))
    size(states, 2) >= 2 || throw(ArgumentError("at least two saved states are required"))
    start_index = clamp(
        floor(Int, (1 - float(tail_fraction)) * size(states, 2)) + 1,
        1,
        size(states, 2) - 1,
    )
    amplitudes = Matrix{Float64}(undef, classified_modes, size(states, 2) - start_index + 1)
    for mode in 1:classified_modes
        state_index = 2mode + 1
        center = (fixed_state[state_index], fixed_state[state_index + 1])
        amplitudes[mode, :] .= _mode_amplitude_series(
            @view(states[:, start_index:end]), mode; center=center)
    end
    means = vec(mean(amplitudes; dims=2))
    split = max(1, size(amplitudes, 2) ÷ 2)
    early = vec(mean(@view(amplitudes[:, 1:split]); dims=2))
    late_start = min(split + 1, size(amplitudes, 2))
    late = vec(mean(@view(amplitudes[:, late_start:end]); dims=2))
    relative_drift = maximum(abs.(late .- early) ./
                             max.(means, float(threshold)))
    ranking = sortperm(means; rev=true)
    dominant = means[ranking[1]] >= threshold ? Int8(ranking[1]) : Int8(0)
    second = length(ranking) > 1 ? means[ranking[2]] : 0.0
    confidence = dominant == 0 ? 0.0 :
        (means[ranking[1]] - second) / (means[ranking[1]] + second + eps(Float64))
    return (
        label=dominant,
        mean_amplitudes=means,
        register_contrast=(means[1] - means[2]) /
                          (means[1] + means[2] + eps(Float64)),
        confidence=confidence,
        relative_drift=relative_drift,
        stationary=relative_drift <= stationarity_tolerance,
        tail_start_index=start_index,
    )
end

function _integrate_sax_transition(
        initial_state::AbstractVector{<:Real},
        gamma::Real,
        zeta::Real,
        model_p::NamedTuple,
        settings::SaxTransitionSettings)
    parameters = set_parameters(
        float(gamma), float(zeta), model_p, Int64(settings.nmodes))
    problem = ODEProblem(
        saxRN!,
        collect(float.(initial_state)),
        (0.0, settings.integration_time),
        parameters,
    )
    solution = solve(
        problem,
        Tsit5();
        saveat=settings.saveat,
        reltol=settings.reltol,
        abstol=settings.abstol,
    )
    DifferentialEquations.SciMLBase.successful_retcode(solution) || error(
        "transition integration failed at gamma=$(gamma), zeta=$(zeta): $(solution.retcode)",
    )
    states = Array(solution)
    fixed_state, fixed_residual = _estimate_fixed_point(
        gamma, zeta, model_p; nmodes=settings.nmodes)
    classified_modes = min(3, settings.nmodes)
    summary = _sax_tail_attractor_summary(
        states,
        fixed_state;
        classified_modes=classified_modes,
        threshold=settings.sustain_threshold,
        tail_fraction=settings.tail_fraction,
        stationarity_tolerance=settings.stationarity_tolerance,
    )
    return (
        terminal_state=copy(states[:, end]),
        fixed_residual=fixed_residual,
        summary=summary,
    )
end

function _sax_hysteresis_direction(
        gamma_path,
        zeta::Real,
        model_p::NamedTuple,
        settings::SaxTransitionSettings,
        excitation_mode::Int;
        verbosity::Int=0)
    first_gamma = first(gamma_path)
    state, _ = _estimate_fixed_point(
        first_gamma, zeta, model_p; nmodes=settings.nmodes)
    state[2excitation_mode + 1] += settings.endpoint_excitation
    count = length(gamma_path)
    labels = Vector{Int8}(undef, count)
    contrasts = Vector{Float64}(undef, count)
    confidences = Vector{Float64}(undef, count)
    drifts = Vector{Float64}(undef, count)
    stationary = BitVector(undef, count)
    amplitudes = Matrix{Float64}(undef, min(3, settings.nmodes), count)
    terminal_states = Matrix{Float64}(undef, length(state), count)
    for (index, gamma) in enumerate(gamma_path)
        run = _integrate_sax_transition(state, gamma, zeta, model_p, settings)
        state = run.terminal_state
        labels[index] = run.summary.label
        contrasts[index] = run.summary.register_contrast
        confidences[index] = run.summary.confidence
        drifts[index] = run.summary.relative_drift
        stationary[index] = run.summary.stationary
        amplitudes[:, index] .= run.summary.mean_amplitudes
        terminal_states[:, index] .= state
        if verbosity > 0
            @info(
                "Hysteresis continuation step",
                direction=gamma_path[1] <= gamma_path[end] ? :increasing : :decreasing,
                step=index,
                steps=count,
                gamma=float(gamma),
                zeta=float(zeta),
                label=labels[index],
                stationary=stationary[index],
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
        mean_amplitudes=amplitudes,
        terminal_states=terminal_states,
    )
end

"""
    sax_hysteresis_scan(model_p, zeta; settings, gamma_values=nothing)

Perform increasing- and decreasing-gamma continuation by time integration. The
terminal state at one gamma is the initial state at the next. The increasing
path starts with a mode-1 perturbation and the decreasing path with mode 2.
Different register labels or contrasts at the same gamma are counted as direct
evidence of coexistence and hysteresis only when both tails pass the
stationarity check; unresolved points are reported separately.
"""
function sax_hysteresis_scan(
        model_p::NamedTuple,
        zeta::Real;
        settings::SaxTransitionSettings=SaxTransitionSettings(),
        gamma_values::Union{Nothing,AbstractVector}=nothing,
        verbosity::Int=0)
    _validate_sax_transition_settings(settings)
    gamma = gamma_values === nothing ?
        collect(range(settings.gamma_window...; length=settings.gamma_points)) :
        sort(unique(collect(float.(gamma_values))))
    length(gamma) >= 2 || throw(ArgumentError("at least two gamma values are required"))
    increasing = _sax_hysteresis_direction(
        gamma, zeta, model_p, settings, 1; verbosity=verbosity)
    decreasing_raw = _sax_hysteresis_direction(
        reverse(gamma), zeta, model_p, settings, 2; verbosity=verbosity)
    decreasing = map(values(decreasing_raw)) do value
        value isa AbstractMatrix ? reverse(value; dims=2) : reverse(value)
    end
    decreasing = NamedTuple{keys(decreasing_raw)}(Tuple(decreasing))
    disagreement = increasing.labels .!= decreasing.labels
    contrast_gap = abs.(increasing.register_contrast .- decreasing.register_contrast)
    stationary_pair = increasing.stationary .& decreasing.stationary
    coexistence_mask = stationary_pair .&
        (disagreement .| (contrast_gap .> 0.5))
    return (
        analysis=:hysteresis,
        zeta=float(zeta),
        gamma=gamma,
        increasing=increasing,
        decreasing=decreasing,
        label_disagreement=disagreement,
        contrast_gap=contrast_gap,
        stationary_pair=stationary_pair,
        coexistence_gamma=gamma[coexistence_mask],
        unresolved_gamma=gamma[.!stationary_pair],
        configuration=(
            zeta=float(zeta),
            gamma=gamma,
            settings=_portable_sax_transition_settings(settings),
        ),
    )
end

"""
    sax_initial_condition_basin(model_p, gamma, zeta; settings, parallel=true)

Classify a two-dimensional initial-condition section through the equilibrium.
The axes perturb the pressure coordinates of acoustic modes 1 and 2. Every
point is integrated independently, so this diagnostic can use Julia threads.
The output includes convergence flags and confidence, not only register labels.
"""
function sax_initial_condition_basin(
        model_p::NamedTuple,
        gamma::Real,
        zeta::Real;
        settings::SaxTransitionSettings=SaxTransitionSettings(),
        mode1_offsets::Union{Nothing,AbstractVector}=nothing,
        mode2_offsets::Union{Nothing,AbstractVector}=nothing,
        parallel::Bool=true,
        verbosity::Int=0)
    _validate_sax_transition_settings(settings)
    offsets1 = mode1_offsets === nothing ?
        collect(range(-settings.ic_span, settings.ic_span; length=settings.ic_points)) :
        sort(unique(collect(float.(mode1_offsets))))
    offsets2 = mode2_offsets === nothing ?
        collect(range(-settings.ic_span, settings.ic_span; length=settings.ic_points)) :
        sort(unique(collect(float.(mode2_offsets))))
    isempty(offsets1) && throw(ArgumentError("mode1_offsets cannot be empty"))
    isempty(offsets2) && throw(ArgumentError("mode2_offsets cannot be empty"))
    fixed_state, fixed_residual = _estimate_fixed_point(
        gamma, zeta, model_p; nmodes=settings.nmodes)
    labels = Matrix{Int8}(undef, length(offsets2), length(offsets1))
    confidence = Matrix{Float64}(undef, size(labels))
    relative_drift = Matrix{Float64}(undef, size(labels))
    stationary = BitMatrix(undef, size(labels))
    completed = Threads.Atomic{Int}(0)
    total = length(labels)

    function evaluate!(linear_index)
        row, column = Tuple(CartesianIndices(labels)[linear_index])
        initial = copy(fixed_state)
        initial[3] += offsets1[column]
        initial[5] += offsets2[row]
        run = _integrate_sax_transition(
            initial, gamma, zeta, model_p, settings)
        labels[row, column] = run.summary.label
        confidence[row, column] = run.summary.confidence
        relative_drift[row, column] = run.summary.relative_drift
        stationary[row, column] = run.summary.stationary
        done = Threads.atomic_add!(completed, 1) + 1
        if verbosity > 0 && (done == total || done % max(1, total ÷ 20) == 0)
            @info(
                "Initial-condition basin progress",
                completed=done,
                total,
                percent=round(100done / total; digits=1),
                gamma=float(gamma),
                zeta=float(zeta),
                threads=Threads.nthreads(),
            )
        end
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
    uncertainty = sax_basin_uncertainty(
        uncertainty_labels, offsets1, offsets2)
    return (
        analysis=:initial_condition_basin,
        gamma=float(gamma),
        zeta=float(zeta),
        mode1_offsets=offsets1,
        mode2_offsets=offsets2,
        labels=labels,
        confidence=confidence,
        relative_drift=relative_drift,
        stationary=stationary,
        fixed_residual=fixed_residual,
        uncertainty=uncertainty,
        configuration=(
            gamma=float(gamma),
            zeta=float(zeta),
            mode1_offsets=offsets1,
            mode2_offsets=offsets2,
            settings=_portable_sax_transition_settings(settings),
        ),
    )
end

"""
    sax_basin_uncertainty(labels, x, y; pixel_scales=nothing)

Estimate an uncertainty exponent from the fraction of label-disagreeing pairs
separated by several grid spacings. This is a resolution diagnostic, not proof
of a fractal basin. A smooth boundary in a two-dimensional section normally
gives an exponent near one; a well-resolved exponent between zero and one is
consistent with a fractal boundary.
"""
function sax_basin_uncertainty(
        labels::AbstractMatrix{<:Integer},
        x::AbstractVector{<:Real},
        y::AbstractVector{<:Real};
        pixel_scales::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    size(labels) == (length(y), length(x)) || throw(DimensionMismatch(
        "labels must have size (length(y), length(x))",
    ))
    maximum_scale = max(1, min(size(labels)...) ÷ 4)
    scales = pixel_scales === nothing ?
        collect(1:min(6, maximum_scale)) :
        sort(unique(Int.(pixel_scales)))
    filter!(scale -> 1 <= scale <= maximum_scale, scales)
    isempty(scales) && return (
        exponent=NaN, intercept=NaN, r_squared=NaN,
        epsilon=Float64[], uncertain_fraction=Float64[], pair_count=Int[],
        interpretation=:insufficient_resolution,
    )
    dx = length(x) > 1 ? median(abs.(diff(float.(x)))) : 1.0
    dy = length(y) > 1 ? median(abs.(diff(float.(y)))) : 1.0
    epsilon = Float64[]
    fractions = Float64[]
    pair_counts = Int[]
    for scale in scales
        differing = 0
        pairs = 0
        for row in axes(labels, 1), column in axes(labels, 2)
            label = labels[row, column]
            label == 0 && continue
            if column + scale <= size(labels, 2)
                other = labels[row, column + scale]
                other == 0 || begin
                    differing += label != other
                    pairs += 1
                end
            end
            if row + scale <= size(labels, 1)
                other = labels[row + scale, column]
                other == 0 || begin
                    differing += label != other
                    pairs += 1
                end
            end
        end
        push!(epsilon, scale * sqrt((dx^2 + dy^2) / 2))
        push!(fractions, pairs == 0 ? NaN : differing / pairs)
        push!(pair_counts, pairs)
    end
    usable = findall(index -> isfinite(fractions[index]) && fractions[index] > 0,
                     eachindex(fractions))
    if length(usable) < 3
        return (
            exponent=NaN, intercept=NaN, r_squared=NaN,
            epsilon=epsilon, uncertain_fraction=fractions, pair_count=pair_counts,
            interpretation=:insufficient_resolution,
        )
    end
    design = hcat(ones(length(usable)), log.(epsilon[usable]))
    response = log.(fractions[usable])
    coefficients = design \ response
    fitted = design * coefficients
    total_variation = sum(abs2, response .- mean(response))
    r_squared = total_variation <= eps(Float64) ? NaN :
        1 - sum(abs2, response .- fitted) / total_variation
    exponent = coefficients[2]
    interpretation = r_squared < 0.8 ? :poor_scaling :
        0 < exponent < 0.8 ? :fractal_consistent :
        0.8 <= exponent <= 1.2 ? :smooth_consistent : :unresolved
    return (
        exponent=exponent,
        intercept=coefficients[1],
        r_squared=r_squared,
        epsilon=epsilon,
        uncertain_fraction=fractions,
        pair_count=pair_counts,
        interpretation=interpretation,
    )
end

"""Plot forward/backward register contrast and mark nonstationary samples."""
function plot_sax_hysteresis(
        result;
        title::AbstractString="Register hysteresis",
        size::Tuple{Int,Int}=(780, 430))
    axis = plot(
        result.gamma,
        result.increasing.register_contrast;
        marker=:circle,
        linewidth=2.2,
        color=colorant"#D73027",
        label="increasing gamma",
        xlabel="gamma",
        ylabel="(A1 - A2) / (A1 + A2)",
        ylims=(-1.05, 1.05),
        title=title,
        size=size,
        framestyle=:box,
    )
    plot!(axis, result.gamma, result.decreasing.register_contrast;
          marker=:diamond, linewidth=2.2, color=colorant"#2166AC",
          label="decreasing gamma")
    hline!(axis, [0.0]; color=:gray45, linestyle=:dot, linewidth=1, label="")
    for path in (result.increasing, result.decreasing)
        bad = .!path.stationary
        any(bad) && scatter!(axis, result.gamma[bad], path.register_contrast[bad];
                             marker=:xcross, markersize=7, color=:black,
                             label="nonstationary tail")
    end
    return axis
end

"""Plot the mode-1/mode-2 initial-condition basin section."""
function plot_sax_initial_condition_basin(
        result;
        title::AbstractString="Initial-condition basin section",
        size::Tuple{Int,Int}=(620, 520))
    palette = cgrad([
        colorant"#BDBDBD",
        colorant"#D73027",
        colorant"#2166AC",
        colorant"#1A9850",
    ], 4; categorical=true)
    axis = heatmap(
        result.mode1_offsets,
        result.mode2_offsets,
        result.labels;
        color=palette,
        clims=(-0.5, 3.5),
        colorbar_ticks=([0, 1, 2, 3], ["silent", "mode 1", "mode 2", "mode 3"]),
        xlabel="initial mode-1 pressure offset",
        ylabel="initial mode-2 pressure offset",
        title=title,
        size=size,
        aspect_ratio=:equal,
        framestyle=:box,
    )
    unresolved = findall(.!result.stationary)
    if !isempty(unresolved)
        scatter!(
            axis,
            [result.mode1_offsets[index[2]] for index in unresolved],
            [result.mode2_offsets[index[1]] for index in unresolved];
            marker=:xcross,
            markersize=3,
            color=:black,
            label="nonstationary tail",
        )
    end
    scatter!(axis, [0.0], [0.0]; marker=:cross, color=:black,
             markersize=6, label="equilibrium")
    return axis
end

const SAX_TRANSITION_CACHE_SCHEMA_VERSION = 1

"""Atomically save a portable hysteresis or initial-condition result."""
function save_sax_transition_cache(
        path::AbstractString,
        result,
        model_p::NamedTuple;
        fingering::AbstractString)
    result.analysis in (:hysteresis, :initial_condition_basin) ||
        throw(ArgumentError("unsupported transition analysis $(result.analysis)"))
    settings = result.configuration.settings
    cache = (
        schema_version=SAX_TRANSITION_CACHE_SCHEMA_VERSION,
        fingering=String(fingering),
        analysis=result.analysis,
        configuration=result.configuration,
        model_signature=_sax_bifurcation_model_signature(model_p, settings.nmodes),
        saved_at_unix=time(),
        result=result,
    )
    _atomic_jld2_save(path; cache)
    return result
end

"""
    load_sax_transition_cache(path, model_p; fingering, analysis, configuration)

Load a transition result only when its analysis type, grid, numerical settings,
fingering, and modal model signature exactly match the requested calculation.
"""
function load_sax_transition_cache(
        path::AbstractString,
        model_p::NamedTuple;
        fingering::AbstractString,
        analysis::Symbol,
        configuration)
    isfile(path) || return (status=:missing, result=nothing, reason="cache file is absent")
    stored = try
        Logging.with_logger(Logging.NullLogger()) do
            JLD2.load(path, "cache")
        end
    catch err
        return (status=:corrupt, result=nothing, reason=sprint(showerror, err))
    end
    settings = configuration.settings
    expected_model = _sax_bifurcation_model_signature(model_p, settings.nmodes)
    checks = try
        (
            stored.schema_version == SAX_TRANSITION_CACHE_SCHEMA_VERSION =>
                "cache schema changed",
            stored.fingering == String(fingering) => "fingering changed",
            stored.analysis == analysis => "analysis type changed",
            isequal(stored.configuration, configuration) => "configuration changed",
            isequal(stored.model_signature, expected_model) =>
                "acoustic model parameters changed",
        )
    catch err
        return (status=:corrupt, result=nothing, reason=sprint(showerror, err))
    end
    for (valid, reason) in checks
        valid || return (status=:incompatible, result=nothing, reason=reason)
    end
    return (status=:valid, result=stored.result, reason="")
end
