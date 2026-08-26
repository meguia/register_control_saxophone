# in VSCode expand/collapse regions with Ctrl+K followed by Ctrl+L
# to collapse all regions: Ctrl+K followed by Ctrl+0
# to expand all regions: Ctrl+K followed by Ctrl+J

#region IMPORTS

using Measures
using Plots
using Random
using Statistics

const HAS_OPTIMALTRANSPORT = Base.find_package("OptimalTransport") !== nothing
if HAS_OPTIMALTRANSPORT
    using OptimalTransport
end

#endregion

#region API CONTRACT

if !isdefined(@__MODULE__, :RT_SAX_STATISTICS_PUBLIC_API)
    const RT_SAX_STATISTICS_PUBLIC_API = (
        :statistics_public_api_symbols,
        :statistics_internal_helper_symbols,
        :statistics_public_api,
        :statistics_missing_symbols,
        :statistics_validate_dependencies,
        :trial_trajectory,
        :onoff_regions,
        :stable_windows_nonlegato,
        :segment_labels,
        :points_in_window,
        :kde_overlap_metrics,
        :compare_interval_metrics,
        :compute_interval_pair_rows,
        :WassersteinGridSpec,
        :KDEWassersteinResult,
        :normalize_point_set,
        :normalize_point_sets,
        :buffer_point_set,
        :select_trials_by_condition,
        :compute_kde_wasserstein_distances,
        :compute_kde_wasserstein_distances_optimaltransport,
        :collect_wasserstein_reference_point_sets,
        :compute_global_wasserstein_scales,
        :permutation_test_wasserstein_between_sets,
        :permutation_test_wasserstein_rows_by_family,
        :permutation_test_wasserstein_rows_by_family_subject_stratified,
        :nonlegato_point_sets_from_trial,
        :nonlegato_point_sets_from_trials,
        :compute_nonlegato_trial_wasserstein_distances,
        :compute_nonlegato_trial_wasserstein_distances_optimaltransport,
        :compute_nonlegato_pooled_wasserstein_distances,
        :compute_nonlegato_pooled_wasserstein_distances_optimaltransport,
        :compute_paper_wasserstein_summary,
        :plot_paper_wasserstein_figure,
        :occupancy_map,
        :occupancy_overlap,
        :plot_kde_contour,
    )
end

if !isdefined(@__MODULE__, :RT_SAX_STATISTICS_INTERNAL_HELPERS)
    const RT_SAX_STATISTICS_INTERNAL_HELPERS = (
        :_time_axis_seconds,
        :_default_calibrations,
        :_pf_interval_from_timewindow_fallback,
        :_pf_interval_from_timewindow_safe,
        :_overtone_interval_points,
        :stable_windows_legato,
        :legato_segment_labels,
        :_legato_note_points_from_trial,
        :_parse_wasserstein_scales,
        :_coerce_wasserstein_grid_spec,
        :_normalize_point_set_matrix,
        :_grid_from_normalized_point_sets,
        :_points_from_mask,
        :elliptical_offsets,
        :dilate_mask,
        :_sinkhorn_wasserstein_distance,
        :_sinkhorn_wasserstein_distance_optimaltransport,
        :_pairwise_cost_matrix,
    )
end

"""
Return the stable public API symbols for the statistics layer.
"""
statistics_public_api_symbols() = RT_SAX_STATISTICS_PUBLIC_API

"""
Return internal helper symbols (implementation details, not API contract).
"""
statistics_internal_helper_symbols() = RT_SAX_STATISTICS_INTERNAL_HELPERS

"""
Return a NamedTuple with currently bound public API callables.
Useful for introspection in Pluto and scripts.
"""
function statistics_public_api()
    pairs = Pair{Symbol,Any}[]
    for name in RT_SAX_STATISTICS_PUBLIC_API
        isdefined(@__MODULE__, name) || continue
        push!(pairs, name => getfield(@__MODULE__, name))
    end
    return (; pairs...)
end

"""
Return module symbols declared in the API/helper contracts that are currently missing.
Useful to catch incomplete refactors early.
"""
function statistics_missing_symbols()
    required = unique(vcat(
        collect(RT_SAX_STATISTICS_PUBLIC_API),
        collect(RT_SAX_STATISTICS_INTERNAL_HELPERS),
    ))

    missing = Symbol[]
    for name in required
        isdefined(@__MODULE__, name) || push!(missing, name)
    end
    return missing
end

"""
Validate that all symbols declared in API/helper contracts are defined.
Set `strict = false` to return a warning instead of throwing.
"""
function statistics_validate_dependencies(; strict::Bool = true)
    missing = statistics_missing_symbols()
    isempty(missing) && return true

    msg = "Missing statistics symbols: " * join(string.(missing), ", ")
    if strict
        error(msg)
    else
        @warn msg
        return false
    end
end

#endregion

#region TYPES

if !isdefined(@__MODULE__, :KDETrialResult)
    Base.@kwdef struct KDETrialResult
        subject_id::String
        type::Symbol
        task::Symbol
        take::Int
        block::Int
        success::Bool
        space::Symbol
        label1::Symbol
        label2::Symbol
        n1::Int
        n2::Int
        overlap::Float64
        hellinger::Float64
        centroid_dist::Float64
        bandwidth_x::Float64
        bandwidth_y::Float64
        window1::Tuple{Float64,Float64}
        window2::Tuple{Float64,Float64}
    end
end

#endregion

#region CORE TRAJECTORY UTILITIES

function interp1_linear(t::AbstractVector{<:Real},
                        x::AbstractVector{<:Real},
                        tq::AbstractVector{<:Real})
    @assert length(t) == length(x)
    @assert issorted(t)

    y = similar(tq, Float64)

    for k in eachindex(tq)
        tau = tq[k]

        if tau <= t[1]
            y[k] = x[1]
        elseif tau >= t[end]
            y[k] = x[end]
        else
            i = searchsortedlast(t, tau)
            t0, t1 = t[i], t[i + 1]
            x0, x1 = x[i], x[i + 1]
            alpha = (tau - t0) / (t1 - t0)
            y[k] = (1 - alpha) * x0 + alpha * x1
        end
    end

    return y
end

pressure_to_gamma(P; P_sat = 1.6) = clamp(P / P_sat, 0.0, 1.0)

force_to_zeta(F; Fmin = 5.9, Fmax = 29.8) =
    clamp((Fmax - F) / (Fmax - Fmin), 0.0, 1.0)

function _time_axis_seconds(t::AbstractVector{<:Real})
    tf = Float64.(t)
    isempty(tf) && return tf

    # Experiment sensor streams are typically stored in milliseconds,
    # while onoff windows are in seconds.
    if maximum(tf) > 100.0
        return tf ./ 1000.0
    end

    return tf
end

function trial_trajectory(tr;
    dt::Float64 = 0.001,
    pressure_calib = pressure_from_adc,
    force_calib = force_newton_from_adc,
    space::Symbol = :physical,
    P_sat::Float64 = 1.6,
    Fmin::Float64 = 5.9,
    Fmax::Float64 = 29.8
)
    t1 = _time_axis_seconds(tr.t1)
    t2 = _time_axis_seconds(tr.t2)

    tmin = max(first(t1), first(t2))
    tmax = min(last(t1), last(t2))
    @assert tmax > tmin "No temporal overlap between pressure and force traces."

    tq = collect(tmin:dt:tmax)

    p_raw = interp1_linear(t1, tr.v1, tq)
    f_raw = interp1_linear(t2, tr.v2, tq)

    P = pressure_calib.(p_raw)
    F = force_calib.(f_raw)

    if space == :physical
        return (t = tq, x = P, y = F, xname = :pressure, yname = :force)
    elseif space == :parameter
        tr.type == :Model || error("Parameter-space analysis is only defined for :Model trials.")
        gamma = pressure_to_gamma.(P; P_sat = P_sat)
        zeta = force_to_zeta.(F; Fmin = Fmin, Fmax = Fmax)
        return (t = tq, x = gamma, y = zeta, xname = :gamma, yname = :zeta)
    else
        error("space must be :physical or :parameter")
    end
end

function onoff_regions(tr; mode::Union{Nothing,Int} = nothing)
    hasproperty(tr, :onoff) || return NTuple{3,Float64}[]

    raw = getproperty(tr, :onoff)
    segs = NTuple{3,Float64}[]

    for seg in raw
        if length(seg) == 3
            push!(segs, (Float64(seg[1]), Float64(seg[2]), Float64(seg[3])))
        elseif length(seg) == 2
            # Legacy format without mode label.
            push!(segs, (Float64(seg[1]), Float64(seg[2]), NaN))
        end
    end

    sort!(segs, by = s -> s[1])

    mode === nothing && return segs

    # If mode labels are present, filter by requested mode.
    has_mode_info = any(s -> isfinite(s[3]), segs)
    if has_mode_info
        return [s for s in segs if isfinite(s[3]) && round(Int, s[3]) == mode]
    end

    # Legacy (2-tuple) data: return chronological segments as best effort.
    return segs
end

function stable_windows_nonlegato(tr;
    mode::Int = 2,
    inner_pad::Float64 = 0.05,
    min_dur::Float64 = 0.05,
    edge_exclusion_frac::Float64 = 0.10
)
    segs = onoff_regions(tr; mode = mode)

    length(segs) >= 2 || error("Need at least two mode-$mode regions for a non-legato trial.")

    # Build candidate trimmed windows and keep only valid-duration segments.
    candidates = NamedTuple{(:w, :dur, :mid),Tuple{Tuple{Float64,Float64},Float64,Float64}}[]
    for seg in segs
        w = (seg[1] + inner_pad, seg[2] - inner_pad)
        dur = w[2] - w[1]
        if dur >= min_dur
            mid = 0.5 * (w[1] + w[2])
            push!(candidates, (w = w, dur = dur, mid = mid))
        end
    end

    length(candidates) >= 2 || error("Need at least two valid stable windows after trimming.")

    # Exclude boundary windows, which are often onset/offset artifacts.
    mids = [c.mid for c in candidates]
    mmin = minimum(mids)
    mmax = maximum(mids)
    mspan = mmax - mmin

    central_candidates = candidates
    if edge_exclusion_frac > 0 && mspan > 0
        lo = mmin + edge_exclusion_frac * mspan
        hi = mmax - edge_exclusion_frac * mspan
        central_candidates = [c for c in candidates if lo <= c.mid <= hi]
        if length(central_candidates) < 2
            central_candidates = candidates
        end
    end

    # Prefer longest central windows, then restore chronological order.
    sort!(central_candidates; by = c -> c.dur, rev = true)
    top2 = [central_candidates[1].w, central_candidates[2].w]
    sort!(top2; by = w -> w[1])

    return top2[1], top2[2]
end

function segment_labels(task::Symbol)
    if task == :NonlegatoAsc
        return (:low, :high)
    elseif task == :NonlegatoDesc
        return (:high, :low)
    else
        error("segment_labels is implemented only for :NonlegatoAsc and :NonlegatoDesc")
    end
end

function points_in_window(t, x, y, win::Tuple{Float64,Float64})
    idx = findall(i -> win[1] <= t[i] <= win[2], eachindex(t))

    # Robustness for mixed time units between trajectory (ms vs s)
    # and onoff windows. Prefer the scale that captures more samples.
    if length(idx) <= 3
        win_ms = (1000.0 * win[1], 1000.0 * win[2])
        idx_ms = findall(i -> win_ms[1] <= t[i] <= win_ms[2], eachindex(t))

        win_s = (win[1] / 1000.0, win[2] / 1000.0)
        idx_s = findall(i -> win_s[1] <= t[i] <= win_s[2], eachindex(t))

        if length(idx_ms) > length(idx)
            idx = idx_ms
        end
        if length(idx_s) > length(idx)
            idx = idx_s
        end
    end

    length(idx) > 3 || error("Too few samples inside window $(win).")
    return hcat(x[idx], y[idx])
end

function _default_calibrations(; pressure_kwargs = NamedTuple(), force_kwargs = NamedTuple())
    mod = @__MODULE__
    pressure_calib = if isdefined(mod, :pressure_from_adc)
        x -> getfield(mod, :pressure_from_adc)(x; pressure_kwargs...)
    else
        x -> Float64(x)
    end

    force_calib = if isdefined(mod, :force_newton_from_adc)
        x -> getfield(mod, :force_newton_from_adc)(x; force_kwargs...)
    else
        x -> Float64(x)
    end

    return pressure_calib, force_calib
end

function _pf_interval_from_timewindow_fallback(tr, t_start::Real, t_stop::Real;
                                               axis_scaling::Symbol = :physical,
                                               pressure_kwargs = NamedTuple(),
                                               force_kwargs = NamedTuple())
    pressure_calib, force_calib = _default_calibrations(; pressure_kwargs=pressure_kwargs,
                                                        force_kwargs=force_kwargs)

    if axis_scaling === :raw
        traj = trial_trajectory(tr;
                                pressure_calib = x -> Float64(x),
                                force_calib = x -> Float64(x),
                                space = :physical)
    elseif axis_scaling === :physical
        traj = trial_trajectory(tr;
                                pressure_calib = pressure_calib,
                                force_calib = force_calib,
                                space = :physical)
    elseif axis_scaling === :parameter || axis_scaling === :parameters
        traj = trial_trajectory(tr;
                                pressure_calib = pressure_calib,
                                force_calib = force_calib,
                                space = :parameter)
    else
        error("axis_scaling must be one of :raw, :physical, :parameter, :parameters")
    end

    idx = findall(i -> t_start <= traj.t[i] <= t_stop, eachindex(traj.t))
    isempty(idx) && return nothing

    return (t = traj.t[idx], pressure = traj.x[idx], force = traj.y[idx])
end

function _pf_interval_from_timewindow_safe(tr, t_start::Real, t_stop::Real;
                                           axis_scaling::Symbol = :physical,
                                           pressure_kwargs = NamedTuple(),
                                           force_kwargs = NamedTuple(),
                                           param_map_override = nothing,
                                           param_map_config::AbstractString = "./rt_sax_configuration.toml")
    mod = @__MODULE__
    if isdefined(mod, :_pf_interval_from_timewindow)
        return getfield(mod, :_pf_interval_from_timewindow)(
            tr, t_start, t_stop;
            axis_scaling = axis_scaling,
            pressure_kwargs = pressure_kwargs,
            force_kwargs = force_kwargs,
            param_map_override = param_map_override,
            param_map_config = param_map_config
        )
    end

    # Fallback path when analysis helper is not bound in Main (common in some Pluto sessions).
    return _pf_interval_from_timewindow_fallback(
        tr, t_start, t_stop;
        axis_scaling = axis_scaling,
        pressure_kwargs = pressure_kwargs,
        force_kwargs = force_kwargs
    )
end

function _overtone_interval_points(tr;
    trim_start_ms::Real = 100,
    trim_end_ms::Real = 100,
    axis_scaling::Symbol = :physical
)
    tr.task == :Overtone || error("_overtone_interval_points expects a multiphonic (:Overtone) trial.")

    regions = onoff_regions(tr; mode = 2)
    length(regions) >= 1 || error("No multiphonic interval found for trial.")

    t_start, t_stop, _ = regions[1]
    interval = _pf_interval_from_timewindow_safe(
        tr,
        t_start + trim_start_ms / 1000,
        t_stop - trim_end_ms / 1000;
        axis_scaling = axis_scaling,
        pressure_kwargs = NamedTuple(),
        force_kwargs = NamedTuple()
    )

    interval === nothing && error("Empty multiphonic interval after trimming.")
    isempty(interval.pressure) && error("Empty multiphonic interval after trimming.")

    return hcat(Float64.(interval.pressure), Float64.(interval.force))
end

# Split a legato trial's single stable mode-2 region into two half-windows.
function stable_windows_legato(tr;
    inner_pad::Float64 = 0.05,
    split_frac::Float64 = 0.5,
    min_dur::Float64 = 0.05
)
    segs2 = onoff_regions(tr; mode = 2)
    isempty(segs2) && error("stable_windows_legato: no mode-2 region for subject=$(tr.subject_id), block=$(tr.block).")
    seg2 = argmax(s -> s[2] - s[1], segs2)

    t_start = seg2[1] + inner_pad
    t_end   = seg2[2] - inner_pad

    (t_end - t_start) >= 2.0 * min_dur ||
        error("stable_windows_legato: region too short after inner_pad for subject=$(tr.subject_id).")

    segs1 = onoff_regions(tr; mode = 1)
    m1_clips = [(max(s[1], t_start), min(s[2], t_end))
                for s in segs1 if s[2] > t_start && s[1] < t_end]

    t_split = if isempty(m1_clips)
        t_start + split_frac * (t_end - t_start)
    else
        t_m1_end   = maximum(c[2] for c in m1_clips)
        t_m1_start = minimum(c[1] for c in m1_clips)
        frac_start = (t_m1_start - t_start) / (t_end - t_start)

        if frac_start < 0.5
            t_m1_end
        else
            t_m1_start
        end
    end

    t_split = clamp(t_split, t_start + min_dur, t_end - min_dur)

    return (t_start, t_split), (t_split, t_end)
end

function legato_segment_labels(task::Symbol)
    if task == :LegatoAsc
        return (:low, :high)
    elseif task == :LegatoDesc
        return (:high, :low)
    else
        error("legato_segment_labels: expected :LegatoAsc or :LegatoDesc, got :$task")
    end
end

function _legato_note_points_from_trial(tr;
    space::Symbol = :physical,
    pressure_calib = pressure_from_adc,
    force_calib = force_newton_from_adc,
    inner_pad::Float64 = 0.05,
    split_frac::Float64 = 0.5
)
    traj = trial_trajectory(
        tr;
        space = space,
        pressure_calib = pressure_calib,
        force_calib = force_calib
    )

    n = minimum((length(tr.t), length(tr.a1), length(tr.a2)))
    n >= 2 || error("_legato_note_points_from_trial: insufficient modal timeline samples.")

    t_mode = _time_axis_seconds(tr.t[1:n])
    regions1 = onoff_regions(tr; mode = 1)

    segs2 = onoff_regions(tr; mode = 2)
    isempty(segs2) && error("_legato_note_points_from_trial: no mode-2 region.")
    seg2 = argmax(s -> s[2] - s[1], segs2)
    t2_start = seg2[1] + inner_pad
    t2_end   = seg2[2] - inner_pad
    t2_start < t2_end || error("_legato_note_points_from_trial: invalid padded mode-2 window.")

    mode1_active = falses(length(t_mode))
    for (ts, te, _) in regions1
        i1 = findfirst(>=(ts), t_mode)
        i2 = findlast(<=(te), t_mode)
        if !isnothing(i1) && !isnothing(i2) && i1 <= i2
            mode1_active[i1:i2] .= true
        end
    end

    mode2_active = falses(length(t_mode))
    i1 = findfirst(>=(t2_start), t_mode)
    i2 = findlast(<=(t2_end), t_mode)
    if !isnothing(i1) && !isnothing(i2) && i1 <= i2
        mode2_active[i1:i2] .= true
    end

    m1_traj = falses(length(traj.t))
    m2_traj = falses(length(traj.t))
    for k in eachindex(traj.t)
        im = argmin(abs.(t_mode .- traj.t[k]))
        m1_traj[k] = mode1_active[im]
        m2_traj[k] = mode2_active[im]
    end

    low_idx = findall(m1_traj .& m2_traj)
    high_idx = findall((.!m1_traj) .& m2_traj)

    if length(low_idx) <= 3 || length(high_idx) <= 3
        w1, w2 = stable_windows_legato(tr; inner_pad = inner_pad, split_frac = split_frac)
        pts1 = points_in_window(traj.t, traj.x, traj.y, w1)
        pts2 = points_in_window(traj.t, traj.x, traj.y, w2)
        lab1, _ = legato_segment_labels(tr.task)
        low  = lab1 == :low  ? pts1 : pts2
        high = lab1 == :high ? pts1 : pts2
        return (low = low, high = high)
    end

    low = hcat(traj.x[low_idx], traj.y[low_idx])
    high = hcat(traj.x[high_idx], traj.y[high_idx])
    return (low = low, high = high)
end

#endregion

#region KDE OVERLAP METRICS

function silverman_bandwidth(v::AbstractVector{<:Real})
    n = length(v)
    s = std(v)
    if !isfinite(s) || s <= 0
        return NaN
    end
    return 1.06 * s * n^(-1 / 5)
end

function kde2d_grid(points::AbstractMatrix{<:Real};
    nx::Int = 80,
    ny::Int = 80,
    xgrid = nothing,
    ygrid = nothing,
    hx = nothing,
    hy = nothing,
    pad_frac::Float64 = 0.10
)
    @assert size(points, 2) == 2

    x = points[:, 1]
    y = points[:, 2]
    n = length(x)

    hx === nothing && (hx = silverman_bandwidth(x))
    hy === nothing && (hy = silverman_bandwidth(y))

    if xgrid === nothing
        xmin, xmax = minimum(x), maximum(x)
        xr = xmax - xmin
        xr = xr > 0 ? xr : 1.0
        xgrid = collect(range(xmin - pad_frac * xr, xmax + pad_frac * xr, length = nx))
    end
    if ygrid === nothing
        ymin, ymax = minimum(y), maximum(y)
        yr = ymax - ymin
        yr = yr > 0 ? yr : 1.0
        ygrid = collect(range(ymin - pad_frac * yr, ymax + pad_frac * yr, length = ny))
    end

    dens = zeros(Float64, length(xgrid), length(ygrid))
    normconst = 1 / (2 * pi * hx * hy * n)

    for i in eachindex(xgrid)
        gx = xgrid[i]
        for j in eachindex(ygrid)
            gy = ygrid[j]
            s = 0.0
            for k in 1:n
                dx = (gx - x[k]) / hx
                dy = (gy - y[k]) / hy
                s += exp(-0.5 * (dx * dx + dy * dy))
            end
            dens[i, j] = normconst * s
        end
    end

    dxg = xgrid[2] - xgrid[1]
    dyg = ygrid[2] - ygrid[1]
    Z = sum(dens) * dxg * dyg
    if !isfinite(Z) || Z <= 0
        error("KDE normalization failed: Z=$Z, hx=$hx, hy=$hy")
    end
    dens ./= Z

    return xgrid, ygrid, dens, hx, hy
end

function kde_overlap_metrics(points1::AbstractMatrix{<:Real},
                             points2::AbstractMatrix{<:Real};
    nx::Int = 80,
    ny::Int = 80,
    hx = nothing,
    hy = nothing
)
    @assert size(points1, 2) == 2
    @assert size(points2, 2) == 2

    allpts = vcat(points1, points2)

    x = allpts[:, 1]
    y = allpts[:, 2]

    hx === nothing && (hx = silverman_bandwidth(x))
    hy === nothing && (hy = silverman_bandwidth(y))

    xgrid, ygrid, _, _, _ = kde2d_grid(allpts; nx = nx, ny = ny, hx = hx, hy = hy)
    _, _, d1, _, _ = kde2d_grid(points1; xgrid = xgrid, ygrid = ygrid, hx = hx, hy = hy)
    _, _, d2, _, _ = kde2d_grid(points2; xgrid = xgrid, ygrid = ygrid, hx = hx, hy = hy)

    dxg = xgrid[2] - xgrid[1]
    dyg = ygrid[2] - ygrid[1]

    if !isfinite(hx) || hx <= 0
        hx = 2 * dxg
    else
        hx = max(hx, 2 * dxg)
    end

    if !isfinite(hy) || hy <= 0
        hy = 2 * dyg
    else
        hy = max(hy, 2 * dyg)
    end

    overlap = sum(min.(d1, d2)) * dxg * dyg

    bc = sum(sqrt.(d1 .* d2)) * dxg * dyg
    hellinger = sqrt(max(0.0, 1.0 - bc))

    c1 = vec(mean(points1; dims = 1))
    c2 = vec(mean(points2; dims = 1))
    centroid_dist = sqrt(sum((c1 .- c2) .^ 2))

    return (
        overlap = overlap,
        hellinger = hellinger,
        centroid_dist = centroid_dist,
        xgrid = xgrid,
        ygrid = ygrid,
        d1 = d1,
        d2 = d2,
        hx = hx,
        hy = hy,
        centroid1 = c1,
        centroid2 = c2
    )

end

function occupancy_map(points::AbstractMatrix{<:Real};
    nx::Int = 80,
    ny::Int = 80,
    xedges = nothing,
    yedges = nothing,
    pad_frac::Real = 0.05
)
    @assert size(points, 2) == 2

    x = points[:, 1]
    y = points[:, 2]

    if xedges === nothing
        xmin, xmax = minimum(x), maximum(x)
        xr = xmax - xmin
        xr = xr > 0 ? xr : 1.0
        xedges = collect(range(xmin - pad_frac * xr, xmax + pad_frac * xr, length = nx + 1))
    end

    if yedges === nothing
        ymin, ymax = minimum(y), maximum(y)
        yr = ymax - ymin
        yr = yr > 0 ? yr : 1.0
        yedges = collect(range(ymin - pad_frac * yr, ymax + pad_frac * yr, length = ny + 1))
    end

    H = zeros(Float64, ny, nx)

    for i in 1:size(points, 1)
        xi = points[i, 1]
        yi = points[i, 2]

        ix = searchsortedlast(xedges, xi)
        iy = searchsortedlast(yedges, yi)

        if 1 <= ix <= nx && 1 <= iy <= ny
            H[iy, ix] += 1
        end
    end

    s = sum(H)
    if s > 0
        H ./= s
    end

    return xedges, yedges, H
end

occupancy_overlap(H1::AbstractMatrix, H2::AbstractMatrix) = sum(min.(H1, H2))

function elliptical_offsets(rx::Int, ry::Int)
    rx >= 0 || error("rx must be non-negative")
    ry >= 0 || error("ry must be non-negative")

    offs = Tuple{Int,Int}[]
    for dy in -ry:ry, dx in -rx:rx
        val = (rx == 0 ? (dx == 0 ? 0.0 : Inf) : (dx / rx)^2) +
              (ry == 0 ? (dy == 0 ? 0.0 : Inf) : (dy / ry)^2)
        if val <= 1.0
            push!(offs, (dx, dy))
        end
    end
    return offs
end

function dilate_mask(mask::BitMatrix; rx::Int = 2, ry::Int = 2)
    ny, nx = size(mask)
    out = falses(ny, nx)
    offs = elliptical_offsets(rx, ry)

    for iy in 1:ny, ix in 1:nx
        mask[iy, ix] || continue
        for (dx, dy) in offs
            jx = ix + dx
            jy = iy + dy
            if 1 <= jx <= nx && 1 <= jy <= ny
                out[jy, jx] = true
            end
        end
    end

    return out
end

"""
Compute all pairwise comparison metrics between two point sets representing
two intervals in control space.

This helper is interval-source agnostic: callers can provide points from
nonlegato windows, legato mode masks, same-note pairings, or multiphonic windows.
"""
function compare_interval_metrics(points1::AbstractMatrix{<:Real},
                                  points2::AbstractMatrix{<:Real};
    nx::Int = 80,
    ny::Int = 80,
    pad_frac::Real = 0.05
)
    size(points1, 2) == 2 || error("compare_interval_metrics: points1 must be N x 2")
    size(points2, 2) == 2 || error("compare_interval_metrics: points2 must be N x 2")
    size(points1, 1) > 3 || error("compare_interval_metrics: points1 needs more than 3 samples")
    size(points2, 1) > 3 || error("compare_interval_metrics: points2 needs more than 3 samples")

    kde = kde_overlap_metrics(points1, points2; nx = nx, ny = ny)

    allpts = vcat(points1, points2)
    xedges, yedges, _ = occupancy_map(allpts; nx = nx, ny = ny, pad_frac = pad_frac)
    _, _, H1 = occupancy_map(points1; xedges = xedges, yedges = yedges)
    _, _, H2 = occupancy_map(points2; xedges = xedges, yedges = yedges)

    return (
        overlap = kde.overlap,
        hellinger = kde.hellinger,
        centroid_dist = kde.centroid_dist,
        occupancy_overlap = occupancy_overlap(H1, H2),
        bandwidth_x = kde.hx,
        bandwidth_y = kde.hy,
    )
end

"""
Compute interval-comparison metrics for a vector of prepared pair records.

Each `pairs` item must contain two matrix fields (default keys: `:left_points`,
`:right_points`) and any number of metadata fields. The output keeps all metadata
and appends:
- `n_left`, `n_right`
- `kde_overlap`, `hellinger`, `centroid_dist`, `occupancy_overlap`
- `bandwidth_x`, `bandwidth_y`

Pairs with fewer than 4 samples in either interval are skipped.
"""
function compute_interval_pair_rows(pairs::AbstractVector;
    left_points_key::Symbol = :left_points,
    right_points_key::Symbol = :right_points,
    nx::Int = 80,
    ny::Int = 80,
    pad_frac::Real = 0.05
)
    out = NamedTuple[]

    for p in pairs
        names = propertynames(p)
        (left_points_key in names && right_points_key in names) || continue

        left_points = getproperty(p, left_points_key)
        right_points = getproperty(p, right_points_key)

        size(left_points, 1) > 3 || continue
        size(right_points, 1) > 3 || continue

        m = compare_interval_metrics(left_points, right_points; nx = nx, ny = ny, pad_frac = pad_frac)

        meta_pairs = Pair{Symbol,Any}[]
        for nm in names
            (nm == left_points_key || nm == right_points_key) && continue
            push!(meta_pairs, nm => getproperty(p, nm))
        end
        meta = (; meta_pairs...)

        push!(out, merge(meta, (
            n_left = size(left_points, 1),
            n_right = size(right_points, 1),
            kde_overlap = m.overlap,
            hellinger = m.hellinger,
            centroid_dist = m.centroid_dist,
            occupancy_overlap = m.occupancy_overlap,
            bandwidth_x = m.bandwidth_x,
            bandwidth_y = m.bandwidth_y,
        )))
    end

    return out
end

#region WASSERSTEIN DISTANCE METRICS

Base.@kwdef struct WassersteinGridSpec
    nx::Int = 40
    ny::Int = 40
    pad_frac::Float64 = 0.08
end

if !isdefined(@__MODULE__, :KDEWassersteinResult)
    Base.@kwdef struct KDEWassersteinResult
        labels::Vector{Any}
        raw_point_sets::Vector{Matrix{Float64}}
        normalized_point_sets::Vector{Matrix{Float64}}
        buffered_point_sets::Union{Nothing,Vector{Matrix{Float64}}}
        support_point_sets::Vector{Matrix{Float64}}
        xedges::Vector{Float64}
        yedges::Vector{Float64}
        xgrid::Vector{Float64}
        ygrid::Vector{Float64}
        kde_grids::Vector{Matrix{Float64}}
        wasserstein_distance_matrix::Matrix{Float64}
        scales::NTuple{2,Float64}
        bandwidth::Float64
        buffer_size::Union{Nothing,Float64}
        grid_spec::WassersteinGridSpec
        sinkhorn_reg::Float64
        sinkhorn_maxiter::Int
        sinkhorn_tol::Float64
    end
end

function _parse_wasserstein_scales(scales)
    values = collect(scales)
    length(values) == 2 || error("scales must contain exactly two values")

    sx = Float64(values[1])
    sy = Float64(values[2])
    (isfinite(sx) && sx > 0) || error("x scale must be positive")
    (isfinite(sy) && sy > 0) || error("y scale must be positive")

    return (sx, sy)
end

function _coerce_wasserstein_grid_spec(grid_spec)
    grid_spec isa WassersteinGridSpec && return grid_spec
    grid_spec isa NamedTuple || error("grid_spec must be a WassersteinGridSpec or NamedTuple")
    hasproperty(grid_spec, :nx) || error("grid_spec needs an :nx field")
    hasproperty(grid_spec, :ny) || error("grid_spec needs an :ny field")
    hasproperty(grid_spec, :pad_frac) || error("grid_spec needs a :pad_frac field")

    return WassersteinGridSpec(
        nx = Int(grid_spec.nx),
        ny = Int(grid_spec.ny),
        pad_frac = Float64(grid_spec.pad_frac)
    )
end

function _normalize_point_set_matrix(points::AbstractMatrix{<:Real}, scales::NTuple{2,Float64})
    size(points, 2) == 2 || error("All point sets must be two-dimensional")
    size(points, 1) > 0 || error("Point sets must contain at least one point")

    sx, sy = scales
    return hcat(Float64.(points[:, 1]) ./ sx, Float64.(points[:, 2]) ./ sy)
end

function normalize_point_set(points::AbstractMatrix{<:Real}; scales)
    return _normalize_point_set_matrix(points, _parse_wasserstein_scales(scales))
end

function normalize_point_sets(point_sets::AbstractVector; scales)
    parsed_scales = _parse_wasserstein_scales(scales)
    return [_normalize_point_set_matrix(points, parsed_scales) for points in point_sets]
end

function _grid_from_normalized_point_sets(point_sets::AbstractVector{<:AbstractMatrix};
    grid_spec::WassersteinGridSpec,
    buffer_size::Union{Nothing,Float64} = nothing
)
    all_points = reduce(vcat, point_sets)
    x = all_points[:, 1]
    y = all_points[:, 2]

    xspan = maximum(x) - minimum(x)
    yspan = maximum(y) - minimum(y)
    xpad = max(grid_spec.pad_frac * max(xspan, eps(Float64)), buffer_size === nothing ? 0.0 : buffer_size)
    ypad = max(grid_spec.pad_frac * max(yspan, eps(Float64)), buffer_size === nothing ? 0.0 : buffer_size)

    grid_spec.nx >= 2 || error("grid_spec.nx must be at least 2")
    grid_spec.ny >= 2 || error("grid_spec.ny must be at least 2")

    xedges = collect(range(minimum(x) - xpad, maximum(x) + xpad, length = grid_spec.nx + 1))
    yedges = collect(range(minimum(y) - ypad, maximum(y) + ypad, length = grid_spec.ny + 1))

    xgrid = 0.5 .* (xedges[1:end-1] .+ xedges[2:end])
    ygrid = 0.5 .* (yedges[1:end-1] .+ yedges[2:end])

    return xedges, yedges, xgrid, ygrid
end

function _points_from_mask(mask::AbstractMatrix{Bool}, xgrid::AbstractVector{<:Real}, ygrid::AbstractVector{<:Real})
    n = count(mask)
    pts = Matrix{Float64}(undef, n, 2)
    idx = 1

    for j in eachindex(ygrid)
        for i in eachindex(xgrid)
            mask[j, i] || continue
            pts[idx, 1] = Float64(xgrid[i])
            pts[idx, 2] = Float64(ygrid[j])
            idx += 1
        end
    end

    return pts
end

function buffer_point_set(points::AbstractMatrix{<:Real};
    buffer_size::Real,
    xedges::Union{Nothing,AbstractVector{<:Real}} = nothing,
    yedges::Union{Nothing,AbstractVector{<:Real}} = nothing,
    grid_spec::Union{WassersteinGridSpec,NamedTuple} = WassersteinGridSpec()
)
    buffer_size >= 0 || error("buffer_size must be non-negative")
    normalized = _normalize_point_set_matrix(points, (1.0, 1.0))
    spec = _coerce_wasserstein_grid_spec(grid_spec)

    if xedges === nothing || yedges === nothing
        xedges, yedges, _, _ = _grid_from_normalized_point_sets([normalized]; grid_spec = spec, buffer_size = Float64(buffer_size))
    end

    _, _, H = occupancy_map(normalized; xedges = xedges, yedges = yedges)
    mask = H .> 0
    dx = xedges[2] - xedges[1]
    dy = yedges[2] - yedges[1]
    rx = max(0, ceil(Int, Float64(buffer_size) / dx))
    ry = max(0, ceil(Int, Float64(buffer_size) / dy))
    buffered_mask = dilate_mask(BitMatrix(mask); rx = rx, ry = ry)

    xgrid = 0.5 .* (xedges[1:end-1] .+ xedges[2:end])
    ygrid = 0.5 .* (yedges[1:end-1] .+ yedges[2:end])
    buffered_points = _points_from_mask(buffered_mask, xgrid, ygrid)

    return (
        buffered_points = buffered_points,
        buffered_mask = buffered_mask,
        xedges = collect(xedges),
        yedges = collect(yedges),
    )
end

function _pairwise_cost_matrix(coords::AbstractMatrix{<:Real})
    n = size(coords, 1)
    cost = Matrix{Float64}(undef, n, n)

    for i in 1:n
        cost[i, i] = 0.0
        for j in (i + 1):n
            dx = Float64(coords[i, 1] - coords[j, 1])
            dy = Float64(coords[i, 2] - coords[j, 2])
            d = sqrt(dx * dx + dy * dy)
            cost[i, j] = d
            cost[j, i] = d
        end
    end

    return cost
end

function _sinkhorn_wasserstein_distance(p::AbstractVector{<:Real}, q::AbstractVector{<:Real}, cost::AbstractMatrix{<:Real};
    reg::Float64,
    maxiter::Int = 1000,
    tol::Float64 = 1e-7
)
    reg > 0 || error("Sinkhorn regularization must be positive")

    pvec = max.(Float64.(p), 0.0)
    qvec = max.(Float64.(q), 0.0)
    sp = sum(pvec)
    sq = sum(qvec)
    sp > 0 || error("First distribution has zero mass")
    sq > 0 || error("Second distribution has zero mass")
    pvec ./= sp
    qvec ./= sq

    C = Float64.(cost)
    K = exp.(-C ./ reg)
    eps_mass = 1e-15
    u = fill(1.0, length(pvec))
    v = fill(1.0, length(qvec))

    for _ in 1:maxiter
        u_prev = copy(u)
        Kv = K * v
        u = pvec ./ max.(Kv, eps_mass)
        Ktu = K' * u
        v = qvec ./ max.(Ktu, eps_mass)

        denom = max.(abs.(u_prev), eps_mass)
        rel_change = maximum(abs.(u .- u_prev) ./ denom)
        rel_change < tol && break
    end

    transport = (u .* K) .* transpose(v)
    return sum(transport .* C)
end

function _sinkhorn_wasserstein_distance_optimaltransport(
    p::AbstractVector{<:Real},
    q::AbstractVector{<:Real},
    cost::AbstractMatrix{<:Real};
    reg::Float64,
    maxiter::Int = 1000,
    tol::Float64 = 1e-7,
)
    HAS_OPTIMALTRANSPORT || error("OptimalTransport.jl is not available. Install it with: import Pkg; Pkg.add(\"OptimalTransport\")")
    reg > 0 || error("Sinkhorn regularization must be positive")

    pvec = max.(Float64.(p), 0.0)
    qvec = max.(Float64.(q), 0.0)
    sp = sum(pvec)
    sq = sum(qvec)
    sp > 0 || error("First distribution has zero mass")
    sq > 0 || error("Second distribution has zero mass")
    pvec ./= sp
    qvec ./= sq

    C = Float64.(cost)
    return Float64(OptimalTransport.sinkhorn2(pvec, qvec, C, reg; maxiter = maxiter, rtol = tol))
end

function select_trials_by_condition(trials::AbstractVector;
    task::Union{Nothing,Symbol} = nothing,
    typ::Union{Nothing,Symbol} = nothing,
    subject_id::Union{Nothing,String} = nothing,
    only_success::Bool = true,
    exclude_subject_id::Union{Nothing,String} = "13"
)
    selected = eltype(trials)[]

    for tr in trials
        only_success && !tr.success && continue
        task !== nothing && tr.task != task && continue
        typ !== nothing && tr.type != typ && continue
        subject_id !== nothing && tr.subject_id != subject_id && continue
        exclude_subject_id !== nothing && tr.subject_id == exclude_subject_id && continue
        push!(selected, tr)
    end

    return selected
end

function nonlegato_point_sets_from_trial(tr;
    trim_start_ms::Real = 0,
    trim_end_ms::Real = 0,
    space::Symbol = :physical,
    pressure_calib = pressure_from_adc,
    force_calib = force_newton_from_adc,
    inner_pad::Float64 = 0.05,
    min_dur::Float64 = 0.05,
    edge_exclusion_frac::Float64 = 0.15
)
    tr.task in (:NonlegatoAsc, :NonlegatoDesc) || error("nonlegato_point_sets_from_trial only supports non-legato trials")

    traj = trial_trajectory(
        tr;
        space = space,
        pressure_calib = pressure_calib,
        force_calib = force_calib,
    )

    w1, w2 = stable_windows_nonlegato(
        tr;
        inner_pad = inner_pad,
        min_dur = min_dur,
        edge_exclusion_frac = edge_exclusion_frac
    )

    trim_start_s = Float64(trim_start_ms) / 1000
    trim_end_s = Float64(trim_end_ms) / 1000

    w1_trim = (w1[1] + trim_start_s, w1[2] - trim_end_s)
    w2_trim = (w2[1] + trim_start_s, w2[2] - trim_end_s)
    (w1_trim[1] < w1_trim[2] && w2_trim[1] < w2_trim[2]) || error("Trimming removed one of the non-legato windows")

    pts1 = points_in_window(traj.t, traj.x, traj.y, w1_trim)
    pts2 = points_in_window(traj.t, traj.x, traj.y, w2_trim)
    low_label, _ = segment_labels(tr.task)

    if low_label == :low
        return (low = pts1, high = pts2, windows = (low = w1_trim, high = w2_trim))
    else
        return (low = pts2, high = pts1, windows = (low = w2_trim, high = w1_trim))
    end
end

function nonlegato_point_sets_from_trials(trials::AbstractVector;
    task::Union{Nothing,Symbol} = nothing,
    typ::Union{Nothing,Symbol} = nothing,
    subject_id::Union{Nothing,String} = nothing,
    only_success::Bool = true,
    exclude_subject_id::Union{Nothing,String} = "13",
    trim_start_ms::Real = 0,
    trim_end_ms::Real = 0,
    space::Symbol = :physical,
    pressure_calib = pressure_from_adc,
    force_calib = force_newton_from_adc,
    inner_pad::Float64 = 0.05,
    min_dur::Float64 = 0.05,
    edge_exclusion_frac::Float64 = 0.15,
    pool::Bool = true
)
    selected = select_trials_by_condition(
        trials;
        task = task,
        typ = typ,
        subject_id = subject_id,
        only_success = only_success,
        exclude_subject_id = exclude_subject_id
    )

    usable_trials = eltype(trials)[]
    low_sets = Matrix{Float64}[]
    high_sets = Matrix{Float64}[]
    per_trial = NamedTuple[]

    for tr in selected
        try
            pts = nonlegato_point_sets_from_trial(
                tr;
                trim_start_ms = trim_start_ms,
                trim_end_ms = trim_end_ms,
                space = space,
                pressure_calib = pressure_calib,
                force_calib = force_calib,
                inner_pad = inner_pad,
                min_dur = min_dur,
                edge_exclusion_frac = edge_exclusion_frac
            )

            push!(usable_trials, tr)
            push!(low_sets, pts.low)
            push!(high_sets, pts.high)
            push!(per_trial, (trial = tr, low = pts.low, high = pts.high, windows = pts.windows))
        catch
            # Skip trials that do not yield valid non-legato windows after trimming.
        end
    end

    pooled = pool ? (
        low = isempty(low_sets) ? zeros(0, 2) : reduce(vcat, low_sets),
        high = isempty(high_sets) ? zeros(0, 2) : reduce(vcat, high_sets)
    ) : nothing

    return (
        selected_trials = usable_trials,
        low_sets = low_sets,
        high_sets = high_sets,
        per_trial = per_trial,
        pooled = pooled,
    )
end

function _finite_point_rows(points::AbstractMatrix)
    size(points, 2) == 2 || error("Point sets must be two-dimensional")
    keep = vec(all(isfinite, points; dims = 2))
    return Matrix{Float64}(points[keep, :])
end

"""
    collect_wasserstein_reference_point_sets(trials; ...)

Collect the retained pressure-force point sets used to estimate the pooled
global SD scaling for Wasserstein analyses. The collection mirrors the active
comparison families: non-legato low/high, legato low/high, and multiphonic regions.
"""
function collect_wasserstein_reference_point_sets(trials::AbstractVector;
    cond_specs = (
        (task = :NonlegatoAsc, typ = :Model, label = "Asc Model"),
        (task = :NonlegatoDesc, typ = :Model, label = "Desc Model"),
        (task = :NonlegatoAsc, typ = :Real, label = "Asc Instrument"),
        (task = :NonlegatoDesc, typ = :Real, label = "Desc Instrument"),
    ),
    legato_specs = (
        (task = :LegatoAsc, typ = :Model, label = "Asc Model"),
        (task = :LegatoDesc, typ = :Model, label = "Desc Model"),
        (task = :LegatoAsc, typ = :Real, label = "Asc Instrument"),
        (task = :LegatoDesc, typ = :Real, label = "Desc Instrument"),
    ),
    exclude_subject_id::Union{Nothing,String} = "13",
    trim_start_ms::Real = 100,
    trim_end_ms::Real = 100,
    space::Symbol = :physical,
    pressure_calib = pressure_from_adc,
    force_calib = force_newton_from_adc,
    inner_pad::Float64 = 0.05,
    edge_exclusion_frac::Float64 = 0.15,
)
    point_sets = Matrix{Float64}[]

    function _push_if_nonempty(points)
        finite = _finite_point_rows(points)
        size(finite, 1) > 0 && push!(point_sets, finite)
        return nothing
    end

    for c in cond_specs
        selected = select_trials_by_condition(
            trials;
            task = c.task,
            typ = c.typ,
            only_success = true,
            exclude_subject_id = exclude_subject_id,
        )

        for tr in selected
            pts = try
                nonlegato_point_sets_from_trial(
                    tr;
                    trim_start_ms = trim_start_ms,
                    trim_end_ms = trim_end_ms,
                    space = space,
                    pressure_calib = pressure_calib,
                    force_calib = force_calib,
                    inner_pad = inner_pad,
                    edge_exclusion_frac = edge_exclusion_frac,
                )
            catch
                nothing
            end

            pts === nothing && continue
            _push_if_nonempty(pts.low)
            _push_if_nonempty(pts.high)
        end
    end

    for c in legato_specs
        selected = select_trials_by_condition(
            trials;
            task = c.task,
            typ = c.typ,
            only_success = true,
            exclude_subject_id = exclude_subject_id,
        )

        for tr in selected
            pts = try
                _legato_note_points_from_trial(
                    tr;
                    space = space,
                    pressure_calib = pressure_calib,
                    force_calib = force_calib,
                    inner_pad = inner_pad,
                )
            catch
                nothing
            end

            pts === nothing && continue
            _push_if_nonempty(pts.low)
            _push_if_nonempty(pts.high)
        end
    end

    overtone_types = unique([c.typ for c in cond_specs])
    for typ in overtone_types
        selected = select_trials_by_condition(
            trials;
            task = :Overtone,
            typ = typ,
            only_success = true,
            exclude_subject_id = exclude_subject_id,
        )

        for tr in selected
            pts = try
                _overtone_interval_points(
                    tr;
                    trim_start_ms = trim_start_ms,
                    trim_end_ms = trim_end_ms,
                    axis_scaling = space,
                )
            catch
                nothing
            end

            pts === nothing && continue
            _push_if_nonempty(pts)
        end
    end

    return point_sets
end

"""
    compute_global_wasserstein_scales(trials; corrected = true, ...)

Estimate the pooled global pressure/force SDs used as Wasserstein axis scales.
The returned `scales` tuple is `(pressure_sd, force_sd)`. Means are reported for
auditability but are not subtracted; this keeps physical magnitudes positive
while defining the ground metric in global-SD units.
"""
function compute_global_wasserstein_scales(trials::AbstractVector;
    corrected::Bool = true,
    kwargs...
)
    point_sets = collect_wasserstein_reference_point_sets(trials; kwargs...)
    isempty(point_sets) && error("No retained pressure-force samples available for Wasserstein scaling")

    all_points = reduce(vcat, point_sets)
    size(all_points, 1) > 1 || error("At least two retained points are required for Wasserstein scaling")

    pressure = all_points[:, 1]
    force = all_points[:, 2]
    pressure_sd = std(pressure; corrected = corrected)
    force_sd = std(force; corrected = corrected)

    (isfinite(pressure_sd) && pressure_sd > 0) || error("Global pressure SD must be finite and positive")
    (isfinite(force_sd) && force_sd > 0) || error("Global force SD must be finite and positive")

    return (
        normalization = :global_sd_no_centering,
        scales = (Float64(pressure_sd), Float64(force_sd)),
        pressure_mean = mean(pressure),
        force_mean = mean(force),
        pressure_sd = Float64(pressure_sd),
        force_sd = Float64(force_sd),
        n_points = size(all_points, 1),
        n_point_sets = length(point_sets),
        corrected = corrected,
    )
end

function compute_kde_wasserstein_distances(point_sets::AbstractVector;
    scales,
    bandwidth::Real,
    buffer_size::Union{Nothing,Real} = nothing,
    grid_spec = WassersteinGridSpec(),
    sinkhorn_reg::Union{Nothing,Real} = nothing,
    sinkhorn_maxiter::Int = 1000,
    sinkhorn_tol::Float64 = 1e-7,
    labels = nothing
)
    length(point_sets) >= 2 || error("Need at least two point sets")

    parsed_scales = _parse_wasserstein_scales(scales)
    bw = Float64(bandwidth)
    (isfinite(bw) && bw > 0) || error("bandwidth must be positive")

    buf = if buffer_size === nothing
        nothing
    else
        bs = Float64(buffer_size)
        bs >= 0 || error("buffer_size must be non-negative")
        bs
    end

    spec = _coerce_wasserstein_grid_spec(grid_spec)
    raw_sets = [Matrix{Float64}(points) for points in point_sets]
    normalized_sets = [_normalize_point_set_matrix(points, parsed_scales) for points in raw_sets]

    xedges, yedges, xgrid, ygrid = _grid_from_normalized_point_sets(
        normalized_sets;
        grid_spec = spec,
        buffer_size = buf
    )

    support_sets = Matrix{Float64}[]
    buffered_sets = buf === nothing ? nothing : Matrix{Float64}[]

    for points in normalized_sets
        if buf === nothing
            push!(support_sets, points)
        else
            buffered = buffer_point_set(
                points;
                buffer_size = buf,
                xedges = xedges,
                yedges = yedges,
                grid_spec = spec
            )
            push!(support_sets, buffered.buffered_points)
            push!(buffered_sets, buffered.buffered_points)
        end
    end

    kde_grids = Matrix{Float64}[]
    dx = length(xgrid) > 1 ? xgrid[2] - xgrid[1] : 1.0
    dy = length(ygrid) > 1 ? ygrid[2] - ygrid[1] : 1.0
    cell_area = dx * dy

    for points in support_sets
        _, _, dens, _, _ = kde2d_grid(points; xgrid = xgrid, ygrid = ygrid, hx = bw, hy = bw)
        push!(kde_grids, dens)
    end

    coords = Matrix{Float64}(undef, length(xgrid) * length(ygrid), 2)
    idx = 1
    for j in eachindex(ygrid)
        for i in eachindex(xgrid)
            coords[idx, 1] = xgrid[i]
            coords[idx, 2] = ygrid[j]
            idx += 1
        end
    end
    cost = _pairwise_cost_matrix(coords)

    transport_reg = Float64(sinkhorn_reg === nothing ? max(1e-3, 0.5 * bw) : sinkhorn_reg)
    transport_reg > 0 || error("sinkhorn_reg must be positive")

    masses = [max.(vec(dens) .* cell_area, 0.0) for dens in kde_grids]
    for mass in masses
        s = sum(mass)
        s > 0 || error("One KDE grid has zero mass")
        mass ./= s
    end

    nsets = length(masses)
    wasserstein = zeros(Float64, nsets, nsets)
    for i in 1:nsets
        for j in (i + 1):nsets
            d = _sinkhorn_wasserstein_distance(
                masses[i],
                masses[j],
                cost;
                reg = transport_reg,
                maxiter = sinkhorn_maxiter,
                tol = sinkhorn_tol
            )
            wasserstein[i, j] = d
            wasserstein[j, i] = d
        end
    end

    if labels === nothing
        labels = collect(1:nsets)
    end

    return KDEWassersteinResult(
        labels = collect(labels),
        raw_point_sets = raw_sets,
        normalized_point_sets = normalized_sets,
        buffered_point_sets = buffered_sets,
        support_point_sets = support_sets,
        xedges = xedges,
        yedges = yedges,
        xgrid = xgrid,
        ygrid = ygrid,
        kde_grids = kde_grids,
        wasserstein_distance_matrix = wasserstein,
        scales = parsed_scales,
        bandwidth = bw,
        buffer_size = buf,
        grid_spec = spec,
        sinkhorn_reg = transport_reg,
        sinkhorn_maxiter = sinkhorn_maxiter,
        sinkhorn_tol = sinkhorn_tol,
    )
end

function compute_kde_wasserstein_distances_optimaltransport(point_sets::AbstractVector;
    scales,
    bandwidth::Real,
    buffer_size::Union{Nothing,Real} = nothing,
    grid_spec = WassersteinGridSpec(),
    sinkhorn_reg::Union{Nothing,Real} = nothing,
    sinkhorn_maxiter::Int = 1000,
    sinkhorn_tol::Float64 = 1e-7,
    labels = nothing
)
    length(point_sets) >= 2 || error("Need at least two point sets")

    parsed_scales = _parse_wasserstein_scales(scales)
    bw = Float64(bandwidth)
    (isfinite(bw) && bw > 0) || error("bandwidth must be positive")

    buf = if buffer_size === nothing
        nothing
    else
        bs = Float64(buffer_size)
        bs >= 0 || error("buffer_size must be non-negative")
        bs
    end

    spec = _coerce_wasserstein_grid_spec(grid_spec)
    raw_sets = [Matrix{Float64}(points) for points in point_sets]
    normalized_sets = [_normalize_point_set_matrix(points, parsed_scales) for points in raw_sets]

    xedges, yedges, xgrid, ygrid = _grid_from_normalized_point_sets(
        normalized_sets;
        grid_spec = spec,
        buffer_size = buf
    )

    support_sets = Matrix{Float64}[]
    buffered_sets = buf === nothing ? nothing : Matrix{Float64}[]

    for points in normalized_sets
        if buf === nothing
            push!(support_sets, points)
        else
            buffered = buffer_point_set(
                points;
                buffer_size = buf,
                xedges = xedges,
                yedges = yedges,
                grid_spec = spec
            )
            push!(support_sets, buffered.buffered_points)
            push!(buffered_sets, buffered.buffered_points)
        end
    end

    kde_grids = Matrix{Float64}[]
    dx = length(xgrid) > 1 ? xgrid[2] - xgrid[1] : 1.0
    dy = length(ygrid) > 1 ? ygrid[2] - ygrid[1] : 1.0
    cell_area = dx * dy

    for points in support_sets
        _, _, dens, _, _ = kde2d_grid(points; xgrid = xgrid, ygrid = ygrid, hx = bw, hy = bw)
        push!(kde_grids, dens)
    end

    coords = Matrix{Float64}(undef, length(xgrid) * length(ygrid), 2)
    idx = 1
    for j in eachindex(ygrid)
        for i in eachindex(xgrid)
            coords[idx, 1] = xgrid[i]
            coords[idx, 2] = ygrid[j]
            idx += 1
        end
    end
    cost = _pairwise_cost_matrix(coords)

    transport_reg = Float64(sinkhorn_reg === nothing ? max(1e-3, 0.5 * bw) : sinkhorn_reg)
    transport_reg > 0 || error("sinkhorn_reg must be positive")

    masses = [max.(vec(dens) .* cell_area, 0.0) for dens in kde_grids]
    for mass in masses
        s = sum(mass)
        s > 0 || error("One KDE grid has zero mass")
        mass ./= s
    end

    nsets = length(masses)
    wasserstein = zeros(Float64, nsets, nsets)
    for i in 1:nsets
        for j in (i + 1):nsets
            d = _sinkhorn_wasserstein_distance_optimaltransport(
                masses[i],
                masses[j],
                cost;
                reg = transport_reg,
                maxiter = sinkhorn_maxiter,
                tol = sinkhorn_tol
            )
            wasserstein[i, j] = d
            wasserstein[j, i] = d
        end
    end

    if labels === nothing
        labels = collect(1:nsets)
    end

    return KDEWassersteinResult(
        labels = collect(labels),
        raw_point_sets = raw_sets,
        normalized_point_sets = normalized_sets,
        buffered_point_sets = buffered_sets,
        support_point_sets = support_sets,
        xedges = xedges,
        yedges = yedges,
        xgrid = xgrid,
        ygrid = ygrid,
        kde_grids = kde_grids,
        wasserstein_distance_matrix = wasserstein,
        scales = parsed_scales,
        bandwidth = bw,
        buffer_size = buf,
        grid_spec = spec,
        sinkhorn_reg = transport_reg,
        sinkhorn_maxiter = sinkhorn_maxiter,
        sinkhorn_tol = sinkhorn_tol,
    )
end

"""
    permutation_test_wasserstein_between_sets(points_a, points_b;
        label_a = nothing,
        label_b = nothing,
        n_permutations = 1000,
        random_seed = nothing,
        distance_kwargs = nothing,
    )

Run a one-sided permutation test for the observed Wasserstein distance between
two 2D point sets under the null hypothesis that both sets were sampled from
the same underlying distribution.

The core distance computation is delegated directly to
`compute_kde_wasserstein_distances([points_a, points_b]; ...)`, so callers must
provide the same keyword arguments required by that function via
`distance_kwargs`, typically at least `scales` and `bandwidth`.
"""
function permutation_test_wasserstein_between_sets(points_a::AbstractMatrix{<:Real},
                                                   points_b::AbstractMatrix{<:Real};
    label_a = nothing,
    label_b = nothing,
    n_permutations::Int = 1000,
    random_seed::Union{Nothing,Integer} = nothing,
    distance_kwargs = nothing,
)
    size(points_a, 2) == 2 || error("points_a must be an N x 2 matrix")
    size(points_b, 2) == 2 || error("points_b must be an N x 2 matrix")
    size(points_a, 1) > 3 || error("points_a must contain at least 4 samples")
    size(points_b, 1) > 3 || error("points_b must contain at least 4 samples")
    all(isfinite, points_a) || error("points_a contains non-finite values")
    all(isfinite, points_b) || error("points_b contains non-finite values")
    n_permutations > 0 || error("n_permutations must be positive")

    kwargs_nt = if distance_kwargs === nothing
        (;)
    elseif distance_kwargs isa NamedTuple
        distance_kwargs
    elseif distance_kwargs isa AbstractDict
        (; (Symbol(k) => v for (k, v) in pairs(distance_kwargs))...)
    else
        error("distance_kwargs must be nothing, a NamedTuple, or an AbstractDict")
    end

    hasproperty(kwargs_nt, :scales) || error("distance_kwargs must include :scales")
    hasproperty(kwargs_nt, :bandwidth) || error("distance_kwargs must include :bandwidth")
    distance_method = hasproperty(kwargs_nt, :method) ? kwargs_nt.method : :internal
    distance_method in (:internal, :optimaltransport) || error("Unsupported distance method $(repr(distance_method)). Use :internal or :optimaltransport")
    compute_kwargs = (; (k => getproperty(kwargs_nt, k) for k in keys(kwargs_nt) if k != :method)...)

    name_a = isnothing(label_a) ? "A" : label_a
    name_b = isnothing(label_b) ? "B" : label_b
    labels = [name_a, name_b]

    function _pair_result(a, b)
        if distance_method == :optimaltransport
            return compute_kde_wasserstein_distances_optimaltransport(
                [a, b];
                compute_kwargs...,
                labels = labels,
            )
        else
            return compute_kde_wasserstein_distances(
                [a, b];
                compute_kwargs...,
                labels = labels,
            )
        end
    end

    observed = _pair_result(Matrix{Float64}(points_a), Matrix{Float64}(points_b))
    observed_distance = observed.wasserstein_distance_matrix[1, 2]

    combined = vcat(Matrix{Float64}(points_a), Matrix{Float64}(points_b))
    n_a = size(points_a, 1)
    n_b = size(points_b, 1)
    total_n = n_a + n_b
    rng = isnothing(random_seed) ? Random.default_rng() : Random.MersenneTwister(random_seed)

    null_distances = Vector{Float64}(undef, n_permutations)
    for perm_idx in 1:n_permutations
        perm = Random.randperm(rng, total_n)
        perm_a = combined[perm[1:n_a], :]
        perm_b = combined[perm[(n_a + 1):end], :]

        permuted = _pair_result(perm_a, perm_b)
        null_distances[perm_idx] = permuted.wasserstein_distance_matrix[1, 2]
    end

    exceedance_count = count(d -> d >= observed_distance, null_distances)
    p_value = (exceedance_count + 1) / (n_permutations + 1)

    return (
        label_a = name_a,
        label_b = name_b,
        n_a = n_a,
        n_b = n_b,
        observed_distance = observed_distance,
        observed_result = observed,
        null_distances = null_distances,
        null_mean = mean(null_distances),
        null_sd = length(null_distances) > 1 ? std(null_distances) : 0.0,
        exceedance_count = exceedance_count,
        p_value = p_value,
        alternative = :greater,
        n_permutations = n_permutations,
        random_seed = random_seed,
        distance_kwargs = kwargs_nt,
    )
end

"""
    permutation_test_wasserstein_rows_by_family(trials;
        family = :all,
        cond_specs = (...),
        legato_specs = (...),
        exclude_subject_id = "13",
        trim_start_ms = 100,
        trim_end_ms = 100,
        space = :physical,
        pressure_calib = pressure_from_adc,
        force_calib = force_newton_from_adc,
        inner_pad = 0.05,
        edge_exclusion_frac = 0.15,
        n_permutations = 200,
        random_seed = nothing,
        distance_kwargs,
    )

Build permutation-test rows for one family (or all families) used in the Pluto
distance notebook. The output row schema matches the notebook table
(`family`, `comparison`, `task`, `typ`, `note`, `n_a`, `n_b`, `observed`,
`null_mean`, `null_sd`, `p_value`, `status`).

Supported `family` values:
- `:all`
- `:nonlegato_low_high`
- `:same_note_asc_desc`
- `:overtone_vs_nonlegato`
- `:legato_low_high`
"""
function permutation_test_wasserstein_rows_by_family(trials::AbstractVector;
    family::Symbol = :all,
    cond_specs = (
        (task = :NonlegatoAsc, typ = :Model, label = "Asc Model"),
        (task = :NonlegatoDesc, typ = :Model, label = "Desc Model"),
        (task = :NonlegatoAsc, typ = :Real, label = "Asc Instrument"),
        (task = :NonlegatoDesc, typ = :Real, label = "Desc Instrument"),
    ),
    legato_specs = (
        (task = :LegatoAsc, typ = :Model, label = "Asc Model"),
        (task = :LegatoDesc, typ = :Model, label = "Desc Model"),
        (task = :LegatoAsc, typ = :Real, label = "Asc Instrument"),
        (task = :LegatoDesc, typ = :Real, label = "Desc Instrument"),
    ),
    exclude_subject_id::Union{Nothing,String} = "13",
    trim_start_ms::Real = 100,
    trim_end_ms::Real = 100,
    space::Symbol = :physical,
    pressure_calib = pressure_from_adc,
    force_calib = force_newton_from_adc,
    inner_pad::Float64 = 0.05,
    edge_exclusion_frac::Float64 = 0.15,
    n_permutations::Int = 200,
    random_seed::Union{Nothing,Integer} = nothing,
    distance_kwargs = nothing,
)
    allowed_families = Set((
        :all,
        :nonlegato_low_high,
        :same_note_asc_desc,
        :overtone_vs_nonlegato,
        :legato_low_high,
    ))
    family in allowed_families || error("Unsupported family $(repr(family)). Allowed: $(collect(allowed_families))")
    n_permutations > 0 || error("n_permutations must be positive")

    kwargs_nt = if distance_kwargs === nothing
        (;)
    elseif distance_kwargs isa NamedTuple
        distance_kwargs
    elseif distance_kwargs isa AbstractDict
        (; (Symbol(k) => v for (k, v) in pairs(distance_kwargs))...)
    else
        error("distance_kwargs must be nothing, a NamedTuple, or an AbstractDict")
    end
    hasproperty(kwargs_nt, :scales) || error("distance_kwargs must include :scales")
    hasproperty(kwargs_nt, :bandwidth) || error("distance_kwargs must include :bandwidth")
    distance_method = hasproperty(kwargs_nt, :method) ? kwargs_nt.method : :internal
    distance_method in (:internal, :optimaltransport) || error("Unsupported distance method $(repr(distance_method)). Use :internal or :optimaltransport")
    compute_kwargs = (; (k => getproperty(kwargs_nt, k) for k in keys(kwargs_nt) if k != :method)...)

    rows = NamedTuple[]
    seed_counter_ref = Ref(0)

    _run_family(sym::Symbol) = (family == :all || family == sym)

    function _next_seed()
        if random_seed === nothing
            return nothing
        end
        seed_counter_ref[] += 1
        return Int(random_seed) + seed_counter_ref[]
    end

    function _finite_points(pts::AbstractMatrix)
        keep = vec(all(isfinite, pts; dims = 2))
        return pts[keep, :]
    end

    function _push_perm_row(points_a::AbstractMatrix, points_b::AbstractMatrix;
        family,
        comparison,
        task,
        typ,
        note,
        label_a,
        label_b,
    )
        pts_a = _finite_points(points_a)
        pts_b = _finite_points(points_b)

        if size(pts_a, 1) <= 3 || size(pts_b, 1) <= 3
            push!(rows, (
                statistic_level = "global_pooled",
                family = family,
                comparison = comparison,
                task = task,
                typ = typ,
                note = note,
                label_a = label_a,
                label_b = label_b,
                n_a = size(pts_a, 1),
                n_b = size(pts_b, 1),
                observed = NaN,
                null_mean = NaN,
                null_sd = NaN,
                p_value = NaN,
                status = "insufficient samples",
            ))
            return
        end

        try
            out = permutation_test_wasserstein_between_sets(
                pts_a,
                pts_b;
                label_a = label_a,
                label_b = label_b,
                n_permutations = n_permutations,
                random_seed = _next_seed(),
                distance_kwargs = kwargs_nt,
            )

            push!(rows, (
                statistic_level = "global_pooled",
                family = family,
                comparison = comparison,
                task = task,
                typ = typ,
                note = note,
                label_a = String(out.label_a),
                label_b = String(out.label_b),
                n_a = out.n_a,
                n_b = out.n_b,
                observed = out.observed_distance,
                null_mean = out.null_mean,
                null_sd = out.null_sd,
                p_value = out.p_value,
                status = "ok",
            ))
        catch err
            err_msg = sprint(showerror, err)
            err_label = if err isa UndefVarError
                "UndefVarError($(err.var)): $(err_msg)"
            else
                err_msg
            end

            push!(rows, (
                statistic_level = "global_pooled",
                family = family,
                comparison = comparison,
                task = task,
                typ = typ,
                note = note,
                label_a = label_a,
                label_b = label_b,
                n_a = size(pts_a, 1),
                n_b = size(pts_b, 1),
                observed = NaN,
                null_mean = NaN,
                null_sd = NaN,
                p_value = NaN,
                status = "error: $(err_label)",
            ))
        end
    end

    if _run_family(:nonlegato_low_high)
        for c in cond_specs
            grouped = try
                nonlegato_point_sets_from_trials(
                    trials;
                    task = c.task,
                    typ = c.typ,
                    only_success = true,
                    exclude_subject_id = exclude_subject_id,
                    trim_start_ms = trim_start_ms,
                    trim_end_ms = trim_end_ms,
                    space = space,
                    pressure_calib = pressure_calib,
                    force_calib = force_calib,
                    inner_pad = inner_pad,
                    edge_exclusion_frac = edge_exclusion_frac,
                    pool = true,
                )
            catch
                nothing
            end

            if grouped === nothing || grouped.pooled === nothing
                _push_perm_row(zeros(0, 2), zeros(0, 2);
                    family = "1) Nonlegato low vs high",
                    comparison = "Pooled by condition",
                    task = string(c.task),
                    typ = string(c.typ),
                    note = "low-vs-high",
                    label_a = "low",
                    label_b = "high",
                )
            else
                _push_perm_row(grouped.pooled.low, grouped.pooled.high;
                    family = "1) Nonlegato low vs high",
                    comparison = "Pooled by condition",
                    task = string(c.task),
                    typ = string(c.typ),
                    note = "low-vs-high",
                    label_a = "low",
                    label_b = "high",
                )
            end
        end
    end

    if _run_family(:same_note_asc_desc)
        for typ in (:Model, :Real)
            asc_pool = try
                nonlegato_point_sets_from_trials(
                    trials;
                    task = :NonlegatoAsc,
                    typ = typ,
                    only_success = true,
                    exclude_subject_id = exclude_subject_id,
                    trim_start_ms = trim_start_ms,
                    trim_end_ms = trim_end_ms,
                    space = space,
                    pressure_calib = pressure_calib,
                    force_calib = force_calib,
                    inner_pad = inner_pad,
                    edge_exclusion_frac = edge_exclusion_frac,
                    pool = true,
                )
            catch
                nothing
            end
            desc_pool = try
                nonlegato_point_sets_from_trials(
                    trials;
                    task = :NonlegatoDesc,
                    typ = typ,
                    only_success = true,
                    exclude_subject_id = exclude_subject_id,
                    trim_start_ms = trim_start_ms,
                    trim_end_ms = trim_end_ms,
                    space = space,
                    pressure_calib = pressure_calib,
                    force_calib = force_calib,
                    inner_pad = inner_pad,
                    edge_exclusion_frac = edge_exclusion_frac,
                    pool = true,
                )
            catch
                nothing
            end

            for note in (:low, :high)
                pts_asc = (asc_pool === nothing || asc_pool.pooled === nothing) ? zeros(0, 2) : getfield(asc_pool.pooled, note)
                pts_desc = (desc_pool === nothing || desc_pool.pooled === nothing) ? zeros(0, 2) : getfield(desc_pool.pooled, note)

                _push_perm_row(pts_asc, pts_desc;
                    family = "2) Same note Asc vs Desc",
                    comparison = "Pooled by type+note",
                    task = "Asc-vs-Desc",
                    typ = string(typ),
                    note = string(note),
                    label_a = "Asc",
                    label_b = "Desc",
                )
            end
        end
    end

    if _run_family(:overtone_vs_nonlegato)
        overtone_chunks_by_type = Dict{Symbol,Vector{Matrix{Float64}}}()
        overtone_trials = filter(
            tr -> tr.success && tr.subject_id != exclude_subject_id && tr.task == :Overtone && tr.type in (:Model, :Real),
            trials,
        )
        for tr in overtone_trials
            try
                pts = _overtone_interval_points(
                    tr;
                    trim_start_ms = trim_start_ms,
                    trim_end_ms = trim_end_ms,
                    axis_scaling = space,
                )
                push!(get!(overtone_chunks_by_type, tr.type, Matrix{Float64}[]), pts)
            catch
            end
        end

        for c in cond_specs
            overtone_pool = isempty(get(overtone_chunks_by_type, c.typ, Matrix{Float64}[])) ? zeros(0, 2) : reduce(vcat, get(overtone_chunks_by_type, c.typ, Matrix{Float64}[]))

            nl_pool = try
                nonlegato_point_sets_from_trials(
                    trials;
                    task = c.task,
                    typ = c.typ,
                    only_success = true,
                    exclude_subject_id = exclude_subject_id,
                    trim_start_ms = trim_start_ms,
                    trim_end_ms = trim_end_ms,
                    space = space,
                    pressure_calib = pressure_calib,
                    force_calib = force_calib,
                    inner_pad = inner_pad,
                    edge_exclusion_frac = edge_exclusion_frac,
                    pool = true,
                )
            catch
                nothing
            end

            for note in (:low, :high)
                note_pts = (nl_pool === nothing || nl_pool.pooled === nothing) ? zeros(0, 2) : getfield(nl_pool.pooled, note)
                _push_perm_row(overtone_pool, note_pts;
                    family = "3) Multiphonic vs nonlegato note",
                    comparison = "Pooled by condition+note",
                    task = string(c.task),
                    typ = string(c.typ),
                    note = string(note),
                    label_a = "Multiphonic",
                    label_b = string(note),
                )
            end
        end
    end

    if _run_family(:legato_low_high)
        for c in legato_specs
            leg_trials = filter(
                tr -> tr.success && tr.subject_id != exclude_subject_id && tr.task == c.task && tr.type == c.typ,
                trials,
            )

            legato_low_chunks = Matrix{Float64}[]
            legato_high_chunks = Matrix{Float64}[]
            for tr in leg_trials
                try
                    p = _legato_note_points_from_trial(
                        tr;
                        space = space,
                        pressure_calib = pressure_calib,
                        force_calib = force_calib,
                        inner_pad = inner_pad,
                    )
                    push!(legato_low_chunks, p.low)
                    push!(legato_high_chunks, p.high)
                catch
                end
            end

            low_pool = isempty(legato_low_chunks) ? zeros(0, 2) : reduce(vcat, legato_low_chunks)
            high_pool = isempty(legato_high_chunks) ? zeros(0, 2) : reduce(vcat, legato_high_chunks)

            _push_perm_row(low_pool, high_pool;
                family = "4) Legato low vs high",
                comparison = "Pooled by condition",
                task = string(c.task),
                typ = string(c.typ),
                note = "low-vs-high",
                label_a = "low",
                label_b = "high",
            )
        end
    end

    return rows
end

"""
    permutation_test_wasserstein_rows_by_family_subject_stratified(trials;
        family = :all,
        ...
    )

Subject-stratified permutation tests for the same four family definitions used by
the 08 notebook. For each row, the test statistic is the mean of per-subject
Wasserstein distances, and permutations are performed independently within each
subject (preserving each subject's sample counts).
"""
function permutation_test_wasserstein_rows_by_family_subject_stratified(trials::AbstractVector;
    family::Symbol = :all,
    cond_specs = (
        (task = :NonlegatoAsc, typ = :Model, label = "Asc Model"),
        (task = :NonlegatoDesc, typ = :Model, label = "Desc Model"),
        (task = :NonlegatoAsc, typ = :Real, label = "Asc Instrument"),
        (task = :NonlegatoDesc, typ = :Real, label = "Desc Instrument"),
    ),
    legato_specs = (
        (task = :LegatoAsc, typ = :Model, label = "Asc Model"),
        (task = :LegatoDesc, typ = :Model, label = "Desc Model"),
        (task = :LegatoAsc, typ = :Real, label = "Asc Instrument"),
        (task = :LegatoDesc, typ = :Real, label = "Desc Instrument"),
    ),
    exclude_subject_id::Union{Nothing,String} = "13",
    trim_start_ms::Real = 100,
    trim_end_ms::Real = 100,
    space::Symbol = :physical,
    pressure_calib = pressure_from_adc,
    force_calib = force_newton_from_adc,
    inner_pad::Float64 = 0.05,
    edge_exclusion_frac::Float64 = 0.15,
    n_permutations::Int = 200,
    random_seed::Union{Nothing,Integer} = nothing,
    max_points_per_set::Union{Nothing,Int} = nothing,
    subsample_seed::Union{Nothing,Integer} = nothing,
    distance_kwargs = nothing,
)
    allowed_families = Set((
        :all,
        :nonlegato_low_high,
        :same_note_asc_desc,
        :overtone_vs_nonlegato,
        :legato_low_high,
    ))
    family in allowed_families || error("Unsupported family $(repr(family)). Allowed: $(collect(allowed_families))")
    n_permutations > 0 || error("n_permutations must be positive")
    if !(max_points_per_set === nothing)
        max_points_per_set > 3 || error("max_points_per_set must be > 3 when provided")
    end

    kwargs_nt = if distance_kwargs === nothing
        (;)
    elseif distance_kwargs isa NamedTuple
        distance_kwargs
    elseif distance_kwargs isa AbstractDict
        (; (Symbol(k) => v for (k, v) in pairs(distance_kwargs))...)
    else
        error("distance_kwargs must be nothing, a NamedTuple, or an AbstractDict")
    end
    hasproperty(kwargs_nt, :scales) || error("distance_kwargs must include :scales")
    hasproperty(kwargs_nt, :bandwidth) || error("distance_kwargs must include :bandwidth")
    distance_method = hasproperty(kwargs_nt, :method) ? kwargs_nt.method : :internal
    distance_method in (:internal, :optimaltransport) || error("Unsupported distance method $(repr(distance_method)). Use :internal or :optimaltransport")
    compute_kwargs = (; (k => getproperty(kwargs_nt, k) for k in keys(kwargs_nt) if k != :method)...)

    rows = NamedTuple[]
    seed_counter_ref = Ref(0)

    _run_family(sym::Symbol) = (family == :all || family == sym)

    function _next_seed()
        if random_seed === nothing
            return nothing
        end
        seed_counter_ref[] += 1
        return Int(random_seed) + seed_counter_ref[]
    end

    function _finite_points(pts::AbstractMatrix)
        keep = vec(all(isfinite, pts; dims = 2))
        return pts[keep, :]
    end

    rng_subsample = isnothing(subsample_seed) ? Random.default_rng() : Random.MersenneTwister(subsample_seed)

    function _maybe_subsample_points(pts::AbstractMatrix)
        max_points_per_set === nothing && return pts
        n = size(pts, 1)
        n <= max_points_per_set && return pts
        idx = Random.randperm(rng_subsample, n)[1:max_points_per_set]
        return pts[idx, :]
    end

    function _eligible_subject_ids(selector)
        ids = String[]
        for tr in trials
            tr.success || continue
            tr.subject_id == exclude_subject_id && continue
            selector(tr) || continue
            push!(ids, tr.subject_id)
        end
        return sort(unique(ids))
    end

    function _subject_stratified_perm_row(subject_pairs;
        family,
        comparison,
        task,
        typ,
        note,
        label_a,
        label_b,
        statistic_level = "subject_mean",
    )
        # Clean and keep only subjects with enough finite samples in both sets.
        prepared = NamedTuple[]
        for sp in subject_pairs
            pts_a_full = _finite_points(sp.a)
            pts_b_full = _finite_points(sp.b)
            pts_a = _maybe_subsample_points(pts_a_full)
            pts_b = _maybe_subsample_points(pts_b_full)
            if size(pts_a, 1) > 3 && size(pts_b, 1) > 3
                push!(prepared, (
                    subject_id = sp.subject_id,
                    a = Matrix{Float64}(pts_a),
                    b = Matrix{Float64}(pts_b),
                    n_a_full = size(pts_a_full, 1),
                    n_b_full = size(pts_b_full, 1),
                    n_a = size(pts_a, 1),
                    n_b = size(pts_b, 1),
                ))
            end
        end

        if isempty(prepared)
            push!(rows, (
                statistic_level = statistic_level,
                family = family,
                comparison = comparison,
                task = task,
                typ = typ,
                note = note,
                label_a = label_a,
                label_b = label_b,
                n_a_full = 0,
                n_b_full = 0,
                n_a = 0,
                n_b = 0,
                observed = NaN,
                observed_sd = NaN,
                null_mean = NaN,
                null_sd = NaN,
                p_value = NaN,
                n_units = 0,
                n_units_used = 0,
                n_perm_requested = n_permutations,
                n_perm_valid = 0,
                status = "insufficient samples",
            ))
            return
        end

        labels = [label_a, label_b]
        observed_by_subject = Float64[]
        valid_subjects = NamedTuple[]
        dropped_subjects = 0

        function _pair_distance(points_a, points_b)
            result = if distance_method == :optimaltransport
                compute_kde_wasserstein_distances_optimaltransport(
                    [points_a, points_b];
                    compute_kwargs...,
                    labels = labels,
                )
            else
                compute_kde_wasserstein_distances(
                    [points_a, points_b];
                    compute_kwargs...,
                    labels = labels,
                )
            end
            return result.wasserstein_distance_matrix[1, 2]
        end

        for sp in prepared
            try
                push!(observed_by_subject, _pair_distance(sp.a, sp.b))
                push!(valid_subjects, sp)
            catch
                dropped_subjects += 1
            end
        end

        if isempty(valid_subjects)
            push!(rows, (
                statistic_level = statistic_level,
                family = family,
                comparison = comparison,
                task = task,
                typ = typ,
                note = note,
                label_a = label_a,
                label_b = label_b,
                n_a_full = 0,
                n_b_full = 0,
                n_a = 0,
                n_b = 0,
                observed = NaN,
                observed_sd = NaN,
                null_mean = NaN,
                null_sd = NaN,
                p_value = NaN,
                n_units = length(prepared),
                n_units_used = 0,
                n_perm_requested = n_permutations,
                n_perm_valid = 0,
                status = "error: no valid units after KDE filtering",
            ))
            return
        end

        observed_distance = mean(observed_by_subject)
        observed_sd = length(observed_by_subject) > 1 ? std(observed_by_subject) : 0.0
        total_n_a_full = sum(sp -> sp.n_a_full, valid_subjects)
        total_n_b_full = sum(sp -> sp.n_b_full, valid_subjects)
        total_n_a = sum(sp -> sp.n_a, valid_subjects)
        total_n_b = sum(sp -> sp.n_b, valid_subjects)

        rng = isnothing(random_seed) ? Random.default_rng() : Random.MersenneTwister(_next_seed())
        null_distances = Float64[]
        dropped_perm_draws = 0

        for perm_idx in 1:n_permutations
            perm_subject_vals = Float64[]
            for sp in valid_subjects
                combined = vcat(sp.a, sp.b)
                total_n = sp.n_a + sp.n_b
                perm = Random.randperm(rng, total_n)
                perm_a = combined[perm[1:sp.n_a], :]
                perm_b = combined[perm[(sp.n_a + 1):end], :]

                try
                    push!(perm_subject_vals, _pair_distance(perm_a, perm_b))
                catch
                end
            end

            if isempty(perm_subject_vals)
                dropped_perm_draws += 1
            else
                push!(null_distances, mean(perm_subject_vals))
            end
        end

        if isempty(null_distances)
            push!(rows, (
                statistic_level = statistic_level,
                family = family,
                comparison = comparison,
                task = task,
                typ = typ,
                note = note,
                label_a = label_a,
                label_b = label_b,
                n_a_full = total_n_a_full,
                n_b_full = total_n_b_full,
                n_a = total_n_a,
                n_b = total_n_b,
                observed = observed_distance,
                observed_sd = observed_sd,
                null_mean = NaN,
                null_sd = NaN,
                p_value = NaN,
                n_units = length(prepared),
                n_units_used = length(valid_subjects),
                n_perm_requested = n_permutations,
                n_perm_valid = 0,
                status = "error: all permutation draws failed",
            ))
            return
        end

        exceedance_count = count(d -> d >= observed_distance, null_distances)
        p_value = (exceedance_count + 1) / (length(null_distances) + 1)

        push!(rows, (
            statistic_level = statistic_level,
            family = family,
            comparison = comparison,
            task = task,
            typ = typ,
            note = note,
            label_a = label_a,
            label_b = label_b,
            n_a_full = total_n_a_full,
            n_b_full = total_n_b_full,
            n_a = total_n_a,
            n_b = total_n_b,
            observed = observed_distance,
            observed_sd = observed_sd,
            null_mean = mean(null_distances),
            null_sd = length(null_distances) > 1 ? std(null_distances) : 0.0,
            p_value = p_value,
            n_units = length(prepared),
            n_units_used = length(valid_subjects),
            n_perm_requested = n_permutations,
            n_perm_valid = length(null_distances),
            status = "ok",
        ))
    end

    if _run_family(:nonlegato_low_high)
        for c in cond_specs
            pairs = NamedTuple[]
            selected = filter(
                tr -> tr.success && tr.subject_id != exclude_subject_id && tr.task == c.task && tr.type == c.typ,
                trials,
            )

            for tr in selected
                pts = try
                    nonlegato_point_sets_from_trial(
                        tr;
                        trim_start_ms = trim_start_ms,
                        trim_end_ms = trim_end_ms,
                        space = space,
                        pressure_calib = pressure_calib,
                        force_calib = force_calib,
                        inner_pad = inner_pad,
                        edge_exclusion_frac = edge_exclusion_frac,
                    )
                catch
                    nothing
                end

                if !(pts === nothing)
                    push!(pairs, (
                        subject_id = tr.subject_id,
                        a = pts.low,
                        b = pts.high,
                    ))
                end
            end

            _subject_stratified_perm_row(pairs;
                family = "1) Nonlegato low vs high",
                comparison = "Subject-stratified",
                task = string(c.task),
                typ = string(c.typ),
                note = "low-vs-high",
                label_a = "low",
                label_b = "high",
                statistic_level = "trial_mean",
            )
        end
    end

    if _run_family(:same_note_asc_desc)
        for typ in (:Model, :Real)
            subj_ids = _eligible_subject_ids(tr -> tr.type == typ && tr.task in (:NonlegatoAsc, :NonlegatoDesc))
            for note in (:low, :high)
                pairs = NamedTuple[]
                for sid in subj_ids
                    asc_pool = try
                        nonlegato_point_sets_from_trials(
                            trials;
                            task = :NonlegatoAsc,
                            typ = typ,
                            subject_id = sid,
                            only_success = true,
                            exclude_subject_id = exclude_subject_id,
                            trim_start_ms = trim_start_ms,
                            trim_end_ms = trim_end_ms,
                            space = space,
                            pressure_calib = pressure_calib,
                            force_calib = force_calib,
                            inner_pad = inner_pad,
                            edge_exclusion_frac = edge_exclusion_frac,
                            pool = true,
                        )
                    catch
                        nothing
                    end
                    desc_pool = try
                        nonlegato_point_sets_from_trials(
                            trials;
                            task = :NonlegatoDesc,
                            typ = typ,
                            subject_id = sid,
                            only_success = true,
                            exclude_subject_id = exclude_subject_id,
                            trim_start_ms = trim_start_ms,
                            trim_end_ms = trim_end_ms,
                            space = space,
                            pressure_calib = pressure_calib,
                            force_calib = force_calib,
                            inner_pad = inner_pad,
                            edge_exclusion_frac = edge_exclusion_frac,
                            pool = true,
                        )
                    catch
                        nothing
                    end

                    if !(asc_pool === nothing || asc_pool.pooled === nothing || desc_pool === nothing || desc_pool.pooled === nothing)
                        push!(pairs, (
                            subject_id = sid,
                            a = getfield(asc_pool.pooled, note),
                            b = getfield(desc_pool.pooled, note),
                        ))
                    end
                end

                _subject_stratified_perm_row(pairs;
                    family = "2) Same note Asc vs Desc",
                    comparison = "Subject-stratified",
                    task = "Asc-vs-Desc",
                    typ = string(typ),
                    note = string(note),
                    label_a = "Asc",
                    label_b = "Desc",
                    statistic_level = "subject_mean",
                )
            end
        end
    end

    if _run_family(:overtone_vs_nonlegato)
        for c in cond_specs
            subj_ids = _eligible_subject_ids(tr -> tr.type == c.typ && tr.task in (:Overtone, c.task))
            for note in (:low, :high)
                pairs = NamedTuple[]
                for sid in subj_ids
                    overtone_chunks = Matrix{Float64}[]
                    subj_overtones = filter(
                        tr -> tr.success && tr.subject_id == sid && tr.subject_id != exclude_subject_id && tr.task == :Overtone && tr.type == c.typ,
                        trials,
                    )
                    for tr in subj_overtones
                        try
                            pts = _overtone_interval_points(
                                tr;
                                trim_start_ms = trim_start_ms,
                                trim_end_ms = trim_end_ms,
                                axis_scaling = space,
                            )
                            push!(overtone_chunks, pts)
                        catch
                        end
                    end

                    nl_pool = try
                        nonlegato_point_sets_from_trials(
                            trials;
                            task = c.task,
                            typ = c.typ,
                            subject_id = sid,
                            only_success = true,
                            exclude_subject_id = exclude_subject_id,
                            trim_start_ms = trim_start_ms,
                            trim_end_ms = trim_end_ms,
                            space = space,
                            pressure_calib = pressure_calib,
                            force_calib = force_calib,
                            inner_pad = inner_pad,
                            edge_exclusion_frac = edge_exclusion_frac,
                            pool = true,
                        )
                    catch
                        nothing
                    end

                    if !isempty(overtone_chunks) && !(nl_pool === nothing || nl_pool.pooled === nothing)
                        push!(pairs, (
                            subject_id = sid,
                            a = reduce(vcat, overtone_chunks),
                            b = getfield(nl_pool.pooled, note),
                        ))
                    end
                end

                _subject_stratified_perm_row(pairs;
                    family = "3) Multiphonic vs nonlegato note",
                    comparison = "Subject-stratified",
                    task = string(c.task),
                    typ = string(c.typ),
                    note = string(note),
                    label_a = "Multiphonic",
                    label_b = string(note),
                    statistic_level = "subject_mean",
                )
            end
        end
    end

    if _run_family(:legato_low_high)
        for c in legato_specs
            pairs = NamedTuple[]
            selected = filter(
                tr -> tr.success && tr.subject_id != exclude_subject_id && tr.task == c.task && tr.type == c.typ,
                trials,
            )
            for tr in selected
                p = try
                    _legato_note_points_from_trial(
                        tr;
                        space = space,
                        pressure_calib = pressure_calib,
                        force_calib = force_calib,
                        inner_pad = inner_pad,
                    )
                catch
                    nothing
                end

                if !(p === nothing)
                    push!(pairs, (
                        subject_id = tr.subject_id,
                        a = p.low,
                        b = p.high,
                    ))
                end
            end

            _subject_stratified_perm_row(pairs;
                family = "4) Legato low vs high",
                comparison = "Subject-stratified",
                task = string(c.task),
                typ = string(c.typ),
                note = "low-vs-high",
                label_a = "low",
                label_b = "high",
                statistic_level = "trial_mean",
            )
        end
    end

    return rows
end

function compute_nonlegato_trial_wasserstein_distances(tr;
    scales,
    bandwidth::Real,
    buffer_size::Union{Nothing,Real} = nothing,
    grid_spec = WassersteinGridSpec(),
    trim_start_ms::Real = 0,
    trim_end_ms::Real = 0,
    space::Symbol = :physical,
    pressure_calib = pressure_from_adc,
    force_calib = force_newton_from_adc,
    inner_pad::Float64 = 0.05,
    min_dur::Float64 = 0.05,
    edge_exclusion_frac::Float64 = 0.15,
    sinkhorn_reg::Union{Nothing,Real} = nothing,
    sinkhorn_maxiter::Int = 1000,
    sinkhorn_tol::Float64 = 1e-7
)
    pts = nonlegato_point_sets_from_trial(
        tr;
        trim_start_ms = trim_start_ms,
        trim_end_ms = trim_end_ms,
        space = space,
        pressure_calib = pressure_calib,
        force_calib = force_calib,
        inner_pad = inner_pad,
        min_dur = min_dur,
        edge_exclusion_frac = edge_exclusion_frac
    )

    return compute_kde_wasserstein_distances(
        [pts.low, pts.high];
        scales = scales,
        bandwidth = bandwidth,
        buffer_size = buffer_size,
        grid_spec = grid_spec,
        sinkhorn_reg = sinkhorn_reg,
        sinkhorn_maxiter = sinkhorn_maxiter,
        sinkhorn_tol = sinkhorn_tol,
        labels = ["low", "high"]
    )
end

function compute_nonlegato_trial_wasserstein_distances_optimaltransport(tr;
    scales,
    bandwidth::Real,
    buffer_size::Union{Nothing,Real} = nothing,
    grid_spec = WassersteinGridSpec(),
    trim_start_ms::Real = 0,
    trim_end_ms::Real = 0,
    space::Symbol = :physical,
    pressure_calib = pressure_from_adc,
    force_calib = force_newton_from_adc,
    inner_pad::Float64 = 0.05,
    min_dur::Float64 = 0.05,
    edge_exclusion_frac::Float64 = 0.15,
    sinkhorn_reg::Union{Nothing,Real} = nothing,
    sinkhorn_maxiter::Int = 1000,
    sinkhorn_tol::Float64 = 1e-7
)
    pts = nonlegato_point_sets_from_trial(
        tr;
        trim_start_ms = trim_start_ms,
        trim_end_ms = trim_end_ms,
        space = space,
        pressure_calib = pressure_calib,
        force_calib = force_calib,
        inner_pad = inner_pad,
        min_dur = min_dur,
        edge_exclusion_frac = edge_exclusion_frac
    )

    return compute_kde_wasserstein_distances_optimaltransport(
        [pts.low, pts.high];
        scales = scales,
        bandwidth = bandwidth,
        buffer_size = buffer_size,
        grid_spec = grid_spec,
        sinkhorn_reg = sinkhorn_reg,
        sinkhorn_maxiter = sinkhorn_maxiter,
        sinkhorn_tol = sinkhorn_tol,
        labels = ["low", "high"]
    )
end

function compute_nonlegato_pooled_wasserstein_distances(trials::AbstractVector;
    task::Union{Nothing,Symbol} = nothing,
    typ::Union{Nothing,Symbol} = nothing,
    subject_id::Union{Nothing,String} = nothing,
    scales,
    bandwidth::Real,
    buffer_size::Union{Nothing,Real} = nothing,
    grid_spec = WassersteinGridSpec(),
    trim_start_ms::Real = 0,
    trim_end_ms::Real = 0,
    space::Symbol = :physical,
    pressure_calib = pressure_from_adc,
    force_calib = force_newton_from_adc,
    inner_pad::Float64 = 0.05,
    min_dur::Float64 = 0.05,
    edge_exclusion_frac::Float64 = 0.15,
    only_success::Bool = true,
    exclude_subject_id::Union{Nothing,String} = "13",
    sinkhorn_reg::Union{Nothing,Real} = nothing,
    sinkhorn_maxiter::Int = 1000,
    sinkhorn_tol::Float64 = 1e-7,
    pool::Bool = true
)
    grouped = nonlegato_point_sets_from_trials(
        trials;
        task = task,
        typ = typ,
        subject_id = subject_id,
        only_success = only_success,
        exclude_subject_id = exclude_subject_id,
        trim_start_ms = trim_start_ms,
        trim_end_ms = trim_end_ms,
        space = space,
        pressure_calib = pressure_calib,
        force_calib = force_calib,
        inner_pad = inner_pad,
        min_dur = min_dur,
        edge_exclusion_frac = edge_exclusion_frac,
        pool = pool
    )

    pooled = grouped.pooled === nothing ? (
        low = isempty(grouped.low_sets) ? zeros(0, 2) : reduce(vcat, grouped.low_sets),
        high = isempty(grouped.high_sets) ? zeros(0, 2) : reduce(vcat, grouped.high_sets)
    ) : grouped.pooled

    result = compute_kde_wasserstein_distances(
        [pooled.low, pooled.high];
        scales = scales,
        bandwidth = bandwidth,
        buffer_size = buffer_size,
        grid_spec = grid_spec,
        sinkhorn_reg = sinkhorn_reg,
        sinkhorn_maxiter = sinkhorn_maxiter,
        sinkhorn_tol = sinkhorn_tol,
        labels = ["low", "high"]
    )

    return (
        selected_trials = grouped.selected_trials,
        per_trial = grouped.per_trial,
        pooled_point_sets = pooled,
        wasserstein = result,
    )
end

function compute_nonlegato_pooled_wasserstein_distances_optimaltransport(trials::AbstractVector;
    task::Union{Nothing,Symbol} = nothing,
    typ::Union{Nothing,Symbol} = nothing,
    subject_id::Union{Nothing,String} = nothing,
    scales,
    bandwidth::Real,
    buffer_size::Union{Nothing,Real} = nothing,
    grid_spec = WassersteinGridSpec(),
    trim_start_ms::Real = 0,
    trim_end_ms::Real = 0,
    space::Symbol = :physical,
    pressure_calib = pressure_from_adc,
    force_calib = force_newton_from_adc,
    inner_pad::Float64 = 0.05,
    min_dur::Float64 = 0.05,
    edge_exclusion_frac::Float64 = 0.15,
    only_success::Bool = true,
    exclude_subject_id::Union{Nothing,String} = "13",
    sinkhorn_reg::Union{Nothing,Real} = nothing,
    sinkhorn_maxiter::Int = 1000,
    sinkhorn_tol::Float64 = 1e-7,
    pool::Bool = true
)
    grouped = nonlegato_point_sets_from_trials(
        trials;
        task = task,
        typ = typ,
        subject_id = subject_id,
        only_success = only_success,
        exclude_subject_id = exclude_subject_id,
        trim_start_ms = trim_start_ms,
        trim_end_ms = trim_end_ms,
        space = space,
        pressure_calib = pressure_calib,
        force_calib = force_calib,
        inner_pad = inner_pad,
        min_dur = min_dur,
        edge_exclusion_frac = edge_exclusion_frac,
        pool = pool
    )

    pooled = grouped.pooled === nothing ? (
        low = isempty(grouped.low_sets) ? zeros(0, 2) : reduce(vcat, grouped.low_sets),
        high = isempty(grouped.high_sets) ? zeros(0, 2) : reduce(vcat, grouped.high_sets)
    ) : grouped.pooled

    result = compute_kde_wasserstein_distances_optimaltransport(
        [pooled.low, pooled.high];
        scales = scales,
        bandwidth = bandwidth,
        buffer_size = buffer_size,
        grid_spec = grid_spec,
        sinkhorn_reg = sinkhorn_reg,
        sinkhorn_maxiter = sinkhorn_maxiter,
        sinkhorn_tol = sinkhorn_tol,
        labels = ["low", "high"]
    )

    return (
        selected_trials = grouped.selected_trials,
        per_trial = grouped.per_trial,
        pooled_point_sets = pooled,
        wasserstein = result,
    )
end

"""
    plot_kde_contour(point_sets; x_scale, y_scale, bandwidth, kwargs...)

Plot KDE density as a filled contour for one or more sets of physical-space points.
Normalisation and optional buffering are applied internally before computing the KDE.

# Arguments
- `point_sets`:      A single `Matrix` (N×2) or a `Vector` of matrices. Columns are [x, y] in physical units.
- `x_scale`:         Normalisation divisor for x (e.g. `1.8` for pressure in kPa).
- `y_scale`:         Normalisation divisor for y (e.g. `10.0` for force in N).
- `bandwidth`:       KDE smoothing bandwidth in **normalised** units (tune this to compare smoothing levels).
- `buffer_size`:     Pre-KDE buffering radius in normalised units; `nothing` = disabled.
- `grid_spec`:       `WassersteinGridSpec` — grid resolution and padding.
- `xlabel`, `ylabel`, `title`: Axis/plot labels.
- `levels`:          Number of filled-contour levels.
- `show_points`:     Overlay raw data points as a semi-transparent scatter layer.
- `labels`:          Subplot title string(s) for each set (overrides `title` when multiple sets).
- `original_units`:  When `true` (default) axis values are shown in physical units.
- `subplot_size`:    `(width, height)` in pixels per subplot panel.
"""
function plot_kde_contour(
    point_sets;
    x_scale::Real,
    y_scale::Real,
    bandwidth::Real,
    buffer_size::Union{Nothing,Real} = nothing,
    grid_spec = WassersteinGridSpec(nx = 60, ny = 60, pad_frac = 0.10),
    xlabel::String = "Pressure (kPa)",
    ylabel::String = "Force (N)",
    title::String = "",
    levels::Int = 10,
    show_points::Bool = true,
    labels = nothing,
    original_units::Bool = true,
    subplot_size::Tuple{Int,Int} = (480, 420)
)
    # Canonicalise to a vector of matrices.
    sets = if isa(point_sets, AbstractMatrix)
        [Matrix{Float64}(point_sets)]
    else
        [Matrix{Float64}(p) for p in point_sets]
    end
    isempty(sets) && error("plot_kde_contour: no point sets provided")

    parsed_scales = _parse_wasserstein_scales((Float64(x_scale), Float64(y_scale)))
    bw = Float64(bandwidth)
    spec = _coerce_wasserstein_grid_spec(grid_spec)
    buf = buffer_size === nothing ? nothing : Float64(buffer_size)

    # Filter non-finite rows and normalise each set.
    normalized = Vector{Matrix{Float64}}(undef, length(sets))
    for (i, pts) in enumerate(sets)
        fpts = pts[vec(all(isfinite, pts; dims = 2)), :]
        size(fpts, 1) > 3 || error("plot_kde_contour: set $i has fewer than 4 finite points after filtering")
        normalized[i] = _normalize_point_set_matrix(fpts, parsed_scales)
    end

    # Build a shared normalised grid over all sets.
    xedges, yedges, xgrid, ygrid = _grid_from_normalized_point_sets(
        normalized;
        grid_spec = spec,
        buffer_size = buf
    )

    # Optional per-set buffering, then KDE.
    kde_grids = Vector{Matrix{Float64}}(undef, length(sets))
    for (i, pts) in enumerate(normalized)
        support = if buf === nothing
            pts
        else
            buffer_point_set(
                pts;
                buffer_size = buf,
                xedges = xedges,
                yedges = yedges,
                grid_spec = spec
            ).buffered_points
        end
        _, _, dens, _, _ = kde2d_grid(support; xgrid = xgrid, ygrid = ygrid, hx = bw, hy = bw)
        kde_grids[i] = dens
    end

    # Back-scale the grid axes to physical or keep normalised.
    xplot = original_units ? xgrid .* Float64(x_scale) : xgrid
    yplot = original_units ? ygrid .* Float64(y_scale) : ygrid

    # Resolve per-panel labels.
    lbs = if labels === nothing
        fill(title, length(sets))
    elseif isa(labels, AbstractString)
        fill(labels, length(sets))
    else
        collect(string.(labels))
    end

    nsets = length(sets)
    fig_w, fig_h = subplot_size
    plt = plot(
        layout = (1, nsets),
        size = (fig_w * nsets, fig_h),
        left_margin = 8mm,
        bottom_margin = 8mm,
    )

    for i in 1:nsets
        # kde2d_grid returns dens[ix, iy]; contourf expects z[iy, ix] → transpose.
        dens = kde_grids[i]
        contourf!(
            plt, xplot, yplot, dens';
            subplot = i,
            levels = levels,
            xlabel = xlabel,
            ylabel = i == 1 ? ylabel : "",
            title = get(lbs, i, ""),
            color = :viridis,
            linewidth = 0,
            colorbar = true,
        )
        if show_points
            sc_x = original_units ? sets[i][:, 1] : normalized[i][:, 1]
            sc_y = original_units ? sets[i][:, 2] : normalized[i][:, 2]
            scatter!(
                plt, sc_x, sc_y;
                subplot = i,
                color = :white,
                alpha = 0.25,
                markersize = 2,
                markerstrokewidth = 0,
                label = false,
            )
        end
    end

    return plt
end

# Old overlap/centroid metrics are retained only for compatibility with older notebooks and logs.
# New analyses should prefer Wasserstein-based comparison because it jointly captures displacement,
# spread, and partial overlap between distributions.

#endregion



#endregion

#region PAPER FIGURE 5: WASSERSTEIN SUMMARY

include(joinpath(@__DIR__, "paper_wasserstein.jl"))

#endregion
