"""
Utilities used by the Figure 6 block of `rt_sax_experiment_analysis.jl`.

The compact JLD2 product contains only portable arrays and named tuples from
the completed, non-regularized fixed-parameter map.  Continuation supplied
finite-amplitude candidate states, but every retained map entry was validated
again by direct integration of the experiment model.
"""

if !isdefined(@__MODULE__, :RT_SAX_PROJECT_ROOT)
    const RT_SAX_PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
end

if !isdefined(@__MODULE__, :RT_SAX_PAPER_SUBJECTS)
    const RT_SAX_PAPER_SUBJECTS =
        ["13", "22", "27", "33", "34", "37", "49", "50",
         "64", "70", "80", "83", "88", "90"]
end

"""Load and validate the portable final non-regularized map used in Figure 6."""
function load_portable_multistability_map(
        path::AbstractString=joinpath(
            RT_SAX_PROJECT_ROOT, "data", "derived",
            "multistability_map_v1.jld2"))
    isfile(path) || error("Portable multistability map not found: $path")
    product = JLD2.load(path, "map")
    validate_multistability_map(product)
    return product
end

"""Validate the map schema, dimensions, logical masks, and published counts."""
function validate_multistability_map(map; strict_reference::Bool=true)
    map.schema_version == 1 ||
        error("Unsupported multistability schema $(map.schema_version)")
    dimensions = (length(map.zeta), length(map.gamma))
    for field in (:known, :t1, :t2, :other, :t1_high_p2,
                  :low_p1_high_p2)
        size(getproperty(map, field)) == dimensions ||
            error("Map field $field has wrong dimensions")
    end
    issorted(map.gamma) || error("Gamma axis is not sorted")
    issorted(map.zeta) || error("Zeta axis is not sorted")
    all(map.t1 .<= map.known) || error("T1 contains unknown cells")
    all(map.t2 .<= map.known) || error("T2 contains unknown cells")

    derived = (
        expected_points=prod(dimensions),
        known=count(map.known),
        t1=count(map.t1),
        t2=count(map.t2),
        overlap=count(map.t1 .& map.t2),
        mixed_completed=length(map.mixed_points),
        mixed_at_cutoff=count(point -> point.right_censored,
                              map.mixed_points),
    )
    for name in propertynames(derived)
        hasproperty(map.counts, name) || error("Map counts omit $name")
        getproperty(map.counts, name) == getproperty(derived, name) ||
            error("Map count mismatch for $name")
    end

    if strict_reference
        dimensions == (72, 389) ||
            error("Expected the final 72 × 389 grid, found $dimensions")
        derived.expected_points == 28_008 || error("Unexpected grid size")
        derived.t1 == 16_635 || error("Unexpected T1 count")
        derived.t2 == 9_095 || error("Unexpected T2 count")
        derived.overlap == 8_661 || error("Unexpected overlap count")
        derived.mixed_completed == 1_181 ||
            error("Unexpected mixed-experiment count")
        derived.mixed_at_cutoff == 269 ||
            error("Unexpected mixed-at-cutoff count")
    end
    return derived
end

function _paper_gaussian_score(gamma, zeta, mask;
        known=trues(size(mask)), sigma_gamma, sigma_zeta,
        radius_sigma::Real=2.5)
    score = fill(NaN, size(mask))
    support = zeros(Int, size(mask))
    for row in axes(mask, 1), column in axes(mask, 2)
        row_first = searchsortedfirst(
            zeta, zeta[row] - radius_sigma * sigma_zeta)
        row_last = searchsortedlast(
            zeta, zeta[row] + radius_sigma * sigma_zeta)
        column_first = searchsortedfirst(
            gamma, gamma[column] - radius_sigma * sigma_gamma)
        column_last = searchsortedlast(
            gamma, gamma[column] + radius_sigma * sigma_gamma)
        row_range = row_first:row_last
        column_range = column_first:column_last
        numerator = 0.0
        denominator = 0.0
        samples = 0
        for rr in row_range, cc in column_range
            known[rr, cc] || continue
            distance = ((gamma[cc] - gamma[column]) / sigma_gamma)^2 +
                       ((zeta[rr] - zeta[row]) / sigma_zeta)^2
            weight = exp(-0.5 * distance)
            numerator += weight * mask[rr, cc]
            denominator += weight
            samples += 1
        end
        score[row, column] = denominator > 0 ? numerator / denominator : NaN
        support[row, column] = samples
    end
    return (score=score, support=support)
end

function _paper_refine_field(gamma, zeta, values, factor::Int)
    factor >= 1 || error("Display refinement must be positive")
    factor == 1 && return (
        gamma=Float64.(gamma), zeta=Float64.(zeta),
        values=Float64.(values))
    dense_gamma = collect(range(
        first(gamma), last(gamma), length=(length(gamma) - 1) * factor + 1))
    dense_zeta = collect(range(
        first(zeta), last(zeta), length=(length(zeta) - 1) * factor + 1))
    dense = fill(NaN, length(dense_zeta), length(dense_gamma))
    for (row, zvalue) in pairs(dense_zeta),
            (column, gvalue) in pairs(dense_gamma)
        source_column = clamp(searchsortedlast(gamma, gvalue), 1,
                              length(gamma) - 1)
        source_row = clamp(searchsortedlast(zeta, zvalue), 1,
                           length(zeta) - 1)
        gc = (gvalue - gamma[source_column]) /
             (gamma[source_column + 1] - gamma[source_column])
        zc = (zvalue - zeta[source_row]) /
             (zeta[source_row + 1] - zeta[source_row])
        corners = (
            values[source_row, source_column],
            values[source_row, source_column + 1],
            values[source_row + 1, source_column],
            values[source_row + 1, source_column + 1],
        )
        any(isnan, corners) && continue
        dense[row, column] =
            (1 - zc) * ((1 - gc) * corners[1] + gc * corners[2]) +
            zc * ((1 - gc) * corners[3] + gc * corners[4])
    end
    return (gamma=dense_gamma, zeta=dense_zeta, values=dense)
end

"""
Build the presentation fields for Figure 6 from the immutable raw masks.

Smoothing and interpolation change only the plotted local-support fields; the
raw point classifications in the JLD2 file are never modified.  The defaults
are the values used for the submitted figure.
"""
function paper_multistability_fields(map;
        smoothing_factor::Real=0.7,
        threshold::Real=0.7,
        base_sigma_gamma::Real=0.020,
        base_sigma_zeta::Real=0.040,
        display_refinement::Int=5,
        mixed_sigma_gamma::Real=0.030,
        mixed_sigma_zeta::Real=0.060,
        mixed_threshold::Real=0.55,
        mixed_minimum_support::Int=3)
    smoothing_factor >= 0 || error("Smoothing factor must be non-negative")
    0 < threshold < 1 || error("Threshold must lie in (0, 1)")
    sigma_gamma = smoothing_factor * base_sigma_gamma
    sigma_zeta = smoothing_factor * base_sigma_zeta

    if smoothing_factor == 0
        raw(mask) = (
            score=Base.map((value, known) -> known ? Float64(value) : NaN,
                           mask, map.known),
            support=Int.(map.known),
        )
        t1_raw, t2_raw, p2_raw =
            raw(map.t1), raw(map.t2), raw(map.low_p1_high_p2)
        refinement = 1
    else
        t1_raw = _paper_gaussian_score(
            map.gamma, map.zeta, map.t1; known=map.known,
            sigma_gamma, sigma_zeta)
        t2_raw = _paper_gaussian_score(
            map.gamma, map.zeta, map.t2; known=map.known,
            sigma_gamma, sigma_zeta)
        p2_raw = _paper_gaussian_score(
            map.gamma, map.zeta, map.low_p1_high_p2; known=map.known,
            sigma_gamma, sigma_zeta)
        refinement = display_refinement
    end

    t1 = _paper_refine_field(map.gamma, map.zeta, t1_raw.score, refinement)
    t2 = _paper_refine_field(map.gamma, map.zeta, t2_raw.score, refinement)
    p2 = _paper_refine_field(map.gamma, map.zeta, p2_raw.score, refinement)
    classes = zeros(Int8, size(t1.values))
    t1_support = t1.values .>= threshold
    t2_support = t2.values .>= threshold
    classes[t1_support .& .!t2_support] .= 1
    classes[t2_support .& .!t1_support] .= 2
    classes[t1_support .& t2_support] .= 3

    mixed_score = zeros(size(map.t1))
    mixed_support = zeros(Int, size(map.t1))
    for row in axes(map.t1, 1), column in axes(map.t1, 2)
        numerator = 0.0
        denominator = 0.0
        samples = 0
        for point in map.mixed_points
            dg = (map.gamma[column] - point.gamma) / mixed_sigma_gamma
            dz = (map.zeta[row] - point.zeta) / mixed_sigma_zeta
            distance = dg^2 + dz^2
            distance <= 2.5^2 || continue
            weight = exp(-0.5 * distance)
            numerator += weight * point.right_censored
            denominator += weight
            samples += 1
        end
        mixed_score[row, column] =
            denominator > 0 ? numerator / denominator : 0.0
        mixed_support[row, column] = samples
    end
    mixed_dense = _paper_refine_field(
        map.gamma, map.zeta, mixed_score, refinement)
    support_dense = _paper_refine_field(
        map.gamma, map.zeta, mixed_support, refinement)
    eligibility = (classes .== 3) .| (p2.values .>= threshold)
    mixed_region =
        (mixed_dense.values .>= mixed_threshold) .&
        (support_dense.values .>= mixed_minimum_support) .& eligibility

    return (;
        gamma=t1.gamma, zeta=t1.zeta,
        t1=t1.values, t2=t2.values, classes,
        mixed_score=mixed_dense.values,
        mixed_support=support_dense.values,
        mixed_region,
        threshold=Float64(threshold),
        smoothing_factor=Float64(smoothing_factor),
        sigma_gamma, sigma_zeta,
        raw=(t1=t1_raw, t2=t2_raw),
    )
end

function _paper_nearest_index(axis, value)
    upper = searchsortedfirst(axis, value)
    upper <= 1 && return 1
    upper > length(axis) && return length(axis)
    lower = upper - 1
    return abs(axis[lower] - value) <= abs(axis[upper] - value) ?
        lower : upper
end

function _paper_add_patterns!(axis, fields;
        spacing::Real=0.01, decimation::Int=1,
        linewidth::Real=0.75, alpha::Real=0.8)
    step = spacing * decimation
    gamma = collect(first(fields.gamma):step:last(fields.gamma))
    zeta = collect(first(fields.zeta):step:last(fields.zeta))
    half = 0.36 * step
    for zvalue in zeta, gvalue in gamma
        class = fields.classes[
            _paper_nearest_index(fields.zeta, zvalue),
            _paper_nearest_index(fields.gamma, gvalue)]
        if class in (1, 3)
            plot!(axis, [gvalue - half, gvalue + half],
                  [zvalue - half, zvalue + half];
                  color=:black, linewidth, alpha, label="")
        end
        if class in (2, 3)
            plot!(axis, [gvalue - half, gvalue + half],
                  [zvalue + half, zvalue - half];
                  color=:black, linewidth, alpha, label="")
        end
    end
    return axis
end

function _paper_parameter_trajectory(trial)
    length(trial.param_map) >= 2 || error("Trial has no parameter map")
    gamma_raw = map(value -> _mapvals(value, trial.param_map[1]), trial.v1)
    zeta_raw = map(value -> _mapvals(value, trial.param_map[2]), trial.v2)
    t_ms, gamma, zeta = _align_two_series(
        trial.t1, gamma_raw, trial.t2, zeta_raw; mode=:auto)
    return (t=Float64.(t_ms) ./ 1000, x=Float64.(gamma), y=Float64.(zeta))
end

function _paper_points_in_window(trajectory, window)
    mask = (trajectory.t .>= window[1]) .& (trajectory.t .<= window[2])
    count(mask) > 0 || return zeros(0, 2)
    return hcat(trajectory.x[mask], trajectory.y[mask])
end

function _paper_trimmed_window(region, trim_start_ms, trim_end_ms)
    window = (region[1] + trim_start_ms / 1000,
              region[2] - trim_end_ms / 1000)
    window[2] > window[1] || error("Safety trim removed the interval")
    return window
end

"""Collect successful model-condition samples over the 14-subject cohort."""
function paper_figure6_gesture_points(trials;
        subject_ids=RT_SAX_PAPER_SUBJECTS,
        nonlegato_trim_start_ms::Real=200,
        nonlegato_trim_end_ms::Real=200,
        multiphonic_trim_start_ms::Real=200,
        multiphonic_trim_end_ms::Real=300)
    allowed = Set(String.(subject_ids))
    low = zeros(0, 2)
    high = zeros(0, 2)
    multiphonic = zeros(0, 2)
    for trial in trials
        trial.type == :Model && trial.success &&
            trial.subject_id in allowed || continue
        trajectory = try
            _paper_parameter_trajectory(trial)
        catch
            continue
        end
        if trial.task in (:NonlegatoAsc, :NonlegatoDesc)
            regions = onoff_regions(trial; mode=2)
            length(regions) >= 2 || continue
            first_points = try
                _paper_points_in_window(
                    trajectory,
                    _paper_trimmed_window(
                        regions[1], nonlegato_trim_start_ms,
                        nonlegato_trim_end_ms))
            catch
                zeros(0, 2)
            end
            second_points = try
                _paper_points_in_window(
                    trajectory,
                    _paper_trimmed_window(
                        regions[2], nonlegato_trim_start_ms,
                        nonlegato_trim_end_ms))
            catch
                zeros(0, 2)
            end
            if trial.task == :NonlegatoAsc
                low = vcat(low, first_points)
                high = vcat(high, second_points)
            else
                high = vcat(high, first_points)
                low = vcat(low, second_points)
            end
        elseif trial.task == :Overtone
            regions = onoff_regions(trial; mode=2)
            isempty(regions) && continue
            region = argmax(r -> r[2] - r[1], regions)
            points = try
                _paper_points_in_window(
                    trajectory,
                    _paper_trimmed_window(
                        region, multiphonic_trim_start_ms,
                        multiphonic_trim_end_ms))
            catch
                zeros(0, 2)
            end
            multiphonic = vcat(multiphonic, points)
        end
    end
    return (; low, high, multiphonic)
end

"""Plot the final Figure 6 map, optionally with experimental gesture samples."""
function plot_paper_multistability_map(map;
        fields=paper_multistability_fields(map),
        trajectories=nothing,
        gamma_limits=(0.02, 0.99),
        zeta_limits=extrema(map.zeta),
        pattern_spacing::Real=0.01,
        pattern_decimation::Int=1,
        mixed_alpha::Real=0.45,
        trajectory_alpha::Real=0.5,
        trajectory_markersize::Real=4.5,
        figure_size=(1120, 800))
    axis = plot(;
        xlabel="γ", ylabel="ζ", xlims=gamma_limits, ylims=zeta_limits,
        legend=:bottomleft, legendfontsize=11, framestyle=:box,
        size=figure_size, background_color=:white, foreground_color=:black)

    mixed_values = fill(NaN, size(fields.mixed_region))
    mixed_values[fields.mixed_region] .= 1.0
    heatmap!(axis, fields.gamma, fields.zeta, mixed_values;
        color=cgrad([:gray65, :gray65]), clims=(0, 1), colorbar=false,
        alpha=mixed_alpha, label="")
    _paper_add_patterns!(axis, fields;
        spacing=pattern_spacing, decimation=pattern_decimation)
    contour!(axis, fields.gamma, fields.zeta, fields.t1;
        levels=[fields.threshold], color=:gray25, linewidth=0.8,
        label="", colorbar_entry=false)
    contour!(axis, fields.gamma, fields.zeta, fields.t2;
        levels=[fields.threshold], color=:gray25, linewidth=0.8,
        label="", colorbar_entry=false)

    # Compact legend keys. The figure caption defines the actual hatch motifs.
    plot!(axis, [NaN], [NaN]; color=:black, label="T₁")
    plot!(axis, [NaN], [NaN]; color=:black, linestyle=:dash, label="T₂")
    scatter!(axis, [NaN], [NaN]; marker=:x, color=:black,
             label="T₁ + T₂")
    scatter!(axis, [NaN], [NaN]; marker=:square, color=:gray65,
             markerstrokecolor=:gray35, label="Mixed")

    if !isnothing(trajectories)
        for (points, color, label) in (
                (trajectories.low, :red, "Low"),
                (trajectories.high, :blue, "High"),
                (trajectories.multiphonic, :purple, "Multiphonic"))
            size(points, 1) == 0 && continue
            scatter!(axis, points[:, 1], points[:, 2];
                color, alpha=trajectory_alpha,
                markersize=trajectory_markersize,
                markerstrokewidth=0, label)
        end
    end
    return axis
end

"""Save one plot as PNG, SVG, and PDF and return the written paths."""
function save_paper_figure(axis, stem::AbstractString)
    mkpath(dirname(stem))
    paths = String[]
    for extension in ("png", "svg", "pdf")
        path = "$stem.$extension"
        savefig(axis, path)
        push!(paths, path)
    end
    return paths
end
