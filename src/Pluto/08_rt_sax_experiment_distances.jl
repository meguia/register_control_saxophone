### A Pluto.jl notebook ###
# v0.20.27

using Markdown
using InteractiveUtils

# ╔═╡ 2f2f95a6-8d6a-4b5d-9a66-4820438f4c9a
import Pkg; Pkg.activate()

# ╔═╡ f00f4a64-4c5a-4f07-a26f-53e05fc8f5fc
using Plots, ProjectRoot, JLD2, Statistics, Measures

# ╔═╡ 7b7cae7f-5de4-4dcf-9ed5-56cbfb94531b
begin
    session_path = @projectroot("src", "sessions")
    data_path = joinpath(session_path, "processed_data")
    include("../rt_sax_control.jl")
    include("../rt_sax_experiment_analysis.jl")
    include("../rt_sax_experiment_statistics.jl")
    statistics_validate_dependencies()
end

# ╔═╡ 65f73f18-b34f-4cdb-9ec8-117ce4f6d95c
begin
    if !isdefined(Main, :Trial)
        @eval Main const Trial = $(Trial)
    end
    @load joinpath(data_path, "all_model_trials_wamplitudes_onoff.jld2") model_trials
    @load joinpath(data_path, "all_real_trials_wamplitudes_onoff.jld2") real_trials
end

# ╔═╡ f6736de2-85b9-48ed-9f77-a87f8fcb53fd


# ╔═╡ 1027f4a4-281f-4d97-b896-4709e8e2bb0f
begin
    # Ensure mode regions are available for all helper extractors.
    trials = vcat(model_trials, real_trials)
end

# ╔═╡ d3ca3b44-bcf8-4dd4-b96a-ec87c4e2a8af
begin
	# correction (temporal)
	M83 = filter_trials(trials, subject_ids = ["83"], blocks = [2],task=:Overtone, type=:Real)[1]
	M83.success = false
end	

# ╔═╡ 1f7df18f-cb85-4c19-a802-a049f8e8da14
md"""
# 08 - Wasserstein Distance Summaries

Simplified notebook focused on Wasserstein comparisons only.
No 2D maps are shown, only summary bar plots and tables.
"""

# ╔═╡ 3991afce-c191-4ec2-925a-cdd35f435b40
begin
    cond_specs = [
        (task = :NonlegatoAsc, typ = :Model, label = "Asc Model"),
        (task = :NonlegatoDesc, typ = :Model, label = "Desc Model"),
        (task = :NonlegatoAsc, typ = :Real, label = "Asc Instrument"),
        (task = :NonlegatoDesc, typ = :Real, label = "Desc Instrument"),
    ]

    legato_specs = [
        (task = :LegatoAsc, typ = :Model, label = "Asc Model"),
        (task = :LegatoDesc, typ = :Model, label = "Desc Model"),
        (task = :LegatoAsc, typ = :Real, label = "Asc Instrument"),
        (task = :LegatoDesc, typ = :Real, label = "Desc Instrument"),
    ]

    # Tunable KDE/transport parameters in normalized units.
    W_BANDWIDTH = 0.03
    W_BUFFER = 0.03
    W_GRID = WassersteinGridSpec(nx = 30, ny = 30, pad_frac = 0.08)

    # Global Wasserstein backend for the entire notebook.
    # Options: :internal (default Sinkhorn in this repo) or :optimaltransport (OptimalTransport.jl).
    W_METHOD = :optimaltransport
    W_SINKHORN_REG = 0.10
    W_SINKHORN_MAXITER = 50_000
    W_SINKHORN_TOL = 1e-5

    # Common extraction settings.
    EXCLUDE_SUBJECT = "97"
    TRIM_MS = 100
    INNER_PAD = 0.05
    EDGE_EXCL = 0.15

    # Global pooled-SD scaling (physical -> normalized units), without mean subtraction.
    W_GLOBAL_SCALE_INFO = compute_global_wasserstein_scales(
        trials;
        cond_specs = cond_specs,
        legato_specs = legato_specs,
        exclude_subject_id = EXCLUDE_SUBJECT,
        trim_start_ms = TRIM_MS,
        trim_end_ms = TRIM_MS,
        space = :physical,
        pressure_calib = pressure_from_adc,
        force_calib = force_newton_from_adc,
        inner_pad = INNER_PAD,
        edge_exclusion_frac = EDGE_EXCL,
    )
    W_SCALES = W_GLOBAL_SCALE_INFO.scales
    X_SCALE_KPA = W_SCALES[1]
    Y_SCALE_N = W_SCALES[2]

    # Permutation-test controls (used in Section 4b).
    PERM_N_PERMUTATIONS = 20
    PERM_RANDOM_SEED = 42
    PERM_RESULTS_DIRNAME = "permutation_tests_N100"
    PERM_RESULTS_DIR = joinpath(data_path, PERM_RESULTS_DIRNAME)
end

# ╔═╡ 0f29e76e-ef24-48ee-80df-a7a48e2bf8f0
begin
    _fmt(x) = !isfinite(x) ? "NaN" : (abs(x) >= 1e-3 ? string(round(x, digits = 3)) : string(round(x, sigdigits = 3)))

    function _finite_points(pts::AbstractMatrix)
        keep = vec(all(isfinite, pts; dims = 2))
        return pts[keep, :]
    end

    function _mean_sd_n(vals)
        v = [Float64(x) for x in vals if isfinite(x)]
        n = length(v)
        if n == 0
            return (mean = NaN, sd = NaN, n = 0)
        elseif n == 1
            return (mean = v[1], sd = 0.0, n = 1)
        else
            return (mean = mean(v), sd = std(v), n = n)
        end
    end

    function _plot_group_bar(rows; title, ylabel, colors = :steelblue)
        means = Float64[]
        sds = Float64[]
        labels = String[]
        for r in rows
            push!(means, Float64(r.mean))
            push!(sds, Float64(r.sd))
            push!(labels, String(r.label))
        end

        bar(
            1:length(rows), means;
            yerror = sds,
            xticks = (1:length(rows), labels),
            legend = false,
            title = title,
            ylabel = ylabel,
            xlabel = "",
            color = colors,
            size = (900, 440),
            bottom_margin = 12mm,
            left_margin = 8mm,
            top_margin = 3mm,
            xrotation = 15,
            bar_width = 0.6,
            linewidth = 0,
            grid = :y,
            gridalpha = 0.3,
            framestyle = :box,
        )
    end

    # Returns a per-bar color vector: steelblue for Model, coral for Instrument.
    function _bar_colors(rows)
        [begin
            t = hasproperty(r, :typ) ? r.typ : (hasproperty(r, :type) ? r.type : :unknown)
            t == :Model ? :steelblue : :coral
        end for r in rows]
    end

    function _rows_markdown(rows; headers, values_fn)
        lines = String[]
        push!(lines, "| " * join(headers, " | ") * " |")
        push!(lines, "| " * join(fill("---", length(headers)), " | ") * " |")
        for r in rows
            vals = values_fn(r)
            push!(lines, "| " * join(vals, " | ") * " |")
        end
        Markdown.parse(join(lines, "\n"))
    end

    function _load_perm_rows(file::AbstractString)
        isfile(file) || error("Missing file: $(file). Run scripts/run_permutation_families.jl first.")
        return JLD2.jldopen(file, "r") do f
            haskey(f, "rows") || error("File does not contain key 'rows': $(file)")
            if !haskey(f, "W_GLOBAL_SCALE_INFO")
                return [_stale_perm_row(file, "stale: missing W_GLOBAL_SCALE_INFO; rerun scripts/run_permutation_families.jl")]
            end
            scale_info = read(f, "W_GLOBAL_SCALE_INFO")
            saved_scales = scale_info.scales
            if !(isapprox(saved_scales[1], W_SCALES[1]; rtol = 1e-10, atol = 1e-12) &&
                 isapprox(saved_scales[2], W_SCALES[2]; rtol = 1e-10, atol = 1e-12))
                return [_stale_perm_row(file, "stale: saved scales do not match current global SD scales")]
            end
            if !haskey(f, "EXCLUDE_SUBJECT")
                return [_stale_perm_row(file, "stale: missing EXCLUDE_SUBJECT; rerun scripts/run_permutation_families.jl")]
            end
            saved_exclude_subject = read(f, "EXCLUDE_SUBJECT")
            if saved_exclude_subject != EXCLUDE_SUBJECT
                return [_stale_perm_row(file, "stale: saved EXCLUDE_SUBJECT=$(saved_exclude_subject) does not match current EXCLUDE_SUBJECT=$(EXCLUDE_SUBJECT)")]
            end
            if !haskey(f, "W_METHOD")
                return [_stale_perm_row(file, "stale: missing W_METHOD; rerun scripts/run_permutation_families.jl")]
            end
            saved_method = read(f, "W_METHOD")
            if saved_method != W_METHOD
                return [_stale_perm_row(file, "stale: saved W_METHOD=$(saved_method) does not match current W_METHOD=$(W_METHOD)")]
            end
            if !haskey(f, "W_SINKHORN_MAXITER")
                return [_stale_perm_row(file, "stale: missing W_SINKHORN_MAXITER; rerun scripts/run_permutation_families.jl")]
            end
            saved_sinkhorn_maxiter = read(f, "W_SINKHORN_MAXITER")
            if saved_sinkhorn_maxiter != W_SINKHORN_MAXITER
                return [_stale_perm_row(file, "stale: saved W_SINKHORN_MAXITER=$(saved_sinkhorn_maxiter) does not match current W_SINKHORN_MAXITER=$(W_SINKHORN_MAXITER)")]
            end
            if !haskey(f, "W_SINKHORN_REG")
                return [_stale_perm_row(file, "stale: missing W_SINKHORN_REG; rerun scripts/run_permutation_families.jl")]
            end
            saved_sinkhorn_reg = read(f, "W_SINKHORN_REG")
            if !isapprox(saved_sinkhorn_reg, W_SINKHORN_REG; rtol = 1e-10, atol = 1e-12)
                return [_stale_perm_row(file, "stale: saved W_SINKHORN_REG=$(saved_sinkhorn_reg) does not match current W_SINKHORN_REG=$(W_SINKHORN_REG)")]
            end
            if !haskey(f, "W_SINKHORN_TOL")
                return [_stale_perm_row(file, "stale: missing W_SINKHORN_TOL; rerun scripts/run_permutation_families.jl")]
            end
            saved_sinkhorn_tol = read(f, "W_SINKHORN_TOL")
            if !isapprox(saved_sinkhorn_tol, W_SINKHORN_TOL; rtol = 1e-10, atol = 1e-12)
                return [_stale_perm_row(file, "stale: saved W_SINKHORN_TOL=$(saved_sinkhorn_tol) does not match current W_SINKHORN_TOL=$(W_SINKHORN_TOL)")]
            end
            read(f, "rows")
        end
    end

    function _stale_perm_row(file::AbstractString, status::AbstractString)
        return (
            statistic_level = "stale_file",
            family = basename(file),
            comparison = "not loaded",
            task = "",
            typ = "",
            note = "",
            label_a = "",
            label_b = "",
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
            n_perm_requested = 0,
            n_perm_valid = 0,
            status = String(status),
        )
    end

    function _compute_kde_wasserstein_selected(point_sets; labels = nothing)
        if W_METHOD == :internal
            return compute_kde_wasserstein_distances(
                point_sets;
                scales = W_SCALES,
                bandwidth = W_BANDWIDTH,
                buffer_size = W_BUFFER,
                grid_spec = W_GRID,
                sinkhorn_reg = W_SINKHORN_REG,
                sinkhorn_maxiter = W_SINKHORN_MAXITER,
                sinkhorn_tol = W_SINKHORN_TOL,
                labels = labels,
            )
        elseif W_METHOD == :optimaltransport
            return compute_kde_wasserstein_distances_optimaltransport(
                point_sets;
                scales = W_SCALES,
                bandwidth = W_BANDWIDTH,
                buffer_size = W_BUFFER,
                grid_spec = W_GRID,
                sinkhorn_reg = W_SINKHORN_REG,
                sinkhorn_maxiter = W_SINKHORN_MAXITER,
                sinkhorn_tol = W_SINKHORN_TOL,
                labels = labels,
            )
        else
            error("Unsupported W_METHOD=$(W_METHOD). Use :internal or :optimaltransport")
        end
    end

    function _compute_nonlegato_trial_wd_selected(tr)
        if W_METHOD == :internal
            return compute_nonlegato_trial_wasserstein_distances(
                tr;
                scales = W_SCALES,
                bandwidth = W_BANDWIDTH,
                buffer_size = W_BUFFER,
                grid_spec = W_GRID,
                sinkhorn_reg = W_SINKHORN_REG,
                sinkhorn_maxiter = W_SINKHORN_MAXITER,
                sinkhorn_tol = W_SINKHORN_TOL,
                trim_start_ms = TRIM_MS,
                trim_end_ms = TRIM_MS,
                space = :physical,
                pressure_calib = pressure_from_adc,
                force_calib = force_newton_from_adc,
                inner_pad = INNER_PAD,
                edge_exclusion_frac = EDGE_EXCL,
            )
        elseif W_METHOD == :optimaltransport
            return compute_nonlegato_trial_wasserstein_distances_optimaltransport(
                tr;
                scales = W_SCALES,
                bandwidth = W_BANDWIDTH,
                buffer_size = W_BUFFER,
                grid_spec = W_GRID,
                sinkhorn_reg = W_SINKHORN_REG,
                sinkhorn_maxiter = W_SINKHORN_MAXITER,
                sinkhorn_tol = W_SINKHORN_TOL,
                trim_start_ms = TRIM_MS,
                trim_end_ms = TRIM_MS,
                space = :physical,
                pressure_calib = pressure_from_adc,
                force_calib = force_newton_from_adc,
                inner_pad = INNER_PAD,
                edge_exclusion_frac = EDGE_EXCL,
            )
        else
            error("Unsupported W_METHOD=$(W_METHOD). Use :internal or :optimaltransport")
        end
    end

    function _compute_nonlegato_pooled_wd_selected(trials; task, typ)
        if W_METHOD == :internal
            return compute_nonlegato_pooled_wasserstein_distances(
                trials;
                task = task,
                typ = typ,
                scales = W_SCALES,
                bandwidth = W_BANDWIDTH,
                buffer_size = W_BUFFER,
                grid_spec = W_GRID,
                sinkhorn_reg = W_SINKHORN_REG,
                sinkhorn_maxiter = W_SINKHORN_MAXITER,
                sinkhorn_tol = W_SINKHORN_TOL,
                trim_start_ms = TRIM_MS,
                trim_end_ms = TRIM_MS,
                space = :physical,
                pressure_calib = pressure_from_adc,
                force_calib = force_newton_from_adc,
                inner_pad = INNER_PAD,
                edge_exclusion_frac = EDGE_EXCL,
                only_success = true,
                exclude_subject_id = EXCLUDE_SUBJECT,
            )
        elseif W_METHOD == :optimaltransport
            return compute_nonlegato_pooled_wasserstein_distances_optimaltransport(
                trials;
                task = task,
                typ = typ,
                scales = W_SCALES,
                bandwidth = W_BANDWIDTH,
                buffer_size = W_BUFFER,
                grid_spec = W_GRID,
                trim_start_ms = TRIM_MS,
                trim_end_ms = TRIM_MS,
                space = :physical,
                pressure_calib = pressure_from_adc,
                force_calib = force_newton_from_adc,
                inner_pad = INNER_PAD,
                edge_exclusion_frac = EDGE_EXCL,
                only_success = true,
                exclude_subject_id = EXCLUDE_SUBJECT,
            )
        else
            error("Unsupported W_METHOD=$(W_METHOD). Use :internal or :optimaltransport")
        end
    end

    nothing
end

# ╔═╡ 6c1f4d9e-1cb8-4d1f-b3cf-2f6d9c4a8e17
md"""
## Analysis Map

This notebook uses the Wasserstein pipeline only.

- Intra-subject comparisons: Sections 1, 2, 3, and 4 compute one distance per trial or one distance per subject by pairing point sets within the same subject.
- Inter-subject pooled comparisons: the pooled rows in Sections 1 through 4 concatenate point sets across subjects for the same condition before computing Wasserstein distance.
- Diagnostic KDE inspection: Section 5 is optional and uses buffered KDE only for visualization; it is not part of the core Wasserstein analysis.
"""

# ╔═╡ 0bb8c9f7-c8db-45b0-8833-84e21297f812
md"""
## 1) Nonlegato High vs Low

A) Intra-subject, per trial
B) Inter-subject pooled, per condition
"""

# ╔═╡ 209ce92f-0ee4-4fc6-a5fa-bd2b66a81d9a
nonlegato_trial_data = begin
    nonlegato_trial_rows = NamedTuple[]

    selected_nonlegato = filter(
        tr -> tr.success && tr.subject_id != EXCLUDE_SUBJECT && tr.task in (:NonlegatoAsc, :NonlegatoDesc) && tr.type in (:Model, :Real),
        trials
    )

    for tr in selected_nonlegato
        try
            wd = _compute_nonlegato_trial_wd_selected(tr)

            d = wd.wasserstein_distance_matrix[1, 2]
            push!(nonlegato_trial_rows, (
                subject_id = tr.subject_id,
                task = tr.task,
                type = tr.type,
                block = tr.block,
                take = tr.take,
                distance = d,
            ))
        catch
            # Skip trials that do not yield valid windows.
        end
    end

    nonlegato_trial_summary = NamedTuple[]
    for c in cond_specs
        grp = filter(r -> r.task == c.task && r.type == c.typ, nonlegato_trial_rows)
        stats = _mean_sd_n((r.distance for r in grp))
        push!(nonlegato_trial_summary, merge(c, (mean = stats.mean, sd = stats.sd, n = stats.n)))
    end

    nonlegato_trial_plot = _plot_group_bar(
        nonlegato_trial_summary;
        title = "Nonlegato Low vs High (Intra-trial)",
        ylabel = "Wasserstein distance",
        colors = _bar_colors(nonlegato_trial_summary),
    )

    (
        rows = nonlegato_trial_rows,
        summary = nonlegato_trial_summary,
        plot = nonlegato_trial_plot,
    )
end

# ╔═╡ c37269ce-2f86-4d1e-b88f-fb370c560f85
nonlegato_trial_data.plot

# ╔═╡ d4f697de-a062-4fe8-a901-37d652f0b469
_rows_markdown(
    nonlegato_trial_data.summary;
    headers = ["Task", "Type", "Mean", "SD", "n"],
    values_fn = r -> [string(r.task), string(r.typ), _fmt(r.mean), _fmt(r.sd), string(r.n)]
)

# ╔═╡ 52fb3e4d-cd50-4361-a7ca-ee5f9d63400c
pooled_nonlegato_data = begin
    pooled_nonlegato_rows = NamedTuple[]

    for c in cond_specs
        try
            out = _compute_nonlegato_pooled_wd_selected(trials; task = c.task, typ = c.typ)
            d = out.wasserstein.wasserstein_distance_matrix[1, 2]
            push!(pooled_nonlegato_rows, merge(c, (distance = d, n_trials = length(out.selected_trials))))
        catch
            push!(pooled_nonlegato_rows, merge(c, (distance = NaN, n_trials = 0)))
        end
    end

    pooled_nonlegato_plot = _plot_group_bar(
        [merge(r, (mean = r.distance, sd = 0.0, label = r.label)) for r in pooled_nonlegato_rows];
        title = "Nonlegato Low vs High (Pooled)",
        ylabel = "Wasserstein distance",
        colors = _bar_colors(pooled_nonlegato_rows),
    )

    (rows = pooled_nonlegato_rows, plot = pooled_nonlegato_plot)
end

# ╔═╡ 2762621d-19f4-4bdd-aaab-bb86f9b3a90f
pooled_nonlegato_data.plot

# ╔═╡ 5dde932c-8227-427f-8f5b-f63f13306315
_rows_markdown(
    pooled_nonlegato_data.rows;
    headers = ["Task", "Type", "Pooled distance", "n trials"],
    values_fn = r -> [string(r.task), string(r.typ), _fmt(r.distance), string(r.n_trials)]
)

# ╔═╡ efcc4058-473d-4aa4-b2fd-97cf6f9f70b7
md"""
## 2) Same Note: Asc vs Desc (Intra-subject + pooled)

Compare low with low and high with high, separately for Model and Instrument.
"""

# ╔═╡ e243c0e7-2e7e-4c64-9efd-7d70d4844961
same_note_data = begin
    same_note_subject_rows = NamedTuple[]
    same_note_pooled_rows = NamedTuple[]

    for typ in (:Model, :Real)
        subj_ids = sort(unique(tr.subject_id for tr in trials if tr.type == typ && tr.subject_id != EXCLUDE_SUBJECT))

        for sid in subj_ids
            asc = try
                nonlegato_point_sets_from_trials(
                    trials;
                    task = :NonlegatoAsc,
                    typ = typ,
                    subject_id = sid,
                    only_success = true,
                    exclude_subject_id = nothing,
                    trim_start_ms = TRIM_MS,
                    trim_end_ms = TRIM_MS,
                    space = :physical,
                    pressure_calib = pressure_from_adc,
                    force_calib = force_newton_from_adc,
                    inner_pad = INNER_PAD,
                    edge_exclusion_frac = EDGE_EXCL,
                    pool = true,
                )
            catch
                nothing
            end

            desc = try
                nonlegato_point_sets_from_trials(
                    trials;
                    task = :NonlegatoDesc,
                    typ = typ,
                    subject_id = sid,
                    only_success = true,
                    exclude_subject_id = nothing,
                    trim_start_ms = TRIM_MS,
                    trim_end_ms = TRIM_MS,
                    space = :physical,
                    pressure_calib = pressure_from_adc,
                    force_calib = force_newton_from_adc,
                    inner_pad = INNER_PAD,
                    edge_exclusion_frac = EDGE_EXCL,
                    pool = true,
                )
            catch
                nothing
            end

            isnothing(asc) && continue
            isnothing(desc) && continue

            for note in (:low, :high)
                pts_asc = _finite_points(getfield(asc.pooled, note))
                pts_desc = _finite_points(getfield(desc.pooled, note))
                size(pts_asc, 1) > 3 || continue
                size(pts_desc, 1) > 3 || continue

                try
                    wd = _compute_kde_wasserstein_selected([pts_asc, pts_desc]; labels = ["Asc", "Desc"])

                    push!(same_note_subject_rows, (
                        type = typ,
                        note = note,
                        subject_id = sid,
                        distance = wd.wasserstein_distance_matrix[1, 2],
                    ))
                catch
                    # Skip rare invalid KDE cases.
                end
            end
        end

        asc_pool = try
            nonlegato_point_sets_from_trials(trials; task = :NonlegatoAsc, typ = typ, only_success = true, exclude_subject_id = EXCLUDE_SUBJECT, trim_start_ms = TRIM_MS, trim_end_ms = TRIM_MS, space = :physical, pressure_calib = pressure_from_adc, force_calib = force_newton_from_adc, inner_pad = INNER_PAD, edge_exclusion_frac = EDGE_EXCL, pool = true)
        catch
            nothing
        end
        desc_pool = try
            nonlegato_point_sets_from_trials(trials; task = :NonlegatoDesc, typ = typ, only_success = true, exclude_subject_id = EXCLUDE_SUBJECT, trim_start_ms = TRIM_MS, trim_end_ms = TRIM_MS, space = :physical, pressure_calib = pressure_from_adc, force_calib = force_newton_from_adc, inner_pad = INNER_PAD, edge_exclusion_frac = EDGE_EXCL, pool = true)
        catch
            nothing
        end

        if isnothing(asc_pool) || isnothing(desc_pool)
            for note in (:low, :high)
                push!(same_note_pooled_rows, (type = typ, note = note, distance = NaN))
            end
            continue
        end

        for note in (:low, :high)
            pts_asc = _finite_points(getfield(asc_pool.pooled, note))
            pts_desc = _finite_points(getfield(desc_pool.pooled, note))
            if size(pts_asc, 1) > 3 && size(pts_desc, 1) > 3
                try
                    wd = _compute_kde_wasserstein_selected([pts_asc, pts_desc]; labels = ["Asc", "Desc"])
                    push!(same_note_pooled_rows, (type = typ, note = note, distance = wd.wasserstein_distance_matrix[1, 2]))
                catch
                    push!(same_note_pooled_rows, (type = typ, note = note, distance = NaN))
                end
            else
                push!(same_note_pooled_rows, (type = typ, note = note, distance = NaN))
            end
        end
    end

    same_note_subject_summary = NamedTuple[]
    for typ in (:Model, :Real), note in (:low, :high)
        grp = filter(r -> r.type == typ && r.note == note, same_note_subject_rows)
        stats = _mean_sd_n((r.distance for r in grp))
        label = string(note) * " " * (typ == :Model ? "Model" : "Instrument")
        push!(same_note_subject_summary, (type = typ, note = note, label = label, mean = stats.mean, sd = stats.sd, n = stats.n))
    end

    same_note_plot = _plot_group_bar(
        same_note_subject_summary;
        title = "Same Note Asc vs Desc (Subject-level)",
        ylabel = "Wasserstein distance",
        colors = _bar_colors(same_note_subject_summary),
    )

    (
        subject_rows = same_note_subject_rows,
        subject_summary = same_note_subject_summary,
        pooled_rows = same_note_pooled_rows,
        plot = same_note_plot,
    )
end

# ╔═╡ 9f1b0b76-fa2d-45bf-8517-8f6d0ab0d31f
same_note_data.plot

# ╔═╡ e6ec8ad8-b4c6-45f1-a90e-9dab2a565616
_rows_markdown(
    same_note_data.subject_summary;
    headers = ["Type", "Note", "Mean", "SD", "n"],
    values_fn = r -> [string(r.type), string(r.note), _fmt(r.mean), _fmt(r.sd), string(r.n)]
)

# ╔═╡ 29e7092f-f2f0-495d-a24f-6e88c286af75
_rows_markdown(
    same_note_data.pooled_rows;
    headers = ["Type", "Note", "Pooled distance"],
    values_fn = r -> [string(r.type), string(r.note), _fmt(r.distance)]
)

# ╔═╡ 5909c7b4-3bb2-44ef-8ca5-a06ec4bcbb95
md"""
## 3) Multiphonic vs Nonlegato Note (Intra-subject + pooled)

For each nonlegato condition and note, compare multiphonic pooled points (same type)
against the condition-specific nonlegato note points.
"""

# ╔═╡ 981d53dc-c786-463a-a208-5cb443535634
overtone_vs_nonlegato_data = begin
    section3_overtone_subject_chunks = Dict{Tuple{Symbol,String}, Vector{Matrix{Float64}}}()

    section3_overtone_trials = filter(
        tr -> tr.success && tr.subject_id != EXCLUDE_SUBJECT && tr.task == :Overtone && tr.type in (:Model, :Real),
        trials
    )

    for tr in section3_overtone_trials
        try
            pts = _overtone_interval_points(
                tr;
                trim_start_ms = TRIM_MS,
                trim_end_ms = TRIM_MS,
                axis_scaling = :physical,
            )
            push!(get!(section3_overtone_subject_chunks, (tr.type, tr.subject_id), Matrix{Float64}[]), pts)
        catch
            # Skip unusable multiphonic intervals.
        end
    end

    overtone_vs_nonlegato_subject_rows = NamedTuple[]

    for c in cond_specs
        for sid in sort(unique(tr.subject_id for tr in trials if tr.type == c.typ && tr.subject_id != EXCLUDE_SUBJECT))
            overtone_pts = isempty(get(section3_overtone_subject_chunks, (c.typ, sid), Matrix{Float64}[])) ? zeros(0, 2) : reduce(vcat, get(section3_overtone_subject_chunks, (c.typ, sid), Matrix{Float64}[]))
            overtone_pts = _finite_points(overtone_pts)

            nl = try
                nonlegato_point_sets_from_trials(
                    trials;
                    task = c.task,
                    typ = c.typ,
                    subject_id = sid,
                    only_success = true,
                    exclude_subject_id = nothing,
                    trim_start_ms = TRIM_MS,
                    trim_end_ms = TRIM_MS,
                    space = :physical,
                    pressure_calib = pressure_from_adc,
                    force_calib = force_newton_from_adc,
                    inner_pad = INNER_PAD,
                    edge_exclusion_frac = EDGE_EXCL,
                    pool = true,
                )
            catch
                nothing
            end

            isnothing(nl) && continue

            for note in (:low, :high)
                note_pts = _finite_points(getfield(nl.pooled, note))
                size(overtone_pts, 1) > 3 || continue
                size(note_pts, 1) > 3 || continue

                try
                    wd = _compute_kde_wasserstein_selected([overtone_pts, note_pts]; labels = ["Multiphonic", string(note)])

                    push!(overtone_vs_nonlegato_subject_rows, (
                        task = c.task,
                        type = c.typ,
                        note = note,
                        subject_id = sid,
                        distance = wd.wasserstein_distance_matrix[1, 2],
                    ))
                catch
                    # Skip rare invalid KDE cases.
                end
            end
        end
    end

    overtone_vs_nonlegato_summary = NamedTuple[]
    for c in cond_specs, note in (:low, :high)
        grp = filter(r -> r.task == c.task && r.type == c.typ && r.note == note, overtone_vs_nonlegato_subject_rows)
        stats = _mean_sd_n((r.distance for r in grp))
        label = string(c.task == :NonlegatoAsc ? "Asc" : "Desc") * " " * string(note) * " " * (c.typ == :Model ? "Model" : "Inst")
        push!(overtone_vs_nonlegato_summary, (task = c.task, typ = c.typ, note = note, label = label, mean = stats.mean, sd = stats.sd, n = stats.n))
    end

    overtone_vs_nonlegato_plot = _plot_group_bar(
        overtone_vs_nonlegato_summary;
        title = "Multiphonic vs Nonlegato Note (Subject-level)",
        ylabel = "Wasserstein distance",
        colors = _bar_colors(overtone_vs_nonlegato_summary),
    )

    (rows = overtone_vs_nonlegato_subject_rows, summary = overtone_vs_nonlegato_summary, plot = overtone_vs_nonlegato_plot)
end

# ╔═╡ 4d81039f-2d87-4a2f-b2e6-9b8a4ecee9f2
overtone_vs_nonlegato_data.plot

# ╔═╡ 2ec5df38-d7af-427f-8e8b-f7e2ef6c84d2
_rows_markdown(
    overtone_vs_nonlegato_data.summary;
    headers = ["Task", "Type", "Note", "Mean", "SD", "n"],
    values_fn = r -> [string(r.task), string(r.typ), string(r.note), _fmt(r.mean), _fmt(r.sd), string(r.n)]
)

# ╔═╡ 2d91fe4b-545f-4513-97ba-bbe1f8f95f7e
md"""
## 4) Legato High vs Low (Intra-subject + pooled)

Distance between low/high note regions in legato trials.
"""

# ╔═╡ d88d3f27-7557-49d4-8de5-b43bfe4f1774
legato_data = begin
    legato_subject_rows = NamedTuple[]

    selected_legato = filter(
        tr -> tr.success && tr.subject_id != EXCLUDE_SUBJECT && tr.task in (:LegatoAsc, :LegatoDesc) && tr.type in (:Model, :Real),
        trials
    )

    for tr in selected_legato
        try
            pts = _legato_note_points_from_trial(
                tr;
                space = :physical,
                pressure_calib = pressure_from_adc,
                force_calib = force_newton_from_adc,
                inner_pad = INNER_PAD,
            )

            wd = _compute_kde_wasserstein_selected([pts.low, pts.high]; labels = ["low", "high"])

            push!(legato_subject_rows, (
                task = tr.task,
                type = tr.type,
                subject_id = tr.subject_id,
                block = tr.block,
                take = tr.take,
                distance = wd.wasserstein_distance_matrix[1, 2],
            ))
        catch
            # Skip unusable legato trials.
        end
    end

    legato_subject_summary = NamedTuple[]
    for c in legato_specs
        grp = filter(r -> r.task == c.task && r.type == c.typ, legato_subject_rows)
        stats = _mean_sd_n((r.distance for r in grp))
        push!(legato_subject_summary, merge(c, (mean = stats.mean, sd = stats.sd, n = stats.n)))
    end

    legato_subject_plot = _plot_group_bar(
        legato_subject_summary;
        title = "Legato Low vs High (Subject-level)",
        ylabel = "Wasserstein distance",
        colors = _bar_colors(legato_subject_summary),
    )

    # Pooled by condition.
    pooled_legato_rows = NamedTuple[]
    for c in legato_specs
        leg_trials = filter(
            tr -> tr.success && tr.subject_id != EXCLUDE_SUBJECT && tr.task == c.task && tr.type == c.typ,
            trials
        )

        legato_low_chunks = Matrix{Float64}[]
        legato_high_chunks = Matrix{Float64}[]
        for tr in leg_trials
            try
                p = _legato_note_points_from_trial(
                    tr;
                    space = :physical,
                    pressure_calib = pressure_from_adc,
                    force_calib = force_newton_from_adc,
                    inner_pad = INNER_PAD,
                )
                push!(legato_low_chunks, p.low)
                push!(legato_high_chunks, p.high)
            catch
            end
        end

        low_pool = isempty(legato_low_chunks) ? zeros(0, 2) : reduce(vcat, legato_low_chunks)
        high_pool = isempty(legato_high_chunks) ? zeros(0, 2) : reduce(vcat, legato_high_chunks)

        if size(low_pool, 1) > 3 && size(high_pool, 1) > 3
            wd = _compute_kde_wasserstein_selected([low_pool, high_pool]; labels = ["low", "high"])
            push!(pooled_legato_rows, merge(c, (distance = wd.wasserstein_distance_matrix[1, 2], n_trials = length(leg_trials))))
        else
            push!(pooled_legato_rows, merge(c, (distance = NaN, n_trials = 0)))
        end
    end

    pooled_legato_plot = _plot_group_bar(
        [merge(r, (mean = r.distance, sd = 0.0, label = r.label)) for r in pooled_legato_rows];
        title = "Legato Low vs High (Pooled)",
        ylabel = "Wasserstein distance",
        colors = _bar_colors(pooled_legato_rows),
    )

    (
        rows = legato_subject_rows,
        summary = legato_subject_summary,
        plot = legato_subject_plot,
        pooled_rows = pooled_legato_rows,
        pooled_plot = pooled_legato_plot,
    )
end

# ╔═╡ b7ac9d6f-590d-4e56-beb6-8eaf53039880
legato_data.plot

# ╔═╡ 58e348d5-6f4c-4f8e-9588-f8dcb7145a8e
_rows_markdown(
    legato_data.summary;
    headers = ["Task", "Type", "Mean", "SD", "n"],
    values_fn = r -> [string(r.task), string(r.typ), _fmt(r.mean), _fmt(r.sd), string(r.n)]
)

# ╔═╡ 87cf3c09-4b8c-4ece-a6f7-cbeb08f10437
legato_data.pooled_plot

# ╔═╡ f18f0a63-556f-4ebf-9e0d-ebf2c3c6f84e
_rows_markdown(
    legato_data.pooled_rows;
    headers = ["Task", "Type", "Pooled distance", "n trials"],
    values_fn = r -> [string(r.task), string(r.typ), _fmt(r.distance), string(r.n_trials)]
)

# ╔═╡ a9b8c7d6-5e4f-3a2b-1c0d-ef9a8b7c6d5e
md"""
## Distance Summary

All six Wasserstein bar charts in a single figure.
Blue = Model · Orange = Instrument
"""

# ╔═╡ b0c1d2e3-4f5a-6b7c-8d9e-0f1a2b3c4d5e
let
    DIST_SUMMARY_YLIMS = (0.0, 3.0)
    DIST_SUMMARY_SIZE = (1400, 1200)
    DIST_SUMMARY_TITLEFONT = 26
    DIST_SUMMARY_TICKFONT = 18
    DIST_SUMMARY_GUIDEFONT = 20
    DIST_SUMMARY_PLOTTITLEFONT = 24

    function _condition_code(task, typ)
        dir_code = occursin("Asc", string(task)) ? "A" : "D"
        type_code = typ == :Model ? "M" : "R"
        return dir_code * type_code
    end

    function _type_code(typ)
        return typ == :Model ? "M" : "R"
    end

    function _note_code(note)
        return uppercase(string(note))
    end

    function _distance_summary_label(r)
        if hasproperty(r, :note) && hasproperty(r, :task) && hasproperty(r, :typ)
            return _note_code(r.note) * "\n" * _condition_code(r.task, r.typ)
        elseif hasproperty(r, :note) && hasproperty(r, :type)
            return _note_code(r.note) * " " * _type_code(r.type)
        elseif hasproperty(r, :task) && hasproperty(r, :typ)
            return _condition_code(r.task, r.typ)
        elseif hasproperty(r, :label)
            return String(r.label)
        else
            return ""
        end
    end

    function _sp_bar(rows; title, show_ylabel = false)
        means  = [Float64(r.mean) for r in rows]
        sds    = [Float64(r.sd)   for r in rows]
        labels = [_distance_summary_label(r) for r in rows]
        colors = _bar_colors(rows)
        bar(
            1:length(rows), means;
            yerror        = sds,
            xticks        = (1:length(rows), labels),
            legend        = false,
            title         = title,
            ylabel        = show_ylabel ? "W distance" : "",
            xlabel        = "",
            color         = colors,
            ylims         = DIST_SUMMARY_YLIMS,
            xrotation     = 0,
            bar_width     = 0.6,
            linewidth     = 2,
            grid          = :y,
            gridalpha     = 0.3,
            framestyle    = :box,
            titlefontsize = DIST_SUMMARY_TITLEFONT,
            tickfontsize  = DIST_SUMMARY_TICKFONT,
            guidefontsize = DIST_SUMMARY_GUIDEFONT,
            bottom_margin = 9mm,
            left_margin   = show_ylabel ? 11mm : 4mm,
            top_margin    = 4mm,
            right_margin  = 3mm,
        )
    end

    _nl_pooled  = [(task = r.task, typ = r.typ, mean = r.distance, sd = 0.0)
                   for r in pooled_nonlegato_data.rows]
    _leg_pooled = [(task = r.task, typ = r.typ, mean = r.distance, sd = 0.0)
                   for r in legato_data.pooled_rows]
    _multi_low = [(task = r.task, typ = r.typ, mean = r.mean, sd = r.sd)
                  for r in overtone_vs_nonlegato_data.summary if r.note == :low]
    _multi_high = [(task = r.task, typ = r.typ, mean = r.mean, sd = r.sd)
                   for r in overtone_vs_nonlegato_data.summary if r.note == :high]

    p1 = _sp_bar(nonlegato_trial_data.summary;       title = "Nonlegato low-high (trial)", show_ylabel = true)
    p2 = _sp_bar(_nl_pooled;                         title = "Nonlegato low-high (pooled)")
    p3 = _sp_bar(legato_data.summary;                title = "Legato low-high (trial)", show_ylabel = true)
    p4 = _sp_bar(_leg_pooled;                        title = "Legato low-high (pooled)")
    p5 = _sp_bar(_multi_low;                         title = "Multiphonic vs nonlegato low", show_ylabel = true)
    p6 = _sp_bar(_multi_high;                        title = "Multiphonic vs nonlegato high")
    p7 = _sp_bar(same_note_data.subject_summary;     title = "Same note asc-desc", show_ylabel = true)
    p8 = plot([NaN], [NaN]; legend = false, framestyle = :none, grid = false, xticks = ([], []), yticks = ([], []))

    pfig = plot(
        p1, p2, p3, p4, p5, p6, p7, p8;
        layout             = (4, 2),
        size               = DIST_SUMMARY_SIZE,
        plot_title         = "",
        plot_titlefontsize = DIST_SUMMARY_PLOTTITLEFONT,
    )
    savefig(pfig, "Figure5.png")
    savefig(pfig, "Figure5.svg")
	pfig
end

# ╔═╡ 3f4a6b8c-1d2e-4f5a-9b0c-7d8e9f0a1b2c
md"""
## 5) Permutation Tests (Null: Same Underlying 2D Distribution)

This section loads precomputed permutation-test rows (generated by
`scripts/run_permutation_families.jl`) and reports whether the observed
Wasserstein distance is larger than expected under random relabeling.
"""

# ╔═╡ 4a5b6c7d-2e3f-4a5b-8c9d-0e1f2a3b4c5d
permutation_rows_nonlegato_low_high = let
    _file = joinpath(PERM_RESULTS_DIR, "permutation_rows_nonlegato_low_high.jld2")
    _load_perm_rows(_file)
end

# ╔═╡ 5b6c7d8e-3f4a-4b5c-9d0e-1f2a3b4c5d6e
permutation_rows_same_note_asc_desc = let
    _file = joinpath(PERM_RESULTS_DIR, "permutation_rows_same_note_asc_desc.jld2")
    _load_perm_rows(_file)
end

# ╔═╡ 6c7d8e9f-4a5b-4c6d-9e0f-2a3b4c5d6e7f
permutation_rows_overtone_vs_nonlegato = let
    _file = joinpath(PERM_RESULTS_DIR, "permutation_rows_overtone_vs_nonlegato.jld2")
    _load_perm_rows(_file)
end

# ╔═╡ 7d8e9f0a-5b6c-4d7e-9f0a-3b4c5d6e7f8a
permutation_rows_legato_low_high = let
    _file = joinpath(PERM_RESULTS_DIR, "permutation_rows_legato_low_high.jld2")
    _load_perm_rows(_file)
end

# ╔═╡ 8e9f0a1b-6c7d-4e8f-9a0b-4c5d6e7f8a9b
permutation_test_rows = vcat(
    permutation_rows_nonlegato_low_high,
    permutation_rows_same_note_asc_desc,
    permutation_rows_overtone_vs_nonlegato,
    permutation_rows_legato_low_high,
)

# ╔═╡ 9f0a1b2c-7d8e-4f9a-0b1c-5d6e7f8a9b0c
_rows_markdown(
    permutation_test_rows;
    headers = ["Statistic level", "Family", "Comparison", "Task", "Type", "Note", "nA", "nB", "Observed", "Observed SD", "n units", "n units used", "Null mean", "Null sd", "n perm valid", "p (greater)", "Status"],
    values_fn = r -> [
        get(r, :statistic_level, "unknown"),
        r.family,
        r.comparison,
        r.task,
        r.typ,
        r.note,
        string(r.n_a),
        string(r.n_b),
        _fmt(r.observed),
        _fmt(get(r, :observed_sd, NaN)),
        string(get(r, :n_units, -1)),
        string(get(r, :n_units_used, -1)),
        _fmt(r.null_mean),
        _fmt(r.null_sd),
        string(get(r, :n_perm_valid, -1)),
        _fmt(r.p_value),
        r.status,
    ]
)

# ╔═╡ a0b1c2d3-8e9f-401a-1b2c-6e7f8a9b0c1d
permutation_test_rows_ok_sorted = begin
    ok_rows = filter(r -> r.status == "ok" && isfinite(r.p_value), permutation_test_rows)
    sort(ok_rows; by = r -> r.p_value)
end

# ╔═╡ b1c2d3e4-9f0a-412b-2c3d-7f8a9b0c1d2e
_rows_markdown(
    permutation_test_rows_ok_sorted;
    headers = ["Statistic level", "Family", "Comparison", "Task", "Type", "Note", "nA", "nB", "Observed", "Observed SD", "n units", "n units used", "Null mean", "Null sd", "n perm valid", "p (greater)"],
    values_fn = r -> [
        get(r, :statistic_level, "unknown"),
        r.family,
        r.comparison,
        r.task,
        r.typ,
        r.note,
        string(r.n_a),
        string(r.n_b),
        _fmt(r.observed),
        _fmt(get(r, :observed_sd, NaN)),
        string(get(r, :n_units, -1)),
        string(get(r, :n_units_used, -1)),
        _fmt(r.null_mean),
        _fmt(r.null_sd),
        string(get(r, :n_perm_valid, -1)),
        _fmt(r.p_value),
    ]
)

# ╔═╡ 54c5f93e-9096-4dac-a4d1-a17568f2a2db
md"""
### Notes

- All distances shown here come from the normalized full Wasserstein pipeline.
- Active backend for all sections: `W_METHOD` (`:internal` or `:optimaltransport`).
- Normalization used in this test notebook:
  - x scale (pressure): 1.8 kPa
  - y scale (force): 10 N
- Legacy KDE-overlap and centroid metrics are intentionally not used in this notebook.
"""

# ╔═╡ a3c5e7b9-1d2f-4a6c-8e0b-2d4f6a8c0e1f
md"""
## 6) KDE Density Inspection

Use `plot_kde_contour` to visualise the KDE for any point set and explore the effect of different bandwidths.
This section is diagnostic only; it is not part of the core Wasserstein analysis.

- **Change `KDE_INSPECT_BW`** to compare smoothing levels (try 0.03, 0.06, 0.12 …).
- **Change `INSPECT_SUBJECT`**, `INSPECT_TASK`, or `INSPECT_TYPE` to pick a different condition.

*Both a single-subject example (pooled over that subject's trials) and a fully pooled example are shown.*
"""

# ╔═╡ b2d4f6a8-0c1e-4b7d-9f1a-3e5c7a9b1d2f
begin
    KDE_INSPECT_BW  = 0.03    # bandwidth in normalised units — change to inspect
    KDE_INSPECT_BUF = 0.03   # buffer size (normalised units); set to nothing to disable
    INSPECT_SUBJECT = "27"       # ← change to any valid subject_id
    INSPECT_TASK    = :NonlegatoAsc   # :NonlegatoAsc or :NonlegatoDesc
    INSPECT_TYPE    = :Model          # :Model or :Real
end

# ╔═╡ c1e3a5f7-9b0d-4c2e-8a4f-6b8d0e2a4c6f
begin
    _subj_pts = try
        nonlegato_point_sets_from_trials(
            trials;
            task = INSPECT_TASK,
            typ  = INSPECT_TYPE,
            subject_id = INSPECT_SUBJECT,
            only_success = true,
            exclude_subject_id = nothing,
            trim_start_ms  = TRIM_MS,
            trim_end_ms    = TRIM_MS,
            space          = :physical,
            pressure_calib = pressure_from_adc,
            force_calib    = force_newton_from_adc,
            inner_pad        = INNER_PAD,
            edge_exclusion_frac = EDGE_EXCL,
            pool = true,
        )
    catch _e
        nothing
    end

    if _subj_pts !== nothing &&
            size(_subj_pts.pooled.low,  1) > 3 &&
            size(_subj_pts.pooled.high, 1) > 3
        plot_kde_contour(
            [_subj_pts.pooled.low, _subj_pts.pooled.high];
            x_scale    = X_SCALE_KPA,
            y_scale    = Y_SCALE_N,
            bandwidth  = KDE_INSPECT_BW,
            buffer_size = KDE_INSPECT_BUF,
            grid_spec  = WassersteinGridSpec(nx = 60, ny = 60, pad_frac = 0.10),
            xlabel     = "Pressure (kPa)",
            ylabel     = "Force (N)",
            title      = "Subject $INSPECT_SUBJECT / $INSPECT_TASK / $INSPECT_TYPE  (bw=$(KDE_INSPECT_BW))",
            labels     = ["low note", "high note"],
            show_points = true,
            levels     = 12,
        )
    else
        md"⚠ No valid data for subject **$INSPECT_SUBJECT** · **$INSPECT_TASK** / **$INSPECT_TYPE**."
    end
end

# ╔═╡ d0f2c4a6-8e9b-4d1f-bf3a-5e7b9d1f3c5a
begin
    _pool_pts = try
        nonlegato_point_sets_from_trials(
            trials;
            task = INSPECT_TASK,
            typ  = INSPECT_TYPE,
            only_success = true,
            exclude_subject_id = EXCLUDE_SUBJECT,
            trim_start_ms  = TRIM_MS,
            trim_end_ms    = TRIM_MS,
            space          = :physical,
            pressure_calib = pressure_from_adc,
            force_calib    = force_newton_from_adc,
            inner_pad        = INNER_PAD,
            edge_exclusion_frac = EDGE_EXCL,
            pool = true,
        )
    catch _e
        nothing
    end

    if _pool_pts !== nothing &&
            size(_pool_pts.pooled.low,  1) > 3 &&
            size(_pool_pts.pooled.high, 1) > 3
        plot_kde_contour(
            [_pool_pts.pooled.low, _pool_pts.pooled.high];
            x_scale    = X_SCALE_KPA,
            y_scale    = Y_SCALE_N,
            bandwidth  = KDE_INSPECT_BW,
            buffer_size = KDE_INSPECT_BUF,
            grid_spec  = WassersteinGridSpec(nx = 60, ny = 60, pad_frac = 0.10),
            xlabel     = "Pressure (kPa)",
            ylabel     = "Force (N)",
            title      = "Pooled / $INSPECT_TASK / $INSPECT_TYPE  (bw=$(KDE_INSPECT_BW))",
            labels     = ["low note", "high note"],
            show_points = true,
            levels     = 12,
        )
    else
        md"⚠ No valid pooled data for **$INSPECT_TASK** / **$INSPECT_TYPE**."
    end
end

# ╔═╡ c5dc7d7e-4da5-45a7-bde9-1a5c1cbd3730
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 1600px;
    	padding-left: max(160px, 10%);
    	padding-right: max(160px, 10%);
	}
	input[type*="range"] {
		width: 90%;
	}
</style>
"""

# ╔═╡ Cell order:
# ╠═2f2f95a6-8d6a-4b5d-9a66-4820438f4c9a
# ╠═f00f4a64-4c5a-4f07-a26f-53e05fc8f5fc
# ╠═7b7cae7f-5de4-4dcf-9ed5-56cbfb94531b
# ╠═65f73f18-b34f-4cdb-9ec8-117ce4f6d95c
# ╠═f6736de2-85b9-48ed-9f77-a87f8fcb53fd
# ╠═1027f4a4-281f-4d97-b896-4709e8e2bb0f
# ╠═d3ca3b44-bcf8-4dd4-b96a-ec87c4e2a8af
# ╟─1f7df18f-cb85-4c19-a802-a049f8e8da14
# ╠═3991afce-c191-4ec2-925a-cdd35f435b40
# ╟─0f29e76e-ef24-48ee-80df-a7a48e2bf8f0
# ╟─6c1f4d9e-1cb8-4d1f-b3cf-2f6d9c4a8e17
# ╟─0bb8c9f7-c8db-45b0-8833-84e21297f812
# ╟─209ce92f-0ee4-4fc6-a5fa-bd2b66a81d9a
# ╠═c37269ce-2f86-4d1e-b88f-fb370c560f85
# ╟─d4f697de-a062-4fe8-a901-37d652f0b469
# ╟─52fb3e4d-cd50-4361-a7ca-ee5f9d63400c
# ╠═2762621d-19f4-4bdd-aaab-bb86f9b3a90f
# ╟─5dde932c-8227-427f-8f5b-f63f13306315
# ╟─efcc4058-473d-4aa4-b2fd-97cf6f9f70b7
# ╟─e243c0e7-2e7e-4c64-9efd-7d70d4844961
# ╠═9f1b0b76-fa2d-45bf-8517-8f6d0ab0d31f
# ╟─e6ec8ad8-b4c6-45f1-a90e-9dab2a565616
# ╟─29e7092f-f2f0-495d-a24f-6e88c286af75
# ╟─5909c7b4-3bb2-44ef-8ca5-a06ec4bcbb95
# ╠═981d53dc-c786-463a-a208-5cb443535634
# ╠═4d81039f-2d87-4a2f-b2e6-9b8a4ecee9f2
# ╟─2ec5df38-d7af-427f-8e8b-f7e2ef6c84d2
# ╟─2d91fe4b-545f-4513-97ba-bbe1f8f95f7e
# ╟─d88d3f27-7557-49d4-8de5-b43bfe4f1774
# ╠═b7ac9d6f-590d-4e56-beb6-8eaf53039880
# ╟─58e348d5-6f4c-4f8e-9588-f8dcb7145a8e
# ╠═87cf3c09-4b8c-4ece-a6f7-cbeb08f10437
# ╟─f18f0a63-556f-4ebf-9e0d-ebf2c3c6f84e
# ╟─a9b8c7d6-5e4f-3a2b-1c0d-ef9a8b7c6d5e
# ╠═b0c1d2e3-4f5a-6b7c-8d9e-0f1a2b3c4d5e
# ╟─3f4a6b8c-1d2e-4f5a-9b0c-7d8e9f0a1b2c
# ╠═4a5b6c7d-2e3f-4a5b-8c9d-0e1f2a3b4c5d
# ╠═5b6c7d8e-3f4a-4b5c-9d0e-1f2a3b4c5d6e
# ╠═6c7d8e9f-4a5b-4c6d-9e0f-2a3b4c5d6e7f
# ╠═7d8e9f0a-5b6c-4d7e-9f0a-3b4c5d6e7f8a
# ╠═8e9f0a1b-6c7d-4e8f-9a0b-4c5d6e7f8a9b
# ╠═9f0a1b2c-7d8e-4f9a-0b1c-5d6e7f8a9b0c
# ╠═a0b1c2d3-8e9f-401a-1b2c-6e7f8a9b0c1d
# ╟─b1c2d3e4-9f0a-412b-2c3d-7f8a9b0c1d2e
# ╟─54c5f93e-9096-4dac-a4d1-a17568f2a2db
# ╟─a3c5e7b9-1d2f-4a6c-8e0b-2d4f6a8c0e1f
# ╠═b2d4f6a8-0c1e-4b7d-9f1a-3e5c7a9b1d2f
# ╟─c1e3a5f7-9b0d-4c2e-8a4f-6b8d0e2a4c6f
# ╟─d0f2c4a6-8e9b-4d1f-bf3a-5e7b9d1f3c5a
# ╠═c5dc7d7e-4da5-45a7-bde9-1a5c1cbd3730
