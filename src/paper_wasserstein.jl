"""Paper-level Wasserstein summaries used to build Figure 5."""

function _paper_finite_points(points::AbstractMatrix)
    keep = vec(all(isfinite, points; dims=2))
    return points[keep, :]
end

function _paper_mean_sd_n(values)
    finite = [Float64(value) for value in values if isfinite(value)]
    isempty(finite) && return (mean=NaN, sd=NaN, n=0)
    length(finite) == 1 && return (mean=only(finite), sd=0.0, n=1)
    return (mean=mean(finite), sd=std(finite), n=length(finite))
end

function _paper_wasserstein(point_sets;
        scales, bandwidth, buffer_size, grid_spec,
        sinkhorn_reg, sinkhorn_maxiter, sinkhorn_tol,
        labels=nothing, method::Symbol=:optimaltransport)
    kwargs = (
        scales=scales,
        bandwidth=bandwidth,
        buffer_size=buffer_size,
        grid_spec=grid_spec,
        sinkhorn_reg=sinkhorn_reg,
        sinkhorn_maxiter=sinkhorn_maxiter,
        sinkhorn_tol=sinkhorn_tol,
        labels=labels,
    )
    if method == :optimaltransport
        return compute_kde_wasserstein_distances_optimaltransport(
            point_sets; kwargs...)
    elseif method == :internal
        return compute_kde_wasserstein_distances(point_sets; kwargs...)
    end
    error("Unsupported Wasserstein method $method")
end

"""
Compute every Wasserstein summary displayed in Figure 5.

The defaults reproduce the submitted analysis: physical pressure-force space,
global pooled-SD scaling without mean subtraction, a 30 × 30 KDE grid, and
the `OptimalTransport.jl` Sinkhorn backend.
"""
function compute_paper_wasserstein_summary(trials;
        exclude_subject_id::Union{Nothing,AbstractString}="97",
        trim_ms::Real=100,
        inner_pad::Real=0.05,
        edge_exclusion_frac::Real=0.15,
        bandwidth::Real=0.03,
        buffer_size::Real=0.03,
        grid_spec=WassersteinGridSpec(nx=30, ny=30, pad_frac=0.08),
        method::Symbol=:optimaltransport,
        sinkhorn_reg::Real=0.10,
        sinkhorn_maxiter::Int=50_000,
        sinkhorn_tol::Real=1e-5)
    condition_specs = [
        (task=:NonlegatoAsc, typ=:Model, label="Asc Model"),
        (task=:NonlegatoDesc, typ=:Model, label="Desc Model"),
        (task=:NonlegatoAsc, typ=:Real, label="Asc Instrument"),
        (task=:NonlegatoDesc, typ=:Real, label="Desc Instrument"),
    ]
    legato_specs = [
        (task=:LegatoAsc, typ=:Model, label="Asc Model"),
        (task=:LegatoDesc, typ=:Model, label="Desc Model"),
        (task=:LegatoAsc, typ=:Real, label="Asc Instrument"),
        (task=:LegatoDesc, typ=:Real, label="Desc Instrument"),
    ]
    scale_info = compute_global_wasserstein_scales(
        trials;
        cond_specs=condition_specs,
        legato_specs,
        exclude_subject_id,
        trim_start_ms=trim_ms,
        trim_end_ms=trim_ms,
        space=:physical,
        pressure_calib=pressure_from_adc,
        force_calib=force_newton_from_adc,
        inner_pad,
        edge_exclusion_frac,
    )
    scales = scale_info.scales
    failures = NamedTuple[]

    distance(point_sets; labels=nothing) = _paper_wasserstein(
        point_sets;
        scales,
        bandwidth,
        buffer_size,
        grid_spec,
        sinkhorn_reg,
        sinkhorn_maxiter,
        sinkhorn_tol,
        labels,
        method,
    )

    nonlegato_trial_rows = NamedTuple[]
    selected_nonlegato = filter(trials) do trial
        trial.success &&
        (isnothing(exclude_subject_id) ||
         trial.subject_id != exclude_subject_id) &&
        trial.task in (:NonlegatoAsc, :NonlegatoDesc) &&
        trial.type in (:Model, :Real)
    end
    for trial in selected_nonlegato
        try
            result = if method == :optimaltransport
                compute_nonlegato_trial_wasserstein_distances_optimaltransport(
                    trial;
                    scales,
                    bandwidth,
                    buffer_size,
                    grid_spec,
                    sinkhorn_reg,
                    sinkhorn_maxiter,
                    sinkhorn_tol,
                    trim_start_ms=trim_ms,
                    trim_end_ms=trim_ms,
                    space=:physical,
                    pressure_calib=pressure_from_adc,
                    force_calib=force_newton_from_adc,
                    inner_pad,
                    edge_exclusion_frac,
                )
            else
                compute_nonlegato_trial_wasserstein_distances(
                    trial;
                    scales,
                    bandwidth,
                    buffer_size,
                    grid_spec,
                    sinkhorn_reg,
                    sinkhorn_maxiter,
                    sinkhorn_tol,
                    trim_start_ms=trim_ms,
                    trim_end_ms=trim_ms,
                    space=:physical,
                    pressure_calib=pressure_from_adc,
                    force_calib=force_newton_from_adc,
                    inner_pad,
                    edge_exclusion_frac,
                )
            end
            push!(nonlegato_trial_rows, (
                subject_id=trial.subject_id,
                task=trial.task,
                type=trial.type,
                block=trial.block,
                take=trial.take,
                distance=result.wasserstein_distance_matrix[1, 2],
            ))
        catch exception
            push!(failures, (
                stage=:nonlegato_trial,
                subject_id=trial.subject_id,
                task=trial.task,
                type=trial.type,
                error=sprint(showerror, exception),
            ))
        end
    end
    nonlegato_trial_summary = NamedTuple[]
    for condition in condition_specs
        group = filter(row ->
            row.task == condition.task && row.type == condition.typ,
            nonlegato_trial_rows)
        stats = _paper_mean_sd_n(row.distance for row in group)
        push!(nonlegato_trial_summary, merge(condition, stats))
    end

    nonlegato_pooled_rows = NamedTuple[]
    for condition in condition_specs
        try
            result = if method == :optimaltransport
                compute_nonlegato_pooled_wasserstein_distances_optimaltransport(
                    trials;
                    task=condition.task,
                    typ=condition.typ,
                    scales,
                    bandwidth,
                    buffer_size,
                    grid_spec,
                    trim_start_ms=trim_ms,
                    trim_end_ms=trim_ms,
                    space=:physical,
                    pressure_calib=pressure_from_adc,
                    force_calib=force_newton_from_adc,
                    inner_pad,
                    edge_exclusion_frac,
                    only_success=true,
                    exclude_subject_id,
                )
            else
                compute_nonlegato_pooled_wasserstein_distances(
                    trials;
                    task=condition.task,
                    typ=condition.typ,
                    scales,
                    bandwidth,
                    buffer_size,
                    grid_spec,
                    sinkhorn_reg,
                    sinkhorn_maxiter,
                    sinkhorn_tol,
                    trim_start_ms=trim_ms,
                    trim_end_ms=trim_ms,
                    space=:physical,
                    pressure_calib=pressure_from_adc,
                    force_calib=force_newton_from_adc,
                    inner_pad,
                    edge_exclusion_frac,
                    only_success=true,
                    exclude_subject_id,
                )
            end
            push!(nonlegato_pooled_rows, merge(condition, (
                distance=result.wasserstein.wasserstein_distance_matrix[1, 2],
                n_trials=length(result.selected_trials),
            )))
        catch exception
            push!(nonlegato_pooled_rows,
                  merge(condition, (distance=NaN, n_trials=0)))
            push!(failures, (
                stage=:nonlegato_pooled,
                subject_id="",
                task=condition.task,
                type=condition.typ,
                error=sprint(showerror, exception),
            ))
        end
    end

    same_note_subject_rows = NamedTuple[]
    same_note_pooled_rows = NamedTuple[]
    for typ in (:Model, :Real)
        subject_ids = sort(unique(
            trial.subject_id for trial in trials
            if trial.type == typ &&
               (isnothing(exclude_subject_id) ||
                trial.subject_id != exclude_subject_id)))
        for subject_id in subject_ids
            asc = try
                nonlegato_point_sets_from_trials(
                    trials;
                    task=:NonlegatoAsc,
                    typ,
                    subject_id,
                    only_success=true,
                    exclude_subject_id=nothing,
                    trim_start_ms=trim_ms,
                    trim_end_ms=trim_ms,
                    space=:physical,
                    pressure_calib=pressure_from_adc,
                    force_calib=force_newton_from_adc,
                    inner_pad,
                    edge_exclusion_frac,
                    pool=true,
                )
            catch
                nothing
            end
            desc = try
                nonlegato_point_sets_from_trials(
                    trials;
                    task=:NonlegatoDesc,
                    typ,
                    subject_id,
                    only_success=true,
                    exclude_subject_id=nothing,
                    trim_start_ms=trim_ms,
                    trim_end_ms=trim_ms,
                    space=:physical,
                    pressure_calib=pressure_from_adc,
                    force_calib=force_newton_from_adc,
                    inner_pad,
                    edge_exclusion_frac,
                    pool=true,
                )
            catch
                nothing
            end
            isnothing(asc) && continue
            isnothing(desc) && continue
            for note in (:low, :high)
                asc_points = _paper_finite_points(getfield(asc.pooled, note))
                desc_points = _paper_finite_points(getfield(desc.pooled, note))
                size(asc_points, 1) > 3 || continue
                size(desc_points, 1) > 3 || continue
                try
                    result = distance(
                        [asc_points, desc_points]; labels=["Asc", "Desc"])
                    push!(same_note_subject_rows, (
                        type=typ,
                        note,
                        subject_id,
                        distance=result.wasserstein_distance_matrix[1, 2],
                    ))
                catch exception
                    push!(failures, (
                        stage=:same_note_subject,
                        subject_id,
                        task=note,
                        type=typ,
                        error=sprint(showerror, exception),
                    ))
                end
            end
        end

        asc_pool = try
            nonlegato_point_sets_from_trials(
                trials; task=:NonlegatoAsc, typ, only_success=true,
                exclude_subject_id, trim_start_ms=trim_ms,
                trim_end_ms=trim_ms, space=:physical,
                pressure_calib=pressure_from_adc,
                force_calib=force_newton_from_adc,
                inner_pad, edge_exclusion_frac, pool=true)
        catch
            nothing
        end
        desc_pool = try
            nonlegato_point_sets_from_trials(
                trials; task=:NonlegatoDesc, typ, only_success=true,
                exclude_subject_id, trim_start_ms=trim_ms,
                trim_end_ms=trim_ms, space=:physical,
                pressure_calib=pressure_from_adc,
                force_calib=force_newton_from_adc,
                inner_pad, edge_exclusion_frac, pool=true)
        catch
            nothing
        end
        for note in (:low, :high)
            if isnothing(asc_pool) || isnothing(desc_pool)
                push!(same_note_pooled_rows,
                      (type=typ, note, distance=NaN))
                continue
            end
            asc_points = _paper_finite_points(getfield(asc_pool.pooled, note))
            desc_points = _paper_finite_points(getfield(desc_pool.pooled, note))
            value = if size(asc_points, 1) > 3 && size(desc_points, 1) > 3
                try
                    distance([asc_points, desc_points];
                             labels=["Asc", "Desc"]).wasserstein_distance_matrix[1, 2]
                catch
                    NaN
                end
            else
                NaN
            end
            push!(same_note_pooled_rows, (type=typ, note, distance=value))
        end
    end
    same_note_subject_summary = NamedTuple[]
    for typ in (:Model, :Real), note in (:low, :high)
        group = filter(row -> row.type == typ && row.note == note,
                       same_note_subject_rows)
        stats = _paper_mean_sd_n(row.distance for row in group)
        label = string(note) * " " *
                (typ == :Model ? "Model" : "Instrument")
        push!(same_note_subject_summary,
              (; type=typ, note, label, stats...))
    end

    multiphonic_chunks = Dict{
        Tuple{Symbol,String},Vector{Matrix{Float64}}}()
    selected_multiphonic = filter(trials) do trial
        trial.success &&
        (isnothing(exclude_subject_id) ||
         trial.subject_id != exclude_subject_id) &&
        trial.task == :Overtone && trial.type in (:Model, :Real)
    end
    for trial in selected_multiphonic
        try
            points = _overtone_interval_points(
                trial;
                trim_start_ms=trim_ms,
                trim_end_ms=trim_ms,
                axis_scaling=:physical,
            )
            push!(get!(multiphonic_chunks,
                       (trial.type, trial.subject_id), Matrix{Float64}[]),
                  points)
        catch exception
            push!(failures, (
                stage=:multiphonic_points,
                subject_id=trial.subject_id,
                task=trial.task,
                type=trial.type,
                error=sprint(showerror, exception),
            ))
        end
    end
    multiphonic_rows = NamedTuple[]
    for condition in condition_specs
        subject_ids = sort(unique(
            trial.subject_id for trial in trials
            if trial.type == condition.typ &&
               (isnothing(exclude_subject_id) ||
                trial.subject_id != exclude_subject_id)))
        for subject_id in subject_ids
            chunks = get(multiphonic_chunks,
                         (condition.typ, subject_id), Matrix{Float64}[])
            multiphonic_points = isempty(chunks) ?
                zeros(0, 2) : _paper_finite_points(reduce(vcat, chunks))
            nonlegato = try
                nonlegato_point_sets_from_trials(
                    trials;
                    task=condition.task,
                    typ=condition.typ,
                    subject_id,
                    only_success=true,
                    exclude_subject_id=nothing,
                    trim_start_ms=trim_ms,
                    trim_end_ms=trim_ms,
                    space=:physical,
                    pressure_calib=pressure_from_adc,
                    force_calib=force_newton_from_adc,
                    inner_pad,
                    edge_exclusion_frac,
                    pool=true,
                )
            catch
                nothing
            end
            isnothing(nonlegato) && continue
            for note in (:low, :high)
                note_points = _paper_finite_points(
                    getfield(nonlegato.pooled, note))
                size(multiphonic_points, 1) > 3 || continue
                size(note_points, 1) > 3 || continue
                try
                    result = distance(
                        [multiphonic_points, note_points];
                        labels=["Multiphonic", string(note)])
                    push!(multiphonic_rows, (
                        task=condition.task,
                        type=condition.typ,
                        note,
                        subject_id,
                        distance=result.wasserstein_distance_matrix[1, 2],
                    ))
                catch exception
                    push!(failures, (
                        stage=:multiphonic_distance,
                        subject_id,
                        task=condition.task,
                        type=condition.typ,
                        error=sprint(showerror, exception),
                    ))
                end
            end
        end
    end
    multiphonic_summary = NamedTuple[]
    for condition in condition_specs, note in (:low, :high)
        group = filter(row ->
            row.task == condition.task && row.type == condition.typ &&
            row.note == note,
            multiphonic_rows)
        stats = _paper_mean_sd_n(row.distance for row in group)
        label = string(condition.task == :NonlegatoAsc ? "Asc" : "Desc") *
                " " * string(note) * " " *
                (condition.typ == :Model ? "Model" : "Inst")
        push!(multiphonic_summary,
              (; task=condition.task, typ=condition.typ, note, label,
               stats...))
    end

    legato_rows = NamedTuple[]
    selected_legato = filter(trials) do trial
        trial.success &&
        (isnothing(exclude_subject_id) ||
         trial.subject_id != exclude_subject_id) &&
        trial.task in (:LegatoAsc, :LegatoDesc) &&
        trial.type in (:Model, :Real)
    end
    for trial in selected_legato
        try
            points = _legato_note_points_from_trial(
                trial;
                space=:physical,
                pressure_calib=pressure_from_adc,
                force_calib=force_newton_from_adc,
                inner_pad,
            )
            result = distance([points.low, points.high];
                              labels=["low", "high"])
            push!(legato_rows, (
                task=trial.task,
                type=trial.type,
                subject_id=trial.subject_id,
                block=trial.block,
                take=trial.take,
                distance=result.wasserstein_distance_matrix[1, 2],
            ))
        catch exception
            push!(failures, (
                stage=:legato_trial,
                subject_id=trial.subject_id,
                task=trial.task,
                type=trial.type,
                error=sprint(showerror, exception),
            ))
        end
    end
    legato_summary = NamedTuple[]
    for condition in legato_specs
        group = filter(row ->
            row.task == condition.task && row.type == condition.typ,
            legato_rows)
        stats = _paper_mean_sd_n(row.distance for row in group)
        push!(legato_summary, merge(condition, stats))
    end
    legato_pooled_rows = NamedTuple[]
    for condition in legato_specs
        condition_trials = filter(trials) do trial
            trial.success &&
            (isnothing(exclude_subject_id) ||
             trial.subject_id != exclude_subject_id) &&
            trial.task == condition.task && trial.type == condition.typ
        end
        low_chunks = Matrix{Float64}[]
        high_chunks = Matrix{Float64}[]
        for trial in condition_trials
            try
                points = _legato_note_points_from_trial(
                    trial;
                    space=:physical,
                    pressure_calib=pressure_from_adc,
                    force_calib=force_newton_from_adc,
                    inner_pad,
                )
                push!(low_chunks, points.low)
                push!(high_chunks, points.high)
            catch
            end
        end
        low_pool = isempty(low_chunks) ? zeros(0, 2) : reduce(vcat, low_chunks)
        high_pool = isempty(high_chunks) ? zeros(0, 2) : reduce(vcat, high_chunks)
        value = if size(low_pool, 1) > 3 && size(high_pool, 1) > 3
            distance([low_pool, high_pool];
                     labels=["low", "high"]).wasserstein_distance_matrix[1, 2]
        else
            NaN
        end
        push!(legato_pooled_rows, merge(condition, (
            distance=value,
            n_trials=length(condition_trials),
        )))
    end

    settings = (;
        exclude_subject_id,
        trim_ms=Float64(trim_ms),
        inner_pad=Float64(inner_pad),
        edge_exclusion_frac=Float64(edge_exclusion_frac),
        bandwidth=Float64(bandwidth),
        buffer_size=Float64(buffer_size),
        grid_spec,
        method,
        sinkhorn_reg=Float64(sinkhorn_reg),
        sinkhorn_maxiter,
        sinkhorn_tol=Float64(sinkhorn_tol),
    )
    return (;
        settings,
        scale_info,
        nonlegato_trial=(rows=nonlegato_trial_rows,
                         summary=nonlegato_trial_summary),
        nonlegato_pooled=(rows=nonlegato_pooled_rows,),
        same_note=(subject_rows=same_note_subject_rows,
                   subject_summary=same_note_subject_summary,
                   pooled_rows=same_note_pooled_rows),
        multiphonic=(rows=multiphonic_rows,
                     summary=multiphonic_summary),
        legato=(rows=legato_rows, summary=legato_summary,
                pooled_rows=legato_pooled_rows),
        failures,
    )
end

function _paper_distance_label(row)
    direction = hasproperty(row, :task) ?
        (occursin("Asc", string(row.task)) ? "A" : "D") : ""
    typ = hasproperty(row, :typ) ? row.typ :
          (hasproperty(row, :type) ? row.type : :unknown)
    type_code = typ == :Model ? "M" : "R"
    if hasproperty(row, :note) && hasproperty(row, :task)
        return uppercase(string(row.note)) * "\n" * direction * type_code
    elseif hasproperty(row, :note)
        return uppercase(string(row.note)) * " " * type_code
    elseif hasproperty(row, :task)
        return direction * type_code
    end
    return hasproperty(row, :label) ? String(row.label) : ""
end

function _paper_distance_colors(rows)
    return [begin
        typ = hasproperty(row, :typ) ? row.typ :
              (hasproperty(row, :type) ? row.type : :unknown)
        typ == :Model ? :steelblue : :coral
    end for row in rows]
end

function _paper_distance_bar(rows; title, show_ylabel::Bool=false)
    bar(
        1:length(rows), [Float64(row.mean) for row in rows];
        yerror=[Float64(row.sd) for row in rows],
        xticks=(1:length(rows), _paper_distance_label.(rows)),
        legend=false,
        title,
        ylabel=show_ylabel ? "W distance" : "",
        color=_paper_distance_colors(rows),
        ylims=(0.0, 3.0),
        bar_width=0.6,
        linewidth=2,
        grid=:y,
        gridalpha=0.3,
        framestyle=:box,
        titlefontsize=26,
        tickfontsize=18,
        guidefontsize=20,
        bottom_margin=9mm,
        left_margin=show_ylabel ? 11mm : 4mm,
        top_margin=4mm,
        right_margin=3mm,
    )
end

"""Build the seven-panel Wasserstein summary used as paper Figure 5."""
function plot_paper_wasserstein_figure(summary; size=(1400, 1200))
    nonlegato_pooled = [(
        task=row.task, typ=row.typ, mean=row.distance, sd=0.0)
        for row in summary.nonlegato_pooled.rows]
    legato_pooled = [(
        task=row.task, typ=row.typ, mean=row.distance, sd=0.0)
        for row in summary.legato.pooled_rows]
    multiphonic_low = [(
        task=row.task, typ=row.typ, mean=row.mean, sd=row.sd)
        for row in summary.multiphonic.summary if row.note == :low]
    multiphonic_high = [(
        task=row.task, typ=row.typ, mean=row.mean, sd=row.sd)
        for row in summary.multiphonic.summary if row.note == :high]

    panels = (
        _paper_distance_bar(summary.nonlegato_trial.summary;
            title="Nonlegato low-high (trial)", show_ylabel=true),
        _paper_distance_bar(nonlegato_pooled;
            title="Nonlegato low-high (pooled)"),
        _paper_distance_bar(summary.legato.summary;
            title="Legato low-high (trial)", show_ylabel=true),
        _paper_distance_bar(legato_pooled;
            title="Legato low-high (pooled)"),
        _paper_distance_bar(multiphonic_low;
            title="Multiphonic vs nonlegato low", show_ylabel=true),
        _paper_distance_bar(multiphonic_high;
            title="Multiphonic vs nonlegato high"),
        _paper_distance_bar(summary.same_note.subject_summary;
            title="Same note asc-desc", show_ylabel=true),
        plot([NaN], [NaN]; legend=false, framestyle=:none, grid=false,
             xticks=([], []), yticks=([], [])),
    )
    return plot(panels...; layout=(4, 2), size, plot_title="",
                plot_titlefontsize=24)
end
