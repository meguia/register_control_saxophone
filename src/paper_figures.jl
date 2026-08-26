"""Functions that assemble paper Figures 3 and 4 from processed trials."""

function _paper_task_title(task::Symbol)
    task == :NonlegatoAsc && return "Non-legato ascending"
    task == :NonlegatoDesc && return "Non-legato descending"
    task == :LegatoAsc && return "Legato ascending"
    task == :LegatoDesc && return "Legato descending"
    task == :Overtone && return "Multiphonic"
    return String(task)
end

function _paper_trial_panel(trials, subjects, task, condition;
        show_xlabel::Bool=true,
        show_ylabel::Bool=true,
        title_text=nothing,
        highlight=("13", 1))
    axis = plot_task_pf_mode2(
        trials,
        task,
        subjects,
        condition;
        ms=10,
        markeralpha=0.02,
        lw=1,
        linealpha=0.6,
        line_darkening=0.6,
        up_ms=4,
        down_ms=4,
        spline_nsamp=40,
        display=true,
        axis_scaling=:physical,
        draw_gray=false,
        hl_lw=3.5,
        hl_ms=6.0,
        hl_linealpha=1.0,
        trim_start_ms=100.0,
        trim_end_ms=100.0,
        highlight,
        tickfontsize=18,
        titlefontsize=24,
        labelfontsize=22,
        xlims=(0.0, 2.0),
        ylims=(0.0, 8.0),
    )
    plot!(axis;
        title=isnothing(title_text) ? _paper_task_title(task) : title_text,
        xlabel=show_xlabel ? "Pressure (kPa)" : "",
        ylabel=show_ylabel ? "Force (N)" : "",
    )
    return axis
end

function _paper_blank_trial_panel()
    return plot(
        [NaN], [NaN]; legend=false, framestyle=:none, grid=false,
        xticks=([], []), yticks=([], []),
        xlims=(0.0, 2.0), ylims=(0.0, 8.0))
end

"""
Assemble Figure 3 (`condition=:Model`) or Figure 4 (`condition=:Real`).
"""
function plot_paper_trial_figure(trials;
        condition::Symbol,
        subjects=RT_SAX_PAPER_SUBJECTS,
        size=(1200, 1500))
    condition in (:Model, :Real) ||
        error("condition must be :Model or :Real")
    condition_label = condition == :Model ? "Model" : "Instrument"
    panels = (
        _paper_trial_panel(trials, subjects, :NonlegatoAsc,
            condition_label; show_xlabel=false, show_ylabel=true),
        _paper_trial_panel(trials, subjects, :NonlegatoDesc,
            condition_label; show_xlabel=false, show_ylabel=false),
        _paper_trial_panel(trials, subjects, :LegatoAsc,
            condition_label; show_xlabel=false, show_ylabel=true),
        _paper_trial_panel(trials, subjects, :LegatoDesc,
            condition_label; show_xlabel=false, show_ylabel=false),
        _paper_trial_panel(trials, subjects, :Overtone,
            condition_label; show_xlabel=true, show_ylabel=true,
            title_text="Multiphonic"),
        _paper_blank_trial_panel(),
    )
    return plot(panels...; layout=(3, 2), margin=6mm, size)
end
