# Fixed-zeta periodic-orbit amplitude diagrams for comparison with Colinot's
# harmonic-balance Figures 3.26 and 3.27. The calculation is deliberately
# isolated from the two-parameter continuation caches: it follows complete
# periodic branches in gamma and therefore retains stability and amplitude
# information that a curve in the (gamma, zeta) plane does not contain.

Base.@kwdef struct SaxFixedZetaAmplitudeSettings
    nmodes::Int = 8
    zeta::Float64 = 0.6
    modes::Tuple{Vararg{Int}} = (1, 2)
    hopf_gamma_hints::Tuple{Vararg{Float64}} = (0.40, 0.43)
    gamma_range::Tuple{Float64,Float64} = (0.30, 2.00)
    contact_stiffness::Float64 = 100.0
    collocation_intervals::Int = 40
    collocation_degree::Int = 4
    po_ds::Float64 = 1e-3
    po_dsmax::Float64 = 6e-3
    po_max_steps::Int = 900
    po_save_sol_every_step::Int = 5
    hopf_scan_points::Int = 681
    include_period_two::Bool = true
    p2_ds::Float64 = 5e-4
    p2_dsmax::Float64 = 3e-3
    p2_max_steps::Int = 700
    floquet_solver::Symbol = :periodic_schur
    newton_tol::Float64 = 1e-10
    stability_tol::Float64 = 1e-8
end

"""Resolution presets for the fixed-zeta amplitude comparison."""
function sax_fixed_zeta_amplitude_settings(profile::Symbol=:final; kwargs...)
    profile in (:smoke, :pilot, :final) || throw(ArgumentError(
        "profile must be :smoke, :pilot, or :final"))
    resolution = if profile == :smoke
        (
            gamma_range=(0.35, 0.55),
            collocation_intervals=8,
            collocation_degree=2,
            po_ds=4e-3,
            po_dsmax=1e-2,
            po_max_steps=35,
            po_save_sol_every_step=1,
            hopf_scan_points=81,
            include_period_two=false,
            p2_max_steps=10,
        )
    elseif profile == :pilot
        (
            gamma_range=(0.30, 1.00),
            collocation_intervals=25,
            collocation_degree=3,
            po_ds=2e-3,
            po_dsmax=8e-3,
            po_max_steps=450,
            po_save_sol_every_step=5,
            hopf_scan_points=281,
            include_period_two=true,
            p2_max_steps=350,
        )
    else
        NamedTuple()
    end
    settings = SaxFixedZetaAmplitudeSettings(; merge(resolution, (; kwargs...))...)
    return _validate_sax_fixed_zeta_amplitude_settings(settings)
end

function _validate_sax_fixed_zeta_amplitude_settings(
        settings::SaxFixedZetaAmplitudeSettings)
    settings.nmodes > 0 || throw(ArgumentError("nmodes must be positive"))
    0 < settings.zeta || throw(ArgumentError("zeta must be positive"))
    settings.gamma_range[1] < settings.gamma_range[2] ||
        throw(ArgumentError("gamma_range must be increasing"))
    !isempty(settings.modes) || throw(ArgumentError("modes cannot be empty"))
    all(mode -> 1 <= mode <= settings.nmodes, settings.modes) ||
        throw(ArgumentError("every mode must lie in 1:$(settings.nmodes)"))
    length(unique(settings.modes)) == length(settings.modes) ||
        throw(ArgumentError("modes must be unique"))
    length(settings.hopf_gamma_hints) == length(settings.modes) ||
        throw(ArgumentError(
            "hopf_gamma_hints must contain one value for every selected mode"))
    all(hint -> settings.gamma_range[1] <= hint <= settings.gamma_range[2],
        settings.hopf_gamma_hints) || throw(ArgumentError(
            "every Hopf gamma hint must lie inside gamma_range"))
    settings.contact_stiffness >= 0 ||
        throw(ArgumentError("contact_stiffness must be nonnegative"))
    settings.collocation_intervals >= 5 ||
        throw(ArgumentError("at least five collocation intervals are required"))
    settings.collocation_degree >= 2 ||
        throw(ArgumentError("collocation_degree must be at least two"))
    settings.po_max_steps > 0 || throw(ArgumentError("po_max_steps must be positive"))
    settings.p2_max_steps > 0 || throw(ArgumentError("p2_max_steps must be positive"))
    settings.floquet_solver == :periodic_schur || throw(ArgumentError(
        "the canonical fixed-zeta amplitude run requires :periodic_schur",
    ))
    settings.hopf_scan_points >= 21 ||
        throw(ArgumentError("hopf_scan_points must be at least 21"))
    return settings
end

function _portable_sax_fixed_zeta_amplitude_settings(
        settings::SaxFixedZetaAmplitudeSettings)
    names = fieldnames(SaxFixedZetaAmplitudeSettings)
    return NamedTuple{names}(Tuple(getfield(settings, name) for name in names))
end

function _sax_fixed_zeta_tag(value::Real; digits::Int=3)
    return replace(@sprintf("%.*f", digits, float(value)), "." => "p", "-" => "m")
end

"""Return the isolated directory and component paths for one amplitude run."""
function sax_fixed_zeta_amplitude_paths(
        root::AbstractString;
        settings::SaxFixedZetaAmplitudeSettings=SaxFixedZetaAmplitudeSettings())
    _validate_sax_fixed_zeta_amplitude_settings(settings)
    zeta_tag = _sax_fixed_zeta_tag(settings.zeta)
    stiffness_tag = _sax_fixed_zeta_tag(settings.contact_stiffness; digits=1)
    directory = joinpath(
        root, "fixed_zeta_amplitude_z$(zeta_tag)_kc$(stiffness_tag)")
    return (
        directory=directory,
        p1=mode -> joinpath(directory, "p1_mode$(Int(mode)).jld2"),
        p2=key -> joinpath(directory, "p2_$(key).jld2"),
        p2_checkpoints=key -> joinpath(directory, "p2_$(key)_checkpoints.jld2"),
    )
end

const SAX_FIXED_ZETA_AMPLITUDE_CACHE_SCHEMA_VERSION = 2

function _sax_fixed_zeta_floquet_solver(
        settings::SaxFixedZetaAmplitudeSettings)
    settings.floquet_solver == :periodic_schur || throw(ArgumentError(
        "unsupported fixed-zeta Floquet solver $(settings.floquet_solver)",
    ))
    return SaxFloquetPQZ(
        cyclic_retries=8,
        fallback_to_floquet_coll=false,
    )
end

function _save_sax_fixed_zeta_amplitude_cache(
        path::AbstractString,
        kind::Symbol,
        payload,
        model_p::NamedTuple,
        settings::SaxFixedZetaAmplitudeSettings)
    cache = (
        schema_version=SAX_FIXED_ZETA_AMPLITUDE_CACHE_SCHEMA_VERSION,
        analysis=:fixed_zeta_periodic_amplitude,
        kind=kind,
        settings_signature=_portable_sax_fixed_zeta_amplitude_settings(settings),
        model_signature=_sax_bifurcation_model_signature(model_p, settings.nmodes),
        saved_at_unix=time(),
        payload=payload,
    )
    _atomic_jld2_save(path; cache)
    return payload
end

function _load_sax_fixed_zeta_amplitude_cache(
        path::AbstractString,
        kind::Symbol,
        model_p::NamedTuple,
        settings::SaxFixedZetaAmplitudeSettings)
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
            stored.schema_version == SAX_FIXED_ZETA_AMPLITUDE_CACHE_SCHEMA_VERSION =>
                "cache schema changed",
            stored.analysis == :fixed_zeta_periodic_amplitude =>
                "analysis kind changed",
            stored.kind == kind => "component kind changed",
            isequal(stored.settings_signature,
                    _portable_sax_fixed_zeta_amplitude_settings(settings)) =>
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
    if kind == :p1
        mode = Int(stored.payload.branch.mode)
        mode_index = findfirst(==(mode), settings.modes)
        isnothing(mode_index) && return (
            status=:incompatible,
            payload=nothing,
            reason="stored P1 mode is not requested",
        )
        expected_gamma = settings.hopf_gamma_hints[mode_index]
        stored_gamma = float(stored.payload.branch.hopf.gamma)
        abs(stored_gamma - expected_gamma) <= 0.25 || return (
            status=:incompatible,
            payload=nothing,
            reason="stored P1 branch uses the wrong Hopf crossing",
        )
        all(seed -> seed.type == :pd, stored.payload.pd_seeds) || return (
            status=:incompatible,
            payload=nothing,
            reason="stored P1 cache contains non-PD branch-switch seeds",
        )
    end
    return (status=:valid, payload=stored.payload, reason="")
end

function _sax_fixed_zeta_model(
        model_p::NamedTuple,
        settings::SaxFixedZetaAmplitudeSettings)
    return merge(model_p, (contact_stiffness=settings.contact_stiffness,))
end

function _sax_fixed_zeta_bifurcation_settings(
        settings::SaxFixedZetaAmplitudeSettings)
    zeta_half_width = max(1e-6, abs(settings.zeta) * 1e-6)
    return sax_bifurcation_settings(
        :final;
        nmodes=settings.nmodes,
        gamma_range=settings.gamma_range,
        zeta_range=(settings.zeta - zeta_half_width,
                    settings.zeta + zeta_half_width),
        zeta_seeds=(settings.zeta,),
        po_collocation_intervals=settings.collocation_intervals,
        po_collocation_degree=settings.collocation_degree,
        po_linear_solver=:condensed,
        po_ds=settings.po_ds,
        po_dsmax=settings.po_dsmax,
        po_max_steps=settings.po_max_steps,
        po_save_sol_every_step=settings.po_save_sol_every_step,
        newton_tol=settings.newton_tol,
        stability_tol=settings.stability_tol,
    )
end

function _sax_fixed_zeta_hopf_settings(
        settings::SaxFixedZetaAmplitudeSettings)
    half_width = max(1e-6, abs(settings.zeta) * 1e-6)
    return SaxPDRescueSettings(
        nmodes=settings.nmodes,
        gamma_range=settings.gamma_range,
        zeta_range=(settings.zeta - half_width, settings.zeta + half_width),
        seed_zetas=(settings.zeta,),
        po_collocation_intervals=settings.collocation_intervals,
        po_collocation_degree=settings.collocation_degree,
        po_ds=settings.po_ds,
        po_dsmax=settings.po_dsmax,
        po_max_steps=settings.po_max_steps,
        po_save_sol_every_step=settings.po_save_sol_every_step,
        newton_tol=settings.newton_tol,
        stability_tol=settings.stability_tol,
        hopf_scan_points=settings.hopf_scan_points,
    )
end

function _sax_fixed_zeta_specialpoints(branch)
    points = Any[]
    for special in branch.specialpoint
        special.type in (:fold, :pd, :ns) || continue
        special.status == :converged || continue
        index = clamp(Int(special.idx), 1, length(branch))
        push!(points, (
            type=special.type,
            status=special.status,
            gamma=float(special.param),
            pressure_l2=float(branch.branch.pressure_l2[index]),
            period=float(branch.branch.period[index]),
            stable=branch.branch.stable[index] === true,
            n_unstable=Int(branch.branch.n_unstable[index]),
            precision=float(special.precision),
            index=index,
            step=Int(special.step),
        ))
    end
    return points
end

function _sax_fixed_zeta_portable_p1(branch, hopf, mode::Integer)
    stable = map(value -> value === true, branch.branch.stable)
    return (
        kind=:p1,
        mode=Int(mode),
        gamma=collect(float.(branch.branch.gamma)),
        zeta=fill(float(hopf.zeta), length(branch)),
        pressure_l2=collect(float.(branch.branch.pressure_l2)),
        pressure_amplitude=collect(float.(branch.branch.pressure_amplitude)),
        period=collect(float.(branch.branch.period)),
        stable=collect(stable),
        n_unstable=collect(Int.(branch.branch.n_unstable)),
        step=collect(Int.(branch.branch.step)),
        hopf=(gamma=float(hopf.gamma), zeta=float(hopf.zeta),
              mode=Int(mode), frequency=float(hopf.frequency)),
        specialpoints=_sax_fixed_zeta_specialpoints(branch),
        diagnostics=_sax_branch_terminal_diagnostics(branch, :gamma),
    )
end

function _sax_fixed_zeta_p2_key(seed)
    gamma_tag = replace(@sprintf("%.9f", float(seed.gamma)), "." => "p", "-" => "m")
    return "m$(Int(seed.mode))_g$(gamma_tag)"
end

function _sax_fixed_zeta_period_two_settings(
        seed,
        settings::SaxFixedZetaAmplitudeSettings)
    return SaxPeriodTwoSettings(
        nmodes=settings.nmodes,
        mode=Int(seed.mode),
        gamma_hint=float(seed.gamma),
        gamma_range=settings.gamma_range,
        collocation_intervals=settings.collocation_intervals,
        collocation_degree=settings.collocation_degree,
        ds=settings.p2_ds,
        dsmax=settings.p2_dsmax,
        max_steps=settings.p2_max_steps,
        save_sol_every_step=settings.po_save_sol_every_step,
        checkpoint_every=5,
        progress_every=5,
        newton_tol=settings.newton_tol,
        stability_tol=settings.stability_tol,
    )
end

function _sax_fixed_zeta_portable_p2(run, seed)
    curve = run.curve
    specialpoints = map(run.specialpoints) do point
        index = clamp(Int(point.index), 1, length(curve.gamma))
        merge(point, (
            pressure_l2=float(curve.pressure_l2[index]),
            period=float(curve.period[index]),
            stable=curve.stable[index] === true,
            n_unstable=Int(curve.n_unstable[index]),
        ))
    end
    return merge(curve, (
        specialpoints=collect(specialpoints),
        parent_pd=(
            gamma=float(seed.gamma),
            zeta=float(seed.zeta),
            mode=Int(seed.mode),
            localization_precision=float(seed.localization_precision),
        ),
        initial_period_ratio=float(run.initial_period_ratio),
        diagnostics=run.curve.diagnostics,
    ))
end

function _sax_unique_fixed_zeta_pd_seeds(branch_payloads)
    seeds = Any[]
    for payload in branch_payloads, candidate in payload.pd_seeds
        candidate.type == :pd || continue
        duplicate = any(seed ->
            seed.mode == candidate.mode &&
            abs(seed.gamma - candidate.gamma) <= 1e-4,
            seeds,
        )
        duplicate || push!(seeds, candidate)
    end
    return seeds
end

"""
    load_sax_fixed_zeta_amplitude_progress(model, root; settings)

Load only atomically committed P1 and P2 components. The returned partial
result is safe to plot while another Julia process is continuing a branch.
"""
function load_sax_fixed_zeta_amplitude_progress(
        model_p::NamedTuple,
        root::AbstractString;
        settings::SaxFixedZetaAmplitudeSettings=SaxFixedZetaAmplitudeSettings())
    _validate_sax_fixed_zeta_amplitude_settings(settings)
    model = _sax_fixed_zeta_model(model_p, settings)
    paths = sax_fixed_zeta_amplitude_paths(root; settings=settings)
    p1_payloads = Any[]
    p1_status = Any[]
    for mode in settings.modes
        loaded = _load_sax_fixed_zeta_amplitude_cache(
            paths.p1(mode), :p1, model, settings)
        push!(p1_status, (mode=Int(mode), status=loaded.status,
                          reason=loaded.reason, path=paths.p1(mode)))
        loaded.status == :valid && push!(p1_payloads, loaded.payload)
    end
    seeds = _sax_unique_fixed_zeta_pd_seeds(p1_payloads)
    p2_branches = Any[]
    p2_status = Any[]
    if settings.include_period_two
        for seed in seeds
            key = _sax_fixed_zeta_p2_key(seed)
            loaded = _load_sax_fixed_zeta_amplitude_cache(
                paths.p2(key), :p2, model, settings)
            push!(p2_status, (key=key, mode=Int(seed.mode),
                              gamma=float(seed.gamma), status=loaded.status,
                              reason=loaded.reason, path=paths.p2(key)))
            loaded.status == :valid && push!(p2_branches, loaded.payload)
        end
    end
    p1_complete = length(p1_payloads) == length(settings.modes)
    p2_complete = !settings.include_period_two ||
        length(p2_branches) == length(seeds)
    status = p1_complete && p2_complete ? :complete :
             isempty(p1_payloads) ? :missing : :partial
    return (
        analysis=:fixed_zeta_periodic_amplitude,
        status=status,
        settings=_portable_sax_fixed_zeta_amplitude_settings(settings),
        regularization=hasproperty(model, :sax_regularization) ?
            model.sax_regularization : nothing,
        paths=paths,
        p1_branches=[payload.branch for payload in p1_payloads],
        p2_branches=p2_branches,
        hopf_points=[payload.branch.hopf for payload in p1_payloads],
        pd_seeds=seeds,
        p1_status=p1_status,
        p2_status=p2_status,
        counts=(
            p1=length(p1_payloads),
            expected_p1=length(settings.modes),
            p2=length(p2_branches),
            expected_p2=settings.include_period_two ? length(seeds) : 0,
        ),
    )
end

"""
    sax_fixed_zeta_amplitude_events(progress; gamma_limits=(0.30, 1.00))

Return a deduplicated, gamma-ordered table of Hopf and periodic-orbit
bifurcations in a selected plotting interval. This is also a consistency
diagnostic for the figure: the event coordinates are read from the same
portable branches that provide the plotted L2 amplitudes. Unclassified branch
points are omitted rather than being guessed as NS events.
"""
function sax_fixed_zeta_amplitude_events(
        progress;
        gamma_limits::Tuple{<:Real,<:Real}=(0.30, 1.00),
        gamma_tolerance::Real=1e-4,
        amplitude_tolerance::Real=2e-3)
    lower, upper = float.(gamma_limits)
    rows = Any[]
    for hopf in progress.hopf_points
        lower <= hopf.gamma <= upper || continue
        push!(rows, (
            type=:hopf,
            label="H$(Int(hopf.mode))",
            orbit=:equilibrium,
            mode=Int(hopf.mode),
            gamma=float(hopf.gamma),
            pressure_l2=0.0,
            stable=missing,
        ))
    end
    for branch in Any[progress.p1_branches...; progress.p2_branches...]
        for point in branch.specialpoints
            lower <= point.gamma <= upper || continue
            point.type in (:fold, :pd, :ns) || continue
            label = point.type == :fold ? "F" :
                    point.type == :pd ? "PD" : "NS"
            candidate = (
                type=point.type,
                label=label,
                orbit=branch.kind,
                mode=Int(branch.mode),
                gamma=float(point.gamma),
                pressure_l2=float(point.pressure_l2),
                stable=point.stable === true,
            )
            duplicate = any(row ->
                row.type == candidate.type &&
                row.mode == candidate.mode &&
                abs(row.gamma - candidate.gamma) <= gamma_tolerance &&
                abs(row.pressure_l2 - candidate.pressure_l2) <=
                    amplitude_tolerance,
                rows,
            )
            duplicate || push!(rows, candidate)
        end
    end
    sort!(rows; by=row -> (row.gamma, row.mode, String(row.label)))
    return rows
end

"""
    compute_sax_fixed_zeta_amplitude(model, root; settings, resume=true)

At the requested fixed `zeta`, locate the selected modal Hopf points, continue
their P1 periodic-orbit families in `gamma`, and optionally branch-switch to
P2 at every distinct period-doubling point. P1 and P2 stability and event
detection both use strict generalized Periodic Schur, without a FloquetColl
fallback. Each completed component is saved atomically before the next one
starts.
"""
function compute_sax_fixed_zeta_amplitude(
        model_p::NamedTuple,
        root::AbstractString;
        settings::SaxFixedZetaAmplitudeSettings=SaxFixedZetaAmplitudeSettings(),
        resume::Bool=true,
        verbosity::Integer=1)
    _validate_sax_fixed_zeta_amplitude_settings(settings)
    model = _sax_fixed_zeta_model(model_p, settings)
    paths = sax_fixed_zeta_amplitude_paths(root; settings=settings)
    mkpath(paths.directory)
    bif_settings = _sax_fixed_zeta_bifurcation_settings(settings)
    hopf_settings = _sax_fixed_zeta_hopf_settings(settings)

    for (mode, gamma_hint) in zip(
            settings.modes, settings.hopf_gamma_hints)
        cached = resume ? _load_sax_fixed_zeta_amplitude_cache(
            paths.p1(mode), :p1, model, settings) :
            (status=:missing, payload=nothing, reason="resume disabled")
        cached.status == :valid && continue
        verbosity > 0 && @info(
            "Fixed-zeta P1 amplitude branch started",
            zeta=settings.zeta,
            mode,
            gamma_range=settings.gamma_range,
            collocation="$(settings.collocation_intervals)x$(settings.collocation_degree)",
        )
        hopf = refine_sax_hopf_checkpoint(
            model,
            settings.zeta,
            mode;
            settings=hopf_settings,
            gamma_hint=gamma_hint,
        )
        branch = continue_sax_periodic_orbits(
            hopf, model;
            settings=bif_settings,
            verbosity=Int(verbosity),
            eigsolver=_sax_fixed_zeta_floquet_solver(settings),
        )
        portable = _sax_fixed_zeta_portable_p1(branch, hopf, mode)
        pd_seeds = filter(
            checkpoint -> checkpoint.type == :pd,
            _sax_periodic_bifurcation_checkpoints(branch, hopf, mode),
        )
        payload = (branch=portable, pd_seeds=pd_seeds)
        _save_sax_fixed_zeta_amplitude_cache(
            paths.p1(mode), :p1, payload, model, settings)
        verbosity > 0 && @info(
            "Fixed-zeta P1 amplitude branch completed",
            mode,
            points=length(portable.gamma),
            folds=count(point -> point.type == :fold, portable.specialpoints),
            period_doublings=count(point -> point.type == :pd, portable.specialpoints),
            neimark_sacker=count(point -> point.type == :ns, portable.specialpoints),
        )
    end

    progress = load_sax_fixed_zeta_amplitude_progress(
        model_p, root; settings=settings)
    if settings.include_period_two
        for seed in progress.pd_seeds
            key = _sax_fixed_zeta_p2_key(seed)
            cached = resume ? _load_sax_fixed_zeta_amplitude_cache(
                paths.p2(key), :p2, model, settings) :
                (status=:missing, payload=nothing, reason="resume disabled")
            cached.status == :valid && continue
            verbosity > 0 && @info(
                "Fixed-zeta P2 branch switch started",
                mode=seed.mode,
                gamma=seed.gamma,
                zeta=seed.zeta,
            )
            run = continue_sax_period_two(
                seed, model;
                settings=_sax_fixed_zeta_period_two_settings(seed, settings),
                checkpoint_path=paths.p2_checkpoints(key),
                eigsolver=_sax_fixed_zeta_floquet_solver(settings),
                detect_bifurcation=3,
                save_eigenvectors=true,
                verbosity=Int(verbosity),
            )
            portable = _sax_fixed_zeta_portable_p2(run, seed)
            _save_sax_fixed_zeta_amplitude_cache(
                paths.p2(key), :p2, portable, model, settings)
            verbosity > 0 && @info(
                "Fixed-zeta P2 branch completed",
                mode=seed.mode,
                parent_gamma=seed.gamma,
                points=length(portable.gamma),
            )
        end
    end
    return load_sax_fixed_zeta_amplitude_progress(
        model_p, root; settings=settings)
end

function _sax_plot_amplitude_segments!(
        figure,
        curve;
        stable_color="#D55E00",
        unstable_color="#4D4D4D")
    count = length(curve.gamma)
    count == 0 && return figure
    start = 1
    for stop in 2:(count + 1)
        split = stop == count + 1 || curve.stable[stop] != curve.stable[stop - 1]
        split || continue
        indices = start:(stop - 1)
        is_stable = curve.stable[start]
        plot!(
            figure,
            curve.gamma[indices],
            curve.pressure_l2[indices];
            color=is_stable ? stable_color : unstable_color,
            linewidth=is_stable ? 3.2 : 2.8,
            linestyle=is_stable ? :solid : :dash,
            label="",
        )
        start = stop
    end
    return figure
end

"""
    plot_sax_fixed_zeta_amplitude(progress; gamma_limits=nothing)

Plot Colinot's fixed-zeta observable from collocation continuation. Thick
orange segments are Floquet-stable and dashed dark-gray segments are unstable.
Hopf, fold, period-doubling, and Neimark-Sacker events are shown with markers
rather than repeated text annotations. `gamma_limits` can be `(0.30, 1.0)` for
the Figure 3.27-style moderate-pressure close-up.
"""
function plot_sax_fixed_zeta_amplitude(
        progress;
        gamma_limits::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
        show_bifurcation_labels::Bool=true,
        legend=:topright,
        title::AbstractString="")
    branches = Any[progress.p1_branches...; progress.p2_branches...]
    selected_limits = isnothing(gamma_limits) ?
        Tuple(float.(progress.settings.gamma_range)) :
        Tuple(float.(gamma_limits))
    maximum_amplitude = maximum((maximum(branch.pressure_l2)
                                 for branch in branches); init=1e-6)
    figure = plot(
        xlabel="gamma",
        ylabel="L2 pressure amplitude",
        xlims=selected_limits,
        ylims=(-0.035maximum_amplitude, 1.06maximum_amplitude),
        legend=legend,
        title=title,
        framestyle=:box,
        grid=false,
    )
    isempty(branches) && return figure
    stable_color = progress.settings.contact_stiffness == 0 ?
        "#0072B2" : "#D55E00"
    unstable_color = "#4D4D4D"
    plot!(figure, [NaN], [NaN]; color=stable_color, linewidth=3.2,
          linestyle=:solid, label="Stable periodic orbit")
    plot!(figure, [NaN], [NaN]; color=unstable_color, linewidth=2.8,
          linestyle=:dash, label="Unstable periodic orbit")
    for branch in branches
        _sax_plot_amplitude_segments!(
            figure, branch;
            stable_color=stable_color,
            unstable_color=unstable_color,
        )
    end
    if show_bifurcation_labels
        event_styles = Dict(
            "H1" => (marker=:diamond, color="#D55E00"),
            "H2" => (marker=:diamond, color="#0072B2"),
            "F"  => (marker=:rect, color="#CC79A7"),
            "PD" => (marker=:utriangle, color="#E69F00"),
            "NS" => (marker=:circle, color="#009E73"),
        )
        events = sax_fixed_zeta_amplitude_events(
            progress; gamma_limits=selected_limits)
        for label in ("H1", "H2", "F", "PD", "NS")
            selected = filter(event -> event.label == label, events)
            isempty(selected) && continue
            style = event_styles[label]
            scatter!(
                figure,
                getproperty.(selected, :gamma),
                getproperty.(selected, :pressure_l2);
                marker=style.marker,
                markersize=7,
                markercolor=style.color,
                markerstrokecolor=:black,
                markerstrokewidth=0.8,
                label=label,
            )
        end
    end
    return figure
end
