# Focused, cacheable diagnostics for the unresolved high-gamma transition.
#
# This file is included after the ordinary bifurcation, PD-rescue, and
# transition-refinement layers.  It deliberately writes only to a separate
# transition-mechanism directory.  Existing Final, rescue, sweep, session, and
# processed-experiment caches remain read-only inputs.

const SAX_TRANSITION_MECHANISM_SCHEMA_VERSION = 2
const SAX_TRANSITION_MECHANISM_CHECKPOINT_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

"""Controls for validation and directional continuation of the high-gamma NS seed."""
Base.@kwdef struct SaxHighGammaNSSettings
    nmodes::Int = 8
    mode::Int = 2
    seed_zeta::Float64 = 0.25
    gamma_hint::Float64 = 0.6043477221884095
    zeta_range::Tuple{Float64,Float64} = (0.001, 0.99)
    collocation_intervals::Int = 40
    collocation_degree::Int = 4
    ds::Float64 = 5e-5
    dsmin::Float64 = 1e-10
    dsmax::Float64 = 2e-4
    max_steps::Int = 1200
    save_sol_every_step::Int = 5
    checkpoint_every::Int = 5
    progress_every::Int = 5
    codim2_detection::Int = 2
    newton_tol::Float64 = 1e-10
    newton_max_iterations::Int = 45
    stability_tol::Float64 = 1e-8
    validation_residual_tol::Float64 = 1e-8
    validation_floquet_tol::Float64 = 2e-3
    r2_angle_warning::Float64 = 0.10
    formulations::Tuple{Vararg{Symbol}} = (:minaug, :matrix_based)
end

"""Controls for branch switching from the known PD orbit to the period-two family."""
Base.@kwdef struct SaxPeriodTwoSettings
    nmodes::Int = 8
    mode::Int = 2
    gamma_hint::Float64 = 0.44926096499365253
    gamma_range::Tuple{Float64,Float64} = (0.35, 0.75)
    collocation_intervals::Int = 40
    collocation_degree::Int = 4
    delta_gamma_candidates::Tuple{Vararg{Float64}} =
        (-1e-3, 1e-3, -3e-3, 3e-3)
    amplitude_factors::Tuple{Vararg{Float64}} = (0.05, 0.10, 0.20)
    ds::Float64 = 5e-4
    dsmin::Float64 = 1e-8
    dsmax::Float64 = 3e-3
    max_steps::Int = 350
    save_sol_every_step::Int = 5
    checkpoint_every::Int = 5
    progress_every::Int = 5
    newton_tol::Float64 = 1e-10
    newton_max_iterations::Int = 40
    stability_tol::Float64 = 1e-8
    validation_residual_tol::Float64 = 1e-8
    validation_floquet_tol::Float64 = 2e-3
end

"""Controls for independent fixed-zeta parent-orbit and Floquet slices."""
Base.@kwdef struct SaxFloquetSliceSettings
    nmodes::Int = 8
    mode::Int = 2
    zeta_values::Tuple{Vararg{Float64}} =
        Tuple(collect(range(0.10, 0.90; length=17)))
    gamma_range::Tuple{Float64,Float64} = (0.35, 0.75)
    gamma_grid_points::Int = 161
    collocation_intervals::Int = 40
    collocation_degree::Int = 4
    po_ds::Float64 = 1e-3
    po_dsmax::Float64 = 5e-3
    po_max_steps::Int = 500
    hopf_scan_points::Int = 241
    flip_angle_window::Float64 = 0.20
    newton_tol::Float64 = 1e-10
    stability_tol::Float64 = 1e-8
end

"""
Controls for the mixed-mode escape-time surface.

Every point starts from its equilibrium plus equal pressure-coordinate
perturbations of modes 1 and 2.  `mixedness_threshold` is applied to
`4A1*A2/(A1+A2)^2`.  Escape is accepted only when the signal is active and the
mixedness remains below the threshold for `hold_cycles` mode-1 periods.
"""
Base.@kwdef struct SaxMixedEscapeSettings
    nmodes::Int = 8
    gamma_range::Tuple{Float64,Float64} = (0.46, 0.70)
    zeta_range::Tuple{Float64,Float64} = (0.25, 0.80)
    gamma_points::Int = 61
    zeta_points::Int = 31
    maximum_time::Float64 = 6000.0
    saveat::Float64 = 0.5
    mixed_excitation::Float64 = 1e-2
    activation_threshold::Float64 = 2e-3
    mixedness_threshold::Float64 = 0.55
    hold_cycles::Float64 = 5.0
    reltol::Float64 = 1e-7
    abstol::Float64 = 1e-9
end

function sax_transition_mechanism_settings(profile::Symbol=:final)
    profile in (:smoke, :pilot, :final) || throw(ArgumentError(
        "transition-mechanism profile must be :smoke, :pilot, or :final",
    ))
    if profile == :smoke
        return (
            ns=SaxHighGammaNSSettings(
                max_steps=2, checkpoint_every=1, progress_every=1,
                codim2_detection=0, newton_max_iterations=6,
                formulations=(:matrix_based,),
            ),
            p2=SaxPeriodTwoSettings(max_steps=3, save_sol_every_step=1,
                                    checkpoint_every=1, progress_every=1,
                                    delta_gamma_candidates=(-1e-3,),
                                    amplitude_factors=(0.05,)),
            floquet=SaxFloquetSliceSettings(
                zeta_values=(0.25,), gamma_range=(0.40, 0.68),
                gamma_grid_points=31, collocation_intervals=25,
                collocation_degree=3, po_ds=2e-3, po_dsmax=6e-3,
                po_max_steps=180, hopf_scan_points=101),
            escape=SaxMixedEscapeSettings(
                gamma_range=(0.55, 0.65), zeta_range=(0.50, 0.65),
                gamma_points=3, zeta_points=3, maximum_time=8.0,
                saveat=0.25, hold_cycles=0.25),
        )
    elseif profile == :pilot
        return (
            ns=SaxHighGammaNSSettings(max_steps=250),
            p2=SaxPeriodTwoSettings(max_steps=120),
            floquet=SaxFloquetSliceSettings(
                zeta_values=Tuple(collect(range(0.15, 0.85; length=9))),
                gamma_grid_points=81, collocation_intervals=25,
                collocation_degree=3, po_max_steps=320,
                hopf_scan_points=161),
            escape=SaxMixedEscapeSettings(
                gamma_points=31, zeta_points=17, maximum_time=2000.0),
        )
    end
    return (
        ns=SaxHighGammaNSSettings(),
        p2=SaxPeriodTwoSettings(),
        floquet=SaxFloquetSliceSettings(),
        escape=SaxMixedEscapeSettings(),
    )
end

_portable_sax_mechanism_settings(settings) =
    NamedTuple{fieldnames(typeof(settings))}(
        Tuple(getfield(settings, name) for name in fieldnames(typeof(settings))),
    )

function _validate_sax_high_gamma_ns_settings(settings::SaxHighGammaNSSettings)
    1 <= settings.mode <= settings.nmodes ||
        throw(ArgumentError("NS mode must lie in 1:$(settings.nmodes)"))
    settings.zeta_range[1] <= settings.seed_zeta <= settings.zeta_range[2] ||
        throw(ArgumentError("NS seed_zeta must lie inside zeta_range"))
    0 < settings.dsmin <= abs(settings.ds) <= settings.dsmax ||
        throw(ArgumentError("require 0 < dsmin <= |ds| <= dsmax for NS"))
    settings.max_steps > 0 || throw(ArgumentError("NS max_steps must be positive"))
    settings.codim2_detection in 0:3 ||
        throw(ArgumentError("NS codim2_detection must lie in 0:3"))
    settings.collocation_intervals >= 5 ||
        throw(ArgumentError("NS collocation requires at least five intervals"))
    settings.collocation_degree >= 2 ||
        throw(ArgumentError("NS collocation degree must be at least two"))
    isempty(settings.formulations) &&
        throw(ArgumentError("NS formulations cannot be empty"))
    all(formulation -> formulation in (:minaug, :matrix_based),
        settings.formulations) || throw(ArgumentError(
        "NS formulations must contain only :minaug or :matrix_based"))
    return settings
end

function _validate_sax_period_two_settings(settings::SaxPeriodTwoSettings)
    settings.gamma_range[1] < settings.gamma_range[2] ||
        throw(ArgumentError("P2 gamma_range must be increasing"))
    isempty(settings.delta_gamma_candidates) &&
        throw(ArgumentError("P2 delta_gamma_candidates cannot be empty"))
    isempty(settings.amplitude_factors) &&
        throw(ArgumentError("P2 amplitude_factors cannot be empty"))
    all(value -> !iszero(value), settings.delta_gamma_candidates) ||
        throw(ArgumentError("P2 predictor offsets must be nonzero"))
    all(>(0), settings.amplitude_factors) ||
        throw(ArgumentError("P2 amplitude factors must be positive"))
    0 < settings.dsmin <= abs(settings.ds) <= settings.dsmax ||
        throw(ArgumentError("require 0 < dsmin <= |ds| <= dsmax for P2"))
    return settings
end

function _validate_sax_floquet_slice_settings(settings::SaxFloquetSliceSettings)
    isempty(settings.zeta_values) &&
        throw(ArgumentError("Floquet zeta_values cannot be empty"))
    settings.gamma_grid_points >= 3 ||
        throw(ArgumentError("Floquet gamma_grid_points must be at least three"))
    settings.gamma_range[1] < settings.gamma_range[2] ||
        throw(ArgumentError("Floquet gamma_range must be increasing"))
    settings.flip_angle_window > 0 ||
        throw(ArgumentError("flip_angle_window must be positive"))
    return settings
end

function _validate_sax_mixed_escape_settings(settings::SaxMixedEscapeSettings)
    settings.nmodes >= 2 || throw(ArgumentError("escape surface needs two modes"))
    settings.gamma_points >= 2 || throw(ArgumentError("gamma_points must be at least two"))
    settings.zeta_points >= 2 || throw(ArgumentError("zeta_points must be at least two"))
    settings.maximum_time > 0 || throw(ArgumentError("maximum_time must be positive"))
    settings.saveat > 0 || throw(ArgumentError("saveat must be positive"))
    settings.mixed_excitation > 0 ||
        throw(ArgumentError("mixed_excitation must be positive"))
    settings.activation_threshold > 0 ||
        throw(ArgumentError("activation_threshold must be positive"))
    0 < settings.mixedness_threshold < 1 ||
        throw(ArgumentError("mixedness_threshold must lie inside (0,1)"))
    settings.hold_cycles > 0 || throw(ArgumentError("hold_cycles must be positive"))
    return settings
end

# ---------------------------------------------------------------------------
# Shared cache and checkpoint helpers
# ---------------------------------------------------------------------------

function _save_sax_transition_mechanism_cache(path::AbstractString,
                                               kind::Symbol,
                                               payload,
                                               model_p::NamedTuple,
                                               settings)
    cache = (
        schema_version=SAX_TRANSITION_MECHANISM_SCHEMA_VERSION,
        cache_kind=kind,
        settings_signature=_portable_sax_mechanism_settings(settings),
        model_signature=_sax_bifurcation_model_signature(
            model_p, getfield(settings, :nmodes)),
        saved_at_unix=time(),
        payload=payload,
    )
    _atomic_jld2_save(path; cache)
    return payload
end

function _load_sax_transition_mechanism_cache(path::AbstractString,
                                               kind::Symbol,
                                               model_p::NamedTuple,
                                               settings)
    isfile(path) || return (status=:missing, payload=nothing,
                            reason="cache is absent")
    stored = try
        Logging.with_logger(Logging.NullLogger()) do
            JLD2.load(path, "cache")
        end
    catch err
        return (status=:corrupt, payload=nothing, reason=sprint(showerror, err))
    end
    compatible = try
        stored.schema_version == SAX_TRANSITION_MECHANISM_SCHEMA_VERSION &&
        stored.cache_kind == kind &&
        isequal(stored.settings_signature,
                _portable_sax_mechanism_settings(settings)) &&
        isequal(stored.model_signature,
                _sax_bifurcation_model_signature(
                    model_p, getfield(settings, :nmodes)))
    catch
        false
    end
    compatible || return (status=:incompatible, payload=nothing,
                           reason="schema, model, or settings changed")
    return (status=:valid, payload=stored.payload, reason="compatible cache")
end

"""
    load_sax_periodic_stage_checkpoint(stage_directory, model_p;
                                       kind, mode, gamma_hint, settings)

Load one converged full periodic-orbit checkpoint from the ordinary Final
stage cache.  The source cache is only read.  This is the preferred source for
the known PD point and the high-gamma mode-2 NS point because it preserves the
validated `40 x 4` orbit without recomputing the parent branch.
"""
function load_sax_periodic_stage_checkpoint(
        stage_directory::AbstractString,
        model_p::NamedTuple;
        kind::Symbol,
        mode::Integer,
        gamma_hint::Real,
        settings::SaxBifurcationSettings=sax_bifurcation_settings(:final))
    kind in (:fold, :pd, :ns) ||
        throw(ArgumentError("periodic checkpoint kind must be :fold, :pd, or :ns"))
    path = _sax_stage_cache_path(stage_directory, :periodic)
    loaded = _load_sax_stage_cache(path, :periodic, model_p, settings)
    loaded.status == :valid || return (
        status=loaded.status,
        checkpoint=nothing,
        reason=loaded.reason,
        path=path,
    )
    candidates = [checkpoint for checkpoint in loaded.payload.periodic_checkpoints
                  if checkpoint.type == kind && checkpoint.mode == Int(mode) &&
                     checkpoint.localization_status == :converged]
    isempty(candidates) && return (
        status=:missing,
        checkpoint=nothing,
        reason="no converged mode-$(Int(mode)) $(kind) checkpoint is stored",
        path=path,
    )
    selected = candidates[argmin(
        abs(checkpoint.gamma - float(gamma_hint)) for checkpoint in candidates)]
    return (status=:valid, checkpoint=selected,
            reason="converged ordinary-stage checkpoint", path=path)
end

function _sax_mechanism_bifurcation_settings(nmodes::Integer,
                                              intervals::Integer,
                                              degree::Integer;
                                              gamma_range=(0.30, 0.99),
                                              zeta_range=(0.001, 0.99),
                                              po_ds=1e-3,
                                              po_dsmax=5e-3,
                                              po_max_steps=500,
                                              po_save_sol_every_step=5,
                                              newton_tol=1e-10,
                                              stability_tol=1e-8)
    return sax_bifurcation_settings(
        :final;
        nmodes=Int(nmodes),
        gamma_range=Tuple(float.(gamma_range)),
        zeta_range=Tuple(float.(zeta_range)),
        po_collocation_intervals=Int(intervals),
        po_collocation_degree=Int(degree),
        po_linear_solver=:condensed,
        po_ds=float(po_ds),
        po_dsmax=float(po_dsmax),
        po_max_steps=Int(po_max_steps),
        po_save_sol_every_step=Int(po_save_sol_every_step),
        newton_tol=float(newton_tol),
        stability_tol=float(stability_tol),
    )
end

function _sax_periodic_orbit_validation(checkpoint,
                                        model_p::NamedTuple,
                                        settings::SaxBifurcationSettings;
                                        target::Symbol,
                                        residual_tolerance::Real=1e-8,
                                        floquet_tolerance::Real=2e-3,
                                        r2_angle_warning::Real=0.10)
    target in (:pd, :ns) || throw(ArgumentError("target must be :pd or :ns"))
    wrapper, parameters = _sax_periodic_wrapper(checkpoint, model_p, settings)
    residual_norm = norm(BK.residual(wrapper, checkpoint.solution, parameters), Inf)
    jacobian = BK.jacobian(wrapper, checkpoint.solution, parameters)
    collocation = BK.get_discretization(wrapper)
    exponents, _, converged, iterations = BK.FloquetColl()(
        collocation, jacobian, 2 + 2 * settings.nmodes)
    values = ComplexF64.(exponents)
    neutral_index = argmin(abs.(values))
    neutral = values[neutral_index]
    available = [index for index in eachindex(values) if index != neutral_index]
    # Several collocation exponents may share the same principal angle.  This
    # happens at a flip because every negative real multiplier is represented
    # with imaginary part pi.  Select the exponent closest to the *target point*
    # on the imaginary axis, not merely the one with the closest angle.
    critical_index = if target == :pd
        available[argmin(hypot(
            real(values[index]) - real(neutral),
            abs(abs(imag(values[index])) - pi),
        ) for index in available)]
    else
        theta = abs(float(checkpoint.floquet_angle))
        available[argmin(hypot(
            real(values[index]) - real(neutral),
            abs(abs(imag(values[index])) - theta),
        ) for index in available)]
    end
    critical = values[critical_index]
    angle = abs(imag(critical))
    angle_to_pi = abs(pi - angle)
    corrected_growth = real(critical) - real(neutral)
    critical_condition = target == :pd ?
        hypot(corrected_growth, angle_to_pi) : abs(corrected_growth)
    orbit_record = _record_sax_periodic_orbit(
        checkpoint.solution,
        (prob=wrapper, p=parameters);
        grazing_velocity_tol=settings.periodic_grazing_velocity_tol,
    )
    valid = residual_norm <= float(residual_tolerance) &&
            critical_condition <= float(floquet_tolerance) &&
            orbit_record.minimum_absolute_pressure_drop > settings.smoothness_tol &&
            !orbit_record.possible_grazing_contact
    return (
        valid=valid,
        target=target,
        gamma=float(checkpoint.gamma),
        zeta=float(checkpoint.zeta),
        localization_status=checkpoint.localization_status,
        localization_precision=float(checkpoint.localization_precision),
        residual_norm=float(residual_norm),
        floquet_converged=Bool(converged),
        floquet_iterations=Int(iterations),
        neutral_exponent=neutral,
        critical_exponent=critical,
        corrected_critical_growth=float(corrected_growth),
        floquet_angle=float(angle),
        angle_to_pi=float(angle_to_pi),
        near_r2=angle_to_pi <= float(r2_angle_warning),
        critical_condition=float(critical_condition),
        orbit=orbit_record,
    )
end

# ---------------------------------------------------------------------------
# 1. High-gamma mode-2 Neimark-Sacker validation and continuation
# ---------------------------------------------------------------------------

_sax_mechanism_ns_formulation(formulation::Symbol) =
    formulation == :minaug ? BK.MinAug() :
    formulation == :matrix_based ? BK.MinAugMatrixBased() :
    throw(ArgumentError("NS formulation must be :minaug or :matrix_based"))

function _sax_mechanism_ns_initial_vectors(wrapper,
                                           checkpoint,
                                           parameters,
                                           state_dimension::Integer)
    jacobian = BK.jacobian(wrapper, checkpoint.solution, parameters)
    bordered = Complex.(copy(jacobian))
    dimension = size(bordered, 1)
    border = sin.(collect(1.0:dimension))
    bordered[end, :] .= border
    bordered[:, end] .= reverse(border)
    bordered[end, end] = 0
    bordered[end-state_dimension:end-1, end-state_dimension:end-1] .=
        I(state_dimension) .* cis(checkpoint.floquet_angle)
    rhs = zeros(eltype(bordered), dimension)
    rhs[end] = 1
    right = (bordered \ rhs)[begin:end-1]
    left = (adjoint(bordered) \ rhs)[begin:end-1]
    right ./= norm(right)
    left ./= norm(left)
    return left, right
end

function _sax_mechanism_ns_state(z)
    inner = BK.getvec(z)
    orbit, gamma, theta = if inner isa BK.BorderedArray
        (
            collect(float.(BK.getvec(inner))),
            float(inner.p[1]),
            float(inner.p[2]),
        )
    else
        length(inner) >= 3 || error("flattened NS state has fewer than three entries")
        (
            collect(float.(@view inner[begin:end-2])),
            float(inner[end-1]),
            float(inner[end]),
        )
    end
    zeta = float(z isa BK.BorderedArray ? z.p : BK.getp(z))
    return (orbit=orbit, gamma=gamma, zeta=zeta, theta=theta)
end

function _sax_mechanism_ns_checkpoint(z,
                                      seed,
                                      formulation::Symbol,
                                      direction::Symbol,
                                      accepted_step::Integer)
    state = _sax_mechanism_ns_state(z)
    return (
        key="high_gamma_ns_$(formulation)_$(direction)_step$(Int(accepted_step))",
        type=:ns,
        mode=Int(seed.mode),
        source_hopf_key=seed.source_hopf_key,
        specialpoint_index=0,
        localization_status=:accepted_continuation_step,
        localization_precision=NaN,
        gamma=state.gamma,
        zeta=state.zeta,
        floquet_angle=state.theta,
        solution=state.orbit,
    )
end

function _load_sax_mechanism_ns_checkpoints(path::AbstractString,
                                            model_p::NamedTuple,
                                            settings::SaxHighGammaNSSettings;
                                            formulation::Union{Nothing,Symbol}=nothing,
                                            direction::Union{Nothing,Symbol}=nothing,
                                            source_seed_key::Union{Nothing,AbstractString}=nothing)
    isfile(path) || return (status=:missing, records=Any[],
                            reason="checkpoint cache is absent")
    stored = try
        Logging.with_logger(Logging.NullLogger()) do
            JLD2.load(path, "checkpoint_cache")
        end
    catch err
        return (status=:corrupt, records=Any[], reason=sprint(showerror, err))
    end
    compatible = try
        stored.schema_version == SAX_TRANSITION_MECHANISM_CHECKPOINT_SCHEMA_VERSION &&
        stored.cache_kind == :high_gamma_ns_checkpoints &&
        !isempty(stored.checkpoints) &&
        (isnothing(formulation) || stored.formulation == formulation) &&
        (isnothing(direction) || stored.direction == direction) &&
        (isnothing(source_seed_key) ||
         stored.source_seed_key == source_seed_key) &&
        isequal(stored.settings_signature,
                _portable_sax_mechanism_settings(settings)) &&
        isequal(stored.model_signature,
                _sax_bifurcation_model_signature(model_p, settings.nmodes))
    catch
        false
    end
    compatible || return (status=:incompatible, records=Any[],
                           reason="checkpoint schema, model, or settings changed")
    return (status=:valid, records=Any[stored.checkpoints...],
            reason="compatible checkpoints")
end

function _sax_mechanism_ns_callback(path::AbstractString,
                                    seed,
                                    model_p::NamedTuple,
                                    settings::SaxHighGammaNSSettings,
                                    formulation::Symbol,
                                    direction::Symbol,
                                    verbosity::Integer)
    loaded = _load_sax_mechanism_ns_checkpoints(
        path, model_p, settings;
        formulation=formulation,
        direction=direction,
        source_seed_key=seed.key,
    )
    checkpoints = loaded.status == :valid ? copy(loaded.records) : Any[]
    step_offset = isempty(checkpoints) ? 0 : checkpoints[end].accepted_step
    started_ns = time_ns()
    callback = function (z, tangent, step, branch; kwargs...)
        solver_state = get(kwargs, :state, nothing)
        !isnothing(solver_state) && BK.in_bisection(solver_state) && return true
        accepted_step = step_offset + Int(step)
        orbit_checkpoint = _sax_mechanism_ns_checkpoint(
            z, seed, formulation, direction, accepted_step)
        ds = try
            float(solver_state.ds)
        catch
            NaN
        end
        if verbosity > 0 &&
                (step == 1 || step % settings.progress_every == 0)
            @info(
                "High-gamma NS progress",
                formulation,
                direction,
                accepted_step,
                local_step=step,
                step_limit=settings.max_steps,
                gamma=orbit_checkpoint.gamma,
                zeta=orbit_checkpoint.zeta,
                theta=orbit_checkpoint.floquet_angle,
                angle_to_pi=abs(pi - abs(orbit_checkpoint.floquet_angle)),
                ds,
                elapsed_seconds=_sax_elapsed_seconds(started_ns),
            )
        end
        if step == 1 || step % settings.checkpoint_every == 0
            push!(checkpoints, (
                accepted_step=accepted_step,
                saved_at_unix=time(),
                gamma=orbit_checkpoint.gamma,
                zeta=orbit_checkpoint.zeta,
                theta=orbit_checkpoint.floquet_angle,
                angle_to_pi=abs(pi - abs(orbit_checkpoint.floquet_angle)),
                ds=ds,
                orbit_checkpoint=orbit_checkpoint,
                augmented_solution=_portable_sax_pd_augmented(z),
                augmented_tangent=_portable_sax_pd_augmented(tangent),
            ))
            checkpoint_cache = (
                schema_version=SAX_TRANSITION_MECHANISM_CHECKPOINT_SCHEMA_VERSION,
                cache_kind=:high_gamma_ns_checkpoints,
                settings_signature=_portable_sax_mechanism_settings(settings),
                model_signature=_sax_bifurcation_model_signature(
                    model_p, settings.nmodes),
                formulation=formulation,
                direction=direction,
                source_seed_key=seed.key,
                updated_at_unix=time(),
                checkpoints=copy(checkpoints),
            )
            _atomic_jld2_save(path; checkpoint_cache)
        end
        return true
    end
    return callback, checkpoints
end

function _sax_mechanism_ns_options(settings::SaxHighGammaNSSettings,
                                   direction::Symbol)
    direction in (:positive, :negative) ||
        throw(ArgumentError("NS direction must be :positive or :negative"))
    signed_ds = direction == :positive ? abs(settings.ds) : -abs(settings.ds)
    options = BK.ContinuationPar(
        p_min=settings.zeta_range[1],
        p_max=settings.zeta_range[2],
        ds=signed_ds,
        dsmin=settings.dsmin,
        dsmax=settings.dsmax,
        max_steps=settings.max_steps,
        detect_bifurcation=0,
        nev=2 + 2 * settings.nmodes,
        save_eigenvectors=settings.codim2_detection > 0,
        save_sol_every_step=settings.save_sol_every_step,
        tol_stability=settings.stability_tol,
        newton_options=BK.NewtonPar(
            tol=settings.newton_tol,
            max_iterations=settings.newton_max_iterations,
            verbose=false,
        ),
    )
    return BK.detect_codim2_parameters(
        settings.codim2_detection,
        options;
        update_minaug_every_step=1,
    )
end

function _sax_ns_curve_from_checkpoint_records(records,
                                               seed,
                                               formulation::Symbol,
                                               direction::Symbol,
                                               terminal_status::Symbol,
                                               terminal_error)
    return (
        kind=:ns,
        gamma=[float(record.gamma) for record in records],
        zeta=[float(record.zeta) for record in records],
        theta=[float(record.theta) for record in records],
        mode=Int(seed.mode),
        frequency=NaN,
        source=(checkpoint=seed.key, formulation=formulation,
                direction=direction, high_gamma=true),
        diagnostics=(
            point_count=length(records),
            active_parameter=:zeta,
            terminal_status=terminal_status,
            terminal_error=terminal_error,
            minimum_angle_to_pi=isempty(records) ? NaN :
                minimum(record.angle_to_pi for record in records),
        ),
    )
end

function _sax_mechanism_ns_r2_points(branch)
    points = Any[]
    for special in branch.specialpoint
        special.type in (:R2, :gpdR2) || continue
        gamma = try
            float(special.printsol.gamma)
        catch
            NaN
        end
        zeta = try
            float(special.printsol.zeta)
        catch
            try
                float(special.param)
            catch
                NaN
            end
        end
        theta = try
            float(special.printsol.ωₙₛ)
        catch
            NaN
        end
        push!(points, (
            type=:R2,
            status=special.status,
            gamma=gamma,
            zeta=zeta,
            theta=theta,
            precision=float(special.precision),
        ))
    end
    return points
end

"""
    continue_sax_high_gamma_ns(seed, model_p; settings, formulation,
                               direction, checkpoint_path, verbosity)

Continue one side of the high-gamma NS locus.  Each accepted group of steps is
saved atomically, and a failure returns the checkpoint-derived partial curve
instead of discarding it.
"""
function continue_sax_high_gamma_ns(
        seed::NamedTuple,
        model_p::NamedTuple;
        settings::SaxHighGammaNSSettings=SaxHighGammaNSSettings(),
        formulation::Symbol=:minaug,
        direction::Symbol=:positive,
        checkpoint_path::AbstractString,
        verbosity::Integer=1)
    _validate_sax_high_gamma_ns_settings(settings)
    seed.type == :ns || throw(ArgumentError("high-gamma NS seed must have type :ns"))
    bif_settings = _sax_mechanism_bifurcation_settings(
        settings.nmodes,
        settings.collocation_intervals,
        settings.collocation_degree;
        zeta_range=settings.zeta_range,
        newton_tol=settings.newton_tol,
        stability_tol=settings.stability_tol,
    )
    wrapper, parameters = _sax_periodic_wrapper(seed, model_p, bif_settings)
    state_dimension, _, _ = size(BK.get_discretization(wrapper))
    left, right = _sax_mechanism_ns_initial_vectors(
        wrapper, seed, parameters, state_dimension)
    initial = BK.BorderedArray(
        copy(seed.solution), [seed.gamma, seed.floquet_angle])
    options = _sax_mechanism_ns_options(settings, direction)
    callback, records = _sax_mechanism_ns_callback(
        checkpoint_path, seed, model_p, settings,
        formulation, direction, verbosity)
    branch = nothing
    terminal_error = nothing
    started_ns = time_ns()
    try
        branch = BK.continuation_ns(
            wrapper,
            BK.PALC(),
            initial,
            parameters,
            (BK.@optic _.gamma),
            (BK.@optic _.zeta),
            left,
            right,
            options;
            update_minaug_every_step=1,
            bdlinsolver=BK.MatrixBLS(),
            jacobian_ma=_sax_mechanism_ns_formulation(formulation),
            compute_eigen_elements=settings.codim2_detection > 0,
            usehessian=false,
            bothside=false,
            normC=BK.norminf,
            finalise_solution=callback,
            verbosity=max(0, Int(verbosity) - 1),
            plot=false,
            kind=BK.NSPeriodicOrbitCont(),
        )
    catch err
        err isa InterruptException && rethrow()
        terminal_error = (
            exception_type=Symbol(nameof(typeof(err))),
            error=sprint(showerror, err),
        )
    end
    loaded = _load_sax_mechanism_ns_checkpoints(
        checkpoint_path, model_p, settings;
        formulation=formulation,
        direction=direction,
        source_seed_key=seed.key,
    )
    records = loaded.status == :valid ? loaded.records : records
    terminal_status = if isnothing(branch)
        isempty(records) ? :failed : :partial_failure
    else
        diagnostics = _sax_branch_terminal_diagnostics(branch, :zeta)
        classifications = [endpoint.classification for endpoint in diagnostics.endpoints]
        any(classification -> classification in (
            :lower_parameter_boundary, :upper_parameter_boundary),
            classifications) ? :complete : :partial_returned
    end
    curve = if isnothing(branch)
        _sax_ns_curve_from_checkpoint_records(
            records, seed, formulation, direction,
            terminal_status, terminal_error)
    else
        summary = _curve_summary(
            branch,
            :ns;
            mode=seed.mode,
            source=(checkpoint=seed.key, formulation=formulation,
                    direction=direction, high_gamma=true),
            active_parameter=:zeta,
        )
        _portable_sax_curve(merge(summary, (
            diagnostics=merge(summary.diagnostics, (
                terminal_status=terminal_status,
                terminal_error=terminal_error,
            )),
        )))
    end
    r2_points = isnothing(branch) ? Any[] : try
        _sax_mechanism_ns_r2_points(branch)
    catch
        Any[]
    end
    return (
        status=terminal_status,
        curve=curve,
        seed=seed,
        formulation=formulation,
        direction=direction,
        checkpoint_path=abspath(checkpoint_path),
        checkpoint_count=length(records),
        r2_points=r2_points,
        terminal_error=terminal_error,
        elapsed_seconds=_sax_elapsed_seconds(started_ns),
    )
end

function _sax_high_gamma_ns_component_paths(directory::AbstractString,
                                            formulation::Symbol,
                                            direction::Symbol)
    stem = "high_gamma_ns_$(formulation)_$(direction)"
    return (
        result=joinpath(directory, "$(stem).jld2"),
        checkpoints=joinpath(directory, "$(stem)_checkpoints.jld2"),
    )
end

"""
    compute_sax_high_gamma_ns(model_p, stage_directory, output_directory;
                              settings, resume=true, verbosity=1)

Validate the stored high-gamma mode-2 NS orbit and continue it independently
in both zeta directions.  Formulations are tried in `settings.formulations`
order and later formulations are fallbacks when no accepted step was saved.
The validation is cached before either expensive augmented solve begins.
"""
function compute_sax_high_gamma_ns(
        model_p::NamedTuple,
        stage_directory::AbstractString,
        output_directory::AbstractString;
        settings::SaxHighGammaNSSettings=SaxHighGammaNSSettings(),
        resume::Bool=true,
        verbosity::Integer=1)
    _validate_sax_high_gamma_ns_settings(settings)
    mkpath(output_directory)
    final_settings = sax_bifurcation_settings(:final)
    source = load_sax_periodic_stage_checkpoint(
        stage_directory,
        model_p;
        kind=:ns,
        mode=settings.mode,
        gamma_hint=settings.gamma_hint,
        settings=final_settings,
    )
    source.status == :valid || error(
        "high-gamma NS source is unavailable: $(source.reason)")
    seed = source.checkpoint
    bif_settings = _sax_mechanism_bifurcation_settings(
        settings.nmodes,
        settings.collocation_intervals,
        settings.collocation_degree;
        zeta_range=settings.zeta_range,
        newton_tol=settings.newton_tol,
        stability_tol=settings.stability_tol,
    )
    validation = _sax_periodic_orbit_validation(
        seed,
        model_p,
        bif_settings;
        target=:ns,
        residual_tolerance=settings.validation_residual_tol,
        floquet_tolerance=settings.validation_floquet_tol,
        r2_angle_warning=settings.r2_angle_warning,
    )
    validation.valid || error(
        "stored high-gamma NS checkpoint failed validation: $(validation)")
    validation_result = (
        analysis=:high_gamma_ns_validation,
        seed=seed,
        source_path=abspath(source.path),
        validation=validation,
    )
    _save_sax_transition_mechanism_cache(
        joinpath(output_directory, "high_gamma_ns_validation.jld2"),
        :high_gamma_ns_validation,
        validation_result,
        model_p,
        settings,
    )

    components = Any[]
    failures = Any[]
    for direction in (:positive, :negative)
        primary_completed = false
        for formulation in settings.formulations
            primary_completed && break
            paths = _sax_high_gamma_ns_component_paths(
                output_directory, formulation, direction)
            cached = resume ? _load_sax_transition_mechanism_cache(
                paths.result, :high_gamma_ns_component,
                model_p, settings) :
                (status=:missing, payload=nothing, reason="resume disabled")
            component = if cached.status == :valid
                cached.payload
            else
                verbosity > 0 && @info(
                    "Starting high-gamma NS component",
                    formulation,
                    direction,
                    gamma=seed.gamma,
                    zeta=seed.zeta,
                    theta=seed.floquet_angle,
                )
                run = continue_sax_high_gamma_ns(
                    seed,
                    model_p;
                    settings=settings,
                    formulation=formulation,
                    direction=direction,
                    checkpoint_path=paths.checkpoints,
                    verbosity=verbosity,
                )
                _save_sax_transition_mechanism_cache(
                    paths.result, :high_gamma_ns_component,
                    run, model_p, settings)
            end
            push!(components, component)
            primary_completed = component.checkpoint_count > 0
            if !primary_completed
                push!(failures, (
                    stage=:high_gamma_ns,
                    formulation=formulation,
                    direction=direction,
                    status=component.status,
                    error=component.terminal_error,
                ))
            end
        end
    end
    result = (
        analysis=:high_gamma_ns,
        seed=seed,
        source_path=abspath(source.path),
        validation=validation,
        components=components,
        failures=failures,
    )
    _save_sax_transition_mechanism_cache(
        joinpath(output_directory, "high_gamma_ns_manifest.jld2"),
        :high_gamma_ns_manifest,
        result,
        model_p,
        settings,
    )
    return result
end

# ---------------------------------------------------------------------------
# 2. Branch switching from the known PD point to the period-two family
# ---------------------------------------------------------------------------

function _sax_period_two_options(
        settings::SaxPeriodTwoSettings;
        detect_bifurcation::Integer=3,
        save_eigenvectors::Bool=true)
    return BK.ContinuationPar(
        p_min=settings.gamma_range[1],
        p_max=settings.gamma_range[2],
        ds=settings.ds,
        dsmin=settings.dsmin,
        dsmax=settings.dsmax,
        max_steps=settings.max_steps,
        detect_bifurcation=Int(detect_bifurcation),
        nev=2 + 2 * settings.nmodes,
        save_eigenvectors=save_eigenvectors,
        save_sol_every_step=settings.save_sol_every_step,
        n_inversion=8,
        max_bisection_steps=35,
        tol_stability=settings.stability_tol,
        newton_options=BK.NewtonPar(
            tol=settings.newton_tol,
            max_iterations=settings.newton_max_iterations,
            verbose=false,
        ),
    )
end

function _sax_period_two_floquet_samples(
        branch,
        zeta::Real;
        include_solutions::Bool=false)
    rows = Dict(Int(row.step) => row for row in branch.branch)
    solutions = include_solutions ?
        Dict(Int(point.step) => collect(float.(point.x)) for point in branch.sol) :
        Dict{Int,Vector{Float64}}()
    samples = Any[]
    for eigen_record in branch.eig
        hasproperty(eigen_record, :converged) && !eigen_record.converged && continue
        step = Int(eigen_record.step)
        haskey(rows, step) || continue
        row = rows[step]
        values = ComplexF64.(eigen_record.eigenvals)
        isempty(values) && continue
        sample = (
            continuation_step=step,
            gamma=float(row.gamma),
            zeta=float(zeta),
            period=float(row.period),
            exponents=values,
        )
        if include_solutions && haskey(solutions, step)
            sample = merge(sample, (solution=solutions[step],))
        end
        push!(samples, sample)
    end
    sort!(samples; by=sample -> sample.continuation_step)
    return samples
end

function _sax_period_two_callback(path::AbstractString,
                                  seed,
                                  model_p::NamedTuple,
                                  settings::SaxPeriodTwoSettings,
                                  verbosity::Integer)
    checkpoints = Any[]
    started_ns = time_ns()
    callback = function (z, tangent, step, branch; kwargs...)
        solver_state = get(kwargs, :state, nothing)
        !isnothing(solver_state) && BK.in_bisection(solver_state) && return true
        gamma = float(BK.getp(z))
        solution = collect(float.(BK.getvec(z)))
        ds = try
            float(solver_state.ds)
        catch
            NaN
        end
        if verbosity > 0 &&
                (step == 1 || step % settings.progress_every == 0)
            @info(
                "Period-two branch progress",
                accepted_step=Int(step),
                step_limit=settings.max_steps,
                gamma,
                zeta=seed.zeta,
                ds,
                elapsed_seconds=_sax_elapsed_seconds(started_ns),
            )
        end
        if step == 1 || step % settings.checkpoint_every == 0
            push!(checkpoints, (
                accepted_step=Int(step),
                saved_at_unix=time(),
                gamma=gamma,
                zeta=float(seed.zeta),
                solution=solution,
                tangent=_portable_sax_pd_augmented(tangent),
                ds=ds,
            ))
            checkpoint_cache = (
                schema_version=SAX_TRANSITION_MECHANISM_CHECKPOINT_SCHEMA_VERSION,
                cache_kind=:period_two_checkpoints,
                settings_signature=_portable_sax_mechanism_settings(settings),
                model_signature=_sax_bifurcation_model_signature(
                    model_p, settings.nmodes),
                source_seed_key=seed.key,
                doubled_collocation_intervals=2 * settings.collocation_intervals,
                collocation_degree=settings.collocation_degree,
                updated_at_unix=time(),
                checkpoints=copy(checkpoints),
            )
            _atomic_jld2_save(path; checkpoint_cache)
        end
        return true
    end
    return callback, checkpoints
end

function _sax_period_two_specialpoints(branch)
    return [(
        type=special.type,
        status=special.status,
        gamma=float(special.param),
        precision=float(special.precision),
        index=Int(special.idx),
        step=Int(special.step),
    ) for special in branch.specialpoint if special.type != :endpoint]
end

"""
    continue_sax_period_two(pd_seed, model_p; settings, checkpoint_path,
                            verbosity=1)

Construct the period-doubling eigenfunction directly from the portable Final
PD orbit, double the collocation mesh and period, correct the predictor, and
continue the resulting period-two family.  Predictor signs and amplitudes are
tried in a deterministic order because only one side exists for a given flip
criticality.
"""
function continue_sax_period_two(
        pd_seed::NamedTuple,
        model_p::NamedTuple;
        settings::SaxPeriodTwoSettings=SaxPeriodTwoSettings(),
        checkpoint_path::AbstractString,
        eigsolver=nothing,
        detect_bifurcation::Integer=3,
        save_eigenvectors::Bool=true,
        include_floquet_solutions::Bool=false,
        verbosity::Integer=1)
    _validate_sax_period_two_settings(settings)
    pd_seed.type == :pd || throw(ArgumentError("period-two seed must have type :pd"))
    bif_settings = _sax_mechanism_bifurcation_settings(
        settings.nmodes,
        settings.collocation_intervals,
        settings.collocation_degree;
        gamma_range=settings.gamma_range,
        newton_tol=settings.newton_tol,
        stability_tol=settings.stability_tol,
    )
    parent_validation = _sax_periodic_orbit_validation(
        pd_seed,
        model_p,
        bif_settings;
        target=:pd,
        residual_tolerance=settings.validation_residual_tol,
        floquet_tolerance=settings.validation_floquet_tol,
    )
    parent_validation.valid || error(
        "stored PD checkpoint failed validation: $(parent_validation)")
    wrapper, parameters = _sax_periodic_wrapper(pd_seed, model_p, bif_settings)
    parent_period = BK.getperiod(wrapper, pd_seed.solution, parameters)
    pd_point = BK.PeriodDoubling(
        copy(pd_seed.solution),
        nothing,
        pd_seed.gamma,
        parameters,
        (BK.@optic _.gamma),
        nothing,
        nothing,
        nothing,
        :none,
    )
    # The simplified Iooss calculation supplies the anti-periodic eigenvector
    # but avoids expensive third-order coefficients.  Predictor direction is
    # tested explicitly below, so no criticality assumption is needed.
    normal_form = BK.period_doubling_normal_form_iooss(
        wrapper,
        pd_point;
        detailed=Val(false),
        verbose=verbosity > 1,
    )
    attempts = Any[]
    for delta_gamma in settings.delta_gamma_candidates,
            amplitude_factor in settings.amplitude_factors
        try
            verbosity > 0 && @info(
                "Trying period-two branch switch",
                delta_gamma,
                amplitude_factor,
                parent_gamma=pd_seed.gamma,
            )
            predictor = BK.predictor(
                normal_form,
                delta_gamma,
                amplitude_factor;
                override=true,
            )
            discretization = BK.get_discretization(predictor.prob)
            options = BK._update_cont_params(
                _sax_period_two_options(
                    settings;
                    detect_bifurcation=detect_bifurcation,
                    save_eigenvectors=save_eigenvectors,
                ),
                discretization,
                predictor.orbitguess,
            )
            discretization = BK._set_params_in_po(
                discretization,
                BK.set(parameters, (BK.@optic _.gamma), predictor.pnew),
            )
            callback, checkpoints = _sax_period_two_callback(
                checkpoint_path, pd_seed, model_p, settings, verbosity)
            common = (
                linear_algo=BK.COPBLS(),
                record_from_solution=(x, info; kwargs...) ->
                    _record_sax_periodic_orbit(
                        x,
                        info;
                        grazing_velocity_tol=1e-5,
                        kwargs...,
                    ),
                normC=BK.norminf,
                finalise_solution=callback,
                verbosity=max(0, Int(verbosity) - 1),
                plot=false,
            )
            branch = isnothing(eigsolver) ? BK.continuation(
                discretization,
                predictor.orbitguess,
                BK.PALC(),
                options;
                common...,
            ) : BK.continuation(
                discretization,
                predictor.orbitguess,
                BK.PALC(),
                options;
                eigsolver=eigsolver,
                common...,
            )
            length(branch) >= 2 || error("period-two branch returned fewer than two points")
            gamma, zeta = _sax_branch_coordinates(branch)
            periods = collect(float.(branch.branch.period))
            initial_period_ratio = first(periods) / float(parent_period)
            1.5 <= initial_period_ratio <= 2.5 || error(
                "branch-switch period ratio $(initial_period_ratio) is not period two")
            curve = (
                kind=:p2,
                gamma=gamma,
                zeta=zeta,
                period=periods,
                pressure_amplitude=collect(float.(branch.branch.pressure_amplitude)),
                pressure_l2=collect(float.(branch.branch.pressure_l2)),
                stable=collect(map(value -> value === true,
                                   branch.branch.stable)),
                n_unstable=collect(Int.(branch.branch.n_unstable)),
                mode=Int(pd_seed.mode),
                frequency=NaN,
                source=(checkpoint=pd_seed.key,
                        delta_gamma=float(delta_gamma),
                        amplitude_factor=float(amplitude_factor)),
                diagnostics=_sax_branch_terminal_diagnostics(branch, :gamma),
            )
            return (
                analysis=:period_two_branch,
                curve=curve,
                seed=pd_seed,
                parent_validation=parent_validation,
                parent_period=float(parent_period),
                initial_period_ratio=float(initial_period_ratio),
                doubled_collocation_intervals=2 * settings.collocation_intervals,
                collocation_degree=settings.collocation_degree,
                specialpoints=_sax_period_two_specialpoints(branch),
                floquet_samples=_sax_period_two_floquet_samples(
                    branch, pd_seed.zeta;
                    include_solutions=include_floquet_solutions),
                floquet_solver=isnothing(eigsolver) ? :default :
                    Symbol(nameof(typeof(eigsolver))),
                attempts=vcat(attempts, [(
                    delta_gamma=float(delta_gamma),
                    amplitude_factor=float(amplitude_factor),
                    status=:converged,
                    points=length(branch),
                )]),
                checkpoint_path=abspath(checkpoint_path),
                checkpoint_count=length(checkpoints),
            )
        catch err
            err isa InterruptException && rethrow()
            push!(attempts, (
                delta_gamma=float(delta_gamma),
                amplitude_factor=float(amplitude_factor),
                status=:failed,
                exception_type=Symbol(nameof(typeof(err))),
                error=sprint(showerror, err),
            ))
        end
    end
    error("all period-two branch-switch attempts failed: $(attempts)")
end

"""Load the Final PD checkpoint, branch-switch to P2, and save a portable result."""
function compute_sax_period_two(
        model_p::NamedTuple,
        stage_directory::AbstractString,
        output_directory::AbstractString;
        settings::SaxPeriodTwoSettings=SaxPeriodTwoSettings(),
        resume::Bool=true,
        verbosity::Integer=1)
    _validate_sax_period_two_settings(settings)
    mkpath(output_directory)
    result_path = joinpath(output_directory, "period_two_branch.jld2")
    cached = resume ? _load_sax_transition_mechanism_cache(
        result_path, :period_two_branch, model_p, settings) :
        (status=:missing, payload=nothing, reason="resume disabled")
    cached.status == :valid && return cached.payload
    source = load_sax_periodic_stage_checkpoint(
        stage_directory,
        model_p;
        kind=:pd,
        mode=settings.mode,
        gamma_hint=settings.gamma_hint,
        settings=sax_bifurcation_settings(:final),
    )
    source.status == :valid || error(
        "Final PD checkpoint is unavailable: $(source.reason)")
    result = continue_sax_period_two(
        source.checkpoint,
        model_p;
        settings=settings,
        checkpoint_path=joinpath(output_directory, "period_two_checkpoints.jld2"),
        verbosity=verbosity,
    )
    return _save_sax_transition_mechanism_cache(
        result_path, :period_two_branch, result, model_p, settings)
end

# ---------------------------------------------------------------------------
# 3. Independent fixed-zeta Floquet slicing
# ---------------------------------------------------------------------------

function _sax_floquet_sample(values,
                             gamma::Real,
                             zeta::Real,
                             period::Real,
                             branch_index::Integer,
                             flip_angle_window::Real)
    exponents = ComplexF64.(values)
    isempty(exponents) && return nothing
    neutral_index = argmin(abs.(exponents))
    neutral = exponents[neutral_index]
    available = [index for index in eachindex(exponents) if index != neutral_index]
    isempty(available) && return nothing
    # A negative multiplier far inside the unit circle also has principal
    # exponent angle pi.  Selecting by angle alone therefore reports the wrong
    # transverse direction whenever a near-R2 complex pair is much closer to
    # the unit circle.  Distance to the corrected flip point resolves the tie.
    flip_index = available[argmin(hypot(
        real(exponents[index]) - real(neutral),
        abs(abs(imag(exponents[index])) - pi),
    ) for index in available)]
    flip = exponents[flip_index]
    corrected_flip_growth = real(flip) - real(neutral)
    angle_to_pi = abs(pi - abs(imag(flip)))
    multiplier = exp(flip)
    dominant_index = available[argmax(real(exponents[index]) for index in available)]
    dominant = exponents[dominant_index]
    return (
        branch_index=Int(branch_index),
        gamma=float(gamma),
        zeta=float(zeta),
        period=float(period),
        neutral_exponent=neutral,
        dominant_exponent=dominant,
        dominant_growth=float(real(dominant) - real(neutral)),
        dominant_angle=float(abs(imag(dominant))),
        dominant_angle_to_pi=float(abs(pi - abs(imag(dominant)))),
        flip_exponent=flip,
        flip_growth=float(corrected_flip_growth),
        flip_angle=float(abs(imag(flip))),
        flip_angle_to_pi=float(angle_to_pi),
        flip_distance=float(hypot(corrected_flip_growth, angle_to_pi)),
        flip_multiplier=multiplier,
        flip_candidate=angle_to_pi <= float(flip_angle_window),
    )
end

function _sax_floquet_branch_samples(branch,
                                     zeta::Real,
                                     settings::SaxFloquetSliceSettings)
    count = min(length(branch), length(branch.eig))
    samples = Any[]
    for index in 1:count
        eigen_record = branch.eig[index]
        hasproperty(eigen_record, :eigenvals) || continue
        gamma = branch.branch.gamma[index]
        period = branch.branch.period[index]
        sample = _sax_floquet_sample(
            eigen_record.eigenvals,
            gamma,
            zeta,
            period,
            index,
            settings.flip_angle_window,
        )
        isnothing(sample) || push!(samples, sample)
    end
    return samples
end

function _sax_floquet_slice_bifurcations(branch, hopf, mode::Integer)
    checkpoints = _sax_periodic_bifurcation_checkpoints(branch, hopf, mode)
    portable = map(checkpoints) do checkpoint
        (
            type=checkpoint.type,
            mode=checkpoint.mode,
            gamma=checkpoint.gamma,
            zeta=checkpoint.zeta,
            floquet_angle=checkpoint.floquet_angle,
            localization_status=checkpoint.localization_status,
            localization_precision=checkpoint.localization_precision,
        )
    end
    return portable
end

"""
    compute_sax_floquet_slice(model_p, zeta; settings, verbosity=1)

Independently refine the mode-2 Hopf point at one fixed zeta, continue its
period-one family in gamma, record the full Floquet spectrum at every accepted
point, and retain independently localized PD/NS/fold roots.  No tangent or
augmented state from the two-parameter PD curve is reused.
"""
function compute_sax_floquet_slice(
        model_p::NamedTuple,
        zeta::Real;
        settings::SaxFloquetSliceSettings=SaxFloquetSliceSettings(),
        verbosity::Integer=1)
    _validate_sax_floquet_slice_settings(settings)
    rescue_settings = SaxPDRescueSettings(
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
        po_save_sol_every_step=0,
        newton_tol=settings.newton_tol,
        stability_tol=settings.stability_tol,
        hopf_scan_points=settings.hopf_scan_points,
    )
    hopf = refine_sax_hopf_checkpoint(
        model_p,
        zeta,
        settings.mode;
        settings=rescue_settings,
    )
    bif_settings = _sax_pd_rescue_bifurcation_settings(rescue_settings)
    started_ns = time_ns()
    branch = continue_sax_periodic_orbits(
        hopf,
        model_p;
        settings=bif_settings,
        verbosity=Int(verbosity),
    )
    samples = _sax_floquet_branch_samples(branch, zeta, settings)
    bifurcations = _sax_floquet_slice_bifurcations(
        branch, hopf, settings.mode)
    return (
        analysis=:transverse_floquet_slice,
        zeta=float(zeta),
        hopf_checkpoint=hopf,
        samples=samples,
        bifurcations=bifurcations,
        pd_points=[point for point in bifurcations if point.type == :pd],
        ns_points=[point for point in bifurcations if point.type == :ns],
        fold_points=[point for point in bifurcations if point.type == :fold],
        diagnostics=_sax_branch_terminal_diagnostics(branch, :gamma),
        elapsed_seconds=_sax_elapsed_seconds(started_ns),
    )
end

function _sax_floquet_unique_samples(samples)
    isempty(samples) && return Any[]
    ordered = sort(collect(samples); by=sample -> sample.gamma)
    groups = Vector{Vector{Any}}()
    for sample in ordered
        if isempty(groups) ||
                abs(sample.gamma - last(last(groups)).gamma) > 1e-7
            push!(groups, Any[sample])
        else
            push!(last(groups), sample)
        end
    end
    # Both-sided Hopf continuation often returns the same physical family
    # twice with opposite arclength orientation.  At repeated gamma choose the
    # least-stable transverse sample.  The raw multivalued samples remain in
    # each slice and are never discarded from the cache.
    return [group[argmax(sample.dominant_growth for sample in group)]
            for group in groups]
end

function _sax_transverse_roots(samples,
                               flip_angle_window::Real;
                               zeta::Real=NaN)
    length(samples) >= 2 || return Any[]
    roots = Any[]
    for (left, right) in zip(samples[begin:end-1], samples[begin+1:end])
        g1 = float(left.dominant_growth)
        g2 = float(right.dominant_growth)
        isfinite(g1) && isfinite(g2) || continue
        signbit(g1) == signbit(g2) && !iszero(g1) && !iszero(g2) && continue
        denominator = g2 - g1
        fraction = abs(denominator) <= eps(Float64) ? 0.5 : clamp(-g1 / denominator, 0, 1)
        gamma = (1 - fraction) * left.gamma + fraction * right.gamma
        angle = (1 - fraction) * left.dominant_angle + fraction * right.dominant_angle
        angle_to_pi = abs(pi - angle)
        push!(roots, (
            gamma=float(gamma),
            zeta=isfinite(float(zeta)) ? float(zeta) : float(left.zeta),
            angle=float(angle),
            angle_to_pi=float(angle_to_pi),
            classification=angle_to_pi <= float(flip_angle_window) ?
                :near_r2_transverse_root : :ns_transverse_root,
            source=:linear_interpolation_of_independent_floquet_samples,
        ))
    end
    return roots
end

function _sax_linear_sample(x, xs, ys)
    x < first(xs) && return NaN
    x > last(xs) && return NaN
    right = searchsortedfirst(xs, x)
    right == 1 && return float(first(ys))
    right > length(xs) && return float(last(ys))
    left = right - 1
    denominator = xs[right] - xs[left]
    abs(denominator) <= eps(Float64) && return float(ys[left])
    fraction = (x - xs[left]) / denominator
    return float((1 - fraction) * ys[left] + fraction * ys[right])
end

"""Regularize raw transverse slices only for heatmap display."""
function assemble_sax_floquet_slice_map(
        slices,
        settings::SaxFloquetSliceSettings=SaxFloquetSliceSettings())
    gamma = collect(range(
        settings.gamma_range[1], settings.gamma_range[2];
        length=settings.gamma_grid_points,
    ))
    ordered_slices = sort(collect(slices); by=slice -> slice.zeta)
    zeta = [float(slice.zeta) for slice in ordered_slices]
    dimensions = (length(zeta), length(gamma))
    flip_growth = fill(NaN, dimensions)
    flip_angle_to_pi = fill(NaN, dimensions)
    flip_distance = fill(NaN, dimensions)
    dominant_growth = fill(NaN, dimensions)
    dominant_angle = fill(NaN, dimensions)
    dominant_angle_to_pi = fill(NaN, dimensions)
    period = fill(NaN, dimensions)
    for (row, slice) in enumerate(ordered_slices)
        samples = _sax_floquet_unique_samples(slice.samples)
        length(samples) >= 2 || continue
        xs = [sample.gamma for sample in samples]
        for (column, value) in enumerate(gamma)
            flip_growth[row, column] = _sax_linear_sample(
                value, xs, [sample.flip_growth for sample in samples])
            flip_angle_to_pi[row, column] = _sax_linear_sample(
                value, xs, [sample.flip_angle_to_pi for sample in samples])
            flip_distance[row, column] = _sax_linear_sample(
                value, xs, [sample.flip_distance for sample in samples])
            dominant_growth[row, column] = _sax_linear_sample(
                value, xs, [sample.dominant_growth for sample in samples])
            dominant_angle[row, column] = _sax_linear_sample(
                value, xs, [sample.dominant_angle for sample in samples])
            dominant_angle_to_pi[row, column] = _sax_linear_sample(
                value, xs, [sample.dominant_angle_to_pi for sample in samples])
            period[row, column] = _sax_linear_sample(
                value, xs, [sample.period for sample in samples])
        end
    end
    roots = [point for slice in ordered_slices for point in slice.pd_points]
    transverse_roots = [root for slice in ordered_slices for root in
        _sax_transverse_roots(
            _sax_floquet_unique_samples(slice.samples),
            settings.flip_angle_window,
            zeta=slice.zeta,
        )]
    return (
        gamma=gamma,
        zeta=zeta,
        flip_growth=flip_growth,
        flip_angle_to_pi=flip_angle_to_pi,
        flip_distance=flip_distance,
        dominant_growth=dominant_growth,
        dominant_angle=dominant_angle,
        dominant_angle_to_pi=dominant_angle_to_pi,
        period=period,
        pd_roots=roots,
        transverse_roots=transverse_roots,
        note="Regular arrays are display interpolants; raw branch samples and localized roots are authoritative",
    )
end

_sax_floquet_zeta_tag(zeta::Real) =
    replace(@sprintf("%.6f", float(zeta)), "." => "p")

"""Compute every fixed-zeta slice with one atomic cache per completed slice."""
function compute_sax_transverse_floquet_map(
        model_p::NamedTuple,
        output_directory::AbstractString;
        settings::SaxFloquetSliceSettings=SaxFloquetSliceSettings(),
        resume::Bool=true,
        verbosity::Integer=1)
    _validate_sax_floquet_slice_settings(settings)
    mkpath(output_directory)
    slices = Any[]
    failures = Any[]
    for (index, zeta) in enumerate(settings.zeta_values)
        path = joinpath(
            output_directory,
            "floquet_slice_z$(_sax_floquet_zeta_tag(zeta)).jld2",
        )
        cached = resume ? _load_sax_transition_mechanism_cache(
            path, :transverse_floquet_slice, model_p, settings) :
            (status=:missing, payload=nothing, reason="resume disabled")
        if cached.status == :valid
            push!(slices, cached.payload)
            continue
        end
        verbosity > 0 && @info(
            "Starting transverse Floquet slice",
            slice=index,
            slices=length(settings.zeta_values),
            zeta,
            collocation="$(settings.collocation_intervals)x$(settings.collocation_degree)",
        )
        try
            slice = compute_sax_floquet_slice(
                model_p, zeta; settings=settings, verbosity=verbosity)
            push!(slices, _save_sax_transition_mechanism_cache(
                path, :transverse_floquet_slice, slice, model_p, settings))
        catch err
            err isa InterruptException && rethrow()
            failure = (
                zeta=float(zeta),
                exception_type=Symbol(nameof(typeof(err))),
                error=sprint(showerror, err),
            )
            push!(failures, failure)
            verbosity > 0 && @warn "Transverse Floquet slice failed" failure
        end
    end
    result = (
        analysis=:transverse_floquet_map,
        settings=_portable_sax_mechanism_settings(settings),
        slices=sort(slices; by=slice -> slice.zeta),
        map=assemble_sax_floquet_slice_map(slices, settings),
        failures=failures,
    )
    _save_sax_transition_mechanism_cache(
        joinpath(output_directory, "transverse_floquet_map.jld2"),
        :transverse_floquet_map,
        result,
        model_p,
        settings,
    )
    return result
end

"""Plot a regularized transverse Floquet metric with localized PD roots."""
function plot_sax_transverse_floquet_map(result; metric::Symbol=:flip_growth)
    metric in (:flip_growth, :flip_distance, :dominant_growth) ||
        throw(ArgumentError("metric must be :flip_growth, :flip_distance, or :dominant_growth"))
    values = getproperty(result.map, metric)
    display_values = metric == :flip_distance ?
        log10.(max.(values, eps(Float64))) : values
    title = metric == :flip_growth ? "Corrected flip Floquet growth" :
        metric == :flip_distance ? "log10 distance to flip condition" :
        "Dominant nontrivial Floquet growth"
    p = heatmap(
        result.map.gamma,
        result.map.zeta,
        display_values;
        xlabel="gamma",
        ylabel="zeta",
        title=title,
        color=:balance,
        colorbar=true,
    )
    if !isempty(result.map.pd_roots)
        scatter!(
            p,
            [point.gamma for point in result.map.pd_roots],
            [point.zeta for point in result.map.pd_roots];
            marker=:circle,
            markercolor=:black,
            markerstrokecolor=:white,
            markersize=4,
            label="localized PD roots",
        )
    end
    if !isempty(result.map.transverse_roots)
        scatter!(
            p,
            [point.gamma for point in result.map.transverse_roots],
            [point.zeta for point in result.map.transverse_roots];
            marker=:diamond,
            markercolor=:white,
            markerstrokecolor=:black,
            markersize=4,
            label="Floquet sign-change roots",
        )
    end
    return p
end

# ---------------------------------------------------------------------------
# 4. Mixed-mode escape-time surface
# ---------------------------------------------------------------------------

"""
    _sax_mixed_escape_metrics(times, amplitude1, amplitude2; ...)

Measure how long an initially active mixed state persists.  A completed escape
requires an active signal with mixedness below threshold for one uninterrupted
hold window.  Runs that remain mixed until the integration limit are explicitly
right-censored instead of being treated as measured escape times.
"""
function _sax_mixed_escape_metrics(
        times::AbstractVector{<:Real},
        amplitude1::AbstractVector{<:Real},
        amplitude2::AbstractVector{<:Real};
        activation_threshold::Real,
        mixedness_threshold::Real,
        hold_time::Real)
    length(times) == length(amplitude1) == length(amplitude2) ||
        throw(DimensionMismatch("time and modal-amplitude vectors must agree"))
    length(times) >= 2 || throw(ArgumentError("at least two time samples are required"))
    total = float.(amplitude1) .+ float.(amplitude2)
    mixedness = 4 .* float.(amplitude1) .* float.(amplitude2) ./
        max.(total .^ 2, eps(Float64))
    active = total .>= float(activation_threshold)
    mixed = active .& (mixedness .>= float(mixedness_threshold))
    start_index = findfirst(mixed)
    if isnothing(start_index)
        return (
            status=:never_mixed,
            escape_time=NaN,
            right_censored=false,
            escape_index=0,
            escape_register=Int8(0),
            mixed_fraction=0.0,
            peak_mixedness=maximum(mixedness),
            peak_total_amplitude=maximum(total),
            final_mixedness=mixedness[end],
            final_total_amplitude=total[end],
        )
    end
    time_step = median(diff(float.(times)))
    hold_samples = max(1, ceil(Int, float(hold_time) / time_step))
    escape_index = 0
    for index in start_index:(length(times) - hold_samples + 1)
        window = index:(index + hold_samples - 1)
        if all(@view(active[window])) &&
                all((@view mixedness[window]) .< float(mixedness_threshold))
            escape_index = index
            break
        end
    end
    considered = start_index:length(times)
    if escape_index == 0
        tail_count = min(hold_samples, length(times) - start_index + 1)
        tail = (length(times) - tail_count + 1):length(times)
        inactive = !any(@view(active[tail]))
        return (
            status=inactive ? :inactive_tail : :right_censored,
            escape_time=float(times[end] - times[start_index]),
            right_censored=!inactive,
            escape_index=0,
            escape_register=Int8(0),
            mixed_fraction=mean(@view mixedness[considered]),
            peak_mixedness=maximum(@view mixedness[considered]),
            peak_total_amplitude=maximum(@view total[considered]),
            final_mixedness=mixedness[end],
            final_total_amplitude=total[end],
        )
    end
    window = escape_index:(escape_index + hold_samples - 1)
    mean1 = mean(@view amplitude1[window])
    mean2 = mean(@view amplitude2[window])
    register = mean1 > mean2 ? Int8(1) : Int8(2)
    return (
        status=:escaped,
        escape_time=float(times[escape_index] - times[start_index]),
        right_censored=false,
        escape_index=escape_index,
        escape_register=register,
        mixed_fraction=mean(@view mixedness[considered]),
        peak_mixedness=maximum(@view mixedness[considered]),
        peak_total_amplitude=maximum(@view total[considered]),
        final_mixedness=mixedness[end],
        final_total_amplitude=total[end],
    )
end

"""Compute one reproducible equal-mode perturbation escape experiment."""
function sax_mixed_mode_escape(
        model_p::NamedTuple,
        gamma::Real,
        zeta::Real;
        settings::SaxMixedEscapeSettings=SaxMixedEscapeSettings(),
        retain_series::Bool=false)
    _validate_sax_mixed_escape_settings(settings)
    fixed_state, fixed_residual = _estimate_fixed_point(
        gamma, zeta, model_p; nmodes=settings.nmodes)
    initial_state = copy(fixed_state)
    initial_state[3] += settings.mixed_excitation
    initial_state[5] += settings.mixed_excitation
    parameters = set_parameters(
        float(gamma), float(zeta), model_p, Int64(settings.nmodes))
    problem = ODEProblem(
        saxRN!, initial_state, (0.0, settings.maximum_time), parameters)
    solution = solve(
        problem,
        Tsit5();
        saveat=settings.saveat,
        reltol=settings.reltol,
        abstol=settings.abstol,
    )
    DifferentialEquations.SciMLBase.successful_retcode(solution) || error(
        "mixed-mode escape integration failed at gamma=$(gamma), zeta=$(zeta): $(solution.retcode)",
    )
    states = Array(solution)
    amplitude1 = _mode_amplitude_series(
        states, 1; center=(fixed_state[3], fixed_state[4]))
    amplitude2 = _mode_amplitude_series(
        states, 2; center=(fixed_state[5], fixed_state[6]))
    hold_time = settings.hold_cycles * 2 * pi / float(model_p.ω[1])
    metrics = _sax_mixed_escape_metrics(
        solution.t,
        amplitude1,
        amplitude2;
        activation_threshold=settings.activation_threshold,
        mixedness_threshold=settings.mixedness_threshold,
        hold_time=hold_time,
    )
    tail_start = max(1, floor(Int, 0.9 * length(amplitude1)))
    tail1 = mean(@view amplitude1[tail_start:end])
    tail2 = mean(@view amplitude2[tail_start:end])
    final_register = max(tail1, tail2) < settings.activation_threshold ?
        Int8(0) : tail1 > tail2 ? Int8(1) : Int8(2)
    return merge(metrics, (
        gamma=float(gamma),
        zeta=float(zeta),
        fixed_residual=float(fixed_residual),
        hold_time=float(hold_time),
        final_register=final_register,
        final_mode_amplitudes=(float(tail1), float(tail2)),
        integrated_time=float(last(solution.t)),
        series=retain_series ? (
            time=collect(float.(solution.t)),
            amplitude1=collect(float.(amplitude1)),
            amplitude2=collect(float.(amplitude2)),
        ) : nothing,
    ))
end

function _sax_escape_row_path(directory::AbstractString, zeta::Real)
    return joinpath(
        directory,
        "mixed_escape_z$(_sax_floquet_zeta_tag(zeta)).jld2",
    )
end

function _compute_sax_mixed_escape_row(model_p::NamedTuple,
                                       gamma,
                                       zeta::Real,
                                       settings::SaxMixedEscapeSettings;
                                       parallel::Bool=true,
                                       verbosity::Integer=1)
    results = Vector{Any}(undef, length(gamma))
    completed = Threads.Atomic{Int}(0)
    function evaluate!(index)
        results[index] = try
            sax_mixed_mode_escape(
                model_p, gamma[index], zeta; settings=settings)
        catch err
            (
                gamma=float(gamma[index]),
                zeta=float(zeta),
                status=:error,
                escape_time=NaN,
                right_censored=false,
                escape_register=Int8(0),
                mixed_fraction=NaN,
                peak_mixedness=NaN,
                peak_total_amplitude=NaN,
                final_mixedness=NaN,
                final_total_amplitude=NaN,
                final_register=Int8(0),
                exception_type=Symbol(nameof(typeof(err))),
                error=sprint(showerror, err),
            )
        end
        done = Threads.atomic_add!(completed, 1) + 1
        if verbosity > 0 &&
                (done == length(gamma) ||
                 done % max(1, div(length(gamma), 10)) == 0)
            @info(
                "Mixed-mode escape row progress",
                zeta=float(zeta),
                completed=done,
                total=length(gamma),
                threads=Threads.nthreads(),
            )
        end
        return nothing
    end
    if parallel && Threads.nthreads() > 1
        Threads.@threads for index in eachindex(gamma)
            evaluate!(index)
        end
    else
        for index in eachindex(gamma)
            evaluate!(index)
        end
    end
    return (
        analysis=:mixed_escape_row,
        zeta=float(zeta),
        gamma=collect(float.(gamma)),
        points=results,
    )
end

function _assemble_sax_mixed_escape_rows(rows,
                                         settings::SaxMixedEscapeSettings)
    ordered_rows = sort(collect(rows); by=row -> row.zeta)
    gamma = collect(range(
        settings.gamma_range[1], settings.gamma_range[2];
        length=settings.gamma_points,
    ))
    zeta = [float(row.zeta) for row in ordered_rows]
    escape_time = fill(NaN, length(zeta), length(gamma))
    right_censored = falses(length(zeta), length(gamma))
    status = Matrix{Symbol}(undef, length(zeta), length(gamma))
    escape_register = zeros(Int8, length(zeta), length(gamma))
    final_register = zeros(Int8, length(zeta), length(gamma))
    mixed_fraction = fill(NaN, length(zeta), length(gamma))
    peak_mixedness = fill(NaN, length(zeta), length(gamma))
    for (row_index, row) in enumerate(ordered_rows),
            column in eachindex(row.points)
        point = row.points[column]
        escape_time[row_index, column] = point.escape_time
        right_censored[row_index, column] = point.right_censored
        status[row_index, column] = point.status
        escape_register[row_index, column] = point.escape_register
        final_register[row_index, column] = point.final_register
        mixed_fraction[row_index, column] = point.mixed_fraction
        peak_mixedness[row_index, column] = point.peak_mixedness
    end
    return (
        gamma=gamma,
        zeta=zeta,
        escape_time=escape_time,
        right_censored=right_censored,
        status=status,
        escape_register=escape_register,
        final_register=final_register,
        mixed_fraction=mixed_fraction,
        peak_mixedness=peak_mixedness,
        rows=ordered_rows,
    )
end

"""
    compute_sax_mixed_escape_surface(model_p, output_directory; settings,
                                     resume=true, parallel=true, verbosity=1)

Compute a right-censoring-aware escape-time surface.  Every zeta row is saved
atomically as soon as it finishes, so an overnight run can resume without
repeating completed integrations.
"""
function compute_sax_mixed_escape_surface(
        model_p::NamedTuple,
        output_directory::AbstractString;
        settings::SaxMixedEscapeSettings=SaxMixedEscapeSettings(),
        resume::Bool=true,
        parallel::Bool=true,
        verbosity::Integer=1)
    _validate_sax_mixed_escape_settings(settings)
    mkpath(output_directory)
    gamma = collect(range(
        settings.gamma_range[1], settings.gamma_range[2];
        length=settings.gamma_points,
    ))
    zeta = collect(range(
        settings.zeta_range[1], settings.zeta_range[2];
        length=settings.zeta_points,
    ))
    rows = Any[]
    for (row_index, value) in enumerate(zeta)
        path = _sax_escape_row_path(output_directory, value)
        cached = resume ? _load_sax_transition_mechanism_cache(
            path, :mixed_escape_row, model_p, settings) :
            (status=:missing, payload=nothing, reason="resume disabled")
        row = if cached.status == :valid
            cached.payload
        else
            verbosity > 0 && @info(
                "Starting mixed-mode escape row",
                row=row_index,
                rows=length(zeta),
                zeta=value,
                gamma_points=length(gamma),
                maximum_time=settings.maximum_time,
            )
            computed = _compute_sax_mixed_escape_row(
                model_p,
                gamma,
                value,
                settings;
                parallel=parallel,
                verbosity=verbosity,
            )
            _save_sax_transition_mechanism_cache(
                path, :mixed_escape_row, computed, model_p, settings)
        end
        push!(rows, row)
    end
    assembled = _assemble_sax_mixed_escape_rows(rows, settings)
    result = merge(assembled, (
        analysis=:mixed_mode_escape_surface,
        settings=_portable_sax_mechanism_settings(settings),
        protocol=(
            initial_condition=:equal_mode_1_mode_2_pressure_perturbation,
            mixedness_formula="4*A1*A2/(A1+A2)^2",
            right_censored_value=settings.maximum_time,
        ),
    ))
    return _save_sax_transition_mechanism_cache(
        joinpath(output_directory, "mixed_escape_surface.jld2"),
        :mixed_escape_surface,
        result,
        model_p,
        settings,
    )
end

"""Plot log10 mixed-mode escape time and mark the censoring boundary."""
function plot_sax_mixed_escape_surface(result)
    values = log10.(max.(result.escape_time, eps(Float64)))
    p = heatmap(
        result.gamma,
        result.zeta,
        values;
        xlabel="gamma",
        ylabel="zeta",
        title="Mixed-mode escape time",
        color=:viridis,
        colorbar_title="log10 time",
    )
    if any(result.right_censored) && !all(result.right_censored)
        contour!(
            p,
            result.gamma,
            result.zeta,
            Float64.(result.right_censored);
            levels=[0.5],
            color=:white,
            linewidth=2,
            label="right-censoring boundary",
        )
    end
    return p
end

# ---------------------------------------------------------------------------
# Read-only progress loading for Pluto notebooks
# ---------------------------------------------------------------------------

function _sax_provisional_ns_curve(records,
                                   formulation::Symbol,
                                   direction::Symbol)
    return (
        kind=:ns,
        mode=Int(first(records).orbit_checkpoint.mode),
        gamma=[float(record.gamma) for record in records],
        zeta=[float(record.zeta) for record in records],
        theta=[float(record.theta) for record in records],
        frequency=fill(NaN, length(records)),
        source=(
            analysis=:high_gamma_ns,
            formulation=formulation,
            direction=direction,
            provisional=true,
        ),
        diagnostics=(
            terminal_status=:checkpoint_in_progress,
            point_count=length(records),
            accepted_step=last(records).accepted_step,
        ),
    )
end

function _load_sax_period_two_checkpoint_curve(path::AbstractString,
                                               model_p::NamedTuple,
                                               settings::SaxPeriodTwoSettings)
    isfile(path) || return (status=:missing, curve=nothing,
                            reason="checkpoint cache is absent")
    stored = try
        Logging.with_logger(Logging.NullLogger()) do
            JLD2.load(path, "checkpoint_cache")
        end
    catch err
        return (status=:corrupt, curve=nothing, reason=sprint(showerror, err))
    end
    compatible = try
        stored.schema_version == SAX_TRANSITION_MECHANISM_CHECKPOINT_SCHEMA_VERSION &&
        stored.cache_kind == :period_two_checkpoints &&
        isequal(stored.settings_signature,
                _portable_sax_mechanism_settings(settings)) &&
        isequal(stored.model_signature,
                _sax_bifurcation_model_signature(model_p, settings.nmodes)) &&
        !isempty(stored.checkpoints)
    catch
        false
    end
    compatible || return (status=:incompatible, curve=nothing,
                           reason="checkpoint schema, model, or settings changed")
    records = stored.checkpoints
    curve = (
        kind=:p2,
        gamma=[float(record.gamma) for record in records],
        zeta=[float(record.zeta) for record in records],
        mode=Int(settings.mode),
        frequency=NaN,
        source=(analysis=:period_two, provisional=true),
        diagnostics=(
            terminal_status=:checkpoint_in_progress,
            point_count=length(records),
            accepted_step=last(records).accepted_step,
        ),
    )
    return (status=:checkpoint, curve=curve,
            reason="compatible period-two checkpoints")
end

"""
    load_sax_transition_mechanism_progress(model_p, directory; profile=:final)

Load completed transition-mechanism products and safely fall back to atomic
NS/P2 checkpoints, completed Floquet slices, and completed escape rows.  The
function is read-only and is safe to call while the external runner is active.
Only files whose schema, model, and complete profile settings match are used.
"""
function load_sax_transition_mechanism_progress(
        model_p::NamedTuple,
        directory::AbstractString;
        profile::Symbol=:final)
    settings = sax_transition_mechanism_settings(profile)
    ns_directory = joinpath(directory, "high_gamma_ns")
    p2_directory = joinpath(directory, "period_two")
    floquet_directory = joinpath(directory, "floquet_slices")
    escape_directory = joinpath(directory, "mixed_escape")

    ns_manifest = _load_sax_transition_mechanism_cache(
        joinpath(ns_directory, "high_gamma_ns_manifest.jld2"),
        :high_gamma_ns_manifest, model_p, settings.ns)
    ns_validation = _load_sax_transition_mechanism_cache(
        joinpath(ns_directory, "high_gamma_ns_validation.jld2"),
        :high_gamma_ns_validation, model_p, settings.ns)
    ns_curves = Any[]
    ns_rows = Any[]
    completed_components = Set{Tuple{Symbol,Symbol}}()
    if ns_manifest.status == :valid
        for component in ns_manifest.payload.components
            provisional = component.status != :complete
            if !isempty(component.curve.gamma)
                curve = provisional ? merge(component.curve, (
                    source=merge(component.curve.source, (provisional=true,)),
                )) : component.curve
                push!(ns_curves, curve)
            end
            push!(completed_components,
                  (component.formulation, component.direction))
            push!(ns_rows, (
                formulation=component.formulation,
                direction=component.direction,
                status=component.status,
                points=length(component.curve.gamma),
                checkpoint_count=component.checkpoint_count,
                accepted_step=missing,
                gamma=isempty(component.curve.gamma) ? missing :
                    float(last(component.curve.gamma)),
                zeta=isempty(component.curve.zeta) ? missing :
                    float(last(component.curve.zeta)),
                terminal_error=component.terminal_error,
                provisional=provisional,
            ))
        end
    end
    for formulation in settings.ns.formulations, direction in (:positive, :negative)
        (formulation, direction) in completed_components && continue
        paths = _sax_high_gamma_ns_component_paths(
            ns_directory, formulation, direction)
        component = _load_sax_transition_mechanism_cache(
            paths.result,
            :high_gamma_ns_component,
            model_p,
            settings.ns,
        )
        if component.status == :valid
            provisional = component.payload.status != :complete
            if !isempty(component.payload.curve.gamma)
                curve = provisional ? merge(component.payload.curve, (
                    source=merge(component.payload.curve.source,
                                 (provisional=true,)),
                )) : component.payload.curve
                push!(ns_curves, curve)
            end
            push!(completed_components, (formulation, direction))
            push!(ns_rows, (
                formulation=formulation,
                direction=direction,
                status=component.payload.status,
                points=length(component.payload.curve.gamma),
                checkpoint_count=component.payload.checkpoint_count,
                accepted_step=missing,
                gamma=isempty(component.payload.curve.gamma) ? missing :
                    float(last(component.payload.curve.gamma)),
                zeta=isempty(component.payload.curve.zeta) ? missing :
                    float(last(component.payload.curve.zeta)),
                terminal_error=component.payload.terminal_error,
                provisional=provisional,
            ))
            continue
        end
        loaded = _load_sax_mechanism_ns_checkpoints(
            paths.checkpoints,
            model_p,
            settings.ns;
            formulation=formulation,
            direction=direction,
        )
        loaded.status == :valid || continue
        push!(ns_curves, _sax_provisional_ns_curve(
            loaded.records, formulation, direction))
        push!(ns_rows, (
            formulation=formulation,
            direction=direction,
            status=:checkpoint,
            points=length(loaded.records),
            checkpoint_count=length(loaded.records),
            accepted_step=last(loaded.records).accepted_step,
            gamma=float(last(loaded.records).gamma),
            zeta=float(last(loaded.records).zeta),
            terminal_error=nothing,
            provisional=true,
        ))
    end

    p2_completed = _load_sax_transition_mechanism_cache(
        joinpath(p2_directory, "period_two_branch.jld2"),
        :period_two_branch, model_p, settings.p2)
    p2 = if p2_completed.status == :valid
        (status=:complete, curve=p2_completed.payload.curve,
         result=p2_completed.payload, reason="compatible completed P2 branch")
    else
        checkpoint = _load_sax_period_two_checkpoint_curve(
            joinpath(p2_directory, "period_two_checkpoints.jld2"),
            model_p,
            settings.p2,
        )
        (status=checkpoint.status, curve=checkpoint.curve,
         result=nothing, reason=checkpoint.reason)
    end

    floquet_completed = _load_sax_transition_mechanism_cache(
        joinpath(floquet_directory, "transverse_floquet_map.jld2"),
        :transverse_floquet_map, model_p, settings.floquet)
    floquet_slices = Any[]
    if floquet_completed.status == :valid
        append!(floquet_slices, floquet_completed.payload.slices)
    else
        for zeta in settings.floquet.zeta_values
            path = joinpath(
                floquet_directory,
                "floquet_slice_z$(_sax_floquet_zeta_tag(zeta)).jld2",
            )
            loaded = _load_sax_transition_mechanism_cache(
                path, :transverse_floquet_slice, model_p, settings.floquet)
            loaded.status == :valid && push!(floquet_slices, loaded.payload)
        end
    end
    floquet_result = if floquet_completed.status == :valid
        floquet_completed.payload
    elseif isempty(floquet_slices)
        nothing
    else
        (
            analysis=:transverse_floquet_map,
            settings=_portable_sax_mechanism_settings(settings.floquet),
            slices=sort(floquet_slices; by=slice -> slice.zeta),
            map=assemble_sax_floquet_slice_map(floquet_slices, settings.floquet),
            failures=Any[],
            provisional=true,
        )
    end

    escape_completed = _load_sax_transition_mechanism_cache(
        joinpath(escape_directory, "mixed_escape_surface.jld2"),
        :mixed_escape_surface, model_p, settings.escape)
    escape_rows = Any[]
    if escape_completed.status == :valid
        append!(escape_rows, escape_completed.payload.rows)
    else
        expected_zeta = range(
            settings.escape.zeta_range[1], settings.escape.zeta_range[2];
            length=settings.escape.zeta_points,
        )
        for zeta in expected_zeta
            loaded = _load_sax_transition_mechanism_cache(
                _sax_escape_row_path(escape_directory, zeta),
                :mixed_escape_row,
                model_p,
                settings.escape,
            )
            loaded.status == :valid && push!(escape_rows, loaded.payload)
        end
    end
    escape_result = if escape_completed.status == :valid
        escape_completed.payload
    elseif isempty(escape_rows)
        nothing
    else
        merge(_assemble_sax_mixed_escape_rows(escape_rows, settings.escape), (
            analysis=:mixed_mode_escape_surface,
            settings=_portable_sax_mechanism_settings(settings.escape),
            provisional=true,
        ))
    end

    ns_status = if ns_manifest.status == :valid
        isempty(ns_curves) ? :validation_only :
            isempty(ns_manifest.payload.failures) ? :complete : :partial
    elseif isempty(ns_curves)
        ns_manifest.status
    else
        :checkpoint
    end

    return (
        profile=profile,
        directory=abspath(directory),
        settings=settings,
        ns=(
            status=ns_status,
            curves=ns_curves,
            rows=ns_rows,
            validation=ns_manifest.status == :valid ?
                ns_manifest.payload.validation :
                ns_validation.status == :valid ?
                    ns_validation.payload.validation : nothing,
            manifest=ns_manifest.status == :valid ? ns_manifest.payload : nothing,
        ),
        p2=p2,
        floquet=(
            status=floquet_completed.status == :valid ? :complete :
                isempty(floquet_slices) ? floquet_completed.status : :partial,
            result=floquet_result,
            completed_slices=length(floquet_slices),
            expected_slices=length(settings.floquet.zeta_values),
        ),
        escape=(
            status=escape_completed.status == :valid ? :complete :
                isempty(escape_rows) ? escape_completed.status : :partial,
            result=escape_result,
            completed_rows=length(escape_rows),
            expected_rows=settings.escape.zeta_points,
        ),
    )
end

"""Overlay completed or checkpoint transition-mechanism evidence on a diagram."""
function overlay_sax_transition_mechanisms!(
        figure,
        progress;
        subplot::Integer=1,
        show_labels::Bool=true,
        compact_labels::Bool=false,
        include_escape_contours::Bool=false)
    axis = length(figure.subplots) > 1 ? figure[Int(subplot)] : figure
    ns_labeled = false
    for curve in progress.ns.curves
        provisional = hasproperty(curve.source, :provisional) &&
            curve.source.provisional
        plot!(
            axis,
            curve.gamma,
            curve.zeta;
            color=:purple,
            linestyle=:solid,
            linewidth=provisional ? 2.0 : 3.0,
            alpha=provisional ? 0.65 : 1.0,
            marker=provisional ? :circle : :none,
            markersize=provisional ? 2.2 : 0,
            label=show_labels && !ns_labeled ?
                (compact_labels ? (provisional ? "NS*" : "NS-H") :
                 provisional ? "NS checkpoint" : "high-gamma NS") : "",
        )
        ns_labeled = true
    end
    if !isnothing(progress.p2.curve) && !isempty(progress.p2.curve.gamma)
        provisional = progress.p2.status != :complete
        plot!(
            axis,
            progress.p2.curve.gamma,
            progress.p2.curve.zeta;
            color=:teal,
            linestyle=:solid,
            linewidth=provisional ? 2.0 : 3.0,
            alpha=provisional ? 0.65 : 1.0,
            marker=:diamond,
            markersize=2.5,
            label=show_labels ?
                (compact_labels ? "P2" :
                 provisional ? "P2 checkpoint" : "P2 family") : "",
        )
    end
    if !isnothing(progress.floquet.result)
        roots = progress.floquet.result.map.transverse_roots
        if !isempty(roots)
            scatter!(
                axis,
                [root.gamma for root in roots],
                [root.zeta for root in roots];
                marker=:diamond,
                markercolor=:white,
                markerstrokecolor=:black,
                markerstrokewidth=1.2,
                markersize=5,
                label=show_labels ?
                    (compact_labels ? "FZ" : "Floquet zero") : "",
            )
        end
    end
    if include_escape_contours && !isnothing(progress.escape.result)
        escape = progress.escape.result
        finite_values = filter(isfinite, vec(escape.escape_time))
        if length(escape.zeta) >= 2 && length(escape.gamma) >= 2 &&
                !isempty(finite_values) &&
                maximum(finite_values) > minimum(finite_values)
            contour!(
                axis,
                escape.gamma,
                escape.zeta,
                log10.(max.(escape.escape_time, eps(Float64)));
                levels=4,
                color=:darkgray,
                linewidth=1.2,
                colorbar=false,
                label="",
            )
            show_labels && plot!(
                axis, [NaN], [NaN]; color=:darkgray, linewidth=1.2,
                label=compact_labels ? "tau-mix" : "escape-time contours")
        end
    end
    return figure
end
