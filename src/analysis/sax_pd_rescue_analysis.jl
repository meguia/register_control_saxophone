# This file implements a deliberately isolated period-doubling rescue run.
# It is included after sax_bifurcation_analysis.jl and therefore reuses the
# analytic saxophone model, collocation wrapper, and atomic JLD2 writer defined
# there.  Nothing in this file writes to the ordinary bifurcation caches.

#region SETTINGS AND CACHE LAYOUT

"""
    SaxPDRescueSettings(; kwargs...)

Numerical controls for a high-resolution, directional continuation of the
period-doubling locus.  The defaults implement the conservative rescue plan:

- final `40 × 4` periodic-orbit collocation
- independent PD seeds at `zeta = 0.25`, `0.75`, and `0.80`
- separate positive and negative continuations with `bothside=false`
- `ds = ±1e-4`, `dsmin = 1e-10`, and `dsmax = 5e-4`
- codimension-two event detection disabled during the first pass
- full augmented state and tangent checkpointed every five accepted steps

The two minimally augmented Jacobian formulations are selected per run with
`:matrix_based` (`BK.MinAugMatrixBased()`) or `:minaug` (`BK.MinAug()`).
"""
Base.@kwdef struct SaxPDRescueSettings
    nmodes::Int = 8
    gamma_range::Tuple{Float64,Float64} = (0.30, 0.99)
    zeta_range::Tuple{Float64,Float64} = (0.001, 0.99)
    seed_zetas::Tuple{Vararg{Float64}} = (0.25, 0.75, 0.80)

    po_collocation_intervals::Int = 40
    po_collocation_degree::Int = 4
    po_linear_solver::Symbol = :condensed
    po_ds::Float64 = 1e-3
    po_dsmax::Float64 = 6e-3
    po_max_steps::Int = 700
    po_save_sol_every_step::Int = 5

    po_curve_ds::Float64 = 1e-4
    po_curve_dsmin::Float64 = 1e-10
    po_curve_dsmax::Float64 = 5e-4
    po_curve_max_steps::Int = 1500
    po_curve_save_sol_every_step::Int = 5
    checkpoint_every::Int = 5
    progress_every::Int = 5
    codim2_detection::Int = 0

    newton_tol::Float64 = 1e-10
    newton_max_iterations::Int = 45
    stability_tol::Float64 = 1e-8
    hopf_scan_points::Int = 241
    hopf_growth_tolerance::Float64 = 1e-11
end

function _validate_sax_pd_rescue_settings(settings::SaxPDRescueSettings)
    settings.nmodes > 0 || throw(ArgumentError("nmodes must be positive"))
    settings.gamma_range[1] < settings.gamma_range[2] ||
        throw(ArgumentError("gamma_range must be increasing"))
    settings.zeta_range[1] < settings.zeta_range[2] ||
        throw(ArgumentError("zeta_range must be increasing"))
    all(settings.zeta_range[1] .<= settings.seed_zetas .<= settings.zeta_range[2]) ||
        throw(ArgumentError("every PD seed zeta must lie inside zeta_range"))
    settings.po_collocation_intervals >= 5 ||
        throw(ArgumentError("at least five collocation intervals are required"))
    settings.po_collocation_degree >= 2 ||
        throw(ArgumentError("collocation degree must be at least two"))
    settings.po_linear_solver in (:condensed, :dense) ||
        throw(ArgumentError("po_linear_solver must be :condensed or :dense"))
    0 < settings.po_curve_dsmin <= abs(settings.po_curve_ds) <= settings.po_curve_dsmax ||
        throw(ArgumentError("require 0 < po_curve_dsmin <= |po_curve_ds| <= po_curve_dsmax"))
    settings.po_curve_max_steps > 0 ||
        throw(ArgumentError("po_curve_max_steps must be positive"))
    settings.checkpoint_every > 0 ||
        throw(ArgumentError("checkpoint_every must be positive"))
    settings.progress_every > 0 ||
        throw(ArgumentError("progress_every must be positive"))
    settings.codim2_detection in 0:3 ||
        throw(ArgumentError("codim2_detection must lie in 0:3"))
    settings.newton_max_iterations > 0 ||
        throw(ArgumentError("newton_max_iterations must be positive"))
    settings.hopf_scan_points >= 11 ||
        throw(ArgumentError("hopf_scan_points must be at least 11"))
    return settings
end

function _sax_pd_rescue_bifurcation_settings(settings::SaxPDRescueSettings)
    _validate_sax_pd_rescue_settings(settings)
    return sax_bifurcation_settings(
        :final;
        nmodes=settings.nmodes,
        gamma_range=settings.gamma_range,
        zeta_range=settings.zeta_range,
        zeta_seeds=settings.seed_zetas,
        po_collocation_intervals=settings.po_collocation_intervals,
        po_collocation_degree=settings.po_collocation_degree,
        po_linear_solver=settings.po_linear_solver,
        po_ds=settings.po_ds,
        po_dsmax=settings.po_dsmax,
        po_max_steps=settings.po_max_steps,
        po_save_sol_every_step=settings.po_save_sol_every_step,
        po_curve_ds=settings.po_curve_ds,
        po_curve_dsmax=settings.po_curve_dsmax,
        po_curve_max_steps=settings.po_curve_max_steps,
        po_curve_save_sol_every_step=settings.po_curve_save_sol_every_step,
        newton_tol=settings.newton_tol,
        stability_tol=settings.stability_tol,
    )
end

function _portable_sax_pd_rescue_settings(settings::SaxPDRescueSettings)
    names = fieldnames(SaxPDRescueSettings)
    return NamedTuple{names}(Tuple(getfield(settings, name) for name in names))
end

const SAX_PD_RESCUE_CACHE_SCHEMA_VERSION = 1
const SAX_PD_RESCUE_CHECKPOINT_SCHEMA_VERSION = 1

_sax_pd_zeta_tag(zeta::Real) = replace(@sprintf("%.3f", float(zeta)), "." => "p")

"""
    sax_pd_rescue_cache_paths(directory, zeta, formulation, direction)

Return the isolated seed, component, and step-checkpoint paths for one rescue
continuation.  These names never overlap the normal `bifurcation_curves_*.jld2`
or `stages*` paths.
"""
function sax_pd_rescue_cache_paths(directory::AbstractString,
                                   zeta::Real,
                                   formulation::Symbol=:matrix_based,
                                   direction::Symbol=:positive)
    formulation in (:matrix_based, :minaug) || throw(ArgumentError(
        "formulation must be :matrix_based or :minaug",
    ))
    direction in (:positive, :negative) || throw(ArgumentError(
        "direction must be :positive or :negative",
    ))
    tag = _sax_pd_zeta_tag(zeta)
    stem = "pd_z$(tag)_$(formulation)_$(direction)"
    return (
        seed=joinpath(directory, "seed_z$(tag).jld2"),
        result=joinpath(directory, "$(stem).jld2"),
        checkpoints=joinpath(directory, "$(stem)_checkpoints.jld2"),
    )
end

#endregion

#region INDEPENDENT FINAL-RESOLUTION PD SEEDS

function _sax_modal_growth(model_p::NamedTuple,
                           gamma::Real,
                           zeta::Real,
                           mode::Integer,
                           settings::SaxBifurcationSettings)
    spectrum, equilibrium, residual, parameters = _equilibrium_spectrum(
        model_p, gamma, zeta, settings)
    eigenvalue = _sax_modal_eigenvalue(spectrum, model_p, mode)
    return (
        growth=float(real(eigenvalue)),
        eigenvalue=ComplexF64(eigenvalue),
        equilibrium=collect(float.(equilibrium)),
        residual=float(residual),
        parameters=parameters,
    )
end

"""
    refine_sax_hopf_checkpoint(model_p, zeta, mode; settings, gamma_hint=nothing)

Construct an accurate portable Hopf checkpoint without reusing a periodic
orbit.  The routine scans the requested gamma interval for sign changes of the
selected modal growth rate and then bisects the nearest crossing.  A scalar
`gamma_hint` only selects among multiple crossings; the returned equilibrium,
frequency, and gamma are recomputed from the analytic 18-state Jacobian.
"""
function refine_sax_hopf_checkpoint(
        model_p::NamedTuple,
        zeta::Real,
        mode::Integer;
        settings::SaxPDRescueSettings=SaxPDRescueSettings(),
        gamma_hint::Union{Nothing,Real}=nothing)
    _validate_sax_pd_rescue_settings(settings)
    1 <= mode <= settings.nmodes || throw(ArgumentError(
        "mode must lie in 1:$(settings.nmodes)",
    ))
    settings.zeta_range[1] <= zeta <= settings.zeta_range[2] ||
        throw(ArgumentError("zeta=$zeta lies outside settings.zeta_range"))
    bif_settings = _sax_pd_rescue_bifurcation_settings(settings)

    gamma_grid = collect(range(
        settings.gamma_range[1], settings.gamma_range[2];
        length=settings.hopf_scan_points,
    ))
    growth = fill(NaN, length(gamma_grid))
    for index in eachindex(gamma_grid)
        growth[index] = try
            _sax_modal_growth(
                model_p, gamma_grid[index], zeta, mode, bif_settings).growth
        catch
            NaN
        end
    end

    brackets = Tuple{Float64,Float64}[]
    for index in 1:(length(gamma_grid) - 1)
        left_value, right_value = growth[index], growth[index + 1]
        isfinite(left_value) && isfinite(right_value) || continue
        if left_value == 0
            push!(brackets, (gamma_grid[index], gamma_grid[index]))
        elseif signbit(left_value) != signbit(right_value)
            push!(brackets, (gamma_grid[index], gamma_grid[index + 1]))
        end
    end
    isempty(brackets) && error(
        "no mode-$mode Hopf crossing was bracketed at zeta=$(float(zeta)) " *
        "inside gamma=$(settings.gamma_range)",
    )

    target = isnothing(gamma_hint) ? sum(settings.gamma_range) / 2 : float(gamma_hint)
    bracket = brackets[argmin(abs((left + right) / 2 - target)
                              for (left, right) in brackets)]
    left, right = bracket
    left_growth = _sax_modal_growth(model_p, left, zeta, mode, bif_settings).growth
    if left != right
        for _ in 1:80
            middle = (left + right) / 2
            middle_growth = _sax_modal_growth(
                model_p, middle, zeta, mode, bif_settings).growth
            if abs(middle_growth) <= settings.hopf_growth_tolerance ||
                    abs(right - left) <= 10eps(max(abs(middle), 1.0))
                left = right = middle
                break
            elseif signbit(left_growth) == signbit(middle_growth)
                left = middle
                left_growth = middle_growth
            else
                right = middle
            end
        end
    end

    gamma = (left + right) / 2
    refined = _sax_modal_growth(model_p, gamma, zeta, mode, bif_settings)
    abs(refined.growth) <= 10settings.hopf_growth_tolerance || @warn(
        "refined Hopf seed retains a visible modal growth rate",
        gamma,
        zeta,
        mode,
        growth=refined.growth,
    )
    frequency = abs(float(imag(refined.eigenvalue)))
    key = "hopf_independent_m$(Int(mode))_g$(round(gamma; digits=10))_z$(round(float(zeta); digits=10))"
    return (
        key=key,
        mode=Int(mode),
        gamma=float(gamma),
        zeta=float(zeta),
        frequency=frequency,
        state=refined.equilibrium,
        source=(method=:analytic_modal_growth_bisection, gamma_hint=gamma_hint),
        modal_growth=refined.growth,
        equilibrium_residual=refined.residual,
    )
end

"""
    discover_sax_pd_seed(model_p, zeta; mode=2, gamma_hint=nothing,
                         pd_gamma_hint=nothing, settings, verbosity=1)

Compute a fresh final-resolution periodic branch from an independently refined
Hopf point and select a localized PD point on that branch.  `pd_gamma_hint` is
used only to choose among multiple detected PD points.  The returned
`pd_checkpoint` contains the full `40 × 4` orbit and is therefore independent
of the old `25 × 3` Pilot checkpoint.
"""
function discover_sax_pd_seed(
        model_p::NamedTuple,
        zeta::Real;
        mode::Integer=2,
        gamma_hint::Union{Nothing,Real}=nothing,
        pd_gamma_hint::Union{Nothing,Real}=nothing,
        settings::SaxPDRescueSettings=SaxPDRescueSettings(),
        verbosity::Int=1)
    _validate_sax_pd_rescue_settings(settings)
    hopf_checkpoint = refine_sax_hopf_checkpoint(
        model_p,
        zeta,
        mode;
        settings=settings,
        gamma_hint=gamma_hint,
    )
    bif_settings = _sax_pd_rescue_bifurcation_settings(settings)
    periodic_branch = continue_sax_periodic_orbits(
        hopf_checkpoint,
        model_p;
        settings=bif_settings,
        verbosity=verbosity,
    )
    candidates = [candidate for candidate in _sax_periodic_bifurcation_checkpoints(
        periodic_branch, hopf_checkpoint, mode) if candidate.type == :pd]
    isempty(candidates) && error(
        "the final-resolution periodic branch from zeta=$(float(zeta)) " *
        "did not localize a PD point",
    )
    target = isnothing(pd_gamma_hint) ? hopf_checkpoint.gamma : float(pd_gamma_hint)
    selected = candidates[argmin(abs(candidate.gamma - target) for candidate in candidates)]
    return (
        hopf_checkpoint=hopf_checkpoint,
        pd_checkpoint=selected,
        pd_candidates=candidates,
        periodic_diagnostics=_sax_branch_terminal_diagnostics(periodic_branch, :gamma),
        periodic_branch=periodic_branch,
    )
end

function _portable_sax_pd_seed(result)
    return (
        hopf_checkpoint=result.hopf_checkpoint,
        pd_checkpoint=result.pd_checkpoint,
        pd_candidates=copy(result.pd_candidates),
        periodic_diagnostics=result.periodic_diagnostics,
    )
end

function save_sax_pd_seed_cache(path::AbstractString,
                                result,
                                model_p::NamedTuple;
                                settings::SaxPDRescueSettings=SaxPDRescueSettings())
    cache = (
        schema_version=SAX_PD_RESCUE_CACHE_SCHEMA_VERSION,
        cache_kind=:pd_seed,
        settings_signature=_portable_sax_pd_rescue_settings(settings),
        model_signature=_sax_bifurcation_model_signature(model_p, settings.nmodes),
        saved_at_unix=time(),
        seed=_portable_sax_pd_seed(result),
    )
    _atomic_jld2_save(path; cache)
    return cache.seed
end

function load_sax_pd_seed_cache(path::AbstractString,
                                model_p::NamedTuple;
                                settings::SaxPDRescueSettings=SaxPDRescueSettings(),
                                target_zeta::Union{Nothing,Real}=nothing)
    isfile(path) || return (status=:missing, seed=nothing, reason="seed cache is absent")
    stored = try
        Logging.with_logger(Logging.NullLogger()) do
            JLD2.load(path, "cache")
        end
    catch err
        return (status=:corrupt, seed=nothing, reason=sprint(showerror, err))
    end
    compatible = try
        stored.schema_version == SAX_PD_RESCUE_CACHE_SCHEMA_VERSION &&
        stored.cache_kind == :pd_seed &&
        isequal(stored.settings_signature, _portable_sax_pd_rescue_settings(settings)) &&
        isequal(stored.model_signature,
                _sax_bifurcation_model_signature(model_p, settings.nmodes)) &&
        (isnothing(target_zeta) || isapprox(
            stored.seed.pd_checkpoint.zeta,
            float(target_zeta);
            atol=1e-10,
            rtol=0,
        ))
    catch
        false
    end
    compatible || return (
        status=:incompatible,
        seed=nothing,
        reason="seed cache has a different schema, model, settings, or target zeta",
    )
    return (status=:valid, seed=stored.seed, reason="compatible seed cache")
end

#endregion

#region DIRECTIONAL PD CONTINUATION AND FULL CHECKPOINTS

_sax_pd_formulation(formulation::Symbol) = formulation == :matrix_based ?
    BK.MinAugMatrixBased() : formulation == :minaug ? BK.MinAug() :
    throw(ArgumentError("formulation must be :matrix_based or :minaug"))

function _sax_pd_initial_vectors(wrapper, solution, parameters, state_dimension)
    jacobian = BK.jacobian(wrapper, solution, parameters)
    bordered = copy(jacobian)
    matrix_dimension = size(bordered, 1)
    deterministic_border = sin.(collect(1.0:matrix_dimension))
    bordered[end, :] .= deterministic_border
    bordered[:, end] .= reverse(deterministic_border)
    bordered[end, end] = 0
    bordered[end-state_dimension:end-1, 1:state_dimension] .= I(state_dimension)
    rhs = zeros(eltype(bordered), matrix_dimension)
    rhs[end] = 1
    right = (bordered \ rhs)[begin:end-1]
    left = (adjoint(bordered) \ rhs)[begin:end-1]
    right ./= norm(right)
    left ./= norm(left)
    return left, right
end

function _portable_sax_pd_augmented(value)
    if value isa BK.BorderedArray
        return (
            representation=:bordered,
            u=_portable_sax_pd_augmented(value.u),
            p=_portable_sax_pd_augmented(value.p),
        )
    elseif value isa AbstractArray
        return collect(value)
    elseif value isa Number
        return value
    elseif isnothing(value)
        return nothing
    end
    return string(value)
end

function _sax_pd_orbit_checkpoint(z,
                                  seed,
                                  formulation::Symbol,
                                  direction::Symbol,
                                  step::Integer)
    inner = BK.getvec(z)
    orbit = collect(float.(BK.getvec(inner)))
    gamma = float(BK.getp(inner))
    zeta = float(BK.getp(z))
    key = "pd_rescue_$(formulation)_$(direction)_step$(Int(step))_g$(round(gamma; digits=10))_z$(round(zeta; digits=10))"
    return (
        key=key,
        type=:pd,
        mode=Int(seed.mode),
        source_hopf_key=seed.source_hopf_key,
        specialpoint_index=0,
        localization_status=:accepted_rescue_step,
        localization_precision=NaN,
        gamma=gamma,
        zeta=zeta,
        floquet_angle=NaN,
        solution=orbit,
    )
end

function _sax_pd_checkpoint_callback(
        path::AbstractString,
        seed,
        model_p::NamedTuple,
        settings::SaxPDRescueSettings,
        formulation::Symbol,
        direction::Symbol,
        verbosity::Integer)
    checkpoints = Any[]
    if isfile(path)
        stored = try
            Logging.with_logger(Logging.NullLogger()) do
                JLD2.load(path, "checkpoint_cache")
            end
        catch
            nothing
        end
        reusable = try
            stored.schema_version == SAX_PD_RESCUE_CHECKPOINT_SCHEMA_VERSION &&
            stored.cache_kind == :pd_augmented_checkpoints &&
            stored.formulation == formulation &&
            stored.direction == direction &&
            isequal(stored.settings_signature,
                    _portable_sax_pd_rescue_settings(settings)) &&
            isequal(stored.model_signature,
                    _sax_bifurcation_model_signature(model_p, settings.nmodes))
        catch
            false
        end
        reusable && append!(checkpoints, stored.checkpoints)
    end
    step_offset = isempty(checkpoints) ? 0 : checkpoints[end].accepted_step
    started_ns = time_ns()

    callback = function (z, tangent, step, branch; kwargs...)
        state = get(kwargs, :state, nothing)
        BK.in_bisection(state) && return true
        global_step = step_offset + Int(step)
        orbit_checkpoint = _sax_pd_orbit_checkpoint(
            z, seed, formulation, direction, global_step)

        if verbosity > 0 && (step == 1 || step % settings.progress_every == 0)
            step_size = try
                float(state.ds)
            catch
                NaN
            end
            @info "PD rescue progress" formulation direction accepted_step=global_step local_step=step step_limit=settings.po_curve_max_steps gamma=orbit_checkpoint.gamma zeta=orbit_checkpoint.zeta ds=step_size elapsed_seconds=_sax_elapsed_seconds(started_ns)
        end

        if step == 1 || step % settings.checkpoint_every == 0
            push!(checkpoints, (
                accepted_step=global_step,
                saved_at_unix=time(),
                gamma=orbit_checkpoint.gamma,
                zeta=orbit_checkpoint.zeta,
                orbit_checkpoint=orbit_checkpoint,
                augmented_solution=_portable_sax_pd_augmented(z),
                augmented_tangent=_portable_sax_pd_augmented(tangent),
            ))
            checkpoint_cache = (
                schema_version=SAX_PD_RESCUE_CHECKPOINT_SCHEMA_VERSION,
                cache_kind=:pd_augmented_checkpoints,
                settings_signature=_portable_sax_pd_rescue_settings(settings),
                model_signature=_sax_bifurcation_model_signature(
                    model_p, settings.nmodes),
                formulation=formulation,
                direction=direction,
                latest_source_seed_key=seed.key,
                updated_at_unix=time(),
                checkpoints=copy(checkpoints),
            )
            _atomic_jld2_save(path; checkpoint_cache)
        end
        return true
    end
    return callback, checkpoints
end

function _sax_pd_rescue_options(settings::SaxPDRescueSettings,
                                direction::Symbol)
    direction in (:positive, :negative) || throw(ArgumentError(
        "direction must be :positive or :negative",
    ))
    signed_ds = direction == :positive ?
        abs(settings.po_curve_ds) : -abs(settings.po_curve_ds)
    options = BK.ContinuationPar(
        p_min=settings.zeta_range[1],
        p_max=settings.zeta_range[2],
        ds=signed_ds,
        dsmin=settings.po_curve_dsmin,
        dsmax=settings.po_curve_dsmax,
        max_steps=settings.po_curve_max_steps,
        detect_bifurcation=0,
        nev=2 + 2settings.nmodes,
        save_eigenvectors=settings.codim2_detection > 0,
        save_sol_every_step=settings.po_curve_save_sol_every_step,
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

"""
    continue_sax_pd_rescue(seed, model_p; settings, direction, formulation,
                           checkpoint_path, verbosity=1)

Continue one side of a PD curve.  This function always calls BifurcationKit
with `bothside=false`; the sign of `ds` is set explicitly by `direction`.
`checkpoint_path` receives atomic JLD2 snapshots of the complete nested
augmented solution and tangent, together with a portable periodic-orbit PD
checkpoint that can initialize a later rescue attempt.
"""
function continue_sax_pd_rescue(
        seed::NamedTuple,
        model_p::NamedTuple;
        settings::SaxPDRescueSettings=SaxPDRescueSettings(),
        direction::Symbol=:positive,
        formulation::Symbol=:matrix_based,
        checkpoint_path::AbstractString,
        verbosity::Int=1)
    _validate_sax_pd_rescue_settings(settings)
    seed.type == :pd || throw(ArgumentError("rescue seed must have type :pd"))
    jacobian_formulation = _sax_pd_formulation(formulation)
    bif_settings = _sax_pd_rescue_bifurcation_settings(settings)
    wrapper, parameters = _sax_periodic_wrapper(seed, model_p, bif_settings)
    collocation = BK.get_discretization(wrapper)
    state_dimension, _, _ = size(collocation)
    left, right = _sax_pd_initial_vectors(
        wrapper, seed.solution, parameters, state_dimension)
    initial = BK.BorderedArray(copy(seed.solution), seed.gamma)
    options = _sax_pd_rescue_options(settings, direction)
    callback, checkpoints = _sax_pd_checkpoint_callback(
        checkpoint_path,
        seed,
        model_p,
        settings,
        formulation,
        direction,
        verbosity,
    )

    started_ns = time_ns()
    branch = BK.continuation_pd(
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
        jacobian_ma=jacobian_formulation,
        compute_eigen_elements=settings.codim2_detection > 0,
        usehessian=false,
        bothside=false,
        normC=BK.norminf,
        finalise_solution=callback,
        verbosity=max(0, verbosity - 1),
        plot=false,
        kind=BK.PDPeriodicOrbitCont(),
    )
    gamma, zeta = _sax_branch_coordinates(branch)
    curve = (
        kind=:pd,
        gamma=gamma,
        zeta=zeta,
        mode=Int(seed.mode),
        frequency=NaN,
        source=(
            checkpoint=seed.key,
            formulation=formulation,
            direction=direction,
        ),
        diagnostics=_sax_branch_terminal_diagnostics(branch, :zeta),
    )
    return (
        curve=curve,
        branch=branch,
        seed=seed,
        formulation=formulation,
        direction=direction,
        elapsed_seconds=_sax_elapsed_seconds(started_ns),
        checkpoint_path=abspath(checkpoint_path),
        checkpoint_count=length(checkpoints),
    )
end

function save_sax_pd_rescue_result(path::AbstractString,
                                   result,
                                   model_p::NamedTuple;
                                   settings::SaxPDRescueSettings=SaxPDRescueSettings())
    portable = (
        curve=result.curve,
        seed=result.seed,
        formulation=result.formulation,
        direction=result.direction,
        elapsed_seconds=result.elapsed_seconds,
        checkpoint_path=result.checkpoint_path,
        checkpoint_count=result.checkpoint_count,
    )
    cache = (
        schema_version=SAX_PD_RESCUE_CACHE_SCHEMA_VERSION,
        cache_kind=:pd_directional_result,
        settings_signature=_portable_sax_pd_rescue_settings(settings),
        model_signature=_sax_bifurcation_model_signature(model_p, settings.nmodes),
        saved_at_unix=time(),
        result=portable,
    )
    _atomic_jld2_save(path; cache)
    return portable
end

function load_sax_pd_rescue_result(
        path::AbstractString,
        model_p::NamedTuple;
        settings::SaxPDRescueSettings=SaxPDRescueSettings(),
        formulation::Union{Nothing,Symbol}=nothing,
        direction::Union{Nothing,Symbol}=nothing,
        seed_key::Union{Nothing,AbstractString}=nothing)
    isfile(path) || return (status=:missing, result=nothing, reason="result cache is absent")
    stored = try
        Logging.with_logger(Logging.NullLogger()) do
            JLD2.load(path, "cache")
        end
    catch err
        return (status=:corrupt, result=nothing, reason=sprint(showerror, err))
    end
    compatible = try
        stored.schema_version == SAX_PD_RESCUE_CACHE_SCHEMA_VERSION &&
        stored.cache_kind == :pd_directional_result &&
        isequal(stored.settings_signature, _portable_sax_pd_rescue_settings(settings)) &&
        isequal(stored.model_signature,
                _sax_bifurcation_model_signature(model_p, settings.nmodes)) &&
        (isnothing(formulation) || stored.result.formulation == formulation) &&
        (isnothing(direction) || stored.result.direction == direction) &&
        (isnothing(seed_key) || stored.result.seed.key == seed_key)
    catch
        false
    end
    compatible || return (
        status=:incompatible,
        result=nothing,
        reason="result cache has a different schema, model, settings, seed, or run direction",
    )
    return (status=:valid, result=stored.result, reason="compatible rescue result")
end

"""
    load_sax_pd_rescue_checkpoint(path; accepted_step=nothing)

Load a portable PD orbit from the full augmented checkpoint file.  By default
the last saved accepted step is returned.  Passing `accepted_step` chooses the
latest checkpoint at or before that step.  The returned orbit can be supplied
directly to `continue_sax_pd_rescue`, including with the opposite direction or
the other minimally augmented formulation.
"""
function load_sax_pd_rescue_checkpoint(
        path::AbstractString;
        accepted_step::Union{Nothing,Integer}=nothing,
        model_p::Union{Nothing,NamedTuple}=nothing,
        settings::Union{Nothing,SaxPDRescueSettings}=nothing,
        formulation::Union{Nothing,Symbol}=nothing,
        direction::Union{Nothing,Symbol}=nothing)
    isfile(path) || return (status=:missing, checkpoint=nothing, record=nothing,
                            reason="checkpoint cache is absent")
    stored = try
        Logging.with_logger(Logging.NullLogger()) do
            JLD2.load(path, "checkpoint_cache")
        end
    catch err
        return (status=:corrupt, checkpoint=nothing, record=nothing,
                reason=sprint(showerror, err))
    end
    valid = try
        stored.schema_version == SAX_PD_RESCUE_CHECKPOINT_SCHEMA_VERSION &&
        stored.cache_kind == :pd_augmented_checkpoints &&
        !isempty(stored.checkpoints) &&
        (isnothing(formulation) || stored.formulation == formulation) &&
        (isnothing(direction) || stored.direction == direction) &&
        (isnothing(settings) || isequal(
            stored.settings_signature,
            _portable_sax_pd_rescue_settings(settings),
        )) &&
        (isnothing(model_p) || isnothing(settings) || isequal(
            stored.model_signature,
            _sax_bifurcation_model_signature(model_p, settings.nmodes),
        ))
    catch
        false
    end
    valid || return (status=:incompatible, checkpoint=nothing, record=nothing,
                     reason="checkpoint cache schema is incompatible or empty")
    candidates = isnothing(accepted_step) ? stored.checkpoints :
        [record for record in stored.checkpoints
         if record.accepted_step <= Int(accepted_step)]
    isempty(candidates) && return (
        status=:missing,
        checkpoint=nothing,
        record=nothing,
        reason="no saved checkpoint exists at or before accepted_step=$(accepted_step)",
    )
    record = candidates[end]
    return (status=:valid, checkpoint=record.orbit_checkpoint, record=record,
            reason="portable orbit and full augmented state are available")
end

#endregion

#region RESTARTABLE SUITE

function _sax_pd_hint(curves, kind::Symbol, mode::Integer, zeta::Real)
    candidates = [curve for curve in curves
                  if curve.kind == kind && curve.mode == mode &&
                     minimum(curve.zeta) <= zeta <= maximum(curve.zeta)]
    isempty(candidates) && return nothing
    return _interpolate_curve_gamma(first(candidates), zeta)
end

"""
    compute_sax_pd_rescue_suite(model_p, directory; settings, hint_curves,
                                overwrite=false, verbosity=1)

Run the complete isolated rescue matrix.  For every configured zeta it first
loads or independently computes a final-resolution PD seed.  It then runs both
directions for both minimally augmented formulations and saves each component
immediately.  A failed component is recorded and does not prevent the remaining
independent runs from starting.

`hint_curves` may contain old portable Hopf/PD curves.  They supply scalar
gamma selection hints only; no old periodic orbit or tangent is reused.
"""
function compute_sax_pd_rescue_suite(
        model_p::NamedTuple,
        directory::AbstractString;
        settings::SaxPDRescueSettings=SaxPDRescueSettings(),
        hint_curves=Any[],
        overwrite::Bool=false,
        verbosity::Int=1)
    _validate_sax_pd_rescue_settings(settings)
    mkpath(directory)
    seeds = Dict{Float64,Any}()
    results = Any[]
    failures = Any[]

    for zeta in settings.seed_zetas
        paths = sax_pd_rescue_cache_paths(directory, zeta)
        seed_state = overwrite ?
            (status=:missing, seed=nothing, reason="overwrite requested") :
            load_sax_pd_seed_cache(
                paths.seed, model_p; settings=settings, target_zeta=zeta)
        seed = if seed_state.status == :valid
            seed_state.seed
        else
            verbosity > 0 && @info(
                "Discovering independent final-resolution PD seed",
                zeta,
                collocation="$(settings.po_collocation_intervals)×$(settings.po_collocation_degree)",
            )
            try
                discovered = discover_sax_pd_seed(
                    model_p,
                    zeta;
                    mode=2,
                    gamma_hint=_sax_pd_hint(hint_curves, :hopf, 2, zeta),
                    pd_gamma_hint=_sax_pd_hint(hint_curves, :pd, 2, zeta),
                    settings=settings,
                    verbosity=verbosity,
                )
                save_sax_pd_seed_cache(
                    paths.seed, discovered, model_p; settings=settings)
            catch err
                push!(failures, (
                    stage=:seed,
                    zeta=float(zeta),
                    exception_type=Symbol(nameof(typeof(err))),
                    error=sprint(showerror, err),
                ))
                nothing
            end
        end
        isnothing(seed) && continue
        seeds[float(zeta)] = seed

        # Run the specialized formulation first because it avoids forming the
        # full augmented Jacobian.  The matrix-based calls remain independent
        # comparison runs and are still cached separately.
        for formulation in (:minaug, :matrix_based),
                direction in (:positive, :negative)
            component_paths = sax_pd_rescue_cache_paths(
                directory, zeta, formulation, direction)
            cached = overwrite ?
                (status=:missing, result=nothing, reason="overwrite requested") :
                load_sax_pd_rescue_result(
                    component_paths.result,
                    model_p;
                    settings=settings,
                    formulation=formulation,
                    direction=direction,
                )
            if cached.status == :valid
                push!(results, cached.result)
                continue
            end

            verbosity > 0 && @info(
                "Starting isolated PD rescue component",
                zeta,
                formulation,
                direction,
                ds=direction == :positive ? abs(settings.po_curve_ds) : -abs(settings.po_curve_ds),
                dsmin=settings.po_curve_dsmin,
                dsmax=settings.po_curve_dsmax,
                codim2_detection=settings.codim2_detection,
            )
            try
                restart_state = load_sax_pd_rescue_checkpoint(
                    component_paths.checkpoints;
                    model_p=model_p,
                    settings=settings,
                    formulation=formulation,
                    direction=direction,
                )
                run_seed = restart_state.status == :valid ?
                    restart_state.checkpoint : seed.pd_checkpoint
                if restart_state.status == :valid && verbosity > 0
                    @info "Resuming PD rescue from the latest portable orbit checkpoint" zeta formulation direction accepted_step=restart_state.record.accepted_step gamma=run_seed.gamma checkpoint_zeta=run_seed.zeta
                end
                raw = continue_sax_pd_rescue(
                    run_seed,
                    model_p;
                    settings=settings,
                    formulation=formulation,
                    direction=direction,
                    checkpoint_path=component_paths.checkpoints,
                    verbosity=verbosity,
                )
                push!(results, save_sax_pd_rescue_result(
                    component_paths.result, raw, model_p; settings=settings))
            catch err
                push!(failures, (
                    stage=:pd_curve,
                    zeta=float(zeta),
                    formulation=formulation,
                    direction=direction,
                    exception_type=Symbol(nameof(typeof(err))),
                    error=sprint(showerror, err),
                    checkpoint_path=abspath(component_paths.checkpoints),
                ))
            end
        end
    end

    suite = (
        schema_version=SAX_PD_RESCUE_CACHE_SCHEMA_VERSION,
        settings_signature=_portable_sax_pd_rescue_settings(settings),
        model_signature=_sax_bifurcation_model_signature(model_p, settings.nmodes),
        updated_at_unix=time(),
        seeds=seeds,
        results=results,
        failures=failures,
    )
    _atomic_jld2_save(joinpath(directory, "pd_rescue_manifest.jld2"); suite)
    return suite
end

"""Load every compatible component currently available in a rescue directory."""
function load_sax_pd_rescue_suite(
        model_p::NamedTuple,
        directory::AbstractString;
        settings::SaxPDRescueSettings=SaxPDRescueSettings(),
        include_checkpoints::Bool=false)
    seeds = Dict{Float64,Any}()
    results = Any[]
    status = Any[]
    for zeta in settings.seed_zetas
        paths = sax_pd_rescue_cache_paths(directory, zeta)
        seed_state = load_sax_pd_seed_cache(
            paths.seed, model_p; settings=settings, target_zeta=zeta)
        push!(status, (zeta=float(zeta), item=:seed, state=seed_state.status,
                       reason=seed_state.reason))
        seed_state.status == :valid || continue
        seeds[float(zeta)] = seed_state.seed
        for formulation in (:minaug, :matrix_based),
                direction in (:positive, :negative)
            component_paths = sax_pd_rescue_cache_paths(
                directory, zeta, formulation, direction)
            result_state = load_sax_pd_rescue_result(
                component_paths.result,
                model_p;
                settings=settings,
                formulation=formulation,
                direction=direction,
            )
            push!(status, (
                zeta=float(zeta),
                item=Symbol("$(formulation)_$(direction)"),
                state=result_state.status,
                reason=result_state.reason,
            ))
            if result_state.status == :valid
                push!(results, result_state.result)
            elseif include_checkpoints
                # An interrupted continuation has no ordinary result cache,
                # but its atomic checkpoint file still contains a valid,
                # decimated trace.  Expose that trace using the same small
                # interface as a completed result so plotting remains useful
                # while the expensive run is still in progress.
                checkpoint_state = load_sax_pd_rescue_checkpoint(
                    component_paths.checkpoints;
                    model_p=model_p,
                    settings=settings,
                    formulation=formulation,
                    direction=direction,
                )
                if checkpoint_state.status == :valid
                    records = try
                        stored = Logging.with_logger(Logging.NullLogger()) do
                            JLD2.load(component_paths.checkpoints, "checkpoint_cache")
                        end
                        stored.checkpoints
                    catch
                        Any[]
                    end
                    if !isempty(records)
                        push!(results, (
                            formulation=formulation,
                            direction=direction,
                            provisional=true,
                            accepted_step=checkpoint_state.record.accepted_step,
                            curve=(
                                gamma=[float(r.orbit_checkpoint.gamma) for r in records],
                                zeta=[float(r.orbit_checkpoint.zeta) for r in records],
                            ),
                        ))
                        push!(status, (
                            zeta=float(zeta),
                            item=Symbol("$(formulation)_$(direction)_checkpoint"),
                            state=:checkpoint,
                            reason="using $(checkpoint_state.record.accepted_step) saved accepted steps",
                        ))
                    end
                end
            end
        end
    end
    return (seeds=seeds, results=results, status=status)
end

"""
    plot_sax_pd_rescue_comparison(suite; reference=nothing)

Plot every available directional rescue component in the gamma-zeta plane.
Matrix-based and specialized minimally augmented formulations use different
colors; positive and negative calls use solid and dashed lines.  If supplied,
`reference` is a portable ordinary bifurcation result whose Hopf and original
PD curves are drawn in the background.
"""
function plot_sax_pd_rescue_comparison(suite; reference=nothing)
    p = plot(
        ;
        xlabel="gamma",
        ylabel="zeta",
        xlim=(0.30, 1.00),
        ylim=(0.001, 0.99),
        legend=:outerright,
        title="Directional period-doubling rescue",
        size=(920, 620),
    )
    if !isnothing(reference)
        for curve in reference.hopf_curves
            plot!(p, curve.gamma, curve.zeta;
                  color=:gray45, linewidth=1.2, alpha=0.65,
                  label="")
        end
        for (index, curve) in enumerate(reference.pd_curves)
            plot!(p, curve.gamma, curve.zeta;
                  color=:black, linewidth=2, alpha=0.65,
                  linestyle=:dot,
                  label=index == 1 ? "original Pilot PD" : "")
        end
    end

    seen = Set{Tuple{Symbol,Symbol}}()
    for result in suite.results
        formulation = result.formulation
        direction = result.direction
        identity = (formulation, direction)
        color = formulation == :matrix_based ? :darkorange2 : :dodgerblue3
        provisional = get(result, :provisional, false)
        linestyle = provisional ? :dot :
            (direction == :positive ? :solid : :dash)
        label = identity in seen ? "" :
            "$(formulation), $(direction) $(provisional ? "checkpoint" : "ds")"
        push!(seen, identity)
        plot!(p, result.curve.gamma, result.curve.zeta;
              color=color, linestyle=linestyle, linewidth=2,
              alpha=provisional ? 0.65 : 0.8, label=label)
    end

    ordered_seeds = sort(collect(values(suite.seeds));
                         by=seed -> seed.pd_checkpoint.zeta)
    for (index, seed) in enumerate(ordered_seeds)
        point = seed.pd_checkpoint
        scatter!(p, [point.gamma], [point.zeta];
                 color=:black, marker=:diamond, markersize=5,
                 markerstrokewidth=0.7,
                 label=index == 1 ? "independent final PD seeds" : "")
    end
    return p
end

#endregion
