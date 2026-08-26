# This file is included by rt_sax_experiment_analysis.jl after the simulation
# and mode-persistence functions.  It deliberately uses the qualified `BK.`
# prefix: BifurcationKit exports many generic names (`continuation`, `newton`,
# `get_normal_form`, ...), and qualification makes the nontrivial calls below
# much easier to audit in a notebook.

import BifurcationKit as BK
import Logging

#region MODEL BIFURCATION SETTINGS AND VECTOR FIELD

"""
Numerical controls for the two-parameter saxophone bifurcation analysis.

The defaults target the 18-state, eight-acoustic-mode model.  `zeta_seeds`
contains the horizontal slices from which one-parameter equilibrium branches
are first continued in `gamma`.  Continuation is local, so using several seeds
is the inexpensive way to reduce the chance of missing disconnected Hopf
curves.  It is not a mathematical proof that every component was found.

The continuation stages have separate step sizes because periodic-orbit and
codimension-two augmented systems are appreciably harder than equilibria.
"""
Base.@kwdef struct SaxBifurcationSettings
    nmodes::Int = 8
    gamma_range::Tuple{Float64,Float64} = (0.30, 0.99)
    zeta_range::Tuple{Float64,Float64} = (0.001, 0.99)
    zeta_seeds::Tuple{Vararg{Float64}} = (0.25, 0.55, 0.85)

    equilibrium_ds::Float64 = 2e-3
    equilibrium_dsmax::Float64 = 1e-2
    equilibrium_max_steps::Int = 900

    hopf_ds::Float64 = 1e-3
    hopf_dsmax::Float64 = 8e-3
    hopf_max_steps::Int = 700

    po_ds::Float64 = 1e-3
    po_dsmax::Float64 = 6e-3
    po_max_steps::Int = 500
    po_collocation_intervals::Int = 40
    po_collocation_degree::Int = 4
    po_linear_solver::Symbol = :condensed
    po_save_sol_every_step::Int = 10

    po_curve_ds::Float64 = 5e-4
    po_curve_dsmax::Float64 = 4e-3
    po_curve_max_steps::Int = 500
    po_curve_save_sol_every_step::Int = 10
    gh_fold_predictor_amplitude::Float64 = 1e-2

    newton_tol::Float64 = 1e-10
    stability_tol::Float64 = 1e-8
    smoothness_tol::Float64 = 1e-7
    validation_residual_tol::Float64 = 1e-8
    validation_eigen_tol::Float64 = 2e-6
    validation_frequency_separation::Float64 = 1e-4
    validation_transversality_tol::Float64 = 1e-5
    validation_parameter_step::Float64 = 1e-5
    validation_lyapunov_tol::Float64 = 1e-4

    # The contact term and the max(reed opening, 0) flow law are continuous at
    # reed closure.  A periodic orbit may therefore cross that surface
    # transversely and use the piecewise Jacobian almost everywhere.  Grazing
    # contact still requires a nonsmooth bifurcation method and is flagged in
    # the orbit record.  Flow reversal remains strict because sqrt(abs(d)) has
    # an unbounded derivative at d=0.
    allow_transverse_reed_contact::Bool = true
    periodic_grazing_velocity_tol::Float64 = 1e-5
end

"""
    sax_bifurcation_settings(profile=:final; kwargs...)

Construct the continuation settings for either the exploratory `:pilot`
profile or the publication-resolution `:final` profile. The pilot uses a
`25 × 3` periodic-orbit collocation mesh; the final profile uses `40 × 4`.
Both use BifurcationKit's collocation-condensation solver by default.

Keyword arguments override the selected profile, which is useful for focused
convergence studies and small smoke tests.
"""
function sax_bifurcation_settings(profile::Symbol=:final; kwargs...)
    profile in (:pilot, :final) || throw(ArgumentError(
        "bifurcation profile must be :pilot or :final, found $profile",
    ))
    resolution = profile == :pilot ?
        (po_collocation_intervals=25, po_collocation_degree=3) :
        (po_collocation_intervals=40, po_collocation_degree=4)
    return SaxBifurcationSettings(; merge(resolution, (; kwargs...))...)
end

_sax_elapsed_seconds(started_ns::UInt64) =
    round((time_ns() - started_ns) / 1e9; digits=1)

"""
    _sax_continuation_progress(stage, step_limit, verbosity; every_steps)

Create a throttled BifurcationKit `finalise_solution` callback.  At verbosity
level 1 it emits an `@info` record after the first accepted continuation step
and then every `every_steps` accepted steps.  Pluto displays these records
while the cell is still running, unlike ordinary `println` output sent to the
Julia process.  The callback always returns `true`, so it observes progress
without changing the continuation.
"""
function _sax_continuation_progress(
        stage::AbstractString,
        step_limit::Integer,
        verbosity::Integer;
        every_steps::Integer)
    every_steps > 0 || throw(ArgumentError("every_steps must be positive"))
    started_ns = time_ns()

    return function (z, tangent, step, branch; kwargs...)
        if verbosity > 0 && (step == 1 || step % every_steps == 0)
            state = get(kwargs, :state, nothing)
            parameter = try
                isnothing(state) ? float(z.p) : float(BK.getp(state))
            catch
                NaN
            end
            @info "Sax continuation progress" stage accepted_step=step step_limit parameter elapsed_seconds=_sax_elapsed_seconds(started_ns)
        end
        return true
    end
end

"""
    sax_bifurcation_parameters(model_p; gamma, zeta, nmodes=8)

Create the immutable parameter `NamedTuple` used by BifurcationKit.

The continuation parameters must be named fields so that
`BK.@optic _.gamma` and `BK.@optic _.zeta` can independently vary the two
coordinates.  For `nmodes == 8` the corresponding state and Jacobian have
dimension `2 + 2nmodes == 18`.
"""
function sax_bifurcation_parameters(model_p::NamedTuple;
                                    gamma::Real,
                                    zeta::Real,
                                    nmodes::Int=8)
    nmodes > 0 || throw(ArgumentError("nmodes must be positive"))
    length(model_p.α) >= nmodes || throw(ArgumentError("model_p.α has fewer than $nmodes entries"))
    length(model_p.ω) >= nmodes || throw(ArgumentError("model_p.ω has fewer than $nmodes entries"))
    length(model_p.C) >= nmodes || throw(ArgumentError("model_p.C has fewer than $nmodes entries"))

    alpha = collect(float.(model_p.α[1:nmodes]))
    omega = collect(float.(model_p.ω[1:nmodes]))
    coupling = collect(float.(model_p.C[1:nmodes]))
    all(alpha .> 0) || throw(ArgumentError("all modal damping coefficients α must be positive"))
    all(omega .> 0) || throw(ArgumentError("all modal angular frequencies ω must be positive"))

    parameters = (
        gamma = float(gamma),
        zeta = float(zeta),
        alpha = alpha,
        omega = omega,
        coupling = coupling,
        nmodes = nmodes,
        reed_damping = 4.224,
        reed_stiffness = 17.842176,
        contact_stiffness = hasproperty(model_p, :contact_stiffness) ?
            float(model_p.contact_stiffness) : 100.0,
    )
    return hasproperty(model_p, :sax_regularization) ?
        merge(parameters, (sax_regularization=model_p.sax_regularization,)) :
        parameters
end

"""
    sax_bifurcation_residual(u, p)

Out-of-place equilibrium residual `F(u,p)` for BifurcationKit.

For an ordinary parameter tuple this is algebraically the same vector field as
`saxRN!`. When the parameters carry a `sax_regularization` signature, it
dispatches to the parallel Colinot-regularized vector field. The first two
states are reed displacement/velocity. States `2m+1, 2m+2` are the pressure
and quadrature coordinates of acoustic mode `m`.
"""
function sax_bifurcation_residual(u::AbstractVector, p::NamedTuple)
    hasproperty(p, :sax_regularization) &&
        return sax_regularized_bifurcation_residual(u, p)
    n = 2 + 2p.nmodes
    length(u) == n || throw(DimensionMismatch("expected $n states, received $(length(u))"))

    T = promote_type(eltype(u), typeof(p.gamma), typeof(p.zeta))
    du = zeros(T, n)

    pressure = sum(@view u[3:2:end])
    reed_coordinate = real(u[1]) + one(real(u[1]))
    closed_part = min(reed_coordinate, zero(reed_coordinate))
    contact_force = p.contact_stiffness * closed_part^2 * (one(T) - u[2])

    opening = max(reed_coordinate, zero(reed_coordinate))
    pressure_drop = p.gamma - pressure
    flow = p.zeta * opening * sign(pressure_drop) * sqrt(abs(pressure_drop))

    du[1] = u[2]
    du[2] = -p.reed_damping * u[2] +
            p.reed_stiffness * (pressure - p.gamma - u[1] + contact_force)

    @inbounds for mode in 1:p.nmodes
        i = 2mode + 1
        du[i] = -p.alpha[mode] * u[i] - 2p.omega[mode] * u[i + 1] +
                2p.coupling[mode] * flow
        du[i + 1] = -p.alpha[mode] * u[i + 1] + 0.5p.omega[mode] * u[i]
    end
    return du
end

"""
    sax_smoothness_margins(u, p)

Return signed locations and absolute distances to the two nonsmooth surfaces.

`reed_opening > 0` means the reed is open.  `pressure_drop > 0` means forward
flow. For the historical piecewise model, classical equilibrium, Hopf,
period-doubling, and Neimark-Sacker continuation is valid only while both
distances remain safely nonzero. For a model tagged by `regularize_sax_model`,
the same fields are retained as physical proximity diagnostics but are no
longer differentiability restrictions.
"""
function sax_smoothness_margins(u::AbstractVector, p::NamedTuple)
    pressure = sum(@view u[3:2:end])
    reed_opening = real(u[1]) + 1
    pressure_drop = real(p.gamma - pressure)
    return (
        reed_opening = float(reed_opening),
        pressure_drop = float(pressure_drop),
        reed_distance = float(abs(reed_opening)),
        flow_reversal_distance = float(abs(pressure_drop)),
        open_reed = reed_opening > 0,
        forward_flow = pressure_drop > 0,
        regularized = hasproperty(p, :sax_regularization),
        eta_opening = hasproperty(p, :sax_regularization) ?
            float(p.sax_regularization.eta_opening) : 0.0,
        eta_flow = hasproperty(p, :sax_regularization) ?
            float(p.sax_regularization.eta_flow) : 0.0,
    )
end

"""
    sax_bifurcation_jacobian!(J, u, p; smoothness_tol=1e-10, strict=true)

Fill the explicit analytic Jacobian `∂F/∂u`.

For eight modes this is the requested `18 × 18` matrix.  Let
`P = sum(u[3:2:end])`, `d = gamma-P`, and `r = u[1]+1`.  Away from `d=0`,
the derivative of `sign(d)sqrt(abs(d))` is `1/(2sqrt(abs(d)))`.  This creates
a rank-one coupling from every modal pressure coordinate into every driven
modal equation.  Reed contact contributes only on the `r<0` branch.

With `strict=true`, a `DomainError` is raised inside `smoothness_tol` of either
nonsmooth surface.  This is intentional: inserting an arbitrary derivative at
a corner could create a spurious Hopf or Floquet crossing.

When `p` carries a `sax_regularization` signature, this function dispatches to
the globally smooth analytic Jacobian. The strict surface checks then do not
apply because both formerly singular derivatives are finite.
"""
function sax_bifurcation_jacobian!(J::AbstractMatrix,
                                   u::AbstractVector,
                                   p::NamedTuple;
                                   smoothness_tol::Real=1e-10,
                                   strict::Bool=true,
                                   strict_reed::Bool=strict,
                                   strict_flow::Bool=strict)
    hasproperty(p, :sax_regularization) &&
        return sax_regularized_bifurcation_jacobian!(J, u, p)
    n = 2 + 2p.nmodes
    size(J) == (n, n) || throw(DimensionMismatch("Jacobian must have size ($n, $n)"))
    length(u) == n || throw(DimensionMismatch("expected $n states, received $(length(u))"))

    pressure = sum(@view u[3:2:end])
    reed_coordinate = real(u[1]) + one(real(u[1]))
    pressure_drop = real(p.gamma - pressure)

    if strict_reed && abs(reed_coordinate) <= smoothness_tol
        throw(DomainError(reed_coordinate,
            "analytic Jacobian is undefined at the reed-contact surface u[1]+1=0"))
    end
    if strict_flow && abs(pressure_drop) <= smoothness_tol
        throw(DomainError(pressure_drop,
            "analytic Jacobian is undefined at the flow-reversal surface gamma-P=0"))
    end

    fill!(J, zero(eltype(J)))

    # Contact force Fc = kc*r^2*(1-u2) for r<0, and zero for r>0.
    if reed_coordinate < 0
        dcontact_du1 = 2p.contact_stiffness * reed_coordinate * (1 - u[2])
        dcontact_du2 = -p.contact_stiffness * reed_coordinate^2
    else
        dcontact_du1 = zero(eltype(J))
        dcontact_du2 = zero(eltype(J))
    end

    # Flow q = zeta*max(r,0)*sign(d)*sqrt(abs(d)).
    # The fallback derivative at d=0 is only reachable with strict=false and is
    # useful for diagnostics, never for a reported validated bifurcation.
    opening = max(reed_coordinate, zero(reed_coordinate))
    root_flow = sign(pressure_drop) * sqrt(abs(pressure_drop))
    droot_dd = abs(pressure_drop) > smoothness_tol ?
               inv(2sqrt(abs(pressure_drop))) : zero(eltype(J))
    dflow_du1 = p.zeta * (reed_coordinate > 0 ? one(eltype(J)) : zero(eltype(J))) * root_flow
    dflow_dpressure_state = -p.zeta * opening * droot_dd

    J[1, 2] = 1
    J[2, 1] = p.reed_stiffness * (-1 + dcontact_du1)
    J[2, 2] = -p.reed_damping + p.reed_stiffness * dcontact_du2
    @inbounds for source_mode in 1:p.nmodes
        J[2, 2source_mode + 1] = p.reed_stiffness
    end

    @inbounds for mode in 1:p.nmodes
        i = 2mode + 1
        drive_gain = 2p.coupling[mode]

        J[i, 1] = drive_gain * dflow_du1
        J[i, i] += -p.alpha[mode]
        J[i, i + 1] = -2p.omega[mode]
        for source_mode in 1:p.nmodes
            J[i, 2source_mode + 1] += drive_gain * dflow_dpressure_state
        end

        J[i + 1, i] = 0.5p.omega[mode]
        J[i + 1, i + 1] = -p.alpha[mode]
    end
    return J
end

"""Allocate and return the explicit analytic saxophone equilibrium Jacobian."""
function sax_bifurcation_jacobian(u::AbstractVector,
                                  p::NamedTuple;
                                  kwargs...)
    T = promote_type(eltype(u), typeof(p.gamma), typeof(p.zeta))
    J = zeros(T, length(u), length(u))
    return sax_bifurcation_jacobian!(J, u, p; kwargs...)
end

"""
    check_sax_bifurcation_jacobian(u, p; step=1e-7)

Compare the analytic Jacobian with a centered finite-difference Jacobian.
The returned relative Frobenius error is the primary regression diagnostic;
`maximum_absolute_error` helps identify a single incorrectly filled entry.
"""
function check_sax_bifurcation_jacobian(u::AbstractVector,
                                        p::NamedTuple;
                                        step::Real=1e-7,
                                        smoothness_tol::Real=1e-10)
    margins = sax_smoothness_margins(u, p)
    if !margins.regularized
        minimum((margins.reed_distance, margins.flow_reversal_distance)) > 10step ||
            throw(ArgumentError("finite-difference stencil is too close to a nonsmooth surface"))
    end

    analytic = sax_bifurcation_jacobian(u, p;
                                        smoothness_tol=smoothness_tol,
                                        strict=true)
    finite_difference = similar(analytic)
    for j in eachindex(u)
        h = float(step) * max(1.0, abs(float(real(u[j]))))
        up = copy(u); up[j] += h
        um = copy(u); um[j] -= h
        finite_difference[:, j] .=
            (sax_bifurcation_residual(up, p) .- sax_bifurcation_residual(um, p)) ./ (2h)
    end

    delta = analytic - finite_difference
    scale = max(norm(finite_difference), eps(Float64))
    return (
        analytic = analytic,
        finite_difference = finite_difference,
        maximum_absolute_error = maximum(abs, delta),
        relative_frobenius_error = norm(delta) / scale,
        margins = margins,
    )
end

function _validate_sax_bifurcation_settings(settings::SaxBifurcationSettings)
    settings.nmodes > 0 || throw(ArgumentError("settings.nmodes must be positive"))
    settings.gamma_range[1] < settings.gamma_range[2] ||
        throw(ArgumentError("gamma_range must be increasing"))
    settings.zeta_range[1] < settings.zeta_range[2] ||
        throw(ArgumentError("zeta_range must be increasing"))
    all(settings.zeta_range[1] .<= settings.zeta_seeds .<= settings.zeta_range[2]) ||
        throw(ArgumentError("every zeta seed must lie inside zeta_range"))
    settings.po_collocation_intervals >= 5 ||
        throw(ArgumentError("at least five collocation intervals are required"))
    settings.po_collocation_degree >= 2 ||
        throw(ArgumentError("collocation degree must be at least two"))
    settings.po_linear_solver in (:condensed, :dense) ||
        throw(ArgumentError("po_linear_solver must be :condensed or :dense"))
    settings.po_save_sol_every_step >= 0 ||
        throw(ArgumentError("po_save_sol_every_step must be non-negative"))
    settings.po_curve_save_sol_every_step >= 0 ||
        throw(ArgumentError("po_curve_save_sol_every_step must be non-negative"))
    settings.gh_fold_predictor_amplitude > 0 ||
        throw(ArgumentError("gh_fold_predictor_amplitude must be positive"))
    settings.validation_lyapunov_tol > 0 ||
        throw(ArgumentError("validation_lyapunov_tol must be positive"))
    return settings
end

function _sax_bifurcation_problem(model_p::NamedTuple,
                                  gamma::Real,
                                  zeta::Real,
                                  settings::SaxBifurcationSettings)
    p = sax_bifurcation_parameters(model_p;
                                   gamma=gamma,
                                   zeta=zeta,
                                   nmodes=settings.nmodes)
    u0, fixed_residual = _estimate_fixed_point(gamma, zeta, model_p;
                                               nmodes=settings.nmodes,
                                               pert=0.0)
    fixed_residual <= max(100settings.newton_tol, settings.validation_residual_tol) ||
        @warn "analytic fixed-point seed has a larger than expected residual" gamma zeta fixed_residual

    # The record deliberately ignores its second argument.  For equilibrium
    # continuation it is gamma; for a Hopf curve BifurcationKit supplies zeta.
    # BifurcationKit itself records the active parameter fields.
    record_state = (u, _parameter; kwargs...) -> (
        state_norm = norm(u),
        acoustic_pressure = sum(@view u[3:2:end]),
    )
    jac = (u, pars) -> sax_bifurcation_jacobian(
        u, pars;
        smoothness_tol=settings.smoothness_tol,
        strict=true,
        strict_reed=!settings.allow_transverse_reed_contact,
        strict_flow=true,
    )
    problem = BK.BifurcationProblem(
        sax_bifurcation_residual,
        u0,
        p,
        (BK.@optic _.gamma);
        J=jac,
        record_from_solution=record_state,
    )
    return problem, u0, fixed_residual
end

#endregion

#region EQUILIBRIUM, HOPF, AND DOUBLE-HOPF CONTINUATION

"""
    continue_sax_equilibria(model_p, zeta; settings=..., verbosity=0)

Continue the equilibrium branch in `gamma` at fixed `zeta` and precisely locate
Hopf crossings.  All 18 equilibrium eigenvalues are retained by default, which
is inexpensive for this model and necessary for reliable double-Hopf detection.
"""
function continue_sax_equilibria(model_p::NamedTuple,
                                 zeta::Real;
                                 settings::SaxBifurcationSettings=SaxBifurcationSettings(),
                                 verbosity::Int=0)
    _validate_sax_bifurcation_settings(settings)
    settings.zeta_range[1] <= zeta <= settings.zeta_range[2] ||
        throw(ArgumentError("zeta=$zeta lies outside settings.zeta_range"))

    gamma0 = sum(settings.gamma_range) / 2
    problem, u0, fixed_residual = _sax_bifurcation_problem(model_p, gamma0, zeta, settings)
    nstate = length(u0)
    nstate == 18 || settings.nmodes != 8 ||
        error("the eight-mode bifurcation problem must have exactly 18 states")

    options = BK.ContinuationPar(
        p_min=settings.gamma_range[1],
        p_max=settings.gamma_range[2],
        ds=settings.equilibrium_ds,
        dsmin=min(1e-6, abs(settings.equilibrium_ds) / 100),
        dsmax=settings.equilibrium_dsmax,
        max_steps=settings.equilibrium_max_steps,
        detect_bifurcation=3,
        detect_fold=true,
        nev=nstate,
        save_eigenvectors=true,
        save_sol_every_step=1,
        n_inversion=8,
        max_bisection_steps=35,
        tol_stability=settings.stability_tol,
        newton_options=BK.NewtonPar(
            tol=settings.newton_tol,
            max_iterations=25,
            verbose=false,
        ),
    )

    branch = BK.continuation(
        problem,
        BK.PALC(),
        options;
        bothside=true,
        normC=BK.norminf,
        finalise_solution=_sax_continuation_progress(
            "equilibrium branch",
            settings.equilibrium_max_steps,
            verbosity;
            every_steps=25,
        ),
        # Level 1 is reserved for concise Pluto-visible progress.  Level 2 and
        # above also enable BifurcationKit's detailed stdout iteration trace.
        verbosity=max(0, verbosity - 1),
        plot=false,
    )
    return (
        branch=branch,
        seed=(gamma=gamma0, zeta=float(zeta)),
        fixed_state=u0,
        fixed_residual=fixed_residual,
        hopf_specialpoint_indices=findall(sp -> sp.type == :hopf, branch.specialpoint),
    )
end

"""
    continue_sax_hopf_curve(equilibrium_branch, hopf_index; settings=..., verbosity=0)

Continue one equilibrium Hopf point in the second parameter `zeta`.  The result
is a codimension-one curve in `(gamma,zeta)`.  BifurcationKit's minimally
augmented Hopf system is used, with the explicit state Jacobian supplied by this
file.  Codimension-two detection is enabled so `:hh` points are localized on
the curve.
"""
function continue_sax_hopf_curve(equilibrium_branch,
                                 hopf_index::Integer;
                                 settings::SaxBifurcationSettings=SaxBifurcationSettings(),
                                 verbosity::Int=0)
    _validate_sax_bifurcation_settings(settings)
    1 <= hopf_index <= length(equilibrium_branch.specialpoint) ||
        throw(BoundsError(equilibrium_branch.specialpoint, hopf_index))
    equilibrium_branch.specialpoint[hopf_index].type == :hopf ||
        throw(ArgumentError("special point $hopf_index is not a Hopf point"))

    options = BK.ContinuationPar(
        p_min=settings.zeta_range[1],
        p_max=settings.zeta_range[2],
        ds=settings.hopf_ds,
        dsmin=min(1e-6, abs(settings.hopf_ds) / 100),
        dsmax=settings.hopf_dsmax,
        max_steps=settings.hopf_max_steps,
        detect_bifurcation=0,
        nev=2 + 2settings.nmodes,
        save_eigenvectors=true,
        save_sol_every_step=1,
        tol_stability=settings.stability_tol,
        newton_options=BK.NewtonPar(
            tol=settings.newton_tol,
            max_iterations=30,
            verbose=false,
        ),
    )

    return BK.continuation(
        equilibrium_branch,
        Int(hopf_index),
        (BK.@optic _.zeta),
        options;
        detect_codim2_bifurcation=2,
        update_minaug_every_step=1,
        start_with_eigen=true,
        jacobian_ma=BK.MinAug(),
        bdlinsolver=BK.MatrixBLS(),
        bothside=true,
        normC=BK.norminf,
        finalise_solution=_sax_continuation_progress(
            "Hopf curve",
            settings.hopf_max_steps,
            verbosity;
            every_steps=10,
        ),
        verbosity=max(0, verbosity - 1),
        plot=false,
    )
end

function _upper_half_plane_pairs(values::AbstractVector{<:Complex}; imag_tol::Real=1e-8)
    candidates = [lambda for lambda in values if imag(lambda) > imag_tol]
    sort!(candidates; by=lambda -> (abs(real(lambda)), imag(lambda)))
    return candidates
end

function _match_eigenvalue(values::AbstractVector{<:Complex}, target::Complex)
    isempty(values) && error("empty eigenvalue spectrum")
    return values[argmin(abs.(values .- target))]
end

function _equilibrium_spectrum(model_p::NamedTuple,
                               gamma::Real,
                               zeta::Real,
                               settings::SaxBifurcationSettings)
    p = sax_bifurcation_parameters(model_p;
                                   gamma=gamma,
                                   zeta=zeta,
                                   nmodes=settings.nmodes)
    u, residual = _estimate_fixed_point(gamma, zeta, model_p;
                                        nmodes=settings.nmodes,
                                        pert=0.0)
    J = sax_bifurcation_jacobian(u, p;
                                 smoothness_tol=settings.smoothness_tol,
                                 strict=true)
    return eigvals(J), u, residual, p
end

"""
    validate_sax_double_hopf(hopf_branch, specialpoint_index, model_p; settings=...)

Numerically validate a BifurcationKit `:hh` candidate using independent checks:

1. The equilibrium residual is small
2. Two distinct conjugate eigenvalue pairs have real parts near zero
3. The point is away from both nonsmooth switching surfaces
4. The `2 × 2` matrix of the two growth-rate derivatives with respect to
   `(gamma,zeta)` is nonsingular
5. The Hopf-Hopf normal form can be evaluated

This is a strong floating-point validation, not an interval-arithmetic proof.
The `transversality_matrix` is especially important: its determinant verifies
that the two neutral pairs cross independently in the two-parameter plane.
"""
function validate_sax_double_hopf(hopf_branch,
                                  specialpoint_index::Integer,
                                  model_p::NamedTuple;
                                  settings::SaxBifurcationSettings=SaxBifurcationSettings())
    1 <= specialpoint_index <= length(hopf_branch.specialpoint) ||
        throw(BoundsError(hopf_branch.specialpoint, specialpoint_index))
    special = hopf_branch.specialpoint[specialpoint_index]
    special.type == :hh || throw(ArgumentError("special point is $(special.type), not :hh"))

    # For a Hopf curve continued with lens1=gamma and lens2=zeta,
    # BifurcationKit stores gamma in the bordered solution and zeta in `param`.
    gamma = float(special.x.p1)
    zeta = float(special.param)
    spectrum, equilibrium, residual, p = _equilibrium_spectrum(
        model_p, gamma, zeta, settings)
    candidates = _upper_half_plane_pairs(complex.(spectrum))

    reasons = String[]
    special.status == :converged || push!(reasons, "BifurcationKit localization status is $(special.status)")
    residual <= settings.validation_residual_tol ||
        push!(reasons, "equilibrium residual $residual exceeds tolerance")

    if length(candidates) < 2
        push!(reasons, "fewer than two positive-frequency eigenvalue pairs were found")
        critical = ComplexF64[]
    else
        critical = ComplexF64[candidates[1], candidates[2]]
        maximum(abs.(real.(critical))) <= settings.validation_eigen_tol ||
            push!(reasons, "the two candidate growth rates are not both sufficiently close to zero")
        abs(imag(critical[1]) - imag(critical[2])) >= settings.validation_frequency_separation ||
            push!(reasons, "the two candidate Hopf frequencies are not distinct")
    end

    margins = sax_smoothness_margins(equilibrium, p)
    margins.reed_distance > settings.smoothness_tol ||
        push!(reasons, "point is too close to reed closure")
    margins.flow_reversal_distance > settings.smoothness_tol ||
        push!(reasons, "point is too close to flow reversal")

    transversality = fill(NaN, 2, 2)
    determinant = NaN
    if length(critical) == 2
        h = settings.validation_parameter_step
        for (j, (dgamma, dzeta)) in enumerate(((h, 0.0), (0.0, h)))
            plus, _, _, _ = _equilibrium_spectrum(
                model_p, gamma + dgamma, zeta + dzeta, settings)
            minus, _, _, _ = _equilibrium_spectrum(
                model_p, gamma - dgamma, zeta - dzeta, settings)
            plus = complex.(plus); minus = complex.(minus)
            for pair in 1:2
                lambda_plus = _match_eigenvalue(plus, critical[pair])
                lambda_minus = _match_eigenvalue(minus, critical[pair])
                transversality[pair, j] = real(lambda_plus - lambda_minus) / (2h)
            end
        end
        determinant = det(transversality)
        abs(determinant) >= settings.validation_transversality_tol ||
            push!(reasons, "the two growth-rate gradients are nearly linearly dependent")
    end

    normal_form = nothing
    normal_form_error = nothing
    try
        # Finite-difference multilinear forms are chosen because the vector
        # field is only piecewise smooth; AD through max/min is less transparent.
        normal_form = BK.get_normal_form(
            hopf_branch,
            Int(specialpoint_index);
            nev=2 + 2settings.nmodes,
            autodiff=false,
            verbose=false,
        )
    catch err
        normal_form_error = sprint(showerror, err)
        push!(reasons, "Hopf-Hopf normal-form evaluation failed")
    end

    return (
        valid=isempty(reasons),
        gamma=gamma,
        zeta=zeta,
        frequencies=imag.(critical),
        eigenvalues=critical,
        residual=residual,
        margins=margins,
        transversality_matrix=transversality,
        transversality_determinant=determinant,
        localization_status=special.status,
        normal_form=normal_form,
        normal_form_error=normal_form_error,
        reasons=reasons,
        specialpoint_index=Int(specialpoint_index),
    )
end

"""
    validate_sax_generalized_hopf(hopf_branch, specialpoint_index, model_p; settings)

Validate a generalized-Hopf (Bautin) point localized on a Hopf curve. The
equilibrium residual and switching-surface margins are checked independently,
and BifurcationKit's Bautin normal form is evaluated to obtain the second
Lyapunov coefficient. This remains a floating-point validation rather than an
interval proof.
"""
function validate_sax_generalized_hopf(
        hopf_branch,
        specialpoint_index::Integer,
        model_p::NamedTuple;
        settings::SaxBifurcationSettings=SaxBifurcationSettings())
    1 <= specialpoint_index <= length(hopf_branch.specialpoint) ||
        throw(BoundsError(hopf_branch.specialpoint, specialpoint_index))
    special = hopf_branch.specialpoint[Int(specialpoint_index)]
    special.type == :gh || throw(ArgumentError(
        "special point is $(special.type), not :gh",
    ))

    gamma = float(special.x.p1)
    zeta = float(special.param)
    spectrum, equilibrium, residual, parameters = _equilibrium_spectrum(
        model_p, gamma, zeta, settings)
    frequency = abs(float(special.x.ω))
    critical = _match_eigenvalue(complex.(spectrum), Complex(0, frequency))
    margins = sax_smoothness_margins(equilibrium, parameters)
    reasons = String[]
    special.status == :converged || push!(
        reasons,
        "BifurcationKit localization status is $(special.status)",
    )
    residual <= settings.validation_residual_tol || push!(
        reasons,
        "equilibrium residual $residual exceeds tolerance",
    )
    abs(real(critical)) <= settings.validation_eigen_tol || push!(
        reasons,
        "the critical Hopf eigenvalue is not sufficiently neutral",
    )
    margins.reed_distance > settings.smoothness_tol ||
        push!(reasons, "point is too close to reed closure")
    margins.flow_reversal_distance > settings.smoothness_tol ||
        push!(reasons, "point is too close to flow reversal")

    normal_form = nothing
    normal_form_error = nothing
    second_lyapunov = NaN
    normal_form_first_lyapunov = NaN
    first_lyapunov = hasproperty(special.printsol, :l1) ?
        float(real(special.printsol.l1)) : NaN
    if !isfinite(first_lyapunov)
        push!(reasons, "the localized first Lyapunov coefficient is not finite")
    elseif abs(first_lyapunov) > settings.validation_lyapunov_tol
        push!(
            reasons,
            "the localized first Lyapunov coefficient is not sufficiently close to zero",
        )
    end
    try
        normal_form = BK.get_normal_form(
            hopf_branch,
            Int(specialpoint_index);
            nev=2 + 2settings.nmodes,
            autodiff=false,
            verbose=false,
        )
        normal_form_first_lyapunov = float(real(normal_form.nf.l1))
        second_lyapunov = float(real(normal_form.nf.l2))
        if !isfinite(normal_form_first_lyapunov)
            push!(
                reasons,
                "the normal-form first Lyapunov coefficient is not finite",
            )
        elseif abs(normal_form_first_lyapunov) > settings.validation_lyapunov_tol
            push!(
                reasons,
                "the independently evaluated first Lyapunov coefficient is not sufficiently close to zero",
            )
        end
        isfinite(second_lyapunov) || push!(
            reasons,
            "the second Lyapunov coefficient is not finite",
        )
    catch err
        normal_form_error = sprint(showerror, err)
        push!(reasons, "Bautin normal-form evaluation failed")
    end

    return (
        valid=isempty(reasons),
        gamma=gamma,
        zeta=zeta,
        frequency=frequency,
        eigenvalue=critical,
        first_lyapunov=first_lyapunov,
        normal_form_first_lyapunov=normal_form_first_lyapunov,
        second_lyapunov=second_lyapunov,
        residual=residual,
        margins=margins,
        localization_status=special.status,
        normal_form=normal_form,
        normal_form_error=normal_form_error,
        reasons=reasons,
        specialpoint_index=Int(specialpoint_index),
    )
end

function _sax_modal_eigenvalue(spectrum,
                               model_p::NamedTuple,
                               mode::Integer)
    upper = ComplexF64[lambda for lambda in complex.(spectrum) if imag(lambda) > 0]
    isempty(upper) && error("equilibrium spectrum has no positive-frequency eigenvalues")
    target = float(model_p.ω[Int(mode)])
    return upper[argmin(abs.(imag.(upper) .- target))]
end

"""
    validate_sax_double_hopf_intersection(gamma, zeta, modes, model_p; settings)

Refine an intersection of two independently continued Hopf loci by solving for
the two modal growth rates simultaneously. This is independent of
BifurcationKit's codimension-two event detector and therefore catches a
double-Hopf point even when no `:hh` special point was emitted.

The returned point passes the same residual, smoothness, eigenvalue, frequency,
and transversality tests as `validate_sax_double_hopf`. The event-attached
BifurcationKit normal form is not available without a localized special point,
so this result records `normal_form_evaluated=false`. Use
`sax_double_hopf_normal_form` to derive the equivalent cubic interaction
coefficients directly at this independently localized equilibrium.
"""
function validate_sax_double_hopf_intersection(
        gamma::Real,
        zeta::Real,
        modes::Tuple{Int,Int},
        model_p::NamedTuple;
        settings::SaxBifurcationSettings=SaxBifurcationSettings(),
        max_iterations::Integer=12)
    modes[1] != modes[2] || throw(ArgumentError("double-Hopf modes must differ"))
    x = Float64[gamma, zeta]
    converged = false
    escaped_parameter_box = false
    h = settings.validation_parameter_step

    # An interpolated curve intersection is only a predictor.  In particular,
    # a self-overlapping or weakly resolved portable curve can give Newton a
    # predictor whose first correction is far outside the requested diagram.
    # Keep the last valid iterate instead of evaluating the equilibrium model
    # at that escaped point.  The margin also keeps the centered finite
    # differences below inside the parameter rectangle.
    function inside_parameter_box(point; margin::Real=0.0)
        return all(isfinite, point) &&
            settings.gamma_range[1] + margin <= point[1] <=
                settings.gamma_range[2] - margin &&
            settings.zeta_range[1] + margin <= point[2] <=
                settings.zeta_range[2] - margin
    end
    inside_parameter_box(x; margin=h) || throw(ArgumentError(
        "double-Hopf predictor lies outside the configured parameter rectangle",
    ))

    function critical_spectrum(point)
        spectrum, equilibrium, residual, p = _equilibrium_spectrum(
            model_p, point[1], point[2], settings)
        critical = ComplexF64[
            _sax_modal_eigenvalue(spectrum, model_p, modes[1]),
            _sax_modal_eigenvalue(spectrum, model_p, modes[2]),
        ]
        return critical, equilibrium, residual, p
    end

    transversality = fill(NaN, 2, 2)
    for _ in 1:Int(max_iterations)
        critical, _, _, _ = critical_spectrum(x)
        growth = real.(critical)
        if maximum(abs, growth) <= settings.validation_eigen_tol / 100
            converged = true
            break
        end

        for coordinate in 1:2
            plus = copy(x); plus[coordinate] += h
            minus = copy(x); minus[coordinate] -= h
            plus_spectrum = first(_equilibrium_spectrum(
                model_p, plus[1], plus[2], settings))
            minus_spectrum = first(_equilibrium_spectrum(
                model_p, minus[1], minus[2], settings))
            for pair in 1:2
                lambda_plus = _match_eigenvalue(complex.(plus_spectrum), critical[pair])
                lambda_minus = _match_eigenvalue(complex.(minus_spectrum), critical[pair])
                transversality[pair, coordinate] =
                    real(lambda_plus - lambda_minus) / (2h)
            end
        end
        isfinite(det(transversality)) || break
        correction = transversality \ growth
        all(isfinite, correction) || break
        trial = x .- correction
        if !inside_parameter_box(trial; margin=h)
            escaped_parameter_box = true
            break
        end
        x .= trial
    end

    critical, equilibrium, residual, p = critical_spectrum(x)
    for coordinate in 1:2
        plus = copy(x); plus[coordinate] += h
        minus = copy(x); minus[coordinate] -= h
        plus_spectrum = first(_equilibrium_spectrum(
            model_p, plus[1], plus[2], settings))
        minus_spectrum = first(_equilibrium_spectrum(
            model_p, minus[1], minus[2], settings))
        for pair in 1:2
            lambda_plus = _match_eigenvalue(complex.(plus_spectrum), critical[pair])
            lambda_minus = _match_eigenvalue(complex.(minus_spectrum), critical[pair])
            transversality[pair, coordinate] =
                real(lambda_plus - lambda_minus) / (2h)
        end
    end
    determinant = det(transversality)
    margins = sax_smoothness_margins(equilibrium, p)
    reasons = String[]
    converged || push!(reasons, "modal growth-rate Newton refinement did not converge")
    escaped_parameter_box && push!(
        reasons,
        "modal growth-rate Newton step left the configured parameter rectangle",
    )
    residual <= settings.validation_residual_tol ||
        push!(reasons, "equilibrium residual $residual exceeds tolerance")
    maximum(abs.(real.(critical))) <= settings.validation_eigen_tol ||
        push!(reasons, "the two modal growth rates are not sufficiently close to zero")
    abs(imag(critical[1]) - imag(critical[2])) >=
        settings.validation_frequency_separation ||
        push!(reasons, "the two candidate Hopf frequencies are not distinct")
    margins.reed_distance > settings.smoothness_tol ||
        push!(reasons, "point is too close to reed closure")
    margins.flow_reversal_distance > settings.smoothness_tol ||
        push!(reasons, "point is too close to flow reversal")
    abs(determinant) >= settings.validation_transversality_tol ||
        push!(reasons, "the two growth-rate gradients are nearly linearly dependent")

    return (
        valid=isempty(reasons),
        gamma=x[1],
        zeta=x[2],
        modes=modes,
        frequencies=imag.(critical),
        eigenvalues=critical,
        residual=residual,
        margins=margins,
        transversality_matrix=transversality,
        transversality_determinant=determinant,
        localization_status=:analytic_intersection,
        validation_method=:independent_hopf_intersection,
        normal_form=nothing,
        normal_form_evaluated=false,
        normal_form_error=nothing,
        reasons=reasons,
        specialpoint_index=0,
    )
end

function _sax_double_hopf_eigenvectors(A, targets)
    right_decomposition = eigen(complex.(A))
    left_decomposition = eigen(adjoint(complex.(A)))
    used = Set{Int}()
    eigenvalues = ComplexF64[]
    right_vectors = Vector{ComplexF64}[]
    left_vectors = Vector{ComplexF64}[]
    for target in targets
        available = [index for index in eachindex(right_decomposition.values)
                     if !(index in used)]
        index = available[argmin(abs.(right_decomposition.values[available] .- target))]
        push!(used, index)
        eigenvalue = ComplexF64(right_decomposition.values[index])
        q = ComplexF64.(right_decomposition.vectors[:, index])
        q ./= norm(q)
        left_index = argmin(abs.(left_decomposition.values .- conj(eigenvalue)))
        p = ComplexF64.(left_decomposition.vectors[:, left_index])
        p ./= dot(q, p)
        push!(eigenvalues, eigenvalue)
        push!(right_vectors, q)
        push!(left_vectors, p)
    end
    return eigenvalues, right_vectors, left_vectors
end

function _sax_low_order_frequency_resonance(omega1::Real, omega2::Real;
                                             maximum_order::Integer=4,
                                             tolerance::Real=0.02)
    candidates = [
        (m=m, n=n, detuning=m * omega1 - n * omega2,
         relative_detuning=abs(m * omega1 - n * omega2) /
                           max(abs(m * omega1), abs(n * omega2)))
        for m in 1:Int(maximum_order), n in 1:Int(maximum_order)
        if m != n
    ]
    closest = candidates[argmin(getproperty.(candidates, :relative_detuning))]
    return merge(closest, (
        near=closest.relative_detuning <= tolerance,
        tolerance=float(tolerance),
    ))
end

"""
    sax_double_hopf_normal_form(point, model_p; settings, resonance_tolerance=0.02)

Compute the cubic Hopf-Hopf interaction normal form directly at an independently
localized double-Hopf equilibrium. This fills the gap left when the point was
found by intersecting two Hopf curves rather than by BifurcationKit's `:hh`
event detector.

The complex normal form uses BifurcationKit's conventions and Euclidean
eigenvector scaling:

`z1' = (beta1 + i*omega1)z1 + G2100/2*z1*abs2(z1) + G1011*z1*abs2(z2)`

`z2' = (beta2 + i*omega2)z2 + G1110*z2*abs2(z1) + G0021/2*z2*abs2(z2)`

where `beta = real(Gamma) * [gamma-gamma_HH, zeta-zeta_HH]`. The returned
amplitude matrix contains the real cubic coefficients. Two quadratic
parameter predictors give the tangent directions of the Neimark-Sacker curves
expected to emanate from the two small-amplitude periodic-orbit families.

Because a low-order frequency resonance changes the normal form, the result
also reports the nearest resonance and the condition numbers of every
homological solve. Numerical validity here remains a floating-point statement.
"""
function sax_double_hopf_normal_form(
        point,
        model_p::NamedTuple;
        settings::SaxBifurcationSettings=SaxBifurcationSettings(),
        parameter_step::Real=settings.validation_parameter_step,
        resonance_tolerance::Real=0.02)
    point.valid || throw(ArgumentError("the double-Hopf point is not validated"))
    length(point.frequencies) == 2 || throw(ArgumentError(
        "the double-Hopf point must contain two frequencies",
    ))
    parameter_step > 0 || throw(ArgumentError("parameter_step must be positive"))

    ordering = sortperm(float.(point.frequencies); rev=true)
    frequencies = float.(point.frequencies[ordering])
    modes = hasproperty(point, :modes) && !isnothing(point.modes) ?
        Tuple(Int.(collect(point.modes)[ordering])) : (0, 0)
    targets = hasproperty(point, :eigenvalues) && length(point.eigenvalues) == 2 ?
        ComplexF64.(point.eigenvalues[ordering]) : ComplexF64.(im .* frequencies)

    problem, equilibrium, residual = _sax_bifurcation_problem(
        model_p, point.gamma, point.zeta, settings)
    parameters = sax_bifurcation_parameters(
        model_p;
        gamma=point.gamma,
        zeta=point.zeta,
        nmodes=settings.nmodes,
    )
    A = sax_bifurcation_jacobian(
        equilibrium,
        parameters;
        smoothness_tol=settings.smoothness_tol,
        strict=true,
    )
    eigenvalues, q, p = _sax_double_hopf_eigenvectors(A, targets)
    omega1, omega2 = imag(eigenvalues[1]), imag(eigenvalues[2])
    omega1 > omega2 > 0 || throw(ArgumentError(
        "the critical frequencies could not be ordered as omega1 > omega2 > 0",
    ))

    vector_field = problem.VF
    B = BK.BilinearMap((dx1, dx2) ->
        BK.d2F(vector_field, equilibrium, parameters, dx1, dx2))
    C = BK.TrilinearMap((dx1, dx2, dx3) ->
        BK.d3F(vector_field, equilibrium, parameters, dx1, dx2, dx3))
    q1, q2 = q
    p1, p2 = p
    cq1, cq2 = conj(q1), conj(q2)
    lambda1, lambda2 = eigenvalues
    identity_matrix = Matrix{ComplexF64}(I, length(equilibrium), length(equilibrium))
    complex_A = complex.(A)

    M2000 = 2lambda1 .* identity_matrix .- complex_A
    M0020 = 2lambda2 .* identity_matrix .- complex_A
    M1010 = im * (omega1 + omega2) .* identity_matrix .- complex_A
    M1001 = im * (omega1 - omega2) .* identity_matrix .- complex_A
    h2000 = M2000 \ B(q1, q1)
    h0020 = M0020 \ B(q2, q2)
    h1010 = M1010 \ B(q1, q2)
    h1001 = M1001 \ B(q1, cq2)
    h1100 = -(complex_A \ B(q1, cq1))
    h0011 = -(complex_A \ B(q2, cq2))

    G2100 = dot(p1,
        C(q1, q1, cq1) + B(h2000, cq1) + 2B(h1100, q1))
    G0021 = dot(p2,
        C(q2, q2, cq2) + B(h0020, cq2) + 2B(h0011, q2))
    G1110 = dot(p2,
        C(q1, cq1, q2) + B(h1100, q2) +
        B(h1010, cq1) + B(conj(h1001), q1))
    G1011 = dot(p1,
        C(q1, q2, cq2) + B(h1010, cq2) +
        B(h1001, q2) + B(h0011, q1))

    Gamma = zeros(ComplexF64, 2, 2)
    base_parameters = (float(point.gamma), float(point.zeta))
    for coordinate in 1:2
        step = float(parameter_step) * max(1.0, abs(base_parameters[coordinate]))
        plus = collect(base_parameters); plus[coordinate] += step
        minus = collect(base_parameters); minus[coordinate] -= step
        plus_spectrum = complex.(first(_equilibrium_spectrum(
            model_p, plus[1], plus[2], settings)))
        minus_spectrum = complex.(first(_equilibrium_spectrum(
            model_p, minus[1], minus[2], settings)))
        for pair in 1:2
            lambda_plus = _match_eigenvalue(plus_spectrum, eigenvalues[pair])
            lambda_minus = _match_eigenvalue(minus_spectrum, eigenvalues[pair])
            Gamma[pair, coordinate] = (lambda_plus - lambda_minus) / (2step)
        end
    end

    complex_amplitude_matrix = [G2100 / 2 G1011; G1110 G0021 / 2]
    amplitude_matrix = real.(complex_amplitude_matrix)
    alpha1 = real.(Gamma) \ [amplitude_matrix[1, 1], amplitude_matrix[2, 1]]
    alpha2 = real.(Gamma) \ [amplitude_matrix[1, 2], amplitude_matrix[2, 2]]
    predictor1 = -alpha1
    predictor2 = -alpha2
    direction1 = predictor1 / norm(predictor1)
    direction2 = predictor2 / norm(predictor2)
    domega1 = [imag(G2100) / 2, imag(G1110)] - imag.(Gamma) * alpha1
    domega2 = [imag(G1011), imag(G0021) / 2] - imag.(Gamma) * alpha2

    gram = ComplexF64[dot(p[row], q[column]) for row in 1:2, column in 1:2]
    transversality_reference = hasproperty(point, :transversality_matrix) ?
        point.transversality_matrix[ordering, :] : fill(NaN, 2, 2)
    resonance = _sax_low_order_frequency_resonance(
        omega1, omega2; tolerance=resonance_tolerance)
    margins = sax_smoothness_margins(equilibrium, parameters)
    homological_condition_numbers = (
        h2000=cond(M2000),
        h0020=cond(M0020),
        h1010=cond(M1010),
        h1001=cond(M1001),
        h1100=cond(complex_A),
        h0011=cond(complex_A),
    )
    finite_coefficients = all(isfinite, real.(complex_amplitude_matrix)) &&
                          all(isfinite, imag.(complex_amplitude_matrix))
    numerical_valid = residual <= settings.validation_residual_tol &&
        norm(gram - I, Inf) <= 1e-6 && finite_coefficients &&
        abs(det(real.(Gamma))) >= settings.validation_transversality_tol

    return (
        numerical_valid=numerical_valid,
        gamma=float(point.gamma),
        zeta=float(point.zeta),
        modes=modes,
        eigenvalues=eigenvalues,
        frequencies=(omega1, omega2),
        frequency_ratio=omega1 / omega2,
        resonance=resonance,
        Gamma=Gamma,
        cubic_coefficients=(
            G2100=G2100,
            G1011=G1011,
            G1110=G1110,
            G0021=G0021,
        ),
        complex_amplitude_matrix=complex_amplitude_matrix,
        amplitude_matrix=amplitude_matrix,
        amplitude_determinant=det(amplitude_matrix),
        ns_predictors=(
            branch1=(periodic_mode=modes[1], alpha=alpha1,
                     parameter_quadratic_coefficient=predictor1,
                     direction=direction1, frequency_correction=domega1),
            branch2=(periodic_mode=modes[2], alpha=alpha2,
                     parameter_quadratic_coefficient=predictor2,
                     direction=direction2, frequency_correction=domega2),
        ),
        equilibrium_residual=residual,
        smoothness_margins=margins,
        biorthogonality_matrix=gram,
        biorthogonality_error=norm(gram - I, Inf),
        transversality_reference=transversality_reference,
        transversality_difference=real.(Gamma) - transversality_reference,
        homological_condition_numbers=homological_condition_numbers,
        interpretation_status=resonance.near ? :near_low_order_resonance :
                              :nonresonant_cubic_normal_form,
    )
end

"""
    sax_double_hopf_curve_connections(point, result; normal_form=nothing)

Measure the closest approach of every computed PD and NS curve to a validated
double-Hopf point. A curve is called `locally_connected` only when it enters the
specified parameter-space tolerance. If a normal form is supplied, NS tangents
are compared with its two predicted NS directions. This is a numerical topology
diagnostic, not a proof that a curve is globally unique.
"""
function sax_double_hopf_curve_connections(
        point,
        result;
        normal_form=nothing,
        additional_ns_curves=Any[],
        tolerance::Real=5e-3)
    tolerance > 0 || throw(ArgumentError("tolerance must be positive"))
    curve_groups = (
        (:pd, Any[result.pd_curves...]),
        (:ns, vcat(Any[result.ns_curves...], Any[additional_ns_curves...])),
    )
    rows = Any[]
    for (kind, curves) in curve_groups, (curve_index, curve) in enumerate(curves)
        isempty(curve.gamma) && continue
        distances = hypot.(curve.gamma .- point.gamma, curve.zeta .- point.zeta)
        nearest_index = argmin(distances)
        endpoint_distance = min(first(distances), last(distances))
        tangent = fill(NaN, 2)
        predictor = :none
        alignment = NaN
        if length(distances) >= 2
            left = max(1, nearest_index - 1)
            right = min(length(distances), nearest_index + 1)
            tangent .= [curve.gamma[right] - curve.gamma[left],
                        curve.zeta[right] - curve.zeta[left]]
            tangent_norm = norm(tangent)
            tangent_norm > 0 && (tangent ./= tangent_norm)
        end
        if kind == :ns && !isnothing(normal_form) && all(isfinite, tangent)
            directions = (
                branch1=normal_form.ns_predictors.branch1.direction,
                branch2=normal_form.ns_predictors.branch2.direction,
            )
            alignments = (
                branch1=abs(dot(tangent, directions.branch1)),
                branch2=abs(dot(tangent, directions.branch2)),
            )
            predictor = alignments.branch1 >= alignments.branch2 ? :branch1 : :branch2
            alignment = getproperty(alignments, predictor)
        end
        push!(rows, (
            kind=kind,
            curve_index=curve_index,
            point_count=length(distances),
            minimum_distance=distances[nearest_index],
            nearest_gamma=curve.gamma[nearest_index],
            nearest_zeta=curve.zeta[nearest_index],
            nearest_index=nearest_index,
            endpoint_distance=endpoint_distance,
            locally_connected=distances[nearest_index] <= tolerance,
            endpoint_at_double_hopf=endpoint_distance <= tolerance,
            nearest_predictor=predictor,
            tangent_alignment=alignment,
            source=hasproperty(curve, :source) ? curve.source : nothing,
        ))
    end
    return (
        tolerance=float(tolerance),
        rows=rows,
        ns_locally_connected=any(row -> row.kind == :ns && row.locally_connected, rows),
        pd_locally_connected=any(row -> row.kind == :pd && row.locally_connected, rows),
    )
end

"""Find and independently validate intersections between different modal Hopf loci."""
function find_sax_double_hopf_intersections(
        hopf_curves,
        model_p::NamedTuple;
        settings::SaxBifurcationSettings=SaxBifurcationSettings(),
        comparison_points::Integer=513)
    candidates = Any[]
    h = settings.validation_parameter_step
    for first_index in eachindex(hopf_curves)
        first_curve = hopf_curves[first_index]
        first_curve.mode > 0 || continue
        for second_index in (first_index + 1):length(hopf_curves)
            second_curve = hopf_curves[second_index]
            second_curve.mode > 0 || continue
            first_curve.mode != second_curve.mode || continue
            # Two-parameter Hopf continuations can temporarily wander outside
            # the plotting rectangle in the dependent parameter.  Restrict
            # the interpolation overlap before constructing HH predictors.
            low = max(
                minimum(first_curve.zeta),
                minimum(second_curve.zeta),
                settings.zeta_range[1] + h,
            )
            high = min(
                maximum(first_curve.zeta),
                maximum(second_curve.zeta),
                settings.zeta_range[2] - h,
            )
            low < high || continue
            grid = collect(range(low, high; length=max(3, Int(comparison_points))))
            difference = [
                _interpolate_curve_gamma(first_curve, z) -
                _interpolate_curve_gamma(second_curve, z)
                for z in grid
            ]
            for interval in 1:(length(grid) - 1)
                difference[interval] * difference[interval + 1] <= 0 || continue
                left, right = grid[interval], grid[interval + 1]
                for _ in 1:60
                    midpoint = (left + right) / 2
                    left_value = _interpolate_curve_gamma(first_curve, left) -
                                 _interpolate_curve_gamma(second_curve, left)
                    midpoint_value = _interpolate_curve_gamma(first_curve, midpoint) -
                                     _interpolate_curve_gamma(second_curve, midpoint)
                    if left_value * midpoint_value <= 0
                        right = midpoint
                    else
                        left = midpoint
                    end
                end
                candidate_zeta = (left + right) / 2
                candidate_gamma = (
                    _interpolate_curve_gamma(first_curve, candidate_zeta) +
                    _interpolate_curve_gamma(second_curve, candidate_zeta)
                ) / 2
                isfinite(candidate_gamma) || continue
                settings.gamma_range[1] + h <= candidate_gamma <=
                    settings.gamma_range[2] - h || continue
                validation = try
                    validate_sax_double_hopf_intersection(
                        candidate_gamma,
                        candidate_zeta,
                        (Int(first_curve.mode), Int(second_curve.mode)),
                        model_p;
                        settings=settings,
                    )
                catch err
                    err isa InterruptException && rethrow()
                    # Independent intersection validation supplements the
                    # event-localized HH detector.  A bad interpolated
                    # predictor must not discard the underlying Hopf curves.
                    continue
                end
                duplicate = any(point ->
                    hypot(point.gamma - validation.gamma,
                          point.zeta - validation.zeta) < 5e-4,
                    candidates)
                duplicate || push!(candidates, validation)
            end
        end
    end
    return candidates
end

#endregion

#region PERIODIC ORBITS, PERIOD DOUBLING, AND NEIMARK-SACKER CURVES

const _SAX_PRESSURE_L2_QUADRATURE_CACHE =
    Dict{Int,NamedTuple{(:weights, :interpolation),Tuple{Vector{Float64},Matrix{Float64}}}}()

"""
    _sax_periodic_pressure_l2(collocation, orbit)

Return the normalized pressure L2 amplitude
`sqrt(integral_0^1 p(tau)^2 d tau)` of a collocation orbit. Because physical
time is `t = T*tau`, this is identical to
`sqrt((1/T) * integral_0^T p(t)^2 dt)` and is therefore an RMS pressure.

On each collocation interval the pressure is a polynomial of degree `m`.
An `(m + 1)`-point Gauss-Legendre rule integrates its square exactly. This is
preferable to treating the nonuniform collocation nodes as uniformly sampled
time data and gives the same observable used in a harmonic-balance amplitude
diagram without depending on the periodic-orbit period.
"""
function _sax_periodic_pressure_l2(collocation, orbit)
    pressure = vec(sum(@view orbit.u[3:2:end, :]; dims=1))
    if !(collocation isa BK.Collocation)
        duration = float(last(orbit.t) - first(orbit.t))
        duration > 0 || return NaN
        integral = sum(eachindex(pressure)[1:end-1]) do index
            dt = float(orbit.t[index + 1] - orbit.t[index])
            dt * (abs2(pressure[index]) + abs2(pressure[index + 1])) / 2
        end
        return sqrt(max(float(integral / duration), 0.0))
    end

    _, degree, intervals = size(collocation)
    quadrature = get!(_SAX_PRESSURE_L2_QUADRATURE_CACHE, degree) do
        order = degree + 1
        off_diagonal = [index / sqrt(4index^2 - 1)
                        for index in 1:(order - 1)]
        factorization = eigen(SymTridiagonal(zeros(order), off_diagonal))
        gauss_nodes = collect(float.(factorization.values))
        weights = collect(float.(2 .* abs2.(factorization.vectors[1, :])))
        interpolation = [
            BK.lagrange(node_index, gauss_nodes[gauss_index],
                        BK.get_mesh_coll(collocation))
            for gauss_index in eachindex(gauss_nodes),
                node_index in 1:(degree + 1)
        ]
        (weights=weights, interpolation=interpolation)
    end
    mesh = BK.getmesh(collocation)
    integral = 0.0
    for interval in 1:intervals
        indices = (1:(degree + 1)) .+ (interval - 1) * degree
        nodal_pressure = @view pressure[indices]
        interval_width = float(mesh[interval + 1] - mesh[interval])
        for gauss_index in eachindex(quadrature.weights)
            value = dot(
                @view(quadrature.interpolation[gauss_index, :]),
                nodal_pressure,
            )
            integral += interval_width * quadrature.weights[gauss_index] *
                        abs2(value) / 2
        end
    end
    return sqrt(max(integral, 0.0))
end

function _record_sax_periodic_orbit(x, info;
                                    grazing_velocity_tol::Real=1e-5,
                                    kwargs...)
    orbit = BK.get_periodic_orbit(info.prob, x, info.p)
    pressure = vec(sum(@view orbit.u[3:2:end, :]; dims=1))
    reed_opening = vec(real.(@view orbit.u[1, :])) .+ 1
    reed_velocity = vec(real.(@view orbit.u[2, :]))
    pressure_drop = info.p.gamma .- pressure
    nearest_contact_index = argmin(abs.(reed_opening))
    crosses_reed_contact = minimum(reed_opening) <= 0 <= maximum(reed_opening)
    contact_crossing_speed = abs(reed_velocity[nearest_contact_index])
    return (
        gamma=info.p.gamma,
        zeta=info.p.zeta,
        period=BK.getperiod(info.prob, x, info.p),
        pressure_min=minimum(pressure),
        pressure_max=maximum(pressure),
        pressure_amplitude=maximum(pressure) - minimum(pressure),
        pressure_l2=_sax_periodic_pressure_l2(info.prob, orbit),
        minimum_reed_opening=minimum(reed_opening),
        minimum_absolute_reed_opening=minimum(abs, reed_opening),
        minimum_absolute_pressure_drop=minimum(abs, pressure_drop),
        crosses_reed_contact=crosses_reed_contact,
        contact_crossing_speed=contact_crossing_speed,
        possible_grazing_contact=crosses_reed_contact &&
                                 contact_crossing_speed <= grazing_velocity_tol,
    )
end

"""
    continue_sax_periodic_orbits(hopf_curve, point_index; settings=..., verbosity=0)

Start a collocation branch of periodic orbits from a selected point on a Hopf
curve and continue it in `gamma` at the corresponding fixed `zeta`.

The orbit is represented by piecewise polynomials on
`po_collocation_intervals`, with `po_collocation_degree` nodes per interval.
The default `po_linear_solver=:condensed` combines an in-place analytic dense
collocation Jacobian with BifurcationKit's `COPBLS` condensation solver. Set
`po_linear_solver=:dense` only for solver-validation comparisons with the
previous full dense bordered solve. Bifurcation detection acts on Floquet
multipliers: an additional crossing through `+1` produces `:fold`, a crossing
through `-1` produces `:pd`, and a non-real conjugate pair crossing the unit
circle produces `:ns`.
"""
function continue_sax_periodic_orbits(hopf_curve,
                                      point_index::Integer;
                                      settings::SaxBifurcationSettings=SaxBifurcationSettings(),
                                      verbosity::Int=0)
    _validate_sax_bifurcation_settings(settings)
    1 <= point_index <= length(hopf_curve) || throw(BoundsError(hopf_curve.branch, point_index))

    options = BK.ContinuationPar(
        p_min=settings.gamma_range[1],
        p_max=settings.gamma_range[2],
        ds=settings.po_ds,
        dsmin=min(1e-7, abs(settings.po_ds) / 100),
        dsmax=settings.po_dsmax,
        max_steps=settings.po_max_steps,
        detect_bifurcation=3,
        nev=2 + 2settings.nmodes,
        save_eigenvectors=true,
        save_sol_every_step=settings.po_save_sol_every_step,
        n_inversion=8,
        max_bisection_steps=35,
        tol_stability=settings.stability_tol,
        newton_options=BK.NewtonPar(
            tol=settings.newton_tol,
            max_iterations=30,
            verbose=false,
        ),
    )
    condensed_solver = settings.po_linear_solver == :condensed
    collocation = BK.Collocation(
        settings.po_collocation_intervals,
        settings.po_collocation_degree;
        jacobian=condensed_solver ?
                 BK.DenseAnalyticalInplace() : BK.DenseAnalytical(),
        update_section_every_step=1,
    )
    linear_solver = condensed_solver ? BK.COPBLS() : BK.MatrixBLS()
    record_orbit = (x, info; kwargs...) -> _record_sax_periodic_orbit(
        x,
        info;
        grazing_velocity_tol=settings.periodic_grazing_velocity_tol,
        kwargs...,
    )

    return BK.continuation_from_hopf_point(
        hopf_curve,
        Int(point_index),
        options,
        collocation;
        lens=(BK.@optic _.gamma),
        autodiff_nf=false,
        nev=2 + 2settings.nmodes,
        record_from_solution=record_orbit,
        bothside=true,
        normC=BK.norminf,
        linear_algo=linear_solver,
        finalise_solution=_sax_continuation_progress(
            "periodic-orbit branch",
            settings.po_max_steps,
            verbosity;
            every_steps=5,
        ),
        verbosity=max(0, verbosity - 1),
        plot=false,
    )
end

"""
    continue_sax_periodic_bifurcation_curve(po_branch, specialpoint_index;
                                             settings=..., verbosity=0)

Continue a periodic-orbit `:fold`, `:pd`, or `:ns` special point in
`(gamma,zeta)`.
The augmented collocation problem enforces both the orbit equations and the
critical Floquet condition, producing the desired fold, period-doubling, or
Neimark-Sacker curve.
"""
function continue_sax_periodic_bifurcation_curve(
        po_branch,
        specialpoint_index::Integer;
        settings::SaxBifurcationSettings=SaxBifurcationSettings(),
        verbosity::Int=0)
    _validate_sax_bifurcation_settings(settings)
    1 <= specialpoint_index <= length(po_branch.specialpoint) ||
        throw(BoundsError(po_branch.specialpoint, specialpoint_index))
    bifurcation_type = po_branch.specialpoint[specialpoint_index].type
    bifurcation_type in (:fold, :pd, :ns) || throw(ArgumentError(
        "periodic special point must be :fold, :pd, or :ns, found $bifurcation_type",
    ))

    options = BK.ContinuationPar(
        p_min=settings.zeta_range[1],
        p_max=settings.zeta_range[2],
        ds=settings.po_curve_ds,
        dsmin=min(1e-7, abs(settings.po_curve_ds) / 100),
        dsmax=settings.po_curve_dsmax,
        max_steps=settings.po_curve_max_steps,
        detect_bifurcation=0,
        nev=2 + 2settings.nmodes,
        save_eigenvectors=true,
        save_sol_every_step=settings.po_curve_save_sol_every_step,
        tol_stability=settings.stability_tol,
        newton_options=BK.NewtonPar(
            tol=settings.newton_tol,
            max_iterations=35,
            verbose=false,
        ),
    )

    return BK.continuation(
        po_branch,
        Int(specialpoint_index),
        (BK.@optic _.zeta),
        options;
        detect_codim2_bifurcation=2,
        update_minaug_every_step=1,
        start_with_eigen=false,
        usehessian=false,
        jacobian_ma=BK.MinAugMatrixBased(),
        bothside=true,
        normC=BK.norminf,
        finalise_solution=_sax_continuation_progress(
            "$(uppercase(string(bifurcation_type))) curve",
            settings.po_curve_max_steps,
            verbosity;
            every_steps=5,
        ),
        verbosity=max(0, verbosity - 1),
        plot=false,
    )
end

#endregion

#region ONE-CALL DRIVER AND PLOTTING

function _sax_branch_coordinates(branch)
    fields = propertynames(branch.branch)
    :gamma in fields || error("branch record does not contain gamma")
    :zeta in fields || error("branch record does not contain zeta")
    gamma = collect(float.(getproperty(branch.branch, :gamma)))
    zeta = collect(float.(getproperty(branch.branch, :zeta)))
    keep = isfinite.(gamma) .& isfinite.(zeta)
    return gamma[keep], zeta[keep]
end

function _hopf_curve_mode(branch, model_p::NamedTuple, nmodes::Int)
    frequency_field = Symbol("ωₕ")
    if hasproperty(branch.branch, frequency_field)
        frequencies = abs.(float.(getproperty(branch.branch, frequency_field)))
        frequencies = frequencies[isfinite.(frequencies)]
        if !isempty(frequencies)
            omega = median(frequencies)
            mode = argmin(abs.(float.(model_p.ω[1:nmodes]) .- omega))
            return Int(mode), omega
        end
    end
    return 0, NaN
end

function _sax_restore_endpoint_solution(saved)
    if hasproperty(saved, :x) && hasproperty(saved, :p1)
        border = hasproperty(saved, :ω) ?
            [float(saved.p1), float(saved.ω)] : float(saved.p1)
        return BK.BorderedArray(saved.x, border)
    end
    return saved
end

function _sax_branch_terminal_diagnostics(branch, active_parameter::Symbol)
    records = branch.branch
    fields = propertynames(records)
    point_count = length(branch)
    gamma = :gamma in fields ? collect(float.(records.gamma)) : fill(NaN, point_count)
    zeta = :zeta in fields ? collect(float.(records.zeta)) : fill(NaN, point_count)
    ds = :ds in fields ? collect(float.(records.ds)) : fill(NaN, point_count)
    steps = :step in fields ? collect(Int.(records.step)) : collect(0:(point_count - 1))
    newton_iterations = :itnewton in fields ? collect(Int.(records.itnewton)) : fill(-1, point_count)
    limits = active_parameter == :gamma ?
        branch.contparams.p_min => branch.contparams.p_max :
        branch.contparams.p_min => branch.contparams.p_max
    tolerance = max(10abs(branch.contparams.dsmin), 1e-8)

    specialpoints = [(
        type=sp.type,
        status=sp.status,
        parameter=float(sp.param),
        precision=float(sp.precision),
        interval=(float(sp.interval[1]), float(sp.interval[2])),
        index=Int(sp.idx),
        step=Int(sp.step),
    ) for sp in branch.specialpoint]

    endpoint_specialpoints = [sp for sp in branch.specialpoint if sp.type == :endpoint]
    endpoints = map(endpoint_specialpoints) do endpoint
        index = clamp(endpoint.idx, 1, max(1, point_count))
        coordinate = float(endpoint.param)
        problem = BK.getprob(branch)
        base_parameters = BK.getparams(problem)
        first_parameter = hasproperty(endpoint.x, :p1) ?
            float(endpoint.x.p1) : NaN
        endpoint_gamma = hasproperty(endpoint.printsol, :gamma) ?
            float(endpoint.printsol.gamma) :
            (active_parameter == :gamma ? coordinate :
             (isfinite(first_parameter) ? first_parameter :
              (hasproperty(base_parameters, :gamma) ?
               float(base_parameters.gamma) : gamma[index])))
        endpoint_zeta = hasproperty(endpoint.printsol, :zeta) ?
            float(endpoint.printsol.zeta) :
            (active_parameter == :zeta ? coordinate :
             (hasproperty(base_parameters, :zeta) ?
              float(base_parameters.zeta) : zeta[index]))
        parameters = if hasproperty(base_parameters, :gamma) &&
                        hasproperty(base_parameters, :zeta)
            merge(base_parameters, (gamma=endpoint_gamma, zeta=endpoint_zeta))
        else
            BK.setparam(branch, coordinate)
        end
        classification = if endpoint.status != :converged
            :unlocalized_endpoint
        elseif abs(coordinate - first(limits)) <= tolerance
            :lower_parameter_boundary
        elseif abs(coordinate - last(limits)) <= tolerance
            :upper_parameter_boundary
        elseif abs(ds[index]) <= 1.05abs(branch.contparams.dsmin)
            :minimum_step
        elseif abs(steps[index]) >= branch.contparams.max_steps
            :maximum_steps
        else
            :returned_internal
        end

        residual_norm, equation_residual_norm = try
            restored_solution = _sax_restore_endpoint_solution(endpoint.x)
            residual = BK.residual(problem, restored_solution, parameters)
            base_residual = hasproperty(residual, :u) ? residual.u : residual
            has_phase_condition = branch.kind isa BK.PeriodicOrbitCont ||
                                  branch.kind isa BK.FoldPeriodicOrbitCont ||
                                  branch.kind isa BK.PDPeriodicOrbitCont ||
                                  branch.kind isa BK.NSPeriodicOrbitCont
            equation_residual = has_phase_condition && length(base_residual) > 1 ?
                @view(base_residual[begin:end-1]) : base_residual
            (norm(residual, Inf), norm(equation_residual, Inf))
        catch
            (NaN, NaN)
        end
        (
            index=Int(index),
            gamma=endpoint_gamma,
            zeta=endpoint_zeta,
            active_parameter=active_parameter,
            parameter=coordinate,
            classification=classification,
            ds=ds[index],
            accepted_step=steps[index],
            newton_iterations=newton_iterations[index],
            residual_norm=residual_norm,
            equation_residual_norm=equation_residual_norm,
            localization_status=endpoint.status,
        )
    end

    return (
        point_count=point_count,
        active_parameter=active_parameter,
        parameter_limits=(float(first(limits)), float(last(limits))),
        dsmin=float(branch.contparams.dsmin),
        max_steps=Int(branch.contparams.max_steps),
        endpoints=endpoints,
        specialpoints=specialpoints,
    )
end

function _curve_summary(branch, kind::Symbol;
                        mode::Int=0,
                        frequency::Real=NaN,
                        source=nothing,
                        active_parameter::Symbol=:zeta)
    gamma, zeta = _sax_branch_coordinates(branch)
    return (
        kind=kind,
        gamma=gamma,
        zeta=zeta,
        mode=mode,
        frequency=float(frequency),
        source=source,
        diagnostics=_sax_branch_terminal_diagnostics(branch, active_parameter),
        branch=branch,
    )
end

function _monotone_curve_samples(curve)
    order = sortperm(curve.zeta)
    zeta = collect(float.(curve.zeta[order]))
    gamma = collect(float.(curve.gamma[order]))
    unique_zeta = Float64[]
    unique_gamma = Float64[]
    for (z, g) in zip(zeta, gamma)
        if !isempty(unique_zeta) && isapprox(z, unique_zeta[end]; atol=1e-12, rtol=0)
            unique_gamma[end] = (unique_gamma[end] + g) / 2
        else
            push!(unique_zeta, z)
            push!(unique_gamma, g)
        end
    end
    return unique_zeta, unique_gamma
end

function _interpolate_curve_gamma(curve, zeta::Real)
    z, gamma = _monotone_curve_samples(curve)
    length(z) >= 2 || throw(ArgumentError("curve needs at least two distinct zeta values"))
    z[1] <= zeta <= z[end] || throw(BoundsError(z, zeta))
    right = clamp(searchsortedfirst(z, float(zeta)), 2, length(z))
    left = right - 1
    fraction = (float(zeta) - z[left]) / (z[right] - z[left])
    return gamma[left] + fraction * (gamma[right] - gamma[left])
end

function _curves_are_duplicates(a, b;
                                tolerance::Real=3e-3,
                                comparison_points::Integer=65)
    a.kind == b.kind || return false
    a.mode == b.mode || return false
    isempty(a.gamma) && return isempty(b.gamma)
    isempty(b.gamma) && return false

    # Different continuation runs generally sample the same locus at different
    # zeta values. Compare interpolants on a common grid rather than nearest raw
    # points; nearest-point comparison retained duplicate H1/H2 curves in the
    # first Pilot run merely because their meshes were offset.
    low = max(minimum(a.zeta), minimum(b.zeta))
    high = min(maximum(a.zeta), maximum(b.zeta))
    low < high || return false
    span_a = maximum(a.zeta) - minimum(a.zeta)
    span_b = maximum(b.zeta) - minimum(b.zeta)
    overlap_fraction = (high - low) / max(min(span_a, span_b), eps(Float64))
    overlap_fraction >= 0.8 || return false

    grid = range(low, high; length=max(3, Int(comparison_points)))
    maximum(abs(_interpolate_curve_gamma(a, z) -
                _interpolate_curve_gamma(b, z)) for z in grid) <= tolerance
end

function _push_unique_curve!(curves::Vector, curve; tolerance::Real=3e-3)
    any(existing -> _curves_are_duplicates(existing, curve; tolerance=tolerance), curves) ||
        push!(curves, curve)
    return curves
end

"""
    _checkpoint_is_covered_by_curve(checkpoint, curve; tolerance=1e-3)

Return `true` when an already cached curve of the same bifurcation type and
modal family passes through a periodic-orbit checkpoint.  Several fixed-zeta
periodic branches can detect different points on the same U-shaped two-
parameter component.  Continuing every such point retraces the component and
can cost hours, so Stage 3 uses this geometric test before starting a new run.

The comparison is point-to-polyline rather than point-to-sample.  This keeps
the test insensitive to the continuation mesh while the conservative default
tolerance avoids merging visibly separated components.
"""
function _checkpoint_is_covered_by_curve(
        checkpoint,
        curve;
        tolerance::Real=1e-3)
    checkpoint.type == curve.kind || return false
    checkpoint.mode == curve.mode || return false
    isempty(curve.gamma) && return false

    gamma = float(checkpoint.gamma)
    zeta = float(checkpoint.zeta)
    if length(curve.gamma) == 1
        curve_gamma = float(curve.gamma[1])
        curve_zeta = float(curve.zeta[1])
        return isfinite(curve_gamma) && isfinite(curve_zeta) &&
            hypot(gamma - curve_gamma, zeta - curve_zeta) <= tolerance
    end

    distance = Inf
    for index in 1:(length(curve.gamma) - 1)
        gamma_left = float(curve.gamma[index])
        zeta_left = float(curve.zeta[index])
        gamma_right = float(curve.gamma[index + 1])
        zeta_right = float(curve.zeta[index + 1])
        all(isfinite, (gamma_left, zeta_left, gamma_right, zeta_right)) ||
            continue
        delta_gamma = gamma_right - gamma_left
        delta_zeta = zeta_right - zeta_left
        length_squared = delta_gamma^2 + delta_zeta^2
        fraction = iszero(length_squared) ? 0.0 : clamp(
            ((gamma - gamma_left) * delta_gamma +
             (zeta - zeta_left) * delta_zeta) / length_squared,
            0.0,
            1.0,
        )
        projected_gamma = gamma_left + fraction * delta_gamma
        projected_zeta = zeta_left + fraction * delta_zeta
        distance = min(
            distance,
            hypot(gamma - projected_gamma, zeta - projected_zeta),
        )
        distance <= tolerance && return true
    end
    return false
end

function _sax_pd_curve_orbit_state(value, zeta::Real)
    if hasproperty(value, :x) && hasproperty(value, :p1)
        orbit = value.x isa BK.BorderedArray ? BK.getvec(value.x) : value.x
        return (
            solution=collect(float.(orbit)),
            gamma=float(value.p1),
            zeta=float(zeta),
        )
    end
    inner = BK.getvec(value)
    if hasproperty(inner, :x) && hasproperty(inner, :p1)
        orbit = inner.x isa BK.BorderedArray ? BK.getvec(inner.x) : inner.x
        return (
            solution=collect(float.(orbit)),
            gamma=float(inner.p1),
            zeta=float(zeta),
        )
    end
    if inner isa BK.BorderedArray
        return (
            solution=collect(float.(BK.getvec(inner))),
            gamma=float(BK.getp(inner)),
            zeta=float(zeta),
        )
    end
    value isa BK.BorderedArray || error(
        "unsupported PD augmented state $(typeof(value))",
    )
    return (
        solution=collect(float.(BK.getvec(value))),
        gamma=float(BK.getp(value)),
        zeta=float(zeta),
    )
end

function _sax_pd_r2_neighbour_checkpoints(
        branch,
        source_checkpoint;
        cluster_tolerance::Real=5e-3,
        maximum_step_gap::Integer=25)
    candidates = Any[]
    for special in branch.specialpoint
        special.type in (:R2, :gpdR2) || continue
        special.status == :converged || continue
        state = try
            _sax_pd_curve_orbit_state(special.x, special.param)
        catch
            continue
        end
        push!(candidates, (
            special=special,
            state=state,
            precision=float(special.precision),
        ))
    end
    sort!(candidates; by=item -> item.precision)
    representatives = Any[]
    for candidate in candidates
        duplicate = any(representatives) do stored
            hypot(candidate.state.gamma - stored.state.gamma,
                  candidate.state.zeta - stored.state.zeta) <= cluster_tolerance
        end
        duplicate || push!(representatives, candidate)
    end
    sort!(representatives; by=item -> item.state.gamma)

    saved = sort(collect(branch.sol); by=point -> Int(point.step))
    checkpoints = Any[]
    for (cluster, candidate) in enumerate(representatives)
        special_step = Int(candidate.special.step)
        for side in (:before, :after)
            eligible = side == :before ?
                [point for point in saved if Int(point.step) < special_step] :
                [point for point in saved if Int(point.step) > special_step]
            isempty(eligible) && continue
            selected = eligible[argmin(abs(Int(point.step) - special_step)
                                       for point in eligible)]
            step_gap = abs(Int(selected.step) - special_step)
            step_gap <= maximum_step_gap || continue
            state = try
                _sax_pd_curve_orbit_state(selected.x, selected.p)
            catch
                continue
            end
            key = "pd_r2_cluster$(cluster)_$(side)_$(source_checkpoint.key)_step$(Int(selected.step))"
            push!(checkpoints, (
                key=key,
                type=:pd,
                resonance_type=:R2,
                resonance_cluster=Int(cluster),
                resonance_side=side,
                mode=Int(source_checkpoint.mode),
                source_hopf_key=source_checkpoint.source_hopf_key,
                source_checkpoint_key=source_checkpoint.key,
                specialpoint_index=0,
                localization_status=:converged_pd_neighbour,
                localization_precision=float(candidate.precision),
                gamma=state.gamma,
                zeta=state.zeta,
                floquet_angle=NaN,
                solution=state.solution,
                r2_gamma=candidate.state.gamma,
                r2_zeta=candidate.state.zeta,
                r2_step=special_step,
                continuation_step=Int(selected.step),
                step_gap=step_gap,
            ))
        end
    end
    return checkpoints
end

function _push_unique_periodic_codim2_checkpoint!(checkpoints, checkpoint;
                                                   tolerance::Real=1e-5)
    duplicate = any(checkpoints) do stored
        stored.type == checkpoint.type &&
            stored.mode == checkpoint.mode &&
            hasproperty(stored, :resonance_side) &&
            stored.resonance_side == checkpoint.resonance_side &&
            hypot(stored.gamma - checkpoint.gamma,
                  stored.zeta - checkpoint.zeta) <= tolerance
    end
    duplicate || push!(checkpoints, checkpoint)
    return checkpoints
end

function _select_hopf_point_for_periodic_orbit(hopf_curve, preferred_zeta::Real)
    hasproperty(hopf_curve.branch, :zeta) || error("Hopf curve has no zeta record")
    zeta = collect(float.(hopf_curve.branch.zeta))
    length(zeta) >= 3 || error("Hopf curve is too short to initialize a periodic branch")
    # Avoid the first and last continuation points, whose tangents are usually
    # less accurate, and avoid an already detected codimension-two point.
    candidates = collect(2:(length(zeta) - 1))
    special_steps = Set(sp.idx for sp in hopf_curve.specialpoint if sp.type != :endpoint)
    filtered = [i for i in candidates if !(i in special_steps)]
    isempty(filtered) && (filtered = candidates)
    return filtered[argmin(abs.(zeta[filtered] .- preferred_zeta))]
end

function _sax_hopf_checkpoint(hopf_branch,
                              point_index::Integer,
                              mode::Integer;
                              source=nothing)
    formulation = BK.get_formulation(BK.getprob(hopf_branch))
    saved_point = hopf_branch.sol[Int(point_index)]
    state = collect(float.(BK.get_solution(saved_point.x)))
    frequency = abs(float(BK.get_frequency(saved_point.x, formulation)))
    parameters = BK.getparams(hopf_branch, Int(point_index))
    gamma = float(parameters.gamma)
    zeta = float(parameters.zeta)
    key = "hopf_m$(Int(mode))_g$(round(gamma; digits=10))_z$(round(zeta; digits=10))"
    return (
        key=key,
        mode=Int(mode),
        gamma=gamma,
        zeta=zeta,
        frequency=frequency,
        state=state,
        source=source,
    )
end

function _sax_problem_from_checkpoint(checkpoint,
                                      model_p::NamedTuple,
                                      settings::SaxBifurcationSettings)
    parameters = sax_bifurcation_parameters(
        model_p;
        gamma=checkpoint.gamma,
        zeta=checkpoint.zeta,
        nmodes=settings.nmodes,
    )
    record_state = (u, _parameter; kwargs...) -> (
        state_norm=norm(u),
        acoustic_pressure=sum(@view u[3:2:end]),
    )
    jacobian = (u, pars) -> sax_bifurcation_jacobian(
        u,
        pars;
        smoothness_tol=settings.smoothness_tol,
        strict=true,
        strict_reed=!settings.allow_transverse_reed_contact,
        strict_flow=true,
    )
    problem = BK.BifurcationProblem(
        sax_bifurcation_residual,
        copy(checkpoint.state),
        parameters,
        (BK.@optic _.gamma);
        J=jacobian,
        record_from_solution=record_state,
    )
    return problem, parameters
end

function _sax_periodic_options(
        settings::SaxBifurcationSettings;
        detect_bifurcation::Integer=3,
        save_eigenvectors::Bool=true)
    return BK.ContinuationPar(
        p_min=settings.gamma_range[1],
        p_max=settings.gamma_range[2],
        ds=settings.po_ds,
        dsmin=min(1e-7, abs(settings.po_ds) / 100),
        dsmax=settings.po_dsmax,
        max_steps=settings.po_max_steps,
        detect_bifurcation=Int(detect_bifurcation),
        nev=2 + 2settings.nmodes,
        save_eigenvectors=save_eigenvectors,
        save_sol_every_step=settings.po_save_sol_every_step,
        n_inversion=8,
        max_bisection_steps=35,
        tol_stability=settings.stability_tol,
        newton_options=BK.NewtonPar(
            tol=settings.newton_tol,
            max_iterations=30,
            verbose=false,
        ),
    )
end

function _sax_collocation(settings::SaxBifurcationSettings)
    condensed = settings.po_linear_solver == :condensed
    collocation = BK.Collocation(
        settings.po_collocation_intervals,
        settings.po_collocation_degree;
        jacobian=condensed ? BK.DenseAnalyticalInplace() : BK.DenseAnalytical(),
        update_section_every_step=1,
    )
    return collocation, condensed ? BK.COPBLS() : BK.MatrixBLS()
end

"""
    continue_sax_periodic_orbits(checkpoint, model_p;
                                 settings, verbosity=0, bothside=true,
                                 step_callback=nothing)

Resume periodic-orbit continuation from a portable Hopf checkpoint. The Hopf
eigenvectors and normal form are recomputed from the analytic equilibrium
Jacobian, so the checkpoint contains only arrays and scalar parameters and is
safe to store in JLD2. Set `bothside=false` for a directional continuation;
the sign of `settings.po_ds` then selects the pseudo-arclength orientation.
An optional `step_callback` receives the same arguments as BifurcationKit's
`finalise_solution` callback and may return `false` to stop after an accepted
event without changing the default continuation behavior.
"""
function continue_sax_periodic_orbits(
        checkpoint::NamedTuple,
        model_p::NamedTuple;
        settings::SaxBifurcationSettings=SaxBifurcationSettings(),
        verbosity::Int=0,
        bothside::Bool=true,
        step_callback=nothing,
        eigsolver=nothing,
        detect_bifurcation::Integer=3,
        save_eigenvectors::Bool=true)
    _validate_sax_bifurcation_settings(settings)
    problem, parameters = _sax_problem_from_checkpoint(checkpoint, model_p, settings)
    jacobian = BK.jacobian(problem, checkpoint.state, parameters)
    right_factorization = eigen(jacobian)
    right_index = argmin(abs.(right_factorization.values .-
                             Complex(0, checkpoint.frequency)))
    right = copy(right_factorization.vectors[:, right_index])
    right ./= norm(right)

    left_factorization = eigen(adjoint(jacobian))
    left_index = argmin(abs.(left_factorization.values .-
                            conj(right_factorization.values[right_index])))
    left = copy(left_factorization.vectors[:, left_index])
    left ./= BK.VI.inner(right, left)

    empty_normal_form = BK.HopfNormalForm(
        a=missing,
        b=missing,
        Ψ001=missing,
        Ψ110=missing,
        Ψ200=missing,
    )
    hopf_point = BK.Hopf(
        copy(checkpoint.state),
        nothing,
        checkpoint.gamma,
        checkpoint.frequency,
        parameters,
        (BK.@optic _.gamma),
        right,
        left,
        empty_normal_form,
        :unknown,
    )
    hopf_normal_form = BK.__hopf_normal_form(
        problem,
        hopf_point,
        BK.DefaultLS();
        verbose=verbosity > 1,
        autodiff=false,
        L=jacobian,
    )
    options = _sax_periodic_options(
        settings;
        detect_bifurcation=detect_bifurcation,
        save_eigenvectors=save_eigenvectors,
    )
    collocation, linear_solver = _sax_collocation(settings)
    record_orbit = (x, info; kwargs...) -> _record_sax_periodic_orbit(
        x,
        info;
        grazing_velocity_tol=settings.periodic_grazing_velocity_tol,
        kwargs...,
    )
    progress = _sax_continuation_progress(
        "periodic-orbit branch $(checkpoint.key)",
        settings.po_max_steps,
        verbosity;
        every_steps=5,
    )
    finaliser = isnothing(step_callback) ? progress :
        function (z, tangent, step, branch; kwargs...)
            progress(z, tangent, step, branch; kwargs...) || return false
            return step_callback(z, tangent, step, branch; kwargs...)
        end
    common = (
        alg=BK.PALC(),
        record_from_solution=record_orbit,
        bothside=bothside,
        normC=BK.norminf,
        linear_algo=linear_solver,
        finalise_solution=finaliser,
        verbosity=max(0, verbosity - 1),
        plot=false,
    )
    return isnothing(eigsolver) ? BK._continuation(
        hopf_normal_form,
        problem,
        options,
        collocation;
        common...,
    ) : BK._continuation(
        hopf_normal_form,
        problem,
        options,
        collocation;
        eigsolver=eigsolver,
        common...,
    )
end

function _sax_periodic_bifurcation_checkpoints(
        po_branch,
        hopf_checkpoint,
        mode::Integer)
    checkpoints = Any[]
    for (specialpoint_index, special) in enumerate(po_branch.specialpoint)
        special.type in (:fold, :pd, :ns) || continue
        special.status == :converged || continue
        solution = hasproperty(special.x, :sol) ? special.x.sol : special.x
        parameters = BK.setparam(po_branch, special.param)
        floquet_angle = special.type == :ns ?
            imag(po_branch.eig[special.idx].eigenvals[special.ind_ev]) : NaN
        key = "$(special.type)_$(hopf_checkpoint.key)_sp$(specialpoint_index)"
        push!(checkpoints, (
            key=key,
            type=special.type,
            mode=Int(mode),
            source_hopf_key=hopf_checkpoint.key,
            specialpoint_index=Int(specialpoint_index),
            localization_status=special.status,
            localization_precision=float(special.precision),
            gamma=float(special.param),
            zeta=float(parameters.zeta),
            floquet_angle=float(floquet_angle),
            solution=collect(float.(solution)),
        ))
    end
    return checkpoints
end

function _push_unique_periodic_checkpoint!(checkpoints::Vector,
                                           candidate;
                                           parameter_tolerance::Real=1e-3,
                                           angle_tolerance::Real=1e-2)
    duplicate = any(checkpoints) do checkpoint
        checkpoint.type == candidate.type || return false
        checkpoint.mode == candidate.mode || return false
        hypot(checkpoint.gamma - candidate.gamma,
              checkpoint.zeta - candidate.zeta) <= parameter_tolerance || return false
        candidate.type in (:fold, :pd) ||
            abs(abs(checkpoint.floquet_angle) - abs(candidate.floquet_angle)) <=
                angle_tolerance
    end
    duplicate || push!(checkpoints, candidate)
    return checkpoints
end

function _sax_periodic_wrapper(checkpoint,
                               model_p::NamedTuple,
                               settings::SaxBifurcationSettings)
    state_dimension = 2 + 2settings.nmodes
    problem_checkpoint = (
        state=collect(checkpoint.solution[1:state_dimension]),
        gamma=checkpoint.gamma,
        zeta=checkpoint.zeta,
    )
    problem, parameters = _sax_problem_from_checkpoint(
        problem_checkpoint, model_p, settings)
    orbit_dimension = state_dimension * (
        1 + settings.po_collocation_intervals * settings.po_collocation_degree)
    length(checkpoint.solution) == orbit_dimension + 1 || throw(DimensionMismatch(
        "periodic checkpoint has $(length(checkpoint.solution)) entries; expected $(orbit_dimension + 1)",
    ))
    condensed = settings.po_linear_solver == :condensed
    collocation = BK.Collocation(
        settings.po_collocation_intervals,
        settings.po_collocation_degree;
        N=state_dimension,
        prob_vf=problem,
        ϕ=zeros(orbit_dimension),
        xπ=zeros(orbit_dimension),
        ∂ϕ=zeros(state_dimension,
                  settings.po_collocation_intervals * settings.po_collocation_degree),
        jacobian=condensed ? BK.DenseAnalyticalInplace() : BK.DenseAnalytical(),
        update_section_every_step=1,
    )
    BK.updatesection!(collocation, checkpoint.solution, parameters)
    jacobian = BK._generate_jacobian(
        collocation,
        collocation.jacobian,
        checkpoint.solution,
        parameters,
    )
    record_orbit = (x, info; kwargs...) -> _record_sax_periodic_orbit(
        x,
        info;
        grazing_velocity_tol=settings.periodic_grazing_velocity_tol,
        kwargs...,
    )
    record = BK.RecordForPeriodicOrbits(
        record_orbit,
        BK.record_from_solution(problem),
    )
    wrapper = BK.PeriodicOrbitFunctionalColl(
        collocation,
        jacobian,
        checkpoint.solution,
        nothing,
        record,
    )
    return wrapper, parameters
end

function _sax_periodic_curve_options(settings::SaxBifurcationSettings)
    options = BK.ContinuationPar(
        p_min=settings.zeta_range[1],
        p_max=settings.zeta_range[2],
        ds=settings.po_curve_ds,
        dsmin=min(1e-7, abs(settings.po_curve_ds) / 100),
        dsmax=settings.po_curve_dsmax,
        max_steps=settings.po_curve_max_steps,
        detect_bifurcation=0,
        nev=2 + 2settings.nmodes,
        save_eigenvectors=true,
        save_sol_every_step=settings.po_curve_save_sol_every_step,
        tol_stability=settings.stability_tol,
        newton_options=BK.NewtonPar(
            tol=settings.newton_tol,
            max_iterations=35,
            verbose=false,
        ),
    )
    return BK.detect_codim2_parameters(
        2,
        options;
        update_minaug_every_step=1,
    )
end

"""Resume a fold, PD, or NS two-parameter continuation from a portable orbit checkpoint."""
function continue_sax_periodic_bifurcation_curve(
        checkpoint::NamedTuple,
        model_p::NamedTuple;
        settings::SaxBifurcationSettings=SaxBifurcationSettings(),
        verbosity::Int=0)
    checkpoint.type in (:fold, :pd, :ns) || throw(ArgumentError(
        "periodic checkpoint must have type :fold, :pd, or :ns",
    ))
    wrapper, parameters = _sax_periodic_wrapper(checkpoint, model_p, settings)
    options = _sax_periodic_curve_options(settings)
    collocation = BK.get_discretization(wrapper)
    state_dimension, _, _ = size(collocation)
    jacobian = BK.jacobian(wrapper, checkpoint.solution, parameters)
    progress = _sax_continuation_progress(
        "$(uppercase(string(checkpoint.type))) curve $(checkpoint.key)",
        settings.po_curve_max_steps,
        verbosity;
        every_steps=5,
    )
    lens_gamma = (BK.@optic _.gamma)
    lens_zeta = (BK.@optic _.zeta)

    if checkpoint.type == :fold
        # Reconstruct right and left null vectors using the same bordered solve
        # as BifurcationKit's collocation-specific fold initializer. This is
        # substantially better conditioned than selecting the smallest dense
        # singular vector when the fold localization is only approximate.
        dimension = size(jacobian, 1)
        border_right = sin.(collect(1.0:dimension))
        border_left = cos.(collect(1.0:dimension))
        rhs = zeros(eltype(jacobian), dimension)
        bordered_solver = BK.MatrixBLS()
        right, = bordered_solver(
            jacobian, border_right, border_left, zero(eltype(jacobian)),
            rhs, one(eltype(jacobian)))
        left, = bordered_solver(
            adjoint(jacobian), border_left, border_right,
            zero(eltype(jacobian)), rhs, one(eltype(jacobian)))
        right = collect(right ./ norm(right))
        left = collect(left ./ norm(left))
        initial = BK.BorderedArray(copy(checkpoint.solution), checkpoint.gamma)
        return BK.continuation_fold(
            wrapper,
            BK.PALC(),
            initial,
            parameters,
            lens_gamma,
            lens_zeta,
            right,
            left,
            options;
            update_minaug_every_step=1,
            bdlinsolver=bordered_solver,
            jacobian_ma=BK.MinAugMatrixBased(),
            usehessian=false,
            compute_eigen_elements=true,
            bothside=true,
            normC=BK.norminf,
            finalise_solution=progress,
            verbosity=max(0, verbosity - 1),
            plot=false,
            kind=BK.FoldPeriodicOrbitCont(),
        )
    end

    bordered = checkpoint.type == :ns ? Complex.(copy(jacobian)) : copy(jacobian)
    matrix_dimension = size(bordered, 1)
    deterministic_border = sin.(collect(1.0:matrix_dimension))
    bordered[end, :] .= deterministic_border
    bordered[:, end] .= reverse(deterministic_border)
    bordered[end, end] = 0
    if checkpoint.type == :pd
        bordered[end-state_dimension:end-1, 1:state_dimension] .= I(state_dimension)
    elseif checkpoint.type == :ns
        bordered[end-state_dimension:end-1, end-state_dimension:end-1] .=
            I(state_dimension) .* cis(checkpoint.floquet_angle)
    end
    rhs = zeros(eltype(bordered), matrix_dimension)
    rhs[end] = 1
    right = (bordered \ rhs)[begin:end-1]
    left = (adjoint(bordered) \ rhs)[begin:end-1]
    right ./= norm(right)
    left ./= norm(left)

    if checkpoint.type == :pd
        initial = BK.BorderedArray(copy(checkpoint.solution), checkpoint.gamma)
        return BK.continuation_pd(
            wrapper,
            BK.PALC(),
            initial,
            parameters,
            lens_gamma,
            lens_zeta,
            left,
            right,
            options;
            update_minaug_every_step=1,
            jacobian_ma=BK.MinAugMatrixBased(),
            bothside=true,
            normC=BK.norminf,
            finalise_solution=progress,
            verbosity=max(0, verbosity - 1),
            plot=false,
            kind=BK.PDPeriodicOrbitCont(),
        )
    end

    initial = BK.BorderedArray(
        copy(checkpoint.solution),
        [checkpoint.gamma, checkpoint.floquet_angle],
    )
    return BK.continuation_ns(
        wrapper,
        BK.PALC(),
        initial,
        parameters,
        lens_gamma,
        lens_zeta,
        left,
        right,
        options;
        update_minaug_every_step=1,
        jacobian_ma=BK.MinAugMatrixBased(),
        bothside=true,
        normC=BK.norminf,
        finalise_solution=progress,
        verbosity=max(0, verbosity - 1),
        plot=false,
        kind=BK.NSPeriodicOrbitCont(),
    )
end

"""
    continue_sax_generalized_hopf_fold_curve(hopf_branch, gh_index; settings)

Switch directly from a generalized-Hopf normal form to the emanating fold of
periodic orbits. This complements fold detection along already continued
periodic branches: close to a Bautin point the fold cycles have vanishingly
small amplitude and can otherwise be missed by a coarse periodic branch.
"""
function continue_sax_generalized_hopf_fold_curve(
        hopf_branch,
        gh_index::Integer;
        settings::SaxBifurcationSettings=SaxBifurcationSettings(),
        verbosity::Int=0)
    special = hopf_branch.specialpoint[Int(gh_index)]
    special.type == :gh || throw(ArgumentError(
        "special point is $(special.type), not :gh",
    ))
    options = _sax_periodic_curve_options(settings)
    collocation, _ = _sax_collocation(settings)
    record_orbit = (x, info; kwargs...) -> _record_sax_periodic_orbit(
        x,
        info;
        grazing_velocity_tol=settings.periodic_grazing_velocity_tol,
        kwargs...,
    )
    progress = _sax_continuation_progress(
        "GH fold curve at g=$(round(float(special.x.p1); digits=6)), " *
        "z=$(round(float(special.param); digits=6))",
        settings.po_curve_max_steps,
        verbosity;
        every_steps=5,
    )
    return BK.continuation(
        hopf_branch,
        Int(gh_index),
        options,
        collocation;
        alg=BK.PALC(),
        autodiff=false,
        detect_codim2_bifurcation=0,
        δp=settings.gh_fold_predictor_amplitude,
        record_from_solution=record_orbit,
        bothside=true,
        normC=BK.norminf,
        jacobian_ma=BK.MinAugMatrixBased(),
        usehessian=false,
        finalise_solution=progress,
        verbosity=max(0, verbosity - 1),
        plot=false,
    )
end

function _portable_sax_bifurcation_settings(settings::SaxBifurcationSettings)
    names = fieldnames(SaxBifurcationSettings)
    return NamedTuple{names}(Tuple(getfield(settings, name) for name in names))
end

function _portable_sax_curve(curve)
    return (
        kind=curve.kind,
        gamma=copy(curve.gamma),
        zeta=copy(curve.zeta),
        mode=curve.mode,
        frequency=curve.frequency,
        source=hasproperty(curve, :source) ? curve.source : nothing,
        diagnostics=hasproperty(curve, :diagnostics) ? curve.diagnostics : nothing,
    )
end

function _portable_sax_double_hopf(point)
    return (
        valid=point.valid,
        gamma=point.gamma,
        zeta=point.zeta,
        modes=hasproperty(point, :modes) ? point.modes : nothing,
        frequencies=copy(point.frequencies),
        eigenvalues=copy(point.eigenvalues),
        residual=point.residual,
        margins=point.margins,
        transversality_matrix=copy(point.transversality_matrix),
        transversality_determinant=point.transversality_determinant,
        localization_status=point.localization_status,
        validation_method=hasproperty(point, :validation_method) ?
                          point.validation_method : :bifurcationkit_specialpoint,
        normal_form_evaluated=hasproperty(point, :normal_form_evaluated) ?
                              point.normal_form_evaluated :
                              isnothing(point.normal_form_error),
        normal_form_error=point.normal_form_error,
        reasons=copy(point.reasons),
    )
end

function _portable_sax_generalized_hopf(point)
    return (
        valid=point.valid,
        gamma=point.gamma,
        zeta=point.zeta,
        frequency=point.frequency,
        eigenvalue=point.eigenvalue,
        first_lyapunov=point.first_lyapunov,
        normal_form_first_lyapunov=point.normal_form_first_lyapunov,
        second_lyapunov=point.second_lyapunov,
        residual=point.residual,
        margins=point.margins,
        localization_status=point.localization_status,
        normal_form_evaluated=isnothing(point.normal_form_error),
        normal_form_error=point.normal_form_error,
        reasons=copy(point.reasons),
    )
end

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

const SAX_BIFURCATION_STAGE_CACHE_SCHEMA_VERSION = 2
const _SAX_BIFURCATION_STAGE_NAMES = (:hopf, :periodic, :curves)

_sax_stage_cache_path(directory::AbstractString, stage::Symbol) =
    joinpath(directory, "stage_$(stage).jld2")

function _save_sax_stage_cache(path::AbstractString,
                               stage::Symbol,
                               payload,
                               model_p::NamedTuple,
                               settings::SaxBifurcationSettings)
    cache = (
        schema_version=SAX_BIFURCATION_STAGE_CACHE_SCHEMA_VERSION,
        stage=stage,
        settings_signature=_portable_sax_bifurcation_settings(settings),
        model_signature=_sax_bifurcation_model_signature(model_p, settings.nmodes),
        saved_at_unix=time(),
        payload=payload,
    )
    _atomic_jld2_save(path; cache)
    return payload
end

function _load_sax_stage_cache(path::AbstractString,
                               stage::Symbol,
                               model_p::NamedTuple,
                               settings::SaxBifurcationSettings)
    isfile(path) || return (status=:missing, payload=nothing, reason="stage cache is absent")
    stored = try
        Logging.with_logger(Logging.NullLogger()) do
            JLD2.load(path, "cache")
        end
    catch err
        return (status=:corrupt, payload=nothing, reason=sprint(showerror, err))
    end
    schema_matches = try
        stored.schema_version == SAX_BIFURCATION_STAGE_CACHE_SCHEMA_VERSION
    catch err
        return (status=:corrupt, payload=nothing, reason=sprint(showerror, err))
    end
    schema_matches || return (
        status=:incompatible,
        payload=nothing,
        reason="stage cache schema changed",
    )
    checks = try
        (
            stored.stage == stage => "stage name changed",
            isequal(stored.settings_signature,
                    _portable_sax_bifurcation_settings(settings)) =>
                "continuation settings changed",
            isequal(stored.model_signature,
                    _sax_bifurcation_model_signature(model_p, settings.nmodes)) =>
                "acoustic model parameters changed",
        )
    catch err
        return (status=:corrupt, payload=nothing, reason=sprint(showerror, err))
    end
    for (valid, reason) in checks
        valid || return (status=:incompatible, payload=nothing, reason=reason)
    end
    return (status=:valid, payload=stored.payload, reason="")
end

"""Return compatibility and completion summaries for the three restart caches."""
function sax_bifurcation_stage_cache_status(
        directory::AbstractString,
        model_p::NamedTuple;
        settings::SaxBifurcationSettings=SaxBifurcationSettings())
    return map(_SAX_BIFURCATION_STAGE_NAMES) do stage
        path = _sax_stage_cache_path(directory, stage)
        loaded = _load_sax_stage_cache(path, stage, model_p, settings)
        completed = if loaded.status != :valid
            0
        elseif stage == :hopf
            length(loaded.payload.completed_seeds)
        elseif stage == :periodic
            length(loaded.payload.completed_hopf_keys)
        else
            length(loaded.payload.completed_checkpoint_keys)
        end
        total = if loaded.status != :valid
            0
        elseif stage == :hopf
            length(settings.zeta_seeds)
        elseif stage == :periodic
            hasproperty(loaded.payload, :expected_hopf_keys) ?
                length(loaded.payload.expected_hopf_keys) :
                length(loaded.payload.completed_hopf_keys)
        else
            hasproperty(loaded.payload, :expected_checkpoint_keys) ?
                length(loaded.payload.expected_checkpoint_keys) :
                length(loaded.payload.completed_checkpoint_keys)
        end
        return (stage=stage, status=loaded.status, completed=completed,
                total=total, path=path, reason=loaded.reason)
    end
end

function _record_sax_stage_failure!(failures::Vector, failure)
    if hasproperty(failure, :key)
        filter!(failures) do existing
            !(hasproperty(existing, :key) &&
              existing.key == failure.key &&
              existing.stage == failure.stage)
        end
    end
    push!(failures, failure)
    return failures
end

function _clear_sax_stage_failure!(failures::Vector, stage::Symbol, key::AbstractString)
    filter!(failures) do existing
        !(hasproperty(existing, :key) &&
          existing.key == key &&
          existing.stage == stage)
    end
    return failures
end

"""
    compute_sax_bifurcation_diagram(model_p; settings=..., include_periodic=true,
                                    stage_cache_directory=nothing, resume=true)

Run the complete workflow and return portable `(gamma,zeta)` curve summaries:

1. Continue equilibria in `gamma` from every `zeta_seeds` slice
2. Continue every detected Hopf point in `(gamma,zeta)`
3. Independently validate every detected double-Hopf and generalized-Hopf point
4. Switch valid generalized-Hopf points directly to periodic-orbit fold curves
5. Start a collocation periodic-orbit branch from every unique Hopf curve
6. Detect folds, period-doubling, and Neimark-Sacker points by Floquet
   multipliers using `periodic_eigsolver` when one is supplied
7. Continue every such point as a two-parameter curve

When `stage_cache_directory` is supplied, compatible JLD2 checkpoints are
written after every equilibrium seed, periodic branch, and fold/PD/NS component.
With `resume=true`, a later run reconstructs the required numerical problems
from portable Hopf and periodic-orbit checkpoints and skips completed work.
No function closures or Pluto workspace types are serialized.

Failures are accumulated in `result.failures` rather than aborting the entire
survey. One difficult periodic component therefore does not erase successfully
computed equilibrium/Hopf curves.

`verbosity=1` emits concise `@info` milestones and accepted-step updates that
Pluto can display while the computation is running.  `verbosity=2` additionally
enables BifurcationKit's detailed stdout trace.
"""
function compute_sax_bifurcation_diagram(
        model_p::NamedTuple;
        settings::SaxBifurcationSettings=SaxBifurcationSettings(),
        include_periodic::Bool=true,
        verbosity::Int=1,
        stage_cache_directory::Union{Nothing,AbstractString}=nothing,
        resume::Bool=true,
        periodic_eigsolver=nothing)
    _validate_sax_bifurcation_settings(settings)
    analysis_started_ns = time_ns()
    cache_enabled = !isnothing(stage_cache_directory)
    cache_enabled && mkpath(stage_cache_directory)

    verbosity > 0 && @info(
        "Sax bifurcation analysis started",
        state_dimension=2 + 2settings.nmodes,
        zeta_seeds=length(settings.zeta_seeds),
        include_periodic,
        stage_cache_directory,
        resume,
        periodic_eigsolver=isnothing(periodic_eigsolver) ?
            :bifurcationkit_default : Symbol(nameof(typeof(periodic_eigsolver))),
    )

    # Stage 1: equilibrium slices, unique Hopf loci, portable Hopf seeds, and
# BifurcationKit-localized GH points, and both localized and independently
# intersected HH points.
    hopf_cache = cache_enabled && resume ? _load_sax_stage_cache(
        _sax_stage_cache_path(stage_cache_directory, :hopf),
        :hopf,
        model_p,
        settings,
    ) : (status=:missing, payload=nothing, reason="resume disabled")
    if hopf_cache.status == :valid
        payload = hopf_cache.payload
        completed_seeds = collect(float.(payload.completed_seeds))
        equilibrium_run_count = Int(payload.equilibrium_run_count)
        hopf_curves = Any[payload.hopf_curves...]
        hopf_checkpoints = Any[payload.hopf_checkpoints...]
        double_hopf_points = Any[payload.double_hopf_points...]
        generalized_hopf_points = Any[payload.generalized_hopf_points...]
        generalized_hopf_fold_curves = Any[payload.generalized_hopf_fold_curves...]
        stage1_failures = Any[payload.failures...]
        verbosity > 0 && @info(
            "Stage 1/3 cache loaded",
            completed_seeds=length(completed_seeds),
            unique_hopf_curves=length(hopf_curves),
            hopf_checkpoints=length(hopf_checkpoints),
            double_hopf_points=length(double_hopf_points),
            generalized_hopf_points=length(generalized_hopf_points),
            generalized_hopf_fold_curves=length(generalized_hopf_fold_curves),
        )
    else
        completed_seeds = Float64[]
        equilibrium_run_count = 0
        hopf_curves = Any[]
        hopf_checkpoints = Any[]
        double_hopf_points = Any[]
        generalized_hopf_points = Any[]
        generalized_hopf_fold_curves = Any[]
        stage1_failures = Any[]
    end
    raw_equilibrium_runs = Any[]

    for (seed_number, zeta_seed) in enumerate(settings.zeta_seeds)
        any(seed -> isapprox(seed, zeta_seed; atol=0, rtol=0), completed_seeds) && continue
        slice_started_ns = time_ns()
        verbosity > 0 && @info(
            "Stage 1/3: equilibrium and Hopf seed started",
            seed_number,
            seed_count=length(settings.zeta_seeds),
            zeta=zeta_seed,
        )
        equilibrium_run = try
            continue_sax_equilibria(model_p, zeta_seed;
                                     settings=settings,
                                     verbosity=verbosity)
        catch err
            err isa InterruptException && rethrow()
            push!(stage1_failures, (stage=:equilibrium, zeta=zeta_seed,
                                    exception_type=Symbol(nameof(typeof(err))),
                                    error=sprint(showerror, err)))
            verbosity > 0 && @warn(
                "Equilibrium continuation failed; continuing the survey",
                zeta_seed,
                exception_type=typeof(err),
                error=sprint(showerror, err),
                elapsed_seconds=_sax_elapsed_seconds(slice_started_ns),
                captured_failures=length(stage1_failures),
            )
            nothing
        end
        if !isnothing(equilibrium_run)
            equilibrium_run_count += 1
            push!(raw_equilibrium_runs, equilibrium_run)
            verbosity > 0 && @info(
                "Stage 1/3: equilibrium slice completed",
                seed_number,
                zeta=zeta_seed,
                branch_points=length(equilibrium_run.branch),
                hopf_candidates=length(equilibrium_run.hopf_specialpoint_indices),
                elapsed_seconds=_sax_elapsed_seconds(slice_started_ns),
            )
        end

        hopf_indices = isnothing(equilibrium_run) ?
            Int[] : equilibrium_run.hopf_specialpoint_indices
        for hopf_specialpoint_index in hopf_indices
            hopf_started_ns = time_ns()
            verbosity > 0 && @info(
                "Stage 1/3: Hopf-curve continuation started",
                zeta_seed,
                equilibrium_specialpoint=hopf_specialpoint_index,
            )
            hopf_branch = try
                continue_sax_hopf_curve(
                    equilibrium_run.branch,
                    hopf_specialpoint_index;
                    settings=settings,
                    verbosity=verbosity,
                )
            catch err
                err isa InterruptException && rethrow()
                push!(stage1_failures, (stage=:hopf_curve, zeta=zeta_seed,
                                        index=hopf_specialpoint_index,
                                        exception_type=Symbol(nameof(typeof(err))),
                                        error=sprint(showerror, err)))
                verbosity > 0 && @warn(
                    "Hopf-curve continuation failed; continuing the survey",
                    zeta_seed,
                    hopf_specialpoint_index,
                    exception_type=typeof(err),
                    error=sprint(showerror, err),
                    elapsed_seconds=_sax_elapsed_seconds(hopf_started_ns),
                    captured_failures=length(stage1_failures),
                )
                continue
            end

            mode, omega = _hopf_curve_mode(hopf_branch, model_p, settings.nmodes)
            summary = _curve_summary(
                hopf_branch,
                :hopf;
                mode=mode,
                frequency=omega,
                source=(zeta=zeta_seed, equilibrium_specialpoint=hopf_specialpoint_index),
            )
            before = length(hopf_curves)
            _push_unique_curve!(hopf_curves, summary)
            is_new_curve = length(hopf_curves) > before
            hh_indices = findall(sp -> sp.type == :hh, hopf_branch.specialpoint)
            gh_indices = findall(sp -> sp.type == :gh, hopf_branch.specialpoint)
            verbosity > 0 && @info(
                "Stage 1/3: Hopf curve completed",
                mode,
                curve_points=length(hopf_branch),
                new_unique_curve=is_new_curve,
                double_hopf_candidates=length(hh_indices),
                generalized_hopf_candidates=length(gh_indices),
                elapsed_seconds=_sax_elapsed_seconds(hopf_started_ns),
                unique_hopf_curves=length(hopf_curves),
            )

            # Validate HH points even on a duplicate branch; final coordinate
            # deduplication below retains the strongest available localization.
            for hh_index in hh_indices
                hh_started_ns = time_ns()
                verbosity > 0 && @info(
                    "Stage 1/3: double-Hopf validation started",
                    mode,
                    hopf_specialpoint=hh_index,
                )
                validation = try
                    validate_sax_double_hopf(
                        hopf_branch,
                        hh_index,
                        model_p;
                    settings=settings,
                )
                catch err
                    err isa InterruptException && rethrow()
                    push!(stage1_failures, (stage=:double_hopf_validation,
                                            zeta=zeta_seed,
                                            index=hh_index,
                                            exception_type=Symbol(nameof(typeof(err))),
                                            error=sprint(showerror, err)))
                    verbosity > 0 && @warn(
                        "Double-Hopf validation failed; continuing the survey",
                        hh_index,
                        exception_type=typeof(err),
                        error=sprint(showerror, err),
                        elapsed_seconds=_sax_elapsed_seconds(hh_started_ns),
                        captured_failures=length(stage1_failures),
                    )
                    continue
                end
                duplicate = any(point ->
                    hypot(point.gamma - validation.gamma,
                          point.zeta - validation.zeta) < 5e-4,
                    double_hopf_points)
                duplicate || push!(double_hopf_points, validation)
                verbosity > 0 && @info(
                    "Stage 1/3: double-Hopf validation completed",
                    valid=validation.valid,
                    gamma=validation.gamma,
                    zeta=validation.zeta,
                    reasons=length(validation.reasons),
                    duplicate,
                    elapsed_seconds=_sax_elapsed_seconds(hh_started_ns),
                    retained_double_hopf_points=length(double_hopf_points),
                )
            end

            # A generalized Hopf (Bautin) point is accepted only after the
            # equilibrium, spectral localization, smoothness, and normal-form
            # checks in `validate_sax_generalized_hopf`. Duplicate detections
            # from overlapping Hopf continuations are merged geometrically.
            for gh_index in gh_indices
                gh_started_ns = time_ns()
                verbosity > 0 && @info(
                    "Stage 1/3: generalized-Hopf validation started",
                    mode,
                    hopf_specialpoint=gh_index,
                )
                validation = try
                    validate_sax_generalized_hopf(
                        hopf_branch,
                        gh_index,
                        model_p;
                        settings=settings,
                    )
                catch err
                    err isa InterruptException && rethrow()
                    push!(stage1_failures, (
                        stage=:generalized_hopf_validation,
                        zeta=zeta_seed,
                        index=gh_index,
                        exception_type=Symbol(nameof(typeof(err))),
                        error=sprint(showerror, err),
                    ))
                    verbosity > 0 && @warn(
                        "Generalized-Hopf validation failed; continuing the survey",
                        gh_index,
                        exception_type=typeof(err),
                        error=sprint(showerror, err),
                        elapsed_seconds=_sax_elapsed_seconds(gh_started_ns),
                        captured_failures=length(stage1_failures),
                    )
                    continue
                end
                duplicate = any(point ->
                    hypot(point.gamma - validation.gamma,
                          point.zeta - validation.zeta) < 5e-4,
                    generalized_hopf_points)
                duplicate || push!(generalized_hopf_points, validation)
                verbosity > 0 && @info(
                    "Stage 1/3: generalized-Hopf validation completed",
                    valid=validation.valid,
                    gamma=validation.gamma,
                    zeta=validation.zeta,
                    l1=validation.first_lyapunov,
                    l2=validation.second_lyapunov,
                    reasons=length(validation.reasons),
                    duplicate,
                    elapsed_seconds=_sax_elapsed_seconds(gh_started_ns),
                    retained_generalized_hopf_points=length(generalized_hopf_points),
                )

                # Use the Bautin predictor to start the emanating fold curve
                # while the full Hopf branch and normal-form workspace are
                # available. The portable curve is saved in the stage-1 cache,
                # so resuming does not require serializing those workspaces.
                already_continued = any(curve -> begin
                    source = curve.source
                    hasproperty(source, :generalized_hopf) &&
                        hypot(source.generalized_hopf.gamma - validation.gamma,
                              source.generalized_hopf.zeta - validation.zeta) < 5e-4
                end, generalized_hopf_fold_curves)
                if include_periodic && validation.valid && !already_continued
                    gh_fold_started_ns = time_ns()
                    verbosity > 0 && @info(
                        "Stage 1/3: GH fold-of-periodic-orbits continuation started",
                        gamma=validation.gamma,
                        zeta=validation.zeta,
                        mode,
                    )
                    gh_fold_branch = try
                        continue_sax_generalized_hopf_fold_curve(
                            hopf_branch,
                            gh_index;
                            settings=settings,
                            verbosity=verbosity,
                        )
                    catch err
                        err isa InterruptException && rethrow()
                        push!(stage1_failures, (
                            stage=:generalized_hopf_fold_curve,
                            zeta=zeta_seed,
                            index=gh_index,
                            exception_type=Symbol(nameof(typeof(err))),
                            error=sprint(showerror, err),
                        ))
                        verbosity > 0 && @warn(
                            "GH fold-curve continuation failed; fold detection on periodic branches remains active",
                            gamma=validation.gamma,
                            zeta=validation.zeta,
                            exception_type=typeof(err),
                            error=sprint(showerror, err),
                            elapsed_seconds=_sax_elapsed_seconds(gh_fold_started_ns),
                            captured_failures=length(stage1_failures),
                        )
                        nothing
                    end
                    if !isnothing(gh_fold_branch)
                        fold_summary = _curve_summary(
                            gh_fold_branch,
                            :fold;
                            mode=mode,
                            source=(
                                generalized_hopf=(
                                    gamma=validation.gamma,
                                    zeta=validation.zeta,
                                ),
                                hopf_mode=mode,
                            ),
                            active_parameter=:zeta,
                        )
                        _push_unique_curve!(
                            generalized_hopf_fold_curves,
                            fold_summary,
                        )
                        verbosity > 0 && @info(
                            "Stage 1/3: GH fold-of-periodic-orbits continuation completed",
                            curve_points=length(gh_fold_branch),
                            retained_gh_fold_curves=
                                length(generalized_hopf_fold_curves),
                            elapsed_seconds=
                                _sax_elapsed_seconds(gh_fold_started_ns),
                        )
                    end
                end
            end

            if is_new_curve
                try
                    point_index = _select_hopf_point_for_periodic_orbit(
                        hopf_branch, zeta_seed)
                    push!(hopf_checkpoints, _sax_hopf_checkpoint(
                        hopf_branch,
                        point_index,
                        mode;
                        source=summary.source,
                    ))
                catch err
                    err isa InterruptException && rethrow()
                    push!(stage1_failures, (stage=:periodic_initialization,
                                            zeta=zeta_seed,
                                            exception_type=Symbol(nameof(typeof(err))),
                                            error=sprint(showerror, err)))
                end
            end
        end

        independent_points = try
            find_sax_double_hopf_intersections(
                hopf_curves,
                model_p;
                settings=settings,
            )
        catch err
            err isa InterruptException && rethrow()
            push!(stage1_failures, (
                stage=:independent_double_hopf_intersections,
                zeta=zeta_seed,
                exception_type=Symbol(nameof(typeof(err))),
                error=sprint(showerror, err),
            ))
            verbosity > 0 && @warn(
                "Independent double-Hopf search failed; preserving completed Hopf curves",
                zeta_seed,
                exception_type=typeof(err),
                error=sprint(showerror, err),
                captured_failures=length(stage1_failures),
            )
            Any[]
        end
        for validation in independent_points
            duplicate = any(point ->
                hypot(point.gamma - validation.gamma,
                      point.zeta - validation.zeta) < 5e-4,
                double_hopf_points)
            duplicate || push!(double_hopf_points, validation)
        end
        push!(completed_seeds, float(zeta_seed))
        if cache_enabled
            payload = (
                completed_seeds=copy(completed_seeds),
                equilibrium_run_count=equilibrium_run_count,
                hopf_curves=_portable_sax_curve.(hopf_curves),
                hopf_checkpoints=copy(hopf_checkpoints),
                double_hopf_points=_portable_sax_double_hopf.(double_hopf_points),
                generalized_hopf_points=
                    _portable_sax_generalized_hopf.(generalized_hopf_points),
                generalized_hopf_fold_curves=
                    _portable_sax_curve.(generalized_hopf_fold_curves),
                failures=copy(stage1_failures),
            )
            _save_sax_stage_cache(
                _sax_stage_cache_path(stage_cache_directory, :hopf),
                :hopf,
                payload,
                model_p,
                settings,
            )
        end
    end

    # Stage 2: reconstruct each unique Hopf seed and continue its periodic
    # branch. Every detected fold/PD/NS orbit is stored as a portable checkpoint.
    periodic_cache = cache_enabled && resume ? _load_sax_stage_cache(
        _sax_stage_cache_path(stage_cache_directory, :periodic),
        :periodic,
        model_p,
        settings,
    ) : (status=:missing, payload=nothing, reason="resume disabled")
    if periodic_cache.status == :valid
        payload = periodic_cache.payload
        completed_hopf_keys = String.(payload.completed_hopf_keys)
        periodic_branch_count = Int(payload.periodic_branch_count)
        periodic_checkpoints = Any[payload.periodic_checkpoints...]
        periodic_branch_diagnostics = Any[payload.periodic_branch_diagnostics...]
        stage2_failures = Any[payload.failures...]
        verbosity > 0 && @info(
            "Stage 2/3 cache loaded",
            completed_hopf_branches=length(completed_hopf_keys),
            periodic_branches=periodic_branch_count,
            periodic_bifurcation_checkpoints=length(periodic_checkpoints),
        )
    else
        completed_hopf_keys = String[]
        periodic_branch_count = 0
        periodic_checkpoints = Any[]
        periodic_branch_diagnostics = Any[]
        stage2_failures = Any[]
    end
    raw_periodic_branches = Any[]

    if include_periodic
        for checkpoint in hopf_checkpoints
            checkpoint.key in completed_hopf_keys && continue
            periodic_started_ns = time_ns()
            verbosity > 0 && @info(
                "Stage 2/3: periodic-orbit continuation started",
                key=checkpoint.key,
                mode=checkpoint.mode,
            )
            po_branch = try
                continue_sax_periodic_orbits(
                    checkpoint,
                    model_p;
                    settings=settings,
                    verbosity=verbosity,
                    eigsolver=periodic_eigsolver,
                )
            catch err
                err isa InterruptException && rethrow()
                _record_sax_stage_failure!(stage2_failures, (
                    stage=:periodic_orbit,
                    mode=checkpoint.mode,
                    key=checkpoint.key,
                    exception_type=Symbol(nameof(typeof(err))),
                    error=sprint(showerror, err),
                ))
                verbosity > 0 && @warn(
                    "Periodic-orbit continuation failed; continuing the survey",
                    mode=checkpoint.mode,
                    key=checkpoint.key,
                    exception_type=typeof(err),
                    error=sprint(showerror, err),
                    elapsed_seconds=_sax_elapsed_seconds(periodic_started_ns),
                    captured_failures=length(stage2_failures),
                )
                nothing
            end
            if !isnothing(po_branch)
                _clear_sax_stage_failure!(
                    stage2_failures,
                    :periodic_orbit,
                    checkpoint.key,
                )
                periodic_branch_count += 1
                push!(raw_periodic_branches, po_branch)
                for candidate in _sax_periodic_bifurcation_checkpoints(
                        po_branch, checkpoint, checkpoint.mode)
                    _push_unique_periodic_checkpoint!(
                        periodic_checkpoints, candidate)
                end
                diagnostic = merge(
                    (key=checkpoint.key, mode=checkpoint.mode),
                    _sax_branch_terminal_diagnostics(po_branch, :gamma),
                )
                push!(periodic_branch_diagnostics, diagnostic)
                fold_candidates = count(sp -> sp.type == :fold, po_branch.specialpoint)
                pd_candidates = count(sp -> sp.type == :pd, po_branch.specialpoint)
                ns_candidates = count(sp -> sp.type == :ns, po_branch.specialpoint)
                verbosity > 0 && @info(
                    "Stage 2/3: periodic-orbit branch completed",
                    key=checkpoint.key,
                    mode=checkpoint.mode,
                    branch_points=length(po_branch),
                    fold_candidates,
                    pd_candidates,
                    ns_candidates,
                    elapsed_seconds=_sax_elapsed_seconds(periodic_started_ns),
                    periodic_branches=periodic_branch_count,
                )
                push!(completed_hopf_keys, checkpoint.key)
            end
            if cache_enabled
                payload = (
                    completed_hopf_keys=copy(completed_hopf_keys),
                    expected_hopf_keys=[item.key for item in hopf_checkpoints],
                    periodic_branch_count=periodic_branch_count,
                    periodic_checkpoints=copy(periodic_checkpoints),
                    periodic_branch_diagnostics=copy(periodic_branch_diagnostics),
                    failures=copy(stage2_failures),
                )
                _save_sax_stage_cache(
                    _sax_stage_cache_path(stage_cache_directory, :periodic),
                    :periodic,
                    payload,
                    model_p,
                    settings,
                )
            end
        end
    end

    # Stage 3: each fold/PD/NS component is committed separately. A hard stop during
    # one component leaves all earlier components reusable on the next run.
    curve_cache = cache_enabled && resume ? _load_sax_stage_cache(
        _sax_stage_cache_path(stage_cache_directory, :curves),
        :curves,
        model_p,
        settings,
    ) : (status=:missing, payload=nothing, reason="resume disabled")
    if curve_cache.status == :valid
        payload = curve_cache.payload
        completed_checkpoint_keys = String.(payload.completed_checkpoint_keys)
        fold_curves = Any[payload.fold_curves...]
        pd_curves = Any[payload.pd_curves...]
        ns_curves = Any[payload.ns_curves...]
        periodic_codim2_checkpoints = hasproperty(
            payload, :periodic_codim2_checkpoints) ?
            Any[payload.periodic_codim2_checkpoints...] : Any[]
        stage3_failures = Any[payload.failures...]
        verbosity > 0 && @info(
            "Stage 3/3 cache loaded",
            completed_components=length(completed_checkpoint_keys),
            fold_curves=length(fold_curves),
            pd_curves=length(pd_curves),
            ns_curves=length(ns_curves),
        )
    else
        completed_checkpoint_keys = String[]
        fold_curves = Any[]
        pd_curves = Any[]
        ns_curves = Any[]
        periodic_codim2_checkpoints = Any[]
        stage3_failures = Any[]
    end
    if include_periodic
        for curve in generalized_hopf_fold_curves
            _push_unique_curve!(fold_curves, curve)
        end
    end

    commit_stage3_cache!() = if cache_enabled
        payload = (
            completed_checkpoint_keys=copy(completed_checkpoint_keys),
            expected_checkpoint_keys=[item.key for item in periodic_checkpoints],
            fold_curves=_portable_sax_curve.(fold_curves),
            pd_curves=_portable_sax_curve.(pd_curves),
            ns_curves=_portable_sax_curve.(ns_curves),
            periodic_codim2_checkpoints=copy(periodic_codim2_checkpoints),
            failures=copy(stage3_failures),
        )
        _save_sax_stage_cache(
            _sax_stage_cache_path(stage_cache_directory, :curves),
            :curves,
            payload,
            model_p,
            settings,
        )
    end

    if include_periodic
        for checkpoint in periodic_checkpoints
            checkpoint.key in completed_checkpoint_keys && continue
            destination = checkpoint.type == :fold ? fold_curves :
                          checkpoint.type == :pd ? pd_curves : ns_curves
            covering_index = findfirst(
                curve -> _checkpoint_is_covered_by_curve(checkpoint, curve),
                destination,
            )
            if !isnothing(covering_index)
                push!(completed_checkpoint_keys, checkpoint.key)
                verbosity > 0 && @info(
                    "Stage 3/3: checkpoint already covered by cached curve",
                    key=checkpoint.key,
                    type=checkpoint.type,
                    mode=checkpoint.mode,
                    covering_curve=covering_index,
                )
                commit_stage3_cache!()
                continue
            end
            curve_started_ns = time_ns()
            verbosity > 0 && @info(
                "Stage 3/3: periodic bifurcation curve started",
                key=checkpoint.key,
                type=checkpoint.type,
                mode=checkpoint.mode,
            )
            curve_branch = try
                continue_sax_periodic_bifurcation_curve(
                    checkpoint,
                    model_p;
                    settings=settings,
                    verbosity=verbosity,
                )
            catch err
                err isa InterruptException && rethrow()
                _record_sax_stage_failure!(stage3_failures, (
                    stage=Symbol("$(checkpoint.type)_curve"),
                    mode=checkpoint.mode,
                    key=checkpoint.key,
                    exception_type=Symbol(nameof(typeof(err))),
                    error=sprint(showerror, err),
                ))
                verbosity > 0 && @warn(
                    "Periodic bifurcation-curve continuation failed; continuing the survey",
                    key=checkpoint.key,
                    type=checkpoint.type,
                    mode=checkpoint.mode,
                    exception_type=typeof(err),
                    error=sprint(showerror, err),
                    elapsed_seconds=_sax_elapsed_seconds(curve_started_ns),
                    captured_failures=length(stage3_failures),
                )
                nothing
            end
            if !isnothing(curve_branch)
                curve_stage = Symbol("$(checkpoint.type)_curve")
                _clear_sax_stage_failure!(
                    stage3_failures,
                    curve_stage,
                    checkpoint.key,
                )
                curve = _curve_summary(
                    curve_branch,
                    checkpoint.type;
                    mode=checkpoint.mode,
                    source=(checkpoint=checkpoint.key,
                            hopf=checkpoint.source_hopf_key),
                    active_parameter=:zeta,
                )
                _push_unique_curve!(destination, curve)
                if checkpoint.type == :pd
                    for candidate in _sax_pd_r2_neighbour_checkpoints(
                            curve_branch, checkpoint)
                        _push_unique_periodic_codim2_checkpoint!(
                            periodic_codim2_checkpoints, candidate)
                    end
                end
                verbosity > 0 && @info(
                    "Stage 3/3: periodic bifurcation curve completed",
                    key=checkpoint.key,
                    type=checkpoint.type,
                    mode=checkpoint.mode,
                    curve_points=length(curve_branch),
                    retained_curves=length(destination),
                    elapsed_seconds=_sax_elapsed_seconds(curve_started_ns),
                )
                push!(completed_checkpoint_keys, checkpoint.key)
            end
            commit_stage3_cache!()
        end
    end

    failures = vcat(stage1_failures, stage2_failures, stage3_failures)

    verbosity > 0 && @info(
        "Sax bifurcation analysis finished",
        elapsed_seconds=_sax_elapsed_seconds(analysis_started_ns),
        equilibrium_runs=equilibrium_run_count,
        hopf_curves=length(hopf_curves),
        double_hopf_points=length(double_hopf_points),
        generalized_hopf_points=length(generalized_hopf_points),
        periodic_branches=periodic_branch_count,
        fold_curves=length(fold_curves),
        pd_curves=length(pd_curves),
        ns_curves=length(ns_curves),
        captured_failures=length(failures),
    )

    return (
        settings=settings,
        model_parameters=model_p,
        equilibrium_runs=raw_equilibrium_runs,
        equilibrium_run_count=equilibrium_run_count,
        hopf_curves=hopf_curves,
        double_hopf_points=double_hopf_points,
        generalized_hopf_points=generalized_hopf_points,
        periodic_branches=raw_periodic_branches,
        periodic_branch_count=periodic_branch_count,
        periodic_branch_diagnostics=periodic_branch_diagnostics,
        periodic_codim2_checkpoints=periodic_codim2_checkpoints,
        fold_curves=fold_curves,
        pd_curves=pd_curves,
        ns_curves=ns_curves,
        failures=failures,
        metadata=(
            state_dimension=2 + 2settings.nmodes,
            jacobian=hasproperty(model_p, :sax_regularization) ?
                :analytic_regularized : :analytic_piecewise_smooth,
            periodic_discretization=:orthogonal_collocation,
            periodic_jacobian=settings.po_linear_solver == :condensed ?
                              :dense_analytical_inplace : :dense_analytical,
            periodic_linear_solver=settings.po_linear_solver,
            double_hopf_validation=:specialpoint_plus_independent_intersection,
            generalized_hopf_validation=:spectral_smoothness_and_normal_form,
            stage_cache_directory=stage_cache_directory,
            resumed=resume,
        ),
    )
end

"""
    sax_bifurcation_curve_data(result)

Remove solver objects and normal-form workspaces from a full result, retaining
only portable curve coordinates, validation diagnostics, settings, and failure
messages.  This representation is suitable for JLD2 output and for plotting in
a later Julia session without serializing BifurcationKit's function closures.
"""
function sax_bifurcation_curve_data(result)
    equilibrium_count = hasproperty(result, :equilibrium_run_count) ?
        result.equilibrium_run_count : length(result.equilibrium_runs)
    periodic_count = hasproperty(result, :periodic_branch_count) ?
        result.periodic_branch_count : length(result.periodic_branches)
    return (
        settings=_portable_sax_bifurcation_settings(result.settings),
        counts=(
            equilibrium_runs=equilibrium_count,
            hopf_curves=length(result.hopf_curves),
            double_hopf_points=length(result.double_hopf_points),
            generalized_hopf_points=length(result.generalized_hopf_points),
            periodic_branches=periodic_count,
            fold_curves=length(result.fold_curves),
            pd_curves=length(result.pd_curves),
            ns_curves=length(result.ns_curves),
            failures=length(result.failures),
        ),
        hopf_curves=_portable_sax_curve.(result.hopf_curves),
        double_hopf_points=_portable_sax_double_hopf.(result.double_hopf_points),
        generalized_hopf_points=
            _portable_sax_generalized_hopf.(result.generalized_hopf_points),
        periodic_branch_diagnostics=hasproperty(result, :periodic_branch_diagnostics) ?
            copy(result.periodic_branch_diagnostics) : Any[],
        periodic_codim2_checkpoints=
            hasproperty(result, :periodic_codim2_checkpoints) ?
            copy(result.periodic_codim2_checkpoints) : Any[],
        fold_curves=_portable_sax_curve.(result.fold_curves),
        pd_curves=_portable_sax_curve.(result.pd_curves),
        ns_curves=_portable_sax_curve.(result.ns_curves),
        failures=copy(result.failures),
        metadata=result.metadata,
    )
end

const SAX_BIFURCATION_CACHE_SCHEMA_VERSION = 3

function _sax_bifurcation_model_signature(model_p::NamedTuple, nmodes::Int)
    signature = (
        nmodes=nmodes,
        alpha=collect(Float64.(model_p.α[1:nmodes])),
        omega=collect(Float64.(model_p.ω[1:nmodes])),
        coupling=collect(Float64.(model_p.C[1:nmodes])),
    )
    hasproperty(model_p, :contact_stiffness) &&
        (signature = merge(signature, (
            contact_stiffness=float(model_p.contact_stiffness),)))
    return hasproperty(model_p, :sax_regularization) ?
        merge(signature, (sax_regularization=model_p.sax_regularization,)) :
        signature
end

"""
    save_sax_bifurcation_cache(path, result, model_p; fingering)

Atomically save the portable bifurcation curves and the exact model/settings
signature needed to decide whether the cache can be reused. Raw
BifurcationKit branches are intentionally excluded because they contain solver
workspaces and function closures that are not stable JLD2 interchange data.
"""
function save_sax_bifurcation_cache(path::AbstractString,
                                    result,
                                    model_p::NamedTuple;
                                    fingering::AbstractString)
    portable = sax_bifurcation_curve_data(result)
    cache = (
        schema_version=SAX_BIFURCATION_CACHE_SCHEMA_VERSION,
        fingering=String(fingering),
        settings_signature=portable.settings,
        model_signature=_sax_bifurcation_model_signature(
            model_p,
            portable.settings.nmodes,
        ),
        saved_at_unix=time(),
        bifurcation=portable,
    )

    _atomic_jld2_save(path; cache)
    return portable
end

"""
    load_sax_bifurcation_cache(path, model_p; fingering, settings)

Load a portable cache only when its schema, fingering, continuation settings,
and acoustic-mode parameters exactly match the requested calculation. The
return value has fields `status`, `result`, and `reason`; `status` is one of
`:valid`, `:missing`, `:incompatible`, or `:corrupt`.
"""
function load_sax_bifurcation_cache(path::AbstractString,
                                    model_p::NamedTuple;
                                    fingering::AbstractString,
                                    settings::SaxBifurcationSettings)
    isfile(path) || return (
        status=:missing,
        result=nothing,
        reason="cache file is absent",
    )

    stored = try
        data = Logging.with_logger(Logging.NullLogger()) do
            JLD2.load(path)
        end
        haskey(data, "cache") || error("missing cache payload")
        data["cache"]
    catch err
        return (
            status=:corrupt,
            result=nothing,
            reason=sprint(showerror, err),
        )
    end

    schema_matches = try
        stored.schema_version == SAX_BIFURCATION_CACHE_SCHEMA_VERSION
    catch err
        return (
            status=:corrupt,
            result=nothing,
            reason=sprint(showerror, err),
        )
    end
    schema_matches || return (
        status=:incompatible,
        result=nothing,
        reason="cache schema changed",
    )

    expected_signature = _sax_bifurcation_model_signature(
        model_p,
        settings.nmodes,
    )
    checks = try
        (
            (stored.fingering == String(fingering), "fingering changed"),
            (isequal(stored.settings_signature,
                     _portable_sax_bifurcation_settings(settings)),
             "continuation settings changed"),
            (isequal(stored.model_signature, expected_signature),
             "acoustic model parameters changed"),
        )
    catch err
        return (
            status=:corrupt,
            result=nothing,
            reason=sprint(showerror, err),
        )
    end
    for (valid, reason) in checks
        valid || return (
            status=:incompatible,
            result=nothing,
            reason=reason,
        )
    end
    result = try
        stored.bifurcation
    catch err
        return (
            status=:corrupt,
            result=nothing,
            reason=sprint(showerror, err),
        )
    end
    return (status=:valid, result=result, reason="")
end

const _SAX_HOPF_MODE_COLORS = (
    colorant"#D73027", colorant"#2166AC", colorant"#1A9850",
    colorant"#984EA3", colorant"#FF7F00", colorant"#A65628",
    colorant"#F781BF", colorant"#4D4D4D",
)

function _sax_mode_color(mode::Integer)
    1 <= mode <= length(_SAX_HOPF_MODE_COLORS) ?
        _SAX_HOPF_MODE_COLORS[mode] : colorant"#202020"
end

function _plot_sax_curves!(axis, result;
                           show_labels::Bool=true,
                           hopf_lw::Real=2.4,
                           fold_lw::Real=2.4,
                           fold_color=colorant"#018571",
                           fold_mode_colors=nothing,
                           fold_linestyle::Symbol=:dashdot,
                           pd_color=colorant"#E66101",
                           pd_mode_colors=nothing,
                           ns_color=colorant"#7B3294",
                           ns_mode_colors=nothing,
                           pd_lw::Real=2.2,
                           ns_lw::Real=2.2,
                           alpha::Real=0.95,
                           show_periodic_checkpoints::Bool=true,
                           show_invalid_hh::Bool=false,
                           show_invalid_gh::Bool=false)
    used_labels = Set{String}()
    label_once(text) = if show_labels && !(text in used_labels)
        push!(used_labels, text)
        text
    else
        ""
    end

    for curve in result.hopf_curves
        label = curve.mode > 0 ? "Hopf H$(curve.mode)" : "Hopf"
        plot!(axis, curve.gamma, curve.zeta;
              color=_sax_mode_color(curve.mode),
              linestyle=:solid,
              linewidth=hopf_lw,
              alpha=alpha,
              label=label_once(label))
    end
    fold_curve_color(curve) = if isnothing(fold_mode_colors) ||
            !hasproperty(curve, :mode)
        fold_color
    else
        get(fold_mode_colors, Int(curve.mode), fold_color)
    end
    for curve in result.fold_curves
        curve_color = fold_curve_color(curve)
        curve_label = if isnothing(fold_mode_colors) ||
                !hasproperty(curve, :mode) || curve.mode <= 0
            "fold of periodic orbits"
        else
            "Fold$(curve.mode)"
        end
        plot!(axis, curve.gamma, curve.zeta;
              color=curve_color,
              linestyle=fold_linestyle,
              linewidth=fold_lw,
              alpha=alpha,
              label=label_once(curve_label))
    end
    for curve in result.pd_curves
        curve_mode = hasproperty(curve, :mode) ? Int(curve.mode) : 0
        curve_color = isnothing(pd_mode_colors) ? pd_color :
            get(pd_mode_colors, curve_mode, pd_color)
        curve_label = isnothing(pd_mode_colors) || curve_mode <= 0 ?
            "period doubling" : "PD$(curve_mode)"
        plot!(axis, curve.gamma, curve.zeta;
              color=curve_color,
              linestyle=:dash,
              linewidth=pd_lw,
              alpha=alpha,
              label=label_once(curve_label))
    end
    for curve in result.ns_curves
        curve_mode = hasproperty(curve, :mode) ? Int(curve.mode) : 0
        curve_color = isnothing(ns_mode_colors) ? ns_color :
            get(ns_mode_colors, curve_mode, ns_color)
        curve_label = isnothing(ns_mode_colors) || curve_mode <= 0 ?
            "Neimark-Sacker" : "NS$(curve_mode)"
        plot!(axis, curve.gamma, curve.zeta;
              color=curve_color,
              linestyle=:dot,
              linewidth=ns_lw,
              alpha=alpha,
              label=label_once(curve_label))
    end
    if show_periodic_checkpoints &&
            hasproperty(result, :periodic_bifurcation_checkpoints)
        for checkpoint in result.periodic_bifurcation_checkpoints
            checkpoint.localization_status == :converged || continue
            marker, color, label = if checkpoint.type == :fold
                checkpoint_color = if isnothing(fold_mode_colors) ||
                        !hasproperty(checkpoint, :mode)
                    fold_color
                else
                    get(fold_mode_colors, Int(checkpoint.mode), fold_color)
                end
                (:rect, checkpoint_color, "localized fold seed")
            elseif checkpoint.type == :pd
                checkpoint_mode = hasproperty(checkpoint, :mode) ?
                    Int(checkpoint.mode) : 0
                checkpoint_color = isnothing(pd_mode_colors) ? pd_color :
                    get(pd_mode_colors, checkpoint_mode, pd_color)
                (:utriangle, checkpoint_color, "localized PD seed")
            elseif checkpoint.type == :ns
                checkpoint_mode = hasproperty(checkpoint, :mode) ?
                    Int(checkpoint.mode) : 0
                checkpoint_color = isnothing(ns_mode_colors) ? ns_color :
                    get(ns_mode_colors, checkpoint_mode, ns_color)
                (:circle, checkpoint_color, "localized NS seed")
            else
                continue
            end
            scatter!(axis, [checkpoint.gamma], [checkpoint.zeta];
                     marker=marker, markersize=7,
                     markercolor=:white, markerstrokecolor=color,
                     markerstrokewidth=2,
                     label=label_once(label))
        end
    end
    for point in result.double_hopf_points
        if point.valid
            scatter!(axis, [point.gamma], [point.zeta];
                     marker=:star5, markersize=10,
                     color=:black, markerstrokecolor=:white,
                     markerstrokewidth=0.8,
                     label=label_once("double Hopf"))
        elseif show_invalid_hh
            scatter!(axis, [point.gamma], [point.zeta];
                     marker=:xcross, markersize=7,
                     color=:gray35,
                     label=label_once("unvalidated HH candidate"))
        end
    end
    for point in result.generalized_hopf_points
        if point.valid
            scatter!(axis, [point.gamma], [point.zeta];
                     marker=:diamond, markersize=8,
                     color=colorant"#018571", markerstrokecolor=:white,
                     markerstrokewidth=0.8,
                     label=label_once("generalized Hopf"))
        elseif show_invalid_gh
            scatter!(axis, [point.gamma], [point.zeta];
                     marker=:diamond, markersize=6,
                     color=:gray55, markerstrokecolor=:gray20,
                     markerstrokewidth=0.6,
                     label=label_once("unvalidated GH candidate"))
        end
    end
    return axis
end

"""Plot Hopf, generalized/double-Hopf, fold, period-doubling, and NS loci."""
function plot_sax_bifurcation_diagram(result;
                                      title::AbstractString="Saxophone model bifurcation diagram",
                                      size::Tuple{Int,Int}=(850, 650),
                                      legend=:outerright,
                                      show_invalid_hh::Bool=false,
                                      show_invalid_gh::Bool=false,
                                      fold_color=colorant"#018571",
                                      fold_mode_colors=nothing,
                                      fold_linestyle::Symbol=:dashdot,
                                      pd_color=colorant"#E66101",
                                      pd_mode_colors=nothing,
                                      ns_color=colorant"#7B3294",
                                      ns_mode_colors=nothing,
                                      plot_kwargs...)
    settings = result.settings
    axis = plot(
        ;
        xlabel="γ",
        ylabel="ζ",
        xlims=settings.gamma_range,
        ylims=settings.zeta_range,
        title=title,
        size=size,
        legend=legend,
        framestyle=:box,
        grid=:lightgray,
        plot_kwargs...,
    )
    return _plot_sax_curves!(
        axis,
        result;
        show_labels=true,
        show_invalid_hh=show_invalid_hh,
        show_invalid_gh=show_invalid_gh,
        fold_color=fold_color,
        fold_mode_colors=fold_mode_colors,
        fold_linestyle=fold_linestyle,
        pd_color=pd_color,
        pd_mode_colors=pd_mode_colors,
        ns_color=ns_color,
        ns_mode_colors=ns_mode_colors,
    )
end

"""
    overlay_sax_bifurcation!(plot_object, result; subplot=1)

Overlay the bifurcation set on an existing plot.  `subplot=1` is the correct
default for the two-panel object returned by `plot_sweep_mode_regions_pattern`,
whose second panel is only the pattern legend.
"""
function overlay_sax_bifurcation!(plot_object, result;
                                  subplot::Int=1,
                                  show_labels::Bool=false,
                                  kwargs...)
    axis = length(plot_object.subplots) > 1 ? plot_object[subplot] : plot_object
    _plot_sax_curves!(axis, result; show_labels=show_labels, kwargs...)
    return plot_object
end

"""
    plot_sax_bifurcation_on_persistence(maps, result; patterned=true)

Build the existing mode-persistence figure and overlay the bifurcation curves
on its main `(gamma,zeta)` panel.  The persistence regions describe finite-
amplitude attractor selection; the overlaid curves describe local changes of
equilibria or periodic orbits.  They are complementary and need not coincide.
"""
function plot_sax_bifurcation_on_persistence(maps, result;
                                             patterned::Bool=true,
                                             show_labels::Bool=false,
                                             persistence_kwargs=NamedTuple(),
                                             overlay_kwargs...)
    figure = patterned ?
        plot_sweep_mode_regions_pattern(maps; persistence_kwargs...) :
        plot_sweep_mode_regions(maps; persistence_kwargs...)
    return overlay_sax_bifurcation!(
        figure,
        result;
        subplot=1,
        show_labels=show_labels,
        overlay_kwargs...,
    )
end

#endregion
