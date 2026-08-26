# Smooth saxophone equations used by the parallel regularization study.
#
# This file is intentionally separate from the historical, piecewise model.
# The continuation infrastructure dispatches here only when `model_p` carries
# the `sax_regularization` field created by `regularize_sax_model`.

const SAX_REGULARIZATION_SCHEMA_VERSION = 1
const SAX_REGULARIZATION_LAW = :colinot_smooth_absolute_value

"""
    SaxRegularizationSettings(; eta=1e-3)

Numerical definition of the smooth saxophone model. Following Colinot, every
absolute value is replaced by `sqrt(z^2 + eta)`. The same positive `eta` is
used for reed opening, reed contact, and flow reversal unless one of the three
component values is explicitly overridden.

`eta` is added to a squared dimensionless variable, so it is not the square of
a smoothing width. The commonly used reference value is `1e-3`.
"""
Base.@kwdef struct SaxRegularizationSettings
    schema_version::Int = SAX_REGULARIZATION_SCHEMA_VERSION
    law::Symbol = SAX_REGULARIZATION_LAW
    eta::Float64 = 1e-3
    eta_opening::Float64 = eta
    eta_contact::Float64 = eta
    eta_flow::Float64 = eta
end

function _validate_sax_regularization(settings::SaxRegularizationSettings)
    settings.schema_version == SAX_REGULARIZATION_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported saxophone regularization schema"))
    settings.law == SAX_REGULARIZATION_LAW ||
        throw(ArgumentError("unsupported saxophone regularization law $(settings.law)"))
    all(>(0), (settings.eta, settings.eta_opening,
               settings.eta_contact, settings.eta_flow)) ||
        throw(ArgumentError("all regularization eta values must be positive"))
    return settings
end

"""Return a serialization-stable signature for a regularization choice."""
function sax_regularization_signature(settings::SaxRegularizationSettings)
    _validate_sax_regularization(settings)
    return (
        schema_version=settings.schema_version,
        law=settings.law,
        eta=settings.eta,
        eta_opening=settings.eta_opening,
        eta_contact=settings.eta_contact,
        eta_flow=settings.eta_flow,
    )
end

"""
    regularize_sax_model(model_p; eta=1e-3, settings=nothing)

Return a copy of the acoustic parameter tuple tagged for the smooth equations.
The original `model_p` is never mutated. The tag is propagated into all JLD2
model signatures, which prevents regularized and piecewise caches from being
mistaken for one another.
"""
function regularize_sax_model(
        model_p::NamedTuple;
        eta::Real=1e-3,
        settings::Union{Nothing,SaxRegularizationSettings}=nothing)
    selected = isnothing(settings) ?
        SaxRegularizationSettings(eta=float(eta)) : settings
    signature = sax_regularization_signature(selected)
    return merge(model_p, (sax_regularization=signature,))
end

is_regularized_sax_model(model_p::NamedTuple) =
    hasproperty(model_p, :sax_regularization)

function _sax_regularization(source::NamedTuple)
    hasproperty(source, :sax_regularization) || return nothing
    regularization = source.sax_regularization
    regularization.law == SAX_REGULARIZATION_LAW || throw(ArgumentError(
        "unsupported saxophone regularization law $(regularization.law)",
    ))
    all(>(0), (regularization.eta_opening,
               regularization.eta_contact,
               regularization.eta_flow)) || throw(ArgumentError(
        "all regularization eta values must be positive",
    ))
    return regularization
end

"""Colinot's smooth approximation of `abs(x)`."""
sax_smooth_absolute(x, eta::Real) = sqrt(x * x + eta)

"""Colinot's smooth approximation of `max(x, 0)`."""
sax_smooth_positive(x, eta::Real) =
    (x + sax_smooth_absolute(x, eta)) / 2

"""Smooth approximation of `min(x, 0)` using the same absolute value."""
sax_smooth_negative(x, eta::Real) =
    (x - sax_smooth_absolute(x, eta)) / 2

"""Derivative of `sax_smooth_positive` with respect to its first argument."""
sax_smooth_positive_derivative(x, eta::Real) =
    (one(x) + x / sax_smooth_absolute(x, eta)) / 2

"""Derivative of `sax_smooth_negative` with respect to its first argument."""
sax_smooth_negative_derivative(x, eta::Real) =
    (one(x) - x / sax_smooth_absolute(x, eta)) / 2

"""
    sax_smooth_signed_square_root(d, eta)

Smooth odd replacement for `sign(d) * sqrt(abs(d))`. Replacing
`abs(d)` by `sqrt(d^2 + eta)` and `sign(d)` by
`d / sqrt(d^2 + eta)` gives exactly `d / (d^2 + eta)^(1/4)`.
Unlike smoothing the absolute value while retaining `sign(d)`, this function
is continuous, differentiable, and zero at flow reversal.
"""
sax_smooth_signed_square_root(d, eta::Real) =
    d / sqrt(sax_smooth_absolute(d, eta))

"""Derivative of `sax_smooth_signed_square_root` with respect to `d`."""
function sax_smooth_signed_square_root_derivative(d, eta::Real)
    base = d * d + eta
    return (eta + d * d / 2) / base^(5 / 4)
end

"""
    sax_regularized_dynamics!(du, u, p, t=nothing)

Evaluate the Colinot-regularized saxophone vector field in place. `p` must be
the named parameter tuple created by `sax_bifurcation_parameters`. The unused
time argument gives the function the standard `ODEProblem` signature while
the same implementation remains usable by continuation residuals.
"""
function sax_regularized_dynamics!(
        du::AbstractVector,
        u::AbstractVector,
        p::NamedTuple,
        t=nothing)
    regularization = something(_sax_regularization(p))
    n = 2 + 2p.nmodes
    length(u) == n || throw(DimensionMismatch(
        "expected $n states, received $(length(u))"))
    length(du) == n || throw(DimensionMismatch(
        "expected $n derivative entries, received $(length(du))"))
    pressure = sum(@view u[3:2:end])
    reed_coordinate = real(u[1]) + one(real(u[1]))
    closed_part = sax_smooth_negative(
        reed_coordinate, regularization.eta_contact)
    contact_force = p.contact_stiffness * closed_part^2 *
        (one(eltype(du)) - u[2])
    opening = sax_smooth_positive(
        reed_coordinate, regularization.eta_opening)
    pressure_drop = p.gamma - pressure
    flow = p.zeta * opening * sax_smooth_signed_square_root(
        pressure_drop, regularization.eta_flow)

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
    sax_regularized_bifurcation_residual(u, p)

Smooth 18-state saxophone vector field. `p` must contain the acoustic fields
created by `sax_bifurcation_parameters` and a `sax_regularization` signature.
"""
function sax_regularized_bifurcation_residual(
        u::AbstractVector,
        p::NamedTuple)
    T = promote_type(eltype(u), typeof(p.gamma), typeof(p.zeta))
    du = zeros(T, length(u))
    return sax_regularized_dynamics!(du, u, p)
end

"""Fill the exact analytic Jacobian of the regularized vector field."""
function sax_regularized_bifurcation_jacobian!(
        J::AbstractMatrix,
        u::AbstractVector,
        p::NamedTuple)
    regularization = something(_sax_regularization(p))
    n = 2 + 2p.nmodes
    size(J) == (n, n) || throw(DimensionMismatch(
        "Jacobian must have size ($n, $n)"))
    length(u) == n || throw(DimensionMismatch(
        "expected $n states, received $(length(u))"))
    fill!(J, zero(eltype(J)))

    pressure = sum(@view u[3:2:end])
    reed_coordinate = real(u[1]) + one(real(u[1]))
    pressure_drop = real(p.gamma - pressure)

    closed_part = sax_smooth_negative(
        reed_coordinate, regularization.eta_contact)
    dclosed_du1 = sax_smooth_negative_derivative(
        reed_coordinate, regularization.eta_contact)
    dcontact_du1 = 2p.contact_stiffness * closed_part * dclosed_du1 *
                   (one(eltype(J)) - u[2])
    dcontact_du2 = -p.contact_stiffness * closed_part^2

    opening = sax_smooth_positive(
        reed_coordinate, regularization.eta_opening)
    dopening_du1 = sax_smooth_positive_derivative(
        reed_coordinate, regularization.eta_opening)
    root_flow = sax_smooth_signed_square_root(
        pressure_drop, regularization.eta_flow)
    droot_dd = sax_smooth_signed_square_root_derivative(
        pressure_drop, regularization.eta_flow)
    dflow_du1 = p.zeta * dopening_du1 * root_flow
    dflow_dpressure_state = -p.zeta * opening * droot_dd

    J[1, 2] = one(eltype(J))
    J[2, 1] = p.reed_stiffness * (-one(eltype(J)) + dcontact_du1)
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
            J[i, 2source_mode + 1] +=
                drive_gain * dflow_dpressure_state
        end
        J[i + 1, i] = 0.5p.omega[mode]
        J[i + 1, i + 1] = -p.alpha[mode]
    end
    return J
end

"""
    sax_regularized_fixed_point(gamma, zeta, model_p; nmodes=8, pert=0)

Compute the regularized equilibrium from its two scalar balance equations and
return the 18-state vector plus the full vector-field residual norm. A damped
Newton iteration is used only for this inexpensive seed calculation.
"""
function sax_regularized_fixed_point(
        gamma::Real,
        zeta::Real,
        model_p::NamedTuple;
        nmodes::Int=8,
        pert::Real=0.0)
    regularization = something(_sax_regularization(model_p))
    gamma_f, zeta_f = float(gamma), float(zeta)
    alpha = collect(float.(model_p.α[1:nmodes]))
    omega = collect(float.(model_p.ω[1:nmodes]))
    coupling = collect(float.(model_p.C[1:nmodes]))
    A = @. (alpha + omega * omega / alpha) / (2coupling * zeta_f)
    K = sum(inv.(A))

    # The raw open-reed equilibrium is an effective initial guess. The smooth
    # equations are then solved directly, including the regularized contact.
    pressure = clamp(min(0.002nmodes, gamma_f / 2), 0.0, gamma_f)
    reed_state = pressure - gamma_f
    residual_norm = Inf
    for _ in 1:80
        reed_coordinate = reed_state + 1
        pressure_drop = gamma_f - pressure
        opening = sax_smooth_positive(
            reed_coordinate, regularization.eta_opening)
        dopening = sax_smooth_positive_derivative(
            reed_coordinate, regularization.eta_opening)
        closed = sax_smooth_negative(
            reed_coordinate, regularization.eta_contact)
        dclosed = sax_smooth_negative_derivative(
            reed_coordinate, regularization.eta_contact)
        root_flow = sax_smooth_signed_square_root(
            pressure_drop, regularization.eta_flow)
        droot = sax_smooth_signed_square_root_derivative(
            pressure_drop, regularization.eta_flow)

        f1 = pressure - K * opening * root_flow
        f2 = reed_state - pressure + gamma_f -
             100.0 * closed^2
        residual_norm = hypot(f1, f2)
        residual_norm <= 1e-13 && break
        jacobian = [
            -K * dopening * root_flow  1 + K * opening * droot
            1 - 200.0 * closed * dclosed  -1
        ]
        step = jacobian \ [f1, f2]
        accepted = false
        damping = 1.0
        for _ in 1:20
            candidate_x = reed_state - damping * step[1]
            candidate_p = pressure - damping * step[2]
            candidate_r = candidate_x + 1
            candidate_d = gamma_f - candidate_p
            c1 = candidate_p - K * sax_smooth_positive(
                candidate_r, regularization.eta_opening) *
                sax_smooth_signed_square_root(
                    candidate_d, regularization.eta_flow)
            candidate_closed = sax_smooth_negative(
                candidate_r, regularization.eta_contact)
            c2 = candidate_x - candidate_p + gamma_f -
                 100.0 * candidate_closed^2
            if hypot(c1, c2) < residual_norm
                reed_state, pressure = candidate_x, candidate_p
                accepted = true
                break
            end
            damping /= 2
        end
        accepted || error(
            "regularized equilibrium Newton line search failed at gamma=$(gamma), zeta=$(zeta)")
    end
    residual_norm <= 1e-9 || error(
        "regularized equilibrium did not converge at gamma=$(gamma), zeta=$(zeta); residual=$(residual_norm)")

    reed_coordinate = reed_state + 1
    pressure_drop = gamma_f - pressure
    normalized_flow = sax_smooth_positive(
        reed_coordinate, regularization.eta_opening) *
        sax_smooth_signed_square_root(
            pressure_drop, regularization.eta_flow)
    pressure_modes = normalized_flow ./ A
    quadratures = @. 0.5 * pressure_modes * omega / alpha
    state = vcat(
        [reed_state + float(pert), 0.0],
        collect(Iterators.flatten(zip(pressure_modes, quadratures))),
    )
    parameters = (
        gamma=gamma_f,
        zeta=zeta_f,
        alpha=alpha,
        omega=omega,
        coupling=coupling,
        nmodes=nmodes,
        reed_damping=4.224,
        reed_stiffness=17.842176,
        contact_stiffness=hasproperty(model_p, :contact_stiffness) ?
            float(model_p.contact_stiffness) : 100.0,
        sax_regularization=regularization,
    )
    derivative = sax_regularized_bifurcation_residual(state, parameters)
    return state, norm(derivative)
end

"""
    sax_regularization_diagnostics(; eta=1e-3, samples=2001)

Quantify the approximation error and the finite slope at both former
singularities. This is a numerical-model diagnostic, not a bifurcation result.
"""
function sax_regularization_diagnostics(;
        eta::Real=1e-3,
        samples::Int=2001,
        interval::Tuple{<:Real,<:Real}=(-1.0, 1.0))
    eta > 0 || throw(ArgumentError("eta must be positive"))
    samples >= 3 || throw(ArgumentError("samples must be at least three"))
    grid = range(float(interval[1]), float(interval[2]); length=samples)
    raw_opening = max.(grid, 0)
    smooth_opening = sax_smooth_positive.(grid, eta)
    raw_flow = sign.(grid) .* sqrt.(abs.(grid))
    smooth_flow = sax_smooth_signed_square_root.(grid, eta)
    return (
        eta=float(eta),
        maximum_opening_error=maximum(abs.(smooth_opening .- raw_opening)),
        maximum_flow_error=maximum(abs.(smooth_flow .- raw_flow)),
        opening_at_closure=sax_smooth_positive(0.0, eta),
        flow_at_reversal=sax_smooth_signed_square_root(0.0, eta),
        opening_slope_at_closure=sax_smooth_positive_derivative(0.0, eta),
        flow_slope_at_reversal=
            sax_smooth_signed_square_root_derivative(0.0, eta),
    )
end
