# Audio-free transcription of the saxophone equations in rt_sax_control.jl.
# The historical real-time file remains canonical and untouched; this copy
# lets batch analysis run without importing audio or serial packages.

function set_parameters(γ::Float64, ζ::Float64,
                        model_p::NamedTuple, nmodes::Int64)
    nparfix = 2
    pars = zeros(nparfix + 3nmodes)
    pars[1:nparfix] = [γ, ζ]
    pars[nparfix + 1:nparfix + nmodes] = model_p.α[1:nmodes]
    pars[nparfix + nmodes + 1:nparfix + 2nmodes] = model_p.ω[1:nmodes]
    pars[nparfix + 2nmodes + 1:nparfix + 3nmodes] = model_p.C[1:nmodes]
    return pars
end

function saxRN!(dx, x, p, t)
    nparfix = 2
    nmodes = Int(length(x) / 2 - 1)
    γ, ζ = p[1:nparfix]
    α = p[nparfix + 1:nparfix + nmodes]
    ω = p[nparfix + nmodes + 1:nparfix + 2nmodes]
    C = p[nparfix + 2nmodes + 1:nparfix + 3nmodes]
    P = sum(x[3:2:end])
    Fc = 100.0 * min(real(x[1]) + 1, 0)^2 * (1 - x[2])
    u = ζ * max(real(x[1]) + 1, 0) * sign(γ - P) * sqrt(abs(γ - P))
    dx[1] = x[2]
    dx[2] = -4.224x[2] + 17.842176 * (P - γ - x[1] + Fc)
    @inbounds for mode in 1:nmodes
        index = 2mode + 1
        dx[index] = -α[mode] * x[index] - 2ω[mode] * x[index + 1] +
                    2C[mode] * u
        dx[index + 1] = -α[mode] * x[index + 1] + 0.5ω[mode] * x[index]
    end
    return dx
end

function _mode_amplitude_series(
        states::AbstractMatrix{<:Real},
        mode::Integer;
        center::Union{Nothing,Tuple{<:Real,<:Real}}=nothing)
    index = 2mode + 1
    @assert index + 1 <= size(states, 1) "Requested mode index exceeds state dimension"
    c1, c2 = isnothing(center) ? (0.0, 0.0) :
        (float(center[1]), float(center[2]))
    return sqrt.((states[index, :] .- c1).^2 .+
                 (states[index + 1, :] .- c2).^2)
end

"""Estimate the analytic equilibrium state and its vector-field residual."""
function _estimate_fixed_point(
        γ::Real,
        ζ::Real,
        model_p::NamedTuple;
        nmodes::Int=8,
        pert::Real=0.0)
    if hasproperty(model_p, :sax_regularization)
        isdefined(@__MODULE__, :sax_regularized_fixed_point) || error(
            "the regularized model layer must be included before estimating this fixed point",
        )
        return sax_regularized_fixed_point(
            γ, ζ, model_p; nmodes=nmodes, pert=pert)
    end
    γf, ζf = float(γ), float(ζ)
    α = float.(model_p.α[1:nmodes])
    ω = float.(model_p.ω[1:nmodes])
    C = float.(model_p.C[1:nmodes])
    A = @. (α + ω * ω / α) / (2.0 * C * ζf)

    lower = 0.0
    upper = min(γf, 0.002nmodes)
    lower < upper || error(
        "Empty feasible interval for p: [$(lower), $(upper)] (gamma=$(γ))",
    )
    K = sum(inv.(A))
    residual_function(p) =
        K * (p - γf + 1.0) * sqrt(max(γf - p, 0.0)) - p
    lower_value = residual_function(lower)
    upper_value = residual_function(upper)
    lower_value * upper_value <= 0 || error(
        "No sign change in equilibrium bracket at gamma=$(γ), zeta=$(ζ)",
    )

    atol, rtol = 1e-12, 1e-10
    for _ in 1:100
        middle = 0.5 * (lower + upper)
        middle_value = residual_function(middle)
        if abs(middle_value) <= atol ||
                (upper - lower) <= max(
                    atol, rtol * max(abs(lower), abs(upper), 1.0))
            lower = upper = middle
            break
        end
        if signbit(lower_value) == signbit(middle_value)
            lower, lower_value = middle, middle_value
        else
            upper, upper_value = middle, middle_value
        end
    end

    pstar = 0.5 * (lower + upper)
    drive = (pstar - γf + 1.0) * sqrt(max(γf - pstar, 0.0))
    p0 = max.(drive ./ A, 0.0)
    q0 = @. 0.5 * p0 * ω / α
    x0 = sum(p0) - γf + pert
    state = vcat([x0, 0.0], collect(Iterators.flatten(zip(p0, q0))))

    parameters = set_parameters(γf, ζf, model_p, Int64(nmodes))
    derivative = similar(state)
    saxRN!(derivative, state, parameters, 0.0)
    return state, sqrt(sum(abs2, derivative))
end
