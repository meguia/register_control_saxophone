using Peaks, LsqFit, Optim


function base_model_complex(x::Vector{Float64}, p::Vector{Float64})
	h, w, ω = p
	@. h * w / (w + (x - ω) * 1im) + h * w / (w + (x + ω) * 1im)
end

function base_model_real(x::Vector{Float64}, p::Vector{Float64})
	abs.(base_model_complex(x, p))
end

function model(x::Vector{Float64}, p::Vector{Float64})
	abs.(sum([base_model_complex(x, p[i:i+2]) for i in 1:3:length(p)]))
end

function model_complex(x::Vector{Float64}, p::Vector{ComplexF64})
	Z(f, h, s) = @. real(h) * real(s) / (f * 1im - s) + real(h) * real(s) / (f * 1im - conj(s))
	sum([Z(x, (p[i], p[i+1])...) for i in 1:2:length(p)])
end

function fit_params(Z_c, x_range::StepRangeLen;
					r_lower_bounds = [.01, Inf, .01],
					r_upper_bounds = [.01, Inf, .01],
					ComplexFit = false)
	y = abs.(Z_c)
	
	i_max, m_max = findmaxima(y)
	n = length(i_max)

	i_min = vcat(1, findminima(y)[1], length(y))

	# p_a = [[a, b, c], [a, b, c], [a, b, c]]
	p_a::Vector{Vector{Float64}} = []
	for (i, j) in enumerate(i_max)
		push!(p_a, [getindex(m_max, i), .1, getindex(x_range, j)])
	end

	x = collect(x_range)
	
	h::Vector{Float64} = []
	w::Vector{Float64} = []
	ω::Vector{Float64} = []
	r::Vector{Float64} = []

	# Perform a first fit to get an initial guess for the parameters:
	# We fit the (base) model to each peak individually.
	for i in 1:n
		x_::Vector{Float64} = x[i_min[i]:i_min[i+1]]
		y_::Vector{Float64} = y[i_min[i]:i_min[i+1]]
		p_::Vector{Float64} = p_a[i]
		p_lower_bounds::Vector{Float64} = []
		p_upper_bounds::Vector{Float64} = []

		push!(p_lower_bounds, (p_ .* (1. .- r_lower_bounds))...)
		push!(p_upper_bounds, (p_ .* (1. .+ r_upper_bounds))...)

		fit_a = LsqFit.curve_fit(base_model_real, x_, y_, p_; lower = p_lower_bounds, upper = p_upper_bounds)
		push!.([h, w, ω], fit_a.param)
	end
	
	# Use the previous fit as an initial guess for the sum of peaks

	lower_end = i_min[1]
	upper_end = i_min[end]

	x__::Vector{Float64} = x[lower_end:upper_end]
	y__::Vector{Float64} = abs.(Z_c[lower_end:upper_end])
	p__::Vector{Float64} = []
	p__lower_bounds::Vector{Float64} = []
	p__upper_bounds::Vector{Float64} = []
	for i in 1:n
		push!(p__, h[i], w[i], ω[i])
		push!(p__lower_bounds, h[i] - abs(h[i]) * 0.1, w[i] - abs(w[i]) * 0.1, ω[i] - abs(ω[i]) * 0.1)
		push!(p__upper_bounds, h[i] + abs(h[i]) * 0.1, w[i] + abs(w[i]) * 0.1, ω[i] + abs(ω[i]) * 0.1)
	end

	for (i, p) in enumerate(p__)
		p >= p__lower_bounds[i] && p <= p__upper_bounds[i] || println("Warning: p[$i] = $p out of bounds (l = $(p__lower_bounds[i]), u = $(p__upper_bounds[i]))")
	end

	fit_b = LsqFit.curve_fit(model, x__, y__, p__)

	for i in 1:n
		h[i] = fit_b.param[3*i-2]
		w[i] = fit_b.param[3*i-1]
		ω[i] = fit_b.param[3*i]
	end

	
	# Complex fit using Optim
	if ComplexFit
		y___::Vector{ComplexF64} = Z_c
		p___::Vector{ComplexF64} = vcat([[h + 0im, w - ω * 1im] for (h, w, ω) in zip(h, w, ω)]...)

		function o_objective(p)
			r_ = sum((model_complex(x__, p) - y___) .^ 2)
			return (real(r_) - imag(r_)) ^ 2
		end

	fit_c = Optim.optimize(o_objective, p___ .* 0.1, p___ .* 1.9, p___, SAMIN(; rt = 0.95), Optim.Options(iterations = 10^7))

		for i in 1:n
			h[i] = real(fit_c.minimizer[2*i-1])
			w[i] = real(fit_c.minimizer[2*i])
			ω[i] = imag(fit_c.minimizer[2*i])
		end
	end
	
	return (ω = ω, α = w, C = h .* w, r = fit_b.resid)
end



"""
Build an impedance curve as a function of frequency from poles and residues.
"""
function Z(f, ω::Vector{Float64}, α::Vector{Float64}, C::Vector{Float64}; r_sum = true)
	s = -α + 1im * ω
	z = @. C' / (1im * f - s') + conj(C') / (1im * f - conj(s'))
	r = r_sum ? sum(z, dims = 2)[:, 1] : z
end
;