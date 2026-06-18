### A Pluto.jl notebook ###
# v0.20.27

using Markdown
using InteractiveUtils

# ╔═╡ ca732298-4177-48cd-8d51-9dbbcbd9a134
import Pkg; Pkg.activate()

# ╔═╡ bab5ca50-2d22-11f1-1168-f7d34e6cb5e7
using Plots, ProjectRoot, JLD2, Measures

# ╔═╡ 7ffe4963-f7d9-410d-b19b-345bdc3bb787
include("../rt_sax_experiment_analysis.jl");

# ╔═╡ ce8f2d66-85d7-4407-9813-b39a892072b8
session_path = @projectroot("src","sessions")

# ╔═╡ c4a86461-98f1-466d-adef-604954a95905
data_path = joinpath(session_path,"processed_data")

# ╔═╡ 511948c0-5638-415c-aa88-64a0efc8da50
# This is necessary because the jld2 was saved by the REPL
if !isdefined(Main, :Trial)
	@eval Main const Trial = $(Trial)
end

# ╔═╡ 1e440798-1951-4bf2-ae32-a62d8fef0541
subjects = list_subjects(session_path,"rt_sax_experiment_sept_2025.csv")

# ╔═╡ 76374dba-6d99-43b4-8db1-b336e27b8857
@load joinpath(data_path,"all_model_trials_wamplitudes_onoff.jld2") model_trials

# ╔═╡ 6c07aa19-c5dd-4c6a-beb5-67c5cb1f485c
@load joinpath(data_path,"all_real_trials_wamplitudes_onoff.jld2") real_trials

# ╔═╡ 1cca61ee-aa4b-4ffa-bb3d-e0c909649ff3
# correction (temporal)
M83 = filter_trials(real_trials, subject_ids = ["83"], blocks = [2],task=:Overtone)[1]

# ╔═╡ 3b1043d1-c7ea-469b-841e-1d3de32d911f
M83.success = false

# ╔═╡ 018b575d-9e1b-47a0-a6fa-5d14e5dc2cb9
begin
	PLOT_ALL_LEFT_MARGIN = 12mm
	PLOT_ALL_RIGHT_MARGIN = 0mm
	PLOT_ALL_TOP_MARGIN = 0mm
	PLOT_ALL_BOTTOM_MARGIN = 0mm
	PLOT_ALL_FAILED_ALPHA = 0.3
	PLOT_ALL_SUCCESS_ALPHA = 1.0
	PLOT_ALL_XLABEL = "Time (ms)"
	PLOT_ALL_NCOLS = 2
	PLOT_ALL_PANEL_HEIGHT = 120

	function plot_task_wregions_03(
		trial_list, task, subject_list, condition;
		trim_start_ms::Real = 0.0,
		trim_end_ms::Real = 0.0,
		display = false,
		remove = Int[],
		ncols = PLOT_ALL_NCOLS,
		overlay_controls = true,
		show_trim_guides::Bool = false,
		controls_alpha = 0.8,
		controls_ls = :dot,
		use_absmax = true,
		left_margin = PLOT_ALL_LEFT_MARGIN,
		right_margin = PLOT_ALL_RIGHT_MARGIN,
		top_margin = PLOT_ALL_TOP_MARGIN,
		bottom_margin = PLOT_ALL_BOTTOM_MARGIN,
		failed_alpha = PLOT_ALL_FAILED_ALPHA,
		success_alpha = PLOT_ALL_SUCCESS_ALPHA,
		xlabel_text = PLOT_ALL_XLABEL,
	)
		trials_all = filter_trials(trial_list; task = task, subject_ids = subject_list, success_only = false)
		trials = copy(trials_all)
		sort!(trials, by = tr -> (something(tryparse(Int, tr.subject_id), typemax(Int)), tr.block, tr.order))
		isempty(remove) || deleteat!(trials, sort(unique(remove)))

		plts = []

		for tr in trials
			t = tr.t
			a1 = tr.a1
			a2 = tr.a2
			(isempty(a1) || isempty(a2)) && continue

			ymax = max(maximum(a1), maximum(a2))
			line_alpha = tr.success ? success_alpha : failed_alpha

			p = plot(t, a1; c = :red, lw = 1.5, alpha = line_alpha, label = "")
			plot!(p, t, a2; c = :blue, lw = 1.5, alpha = line_alpha, label = "")

			if show_trim_guides
				trim_start_ms > 0.0 && vline!(p, [first(t) + trim_start_ms]; c = :black, ls = :dash, alpha = 0.45, lw = 1.2, label = "")
				trim_end_ms > 0.0 && vline!(p, [last(t) - trim_end_ms]; c = :black, ls = :dash, alpha = 0.45, lw = 1.2, label = "")
			end

			if overlay_controls && !isempty(tr.v1) && !isempty(tr.v2)
				t1 = tr.t1
				t2 = tr.t2
				s1 = use_absmax ? maximum(abs.(tr.v1)) : maximum(tr.v1)
				s2 = use_absmax ? maximum(abs.(tr.v2)) : maximum(tr.v2)
				s1 > 0 && plot!(p, t1, tr.v1 .* (ymax / s1); c = :magenta, lw = 1.2, ls = controls_ls, alpha = controls_alpha * line_alpha, label = "")
				s2 > 0 && plot!(p, t2, tr.v2 .* (ymax / s2); c = :green, lw = 1.2, ls = controls_ls, alpha = controls_alpha * line_alpha, label = "")
			end

			txtcolor = tr.success ? :black : :red

			if tr.success
				tmin, tmax = first(t), last(t)
				regions1 = Tuple{Float64,Float64,Float64}[]
				regions2 = Tuple{Float64,Float64,Float64}[]
				for (ts, te, m) in onoff_regions(tr; mode = 1)
					ts_eff = clamp(1000 * ts + trim_start_ms, tmin, tmax)
					te_eff = clamp(1000 * te - trim_end_ms, tmin, tmax)
					ts_eff < te_eff || continue
					push!(regions1, (ts_eff, te_eff, m))
				end
				for (ts, te, m) in onoff_regions(tr; mode = 2)
					ts_eff = clamp(1000 * ts + trim_start_ms, tmin, tmax)
					te_eff = clamp(1000 * te - trim_end_ms, tmin, tmax)
					ts_eff < te_eff || continue
					push!(regions2, (ts_eff, te_eff, m))
				end

				up1 = [argmin(abs.(t .- seg[1])) for seg in regions1]
				down1 = [argmin(abs.(t .- seg[2])) for seg in regions1]
				up2 = [argmin(abs.(t .- seg[1])) for seg in regions2]
				down2 = [argmin(abs.(t .- seg[2])) for seg in regions2]

				scatter!(p, t[up1], a1[up1]; c = :white, m = :circle, msw = 1, msc = :red, alpha = 0.7, label = "")
				scatter!(p, t[down1], a1[down1]; c = :white, m = :square, msw = 1, msc = :red, alpha = 0.7, label = "")
				scatter!(p, t[up2], a2[up2]; c = :white, m = :circle, msw = 1, msc = :blue, alpha = 0.7, label = "")
				scatter!(p, t[down2], a2[down2]; c = :white, m = :square, msw = 1, msc = :blue, alpha = 0.7, label = "")
			end

			xmin, xmax = first(t), last(t)
			annotate!(p, xmin + 0.05 * (xmax - xmin), 0.85 * ymax, text(tr.subject_id, txtcolor, 8))
			push!(plts, p)
		end

		nrows = ceil(Int, length(plts) / ncols)
		for (i, p) in enumerate(plts)
			is_bottom = i > (nrows - 1) * ncols
			xlabel!(p, is_bottom ? xlabel_text : "")
		end 
		label_task = Dict(
    		"NonlegatoAsc"  => "Non-legato Ascending",
    		"NonlegatoDesc" => "Non-legato Descending",
			"LegatoAsc"  => "Legato Ascending",
    		"LegatoDesc" => "Legato Descending",
			"Overtone" => "Multiphonic"
		)	
		label_condition = Dict(
			"Model" => "Model",
			"Instrument" => "Real-instrument"
		)
		function make_label(label::String, label_dict::Dict)
    		return label_dict[label]
		end
		pp = plot(
			plts...;
			layout = (nrows, ncols),
			size = (1000, PLOT_ALL_PANEL_HEIGHT * nrows),
			plot_title =  make_label(String(task),label_task) * " | " * make_label(condition, label_condition),
			left_margin = left_margin,
			right_margin = right_margin,
			top_margin = top_margin,
			bottom_margin = bottom_margin,
		)

		display && return pp
		savefig(pp, "Figures/" * condition * "_" * String(task) * "_regions.png")
		return nothing
	end
end

# ╔═╡ 801e458e-3842-4d4e-8baa-5cbaea93df81
fill_onoff_from_onsets!(model_trials; thr2=0.1)

# ╔═╡ 5d07d364-3665-4178-9cf3-0bc60999f4be
fill_onoff_from_onsets!(real_trials)

# ╔═╡ fd386a54-1bfa-449e-8e7e-b72dee34f216
begin
	fill_onoff_from_onsets!(model_trials; thr2=0.8)
	plot_task_wregions_03(model_trials, :NonlegatoAsc, subjects,"Model";display=true,trim_start_ms=100.0, trim_end_ms=100.0)
end	

# ╔═╡ 3c02f76e-a7aa-48cc-818a-17de91b00c75
plot_task_wregions_03(model_trials, :NonlegatoDesc, subjects,"Model";display=true,trim_start_ms=100.0, trim_end_ms=100.0)

# ╔═╡ 154bcd97-27b4-491c-a3cc-dfa6f1435e41
plot_task_wregions_03(real_trials, :NonlegatoAsc, subjects,"Instrument";display=true)

# ╔═╡ 8090c1d9-9dd9-4737-a0df-a1e886e3b08d
plot_task_wregions_03(real_trials, :NonlegatoDesc, subjects,"Instrument";display=true)

# ╔═╡ dec5238a-2359-4ba1-ab42-a1b5b8f38587
plot_task_wregions_03(model_trials, :LegatoDesc, subjects,"Model";display=true,remove=[])

# ╔═╡ f0ff4170-a0b2-4064-8702-cafaa511bc9b
plot_task_wregions_03(model_trials, :LegatoAsc, subjects,"Model";display=true,remove=[])

# ╔═╡ 99e9c24c-3797-4272-bef9-493032e82c5b
plot_task_wregions_03(real_trials, :LegatoDesc, subjects,"Instrument";display=true)

# ╔═╡ d6e314c4-aaab-4984-836e-d766e29d8755
plot_task_wregions_03(real_trials, :LegatoAsc, subjects,"Instrument";display=true)

# ╔═╡ 523aea75-02ed-44ce-b4da-4e496f7407da
plot_task_wregions_03(real_trials, :Overtone, subjects,"Instrument";display=true)

# ╔═╡ 26b1a9d8-e196-4d0d-a0f1-d9c9c72ff174
plot_task_wregions_03(model_trials, :Overtone, subjects,"Model";display=true)

# ╔═╡ e3103880-8f7d-40a6-85ac-5bae9d3479b3
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 1400px;
    	padding-left: max(160px, 10%);
    	padding-right: max(160px, 10%);
	}
	input[type*="range"] {
		width: 90%;
	}
</style>
"""

# ╔═╡ Cell order:
# ╠═ca732298-4177-48cd-8d51-9dbbcbd9a134
# ╠═bab5ca50-2d22-11f1-1168-f7d34e6cb5e7
# ╠═7ffe4963-f7d9-410d-b19b-345bdc3bb787
# ╠═ce8f2d66-85d7-4407-9813-b39a892072b8
# ╠═c4a86461-98f1-466d-adef-604954a95905
# ╠═511948c0-5638-415c-aa88-64a0efc8da50
# ╠═1e440798-1951-4bf2-ae32-a62d8fef0541
# ╠═76374dba-6d99-43b4-8db1-b336e27b8857
# ╠═6c07aa19-c5dd-4c6a-beb5-67c5cb1f485c
# ╠═1cca61ee-aa4b-4ffa-bb3d-e0c909649ff3
# ╠═3b1043d1-c7ea-469b-841e-1d3de32d911f
# ╠═018b575d-9e1b-47a0-a6fa-5d14e5dc2cb9
# ╠═801e458e-3842-4d4e-8baa-5cbaea93df81
# ╠═5d07d364-3665-4178-9cf3-0bc60999f4be
# ╠═fd386a54-1bfa-449e-8e7e-b72dee34f216
# ╠═3c02f76e-a7aa-48cc-818a-17de91b00c75
# ╠═154bcd97-27b4-491c-a3cc-dfa6f1435e41
# ╠═8090c1d9-9dd9-4737-a0df-a1e886e3b08d
# ╠═dec5238a-2359-4ba1-ab42-a1b5b8f38587
# ╠═f0ff4170-a0b2-4064-8702-cafaa511bc9b
# ╠═99e9c24c-3797-4272-bef9-493032e82c5b
# ╠═d6e314c4-aaab-4984-836e-d766e29d8755
# ╠═523aea75-02ed-44ce-b4da-4e496f7407da
# ╠═26b1a9d8-e196-4d0d-a0f1-d9c9c72ff174
# ╠═e3103880-8f7d-40a6-85ac-5bae9d3479b3
