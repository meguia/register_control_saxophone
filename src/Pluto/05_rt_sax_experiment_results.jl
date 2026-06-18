### A Pluto.jl notebook ###
# v0.20.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 5d954762-35ab-11f1-2308-851b8fbd36ac
import Pkg; Pkg.activate()

# ╔═╡ c5f76341-be5a-40e6-bc4e-ab2eff0ac7f5
using Plots, ProjectRoot, JLD2, PlutoUI, LaTeXStrings, Measures

# ╔═╡ b99a2501-caa7-4eb7-9946-4f7b313cd480
include("../rt_sax_experiment_analysis.jl");

# ╔═╡ 7b5f1adc-698a-446c-bafa-653553fbd7c5
include(@projectroot("src", "rt_sax_control.jl"))

# ╔═╡ ac9de1c5-2381-44f2-ace9-d59f794b6489
session_path = @projectroot("src","sessions")

# ╔═╡ b40831e0-145a-47b9-8b57-5e46411746a7
data_path = joinpath(session_path,"processed_data")

# ╔═╡ 713bbe11-6980-44c0-8427-b2eeb93a1bf5
# This is necessary because the jld2 was saved by the REPL
if !isdefined(Main, :Trial)
    @eval Main const Trial = $(Trial)
end

# ╔═╡ 84c5277c-6042-48bd-a194-ac6571a2c8f9
subjects = list_subjects(session_path,"rt_sax_experiment_sept_2025.csv")

# ╔═╡ a5ada751-9492-4ef6-bf60-bdaad01a8e71
@load joinpath(data_path,"all_model_trials_wamplitudes_onoff.jld2") model_trials

# ╔═╡ 24de0e30-e346-466a-a9d4-c13d0bbef898
@load joinpath(data_path,"all_real_trials_wamplitudes_onoff.jld2") real_trials

# ╔═╡ f8eadaae-3560-4d2d-8d73-ea2fedb7406a
begin
	# correction (temporal)
	M83 = filter_trials(real_trials, subject_ids = ["83"], blocks = [2],task=:Overtone)[1]
	M83.success = false
end	

# ╔═╡ e42e8cba-d8ab-4045-a699-286543764a3f
begin
	vv = 0:15000
	nn = force_newton_from_adc(vv)
	nn2 = newton_from_adc(vv)
	plot(vv,nn,xlabel="ADC value",ylabel="F (N)",label="Paper with correction")
	plot!(vv,nn2,label="Our Data")
	plot!([0,:15000],[0,0],c=:black,ls=:dash)
end	

# ╔═╡ f2f9562c-a535-44bb-b04a-ef95b87acdbe
model_trials[12]

# ╔═╡ ac102238-3ce3-4eba-a780-adbbeeb2e1ce
pf_opts = (
    ms = 10,
    markeralpha = 0.02,
    lw = 1,
    linealpha = 0.6,
    line_darkening = 0.6,
    up_ms = 4,
    down_ms = 4,
    spline_nsamp = 40,
    display = true,
    size = (800, 600),
    axis_scaling = :physical,
    draw_gray = false,
	hl_lw = 3.5,
    hl_ms = 6.0,
    hl_linealpha = 1.0,
	trim_start_ms=100.0, 
	trim_end_ms=100.0
)

# ╔═╡ c0d399e6-59fb-4af0-963f-b28b257317d2
length(model_trials[20].a1)

# ╔═╡ 65721f29-ff55-4420-b786-0b9f8717cae8
md"""
subject: $(@bind subject_id Select(subjects))  
block: $(@bind block Select([1,2]))
task: $(@bind task Select(collect(TASKS)[[4,5,2,3,6]]))
type: $(@bind type Select(["Model","Instrument"]))
"""

# ╔═╡ 1a6f8a6a-8f2b-43aa-a4d3-2d1cb1a2bb91
begin
	RESULTS_XLIMS = (0.0, 2.0)
	RESULTS_YLIMS = (0.0, 8.0)
	RESULTS_TICKFONT = 18
	RESULTS_TITLEFONT = 24
	RESULTS_FIGURE_TITLEFONT = 24
	RESULTS_FIGURE_TITLEVSPAN = 0.08
	RESULTS_LABELFONT = 22
	RESULTS_PAIR_SIZE = (1200, 500)
	RESULTS_COMPOSITE_SIZE = (1200, 1500)

	_task_display_label(task) = task == :Overtone ? "Multiphonic" : String(task)
	_condition_display_label(condition) = condition == "Instrument" ? "Real-Instrument" : String(condition)
	function _task_panel_title(task)
		if task == :NonlegatoAsc
			return "Non-legato ascending"
		elseif task == :NonlegatoDesc
			return "Non-legato descending"
		elseif task == :LegatoAsc
			return "Legato ascending"
		elseif task == :LegatoDesc
			return "Legato descending"
		elseif task == :Overtone
			return "Multiphonic"
		else
			return String(task)
		end
	end

	function _result_pf_plot(trial_list, task, condition; show_xlabel = true, show_ylabel = true, title_text = nothing, kwargs...)
		p = plot_task_pf_mode2(
			trial_list,
			task,
			subjects,
			condition;
			pf_opts...,
			highlight = (subject_id, block),
			tickfontsize = RESULTS_TICKFONT,
			titlefontsize = RESULTS_TITLEFONT,
			labelfontsize = RESULTS_LABELFONT,
			xlims = RESULTS_XLIMS,
			ylims = RESULTS_YLIMS,
			kwargs...,
		)
		plot!(
			p;
			title = isnothing(title_text) ? _task_panel_title(task) : title_text,
			xlabel = show_xlabel ? "Pressure (kPa)" : "",
			ylabel = show_ylabel ? "Force (N)" : "",
		)
		return p
	end

	function _blank_result_panel()
		plot(
			[NaN], [NaN];
			legend = false,
			framestyle = :none,
			grid = false,
			xticks = ([], []),
			yticks = ([], []),
			xlims = RESULTS_XLIMS,
			ylims = RESULTS_YLIMS,
		)
	end
end

# ╔═╡ 3f423dd0-cbad-449c-a650-3a014dd14549
trials = (type == "Model" ? model_trials : real_trials);

# ╔═╡ f6d2eca2-41b0-47d8-93e2-5330bd879a32
begin
	_result_pf_plot(trials, task, type; margin = 3mm)
end

# ╔═╡ fe8e41f9-5900-4de5-8805-60c3e76b9402
begin 
	p1 = _result_pf_plot(model_trials, :NonlegatoAsc, "Model"; show_xlabel = true, show_ylabel = true)
	p2 = _result_pf_plot(model_trials, :NonlegatoDesc, "Model"; show_xlabel = true, show_ylabel = false)
	plot(p1,p2,margin=6mm,size=RESULTS_PAIR_SIZE)
end	

# ╔═╡ c11dabfa-0fc1-4a7c-be35-3260b1648ea0
begin 
	p3 = _result_pf_plot(real_trials, :NonlegatoAsc, "Instrument"; show_xlabel = true, show_ylabel = true)
	p4 = _result_pf_plot(real_trials, :NonlegatoDesc, "Instrument"; show_xlabel = true, show_ylabel = false)
	plot(p3,p4,margin=6mm,size=RESULTS_PAIR_SIZE)
end	

# ╔═╡ 3028e1dc-649f-4722-bdc6-755adfdf3b3b
begin 
	p5 = _result_pf_plot(model_trials, :LegatoAsc, "Model"; show_xlabel = true, show_ylabel = true)
	p6 = _result_pf_plot(model_trials, :LegatoDesc, "Model"; show_xlabel = true, show_ylabel = false)
	plot(p5,p6,margin=6mm,size=RESULTS_PAIR_SIZE)
end	

# ╔═╡ fe3a1b74-b65a-4d26-a892-074e105218e1
begin 
	p7 = _result_pf_plot(real_trials, :LegatoAsc, "Instrument"; show_xlabel = true, show_ylabel = true)
	p8 = _result_pf_plot(real_trials, :LegatoDesc, "Instrument"; show_xlabel = true, show_ylabel = false)
	plot(p7,p8,margin=6mm,size=RESULTS_PAIR_SIZE)
end	

# ╔═╡ 8d0a09aa-a3ab-4dac-9dd8-7715f25c4d37
begin 
	p9 = _result_pf_plot(model_trials, :Overtone, "Model"; show_xlabel = true, show_ylabel = true, title_text = "Model")
	p10 = _result_pf_plot(real_trials, :Overtone, "Instrument"; show_xlabel = true, show_ylabel = false, title_text = "Real-Instrument")
	pmulti = plot(p9,p10,margin=8mm,size=RESULTS_PAIR_SIZE)
	pmulti
end	

# ╔═╡ 459b76cb-c5cc-45a4-b64a-b361551127e6
real_trials

# ╔═╡ d0c604c5-1000-492b-a9d3-1f5ec287e0ad
begin
	pmodel = plot(
		_result_pf_plot(model_trials, :NonlegatoAsc, "Model"; show_xlabel = false, show_ylabel = true),
		_result_pf_plot(model_trials, :NonlegatoDesc, "Model"; show_xlabel = false, show_ylabel = false),
		_result_pf_plot(model_trials, :LegatoAsc, "Model"; show_xlabel = false, show_ylabel = true),
		_result_pf_plot(model_trials, :LegatoDesc, "Model"; show_xlabel = false, show_ylabel = false),
		_result_pf_plot(model_trials, :Overtone, "Model"; show_xlabel = true, show_ylabel = true),
		_blank_result_panel(),
		layout=(3,2),margin=6mm,size=RESULTS_COMPOSITE_SIZE,
	)
	savefig(pmodel, "Figure3.png")
	savefig(pmodel, "Figure3.svg")
	pmodel
end	

# ╔═╡ e9689cbd-4c22-4c4c-81ec-e28b5ea239cc
begin
	preal = plot(
		_result_pf_plot(real_trials, :NonlegatoAsc, "Instrument"; show_xlabel = false, show_ylabel = true),
		_result_pf_plot(real_trials, :NonlegatoDesc, "Instrument"; show_xlabel = false, show_ylabel = false),
		_result_pf_plot(real_trials, :LegatoAsc, "Instrument"; show_xlabel = false, show_ylabel = true),
		_result_pf_plot(real_trials, :LegatoDesc, "Instrument"; show_xlabel = false, show_ylabel = false),
		_result_pf_plot(real_trials, :Overtone, "Instrument"; show_xlabel = true, show_ylabel = true, title_text = "Multiphonic"),
		_blank_result_panel(),
		layout=(3,2),margin=6mm,size=RESULTS_COMPOSITE_SIZE,
	)
	savefig(preal, "Figure4.png")
	savefig(preal, "Figure4.svg")
	preal
end	

# ╔═╡ 0e8db6a4-4c2e-4f15-90c5-da1f5bba9984
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
# ╠═5d954762-35ab-11f1-2308-851b8fbd36ac
# ╠═c5f76341-be5a-40e6-bc4e-ab2eff0ac7f5
# ╠═b99a2501-caa7-4eb7-9946-4f7b313cd480
# ╠═7b5f1adc-698a-446c-bafa-653553fbd7c5
# ╠═ac9de1c5-2381-44f2-ace9-d59f794b6489
# ╠═b40831e0-145a-47b9-8b57-5e46411746a7
# ╠═713bbe11-6980-44c0-8427-b2eeb93a1bf5
# ╠═84c5277c-6042-48bd-a194-ac6571a2c8f9
# ╠═a5ada751-9492-4ef6-bf60-bdaad01a8e71
# ╠═24de0e30-e346-466a-a9d4-c13d0bbef898
# ╠═f8eadaae-3560-4d2d-8d73-ea2fedb7406a
# ╠═e42e8cba-d8ab-4045-a699-286543764a3f
# ╠═f2f9562c-a535-44bb-b04a-ef95b87acdbe
# ╠═ac102238-3ce3-4eba-a780-adbbeeb2e1ce
# ╠═1a6f8a6a-8f2b-43aa-a4d3-2d1cb1a2bb91
# ╠═c0d399e6-59fb-4af0-963f-b28b257317d2
# ╟─65721f29-ff55-4420-b786-0b9f8717cae8
# ╠═3f423dd0-cbad-449c-a650-3a014dd14549
# ╠═f6d2eca2-41b0-47d8-93e2-5330bd879a32
# ╠═fe8e41f9-5900-4de5-8805-60c3e76b9402
# ╠═c11dabfa-0fc1-4a7c-be35-3260b1648ea0
# ╠═3028e1dc-649f-4722-bdc6-755adfdf3b3b
# ╠═fe3a1b74-b65a-4d26-a892-074e105218e1
# ╠═8d0a09aa-a3ab-4dac-9dd8-7715f25c4d37
# ╠═459b76cb-c5cc-45a4-b64a-b361551127e6
# ╠═d0c604c5-1000-492b-a9d3-1f5ec287e0ad
# ╠═e9689cbd-4c22-4c4c-81ec-e28b5ea239cc
# ╠═0e8db6a4-4c2e-4f15-90c5-da1f5bba9984
