using Plots, DifferentialEquations, RealTimeAudioDiffEq, LibSerialPort 
using Glob, JLD2, ProjectRoot, OrderedCollections, Colors, Dierckx
using Statistics, StatsBase, Printf, WAV, FFTW
using DSP
using Measures
import Base: isless

include("./rt_sax_control.jl")

# in VSCode expand/collapse regions with Ctrl+K followed by Ctrl+L
# to collapse all regions: Ctrl+K followed by Ctrl+0
# to expand all regions: Ctrl+K followed by Ctrl+J

#region API CONTRACT

if !isdefined(@__MODULE__, :RT_SAX_ANALYSIS_PUBLIC_API)
    const RT_SAX_ANALYSIS_PUBLIC_API = (
        :list_subjects,
        :trial_audio_block_index,
        :assign_audiofile_from_metadata!,
        :assign_audiofiles_from_metadata!,
        :find_general_log,
        :lastmatch,
        :newton_from_adc,
        :force_newton_from_adc,
        :pressure_from_adc,
        :parse_block_params,
        :parse_block_tasklist,
        :load_trial_data,
        :discover_blocks,
        :build_trials_for_block,
        :load_trial_choices_csv,
        :select_chosen_trials_from_choices,
        :load_subject_trials_model,
        :load_subject_trials_real,
        :filter_trials,
        :all_trials,
        :all_trials_wamplitudes_onoff,
        :get_mode_amps,
        :fill_mode_amplitudes_from_trial_audio!,
        :simulate_mode_amplitudes!,
        :onoff_regions,
        :load_overtone_onoff_table,
        :assign_overtone_onoff_from_table!,
        :fill_onoff_from_onsets!,
        :postprocess_trial!,
        :postprocess_trials!,
        :postprocess_trials_by_type!,
        :plot_task_pf_mode2,
        :plot_task_wregions,
        :plot_gamma_zeta_modes!,
        :simulate_modes_at_point,
        :sweep_modes_grid,
        :plot_sweep_mode_regions,
        :plot_sweep_mode_regions_pattern,
        :plot_trials_values,
        :subject_overview,
        :print_subject_overview,
        :make_default_overrides,
        :apply_override_choices!,
        :trajectories,
        :compute_sol,
        :play_task,
        :assign_onoff_by_beeps!,
        :assign_onoff!,
        :detect_onoff_from_signal,
        :detect_tones,
        :evaluate_results_kde,
    )
end

if !isdefined(@__MODULE__, :RT_SAX_ANALYSIS_INTERNAL_HELPERS)
    const RT_SAX_ANALYSIS_INTERNAL_HELPERS = (
        :_mapvals,
        :_lininterp_vec,
        :_align_two_series,
        :_param_map_from_config,
        :_resolve_param_map,
        :_darker_color,
        :_longest_onset_region,
        :_has_mode_regions,
        :_mode_time_mask,
        :_default_thr2_for_task,
        :_safe_parse_float,
        :_safe_parse_bool,
        :_resolve_default_overtone_csv,
        :_trial_has_mode_amplitudes,
        :_compress_curve,
        :_arclength_parameter,
        :_fit_parametric_spline_pf,
        :_pf_interval_from_timewindow,
        :_note_color_by_order,
        :_bool_runs,
        :_plot_pf_segment!,
        :_mode_amplitude_series,
        :_estimate_fixed_point,
        :_trial_onoff_start_time,
    )
end

"""
Return the stable public API symbols for the analysis layer.
"""
analysis_public_api_symbols() = RT_SAX_ANALYSIS_PUBLIC_API

"""
Return internal helper symbols (implementation details, not API contract).
"""
analysis_internal_helper_symbols() = RT_SAX_ANALYSIS_INTERNAL_HELPERS

"""
Return a NamedTuple with currently bound public API callables.
Useful for introspection in Pluto and scripts.
"""
function analysis_public_api()
    pairs = Pair{Symbol,Any}[]
    for name in RT_SAX_ANALYSIS_PUBLIC_API
        isdefined(@__MODULE__, name) || continue
        push!(pairs, name => getfield(@__MODULE__, name))
    end
    return (; pairs...)
end

#endregion


#region TYPES: Trial, BlockRef

if !isdefined(@__MODULE__, :TASKS)
    const TASKS = (:Practice, :LegatoAsc, :LegatoDesc, :NonlegatoAsc, :NonlegatoDesc, :Overtone)
end

if !isdefined(@__MODULE__, :DEFAULT_EXPECTED_F0_HZ)
    const DEFAULT_EXPECTED_F0_HZ = 185.0
end

if !isdefined(@__MODULE__, :DEFAULT_NONLEGATO_THR2)
    const DEFAULT_NONLEGATO_THR2 = 0.8
end

if !isdefined(@__MODULE__, :DEFAULT_LEGATO_THR2)
    const DEFAULT_LEGATO_THR2 = 0.6
end

if !isdefined(@__MODULE__, :DEFAULT_REAL_BLOCK_ORDER)
    const DEFAULT_REAL_BLOCK_ORDER = Dict(
        "22" => 1, "13" => 1, "49" => 2, "80" => 2, "27" => 1, "70" => 1,
        "33" => 1, "90" => 2, "50" => 2, "83" => 2, "34" => 1, "88" => 1,
        "64" => 1, "37" => 2
    )
end

if !isdefined(@__MODULE__, :DEFAULT_MISSING_AUDIO_BLOCK)
    const DEFAULT_MISSING_AUDIO_BLOCK = Dict("27" => 4, "49" => 1, "83" => 4)
end

# Container for one experiment trial (metadata, sensors, and derived mode features).

if !isdefined(@__MODULE__, :Trial)
    mutable struct Trial
        subject_id::String
        type::Symbol          # :Model or :Real
        fingering::String     # "D4", "Dx4", etc.
        block::Int            # block index (1..N)
        task::Symbol          # :Overtone, :LegatoAsc, etc.
        take::Int             # repetition index within the block for that task (1..N)
        order::Int            # absolute order within block (1..total trials)
        success::Bool         # whether the trial was successful (True/False)
        datafile::String      # data file (relative path)
        audiofile::String     # continuous audio file of the trial
        ts::Float64           # time scaling for the NLDSax model (to match the note frequency)
        u0::Vector{Float64}   # from block log
        p0::Vector{Float64}   # from block log
        param_map::Vector{Tuple{Int64,Int64,Float64, Float64}} # from block log
        t1::Vector{Float64}; v1::Vector{Float64} # pressure sensor values recorded at times t1
        t2::Vector{Float64}; v2::Vector{Float64} # force sensor values recorded at times t2
        t::Vector{Float64}; a1::Vector{Float64}; a2::Vector{Float64}; a3::Vector{Float64} # time t and amplitudes of modes 1 to 3 (a1-a3)after segmentation
        t0::Vector{Float64}; u3::Vector{Float64}; u5::Vector{Float64}; u7::Vector{Float64} # time u0 and amplitudes of modes 1 to 3  (variables u3-u7)of the model   
        onoff::Vector{Tuple{Float64,Float64,Float64}} # cached stable note regions [(start, stop, mode), ...]
        duration::Float64 # trial duration in milliseconds
    end
end

#
# Sort Trials lexicographically by task name.
# 

isless(a::Trial, b::Trial) = isless(String(a.task), String(b.task))

#
# Reference to a discovered block (= one log) for a fingering.
#

if !isdefined(@__MODULE__, :BlockRef)
    struct BlockRef
        fingering::String
        block_idx::Int
        logfile::String
        audiofile::String
    end
end


#endregion

#region UTILITIES: ARRAYS & STRINGS

# If not already defined elsewhere:
function _mapvals(x, p::Tuple{<:Real,<:Real,<:Real,<:Real})
    a,b,c,d = p
    if x < a
        return c
    elseif x > b
        return d
    else
        return (x - a) * (d - c) / (b - a) + c
    end
end

"""
Return deduplicated, numerically sorted subject IDs from the experiment log.
Used by Pluto notebooks and scripts as the standard subject-discovery entry point.
"""
function list_subjects(session_path::String, logs_file::String)
    subjects = String[]
    logs_path = joinpath(session_path, logs_file)
    for line in eachline(logs_path)
        elms = split(line,",")
        if tryparse(Int64,elms[1]) !== nothing
            push!(subjects, strip(elms[1]))
        end
    end
    unique!(subjects)
    sort!(subjects; by = s -> something(tryparse(Int, s), typemax(Int)))
    return subjects
end

"""
Map trial metadata `(subject_id, type, block)` to global recording block index (1..4).

This reproduces the `get_block(...)` logic from the historical Assign_audiofile notebook.
"""
function trial_audio_block_index(subject_id::String, trial_type::Symbol, trial_block::Int;
                                 real_block_order::Dict{String,Int}=DEFAULT_REAL_BLOCK_ORDER)
    haskey(real_block_order, subject_id) ||
        error("Missing real-block order mapping for subject $subject_id")

    tmp_b = Bool(real_block_order[subject_id] - 1) ⊻ (trial_type == :Model)
    return Int(tmp_b) + 1 + Int(trial_block == 2) * 2
end

"""
Assign `trial.audiofile` using self-descriptive naming inferred from trial metadata.

Default generated pattern:
- `S<subject>_<Type>_B<block>_<Task>.wav`

Legacy pattern (Assign_audiofile notebook):
- `S<subject>_click_0<block>_segment_<order>.wav`

Special cases mirror notebook behavior:
- subject `97` is skipped by default
- subjects in `missing_audio_block` shift segment index by `-1` on that block
- resulting `order <= 0` is marked as `"missing"`
"""
function assign_audiofile_from_metadata!(trial::Trial;
                                         overwrite::Bool=false,
                                         naming_style::Symbol=:self_descriptive,
                                         skip_subject_ids::Vector{String}=["97"],
                                         missing_audio_block::Dict{String,Int}=DEFAULT_MISSING_AUDIO_BLOCK,
                                         real_block_order::Dict{String,Int}=DEFAULT_REAL_BLOCK_ORDER)
    audio_now = strip(trial.audiofile)
    if !overwrite && !isempty(audio_now) && lowercase(audio_now) != "missing"
        return trial
    end

    trial.subject_id in skip_subject_ids && return trial

    if naming_style == :self_descriptive
        type_name = string(trial.type)
        task_name = string(trial.task)
        trial.audiofile = "S$(trial.subject_id)_$(type_name)_B$(trial.block)_$(task_name).wav"
    elseif naming_style == :click_segment
        block = trial_audio_block_index(trial.subject_id, trial.type, trial.block;
                                        real_block_order=real_block_order)
        order = trial.order

        if haskey(missing_audio_block, trial.subject_id) && block == missing_audio_block[trial.subject_id]
            order -= 1
        end

        if order <= 0
            trial.audiofile = "missing"
        else
            trial.audiofile = "S$(trial.subject_id)_click_0$(block)_segment_$(order).wav"
        end
    else
        error("Unsupported naming_style: $naming_style. Use :self_descriptive or :click_segment")
    end

    return trial
end

"""
Vectorized wrapper for `assign_audiofile_from_metadata!`.
"""
function assign_audiofiles_from_metadata!(trials::Vector{Trial}; kwargs...)
    for tr in trials
        assign_audiofile_from_metadata!(tr; kwargs...)
    end
    return trials
end

# Return substring between the first '[' and its matching ']' scanning forward from from_idx
function extract_bracket_block(s::String, from_idx::Int)
    ib = findnext('[', s, from_idx)
    ib === nothing && return ""
    # Scan forward to find the matching ']'
    depth = 0
    for i in ib:length(s)
        c = s[i]
        if c == '['
            depth += 1
        elseif c == ']'
            depth -= 1
            if depth == 0
                return s[ib+1:i-1]  # contents without the surrounding brackets
            end
        end
    end
    return ""  # not found
end

"Find the first occurrence of `label` and return index just after it; 0 if missing."
# implement after_label_index behavior for this analysis pipeline.
function after_label_index(s::String, label::String)
    p = findfirst(label, s)
    p === nothing && return 0
    return p.stop
end

"Parse a comma-separated list of Float64 (strip spaces first)."
parse_float_list(sub::String) = parse.(Float64, split(replace(sub, ' '=>""), ','))

#
# Linear interpolation of (t_src,y_src) at times t_eval. Returns interpolated values as a Vector{Float64}.
#
function _lininterp_vec(t_src::AbstractVector{<:Real},
                       y_src::AbstractVector{<:Real},
                       t_eval::AbstractVector{<:Real})
    @assert length(t_src) == length(y_src) "_lininterp_vec: length mismatch"
    out = similar(collect(t_eval), Float64)
    for i in eachindex(t_eval)
        t = t_eval[i]
        j = searchsortedlast(t_src, t)
        if j <= 0
            out[i] = y_src[1]
        elseif j >= length(t_src)
            out[i] = y_src[end]
        else
            τ = (t - t_src[j]) / (t_src[j+1] - t_src[j])
            out[i] = (1-τ)*y_src[j] + τ*y_src[j+1]
        end
    end
    return out
end

#
# Align two time series (t1,y1) and (t2,y2) to a common time vector.
# Arguments:
# - t1, y1: time and values of first series
# - t2, y2: time and values of second series
# Keyword arguments:
#     - mode::Symbol = :auto
#     - :auto → if t1≈t2 (len equal and max |Δ|≤tol), use avg times; else :union
#     - :t1/:t2 → evaluate both series on t1 / t2 respectively
#     - :union  → evaluate both on sort(unique(vcat(t1,t2)))
#     - tol_ms::Real = 0.5   # tolerance in milliseconds for :auto mode
# Returns: (teval, y1e, y2e)
#  
function _align_two_series(t1::Vector{<:Real}, y1::Vector{<:Real},
                          t2::Vector{<:Real}, y2::Vector{<:Real};
                          mode::Symbol=:auto, tol_ms::Real=0.5)
    @assert length(t1) == length(y1) "_align_two_series: (t1,y1) length mismatch"
    @assert length(t2) == length(y2) "_align_two_series: (t2,y2) length mismatch"

    if mode === :auto
        if !isempty(t1) && length(t1) == length(t2) && maximum(abs.(t1 .- t2)) ≤ tol_ms
            teval = 0.5 .* (t1 .+ t2)
            y1e = _lininterp_vec(t1, y1, teval)
            y2e = _lininterp_vec(t2, y2, teval)
            return teval, y1e, y2e
        else
            mode = :union
        end
    end

    if mode === :t1
        teval = t1
        y1e = copy(y1)
        y2e = _lininterp_vec(t2, y2, teval)
        return teval, y1e, y2e
    elseif mode === :t2
        teval = t2
        y1e = _lininterp_vec(t1, y1, teval)
        y2e = copy(y2)
        return teval, y1e, y2e
    else
        # :union
        teval = sort!(unique(vcat(t1, t2)))
        y1e = _lininterp_vec(t1, y1, teval)
        y2e = _lininterp_vec(t2, y2, teval)
        return teval, y1e, y2e
    end
end

#endregion

#region UTILITIES: FILES

#
# Find the general log inside logs_dir whose name starts with "rt_sax_experiment"
#
function find_general_log(logs_dir::String)
    files = glob("rt_sax_experiment*", logs_dir)
    isempty(files) && error("No general log in $logs_dir (expected a file starting with 'rt_sax_experiment').")
    sort(files)
    return files[end]  # take the newest if there are several
end

#
# Return the lexicographically last file that matches a glob pattern in `path`; '' if none.
#
function lastmatch(globpat::AbstractString, path::AbstractString)
    files = glob(globpat, path)
    isempty(files) && return ""
    sort(files)
    return files[end]
end

# ----------- Project root helpers (from ProjectRoot.jl) -----------
to_project_relative(p::AbstractString; root::AbstractString) = begin
    isempty(p) && return ""
    q = replace(p, '\\' => '/'); r = replace(root, '\\' => '/')
    # already relative?
    if !(startswith(q, "/") || occursin(":/", q)); return q; end
    # exact prefix
    if startswith(q, r*"/"); return q[length(r)+2:end]; end
    # anchor by project folder name
    pname = split(r, '/')[end]; anchor = "/"*pname*"/"
    a = findfirst(anchor, q); a !== nothing ? q[a.stop+1:end] : split(q,'/')[end]
end

project_abs(rel::AbstractString; root::AbstractString) = begin
    isempty(rel) && return ""
    q = replace(rel, '\\' => '/')
    (startswith(q,"/") || occursin(":/",q)) ? normpath(q) : normpath(joinpath(root, q))
end

#endregion

#region UTILITIES: CONVERSIONS

"""
Convert force-sensor ADC values to Newtons using calibration parameters.
Supports scalar and array inputs through method overloading.
"""
function newton_from_adc(adc::Real; bits=15, Vref=5.0, Rm=1000.0, Vadc=6.144)
    max_adc = Int(2^bits - 1)
	
    Vin = (adc * Vadc) ./ max_adc
    
    # from FSR 402 datasheet
    Rfsr = Rm * (Vref .- Vin) ./ Vin
    
    y = 1 ./ Rfsr # conductancia
    
    # curva de ajuste (our data)
    a = 1.569e-6
	x = y ./ a # gramos
    newtons = x .* 9.8e-3

    return newtons
end

# implement newton_from_adc behavior for this analysis pipeline.
function newton_from_adc(adc::AbstractArray; kwargs...)
    return newton_from_adc.(adc; kwargs...)
end

"""
Convert ADC values to force in Newtons with the physical full-scale model.
Used in notebooks when plotting force in physical units.
"""
function force_newton_from_adc(adc::Real; Vfs_adc=6.144, Vplus=5.0, Rm=1000.0)
    # ADS1115 single-ended: rango útil 0..32767
    adc_clamped = clamp(Float64(adc), 0.0, 32767.0)

    # Voltaje medido por el ADC
    V = adc_clamped * (Vfs_adc / 32768.0)

    # Protección de bordes
    if V <= 0.0
        return 0.0
    elseif V >= Vplus
        return Inf   # fuera del rango físico del divisor (o saturación)
    end

    # Conductividad del FSR [S]
    G = V / (Rm * (Vplus - V))

    # Datos de calibración adoptados para Interlink FSR-402, RM = 1k
    G300 = 3.750e-4
    G500 = 5.497e-4
    a = (G500 - G300) / (500.0 - 300.0)   # 8.735e-7 S/g

    # Tramo lineal alto
    if G >= G300
        m_g = 300.0 + (G - G300) / a
        return m_g * 9.80665e-3
    end

    # Tramo bajo: G(u) = c2*u^2 - c3*u^3, con u = m/300 en [0,1]
    c2 = 8.6295e-4
    c3 = 4.8795e-4

    # Inversión numérica robusta por bisección
    lo, hi = 0.0, 1.0
    for _ in 1:60
        u = (lo + hi) / 2
        Gu = c2*u^2 - c3*u^3
        if Gu < G
            lo = u
        else
            hi = u
        end
    end

    u = (lo + hi) / 2
    m_g = 300.0 * u
    return m_g * 9.80665e-3
end

# Versión vectorial
force_newton_from_adc(adc::AbstractArray; kwargs...) = force_newton_from_adc.(adc; kwargs...)

# Converts ADC readings to pressure in kPa.
# pressure_from_adc(adc; bits=14, Vref=5.0, Rg=220.0, Voffset=0.0)
# Calibration model for the pressure sensor.
# Parameters:
#  - adc: ADC reading (scalar or vector)
#  - bits: ADC resolution (default 15 -> max = 32768)
#  - Vref: ADC reference voltage in V (default 5.0)
#  - Rg: gain resistor of the AD620 in ohms (default 220.0)
#  - Voffset: amplifier output offset at zero pressure in V
# Returns: vector of pressures in kpa (negative values truncated to 0)

function pressure_from_adc(adc; bits=15, Vref=5.0, Rg=220.0, Voffset=0.0, sens=0.012)
    max_adc = float(2^bits - 1)
    adc_f = float.(adc)                       # vectorized
    Vout = adc_f ./ max_adc .* Vref           # 
    G = 1.0 + 49400.0 / float(Rg)            # gain
    sens_tot = sens * G                      # V per pressure unit (total)
    pressure_units = (Vout .- Voffset) ./ sens_tot  # vectorized
    pressure_units = max.(pressure_units, 0.0)       # avoid negative pressures
    kpa = pressure_units .* 6.89476
    return kpa
end

# implement _param_map_from_config behavior for this analysis pipeline.
function _param_map_from_config(configfile::AbstractString="./rt_sax_configuration.toml")
    cfg = TOML.parsefile(configfile)
    haskey(cfg, "param_map") || error("Missing 'param_map' in config file: $configfile")
    raw = cfg["param_map"]
    length(raw) >= 2 || error("'param_map' must contain at least 2 mappings in: $configfile")
    return [
        (float(raw[1][1]), float(raw[1][2]), float(raw[1][3]), float(raw[1][4])),
        (float(raw[2][1]), float(raw[2][2]), float(raw[2][3]), float(raw[2][4]))
    ]
end

# implement _resolve_param_map behavior for this analysis pipeline.
function _resolve_param_map(tr;
                            param_map_override=nothing,
                            param_map_config::AbstractString="./rt_sax_configuration.toml")
    if param_map_override !== nothing
        return param_map_override
    end
    if hasproperty(tr, :param_map)
        pm = getproperty(tr, :param_map)
        if !isnothing(pm) && length(pm) >= 2
            return pm
        end
    end
    return _param_map_from_config(param_map_config)
end


#endregion

#region UTILITIES: GRAPHICS

# Helpers for color and spline path drawing

function _darker_color(c, f::Real=0.65)
    col = RGB(parse(Colorant, string(c)))
    return RGB(clamp(red(col) * f, 0, 1), clamp(green(col) * f, 0, 1), clamp(blue(col) * f, 0, 1))
end

#endregion

#region UTILITIES: AUDIO & SIGNAL

function hilbert_envelope(x::AbstractVector{<:Real})
    N = length(x)
    X = fft(x)
    H = zeros(eltype(X), N)
    if iseven(N)
        H[1] = 1
        H[2:(N÷2)] .= 2
        H[N÷2+1] = 1
    else
        H[1] = 1
        H[2:((N+1)÷2)] .= 2
    end
    xa = ifft(X .* H)
    return abs.(xa)
end

# In-place moving average smoothing (window size in samples)
function smooth_movingavg!(y::Vector{Float64}, win::Int)
    win ≤ 1 && return y
    w = ones(Float64, win) ./ win
    pad = win ÷ 2
    ypad = vcat(fill(first(y), pad), y, fill(last(y), pad))
    conv = real(ifft(fft(ypad) .* fft(vcat(w, zeros(length(ypad)-length(w))))))
    copyto!(y, @view conv[pad+1:pad+length(y)])
    return y
end

#
# Detect (on, off) segments from a signal using a smoothed |signal| envelope and a
# single z-threshold above the envelope mean.
#
# Inputs
# - t :: Vector{<:Real}   # time in seconds (monotonic, same length as v)
# - v :: Vector{<:Real}   # signal
#
# Keywords
# - z         :: Real = 0.1      # threshold = mean(env) + z*std(env)
# - smooth_ms :: Real = 10.0     # moving-average window (ms) for the envelope
# - minlen    :: Real = 0.20     # minimum segment length (s)
#
# Returns
# - Vector{Tuple{Float64,Float64}} with (on, off) in seconds.
#
function parse_block_params(all_log::String)
    # u0
    u0 = Float64[]
    iu = after_label_index(all_log, "u0:")
    if iu != 0
        inner = extract_bracket_block(all_log, iu)
        !isempty(inner) && (u0 = parse_float_list(inner))
    end

    # p0
    p0 = Float64[]
    ip = after_label_index(all_log, "p:")
    if ip != 0
        inner = extract_bracket_block(all_log, ip)
        !isempty(inner) && (p0 = parse_float_list(inner))
    end

    # ts
    ts = NaN
    its = after_label_index(all_log, "ts:")
    if its != 0
        line_end = findnext('\n', all_log, its)
        line_end = line_end === nothing ? lastindex(all_log) : line_end - 1
        ts_str = strip(all_log[its+1:line_end])
        try ts = parse(Float64, ts_str) catch; ts = NaN end
    end

    # param_map
    param_map = Tuple{Int64,Int64,Float64, Float64}[]
    ipm = after_label_index(all_log, "param_map")
    if ipm != 0
        inner = extract_bracket_block(all_log, ipm)
        if !isempty(inner)
            inner2 = replace(inner, ' '=>"")
            parts = split(inner2, ['(', ')'])
            tuplestrs = [p for p in parts if !isempty(p) && p != "," && occursin(",", p)]
            for tup in tuplestrs
                vals = split(tup, ',')
                if length(vals) == 4
                    push!(param_map, (parse(Int64, vals[1]),
                                      parse(Int64, vals[2]),
                                      parse(Float64, vals[3]),
                                      parse(Float64, vals[4])))
                end
            end
        end
    end
    return u0, p0, ts, param_map
end

#
# Parse an ordered list of (task::Symbol, datafile::String) from the block log.
# Assumes adjacent lines:
#   Info: ("task", "LegatoAsc")
#   Info: ("datafile", "/abs/path/file.dat")
# Pairs them in order.
#
function parse_block_tasklist(all_log::String)
    lines = split(all_log, '\n')
    pairs = Tuple{Symbol,String}[]
    pending_task = nothing :: Union{Nothing,Symbol}
    for ln in lines
        if occursin("(\"task\"", ln)
            toks = split(ln, '"')
            if length(toks) >= 4
                pending_task = Symbol(toks[4])
            end
        elseif occursin("(\"datafile\"", ln)
            toks = split(ln, '"')
            if length(toks) >= 4 && pending_task !== nothing
                push!(pairs, (pending_task::Symbol, toks[4]))
                pending_task = nothing
            end
        end
    end
    return pairs
end

#endregion

#region DATA LOADING

#
# Load a .dat file and return aligned (milliseconds) time vectors and duration.
# It uses `root` to resolve relative paths.
# and the function read_data from rt_sax_control.jl
#
function load_trial_data(datafile::AbstractString;root::String=@projectroot)
    datafile_abs = project_abs(datafile;root=root)
    t1, t2, v1, v2 = read_data(datafile_abs)            # times in nanoseconds
    t0 = min(t1[1], t2[1])
    t1s = (t1 .- t0) ./ 1e6                         # milliseconds, start at 0
    t2s = (t2 .- t0) ./ 1e6
    dur = max(t1s[end], t2s[end])
    return t1s, t2s, v1, v2, dur
end


# Discover subject blocks in chronological order from the session general log.
function discover_blocks(subject_id::String, typ::Symbol, logs_dir::String;
                         audio_dir::Union{Nothing,String}=nothing,
                         general_log_file::Union{Nothing,String}=nothing)

    genlog = if general_log_file === nothing
        find_general_log(logs_dir)
    else
        isabspath(general_log_file) ? general_log_file : joinpath(logs_dir, general_log_file)
    end
    isfile(genlog) || error("General log not found: $genlog")

    # Parse minimal columns: subject_id,type,fingering,order,logfile,...
    # Keep tuples: (order::Int, typeSym::Symbol, fingering::String, log_basename::String)
    entries = Tuple{Int,Symbol,String,String}[]
    for ln in eachline(genlog)
        s = strip(ln); isempty(s) && continue
        cols = split(s, ','); length(cols) < 5 && continue
        strip(cols[1]) == subject_id || continue

        type_sym = (lowercase(strip(cols[2])) == "real") ? :Real : :Model
        fing     = strip(cols[3])
        ord      = try parse(Int, strip(cols[4])) catch; continue end
        logb     = splitdir(strip(cols[5]))[2]  # basename only

        push!(entries, (ord, type_sym, fing, logb))
    end
    sort!(entries, by = e -> e[1])  # chronological by global order

    # Build BlockRefs only for requested type; WAV by GLOBAL order
    per_fing_counts = Dict{String,Int}()
    refs = BlockRef[]

    for (ord, type_sym, fing, logb) in entries
        # compute audio path for this global order, if audio_dir is provided
        audiof = ""
        if audio_dir !== nothing
            idx = lpad(string(ord), 2, '0')  # "01", "02", ...
            audiof = joinpath(audio_dir, "S$(subject_id)_$(idx).wav")
        end

        # only keep rows matching the requested type
        if type_sym == typ
            # per-fingering block index for this type in chrono order
            bidx = get!(per_fing_counts, fing, 0) + 1
            per_fing_counts[fing] = bidx

            # resolve the logfile (prefer exact basename; fallback to a glob)
            candidate = joinpath(logs_dir, logb)
            logfile = isfile(candidate) ? candidate :
                      lastmatch("S$(subject_id)_$(String(typ))_$(fing)*.log", logs_dir)
            logfile == "" && error("Missing block log for subject=$(subject_id) type=$(typ) fing=$(fing) (order=$(ord)).")

            push!(refs, BlockRef(fing, bidx, logfile, audiof))
        end
    end

    return refs
end


#
# Build trials for one block (BlockRef) and subject/type in `path`.
# Uses `root` to resolve relative paths.
#

function build_trials_for_block(ref::BlockRef, subject_id::String, type::Symbol, root::AbstractString=@projectroot)
    all_log = read(ref.logfile, String)
    u0, p0, ts, param_map = parse_block_params(all_log)
    tasklist = parse_block_tasklist(all_log)

    trials = Trial[]
    takes_by_task = Dict{Symbol,Int}()

    for (ord, (task, datafile)) in enumerate(tasklist)
        take = get!(takes_by_task, task, 0) + 1
        takes_by_task[task] = take
        datafile_rel = to_project_relative(datafile; root=root)
        t1, t2, v1, v2, dur = load_trial_data(datafile_rel; root=root)
        push!(trials, Trial(
            subject_id,
            type,
            ref.fingering,
            ref.block_idx,
            task,
            take,
            ord,
            false,
            datafile_rel,
            ref.audiofile,
            ts,
            u0,
            p0,
            param_map,
            t1,
            v1,
            t2,
            v2,
            Float64[],  # t
            Float64[],  # a1
            Float64[],  # a2
            Float64[],  # a3
            Float64[],  # t0
            Float64[],  # u3
            Float64[],  # u5
            Float64[],  # u7
            Tuple{Float64,Float64,Float64}[],  # onoff
            dur
        ))
    end
    return trials
end

# Parse the CSV with columns: subject, block, task, trial, success
# Example row: 22,1,:LegatoAsc,0,FALSE
function load_trial_choices_csv(csv_path::String)
    # Map: (subject, block, task) => (trial, success)
    choices = Dict{Tuple{String,Int,Symbol},Tuple{Int,Bool}}()

    # small helper: strip spaces and double quotes
    stripq(s) = replace(strip(s), "\"" => "")

    line_no = 0
    for raw in eachline(csv_path)
        line_no += 1
        s = strip(raw)
        isempty(s) && continue

        # if the whole line is quoted, drop the outer quotes
        if startswith(s, "\"") && endswith(s, "\"") && length(s) >= 2
            s = s[2:end-1]
        end

        cols = split(s, ',')
        if length(cols) < 5
            @warn "choices CSV: skipping line $line_no (expected 5 columns, got $(length(cols)))"
            continue
        end
        c = stripq.(cols)

        # subject stays as String (e.g., "22")
        subj = c[1]

        # block and trial must be integers; skip header or malformed rows
        block_i = tryparse(Int, c[2])
        trial_i = tryparse(Int, c[4])
        if block_i === nothing || trial_i === nothing
            # Likely a header row like: subject,block,task,trial,success
            @info "choices CSV: skipping non-data row $line_no: $s"
            continue
        end

        # task: accept ":LegatoAsc" or "LegatoAsc"
        tstr = c[3]
        task_sym = startswith(tstr, ":") ? Symbol(tstr[2:end]) : Symbol(tstr)

        # success: TRUE/FALSE, T/F, 1/0, yes/no (case-insensitive)
        sstr = uppercase(c[5])
        succ = sstr in ("TRUE","T","1","YES")

        choices[(subj, block_i, task_sym)] = (trial_i, succ)
    end

    return choices
end



# Choose one trial per task within *this block* using the CSV choices.
# If a (subject, block, task) row exists, use its (take, success);
# otherwise default to (last take, success=true).
function select_chosen_trials_from_choices(trials::Vector{Trial},
                                           choices::Dict{Tuple{String,Int,Symbol},Tuple{Int,Bool}};
                                           include_practice::Bool=false)
    isempty(trials) && return Trial[]

    subject  = trials[1].subject_id
    blockidx = trials[1].block

    # Collect the max take per task inside this block
    last_take = Dict{Symbol,Int}()
    for tr in trials
        (!include_practice && tr.task == :Practice) && continue
        last_take[tr.task] = max(get(last_take, tr.task, 0), tr.take)
        tr.success = false  # clear all; we will set only the chosen one
    end

    chosen = Trial[]
    for (task, nt) in last_take
        # Lookup (subject, block, task) once
        take_succ = get(choices, (subject, blockidx, task), (0, true))
        wanted_take = take_succ[1] == 0 ? nt : min(take_succ[1], nt)
        mark_success = take_succ[2]

        # Pick the chosen trial (fallback: last occurrence of task)
        sel = findfirst(tr -> tr.task == task && tr.take == wanted_take, trials)
        sel === nothing && (sel = findlast(tr -> tr.task == task, trials))
        trc = trials[sel]
        trc.success = mark_success
        push!(chosen, trc)
    end

    sort!(chosen)  # keep the ordering 
    return chosen
end






# Load selected MODEL trials for all subjects using the choices CSV.
"""
Load, build, and optionally postprocess Model trials for requested subjects.
Core future-pipeline loader used by all_trials variants.
"""
function load_subject_trials_model(subject_list::Vector{String}, path::String,
                                   choices_csv::String;
                                   fs::Real=22.05,
                                   include_practice::Bool=false,
                                   general_log_file::Union{Nothing,String}=nothing,
                                   postprocess::Bool=false,
                                   amplitude_filler::Union{Nothing,Function}=nothing,
                                   simulate_model_amplitudes::Bool=false,
                                   replay_filler::Union{Nothing,Function}=nothing,
                                   postprocess_kwargs...)
    choices = load_trial_choices_csv(choices_csv)

    all = Trial[]
    for subject_id in sort(subject_list; by = s -> something(tryparse(Int, s), typemax(Int)))
        refs = discover_blocks(subject_id, :Model, path; general_log_file=general_log_file)
        for ref in refs
            trials = build_trials_for_block(ref, subject_id, :Model, path)
            chosen = select_chosen_trials_from_choices(trials, choices;
                                                       include_practice=include_practice)
            append!(all, chosen)
        end
    end
    postprocess && postprocess_trials!(all;
                                       amplitude_filler=amplitude_filler,
                                       simulate_model_amplitudes=simulate_model_amplitudes,
                                       replay_filler=replay_filler,
                                       postprocess_kwargs...)
    return all
end

# implement load_subject_trials_model behavior for this analysis pipeline.
function load_subject_trials_model(subject_list::Vector{String}, path::String;
                                   choices_csv::String,
                                   fs::Real=22.05,
                                   include_practice::Bool=false,
                                                                     general_log_file::Union{Nothing,String}=nothing,
                                                                     postprocess::Bool=false,
                                                                     amplitude_filler::Union{Nothing,Function}=nothing,
                                                                     simulate_model_amplitudes::Bool=false,
                                                                     replay_filler::Union{Nothing,Function}=nothing,
                                                                     postprocess_kwargs...)
    return load_subject_trials_model(subject_list, path, choices_csv;
                                     fs=fs,
                                     include_practice=include_practice,
                                                                         general_log_file=general_log_file,
                                                                         postprocess=postprocess,
                                                                         amplitude_filler=amplitude_filler,
                                                                         simulate_model_amplitudes=simulate_model_amplitudes,
                                                                         replay_filler=replay_filler,
                                                                         postprocess_kwargs...)
end

# Load selected REAL trials for all subjects using the choices CSV.
"""
Load, build, and optionally postprocess Real trials for requested subjects.
Core future-pipeline loader used by all_trials variants.
"""
function load_subject_trials_real(subject_list::Vector{String}, path::String,
                                   choices_csv::String;
                                   fs::Real=22.05,
                                   include_practice::Bool=false,
                                   general_log_file::Union{Nothing,String}=nothing,
                                   postprocess::Bool=false,
                                   amplitude_filler::Union{Nothing,Function}=nothing,
                                   postprocess_kwargs...)
    choices = load_trial_choices_csv(choices_csv)

    all = Trial[]
    for subject_id in sort(subject_list; by = s -> something(tryparse(Int, s), typemax(Int)))
        refs = discover_blocks(subject_id, :Real, path; general_log_file=general_log_file)
        for ref in refs
            trials = build_trials_for_block(ref, subject_id, :Real, path)
            chosen = select_chosen_trials_from_choices(trials, choices;
                                                       include_practice=include_practice)
            append!(all, chosen)
        end
    end
    postprocess && postprocess_trials!(all;
                                       amplitude_filler=amplitude_filler,
                                       postprocess_kwargs...)
    return all
end

# implement load_subject_trials_real behavior for this analysis pipeline.
function load_subject_trials_real(subject_list::Vector{String}, path::String;
                                  choices_csv::String,
                                  fs::Real=22.05,
                                  include_practice::Bool=false,
                                                                    general_log_file::Union{Nothing,String}=nothing,
                                                                    postprocess::Bool=false,
                                                                    amplitude_filler::Union{Nothing,Function}=nothing,
                                                                    postprocess_kwargs...)
    return load_subject_trials_real(subject_list, path, choices_csv;
                                    fs=fs,
                                    include_practice=include_practice,
                                                                        general_log_file=general_log_file,
                                                                        postprocess=postprocess,
                                                                        amplitude_filler=amplitude_filler,
                                                                        postprocess_kwargs...)
end


"""
Filter trial vectors by metadata constraints (task, subject, block, success, etc.).
Core selector used by notebooks and plotting routines.
"""
function filter_trials(trs::Vector{Trial};
                       task_idx::Union{Nothing,Int}=nothing,
                       task::Union{Nothing,Symbol}=nothing,
                       success_only::Bool=false,
                       type::Union{Nothing,Symbol}=nothing,
                       subject_ids::Union{Nothing,Vector{String}}=nothing,
                       fingering::Union{Nothing,String}=nothing,
                       blocks::Union{Nothing,Vector{Int}}=nothing,
                       include_practice::Bool=false)
    # allow index addressing
    task_sym = task
    if task_sym === nothing && task_idx !== nothing
        @assert 1 <= task_idx <= length(TASKS) "task_idx out of range"
        task_sym = TASKS[task_idx]
    end

    return [tr for tr in trs if
        (include_practice || tr.task != :Practice) &&
        (task_sym === nothing || tr.task == task_sym) &&
        (!success_only || tr.success) &&
        (type === nothing || tr.type == type) &&
        (subject_ids === nothing || (tr.subject_id in subject_ids)) &&
        (fingering === nothing || tr.fingering == fingering) &&
        (blocks === nothing || (tr.block in blocks))
    ]
end


#endregion

#region POSTPROCESSING PIPELINE



"""
Load and postprocess Model/Real trials for all selected subjects.
Future high-level batch pipeline entry point.
"""
function all_trials(session_dir::String, log_file::String, choices_file::String;
                    type::Symbol=:Both,
                    root::String=@projectroot,
                    skip_subjects::Vector{String}=String[],
                    postprocess::Bool=false,
                    real_amplitude_filler::Union{Nothing,Function}=nothing,
                    model_amplitude_filler::Union{Nothing,Function}=nothing,
                    simulate_model_amplitudes::Bool=false,
                    replay_filler::Union{Nothing,Function}=nothing,
                    postprocess_kwargs...)
    session_path = joinpath(root, "src", session_dir)
    choices_path = joinpath(session_path, choices_file)
    save_path = joinpath(session_path, "processed_data")
    subjects = list_subjects(session_path, log_file)
    filter!(s -> !(s in skip_subjects), subjects)
    if type == :Real
        real_trials = load_subject_trials_real(subjects, session_path, choices_path;
                                               general_log_file=log_file,
                                               postprocess=postprocess,
                                               amplitude_filler=real_amplitude_filler,
                                               postprocess_kwargs...)
        @save joinpath(save_path, "all_real_trials.jld2") real_trials
        return real_trials
    elseif type == :Model
        model_trials = load_subject_trials_model(subjects, session_path, choices_path;
                                                 general_log_file=log_file,
                                                 postprocess=postprocess,
                                                 amplitude_filler=model_amplitude_filler,
                                                 simulate_model_amplitudes=simulate_model_amplitudes,
                                                 replay_filler=replay_filler,
                                                 postprocess_kwargs...)
        @save joinpath(save_path, "all_model_trials.jld2") model_trials
        return model_trials
    elseif type == :Both
        model_trials = load_subject_trials_model(subjects, session_path, choices_path;
                                                 general_log_file=log_file,
                                                 postprocess=postprocess,
                                                 amplitude_filler=model_amplitude_filler,
                                                 simulate_model_amplitudes=simulate_model_amplitudes,
                                                 replay_filler=replay_filler,
                                                 postprocess_kwargs...)
        real_trials = load_subject_trials_real(subjects, session_path, choices_path;
                                               general_log_file=log_file,
                                               postprocess=postprocess,
                                               amplitude_filler=real_amplitude_filler,
                                               postprocess_kwargs...)
        @save joinpath(save_path, "all_trials.jld2") model_trials real_trials
        return model_trials, real_trials
    else
        error("type must be :Real or :Model or :Both")
    end
end

"""
Load selected trials and run the full postprocessing chain by type:
1) keep selected trial metadata,
2) fill `a1/a2/a3` from WAV,
3) optionally simulate model state traces (`u3/u5/u7`) for Model,
4) assign `onoff` using threshold rules and optional overtone CSV overrides.

Expected audio layout by default:
- `joinpath(@projectroot, "chopped", "S<subject_id>", trial.audiofile)`

When overtone CSVs are not provided explicitly, this function looks under
`src/sessions/reviewed_data/` for type-specific files such as:
- `overtone_real_table.csv`
- `overtone_model_table.csv`
- `Real_overtones_onoff.csv` / `Model_overtones_onoff.csv`
"""
function all_trials_wamplitudes_onoff(session_dir::String, log_file::String, choices_file::String;
                                      type::Symbol=:Both,
                                      root::String=@projectroot,
                                      skip_subjects::Vector{String}=String[],
                                      assign_audiofiles::Bool=true,
                                      audio_assignment_overwrite::Bool=false,
                                      audio_skip_subject_ids::Vector{String}=["97"],
                                      expected_f0::Real=DEFAULT_EXPECTED_F0_HZ,
                                      t_offset_ms::Real=940.0,
                                      real_audio_root::Union{Nothing,AbstractString}=nothing,
                                      model_audio_root::Union{Nothing,AbstractString}=nothing,
                                      overtone_real_csv::Union{Nothing,AbstractString}=nothing,
                                      overtone_model_csv::Union{Nothing,AbstractString}=nothing,
                                      simulate_model_states::Bool=false,
                                      nonlegato_thr2::Real=DEFAULT_NONLEGATO_THR2,
                                      legato_thr2::Real=DEFAULT_LEGATO_THR2,
                                      onset_kwargs...)
    session_path = joinpath(root, "src", session_dir)
    choices_path = joinpath(session_path, choices_file)
    save_path = joinpath(session_path, "processed_data")
    subjects = list_subjects(session_path, log_file)
    filter!(s -> !(s in skip_subjects), subjects)

    real_filler = mode_amplitudes_from_trial_audio_wrapper(; expected_f0=expected_f0,
                                                           t_offset_ms=t_offset_ms,
                                                           audio_root=real_audio_root)
    model_filler = mode_amplitudes_from_trial_audio_wrapper(; expected_f0=expected_f0,
                                                            t_offset_ms=t_offset_ms,
                                                            audio_root=model_audio_root)

    real_overtone = isnothing(overtone_real_csv) ? _resolve_default_overtone_csv(session_path, :Real) : String(overtone_real_csv)
    model_overtone = isnothing(overtone_model_csv) ? _resolve_default_overtone_csv(session_path, :Model) : String(overtone_model_csv)

    if type == :Real
        real_trials = load_subject_trials_real(subjects, session_path, choices_path;
                                               general_log_file=log_file,
                                               postprocess=false)
        if assign_audiofiles
            assign_audiofiles_from_metadata!(real_trials;
                                             overwrite=audio_assignment_overwrite,
                                             skip_subject_ids=audio_skip_subject_ids)
        end
        postprocess_trials_by_type!(real_trials;
                                    type=:Real,
                                    amplitude_filler=real_filler,
                                    overtone_onoff_csv=real_overtone,
                                    nonlegato_thr2=nonlegato_thr2,
                                    legato_thr2=legato_thr2,
                                    onset_kwargs...)
        @save joinpath(save_path, "all_real_trials_wamplitudes_onoff.jld2") real_trials
        return real_trials
    elseif type == :Model
        model_trials = load_subject_trials_model(subjects, session_path, choices_path;
                                                 general_log_file=log_file,
                                                 postprocess=false)
        if assign_audiofiles
            assign_audiofiles_from_metadata!(model_trials;
                                             overwrite=audio_assignment_overwrite,
                                             skip_subject_ids=audio_skip_subject_ids)
        end
        postprocess_trials_by_type!(model_trials;
                                    type=:Model,
                                    amplitude_filler=model_filler,
                                    simulate_model_states=simulate_model_states,
                                    overtone_onoff_csv=model_overtone,
                                    nonlegato_thr2=nonlegato_thr2,
                                    legato_thr2=legato_thr2,
                                    onset_kwargs...)
        @save joinpath(save_path, "all_model_trials_wamplitudes_onoff.jld2") model_trials
        return model_trials
    elseif type == :Both
        model_trials = load_subject_trials_model(subjects, session_path, choices_path;
                                                 general_log_file=log_file,
                                                 postprocess=false)
        real_trials = load_subject_trials_real(subjects, session_path, choices_path;
                                               general_log_file=log_file,
                                               postprocess=false)

        if assign_audiofiles
            assign_audiofiles_from_metadata!(model_trials;
                                             overwrite=audio_assignment_overwrite,
                                             skip_subject_ids=audio_skip_subject_ids)
            assign_audiofiles_from_metadata!(real_trials;
                                             overwrite=audio_assignment_overwrite,
                                             skip_subject_ids=audio_skip_subject_ids)
        end

        postprocess_trials_by_type!(model_trials;
                                    type=:Model,
                                    amplitude_filler=model_filler,
                                    simulate_model_states=simulate_model_states,
                                    overtone_onoff_csv=model_overtone,
                                    nonlegato_thr2=nonlegato_thr2,
                                    legato_thr2=legato_thr2,
                                    onset_kwargs...)
        postprocess_trials_by_type!(real_trials;
                                    type=:Real,
                                    amplitude_filler=real_filler,
                                    overtone_onoff_csv=real_overtone,
                                    nonlegato_thr2=nonlegato_thr2,
                                    legato_thr2=legato_thr2,
                                    onset_kwargs...)

        @save joinpath(save_path, "all_trials_wamplitudes_onoff.jld2") model_trials real_trials
        return model_trials, real_trials
    else
        error("type must be :Real or :Model or :Both")
    end
end


@inline now_s() = Base.time_ns() * 1e-9

# implement start_model_from_trial behavior for this analysis pipeline.
function start_model_from_trial(trial; gain::Real=0.2,
                                channel_map::AbstractVector{<:AbstractVector{<:Integer}}=[[3],[5]])
    @assert hasproperty(trial,:u0) && !isempty(trial.u0)
    @assert hasproperty(trial,:p0) && !isempty(trial.p0)
    source = DESource(saxRN!, trial.u0, trial.p0; channel_map=channel_map)
    output_device = get_default_output_device()
    start_DESource(source, output_device; buffer_size=convert(UInt32,1024))
    ts_val = hasproperty(trial,:ts) ? float(trial.ts) : 1.0
    @atomic source.data.control.ts   = ts_val
    @atomic source.data.control.gain = float(gain)
    return source
end

# implement make_replay_events behavior for this analysis pipeline.
function make_replay_events(trial; time_units::Symbol=:ms, use_avg::Bool=false)
    t1 = hasproperty(trial,:t1) ? trial.t1 : Float64[]
    t2 = hasproperty(trial,:t2) ? trial.t2 : Float64[]
    v1 = hasproperty(trial,:v1) ? trial.v1 : Float64[]
    v2 = hasproperty(trial,:v2) ? trial.v2 : Float64[]
    @assert length(t1)==length(v1) && length(t2)==length(v2)
    isempty(t1) && isempty(t2) && return Tuple{Float64,Int,Int}[]
    scale = time_units === :ms ? 1e-3 : time_units === :ns ? 1e-9 : 1.0
    t1s = Float64.(t1) .* scale
    t2s = Float64.(t2) .* scale
    t0  = minimum(vcat(t1s, t2s))
    t1s .-= t0
    t2s .-= t0
    events = Tuple{Float64,Int,Int}[]
    if use_avg && !isempty(t1s) && !isempty(t2s) && length(t1s)==length(t2s)
        te = 0.5 .* (t1s .+ t2s)
        append!(events, ((te[i], 1, Int(v1[i])) for i in eachindex(te)))
        append!(events, ((te[i], 2, Int(v2[i])) for i in eachindex(te)))
    else
        append!(events, ((t1s[i], 1, Int(v1[i])) for i in eachindex(t1s)))
        append!(events, ((t2s[i], 2, Int(v2[i])) for i in eachindex(t2s)))
    end
    sort!(events, by = x -> x[1])
    return events
end

# implement feed_channel_from_events! behavior for this analysis pipeline.
function feed_channel_from_events!(mgr::RTSaxSerialManager,
                                   events::Vector{Tuple{Float64,Int,Int}})
    isempty(events) && return nothing
    chan = mgr.chan
    isopen(chan) || error("feed_channel_from_events!: channel is closed")
    t0 = now_s()
    for (t_rel, id, val) in events
        while true
            d = t_rel - (now_s() - t0)
            d <= 0 && break
            sleep(d <= 0.005 ? d : min(d, 0.02))
        end
        while true
            isopen(chan) || error("feed_channel_from_events!: channel was closed")
            length(chan.data) < chan.sz_max && break
            sleep(1e-4)
            yield()
        end
        put!(chan, (id, val))
        yield()
    end
    return nothing
end

# implement replay_trial! behavior for this analysis pipeline.
function replay_trial!(trial; gain::Real=0.2,
                       time_units::Symbol=:ms, use_avg::Bool=false,
                       channel_map::AbstractVector{<:AbstractVector{<:Integer}}=[[3],[5]],
                       warmup_s::Real = 0.5,
                       mgr_capacity::Int = 65_536)
    @assert hasproperty(trial,:param_map)
    events = make_replay_events(trial; time_units=time_units, use_avg=use_avg)
    source = start_model_from_trial(trial; gain=gain, channel_map=channel_map)
    mgr = RTSaxSerialManager("REPLAY",115200; buf_size=mgr_capacity)
    @atomic mgr.reader_stop = true
    try
        start_param_updater_only!(mgr, source, trial.param_map)
        warmup_s > 0 && sleep(warmup_s)
        feeder_task = Threads.@spawn feed_channel_from_events!(mgr, events)
        wait(feeder_task)
    finally
        stop_update!(mgr)
        stop_model!(source)
    end
    return nothing
end

# implement get_mode_amps behavior for this analysis pipeline.
function get_mode_amps(x::AbstractVector{<:Real}, fs::Real, expected_f0::Real)::Vector{Vector{Float64}}
    fs_hz = Float64(fs)
    f0_hz = Float64(expected_f0)
    fs_hz > 0 || error("Sampling frequency fs must be > 0, got $(fs_hz)")
    f0_hz > 0 || error("expected_f0 must be > 0, got $(f0_hz)")

    x_f = Float64.(x)
    freqs = [f0_hz, 2 * f0_hz, 3 * f0_hz]
    bw = 30.0
    order = 3
    a = zeros(Float64, length(x_f), length(freqs))

    for (n, f) in enumerate(freqs)
        f1 = max(0.1, f - bw)
        f2 = min(fs_hz / 2 - 1, f + bw)
        bp = digitalfilter(Bandpass(f1, f2), Butterworth(order); fs = fs_hz)
        y = filtfilt(bp, x_f)
        a[:, n] = abs.(hilbert(y))
    end

    return collect.(eachcol(a))
end

# implement fill_mode_amplitudes_from_trial_audio! behavior for this analysis pipeline.
function fill_mode_amplitudes_from_trial_audio!(trial;
                                                expected_f0::Real=DEFAULT_EXPECTED_F0_HZ,
                                                t_offset_ms::Real=940.0,
                                                audio_root::Union{Nothing,AbstractString}=nothing)
    audio_name = strip(trial.audiofile)
    if isempty(audio_name) || lowercase(audio_name) == "missing"
        return trial
    end

    # implement ms_to_sample_local behavior for this analysis pipeline.
    function ms_to_sample_local(ms::Real, fs::Real)
        return floor(Int, (ms / 1000) * fs) + 1
    end

    # implement resolve_audio_path behavior for this analysis pipeline.
    function resolve_audio_path(audio_name::AbstractString)
        if isabspath(audio_name)
            return normpath(audio_name)
        end

        root = isnothing(audio_root) ? joinpath(@projectroot, "chopped") : String(audio_root)
        candidates = String[
            joinpath(root, "S$(trial.subject_id)", audio_name),
            joinpath(root, audio_name)
        ]

        for candidate in candidates
            if isfile(candidate)
                return normpath(candidate)
            end
        end

        error("Audio file not found for trial $(trial.subject_id) block $(trial.block) task $(trial.task): $(audio_name)")
    end

    t_ref = !isempty(trial.t1) ? trial.t1 : trial.t2
    isempty(t_ref) && return trial

    audio_path = resolve_audio_path(audio_name)
    y, fs = wavread(audio_path)
    audio_signal = ndims(y) == 1 ? Float64.(y) : Float64.(y[:, 1])
    mode_amps = get_mode_amps(audio_signal, Float64(fs), Float64(expected_f0))

    t_shifted = Float64.(t_ref) .+ float(t_offset_ms)
    sampled = [Float64[] for _ in 1:3]

    for mode in 1:3
        amp_trace = mode_amps[mode]
        n_trace = length(amp_trace)
        for time_ms in t_shifted
            sample_index = clamp(ms_to_sample_local(time_ms, fs), 1, n_trace)
            push!(sampled[mode], amp_trace[sample_index])
        end
    end

    trial.t = collect(Float64.(t_ref))
    trial.a1 = sampled[1]
    trial.a2 = sampled[2]
    trial.a3 = sampled[3]
    return trial
end

# implement mode_amplitudes_from_trial_audio_wrapper behavior for this analysis pipeline.
function mode_amplitudes_from_trial_audio_wrapper(;
                                                  expected_f0::Real=DEFAULT_EXPECTED_F0_HZ,
                                                  t_offset_ms::Real=940.0,
                                                  audio_root::Union{Nothing,AbstractString}=nothing)
    return trial -> fill_mode_amplitudes_from_trial_audio!(trial;
                                                            expected_f0=expected_f0,
                                                            t_offset_ms=t_offset_ms,
                                                            audio_root=audio_root)
end

# implement simulate_mode_amplitudes! behavior for this analysis pipeline.
"""
Simulate model dynamics to generate amplitude envelopes for one trial.
Used in postprocessing when measured amplitudes are unavailable.
"""
function simulate_mode_amplitudes!(trial;
    f = saxRN!, solver = Tsit5(),
    dt::Real = 0.1,
    amp_method::Symbol = :quadrature,
    smooth_ms::Real = 0.0,
    unify_eval::Bool = false, tol_ms::Real = 1.0,
    write_amplitudes::Bool = true)
    t1 = hasproperty(trial, :t1) ? trial.t1 : Float64[]
    t2 = hasproperty(trial, :t2) ? trial.t2 : Float64[]
    v1 = hasproperty(trial, :v1) ? trial.v1 : Float64[]
    v2 = hasproperty(trial, :v2) ? trial.v2 : Float64[]
    if isempty(t1) && isempty(t2)
        trial.t0 = Float64[]
        trial.u3 = Float64[]
        trial.u5 = Float64[]
        trial.u7 = Float64[]
        if write_amplitudes
            trial.t = Float64[]
            trial.a1 = Float64[]
            trial.a2 = Float64[]
            trial.a3 = Float64[]
        end
        return trial
    end
    pm = trial.param_map
    γ = isempty(v1) ? Float64[] : map(x -> _mapvals(x, pm[1]), v1)
    ζ = isempty(v2) ? Float64[] : map(x -> _mapvals(x, pm[2]), v2)
    all_times = vcat(t1, t2)
    tmin, tmax = minimum(all_times), maximum(all_times)
    tgrid = collect(tmin:dt:tmax)
    trial.t = tgrid
    pγ = Ref(1)
    cbγ = isempty(t1) ? nothing : PresetTimeCallback(t1, (integ)->begin integ.p[1] = γ[pγ[]]; pγ[] += 1 end; save_positions=(false,false))
    pζ = Ref(1)
    cbζ = isempty(t2) ? nothing : PresetTimeCallback(t2, (integ)->begin integ.p[2] = ζ[pζ[]]; pζ[] += 1 end; save_positions=(false,false))
    cbset = cbγ === nothing ? (cbζ === nothing ? nothing : cbζ) : (cbζ === nothing ? cbγ : CallbackSet(cbγ, cbζ))
    u0 = copy(trial.u0); p0 = copy(trial.p0)
    !isempty(t1) && (t1[1] == tmin || isapprox(t1[1], tmin; atol=1e-12)) && (p0[1] = γ[1])
    !isempty(t2) && (t2[1] == tmin || isapprox(t2[1], tmin; atol=1e-12)) && (p0[2] = ζ[1])
    prob = ODEProblem(f, u0, (tmin, tmax), p0)
    sol  = solve(prob, solver; callback=cbset, saveat=tgrid)
    S = Array(sol)
    trial.t0 = collect(tgrid)
    trial.u3 = collect(@view S[3, :])
    trial.u5 = collect(@view S[5, :])
    trial.u7 = collect(@view S[7, :])

    x3 = @view S[3, :]; x4 = @view S[4, :]
    x5 = @view S[5, :]; x6 = @view S[6, :]
    x7 = @view S[7, :]; x8 = @view S[8, :]
    env1 = amp_method === :hilbert ? sqrt.(hilbert_envelope(collect(x3)).^2 .+ hilbert_envelope(collect(x4)).^2) : sqrt.(x3.^2 .+ x4.^2)
    env2 = amp_method === :hilbert ? sqrt.(hilbert_envelope(collect(x5)).^2 .+ hilbert_envelope(collect(x6)).^2) : sqrt.(x5.^2 .+ x6.^2)
    env3 = amp_method === :hilbert ? sqrt.(hilbert_envelope(collect(x7)).^2 .+ hilbert_envelope(collect(x8)).^2) : sqrt.(x7.^2 .+ x8.^2)
    if smooth_ms > 0
        win = max(1, round(Int, smooth_ms / dt))
        smooth_movingavg!(env1, win); smooth_movingavg!(env2, win); smooth_movingavg!(env3, win)
    end
    # implement interp_on_grid behavior for this analysis pipeline.
    function interp_on_grid(env::Vector{Float64}, tvec::Vector{Float64})
        out = similar(tvec)
        for i in eachindex(tvec)
            t = tvec[i]; j = searchsortedlast(tgrid, t)
            if j <= 0
                out[i] = env[1]
            elseif j >= length(tgrid)
                out[i] = env[end]
            else
                τ = (t - tgrid[j]) / (tgrid[j+1] - tgrid[j])
                out[i] = (1-τ)*env[j] + τ*env[j+1]
            end
        end
        return out
    end
    if write_amplitudes
        if unify_eval && !isempty(t1) && !isempty(t2) && length(t1) == length(t2) && maximum(abs.(t1 .- t2)) ≤ tol_ms
            teval = 0.5 .* (t1 .+ t2)
            trial.t = teval
            trial.a1 = interp_on_grid(env1, teval)
            trial.a2 = interp_on_grid(env2, teval)
            trial.a3 = interp_on_grid(env3, teval)
        else
            teval = !isempty(t1) ? t1 : t2
            trial.t = teval
            trial.a1 = isempty(teval) ? Float64[] : interp_on_grid(env1, teval)
            trial.a2 = isempty(teval) ? Float64[] : interp_on_grid(env2, teval)
            trial.a3 = isempty(teval) ? Float64[] : interp_on_grid(env3, teval)
        end
    end
    return trial
end



# Onset detection

if !isdefined(@__MODULE__, :OnsetRegion)
    const OnsetRegion = NamedTuple{
        (:onset, :offset, :up, :down, :peak, :peak_amp, :mode),
        Tuple{Int,Int,Int,Int,Int,Float64,Float64}
    }
end


# implement onsets behavior for this analysis pipeline.
"""
Detect onset/offset regions from amplitude traces with configurable thresholds.
Provides segmentation primitives for on/off region assignment.
"""
function onsets(amps; thr1=0.05, thr2=0.5, iterations=1, nbins=200, minpeak=0.0, trim_start_ms::Real=0.0, trim_end_ms::Real=0.0, ts::Union{Nothing,Real}=nothing)
    a = collect(float.(amps))
    n = length(a)
    n == 0 && return OnsetRegion[]

    regions = OnsetRegion[]
    floorval = minimum(a)

    trim_start_n = 0
    trim_end_n = 0
    if (trim_start_ms > 0.0 || trim_end_ms > 0.0) && ts !== nothing
        ts_f = float(ts)
        ts_f > 0 || error("ts must be positive for trimming")
        trim_start_n = trim_start_ms > 0.0 ? floor(Int, trim_start_ms / ts_f) : 0
        trim_end_n = trim_end_ms > 0.0 ? floor(Int, trim_end_ms / ts_f) : 0
    end

    for _ in 1:iterations
        peak_amp, peak = findmax(a)

        if !isfinite(peak_amp) || peak_amp <= minpeak
            break
        end
        onset = 1
        for i in peak:-1:1
            if a[i] < peak_amp * thr1
                onset = i
                break
            end
        end
        offset = n
        for i in peak:n
            if a[i] < peak_amp * thr1
                offset = i
                break
            end
        end

        seg = @view a[onset:offset]

        mode_val =
            if maximum(seg) == minimum(seg)
                float(seg[1])
            else
                edges = range(minimum(seg), peak_amp; length=nbins+1)
                h = fit(Histogram, seg, edges)
                centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
                centers[argmax(h.weights)]
            end

        up = onset
        for i in onset:offset
            if a[i] > mode_val * thr2
                up = i
                break
            end
        end

        down = offset
        for i in offset:-1:onset
            if a[i] > mode_val * thr2
                down = i
                break
            end
        end

        # Trim shortens only the effective up/down interval inside the detected region.
        if trim_start_n > 0
            up = max(up, min(offset, onset + trim_start_n))
        end
        if trim_end_n > 0
            down = min(down, max(onset, offset - trim_end_n))
        end

        push!(regions, (
            onset = onset,
            offset = offset,
            up = up,
            down = down,
            peak = peak,
            peak_amp = float(peak_amp),
            mode = float(mode_val)
        ))
        # suppress the detected region before the next iteration
        a[onset:offset] .= floorval
    end
    sort!(regions, by = r -> r.onset)
    return regions
end

# implement _longest_onset_region behavior for this analysis pipeline.
function _longest_onset_region(regions)
    isempty(regions) && return typeof(regions)()
    lens = [r.down - r.up for r in regions]
    return [regions[argmax(lens)]]
end

# implement _has_mode_regions behavior for this analysis pipeline.
function _has_mode_regions(tr::Trial)
    return any(seg -> seg[3] > 0.0, tr.onoff)
end

# implement onoff_regions behavior for this analysis pipeline.
function onoff_regions(tr::Trial; mode::Union{Nothing,Int}=nothing)
    regions = Tuple{Float64,Float64,Float64}[]
    for seg in tr.onoff
        seg_mode = round(Int, seg[3])
        seg_mode <= 0 && continue
        if isnothing(mode) || seg_mode == mode
            push!(regions, seg)
        end
    end
    sort!(regions, by = seg -> (seg[1], seg[2], seg[3]))
    return regions
end

# implement _mode_time_mask behavior for this analysis pipeline.
function _mode_time_mask(t_mode_s::AbstractVector{<:Real}, regions::Vector{Tuple{Float64,Float64,Float64}})
    mask = falses(length(t_mode_s))
    for (t_start, t_stop, _) in regions
        i1 = findfirst(>=(t_start), t_mode_s)
        i2 = findlast(<=(t_stop), t_mode_s)
        if !isnothing(i1) && !isnothing(i2) && i1 <= i2
            mask[i1:i2] .= true
        end
    end
    return mask
end

# implement _default_thr2_for_task behavior for this analysis pipeline.
function _default_thr2_for_task(task)
    return (is_legato(task) || task == :Overtone) ? DEFAULT_LEGATO_THR2 : DEFAULT_NONLEGATO_THR2
end

# implement _safe_parse_float behavior for this analysis pipeline.
function _safe_parse_float(x)
    if x isa Real
        return Float64(x)
    end
    s = lowercase(strip(String(x)))
    isempty(s) && return nothing
    s in ("na", "nan", "none", "missing", "null") && return nothing
    return tryparse(Float64, s)
end

# implement _safe_parse_bool behavior for this analysis pipeline.
function _safe_parse_bool(x)
    s = uppercase(strip(String(x)))
    return s in ("TRUE", "T", "1", "YES", "Y")
end

# implement load_overtone_onoff_table behavior for this analysis pipeline.
"""
Load reviewed overtone segments from a type-specific CSV and index them by `(subject, block)`.
Expected columns: `subject`, `block`, `success`, `stability`, `start`, `end`.
Rows where `success=false` or `start`/`end` are empty are recorded as unsuccessful
and will cause `assign_overtone_onoff_from_table!` to clear `onoff` rather than set it.
"""
function load_overtone_onoff_table(csv_path::AbstractString)
    isfile(csv_path) || error("Overtone on/off CSV not found: $(csv_path)")

    rows = Dict{Tuple{String,Int},Tuple{Float64,Float64,Bool,String}}()
    headers = String[]

    for (i, raw) in enumerate(eachline(csv_path))
        line = strip(raw)
        isempty(line) && continue
        cols = strip.(replace.(split(line, ','), '"' => ""))
        if i == 1
            headers = lowercase.(cols)
            continue
        end

        # pad short rows so missing trailing columns are treated as empty
        padded = length(cols) < length(headers) ?
            vcat(cols, fill("", length(headers) - length(cols))) : cols
        row = Dict{String,String}(headers[j] => padded[j] for j in eachindex(headers))

        subj = get(row, "subject", "")
        isempty(subj) && continue
        block = tryparse(Int, get(row, "block", ""))
        block === nothing && continue

        success = _safe_parse_bool(get(row, "success", "false"))
        stability = get(row, "stability", "")
        start_s = _safe_parse_float(get(row, "start", ""))
        stop_s  = _safe_parse_float(get(row, "end", ""))

        # record the row regardless; success=false or empty interval disables onoff
        if !success || isnothing(start_s) || isnothing(stop_s)
            rows[(subj, block)] = (0.0, 0.0, false, stability)
        else
            rows[(subj, block)] = (start_s, stop_s, true, stability)
        end
    end

    return rows
end

# implement assign_overtone_onoff_from_table! behavior for this analysis pipeline.
"""
Assign overtone `onoff` from reviewed CSV intervals when a matching row exists.
Table is keyed by `(subject, block)` with values `(start, end, success, stability)`.
If the row exists but `success=false` or the interval is empty, `onoff` is cleared.
Returns `true` when the row was found (and assignment applied or cleared), `false` if no row.
"""
function assign_overtone_onoff_from_table!(trial::Trial,
                                           table::Dict{Tuple{String,Int},Tuple{Float64,Float64,Bool,String}};
                                           mode::Real=2.0,
                                           force::Bool=true)
    trial.task == :Overtone || return false
    if !force && _has_mode_regions(trial)
        return false
    end

    row = get(table, (trial.subject_id, trial.block), nothing)
    row === nothing && return false

    start_s, stop_s, success, _stability = row
    if success && isfinite(start_s) && isfinite(stop_s) && stop_s > start_s
        trial.onoff = [(start_s, stop_s, Float64(mode))]
    else
        trial.onoff = Tuple{Float64,Float64,Float64}[]
    end
    return true
end

# implement _resolve_default_overtone_csv behavior for this analysis pipeline.
function _resolve_default_overtone_csv(session_path::AbstractString, type::Symbol)
    type_name = lowercase(String(type))
    candidates = String[
        joinpath(session_path, "reviewed_data", "$(type_name)_overtone_onoff.csv"),
        joinpath(session_path, "reviewed_data", "$(String(type))_overtones_onoff.csv"),
        joinpath(session_path, "reviewed_data", "$(uppercasefirst(type_name))_overtones_onoff.csv"),
        joinpath(session_path, "reviewed_data", "$(type_name)_overtones_onoff.csv"),
        joinpath(session_path, "reviewed_data", "overtone_$(type_name)_table.csv")
    ]
    for p in candidates
        if isfile(p)
            return p
        end
    end
    return nothing
end

# implement fill_onoff_from_onsets! behavior for this analysis pipeline.
"""
Infer stable on/off regions from amplitude trajectories and write them in-place.
Used directly in notebooks and the postprocessing pipeline.
"""
function fill_onoff_from_onsets!(trial::Trial;
                                 thr1=0.05,
                                 thr2::Union{Nothing,Real}=nothing,
                                 iterations::Int=2,
                                 trim_start_ms::Real=0.0,
                                 trim_end_ms::Real=0.0,
                                 force::Bool=true)
    if !force && _has_mode_regions(trial)
        return trial
    end

    trial.onoff = Tuple{Float64,Float64,Float64}[]

    n = minimum((length(trial.t), length(trial.a1), length(trial.a2)))
    n < 2 && return trial

    t_mode_s = trial.t[1:n] ./ 1000
    a1 = trial.a1[1:n]
    a2 = trial.a2[1:n]
    ts_ms = (n >= 2 && trial.t[2] > trial.t[1]) ? (trial.t[2] - trial.t[1]) : nothing
    thr2_eff = isnothing(thr2) ? _default_thr2_for_task(trial.task) : Float64(thr2)

    single_region = is_legato(trial.task) || trial.task == :Overtone
    task_iterations = single_region ? 1 : max(iterations, 2)
    regions1 = onsets(a1; thr1=thr1, thr2=thr2_eff,
                      iterations=task_iterations,
                      trim_start_ms=trim_start_ms, trim_end_ms=trim_end_ms, ts=ts_ms)
    regions2 = onsets(a2; thr1=thr1, thr2=thr2_eff,
                      iterations=task_iterations,
                      trim_start_ms=trim_start_ms, trim_end_ms=trim_end_ms, ts=ts_ms)

    if is_legato(trial.task)
        regions1 = _longest_onset_region(regions1)
        regions2 = _longest_onset_region(regions2)
    elseif trial.task == :Overtone
        regions1 = typeof(regions1)()
        regions2 = _longest_onset_region(regions2)
    end

    for r in regions1
        push!(trial.onoff, (t_mode_s[r.up], t_mode_s[r.down], 1.0))
    end
    for r in regions2
        push!(trial.onoff, (t_mode_s[r.up], t_mode_s[r.down], 2.0))
    end

    sort!(trial.onoff, by = seg -> (seg[1], seg[2], seg[3]))
    return trial
end

# implement fill_onoff_from_onsets! behavior for this analysis pipeline.
function fill_onoff_from_onsets!(trials::Vector{Trial}; kwargs...)
    for tr in trials
        fill_onoff_from_onsets!(tr; kwargs...)
    end
    return trials
end

# implement _trial_has_mode_amplitudes behavior for this analysis pipeline.
function _trial_has_mode_amplitudes(trial::Trial)
    n = minimum((length(trial.t), length(trial.a1), length(trial.a2)))
    return n >= 2
end

"""
Apply configured postprocessing operations to one trial in-place.
Coordinates amplitude fill/simulation and on/off assignment.
"""
function postprocess_trial!(trial::Trial;
                            amplitude_filler::Union{Nothing,Function}=nothing,
                            simulate_model_amplitudes::Bool=false,
                            replay_filler::Union{Nothing,Function}=nothing,
                            fill_onoff::Bool=true,
                            force_onoff::Bool=true,
                            onset_kwargs...)
    if amplitude_filler !== nothing
        amplitude_filler(trial)
    elseif simulate_model_amplitudes && trial.type == :Model
        simulate_mode_amplitudes!(trial)
    end

    if replay_filler !== nothing && trial.type == :Model
        replay_filler(trial)
    end

    if fill_onoff && _trial_has_mode_amplitudes(trial)
        fill_onoff_from_onsets!(trial; force=force_onoff, onset_kwargs...)
    end

    return trial
end

# implement postprocess_trials! behavior for this analysis pipeline.
"""
Apply postprocess_trial! to all trials in-place.
Future pipeline endpoint for bulk postprocessing.
"""
function postprocess_trials!(trials::Vector{Trial}; kwargs...)
    for tr in trials
        postprocess_trial!(tr; kwargs...)
    end
    return trials
end

"""
Run the postprocessing sequence in type-aware stages:
1) keep loaded trial metadata,
2) fill `a1/a2/a3` from trial WAV audio,
3) optionally fill model state traces (`u3/u5/u7`) via simulation,
4) assign `onoff` with task-dependent thresholds and optional overtone table override.
"""
function postprocess_trials_by_type!(trials::Vector{Trial};
                                     type::Union{Nothing,Symbol}=nothing,
                                     amplitude_filler::Union{Nothing,Function}=nothing,
                                     simulate_model_states::Bool=false,
                                     overtone_onoff_csv::Union{Nothing,AbstractString}=nothing,
                                     nonlegato_thr2::Real=DEFAULT_NONLEGATO_THR2,
                                     legato_thr2::Real=DEFAULT_LEGATO_THR2,
                                     fill_onoff::Bool=true,
                                     force_onoff::Bool=true,
                                     onset_kwargs...)
    isempty(trials) && return trials

    trial_type = isnothing(type) ? trials[1].type : type

    if amplitude_filler !== nothing
        for tr in trials
            amplitude_filler(tr)
        end
    end

    if simulate_model_states && trial_type == :Model
        for tr in trials
            simulate_mode_amplitudes!(tr; write_amplitudes=false)
        end
    end

    fill_onoff || return trials

    overtone_table = nothing
    if overtone_onoff_csv !== nothing && isfile(String(overtone_onoff_csv))
        overtone_table = load_overtone_onoff_table(String(overtone_onoff_csv))
    end

    for tr in trials
        if tr.task == :Overtone && overtone_table !== nothing
            used_table = assign_overtone_onoff_from_table!(tr, overtone_table; force=force_onoff)
            used_table && continue
        end

        _trial_has_mode_amplitudes(tr) || continue
        thr2_task = is_legato(tr.task) ? Float64(legato_thr2) : Float64(nonlegato_thr2)
        if tr.task == :Overtone
            thr2_task = Float64(legato_thr2)
        end

        if haskey(onset_kwargs, :thr2)
            fill_onoff_from_onsets!(tr; force=force_onoff, onset_kwargs...)
        else
            fill_onoff_from_onsets!(tr; thr2=thr2_task, force=force_onoff, onset_kwargs...)
        end
    end

    return trials
end


#endregion

#region PLOTTING RESULTS


function _compress_curve(x::AbstractVector, y::AbstractVector; tol::Real=1e-10)
    n = length(x)
    keep = trues(n)
    for i in 2:n
        if hypot(x[i] - x[i-1], y[i] - y[i-1]) <= tol
            keep[i] = false
        end
    end
    return collect(float.(x[keep])), collect(float.(y[keep]))
end

# implement _arclength_parameter behavior for this analysis pipeline.
function _arclength_parameter(x::AbstractVector, y::AbstractVector)
    ds = hypot.(diff(x), diff(y))
    u = zeros(Float64, length(x))
    u[2:end] .= cumsum(ds)
    L = u[end]
    L > 0 || error("curve has zero total arc length")
    u ./= L
    return u
end

# implement _fit_parametric_spline_pf behavior for this analysis pipeline.
function _fit_parametric_spline_pf(
    pressure::AbstractVector,
    force::AbstractVector;
    k::Int = 3,
    nsamp::Int = 80,
    tol::Real = 1e-10,
    normalize_axes::Bool = true
)
    x_raw = collect(float.(pressure))
    y_raw = collect(float.(force))

    if normalize_axes
        x0 = minimum(x_raw)
        y0 = minimum(y_raw)
        xscale = max(maximum(x_raw) - x0, eps(Float64))
        yscale = max(maximum(y_raw) - y0, eps(Float64))
        x_in = (x_raw .- x0) ./ xscale
        y_in = (y_raw .- y0) ./ yscale
    else
        x0 = 0.0
        y0 = 0.0
        xscale = 1.0
        yscale = 1.0
        x_in = x_raw
        y_in = y_raw
    end

    x, y = _compress_curve(x_in, y_in; tol=tol)
    n = length(x)
    
    # Degenerate case: not enough distinct points after compression → fallback to linear interpolation
    if n < 2
        # Return uniformly interpolated polyline
        uf_nsamp = collect(range(0.0, 1.0, length=nsamp))
        x_interp = x_in[1] .+ (x_in[end] - x_in[1]) .* uf_nsamp
        y_interp = y_in[1] .+ (y_in[end] - y_in[1]) .* uf_nsamp
        pressure_f = x_interp .* xscale .+ x0
        force_f    = y_interp .* yscale .+ y0
        return pressure_f, force_f
    end

    kk = min(k, n - 1, 5)
    u  = _arclength_parameter(x, y)
    uf = collect(range(0.0, 1.0, length=nsamp))

    X = Matrix{Float64}(undef, 2, n)
    X[1, :] .= x
    X[2, :] .= y

    spl = ParametricSpline(u, X; k=kk, s=0.0)
    XYf = evaluate(spl, uf)

    pressure_f = vec(XYf[1, :]) .* xscale .+ x0
    force_f    = vec(XYf[2, :]) .* yscale .+ y0

    return pressure_f, force_f
end

# implement _pf_interval_from_timewindow behavior for this analysis pipeline.
function _pf_interval_from_timewindow(tr, t_start::Real, t_stop::Real;
                                     axis_scaling::Symbol=:raw,
                                     pressure_kwargs=NamedTuple(),
                                     force_kwargs=NamedTuple(),
                                     param_map_override=nothing,
                                     param_map_config::AbstractString="./rt_sax_configuration.toml")
    if axis_scaling === :raw
        pressure = tr.v1
        force = tr.v2
    elseif axis_scaling === :physical
        pressure = pressure_from_adc(tr.v1; pressure_kwargs...)
        force = newton_from_adc(tr.v2; force_kwargs...)
    elseif axis_scaling === :parameters
        pm = _resolve_param_map(tr;
                                param_map_override=param_map_override,
                                param_map_config=param_map_config)
        pressure = map(x -> _mapvals(x, pm[1]), tr.v1)
        force = map(x -> _mapvals(x, pm[2]), tr.v2)
    else
        error("axis_scaling must be one of :raw, :physical, :parameters")
    end

    tpf      = tr.t1 ./ 1000                # assume t1 ≈ t2

    i1 = findfirst(>=(t_start), tpf)
    i2 = findlast(<=(t_stop), tpf)

    if isnothing(i1) || isnothing(i2) || i1 > i2
        return nothing
    end

    return (
        t = tpf[i1:i2],
        pressure = pressure[i1:i2],
        force = force[i1:i2]
    )
end

# Task logic helpers

is_legato(task::Symbol) = occursin("Legato", String(task))
is_desc(task::Symbol)   = occursin("Desc", String(task))

# implement active_mask behavior for this analysis pipeline.
function active_mask(n::Int, regions)
    mask = falses(n)
    for r in regions
        i = clamp(r.onset, 1, n)
        j = clamp(r.offset, 1, n)
        i <= j && (mask[i:j] .= true)
    end
    return mask
end

# implement _note_color_by_order behavior for this analysis pipeline.
function _note_color_by_order(task::Symbol, k::Int)
    if task == :Overtone
        return k == 1 ? :magenta : :blue
    end
    if is_desc(task)
        return k == 1 ? :blue : :red
    else
        return k == 1 ? :red : :blue
    end
end

# implement _bool_runs behavior for this analysis pipeline.
function _bool_runs(mask::AbstractVector{Bool}, i::Int, j::Int)
    i > j && return Tuple{Int,Int,Bool}[]
    runs = Tuple{Int,Int,Bool}[]
    s = i
    current = mask[i]
    for k in (i+1):j
        if mask[k] != current
            push!(runs, (s, k-1, current))
            s = k
            current = mask[k]
        end
    end
    push!(runs, (s, j, current))
    return runs
end

# Segment drawer in Pressure–Force space

function _plot_pf_segment!(
    p, pressure, force, i::Int, j::Int;
    color=:gray70,
    lw=1.5,
    linealpha=0.8,
    ms=2.5,
    markeralpha=0.10,
    add_arrow=false,
    markers=true,
    draw_markers=true,
    draw_curve=true,
    line_darkening=0.65,
    up_idx::Union{Nothing,Int}=nothing,
    down_idx::Union{Nothing,Int}=nothing,
    up_ms=5.5,
    up_msw=1.2,
    down_ms=6.0,
    down_msw=1.2,
    use_spline::Bool=true,
    spline_normalize_axes::Bool=true,
    spline_k::Int=3,
    spline_nsamp::Int=80,
    spline_tol::Real=1e-10
)
    i > j && return p
    idx = i:j
    line_color = _darker_color(color, line_darkening)

    if draw_markers && markers
        scatter!(p, pressure[idx], force[idx]; color=color, ms=ms, msw=0, markeralpha=markeralpha, label="", colorbar_entry=false)
    end

    if draw_curve && length(idx) >= 2
        if use_spline
            pressure_f, force_f = _fit_parametric_spline_pf(pressure[idx], force[idx]; k=spline_k, nsamp=spline_nsamp, tol=spline_tol, normalize_axes=spline_normalize_axes)

            plot!(p, pressure_f, force_f; color=line_color, lw=lw, linealpha=linealpha, arrow=add_arrow, label="", colorbar_entry=false)
        else
            plot!(p, pressure[idx], force[idx]; color=line_color, lw=lw, linealpha=linealpha, arrow=add_arrow, label="", colorbar_entry=false)
        end
    end

    if draw_markers && !isnothing(up_idx) && i <= up_idx <= j
        scatter!(p, [pressure[up_idx]], [force[up_idx]]; c=:white, m=:circle, ms=up_ms, msw=up_msw, msc=line_color, alpha=0.95, label="", colorbar_entry=false)
    end

    if draw_markers && !isnothing(down_idx) && i <= down_idx <= j
        scatter!(p, [pressure[down_idx]], [force[down_idx]]; c=:white, m=:utriangle, ms=down_ms, msw=down_msw, msc=line_color, alpha=0.95, label="", colorbar_entry=false)
    end

    return p
end

# One mode-2 region in PF space

function _plot_pf_segment!(
    p, pressure, force, r2, mode1_active, task, k;
    pass::Symbol = :both,   # :markers or :curves
    draw_gray::Bool=true,
    gray_color=:gray70,
    lw=1.5,
    linealpha=0.8,
    ms=2.5,
    markeralpha=0.10,
    add_arrow=false,
    markers=true,
    line_darkening=0.65,
    up_ms=5.5,
    up_msw=1.2,
    down_ms=6.0,
    down_msw=1.2,
    use_spline::Bool=true,
    spline_normalize_axes::Bool=true,
    spline_k::Int=3,
    spline_nsamp::Int=80,
    spline_tol::Real=1e-10
)
    onset  = r2.onset
    offset = r2.offset
    up     = r2.up
    down   = r2.down

    if pass == :curves && draw_gray
        _plot_pf_segment!(p, pressure, force, onset, offset; color=gray_color, lw=lw, linealpha=linealpha, ms=ms, markeralpha=markeralpha, add_arrow=false, markers=false, draw_markers=false, draw_curve=true, line_darkening=line_darkening, up_idx=nothing, down_idx=nothing, use_spline=use_spline, spline_normalize_axes=spline_normalize_axes, spline_k=spline_k, spline_nsamp=spline_nsamp, spline_tol=spline_tol)
    end

    if up <= down
        if is_legato(task)
            runs = _bool_runs(mode1_active, up, down)
            for (ri, (i1, i2, m1on)) in enumerate(runs)
                c = m1on ? :red : :blue
                # extend end by 1 to share boundary point with next run (connects curves)
                i2_ext = (pass == :curves && ri < length(runs)) ? i2 + 1 : i2
                _plot_pf_segment!(p, pressure, force, i1, i2_ext; color=c, lw=lw, linealpha=linealpha, ms=ms, markeralpha=markeralpha, add_arrow=(pass == :curves ? add_arrow : false), markers=(pass == :endpoints ? false : markers), draw_markers=(pass == :markers || pass == :endpoints), draw_curve=(pass == :curves), line_darkening=line_darkening, up_idx=(pass == :endpoints && i1 == up ? up : nothing), down_idx=(pass == :endpoints && i2 == down ? down : nothing), up_ms=up_ms, up_msw=up_msw, down_ms=down_ms, down_msw=down_msw, use_spline=use_spline, spline_normalize_axes=spline_normalize_axes, spline_k=spline_k, spline_nsamp=spline_nsamp, spline_tol=spline_tol)
            end
        else
            c = _note_color_by_order(task, k)
            _plot_pf_segment!(p, pressure, force, up, down; color=c, lw=lw, linealpha=linealpha, ms=ms, markeralpha=markeralpha, add_arrow=(pass == :curves ? add_arrow : false), markers=(pass == :endpoints ? false : markers), draw_markers=(pass == :markers || pass == :endpoints), draw_curve=(pass == :curves), line_darkening=line_darkening, up_idx=(pass == :endpoints ? up : nothing), down_idx=(pass == :endpoints ? down : nothing), up_ms=up_ms, up_msw=up_msw, down_ms=down_ms, down_msw=down_msw, use_spline=use_spline, spline_normalize_axes=spline_normalize_axes, spline_k=spline_k, spline_nsamp=spline_nsamp, spline_tol=spline_tol)
        end
    end

    return p
end

# Plot in PF Space with mode-2 regions highlighted
# mandatory arguments: trial_list, task, subject_list, condition
# where trial_list is the full list of trials to filter from, 
# and the others specify the filtering criteria. condition

"""
Plot pressure-force trajectories for one task with mode-2 highlighting.
Primary output figure function used by Pluto notebooks.
"""
function plot_task_pf_mode2(
    trial_list, task, subject_list, condition;
    trim_start_ms::Real=0.0, trim_end_ms::Real=0.0,
    display=false, remove=Int[],
    axis_scaling::Symbol=:raw,
    use_physical_units::Union{Nothing,Bool}=nothing,
    pressure_kwargs=NamedTuple(),
    force_kwargs=NamedTuple(),
    param_map_override=nothing,
    param_map_config::AbstractString="./rt_sax_configuration.toml",
    lw=1.5,
    linealpha=0.6,
    ms=2.0,
    markeralpha=0.10,
    add_arrow=false,
    markers=true,
    draw_gray::Bool=true,
    gray_color=:gray70,
    line_darkening=0.65,
    up_ms=5.5,
    up_msw=1.2,
    down_ms=6.0,
    down_msw=1.2,
    use_spline::Bool=true,
    spline_normalize_axes::Union{Nothing,Bool}=nothing,
    spline_k::Int=3,
    spline_nsamp::Int=80,
    spline_tol::Real=1e-10,
    highlight::Union{Nothing,Tuple{String,Int}}=nothing,
    hl_lw=3.5,
    hl_ms=6.0,
    hl_linealpha=1.0,
    plot_kwargs...
)
    trials_all = filter_trials(trial_list; task = task, subject_ids = subject_list, success_only = false)
    trials = copy(trials_all)
    sort!(trials, by = tr -> (something(tryparse(Int, tr.subject_id), typemax(Int)), tr.block, tr.order))
    isempty(remove) || deleteat!(trials, sort(unique(remove)))
    data = NamedTuple[]

    if use_physical_units !== nothing
        axis_scaling = use_physical_units ? :physical : :raw
    end

    # Per-segment axis normalization is useful for raw/physical units,
    # but it can over-distort gamma-zeta trajectories.
    spline_norm_eff = isnothing(spline_normalize_axes) ? (axis_scaling != :parameters) : spline_normalize_axes
    draw_gray_eff = task == :Overtone ? false : draw_gray

    for tr in trials
        tr.success || continue

        n = minimum((length(tr.t), length(tr.a1), length(tr.a2)))
        n < 2 && continue
        t_mode = tr.t[1:n] ./ 1000
        regions1 = task == :Overtone ? Tuple{Float64,Float64,Float64}[] : onoff_regions(tr; mode=1)
        regions2_all = onoff_regions(tr; mode=2)
        regions2 = if task == :Overtone && !isempty(regions2_all)
            [regions2_all[1]]
        else
            regions2_all
        end
        isempty(regions2) && continue
        mode1_active = task == :Overtone ? falses(length(t_mode)) : _mode_time_mask(t_mode, regions1)
        region_data = NamedTuple[]
        trim_start_s = trim_start_ms / 1000
        trim_end_s = trim_end_ms / 1000
        tmin, tmax = first(t_mode), last(t_mode)

        for (t_start, t_stop, _) in regions2
            t_start_eff = clamp(t_start + trim_start_s, tmin, tmax)
            t_stop_eff = clamp(t_stop - trim_end_s, tmin, tmax)
            t_start_eff < t_stop_eff || continue

            pf = _pf_interval_from_timewindow(tr, t_start_eff, t_stop_eff; axis_scaling=axis_scaling, pressure_kwargs=pressure_kwargs, force_kwargs=force_kwargs, param_map_override=param_map_override, param_map_config=param_map_config)
            pf === nothing && continue
            # map onset/up/down/offset to common interval indices
            nseg = length(pf.t)
            onset_i  = 1
            offset_i = nseg

            t_up   = t_start_eff
            t_down = t_stop_eff

            up_i   = argmin(abs.(pf.t .- t_up))
            down_i = argmin(abs.(pf.t .- t_down))

            # for legato coloring, transfer the mode1 activity mask by time
            mode1_seg = falses(nseg)
            for k in eachindex(pf.t)
                tm = pf.t[k]
                im = argmin(abs.(t_mode .- tm))
                mode1_seg[k] = mode1_active[im]
            end

            push!(region_data, (
                pressure = pf.pressure,
                force = pf.force,
                mode1_active = mode1_seg,
                r2 = (
                    onset = onset_i,
                    offset = offset_i,
                    up = up_i,
                    down = down_i
                )
            ))
        end

        if !isempty(region_data)
            is_hl = !isnothing(highlight) &&
                    tr.subject_id == highlight[1] &&
                    tr.block == highlight[2]
            push!(data, (regions = region_data, highlighted = is_hl))
        end
    end

    xlabel_text, ylabel_text = if axis_scaling === :physical
        ("Pressure (kPa)", "Force (N)")
    elseif axis_scaling === :parameters
        ("γ (gamma)", "ζ (zeta)")
    else
        ("Pressure Sensor (a. u.)", "Force Sensor (a. u.)")
    end

    p = plot(; xlabel=xlabel_text, ylabel=ylabel_text, legend=false, colorbar=false, title=condition * " | " * String(task), plot_kwargs...)

    # pass 1: all scatter blobs (small dots, no up/down endpoint markers)
    for hl_filter in (false, true)
        for d in filter(d -> d.highlighted == hl_filter, data)
            for (k, rr) in enumerate(d.regions)
                _plot_pf_segment!(p, rr.pressure, rr.force, rr.r2, rr.mode1_active, task, k; pass=:markers, draw_gray=draw_gray_eff, gray_color=gray_color, lw=lw, linealpha=linealpha, ms=ms, markeralpha=markeralpha, add_arrow=false, markers=markers, line_darkening=line_darkening, up_ms=up_ms, up_msw=up_msw, down_ms=down_ms, down_msw=down_msw, use_spline=use_spline, spline_normalize_axes=spline_norm_eff, spline_k=spline_k, spline_nsamp=spline_nsamp, spline_tol=spline_tol)
            end
        end
    end

    # pass 2: all curves
    for hl_filter in (false, true)
        for d in filter(d -> d.highlighted == hl_filter, data)
            for (k, rr) in enumerate(d.regions)
                _plot_pf_segment!(p, rr.pressure, rr.force, rr.r2, rr.mode1_active, task, k; pass=:curves, draw_gray=draw_gray_eff, gray_color=gray_color, lw=d.highlighted ? hl_lw : lw, linealpha=d.highlighted ? hl_linealpha : linealpha, ms=ms, markeralpha=markeralpha, add_arrow=add_arrow, markers=false, line_darkening=line_darkening, up_ms=up_ms, up_msw=up_msw, down_ms=down_ms, down_msw=down_msw, use_spline=use_spline, spline_normalize_axes=spline_norm_eff, spline_k=spline_k, spline_nsamp=spline_nsamp, spline_tol=spline_tol)
            end
        end
    end

    # pass 3: up/down endpoint markers on top of everything
    for hl_filter in (false, true)
        for d in filter(d -> d.highlighted == hl_filter, data)
            for (k, rr) in enumerate(d.regions)
                _plot_pf_segment!(p, rr.pressure, rr.force, rr.r2, rr.mode1_active, task, k; pass=:endpoints, draw_gray=false, gray_color=gray_color, lw=lw, linealpha=linealpha, ms=ms, markeralpha=markeralpha, add_arrow=false, markers=markers, line_darkening=line_darkening, up_ms=d.highlighted ? hl_ms : up_ms, up_msw=up_msw, down_ms=d.highlighted ? hl_ms : down_ms, down_msw=down_msw, use_spline=use_spline, spline_normalize_axes=spline_norm_eff, spline_k=spline_k, spline_nsamp=spline_nsamp, spline_tol=spline_tol)
            end
        end
    end

    if display
        return p
    else
        savefig(p, "Figures/" * condition * "_" * String(task) * "_pressure_force_mode2.png")
        return nothing
    end
end

# Panels to show all traces for a task

"""
Plot per-trial amplitude traces with detected regions and optional overlays.
Used by notebooks for segmentation and task-level inspection.
"""
function plot_task_wregions(
    trial_list, task, subject_list, condition;
    trim_start_ms::Real=0.0, trim_end_ms::Real=0.0,
    display=false, remove=Int[], ncols=2,
    overlay_controls=true,
    show_trim_guides::Bool=false,
    controls_alpha=0.8,
    controls_ls=:dot,
    use_absmax=true
)
    trials_all = filter_trials(trial_list; task=task, subject_ids=subject_list, success_only=false)

    trials = copy(trials_all)
    sort!(trials, by = tr -> (something(tryparse(Int, tr.subject_id), typemax(Int)), tr.block, tr.order))
    isempty(remove) || deleteat!(trials, sort(unique(remove)))

    plts = []

    for tr in trials
        t  = tr.t ./ 1000
        a1 = tr.a1
        a2 = tr.a2

        # Guard against empty or incomplete trial data
        if isempty(a1) || isempty(a2)
            continue
        end

        ymax = max(maximum(a1), maximum(a2))
        trim_start_s = trim_start_ms / 1000
        trim_end_s = trim_end_ms / 1000

        p = plot(t, a1; c=:red,  lw=1.5, label="")
        plot!(p, t, a2; c=:blue, lw=1.5, label="")

        if show_trim_guides
            if trim_start_ms > 0.0
                t_start_keep = first(t) + trim_start_ms / 1000
                vline!(p, [t_start_keep]; c=:black, ls=:dash, alpha=0.45, lw=1.2, label="")
            end
            if trim_end_ms > 0.0
                t_end_keep = last(t) - trim_end_ms / 1000
                vline!(p, [t_end_keep]; c=:black, ls=:dash, alpha=0.45, lw=1.2, label="")
            end
        end

        # overlay v1 and v2, rescaled to amplitude range
        if overlay_controls && !isempty(tr.v1) && !isempty(tr.v2)
            t1 = tr.t1 ./ 1000
            t2 = tr.t2 ./ 1000

            s1 = use_absmax ? maximum(abs.(tr.v1)) : maximum(tr.v1)
            s2 = use_absmax ? maximum(abs.(tr.v2)) : maximum(tr.v2)

            if s1 > 0
                v1_plot = tr.v1 .* (ymax / s1)
                plot!(p, t1, v1_plot; c=:magenta, lw=1.2, ls=controls_ls, alpha=controls_alpha, label="")
            end

            if s2 > 0
                v2_plot = tr.v2 .* (ymax / s2)
                plot!(p, t2, v2_plot; c=:green, lw=1.2, ls=controls_ls, alpha=controls_alpha, label="")
            end
        end

        txtcolor = tr.success ? :black : :red

        if tr.success
            tmin, tmax = first(t), last(t)
            regions1 = Tuple{Float64,Float64,Float64}[]
            regions2 = Tuple{Float64,Float64,Float64}[]
            for (ts, te, m) in onoff_regions(tr; mode=1)
                ts_eff = clamp(ts + trim_start_s, tmin, tmax)
                te_eff = clamp(te - trim_end_s, tmin, tmax)
                ts_eff < te_eff || continue
                push!(regions1, (ts_eff, te_eff, m))
            end
            for (ts, te, m) in onoff_regions(tr; mode=2)
                ts_eff = clamp(ts + trim_start_s, tmin, tmax)
                te_eff = clamp(te - trim_end_s, tmin, tmax)
                ts_eff < te_eff || continue
                push!(regions2, (ts_eff, te_eff, m))
            end

            up1 = [argmin(abs.(t .- seg[1])) for seg in regions1]
            down1 = [argmin(abs.(t .- seg[2])) for seg in regions1]
            up2 = [argmin(abs.(t .- seg[1])) for seg in regions2]
            down2 = [argmin(abs.(t .- seg[2])) for seg in regions2]

            scatter!(p, t[up1],   a1[up1];   c=:white, m=:circle, msw=1, msc=:red,  alpha=0.7, label="")
            scatter!(p, t[down1], a1[down1]; c=:white, m=:square, msw=1, msc=:red,  alpha=0.7, label="")
            scatter!(p, t[up2],   a2[up2];   c=:white, m=:circle, msw=1, msc=:blue, alpha=0.7, label="")
            scatter!(p, t[down2], a2[down2]; c=:white, m=:square, msw=1, msc=:blue, alpha=0.7, label="")
        end

        xmin, xmax = first(t), last(t)
        annotate!(p, xmin + 0.05*(xmax - xmin), 0.85*ymax, text(tr.subject_id, txtcolor, 8))

        push!(plts, p)
    end

    nrows = ceil(Int, length(plts) / ncols)
    pp = plot(plts...; layout=(nrows, ncols), size=(1000, 150*nrows), plot_title=condition * " | " * String(task))

    if display
        return pp
    else
        savefig(pp, "Figures/" * condition * "_" * String(task) * "_regions.png")
        return nothing
    end
end

# Plot a trial in ζ-γ space and color points by mode amplitudes.
"""
Overlay one trial in gamma-zeta space and color by modal activity.
Used in Pluto outputs for trajectory visualization.
"""
function plot_gamma_zeta_modes!(plt, trial;
    mode::Symbol = :predominant,
    amp_thresh::Real = 0.01,
    align::Symbol = :auto,
    tol_ms::Real = 0.5,
    colors_pred = (colorant"#D81B60", colorant"#1E88E5", colorant"#BDBDBD"),
    rgb_a3::Union{Nothing,AbstractVector{<:Real}} = nothing,
    rgb_norm::Symbol = :global,
    scatter_ms::Real = 4,
    alpha::Real = 0.9,
    draw_path::Symbol = :none,
    lw::Real = 1.0,
    apply_map::Bool = true,
    param_map_override = nothing,
    show_legend::Bool = false)

    # 1) Map v1→γ and v2→ζ
    @assert hasproperty(trial, :v1) && hasproperty(trial, :v2)
    @assert hasproperty(trial, :t1) && hasproperty(trial, :t2)
    @assert hasproperty(trial, :a1) && hasproperty(trial, :a2)

    pm = if param_map_override !== nothing
        param_map_override
    else
        @assert hasproperty(trial, :param_map) "Trial missing param_map and no override provided"
        trial.param_map
    end

    γ_raw = apply_map ? map(x->_mapvals(x, pm[1]), trial.v1) : trial.v1
    ζ_raw = apply_map ? map(x->_mapvals(x, pm[2]), trial.v2) : trial.v2

    # 2) Align γ, ζ, a1, a2 to a common time vector
    teval, γ, ζ = _align_two_series(trial.t1, γ_raw, trial.t2, ζ_raw; mode=align, tol_ms=tol_ms)
    _,   a1, a2 = _align_two_series(trial.t1, trial.a1, trial.t2, trial.a2; mode=align, tol_ms=tol_ms)

    # Optional path line under scatter
    if plt === nothing
        plt = plot()
    end
    if draw_path != :none
        plot!(plt, γ, ζ, color=:gray80, lw=(draw_path==:thin ? 0.5 : lw), alpha=0.6, label=false)
    end

    # 3) Build per-point colors
    if mode === :predominant
        col1, col2, col_silent = colors_pred
        colors_pts = Vector{Any}(undef, length(teval))
        for i in eachindex(teval)
            m1 = a1[i]; m2 = a2[i]
            maxamp = max(m1, m2)
            if maxamp < amp_thresh
                colors_pts[i] = RGBA(col_silent, alpha)
            else
                colors_pts[i] = (m1 >= m2) ? RGBA(col1, alpha) : RGBA(col2, alpha)
            end
        end
        scatter!(plt, γ, ζ; ms=scatter_ms, ma=alpha, msw=0, color=colors_pts, label=false)

        if show_legend
            scatter!(plt, [NaN], [NaN], color=RGBA(col1, alpha), ms=scatter_ms, label="mode 1 ≥ thresh")
            scatter!(plt, [NaN], [NaN], color=RGBA(col2, alpha), ms=scatter_ms, label="mode 2 ≥ thresh")
            scatter!(plt, [NaN], [NaN], color=RGBA(col_silent, alpha), ms=scatter_ms, label="silent")
        end

    elseif mode === :rgb
        # Prepare channels
        a3 = rgb_a3
        # Normalize channels
        if rgb_norm === :global
            denom = maximum(vcat(a1, a2, (a3 === nothing ? [0.0] : a3)))
            denom = denom > 0 ? denom : 1.0
            R = a1 ./ denom
            G = (a3 === nothing ? zeros(length(a1)) : (a3 ./ denom))
            B = a2 ./ denom
        else
            # :perpoint normalization by local max
            R = similar(a1); B = similar(a2)
            G = (a3 === nothing ? zeros(length(a1)) : similar(a3))
            for i in eachindex(a1)
                m = maximum((a3 === nothing) ? ((a1[i], a2[i])) : ((a1[i], a2[i], a3[i])))
                m = m > 0 ? m : 1.0
                R[i] = a1[i] / m
                B[i] = a2[i] / m
                if a3 !== nothing; G[i] = a3[i] / m; end
            end
        end
        # Apply threshold → desaturate to gray for near-silent
        colors_pts = Vector{Any}(undef, length(teval))
        for i in eachindex(teval)
            if max(a1[i], a2[i], (a3 === nothing ? 0.0 : a3[i])) < amp_thresh
                colors_pts[i] = RGBA(0.7, 0.7, 0.7, alpha)
            else
                g = (a3 === nothing ? 0.0 : clamp(G[i], 0, 1))
                colors_pts[i] = RGBA(clamp(R[i], 0, 1), g, clamp(B[i], 0, 1), alpha)
            end
        end
        scatter!(plt, γ, ζ; ms=scatter_ms, ma=alpha, msw=0, color=colors_pts, label=false)

        if show_legend
            scatter!(plt, [NaN], [NaN], color=RGBA(1, 0, 0, alpha), ms=scatter_ms, label="a1 → R")
            a3 === nothing || scatter!(plt, [NaN], [NaN], color=RGBA(0, 1, 0, alpha), ms=scatter_ms, label="a3 → G")
            scatter!(plt, [NaN], [NaN], color=RGBA(0, 0, 1, alpha), ms=scatter_ms, label="a2 → B")
            scatter!(plt, [NaN], [NaN], color=RGBA(0.7, 0.7, 0.7, alpha), ms=scatter_ms, label="silent (< thresh)")
        end

    else
        error("Unknown mode = $mode. Use :predominant or :rgb.")
    end

    xlabel!(plt, "γ (gamma)")
    ylabel!(plt, "ζ (zeta)")
    show_legend || plot!(plt, legend=false)
end


#endregion

#region SIMULATION


function _mode_amplitude_series(S::AbstractMatrix{<:Real}, mode::Integer;
                               center::Union{Nothing,Tuple{<:Real,<:Real}}=nothing)
    i = 2 * mode + 1
    @assert i + 1 <= size(S, 1) "Requested mode index exceeds state dimension"
    c1, c2 = center === nothing ? (0.0, 0.0) : (float(center[1]), float(center[2]))
    return sqrt.((S[i, :] .- c1).^2 .+ (S[i + 1, :] .- c2).^2)
end

"""
Estimate fixed-point state and residual for a gamma-zeta parameter pair.
Used directly in simulation notebooks and sweep routines.
"""
function _estimate_fixed_point(γ::Real, ζ::Real, model_p::NamedTuple;
                               nmodes::Int=8,
                               pert::Real=0.0)
    γf, ζf = float(γ), float(ζ)
    α = float.(model_p.α[1:nmodes]); ω = float.(model_p.ω[1:nmodes]); C = float.(model_p.C[1:nmodes])
    A = @. (α + ω*ω/α) / (2.0 * C * ζf)
 
    # Hardcoded bracket near known fixed-point magnitudes: p_n in [0, 0.002] (odd states).
    a = 0.0
    b = min(γf, 0.002 * nmodes)
    a < b || error("Empty feasible interval for p: [$a, $b] (γ=$(γ))")

    K = sum(inv.(A))
    f(p) = K * (p - γf + 1.0) * sqrt(max(γf - p, 0.0)) - p
    fa, fb = f(a), f(b)
    fa * fb <= 0 || error("No sign change in bracket [$a, $b] for γ=$(γ), ζ=$(ζ)")

    atol, rtol = 1e-12, 1e-10
    for _ in 1:100
        m = 0.5 * (a + b)
        fm = f(m)
        (abs(fm) <= atol || (b - a) <= max(atol, rtol * max(abs(a), abs(b), 1.0))) && (a = b = m; break)
        if signbit(fa) == signbit(fm)
            a, fa = m, fm
        else
            b, fb = m, fm
        end
    end

    pstar = 0.5 * (a + b)
    drive = (pstar - γf + 1.0) * sqrt(max(γf - pstar, 0.0))
    p0 = max.(drive ./ A, 0.0)
    q0 = @. 0.5 * p0 * ω / α
    x0 = sum(p0) - γf + pert
    u = vcat([x0, 0.0], collect(Iterators.flatten(zip(p0, q0))))

    p = set_parameters(float(γ), float(ζ), model_p, Int64(nmodes))
    dx = similar(u)
    saxRN!(dx, u, p, 0.0)
    residual = sqrt(sum(abs2, dx))
    
    return u, residual
end

# Simulate one (gamma,zeta) point and return:
# - grow_mode{1,2,3}: from fixed point + small perturbation, mode grows above threshold
# - dominant_mode{1,2,3}: for each sustain excitation IC (mode1/mode2/mode3),
#   dominant mode at the end (0 if no mode exceeds threshold)
function simulate_modes_at_point(γ::Real, ζ::Real, model_p::NamedTuple;
                                 nmodes::Int=8,
                                 fixed_perturb::Real=1e-5,
                                 excite_amp::Real=1e-3,
                                 t_growth::Real=500.0,
                                 t_sustain::Real=500.0,
                                 growth_threshold::Real=1e-2,
                                 sustain_threshold::Real=1e-2,
                                 sustain_tail_fraction::Real=0.25,
                                 solver=Tsit5(),
                                 saveat_growth::Real=0.4,
                                 saveat_sustain::Real=0.4,
                                 reltol::Real=1e-6,
                                 abstol::Real=1e-8)
    @assert nmodes >= 3 "simulate_modes_at_point requires nmodes >= 3"
    p = set_parameters(float(γ), float(ζ), model_p, Int64(nmodes))

    ufp, fixed_residual = _estimate_fixed_point(γ, ζ, model_p;
                                                nmodes=nmodes,
                                                pert=0.0)

    tail_fraction = clamp(float(sustain_tail_fraction), 0.0, 1.0)

    # ── Growth IC: fixed point + small perturbation on x[1] ──────────────────
    ug = copy(ufp)
    ug[1] += float(fixed_perturb)
    prob_g = ODEProblem(saxRN!, ug, (0.0, float(t_growth)), p)
    sol_g  = solve(prob_g, solver; saveat=saveat_growth, reltol=reltol, abstol=abstol)
    Sg     = Array(sol_g)

    grow_amps = ntuple(3) do m
        i   = 2 * m + 1
        amp = _mode_amplitude_series(Sg, m; center=(ufp[i], ufp[i + 1]))
        maximum(amp)
    end
    grow = ntuple(m -> Int8(grow_amps[m] >= growth_threshold), 3)

    # ── Sustain ICs: mode-specific excitation ────────────────────────────────
    dominant = ntuple(3) do m
        us     = copy(ufp)
        us[2 * m + 1] += float(excite_amp)
        prob_s = ODEProblem(saxRN!, us, (0.0, float(t_sustain)), p)
        sol_s  = solve(prob_s, solver; saveat=saveat_sustain, reltol=reltol, abstol=abstol)
        Ss     = Array(sol_s)
        amps_tail = ntuple(3) do k
            i   = 2 * k + 1
            amp = _mode_amplitude_series(Ss, k; center=(ufp[i], ufp[i + 1]))
            i0  = max(1, floor(Int, (1.0 - tail_fraction) * length(amp)))
            mean(@view amp[i0:end])
        end
        best_k = argmax(amps_tail)
        amps_tail[best_k] >= sustain_threshold ? Int8(best_k) : Int8(0)
    end

    return (
        grow_mode1    = grow[1],
        grow_mode2    = grow[2],
        grow_mode3    = grow[3],
        dominant_mode1 = dominant[1],
        dominant_mode2 = dominant[2],
        dominant_mode3 = dominant[3],
        fixed_state   = ufp,
        fixed_residual = fixed_residual
    )
end

"""
Sweep gamma-zeta grids and compute growth/dominance mode maps.
Primary simulation-output generator for scripts in src/scripts.
"""
function sweep_modes_grid(model_p::NamedTuple;
                                gamma_range::Tuple{<:Real,<:Real}=(0.3, 0.99),
                                zeta_range::Tuple{<:Real,<:Real}=(0.1, 0.99),
                                ngamma::Int=80,
                                nzeta::Int=80,
                                gamma_values::Union{Nothing,AbstractVector}=nothing,
                                zeta_values::Union{Nothing,AbstractVector}=nothing,
                                parallel::Bool=true,
                                print_progress::Bool=false,
                                progress_every_rows::Int=5,
                                kwargs...)
    γvals = gamma_values === nothing ? collect(range(float(gamma_range[1]), float(gamma_range[2]), length=ngamma)) : collect(float.(gamma_values))
    ζvals = zeta_values === nothing ? collect(range(float(zeta_range[1]), float(zeta_range[2]), length=nzeta)) : collect(float.(zeta_values))

    nγ = length(γvals)
    nζ = length(ζvals)

    grow_mode1 = zeros(Int8, nζ, nγ)
    grow_mode2 = zeros(Int8, nζ, nγ)
    grow_mode3 = zeros(Int8, nζ, nγ)
    dominant_mode1 = zeros(Int8, nζ, nγ)
    dominant_mode2 = zeros(Int8, nζ, nγ)
    dominant_mode3 = zeros(Int8, nζ, nγ)
    fixed_residual = fill(NaN, nζ, nγ)

    t0 = time()
    progress_step = max(1, progress_every_rows)
    progress_lock = ReentrantLock()

    if parallel && Threads.nthreads() > 1
        done_rows = Threads.Atomic{Int}(0)
        Threads.@threads for iz in eachindex(ζvals)
            ζ = ζvals[iz]
            for (iγ, γ) in enumerate(γvals)
                result = simulate_modes_at_point(γ, ζ, model_p; kwargs...)
                grow_mode1[iz, iγ]    = result.grow_mode1
                grow_mode2[iz, iγ]    = result.grow_mode2
                grow_mode3[iz, iγ]    = result.grow_mode3
                dominant_mode1[iz, iγ] = result.dominant_mode1
                dominant_mode2[iz, iγ] = result.dominant_mode2
                dominant_mode3[iz, iγ] = result.dominant_mode3
                fixed_residual[iz, iγ] = result.fixed_residual
            end
            if print_progress
                rows_done = Threads.atomic_add!(done_rows, 1) + 1
                if rows_done % progress_step == 0 || rows_done == nζ
                    elapsed = round(time() - t0; digits=1)
                    pct = round(100 * rows_done / nζ; digits=1)
                    lock(progress_lock) do
                        println("  sweep progress: $(rows_done)/$(nζ) rows ($(pct)%), elapsed $(elapsed)s")
                        flush(stdout)
                    end
                end
            end
        end
    else
        for (iz, ζ) in enumerate(ζvals)
            for (iγ, γ) in enumerate(γvals)
                result = simulate_modes_at_point(γ, ζ, model_p; kwargs...)
                grow_mode1[iz, iγ]    = result.grow_mode1
                grow_mode2[iz, iγ]    = result.grow_mode2
                grow_mode3[iz, iγ]    = result.grow_mode3
                dominant_mode1[iz, iγ] = result.dominant_mode1
                dominant_mode2[iz, iγ] = result.dominant_mode2
                dominant_mode3[iz, iγ] = result.dominant_mode3
                fixed_residual[iz, iγ] = result.fixed_residual
            end
            if print_progress && (iz % progress_step == 0 || iz == nζ)
                elapsed = round(time() - t0; digits=1)
                pct = round(100 * iz / nζ; digits=1)
                println("  sweep progress: $(iz)/$(nζ) rows ($(pct)%), elapsed $(elapsed)s")
                flush(stdout)
            end
        end
    end

    return (
        gamma_values = γvals,
        zeta_values = ζvals,
        grow_mode1 = grow_mode1,
        grow_mode2 = grow_mode2,
        grow_mode3 = grow_mode3,
        dominant_mode1 = dominant_mode1,
        dominant_mode2 = dominant_mode2,
        dominant_mode3 = dominant_mode3,
        fixed_residual = fixed_residual
    )
end

# Plot dominant-mode regions from sweep_modes_grid and overlay growth boundaries.
# Heatmap classes (from dominant_mode1, dominant_mode2):
#   0 -> other / none
#   1 -> (1,0)
#   2 -> (0,2)
#   3 -> (1,2)
#   4 -> (2,2)
#   5 -> (1,1)
function plot_sweep_mode_regions(maps;
                                 title::AbstractString="Dominant Regions with Growth Boundaries",
                                 xlabel::AbstractString="gamma",
                                 ylabel::AbstractString="zeta",
                                 dom10_color=colorant"#FFB6C1",
                                 dom02_color=colorant"#5DADE2",
                                 dom12_color=:purple,
                                 dom22_color=:blue,
                                 dom11_color=:red,
                                 dom21_color=:gray,
                                 dom3_color=:green,
                                 none_color=colorant"#F2F2F2",
                                 grow1_color=:black,
                                 grow2_color=:black,
                                 grow1_style=:solid,
                                 grow2_style=:dash,
                                 grow1_lw::Real=2.8,
                                 grow2_lw::Real=2.4,
                                 grow_alpha::Real=1.0,
                                 plot_kwargs...)
    @assert haskey(maps, :gamma_values) "maps must contain :gamma_values"
    @assert haskey(maps, :zeta_values) "maps must contain :zeta_values"
    @assert haskey(maps, :dominant_mode1) "maps must contain :dominant_mode1"
    @assert haskey(maps, :dominant_mode2) "maps must contain :dominant_mode2"
    @assert haskey(maps, :grow_mode1) "maps must contain :grow_mode1"
    @assert haskey(maps, :grow_mode2) "maps must contain :grow_mode2"

    γvals = collect(float.(maps.gamma_values))
    ζvals = collect(float.(maps.zeta_values))

    d1 = Int8.(maps.dominant_mode1)
    d2 = Int8.(maps.dominant_mode2)
    g1 = Float64.(maps.grow_mode1 .> 0)
    g2 = Float64.(maps.grow_mode2 .> 0)

    @assert size(d1) == (length(ζvals), length(γvals)) "dominant_mode1 size must be (length(zeta_values), length(gamma_values))"
    @assert size(d2) == size(d1) "dominant_mode2 must have same size as dominant_mode1"
    @assert size(g1) == size(d1) "grow_mode1 must have same size as dominant_mode1"
    @assert size(g2) == size(d1) "grow_mode2 must have same size as dominant_mode1"

    dominant_regions = zeros(Int8, size(d1))
    dominant_regions[(d1 .== 1) .& (d2 .== 0)] .= 1
    dominant_regions[(d1 .== 0) .& (d2 .== 2)] .= 2
    dominant_regions[(d1 .== 1) .& (d2 .== 2)] .= 3
    dominant_regions[(d1 .== 2) .& (d2 .== 2)] .= 4
    dominant_regions[(d1 .== 1) .& (d2 .== 1)] .= 5
    dominant_regions[(d1 .== 2) .& (d2 .== 1)] .= 6
    dominant_regions[(d1 .== 3) .| (d2 .== 3)] .= 7
    # Suppress classes 21 (/) and 3 (.) in pattern rendering.
    dominant_regions[(dominant_regions .== 6) .| (dominant_regions .== 7)] .= 0

    cmap = cgrad([none_color, dom10_color, dom02_color, dom12_color, dom22_color, dom11_color, dom21_color, dom3_color], categorical=true)
    cb_ticks = collect(0:7)
    cb_labels = ["00", "10", "02", "12", "22", "11", "21", "3"]

    p = heatmap(γvals, ζvals, dominant_regions; clims=(-0.5, 7.5), c=cmap, xlabel=xlabel, ylabel=ylabel, title=title, colorbar=false, legend=:topright, plot_kwargs...)

    contour!(p, γvals, ζvals, g1; levels=[0.5], c=grow1_color, lw=grow1_lw, alpha=grow_alpha, linestyle=grow1_style, colorbar_entry=false, label="grow_mode1 boundary")

    contour!(p, γvals, ζvals, g2; levels=[0.5], c=grow2_color, lw=grow2_lw, alpha=grow_alpha, linestyle=grow2_style, colorbar_entry=false, label="grow_mode2 boundary")

    p_cb = heatmap([1.0], cb_ticks, reshape(cb_ticks, :, 1); clims=(-0.5, 7.5), c=cmap, colorbar=false, xticks=false, xlabel="", yticks=(cb_ticks, cb_labels), ylabel="dom", ymirror=true, framestyle=:box)

    return plot(p, p_cb; layout=@layout([a{0.93w} b{0.07w}]))
end

"""
Render print-friendly patterned mode-region maps from sweep outputs.
Used by scripts/notebooks as final simulation visualization output.
"""
function plot_sweep_mode_regions_pattern(maps;
                                         title::AbstractString="",
                                         xlabel::AbstractString="γ",
                                         ylabel::AbstractString="ζ",
                                         bg_gray::Real=0.90,
                                         pattern_decimation=:auto,
                                         auto_pattern_decimation::Bool=true,
                                         max_motif_cells::Int=12_000,
                                         motif_stride::Int=1,
                                         pattern_color=:black,
                                         pattern10_color=RGBA(0.30, 0.30, 0.30, 0.95),
                                         pattern02_color=RGBA(0.52, 0.52, 0.52, 0.95),
                                         pattern_dark_alpha::Real=0.92,
                                         pattern_dark_lw::Real=1.6,
                                         pattern_alpha::Real=0.60,
                                         pattern_lw::Real=1.1,
                                         region_sep_color=:gray15,
                                         region_sep_lw::Real=1.2,
                                         region_sep_alpha::Real=1.0,
                                         grow1_color=:black,
                                         grow2_color=:black,
                                         grow1_style=:solid,
                                         grow2_style=:dash,
                                         grow1_lw::Real=2.8,
                                         grow2_lw::Real=2.4,
                                         grow_alpha::Real=1.0,
                                         show_growth_boundaries::Bool=false,
                                         plot_kwargs...)
    @assert haskey(maps, :gamma_values) "maps must contain :gamma_values"
    @assert haskey(maps, :zeta_values) "maps must contain :zeta_values"
    @assert haskey(maps, :dominant_mode1) "maps must contain :dominant_mode1"
    @assert haskey(maps, :dominant_mode2) "maps must contain :dominant_mode2"
    @assert haskey(maps, :grow_mode1) "maps must contain :grow_mode1"
    @assert haskey(maps, :grow_mode2) "maps must contain :grow_mode2"

    γvals = collect(float.(maps.gamma_values))
    ζvals = collect(float.(maps.zeta_values))

    d1 = Int8.(maps.dominant_mode1)
    d2 = Int8.(maps.dominant_mode2)
    g1 = Float64.(maps.grow_mode1 .> 0)
    g2 = Float64.(maps.grow_mode2 .> 0)

    @assert size(d1) == (length(ζvals), length(γvals)) "dominant_mode1 size must be (length(zeta_values), length(gamma_values))"
    @assert size(d2) == size(d1) "dominant_mode2 must have same size as dominant_mode1"
    @assert size(g1) == size(d1) "grow_mode1 must have same size as dominant_mode1"
    @assert size(g2) == size(d1) "grow_mode2 must have same size as dominant_mode1"

    dominant_regions = zeros(Int8, size(d1))
    dominant_regions[(d1 .== 1) .& (d2 .== 0)] .= 1
    dominant_regions[(d1 .== 0) .& (d2 .== 2)] .= 2
    dominant_regions[(d1 .== 1) .& (d2 .== 2)] .= 3
    dominant_regions[(d1 .== 2) .& (d2 .== 2)] .= 4
    dominant_regions[(d1 .== 1) .& (d2 .== 1)] .= 5
    dominant_regions[(d1 .== 2) .& (d2 .== 1)] .= 6
    dominant_regions[(d1 .== 3) .| (d2 .== 3)] .= 7

    # Uniform gray background; class identity is encoded by overlaid motifs.
    bg = fill(bg_gray, size(d1))
    graymap = cgrad([RGB(bg_gray, bg_gray, bg_gray), RGB(bg_gray, bg_gray, bg_gray)])
    p = heatmap(γvals, ζvals, bg; c=graymap, clims=(0.0, 1.0), xlabel=xlabel, ylabel=ylabel, title=title, colorbar=false, legend=:topright, guidefontsize=16, tickfontsize=14, left_margin=5mm, right_margin=5mm, top_margin=5mm, bottom_margin=5mm, plot_kwargs...)

    nζ, nγ = size(dominant_regions)

    # Decimation controls:
    # - pattern_decimation = :auto uses max_motif_cells to choose a stride automatically.
    # - pattern_decimation = :none disables decimation (stride = 1).
    # - pattern_decimation = integer N uses fixed stride N.
    # Backward compatibility:
    # - motif_stride > 1 still forces that stride.
    # - auto_pattern_decimation is only consulted when pattern_decimation is not explicitly set.
    stride = 1
    if motif_stride > 1
        stride = max(1, motif_stride)
    elseif pattern_decimation isa Integer
        stride = max(1, Int(pattern_decimation))
    elseif pattern_decimation == :none
        stride = 1
    elseif pattern_decimation == :auto
        total_cells = nζ * nγ
        if total_cells > max_motif_cells
            stride = max(1, ceil(Int, sqrt(total_cells / max_motif_cells)))
        end
    elseif pattern_decimation === nothing
        # Legacy behavior (for older calls): controlled by auto_pattern_decimation.
        stride = 1
        if auto_pattern_decimation
            total_cells = nζ * nγ
            if total_cells > max_motif_cells
                stride = max(1, ceil(Int, sqrt(total_cells / max_motif_cells)))
            end
        end
    else
        throw(ArgumentError("pattern_decimation must be :auto, :none, an integer >= 1, or nothing"))
    end

    # Cell half-size for motif drawing. When decimated, enlarge motifs to preserve coverage.
    dx = length(γvals) > 1 ? minimum(abs.(diff(γvals))) : 0.02
    dz = length(ζvals) > 1 ? minimum(abs.(diff(ζvals))) : 0.02
    hx = 0.50 * dx * stride
    hz = 0.50 * dz * stride

    # Batch motif segments with NaN separators to keep the number of plot series small.
    x10 = Float64[]; y10 = Float64[]
    x02 = Float64[]; y02 = Float64[]
    x12 = Float64[]; y12 = Float64[]
    x22h = Float64[]; y22h = Float64[]
    x22v = Float64[]; y22v = Float64[]
    x11a = Float64[]; y11a = Float64[]
    x11b = Float64[]; y11b = Float64[]

    for ii in 1:stride:nζ, jj in 1:stride:nγ
        cls = dominant_regions[ii, jj]
        cls == 0 && continue
        x = γvals[jj]
        z = ζvals[ii]
        if cls == 1
            append!(x10, (x - hx, x + hx, NaN)); append!(y10, (z, z, NaN))
        elseif cls == 2
            append!(x02, (x, x, NaN)); append!(y02, (z - hz, z + hz, NaN))
        elseif cls == 3
            append!(x12, (x - hx, x + hx, NaN)); append!(y12, (z + hz, z - hz, NaN))
        elseif cls == 4
            append!(x22h, (x - hx, x + hx, NaN)); append!(y22h, (z, z, NaN))
            append!(x22v, (x, x, NaN)); append!(y22v, (z - hz, z + hz, NaN))
        elseif cls == 5
            append!(x11a, (x - hx, x + hx, NaN)); append!(y11a, (z - hz, z + hz, NaN))
            append!(x11b, (x - hx, x + hx, NaN)); append!(y11b, (z + hz, z - hz, NaN))
        end
    end

    !isempty(x10) && plot!(p, x10, y10; linecolor=pattern10_color, lw=pattern_dark_lw, label="", colorbar_entry=false)
    !isempty(x02) && plot!(p, x02, y02; linecolor=pattern02_color, lw=pattern_dark_lw, label="", colorbar_entry=false)
    !isempty(x12) && plot!(p, x12, y12; linecolor=pattern_color, lw=pattern_dark_lw, alpha=pattern_dark_alpha, label="", colorbar_entry=false)
    !isempty(x22h) && plot!(p, x22h, y22h; linecolor=pattern_color, lw=pattern_lw, alpha=pattern_alpha, label="", colorbar_entry=false)
    !isempty(x22v) && plot!(p, x22v, y22v; linecolor=pattern_color, lw=pattern_lw, alpha=pattern_alpha, label="", colorbar_entry=false)
    !isempty(x11a) && plot!(p, x11a, y11a; linecolor=pattern_color, lw=pattern_lw, alpha=pattern_alpha, label="", colorbar_entry=false)
    !isempty(x11b) && plot!(p, x11b, y11b; linecolor=pattern_color, lw=pattern_lw, alpha=pattern_alpha, label="", colorbar_entry=false)

    # Draw contour boundaries per class for smoother, less blocky separators.
    for cls in 1:5
        mask = Float64.(dominant_regions .== Int8(cls))
        contour!(p, γvals, ζvals, mask; levels=[0.5], c=region_sep_color, lw=region_sep_lw, alpha=region_sep_alpha, linestyle=:solid, colorbar_entry=false, label="")
    end

    if show_growth_boundaries
        contour!(p, γvals, ζvals, g1; levels=[0.5], c=grow1_color, lw=grow1_lw, alpha=grow_alpha, linestyle=grow1_style, colorbar_entry=false, label="grow_mode1 boundary")
        contour!(p, γvals, ζvals, g2; levels=[0.5], c=grow2_color, lw=grow2_lw, alpha=grow_alpha, linestyle=grow2_style, colorbar_entry=false, label="grow_mode2 boundary")
    end

    # Custom legend panel: one boxed square per class with labels on the right.
    legend_labels = ["11", "22", "12", "02", "10", "00"]
    legend_classes = Int8[5, 4, 3, 2, 1, 0]
    p_cb = plot(; xlim=(0.0, 3.2), ylim=(0.5, 6.5), xticks=false, yticks=false,
                legend=false, framestyle=:none,
                background_color=RGB(bg_gray, bg_gray, bg_gray),
                left_margin=4mm, right_margin=10mm, top_margin=4mm, bottom_margin=4mm)

    grid_n = 4
    x_offsets = range(-0.30, 0.30; length=grid_n)
    y_offsets = range(-0.30, 0.30; length=grid_n)
    lhx = 0.09  # half-width of motif stroke in legend mini-cells
    lhz = 0.09  # half-height

    for (row, (lbl, cls)) in enumerate(zip(legend_labels, legend_classes))
        cy = 7.0 - row
        x0, x1 = 0.25, 1.25
        y0, y1 = cy - 0.45, cy + 0.45

        # Individual black square per class (including 00).
        plot!(p_cb, [x0, x1], [y0, y0]; linecolor=:black, lw=1.2, label="", colorbar_entry=false)
        plot!(p_cb, [x0, x1], [y1, y1]; linecolor=:black, lw=1.2, label="", colorbar_entry=false)
        plot!(p_cb, [x0, x0], [y0, y1]; linecolor=:black, lw=1.2, label="", colorbar_entry=false)
        plot!(p_cb, [x1, x1], [y0, y1]; linecolor=:black, lw=1.2, label="", colorbar_entry=false)

        # Pattern samples inside each square.
        if cls != 0
            for dx in x_offsets, dz in y_offsets
                x = 0.75 + dx
                y = cy + dz
                if cls == 1
                    plot!(p_cb, [x - lhx, x + lhx], [y, y]; linecolor=pattern10_color, lw=1.5, label="", colorbar_entry=false)
                elseif cls == 2
                    plot!(p_cb, [x, x], [y - lhz, y + lhz]; linecolor=pattern02_color, lw=1.6, label="", colorbar_entry=false)
                elseif cls == 3
                    plot!(p_cb, [x - lhx, x + lhx], [y + lhz, y - lhz]; linecolor=pattern_color, lw=1.5, alpha=pattern_dark_alpha, label="", colorbar_entry=false)
                elseif cls == 4
                    plot!(p_cb, [x - lhx, x + lhx], [y, y]; linecolor=pattern_color, lw=1.2, alpha=pattern_alpha, label="", colorbar_entry=false)
                    plot!(p_cb, [x, x], [y - lhz, y + lhz]; linecolor=pattern_color, lw=1.2, alpha=pattern_alpha, label="", colorbar_entry=false)
                elseif cls == 5
                    plot!(p_cb, [x - lhx, x + lhx], [y - lhz, y + lhz]; linecolor=pattern_color, lw=1.2, alpha=pattern_alpha, label="", colorbar_entry=false)
                    plot!(p_cb, [x - lhx, x + lhx], [y + lhz, y - lhz]; linecolor=pattern_color, lw=1.2, alpha=pattern_alpha, label="", colorbar_entry=false)
                end
            end
        end

        annotate!(p_cb, 1.85, cy, text(lbl, 16, :black))
    end

    p_blank = plot(; framestyle=:none, background_color=:transparent, foreground_color=:transparent, legend=false)
    p_cb_panel = plot(p_blank, p_cb, p_blank; layout=@layout([_{0.10h}; b{0.80h}; _{0.10h}]))
    return plot(p, p_cb_panel; layout=@layout([a{0.85w} b{0.15w}]))
end


#endregion

#region UNUSED


"""
Placeholder for statistical evaluation of experimental results.

Planned scope:
- KDE-based comparison of mode-1 (red) vs mode-2 (blue) occupied areas in parameter space.
- Statistical comparison between trajectories.
"""
function plot_trials_values(trials::AbstractVector;
                            y::Symbol=:v1, x::Symbol=:time,
                            apply_map::Bool=false,
                            param_map_override=nothing,
                            align::Symbol=:start,
                            normalize::Symbol=:none,
                            lw::Real=1.5, α::Real=0.9, show_legend::Bool=false)

    isempty(trials) && return plot()

    # Helper to fetch param_map for a trial
    get_pm = tr -> begin
        if param_map_override !== nothing
            param_map_override
        elseif hasproperty(tr, :param_map) && !isnothing(getproperty(tr,:param_map))
            getproperty(tr, :param_map)
        else
            nothing
        end
    end

    # Map a single series according to var and param_map
    # returns a Vector{Float64} and a tag of origin (:v1 or :v2) to choose time
    function series_from(tr, var::Symbol)
        if var === :gamma || (apply_map && var === :v1)
            pm = get_pm(tr)
            pm === nothing && error("param_map required to map v1→gamma")
            yv = getproperty(tr, :v1)
            return map(x->_mapvals(x, pm[1]), yv), :v1
        elseif var === :zeta || (apply_map && var === :v2)
            pm = get_pm(tr)
            pm === nothing && error("param_map required to map v2→zeta")
            yv = getproperty(tr, :v2)
            return map(x->_mapvals(x, pm[2]), yv), :v2
        elseif var === :v1
            return getproperty(tr, :v1), :v1
        elseif var === :v2
            return getproperty(tr, :v2), :v2
        else
            error("Unsupported variable symbol for series: $var")
        end
    end

    # Choose a time vector given preferred source (:v1 or :v2) and x selector
    function time_from(tr, xsel::Symbol, pref::Symbol)
        if xsel === :t1 && hasproperty(tr,:t1); return getproperty(tr,:t1) end
        if xsel === :t2 && hasproperty(tr,:t2); return getproperty(tr,:t2) end
        if xsel === :time || xsel === :auto
            if pref === :v1 && hasproperty(tr,:t1); return getproperty(tr,:t1) end
            if pref === :v2 && hasproperty(tr,:t2); return getproperty(tr,:t2) end
            if hasproperty(tr,:t);  return getproperty(tr,:t)  end
            if hasproperty(tr,:ts); return getproperty(tr,:ts) end
        end
        return nothing  # fallback to index later
    end

    # Is x time-like?
    is_time_x = (x === :time || x === :t1 || x === :t2 || x === :auto)

    plt = plot()

    for tr in trials
        # Y series (+ origin to pick t1/t2 when x is time-like)
        yv, origin = series_from(tr, y)

        # X axis: either time-like or a second data series
        if is_time_x
            tv = time_from(tr, x, origin)
            tv === nothing && (tv = collect(0:length(yv)-1))
            if align === :start && !isempty(tv)
                tv = tv .- tv[1]
            end
            # Normalize Y (only)
            yplot = copy(yv)
            if normalize === :zscore
                μ, σ = mean(yplot), std(yplot); σ > 0 && (yplot = (yplot .- μ) ./ σ)
            elseif normalize === :minmax
                m, M = minimum(yplot), maximum(yplot); M > m && (yplot = (yplot .- m) ./ (M - m))
            end
            # Label
            sid = hasproperty(tr,:subject_id) ? tr.subject_id : "?"
            blk = hasproperty(tr,:block) ? tr.block : "?"
            task = hasproperty(tr,:task) ? string(tr.task) : "?"
            take = hasproperty(tr,:take) ? tr.take : "?"
            lbl = show_legend ? "S$(sid) B$(blk) $(task) take $(take)" : ""
            plot!(plt, tv, yplot, lw=lw, alpha=α, label=lbl)
        else
            # Parametric: X series
            xv, _ = series_from(tr, x)
            # Enforce same length
            n = min(length(xv), length(yv))
            xv = @view xv[1:n]
            yplot = copy(@view yv[1:n])
            if normalize === :zscore
                μ, σ = mean(yplot), std(yplot); σ > 0 && (yplot = (yplot .- μ) ./ σ)
            elseif normalize === :minmax
                m, M = minimum(yplot), maximum(yplot); M > m && (yplot = (yplot .- m) ./ (M - m))
            end
            sid = hasproperty(tr,:subject_id) ? tr.subject_id : "?"
            blk = hasproperty(tr,:block) ? tr.block : "?"
            task = hasproperty(tr,:task) ? string(tr.task) : "?"
            take = hasproperty(tr,:take) ? tr.take : "?"
            lbl = show_legend ? "S$(sid) B$(blk) $(task) take $(take)" : ""
            plot!(plt, xv, yplot, lw=lw, alpha=α, label=lbl)
        end
    end

    # Axis labels
    function varlabel(sym::Symbol)
        sym === :v1    && return "v1"
        sym === :v2    && return "v2"
        sym === :gamma && return "γ (gamma)"
        sym === :zeta  && return "ζ (zeta)"
        sym === :time  && return "t"
        sym === :t1    && return "t1"
        sym === :t2    && return "t2"
        sym === :index && return "sample"
        return String(sym)
    end
    if is_time_x
        xlabel!(plt, varlabel(x === :auto ? :time : x))
        ylabel!(plt, varlabel(y))
    else
        xlabel!(plt, varlabel(x))
        ylabel!(plt, varlabel(y))
    end
    show_legend || plot!(plt, legend=false)
    return plt
end



# Return WAV files for a subject sorted by numeric suffix.
function list_subject_wavs(subject_id::String, audio_dir::String)
    pats  = ["S$(subject_id)_*.wav", "S$(subject_id)_*.WAV"]
    files = String[]
    for p in pats
        append!(files, glob(p, audio_dir))
    end
    parse_idx(f) = try
        base = splitext(splitdir(f)[2])[1]
        parse(Int, split(base, "_")[end])
    catch
        typemax(Int)
    end
    sort!(files, by=parse_idx)
    return files
end

# Extract fingering from filenames like S<id>_<type>_<FING>_...
function fingering_from_filename(basename::String)
    parts = split(basename, "_")
    return length(parts) >= 3 ? parts[3] : ""
end

# Mark one successful take per task using explicit overrides.
function mark_success_by_overrides!(trials::Vector{Trial}, overrides::Dict{Tuple{String,Int,Symbol},Int})
    last_take = Dict{Symbol,Int}()
    for tr in trials
        tr.task == :Practice && continue
        last_take[tr.task] = max(get(last_take, tr.task, 0), tr.take)
    end
    for (task, ntakes) in last_take
        wanted = get(overrides, (trials[1].fingering, trials[1].block, task), 0)
        wanted = wanted == 0 ? ntakes : wanted
        for tr in trials
            tr.success = tr.success || (tr.task == task && tr.take == wanted)
        end
    end
    return trials
end

# implement subject_overview behavior for this analysis pipeline.
function subject_overview(subject_id::String, type::Symbol, path::String)
    refs = discover_blocks(subject_id, type, path)
    overview = NamedTuple[]
    for ref in refs
        all_log = read(ref.logfile, String)
        pairs = parse_block_tasklist(all_log)
        takes_by_task = Dict{Symbol,Int}()
        ordered_trials = NamedTuple[]
        for (ord, (task, datafile_abs)) in enumerate(pairs)
            take = get!(takes_by_task, task, 0) + 1
            takes_by_task[task] = take
            datafile_rel = to_project_relative(datafile_abs; root=path)
            push!(ordered_trials, (order=ord, task=task, take=take, datafile_rel=datafile_rel))
        end
        tasks_counts = OrderedDict{Symbol,Int}()
        for k in sort(collect(keys(takes_by_task)) .|> String)
            tasks_counts[Symbol(k)] = takes_by_task[Symbol(k)]
        end
        audiofile_rel = to_project_relative(ref.audiofile; root=path)
        push!(overview, (fingering=ref.fingering,
                         block=ref.block_idx,
                         audiofile_rel=audiofile_rel,
                         tasks_counts=tasks_counts,
                         ordered_trials=ordered_trials))
    end
    return overview
end

# implement print_subject_overview behavior for this analysis pipeline.
function print_subject_overview(overview; include_practice::Bool=false)
    for blk in overview
        println("Fingering ", blk.fingering, " | Block ", blk.block,
                isempty(blk.audiofile_rel) ? "" : " | audio: "*blk.audiofile_rel)
        println("  Takes per task:")
        for (task, n) in blk.tasks_counts
            if include_practice || task != :Practice
                @printf "    %-14s %d\n" String(task) n
            end
        end
        println("  Ordered trials (order, task, take, datafile_rel):")
        for tr in blk.ordered_trials
            if include_practice || tr.task != :Practice
                @printf "    %3d  %-14s  %2d  %s\n" tr.order String(tr.task) tr.take tr.datafile_rel
            end
        end
        println()
    end
end

# implement make_default_overrides behavior for this analysis pipeline.
function make_default_overrides(ov; include_practice::Bool=false)
    d = Dict{Tuple{String,Int,Symbol},Int}()
    for blk in ov
        for (task, _n) in blk.tasks_counts
            if include_practice || task != :Practice
                d[(blk.fingering, blk.block, task)] = 0
            end
        end
    end
    return d
end

# implement apply_override_choices! behavior for this analysis pipeline.
function apply_override_choices!(overrides::Dict{Tuple{String,Int,Symbol},Int},
                                 choices::Vector{Tuple{String,Int,Symbol,Int}})
    for (f,b,t,sel) in choices
        overrides[(f,b,t)] = sel
    end
    return overrides
end

# implement trajectories behavior for this analysis pipeline.
function trajectories(allexp::Vector{Trial};
                      param=:pressure, align=:onset, window=(-0.1, 1.5))
    out = OrderedDict{NTuple{4,Any}, Tuple{Vector{Float64},Vector{Float64}}}()
    for tr in allexp
        t  = (param==:pressure ? tr.t1 : tr.t2)
        y  = (param==:pressure ? tr.v1 : tr.v2)
        t0 = (align==:onset) ? _trial_onoff_start_time(tr) : 0.0
        tp = t .- t0
        sel = findall(tp .>= window[1] .&& tp .<= window[2])
        !isempty(sel) && (out[(tr.subject_id, tr.type, tr.fingering, tr.task)] = (tp[sel], y[sel]))
    end
    return out
end

# implement compute_sol behavior for this analysis pipeline.
function compute_sol(t1,t2,γ,ζ,u0, p0, fs)
    dt = 1/fs
    tmax = max(t1[end],t2[end])
    i1 = Ref(1)
    # implement affect_p1! behavior for this analysis pipeline.
    function affect_p1!(integrator)
        if i1[] <= length(γ)
            integrator.p[1] = γ[i1[]]
            i1[] += 1
        end
    end
    i2 = Ref(1)
    # implement affect_p2! behavior for this analysis pipeline.
    function affect_p2!(integrator)
        if i2[] <= length(ζ)
            integrator.p[2] = ζ[i2[]]
            i2[] += 1
        end
    end
    cb1 = PresetTimeCallback(t1, affect_p1!, save_positions=(false, false))
    cb2 = PresetTimeCallback(t2, affect_p2!, save_positions=(false, false))
    prob = ODEProblem(saxRN!,u0,(0,tmax), p0)
    sol = solve(prob, Tsit5(), callback = CallbackSet(cb1,cb2), saveat = dt)
    return dropdims(sum(Array(sol)[3:2:end,:],dims=1),dims=1)/2.0
end

# implement play_task behavior for this analysis pipeline.
function play_task(block,task,data,param_map,u0, p0,fs=22.05)
    val = data[block][task]
    γ = _mapvals.(val[:,2],Ref(param_map[1]))
    ζ = _mapvals.(val[:,4],Ref(param_map[2]))
    s = compute_sol(val[:,1],val[:,2],γ,ζ,u0, p0,fs)
    return s
end

# implement assign_onoff_by_beeps! behavior for this analysis pipeline.
function assign_onoff_by_beeps!(trials::Vector{Trial}; f0::Real=700.0, win_ms::Real=120.0, hop_ms::Real=10.0,
    z_beep::Real=8.0, purity_min::Real=0.40, harmonic_max::Real=0.20,
    min_sep::Real=0.15, offset_start::Real=0.05,
    z::Real=3.0, smooth_ms::Real=10.0, minlen::Real=0.20)
    isempty(trials) && return trials
    sort!(trials, by = t -> t.order)
    audiofile = trials[1].audiofile
    isempty(audiofile) && error("assign_onoff_by_beeps!: missing audiofile for this REAL block")
    y, fs = wavread(audiofile)
    y = ndims(y)==1 ? Float64.(y) : vec(mean(y, dims=2))[:]
    beeps = detect_tones(y, fs, f0;
        profile=:pure, edges=:onsets,
        win_ms=win_ms, hop_ms=hop_ms,
        snr_db=z_beep, purity_min=purity_min,
        harmonic_max=harmonic_max, min_sep=min_sep)
    isempty(beeps) && return trials
    nb = length(beeps); nt = length(trials)
    useN = min(nb, nt)
    T = length(y)/fs
    for i in 1:useN
        dur_s = trials[i].duration / 1000.0
        t0 = beeps[i] + offset_start
        t1 = min(beeps[i] + dur_s, T)
        if t1 <= t0
            trials[i].onoff = [(0.0, 0.0, 0.0)]
            continue
        end
        i0 = clamp(round(Int, t0*fs)+1, 1, length(y))
        i1 = clamp(round(Int, t1*fs), i0, length(y))
        ywin = y[i0:i1]
        ts   = collect(0:length(ywin)-1) ./ fs
        segs = detect_onoff_from_signal(ts, ywin; z=z, smooth_ms=smooth_ms, minlen=minlen)
        trials[i].onoff = isempty(segs) ? [(0.0, (i1 - i0)/fs, 0.0)] : [(segs[1][1], segs[1][2], 0.0)]
    end
    for i in useN+1:nt
        trials[i].onoff = [(0.0, 0.0, 0.0)]
    end
    return trials
end

# implement assign_onoff! behavior for this analysis pipeline.
function assign_onoff!(trials::Vector{Trial};
    beep_f0::Real=700.0, beep_win_ms::Real=120.0, beep_hop_ms::Real=10.0,
    beep_core_bw_hz::Real=6.0,  beep_ring_gap_hz::Real=20.0, beep_ring_bw_hz::Real=80.0,
    beep_snr_db::Real=8.0, beep_purity_min::Real=0.40, beep_harmonic_max::Real=0.20,
    beep_min_frames::Int=2, beep_min_sep::Real=0.15,
    offset_start::Real=0.05,
    tone_f0::Union{Nothing,Real}=nothing,
    tone_n_harmonics::Int=6, tone_include_f0::Bool=true, tone_harm_weights::Symbol=:inv,
    tone_win_ms::Real=92.0, tone_hop_ms::Real=10.0,
    tone_core_bw_hz::Real=10.0, tone_ring_gap_hz::Real=30.0, tone_ring_bw_hz::Real=120.0,
    tone_snr_db::Real=6.0, tone_purity_min::Real=0.30, tone_min_harmonics::Int=3,
    tone_min_frames::Int=2, tone_min_silence_ms::Real=40.0, tone_min_len_ms::Real=80.0,
    z::Real=3.0, smooth_ms::Real=10.0, minlen::Real=0.20)
    isempty(trials) && return trials
    sort!(trials, by = t -> t.order)
    audiofile = trials[1].audiofile
    isempty(audiofile) && error("assign_onoff!: missing audiofile for this block")
    y, fs = wavread(audiofile)
    y = ndims(y)==1 ? Float64.(y) : vec(mean(y, dims=2))[:]
    T = length(y)/fs
    beeps = detect_tones(y, fs, beep_f0;
        profile=:pure, edges=:onsets,
        win_ms=beep_win_ms, hop_ms=beep_hop_ms,
        core_bw_hz=beep_core_bw_hz, ring_gap_hz=beep_ring_gap_hz, ring_bw_hz=beep_ring_bw_hz,
        snr_db=beep_snr_db, purity_min=beep_purity_min, harmonic_max=beep_harmonic_max,
        min_frames=beep_min_frames, min_sep=beep_min_sep)
    useN = min(length(beeps), length(trials))
    for i in 1:useN
        dur_s = trials[i].duration / 1000.0
        t0 = beeps[i] + offset_start
        t1 = min(beeps[i] + dur_s, T)
        if t1 <= t0
            trials[i].onoff = [(0.0, 0.0, 0.0)]
            continue
        end
        i0 = clamp(round(Int, t0*fs)+1, 1, length(y))
        i1 = clamp(round(Int, t1*fs), i0, length(y))
        ywin = y[i0:i1]
        ts = collect(0:length(ywin)-1) ./ fs
        segs = Tuple{Float64,Float64}[]
        if tone_f0 !== nothing
            segs = detect_tones(ywin, fs, tone_f0;
                profile=:harmonic, edges=:segments,
                n_harmonics=tone_n_harmonics, include_f0=tone_include_f0, harm_weights=tone_harm_weights,
                win_ms=tone_win_ms, hop_ms=tone_hop_ms,
                core_bw_hz=tone_core_bw_hz, ring_gap_hz=tone_ring_gap_hz, ring_bw_hz=tone_ring_bw_hz,
                snr_db=tone_snr_db, purity_min=tone_purity_min, min_harmonics=tone_min_harmonics,
                min_frames=tone_min_frames, min_silence_ms=tone_min_silence_ms, min_len_ms=tone_min_len_ms)
        end
        if isempty(segs)
            segs_amp = detect_onoff_from_signal(ts, ywin; z=z, smooth_ms=smooth_ms, minlen=minlen)
            if isempty(segs_amp)
                trials[i].onoff = [(0.0, (i1 - i0)/fs, 0.0)]
            else
                lens = map(s -> s[2]-s[1], segs_amp)
                seg = segs_amp[findmax(lens)[2]]
                trials[i].onoff = [(seg[1], seg[2], 0.0)]
            end
        else
            lens = map(s -> s[2]-s[1], segs)
            seg = segs[findmax(lens)[2]]
            trials[i].onoff = [(seg[1], seg[2], 0.0)]
        end
    end
    for i in useN+1:length(trials)
        trials[i].onoff = [(0.0, 0.0, 0.0)]
    end
    return trials
end





# Moved from active regions: currently not used by Pluto outputs
function detect_onoff_from_signal(t::Vector{<:Real}, v::Vector{<:Real};   
                                  z::Real=0.1, smooth_ms::Real=10.0, minlen::Real=0.20)

    length(t) == length(v) || throw(ArgumentError("t and v must have same length"))
    N = length(t)
    N >= 3 || return Tuple{Float64,Float64}[]

    t64 = Float64.(t)
    v64 = Float64.(v)

    # estimate dt and smoothing window
    dt = median(diff(t64))
    dt > 0 || throw(ArgumentError("t must be strictly increasing"))
    w = max(1, round(Int, smooth_ms*1e-3/dt))

    # centered moving average of |v|
    function movmean_abs(x::Vector{Float64}, win::Int)
        win <= 1 && return abs.(x)
        half = win ÷ 2
        xa = abs.(x)
        cs = cumsum(vcat(0.0, xa))
        y = similar(xa)
        @inbounds for i in 1:length(xa)
            lo = max(1, i - half)
            hi = min(length(xa), i + (win - 1 - half))
            y[i] = (cs[hi] - cs[lo - 1]) / (hi - lo + 1)
        end
        return y
    end

    env = movmean_abs(v64, w)
    μ = mean(env)
    σ = std(env) + eps()
    thr = μ + z*σ

    above = env .>= thr

    # find rising/falling edges
    segs = Tuple{Float64,Float64}[]
    in_seg = false
    start_t = 0.0

    for i in 1:N
        if !in_seg && above[i]
            in_seg = true
            start_t = t64[i]
        elseif in_seg && !above[i]
            in_seg = false
            stop_t = t64[i]
            if stop_t - start_t >= minlen
                push!(segs, (start_t, stop_t))
            end
        end
    end
    if in_seg
        stop_t = t64[end]
        if stop_t - start_t >= minlen
            push!(segs, (start_t, stop_t))
        end
    end

    return segs
end

#
# Detect tones (pure or harmonic) centered at f0.
#
# Inputs
# - signal :: Vector{<:Real}
# - fs     :: Real                # sampling rate [Hz]
# - f0     :: Real                # target fundamental [Hz]
#
# Keywords (sensible defaults below)
# - profile        :: Symbol = :pure       # :pure (beeps) or :harmonic (sax notes)
# - edges          :: Symbol = :onsets     # :onsets or :segments
# - n_harmonics    :: Int    = 6           # number of harmonics ABOVE f0 to include when profile=:harmonic
# - include_f0     :: Bool   = true        # include the fundamental band in the harmonic stack
# - harm_weights   :: Symbol = :inv        # :equal | :inv | :inv2 (weights for harmonic sum)
# - win_ms         :: Real   = 120.0       # STFT window length [ms]
# - hop_ms         :: Real   = 10.0        # STFT hop [ms]
# - core_bw_hz     :: Real   = 8.0         # half-bandwidth for each (h*f0) core band [Hz]
# - ring_gap_hz    :: Real   = 25.0        # guard-band just outside core before "ring"
# - ring_bw_hz     :: Real   = 100.0       # width of ring band per harmonic (local noise ref)
# - snr_db         :: Real   = 8.0         # min SNR (core vs ring) in dB (overall)
# - purity_min     :: Real   = 0.35        # min purity = E_core_stack / (E_total - E_core_stack)
# - min_harmonics  :: Int    = 2           # min #harmonics individually above SNR (only for :harmonic)
# - harmonic_max   :: Real   = 0.20        # max (E_2f0+E_3f0)/E_f0 for :pure (penalize harmonics)
# - min_frames     :: Int    = 2           # require this many consecutive positive frames
# - min_silence_ms :: Real   = 30.0        # fill short 0-gaps inside segments up to this (ms)
# - min_len_ms     :: Real   = 60.0        # drop segments shorter than this (ms), only for :segments
# - min_sep        :: Real   = 0.15        # separate onsets by at least this (s), only for :onsets)
#
# Returns
# - if edges == :onsets   → Vector{Float64} of onset times (s)
# - if edges == :segments → Vector{Tuple{Float64,Float64}} of (on, off) in seconds
#

function detect_tones(signal::Vector{<:Real}, fs::Real, f0::Real;
    profile::Symbol = :pure, edges::Symbol = :onsets,
    n_harmonics::Int = 6, include_f0::Bool = true, harm_weights::Symbol = :inv,
    win_ms::Real = 120.0, hop_ms::Real = 10.0,
    core_bw_hz::Real = 8.0, ring_gap_hz::Real = 25.0, ring_bw_hz::Real = 100.0,
    snr_db::Real = 8.0, purity_min::Real = 0.35, min_harmonics::Int = 2,
    harmonic_max::Real = 0.20, min_frames::Int = 2,
    min_silence_ms::Real = 30.0, min_len_ms::Real = 60.0, min_sep::Real = 0.15)

    x = Float64.(signal)
    Nw_target = max(256, round(Int, fs*win_ms/1_000))
    Nw = 1 << ceil(Int, log2(Nw_target))          # next power of 2
    H  = max(1, round(Int, fs*hop_ms/1_000))
    w  = 0.5 .- 0.5 .* cos.(2π .* (0:Nw-1) ./ Nw) # Hann
    K  = Nw ÷ 2 + 1

    hz2bin(f) = clamp(round(Int, f*Nw/fs) + 1, 1, K)

    # list of harmonic indices to use
    Hs = include_f0 ? collect(1:(1+n_harmonics)) : collect(2:(1+n_harmonics))
    Hs = [h for h in Hs if h*f0 < fs/2 - 1]       # keep under Nyquist
    isempty(Hs) && return (edges === :segments ? Tuple{Float64,Float64}[] : Float64[])

    # harmonic weights
    wh = if harm_weights === :equal
        ones(Float64, length(Hs))
    elseif harm_weights === :inv
        1.0 ./ Float64.(Hs)
    elseif harm_weights === :inv2
        1.0 ./ (Float64.(Hs).^2)
    else
        ones(Float64, length(Hs))
    end
    wh ./= sum(wh)  # normalize

    # helpers
    @inline band_energy(Y, lo, hi) = (lo<=hi ? sum(@view Y[lo:hi]) : 0.0)

    # per-frame loop
    flags = Bool[]               # decision per frame
    times = Float64[]            # frame center times
    score_core = Float64[]       # total core energy (for onset ranking if needed)
    idx = 1
    eps1 = 1e-12
    snr_lin_thresh = 10.0^(snr_db/10)

    while idx + Nw - 1 <= length(x)
        @views frame = x[idx:idx+Nw-1]
        Y = abs2.(rfft(frame .* w))  # power spectrum
        E_tot = sum(Y)

        # accumulate across harmonics
        E_core_stack = 0.0
        E_ring_stack = 0.0
        n_h_ok = 0

        for (j, h) in enumerate(Hs)
            k0    = hz2bin(h*f0)
            dcore = max(1, hz2bin(h*f0 + core_bw_hz) - k0)
            dgap  = max(1, hz2bin(h*f0 + ring_gap_hz) - k0)
            dring = max(dgap+1, hz2bin(h*f0 + ring_gap_hz + ring_bw_hz) - k0)

            k_lo = max(1, k0 - dcore); k_hi = min(K, k0 + dcore)
            r1_lo = max(1, k0 + dgap);  r1_hi = min(K, k0 + dring)
            r2_lo = max(1, k0 - dring); r2_hi = max(1, k0 - dgap)

            E_core_h = band_energy(Y, k_lo, k_hi)
            nr1 = max(0, r1_hi - r1_lo + 1); nr2 = max(0, r2_hi - r2_lo + 1)
            E_ring_h = band_energy(Y, r1_lo, r1_hi) + band_energy(Y, r2_lo, r2_hi)
            nrbins = max(1, nr1 + nr2)
            E_ring_avg_h = E_ring_h / nrbins

            # individual harmonic SNR test (for :harmonic profile’s min_harmonics)
            if E_core_h / (E_ring_avg_h + eps1) >= snr_lin_thresh
                n_h_ok += 1
            end

            E_core_stack += wh[j] * E_core_h
            E_ring_stack += wh[j] * E_ring_avg_h
        end

        E_rest = max(eps1, E_tot - E_core_stack)  # rest of spectrum

        snr_lin = E_core_stack / (E_ring_stack + eps1)
        purity  = E_core_stack / (E_rest + eps1)

        ok = (snr_lin >= snr_lin_thresh) & (purity >= purity_min)

        if profile === :pure
            # penalize harmonic presence: compare (approx) 2f0+3f0 vs f0
            # estimate f0 energy as the first term (if included); else relax
            if include_f0 && !isempty(Hs) && Hs[1] == 1
                # recompute a quick 2f0+3f0 vs f0 ratio
                kf0 = hz2bin(f0); d0 = max(1, hz2bin(f0 + core_bw_hz) - kf0)
                E_f0 = band_energy(Y, max(1, kf0 - d0), min(K, kf0 + d0))
                E_h  = 0.0
                for h in (2,3)
                    kh = hz2bin(h*f0); dh = max(1, hz2bin(h*f0 + core_bw_hz) - kh)
                    E_h += band_energy(Y, max(1, kh - dh), min(K, kh + dh))
                end
                ok &= (E_h / (E_f0 + eps1) <= harmonic_max)
            end
        else
            # require multiple harmonics individually "good"
            ok &= (n_h_ok >= min_harmonics)
        end

        push!(flags, ok)
        push!(times, (idx + Nw/2 - 1) / fs)
        push!(score_core, E_core_stack)
        idx += H
    end

    # temporal persistence
    good = falses(length(flags))
    run = 0
    for i in eachindex(flags)
        run = flags[i] ? run + 1 : 0
        good[i] = run >= min_frames
    end

    # fill short gaps (morphological closing) up to min_silence_ms
    if any(good)
        gap_max = max(0, round(Int, (min_silence_ms/1_000) / (hop_ms/1_000)))
        if gap_max > 0
            i = 1
            while i <= length(good)
                if good[i]
                    # skip inside segment
                    while i <= length(good) && good[i]; i += 1; end
                    # count zeros gap
                    zstart = i
                    while i <= length(good) && !good[i]; i += 1; end
                    zend = i - 1
                    zlen = (zstart<=zend) ? (zend - zstart + 1) : 0
                    if zlen > 0 && zlen <= gap_max && i <= length(good) && good[i]
                        fill!(view(good, zstart:zend), true)
                    end
                else
                    i += 1
                end
            end
        end
    end

    if edges === :segments
        # build (on, off) from good[]
        segs = Tuple{Float64,Float64}[]
        i = 1
        while i <= length(good)
            if good[i]
                j = i
                while j < length(good) && good[j+1]; j += 1; end
                t_on  = times[i]
                t_off = times[j]
                if (t_off - t_on) >= (min_len_ms/1_000)
                    push!(segs, (t_on, t_off))
                end
                i = j + 1
            else
                i += 1
            end
        end
        return segs
    else
        # onsets only: take transitions 0->1 and enforce min_sep
        onsets = Float64[]
        prev = false
        for i in eachindex(good)
            if good[i] && !prev
                push!(onsets, times[i])
            end
            prev = good[i]
        end
        # enforce min_sep
        if length(onsets) > 1
            filt = Float64[onsets[1]]
            for k in 2:length(onsets)
                if onsets[k] - filt[end] >= min_sep
                    push!(filt, onsets[k])
                end
            end
            onsets = filt
        end
        return onsets
    end
end

# implement _trial_onoff_start_time behavior for this analysis pipeline.
function _trial_onoff_start_time(tr::Trial)
    isempty(tr.onoff) && return 0.0
    return tr.onoff[1][1]
end

# implement evaluate_results_kde behavior for this analysis pipeline.
function evaluate_results_kde(; kwargs...)
    # Intentionally left empty for future implementation.
    return nothing
end


#
# plot_trials_values(trials; y=:v1, x=:time, apply_map=false,
#                    param_map_override=nothing, align=:start, normalize=:none,
#                    lw=1.5, α=0.9, show_legend=false)
#
# Plot either against time or as a parametric curve.
#
# Arguments
# - trials  :: Vector{Trial}
#
# Keywords
# - y       :: Symbol  # one of :v1, :v2, :gamma, :zeta
# - x       :: Symbol  # :time, :t1, :t2, :index, or one of :v1, :v2, :gamma, :zeta
# - apply_map :: Bool  # if true and y/x are :v1/:v2, map to :gamma/:zeta using param_map
# - param_map_override :: Union{Nothing,Vector{Tuple}}  # optional [ (a,b,c,d), (a,b,c,d) ]
# - align   :: :start | :none   # only applies when x is time-like (:time/:t1/:t2)
# - normalize :: :none | :zscore | :minmax  # applied to Y only
# - lw, α, show_legend :: styling
#
# Assumptions
# - Each Trial may carry: v1, v2, t1, t2, t/ts, subject_id, block, task, take, param_map
#

#endregion