using Dates
using JLD2

include(joinpath(@__DIR__, "..", "src", "rt_sax_experiment_analysis.jl"))

const ROOT = normpath(joinpath(@__DIR__, ".."))
const SESSION_REL = "sessions"
const LOG_CSV = "rt_sax_experiment_sept_2025.csv"
const REAL_CHOICES = joinpath("reviewed_data", "real_choices.csv")
const MODEL_CHOICES = joinpath("reviewed_data", "model_choices.csv")
const REAL_OVERTONE = joinpath(ROOT, "src", "sessions", "reviewed_data", "real_overtone_onoff.csv")
const MODEL_OVERTONE = joinpath(ROOT, "src", "sessions", "reviewed_data", "model_overtone_onoff.csv")
const AUDIO_ROOT = joinpath(ROOT, "audiofiles")
const PROC_DIR = joinpath(ROOT, "src", "sessions", "processed_data")
const REAL_OUT = joinpath(PROC_DIR, "all_real_trials_wamplitudes_onoff.jld2")
const MODEL_OUT = joinpath(PROC_DIR, "all_model_trials_wamplitudes_onoff.jld2")
const LOG_OUT = joinpath(ROOT, "logs", "postprocess_log.md")

function pct(n::Int, d::Int)
    d == 0 && return "NaN"
    return string(round(100n / d; digits=1), "%")
end

function task_stats(trials)
    taskset = Dict(
        :Nonlegato => filter(t -> t.task in (:NonlegatoAsc, :NonlegatoDesc), trials),
        :Legato => filter(t -> t.task in (:LegatoAsc, :LegatoDesc), trials),
        :Overtone => filter(t -> t.task == :Overtone, trials),
    )

    nonleg = taskset[:Nonlegato]
    leg = taskset[:Legato]
    over = taskset[:Overtone]

    nonleg_ok = count(t -> length(onoff_regions(t; mode=1)) == 2 && length(onoff_regions(t; mode=2)) == 2, nonleg)
    leg_ok = count(t -> length(onoff_regions(t; mode=1)) == 1 && length(onoff_regions(t; mode=2)) == 1, leg)
    overtone_mode2_eq1 = count(t -> length(onoff_regions(t; mode=2)) == 1, over)
    overtone_with_any = count(t -> !isempty(onoff_regions(t; mode=2)), over)

    return (
        nonleg_total = length(nonleg),
        nonleg_ok = nonleg_ok,
        nonleg_missing = length(nonleg) - nonleg_ok,
        leg_total = length(leg),
        leg_ok = leg_ok,
        leg_missing = length(leg) - leg_ok,
        overtone_total = length(over),
        overtone_mode2_eq1 = overtone_mode2_eq1,
        overtone_with_any = overtone_with_any,
    )
end

function dataset_stats(label::String, trials)
    total = length(trials)
    audio_filled = count(t -> !isempty(t.a1), trials)
    missing_audio = count(t -> isempty(strip(t.audiofile)) || lowercase(strip(t.audiofile)) == "missing", trials)
    ts = task_stats(trials)

    return (
        label = label,
        total = total,
        audio_filled = audio_filled,
        missing_audio = missing_audio,
        success_rate = pct(audio_filled, total),
        task = ts,
    )
end

function to_markdown(start_ts::DateTime, end_ts::DateTime, real_stats, model_stats)
    return """
# Postprocess Log - Final Report

**Date:** $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))  
**Workspace:** register_control_saxophone  
**Status:** COMPLETE - Postprocess rerun finished and outputs regenerated

## Objective
Rerun the full postprocessing pipeline, regenerate production outputs, and overwrite this log with fresh metrics.

## Inputs Used
- Session log: src/sessions/rt_sax_experiment_sept_2025.csv
- Real choices: src/sessions/reviewed_data/real_choices.csv
- Model choices: src/sessions/reviewed_data/model_choices.csv
- Real overtone table: src/sessions/reviewed_data/real_overtone_onoff.csv
- Model overtone table: src/sessions/reviewed_data/model_overtone_onoff.csv
- Audio root: audiofiles/

## Execution Trace
- START: $(start_ts)
- END:   $(end_ts)
- Duration: $(round(Dates.value(end_ts - start_ts)/1000; digits=2)) seconds

## Output Files (Regenerated)
- src/sessions/processed_data/all_real_trials_wamplitudes_onoff.jld2
- src/sessions/processed_data/all_model_trials_wamplitudes_onoff.jld2

## Trial Counts & Audio Coverage
| Metric | Real | Model |
|---|---:|---:|
| Total Trials | $(real_stats.total) | $(model_stats.total) |
| With audio-derived amplitudes (a1 non-empty) | $(real_stats.audio_filled) | $(model_stats.audio_filled) |
| Missing audiofile | $(real_stats.missing_audio) | $(model_stats.missing_audio) |
| Success Rate | $(real_stats.success_rate) | $(model_stats.success_rate) |

## Onoff Region Cardinality Checks

### Nonlegato (expected 2 mode1 + 2 mode2)
| Dataset | Total | Correct | Not matching |
|---|---:|---:|---:|
| Real | $(real_stats.task.nonleg_total) | $(real_stats.task.nonleg_ok) | $(real_stats.task.nonleg_missing) |
| Model | $(model_stats.task.nonleg_total) | $(model_stats.task.nonleg_ok) | $(model_stats.task.nonleg_missing) |

### Legato (expected 1 mode1 + 1 mode2)
| Dataset | Total | Correct | Not matching |
|---|---:|---:|---:|
| Real | $(real_stats.task.leg_total) | $(real_stats.task.leg_ok) | $(real_stats.task.leg_missing) |
| Model | $(model_stats.task.leg_total) | $(model_stats.task.leg_ok) | $(model_stats.task.leg_missing) |

### Overtone (expected mode2 only)
| Dataset | Total | Exactly 1 mode2 | At least 1 mode2 |
|---|---:|---:|---:|
| Real | $(real_stats.task.overtone_total) | $(real_stats.task.overtone_mode2_eq1) | $(real_stats.task.overtone_with_any) |
| Model | $(model_stats.task.overtone_total) | $(model_stats.task.overtone_mode2_eq1) | $(model_stats.task.overtone_with_any) |

## Conclusion
New production JLD2 outputs were regenerated.
This log has been overwritten with metrics from this rerun.
"""
end

function main()
    start_ts = now()

    println("Starting full postprocess rerun...")

    real_trials = all_trials_wamplitudes_onoff(
        SESSION_REL,
        LOG_CSV,
        REAL_CHOICES;
        type = :Real,
        root = ROOT,
        real_audio_root = AUDIO_ROOT,
        overtone_real_csv = REAL_OVERTONE
    )

    model_trials = all_trials_wamplitudes_onoff(
        SESSION_REL,
        LOG_CSV,
        MODEL_CHOICES;
        type = :Model,
        root = ROOT,
        model_audio_root = AUDIO_ROOT,
        overtone_model_csv = MODEL_OVERTONE
    )

    # Ensure saved outputs exist.
    isfile(REAL_OUT) || error("Expected output not found: $REAL_OUT")
    isfile(MODEL_OUT) || error("Expected output not found: $MODEL_OUT")

    real_stats = dataset_stats("Real", real_trials)
    model_stats = dataset_stats("Model", model_trials)

    end_ts = now()
    md = to_markdown(start_ts, end_ts, real_stats, model_stats)
    mkpath(dirname(LOG_OUT))
    open(LOG_OUT, "w") do io
        write(io, md)
    end

    println("Done.")
    println("Real trials: $(real_stats.total), audio filled: $(real_stats.audio_filled)")
    println("Model trials: $(model_stats.total), audio filled: $(model_stats.audio_filled)")
    println("Wrote log: $LOG_OUT")
end

main()
