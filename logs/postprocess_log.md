# Postprocess Log - Final Report

**Date:** 2026-06-18 11:06:04  
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
- Excluded subjects before trial loading: 97

## Execution Trace
- START: 2026-06-18T11:02:58.452
- END:   2026-06-18T11:06:04.845
- Duration: 186.39 seconds

## Output Files (Regenerated)
- src/sessions/processed_data/all_real_trials_wamplitudes_onoff.jld2
- src/sessions/processed_data/all_model_trials_wamplitudes_onoff.jld2

## Trial Counts & Audio Coverage
| Metric | Real | Model |
|---|---:|---:|
| Total Trials | 140 | 140 |
| With audio-derived amplitudes (a1 non-empty) | 140 | 140 |
| Missing audiofile | 0 | 0 |
| Success Rate | 100.0% | 100.0% |

## Onoff Region Cardinality Checks

### Nonlegato (expected 2 mode1 + 2 mode2)
| Dataset | Total | Correct | Not matching |
|---|---:|---:|---:|
| Real | 56 | 56 | 0 |
| Model | 56 | 56 | 0 |

### Legato (expected 1 mode1 + 1 mode2)
| Dataset | Total | Correct | Not matching |
|---|---:|---:|---:|
| Real | 56 | 56 | 0 |
| Model | 56 | 56 | 0 |

### Overtone (expected mode2 only)
| Dataset | Total | Exactly 1 mode2 | At least 1 mode2 |
|---|---:|---:|---:|
| Real | 28 | 28 | 28 |
| Model | 28 | 28 | 28 |

## Conclusion
New production JLD2 outputs were regenerated.
This log has been overwritten with metrics from this rerun.
