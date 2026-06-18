# Postprocess Log - Final Report

**Date:** 2026-06-18 03:32:33  
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
- START: 2026-06-18T03:26:51.751
- END:   2026-06-18T03:32:33.857
- Duration: 342.11 seconds

## Output Files (Regenerated)
- src/sessions/processed_data/all_real_trials_wamplitudes_onoff.jld2
- src/sessions/processed_data/all_model_trials_wamplitudes_onoff.jld2

## Trial Counts & Audio Coverage
| Metric | Real | Model |
|---|---:|---:|
| Total Trials | 145 | 150 |
| With audio-derived amplitudes (a1 non-empty) | 140 | 140 |
| Missing audiofile | 5 | 10 |
| Success Rate | 96.6% | 93.3% |

## Onoff Region Cardinality Checks

### Nonlegato (expected 2 mode1 + 2 mode2)
| Dataset | Total | Correct | Not matching |
|---|---:|---:|---:|
| Real | 58 | 56 | 2 |
| Model | 60 | 56 | 4 |

### Legato (expected 1 mode1 + 1 mode2)
| Dataset | Total | Correct | Not matching |
|---|---:|---:|---:|
| Real | 58 | 56 | 2 |
| Model | 60 | 56 | 4 |

### Overtone (expected mode2 only)
| Dataset | Total | Exactly 1 mode2 | At least 1 mode2 |
|---|---:|---:|---:|
| Real | 29 | 28 | 28 |
| Model | 30 | 28 | 28 |

## Conclusion
New production JLD2 outputs were regenerated.
This log has been overwritten with metrics from this rerun.
