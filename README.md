# Register-Control Saxophone Experiment

Clean repository for the register-control saxophone experiment. It contains the code and curated data needed to run the experiment, rebuild the processed trial datasets, and reproduce the analysis/figure pipeline.

## Layout

- `audiofiles/`: task audio files used during postprocessing and mode-amplitude extraction.
- `scripts/`: reproducible processing/statistics scripts.
  - `postprocess_refresh.jl`: rebuilds processed trial JLD2 files.
  - `run_permutation_families.jl`: computes permutation-test rows.
- `src/`: experiment runtime, postprocessing, and analysis code.
  - `generate_blocks.jl`: generates experiment blocks.
  - `rt_sax_experiment_block.jl`: runs experiment blocks.
  - `rt_sax_control.jl` and `rt_serial.jl`: real-time control and serial IO.
  - `rt_sax_experiment_analysis.jl`: postprocessing pipeline.
  - `rt_sax_experiment_statistics.jl`: Wasserstein/statistical analysis.
  - `Pluto/`: notebooks for QA and Figures 3-7 workflow.
  - `sessions/`: raw session logs/data, reviewed CSVs, and processed JLD2 outputs.
  - `impedances/alto/`: alto saxophone impedance files used by the model/control code.

## Julia Setup

From the repository root:

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Some runtime hardware paths depend on serial/audio devices and may need local configuration in `src/rt_sax_configuration.toml`.

## Rebuild Processed Data

```julia
include("scripts/postprocess_refresh.jl")
```

This regenerates:

- `src/sessions/processed_data/all_real_trials_wamplitudes_onoff.jld2`
- `src/sessions/processed_data/all_model_trials_wamplitudes_onoff.jld2`

## Figures And Statistics

Use the Pluto notebooks:

- `src/Pluto/03_rt_sax_experiment_plot_all.jl`: trial QA plots.
- `src/Pluto/05_rt_sax_experiment_results.jl`: Figures 3 and 4.
- `src/Pluto/08_rt_sax_experiment_distances.jl`: distance summaries and permutation table.

Permutation rows are generated outside Pluto:

```julia
ENV["RTSAX_PERM_N_PERMUTATIONS"] = "1000"
ENV["RTSAX_PERM_OUT_DIR"] = joinpath(pwd(), "src", "sessions", "processed_data", "permutation_tests")
include("scripts/run_permutation_families.jl")
```

For quicker preliminary runs, use a different output folder such as `permutation_tests_N100`.

## Notes

- Subject 97 is excluded from the current analysis; subject 13 is retained.
- The current Wasserstein ground metric uses global standard-deviation scaling without mean subtraction.
- The code still uses the internal task name `Overtone` for what is labelled as `Multiphonic` in figures and tables.
