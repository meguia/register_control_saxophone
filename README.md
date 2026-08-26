# Register control in saxophone playing

This is the compact publication branch for the register-control saxophone
experiment. It contains the original experiment program, the paper cohort's
raw sensor sessions and WAV files, and the minimum analysis needed to rebuild
paper Figures 3, 4, 5, and 6.

There is deliberately no Julia package module and no notebook dependency. The
workflow uses the original source files through `include` and five numbered
scripts.

## Repository contents

```text
register_control_saxophone/
├── audiofiles/                         # 280 WAV files, 14 paper subjects
├── data/derived/
│   └── multistability_map_v1.jld2      # compact validated Figure 6 map
├── scripts/
│   ├── 01_postprocess.jl
│   ├── 02_wasserstein_distances.jl
│   ├── 03_permutation_tests.jl
│   ├── 04_recompute_figure6_map.jl
│   └── 05_generate_figures.jl
├── src/
│   ├── rt_sax_control.jl               # original real-time model/control
│   ├── rt_sax_experiment_block.jl      # original experiment procedure
│   ├── rt_serial.jl                    # original serial protocol
│   ├── rt_sax_experiment_test.jl       # original hardware test
│   ├── generate_blocks.jl              # original counterbalancing utility
│   ├── rt_sax_configuration.toml
│   ├── subjects_config.json
│   ├── arduino/RT_Sax_ADC/RT_Sax_ADC.ino
│   ├── impedances/alto/Dx4.jld2
│   ├── sax_model_core.jl               # audio-free copy of model equations
│   ├── rt_sax_experiment_analysis.jl   # loading, postprocessing, plots
│   ├── rt_sax_experiment_statistics.jl # Wasserstein and permutation tests
│   ├── analysis/                       # continuation internals
│   ├── regularized_bifurcation/        # continuation guide
│   ├── nonregularized_multistability/  # exact-model validation map
│   └── sessions/
│       ├── rt_sax_experiment_sept_2025.csv
│       ├── reviewed_data/              # hand-reviewed choices and intervals
│       ├── *.log and *.dat             # raw paper-cohort sessions
│       └── processed_data/             # generated and git-ignored
└── results/                             # generated and git-ignored
```

The publication cohort contains subjects 13, 22, 27, 33, 34, 37, 49, 50, 64,
70, 80, 83, 88, and 90. The retained raw inputs are 56 session-log records, 56
block logs, 686 sensor `.dat` files, 280 WAV files, and eight reviewed CSV
tables. Subject 97 was a pilot and is not included.

## Julia environment

Julia 1.12 is recommended. From the repository root:

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
```

The project is an application environment, not a Julia package. Do not run
`using RegisterControlSaxophone`; include the entry file or numbered script
shown below.

## Performing the experiment

The hardware-facing procedure is preserved from the original repository. The
following files are intentionally untouched:

- `src/rt_sax_control.jl`
- `src/rt_sax_experiment_block.jl`
- `src/rt_serial.jl`
- `src/rt_sax_experiment_test.jl`
- `src/generate_blocks.jl`
- `src/rt_sax_configuration.toml`
- `src/subjects_config.json`
- `src/arduino/RT_Sax_ADC/RT_Sax_ADC.ino`

Upload the Arduino sketch first. Then edit only the local serial port and, if
needed, gain in `src/rt_sax_configuration.toml`. For example, use `COM3` on
Windows or the appropriate `/dev/...` device on Linux/macOS.

Start Julia from `src` because the historical procedure uses relative paths:

```julia
cd("src")
import Pkg
Pkg.activate("..")
include("rt_sax_experiment_block.jl")
run_block("13", 1)
```

During a block, keys `0` through `5` start the displayed task, `S` stops a
task, `P` stops and prints a summary, and `Q` closes the block. The experiment
writes its `.log`, `.dat`, and session-index records below `src/sessions`.

To test audio and serial acquisition without running a block:

```julia
cd("src")
import Pkg
Pkg.activate("..")
include("rt_sax_experiment_test.jl")
manager, source, parameter_map = run_test("13", "Dx4")
start_update!(manager, source, parameter_map)
# perform a short test
stop_model(manager, source)
```

`generate_blocks.jl` can create a new randomized schedule, but do not overwrite
the deposited `subjects_config.json` when reproducing the published experiment.

Batch analysis and continuation load `src/sax_model_core.jl` so that they do
not initialize an audio device or serial port. Its `set_parameters` and
`saxRN!` definitions are a direct, audio-free transcription of the equations
in the untouched `rt_sax_control.jl`; it does not replace the experiment
runtime.

## Reproducing Figures 3–6

Run all commands from the repository root. On Windows, the same commands can
be entered in the Julia REPL with forward slashes in paths.

### Step 1: rebuild processed trials

Shell:

```sh
julia --project=. --threads=auto scripts/01_postprocess.jl
```

REPL:

```julia
include("scripts/01_postprocess.jl")
```

This reads the session index, reviewed choices, raw `.dat` files, and WAV
files. It writes:

- `src/sessions/processed_data/all_model_trials_wamplitudes_onoff.jld2`
- `src/sessions/processed_data/all_real_trials_wamplitudes_onoff.jld2`
- `src/sessions/processed_data/postprocess_summary.txt`

### Step 2: compute the Figure 5 Wasserstein distances

Shell:

```sh
julia --project=. --threads=auto scripts/02_wasserstein_distances.jl
```

REPL:

```julia
include("scripts/02_wasserstein_distances.jl")
```

The analysis uses physical pressure-force coordinates, global pooled standard
deviations as scale factors without mean subtraction, a 30 × 30 KDE grid, and
the `OptimalTransport.jl` Sinkhorn implementation. It writes the complete JLD2
summary and a readable CSV table below `src/sessions/processed_data`.

### Step 3: run the permutation tests

The paper setting is 1000 permutations for each of four comparison families.
The families run concurrently, so use several Julia threads.

Shell:

```sh
julia --project=. --threads=4 scripts/03_permutation_tests.jl
```

REPL, including the explicit paper setting:

```julia
ENV["RTSAX_PERM_N_PERMUTATIONS"] = "1000"
include("scripts/03_permutation_tests.jl")
```

For a smoke test, set `RTSAX_PERM_N_PERMUTATIONS` to `10`. Results are written
to `src/sessions/processed_data/permutation_tests`. Optional positional family
names are `nonlegato_low_high`, `same_note_asc_desc`,
`overtone_vs_nonlegato`, and `legato_low_high`.

The null distributions are produced by subject-stratified relabeling: labels
are exchanged only within the relevant subject and comparison stratum, so the
repeated-measures structure is preserved. Each family has a fixed random-seed
offset, making reruns reproducible.

### Step 4: optionally recompute the continuation-guided Figure 6 map

This expensive step is not required to draw the published figure: the compact
validated final product is deposited as
`data/derived/multistability_map_v1.jld2`.

To reproduce that product from the differential equations, the pipeline first
continues periodic solutions of the smoothly regularized model with
`eta = 0.001`. It computes the equilibrium Hopf curves, codimension-two points,
periodic branches, folds, period doubling, Neimark-Sacker candidates,
resonance information, and Periodic-Schur Floquet stability. Those continued
states are only guides. The second stage transports them to the original
non-regularized 18-variable equations, integrates them directly, checks their
recurrence and attraction with two solvers and finite perturbations, and runs
the basin-edge mixed-response calculations. Compatible point checkpoints are
reused automatically.

Shell, full publication grid:

```sh
julia --project=. --threads=16 scripts/04_recompute_figure6_map.jl --profile=final
```

REPL, full publication grid:

```julia
include("scripts/04_recompute_figure6_map.jl")
result = run_figure6_computation(profile=:final)
```

If the regularized guide is already complete, use:

```julia
result = run_figure6_computation(
    profile=:final,
    run_regularized=false,
)
```

A short installation test is available as `profile=:smoke`. The full run is
restartable and can take many hours. A completed final run exports
`src/sessions/processed_data/multistability_map_recomputed.jld2`. The figure
script selects this file automatically; otherwise it selects the deposited
compact map. `RTSAX_FIGURE6_MAP` can point to another compatible final map.

### Step 5: generate Figures 3, 4, 5, and 6

Shell:

```sh
julia --project=. --threads=auto scripts/05_generate_figures.jl
```

REPL:

```julia
include("scripts/05_generate_figures.jl")
```

The script writes PNG, SVG, and PDF versions to `results`:

- Figure 3: model-condition pressure-force trajectories
- Figure 4: real-instrument pressure-force trajectories
- Figure 5: seven Wasserstein-distance summaries
- Figure 6: the fixed-parameter response map with successful model-condition
  non-legato and performed-multiphonic samples

If the Figure 5 summary is absent, the figure script computes it automatically.

## Figure 6 interpretation

The raw map is a fixed-parameter accessibility calculation. `T1` and `T2` are
the two principal minimal recurrence-time classes of the complete 18-variable
state. They are not synonymous with the continuation labels P1 and P2: P1 and
P2 describe branch provenance relative to a period-doubling event, whereas
`T1` and `T2` describe the measured global recurrence time. Because the first
two resonances are close to an octave, a high-family P2 response can recur with
approximately `T1`.

The `T1 + T2` hatch means that both response classes have local smoothed
support; it does not assert that both were validated at every displayed pixel
or that a single trajectory has two minimal periods. `Mixed` shows local
support for boundary-initialized simulations that still contained substantial
activity in both acoustic modes at the 3 s cutoff. It is not an automatic
classification of a stable multiphonic, torus, or chaotic attractor. Purple
points retain their experimental meaning: they are samples from the performed
multiphonic task.

For presentation, Figure 6 applies Gaussian smoothing factor `0.7` and support
threshold `0.7` to the discrete validated masks. Smoothing changes only the
drawn local-support fields; the raw classifications and counts remain in the
JLD2 product. The deposited final map validates a 389 × 72 grid with 28,008
points, 16,635 `T1` cells, 9,095 `T2` cells, 8,661 sampled overlaps, 1,181
completed mixed-edge simulations, and 269 simulations still mixed at cutoff.

## Integrity check for the experiment procedure

To confirm that the publication branch did not alter the historical runtime,
compare it with the target repository's `main` branch:

```sh
git diff origin/main -- \
  src/rt_sax_control.jl \
  src/rt_sax_experiment_block.jl \
  src/rt_serial.jl \
  src/rt_sax_experiment_test.jl \
  src/generate_blocks.jl \
  src/rt_sax_configuration.toml \
  src/subjects_config.json \
  src/arduino/RT_Sax_ADC/RT_Sax_ADC.ino
```

The command should print nothing.
