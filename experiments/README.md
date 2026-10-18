# Experiments: the code behind the paper's numbers

These notebooks contain the experiment code **exactly as it was run** for the paper, in the order it was run. The only changes
from our working notebooks are:

* result paths are relative to one folder, `RESULTS_ROOT` (default `results/`, set in `setup/s0_paths.py`);
* notes to ourselves ("paste this output ...") were removed from comments and messages;
* functions that the final experiments never call were left out (an old baseline runner, and the step-by-step gating code that
  the final method replaced by a single check before motion), together with small self-test prints.

The printed output under each cell is the output of our own run (paths shortened; long progress logs shortened to their first and
last lines), so the notebooks can be read without running them.

## How to run

Each notebook starts with one set-up cell that runs the scripts in `setup/` with `%run -i`, so all cells share one namespace,
as in the original notebooks. Then run the cells from top to bottom. Every long cell is **resumable**: it appends one row per
case to a CSV file and skips the cases that are already saved, so it can be stopped and started again. With the result files of
our run in `results/`, the run cells print "0 to run" and the results cells rebuild every table in seconds.

| Set-up script | Contents |
|---|---|
| `s0_paths.py` | result folders |
| `s1_models_and_simulator.py` | SmolVLA, LIBERO helpers, instruction parser, Grounding DINO, a test environment with depth |
| `s2_engine.py` | the evidence engine (grounding, camera model, graded relations, G / S / D) and the experiment helpers (episode set-up, the seven test conditions) |
| `s3_calibrate_camera.py` | camera calibration (run once per session; moves the gripper and checks the projection) |
| `s4_engine_fixes.py` | the final version of the scene verification (`perceive_scene`) |
| `s5_policy_helpers.py` | SmolVLA's action for one observation, the action-heading term A (tested and removed), the evidence picture |
| `s6_study1_setup.py` | Study 1: splits, result files, verification of one case, offline gate, statistics |
| `s7_study2_setup.py` | Study 2: OWLv2, the hybrid detector, the gate runner and comparison tables |
| `s8_two_policies_and_judge_setup.py` | one runner for all gate variants, GPU memory planning, tables for two policies and the VLM judge |

## Which notebook gives which result

| Notebook | Paper | Main result files |
|---|---|---|
| `01_study1_first_gate.ipynb` | Section IV: Tables II and III, data of Figs. 2 and 3 | `v4/verify_v4.csv`, `v4/rollouts_v4.csv`, `v4/baseline_clean_v4.csv`, `v5/*` |
| `02_study2_hybrid_gate.ipynb` | Sections V-A, V-B, V-E: Tables IV and V | `v6/verify_owl*.csv`, `v6/fresh_verify_*.csv`, `v6/fresh_rollouts.csv` |
| `03_second_policy_and_vlm_judge.ipynb` | Sections V-B, V-C: Tables V and VI, data of Fig. 5 | `v7/xvla_rollouts.csv`, `v7/vlm_judge_*.csv` |
| `04_attached_run_and_demo.ipynb` | Section V-D: Table VII, the demo video | `v7/live_*.csv`, `v7/demo/` |

Development and test data were kept apart: Study 1 developed on initial states 0-1 and tested on 2-4; Study 2 developed on
states 0-2 and tested on 5-9, which no earlier experiment had touched. The freeze cells refuse to overwrite a frozen settings file,
and the test sweeps refuse to start before it exists.

## Not included

* The stand-alone comparison of four detectors outside the gate (Grounding DINO, MM-Grounding-DINO, LLMDet and OWLv2 on the 70
  development scenes of initial state 0; Section V-A and Fig. 4 of the paper) was run in a separate notebook that is not part of
  this repository yet. The gate-level detector experiments of Table IV are included (notebook 02).
* Earlier development notebooks (versions 1 to 3 of the method) and optional experiments that are not in the paper.

## Hardware and time

One NVIDIA GeForce RTX 5050 laptop GPU (8 GB) under WSL2. The detectors stay on the GPU; only one robot policy (or the VLM) is
loaded next to them at a time. The time of every long cell is written in its title.
