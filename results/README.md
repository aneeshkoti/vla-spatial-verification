# Result files

The result tables of our runs go here, in the same folders the experiment notebooks write to. They are copied from the results
folder of a run with

```bash
python scripts/collect_results.py --src ~/smolvla_project/results
```

which copies only the tables (`.csv`, `.json`, `.jsonl`; no images or videos) and writes `MANIFEST.json` with the size and SHA-1
of every file. Then `python scripts/reproduce_numbers.py` recomputes the paper's numbers from them.

| Folder | What | Main files |
|---|---|---|
| `v2/` | the first SmolVLA baseline (used once, to check that baseline episodes are reproducible) | `baseline_clean.csv` |
| `v4/` | Study 1: SmolVLA baseline, the gate's log for the 350 development and test cases, rollouts, the frozen settings | `baseline_clean_v4.csv`, `verify_v4.csv`, `verify_candidates_v4.jsonl`, `rollouts_v4.csv`, `frozen_config.json`, `tables/` |
| `v5/` | Study 1 analyses: oracle identity, moved plate, official LeRobot evaluation, comparison tables | `oracle_identity_test.csv`, `memorisation_test.csv`, `official_eval_summary.json`, `tables/` |
| `v6/` | Study 2: detector variants on development scenes, the frozen hybrid settings, both gates on the 350 fresh scenes, SmolVLA rollouts | `verify_owl*.csv`, `frozen_config_v6.json`, `fresh_verify_gdino.csv`, `fresh_verify_hybrid.csv`, `fresh_verify_hybrid_oracle.csv`, `fresh_rollouts.csv` |
| `v7/` | X-VLA rollouts, the VLM judge, the attached run | `xvla_rollouts.csv`, `vlm_judge_dev.csv`, `vlm_judge_fresh.csv`, `vlm_judge_choice.json`, `live_*.csv` |

## Columns of the gate logs (`verify_v4.csv`, `fresh_verify_*.csv`)

One row per case (task, initial state, condition). Suffix `_0` = first grounding pass, `_1` = the re-grounding pass (synonyms).

| Column | Meaning |
|---|---|
| `kind`, `valid` | test condition; 1 if the instruction matches the scene |
| `G_0`, `S_0`, `D_0` | evidence of the first pass |
| `S_rel_0`, `D_rel_0` | 2D / 3D relation degree before the grounding-trust factor |
| `amb_0` | ambiguity factor (1 = one bowl fits clearly, 0.5 = two bowls fit equally well) |
| `t_conf_0`, `ref_confs_0`, `goal_conf_0` | detector confidences of the selected target, references and goal |
| `sel_0`, `right_target_0`, `right_ref_0`, `right_goal_0` | analysis only (from the simulator): which object was selected, and whether it was the right one |
| `t_perceive0_s`, `t_perceive1_s` | time of each grounding pass (seconds) |

The decision is not stored in these files: it is recomputed from them with the frozen settings (`vla_gate.decision.decide_all`),
so the same file gives the decision of any weighting or threshold that was evaluated on development scenes.
