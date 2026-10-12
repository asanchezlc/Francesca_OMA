# OMA pipeline: Méndez-Núñez bridge (Case Study II)

Run every script from the repository root with the project environment:

```
.venv\Scripts\python.exe <script>.py
```

Records (not in git): `C:\Users\User\Documents\DOCTORADO\Francesca\MendezNuñez\`
- `Roberto_signal_Puente_Circunv.xlsx`: training record (submitted paper).
- `Accelerations_21.xlsx` … `Accelerations_28.xlsx`: independent test records (major revision),
  converted from the `.tdms` files with `z_from_tdms_to_xlsx.py`.

## A. Submitted paper

| Script | Reads | Writes | Used for |
|---|---|---|---|
| `OMA_Code_1_EFDD.py` | training record | `results/EFDD_results.json` (`Fn_oma_Hz`, `Phi_oma`, `Phi_oma_real`, `FIM`, `detFIM`, `Ed`), `results/plots/EFDD_peak_picking.png` | Source of `EFDD_results.json` for the sensor-selection scripts |
| `qaoa-fm_OMA.py`, `brute_force_sensor_selection.py`, `ga_sensor_selection_pymoo.py` | `results/EFDD_results.json` (the `Ed` vector, FIM utility term) | (selection results) | Sensor selection (QUBO / brute force / GA). The last two load `qaoa-fm_OMA.py` by path |
| `OMA_Code_1_COVSSI_subsets.py` | training record | `results/autossi_results.json`, `results/autossi_subsets_results.json`, `results/plots/autossi_*.png`, `results/plots/AutoMAC_Global.png` | Table 6: COV-SSI on the full network and on {S2,S3,S6}, {S1,S2,S8}, {S2,S3,S8} |

Order: `OMA_Code_1_EFDD.py` before the selection scripts. `OMA_Code_1_COVSSI_subsets.py` is independent.

The Table 6 version of `OMA_Code_1_COVSSI_subsets.py` (subset matching with `factor_freq = 0.98`,
`factor_mac = 0.02`; `min_pctg = 0.3` for {S2,S3,S6}) is tagged `submitted-table6` in git.

## B. Major revision: out-of-sample validation

Run in this order:

| Step | Script | Reads | Writes (all under `results/`) | Runtime |
|---|---|---|---|---|
| 1 | `OMA_Code_1_OOS_1_run_oma.py` | training record + `Accelerations_*.xlsx` | `oos_bridge/modal_results.json` (Fn, Phi of every global and subset run), `oos_bridge/settings.json` (settings, file hashes), `oos_bridge/excluded_records.csv`, `oos_bridge/plots/`, `oos_bridge/run_log.txt` | ~10 min |
| 2 | `OMA_Code_1_OOS_2_reference_modes.py` | `Accelerations_21.xlsx` (global OMA rerun), `oos_bridge/modal_results.json`, `oos_bridge/settings.json` | `reference_modes/`: `reference_modes.json` / `.csv` (Ref_1..Ref_11), `reference_autoMAC.*`, `consistency_*.csv`, `settings.json` | ~1 min |
| 3 | `OMA_Code_1_OOS_3_validate_subsets.py` | `oos_bridge/modal_results.json`, `oos_bridge/settings.json`, `reference_modes/reference_modes.json`, `reference_modes/settings.json`, `reference_modes/consistency_MAC_assigned.csv` | `oos_bridge/v3/`: `settings.json`, `global_ref_labelling.csv`, `per_mode_matching.csv`, `per_record_subset_metrics.csv`, `aggregated_metrics.csv`, `ref_mode_recovery.csv`, `training_in_sample.csv`, `training_vs_table6.csv`, `run_log.txt` | seconds |
| 4 | `OMA_Code_1_OOS_4_all_layouts_run.py` | training record + `Accelerations_*.xlsx`, `oos_bridge/modal_results.json`, `oos_bridge/settings.json` | `oos_bridge/all_layouts/`: `modes/<record>.jsonl` (subset modes of all 120 three-sensor layouts, resumable store), `sanity_check_table6_layouts.csv`, `failures.csv`, `settings.json`, `run_log.txt` | ~25 min with `N_JOBS = 8` |
| 5 | `OMA_Code_1_OOS_5_all_layouts_analysis.py` | `oos_bridge/all_layouts/modes/`, the step 1-3 inputs above, `oos_bridge/v3/per_record_subset_metrics.csv`, and the inputs of J (training record, `signal_Geo.xlsx`, `results/EFDD_results.json`, `bruteforce_results/bruteforce_ranking_all_combinations.csv`) | `oos_bridge/all_layouts/analysis/`: `per_layout_record_metrics.csv`, `per_layout_summary.csv`, `table6_layouts_rank.csv`, `correlations.csv`, `topk_by_J.csv`, `scatter_*.png`, `settings.json`, `run_log.txt` | ~1 min |

- Step 1 runs the global OMA of every record and the subset OMA of the three Table-6 layouts;
  step 4 runs the subset OMA of all 120 layouts. Steps 2-5, the debug tool and the superseded
  v2 import step 1 as a module
  (`import OMA_Code_1_OOS_1_run_oma as v1`) and take every OMA and matching constant from it.
  If you change a constant or a subset in step 1, rerun step 1 before the later steps.
- Steps 4-5 extend step 3 from the three Table-6 layouts to all 120 three-sensor layouts
  (reviewer comment R2-C3). Step 4 runs the subset OMA of every (record, layout) pair with
  the step 1 settings and skips the pairs already stored when restarted; the results do not
  depend on `N_JOBS`. Step 5 imports the matching and metrics of step 3, and recomputes the
  objective J of the submitted paper with the functions of `brute_force_sensor_selection.py`
  (checked against its saved ranking).
- Step 1 also writes matching tables of its own (`per_mode_matching.csv`, `aggregated_metrics.csv`,
  …) with the original matcher; they are superseded by `oos_bridge/v3/`.
- Step 3 evaluates the subsets against the reference modes only: global modes without a Ref
  label (training: 5.232 and 6.085 Hz) are excluded.

Interactive tool: `OMA_Code_1_OOS_debug.py` reruns the global OMA of selected records with
editable settings and shows the figures in the VS Code debugger. It writes nothing.

## Superseded scripts (`z_` prefix)

| Script | Status |
|---|---|
| `z_OMA_Code_1_COVSSI_subsets_oos_v2.py` | Re-matching against every global mode of a record; outputs in `results/oos_bridge/v2/`. Kept for traceability. Nothing imports it: step 3 contains a verified copy of its matcher. |
| `z_from_tdms_to_xlsx.py` | One-off conversion of the `.tdms` test records to `.xlsx`. |

## Renamed scripts

| Previous name | Current name |
|---|---|
| `OMA_Code_1.py` | `OMA_Code_1_EFDD.py` |
| `OMA_Code_1_COVSSI_subsets_oos.py` | `OMA_Code_1_OOS_1_run_oma.py` |
| `OMA_Code_1_COVSSI_OBTAIN_REF_MODES.py` | `OMA_Code_1_OOS_2_reference_modes.py` |
| `OMA_Code_1_COVSSI_subsets_DEBUG.py` | `OMA_Code_1_OOS_debug.py` |
| `OMA_Code_1_COVSSI_subsets_oos_v2.py` | `z_OMA_Code_1_COVSSI_subsets_oos_v2.py` |

Outputs written before the renaming (`results/oos_bridge/v2/settings.json`,
`results/reference_modes/settings.json`) still mention the previous names.
