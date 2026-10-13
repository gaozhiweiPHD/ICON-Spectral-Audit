# Current manuscript figure map

Numbering comes from the actual manuscript, including appendices.
Legacy filenames fig07, fig10, fig13 and fig19 correspond to current Figures 8, 11, 14 and 20.
Run each command group once: shared commands generate several figures and refuse overwrites.
For Figure 22 substitute each of the three PDE names.
Redraws preserve scientific arrays but need not match the archived raster/layout exactly.
Figures 16/17 use export-only commands because historical array-only redraw cannot be certified.
All 24 original assets can be exported with python scripts/export_figures.py --output-dir outputs/manuscript-assets.

| Figure / label | Current assets | Source data | Reproduction command |
|---|---|---|---|
| 1 `fig:adv_rel_l2` | `figures/manuscript/advection_spectral_robustness_summary.pdf` | results/revision/prompt_variability/advection_prompt_stats_full_range.csv | `python scripts/revision/plot_prompt_variability.py --pde advection --output-dir outputs/adv-summary` |
| 2 `fig:adv_query_spectrum` | `figures/manuscript/advection_query_spectrum_over_c_with_cumulative_relative_error.png` | experiments/advection/figures/current/advection_aligned_paper_style_diagnostics_results.npz | `python scripts/plot/render_regime_comparisons.py --output-dir outputs/regimes` |
| 3 `fig:adv_phase_block_error` | `figures/manuscript/advection_phase_block_error_over_c.png` | same advection NPZ as Figure 2 | `python scripts/plot/redraw_detail_panels.py --output-dir outputs/details` |
| 4 `fig:adv_feature_map` | `figures/manuscript/fig04_square_feature_map.pdf` | same advection NPZ as Figure 2 | `python scripts/plot/replot_square_feature_maps.py --output-dir outputs/maps` |
| 5 `fig:adv_demo_noise` | `figures/manuscript/advection_demo_parameter_noise.pdf` | results/revision/demo_parameter_noise/advection_demo_parameter_noise.csv | `python scripts/revision/plot_demo_parameter_noise.py --output outputs/noise.png` |
| 6 `fig:burgers_pred_error` | `figures/manuscript/burgers_spectral_robustness_summary.pdf` | results/revision/prompt_variability/burgers_prompt_stats_full_range.csv | `python scripts/revision/plot_prompt_variability.py --pde burgers --output-dir outputs/burgers-summary` |
| 7 `fig:burgers_base_context_spectrum` | `figures/manuscript/burgers_base_state_context_spectrum_with_cumulative_relative_error.png` | experiments/burgers/figures/current/burgers_paper_style_diagnostics_results.npz | `python scripts/plot/plot_current_burgers.py --output-dir outputs/burgers` |
| 8 `fig:burgers_feature_map` | `figures/manuscript/fig07_square_feature_map.pdf` | experiments/revision/feature_map/burgers_near_shock_16_pairs.npz | `python scripts/plot/replot_square_feature_maps.py --output-dir outputs/maps` |
| 9 `fig:burgers_wrong_prediction` | `figures/manuscript/burgers_wrong_relative_prediction_error_over_hidden_viscosity.png` | experiments/burgers/corrupted_prompt/run/wrong_eval_figs/burgers_wrong_negative_control_results.npz | `python scripts/plot/render_regime_comparisons.py --output-dir outputs/regimes` |
| 10 `fig:burgers_wrong_query_spectrum` | `figures/manuscript/burgers_wrong_query_spectrum_with_cumulative_relative_error.png` | same corrupted-prompt NPZ as Figure 9 | `python scripts/plot/render_regime_comparisons.py --output-dir outputs/regimes` |
| 11 `fig:burgers_wrong_feature_map` | `figures/manuscript/fig10_square_feature_map.pdf` | same corrupted-prompt NPZ as Figure 9; 14 Fourier pairs | `python scripts/plot/replot_square_feature_maps.py --output-dir outputs/maps` |
| 12 `fig:ac_pred_error` | `figures/manuscript/allen_cahn_spectral_robustness_summary.pdf` | results/revision/prompt_variability/allen_cahn_prompt_stats_full_range.csv | `python scripts/revision/plot_prompt_variability.py --pde allen_cahn --output-dir outputs/ac-summary` |
| 13 `fig:ac_base_context_spectrum` | `figures/manuscript/allen_cahn_base_state_context_spectrum_with_cumulative_relative_error.png` | experiments/allen_cahn/figures/current/allen_cahn_paper_style_diagnostics_results.npz | `python scripts/plot/redraw_detail_panels.py --output-dir outputs/details` |
| 14 `fig:ac_feature_map` | `figures/manuscript/fig13_square_feature_map.pdf` | same Allen–Cahn NPZ as Figure 13; interface base state | `python scripts/plot/replot_square_feature_maps.py --output-dir outputs/maps` |
| 15 `fig:burgers_base_pred` | `figures/manuscript/burgers_base_state_predictions.png` | Burgers main NPZ base_prediction_* arrays | `python scripts/plot/redraw_detail_panels.py --output-dir outputs/details` |
| 16 `fig:ac_base_pred` | `figures/manuscript/allen_cahn_base_state_predictions.png` | archived image only; base prediction arrays absent from saved Allen–Cahn NPZ | `python scripts/export_figures.py --output-dir outputs/manuscript-assets` |
| 17 `fig:adv_ablation_prediction` | `figures/manuscript/advection_ablation_prediction_error_over_c.png` | archived image; prediction means/SDs missing from ablation NPZ | `python scripts/export_figures.py --output-dir outputs/manuscript-assets` |
| 18 `fig:adv_ablation_query_spectrum` | `figures/manuscript/advection_ablation_query_spectrum_c0.80.png` | experiments/advection/bandwidth_ablation/figures/advection_mode_ablation_results.npz | `python scripts/plot/redraw_detail_panels.py --output-dir outputs/details` |
| 19 `fig:adv_ablation_phase` | `figures/manuscript/advection_ablation_phase_block_error_c0.80.png` | same bandwidth-ablation NPZ as Figure 18 | `python scripts/plot/redraw_detail_panels.py --output-dir outputs/details` |
| 20 `fig:adv_ablation_feature_map` | `figures/manuscript/fig19_square_feature_map.pdf` | same bandwidth-ablation NPZ as Figure 18 | `python scripts/plot/replot_square_feature_maps.py --output-dir outputs/maps` |
| 21 `fig:training_proximity` | `figures/manuscript/advection_training_proximity.pdf` | experiments/revision/training_proximity/advection_training_proximity.csv | `python scripts/revision/plot_training_proximity.py --pde advection --output outputs/proximity.png` |
| 22 `fig:context_size` | `figures/manuscript/advection_context_size.pdf`<br>`figures/manuscript/burgers_context_size.pdf`<br>`figures/manuscript/allen_cahn_context_size.pdf` | experiments/revision/context_size/{advection,burgers,allen_cahn}_context_size.csv | `python scripts/revision/plot_context_size.py --pde PDE --output outputs/PDE-context.png` |
