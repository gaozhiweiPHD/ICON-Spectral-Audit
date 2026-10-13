# Context-size ablation

The existing checkpoint can be evaluated with fewer than five demonstrations.
The attention mask and sequence construction are data-shape dependent, so
`N_demo = 1, 3, 5` require no retraining.

It cannot be evaluated faithfully with `N_demo = 10` or `20`: the trained
`func_pos_embedding` has `demo_max_num * 3 = 18` entries, exactly enough for
five demonstrations plus one query. A longer prompt requests unseen and
out-of-range positional indices and is incompatible with the checkpoint.
Changing/expanding this parameter would alter the architecture and require a
policy for untrained weights, so it is not done silently.

Run the supported evaluation with:

```bash
python scripts/revision/run_context_size.py --pde advection --context-sizes 1 3 5 --repeats 10
# Repeat with --pde burgers and --pde allen_cahn, or submit slurm/context_size.sbatch.
```

The script records raw rows and refuses unsupported sizes unless the model
architecture is changed in a separate, explicitly approved experiment.
