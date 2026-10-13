"""Integrity and scientific-array invariants, independent of model execution."""
import json,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
class ReleaseTests(unittest.TestCase):
    def test_current_manuscript_assets(self):
        rows=json.loads((ROOT/"docs/MANUSCRIPT_FIGURES.json").read_text())
        self.assertEqual(len(rows),22)
        self.assertEqual(sum(len(r["assets"]) for r in rows),24)
        for row in rows:
            for rel in row["assets"]:
                self.assertTrue((ROOT/rel).is_file(),rel)
                self.assertTrue((ROOT/rel).resolve().is_relative_to(ROOT))
    def test_projected_jacobian_error_arrays(self):
        cases=[
          ("experiments/advection/figures/current/advection_aligned_paper_style_diagnostics_results.npz","A_true","A_icon","A_error"),
          ("experiments/revision/feature_map/burgers_near_shock_16_pairs.npz","feature_near_shock_A_true","feature_near_shock_A_icon","feature_near_shock_A_err"),
          ("experiments/allen_cahn/figures/current/allen_cahn_paper_style_diagnostics_results.npz","feature_interface_A_true","feature_interface_A_icon","feature_interface_A_err")]
        for rel,truth,icon,error in cases:
            with np.load(ROOT/rel,allow_pickle=False) as data:
                self.assertEqual(data[truth].shape,(32,32))
                np.testing.assert_allclose(data[icon]-data[truth],data[error],rtol=1e-5,atol=1e-6)
    def test_synthetic_example_shapes(self):
        for pde in ("advection","burgers","allen_cahn"):
            with np.load(ROOT/"data/samples"/(pde+".npz"),allow_pickle=False) as d:
                for key in ("cond_k","cond_v","qoi_k","qoi_v"):
                    self.assertEqual(d[key].shape,(1,6,100,1))
                    self.assertTrue(np.isfinite(d[key]).all())
if __name__=="__main__": unittest.main()
