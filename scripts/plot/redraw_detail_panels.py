#!/usr/bin/env python3
"""Rebuild detailed panels from archived arrays. Layouts need not be byte-identical."""
import argparse,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/plot"))
from render_regime_comparisons import load_plots
sys.path.insert(0,str(ROOT/"src"))
from plotting.style import apply_paper_style
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    apply_paper_style()
    d=np.load(ROOT/"experiments/advection/figures/current/advection_aligned_paper_style_diagnostics_results.npz")
    result={"phase_block_error":{"c_values":d["phase_c_values"],"k":d["phase_k"],"block_error":d["phase_block_error"]}}
    ns=load_plots("experiments/advection/evaluation/evaluate_advection.py",["plot_phase_block_error"],out)
    ns["plot_phase_block_error"](result)
    d=np.load(ROOT/"experiments/allen_cahn/figures/current/allen_cahn_paper_style_diagnostics_results.npz")
    result={"base_context":{kind:{field:d[f"base_{kind}_{field}"] for field in
        ("k","eps_values","icon_mean","cum_rel_err_cos","cum_rel_err_sin")} for kind in ("near_zero","moderate","interface")}}
    ns=load_plots("experiments/allen_cahn/evaluation/evaluate_allen_cahn.py",["plot_base_state_context_spectrum"],out,base_label=lambda k:k.replace("_"," "))
    ns["plot_base_state_context_spectrum"](result)
    d=np.load(ROOT/"experiments/advection/bandwidth_ablation/figures/advection_mode_ablation_results.npz")
    spectrum={"k":d["spectrum_k"],"true":d["spectrum_true"],"models":{f"train modes={m}":{
        "mean":d[f"spectrum_mode{m}"],"cum_cos":d[f"spectrum_mode{m}_cum_cos"],"cum_sin":d[f"spectrum_mode{m}_cum_sin"]}
        for m in (8,16)}}
    feature={"models":{f"train modes={m}":{"block_err":d[f"feature_block_err_mode{m}"],
        "rel_F":float(np.linalg.norm(d[f"feature_A_mode{m}"]-d["feature_A_true"])/np.linalg.norm(d["feature_A_true"]))} for m in (8,16)}}
    def save(fig,name):
        path=out/name
        if path.exists(): raise FileExistsError(path)
        fig.savefig(path,bbox_inches="tight"); fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight"); plt.close(fig)
    def bandwidth(axes):
        for ax in np.asarray(axes,dtype=object).ravel():
            ax.axvline(8,linestyle=":",color="0.35"); ax.axvline(16,linestyle="--",color="0.15")
    ns=load_plots("experiments/advection/bandwidth_ablation/evaluate_mode_ablation.py",
        ["plot_query_spectrum","plot_phase_block_error"],out,add_training_bandwidth_lines=bandwidth,
        _savefig=save,REPRESENTATIVE_C=0.8,FEATURE_C=0.8,FEATURE_PAIR_MODES=16,TRAIN_BW8=8,TRAIN_BW16=16)
    ns["plot_query_spectrum"]({"spectrum":spectrum}); ns["plot_phase_block_error"]({"feature":feature})
    d=np.load(ROOT/"experiments/burgers/figures/current/burgers_paper_style_diagnostics_results.npz")
    fig,axes=plt.subplots(3,1,figsize=(7,8),constrained_layout=True)
    for ax,kind in zip(axes,("smooth","steep","near_shock")):
        x=d[f"base_prediction_{kind}_x"]
        for field,label in (("u0","initial"),("true","reference"),("pred","ICON")):
            ax.plot(x,d[f"base_prediction_{kind}_{field}"],label=label)
        ax.set_title(kind.replace("_"," ")); ax.set_xlabel("x"); ax.grid(True); ax.legend()
    save(fig,"burgers_base_state_predictions.png")
if __name__=="__main__": main()
