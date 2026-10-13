# -*- coding: utf-8 -*-
"""
Plot-only / evaluation-only diagnostics for the corrupted-prompt Burgers ICON.

This script does NOT train. It is intended for the negative-control Burgers run,
where the model was trained on corrupted prompts: the context/demo pairs were
assembled from one hidden viscosity c_demo, while the query pair was assembled
from a different hidden viscosity c_query.

The evaluation here is clean: all test prompts are generated from the correct
Burgers solution operator with one consistent viscosity per prompt. Therefore,
any prediction or spectral degradation is a property of the corrupted-trained
model, not of the evaluation data.

Figures saved by default:
  1. burgers_wrong_relative_prediction_error_over_hidden_viscosity.png
  2. burgers_wrong_query_spectrum_with_cumulative_relative_error.png
     (with clean-trained error curves if a clean checkpoint is found)
  3. burgers_wrong_fourier_feature_blocks_cos_sin_true_icon_error.png
     (cos/sin Fourier blocks plotted separately)
  4. burgers_wrong_representative_prediction.png
  5. burgers_wrong_negative_control_results.npz

Optional clean-model comparison:
  If a clean Burgers checkpoint is specified or found, the prediction-error,
  cumulative spectral-error, representative-spectrum, and feature-map figures
  also overlay the clean-trained model. Otherwise only the corrupted-trained
  model is plotted.

Put this file in the same folder as models_utils.py and run:
    python plot_burgers_wrong_data_negative_control.py

No argparse is used, so it is safe to run from a notebook with:
    %run plot_burgers_wrong_data_negative_control.py
"""

from __future__ import annotations

from collections import namedtuple
from functools import partial
from typing import Tuple
import glob
import os
import sys
from pathlib import Path

import numpy as np
if os.environ.get("ICON_NUMERICS_ONLY") == "1":
    plt = None
    Line2D = None
else:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

import jax
import jax.numpy as jnp
import jax.tree_util as tree
from flax import linen as nn
from flax.linen import MultiHeadDotProductAttention
from flax import serialization
from einshape import jax_einshape as einshape

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from common import models_utils as mu
from common.project import BURGERS_CHECKPOINT_DIR, BURGERS_CORRUPTED_RUN_DIR
if plt is None:
    apply_shared_plot_style = lambda: None
else:
    from plotting.style import apply_paper_style as apply_shared_plot_style
    from plotting.regimes import (style_parameter_curves, style_parameter_sweep, add_regime_key, comparison_filename)


# =============================================================================
# 0. Configuration: match the corrupted Burgers training run
# =============================================================================

seed = 2026

# Prompt/data settings.
num_pairs = 6
num_demos = 5
n_grid = 100
n_modes = 16
fourier_decay = 1.0
bias_scale = 0.0
initial_amp = 0.7
normalize_each_function = False

# Burgers PDE.
T_final = 0.25
c_range: Tuple[float, float] = (0.005, 0.05)
n_solver_steps = 800

# Model settings.
MODEL_SIZE = "paper_like"
if MODEL_SIZE == "paper_like":
    model_config = {
        "demo_max_num": num_pairs,
        "index_mode": "learn",
        "transformer": {
            "n_layers": 6,
            "n_heads": 8,
            "head_dim": 32,
            "model_dim": 256,
            "widening_factor": 4,
        },
        "out_dim": 1,
    }
elif MODEL_SIZE == "small":
    model_config = {
        "demo_max_num": num_pairs,
        "index_mode": "learn",
        "transformer": {
            "n_layers": 4,
            "n_heads": 4,
            "head_dim": 32,
            "model_dim": 128,
            "widening_factor": 4,
        },
        "out_dim": 1,
    }
else:
    raise ValueError(MODEL_SIZE)

# -----------------------------------------------------------------------------
# Checkpoints.
# -----------------------------------------------------------------------------
# Corrupted model checkpoint. If WRONG_CKPT_PATH is None, the latest
# params_step_*.msgpack in WRONG_CKPT_DIR is loaded.
WRONG_CKPT_DIR = str(BURGERS_CORRUPTED_RUN_DIR / "ckpt")
WRONG_CKPT_PATH = None

# Optional clean model checkpoint for visual comparison.
# Leave CLEAN_CKPT_PATH=None to auto-search the candidate directories below.
# Set LOAD_CLEAN_COMPARISON=False if you only want corrupted-model figures.
LOAD_CLEAN_COMPARISON = True
CLEAN_CKPT_PATH = str(BURGERS_CHECKPOINT_DIR)
CLEAN_CKPT_DIR_CANDIDATES = [
    "./icon_burgers_viscosity_ckpt",
    "../icon_burgers_viscosity_ckpt",
    "./burgers_train_true_spectrum_ckpt",
    "../burgers_train_true_spectrum_ckpt",
]
# Example:
# CLEAN_CKPT_PATH = "./icon_burgers_viscosity_ckpt/params_step_200000.msgpack"

# Output folder.
fig_dir = str(BURGERS_CORRUPTED_RUN_DIR / "wrong_eval_figs")
os.makedirs(fig_dir, exist_ok=True)
save_figs = True

# Evaluation grids and computational sizes.
paper_c_values = np.asarray([0.002, 0.0035, 0.005, 0.01, 0.02, 0.035, 0.05, 0.07, 0.10])
paper_prediction_prompts_per_c = 10
paper_query_contexts_per_c = 5
paper_query_max_mode = 40
paper_feature_c_value = 0.02
paper_feature_base_kind = "near_shock"  # recommended: "steep" or "near_shock"
paper_feature_max_pair_mode = 14
paper_representative_c = 0.02

PAPER_FEATURE_CMAP = "magma"
PAPER_ERROR_CMAP = "cividis"


# =============================================================================
# 1. Plot style and containers
# =============================================================================

def set_paper_plot_style():
    apply_shared_plot_style()
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2.0,
        "axes.linewidth": 1.0,
        "grid.alpha": 0.30,
        "figure.dpi": 120,
    })


if plt is not None:
    set_paper_plot_style()

Data = namedtuple(
    "Data",
    [
        "demo_cond_k", "demo_cond_v", "demo_cond_mask",
        "demo_qoi_k", "demo_qoi_v", "demo_qoi_mask",
        "quest_cond_k", "quest_cond_v", "quest_cond_mask",
        "quest_qoi_k", "quest_qoi_mask",
    ],
)

RawBurgers = namedtuple(
    "RawBurgers",
    ["cond_k", "cond_v", "qoi_k", "qoi_v", "c", "coeff_sin", "coeff_cos", "bias"],
)


# =============================================================================
# 2. Random functions and true Burgers solver
# =============================================================================

def eval_random_fourier(x: jnp.ndarray, coeff_sin: jnp.ndarray, coeff_cos: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    n_modes_local = coeff_sin.shape[-1]
    modes = jnp.arange(1, n_modes_local + 1, dtype=x.dtype)
    phase = 2.0 * jnp.pi * x[..., None] * modes
    y = jnp.sum(
        coeff_sin[..., None, :] * jnp.sin(phase)
        + coeff_cos[..., None, :] * jnp.cos(phase),
        axis=-1,
    )
    return y + bias[..., None]


def maybe_normalize_pair(cond_v: jnp.ndarray, qoi_v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    if not normalize_each_function:
        return cond_v, qoi_v
    mean = jnp.mean(cond_v, axis=2, keepdims=True)
    std = jnp.std(cond_v, axis=2, keepdims=True) + 1e-6
    return (cond_v - mean) / std, (qoi_v - mean) / std


def _burgers_rhs(u: jnp.ndarray, c: jnp.ndarray, k: jnp.ndarray, dealias: jnp.ndarray) -> jnp.ndarray:
    u_hat = jnp.fft.fft(u, axis=-1)
    u2_hat = jnp.fft.fft(u * u, axis=-1) * dealias
    nonlin = -0.5 * jnp.fft.ifft(1j * k * u2_hat, axis=-1).real
    diff = c[:, None] * jnp.fft.ifft(-(k ** 2) * u_hat, axis=-1).real
    return nonlin + diff


def _rk4_step(u: jnp.ndarray, c: jnp.ndarray, dt: float, k: jnp.ndarray, dealias: jnp.ndarray) -> jnp.ndarray:
    k1 = _burgers_rhs(u, c, k, dealias)
    k2 = _burgers_rhs(u + 0.5 * dt * k1, c, k, dealias)
    k3 = _burgers_rhs(u + 0.5 * dt * k2, c, k, dealias)
    k4 = _burgers_rhs(u + dt * k3, c, k, dealias)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_burgers_batch(u0: jnp.ndarray, c: jnp.ndarray, T: float, nsteps: int) -> jnp.ndarray:
    N = u0.shape[-1]
    dt_step = T / float(nsteps)
    freq = jnp.fft.fftfreq(N, d=1.0 / N)
    k = 2.0 * jnp.pi * freq
    dealias = (jnp.abs(freq) <= (N / 3.0)).astype(jnp.complex64)

    def body(u, _):
        return _rk4_step(u, c, dt_step, k, dealias), None

    uT, _ = jax.lax.scan(body, u0, xs=None, length=nsteps)
    return uT


solve_burgers_batch_jit = jax.jit(solve_burgers_batch, static_argnames=("nsteps",))


# =============================================================================
# 3. Clean prompt generation for evaluation
# =============================================================================

def generate_burgers_raw_data(
    key: jax.Array,
    num_operators: int,
    pairs_per_operator: int,
    n_grid: int = 100,
    n_modes: int = 16,
    c_range: Tuple[float, float] = (0.005, 0.05),
    T_final: float = 0.25,
    fourier_decay: float = 1.0,
    bias_scale: float = 0.0,
    initial_amp: float = 0.7,
    n_solver_steps: int = 800,
) -> RawBurgers:
    key_c, key_coef, key_bias = jax.random.split(key, 3)

    c = jax.random.uniform(
        key_c,
        shape=(num_operators,),
        minval=c_range[0],
        maxval=c_range[1],
    )

    modes = jnp.arange(1, n_modes + 1, dtype=jnp.float32)
    scale = initial_amp / (modes ** fourier_decay)

    coeff = jax.random.normal(
        key_coef,
        shape=(num_operators, pairs_per_operator, 2, n_modes),
    ) * scale
    coeff_sin = coeff[:, :, 0, :]
    coeff_cos = coeff[:, :, 1, :]
    bias = bias_scale * jax.random.normal(key_bias, shape=(num_operators, pairs_per_operator))

    x = jnp.linspace(0.0, 1.0, n_grid, endpoint=False, dtype=jnp.float32)
    x_full = jnp.broadcast_to(x, (num_operators, pairs_per_operator, n_grid))

    cond_k = x_full[..., None]
    qoi_k = x_full[..., None]
    cond_v0 = eval_random_fourier(x_full, coeff_sin, coeff_cos, bias)

    flat_u0 = cond_v0.reshape(num_operators * pairs_per_operator, n_grid)
    flat_c = jnp.repeat(c, pairs_per_operator)
    flat_uT = solve_burgers_batch_jit(flat_u0, flat_c, T_final, nsteps=n_solver_steps)

    cond_v = cond_v0[..., None]
    qoi_v = flat_uT.reshape(num_operators, pairs_per_operator, n_grid, 1)
    cond_v, qoi_v = maybe_normalize_pair(cond_v, qoi_v)

    return RawBurgers(
        cond_k=cond_k.astype(jnp.float32),
        cond_v=cond_v.astype(jnp.float32),
        qoi_k=qoi_k.astype(jnp.float32),
        qoi_v=qoi_v.astype(jnp.float32),
        c=c.astype(jnp.float32),
        coeff_sin=coeff_sin.astype(jnp.float32),
        coeff_cos=coeff_cos.astype(jnp.float32),
        bias=bias.astype(jnp.float32),
    )


def build_icon_data(raw: RawBurgers, demo_num: int = num_demos) -> tuple[Data, jnp.ndarray]:
    bs = raw.cond_k.shape[0]
    cond_len = raw.cond_k.shape[2]
    qoi_len = raw.qoi_k.shape[2]

    data = Data(
        demo_cond_k=raw.cond_k[:, :demo_num],
        demo_cond_v=raw.cond_v[:, :demo_num],
        demo_cond_mask=jnp.ones((bs, demo_num, cond_len), dtype=bool),
        demo_qoi_k=raw.qoi_k[:, :demo_num],
        demo_qoi_v=raw.qoi_v[:, :demo_num],
        demo_qoi_mask=jnp.ones((bs, demo_num, qoi_len), dtype=bool),
        quest_cond_k=raw.cond_k[:, demo_num:demo_num + 1],
        quest_cond_v=raw.cond_v[:, demo_num:demo_num + 1],
        quest_cond_mask=jnp.ones((bs, 1, cond_len), dtype=bool),
        quest_qoi_k=raw.qoi_k[:, demo_num:demo_num + 1],
        quest_qoi_mask=jnp.ones((bs, 1, qoi_len), dtype=bool),
    )
    label = raw.qoi_v[:, demo_num:demo_num + 1]
    return data, label


def make_burgers_prompts_at_c(
    key: jax.Array,
    c_value: float,
    num_prompts: int,
    pairs_per_prompt: int = num_pairs,
    fixed_query_u0: jnp.ndarray | None = None,
) -> tuple[RawBurgers, Data, jnp.ndarray]:
    raw = generate_burgers_raw_data(
        key,
        num_operators=num_prompts,
        pairs_per_operator=pairs_per_prompt,
        n_grid=n_grid,
        n_modes=n_modes,
        c_range=(float(c_value), float(c_value) + 1e-8),
        T_final=T_final,
        fourier_decay=fourier_decay,
        bias_scale=bias_scale,
        initial_amp=initial_amp,
        n_solver_steps=n_solver_steps,
    )

    if fixed_query_u0 is not None:
        if normalize_each_function:
            raise NotImplementedError("fixed_query_u0 replacement assumes normalize_each_function=False.")
        uq = jnp.asarray(fixed_query_u0).reshape(n_grid)
        uq_batch = jnp.broadcast_to(uq[None, :], (num_prompts, n_grid))
        c_batch = raw.c
        uT_batch = solve_burgers_batch_jit(uq_batch, c_batch, T_final, nsteps=n_solver_steps)
        raw = raw._replace(
            cond_v=raw.cond_v.at[:, num_demos, :, 0].set(uq_batch),
            qoi_v=raw.qoi_v.at[:, num_demos, :, 0].set(uT_batch),
        )

    data, label = build_icon_data(raw, demo_num=num_demos)
    return raw, data, label


def make_burgers_base_state(kind: str, n_grid: int = 100) -> jnp.ndarray:
    x = jnp.linspace(0.0, 1.0, n_grid, endpoint=False, dtype=jnp.float32)
    if kind == "smooth":
        u = jnp.sin(2.0 * jnp.pi * x)
    elif kind == "steep":
        u = jnp.tanh(jnp.sin(2.0 * jnp.pi * x) / 0.22)
    elif kind == "near_shock":
        u = jnp.tanh(jnp.sin(2.0 * jnp.pi * x) / 0.08)
    else:
        raise ValueError(kind)
    u = u - jnp.mean(u)
    u = initial_amp * u / (jnp.max(jnp.abs(u)) + 1e-12)
    return u[None, :, None].astype(jnp.float32)


# =============================================================================
# 4. ICON model
# =============================================================================

class MLP(nn.Module):
    hidden_dim: int
    out_dim: int
    depth: int

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for _ in range(self.depth):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.gelu(x)
        return nn.Dense(features=self.out_dim)(x)


class SelfAttnTransformer(nn.Module):
    n_layers: int
    n_heads: int
    head_dim: int
    model_dim: int
    widening_factor: int

    @nn.compact
    def __call__(self, inputs, mask=None):
        x = inputs
        for _ in range(self.n_layers):
            attn_block = MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.head_dim * self.n_heads,
                out_features=self.model_dim,
            )
            x = attn_block(inputs_q=x, inputs_k=x, inputs_v=x, mask=mask) + x
            x = nn.LayerNorm()(x)
            dense_block = MLP(self.model_dim * self.widening_factor, self.model_dim, 1)
            x = dense_block(x) + x
            x = nn.LayerNorm()(x)
        return x


def build_matrices_from_data_shape(data_shape, mode, shot_num_min, return_shape_list=False):
    demo_num_local = data_shape.demo_cond_k[0]
    demo_cond_len = data_shape.demo_cond_k[1]
    demo_qoi_len = data_shape.demo_qoi_k[1]
    quest_cond_len = data_shape.quest_cond_k[1]
    quest_qoi_len = data_shape.quest_qoi_k[1]

    cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list = mu.build_bool_sequence(
        demo_num_local, mode, shot_num_min
    )
    cond_len_raw = [demo_cond_len] * demo_num_local + [quest_cond_len]
    qoi_kv_len_raw = [demo_qoi_len] * demo_num_local + [quest_qoi_len]
    qoi_k_len_raw = [demo_qoi_len] * demo_num_local + [quest_qoi_len]

    cond_len_list = [a * b for a, b in zip(cond_bool_list, cond_len_raw)]
    qoi_kv_len_list = [a * b for a, b in zip(qoi_kv_bool_list, qoi_kv_len_raw)]
    qoi_k_len_list = [a * b for a, b in zip(qoi_k_bool_list, qoi_k_len_raw)]

    basic_mask = mu.build_basic_mask(cond_len_list, qoi_kv_len_list, qoi_k_len_list)
    index_pos = mu.build_index_integer(cond_len_list, qoi_kv_len_list, qoi_k_len_list)
    out_mask = mu.build_out_mask(
        cond_len_list,
        qoi_kv_len_list,
        qoi_k_len_list,
        num_range=(shot_num_min, demo_num_local + 1),
    )
    if return_shape_list:
        return basic_mask, index_pos, out_mask, cond_len_list, qoi_kv_len_list, qoi_k_len_list
    return basic_mask, index_pos, out_mask


class ICON_LM(nn.Module):
    config: dict
    basic_mask: jnp.ndarray
    index_pos: jnp.ndarray
    out_mask: jnp.ndarray

    def setup(self):
        d = self.config["transformer"]["model_dim"]
        self.pre_projection = nn.Dense(d, name="pre_projection")
        self.func_pos_embedding = nn.Embed(self.config["demo_max_num"] * 3, d, name="func_pos_embedding")
        self.transformer = SelfAttnTransformer(**self.config["transformer"], name="transformer")
        self.post_projection = nn.Dense(self.config["out_dim"], name="post_projection")

    def basic_forward(self, data, mode, index_pos, basic_mask):
        demo_num_local = len(data.demo_cond_k)
        cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list = mu.build_bool_sequence(
            demo_num_local, mode=mode, shot_num_min=0
        )
        sequence = mu.build_data_sequence(data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list)
        token_mask = mu.build_data_mask(data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list)

        sequence = self.pre_projection(sequence)
        sequence = sequence + self.func_pos_embedding(index_pos)

        token_mask = einshape("i->ji", token_mask, j=sequence.shape[0])
        attn_mask = token_mask * basic_mask
        attn_mask = einshape("ij->mij", attn_mask, m=self.config["transformer"]["n_heads"])

        sequence = self.transformer(sequence, mask=attn_mask)
        return self.post_projection(sequence)

    def __call__(self, data):
        seq = self.basic_forward(data, "train", self.index_pos, self.basic_mask)
        return seq[self.out_mask]

    def predict(self, data):
        data_shape = tree.tree_map(lambda x: x.shape, data)
        basic_mask, index_pos, _ = build_matrices_from_data_shape(data_shape, mode="test", shot_num_min=0)
        seq = self.basic_forward(data, "test", index_pos, basic_mask)
        return seq[-data.quest_qoi_mask.shape[-1]:, :]


def find_latest_checkpoint(ckpt_dir: str) -> str:
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "params_step_*.msgpack")))
    if not paths:
        raise FileNotFoundError(f"No params_step_*.msgpack found in {ckpt_dir}")
    return paths[-1]


def build_model_and_template_params():
    key = jax.random.PRNGKey(seed)
    key_data, key_init = jax.random.split(key)
    raw, data, _ = make_burgers_prompts_at_c(
        key_data,
        c_value=0.02,
        num_prompts=1,
        pairs_per_prompt=num_pairs,
        fixed_query_u0=None,
    )
    sample_data = tree.tree_map(lambda x: x[0], data)
    sample_shape = tree.tree_map(lambda x: x.shape, sample_data)
    basic_mask, index_pos, out_mask = build_matrices_from_data_shape(sample_shape, "train", 0)
    model = ICON_LM(model_config, basic_mask=basic_mask, index_pos=index_pos, out_mask=out_mask)
    params_template = model.init({"params": key_init}, sample_data)
    return model, params_template


icon_model, params_template = build_model_and_template_params()


def load_params_from_checkpoint(path_or_dir: str | None, default_dir: str) -> tuple[dict, str]:
    if path_or_dir is None:
        path = find_latest_checkpoint(default_dir)
    elif os.path.isdir(path_or_dir):
        path = find_latest_checkpoint(path_or_dir)
    else:
        path = path_or_dir
    with open(path, "rb") as f:
        params = serialization.from_bytes(params_template, f.read())
    print("Loaded checkpoint:", path)
    return params, path


def find_optional_clean_checkpoint() -> str | None:
    """Return a clean checkpoint path if one is explicitly set or discoverable."""
    if not LOAD_CLEAN_COMPARISON:
        return None
    if CLEAN_CKPT_PATH is not None:
        return CLEAN_CKPT_PATH

    for directory in CLEAN_CKPT_DIR_CANDIDATES:
        if not os.path.isdir(directory):
            continue
        paths = sorted(glob.glob(os.path.join(directory, "params_step_*.msgpack")))
        if paths:
            return paths[-1]
    return None


params_wrong, loaded_wrong_ckpt = load_params_from_checkpoint(WRONG_CKPT_PATH, WRONG_CKPT_DIR)
clean_ckpt_resolved = find_optional_clean_checkpoint()
if clean_ckpt_resolved is not None:
    params_clean, loaded_clean_ckpt = load_params_from_checkpoint(clean_ckpt_resolved, os.path.dirname(clean_ckpt_resolved))
else:
    params_clean, loaded_clean_ckpt = None, None
    if LOAD_CLEAN_COMPARISON:
        print("No clean checkpoint found. Set CLEAN_CKPT_PATH or CLEAN_CKPT_DIR_CANDIDATES if you want clean-trained curves.")


def predict_one(params, data_one):
    return icon_model.apply(params, data_one, method=ICON_LM.predict)


predict_batch = jax.jit(jax.vmap(predict_one, in_axes=(None, 0)))


# =============================================================================
# 5. Jacobian and spectral utilities
# =============================================================================

def true_burgers_map_u0(u0_flat: jnp.ndarray, c_value: float | jnp.ndarray) -> jnp.ndarray:
    u0 = jnp.asarray(u0_flat).reshape(1, n_grid)
    c_arr = jnp.asarray(c_value, dtype=jnp.float32).reshape(1)
    uT = solve_burgers_batch_jit(u0, c_arr, T_final, nsteps=n_solver_steps)
    return uT.reshape(-1)


@partial(jax.jit, static_argnames=("max_mode",))
def icon_mode_gain_by_jvp_fast(params, data_one, max_mode=40):
    u0_flat = data_one.quest_cond_v.reshape(-1)
    n = u0_flat.shape[0]

    def f_query(u_flat):
        data_new = data_one._replace(quest_cond_v=u_flat.reshape(data_one.quest_cond_v.shape))
        return predict_one(params, data_new).reshape(-1)

    x = jnp.arange(n, dtype=jnp.float32) / n
    ks = jnp.arange(1, max_mode + 1, dtype=jnp.float32)[:, None]
    cos_basis = jnp.sqrt(2.0) * jnp.cos(2.0 * jnp.pi * ks * x[None, :])
    sin_basis = jnp.sqrt(2.0) * jnp.sin(2.0 * jnp.pi * ks * x[None, :])
    tangents = jnp.stack([cos_basis, sin_basis], axis=1).reshape(2 * max_mode, n)

    def one_jvp(v):
        _, y = jax.jvp(f_query, (u0_flat,), (v,))
        return y

    responses = jax.vmap(one_jvp)(tangents)
    gains = jnp.linalg.norm(responses, axis=1) / (jnp.linalg.norm(tangents, axis=1) + 1e-12)
    gains = gains.reshape(max_mode, 2)
    k_values = jnp.arange(1, max_mode + 1, dtype=jnp.float32)
    return jnp.stack([k_values, gains[:, 0], gains[:, 1]], axis=1)


@partial(jax.jit, static_argnames=("max_mode",))
def true_mode_gain_by_jvp_u0_fast(u0_flat, c_value, max_mode=40):
    u0_flat = jnp.asarray(u0_flat).reshape(-1)
    n = u0_flat.shape[0]
    f = lambda u: true_burgers_map_u0(u, c_value)

    x = jnp.arange(n, dtype=jnp.float32) / n
    ks = jnp.arange(1, max_mode + 1, dtype=jnp.float32)[:, None]
    cos_basis = jnp.sqrt(2.0) * jnp.cos(2.0 * jnp.pi * ks * x[None, :])
    sin_basis = jnp.sqrt(2.0) * jnp.sin(2.0 * jnp.pi * ks * x[None, :])
    tangents = jnp.stack([cos_basis, sin_basis], axis=1).reshape(2 * max_mode, n)

    def one_jvp(v):
        _, y = jax.jvp(f, (u0_flat,), (v,))
        return y

    responses = jax.vmap(one_jvp)(tangents)
    gains = jnp.linalg.norm(responses, axis=1) / (jnp.linalg.norm(tangents, axis=1) + 1e-12)
    gains = gains.reshape(max_mode, 2)
    k_values = jnp.arange(1, max_mode + 1, dtype=jnp.float32)
    return jnp.stack([k_values, gains[:, 0], gains[:, 1]], axis=1)


def icon_query_map(params, data_one):
    def f_query(u_flat):
        data_new = data_one._replace(quest_cond_v=u_flat.reshape(data_one.quest_cond_v.shape))
        return predict_one(params, data_new).reshape(-1)
    return f_query


def true_jacobian_l2_u0(u0_flat, c_value):
    f = lambda u: true_burgers_map_u0(u, c_value)
    J = jax.jacrev(f)(jnp.asarray(u0_flat).reshape(-1))
    n_out, n_in = J.shape
    return jnp.sqrt((1.0 / n_out) / (1.0 / n_in)) * J


def icon_jacobian_l2(params, data_one):
    u0_flat = data_one.quest_cond_v.reshape(-1)
    J = jax.jacrev(icon_query_map(params, data_one))(u0_flat)
    n_out, n_in = J.shape
    return jnp.sqrt((1.0 / n_out) / (1.0 / n_in)) * J


def fourier_feature_basis(n: int, max_pair_mode: int):
    x = jnp.arange(n, dtype=jnp.float32) / n
    cols = []
    for k in range(1, max_pair_mode + 1):
        c_col = jnp.sqrt(2.0) * jnp.cos(2.0 * jnp.pi * k * x)
        s_col = jnp.sqrt(2.0) * jnp.sin(2.0 * jnp.pi * k * x)
        c_col = c_col / (jnp.linalg.norm(c_col) + 1e-12)
        s_col = s_col / (jnp.linalg.norm(s_col) + 1e-12)
        cols.extend([c_col, s_col])
    return jnp.stack(cols, axis=1)


def fourier_feature_matrix_from_jacobian(J_l2, max_pair_mode: int):
    n_out, n_in = J_l2.shape
    Phi = fourier_feature_basis(n_in, max_pair_mode)
    Psi = fourier_feature_basis(n_out, max_pair_mode)
    return Psi.T @ J_l2 @ Phi


def cumulative_relative_spectral_error(icon_cs: np.ndarray, true_cs: np.ndarray, eps: float = 1e-12):
    """icon_cs and true_cs have shape [num_c, num_modes, 2]."""
    icon_cs = np.asarray(icon_cs)
    true_cs = np.asarray(true_cs)
    diff2 = (icon_cs - true_cs) ** 2
    true2 = true_cs ** 2
    cum_cos = np.sqrt(np.cumsum(diff2[:, :, 0], axis=1)) / (np.sqrt(np.cumsum(true2[:, :, 0], axis=1)) + eps)
    cum_sin = np.sqrt(np.cumsum(diff2[:, :, 1], axis=1)) / (np.sqrt(np.cumsum(true2[:, :, 1], axis=1)) + eps)
    cum_all = np.sqrt(np.cumsum(np.sum(diff2, axis=2), axis=1)) / (np.sqrt(np.cumsum(np.sum(true2, axis=2), axis=1)) + eps)
    return cum_cos, cum_sin, cum_all


def rel_l2_per_prompt(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    target = np.asarray(label)[:, 0]
    pred = np.asarray(pred)
    diff = pred - target
    diff_flat = diff.reshape(diff.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    return np.linalg.norm(diff_flat, axis=1) / (np.linalg.norm(target_flat, axis=1) + 1e-12)


# =============================================================================
# 6. Compute diagnostics
# =============================================================================

def compute_prediction_error_over_c(params, c_values, prompts_per_c=10, key_seed=31415):
    rows = []
    for i, c_value in enumerate(c_values):
        _, data_i, label_i = make_burgers_prompts_at_c(
            jax.random.PRNGKey(key_seed + i),
            c_value=float(c_value),
            num_prompts=prompts_per_c,
            pairs_per_prompt=num_pairs,
            fixed_query_u0=None,
        )
        pred_i = predict_batch(params, data_i)
        err_i = rel_l2_per_prompt(np.asarray(pred_i), np.asarray(label_i))
        rows.append({
            "c": float(c_value),
            "err_mean": float(np.mean(err_i)),
            "err_std": float(np.std(err_i)),
            "err_all": err_i,
        })
    return {
        "c_values": np.asarray([r["c"] for r in rows]),
        "err_mean": np.asarray([r["err_mean"] for r in rows]),
        "err_std": np.asarray([r["err_std"] for r in rows]),
        "err_all": [r["err_all"] for r in rows],
    }


def compute_query_spectrum_over_c(params, fixed_uq, c_values, contexts_per_c=5, max_mode=40, key_seed=2028):
    icon_mean, icon_std, true_list = [], [], []
    for i, c_value in enumerate(c_values):
        _, data_i, _ = make_burgers_prompts_at_c(
            jax.random.PRNGKey(key_seed + i),
            c_value=float(c_value),
            num_prompts=contexts_per_c,
            pairs_per_prompt=num_pairs,
            fixed_query_u0=fixed_uq,
        )
        icon_runs = []
        for r in range(contexts_per_c):
            data_one = tree.tree_map(lambda x: x[r], data_i)
            g_icon = icon_mode_gain_by_jvp_fast(params, data_one, max_mode=max_mode)
            icon_runs.append(np.asarray(g_icon)[:, 1:3])
        icon_runs = np.stack(icon_runs, axis=0)
        icon_mean.append(icon_runs.mean(axis=0))
        icon_std.append(icon_runs.std(axis=0))
        g_true = true_mode_gain_by_jvp_u0_fast(jnp.asarray(fixed_uq).reshape(-1), float(c_value), max_mode=max_mode)
        true_list.append(np.asarray(g_true)[:, 1:3])
    icon_mean = np.stack(icon_mean, axis=0)
    icon_std = np.stack(icon_std, axis=0)
    true_arr = np.stack(true_list, axis=0)
    cum_cos, cum_sin, cum_all = cumulative_relative_spectral_error(icon_mean, true_arr)
    return {
        "k": np.arange(1, max_mode + 1),
        "c_values": np.asarray(c_values, dtype=float),
        "icon_mean": icon_mean,
        "icon_std": icon_std,
        "true": true_arr,
        "cum_rel_err_cos": cum_cos,
        "cum_rel_err_sin": cum_sin,
        "cum_rel_err_all": cum_all,
    }


def compute_representative_prediction(params, c_value=0.02, key_seed=4242):
    raw, data, label = make_burgers_prompts_at_c(
        jax.random.PRNGKey(key_seed),
        c_value=float(c_value),
        num_prompts=1,
        pairs_per_prompt=num_pairs,
        fixed_query_u0=None,
    )
    pred = predict_batch(params, data)
    rel = float(rel_l2_per_prompt(np.asarray(pred), np.asarray(label))[0])
    return {
        "x": np.asarray(data.quest_qoi_k[0, 0, :, 0]),
        "u0": np.asarray(data.quest_cond_v[0, 0, :, 0]),
        "true": np.asarray(label[0, 0, :, 0]),
        "pred": np.asarray(pred[0, :, 0]),
        "c": float(raw.c[0]),
        "rel_l2": rel,
    }


def compute_feature_map(params, base_kind="near_shock", c_value=0.02, max_pair_mode=14, key_seed=7070):
    uq = make_burgers_base_state(base_kind, n_grid=n_grid)
    _, data, _ = make_burgers_prompts_at_c(
        jax.random.PRNGKey(key_seed),
        c_value=float(c_value),
        num_prompts=1,
        pairs_per_prompt=num_pairs,
        fixed_query_u0=uq,
    )
    data_one = tree.tree_map(lambda x: x[0], data)
    u0_flat = data_one.quest_cond_v.reshape(-1)
    J_true = true_jacobian_l2_u0(u0_flat, float(c_value))
    J_icon = icon_jacobian_l2(params, data_one)
    A_true = np.asarray(fourier_feature_matrix_from_jacobian(J_true, max_pair_mode))
    A_icon = np.asarray(fourier_feature_matrix_from_jacobian(J_icon, max_pair_mode))
    A_err = A_icon - A_true
    rel_frob_err = float(np.linalg.norm(A_err) / (np.linalg.norm(A_true) + 1e-12))
    return {
        "base_kind": base_kind,
        "c": float(c_value),
        "max_pair_mode": int(max_pair_mode),
        "A_true": A_true,
        "A_icon": A_icon,
        "A_err": A_err,
        "abs_true": np.abs(A_true),
        "abs_icon": np.abs(A_icon),
        "rel_err_map": np.abs(A_err) / (np.linalg.norm(A_true) + 1e-12),
        "rel_frob_err": rel_frob_err,
    }


def compute_all_diagnostics():
    print("Computing clean-evaluation prediction error for corrupted-trained model...")
    pred_wrong = compute_prediction_error_over_c(
        params_wrong,
        paper_c_values,
        prompts_per_c=paper_prediction_prompts_per_c,
    )
    pred_clean = None
    if params_clean is not None:
        print("Computing prediction error for clean model comparison...")
        pred_clean = compute_prediction_error_over_c(
            params_clean,
            paper_c_values,
            prompts_per_c=paper_prediction_prompts_per_c,
        )

    print("Generating fixed query for spectrum diagnostics...")
    _, data0, _ = make_burgers_prompts_at_c(
        jax.random.PRNGKey(1234),
        c_value=paper_representative_c,
        num_prompts=1,
        pairs_per_prompt=num_pairs,
        fixed_query_u0=None,
    )
    data_one0 = tree.tree_map(lambda x: x[0], data0)
    fixed_uq = np.asarray(data_one0.quest_cond_v[0, :, 0])

    print("Computing query spectrum for corrupted-trained model...")
    query_wrong = compute_query_spectrum_over_c(
        params_wrong,
        fixed_uq=fixed_uq,
        c_values=paper_c_values,
        contexts_per_c=paper_query_contexts_per_c,
        max_mode=paper_query_max_mode,
    )
    # Clean-trained curve is used only in the prediction-error plot.
    # Do not compute or overlay clean spectra in the cumulative spectral-error figure.
    query_clean = None

    print("Computing representative prediction...")
    representative_wrong = compute_representative_prediction(params_wrong, c_value=paper_representative_c)

    print("Computing Fourier feature map for corrupted-trained model...")
    feature_wrong = compute_feature_map(
        params_wrong,
        base_kind=paper_feature_base_kind,
        c_value=paper_feature_c_value,
        max_pair_mode=paper_feature_max_pair_mode,
    )
    # Feature maps are used as a negative-control diagnostic for the corrupted model only.
    feature_clean = None

    return {
        "prediction_wrong": pred_wrong,
        "prediction_clean": pred_clean,
        "query_wrong": query_wrong,
        "query_clean": query_clean,
        "representative_wrong": representative_wrong,
        "feature_wrong": feature_wrong,
        "feature_clean": feature_clean,
        "fixed_uq": fixed_uq,
        "loaded_wrong_ckpt": loaded_wrong_ckpt,
        "loaded_clean_ckpt": loaded_clean_ckpt,
    }


# =============================================================================
# 7. Plotting
# =============================================================================

def save_figure(fig, filename: str):
    path = os.path.join(fig_dir, filename)
    if save_figs:
        fig.savefig(path, dpi=260, bbox_inches="tight")
        if filename.endswith("_regime_style.png"):
            fig.savefig(path[:-4] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)


def plot_representative_prediction(results):
    r = results["representative_wrong"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
    axes[0].plot(r["x"], r["u0"], "k-")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$u_0(x)$")
    axes[0].grid(True)
    axes[0].text(0.03, 0.92, rf"$c={r['c']:.3g}$", transform=axes[0].transAxes,
                 ha="left", va="top", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    axes[1].plot(r["x"], r["true"], "k-", label="true")
    axes[1].plot(r["x"], r["pred"], "r--", label="corrupted-trained ICON")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$u(T,x)$")
    axes[1].grid(True)
    axes[1].legend(frameon=True)
    axes[1].text(0.03, 0.92, rf"Rel-$L^2={r['rel_l2']:.2e}$", transform=axes[1].transAxes,
                 ha="left", va="top", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    save_figure(fig, "burgers_wrong_representative_prediction.png")


def plot_prediction_error_over_c(results):
    w = results["prediction_wrong"]
    c = w["c_values"]
    fig, ax = plt.subplots(figsize=(6.4, 4.1), constrained_layout=True)
    if results["prediction_clean"] is not None:
        cl = results["prediction_clean"]
        ax.plot(cl["c_values"], cl["err_mean"], "o-", label="clean-trained")
    # ax.errorbar(c, w["err_mean"], yerr=w["err_std"], fmt="s--", capsize=3, label="corrupted-trained")
    ax.plot(c, w["err_mean"], "s-", label="corrupted-trained")
    ax.axvspan(c_range[0], c_range[1], alpha=0.18, label="training range")
    ax.set_yscale("log")
    ax.set_xlabel(r"hidden viscosity $c$")
    ax.set_ylabel(r"relative $L^2$ prediction error")
    ax.grid(True, which="both")
    for regime_ax in np.asarray(ax, dtype=object).ravel():
        style_parameter_sweep(regime_ax, 'burgers')
    ax.legend(frameon=True)
    add_regime_key(plt.gcf())
    save_figure(fig, comparison_filename("burgers_wrong_relative_prediction_error_over_hidden_viscosity.png"))


def plot_query_spectrum_wrong(results):
    """
    Corrupted-trained query spectrum over c.

    Top row: corrupted-trained ICON gains.
    Bottom row: accumulated relative errors of the corrupted-trained model only.
    The clean-trained baseline is intentionally not overlaid here; it is only
    shown in the relative L2 prediction-error figure.
    """
    q = results["query_wrong"]

    k = q["k"]
    c_values = q["c_values"]
    icon = q["icon_mean"]
    cum_cos = q["cum_rel_err_cos"]
    cum_sin = q["cum_rel_err_sin"]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True, constrained_layout=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, c_val in enumerate(c_values):
        color = colors[i % len(colors)]

        axes[0, 0].plot(k, icon[i, :, 0], "o-", color=color, label=f"{c_val:g}")
        axes[0, 1].plot(k, icon[i, :, 1], "o-", color=color, label=f"{c_val:g}")
        axes[1, 0].plot(k, cum_cos[i], "o-", color=color)
        axes[1, 1].plot(k, cum_sin[i], "o-", color=color)

    for ax in axes.ravel():
        ax.axvline(n_modes, linestyle="--", color="k", linewidth=1.2)
        ax.grid(True, which="both")

    axes[0, 0].set_ylabel("ICON Jacobian gain")
    axes[1, 0].set_ylabel("cumulative relative error")
    axes[1, 0].set_xlabel(r"Fourier mode $K$")
    axes[1, 1].set_xlabel(r"Fourier mode $K$")

    axes[0, 0].text(0.04, 0.90, "cos", transform=axes[0, 0].transAxes,
                    ha="left", va="top", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    axes[0, 1].text(0.04, 0.90, "sin", transform=axes[0, 1].transAxes,
                    ha="left", va="top", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    axes[1, 0].text(0.04, 0.90, "cos", transform=axes[1, 0].transAxes,
                    ha="left", va="top", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    axes[1, 1].text(0.04, 0.90, "sin", transform=axes[1, 1].transAxes,
                    ha="left", va="top", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    from plotting.regimes import keep_training_parameter_curves
    visible_c_values = keep_training_parameter_curves(axes, c_values, "burgers")
    style_parameter_curves(axes, visible_c_values, "burgers")
    axes[0, 1].legend(title=r"training-range viscosity $c$", ncol=2, frameon=True)
    save_figure(fig, comparison_filename("burgers_wrong_query_spectrum_with_cumulative_relative_error.png"))

def plot_clean_vs_wrong_query_at_representative_c(results, c_value=0.02):
    if results["query_clean"] is None:
        return
    q_w = results["query_wrong"]
    q_c = results["query_clean"]
    c_values = q_w["c_values"]
    idx = int(np.argmin(np.abs(c_values - c_value)))
    k = q_w["k"]
    true = q_w["true"][idx]
    wrong = q_w["icon_mean"][idx]
    clean = q_c["icon_mean"][idx]
    clean_cum_cos, clean_cum_sin, _ = cumulative_relative_spectral_error(clean[None, :, :], true[None, :, :])
    wrong_cum_cos, wrong_cum_sin, _ = cumulative_relative_spectral_error(wrong[None, :, :], true[None, :, :])

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.8), sharex=True, constrained_layout=True)
    labels = ["cos", "sin"]
    for j in range(2):
        axes[0, j].plot(k, true[:, j], "k-", label="true tangent")
        axes[0, j].plot(k, clean[:, j], "o-", label="clean-trained")
        axes[0, j].plot(k, wrong[:, j], "s--", label="corrupted-trained")
        axes[0, j].axvline(n_modes, linestyle="--", color="k", linewidth=1.2)
        axes[0, j].grid(True)
        axes[0, j].text(0.04, 0.90, labels[j], transform=axes[0, j].transAxes,
                        ha="left", va="top", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    axes[1, 0].plot(k, clean_cum_cos[0], "o-", label="clean-trained")
    axes[1, 0].plot(k, wrong_cum_cos[0], "s--", label="corrupted-trained")
    axes[1, 1].plot(k, clean_cum_sin[0], "o-", label="clean-trained")
    axes[1, 1].plot(k, wrong_cum_sin[0], "s--", label="corrupted-trained")
    for ax in axes[1, :]:
        ax.axvline(n_modes, linestyle="--", color="k", linewidth=1.2)
        ax.grid(True)
        ax.set_xlabel(r"Fourier mode $K$")
    axes[0, 0].set_ylabel("Jacobian gain")
    axes[1, 0].set_ylabel("cumulative relative error")
    axes[0, 1].legend(frameon=True)
    axes[1, 0].legend(frameon=True)
    save_figure(fig, "burgers_wrong_vs_clean_query_spectrum_representative_c.png")


def split_cos_sin_blocks(A: np.ndarray) -> dict[str, np.ndarray]:
    """
    Split a paired real-Fourier matrix into cos/sin blocks.

    The Fourier basis is ordered as
        cos(1), sin(1), cos(2), sin(2), ...
    so the blocks are:
        cc: output cos rows, input cos columns
        cs: output cos rows, input sin columns
        sc: output sin rows, input cos columns
        ss: output sin rows, input sin columns.
    """
    return {
        "cc": A[0::2, 0::2],
        "cs": A[0::2, 1::2],
        "sc": A[1::2, 0::2],
        "ss": A[1::2, 1::2],
    }


def block_display_name(block_name: str) -> str:
    names = {
        "cc": r"cos $\to$ cos",
        "cs": r"sin $\to$ cos",
        "sc": r"cos $\to$ sin",
        "ss": r"sin $\to$ sin",
    }
    return names[block_name]


def plot_feature_map_wrong(results):
    """
    Plot Fourier feature maps with cos/sin blocks separated.

    This replaces the previous interleaved 2m x 2m feature-map figure.  Separating
    the four blocks makes it clear whether the corrupted-trained model has the
    correct cos/sin coupling structure.
    """
    fw = results["feature_wrong"]
    fc = results["feature_clean"]

    true_blocks = split_cos_sin_blocks(fw["A_true"])
    wrong_blocks = split_cos_sin_blocks(fw["A_icon"])
    clean_blocks = split_cos_sin_blocks(fc["A_icon"]) if fc is not None else None

    # Error maps are normalized by ||A_true||_F, exactly as in fw["rel_err_map"].
    norm_true = np.linalg.norm(fw["A_true"]) + 1e-12
    wrong_err_blocks = {name: np.abs(wrong_blocks[name] - true_blocks[name]) / norm_true
                        for name in true_blocks}

    clean_err_blocks = None
    if clean_blocks is not None:
        clean_err_blocks = {name: np.abs(clean_blocks[name] - true_blocks[name]) / norm_true
                            for name in true_blocks}

    # Common color scales across all cos/sin blocks.
    feature_vmax = max(
        max(float(np.max(np.abs(M))) for M in true_blocks.values()),
        max(float(np.max(np.abs(M))) for M in wrong_blocks.values()),
        1e-12,
    )
    if clean_blocks is not None:
        feature_vmax = max(feature_vmax, max(float(np.max(np.abs(M))) for M in clean_blocks.values()))

    err_vmax = max(float(np.max(M)) for M in wrong_err_blocks.values()) + 1e-12
    if clean_err_blocks is not None:
        err_vmax = max(err_vmax, max(float(np.max(M)) for M in clean_err_blocks.values()))

    if clean_blocks is not None:
        col_titles = ["true", "clean ICON", "corrupted ICON", "clean error", "corrupted error"]
        ncols = 5
        fig_width = 15.0
    else:
        col_titles = ["true", "corrupted ICON", "corrupted error"]
        ncols = 3
        fig_width = 10.0

    block_order = ["cc", "cs", "sc", "ss"]
    fig, axes = plt.subplots(
        len(block_order),
        ncols,
        figsize=(fig_width, 9.6),
        constrained_layout=True,
    )

    im_feature = None
    im_error = None

    for r, block_name in enumerate(block_order):
        row_items = []
        if clean_blocks is not None:
            row_items = [
                (np.abs(true_blocks[block_name]), "feature"),
                (np.abs(clean_blocks[block_name]), "feature"),
                (np.abs(wrong_blocks[block_name]), "feature"),
                (clean_err_blocks[block_name], "error"),
                (wrong_err_blocks[block_name], "error"),
            ]
        else:
            row_items = [
                (np.abs(true_blocks[block_name]), "feature"),
                (np.abs(wrong_blocks[block_name]), "feature"),
                (wrong_err_blocks[block_name], "error"),
            ]

        for cidx, (M, typ) in enumerate(row_items):
            ax = axes[r, cidx]
            if typ == "feature":
                im_feature = ax.imshow(
                    M, origin="lower", aspect="equal", interpolation="nearest",
                    cmap=PAPER_FEATURE_CMAP, vmin=0.0, vmax=feature_vmax,
                )
            else:
                im_error = ax.imshow(
                    M, origin="lower", aspect="equal", interpolation="nearest",
                    cmap=PAPER_ERROR_CMAP, vmin=0.0, vmax=err_vmax,
                )

            if r == 0:
                ax.set_title(col_titles[cidx], fontsize=11)
            if cidx == 0:
                ax.set_ylabel(block_display_name(block_name), fontsize=11)
            else:
                ax.set_ylabel("")
            if r == len(block_order) - 1:
                ax.set_xlabel("input mode")
            else:
                ax.set_xlabel("")
            ax.tick_params(labelsize=9)

            # Put only the global Rel-F number on the corrupted-error panels.
            if typ == "error" and col_titles[cidx] == "corrupted error":
                ax.text(
                    0.04, 0.94,
                    rf"Rel-F={fw['rel_frob_err']:.2e}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.5,
                    bbox=dict(facecolor="white", alpha=0.78, edgecolor="none"),
                )

    # Shared colorbars for all feature and error maps.
    if im_feature is not None:
        feature_axes = [
            axes[r, cidx]
            for r in range(len(block_order))
            for cidx, title in enumerate(col_titles)
            if "error" not in title
        ]
        cbar = fig.colorbar(im_feature, ax=feature_axes, shrink=0.82)
        cbar.set_label(r"$|A_{\rm block}|$")

    if im_error is not None:
        error_axes = [
            axes[r, cidx]
            for r in range(len(block_order))
            for cidx, title in enumerate(col_titles)
            if "error" in title
        ]
        cbar_err = fig.colorbar(im_error, ax=error_axes, shrink=0.82)
        cbar_err.set_label(r"$|A_{\rm ICON}-A_{\rm true}|/\|A_{\rm true}\|_F$")

    save_figure(fig, "burgers_wrong_fourier_feature_blocks_cos_sin_true_icon_error.png")

    # Also keep a compact three-panel interleaved map for backward compatibility,
    # but make the labels shorter and cleaner.
    panels = [
        (fw["abs_true"], "true", None),
        (fw["abs_icon"], "corrupted ICON", None),
        (fw["rel_err_map"], f"Rel-F={fw['rel_frob_err']:.2e}", "error"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(12.0, 3.8), constrained_layout=True)

    feature_vmax_combined = max(float(np.max(fw["abs_true"])), float(np.max(fw["abs_icon"])), 1e-12)
    err_vmax_combined = max(float(np.max(fw["rel_err_map"])), 1e-12)
    im_feature = None
    im_error = None
    for ax, (M, label, typ) in zip(axes, panels):
        if typ == "error":
            im_error = ax.imshow(M, origin="lower", aspect="equal", interpolation="nearest",
                                 cmap=PAPER_ERROR_CMAP, vmin=0.0, vmax=err_vmax_combined)
        else:
            im_feature = ax.imshow(M, origin="lower", aspect="equal", interpolation="nearest",
                                   cmap=PAPER_FEATURE_CMAP, vmin=0.0, vmax=feature_vmax_combined)
        ax.set_xlabel("input Fourier index")
        ax.set_ylabel("output Fourier index")
        ax.text(0.04, 0.94, label, transform=ax.transAxes,
                ha="left", va="top", bbox=dict(facecolor="white", alpha=0.78, edgecolor="none"))

    if im_feature is not None:
        cbar = fig.colorbar(im_feature, ax=[axes[0], axes[1]], shrink=0.82)
        cbar.set_label(r"$|\Psi^\top J\Phi|$")
    if im_error is not None:
        cbar_err = fig.colorbar(im_error, ax=[axes[-1]], shrink=0.82)
        cbar_err.set_label(r"$|A_{\rm ICON}-A_{\rm true}|/\|A_{\rm true}\|_F$")

    save_figure(fig, "burgers_wrong_fourier_feature_map_true_icon_error.png")

def save_numeric_results(results):
    q = results["query_wrong"]
    fw = results["feature_wrong"]

    save_dict = {
        "wrong_ckpt": np.asarray(results["loaded_wrong_ckpt"]),
        "clean_ckpt": np.asarray("" if results["loaded_clean_ckpt"] is None else results["loaded_clean_ckpt"]),
        "pred_c_values": results["prediction_wrong"]["c_values"],
        "pred_wrong_mean": results["prediction_wrong"]["err_mean"],
        "pred_wrong_std": results["prediction_wrong"]["err_std"],
        "query_k": q["k"],
        "query_c_values": q["c_values"],
        "query_wrong_icon_mean": q["icon_mean"],
        "query_wrong_icon_std": q["icon_std"],
        "query_true": q["true"],
        "query_wrong_cum_rel_err_cos": q["cum_rel_err_cos"],
        "query_wrong_cum_rel_err_sin": q["cum_rel_err_sin"],
        "query_wrong_cum_rel_err_all": q["cum_rel_err_all"],
        "feature_base_kind": np.asarray(fw["base_kind"]),
        "feature_c": np.asarray(fw["c"]),
        "feature_A_true": fw["A_true"],
        "feature_A_icon_wrong": fw["A_icon"],
        "feature_rel_err_map_wrong": fw["rel_err_map"],
        "feature_rel_frob_err_wrong": np.asarray(fw["rel_frob_err"]),
    }

    if results["prediction_clean"] is not None:
        save_dict.update({
            "pred_clean_mean": results["prediction_clean"]["err_mean"],
            "pred_clean_std": results["prediction_clean"]["err_std"],
        })

    if results["query_clean"] is not None:
        qc = results["query_clean"]
        save_dict.update({
            "query_clean_icon_mean": qc["icon_mean"],
            "query_clean_icon_std": qc["icon_std"],
            "query_clean_cum_rel_err_cos": qc["cum_rel_err_cos"],
            "query_clean_cum_rel_err_sin": qc["cum_rel_err_sin"],
            "query_clean_cum_rel_err_all": qc["cum_rel_err_all"],
        })

    if results["feature_clean"] is not None:
        save_dict.update({
            "feature_A_icon_clean": results["feature_clean"]["A_icon"],
            "feature_rel_err_map_clean": results["feature_clean"]["rel_err_map"],
            "feature_rel_frob_err_clean": np.asarray(results["feature_clean"]["rel_frob_err"]),
        })

    np.savez(os.path.join(fig_dir, "burgers_wrong_negative_control_results.npz"), **save_dict)
    print("Saved arrays to", os.path.join(fig_dir, "burgers_wrong_negative_control_results.npz"))

def plot_all_figures(results):
    plot_representative_prediction(results)
    plot_prediction_error_over_c(results)
    plot_query_spectrum_wrong(results)
    plot_feature_map_wrong(results)
    save_numeric_results(results)


# =============================================================================
# 8. Main
# =============================================================================

if __name__ == "__main__":
    print("Negative-control Burgers evaluation")
    print("Wrong checkpoint:", loaded_wrong_ckpt)
    if loaded_clean_ckpt is not None:
        print("Clean checkpoint:", loaded_clean_ckpt)
    print("Figures will be saved to:", fig_dir)
    results = compute_all_diagnostics()
    plot_all_figures(results)
