# -*- coding: utf-8 -*-
"""
Advection mode-bandwidth ablation plotting script.

Compare two hidden-shift/advection ICON checkpoints trained with different
random Fourier bandwidths, e.g. n_train=8 versus n_train=16.

This script DOES NOT train. It loads two checkpoints with the same ICON
architecture and evaluates them on the same clean advection test prompts.

Recommended figures for an ablation section:
  1. Prediction relative L2 error over hidden speed c, evaluated on
     test functions with bandwidth 8 and bandwidth 16.
  2. Query Jacobian spectrum at a representative c, comparing true shift,
     ICON trained with 8 modes, and ICON trained with 16 modes.
  3. Per-mode Fourier block/phase error at the same representative c.
  4. Fourier feature maps: exact block-rotation map, learned maps from
     both checkpoints, and normalized error maps.

Edit CKPT8_DIR/CKPT16_DIR or CKPT8_PATH/CKPT16_PATH below before running.

Terminal:
    python plot_advection_mode_ablation.py

Notebook:
    %run plot_advection_mode_ablation.py
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from collections import namedtuple
from functools import partial
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

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
from common.project import (
    ADVECTION_ABLATION_FIGURE_DIR,
    ADVECTION_CHECKPOINT_DIR,
    ADVECTION_FREQUENCY8_DIR,
)
from plotting.style import apply_paper_style as apply_shared_plot_style
from plotting.regimes import (style_parameter_curves, style_parameter_sweep, add_regime_key, comparison_filename)


# =============================================================================
# 0. User configuration
# =============================================================================

seed = 0

# -----------------------------------------------------------------------------
# Checkpoints.
# -----------------------------------------------------------------------------
# Set explicit paths if possible.  If a path is None, the script uses the latest
# params_step_*.msgpack in the corresponding directory.
CKPT8_PATH = None
CKPT8_DIR = str(ADVECTION_FREQUENCY8_DIR / "ckpt")

CKPT16_PATH = None
CKPT16_DIR = str(ADVECTION_CHECKPOINT_DIR)

# Examples:
# CKPT8_PATH = "./icon_paperlike_advection_offline_ckpt/params_step_150000.msgpack"
# CKPT16_PATH = str(ADVECTION_CHECKPOINT_DIR / "params_step_200000.msgpack")

fig_dir = str(ADVECTION_ABLATION_FIGURE_DIR)
os.makedirs(fig_dir, exist_ok=True)
save_figs = True

# Prompt/model settings.  These must match the model architecture.
num_pairs = 6
_demo_num = 5
n_grid = 100

# Random function settings used for evaluation.
fourier_decay = 1.0
initial_amp = 0.7
bias_scale = 0.0
normalize_each_function = False

# Hidden speed distribution.
dt = 0.5
c_range: Tuple[float, float] = (0.5, 1.0)

# Architecture.
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
    raise ValueError(f"Unknown MODEL_SIZE={MODEL_SIZE!r}")

# Ablation settings.
TRAIN_BW8 = 8
TRAIN_BW16 = 16
PRED_EVAL_BANDWIDTHS = (8, 16)
PRED_C_VALUES = np.asarray([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20])
PRED_PROMPTS_PER_C = 12

REPRESENTATIVE_C = 0.80
SPECTRUM_EVAL_BANDWIDTH = 16
SPECTRUM_CONTEXTS = 5
MAX_MODE = 40

FEATURE_C = 0.80
FEATURE_EVAL_BANDWIDTH = 16
FEATURE_PAIR_MODES = 16


# =============================================================================
# 1. Style and containers
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

RawShift = namedtuple(
    "RawShift",
    ["cond_k", "cond_v", "qoi_k", "qoi_v", "c", "shift", "coeff_sin", "coeff_cos", "bias"],
)


def _savefig(fig, filename: str):
    path = os.path.join(fig_dir, filename)
    if save_figs:
        fig.savefig(path, dpi=240, bbox_inches="tight")
        if filename.endswith("_regime_style.png"):
            fig.savefig(path[:-4] + ".pdf", bbox_inches="tight")
    print("saved:", path)


# =============================================================================
# 2. Data generation
# =============================================================================

def eval_random_fourier(x, coeff_sin, coeff_cos, bias):
    n_modes_local = coeff_sin.shape[-1]
    modes = jnp.arange(1, n_modes_local + 1, dtype=x.dtype)
    phase = 2.0 * jnp.pi * x[..., None] * modes
    y = jnp.sum(
        coeff_sin[..., None, :] * jnp.sin(phase)
        + coeff_cos[..., None, :] * jnp.cos(phase),
        axis=-1,
    )
    return y + bias[..., None]


def maybe_normalize_pair(cond_v, qoi_v):
    if not normalize_each_function:
        return cond_v, qoi_v
    mean = jnp.mean(cond_v, axis=2, keepdims=True)
    std = jnp.std(cond_v, axis=2, keepdims=True) + 1e-6
    return (cond_v - mean) / std, (qoi_v - mean) / std


def generate_shift_raw_data(
    key: jax.Array,
    num_operators: int,
    pairs_per_operator: int,
    n_grid: int = 100,
    n_modes: int = 16,
    c_range: Tuple[float, float] = (0.5, 1.0),
    dt: float = 0.5,
    fourier_decay: float = 1.0,
    initial_amp: float = 0.7,
    bias_scale: float = 0.0,
) -> RawShift:
    key_c, key_coef, key_bias = jax.random.split(key, 3)
    c = jax.random.uniform(key_c, shape=(num_operators,), minval=c_range[0], maxval=c_range[1])
    shift = jnp.mod(c * dt, 1.0)

    modes = jnp.arange(1, n_modes + 1, dtype=jnp.float32)
    scale = initial_amp / (modes ** fourier_decay)
    coeff = jax.random.normal(key_coef, shape=(num_operators, pairs_per_operator, 2, n_modes)) * scale
    coeff_sin = coeff[:, :, 0, :]
    coeff_cos = coeff[:, :, 1, :]
    bias = bias_scale * jax.random.normal(key_bias, shape=(num_operators, pairs_per_operator))

    x = jnp.linspace(0.0, 1.0, n_grid, endpoint=False, dtype=jnp.float32)
    x_full = jnp.broadcast_to(x, (num_operators, pairs_per_operator, n_grid))

    cond_k = x_full[..., None]
    qoi_k = x_full[..., None]
    cond_v = eval_random_fourier(x_full, coeff_sin, coeff_cos, bias)[..., None]
    x_shifted = jnp.mod(x_full - shift[:, None, None], 1.0)
    qoi_v = eval_random_fourier(x_shifted, coeff_sin, coeff_cos, bias)[..., None]
    cond_v, qoi_v = maybe_normalize_pair(cond_v, qoi_v)

    return RawShift(
        cond_k=cond_k.astype(jnp.float32),
        cond_v=cond_v.astype(jnp.float32),
        qoi_k=qoi_k.astype(jnp.float32),
        qoi_v=qoi_v.astype(jnp.float32),
        c=c.astype(jnp.float32),
        shift=shift.astype(jnp.float32),
        coeff_sin=coeff_sin.astype(jnp.float32),
        coeff_cos=coeff_cos.astype(jnp.float32),
        bias=bias.astype(jnp.float32),
    )


def build_icon_data(raw: RawShift, demo_num: int = _demo_num) -> tuple[Data, jnp.ndarray]:
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


def periodic_linear_interp_grid(u_grid, x_query):
    u_grid = jnp.asarray(u_grid)
    x_query = jnp.asarray(x_query)
    if u_grid.ndim == 1:
        u_grid = u_grid[None, :]
    if x_query.ndim == 1:
        x_query = x_query[None, :]
    B, N = u_grid.shape
    y = jnp.mod(x_query, 1.0) * N
    i0 = jnp.floor(y).astype(jnp.int32) % N
    i1 = (i0 + 1) % N
    w = y - jnp.floor(y)
    rows = jnp.arange(B)[:, None]
    return (1.0 - w) * u_grid[rows, i0] + w * u_grid[rows, i1]


def make_shift_prompts_at_c(
    key,
    c_value: float,
    num_prompts: int,
    n_modes_eval: int,
    fixed_query_u0=None,
    pairs_per_prompt: int = num_pairs,
):
    raw = generate_shift_raw_data(
        key,
        num_operators=num_prompts,
        pairs_per_operator=pairs_per_prompt,
        n_grid=n_grid,
        n_modes=n_modes_eval,
        c_range=(float(c_value), float(c_value) + 1e-8),
        dt=dt,
        fourier_decay=fourier_decay,
        initial_amp=initial_amp,
        bias_scale=bias_scale,
    )
    if fixed_query_u0 is not None:
        uq = jnp.asarray(fixed_query_u0).reshape(n_grid)
        uq_batch = jnp.broadcast_to(uq[None, :], (num_prompts, n_grid))
        x = raw.qoi_k[:, _demo_num, :, 0]
        x_shifted = jnp.mod(x - raw.shift[:, None], 1.0)
        uT_batch = periodic_linear_interp_grid(uq_batch, x_shifted)
        raw = raw._replace(
            cond_v=raw.cond_v.at[:, _demo_num, :, 0].set(uq_batch),
            qoi_v=raw.qoi_v.at[:, _demo_num, :, 0].set(uT_batch),
        )
    data, label = build_icon_data(raw, demo_num=_demo_num)
    return raw, data, label


def make_fixed_query_u0(key_seed: int, n_modes_eval: int):
    raw = generate_shift_raw_data(
        jax.random.PRNGKey(key_seed),
        num_operators=1,
        pairs_per_operator=num_pairs,
        n_grid=n_grid,
        n_modes=n_modes_eval,
        c_range=(REPRESENTATIVE_C, REPRESENTATIVE_C + 1e-8),
        dt=dt,
        fourier_decay=fourier_decay,
        initial_amp=initial_amp,
        bias_scale=bias_scale,
    )
    return raw.cond_v[0, _demo_num, :, 0]


# =============================================================================
# 3. ICON model
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
            attn = MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.head_dim * self.n_heads,
                out_features=self.model_dim,
            )
            x = attn(inputs_q=x, inputs_k=x, inputs_v=x, mask=mask) + x
            x = nn.LayerNorm()(x)
            ff = MLP(self.model_dim * self.widening_factor, self.model_dim, 1)
            x = ff(x) + x
            x = nn.LayerNorm()(x)
        return x


def build_matrices_from_data_shape(data_shape, mode, shot_num_min):
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


def build_model_and_template():
    key = jax.random.PRNGKey(seed)
    key_data, key_init = jax.random.split(key)
    raw = generate_shift_raw_data(
        key_data,
        num_operators=1,
        pairs_per_operator=num_pairs,
        n_grid=n_grid,
        n_modes=16,
        c_range=c_range,
        dt=dt,
        fourier_decay=fourier_decay,
        initial_amp=initial_amp,
        bias_scale=bias_scale,
    )
    data, _ = build_icon_data(raw, demo_num=_demo_num)
    sample_data = tree.tree_map(lambda x: x[0], data)
    sample_shape = tree.tree_map(lambda x: x.shape, sample_data)
    basic_mask, index_pos, out_mask = build_matrices_from_data_shape(sample_shape, "train", 0)
    model = ICON_LM(model_config, basic_mask=basic_mask, index_pos=index_pos, out_mask=out_mask)
    params_template = model.init({"params": key_init}, sample_data)
    return model, params_template


icon_model, params_template = build_model_and_template()


def predict_one(params, data_one):
    return icon_model.apply(params, data_one, method=ICON_LM.predict)


predict_batch = jax.jit(jax.vmap(predict_one, in_axes=(None, 0)))


def find_latest_checkpoint(directory: str) -> str:
    paths = sorted(glob.glob(os.path.join(directory, "params_step_*.msgpack")))
    if not paths:
        raise FileNotFoundError(
            f"No params_step_*.msgpack found in {directory}. "
            "Edit CKPT8_DIR/CKPT16_DIR or set CKPT8_PATH/CKPT16_PATH explicitly."
        )
    return paths[-1]


def load_params(path: str):
    with open(path, "rb") as f:
        return serialization.from_bytes(params_template, f.read())


# =============================================================================
# 4. Exact shift maps and learned Jacobians
# =============================================================================

def exact_fourier_rotation_block(k_mode: int, c_value: float) -> np.ndarray:
    theta = 2.0 * np.pi * k_mode * float(c_value) * dt
    # Basis order is [cos, sin]. For G u(x)=u(x-s):
    # [cos; sin] maps by [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]].
    return np.asarray(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta),  np.cos(theta)]],
        dtype=float,
    )


def exact_fourier_map(c_value: float, max_pair_mode: int) -> np.ndarray:
    A = np.zeros((2 * max_pair_mode, 2 * max_pair_mode), dtype=float)
    for k_mode in range(1, max_pair_mode + 1):
        block = exact_fourier_rotation_block(k_mode, c_value)
        i = 2 * (k_mode - 1)
        A[i:i + 2, i:i + 2] = block
    return A


def fourier_cos_sin_bases(n: int, max_mode: int):
    x = jnp.arange(n, dtype=jnp.float32) / n
    C, S = [], []
    for k_mode in range(1, max_mode + 1):
        c_col = jnp.sqrt(2.0) * jnp.cos(2.0 * jnp.pi * k_mode * x)
        s_col = jnp.sqrt(2.0) * jnp.sin(2.0 * jnp.pi * k_mode * x)
        C.append(c_col / (jnp.linalg.norm(c_col) + 1e-12))
        S.append(s_col / (jnp.linalg.norm(s_col) + 1e-12))
    return jnp.stack(C, axis=1), jnp.stack(S, axis=1)


def paired_fourier_basis(n: int, max_pair_mode: int):
    C, S = fourier_cos_sin_bases(n, max_pair_mode)
    cols = []
    for k in range(max_pair_mode):
        cols.append(C[:, k])
        cols.append(S[:, k])
    return jnp.stack(cols, axis=1)


def icon_query_map(params, data_one):
    def f_query(u_flat):
        data_new = data_one._replace(quest_cond_v=u_flat.reshape(data_one.quest_cond_v.shape))
        return predict_one(params, data_new).reshape(-1)
    return f_query


@partial(jax.jit, static_argnames=("max_mode",))
def icon_mode_gain_by_jvp_fast(params, data_one, max_mode=40):
    u0_flat = data_one.quest_cond_v.reshape(-1)
    n = u0_flat.shape[0]
    f_query = icon_query_map(params, data_one)

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


def icon_jacobian_l2(params, data_one):
    u0_flat = data_one.quest_cond_v.reshape(-1)
    J = jax.jacrev(icon_query_map(params, data_one))(u0_flat)
    n_out, n_in = J.shape
    return jnp.sqrt((1.0 / n_out) / (1.0 / n_in)) * J


def fourier_project_jacobian(J_l2, max_pair_mode: int):
    n_out, n_in = J_l2.shape
    Phi = paired_fourier_basis(n_in, max_pair_mode)
    Psi = paired_fourier_basis(n_out, max_pair_mode)
    return Psi.T @ J_l2 @ Phi


def cumulative_relative_errors(icon_cs: np.ndarray, true_cs: np.ndarray, eps=1e-12):
    icon_cs = np.asarray(icon_cs)
    true_cs = np.asarray(true_cs)
    diff2 = (icon_cs - true_cs) ** 2
    true2 = true_cs ** 2
    cum_cos = np.sqrt(np.cumsum(diff2[:, 0])) / (np.sqrt(np.cumsum(true2[:, 0])) + eps)
    cum_sin = np.sqrt(np.cumsum(diff2[:, 1])) / (np.sqrt(np.cumsum(true2[:, 1])) + eps)
    cum_all = np.sqrt(np.cumsum(np.sum(diff2, axis=1))) / (
        np.sqrt(np.cumsum(np.sum(true2, axis=1))) + eps
    )
    return cum_cos, cum_sin, cum_all


def block_phase_error(A_icon: np.ndarray, c_value: float, max_pair_mode: int):
    errs = []
    for k_mode in range(1, max_pair_mode + 1):
        i = 2 * (k_mode - 1)
        block_icon = A_icon[i:i + 2, i:i + 2]
        block_true = exact_fourier_rotation_block(k_mode, c_value)
        errs.append(np.linalg.norm(block_icon - block_true, ord="fro") / (np.linalg.norm(block_true, ord="fro") + 1e-12))
    return np.asarray(errs)


# =============================================================================
# 5. Diagnostics
# =============================================================================

def rel_l2_per_prompt(pred, label):
    target = np.asarray(label)[:, 0]
    pred = np.asarray(pred)
    diff = pred - target
    diff_flat = diff.reshape(diff.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    return np.linalg.norm(diff_flat, axis=1) / (np.linalg.norm(target_flat, axis=1) + 1e-12)


def compute_prediction_error_over_c(params_by_model):
    out = {}
    for eval_bw in PRED_EVAL_BANDWIDTHS:
        rows = {name: [] for name in params_by_model}
        for i, c_value in enumerate(PRED_C_VALUES):
            _, data_i, label_i = make_shift_prompts_at_c(
                jax.random.PRNGKey(10000 + 100 * eval_bw + i),
                c_value=float(c_value),
                num_prompts=PRED_PROMPTS_PER_C,
                n_modes_eval=eval_bw,
            )
            for name, p in params_by_model.items():
                pred_i = predict_batch(p, data_i)
                err_i = rel_l2_per_prompt(np.asarray(pred_i), np.asarray(label_i))
                rows[name].append({
                    "mean": float(np.mean(err_i)),
                    "std": float(np.std(err_i)),
                    "all": err_i,
                })
        out[eval_bw] = rows
    return out


def compute_spectrum_at_c(params_by_model):
    fixed_uq = make_fixed_query_u0(24680, SPECTRUM_EVAL_BANDWIDTH)
    _, data_i, _ = make_shift_prompts_at_c(
        jax.random.PRNGKey(13579),
        c_value=REPRESENTATIVE_C,
        num_prompts=SPECTRUM_CONTEXTS,
        n_modes_eval=SPECTRUM_EVAL_BANDWIDTH,
        fixed_query_u0=fixed_uq,
    )
    k = np.arange(1, MAX_MODE + 1)
    true_cs = np.ones((MAX_MODE, 2), dtype=float)
    out = {"k": k, "true": true_cs, "fixed_uq": np.asarray(fixed_uq), "models": {}}
    for name, p in params_by_model.items():
        gains = []
        for j in range(SPECTRUM_CONTEXTS):
            data_one = tree.tree_map(lambda x: x[j], data_i)
            g = icon_mode_gain_by_jvp_fast(p, data_one, max_mode=MAX_MODE)
            gains.append(np.asarray(g[:, 1:3]))
        arr = np.stack(gains, axis=0)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        cum_cos, cum_sin, cum_all = cumulative_relative_errors(mean, true_cs)
        out["models"][name] = {
            "mean": mean,
            "std": std,
            "cum_cos": cum_cos,
            "cum_sin": cum_sin,
            "cum_all": cum_all,
        }
    return out


def compute_feature_maps(params_by_model):
    fixed_uq = make_fixed_query_u0(97531, FEATURE_EVAL_BANDWIDTH)
    _, data_i, _ = make_shift_prompts_at_c(
        jax.random.PRNGKey(86420),
        c_value=FEATURE_C,
        num_prompts=1,
        n_modes_eval=FEATURE_EVAL_BANDWIDTH,
        fixed_query_u0=fixed_uq,
    )
    data_one = tree.tree_map(lambda x: x[0], data_i)
    A_true = exact_fourier_map(FEATURE_C, FEATURE_PAIR_MODES)
    out = {"A_true": A_true, "models": {}}
    denom = np.linalg.norm(A_true, ord="fro") + 1e-12
    for name, p in params_by_model.items():
        J_l2 = icon_jacobian_l2(p, data_one)
        A_icon = np.asarray(fourier_project_jacobian(J_l2, FEATURE_PAIR_MODES))
        E = np.abs(A_icon - A_true) / denom
        block_err = block_phase_error(A_icon, FEATURE_C, FEATURE_PAIR_MODES)
        rel_F = float(np.linalg.norm(A_icon - A_true, ord="fro") / denom)
        out["models"][name] = {
            "A_icon": A_icon,
            "E": E,
            "block_err": block_err,
            "rel_F": rel_F,
        }
    return out


def compute_all_diagnostics():
    ckpt8 = CKPT8_PATH if CKPT8_PATH is not None else find_latest_checkpoint(CKPT8_DIR)
    ckpt16 = CKPT16_PATH if CKPT16_PATH is not None else find_latest_checkpoint(CKPT16_DIR)
    print("Loading 8-mode checkpoint:", ckpt8)
    params8 = load_params(ckpt8)
    print("Loading 16-mode checkpoint:", ckpt16)
    params16 = load_params(ckpt16)
    params_by_model = {
        "train modes = 8": params8,
        "train modes = 16": params16,
    }
    results = {
        "ckpt8": ckpt8,
        "ckpt16": ckpt16,
        "prediction": compute_prediction_error_over_c(params_by_model),
        "spectrum": compute_spectrum_at_c(params_by_model),
        "feature": compute_feature_maps(params_by_model),
    }
    return results


# =============================================================================
# 6. Plotting
# =============================================================================

def add_training_bandwidth_lines(ax):
    ax.axvline(TRAIN_BW8, linestyle=":", color="0.35", lw=1.6)
    ax.axvline(TRAIN_BW16, linestyle="--", color="0.15", lw=1.6)


def plot_prediction_error(results):
    pred = results["prediction"]
    fig, axes = plt.subplots(1, len(PRED_EVAL_BANDWIDTHS), figsize=(11, 4), sharey=True, constrained_layout=True)
    if len(PRED_EVAL_BANDWIDTHS) == 1:
        axes = [axes]
    for ax, eval_bw in zip(axes, PRED_EVAL_BANDWIDTHS):
        for name, rows in pred[eval_bw].items():
            means = np.asarray([r["mean"] for r in rows])
            stds = np.asarray([r["std"] for r in rows])
            ax.semilogy(PRED_C_VALUES, means, "o-", label=name)
            ax.fill_between(PRED_C_VALUES, np.maximum(means - stds, 1e-16), means + stds, alpha=0.15)
        ax.axvspan(c_range[0], c_range[1], alpha=0.16, color="0.7", label="training c range")
        ax.set_xlabel("hidden speed c")
        ax.set_ylabel("relative $L^2$ error")
        ax.grid(True, which="both")
        ax.text(0.04, 0.7, rf"test bandwidth $n_{{eval}}={eval_bw}$", transform=ax.transAxes,
                ha="left", va="top", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    for regime_ax in np.asarray(axes, dtype=object).ravel():
        style_parameter_sweep(regime_ax, 'advection')
    axes[-1].legend(frameon=False)
    add_regime_key(plt.gcf())
    _savefig(fig, comparison_filename("advection_ablation_prediction_error_over_c.png"))
    plt.close(fig)


def plot_query_spectrum(results):
    spec = results["spectrum"]
    k = spec["k"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
    comps = [(0, "cos"), (1, "sin")]

    for col, (j, name_comp) in enumerate(comps):
        ax = axes[0, col]
        ax.plot(k, spec["true"][:, j], "k-", lw=2.2, label="true")
        for name, item in spec["models"].items():
            ax.plot(k, item["mean"][:, j], "o-", label=name)
        add_training_bandwidth_lines(ax)
        ax.set_ylabel("Jacobian gain")
        ax.grid(True)
        ax.text(0.04, 0.92, name_comp, transform=ax.transAxes, ha="left", va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

        ax2 = axes[1, col]
        for name, item in spec["models"].items():
            err = item["cum_cos"] if j == 0 else item["cum_sin"]
            ax2.semilogy(k, err, "o-", label=name)
        add_training_bandwidth_lines(ax2)
        ax2.set_xlabel("Fourier mode K")
        ax2.set_ylabel("cumulative relative error")
        ax2.grid(True, which="both")
        ax2.text(0.04, 0.92, name_comp, transform=ax2.transAxes, ha="left", va="top",
                 bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    axes[0, 1].legend(frameon=False)
    _savefig(fig, f"advection_ablation_query_spectrum_c{REPRESENTATIVE_C:.2f}.png")
    plt.close(fig)


def plot_phase_block_error(results):
    feat = results["feature"]
    modes = np.arange(1, FEATURE_PAIR_MODES + 1)
    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    for name, item in feat["models"].items():
        ax.semilogy(modes, item["block_err"], "o-", label=f"{name}, relF={item['rel_F']:.2e}")
    ax.axvline(TRAIN_BW8, linestyle=":", color="0.35", lw=1.6, label="train bandwidth 8")
    ax.axvline(TRAIN_BW16, linestyle="--", color="0.15", lw=1.6, label="train bandwidth 16")
    ax.set_xlabel("Fourier mode k")
    ax.set_ylabel(r"relative $2\times2$ block error")
    ax.grid(True, which="both")
    ax.legend(frameon=False)
    _savefig(fig, f"advection_ablation_phase_block_error_c{FEATURE_C:.2f}.png")
    plt.close(fig)


def plot_feature_maps(results):
    feat = results["feature"]
    A_true = feat["A_true"]
    A8 = feat["models"]["train modes = 8"]["A_icon"]
    A16 = feat["models"]["train modes = 16"]["A_icon"]
    E8 = feat["models"]["train modes = 8"]["E"]
    E16 = feat["models"]["train modes = 16"]["E"]

    vmax = max(np.max(np.abs(A_true)), np.max(np.abs(A8)), np.max(np.abs(A16)), 1e-12)
    evmax = max(np.max(E8), np.max(E16), 1e-12)

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), constrained_layout=True)
    panels = [
        (A_true, "true", "map"),
        (A8, "ICON, train modes = 8", "map"),
        (A16, "ICON, train modes = 16", "map"),
        (E8, "error, train modes = 8", "err"),
        (E16, "error, train modes = 16", "err"),
    ]
    im_map = None
    im_err = None
    for ax, (M, label, kind) in zip(axes, panels):
        if kind == "map":
            im = ax.imshow(M, origin="lower", aspect="equal", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            im_map = im
        else:
            im = ax.imshow(M, origin="lower", aspect="equal", cmap="cividis", vmin=0.0, vmax=evmax)
            im_err = im
        ax.set_xlabel("input Fourier index")
        ax.set_ylabel("output Fourier index")
        ax.text(0.04, 0.94, label, transform=ax.transAxes, ha="left", va="top",
                fontsize=9, bbox=dict(facecolor="white", alpha=0.78, edgecolor="none"))
    fig.colorbar(im_map, ax=axes[:3], shrink=0.78, location="bottom", pad=0.08)
    fig.colorbar(im_err, ax=axes[3:], shrink=0.78, location="bottom", pad=0.08)
    _savefig(fig, f"advection_ablation_fourier_feature_maps_c{FEATURE_C:.2f}.png")
    plt.close(fig)


def save_numeric_results(results):
    spec = results["spectrum"]
    feat = results["feature"]
    np.savez(
        os.path.join(fig_dir, "advection_mode_ablation_results.npz"),
        ckpt8=np.asarray(results["ckpt8"]),
        ckpt16=np.asarray(results["ckpt16"]),
        pred_c_values=PRED_C_VALUES,
        spectrum_k=spec["k"],
        spectrum_true=spec["true"],
        spectrum_mode8=spec["models"]["train modes = 8"]["mean"],
        spectrum_mode16=spec["models"]["train modes = 16"]["mean"],
        spectrum_mode8_cum_cos=spec["models"]["train modes = 8"]["cum_cos"],
        spectrum_mode8_cum_sin=spec["models"]["train modes = 8"]["cum_sin"],
        spectrum_mode16_cum_cos=spec["models"]["train modes = 16"]["cum_cos"],
        spectrum_mode16_cum_sin=spec["models"]["train modes = 16"]["cum_sin"],
        feature_A_true=feat["A_true"],
        feature_A_mode8=feat["models"]["train modes = 8"]["A_icon"],
        feature_A_mode16=feat["models"]["train modes = 16"]["A_icon"],
        feature_E_mode8=feat["models"]["train modes = 8"]["E"],
        feature_E_mode16=feat["models"]["train modes = 16"]["E"],
        feature_block_err_mode8=feat["models"]["train modes = 8"]["block_err"],
        feature_block_err_mode16=feat["models"]["train modes = 16"]["block_err"],
    )


def plot_all(results):
    plot_prediction_error(results)
    plot_query_spectrum(results)
    plot_phase_block_error(results)
    plot_feature_maps(results)
    save_numeric_results(results)


def main():
    results = compute_all_diagnostics()
    plot_all(results)
    print("\nFinished. Figures saved in:", fig_dir)
    print("Generated files:")
    for name in [
        "advection_ablation_prediction_error_over_c.png",
        f"advection_ablation_query_spectrum_c{REPRESENTATIVE_C:.2f}.png",
        f"advection_ablation_phase_block_error_c{FEATURE_C:.2f}.png",
        f"advection_ablation_fourier_feature_maps_c{FEATURE_C:.2f}.png",
        "advection_mode_ablation_results.npz",
    ]:
        print("  -", os.path.join(fig_dir, name))


if __name__ == "__main__":
    main()
