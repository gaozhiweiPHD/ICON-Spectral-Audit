# -*- coding: utf-8 -*-
"""
Complete Allen--Cahn hidden-epsilon ICON training and paper-style diagnostics,
aligned with the Burgers hidden-viscosity training/evaluation settings.

PDE:
    u_t = eps^2 u_xx + u - u^3,   x in [0,1], periodic.

The hidden operator parameter is eps.  Each prompt contains five demonstration
pairs and one query pair generated with the same eps, but eps is not provided
to ICON.

This script supports full offline prompt-based ICON training and the
paper-style evaluation/plotting pipeline.  In evaluation mode it saves:

  1. relative prediction error over hidden epsilon;
  2. query spectrum over epsilon with cumulative relative spectral error;
  3. predictions for three fixed base states;
  4. base-state context spectra with cumulative relative spectral error;
  5. Fourier feature maps: true, ICON, and absolute error, with shared scales.

Put this file in the same directory as models_utils.py.
"""

from __future__ import annotations

import argparse
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

import optax
from flax import linen as nn
from flax.linen import MultiHeadDotProductAttention
from flax import serialization
from einshape import jax_einshape as einshape

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from common import models_utils as mu
from common.project import ALLEN_CAHN_CHECKPOINT_DIR, ALLEN_CAHN_FIGURE_DIR
from plotting.style import apply_paper_style as apply_shared_plot_style
from plotting.regimes import (style_parameter_curves, style_parameter_sweep, add_regime_key, comparison_filename)


# =============================================================================
# 0. Configuration
# =============================================================================

seed = 0

# Prompt/data settings, aligned with the Burgers hidden-viscosity run.
num_pairs = 6
demo_num = 5
n_grid = 100
n_modes = 16
fourier_decay = 1.0
initial_amp = 0.7
bias_scale = 0.0
normalize_each_function = False

# Allen--Cahn settings: u_t = eps^2 u_xx + u - u^3.
T_final = 0.25
eps_range: Tuple[float, float] = (0.02, 0.08)
n_solver_steps = 800

# Offline pool.
num_train_operators = 2000
pairs_per_operator = 20
num_val_operators = 128
pairs_per_val_operator = num_pairs

# Training setup, aligned with the Burgers hidden-viscosity run.
batch_size = 4
num_steps = 200000
print_every = 2000
learning_rate = 2.0e-4
weight_decay = 1.0e-4
warmup_steps = 2000
clip_norm = 1.0
val_eval_batch_size = 1
save_every = 10000

ckpt_dir = str(ALLEN_CAHN_CHECKPOINT_DIR)
fig_dir = str(ALLEN_CAHN_FIGURE_DIR)
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
save_figs = True

# Paper diagnostics. Use fewer contexts first if debugging speed.
paper_eps_values = np.asarray([0.01, 0.015, 0.02, 0.035, 0.05, 0.065, 0.08, 0.10, 0.12], dtype=float)
paper_query_contexts_per_eps = 5
paper_base_contexts_per_eps = 3
paper_prediction_eqns_per_eps = 8
paper_max_mode = 40
paper_base_max_mode = 30
paper_feature_pair_modes = 16
paper_base_eps_value = 0.05
paper_base_kinds = ("near_zero", "moderate", "interface")

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
        "grid.alpha": 0.32,
        "figure.dpi": 120,
    })


set_paper_plot_style()


# =============================================================================
# 1. Containers
# =============================================================================

Data = namedtuple(
    "Data",
    [
        "demo_cond_k", "demo_cond_v", "demo_cond_mask",
        "demo_qoi_k", "demo_qoi_v", "demo_qoi_mask",
        "quest_cond_k", "quest_cond_v", "quest_cond_mask",
        "quest_qoi_k", "quest_qoi_mask",
    ],
)

RawAC = namedtuple(
    "RawAC",
    ["cond_k", "cond_v", "qoi_k", "qoi_v", "eps", "coeff_sin", "coeff_cos", "bias"],
)


# =============================================================================
# 2. Random functions and Allen--Cahn solver
# =============================================================================

def eval_random_fourier(x, coeff_sin, coeff_cos, bias):
    modes = jnp.arange(1, coeff_sin.shape[-1] + 1, dtype=x.dtype)
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


def _ac_rhs(u, eps, k):
    """u_t = eps^2 u_xx + u - u^3."""
    u_hat = jnp.fft.fft(u, axis=-1)
    u_xx = jnp.fft.ifft(-(k ** 2) * u_hat, axis=-1).real
    return (eps[:, None] ** 2) * u_xx + u - u ** 3


def _rk4_step_ac(u, eps, dt, k):
    k1 = _ac_rhs(u, eps, k)
    k2 = _ac_rhs(u + 0.5 * dt * k1, eps, k)
    k3 = _ac_rhs(u + 0.5 * dt * k2, eps, k)
    k4 = _ac_rhs(u + dt * k3, eps, k)
    u_new = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    # This clip is inherited from the training data generator. For the amplitudes
    # used here it should normally be inactive, but keeping it makes evaluation
    # consistent with training labels.
    return jnp.clip(u_new, -3.0, 3.0)


def solve_allen_cahn_batch(u0, eps, T, nsteps):
    N = u0.shape[-1]
    dt = T / float(nsteps)
    freq = jnp.fft.fftfreq(N, d=1.0 / N)
    k = 2.0 * jnp.pi * freq

    def body(u, _):
        return _rk4_step_ac(u, eps, dt, k), None

    uT, _ = jax.lax.scan(body, u0, xs=None, length=nsteps)
    return uT


solve_allen_cahn_batch_jit = jax.jit(solve_allen_cahn_batch, static_argnames=("nsteps",))


# =============================================================================
# 3. Data generation
# =============================================================================

def generate_allen_cahn_raw_data(
    key,
    num_operators: int,
    pairs_per_operator: int,
    n_grid: int = 100,
    n_modes: int = 8,
    eps_range: Tuple[float, float] = (0.02, 0.08),
    T_final: float = 0.25,
    fourier_decay: float = 1.0,
    bias_scale: float = 0.0,
    initial_amp: float = 0.35,
    n_solver_steps: int = 1000,
):
    key_eps, key_coef, key_bias = jax.random.split(key, 3)
    eps = jax.random.uniform(key_eps, shape=(num_operators,), minval=eps_range[0], maxval=eps_range[1])

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
    cond_v0 = eval_random_fourier(x_full, coeff_sin, coeff_cos, bias)

    flat_u0 = cond_v0.reshape(num_operators * pairs_per_operator, n_grid)
    flat_eps = jnp.repeat(eps, pairs_per_operator)
    flat_uT = solve_allen_cahn_batch_jit(flat_u0, flat_eps, T_final, nsteps=n_solver_steps)

    cond_v = cond_v0[..., None]
    qoi_v = flat_uT.reshape(num_operators, pairs_per_operator, n_grid, 1)
    cond_v, qoi_v = maybe_normalize_pair(cond_v, qoi_v)

    return RawAC(
        cond_k=cond_k.astype(jnp.float32),
        cond_v=cond_v.astype(jnp.float32),
        qoi_k=qoi_k.astype(jnp.float32),
        qoi_v=qoi_v.astype(jnp.float32),
        eps=eps.astype(jnp.float32),
        coeff_sin=coeff_sin.astype(jnp.float32),
        coeff_cos=coeff_cos.astype(jnp.float32),
        bias=bias.astype(jnp.float32),
    )


def build_icon_data(raw, demo_num=demo_num):
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


@jax.jit
def sample_offline_prompt_batch(raw_pool, key):
    key_op, key_pair = jax.random.split(key)
    num_ops = raw_pool.cond_k.shape[0]
    pairs_per_op = raw_pool.cond_k.shape[1]

    op_idx = jax.random.randint(key_op, shape=(batch_size,), minval=0, maxval=num_ops)
    scores = jax.random.uniform(key_pair, shape=(batch_size, pairs_per_op))
    pair_idx = jnp.argsort(scores, axis=1)[:, :num_pairs]

    def gather_field(field):
        selected = field[op_idx]
        return jnp.take_along_axis(selected, pair_idx[:, :, None, None], axis=1)

    raw_batch = RawAC(
        cond_k=gather_field(raw_pool.cond_k),
        cond_v=gather_field(raw_pool.cond_v),
        qoi_k=gather_field(raw_pool.qoi_k),
        qoi_v=gather_field(raw_pool.qoi_v),
        eps=raw_pool.eps[op_idx],
        coeff_sin=jnp.zeros((batch_size, num_pairs, n_modes), dtype=jnp.float32),
        coeff_cos=jnp.zeros((batch_size, num_pairs, n_modes), dtype=jnp.float32),
        bias=jnp.zeros((batch_size, num_pairs), dtype=jnp.float32),
    )
    return build_icon_data(raw_batch, demo_num=demo_num)


def make_test_batch(key_seed=1234, eqns=8, eps_rng=None):
    if eps_rng is None:
        eps_rng = eps_range
    raw = generate_allen_cahn_raw_data(
        jax.random.PRNGKey(key_seed),
        num_operators=eqns,
        pairs_per_operator=num_pairs,
        n_grid=n_grid,
        n_modes=n_modes,
        eps_range=eps_rng,
        T_final=T_final,
        fourier_decay=fourier_decay,
        bias_scale=bias_scale,
        initial_amp=initial_amp,
        n_solver_steps=n_solver_steps,
    )
    data, label = build_icon_data(raw)
    return raw, data, label


def make_fixed_eps_batch(eps_value, key_seed=999, eqns=8):
    tiny = 1e-9
    return make_test_batch(key_seed=key_seed, eqns=eqns, eps_rng=(eps_value, eps_value + tiny))


def make_allen_cahn_prompts_at_eps(
    key,
    eps_value: float,
    num_prompts: int,
    pairs_per_prompt: int = num_pairs,
    fixed_query_u0=None,
):
    raw = generate_allen_cahn_raw_data(
        key,
        num_operators=num_prompts,
        pairs_per_operator=pairs_per_prompt,
        n_grid=n_grid,
        n_modes=n_modes,
        eps_range=(float(eps_value), float(eps_value) + 1e-8),
        T_final=T_final,
        fourier_decay=fourier_decay,
        bias_scale=bias_scale,
        initial_amp=initial_amp,
        n_solver_steps=n_solver_steps,
    )

    if fixed_query_u0 is not None:
        if normalize_each_function:
            raise NotImplementedError("fixed_query_u0 is implemented for normalize_each_function=False only.")
        uq = jnp.asarray(fixed_query_u0).reshape(n_grid)
        uq_batch = jnp.broadcast_to(uq[None, :], (num_prompts, n_grid))
        eps_batch = raw.eps
        uT_batch = solve_allen_cahn_batch_jit(uq_batch, eps_batch, T_final, nsteps=n_solver_steps)
        cond_v = raw.cond_v.at[:, demo_num, :, 0].set(uq_batch)
        qoi_v = raw.qoi_v.at[:, demo_num, :, 0].set(uT_batch)
        raw = raw._replace(cond_v=cond_v, qoi_v=qoi_v)

    data, label = build_icon_data(raw, demo_num=demo_num)
    return raw, data, label


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


def build_model_params():
    key = jax.random.PRNGKey(seed)
    key_data, key_init = jax.random.split(key)
    raw, data, _ = make_test_batch(key_seed=int(key_data[0]), eqns=1, eps_rng=eps_range)
    sample_data = tree.tree_map(lambda x: x[0], data)
    sample_shape = tree.tree_map(lambda x: x.shape, sample_data)
    basic_mask, index_pos, out_mask = build_matrices_from_data_shape(sample_shape, "train", 0)
    model = ICON_LM(model_config, basic_mask=basic_mask, index_pos=index_pos, out_mask=out_mask)
    params = model.init({"params": key_init}, sample_data)
    return model, params


icon_model, params = build_model_params()


def predict_one(params, data_one):
    return icon_model.apply(params, data_one, method=ICON_LM.predict)


predict_batch = jax.jit(jax.vmap(predict_one, in_axes=(None, 0)))


# =============================================================================
# 5. Training utilities
# =============================================================================

def make_optimizer():
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=num_steps,
        end_value=learning_rate * 0.05,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )
    return tx, schedule


tx, lr_schedule = make_optimizer()
opt_state = tx.init(params)


@jax.jit
def loss_fn(params, data, label):
    pred = jax.vmap(lambda d: icon_model.apply(params, d))(data)
    demo_target = data.demo_qoi_v.reshape((data.demo_qoi_v.shape[0], -1, 1))
    query_target = label.reshape((label.shape[0], -1, 1))
    target = jnp.concatenate([demo_target, query_target], axis=1)
    return jnp.mean((pred - target) ** 2)


@jax.jit
def train_step(params, opt_state, raw_pool, key):
    data, label = sample_offline_prompt_batch(raw_pool, key)
    loss, grads = jax.value_and_grad(loss_fn)(params, data, label)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def evaluate_batch(params, data, label):
    pred = predict_batch(params, data)
    target = label[:, 0]
    mse = jnp.mean((pred - target) ** 2)
    rmse = jnp.sqrt(mse)
    nmse = mse / (jnp.var(target) + 1e-12)
    rel_l2 = jnp.linalg.norm(pred - target) / (jnp.linalg.norm(target) + 1e-12)
    return pred, target, mse, rmse, nmse, rel_l2


def save_checkpoint(params, step):
    path = os.path.join(ckpt_dir, f"params_step_{step:06d}.msgpack")
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(params))
    print("Saved checkpoint:", path)
    return path


def load_checkpoint(path):
    global params
    with open(path, "rb") as f:
        params = serialization.from_bytes(params, f.read())
    print("Loaded checkpoint:", path)
    return params


def find_latest_checkpoint():
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "params_step_*.msgpack")))
    if not paths:
        raise FileNotFoundError(f"No params_step_*.msgpack found in {ckpt_dir}")
    return paths[-1]


def train_allen_cahn():
    global params, opt_state

    key = jax.random.PRNGKey(seed)
    key_train, key_val, key_loop = jax.random.split(key, 3)

    print("Generating offline Allen--Cahn training pool...")
    raw_train = generate_allen_cahn_raw_data(
        key_train,
        num_operators=num_train_operators,
        pairs_per_operator=pairs_per_operator,
        n_grid=n_grid,
        n_modes=n_modes,
        eps_range=eps_range,
        T_final=T_final,
        fourier_decay=fourier_decay,
        bias_scale=bias_scale,
        initial_amp=initial_amp,
        n_solver_steps=n_solver_steps,
    )
    raw_train = tree.tree_map(jax.device_put, raw_train)
    print(f"Offline pool: {num_train_operators} x {pairs_per_operator} = {num_train_operators * pairs_per_operator} pairs")

    raw_val = generate_allen_cahn_raw_data(
        key_val,
        num_operators=num_val_operators,
        pairs_per_operator=pairs_per_val_operator,
        n_grid=n_grid,
        n_modes=n_modes,
        eps_range=eps_range,
        T_final=T_final,
        fourier_decay=fourier_decay,
        bias_scale=bias_scale,
        initial_amp=initial_amp,
        n_solver_steps=n_solver_steps,
    )
    val_data, val_label = build_icon_data(raw_val)
    val_data = tree.tree_map(jax.device_put, val_data)
    val_label = jax.device_put(val_label)

    key_loop, key_batch = jax.random.split(key_loop)
    params, opt_state, _ = train_step(params, opt_state, raw_train, key_batch)

    for step in range(1, num_steps + 1):
        key_loop, key_batch = jax.random.split(key_loop)
        params, opt_state, loss = train_step(params, opt_state, raw_train, key_batch)

        if step % print_every == 0 or step == 1:
            _, _, mse, rmse, nmse, rel_l2 = evaluate_batch(params, val_data, val_label)
            print(
                f"step={step:06d}  lr={float(lr_schedule(step)):.3e}  "
                f"train_mse={float(loss):.6e}  val_mse={float(mse):.6e}  "
                f"val_rmse={float(rmse):.6e}  val_nmse={float(nmse):.6e}  "
                f"val_rel_l2={float(rel_l2):.6e}"
            )
        if step % save_every == 0:
            save_checkpoint(params, step)

    save_checkpoint(params, num_steps)
    return params


# =============================================================================
# 6. Unified spectral utilities
# =============================================================================

def cumulative_relative_error(icon_cs, true_cs, eps=1e-12):
    """
    icon_cs, true_cs: [num_parameter_values, num_modes, 2]
    last dimension: 0=cos, 1=sin.
    Returns cumulative relative errors for cos, sin, and combined curves.
    """
    icon_cs = np.asarray(icon_cs)
    true_cs = np.asarray(true_cs)
    diff2 = (icon_cs - true_cs) ** 2
    true2 = true_cs ** 2

    cum_cos = np.sqrt(np.cumsum(diff2[:, :, 0], axis=1)) / (
        np.sqrt(np.cumsum(true2[:, :, 0], axis=1)) + eps
    )
    cum_sin = np.sqrt(np.cumsum(diff2[:, :, 1], axis=1)) / (
        np.sqrt(np.cumsum(true2[:, :, 1], axis=1)) + eps
    )
    cum_all = np.sqrt(np.cumsum(np.sum(diff2, axis=2), axis=1)) / (
        np.sqrt(np.cumsum(np.sum(true2, axis=2), axis=1)) + eps
    )
    return cum_cos, cum_sin, cum_all


def fourier_pair_bases(n: int, max_mode: int):
    x = jnp.arange(n, dtype=jnp.float32) / n
    cols = []
    for k_mode in range(1, max_mode + 1):
        c_col = jnp.sqrt(2.0) * jnp.cos(2.0 * jnp.pi * k_mode * x)
        s_col = jnp.sqrt(2.0) * jnp.sin(2.0 * jnp.pi * k_mode * x)
        c_col = c_col / (jnp.linalg.norm(c_col) + 1e-12)
        s_col = s_col / (jnp.linalg.norm(s_col) + 1e-12)
        cols.extend([c_col, s_col])
    return jnp.stack(cols, axis=1)


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


def true_allen_cahn_map_u0(u0_flat, eps_value):
    u0 = jnp.asarray(u0_flat).reshape(1, n_grid)
    eps_arr = jnp.asarray(eps_value, dtype=jnp.float32).reshape(1)
    uT = solve_allen_cahn_batch_jit(u0, eps_arr, T_final, nsteps=n_solver_steps)
    return uT.reshape(-1)


@partial(jax.jit, static_argnames=("max_mode",))
def true_mode_gain_by_jvp_u0_fast(u0_flat, eps_value, max_mode=40):
    u0_flat = jnp.asarray(u0_flat).reshape(-1)
    n = u0_flat.shape[0]
    f = lambda u: true_allen_cahn_map_u0(u, eps_value)

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


def icon_jacobian_l2(params, data_one):
    u0_flat = data_one.quest_cond_v.reshape(-1)

    def f_query(u_flat):
        data_new = data_one._replace(quest_cond_v=u_flat.reshape(data_one.quest_cond_v.shape))
        return predict_one(params, data_new).reshape(-1)

    J = jax.jacrev(f_query)(u0_flat)
    n_out, n_in = J.shape
    return jnp.sqrt((1.0 / n_out) / (1.0 / n_in)) * J


def true_jacobian_l2_u0(u0_flat, eps_value):
    f = lambda u: true_allen_cahn_map_u0(u, eps_value)
    J = jax.jacrev(f)(jnp.asarray(u0_flat).reshape(-1))
    n_out, n_in = J.shape
    return jnp.sqrt((1.0 / n_out) / (1.0 / n_in)) * J


def fourier_project_jacobian(J_l2, max_pair_mode=12):
    n_out, n_in = J_l2.shape
    Phi = fourier_pair_bases(n_in, max_pair_mode)
    Psi = fourier_pair_bases(n_out, max_pair_mode)
    return Psi.T @ J_l2 @ Phi


def offdiag_ratio(A):
    A = np.asarray(A)
    D = np.diag(np.diag(A))
    return np.linalg.norm(A - D, ord="fro") / (np.linalg.norm(A, ord="fro") + 1e-12)


# =============================================================================
# 7. Base states
# =============================================================================

def make_ac_base_state(kind, n_grid=100):
    x = jnp.linspace(0.0, 1.0, n_grid, endpoint=False, dtype=jnp.float32)
    if kind == "near_zero":
        u = 0.15 * jnp.sin(2.0 * jnp.pi * x)
    elif kind == "moderate":
        u = 0.55 * jnp.sin(2.0 * jnp.pi * x)
    elif kind == "interface":
        width = 0.04
        u = 0.75 * jnp.tanh(jnp.sin(2.0 * jnp.pi * x) / width)
    else:
        raise ValueError(kind)
    u = u - jnp.mean(u)
    u = jnp.clip(u, -0.95, 0.95)
    return u[None, :, None].astype(jnp.float32)


def base_label(kind):
    return {
        "near_zero": "near-zero",
        "moderate": "moderate",
        "interface": "interface-like",
    }[kind]


def make_prompt_with_fixed_eps_and_uq(eps_value, fixed_uq, key_seed):
    raw, data, _ = make_allen_cahn_prompts_at_eps(
        jax.random.PRNGKey(key_seed),
        eps_value=float(eps_value),
        num_prompts=1,
        pairs_per_prompt=num_pairs,
        fixed_query_u0=fixed_uq,
    )
    data_one = tree.tree_map(lambda x: x[0], data)
    return data_one, raw


# =============================================================================
# 8. Computation functions for paper figures
# =============================================================================

def compute_prediction_error_over_eps(eps_values=paper_eps_values, eqns_per_eps=paper_prediction_eqns_per_eps):
    rows = []
    for i, eps_value in enumerate(eps_values):
        raw, data, label = make_fixed_eps_batch(float(eps_value), key_seed=1000 + i, eqns=eqns_per_eps)
        pred, target, mse, rmse, nmse, rel_l2 = evaluate_batch(params, data, label)
        rows.append([float(eps_value), float(mse), float(rel_l2)])
        print(f"eps={eps_value:.6f}  rel_L2={float(rel_l2):.6e}")
    return np.asarray(rows)


def compute_context_spectrum_for_fixed_uq(
    fixed_uq,
    eps_values=paper_eps_values,
    contexts_per_eps=paper_query_contexts_per_eps,
    max_mode=paper_max_mode,
    key_seed=2028,
):
    key0 = jax.random.PRNGKey(key_seed)
    icon_mean = []
    icon_std = []
    true_gains = []

    # compile once
    data0, _ = make_prompt_with_fixed_eps_and_uq(float(eps_values[0]), fixed_uq, key_seed=key_seed)
    _ = icon_mode_gain_by_jvp_fast(params, data0, max_mode=max_mode).block_until_ready()
    _ = true_mode_gain_by_jvp_u0_fast(jnp.asarray(fixed_uq).reshape(-1), float(eps_values[0]), max_mode=max_mode).block_until_ready()

    for i, eps_value in enumerate(eps_values):
        key_i = jax.random.fold_in(key0, i)
        raw_i, data_i, label_i = make_allen_cahn_prompts_at_eps(
            key_i,
            eps_value=float(eps_value),
            num_prompts=contexts_per_eps,
            pairs_per_prompt=num_pairs,
            fixed_query_u0=fixed_uq,
        )
        icon_list = []
        for j in range(contexts_per_eps):
            data_one = tree.tree_map(lambda x: x[j], data_i)
            g_icon = icon_mode_gain_by_jvp_fast(params, data_one, max_mode=max_mode)
            icon_list.append(np.asarray(g_icon[:, 1:3]))
        icon_arr = np.stack(icon_list, axis=0)
        icon_mean.append(icon_arr.mean(axis=0))
        icon_std.append(icon_arr.std(axis=0))

        g_true = true_mode_gain_by_jvp_u0_fast(
            jnp.asarray(fixed_uq).reshape(-1),
            float(eps_value),
            max_mode=max_mode,
        )
        true_gains.append(np.asarray(g_true[:, 1:3]))

    icon_mean = np.stack(icon_mean, axis=0)
    icon_std = np.stack(icon_std, axis=0)
    true_gains = np.stack(true_gains, axis=0)
    cum_cos, cum_sin, cum_all = cumulative_relative_error(icon_mean, true_gains)

    return {
        "k": np.arange(1, max_mode + 1),
        "eps_values": np.asarray(eps_values, dtype=float),
        "icon_mean": icon_mean,
        "icon_std": icon_std,
        "true": true_gains,
        "cum_rel_err_cos": cum_cos,
        "cum_rel_err_sin": cum_sin,
        "cum_rel_err_all": cum_all,
    }


def compute_base_predictions(base_kinds=paper_base_kinds, eps_value=paper_base_eps_value, key_seed=444):
    rows = []
    for i, kind in enumerate(base_kinds):
        uq = make_ac_base_state(kind, n_grid=n_grid)
        data_one, _ = make_prompt_with_fixed_eps_and_uq(float(eps_value), uq, key_seed=key_seed + i)
        pred = np.asarray(predict_one(params, data_one).reshape(-1))
        true_uT = np.asarray(true_allen_cahn_map_u0(jnp.asarray(uq).reshape(-1), float(eps_value)))
        x = np.asarray(data_one.quest_qoi_k[0, :, 0])
        u0 = np.asarray(uq[0, :, 0])
        rel_l2 = np.linalg.norm(pred - true_uT) / (np.linalg.norm(true_uT) + 1e-12)
        rows.append({
            "kind": kind,
            "x": x,
            "u0": u0,
            "true": true_uT,
            "pred": pred,
            "rel_l2": float(rel_l2),
            "eps": float(eps_value),
        })
    return rows


def compute_base_state_context_spectra(
    base_kinds=paper_base_kinds,
    eps_values=(0.02, 0.05, 0.08),
    contexts_per_eps=paper_base_contexts_per_eps,
    max_mode=paper_base_max_mode,
):
    out = {}
    for bi, kind in enumerate(base_kinds):
        uq = make_ac_base_state(kind, n_grid=n_grid)
        out[kind] = compute_context_spectrum_for_fixed_uq(
            uq,
            eps_values=np.asarray(eps_values, dtype=float),
            contexts_per_eps=contexts_per_eps,
            max_mode=max_mode,
            key_seed=13579 + 1000 * bi,
        )
    return out


def compute_feature_maps(
    base_kinds=paper_base_kinds,
    eps_value=paper_base_eps_value,
    max_pair_mode=paper_feature_pair_modes,
):
    results = {}
    for i, kind in enumerate(base_kinds):
        uq = make_ac_base_state(kind, n_grid=n_grid)
        data_one, _ = make_prompt_with_fixed_eps_and_uq(float(eps_value), uq, key_seed=60000 + i)
        J_icon = icon_jacobian_l2(params, data_one)
        J_true = true_jacobian_l2_u0(jnp.asarray(uq).reshape(-1), float(eps_value))
        A_icon = np.asarray(fourier_project_jacobian(J_icon, max_pair_mode=max_pair_mode))
        A_true = np.asarray(fourier_project_jacobian(J_true, max_pair_mode=max_pair_mode))
        A_err = A_icon - A_true
        results[kind] = {
            "A_icon": A_icon,
            "A_true": A_true,
            "A_err": A_err,
            "rel_fro_error": float(np.linalg.norm(A_err, ord="fro") / (np.linalg.norm(A_true, ord="fro") + 1e-12)),
            "roff_icon": float(offdiag_ratio(A_icon)),
            "roff_true": float(offdiag_ratio(A_true)),
        }
    return results


def compute_all_paper_diagnostics():
    print("Computing relative prediction error over hidden epsilon...")
    pred_error = compute_prediction_error_over_eps()

    print("Generating a fixed query state for query-spectrum diagnostics...")
    raw_test, test_data, test_label = make_test_batch(key_seed=2026, eqns=8, eps_rng=eps_range)
    data_one = tree.tree_map(lambda x: x[0], test_data)
    fixed_uq = data_one.quest_cond_v

    print("Computing query spectrum over epsilon...")
    query_spectrum = compute_context_spectrum_for_fixed_uq(
        fixed_uq,
        eps_values=paper_eps_values,
        contexts_per_eps=paper_query_contexts_per_eps,
        max_mode=paper_max_mode,
        key_seed=2028,
    )

    print("Computing base-state predictions...")
    base_predictions = compute_base_predictions()

    print("Computing base-state context spectra...")
    base_context = compute_base_state_context_spectra()

    print("Computing Fourier feature maps...")
    feature_maps = compute_feature_maps()

    return {
        "pred_error": pred_error,
        "query_spectrum": query_spectrum,
        "base_predictions": base_predictions,
        "base_context": base_context,
        "feature_maps": feature_maps,
    }


# =============================================================================
# 9. Plotting functions
# =============================================================================

def savefig(fig, name):
    if save_figs:
        fig.savefig(os.path.join(fig_dir, name + ".png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(fig_dir, name + ".pdf"), bbox_inches="tight")
    plt.show()


def plot_prediction_error_over_eps(results):
    arr = results["pred_error"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.semilogy(arr[:, 0], arr[:, 2], "o-", label=r"relative $L^2$")
    ax.axvspan(eps_range[0], eps_range[1], alpha=0.18, label=r"training range")
    ax.set_xlabel(r"hidden parameter $\varepsilon$")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.grid(True, which="both")
    for regime_ax in np.asarray(ax, dtype=object).ravel():
        style_parameter_sweep(regime_ax, 'allen_cahn')
    ax.legend(frameon=True)
    add_regime_key(plt.gcf())
    savefig(fig, comparison_filename("allen_cahn_relative_prediction_error_over_hidden_epsilon"))


def plot_query_spectrum_with_cumulative_error(results):
    qs = results["query_spectrum"]
    k = qs["k"]
    eps_values = qs["eps_values"]
    icon_mean = qs["icon_mean"]
    cum_cos = qs["cum_rel_err_cos"]
    cum_sin = qs["cum_rel_err_sin"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.0), sharex=True, constrained_layout=True)

    for i, eps_value in enumerate(eps_values):
        axes[0, 0].plot(k, icon_mean[i, :, 0], "o-", label=f"{eps_value:g}")
        axes[0, 1].plot(k, icon_mean[i, :, 1], "s-", label=f"{eps_value:g}")
        axes[1, 0].plot(k, cum_cos[i], "o-")
        axes[1, 1].plot(k, cum_sin[i], "s-")

    for ax in axes.ravel():
        ax.axvline(n_modes, linestyle="--", color="k", linewidth=1.6)
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
    style_parameter_curves(axes, eps_values, 'allen_cahn')
    axes[0, 1].legend(title=r"$\varepsilon$", ncol=2, frameon=True)
    add_regime_key(plt.gcf())
    savefig(fig, comparison_filename("allen_cahn_query_spectrum_over_eps_with_cumulative_relative_error"))


def plot_base_state_predictions(results):
    rows = results["base_predictions"]
    fig, axes = plt.subplots(len(rows), 2, figsize=(9.0, 3.0 * len(rows)), sharex=True, constrained_layout=True)
    for i, row in enumerate(rows):
        axes[i, 0].plot(row["x"], row["u0"], "k-")
        axes[i, 0].set_ylabel(r"$u_0(x)$")
        axes[i, 0].grid(True)
        axes[i, 0].text(0.03, 0.88, base_label(row["kind"]), transform=axes[i, 0].transAxes,
                        ha="left", va="top", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

        axes[i, 1].plot(row["x"], row["true"], "k-", label="true")
        axes[i, 1].plot(row["x"], row["pred"], "r--", label="ICON")
        axes[i, 1].set_ylabel(r"$u(T,x)$")
        axes[i, 1].grid(True)
        axes[i, 1].text(0.03, 0.88, rf"Rel-$L^2={row['rel_l2']:.2e}$", transform=axes[i, 1].transAxes,
                        ha="left", va="top", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
        if i == 0:
            axes[i, 1].legend(frameon=True)

    axes[-1, 0].set_xlabel(r"$x$")
    axes[-1, 1].set_xlabel(r"$x$")
    savefig(fig, "allen_cahn_base_state_predictions")


def plot_base_state_context_spectrum(results):
    bc = results["base_context"]
    kinds = list(bc.keys())
    fig, axes = plt.subplots(len(kinds), 4, figsize=(16.0, 3.2 * len(kinds)), sharex="col", constrained_layout=True)
    if len(kinds) == 1:
        axes = axes[None, :]

    for r, kind in enumerate(kinds):
        item = bc[kind]
        k = item["k"]
        eps_values = item["eps_values"]
        icon_mean = item["icon_mean"]
        cum_cos = item["cum_rel_err_cos"]
        cum_sin = item["cum_rel_err_sin"]

        for i, eps_value in enumerate(eps_values):
            axes[r, 0].plot(k, icon_mean[i, :, 0], "o-", label=f"{eps_value:g}")
            axes[r, 1].plot(k, icon_mean[i, :, 1], "s-", label=f"{eps_value:g}")
            axes[r, 2].plot(k, cum_cos[i], "o-")
            axes[r, 3].plot(k, cum_sin[i], "s-")

        for c in range(4):
            axes[r, c].axvline(n_modes, linestyle="--", color="k", linewidth=1.4)
            axes[r, c].grid(True, which="both")
            axes[r, c].text(0.04, 0.88, base_label(kind), transform=axes[r, c].transAxes,
                            ha="left", va="top", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

        axes[r, 0].set_ylabel("ICON gain")
        axes[r, 2].set_ylabel("cum. rel. error")

    axes[0, 0].set_title("cos gain")
    axes[0, 1].set_title("sin gain")
    axes[0, 2].set_title("cos cumulative error")
    axes[0, 3].set_title("sin cumulative error")
    for c in range(4):
        axes[-1, c].set_xlabel(r"Fourier mode $K$")
    axes[0, 1].legend(title=r"$\varepsilon$", frameon=True, ncol=1)
    savefig(fig, "allen_cahn_base_state_context_spectrum_with_cumulative_relative_error")


def plot_feature_maps(results):
    fmap = results["feature_maps"]
    kinds = list(fmap.keys())

    vmax = max(
        max(float(np.max(np.abs(fmap[k]["A_true"]))), float(np.max(np.abs(fmap[k]["A_icon"]))))
        for k in kinds
    ) + 1e-12
    err_vmax = max(float(np.max(np.abs(fmap[k]["A_err"]))) for k in kinds) + 1e-12

    fig, axes = plt.subplots(len(kinds), 3, figsize=(12.5, 3.7 * len(kinds)), constrained_layout=True)
    if len(kinds) == 1:
        axes = axes[None, :]

    im_main = None
    im_err = None
    for r, kind in enumerate(kinds):
        A_true = np.abs(fmap[kind]["A_true"])
        A_icon = np.abs(fmap[kind]["A_icon"])
        A_err = np.abs(fmap[kind]["A_err"])

        im_main = axes[r, 0].imshow(A_true, origin="lower", aspect="equal", interpolation="nearest", cmap="magma", vmin=0.0, vmax=vmax)
        axes[r, 1].imshow(A_icon, origin="lower", aspect="equal", interpolation="nearest", cmap="magma", vmin=0.0, vmax=vmax)
        im_err = axes[r, 2].imshow(A_err, origin="lower", aspect="equal", interpolation="nearest", cmap="magma", vmin=0.0, vmax=err_vmax)

        axes[r, 0].set_ylabel("output mode index")
        axes[r, 0].text(0.03, 0.93, base_label(kind), transform=axes[r, 0].transAxes,
                        ha="left", va="top", color="white",
                        bbox=dict(facecolor="black", alpha=0.45, edgecolor="none"))
        axes[r, 1].text(0.03, 0.93, rf"$r_{{off}}={fmap[kind]['roff_icon']:.2f}$", transform=axes[r, 1].transAxes,
                        ha="left", va="top", color="white",
                        bbox=dict(facecolor="black", alpha=0.45, edgecolor="none"))
        axes[r, 2].text(0.03, 0.93, rf"Rel-Fro$={fmap[kind]['rel_fro_error']:.2e}$", transform=axes[r, 2].transAxes,
                        ha="left", va="top", color="white",
                        bbox=dict(facecolor="black", alpha=0.45, edgecolor="none"))

        for c in range(3):
            axes[r, c].set_xlabel("input mode index")

    axes[0, 0].set_title("true")
    axes[0, 1].set_title("ICON")
    axes[0, 2].set_title("absolute error")
    cbar1 = fig.colorbar(im_main, ax=axes[:, :2], shrink=0.82)
    cbar1.set_label(r"$|A|$")
    cbar2 = fig.colorbar(im_err, ax=axes[:, 2], shrink=0.82)
    cbar2.set_label(r"$|A_{\rm ICON}-A_{\rm true}|$")
    savefig(fig, "allen_cahn_fourier_feature_maps_true_icon_absolute_error")


def plot_all_paper_figures(results):
    plot_prediction_error_over_eps(results)
    plot_query_spectrum_with_cumulative_error(results)
    plot_base_state_predictions(results)
    plot_base_state_context_spectrum(results)
    plot_feature_maps(results)


def save_numeric_results(results):
    qs = results["query_spectrum"]
    save_dict = {
        "pred_error": results["pred_error"],
        "query_k": qs["k"],
        "query_eps_values": qs["eps_values"],
        "query_icon_mean": qs["icon_mean"],
        "query_icon_std": qs["icon_std"],
        "query_true": qs["true"],
        "query_cum_rel_err_cos": qs["cum_rel_err_cos"],
        "query_cum_rel_err_sin": qs["cum_rel_err_sin"],
        "query_cum_rel_err_all": qs["cum_rel_err_all"],
    }

    for kind, item in results["base_context"].items():
        save_dict[f"base_{kind}_k"] = item["k"]
        save_dict[f"base_{kind}_eps_values"] = item["eps_values"]
        save_dict[f"base_{kind}_icon_mean"] = item["icon_mean"]
        save_dict[f"base_{kind}_true"] = item["true"]
        save_dict[f"base_{kind}_cum_rel_err_cos"] = item["cum_rel_err_cos"]
        save_dict[f"base_{kind}_cum_rel_err_sin"] = item["cum_rel_err_sin"]
        save_dict[f"base_{kind}_cum_rel_err_all"] = item["cum_rel_err_all"]

    for kind, item in results["feature_maps"].items():
        save_dict[f"feature_{kind}_A_true"] = item["A_true"]
        save_dict[f"feature_{kind}_A_icon"] = item["A_icon"]
        save_dict[f"feature_{kind}_A_err"] = item["A_err"]
        save_dict[f"feature_{kind}_rel_fro_error"] = np.asarray(item["rel_fro_error"])
        save_dict[f"feature_{kind}_roff_icon"] = np.asarray(item["roff_icon"])
        save_dict[f"feature_{kind}_roff_true"] = np.asarray(item["roff_true"])

    np.savez(os.path.join(fig_dir, "allen_cahn_paper_style_diagnostics_results.npz"), **save_dict)


# =============================================================================
# 10. Evaluation driver and CLI
# =============================================================================

def eval_allen_cahn_paper_style(ckpt_path=None):
    global params
    if ckpt_path is None:
        ckpt_path = find_latest_checkpoint()
    params = load_checkpoint(ckpt_path)

    print("JAX devices:", jax.devices())
    print("Saving figures to:", fig_dir)
    results = compute_all_paper_diagnostics()
    plot_all_paper_figures(results)
    save_numeric_results(results)
    print("Saved figures and arrays to:", fig_dir)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Allen--Cahn hidden-epsilon ICON paper-style diagnostics.")
    parser.add_argument(
        "--mode",
        type=str,
        default="train_eval",
        choices=["train", "eval", "train_eval"],
        help="Run training, evaluation, or both.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint path for evaluation. If omitted, the latest checkpoint in ckpt_dir is used.",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_allen_cahn()
    elif args.mode == "eval":
        eval_allen_cahn_paper_style(args.ckpt)
    elif args.mode == "train_eval":
        train_allen_cahn()
        final_ckpt = args.ckpt if args.ckpt is not None else os.path.join(ckpt_dir, f"params_step_{num_steps:06d}.msgpack")
        eval_allen_cahn_paper_style(final_ckpt)
