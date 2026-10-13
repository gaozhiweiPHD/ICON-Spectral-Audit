# -*- coding: utf-8 -*-
"""
Plot-only/eval script for the aligned 1D hidden-shift advection ICON experiment.

This version is aligned with the Burgers/Allen--Cahn paper-style setup:

    - 5 demo pairs + 1 query pair per prompt;
    - n_grid = 100 spatial tokens;
    - n_modes = 16 random Fourier bandwidth;
    - initial_amp = 0.7, fourier_decay = 1.0, bias_scale = 0.0;
    - 2000 hidden operators x 20 pairs/operator in the offline training pool;
    - batch_size = 4, num_steps = 200000, AdamW with warmup-cosine schedule.

PDE:
    u_t + c u_x = 0,    x in [0,1], periodic.

For a fixed time step dt, the exact solution operator is the periodic shift
    G_s u(x) = u((x - s) mod 1),   s = c * dt.

The hidden speed c, or equivalently the shift s, is never provided to ICON.
It must be inferred from the context pairs.

This file includes:
    1. offline prompt-based training;
    2. relative prediction error over hidden c;
    3. query/context spectrum over c with cumulative relative gain error;
    4. Fourier feature maps for the learned shift operator and exact shift map;
    5. phase/block-rotation error diagnostics.

Put this file in the same directory as models_utils.py.
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
import optax
from einshape import jax_einshape as einshape

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from common import models_utils as mu
from common.project import ADVECTION_CHECKPOINT_DIR, ADVECTION_FIGURE_DIR
from plotting.style import apply_paper_style as apply_shared_plot_style
from plotting.regimes import (style_parameter_curves, style_parameter_sweep, add_regime_key, comparison_filename)


# =============================================================================
# 0. User configuration: aligned with Burgers
# =============================================================================

seed = 0

# Prompt structure.
num_pairs = 6
demo_num = 5
n_grid = 100

# Random initial condition settings.
n_modes = 16
fourier_decay = 1.0
initial_amp = 0.7
bias_scale = 0.0
normalize_each_function = False

# Hidden speed distribution. The model never receives c.
dt = 0.5
c_range: Tuple[float, float] = (0.5, 1.0)

# Offline dataset size.
offline_train = True
num_train_operators = 2000
pairs_per_operator = 20
num_val_operators = 128

# Training setup.
batch_size = 4
num_steps = 200000
print_every = 2000
learning_rate = 2.0e-4
weight_decay = 1.0e-4
warmup_steps = 2000
clip_norm = 1.0
val_eval_batch_size = 1

save_every = 10000
ckpt_dir = str(ADVECTION_CHECKPOINT_DIR)
os.makedirs(ckpt_dir, exist_ok=True)

# Figures.
fig_dir = str(ADVECTION_FIGURE_DIR)
os.makedirs(fig_dir, exist_ok=True)
save_figs = True

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
    raise ValueError(f"Unknown MODEL_SIZE={MODEL_SIZE}")


# Diagnostics.
paper_c_values = np.asarray([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20], dtype=float)
paper_c_values_id = np.asarray([0.50, 0.60, 0.70, 0.80, 0.90, 1.00], dtype=float)
paper_max_mode = 40
paper_contexts_per_c = 5
paper_feature_pair_modes = 16


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

RawShift = namedtuple(
    "RawShift",
    ["cond_k", "cond_v", "qoi_k", "qoi_v", "c", "shift", "coeff_sin", "coeff_cos", "bias"],
)


# =============================================================================
# 2. Hidden-shift data generation
# =============================================================================

def eval_random_fourier(
    x: jnp.ndarray,
    coeff_sin: jnp.ndarray,
    coeff_cos: jnp.ndarray,
    bias: jnp.ndarray,
) -> jnp.ndarray:
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
    """Generate an offline pool of hidden-shift input-output pairs."""
    key_c, key_coef, key_bias = jax.random.split(key, 3)

    c = jax.random.uniform(
        key_c,
        shape=(num_operators,),
        minval=c_range[0],
        maxval=c_range[1],
    )
    shift = jnp.mod(c * dt, 1.0)

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


def build_icon_data(raw: RawShift, demo_num: int = 5) -> tuple[Data, jnp.ndarray]:
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
def sample_offline_prompt_batch(raw_pool: RawShift, key: jax.Array) -> tuple[Data, jnp.ndarray]:
    key_op, key_pair = jax.random.split(key)
    num_ops = raw_pool.cond_k.shape[0]
    pp = raw_pool.cond_k.shape[1]

    op_idx = jax.random.randint(key_op, shape=(batch_size,), minval=0, maxval=num_ops)
    scores = jax.random.uniform(key_pair, shape=(batch_size, pp))
    pair_idx = jnp.argsort(scores, axis=1)[:, :num_pairs]

    def gather_field(field):
        selected = field[op_idx]
        return jnp.take_along_axis(selected, pair_idx[:, :, None, None], axis=1)

    raw_batch = RawShift(
        cond_k=gather_field(raw_pool.cond_k),
        cond_v=gather_field(raw_pool.cond_v),
        qoi_k=gather_field(raw_pool.qoi_k),
        qoi_v=gather_field(raw_pool.qoi_v),
        c=raw_pool.c[op_idx],
        shift=raw_pool.shift[op_idx],
        coeff_sin=jnp.zeros((batch_size, num_pairs, n_modes), dtype=jnp.float32),
        coeff_cos=jnp.zeros((batch_size, num_pairs, n_modes), dtype=jnp.float32),
        bias=jnp.zeros((batch_size, num_pairs), dtype=jnp.float32),
    )
    return build_icon_data(raw_batch, demo_num=demo_num)


def make_shift_prompts_at_c(
    key: jax.Array,
    c_value: float,
    num_prompts: int,
    pairs_per_prompt: int = num_pairs,
    fixed_query_u0: jnp.ndarray | None = None,
) -> tuple[RawShift, Data, jnp.ndarray]:
    raw = generate_shift_raw_data(
        key,
        num_operators=num_prompts,
        pairs_per_operator=pairs_per_prompt,
        n_grid=n_grid,
        n_modes=n_modes,
        c_range=(float(c_value), float(c_value) + 1e-8),
        dt=dt,
        fourier_decay=fourier_decay,
        initial_amp=initial_amp,
        bias_scale=bias_scale,
    )

    if fixed_query_u0 is not None:
        if normalize_each_function:
            raise NotImplementedError("fixed_query_u0 replacement assumes normalize_each_function=False.")
        uq = jnp.asarray(fixed_query_u0).reshape(n_grid)
        uq_batch = jnp.broadcast_to(uq[None, :], (num_prompts, n_grid))
        shifts = raw.shift
        x = raw.qoi_k[:, demo_num, :, 0]
        x_shifted = jnp.mod(x - shifts[:, None], 1.0)

        # fixed_query_u0 is given on the grid. Since advection is a shift of
        # a grid function, use periodic linear interpolation for labels.
        uT_batch = periodic_linear_interp_grid(uq_batch, x_shifted)

        raw = raw._replace(
            cond_v=raw.cond_v.at[:, demo_num, :, 0].set(uq_batch),
            qoi_v=raw.qoi_v.at[:, demo_num, :, 0].set(uT_batch),
        )

    data, label = build_icon_data(raw, demo_num=demo_num)
    return raw, data, label


def periodic_linear_interp_grid(u_grid: jnp.ndarray, x_query: jnp.ndarray) -> jnp.ndarray:
    """
    Periodic linear interpolation of grid values on x_j=j/N.

    u_grid: [B,N] or [N]
    x_query: [B,Nq] or [Nq]
    """
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
    raw = generate_shift_raw_data(
        key_data,
        num_operators=1,
        pairs_per_operator=num_pairs,
        n_grid=n_grid,
        n_modes=n_modes,
        c_range=c_range,
        dt=dt,
        fourier_decay=fourier_decay,
        initial_amp=initial_amp,
        bias_scale=bias_scale,
    )
    data, _ = build_icon_data(raw, demo_num=demo_num)
    sample_data = tree.tree_map(lambda x: x[0], data)
    sample_shape = tree.tree_map(lambda x: x.shape, sample_data)
    basic_mask, index_pos, out_mask = build_matrices_from_data_shape(sample_shape, "train", 0)
    model = ICON_LM(model_config, basic_mask=basic_mask, index_pos=index_pos, out_mask=out_mask)
    params = model.init({"params": key_init}, sample_data)
    return model, params


icon_model, params = build_model_params()


def forward_one(params, data_one):
    return icon_model.apply(params, data_one)


def predict_one(params, data_one):
    return icon_model.apply(params, data_one, method=ICON_LM.predict)


forward_batch = jax.jit(jax.vmap(forward_one, in_axes=(None, 0)))
predict_batch = jax.jit(jax.vmap(predict_one, in_axes=(None, 0)))


# =============================================================================
# 4. Training utilities
# =============================================================================

def build_train_target(batch_data: Data, batch_label: jnp.ndarray) -> jnp.ndarray:
    demo_qoi_v = jnp.asarray(batch_data.demo_qoi_v)
    quest_qoi_v = jnp.asarray(batch_label)
    if quest_qoi_v.ndim == 3:
        quest_qoi_v = quest_qoi_v[:, None, :, :]
    target = jnp.concatenate([demo_qoi_v, quest_qoi_v], axis=1)
    bs, n, t, d = target.shape
    return target.reshape(bs, n * t, d)


def make_lr_schedule():
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=max(num_steps, warmup_steps + 1),
        end_value=learning_rate * 0.05,
    )


lr_schedule = make_lr_schedule()
tx = optax.chain(
    optax.clip_by_global_norm(clip_norm),
    optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay),
)
opt_state = tx.init(params)


@jax.jit
def train_step(params, opt_state, batch_data, batch_label):
    target = build_train_target(batch_data, batch_label)

    def loss_fn(p):
        pred = forward_batch(p, batch_data)
        return jnp.mean((pred - target) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state_new = tx.update(grads, opt_state, params)
    params_new = optax.apply_updates(params, updates)
    return params_new, opt_state_new, loss


@jax.jit
def eval_sums(params, data, label):
    pred = predict_batch(params, data)
    target = label[:, 0]
    diff = pred - target
    sse = jnp.sum(diff ** 2)
    target_sum = jnp.sum(target)
    target_sq_sum = jnp.sum(target ** 2)
    count = target.size
    return sse, target_sum, target_sq_sum, count


def eval_metrics_batched(params, data, label, batch_size_eval=1):
    n = label.shape[0]
    total_sse = 0.0
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for i in range(0, n, batch_size_eval):
        j = min(i + batch_size_eval, n)
        data_b = tree.tree_map(lambda x: x[i:j], data)
        label_b = label[i:j]
        sse, tsum, tsq, cnt = eval_sums(params, data_b, label_b)
        sse, tsum, tsq = jax.device_get((sse, tsum, tsq))
        total_sse += float(sse)
        total_sum += float(tsum)
        total_sq_sum += float(tsq)
        total_count += int(cnt)

    mse = total_sse / max(total_count, 1)
    rmse = float(np.sqrt(mse))
    mean = total_sum / max(total_count, 1)
    var = total_sq_sum / max(total_count, 1) - mean ** 2
    nmse = mse / (var + 1e-12)
    rel_l2 = float(np.sqrt(total_sse / (total_sq_sum + 1e-12)))
    return mse, rmse, nmse, rel_l2


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


def find_latest_checkpoint(directory: str) -> str:
    paths = sorted(glob.glob(os.path.join(directory, "params_step_*.msgpack")))
    if not paths:
        raise FileNotFoundError(f"No params_step_*.msgpack found in {directory}")
    return paths[-1]


# =============================================================================
# 5. Train
# =============================================================================

def train_advection():
    global params, opt_state

    key = jax.random.PRNGKey(seed)
    key_train, key_val, key_loop = jax.random.split(key, 3)

    print("Generating offline advection training pool...")
    raw_train = generate_shift_raw_data(
        key_train,
        num_operators=num_train_operators,
        pairs_per_operator=pairs_per_operator,
        n_grid=n_grid,
        n_modes=n_modes,
        c_range=c_range,
        dt=dt,
        fourier_decay=fourier_decay,
        initial_amp=initial_amp,
        bias_scale=bias_scale,
    )
    raw_train = tree.tree_map(jax.device_put, raw_train)

    print(
        f"Offline train pool: {num_train_operators} operators x "
        f"{pairs_per_operator} pairs/operator = "
        f"{num_train_operators * pairs_per_operator} function pairs."
    )

    raw_val = generate_shift_raw_data(
        key_val,
        num_operators=num_val_operators,
        pairs_per_operator=num_pairs,
        n_grid=n_grid,
        n_modes=n_modes,
        c_range=c_range,
        dt=dt,
        fourier_decay=fourier_decay,
        initial_amp=initial_amp,
        bias_scale=bias_scale,
    )
    val_data, val_label = build_icon_data(raw_val, demo_num=demo_num)
    val_data = tree.tree_map(jax.device_put, val_data)
    val_label = jax.device_put(val_label)

    Loss, ValMSE, ValRelL2 = [], [], []

    # compile once
    key_loop, key_batch = jax.random.split(key_loop)
    batch_data, batch_label = sample_offline_prompt_batch(raw_train, key_batch)
    params, opt_state, _ = train_step(params, opt_state, batch_data, batch_label)

    print("Configuration:")
    print("  PDE: u_t + c u_x = 0, periodic")
    print(f"  c_range={c_range}, dt={dt}, shift_range=({c_range[0]*dt:.3f},{c_range[1]*dt:.3f})")
    print(f"  n_grid={n_grid}, n_modes={n_modes}, initial_amp={initial_amp}")
    print(f"  offline operators={num_train_operators}, pairs/operator={pairs_per_operator}")
    print(f"  batch_size={batch_size}, num_steps={num_steps}")
    print("Start training...\n")

    for step in range(num_steps + 1):
        key_loop, key_batch = jax.random.split(key_loop)
        batch_data, batch_label = sample_offline_prompt_batch(raw_train, key_batch)
        params, opt_state, train_loss = train_step(params, opt_state, batch_data, batch_label)

        if step % print_every == 0 or step == num_steps:
            val_mse, val_rmse, val_nmse, val_rel_l2 = eval_metrics_batched(
                params, val_data, val_label, val_eval_batch_size
            )
            lr_now = float(lr_schedule(step))
            print(
                f"step={step:06d}  "
                f"lr={lr_now:.3e}  "
                f"train_mse={float(train_loss):.6e}  "
                f"val_mse={val_mse:.6e}  "
                f"val_rmse={val_rmse:.6e}  "
                f"val_nmse={val_nmse:.6e}  "
                f"val_rel_l2={val_rel_l2:.6e}"
            )
            Loss.append(float(train_loss))
            ValMSE.append(float(val_mse))
            ValRelL2.append(float(val_rel_l2))

        if save_every is not None and save_every > 0 and step > 0 and step % save_every == 0:
            save_checkpoint(params, step)

    final_path = save_checkpoint(params, num_steps)
    np.savez(
        os.path.join(fig_dir, "advection_training_curves.npz"),
        Loss=np.asarray(Loss),
        ValMSE=np.asarray(ValMSE),
        ValRelL2=np.asarray(ValRelL2),
        print_every=np.asarray(print_every),
    )
    return final_path


# =============================================================================
# 6. Ground-truth shift maps and spectra
# =============================================================================

def true_advection_map_u0(u0_flat: jnp.ndarray, c_value: float | jnp.ndarray) -> jnp.ndarray:
    u0_flat = jnp.asarray(u0_flat).reshape(-1)
    n = u0_flat.shape[0]
    x = jnp.arange(n, dtype=jnp.float32) / n
    shift = jnp.mod(jnp.asarray(c_value, dtype=jnp.float32) * dt, 1.0)
    x_shifted = jnp.mod(x - shift, 1.0)
    return periodic_linear_interp_grid(u0_flat, x_shifted).reshape(-1)


def exact_fourier_rotation_block(k_mode: int, c_value: float) -> np.ndarray:
    theta = 2.0 * np.pi * k_mode * float(c_value) * dt
    return np.asarray([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ], dtype=float)


def exact_fourier_map(c_value: float, max_pair_mode: int = 16) -> np.ndarray:
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
        c_col = c_col / (jnp.linalg.norm(c_col) + 1e-12)
        s_col = s_col / (jnp.linalg.norm(s_col) + 1e-12)
        C.append(c_col)
        S.append(s_col)
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
        data_new = data_one._replace(
            quest_cond_v=u_flat.reshape(data_one.quest_cond_v.shape)
        )
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


@partial(jax.jit, static_argnames=("max_mode",))
def true_mode_gain_by_jvp_u0_fast(u0_flat, c_value, max_mode=40):
    u0_flat = jnp.asarray(u0_flat).reshape(-1)
    n = u0_flat.shape[0]
    f = lambda u: true_advection_map_u0(u, c_value)

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
    f_query = icon_query_map(params, data_one)
    J = jax.jacrev(f_query)(u0_flat)
    n_out, n_in = J.shape
    return jnp.sqrt((1.0 / n_out) / (1.0 / n_in)) * J


def fourier_project_jacobian(J_l2: jnp.ndarray, max_pair_mode: int = 16):
    n_out, n_in = J_l2.shape
    Phi = paired_fourier_basis(n_in, max_pair_mode)
    Psi = paired_fourier_basis(n_out, max_pair_mode)
    return Psi.T @ J_l2 @ Phi


def cumulative_relative_errors(icon_cs, true_cs, eps=1e-12):
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


# =============================================================================
# 7. Compute paper-style diagnostics
# =============================================================================

def evaluate_prediction(params, data, label):
    pred = predict_batch(params, data)
    target = label[:, 0]
    diff = pred - target
    mse = jnp.mean(diff ** 2)
    rel_l2 = jnp.linalg.norm(diff) / (jnp.linalg.norm(target) + 1e-12)
    return pred, target, mse, rel_l2


def compute_prediction_error_over_c(c_values, eqns_per_c=8, key_seed=4100):
    rows = []
    for i, c_value in enumerate(c_values):
        raw, data, label = make_shift_prompts_at_c(
            jax.random.PRNGKey(key_seed + i),
            c_value=float(c_value),
            num_prompts=eqns_per_c,
            pairs_per_prompt=num_pairs,
        )
        pred, target, mse, rel_l2 = evaluate_prediction(params, data, label)
        rows.append([float(c_value), float(mse), float(rel_l2)])
    return np.asarray(rows)


def compute_context_spectrum(fixed_uq, c_values, num_contexts_per_c=5, max_mode=40, key_seed=2028):
    icon_mean, icon_std, true_gains = [], [], []
    key0 = jax.random.PRNGKey(key_seed)

    for i, c_value in enumerate(c_values):
        key_i = jax.random.fold_in(key0, i)
        raw_i, data_i, _ = make_shift_prompts_at_c(
            key_i,
            c_value=float(c_value),
            num_prompts=num_contexts_per_c,
            pairs_per_prompt=num_pairs,
            fixed_query_u0=fixed_uq,
        )

        icon_runs = []
        for j in range(num_contexts_per_c):
            data_one = tree.tree_map(lambda x: x[j], data_i)
            g_icon = np.asarray(icon_mode_gain_by_jvp_fast(params, data_one, max_mode=max_mode))[:, 1:3]
            icon_runs.append(g_icon)

        icon_runs = np.stack(icon_runs, axis=0)
        icon_mean.append(icon_runs.mean(axis=0))
        icon_std.append(icon_runs.std(axis=0))

        g_true = np.asarray(
            true_mode_gain_by_jvp_u0_fast(jnp.asarray(fixed_uq).reshape(-1), float(c_value), max_mode=max_mode)
        )[:, 1:3]
        true_gains.append(g_true)

    return {
        "k": np.arange(1, max_mode + 1),
        "c_values": np.asarray(c_values, dtype=float),
        "icon_mean": np.stack(icon_mean, axis=0),
        "icon_std": np.stack(icon_std, axis=0),
        "true": np.stack(true_gains, axis=0),
    }


def compute_feature_maps(c_value=0.8, key_seed=8888, max_pair_mode=16):
    raw, data, label = make_shift_prompts_at_c(
        jax.random.PRNGKey(key_seed),
        c_value=float(c_value),
        num_prompts=1,
        pairs_per_prompt=num_pairs,
    )
    data_one = tree.tree_map(lambda x: x[0], data)

    J_icon = icon_jacobian_l2(params, data_one)
    A_icon = np.asarray(fourier_project_jacobian(J_icon, max_pair_mode=max_pair_mode))
    A_true = exact_fourier_map(float(c_value), max_pair_mode=max_pair_mode)

    rel_frob = np.linalg.norm(A_icon - A_true, ord="fro") / (np.linalg.norm(A_true, ord="fro") + 1e-12)
    return {
        "c_value": float(c_value),
        "A_true": A_true,
        "A_icon": A_icon,
        "A_error": A_icon - A_true,
        "rel_frob": rel_frob,
    }


def compute_phase_block_error(c_values, max_pair_mode=16, contexts_per_c=3, key_seed=9991):
    rows = []
    for ci, c_value in enumerate(c_values):
        block_err_runs = []
        for r in range(contexts_per_c):
            raw, data, label = make_shift_prompts_at_c(
                jax.random.PRNGKey(key_seed + 1000 * ci + r),
                c_value=float(c_value),
                num_prompts=1,
                pairs_per_prompt=num_pairs,
            )
            data_one = tree.tree_map(lambda x: x[0], data)
            J_icon = icon_jacobian_l2(params, data_one)
            A_icon = np.asarray(fourier_project_jacobian(J_icon, max_pair_mode=max_pair_mode))
            A_true = exact_fourier_map(float(c_value), max_pair_mode=max_pair_mode)
            errs = []
            for k_mode in range(1, max_pair_mode + 1):
                i = 2 * (k_mode - 1)
                Bi = A_icon[i:i + 2, i:i + 2]
                Bt = A_true[i:i + 2, i:i + 2]
                errs.append(np.linalg.norm(Bi - Bt, ord="fro") / (np.linalg.norm(Bt, ord="fro") + 1e-12))
            block_err_runs.append(errs)
        rows.append(np.mean(np.asarray(block_err_runs), axis=0))
    return {
        "c_values": np.asarray(c_values, dtype=float),
        "k": np.arange(1, max_pair_mode + 1),
        "block_error": np.stack(rows, axis=0),
    }


def compute_all_diagnostics():
    results = {}

    # Fixed query used for context spectrum.
    raw_ref, data_ref, label_ref = make_shift_prompts_at_c(
        jax.random.PRNGKey(1234),
        c_value=0.8,
        num_prompts=8,
        pairs_per_prompt=num_pairs,
    )
    data_one = tree.tree_map(lambda x: x[0], data_ref)
    fixed_uq = np.asarray(data_one.quest_cond_v[0, :, 0])

    # Representative in-distribution prediction at c=0.8.
    pred_ref, target_ref, mse_ref, rel_l2_ref = evaluate_prediction(params, data_ref, label_ref)
    results["representative_prediction"] = {
        "x": np.asarray(data_ref.quest_qoi_k[0, 0, :, 0]),
        "c": float(raw_ref.c[0]),
        "shift": float(raw_ref.shift[0]),
        "u0": np.asarray(data_ref.quest_cond_v[0, 0, :, 0]),
        "true": np.asarray(target_ref[0, :, 0]),
        "pred": np.asarray(pred_ref[0, :, 0]),
        "mse": float(mse_ref),
        "rel_l2": float(rel_l2_ref),
    }

    results["prediction_error"] = compute_prediction_error_over_c(
        paper_c_values,
        eqns_per_c=8,
        key_seed=4100,
    )

    results["query_context"] = compute_context_spectrum(
        fixed_uq=fixed_uq,
        c_values=paper_c_values,
        num_contexts_per_c=paper_contexts_per_c,
        max_mode=paper_max_mode,
        key_seed=2028,
    )

    results["feature_map"] = compute_feature_maps(
        c_value=0.8,
        key_seed=8888,
        max_pair_mode=paper_feature_pair_modes,
    )

    results["phase_block_error"] = compute_phase_block_error(
        paper_c_values_id,
        max_pair_mode=paper_feature_pair_modes,
        contexts_per_c=3,
        key_seed=9991,
    )

    return results


# =============================================================================
# 8. Plotting
# =============================================================================

def save_or_show(path=None):
    """Save and show the current figure without switching layout engines.

    Important: some figures use constrained_layout=True and contain colorbars.
    Calling plt.tight_layout() after a colorbar has been created can trigger
    Matplotlib's error:
        RuntimeError: Colorbar layout of new layout engine not compatible with old engine
    Therefore this helper never calls tight_layout().  The combination of
    constrained_layout=True where needed and bbox_inches="tight" during saving
    is sufficient for the paper figures.
    """
    fig = plt.gcf()
    if save_figs and path is not None:
        fig.savefig(path, dpi=260, bbox_inches="tight")
        if path.endswith(".png"):
            fig.savefig(path[:-4] + ".pdf", bbox_inches="tight")
    plt.show()


def plot_representative_prediction(results):
    r = results["representative_prediction"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)

    axes[0].plot(r["x"], r["u0"], "k-")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$u_q(x)$")
    axes[0].grid(True)
    axes[0].text(0.04, 0.90, "query input", transform=axes[0].transAxes,
                 ha="left", va="top", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    axes[1].plot(r["x"], r["true"], "k-", label="true")
    axes[1].plot(r["x"], r["pred"], "r--", label="ICON")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$u(T,x)$")
    axes[1].grid(True)
    axes[1].legend(frameon=True)
    axes[1].text(0.04, 0.90, rf"$c={r['c']:.2f}$, shift={r['shift']:.2f}\nRel-$L^2={r['rel_l2']:.2e}$",
                 transform=axes[1].transAxes, ha="left", va="top",
                 bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    if save_figs:
        png_path = os.path.join(fig_dir, "advection_representative_prediction.png")
        fig.savefig(png_path, dpi=260, bbox_inches="tight")
        fig.savefig(png_path[:-4] + ".pdf", bbox_inches="tight")
    plt.show()


def plot_prediction_error(results):
    arr = results["prediction_error"]
    c_values = arr[:, 0]
    rel_l2 = arr[:, 2]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.semilogy(c_values, rel_l2, "o-", label=r"relative $L^2$")
    ax.axvspan(c_range[0], c_range[1], alpha=0.18, label="training range")
    ax.set_xlabel(r"hidden speed $c$")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.grid(True, which="both")
    for regime_ax in np.asarray(ax, dtype=object).ravel():
        style_parameter_sweep(regime_ax, 'advection')
    ax.legend(frameon=True)
    add_regime_key(plt.gcf())
    save_or_show(comparison_filename(os.path.join(fig_dir, "advection_relative_prediction_error_over_hidden_speed.png")))


def plot_query_spectrum_with_cumulative_error(results):
    ctx = results["query_context"]
    k = ctx["k"]
    c_values = ctx["c_values"]
    icon_mean = ctx["icon_mean"]
    true = ctx["true"]
    cum_cos, cum_sin, cum_all = cumulative_relative_errors(icon_mean, true)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex=True)

    for i, c_value in enumerate(c_values):
        axes[0, 0].plot(k, icon_mean[i, :, 0], "o-", label=f"{c_value:g}")
        axes[0, 1].plot(k, icon_mean[i, :, 1], "s-", label=f"{c_value:g}")
        axes[1, 0].plot(k, cum_cos[i], "o-")
        axes[1, 1].plot(k, cum_sin[i], "s-")

    for ax in axes.ravel():
        ax.axvline(n_modes, linestyle="--", color="k")
        ax.grid(True, which="both")

    axes[0, 0].set_ylabel("ICON Jacobian gain")
    axes[1, 0].set_ylabel("cumulative relative error")
    axes[0, 0].text(0.04, 0.90, "cos", transform=axes[0, 0].transAxes,
                    ha="left", va="top", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    axes[0, 1].text(0.04, 0.90, "sin", transform=axes[0, 1].transAxes,
                    ha="left", va="top", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    axes[1, 0].text(0.04, 0.90, "cos", transform=axes[1, 0].transAxes,
                    ha="left", va="top", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    axes[1, 1].text(0.04, 0.90, "sin", transform=axes[1, 1].transAxes,
                    ha="left", va="top", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    axes[1, 0].set_xlabel(r"Fourier mode $K$")
    axes[1, 1].set_xlabel(r"Fourier mode $K$")
    from plotting.regimes import keep_training_parameter_curves
    visible_c_values = keep_training_parameter_curves(axes, c_values, "advection")
    style_parameter_curves(axes, visible_c_values, "advection")
    axes[0, 1].legend(title=r"training-range speed $c$", ncol=2, frameon=True)

    save_or_show(comparison_filename(os.path.join(fig_dir, "advection_query_spectrum_training_range.png")))

    return cum_cos, cum_sin, cum_all


def plot_phase_block_error(results):
    r = results["phase_block_error"]
    c_values = r["c_values"]
    k = r["k"]
    block_error = r["block_error"]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, c_value in enumerate(c_values):
        ax.semilogy(k, block_error[i], "o-", label=f"{c_value:g}")
    ax.axvline(n_modes, linestyle="--", color="k")
    ax.set_xlabel(r"Fourier mode $k$")
    ax.set_ylabel("relative block-rotation error")
    ax.grid(True, which="both")
    ax.legend(title=r"hidden speed $c$", ncol=2, frameon=True)
    save_or_show(os.path.join(fig_dir, "advection_phase_block_error_over_c.png"))


def plot_feature_map(results):
    fm = results["feature_map"]
    A_true = fm["A_true"]
    A_icon = fm["A_icon"]
    A_err_abs = np.abs(fm["A_error"])
    c_value = fm["c_value"]

    vmax = max(float(np.max(np.abs(A_true))), float(np.max(np.abs(A_icon))), 1e-12)
    err_vmax = max(float(np.max(A_err_abs)), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), constrained_layout=True)
    panels = [
        (A_true, "true", vmax, "coolwarm", -vmax, vmax),
        (A_icon, "ICON", vmax, "coolwarm", -vmax, vmax),
        (A_err_abs, rf"abs. error, Rel-F={fm['rel_frob']:.2e}", err_vmax, "magma", 0.0, err_vmax),
    ]

    for ax, (M, label, vm, cmap, vmin, vmax_i) in zip(axes, panels):
        im = ax.imshow(M, origin="lower", aspect="equal", interpolation="nearest",
                       cmap=cmap, vmin=vmin, vmax=vmax_i)
        ax.set_xlabel("input Fourier index")
        ax.set_ylabel("output Fourier index")
        ax.text(0.03, 0.94, label, transform=ax.transAxes,
                ha="left", va="top",
                bbox=dict(facecolor="white", alpha=0.78, edgecolor="none"))
        fig.colorbar(im, ax=ax, shrink=0.82)

    save_or_show(os.path.join(fig_dir, "advection_fourier_feature_maps_true_icon_absolute_error.png"))


def save_numeric_results(results, cum_errors):
    ctx = results["query_context"]
    fm = results["feature_map"]
    phase = results["phase_block_error"]
    np.savez(
        os.path.join(fig_dir, "advection_aligned_paper_style_diagnostics_results.npz"),
        prediction_error=results["prediction_error"],
        context_k=ctx["k"],
        context_c_values=ctx["c_values"],
        context_icon_mean=ctx["icon_mean"],
        context_icon_std=ctx["icon_std"],
        context_true=ctx["true"],
        query_cum_rel_err_cos=cum_errors[0],
        query_cum_rel_err_sin=cum_errors[1],
        query_cum_rel_err_all=cum_errors[2],
        feature_c_value=np.asarray(fm["c_value"]),
        A_true=fm["A_true"],
        A_icon=fm["A_icon"],
        A_error=fm["A_error"],
        A_rel_frob=np.asarray(fm["rel_frob"]),
        phase_c_values=phase["c_values"],
        phase_k=phase["k"],
        phase_block_error=phase["block_error"],
    )


def plot_all_figures(results):
    plot_representative_prediction(results)
    plot_prediction_error(results)
    cum_errors = plot_query_spectrum_with_cumulative_error(results)
    plot_phase_block_error(results)
    plot_feature_map(results)
    save_numeric_results(results, cum_errors)


# =============================================================================
# 9. Eval driver and CLI
# =============================================================================

def eval_advection(ckpt_path=None):
    global params
    if ckpt_path is not None:
        params = load_checkpoint(ckpt_path)

    print("Computing advection paper-style diagnostics...")
    results = compute_all_diagnostics()
    plot_all_figures(results)
    print("Saved complete advection figures and arrays to:", fig_dir)
    print("Expected figure files:")
    for name in [
        "advection_representative_prediction.png",
        "advection_relative_prediction_error_over_hidden_speed.png",
        "advection_query_spectrum_over_c_with_cumulative_relative_error.png",
        "advection_phase_block_error_over_c.png",
        "advection_fourier_feature_maps_true_icon_absolute_error.png",
    ]:
        print("  -", os.path.join(fig_dir, name))



# =============================================================================
# Notebook-safe entry point: no argparse, so it will not crash in Jupyter/IPython.
# Edit these two variables if needed.
# =============================================================================
RUN_MODE = "eval"   # "eval", "train", or "train_eval"
CKPT_PATH = None    # e.g. str(ADVECTION_CHECKPOINT_DIR / "params_step_200000.msgpack")
                  # If None in eval mode, the latest params_step_*.msgpack is used.


def main_no_argparse():
    if RUN_MODE == "train":
        train_advection()
    elif RUN_MODE == "eval":
        ckpt_path = CKPT_PATH if CKPT_PATH is not None else find_latest_checkpoint(ckpt_dir)
        eval_advection(ckpt_path)
    elif RUN_MODE == "train_eval":
        final_ckpt = train_advection()
        eval_advection(final_ckpt)
    else:
        raise ValueError(f"Unknown RUN_MODE={RUN_MODE!r}; use 'eval', 'train', or 'train_eval'.")


if __name__ == "__main__":
    main_no_argparse()
