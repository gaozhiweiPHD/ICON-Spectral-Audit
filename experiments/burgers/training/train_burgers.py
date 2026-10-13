# -*- coding: utf-8 -*-
"""
Offline prompt-based ICON training for 1D viscous Burgers equation.

PDE:
    u_t + u u_x = c u_xx,  x in [0, 1] periodic.

Task:
    Each context corresponds to one hidden viscosity c.
    The model never receives c. It sees only several condition--QoI pairs
        u_i(0, x) -> u_i(T, x)
    generated with the same c, and must predict u_q(T, x) for a new query
    initial condition u_q(0, x).

This is the Burgers analogue of the hidden-shift advection ICON experiment.
The data are generated offline first, then prompts are randomly assembled from
that offline pool during training. This mimics the setting of expensive PDE data.

Solver:
    Periodic pseudo-spectral RK4 solver with 2/3 dealiasing.
    The time step should be chosen conservatively for small viscosity / steep data.
    For very small c, increase n_solver_steps and/or solver_grid.

Run:
    Put this file in the same folder as models_utils.py, then run
        python train_icon_burgers_viscosity_offline_jax.py

Required packages:
    jax, flax, optax, einshape, matplotlib, numpy
"""

from __future__ import annotations

from collections import namedtuple
from typing import Tuple
import os
import sys
from pathlib import Path

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
from common.project import BURGERS_CHECKPOINT_DIR, BURGERS_FIGURE_DIR
from plotting.style import apply_paper_style as apply_shared_plot_style
from plotting.regimes import style_parameter_curves, style_parameter_sweep, add_regime_key, comparison_filename


# =============================================================================
# 0. User configuration
# =============================================================================

seed = 0

# Prompt structure.
num_pairs = 6          # 5 demo pairs + 1 question pair
demo_num = 5
n_grid = 100           # tokens per condition/QoI function

# Random initial condition settings.
n_modes = 16
fourier_decay = 1.0
bias_scale = 0.0
initial_amp = 0.7      # smaller is easier / safer for small viscosity
normalize_each_function = False

# Burgers hidden viscosity distribution. The model never receives c.
# For a first run, use (0.005, 0.05). For harder nearly-inviscid cases,
# try (0.001, 0.02) and increase n_solver_steps / solver_grid.
T_final = 0.25
c_range: Tuple[float, float] = (0.005, 0.05)

# Offline dataset size.
offline_train = True
num_train_operators = 2000     # number of hidden c values in offline pool
pairs_per_operator = 20        # input-output pairs per c
num_val_operators = 128

# Solver resolution and accuracy.
# solver_grid is the actual PDE grid. If equal to n_grid, no downsampling is needed.
solver_grid = n_grid
n_solver_steps = 800           # increase for smaller c or sharper initial data
generation_chunk_ops = 64      # generate offline pool in operator chunks

# Training setup.
batch_size = 4
num_steps = 200000
print_every = 2000
learning_rate = 2e-4
weight_decay = 1e-4
warmup_steps = 2000
clip_norm = 1.0
val_eval_batch_size = 1

save_every = 10000
ckpt_dir = str(BURGERS_CHECKPOINT_DIR)
os.makedirs(ckpt_dir, exist_ok=True)

MODEL_SIZE = "paper_like"  # options: 'paper_like', 'small'

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


# =============================================================================
# 1. Data containers
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

RawBurgers = namedtuple(
    "RawBurgers",
    ["cond_k", "cond_v", "qoi_k", "qoi_v", "c", "coeff_sin", "coeff_cos", "bias"],
)


# =============================================================================
# 2. Random initial functions
# =============================================================================

def eval_random_fourier(
    x: jnp.ndarray,
    coeff_sin: jnp.ndarray,
    coeff_cos: jnp.ndarray,
    bias: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate random periodic functions."""
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


# =============================================================================
# 3. Periodic Burgers solver
# =============================================================================

def _burgers_rhs(u: jnp.ndarray, c: jnp.ndarray, k: jnp.ndarray, dealias: jnp.ndarray) -> jnp.ndarray:
    """
    RHS for u_t + u u_x = c u_xx using conservative spectral form:
        u_t = -0.5 * d_x(u^2) + c u_xx.

    u: [B, N]
    c: [B]
    k: [N], angular wave numbers 2*pi*m
    dealias: [N] boolean / float mask for 2/3 dealiasing.
    """
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
    """
    Solve a batch of periodic viscous Burgers equations.

    u0: [B, N]
    c:  [B]
    return u(T): [B, N]

    Notes:
      - Explicit RK4 is simple and robust for the moderate settings here.
      - For smaller viscosity or sharper profiles, increase nsteps and/or N.
    """
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
# 4. Offline data generation
# =============================================================================

def generate_burgers_raw_data(
    key: jax.Array,
    num_operators: int,
    pairs_per_operator: int,
    n_grid: int = 100,
    n_modes: int = 8,
    c_range: Tuple[float, float] = (0.005, 0.05),
    T_final: float = 0.25,
    fourier_decay: float = 1.0,
    bias_scale: float = 0.0,
    initial_amp: float = 0.7,
    n_solver_steps: int = 800,
) -> RawBurgers:
    """
    Generate an offline pool:
        [num_operators, pairs_per_operator] input-output pairs.

    Each operator has one hidden viscosity c. All pairs under the same operator
    use the same c but different initial functions.
    """
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


def concat_raw_burgers(raw_list: list[RawBurgers]) -> RawBurgers:
    return RawBurgers(
        cond_k=jnp.concatenate([r.cond_k for r in raw_list], axis=0),
        cond_v=jnp.concatenate([r.cond_v for r in raw_list], axis=0),
        qoi_k=jnp.concatenate([r.qoi_k for r in raw_list], axis=0),
        qoi_v=jnp.concatenate([r.qoi_v for r in raw_list], axis=0),
        c=jnp.concatenate([r.c for r in raw_list], axis=0),
        coeff_sin=jnp.concatenate([r.coeff_sin for r in raw_list], axis=0),
        coeff_cos=jnp.concatenate([r.coeff_cos for r in raw_list], axis=0),
        bias=jnp.concatenate([r.bias for r in raw_list], axis=0),
    )


def generate_burgers_pool_in_chunks(key: jax.Array, num_operators: int, pairs_per_operator: int) -> RawBurgers:
    """Generate offline pool in chunks to avoid memory spikes."""
    raw_chunks = []
    n_done = 0
    chunk_id = 0
    while n_done < num_operators:
        n_chunk = min(generation_chunk_ops, num_operators - n_done)
        key_i = jax.random.fold_in(key, chunk_id)
        raw_i = generate_burgers_raw_data(
            key_i,
            num_operators=n_chunk,
            pairs_per_operator=pairs_per_operator,
            n_grid=n_grid,
            n_modes=n_modes,
            c_range=c_range,
            T_final=T_final,
            fourier_decay=fourier_decay,
            bias_scale=bias_scale,
            initial_amp=initial_amp,
            n_solver_steps=n_solver_steps,
        )
        raw_chunks.append(jax.device_get(raw_i))
        n_done += n_chunk
        chunk_id += 1
        print(f"  generated {n_done}/{num_operators} operators")
    return concat_raw_burgers(raw_chunks)


# =============================================================================
# 5. ICON data construction and offline prompt sampling
# =============================================================================

def build_icon_data(raw: RawBurgers, demo_num: int = 5) -> tuple[Data, jnp.ndarray]:
    """
    Convert a raw prompt batch into ICON Data.
    raw fields should have shape [bs, num_pairs, n_grid, ...].
    """
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
def sample_offline_prompt_batch(raw_pool: RawBurgers, key: jax.Array) -> tuple[Data, jnp.ndarray]:
    """
    JIT prompt assembly from offline pool.

    Choose batch_size hidden operators, then choose num_pairs stored pairs per
    operator. First demo_num are demos and the last one is the query.
    """
    key_op, key_pair = jax.random.split(key)
    num_ops = raw_pool.cond_k.shape[0]
    pp = raw_pool.cond_k.shape[1]

    op_idx = jax.random.randint(key_op, shape=(batch_size,), minval=0, maxval=num_ops)

    # Randomly select num_pairs distinct-ish pairs by sorting random scores.
    scores = jax.random.uniform(key_pair, shape=(batch_size, pp))
    pair_idx = jnp.argsort(scores, axis=1)[:, :num_pairs]

    def gather_field(field):
        selected = field[op_idx]  # [bs, pp, n_grid, d]
        return jnp.take_along_axis(selected, pair_idx[:, :, None, None], axis=1)

    raw_batch = RawBurgers(
        cond_k=gather_field(raw_pool.cond_k),
        cond_v=gather_field(raw_pool.cond_v),
        qoi_k=gather_field(raw_pool.qoi_k),
        qoi_v=gather_field(raw_pool.qoi_v),
        c=raw_pool.c[op_idx],
        coeff_sin=jnp.zeros((batch_size, num_pairs, n_modes), dtype=jnp.float32),
        coeff_cos=jnp.zeros((batch_size, num_pairs, n_modes), dtype=jnp.float32),
        bias=jnp.zeros((batch_size, num_pairs), dtype=jnp.float32),
    )
    return build_icon_data(raw_batch, demo_num=demo_num)


# =============================================================================
# 6. ICON model
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
        x = nn.Dense(features=self.out_dim)(x)
        return x


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
            dense_block = MLP(
                hidden_dim=self.model_dim * self.widening_factor,
                out_dim=self.model_dim,
                depth=1,
            )
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
        cond_len_list, qoi_kv_len_list, qoi_k_len_list,
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
        self.pre_projection = nn.Dense(
            self.config["transformer"]["model_dim"], name="pre_projection"
        )
        self.func_pos_embedding = nn.Embed(
            self.config["demo_max_num"] * 3,
            self.config["transformer"]["model_dim"],
            name="func_pos_embedding",
        )
        self.transformer = SelfAttnTransformer(
            **self.config["transformer"], name="transformer"
        )
        self.post_projection = nn.Dense(self.config["out_dim"], name="post_projection")

    def basic_forward(self, data, mode, index_pos, basic_mask):
        demo_num_local = len(data.demo_cond_k)
        cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list = mu.build_bool_sequence(
            demo_num_local, mode=mode, shot_num_min=0
        )
        sequence = mu.build_data_sequence(
            data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list
        )
        token_mask = mu.build_data_mask(
            data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list
        )

        sequence = self.pre_projection(sequence)
        sequence = sequence + self.func_pos_embedding(index_pos)

        token_mask = einshape("i->ji", token_mask, j=sequence.shape[0])
        attn_mask = token_mask * basic_mask
        attn_mask = einshape("ij->mij", attn_mask, m=self.config["transformer"]["n_heads"])

        sequence = self.transformer(sequence, mask=attn_mask)
        sequence = self.post_projection(sequence)
        return sequence

    def __call__(self, data):
        sequence = self.basic_forward(data, "train", self.index_pos, self.basic_mask)
        return sequence[self.out_mask]

    def predict(self, data):
        data_shape = tree.tree_map(lambda x: x.shape, data)
        basic_mask, index_pos, _ = build_matrices_from_data_shape(
            data_shape, mode="test", shot_num_min=0
        )
        sequence = self.basic_forward(data, "test", index_pos, basic_mask)
        return sequence[-data.quest_qoi_mask.shape[-1]:, :]


# =============================================================================
# 7. Training utilities
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


# =============================================================================
# 8. Build offline pools, model, optimizer
# =============================================================================

key = jax.random.PRNGKey(seed)
key_init, key_train_pool, key_val_pool, key_loop = jax.random.split(key, 4)

print("Generating offline training pool...")
raw_train_pool = generate_burgers_pool_in_chunks(
    key_train_pool,
    num_operators=num_train_operators,
    pairs_per_operator=pairs_per_operator,
)
raw_train_pool = tree.tree_map(jax.device_put, raw_train_pool)
print(
    f"Offline train pool: {num_train_operators} operators x "
    f"{pairs_per_operator} pairs/operator = "
    f"{num_train_operators * pairs_per_operator} function pairs."
)

print("Generating fixed validation pool...")
raw_val_pool = generate_burgers_pool_in_chunks(
    key_val_pool,
    num_operators=num_val_operators,
    pairs_per_operator=num_pairs,
)
val_data, val_label = build_icon_data(raw_val_pool, demo_num=demo_num)
val_data = tree.tree_map(jax.device_put, val_data)
val_label = jax.device_put(val_label)

sample_data = tree.tree_map(lambda x: x[0], val_data)
sample_shape = tree.tree_map(lambda x: x.shape, sample_data)
basic_mask, index_pos, out_mask = build_matrices_from_data_shape(
    sample_shape, mode="train", shot_num_min=0
)

icon_model = ICON_LM(
    model_config,
    basic_mask=basic_mask,
    index_pos=index_pos,
    out_mask=out_mask,
)
params = icon_model.init({"params": key_init}, sample_data)


def forward_one(p, data_one):
    return icon_model.apply(p, data_one)


def predict_one(p, data_one):
    return icon_model.apply(p, data_one, method=ICON_LM.predict)


forward_batch = jax.vmap(forward_one, in_axes=(None, 0))
predict_batch = jax.vmap(predict_one, in_axes=(None, 0))

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


def save_params(params, step):
    path = os.path.join(ckpt_dir, f"params_step_{step:06d}.msgpack")
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(params))
    return path


# =============================================================================
# 9. Training loop
# =============================================================================

Loss = []
ValMSE = []
ValRelL2 = []

print("Configuration:")
print(f"  PDE: u_t + u u_x = c u_xx, periodic")
print(f"  c_range={c_range}, T_final={T_final}")
print(f"  n_grid={n_grid}, n_modes={n_modes}, n_solver_steps={n_solver_steps}")
print(f"  offline operators={num_train_operators}, pairs/operator={pairs_per_operator}")
print(f"  batch_size={batch_size}, num_steps={num_steps}")
print("Start training...\n")

for step in range(num_steps + 1):
    key_loop, key_batch = jax.random.split(key_loop)

    batch_data, batch_label = sample_offline_prompt_batch(raw_train_pool, key_batch)
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
        path = save_params(params, step)
        print(f"Saved checkpoint to {path}")

final_path = save_params(params, num_steps)
print(f"Final checkpoint saved to {final_path}")



# =============================================================================
# 10. Clean diagnostics and plotting
# =============================================================================

# All figures from this section are saved to the same folder.
fig_dir = str(BURGERS_FIGURE_DIR)
os.makedirs(fig_dir, exist_ok=True)

RUN_FINAL_SPECTRUM_DIAGNOSTICS = True


def set_clean_plot_style():
    apply_shared_plot_style()
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
        "axes.linewidth": 1.0,
        "grid.alpha": 0.35,
        "figure.dpi": 120,
    })


set_clean_plot_style()


def save_current_figure(filename: str):
    path = os.path.join(fig_dir, filename)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    if filename.endswith("_regime_style.png"):
        plt.savefig(path[:-4] + ".pdf", bbox_inches="tight")
    return path


# -----------------------------------------------------------------------------
# Prompt generation at prescribed viscosity and/or prescribed query state
# -----------------------------------------------------------------------------

def make_burgers_prompts_at_c(
    key: jax.Array,
    c_value: float,
    num_prompts: int,
    pairs_per_prompt: int = num_pairs,
    fixed_query_u0: jnp.ndarray | None = None,
) -> tuple[RawBurgers, Data, jnp.ndarray]:
    """
    Generate prompts with prescribed hidden viscosity c.

    If fixed_query_u0 is provided, the query initial condition in every prompt is
    replaced by this same function. The demo pairs are still random and encode
    the chosen viscosity through the context.
    """
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
            raise NotImplementedError(
                "fixed_query_u0 replacement assumes normalize_each_function=False."
            )
        uq = jnp.asarray(fixed_query_u0).reshape(n_grid)
        uq_batch = jnp.broadcast_to(uq[None, :], (num_prompts, n_grid))
        c_batch = raw.c
        uT_batch = solve_burgers_batch_jit(
            uq_batch,
            c_batch,
            T_final,
            nsteps=n_solver_steps,
        )
        raw = raw._replace(
            cond_v=raw.cond_v.at[:, demo_num, :, 0].set(uq_batch),
            qoi_v=raw.qoi_v.at[:, demo_num, :, 0].set(uT_batch),
        )

    data, label = build_icon_data(raw, demo_num=demo_num)
    return raw, data, label


# -----------------------------------------------------------------------------
# Fourier bases and spectra
# -----------------------------------------------------------------------------

def fourier_basis(n: int, kmax: int):
    x = jnp.arange(n, dtype=jnp.float32) / n
    cols = [jnp.ones_like(x)]
    k_mode = 1
    while len(cols) < kmax:
        cols.append(jnp.sqrt(2.0) * jnp.cos(2.0 * jnp.pi * k_mode * x))
        if len(cols) < kmax:
            cols.append(jnp.sqrt(2.0) * jnp.sin(2.0 * jnp.pi * k_mode * x))
        k_mode += 1
    B = jnp.stack(cols[:kmax], axis=1)
    Q, _ = jnp.linalg.qr(B)
    return Q[:, :kmax]


def fourier_cos_sin_bases(n: int, max_mode: int):
    """Euclidean-orthonormal real cosine/sine bases, no constant mode."""
    x = jnp.arange(n, dtype=jnp.float32) / n
    C, S = [], []
    for k_mode in range(1, max_mode + 1):
        c_col = jnp.sqrt(2.0) * jnp.cos(2.0 * jnp.pi * k_mode * x)
        s_col = jnp.sqrt(2.0) * jnp.sin(2.0 * jnp.pi * k_mode * x)
        C.append(c_col / (jnp.linalg.norm(c_col) + 1e-12))
        S.append(s_col / (jnp.linalg.norm(s_col) + 1e-12))
    return jnp.stack(C, axis=1), jnp.stack(S, axis=1)


def effective_rank(s, eps=1e-12):
    return (jnp.sum(s) ** 2) / (jnp.sum(s ** 2) + eps)


def icon_query_map(params, data_one):
    def f_query(u_flat):
        data_new = data_one._replace(
            quest_cond_v=u_flat.reshape(data_one.quest_cond_v.shape)
        )
        return predict_one(params, data_new).reshape(-1)
    return f_query


def true_burgers_map_u0(u0_flat: jnp.ndarray, c_value: float | jnp.ndarray) -> jnp.ndarray:
    """True Burgers solution map u0 -> u(T) at fixed viscosity c."""
    u0 = u0_flat.reshape(1, n_grid)
    c_arr = jnp.asarray(c_value, dtype=jnp.float32).reshape(1)
    uT = solve_burgers_batch_jit(u0, c_arr, T_final, nsteps=n_solver_steps)
    return uT.reshape(-1)


def mode_gain_icon_cos_sin(params, data_one, max_mode=40):
    """Return [k, cos_gain, sin_gain] for ICON Jacobian."""
    u0_flat = data_one.quest_cond_v.reshape(-1)
    f_query = icon_query_map(params, data_one)
    n = u0_flat.shape[0]
    x = jnp.arange(n, dtype=jnp.float32) / n
    rows = []
    for k_mode in range(1, max_mode + 1):
        phi_cos = jnp.sqrt(2.0) * jnp.cos(2.0 * jnp.pi * k_mode * x)
        phi_sin = jnp.sqrt(2.0) * jnp.sin(2.0 * jnp.pi * k_mode * x)
        _, y_cos = jax.jvp(f_query, (u0_flat,), (phi_cos,))
        _, y_sin = jax.jvp(f_query, (u0_flat,), (phi_sin,))
        rows.append([
            k_mode,
            jnp.linalg.norm(y_cos) / (jnp.linalg.norm(phi_cos) + 1e-12),
            jnp.linalg.norm(y_sin) / (jnp.linalg.norm(phi_sin) + 1e-12),
        ])
    return jnp.asarray(rows)


def mode_gain_true_cos_sin(u0_flat, c_value, max_mode=40):
    """Return [k, cos_gain, sin_gain] for the true Burgers tangent map."""
    u0_flat = jnp.asarray(u0_flat).reshape(-1)
    f_true = lambda u: true_burgers_map_u0(u, c_value)
    n = u0_flat.shape[0]
    x = jnp.arange(n, dtype=jnp.float32) / n
    rows = []
    for k_mode in range(1, max_mode + 1):
        phi_cos = jnp.sqrt(2.0) * jnp.cos(2.0 * jnp.pi * k_mode * x)
        phi_sin = jnp.sqrt(2.0) * jnp.sin(2.0 * jnp.pi * k_mode * x)
        _, y_cos = jax.jvp(f_true, (u0_flat,), (phi_cos,))
        _, y_sin = jax.jvp(f_true, (u0_flat,), (phi_sin,))
        rows.append([
            k_mode,
            jnp.linalg.norm(y_cos) / (jnp.linalg.norm(phi_cos) + 1e-12),
            jnp.linalg.norm(y_sin) / (jnp.linalg.norm(phi_sin) + 1e-12),
        ])
    return jnp.asarray(rows)


def jacobian_spectrum_icon(params, data_one, basis_size=41):
    u0_flat = data_one.quest_cond_v.reshape(-1)
    J = jax.jacrev(icon_query_map(params, data_one))(u0_flat)
    n_out, n_in = J.shape
    J_l2 = jnp.sqrt((1.0 / n_out) / (1.0 / n_in)) * J
    Phi = fourier_basis(n_in, min(basis_size, n_in))
    Psi = fourier_basis(n_out, min(basis_size, n_out))
    A = Psi.T @ J_l2 @ Phi
    return {
        "J": J,
        "J_l2": J_l2,
        "A_fourier": A,
        "singular_values_basis": jnp.linalg.svd(A, compute_uv=False),
        "singular_values_full": jnp.linalg.svd(J_l2, compute_uv=False),
    }


def jacobian_spectrum_true(u0_flat, c_value, basis_size=41):
    f_true = lambda u: true_burgers_map_u0(u, c_value)
    J = jax.jacrev(f_true)(jnp.asarray(u0_flat).reshape(-1))
    n_out, n_in = J.shape
    J_l2 = jnp.sqrt((1.0 / n_out) / (1.0 / n_in)) * J
    Phi = fourier_basis(n_in, min(basis_size, n_in))
    Psi = fourier_basis(n_out, min(basis_size, n_out))
    A = Psi.T @ J_l2 @ Phi
    return {
        "J": J,
        "J_l2": J_l2,
        "A_fourier": A,
        "singular_values_basis": jnp.linalg.svd(A, compute_uv=False),
        "singular_values_full": jnp.linalg.svd(J_l2, compute_uv=False),
    }


def fourier_block_projection(J_l2: jnp.ndarray, max_mode: int = 16):
    n_out, n_in = J_l2.shape
    C_in, S_in = fourier_cos_sin_bases(n_in, max_mode)
    C_out, S_out = fourier_cos_sin_bases(n_out, max_mode)
    return {
        "cc": C_out.T @ J_l2 @ C_in,
        "cs": C_out.T @ J_l2 @ S_in,
        "sc": S_out.T @ J_l2 @ C_in,
        "ss": S_out.T @ J_l2 @ S_in,
    }


def true_burgers_sensitivity_c(u0_flat, c_value):
    """d/dc G_c^T(u0), represented as an output function."""
    u0_flat = jnp.asarray(u0_flat).reshape(-1)
    f_c = lambda c_scalar: true_burgers_map_u0(u0_flat, c_scalar)
    return jax.jacfwd(f_c)(jnp.asarray(c_value, dtype=jnp.float32)).reshape(-1)


def fourier_amplitudes_of_function(y, max_mode=40):
    y = jnp.asarray(y).reshape(-1)
    n = y.shape[0]
    x = jnp.arange(n, dtype=jnp.float32) / n
    rows = []
    for k_mode in range(1, max_mode + 1):
        phi_cos = jnp.sqrt(2.0) * jnp.cos(2.0 * jnp.pi * k_mode * x)
        phi_sin = jnp.sqrt(2.0) * jnp.sin(2.0 * jnp.pi * k_mode * x)
        cc = jnp.dot(y, phi_cos) / (jnp.dot(phi_cos, phi_cos) + 1e-12)
        ss = jnp.dot(y, phi_sin) / (jnp.dot(phi_sin, phi_sin) + 1e-12)
        rows.append([k_mode, jnp.abs(cc), jnp.abs(ss), cc, ss])
    return jnp.asarray(rows)


# -----------------------------------------------------------------------------
# Deterministic base states
# -----------------------------------------------------------------------------

def make_burgers_base_state(kind: str, n_grid: int = 100) -> jnp.ndarray:
    """Return shape [1, n_grid, 1]."""
    x = jnp.linspace(0.0, 1.0, n_grid, endpoint=False, dtype=jnp.float32)
    if kind == "smooth":
        u = jnp.sin(2.0 * jnp.pi * x)
    elif kind == "steep":
        u = jnp.tanh(jnp.sin(2.0 * jnp.pi * x) / 0.22)
    elif kind == "near_shock":
        u = jnp.tanh(jnp.sin(2.0 * jnp.pi * x) / 0.08)
    else:
        raise ValueError(f"Unknown base state kind: {kind}")
    u = u - jnp.mean(u)
    u = initial_amp * u / (jnp.max(jnp.abs(u)) + 1e-12)
    return u[None, :, None].astype(jnp.float32)


def make_prompt_with_base_state(c_value: float, base_kind: str, key_seed: int):
    base_uq = make_burgers_base_state(base_kind, n_grid=n_grid)
    raw, data, label = make_burgers_prompts_at_c(
        jax.random.PRNGKey(key_seed),
        c_value=float(c_value),
        num_prompts=1,
        pairs_per_prompt=num_pairs,
        fixed_query_u0=base_uq.reshape(-1),
    )
    return tree.tree_map(lambda x: x[0], data), raw, label, base_uq


# -----------------------------------------------------------------------------
# Compute diagnostics first, plot afterwards
# -----------------------------------------------------------------------------

def evaluate_prediction(params, data, label):
    pred = predict_batch(params, data)
    target = label[:, 0]
    diff = pred - target
    return pred, target, jnp.mean(diff ** 2), jnp.linalg.norm(diff) / (jnp.linalg.norm(target) + 1e-12)


def compute_context_spectrum_cos_sin(fixed_uq, c_values, num_contexts_per_c=5, max_mode=40, key_seed=2028):
    key0 = jax.random.PRNGKey(key_seed)
    icon_mean, icon_std, true_list = [], [], []
    for i, c_value in enumerate(c_values):
        key_i = jax.random.fold_in(key0, i)
        raw_i, data_i, _ = make_burgers_prompts_at_c(
            key_i,
            c_value=float(c_value),
            num_prompts=num_contexts_per_c,
            pairs_per_prompt=num_pairs,
            fixed_query_u0=fixed_uq,
        )
        gains_c = []
        for j in range(num_contexts_per_c):
            data_one = tree.tree_map(lambda x: x[j], data_i)
            gains_c.append(np.asarray(mode_gain_icon_cos_sin(params, data_one, max_mode=max_mode))[:, 1:3])
        gains_c = np.stack(gains_c, axis=0)  # [R, K, 2]
        icon_mean.append(gains_c.mean(axis=0))
        icon_std.append(gains_c.std(axis=0))
        true_list.append(np.asarray(mode_gain_true_cos_sin(jnp.asarray(fixed_uq).reshape(-1), float(c_value), max_mode=max_mode))[:, 1:3])
    return {
        "k": np.arange(1, max_mode + 1),
        "c_values": np.asarray(c_values, dtype=float),
        "icon_mean": np.stack(icon_mean, axis=0),
        "icon_std": np.stack(icon_std, axis=0),
        "true": np.stack(true_list, axis=0),
    }


def compute_base_state_diagnostics(base_kinds, c_value=0.02, contexts_per_base=3, max_mode=40, key_seed=13579):
    out = {
        "base_kinds": tuple(base_kinds),
        "c_value": float(c_value),
        "k": np.arange(1, max_mode + 1),
        "prediction": {},
        "spectrum": {},
    }
    for bi, base_kind in enumerate(base_kinds):
        icon_runs = []
        data_for_prediction = None
        raw_for_prediction = None
        label_for_prediction = None
        for r in range(contexts_per_base):
            seed_i = key_seed + 1000 * bi + r
            data_one, raw, label, base_uq = make_prompt_with_base_state(c_value, base_kind, seed_i)
            g_icon = mode_gain_icon_cos_sin(params, data_one, max_mode=max_mode)
            icon_runs.append(np.asarray(g_icon)[:, 1:3])
            if r == 0:
                data_for_prediction = data_one
                raw_for_prediction = raw
                label_for_prediction = label
        icon_runs = np.stack(icon_runs, axis=0)
        u0_flat = data_for_prediction.quest_cond_v.reshape(-1)
        g_true = mode_gain_true_cos_sin(u0_flat, c_value, max_mode=max_mode)
        pred = np.asarray(predict_one(params, data_for_prediction)).reshape(-1)
        true = np.asarray(label_for_prediction[0, 0]).reshape(-1)
        x = np.asarray(data_for_prediction.quest_qoi_k[0, :, 0])
        out["prediction"][base_kind] = {
            "x": x,
            "u0": np.asarray(data_for_prediction.quest_cond_v[0, :, 0]),
            "true": true,
            "pred": pred,
        }
        out["spectrum"][base_kind] = {
            "icon_mean": icon_runs.mean(axis=0),
            "icon_std": icon_runs.std(axis=0),
            "true": np.asarray(g_true)[:, 1:3],
        }
    return out


def compute_base_state_viscosity_grid(base_kinds, c_values, contexts_per_case=2, max_mode=30, key_seed=97531):
    out = {
        "base_kinds": tuple(base_kinds),
        "c_values": np.asarray(c_values, dtype=float),
        "k": np.arange(1, max_mode + 1),
        "results": {},
    }
    for bi, base_kind in enumerate(base_kinds):
        for ci, c_value in enumerate(c_values):
            icon_runs = []
            data_ref = None
            for r in range(contexts_per_case):
                seed_i = key_seed + 10000 * bi + 1000 * ci + r
                data_one, _, _, _ = make_prompt_with_base_state(float(c_value), base_kind, seed_i)
                icon_runs.append(np.asarray(mode_gain_icon_cos_sin(params, data_one, max_mode=max_mode))[:, 1:3])
                if data_ref is None:
                    data_ref = data_one
            icon_runs = np.stack(icon_runs, axis=0)
            u0_flat = data_ref.quest_cond_v.reshape(-1)
            g_true = mode_gain_true_cos_sin(u0_flat, float(c_value), max_mode=max_mode)
            out["results"][(base_kind, float(c_value))] = {
                "icon_mean": icon_runs.mean(axis=0),
                "icon_std": icon_runs.std(axis=0),
                "true": np.asarray(g_true)[:, 1:3],
            }
    return out


def compute_final_diagnostics():
    """Compute all numerical arrays first; no plotting is done here."""
    results = {}

    # Training curves.
    results["training"] = {
        "Loss": np.asarray(Loss),
        "ValMSE": np.asarray(ValMSE),
        "ValRelL2": np.asarray(ValRelL2),
        "print_every": print_every,
    }

    # In-distribution test prompt and one reference query.
    raw_test, test_data, test_label = make_burgers_prompts_at_c(
        jax.random.PRNGKey(1234),
        c_value=0.02,
        num_prompts=8,
        pairs_per_prompt=num_pairs,
        fixed_query_u0=None,
    )
    pred, target, mse, rel_l2 = evaluate_prediction(params, test_data, test_label)
    data_one = tree.tree_map(lambda x: x[0], test_data)
    c0 = float(raw_test.c[0])
    fixed_uq = np.asarray(data_one.quest_cond_v[0, :, 0])
    x = np.asarray(data_one.quest_qoi_k[0, :, 0])

    results["prediction_example"] = {
        "x": x,
        "c0": c0,
        "u0": fixed_uq,
        "true": np.asarray(target[0]).reshape(-1),
        "pred": np.asarray(pred[0]).reshape(-1),
        "mse": float(mse),
        "rel_l2": float(rel_l2),
    }

    # Query spectrum: no pair gain; store cos/sin separately.
    max_mode = 40
    g_icon = mode_gain_icon_cos_sin(params, data_one, max_mode=max_mode)
    g_true = mode_gain_true_cos_sin(data_one.quest_cond_v.reshape(-1), c0, max_mode=max_mode)
    results["query_gain"] = {
        "k": np.asarray(g_icon[:, 0]),
        "icon": np.asarray(g_icon[:, 1:3]),
        "true": np.asarray(g_true[:, 1:3]),
    }

    # Fourier-projected singular values.
    spec_true = jacobian_spectrum_true(data_one.quest_cond_v.reshape(-1), c0, basis_size=41)
    spec_icon = jacobian_spectrum_icon(params, data_one, basis_size=41)
    results["singular_values"] = {
        "true": np.asarray(spec_true["singular_values_basis"]),
        "icon": np.asarray(spec_icon["singular_values_basis"]),
    }

    # Viscosity sensitivity.
    du_dc = true_burgers_sensitivity_c(fixed_uq, c0)
    amps_dc = fourier_amplitudes_of_function(du_dc, max_mode=40)
    results["c_sensitivity"] = {
        "du_dc": np.asarray(du_dc),
        "amps": np.asarray(amps_dc),
    }

    # Split Fourier blocks.
    blocks_true = fourier_block_projection(spec_true["J_l2"], max_mode=16)
    blocks_icon = fourier_block_projection(spec_icon["J_l2"], max_mode=16)
    blocks_err = {k: blocks_icon[k] - blocks_true[k] for k in blocks_true}
    results["fourier_blocks"] = {
        "true": {k: np.asarray(v) for k, v in blocks_true.items()},
        "icon": {k: np.asarray(v) for k, v in blocks_icon.items()},
        "error": {k: np.asarray(v) for k, v in blocks_err.items()},
    }

    # OOD context spectrum: no pair gain; store cos/sin separately.
    c_values_ood = [0.002, 0.0035, 0.005, 0.01, 0.02, 0.035, 0.05, 0.07, 0.10]
    results["context"] = compute_context_spectrum_cos_sin(
        fixed_uq=fixed_uq,
        c_values=c_values_ood,
        num_contexts_per_c=5,
        max_mode=40,
        key_seed=2028,
    )

    # Base-state predictions and spectra.
    base_kinds = ("smooth", "steep", "near_shock")
    results["base"] = compute_base_state_diagnostics(
        base_kinds=base_kinds,
        c_value=0.02,
        contexts_per_base=3,
        max_mode=40,
        key_seed=13579,
    )
    results["base_grid"] = compute_base_state_viscosity_grid(
        base_kinds=base_kinds,
        c_values=(0.005, 0.02, 0.05),
        contexts_per_case=2,
        max_mode=30,
        key_seed=97531,
    )

    return results


# -----------------------------------------------------------------------------
# Plotting functions: all figures are made from precomputed results.
# No plot titles are used.
# -----------------------------------------------------------------------------

def plot_training_curves(results):
    tr = results["training"]
    plt.figure(figsize=(6.2, 4.0))
    plt.semilogy(np.maximum(tr["Loss"], 1e-16), label="train MSE")
    plt.semilogy(np.maximum(tr["ValMSE"], 1e-16), label="validation MSE")
    plt.xlabel(f"printed index, every {tr['print_every']} steps")
    plt.ylabel("MSE")
    plt.grid(True, which="both")
    plt.legend(frameon=False)
    plt.tight_layout()
    save_current_figure("training_curves.png")
    plt.close()


def plot_prediction_example_clean(results):
    r = results["prediction_example"]
    plt.figure(figsize=(6.2, 4.0))
    plt.plot(r["x"], r["true"], "k-", label="true")
    plt.plot(r["x"], r["pred"], "r--", label="ICON")
    plt.xlabel("x")
    plt.ylabel("u(T,x)")
    plt.grid(True)
    plt.legend(frameon=False)
    plt.tight_layout()
    save_current_figure("prediction_example.png")
    plt.close()


def plot_query_gain_cos_sin(results):
    r = results["query_gain"]
    k = r["k"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)
    labels = ["cosine perturbation", "sine perturbation"]
    for j, ax in enumerate(axes):
        ax.plot(k, r["true"][:, j], "k-o", label="true tangent")
        ax.plot(k, r["icon"][:, j], "r--s", label="ICON Jacobian")
        ax.axvline(n_modes, linestyle="--", color="0.4", label=r"training bandwidth")
        ax.set_xlabel("Fourier mode k")
        ax.set_ylabel("Jacobian gain")
        ax.grid(True)
        ax.text(0.05, 0.92, labels[j], transform=ax.transAxes, ha="left", va="top")
    axes[0].legend(frameon=False)
    plt.tight_layout()
    save_current_figure("true_vs_icon_query_gain_cos_sin.png")
    plt.close()


def plot_singular_values_clean(results):
    r = results["singular_values"]
    s_true = r["true"] / (r["true"][0] + 1e-12)
    s_icon = r["icon"] / (r["icon"][0] + 1e-12)
    plt.figure(figsize=(6.2, 4.0))
    plt.semilogy(s_true, "k-o", label="true tangent")
    plt.semilogy(s_icon, "r--s", label="ICON Jacobian")
    plt.xlabel("index")
    plt.ylabel("normalized singular value")
    plt.grid(True, which="both")
    plt.legend(frameon=False)
    plt.tight_layout()
    save_current_figure("true_vs_icon_singular_values.png")
    plt.close()


def plot_c_sensitivity_clean(results):
    amps = results["c_sensitivity"]["amps"]
    k = amps[:, 0]
    plt.figure(figsize=(6.2, 4.0))
    plt.semilogy(k, amps[:, 1], "o-", label="cos coefficient")
    plt.semilogy(k, amps[:, 2], "s--", label="sin coefficient")
    plt.xlabel("Fourier mode k")
    plt.ylabel(r"amplitude of $\partial_c G_c^T(u_0)$")
    plt.grid(True, which="both")
    plt.legend(frameon=False)
    plt.tight_layout()
    save_current_figure("true_c_sensitivity_spectrum_cos_sin.png")
    plt.close()


def plot_fourier_blocks_clean(results):
    block_names = [
        ("cc", "cos input", "cos output"),
        ("cs", "sin input", "cos output"),
        ("sc", "cos input", "sin output"),
        ("ss", "sin input", "sin output"),
    ]
    for key in ["true", "icon", "error"]:
        blocks = results["fourier_blocks"][key]
        vmax = max(float(np.max(np.abs(blocks[b]))) for b, _, _ in block_names) + 1e-12
        fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0), constrained_layout=True)
        for ax, (b, xlabel, ylabel) in zip(axes.ravel(), block_names):
            im = ax.imshow(blocks[b], origin="lower", aspect="equal", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xticks(np.arange(blocks[b].shape[1]))
            ax.set_yticks(np.arange(blocks[b].shape[0]))
            ax.set_xticklabels(np.arange(1, blocks[b].shape[1] + 1))
            ax.set_yticklabels(np.arange(1, blocks[b].shape[0] + 1))
            fig.colorbar(im, ax=ax, shrink=0.8)
        save_current_figure(f"split_fourier_blocks_{key}.png")
        plt.close()


def plot_context_spectrum_clean(results):
    r = results["context"]
    k = r["k"]
    c_values = r["c_values"]
    for source, arr, filename_prefix in [
        ("ICON", r["icon_mean"], "context_spectrum_icon"),
        ("true", r["true"], "context_spectrum_true"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)
        for j, ax in enumerate(axes):
            for i, c_value in enumerate(c_values):
                ax.plot(k, arr[i, :, j], "o-", label=f"c={c_value:g}")
            ax.axvline(n_modes, linestyle="--", color="0.3")
            ax.set_xlabel("Fourier mode k")
            ax.set_ylabel(f"{source} Jacobian gain")
            ax.grid(True)
            ax.text(0.05, 0.92, "cos" if j == 0 else "sin", transform=ax.transAxes, ha="left", va="top")
        style_parameter_curves(axes, c_values, "burgers")
        axes[-1].legend(frameon=False, ncol=2)
        plt.tight_layout()
        add_regime_key(fig)
        save_current_figure(comparison_filename(f"{filename_prefix}_cos_sin_ood.png"))
        plt.close()

    # Relative error over c, computed from cos and sin gains together, not pair gain.
    rel_errors = np.linalg.norm(r["icon_mean"] - r["true"], axis=(1, 2)) / (
        np.linalg.norm(r["true"], axis=(1, 2)) + 1e-12
    )
    plt.figure(figsize=(6.2, 4.0))
    plt.semilogy(c_values, rel_errors, "o-")
    plt.axvspan(c_range[0], c_range[1], alpha=0.18, label="training range")
    plt.xlabel("hidden viscosity c")
    plt.ylabel("relative spectral error")
    plt.grid(True, which="both")
    style_parameter_sweep(plt.gca(), "burgers")
    plt.legend(frameon=False)
    add_regime_key(plt.gcf())
    plt.tight_layout()
    save_current_figure(comparison_filename("context_spectrum_error_vs_c_cos_sin.png"))
    plt.close()
    return rel_errors


def plot_base_state_predictions(results):
    r = results["base"]
    base_kinds = r["base_kinds"]
    fig, axes = plt.subplots(len(base_kinds), 2, figsize=(9.2, 3.0 * len(base_kinds)), sharex=True)
    if len(base_kinds) == 1:
        axes = axes[None, :]
    for i, kind in enumerate(base_kinds):
        rr = r["prediction"][kind]
        axes[i, 0].plot(rr["x"], rr["u0"], "k-")
        axes[i, 0].set_ylabel(kind.replace("_", " "))
        axes[i, 0].grid(True)
        axes[i, 1].plot(rr["x"], rr["true"], "k-", label="true")
        axes[i, 1].plot(rr["x"], rr["pred"], "r--", label="ICON")
        axes[i, 1].grid(True)
    axes[-1, 0].set_xlabel("x")
    axes[-1, 1].set_xlabel("x")
    axes[0, 0].text(0.05, 0.90, r"$u_0$", transform=axes[0, 0].transAxes)
    axes[0, 1].text(0.05, 0.90, r"$u(T)$", transform=axes[0, 1].transAxes)
    axes[0, 1].legend(frameon=False)
    plt.tight_layout()
    save_current_figure("burgers_base_state_predictions.png")
    plt.close()


def plot_base_state_spectra_clean(results):
    r = results["base"]
    k = r["k"]
    base_kinds = r["base_kinds"]
    fig, axes = plt.subplots(len(base_kinds), 2, figsize=(10.0, 3.0 * len(base_kinds)), sharex=True, sharey=True)
    if len(base_kinds) == 1:
        axes = axes[None, :]
    for i, kind in enumerate(base_kinds):
        spec = r["spectrum"][kind]
        for j in range(2):
            ax = axes[i, j]
            ax.plot(k, spec["true"][:, j], "k-o", label="true tangent")
            ax.plot(k, spec["icon_mean"][:, j], "r--s", label="ICON Jacobian")
            ax.fill_between(
                k,
                spec["icon_mean"][:, j] - spec["icon_std"][:, j],
                spec["icon_mean"][:, j] + spec["icon_std"][:, j],
                alpha=0.18,
            )
            ax.axvline(n_modes, linestyle="--", color="0.3")
            ax.grid(True)
            ax.set_ylabel(kind.replace("_", " ") if j == 0 else "")
            ax.text(0.05, 0.90, "cos" if j == 0 else "sin", transform=ax.transAxes, ha="left", va="top")
    axes[-1, 0].set_xlabel("Fourier mode k")
    axes[-1, 1].set_xlabel("Fourier mode k")
    axes[0, 0].legend(frameon=False)
    plt.tight_layout()
    save_current_figure("burgers_true_vs_icon_base_state_spectra_cos_sin.png")
    plt.close()


def plot_base_state_viscosity_grid_clean(results):
    r = results["base_grid"]
    k = r["k"]
    base_kinds = r["base_kinds"]
    c_values = r["c_values"]
    fig, axes = plt.subplots(len(base_kinds), 2, figsize=(10.0, 3.0 * len(base_kinds)), sharex=True, sharey=True)
    if len(base_kinds) == 1:
        axes = axes[None, :]
    for i, kind in enumerate(base_kinds):
        for j in range(2):
            ax = axes[i, j]
            for c_value in c_values:
                spec = r["results"][(kind, float(c_value))]
                ax.plot(k, spec["true"][:, j], "-", label=f"true c={c_value:g}")
                ax.plot(k, spec["icon_mean"][:, j], "--", label=f"ICON c={c_value:g}")
            ax.axvline(n_modes, linestyle="--", color="0.3")
            ax.grid(True)
            ax.set_ylabel(kind.replace("_", " ") if j == 0 else "")
            ax.text(0.05, 0.90, "cos" if j == 0 else "sin", transform=ax.transAxes, ha="left", va="top")
    axes[-1, 0].set_xlabel("Fourier mode k")
    axes[-1, 1].set_xlabel("Fourier mode k")
    axes[0, 1].legend(frameon=False, ncol=2, fontsize=8)
    plt.tight_layout()
    save_current_figure("burgers_true_vs_icon_base_state_viscosity_grid_cos_sin.png")
    plt.close()


def plot_all_figures(results):
    plot_training_curves(results)
    plot_prediction_example_clean(results)
    plot_query_gain_cos_sin(results)
    plot_singular_values_clean(results)
    plot_c_sensitivity_clean(results)
    plot_fourier_blocks_clean(results)
    plot_context_spectrum_clean(results)
    plot_base_state_predictions(results)
    plot_base_state_spectra_clean(results)
    plot_base_state_viscosity_grid_clean(results)


if RUN_FINAL_SPECTRUM_DIAGNOSTICS:
    print("\nComputing final diagnostics first...")
    final_results = compute_final_diagnostics()

    print("Plotting all final figures...")
    plot_all_figures(final_results)

    # Save arrays for notebook reuse. Dictionaries with tuple keys are saved as
    # object arrays where needed.
    np.savez(
        os.path.join(fig_dir, "burgers_train_clean_diagnostics_results.npz"),
        training_Loss=final_results["training"]["Loss"],
        training_ValMSE=final_results["training"]["ValMSE"],
        prediction_x=final_results["prediction_example"]["x"],
        prediction_u0=final_results["prediction_example"]["u0"],
        prediction_true=final_results["prediction_example"]["true"],
        prediction_icon=final_results["prediction_example"]["pred"],
        query_k=final_results["query_gain"]["k"],
        query_gain_true=final_results["query_gain"]["true"],
        query_gain_icon=final_results["query_gain"]["icon"],
        singular_true=final_results["singular_values"]["true"],
        singular_icon=final_results["singular_values"]["icon"],
        c_sensitivity_amps=final_results["c_sensitivity"]["amps"],
        context_k=final_results["context"]["k"],
        context_c_values=final_results["context"]["c_values"],
        context_icon_mean=final_results["context"]["icon_mean"],
        context_icon_std=final_results["context"]["icon_std"],
        context_true=final_results["context"]["true"],
        base_k=final_results["base"]["k"],
        base_kinds=np.asarray(final_results["base"]["base_kinds"]),
        base_grid_k=final_results["base_grid"]["k"],
        base_grid_c_values=final_results["base_grid"]["c_values"],
        # store complete nested dictionaries as object arrays for convenience
        base_results=np.asarray(final_results["base"], dtype=object),
        base_grid_results=np.asarray(final_results["base_grid"], dtype=object),
        fourier_blocks=np.asarray(final_results["fourier_blocks"], dtype=object),
    )

    print(f"Saved final figures and arrays to {fig_dir}/")
