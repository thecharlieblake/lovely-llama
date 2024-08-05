# type: ignore
import jax.random as jrandom
import numpy as np
import torch
from equinox import filter_grad
from jax import Array, grad, vmap

# `model` from "llama2_c/"
from model import FeedForward as TorchFFNSwiGLU
from model import RMSNorm as TorchRMSNorm
from model import apply_rotary_emb, precompute_freqs_cis
from torch.nn import Parameter

from lovely_llama import FFNSwiGLU, RMSNorm, rope


def to_tensor(jax_array):
    return torch.from_numpy(np.asarray(jax_array).copy())


def to_param(jax_array):
    return Parameter(to_tensor(jax_array))


def assert_allclose(jax_array, torch_tensor):
    np.testing.assert_allclose(
        jax_array, torch_tensor.detach().numpy(), rtol=1e-5, atol=1e-5
    )


# === VARIABLES === #


KEY, X_KEY = jrandom.split(jrandom.PRNGKey(1472), 2)
BATCH_SIZE = 3
D_MODEL = 5
X = jrandom.normal(X_KEY, (BATCH_SIZE, D_MODEL))
torch_X = to_tensor(X)


# === UNIT TESTS === #


def test_ffn_swiglu():
    ffn_swiglu = FFNSwiGLU(D_MODEL, D_MODEL + 1, KEY)
    torch_ffn_swiglu = TorchFFNSwiGLU(D_MODEL, D_MODEL + 1, D_MODEL + 1, dropout=0.0)
    torch_ffn_swiglu.w3.weight = to_param(ffn_swiglu.w_gate.T)
    torch_ffn_swiglu.w1.weight = to_param(ffn_swiglu.w_in.T)
    torch_ffn_swiglu.w2.weight = to_param(ffn_swiglu.w_out.T)

    assert_allclose(vmap(ffn_swiglu)(X), torch_ffn_swiglu(torch_X))

    loss = lambda model, x: vmap(model)(x).mean()  # noqa: E731
    grads = filter_grad(loss)(ffn_swiglu, X)

    assert isinstance(grads, FFNSwiGLU)
    assert isinstance(grads.w_gate, Array)
    assert isinstance(grads.w_in, Array)
    assert isinstance(grads.w_out, Array)
    assert (grads.w_gate != 0).all()
    assert (grads.w_in != 0).all()
    assert (grads.w_out != 0).all()


def test_rms_norm():
    rms_norm = RMSNorm(D_MODEL)
    torch_rms_norm = TorchRMSNorm(D_MODEL, 1e-5)
    torch_rms_norm.weight = to_param(rms_norm.gain)

    assert_allclose(vmap(rms_norm)(X), torch_rms_norm(torch_X))

    loss = lambda model, x: vmap(model)(x).mean()  # noqa: E731
    grads = filter_grad(loss)(rms_norm, X)

    assert isinstance(grads, RMSNorm)
    assert isinstance(grads.gain, Array)
    assert (grads.gain != 0).all()
    assert (grads.gain != 1).all()


def test_rope():
    batch_size = 2
    seq_len = 3
    num_heads = 4
    d_head = 8
    x = jrandom.normal(X_KEY, (batch_size, seq_len, num_heads, d_head))

    rope_full_shape = vmap(vmap(rope, in_axes=1, out_axes=1), in_axes=0)
    y = rope_full_shape(x)  # Map over batch and head dims

    torch_x = to_tensor(x)
    freqs_cos, freqs_sin = precompute_freqs_cis(d_head, seq_len)
    torch_y, _ = apply_rotary_emb(torch_x, torch_x, freqs_cos, freqs_sin)

    assert_allclose(y, torch_y)

    loss = lambda x: rope_full_shape(x).mean()  # noqa: E731
    g = grad(loss)(x)

    assert isinstance(g, Array)
    assert (g != 0).all()
    assert (g != 1).all()
