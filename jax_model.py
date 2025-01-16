from collections.abc import Callable

import equinox as eqx
from jax import Array, lax, vmap
from jax.numpy import arange, array, cos, ones, sin, sqrt
from jax.random import bernoulli, normal, split
from jaxtyping import ArrayLike, Float, Int

IntScalar = Int[Array, ""]
FloatScalar = Float[Array, ""]
FloatPair = Float[Array, "2"]
Layer = Callable[[Float[Array, "seq dim"]], Float[Array, "seq dim"]]


class RMSNorm(eqx.Module):
    """See https://arxiv.org/abs/1910.07467, eqn (4)."""

    g: Float[Array, "dim"]
    eps: float

    def __init__(self, dim: int, eps: float = 1e-5):
        self.g = ones(dim)
        self.eps = eps

    def __call__(self, x: Float[Array, "dim"]) -> Float[Array, "dim"]:
        rms = sqrt((x**2).mean() + self.eps)
        return x / rms * self.g


def rope(x: Float[Array, "seq dim"]) -> Float[Array, "seq dim"]:
    """See https://arxiv.org/abs/2104.09864, eqns (13-15)."""

    def rotate(
        x_pair: FloatPair, seq_idx: IntScalar, dim_2_idx: IntScalar
    ) -> FloatPair:
        angle = seq_idx * 10_000 ** (-2 * dim_2_idx / dim)
        rotation_matrix = array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
        return rotation_matrix @ x_pair

    seq, dim = x.shape
    x_pairs: Float[Array, "seq dim_2 2"] = x.reshape(seq, dim // 2, 2)
    seq_idxs, dim_2_idxs = arange(seq), arange(dim // 2)

    # inner vmap adapts rotate() to batch over 'dim_2' axis, outer vmap over 'seq' axis
    rotate = vmap(vmap(rotate, in_axes=(0, None, 0)), in_axes=(0, 0, None))
    x_pairs = rotate(x_pairs, seq_idxs, dim_2_idxs)

    return x_pairs.reshape(seq, dim)


def swish(x: FloatScalar) -> FloatScalar:
    """See https://arxiv.org/abs/1710.05941v2 (originally coined as SiLU
    https://arxiv.org/abs/1606.08415)."""

    return x * lax.logistic(x)  # def logistic(x): 1 / (1 + exp(-x))


class FFNSwiGLU(eqx.Module):
    """See https://arxiv.org/abs/2002.05202, eqn (6)."""

    w: Float[Array, "dim d_ffn"]
    v: Float[Array, "dim d_ffn"]
    w2: Float[Array, "d_ffn dim"]

    def __init__(self, dim: int, d_ffn: int, key: ArrayLike):
        k_w, k_v, k_w2 = split(key, 3)
        self.w = normal(k_w, (dim, d_ffn))
        self.v = normal(k_v, (dim, d_ffn))
        self.w2 = normal(k_w2, (d_ffn, dim))

    def __call__(self, x: Float[Array, "dim"]) -> Float[Array, "dim"]:
        return (vmap(swish)(x @ self.w) * (x @ self.v)) @ self.w2


class Dropout(eqx.Module):
    p: float

    def __call__(
        self, x: FloatScalar, key: ArrayLike, inference: bool = False
    ) -> FloatScalar:
        if inference:
            return x
        q = 1 - self.p
        mask = bernoulli(key, q, x.shape)
        return mask * x / q


class PreNormResidualBlock(eqx.Module):
    rms_norm: RMSNorm
    residual: Layer
    dropout: Dropout

    def __init__(self, residual: Layer, dim: int, dropout_p: float):
        self.rms_norm = RMSNorm(dim)
        self.residual = residual
        self.dropout = Dropout(dropout_p)  # type: ignore[call-arg]

    def __call__(
        self, x: Float[Array, "seq dim"], key: ArrayLike
    ) -> Float[Array, "seq dim"]:
        skip = x
        x = self.rms_norm(x)
        x = self.residual(x)
        x = vmap(self.dropout)(x, key)
        return x + skip


class TransformerBlock(eqx.Module):

    def __init__(self, dim: int, d_ffn: int, dropout_p: float, key: ArrayLike):
        k_a, k_f = split(key, 2)
        self.attn_block = PreNormResidualBlock(SelfAttention(), dim, dropout_p)
        self.ffn_block = PreNormResidualBlock(
            FFNSwiGLU(dim, d_ffn, k_f), dim, dropout_p
        )

    def __call__(
        self, x: Float[Array, "*batch dim"], key: ArrayLike
    ) -> Float[Array, "*batch dim"]:
        k_a, k_f = split(key, 2)
        x = self.attn_block(x, k_a)
        return self.ffn_block(x, k_f)
