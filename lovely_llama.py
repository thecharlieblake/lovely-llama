from collections.abc import Callable

import equinox as eqx
from jax import Array, vmap
from jax.numpy import arange, array, cos, exp, ones, sin, sqrt
from jax.random import normal, split
from jaxtyping import Float, Int

IntScalar = Int[Array, ""]
FloatScalar = Float[Array, ""]
FloatPair = Float[Array, "2"]


class RMSNorm(eqx.Module):
    """See https://arxiv.org/abs/1910.07467, eqn (4)."""

    gain: Float[Array, " dim"]
    epsilon: float

    def __init__(self, dim: int, epsilon: float = 1e-5):
        self.gain = ones(dim)
        self.epsilon = epsilon

    def __call__(self, x: Float[Array, " dim"]) -> Float[Array, " dim"]:
        rms = sqrt((x**2).mean() + self.epsilon)
        return (x / rms) * self.gain


def rope(x: Float[Array, "seq dim"]) -> Float[Array, "seq dim"]:
    """See https://arxiv.org/abs/2104.09864, eqns (13-15)."""

    def rotate(
        x_pair: FloatPair,
        seq_idx: IntScalar,
        dim_pairs_idx: IntScalar,
    ) -> FloatPair:
        angle = seq_idx * 10_000 ** (-2 * dim_pairs_idx / dim)
        rotation_matrix = array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
        return rotation_matrix @ x_pair

    seq, dim = x.shape
    x_pairs: Float[Array, "seq dim_pairs 2"] = x.reshape(seq, dim // 2, 2)
    seq_idxs, dim_pairs_idx = arange(seq), arange(dim // 2)

    # inner vmap adapts rotate() to batch over 'dim_pairs' axis, outer vmap over 'seq'
    rotate = vmap(vmap(rotate, in_axes=(0, None, 0)), in_axes=(0, 0, None))
    x_pairs = rotate(x_pairs, seq_idxs, dim_pairs_idx)

    return x_pairs.reshape(seq, dim)


def swish(x: FloatScalar) -> FloatScalar:
    """See https://arxiv.org/abs/1710.05941v2 (originally coined as SiLU
    https://arxiv.org/abs/1606.08415)."""

    return x / (1 + exp(-x))


class FFNSwiGLU(eqx.Module):
    """See https://arxiv.org/abs/2002.05202, eqn (6)."""

    w_gate: Float[Array, "dim dim_ffn"]
    w_in: Float[Array, "dim dim_ffn"]
    w_out: Float[Array, "dim_ffn dim"]

    def __init__(self, dim: int, dim_ffn: int, key: Array):
        k_in, k_gate, k_out = split(key, 3)
        self.w_gate = normal(k_gate, (dim, dim_ffn))
        self.w_in = normal(k_in, (dim, dim_ffn))
        self.w_out = normal(k_out, (dim_ffn, dim))

    def __call__(self, x: Float[Array, " dim"]) -> Float[Array, " dim"]:
        gate = x @ self.w_gate
        x = vmap(swish)(x @ self.w_in)
        x *= gate
        return x @ self.w_out


class PreNormResidualBlock(eqx.Module):
    rms_norm: RMSNorm
    residual: Callable[[Float[Array, "seq dim"]], Float[Array, "seq dim"]]

    def __init__(
        self,
        residual: Callable[[Float[Array, "seq dim"]], Float[Array, "seq dim"]],
        dim: int,
    ):
        self.rms_norm = RMSNorm(dim)
        self.residual = residual

    def __call__(self, x: Float[Array, "seq dim"]) -> Float[Array, "seq dim"]:
        skip = x
        x = self.rms_norm(x)
        x = self.residual(x)
        return x + skip


# class TransformerBlock(eqx.Module):
#     def __init__(self, dim: int, d_ffn: int, key: Array):
#         k_a, k_f = split(key, 2)
#         self.attn_block = PreNormResidualBlock(SelfAttention(dim, k_a), dim)
#         self.ffn_block = PreNormResidualBlock(FFNSwiGLU(dim, d_ffn, k_f), dim)

#     def __call__(self, x: Float[Array, "seq dim"]) -> Float[Array, "seq dim"]:
#         x = self.attn_block(x)
#         return self.ffn_block(x)
