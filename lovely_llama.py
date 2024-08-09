from equinox import Module
from equinox.nn import Sequential
from jax import Array, vmap
from jax.numpy import arange, array, cos, exp, log, max, ones, prod, sin, sqrt, tril
from jax.random import normal, split
from jaxtyping import Float, Int, PRNGKeyArray

IntScalar = Int[Array, ""]
FloatScalar = Float[Array, ""]
FloatPair = Float[Array, "2"]


class RMSNorm(Module):
    """See https://arxiv.org/abs/1910.07467, eqn (4)."""

    gain: Float[Array, " dim"]
    epsilon: float

    def __init__(self, dim: int, epsilon: float = 1e-5):
        self.gain = ones(dim)
        self.epsilon = epsilon

    def __call__(self, x: Float[Array, " dim"]) -> Float[Array, " dim"]:
        rms = sqrt((x**2).mean() + self.epsilon)
        return (x / rms) * self.gain


def glorot_normal(
    prng_key: PRNGKeyArray, shape: tuple[int, ...]
) -> Float[Array, "..."]:
    """See https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf, eqn (12)."""

    dim_in, dim_out = prod(shape[-1]), shape[-1]
    sigma_init = sqrt(2 / (dim_in + dim_out))
    return normal(prng_key, shape) * sigma_init


def rope(x: Float[Array, "seq dim"]) -> Float[Array, "seq dim"]:
    """See https://arxiv.org/abs/2104.09864, eqns (13-15)."""

    def rotate(
        x_pair: FloatPair, seq_idx: IntScalar, dim_pairs_idx: IntScalar
    ) -> FloatPair:
        angle = seq_idx * 10_000 ** (-2 * dim_pairs_idx / dim)
        rotation_matrix = array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
        return rotation_matrix @ x_pair

    seq, dim = x.shape
    assert dim % 2 == 0, f"{dim=}"
    x_pairs: Float[Array, "seq dim_pairs 2"] = x.reshape(seq, dim // 2, 2)
    seq_idxs: Float[Array, " seq"] = arange(seq)
    dim_pairs_idxs: Float[Array, " dim_pairs"] = arange(dim // 2)
    rotate = vmap(rotate, (0, None, 0))  # Adds 'dim_pairs' axis to rotate args
    rotate = vmap(rotate, (0, 0, None))  # Adds 'seq' axis to rotate args
    x_pairs = rotate(x_pairs, seq_idxs, dim_pairs_idxs)
    return x_pairs.reshape(seq, dim)


def causal_mask(shape: tuple[int, int]) -> Float[Array, "seq seq"]:
    return log(tril(ones(shape)))


def softmax(x: Float[Array, " seq"]) -> Float[Array, " seq"]:
    e = exp(x - max(x))  # -max(x) for numerical stability (doesn't change softmax val)
    return e / e.sum()


def scaled_dot_product_attention(
    q: Float[Array, "seq_q dim_qk"],
    k: Float[Array, "seq_kv dim_qk"],
    v: Float[Array, "seq_kv dim_v"],
) -> Float[Array, "seq_q dim_v"]:
    """See https://arxiv.org/abs/1706.03762, eqn (1)."""

    dim_qk = q.shape[-1]
    qk = q @ k.T / sqrt(dim_qk)
    qk += causal_mask(qk.shape)  # pyright: ignore [reportArgumentType]
    scores = vmap(softmax)(qk)
    return scores @ v


def attention_head(
    x: Float[Array, "seq dim"],
    w_query: Float[Array, "dim dim_qk"],
    w_key: Float[Array, "dim dim_qk"],
    w_value: Float[Array, "dim dim_v"],
    w_out: Float[Array, "dim_v dim"],
) -> Float[Array, "seq dim"]:
    query = x @ w_query
    key = x @ w_key
    value = x @ w_value
    query, key = rope(query), rope(key)
    x = scaled_dot_product_attention(query, key, value)
    return x @ w_out


class GroupedQueryAttention(Module):
    """See https://arxiv.org/abs/2305.13245, section 2.2."""

    w_query: Float[Array, "groups group_size dim dim_qk"]
    w_key: Float[Array, "groups dim dim_qk"]
    w_value: Float[Array, "groups dim dim_v"]
    w_out: Float[Array, "groups group_size dim_v dim"]

    def __init__(self, dim: int, groups: int, group_size: int, prng_key: PRNGKeyArray):
        assert dim % (groups * group_size) == 0, f"{dim=}, {groups=}, {group_size=}"
        dim_qk = dim_v = dim // (groups * group_size)
        k_query, k_key, k_value, k_out = split(prng_key, 4)
        self.w_query = glorot_normal(k_query, (groups, group_size, dim, dim_qk))
        self.w_key = glorot_normal(k_key, (groups, dim, dim_qk))
        self.w_value = glorot_normal(k_value, (groups, dim, dim_v))
        self.w_out = glorot_normal(k_out, (groups, group_size, dim_v, dim))

    def __call__(self, x: Float[Array, "seq dim"]) -> Float[Array, "seq dim"]:
        attn = vmap(attention_head, (None, 0, None, None, 0))  # Adds 'group_size' axis
        attn = vmap(attn, (None, 0, 0, 0, 0))  # Adds 'groups' axis to attn args
        x = attn(x, self.w_query, self.w_key, self.w_value, self.w_out)
        return x.sum((0, 1))  # sum over 'groups' and 'group_size' axes


def swish(x: FloatScalar) -> FloatScalar:
    """See https://arxiv.org/abs/1710.05941v2 (originally coined as SiLU
    https://arxiv.org/abs/1606.08415)."""

    return x / (1 + exp(-x))


class SwiGLUFFN(Module):
    """See https://arxiv.org/abs/2002.05202, eqn (6)."""

    w_gate: Float[Array, "dim dim_ffn"]
    w_in: Float[Array, "dim dim_ffn"]
    w_out: Float[Array, "dim_ffn dim"]

    def __init__(self, dim: int, dim_ffn: int, prng_key: PRNGKeyArray):
        k_in, k_gate, k_out = split(prng_key, 3)
        self.w_gate = glorot_normal(k_gate, (dim, dim_ffn))
        self.w_in = glorot_normal(k_in, (dim, dim_ffn))
        self.w_out = glorot_normal(k_out, (dim_ffn, dim))

    def __call__(self, x: Float[Array, " dim"]) -> Float[Array, " dim"]:
        gate = x @ self.w_gate
        x = vmap(swish)(x @ self.w_in)
        x *= gate
        return x @ self.w_out


class TransformerBlock(Module):
    attn_norm: RMSNorm
    attn: GroupedQueryAttention
    ffn_norm: RMSNorm
    ffn: SwiGLUFFN

    def __init__(
        self,
        dim: int,
        groups: int,
        group_size: int,
        dim_ffn: int,
        prng_key: PRNGKeyArray,
    ):
        k_attn, k_ffn = split(prng_key, 2)
        self.attn_norm = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, groups, group_size, k_attn)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, dim_ffn, k_ffn)

    def __call__(self, x: Float[Array, "seq dim"], **_) -> Float[Array, "seq dim"]:
        # vmap adds "seq" axis to all non-attn ops, as they each operate per-token
        x += self.attn(vmap(self.attn_norm)(x))
        return x + vmap(self.ffn)(vmap(self.ffn_norm)(x))


class Transformer(Module):
    embedding: Float[Array, "vocab_size dim"]
    blocks: Sequential
    norm: RMSNorm
    unembedding: Float[Array, "dim vocab_size"]

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        groups: int,
        group_size: int,
        dim_ffn: int,
        layers: int,
        prng_key: PRNGKeyArray,
    ):
        k_embed, *k_layers, k_unembed = split(prng_key, layers + 2)
        self.embedding = normal(k_embed, (vocab_size, dim))
        self.blocks = Sequential(
            [
                TransformerBlock(dim, groups, group_size, dim_ffn, k_layer)
                for k_layer in k_layers
            ]
        )
        self.norm = RMSNorm(dim)
        self.unembedding = glorot_normal(k_unembed, (dim, vocab_size))

    def __call__(self, idxs: Int[Array, " seq"]) -> Float[Array, "seq vocab"]:
        x = self.embedding[idxs]
        x = self.blocks(x)
        x = vmap(self.norm)(x)
        return x @ self.unembedding
