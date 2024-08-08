# type: ignore
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import torch
from equinox import filter_grad
from jax import Array, grad, vmap
from model import Attention as TorchSelfAttention
from model import FeedForward as TorchSwiGLUFFN
from model import ModelArgs, apply_rotary_emb, precompute_freqs_cis
from model import RMSNorm as TorchRMSNorm
from model import Transformer as TorchTransformer
from model import TransformerBlock as TorchTransformerBlock
from torch.nn import Parameter

from lovely_llama import (
    GroupedQueryAttention,
    RMSNorm,
    SwiGLUFFN,
    Transformer,
    TransformerBlock,
    rope,
)


def to_tensor(jax_array):
    return torch.from_numpy(np.asarray(jax_array).copy())


def to_param(jax_array):
    return Parameter(to_tensor(jax_array))


def assert_allclose(jax_array, torch_tensor, rtol=1e-5, atol=1e-5):
    np.testing.assert_allclose(
        jax_array, torch_tensor.detach().numpy(), rtol=rtol, atol=atol
    )


# === VARIABLES === #


KEY, X_KEY = jrandom.split(jrandom.PRNGKey(1472), 2)
BATCH_SIZE = 3
D_MODEL = 5
GROUPS = 2
GROUP_SIZE = 7
LAYERS = 4
VOCAB_SIZE = 13
X = jrandom.normal(X_KEY, (BATCH_SIZE, D_MODEL))
torch_X = to_tensor(X)


# === UNIT TESTS === #


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


def test_ffn_swiglu():
    ffn_swiglu = SwiGLUFFN(D_MODEL, D_MODEL + 1, KEY)
    torch_ffn_swiglu = TorchSwiGLUFFN(D_MODEL, D_MODEL + 1, D_MODEL + 1, dropout=0.0)
    torch_ffn_swiglu.w3.weight = to_param(ffn_swiglu.w_gate.T)
    torch_ffn_swiglu.w1.weight = to_param(ffn_swiglu.w_in.T)
    torch_ffn_swiglu.w2.weight = to_param(ffn_swiglu.w_out.T)

    assert_allclose(vmap(ffn_swiglu)(X), torch_ffn_swiglu(torch_X))

    loss = lambda model, x: vmap(model)(x).mean()  # noqa: E731
    grads = filter_grad(loss)(ffn_swiglu, X)

    assert isinstance(grads, SwiGLUFFN)
    assert isinstance(grads.w_gate, Array)
    assert isinstance(grads.w_in, Array)
    assert isinstance(grads.w_out, Array)
    assert (grads.w_gate != 0).all()
    assert (grads.w_in != 0).all()
    assert (grads.w_out != 0).all()


def test_grouped_query_attention():
    dim = D_MODEL * GROUPS * GROUP_SIZE * 2
    self_attention = GroupedQueryAttention(dim, GROUPS, GROUP_SIZE, KEY)
    torch_self_attention = TorchSelfAttention(
        ModelArgs(dim=dim, n_heads=GROUPS * GROUP_SIZE, n_kv_heads=GROUPS)
    )

    torch_self_attention.wq.weight = to_param(
        jnp.reshape(jnp.moveaxis(self_attention.w_query, -2, 0), (dim, -1)).T
    )
    torch_self_attention.wk.weight = to_param(
        jnp.reshape(jnp.moveaxis(self_attention.w_key, -2, 0), (dim, -1)).T
    )
    torch_self_attention.wv.weight = to_param(
        jnp.reshape(jnp.moveaxis(self_attention.w_value, -2, 0), (dim, -1)).T
    )
    torch_self_attention.wo.weight = to_param(
        jnp.reshape(self_attention.w_out, (-1, dim)).T
    )

    seq_len = 11
    x = jrandom.normal(X_KEY, (BATCH_SIZE, seq_len, dim))

    torch_x = to_tensor(x)
    freqs_cos, freqs_sin = precompute_freqs_cis(dim // GROUPS // GROUP_SIZE, seq_len)

    assert_allclose(
        vmap(self_attention)(x),
        torch_self_attention(torch_x, freqs_cos, freqs_sin),
    )

    loss = lambda model, x: vmap(model)(x).mean()  # noqa: E731
    grads = filter_grad(loss)(self_attention, x)

    assert isinstance(grads, GroupedQueryAttention)
    assert isinstance(grads.w_query, Array)
    assert isinstance(grads.w_key, Array)
    assert isinstance(grads.w_value, Array)
    assert isinstance(grads.w_out, Array)
    assert (grads.w_query != 0).all()
    assert (grads.w_key != 0).all()
    assert (grads.w_value != 0).all()
    assert (grads.w_out != 0).all()


def test_transformer_block():
    dim = D_MODEL * GROUPS * GROUP_SIZE * 2
    transformer_block = TransformerBlock(dim, GROUPS, GROUP_SIZE, dim * 3, KEY)
    torch_transformer_block = TorchTransformerBlock(
        0,
        ModelArgs(
            dim=dim, n_heads=GROUPS * GROUP_SIZE, n_kv_heads=GROUPS, hidden_dim=dim * 3
        ),
    )

    torch_transformer_block.attention_norm.weight = to_param(
        transformer_block.attn_norm.gain
    )
    torch_transformer_block.ffn_norm.weight = to_param(transformer_block.ffn_norm.gain)

    torch_transformer_block.attention.wq.weight = to_param(
        jnp.reshape(jnp.moveaxis(transformer_block.attn.w_query, -2, 0), (dim, -1)).T
    )
    torch_transformer_block.attention.wk.weight = to_param(
        jnp.reshape(jnp.moveaxis(transformer_block.attn.w_key, -2, 0), (dim, -1)).T
    )
    torch_transformer_block.attention.wv.weight = to_param(
        jnp.reshape(jnp.moveaxis(transformer_block.attn.w_value, -2, 0), (dim, -1)).T
    )
    torch_transformer_block.attention.wo.weight = to_param(
        jnp.reshape(transformer_block.attn.w_out, (-1, dim)).T
    )
    torch_transformer_block.feed_forward.w3.weight = to_param(
        transformer_block.ffn.w_gate.T
    )
    torch_transformer_block.feed_forward.w1.weight = to_param(
        transformer_block.ffn.w_in.T
    )
    torch_transformer_block.feed_forward.w2.weight = to_param(
        transformer_block.ffn.w_out.T
    )

    seq_len = 11
    x = jrandom.normal(X_KEY, (BATCH_SIZE, seq_len, dim))

    torch_x = to_tensor(x)
    freqs_cos, freqs_sin = precompute_freqs_cis(dim // GROUPS // GROUP_SIZE, seq_len)

    assert_allclose(
        vmap(transformer_block)(x),
        torch_transformer_block(torch_x, freqs_cos, freqs_sin),
    )

    loss = lambda model, x: vmap(model)(x).mean()  # noqa: E731
    grads = filter_grad(loss)(transformer_block, x)

    assert isinstance(grads, TransformerBlock)
    assert isinstance(grads.attn, GroupedQueryAttention)
    assert isinstance(grads.attn.w_query, Array)
    assert isinstance(grads.attn.w_key, Array)
    assert isinstance(grads.attn.w_value, Array)
    assert isinstance(grads.attn.w_out, Array)
    assert (grads.attn.w_query != 0).all()
    assert (grads.attn.w_key != 0).all()
    assert (grads.attn.w_value != 0).all()
    assert (grads.attn.w_out != 0).all()
    assert isinstance(grads.ffn, SwiGLUFFN)
    assert isinstance(grads.ffn.w_gate, Array)
    assert isinstance(grads.ffn.w_in, Array)
    assert isinstance(grads.ffn.w_out, Array)
    assert (grads.ffn.w_gate != 0).all()
    assert (grads.ffn.w_in != 0).all()
    assert (grads.ffn.w_out != 0).all()
    assert isinstance(grads.ffn_norm, RMSNorm)
    assert isinstance(grads.ffn_norm.gain, Array)
    assert (grads.ffn_norm.gain != 0).all()
    assert (grads.ffn_norm.gain != 1).all()
    assert isinstance(grads.attn_norm, RMSNorm)
    assert isinstance(grads.attn_norm.gain, Array)
    assert (grads.attn_norm.gain != 0).all()
    assert (grads.attn_norm.gain != 1).all()


def test_transformer():
    dim = D_MODEL * GROUPS * GROUP_SIZE * 2
    transformer = Transformer(VOCAB_SIZE, dim, GROUPS, GROUP_SIZE, dim * 3, LAYERS, KEY)
    torch_transformer = TorchTransformer(
        ModelArgs(
            dim=dim,
            n_heads=GROUPS * GROUP_SIZE,
            n_kv_heads=GROUPS,
            hidden_dim=dim * 3,
            vocab_size=VOCAB_SIZE,
            n_layers=LAYERS,
        ),
    )

    torch_transformer.tok_embeddings.weight = to_param(transformer.embedding)

    for l in range(LAYERS):  # noqa: E741
        torch_transformer.layers[l].attention_norm.weight = to_param(
            transformer.blocks[l].attn_norm.gain
        )
        torch_transformer.layers[l].ffn_norm.weight = to_param(
            transformer.blocks[l].ffn_norm.gain
        )

        torch_transformer.layers[l].attention.wq.weight = to_param(
            jnp.reshape(
                jnp.moveaxis(transformer.blocks[l].attn.w_query, -2, 0), (dim, -1)
            ).T
        )
        torch_transformer.layers[l].attention.wk.weight = to_param(
            jnp.reshape(
                jnp.moveaxis(transformer.blocks[l].attn.w_key, -2, 0), (dim, -1)
            ).T
        )
        torch_transformer.layers[l].attention.wv.weight = to_param(
            jnp.reshape(
                jnp.moveaxis(transformer.blocks[l].attn.w_value, -2, 0), (dim, -1)
            ).T
        )
        torch_transformer.layers[l].attention.wo.weight = to_param(
            jnp.reshape(transformer.blocks[l].attn.w_out, (-1, dim)).T
        )
        torch_transformer.layers[l].feed_forward.w3.weight = to_param(
            transformer.blocks[l].ffn.w_gate.T
        )
        torch_transformer.layers[l].feed_forward.w1.weight = to_param(
            transformer.blocks[l].ffn.w_in.T
        )
        torch_transformer.layers[l].feed_forward.w2.weight = to_param(
            transformer.blocks[l].ffn.w_out.T
        )

    torch_transformer.norm.weight = to_param(transformer.norm.gain)
    torch_transformer.output.weight = to_param(transformer.unembedding.T)

    seq_len = 11
    idxs = jrandom.randint(X_KEY, (BATCH_SIZE, seq_len), 0, VOCAB_SIZE)

    torch_idxs = to_tensor(idxs)
    dummy_targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_len))

    assert_allclose(
        vmap(transformer)(idxs),
        torch_transformer(torch_idxs, dummy_targets),
        rtol=1e-3,
        atol=1e3,
    )

    loss = lambda model, x: vmap(model)(x).mean()  # noqa: E731
    grads = filter_grad(loss)(transformer, idxs)

    assert isinstance(grads, Transformer)
    assert isinstance(grads.blocks[LAYERS - 2].attn.w_query, Array)
    assert (grads.blocks[LAYERS - 2].attn.w_query != 0).all()
