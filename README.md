# lovely-llama
An implementation of the Llama architecture, to instruct and delight.

## Setup

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) and then run the following, depending on your use-case:

### Users

```
uv sync --no-dev
```

### Developers

Setup:

```
uv sync
git submodule update --init
pre-commit install --hook-type pre-push
```

Formatting / typechecking / testing:

```
uvx ruff format
uvx ty check
uv run pytest --cov
```

Developers may also wish to use the `ruff` and `ty` vscode extensions. Pre-configured setup can be applied via:

```
cp -r .vscode-default .vscode
```

and installing the extensions recommended in the popup.

### Maintainers

```
uv lock --update
```

## Principles for a lovely implementation

The principles I've adopted for a "lovely" implementation:

1. Everything is implemented in one file, from basic `jax.numpy` building blocks
2. The shapes of tensors in a function's parameters are a) explicit and b) minimal
3. The code looks like the corresponding maths (with references from the literature!)
4. No optimizations

These are fulfilled practically via (points corresponding 1-to-1 with the ones above):

1. Everything is tested for correctness against the python implementation in karpathy's [llama2.c repo](https://github.com/karpathy/llama2.c), and made tidy via [ruff](https://docs.astral.sh/ruff/) and [pyright](https://microsoft.github.io/pyright/#/)
2. a) The use of [jaxtyping](https://docs.kidger.site/jaxtyping/) for shape-aware runtime type-checking, b) aggressively `vmap`ping to remove any "batching" dimensions from function parameter-shapes
3. This is made possible because of the vmapping convention (no einsums required!). Some variable names are made more explicit where the maths-naming would be unclear
4. Just don't do it

## Todo

- write out block vmapping in full
- better vmap

- implement training and optim (while keeping training parity with baseline)
- compare model training loss to baseline and fix any issues

## License

This project is licensed under the MIT License (see `LICENSE`).
