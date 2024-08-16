# lovely-llama
An implementation of the Llama architecture, to instruct and delight.

## Setup

### Users

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Developers

```
git submodule update --init
python -m venv .venv
echo 'PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}llama2_c"' >> .venv/bin/activate
source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install --hook-type pre-push
chmod 755 dev
```
and run `./dev` for test, type-checking and formatting (see `./dev --help`).

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

1. compare model training loss to baseline and fix any issues
2. implement training and optim (while keeping training parity with baseline)

## License

This project is licensed under the MIT License (see `LICENSE`). It includes components that are derived from work licensed under the Apache License, Version 2.0 (`dev` script which is derived from https://github.com/graphcore-research/unit-scaling/blob/main/dev, and `typings/jax/` which is derived from https://github.com/google/jax/tree/main/jax/).
