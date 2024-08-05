# lovely-llama
A most pleasing implementation of the Llama architecture, to instruct and delight.

## Setup

### Users
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Developers
```
python -m venv .venv
echo 'PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}llama2_c"' >> .venv/bin/activate
source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
git clone --recursive  # to run tests (against llama.c)
chmod 755 dev
```
and run `./dev` for test, type-checking and formatting (see `./dev --help`).

## Todo

1. ci
2. get implementing!

## License

This project is licensed under the MIT License (see `LICENSE`). However, it includes components that are derived from work licensed under the Apache License, Version 2.0 (`dev` script which is derived from https://github.com/graphcore-research/unit-scaling/blob/main/dev, and `typings/jax/` which is derived from https://github.com/google/jax/tree/main/jax/).
