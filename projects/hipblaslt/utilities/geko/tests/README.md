# Geko tests

Run from the `geko` utilities directory (`projects/hipblaslt/utilities/geko` under your ROCm / hipBLASLt checkout):

```bash
cd ~/rocm-libraries/projects/hipblaslt/utilities/geko

python3 -m pytest tests/ -v -rs \
    --hipblaslt-path ~/rocm-libraries/projects/hipblaslt \
    --config tests/config_generator/fixtures/minimal_config.yaml \
    --workload tests/test_data/workload.yaml
```

Optional ``--hw gfx942`` (etc.) sets ``scripts/configure.py --architecture`` for configure / optimize integration tests; default is ``gfx950``.

Long subprocess / GPU integration tests are marked with `@pytest.mark.slow`. To run the rest of the suite without them:

```bash
python3 -m pytest tests/ -v -rs --skip-slow
```

Equivalent built-in form:

```bash
python3 -m pytest tests/ -v -rs -m "not slow"
```

## Config generator

[tests/config_generator/](config_generator/README.md):

```bash
python3 -m pytest tests/config_generator/ -v -rs \
    --config tests/config_generator/fixtures/minimal_config.yaml \
    --hipblaslt-path ~/rocm-libraries/projects/hipblaslt
```
