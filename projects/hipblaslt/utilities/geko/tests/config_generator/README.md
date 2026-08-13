# Config generator tests

Integration, artifact checks, and MI / optimization / fork-param coverage for `geko.config_generator`.

## Contents

| Module | Role |
|--------|------|
| `artifact_assertions.py` | Output layout, YAML section keywords, shell script checks |
| `test_workflow.py` | CLI subprocess, `run`, artifact validation, `main()` guardrails |
| `test_load_input_workload.py` | `SIZE_OPTION` 2 / `GEMM_LOG_PATH` validation and `parse`-only GEMM list |
| `test_components.py` | MIDesign / optimization params / `generate_fork_params` matrix |
| `fixtures/minimal_config.yaml` | Small YAML for `--config` |

## Running

From the `geko` utilities directory (`projects/hipblaslt/utilities/geko` under your ROCm / hipBLASLt checkout):

```bash
python3 -m pytest tests/config_generator/ -v -rs \
  --config tests/config_generator/fixtures/minimal_config.yaml \
  --hipblaslt-path /path/to/hipblaslt
```

Hermetic-only (no hipBLASLt):

```bash
python3 -m pytest tests/config_generator/test_workflow.py::TestConfigGeneratorMainGuardrails -v
```

Shared CLI options and fixtures: [tests/conftest.py](../conftest.py).

### Pytest markers

| Marker | Meaning |
|--------|---------|
| `cg_integration` | CLI / `run` + artifact checks (needs `--config` + `--hipblaslt-path`) |
| `cg_cli_guard` | `main()` path errors only (hermetic) |
| `cg_components` | MI / opt / fork pipeline (needs `--hipblaslt-path` for Tensile) |

Examples:

```bash
pytest tests/config_generator/ -m cg_cli_guard -v
pytest tests/config_generator/ -m 'cg_integration or cg_components' -v -rs --config ... --hipblaslt-path ...
```
