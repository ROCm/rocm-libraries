# Vendored AOTriton flash tests

AOTriton's SDPA pytest suite, copied byte-identical, with `aotriton_flash.py` replaced by a
hipDNN-backed shim (`shim/aotriton_flash.py`). `vendor.lock.json` records the sha256 of every
file; `sync.py --check` verifies it. CTest runs only the hash check (`--check --no-collect`); the
collection counts are a manual `sync.py --check` (about 5 minutes), and runs compare counts via `collected.txt`.

| | |
|---|---|
| Source | [ROCm/aotriton](https://github.com/ROCm/aotriton) `main`, `0.14b-10` |
| Pin | `030b7b8db8f24aaa13dbf5501f7da33380c78007` |
| Licence | MIT, Copyright (C) 2024-2025 Advanced Micro Devices, Inc. (`vendor/LICENSE`) |

## Contents

| Vendored path | From AOTriton |
|---|---|
| `vendor/modules/flash/tests/**` (25 files, including `adiffs/`) | `modules/flash/tests/**`, except `aotriton_flash.py` |
| `vendor/python/pytest-gpu-lease/pytest_gpu_lease/` (3 files) | same path; defines the `gpu_id` fixture |
| `vendor/LICENSE` | `LICENSE` |

`aotriton_flash.py` is **replaced**: `vendor/` must not contain it, so the shim on `PYTHONPATH`
cannot be shadowed. `shim/pyaotriton/` stubs the remaining `pyaotriton` imports. Vendored files are
never edited; `adiffs/`, `performance_*`, `triton_*` and `pytest2entry.py` are present but unused.

## Collection counts

`pytest --collect-only` on CPU, no errors, at each `FOR_RELEASE` level
(`test_forward.py` is excluded from directory collection and named explicitly):

| `FOR_RELEASE` | default collection | `test_forward.py` |
|---|---|---|
| 0 | 3,915 | 497,088 |
| 1 | 183,159 | 177,408 |
| 2 | 413,559 | 177,408 |
| 3 | 499,407 | 177,408 |

## Refresh

1. Check out AOTriton at the new commit (sparse: `modules/flash/tests`, `python/pytest-gpu-lease`, `LICENSE`).
2. `python aotriton/sync.py --source <checkout>` copies the files, rewrites `vendor.lock.json`
   (pin, hashes, counts; about 5 minutes).
3. Update the pin and counts above, then diff `shim/aotriton_flash.py` against the new upstream
   `aotriton_flash.py` for signature changes and rerun `aotriton/tests`.
