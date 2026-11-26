# Vendored AOTriton flash tests

AOTriton's SDPA pytest suite, copied byte-identical, with `aotriton_flash.py` replaced by a
hipDNN-backed shim (`shim/aotriton_flash.py`). `vendor.lock.json` records the sha256 of every
file; `sync.py --check` verifies it. `--check --no-collect` checks hashes only; the
collection counts are a manual `sync.py --check` (about 5 minutes), and runs compare counts via `collected.txt`.

| | |
|---|---|
| Source | [ROCm/aotriton](https://github.com/ROCm/aotriton) `main`, `0.15b-8` |
| Pin | `a09a19361408e25ac515d4fd59970cd249a1b4a6` |
| Licence | MIT, Copyright (C) 2024-2025 Advanced Micro Devices, Inc. (`vendor/LICENSE`) |

## Contents

| Vendored path | From AOTriton |
|---|---|
| `vendor/modules/flash/tests/**` (28 files, including `adiffs/`) | `modules/flash/tests/**`, except `aotriton_flash.py` |
| `vendor/python/pytest-gpu-lease/pytest_gpu_lease/` (3 files) | same path; defines the `gpu_id` fixture |
| `vendor/LICENSE` | `LICENSE` |

`aotriton_flash.py` is **replaced**: `vendor/` must not contain it, so the shim on `PYTHONPATH`
cannot be shadowed. `shim/pyaotriton/` stubs the remaining `pyaotriton` imports. Vendored files are
never edited; `adiffs/`, `performance_*`, `triton_*`, `pytest2entry.py` and `test_varlen_translation.cc` (C++,
not a pytest module) are present but unused. The suite's own knobs that change what it measures
(calibration, backend pins, `AOTRITON_UT_ARCH`, which narrows head dimensions and sequence lengths
for some architectures) are listed in `aotriton_layout.REFUSED_KNOBS`; the plugin refuses them.

`aotriton/.gitattributes` marks `vendor/**` as `-text`, so a checkout keeps the committed LF bytes
and `sync.py --check` hashes the same bytes on every platform (it hashes working-tree bytes; a
CRLF conversion makes it report `sha256 differs`). The rule lives outside `vendor/` because
`sync.py --source` replaces that directory.

## Collection counts

`pytest --collect-only` on CPU, no errors, at each `FOR_RELEASE` level
(`test_forward.py` is excluded from directory collection and named explicitly):

| `FOR_RELEASE` | default collection | `test_forward.py` |
|---|---|---|
| 0 | 4,658 | 497,264 |
| 1 | 185,252 | 177,584 |
| 2 | 415,652 | 177,584 |
| 3 | 501,500 | 177,584 |

## Refresh

1. Check out AOTriton at the new commit (sparse: `modules/flash/tests`, `python/pytest-gpu-lease`, `LICENSE`).
2. `python aotriton/sync.py --source <checkout>` refuses a checkout with uncommitted changes under the
   vendored paths, copies the files into a temporary directory, swaps it in as `vendor/` and rewrites
   `vendor.lock.json` (pin, hashes, counts; about 5 minutes). If the counting fails, the previous
   `vendor/` and lock stay.
3. Update the pin and counts above, then diff `shim/aotriton_flash.py` against the new upstream
   `aotriton_flash.py` for signature changes and rerun `aotriton/tests`.
