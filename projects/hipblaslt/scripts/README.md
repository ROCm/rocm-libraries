# hipBLASLt scripts

For full build prerequisites and installation, see the [project README](../README.md) and [Building and installing hipBLASLt](../docs/install/building-installing-hipblaslt.rst).

## run_tensile_logic_check.py

Runs **TensileLogic --check-all** on the library logic YAMLs (same check as the pre-build gate). This file is cross-platform (Windows and Unix).

### How to run

From the **hipblaslt project root** (where `library/` and `tensilelite/` live):

```bash
python scripts/run_tensile_logic_check.py
```

**Windows**: `python scripts\run_tensile_logic_check.py` (or use `py` if you have the launcher).

That’s it. Use whatever type of Python you have (system, store, or a venv). If that Python is missing dependencies (e.g. joblib) and the project has a **`.venv`** in the repo root, the script will re-run itself with `.venv\Scripts\python.exe` (Windows) or `.venv/bin/python` (Unix), so you don’t have to remember to activate.

### One-time setup

1. **Build once** so rocisa is present under `build/tensilelite/rocisa/` (nanobind `_rocisa` next to the `rocisa` package) or the legacy `build/tensilelite/rocisa/lib` layout.
2. **Python deps** – either:
   - Use the project **.venv**: `python -m venv .venv` then `.venv\Scripts\pip install -r tensilelite/requirements.txt` (Windows) or `.venv/bin/pip install -r tensilelite/requirements.txt` (Unix), or  
   - Install into your current Python: `pip install -r tensilelite/requirements.txt`.

### Optional: check a single directory

```bash
python scripts/run_tensile_logic_check.py library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/navi33/GridBased
```

### Known-bugs list (ROCM-7144 / validation exceptions)

This script and the CMake pre-build gate explicitly enable TensileLogic's bundled known-bugs list, loaded through package resources, so specific `(logic file path, solution_name)` pairs are skipped. `solution_name` is a solution's `SolutionNameMin`, a content-derived name that stays stable when the library is re-tuned (the positional `SolutionIndex` is not stable, so it is no longer used as the key). Paths in the `known_bugs.yaml` are relative to the library logic root (`library/`), with optional `#` comments and an optional `ticket:` field for Jira keys. Override the list by passing your own `--known-bugs` file; pass an empty YAML file to disable all bundled entries. A direct `TensileLogic --check-all` invocation applies no known-bug skips unless it is given `--known-bugs FILE` or `--use-bundled-known-bugs`.

Documented known bugs are still re-validated on every run instead of being blindly skipped. If a listed solution **now passes** validation (the underlying bug was fixed), the run prints a `Stale known-bugs` warning naming the entry to remove. Pass **`--strict-known-bugs`** to make the run exit non-zero on any stale entry; use that in CI or in the PR that lands the fix, so the fixing PR also removes the listing.

### In CI

`Component CI: hipBLASLt Logic Corpus` ([`component-ci-logic-corpus.yml`](../../../.github/workflows/component-ci-logic-corpus.yml)) runs this script with `--strict-known-bugs` on every pull request that touches the library logic, `known_bugs.yaml` or the other `TensileLogic` sources, or this script. A failure blocks the merge through the required `Component CI Summary`. It runs through `tox -e logic-corpus` in `tensilelite/`, which pip-installs rocisa instead of building hipBLASLt; the script accepts an installed rocisa when no build tree is present.

On a pull request the lane passes `--changed-files=FILE`, the paths the pull request changed, one per line. The script then checks only the architectures that the changed logic YAMLs declare, which keeps the lane to minutes; the whole tree takes about half an hour on four workers. A change to the `TensileLogic` validators, `known_bugs.yaml`, the YAML loader or this script can affect every architecture, so it checks the whole tree, as does an empty value (`--changed-files=`), which is what a manual run of the workflow uses. A change list with none of those in it checks nothing.

### Exit code

- **0** – All solutions passed (Reject = 0).
- **1** – One or more failed; errors are printed.

The full tree (~2246 files) can take several minutes, so passing a subdirectory is useful for quick checks. To tune parallelism: `python scripts/run_tensile_logic_check.py -j 16` (default is 48 workers, capped by CPU count).

### run_tensile_logic_check.sh (Unix only)

Thin wrapper that runs the script with `.venv/bin/python`, if it's present, and with `python3` otherwise. Use the `.py` script directly on Windows.

## rocroller_unbuildable_configs.py

Writes `tensilelite/tensilelite/CustomKernels/rocroller/unbuildable_configs.def`, the rocRoller GEMM configurations that rocRoller fails to build and therefore never runs, from build records. The checked-in records are `scripts/rocroller_build_records.csv`: one line per configuration rocRoller tried to build (layout, element types, tile, workgroup mapping, Stream-K), with `built` or the rocRoller pass that failed. Its `# Source:` and `# rocRoller tree:` lines name the develop commit and the `git rev-parse HEAD:shared/rocroller` tree the records were built with; `write` copies both into the table.

The checked-in records cover the configurations rocRoller tried while choosing kernels for a 72,300-problem gfx950 MX probe (TN, NN, NT and TT, every MX type pair, f16, bf16 and f32 outputs, packed 32-element E8M0 scales): 13,244 configurations, 2,715 of which fail to build. A configuration outside that set is not recorded, even if rocRoller fails to build it.

To regenerate the table from the checked-in records (no GPU needed), and to check that the checked-in files are what the records render:

```bash
python scripts/rocroller_unbuildable_configs.py write scripts/rocroller_build_records.csv \
    --out tensilelite/tensilelite/CustomKernels/rocroller/unbuildable_configs.def \
    --records-out scripts/rocroller_build_records.csv
python scripts/rocroller_unbuildable_configs.py write scripts/rocroller_build_records.csv \
    --out tensilelite/tensilelite/CustomKernels/rocroller/unbuildable_configs.def \
    --records-out scripts/rocroller_build_records.csv --check
```

Rebuilding the records needs a GPU and a `HIPBLASLT_ENABLE_ROCROLLER=ON` build, so CI does not run it. Do it whenever `shared/rocroller` changes, then write the table and the records from the new ones:

```bash
python scripts/rocroller_unbuildable_configs.py build --build <ON build dir> \
    --configs scripts/rocroller_build_records.csv --gpus 0,1 \
    --source "develop <sha>, ROCm <version>: <how the configurations were chosen>" --out records.csv
python scripts/rocroller_unbuildable_configs.py write records.csv \
    --out tensilelite/tensilelite/CustomKernels/rocroller/unbuildable_configs.def \
    --records-out scripts/rocroller_build_records.csv
```

`tensilelite/tensilelite/Tests/unit/test_rocroller_unbuildable_record.py` compares the table's `rocRoller tree` line with `git rev-parse HEAD:shared/rocroller`. When they differ, the test skips and its message gives the commands above, so a `shared/rocroller` change does not fail unrelated pull requests; the table stays in use until it is rebuilt. Set `HIPBLASLT_REQUIRE_ROCROLLER_RECORD_TREE=1` to make the mismatch a failure instead, as a maintainer check before a release or after a rocRoller update. From `tensilelite/`, in the environment the unit tests run in (rocisa importable):

```bash
HIPBLASLT_REQUIRE_ROCROLLER_RECORD_TREE=1 python -m pytest tensilelite/Tests/unit/test_rocroller_unbuildable_record.py
```

`build` builds each configuration with the ON build's `hipblaslt-bench` (one rocRoller solution index each, with temporal loads, on a problem of twice the tile in M and N and four times in K) and appends one record per configuration. It takes the rocRoller tree from the checkout the ON build was configured from (`--rocroller-tree` overrides it). `write` takes records from `build`, as CSV or JSONL (`--source` and `--rocroller-tree` are then required), or from any other log of rocRoller's build attempts in the same format, and records every configuration that failed to build. To add configurations, append them to the `--configs` list.
