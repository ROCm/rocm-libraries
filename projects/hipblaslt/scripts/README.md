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

### Exit code

- **0** – All solutions passed (Reject = 0).
- **1** – One or more failed; errors are printed.

The full tree (~2246 files) can take several minutes, so passing a subdirectory is useful for quick checks. To tune parallelism: `python scripts/run_tensile_logic_check.py -j 16` (default is 48 workers, capped by CPU count).

### run_tensile_logic_check.sh (Unix only)

Thin wrapper that runs the script with `.venv/bin/python`, if it's present, and with `python3` otherwise. Use the `.py` script directly on Windows.
