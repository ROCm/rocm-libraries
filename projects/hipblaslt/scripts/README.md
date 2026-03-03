# hipBLASLt scripts

For full build prerequisites and installation, see the [project README](../README.md) and [Building and installing hipBLASLt](../docs/install/building-installing-hipblaslt.rst).

## run_tensile_logic_check.py

Runs **TensileLogic --check-all** on the library logic YAMLs (same check as the pre-build gate). Cross-platform (Windows and Unix).

### How to run

From the **hipblaslt project root** (where `library/` and `tensilelite/` live):

```bash
python scripts/run_tensile_logic_check.py
```

**Windows:** `python scripts\run_tensile_logic_check.py` (or use `py` if you have the launcher.)

That’s it. Use whatever Python you have (system, store, or a venv). If that Python is missing dependencies (e.g. joblib) and the project has a **`.venv`** in the repo root, the script will re-run itself with `.venv\Scripts\python.exe` (Windows) or `.venv/bin/python` (Unix), so you don’t have to remember to activate.

### One-time setup

1. **Build once** so `build/tensilelite/rocisa/lib` exists (rocisa is built by the normal hipblaslt build).
2. **Python deps** – either:
   - Use the project **.venv**: `python -m venv .venv` then `.venv\Scripts\pip install -r tensilelite/requirements.txt` (Windows) or `.venv/bin/pip install -r tensilelite/requirements.txt` (Unix), or  
   - Install into your current Python: `pip install -r tensilelite/requirements.txt`.

### Optional: check a single directory

```bash
python scripts/run_tensile_logic_check.py library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/navi33/GridBased
```

### Exit code

- **0** – All solutions passed (Reject = 0).
- **1** – One or more failed; errors are printed.

The full tree (~2246 files) can take several minutes; passing a subdirectory is useful for quick checks. To tune parallelism: `python scripts/run_tensile_logic_check.py -j 16` (default is 48 workers, capped by CPU count).

### run_tensile_logic_check.sh (Unix only)

Thin wrapper that runs the script with `.venv/bin/python` if present, else `python3`. Use the `.py` script directly on Windows.
