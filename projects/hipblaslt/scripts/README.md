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

## run_cmake_configure_matrix.sh

Runs the supported local CMake topology matrix for hipBLASLt, rocISA, and hipSPARSELt. It is a
configure-first integration runner: each cell uses an isolated build directory and checks the
generated cache, config/export files, Ninja command, or CPack metadata. It does not replace GPU
execution or Windows package/import testing.

Run it **inside** a prepared ROCm environment. The source-provider cells need an offline nanobind
checkout because rocISA uses FetchContent. The staged cell either accepts an existing provider
prefix or builds a minimal one with `--prepare-stage`.

```bash
projects/hipblaslt/scripts/run_cmake_configure_matrix.sh \
  --nanobind-source /deps/nanobind \
  --prepare-stage \
  --results-dir /results/cmake-matrix
```

For the published local image used by this PR, mount the checkout read-only, a writable results
directory, and an offline nanobind source:

```bash
docker run --rm --network=none \
  --user "$(id -u):$(id -g)" \
  -e MATRIX_IMAGE=pr9248-gfx950:33012920661 \
  --mount type=bind,source="$PWD",target=/src,readonly \
  --mount type=bind,source="$PWD/.cmake-configure-matrix",target=/results \
  --mount type=bind,source="$NANOBIND_SOURCE",target=/deps/nanobind,readonly \
  --workdir /src \
  pr9248-gfx950:33012920661 \
  bash projects/hipblaslt/scripts/run_cmake_configure_matrix.sh \
    --nanobind-source /deps/nanobind --prepare-stage --results-dir /results
```

Use `--list` for the cell names and `--cell NAME` to run a focused subset. Add `--prepare-stage`
or `--stage-prefix` to include the staged/TheRock cell. The YAML cell requires an LLVM package with
its zstd dependency closure, so add `--with-yaml` (or use `--all`) only in such an environment.
Windows, real TheRock superbuild, and packaged Windows import remain separate platform/integration
validations.

### Matrix JSON artifact

Every invocation writes `<results-dir>/cmake-configure-matrix.json`. It is a schema-versioned,
machine-readable record for reviewing or comparing configured trees; it is not a replacement for
the build directories themselves. The runner starts a fresh aggregate on every invocation, even
with `--keep-results`, so a focused rerun cannot retain stale cells. For every started cell it
retains:

- the resolved CMake configure argv, source/build/stage paths, matrix settings, status, and any
  configure failure error/log tail;
- the full parsed `CMakeCache.txt` (including each entry's CMake type), plus the complete
  `compile_commands.json` when the generator supports it;
- `link.txt` commands when CMake creates them, and native CMake File API `codemodel-v2` target/link
  fragments and `cache-v2` replies as the generator-neutral fallback for Ninja;
- hashes and paths for the relevant Ninja/Make generator files, generated package configs/exports,
  CPack files (kept separately), and reachable install scripts.

The runner forces `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` and creates its CMake File API queries before
each configure. A failed configure leaves a `failed` cell with its partial cache and captured log in
the aggregate instead of discarding the result. If `--prepare-stage` fails, the `staged-provider`
cell is recorded as failed and `hipsparselt-staged` remains as a skipped prerequisite consumer.

Set `MATRIX_IMAGE=<published-image-reference>` with Docker's `-e` option to annotate the artifact
with the image used; it is metadata only and does not alter the configure command.
