# hipBLASLt scripts

For full build prerequisites and installation, see the [project README](../README.md) and [Building and installing hipBLASLt](../docs/install/building-installing-hipblaslt.rst).

## run_tensile_logic_check.py

Runs **`tensilelite logic --check-all`** on the library logic YAMLs (the same check as the pre-build gate). This file is cross-platform (Windows and Unix).

### How to run

From the **hipblaslt project root** (where `library/` and `tensilelite/` live):

```bash
python scripts/run_tensile_logic_check.py
```

**Windows**: `python scripts\run_tensile_logic_check.py` (or use `py` if you have the launcher).

The selected Python must already provide the `tensilelite` and `rocisa`
packages. The Python script does not inject checkout or build-tree paths and
does not switch interpreters.

### One-time setup

From `projects/hipblaslt/tensilelite`, prepare the active environment with
`invoke install --gpu-targets <gfx target>`. Alternatively, install compatible
TensileLite and rocisa packages explicitly before invoking this script.

### Optional: check a single directory

```bash
python scripts/run_tensile_logic_check.py library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/navi33/GridBased
```

### Known-bugs list (ROCM-7144 / validation exceptions)

This script and the CMake pre-build gate explicitly enable the bundled known-bugs list for `tensilelite logic`, loaded through package resources, so specific `(logic file path, solution_name)` pairs are skipped. `solution_name` is a solution's `SolutionNameMin`, a content-derived name that stays stable when the library is re-tuned (the positional `SolutionIndex` is not stable, so it is no longer used as the key). Paths in the `known_bugs.yaml` are relative to the library logic root (`library/`), with optional `#` comments and an optional `ticket:` field for Jira keys. Override the list by passing your own `--known-bugs` file; pass an empty YAML file to disable all bundled entries. A direct `tensilelite logic --check-all` invocation applies no known-bug skips unless it is given `--known-bugs FILE` or `--use-bundled-known-bugs`.

Documented known bugs are still re-validated on every run instead of being blindly skipped. If a listed solution **now passes** validation (the underlying bug was fixed), the run prints a `Stale known-bugs` warning naming the entry to remove. Pass **`--strict-known-bugs`** to make the run exit non-zero on any stale entry; use that in CI or in the PR that lands the fix, so the fixing PR also removes the listing.

### Exit code

- **0** – All solutions passed (Reject = 0).
- **1** – One or more failed; errors are printed.

The full tree (~2246 files) can take several minutes, so passing a subdirectory is useful for quick checks. To tune parallelism: `python scripts/run_tensile_logic_check.py -j 16` (default is 48 workers, capped by CPU count).

### run_tensile_logic_check.sh (Unix only)

Thin wrapper that selects `.venv/bin/python`, if present, and otherwise uses
`python3`. The selected interpreter must satisfy the installed-package contract
above. Use the `.py` script directly on Windows.
