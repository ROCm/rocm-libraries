---
name: worktree-venv
description: "Give each git worktree its own Python venv for any rocm-libraries package (rocisa, stinkytofu bindings, ...) without duplicating a multi-GB pip ROCm SDK. Use when concurrent worktrees fight over one shared venv, an editable install from one worktree shadows another, a fresh worktree needs its own build, or 'invoke <task>' picks up the wrong tree."
argument-hint: "[worktree path]"
allowed-tools: Bash, Read, Glob
---

# Per-worktree Python venv

## The problem this solves

An editable install of a compiled package writes a **single global** finder into
the active venv's `site-packages` — for scikit-build-core, `_editable_skbc_<pkg>.pth`
plus a `.py` shim. That shim hard-codes **one** build directory.

With one venv shared across worktrees, whichever worktree installed last owns it.
The others silently import that worktree's binary. Nothing errors: for a code
generator the result is wrong generated output, not a crash.

It is worse with rebuild-on-import (`editable.rebuild`): importing from worktree
B runs `cmake --build` in worktree A's tree.

## Detect before acting — two machine shapes

| ROCm install | How it is found | Layout |
|---|---|---|
| pip wheel (~11GB, inside a venv) | `rocm-sdk path --root` | worktree venv **shares** the base venv's `site-packages` |
| system install | `$ROCM_PATH`, or `amdclang++`/`hipcc`/`hipconfig` on `PATH` | plain **standalone** venv |

Only the pip SDK justifies sharing — it is huge and lives *inside* a venv. A
system install is already outside any venv and costs nothing to reuse.

Probe order (matches `tensilelite/tasks.py::_detect_rocm` so the two cannot
disagree): `rocm-sdk path --root` → `$ROCM_PATH` → marker binaries on `PATH` →
legacy globs (`/opt/rocm*`, `C:\Program Files\AMD\ROCm\*`) → **ask the user**.
Never guess; when non-interactive it exits telling you to pass `--rocm-path`.

## Usage

```bash
python skills/worktree-venv/scripts/setup_worktree_venv.py <worktree>
```

Options: `--base-venv` (default `$BASE_VENV` or `~/.venv`), `--name` (default
`.venv`), `--rocm-path` to skip detection, `--standalone` to force no sharing.

Then per worktree:

```bash
source <worktree>/.venv/bin/activate      # Windows: <worktree>\.venv\Scripts\activate
cd <worktree>/projects/hipblaslt/tensilelite
invoke rocisa                              # installs into THIS worktree only
```

For a **standalone** venv, install the project's own `requirements.txt` first —
`tasks.py` runs `pip install --no-build-isolation`, so the build backend must
already be importable. A shared venv inherits it from the base.

## Verify

```bash
python -c "import rocisa; print(rocisa._rocisa.__file__)"
```

Must resolve under the **current** worktree. Verified end-to-end: a full
`invoke rocisa` in a script-created worktree venv builds stinkytofu (120/120)
and rocisa, and installs only into that worktree — the base venv is untouched.

## Two non-obvious traps

Both were found by actually running a build; neither shows up in import tests.

**1. `rocm-sdk path --bin` is not usable as a `PATH` entry.** The pip SDK's
`_rocm_sdk_devel/bin/amdclang++` is a hardlink of `lib/llvm/bin/amdclang++`, but
its companion `amdllvm` was not hardlinked alongside it, so it dies with
`could not exec …/bin/amdllvm`. Same for `amdclang`, `amdflang`, `amdlld`. This
is independent of venv activation — the driver resolves `amdllvm` from its own
directory, not from `PATH`. The working entry points are the **console-script
trampolines in the base venv's `bin/`**. So for a wheel SDK use `--root` (for
`ROCM_PATH`) and `--cmake` (for `CMAKE_PREFIX_PATH`), and put the *venv's* bin
on `PATH`. Upstream: ROCm/TheRock#7141.

**2. Where the ROCm bin goes in `PATH` decides which venv gets installed into.**
For a wheel SDK the ROCm bin *is* the base venv's bin, which also contains
`pip`, `python` and `invoke`. Prepending it makes `pip install -e` write into
the **base** venv — silently recreating the exact cross-worktree contamination
this skill exists to prevent. The bootstrap therefore inserts the ROCm bin
*after* this venv's own bin, and injects this venv's bin first when running
unactivated (`venv/bin/python -m invoke`).

## Other gotchas

- **`bin`/`Scripts` is never shared** — only `site-packages` is reached.
- **Never `pip install <pkg>` non-editable into the base venv.** It sits second
  on `sys.path` and silently serves a stale binary to any worktree without its
  own install. Keep the base venv for shared deps only.
- **`ROCM_PATH` is needed at import time**, not just build time, when
  `editable.rebuild` is on — `import` shells out to CMake. The bootstrap covers
  anything using the venv's interpreter; external CI runners need it set too.
- Ensure `.venv/` is ignored in the worktree.

## How it works

Two files are written into the worktree venv's `site-packages`:

- `_worktree_base_env.py` — appends the base `site-packages` to `sys.path`
  (append, never insert, so the worktree always wins), and sets `ROCM_PATH`,
  `PATH` and `CMAKE_PREFIX_PATH`.
- `zz_worktree_base_env.pth` — a one-line `import _worktree_base_env`.

A `.pth` line starting with `import` is executed at interpreter startup, so this
applies to **every** entry point — `python`, `pytest`, IDE runners, and anything
they spawn — not just an activated shell. Keeping the logic in a real module
rather than a one-liner keeps it debuggable; same shape scikit-build-core uses
for its own editable shim.

Critically, a `.pth` **path entry** does not run `site.addsitedir()`, so `.pth`
files inside the base `site-packages` are never executed. The base venv's
editable finder therefore cannot hijack the worktree — verified: no
`ScikitBuild` finder appears on the worktree's `sys.meta_path`.

Platform layout differences (`bin` vs `Scripts`, `lib/pythonX.Y` vs `Lib`) are
resolved by asking each interpreter for its own `sysconfig` paths.
