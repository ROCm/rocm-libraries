# TensileLite pre-commit: setup walkthrough

How to install and run the hipblaslt/TensileLite pre-commit hook from a clean
state. The hook runs the unit + characterization tests affected by your staged
TensileLite changes (`scripts/precommit_affected_tests.py`).

## What you are installing

There are **two** pre-commit configs in this repo:

| Config | Scope | What it does |
| --- | --- | --- |
| `/.pre-commit-config.yaml` (repo root) | monorepo lint | black, clang-format, check-yaml, cmake-lint, ... **excludes `projects/hipblaslt/.*`** |
| `projects/hipblaslt/.pre-commit-config.yaml` | hipblaslt only | runs `precommit_affected_tests.py` (the affected unit + char tests) |

You want the **hipblaslt** one for TensileLite work.

### Two facts that bite

1. **One installed hook = one config.** `pre-commit install --config <X>`
   overwrites the single `.git/hooks/pre-commit` file. Installing the hipblaslt
   config **replaces** the root lint config (and vice-versa). You cannot have
   both active from one `pre-commit install`.
2. **Hooks are shared across worktrees.** `.git/hooks` lives in the common git
   dir (`<repo>/.git/hooks`), not per-worktree. Installing/uninstalling from any
   worktree changes the hook for the **main checkout and every worktree**.

## Requirements

The whole setup is two commands, both run **inside a ROCm dev container** (the
hook runs `uv run pytest`, which builds rocisa — a HIP/nanobind native ext —
needing HIP at `/opt/rocm` and a Python with dev headers; the bare host has
neither):

1. `uv sync` — provisions deps + rocisa **and** the `pre-commit` app (it is in
   the `dev` dependency group).
2. `uv run invoke precommit-install` — writes the git hook. uv has no post-sync
   hook, so this is a separate one-time-per-clone step.

Note: `git commit` therefore must be run **from inside the container** (with
`uv` and `LD_LIBRARY_PATH=/opt/rocm/lib`), because the hook shells out to
`uv run pytest`. Committing from the bare host will fail at the rocisa build.

## Steps

All paths are relative to the **repo root** unless noted.

### 1. Provision the env (installs deps, rocisa, and pre-commit)
```bash
cd projects/hipblaslt/tensilelite
uv sync
```

### 2. Install the git hook
```bash
uv run invoke precommit-install
```
This installs the hook pointing at `projects/hipblaslt/.pre-commit-config.yaml`.
If `core.hooksPath` is set to the default hooks dir (a redundant setting that
otherwise makes `pre-commit install` refuse), the task clears it; git-lfs hooks
are unaffected. It bails if `core.hooksPath` points somewhere custom.

Verify:
```bash
grep -- '--config' "$(git rev-parse --git-common-dir)/hooks/pre-commit"
# expect: --config=projects/hipblaslt/.pre-commit-config.yaml
```

### 3. Provision the test backend (inside a ROCm container)

rocisa is a HIP native extension: it will **not** build or import on the bare
host (no HIP cmake config, host Python often lacks dev headers). Use a ROCm
container that has `/opt/rocm` + a headered Python.

#### Verified path: `hipblaslt-tpls:local`
This image has HIP cmake config, Pythons 3.10–3.13 with headers, uv, cmake, g++.
Confirmed working end-to-end (build + import):
Mount the repo at the **same absolute path** inside the container as on the
host. This is required for git worktrees: the worktree's `.git` file holds an
absolute `gitdir:` pointer into `<repo>/.git/worktrees/<name>`, so a different
mount path (e.g. `/work`) makes `git` fail with
`fatal: not a git repository`, which breaks the hook's file detection. Path
parity also keeps the venv's embedded editable-install paths valid — build and
run under the **same** mount, or the `.venv` must be rebuilt.

```bash
# host: launch with the repo mounted at its real path
REPO=/home/davdixon/projects/rocm-libraries
docker run -it --rm \
  -v "$REPO":"$REPO" \
  -w "$REPO"/<path-to-worktree>/projects/hipblaslt/tensilelite \
  hipblaslt-tpls:local bash

# inside container:
rm -rf .venv                 # drop any stale/host-built venv
uv sync                      # uv creates AND populates .venv (deps + rocisa)

# rocisa loads the HIP runtime at import; it lives in /opt/rocm/lib
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
uv run python -c "import rocisa, pytest; print('ok')"   # sanity check
```
Let `uv sync` create `.venv` itself — do **not** pre-make it with
`python -m venv`. A hand-made venv that `uv run` later binds to is left without
the `dev` group, so `pytest` won't be found. Every container Python has dev
headers, so uv's auto-selection is fine here.

`LD_LIBRARY_PATH=/opt/rocm/lib` is required whenever rocisa is imported (tests,
the hook). Export it in the shell you `git commit` from.

The hook runs the tests with **`uv run pytest`** — one tool, no backend
selection or tunables. `uv run` provisions the env from `uv.lock` (deps + the
rocisa build) and runs pytest in it, so the `uv sync` above is really just to
prove it works; the hook would sync on its own. (tox is untouched and still
available if you prefer to run `tox -e unit` by hand — the hook just doesn't use
it.)

### 4. Smoke-test the hook without committing
```bash
cd "$(git rev-parse --show-toplevel)"
pre-commit run --config projects/hipblaslt/.pre-commit-config.yaml \
  tensilelite-affected-tests --all-files
```
Or stage a TensileLite file and `git commit` — the hook runs on staged changes.
Bypass once with `git commit --no-verify`.

## Uninstall / start over
```bash
cd "$(git rev-parse --show-toplevel)"
pre-commit uninstall                                   # removes the shared hook
rm -rf projects/hipblaslt/tensilelite/.venv \          # uv project env
       projects/hipblaslt/tensilelite/rocisa/build \   # compiled rocisa
       projects/hipblaslt/tensilelite/tensile.egg-info
# .tox is also disposable if you used the tox backend:
# rm -rf projects/hipblaslt/tensilelite/.tox
```
Leave `~/.cache/pre-commit` alone unless you want to re-download every hook
repo — it is shared across all your repos, not scoped to this directory.
