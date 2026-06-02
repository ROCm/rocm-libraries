# tensilelite characterization-test dev env

Reproducible local env for the characterization work. Local-only; no push.

## Why this shape

The base image `rocm-libs-bump-e3b:26531351157` carries the ROCm/HIP
toolchain + Python 3.12 but **no Python deps and no rocisa**. rocisa is a
nanobind/CMake extension whose build links `shared/origami` and
`shared/stinkytofu` — monorepo siblings at `rocisa/../../../../shared/` that
are **not** in the image. So:

- The image (`Dockerfile`) bakes only the **Python deps + build tools**.
- **rocisa is built once against the mounted worktree** (where the siblings
  resolve), via the documented `invoke rocisa` editable workflow. Its build
  tree lives in the worktree, so it survives across container runs.

## 1. Build the deps image (once)

The `work/` dir lives at the WORKTREE ROOT (not under tensilelite). The
Dockerfile COPYs nothing, so use the env dir itself as the (tiny) context:

```sh
ENV=work/tensilelite-characterization/env   # relative to the worktree root
docker build -t tensilelite-char:dev -f "$ENV/Dockerfile" "$ENV"
```

## 2. Start a persistent container with the WORKTREE ROOT mounted

The worktree root (monorepo) must be mounted so `shared/origami` +
`shared/stinkytofu` are visible to rocisa's CMake.

```sh
# WT = monorepo worktree root (…/.claude/worktrees/tensilelite-coverage)
WT=$(git rev-parse --show-toplevel)
TL=projects/hipblaslt/tensilelite

docker run -d --name tl-char \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v "$WT":/work -w /work/$TL \
  tensilelite-char:dev sleep infinity
```
(The `--device`/`--group-add` lines are only needed for GPU/client paths;
group-3 unit characterization is CPU-only and works without them.)

## 3. Build rocisa once (inside the container)

```sh
docker exec -w /work/projects/hipblaslt/tensilelite tl-char \
  invoke rocisa
docker exec tl-char python3 -c "import rocisa; from rocisa.enum import DataTypeEnum; print('rocisa OK')"
```

## 4. Run characterization / coverage (repeatable)

⚠️ Pass `--cov` a value that resolves to an existing **directory** (run from
the tensilelite dir, so `Tensile/SolutionStructs/Validators` resolves).
Coverage uses path mode when the arg matches a dir, but IMPORTS the arg as a
module when it doesn't — and importing a rocisa-touching module
re-initializes the `_rocisa` nanobind extension → SIGABRT
(`nanobind: refusing to add duplicate key`). So a dotted name with no
matching path (e.g. `Tensile.SolutionStructs.Validators`) aborts; the dir
path does not. (Verified for editable and non-editable rocisa installs. The
existing tox coverage envs use `--cov=Tensile --cov=rocisa` from this dir,
where both are dirs, so they are fine.)

```sh
docker exec -e PYTHONPATH=/work/projects/hipblaslt/tensilelite \
  -w /work/projects/hipblaslt/tensilelite tl-char \
  pytest -m unit \
    --cov=Tensile/SolutionStructs/Validators \
    --cov-config=pyproject.toml --cov-report=term-missing \
    Tensile/Tests/unit
```
LD_LIBRARY_PATH is baked into the image, so `import rocisa` works without
passing it. This runs the full `-m unit` suite (1186 passed / 201 skipped,
no regression) and reports coverage scoped to the target — coverage
filtering only affects the report, not what runs.

Pre-existing suite (no-regression check) — same invocation as upstream:
```sh
docker exec -w /work/projects/hipblaslt/tensilelite tl-char \
  pytest -m unit Tensile/Tests/unit
```

## Optional: portable fully-baked image

To snapshot rocisa into the image too, after step 3:
`docker commit tl-char tensilelite-char:rocisa`. Caveat: an editable rocisa
keeps build-tree rpaths/`_build_info` pointing into the mounted worktree, so
the committed image still depends on that mount. A truly portable image would
need a non-editable `pip install ./rocisa` with static origami/stinkytofu —
not validated yet; revisit only if a mount-free image is required.
