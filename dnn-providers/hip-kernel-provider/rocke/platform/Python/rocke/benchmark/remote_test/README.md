# ROCKE Remote Test Orchestrator

Thin layer that **builds ROCKE example artifacts locally for multiple GPU
architectures**, rsyncs them to a slurm login node, and runs them
under `srun` on a compute node that matches the requested arch constraint.

## Why

ROCKE builds HSACO + a small manifest in pure Python on this machine; the
launch/verify step needs a real GPU. By splitting build (anywhere) from
run (on a slurm-allocated node matching `--constraint=GFX942`/`GFX1151`),
we get fast iteration without needing the target GPU locally.

## Layout
```
benchmark/remote_test/
├── config.py    # archs, examples, login host, paths (env-overridable)
├── transport.py # ssh + rsync wrappers (BatchMode, ControlMaster reuse)
├── build.py     # invoke example --no-verify --output-dir <stage>
├── slurm.py     # rsync to login node + srun the run_manifest verifier
└── cli.py       # entry: probe | build | run | all
```

## Prerequisites

1. Passwordless SSH to the login node already works from your shell —
   i.e. `ssh <something>` succeeds without a prompt. The orchestrator
   never embeds host/user/key defaults; nothing site-specific lives in
   the repo.

2. **Recommended — SSH alias.** Put the target in `~/.ssh/config`:
   ```sshconfig
   # ~/.ssh/config (NOT in the repo)
   Host rocke-login
       HostName some-login.internal.example
       User <your-login-user>
       IdentityFile ~/.ssh/id_ed25519
       # Add ProxyJump, Port, etc. here as needed.
   ```
   Then point the orchestrator at the alias:
   ```bash
   export ROCKE_REMOTE_HOST=rocke-login
   # ROCKE_REMOTE_USER stays unset — ssh_config supplies it.
   # Optional: shared FS path on the login node, defaults to ~/rocke_remote
   export ROCKE_REMOTE_STAGE=/path/on/login-node/rocke_remote
   ```

3. **Fallback — explicit host + user** (useful for CI where ssh_config
   isn't convenient):
   ```bash
   export ROCKE_REMOTE_HOST=<slurm-login-host>
   export ROCKE_REMOTE_USER=<login-user>
   export ROCKE_REMOTE_KEY=~/.ssh/id_ed25519   # optional
   export ROCKE_REMOTE_STAGE=/path/on/login-node/rocke_remote   # optional
   ```

4. Compute nodes have a ROCm runtime and `python3 + numpy` available
   (the sinfo features should advertise `ROCM*` on every node you target).
   On the node, `rocminfo` must enumerate **at least one GPU agent**
   (`Device Type: GPU`) — if only the CPU agent shows up, the host's
   `amdgpu` DRM driver hasn't bound to the device and no container
   plumbing will rescue it. Common cause on new silicon (e.g. gfx1151
   Strix Halo / PCI 0x1586): the node's DKMS amdgpu predates the PCI
   ID — needs a ROCm 6.4+ driver bundle. The orchestrator runs
   `rocminfo` as a preflight inside the container and exits 42 with a
   clear message if no GPU agent is found, so this surfaces immediately
   instead of as an opaque `hipInit: hipError(100)`.

5. **Docker image on the compute node.** The verifier runs inside
   `docker run` so the ROCm runtime + python deps come from a known
   image rather than the host's `$PATH`. The compute node's docker
   daemon must already have (or be able to pull) the image.
   ```bash
   # Any image with rocm + python3 + numpy works; the CK CI image is
   # a convenient default:
   export ROCKE_DOCKER_IMAGE=rocm/composable_kernel:ck_ub24.04_rocm7.2
   ```
   Leave `ROCKE_DOCKER_IMAGE` unset to run the verifier directly under
   `bash` on the compute node (only works on hosts where `/dev/kfd` is
   accessible without a container).

   The `docker run` invocation hard-codes the flags ROCm needs:
   `--device=/dev/kfd --device /dev/dri:/dev/dri:rw --group-add video
   --ipc=host -v $HOME:$HOME`. Extra flags can be appended via
   `ROCKE_DOCKER_EXTRA_FLAGS` (space-separated, shlex-parsed). The
   default `.rocke_env` template (below) forwards Slurm's GPU-visibility
   env vars so HIP sees only the gres-allocated GPU:
   ```bash
   export ROCKE_DOCKER_EXTRA_FLAGS="-e HIP_VISIBLE_DEVICES \
     -e ROCR_VISIBLE_DEVICES -e GPU_DEVICE_ORDINAL --group-add render"
   ```

### Suggested layout: site env file outside the repo

Keep site-specific values (hostname, image name, etc.) out of the repo
by sourcing them from a file you own — e.g. `~/.rocke_env`:

```bash
# ~/.rocke_env  (NOT checked in)
export ROCKE_REMOTE_HOST=rocke-login                                 # ssh alias
export ROCKE_DOCKER_IMAGE=rocm/composable_kernel:ck_ub24.04_rocm7.2
export ROCKE_DOCKER_EXTRA_FLAGS="-e HIP_VISIBLE_DEVICES \
  -e ROCR_VISIBLE_DEVICES -e GPU_DEVICE_ORDINAL --group-add render"
export PYTHONPATH=/path/to/rocm-libraries/dnn-providers/hip-kernel-provider/rocKE/Python
```

Then `source ~/.rocke_env` before any `cli` invocation.

## Usage

```bash
source ~/.rocke_env   # see "site env file" above

# 1) Verify SSH + show idle nodes per arch
python -m rocke.benchmark.remote_test.cli probe

# 2) Build HSACO + manifest locally for both archs
python -m rocke.benchmark.remote_test.cli build --arch gfx942,gfx1151

# 3) Push + srun the manifest verifier on each arch
python -m rocke.benchmark.remote_test.cli run   --arch gfx942,gfx1151

# Or do build+run together:
python -m rocke.benchmark.remote_test.cli all   --arch gfx942,gfx1151
```

Per-arch examples and shapes are defined in `config.py::ARCHES`. Add a new
arch by registering a new `ArchProfile` there — no other changes needed.

## Slurm dispatch

`run` issues an `srun` of the shape:

```
srun --partition=defq --constraint=GFX1151 --gres=gpu:1 --time=00:15:00 \
     --job-name=rocke-gfx1151 \
     --output=<stage>/gfx1151/srun.out --error=<stage>/gfx1151/srun.err \
     bash -lc '
       docker run --rm --network=host \
         --device=/dev/kfd --device /dev/dri:/dev/dri:rw \
         --volume /dev/dri:/dev/dri:rw -v /var/lib/docker/:/var/lib/docker \
         --group-add video --ipc=host -v $HOME:$HOME \
         -w <stage>/ck_pkg $ROCKE_DOCKER_EXTRA_FLAGS \
         $ROCKE_DOCKER_IMAGE /bin/bash -lc "
           cd <stage>/ck_pkg
           export PYTHONPATH=<stage>/ck_pkg:\$PYTHONPATH
           python3 -m rocke.run_manifest <hsaco> <manifest> --shape M,N,K --verify"'
```

`--constraint` uses the `AVAIL_FEATURES` token (`GFX942` / `GFX1151`) so any
matching idle node gets picked; `--gres=gpu:1` requests one GPU. The
`docker run` wrapper is skipped if `ROCKE_DOCKER_IMAGE` is unset.

## Layout on the login node

```
$ROCKE_REMOTE_STAGE/
├── ck_pkg/rocke/...     # slim mirror of Python/rocke/ used as PYTHONPATH
├── gfx942/               # hsaco + manifest + run_spec.json + srun.{out,err}
└── gfx1151/
```

## Adding more examples

Edit `config.py::ARCHES` — each entry binds an arch to a module that:
- accepts `--no-verify --output-dir <dir>`
- writes a single `*.hsaco` + `manifest.json` in that dir.

Both currently-wired examples (`rocke.examples.gfx942.gemm_demo` and
`rocke.examples.gfx1151.wmma_gemm_verify`) already satisfy this contract.
