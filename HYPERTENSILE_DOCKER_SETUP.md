# Cached sparse ROCm Libraries checkout with GPU Docker access

This setup keeps source changes in a host worktree, uses a host Git mirror as an
object cache, and provides a persistent GPU-enabled container for builds and
tests. Changes are committed and pushed from the host worktree, then pulled
into a matching worktree inside the container.

## Create and update the host Git cache

Create the mirror once at the current cache location:

```bash
mkdir -p "$HOME/.cache/git"

git clone --mirror \
  https://github.com/ROCm/rocm-libraries.git \
  "$HOME/.cache/git/rocm-libraries.git"
```

Refresh the mirror before creating a checkout or updating a container:

```bash
git -C "$HOME/.cache/git/rocm-libraries.git" \
  remote update --prune
```

A clone made with `--reference-if-able` reads existing objects from this
mirror. Objects fetched later by an individual clone are stored in that clone;
they do not update the mirror.

## Create the persistent GPU-enabled container

Verify the host ROCm devices and capture the groups that own them:

```bash
ls -l /dev/kfd /dev/dri

ROCM_VIDEO_GID=$(stat -c '%g' /dev/dri/card0)
ROCM_RENDER_GID=$(stat -c '%g' /dev/kfd)
```

Create the persistent work volume. This is idempotent:

```bash
docker volume create vllm-rcm-a-work
```

Remove an existing container only after confirming important work is stored in
the named `/work` volume:

```bash
docker rm -f vllm-rcm-a
```

Create the container with the minimum device access needed for ROCm:

```bash
export IMAGE_NAME_V=vllm/vllm-openai-rocm:nightly-eed1f3d0c6043bd494424a22443ee198dd56f657

docker run -d \
  --name vllm-rcm-a \
  --restart unless-stopped \
  --entrypoint /bin/sh \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add "$ROCM_VIDEO_GID" \
  --group-add "$ROCM_RENDER_GID" \
  --mount type=bind,src=/home/alvasile/.cache/git,dst=/home/alvasile/.cache/git,readonly \
  --mount type=volume,src=vllm-rcm-a-work,dst=/work \
  "$IMAGE_NAME_V" \
  -c 'while :; do sleep 86400; done'
```

Trust the host-owned mirror from the root user inside the container:

```bash
docker exec vllm-rcm-a \
  git config --global --add safe.directory \
  /home/alvasile/.cache/git/rocm-libraries.git
```

Verify GPU access through PyTorch:

```bash
docker exec vllm-rcm-a python3 -c '
import torch
print(f"torch={torch.__version__}")
print(f"hip={torch.version.hip}")
print(f"available={torch.cuda.is_available()}")
print(f"count={torch.cuda.device_count()}")
assert torch.cuda.is_available() and torch.cuda.device_count() > 0
'
```

## Create the cached sparse clone inside Docker

```bash
docker exec vllm-rcm-a sh -lc '
cd /work

git clone \
  --sparse \
  --filter=blob:none \
  --reference-if-able /home/alvasile/.cache/git/rocm-libraries.git \
  https://github.com/ROCm/rocm-libraries.git \
  rocm-libraries

git -C /work/rocm-libraries sparse-checkout set \
  projects/hipblaslt \
  projects/hipsparselt \
  projects/rocblas \
  shared/origami \
  shared/rocroller \
  shared/mxdatagenerator \
  shared/stinkytofu \
  cmake \
  .github
'
```

Confirm the clone uses the mirror and the real remote:

```bash
docker exec vllm-rcm-a sh -lc '
cd /work/rocm-libraries
cat .git/objects/info/alternates
git remote get-url origin
git status --short --branch
'
```

## Create the host HyperTensile worktree

Run these commands on the host from the main ROCm Libraries checkout:

```bash
cd /home/alvasile/rocm-libraries

git fetch origin develop

git worktree add \
  --no-checkout \
  --no-track \
  -b users/alvasile/hypertensile \
  .codex/worktrees/users-alvasile-hypertensile \
  origin/develop

cd .codex/worktrees/users-alvasile-hypertensile

git sparse-checkout init --cone

git sparse-checkout set \
  projects/hipblaslt \
  projects/hipsparselt \
  projects/rocblas \
  shared/origami \
  shared/rocroller \
  shared/mxdatagenerator \
  shared/stinkytofu \
  cmake \
  .github

git read-tree -mu HEAD
```

Make, commit, and push source changes from this host worktree:

```bash
cd /home/alvasile/rocm-libraries/.codex/worktrees/users-alvasile-hypertensile

git status --short --branch
git add <paths>
git diff --cached --check
git commit
git push --set-upstream origin users/alvasile/hypertensile
```

## Pull the host branch into Docker

Create a matching Docker worktree after the host branch has been pushed:

```bash
docker exec vllm-rcm-a sh -lc '
git -C /work/rocm-libraries fetch origin users/alvasile/hypertensile
mkdir -p /work/worktrees

git -C /work/rocm-libraries worktree add \
  --no-checkout \
  --track \
  -b users/alvasile/hypertensile \
  /work/worktrees/users-alvasile-hypertensile \
  origin/users/alvasile/hypertensile

git -C /work/worktrees/users-alvasile-hypertensile \
  sparse-checkout init --cone

git -C /work/worktrees/users-alvasile-hypertensile \
  sparse-checkout set \
  projects/hipblaslt \
  projects/hipsparselt \
  projects/rocblas \
  shared/origami \
  shared/rocroller \
  shared/mxdatagenerator \
  shared/stinkytofu \
  cmake \
  .github

git -C /work/worktrees/users-alvasile-hypertensile \
  read-tree -mu HEAD
'
```

After later host pushes, update the Docker worktree with:

```bash
docker exec vllm-rcm-a \
  git -C /work/worktrees/users-alvasile-hypertensile \
  pull --ff-only
```

Enter the Docker worktree for builds or tests:

```bash
docker exec -it \
  -w /work/worktrees/users-alvasile-hypertensile \
  vllm-rcm-a \
  bash
```

The corresponding locations are:

```text
Host worktree:       /home/alvasile/rocm-libraries/.codex/worktrees/users-alvasile-hypertensile
Docker worktree:     /work/worktrees/users-alvasile-hypertensile
Host object cache:   /home/alvasile/.cache/git/rocm-libraries.git
```
