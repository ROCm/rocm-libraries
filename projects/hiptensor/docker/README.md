# hipTensor Docker Images

Two Dockerfiles are provided, both targeting Ubuntu 24.04:

| File | Description |
|------|-------------|
| `Dockerfile.ubuntu24.prebuilt` | Downloads a prebuilt ROCm nightly tarball |
| `Dockerfile.ubuntu24.fullbuild` | Clones and builds ROCm from source (TheRock) |

---

## Prebuilt

Downloads a prebuilt ROCm tarball from `rocm.nightlies.amd.com`. Faster to build; suitable for day-to-day development.

**Default build** (gfx94X, tag pinned in Dockerfile):
```bash
docker build -f Dockerfile.ubuntu24.prebuilt -t hiptensor:prebuilt .
```

**Specify a different ASIC and nightly tag** (format: `<version>a<YYYYMMDD>`):
```bash
docker build -f Dockerfile.ubuntu24.prebuilt \
  --build-arg THEROCK_ASIC=gfx90a \
  --build-arg THEROCK_GIT_TAG=7.13.0a20260401 \
  -t hiptensor:prebuilt .
```

**Override the full tarball filename and source:**
```bash
docker build -f Dockerfile.ubuntu24.prebuilt \
  --build-arg THEROCK_TARBALL=therock-dist-linux-gfx950-dcgpu-7.12.0.tar.gz \
  --build-arg THEROCK_URL_BASE=https://repo.amd.com/rocm/tarball/ \
  -t hiptensor:prebuilt .
```

**Available tarball sources** (`THEROCK_URL_BASE`):

| Source | URL |
|--------|-----|
| Nightly builds (default) | `https://rocm.nightlies.amd.com/tarball/` |
| Stable releases | `https://repo.amd.com/rocm/tarball/` |
| Prereleases (QA) | `https://rocm.prereleases.amd.com/tarball/` |
| Dev builds | `https://rocm.devreleases.amd.com/tarball/` |

See the [TheRock releases page](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#browsing-release-tarballs) for details.

---

## Fullbuild

Clones TheRock from GitHub and builds ROCm from source. Takes significantly longer; useful when a prebuilt tarball is unavailable or when custom build options are needed.

**Default build** (gfx94X, latest default branch):
```bash
docker build -f Dockerfile.ubuntu24.fullbuild -t hiptensor:fullbuild .
```

**Specify a different ASIC and commit hash:**
```bash
docker build -f Dockerfile.ubuntu24.fullbuild \
  --build-arg THEROCK_ASIC=gfx950 \
  --build-arg THEROCK_GIT_HASH=abc123def456 \
  -t hiptensor:fullbuild .
```

**Debug build using 32 parallel jobs:**
```bash
docker build -f Dockerfile.ubuntu24.fullbuild \
  --build-arg THEROCK_BUILD_MODE=Debug \
  --build-arg BUILD_JOBS=32 \
  -t hiptensor:debug .
```

**Build with a custom TheRock CMake preset:**
```bash
docker build -f Dockerfile.ubuntu24.fullbuild \
  --build-arg THEROCK_BUILD_MODE=Preset \
  --build-arg THEROCK_BUILD_PRESET=linux-debug-package \
  -t hiptensor:custom-preset .
```

**Build mode options** (`THEROCK_BUILD_MODE`):

| Mode | Description |
|------|-------------|
| `Release` (default) | Standard optimized build |
| `Debug` | Debug symbols, no optimization |
| `Preset` | Use a named TheRock CMake preset (`THEROCK_BUILD_PRESET`) |

---

## Running a Container

Mount your current directory into `/workspace`:
```bash
docker run -itd --name my-container --rm --network=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v $(pwd):/workspace \
  -v $HOME/.ssh:/root/.ssh/ \
  hiptensor:prebuilt
docker exec -it --privileged my-container /bin/bash
```
