# hipTensor Docker Images

Two Dockerfiles are provided, both targeting Ubuntu 24.04:

| File                              | Description                                                    |
|-----------------------------------|----------------------------------------------------------------|
| `Dockerfile.ubuntu24.prebuilt`    | Downloads a prebuilt ROCm tarball (nightly, stable, or latest) |
| `Dockerfile.ubuntu24.fullbuild`   | Clones and builds ROCm from source (TheRock)                   |

---

## Prebuilt

Downloads a prebuilt ROCm tarball from a configurable source (`THEROCK_URL_BASE`). Faster to build; suitable for day-to-day development.

**Default build** (gfx94X, latest available nightly):
```bash
docker build -f Dockerfile.ubuntu24.prebuilt -t hiptensor:prebuilt .
```

`THEROCK_GIT_TAG` defaults to `latest`, which queries `THEROCK_URL_BASE` at build time, finds all tarballs matching the ASIC prefix, and downloads the highest version. ADHOCBUILD tarballs are excluded.

**Select another ASIC** :
```bash
docker build -f Dockerfile.ubuntu24.prebuilt \
  --build-arg THEROCK_ASIC=gfx950 \
  -t hiptensor:prebuilt .
```

**Pin to a specific nightly** (format: `<version>a<YYYYMMDD>`):
```bash
docker build -f Dockerfile.ubuntu24.prebuilt \
  --build-arg THEROCK_ASIC=gfx950 \
  --build-arg THEROCK_GIT_TAG=7.13.0a20260401 \
  -t hiptensor:prebuilt .
```

**Use stable releases instead of nightlies:**
```bash
docker build -f Dockerfile.ubuntu24.prebuilt \
  --build-arg THEROCK_GIT_TAG=latest \
  --build-arg THEROCK_URL_BASE=https://repo.amd.com/rocm/tarball/ \
  -t hiptensor:prebuilt .
```

**Pin to a specific stable release** (format: `<version>`):
```bash
docker build -f Dockerfile.ubuntu24.prebuilt \
  --build-arg THEROCK_GIT_TAG=7.12.0 \
  --build-arg THEROCK_URL_BASE=https://repo.amd.com/rocm/tarball/ \
  -t hiptensor:prebuilt .
```

**Available tarball sources** (`THEROCK_URL_BASE`):

| Source                   | URL                                           |
|--------------------------|-----------------------------------------------|
| Nightly builds (default) | `https://rocm.nightlies.amd.com/tarball/`     |
| Stable releases          | `https://repo.amd.com/rocm/tarball/`          |
| Prereleases (QA)         | `https://rocm.prereleases.amd.com/tarball/`   |
| Dev builds               | `https://rocm.devreleases.amd.com/tarball/`   |

See the [TheRock releases page](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#browsing-release-tarballs) for details.

### ASIC suffix note

The tarball suffix (`-dcgpu`, `-dgpu`, `-all`, or none) is derived automatically from `THEROCK_ASIC`:

| ASIC(s)                                                            | Auto-selected suffix |
|--------------------------------------------------------------------|----------------------|
| `gfx94X`, `gfx90X`, `gfx950`                                       | `-dcgpu`             |
| `gfx101X`, `gfx103X`                                               | `-dgpu`              |
| `gfx110X`, `gfx120X`                                               | `-all`               |
| `gfx900`, `gfx906`, `gfx908`, `gfx90a`, `gfx1150`–`gfx1153`        | _(none)_             |

Use `THEROCK_PREBUILT_ID` only if you need a non-default suffix (e.g. `gfx110X-dgpu` instead of `gfx110X-all`):

```bash
docker build -f Dockerfile.ubuntu24.prebuilt \
  --build-arg THEROCK_PREBUILT_ID=gfx110X-dgpu \
  -t hiptensor:prebuilt .
```

### Providing a full tarball URL

If the tarball source uses a non-standard naming convention, bypass all auto-detection by providing the complete download URL via `THEROCK_TARBALL_URL`. All other `THEROCK_*` arguments are ignored when this is set:

```bash
docker build -f Dockerfile.ubuntu24.prebuilt \
  --build-arg THEROCK_TARBALL_URL=https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx950-dcgpu-7.12.0.tar.gz \
  -t hiptensor:prebuilt .
```

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

| Mode                | Description                                               |
|---------------------|-----------------------------------------------------------|
| `Release` (default) | Standard optimized build                                  |
| `Debug`             | Debug symbols, no optimization                            |
| `Preset`            | Use a named TheRock CMake preset (`THEROCK_BUILD_PRESET`) |

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
