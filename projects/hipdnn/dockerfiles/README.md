# hipDNN Docker Environments

This directory contains the Dockerfile for building the hipDNN development environment Docker image.

## 📋 Prerequisites

- **Docker with Buildx installed** (Version 27.5.1+ recommended)
    - **Why?** This Dockerfile relies on BuildKit (Buildx) to selectively build only the necessary stages. Legacy builders or older versions may attempt to build *all* stages—including those with missing arguments—causing build failures.
- **ROCm-compatible GPU**
    - Required for running applications with GPU support.
- **Sufficient disk space**
    - Required for storing development Docker images.

## 🐳 Ubuntu 24.04

- **File**: [`Dockerfile.ubuntu24`](Dockerfile.ubuntu24)
- **Base Image**: Ubuntu 24.04 LTS
- **Purpose**: Primary development environment for hipDNN
- **Includes**:
  - ROCm development tools
  - CMake build system
  - Google Test framework
  - Ninja build tool

## 🔨 Building the Docker Image

The Dockerfile supports two build types: **prebuilt** (using current per-family multi-arch tarballs) and **fullbuild** (building from source).

### Build Arguments

#### 🔧Select Build Type

| Argument | Default | Description | Valid Values |
|----------|---------|-------------|--------------|
| `BUILD_TYPE` | `prebuilt` | Selects build method | `prebuilt`, `fullbuild` |

#### 🧩 Kernel Packing Tooling

The `devshell` and `hipdnn` images stage the `rocm_kpack` Python tooling at
**`/opt/rocm-kpack/python`**, together with its `msgpack` and `zstandard` dependencies. The
hip-kernel-provider build finds this path on its own, so kernel packing works inside the container
with no CMake flags and no network access at build time. (The `fullbuildshell` target builds from
`base` and does not include it.)

Passing `-DHIPKERNELPROVIDER_KPACK_PYTHON_DIR=<dir>` overrides the staged copy — useful when
building against a local rocm-systems checkout. See
[Building.md](../docs/Building.md#kernel-packing-rocm_kpack).

| Argument | Default | Description |
|----------|---------|-------------|
| `ROCM_SYSTEMS_KPACK_REF` | a pinned commit | Git ref of [rocm-systems](https://github.com/ROCm/rocm-systems) to take `shared/kpack` from. Only `shared/kpack` is checked out. A branch, a tag, or a **full** commit SHA — the git wire protocol cannot fetch an abbreviated one. Independent of `ROCM_SYSTEMS_REF`, which selects TheRock's rocm-systems submodule. |

> [!NOTE]
> The default is a pinned commit, not a branch: Docker keys a layer on the command string, so a
> branch default would let a warm builder cache ship stale tooling. Pass
> `ROCM_SYSTEMS_KPACK_REF=develop` to track the branch. The staged commit is recorded at
> `/opt/rocm-kpack/ROCM_SYSTEMS_COMMIT`.

> [!NOTE]
> The tooling is staged outside `/opt/rocm` because the `devshell` target expects a ROCm install
> to be mounted at `/opt/rocm` at run time, which would shadow anything placed there at build time.

```bash
# Stage kpack from a specific rocm-systems commit
docker build -f Dockerfile.ubuntu24 \
    --build-arg ROCM_SYSTEMS_KPACK_REF=a022846cf553c2b135410a5168f97705f1b9c6ac \
    -t hipdnn:prebuilt .
```

#### 📦 Prebuilt-Only Arguments

> [!NOTE]
> Prebuilt mode downloads pre-compiled binaries through TheRock's `install_rocm_from_artifacts.py` from the current nightly or dev multi-arch tarball feed (much faster than building from source).

> [!NOTE]
> There is currently an issue with using the prebuilt binaries with the gfx90X ASIC family. Refer to this GitHub issue for more details:
https://github.com/ROCm/TheRock/issues/2179

| Argument | Default | Description |
|----------|---------|-------------|
| `THEROCK_RELEASE` | `latest` | Release version to install. Use `latest` to automatically fetch the newest nightly build, or specify a nightly version such as `7.12.0a20260202` or a commit-addressed dev version such as `7.15.0.dev0+<sha>`. TheRock selects the appropriate multi-arch feed. |
| `THEROCK_ASIC` | `gfx94X` | GPU architecture family prefix. Combined with `THEROCK_ASIC_VARIANT` to form the artifact group. |
| `THEROCK_ASIC_VARIANT` | `dcgpu` | GPU variant suffix (e.g., `dcgpu`, `all`, `dgpu`). Combined with `THEROCK_ASIC` to form the artifact group. |
| `THEROCK_ARTIFACT_GROUP` | `$THEROCK_ASIC-$THEROCK_ASIC_VARIANT` | Full artifact group override. The default `gfx94X-dcgpu` selects the current per-family multi-arch tarball. Use `multiarch` only when all GPU kernel packs are required; it is substantially larger. Choose other per-family group names from the [tarball feed](https://rocm.nightlies.amd.com/tarball-multi-arch/). |

#### Version Logging

The prebuilt stage writes the installed TheRock version to `/opt/rocm/THEROCK_VERSION` inside the image. For pinned releases, this contains the exact version string. For `latest` builds, it captures version information from the install output.

#### 🏗️ Fullbuild-Only Arguments

> [!NOTE]
> Fullbuild mode clones and compiles TheRock from source (will take several hours to complete but more flexible)

| Argument | Default | Description |
|----------|---------|-------------|
| `THEROCK_GIT_HASH` | `default` | Specific git commit hash to checkout (uses default branch if not specified). |
| `ROCM_LIBRARIES_REF` | `default` | Specific git commit hash for rocm-libraries submodule to checkout (uses default branch if not specified). |
| `THEROCK_ASIC` | `gfx94X` | GPU architecture target. The values for THEROCK_ASIC for fullbuild mode can be found in the LLVM Target column of the Supported GPU table [here](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html). |
| `THEROCK_BUILD_MODE` | `Release` | Build mode: `Preset` (uses TheRock presets), `Debug`, or `Release` (uses CMake build types). |
| `THEROCK_BUILD_PRESET` | `linux-release-package` | Specify which build preset to use when THEROCK_BUILD_MODE=Preset. |
| `BUILD_JOBS` | `0` | Number of parallel build jobs. 0 uses all available CPU cores. |

### Build Examples

#### 📦 Prebuilt Mode (Recommended)

**Default prebuilt** (latest nightly current per-family multi-arch tarball for gfx94X-dcgpu):
```bash
docker build -f Dockerfile.ubuntu24 -t hipdnn:prebuilt .
```

This automatically downloads and installs the latest nightly `gfx94X-dcgpu` artifact group. It does not use the heavyweight `multiarch` group, which is only for images that need all GPU kernel packs.

**Specific release version**:
```bash
docker build -f Dockerfile.ubuntu24 \
    --build-arg THEROCK_RELEASE=7.12.0a20260202 \
    -t hipdnn:prebuilt .
```

**Specific dev release**:
```bash
docker build -f Dockerfile.ubuntu24 \
    --build-arg THEROCK_RELEASE=7.15.0.dev0+8b3042421581dc0927d7a1278f96f2deeba75286 \
    --build-arg THEROCK_ARTIFACT_GROUP=gfx90a \
    -t hipdnn:prebuilt_dev_gfx90a .
```

**MI200 (gfx90a)**:
```bash
docker build -f Dockerfile.ubuntu24 \
    --build-arg THEROCK_ARTIFACT_GROUP=gfx90a \
    -t hipdnn:prebuilt_gfx90a .
```

**Full artifact group override** (all GPU kernel packs):
```bash
docker build -f Dockerfile.ubuntu24 \
    --build-arg THEROCK_ARTIFACT_GROUP=multiarch \
    -t hipdnn:prebuilt_multiarch .
```

#### 🏗️ Fullbuild Mode

Note that the full build can take several hours to complete.

**Default fullbuild** (gfx94X, default branch):
```bash
docker build -f Dockerfile.ubuntu24 --build-arg BUILD_TYPE=fullbuild -t hipdnn:fullbuild .
```

**Debug build** (gfx94X, debug mode):
```bash
docker build -f Dockerfile.ubuntu24 --build-arg BUILD_TYPE=fullbuild --build-arg THEROCK_BUILD_MODE=Debug -t hipdnn:debug .
```

**Release build with limited cores** (gfx94X, release mode, 4 cores):
```bash
docker build -f Dockerfile.ubuntu24 --build-arg BUILD_TYPE=fullbuild --build-arg THEROCK_BUILD_MODE=Release --build-arg BUILD_JOBS=4 -t hipdnn:release .
```

**Custom preset with all cores** (gfx94X, custom preset):
```bash
docker build -f Dockerfile.ubuntu24 --build-arg BUILD_TYPE=fullbuild --build-arg THEROCK_BUILD_MODE=Preset --build-arg THEROCK_BUILD_PRESET=linux-debug-package -t hipdnn:custom-preset .
```

**Custom ASIC and Git Hash** (gfx950, hash abcd1234):
```bash
docker build -f Dockerfile.ubuntu24 --build-arg BUILD_TYPE=fullbuild --build-arg THEROCK_ASIC=gfx950 --build-arg THEROCK_GIT_HASH=abcd1234 -t hipdnn:custom_source_build .
```

## 🚀 Running the Containers

### Basic Run Command

To run the hipDNN docker environment with full GPU support:

```bash
docker run -it \
  -v /path/to/hipdnn/repo:/workspace/hipdnn \
  --privileged \
  --rm \
  --device=/dev/kfd \
  --device /dev/dri:/dev/dri:rw \
  --volume /dev/dri:/dev/dri:rw \
  -v /var/lib/docker/:/var/lib/docker \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  <image-name>
```

### Command Options Explained

| Option | Purpose |
|--------|---------|
| `-it` | Interactive terminal |
| `-v /path/to/hipdnn/repo:/workspace/hipdnn` | Mount hipDNN source code |
| `--privileged` | Required for GPU access |
| `--rm` | Remove container after exit |
| `--device=/dev/kfd` | AMD GPU compute device |
| `--device /dev/dri:/dev/dri:rw` | Direct Rendering Infrastructure for GPU |
| `--group-add video` | Add container to video group for GPU access |
| `--cap-add=SYS_PTRACE` | Enable debugging capabilities |
| `--security-opt seccomp=unconfined` | Required for some debugging tools |
| `<image-name>` | The tag added to the image with the `-t` option in the build command |

### Example Run Commands

1. Switch to the `rocm-libraries/projects/hipdnn` folder.
2. Using image tagged `hipdnn:prebuilt`:
```bash
docker run -it \
  -v $(pwd):/workspace/hipdnn \
  --privileged \
  --rm \
  --device=/dev/kfd \
  --device /dev/dri:/dev/dri:rw \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  hipdnn:prebuilt
```
### Building hipDNN

Follow the [quick start steps in the build guide](../docs/Building.md#quick-start-guide) to build hipDNN.

## 💡 Tips and Best Practices

1. **Volume Mounting**: Mount your hipDNN source code directory to persist changes:
   ```bash
   -v /absolute/path/to/hipdnn:/workspace/hipdnn
   ```

2. **GPU Verification**: Once inside the container, verify GPU access:
   ```bash
   amd-smi
   ```
   or
   ```bash
   rocminfo
   ```

3. **Development Workflow**: The containers include all necessary development tools, so you can edit code on your host machine and build/test inside the container.

## ⚠️ Troubleshooting

### GPU Not Detected
- Ensure ROCm is properly installed on the host system
- Verify that `/dev/kfd` and `/dev/dri` exist on the host
- Check that your user is in the `video` and `render` groups

### Permission Issues
- Make sure to use the `--privileged` flag
- Verify that the mounted directories have appropriate permissions

### Build Failures
- Ensure all submodules are initialized: `git submodule update --init --recursive`
- Check that the ROCm version in the container matches your GPU requirements

### Fullbuild not updating to latest version of TheRock
- Since the docker build doesn't change when the cloned source is updated, you will need to either provide a new hash for the docker build to rebuild `--build-arg THEROCK_GIT_HASH=abcd1234`, or provide `--no-cache` option when building to force rebuild

### Build issues

**Build fails attempting to build all stages / Argument mismatches:**
- **Symptom:** The build attempts to execute stages you didn't select (e.g., compiling from source in `fullbuild` when you requested `prebuilt`), often failing due to default arguments or configuration conflicts in those unselected stages.
- **Cause:** You are likely using the legacy Docker builder instead of BuildKit (Buildx). The legacy builder attempts to process all stages, whereas BuildKit only builds what is necessary for the target.
- **Solution:**
  - Ensure Docker Buildx is installed and enabled (`docker buildx version`).
  - Upgrade Docker to the recommended version (27.5.1+).
  - If using an older version, try explicitly enabling BuildKit: `DOCKER_BUILDKIT=1 docker build ...`
