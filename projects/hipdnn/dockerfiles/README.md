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

The Dockerfile supports two build types: **prebuilt** (using nightly tarballs) and **fullbuild** (building from source).

### Build Arguments

#### 🔧Select Build Type

| Argument | Default | Description | Valid Values |
|----------|---------|-------------|--------------|
| `BUILD_TYPE` | `prebuilt` | Selects build method | `prebuilt`, `fullbuild` |

#### 📦 Prebuilt-Only Arguments

> [!NOTE]
> Prebuilt mode downloads pre-compiled binaries from TheRock nightly builds (much faster)

> [!NOTE]
> There is currently an issue with using the prebuilt binaries with the gfx90X ASIC family. Refer to this GitHub issue for more details:
https://github.com/ROCm/TheRock/issues/2179


| Argument | Default | Description |
|----------|---------|-------------|
| `THEROCK_GIT_TAG`<br>_(deprecated)_ | `7.0.0rc20250909` | Build tag for [nightly tarballs](https://therock-nightly-tarball.s3.amazonaws.com/). Used to create the precompiled binaries download filename key: `therock-dist-linux-$THEROCK_ASIC-dcgpu-$THEROCK_GIT_TAG.tar.gz`.<br>Alternatively, specify the full prebuilt ID using `THEROCK_PREBUILT_ID`, or the full tarball name using `THEROCK_TARBALL`, below. |
| `THEROCK_ASIC`<br>_(deprecated)_ | `gfx94X` for prebuilt | GPU (dcgpu) architecture target. The values for THEROCK_ASIC for prebuilt mode can be found [here](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#index-page-listing) or parsed directly from the tarball filename keys listed [here](https://therock-nightly-tarball.s3.amazonaws.com/).<br>Future support [here](https://github.com/ROCm/TheRock/blob/main/ROADMAP.md#rocm-on-linux).<br>Note: For prebuilt mode, ensure that the specified dcgpu asic is supported by the prebuilt artifacts as set in `THEROCK_GIT_TAG` described above. |
| `THEROCK_PREBUILT_ID`<br>_(recommended)_ | `$THEROCK_ASIC-dcgpu-$THEROCK_GIT_TAG` | GPU architecture family and build tag. The values for `$THEROCK_ASIC-dcgpu` for prebuilt mode can be found in the _GFX family_ column from the table [here](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#index-page-listing) or parsed directly from the tarball filename keys listed [here](https://therock-nightly-tarball.s3.amazonaws.com/).<br>Future support [here](https://github.com/ROCm/TheRock/blob/main/ROADMAP.md#rocm-on-linux)<br>The resulting tarball filename key will be  `therock-dist-linux-$THEROCK_PREBUILT_ID.tar.gz` |
| `THEROCK_TARBALL` | `therock-dist-linux-gfx94X-dcgpu-7.0.0rc20250909.tar.gz` | Overrides the full tarball path for [nightly tarballs](https://therock-nightly-tarball.s3.amazonaws.com/) (setting this will ignore THEROCK_ASIC, THEROCK_GIT_TAG, and THEROCK_PREBUILT_ID). |

#### 🏗️ Fullbuild-Only Arguments

> [!NOTE]
> Fullbuild mode clones and compiles TheRock from source (will take several hours to complete but more flexible)

| Argument | Default | Description |
|----------|---------|-------------|
| `THEROCK_GIT_HASH` | `default` | Specific git commit hash to checkout (uses default branch if not specified). |
| `ROCM_LIBRARIES_REF` | `default` | Specific git commit hash for rocm-libraries submodule to checkout (uses default branch if not specified). |
| `THEROCK_ASIC` | `gfx90a` for fullbuild | GPU architecture target. The values for THEROCK_ASIC for fullbuild mode can be found in the LLVM Target column of the Supported GPU table [here](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html). |
| `THEROCK_BUILD_MODE` | `Release` | Build mode: `Preset` (uses TheRock presets), `Debug`, or `Release` (uses CMake build types). |
| `THEROCK_BUILD_PRESET` | `linux-release-package` | Specify which build preset to use when THEROCK_BUILD_MODE=Preset. |
| `BUILD_JOBS` | `0` | Number of parallel build jobs. 0 uses all available CPU cores. |

### Build Examples

#### 📦 Prebuilt Mode (Recommended)
The list of latest precompiled tarballs is available [here](https://therock-nightly-tarball.s3.amazonaws.com/)..

**Default prebuilt** (gfx94X, tag 7.0.0rc20250909):
```bash
docker build -f Dockerfile.ubuntu24 -t hipdnn:prebuilt .
```

This will download and install prebuilt binaries using the filename `therock-dist-linux-gfx94X-dcgpu-7.0.0rc20250909.tar.gz` and tags the image as `hipdnn:prebuilt`.

**Override ASIC and Git Tag** (gfx950, tag 7.0.0rc20250908):

```bash
docker build -f Dockerfile.ubuntu24 --build-arg THEROCK_ASIC=gfx950 --build-arg THEROCK_GIT_TAG=7.0.0rc20250908 -t hipdnn:prebuilt_gfx950 .
```

This will download and install prebuilt artifacts using the filename `therock-dist-linux-gfx950-dcgpu-7.0.0rc20250908.tar.gz` and tags the image as `hipdnn:prebuilt_gfx950`.

#### 🏗️ Fullbuild Mode

Note that the full build can take several hours to complete.

**Default fullbuild** (gfx90a, default branch):
```bash
docker build -f Dockerfile.ubuntu24 --build-arg BUILD_TYPE=fullbuild -t hipdnn:fullbuild .
```

**Debug build** (gfx90a, debug mode):
```bash
docker build -f Dockerfile.ubuntu24 --build-arg BUILD_TYPE=fullbuild --build-arg THEROCK_BUILD_MODE=Debug -t hipdnn:debug .
```

**Release build with limited cores** (gfx90a, release mode, 4 cores):
```bash
docker build -f Dockerfile.ubuntu24 --build-arg BUILD_TYPE=fullbuild --build-arg THEROCK_BUILD_MODE=Release --build-arg BUILD_JOBS=4 -t hipdnn:release .
```

**Custom preset with all cores** (gfx90a, custom preset):
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
   rocm-smi
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
