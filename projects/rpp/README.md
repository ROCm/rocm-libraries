[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![doc](https://img.shields.io/badge/doc-rocm.docs.amd.com-blueviolet)](https://rocm.docs.amd.com/projects/rpp/en/latest/index.html)

<p align="center"><img width="70%" src="docs/data/AMD_RPP_logo.png" /></p>

AMD ROCm Performance Primitives (RPP) library is a comprehensive, high-performance computer
vision library for AMD processors that have `HIP`, or `CPU` backends.

<p align="center"><img width="35%" src="docs/data/rpp_structure_4.png" /></p>

> [!NOTE]
> Starting with ROCm 10.1, RPP is built and delivered as part of
> [TheRock](https://github.com/ROCm/TheRock), the unified ROCm build system.
> Earlier standalone RPP releases were delivered with ROCm 7.2.x and prior.
>
> RPP source now lives in the [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries)
> monorepo under [`projects/rpp`](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rpp).
> It was previously developed in the standalone `ROCm/rpp` repository.

## Supported Augmentations / Primitives

RPP supports various 2D image, 3D image (voxel), and audio augmentations and primitives. The tables below show CPU (HOST) and GPU (HIP) support for each functionality. For the authoritative, always up-to-date list, see [Supported functionalities and variants](https://rocm.docs.amd.com/projects/rpp/en/latest/reference/rpp-supported-functionalities.html) in the documentation.

<details>
<summary><b>Color augmentations</b></summary>

| Type | CPU | GPU |
|------|:---:|:---:|
| brightness | ✅ | ✅ |
| gamma correction | ✅ | ✅ |
| blend | ✅ | ✅ |
| hue | ✅ | ✅ |
| saturation | ✅ | ✅ |
| color twist | ✅ | ✅ |
| color jitter | ✅ | ❌ |
| color cast | ✅ | ✅ |
| exposure | ✅ | ✅ |
| contrast | ✅ | ✅ |
| lut | ✅ | ✅ |
| color temperature | ✅ | ✅ |
| histogram equalize | ✅ | ✅ |

</details>

<details>
<summary><b>Effects augmentations</b></summary>

| Type | CPU | GPU |
|------|:---:|:---:|
| gridmask | ✅ | ✅ |
| spatter | ✅ | ✅ |
| salt and pepper noise | ✅ | ✅ |
| shot noise | ✅ | ✅ |
| gaussian noise | ✅ | ✅ |
| non-linear blend | ✅ | ✅ |
| water | ✅ | ✅ |
| ricap | ✅ | ✅ |
| vignette | ✅ | ✅ |
| jitter | ✅ | ✅ |
| erase | ✅ | ✅ |
| random erase | ✅ | ✅ |
| glitch | ✅ | ✅ |
| rain | ✅ | ✅ |
| pixelate | ✅ | ✅ |
| fog | ✅ | ✅ |
| posterize | ✅ | ✅ |
| solarize | ✅ | ✅ |
| snow | ✅ | ✅ |
| channel dropout | ✅ | ✅ |
| cutout dropout | ✅ | ✅ |
| grid dropout | ✅ | ✅ |
| coarse dropout | ✅ | ✅ |

</details>

<details>
<summary><b>Geometric augmentations</b></summary>

| Type | CPU | GPU |
|------|:---:|:---:|
| crop | ✅ | ✅ |
| crop mirror normalize | ✅ | ✅ |
| crop and patch | ✅ | ✅ |
| flip | ✅ | ✅ |
| resize | ✅ | ✅ |
| resize mirror normalize | ✅ | ✅ |
| resize crop mirror | ✅ | ✅ |
| rotate | ✅ | ✅ |
| warp affine | ✅ | ✅ |
| warp perspective | ✅ | ✅ |
| lens correction | ✅ | ✅ |
| fisheye | ✅ | ✅ |
| phase | ✅ | ✅ |
| slice | ✅ | ✅ |
| remap | ✅ | ✅ |
| transpose | ✅ | ✅ |
| concat | ✅ | ✅ |
| jpeg compression distortion | ✅ | ✅ |

</details>

<details>
<summary><b>Morphological operations</b></summary>

| Type | CPU | GPU |
|------|:---:|:---:|
| erode | ✅ | ✅ |
| dilate | ✅ | ✅ |

</details>

<details>
<summary><b>Filter augmentations</b></summary>

| Type | CPU | GPU |
|------|:---:|:---:|
| box filter | ✅ | ✅ |
| median filter | ✅ | ✅ |
| gaussian filter | ✅ | ✅ |
| sobel filter | ✅ | ✅ |
| emboss | ✅ | ✅ |

</details>

<details>
<summary><b>Arithmetic operations</b></summary>

| Type | CPU | GPU |
|------|:---:|:---:|
| add scalar | ✅ | ✅ |
| subtract scalar | ✅ | ✅ |
| multiply scalar | ✅ | ✅ |
| fused multiply add scalar | ✅ | ✅ |
| magnitude | ✅ | ✅ |
| log | ✅ | ✅ |
| log1p | ✅ | ✅ |
| tensor add | ✅ | ✅ |
| tensor subtract | ✅ | ✅ |
| tensor multiply | ✅ | ✅ |
| tensor divide | ✅ | ✅ |

</details>

<details>
<summary><b>Statistical operations</b></summary>

| Type | CPU | GPU |
|------|:---:|:---:|
| tensor sum | ✅ | ✅ |
| tensor min | ✅ | ✅ |
| tensor max | ✅ | ✅ |
| tensor mean | ✅ | ✅ |
| tensor stddev | ✅ | ✅ |
| normalize | ✅ | ✅ |
| threshold | ✅ | ✅ |

</details>

<details>
<summary><b>Bitwise operations</b></summary>

| Type | CPU | GPU |
|------|:---:|:---:|
| bitwise `AND` | ✅ | ✅ |
| bitwise `OR` | ✅ | ✅ |
| bitwise `XOR` | ✅ | ✅ |
| bitwise `NOT` | ✅ | ✅ |
| tensor `AND` tensor | ✅ | ✅ |
| tensor `OR` tensor | ✅ | ✅ |
| tensor `XOR` tensor | ✅ | ✅ |

</details>

<details>
<summary><b>Data exchange operations</b></summary>

| Type | CPU | GPU |
|------|:---:|:---:|
| copy | ✅ | ✅ |
| channel permute | ✅ | ✅ |
| color to greyscale | ✅ | ✅ |
| YUV to RGB | ❌ | ✅ |
| YUV to RGB (cubic vertical upsampling) | ❌ | ✅ |
| YUV to RGB (linear vertical upsampling) | ❌ | ✅ |

</details>

<details>
<summary><b>Audio augmentations</b></summary>

| Type | CPU | GPU |
|------|:---:|:---:|
| non-silent region detection | ✅ | ✅ |
| to decibels | ✅ | ✅ |
| pre-emphasis filter | ✅ | ✅ |
| down mixing | ✅ | ✅ |
| spectrogram | ✅ | ✅ |
| mel filter bank | ✅ | ✅ |
| resample | ✅ | ✅ |
| audio tensor add tensor | ✅ | ✅ |
| audio tensor multiply scalar | ✅ | ✅ |

</details>

<details>
<summary><b>3D image (voxel) augmentations</b></summary>

| Type | CPU | GPU |
|------|:---:|:---:|
| flip (voxel) | ✅ | ✅ |
| gaussian noise (voxel) | ✅ | ✅ |
| add scalar | ✅ | ✅ |
| subtract scalar | ✅ | ✅ |
| multiply scalar | ✅ | ✅ |
| fused multiply add scalar | ✅ | ✅ |
| slice | ✅ | ✅ |
| normalize | ✅ | ✅ |

</details>

## Supported 2D Image Augmentations Samples

<p align="center"><img width="90%" src="docs/data/supported_functionalities_samples.jpg" /></p>

## Supported 3D Image Augmentations Samples

<div align="center">

| &nbsp; | Input<br>(3D voxel image) | &nbsp; |
|:-------------------------:|:-------------------------:|:-------------------------:|
| &nbsp; | ![](docs/data/doxygenInputs/input150x150x4.gif) | &nbsp; |
| add_scalar<br>(3D scalar addition) | subtract_scalar<br>(3D scalar subtraction) | multiply_scalar<br>(3D scalar multiplication) |
| ![](docs/data/doxygenOutputs/arithmetic_operations_add_scalar_150x150x4.gif) | ![](docs/data/doxygenOutputs/arithmetic_operations_subtract_scalar_150x150x4.gif) | ![](docs/data/doxygenOutputs/arithmetic_operations_multiply_scalar_150x150x4.gif) |
| fused_multiply_add_scalar<br>(brightened 3D image) | gaussian_noise<br>(3D noise augmentation) | flip<br>(3D flip augmentation) |
| ![](docs/data/doxygenOutputs/arithmetic_operations_fused_multiply_add_scalar_150x150x4.gif) | ![](docs/data/doxygenOutputs/effects_augmentations_gaussian_noise_150x150x4.gif) | ![](docs/data/doxygenOutputs/geometric_augmentations_flip_150x150x4.gif) |

</div>

slice (3D slice - 100x200 from 240x240x155):

<p align="center"><img src="docs/data/doxygenOutputs/geometric_augmentations_slice_100x200x155.gif" /></p>

## Supported Audio Augmentations Samples

Spectrogram functionality output represented as an image:

<p align="center"><img width="55%" src="docs/data/spectrogramOutput.png" /></p>

## Prerequisites

### Operating Systems
* Linux

>[!NOTE]
> * ROCm supported distributions


### Hardware
* **CPU**: [AMD64](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
* **GPU**: [AMD Radeon&trade; Graphics](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) / [AMD Instinct&trade; Accelerators](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)

> [!IMPORTANT]
> * [ROCm-supported hardware required for HIP backend](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)

### Dependencies
* CMake Version `3.10` or later
* AMD Clang++ compiler (C++17 or higher) - installed with ROCm
* OpenMP - installed with ROCm llvm
* Half - installed with ROCm

## Installation instructions

The installation process uses the following steps:

* [ROCm-supported hardware](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) install verification

* RPP is included with the ROCm Core SDK on Linux. A standard ROCm
installation using the `amdrocm-core-sdk` or `amdrocm-core` meta package
installs the RPP runtime and development package by default. Follow the official
[Install AMD ROCm](https://rocm.docs.amd.com/en/latest/install/rocm.html) guide
and use the selector to choose your GPU, operating system, and install method.
> [!IMPORTANT]
> Use **either** [package install](#package-install) **or** [source build and install](#source-build-and-install) as described below.

### Package install

Install RPP runtime, development, and test packages.
* Runtime package - `amdrocm-rpp` only provides the rpp library `librpp.so`
* Development package - `amdrocm-rpp-dev` (Debian) / `amdrocm-rpp-devel` (RPM) provides the library, header files, and the CMake package config
* Test package - `amdrocm-rpp-test` provides CTest to verify installation

> [!NOTE]
> Package install will auto install all dependencies. `amdrocm-rpp` and the
> development package are already pulled in by the `amdrocm-core` and
> `amdrocm-core-sdk` meta packages, so only `amdrocm-rpp-test` has to be
> installed explicitly on a standard ROCm installation.

> [!IMPORTANT]
> The RPP packages built by TheRock are currently built with audio
> augmentations disabled (`RPP_AUDIO_SUPPORT=OFF`). Build from source to enable
> audio augmentation support.

#### Ubuntu

```shell
sudo apt install amdrocm-rpp amdrocm-rpp-dev amdrocm-rpp-test
```

#### RHEL

```shell
sudo dnf install amdrocm-rpp amdrocm-rpp-devel amdrocm-rpp-test
```

#### SLES

```shell
sudo zypper install amdrocm-rpp amdrocm-rpp-devel amdrocm-rpp-test
```

### Source build and install

RPP is built and released as part of
[TheRock](https://github.com/ROCm/TheRock). Building through TheRock produces
the same artifacts as the released packages and is the recommended path when
you need a build that matches a ROCm release. Follow the
[TheRock build instructions](https://github.com/ROCm/TheRock/blob/main/README.md)
to fetch the sources and configure the build, then build the RPP subproject:

```shell
cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=<your-gpu-family>
ninja -C build rpp+dist
```

RPP can also be built standalone against an existing ROCm installation.

* Clone the ROCm libraries git repository and change into the RPP project directory

  ```shell
  git clone https://github.com/ROCm/rocm-libraries.git
  cd rocm-libraries/projects/rpp
  ```

#### HIP/HOST Backend

  ```shell
  mkdir build && cd build
  cmake ../
  make -j$(nproc)
  sudo make install
  ```
> [!NOTE]
> RPP has both GPU and CPU backends. Building it with HIP backend enables both backends.

### Running Tests
  After installing RPP, refer to the [Verify installation](#verify-installation) section below for instructions on running tests.

## Verify installation

The installer will copy

* Libraries into `${ROCM_PATH}/lib`
* Header files into `${ROCM_PATH}/include/rpp`
* Samples, and test folder into `${ROCM_PATH}/share/rpp`
* Documents folder into `${ROCM_PATH}/share/doc/rpp`

> [!NOTE]
> Packages built by TheRock install into a version-scoped prefix such as
> `/opt/rocm/core-<major>.<minor>` to support side-by-side installations. The
> `/opt/rocm` paths above remain valid because they are maintained as
> `update-alternatives` symlinks into the active prefix.

### Verify with the amdrocm-rpp-test package

Test package will install CTest module to test rpp. Follow below steps to test package install

```shell
mkdir rpp-test && cd rpp-test
cmake ${ROCM_PATH}/share/rpp/test/
ctest -VV
```
> [!NOTE]
> Install test [prerequisites](utilities/test_suite#prerequisites) to run all tests

## Test Functionalities

To test latest Image/Voxel/Audio/Miscellaneous functionalities of RPP using a python script please view [AMD ROCm Performance Primitives (RPP) Test Suite](utilities/test_suite/README.md)

## Adding RPP to your CMake project
To add RPP to your CMake project, you can use the following code after installation:

```cmake
find_package(rpp REQUIRED)
target_link_libraries(your_target PRIVATE rpp::rpp)
```

HIP backend support is automatic: `rpp::rpp` transitively propagates the HIP include paths and link libraries, and `rpp/rpp.h` includes `rpp_backend.h` which sets `RPP_BACKEND_HIP` for your compiled sources.

> [!NOTE]
> `find_package(rpp REQUIRED)` sets the following variables in your CMake project:
> * `rpp_BACKEND_TYPE` - "HIP" or "CPU" — useful for conditional CMake logic (e.g. adding HIP-specific sources)
> * `rpp_AUDIO_AUGMENTATIONS_SUPPORT` - ON or OFF

> [!TIP]
> If CMake is unable to find RPP, the following fixes can be tried:
> * Ensure `${ROCM_PATH}/bin` is in your `PATH`: `export PATH=${ROCM_PATH}/bin:$PATH`.
> * Ensure `CMAKE_PREFIX_PATH` includes `${ROCM_PATH}/lib/cmake`.


## MIVisionX support - OpenVX extension

[MIVisionX](https://github.com/ROCm/MIVisionX) RPP extension
[vx_rpp](https://github.com/ROCm/MIVisionX/tree/master/amd_openvx_extensions/amd_rpp#amd-rpp-extension) supports RPP functionality through the OpenVX Framework.

## Technical support

For RPP questions and feedback, you can contact us at `mivisionx.support@amd.com`.

To submit feature requests and bug reports, use our
[GitHub issues](https://github.com/ROCm/rocm-libraries/issues) page.

## Documentation

You can build our documentation locally using the following code. The Sphinx
build also runs Doxygen to generate the API reference.

```bash
cd docs
pip3 install -r sphinx/requirements.txt
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Release notes

All notable changes for each release are added to our [changelog](CHANGELOG.md).
