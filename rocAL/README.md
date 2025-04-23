[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<p align="center"><img width="70%" src="https://raw.githubusercontent.com/ROCm/rocAL/master/docs/data/rocAL_logo.png" /></p>

> [!NOTE]
> The published documentation is available at [rocAL](https://rocm.docs.amd.com/projects/rocAL/en/latest/) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

The AMD ROCm Augmentation Library (**rocAL**) is designed to efficiently decode and process images and videos from a variety of storage formats and modify them through a processing graph programmable by the user. rocAL currently provides C API.
For more details, go to [rocAL user guide](docs) page.

## Supported Operations

rocAL can be currently used to perform the following operations either with randomized or fixed parameters:

<table>
  <tr>
    <th>Blend</th>
    <th>Blur (Gaussian 3x3)</th>
    <th>Brightness</th>
    <th>Color Temperature</th>
  </tr>
  <tr>
    <th>ColorTwist</th>
    <th>Contrast</th>
    <th>Crop</th>
    <th>Crop Mirror Normalization</th>
  </tr>
  <tr>
    <th>CropResize</th>
    <th>Exposure Modification</th>
    <th>Fisheye Lens</th>
    <th>Flip (Horizontal, Vertical and Both)</th>
  </tr>
  <tr>
    <th>Fog</th>
    <th>Gamma</th>
    <th>Hue</th>
    <th>Jitter</th>
  </tr>
  <tr>
    <th>Lens Correction</th>
    <th>Pixelization</th>
    <th>Raindrops</th>
    <th>Random Crop</th>
  </tr>
  <tr>
    <th>Resize</th>
    <th>Resize Crop Mirror</th>
    <th>Rotation</th>
    <th>Salt And Pepper Noise</th>
  </tr>
  <tr>
    <th>Saturation</th>
    <th>Snowflakes</th>
    <th>Vignette</th>
    <th>Warp Affine</th>
  </tr>
</table>

## Prerequisites

### Operating Systems

* Linux distribution
  + Ubuntu - `22.04` / `24.04`
  + RedHat - `8` / `9`
  + SLES - `15-SP5`

### Hardware

* **CPU**: [AMD64](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
* **GPU**: [AMD Radeon&trade; Graphics](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) / [AMD Instinct&trade; Accelerators](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)

> [!IMPORTANT] 
> * [ROCm-supported hardware required for HIP backend](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
> * `gfx908` or higher GPU required

* Install ROCm `6.4.0` or later with [amdgpu-install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html): **Required** usecase:`rocm`
> [!IMPORTANT]
> `sudo amdgpu-install --usecase=rocm`

### Compiler

* AMD Clang++ Version 18.0.0 or later - installed with ROCm

### Libraries

* CMake Version `3.10` or later

  ```shell
  sudo apt install cmake
  ```

* [MIVisionX](https://github.com/ROCm/MIVisionX) Components: [AMD OpenVX&trade;](https://github.com/ROCm/MIVisionX/tree/master/amd_openvx) and AMD OpenVX&trade; Extensions: `VX_RPP` and `AMD Media`
  ```shell
  sudo apt install mivisionx-dev
  ```

* [Half-precision floating-point](https://half.sourceforge.net) library - Version `1.12.0` or higher
  ```shell
  sudo apt install half
  ```

* [Google Protobuf](https://developers.google.com/protocol-buffers) - Version `3.12.4` or higher
  ```shell
  sudo apt install libprotobuf-dev
  ```

* [LMBD Library](http://www.lmdb.tech/doc/)
  ```shell
  sudo apt install liblmdb-dev
  ```

* [Turbo JPEG](https://libjpeg-turbo.org/)
  ```shell
  sudo apt install libturbojpeg0-dev
  ```

* Python3 and Python3 PIP
  ```shell
  sudo apt install python3-dev python3-pip
  ```

* Python3 Wheel
  ```shell
  sudo apt install python3-wheel
  ```

* rocDecode - **Optional** for source install, but required for package install
  ```shell
  sudo apt install rocdecode-dev
  ```

* [rocJPEG](https://github.com/ROCm/rocJPEG) - **Optional** for source install, but required for package install
  ```shell
  sudo apt install rocjpeg-dev
  ```
 
* [FFMPEG](https://www.ffmpeg.org) - **Optional** for source install, but required for package install
  ```shell
  sudo apt install ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
  ```

* [OpenCV](https://docs.opencv.org/4.6.0/index.html) -  **Optional** for source install, but required for package install
  ```shell
  sudo apt install libopencv-dev
  ```

* [PyBind11](https://github.com/pybind/pybind11) - Manual install
  * Source: `https://github.com/pybind/pybind11`
  * Tag: [v2.11.1](https://github.com/pybind/pybind11/releases/tag/v2.11.1)

* [RapidJSON](https://github.com/Tencent/rapidjson) - Manual install
  * Source: `https://github.com/Tencent/rapidjson.git`
  * Tag: `master`

> [!IMPORTANT]
> * Required compiler support
>   * C++17
>   * OpenMP
>   * Threads
> * On Ubuntu 22.04 - Additional package required: libstdc++-12-dev
>  ```shell
>  sudo apt install libstdc++-12-dev
>  `````

>[!NOTE]
> * All package installs are shown with the `apt` package manager. Use the appropriate package manager for your operating system.

### Prerequisites setup script

For your convenience, we provide the setup script,[rocAL-setup.py](https://github.com/ROCm/rocAL/blob/develop/rocAL-setup.py), which installs all required dependencies. Run this script only once.

```shell
python rocAL-setup.py --directory [setup directory - optional (default:~/)]
                      --rocm_path [ROCm Installation Path - optional (default:/opt/rocm)]
                      --backend   [rocAL Dependency Backend - optional (default:HIP) [options:OCL/HIP]]
                      --ffmpeg    [FFMPEG Installation - optional (default:OFF)[options:ON/OFF]]
                      --reinstall [Reinstall - optional (default:OFF)[options:ON/OFF]]
```

## Installation instructions

The installation process uses the following steps:

* [ROCm-supported hardware](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) install verification

* Install ROCm `6.4.0` or later with [amdgpu-install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html) with `--usecase=rocm`

>[!IMPORTANT]
> Use **either** [package install](#package-install) **or** [source install](#source-install) as described below.

### Package install

Install rocAL runtime, development, and test packages.

* Runtime package - `rocal` only provides the dynamic libraries
* Development package - `rocal-dev`/`rocal-devel` provides the libraries, executables, header files, and samples
* Test package - `rocal-test` provides ctest to verify installation

#### `Ubuntu`

  ```shell
  sudo apt-get install rocal rocal-dev rocal-test
  ```

#### `CentOS`/`RedHat`

  ```shell
  sudo yum install rocal rocal-devel rocal-test
  ```

#### `SLES`

  ```shell
  sudo zypper install rocal rocal-devel rocal-test
  ```

>[!IMPORTANT]
> * `SLES` package install requires `TurboJPEG` manual install
>   ```
>   git clone -b 3.0.2 https://github.com/libjpeg-turbo/libjpeg-turbo.git
>   mkdir tj-build && cd tj-build
>   cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_STATIC=FALSE -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib -DWITH_JPEG8=TRUE ../libjpeg-turbo/
>   make -j8 && sudo make install
>   ```
> * `CentOS`/`RedHat`/`SLES` requires additional `FFMPEG Dev` package manual install
> * rocAL Python module: To use python module, you can set PYTHONPATH:
>   + `export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH`

### Source install

To build rocAL from source and install, follow the steps below:

* Clone rocAL source code

```shell
git clone https://github.com/ROCm/rocAL.git
```

>[!NOTE] 
> rocAL has support for two GPU backends: **OPENCL** and **HIP**:

#### HIP Backend

* Instructions for building rocAL with the **HIP** GPU backend (default GPU backend):
  + run the setup script to install all the dependencies required by the **HIP** GPU backend:
  ```shell
  cd rocAL
  python rocAL-setup.py
  ```

  + run the below commands to build rocAL with the **HIP** GPU backend:
  ```shell
  mkdir build-hip
  cd build-hip
  cmake ../
  make -j8
  sudo cmake --build . --target PyPackageInstall
  sudo make install
  ```
>[!NOTE]
> * `PyPackageInstall` used for rocal_pybind installation


>[!IMPORTANT]
> * Use `-D PYTHON_VERSION_SUGGESTED=3.x` with `cmake` for using a specific Python3 version if required.
> * Use `-D AUDIO_SUPPORT=ON` to enable Audio features, Audio support will be enabled by default with ROCm versions > 6.2

  + run tests - [test option instructions](https://github.com/ROCm/MIVisionX/wiki/CTest)
  ```shell
  make test
  ```

>[!NOTE]
> To run tests with verbose option, use `make test ARGS="-VV"`.

#### OpenCL Backend
* Instructions for building rocAL with [**OPENCL** GPU backend](https://github.com/ROCm/rocAL/wiki/OpenCL-Backend)

>[!NOTE]
> + rocAL_pybind is not supported on OPENCL backend
> + rocAL cannot be installed for both GPU backends in the same default folder (i.e., /opt/rocm/)
> + if an app interested in installing rocAL with both GPU backends, then add **-DCMAKE_INSTALL_PREFIX** in the cmake commands to install rocAL with OPENCL and HIP backends into two separate custom folders.

## Verify installation

* The installer will copy
  * Executables into `/opt/rocm/bin`
  * Libraries into `/opt/rocm/lib`
  * rocal_pybind into `/opt/rocm/lib`
  * Header files into `/opt/rocm/include/rocal`
  * Apps, & Samples folder into `/opt/rocm/share/rocal`
  * Documents folder into `/opt/rocm/share/doc/rocal`

### Verify with rocal-test package

Test package will install ctest module to test rocAL. Follow below steps to test package install

```shell
mkdir rocAL-test && cd rocAL-test
cmake /opt/rocm/share/rocal/test/
ctest -VV
```
>[!NOTE]
> * Make sure all rocAL required libraries are in your PATH
> * `RHEL`/`SLES` - Export FFMPEG libraries into your PATH 
>     + `export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64/:/usr/local/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH` 
> ```shell
> export PATH=$PATH:/opt/rocm/bin
> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
> ```

### Verify rocAL PyBind with rocal-test package

Test package will install ctest module to test rocAL PyBindings. Follow below steps to test package install

```shell
mkdir rocal-pybind-test && cd rocal-pybind-test
cmake /opt/rocm/share/rocal/test/pybind
ctest -VV
```
>[!NOTE]
> * Make sure all rocAL required libraries are in your PATH
> ```shell
> export PATH=$PATH:/opt/rocm/bin
> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
> export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH
> ```

## Documentation

Run the steps below to build documentation locally.

* Sphinx documentation
```bash
cd docs
pip3 install -r sphinx/requirements.txt
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```
* Doxygen
```bash
doxygen .Doxyfile
```

## Technical support

Please email `mivisionx.support@amd.com` for questions, and feedback on rocAL.

Please submit your feature requests, and bug reports on the [GitHub issues](https://github.com/ROCm/rocAL/issues) page.

## Release notes

### Latest release version

[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/ROCm/rocAL?style=for-the-badge)](https://github.com/ROCm/rocAL/releases)

### Changelog

Review all notable [changes](CHANGELOG.md#changelog) with the latest release

### Tested Configurations

* Linux distribution
  * Ubuntu - `22.04` / `24.04`
  * RedHat - `8` / `9`
  * SLES - `15-SP5`
* ROCm: rocm-core - `6.4.0`+
* MIVisionX - `mivisionx-dev`/`mivisionx-devel`
* rocDecode - `rocdecode-dev`/`rocdecode-devel`
* Protobuf - `libprotobuf-dev`/`protobuf-devel`
* TurboJPEG - `libturbojpeg0-dev`/`turbojpeg-devel`
* RapidJSON - `https://github.com/Tencent/rapidjson`
* PyBind11 - [v2.11.1](https://github.com/pybind/pybind11)
* FFMPEG - `ffmpeg` dev package
* OpenCV - `libopencv-dev` / [4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
* libsndfile - [1.0.31](https://github.com/libsndfile/libsndfile/releases/tag/1.0.31)
* Libtar - [v1.2.20](https://repo.or.cz/libtar.git)
* rocAL Setup Script - `V3.0.0`
* Dependencies for all the above packages
