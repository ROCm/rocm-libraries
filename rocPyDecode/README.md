[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# rocDecode Python Binding

> [!NOTE]
> The published documentation is available at [rocPyDecode](https://rocm.docs.amd.com/projects/rocPyDecode/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

The rocDecode Python Binding, rocPyDecode, is a tool that allows users to access rocDecode APIs in both Python and C/C++ languages. It works by connecting Python and C/C++ libraries, enabling function calling and data passing between the two languages. The `rocpydecode.so` library is a wrapper that facilitates the use of rocDecode APIs that are written primarily in C/C++ language within Python.

## Prerequisites

### Operating Systems
* Linux
  * Ubuntu - `22.04` / `24.04`

### Hardware
* **GPU**: [AMD Radeon&trade; Graphics](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) / [AMD Instinct&trade; Accelerators](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)

> [!IMPORTANT] 
> `gfx908` or higher GPU required

* Install ROCm `6.3.0` or later with [amdgpu-install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html): **Required** usecase:`rocm`
> [!IMPORTANT]
> `sudo amdgpu-install --usecase=rocm`

### Compiler
* AMD Clang++ Version 18.0.0 or later - installed with ROCm

### Libraries
* CMake `3.12` or higher

  ```shell
  sudo apt install cmake
  ```

* [rocDecode](https://github.com/ROCm/rocDecode)

  ```shell
  sudo apt install rocdecode-dev
  ```

* [DLPack](https://pypi.org/project/dlpack/)
  
  ```shell
  sudo apt install libdlpack-dev
  ```

* Python3 and Python3 PIP

  ```shell
  sudo apt install python3-dev python3-pip
  ```

* [PyBind11](https://github.com/pybind/pybind11)

  ```shell
  sudo apt install python3-pybind11
  ```

* [pkg-config](https://en.wikipedia.org/wiki/Pkg-config)

  ```shell
  sudo apt install pkg-config
  ```

* [FFmpeg](https://ffmpeg.org/about.html) runtime and headers - for tests and samples

  ```shell
  sudo apt install libavcodec-dev libavformat-dev libavutil-dev
  ```

> [!IMPORTANT]
> * Required compiler support
>   * C++17
>   * Threads

>[!NOTE]
> * All package installs are shown with the `apt` package manager. Use the appropriate package manager for your operating system.

### Prerequisites setup script

For your convenience, we provide the setup script, [rocPyDecode-requirements.py](rocPyDecode-requirements.py), which installs all required dependencies. Run this script only once on bare metal, if using docker please see below instructions.

```shell
python3 rocPyDecode-requirements.py
```

## rocPyDecode install

The installation process uses the following steps:

* [ROCm-supported hardware](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) install verification

* Install ROCm `6.3.0` or later with [amdgpu-install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html) with `--usecase=rocm`

>[!IMPORTANT]
> Use **either** [package install](#package-install) **or** [source install](#source-install) as described below.

### Package install

Install rocPyDecode runtime, and test packages.

* Runtime package - `rocpydecode` only provides the python bindings for rocDecode
* Test package - `rocpydecode-test` provides ctest to verify installation

#### `Ubuntu`

  ```shell
  sudo apt-get install rocpydecode rocpydecode-test
  ```

>[!IMPORTANT]
> Python module: To use python module, set PYTHONPATH:
>   + `export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH`

### Source install

To build rocPyDecode from source and install, follow the steps below:

* Clone rocPyDecode source code

```shell
git clone https://github.com/ROCm/rocPyDecode.git
```

#### CMake install

* Instructions for building rocPyDecode with the **CMake**
  + run the requirements script to install all the dependencies required:
  ```shell
  cd rocPyDecode
  python3 rocPyDecode-requirements.py
  ```

  + run the below commands to build rocPyDecode:
  ```shell
  mkdir build && cd build
  cmake ../
  make -j8
  sudo make install
  ```

>[!IMPORTANT]
> * rocPyDecode will be installed for all Python versions on the system. To install rocPyDecode for a specific Python version, use the cmake `-D PYTHON_VERSION_SUGGESTED=version_num` directive, where version_num is the target Python version.

  + run tests - [test option instructions](https://github.com/ROCm/MIVisionX/wiki/CTest)
  ```shell
  make test
  ```

>[!NOTE]
> To run tests with verbose option, use `make test ARGS="-VV"`.

#### Pip3 install

```shell
cd rocPyDecode
sudo pip3 install .
```
>[!NOTE]
> `sudo` access is required

#### Creating python distribution wheel
* Option 1:
```shell
cd rocPyDecode
sudo python3 build_rocpydecode_wheel.py
```
* Option 2:
```shell
cd rocPyDecode
sudo python3 setup.py bdist_wheel
```
>[!NOTE]
> * Generated `.whl` file will be located under subfolder `./dist/`
> * `sudo` access is required

#### docker environment install

```shell
cd rocPyDecode
python rocPyDecode-docker-install.py 
```
>[!NOTE]
> Do NOT use `sudo`

### creating rocPyDecode conda package
* Information on how to create and install rocPyDecode conda package is located [here](https://github.com/ROCm/rocPyDecode/blob/develop/conda-recipe/README.md).

## Run CTest

This will run python samples and show pass/fail.
>[!NOTE]
> install rocPydecode before running tests

### Dependencies:
```shell
python3 -m pip install --upgrade pip
python3 -m pip install -i https://test.pypi.org/simple hip-python
```

### Run tests with source
```shell
mkdir rocpydecode-test && cd rocpydecode-test
cmake ../rocPyDecode/tests
ctest -VV
```

### Run tests with rocpydecode-test package

Test package will install ctest module to test rocPyDecode. Follow below steps to test package install

```shell
mkdir rocpydecode-test && cd rocpydecode-test
cmake /opt/rocm/share/rocpydecode/tests
ctest -VV
```
>[!NOTE]
> Make sure all required libraries are in your PATH
> ```shell
> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
> export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH
> ```

## Run Sample Scripts

* Sample scripts and instructions to run them can be found [here](samples/)

## Documentation

Run the following code to build our documentation locally.

```shell
cd docs
pip3 install -r sphinx/requirements.txt
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

For more information on documentation builds, refer to the
[Building documentation](https://rocm.docs.amd.com/en/latest/contribute/building.html)
page.

## Tested configurations

* Linux distribution
  * Ubuntu - `22.04` / `24.04`
* ROCm: rocm-core - `6.3.0.60300`+
* AMD Clang++ - Version `18.0.0`+
* CMake - Version `3.12`+
* rocdecode-dev - `0.10.0`+
* libdlpack-dev - `0.6-1`
* python3-pybind11 - `2.9.1-2`
