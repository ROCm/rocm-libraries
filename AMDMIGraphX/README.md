# AMD MIGraphX

AMD MIGraphX is AMD's graph inference engine, which accelerates machine learning model inference.

>[!NOTE]
>The published documentation is available at [MIGraphX](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

> [!NOTE]
> You must [install ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html) before
> installing MIGraphX.

## Installing from binaries

Install binaries using:

```bash
sudo apt update && sudo apt install -y migraphx
```

Header files and libraries are installed under ``/opt/rocm-<version>``, where ``<version>`` is the ROCm
version.

## Building from source

You have three options for building from source:

* [ROCm build tool](#use-the-rocm-build-tool-rbuild): Uses
  [rbuild](https://github.com/RadeonOpenCompute/rbuild) to install prerequisites, then you can build
  the libraries with a single command.

* [CMake](#use-cmake-to-build-migraphx): Uses a script to install prerequisites, then you can use
  CMake to build the source.

* [Docker](#use-docker): Builds a Docker image with all prerequisites installed, then you can build the
  MIGraphX sources inside a Docker container.

### Build prerequisites

The following is a list of prerequisites for building MIGraphX.

* [ROCm CMake modules](https://github.com/RadeonOpenCompute/rocm-cmake) **required**
* [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen) for running on the GPU
* [rocBLAS](https://github.com/ROCmSoftwarePlatform/rocBLAS) for running on the GPU
* [HIP](https://github.com/ROCm-Developer-Tools/HIP) for running on the GPU
* [Protobuf](https://github.com/google/protobuf) for reading [onnx](https://github.com/onnx/onnx)
  files
* [Half](http://half.sourceforge.net/), an IEEE 754-based half-precision floating point library
* [pybind11](https://pybind11.readthedocs.io/en/stable/) for python bindings
* [JSON](https://github.com/nlohmann/json) for model serialization to json string format
* [MessagePack](https://msgpack.org/index.html) for model serialization to binary format
* [SQLite3](https://www.sqlite.org/index.html) to create database of kernels' tuning information or run queries on existing database

### Use the ROCm build tool [rbuild](https://github.com/RadeonOpenCompute/rbuild).

1. Install `rocm-cmake`, `pip3`, `rocblas`, and `miopen-hip`:

    ```bash
    sudo apt install -y rocm-cmake python3-pip rocblas miopen-hip
    ```

2. Install [rbuild](https://github.com/RadeonOpenCompute/rbuild) (sudo may be required):

    ```bash
    pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
    ```

3. Build MIGraphX source code:

    ```bash
    rbuild build -d depend -B build -DGPU_TARGETS=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')
    ```

Once completed, all prerequisites are in the `depend` folder and MIGraphX is in the `build` directory.

> [!NOTE]
> If you get an `rbuild: command not found` error, it's because `rbuild` is installed in `$HOME/.local/bin`,
> which is not in `PATH`. You can either export PATH as `export PATH=$HOME/.local/bin:$PATH` to add
> the folder to `PATH`, or add the option `--prefix /usr/local` in the pip3 command when installing `rbuild`.

### Use CMake to build MIGraphX

1. Install the prerequisites:

    ```bash
    rbuild prepare -d depend
    ```

    This puts all the prerequisites are in `depend` the folder. They can be used in the `cmake`
    configuration as `-DCMAKE_PREFIX_PATH=depend`.

    If you have sudo access, as an alternative to the `rbuild` command, you can install the prerequisites
    in the same way as a Dockerfile, by calling `./tools/install_prereqs.sh`.

    By default, all prerequisites are installed at the default location (`/usr/local`) and are accessible by all
    users. For the default location, `sudo` is required to run the script. You can also specify a different
    location using `./tools/install_prereqs.sh $custom_location`.

2. Go to the project folder and create a `build` directory:

    ```bash
    mkdir build
    cd build
    ```

3. Configure CMake. If the prerequisites are installed at the default location `/usr/local`, use:

    ```bash
    CXX=/opt/rocm/llvm/bin/clang++ cmake .. -DGPU_TARGETS=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')
    ```

    Otherwise, you need to set `-DCMAKE_PREFIX_PATH=$your_loc` to configure CMake.

4. Build MIGraphX source code:

    ```cpp
    make -j$(nproc)
    ```

    You can verify this using:

    ```cpp
    make -j$(nproc) check
    ```

5. Install MIGraphX libraries:

    ```cpp
    make install
    ```

### Use Docker

The easiest way to set up the development environment is to use Docker.

1. With the Dockerfile, build a Docker image:

    ```bash
        docker build -t migraphx .
    ```

2. Enter the development environment using `docker run`:

    ```bash
        docker run --device='/dev/kfd' --device='/dev/dri' -v=`pwd`:/code/AMDMIGraphX -w /code/AMDMIGraphX --group-add video -it migraphx
    ```

3. In the Docker container, all required prerequisites are already installed, so you can go to the folder
    `/code/AMDMIGraphX` and follow the steps (starting from 2) in the
    [Use CMake to build MIGraphX](#use-cmake-to-build-migraphx).

## Using the MIGraphX Python module

To use MIGraphX's Python module, you can set `PYTHONPATH` or use the `.deb` package:

* Setting `PYTHONPATH`:

    ```bash
    export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH
    ```

* Creating the `deb` package:

    ```bash
    make package
    ```

    This provides the path for .deb package.

    To install:

    ```bash
    dpkg -i <path_to_deb_file>
    ```

## Calling MIGraphX APIs

To use MIGraphX's C/C++ API in your CMake project, you must set `CMAKE_PREFIX_PATH` to the
MIGraphX installation location and run:

```cmake
find_package(migraphx)
target_link_libraries(myApp migraphx::c)
```

Where `myApp` is the CMake target in your project.

## Building for development

Using `rbuild`, you can install the dependencies for development with:

```bash
rbuild develop -DGPU_TARGETS=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')
```

This installs development dependencies in the `deps` directory and configures `cmake` to use those
dependencies in the `build` directory. You can change these directories by passing the `--deps-dir` and
`--build-dir` flags to the `rbuild` command:

```bash
rbuild develop --build-dir build_rocm_55 --deps-dir /home/user/deps_dir
```

## Building the documentation

HTML and PDF documentation can be built using:

`cmake --build . --config Release --target doc` **OR** `make doc`

This will build a local searchable web site inside the docs/html folder.

Documentation is built using [Doxygen](http://www.stack.nl/~dimitri/doxygen/download.html) and [rocm-docs-core](https://github.com/RadeonOpenCompute/rocm-docs-core)

Run the steps below to build documentation locally.

```bash
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

Depending on your setup `sudo` may be required for the pip install.

## Formatting the code

All the code is formatted using clang-format. To format a file, use:

```clang
clang-format-10 -style=file -i <path-to-source-file>
```

Also, githooks can be installed to format the code per-commit:

```bash
./.githooks/install
```
