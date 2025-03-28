# hipBLASLt

hipBLASLt is a library that provides general matrix-matrix operations. It has a flexible API that extends
functionalities beyond a traditional BLAS library, such as adding flexibility to matrix data layouts, input
types, compute types, and algorithmic implementations and heuristics.

> [!NOTE]
> The published hipBLASLt documentation is available at [hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the hipBLASLt/docs folder of this repository. As with all ROCm projects, the documentation is open source. For more information, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

hipBLASLt uses the HIP programming language with an underlying optimized generator as its backend
kernel provider.

After you specify a set of options for a matrix-matrix operation, you can reuse these for different
inputs. The general matrix-multiply (GEMM) operation is performed by the `hipblasLtMatmul` API.

The equation is:

```math
D = Activation(alpha \cdot op(A) \cdot op(B) + beta \cdot op(C) + bias)
```

Where *op( )* refers to in-place operations, such as transpose and non-transpose, and *alpha* and
*beta* are scalars.

The activation function supports GELU, ReLU, and Swish (SiLU). the bias vector matches matrix D rows and
broadcasts to all D columns.

For the supported data types, see
[Supported data types](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/data-type-support.html).

## Documentation

Full documentation for hipBLASLt is available at
[rocm.docs.amd.com/projects/hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html).

Run the steps below to build documentation locally.

```bash
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

Alternatively, build with CMake:

```bash
cmake -DBUILD_DOCS=ON ...
```


## Requirements

To install hipBLASLt, you must meet the following requirements:

Required hardware:

* gfx90a card
* gfx94x card
* gfx110x card

Required software:

* Git
* CMake 3.16.8 or later
* python3.7 or later
* python3.7-venv or later
* AMD [ROCm](https://github.com/RadeonOpenCompute/ROCm), version 5.5 or later
* [hipBLAS-common](https://github.com/ROCm/hipBLAS-common)
* [roctracer](https://github.com/ROCm/roctracer)

## Build and install

You can build hipBLASLt using the `install.sh` script:

```bash
# Clone hipBLASLt using git
git clone https://github.com/ROCmSoftwarePlatform/hipBLASLt

# Go to hipBLASLt directory
cd hipBLASLt

# Run install.sh script
# Command line options:
#   -h|--help         - prints help message
#   -i|--install      - install after build
#   -d|--dependencies - install build dependencies
#   -c|--clients      - build library clients too (combines with -i & -d)
#   -g|--debug        - build with debug flag
./install.sh -idc
```

> **_NOTE:_**  To build hipBLASLt for ROCm <= 6.2, pass the `--legacy_hipblas_direct` flag to `install.sh`

## Unit tests

All unit tests are located in `build/release/clients/staging/`. To build these tests, you must build
hipBLASLt with `--clients`.

You can find more information at the following links:

* [hipblaslt-test](clients/gtest/README.md)
* [hipblaslt-bench](clients/benchmarks/README.md)

## TensileLite Host Library Tests
To build and run TensileLite Host Library Tests, use the following commands:
``` 
 cd tensilelite && mkdir build && cd build
 cmake -DTENSILE_DISABLE_CTEST=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ -DTensile_ROOT=$(pwd)/../Tensile ../HostLibraryTests
 make -j
 ./TensileTests 
```

## Contribute

If you want to submit an issue, you can do so on
[GitHub](https://github.com/ROCmSoftwarePlatform/hipBLASLt/issues).

To contribute to our repository, you can create a GitHub pull request.
