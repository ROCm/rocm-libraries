# rocSPARSE

rocSPARSE exposes a common interface that provides Basic Linear Algebra Subroutines (BLAS) for
sparse computation. It is implemented on top of AMD
[ROCm](https://github.com/ROCm/ROCm) runtime and toolchains. rocSPARSE is
created using the [HIP](https://github.com/ROCm/rocm-systems/tree/develop/projects/hip) programming
language and optimized for AMD's latest discrete GPUs.

> [!NOTE]
> For portability, ROCm provides the **[hipSPARSE](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipsparse)** library. hipSPARSE includes a comprehensive, portable interface that supports multiple backends (including rocSPARSE and cuSPARSE). For documentation and examples, see the [hipSPARSE documentation](https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/).

## Documentation

> [!NOTE]
> The published rocSPARSE documentation is available at [rocSPARSE](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the rocSPARSE/docs folder of this repository. As with all ROCm projects, the documentation is open source. For more information, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

To build our documentation locally, run the following code:

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

* Git
* CMake (3.19 or later)
* [ROCm CMake](https://github.com/ROCm/rocm-cmake) (0.11.0 or later)
* AMD [ROCm] 3.5 platform or later

Optional:
* [GoogleTest](https://github.com/google/googletest) (required only for tests)
  * Use `GTEST_ROOT` to specify a location
  * Alternatively, set `FETCHCONTENT_SOURCE_DIR_GOOGLETEST` to a googletest source tree
* The SuiteSparse test matrices (required only for tests), stored with [DVC](https://dvc.org/):
  run `dvc pull -R projects/rocsparse/clients/matrices` from the root of the rocm-libraries repository.
  DVC needs S3 support, for example `pip install 'dvc[s3]'`

By default, CMake doesn't download any of these. Configure with `-DALLOW_FETCH_DEPS=ON` to download
ROCm CMake, GoogleTest and the test matrices when they're missing.

## Build and install

1. Checkout the rocSPARSE code using either a sparse checkout or a full clone of the rocm-libraries repository.

   To limit your local checkout to only the rocSPARSE project, configure ``sparse-checkout`` before cloning.
   This uses the Git partial clone feature (``--filter=blob:none``) to reduce how much data is downloaded.
   Use the following commands for a sparse checkout:

    ```bash

    git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
    cd rocm-libraries
    git sparse-checkout init --cone
    git sparse-checkout set projects/rocsparse .dvc # add projects/rocprim and projects/rocblas to include dependencies
    git checkout develop # or use the branch you want to work with
    ```

   To clone the entire rocm-libraries repository, use the following commands. This process takes more time,
   but is recommended if you want to work with a large number of libraries.

    ```bash

    # Clone rocm-libraries, including rocSPARSE, using Git
    git clone https://github.com/ROCm/rocm-libraries.git

    # Go to rocSPARSE directory
    cd rocm-libraries/projects/rocsparse
    ```

2. Build rocSPARSE using the `install.sh` script.

    ```bash

    # Run install.sh script
    # Command line options:
    #   -h|--help         - prints help message
    #   -i|--install      - install after build
    #   -d|--dependencies - install distro build dependencies (not GoogleTest)
    #   -c|--clients      - build library clients too (combines with -i & -d)
    #   -g|--debug        - build with debug flag
    ./install.sh -dci
    ```

    > [!NOTE]
    > Building the tests (`install.sh -c`, or `-DBUILD_CLIENTS_TESTS=ON`) needs GoogleTest and the
    > SuiteSparse test matrices. Install GoogleTest, for example the `libgtest-dev` package, and run
    > `dvc pull -R clients/matrices` from this directory first (in a sparse checkout, run
    > `git sparse-checkout add .dvc` before that). When you configure with CMake directly,
    > `-DALLOW_FETCH_DEPS=ON` downloads both instead.

3. Compile rocSPARSE (all compiler specifications are automatically determined).

    ```bash

    # Clone rocm-libraries, including rocSPARSE, using Git
    git clone https://github.com/ROCm/rocm-libraries.git

    # Go to rocSPARSE directory
    cd rocm-libraries/projects/rocsparse

    # Pull the test matrices
    dvc pull -R clients/matrices

    # Create and go to the build directory
    mkdir -p build/release; cd build/release

    # Configure rocSPARSE
    # Build options:
    #   BUILD_CLIENTS_TESTS      - build tests (OFF)
    #   BUILD_CLIENTS_BENCHMARKS - build benchmarks (OFF)
    #   BUILD_CLIENTS_SAMPLES    - build examples (ON)
    #   BUILD_VERBOSE            - verbose output (OFF)
    #   BUILD_SHARED_LIBS        - build rocSPARSE as a shared library (ON)
    CXX=/opt/rocm/bin/amdclang++ cmake -DBUILD_CLIENTS_TESTS=ON ../..

    # Build
    make

    # Install
    [sudo] make install
    ```

## Unit tests and benchmarks

To run unit tests, you must build rocSPARSE with `-DBUILD_CLIENTS_TESTS=ON`.

```bash
# Go to rocSPARSE build directory
cd rocm-libraries/projects/rocsparse; cd build/release

# Run all tests
./clients/staging/rocsparse-test
```

To run benchmarks, you must build rocSPARSE with `-DBUILD_CLIENTS_BENCHMARKS=ON`.

```bash
# Go to rocSPARSE build directory
cd rocm-libraries/projects/rocsparse/build/release

# Run benchmark, e.g.
./clients/staging/rocsparse-bench -f hybmv --laplacian-dim 2000 -i 200
```

## Issues

To submit an issue, a bug, or a feature request, use the rocm-libraries GitHub
[issue tracker](https://github.com/ROCm/rocm-libraries/issues).

## License

The [license file](https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocsparse/LICENSE.md) is located in the main
repository.
