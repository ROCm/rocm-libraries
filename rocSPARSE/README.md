# rocSPARSE

rocSPARSE exposes a common interface that provides Basic Linear Algebra Subroutines (BLAS) for
sparse computation. It's implemented on top of AMD
[ROCm](https://github.com/ROCm/ROCm) runtime and toolchains. rocSPARSE is
created using the [HIP](https://github.com/ROCm/HIP/) programming
language and optimized for AMD's latest discrete GPUs.

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
* CMake (3.5 or later)
* AMD [ROCm] 3.5 platform or later

Optional:
* [GoogleTest](https://github.com/google/googletest) (required only for tests)
  * Use `GTEST_ROOT` to specify a location
  * If you don't have GoogleTest installed, CMake automatically downloads and builds it

## Build and install

1. Build rocSPARSE using the `install.sh` script.

    ```bash
    # Clone rocSPARSE using git
    git clone https://github.com/ROCm/rocSPARSE.git

    # Go to rocSPARSE directory
    cd rocSPARSE

    # Run install.sh script
    # Command line options:
    #   -h|--help         - prints help message
    #   -i|--install      - install after build
    #   -d|--dependencies - install build dependencies
    #   -c|--clients      - build library clients too (combines with -i & -d)
    #   -g|--debug        - build with debug flag
    ./install.sh -dci
    ```

2. Compile rocSPARSE (all compiler specifications are automatically determined).

    ```bash
    # Clone rocSPARSE using git
    git clone https://github.com/ROCm/rocSPARSE.git

    # Go to rocSPARSE directory, create and go to the build directory
    cd rocSPARSE; mkdir -p build/release; cd build/release

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
cd rocSPARSE; cd build/release

# Run all tests
./clients/staging/rocsparse-test
```

To run benchmarks, you must build rocSPARSE with `-DBUILD_CLIENTS_BENCHMARKS=ON`.

```bash
# Go to rocSPARSE build directory
cd rocSPARSE/build/release

# Run benchmark, e.g.
./clients/staging/rocsparse-bench -f hybmv --laplacian-dim 2000 -i 200
```

## Issues

To submit an issue, a bug, or a feature request, use the GitHub
[issue tracker](https://github.com/ROCm/rocSPARSE/issues).

## License

Our [license file](https://github.com/ROCm/rocSPARSE) is located in the main
repository.
