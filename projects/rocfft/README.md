# rocFFT

rocFFT is a software library for computing fast Fourier transforms (FFTs) written in the HIP
programming language. It's part of the AMD software ecosystem based on
[ROCm](https://github.com/ROCm/ROCm). The rocFFT library can be used with AMD GPUs.

## Documentation

> [!NOTE]
> The published rocFFT documentation is available at [rocFFT](https://rocm.docs.amd.com/projects/rocFFT/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the projects/rocfft/docs folder of the rocm-libraries repository. As with all ROCm projects, the documentation is open source. For more information, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

To build our documentation locally, use the following code:

```Bash
cd projects/rocfft/docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Build and install

You can install rocFFT using pre-built packages or building from source.

* Installing pre-built packages:

    1. Download the pre-built packages from the
        [ROCm package servers](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html) or use the
        GitHub releases tab to download the source (this may give you a more recent version than the
        pre-built packages).

    2. Run: `sudo apt update && sudo apt install rocfft`

* Building from source:

    rocFFT is compiled with AMD's clang++ and uses CMake. You can specify several options to customize your
    build. The following commands build a shared library for supported AMD GPUs. Run these commands from the `rocm-libraries/projects/rocfft` directory:

    ```bash
    mkdir build && cd build
    cmake -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_C_COMPILER=amdclang ..
    make -j
    ```

    You can compile a static library using the `-DBUILD_SHARED_LIBS=off` option.

    With rocFFT, you can use indirect function calls by default; this requires ROCm 4.3 or higher. You can
    use `-DROCFFT_CALLBACKS_ENABLED=off` with CMake to prevent these calls on older ROCm
    compilers. Note that with this configuration, callbacks won't work correctly.

    rocFFT includes the following clients:

  * `rocfft-bench`: Runs general transforms and is useful for performance analysis
  * `rocfft-test`: Runs various regression tests
  * Various small samples

    | Client | CMake option | Dependencies |
    |:------|:-----------------|:-----------------|
    | `rocfft-bench` | `-DBUILD_CLIENTS_BENCH=on` | hipRAND |
    | `rocfft-test` | `-DBUILD_CLIENTS_TESTS=on` | hipRAND, FFTW, GoogleTest |
    | samples | `-DBUILD_CLIENTS_SAMPLES=on` | None |
    | coverage | `-DBUILD_CODE_COVERAGE=ON` | clang, llvm-cov |

    Clients are not built by default. To build them, use `-DBUILD_CLIENTS=on`. The build process
    downloads and builds GoogleTest and FFTW if they are not already installed.

    Clients can be built separately from the main library. For example, you can build all the clients with
    an existing rocFFT library by invoking CMake from within the `rocFFT-src/clients` folder:

    ```bash
    mkdir build && cd build
    cmake -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_PREFIX_PATH=/path/to/rocFFT-lib ..
    make -j
    ```

    To install client dependencies on Ubuntu, run:

    ```bash
    sudo apt install libgtest-dev libfftw3-dev libboost-dev
    ```

    rocFFT uses version 1.11 of GoogleTest.

    You can generate a test coverage report with the following:
    ```bash
    cmake -DCMAKE_CXX_COMPILER=amdclang++ -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_CODE_COVERAGE=ON <optional: -DCOVERAGE_TEST_OPTIONS="cmdline args to pass to rocfft-test (default: --smoketest)"> ..
    make -j coverage
    ```
    The above will output the coverage report to the terminal and also save an html coverage report to `$PWD/coverage-report`.

## Examples

A summary of the latest functionality and workflow to compute an FFT with rocFFT is available on the
[rocFFT documentation portal](https://rocm.docs.amd.com/projects/rocFFT/en/latest/).

You can find additional examples in the `clients/samples` subdirectory.

## Support

You can report bugs and feature requests through the rocm-libraries GitHub
[issue tracker](https://github.com/ROCm/rocm-libraries/issues).

## Contribute

If you want to contribute to rocFFT, you must follow the [contribution guidelines](https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocfft/.github/CONTRIBUTING.md).
