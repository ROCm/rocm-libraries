# hipFFT

hipFFT is an FFT marshalling library that supports
[rocFFT](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocfft) and
[cuFFT](https://developer.nvidia.com/cufft) backends.

hipFFT exports an interface that doesn't require the client to change, regardless of the chosen backend.
It sits between your application and the backend FFT library, where it marshals inputs to the backend
and marshals results back to your application.

## Documentation

> [!NOTE]
> The published hipFFT documentation is available at [hipFFT](https://rocm.docs.amd.com/projects/hipFFT/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the projects/hipfft/docs folder of the rocm-libraries repository. As with all ROCm projects, the documentation is open source. For more information, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

To build our documentation locally, run the following code:

```bash
cd projects/hipfft/docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Build and install

You can download pre-built packages from the
[ROCm package servers](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html).

If you're using Ubuntu, you can run: `sudo apt update && sudo apt install hipfft`.

### Building from source

To build hipFFT from source, follow these steps:

1. Install the library build dependencies:

   * On AMD platforms, you must install [rocFFT](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocfft).

2. Install the client build dependencies:

   * The clients (samples, tests, etc) included with the hipFFT source depend on hipRAND, FFTW and GoogleTest.

3. Build hipFFT. Run these commands from the `rocm-libraries/projects/hipfft` directory:

    To show all build options:

    ```bash
      mkdir build && cd build
      cmake -LH ..
    ```

Here are some CMake build examples for an AMD GPU:

* Case: Build a project using HIP language APIs + hipFFT with standard host compiler
   * Code: `cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -L ..`
* Case: Build a project using HIP language APIs + hipFFT + device kernels with HIP-Clang
   * Code: `cmake -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_BUILD_TYPE=Release -DBUILD_CLIENTS=ON -L ..`

```note
The `-DBUILD_CLIENTS=ON` option is only allowed with the amdclang++ or HIPCC compilers.
```

## Code Coverage
You can generate a test coverage report with the following:

```bash
cmake -DCMAKE_CXX_COMPILER=amdclang++ -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_CODE_COVERAGE=ON <optional: -DCOVERAGE_TEST_OPTIONS="cmdline args to pass to hipfft-test (default: --smoketest)"> ..
make -j coverage
```
The commands above will output the coverage report to the terminal and save an html coverage report to `$PWD/coverage-report`.  Note that hipFFT uses llvm for code coverage, which only works with clang compilers.

## Porting from CUDA

If you have existing CUDA code and want to transition to HIP, follow these steps:

1. [HIPIFY](https://github.com/ROCm-Developer-Tools/HIPIFY) your code and fix all unsupported CUDA
   features and user-defined macros
2. Build with HIP-Clang to run on an AMD device

More information about porting to HIP is available in the
[HIP porting guide](https://rocm.docs.amd.com/projects/HIP/en/develop/user_guide/hip_porting_guide.html).

## Support

You can report bugs and feature requests through the rocm-libraries GitHub
[issue tracker](https://github.com/ROCm/rocm-libraries/issues).

## Contribute

If you want to contribute to hipFFT, you must follow our [contribution guidelines](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipfft/.github/CONTRIBUTING.md).
