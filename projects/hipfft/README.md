# hipFFT

hipFFT is an FFT marshalling library that supports
[rocFFT](https://github.com/ROCmSoftwarePlatform/rocFFT) and
[cuFFT](https://developer.nvidia.com/cufft) backends.

hipFFT exports an interface that doesn't require the client to change, regardless of the chosen backend.
It sits between your application and the backend FFT library, where it marshals inputs to the backend
and marshals results back to your application.

## Documentation

> [!NOTE]
> The published hipFFT documentation is available at [hipFFT](https://rocm.docs.amd.com/projects/hipFFT/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the hipFFT/docs folder of this repository. As with all ROCm projects, the documentation is open source. For more information, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

To build our documentation locally, run the following code:

```bash
cd docs

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

   * On AMD platforms, you must install [rocFFT](https://github.com/ROCmSoftwarePlatform/rocFFT).
   * On NVIDIA platforms, you must install [cuFFT](https://developer.nvidia.com/cufft).

2. Install the client build dependencies:

   * The clients (samples, tests, etc) included with the hipFFT source depend on hipRAND, FFTW and GoogleTest.

3. Build hipFFT:

    To show all build options:

    ```bash
      mkdir build && cd build
      cmake -LH ..
    ```

Here are some CMake build examples:

* AMD GPU
  * Case: Build a project using HIP language APIs + hipFFT with standard host compiler
    * Code: `cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -L ..`
  * Case: Build a project using HIP language APIs + hipFFT + device kernels with HIP-Clang
    * Code: `cmake -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_BUILD_TYPE=Release -DBUILD_CLIENTS=ON -L ..`
* NVIDIA GPU
  * Case: Build a project using HIP language APIs + hipFFT with standard host compiler
    * Code: `cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_LIB=CUDA -L ..`
  * Case: Build a project using HIP language APIs + hipFFT + device kernels with HIP-NVCC
    * Code: `HIP_PLATFORM=nvidia cmake -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_BUILD_TYPE=Release -DBUILD_CLIENTS=ON -L ..`

```note
The `-DBUILD_CLIENTS=ON` option is only allowed with the amdclang++ or HIPCC compilers.
```

## Porting from CUDA

If you have existing CUDA code and want to transition to HIP, follow these steps:

1. [HIPIFY](https://github.com/ROCm-Developer-Tools/HIPIFY) your code and fix all unsupported CUDA
   features and user-defined macros
2. Build with HIP-NVCC to run on an NVIDIA device
3. Build with HIP-Clang to run on an AMD device

More information about porting to HIP is available in the
[HIP porting guide](https://rocm.docs.amd.com/projects/HIP/en/develop/user_guide/hip_porting_guide.html).

## Support

You can report bugs and feature requests through the GitHub
[issue tracker](https://github.com/ROCm/hipFFT/issues).

## Contribute

If you want to contribute to hipFFT, you must follow our [contribution guidelines](https://github.com/ROCm/hipFFT/blob/develop/.github/CONTRIBUTING.md).
