# rocWMMA

Welcome! rocWMMA is a C++ library for accelerating mixed-precision matrix multiply-accumulate (MMA)
operations leveraging AMD GPU hardware. rocWMMA makes it easier to break down MMA problems
into fragments and distribute block-wise MMA operations in parallel across GPU wavefronts. The API
consists of a header library, that can be used to compile MMA acceleration directly into GPU kernel
device code. This can benefit from compiler optimization in the generation of kernel assembly, and
doesn't incur additional overhead costs of linking to external runtime libraries or having to launch
separate kernels.

rocWMMA includes sample projects to validate and demonstrate API usage. These include simple GEMMs,
performant GEMMs, DLRM, GEMV and hipRTC integration. Community-contributed samples demonstrating
advanced techniques and specialized use cases are available in samples/community/ (opt-in via
ROCWMMA_BUILD_COMMUNITY_SAMPLES=ON).

The test suite includes validation and benchmarking projects that focus on unit testing, GEMMs and DLRM.

> [!NOTE]
> The published rocWMMA documentation is available at [rocWMMA](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `projects/rocwmma/docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).


## Requirements

rocWMMA currently supports the following AMD GPU architectures:

* CDNA class GPU featuring matrix core support: gfx908, gfx90a, gfx942, gfx950 as 'gfx9'
* RDNA class GPU featuring AI acceleration support: gfx1100, gfx1101, gfx1102, gfx1150, gfx1151, gfx1152, gfx1153 as 'gfx11'; gfx1200, gfx1201 as 'gfx12'

Dependencies:

* Minimum ROCm version support is 6.4.
* Minimum cmake version support is 3.14.
* Minimum ROCm-cmake version support is 0.8.0.
* Minimum rocBLAS version support is rocBLAS 4.0.0 for ROCm 6.0* (or ROCm packages rocblas and rocblas-dev).
* Minimum ROCm SMI version support is 7.6.0** (or ROCm packages rocm-smi-lib and librocm-smi-dev).
* Minimum HIP runtime version support is 4.3.0 (or ROCm package hip-runtime-amd).
* Minimum LLVM OpenMP runtime dev package version support is 10.0 (available as ROCm package rocm-llvm-dev).

```note::
    * = if using rocBLAS for validation.
    ** = if building benchmark tests (configuring with ROCWMMA_BUILD_BENCHMARK_TESTS=ON).

    It is best to use available ROCm packages from the same release where applicable.
```

## Build with CMake

For more detailed information, please refer to the [rocWMMA installation guide](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/install/installation.html).

### Project options

|Option|Description|Default value|
|---|---|---|
|GPU_TARGETS|Build code for specific GPU target(s)|gfx908;gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201|
|AMDGPU_TARGETS|(Deprecated) Build code for specific GPU target(s)|gfx908;gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201|
|ROCWMMA_BUILD_TESTS|Build Tests|ON|
|ROCWMMA_BUILD_SAMPLES|Build Samples|ON|
|ROCWMMA_BUILD_COMMUNITY_SAMPLES|Build community-contributed samples|OFF|
|ROCWMMA_BUILD_DOCS|Build doxygen documentation from code|OFF|
|ROCWMMA_BUILD_ASSEMBLY|Generate assembly files|OFF|
|ROCWMMA_BUILD_VALIDATION_TESTS|Build validation tests |ON (requires ROCWMMA_BUILD_TESTS=ON)|
|ROCWMMA_BUILD_REGRESSION_TESTS|Build regression testing coverage |ON (requires ROCWMMA_BUILD_TESTS=ON)|
|ROCWMMA_BUILD_BENCHMARK_TESTS|Build benchmark tests |OFF (requires ROCWMMA_BUILD_TESTS=ON)|
|ROCWMMA_BUILD_EXTENDED_TESTS|Build extended testing coverage |OFF (requires ROCWMMA_BUILD_TESTS=ON)|
|ROCWMMA_VALIDATE_WITH_ROCBLAS|Use rocBLAS for validation tests|ON (requires ROCWMMA_BUILD_VALIDATION_TESTS=ON)|
|ROCWMMA_BENCHMARK_WITH_ROCBLAS|Include rocBLAS benchmarking data|OFF (requires ROCWMMA_BUILD_BENCHMARK_TESTS=ON)|
|ROCWMMA_USE_SYSTEM_GOOGLETEST|Use system Google Test library instead of downloading and building it|OFF (requires ROCWMMA_BUILD_TESTS=ON)|

### Example configurations

By default, the project is configured in release mode and is linked against rocBLAS for validating
results. Here are some configuration examples:

|Configuration|Command|
|---|---|
|Basic|`CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B<build_dir> .`|
|Targeting gfx908|`CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B<build_dir> . -DGPU_TARGETS=gfx908:xnack-` |
|Debug build|`CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B<build_dir> . -DCMAKE_BUILD_TYPE=Debug` |
|Build without rocBLAS (default on)|`CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -B<build_dir> . -DROCWMMA_VALIDATE_WITH_ROCBLAS=OFF -DROCWMMA_BENCHMARK_WITH_ROCBLAS=OFF` |

After configuration, build with `cmake --build <build_dir> -- -j<nproc>`

## Documentation

For more comprehensive documentation on installation, samples and test contents, API reference and programmer's guide you can build the documentation locally in different ways.

### Html

```bash
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

The HTML documentation can be viewed in your browser by opening the `docs/_build/html/index.html` result.

### Pdf

```bash
cd docs

sudo apt-get update
sudo apt-get install doxygen
sudo apt-get install texlive-latex-base texlive-latex-extra

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b latex -d _build/doctrees -D language=en . _build/latex

cd _build/latex

pdflatex rocwmma.tex
```

Running the above commands generates `rocwmma.pdf`.

The latest official documentation for rocWMMA is available at:
[https://rocm.docs.amd.com/projects/rocWMMA/en/latest/index.html](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/index.html).


## Contributing to the rocWMMA Library

Community collaboration is encouraged! If you are considering contributing, please follow the [rocWMMA Contribution Guide](https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocwmma/CONTRIBUTING.md) to get started.
