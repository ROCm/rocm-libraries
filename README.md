# hipTensor

> [!NOTE]
> The published documentation is available at [hipTensor](https://rocm.docs.amd.com/projects/hipTensor/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

Welcome! hiptensor is AMD's C++ library for accelerating tensor primitives using GPU matrix cores on AMD's latest discrete GPUs.

## Requirements

hipTensor currently supports the following AMDGPU architectures:

* CDNA class GPU featuring matrix core support: gfx908, gfx90a, gfx940, gfx941, gfx942, gfx950 as 'gfx9'

:::{note}
Double precision FP64 datatype support requires gfx90a, gfx940, gfx941 or gfx950
:::

Dependencies:

* Minimum ROCm version support is 6.0.
* Minimum cmake version support is 3.14.
* Minimum ROCm-cmake version support is 0.8.0.
* Minimum Composable Kernel version support is composable_kernel 1.1.0 for ROCm 6.0.2 (or ROCm package composablekernel-dev).
* Minimum HIP runtime version support is 4.3.0 (or ROCm package ROCm hip-runtime-amd).
* Minimum LLVM dev package version support is 10.0 (available as ROCm package rocm-llvm-dev).

Optional:

* doxygen (for building documentation)

## Build with CMake

For more detailed information, please refer to the [hipTensor installation guide](https://rocm.docs.amd.com/projects/hipTensor/en/latest/install/installation.html).

### Project options

| Option                              | Description                                         | Default Value                                                    |
|-------------------------------------|-----------------------------------------------------|------------------------------------------------------------------|
| AMDGPU_TARGETS (Deprecated)         | Build code for specific GPU target(s)               | gfx908;gfx90a;gfx940;gfx941;gfx942;gfx950                        |
| GPU_TARGETS                         | Build code for specific GPU target(s)               | gfx908;gfx90a;gfx940;gfx941;gfx942;gfx950                        |
| HIPTENSOR_BUILD_TESTS               | Build Tests                                         | ON                                                               |
| HIPTENSOR_BUILD_SAMPLES             | Build Samples                                       | ON                                                               |
| HIPTENSOR_BUILD_COMPRESSED_DBG      | Enable compressed debug symbols                     | ON                                                               |
| HIPTENSOR_DEFAULT_STRIDES_COL_MAJOR | Set hiptensor default data layout to column major   | ON                                                               |

### Example configurations

By default, the project is configured as Release mode. Here are some of the examples for the configuration:

| Configuration                    | Command                                                                   |
|----------------------------------|---------------------------------------------------------------------------|
| Basic                            | `CC=hipcc CXX=hipcc cmake -B<build_dir> .`                                |
| Targeting gfx908                 | `CC=hipcc CXX=hipcc cmake -B<build_dir> . -DGPU_TARGETS=gfx908:xnack-`    |
| Debug build                      | `CC=hipcc CXX=hipcc cmake -B<build_dir> . -DCMAKE_BUILD_TYPE=Debug`       |
| Build without tests (default on) | `CC=hipcc CXX=hipcc cmake -B<build_dir> . -DHIPTENSOR_BUILD_TESTS=OFF`    |

After configuration, build with `cmake --build <build_dir> -- -j<nproc>`.

## Documentation

For more comprehensive documentation on installation, samples and test contents, API reference and programmer's guide you can build the documentation locally using the following commands:

```bash
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

The HTML documentation can be viewed in your browser by opening docs/_build/html/index.html result.

The latest official documentation for hipTensor is available at:
[https://rocm.docs.amd.com/projects/hipTensor/en/latest/index.html](https://rocm.docs.amd.com/projects/hipTensor/en/latest/index.html).

## Contributing to the hipTensor Library

Community collaboration is encouraged! If you are considering contributing, please follow the [hipTensor Contribution Guide](https://rocm.docs.amd.com/projects/hipTensor/en/latest/contribution/contributors-guide.html) to get started.
