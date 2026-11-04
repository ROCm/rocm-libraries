# hipCCL

hipCCL is AMD's unified package for [rocPRIM](hipccl3/rocprim), [hipCUB](hipccl3/hipcub),
and [rocThrust](hipccl3/rocthrust) - one version, one install layout, one
package, and one `find_package(hipccl)` entry point for all three, mirroring
the role [NVIDIA's CCCL](https://github.com/NVIDIA/cccl) plays for
CUB/Thrust/libcudacxx.

> [!NOTE]
> hipCCL documentation is available in the [`docs`](docs/index.rst) folder of
> this repository. See [`docs/conceptual/hipccl-layout.rst`](docs/conceptual/hipccl-layout.rst)
> for an explanation of the project's two build layouts (`hipccl3`/`hipccl2`)
> and how to choose between them.

## Requirements

* Git
* CMake (3.25.2 or later)
* AMD [ROCm](https://rocm.docs.amd.com/en/latest/) platform
  * Including
    [HIP-clang](https://github.com/ROCm/HIP/blob/master/INSTALL.md#hip-clang)
    compiler
* C++17

## Build and install

hipCCL can be built two ways: through its layout selector, which lets a
single `cmake` invocation choose between the unified `hipccl3` layout
(default) and the `hipccl2` legacy layout, or by building either layout
directly. See [`docs/install/build.rst`](docs/install/build.rst) for the full
list of CMake options and an explanation of each layout.

```shell
git clone https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries

mkdir build; cd build

# Configure hipCCL, setup options for your system.
# CXX must be set to a HIP-capable compiler.
CXX=hipcc cmake -S ../projects/hipccl -B .

# Build
make -j4

# Install
sudo make install
```

Pass `-DHIPCCL_BUILD_LEGACY=ON` to build the `hipccl2` layout instead of the
default, unified `hipccl3` project.

### Using hipCCL

We recommend including hipCCL into a CMake project by using the package
configuration files. The hipCCL package name is `hipccl`.

```cmake
# "/opt/rocm" - default install prefix
find_package(hipccl REQUIRED)                    # all three components
target_link_libraries(<your_target> PRIVATE hipccl::hipccl)

find_package(hipccl REQUIRED COMPONENTS hipcub)   # just one
target_link_libraries(<your_target> PRIVATE hip::hipcub)
```

rocPRIM, hipCUB, and rocThrust also each keep working as independent,
standalone `find_package()` targets (`roc::rocprim`, `hip::hipcub`,
`roc::rocthrust`), exactly as before hipCCL existed - see each component's own
documentation for details:

* [rocPRIM documentation](https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html)
* [hipCUB documentation](https://rocm.docs.amd.com/projects/hipCUB/en/latest/index.html)
* [rocThrust documentation](https://rocm.docs.amd.com/projects/rocThrust/en/latest/index.html)

## Building the documentation locally

```shell
cd projects/hipccl/docs

python -m venv .venv    # create this outside docs/ to avoid polluting the build
source .venv/bin/activate
pip install -r sphinx/requirements.txt

sphinx-build -b html . _build/html
```

You can then open `_build/html/index.html` in your browser to view the
documentation.

## Support

You can report bugs and feature requests through our GitHub
[issue tracker](https://github.com/ROCm/rocm-libraries/issues).

## Contributions and license

Contributions of any kind are most welcome! Contribution instructions are in
the repository's [CONTRIBUTING](../../CONTRIBUTING.md) guide.

Licensing information is in [LICENSE](hipccl3/LICENSE) (the `hipccl3` copy;
`hipccl2` carries its own, currently identical, copy).
