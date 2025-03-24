# ROCm OpenSHMEM (rocSHMEM)

The ROCm OpenSHMEM (rocSHMEM) runtime is part of an AMD and AMD Research
initiative to provide GPU-centric networking through an OpenSHMEM-like interface.
This intra-kernel networking library simplifies application
code complexity and enables more fine-grained communication/computation
overlap than traditional host-driven networking.
rocSHMEM uses a single symmetric heap (SHEAP) that is allocated on GPU memories.

There are currently two backends for rocSHMEM;
IPC and Reverse Offload (RO).
The backends primarily differ in their implementations of intra-kernel networking.
Currently, only the IPC backend is supported.
The RO backend is provided as-is with
no guarantees of support from AMD or AMD Research.

The IPC backend implements communication primitives using load/store operations issued from the GPU.

The Reverse Offload (RO) backend has the GPU runtime forward rocSHMEM networking operations
to the host-side runtime, which calls into a traditional MPI or OpenSHMEM
implementation. This forwarding of requests is transparent to the
programmer, who only sees the GPU-side interface.

## Requirements

rocSHMEM base requirements:
* ROCm v6.2.2 onwards
    *  May work with other versions, but it has not been tested
* AMD GPUs
  * MI250X
  * MI300X
* ROCm-aware Open MPI and UCX as described in
  [Building the Dependencies](#building-the-dependencies)

rocSHMEM only supports HIP applications. There are no plans to port to
OpenCL.

## Building and Installation

rocSHMEM uses the CMake build system. The CMakeLists file contains
additional details about library options.

To create an out-of-source build for the IPC backend:

```
mkdir build
cd build
../scripts/build_configs/ipc_single
```

The build script passes configuration options to CMake to setup canonical builds.
There are other scripts in `./scripts/build_configs`
directory but currently, only `ipc_single` is supported.

By default, the library is installed in `~/rocshmem`. You may provide a
custom install path by supplying it as an argument. For example:

```
../scripts/build_configs/ipc_single /path/to/install
```

## Compiling/Linking and Running with rocSHMEM

rocSHMEM is built as a library that can be statically
linked to your application during compilation using `hipcc`.

During the compilation of your application, include the rocSHMEM header files
and the rocSHMEM library when using hipcc.
Since rocSHMEM depends on MPI you will need to link to an MPI library.
The arguments for MPI linkage must be added manually
as opposed to using mpicc.

When using hipcc directly (as opposed to through a build system), we
recommend performing the compilation and linking steps separately.
At the top of the examples files (`./examples/*`),
example compile and link commands are provided:

```
# Compile
hipcc -c -fgpu-rdc -x hip rocshmem_allreduce_test.cc \
  -I/opt/rocm/include                                \
  -I$ROCSHMEM_INSTALL_DIR/include                    \
  -I$OPENMPI_UCX_INSTALL_DIR/include/

# Link
hipcc -fgpu-rdc --hip-link rocshmem_allreduce_test.o -o rocshmem_allreduce_test \
  $ROCSHMEM_INSTALL_DIR/lib/librocshmem.a                                       \
  $OPENMPI_UCX_INSTALL_DIR/lib/libmpi.so                                        \
  -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64

```

If your project uses cmake,
you may find the
[Using CMake with AMD ROCm](https://rocmdocs.amd.com/en/latest/conceptual/cmake-packages.html)
page useful.

## Runtime Parameters
rocSHMEM has the following enviroment variables:

```
    ROCSHMEM_HEAP_SIZE (default : 1 GB)
                        Defines the size of the rocSHMEM symmetric heap
                        Note the heap is on the GPU memory.

    ROCSHMEM_RO_DISABLE_IPC (default : 0)
                        Disables IPC support for the reverse offload backend.
```

## Examples

rocSHMEM is similar to OpenSHMEM and should be familiar to programmers who
have experience with OpenSHMEM or other PGAS network programming APIs in the
context of CPUs.
The best way to learn how to use rocSHMEM is to read the functions described in
headers in the dirctory `./include/rocshmem/`,
or to look at the provided example code in the `./example/` directory.
The examples can be run like so:

```
mpirun -np 2 ./build/examples/rocshmem_getmem_test
```

## Tests
rocSHMEM is shipped with a functional and unit test suite for the supported rocSHMEM API.
They test Puts, Gets, nonblocking Puts,
nonblocking Gets, Quiets, Atomics, Tests, Wait-untils, Broadcasts, Reductions, and etc.
To run the tests, you may use the driver scripts provided in the `./scripts/` directory:

```
# Run Functional Tests
./scripts/functional_tests/driver.sh ./build/tests/functional_tests/rocshmem_example_driver short <log_directory>

# Run Unit Tests
./scripts/unit_tests/driver.sh ./build/tests/unit_tests/rocshmem_unit_tests all
```

## Building the Dependencies

rocSHMEM requires a ROCm-Aware Open MPI and UCX.
Other MPI implementations, such as MPICH,
_should_ be compatible with rocSHMEM but it has not been thoroughly tested.

To build and configure ROCm-Aware UCX (1.17.0 or later), you need to:

```
git clone https://github.com/openucx/ucx.git -b v1.17.x
cd ucx
./autogen.sh
./configure --prefix=<prefix_dir> --with-rocm=<rocm_path> --enable-mt
make -j 8
make -j 8 install
```

Then, you need to build Open MPI (5.0.6 or later) with UCX support.

```
git clone --recursive https://github.com/open-mpi/ompi.git -b v5.0.x
cd ompi
./autogen.pl
./configure --prefix=<prefix_dir> --with-rocm=<rocm_path> --with-ucx=<ucx_path>
make -j 8
make -j 8 install
```

Alternatively, we have script to install dependencies.
However, it is not gauranteed to work and perform optimally on all platforms.
Configuration options are platform dependent.

```
./scripts/install_dependencies.sh
```

For more information on OpenMPI-UCX support, please visit:
https://rocm.docs.amd.com/en/latest/how-to/gpu-enabled-mpi.html
