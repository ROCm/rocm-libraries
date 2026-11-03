# Fortran API reference

This library provides a pure Fortran interface for the hipRAND API.

This interface is intended to target only Host API functions, and provides a mapping to some of
the C Host API functions in hipRAND. For documentation of these functions, please
refer to the C Host API functions documentation.

## Deprecation notice

This library is currently deprecated in favor of [hipfort](https://github/ROCm/hipfort).

## Build and install

The Fortran interface is installed as part of the hipRAND package. Simply add the build
option `-DBUILD_FORTRAN_WRAPPER=ON` when configuring the project, as below:

```shell
cmake -DBUILD_FORTRAN_WRAPPER=ON ...
```

## Running unit tests

After having configured the project with testing enabled (with option `-DBUILD_TEST=ON`), follow the steps below
to build and run the unit tests.

1. Go to hipRAND build directory
```shell
cd /path/to/hipRAND/build
```

2. Build unit tests for Fortran interface
```shell
cmake --build . --target test_hiprand_fortran_wrapper
```

3. Run unit tests
```shell
./test/test_hiprand_fortran_wrapper
```

## Usage

Below is an example of writing a simple Fortran program that generates a set of uniform values.

```fortran
integer(kind =8) :: gen
real, target, dimension(128) :: h_x
type(c_ptr) :: d_x
integer(c_int) :: status
integer(c_size_t), parameter :: output_size = 128
status = hipMalloc(d_x, output_size * sizeof(h_x(1)))
status = hiprandCreateGenerator(gen, HIPRAND_RNG_PSEUDO_DEFAULT)
status = hiprandGenerateUniform(gen, d_x, output_size)
status = hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), hipMemcpyDeviceToHost)
status = hipFree(d_x)
status = hiprandDestroyGenerator(gen)
```

And when compiling the source code with a Fortran compiler, the following should be linked[^1]:

```shell
# Compile on an NVCC platform (link CUDA libraries: cuda, cudart).
gfortran <input-file>.f90 hip_m.f90 hiprand_m.f90 -lhiprand_fortran -lhiprand -lcuda -lcudart

# Compile on an AMD platform (link HIP library: ${HIP_ROOT_DIR}/lib).
# Note: ${HIP_ROOT_DIR} points to the directory where HIP was installed.
gfortran <input-file>.f90 hip_m.f90 hiprand_m.f90 -lhiprand_fortran -lhiprand -L${HIP_ROOT_DIR}/lib
```

[^1]: `gfortran` is used in this example, however other Fortran compilers should work.
