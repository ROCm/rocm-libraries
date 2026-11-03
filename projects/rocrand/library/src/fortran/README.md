# Fortran API reference

This library provides a pure Fortran interface for the rocRAND API.

This interface is intended to target only Host API functions. It provides a mapping to some of
the C Host API functions in rocRAND. For documentation of these functions, see
the C Host API functions documentation.

## Deprecation notice

This library is currently deprecated in favor of [hipfort](https://github.com/ROCm/hipfort).

## Build and install

The Fortran interface is installed as part of the rocRAND package. Add the build
option `-DBUILD_FORTRAN_WRAPPER=ON` when configuring the project, as shown below:

```shell
cmake -DBUILD_FORTRAN_WRAPPER=ON ...
```

## Running unit tests

Configure the project with testing enabled, using the `-DBUILD_TEST=ON` option, then follow the steps below
to build and run the unit tests.

1. Go to the rocRAND `build` directory:
```shell
cd /path/to/rocRAND/build
```

2. Build the unit tests for the Fortran interface:
```shell
cmake --build . --target test_rocrand_fortran_wrapper
```

3. Run the unit tests:
```shell
./test/test_rocrand_fortran_wrapper
```

## Usage

Here is an example of how to write a simple Fortran program that generates a set of uniform values.

```fortran
integer(kind =8) :: gen
real, target, dimension(128) :: h_x
type(c_ptr) :: d_x
integer(c_int) :: status
integer(c_size_t), parameter :: output_size = 128
status = hipMalloc(d_x, output_size * sizeof(h_x(1)))
status = rocrand_create_generator(gen, ROCRAND_RNG_PSEUDO_DEFAULT)
status = rocrand_generate_uniform(gen, d_x, output_size)
status = hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), hipMemcpyDeviceToHost)
status = hipFree(d_x)
status = rocrand_destroy_generator(gen)
```

When compiling the source code with a Fortran compiler, the following items should be linked[^1]:

```shell

# Compile on an AMD platform (link HIP library: ${HIP_ROOT_DIR}/lib).
# Note: ${HIP_ROOT_DIR} points to the directory where HIP was installed.
gfortran <input-file>.f90 hip_m.f90 rocrand_m.f90  -lrocrand_fortran -lrocrand -L${HIP_ROOT_DIR}/lib
```

[^1]: `gfortran` is used in this example, however, other Fortran compilers should work.
