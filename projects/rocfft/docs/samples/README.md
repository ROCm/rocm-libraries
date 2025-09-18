# Samples to demo using rocfft

## `complex_1d`

You may need to add the directories for amdclang++ and rocFFT to your
`CMAKE_PREFIX_PATH`, and ensure that `amdclang++` is in your `PATH`.

``` bash
$ mkdir build && cd build
$ cmake -DCMAKE_CXX_COMPILER=amdclang++ ..
$ make
```
