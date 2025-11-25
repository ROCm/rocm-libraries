# Tensilelite

## Building and Running Tests

While full test suites can be run with a single `tox` command, developers may wish to
build the hipBLASLt tensilelite client executable (`tensilelite-client`) and run individual tests separately.
This is useful for debugging specific problems or isolating issues in a specific test.

### Run Full Test Suite with Tox

The standard workflow for running the entire test suite is to use `tox`. This command will build
`tensilelite-client` and execute all tests.

```
cd rocm-libraries/projects/hipblaslt/tensilelite
tox -e py3 -- Tensile/Tests -m common
```

### Build client with invoke and Run a Test (Default Path)

This workflow uses `invoke` to build the client into the default `build_tmp` directory.
Tensile will search for `tensilelite-client` in `tensilelite/build_tmp` if `--prebuilt-client`
is not specified.

```
cd rocm-libraries/projects/hipblaslt/tensilelite

# install invoke and rocisa if you haven't already
pip3 install invoke

# build the client to the default location with defaults
invoke build-client

# run an individual test
Tensile/bin/Tensile Tensile/Tests/common/exception/<test>.yaml tensile-out
```

**3. Build with CMake (Custom Location) and Run Test with Path Flag**

This workflow is for when you need to build the client in a location other than the default
`build_tmp` directory. The `--prebuilt-client` flag is then used to specify this custom path when
running a test. Be sure to pass the root directory of the hipblaslt project when configuring.

```
cd rocm-libraries/projects/hipblaslt/tensilelite

# configure in a custom directory (e.g., my-custom-build)
cmake --preset tensilelite -S .. -B my-custom-build

# build
cmake --build my-custom-build --parallel

# run a test, specifying the custom client path with --prebuilt-client
./my-custom-build/Tensile.sh Tensile/Tests/pre_checkin/<test>.yaml tensile-out \
                          --prebuilt-client=my-custom-build/tensilelite-client/tensilelite-client
```

**4. Build with tox (Custom Build Args)**

This workflow uses `tox` with custom CMake arguments, which is useful for creating
specialized builds (e.g., Debug builds) and setting the architecture.

```
# build the client using tox with custom CMake flags
cd rocm-libraries/projects/hipblaslt/tensilelite
TENSILELITE_CLIENT_ARGS="--build-type Debug --gpu-targets gfx90a --clean" tox -e py3 -- Tensile/Tests -m common
```

### Options

* `TENSILELITE_ENABLE_HOST`: Enables generation of tensilelite host (default: `ON`)
* `TENSILELITE_ENABLE_CLIENT`: Enables generation of tensilelite client application (default: `ON`)
* `TENSILELITE_ENABLE_AUTOBUILD`: Generate wrapper scripts that set PYTHONPATH and trigger rebuilds of rocisa (default: `OFF`)
* `TENSILELITE_BUILD_TESTING`: Build tensilelite host library tests (default: `OFF`)
* `GPU_TARGETS:` Semicolon separated list of gfx targets to build

## How to Rebuild Object Codes Directly from Assembly

During the tuning process, it is of interest to modify an assembly file/s and rebuild the corresponding object file/s and then relink the corresponding co file. Currently, we generate additional source files and a script to provide this workflow.

A new `Makefile` is added that manages rebuilding a co file during iterative development when tuning. One modifies an assembly file of interest, then runs `make` and make will detect what file/s changed and rebuild accordingly.

Assumptions:

- Each problem directory contains a library directory with one co file corresponding to one architecture

**Edit**(2025/3/31) ``rocisa`` use the CMake build system instead of the ``virtualenv``. The behavior of the TensileLite changed a bit with only one extra line.

Example:

```cmake -DTENSILE_BIN=Tensile -DDEVELOP_MODE=ON -S <path-to-tensilelite-root> -B <tensile-out>```

The script will be created in the build folder and will be named in Tensile.bat or Tensile.sh depending on the platform. Then you can then run the script under the ``tensile-out`` folder as usual:

```
Tensile.sh <abs-path>/Tensile/Tests/gemm/fp16_use_e.yaml tensile-out
```

or

```
Tensile.bat <abs-path>/Tensile/Tests/gemm/fp16_use_e.yaml tensile-out
```

**You don't need to rerun CMake unless you delete the ``tensile-out`` folder.**

To build asm only:

```
# modify an assembly file in tensile-out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_DB_UserArgs_00/00_Final/source/build_tmp/SOURCE/assembly
make co TENSILE_OUT=tensile-out
# re-run the client
```

The Makefile will set the target based on the name of the co file and sets a default wavefront flag but each of these can be customized as follows:

For 64 wavefront size systems,

```
make co TENSILE_OUT=tensile-out ARCH="gfx942" WAVE=64
```

For 32 wavefront size systems,

```
make co TENSILE_OUT=tensile-out ARCH="gfx1100" WAVE=32
```

In addition, we provide `ASM_ARGS` and `LINK_ARGS` as additional customization points for the assemble and link step respectively. If the architecture cannot be detect corectly, you may need to manually add ``ARCH="gfx942:xnack-"`` to the ``make`` command.
