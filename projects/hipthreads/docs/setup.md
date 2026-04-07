## Setup

**hipThreads prerequisites**:

- Linux OS (Ubuntu 24.04 recommended)
- CMake 3.21+
- Build tools (e.g., `make` or `ninja`)
- **ROCm 7.12+** — hipThreads depends on HIP and libhipcxx. The code samples also use rocThrust utilities. All are included in TheRock builds.

### Installing ROCm

> **Note:** ROCm 7.12 is part of a technology preview release stream (starting from 7.9.0) and is separate from the 7.0–7.2 production releases. The last supported ROCm 7 production release is 7.0.2. For ROCm 7.0.2 setup instructions, see the [0.1.0 release prerequisites](https://github.com/ROCm/hipThreads/blob/release/0.1.0/README.md#prerequisites).

1. Follow the [ROCm 7.12 installation guide](https://rocm.docs.amd.com/en/7.12.0-preview/install/rocm.html) for your GPU and distribution. Install at least the **core-dev** package for your GPU architecture (e.g., `amdrocm-core-dev7.12-gfx120x`). The full **core-sdk** package (e.g., `amdrocm-core-sdk-gfx120x`) also works.

2. Configure your environment:

    ```bash
    export ROCM_PATH=/opt/rocm/core
    export PATH=$PATH:$ROCM_PATH/bin
    export LD_LIBRARY_PATH=$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    ```

    To make this persistent across sessions, add the lines above to `~/.bashrc` and run `source ~/.bashrc`.

3. Verify the installation:

    ```bash
    hipcc --version         # Should print the clang/HIP version
    rocminfo                # Should list detected GPUs and HSA agents
    amd-smi version         # Should show AMDSMI and ROCm version info
    ```

### Build and Installation

By default, hipThreads installs under `$ROCM_PATH` (matching other ROCm components). You can override this by adding `-DCMAKE_INSTALL_PREFIX=<path>` to the CMake configure command.

```bash
git clone https://github.com/ROCm/hipThreads.git
cd hipThreads
cmake -B build
cmake --build ./build
sudo cmake --install ./build
```

> **Note:** Installing to `$ROCM_PATH` usually requires `sudo`.

### How to use hipThreads in a CMake project

To use hipThreads in your own project, add the following lines to your `CMakeLists.txt` file:

```cmake
find_package(hipthreads REQUIRED)

# ...

target_link_libraries(<your_target> hipthreads::hipthreads)
```

If hipThreads is not installed under `$ROCM_PATH`, add `-DCMAKE_PREFIX_PATH=/path/to/hipthreads` to your CMake configure command.
