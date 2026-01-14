## Setup

**hipThreads prerequisites**:

- [ROCm 7.11+](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) (HIP runtime and hipcc)
- [libhipcxx](https://github.com/ROCm/libhipcxx?tab=readme-ov-file#requirements)
- CMake 3.10+ (+ build tools (e.g. `make` or `ninja`))
- Linux OS (Ubuntu 22.04+ is recommended)

**Build and Installation**

- By default hipThreads installs under `/opt/rocm` (matching other ROCm components). You can override this by adding `-DCMAKE_INSTALL_PREFIX=<path>` to the CMake configure command.
- Installing to `/opt/rocm` usually requires `sudo`.

``` bash
  git clone https://github.com/ROCm/hipThreads.git hipthreads
  cd hipthreads
  mkdir build && cd build
  cmake ..
  make -j
  sudo make install
```


**How to use hipThreads in a CMake project**
- To use hipThreads in your own project, add the following lines to your `CMakeLists.txt` file:

``` cpp
  find_package(hipthreads REQUIRED)
  
  [...]

  target_link_libraries(<your_traget> hipthreads::hipthreads)
```