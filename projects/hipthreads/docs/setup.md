## Setup

> **Caution:** **hipThreads currently works only with ROCm 7.0.2** Other ROCm versions (including newer ones) are not supported. Follow the prerequisites below carefully to install the correct versions.

**hipThreads prerequisites**:

- Linux OS (Ubuntu 24.04 recommended)
- CMake 3.21+
- Build tools (e.g., `make` or `ninja`)
- **ROCm 7.0.2** (HIP runtime and hipcc) — **hipThreads currently does not work with other ROCm versions.**
  See the [README](https://github.com/ROCm/hipThreads/blob/release/0.1.0/README.md#Prerequisites) for detailed step-by-step ROCm 7.0.2 installation instructions.

- [libhipcxx v2.7](https://github.com/ROCm/libhipcxx/tree/release/2.7.x?tab=readme-ov-file#installation)
  ```bash
  git clone -b release/2.7.x git@github.com:ROCm/libhipcxx.git # clone and checkout the release/2.7.x
  cd libhipcxx
  mkdir build && cd build
  cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm -DLIBCUDACXX_ENABLE_LIBCUDACXX_TESTS=OFF .. # skip tests to avoid installing lit
  make
  sudo make install
  ```
- The code samples require [rocThrust version **4.2.0**](https://rocm.docs.amd.com/projects/rocThrust/en/latest/install/rocThrust-install-overview.html)
  ```bash
  git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
  cd rocm-libraries
  git sparse-checkout init --cone
  git sparse-checkout set projects/rocthrust
  git checkout release/rocm-rel-7.2
  cd projects/rocthrust
  sudo ./install --install
  ```

**Build and Installation**

- By default hipThreads installs under `/opt/rocm` (matching other ROCm components). You can override this by adding `-DCMAKE_INSTALL_PREFIX=<path>` to the CMake configure command.
- Installing to `/opt/rocm` usually requires `sudo`.

```bash
git clone https://github.com/ROCm/hipThreads.git
cd hipThreads
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

If hipThreads is not installed in the default `/opt/rocm` location, add `-DCMAKE_PREFIX_PATH=/path/to/hipThreads` to your CMake configure command.
