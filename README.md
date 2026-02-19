# hipThreads : C++-style concurrency library for AMD GPUs

> [!CAUTION]
> This release is an *early-access* software technology preview. Running production workloads is *not* recommended.
>
> **hipThreads currently works only with ROCm 7.0.2.** Other ROCm versions (including newer ones) are not supported. **Follow the Prerequisites** section below carefully to install the correct version.

## Introduction

hipThreads is a C++-style concurrency library for AMD GPUs that brings familiar threading abstractions to GPU programming by implementing C++ threading and synchronization primitives for GPU code.

The library offers a compatible interface to the C++ Standard Library threading facilities, you can write familiar concurrency code using `hip::thread`, `hip::mutex`, `hip::lock_guard`, `hip::condition_variable`, 
and other primitives. The library supports cooperative threading, standard synchronization primitives, and multi-fiber execution (width parameter) to leverage GPU SIMD architecture.


## Porting Existing Code

If you have existing CPU code using `std::thread`, porting to GPU with hipThreads requires minimal changes:

1. Replace `std::thread` with `hip::thread` 
2. Add `__device__` annotation to lambdas/functions running on GPU
3. Handle GPU memory allocation (CPU and GPU have separate memory pools)

The familiar threading model remains the same, making GPU acceleration accessible without rewriting your concurrency logic. See the `examples/` directory for detailed porting examples.

## Prerequisites

hipThreads requires the following:

- Linux OS (Ubuntu 24.04 recommended)
- CMake 3.21+
- Build tools (e.g., `make` or `ninja`)
- **ROCm 7.0.2** (HIP runtime and hipcc) — **hipThreads does not work with other ROCm versions.**
  To install ROCm 7.0.2 on Ubuntu, follow these steps in order:
  - Meet the [ROCm installation prerequisites](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/prerequisites.html) (kernel version, permissions, etc.). In particular, add your user to the `video` and `render` groups:
    ```bash
    sudo usermod -a -G video,render $LOGNAME
    ```
  - Install the [AMDGPU kernel driver](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/install/detailed-install/package-manager/package-manager-ubuntu.html), then **reboot**.
  - Install **ROCm 7.0.2** via the [package manager](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager/package-manager-ubuntu.html), but **use the 7.0.2 repository URLs** when registering packages:
    ```bash
    sudo mkdir --parents --mode=0755 /etc/apt/keyrings
    wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
        gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

    sudo tee /etc/apt/sources.list.d/rocm.list << EOF
    deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.0.2/ noble main
    deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/7.0.2/ubuntu/ noble main
    EOF

    sudo tee /etc/apt/preferences.d/rocm-pin-600 << EOF
    Package: *
    Pin: release o=repo.radeon.com
    Pin-Priority: 600
    EOF

    sudo apt update
    sudo apt install rocm
    ```
  - Complete the [post-installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/post-install.html), then **reboot**:
    ```bash
    sudo tee --append /etc/ld.so.conf.d/rocm.conf <<EOF
    /opt/rocm/lib
    /opt/rocm/lib64
    EOF
    sudo ldconfig
    sudo reboot
    ```
  - Verify the installation:
    ```bash
    which hipcc             # /opt/rocm/bin/hipcc
    hipcc --version         # Should print the clang/HIP version
    amd-smi                 # Should show GPU info
    ```
    If `amd-smi` does not work, try: `sudo modprobe amdgpu`
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

## Build and Installation

By default, hipThreads installs under `/opt/rocm` (matching other ROCm components). You can override this by adding `-DCMAKE_INSTALL_PREFIX=<path>` to the CMake configure command.

```bash
git clone https://github.com/ROCm/hipThreads.git
cd hipThreads
cmake -B build
cmake --build ./build
sudo cmake --install ./build
```

> [!NOTE]
>  Installing to `/opt/rocm` usually requires `sudo`.

## Usage

To use hipThreads in your CMake project, add the following to your `CMakeLists.txt`:

```cmake
find_package(hipthreads REQUIRED)

# ...

target_link_libraries(<your_target> hipthreads::hipthreads)
```

If hipThreads is not installed in the default `/opt/rocm` location, add `-DCMAKE_PREFIX_PATH=/path/to/hipThreads` to your CMake configure command.

## Examples

Sample code demonstrating hipThreads usage can be found in the `examples/` directory.

### SAXPY — Incremental GPU Porting

The SAXPY example shows how to incrementally port CPU-parallel algorithms to GPU using `hip::thread`, demonstrating the natural progression from `std::thread` to optimized GPU execution.

To build and run:

```bash
cd examples/saxpy/step3-simdize
cmake -B build
cmake --build ./build
./build/bin/saxpy
```


### llama3.c — LLM Inference

The `examples/llama3.c` directory contains a port of [llama3.c](https://github.com/jameswdelancey/llama3.c) (a minimal LLaMA 3 inference engine in C) to `hip::thread`. See the [llama3.c README](https://github.com/jameswdelancey/llama3.c/blob/master/README.md) for full details on model setup and options.

To build:

```bash
cd examples/llama3.c/step4-simdize
cmake -B build
cmake --build ./build
```

Download and export a model (requires the [Meta LLaMA 3 weights](https://huggingface.co/meta-llama/)):

```bash
python export.py llama3.2_3b_instruct_fp32.bin --meta-llama ../llama3.2-3b-instruct/
```

Run inference or start a chat session:

```bash
./build/bin/llama3 ~/models/llama3.2_3b_instruct_fp32.bin -z ~/models/tokenizer.bin -i "My car" -n 100
./build/bin/llama3 ~/models/llama3.2_3b_instruct_fp32.bin -z ~/models/tokenizer.bin -m chat
```

<details>
<summary>Command-line options</summary>

| Option | Description | Default |
|--------|-------------|---------|
| `-t <float>` | Temperature (0 to inf) | `1.0` |
| `-p <float>` | Top-p sampling (0 to 1) | `0.9` |
| `-s <int>` | Random seed | `time(NULL)` |
| `-n <int>` | Number of steps | `4096` |
| `-i <string>` | Input prompt | — |
| `-z <string>` | Path to tokenizer | — |
| `-m <string>` | Mode: `generate` or `chat` | `generate` |
| `-y <string>` | System prompt (chat mode) | — |

</details>

## Documentation

Documentation is available in multiple forms:
- **API Reference**: Doxygen-generated documentation in the `docs/` directory
- **Source Documentation**: Since the library uses Doxygen-style comments throughout the source files.
You can browse the `inc/` and `src/` directories directly to read the API documentation inline with the code.
- **Tutorials and Examples**: See [our ROCm™ Blogs post](https://rocm.blogs.amd.com/software-tools-optimization/hipthreads-introduction/README.html) for an introduction with detailed examples

### Viewing API Documentation Locally

The API reference is generated with Doxygen. Since we're unable to host it online currently, you can view it locally using any of these methods:

**Option 1: Open .html files directly in browser**

**Option 2: Local HTTP server (recommended for full functionality)**:
```bash
python3 -m http.server 5500 --directory docs/doxygen/html
# Then open http://localhost:5500 in your browser
```

> [!NOTE]
> Opening HTML files directly works for most documentation browsing. Use the HTTP server method if you encounter issues with search functionality or cross-file navigation.

## Key Limitations and Best Practices

While hipThreads mimics the C++ standard library, GPU hardware constraints impose specific rules:

### 1. Avoiding Deadlocks: Synchronous Calls and Scoping
The creation of a `hip::thread` launches a persistent kernel (scheduler) that polls for work. Consequently, calling synchronous HIP functions (like `hipDeviceSynchronize`, synchronous `hipMemcpy`, or `thrust::copy`) will cause deadlocks because they wait for *all* GPU tasks to finish—including the persistent idle kernel.

*   **Solution A (Async APIs)**: Use async HIP functions (e.g., `hipMemcpyAsync`, `hipMemsetAsync`) which do not wait for the idle loop to terminate.
*   **Solution B (Scoping)**: If you must use synchronous calls (e.g., when mixing with rocThrust), wrap your `hip::thread` objects in a **scoped block (`{ ... }`)**. This ensures threads are joined and the persistent kernel is destroyed *before* the synchronous call is made.

### 2. Lambda Annotations
A `hip::thread` constructed on the host cannot accept standard host function pointers or standard lambdas.
*   **Requirement**: You must use **extended lambdas** annotated with `__device__`. 
*   **Device Functions**: Host code cannot reference `__device__` functions directly. To call a device function, wrap it inside a `[] __device__ { ... }` lambda.

### 3. Memory and Data Transfer
Arguments passed to the `hip::thread` constructor must be **TriviallyCopyable** as they are copied by value to the device.
*   **No Complex Types**: Do not pass structures containing `std::vector` or other standard containers.
*   **Raw Pointers**: If passing a pointer, it **must** point to GPU-accessible memory (allocated via `hipMalloc` or similar). Passing a host pointer will cause a crash.
*   **Stack Isolation**: GPU threads have private stacks. 
**Never capture by reference (`[&]`)** if the variable resides on the launching thread's stack. Other threads cannot access that memory. 
Shared data must exist in heap/global memory.

### 4. Synchronization Behavior
GPU synchronization primitives are approximations of their CPU counterparts:
*   **No Preemption or Blocking**: The GPU does not support blocking or hardware preemption. `condition_variable::wait` spins or yields rather than blocking.
*   **Pseudo Yield**: `this_thread::yield` only returns control to the caller when the yieldee has finished. The yieldee will not be interrupted and cannot yield back to the caller.


> [!NOTE]
> For practical demonstrations of these concepts, explore the `examples/` directory. 
> For detailed usage of specific primitives and implementation of edge cases, refer to the unit tests in the `test/` directory.


## License

hipThreads is distributed under the Apache License v2.0 with LLVM Exceptions. See `LICENSE.txt` for details.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
