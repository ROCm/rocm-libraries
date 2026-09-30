# hipThreads : C++-style concurrency library for AMD GPUs

> [!NOTE]
> The published hipThreads documentation is available [here](https://rocm.docs.amd.com/projects/hipThreads/en/latest/) in an organized, easy-to-read format, with search and a table of contents. The documentation source files are located in the `projects/hipThreads/docs/` folder. As with all ROCm projects, the documentation is open source.

## Introduction

hipThreads is a C++-style concurrency library for AMD GPUs that brings familiar threading abstractions to GPU programming by implementing C++ threading and synchronization primitives for GPU code.

The library offers a compatible interface to the C++ Standard Library threading facilities, you can write familiar concurrency code using `hip::wthread`, `hip::mutex`, `hip::lock_guard`, `hip::condition_variable`,
and other primitives. The library supports cooperative threading, standard synchronization primitives, and multi-fiber execution (width parameter) to leverage GPU SIMD architecture.


## Porting Existing Code

If you have existing CPU code using `std::thread`, porting to GPU with hipThreads requires minimal changes:

1. Replace `std::thread` with `hip::wthread`
2. Add `__device__` annotation to lambdas/functions running on GPU
3. Handle GPU memory allocation (CPU and GPU have separate memory pools)

The familiar threading model remains the same, making GPU acceleration accessible without rewriting your concurrency logic. See the `examples/` directory for detailed porting examples.

## Installation

hipThreads ships as part of ROCm. If you install ROCm 10 or later, hipThreads is bundled with it, so
no separate installation is needed. [TheRock](https://github.com/ROCm/TheRock) nightly builds also
include hipThreads in each nightly.

- To install hipThreads on its own, see [Install hipThreads](https://rocm.docs.amd.com/projects/hipThreads/en/latest/install/install.html).
- To build hipThreads from source (on Linux or Windows), see [Build from source](https://rocm.docs.amd.com/projects/hipThreads/en/latest/install/source-build.html). Building from source requires ROCm 7.12 or later.

## Usage

To use hipThreads in your CMake project, add the following to your `CMakeLists.txt`:

```cmake
find_package(hipthreads REQUIRED)

# ...

target_link_libraries(<your_target> hipthreads::hipthreads)
```

If hipThreads is not installed under `$ROCM_PATH`, add `-DCMAKE_PREFIX_PATH=/path/to/hipthreads` to your CMake configure command.

## Examples

Sample code demonstrating hipThreads usage can be found in the `examples/` directory.

### SAXPY — Incremental GPU Porting

The SAXPY example shows how to incrementally port CPU-parallel algorithms to GPU using `hip::wthread`, demonstrating the natural progression from `std::thread` to optimized GPU execution.

To build and run:

```bash
cd examples/saxpy/step3-simdize
cmake -B build
cmake --build ./build
./build/bin/saxpy
```


### llama3.c — LLM Inference

The `examples/llama3.c` directory contains a port of [llama3.c](https://github.com/jameswdelancey/llama3.c) (a minimal LLaMA 3 inference engine in C) to `hip::wthread`. See the [llama3.c README](https://github.com/jameswdelancey/llama3.c/blob/master/README.md) for full details on model setup and options.

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

Full documentation — getting started, the `std::` → `hip::` mapping, limitations, and the complete API reference — is available at [rocm.docs.amd.com/projects/hipThreads](https://rocm.docs.amd.com/projects/hipThreads/en/latest/).

For an introduction with detailed examples, see [our ROCm™ Blogs post](https://rocm.blogs.amd.com/software-tools-optimization/hipthreads-introduction/README.html).

## Key Limitations and Best Practices

GPU hardware imposes constraints that have no counterpart on the CPU (avoiding deadlocks from synchronous HIP calls, `__device__` lambda requirements, TriviallyCopyable arguments, cooperative-only synchronization, and more). These are not compile-time errors, so read them before writing hipThreads code: see [Limitations](https://rocm.docs.amd.com/projects/hipThreads/en/latest/reference/limitations.html).

## Performance Tuning

hipThreads launches a fixed number of scheduler slots ("vcores"), reported by `hip::wthread::hardware_concurrency()`. The number of vcores per WGP defaults to `16`, which works well across the bundled samples on Navi (Radeon) cards, but the best value is architecture- and workload-dependent. You can tune it without rebuilding by setting the `HIPTHREADS_VCORES_PER_WGP` environment variable, or change the compiled-in default with `-DHIPTHREADS_DEFAULT_VCORES_PER_WGP=<n>` when building from source. See [Tune scheduler concurrency](https://rocm.docs.amd.com/projects/hipThreads/en/latest/how-to/tune-scheduler-concurrency.html) for details.

## License

hipThreads is distributed under the Apache License v2.0 with LLVM Exceptions. See `LICENSE.txt` for details.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
