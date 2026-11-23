# Origami: Analytical Solution Selection for GEMM Kernels

Origami predicts GEMM kernel performance and ranks candidate configurations.
The native library builds independently of Python. Its Python bindings live in
[bindings/python/rocm-origami](../../bindings/python/rocm-origami).

Native headers, implementation, and tests remain in `include/`, `src/`, and
`tests/` under this directory.

## Build and test

Use an installed ROCm development environment through `CMAKE_PREFIX_PATH`:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DORIGAMI_BUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
cmake --install build --prefix /path/to/install
```

`ORIGAMI_BUILD_SHARED_LIBS` defaults to `ON`. Explicit static consumers remain
supported. `ORIGAMI_BUILD_TESTING` defaults to `OFF`, and
`ORIGAMI_ENABLE_FETCH` controls dependency fetching. The native build has no
Python options and installs no Python package.

## API example

### C++ API

```cpp
#include "origami/origami.hpp"
#include "origami/types.hpp"
#include <vector>
#include <iostream>

int main() {
    // Get hardware information for device 0
    auto hardware = origami::hardware_t::get_hardware_for_device(0);
    
    // Create a problem description
    origami::problem_t problem;
    problem.size.m = 2048;  // M dimension
    problem.size.n = 2048;  // N dimension
    problem.size.k = 2048;  // K dimension
    problem.batch = 1;
    problem.a_transpose = origami::transpose_t::T;
    problem.b_transpose = origami::transpose_t::N;
    problem.a_dtype = origami::data_type_t::Half;
    problem.b_dtype = origami::data_type_t::Half;
    problem.c_dtype = origami::data_type_t::Half;
    problem.d_dtype = origami::data_type_t::Half;
    problem.mi_dtype = origami::data_type_t::Half;
    problem.a_mx_block_size = 0;
    problem.b_mx_block_size = 0;
    
    // Create candidate configurations
    std::vector<origami::config_t> configs;
    origami::config_t config;
    config.mt.m = 256;  // Macro tile M
    config.mt.n = 256;  // Macro tile N
    config.mt.k = 64;   // Macro tile K
    config.mi.m = 16;   // Matrix instruction M
    config.mi.n = 16;   // Matrix instruction N
    config.mi.k = 32;   // Matrix instruction K
    config.occupancy = 4;
    configs.push_back(config);
    
    // Select best configuration
    auto best_result = origami::select_config(problem, hardware, configs);
    std::cout << "Best latency: " << best_result.latency << std::endl;
    std::cout << "Best config: MT=(" 
              << best_result.config.mt.m << ", " 
              << best_result.config.mt.n << ", " 
              << best_result.config.mt.k << ")" << std::endl;
    
    // Alternative: Simple selection using just M, N, K
    auto best_result_simple = origami::select_config_mnk(2048, 2048, 2048, hardware, configs);
    
    // Rank all configurations by performance
    auto ranked_configs = origami::rank_configs(problem, hardware, configs);
    std::cout << "Top 5 configs:" << std::endl;
    for (size_t i = 0; i < std::min(ranked_configs.size(), size_t(5)); ++i) {
        const auto& result = ranked_configs[i];
        std::cout << "  Rank " << (i+1) << ": latency=" << result.latency 
                  << ", MT=(" << result.config.mt.m << ", " 
                  << result.config.mt.n << ", " << result.config.mt.k << ")" << std::endl;
    }
    
    // Compute performance in GFLOPS
    double gflops = origami::compute_perf_gflops(hardware, problem, best_result.latency);
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    
    return 0;
}
```

## Supported GPUs

| LLVM Target | GPUs | Functional | Optimized |
|-------------|------|------------|-----------|
| gfx942 | MI325X, MI300X, MI300A | ✔️ | ✔️ |
| gfx950 | MI355X, MI350X | ✔️ | ✔️ |
| gfx1100 | Radeon RX 7900 XTX/XT/GRE, Radeon PRO W7900 (Dual Slot), Radeon PRO W7800 (48GB) | ✔️ | |
| gfx1150 | Radeon 890M/880M iGPU | ✔️ | |
| gfx1151 | Radeon 8060S/8050S/8040S iGPU | ✔️ | |
| gfx1152 | Radeon 860M/840M iGPU | ✔️ | |
| gfx1153 | TBA | ✔️ | |
| gfx1200 | Radeon RX 9060 (XT) | ✔️ | |
| gfx1201 | Radeon RX 9070 (XT/GRE), Radeon AI PRO R9700 (D/S) | ✔️ | |
| gfx1250 | TBA | ✔️ | |

For more information on GPU hardware specifications, check out [ROCm documentation](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html).

## Debug Logging

Origami includes built-in debug logging that exposes internal latency model values (cache hit rates, memory/compute latencies, tile parameters, etc.). Debug output requires the `ANALYTICAL_GEMM_DEBUG=1` environment variable to activate the debug code paths.

### Text Log

Write human-readable debug output to a file by setting `ORIGAMI_LOG_FILE`:

```bash
export ANALYTICAL_GEMM_DEBUG=1
export ORIGAMI_LOG_FILE=/tmp/origami.log
```

Each GEMM evaluation produces a block of key-value lines in the log file, e.g.:

```
[DEBUG] gemm.cpp:99 - ======== Origami Debug Info ========
[DEBUG] gemm.cpp:100 - M: 2048
[DEBUG] gemm.cpp:101 - N: 2048
...
[DEBUG] gemm.cpp:116 - total_latency: 45678.9
[DEBUG] gemm.cpp:117 - =================================
```

### CSV Log

Write structured CSV output (one row per GEMM evaluation) by using a `.csv` file extension:

```bash
export ANALYTICAL_GEMM_DEBUG=1
export ORIGAMI_LOG_FILE=/tmp/origami_debug.csv
```

The log format is inferred from the file extension: `.csv` selects CSV mode, anything else selects human-readable text mode.

The CSV file contains columns for every value logged with `OLOG_DEBUG`, such as `M`, `N`, `K`, `L_mem`, `L_compute`, `H_mem_l2_A`, `total_latency`, etc. This is useful for bulk analysis of the latency model across many GEMM problems.

Accumulated rows are flushed to disk in two situations:

1. **At process exit** — the logger destructor writes any remaining rows.
2. **On explicit flush or reconfiguration** — calling `Logger::flush()` writes buffered rows. Calling `Logger::update_from_env()` also flushes before applying the new configuration. Subsequent rows are appended incrementally.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ANALYTICAL_GEMM_DEBUG` | Set to `1` to enable debug code paths in the latency model |
| `ORIGAMI_LOG_FILE` | Path for log output; `.csv` extension selects CSV format, any other extension selects text |

## Contribute

If you want to submit an issue, you can do so on
[GitHub](https://github.com/ROCm/rocm-libraries/issues). To contribute to our repository, you can create a GitHub pull request.

## How to Cite

If you use Origami or reference it in your research, please cite our work:

```bibtex
@misc{Swann:2025:TTB,
  title={{tritonBLAS}: Triton-based Analytical Approach for GEMM Kernel Parameter Selection}, 
  author={Ryan Swann and Muhammad Osama and Xiaohu Guo and Bryant Nelson and Lixun Zhang and Alex Brown and Yen Ong and Ali Yazdani and Sean Siddens and Ganesh Dasika and Alex Underwood},
  year={2025},
  eprint={2512.04226},
  archivePrefix={arXiv},
  primaryClass={cs.DC},
  url={https://arxiv.org/abs/2512.04226},
}
```
