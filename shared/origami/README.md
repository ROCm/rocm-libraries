# Origami: Analytical GEMM Solution Selection

The name "origami" still evokes the elegance of transforming a flat (2-D) sheet into intricate higher dimensional structures. In this context, however, Origami has evolved into a toolset for **GEMM solution selection and optimization**. Inspired by the art of paper folding, the library now enables users to explore a range of tiling and mapping configurations and to make informed decisions on data and computation mapping for high-performance GEMM operations.

Origami provides a rigorous methodology to analyze and select optimal GEMM parameters. It does so by evaluating both **compute** and **memory latencies** across a wide range of tile sizes. The framework computes essential metrics such as:
- **Matrix Instruction (MI) Tiling Counts** and the associated compute latencies.
- **Memory Load Latencies**, considering both L2 and main memory bandwidth constraints.
- **Active Compute Unit Occupancy and Wave Counts**, ensuring realistic mapping to the underlying hardware.
- **Tie-breaking Strategies** using arithmetic intensity and more refined L2 hit rate estimations to resolve candidate configuration ties.

This approach allows programmers to achieve near-optimal performance without manually exploring every possibility. While default (implicit) parameters provide "out-of-the-box" performance, expert users can dive deeper into the analytical model to tune for their specific hardware and problem sizes.

---

## Documentation

- **Getting Started**
- [Programming Abstraction](docs/programming_abstraction.md)
- [Hierarchical Tiling and GEMM Latency Calculation](docs/hierarchical_tiling.md)
- [Usage Examples](docs/examples.md)

---

## Quick Start Guide – Origami

**Origami** provides an end-to-end analytical solution to GEMM parameter selection. It estimates performance by sweeping over candidate tile sizes and selecting the optimal configuration based on latency and arithmetic intensity.

### Building Origami

Assuming you are in the repository root, run:

```bash
# configure
cmake -S . -B build/ -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ -DCMAKE_INSTALL_PREFIX=/opt/rocm 
# build
cmake --build build/ --parallel
```

### Installing Origami

After configuring and building, run the following command to install:

```bash
# install
cmake --target install
```

## API & Usage

The core API is centered around the function `select_best_macro_tile_size` provided in `Utils.hpp`. This function evaluates a list of candidate macro-tile configurations for GEMM operations and selects the one that minimizes total latency while applying a tie-breaking strategy based on problem dimensions. The API provides several additional functions for ranking, sweeping tile configurations, and even selecting the best workgroup multiplier based on L2 hit rate.

### `select_best_macro_tile_size`

**Function Signature:**

```cpp
std::tuple<double, size_t, size_t, size_t, size_t, size_t, size_t> select_best_macro_tile_size(
    size_t M,
    size_t N,
    size_t K,
    Hardware &hardware,
    const std::vector<std::tuple<size_t, size_t, size_t, size_t, size_t, size_t>> &MT_list,
    size_t element_size,
    double H_L2,
    bool debug,
    bool print,
    size_t WGM);
```


**Quick Example:**
```cpp
#include "Utils.hpp"
#include <iostream>
#include <vector>
#include <tuple>

using namespace TensileLite::analytical;

int main() {
    // Define GEMM dimensions
    size_t M = 1024, N = 1024, K = 1024;
    size_t element_size = 2;
    double H_L2 = 0.8;
    bool debug = true;
    bool print = true;
    size_t WGM = 6;
    
    // Instantiate your hardware object with appropriate parameters.
    Hardware hardware;
    
    // Create a list of candidate macro-tile configurations.
    // Each tuple is structured as: (MT_M, MT_N, MT_K, MI_M, MI_N, MI_K).
    std::vector<std::tuple<size_t, size_t, size_t, size_t, size_t, size_t>> MT_list = {
        std::make_tuple(64, 64, 128, 16, 16, 16),
        std::make_tuple(128, 64, 64, 16, 16, 16),
        // You can add more candidate configurations as needed.
    };
    
    // Call select_best_macro_tile_size to obtain the best configuration.
    auto best_tile = select_best_macro_tile_size(M, N, K, hardware, MT_list, element_size, H_L2, debug, print, WGM);
    
    // Unpack and display the resulting configuration.
    double latency = std::get<0>(best_tile);
    size_t MT_M = std::get<1>(best_tile);
    size_t MT_N = std::get<2>(best_tile);
    size_t MT_K = std::get<3>(best_tile);
    size_t MI_M = std::get<4>(best_tile);
    size_t MI_N = std::get<5>(best_tile);
    size_t MI_K = std::get<6>(best_tile);
    
    std::cout << "Selected Macro-Tile Configuration:\n"
              << "  Total Latency: " << latency << "\n"
              << "  MT_M: " << MT_M << ", MT_N: " << MT_N << ", MT_K: " << MT_K << "\n"
              << "  MI_M: " << MI_M << ", MI_N: " << MI_N << ", MI_K: " << MI_K << "\n";
    
    return 0;
}

```