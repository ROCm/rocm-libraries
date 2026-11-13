// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Demonstrates a 2-level workgroup tiling using mosaic's blocked_product.
//
// We have an HxW grid of workgroups. By default, the scheduler assigns
// workgroup IDs in row-major order across the grid. This example remaps the
// IDs so that consecutive IDs land within the same L2_Y x L2_X tile of
// workgroups before moving to the next tile.
//
// The remapping is expressed as:
//   layout = blocked_product(inner_block, outer_tiler)
// where inner_block is the per-tile workgroup ordering (row-major within a
// tile) and outer_tiler is how tiles are arranged across the full grid
// (column-major across tiles).

#include <hip/hip_runtime.h>

#include <cassert>
#include <iomanip>
#include <iostream>
#include <mosaic/algebra.hpp>  // blocked_product
#include <mosaic/layout.hpp>
#include <stdexcept>
#include <string>
#include <unordered_set>

// Tiny shim — replaces mosaic::error::throw_if_exception from the old API.
inline void check_hip(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
    }
}

bool is_bijective(std::size_t* arr, std::size_t size) {
    std::unordered_set<std::size_t> seen;
    for (std::size_t i = 0; i < size; ++i) {
        if (arr[i] >= size || !seen.insert(arr[i]).second) return false;
    }
    return true;
}

void print_grid(std::size_t* arr, std::size_t height, std::size_t width) {
    std::size_t max_val = 0;
    for (std::size_t i = 0; i < height * width; ++i) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    std::size_t cell_width = std::max<std::size_t>(4, std::to_string(max_val).length() + 2);
    for (std::size_t row = 0; row < height; ++row) {
        for (std::size_t col = 0; col < width; ++col) {
            std::cout << std::setw(cell_width) << arr[row * width + col];
        }
        std::cout << "\n";
    }
}

template <std::size_t L2_Y, std::size_t L2_X>
__global__ void kernel(std::size_t* wg_map, std::size_t* layout_map) {
    // Default workgroup ID — row-major into the H x W workgroup grid.
    std::size_t wg_id = blockIdx.y * gridDim.x + blockIdx.x;

    // The 2-level tiled layout, expressed as blocked_product:
    //   inner = row_major(L2_Y, L2_X) — workgroup ordering within a tile
    //   outer = col_major(H/L2_Y, W/L2_X) — how tiles are placed across the grid
    auto inner = mosaic::make_layout(mosaic::make_shape(L2_Y, L2_X),
                                     mosaic::compact_row_major(mosaic::make_shape(L2_Y, L2_X)));
    auto outer = mosaic::make_layout(
        mosaic::make_shape(gridDim.y / L2_Y, gridDim.x / L2_X),
        mosaic::compact_col_major(mosaic::make_shape(gridDim.y / L2_Y, gridDim.x / L2_X)));
    auto l2_layout = mosaic::blocked_product(inner, outer);

    // Evaluate the layout at the flat workgroup ID. The result is the
    // permuted offset — its position in the tiled enumeration order.
    std::size_t idx = static_cast<std::size_t>(l2_layout(wg_id));

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        wg_map[idx] = wg_id;      // permuted ID → original ID
        layout_map[wg_id] = idx;  // original ID → permuted ID
    }
}

int main() {
    constexpr std::size_t HEIGHT = 8;
    constexpr std::size_t WIDTH = 8;
    constexpr std::size_t TOTAL = HEIGHT * WIDTH;
    constexpr std::size_t L2_Y = 2;
    constexpr std::size_t L2_X = 2;

    std::size_t *d_wg_map = nullptr, *d_layout_map = nullptr;
    auto h_wg_map = new std::size_t[TOTAL];
    auto h_layout_map = new std::size_t[TOTAL];

    try {
        check_hip(hipMalloc(&d_wg_map, TOTAL * sizeof(std::size_t)), "hipMalloc d_wg_map");
        check_hip(hipMalloc(&d_layout_map, TOTAL * sizeof(std::size_t)), "hipMalloc d_layout_map");

        dim3 grid_dim(WIDTH, HEIGHT);
        dim3 block_dim(1, 1);

        kernel<L2_Y, L2_X><<<grid_dim, block_dim>>>(d_wg_map, d_layout_map);
        check_hip(hipGetLastError(), "kernel launch");
        check_hip(hipDeviceSynchronize(), "device sync");

        check_hip(hipMemcpy(h_wg_map, d_wg_map, TOTAL * sizeof(std::size_t), hipMemcpyDeviceToHost),
                  "memcpy wg_map");
        check_hip(hipMemcpy(h_layout_map, d_layout_map, TOTAL * sizeof(std::size_t),
                            hipMemcpyDeviceToHost),
                  "memcpy layout_map");

        assert(is_bijective(h_wg_map, TOTAL) && "Mapping is not bijective!");

        std::cout << "map[transform(wg_id)] = wg_id\n";
        print_grid(h_wg_map, HEIGHT, WIDTH);
        std::cout << "map[wg_id] = transform(wg_id)\n";
        print_grid(h_layout_map, HEIGHT, WIDTH);

        check_hip(hipFree(d_wg_map), "hipFree d_wg_map");
        check_hip(hipFree(d_layout_map), "hipFree d_layout_map");
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        delete[] h_wg_map;
        delete[] h_layout_map;
        return 1;
    }

    delete[] h_wg_map;
    delete[] h_layout_map;
    return 0;
}
