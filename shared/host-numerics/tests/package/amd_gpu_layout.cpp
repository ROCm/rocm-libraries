// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <roc/host_numerics/amd_gpu_layout/mx.hpp>
#include <vector>

#ifdef _OPENMP
#error "The installed AMDGPULayout target must not export OpenMP compile flags."
#endif

int main() {
    using namespace roc::host_numerics::amd_gpu_layout;

    std::vector<int> tiledInput{0, 1, 2, 3, 4, 5, 6, 7};
    auto tiled = preSwizzle(tiledInput, {2, 4}, {}, {2, 4});
    if (tiled != tiledInput) return 1;

    std::vector<uint8_t> gfx950Input(32 * 8);
    std::iota(gfx950Input.begin(), gfx950Input.end(), uint8_t{0});
    auto gfx950 = copyMxScaleStorageToPhysicalLayout(
        reinterpret_cast<const std::byte*>(gfx950Input.data()), gfx950Input.size(), {32, 8}, 32,
        MxScaleStorageLayout::Gfx950);
    if (gfx950.size() != 32 * 8) return 2;
    const std::array<std::byte, 8> expectedGfx950Prefix{
        std::byte{0}, std::byte{128}, std::byte{4},  std::byte{132},
        std::byte{8}, std::byte{136}, std::byte{12}, std::byte{140},
    };
    if (!std::equal(expectedGfx950Prefix.begin(), expectedGfx950Prefix.end(), gfx950.begin()))
        return 3;

    std::vector<uint8_t> gfx1250Input(2 * 5);
    std::iota(gfx1250Input.begin(), gfx1250Input.end(), uint8_t{1});
    auto gfx1250 = copyMxScaleStorageToPhysicalLayout(
        reinterpret_cast<const std::byte*>(gfx1250Input.data()), gfx1250Input.size(), {2, 5}, 32,
        MxScaleStorageLayout::Gfx1250);
    const std::array<std::byte, 16> expectedGfx1250{
        std::byte{1},  std::byte{2}, std::byte{3}, std::byte{4}, std::byte{6}, std::byte{7},
        std::byte{8},  std::byte{9}, std::byte{5}, std::byte{0}, std::byte{0}, std::byte{0},
        std::byte{10}, std::byte{0}, std::byte{0}, std::byte{0},
    };
    if (gfx1250 != std::vector<std::byte>(expectedGfx1250.begin(), expectedGfx1250.end())) return 4;

    return 0;
}
