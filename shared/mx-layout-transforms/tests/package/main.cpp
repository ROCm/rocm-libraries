// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <numeric>
#include <roc/mx_layout_transforms/pre_swizzle.hpp>
#include <vector>

int main() {
    using namespace roc::mx_layout_transforms;

    std::vector<int> tiledInput{0, 1, 2, 3, 4, 5, 6, 7};
    auto tiled = preSwizzle(tiledInput, {2, 4}, {}, {2, 4});
    if (tiled != tiledInput) return 1;

    std::vector<uint8_t> gfx950Input(32 * 8);
    std::iota(gfx950Input.begin(), gfx950Input.end(), uint8_t{0});
    auto gfx950 = preSwizzleScalesGFX950(gfx950Input, {32, 8});
    if (gfx950.size() != preSwizzleScalesGFX950PaddedSize(32, 8)) return 2;

    std::vector<uint8_t> gfx1250Input(2 * 5);
    std::iota(gfx1250Input.begin(), gfx1250Input.end(), uint8_t{1});
    auto gfx1250 = preSwizzleScalesGFX1250(gfx1250Input, 2, 5, 32);
    if (gfx1250.size() != preSwizzleScalesGFX1250PaddedSize(2, 5, 32)) return 3;

    return 0;
}
