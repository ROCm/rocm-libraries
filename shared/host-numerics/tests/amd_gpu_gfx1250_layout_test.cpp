// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_amd_gpu_layout_test_support.hpp"

namespace amd_gpu_layout_test {
TEST(MxScaleStorageGFX1250Test, ThrowsOnZeroBlock) {
    std::vector<uint8_t> in(4);
    EXPECT_THROW(copyScaleStorage(in, {1, 4}, 0, MxScaleStorageLayout::Gfx1250),
                 std::runtime_error);
}

TEST(MxScaleStorageGFX1250Test, RejectsUnsupportedBlockAndOverflow) {
    std::vector<uint8_t> input;
    EXPECT_THROW(copyScaleStorage(input, {0, 0}, 64, MxScaleStorageLayout::Gfx1250),
                 std::runtime_error);
    const size_t maximumSize = runtimeSize(std::numeric_limits<size_t>::max());
    EXPECT_THROW(copyScaleStorage(input, {maximumSize, 2}, 16, MxScaleStorageLayout::Gfx1250),
                 std::overflow_error);
}

TEST(MxScaleStorageGFX1250Test, ThrowsOnSizeMismatch) {
    std::vector<uint8_t> in(7);  // slow*fast = 8 expected
    EXPECT_THROW(copyScaleStorage(in, {runtimeSize(2), runtimeSize(4)}, runtimeSize(32),
                                  MxScaleStorageLayout::Gfx1250),
                 std::invalid_argument);
}

TEST(MxScaleStorageGFX1250Test, MapsAlignedFastDim) {
    // mxBlock=32 -> dimk=4. slow=2, fast=8 (=2 tiles). Use values v[s*fast+f] = s*10+f.
    constexpr size_t slow = 2, fast = 8, mxBlock = 32;
    constexpr size_t dimk = 128 / mxBlock;  // = 4
    std::vector<uint8_t> in(slow * fast);
    for (size_t s = 0; s < slow; ++s)
        for (size_t f = 0; f < fast; ++f) in[s * fast + f] = static_cast<uint8_t>(s * 10 + f);

    auto out = copyScaleStorage(in, {slow, fast}, mxBlock, MxScaleStorageLayout::Gfx1250);
    ASSERT_EQ(out.size(), slow * fast);  // no padding

    // Output layout: {numTiles, slow, dimk}
    // out[tile, s, j] should equal in[s, tile*dimk + j]
    size_t const numTiles = fast / dimk;
    for (size_t tile = 0; tile < numTiles; ++tile)
        for (size_t s = 0; s < slow; ++s)
            for (size_t j = 0; j < dimk; ++j) {
                size_t const outIdx = tile * (slow * dimk) + s * dimk + j;
                size_t const inIdx = s * fast + (tile * dimk + j);
                EXPECT_EQ(std::to_integer<uint8_t>(out[outIdx]), in[inIdx])
                    << "tile=" << tile << " s=" << s << " j=" << j;
            }
}

TEST(MxScaleStorageGFX1250Test, PadsFastDimWithZeros) {
    // mxBlock=16 -> dimk=8. slow=3, fast=10 -> paddedFast=16, two tiles, second
    // tile has 6 padded zero scales.
    constexpr size_t slow = 3, fast = 10, mxBlock = 16;
    constexpr size_t dimk = 128 / mxBlock;  // = 8
    std::vector<uint8_t> in(slow * fast);
    for (size_t i = 0; i < in.size(); ++i)
        in[i] = static_cast<uint8_t>(i + 1);  // non-zero so we can spot pads

    auto out = copyScaleStorage(in, {slow, fast}, mxBlock, MxScaleStorageLayout::Gfx1250);
    ASSERT_EQ(out.size(), slow * 16);

    size_t const numTiles = 16 / dimk;
    size_t seenZeros = 0;
    for (size_t tile = 0; tile < numTiles; ++tile)
        for (size_t s = 0; s < slow; ++s)
            for (size_t j = 0; j < dimk; ++j) {
                size_t const outIdx = tile * (slow * dimk) + s * dimk + j;
                size_t const srcFast = tile * dimk + j;
                if (srcFast < fast)
                    EXPECT_EQ(std::to_integer<uint8_t>(out[outIdx]), in[s * fast + srcFast]);
                else {
                    EXPECT_EQ(out[outIdx], std::byte{0});
                    ++seenZeros;
                }
            }
    EXPECT_EQ(seenZeros, slow * (16 - fast));
}

TEST(MxScaleStorageGFX1250Test, MultiplesPreservePayload) {
    // For perfectly-aligned fastDim, the swizzle is a pure permutation: every
    // input byte appears exactly once in the output (unsorted equality).
    constexpr size_t slow = 4, fast = 16, mxBlock = 32;
    std::vector<uint8_t> in(slow * fast);
    for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<uint8_t>(i * 3 + 7);

    auto out = copyScaleStorage(in, {slow, fast}, mxBlock, MxScaleStorageLayout::Gfx1250);
    ASSERT_EQ(out.size(), in.size());

    std::vector<std::byte> sortedIn = encodedBytes(in);
    auto sortedOut = out;
    std::sort(sortedIn.begin(), sortedIn.end());
    std::sort(sortedOut.begin(), sortedOut.end());
    EXPECT_EQ(sortedIn, sortedOut);
}

}  // namespace amd_gpu_layout_test
