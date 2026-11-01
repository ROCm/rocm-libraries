// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// comp_tdm_v3 bf16 gtest, M, N and K that are not multiples of the tile (SMOKE_TEST):
// M, N or K unaligned, all three unaligned with odd strides, a shape below one tile, and M = 1 or N
// = 1.
//
// TDM clips every A/B box to the real extent (it handles ragged extents itself), so A/B never limit
// the shape. C is written by the CShuffle epilogue with vector stores: an unaligned N needs kPadN,
// and N must be a multiple of the C vector size, or a partial vector would write past the row end
// into the next row. For each shape the kernel must accept or reject it by that rule
// (ExpectSupported), and every shape it accepts must be correct:
// - the unpadded instances (V3Bf16Configs) accept an unaligned M or K and reject an unaligned N;
// - the padded instances (V3Bf16PaddedConfigs) also accept an unaligned N, for every N that
//   their C vector size divides (8 by default; 2 and 1 for the narrow-vector instances);
// - the TdmEpilogue instances (V3Bf16TdmEpilogueConfigs), unpadded, accept every shape: the TDM
//   store clips C in hardware.
// Inputs and checks: test_gemm_pipeline_comp_tdm_v3_util.hpp.

#include "test_gemm_pipeline_comp_tdm_v3_util.hpp"

namespace {

struct Shape
{
    ck_tile::index_t M, N, K;
};

template <typename Config>
std::vector<Shape> unaligned_shapes()
{
    constexpr ck_tile::index_t TK = Config::TileK;
    return {
        {300, 256, 2 * TK},     // M unaligned
        {256, 300, 2 * TK},     // N unaligned, N % 8 = 4
        {256, 264, 2 * TK},     // N unaligned, N % 8 = 0
        {256, 256, 2 * TK + 8}, // K unaligned
        {256, 256, TK + 1},     // K unaligned and odd: odd A/B row strides
        {333, 517, 1000},       // M, N and K unaligned and odd
        {37, 45, 40},           // below one tile in M, N and K
        {37, 48, 40},           // below one tile, N % 8 = 0
        {1, 256, 2 * TK},       // M = 1
        {256, 1, 2 * TK},       // N = 1
    };
}

template <typename Config>
class TestCompTDMV3Bf16UnalignedMnk : public TestCkTileGemmPipelineCompTDMV3<Config>
{
    protected:
    void RunUnaligned(Init init)
    {
        int ran = 0;
        for(const Shape& s : unaligned_shapes<Config>())
        {
            if(this->ExpectSupported(s.M, s.N, s.K))
            {
                this->Run(s.M, s.N, s.K, init);
                ++ran;
            }
        }
        EXPECT_GT(ran, 0);
    }
};
template <typename Config>
class TestCompTDMV3Bf16UnalignedMnkTdmEpilogue : public TestCompTDMV3Bf16UnalignedMnk<Config>
{
};
template <typename Config>
class TestCompTDMV3Bf16UnalignedMnkPadded : public TestCompTDMV3Bf16UnalignedMnk<Config>
{
};

TYPED_TEST_SUITE(TestCompTDMV3Bf16UnalignedMnk, V3Bf16Configs);
TYPED_TEST(TestCompTDMV3Bf16UnalignedMnk, IntUnaligned) { this->RunUnaligned(Init::Int); }
TYPED_TEST(TestCompTDMV3Bf16UnalignedMnk, RandomUnaligned) { this->RunUnaligned(Init::Random); }

TYPED_TEST_SUITE(TestCompTDMV3Bf16UnalignedMnkPadded, V3Bf16PaddedConfigs);
TYPED_TEST(TestCompTDMV3Bf16UnalignedMnkPadded, IntUnaligned) { this->RunUnaligned(Init::Int); }
TYPED_TEST(TestCompTDMV3Bf16UnalignedMnkPadded, RandomUnaligned)
{
    this->RunUnaligned(Init::Random);
}

TYPED_TEST_SUITE(TestCompTDMV3Bf16UnalignedMnkTdmEpilogue, V3Bf16TdmEpilogueConfigs);
TYPED_TEST(TestCompTDMV3Bf16UnalignedMnkTdmEpilogue, IntUnaligned)
{
    this->RunUnaligned(Init::Int);
}
TYPED_TEST(TestCompTDMV3Bf16UnalignedMnkTdmEpilogue, RandomUnaligned)
{
    this->RunUnaligned(Init::Random);
}

} // namespace
