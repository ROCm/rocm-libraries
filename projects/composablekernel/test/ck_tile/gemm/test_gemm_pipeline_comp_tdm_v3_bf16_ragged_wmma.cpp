// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// comp_tdm_v3 bf16 gtest, shapes that are not multiples of the tile (SMOKE_TEST): M, N or K
// ragged, everything ragged with odd strides, a shape below one tile, and M = 1 or N = 1.
//
// TDM clips every A/B box to the real extent, so A/B never limit the shape. C is written by the
// CShuffle epilogue with vector stores: a ragged N tile needs kPadN, and N must be a multiple of
// the C vector size, or a partial vector would write past the row end into the next row. For each
// shape the kernel must accept or reject it by that rule (ExpectSupported), and every shape it
// accepts must be correct:
// - the unpadded instances (V3Bf16Configs) accept M or K ragged and reject a ragged N tile;
// - the padded instances (V3Bf16PaddedConfigs) also accept a ragged N tile, for every N that
//   their C vector size divides (8 by default; 2 and 1 for the narrow-vector instances).
// Inputs and checks: test_gemm_pipeline_comp_tdm_v3_util.hpp.

#include "test_gemm_pipeline_comp_tdm_v3_util.hpp"

namespace {

struct Shape
{
    ck_tile::index_t M, N, K;
};

template <typename Config>
std::vector<Shape> ragged_shapes()
{
    constexpr ck_tile::index_t TK = Config::TileK;
    return {
        {300, 256, 2 * TK},     // M ragged
        {256, 300, 2 * TK},     // N ragged, N % 8 = 4
        {256, 264, 2 * TK},     // N ragged, N % 8 = 0
        {256, 256, 2 * TK + 8}, // K ragged
        {256, 256, TK + 1},     // K ragged and odd: odd A/B row strides
        {333, 517, 1000},       // everything ragged and odd
        {37, 45, 40},           // below one tile in M, N and K
        {37, 48, 40},           // below one tile, N % 8 = 0
        {1, 256, 2 * TK},       // M = 1
        {256, 1, 2 * TK},       // N = 1
    };
}

template <typename Config>
class TestCompTDMV3Bf16Ragged : public TestCkTileGemmPipelineCompTDMV3<Config>
{
    protected:
    void RunRagged(Init init)
    {
        int ran = 0;
        for(const Shape& s : ragged_shapes<Config>())
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
class TestCompTDMV3Bf16RaggedPadded : public TestCompTDMV3Bf16Ragged<Config>
{
};

TYPED_TEST_SUITE(TestCompTDMV3Bf16Ragged, V3Bf16Configs);
TYPED_TEST(TestCompTDMV3Bf16Ragged, IntRagged) { this->RunRagged(Init::Int); }
TYPED_TEST(TestCompTDMV3Bf16Ragged, RandomRagged) { this->RunRagged(Init::Random); }

TYPED_TEST_SUITE(TestCompTDMV3Bf16RaggedPadded, V3Bf16PaddedConfigs);
TYPED_TEST(TestCompTDMV3Bf16RaggedPadded, IntRagged) { this->RunRagged(Init::Int); }
TYPED_TEST(TestCompTDMV3Bf16RaggedPadded, RandomRagged) { this->RunRagged(Init::Random); }

} // namespace
