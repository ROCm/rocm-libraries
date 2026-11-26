// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// MX MoE GEMM on gfx1250, a4w4 (MXFP4 x MXFP4) and a8w4 (MX FP8 x MXFP4): MoeFlatmmKernel on
// MXFlatmmPipelineAGmemBGmemCRegV1, one FFN GEMM per launch (gemm1_gate_up; gemm2 with atomic
// combine; gemm2_per_slot followed by MoeTopkCombineKernel), checked against a CPU reference. The
// problem setup (random top-k routing sorted into 32-row expert tiles, MX data and scale layouts,
// reference) is shared with tile_example_mx_moe_flatmm.
//
// The layer cases chain gemm1_gate_up_mx_quant (SiLU(gate) * up quantized to MXFP4 in the
// epilogue) into the down projection, as two kernels or in one MoeMxPersistentKernel launch.
//
// The cases cover the token gather (row-mapping data), padding rows (token counts that are not a
// multiple of the tile), experts spanning several tiles, many small experts, and one token.

#include <gtest/gtest.h>

#include "mx_moe_flatmm.hpp"

namespace {

struct MoeCase
{
    const char* name;
    int experts;
    int num_tokens;
    int topk;
    int N;
    int K;
    int init; // 0 random, 1 constant, 2 row r of A = v(r)
};

std::ostream& operator<<(std::ostream& os, const MoeCase& c) { return os << c.name; }

// gemm1_gate_up: N = 2 x intermediate, K = hidden. gemm2: N = hidden, K = intermediate.
const MoeCase kCases[] = {
    {"Constant", 4, 128, 2, 512, 512, 1},
    {"RowMapping", 4, 128, 2, 512, 512, 2},
    {"Random", 4, 128, 2, 512, 512, 0},
    {"SeveralTilesPerExpert", 2, 256, 2, 512, 512, 0},
    {"PaddingRows", 8, 37, 2, 512, 512, 0},
    {"ManyExperts", 64, 64, 8, 1536, 768, 0},
    {"OneToken", 8, 1, 2, 512, 512, 0},
};

template <ck_tile::MoeFlatmmKind kind, typename ADataType>
void run_case(const MoeCase& t)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "the MX MoE path is implemented for gfx1250 only";

    MxMoeCase c;
    c.experts    = t.experts;
    c.num_tokens = t.num_tokens;
    c.topk       = t.topk;
    c.N          = t.N;
    c.K          = t.K;
    c.init       = t.init;
    c.verbose    = false;

    const MxMoeResult r = run_mx_moe_case<kind, ADataType>(c);
    EXPECT_TRUE(r.pass);
    if(std::string(t.name) == "SeveralTilesPerExpert")
        EXPECT_GT(r.max_rows_per_expert, MXMoeFlatmmConfig::M_Tile)
            << "the case must route more rows to one expert than one M tile holds";
}

// a4w4: MXFP4 activations; a8w4: MX FP8 activations. Weights are MXFP4 in both.
class MxMoeFlatmmA4W4 : public ::testing::TestWithParam<MoeCase>
{
};
class MxMoeFlatmmA8W4 : public ::testing::TestWithParam<MoeCase>
{
};

#define MX_MOE_TESTS(Fixture, AType)                                                            \
    TEST_P(Fixture, Gemm1GateUp)                                                                \
    {                                                                                           \
        run_case<ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up, AType>(GetParam());                \
    }                                                                                           \
    TEST_P(Fixture, Gemm2) { run_case<ck_tile::MoeFlatmmKind::kFFN_gemm2, AType>(GetParam()); } \
    TEST_P(Fixture, Gemm2PerSlot)                                                               \
    {                                                                                           \
        run_case<ck_tile::MoeFlatmmKind::kFFN_gemm2_per_slot, AType>(GetParam());               \
    }                                                                                           \
    INSTANTIATE_TEST_SUITE_P(                                                                   \
        Mx, Fixture, ::testing::ValuesIn(kCases), [](const auto& param_info) {                  \
            return std::string(param_info.param.name);                                          \
        })

MX_MOE_TESTS(MxMoeFlatmmA4W4, ck_tile::pk_fp4_t);
MX_MOE_TESTS(MxMoeFlatmmA8W4, ck_tile::fp8_t);

// A whole layer, a4w4 and a8w4: gemm1_gate_up_mx_quant / _mxfp8_quant writes the intermediate as
// MXFP4 / MX FP8 (checked against a CPU quantization; a value on a rounding midpoint may round
// either way, see run_mx_moe_layer), then the down projection reads it as is.
struct LayerCase
{
    const char* name;
    int experts;
    int num_tokens;
    int topk;
    int hidden;
    int inter;
    int init;
};

std::ostream& operator<<(std::ostream& os, const LayerCase& c) { return os << c.name; }

const LayerCase kLayerCases[] = {
    {"Constant", 4, 128, 2, 512, 512, 1},
    {"Random", 4, 128, 2, 512, 512, 0},
    {"SeveralTilesPerExpert", 2, 256, 2, 512, 512, 0},
    {"PaddingRows", 8, 37, 2, 512, 512, 0},
    {"ManyExperts", 64, 64, 8, 512, 768, 0},
    {"OneToken", 8, 1, 2, 512, 512, 0},
    // a4w4: intermediate rows that are not whole 256-byte blocks, so the persistent kernel pads
    // the row stride, which the down projection's gathered A must honour. (a8w4 rows, one byte per
    // value, are whole blocks for every intermediate size.)
    {"PaddedIntermediateStride", 64, 256, 8, 512, 1280, 0},
};

template <typename AType>
void run_layer(const LayerCase& t, bool per_slot, bool persistent = false)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "the MX MoE path is implemented for gfx1250 only";
    MxMoeLayerCase c;
    c.experts    = t.experts;
    c.num_tokens = t.num_tokens;
    c.topk       = t.topk;
    c.hidden     = t.hidden;
    c.inter      = t.inter;
    c.init       = t.init;
    c.per_slot   = per_slot;
    c.persistent = persistent;
    c.verbose    = false;
    const auto r = run_mx_moe_layer<AType>(c);
    EXPECT_EQ(r.intermediate_mismatch, 0.0) << "the MX intermediate differs from the CPU";
    EXPECT_TRUE(r.pass);
}

class MxMoeLayerA4W4 : public ::testing::TestWithParam<LayerCase>
{
};
class MxMoeLayerA8W4 : public ::testing::TestWithParam<LayerCase>
{
};

// Two kernels per-slot and atomic; the whole layer in one MoeMxPersistentKernel launch (gate_up and
// down tasks synchronized by per-row-tile counters), launched twice so that the second runs on the
// counters the first reset; and whatever plan choose_mx_moe_layer_plan picks for the case.
#define MX_MOE_LAYER_TESTS(Suite, AType)                                                         \
    TEST_P(Suite, PerSlot) { run_layer<AType>(GetParam(), true); }                               \
    TEST_P(Suite, Atomic) { run_layer<AType>(GetParam(), false); }                               \
    TEST_P(Suite, PersistentPerSlot) { run_layer<AType>(GetParam(), true, true); }               \
    TEST_P(Suite, PersistentAtomic) { run_layer<AType>(GetParam(), false, true); }               \
    TEST_P(Suite, AutoPlan)                                                                      \
    {                                                                                            \
        const auto& t = GetParam();                                                              \
        const auto plan =                                                                        \
            choose_mx_moe_layer_plan<AType>(t.experts, t.num_tokens, t.topk, t.hidden, t.inter); \
        run_layer<AType>(t, plan.per_slot, plan.persistent);                                     \
    }                                                                                            \
    INSTANTIATE_TEST_SUITE_P(                                                                    \
        Layer, Suite, ::testing::ValuesIn(kLayerCases), [](const auto& param_info) {             \
            return std::string(param_info.param.name);                                           \
        })

MX_MOE_LAYER_TESTS(MxMoeLayerA4W4, ck_tile::pk_fp4_t);
MX_MOE_LAYER_TESTS(MxMoeLayerA8W4, ck_tile::fp8_t);

} // namespace
