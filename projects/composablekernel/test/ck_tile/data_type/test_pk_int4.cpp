// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

using ck_tile::bf16_t;
using ck_tile::bf16x2_t;
using ck_tile::fp16x2_t;
using ck_tile::fp32x2_t;
using ck_tile::half_t;
using ck_tile::pk_int4_t;

TEST(PackedInt4, ConvertToFloat)
{
#ifdef CK_TILE_USE_PK4_LAYOUT_SHUFFLE
    constexpr float first_input_val  = 7.f;
    constexpr float second_input_val = -1.f;
#else
    constexpr float first_input_val  = -1.f;
    constexpr float second_input_val = 7.f;
#endif
    uint8_t data = 0b11110111; // {-1, 7}
    pk_int4_t in = ck_tile::bit_cast<int8_t>(data);
    fp32x2_t out = ck_tile::pk_int4_t_to_fp32x2_t(in);

    EXPECT_EQ(out.x, first_input_val);
    EXPECT_EQ(out.y, second_input_val);
}

TEST(PackedInt4, ConvertToHalf)
{
#ifdef CK_TILE_USE_PK4_LAYOUT_SHUFFLE
    const half_t first_input_val  = ck_tile::type_convert<half_t>(7.f);
    const half_t second_input_val = ck_tile::type_convert<half_t>(-1.f);
#else
    const half_t first_input_val  = ck_tile::type_convert<half_t>(-1.f);
    const half_t second_input_val = ck_tile::type_convert<half_t>(7.f);
#endif
    uint8_t data = 0b11110111; // {-1, 7}
    pk_int4_t in = ck_tile::bit_cast<int8_t>(data);
    fp16x2_t out = ck_tile::pk_int4_t_to_halfx2_t(in);

    EXPECT_EQ(out.x, first_input_val);
    EXPECT_EQ(out.y, second_input_val);
}

TEST(PackedInt4, ConvertToBHalf)
{
#ifdef CK_TILE_USE_PK4_LAYOUT_SHUFFLE
    const bf16_t first_input_val  = ck_tile::type_convert<bf16_t>(7.f);
    const bf16_t second_input_val = ck_tile::type_convert<bf16_t>(-1.f);
#else
    const bf16_t first_input_val  = ck_tile::type_convert<bf16_t>(-1.f);
    const bf16_t second_input_val = ck_tile::type_convert<bf16_t>(7.f);
#endif
    uint8_t data = 0b11110111; // {-1, 7}
    pk_int4_t in = ck_tile::bit_cast<int8_t>(data);
    bf16x2_t out = ck_tile::pk_int4_t_to_bfloat16x2_t(in);

    EXPECT_EQ(out.x, first_input_val);
    EXPECT_EQ(out.y, second_input_val);
}

TEST(PackedInt4, HostTensorOddSpaceSizeRoundsUp)
{
    ck_tile::HostTensor<pk_int4_t> t({3});
    EXPECT_EQ(t.get_element_space_size(), 2u);
    EXPECT_EQ(t.get_element_space_size_in_bytes(), 2u * sizeof(pk_int4_t));
    EXPECT_LT(t.GetOffsetFromMultiIndex(2), t.mData.size());

    ck_tile::HostTensor<pk_int4_t> u({1});
    EXPECT_EQ(u.get_element_space_size(), 1u);

    ck_tile::HostTensor<pk_int4_t> e({4});
    EXPECT_EQ(e.get_element_space_size(), 2u);

    ck_tile::HostTensor<pk_int4_t> m({3, 5});
    EXPECT_EQ(m.get_element_space_size(), 8u);
}
