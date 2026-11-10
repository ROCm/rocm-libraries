// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Instances comp_tdm_v4 must reject at compile time; CMake compiles each MX_TDM_V4_REJECT case
// with -fsyntax-only and expects it to fail.

#include "test_mx_gemm_tdm_v4_util.hpp"

namespace {

using namespace mx_tdm_v4_test;
using ck_tile::fp8_t;
using ck_tile::MxTdmV4HalfPlr;
using ck_tile::pk_fp4_t;

#if MX_TDM_V4_REJECT == 1
// half_plr 'b' without local prefetch.
using Bad = Kernel<pk_fp4_t,
                   pk_fp4_t,
                   fp8_t,
                   256,
                   256,
                   512,
                   2,
                   2,
                   32,
                   Tu<4, 4, 4, -1, 0>,
                   Shape<false, false, 0, 0, 8>>;
#elif MX_TDM_V4_REJECT == 2
// 8-bit types on the 32x16 FP4 atom.
using Bad =
    Kernel<fp8_t, fp8_t, fp8_t, 128, 128, 256, 2, 2, 32, Tu<2, 2>, Shape<false, false, 0, 0, 4>>;
#elif MX_TDM_V4_REJECT == 3
// A 16-element D store with an N repeat of 8.
using Bad = Kernel<pk_fp4_t,
                   pk_fp4_t,
                   fp8_t,
                   256,
                   256,
                   512,
                   2,
                   2,
                   32,
                   Tu<4, 4>,
                   Shape<false, false, 0, 0, 16>>;
#elif MX_TDM_V4_REJECT == 4
// GL2 prefetch distance 1.
using Bad = Kernel<pk_fp4_t,
                   pk_fp4_t,
                   fp8_t,
                   256,
                   256,
                   512,
                   2,
                   2,
                   32,
                   Tu<4, 4, 1>,
                   Shape<false, false, 0, 0, 8>>;
#elif MX_TDM_V4_REJECT == 5
// A cluster of 32 workgroups.
using Bad = Kernel<pk_fp4_t,
                   pk_fp4_t,
                   fp8_t,
                   256,
                   256,
                   512,
                   2,
                   2,
                   32,
                   Tu<8, 4>,
                   Shape<false, false, 0, 0, 8>>;
#else
#error "set MX_TDM_V4_REJECT"
#endif

[[maybe_unused]] const std::string name = Bad::GetName();
[[maybe_unused]] constexpr auto smem    = Bad::GetSmemSize();

} // namespace
