// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "base.hpp"
#include "code.hpp"
#include "container.hpp"
#include "instruction/common.hpp"
#include "instruction/cvt.hpp"
#include "instruction/mem.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

static constexpr IsaVersion gfx950 = {9, 5, 0};
static constexpr int32_t wavefrontSize = 64;

// rocIsa keeps the selected kernel per thread, so each test selects it.
static void selectGfx950() {
  rocisa::rocIsa &isa = rocisa::rocIsa::getInstance();
  isa.init(gfx950, ROCISA_CORE_TEST_ASSEMBLER);
  isa.setKernel(gfx950, wavefrontSize);
}

static void expectContains(std::string_view text, std::string_view needle) {
  EXPECT_NE(text.find(needle), std::string_view::npos)
      << "missing '" << needle << "' in:\n"
      << text;
}

TEST(RocisaCore, EmitsGfx950Assembly) {
  selectGfx950();
  rocisa::Module module("smoke");
  module.addT<rocisa::SLoadB64>(rocisa::sgpr(4, 2), rocisa::sgpr(0, 2), 0);
  module.addT<rocisa::SWaitCnt>(-1, -1, -1, 0);
  module.addT<rocisa::SMovB32>(rocisa::sgpr(6), 1);
  module.addT<rocisa::VMovB32>(rocisa::vgpr(1), 0);
  module.addT<rocisa::GlobalLoadB32>(rocisa::vgpr(2), rocisa::vgpr(1),
                                     rocisa::sgpr(4, 2));
  module.addT<rocisa::SWaitCnt>(0, -1, -1, -1);
  module.addT<rocisa::VAddF32>(rocisa::vgpr(3), rocisa::vgpr(2),
                               rocisa::vgpr(2));
  module.addT<rocisa::SEndpgm>();
  std::string text = module.toString();
  expectContains(text, "s_load_dwordx2 s[4:5], s[0:1], 0");
  expectContains(text, "s_waitcnt lgkmcnt(0)");
  expectContains(text, "s_mov_b32 s6, 1");
  expectContains(text, "v_mov_b32 v1, 0");
  expectContains(text, "global_load_dword v2, v1, s[4:5]");
  expectContains(text, "s_waitcnt vmcnt(0)");
  expectContains(text, "v_add_f32 v3, v2, v2");
  expectContains(text, "s_endpgm");
}

TEST(RocisaCore, EmitsScaledPackedFp8ToBf16Convert) {
  selectGfx950();
  rocisa::Module module("cvt");
  module.addT<rocisa::VCvtScalePkFP8toBF16>(rocisa::vgpr(78), rocisa::vgpr(0),
                                            rocisa::vgpr(76));
  module.addT<rocisa::VCvtScalePkFP8toBF16>(rocisa::vgpr(0), rocisa::vgpr(0),
                                            rocisa::vgpr(76), std::nullopt,
                                            rocisa::VOP3PModifiers({1, 0, 0}));
  std::string text = module.toString();
  expectContains(text, "v_cvt_scalef32_pk_bf16_fp8 v78, v0, v76");
  expectContains(text, "v_cvt_scalef32_pk_bf16_fp8 v0, v0, v76 op_sel:[1,0,0]");
}

TEST(RocisaCore, EmitsArithmeticShiftRight64) {
  selectGfx950();
  rocisa::Module module("ashr64");
  module.addT<rocisa::SAShiftRightI64>(rocisa::sgpr(4, 2), rocisa::sgpr(6),
                                       rocisa::sgpr(0, 2));
  module.addT<rocisa::SAShiftRightI64>(rocisa::sgpr(4, 2), 3,
                                       rocisa::sgpr(0, 2));
  module.addT<rocisa::VAShiftRightI64>(rocisa::vgpr(2, 2), rocisa::vgpr(4),
                                       rocisa::vgpr(0, 2));
  module.addT<rocisa::VAShiftRightI64>(rocisa::vgpr(2, 2), 3,
                                       rocisa::vgpr(0, 2));
  module.addT<rocisa::VAShiftRightI64>(rocisa::vgpr(2, 2), rocisa::sgpr(6),
                                       rocisa::vgpr(0, 2));
  std::string text = module.toString();
  expectContains(text, "s_ashr_i64 s[4:5], s[0:1], s6");
  expectContains(text, "s_ashr_i64 s[4:5], s[0:1], 3");
  expectContains(text, "v_ashrrev_i64 v[2:3], v4, v[0:1]");
  expectContains(text, "v_ashrrev_i64 v[2:3], 3, v[0:1]");
  expectContains(text, "v_ashrrev_i64 v[2:3], s6, v[0:1]");
}

// Builds the cross-lane reduction forms: full bank mask and optional
// bound_ctrl.
static rocisa::DPPModifiers reductionDpp(int32_t rowBcast,
                                         const std::vector<int32_t> &quadPerm,
                                         bool rowMirror, bool rowHalfMirror,
                                         int32_t rowMask, int32_t boundCtrl) {
  return rocisa::DPPModifiers(-1, rowBcast, boundCtrl, quadPerm, -1, rowMirror,
                              rowHalfMirror, rowMask, 0xf);
}

TEST(RocisaCore, EmitsDppCrossLaneReduction) {
  selectGfx950();
  EXPECT_TRUE(rocisa::DPPModifiers().toString().empty());
  EXPECT_EQ(rocisa::DPPModifiers(2, -1, 0).toString(),
            " row_shr:2 bound_ctrl:0");

  rocisa::Module module("dpp");
  module.addT<rocisa::VAddF32>(
      rocisa::vgpr(0), rocisa::vgpr(64), rocisa::vgpr(64),
      reductionDpp(-1, {1, 0, 3, 2}, false, false, 0xf, 1));
  module.addT<rocisa::VAddF32>(
      rocisa::vgpr(0), rocisa::vgpr(0), rocisa::vgpr(0),
      reductionDpp(-1, {2, 3, 0, 1}, false, false, 0xf, 1));
  module.addT<rocisa::VAddF32>(rocisa::vgpr(0), rocisa::vgpr(0),
                               rocisa::vgpr(0),
                               reductionDpp(-1, {}, false, true, 0xf, 1));
  module.addT<rocisa::VAddF32>(rocisa::vgpr(0), rocisa::vgpr(0),
                               rocisa::vgpr(0),
                               reductionDpp(-1, {}, true, false, 0xf, 1));
  module.addT<rocisa::VAddF32>(rocisa::vgpr(0), rocisa::vgpr(0),
                               rocisa::vgpr(0),
                               reductionDpp(15, {}, false, false, 0xa, -1));
  module.addT<rocisa::VAddF32>(rocisa::vgpr(0), rocisa::vgpr(0),
                               rocisa::vgpr(0),
                               reductionDpp(31, {}, false, false, 0xf, 1));
  std::string text = module.toString();
  expectContains(text,
                 "v_add_f32 v0, v64, v64 quad_perm:[1,0,3,2] row_mask:0xf "
                 "bank_mask:0xf bound_ctrl:1");
  expectContains(text, "v_add_f32 v0, v0, v0 quad_perm:[2,3,0,1] row_mask:0xf "
                       "bank_mask:0xf bound_ctrl:1");
  expectContains(text, "v_add_f32 v0, v0, v0 row_half_mirror row_mask:0xf "
                       "bank_mask:0xf bound_ctrl:1");
  expectContains(text, "v_add_f32 v0, v0, v0 row_mirror row_mask:0xf "
                       "bank_mask:0xf bound_ctrl:1");
  expectContains(text, "v_add_f32 v0, v0, v0 row_bcast:15 row_mask:0xa "
                       "bank_mask:0xf\n");
  expectContains(text, "v_add_f32 v0, v0, v0 row_bcast:31 row_mask:0xf "
                       "bank_mask:0xf bound_ctrl:1");
}
