// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "base.hpp"
#include "code.hpp"
#include "container.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"

#include <gtest/gtest.h>

#include <string>
#include <string_view>

namespace {

constexpr IsaVersion gfx950 = {9, 5, 0};
constexpr int wavefrontSize = 64;

// rocIsa keeps the selected kernel per thread, so each test selects it.
void selectGfx950() {
  rocisa::rocIsa &isa = rocisa::rocIsa::getInstance();
  isa.init(gfx950, ROCISA_CORE_TEST_ASSEMBLER);
  isa.setKernel(gfx950, wavefrontSize);
}

void expectContains(std::string_view text, std::string_view needle) {
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

} // namespace
