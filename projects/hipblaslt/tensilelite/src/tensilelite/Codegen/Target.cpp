// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/Codegen/Target.hpp>

#include "base.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace rocm::tensilelite::codegen;

Target::Target(IsaVersion isa, int32_t wavefront, int32_t xcds,
               std::string_view arch, std::string_view isaFullName)
    : isa(isa), wavefront(wavefront), xcds(xcds), arch(arch),
      isaFullName(isaFullName) {
  rocisa::rocIsa &rocIsa = rocisa::rocIsa::getInstance();
  rocIsa.init(isa, TENSILELITE_CODEGEN_ASSEMBLER);
  // getIsaInfo returns by value, so keep the copy alive.
  rocisa::IsaInfo info = rocIsa.getIsaInfo(isa);
  maxVgprs = info.reg_caps.at("MaxVgpr");
  maxSgprs = info.reg_caps.at("MaxSgpr");
  maxUnifiedVgprs = info.reg_caps.at("PhysicalMaxVgpr");
  maxLdsBytes = info.arch_caps.at("DeviceLDS");
  maxVmcnts = info.asm_caps.at("MaxVmcnt");
  maxLgkmcnts = info.asm_caps.at("MaxLgkmcnt");
  // rocisa has no AGPR cap; this mirrors the kernel writer's maxLimitAgprs.
  maxAgprs = maxUnifiedVgprs - maxVgprs;
}

Target Target::get(std::string_view arch) {
  if (arch != "gfx950")
    throw std::invalid_argument("unsupported codegen target '" +
                                std::string(arch) +
                                "'; supported targets: gfx950");
  // A function-local static runs rocisa's expensive, unsynchronized init once.
  // rocisa has no XCD cap, so the count is a literal.
  static const Target gfx950({9, 5, 0}, 64, 8, "gfx950",
                             "amdgcn-amd-amdhsa--gfx950");
  return gfx950;
}

void Target::selectForCurrentThread() const {
  rocisa::rocIsa::getInstance().setKernel(isa, wavefront);
}
