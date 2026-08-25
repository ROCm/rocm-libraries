// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "stinkytofu/hardware/ToolchainCaps.hpp"

#include <cassert>
#include <cstdint>
#include <mutex>
#include <string>

#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/ComgrProbe.hpp"

namespace stinkytofu {
namespace {

std::string formatIsaName(const ArchHelper::ArchInfo* info) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string name = "amdgcn-amd-amdhsa--gfx";
    name += std::to_string(info->major);
    name += std::to_string(info->minor);
    name += kHex[info->stepping & 0xF];
    return name;
}

struct ArchCapsEntry {
    std::once_flag flag;
    AsmCapsConfig caps;
};

constexpr size_t kMaxArchs = 8;
ArchCapsEntry g_cache[kMaxArchs];

void doProbe(GfxArchID archID, ArchCapsEntry& entry) {
    if (!hasComgrSupport()) return;

    const auto* info = ArchHelper::getInstance().getArchInfo(archID);
    if (!info) return;

    std::string isaName = formatIsaName(info);
    uint32_t ws = info->waveFrontSize;

    bool hasMsb = tryAssembleWithComgr("s_set_vgpr_msb 0", isaName, ws);
    bool hasMsb16 = hasMsb && tryAssembleWithComgr("s_set_vgpr_msb 0x0101", isaName, ws);

    entry.caps.vgprMsbMode = hasMsb16 ? VgprMsbMode::Msb16
                             : hasMsb ? VgprMsbMode::Msb8
                                      : VgprMsbMode::None;
}

}  // namespace

AsmCapsConfig ToolchainCaps::probe(GfxArchID archID) {
    auto idx = static_cast<size_t>(archID);
    if (idx >= kMaxArchs) return {};
    auto& entry = g_cache[idx];
    std::call_once(entry.flag, doProbe, archID, std::ref(entry));
    return entry.caps;
}

}  // namespace stinkytofu
