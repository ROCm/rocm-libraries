// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "stinkytofu/hardware/ToolchainCaps.hpp"

#include <cassert>
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

/// Static, hand-maintained fallback used when the installed comgr does not
/// know about a given ISA (e.g. a newer arch shipped before the platform
/// toolchain is updated).  Without this, runtime probing reports None and
/// downstream passes (e.g. InsertVgprMsbPass) silently no-op.
AsmCapsConfig staticFallbackCaps(GfxArchID archID) {
    AsmCapsConfig caps;
    switch (archID) {
        case GfxArchID::Gfx1250:
            caps.vgprMsbMode = VgprMsbMode::Msb16;
            break;
        default:
            break;
    }
    return caps;
}

void doProbe(GfxArchID archID, ArchCapsEntry& entry) {
    const auto* info = ArchHelper::getInstance().getArchInfo(archID);
    if (!info) return;

    std::string isaName = formatIsaName(info);
    uint32_t ws = info->waveFrontSize;

    // If comgr support isn't available, or the installed comgr doesn't list
    // this ISA, fall back to the static known-arch table so newer archs work
    // on older toolchains.
    if (!hasComgrSupport() || !comgrSupportsIsa(isaName)) {
        entry.caps = staticFallbackCaps(archID);
        return;
    }

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
