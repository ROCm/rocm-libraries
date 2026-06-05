/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "stinkytofu/hardware/HwReg.hpp"

#include <algorithm>

#include "stinkytofu/Config/Config.h"
#include "stinkytofu/hardware/ArchHelper.hpp"

namespace stinkytofu {
namespace HwReg {

// Row type referenced by the generated `kHwregByName_<Arch>[]` / `kHwregById_<Arch>[]` tables.
struct NamedId {
    std::string_view name;
    Id id;
};

namespace {

// Per-arch generated name tables.
#ifdef STINKYTOFU_ARCH_GFX1250
#include "hardware/generated/Gfx1250_hwreg.inc"
#endif

bool isGfx12Plus(GfxArchID arch) {
    const auto* info = ArchHelper::getInstance().getArchInfo(arch);
    return info && info->major >= 12;
}

}  // namespace

// Dispatch to the per-arch nameToId<Arch> / idToName<Arch> generated alongside
// the kHwreg{ByName,ById}_<Arch> tables by tablegen.
bool nameToId(GfxArchID arch, std::string_view name, Id& out) {
#ifdef STINKYTOFU_ARCH_GFX1250
    if (arch == GfxArchID::Gfx1250) return nameToIdGfx1250(name, out);
#endif
    (void)arch;
    (void)name;
    (void)out;
    return false;
}

std::string_view idToName(GfxArchID arch, uint16_t id) {
#ifdef STINKYTOFU_ARCH_GFX1250
    if (arch == GfxArchID::Gfx1250) return idToNameGfx1250(id);
#endif
    (void)arch;
    (void)id;
    return {};
}

uint16_t schedModeId(GfxArchID arch) {
    if (isGfx12Plus(arch)) return ID_SCHED_MODE_GFX12;
    return 0;
}

SubField schedModeDepMode(GfxArchID arch) {
    if (isGfx12Plus(arch)) return {/*offset=*/0, /*size=*/2};
    return {0, 0};
}

SubField schedModeDisableXdlArbStall(GfxArchID arch) {
    if (isGfx12Plus(arch)) return {/*offset=*/4, /*size=*/1};
    return {0, 0};
}

}  // namespace HwReg
}  // namespace stinkytofu
