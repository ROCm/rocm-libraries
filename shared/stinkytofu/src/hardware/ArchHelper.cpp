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
#include "stinkytofu/hardware/ArchHelper.hpp"

#include <unordered_map>

#include "stinkytofu/Config/Config.h"

namespace {
#define GET_ISAINFO_UNIFIED_OPCODES
#include "hardware/gfxIsa.inc"
}  // namespace

/* Architecture-specific headers (GfxXXX.hpp defines GfxXXXArchInfo) - auto-generated in build dir
 */
#include "arch_headers/ArchHelper_includes.inc"

namespace stinkytofu {
ArchHelper::ArchHelper() {
// Populate the fixed list of architecture infos
#define STINKYTOFU_ARCH(archName) \
    registeredArchInfos.push_back(std::make_unique<archName##ArchInfo>());
#include "Config/Archs.def"
}

const ArchHelper::ArchInfo* ArchHelper::getArchInfo(GfxArchID arch) const {
    return registeredArchInfos[static_cast<int>(arch)].get();
}

const ArchHelper::ArchInfo* ArchHelper::getArchInfo(uint32_t major, uint32_t minor,
                                                    uint32_t stepping) const {
    for (const auto& archInfo : registeredArchInfos) {
        if (archInfo->major == major && archInfo->minor == minor &&
            archInfo->stepping == stepping) {
            return archInfo.get();
        }
    }
    return nullptr;
}

const GfxArchID ArchHelper::getGfxArchID(uint32_t major, uint32_t minor, uint32_t stepping) const {
    for (size_t i = 0; i < registeredArchInfos.size(); ++i) {
        const auto& archInfo = registeredArchInfos[i];
        if (archInfo->major == major && archInfo->minor == minor &&
            archInfo->stepping == stepping) {
            return static_cast<GfxArchID>(i);
        }
    }
    assert(false && "Unsupported GfxArchID");
}

}  // namespace stinkytofu
