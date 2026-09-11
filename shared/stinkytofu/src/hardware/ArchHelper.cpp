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

#include <cassert>
#include <cstdint>
#include <string>
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
    // Bounds-checked: a GfxArchID is an index into the per-build arch list, and a
    // build registers only the one stepping it selected. Callers that derive an id
    // arithmetically (e.g. to probe a "some other arch" path) can land past the end,
    // so return nullptr rather than indexing out of range. This is what makes the
    // nullptr guards in isGfx125() and the getWaveFrontSize()-family asserts real.
    const auto idx = static_cast<size_t>(arch);
    if (idx >= registeredArchInfos.size()) return nullptr;
    return registeredArchInfos[idx].get();
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

const ArchHelper::ArchInfo* ArchHelper::getArchInfo(const std::string& name) const {
    for (const auto& archInfo : registeredArchInfos) {
        if (archInfo && archInfo->name == name) {
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
            return static_cast<GfxArchID>(
                i);  // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
        }
    }
    assert(false && "Unsupported GfxArchID");
    return static_cast<GfxArchID>(0);
}

GfxArchID ArchHelper::getGfxArchID(const std::string& name) const {
    for (size_t i = 0; i < registeredArchInfos.size(); ++i) {
        const auto& archInfo = registeredArchInfos[i];
        if (archInfo && archInfo->name == name) {
            return static_cast<GfxArchID>(
                i);  // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
        }
    }
    // Exceptions are disabled in this build, so mirror the triple overload: assert in debug and
    // fall back to the first-registered arch in release rather than surfacing an error.
    assert(false && "Unknown stinkytofu arch name");
    return static_cast<GfxArchID>(0);
}

}  // namespace stinkytofu
