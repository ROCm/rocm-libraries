/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

#pragma once

#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"

namespace stinkytofu {
// ArchHelper provides a centralized interface for querying architecture-specific
// information such as instruction descriptors, opcode mappings, and mnemonic lookups.
//
// This class eliminates the need for scattered switch statements throughout the codebase
// by encapsulating all architecture-specific queries in one place.
class STINKYTOFU_EXPORT ArchHelper {
   public:
    struct ArchInfo {
        ArchInfo(uint32_t major, uint32_t minor, uint32_t stepping, uint32_t waveFrontSize,
                 uint32_t totalVgprPerSimd = 0, uint32_t vgprAllocGranule = 0)
            : major(major),
              minor(minor),
              stepping(stepping),
              waveFrontSize(waveFrontSize),
              totalVgprPerSimd(totalVgprPerSimd),
              vgprAllocGranule(vgprAllocGranule) {}

        virtual ~ArchInfo() = default;

        virtual stinkytofu::IsaOpcode getIsaOpcode(
            stinkytofu::UnifiedOpcode unifiedOpcode) const = 0;
        virtual const HwInstDesc* getMCIDTable() const = 0;
        virtual const std::unordered_map<std::string, uint16_t>& getMnemonicToIsaOpcodeMap()
            const = 0;

        const uint32_t major;
        const uint32_t minor;
        const uint32_t stepping;

        const uint32_t waveFrontSize;

        const uint32_t totalVgprPerSimd;
        const uint32_t vgprAllocGranule;
    };

   public:
    static const ArchHelper& getInstance() {
        static ArchHelper instance;
        return instance;
    }

    const ArchInfo* getArchInfo(GfxArchID arch) const;

    const ArchInfo* getArchInfo(uint32_t major, uint32_t minor, uint32_t stepping) const;

    const GfxArchID getGfxArchID(uint32_t major, uint32_t minor, uint32_t stepping) const;

   private:
    // Private constructor: Populate the fixed list here
    ArchHelper();
    ArchHelper(const ArchHelper&) = delete;
    ArchHelper& operator=(const ArchHelper&) = delete;

    std::vector<std::unique_ptr<ArchInfo>> registeredArchInfos;
};

inline GfxArchID getGfxArchID(uint32_t major, uint32_t minor, uint32_t stepping) {
    return ArchHelper::getInstance().getGfxArchID(major, minor, stepping);
}

inline uint32_t getWaveFrontSize(GfxArchID archID) {
    const auto* archInfo = ArchHelper::getInstance().getArchInfo(archID);
    assert(archInfo && "Invalid GfxArchID");
    return archInfo->waveFrontSize;
}

inline uint32_t getWaveFrontSize(uint32_t major, uint32_t minor, uint32_t stepping) {
    return getWaveFrontSize(getGfxArchID(major, minor, stepping));
}

inline uint32_t getTotalVgprPerSimd(GfxArchID archID) {
    const auto* archInfo = ArchHelper::getInstance().getArchInfo(archID);
    assert(archInfo && "Invalid GfxArchID");
    return archInfo->totalVgprPerSimd;
}

inline uint32_t getVgprAllocGranule(GfxArchID archID) {
    const auto* archInfo = ArchHelper::getInstance().getArchInfo(archID);
    assert(archInfo && "Invalid GfxArchID");
    return archInfo->vgprAllocGranule;
}

// Waves per SIMD a kernel using `kernelVgprs` per wave can achieve on `archID`.
// Returns std::numeric_limits<int>::max() for inputs without enough information
// to compute occupancy: unknown arch caps, unknown allocation (`kernelVgprs == 0`),
// or a kernel that asks for more VGPRs than the SIMD physically has.
inline int getWavesPerSimd(GfxArchID archID, int kernelVgprs) {
    assert(kernelVgprs >= 0 && "kernelVgprs must be non-negative");
    const uint32_t total = getTotalVgprPerSimd(archID);
    const uint32_t granule = getVgprAllocGranule(archID);
    if (total == 0 || granule == 0) return std::numeric_limits<int>::max();
    if (kernelVgprs <= 0) return std::numeric_limits<int>::max();
    const uint32_t rounded = ((kernelVgprs + granule - 1) / granule) * granule;
    if (rounded > total) return std::numeric_limits<int>::max();
    return static_cast<int>(total / rounded);
}

inline std::string getArchName(GfxArchID archID) {
    const auto* archInfo = ArchHelper::getInstance().getArchInfo(archID);
    if (!archInfo) return "gfx_unknown";
    return "gfx" + std::to_string(archInfo->major) + std::to_string(archInfo->minor) +
           std::to_string(archInfo->stepping);
}

}  // namespace stinkytofu
