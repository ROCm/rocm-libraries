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
#include <utility>
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
        ArchInfo(std::string name, uint32_t major, uint32_t minor, uint32_t stepping,
                 uint32_t waveFrontSize, uint32_t totalVgprPerSimd = 0,
                 uint32_t vgprAllocGranule = 0)
            : name(std::move(name)),
              major(major),
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

        // True on targets where ECC forces every VGPR write to be 32 bits wide.
        // On such targets a D16 VMEM op zero-fills the non-data 16-bit half, and
        // True16 VALU performs the write at full-DWORD granularity.
        virtual bool hasD16Writes32BitVgpr() const = 0;

        // Concrete stepping identity (lowercase, e.g. "gfx1250" vs "gfx1250v0"). Distinguishes
        // steppings that share an ISA triple, which the triple alone cannot. Set by the generated
        // ArchInfo subclass; used by getGfxArchID(name) to resolve an identity the triple can't.
        const std::string name;

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

    // Resolve by concrete identity name (ArchInfo::name), disambiguating steppings that share an
    // ISA triple (e.g. gfx1250 v1 vs gfx1250v0). Unlike getGfxArchID(const std::string&), an
    // unknown name returns nullptr instead of asserting, so callers can probe a name and fall
    // back to legacy triple parsing without aborting in debug builds.
    const ArchInfo* getArchInfo(const std::string& name) const;

    const GfxArchID getGfxArchID(uint32_t major, uint32_t minor, uint32_t stepping) const;

    // Resolve by concrete identity name (ArchInfo::name), disambiguating steppings that share an
    // ISA triple. On an unknown name it asserts (debug) and falls back to the first-registered
    // arch (release), matching the triple overload; exceptions are disabled in this build.
    GfxArchID getGfxArchID(const std::string& name) const;

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

inline GfxArchID getGfxArchID(const std::string& name) {
    return ArchHelper::getInstance().getGfxArchID(name);
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

// True for the gfx12.5 family from a raw ISA-triple major/minor.
//
// Deliberately minor-exact, not major>=12: gfx1200/gfx1201 ({12,0,x}) are a different family
// that the gfx12.5 callers do not yet handle (see the FIXMEs at the call sites), and widening
// to major>=12 would silently pull them into gfx12.5's paths. (Contrast HwReg.cpp's private
// isGfx12Plus, which is intentionally the broader major>=12 test for scheduling-mode fields.)
//
// Nothing exercises the minor-exactness: a build registers a single stepping, so every call
// evaluates at (12,5) and a widened predicate would behave identically until a gfx12.0 arch is
// registered. Widen it only alongside the call sites' FIXMEs.
inline constexpr bool isGfx125(uint32_t major, uint32_t minor) {
    return major == 12 && minor == 5;
}

// True for the gfx12.5 family, i.e. both the Gfx1250 (v1) and Gfx1250v0 (v0) steppings of the
// {12,5,x} ISA triple. Prefer this over comparing against GfxArchID::Gfx1250 directly: that
// enumerator does not exist in a build that selected only the v0 stepping, so naming it breaks
// the v0-only build. The two steppings are behaviourally identical outside instruction timing,
// so any code that today special-cases Gfx1250 wants both.
inline bool isGfx125(GfxArchID archID) {
    const auto* archInfo = ArchHelper::getInstance().getArchInfo(archID);
    return archInfo != nullptr && isGfx125(archInfo->major, archInfo->minor);
}

}  // namespace stinkytofu
