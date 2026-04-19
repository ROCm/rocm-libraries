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

#pragma once
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "stinkytofu/hardware/GfxIsa.hpp"

namespace stinkytofu {
// The base gfx instruction.
struct GfxInstDef {
    std::string name;

    HwInstDesc hwInstDesc;

    // for debugging
    std::string definedFile;
    int definedLine = 0;

    GfxInstDef() = default;

    explicit GfxInstDef(InstFlagSet flags) : hwInstDesc() {
        hwInstDesc.flags = flags;
    }

    explicit GfxInstDef(std::initializer_list<InstFlag> flags) : GfxInstDef(makeFlagSet(flags)) {}

    static std::string getFlagStr(InstFlag flag) {
        switch (flag) {
#define MACRO(flagName) \
    case flagName:      \
        return #flagName;
#include "stinkytofu/hardware/Flags.def"

#undef MACRO
            default:
                assert(false && "Unknown InstFlag");
                return "Unknown";
        }
    }

    bool is(InstFlag f) const {
        return hwInstDesc.flags.test(f);
    }

    std::string getFlagsStr() const {
        std::string res;
        for (InstFlag f = IF_BEGIN; f < IF_COUNT; f = InstFlag(f + 1)) {
            if (is(f)) {
                if (!res.empty()) {
                    res += ", ";
                }
                res += getFlagStr(f);
            }
        }
        return res;
    }

    virtual ~GfxInstDef() = default;
};

// GpuArch is a collection of instructions for a specific GPU architecture
// (named by the string).
//
// It contains a list of instructions with their properties (e.g. cycle,
// latency, .. etc), and provide query and add/remove functions.
//
// It also contains a error flag to indicate if there is any error during
// the initialization.
//
// Note the finalize function must be called after no more changes are made
// to the instructions.
struct GpuArch {
    using List = std::vector<std::unique_ptr<GfxInstDef>>;

    struct RegisterLimits {
        unsigned maxVGPR = 256;  // Vector GPRs (v0-v255)
        unsigned maxSGPR = 128;  // Scalar GPRs (s0-s127)
        unsigned maxAGPR = 64;   // Accumulator GPRs (a0-a63, acc0-acc63)
    };

   private:
    List instructions;

    std::unordered_map<std::string, GfxInstDef*> added;

    bool error = false;

    bool finalized = false;

    std::string name;

    uint32_t waveFrontSize = 64;  // Default to 64, can be overridden per architecture

    RegisterLimits registerLimits;  // Register limits for this architecture

    // Cost map system: instruction-specific costs (issue, latency)
    std::unordered_map<std::string, std::pair<uint16_t, uint16_t>> instructionCosts_;

    // Per-architecture defaults (must be explicitly set, 0 = not set/error)
    uint16_t defaultCycle_ = 0;
    uint16_t defaultLatency_ = 0;

    GfxInstDef* getInst(const std::string& name);

    // mapping from logical instruction name to architecture-specific instruction name (StinkyTofu
    // Logical IR)
    std::unordered_map<std::string, std::string> logicalToArchMap;
    // mapping from Rocisa (tensilelite) type name to architecture-specific mnemonic; used for
    // RocisaGfx*Mappings.inc
    std::unordered_map<std::string, std::string> rocisaToArchMap;
    std::unordered_map<std::string, std::string> rocisaConversionMap;

   public:
    GpuArch(const std::string& name) : name(name) {}

    const std::string& getName() const {
        return name;
    }

    uint32_t getWaveFrontSize() const {
        return waveFrontSize;
    }

    void setWaveFrontSize(uint32_t size) {
        waveFrontSize = size;
    }

    const RegisterLimits& getRegisterLimits() const {
        return registerLimits;
    }

    void setRegisterLimits(const RegisterLimits& limits) {
        registerLimits = limits;
    }

    const List& getInstructions() const {
        return instructions;
    }

    bool has(const std::string& name) const {
        return added.find(name) != added.end();
    }

    const GfxInstDef* getInst(const std::string& name) const {
        return added.find(name)->second;
    }

    bool add(std::unique_ptr<GfxInstDef> inst);
    bool erase(const std::string& name);

    // Cost map system: Set default costs for this architecture
    // Must be called before applyInstructionCosts()
    // cycle and latency must be non-zero
    void setDefaultCosts(uint16_t cycle, uint16_t latency);

    // Cost map system: Register instruction-specific cost
    // Used for instructions that differ from defaults
    void setInstructionCost(const std::string& opcode, uint16_t cycle, uint16_t latency);

    // Cost map system: Apply costs to all instructions with strict validation
    // Returns false if:
    //   - setDefaultCosts() was not called
    //   - Any instruction ends up with 0 issue or latency
    // Prints detailed error messages
    bool applyInstructionCosts();

    void finalize(uint16_t startOpcode = 0) {
        assert(!finalized && "GpuArch already finalized");
        finalized = true;

        assert((instructions.size() + startOpcode) <= UINT16_MAX &&
               "Running out of opcodes! Please expand the opcode type to uint32_t!");
        for (size_t i = 0; i < instructions.size(); ++i)
            instructions[i]->hwInstDesc.isaOpcode = static_cast<uint16_t>(i) + startOpcode;
    }

    bool hasError() const {
        return error;
    }

    void setLogicalToArchMap(const std::unordered_map<std::string, std::string>&& map) {
        logicalToArchMap = std::move(map);
    }

    const std::unordered_map<std::string, std::string>& getLogicalToArchMap() const {
        return logicalToArchMap;
    }

    void setRocisaToArchMap(const std::unordered_map<std::string, std::string>&& map) {
        rocisaToArchMap = std::move(map);
    }

    const std::unordered_map<std::string, std::string>& getRocisaToArchMap() const {
        return rocisaToArchMap;
    }

    void setRocisaConversionMap(const std::unordered_map<std::string, std::string>&& map) {
        rocisaConversionMap = std::move(map);
    }

    const std::unordered_map<std::string, std::string>& getRocisaConversionMap() const {
        return rocisaConversionMap;
    }
};

// keep the last `keepDepth` directories and the filename.
std::string getReducedFilename(const char* filename, unsigned keepDepth);

// generic "def" like TableGen
template <typename T, typename... Args>
T* defT(const std::string name, GpuArch& registry, const char* file, size_t line, Args&&... args) {
    auto inst = std::make_unique<T>(std::forward<Args>(args)...);
    T* ptr = inst.get();

    inst->name = name;
    inst->definedFile = getReducedFilename(file, 0);
    inst->definedLine = line;
    registry.add(std::move(inst));

    return ptr;
}

// Overload: create GfxInstDef with flags (no subclass). Used by generated _init.inc.
inline GfxInstDef* defT(const std::string& name, GpuArch& registry, const char* file, size_t line,
                        std::initializer_list<InstFlag> flags) {
    auto inst = std::make_unique<GfxInstDef>(makeFlagSet(flags));
    GfxInstDef* ptr = inst.get();

    inst->name = name;
    inst->definedFile = getReducedFilename(file, 0);
    inst->definedLine = static_cast<int>(line);
    registry.add(std::move(inst));

    return ptr;
}

#define DEF_T(name, ...) defT(name, registry, __FILE__, __LINE__, {__VA_ARGS__})

}  // namespace stinkytofu
