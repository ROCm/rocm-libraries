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
 * THE SOFTWARE IS PROVIDED "AS IS") WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
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
#include <functional>
#include <memory>
#include <vector>

#include "gfx/InstDefDSL.hpp"

namespace stinkytofu {
// Each GPU architecture is defined in a separate file.
// GpuArchManager::initAllArchs initializes all of the architectures.
//
// GpuArchManager also manages a unified opcode table for all of the
// architectures.
class GpuArchManager {
   private:
    std::vector<std::unique_ptr<GpuArch>> instructionsByArch;

    // All of the opcodes for all of the architectures.
    // Sorted by alphabetical order of the mnemonic strings.
    std::vector<std::string> allOpcodes;

    void addArch(const std::string& arch, const std::function<void(GpuArch&)>& defineInsts,
                 const std::function<void(GpuArch&)>& setLogicalToArchMap,
                 const std::function<void(GpuArch&)>& setRocisaToArchMap,
                 const std::function<void(GpuArch&)>& setRocisaConversionMap);

    void enumAllOpcodes();

    bool error = false;

   public:
    bool hasError() const {
        return error;
    }

    const std::vector<std::string>& getAllOpcodes() const {
        return allOpcodes;
    }

    const GpuArch* getArch(const std::string& archName) const {
        for (const auto& arch : instructionsByArch)
            if (arch->getName() == archName) return arch.get();

        assert(false && "Internal error: Arch not found");
        return nullptr;
    }

    std::vector<std::string> getRegisteredArchNames() const {
        std::vector<std::string> archs;
        for (auto& arch : instructionsByArch) archs.push_back(arch->getName());
        return archs;
    }

    std::vector<const GpuArch*> getRegisteredArchs() const {
        std::vector<const GpuArch*> archs;
        for (auto& arch : instructionsByArch) archs.push_back(arch.get());
        return archs;
    }

    static bool initAllArchs(GpuArchManager& manager);
};

}  // namespace stinkytofu
