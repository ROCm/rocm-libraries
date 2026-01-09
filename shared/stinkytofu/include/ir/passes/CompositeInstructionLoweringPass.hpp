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

#include "ir/passes/PassManager.hpp"
#include "stinkytofu.hpp"
#include <vector>

namespace stinkytofu
{
    // Forward declarations
    class IRInstruction;

    /**
     * @brief Expands composite IR instructions based on architecture capabilities
     *
     * Composite instructions are high-level operations that may map to:
     * - Single instruction if architecture supports it (e.g., v_pk_add_f32)
     * - Multiple instructions if architecture doesn't support it (e.g., 2x v_add_f32)
     *
     * Examples:
     * - VAddPKF32:
     *   - gfx9+: Single v_pk_add_f32 instruction
     *   - Fallback: Two v_add_f32 instructions (low/high)
     *
     * - VMovB64:
     *   - If supported: Single v_mov_b64 instruction
     *   - Fallback: Two v_mov_b32 instructions
     *
     * - VLShiftLeftOrB32:
     *   - If supported: Single v_lshl_or_b32 instruction
     *   - Fallback: v_lshlrev_b32 + v_or_b32
     *
     * This pass runs BEFORE ToStinkyAsmPass and produces simple IR instructions
     * that have 1:1 mappings to assembly.
     */
    class CompositeInstructionLoweringPass : public IRInstTransformPass
    {
    public:
        /**
         * @brief Get the name of this pass
         */
        const char* getName() const override
        {
            return "CompositeInstructionLoweringPass";
        }

        /**
         * @brief Transform (expand) a composite IR instruction
         *
         * @param irInst IR instruction (composite or simple)
         * @param arch Target architecture for capability queries
         * @return Vector of simple IR instructions (may be 1 or more)
         *
         * @note The returned instructions are newly allocated and caller takes ownership
         */
        std::vector<IRInstruction*> transform(IRInstruction* irInst, GfxArchID arch) override;

    private:
        // Helper to check if instruction is supported on target architecture
        bool isInstructionSupported(const std::string& mnemonic, GfxArchID arch) const;

        // Architecture capability queries (use isInstructionSupported)
        bool hasVPKAddF32(GfxArchID arch) const;
        bool hasVPKMulF32(GfxArchID arch) const;
        bool hasVMovB64(GfxArchID arch) const;
        bool hasVLShlOrB32(GfxArchID arch) const;
    };
} // namespace stinkytofu
