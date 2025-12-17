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

#include "ir/asm/StinkyAsmIR.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace stinkytofu
{
    // Forward declarations
    class StinkyIRImpl;
    class StinkyTofu;
    struct StinkyInstruction;
    struct StinkyRegister;

    /**
     * @brief High-level IR builder for StinkyTofu
     * 
     * StinkyIR provides high-level functions that generate complex instruction sequences.
     * These functions (@function) implement common algorithms like division, multiplication,
     * and branching logic. Each function uses a provided StinkyTofu builder to create
     * instructions that can be added to an IRListModule.
     * 
     * Usage:
     *   StinkyTofu st([9, 4, 2]);  // Builder that owns instructions
     *   StinkyIR ir([9, 4, 2]);     // High-level function generator
     *   auto module = st.createIRList("kernel");
     *   auto insts = ir.vectorStaticDivide(st, qReg, dReg, divisor, tmpVgpr);
     *   module.add(insts);  // Module takes ownership
     */
    class StinkyIR
    {
    public:
        /**
         * @brief Construct a high-level IR function generator
         * @param arch Architecture version as [major, minor, stepping]
         */
        explicit StinkyIR(std::array<int, 3> arch);
        ~StinkyIR();

        // Move semantics
        StinkyIR(StinkyIR&&) noexcept;
        StinkyIR& operator=(StinkyIR&&) noexcept;

        // Disable copy
        StinkyIR(const StinkyIR&)            = delete;
        StinkyIR& operator=(const StinkyIR&) = delete;

        // ========================================================================
        // Division & Remainder Functions (@function)
        // ========================================================================

        /**
         * @brief Static division with power-of-2 or magic number algorithm (@function).
         * 
         * Divides a vector register by a compile-time constant divisor.
         * Automatically selects optimal implementation:
         * - Power of 2: Simple shift operation
         * - Other: Magic number multiplication + shift
         * 
         * @param builder StinkyTofu builder that creates and owns instructions
         * @param qReg Quotient destination VGPR index
         * @param dReg Dividend source VGPR index
         * @param divisor Compile-time constant divisor
         * @param tmpVgpr Vector of 2 temporary VGPRs (only needed for non-power-of-2)
         * @param comment Optional comment
         * @return Vector of instructions implementing the division
         */
        std::vector<StinkyInstruction*> vectorStaticDivide(StinkyTofu&                  builder,
                                                           uint32_t                     qReg,
                                                           uint32_t                     dReg,
                                                           int                          divisor,
                                                           const std::vector<uint32_t>& tmpVgpr,
                                                           const std::string& comment = "");

        /**
         * @brief Static division and remainder with power-of-2 or magic number (@function).
         * 
         * Computes both quotient and remainder for division by a constant.
         * 
         * @param builder StinkyTofu builder that creates and owns instructions
         * @param qReg Quotient destination VGPR index
         * @param rReg Remainder destination VGPR index
         * @param dReg Dividend source VGPR index
         * @param divisor Compile-time constant divisor
         * @param tmpVgpr Vector of 2 temporary VGPRs
         * @param doRemainder If false, only compute quotient
         * @param comment Optional comment
         * @return Vector of instructions
         */
        std::vector<StinkyInstruction*>
            vectorStaticDivideAndRemainder(StinkyTofu&                  builder,
                                           uint32_t                     qReg,
                                           uint32_t                     rReg,
                                           uint32_t                     dReg,
                                           int                          divisor,
                                           const std::vector<uint32_t>& tmpVgpr,
                                           bool                         doRemainder = true,
                                           const std::string&           comment     = "");

        /**
         * @brief Vector unsigned 32-bit division using floating-point reciprocal (@function).
         * 
         * Dynamic division using FP32 reciprocal approximation.
         * Handles arbitrary divisors at runtime.
         * 
         * @param builder StinkyTofu builder that creates and owns instructions
         * @param qReg Quotient destination VGPR
         * @param dReg Dividend VGPR
         * @param divReg Divisor VGPR
         * @param rReg Remainder VGPR (used as temp)
         * @param doRemainder Compute remainder if true
         * @param comment Optional comment
         * @return Vector of instructions
         */
        std::vector<StinkyInstruction*> vectorUInt32DivideAndRemainder(StinkyTofu& builder,
                                                                       uint32_t    qReg,
                                                                       uint32_t    dReg,
                                                                       uint32_t    divReg,
                                                                       uint32_t    rReg,
                                                                       bool doRemainder = true,
                                                                       const std::string& comment
                                                                       = "");

        /**
         * @brief Scalar static division with magic number algorithm (@function).
         * 
         * @param builder StinkyTofu builder that creates and owns instructions
         * @param qReg Quotient destination SGPR
         * @param rReg Remainder destination SGPR
         * @param dReg Dividend SGPR
         * @param divisor Constant divisor
         * @param tmpSgpr Vector of 2 temporary SGPRs
         * @param doRemainder 0=quotient only, 1=both, 2=remainder only
         * @param comment Optional comment
         * @return Vector of instructions
         */
        std::vector<StinkyInstruction*>
            scalarStaticDivideAndRemainder(StinkyTofu&                  builder,
                                           uint32_t                     qReg,
                                           uint32_t                     rReg,
                                           uint32_t                     dReg,
                                           int                          divisor,
                                           const std::vector<uint32_t>& tmpSgpr,
                                           int                          doRemainder = 1,
                                           const std::string&           comment     = "");

        // ========================================================================
        // Multiplication Functions (@function)
        // ========================================================================

        /**
         * @brief Vector multiplication by compile-time constant (@function).
         * 
         * Optimizes multiplication by using shifts/adds when beneficial.
         * 
         * @param builder StinkyTofu builder that creates and owns instructions
         * @param productReg Destination VGPR
         * @param operandReg Source VGPR
         * @param multiplier Constant multiplier
         * @param tmpSgpr Optional temporary SGPR for large constants
         * @param comment Optional comment
         * @return Vector of instructions
         */
        std::vector<StinkyInstruction*> vectorStaticMultiply(StinkyTofu& builder,
                                                             uint32_t    productReg,
                                                             uint32_t    operandReg,
                                                             int         multiplier,
                                                             const std::vector<uint32_t>& tmpSgpr,
                                                             const std::string& comment = "");

        /**
         * @brief Multiply vector register by bytes-per-element (@function).
         * 
         * Common operation for address calculation. Optimized for
         * common BPE values (0.5, 0.75, 1, 2, 4, 8, 16).
         * 
         * @param builder StinkyTofu builder that creates and owns instructions
         * @param dstReg Destination VGPR
         * @param srcReg Source VGPR
         * @param bpe Bytes per element (0.5, 0.75, 1, 2, 4, 8, etc.)
         * @param comment Optional comment
         * @return Vector of instructions
         */
        std::vector<StinkyInstruction*> vectorMultiplyBpe(StinkyTofu&        builder,
                                                          uint32_t           dstReg,
                                                          uint32_t           srcReg,
                                                          float              bpe,
                                                          const std::string& comment = "");

        /**
         * @brief Multiply 64-bit vector register by BPE (@function).
         * 
         * @param builder StinkyTofu builder that creates and owns instructions
         * @param dstReg Destination VGPR pair (dstReg, dstReg+1)
         * @param srcReg Source VGPR pair
         * @param bpe Bytes per element
         * @param tmpReg Temporary VGPR for non-power-of-2 BPE
         * @param comment Optional comment
         * @return Vector of instructions
         */
        std::vector<StinkyInstruction*> vectorMultiply64Bpe(StinkyTofu&        builder,
                                                            uint32_t           dstReg,
                                                            uint32_t           srcReg,
                                                            float              bpe,
                                                            uint32_t           tmpReg,
                                                            const std::string& comment = "");

        /**
         * @brief Scalar multiplication by BPE (@function).
         * 
         * @param builder StinkyTofu builder that creates and owns instructions
         * @param dstReg Destination SGPR
         * @param srcReg Source SGPR
         * @param bpe Bytes per element
         * @param comment Optional comment
         * @return Vector of instructions
         */
        std::vector<StinkyInstruction*> scalarMultiplyBpe(StinkyTofu&        builder,
                                                          uint32_t           dstReg,
                                                          uint32_t           srcReg,
                                                          float              bpe,
                                                          const std::string& comment = "");

        // ========================================================================
        // Branching Functions (@function)
        // ========================================================================

        /**
         * @brief Branch if scalar register is zero (@function).
         * 
         * Implements conditional branching based on scalar comparison.
         * 
         * @param builder StinkyTofu builder that creates and owns instructions
         * @param sgprName SGPR name/index to test
         * @param tmpSgpr Temporary SGPR index
         * @param label Target label for branch
         * @param comment Optional comment
         * @return Vector of instructions
         */
        std::vector<StinkyInstruction*> BranchIfZero(StinkyTofu&        builder,
                                                     uint32_t           sgprName,
                                                     uint32_t           tmpSgpr,
                                                     const std::string& label,
                                                     const std::string& comment = "");

        /**
         * @brief Branch if scalar register is not zero (@function).
         * 
         * @param builder StinkyTofu builder that creates and owns instructions
         * @param sgprName SGPR to test
         * @param label Target label
         * @param comment Optional comment
         * @return Vector of instructions
         */
        std::vector<StinkyInstruction*> BranchIfNotZero(StinkyTofu&        builder,
                                                        uint32_t           sgprName,
                                                        const std::string& label,
                                                        const std::string& comment = "");

        /**
         * @brief Branch if scalar register is zero with data type support (@function).
         * 
         * Supports: "i32", "i64", "f32", "f64"
         * 
         * @param builder StinkyTofu builder that creates and owns instructions
         * @param sgprName SGPR to test
         * @param dataType Data type ("i32", "i64", "f32", "f64")
         * @param tmpVgpr Temporary VGPR index (for FP comparisons)
         * @param label Target label for branch
         * @param comment Optional comment
         * @return Vector of instructions
         */
        std::vector<StinkyInstruction*> BranchIfZeroTyped(StinkyTofu&        builder,
                                                          uint32_t           sgprName,
                                                          const std::string& dataType,
                                                          uint32_t           tmpVgpr,
                                                          const std::string& label,
                                                          const std::string& comment = "");

        /**
         * @brief Branch if scalar register is not zero with data type support (@function).
         * 
         * Supports: "i32", "i64", "f32", "f64"
         * 
         * @param builder StinkyTofu builder that creates and owns instructions
         * @param sgprName SGPR to test
         * @param dataType Data type ("i32", "i64", "f32", "f64")
         * @param label Target label for branch
         * @param comment Optional comment
         * @return Vector of instructions
         */
        std::vector<StinkyInstruction*> BranchIfNotZeroTyped(StinkyTofu&        builder,
                                                             uint32_t           sgprName,
                                                             const std::string& dataType,
                                                             const std::string& label,
                                                             const std::string& comment = "");

        // ============================================================================
        // Casting Functions (f_cast.hpp)
        // ============================================================================

        /**
         * @brief Saturate cast integer to bounds (@function).
         * 
         * Clamps an integer value to [lowerBound, upperBound] range.
         * Supports different saturation modes:
         * - "normal": Both bounds (uses v_med3_i32)
         * - "upper": Upper bound only (uses v_min_i32)
         * - "lower": Lower bound only (requires v_max_i32, not yet implemented)
         * - "none": No saturation
         * 
         * @param builder StinkyTofu builder that creates and owns instructions
         * @param valueReg VGPR containing value to clamp (modified in-place)
         * @param tmpVgpr Temporary VGPR index (for upper bound)
         * @param tmpSgpr Temporary SGPR index (for lower bound)
         * @param lowerBound Lower bound value
         * @param upperBound Upper bound value
         * @param saturateType Saturation type: "normal", "upper", "lower", "none"
         * @param initGpr If true, initialize temp registers with bounds
         * @param comment Optional comment
         * @return Vector of instructions
         */
        std::vector<StinkyInstruction*> VSaturateCastInt(StinkyTofu&        builder,
                                                         uint32_t           valueReg,
                                                         uint32_t           tmpVgpr,
                                                         uint32_t           tmpSgpr,
                                                         int32_t            lowerBound,
                                                         int32_t            upperBound,
                                                         const std::string& saturateType = "normal",
                                                         bool               initGpr      = true,
                                                         const std::string& comment      = "");

        // ============================================================================
        // Memory & Synchronization Functions (functions.hpp)
        // ============================================================================

        /**
         * @brief Initialize LDS (Local Data Share) memory (@function).
         * 
         * This function initializes LDS memory by:
         * 1. Synchronizing threads with barrier
         * 2. Setting up per-thread addresses
         * 3. Writing initialization value to LDS in parallel
         * 4. Synchronizing again to ensure completion
         * 
         * @param builder StinkyTofu builder that creates and owns instructions
         * @param tmpVgprStart Starting VGPR index for temps (needs 2 consecutive VGPRs)
         * @param serialVgpr VGPR containing thread serial number
         * @param numThreads Number of threads participating
         * @param ldsNumElements Number of LDS elements to initialize
         * @param initValue Value to initialize LDS with (default: 0)
         * @param comment Optional comment
         * @return Vector of instructions
         */
        std::vector<StinkyInstruction*> DSInit(StinkyTofu&        builder,
                                               uint32_t           tmpVgprStart,
                                               uint32_t           serialVgpr,
                                               uint32_t           numThreads,
                                               uint32_t           ldsNumElements,
                                               int32_t            initValue = 0,
                                               const std::string& comment   = "");

    private:
        std::unique_ptr<StinkyIRImpl> pImpl;
    };

    /**
     * @brief Helper class for loading kernel arguments
     * 
     * ArgumentLoader manages the state and offset tracking needed to load
     * kernel arguments from memory into SGPRs using SLoadBX instructions.
     * It automatically selects the appropriate instruction size (32/64/128/256/512)
     * and advances the offset after each load.
     * 
     * Usage:
     *   StinkyAsmIR st([9, 4, 2]);
     *   auto module = st.createIRList("kernel");
     *   ArgumentLoader loader(st);
     *   
     *   // Load a 32-bit argument
     *   auto insts = loader.loadKernArg(0, 2, 1, true);  // dst=s0, src=s[2:3], dword=1
     *   module.add(insts);
     *   
     *   // Load a 64-bit argument (2 dwords)
     *   insts = loader.loadKernArg(1, 2, 2, true);       // dst=s[1:2], dword=2
     *   module.add(insts);
     */
    class ArgumentLoader
    {
    public:
        /**
         * @brief Construct an ArgumentLoader
         * @param builder StinkyTofu builder that creates and owns instructions
         */
        explicit ArgumentLoader(StinkyTofu& builder);

        /**
         * @brief Reset the kernel argument offset to 0
         */
        void resetOffset();

        /**
         * @brief Set the kernel argument offset
         * @param offset New offset value in bytes
         */
        void setOffset(int offset);

        /**
         * @brief Get the current kernel argument offset
         * @return Current offset in bytes
         */
        int getOffset() const;

        /**
         * @brief Load a kernel argument using SLoadBX instruction
         * 
         * Generates an SLoadBX instruction (B32/B64/B128/B256/B512) based on
         * the dword count. The instruction loads from kernarg memory at
         * [srcAddr + kernArgOffset] into the destination SGPR(s).
         * 
         * After loading, kernArgOffset is automatically advanced by (dword * 4)
         * bytes, unless sgprOffset is provided.
         * 
         * @param dstSgpr Destination SGPR index
         * @param srcAddr Source address SGPR pair index (64-bit pointer)
         * @param dword Number of dwords to load (1, 2, 4, 8, 16)
         * @param writeSgpr If false, only advance offset without generating instruction
         * @param sgprOffset Optional explicit offset (if provided, kernArgOffset is not auto-advanced)
         * @param comment Optional comment (defaults to showing current offset)
         * @return Vector of instructions (empty if writeSgpr=false)
         */
        std::vector<StinkyInstruction*> loadKernArg(uint32_t           dstSgpr,
                                                    uint32_t           srcAddr,
                                                    int                dword,
                                                    bool               writeSgpr  = true,
                                                    std::optional<int> sgprOffset = std::nullopt,
                                                    const std::string& comment    = "");

        /**
         * @brief Load all kernel arguments efficiently
         * 
         * Loads a contiguous range of SGPRs from kernel arguments, automatically
         * selecting the largest possible SLoadBX instructions (B512 -> B256 -> B128 -> B64 -> B32)
         * based on SGPR alignment and remaining count.
         * 
         * This minimizes the number of load instructions by using wider loads when possible.
         * 
         * @param sgprStartIndex Starting SGPR index to load into
         * @param srcAddr Source address SGPR pair index (64-bit pointer)
         * @param numSgprToLoad Total number of SGPRs to load
         * @param numSgprPreload Number of SGPRs already preloaded (skip these)
         * @return Vector of instructions
         */
        std::vector<StinkyInstruction*> loadAllKernArg(uint32_t sgprStartIndex,
                                                       uint32_t srcAddr,
                                                       int      numSgprToLoad,
                                                       int      numSgprPreload = 0);

    private:
        StinkyTofu& builder;
        int         kernArgOffset;
    };

    // Backward compatibility: StinkyTofu is now an alias for StinkyAsmIR
    // StinkyAsmIR is defined in StinkyTofu.hpp
    // Users can still use "StinkyTofu" name in their code

} // namespace stinkytofu
