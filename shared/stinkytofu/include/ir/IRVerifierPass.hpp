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

#include "stinkytofu.hpp"
#include <memory>
#include <string>
#include <vector>

namespace stinkytofu
{
    // Forward declarations
    class LogicalInstruction;
    class StinkyInstruction;

    /// Configuration for IR verification
    struct IRVerifierConfig
    {
        bool abortOnError        = true; ///< Abort on first error
        bool verbose             = false; ///< Print detailed messages
        bool checkRegisterWidths = true; ///< Verify register widths match instruction requirements
        bool checkRegisterRanges
            = false; ///< Verify register indices are within valid ranges (requires arch info)
    };

    // ===========================================================================
    // Standalone Validation Functions
    // ===========================================================================

    /// Validate Logical IR structure
    /// Returns error message if invalid, empty string if valid
    std::string validateLogicalIR(Function& func, const IRVerifierConfig& config = {});

    /// Validate StinkyTofu Assembly IR structure
    /// Returns error message if invalid, empty string if valid
    std::string validateStinkyIR(Function& func, const IRVerifierConfig& config = {});

    // ===========================================================================
    // Thin Pass Wrappers (for pipeline integration)
    // ===========================================================================

    /// Pass that verifies Logical IR (thin wrapper around validateLogicalIR)
    class LogicalIRVerifierPass : public Pass
    {
    public:
        static char ID;

        explicit LogicalIRVerifierPass(IRVerifierConfig config = {})
            : config_(std::move(config))
        {
        }

        PassID getPassID() const override
        {
            return &ID;
        }
        const char* getName() const override
        {
            return "LogicalIRVerifier";
        }

        void run(Function& func, PassContext& ctx) override;

    private:
        IRVerifierConfig config_;
    };

    /// Pass that verifies StinkyTofu Assembly IR (thin wrapper around validateStinkyIR)
    class StinkyIRVerifierPass : public Pass
    {
    public:
        static char ID;

        explicit StinkyIRVerifierPass(IRVerifierConfig config = {})
            : config_(std::move(config))
        {
        }

        PassID getPassID() const override
        {
            return &ID;
        }
        const char* getName() const override
        {
            return "StinkyIRVerifier";
        }

        void run(Function& func, PassContext& ctx) override;

    private:
        IRVerifierConfig config_;
    };

    /// Factory functions for creating verifier passes

    /// Create a Logical IR verifier pass
    inline std::unique_ptr<Pass> createLogicalIRVerifierPass(IRVerifierConfig config = {})
    {
        return std::make_unique<LogicalIRVerifierPass>(std::move(config));
    }

    /// Create a StinkyTofu Assembly IR verifier pass
    inline std::unique_ptr<Pass> createStinkyIRVerifierPass(IRVerifierConfig config = {})
    {
        return std::make_unique<StinkyIRVerifierPass>(std::move(config));
    }

} // namespace stinkytofu
