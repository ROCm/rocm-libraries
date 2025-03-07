/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/CodeGen/SubInstruction.hpp>

namespace rocRoller
{
    Instruction ScalarSubInt32(ContextPtr         ctx,
                               Register::ValuePtr dest,
                               Register::ValuePtr lhs,
                               Register::ValuePtr rhs,
                               std::string        comment)
    {
        auto gpu = ctx->targetArchitecture().target();
        if(gpu.isCDNAGPU() || (gpu.isRDNAGPU() && gpu.gfx < GPUArchitectureGFX::GFX1200))
        {
            return Instruction("s_sub_i32", {dest}, {lhs, rhs}, {}, comment);
        }
        else if(gpu.isRDNA4GPU())
        {
            return Instruction("s_sub_co_i32", {dest}, {lhs, rhs}, {}, comment);
        }
        else
        {
            Throw<FatalError>(
                std::format("ScalarSubInt32 not implemented for {}.", gpu.toString()));
        }
    }

    Instruction ScalarSubUInt32(ContextPtr         ctx,
                                Register::ValuePtr dest,
                                Register::ValuePtr lhs,
                                Register::ValuePtr rhs,
                                std::string        comment)
    {
        auto gpu = ctx->targetArchitecture().target();
        if(gpu.isCDNAGPU() || (gpu.isRDNAGPU() && gpu.gfx < GPUArchitectureGFX::GFX1200))
        {
            return Instruction("s_sub_u32", {dest}, {lhs, rhs}, {}, comment);
        }
        else if(gpu.isRDNA4GPU())
        {
            return Instruction("s_sub_co_u32", {dest}, {lhs, rhs}, {}, comment);
        }
        else
        {
            Throw<FatalError>(
                std::format("ScalarSubUInt32 not implemented for {}\n", gpu.toString()));
        }
    }

    Instruction ScalarSubUInt32CarryInOut(ContextPtr         ctx,
                                          Register::ValuePtr dest,
                                          Register::ValuePtr lhs,
                                          Register::ValuePtr rhs,
                                          std::string        comment)
    {
        auto gpu = ctx->targetArchitecture().target();
        if(gpu.isCDNAGPU() || (gpu.isRDNAGPU() && gpu.gfx < GPUArchitectureGFX::GFX1200))
        {
            return Instruction("s_subb_u32", {dest}, {lhs, rhs}, {}, comment);
        }
        else if(gpu.isRDNA4GPU())
        {
            return Instruction("s_sub_co_ci_u32", {dest}, {lhs, rhs}, {}, comment);
        }
        else
        {
            Throw<FatalError>(
                std::format("ScalarSubUInt32CarryInOut not implemented for {}\n", gpu.toString()));
        }
    }

    Instruction VectorSubUInt32(ContextPtr         ctx,
                                Register::ValuePtr dest,
                                Register::ValuePtr lhs,
                                Register::ValuePtr rhs,
                                std::string        comment)
    {
        auto gpu = ctx->targetArchitecture().target();
        if(gpu.isCDNAGPU())
        {
            return Instruction("v_sub_u32", {dest}, {lhs, rhs}, {}, comment);
        }
        else if(gpu.isRDNAGPU())
        {
            return Instruction("v_sub_nc_u32", {dest}, {lhs, rhs}, {}, comment);
        }
        else
        {
            Throw<FatalError>(
                std::format("VectorSubUInt32 not implemented for {}\n", gpu.toString()));
        }
    }

    Instruction VectorSubUInt32CarryOut(ContextPtr         ctx,
                                        Register::ValuePtr dest,
                                        Register::ValuePtr carry,
                                        Register::ValuePtr lhs,
                                        Register::ValuePtr rhs,
                                        std::string        comment)
    {
        auto gpu = ctx->targetArchitecture().target();
        if(gpu.isCDNAGPU() || gpu.isRDNAGPU())
        {
            return Instruction("v_sub_co_u32", {dest, carry}, {lhs, rhs}, {}, comment);
        }
        else
        {
            Throw<FatalError>(
                std::format("VectorSubUInt32CarryOut not implemented for {}\n", gpu.toString()));
        }
    }

    Instruction VectorSubUInt32CarryOut(ContextPtr         ctx,
                                        Register::ValuePtr dest,
                                        Register::ValuePtr lhs,
                                        Register::ValuePtr rhs,
                                        std::string        comment)
    {
        return VectorSubUInt32CarryOut(ctx, dest, ctx->getVCC(), lhs, rhs, comment);
    }

    Instruction VectorSubUInt32CarryInOut(ContextPtr         ctx,
                                          Register::ValuePtr dest,
                                          Register::ValuePtr carryOut,
                                          Register::ValuePtr carryIn,
                                          Register::ValuePtr lhs,
                                          Register::ValuePtr rhs,
                                          std::string        comment)
    {
        auto gpu = ctx->targetArchitecture().target();
        if(gpu.isCDNAGPU())
        {
            return Instruction("v_subb_co_u32", {dest, carryOut}, {lhs, rhs, carryIn}, {}, comment);
        }
        else if(gpu.isRDNAGPU())
        {
            return Instruction(
                "v_sub_co_ci_u32", {dest, carryOut}, {lhs, rhs, carryIn}, {}, comment);
        }
        else
        {
            Throw<FatalError>(
                std::format("VectorSubUInt32CarryInOut not implemented for {}\n", gpu.toString()));
        }
    }

    Instruction VectorSubUInt32CarryInOut(ContextPtr         ctx,
                                          Register::ValuePtr dest,
                                          Register::ValuePtr lhs,
                                          Register::ValuePtr rhs,
                                          std::string        comment)
    {
        auto vcc = ctx->getVCC();
        return VectorSubUInt32CarryInOut(ctx, dest, vcc, vcc, lhs, rhs, comment);
    }

    Instruction VectorSubRevUInt32(ContextPtr         ctx,
                                   Register::ValuePtr dest,
                                   Register::ValuePtr lhs,
                                   Register::ValuePtr rhs,
                                   std::string        comment)
    {
        auto gpu = ctx->targetArchitecture().target();
        if(gpu.isCDNAGPU())
        {
            return Instruction("v_subrev_u32", {dest}, {lhs, rhs}, {}, comment);
        }
        else if(gpu.isRDNAGPU())
        {
            return Instruction("v_subrev_nc_u32", {dest}, {lhs, rhs}, {}, comment);
        }
        else
        {
            Throw<FatalError>(
                std::format("VectorSubRevUInt32 not implemented for {}\n", gpu.toString()));
        }
    }

    Instruction VectorSubRevUInt32CarryOut(ContextPtr         ctx,
                                           Register::ValuePtr dest,
                                           Register::ValuePtr carryOut,
                                           Register::ValuePtr lhs,
                                           Register::ValuePtr rhs,
                                           std::string        comment)
    {

        auto gpu = ctx->targetArchitecture().target();
        if(gpu.isCDNAGPU() || gpu.isRDNAGPU())
        {
            return Instruction("v_subrev_co_u32", {dest, carryOut}, {lhs, rhs}, {}, comment);
        }
        else
        {
            Throw<FatalError>(
                std::format("VectorSubRevUInt32CarryOut not implemented for {}\n", gpu.toString()));
        }
    }

    Instruction VectorSubRevUInt32CarryOut(ContextPtr         ctx,
                                           Register::ValuePtr dest,
                                           Register::ValuePtr lhs,
                                           Register::ValuePtr rhs,
                                           std::string        comment)
    {

        auto vcc = ctx->getVCC();
        return VectorSubRevUInt32CarryOut(ctx, dest, vcc, lhs, rhs, comment);
    }

    Instruction VectorSubRevUInt32CarryInOut(ContextPtr         ctx,
                                             Register::ValuePtr dest,
                                             Register::ValuePtr carryOut,
                                             Register::ValuePtr carryIn,
                                             Register::ValuePtr lhs,
                                             Register::ValuePtr rhs,
                                             std::string        comment)
    {

        auto gpu = ctx->targetArchitecture().target();
        if(gpu.isCDNAGPU())
        {
            return Instruction(
                "v_subbrev_co_u32", {dest, carryOut}, {lhs, rhs, carryIn}, {}, comment);
        }
        else if(gpu.isRDNAGPU())
        {
            return Instruction(
                "v_subrev_co_ci_u32", {dest, carryOut}, {lhs, rhs, carryIn}, {}, comment);
        }
        else
        {
            Throw<FatalError>(std::format("VectorSubRevUInt32CarryInOut not implemented for {}\n",
                                          gpu.toString()));
        }
    }

    Instruction VectorSubRevUInt32CarryInOut(ContextPtr         ctx,
                                             Register::ValuePtr dest,
                                             Register::ValuePtr lhs,
                                             Register::ValuePtr rhs,
                                             std::string        comment)
    {
        auto vcc = ctx->getVCC();
        return VectorSubRevUInt32CarryInOut(ctx, dest, vcc, vcc, lhs, rhs, comment);
    }
}
