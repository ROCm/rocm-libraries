// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/SubInstruction.hpp>

namespace rocRoller
{
    Instruction ScalarSubInt32(ContextPtr         ctx,
                               Register::ValuePtr dest,
                               Register::ValuePtr lhs,
                               Register::ValuePtr rhs,
                               std::string        comment)
    {
        auto const& arch = ctx->targetArchitecture();
        if(arch.HasCapability(GPUCapability::HasExplicitScalarCO))
        {
            return Instruction("s_sub_co_i32", {dest}, {lhs, rhs}, {}, comment);
        }
        else
        {
            return Instruction("s_sub_i32", {dest}, {lhs, rhs}, {}, comment);
        }
    }

    Instruction ScalarSubUInt32(ContextPtr         ctx,
                                Register::ValuePtr dest,
                                Register::ValuePtr lhs,
                                Register::ValuePtr rhs,
                                std::string        comment)
    {
        auto const& arch = ctx->targetArchitecture();
        if(arch.HasCapability(GPUCapability::HasExplicitScalarCO))
        {
            return Instruction("s_sub_co_u32", {dest}, {lhs, rhs}, {}, comment);
        }
        else
        {
            return Instruction("s_sub_u32", {dest}, {lhs, rhs}, {}, comment);
        }
    }

    Instruction ScalarSubUInt32CarryInOut(ContextPtr         ctx,
                                          Register::ValuePtr dest,
                                          Register::ValuePtr lhs,
                                          Register::ValuePtr rhs,
                                          std::string        comment)
    {
        auto const& arch = ctx->targetArchitecture();
        if(arch.HasCapability(GPUCapability::HasExplicitScalarCOCI))
        {
            return Instruction("s_sub_co_ci_u32", {dest}, {lhs, rhs}, {}, comment);
        }
        else
        {
            return Instruction("s_subb_u32", {dest}, {lhs, rhs}, {}, comment);
        }
    }

    Instruction VectorSubUInt32(ContextPtr         ctx,
                                Register::ValuePtr dest,
                                Register::ValuePtr lhs,
                                Register::ValuePtr rhs,
                                std::string        comment)
    {
        auto const& arch = ctx->targetArchitecture();
        if(arch.HasCapability(GPUCapability::HasExplicitNC))
        {
            return Instruction("v_sub_nc_u32", {dest}, {lhs, rhs}, {}, comment);
        }
        else
        {
            return Instruction("v_sub_u32", {dest}, {lhs, rhs}, {}, comment);
        }
    }

    Instruction VectorSubUInt32CarryOut(ContextPtr         ctx,
                                        Register::ValuePtr dest,
                                        Register::ValuePtr carry,
                                        Register::ValuePtr lhs,
                                        Register::ValuePtr rhs,
                                        std::string        comment)
    {
        auto const& arch = ctx->targetArchitecture();
        if(arch.HasCapability(GPUCapability::HasExplicitVectorCO))
        {
            return Instruction("v_sub_co_u32", {dest, carry}, {lhs, rhs}, {}, comment);
        }
        else
        {
            Throw<FatalError>(fmt::format("VectorSubUInt32CarryOut not implemented for {}\n",
                                          arch.target().toString()));
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
        auto const& arch = ctx->targetArchitecture();
        if(arch.HasCapability(GPUCapability::HasExplicitVectorCOCI))
        {
            return Instruction(
                "v_sub_co_ci_u32", {dest, carryOut}, {lhs, rhs, carryIn}, {}, comment);
        }
        else
        {
            return Instruction("v_subb_co_u32", {dest, carryOut}, {lhs, rhs, carryIn}, {}, comment);
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
        auto const& arch = ctx->targetArchitecture();
        if(arch.HasCapability(GPUCapability::HasExplicitVectorRevNC))
        {
            return Instruction("v_subrev_nc_u32", {dest}, {lhs, rhs}, {}, comment);
        }
        else
        {
            return Instruction("v_subrev_u32", {dest}, {lhs, rhs}, {}, comment);
        }
    }

    Instruction VectorSubRevUInt32CarryOut(ContextPtr         ctx,
                                           Register::ValuePtr dest,
                                           Register::ValuePtr carryOut,
                                           Register::ValuePtr lhs,
                                           Register::ValuePtr rhs,
                                           std::string        comment)
    {

        auto const& arch = ctx->targetArchitecture();
        if(arch.HasCapability(GPUCapability::HasExplicitVectorRevCO))
        {
            return Instruction("v_subrev_co_u32", {dest, carryOut}, {lhs, rhs}, {}, comment);
        }
        else
        {
            Throw<FatalError>(fmt::format("VectorSubRevUInt32CarryOut not implemented for {}\n",
                                          arch.target().toString()));
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

        auto const& arch = ctx->targetArchitecture();
        if(arch.HasCapability(GPUCapability::HasExplicitVectorRevCOCI))
        {
            return Instruction(
                "v_subrev_co_ci_u32", {dest, carryOut}, {lhs, rhs, carryIn}, {}, comment);
        }
        else
        {
            return Instruction(
                "v_subbrev_co_u32", {dest, carryOut}, {lhs, rhs, carryIn}, {}, comment);
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
