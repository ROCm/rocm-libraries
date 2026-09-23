// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/AddInstruction.hpp>

namespace rocRoller
{
    Instruction ScalarAddInt32(ContextPtr         ctx,
                               Register::ValuePtr dest,
                               Register::ValuePtr lhs,
                               Register::ValuePtr rhs,
                               std::string        comment)
    {
        auto const& arch = ctx->targetArchitecture();
        if(arch.HasCapability(GPUCapability::HasExplicitScalarCO))
        {
            return Instruction("s_add_co_i32", {dest}, {lhs, rhs}, {}, comment);
        }
        else
        {
            return Instruction("s_add_i32", {dest}, {lhs, rhs}, {}, comment);
        }
    }

    Instruction ScalarAddUInt32(ContextPtr         ctx,
                                Register::ValuePtr dest,
                                Register::ValuePtr lhs,
                                Register::ValuePtr rhs,
                                std::string        comment)
    {
        auto const& arch = ctx->targetArchitecture();
        if(arch.HasCapability(GPUCapability::HasExplicitScalarCO))
        {
            return Instruction("s_add_co_u32", {dest}, {lhs, rhs}, {}, comment);
        }
        else
        {
            return Instruction("s_add_u32", {dest}, {lhs, rhs}, {}, comment);
        }
    }

    Instruction ScalarAddUInt32CarryInOut(ContextPtr         ctx,
                                          Register::ValuePtr dest,
                                          Register::ValuePtr lhs,
                                          Register::ValuePtr rhs,
                                          std::string        comment)
    {
        auto const& arch = ctx->targetArchitecture();
        if(arch.HasCapability(GPUCapability::HasExplicitScalarCOCI))
        {
            return Instruction("s_add_co_ci_u32", {dest}, {lhs, rhs}, {}, comment);
        }
        else
        {
            return Instruction("s_addc_u32", {dest}, {lhs, rhs}, {}, comment);
        }
    }

    Instruction VectorAddUInt32(ContextPtr         ctx,
                                Register::ValuePtr dest,
                                Register::ValuePtr lhs,
                                Register::ValuePtr rhs,
                                std::string        comment)
    {
        auto const& arch = ctx->targetArchitecture();
        if(arch.HasCapability(GPUCapability::HasExplicitNC))
        {
            return Instruction("v_add_nc_u32", {dest}, {lhs, rhs}, {}, comment);
        }
        else
        {
            return Instruction("v_add_u32", {dest}, {lhs, rhs}, {}, comment);
        }
    }

    Instruction VectorAddUInt32CarryOut(ContextPtr         ctx,
                                        Register::ValuePtr dest,
                                        Register::ValuePtr carry,
                                        Register::ValuePtr lhs,
                                        Register::ValuePtr rhs,
                                        std::string        comment)
    {
        auto const& arch = ctx->targetArchitecture();
        if(arch.HasCapability(GPUCapability::HasExplicitVectorCO))
        {
            return Instruction("v_add_co_u32", {dest, carry}, {lhs, rhs}, {}, comment);
        }
        else
        {
            Throw<FatalError>(fmt::format("VectorAddUInt32CarryOut not implemented for {}.\n",
                                          arch.target().toString()));
        }
    }

    Instruction VectorAddUInt32CarryOut(ContextPtr         ctx,
                                        Register::ValuePtr dest,
                                        Register::ValuePtr lhs,
                                        Register::ValuePtr rhs,
                                        std::string        comment)
    {
        return VectorAddUInt32CarryOut(ctx, dest, ctx->getVCC(), lhs, rhs);
    }

    Instruction VectorAddUInt32CarryInOut(ContextPtr         ctx,
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
                "v_add_co_ci_u32", {dest, carryOut}, {lhs, rhs, carryIn}, {}, comment);
        }
        else
        {
            return Instruction("v_addc_co_u32", {dest, carryOut}, {lhs, rhs, carryIn}, {}, comment);
        }
    }

    Instruction VectorAddUInt32CarryInOut(ContextPtr         ctx,
                                          Register::ValuePtr dest,
                                          Register::ValuePtr lhs,
                                          Register::ValuePtr rhs,
                                          std::string        comment)
    {
        auto vcc = ctx->getVCC();
        return VectorAddUInt32CarryInOut(ctx, dest, vcc, vcc, lhs, rhs);
    }

    Instruction VectorAdd3UInt32(ContextPtr         ctx,
                                 Register::ValuePtr dest,
                                 Register::ValuePtr lhs,
                                 Register::ValuePtr rhs1,
                                 Register::ValuePtr rhs2,
                                 std::string        comment)
    {
        auto const& arch = ctx->targetArchitecture();
        if(arch.HasCapability(GPUCapability::v_add3_u32))
        {
            return Instruction("v_add3_u32", {dest}, {lhs, rhs1, rhs2}, {}, comment);
        }
        else
        {
            Throw<FatalError>(fmt::format("VectorAdd3UInt32 not implemented for {}.\n",
                                          arch.target().toString()));
        }
    }
}
