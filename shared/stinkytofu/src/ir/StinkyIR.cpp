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

#include "ir/StinkyIR.hpp"
#include "StinkyBuilder.hpp"
#include <cmath>

namespace stinkytofu
{
    // Anonymous namespace for local helper functions
    namespace
    {
        inline StinkyRegister imm(int32_t value)
        {
            return StinkyRegister(value);
        }

        inline StinkyRegister imm(double value)
        {
            return StinkyRegister(value);
        }

        inline StinkyRegister vcc()
        {
            return StinkyRegister("vcc", 0, 1);
        }
    } // anonymous namespace

    // Implementation class stores only architecture info
    class StinkyIRImpl
    {
    public:
        std::array<int, 3> arch;

        explicit StinkyIRImpl(std::array<int, 3> arch)
            : arch(arch)
        {
        }
    };

    StinkyIR::StinkyIR(std::array<int, 3> arch)
        : pImpl(std::make_unique<StinkyIRImpl>(arch))
    {
    }

    StinkyIR::~StinkyIR()                              = default;
    StinkyIR::StinkyIR(StinkyIR&&) noexcept            = default;
    StinkyIR& StinkyIR::operator=(StinkyIR&&) noexcept = default;

    std::array<int, 3> StinkyIR::getArch() const
    {
        return pImpl->arch;
    }

    // ============================================================================
    // Division Functions
    // ============================================================================

    Expected<std::vector<StinkyInstruction*>>
        StinkyIR::vectorStaticDivideAndRemainder(StinkyTofu&                  builder,
                                                 uint32_t                     qReg,
                                                 uint32_t                     rReg,
                                                 uint32_t                     dReg,
                                                 int                          divisor,
                                                 const std::vector<uint32_t>& tmpVgpr,
                                                 bool                         doRemainder,
                                                 const std::string&           comment)
    {
        std::vector<StinkyInstruction*> result;
        StinkyTofu&                     st = builder;

        std::string dComment = comment.empty()
                                   ? "v" + std::to_string(qReg) + " = v" + std::to_string(dReg)
                                         + " / " + std::to_string(divisor)
                                   : comment;
        std::string rComment = comment.empty()
                                   ? "v" + std::to_string(rReg) + " = v" + std::to_string(dReg)
                                         + " % " + std::to_string(divisor)
                                   : comment;

        // Check if divisor is power of 2
        if((divisor & (divisor - 1)) == 0)
        {
            // Power of 2: use simple shift
            int divisor_log2 = static_cast<int>(std::log2(divisor));

            auto inst = st.VLShiftRightB32(vgpr(qReg), imm(divisor_log2), vgpr(dReg), dComment);
            result.insert(result.end(), inst.begin(), inst.end());

            if(doRemainder)
            {
                auto rimst = st.VAndB32(vgpr(rReg), imm(divisor - 1), vgpr(dReg), rComment);
                result.insert(result.end(), rimst.begin(), rimst.end());
            }
        }
        else
        {
            // Non-power of 2: use magic number algorithm
            assert(tmpVgpr.size() >= 2 && "tmpVgpr must have at least 2 registers");
            if(tmpVgpr.size() < 2)
            {
                return Expected<std::vector<StinkyInstruction*>>::Error(
                    "vectorStaticDivideAndRemainder: tmpVgpr must have at least 2 registers");
            }

            uint32_t tmpVgpr0 = tmpVgpr[0];
            uint32_t tmpVgpr1 = tmpVgpr[1];

            int shift = 32 + 1;
            int magic = ((1ULL << shift) / divisor) + 1;

            // Multiply by magic number
            if(magic <= 64 && magic >= -16)
            {
                auto inst = st.VMulHIU32(vgpr(tmpVgpr1), vgpr(dReg), imm(magic), dComment);
                result.insert(result.end(), inst.begin(), inst.end());
                inst = st.VMulLOU32(vgpr(tmpVgpr0), vgpr(dReg), imm(magic), dComment);
                result.insert(result.end(), inst.begin(), inst.end());
            }
            else
            {
                auto inst = st.VMovB32(vgpr(tmpVgpr0), imm(magic));
                result.insert(result.end(), inst.begin(), inst.end());
                inst = st.VMulHIU32(vgpr(tmpVgpr1), vgpr(dReg), vgpr(tmpVgpr0), dComment);
                result.insert(result.end(), inst.begin(), inst.end());
                inst = st.VMulLOU32(vgpr(tmpVgpr0), vgpr(dReg), vgpr(tmpVgpr0), dComment);
                result.insert(result.end(), inst.begin(), inst.end());
            }

            // Shift right
            auto inst
                = st.VLShiftRightB64(vgpr(tmpVgpr0, 2), imm(shift), vgpr(tmpVgpr0, 2), dComment);
            result.insert(result.end(), inst.begin(), inst.end());

            // Move quotient to destination
            inst = st.VMovB32(vgpr(qReg), vgpr(tmpVgpr0), dComment);
            result.insert(result.end(), inst.begin(), inst.end());

            if(doRemainder)
            {
                // Calculate remainder: r = d - q * divisor
                if(divisor <= 64 && divisor >= -16)
                {
                    auto rinst = st.VMulLOU32(vgpr(tmpVgpr0), vgpr(qReg), imm(divisor), rComment);
                    result.insert(result.end(), rinst.begin(), rinst.end());
                }
                else
                {
                    auto rinst = st.VMovB32(vgpr(tmpVgpr1), imm(divisor));
                    result.insert(result.end(), rinst.begin(), rinst.end());
                    rinst = st.VMulLOU32(vgpr(tmpVgpr0), vgpr(qReg), vgpr(tmpVgpr1), rComment);
                    result.insert(result.end(), rinst.begin(), rinst.end());
                }
                auto rinst = st.VSubU32(vgpr(rReg), vgpr(dReg), vgpr(tmpVgpr0), rComment);
                result.insert(result.end(), rinst.begin(), rinst.end());
            }
        }

        return result;
    }

    Expected<std::vector<StinkyInstruction*>>
        StinkyIR::vectorStaticDivide(StinkyTofu&                  builder,
                                     uint32_t                     qReg,
                                     uint32_t                     dReg,
                                     int                          divisor,
                                     const std::vector<uint32_t>& tmpVgpr,
                                     const std::string&           comment)
    {
        // Call divide and remainder but skip remainder calculation
        return vectorStaticDivideAndRemainder(
            builder, qReg, 0, dReg, divisor, tmpVgpr, false, comment);
    }

    std::vector<StinkyInstruction*>
        StinkyIR::vectorUInt32DivideAndRemainder(StinkyTofu&        builder,
                                                 uint32_t           qReg,
                                                 uint32_t           dReg,
                                                 uint32_t           divReg,
                                                 uint32_t           rReg,
                                                 bool               doRemainder,
                                                 const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        StinkyTofu&                     st = builder;

        std::string dComment = comment.empty()
                                   ? "v" + std::to_string(qReg) + " = v" + std::to_string(dReg)
                                         + " / v" + std::to_string(divReg)
                                   : comment;
        std::string rComment = comment.empty()
                                   ? "v" + std::to_string(rReg) + " = v" + std::to_string(dReg)
                                         + " % v" + std::to_string(divReg)
                                   : comment;

        // Use FP32 reciprocal for dynamic division
        auto inst = st.VCvtU32toF32(vgpr(qReg), vgpr(divReg), dComment);
        result.insert(result.end(), inst.begin(), inst.end());

        inst = st.VRcpF32(vgpr(qReg), vgpr(qReg), dComment);
        result.insert(result.end(), inst.begin(), inst.end());

        inst = st.VCvtU32toF32(vgpr(rReg), vgpr(dReg), dComment);
        result.insert(result.end(), inst.begin(), inst.end());

        inst = st.VMulF32(vgpr(qReg), vgpr(qReg), vgpr(rReg), dComment);
        result.insert(result.end(), inst.begin(), inst.end());

        inst = st.VCvtF32toU32(vgpr(qReg), vgpr(qReg), dComment);
        result.insert(result.end(), inst.begin(), inst.end());

        inst = st.VMulLOU32(vgpr(rReg), vgpr(qReg), vgpr(divReg), dComment);
        result.insert(result.end(), inst.begin(), inst.end());

        inst = st.VSubU32(vgpr(rReg), vgpr(dReg), vgpr(rReg), dComment);
        result.insert(result.end(), inst.begin(), inst.end());

        // Check if rReg == divReg, if so increment quotient
        StinkyRegister execReg("exec", 0, 1);
        inst = st.VCmpXEqU32(execReg, vgpr(rReg), vgpr(divReg), dComment);
        result.insert(result.end(), inst.begin(), inst.end());

        inst = st.VAddU32(vgpr(qReg), imm(1), vgpr(qReg), dComment);
        result.insert(result.end(), inst.begin(), inst.end());

        if(doRemainder)
        {
            inst = st.VMovB32(vgpr(rReg), imm(0), rComment);
            result.insert(result.end(), inst.begin(), inst.end());
        }

        // Reset EXEC mask
        inst = st.SMovB64(execReg, imm(-1), dComment);
        result.insert(result.end(), inst.begin(), inst.end());

        return result;
    }

    Expected<std::vector<StinkyInstruction*>>
        StinkyIR::scalarStaticDivideAndRemainder(StinkyTofu&                  builder,
                                                 uint32_t                     qReg,
                                                 uint32_t                     rReg,
                                                 uint32_t                     dReg,
                                                 int                          divisor,
                                                 const std::vector<uint32_t>& tmpSgpr,
                                                 int                          doRemainder,
                                                 const std::string&           comment)
    {
        std::vector<StinkyInstruction*> result;
        StinkyTofu&                     st = builder;

        // Similar to vector version but using scalar instructions
        if((divisor & (divisor - 1)) == 0)
        {
            // Power of 2
            int divisor_log2 = static_cast<int>(std::log2(divisor));

            if(doRemainder != 2)
            {
                auto inst = st.SLShiftRightB32(sgpr(qReg), imm(divisor_log2), sgpr(dReg), comment);
                result.insert(result.end(), inst.begin(), inst.end());
            }

            if(doRemainder)
            {
                auto inst = st.SAndB32(sgpr(rReg), imm(divisor - 1), sgpr(dReg), comment);
                result.insert(result.end(), inst.begin(), inst.end());
            }
        }
        else
        {
            assert(tmpSgpr.size() >= 2 && "tmpSgpr must have at least 2 registers");
            if(tmpSgpr.size() < 2)
            {
                return Expected<std::vector<StinkyInstruction*>>::Error(
                    "scalarStaticDivideAndRemainder: tmpSgpr must have at least 2 registers");
            }

            uint32_t tmpSgpr0 = tmpSgpr[0];
            uint32_t tmpSgpr1 = tmpSgpr[1];

            int shift   = 32 + 1;
            int magic   = ((1ULL << shift) / divisor) + 1;
            int magicHi = magic >> 16;
            int magicLo = magic & 0xFFFF;

            auto inst = st.SMovB32(
                sgpr(tmpSgpr1), imm(0), "STATIC_DIV: divisor=" + std::to_string(divisor));
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.SMulI32(
                sgpr(tmpSgpr0), imm(magicHi), sgpr(dReg), "tmp1 = dividend * magic hi");
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.SLShiftLeftB64(
                sgpr(tmpSgpr0, 2), imm(16), sgpr(tmpSgpr0, 2), "left shift 16 bits");
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.SMulI32(sgpr(qReg), sgpr(dReg), imm(magicLo), "tmp0 = dividend * magic lo");
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.SAddU32(sgpr(tmpSgpr0), sgpr(qReg), sgpr(tmpSgpr0), "add lo");
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.SAddCU32(sgpr(tmpSgpr1), sgpr(tmpSgpr1), imm(0), "add hi");
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.SLShiftRightB64(sgpr(tmpSgpr0, 2),
                                      imm(shift),
                                      sgpr(tmpSgpr0, 2),
                                      "tmp1 = (dividend * magic) << shift");
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.SMovB32(sgpr(qReg), sgpr(tmpSgpr0), "quotient");
            result.insert(result.end(), inst.begin(), inst.end());

            if(doRemainder)
            {
                inst = st.SMulI32(sgpr(tmpSgpr0), sgpr(qReg), imm(divisor), "quotient*divisor");
                result.insert(result.end(), inst.begin(), inst.end());

                inst = st.SSubU32(
                    sgpr(rReg), sgpr(dReg), sgpr(tmpSgpr0), "rReg = dividend - quotient*divisor");
                result.insert(result.end(), inst.begin(), inst.end());
            }
        }

        return result;
    }

    // ============================================================================
    // Multiplication Functions
    // ============================================================================

    Expected<std::vector<StinkyInstruction*>>
        StinkyIR::vectorStaticMultiply(StinkyTofu&                  builder,
                                       uint32_t                     productReg,
                                       uint32_t                     operandReg,
                                       int                          multiplier,
                                       const std::vector<uint32_t>& tmpSgpr,
                                       const std::string&           comment)
    {
        std::vector<StinkyInstruction*> result;
        StinkyTofu&                     st = builder;

        // TODO: Implement optimized multiplication using shifts/adds
        // For now, simple implementation
        if(multiplier <= 64 && multiplier >= -16)
        {
            auto inst = st.VMulLOU32(vgpr(productReg), imm(multiplier), vgpr(operandReg), comment);
            result.insert(result.end(), inst.begin(), inst.end());
        }
        else
        {
            assert(!tmpSgpr.empty() && "tmpSgpr required for large multipliers");
            if(tmpSgpr.empty())
            {
                return Expected<std::vector<StinkyInstruction*>>::Error(
                    "vectorStaticMultiply: tmpSgpr required for large multipliers");
            }
            auto inst = st.SMovB32(sgpr(tmpSgpr[0]), imm(multiplier), comment);
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.VMulLOU32(vgpr(productReg), sgpr(tmpSgpr[0]), vgpr(operandReg), comment);
            result.insert(result.end(), inst.begin(), inst.end());
        }

        return result;
    }

    std::vector<StinkyInstruction*> StinkyIR::vectorMultiplyBpe(StinkyTofu&        builder,
                                                                uint32_t           dstReg,
                                                                uint32_t           srcReg,
                                                                float              bpe,
                                                                const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        StinkyTofu&                     st = builder;

        std::string mcomment = comment + " (multiply bpe)";

        if(bpe == 0.5f)
        {
            auto inst = st.VLShiftRightB32(vgpr(dstReg), imm(1), vgpr(srcReg), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());
        }
        else if(bpe == 0.75f)
        {
            auto inst = st.VMulLOU32(vgpr(dstReg), imm(6), vgpr(srcReg), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.VLShiftRightB32(vgpr(dstReg), imm(3), vgpr(dstReg), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());
        }
        else
        {
            int bpe_log2 = static_cast<int>(std::log2(bpe));
            if(bpe_log2 == 0 && dstReg == srcReg)
            {
                // bpe is 1, do nothing (just add comment)
                // No instruction needed
            }
            else
            {
                auto inst = st.VLShiftLeftB32(vgpr(dstReg), imm(bpe_log2), vgpr(srcReg), mcomment);
                result.insert(result.end(), inst.begin(), inst.end());
            }
        }

        return result;
    }

    std::vector<StinkyInstruction*> StinkyIR::vectorMultiply64Bpe(StinkyTofu&        builder,
                                                                  uint32_t           dstReg,
                                                                  uint32_t           srcReg,
                                                                  float              bpe,
                                                                  uint32_t           tmpReg,
                                                                  const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        StinkyTofu&                     st = builder;

        std::string mcomment = comment + " (multiply bpe 64)";

        if(bpe == 0.5f)
        {
            auto inst = st.VLShiftRightB64(vgpr(dstReg, 2), imm(1), vgpr(srcReg, 2), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());
        }
        else if(bpe == 0.75f)
        {
            auto inst = st.VMovB32(vgpr(tmpReg), vgpr(srcReg + 1), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.VMulHIU32(vgpr(dstReg + 1), imm(6), vgpr(srcReg), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.VMulLOU32(vgpr(dstReg), imm(6), vgpr(srcReg), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.VMulLOU32(vgpr(tmpReg), imm(6), vgpr(tmpReg), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.VAddU32(vgpr(dstReg + 1), vgpr(dstReg + 1), vgpr(tmpReg), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.VLShiftRightB64(vgpr(dstReg, 2), imm(3), vgpr(dstReg, 2), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());
        }
        else
        {
            int bpe_log2 = static_cast<int>(std::log2(bpe));
            if(bpe_log2 == 0 && dstReg == srcReg)
            {
                // bpe is 1, do nothing
            }
            else
            {
                auto inst
                    = st.VLShiftLeftB64(vgpr(dstReg, 2), imm(bpe_log2), vgpr(srcReg, 2), mcomment);
                result.insert(result.end(), inst.begin(), inst.end());
            }
        }

        return result;
    }

    std::vector<StinkyInstruction*> StinkyIR::scalarMultiplyBpe(StinkyTofu&        builder,
                                                                uint32_t           dstReg,
                                                                uint32_t           srcReg,
                                                                float              bpe,
                                                                const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        StinkyTofu&                     st = builder;

        std::string mcomment = comment + " (multiply bpe)";

        if(bpe == 0.5f)
        {
            auto inst = st.SLShiftRightB32(sgpr(dstReg), imm(1), sgpr(srcReg), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());
        }
        else if(bpe == 0.75f)
        {
            auto inst = st.SMulI32(sgpr(dstReg), imm(6), sgpr(srcReg), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.SLShiftRightB32(sgpr(dstReg), imm(3), sgpr(dstReg), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());
        }
        else
        {
            int bpe_log2 = static_cast<int>(std::log2(bpe));
            if(bpe_log2 == 0 && dstReg == srcReg)
            {
                // bpe is 1, do nothing
            }
            else
            {
                auto inst = st.SLShiftLeftB32(sgpr(dstReg), imm(bpe_log2), sgpr(srcReg), mcomment);
                result.insert(result.end(), inst.begin(), inst.end());
            }
        }

        return result;
    }

    // ============================================================================
    // Branching Functions
    // ============================================================================

    std::vector<StinkyInstruction*> StinkyIR::BranchIfZero(StinkyTofu&        builder,
                                                           uint32_t           sgprName,
                                                           uint32_t           tmpSgpr,
                                                           const std::string& label,
                                                           const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        StinkyTofu&                     st = builder;

        // Compare sgpr with 0
        auto inst = st.SCmpEQU32(sgpr(sgprName), imm(0), comment);
        result.insert(result.end(), inst.begin(), inst.end());

        // Branch if SCC == 1 (equal)
        inst = st.SCBranchSCC1(label, comment);
        result.insert(result.end(), inst.begin(), inst.end());

        return result;
    }

    std::vector<StinkyInstruction*> StinkyIR::BranchIfNotZero(StinkyTofu&        builder,
                                                              uint32_t           sgprName,
                                                              const std::string& label,
                                                              const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        StinkyTofu&                     st = builder;

        // Compare sgpr with 0
        auto inst = st.SCmpEQU32(sgpr(sgprName), imm(0), comment);
        result.insert(result.end(), inst.begin(), inst.end());

        // Branch if SCC == 0 (not equal)
        inst = st.SCBranchSCC0(label, comment);
        result.insert(result.end(), inst.begin(), inst.end());

        return result;
    }

    Expected<std::vector<StinkyInstruction*>>
        StinkyIR::BranchIfZeroTyped(StinkyTofu&        builder,
                                    uint32_t           sgprName,
                                    const std::string& dataType,
                                    uint32_t           tmpVgpr,
                                    const std::string& label,
                                    const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        StinkyTofu&                     st = builder;

        std::string mcomment
            = comment.empty() ? ("branch if s" + std::to_string(sgprName) + " == 0") : comment;

        if(dataType == "i32")
        {
            // Int32: s_cmp_eq_u32 + s_cbranch_scc1
            auto inst = st.SCmpEQU32(sgpr(sgprName), imm(0), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.SCBranchSCC1(label, mcomment);
            result.insert(result.end(), inst.begin(), inst.end());
        }
        else if(dataType == "i64")
        {
            // Int64: s_cmp_eq_u64 + s_cbranch_scc1
            auto inst = st.SCmpEQU64(sgpr(sgprName, 2), imm(0), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.SCBranchSCC1(label, mcomment);
            result.insert(result.end(), inst.begin(), inst.end());
        }
        else if(dataType == "f32")
        {
            // Float32: v_cmp_eq_f32 + s_cbranch_vccnz
            auto inst = st.VCmpEQF32(vcc(), sgpr(sgprName), imm(0.0f), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.SCBranchVCCNZ(label, mcomment);
            result.insert(result.end(), inst.begin(), inst.end());
        }
        else if(dataType == "f64")
        {
            // Float64: v_cmp_eq_f64 + s_cbranch_vccnz
            auto inst = st.VCmpEQF64(vcc(), sgpr(sgprName, 2), imm(0.0), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.SCBranchVCCNZ(label, mcomment);
            result.insert(result.end(), inst.begin(), inst.end());
        }
        else
        {
            return Expected<std::vector<StinkyInstruction*>>::Error(
                "Unsupported data type for BranchIfZeroTyped: " + dataType
                + ". Supported types: i32, i64, f32, f64");
        }

        return result;
    }

    Expected<std::vector<StinkyInstruction*>>
        StinkyIR::BranchIfNotZeroTyped(StinkyTofu&        builder,
                                       uint32_t           sgprName,
                                       const std::string& dataType,
                                       const std::string& label,
                                       const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        StinkyTofu&                     st = builder;

        std::string mcomment
            = comment.empty() ? ("branch if s" + std::to_string(sgprName) + " != 0") : comment;

        if(dataType == "i32")
        {
            // Int32: s_cmp_eq_u32 + s_cbranch_scc0
            auto inst = st.SCmpEQU32(sgpr(sgprName), imm(0), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.SCBranchSCC0(label, mcomment);
            result.insert(result.end(), inst.begin(), inst.end());
        }
        else if(dataType == "i64")
        {
            // Int64: s_cmp_eq_u64 + s_cbranch_scc0
            auto inst = st.SCmpEQU64(sgpr(sgprName, 2), imm(0), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.SCBranchSCC0(label, mcomment);
            result.insert(result.end(), inst.begin(), inst.end());
        }
        else if(dataType == "f32")
        {
            // Float32: v_cmp_eq_f32 + s_cbranch_vccz
            auto inst = st.VCmpEQF32(vcc(), sgpr(sgprName), imm(0.0f), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.SCBranchVCCZ(label, mcomment);
            result.insert(result.end(), inst.begin(), inst.end());
        }
        else if(dataType == "f64")
        {
            // Float64: v_cmp_eq_f64 + s_cbranch_vccz
            auto inst = st.VCmpEQF64(vcc(), sgpr(sgprName, 2), imm(0.0), mcomment);
            result.insert(result.end(), inst.begin(), inst.end());

            inst = st.SCBranchVCCZ(label, mcomment);
            result.insert(result.end(), inst.begin(), inst.end());
        }
        else
        {
            return Expected<std::vector<StinkyInstruction*>>::Error(
                "Unsupported data type for BranchIfNotZeroTyped: " + dataType
                + ". Supported types: i32, i64, f32, f64");
        }

        return result;
    }

    // ============================================================================
    // Casting Functions (f_cast.hpp)
    // ============================================================================

    Expected<std::vector<StinkyInstruction*>>
        StinkyIR::VSaturateCastInt(StinkyTofu&        builder,
                                   uint32_t           valueReg,
                                   uint32_t           tmpVgpr,
                                   uint32_t           tmpSgpr,
                                   int32_t            lowerBound,
                                   int32_t            upperBound,
                                   const std::string& saturateType,
                                   bool               initGpr,
                                   const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        StinkyTofu&                     st = builder;

        std::string mcomment = comment.empty() ? ("saturate cast: " + std::to_string(lowerBound)
                                                  + " <= x <= " + std::to_string(upperBound))
                                               : comment;

        if(saturateType == "normal")
        {
            // Normal case: clamp to [lowerBound, upperBound] using v_med3_i32
            // med3(x, lower, upper) = median of {x, lower, upper}
            // This effectively does: max(lower, min(upper, x))

            if(initGpr)
            {
                // Initialize temp registers with bounds
                auto inst = st.SMovkI32(
                    sgpr(tmpSgpr), imm(lowerBound), "lower bound = " + std::to_string(lowerBound));
                result.insert(result.end(), inst.begin(), inst.end());

                inst = st.VMovB32(
                    vgpr(tmpVgpr), imm(upperBound), "upper bound = " + std::to_string(upperBound));
                result.insert(result.end(), inst.begin(), inst.end());
            }

            // Use v_med3_i32 to clamp: result = median(value, lower, upper)
            auto inst = st.VMed3I32(vgpr(valueReg),
                                    vgpr(valueReg),
                                    sgpr(tmpSgpr),
                                    vgpr(tmpVgpr),
                                    "x = median(x, " + std::to_string(lowerBound) + ", "
                                        + std::to_string(upperBound) + ")");
            result.insert(result.end(), inst.begin(), inst.end());
        }
        else if(saturateType == "upper")
        {
            // Upper bound only: use v_min_i32
            auto inst = st.VMinI32(vgpr(valueReg),
                                   imm(upperBound),
                                   vgpr(valueReg),
                                   "x = min(x, " + std::to_string(upperBound) + ")");
            result.insert(result.end(), inst.begin(), inst.end());
        }
        else if(saturateType == "lower")
        {
            // Lower bound only: would use v_max_i32
            // NOTE: v_max_i32 is not yet implemented in StinkyTofu
            return Expected<std::vector<StinkyInstruction*>>::Error(
                "VSaturateCastInt with saturateType='lower' requires v_max_i32 instruction "
                "which is not yet implemented in StinkyTofu. Use 'normal' or 'upper' instead.");
        }
        else if(saturateType == "none")
        {
            // Do nothing - no saturation
        }
        else
        {
            return Expected<std::vector<StinkyInstruction*>>::Error(
                "Invalid saturateType: " + saturateType
                + ". Supported types: 'normal', 'upper', 'lower', 'none'");
        }

        return result;
    }

    // ============================================================================
    // Memory & Synchronization Functions (functions.hpp)
    // ============================================================================

    Expected<std::vector<StinkyInstruction*>> StinkyIR::DSInit(StinkyTofu&        builder,
                                                               uint32_t           tmpVgprStart,
                                                               uint32_t           serialVgpr,
                                                               uint32_t           numThreads,
                                                               uint32_t           ldsNumElements,
                                                               int32_t            initValue,
                                                               const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        StinkyTofu&                     st = builder;

        assert(tmpVgprStart + 1 < 256 && "DSInit requires 2 consecutive VGPRs");
        if(tmpVgprStart + 1 >= 256)
        {
            return Expected<std::vector<StinkyInstruction*>>::Error(
                "DSInit requires 2 consecutive VGPRs starting at tmpVgprStart (max: 254)");
        }

        uint32_t tmpValueReg = tmpVgprStart; // VGPR for init value
        uint32_t tmpAddrReg  = tmpVgprStart + 1; // VGPR for LDS address

        std::string mcomment = comment.empty() ? "initialize LDS" : comment;

        // Step 1: Wait for any pending memory operations
        auto inst = st.SWaitCnt(0, 0, 0, 0, "wait before LDS init");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 2: Barrier to synchronize all threads
        inst = st.SBarrier("sync before LDS init");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 3: Load init value into VGPR
        inst = st.VMovB32(vgpr(tmpValueReg),
                          imm(static_cast<int32_t>(initValue)),
                          "init value = " + std::to_string(initValue));
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 4: Calculate per-thread LDS address
        // address = serial * 4 (multiply by 4 bytes per dword)
        inst = st.VLShiftLeftB32(vgpr(tmpAddrReg), imm(2), vgpr(serialVgpr), "addr = serial << 2");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 5: Write to LDS in a loop
        // Each thread writes multiple times to cover all LDS elements
        uint32_t writesPerThread = ((ldsNumElements - 1) / numThreads / 4) + 1;

        for(uint32_t i = 0; i < writesPerThread; ++i)
        {
            // Note: In the original rocisa code, offset is passed via DSModifiers
            // In StinkyTofu, we need to add offset to address or use instruction with offset
            // For simplicity, if offset is needed, we'd add it to the address register first
            // But DSStoreB32 in StinkyTofu appears to only take addr and src

            if(i == 0)
            {
                // First write: use address as-is
                inst = st.DSStoreB32(vgpr(tmpAddrReg),
                                     vgpr(tmpValueReg),
                                     "LDS[addr] = " + std::to_string(initValue));
                result.insert(result.end(), inst.begin(), inst.end());
            }
            else
            {
                // Subsequent writes: add offset to address
                uint32_t offset = i * numThreads * 4;
                inst            = st.VAddU32(vgpr(tmpAddrReg),
                                  vgpr(tmpAddrReg),
                                  imm(static_cast<int32_t>(numThreads * 4)),
                                  "addr += " + std::to_string(numThreads * 4));
                result.insert(result.end(), inst.begin(), inst.end());

                inst = st.DSStoreB32(vgpr(tmpAddrReg),
                                     vgpr(tmpValueReg),
                                     "LDS[addr+" + std::to_string(offset)
                                         + "] = " + std::to_string(initValue));
                result.insert(result.end(), inst.begin(), inst.end());
            }
        }

        // Step 6: Wait for LDS writes to complete
        inst = st.SWaitCnt(0, 0, 0, 0, "wait for LDS init to complete");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 7: Final barrier to ensure all threads have finished
        inst = st.SBarrier("sync after LDS init");
        result.insert(result.end(), inst.begin(), inst.end());

        return result;
    }

    // ============================================================================
    // ArgumentLoader Implementation
    // ============================================================================

    ArgumentLoader::ArgumentLoader(StinkyTofu& builder)
        : builder(builder)
        , kernArgOffset(0)
    {
    }

    void ArgumentLoader::resetOffset()
    {
        kernArgOffset = 0;
    }

    void ArgumentLoader::setOffset(int offset)
    {
        kernArgOffset = offset;
    }

    int ArgumentLoader::getOffset() const
    {
        return kernArgOffset;
    }

    Expected<std::vector<StinkyInstruction*>>
        ArgumentLoader::loadKernArg(uint32_t           dstSgpr,
                                    uint32_t           srcAddr,
                                    int                dword,
                                    bool               writeSgpr,
                                    std::optional<int> sgprOffset,
                                    const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        int                             size = dword * 4;

        if(writeSgpr)
        {
            // Determine which offset to use
            int offsetToUse = sgprOffset.has_value() ? *sgprOffset : kernArgOffset;

            // Generate comment showing offset
            std::string finalComment = comment.empty()
                                           ? ("load kernarg offset " + std::to_string(offsetToUse))
                                           : comment;

            // Select appropriate SLoadBX instruction based on dword count
            std::vector<StinkyInstruction*> inst;
            switch(dword)
            {
            case 1: // 32-bit (B32)
                inst = builder.SLoadB32(
                    sgpr(dstSgpr, dword), sgpr(srcAddr, 2), offsetToUse, finalComment);
                break;
            case 2: // 64-bit (B64)
                inst = builder.SLoadB64(
                    sgpr(dstSgpr, dword), sgpr(srcAddr, 2), offsetToUse, finalComment);
                break;
            case 4: // 128-bit (B128)
                inst = builder.SLoadB128(
                    sgpr(dstSgpr, dword), sgpr(srcAddr, 2), offsetToUse, finalComment);
                break;
            case 8: // 256-bit (B256)
                inst = builder.SLoadB256(
                    sgpr(dstSgpr, dword), sgpr(srcAddr, 2), offsetToUse, finalComment);
                break;
            case 16: // 512-bit (B512)
                inst = builder.SLoadB512(
                    sgpr(dstSgpr, dword), sgpr(srcAddr, 2), offsetToUse, finalComment);
                break;
            default:
                return Expected<std::vector<StinkyInstruction*>>::Error(
                    "Invalid dword size for loadKernArg: " + std::to_string(dword)
                    + ". Valid values: 1, 2, 4, 8, 16");
            }

            result.insert(result.end(), inst.begin(), inst.end());
        }

        // Advance offset only if sgprOffset was not provided
        if(!sgprOffset.has_value())
        {
            kernArgOffset += size;
        }

        return result;
    }

    Expected<std::vector<StinkyInstruction*>> ArgumentLoader::loadAllKernArg(
        uint32_t sgprStartIndex, uint32_t srcAddr, int numSgprToLoad, int numSgprPreload)
    {
        std::vector<StinkyInstruction*> result;

        // Adjust for preloaded SGPRs
        int actualLoad = numSgprToLoad - numSgprPreload;
        sgprStartIndex += numSgprPreload;
        kernArgOffset += numSgprPreload * 4;

        // Load remaining SGPRs using largest possible chunks
        while(actualLoad > 0)
        {
            // Try sizes: 16, 8, 4, 2, 1 (dwords)
            int i = 16;
            while(i >= 1)
            {
                bool isSgprAligned = false;

                // Check SGPR alignment requirements
                if((i >= 4) && (sgprStartIndex % 4 == 0))
                {
                    isSgprAligned = true; // Aligned for 128-bit+ loads
                }
                else if((i == 2) && (sgprStartIndex % 2 == 0))
                {
                    isSgprAligned = true; // Aligned for 64-bit loads
                }
                else if(i == 1)
                {
                    isSgprAligned = true; // Always aligned for 32-bit loads
                }

                // If aligned and enough SGPRs remaining, generate load instruction
                if(isSgprAligned && actualLoad >= i)
                {
                    actualLoad -= i;

                    std::string kernArgOffsetStr = std::to_string(kernArgOffset);

                    std::vector<StinkyInstruction*> inst;
                    switch(i)
                    {
                    case 16: // B512
                        inst = builder.SLoadB512(sgpr(sgprStartIndex, i),
                                                 sgpr(srcAddr, 2),
                                                 kernArgOffset,
                                                 kernArgOffsetStr);
                        break;
                    case 8: // B256
                        inst = builder.SLoadB256(sgpr(sgprStartIndex, i),
                                                 sgpr(srcAddr, 2),
                                                 kernArgOffset,
                                                 kernArgOffsetStr);
                        break;
                    case 4: // B128
                        inst = builder.SLoadB128(sgpr(sgprStartIndex, i),
                                                 sgpr(srcAddr, 2),
                                                 kernArgOffset,
                                                 kernArgOffsetStr);
                        break;
                    case 2: // B64
                        inst = builder.SLoadB64(sgpr(sgprStartIndex, i),
                                                sgpr(srcAddr, 2),
                                                kernArgOffset,
                                                kernArgOffsetStr);
                        break;
                    case 1: // B32
                        inst = builder.SLoadB32(sgpr(sgprStartIndex, i),
                                                sgpr(srcAddr, 2),
                                                kernArgOffset,
                                                kernArgOffsetStr);
                        break;
                    default:
                        return Expected<std::vector<StinkyInstruction*>>::Error(
                            "Invalid SGPR size in loadAllKernArg: " + std::to_string(i));
                    }

                    result.insert(result.end(), inst.begin(), inst.end());

                    sgprStartIndex += i;
                    kernArgOffset += i * 4;
                    break; // Move to next chunk
                }

                i /= 2; // Try smaller size
            }
        }

        return result;
    }

    // ========================================================================
    // Activation Magic Numbers (from Activation.py)
    // ========================================================================

    namespace
    {
        // Union to convert hex bit pattern to float
        union FloatUnion
        {
            uint32_t u;
            float    f;
            constexpr FloatUnion(uint32_t bits)
                : u(bits)
            {
            }
        };

        // All magic numbers from ActivationMagicNumbers in Activation.py
        constexpr uint32_t FloatGeluK0Bits   = 0x3F4C422A; // 0.797884583
        constexpr uint32_t FloatGeluK1Bits   = 0x3D372713; // 0.044715
        constexpr uint32_t Float16GeluK1Bits = 0x29B9; // 0.044715 (f16)
        constexpr uint32_t FloatDGeluK0Bits  = 0x3D5B33B3; // 0.0535161
        constexpr uint32_t FloatDGeluK1Bits  = 0x3ECC4220; // 0.398942
        constexpr uint32_t FloatDGeluK2Bits  = 0x3D12220C; // 0.035677
        constexpr uint32_t FloatDGeluK3Bits  = 0x3F4C4231; // 0.797885

        // log2(e) = 1.442695040888963 for exp implementation
        constexpr uint32_t Log2EBits = 0x3FB8AA3B; // 1.442695040888963

        // Getter functions
        inline float getFloatGeluK0()
        {
            return FloatUnion(FloatGeluK0Bits).f;
        }
        inline float getFloatGeluK1()
        {
            return FloatUnion(FloatGeluK1Bits).f;
        }
        inline float getFloat16GeluK1()
        {
            return FloatUnion(Float16GeluK1Bits).f;
        }
        inline float getFloatDGeluK0()
        {
            return FloatUnion(FloatDGeluK0Bits).f;
        }
        inline float getFloatDGeluK1()
        {
            return FloatUnion(FloatDGeluK1Bits).f;
        }
        inline float getFloatDGeluK2()
        {
            return FloatUnion(FloatDGeluK2Bits).f;
        }
        inline float getFloatDGeluK3()
        {
            return FloatUnion(FloatDGeluK3Bits).f;
        }
        inline float getLog2E()
        {
            return FloatUnion(Log2EBits).f;
        }
    }

    // ========================================================================
    // Activation Helper Functions
    // ========================================================================

    // exp(x) = 2^(x * log2(e))
    // Modifies vgprInOut in place
    static void expF32InPlace(StinkyTofu&                      builder,
                              uint32_t                         vgprInOut,
                              std::vector<StinkyInstruction*>& result)
    {
        // Step 1: x = x * log2(e)
        auto inst
            = builder.VMulF32(vgpr(vgprInOut), imm(getLog2E()), vgpr(vgprInOut), "x * log2(e)");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 2: x = 2^x
        inst = builder.VExpF32(vgpr(vgprInOut), vgpr(vgprInOut), "2^x");
        result.insert(result.end(), inst.begin(), inst.end());
    }

    static void expF16InPlace(StinkyTofu&                      builder,
                              uint32_t                         vgprInOut,
                              std::vector<StinkyInstruction*>& result)
    {
        // Step 1: x = x * log2(e)
        auto inst
            = builder.VMulF16(vgpr(vgprInOut), imm(getLog2E()), vgpr(vgprInOut), "x * log2(e)");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 2: x = 2^x
        inst = builder.VExpF16(vgpr(vgprInOut), vgpr(vgprInOut), "2^x");
        result.insert(result.end(), inst.begin(), inst.end());
    }

    // tanh(x) = 1 - 2/(exp(2x) + 1)
    // Modifies vgprInOut in place
    static void tanhF32InPlace(StinkyTofu&                      builder,
                               uint32_t                         vgprInOut,
                               std::vector<StinkyInstruction*>& result)
    {
        // Step 1: x = 2 * x
        auto inst = builder.VMulF32(vgpr(vgprInOut), imm(2.0), vgpr(vgprInOut), "2 * x");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 2: x = exp(x)
        expF32InPlace(builder, vgprInOut, result);

        // Step 3: x = 1.0 + x  (now x = exp(2x) + 1)
        inst = builder.VAddF32(vgpr(vgprInOut), imm(1.0), vgpr(vgprInOut), "1 + exp(2x)");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 4: x = 1/x
        inst = builder.VRcpF32(vgpr(vgprInOut), vgpr(vgprInOut), "1 / (1 + exp(2x))");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 5: x = -2*x + 1  (final tanh)
        inst = builder.VFmaF32(
            vgpr(vgprInOut), imm(-2.0), vgpr(vgprInOut), imm(1.0), "tanh(x) = 1 - 2/(exp(2x) + 1)");
        result.insert(result.end(), inst.begin(), inst.end());
    }

    static void tanhF16InPlace(StinkyTofu&                      builder,
                               uint32_t                         vgprInOut,
                               std::vector<StinkyInstruction*>& result)
    {
        // Step 1: x = 2 * x
        auto inst = builder.VMulF16(vgpr(vgprInOut), imm(2.0), vgpr(vgprInOut), "2 * x");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 2: x = exp(x)
        expF16InPlace(builder, vgprInOut, result);

        // Step 3: x = 1.0 + x  (now x = exp(2x) + 1)
        inst = builder.VAddF16(vgpr(vgprInOut), imm(1.0), vgpr(vgprInOut), "1 + exp(2x)");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 4: x = 1/x
        inst = builder.VRcpF16(vgpr(vgprInOut), vgpr(vgprInOut), "1 / (1 + exp(2x))");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 5: x = -2*x + 1  (final tanh)
        inst = builder.VFmaF16(
            vgpr(vgprInOut), imm(-2.0), vgpr(vgprInOut), imm(1.0), "tanh(x) = 1 - 2/(exp(2x) + 1)");
        result.insert(result.end(), inst.begin(), inst.end());
    }

    // ========================================================================
    // Activation Functions
    // ========================================================================

    // ReLU: max(0, x)
    std::vector<StinkyInstruction*>
        StinkyIR::reluF16(StinkyTofu& builder, uint32_t vgprIn, uint32_t vgprOut)
    {
        auto result = builder.VMaxF16(vgpr(vgprOut), vgpr(vgprIn), imm(0.0), "x = max(0, x)");
        return result;
    }

    std::vector<StinkyInstruction*>
        StinkyIR::reluF32(StinkyTofu& builder, uint32_t vgprIn, uint32_t vgprOut)
    {
        auto result = builder.VMaxF32(vgpr(vgprOut), vgpr(vgprIn), imm(0.0), "x = max(0, x)");
        return result;
    }

    std::vector<StinkyInstruction*>
        StinkyIR::reluF64(StinkyTofu& builder, uint32_t vgprIn, uint32_t vgprOut)
    {
        auto result = builder.VMaxF64(vgpr(vgprOut, 2), vgpr(vgprIn, 2), imm(0.0), "x = max(0, x)");
        return result;
    }

    std::vector<StinkyInstruction*>
        StinkyIR::reluI32(StinkyTofu& builder, uint32_t vgprIn, uint32_t vgprOut)
    {
        auto result = builder.VMaxI32(vgpr(vgprOut), vgpr(vgprIn), imm(0), "x = max(0, x)");
        return result;
    }

    // Leaky ReLU: x >= 0 ? x : alpha * x
    // Uses conditional mask to select between x and alpha*x based on sign
    std::vector<StinkyInstruction*> StinkyIR::leakyReluF16(StinkyTofu&    builder,
                                                           uint32_t       vgprIn,
                                                           uint32_t       vgprOut,
                                                           StinkyRegister alpha)
    {
        std::vector<StinkyInstruction*> result;

        // tmp = alpha * x
        auto inst = builder.VMulF16(vgpr(vgprOut), alpha, vgpr(vgprIn), "tmp = alpha * x");
        result.insert(result.end(), inst.begin(), inst.end());

        // cmp = (x >= 0) -> sets VCC
        inst = builder.VCmpGEF16(vcc(), vgpr(vgprIn), imm(0.0), "x >= 0 ?");
        result.insert(result.end(), inst.begin(), inst.end());

        // out = cmp ? x : tmp (if x >= 0 use x, else use alpha*x)
        inst = builder.VCndMaskB32(
            vgpr(vgprOut), vgpr(vgprOut), vgpr(vgprIn), "select x if >= 0, else alpha*x");
        result.insert(result.end(), inst.begin(), inst.end());

        return result;
    }

    std::vector<StinkyInstruction*> StinkyIR::leakyReluF32(StinkyTofu&    builder,
                                                           uint32_t       vgprIn,
                                                           uint32_t       vgprOut,
                                                           StinkyRegister alpha)
    {
        std::vector<StinkyInstruction*> result;

        // tmp = alpha * x
        auto inst = builder.VMulF32(vgpr(vgprOut), alpha, vgpr(vgprIn), "tmp = alpha * x");
        result.insert(result.end(), inst.begin(), inst.end());

        // cmp = (x >= 0) -> sets VCC
        inst = builder.VCmpGEF32(vcc(), vgpr(vgprIn), imm(0.0), "x >= 0 ?");
        result.insert(result.end(), inst.begin(), inst.end());

        // out = cmp ? x : tmp (if x >= 0 use x, else use alpha*x)
        inst = builder.VCndMaskB32(
            vgpr(vgprOut), vgpr(vgprOut), vgpr(vgprIn), "select x if >= 0, else alpha*x");
        result.insert(result.end(), inst.begin(), inst.end());

        return result;
    }

    // GELU: 0.5 * x * (1 + tanh(k0 * x * (1 + k1 * x * x)))
    std::vector<StinkyInstruction*>
        StinkyIR::geluF32(StinkyTofu& builder, uint32_t vgprIn, uint32_t vgprOut, uint32_t tmpVgpr)
    {
        std::vector<StinkyInstruction*> result;

        // tmp = x * x
        auto inst = builder.VMulF32(vgpr(tmpVgpr), vgpr(vgprIn), vgpr(vgprIn), "x * x");
        result.insert(result.end(), inst.begin(), inst.end());

        // tmp = x^2 * k1 + 1 (FMA)
        inst = builder.VFmaF32(
            vgpr(tmpVgpr), vgpr(tmpVgpr), imm(getFloatGeluK1()), imm(1.0), "x^2 * k1 + 1");
        result.insert(result.end(), inst.begin(), inst.end());

        // tmp = x * (x^2 * k1 + 1)
        inst = builder.VMulF32(vgpr(tmpVgpr), vgpr(vgprIn), vgpr(tmpVgpr), "x * (x^2 * k1 + 1)");
        result.insert(result.end(), inst.begin(), inst.end());

        // tmp = k0 * x * (x^2 * k1 + 1)
        inst = builder.VMulF32(
            vgpr(tmpVgpr), imm(getFloatGeluK0()), vgpr(tmpVgpr), "k0 * x * (x^2 * k1 + 1)");
        result.insert(result.end(), inst.begin(), inst.end());

        // tmp = tanh(tmp)
        tanhF32InPlace(builder, tmpVgpr, result);

        // tmp = 1 + tanh(...)
        inst = builder.VAddF32(vgpr(tmpVgpr), imm(1.0), vgpr(tmpVgpr), "1 + tanh(...)");
        result.insert(result.end(), inst.begin(), inst.end());

        // tmp = x * (1 + tanh(...))
        inst = builder.VMulF32(vgpr(tmpVgpr), vgpr(vgprIn), vgpr(tmpVgpr), "x * (1 + tanh(...))");
        result.insert(result.end(), inst.begin(), inst.end());

        // out = 0.5 * tmp
        inst = builder.VMulF32(vgpr(vgprOut), imm(0.5), vgpr(tmpVgpr), "0.5 * x * (1 + tanh(...))");
        result.insert(result.end(), inst.begin(), inst.end());

        return result;
    }

    std::vector<StinkyInstruction*>
        StinkyIR::geluF16(StinkyTofu& builder, uint32_t vgprIn, uint32_t vgprOut, uint32_t tmpVgpr)
    {
        std::vector<StinkyInstruction*> result;

        // tmp = x * x
        auto inst = builder.VMulF16(vgpr(tmpVgpr), vgpr(vgprIn), vgpr(vgprIn), "x * x");
        result.insert(result.end(), inst.begin(), inst.end());

        // tmp = x^2 * k1 + 1 (FMA)
        inst = builder.VFmaF16(
            vgpr(tmpVgpr), vgpr(tmpVgpr), imm(getFloatGeluK1()), imm(1.0), "x^2 * k1 + 1");
        result.insert(result.end(), inst.begin(), inst.end());

        // tmp = x * (x^2 * k1 + 1)
        inst = builder.VMulF16(vgpr(tmpVgpr), vgpr(vgprIn), vgpr(tmpVgpr), "x * (x^2 * k1 + 1)");
        result.insert(result.end(), inst.begin(), inst.end());

        // tmp = k0 * x * (x^2 * k1 + 1) - using exact bit pattern
        inst = builder.VMulF16(
            vgpr(tmpVgpr), imm(getFloatGeluK0()), vgpr(tmpVgpr), "k0 * x * (x^2 * k1 + 1)");
        result.insert(result.end(), inst.begin(), inst.end());

        // tmp = tanh(tmp)
        tanhF16InPlace(builder, tmpVgpr, result);

        // tmp = 1 + tanh(...)
        inst = builder.VAddF16(vgpr(tmpVgpr), imm(1.0), vgpr(tmpVgpr), "1 + tanh(...)");
        result.insert(result.end(), inst.begin(), inst.end());

        // tmp = x * (1 + tanh(...))
        inst = builder.VMulF16(vgpr(tmpVgpr), vgpr(vgprIn), vgpr(tmpVgpr), "x * (1 + tanh(...))");
        result.insert(result.end(), inst.begin(), inst.end());

        // out = 0.5 * tmp
        inst = builder.VMulF16(vgpr(vgprOut), imm(0.5), vgpr(tmpVgpr), "0.5 * x * (1 + tanh(...))");
        result.insert(result.end(), inst.begin(), inst.end());

        return result;
    }

    // Sigmoid: 1 / (1 + exp(-x))
    std::vector<StinkyInstruction*> StinkyIR::sigmoidF32(StinkyTofu& builder,
                                                         uint32_t    vgprIn,
                                                         uint32_t    vgprOut,
                                                         uint32_t    tmpVgpr)
    {
        std::vector<StinkyInstruction*> result;

        // out = -x
        auto inst = builder.VMulF32(vgpr(vgprOut), imm(-1.0), vgpr(vgprIn), "-x");
        result.insert(result.end(), inst.begin(), inst.end());

        // out = exp(-x)
        expF32InPlace(builder, vgprOut, result);

        // out = 1 + exp(-x)
        inst = builder.VAddF32(vgpr(vgprOut), imm(1.0), vgpr(vgprOut), "1 + exp(-x)");
        result.insert(result.end(), inst.begin(), inst.end());

        // out = 1 / (1 + exp(-x))
        inst = builder.VRcpF32(vgpr(vgprOut), vgpr(vgprOut), "1 / (1 + exp(-x))");
        result.insert(result.end(), inst.begin(), inst.end());

        return result;
    }

    std::vector<StinkyInstruction*> StinkyIR::sigmoidF16(StinkyTofu& builder,
                                                         uint32_t    vgprIn,
                                                         uint32_t    vgprOut,
                                                         uint32_t    tmpVgpr)
    {
        std::vector<StinkyInstruction*> result;

        // out = -x
        auto inst = builder.VMulF16(vgpr(vgprOut), imm(-1.0), vgpr(vgprIn), "-x");
        result.insert(result.end(), inst.begin(), inst.end());

        // out = exp(-x)
        expF16InPlace(builder, vgprOut, result);

        // out = 1 + exp(-x)
        inst = builder.VAddF16(vgpr(vgprOut), imm(1.0), vgpr(vgprOut), "1 + exp(-x)");
        result.insert(result.end(), inst.begin(), inst.end());

        // out = 1 / (1 + exp(-x))
        inst = builder.VRcpF16(vgpr(vgprOut), vgpr(vgprOut), "1 / (1 + exp(-x))");
        result.insert(result.end(), inst.begin(), inst.end());

        return result;
    }

    // Abs: abs(x) - TODO: Implement proper abs with VOP3 modifiers
    std::vector<StinkyInstruction*>
        StinkyIR::absF16(StinkyTofu& builder, uint32_t vgprIn, uint32_t vgprOut)
    {
        // Placeholder: use AND with bitmask to clear sign bit
        auto result
            = builder.VAndB32(vgpr(vgprOut), imm(0x7FFF7FFF), vgpr(vgprIn), "abs(x) via and");
        return result;
    }

    std::vector<StinkyInstruction*>
        StinkyIR::absF32(StinkyTofu& builder, uint32_t vgprIn, uint32_t vgprOut)
    {
        // Placeholder: use AND with bitmask to clear sign bit
        auto result
            = builder.VAndB32(vgpr(vgprOut), imm(0x7FFFFFFF), vgpr(vgprIn), "abs(x) via and");
        return result;
    }

    // Clamp: max(alpha, min(x, beta))
    // Python order: first min, then max (matches Activation.py line 913-914)
    std::vector<StinkyInstruction*> StinkyIR::clampF16(StinkyTofu&    builder,
                                                       uint32_t       vgprIn,
                                                       uint32_t       vgprOut,
                                                       StinkyRegister alpha,
                                                       StinkyRegister beta)
    {
        std::vector<StinkyInstruction*> result;
        // Step 1: min(x, beta)
        auto inst = builder.VMinF16(vgpr(vgprOut), beta, vgpr(vgprIn), "min(x, beta)");
        result.insert(result.end(), inst.begin(), inst.end());
        // Step 2: max(alpha, min(x, beta))
        inst = builder.VMaxF16(vgpr(vgprOut), alpha, vgpr(vgprOut), "max(alpha, min(x, beta))");
        result.insert(result.end(), inst.begin(), inst.end());
        return result;
    }

    std::vector<StinkyInstruction*> StinkyIR::clampF32(StinkyTofu&    builder,
                                                       uint32_t       vgprIn,
                                                       uint32_t       vgprOut,
                                                       StinkyRegister alpha,
                                                       StinkyRegister beta)
    {
        std::vector<StinkyInstruction*> result;
        // Step 1: min(x, beta)
        auto inst = builder.VMinF32(vgpr(vgprOut), beta, vgpr(vgprIn), "min(x, beta)");
        result.insert(result.end(), inst.begin(), inst.end());
        // Step 2: max(alpha, min(x, beta))
        inst = builder.VMaxF32(vgpr(vgprOut), alpha, vgpr(vgprOut), "max(alpha, min(x, beta))");
        result.insert(result.end(), inst.begin(), inst.end());
        return result;
    }

    // Silu (Swish-1): x * sigmoid(x)
    std::vector<StinkyInstruction*>
        StinkyIR::siluF16(StinkyTofu& builder, uint32_t vgprIn, uint32_t vgprOut, uint32_t tmpVgpr)
    {
        std::vector<StinkyInstruction*> result;

        // Step 1: tmp = sigmoid(x)
        auto sigmoid_result = sigmoidF16(builder, vgprIn, tmpVgpr, tmpVgpr);
        result.insert(result.end(), sigmoid_result.begin(), sigmoid_result.end());

        // Step 2: out = x * sigmoid(x)
        auto inst = builder.VMulF16(vgpr(vgprOut), vgpr(vgprIn), vgpr(tmpVgpr), "x * sigmoid(x)");
        result.insert(result.end(), inst.begin(), inst.end());

        return result;
    }

    std::vector<StinkyInstruction*>
        StinkyIR::siluF32(StinkyTofu& builder, uint32_t vgprIn, uint32_t vgprOut, uint32_t tmpVgpr)
    {
        std::vector<StinkyInstruction*> result;

        // Step 1: tmp = sigmoid(x)
        auto sigmoid_result = sigmoidF32(builder, vgprIn, tmpVgpr, tmpVgpr);
        result.insert(result.end(), sigmoid_result.begin(), sigmoid_result.end());

        // Step 2: out = x * sigmoid(x)
        auto inst = builder.VMulF32(vgpr(vgprOut), vgpr(vgprIn), vgpr(tmpVgpr), "x * sigmoid(x)");
        result.insert(result.end(), inst.begin(), inst.end());

        return result;
    }

    // Swish: x * sigmoid(beta * x)
    std::vector<StinkyInstruction*> StinkyIR::swishF16(StinkyTofu&    builder,
                                                       uint32_t       vgprIn,
                                                       uint32_t       vgprOut,
                                                       StinkyRegister beta,
                                                       uint32_t       tmpVgpr1,
                                                       uint32_t       tmpVgpr2)
    {
        std::vector<StinkyInstruction*> result;

        // Step 1: tmp1 = x * beta
        auto inst = builder.VMulF16(vgpr(tmpVgpr1), vgpr(vgprIn), beta, "x * beta");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 2: tmp2 = sigmoid(x * beta)
        auto sigmoid_result = sigmoidF16(builder, tmpVgpr1, tmpVgpr2, tmpVgpr2);
        result.insert(result.end(), sigmoid_result.begin(), sigmoid_result.end());

        // Step 3: out = x * sigmoid(x * beta)
        inst
            = builder.VMulF16(vgpr(vgprOut), vgpr(vgprIn), vgpr(tmpVgpr2), "x * sigmoid(x * beta)");
        result.insert(result.end(), inst.begin(), inst.end());

        return result;
    }

    std::vector<StinkyInstruction*> StinkyIR::swishF32(StinkyTofu&    builder,
                                                       uint32_t       vgprIn,
                                                       uint32_t       vgprOut,
                                                       StinkyRegister beta,
                                                       uint32_t       tmpVgpr1,
                                                       uint32_t       tmpVgpr2)
    {
        std::vector<StinkyInstruction*> result;

        // Step 1: tmp1 = x * beta
        auto inst = builder.VMulF32(vgpr(tmpVgpr1), vgpr(vgprIn), beta, "x * beta");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 2: tmp2 = sigmoid(x * beta)
        auto sigmoid_result = sigmoidF32(builder, tmpVgpr1, tmpVgpr2, tmpVgpr2);
        result.insert(result.end(), sigmoid_result.begin(), sigmoid_result.end());

        // Step 3: out = x * sigmoid(x * beta)
        inst
            = builder.VMulF32(vgpr(vgprOut), vgpr(vgprIn), vgpr(tmpVgpr2), "x * sigmoid(x * beta)");
        result.insert(result.end(), inst.begin(), inst.end());

        return result;
    }

    // Clipped ReLU: if x > alpha: min(x, beta) else: min(0, beta)
    std::vector<StinkyInstruction*> StinkyIR::clippedReluF32(StinkyTofu&    builder,
                                                             uint32_t       vgprIn,
                                                             uint32_t       vgprOut,
                                                             StinkyRegister alpha,
                                                             StinkyRegister beta,
                                                             uint32_t       tmpVgpr)
    {
        std::vector<StinkyInstruction*> result;

        // Step 1: cmp = (x > alpha) -> sets VCC
        auto inst = builder.VCmpGTF32(vcc(), vgpr(vgprIn), alpha, "x > alpha ?");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 2: tmp1 = min(x, beta)
        inst = builder.VMinF32(vgpr(vgprOut), beta, vgpr(vgprIn), "min(x, beta)");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 3: tmp2 = min(0, beta)
        inst = builder.VMinF32(vgpr(tmpVgpr), beta, imm(0.0), "min(0, beta)");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 4: out = cmp ? tmp1 : tmp2
        inst = builder.VCndMaskB32(vgpr(vgprOut),
                                   vgpr(tmpVgpr),
                                   vgpr(vgprOut),
                                   "select min(x,beta) if x>alpha, else min(0,beta)");
        result.insert(result.end(), inst.begin(), inst.end());

        return result;
    }

    std::vector<StinkyInstruction*> StinkyIR::clippedReluF16(StinkyTofu&    builder,
                                                             uint32_t       vgprIn,
                                                             uint32_t       vgprOut,
                                                             StinkyRegister alpha,
                                                             StinkyRegister beta,
                                                             uint32_t       tmpVgpr)
    {
        std::vector<StinkyInstruction*> result;

        // Step 1: cmp = (x > alpha) -> sets VCC
        auto inst = builder.VCmpGTF16(vcc(), vgpr(vgprIn), alpha, "x > alpha ?");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 2: tmp1 = min(x, beta)
        inst = builder.VMinF16(vgpr(vgprOut), beta, vgpr(vgprIn), "min(x, beta)");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 3: tmp2 = min(0, beta)
        inst = builder.VMinF16(vgpr(tmpVgpr), beta, imm(0.0), "min(0, beta)");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 4: out = cmp ? tmp1 : tmp2
        inst = builder.VCndMaskB32(vgpr(vgprOut),
                                   vgpr(tmpVgpr),
                                   vgpr(vgprOut),
                                   "select min(x,beta) if x>alpha, else min(0,beta)");
        result.insert(result.end(), inst.begin(), inst.end());

        return result;
    }

    std::vector<StinkyInstruction*> StinkyIR::clippedReluF64(StinkyTofu&    builder,
                                                             uint32_t       vgprIn,
                                                             uint32_t       vgprOut,
                                                             StinkyRegister alpha,
                                                             StinkyRegister beta,
                                                             uint32_t       tmpVgpr)
    {
        std::vector<StinkyInstruction*> result;

        // Step 1: cmp = (x > alpha) -> sets VCC
        auto inst = builder.VCmpGTF64(vcc(), vgpr(vgprIn, 2), alpha, "x > alpha ?");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 2: tmp1 = min(x, beta)
        inst = builder.VMinF64(vgpr(vgprOut, 2), beta, vgpr(vgprIn, 2), "min(x, beta)");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 3: tmp2 = min(0, beta)
        inst = builder.VMinF64(vgpr(tmpVgpr, 2), beta, imm(0.0), "min(0, beta)");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 4: out = cmp ? tmp1 : tmp2 (need two VCndMask for f64, low and high)
        inst = builder.VCndMaskB32(vgpr(vgprOut),
                                   vgpr(tmpVgpr),
                                   vgpr(vgprOut),
                                   "select min(x,beta) if x>alpha, else min(0,beta) [low]");
        result.insert(result.end(), inst.begin(), inst.end());
        inst = builder.VCndMaskB32(vgpr(vgprOut + 1),
                                   vgpr(tmpVgpr + 1),
                                   vgpr(vgprOut + 1),
                                   "select min(x,beta) if x>alpha, else min(0,beta) [high]");
        result.insert(result.end(), inst.begin(), inst.end());

        return result;
    }

    std::vector<StinkyInstruction*> StinkyIR::clippedReluI32(StinkyTofu&    builder,
                                                             uint32_t       vgprIn,
                                                             uint32_t       vgprOut,
                                                             StinkyRegister alpha,
                                                             StinkyRegister beta,
                                                             uint32_t       tmpVgpr)
    {
        std::vector<StinkyInstruction*> result;

        // Step 1: cmp = (x > alpha) -> sets VCC
        auto inst = builder.VCmpGTI32(vcc(), vgpr(vgprIn), alpha, "x > alpha ?");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 2: tmp1 = min(x, beta)
        inst = builder.VMinI32(vgpr(vgprOut), beta, vgpr(vgprIn), "min(x, beta)");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 3: tmp2 = min(0, beta)
        inst = builder.VMinI32(vgpr(tmpVgpr), beta, imm(0), "min(0, beta)");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 4: out = cmp ? tmp1 : tmp2
        inst = builder.VCndMaskB32(vgpr(vgprOut),
                                   vgpr(tmpVgpr),
                                   vgpr(vgprOut),
                                   "select min(x,beta) if x>alpha, else min(0,beta)");
        result.insert(result.end(), inst.begin(), inst.end());

        return result;
    }

    // DGelu (Gradient of GELU): 0.5 * tanh(xx) + x1 * (4 / (exp(-xx) + exp(xx))^2) + 0.5
    // Python: lines 787-850 (f32 only)
    std::vector<StinkyInstruction*> StinkyIR::dgeluF32(StinkyTofu& builder,
                                                       uint32_t    vgprIn,
                                                       uint32_t    vgprOut,
                                                       uint32_t    tmpVgpr1,
                                                       uint32_t    tmpVgpr2,
                                                       uint32_t    tmpVgpr3)
    {
        std::vector<StinkyInstruction*> result;

        // Step 1: tmp1 = x * x
        auto inst = builder.VMulF32(vgpr(tmpVgpr1), vgpr(vgprIn), vgpr(vgprIn), "tmp1 = x * x");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 2: tmp1 = x * x * x = x^3
        inst = builder.VMulF32(vgpr(tmpVgpr1), vgpr(tmpVgpr1), vgpr(vgprIn), "tmp1 = x^3");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 3: tmp2 = 0.398942 * x (FloatDGeluK1)
        inst = builder.VMulF32(
            vgpr(tmpVgpr2), imm(getFloatDGeluK1()), vgpr(vgprIn), "tmp2 = 0.398942 * x");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 4: tmp2 = 0.0535161 * x^3 + tmp2 (FloatDGeluK0)
        inst = builder.VFmaF32(vgpr(tmpVgpr2),
                               imm(getFloatDGeluK0()),
                               vgpr(tmpVgpr1),
                               vgpr(tmpVgpr2),
                               "tmp2 = 0.0535161 * x^3 + tmp2");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 5: tmp3 = 0.797885 * x (FloatDGeluK3)
        inst = builder.VMulF32(
            vgpr(tmpVgpr3), imm(getFloatDGeluK3()), vgpr(vgprIn), "tmp3 = 0.797885 * x");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 6: tmp1 = 0.035677 * x^3 + tmp3 (FloatDGeluK2)  [tmp1 = xx]
        inst = builder.VFmaF32(vgpr(tmpVgpr1),
                               imm(getFloatDGeluK2()),
                               vgpr(tmpVgpr1),
                               vgpr(tmpVgpr3),
                               "tmp1 = 0.035677 * x^3 + tmp3 = xx");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 7: tmp3 = exp(xx), preserving tmp1
        inst = builder.VMovB32(vgpr(tmpVgpr3), vgpr(tmpVgpr1), "copy xx to tmp3");
        result.insert(result.end(), inst.begin(), inst.end());
        expF32InPlace(builder, tmpVgpr3, result); // tmp3 = exp(xx)

        // Step 8: tmp1 = -xx
        inst = builder.VMulF32(vgpr(tmpVgpr1), imm(-1.0), vgpr(tmpVgpr1), "tmp1 = -xx");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 9: tmp1 = exp(-xx)
        expF32InPlace(builder, tmpVgpr1, result);

        // Step 10: out = exp(xx) + exp(-exp(xx))
        inst = builder.VAddF32(
            vgpr(vgprOut), vgpr(tmpVgpr3), vgpr(tmpVgpr1), "out = exp(xx) + exp(-xx)");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 11: tmp1 = exp(xx) - exp(-exp(xx))
        inst = builder.VSubF32(
            vgpr(tmpVgpr1), vgpr(tmpVgpr3), vgpr(tmpVgpr1), "tmp1 = exp(xx) - exp(-xx)");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 12: tmp3 = 1 / out
        inst = builder.VRcpF32(vgpr(tmpVgpr3), vgpr(vgprOut), "tmp3 = 1 / (exp(xx) + exp(-xx))");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 13: tmp3 = tmp1 * tmp3 = tanh(xx)
        inst = builder.VMulF32(vgpr(tmpVgpr3), vgpr(tmpVgpr1), vgpr(tmpVgpr3), "tmp3 = tanh(xx)");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 14: tmp1 = 0.5 * tmp3
        inst = builder.VMulF32(vgpr(tmpVgpr1), imm(0.5), vgpr(tmpVgpr3), "tmp1 = 0.5 * tanh(xx)");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 15: out = out * out
        inst = builder.VMulF32(
            vgpr(vgprOut), vgpr(vgprOut), vgpr(vgprOut), "out = (exp(xx) + exp(-xx))^2");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 16: out = 1 / out = 1 / (exp(xx) + exp(-xx))^2
        inst = builder.VRcpF32(vgpr(vgprOut), vgpr(vgprOut), "out = 1 / (exp(xx) + exp(-xx))^2");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 17: out = 4 * out
        inst = builder.VMulF32(
            vgpr(vgprOut), imm(4.0), vgpr(vgprOut), "out = 4 / (exp(xx) + exp(-xx))^2");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 18: out = out * tmp2 + tmp1
        inst = builder.VFmaF32(vgpr(vgprOut),
                               vgpr(vgprOut),
                               vgpr(tmpVgpr2),
                               vgpr(tmpVgpr1),
                               "out = x1 * (4/(exp + exp)^2) + 0.5*tanh");
        result.insert(result.end(), inst.begin(), inst.end());

        // Step 19: out = out + 0.5
        inst = builder.VAddF32(vgpr(vgprOut), imm(0.5), vgpr(vgprOut), "out = out + 0.5");
        result.insert(result.end(), inst.begin(), inst.end());

        return result;
    }
} // namespace stinkytofu
