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

#include "StinkyIR.hpp"
#include "StinkyTofu.hpp"
#include <cmath>
#include <stdexcept>

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

    // ============================================================================
    // Division Functions
    // ============================================================================

    std::vector<StinkyInstruction*>
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
            if(tmpVgpr.size() < 2)
            {
                throw std::runtime_error(
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

    std::vector<StinkyInstruction*>
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

    std::vector<StinkyInstruction*>
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
            if(tmpSgpr.size() < 2)
            {
                throw std::runtime_error(
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

    std::vector<StinkyInstruction*>
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
            if(tmpSgpr.empty())
            {
                throw std::runtime_error(
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

    std::vector<StinkyInstruction*> StinkyIR::BranchIfZeroTyped(StinkyTofu&        builder,
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
            throw std::runtime_error("Unsupported data type for BranchIfZeroTyped: " + dataType
                                     + ". Supported types: i32, i64, f32, f64");
        }

        return result;
    }

    std::vector<StinkyInstruction*> StinkyIR::BranchIfNotZeroTyped(StinkyTofu&        builder,
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
            throw std::runtime_error("Unsupported data type for BranchIfNotZeroTyped: " + dataType
                                     + ". Supported types: i32, i64, f32, f64");
        }

        return result;
    }

    // ============================================================================
    // Casting Functions (f_cast.hpp)
    // ============================================================================

    std::vector<StinkyInstruction*> StinkyIR::VSaturateCastInt(StinkyTofu&        builder,
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
            throw std::runtime_error(
                "VSaturateCastInt with saturateType='lower' requires v_max_i32 instruction "
                "which is not yet implemented in StinkyTofu. Use 'normal' or 'upper' instead.");
        }
        else if(saturateType == "none")
        {
            // Do nothing - no saturation
        }
        else
        {
            throw std::runtime_error("Invalid saturateType: " + saturateType
                                     + ". Supported types: 'normal', 'upper', 'lower', 'none'");
        }

        return result;
    }

    // ============================================================================
    // Memory & Synchronization Functions (functions.hpp)
    // ============================================================================

    std::vector<StinkyInstruction*> StinkyIR::DSInit(StinkyTofu&        builder,
                                                     uint32_t           tmpVgprStart,
                                                     uint32_t           serialVgpr,
                                                     uint32_t           numThreads,
                                                     uint32_t           ldsNumElements,
                                                     int32_t            initValue,
                                                     const std::string& comment)
    {
        std::vector<StinkyInstruction*> result;
        StinkyTofu&                     st = builder;

        if(tmpVgprStart + 1 >= 256)
        {
            throw std::runtime_error(
                "DSInit requires 2 consecutive VGPRs starting at tmpVgprStart");
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
            // But DSWriteB32 in StinkyTofu appears to only take addr and src

            if(i == 0)
            {
                // First write: use address as-is
                inst = st.DSWriteB32(vgpr(tmpAddrReg),
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

                inst = st.DSWriteB32(vgpr(tmpAddrReg),
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

    std::vector<StinkyInstruction*> ArgumentLoader::loadKernArg(uint32_t           dstSgpr,
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
                throw std::invalid_argument("Invalid dword size for loadKernArg: "
                                            + std::to_string(dword)
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

    std::vector<StinkyInstruction*> ArgumentLoader::loadAllKernArg(uint32_t sgprStartIndex,
                                                                   uint32_t srcAddr,
                                                                   int      numSgprToLoad,
                                                                   int      numSgprPreload)
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
                        throw std::invalid_argument("Invalid SGPR size in loadAllKernArg: "
                                                    + std::to_string(i));
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

} // namespace stinkytofu
