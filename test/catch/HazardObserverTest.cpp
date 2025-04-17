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

#include <cmath>
#include <memory>

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

#include <common/SourceMatcher.hpp>
#include <common/TestValues.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_contains.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using namespace rocRoller;

namespace HazardObserverTest
{
    using Catch::Matchers::ContainsSubstring;

    void peekAndSchedule(TestContext& context, Instruction& inst, uint expectedNops = 0)
    {
        auto peeked = context->observer()->peek(inst);
        CHECK(peeked.nops == expectedNops);
        context->schedule(inst);
    }

    std::vector<rocRoller::Register::ValuePtr>
        createRegisters(TestContext&                           context,
                        rocRoller::Register::Type const        regType,
                        rocRoller::DataType const              dataType,
                        size_t const                           amount,
                        int const                              regCount     = 1,
                        rocRoller::Register::AllocationOptions allocOptions = {})
    {
        std::vector<rocRoller::Register::ValuePtr> regs;
        for(size_t i = 0; i < amount; i++)
        {
            auto reg = std::make_shared<rocRoller::Register::Value>(
                context.get(), regType, dataType, regCount, allocOptions);
            try
            {
                reg->allocateNow();
            }
            catch(...)
            {
                std::cout << i << std::endl;
                throw;
            }
            regs.push_back(reg);
        }
        return regs;
    }

    TEST_CASE("VALUWriteVCC", "[observer]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            if(arch.isRDNAGPU())
            {
                SKIP("RDNA not supported yet");
            }

            SECTION("v_readlane (2nd op) read as laneselect")
            {
                auto context = TestContext::ForTarget(arch);
                auto v = createRegisters(context, Register::Type::Vector, DataType::UInt32, 5);
                auto s = createRegisters(context, Register::Type::Scalar, DataType::UInt32, 2);

                std::vector<Instruction> insts
                    = {Instruction(
                           "v_readlane_b32", {s[0]}, {v[0], Register::Value::Literal(0)}, {}, ""),
                       Instruction("v_readlane_b32", {s[1]}, {v[1], s[0]}, {}, ""),
                       Instruction("s_endpgm", {}, {}, {}, "")};

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1], 4);

                CHECK_THAT(context.output(), ContainsSubstring("s_nop 3"));
            }

            SECTION("v_* (2nd op) read SGPR as constant")
            {

                auto context = TestContext::ForTarget(arch);
                auto v = createRegisters(context, Register::Type::Vector, DataType::UInt32, 5);
                auto s = createRegisters(context, Register::Type::Scalar, DataType::UInt32, 2);

                std::vector<Instruction> insts = {
                    Instruction(
                        "v_readlane_b32", {s[0]}, {v[0], Register::Value::Literal(0)}, {}, ""),
                    Instruction("v_cndmask_b32", {v[0]}, {s[0], v[3], context->getVCC()}, {}, ""),
                    Instruction("s_endpgm", {}, {}, {}, "")};

                if(arch.isCDNA1GPU() || arch.isCDNA2GPU())
                {
                    peekAndSchedule(context, insts[0]);
                    peekAndSchedule(context, insts[1], 1);

                    CHECK_THAT(context.output(), ContainsSubstring("s_nop 0"));
                }
                else
                {
                    // NOPs are required on 94X arch
                    peekAndSchedule(context, insts[0]);
                    peekAndSchedule(context, insts[1], 2);

                    CHECK_THAT(context.output(), ContainsSubstring("s_nop 1"));
                }
            }

            SECTION("v_addc* reading VCC as carry")
            {

                auto context = TestContext::ForTarget(arch);
                auto v = createRegisters(context, Register::Type::Vector, DataType::UInt32, 6);

                std::vector<Instruction> insts
                    = {Instruction("v_add_co_u32", {v[0], context->getVCC()}, {v[1], v[2]}, {}, ""),
                       Instruction("v_addc_co_u32",
                                   {v[5], context->getVCC()},
                                   {v[3], v[4], context->getVCC()},
                                   {},
                                   ""),
                       Instruction("s_endpgm", {}, {}, {}, "")};

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);

                CHECK_THAT(context.output(), !(ContainsSubstring("s_nop")));
            }

            SECTION("v_subb* reading VCC as carry")
            {

                auto context = TestContext::ForTarget(arch);
                auto v = createRegisters(context, Register::Type::Vector, DataType::UInt32, 6);

                std::vector<Instruction> insts
                    = {Instruction("v_sub_co_u32", {v[0], context->getVCC()}, {v[1], v[2]}, {}, ""),
                       Instruction("v_subb_co_u32",
                                   {v[5], context->getVCC()},
                                   {v[3], v[4], context->getVCC()},
                                   {},
                                   ""),
                       Instruction("s_endpgm", {}, {}, {}, "")};

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);

                CHECK_THAT(context.output(), !(ContainsSubstring("s_nop")));
            }

            SECTION("v_addc* reading VCC as non-carry")
            {

                auto context = TestContext::ForTarget(arch);
                auto v = createRegisters(context, Register::Type::Vector, DataType::UInt32, 6);

                std::vector<Instruction> insts
                    = {Instruction("v_add_co_u32", {v[0], context->getVCC()}, {v[1], v[2]}, {}, ""),
                       Instruction("v_addc_co_u32",
                                   {v[5], context->getVCC()},
                                   {v[3], context->getVCC(), context->getVCC()},
                                   {},
                                   ""),
                       Instruction("s_endpgm", {}, {}, {}, "")};

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1], 1);

                CHECK_THAT(context.output(), ContainsSubstring("s_nop 0"));
            }

            SECTION("v_addc* reading SGPR as non-carry")
            {

                auto context = TestContext::ForTarget(arch);
                auto v = createRegisters(context, Register::Type::Vector, DataType::UInt32, 6);
                auto s = createRegisters(context, Register::Type::Scalar, DataType::UInt32, 2);

                std::vector<Instruction> insts
                    = {Instruction("v_add_co_u32", {s[0], context->getVCC()}, {v[1], v[2]}, {}, ""),
                       Instruction("v_addc_co_u32",
                                   {v[5], context->getVCC()},
                                   {v[3], s[0], context->getVCC()},
                                   {},
                                   ""),
                       Instruction("s_endpgm", {}, {}, {}, "")};

                if(arch.isCDNA1GPU() || arch.isCDNA2GPU())
                {
                    peekAndSchedule(context, insts[0]);
                    peekAndSchedule(context, insts[1], 1);

                    CHECK_THAT(context.output(), ContainsSubstring("s_nop 0"));
                }
                else
                {
                    // NOPs are required on 94X arch
                    peekAndSchedule(context, insts[0]);
                    peekAndSchedule(context, insts[1], 2);

                    CHECK_THAT(context.output(), ContainsSubstring("s_nop 1"));
                }
            }

            SECTION("v_addc* reading SGPR as non-carry")
            {

                auto context = TestContext::ForTarget(arch);
                auto v = createRegisters(context, Register::Type::Vector, DataType::UInt32, 6);
                auto s = createRegisters(context, Register::Type::Scalar, DataType::UInt32, 2);

                std::vector<Instruction> insts
                    = {Instruction("v_add_co_u32", {s[0], context->getVCC()}, {v[1], v[2]}, {}, ""),
                       Instruction("v_addc_co_u32",
                                   {v[5], context->getVCC()},
                                   {v[3], s[0], context->getVCC()},
                                   {},
                                   ""),
                       Instruction("s_endpgm", {}, {}, {}, "")};

                if(arch.isCDNA1GPU() || arch.isCDNA2GPU())
                {
                    peekAndSchedule(context, insts[0]);
                    peekAndSchedule(context, insts[1], 1);

                    CHECK_THAT(context.output(), ContainsSubstring("s_nop 0"));
                }
                else
                {
                    // NOPs are required on 94X arch
                    peekAndSchedule(context, insts[0]);
                    peekAndSchedule(context, insts[1], 2);

                    CHECK_THAT(context.output(), ContainsSubstring("s_nop 1"));
                }
            }

            SECTION("This example can be found in basic I32 division")
            {
                auto context = TestContext::ForTarget(arch);
                auto v = createRegisters(context, Register::Type::Vector, DataType::UInt32, 5);
                auto s = createRegisters(context, Register::Type::Scalar, DataType::UInt32, 2);

                std::vector<Instruction> insts = {
                    Instruction("v_cmp_ge_u32", {context->getVCC()}, {v[0], v[1]}, {}, ""),
                    Instruction("v_cndmask_b32", {v[0]}, {v[2], v[3], context->getVCC()}, {}, ""),
                    Instruction("s_endpgm", {}, {}, {}, "")};

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1], 1);

                CHECK_THAT(context.output(), ContainsSubstring("s_nop 0"));
            }

            // TODO Fix implicit VCC observer and enable this test
#if 0
            SECTION("Implicit VCC write")
            {
            std::vector<Instruction> insts
                = {Instruction("v_add_co_u32", {v[3]}, {v[0], v[1]}, {}, ""),
                Instruction("v_cndmask_b32", {v[4]}, {v[4], v[3], context->getVCC()}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(context,insts[0]);
            peekAndSchedule(context,insts[1], 2);

            CHECK_THAT(context.output(), ContainsSubstring("s_nop 1"));
            }
#endif
        }
    }

    TEST_CASE("VALUTrans94X", "[observer]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            if(arch.isRDNAGPU())
            {
                SKIP("RDNA not supported yet");
            }

            SECTION("Has hazard with 2nd op (non-trans) accessing the same register")
            {
                auto context = TestContext::ForTarget(arch);
                auto v       = createRegisters(context, Register::Type::Vector, DataType::Float, 2);

                std::vector<Instruction> insts = {
                    Instruction("v_rcp_iflag_f32", {v[0]}, {v[0]}, {}, ""),
                    Instruction(
                        "v_mul_f32", {v[0]}, {Register::Value::Literal(0x4f7ffffe), v[0]}, {}, ""),
                    Instruction("s_endpgm", {}, {}, {}, "")};

                if(arch.isCDNA1GPU() || arch.isCDNA2GPU())
                {
                    peekAndSchedule(context, insts[0]);
                    peekAndSchedule(context, insts[1]);

                    CHECK_THAT(context.output(), !(ContainsSubstring("s_nop")));
                }
                else
                {
                    // NOPs are required on 94X arch
                    peekAndSchedule(context, insts[0]);
                    peekAndSchedule(context, insts[1], 1);

                    CHECK_THAT(context.output(), ContainsSubstring("s_nop 0"));
                }
            }

            SECTION("No hazard (2nd non-trans op accessing different VGPR)")
            {
                auto context = TestContext::ForTarget(arch);
                auto v       = createRegisters(context, Register::Type::Vector, DataType::Float, 2);

                std::vector<Instruction> insts = {
                    Instruction("v_rcp_iflag_f32", {v[0]}, {v[0]}, {}, ""),
                    Instruction(
                        "v_mul_f32", {v[1]}, {Register::Value::Literal(0x4f7ffffe), v[1]}, {}, ""),
                    Instruction("s_endpgm", {}, {}, {}, "")};

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);

                CHECK_THAT(context.output(), !(ContainsSubstring("s_nop")));
            }

            SECTION("No hazard (2nd op is a trans op)")
            {
                auto context = TestContext::ForTarget(arch);
                auto v       = createRegisters(context, Register::Type::Vector, DataType::Float, 2);

                std::vector<Instruction> insts
                    = {Instruction("v_rcp_iflag_f32", {v[0]}, {v[0]}, {}, ""),
                       Instruction("v_rcp_f32", {v[0]}, {v[0]}, {}, ""),
                       Instruction("s_endpgm", {}, {}, {}, "")};

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);

                CHECK_THAT(context.output(), !(ContainsSubstring("s_nop")));
            }

            SECTION("No hazard")
            {
                auto context = TestContext::ForTarget(arch);
                auto v       = createRegisters(context, Register::Type::Vector, DataType::Float, 2);
                auto vu = createRegisters(context, Register::Type::Vector, DataType::UInt32, 2);

                std::vector<Instruction> insts = {
                    Instruction("v_rcp_iflag_f32", {v[0]}, {v[0]}, {}, ""),
                    Instruction("v_add_u32", {vu[0]}, {vu[0], vu[1]}, {}, ""),
                    Instruction(
                        "v_mul_f32", {v[0]}, {Register::Value::Literal(0x4f7ffffe), v[0]}, {}, ""),
                    Instruction("s_endpgm", {}, {}, {}, "")};

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);
                peekAndSchedule(context, insts[2]);

                CHECK_THAT(context.output(), !(ContainsSubstring("s_nop")));
            }
        }
    }

    TEST_CASE("VALUWriteReadlane94X", "[observer]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            if(arch.isRDNAGPU())
            {
                SKIP("RDNA not supported yet");
            }

            SECTION("Hazard with VALU write followed by a readlane or permlane")
            {
                auto context = TestContext::ForTarget(arch);
                auto v = createRegisters(context, Register::Type::Vector, DataType::UInt32, 9);
                auto s = createRegisters(context, Register::Type::Scalar, DataType::UInt32, 2);

                std::vector<Instruction> insts
                    = {Instruction("v_subrev_u32", {v[0]}, {s[0], v[0]}, {}, ""),
                       Instruction(
                           "v_readlane_b32", {s[1]}, {v[0], Register::Value::Literal(0)}, {}, ""),
                       Instruction("s_endpgm", {}, {}, {}, "")};

                if(arch.isCDNA1GPU() || arch.isCDNA2GPU())
                {
                    peekAndSchedule(context, insts[0]);
                    peekAndSchedule(context, insts[1]);
                    peekAndSchedule(context, insts[2]);

                    CHECK_THAT(context.output(), !(ContainsSubstring("s_nop")));
                }
                else
                {
                    // NOPs are required on 94X arch
                    peekAndSchedule(context, insts[0]);
                    peekAndSchedule(context, insts[1], 1);
                    peekAndSchedule(context, insts[2]);

                    CHECK_THAT(context.output(), ContainsSubstring("s_nop 0"));
                }
            }

            SECTION("No hazard (accessing different VGPRs)")
            {
                auto context = TestContext::ForTarget(arch);
                auto v = createRegisters(context, Register::Type::Vector, DataType::UInt32, 3);
                auto s = createRegisters(context, Register::Type::Scalar, DataType::UInt32, 2);

                std::vector<Instruction> insts
                    = {Instruction("v_subrev_u32", {v[0]}, {s[0], v[0]}, {}, ""),
                       Instruction(
                           "v_readlane_b32", {s[1]}, {v[1], Register::Value::Literal(0)}, {}, ""),
                       Instruction("s_endpgm", {}, {}, {}, "")};

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);
                peekAndSchedule(context, insts[2]);

                CHECK_THAT(context.output(), !(ContainsSubstring("s_nop")));
            }

            SECTION("Tests for v_permlane*, only on GFX950")
            {
                if(!arch.isCDNA35GPU())
                {
                    SKIP("Only test GFX950");
                }

                SECTION("Hazard with 2nd op accessing the same register")
                {
                    auto context = TestContext::ForTarget(arch);
                    auto v = createRegisters(context, Register::Type::Vector, DataType::UInt32, 9);
                    auto s = createRegisters(context, Register::Type::Scalar, DataType::UInt32, 2);

                    std::vector<Instruction> insts
                        = {Instruction("v_subrev_u32", {v[0]}, {s[0], v[0]}, {}, ""),
                           Instruction::InoutInstruction(
                               "v_permlane32_swap_b32", {v[0], v[1]}, {}, ""),
                           Instruction("s_endpgm", {}, {}, {}, "")};

                    peekAndSchedule(context, insts[0]);
                    peekAndSchedule(context, insts[1], 2);
                    peekAndSchedule(context, insts[2]);
                    CHECK_THAT(context.output(), ContainsSubstring("s_nop 1"));
                }

                SECTION("No hazard. Access different registers.")
                {
                    auto context = TestContext::ForTarget(arch);
                    auto v = createRegisters(context, Register::Type::Vector, DataType::UInt32, 9);
                    auto s = createRegisters(context, Register::Type::Scalar, DataType::UInt32, 2);

                    std::vector<Instruction> insts
                        = {Instruction::InoutInstruction(
                               "v_permlane16_swap_b32", {v[3], v[4]}, {}, ""),
                           Instruction::InoutInstruction(
                               "v_permlane32_swap_b32", {v[5], v[6]}, {}, ""),
                           Instruction("s_endpgm", {}, {}, {}, "")};

                    peekAndSchedule(context, insts[0]);
                    peekAndSchedule(context, insts[1]);
                    peekAndSchedule(context, insts[2]);

                    CHECK_THAT(context.output(), !(ContainsSubstring("s_nop")));
                }
            }
        }
    }

    TEST_CASE("VCMPX_EXEC94X", "[observer]")
    {

        SUPPORTED_ARCH_SECTION(arch)
        {
            if(arch.isRDNAGPU())
            {
                SKIP("RDNA not supported yet");
            }

            SECTION("Hazard on 94X with 2nd op is v_readlane")
            {
                auto context = TestContext::ForTarget(arch);
                auto v = createRegisters(context, Register::Type::Vector, DataType::UInt32, 3);
                auto s = createRegisters(context, Register::Type::Scalar, DataType::UInt32, 2);

                std::vector<Instruction> insts
                    = {Instruction("v_cmpx_eq_u32", {}, {s[0], v[0]}, {}, ""),
                       Instruction(
                           "v_readlane_b32", {s[1]}, {v[0], Register::Value::Literal(0)}, {}, ""),
                       Instruction("s_endpgm", {}, {}, {}, "")};

                if(arch.isCDNA1GPU() || arch.isCDNA2GPU())
                {
                    peekAndSchedule(context, insts[0]);
                    peekAndSchedule(context, insts[1]);

                    CHECK_THAT(context.output(), !(ContainsSubstring("s_nop")));
                }
                else
                {
                    // NOPs are required on 94X arch
                    peekAndSchedule(context, insts[0]);
                    peekAndSchedule(context, insts[1], 4);

                    CHECK_THAT(context.output(), ContainsSubstring("s_nop 3"));
                }
            }

            SECTION("Hazard on 94X with 2nd op reads EXEC")
            {
                auto context = TestContext::ForTarget(arch);
                auto v = createRegisters(context, Register::Type::Vector, DataType::UInt32, 3);
                auto s = createRegisters(context, Register::Type::Scalar, DataType::UInt32, 2);

                std::vector<Instruction> insts
                    = {Instruction("v_cmpx_eq_u32", {}, {s[0], v[0]}, {}, ""),
                       Instruction("v_xor_b32", {v[1]}, {v[0], context->getExec()}, {}, ""),
                       Instruction("s_endpgm", {}, {}, {}, "")};

                if(arch.isCDNA1GPU() || arch.isCDNA2GPU())
                {
                    peekAndSchedule(context, insts[0]);
                    peekAndSchedule(context, insts[1]);

                    CHECK_THAT(context.output(), !(ContainsSubstring("s_nop")));
                }
                else
                {
                    // NOPs are required on 94X arch
                    peekAndSchedule(context, insts[0]);
                    peekAndSchedule(context, insts[1], 2);

                    CHECK_THAT(context.output(), ContainsSubstring("s_nop 1"));
                }
            }

            SECTION("No hazard on with 2nd op not reading EXEC")
            {
                auto context = TestContext::ForTarget(arch);
                auto v = createRegisters(context, Register::Type::Vector, DataType::UInt32, 3);
                auto s = createRegisters(context, Register::Type::Scalar, DataType::UInt32, 2);

                std::vector<Instruction> insts
                    = {Instruction("v_cmpx_eq_u32", {}, {s[0], v[0]}, {}, ""),
                       Instruction("v_xor_b32", {v[1]}, {v[0], v[1]}, {}, ""),
                       Instruction("s_endpgm", {}, {}, {}, "")};

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);

                CHECK_THAT(context.output(), !(ContainsSubstring("s_nop")));
            }

            SECTION("Test v_permlane, only on GFX950 ")
            {
                // Tests for v_permlane*, only on GFX950
                if(not arch.isCDNA35GPU())
                {
                    SKIP("v_permlane only available on GFX950");
                }

                auto context = TestContext::ForTarget(arch);
                auto v = createRegisters(context, Register::Type::Vector, DataType::UInt32, 3);
                auto s = createRegisters(context, Register::Type::Scalar, DataType::UInt32, 2);

                std::vector<Instruction> insts
                    = {Instruction("v_cmpx_eq_u32", {}, {s[0], v[0]}, {}, ""),
                       Instruction::InoutInstruction("v_permlane32_swap_b32", {v[1], v[2]}, {}, ""),
                       Instruction("s_endpgm", {}, {}, {}, "")};

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1], 4);

                CHECK_THAT(context.output(), ContainsSubstring("s_nop 3"));
            }
        }
    }

    TEST_CASE("OPSELxSDWA94X", "[observer]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            if(arch.isRDNAGPU())
            {
                SKIP("RDNA not supported yet");
            }

            auto context = TestContext::ForTarget(arch);
            auto v       = createRegisters(context, Register::Type::Vector, DataType::UInt32, 2);

            // OPSEL
            std::vector<Instruction> insts
                = {Instruction("v_fma_f16", {v[1]}, {v[0], v[0], v[0]}, {"op_sel:[0,1,1]"}, ""),
                   Instruction("v_mov_b32", {v[0]}, {v[1]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            if(arch.isCDNA1GPU() || arch.isCDNA2GPU())
            {
                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);

                CHECK_THAT(context.output(), !(ContainsSubstring("s_nop")));
            }
            else
            {
                // NOPs are required on 94X arch
                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1], 1);

                CHECK_THAT(context.output(), ContainsSubstring("s_nop 0"));
            }
        }
    }

    TEST_CASE("VALUWriteSGPRVMEM", "[observer]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            if(arch.isRDNAGPU())
            {
                SKIP("RDNA not supported yet");
            }

            auto context = TestContext::ForTarget(arch);

            auto v = createRegisters(context, Register::Type::Vector, DataType::UInt32, 2);
            auto s = createRegisters(context, Register::Type::Scalar, DataType::UInt32, 1);

            std::vector<Instruction> insts = {
                Instruction("v_readlane_b32", {s[0]}, {v[0], Register::Value::Literal(0)}, {}, ""),
                Instruction(
                    "buffer_load_dword", {v[1]}, {s[0], Register::Value::Literal(0)}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, "")};
            peekAndSchedule(context, insts[0]);
            peekAndSchedule(context, insts[1], 5);

            CHECK_THAT(context.output(), ContainsSubstring("s_nop 4"));
        }
    }

    TEST_CASE("BufferStoreWrite", "[observer]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            if(arch.isRDNAGPU())
            {
                SKIP("RDNA not supported yet");
            }

            SECTION("NOPs added for buffer_store_dwordx4 followed by VALU")
            {
                auto context = TestContext::ForTarget(arch);
                auto v = createRegisters(context, Register::Type::Vector, DataType::Float, 1, 4)[0];
                auto addr = createRegisters(context, Register::Type::Vector, DataType::Raw32, 1)[0];
                auto a    = createRegisters(
                    context, Register::Type::Accumulator, DataType::Float, 1, 4)[0];
                auto s = createRegisters(context, Register::Type::Scalar, DataType::Raw32, 1, 4)[0];

                std::vector<Instruction> insts
                    = {Instruction("buffer_store_dwordx4",
                                   {},
                                   {v, addr, s, Register::Value::Literal(0)},
                                   {"offen"},
                                   ""),
                       Instruction("v_accvgpr_read", {v->subset({0})}, {a->subset({0})}, {}, ""),
                       Instruction("v_accvgpr_read", {v->subset({1})}, {a->subset({1})}, {}, ""),
                       Instruction("v_accvgpr_read", {v->subset({2})}, {a->subset({2})}, {}, ""),
                       Instruction("v_accvgpr_read", {v->subset({3})}, {a->subset({3})}, {}, ""),
                       Instruction("s_endpgm", {}, {}, {}, "")};

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(
                    context, insts[1], (arch.isCDNA1GPU() || arch.isCDNA2GPU()) ? 1 : 2);
                peekAndSchedule(context, insts[2]);
                peekAndSchedule(context, insts[3]);
                peekAndSchedule(context, insts[4]);
                peekAndSchedule(context, insts[5]);

                CHECK_THAT(context.output(), ContainsSubstring("s_nop"));
            }

            SECTION("No NOPs when soffset is an SGPR")
            {
                auto context = TestContext::ForTarget(arch);
                auto v = createRegisters(context, Register::Type::Vector, DataType::Float, 1, 4)[0];
                auto addr = createRegisters(context, Register::Type::Vector, DataType::Raw32, 1)[0];
                auto a    = createRegisters(
                    context, Register::Type::Accumulator, DataType::Float, 1, 4)[0];
                auto s = createRegisters(context, Register::Type::Scalar, DataType::Raw32, 1, 4)[0];
                auto soffset
                    = createRegisters(context, Register::Type::Scalar, DataType::Raw32, 1, 1)[0];

                std::vector<Instruction> insts = {
                    Instruction("buffer_store_dwordx4", {}, {v, addr, s, soffset}, {"offen"}, ""),
                    Instruction("v_accvgpr_read", {v->subset({0})}, {a->subset({0})}, {}, ""),
                    Instruction("v_accvgpr_read", {v->subset({1})}, {a->subset({1})}, {}, ""),
                    Instruction("v_accvgpr_read", {v->subset({2})}, {a->subset({2})}, {}, ""),
                    Instruction("v_accvgpr_read", {v->subset({3})}, {a->subset({3})}, {}, ""),
                    Instruction("s_endpgm", {}, {}, {}, "")};

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);
                peekAndSchedule(context, insts[2]);
                peekAndSchedule(context, insts[3]);
                peekAndSchedule(context, insts[4]);
                peekAndSchedule(context, insts[5]);

                CHECK_THAT(context.output(), !ContainsSubstring("s_nop"));
            }

            SECTION("No NOPs after buffer_store_dwordx4")
            {
                auto context = TestContext::ForTarget(arch);
                auto v1
                    = createRegisters(context, Register::Type::Vector, DataType::Float, 1, 4)[0];
                auto v2
                    = createRegisters(context, Register::Type::Vector, DataType::Float, 1, 4)[0];
                auto addr = createRegisters(context, Register::Type::Vector, DataType::Raw32, 1)[0];
                auto a    = createRegisters(
                    context, Register::Type::Accumulator, DataType::Float, 1, 4)[0];
                auto s = createRegisters(context, Register::Type::Scalar, DataType::Raw32, 1, 4)[0];
                auto soffset
                    = createRegisters(context, Register::Type::Scalar, DataType::Raw32, 1, 1)[0];

                std::vector<Instruction> insts = {
                    Instruction("buffer_store_dwordx4", {}, {v1, addr, s, soffset}, {"offen"}, ""),
                    Instruction("v_accvgpr_read", {v2->subset({0})}, {a->subset({0})}, {}, ""),
                    Instruction("v_accvgpr_read", {v2->subset({1})}, {a->subset({1})}, {}, ""),
                    Instruction("v_accvgpr_read", {v2->subset({2})}, {a->subset({2})}, {}, ""),
                    Instruction("v_accvgpr_read", {v2->subset({3})}, {a->subset({3})}, {}, ""),
                    Instruction("s_endpgm", {}, {}, {}, "")};

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);
                peekAndSchedule(context, insts[2]);
                peekAndSchedule(context, insts[3]);
                peekAndSchedule(context, insts[4]);
                peekAndSchedule(context, insts[5]);

                CHECK_THAT(context.output(), !ContainsSubstring("s_nop"));
            }
        }
    }

}
