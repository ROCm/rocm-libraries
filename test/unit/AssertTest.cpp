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

#include "gtest/gtest.h"

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/AssertOpKinds.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include "GPUContextFixture.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;
using namespace rocRoller::KernelGraph::ControlGraph;
using ::testing::HasSubstr;

namespace AssertTest
{
    class GPU_AssertTest : public GPUContextFixtureParam<std::tuple<AssertOpKind, std::string>>
    {
    };

    TEST_P(GPU_AssertTest, GPU_Assert)
    {
        auto const& arch = m_context->targetArchitecture();
        auto        gpu  = arch.target();
        if(gpu.isCDNA1GPU())
            GTEST_SKIP() << "Skipping GPU AssertTest for CDNA1GPU" << gpu.toString();

        AssertOpKind assertOpKind;
        std::string  outputMsg;
        std::tie(assertOpKind, outputMsg) = std::get<1>(GetParam());

        ::testing::FLAGS_gtest_death_test_style = "threadsafe";

        if(assertOpKind == AssertOpKind::Count)
        {
            EXPECT_THAT(
                [&]() {
                    rocRoller::KernelGraph::KernelGraph kgraph;
                    auto                                k = m_context->kernel();
                    k->setKernelDimensions(1);
                    setKernelOptions({.assertOpKind = assertOpKind});
                    auto assertOp = kgraph.control.addElement(AssertOp{});
                    m_context->schedule(rocRoller::KernelGraph::generate(kgraph, k));
                },
                ::testing::ThrowsMessage<FatalError>(HasSubstr(outputMsg)));
        }
        else
        {
            auto one  = Expression::literal(1);
            auto zero = Expression::literal(0);

            rocRoller::KernelGraph::KernelGraph kgraph;

            auto k = m_context->kernel();
            k->setKernelDimensions(1);
            k->setWorkitemCount({one, one, one});
            k->setWorkgroupSize({1, 1, 1});
            setKernelOptions({.assertOpKind = assertOpKind});
            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());
            k->setDynamicSharedMemBytes(zero);

            auto                    testReg = kgraph.coordinates.addElement(Linear());
            Expression::DataFlowTag testRegTag{testReg, Register::Type::Scalar, DataType::UInt32};
            auto testRegExpr = std::make_shared<Expression::Expression>(testRegTag);

            int setToZero = kgraph.control.addElement(Assign{Register::Type::Scalar, zero});

            kgraph.mapper.connect(setToZero, testReg, NaryArgument::DEST);

            auto assertOp = kgraph.control.addElement(AssertOp{"Assert Test", testRegExpr == one});

            auto assignOne = kgraph.control.addElement(Assign{Register::Type::Scalar, one});

            auto kernelNode = kgraph.control.addElement(Kernel());
            kgraph.control.addElement(Body(), {kernelNode}, {setToZero});
            kgraph.control.addElement(Sequence(), {setToZero}, {assertOp});
            kgraph.control.addElement(Sequence(), {assertOp}, {assignOne});

            m_context->schedule(rocRoller::KernelGraph::generate(kgraph, k));

            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
            if(arch.HasCapability(GPUCapability::WorkgroupIdxViaTTMP))
            {
                EXPECT_THAT(output(), testing::HasSubstr("s_mov_b32 s2, 0"));
            }
            else
            {
                EXPECT_THAT(output(), testing::HasSubstr("s_mov_b32 s3, 0"));
            }
            if(assertOpKind != AssertOpKind::NoOp)
            {
                if(arch.HasCapability(GPUCapability::WorkgroupIdxViaTTMP))
                {
                    EXPECT_THAT(output(), testing::HasSubstr("s_cmp_eq_i32 s2, 1"));
                }
                else
                {
                    EXPECT_THAT(output(), testing::HasSubstr("s_cmp_eq_i32 s3, 1"));
                }
                EXPECT_THAT(output(), testing::HasSubstr("s_cbranch_scc1"));
                EXPECT_THAT(output(), testing::HasSubstr("AssertFailed"));
                EXPECT_THAT(output(),
                            testing::HasSubstr(
                                fmt::format("// (op {}) Lock for Assert Assert Test", assertOp)));
                EXPECT_THAT(output(),
                            testing::HasSubstr(
                                fmt::format("// (op {}) Unlock for Assert Assert Test", assertOp)));
                if(assertOpKind == AssertOpKind::STrap)
                {
                    EXPECT_THAT(output(), testing::HasSubstr("s_trap 2"));
                }
                else
                { // MEMORY_VIOLATION
                    auto const HasVMov64 = arch.HasCapability(GPUCapability::v_mov_b64);
                    if(HasVMov64)
                    {
                        EXPECT_THAT(output(), testing::HasSubstr("v_mov_b64 v[2:3], 0"));
                    }
                    else
                    {
                        EXPECT_THAT(output(), testing::HasSubstr("v_mov_b32 v2, 0"));
                        EXPECT_THAT(output(), testing::HasSubstr("v_mov_b32 v3, 0"));
                    }
                    EXPECT_THAT(output(), testing::HasSubstr("v_mov_b32 v1, 42"));
                    EXPECT_THAT(output(), testing::HasSubstr("global_store_dword v[2:3], v1 off"));
                }
                EXPECT_THAT(output(), testing::HasSubstr("AssertPassed"));
                if(arch.HasCapability(GPUCapability::WorkgroupIdxViaTTMP))
                {
                    EXPECT_THAT(output(), testing::HasSubstr("s_mov_b32 s3, 1"));
                }
                else
                {
                    EXPECT_THAT(output(), testing::HasSubstr("s_mov_b32 s4, 1"));
                }
            }
            else
            {
                EXPECT_THAT(output(), testing::HasSubstr("AssertOpKind == NoOp"));
            }

            if(isLocalDevice())
            {
                KernelArguments kargs;

                KernelInvocation kinv;
                kinv.workitemCount = {1, 1, 1};
                kinv.workgroupSize = {1, 1, 1};

                auto executableKernel = m_context->instructions()->getExecutableKernel();

                const auto runTest = [&]() {
                    executableKernel->executeKernel(kargs, kinv);
                    // Need to wait for signal, otherwise child process may terminate before signal is sent
                    (void)hipDeviceSynchronize();
                };
                if(assertOpKind != AssertOpKind::NoOp)
                {
                    EXPECT_EXIT({ runTest(); }, ::testing::KilledBySignal(SIGABRT), outputMsg);
                }
                else
                {
                    runTest();
                }
            }
        }
    }

    TEST_P(GPU_AssertTest, GPU_UnconditionalAssert)
    {
        auto const& arch = m_context->targetArchitecture();
        auto        gpu  = arch.target();
        if(gpu.isCDNA1GPU())
            GTEST_SKIP() << "Skipping GPU AssertTest for CDNA1GPU" << gpu.toString();

        AssertOpKind assertOpKind;
        std::string  outputMsg;
        std::tie(assertOpKind, outputMsg) = std::get<1>(GetParam());

        ::testing::FLAGS_gtest_death_test_style = "threadsafe";

        if(assertOpKind == AssertOpKind::Count)
        {
            EXPECT_THAT(
                [&]() {
                    rocRoller::KernelGraph::KernelGraph kgraph;
                    auto                                k = m_context->kernel();
                    k->setKernelDimensions(1);
                    setKernelOptions({.assertOpKind = assertOpKind});
                    auto assertOp = kgraph.control.addElement(AssertOp{});
                    m_context->schedule(rocRoller::KernelGraph::generate(kgraph, k));
                },
                ::testing::ThrowsMessage<FatalError>(HasSubstr(outputMsg)));
        }
        else
        {
            auto one  = Expression::literal(1);
            auto zero = Expression::literal(0);

            rocRoller::KernelGraph::KernelGraph kgraph;

            auto k = m_context->kernel();
            k->setKernelDimensions(1);
            k->setWorkitemCount({one, one, one});
            k->setWorkgroupSize({1, 1, 1});
            setKernelOptions({.assertOpKind = assertOpKind});
            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());
            k->setDynamicSharedMemBytes(zero);

            int setToZero = kgraph.control.addElement(Assign{Register::Type::Scalar, zero});

            auto assertOp = kgraph.control.addElement(AssertOp{"Unconditional Assert"});

            auto assignOne = kgraph.control.addElement(Assign{Register::Type::Scalar, one});

            auto kernelNode = kgraph.control.addElement(Kernel());
            kgraph.control.addElement(Body(), {kernelNode}, {setToZero});
            kgraph.control.addElement(Sequence(), {setToZero}, {assertOp});
            kgraph.control.addElement(Sequence(), {assertOp}, {assignOne});

            m_context->schedule(rocRoller::KernelGraph::generate(kgraph, k));

            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
            if(assertOpKind != AssertOpKind::NoOp)
            {
                if(assertOpKind == AssertOpKind::STrap)
                {
                    EXPECT_THAT(output(), testing::HasSubstr("s_trap 2"));
                }
                else
                { // MEMORY_VIOLATION
                    auto const HasVMov64
                        = m_context->targetArchitecture().HasCapability(GPUCapability::v_mov_b64);
                    if(HasVMov64)
                    {
                        EXPECT_THAT(output(), testing::HasSubstr("v_mov_b64 v[2:3], 0"));
                    }
                    else
                    {
                        EXPECT_THAT(output(), testing::HasSubstr("v_mov_b32 v2, 0"));
                        EXPECT_THAT(output(), testing::HasSubstr("v_mov_b32 v3, 0"));
                    }
                    EXPECT_THAT(output(), testing::HasSubstr("v_mov_b32 v1, 42"));
                    EXPECT_THAT(output(), testing::HasSubstr("global_store_dword v[2:3], v1 off"));
                }
            }
            else
            {
                EXPECT_THAT(output(), testing::HasSubstr("AssertOpKind == NoOp"));
            }

            if(isLocalDevice())
            {
                KernelArguments kargs;

                KernelInvocation kinv;
                kinv.workitemCount = {1, 1, 1};
                kinv.workgroupSize = {1, 1, 1};

                auto executableKernel = m_context->instructions()->getExecutableKernel();

                const auto runTest = [&]() {
                    executableKernel->executeKernel(kargs, kinv);
                    // Need to wait for signal, otherwise child process may terminate before signal is sent
                    (void)hipDeviceSynchronize();
                };
                if(assertOpKind != AssertOpKind::NoOp)
                {
                    EXPECT_EXIT({ runTest(); }, ::testing::KilledBySignal(SIGABRT), outputMsg);
                }
                else
                {
                    runTest();
                }
            }
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        AssertTest,
        GPU_AssertTest,
        ::testing::Combine(
            supportedISAValues(),
            ::testing::Values(std::tuple(AssertOpKind::MemoryViolation, "Memory access fault"),
                              std::tuple(AssertOpKind::STrap, "HSA_STATUS_ERROR_EXCEPTION"),
                              std::tuple(AssertOpKind::NoOp, "AssertOpKind == NoOp"),
                              std::tuple(AssertOpKind::Count, "Invalid AssertOpKind"))));
}
