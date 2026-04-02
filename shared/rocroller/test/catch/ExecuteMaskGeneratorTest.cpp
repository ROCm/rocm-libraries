// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "TestContext.hpp"

#include <common/Utilities.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_runtime.h>
#endif

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;
using namespace rocRoller::KernelGraph::ControlGraph;
using namespace Catch::Matchers;

namespace ExecuteMaskGeneratorTest
{
    namespace kg = rocRoller::KernelGraph;

    /**
     * Builds a minimal KernelGraph for testing Exec and BranchAndExec ConditionalOp modes.
     *
     * Graph structure:
     *   Kernel
     *     +-- Body -> conditional (ConditionalOp with workitem ID odd/even condition)
     *           +-- Body -> trueOp  (assigns 1u to destVGPR)
     *           +-- Else -> falseOp (assigns 2u to destVGPR) [if withElseBody]
     *
     * The condition ((workitemId & 1u) == 0u) checks whether the workitem ID is even.
     * Comparing a VGPR expression to a scalar literal produces a VCC result, as required
     * by Exec and BranchAndExec modes.
     */
    kg::KernelGraph
        buildConditionalGraph(OpMode mode, bool withElseBody, Register::ValuePtr workitemIdReg)
    {
        kg::KernelGraph kgraph;

        auto zero = Expression::literal(0u);
        auto one  = Expression::literal(1u);
        auto two  = Expression::literal(2u);

        // Workitem ID VGPR expression: (workitemId & 1u) == 0u is true for even lanes.
        auto workitemId = workitemIdReg->expression();
        auto isEven     = (workitemId & one) == zero;

        // Destination VGPR for body and else assigns.
        auto destVGPR = kgraph.coordinates.addElement(VGPR());

        auto initOp = kgraph.control.addElement(Assign{Register::Type::Vector, zero});
        kgraph.mapper.connect(initOp, destVGPR, NaryArgument::DEST);

        // True body: assign 1 to destVGPR.
        auto trueOp = kgraph.control.addElement(Assign{Register::Type::Vector, one});
        kgraph.mapper.connect(trueOp, destVGPR, NaryArgument::DEST);

        // ConditionalOp: (workitemId & 1u) == 0u is a VGPR comparison whose result lands
        // in VCC, satisfying the Exec/BranchAndExec requirement.
        auto conditional
            = kgraph.control.addElement(ConditionalOp{isEven, mode, "Exec Conditional"});

        auto kernel = kgraph.control.addElement(Kernel());
        kgraph.control.addElement(Body(), {kernel}, {initOp});
        kgraph.control.addElement(Sequence(), {initOp}, {conditional});
        kgraph.control.addElement(Body(), {conditional}, {trueOp});

        if(withElseBody)
        {
            // False body: assign 2 to destVGPR.
            auto falseOp = kgraph.control.addElement(Assign{Register::Type::Vector, two});
            kgraph.mapper.connect(falseOp, destVGPR, NaryArgument::DEST);
            kgraph.control.addElement(Else(), {conditional}, {falseOp});
        }

        return kgraph;
    }

    /**
     * Like buildConditionalGraph, but also adds a StoreVGPR that writes the per-lane
     * result (1u for true lanes, 2u for false lanes) to a global output buffer at
     * output[workitemId].  Requires a kernel argument named "output" (UInt32 pointer).
     */
    kg::KernelGraph buildConditionalGraphWithStore(OpMode             mode,
                                                   bool               withElseBody,
                                                   Register::ValuePtr workitemIdReg,
                                                   uint32_t           wavefrontSize)
    {
        kg::KernelGraph kgraph;

        auto zero = Expression::literal(0u);
        auto one  = Expression::literal(1u);
        auto two  = Expression::literal(2u);

        auto workitemId = workitemIdReg->expression();
        auto isEven     = (workitemId & one) == zero;

        // Destination VGPR for body and else assigns.
        auto destVGPR = kgraph.coordinates.addElement(VGPR());

        // Pre-initialize destVGPR to 0 so lanes that skip the true body have a
        // known value.
        auto initOp = kgraph.control.addElement(Assign{Register::Type::Vector, zero});
        kgraph.mapper.connect(initOp, destVGPR, NaryArgument::DEST);

        // True body: assign 1 to destVGPR.
        auto trueOp = kgraph.control.addElement(Assign{Register::Type::Vector, one});
        kgraph.mapper.connect(trueOp, destVGPR, NaryArgument::DEST);

        auto conditional
            = kgraph.control.addElement(ConditionalOp{isEven, mode, "Exec Conditional"});

        auto kernel = kgraph.control.addElement(Kernel());
        kgraph.control.addElement(Body(), {kernel}, {initOp});
        kgraph.control.addElement(Sequence(), {initOp}, {conditional});
        kgraph.control.addElement(Body(), {conditional}, {trueOp});

        if(withElseBody)
        {
            // False body: assign 2 to destVGPR.
            auto falseOp = kgraph.control.addElement(Assign{Register::Type::Vector, two});
            kgraph.mapper.connect(falseOp, destVGPR, NaryArgument::DEST);
            kgraph.control.addElement(Else(), {conditional}, {falseOp});
        }

        // Store each lane's result to output[workitemId].
        auto wfSizeExpr = Expression::literal(wavefrontSize);
        auto workitem0  = kgraph.coordinates.addElement(Workitem(0, wfSizeExpr));
        auto user       = kgraph.coordinates.addElement(User({}, "output"));
        kgraph.coordinates.addElement(PassThrough(), {workitem0}, {user});
        kgraph.coordinates.addElement(PassThrough(), {user}, {destVGPR});

        auto storeOp = kgraph.control.addElement(StoreVGPR{});
        kgraph.mapper.connect<User>(storeOp, user);
        kgraph.mapper.connect<VGPR>(storeOp, destVGPR);
        kgraph.control.addElement(Sequence(), {conditional}, {storeOp});

        return kgraph;
    }

    TEST_CASE("ExecuteMaskGenerator - Exec mode, true body only",
              "[exec-mask][codegen][kernel-graph]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            auto testCtx = TestContext::ForTarget(arch);
            auto ctx     = testCtx.get();
            auto k       = ctx->kernel();

            ctx->schedule(k->preamble());
            ctx->schedule(k->prolog());
            auto kgraph = buildConditionalGraph(OpMode::Exec, false, k->workitemIndex()[0]);
            ctx->schedule(rocRoller::KernelGraph::generate(kgraph, k));

            auto output = testCtx.output();

            if(k->wavefront_size() == 64)
            {
                CHECK_THAT(output, ContainsSubstring("s_and_saveexec_b64"));
            }
            else
            {
                CHECK_THAT(output, ContainsSubstring("s_and_saveexec_b32"));
            }

            // No else body: complement mask instruction must not be emitted.
            CHECK_THAT(output, !ContainsSubstring("s_andn1_saveexec"));

            // Exec mode uses EXEC masking, not scalar SCC-based branches.
            CHECK_THAT(output, !ContainsSubstring("s_cbranch_scc0"));
            // Exec mode uses EXEC masking, not VCC-based branches.
            CHECK_THAT(output, !ContainsSubstring("s_cbranch_vcc"));
        }
    }

    TEST_CASE("ExecuteMaskGenerator - Exec mode, true and else bodies",
              "[exec-mask][codegen][kernel-graph]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            auto testCtx = TestContext::ForTarget(arch);
            auto ctx     = testCtx.get();
            auto k       = ctx->kernel();

            ctx->schedule(k->preamble());
            ctx->schedule(k->prolog());
            auto kgraph = buildConditionalGraph(OpMode::Exec, true, k->workitemIndex()[0]);
            ctx->schedule(rocRoller::KernelGraph::generate(kgraph, k));

            auto output = testCtx.output();

            if(k->wavefront_size() == 64)
            {
                CHECK_THAT(output, ContainsSubstring("s_and_saveexec_b64"));
                CHECK_THAT(output, ContainsSubstring("s_andn1_saveexec_b64"));
            }
            else
            {
                CHECK_THAT(output, ContainsSubstring("s_and_saveexec_b32"));
                CHECK_THAT(output, ContainsSubstring("s_andn1_saveexec_b32"));
            }

            // Exec mode uses EXEC masking, not scalar SCC-based branches.
            CHECK_THAT(output, !ContainsSubstring("s_cbranch_scc0"));
            // Exec mode uses EXEC masking, not VCC-based branches.
            CHECK_THAT(output, !ContainsSubstring("s_cbranch_vcc"));
        }
    }

    TEST_CASE("ExecuteMaskGenerator - BranchAndExec mode, true body only",
              "[exec-mask][codegen][kernel-graph]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            auto testCtx = TestContext::ForTarget(arch);
            auto ctx     = testCtx.get();
            auto k       = ctx->kernel();

            ctx->schedule(k->preamble());
            ctx->schedule(k->prolog());
            auto kgraph
                = buildConditionalGraph(OpMode::BranchAndExec, false, k->workitemIndex()[0]);
            ctx->schedule(rocRoller::KernelGraph::generate(kgraph, k));

            auto output = testCtx.output();

            if(k->wavefront_size() == 64)
            {
                CHECK_THAT(output, ContainsSubstring("s_and_saveexec_b64"));
            }
            else
            {
                CHECK_THAT(output, ContainsSubstring("s_and_saveexec_b32"));
            }

            // BranchAndExec emits EXECZ-based branches and corresponding labels.
            CHECK_THAT(output, ContainsSubstring("ELSE_Conditional_EXECZ_"));
            CHECK_THAT(output, ContainsSubstring("EXIT_Conditional_EXECZ_"));
            CHECK_THAT(output, ContainsSubstring("s_cbranch_execz"));
            // Unconditional branch from the end of the true body to the exit label.
            CHECK_THAT(output, ContainsSubstring("s_branch"));

            // BranchAndExec does not use scalar SCC-based branches.
            CHECK_THAT(output, !ContainsSubstring("s_cbranch_scc0"));
            // BranchAndExec does not use VCC-based branches.
            CHECK_THAT(output, !ContainsSubstring("s_cbranch_vcc"));
        }
    }

    TEST_CASE("ExecuteMaskGenerator - BranchAndExec mode, true and else bodies",
              "[exec-mask][codegen][kernel-graph]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            auto testCtx = TestContext::ForTarget(arch);
            auto ctx     = testCtx.get();
            auto k       = ctx->kernel();

            ctx->schedule(k->preamble());
            ctx->schedule(k->prolog());
            auto kgraph = buildConditionalGraph(OpMode::BranchAndExec, true, k->workitemIndex()[0]);
            ctx->schedule(rocRoller::KernelGraph::generate(kgraph, k));

            auto output = testCtx.output();

            if(k->wavefront_size() == 64)
            {
                CHECK_THAT(output, ContainsSubstring("s_and_saveexec_b64"));
                CHECK_THAT(output, ContainsSubstring("s_andn1_saveexec_b64"));
            }
            else
            {
                CHECK_THAT(output, ContainsSubstring("s_and_saveexec_b32"));
                CHECK_THAT(output, ContainsSubstring("s_andn1_saveexec_b32"));
            }

            // BranchAndExec emits EXECZ-based branches and corresponding labels.
            CHECK_THAT(output, ContainsSubstring("ELSE_Conditional_EXECZ_"));
            CHECK_THAT(output, ContainsSubstring("EXIT_Conditional_EXECZ_"));
            CHECK_THAT(output, ContainsSubstring("s_cbranch_execz"));
            // Unconditional branch from the end of the true body to the exit label.
            CHECK_THAT(output, ContainsSubstring("s_branch"));

            // BranchAndExec does not use scalar SCC-based branches.
            CHECK_THAT(output, !ContainsSubstring("s_cbranch_scc0"));
            // BranchAndExec does not use VCC-based branches.
            CHECK_THAT(output, !ContainsSubstring("s_cbranch_vcc"));
        }
    }

    // Helper used by the GPU execution tests below.
    void runGPUExecutionTest(OpMode mode, bool withElseBody)
    {
        auto testCtx = TestContext::ForTestDevice();
        auto ctx     = testCtx.get();
        auto k       = ctx->kernel();

        auto wfSize = static_cast<uint32_t>(k->wavefront_size());

        k->addArgument(
            {"output", {DataType::UInt32, PointerType::PointerGlobal}, DataDirection::WriteOnly});
        k->setKernelDimensions(1);
        k->setWorkitemCount(
            {Expression::literal(wfSize), Expression::literal(1u), Expression::literal(1u)});
        k->setWorkgroupSize({wfSize, 1, 1});

        ctx->schedule(k->preamble());
        ctx->schedule(k->prolog());
        auto kgraph
            = buildConditionalGraphWithStore(mode, withElseBody, k->workitemIndex()[0], wfSize);
        ctx->schedule(rocRoller::KernelGraph::generate(kgraph, k));
        ctx->schedule(k->postamble());
        ctx->schedule(k->amdgpu_metadata());

        if(ctx->hipDeviceIndex() < 0)
            return;

        auto deviceOutput = make_shared_device<uint32_t>(wfSize, 0u);

        KernelArguments kargs(false);
        kargs.append("output", deviceOutput.get());

        KernelInvocation kinv;
        kinv.workitemCount = {wfSize, 1, 1};
        kinv.workgroupSize = {wfSize, 1, 1};

        ctx->instructions()->getExecutableKernel()->executeKernel(kargs, kinv);

        std::vector<uint32_t> hostOutput(wfSize);
        REQUIRE_THAT(
            hipMemcpy(
                hostOutput.data(), deviceOutput.get(), wfSize * sizeof(uint32_t), hipMemcpyDefault),
            HasHipSuccess(0));

        // Even lanes (workitemId & 1 == 0) execute the true body -> 1.
        // Odd lanes:
        //   Exec mode:           else body runs per-lane -> 2, or skipped -> 0.
        //   BranchAndExec mode:  else label only reached when EXEC==0 (no true lanes at all);
        //                        with mixed lanes some true lanes exist, so else body never
        //                        runs -> odd lanes retain their pre-initialized value of 0.
        bool                  elseBodyRunsPerLane = withElseBody && (mode == OpMode::Exec);
        std::vector<uint32_t> expected(wfSize);
        for(uint32_t i = 0; i < wfSize; ++i)
            expected[i] = (i % 2 == 0) ? 1u : (elseBodyRunsPerLane ? 2u : 0u);

        CHECK(hostOutput == expected);
    }

    TEST_CASE("ExecuteMaskGenerator - Exec mode, true body only (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTest(OpMode::Exec, false);
    }

    TEST_CASE("ExecuteMaskGenerator - Exec mode, true and else bodies (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTest(OpMode::Exec, true);
    }

    TEST_CASE("ExecuteMaskGenerator - BranchAndExec mode, true body only (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTest(OpMode::BranchAndExec, false);
    }

    TEST_CASE("ExecuteMaskGenerator - BranchAndExec mode, true and else bodies (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTest(OpMode::BranchAndExec, true);
    }

} // namespace ExecuteMaskGeneratorTest
