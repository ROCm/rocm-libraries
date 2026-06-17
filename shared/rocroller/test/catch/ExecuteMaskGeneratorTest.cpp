// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <sstream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "TestContext.hpp"

#include <common/Utilities.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/GPUArchitecture/GPUCapability.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

#include <hip/hip_runtime.h>

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;
using namespace rocRoller::KernelGraph::ControlGraph;
using namespace Catch::Matchers;

namespace ExecuteMaskGeneratorTest
{
    namespace kg = rocRoller::KernelGraph;

    // Remove lines belonging to the workgroup-cluster prolog branch so that
    // codegen assertions can reject unexpected s_cbranch_vcc without tripping
    // on the legitimate s_cbranch_vccz emitted by the cluster detection in
    // AssemblyKernel::prolog().  Returns the input unchanged when the target
    // does not support workgroup clusters.
    std::string StripClusterBranchLines(const std::string& asmOutput, ContextPtr ctx)
    {
        if(!ctx->targetArchitecture().HasCapability(GPUCapability::HasWorkgroupClusters))
            return asmOutput;

        std::istringstream stream(asmOutput);
        std::string        result;
        std::string        line;
        while(std::getline(stream, line))
        {
            if(line.find("NotInCluster") == std::string::npos)
            {
                result += line;
                result += '\n';
            }
        }
        return result;
    }

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
    kg::KernelGraph buildConditionalGraph(ConditionalMode    mode,
                                          bool               withElseBody,
                                          Register::ValuePtr workitemIdReg)
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
            auto kgraph
                = buildConditionalGraph(ConditionalMode::Exec, false, k->workitemIndex()[0]);
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
            CHECK_THAT(StripClusterBranchLines(output, ctx), !ContainsSubstring("s_cbranch_vcc"));
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
            auto kgraph = buildConditionalGraph(ConditionalMode::Exec, true, k->workitemIndex()[0]);
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
            CHECK_THAT(StripClusterBranchLines(output, ctx), !ContainsSubstring("s_cbranch_vcc"));
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
            auto kgraph = buildConditionalGraph(
                ConditionalMode::BranchAndExec, false, k->workitemIndex()[0]);
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
            CHECK_THAT(StripClusterBranchLines(output, ctx), !ContainsSubstring("s_cbranch_vcc"));
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
            auto kgraph = buildConditionalGraph(
                ConditionalMode::BranchAndExec, true, k->workitemIndex()[0]);
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
            CHECK_THAT(StripClusterBranchLines(output, ctx), !ContainsSubstring("s_cbranch_vcc"));
        }
    }

    /**
     * Like buildConditionalGraph, but also adds a StoreVGPR that writes the per-lane
     * result (1u for true lanes, 2u for false lanes) to a global output buffer at
     * output[workitemId].  Requires a kernel argument named "output" (UInt32 pointer).
     *
     * Graph structure:
     *   Control graph:
     *     Kernel
     *       +-- Body     -> initOp      (assign 0u to destVGPR)
     *       +-- Sequence -> conditional (ConditionalOp: (workitemId & 1u) == 0u)
     *             +-- Body     -> trueOp  (assign 1u to destVGPR)
     *             +-- Else     -> falseOp (assign 2u to destVGPR) [if withElseBody]
     *             +-- Sequence -> storeOp (StoreVGPR: destVGPR -> output[workitemId])
     *   Coordinate graph:
     *     Workitem(0) --PassThrough--> User("output") --PassThrough--> destVGPR (VGPR)
     */
    kg::KernelGraph buildConditionalGraphWithStore(ConditionalMode    mode,
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

    // Helper used by the GPU execution tests below.
    void runGPUExecutionTest(ConditionalMode mode, bool withElseBody)
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
            SKIP("No HIP device present.");

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
        bool                  elseBodyRunsPerLane = withElseBody && (mode == ConditionalMode::Exec);
        std::vector<uint32_t> expected(wfSize);
        for(uint32_t i = 0; i < wfSize; ++i)
            expected[i] = (i % 2 == 0) ? 1u : (elseBodyRunsPerLane ? 2u : 0u);

        CHECK(hostOutput == expected);
    }

    TEST_CASE("ExecuteMaskGenerator - Exec mode, true body only (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTest(ConditionalMode::Exec, false);
    }

    TEST_CASE("ExecuteMaskGenerator - Exec mode, true and else bodies (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTest(ConditionalMode::Exec, true);
    }

    TEST_CASE("ExecuteMaskGenerator - BranchAndExec mode, true body only (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTest(ConditionalMode::BranchAndExec, false);
    }

    TEST_CASE("ExecuteMaskGenerator - BranchAndExec mode, true and else bodies (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTest(ConditionalMode::BranchAndExec, true);
    }

    /**
     * Like buildConditionalGraphWithStore, but uses a condition that is false for every
     * active lane so that s_and_saveexec_* produces EXEC==0 and the EXECZ branch is taken.
     *
     * Condition: workitemId >= wavefrontSize
     * workitemId is always in [0, wavefrontSize), so this is always false for every lane,
     * VCC==0, and EXECZ is set after s_and_saveexec_*.
     *
     * Graph structure:
     *   Kernel
     *     +-- Body     -> initOp      (assign 0u to destVGPR)
     *     +-- Sequence -> conditional (BranchAndExec: alwaysFalse)
     *           +-- Body     -> trueOp  (assign 1u) [skipped: EXECZ branch taken]
     *           +-- Else     -> falseOp (assign 2u) [if withElseBody]
     *           +-- Sequence -> storeOp (StoreVGPR: destVGPR -> output[workitemId])
     *
     * Expected per-lane output:
     *   withElseBody==true:  2u for all lanes (else body runs with EXEC restored to full mask)
     *   withElseBody==false: 0u for all lanes (no body runs; pre-init value retained)
     */
    kg::KernelGraph buildAlwaysFalseConditionalGraphWithStore(ConditionalMode    mode,
                                                              bool               withElseBody,
                                                              Register::ValuePtr workitemIdReg,
                                                              uint32_t           wavefrontSize)
    {
        kg::KernelGraph kgraph;

        auto zero = Expression::literal(0u);
        auto one  = Expression::literal(1u);
        auto two  = Expression::literal(2u);

        // Always-false VCC condition: workitemId >= wavefrontSize.
        // workitemId is always in [0, wfSize), so this is always false for every lane,
        // and s_and_saveexec_* produces EXEC==0 (EXECZ set).
        auto workitemId  = workitemIdReg->expression();
        auto alwaysFalse = workitemId >= Expression::literal(wavefrontSize);

        auto destVGPR = kgraph.coordinates.addElement(VGPR());

        auto initOp = kgraph.control.addElement(Assign{Register::Type::Vector, zero});
        kgraph.mapper.connect(initOp, destVGPR, NaryArgument::DEST);

        auto trueOp = kgraph.control.addElement(Assign{Register::Type::Vector, one});
        kgraph.mapper.connect(trueOp, destVGPR, NaryArgument::DEST);

        auto conditional = kgraph.control.addElement(
            ConditionalOp{alwaysFalse, mode, "AlwaysFalse Conditional"});

        auto kernel = kgraph.control.addElement(Kernel());
        kgraph.control.addElement(Body(), {kernel}, {initOp});
        kgraph.control.addElement(Sequence(), {initOp}, {conditional});
        kgraph.control.addElement(Body(), {conditional}, {trueOp});

        if(withElseBody)
        {
            auto falseOp = kgraph.control.addElement(Assign{Register::Type::Vector, two});
            kgraph.mapper.connect(falseOp, destVGPR, NaryArgument::DEST);
            kgraph.control.addElement(Else(), {conditional}, {falseOp});
        }

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

    // Helper for EXECZ-forced GPU execution tests.
    // The condition is always false for every lane so s_and_saveexec_* produces
    // EXEC==0, EXECZ is set, and the EXECZ branch is taken at runtime.
    void runGPUExecutionTestEXECZ(bool withElseBody)
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
        auto kgraph = buildAlwaysFalseConditionalGraphWithStore(
            ConditionalMode::BranchAndExec, withElseBody, k->workitemIndex()[0], wfSize);
        ctx->schedule(rocRoller::KernelGraph::generate(kgraph, k));
        ctx->schedule(k->postamble());
        ctx->schedule(k->amdgpu_metadata());

        if(ctx->hipDeviceIndex() < 0)
            SKIP("No HIP device present.");

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

        // The condition (workitemId >= wavefrontSize) is false for every lane.
        // s_and_saveexec_* produces EXEC==0 so EXECZ is set and the s_cbranch_execz
        // over the true body is taken; the true body is never executed.
        //
        // With else body: EXEC is restored to the original mask, s_andn1_saveexec_*
        //   computes EXEC AND NOT(vcc) = originalEXEC AND ~0 = originalEXEC, leaving
        //   all lanes active for the else body (assign 2u).
        // Without else body: destVGPR retains its pre-initialized value (0u) for all lanes,
        //   and EXEC is correctly restored to the original mask at the exit label.
        uint32_t              expectedValue = withElseBody ? 2u : 0u;
        std::vector<uint32_t> expected(wfSize, expectedValue);

        CHECK(hostOutput == expected);
    }

    TEST_CASE("ExecuteMaskGenerator - BranchAndExec mode, EXECZ forced (all-lanes false, true body "
              "only) (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestEXECZ(false);
    }

    TEST_CASE("ExecuteMaskGenerator - BranchAndExec mode, EXECZ forced (all-lanes false, with else "
              "body) (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestEXECZ(true);
    }

    /**
     * Like buildConditionalGraphWithStore, but adds a nested ConditionalOp inside the
     * true body that overwrites destVGPR with 4u for lanes whose workitem ID is
     * divisible by 4 (workitemId & 3u == 0u), and optionally 8u for the inner false
     * lanes (even but not div-by-4).
     *
     * True-body execution sequence for a given lane:
     *   1. Assign 1u  to destVGPR  (outer true body).
     *   2. If (workitemId & 3u) == 0u, assign 4u to destVGPR (inner true);
     *      else if withInnerElseBody, assign 8u to destVGPR (inner else).
     *
     * Expected per-lane result:
     *   workitemId % 4 == 0  -> 4u
     *   workitemId % 2 == 0  -> 8u if (innerMode==Exec && withInnerElseBody), else 1u
     *   odd workitemId       -> 2u if (outerMode==Exec && withOuterElseBody), else 0u
     *
     * outerMode and innerMode may differ to test mixed-mode nesting.
     *
     * Graph structure:
     *   Control graph:
     *     Kernel
     *       +-- Body     -> initOp      (assign 0u to destVGPR)
     *       +-- Sequence -> conditional (outer ConditionalOp: (workitemId & 1u) == 0u)
     *             +-- Body     -> trueOp  (assign 1u to destVGPR)
     *                              -> innerConditional (ConditionalOp: (workitemId & 3u) == 0u)
     *                                   +-- Body -> innerTrueOp  (assign 4u to destVGPR)
     *                                   +-- Else -> innerFalseOp (assign 8u to destVGPR) [if withInnerElseBody]
     *             +-- Else     -> falseOp  (assign 2u to destVGPR) [if withOuterElseBody]
     *             +-- Sequence -> storeOp  (StoreVGPR: destVGPR -> output[workitemId])
     *   Coordinate graph:
     *     Workitem(0) --PassThrough--> User("output") --PassThrough--> destVGPR (VGPR)
     */
    kg::KernelGraph buildNestedConditionalGraphWithStore(ConditionalMode    outerMode,
                                                         ConditionalMode    innerMode,
                                                         bool               withOuterElseBody,
                                                         bool               withInnerElseBody,
                                                         Register::ValuePtr workitemIdReg,
                                                         uint32_t           wavefrontSize)
    {
        kg::KernelGraph kgraph;

        auto zero  = Expression::literal(0u);
        auto one   = Expression::literal(1u);
        auto two   = Expression::literal(2u);
        auto three = Expression::literal(3u);
        auto four  = Expression::literal(4u);
        auto eight = Expression::literal(8u);

        auto workitemId = workitemIdReg->expression();
        auto isEven     = (workitemId & one) == zero;
        auto isDivBy4   = (workitemId & three) == zero;

        // Destination VGPR written by all assign ops.
        auto destVGPR = kgraph.coordinates.addElement(VGPR());

        // Pre-initialize destVGPR to 0.
        auto initOp = kgraph.control.addElement(Assign{Register::Type::Vector, zero});
        kgraph.mapper.connect(initOp, destVGPR, NaryArgument::DEST);

        // Outer true body: assign 1 to destVGPR.
        auto trueOp = kgraph.control.addElement(Assign{Register::Type::Vector, one});
        kgraph.mapper.connect(trueOp, destVGPR, NaryArgument::DEST);

        // Inner conditional (nested inside outer true body): assign 4 for div-by-4 lanes.
        auto innerTrueOp = kgraph.control.addElement(Assign{Register::Type::Vector, four});
        kgraph.mapper.connect(innerTrueOp, destVGPR, NaryArgument::DEST);

        auto innerConditional
            = kgraph.control.addElement(ConditionalOp{isDivBy4, innerMode, "DivBy4 Conditional"});
        kgraph.control.addElement(Body(), {innerConditional}, {innerTrueOp});

        if(withInnerElseBody)
        {
            // Inner false body: assign 8 to destVGPR (even but not div-by-4).
            auto innerFalseOp = kgraph.control.addElement(Assign{Register::Type::Vector, eight});
            kgraph.mapper.connect(innerFalseOp, destVGPR, NaryArgument::DEST);
            kgraph.control.addElement(Else(), {innerConditional}, {innerFalseOp});
        }

        auto conditional
            = kgraph.control.addElement(ConditionalOp{isEven, outerMode, "Exec Conditional"});

        auto kernel = kgraph.control.addElement(Kernel());
        kgraph.control.addElement(Body(), {kernel}, {initOp});
        kgraph.control.addElement(Sequence(), {initOp}, {conditional});
        // True body: trueOp then innerConditional.
        kgraph.control.addElement(Body(), {conditional}, {trueOp});
        kgraph.control.addElement(Sequence(), {trueOp}, {innerConditional});

        if(withOuterElseBody)
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

    // Helper used by the nested-conditional GPU execution tests below.
    // outerMode and innerMode may differ to exercise mixed-mode nesting.
    void runGPUExecutionTestNested(ConditionalMode outerMode,
                                   ConditionalMode innerMode,
                                   bool            withOuterElseBody,
                                   bool            withInnerElseBody)
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
        auto kgraph = buildNestedConditionalGraphWithStore(outerMode,
                                                           innerMode,
                                                           withOuterElseBody,
                                                           withInnerElseBody,
                                                           k->workitemIndex()[0],
                                                           wfSize);
        ctx->schedule(rocRoller::KernelGraph::generate(kgraph, k));
        ctx->schedule(k->postamble());
        ctx->schedule(k->amdgpu_metadata());

        if(ctx->hipDeviceIndex() < 0)
            SKIP("No HIP device present.");

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

        // Exec mode else body runs per-lane; BranchAndExec else body only runs when the
        // entire EXEC mask is zero (all active lanes failed the condition), which never
        // happens for the inner condition across a mixed warp, so it effectively never runs.
        bool outerElseRunsPerLane = withOuterElseBody && (outerMode == ConditionalMode::Exec);
        bool innerElseRunsPerLane = withInnerElseBody && (innerMode == ConditionalMode::Exec);

        std::vector<uint32_t> expected(wfSize);
        for(uint32_t i = 0; i < wfSize; ++i)
        {
            if(i % 4 == 0)
                expected[i] = 4u; // outer true + inner true
            else if(i % 2 == 0)
                expected[i] = innerElseRunsPerLane ? 8u : 1u; // outer true + inner false/skipped
            else
                expected[i] = outerElseRunsPerLane ? 2u : 0u; // outer false/skipped
        }

        CHECK(hostOutput == expected);
    }

    TEST_CASE("ExecuteMaskGenerator - Exec mode, nested conditional (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNested(ConditionalMode::Exec, ConditionalMode::Exec, false, false);
    }

    TEST_CASE(
        "ExecuteMaskGenerator - Exec mode, nested conditional with outer else (GPU execution)",
        "[exec-mask][gpu]")
    {
        runGPUExecutionTestNested(ConditionalMode::Exec, ConditionalMode::Exec, true, false);
    }

    TEST_CASE(
        "ExecuteMaskGenerator - Exec mode, nested conditional with inner else (GPU execution)",
        "[exec-mask][gpu]")
    {
        runGPUExecutionTestNested(ConditionalMode::Exec, ConditionalMode::Exec, false, true);
    }

    TEST_CASE("ExecuteMaskGenerator - Exec mode, nested conditional with outer and inner else (GPU "
              "execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNested(ConditionalMode::Exec, ConditionalMode::Exec, true, true);
    }

    TEST_CASE("ExecuteMaskGenerator - BranchAndExec mode, nested conditional (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNested(
            ConditionalMode::BranchAndExec, ConditionalMode::BranchAndExec, false, false);
    }

    TEST_CASE("ExecuteMaskGenerator - BranchAndExec mode, nested conditional with outer else (GPU "
              "execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNested(
            ConditionalMode::BranchAndExec, ConditionalMode::BranchAndExec, true, false);
    }

    TEST_CASE("ExecuteMaskGenerator - BranchAndExec mode, nested conditional with inner else (GPU "
              "execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNested(
            ConditionalMode::BranchAndExec, ConditionalMode::BranchAndExec, false, true);
    }

    TEST_CASE("ExecuteMaskGenerator - BranchAndExec mode, nested conditional with outer and inner "
              "else (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNested(
            ConditionalMode::BranchAndExec, ConditionalMode::BranchAndExec, true, true);
    }

    // Mixed-mode nesting: outer Exec, inner BranchAndExec.
    TEST_CASE("ExecuteMaskGenerator - Exec outer / BranchAndExec inner, nested conditional (GPU "
              "execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNested(
            ConditionalMode::Exec, ConditionalMode::BranchAndExec, false, false);
    }

    TEST_CASE("ExecuteMaskGenerator - Exec outer / BranchAndExec inner, nested conditional with "
              "outer else (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNested(
            ConditionalMode::Exec, ConditionalMode::BranchAndExec, true, false);
    }

    TEST_CASE("ExecuteMaskGenerator - Exec outer / BranchAndExec inner, nested conditional with "
              "inner else (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNested(
            ConditionalMode::Exec, ConditionalMode::BranchAndExec, false, true);
    }

    TEST_CASE("ExecuteMaskGenerator - Exec outer / BranchAndExec inner, nested conditional with "
              "outer and inner else (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNested(
            ConditionalMode::Exec, ConditionalMode::BranchAndExec, true, true);
    }

    // Mixed-mode nesting: outer BranchAndExec, inner Exec.
    TEST_CASE("ExecuteMaskGenerator - BranchAndExec outer / Exec inner, nested conditional (GPU "
              "execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNested(
            ConditionalMode::BranchAndExec, ConditionalMode::Exec, false, false);
    }

    TEST_CASE("ExecuteMaskGenerator - BranchAndExec outer / Exec inner, nested conditional with "
              "outer else (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNested(
            ConditionalMode::BranchAndExec, ConditionalMode::Exec, true, false);
    }

    TEST_CASE("ExecuteMaskGenerator - BranchAndExec outer / Exec inner, nested conditional with "
              "inner else (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNested(
            ConditionalMode::BranchAndExec, ConditionalMode::Exec, false, true);
    }

    TEST_CASE("ExecuteMaskGenerator - BranchAndExec outer / Exec inner, nested conditional with "
              "outer and inner else (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNested(
            ConditionalMode::BranchAndExec, ConditionalMode::Exec, true, true);
    }

    /**
     * Builds a graph with a nested ConditionalOp inside the ELSE body of the outer conditional.
     *
     * Graph structure:
     *   Kernel
     *     +-- Body    -> initOp  (assign 0u)
     *     +-- Sequence -> outer conditional (isEven)
     *           +-- Body -> trueOp         (assign 1u)
     *           +-- Else -> outerFalseOp   (assign 2u)
     *                     -> innerConditional (isLowOdd = i%4==1)
     *                           +-- Body -> innerTrueOp  (assign 5u)
     *                           +-- Else -> innerFalseOp (assign 3u) [if withInnerElseBody]
     *
     * Expected per-lane result (outerMode==Exec):
     *   even         -> 1u  (outer true)
     *   i % 4 == 1   -> 5u  (outer else + inner true:  bit 1 clear among odd lanes)
     *   i % 4 == 3   -> 3u if (innerMode==Exec && withInnerElseBody), else 2u
     *
     * For outerMode==BranchAndExec the outer else body never runs (EXECZ is never set
     * with a mixed warp), so all odd lanes stay at 0u.
     */
    kg::KernelGraph buildNestedInElseConditionalGraphWithStore(ConditionalMode    outerMode,
                                                               ConditionalMode    innerMode,
                                                               bool               withInnerElseBody,
                                                               Register::ValuePtr workitemIdReg,
                                                               uint32_t           wavefrontSize)
    {
        kg::KernelGraph kgraph;

        auto zero  = Expression::literal(0u);
        auto one   = Expression::literal(1u);
        auto two   = Expression::literal(2u);
        auto three = Expression::literal(3u);
        auto five  = Expression::literal(5u);

        auto workitemId = workitemIdReg->expression();
        auto isEven     = (workitemId & one) == zero;
        // Among odd lanes, bit 1 is clear for lanes 1,5,9,... and set for 3,7,11,...
        // This gives a non-trivial partition of the active (odd) lanes in the else body.
        auto isBit1Clear = (workitemId & two) == zero;

        auto destVGPR = kgraph.coordinates.addElement(VGPR());

        auto initOp = kgraph.control.addElement(Assign{Register::Type::Vector, zero});
        kgraph.mapper.connect(initOp, destVGPR, NaryArgument::DEST);

        auto trueOp = kgraph.control.addElement(Assign{Register::Type::Vector, one});
        kgraph.mapper.connect(trueOp, destVGPR, NaryArgument::DEST);

        // Outer else body: assign 2, then run inner conditional.
        auto outerFalseOp = kgraph.control.addElement(Assign{Register::Type::Vector, two});
        kgraph.mapper.connect(outerFalseOp, destVGPR, NaryArgument::DEST);

        auto innerTrueOp = kgraph.control.addElement(Assign{Register::Type::Vector, five});
        kgraph.mapper.connect(innerTrueOp, destVGPR, NaryArgument::DEST);

        auto innerConditional = kgraph.control.addElement(
            ConditionalOp{isBit1Clear, innerMode, "Bit1Clear Conditional"});
        kgraph.control.addElement(Body(), {innerConditional}, {innerTrueOp});

        if(withInnerElseBody)
        {
            // Inner false body: assign 3 to destVGPR (odd lanes where bit 1 is set, i%4==3).
            auto innerFalseOp = kgraph.control.addElement(Assign{Register::Type::Vector, three});
            kgraph.mapper.connect(innerFalseOp, destVGPR, NaryArgument::DEST);
            kgraph.control.addElement(Else(), {innerConditional}, {innerFalseOp});
        }

        auto conditional
            = kgraph.control.addElement(ConditionalOp{isEven, outerMode, "Exec Conditional"});

        auto kernel = kgraph.control.addElement(Kernel());
        kgraph.control.addElement(Body(), {kernel}, {initOp});
        kgraph.control.addElement(Sequence(), {initOp}, {conditional});
        kgraph.control.addElement(Body(), {conditional}, {trueOp});
        // Else body: outerFalseOp then innerConditional.
        kgraph.control.addElement(Else(), {conditional}, {outerFalseOp});
        kgraph.control.addElement(Sequence(), {outerFalseOp}, {innerConditional});

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

    void runGPUExecutionTestNestedInElse(ConditionalMode outerMode,
                                         ConditionalMode innerMode,
                                         bool            withInnerElseBody)
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
        auto kgraph = buildNestedInElseConditionalGraphWithStore(
            outerMode, innerMode, withInnerElseBody, k->workitemIndex()[0], wfSize);
        ctx->schedule(rocRoller::KernelGraph::generate(kgraph, k));
        ctx->schedule(k->postamble());
        ctx->schedule(k->amdgpu_metadata());

        if(ctx->hipDeviceIndex() < 0)
            SKIP("No HIP device present.");

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

        // For outerMode==Exec the else body runs per-lane for odd lanes.
        // isBit1Clear is true when bit 1 of workitemId is 0, i.e. i%4==1 (lanes 1,5,9,...).
        // For outerMode==BranchAndExec the else body only runs when all active lanes
        // fail isEven, which never happens with a mixed warp, so odd lanes stay at 0u.
        // The inner else (assign 3u) runs per-lane only when innerMode==Exec && withInnerElseBody.
        bool innerElseRunsPerLane = withInnerElseBody && (innerMode == ConditionalMode::Exec);

        std::vector<uint32_t> expected(wfSize);
        for(uint32_t i = 0; i < wfSize; ++i)
        {
            if(i % 2 == 0)
                expected[i] = 1u; // outer true
            else if(outerMode == ConditionalMode::Exec)
                expected[i] = (i % 4 == 1) ? 5u : (innerElseRunsPerLane ? 3u : 2u);
            else
                expected[i] = 0u; // BranchAndExec outer else never runs per-lane
        }

        CHECK(hostOutput == expected);
    }

    TEST_CASE("ExecuteMaskGenerator - Exec outer / Exec inner, nested conditional in else (GPU "
              "execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNestedInElse(ConditionalMode::Exec, ConditionalMode::Exec, false);
    }

    TEST_CASE("ExecuteMaskGenerator - Exec outer / BranchAndExec inner, nested conditional in else "
              "(GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNestedInElse(
            ConditionalMode::Exec, ConditionalMode::BranchAndExec, false);
    }

    TEST_CASE("ExecuteMaskGenerator - BranchAndExec outer / Exec inner, nested conditional in else "
              "(GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNestedInElse(
            ConditionalMode::BranchAndExec, ConditionalMode::Exec, false);
    }

    TEST_CASE("ExecuteMaskGenerator - BranchAndExec outer / BranchAndExec inner, nested "
              "conditional in else (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNestedInElse(
            ConditionalMode::BranchAndExec, ConditionalMode::BranchAndExec, false);
    }

    TEST_CASE("ExecuteMaskGenerator - Exec outer / Exec inner, nested conditional with inner else "
              "in else (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNestedInElse(ConditionalMode::Exec, ConditionalMode::Exec, true);
    }

    TEST_CASE("ExecuteMaskGenerator - Exec outer / BranchAndExec inner, nested conditional with "
              "inner else in else (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNestedInElse(
            ConditionalMode::Exec, ConditionalMode::BranchAndExec, true);
    }

    TEST_CASE("ExecuteMaskGenerator - BranchAndExec outer / Exec inner, nested conditional with "
              "inner else in else (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNestedInElse(
            ConditionalMode::BranchAndExec, ConditionalMode::Exec, true);
    }

    TEST_CASE("ExecuteMaskGenerator - BranchAndExec outer / BranchAndExec inner, nested "
              "conditional with inner else in else (GPU execution)",
              "[exec-mask][gpu]")
    {
        runGPUExecutionTestNestedInElse(
            ConditionalMode::BranchAndExec, ConditionalMode::BranchAndExec, true);
    }

} // namespace ExecuteMaskGeneratorTest
