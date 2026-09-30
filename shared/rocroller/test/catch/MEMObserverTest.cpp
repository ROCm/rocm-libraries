// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <memory>

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

#include <common/Scheduling.hpp>
#include <common/SourceMatcher.hpp>
#include <common/TestValues.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/GPUArchitecture/GPUCapability.hpp>
#include <rocRoller/Scheduling/Observers/FunctionalUnit/MEMObserver.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_contains.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using namespace rocRoller;

namespace MEMObserverTest
{
    using Catch::Matchers::ContainsSubstring;

    void peekAndSchedule(TestContext& context, Instruction& inst, uint expectedStalls = 0)
    {
        auto peeked = context->observer()->peek(inst);
        CHECK(peeked.stallCycles == expectedStalls);
        context->schedule(inst);
    }

    TEST_CASE("MEMObservers predicts Stall Cycles", "[observer]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            if(!TestContext::ForTarget(arch)->targetArchitecture().HasCapability(
                   GPUCapability::HasAccCD))
            {
                SKIP("Architecture " + arch.toString() + " does not use Accumulator registers.");
            }

            SECTION("VMEM Instructions stall")
            {
                auto context = TestContext::ForTarget(arch);
                auto weights = Scheduling::VMEMObserver::getWeights(context.get());

                if(weights.vmemQueueSize != 3)
                {
                    SKIP("Test tailored to vmemQueueSize == 3");
                }

                auto s    = context.createRegisters(Register::Type::Scalar, DataType::UInt32, 8);
                auto zero = Register::Value::Literal(0);

                std::vector<Instruction> insts = {
                    Instruction("buffer_load_dwordx2", {s[1]}, {s[0], zero}, {}, ""),
                    Instruction("buffer_load_dwordx2", {s[3]}, {s[2], zero}, {}, ""),
                    Instruction("buffer_load_dwordx2", {s[5]}, {s[4], zero}, {}, ""),
                    Instruction("buffer_load_dwordx2", {s[7]}, {s[6], zero}, {}, ""),
                };

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);
                peekAndSchedule(context, insts[2]);
                peekAndSchedule(context, insts[3], weights.vmemCycles - 2); // e.g. 384 - 3 + 1

                CHECK_THAT(context.output(), ContainsSubstring("CBNW: 0"));
                CHECK_THAT(context.output(), ContainsSubstring("Inc: 3"));
            }

            SECTION("DS Instructions stall")
            {
                auto context = TestContext::ForTarget(arch);
                auto weights = Scheduling::DSMEMObserver::getWeights(context.get());

                if(weights.vmemQueueSize != 3)
                {
                    SKIP("Test tailored to vmemQueueSize == 3");
                }

                auto s    = context.createRegisters(Register::Type::Scalar, DataType::UInt32, 8);
                auto zero = Register::Value::Literal(0);

                std::vector<Instruction> insts = {
                    Instruction("ds_read_b32", {s[1]}, {s[0], zero}, {}, ""),
                    Instruction("ds_read_b32", {s[3]}, {s[2], zero}, {}, ""),
                    Instruction("ds_read_b32", {s[5]}, {s[4], zero}, {}, ""),
                    Instruction("ds_read_b32", {s[7]}, {s[6], zero}, {}, ""),
                };

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);
                peekAndSchedule(context, insts[2]);
                peekAndSchedule(context, insts[3], weights.dsmemCycles - 2); // e.g. 384 - 3 + 1

                CHECK_THAT(context.output(), ContainsSubstring("CBNW: 0"));
                CHECK_THAT(context.output(), ContainsSubstring("Inc: 3"));
            }

            SECTION("No stalls if waitcounts used for loads")
            {
                auto context = TestContext::ForTarget(arch);
                auto s       = context.createRegisters(Register::Type::Scalar, DataType::UInt32, 8);
                auto zero    = Register::Value::Literal(0);

                std::vector<Instruction> insts = {
                    Instruction("buffer_load_dwordx2", {s[1]}, {s[0], zero}, {}, ""),
                    Instruction::Wait(WaitCount::LoadCnt(context->targetArchitecture(), 0)),
                    Instruction("buffer_load_dwordx2", {s[3]}, {s[2], zero}, {}, ""),
                    Instruction("buffer_load_dwordx2", {s[5]}, {s[4], zero}, {}, ""),
                    Instruction("buffer_load_dwordx2", {s[7]}, {s[6], zero}, {}, ""),
                };

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);
                peekAndSchedule(context, insts[2]);
                peekAndSchedule(context, insts[3]);
                peekAndSchedule(context, insts[4]);
            }

            SECTION("No stalls if waitcounts used for stores")
            {
                auto context = TestContext::ForTarget(arch);
                auto s       = context.createRegisters(Register::Type::Scalar, DataType::UInt32, 8);
                auto zero    = Register::Value::Literal(0);

                std::vector<Instruction> insts = {
                    Instruction("buffer_store_dwordx2", {s[1]}, {s[0], zero}, {}, ""),
                    Instruction::Wait(WaitCount::StoreCnt(context->targetArchitecture(), 0)),
                    Instruction("buffer_store_dwordx2", {s[3]}, {s[2], zero}, {}, ""),
                    Instruction("buffer_store_dwordx2", {s[5]}, {s[4], zero}, {}, ""),
                    Instruction("buffer_store_dwordx2", {s[7]}, {s[6], zero}, {}, ""),
                };

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);
                peekAndSchedule(context, insts[2]);
                peekAndSchedule(context, insts[3]);
                peekAndSchedule(context, insts[4]);
            }

            SECTION("Instructions with latency affect tracked cycles")
            {
                auto context = TestContext::ForTarget(arch);
                auto s       = context.createRegisters(Register::Type::Scalar, DataType::UInt32, 4);
                auto v_f16   = context.createRegisters(Register::Type::Vector, DataType::Half, 3);
                auto zero    = Register::Value::Literal(0);
                std::vector<Register::ValuePtr> a;
                std::string                     mi = "";

                auto const& architecture = context.get()->targetArchitecture();

                if(architecture.HasCapability(GPUCapability::HasMFMA))
                {
                    a  = context.createRegisters(Register::Type::Accumulator, DataType::Float, 2);
                    mi = "v_mfma_f32_32x32x8f16";
                }
                else if(architecture.HasCapability(GPUCapability::HasWMMA_f32_16x16x32_f16))
                {
                    a  = context.createRegisters(Register::Type::Vector, DataType::Float, 2);
                    mi = "v_wmma_f32_16x16x32_f16";
                }
                else
                {
                    SKIP("Unsupported Matrix Instruction.");
                }

                std::vector<Instruction> insts{
                    Instruction("buffer_store_dwordx2", {s[1]}, {s[0], zero}, {}, ""),
                    Instruction(mi, {a[0]}, {v_f16[0], v_f16[1], a[0]}, {}, ""),
                    Instruction(mi, {a[1]}, {v_f16[1], v_f16[2], a[1]}, {}, ""),
                    Instruction("buffer_store_dwordx2", {s[3]}, {s[2], zero}, {}, ""),
                };

                auto cycles = architecture.GetInstructionInfo(mi).getLatency();

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);
                peekAndSchedule(context, insts[2], cycles); // latency caused by insts[1]
                peekAndSchedule(context, insts[3]);

                CHECK_THAT(context.output(),
                           ContainsSubstring("current "
                                             + std::to_string(4 + cycles))); // 4 insts + latency
            }

            SECTION("VMEM Instructions with dependency")
            {
                auto context = TestContext::ForTarget(arch);
                auto weights = Scheduling::VMEMObserver::getWeights(context.get());

                if(weights.vmemQueueSize != 3)
                {
                    SKIP("Test tailored to vmemQueueSize == 3");
                }

                auto s    = context.createRegisters(Register::Type::Scalar, DataType::UInt32, 13);
                auto zero = Register::Value::Literal(0);

                std::vector<Instruction> insts = {
                    Instruction("buffer_load_dwordx2", {s[1]}, {s[0], zero}, {}, ""), // pc+=1
                    Instruction("buffer_load_dwordx2", {s[3]}, {s[2], zero}, {}, ""), // pc+=1
                    Instruction("buffer_load_dwordx2", {s[5]}, {s[4], zero}, {}, ""), // pc+=1
                    Instruction("buffer_load_dwordx2",
                                {s[7]},
                                {s[6], zero},
                                {},
                                ""), // pc=inst[0].expected+1, expected=pc+vmemCycles

                    Instruction("buffer_load_dwordx2",
                                {s[8]},
                                {s[3], zero},
                                {},
                                ""), // pc+=2 (including s_waitcnt)

                    Instruction("buffer_load_dwordx2", {s[10]}, {s[9], zero}, {}, ""), // pc+=1
                    Instruction("buffer_load_dwordx2",
                                {s[12]},
                                {s[11], zero},
                                {},
                                ""), // pc=inst[3].expected+1
                };

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);
                peekAndSchedule(context, insts[2]);
                peekAndSchedule(context, insts[3], weights.vmemCycles - 2); // inst[0].expected - pc
                peekAndSchedule(context, insts[4]); // dependency. generate waitcount.

                peekAndSchedule(context, insts[5]);
                peekAndSchedule(context, insts[6], weights.vmemCycles - 3); // inst[3].expected - pc

                CHECK_THAT(context.output(), ContainsSubstring("CBNW: 0, Inc: 3"));
                CHECK_THAT(context.output(), ContainsSubstring("CBNW: 1, Inc: 3"));

                CHECK_THAT(context.output(), ContainsSubstring("s_waitcnt vmcnt(2)"));
            }
        }
    }

    struct WeaveLDSAndSAddKernel : public AssemblyTestKernel
    {
        WeaveLDSAndSAddKernel(ContextPtr context, size_t strideMultiplier)
            : AssemblyTestKernel(context)
            , m_strideMultiplier(strideMultiplier)
        {
        }

        size_t m_strideMultiplier;

        void generate() override
        {
            auto k = m_context->kernel();
            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            const auto workgroupSize = 64;
            const auto instrDwords   = 4;

            auto ldsWithOffset = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::UInt32, 1);
            auto workitemIndex = m_context->kernel()->workitemIndex()[0];

            const auto baseAddresses
                = generateLDSAddresses(workgroupSize, m_strideMultiplier, instrDwords);

            auto agpr
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Accumulator,
                                               DataType::Float,
                                               16,
                                               Register::AllocationOptions::FullyContiguous());
            agpr->allocateNow();

            auto v0 = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   DataType::Half,
                                                   1,
                                                   Register::AllocationOptions::FullyContiguous());
            v0->allocateNow();

            auto v1 = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   DataType::Half,
                                                   1,
                                                   Register::AllocationOptions::FullyContiguous());
            v1->allocateNow();

            int counter = 0;

            auto ldsData = Register::Value::AllocateLDS(m_context, DataType::Raw32, 64);

            auto m_ldsDst = Register::Value::Placeholder(
                m_context,
                Register::Type::Vector,
                DataType::Raw32,
                64,
                Register::AllocationOptions{.contiguousChunkWidth = Register::FULLY_CONTIGUOUS,
                                            .alignment            = static_cast<int>(instrDwords)});

            auto prereq = [&]() -> Generator<Instruction> {
                co_yield Expression::generate(
                    ldsWithOffset,
                    Expression::literal(ldsData->getLDSAllocation()->offset())
                        + workitemIndex->expression()
                              * Expression::literal(
                                  (4 * m_strideMultiplier * instrDwords)
                                      % ldsData->getLDSAllocation()->size(),
                                  resultType(workitemIndex->expression()).varType),
                    m_context);

                co_yield m_ldsDst->allocate();
            };

            auto ldsInstructions = [&]() -> Generator<Instruction> {
                for(int i = 0; i < 12; ++i)
                {
                    const auto [start, end]
                        = getAlignedSubset(m_ldsDst->registerCount(), instrDwords, counter++);
                    auto dstRegs = m_ldsDst->subset(Generated(iota(start, end)));
                    for(auto inst :
                        m_context->mem()->loadLocal(dstRegs, ldsWithOffset, 0, 4 * instrDwords))
                    {
                        inst.setModelledAddresses(baseAddresses);
                        co_yield inst;
                    }
                }
                co_yield Instruction::Wait(WaitCount::Zero(m_context->targetArchitecture()));
            };

            auto addInstructions = [&]() -> Generator<Instruction> {
                for(int i = 1; i < 12; ++i)
                {
                    co_yield Instruction("v_mfma_f32_32x32x2f32", {agpr}, {v0, v1, agpr}, {}, "");
                }
            };

            std::vector<Generator<Instruction>> sequences;
            sequences.push_back(addInstructions());
            sequences.push_back(ldsInstructions());

            std::shared_ptr<Scheduling::Scheduler> scheduler
                = Component::GetNew<Scheduling::Scheduler>(Scheduling::SchedulerProcedure::Priority,
                                                           Scheduling::CostFunction::LinearWeighted,
                                                           m_context);

            m_context->schedule(prereq());
            m_context->schedule((*scheduler)(sequences));
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }
    };

    TEST_CASE("DSObserver scheduling", "[rocprofiler][lds-model][gpu]")
    {
        /*
        Schedules a stream of ds_read_b128s and a stream of mfmas (does no meaningful work)
        Checks scheduler interweaves based on information from observers
        */
        const auto dsObserver
            = GENERATE(DSObserverType::WeightlessDSMemObserver, DSObserverType::DSMEMObserver);
        const auto strideMultiplier = GENERATE(1, 16);

        KernelOptions kernelOps;
        kernelOps->dsObserver = dsObserver;

        auto context = TestContext::ForTestDevice(kernelOps);

        if(not context->targetArchitecture().target().isCDNA4GPU())
        {
            // LDS bandwidth and observer weights being different in different architectures
            // results in different scheduling decisions.
            // Purpose of this test is to verify scheduling differences between the two observers.
            SKIP("Test tailored for gfx950 behavior");
        }

        WeaveLDSAndSAddKernel testKernel(context.get(), strideMultiplier);
        testKernel({});

        SECTION(fmt::format("{}, stride {}", toString(dsObserver), strideMultiplier))
        {
            std::string expected;
            if(strideMultiplier == 1 && dsObserver == DSObserverType::WeightlessDSMemObserver)
            {
                expected = R"(
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    ds_read_b128 v[8:11], v5
                    ds_read_b128 v[12:15], v5
                    ds_read_b128 v[16:19], v5
                    ds_read_b128 v[20:23], v5
                    ds_read_b128 v[24:27], v5
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    ds_read_b128 v[28:31], v5
                    ds_read_b128 v[32:35], v5
                    ds_read_b128 v[36:39], v5
                    ds_read_b128 v[40:43], v5
                    ds_read_b128 v[44:47], v5
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    ds_read_b128 v[48:51], v5
                    ds_read_b128 v[52:55], v5
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                )";
            }
            else if(strideMultiplier == 16 && dsObserver == DSObserverType::WeightlessDSMemObserver)
            {
                expected = R"(
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    ds_read_b128 v[8:11], v5
                    ds_read_b128 v[12:15], v5
                    ds_read_b128 v[16:19], v5
                    ds_read_b128 v[20:23], v5
                    ds_read_b128 v[24:27], v5
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    ds_read_b128 v[28:31], v5
                    ds_read_b128 v[32:35], v5
                    ds_read_b128 v[36:39], v5
                    ds_read_b128 v[40:43], v5
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    ds_read_b128 v[44:47], v5
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    ds_read_b128 v[48:51], v5
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    ds_read_b128 v[52:55], v5
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)
                )";
            }
            else if(dsObserver == DSObserverType::DSMEMObserver)
            {
                expected = R"(
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    ds_read_b128 v[8:11], v5
                    ds_read_b128 v[12:15], v5
                    ds_read_b128 v[16:19], v5
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    ds_read_b128 v[20:23], v5
                    ds_read_b128 v[24:27], v5
                    ds_read_b128 v[28:31], v5
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    v_mfma_f32_32x32x2f32 a[0:15], v0, v4, a[0:15]
                    ds_read_b128 v[32:35], v5
                    ds_read_b128 v[36:39], v5
                    ds_read_b128 v[40:43], v5
                    ds_read_b128 v[44:47], v5
                    ds_read_b128 v[48:51], v5
                    ds_read_b128 v[52:55], v5
                    s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)
                )";
            }
            else
            {
                FAIL("Unknown DSObserverType");
            }
            CHECK_THAT(NormalizedSource(context.output()),
                       ContainsSubstring(NormalizedSource(expected)));
        }
    }
}
