// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "LDSTestUtils.hpp"
#include "Agent.hpp"
#include "Utils.hpp"

#include <algorithm>
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/Scheduling.hpp>
#include <fmt/format.h>

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Scheduling/LDSModel.hpp>

namespace rocRoller
{
    LDSTestKernelBase::LDSTestKernelBase(ContextPtr                 context,
                                         uint32_t                   workgroupSize,
                                         size_t                     instrDwords,
                                         size_t                     strideMultiplier,
                                         const std::vector<size_t>& baseAddresses,
                                         bool                       write)
        : AssemblyTestKernel(context)
        , m_workgroupSize(workgroupSize)
        , m_instrDwords(instrDwords)
        , m_strideMultiplier(strideMultiplier)
        , m_baseAddresses(baseAddresses)
        , m_write(write)
    {
        auto k = m_context->kernel();
        k->setKernelDimensions(1);

        const auto one  = std::make_shared<Expression::Expression>(1u);
        const auto zero = std::make_shared<Expression::Expression>(0u);

        auto workitemCount = Expression::literal(m_workgroupSize * 256);
        k->setWorkgroupSize({m_workgroupSize, 1, 1});
        k->setWorkitemCount({workitemCount, one, one});
        k->setDynamicSharedMemBytes(zero);
    }

    void LDSTestKernelBase::operator()()
    {
        KernelInvocation    invocation{{m_workgroupSize * 256, 1, 1}, {m_workgroupSize, 1, 1}, 0};
        AssemblyTestKernel::operator()(invocation);
    }

    const std::vector<Instruction>& LDSTestKernelBase::getInstructions() const
    {
        return m_instructions;
    }

    std::string LDSTestKernelBase::getSectionName() const
    {
        return fmt::format("{} b{} s{} wgs{}.",
                           m_write ? "write" : "read",
                           m_instrDwords * 32,
                           m_strideMultiplier,
                           m_workgroupSize);
    }

    void LDSTestKernelBase::generate()
    {
        auto k = m_context->kernel();

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto ldsData = Register::Value::AllocateLDS(
            m_context,
            DataType::Raw32,
            m_context->targetArchitecture().GetCapability(GPUCapability::MaxLdsSize) / 4);

        m_ldsWithOffset
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::UInt32, 1);
        m_workitemIndex = m_context->kernel()->workitemIndex()[0];

        auto kb = [&]() -> Generator<Instruction> {
            co_yield Expression::generate(
                m_ldsWithOffset,
                Expression::literal(ldsData->getLDSAllocation()->offset())
                    + m_workitemIndex->expression()
                          * Expression::literal((4 * m_strideMultiplier * m_instrDwords)
                                                    % ldsData->getLDSAllocation()->size(),
                                                resultType(m_workitemIndex->expression()).varType),
                m_context);

            m_ldsDst = Register::Value::Placeholder(
                m_context,
                Register::Type::Vector,
                DataType::Raw32,
                248,
                Register::AllocationOptions{.contiguousChunkWidth = Register::FULLY_CONTIGUOUS,
                                            .alignment = static_cast<int>(m_instrDwords)});
            co_yield m_ldsDst->allocate();

            co_yield m_context->mem()->barrier({});

            co_yield generateKernelBody();
        };

        m_instructions.clear();
        for(auto inst : kb())
        {
            if(GPUInstructionInfo::isLDS(inst.getOpCode()))
                inst.setModelledAddresses(m_baseAddresses);
            m_context->schedule(inst);
            m_instructions.push_back(inst);
        }
        m_instructions.push_back(Instruction("s_endpgm", {}, {}, {}, ""));

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

    KernelLatencyResults runKernelAndCollectLatencies(TestContext&       context,
                                                      LDSTestKernelBase& kernel)
    {
        std::vector<std::vector<rocRoller::profiler::InstructionProfile>> allLatencies;

        for(int run = 0; run < NUM_RUNS; ++run)
        {
            const auto latencies = rocRoller::profiler::loopUntilDispatchData([&]() { kernel(); });
            allLatencies.push_back(latencies);
        }

        const auto& instructions = kernel.getInstructions();

        const auto filteredInstructions
            = filterAndVerifyInstructions(instructions, allLatencies[0]);

        size_t expectedSize = allLatencies[0].size();
        REQUIRE(std::all_of(
            allLatencies.begin(), allLatencies.end(), [expectedSize](const auto& latencies) {
                return latencies.size() == expectedSize;
            }));

        std::vector<std::tuple<std::string, size_t>> medianLatencies;
        for(size_t i = 0; i < filteredInstructions.size(); ++i)
        {
            std::vector<uint64_t> latenciesPerRun;
            for(const auto& runLatencies : allLatencies)
            {
                latenciesPerRun.push_back(runLatencies[i].meanLatency());
            }
            auto       medianLatency = MedianOfOddElements(latenciesPerRun);
            const auto instrString   = allLatencies[0][i].instruction;
            medianLatencies.push_back(std::make_tuple(instrString, medianLatency));
        }

        auto infoStr = formatLatencyComparison(filteredInstructions, medianLatencies);

        auto analysis = analyzeLatencyDeltas(filteredInstructions, medianLatencies);
        infoStr += fmt::format(
            "\nTotal delta: {}, Total absolute delta: {}, Incorrect predictions: {}/{}",
            analysis.totalDelta,
            analysis.totalAbsoluteDelta,
            analysis.incorrectPredictionCount,
            filteredInstructions.size() - 1);

        return KernelLatencyResults{.filteredInstructions = std::move(filteredInstructions),
                                    .medianLatencies      = std::move(medianLatencies),
                                    .infoStr              = infoStr};
    }

    LatencyAnalysisResult
        analyzeLatencyDeltas(const std::vector<Instruction>& filteredInstructions,
                             const std::vector<std::tuple<std::string, size_t>>& medianLatencies)
    {
        LatencyAnalysisResult result = {0, 0, 0};

        // Skip the last instruction (s_endpgm)
        for(size_t i = 0; i < filteredInstructions.size() - 1; ++i)
        {
            const auto& inst = filteredInstructions[i];

            int modelLatency  = inst.totalCycles() * 4;
            int actualLatency = std::get<1>(medianLatencies[i]);
            int delta         = actualLatency - modelLatency;

            result.totalDelta += delta;
            result.totalAbsoluteDelta += std::abs(delta);

            if(delta != 0)
            {
                result.incorrectPredictionCount++;
            }
        }

        return result;
    }

    ParameterizedLDSKernel::ParameterizedLDSKernel(ContextPtr                 context,
                                                   uint32_t                   workgroupSize,
                                                   size_t                     instrDwords,
                                                   size_t                     strideMultiplier,
                                                   const std::vector<size_t>& baseAddresses,
                                                   bool                       write,
                                                   BodyGenerator              bodyGen)
        : LDSTestKernelBase(
            context, workgroupSize, instrDwords, strideMultiplier, baseAddresses, write)
        , m_bodyGenerator(bodyGen)
    {
    }

    Generator<Instruction> ParameterizedLDSKernel::scheduleLdsInstruction(int& counter)
    {
        const auto [start, end]
            = getAlignedSubset(m_ldsDst->registerCount(), m_instrDwords, counter++);
        auto dstRegs = m_ldsDst->subset(Generated(iota(start, end)));
        if(m_write)
            co_yield m_context->mem()->storeLocal(m_ldsWithOffset, dstRegs, 0, 4 * m_instrDwords);
        else
            co_yield m_context->mem()->loadLocal(dstRegs, m_ldsWithOffset, 0, 4 * m_instrDwords);
    }

    ContextPtr ParameterizedLDSKernel::getContext() const
    {
        return m_context;
    }

    Generator<Instruction> ParameterizedLDSKernel::generateKernelBody()
    {
        return m_bodyGenerator(this);
    }

    void runLDSTest(const LDSTestConfig& config)
    {
        using namespace Scheduling::LDSModel;

        const auto workgroupSize
            = config.useMultipleWorkgroupSizes ? GENERATE(64u, 128u, 256u) : 64u;

        const int  instrDwords      = GENERATE(1, 2, 4);
        const int  strideMultiplier = GENERATE(1, 2, 4, 8, 16);
        const bool write            = GENERATE(true, false);

        const auto baseAddresses
            = generateLDSAddresses(workgroupSize, strideMultiplier, instrDwords);

        profiler::reset();

        KernelOptions kernelOpts;
        kernelOpts->dsObserver = DSObserverType::WeightlessDSMemObserver;
        auto context           = TestContext::ForTestDevice(kernelOpts, "");

        if(not context->targetArchitecture().target().isCDNA4GPU())
        {
            SKIP("Currently only testing on gfx950");
        }

        ParameterizedLDSKernel kernel(context.get(),
                                      workgroupSize,
                                      instrDwords,
                                      strideMultiplier,
                                      baseAddresses,
                                      write,
                                      config.kernelBodyGen);

        SECTION(kernel.getSectionName())
        {
            auto result = runKernelAndCollectLatencies(context, kernel);
            auto analysis
                = analyzeLatencyDeltas(result.filteredInstructions, result.medianLatencies);

            config.validationFunc(
                result, analysis, instrDwords, strideMultiplier, write, workgroupSize);
        }
    }

} // namespace rocRoller
