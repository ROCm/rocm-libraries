// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "../catch/TestContext.hpp"
#include "../catch/TestKernels.hpp"
#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/KernelOptions.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Scheduling/LDSModel.hpp>
#include <rocRoller/Scheduling/Observers/FunctionalUnit/MEMObserver.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include <functional>
#include <string>
#include <tuple>
#include <vector>

namespace rocRoller
{
    /**
     * @brief Results from kernel profiling and latency collection
     */
    struct KernelLatencyResults
    {
        std::vector<Instruction>                     filteredInstructions;
        std::vector<std::tuple<std::string, size_t>> medianLatencies;
        std::string                                  infoStr;
    };

    /**
     * @brief Results from latency delta analysis
     */
    struct LatencyAnalysisResult
    {
        int totalDelta;
        int totalAbsoluteDelta;
        int incorrectPredictionCount;
    };

    /**
     * @brief Base class for LDS test kernels with shared functionality
     */
    class LDSTestKernelBase : public AssemblyTestKernel
    {
    public:
        LDSTestKernelBase(ContextPtr                 context,
                          uint32_t                   workgroupSize,
                          size_t                     instrDwords,
                          size_t                     strideMultiplier,
                          const std::vector<size_t>& baseAddresses,
                          bool                       write);

        void operator()();

        const std::vector<Instruction>& getInstructions() const;

        std::string getSectionName() const;

    protected:
        virtual void generate() override;

        virtual Generator<Instruction> generateKernelBody() = 0;

    protected:
        uint32_t                 m_workgroupSize;
        size_t                   m_instrDwords;
        size_t                   m_strideMultiplier;
        std::vector<size_t>      m_baseAddresses;
        bool                     m_write;
        std::vector<Instruction> m_instructions;

        std::shared_ptr<Register::Value> m_ldsDst;
        std::shared_ptr<Register::Value> m_ldsWithOffset;
        std::shared_ptr<Register::Value> m_workitemIndex;
    };

    /**
     * @brief Helper function for profiling and collecting median latencies
     */
    KernelLatencyResults runKernelAndCollectLatencies(TestContext&       context,
                                                      LDSTestKernelBase& kernel);

    /**
     * @brief Analyzes latency deltas between model predictions and profiler measurements
     *
     * @param filteredInstructions The list of instructions to analyze
     * @param medianLatencies The measured latencies for each instruction
     * @return Structure containing total delta, absolute delta, and incorrect prediction count
     */
    LatencyAnalysisResult
        analyzeLatencyDeltas(const std::vector<Instruction>& filteredInstructions,
                             const std::vector<std::tuple<std::string, size_t>>& medianLatencies);

    /**
     * @brief Generic parameterized kernel class for LDS tests
     */
    class ParameterizedLDSKernel : public LDSTestKernelBase
    {
    public:
        using BodyGenerator = std::function<Generator<Instruction>(ParameterizedLDSKernel*)>;

        ParameterizedLDSKernel(ContextPtr                 context,
                               uint32_t                   workgroupSize,
                               size_t                     instrDwords,
                               size_t                     strideMultiplier,
                               const std::vector<size_t>& baseAddresses,
                               bool                       write,
                               BodyGenerator              bodyGen);

        Generator<Instruction> scheduleLdsInstruction(int& counter);

        ContextPtr getContext() const;

    protected:
        Generator<Instruction> generateKernelBody() override;

    private:
        BodyGenerator m_bodyGenerator;
    };

    /**
     * @brief Configuration for LDS test cases
     */
    struct LDSTestConfig
    {
        bool                                                           useMultipleWorkgroupSizes;
        std::function<Generator<Instruction>(ParameterizedLDSKernel*)> kernelBodyGen;
        std::function<void(
            const KernelLatencyResults&, const LatencyAnalysisResult&, int, int, bool, uint32_t)>
            validationFunc;
    };

    /**
     * @brief Common test harness function for LDS tests
     * Handles workgroup/item set up, kernel generation, profiling, latency collection, and validation based on the provided configuration
     * Test need only implement the kernel body generation and validation logic
     */
    void runLDSTest(const LDSTestConfig& config);

} // namespace rocRoller
