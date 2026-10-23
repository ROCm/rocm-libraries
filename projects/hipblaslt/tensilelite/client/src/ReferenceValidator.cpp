/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "ReferenceValidator.hpp"
#include "ResultReporter.hpp"
#include "TimingInstrumentation.hpp"
#include <TensileLite/Client/hostnumerics/GemmInvocationAdapter.hpp>
#include <TensileLite/Client/hostnumerics/HostNumericsBridge.hpp>
#include <hipblaslt/client/FreivaldsCheck.hpp>
#include <hipblaslt/client/FreivaldsDevice.hpp>

#include <TensileLite/Client/hostnumerics/Reference.hpp>

#include <Tensile/DataTypes.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <cstddef>
#include <limits>
#include <sstream>

namespace TensileLite
{
    namespace Client
    {
        ReferenceValidator::ReferenceValidator(po::variables_map const&            args,
                                               std::shared_ptr<DataInitialization> dataInit,
                                               hipStream_t                         stream)
            : m_stream(stream)
            , m_dataInit(dataInit)
        {
            m_elementsToValidate = args["num-elements-to-validate"].as<int>();
            m_printValids        = args["print-valids"].as<bool>();
            m_printMax           = args["print-max"].as<int>();

            m_printTensorA             = args["print-tensor-a"].as<bool>();
            m_printTensorB             = args["print-tensor-b"].as<bool>();
            m_printTensorC             = args["print-tensor-c"].as<bool>();
            m_printTensorD             = args["print-tensor-d"].as<bool>();
            m_printTensorRef           = args["print-tensor-ref"].as<bool>();
            m_printTensorBias          = args["print-tensor-bias"].as<bool>();
            m_printTensorGate          = args["print-tensor-gate"].as<bool>();
            m_printTensorScaleAlphaVec = args["print-tensor-scale-alpha-vec"].as<bool>();
            m_printTensorAmaxD         = args["print-tensor-amaxd"].as<bool>();

            m_printAny = m_printTensorA || m_printTensorB || m_printTensorC || m_printTensorD
                         || m_printTensorRef || m_printTensorBias || m_printTensorGate
                         || m_printTensorAmaxD || m_printTensorScaleAlphaVec;

            m_freivalds = args.count("freivalds") && args["freivalds"].as<bool>();
            if(m_freivalds && m_printTensorRef)
                throw std::invalid_argument(
                    "Freivalds does not compute a reference tensor to print.");
            m_enabled = m_freivalds || m_elementsToValidate != 0 || m_printAny;
        }

        bool ReferenceValidator::needMoreBenchmarkRuns() const
        {
            if(m_enabled && m_numBenchmarkRuns == 0)
                return true;

            return false;
        }

        void ReferenceValidator::preBenchmarkRun() {}

        void ReferenceValidator::postBenchmarkRun()
        {
            m_numBenchmarkRuns++;
        }

        void ReferenceValidator::preProblem(ContractionProblem* const problem)
        {
            m_outputSelections.clear();
            if(m_enabled)
            {
                m_problem = problem;

                // Report problem context for timing correlation
                if(auto gemm = dynamic_cast<ContractionProblemGemm*>(problem))
                {
                    if(!m_freivalds)
                        m_outputSelections.push_back(referenceOutputSelection(
                            gemm->d(), static_cast<size_t>(m_elementsToValidate)));
                    size_t M          = gemm->freeSizeA(0);
                    size_t N          = gemm->freeSizeB(0);
                    size_t K          = gemm->boundSize(0);
                    size_t batchCount = gemm->batchSize(0);
                    reportProblemContext(M,
                                         N,
                                         K,
                                         batchCount,
                                         TensileLite::ToString(gemm->a().dataType()),
                                         TensileLite::ToString(gemm->d().dataType()));
                }
                else if(auto grouped = dynamic_cast<ContractionProblemGroupedGemm*>(problem))
                {
                    size_t totalGemms = grouped->gemms.size();
                    m_outputSelections.reserve(totalGemms);
                    for(size_t i = 0; i < totalGemms; i++)
                    {
                        auto& g = grouped->gemms[i];
                        if(!m_freivalds)
                            m_outputSelections.push_back(referenceOutputSelection(
                                g.d(), static_cast<size_t>(m_elementsToValidate)));
                        size_t M          = g.freeSizeA(0);
                        size_t N          = g.freeSizeB(0);
                        size_t K          = g.boundSize(0);
                        size_t batchCount = g.batchSize(0);
                        reportGroupedProblemContext(i,
                                                    totalGemms,
                                                    M,
                                                    N,
                                                    K,
                                                    batchCount,
                                                    TensileLite::ToString(g.a().dataType()),
                                                    TensileLite::ToString(g.d().dataType()));
                    }
                }

                {
                    ScopedTimer timer("cpu_data_init");
                    m_referenceInputs = m_dataInit->prepareCPUInputs(problem);
                }

                {
                    ScopedTimer timer(m_freivalds ? "freivalds_prepare" : "cpu_reference_gemm");
                    if(m_freivalds)
                        prepareFreivalds();
                    else
                        SolveCPU(problem, m_referenceInputs.get(), m_outputSelections);
                }
            }
        }

        void ReferenceValidator::preSolution(ContractionSolution* const solution)
        {
            m_validatedSolution = false;
            m_errorInSolution   = false;
            m_executedSolution  = false;

            // Re-run CPU reference after DataInitialization refreshes MX inputs.
            if(!m_enabled || m_problem == nullptr || m_referenceInputs == nullptr
               || solution == nullptr)
                return;

            if(auto* gemm = dynamic_cast<ContractionProblemGemm*>(m_problem))
            {
                // Match DataInitialization MX gate.
                if(!isMXProblem(*gemm))
                    return;
                // Only recompute when DataInitialization actually refreshes MX
                // inputs for this solution (solution-dependent HostPreSwizzle).
                // Otherwise the preProblem reference is unchanged, so reuse it
                // instead of paying a full dense reference GEMM per solution.
                if(!m_dataInit->referenceNeedsPerSolutionRecompute(*gemm, solution))
                    return;
                ScopedTimer timer(m_freivalds ? "freivalds_prepare_per_solution"
                                              : "cpu_reference_gemm_per_solution");
                if(m_freivalds)
                    prepareFreivalds();
                else
                    SolveCPU(m_problem, m_referenceInputs.get(), m_outputSelections);
            }
        }

        bool ReferenceValidator::needMoreRunsInSolution() const
        {
            if(m_enabled && !m_validatedSolution)
                return true;

            return false;
        }

        size_t ReferenceValidator::numWarmupRuns()
        {
            if(m_enabled && !m_validatedSolution)
                return 1;

            return 0;
        }

        void ReferenceValidator::setNumWarmupRuns(size_t count) {}

        void ReferenceValidator::preWarmup() {}

        void ReferenceValidator::postWarmup(TimingEvents const& startEvents,
                                            TimingEvents const& stopEvents,
                                            hipStream_t const&  stream)
        {
            m_executedSolution = true;
        }

        void ReferenceValidator::prepareFreivalds()
        {
            using namespace roc::hostnumerics;
            m_freivaldsPlans.clear();
            auto prepare = [&](const ContractionProblemGemm& problem,
                               const ContractionInputs&      inputs) {
                if(!problem.stridedBatched())
                    throw std::invalid_argument("TensileLite Freivalds requires strided batches.");
                auto translated = hostnumerics::translateGemmInvocation(
                    problem, inputs, OutputSelection::all({}));
                if(const auto* failure = std::get_if<hostnumerics::TranslationFailure>(&translated))
                    throw std::invalid_argument("Freivalds: " + failure->reason);
                const auto& adapter = std::get<hostnumerics::GemmInvocationAdapter>(translated);
                std::vector<FreivaldsBatch> batches;
                for(size_t batch = 0; batch < adapter.batchCount(); ++batch)
                {
                    auto prepared = adapter.prepareFreivalds(batch, {}, {});
                    if(const auto* failure = std::get_if<FreivaldsPreparationFailure>(&prepared))
                        throw std::invalid_argument("Freivalds: " + failure->reason);
                    batches.emplace_back(std::get<FreivaldsPlan>(std::move(prepared)),
                                         adapter.outputLayout(batch));
                }
                m_freivaldsPlans.push_back(std::move(batches));
            };
            if(const auto* grouped = dynamic_cast<const ContractionProblemGroupedGemm*>(m_problem))
            {
                const auto& inputs
                    = dynamic_cast<const ContractionGroupedInputs&>(*m_referenceInputs);
                for(size_t i = 0; i < grouped->gemms.size(); ++i)
                    prepare(grouped->gemms[i], inputs.grouped.at(i));
            }
            else
                prepare(dynamic_cast<const ContractionProblemGemm&>(*m_problem),
                        dynamic_cast<const ContractionInputs&>(*m_referenceInputs));
        }

        bool ReferenceValidator::validateFreivalds(const ProblemInputs& inputs)
        {
            using namespace roc::hostnumerics;
            int             device = 0;
            hipDeviceProp_t properties{};
            HIP_CHECK_EXC(hipGetDevice(&device));
            HIP_CHECK_EXC(hipGetDeviceProperties(&properties, device));
            bool   passed          = true;
            size_t checkedElements = 0, readbackBytes = 0;
            bool   fallback = false;
            auto   check    = [&](size_t                        gemm,
                             const ContractionProblemGemm& problem,
                             const ContractionInputs&      reference,
                             const ContractionInputs&      output) {
                const auto* data = static_cast<const std::byte*>(output.d) + output.batchOffsetD;
                for(size_t batch = 0; batch < m_freivaldsPlans.at(gemm).size(); ++batch)
                {
                    const auto& [plan, layout] = m_freivaldsPlans[gemm][batch];
                    auto observation           = hipblaslt::client::observeFreivalds(
                        plan.copyRequest(), data, layout, m_stream);
                    const auto options = hipblaslt::client::freivaldsCheckOptions(
                        properties.major,
                        plan,
                        problem.boundSize(0),
                        toHostNumericsScalarType(problem.a().dataType()),
                        toHostNumericsScalarType(problem.b().dataType()),
                        toHostNumericsScalarType(problem.computeType()));
                    const auto result = std::visit(
                        [&](const auto& value) {
                            return plan.check(value, std::numeric_limits<size_t>::max(), options);
                        },
                        observation.output);
                    checkedElements += result.checkedElements;
                    readbackBytes += observation.readbackBytes;
                    fallback |= std::holds_alternative<Tensor>(observation.output);
                    if(result.status != VerificationStatus::Passed)
                    {
                        passed = false;
                        std::cerr << "Freivalds failed for GEMM " << gemm << ", batch " << batch
                                  << ": " << result.reason << " (" << result.badRowCount
                                  << " rows, " << result.badColumnCount << " columns, "
                                  << result.badElementCount << " exceptional elements)\n";
                    }
                }
                if(m_printAny)
                    printTensors(problem, reference, output);
                if(m_dataInit->getCurBoundsCheck() == BoundsCheckMode::NaN
                   && problem.d().totalLogicalElements() != 0)
                {
                    // Preserve the requested guard check without a numerical reference.
                    // Its existing readback includes D; account for this explicit cost.
                    const size_t index = static_cast<size_t>(ContractionProblemGemm::TENSOR::D);
                    passed &= !checkResults(problem.d(),
                                            reference.d,
                                            output.d,
                                            output.maxElements.at(index),
                                            output.gpu,
                                            OutputSelection::explicitIndices({}, {}),
                                            -1);
                    readbackBytes
                        += output.maxElements.at(index) * size_t(problem.d().elementBytes());
                    fallback = true;
                }
            };
            if(const auto* grouped = dynamic_cast<const ContractionGroupedInputs*>(&inputs))
            {
                const auto& problems
                    = dynamic_cast<const ContractionProblemGroupedGemm&>(*m_problem);
                const auto& references
                    = dynamic_cast<const ContractionGroupedInputs&>(*m_referenceInputs);
                for(size_t i = 0; i < grouped->grouped.size(); ++i)
                    check(i, problems.gemms.at(i), references.grouped.at(i), grouped->grouped[i]);
            }
            else
                check(0,
                      dynamic_cast<const ContractionProblemGemm&>(*m_problem),
                      dynamic_cast<const ContractionInputs&>(*m_referenceInputs),
                      dynamic_cast<const ContractionInputs&>(inputs));
            std::cout << "Freivalds: checked_elements=" << checkedElements
                      << ", readback_bytes=" << readbackBytes
                      << ", full_output_fallback=" << fallback << std::endl;
            m_errorInSolution |= !passed;
            return passed;
        }

        bool ReferenceValidator::validateSolution(std::shared_ptr<ProblemInputs> inputs)
        {
            if(!m_enabled)
                return false;

            if(m_freivalds)
                return validateFreivalds(*inputs);
            bool rv = false;

            if(m_elementsToValidate != 0)
            {
                if(auto problems = dynamic_cast<ContractionProblemGroupedGemm*>(m_problem))
                {
                    auto reference
                        = dynamic_cast<ContractionGroupedInputs const&>(*m_referenceInputs);
                    auto result = dynamic_cast<ContractionGroupedInputs const&>(*inputs);
                    rv          = true;
                    for(size_t j = 0; j < problems->gemms.size(); j++)
                    {
                        rv &= validate(problems->gemms[j],
                                       reference.grouped[j],
                                       result.grouped[j],
                                       m_outputSelections.at(j));
                    }
                }
                else if(auto problem = dynamic_cast<ContractionProblemGemm*>(m_problem))
                {
                    auto reference = dynamic_cast<ContractionInputs const&>(*m_referenceInputs);
                    auto result    = dynamic_cast<ContractionInputs const&>(*inputs);
                    rv = validate(*problem, reference, result, m_outputSelections.front());
                }
                else
                {
                    throw std::runtime_error("Failed to cast to any ContractionProblem.");
                }
            }

            return rv;
        }

        void ReferenceValidator::validateWarmups(std::shared_ptr<ProblemInputs> inputs,
                                                 TimingEvents const&            startEvents,
                                                 TimingEvents const&            stopEvents)
        {
            if(m_enabled && !m_validatedSolution)
            {
                validateSolution(inputs);
                m_validatedSolution = true;
            }
        }

        bool ReferenceValidator::checkResults(
            TensorDescriptor const&                   tensor,
            void const*                               refPtr,
            void const*                               resPtr,
            size_t                                    maxElements,
            bool                                      isgpu,
            const roc::hostnumerics::OutputSelection& outputSelection,
            double                                    threshold)
        {
            const roc::hostnumerics::ScalarType scalarType
                = toHostNumericsScalarType(tensor.dataType());
            const size_t storageBits = roc::hostnumerics::scalarTypeInfo(scalarType).storageBits;
            if(storageBits % 8 != 0)
            {
                throw std::runtime_error(
                    "Sub-byte output validation requires a packed readback adapter.");
            }

            const size_t elementBytes       = storageBits / 8;
            size_t       elementsToCopy     = tensor.totalAllocatedElements();
            size_t       elementsBeforeData = 0;
            size_t       elementsAfterData  = 0;

            const BoundsCheckMode boundsCheck = m_dataInit->getCurBoundsCheck();
            if(boundsCheck == BoundsCheckMode::NaN)
                elementsToCopy = maxElements;

            const bool hasNullPointer  = resPtr == nullptr || refPtr == nullptr;
            const bool hasZeroElements = elementsToCopy == 0 || maxElements == 0;
            if(shouldSkipNullTensor(tensor.getName(), hasNullPointer, hasZeroElements))
                return true;
            if(hasNullPointer || hasZeroElements)
            {
                std::stringstream ss;
                ss << "Unexpected null pointer or no data for tensor " << tensor.getName()
                   << " (result=" << resPtr << ", reference=" << refPtr
                   << ", elementsToCopy=" << elementsToCopy << ", maxElements=" << maxElements
                   << ")";
                throw std::runtime_error(ss.str());
            }

            if(elementsToCopy > std::numeric_limits<size_t>::max() / elementBytes)
                throw std::overflow_error("Validation readback byte count overflow.");
            const size_t bytesToCopy = elementsToCopy * elementBytes;
            allocateResultBuffer(bytesToCopy);

            void const* copySource = resPtr;
            if(boundsCheck == BoundsCheckMode::NaN)
            {
                if(maxElements < tensor.totalAllocatedElements())
                    throw std::runtime_error(
                        "Validation guard allocation is smaller than the tensor.");
                const ptrdiff_t paddingElements = maxElements - tensor.totalAllocatedElements();
                size_t paddingBytes = multiplyElementSize(paddingElements, tensor.elementBytes());
                const size_t alignmentBytes
                    = 2 * static_cast<size_t>(std::ceil(std::max(1.0f, tensor.elementBytes())));
                paddingBytes                 = paddingBytes / alignmentBytes * alignmentBytes;
                const size_t bytesBeforeData = paddingBytes / 2;
                if(bytesBeforeData % elementBytes != 0)
                    throw std::runtime_error("Validation guard offset is not element-aligned.");

                copySource         = static_cast<uint8_t const*>(resPtr) - bytesBeforeData;
                elementsBeforeData = bytesBeforeData / elementBytes;
            }

            if(elementsToCopy < elementsBeforeData + tensor.totalAllocatedElements())
                throw std::runtime_error("Validation guard allocation is smaller than the tensor.");
            elementsAfterData
                = elementsToCopy - elementsBeforeData - tensor.totalAllocatedElements();

            {
                ScopedTimer timer("validate_gpu_readback");
                const auto  copyKind = isgpu ? hipMemcpyDeviceToHost : hipMemcpyHostToHost;
                HIP_CHECK_EXC(
                    hipMemcpy(m_cpuResultBuffer.get(), copySource, bytesToCopy, copyKind));
            }

            std::vector<size_t>    dimensions(tensor.sizes().begin(), tensor.sizes().end());
            std::vector<ptrdiff_t> strides;
            strides.reserve(tensor.strides().size());
            for(const auto stride : tensor.strides())
                strides.push_back(static_cast<ptrdiff_t>(stride));
            const roc::hostnumerics::Layout layout(
                roc::hostnumerics::Shape(std::move(dimensions)), std::move(strides), 0);

            if(tensor.totalAllocatedElements() > std::numeric_limits<size_t>::max() / elementBytes)
                throw std::overflow_error("Validation tensor byte count overflow.");
            const size_t allocatedBytes = tensor.totalAllocatedElements() * elementBytes;
            const auto   referenceStorage
                = std::span<const std::byte>(static_cast<const std::byte*>(refPtr), allocatedBytes);
            const auto resultStorage
                = std::span<std::byte>(reinterpret_cast<std::byte*>(m_cpuResultBuffer.get())
                                           + elementsBeforeData * elementBytes,
                                       allocatedBytes);
            const roc::hostnumerics::Tensor resultTensor
                = roc::hostnumerics::Tensor::shareExternalMutableBackingStorage(
                    scalarType, layout, m_cpuResultBuffer, resultStorage);
            const roc::hostnumerics::Tensor referenceTensor
                = roc::hostnumerics::Tensor::copyEncodedBackingStorage(
                    scalarType, layout, referenceStorage);

            roc::hostnumerics::ComparisonOptions options
                = validationComparisonOptions(tensor.dataType(), threshold);
            options.selection                    = outputSelection;
            options.computeElementwiseStatistics = false;
            options.computeFrobenius             = false;
            options.reportMatchingElements       = m_printValids;
            options.maxReportedMismatches = m_printMax > 0 ? static_cast<size_t>(m_printMax) : 0;

            roc::hostnumerics::ComparisonReport comparison;
            {
                ScopedTimer timer("validate_element_comparison");
                comparison = roc::hostnumerics::compare(resultTensor, referenceTensor, options);
            }

            const bool isComplex = roc::hostnumerics::scalarTypeInfo(scalarType).category
                                   == roc::hostnumerics::ScalarCategory::Complex;
            const auto& samples
                = m_printValids ? comparison.reportedComparisons : comparison.reportedMismatches;
            if(!samples.empty() && m_printMax > 0)
            {
                ScopedTimer timer("validate_mismatch_printing");
                std::cout << "Index:  Device | Reference" << std::endl;
                size_t printed = 0;
                for(const auto& sample : samples)
                {
                    std::cout << "[" << printed++ << "] elem=" << sample.index
                              << " idx=" << sample.observedOffset << ": ";
                    if(isComplex)
                    {
                        std::cout << "(" << sample.observed << "," << sample.observedImaginary
                                  << ")" << (sample.matched ? "==" : "!=") << "(" << sample.expected
                                  << "," << sample.expectedImaginary << ")";
                    }
                    else
                    {
                        std::cout << sample.observed << (sample.matched ? "==" : "!=")
                                  << sample.expected;
                    }
                    std::cout << std::endl;
                }
            }
            if(comparison.mismatches != 0 && m_printMax > 0)
            {
                std::cout << "Found " << comparison.mismatches << " incorrect values in "
                          << comparison.compared << " total values compared." << std::endl;
            }

            roc::hostnumerics::SentinelReport sentinel;
            const auto                        completeStorage = std::span<const std::byte>(
                reinterpret_cast<const std::byte*>(m_cpuResultBuffer.get()), bytesToCopy);
            if(elementsBeforeData != 0)
            {
                sentinel.append(roc::hostnumerics::checkUnwrittenSentinel(
                                    scalarType,
                                    completeStorage,
                                    0,
                                    elementsBeforeData,
                                    roc::hostnumerics::SentinelRegion::Before,
                                    options.maxReportedMismatches),
                                options.maxReportedMismatches);
            }
            if(boundsCheck == BoundsCheckMode::NaN && (outputSelection.selectsAll() || m_freivalds))
            {
                sentinel.append(roc::hostnumerics::checkUnusedTensorStorage(
                                    resultTensor,
                                    tensor.totalAllocatedElements(),
                                    roc::hostnumerics::SentinelRegion::Inside,
                                    options.maxReportedMismatches),
                                options.maxReportedMismatches);
            }
            if(elementsAfterData != 0)
            {
                sentinel.append(roc::hostnumerics::checkUnwrittenSentinel(
                                    scalarType,
                                    completeStorage,
                                    elementsBeforeData + tensor.totalAllocatedElements(),
                                    elementsAfterData,
                                    roc::hostnumerics::SentinelRegion::After,
                                    options.maxReportedMismatches),
                                options.maxReportedMismatches);
            }

            if(sentinel.checked != 0 && m_printMax > 0)
            {
                std::cout << "Performed bounds check on " << sentinel.checked << " elements."
                          << std::endl;
            }
            for(const auto& mismatch : sentinel.reportedMismatches)
            {
                const char* location = "near";
                switch(mismatch.region)
                {
                case roc::hostnumerics::SentinelRegion::Before:
                    location = "before";
                    break;
                case roc::hostnumerics::SentinelRegion::Inside:
                    location = "inside";
                    break;
                case roc::hostnumerics::SentinelRegion::After:
                    location = "after";
                    break;
                case roc::hostnumerics::SentinelRegion::Unspecified:
                case roc::hostnumerics::SentinelRegion::Count:
                    break;
                }
                std::cout << "Value written " << location << " output buffer at index "
                          << mismatch.index << ": found " << mismatch.observed.real
                          << " instead of the unwritten sentinel" << std::endl;
            }

            const bool failed = !comparison.passed() || !sentinel.passed();
            if(failed)
            {
                m_errorInSolution = true;
                m_error           = true;
                std::cout << "Check failed in output tensor: " << tensor << std::endl;
            }
            return failed;
        }

        bool ReferenceValidator::shouldSkipNullTensor(const std::string& tensorName,
                                                      bool               hasNullPointer,
                                                      bool               hasZeroElements) const
        {
            // Only output tensors reach this function (filtered by isOutput() check)
            // Output tensors should never have null pointers or zero elements
            return false;
        }

        bool ReferenceValidator::validate(ContractionProblemGemm const&             problem,
                                          ContractionInputs const&                  reference,
                                          ContractionInputs const&                  result,
                                          const roc::hostnumerics::OutputSelection& outputSelection)
        {
            if(problem.tensors().empty())
                return false;

            bool rv = true;

            if(m_printAny)
                printTensors(problem, reference, result);

            auto   k = problem.transA() ? problem.a().sizes().at(0) : problem.a().sizes().at(1);
            bool   isTF32    = (problem.f32XdlMathOp() == rocisa::DataType::XFloat32);
            bool   isTF32x1  = (problem.computeInputTypeA() == rocisa::DataType::BFloat16
                             && problem.computeInputTypeB() == rocisa::DataType::BFloat16
                             && problem.computeType() == rocisa::DataType::Float
                             && problem.a().dataType() == rocisa::DataType::Float
                             && problem.b().dataType() == rocisa::DataType::Float);
            double threshold = -1.0;
            if(isTF32)
            {
                threshold = 0.01 * sqrt(double(k));
            }
            else if(isTF32x1)
            {
                threshold = 0.3 * sqrt(double(k));
            }

            for(size_t i = 0; i < problem.tensors().size(); i++)
            {
                auto& tensor = problem.tensors()[i];
                if(!tensor.isOutput())
                    continue;

                const auto tensorRole = static_cast<ContractionProblemGemm::TENSOR>(i);
                const roc::hostnumerics::OutputSelection comparisonSelection
                    = tensorRole == ContractionProblemGemm::TENSOR::D
                              || tensorRole == ContractionProblemGemm::TENSOR::E
                          ? outputSelection
                          : referenceOutputSelection(tensor,
                                                     static_cast<size_t>(m_elementsToValidate));

                void const* refPtr = nullptr;
                void const* resPtr = nullptr;
                switch(tensorRole)
                {
                case ContractionProblemGemm::TENSOR::A:
                {
                    refPtr = reference.a;
                    resPtr = result.a;
                }
                break;
                case ContractionProblemGemm::TENSOR::B:
                {
                    refPtr = reference.b;
                    resPtr = result.b;
                }
                break;
                case ContractionProblemGemm::TENSOR::C:
                {
                    refPtr = reference.c;
                    resPtr = result.c;
                }
                break;
                case ContractionProblemGemm::TENSOR::D:
                {
                    refPtr = reference.d;
                    resPtr = result.d;
                }
                break;
                case ContractionProblemGemm::TENSOR::E:
                {
                    refPtr = reference.e;
                    resPtr = result.e;
                }
                break;
                case ContractionProblemGemm::TENSOR::BIAS:
                {
                    refPtr = reference.bias;
                    resPtr = result.bias;
                }
                break;
                case ContractionProblemGemm::TENSOR::GATE_RESIDUAL:
                {
                    refPtr = reference.gateResidual;
                    resPtr = result.gateResidual;
                }
                break;
                case ContractionProblemGemm::TENSOR::SCALEA:
                {
                    refPtr = reference.scaleA;
                    resPtr = result.scaleA;
                }
                break;
                case ContractionProblemGemm::TENSOR::SCALEB:
                {
                    refPtr = reference.scaleB;
                    resPtr = result.scaleB;
                }
                break;
                case ContractionProblemGemm::TENSOR::SCALEC:
                {
                    refPtr = reference.scaleC;
                    resPtr = result.scaleC;
                }
                break;
                case ContractionProblemGemm::TENSOR::SCALED:
                {
                    refPtr = reference.scaleD;
                    resPtr = result.scaleD;
                }
                break;
                case ContractionProblemGemm::TENSOR::SCALEALPHAVEC:
                {
                    refPtr = reference.scaleAlphaVec;
                    resPtr = result.scaleAlphaVec;
                }
                break;
                case ContractionProblemGemm::TENSOR::Synchronizer:
                {
                    refPtr = reference.Synchronizer;
                    resPtr = result.Synchronizer;
                }
                break;
                case ContractionProblemGemm::TENSOR::AMAXD:
                {
                    refPtr = reference.amaxD;
                    resPtr = result.amaxD;
                }

                break;
                default:
                    throw std::runtime_error("Unrecognized output tensor.");
                }

                TensorDescriptor const* validationTensor = &tensor;

                if(Debug::Instance().printTensorInfo())
                    std::cout << "Validating tensor " << validationTensor->getName()
                              << ", cpu pointer " << refPtr << ", gpu pointer " << resPtr
                              << ", size = " << result.maxElements[i] << std::endl;

                // Check if we should skip this tensor due to null pointers or zero elements
                bool hasNullPointer  = (resPtr == nullptr || refPtr == nullptr);
                bool hasZeroElements = (result.maxElements[i] == 0);

                if(shouldSkipNullTensor(tensor.getName(), hasNullPointer, hasZeroElements))
                {
                    continue;
                }

                // If we reach here with null pointers or zero elements, it's an error
                if(hasNullPointer || hasZeroElements)
                {
                    std::stringstream ss;
                    ss << "Unexpected null pointer or zero elements for tensor " << tensor.getName()
                       << " (resPtr=" << resPtr << ", refPtr=" << refPtr
                       << ", maxElements=" << result.maxElements[i] << ")";
                    throw std::runtime_error(ss.str());
                }

                rv &= checkResults(*validationTensor,
                                   refPtr,
                                   resPtr,
                                   result.maxElements[i],
                                   result.gpu,
                                   comparisonSelection,
                                   threshold);
            }
            return rv;
        }

        void ReferenceValidator::allocateResultBuffer(size_t bytes)
        {
            // Only skip reallocation if size matches AND buffer is valid
            if(m_cpuResultBufferSize == bytes && m_cpuResultBuffer.get() != nullptr)
                return;

            m_cpuResultBuffer.reset();

            uint8_t* buffer;
            HIP_CHECK_EXC(hipHostMalloc((void**)&buffer, bytes, 0));
            m_cpuResultBuffer.reset(buffer, [](uint8_t* p) { HIP_CHECK_EXC(hipHostFree(p)); });
            m_cpuResultBufferSize = bytes;
        }

        void ReferenceValidator::printTensors(ContractionProblemGemm const& problem,
                                              ContractionInputs const&      reference,
                                              ContractionInputs const&      result)
        {
            size_t requiredBufferSize = 0;

            std::cout << "reference alpha: " << ToString(reference.alpha)
                      << ", beta: " << ToString(reference.beta) << std::endl;
            std::cout << "result    alpha: " << ToString(result.alpha)
                      << ", beta: " << ToString(result.beta) << std::endl;

            if(m_printTensorA)
                requiredBufferSize
                    = std::max(requiredBufferSize, problem.a().totalAllocatedBytes());
            if(m_printTensorB)
                requiredBufferSize
                    = std::max(requiredBufferSize, problem.b().totalAllocatedBytes());
            if(m_printTensorC)
                requiredBufferSize
                    = std::max(requiredBufferSize, problem.c().totalAllocatedBytes());
            if(m_printTensorD)
                requiredBufferSize
                    = std::max(requiredBufferSize, problem.d().totalAllocatedBytes());
            if(m_printTensorRef)
                requiredBufferSize
                    = std::max(requiredBufferSize, problem.d().totalAllocatedBytes());
            if(m_printTensorBias)
                requiredBufferSize
                    = std::max(requiredBufferSize, problem.bias().totalAllocatedBytes());
            if(m_printTensorGate)
                requiredBufferSize
                    = std::max(requiredBufferSize, problem.gateResidual().totalAllocatedBytes());
            if(m_printTensorScaleAlphaVec)
                requiredBufferSize
                    = std::max(requiredBufferSize, problem.scaleAlphaVec().totalAllocatedBytes());
            if(m_printTensorAmaxD)
                requiredBufferSize
                    = std::max(requiredBufferSize, problem.amaxd().totalAllocatedBytes());

            allocateResultBuffer(requiredBufferSize);

            if(m_printTensorA)
            {
                m_reporter->logTensor(
                    LogLevel::Verbose, "A", reference.a, problem.a(), reference.a);
                if(problem.a().dataType() == rocisa::DataType::Float4 && problem.mxBlockA() > 0)
                {
                    m_reporter->logTensor(
                        LogLevel::Verbose, "MXSA", reference.mxsa, problem.mxsa(), reference.mxsa);
                }
                if(problem.sparse() && problem.sparse() != 2)
                {
                    m_reporter->logTensor(LogLevel::Verbose,
                                          "Compressed A",
                                          reference.compressed,
                                          problem.compressed(),
                                          reference.compressed);
                }
            }

            if(m_printTensorB)
            {
                m_reporter->logTensor(
                    LogLevel::Verbose, "B", reference.b, problem.b(), reference.b);
                if(problem.b().dataType() == rocisa::DataType::Float4 && problem.mxBlockB() > 0)
                {
                    m_reporter->logTensor(
                        LogLevel::Verbose, "MXSB", reference.mxsb, problem.mxsb(), reference.mxsb);
                }
                if(problem.sparse() && problem.sparse() == 2)
                {
                    m_reporter->logTensor(LogLevel::Verbose,
                                          "Compressed B",
                                          reference.compressed,
                                          problem.compressed(),
                                          reference.compressed);
                }
            }

            if(m_printTensorA || m_printTensorB)
            {
                if(problem.sparse())
                {
                    m_reporter->logTensor(LogLevel::Verbose,
                                          "Metadata",
                                          reference.metadata,
                                          problem.metadata(),
                                          reference.metadata);
                }
            }

            if(result.c == result.d && (m_printTensorC || m_printTensorD))
            {
                // If the pointers are the same, only print the buffer once.
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.get(),
                                        result.c,
                                        problem.c().totalAllocatedBytes(),
                                        hipMemcpyDeviceToHost));
                m_reporter->logTensor(
                    LogLevel::Verbose, "C_D", m_cpuResultBuffer.get(), problem.c(), result.c);
            }
            else
            {
                if(m_printTensorC)
                {
                    HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.get(),
                                            result.c,
                                            problem.c().totalAllocatedBytes(),
                                            hipMemcpyDeviceToHost));
                    m_reporter->logTensor(
                        LogLevel::Verbose, "C", m_cpuResultBuffer.get(), problem.c(), result.c);
                }

                if(m_printTensorD)
                {
                    HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.get(),
                                            result.d,
                                            problem.d().totalAllocatedBytes(),
                                            hipMemcpyDeviceToHost));
                    m_reporter->logTensor(
                        LogLevel::Verbose, "D", m_cpuResultBuffer.get(), problem.d(), result.d);
                }
            }

            if(m_printTensorRef)
            {
                m_reporter->logTensor(
                    LogLevel::Verbose, "Ref", reference.d, problem.d(), reference.d);
            }

            if(m_printTensorBias)
            {
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.get(),
                                        result.bias,
                                        problem.bias().totalAllocatedBytes(),
                                        hipMemcpyDeviceToHost));
                m_reporter->logTensor(LogLevel::Verbose,
                                      "bias",
                                      m_cpuResultBuffer.get(),
                                      problem.bias(),
                                      result.bias);
            }
            if(m_printTensorGate)
            {
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.get(),
                                        result.gateResidual,
                                        problem.gateResidual().totalAllocatedBytes(),
                                        hipMemcpyDeviceToHost));
                m_reporter->logTensor(LogLevel::Verbose,
                                      "gateResidual",
                                      m_cpuResultBuffer.get(),
                                      problem.gateResidual(),
                                      result.gateResidual);
            }
            if(m_printTensorScaleAlphaVec)
            {
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.get(),
                                        result.scaleAlphaVec,
                                        problem.scaleAlphaVec().totalAllocatedBytes(),
                                        hipMemcpyDeviceToHost));
                m_reporter->logTensor(LogLevel::Verbose,
                                      "scaleAlphaVec",
                                      m_cpuResultBuffer.get(),
                                      problem.scaleAlphaVec(),
                                      result.scaleAlphaVec);
            }

            if(m_printTensorAmaxD)
            {
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.get(),
                                        result.amaxD,
                                        problem.amaxd().totalAllocatedBytes(),
                                        hipMemcpyDeviceToHost));
                m_reporter->logTensor(LogLevel::Verbose,
                                      "AmaxD Ref",
                                      reference.amaxD,
                                      problem.amaxd(),
                                      reference.amaxD);
                m_reporter->logTensor(LogLevel::Verbose,
                                      "AmaxD GPU",
                                      m_cpuResultBuffer.get(),
                                      problem.amaxd(),
                                      result.amaxD);
            }
        }

        void ReferenceValidator::postSolution()
        {
            ScopedTimer timer("post_solution_validation");
            if(!m_executedSolution)
                return;

            if(m_enabled && !m_validatedSolution)
                return;

            if(m_freivalds || m_elementsToValidate != 0)
            {
                if(m_errorInSolution)
                {
                    m_errorsReported++;
                    m_reporter->report(ResultKey::Validation, "FAILED");
                }
                else
                    m_reporter->report(ResultKey::Validation, "PASSED");
            }
            else
            {
                m_reporter->report(ResultKey::Validation, "NO_CHECK");
            }

            m_errorInSolution = false;
        }

        void ReferenceValidator::postProblem() {}

        void ReferenceValidator::finalizeReport() {}

        int ReferenceValidator::error() const
        {
            return m_errorsReported;
        }
    } // namespace Client
} // namespace TensileLite
