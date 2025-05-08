/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "DataInitializationTyped.hpp"
#include "ResultComparison.hpp"
#include "ResultReporter.hpp"

#include "Reference.hpp"

#include <Tensile/DataTypes.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <cstddef>

namespace Tensile
{
    namespace Client
    {
        ReferenceValidator::ReferenceValidator(po::variables_map const&            args,
                                               std::shared_ptr<DataInitialization> dataInit,
                                               hipStream_t                         stream)

            : m_dataInit(dataInit)
            , m_stream(stream)
        {
            m_elementsToValidate = args["num-elements-to-validate"].as<int>();
            m_printValids        = args["print-valids"].as<bool>();
            m_printMax           = args["print-max"].as<int>();

            m_printTensorA   = args["print-tensor-a"].as<bool>();
            m_printTensorB   = args["print-tensor-b"].as<bool>();
            m_printTensorC   = args["print-tensor-c"].as<bool>();
            m_printTensorD   = args["print-tensor-d"].as<bool>();
            m_printTensorRef = args["print-tensor-ref"].as<bool>();

            m_printAny = m_printTensorA || m_printTensorB || m_printTensorC || m_printTensorD
                         || m_printTensorRef;

            m_convolutionVsContraction = args["convolution-vs-contraction"].as<bool>();
            if(args.count("convolution-identifier"))
                m_convolutionProblem.FromIdentifier(
                    args["convolution-identifier"].as<std::string>());

            m_enabled = m_elementsToValidate != 0 || m_printAny;
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

        void ReferenceValidator::preProblem(ContractionProblem const& problem)
        {
            if(m_enabled)
            {
                m_problem          = problem;
                m_referenceInputs  = m_dataInit->prepareCPUInputs(problem);
                m_validationStride = 1;
                if(m_elementsToValidate > 0
                   && m_elementsToValidate < problem.d().totalLogicalElements())
                    m_validationStride
                        = NextPrime(problem.d().totalAllocatedElements() / m_elementsToValidate);

                SolveCPU(problem, *m_referenceInputs, m_validationStride);

                if(m_convolutionVsContraction)
                {

                    SolveCPUConvolution(
                        m_convolutionProblem, problem, *(m_dataInit->cpuConvInputs()));
                    // std::cout << "ValidateConv--Start\n";
                    m_errorInConvolutionVsContraction = validateSolution(
                        m_dataInit->cpuConvInputs()); // validate conv against reference
                    // TODO - print problem dimensions??
                    std::cout << m_convolutionProblem << " vs " << problem.operationIdentifier()
                              << " :  ";
                    if(m_errorInConvolutionVsContraction)
                    {
                        std::cout << "FAILED_CONV";
                    }
                    else
                    {
                        std::cout << "PASSED_CONV";
                    }
                    std::cout << "\n";
                }
            }
        }

        void ReferenceValidator::preSolution(ContractionSolution const& solution)
        {
            m_validatedSolution = false;
            m_errorInSolution   = false;
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

        void ReferenceValidator::postWarmup() {}

        template <typename ManagedInputs>
        bool ReferenceValidator::validateSolutionCast(std::shared_ptr<ContractionInputs> inputs)
        {
            auto const& typedReference = dynamic_cast<ManagedInputs const&>(*m_referenceInputs);
            auto const& typedResult    = dynamic_cast<ManagedInputs const&>(*inputs);

            auto rv = validateTyped(typedReference, typedResult);

            if(0 and inputs == m_dataInit->cpuConvInputs())
            {
                m_reporter->logTensor(
                    LogLevel::Verbose, "Aval-conv", typedResult.a, m_problem.a(), nullptr);
                m_reporter->logTensor(
                    LogLevel::Verbose, "Bval-conv", typedResult.b, m_problem.b(), nullptr);
                m_reporter->logTensor(
                    LogLevel::Verbose, "Dval-conv", typedResult.d, m_problem.d(), nullptr);
                m_reporter->logTensor(LogLevel::Verbose,
                                      "Bval-contraction",
                                      typedReference.b,
                                      m_problem.b(),
                                      nullptr);
                m_reporter->logTensor(LogLevel::Verbose,
                                      "Dval-contraction",
                                      typedReference.d,
                                      m_problem.d(),
                                      nullptr);
            }

            return rv;
        }

        bool ReferenceValidator::validateSolution(std::shared_ptr<ContractionInputs> inputs)
        {
            // retreive alpha/beta type set via setAlpha/BetaType()
            auto alphaType = m_problem.alphaType();
            auto betaType  = m_problem.betaType();

            // Backward-compatible: when setAlpha/BetaType() wasn't called, use the old way
            // Could remove after rocBLAS is updated
            if(alphaType == DataType::None)
            {
                alphaType = m_problem.a().dataType() == DataType::BFloat16
                                ? DataType::Float
                                : m_problem.d().dataType();
            }
            if(betaType == DataType::None)
            {
                betaType = alphaType;
            }

            auto contractionInputsTypeId = ContractionInputs::TypeId(m_problem.a().dataType(),
                                                                     m_problem.b().dataType(),
                                                                     m_problem.c().dataType(),
                                                                     m_problem.d().dataType(),
                                                                     alphaType,
                                                                     betaType);

            switch(contractionInputsTypeId)
            {
            case ManagedContractionInputs_S_S_S::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_S_S_S>(inputs);
            }
            case ManagedContractionInputs_D_D_D::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_D_D_D>(inputs);
            }
            case ManagedContractionInputs_C_C_C::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_C_C_C>(inputs);
            }
            case ManagedContractionInputs_Z_Z_Z::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_Z_Z_Z>(inputs);
            }
#ifdef TENSILE_USE_HALF
            case ManagedContractionInputs_H_H_H::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_H_H_H>(inputs);
            }
            case ManagedContractionInputs_H_H_S::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_H_H_S>(inputs);
            }
            case ManagedContractionInputs_H_S_S::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_H_S_S>(inputs);
            }
#endif // TENSILE_USE_HALF
            case ManagedContractionInputs_I8x4_I32_I32::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_I8x4_I32_I32>(inputs);
            }
            case ManagedContractionInputs_I32_I32_I32::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_I32_I32_I32>(inputs);
            }
            case ManagedContractionInputs_I8_I32_I32::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_I8_I32_I32>(inputs);
            }
#ifdef TENSILE_USE_BF16
            case ManagedContractionInputs_B_B_S::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_B_B_S>(inputs);
            }
            case ManagedContractionInputs_B_S_S::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_B_S_S>(inputs);
            }
#endif // TENSILE_USE_BF16
#ifdef TENSILE_USE_FP8_BF8
            case ManagedContractionInputs_F8_F8_S::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_F8_F8_S>(inputs);
            }
            case ManagedContractionInputs_F8_S_S::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_F8_S_S>(inputs);
            }
            case ManagedContractionInputs_B8_B8_S::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_B8_B8_S>(inputs);
            }
            case ManagedContractionInputs_B8_S_S::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_B8_S_S>(inputs);
            }
            // hybrid cases: F8B8SS, B8F8SS
            case ManagedContractionInputs_F8B8_S_S::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_F8B8_S_S>(inputs);
            }
            case ManagedContractionInputs_B8F8_S_S::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_B8F8_S_S>(inputs);
            }
            // hybrid cases with To = BF8
            case ManagedContractionInputs_F8B8_B8_S::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_F8B8_B8_S>(inputs);
            }
            case ManagedContractionInputs_B8F8_B8_S::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_B8F8_B8_S>(inputs);
            }
            // cases with To = F16
            case ManagedContractionInputs_F8_H_S::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_F8_H_S>(inputs);
            }
            case ManagedContractionInputs_B8_H_S::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_B8_H_S>(inputs);
            }
            case ManagedContractionInputs_F8B8_H_S::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_F8B8_H_S>(inputs);
            }
            case ManagedContractionInputs_B8F8_H_S::TypeId():
            {
                return validateSolutionCast<ManagedContractionInputs_B8F8_H_S>(inputs);
            }
#endif // TENSILE_USE_FP8_BF8
            default:;
            }

            throw std::runtime_error("Data type not implemented.");
        }

        void ReferenceValidator::validateWarmups(std::shared_ptr<ContractionInputs> inputs,
                                                 TimingEvents const&                startEvents,
                                                 TimingEvents const&                stopEvents)
        {
            if(m_enabled && !m_validatedSolution)
            {
                validateSolution(inputs);
                m_validatedSolution = true;
            }
        }

        template <typename ManagedInputs>
        bool ReferenceValidator::validateTyped(ManagedInputs const& reference,
                                               ManagedInputs const& result)
        {
            bool rv = false;
            if(!m_enabled)
                return rv;

            if(m_printAny)
                printTensorsTyped(reference, result);

            if(m_elementsToValidate != 0)
            {
                // RMSComparison<typename ManagedInputs::DType> compareValid(1e-7,
                // m_printMax > 0);
                PointwiseComparison<typename ManagedInputs::DType> compareValid(
                    m_printValids, m_printMax, m_printMax > 0);
                InvalidComparison<typename ManagedInputs::DType> compareInvalid(m_printMax,
                                                                                m_printMax > 0);
                rv = checkResultsTyped(reference, result, compareValid, compareInvalid);
            }

            return rv;
        }

        void ReferenceValidator::allocateResultBuffer(size_t bytes)
        {
            if(m_cpuResultBufferSize == bytes)
                return;
            m_cpuResultBuffer.reset();

            uint8_t* buffer;
            HIP_CHECK_EXC(hipHostMalloc(&buffer, bytes, 0));
            m_cpuResultBuffer.reset(buffer, hipFree);
            m_cpuResultBufferSize = bytes;
        }

        template <typename ManagedInputs>
        void ReferenceValidator::printTensorsTyped(ManagedInputs const& reference,
                                                   ManagedInputs const& result)
        {
            size_t requiredBufferSize = 0;

            std::cout << "reference alpha: " << reference.alpha << ", beta: " << reference.beta
                      << std::endl;
            std::cout << "result    alpha: " << result.alpha << ", beta: " << result.beta
                      << std::endl;

            if(m_printTensorA)
                requiredBufferSize
                    = std::max(requiredBufferSize, m_problem.a().totalAllocatedBytes());
            if(m_printTensorB)
                requiredBufferSize
                    = std::max(requiredBufferSize, m_problem.b().totalAllocatedBytes());
            if(m_printTensorC)
                requiredBufferSize
                    = std::max(requiredBufferSize, m_problem.c().totalAllocatedBytes());
            if(m_printTensorD)
                requiredBufferSize
                    = std::max(requiredBufferSize, m_problem.d().totalAllocatedBytes());
            if(m_printTensorRef)
                requiredBufferSize
                    = std::max(requiredBufferSize, m_problem.d().totalAllocatedBytes());

            if(m_cpuResultBufferSize < requiredBufferSize)
                allocateResultBuffer(requiredBufferSize);

            if(m_printTensorA)
            {
                HIP_CHECK_EXC(hipMemcpyAsync(m_cpuResultBuffer.get(),
                                             result.a,
                                             m_problem.a().totalAllocatedBytes(),
                                             hipMemcpyDeviceToHost,
                                             m_stream));
                HIP_CHECK_EXC(hipStreamSynchronize(m_stream));
                auto const* buffer = reinterpret_cast<typename ManagedInputs::AType const*>(
                    m_cpuResultBuffer.get());

                m_reporter->logTensor(LogLevel::Verbose, "A", buffer, m_problem.a(), result.a);
            }

            if(m_printTensorB)
            {
                HIP_CHECK_EXC(hipMemcpyAsync(m_cpuResultBuffer.get(),
                                             result.b,
                                             m_problem.b().totalAllocatedBytes(),
                                             hipMemcpyDeviceToHost,
                                             m_stream));
                HIP_CHECK_EXC(hipStreamSynchronize(m_stream));
                auto const* buffer = reinterpret_cast<typename ManagedInputs::BType const*>(
                    m_cpuResultBuffer.get());

                m_reporter->logTensor(LogLevel::Verbose, "B", buffer, m_problem.b(), result.b);
            }

            if(result.c == result.d && (m_printTensorC || m_printTensorD))
            {
                // If the pointers are the same, only print the buffer once.
                HIP_CHECK_EXC(hipMemcpyAsync(m_cpuResultBuffer.get(),
                                             result.c,
                                             m_problem.c().totalAllocatedBytes(),
                                             hipMemcpyDeviceToHost,
                                             m_stream));
                HIP_CHECK_EXC(hipStreamSynchronize(m_stream));
                auto const* buffer = reinterpret_cast<typename ManagedInputs::CType const*>(
                    m_cpuResultBuffer.get());

                m_reporter->logTensor(LogLevel::Verbose, "C_D", buffer, m_problem.c(), result.c);
            }
            else
            {
                if(m_printTensorC)
                {
                    HIP_CHECK_EXC(hipMemcpyAsync(m_cpuResultBuffer.get(),
                                                 result.c,
                                                 m_problem.c().totalAllocatedBytes(),
                                                 hipMemcpyDeviceToHost,
                                                 m_stream));
                    HIP_CHECK_EXC(hipStreamSynchronize(m_stream));
                    auto const* buffer = reinterpret_cast<typename ManagedInputs::CType const*>(
                        m_cpuResultBuffer.get());

                    m_reporter->logTensor(LogLevel::Verbose, "C", buffer, m_problem.c(), result.c);
                }

                if(m_printTensorD)
                {
                    HIP_CHECK_EXC(hipMemcpyAsync(m_cpuResultBuffer.get(),
                                                 result.d,
                                                 m_problem.d().totalAllocatedBytes(),
                                                 hipMemcpyDeviceToHost,
                                                 m_stream));
                    HIP_CHECK_EXC(hipStreamSynchronize(m_stream));
                    auto const* buffer = reinterpret_cast<typename ManagedInputs::DType const*>(
                        m_cpuResultBuffer.get());

                    m_reporter->logTensor(LogLevel::Verbose, "D", buffer, m_problem.d(), result.d);
                }
            }

            if(m_printTensorRef)
            {
                m_reporter->logTensor(
                    LogLevel::Verbose, "Ref", reference.d, m_problem.d(), reference.d);
            }
        }

        template <typename ManagedInputs, typename CompareValid, typename CompareInvalid>
        bool ReferenceValidator::checkResultsTyped(ManagedInputs const& reference,
                                                   ManagedInputs const& result,
                                                   CompareValid&        compareValid,
                                                   CompareInvalid&      compareInvalid)
        {
            using Type         = typename ManagedInputs::DType;
            auto const& tensor = m_problem.d();

            size_t elementsToCopy       = tensor.totalAllocatedElements();
            size_t elementsOffsetToCopy = 0;
            size_t elementsBeforeData   = 0;
            size_t elementsAfterData    = 0;

            BoundsCheckMode boundsCheck = m_dataInit->getCurBoundsCheck();
            if(boundsCheck == BoundsCheckMode::NaN)
                elementsToCopy = result.dElements;
            size_t bytesToCopy = elementsToCopy * sizeof(Type);

            if(m_cpuResultBufferSize < bytesToCopy)
                allocateResultBuffer(bytesToCopy);

            if(boundsCheck == BoundsCheckMode::GuardPageBack)
                elementsOffsetToCopy = result.dElements - tensor.totalAllocatedElements();

            auto copykind = result.gpu ? hipMemcpyDeviceToHost : hipMemcpyHostToHost;

            HIP_CHECK_EXC(hipMemcpyAsync(m_cpuResultBuffer.get(),
                                         result.managedD.get() + elementsOffsetToCopy,
                                         bytesToCopy,
                                         copykind,
                                         m_stream));
            HIP_CHECK_EXC(hipStreamSynchronize(m_stream));

            if(boundsCheck == BoundsCheckMode::NaN)
            {
                elementsBeforeData = result.d - result.managedD.get();
                elementsAfterData
                    = elementsToCopy - (tensor.totalAllocatedElements() + elementsBeforeData);
            }
            // If there was extra data allocated before the tensor to do bounds
            // checking, resultBuffer is the whole allocation, while resultData
            // points directly to the result.
            Type const* resultBuffer    = reinterpret_cast<Type const*>(m_cpuResultBuffer.get());
            Type const* resultData      = resultBuffer + elementsBeforeData;
            Type const* resultAfterData = resultData + tensor.totalAllocatedElements();

            size_t boundsCheckElements = 0;

            for(ptrdiff_t i = 0; i < elementsBeforeData; i++)
            {
                boundsCheckElements++;
                compareInvalid.before(resultBuffer[i], i, elementsBeforeData);
            }

            if(m_validationStride == 1)
            {
                std::vector<size_t> coord(tensor.dimensions());
                size_t outerCount = CoordCount(tensor.sizes().begin() + 1, tensor.sizes().end());

                size_t       prevBaseIndex = 0;
                const size_t innerDimSize  = tensor.sizes()[0];
                const size_t initialStride = tensor.strides()[0];

                for(size_t i = 0; i < outerCount; i++)
                {
                    CoordNumbered(i,
                                  coord.begin() + 1,
                                  coord.end(),
                                  tensor.sizes().begin() + 1,
                                  tensor.sizes().end());
                    size_t baseElemIndex = tensor.index(coord);

                    if(boundsCheck == BoundsCheckMode::NaN && baseElemIndex != 0
                       && baseElemIndex != prevBaseIndex + innerDimSize)
                    {
                        for(auto innerIndex = prevBaseIndex + innerDimSize;
                            innerIndex < baseElemIndex;
                            innerIndex++)
                        {
                            compareInvalid.inside(
                                resultData[innerIndex], innerIndex, baseElemIndex);
                        }
                    }

                    prevBaseIndex = baseElemIndex;

                    for(size_t j = 0; j < innerDimSize; j++)
                    {
                        size_t elemIndex = baseElemIndex + (j * initialStride);

                        Type referenceValue = reference.d[elemIndex];
                        Type resultValue    = resultData[elemIndex];

                        compareValid(
                            referenceValue, resultValue, elemIndex, (i * tensor.sizes()[0]) + j);
                    }
                }
            }
            else
            {
                std::vector<size_t> coord(tensor.dimensions());
                for(size_t elemNumber = 0; elemNumber < tensor.totalLogicalElements();
                    elemNumber += m_validationStride)
                {
                    CoordNumbered(elemNumber,
                                  coord.begin(),
                                  coord.end(),
                                  tensor.sizes().begin(),
                                  tensor.sizes().end());
                    size_t elemIndex = tensor.index(coord);

                    Type referenceValue = reference.d[elemIndex];
                    Type resultValue    = resultData[elemIndex];

                    compareValid(referenceValue, resultValue, elemIndex, elemNumber);
                }
            }

            for(ptrdiff_t i = 0; i < elementsAfterData; i++)
            {
                compareInvalid.after(resultAfterData[i], i, elementsAfterData);
            }

            if(boundsCheckElements > 0)
                std::cout << "Performed bounds check on " << boundsCheckElements << " elements ("
                          << elementsBeforeData << " before data)" << std::endl;

            compareValid.report();
            compareInvalid.report();

            if(compareValid.error() || compareInvalid.error())
            {
                m_errorInSolution = true;
                m_error           = true;

                return true;
            }

            return false;
        }

        void ReferenceValidator::postSolution()
        {
            if(m_enabled && !m_validatedSolution)
                return;

            if(m_elementsToValidate != 0)
            {
                if(m_errorInConvolutionVsContraction)
                {
                    m_errorsReported++;
                    m_reporter->report(ResultKey::Validation, "FAILED_CONV");
                }
                else if(m_errorInSolution)
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
} // namespace Tensile
