/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "ResultComparison.hpp"
#include "ResultReporter.hpp"
#include "TimingInstrumentation.hpp"

#include "Reference.hpp"

#include <Tensile/DataTypes.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <cstddef>
#include <sstream>

namespace TensileLite
{
    namespace Client
    {
        ReferenceValidator::ReferenceValidator(po::variables_map const&            args,
                                               std::shared_ptr<DataInitialization> dataInit)
            : m_dataInit(dataInit)
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
            m_printTensorScaleAlphaVec = args["print-tensor-scale-alpha-vec"].as<bool>();
            m_printTensorAmaxD         = args["print-tensor-amaxd"].as<bool>();

            m_printAny = m_printTensorA || m_printTensorB || m_printTensorC || m_printTensorD
                         || m_printTensorRef || m_printTensorBias || m_printTensorAmaxD;

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

        void ReferenceValidator::preProblem(ContractionProblem* const problem)
        {
            if(m_enabled)
            {
                m_problem = problem;

                // Report problem context for timing correlation
                if(auto gemm = dynamic_cast<ContractionProblemGemm*>(problem))
                {
                    size_t M          = gemm->freeSizeA(0);
                    size_t N          = gemm->freeSizeB(0);
                    size_t K          = gemm->boundSize(0);
                    size_t batchCount = gemm->batchSize(0);
                    reportProblemContext(M, N, K, batchCount,
                                         TensileLite::ToString(gemm->a().dataType()),
                                         TensileLite::ToString(gemm->d().dataType()));
                }
                else if(auto grouped = dynamic_cast<ContractionProblemGroupedGemm*>(problem))
                {
                    size_t totalGemms = grouped->gemms.size();
                    for(size_t i = 0; i < totalGemms; i++)
                    {
                        auto&  g          = grouped->gemms[i];
                        size_t M          = g.freeSizeA(0);
                        size_t N          = g.freeSizeB(0);
                        size_t K          = g.boundSize(0);
                        size_t batchCount = g.batchSize(0);
                        reportGroupedProblemContext(i, totalGemms, M, N, K, batchCount,
                                                    TensileLite::ToString(g.a().dataType()),
                                                    TensileLite::ToString(g.d().dataType()));
                    }
                }

                {
                    ScopedTimer timer("cpu_data_init");
                    m_referenceInputs = m_dataInit->prepareCPUInputs(problem);
                }

                {
                    ScopedTimer timer("cpu_reference_gemm");
                    SolveCPU(problem, m_referenceInputs.get(), m_elementsToValidate);
                }
            }
        }

        void ReferenceValidator::preSolution(ContractionSolution* const solution)
        {
            m_validatedSolution = false;
            m_errorInSolution   = false;
            m_executedSolution  = false;

            // MXFP4/MXFP8 data can depend on the selected solution (e.g. MI-based preSwizzle when
            // enabled). prepareCPUInputs runs in preProblem before preSolution, so the initial
            // SolveCPU may not match GPU inputs refreshed in DataInitialization::preSolution.
            // Re-run the CPU reference after that refresh (CPU "current" buffers are synced there
            // when the MX path runs).
            if(!m_enabled || m_problem == nullptr || m_referenceInputs == nullptr
               || solution == nullptr)
                return;

            if(auto* gemm = dynamic_cast<ContractionProblemGemm*>(m_problem))
            {
                // Must mirror DataInitialization::preSolution's gate.
                if(!isMXProblem(*gemm))
                    return;
                ScopedTimer timer("cpu_reference_gemm_per_solution");
                SolveCPU(m_problem, m_referenceInputs.get(), m_elementsToValidate);
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

        bool ReferenceValidator::validateSolution(std::shared_ptr<ProblemInputs> inputs)
        {
            if(!m_enabled)
                return false;

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
                        rv &= validate(problems->gemms[j], reference.grouped[j], result.grouped[j]);
                    }
                }
                else if(auto problem = dynamic_cast<ContractionProblemGemm*>(m_problem))
                {
                    auto reference = dynamic_cast<ContractionInputs const&>(*m_referenceInputs);
                    auto result    = dynamic_cast<ContractionInputs const&>(*inputs);
                    rv             = validate(*problem, reference, result);
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

        bool ReferenceValidator::checkResults(TensorDescriptor const& tensor,
                                              void const*             refPtr,
                                              void const*             resPtr,
                                              size_t                  maxElements,
                                              bool                    isgpu,
                                              size_t                  validationStride,
                                              double                  threshold)
        {
            bool rv = false;
            switch(tensor.dataType())
            {
            case rocisa::DataType::Float:
            {
                rv = checkResultsTyped(tensor,
                                       (float const*)refPtr,
                                       (float const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::Double:
            {
                rv = checkResultsTyped(tensor,
                                       (double const*)refPtr,
                                       (double const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::ComplexFloat:
            {
                rv = checkResultsTyped(tensor,
                                       (std::complex<float> const*)refPtr,
                                       (std::complex<float> const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::ComplexDouble:
            {
                rv = checkResultsTyped(tensor,
                                       (std::complex<double> const*)refPtr,
                                       (std::complex<double> const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::Half:
            {
                rv = checkResultsTyped(tensor,
                                       (Half const*)refPtr,
                                       (Half const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::Float8:
            {
                rv = checkResultsTyped(tensor,
                                       (Float8 const*)refPtr,
                                       (Float8 const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::BFloat8:
            {
                rv = checkResultsTyped(tensor,
                                       (BFloat8 const*)refPtr,
                                       (BFloat8 const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::Float8_fnuz:
            {
                rv = checkResultsTyped(tensor,
                                       (Float8_fnuz const*)refPtr,
                                       (Float8_fnuz const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::BFloat8_fnuz:
            {
                rv = checkResultsTyped(tensor,
                                       (BFloat8_fnuz const*)refPtr,
                                       (BFloat8_fnuz const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::Int8x4:
            {
                throw std::runtime_error("Unsupported validator data type Int8x4 for output.");
            }
            break;
            case rocisa::DataType::Int32:
            {
                rv = checkResultsTyped(tensor,
                                       (int32_t const*)refPtr,
                                       (int32_t const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::BFloat16:
            {
                rv = checkResultsTyped(tensor,
                                       (BFloat16 const*)refPtr,
                                       (BFloat16 const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            case rocisa::DataType::Int8:
            {
                rv = checkResultsTyped(tensor,
                                       (int8_t const*)refPtr,
                                       (int8_t const*)resPtr,
                                       maxElements,
                                       isgpu,
                                       validationStride,
                                       threshold);
            }
            break;
            default:
                throw std::runtime_error("Unsupported validator data type");
            }
            if(rv)
            {
                std::cout << "Check failed in output tensor: " << tensor << std::endl;
            }
            return rv;
        }

        bool ReferenceValidator::shouldSkipNullTensor(const std::string& tensorName,
                                                      bool hasNullPointer,
                                                      bool hasZeroElements) const
        {
            // Only output tensors reach this function (filtered by isOutput() check)
            // Output tensors should never have null pointers or zero elements
            return false;
        }

        bool ReferenceValidator::validate(ContractionProblemGemm const& problem,
                                          ContractionInputs const&      reference,
                                          ContractionInputs const&      result)
        {
            if(problem.tensors().empty())
                return false;

            bool rv = true;

            if(m_printAny)
                printTensors(problem, reference, result);

            auto k = problem.transA() ? problem.a().sizes().at(0) : problem.a().sizes().at(1);
            bool isTF32 = (problem.f32XdlMathOp() == rocisa::DataType::XFloat32);
            bool isTF32x1 = (problem.computeInputTypeA() == rocisa::DataType::BFloat16
                && problem.computeInputTypeB() == rocisa::DataType::BFloat16
                && problem.computeType() == rocisa::DataType::Float
                && problem.a().dataType() == rocisa::DataType::Float
                && problem.b().dataType() == rocisa::DataType::Float);
            double threshold = -1.0;
            if (isTF32) {
                threshold = 0.01 * sqrt(double(k));
            } else if (isTF32x1) {
                threshold = 0.3 * sqrt(double(k));
            }

            for(size_t i = 0; i < problem.tensors().size(); i++)
            {
                auto& tensor = problem.tensors()[i];
                if(!tensor.isOutput())
                    continue;

                size_t validationStride = 1;
                if(m_elementsToValidate > 0 && m_elementsToValidate < tensor.totalLogicalElements())
                    validationStride
                        = NextPrime(tensor.totalAllocatedElements() / m_elementsToValidate);

                void const* refPtr = nullptr;
                void const* resPtr = nullptr;
                switch(static_cast<ContractionProblemGemm::TENSOR>(i))
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

                if(Debug::Instance().printTensorInfo())
                    std::cout << "Validating tensor " << tensor.getName() << ", cpu pointer "
                              << refPtr << ", gpu pointer " << resPtr
                              << ", size = " << result.maxElements[i] << std::endl;

                // Check if we should skip this tensor due to null pointers or zero elements
                bool hasNullPointer = (resPtr == nullptr || refPtr == nullptr);
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

                rv &= checkResults(
                    tensor, refPtr, resPtr, result.maxElements[i], result.gpu, validationStride, threshold);
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
                if(problem.a().dataType() == rocisa::DataType::Float4
                   && problem.mxBlockA() > 0)
                {
                    m_reporter->logTensor(LogLevel::Verbose,
                                          "MXSA",
                                          reference.mxsa,
                                          problem.mxsa(),
                                          reference.mxsa);
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
                if(problem.b().dataType() == rocisa::DataType::Float4
                   && problem.mxBlockB() > 0)
                {
                    m_reporter->logTensor(LogLevel::Verbose,
                                          "MXSB",
                                          reference.mxsb,
                                          problem.mxsb(),
                                          reference.mxsb);
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

        template <typename ValidType, typename Comparator>
        void forEachElement(TensorDescriptor const& tensor,
                            ValidType const*        reference,
                            ValidType const*        resultData,
                            size_t                  validationStride,
                            Comparator&             compare)
        {
            if(validationStride == 1)
            {
                std::vector<size_t> coord(tensor.dimensions());
                size_t outerCount
                    = CoordCount(tensor.sizes().begin() + 1, tensor.sizes().end());

                size_t       elemNumberBase = 0;
                const size_t innerDimSize   = tensor.sizes()[0];
                const size_t initialStride  = tensor.strides()[0];

                for(size_t i = 0; i < outerCount; i++)
                {
                    CoordNumbered(i,
                                  coord.begin() + 1,
                                  coord.end(),
                                  tensor.sizes().begin() + 1,
                                  tensor.sizes().end());
                    size_t baseElemIndex = tensor.index(coord);

                    for(size_t j = 0; j < innerDimSize; j++)
                    {
                        size_t elemIndex  = baseElemIndex + (j * initialStride);
                        size_t elemNumber = elemNumberBase + j;

                        compare(reference[elemIndex], resultData[elemIndex],
                                elemIndex, elemNumber);
                    }
                    elemNumberBase += innerDimSize;
                }
            }
            else
            {
                std::vector<size_t> coord(tensor.dimensions());
                for(size_t elemNumber = 0;
                    elemNumber < tensor.totalLogicalElements();
                    elemNumber += validationStride)
                {
                    CoordNumbered(elemNumber,
                                  coord.begin(),
                                  coord.end(),
                                  tensor.sizes().begin(),
                                  tensor.sizes().end());
                    size_t elemIndex = tensor.index(coord);

                    compare(reference[elemIndex], resultData[elemIndex],
                            elemIndex, elemNumber);
                }
            }
        }

        template <typename ValidType>
        bool ReferenceValidator::checkResultsTyped(TensorDescriptor const& tensor,
                                                   ValidType const*        reference,
                                                   ValidType const*        result,
                                                   size_t                  maxElement,
                                                   bool                    isgpu,
                                                   size_t                  validationStride,
                                                   double                  threshold)
        {
            size_t elementsToCopy       = tensor.totalAllocatedElements();
            size_t elementsOffsetToCopy = 0;
            size_t elementsBeforeData   = 0;
            size_t elementsAfterData    = 0;

            BoundsCheckMode boundsCheck = m_dataInit->getCurBoundsCheck();
            // For NaN bounds checking, copy the full padded buffer from GPU for all tensors
            if(boundsCheck == BoundsCheckMode::NaN)
                elementsToCopy = maxElement;
            size_t bytesToCopy = elementsToCopy * sizeof(ValidType);

            // Check if we should skip this tensor due to null pointers or no data
            bool hasNullPointer = (result == nullptr || reference == nullptr);
            bool hasZeroElements = (bytesToCopy == 0 || maxElement == 0);

            if(shouldSkipNullTensor(tensor.getName(), hasNullPointer, hasZeroElements))
            {
                return true;
            }

            // If we reach here with null pointers or no data, it's an error
            if(hasNullPointer || hasZeroElements)
            {
                std::stringstream ss;
                ss << "Unexpected null pointer or no data for tensor " << tensor.getName()
                   << " (result=" << result << ", reference=" << reference
                   << ", bytesToCopy=" << bytesToCopy << ", maxElement=" << maxElement << ")";
                throw std::runtime_error(ss.str());
            }

            allocateResultBuffer(bytesToCopy);

            auto copykind = isgpu ? hipMemcpyDeviceToHost : hipMemcpyHostToHost;

            // For NaN bounds checking, the result pointer points to valid data (middle of buffer)
            // We need to adjust it back to buffer start to copy the NaN padding
            void const* copySource = result;
            if(boundsCheck == BoundsCheckMode::NaN)
            {
                // Match the EXACT allocation logic in copyBadInputBuffers:
                // dPadding = totalElements - totalAllocatedElements()  (in elements)
                // dPadding = multiplyElementSize(dPadding, elementBytes())  (convert to bytes)
                // dPadding = round to multiple of (2 * ceil(max(1, elementBytes)))  (ensure alignment)
                // dstOffset = dst + dPadding / 2  (divide bytes by 2)
                ptrdiff_t paddingElements = maxElement - tensor.totalAllocatedElements();
                size_t paddingBytes = multiplyElementSize(paddingElements, tensor.elementBytes());

                // Ensure paddingBytes/2 is properly aligned for the element type
                // Match the exact rounding logic from copyBadInputBuffers
                float elementBytes = tensor.elementBytes();
                size_t alignmentBytes = 2 * static_cast<size_t>(std::ceil(std::max(1.0f, elementBytes)));
                paddingBytes = (paddingBytes / alignmentBytes) * alignmentBytes;

                size_t bytesBeforeData = paddingBytes / 2;

                copySource = (uint8_t const*)result - bytesBeforeData;

                // Calculate elementsBeforeData for bounds checking display
                // Note: for sub-byte types this may not be exact due to rounding
                elementsBeforeData = bytesBeforeData / std::max(static_cast<size_t>(1),
                                                                 static_cast<size_t>(tensor.elementBytes()));
                elementsAfterData
                    = elementsToCopy - (tensor.totalAllocatedElements() + elementsBeforeData);
            }

            {
                ScopedTimer timer("validate_gpu_readback");
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.get(), copySource, bytesToCopy, copykind));
            }
            // If there was extra data allocated before the tensor to do bounds
            // checking, resultBuffer is the whole allocation, while resultData
            // points directly to the result.
            ValidType const* resultBuffer
                = reinterpret_cast<ValidType const*>(m_cpuResultBuffer.get());
            ValidType const* resultData      = resultBuffer + elementsBeforeData;
            ValidType const* resultAfterData = resultData + tensor.totalAllocatedElements();

            FastPointwiseComparison<ValidType> compareValid(m_printMax > 0, threshold);
            InvalidComparison<ValidType>   compareInvalid(m_printMax, m_printMax > 0);

            size_t boundsCheckElements = 0;

            {
                ScopedTimer timer("validate_element_comparison");

                for(ptrdiff_t i = 0; i < elementsBeforeData; i++)
                {
                    boundsCheckElements++;
                    compareInvalid.before(resultBuffer[i], i, elementsBeforeData);
                }

                forEachElement(tensor, reference, resultData, validationStride, compareValid);

                if(boundsCheck == BoundsCheckMode::NaN && validationStride == 1)
                {
                    std::vector<size_t> coord(tensor.dimensions());
                    size_t outerCount
                        = CoordCount(tensor.sizes().begin() + 1, tensor.sizes().end());
                    size_t       prevBaseIndex = 0;
                    const size_t innerDimSize  = tensor.sizes()[0];

                    for(size_t i = 0; i < outerCount; i++)
                    {
                        CoordNumbered(i,
                                      coord.begin() + 1,
                                      coord.end(),
                                      tensor.sizes().begin() + 1,
                                      tensor.sizes().end());
                        size_t baseElemIndex = tensor.index(coord);

                        if(baseElemIndex != 0
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
                    }
                }

                for(ptrdiff_t i = 0; i < elementsAfterData; i++)
                {
                    compareInvalid.after(resultAfterData[i], i, elementsAfterData);
                }
            }

            if(boundsCheckElements > 0)
                std::cout << "Performed bounds check on " << boundsCheckElements << " elements ("
                          << elementsBeforeData << " before data)" << std::endl;

            if((compareValid.errorCount() > 0 || m_printValids) && m_printMax > 0)
            {
                ScopedTimer timer("validate_mismatch_printing");

                PointwiseComparison<ValidType> comparePrint(
                    m_printValids, m_printMax, false, threshold);

                forEachElement(tensor, reference, resultData, validationStride, comparePrint);
            }

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
            ScopedTimer timer("post_solution_validation");
            if(!m_executedSolution)
                return;

            if(m_enabled && !m_validatedSolution)
                return;

            if(m_elementsToValidate != 0)
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
