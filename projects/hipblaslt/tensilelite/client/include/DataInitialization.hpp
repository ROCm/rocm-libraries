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

#pragma once

#include "ProgramOptions.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include "ClientProblemFactory.hpp"
#include "Rotating.hpp"

#include <cstddef>
#include <random>

#include "RunListener.hpp"

namespace TensileLite
{
    namespace Client
    {
        inline bool isMXFP4Tensor(const TensorDescriptor& tensor, size_t mxBlock)
        {
            return tensor.dataType() == rocisa::DataType::Float4 && mxBlock > 0;
        }

        inline bool isMXFP4Problem(const ContractionProblemGemm& problem)
        {
            return isMXFP4Tensor(problem.a(), problem.mxBlockA())
                || isMXFP4Tensor(problem.b(), problem.mxBlockB());
        }

        // Problem-indept. from 0~7, and 16, and 23~26 (fixed values for every problem)
        // And problem-dept. from 8~15 (values depend on problem)
        // RandomNegPosLimited: integer -128~128. fp -1.0~1.0
        enum class InitMode
        {
            Zero = 0, // 0
            One, // 1
            Two, // 2
            Random, // 3
            NaN, // 4
            Inf, // 5
            BadInput, // 6
            BadOutput, // 7
            SerialIdx, // 8
            SerialDim0, // 9
            SerialDim1, // 10
            Identity, // 11
            TrigSin, // 12
            TrigCos, // 13
            TrigAbsSin, // 14
            TrigAbsCos, // 15
            RandomNarrow, // 16
            NegOne, // 17
            Max, // 18
            DenormMin, // 19
            DenormMax, // 20
            RandomNegPosLimited, // 21
            Free, // 22
            TrigIndSin, // 23
            TrigIndCos, // 24
            TrigIndAbsSin, // 25
            TrigIndAbsCos, // 26
            Count
        };

        static int getThreadLocalRandInt()
        {
            thread_local std::mt19937          generator(std::random_device{}());
            std::uniform_int_distribution<int> distribution;
            return distribution(generator);
        }

        static bool IsProblemDependent(InitMode const& mode)
        {
            return mode == InitMode::SerialIdx || mode == InitMode::SerialDim0
                   || mode == InitMode::SerialDim1 || mode == InitMode::Identity
                   || mode == InitMode::TrigSin || mode == InitMode::TrigCos
                   || mode == InitMode::TrigAbsSin || mode == InitMode::TrigAbsCos;
        }

        std::string ToString(InitMode mode);

        std::ostream& operator<<(std::ostream& stream, InitMode const& mode);
        std::istream& operator>>(std::istream& stream, InitMode& mode);

        const int pageSize = 2 * 1024 * 1024;

        enum class BoundsCheckMode
        {
            Disable = 0,
            NaN,
            GuardPageFront,
            GuardPageBack,
            GuardPageAll,
            MaxMode
        };

        std::ostream& operator<<(std::ostream& stream, BoundsCheckMode const& mode);
        std::istream& operator>>(std::istream& stream, BoundsCheckMode& mode);

        enum class PruneSparseMode
        {
            PruneRandom = 0, // random
            PruneXX00, // XX00  0x4
            PruneX0X0, // X0X0  0x8
            Prune0XX0, // 0XX0  0x9
            PruneX00X, // X00X  0xc
            Prune0X0X, // 0X0X  0xd
            Prune00XX, // 00XX  0xe
            MaxPruneMode
        };

        std::ostream& operator<<(std::ostream& stream, PruneSparseMode const& mode);
        std::istream& operator>>(std::istream& stream, PruneSparseMode& mode);
        class DataInitialization : public RunListener
        {
        public:
            static double GetRepresentativeBetaValue(po::variables_map const& args);

            DataInitialization(po::variables_map const&    args,
                               ClientProblemFactory const& problemFactory);
            ~DataInitialization();

            /**
             * Returns a ContractionInputs object with pointers to CPU memory,
             * suitable for using to calculate reference results.
             */
            std::shared_ptr<ProblemInputs> prepareCPUInputs(ContractionProblem const* problem)
            {
                if(auto groupedProblem
                   = dynamic_cast<ContractionProblemGroupedGemm const*>(problem))
                {
                    return prepareCPUInputs(*groupedProblem);
                }
                else if(auto gemmProblem = dynamic_cast<ContractionProblemGemm const*>(problem))
                {
                    return prepareCPUInputs(*gemmProblem);
                }
                else
                {
                    throw std::runtime_error(
                        "[DataInitialization] Failed to cast to any ContractionProblem");
                }
            }

            std::shared_ptr<ProblemInputs>
                prepareCPUInputs(ContractionProblemGroupedGemm const& problem)
            {
                if(m_cpuInit && m_curBoundsCheck == BoundsCheckMode::Disable
                   && !m_problemDependentData)
                {
                    std::vector<void**> bPtr;
                    if(m_elementsToValidate)
                        resetOutput(m_cpuPtrs,
                                    bPtr,
                                    m_maxElements,
                                    m_groupedOffsets,
                                    problem.gemms[0],
                                    hipMemcpyHostToHost);
                }
                else
                {
                    if(m_problemDependentData)
                        initializeCPUInputs(problem);
                    std::vector<void**> bPtr;
                    copyInputs(m_cpuPtrs,
                               bPtr,
                               m_maxElements,
                               m_groupedOffsets,
                               problem.gemms[0],
                               hipMemcpyHostToHost);
                    m_cpuInit = false;
                }
                initializeConstantInputs(problem.gemms[0]);

                return ConvertToProblemInputs(problem.gemms[0], false);
            }

            std::shared_ptr<ProblemInputs> prepareCPUInputs(ContractionProblemGemm const& problem)
            {
                if(m_cpuInit && m_curBoundsCheck == BoundsCheckMode::Disable
                   && !m_problemDependentData)
                {
                    std::vector<void**> bPtr;
                    if(m_elementsToValidate)
                        resetOutput(m_cpuPtrs,
                                    bPtr,
                                    m_maxElements,
                                    m_groupedOffsets,
                                    problem,
                                    hipMemcpyHostToHost);
                }
                else
                {
                    if(m_problemDependentData)
                        initializeCPUInputs(problem);
                    std::vector<void**> bPtr;
                    copyInputs(m_cpuPtrs,
                               bPtr,
                               m_maxElements,
                               m_groupedOffsets,
                               problem,
                               hipMemcpyHostToHost);
                    m_cpuInit = false;
                }
                initializeConstantInputs(problem);

                return ConvertToProblemInputs(problem, false);
            }

            /**
   * Returns a ProblemInputs object with pointers to GPU memory,
   * suitable for using to run the kernel.
   */
            // A temporarily wrapper
            std::shared_ptr<ProblemInputs> prepareGPUInputs(ContractionProblem const* problem)
            {
                if(auto groupedProblem
                   = dynamic_cast<ContractionProblemGroupedGemm const*>(problem))
                {
                    return prepareGPUInputs(*groupedProblem);
                }
                else if(auto gemmProblem = dynamic_cast<ContractionProblemGemm const*>(problem))
                {
                    return prepareGPUInputs(*gemmProblem);
                }
                else
                    throw std::runtime_error("Failed to cast to any ContractionProblem.");
            }

            std::shared_ptr<ProblemInputs>
                prepareGPUInputs(ContractionProblemGroupedGemm const& problem)
            {
                if(m_numRunsInSolution > 0 && m_curBoundsCheck == BoundsCheckMode::GuardPageFront
                   && m_boundsCheck == BoundsCheckMode::GuardPageAll)
                    m_curBoundsCheck = BoundsCheckMode::GuardPageBack;

                hipMemcpyKind kind;

                if(m_keepPristineCopyOnGPU && !m_problemDependentData)
                {
                    // use gpu pristine
                    kind = hipMemcpyDeviceToDevice;
                }
                else
                {
                    // use cpu pristine
                    kind = hipMemcpyHostToDevice;
                }

                if(m_gpuInit && m_curBoundsCheck == BoundsCheckMode::Disable
                   && !m_problemDependentData)
                {
                    if(m_elementsToValidate)
                    {
                        resetOutput(m_gpuPtrs,
                                    m_gpuBatchPtrs,
                                    m_maxElements,
                                    m_groupedOffsets,
                                    problem.gemms[0],
                                    kind);
                    }
                    return m_cachedGPUInputs;
                }
                else
                {
                    // Update CPU Inputs if prepareGPUInputs is not called.
                    if(m_cpuPtrs.empty() && m_problemDependentData)
                        initializeCPUInputs(problem);
                    if(m_problemDependentData)
                        copyValidToGPUBuffer(problem.gemms[0]);

                    // gpu to gpu
                    copyInputs(m_gpuPtrs,
                               m_gpuBatchPtrs,
                               m_maxElements,
                               m_groupedOffsets,
                               problem.gemms[0],
                               hipMemcpyDeviceToDevice);
                    m_gpuInit = true;
                }
                initializeGPUBatchedInputs(problem.gemms[0]);

                if(m_cpuPtrs.empty())
                    initializeConstantInputs(problem.gemms[0]);

                m_cachedGPUInputs = ConvertToProblemInputs(problem.gemms[0], true);
                return m_cachedGPUInputs;
            }

            std::shared_ptr<ProblemInputs> prepareGPUInputs(ContractionProblemGemm const& problem)
            {
                if(m_numRunsInSolution > 0 && m_curBoundsCheck == BoundsCheckMode::GuardPageFront
                   && m_boundsCheck == BoundsCheckMode::GuardPageAll)
                    m_curBoundsCheck = BoundsCheckMode::GuardPageBack;

                hipMemcpyKind kind;

                bool needSwizzle = problem.swizzleTensorA() || problem.swizzleTensorB();

                if(m_keepPristineCopyOnGPU && !m_problemDependentData)
                {
                    // use gpu pristine
                    kind = hipMemcpyDeviceToDevice;
                }
                else
                {
                    // use cpu pristine
                    kind = hipMemcpyHostToDevice;
                }

                if(m_gpuInit && m_curBoundsCheck == BoundsCheckMode::Disable
                   && !m_problemDependentData && !needSwizzle)
                {
                    if(m_elementsToValidate)
                    {
                        resetOutput(m_gpuPtrs,
                                    m_gpuBatchPtrs,
                                    m_maxElements,
                                    m_groupedOffsets,
                                    problem,
                                    kind);
                    }
                    return m_cachedGPUInputs;
                }
                else
                {
                    // Update CPU Inputs if prepareGPUInputs is not called.
                    if(m_cpuPtrs.empty() && m_problemDependentData)
                        initializeCPUInputs(problem);
                    if(m_problemDependentData)
                        copyValidToGPUBuffer(problem);
                    if(needSwizzle)
                        copySwizzledToGPUBuffer(problem);

                    // gpu to gpu
                    copyInputs(m_gpuPtrs,
                               m_gpuBatchPtrs,
                               m_maxElements,
                               m_groupedOffsets,
                               problem,
                               hipMemcpyDeviceToDevice);
                    if(m_rotatingMode == 1 && m_rotatingBuffer > 0)
                    {
                        auto mem = m_rm->getRotatingMemory();
                        // init mode 1 rotating data
                        for(size_t j = 1; j < mem.size(); j++)
                            for(size_t i = 0; i < m_vdata.size(); i++)
                            {
                                auto& desc = problem.tensors()[i];
                                auto  it   = m_vdata[i].pristine.find(desc.dataType());
                                if(it != m_vdata[i].pristine.end())
                                {
                                    auto& p = it->second;
                                    if(i <= ContractionProblemGemm::TENSOR::METADATA)
                                        HIP_CHECK_EXC(hipMemcpy(mem[j][i].data.get(),
                                                                p.gpuInput.current.get(),
                                                                mem[j][i].size,
                                                                hipMemcpyDeviceToDevice));
                                }
                            }
                    }
                    m_gpuInit = true;
                }
                initializeGPUBatchedInputs(problem);

                if(m_cpuPtrs.empty())
                    initializeConstantInputs(problem);

                m_cachedGPUInputs = ConvertToProblemInputs(problem, true);
                return m_cachedGPUInputs;
            }

            std::vector<std::shared_ptr<ProblemInputs>>
                prepareRotatingGPUOutput(int32_t                        maxRotatingBufferNum,
                                         ContractionProblem const*      problem,
                                         std::shared_ptr<ProblemInputs> inputs,
                                         hipStream_t                    stream);

            template <typename S>
            void initArray(rocisa::DataType dataType, InitMode initMode, void* array, S descriptor)
            {
                switch(dataType)
                {
                case rocisa::DataType::Float:
                    initArray<float>(initMode, static_cast<float*>(array), descriptor);
                    break;
                case rocisa::DataType::Double:
                    initArray<double>(initMode, static_cast<double*>(array), descriptor);
                    break;
                case rocisa::DataType::Half:
                    initArray<Half>(initMode, static_cast<Half*>(array), descriptor);
                    break;
                case rocisa::DataType::Int32:
                    initArray<int32_t>(initMode, static_cast<int32_t*>(array), descriptor);
                    break;
                case rocisa::DataType::BFloat16:
                    initArray<BFloat16>(initMode, static_cast<BFloat16*>(array), descriptor);
                    break;
                case rocisa::DataType::Int8:
                    initArray<int8_t>(initMode, static_cast<int8_t*>(array), descriptor);
                    break;
                case rocisa::DataType::Float8:
                    initArray<Float8>(initMode, static_cast<Float8*>(array), descriptor);
                    break;
                case rocisa::DataType::BFloat8:
                    initArray<BFloat8>(initMode, static_cast<BFloat8*>(array), descriptor);
                    break;
                case rocisa::DataType::Float8_fnuz:
                    initArray<Float8_fnuz>(initMode, static_cast<Float8_fnuz*>(array), descriptor);
                    break;
                case rocisa::DataType::BFloat8_fnuz:
                    initArray<BFloat8_fnuz>(
                        initMode, static_cast<BFloat8_fnuz*>(array), descriptor);
                    break;
#ifndef _WIN32
#ifdef TENSILE_USE_FP6
                case rocisa::DataType::Float6:
                    initArray<Float6x32>(initMode, static_cast<Float6x32*>(array), descriptor);
                    break;
#endif // #ifdef TENSILE_USE_FP6
#ifdef TENSILE_USE_BF6
                case rocisa::DataType::BFloat6:
                    initArray<BFloat6x32>(initMode, static_cast<BFloat6x32*>(array), descriptor);
                    break;
#endif // #ifdef TENSILE_USE_BF6
#ifdef TENSILE_USE_FP4
                case rocisa::DataType::Float4:
                    initArray<Float4x2>(initMode, static_cast<Float4x2*>(array), descriptor);
                    break;
#endif // #ifdef TENSILE_USE_FP4
#endif // !_WIN32
                case rocisa::DataType::E8:
                    initArray<E8>(initMode, static_cast<E8*>(array), descriptor);
                    break;
                case rocisa::DataType::E5M3:
                    initArray<E5M3>(initMode, static_cast<E5M3*>(array), descriptor);
                    break;
                case rocisa::DataType::Int64:
                case rocisa::DataType::XFloat32:
                case rocisa::DataType::ComplexFloat:
                    initArray<std::complex<float>>(initMode, static_cast<std::complex<float>*>(array), descriptor);
                    break;
                case rocisa::DataType::ComplexDouble:
                    initArray<std::complex<double>>(initMode, static_cast<std::complex<double>*>(array), descriptor);
                    break;
                case rocisa::DataType::Int8x4:
                case rocisa::DataType::Count:
                case rocisa::DataType::Float8BFloat8:
                case rocisa::DataType::BFloat8Float8:
                case rocisa::DataType::Float8BFloat8_fnuz:
                case rocisa::DataType::BFloat8Float8_fnuz:
#ifdef _WIN32
                case rocisa::DataType::Float6:
                case rocisa::DataType::BFloat6:
                case rocisa::DataType::Float4:
#endif // _WIN32
                ;
                }
            }

            template <typename T>
            static inline T convertDoubleTo(double value);

            template <typename T>
            static T getValue(InitMode mode, double value = 0.0)
            {
                switch(mode)
                {
                case InitMode::Zero:
                    return getValue<T, InitMode::Zero>();
                case InitMode::One:
                    return getValue<T, InitMode::One>();
                case InitMode::Two:
                    return getValue<T, InitMode::Two>();
                case InitMode::Random:
                    return getValue<T, InitMode::Random>();
                case InitMode::RandomNarrow:
                    return getValue<T, InitMode::RandomNarrow>();
                case InitMode::NaN:
                    return getValue<T, InitMode::NaN>();
                case InitMode::Inf:
                    return getValue<T, InitMode::Inf>();
                case InitMode::BadInput:
                    return getValue<T, InitMode::BadInput>();
                case InitMode::BadOutput:
                    return getValue<T, InitMode::BadOutput>();
                case InitMode::NegOne:
                    return getValue<T, InitMode::NegOne>();
                case InitMode::Max:
                    return getValue<T, InitMode::Max>();
                case InitMode::DenormMin:
                    return getValue<T, InitMode::DenormMin>();
                case InitMode::DenormMax:
                    return getValue<T, InitMode::DenormMax>();
                case InitMode::RandomNegPosLimited:
                    return getValue<T, InitMode::RandomNegPosLimited>();
                case InitMode::Free:
                    return convertDoubleTo<T>(value);
                case InitMode::SerialIdx:
                case InitMode::SerialDim0:
                case InitMode::SerialDim1:
                case InitMode::Identity:
                case InitMode::TrigSin:
                case InitMode::TrigCos:
                case InitMode::TrigAbsSin:
                case InitMode::TrigAbsCos:
                case InitMode::TrigIndSin:
                case InitMode::TrigIndCos:
                case InitMode::TrigIndAbsSin:
                case InitMode::TrigIndAbsCos:
                case InitMode::Count:
                    throw std::runtime_error("Invalid InitMode.");
                }
            }

            template <typename T, InitMode Mode>
            static inline T getValue();

            template <typename T>
            static inline T ConvertTo(size_t i);

            template <typename T>
            static inline T getTrigValue(int idx, bool useCos, bool useAbs);

            template <typename T>
            static bool isBadInput(T value);

            template <typename T>
            static bool isBadOutput(T value);

            // Fills max buffer size
            template <typename T>
            void initArray(InitMode mode, T* array, size_t elements)
            {
                switch(mode)
                {
                case InitMode::Zero:
                    initArray<T, InitMode::Zero>(array, elements);
                    break;
                case InitMode::One:
                    initArray<T, InitMode::One>(array, elements);
                    break;
                case InitMode::Two:
                    initArray<T, InitMode::Two>(array, elements);
                    break;
                case InitMode::Random:
                    initArray<T, InitMode::Random>(array, elements);
                    break;
                case InitMode::RandomNarrow:
                    initArray<T, InitMode::RandomNarrow>(array, elements);
                    break;
                case InitMode::NaN:
                    initArray<T, InitMode::NaN>(array, elements);
                    break;
                case InitMode::Inf:
                    initArray<T, InitMode::Inf>(array, elements);
                    break;
                case InitMode::BadInput:
                    initArray<T, InitMode::BadInput>(array, elements);
                    break;
                case InitMode::BadOutput:
                    initArray<T, InitMode::BadOutput>(array, elements);
                    break;
                case InitMode::NegOne:
                    initArray<T, InitMode::NegOne>(array, elements);
                    break;
                case InitMode::Max:
                    initArray<T, InitMode::Max>(array, elements);
                    break;
                case InitMode::DenormMin:
                    initArray<T, InitMode::DenormMin>(array, elements);
                    break;
                case InitMode::DenormMax:
                    initArray<T, InitMode::DenormMax>(array, elements);
                case InitMode::RandomNegPosLimited:
                    initArray<T, InitMode::RandomNegPosLimited>(array, elements);
                    break;
                case InitMode::SerialIdx:
                    initArrayConvert<T>(array, elements);
                    break;
                case InitMode::Free:
                case InitMode::SerialDim0:
                case InitMode::SerialDim1:
                case InitMode::Identity:
                case InitMode::TrigSin:
                case InitMode::TrigCos:
                case InitMode::TrigAbsSin:
                case InitMode::TrigAbsCos:
                case InitMode::TrigIndSin:
                    initArrayTrig<T, false, false>(array, elements);
                    break;
                case InitMode::TrigIndCos:
                    initArrayTrig<T, true, false>(array, elements);
                    break;
                case InitMode::TrigIndAbsSin:
                    initArrayTrig<T, false, true>(array, elements);
                    break;
                case InitMode::TrigIndAbsCos:
                    initArrayTrig<T, true, true>(array, elements);
                    break;
                case InitMode::Count:
                    throw std::runtime_error("Invalid InitMode.");
                }
            }

            // For problem dependent data initialization
            template <typename T>
            void initArray(InitMode mode, T* array, TensorDescriptor const& tensor)
            {
                switch(mode)
                {
                case InitMode::Zero:
                    initArray<T, InitMode::Zero>(array, tensor);
                    break;
                case InitMode::One:
                    initArray<T, InitMode::One>(array, tensor);
                    break;
                case InitMode::Two:
                    initArray<T, InitMode::Two>(array, tensor);
                    break;
                case InitMode::Random:
                    initArray<T, InitMode::Random>(array, tensor);
                    break;
                case InitMode::RandomNarrow:
                    initArray<T, InitMode::RandomNarrow>(array, tensor);
                    break;
                case InitMode::NaN:
                    initArray<T, InitMode::NaN>(array, tensor);
                    break;
                case InitMode::Inf:
                    initArray<T, InitMode::Inf>(array, tensor);
                    break;
                case InitMode::BadInput:
                    initArray<T, InitMode::BadInput>(array, tensor);
                    break;
                case InitMode::BadOutput:
                    initArray<T, InitMode::BadOutput>(array, tensor);
                    break;
                case InitMode::NegOne:
                    initArray<T, InitMode::NegOne>(array, tensor);
                    break;
                case InitMode::Max:
                    initArray<T, InitMode::Max>(array, tensor);
                    break;
                case InitMode::DenormMin:
                    initArray<T, InitMode::DenormMin>(array, tensor);
                    break;
                case InitMode::DenormMax:
                    initArray<T, InitMode::DenormMax>(array, tensor);
                    break;
                case InitMode::SerialIdx:
                    initArraySerialIdx<T>(array, tensor);
                    break;
                case InitMode::SerialDim0:
                    initArraySerialDim<T>(array, 0, tensor);
                    break;
                case InitMode::SerialDim1:
                    initArraySerialDim<T>(array, 1, tensor);
                    break;
                case InitMode::Identity:
                    initArrayIdentity<T>(array, tensor);
                    break;
                case InitMode::TrigSin:
                    initArrayTrig<T, false, false>(array, tensor);
                    break;
                case InitMode::TrigCos:
                    initArrayTrig<T, true, false>(array, tensor);
                    break;
                case InitMode::TrigAbsSin:
                    initArrayTrig<T, false, true>(array, tensor);
                    break;
                case InitMode::TrigAbsCos:
                    initArrayTrig<T, true, true>(array, tensor);
                    break;
                case InitMode::TrigIndSin:
                    initArrayTrig<T, false, false>(array, tensor);
                    break;
                case InitMode::TrigIndCos:
                    initArrayTrig<T, true, false>(array, tensor);
                    break;
                case InitMode::TrigIndAbsSin:
                    initArrayTrig<T, false, true>(array, tensor);
                    break;
                case InitMode::TrigIndAbsCos:
                    initArrayTrig<T, true, true>(array, tensor);
                    break;
                case InitMode::RandomNegPosLimited:
                    initArray<T, InitMode::RandomNegPosLimited>(array, tensor);
                    break;
                case InitMode::Free:
                case InitMode::Count:
                    throw std::runtime_error("Invalid InitMode.");
                }
            }

            template <typename T, InitMode Mode>
            void initArray(T* array, size_t elements)
            {
                size_t numPacks = elements / TypeInfo<T>::Packing;
#pragma omp parallel for
                for(size_t i = 0; i < numPacks; i++)
                {
                    array[i] = getValue<T, Mode>();
                }
            }

            template <typename T>
            void initArrayConvert(T* array, size_t elements)
            {
                size_t numPacks = elements / TypeInfo<T>::Packing;
#pragma omp parallel for
                for(size_t i = 0; i < numPacks; i++)
                {
                    array[i] = ConvertTo<T>(i);
                }
            }

            template <typename T, InitMode Mode>
            void initArray(T* array, TensorDescriptor const& tensor)
            {
                size_t elements = tensor.totalAllocatedElements();
                initArray<T, Mode>(array, elements);
            }

            template <typename T>
            void initArraySerialIdx(T* array, TensorDescriptor const& tensor)
            {
                auto const& sizes = tensor.sizes();
                auto        count = CoordCount(sizes.begin(), sizes.end());
#pragma omp parallel for
                for(size_t idx = 0; idx < count; idx += TypeInfo<T>::Packing)
                {
                    std::vector<size_t> coord(tensor.dimensions(), 0);
                    CoordNumbered(idx, coord.begin(), coord.end(), sizes.begin(), sizes.end());
                    size_t tensorIndex = tensor.index(coord) / TypeInfo<T>::Packing;
                    array[tensorIndex] = ConvertTo<T>(idx);
                }
            }

            template <typename T>
            void initArraySerialDim(T* array, int dim, TensorDescriptor const& tensor)
            {
                auto const& sizes = tensor.sizes();
                auto        count = CoordCount(sizes.begin(), sizes.end());
#pragma omp parallel for
                for(size_t idx = 0; idx < count; idx += TypeInfo<T>::Packing)
                {
                    std::vector<size_t> coord(tensor.dimensions(), 0);
                    CoordNumbered(idx, coord.begin(), coord.end(), sizes.begin(), sizes.end());
                    size_t tensorIndex = tensor.index(coord) / TypeInfo<T>::Packing;
                    array[tensorIndex] = ConvertTo<T>(coord[dim]);
                }
            }

            template <>
            void initArraySerialDim<Half>(Half* array, int dim, TensorDescriptor const& tensor)
            {
                union
                {
                    uint16_t bits;
                    Half     value;
                } x;

                auto const& sizes = tensor.sizes();
                auto        count = CoordCount(sizes.begin(), sizes.end());
#pragma omp parallel for
                for(size_t idx = 0; idx < count; idx++)
                {
                    std::vector<size_t> coord(tensor.dimensions(), 0);
                    CoordNumbered(idx, coord.begin(), coord.end(), sizes.begin(), sizes.end());
                    x.bits                     = static_cast<uint16_t>(coord[dim]);
                    array[tensor.index(coord)] = x.value;
                }
            }

            template <typename T>
            void initArrayIdentity(T* array, TensorDescriptor const& tensor)
            {
                auto const& sizes = tensor.sizes();
                auto        count = CoordCount(sizes.begin(), sizes.end());
#pragma omp parallel for
                for(size_t idx = 0; idx < count; idx += TypeInfo<T>::Packing)
                {
                    std::vector<size_t> coord(tensor.dimensions(), 0);
                    CoordNumbered(idx, coord.begin(), coord.end(), sizes.begin(), sizes.end());
                    size_t tensorIndex = tensor.index(coord) / TypeInfo<T>::Packing;
                    array[tensorIndex] = ConvertTo<T>(coord[0] == coord[1] ? 1 : 0);
                }
            }

            template <typename T, bool useCos, bool useAbs>
            void initArrayTrig(T* array, TensorDescriptor const& tensor)
            {
                auto const& sizes = tensor.sizes();
                auto        count = CoordCount(sizes.begin(), sizes.end());
#pragma omp parallel for
                for(size_t idx = 0; idx < count; idx += TypeInfo<T>::Packing)
                {
                    std::vector<size_t> coord(tensor.dimensions(), 0);
                    CoordNumbered(idx, coord.begin(), coord.end(), sizes.begin(), sizes.end());
                    size_t tensorIndex = tensor.index(coord) / TypeInfo<T>::Packing;
                    array[tensorIndex] = getTrigValue<T>(idx, useCos, useAbs);
                }
            }

            template <typename T, bool useCos, bool useAbs>
            void initArrayTrig(T* array, size_t elements)
            {
#pragma omp parallel for
                for(size_t i = 0; i < elements; i += TypeInfo<T>::Packing)
                {
                    array[i / TypeInfo<T>::Packing] = getTrigValue<T>(i, useCos, useAbs);
                }
            }

            size_t workspaceSize() const
            {
                return m_workspaceSize;
            }

            BoundsCheckMode getCurBoundsCheck()
            {
                return m_curBoundsCheck;
            }

            virtual bool needMoreBenchmarkRuns() const override
            {
                return false;
            }
            virtual void preBenchmarkRun() override {}
            virtual void postBenchmarkRun() override {}
            virtual void preProblem(ContractionProblem* const problem) override
            {
                m_currentGemmProblem
                    = dynamic_cast<ContractionProblemGemm const*>(problem);
                m_currentSolution = nullptr;
            }
            virtual void postProblem() override {}
            virtual void preSolution(ContractionSolution* const solution) override
            {
                m_currentSolution = solution;
                // Re-init MX FP4 inputs once the solution is known (MI-based preSwizzle when enabled).
                // Gate on m_mxScaleFormat so we only re-init when the user requested an MX scale layout;
                // useScaleAB may be empty for MX kernels that use MXSA/MXSB, so do not gate on it.
                if(m_currentSolution != nullptr
                   && m_mxScaleFormat > 0
                   && m_currentGemmProblem != nullptr
                   && !m_gpuPtrs.empty())
                {
                    bool isMXFP4 = isMXFP4Problem(*m_currentGemmProblem);
                    if(isMXFP4)
                    {
                        initializeMXDataForFP4(*m_currentGemmProblem);
                        copyValidToGPUBuffer(*m_currentGemmProblem);
                        copyInputs(m_gpuPtrs,
                                   m_gpuBatchPtrs,
                                   m_maxElements,
                                   m_groupedOffsets,
                                   *m_currentGemmProblem,
                                   hipMemcpyDeviceToDevice);
                    }
                }
            }
            virtual void postSolution() override {}
            virtual bool needMoreRunsInSolution() const override
            {
                return m_numRunsInSolution < m_numRunsPerSolution;
            };

            virtual size_t numWarmupRuns() override
            {
                return 0;
            };
            virtual void setNumWarmupRuns(size_t count) override {}
            virtual void preWarmup() override {}
            virtual void postWarmup(TimingEvents const& startEvents,
                                    TimingEvents const& stopEvents,
                                    hipStream_t const&  stream) override
            {
            }
            virtual void validateWarmups(std::shared_ptr<ProblemInputs> inputs,
                                         TimingEvents const&            startEvents,
                                         TimingEvents const&            stopEvents) override
            {
                m_numRunsInSolution++;
            }

            virtual size_t numSyncs() override
            {
                return 0;
            }
            virtual void setNumSyncs(size_t count) override {}
            virtual void preSyncs() override {}
            virtual void postSyncs() override {}

            virtual size_t numEnqueuesPerSync() override
            {
                return 0;
            }
            virtual void setNumEnqueuesPerSync(size_t count) override {}
            virtual void preEnqueues(hipStream_t const& stream) override {}
            virtual void postEnqueues(TimingEvents const& startEvents,
                                      TimingEvents const& stopEvents,
                                      hipStream_t const&  stream) override
            {
            }
            virtual void validateEnqueues(std::shared_ptr<ProblemInputs> inputs,
                                          TimingEvents const&            startEvents,
                                          TimingEvents const&            stopEvents) override
            {
            }

            virtual void finalizeReport() override {}

            virtual int error() const override
            {
                return 0;
            }

        protected:
            // Memory input for class DataInitialization
            struct MemoryInput
            {
                std::shared_ptr<void>  current;
                std::shared_ptr<void>  valid;
                std::shared_ptr<void>  bad;
                std::shared_ptr<void*> batch;
            };

            // Pristine unit for each allocated memory
            struct PristineUnit
            {
                size_t                        maxElements;
                std::vector<size_t>           groupedGemmOffsets;
                std::vector<TensorDescriptor> initDescriptor;
                MemoryInput                   cpuInput;
                MemoryInput                   gpuInput;

                MemoryInput& getInputByKind(hipMemcpyKind kind)
                {
                    if(kind == hipMemcpyHostToHost || kind == hipMemcpyDeviceToHost)
                        return cpuInput;
                    return gpuInput;
                }
            };

            // Properties for each tensor (arranged in index)
            struct VectorDataInitProperties
            {
                std::string                              name;
                InitMode                                 init;
                std::map<rocisa::DataType, PristineUnit> pristine;
            };

            // Properties for each constants (arranged in index)
            struct ConstDataInitProperties
            {
                std::string      name;
                InitMode         init;
                rocisa::DataType dataType;
                double           freeValue; // For InitMode::Free
                ConstantVariant  value;
            };

            void allocNewCPUInputs();

            void allocNewGPUInputs();

            void copyValidToGPUBuffer(ContractionProblemGemm const& problem);

            void copySwizzledToGPUBuffer(ContractionProblemGemm const& problem);

            void initializeGPUBatchedInputs(ContractionProblemGemm const& problem);

            void initializeCPUInputs(ContractionProblemGroupedGemm const& problem);
            void initializeCPUInputs(ContractionProblemGemm const& problem);

            void initializeConstantInputs(ContractionProblemGemm const& problem);

            void initializeMXDataForFP4(ContractionProblemGemm const& problem);

            void copyInputs(std::vector<void*>&               ptrs,
                            std::vector<void**>&              batchPtrs,
                            std::vector<size_t>&              maxElements,
                            std::vector<std::vector<size_t>>& offsets,
                            ContractionProblemGemm const&     problem,
                            hipMemcpyKind                     kind);

            void resetOutput(std::vector<void*>&               ptrs,
                             std::vector<void**>&              batchPtrs,
                             std::vector<size_t>&              maxElements,
                             std::vector<std::vector<size_t>>& offsets,
                             ContractionProblemGemm const&     problem,
                             hipMemcpyKind                     kind);

            template <typename T>
            void setContractionInputs(std::vector<T*>&                      ptrs,
                                      std::vector<void**>&                  batchPtrs,
                                      void*                                 ws,
                                      std::vector<ConstDataInitProperties>& cdata,
                                      std::vector<size_t>                   maxElements,
                                      bool                                  isGPU,
                                      ContractionInputs*                    inputs);

            void setContractionGroupedInputs(std::vector<void*>&                     ptrs,
                                             std::vector<void**>&                    batchPtrs,
                                             void*                                   ws,
                                             std::vector<ConstDataInitProperties>&   cdata,
                                             bool                                    isGPU,
                                             ContractionProblemGemm const&           problem,
                                             std::vector<std::vector<size_t>> const& offsets,
                                             ContractionGroupedInputs*               inputs);

            std::shared_ptr<ProblemInputs>
                ConvertToProblemInputs(ContractionProblemGemm const& problem, bool isGPU);

            std::vector<VectorDataInitProperties> m_vdata;
            std::vector<void*>                    m_cpuPtrs;
            std::vector<void*>                    m_gpuPtrs;
            std::vector<std::vector<size_t>>      m_groupedOffsets;
            std::vector<size_t>                   m_maxElements;
            std::vector<void**>                   m_gpuBatchPtrs;
            std::shared_ptr<void>                 m_workspacePristine;
            std::vector<ConstDataInitProperties>  m_cdata;

            bool m_cpuInit = false;
            bool m_gpuInit = false;

            std::shared_ptr<ProblemInputs> m_cachedGPUInputs;

            size_t m_maxBatch;

            size_t m_workspaceSize;

            bool m_stridedBatched;

            int    m_sparse;
            size_t m_aMaxLogicalElements; //for sparse

            bool m_cEqualsD;

            ActivationType m_activationType;

            int m_elementsToValidate = 0;

            /// If true, we will allocate an extra copy of the inputs on the GPU.
            /// This will improve performance as we don't have to copy from the CPU
            /// with each kernel launch, but it will use extra memory.
            bool m_keepPristineCopyOnGPU = true;

            /// If set "::NaN", we will initialize all out-of-bounds inputs to NaN, and
            /// all out-of-bounds outputs to a known value. This allows us to
            /// verify that out-of-bounds values are not used or written to.
            /// If set "::GuardPageFront/::GuardPageBack", we will allocate matrix memory
            /// with page aligned, and put matrix start/end address to memory start/end address.
            /// Out-of-bounds access would trigger memory segmentation faults.
            /// m_boundsCheck keep the setting from args.
            /// m_curBoundsCheck keep the current running boundsCheck mode.
            /// If set "::GuardPageAll", DataInit would need 2 runs per solution.
            /// First run would apply "::GuardPageFront" and second run would apply "::GuardPageBack".
            BoundsCheckMode m_boundsCheck        = BoundsCheckMode::Disable;
            BoundsCheckMode m_curBoundsCheck     = BoundsCheckMode::Disable;
            int             m_numRunsPerSolution = 0;
            int             m_numRunsInSolution  = 0;

            PruneSparseMode m_pruneMode = PruneSparseMode::PruneRandom;
            /// If true, the data is dependent on the problem size (e.g. serial)
            /// and must be reinitialized for each problem. Pristine copy on GPU
            /// cannot be used with problem dependent data.
            bool m_problemDependentData = false;

            int64_t                         m_rotatingBuffer = 0;
            std::shared_ptr<RotatingMemory> m_rm;
            int32_t                         m_rotatingMode = 0;

            ContractionSolution const*  m_currentSolution   = nullptr;
            ContractionProblemGemm const* m_currentGemmProblem = nullptr;

            int m_mxScaleFormat = 0;
        };

        template <>
        inline float DataInitialization::getValue<float, InitMode::Zero>()
        {
            return 0.0f;
        }
        template <>
        inline float DataInitialization::getValue<float, InitMode::One>()
        {
            return 1.0f;
        }
        template <>
        inline float DataInitialization::getValue<float, InitMode::Two>()
        {
            return 2.0f;
        }
        template <>
        inline float DataInitialization::getValue<float, InitMode::NegOne>()
        {
            return -1.0f;
        }
        template <>
        inline float DataInitialization::getValue<float, InitMode::Max>()
        {
            return std::numeric_limits<float>::max();
        }
        template <>
        inline float DataInitialization::getValue<float, InitMode::DenormMin>()
        {
            return std::numeric_limits<float>::denorm_min();
        }
        template <>
        inline float DataInitialization::getValue<float, InitMode::DenormMax>()
        {
            union
            {
                uint32_t bits;
                float    value;
            } x;

            x.bits = 0x007FFFFF;
            return x.value;
        }
        template <>
        inline float DataInitialization::getValue<float, InitMode::NaN>()
        {
            return std::numeric_limits<float>::quiet_NaN();
        }
        template <>
        inline float DataInitialization::getValue<float, InitMode::Inf>()
        {
            return std::numeric_limits<float>::infinity();
        }

        template <>
        inline float DataInitialization::getValue<float, InitMode::Random>()
        {
            return static_cast<float>((getThreadLocalRandInt() % 201) - 100);
        }

        template <>
        inline float DataInitialization::getValue<float, InitMode::BadInput>()
        {
            return getValue<float, InitMode::NaN>();
        }
        template <>
        inline float DataInitialization::getValue<float, InitMode::BadOutput>()
        {
            return getValue<float, InitMode::Inf>();
        }

        template <>
        inline double DataInitialization::getValue<double, InitMode::Zero>()
        {
            return 0.0;
        }
        template <>
        inline double DataInitialization::getValue<double, InitMode::One>()
        {
            return 1.0;
        }
        template <>
        inline double DataInitialization::getValue<double, InitMode::Two>()
        {
            return 2.0;
        }
        template <>
        inline double DataInitialization::getValue<double, InitMode::NegOne>()
        {
            return -1.0;
        }
        template <>
        inline double DataInitialization::getValue<double, InitMode::Max>()
        {
            return std::numeric_limits<double>::max();
        }
        template <>
        inline double DataInitialization::getValue<double, InitMode::DenormMin>()
        {
            return std::numeric_limits<double>::denorm_min();
        }
        template <>
        inline double DataInitialization::getValue<double, InitMode::DenormMax>()
        {
            union
            {
                uint64_t bits;
                double   value;
            } x;

            x.bits = 0x000FFFFFFFFFFFFF;
            return x.value;
        }
        template <>
        inline double DataInitialization::getValue<double, InitMode::NaN>()
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        template <>
        inline double DataInitialization::getValue<double, InitMode::Inf>()
        {
            return std::numeric_limits<double>::infinity();
        }

        template <>
        inline double DataInitialization::getValue<double, InitMode::Random>()
        {
            return static_cast<double>((getThreadLocalRandInt() % 2001) - 1000);
        }

        template <>
        inline double DataInitialization::getValue<double, InitMode::BadInput>()
        {
            return getValue<double, InitMode::NaN>();
        }
        template <>
        inline double DataInitialization::getValue<double, InitMode::BadOutput>()
        {
            return getValue<double, InitMode::Inf>();
        }

        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::Zero>()
        {
            return std::complex<float>(0.0f, 0.0f);
        }
        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::One>()
        {
            return std::complex<float>(1.0f, 0.0f);
        }
        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::Two>()
        {
            return std::complex<float>(2.0f, 0.0f);
        }
        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::NegOne>()
        {
            return std::complex<float>(-1.0f, 0.0f);
        }

        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::Max>()
        {
            return std::complex<float>(std::numeric_limits<float>::max(),
                                       std::numeric_limits<float>::max());
        }

        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::DenormMin>()
        {
            return std::complex<float>(std::numeric_limits<float>::denorm_min(),
                                       std::numeric_limits<float>::denorm_min());
        }

        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::DenormMax>()
        {
            union
            {
                uint32_t bits;
                float    value;
            } x;

            x.bits = 0x007FFFFF;
            return std::complex<float>(x.value, x.value);
        }

        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::NaN>()
        {
            return std::complex<float>(std::numeric_limits<float>::quiet_NaN(),
                                       std::numeric_limits<float>::quiet_NaN());
        }

        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::Inf>()
        {
            return std::complex<float>(std::numeric_limits<float>::infinity(),
                                       std::numeric_limits<float>::infinity());
        }

        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::Random>()
        {
            return std::complex<float>(getValue<float, InitMode::Random>(),
                                       getValue<float, InitMode::Random>());
        }

        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::BadInput>()
        {
            return getValue<std::complex<float>, InitMode::NaN>();
        }
        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::BadOutput>()
        {
            return getValue<std::complex<float>, InitMode::Inf>();
        }

        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::Zero>()
        {
            return std::complex<double>(0.0, 0.0);
        }
        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::One>()
        {
            return std::complex<double>(1.0, 0.0);
        }
        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::Two>()
        {
            return std::complex<double>(2.0, 0.0);
        }
        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::NegOne>()
        {
            return std::complex<double>(-1.0, 0.0);
        }
        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::Max>()
        {
            return std::complex<double>(std::numeric_limits<double>::max(),
                                        std::numeric_limits<double>::max());
        }
        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::DenormMin>()
        {
            return std::complex<double>(std::numeric_limits<double>::denorm_min(),
                                        std::numeric_limits<double>::denorm_min());
        }
        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::DenormMax>()
        {
            union
            {
                uint64_t bits;
                double   value;
            } x;

            x.bits = 0x000FFFFFFFFFFFFF;
            return std::complex<double>(x.value, x.value);
        }
        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::NaN>()
        {
            return std::complex<double>(std::numeric_limits<double>::quiet_NaN(),
                                        std::numeric_limits<double>::quiet_NaN());
        }
        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::Inf>()
        {
            return std::complex<double>(std::numeric_limits<double>::infinity(),
                                        std::numeric_limits<double>::infinity());
        }

        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::Random>()
        {
            return std::complex<double>(getValue<double, InitMode::Random>(),
                                        getValue<double, InitMode::Random>());
        }

        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::BadInput>()
        {
            return getValue<std::complex<double>, InitMode::NaN>();
        }
        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::BadOutput>()
        {
            return getValue<std::complex<double>, InitMode::Inf>();
        }

        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::Zero>()
        {
            return 0;
        }
        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::One>()
        {
            return 1;
        }
        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::Two>()
        {
            return 2;
        }
        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::NegOne>()
        {
            return -1;
        }
        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::Max>()
        {
            return std::numeric_limits<int32_t>::max();
        }
        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::DenormMin>()
        {
            throw std::runtime_error("DenormMin not available for int32_t.");
        }
        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::DenormMax>()
        {
            throw std::runtime_error("DenormMax not available for int32_t.");
        }
        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::NaN>()
        {
            throw std::runtime_error("NaN not available for int32_t.");
        }
        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::Inf>()
        {
            throw std::runtime_error("Inf not available for int32_t.");
        }

        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::Random>()
        {
            return getThreadLocalRandInt() % 7 - 3;
        }

        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::BadInput>()
        {
            return std::numeric_limits<int32_t>::max();
        }
        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::BadOutput>()
        {
            return std::numeric_limits<int32_t>::min();
        }

        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::Zero>()
        {
            return Int8x4{0, 0, 0, 0};
        }
        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::One>()
        {
            return Int8x4{1, 1, 1, 1};
        }
        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::Two>()
        {
            return Int8x4{2, 2, 2, 2};
        }
        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::NegOne>()
        {
            return Int8x4{-1, -1, -1, -1};
        }
        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::Max>()
        {
            return Int8x4{127, 127, 127, 127};
        }
        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::DenormMin>()
        {
            throw std::runtime_error("DenormMin not available for Int8x4.");
        }
        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::DenormMax>()
        {
            throw std::runtime_error("DenormMax not available for Int8x4.");
        }
        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::NaN>()
        {
            throw std::runtime_error("NaN not available for Int8x4.");
        }
        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::Inf>()
        {
            throw std::runtime_error("Inf not available for Int8x4.");
        }

        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::Random>()
        {
            return Int8x4{static_cast<int8_t>((getThreadLocalRandInt() % 7) - 3),
                          static_cast<int8_t>((getThreadLocalRandInt() % 7) - 3),
                          static_cast<int8_t>((getThreadLocalRandInt() % 7) - 3),
                          static_cast<int8_t>((getThreadLocalRandInt() % 7) - 3)};
        }

        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::BadInput>()
        {
            auto val = std::numeric_limits<int8_t>::max();
            return Int8x4{val, val, val, val};
        }

        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::BadOutput>()
        {
            auto val = std::numeric_limits<int8_t>::min();
            return Int8x4{val, val, val, val};
        }

        template <>
        inline Half DataInitialization::getValue<Half, InitMode::Zero>()
        {
            return static_cast<Half>(0);
        }
        template <>
        inline Half DataInitialization::getValue<Half, InitMode::One>()
        {
            return static_cast<Half>(1);
        }
        template <>
        inline Half DataInitialization::getValue<Half, InitMode::Two>()
        {
            return static_cast<Half>(2);
        }
        template <>
        inline Half DataInitialization::getValue<Half, InitMode::NegOne>()
        {
            return static_cast<Half>(-1);
        }
        template <>
        inline Half DataInitialization::getValue<Half, InitMode::Max>()
        {
            return static_cast<Half>(65504.0);
        }
        template <>
        inline Half DataInitialization::getValue<Half, InitMode::DenormMin>()
        {
            union
            {
                uint16_t bits;
                Half     value;
            } x;

            x.bits = 0x0001;
            return x.value;
        }
        template <>
        inline Half DataInitialization::getValue<Half, InitMode::DenormMax>()
        {
            union
            {
                uint16_t bits;
                Half     value;
            } x;

            x.bits = 0x03FF;
            return x.value;
        }
        template <>
        inline Half DataInitialization::getValue<Half, InitMode::NaN>()
        {
            union
            {
                uint16_t bits;
                Half     value;
            } x;

            x.bits = 0xFFFF;
            return x.value;
        }
        template <>
        inline Half DataInitialization::getValue<Half, InitMode::Inf>()
        {
            union
            {
                uint16_t bits;
                Half     value;
            } x;

            x.bits = 0x7C00;
            return x.value;
        }

        template <>
        inline Half DataInitialization::getValue<Half, InitMode::Random>()
        {
            return static_cast<Half>((getThreadLocalRandInt() % 7) - 3);
        }

        template <>
        inline Half DataInitialization::getValue<Half, InitMode::BadInput>()
        {
            return getValue<Half, InitMode::NaN>();
        }
        template <>
        inline Half DataInitialization::getValue<Half, InitMode::BadOutput>()
        {
            return getValue<Half, InitMode::Inf>();
        }

        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::Zero>()
        {
            return static_cast<BFloat16>(0);
        }
        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::One>()
        {
            return static_cast<BFloat16>(1);
        }
        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::Two>()
        {
            return static_cast<BFloat16>(2);
        }
        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::NegOne>()
        {
            return static_cast<BFloat16>(-1);
        }
        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::Max>()
        {
            return static_cast<BFloat16>(DataInitialization::getValue<float, InitMode::Max>());
        }
        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::DenormMin>()
        {
            BFloat16 bf16;
            bf16.data = 0x0001;
            return bf16;
        }
        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::DenormMax>()
        {
            BFloat16 bf16;
            bf16.data = 0x007F;
            return bf16;
        }
        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::NaN>()
        {
            return static_cast<BFloat16>(std::numeric_limits<float>::quiet_NaN());
        }
        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::Inf>()
        {
            return static_cast<BFloat16>(std::numeric_limits<float>::infinity());
        }

        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::Random>()
        {
            return static_cast<BFloat16>((getThreadLocalRandInt() % 7) - 3);
        }

        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::BadInput>()
        {
            return getValue<BFloat16, InitMode::NaN>();
        }
        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::BadOutput>()
        {
            return getValue<BFloat16, InitMode::Inf>();
        }

        // Float8 OCP E4M3
        // NOTE: f8/bf8 has special overloading with unit8_t, so using float val and downcast it to f8/bf8 here
        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::Zero>()
        {
            return static_cast<Float8>(0.0f);
        }
        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::One>()
        {
            return static_cast<Float8>(1.0f);
        }
        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::Two>()
        {
            return static_cast<Float8>(2.0f);
        }
        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::NegOne>()
        {
            return static_cast<Float8>(-1.0f);
        }
        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::Max>()
        {
            // NOTE: sign=0 exp=1111, mantissa=110.. bit pattern : 0x7E
            union
            {
                uint8_t bits;
                Float8  value;
            } x;

            x.bits = 0x7E;
            return x.value;
        }
        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::DenormMin>()
        {
            // NOTE: sign=0 exp=0000, mantissa=001.. bit pattern : 0x01
            //return static_cast<Float8>(0.0009765625f);
            union
            {
                uint8_t bits;
                Float8  value;
            } x;

            x.bits = 0x01;
            return x.value;
        }
        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::DenormMax>()
        {
            // NOTE: sign=0 exp=0000, mantissa=111.. bit pattern : 0x07
            union
            {
                uint8_t bits;
                Float8  value;
            } x;

            x.bits = 0x07;
            return x.value;
        }
        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::NaN>()
        {
            {
                union
                {
                    uint8_t bits;
                    Float8  value;
                } x;

                x.bits = 0x7F; // positive NaN = 0 1111 111
                return x.value;
            }
        }
        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::Inf>()
        {
            {
                union
                {
                    uint8_t bits;
                    Float8  value;
                } x;

                x.bits = 0x7F; // no inf in ocp fp8! returning NaN
                return x.value;
            }
        }

        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::Random>()
        {
            return static_cast<Float8>((float)((getThreadLocalRandInt() % 7) - 3));
        }

        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::BadInput>()
        {
            return getValue<Float8, InitMode::NaN>();
        }
        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::BadOutput>()
        {
            return getValue<Float8, InitMode::Inf>();
        }

        // BFloat8 OCPFP8 E5M2
        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::Zero>()
        {
            return static_cast<BFloat8>(0.0f);
        }
        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::One>()
        {
            return static_cast<BFloat8>(1.0f);
        }
        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::Two>()
        {
            return static_cast<BFloat8>(2.0f);
        }
        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::NegOne>()
        {
            return static_cast<BFloat8>(-1.0f);
        }
        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::NaN>()
        {
            {
                union
                {
                    uint8_t bits;
                    BFloat8 value;
                } x;

                x.bits = 0x7F; // positive NaN
                return x.value;
            }
        }
        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::Inf>()
        {
            {
                union
                {
                    uint8_t bits;
                    BFloat8 value;
                } x;

                x.bits = 0x7C; // 0111 1100
                return x.value;
            }
        }

        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::Random>()
        {
            return static_cast<BFloat8>((float)((getThreadLocalRandInt() % 7) - 3));
        }

        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::BadInput>()
        {
            return getValue<BFloat8, InitMode::NaN>();
        }
        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::BadOutput>()
        {
            return getValue<BFloat8, InitMode::Inf>();
        }
        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::Max>()
        {
            union
            {
                uint8_t bits;
                BFloat8 value;
            } x;

            x.bits = 0x7B; // 0 11110 11
            return x.value;
        }
        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::DenormMin>()
        {
            union
            {
                uint8_t bits;
                BFloat8 value;
            } x;

            x.bits = 0x01;
            return x.value;
        }
        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::DenormMax>()
        {
            union
            {
                uint8_t bits;
                BFloat8 value;
            } x;

            x.bits = 0x03;
            return x.value;
        }

        //  NANOO
        // NOTE: f8/bf8 has special overloading with unit8_t, so using float val and downcast it to f8/bf8 here
        template <>
        inline Float8_fnuz DataInitialization::getValue<Float8_fnuz, InitMode::Zero>()
        {
            return static_cast<Float8_fnuz>(0.0f);
        }
        template <>
        inline Float8_fnuz DataInitialization::getValue<Float8_fnuz, InitMode::One>()
        {
            return static_cast<Float8_fnuz>(1.0f);
        }
        template <>
        inline Float8_fnuz DataInitialization::getValue<Float8_fnuz, InitMode::Two>()
        {
            return static_cast<Float8_fnuz>(2.0f);
        }
        template <>
        inline Float8_fnuz DataInitialization::getValue<Float8_fnuz, InitMode::NegOne>()
        {
            return static_cast<Float8_fnuz>(-1.0f);
        }
        template <>
        inline Float8_fnuz DataInitialization::getValue<Float8_fnuz, InitMode::Max>()
        {
            // NOTE: sign=0 exp=1111, mantissa=111.. bit pattern : 0111 1111 -> 0x7F
            union
            {
                uint8_t     bits;
                Float8_fnuz value;
            } x;

            x.bits = 0x7F;
            return x.value;
        }
        template <>
        inline Float8_fnuz DataInitialization::getValue<Float8_fnuz, InitMode::DenormMin>()
        {
            // NOTE: sign=0 exp=0000, mantissa=001.. bit pattern : 0x01
            //return static_cast<>(0.0009765625f);
            union
            {
                uint8_t     bits;
                Float8_fnuz value;
            } x;

            x.bits = 0x01;
            return x.value;
        }
        template <>
        inline Float8_fnuz DataInitialization::getValue<Float8_fnuz, InitMode::DenormMax>()
        {
            // NOTE: sign=0 exp=0000, mantissa=111.. bit pattern : 0x07
            union
            {
                uint8_t     bits;
                Float8_fnuz value;
            } x;

            x.bits = 0x07;
            return x.value;
        }
        template <>
        inline Float8_fnuz DataInitialization::getValue<Float8_fnuz, InitMode::NaN>()
        {
            {
                union
                {
                    uint8_t     bits;
                    Float8_fnuz value;
                } x;

                x.bits = 0x80; // NaN = 1 0000 000
                return x.value;
            }
        }
        template <>
        inline Float8_fnuz DataInitialization::getValue<Float8_fnuz, InitMode::Inf>()
        {
            {
                union
                {
                    uint8_t     bits;
                    Float8_fnuz value;
                } x;

                x.bits = 0x80; // no inf! returning NaN for now!
                return x.value;
            }
        }

        template <>
        inline Float8_fnuz DataInitialization::getValue<Float8_fnuz, InitMode::Random>()
        {
            return static_cast<Float8_fnuz>((float)((getThreadLocalRandInt() % 7) - 3));
        }

        template <>
        inline Float8_fnuz DataInitialization::getValue<Float8_fnuz, InitMode::BadInput>()
        {
            return getValue<Float8_fnuz, InitMode::NaN>();
        }
        template <>
        inline Float8_fnuz DataInitialization::getValue<Float8_fnuz, InitMode::BadOutput>()
        {
            return getValue<Float8_fnuz, InitMode::Inf>();
        }

        // BFloat8_fnuzNANOO
        template <>
        inline BFloat8_fnuz DataInitialization::getValue<BFloat8_fnuz, InitMode::Zero>()
        {
            return static_cast<BFloat8_fnuz>(0.0f);
        }
        template <>
        inline BFloat8_fnuz DataInitialization::getValue<BFloat8_fnuz, InitMode::One>()
        {
            return static_cast<BFloat8_fnuz>(1.0f);
        }
        template <>
        inline BFloat8_fnuz DataInitialization::getValue<BFloat8_fnuz, InitMode::Two>()
        {
            return static_cast<BFloat8_fnuz>(2.0f);
        }
        template <>
        inline BFloat8_fnuz DataInitialization::getValue<BFloat8_fnuz, InitMode::NegOne>()
        {
            return static_cast<BFloat8_fnuz>(-1.0f);
        }
        template <>
        inline BFloat8_fnuz DataInitialization::getValue<BFloat8_fnuz, InitMode::NaN>()
        {
            {
                union
                {
                    uint8_t      bits;
                    BFloat8_fnuz value;
                } x;

                x.bits = 0x80; // NaN
                return x.value;
            }
        }
        template <>
        inline BFloat8_fnuz DataInitialization::getValue<BFloat8_fnuz, InitMode::Inf>()
        {
            {
                union
                {
                    uint8_t      bits;
                    BFloat8_fnuz value;
                } x;

                x.bits = 0x80; // no inf, returning nan
                return x.value;
            }
        }

        template <>
        inline BFloat8_fnuz DataInitialization::getValue<BFloat8_fnuz, InitMode::Random>()
        {
            return static_cast<BFloat8_fnuz>((float)((getThreadLocalRandInt() % 7) - 3));
        }

        template <>
        inline BFloat8_fnuz DataInitialization::getValue<BFloat8_fnuz, InitMode::BadInput>()
        {
            return getValue<BFloat8_fnuz, InitMode::NaN>();
        }
        template <>
        inline BFloat8_fnuz DataInitialization::getValue<BFloat8_fnuz, InitMode::BadOutput>()
        {
            return getValue<BFloat8_fnuz, InitMode::Inf>();
        }
        template <>
        inline BFloat8_fnuz DataInitialization::getValue<BFloat8_fnuz, InitMode::Max>()
        {
            union
            {
                uint8_t      bits;
                BFloat8_fnuz value;
            } x;

            x.bits = 0x7F; // 0 11111 11 -> 0111 1111 -> 0x7B
            return x.value;
        }
        template <>
        inline BFloat8_fnuz DataInitialization::getValue<BFloat8_fnuz, InitMode::DenormMin>()
        {
            union
            {
                uint8_t      bits;
                BFloat8_fnuz value;
            } x;

            x.bits = 0x01;
            return x.value;
        }
        template <>
        inline BFloat8_fnuz DataInitialization::getValue<BFloat8_fnuz, InitMode::DenormMax>()
        {
            union
            {
                uint8_t      bits;
                BFloat8_fnuz value;
            } x;

            x.bits = 0x03;
            return x.value;
        }
        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::Zero>()
        {
            return 0;
        }
        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::One>()
        {
            return 1;
        }
        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::Two>()
        {
            return 2;
        }
        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::NegOne>()
        {
            return -1;
        }
        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::Max>()
        {
            return 127;
        }
        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::DenormMin>()
        {
            throw std::runtime_error("DenormMin not available for int8_t.");
        }
        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::DenormMax>()
        {
            throw std::runtime_error("DenormMax not available for int8_t.");
        }
        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::NaN>()
        {
            throw std::runtime_error("NaN not available for int8_t.");
        }
        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::Inf>()
        {
            throw std::runtime_error("Inf not available for int8_t.");
        }

        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::Random>()
        {
            return static_cast<int8_t>((getThreadLocalRandInt() % 7) - 3);
        }

        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::BadInput>()
        {
            return std::numeric_limits<int8_t>::max();
        }
        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::BadOutput>()
        {
            return std::numeric_limits<int8_t>::min();
        }
#ifndef _WIN32
#ifdef TENSILE_USE_FP6
        template <>
        inline Float6x32 DataInitialization::getValue<Float6x32, InitMode::Zero>()
        {
            return Float6x32(0.0f);
        }
        template <>
        inline Float6x32 DataInitialization::getValue<Float6x32, InitMode::One>()
        {
            return Float6x32(1.0f);
        }
        template <>
        inline Float6x32 DataInitialization::getValue<Float6x32, InitMode::Two>()
        {
            return Float6x32(2.0f);
        }
        template <>
        inline Float6x32 DataInitialization::getValue<Float6x32, InitMode::NegOne>()
        {
            return Float6x32(-1.0f);
        }
        template <>
        inline Float6x32 DataInitialization::getValue<Float6x32, InitMode::Max>()
        {
            return Float6x32(7.5f);
        }
        template <>
        inline Float6x32 DataInitialization::getValue<Float6x32, InitMode::DenormMin>()
        {
            return Float6x32(0.125f);
        }
        template <>
        inline Float6x32 DataInitialization::getValue<Float6x32, InitMode::DenormMax>()
        {
            return Float6x32(0.875f);
        }
        template <>
        inline Float6x32 DataInitialization::getValue<Float6x32, InitMode::NaN>()
        {
            throw std::runtime_error("NaN not available for float6.");
        }
        template <>
        inline Float6x32 DataInitialization::getValue<Float6x32, InitMode::Inf>()
        {
            throw std::runtime_error("Inf not available for float6.");
        }
        template <>
        inline Float6x32 DataInitialization::getValue<Float6x32, InitMode::Random>()
        {
            float v0  = static_cast<float>((rand() % 7) - 3);
            float v1  = static_cast<float>((rand() % 7) - 3);
            float v2  = static_cast<float>((rand() % 7) - 3);
            float v3  = static_cast<float>((rand() % 7) - 3);
            float v4  = static_cast<float>((rand() % 7) - 3);
            float v5  = static_cast<float>((rand() % 7) - 3);
            float v6  = static_cast<float>((rand() % 7) - 3);
            float v7  = static_cast<float>((rand() % 7) - 3);
            float v8  = static_cast<float>((rand() % 7) - 3);
            float v9  = static_cast<float>((rand() % 7) - 3);
            float v10 = static_cast<float>((rand() % 7) - 3);
            float v11 = static_cast<float>((rand() % 7) - 3);
            float v12 = static_cast<float>((rand() % 7) - 3);
            float v13 = static_cast<float>((rand() % 7) - 3);
            float v14 = static_cast<float>((rand() % 7) - 3);
            float v15 = static_cast<float>((rand() % 7) - 3);
            float v16 = static_cast<float>((rand() % 7) - 3);
            float v17 = static_cast<float>((rand() % 7) - 3);
            float v18 = static_cast<float>((rand() % 7) - 3);
            float v19 = static_cast<float>((rand() % 7) - 3);
            float v20 = static_cast<float>((rand() % 7) - 3);
            float v21 = static_cast<float>((rand() % 7) - 3);
            float v22 = static_cast<float>((rand() % 7) - 3);
            float v23 = static_cast<float>((rand() % 7) - 3);
            float v24 = static_cast<float>((rand() % 7) - 3);
            float v25 = static_cast<float>((rand() % 7) - 3);
            float v26 = static_cast<float>((rand() % 7) - 3);
            float v27 = static_cast<float>((rand() % 7) - 3);
            float v28 = static_cast<float>((rand() % 7) - 3);
            float v29 = static_cast<float>((rand() % 7) - 3);
            float v30 = static_cast<float>((rand() % 7) - 3);
            float v31 = static_cast<float>((rand() % 7) - 3);

            return Float6x32(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9,
                            v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
                            v20, v21, v22, v23, v24, v25, v26, v27, v28, v29,
                            v30, v31);
        }
        template <>
        inline Float6x32 DataInitialization::getValue<Float6x32, InitMode::BadInput>()
        {
            throw std::runtime_error("BadInput not available for float4.");
        }
        template <>
        inline Float6x32 DataInitialization::getValue<Float6x32, InitMode::BadOutput>()
        {
            throw std::runtime_error("BadOutput not available for float4.");
        }
#endif // #ifdef TENSILE_USE_FP6

#ifdef TENSILE_USE_BF6
        template <>
        inline BFloat6x32 DataInitialization::getValue<BFloat6x32, InitMode::Zero>()
        {
            return BFloat6x32(0.0f);
        }
        template <>
        inline BFloat6x32 DataInitialization::getValue<BFloat6x32, InitMode::One>()
        {
            return BFloat6x32(1.0f);
        }
        template <>
        inline BFloat6x32 DataInitialization::getValue<BFloat6x32, InitMode::Two>()
        {
            return BFloat6x32(2.0f);
        }
        template <>
        inline BFloat6x32 DataInitialization::getValue<BFloat6x32, InitMode::NegOne>()
        {
            return BFloat6x32(-1.0f);
        }
        template <>
        inline BFloat6x32 DataInitialization::getValue<BFloat6x32, InitMode::Max>()
        {
            return BFloat6x32(7.5f);
        }
        template <>
        inline BFloat6x32 DataInitialization::getValue<BFloat6x32, InitMode::DenormMin>()
        {
            return BFloat6x32(0.125f);
        }
        template <>
        inline BFloat6x32 DataInitialization::getValue<BFloat6x32, InitMode::DenormMax>()
        {
            return BFloat6x32(0.875f);
        }
        template <>
        inline BFloat6x32 DataInitialization::getValue<BFloat6x32, InitMode::NaN>()
        {
            throw std::runtime_error("NaN not available for float6.");
        }
        template <>
        inline BFloat6x32 DataInitialization::getValue<BFloat6x32, InitMode::Inf>()
        {
            throw std::runtime_error("Inf not available for float6.");
        }
        template <>
        inline BFloat6x32 DataInitialization::getValue<BFloat6x32, InitMode::Random>()
        {
            BFloat6x32 ret;

            float v0  = static_cast<float>((rand() % 7) - 3);
            float v1  = static_cast<float>((rand() % 7) - 3);
            float v2  = static_cast<float>((rand() % 7) - 3);
            float v3  = static_cast<float>((rand() % 7) - 3);
            float v4  = static_cast<float>((rand() % 7) - 3);
            float v5  = static_cast<float>((rand() % 7) - 3);
            float v6  = static_cast<float>((rand() % 7) - 3);
            float v7  = static_cast<float>((rand() % 7) - 3);
            float v8  = static_cast<float>((rand() % 7) - 3);
            float v9  = static_cast<float>((rand() % 7) - 3);
            float v10 = static_cast<float>((rand() % 7) - 3);
            float v11 = static_cast<float>((rand() % 7) - 3);
            float v12 = static_cast<float>((rand() % 7) - 3);
            float v13 = static_cast<float>((rand() % 7) - 3);
            float v14 = static_cast<float>((rand() % 7) - 3);
            float v15 = static_cast<float>((rand() % 7) - 3);
            float v16 = static_cast<float>((rand() % 7) - 3);
            float v17 = static_cast<float>((rand() % 7) - 3);
            float v18 = static_cast<float>((rand() % 7) - 3);
            float v19 = static_cast<float>((rand() % 7) - 3);
            float v20 = static_cast<float>((rand() % 7) - 3);
            float v21 = static_cast<float>((rand() % 7) - 3);
            float v22 = static_cast<float>((rand() % 7) - 3);
            float v23 = static_cast<float>((rand() % 7) - 3);
            float v24 = static_cast<float>((rand() % 7) - 3);
            float v25 = static_cast<float>((rand() % 7) - 3);
            float v26 = static_cast<float>((rand() % 7) - 3);
            float v27 = static_cast<float>((rand() % 7) - 3);
            float v28 = static_cast<float>((rand() % 7) - 3);
            float v29 = static_cast<float>((rand() % 7) - 3);
            float v30 = static_cast<float>((rand() % 7) - 3);
            float v31 = static_cast<float>((rand() % 7) - 3);

            return BFloat6x32(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9,
                            v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
                            v20, v21, v22, v23, v24, v25, v26, v27, v28, v29,
                            v30, v31);

        }
        template <>
        inline BFloat6x32 DataInitialization::getValue<BFloat6x32, InitMode::BadInput>()
        {
            throw std::runtime_error("BadInput not available for float4.");
        }
        template <>
        inline BFloat6x32 DataInitialization::getValue<BFloat6x32, InitMode::BadOutput>()
        {
            throw std::runtime_error("BadOutput not available for float4.");
        }
#endif // #ifdef TENSILE_USE_BF6


#ifdef TENSILE_USE_FP4
        template <>
        inline Float4x2 DataInitialization::getValue<Float4x2, InitMode::Zero>()
        {
            union
            {
                uint8_t bits;
                Float4x2 value;
            } x;

            x.bits = 0x0;
            return x.value;
        }
        template <>
        inline Float4x2 DataInitialization::getValue<Float4x2, InitMode::One>()
        {
            union
            {
                uint8_t bits;
                Float4x2 value;
            } x;

            x.bits = 0x22;
            return x.value;
        }
        template <>
        inline Float4x2 DataInitialization::getValue<Float4x2, InitMode::Two>()
        {
            union
            {
                uint8_t bits;
                Float4x2 value;
            } x;

            x.bits = 0x44;
            return x.value;
        }
        template <>
        inline Float4x2 DataInitialization::getValue<Float4x2, InitMode::NegOne>()
        {
            union
            {
                uint8_t bits;
                Float4x2 value;
            } x;

            x.bits = 0xaa;
            return x.value;
        }
        template <>
        inline Float4x2 DataInitialization::getValue<Float4x2, InitMode::Max>()
        {
            union
            {
                uint8_t bits;
                Float4x2 value;
            } x;

            x.bits = 0x77;
            return x.value;
        }
        template <>
        inline Float4x2 DataInitialization::getValue<Float4x2, InitMode::DenormMin>()
        {
            union
            {
                uint8_t bits;
                Float4x2 value;
            } x;

            x.bits = 0x11;
            return x.value;
        }
        template <>
        inline Float4x2 DataInitialization::getValue<Float4x2, InitMode::DenormMax>()
        {
            union
            {
                uint8_t bits;
                Float4x2 value;
            } x;

            x.bits = 0x11;
            return x.value;
        }
        template <>
        inline Float4x2 DataInitialization::getValue<Float4x2, InitMode::NaN>()
        {
            throw std::runtime_error("NaN not available for float4.");
        }
        template <>
        inline Float4x2 DataInitialization::getValue<Float4x2, InitMode::Inf>()
        {
            throw std::runtime_error("Inf not available for float4.");
        }
        template <>
        inline Float4x2 DataInitialization::getValue<Float4x2, InitMode::Random>()
        {
            union
            {
                uint8_t bits;
                Float4x2 value;
            } x;

            uint8_t val0 = static_cast<uint8_t>(rand() % 15);
            uint8_t val1 = static_cast<uint8_t>(rand() % 15);
            x.bits = (val1 << 4) | val0;
            return x.value;
        }
        template <>
        inline Float4x2 DataInitialization::getValue<Float4x2, InitMode::BadInput>()
        {
            throw std::runtime_error("BadInput not available for float4.");
        }
        template <>
        inline Float4x2 DataInitialization::getValue<Float4x2, InitMode::BadOutput>()
        {
            throw std::runtime_error("BadOutput not available for float4.");
        }
#endif // #ifdef TENSILE_USE_FP4
#endif // !_WIN32

        template <>
        inline E8 DataInitialization::getValue<E8, InitMode::Zero>()
        {
            throw std::runtime_error("Zero not available for E8.");
        }
        template <>
        inline E8 DataInitialization::getValue<E8, InitMode::One>()
        {
            return E8(1.0f);
        }
        template <>
        inline E8 DataInitialization::getValue<E8, InitMode::Two>()
        {
            return E8(2.0f);
        }
        template <>
        inline E8 DataInitialization::getValue<E8, InitMode::NegOne>()
        {
            throw std::runtime_error("-1 not available for E8.");
        }
        template <>
        inline E8 DataInitialization::getValue<E8, InitMode::Max>()
        {
            return E8(0xfe);
        }
        template <>
        inline E8 DataInitialization::getValue<E8, InitMode::DenormMin>()
        {
            throw std::runtime_error("DenormMin not available for E8.");
        }
        template <>
        inline E8 DataInitialization::getValue<E8, InitMode::DenormMax>()
        {
            throw std::runtime_error("DenormMax not available for E8.");
        }
        template <>
        inline E8 DataInitialization::getValue<E8, InitMode::NaN>()
        {
            return E8(0xff);
        }
        template <>
        inline E8 DataInitialization::getValue<E8, InitMode::Inf>()
        {
            throw std::runtime_error("Inf not available for E8.");
        }
        template <>
        inline E8 DataInitialization::getValue<E8, InitMode::Random>()
        {
            return E8(static_cast<uint8_t>((rand() % 7) - 3 + 127));
        }
        template <>
        inline E8 DataInitialization::getValue<E8, InitMode::BadInput>()
        {
            return E8(0xff);
        }
        template <>
        inline E8 DataInitialization::getValue<E8, InitMode::BadOutput>()
        {
            return E8(0xff);
        }

        template <>
        inline E5M3 DataInitialization::getValue<E5M3, InitMode::Zero>()
        {
            return 0;
        }
        template <>
        inline E5M3 DataInitialization::getValue<E5M3, InitMode::One>()
        {
            return E5M3(1.0f);
        }
        template <>
        inline E5M3 DataInitialization::getValue<E5M3, InitMode::Two>()
        {
            return E5M3(2.0f);
        }
        template <>
        inline E5M3 DataInitialization::getValue<E5M3, InitMode::NegOne>()
        {
            throw std::runtime_error("-1 not available for E5M3.");
        }
        template <>
        inline E5M3 DataInitialization::getValue<E5M3, InitMode::Max>()
        {
            uint8_t x = 0xfe;
            return reinterpret_cast<E5M3&>(x);
        }
        template <>
        inline E5M3 DataInitialization::getValue<E5M3, InitMode::DenormMin>()
        {
            uint8_t x = 0x01;
            return reinterpret_cast<E5M3&>(x);
        }
        template <>
        inline E5M3 DataInitialization::getValue<E5M3, InitMode::DenormMax>()
        {
            uint8_t x = 0x07;
            return reinterpret_cast<E5M3&>(x);
        }
        template <>
        inline E5M3 DataInitialization::getValue<E5M3, InitMode::NaN>()
        {
            uint8_t x = 0xff;
            return reinterpret_cast<E5M3&>(x);
        }
        template <>
        inline E5M3 DataInitialization::getValue<E5M3, InitMode::Inf>()
        {
            throw std::runtime_error("Inf not available for E5M3.");
        }
        template <>
        inline E5M3 DataInitialization::getValue<E5M3, InitMode::Random>()
        {
            return E5M3((rand() % 7) - 3);
        }
        template <>
        inline E5M3 DataInitialization::getValue<E5M3, InitMode::BadInput>()
        {
            uint8_t x = 0xff;
            return reinterpret_cast<E5M3&>(x);
        }
        template <>
        inline E5M3 DataInitialization::getValue<E5M3, InitMode::BadOutput>()
        {
            uint8_t x = 0xff;
            return reinterpret_cast<E5M3&>(x);
        }

        template <>
        inline bool DataInitialization::isBadInput<float>(float value)
        {
            return std::isnan(value);
        }

        template <>
        inline bool DataInitialization::isBadInput<double>(double value)
        {
            return std::isnan(value);
        }

        template <>
        inline bool DataInitialization::isBadInput<std::complex<float>>(std::complex<float> value)
        {
            return std::isnan(value.real()) && std::isnan(value.imag());
        }

        template <>
        inline bool DataInitialization::isBadInput<std::complex<double>>(std::complex<double> value)
        {
            return std::isnan(value.real()) && std::isnan(value.imag());
        }

        template <>
        inline bool DataInitialization::isBadInput<int32_t>(int32_t value)
        {
            return value == DataInitialization::getValue<int32_t, InitMode::BadInput>();
        }

        template <>
        inline bool DataInitialization::isBadInput<Int8x4>(Int8x4 value)
        {
            return value == DataInitialization::getValue<Int8x4, InitMode::BadInput>();
        }

        template <>
        inline bool DataInitialization::isBadInput<Half>(Half value)
        {
            return std::isnan(static_cast<float>(value));
        }

        template <>
        inline bool DataInitialization::isBadInput<BFloat16>(BFloat16 value)
        {
            return std::isnan(value);
        }

        template <>
        inline bool DataInitialization::isBadInput<Float8>(Float8 value)
        {
            return std::isnan(value);
        }

        template <>
        inline bool DataInitialization::isBadInput<BFloat8>(BFloat8 value)
        {
            return std::isnan(value);
        }

        template <>
        inline bool DataInitialization::isBadInput<Float8_fnuz>(Float8_fnuz value)
        {
            return std::isnan(value);
        }

        template <>
        inline bool DataInitialization::isBadInput<BFloat8_fnuz>(BFloat8_fnuz value)
        {
            return std::isnan(value);
        }

        template <>
        inline bool DataInitialization::isBadInput<int8_t>(int8_t value)
        {
            return value == DataInitialization::getValue<int8_t, InitMode::BadInput>();
        }

#ifndef _WIN32
#ifdef TENSILE_USE_FP6
        template <>
        inline bool DataInitialization::isBadInput<Float6x32>(Float6x32 value)
        {
            return false;
        }
#endif // #ifdef TENSILE_USE_FP6

#ifdef TENSILE_USE_BF6
        template <>
        inline bool DataInitialization::isBadInput<BFloat6x32>(BFloat6x32 value)
        {
            return false;
        }
#endif // #ifdef TENSILE_USE_BF6

#ifdef TENSILE_USE_FP4
        template <>
        inline bool DataInitialization::isBadInput<Float4x2>(Float4x2 value)
        {
            return false;
        }
#endif // #ifdef TENSILE_USE_FP4
#endif // !_WIN32

        template <>
        inline bool DataInitialization::isBadInput<E8>(E8 value)
        {
            return value.data == 0xff;
        }

        template <>
        inline bool DataInitialization::isBadInput<E5M3>(E5M3 value)
        {
            return value.data == 0xff;
        }

        template <>
        inline bool DataInitialization::isBadOutput<float>(float value)
        {
            return std::isinf(value);
        }

        template <>
        inline bool DataInitialization::isBadOutput<double>(double value)
        {
            return std::isinf(value);
        }

        template <>
        inline bool DataInitialization::isBadOutput<std::complex<float>>(std::complex<float> value)
        {
            return std::isinf(value.real()) && std::isinf(value.imag());
        }

        template <>
        inline bool
            DataInitialization::isBadOutput<std::complex<double>>(std::complex<double> value)
        {
            return std::isinf(value.real()) && std::isinf(value.imag());
        }

        template <>
        inline bool DataInitialization::isBadOutput<int32_t>(int32_t value)
        {
            return value == DataInitialization::getValue<int32_t, InitMode::BadOutput>();
        }

        template <>
        inline bool DataInitialization::isBadOutput<Int8x4>(Int8x4 value)
        {
            return value == DataInitialization::getValue<Int8x4, InitMode::BadOutput>();
        }

        template <>
        inline bool DataInitialization::isBadOutput<Half>(Half value)
        {
            return std::isinf(static_cast<float>(value));
        }

        template <>
        inline bool DataInitialization::isBadOutput<Float8>(Float8 value)
        {
            return std::isinf(static_cast<float>(value));
        }

        template <>
        inline bool DataInitialization::isBadOutput<BFloat8>(BFloat8 value)
        {
            return std::isinf(static_cast<float>(value));
        }

        template <>
        inline bool DataInitialization::isBadOutput<Float8_fnuz>(Float8_fnuz value)
        {
            return std::isinf(static_cast<float>(value));
        }

        template <>
        inline bool DataInitialization::isBadOutput<BFloat8_fnuz>(BFloat8_fnuz value)
        {
            return std::isinf(static_cast<float>(value));
        }

        template <>
        inline bool DataInitialization::isBadOutput<BFloat16>(BFloat16 value)
        {
            return std::isinf(static_cast<float>(value));
        }

        template <>
        inline bool DataInitialization::isBadOutput<int8_t>(int8_t value)
        {
            return value == DataInitialization::getValue<int8_t, InitMode::BadOutput>();
        }

#ifndef _WIN32
#ifdef TENSILE_USE_FP6
        template <>
        inline bool DataInitialization::isBadOutput<Float6x32>(Float6x32 value)
        {
            return false;
        }
#endif // #ifdef TENSILE_USE_FP6

#ifdef TENSILE_USE_BF6
        template <>
        inline bool DataInitialization::isBadOutput<BFloat6x32>(BFloat6x32 value)
        {
            return false;
        }
#endif // #ifdef TENSILE_USE_BF6

#ifdef TENSILE_USE_FP4
        template <>
        inline bool DataInitialization::isBadOutput<Float4x2>(Float4x2 value)
        {
            return false;
        }
#endif // #ifdef TENSILE_USE_FP4
#endif // !_WIN32

        template <>
        inline bool DataInitialization::isBadOutput<E8>(E8 value)
        {
            return value.data == 0xff;
        }

        template <>
        inline bool DataInitialization::isBadOutput<E5M3>(E5M3 value)
        {
            return value.data == 0xff;
        }

        template <>
        inline float DataInitialization::getTrigValue<float>(int idx, bool useCos, bool useAbs)
        {
            float val = useCos ? cos(idx) : sin(idx);
            if(useAbs)
                val = std::fabs(val);
            return val;
        }

        template <>
        inline double DataInitialization::getTrigValue<double>(int idx, bool useCos, bool useAbs)
        {
            double val = useCos ? cos(idx) : sin(idx);
            if(useAbs)
                val = std::fabs(val);
            return val;
        }

        template <>
        inline Half DataInitialization::getTrigValue<Half>(int idx, bool useCos, bool useAbs)
        {
            return static_cast<Half>(getTrigValue<float>(idx, useCos, useAbs));
        }

        template <>
        inline BFloat16
            DataInitialization::getTrigValue<BFloat16>(int idx, bool useCos, bool useAbs)
        {
            return static_cast<BFloat16>(getTrigValue<float>(idx, useCos, useAbs));
        }

        template <>
        inline Float8 DataInitialization::getTrigValue<Float8>(int idx, bool useCos, bool useAbs)
        {
            return static_cast<Float8>(getTrigValue<float>(idx, useCos, useAbs));
        }

        template <>
        inline BFloat8 DataInitialization::getTrigValue<BFloat8>(int idx, bool useCos, bool useAbs)
        {
            return static_cast<BFloat8>(getTrigValue<float>(idx, useCos, useAbs));
        }

        template <>
        inline Float8_fnuz
            DataInitialization::getTrigValue<Float8_fnuz>(int idx, bool useCos, bool useAbs)
        {
            return static_cast<Float8_fnuz>(getTrigValue<float>(idx, useCos, useAbs));
        }

        template <>
        inline BFloat8_fnuz
            DataInitialization::getTrigValue<BFloat8_fnuz>(int idx, bool useCos, bool useAbs)
        {
            return static_cast<BFloat8_fnuz>(getTrigValue<float>(idx, useCos, useAbs));
        }

        template <>
        inline int32_t DataInitialization::getTrigValue<int32_t>(int idx, bool useCos, bool useAbs)
        {
            throw std::runtime_error("Trig not available for int32_t.");
        }

        template <>
        inline Int8x4 DataInitialization::getTrigValue<Int8x4>(int idx, bool useCos, bool useAbs)
        {
            throw std::runtime_error("Trig not available for Int8x4.");
        }

        template <>
        inline int8_t DataInitialization::getTrigValue<int8_t>(int idx, bool useCos, bool useAbs)
        {
            throw std::runtime_error("Trig not available for Int8.");
        }

#ifndef _WIN32
#ifdef TENSILE_USE_FP6
        template <>
        inline Float6x32 DataInitialization::getTrigValue<Float6x32>(int idx, bool useCos, bool useAbs)
        {
            float v0  = getTrigValue<float>(idx, useCos, useAbs);
            float v1  = getTrigValue<float>(idx, useCos, useAbs);
            float v2  = getTrigValue<float>(idx, useCos, useAbs);
            float v3  = getTrigValue<float>(idx, useCos, useAbs);
            float v4  = getTrigValue<float>(idx, useCos, useAbs);
            float v5  = getTrigValue<float>(idx, useCos, useAbs);
            float v6  = getTrigValue<float>(idx, useCos, useAbs);
            float v7  = getTrigValue<float>(idx, useCos, useAbs);
            float v8  = getTrigValue<float>(idx, useCos, useAbs);
            float v9  = getTrigValue<float>(idx, useCos, useAbs);
            float v10 = getTrigValue<float>(idx, useCos, useAbs);
            float v11 = getTrigValue<float>(idx, useCos, useAbs);
            float v12 = getTrigValue<float>(idx, useCos, useAbs);
            float v13 = getTrigValue<float>(idx, useCos, useAbs);
            float v14 = getTrigValue<float>(idx, useCos, useAbs);
            float v15 = getTrigValue<float>(idx, useCos, useAbs);
            float v16 = getTrigValue<float>(idx, useCos, useAbs);
            float v17 = getTrigValue<float>(idx, useCos, useAbs);
            float v18 = getTrigValue<float>(idx, useCos, useAbs);
            float v19 = getTrigValue<float>(idx, useCos, useAbs);
            float v20 = getTrigValue<float>(idx, useCos, useAbs);
            float v21 = getTrigValue<float>(idx, useCos, useAbs);
            float v22 = getTrigValue<float>(idx, useCos, useAbs);
            float v23 = getTrigValue<float>(idx, useCos, useAbs);
            float v24 = getTrigValue<float>(idx, useCos, useAbs);
            float v25 = getTrigValue<float>(idx, useCos, useAbs);
            float v26 = getTrigValue<float>(idx, useCos, useAbs);
            float v27 = getTrigValue<float>(idx, useCos, useAbs);
            float v28 = getTrigValue<float>(idx, useCos, useAbs);
            float v29 = getTrigValue<float>(idx, useCos, useAbs);
            float v30 = getTrigValue<float>(idx, useCos, useAbs);
            float v31 = getTrigValue<float>(idx, useCos, useAbs);

            return Float6x32(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9,
                            v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
                            v20, v21, v22, v23, v24, v25, v26, v27, v28, v29,
                            v30, v31);
        }
#endif // #ifdef TENSILE_USE_FP6

#ifdef TENSILE_USE_BF6
        template <>
        inline BFloat6x32 DataInitialization::getTrigValue<BFloat6x32>(int idx, bool useCos, bool useAbs)
        {
            float v0  = getTrigValue<float>(idx, useCos, useAbs);
            float v1  = getTrigValue<float>(idx, useCos, useAbs);
            float v2  = getTrigValue<float>(idx, useCos, useAbs);
            float v3  = getTrigValue<float>(idx, useCos, useAbs);
            float v4  = getTrigValue<float>(idx, useCos, useAbs);
            float v5  = getTrigValue<float>(idx, useCos, useAbs);
            float v6  = getTrigValue<float>(idx, useCos, useAbs);
            float v7  = getTrigValue<float>(idx, useCos, useAbs);
            float v8  = getTrigValue<float>(idx, useCos, useAbs);
            float v9  = getTrigValue<float>(idx, useCos, useAbs);
            float v10 = getTrigValue<float>(idx, useCos, useAbs);
            float v11 = getTrigValue<float>(idx, useCos, useAbs);
            float v12 = getTrigValue<float>(idx, useCos, useAbs);
            float v13 = getTrigValue<float>(idx, useCos, useAbs);
            float v14 = getTrigValue<float>(idx, useCos, useAbs);
            float v15 = getTrigValue<float>(idx, useCos, useAbs);
            float v16 = getTrigValue<float>(idx, useCos, useAbs);
            float v17 = getTrigValue<float>(idx, useCos, useAbs);
            float v18 = getTrigValue<float>(idx, useCos, useAbs);
            float v19 = getTrigValue<float>(idx, useCos, useAbs);
            float v20 = getTrigValue<float>(idx, useCos, useAbs);
            float v21 = getTrigValue<float>(idx, useCos, useAbs);
            float v22 = getTrigValue<float>(idx, useCos, useAbs);
            float v23 = getTrigValue<float>(idx, useCos, useAbs);
            float v24 = getTrigValue<float>(idx, useCos, useAbs);
            float v25 = getTrigValue<float>(idx, useCos, useAbs);
            float v26 = getTrigValue<float>(idx, useCos, useAbs);
            float v27 = getTrigValue<float>(idx, useCos, useAbs);
            float v28 = getTrigValue<float>(idx, useCos, useAbs);
            float v29 = getTrigValue<float>(idx, useCos, useAbs);
            float v30 = getTrigValue<float>(idx, useCos, useAbs);
            float v31 = getTrigValue<float>(idx, useCos, useAbs);

            return BFloat6x32(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9,
                            v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
                            v20, v21, v22, v23, v24, v25, v26, v27, v28, v29,
                            v30, v31);
        }
#endif // #ifdef TENSILE_USE_BF6

#ifdef TENSILE_USE_FP4
        template <>
        inline Float4x2 DataInitialization::getTrigValue<Float4x2>(int idx, bool useCos, bool useAbs)
        {
            float val0 = getTrigValue<float>(idx, useCos, useAbs);
            float val1 = getTrigValue<float>(idx, useCos, useAbs);
            return Float4x2(val0, val1);
        }
#endif // #ifdef TENSILE_USE_FP4
#endif // !_WIN32

        template <>
        inline E8 DataInitialization::getTrigValue<E8>(int idx, bool useCos, bool useAbs)
        {
            return E8(getTrigValue<float>(idx, useCos, useAbs));
        }

        template <>
        inline std::complex<float>
            DataInitialization::getTrigValue<std::complex<float>>(int idx, bool useCos, bool useAbs)
        {
            return std::complex<float>(getTrigValue<float>(idx, useCos, useAbs),
                                       getTrigValue<float>(idx, useCos, useAbs));
        }

        template <>
        inline std::complex<double> DataInitialization::getTrigValue<std::complex<double>>(
            int idx, bool useCos, bool useAbs)
        {
            return std::complex<double>(getTrigValue<double>(idx, useCos, useAbs),
                                        getTrigValue<double>(idx, useCos, useAbs));
        }

        template <>
        inline E5M3 DataInitialization::getTrigValue<E5M3>(int idx, bool useCos, bool useAbs)
        {
            return E5M3(getTrigValue<float>(idx, useCos, useAbs));
        }

        template <typename>
        struct FP_PARAM;

        template <>
        struct FP_PARAM<double>
        {
            using UINT_T                = uint64_t;
            static constexpr int NUMSIG = 52;
            static constexpr int NUMEXP = 11;
        };

        template <>
        struct FP_PARAM<float>
        {
            using UINT_T                = uint32_t;
            static constexpr int NUMSIG = 23;
            static constexpr int NUMEXP = 8;
        };

        template <>
        struct FP_PARAM<BFloat16>
        {
            using UINT_T                = uint16_t;
            static constexpr int NUMSIG = 7;
            static constexpr int NUMEXP = 8;
        };

        template <>
        struct FP_PARAM<Half>
        {
            using UINT_T                = uint16_t;
            static constexpr int NUMSIG = 10;
            static constexpr int NUMEXP = 5;
        };

        template <>
        struct FP_PARAM<Float8>
        {
            using UINT_T                = uint8_t;
            static constexpr int NUMSIG = 3;
            static constexpr int NUMEXP = 4;
        };

        template <>
        struct FP_PARAM<BFloat8>
        {
            using UINT_T                = uint8_t;
            static constexpr int NUMSIG = 2;
            static constexpr int NUMEXP = 5;
        };

        template <>
        struct FP_PARAM<Float8_fnuz>
        {
            using UINT_T                = uint8_t;
            static constexpr int NUMSIG = 3;
            static constexpr int NUMEXP = 4;
        };

        template <>
        struct FP_PARAM<BFloat8_fnuz>
        {
            using UINT_T                = uint8_t;
            static constexpr int NUMSIG = 2;
            static constexpr int NUMEXP = 5;
        };

        template <>
        struct FP_PARAM<E8>
        {
            using UINT_T                = uint8_t;
            static constexpr int NUMSIG = 0;
            static constexpr int NUMEXP = 8;
        };

        template <>
        struct FP_PARAM<E5M3>
        {
            using UINT_T                = uint8_t;
            static constexpr int NUMSIG = 3;
            static constexpr int NUMEXP = 5;
        };

        template <typename T>
        struct rocm_random_common : FP_PARAM<T>
        {
            using typename FP_PARAM<T>::UINT_T;
            using FP_PARAM<T>::NUMSIG;
            using FP_PARAM<T>::NUMEXP;
            using random_fp_int_dist = std::uniform_int_distribution<UINT_T>;

            static_assert(sizeof(UINT_T) == sizeof(T), "Type sizes do not match");
            static constexpr UINT_T expmask = (((UINT_T)1 << NUMEXP) - 1) << NUMSIG;
            static constexpr UINT_T expbias = ((UINT_T)1 << (NUMEXP - 1)) - 1;
            inline static T         signsig_exp(UINT_T signsig, UINT_T exp)
            {
                union
                {
                    UINT_T u;
                    T      fp;
                };
                u = signsig & ~expmask | ((exp + expbias) << NUMSIG) & expmask;
                return fp;
            }
        };

        template <>
        inline BFloat16
            rocm_random_common<BFloat16>::signsig_exp(FP_PARAM<BFloat16>::UINT_T signsig,
                                                      FP_PARAM<BFloat16>::UINT_T exp)
        {
            FP_PARAM<BFloat16>::UINT_T u;
            u = signsig & ~expmask | ((exp + expbias) << NUMSIG) & expmask;
            return static_cast<BFloat16>(u);
        }

        template <typename T, int LOW_EXP, int HIGH_EXP>
        struct rocm_random : rocm_random_common<T>
        {
            using typename rocm_random_common<T>::random_fp_int_dist;
            __attribute__((flatten)) T operator()()
            {
                static std::mt19937 rng;
                int                 exp = std::uniform_int_distribution<int>{}(rng);
                exp                     = exp % (HIGH_EXP - LOW_EXP + 1) + LOW_EXP;
                return this->signsig_exp(random_fp_int_dist{}(rng), exp);
            }
        };

        template <typename T>
        struct rocm_random_narrow_range;

        template <>
        struct rocm_random_narrow_range<double> : rocm_random<double, -189, 0>
        {
        };

        template <>
        struct rocm_random_narrow_range<float> : rocm_random<float, -100, 0>
        {
        };

        template <>
        struct rocm_random_narrow_range<BFloat16> : rocm_random<BFloat16, -100, 0>
        {
        };

        template <>
        struct rocm_random_narrow_range<Half> : rocm_random<Half, -100, 0>
        {
        };

        // TODO: need to check for fp8 and bf8
        template <>
        struct rocm_random_narrow_range<Float8> : rocm_random<Float8, -100, 0>
        {
        };

        template <>
        struct rocm_random_narrow_range<BFloat8> : rocm_random<BFloat8, -100, 0>
        {
        };

        template <>
        struct rocm_random_narrow_range<Float8_fnuz> : rocm_random<Float8_fnuz, -100, 0>
        {
        };

        template <>
        struct rocm_random_narrow_range<BFloat8_fnuz> : rocm_random<BFloat8_fnuz, -100, 0>
        {
        };

        template <>
        struct rocm_random_narrow_range<E8> : rocm_random<E8, -100, 0>
        {
        };

        template <>
        struct rocm_random_narrow_range<E5M3> : rocm_random<E5M3, -100, 0>
        {
        };

        template <>
        inline float DataInitialization::getValue<float, InitMode::RandomNarrow>()
        {
            return rocm_random_narrow_range<float>{}();
        }

        template <>
        inline double DataInitialization::getValue<double, InitMode::RandomNarrow>()
        {
            return rocm_random_narrow_range<double>{}();
        }

        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::RandomNarrow>()
        {
            return rocm_random_narrow_range<BFloat16>{}();
        }

        template <>
        inline Half DataInitialization::getValue<Half, InitMode::RandomNarrow>()
        {
            return rocm_random_narrow_range<Half>{}();
        }

        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::RandomNarrow>()
        {
#if _WIN32
            //msvc's STL implementation follows [rand.req.genl](1.5), so Float8 as template arg
            //is not allowed
            return Float8(rocm_random_narrow_range<float>{}());
#else
            return rocm_random_narrow_range<Float8>{}();
#endif
        }

        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::RandomNarrow>()
        {
#if _WIN32
            //msvc's STL implementation follows [rand.req.genl](1.5), so BFloat8 as template arg
            //is not allowed
            return BFloat8(rocm_random_narrow_range<float>{}());
#else
            return rocm_random_narrow_range<BFloat8>{}();
#endif
        }

        template <>
        inline Float8_fnuz DataInitialization::getValue<Float8_fnuz, InitMode::RandomNarrow>()
        {
            return rocm_random_narrow_range<Float8_fnuz>{}();
        }

        template <>
        inline BFloat8_fnuz DataInitialization::getValue<BFloat8_fnuz, InitMode::RandomNarrow>()
        {
            return rocm_random_narrow_range<BFloat8_fnuz>{}();
        }

        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::RandomNarrow>()
        {
            return std::complex<float>(rocm_random_narrow_range<float>{}(),
                                       rocm_random_narrow_range<float>{}());
        }

        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::RandomNarrow>()
        {
            return std::complex<double>(rocm_random_narrow_range<double>{}(),
                                        rocm_random_narrow_range<double>{}());
        }

        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::RandomNarrow>()
        {
            return getValue<int32_t, InitMode::Random>();
        }

        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::RandomNarrow>()
        {
            return getValue<Int8x4, InitMode::Random>();
        }

        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::RandomNarrow>()
        {
            return getValue<int8_t, InitMode::Random>();
        }

#ifndef _WIN32
#ifdef TENSILE_USE_FP6
        template <>
        inline Float6x32 DataInitialization::getValue<Float6x32, InitMode::RandomNarrow>()
        {
            float v0  = rocm_random_narrow_range<float>{}();
            float v1  = rocm_random_narrow_range<float>{}();
            float v2  = rocm_random_narrow_range<float>{}();
            float v3  = rocm_random_narrow_range<float>{}();
            float v4  = rocm_random_narrow_range<float>{}();
            float v5  = rocm_random_narrow_range<float>{}();
            float v6  = rocm_random_narrow_range<float>{}();
            float v7  = rocm_random_narrow_range<float>{}();
            float v8  = rocm_random_narrow_range<float>{}();
            float v9  = rocm_random_narrow_range<float>{}();
            float v10 = rocm_random_narrow_range<float>{}();
            float v11 = rocm_random_narrow_range<float>{}();
            float v12 = rocm_random_narrow_range<float>{}();
            float v13 = rocm_random_narrow_range<float>{}();
            float v14 = rocm_random_narrow_range<float>{}();
            float v15 = rocm_random_narrow_range<float>{}();
            float v16 = rocm_random_narrow_range<float>{}();
            float v17 = rocm_random_narrow_range<float>{}();
            float v18 = rocm_random_narrow_range<float>{}();
            float v19 = rocm_random_narrow_range<float>{}();
            float v20 = rocm_random_narrow_range<float>{}();
            float v21 = rocm_random_narrow_range<float>{}();
            float v22 = rocm_random_narrow_range<float>{}();
            float v23 = rocm_random_narrow_range<float>{}();
            float v24 = rocm_random_narrow_range<float>{}();
            float v25 = rocm_random_narrow_range<float>{}();
            float v26 = rocm_random_narrow_range<float>{}();
            float v27 = rocm_random_narrow_range<float>{}();
            float v28 = rocm_random_narrow_range<float>{}();
            float v29 = rocm_random_narrow_range<float>{}();
            float v30 = rocm_random_narrow_range<float>{}();
            float v31 = rocm_random_narrow_range<float>{}();

            return Float6x32(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9,
                            v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
                            v20, v21, v22, v23, v24, v25, v26, v27, v28, v29,
                            v30, v31);
        }
#endif // #ifdef TENSILE_USE_FP6

#ifdef TENSILE_USE_BF6
        template <>
        inline BFloat6x32 DataInitialization::getValue<BFloat6x32, InitMode::RandomNarrow>()
        {
            float v0  = rocm_random_narrow_range<float>{}();
            float v1  = rocm_random_narrow_range<float>{}();
            float v2  = rocm_random_narrow_range<float>{}();
            float v3  = rocm_random_narrow_range<float>{}();
            float v4  = rocm_random_narrow_range<float>{}();
            float v5  = rocm_random_narrow_range<float>{}();
            float v6  = rocm_random_narrow_range<float>{}();
            float v7  = rocm_random_narrow_range<float>{}();
            float v8  = rocm_random_narrow_range<float>{}();
            float v9  = rocm_random_narrow_range<float>{}();
            float v10 = rocm_random_narrow_range<float>{}();
            float v11 = rocm_random_narrow_range<float>{}();
            float v12 = rocm_random_narrow_range<float>{}();
            float v13 = rocm_random_narrow_range<float>{}();
            float v14 = rocm_random_narrow_range<float>{}();
            float v15 = rocm_random_narrow_range<float>{}();
            float v16 = rocm_random_narrow_range<float>{}();
            float v17 = rocm_random_narrow_range<float>{}();
            float v18 = rocm_random_narrow_range<float>{}();
            float v19 = rocm_random_narrow_range<float>{}();
            float v20 = rocm_random_narrow_range<float>{}();
            float v21 = rocm_random_narrow_range<float>{}();
            float v22 = rocm_random_narrow_range<float>{}();
            float v23 = rocm_random_narrow_range<float>{}();
            float v24 = rocm_random_narrow_range<float>{}();
            float v25 = rocm_random_narrow_range<float>{}();
            float v26 = rocm_random_narrow_range<float>{}();
            float v27 = rocm_random_narrow_range<float>{}();
            float v28 = rocm_random_narrow_range<float>{}();
            float v29 = rocm_random_narrow_range<float>{}();
            float v30 = rocm_random_narrow_range<float>{}();
            float v31 = rocm_random_narrow_range<float>{}();

            return BFloat6x32(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9,
                            v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
                            v20, v21, v22, v23, v24, v25, v26, v27, v28, v29,
                            v30, v31);
        }
#endif // #ifdef TENSILE_USE_BF6

#ifdef TENSILE_USE_FP4
        template <>
        inline Float4x2 DataInitialization::getValue<Float4x2, InitMode::RandomNarrow>()
        {
            return getValue<Float4x2, InitMode::Random>();
        }
#endif // #ifdef TENSILE_USE_FP4
#endif // !_WIN32

        template <>
        inline E8 DataInitialization::getValue<E8, InitMode::RandomNarrow>()
        {
            return getValue<E8, InitMode::Random>();
        }

        template <>
        inline E5M3 DataInitialization::getValue<E5M3, InitMode::RandomNarrow>()
        {
            return getValue<E5M3, InitMode::Random>();
        }

        template <typename T>
        inline T getValueWithUpperLowerBoundFP(double upper = 1.0, double lower = -1.0)
        {
            return static_cast<T>(lower
                                  + static_cast<double>(getThreadLocalRandInt())
                                        / static_cast<double>(RAND_MAX / (upper - lower)));
        }

        template <typename T>
        inline T getValueWithUpperLowerBoundInteger(int upper = 128, int lower = -128)
        {
            return static_cast<T>(lower + getThreadLocalRandInt() % (upper - lower + 1));
        }

        template <>
        inline float DataInitialization::getValue<float, InitMode::RandomNegPosLimited>()
        {
            return getValueWithUpperLowerBoundFP<float>();
        }

        template <>
        inline double DataInitialization::getValue<double, InitMode::RandomNegPosLimited>()
        {
            return getValueWithUpperLowerBoundFP<double>();
        }

        template <>
        inline BFloat16 DataInitialization::getValue<BFloat16, InitMode::RandomNegPosLimited>()
        {
            return getValueWithUpperLowerBoundFP<BFloat16>();
        }

        template <>
        inline Float8 DataInitialization::getValue<Float8, InitMode::RandomNegPosLimited>()
        {
            //return static_cast<Float8>(getValueWithUpperLowerBoundFP<float>());
            return getValueWithUpperLowerBoundFP<Float8>();
        }

        template <>
        inline BFloat8 DataInitialization::getValue<BFloat8, InitMode::RandomNegPosLimited>()
        {
            //return static_cast<BFloat8>(getValueWithUpperLowerBoundFP<float>());
            return getValueWithUpperLowerBoundFP<BFloat8>();
        }

        template <>
        inline Float8_fnuz
            DataInitialization::getValue<Float8_fnuz, InitMode::RandomNegPosLimited>()
        {
            //return static_cast<Float8_fnuz>(getValueWithUpperLowerBoundFP<float>());
            return getValueWithUpperLowerBoundFP<Float8_fnuz>();
        }

        template <>
        inline BFloat8_fnuz
            DataInitialization::getValue<BFloat8_fnuz, InitMode::RandomNegPosLimited>()
        {
            //return static_cast<BFloat8_fnuz>(getValueWithUpperLowerBoundFP<float>());
            return getValueWithUpperLowerBoundFP<BFloat8_fnuz>();
        }

        template <>
        inline Half DataInitialization::getValue<Half, InitMode::RandomNegPosLimited>()
        {
            return getValueWithUpperLowerBoundFP<Half>();
        }

        template <>
        inline std::complex<float>
            DataInitialization::getValue<std::complex<float>, InitMode::RandomNegPosLimited>()
        {
            return std::complex<float>(getValueWithUpperLowerBoundFP<float>(),
                                       getValueWithUpperLowerBoundFP<float>());
        }

        template <>
        inline std::complex<double>
            DataInitialization::getValue<std::complex<double>, InitMode::RandomNegPosLimited>()
        {
            return std::complex<double>(getValueWithUpperLowerBoundFP<double>(),
                                        getValueWithUpperLowerBoundFP<double>());
        }

        template <>
        inline int32_t DataInitialization::getValue<int32_t, InitMode::RandomNegPosLimited>()
        {
            return getValueWithUpperLowerBoundInteger<int32_t>();
        }

        template <>
        inline Int8x4 DataInitialization::getValue<Int8x4, InitMode::RandomNegPosLimited>()
        {
            return Int8x4{getValueWithUpperLowerBoundInteger<int8_t>(),
                          getValueWithUpperLowerBoundInteger<int8_t>(),
                          getValueWithUpperLowerBoundInteger<int8_t>(),
                          getValueWithUpperLowerBoundInteger<int8_t>()};
        }

        template <>
        inline int8_t DataInitialization::getValue<int8_t, InitMode::RandomNegPosLimited>()
        {
            return getValueWithUpperLowerBoundInteger<int8_t>();
        }

#ifndef _WIN32
#ifdef TENSILE_USE_FP6
        template <>
        inline Float6x32 DataInitialization::getValue<Float6x32, InitMode::RandomNegPosLimited>()
        {
            float v0  = getValueWithUpperLowerBoundFP<float>();
            float v1  = getValueWithUpperLowerBoundFP<float>();
            float v2  = getValueWithUpperLowerBoundFP<float>();
            float v3  = getValueWithUpperLowerBoundFP<float>();
            float v4  = getValueWithUpperLowerBoundFP<float>();
            float v5  = getValueWithUpperLowerBoundFP<float>();
            float v6  = getValueWithUpperLowerBoundFP<float>();
            float v7  = getValueWithUpperLowerBoundFP<float>();
            float v8  = getValueWithUpperLowerBoundFP<float>();
            float v9  = getValueWithUpperLowerBoundFP<float>();
            float v10 = getValueWithUpperLowerBoundFP<float>();
            float v11 = getValueWithUpperLowerBoundFP<float>();
            float v12 = getValueWithUpperLowerBoundFP<float>();
            float v13 = getValueWithUpperLowerBoundFP<float>();
            float v14 = getValueWithUpperLowerBoundFP<float>();
            float v15 = getValueWithUpperLowerBoundFP<float>();
            float v16 = getValueWithUpperLowerBoundFP<float>();
            float v17 = getValueWithUpperLowerBoundFP<float>();
            float v18 = getValueWithUpperLowerBoundFP<float>();
            float v19 = getValueWithUpperLowerBoundFP<float>();
            float v20 = getValueWithUpperLowerBoundFP<float>();
            float v21 = getValueWithUpperLowerBoundFP<float>();
            float v22 = getValueWithUpperLowerBoundFP<float>();
            float v23 = getValueWithUpperLowerBoundFP<float>();
            float v24 = getValueWithUpperLowerBoundFP<float>();
            float v25 = getValueWithUpperLowerBoundFP<float>();
            float v26 = getValueWithUpperLowerBoundFP<float>();
            float v27 = getValueWithUpperLowerBoundFP<float>();
            float v28 = getValueWithUpperLowerBoundFP<float>();
            float v29 = getValueWithUpperLowerBoundFP<float>();
            float v30 = getValueWithUpperLowerBoundFP<float>();
            float v31 = getValueWithUpperLowerBoundFP<float>();

            return Float6x32(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9,
                            v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
                            v20, v21, v22, v23, v24, v25, v26, v27, v28, v29,
                            v30, v31);
        }
#endif // #ifdef TENSILE_USE_FP6

#ifdef TENSILE_USE_BF6
        template <>
        inline BFloat6x32 DataInitialization::getValue<BFloat6x32, InitMode::RandomNegPosLimited>()
        {
            float v0  = getValueWithUpperLowerBoundFP<float>();
            float v1  = getValueWithUpperLowerBoundFP<float>();
            float v2  = getValueWithUpperLowerBoundFP<float>();
            float v3  = getValueWithUpperLowerBoundFP<float>();
            float v4  = getValueWithUpperLowerBoundFP<float>();
            float v5  = getValueWithUpperLowerBoundFP<float>();
            float v6  = getValueWithUpperLowerBoundFP<float>();
            float v7  = getValueWithUpperLowerBoundFP<float>();
            float v8  = getValueWithUpperLowerBoundFP<float>();
            float v9  = getValueWithUpperLowerBoundFP<float>();
            float v10 = getValueWithUpperLowerBoundFP<float>();
            float v11 = getValueWithUpperLowerBoundFP<float>();
            float v12 = getValueWithUpperLowerBoundFP<float>();
            float v13 = getValueWithUpperLowerBoundFP<float>();
            float v14 = getValueWithUpperLowerBoundFP<float>();
            float v15 = getValueWithUpperLowerBoundFP<float>();
            float v16 = getValueWithUpperLowerBoundFP<float>();
            float v17 = getValueWithUpperLowerBoundFP<float>();
            float v18 = getValueWithUpperLowerBoundFP<float>();
            float v19 = getValueWithUpperLowerBoundFP<float>();
            float v20 = getValueWithUpperLowerBoundFP<float>();
            float v21 = getValueWithUpperLowerBoundFP<float>();
            float v22 = getValueWithUpperLowerBoundFP<float>();
            float v23 = getValueWithUpperLowerBoundFP<float>();
            float v24 = getValueWithUpperLowerBoundFP<float>();
            float v25 = getValueWithUpperLowerBoundFP<float>();
            float v26 = getValueWithUpperLowerBoundFP<float>();
            float v27 = getValueWithUpperLowerBoundFP<float>();
            float v28 = getValueWithUpperLowerBoundFP<float>();
            float v29 = getValueWithUpperLowerBoundFP<float>();
            float v30 = getValueWithUpperLowerBoundFP<float>();
            float v31 = getValueWithUpperLowerBoundFP<float>();

            return BFloat6x32(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9,
                            v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
                            v20, v21, v22, v23, v24, v25, v26, v27, v28, v29,
                            v30, v31);
        }
#endif // #ifdef TENSILE_USE_BF6


#ifdef TENSILE_USE_FP4
        template <>
        inline Float4x2 DataInitialization::getValue<Float4x2, InitMode::RandomNegPosLimited>()
        {
            return Float4x2(getValueWithUpperLowerBoundFP<float>(),
                            getValueWithUpperLowerBoundFP<float>());
        }
#endif // #ifdef TENSILE_USE_FP4
#endif // !_WIN32

        template <>
        inline E8 DataInitialization::getValue<E8, InitMode::RandomNegPosLimited>()
        {
            return getValueWithUpperLowerBoundInteger<E8>();
        }

        template <>
        inline E5M3 DataInitialization::getValue<E5M3, InitMode::RandomNegPosLimited>()
        {
            return getValueWithUpperLowerBoundInteger<E5M3>();
        }

        template <>
        inline float DataInitialization::ConvertTo<float>(size_t i)
        {
            return static_cast<float>(i);
        }

        template <>
        inline double DataInitialization::ConvertTo<double>(size_t i)
        {
            return static_cast<double>(i);
        }

        template <>
        inline BFloat16 DataInitialization::ConvertTo<BFloat16>(size_t i)
        {
            return static_cast<BFloat16>(i);
        }

        template <>
        inline Float8 DataInitialization::ConvertTo<Float8>(size_t i)
        {
            return static_cast<Float8>(i);
        }

        template <>
        inline BFloat8 DataInitialization::ConvertTo<BFloat8>(size_t i)
        {
            return static_cast<BFloat8>(i);
        }

        template <>
        inline Float8_fnuz DataInitialization::ConvertTo<Float8_fnuz>(size_t i)
        {
            return static_cast<Float8_fnuz>(i);
        }

        template <>
        inline BFloat8_fnuz DataInitialization::ConvertTo<BFloat8_fnuz>(size_t i)
        {
            return static_cast<BFloat8_fnuz>(i);
        }

        template <>
        inline Half DataInitialization::ConvertTo<Half>(size_t i)
        {
            return static_cast<Half>(i);
        }

        template <>
        inline std::complex<float> DataInitialization::ConvertTo<std::complex<float>>(size_t i)
        {
            return std::complex<float>(static_cast<float>(i), static_cast<float>(i));
        }

        template <>
        inline std::complex<double> DataInitialization::ConvertTo<std::complex<double>>(size_t i)
        {
            return std::complex<double>(static_cast<double>(i), static_cast<double>(i));
        }

        template <>
        inline int32_t DataInitialization::ConvertTo<int32_t>(size_t i)
        {
            return static_cast<int32_t>(i);
        }

        template <>
        inline Int8x4 DataInitialization::ConvertTo<Int8x4>(size_t i)
        {
            return Int8x4{static_cast<int8_t>(i),
                          static_cast<int8_t>(i),
                          static_cast<int8_t>(i),
                          static_cast<int8_t>(i)};
        }

        template <>
        inline int8_t DataInitialization::ConvertTo<int8_t>(size_t i)
        {
            return static_cast<int8_t>(i);
        }

#ifndef _WIN32
#ifdef TENSILE_USE_FP6
        template <>
        inline Float6x32 DataInitialization::ConvertTo<Float6x32>(size_t i)
        {
            float v0  = float(i);
            float v1  = float(i);
            float v2  = float(i);
            float v3  = float(i);
            float v4  = float(i);
            float v5  = float(i);
            float v6  = float(i);
            float v7  = float(i);
            float v8  = float(i);
            float v9  = float(i);
            float v10 = float(i);
            float v11 = float(i);
            float v12 = float(i);
            float v13 = float(i);
            float v14 = float(i);
            float v15 = float(i);
            float v16 = float(i);
            float v17 = float(i);
            float v18 = float(i);
            float v19 = float(i);
            float v20 = float(i);
            float v21 = float(i);
            float v22 = float(i);
            float v23 = float(i);
            float v24 = float(i);
            float v25 = float(i);
            float v26 = float(i);
            float v27 = float(i);
            float v28 = float(i);
            float v29 = float(i);
            float v30 = float(i);
            float v31 = float(i);

            return Float6x32(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9,
                            v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
                            v20, v21, v22, v23, v24, v25, v26, v27, v28, v29,
                            v30, v31);
        }
#endif // #ifdef TENSILE_USE_FP6

#ifdef TENSILE_USE_BF6
        template <>
        inline BFloat6x32 DataInitialization::ConvertTo<BFloat6x32>(size_t i)
        {
            float v0  = float(i);
            float v1  = float(i);
            float v2  = float(i);
            float v3  = float(i);
            float v4  = float(i);
            float v5  = float(i);
            float v6  = float(i);
            float v7  = float(i);
            float v8  = float(i);
            float v9  = float(i);
            float v10 = float(i);
            float v11 = float(i);
            float v12 = float(i);
            float v13 = float(i);
            float v14 = float(i);
            float v15 = float(i);
            float v16 = float(i);
            float v17 = float(i);
            float v18 = float(i);
            float v19 = float(i);
            float v20 = float(i);
            float v21 = float(i);
            float v22 = float(i);
            float v23 = float(i);
            float v24 = float(i);
            float v25 = float(i);
            float v26 = float(i);
            float v27 = float(i);
            float v28 = float(i);
            float v29 = float(i);
            float v30 = float(i);
            float v31 = float(i);

            return BFloat6x32(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9,
                            v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
                            v20, v21, v22, v23, v24, v25, v26, v27, v28, v29,
                            v30, v31);
        }
#endif // #ifdef TENSILE_USE_BF6

#ifdef TENSILE_USE_FP4
        template <>
        inline Float4x2 DataInitialization::ConvertTo<Float4x2>(size_t i)
        {
            return Float4x2(float(i), float(i));
        }
#endif // #ifdef TENSILE_USE_FP4
#endif // !_WIN32

        template <>
        inline E8 DataInitialization::ConvertTo<E8>(size_t i)
        {
            return static_cast<E8>(static_cast<float>(i));
        }

        template <>
        inline E5M3 DataInitialization::ConvertTo<E5M3>(size_t i)
        {
            return static_cast<E5M3>(static_cast<float>(i));
        }

        template <>
        inline float DataInitialization::convertDoubleTo<float>(double value)
        {
            return static_cast<float>(value);
        }

        template <>
        inline double DataInitialization::convertDoubleTo<double>(double value)
        {
            return value;
        }

        template <>
        inline BFloat16 DataInitialization::convertDoubleTo<BFloat16>(double value)
        {
            return static_cast<BFloat16>(value);
        }

        template <>
        inline Half DataInitialization::convertDoubleTo<Half>(double value)
        {
            return static_cast<Half>(value);
        }

        template <>
        inline std::complex<float>
            DataInitialization::convertDoubleTo<std::complex<float>>(double value)
        {
            throw std::runtime_error("convertDoubleTo not available for std::complex<float>.");
        }

        template <>
        inline std::complex<double>
            DataInitialization::convertDoubleTo<std::complex<double>>(double value)
        {
            throw std::runtime_error("convertDoubleTo not available for std::complex<double>.");
        }

        template <>
        inline int32_t DataInitialization::convertDoubleTo<int32_t>(double value)
        {
            return static_cast<int32_t>(value);
        }

        template <>
        inline Int8x4 DataInitialization::convertDoubleTo<Int8x4>(double value)
        {
            throw std::runtime_error("convertDoubleTo not available for Int8x4.");
        }

        template <>
        inline int8_t DataInitialization::convertDoubleTo<int8_t>(double value)
        {
            return static_cast<int8_t>(value);
        }

        template <>
        inline Float8 DataInitialization::convertDoubleTo<Float8>(double value)
        {
            return static_cast<Float8>(value);
        }

        template <>
        inline BFloat8 DataInitialization::convertDoubleTo<BFloat8>(double value)
        {
            return static_cast<BFloat8>(value);
        }

        template <>
        inline Float8_fnuz DataInitialization::convertDoubleTo<Float8_fnuz>(double value)
        {
            return static_cast<Float8_fnuz>(value);
        }

        template <>
        inline BFloat8_fnuz DataInitialization::convertDoubleTo<BFloat8_fnuz>(double value)
        {
            return static_cast<BFloat8_fnuz>(value);
        }
#ifndef _WIN32
#ifdef TENSILE_USE_FP6
        template <>
        inline Float6x32 DataInitialization::convertDoubleTo<Float6x32>(double value)
        {
            float v0  = float(value);
            float v1  = float(value);
            float v2  = float(value);
            float v3  = float(value);
            float v4  = float(value);
            float v5  = float(value);
            float v6  = float(value);
            float v7  = float(value);
            float v8  = float(value);
            float v9  = float(value);
            float v10 = float(value);
            float v11 = float(value);
            float v12 = float(value);
            float v13 = float(value);
            float v14 = float(value);
            float v15 = float(value);
            float v16 = float(value);
            float v17 = float(value);
            float v18 = float(value);
            float v19 = float(value);
            float v20 = float(value);
            float v21 = float(value);
            float v22 = float(value);
            float v23 = float(value);
            float v24 = float(value);
            float v25 = float(value);
            float v26 = float(value);
            float v27 = float(value);
            float v28 = float(value);
            float v29 = float(value);
            float v30 = float(value);
            float v31 = float(value);

            return Float6x32(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9,
                            v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
                            v20, v21, v22, v23, v24, v25, v26, v27, v28, v29,
                            v30, v31);
        }
#endif // #ifdef TENSILE_USE_FP6

#ifdef TENSILE_USE_BF6
        template <>
        inline BFloat6x32 DataInitialization::convertDoubleTo<BFloat6x32>(double value)
        {
            float v0  = float(value);
            float v1  = float(value);
            float v2  = float(value);
            float v3  = float(value);
            float v4  = float(value);
            float v5  = float(value);
            float v6  = float(value);
            float v7  = float(value);
            float v8  = float(value);
            float v9  = float(value);
            float v10 = float(value);
            float v11 = float(value);
            float v12 = float(value);
            float v13 = float(value);
            float v14 = float(value);
            float v15 = float(value);
            float v16 = float(value);
            float v17 = float(value);
            float v18 = float(value);
            float v19 = float(value);
            float v20 = float(value);
            float v21 = float(value);
            float v22 = float(value);
            float v23 = float(value);
            float v24 = float(value);
            float v25 = float(value);
            float v26 = float(value);
            float v27 = float(value);
            float v28 = float(value);
            float v29 = float(value);
            float v30 = float(value);
            float v31 = float(value);

            return BFloat6x32(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9,
                            v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
                            v20, v21, v22, v23, v24, v25, v26, v27, v28, v29,
                            v30, v31);
        }
#endif // #ifdef TENSILE_USE_BF6

#ifdef TENSILE_USE_FP4
        template <>
        inline Float4x2 DataInitialization::convertDoubleTo<Float4x2>(double value)
        {
            return Float4x2(float(value), float(value));
        }
#endif // #ifdef TENSILE_USE_FP4
#endif // !_WIN32

        template <>
        inline E8 DataInitialization::convertDoubleTo<E8>(double value)
        {
            return static_cast<E8>(static_cast<float>(value));
        }

        template <>
        inline E5M3 DataInitialization::convertDoubleTo<E5M3>(double value)
        {
            return static_cast<E5M3>(static_cast<float>(value));
        }
    } // namespace Client
} // namespace TensileLite
