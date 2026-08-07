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

#include <mxDataGen.hpp>

#include <cstddef>

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

        inline bool isMXTensor(const TensorDescriptor& tensor, size_t mxBlock)
        {
            if(mxBlock == 0)
                return false;
            auto dt = tensor.dataType();
            return dt == rocisa::DataType::Float4
                || dt == rocisa::DataType::Float6
                || dt == rocisa::DataType::BFloat6
                || dt == rocisa::DataType::Float8
                || dt == rocisa::DataType::BFloat8;
        }

        inline bool isMXProblem(const ContractionProblemGemm& problem)
        {
            return isMXTensor(problem.a(), problem.mxBlockA())
                || isMXTensor(problem.b(), problem.mxBlockB());
        }

        // Problem-indept. from 0~7, and 16, and 23~26 (fixed values for every problem)
        // And problem-dept. from 8~15 (values depend on problem)
        // RandomNegPosLimited: integer -128~128. fp -1.0~1.0
        // UniformLowPrecision (27): uniform random in [-maxVal, maxVal] where maxVal is
        //   the maximum representable value of the target type. Only supported for
        //   low-precision packed types (FP4, FP6, BF6). Produces significantly fewer
        //   zeros than Random by covering the full representable range uniformly.
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
            UniformLowPrecision, // 27
            Count
        };

        bool tryHostValidationInitialize(
            rocisa::DataType dataType, InitMode mode, void* array, size_t elements);
        bool tryHostValidationInitialize(rocisa::DataType dataType,
                                         InitMode         mode,
                                         void*            array,
                                         size_t           elements,
                                         double           freeValue);
        bool tryHostValidationInitialize(rocisa::DataType        dataType,
                                         InitMode                 mode,
                                         void*                    array,
                                         TensorDescriptor const& descriptor);
        double hostValidationDoubleValue(InitMode mode, double freeValue = 0.0);
        double hostValidationUniformDouble(double lower, double upper);

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

        void initCPUSparseInput(PruneSparseMode         mode,
                                void*                   dstPruned,
                                void*                   dstCompressed,
                                void*                   dstMeta,
                                TensorDescriptor const& tensor,
                                TensorDescriptor const& tensorC,
                                TensorDescriptor const& tensorMeta,
                                size_t                  dim,
                                bool                    metadataLayout);

        class DataInitialization : public RunListener
        {
        public:
            static double GetRepresentativeBetaValue(po::variables_map const& args);

            DataInitialization(po::variables_map const&    args,
                               ClientProblemFactory const& problemFactory);
            ~DataInitialization();

            // True when the CPU reference must be recomputed for this solution
            // because DataInitialization refreshes MX inputs per solution
            // (solution-dependent HostPreSwizzle, gfx950). When false (e.g.
            // gfx1250, gfx942, non-MX) the per-problem reference is still valid
            // and can be reused across all solutions.
            bool referenceNeedsPerSolutionRecompute(ContractionProblemGemm const& problem,
                                                    ContractionSolution const*    solution) const
            {
                return needsSolutionDependentMXPreswizzle(problem, solution);
            }

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
                bool needMXSwizzle = (problem.mxBlockA() != 0) || (problem.mxBlockB() != 0);

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
                   && !m_problemDependentData && !needSwizzle && !needMXSwizzle)
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
                    if(needSwizzle || needMXSwizzle)
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
                if(tryHostValidationInitialize(dataType, initMode, array, descriptor))
                    return;
                throw std::invalid_argument(
                    "TensileLite CPU initialization mode/type is not represented "
                    "by host-validation.");
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
                // Re-init MX inputs for solution-dependent HostPreSwizzle.
                if(m_currentSolution != nullptr
                   && m_currentGemmProblem != nullptr
                   && !m_gpuPtrs.empty()
                   && needsSolutionDependentMXPreswizzle(*m_currentGemmProblem,
                                                         m_currentSolution))
                {
                    initializeMXData(*m_currentGemmProblem);
                    copyValidToGPUBuffer(*m_currentGemmProblem);
                    copyInputs(m_gpuPtrs,
                               m_gpuBatchPtrs,
                               m_maxElements,
                               m_groupedOffsets,
                               *m_currentGemmProblem,
                               hipMemcpyDeviceToDevice);
                    // Sync CPU current buffers so the reference matches GPU data.
                    for(int ti : {ContractionProblemGemm::TENSOR::A,
                                  ContractionProblemGemm::TENSOR::B,
                                  ContractionProblemGemm::TENSOR::MXSA,
                                  ContractionProblemGemm::TENSOR::MXSB})
                    {
                        auto& desc = m_currentGemmProblem->tensors()[ti];
                        auto  it   = m_vdata[ti].pristine.find(desc.dataType());
                        if(it == m_vdata[ti].pristine.end())
                            continue;
                        auto& p = it->second;
                        if(p.cpuInput.valid && p.cpuInput.current)
                        {
                            size_t bytes = multiplyElementSize(
                                p.maxElements, desc.elementBytes());
                            std::memcpy(p.cpuInput.current.get(),
                                        p.cpuInput.valid.get(),
                                        bytes);
                        }
                    }
                }
            }
            virtual void postSolution() override
            {
                if(m_boundsCheck == BoundsCheckMode::GuardPageAll)
                {
                    m_numRunsInSolution = 0;
                    m_curBoundsCheck    = BoundsCheckMode::GuardPageFront;
                }
            }
            virtual bool needMoreRunsInSolution() const override
            {
                return m_numRunsInSolution < m_numRunsPerSolution;
            };

            virtual size_t numWarmupRuns() override
            {
                if(m_numRunsInSolution < m_numRunsPerSolution)
                    return 1;
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

            void initializeMXData(ContractionProblemGemm const& problem);

            // True when swizzled MX scales depend on the solution.
            bool needsSolutionDependentMXPreswizzle(
                ContractionProblemGemm const& problem,
                ContractionSolution const*    solution) const
            {
                return isMXProblem(problem) && m_mxScaleFormat > 0
                       && m_mxScaleLayout == MXScaleLayout::GFX950
                       && solution != nullptr
                       && solution->problemType.mxScaleFormat == 1;
            }

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
            std::vector<std::shared_ptr<void>>    m_guardPages;
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
            MXScaleLayout m_mxScaleLayout = MXScaleLayout::None;
            // Set by initializeMXData when a preswizzled scale was uploaded
            // straight into gpuInput.valid (i.e. copySwizzledToGPUBuffer can
            // hand back gpuInput.valid as-is rather than re-swizzling).
            bool m_mxPreswizzledA = false;
            bool m_mxPreswizzledB = false;
        };

    } // namespace Client
} // namespace TensileLite
