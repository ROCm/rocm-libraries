/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "DataInitialization.hpp"

#include <hip/hip_runtime.h>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/hip/HipUtils.hpp>

namespace Tensile
{
    namespace Client
    {
        template <typename A,
                  typename B     = A,
                  typename C     = B,
                  typename D     = C,
                  typename Alpha = D,
                  typename Beta  = Alpha>
        struct ManagedContractionInputs : public TypedContractionInputs<A, B, C, D, Alpha, Beta>
        {
            using Base      = TypedContractionInputs<A, B, C, D, Alpha, Beta>;
            using AType     = A;
            using BType     = B;
            using CType     = C;
            using DType     = D;
            using AlphaType = Alpha;
            using BetaType  = Beta;

            ManagedContractionInputs(std::shared_ptr<A>    _a,
                                     std::shared_ptr<B>    _b,
                                     std::shared_ptr<C>    _c,
                                     std::shared_ptr<D>    _d,
                                     std::shared_ptr<A*>   _batchA,
                                     std::shared_ptr<B*>   _batchB,
                                     std::shared_ptr<C*>   _batchC,
                                     std::shared_ptr<D*>   _batchD,
                                     size_t                _aElements,
                                     size_t                _bElements,
                                     size_t                _cElements,
                                     size_t                _dElements,
                                     Alpha                 _alpha,
                                     Beta                  _beta,
                                     bool                  _gpu,
                                     std::shared_ptr<void> _ws            = nullptr,
                                     size_t                _workspaceSize = 0)

                : Base(_a.get(),
                       _b.get(),
                       _c.get(),
                       _d.get(),
                       _batchA.get(),
                       _batchB.get(),
                       _batchC.get(),
                       _batchD.get(),
                       _alpha,
                       _beta,
                       _ws.get())
                , managedA(_a)
                , managedB(_b)
                , managedC(_c)
                , managedD(_d)
                , managedBatchA(_batchA)
                , managedBatchB(_batchB)
                , managedBatchC(_batchC)
                , managedBatchD(_batchD)
                , aElements(_aElements)
                , bElements(_bElements)
                , cElements(_cElements)
                , dElements(_dElements)
                , gpu(_gpu)
                , managedWS(_ws)
                , workspaceSize(_workspaceSize)
            {
            }

            ~ManagedContractionInputs() = default;

            std::shared_ptr<A>    managedA;
            std::shared_ptr<B>    managedB;
            std::shared_ptr<C>    managedC;
            std::shared_ptr<D>    managedD;
            std::shared_ptr<A*>   managedBatchA;
            std::shared_ptr<B*>   managedBatchB;
            std::shared_ptr<C*>   managedBatchC;
            std::shared_ptr<D*>   managedBatchD;
            std::shared_ptr<void> managedWS;

            size_t aElements;
            size_t bElements;
            size_t cElements;
            size_t dElements;
            size_t workspaceSize;

            bool gpu;
        };

        template <typename TypedInputs>
        class TypedDataInitialization : public DataInitialization
        {
        public:
            using AType     = typename TypedInputs::AType;
            using BType     = typename TypedInputs::BType;
            using CType     = typename TypedInputs::CType;
            using DType     = typename TypedInputs::DType;
            using AlphaType = typename TypedInputs::AlphaType;
            using BetaType  = typename TypedInputs::BetaType;
            using ManagedInputs
                = ManagedContractionInputs<AType, BType, CType, DType, AlphaType, BetaType>;

            TypedDataInitialization(po::variables_map const&    args,
                                    ClientProblemFactory const& problemFactory,
                                    size_t                      maxWorkspaceSize = 0)
                : DataInitialization(args, problemFactory, maxWorkspaceSize)
            {
            }

            virtual std::shared_ptr<ContractionInputs>
                prepareCPUInputs(ContractionProblem const& problem)
            {
                return prepareCPUInputsTyped(problem);
            }

            virtual std::shared_ptr<ContractionInputs>
                prepareGPUInputs(ContractionProblem const& problem)
            {
                if(m_numRunsInSolution > 0 && m_curBoundsCheck == BoundsCheckMode::GuardPageFront
                   && m_boundsCheck == BoundsCheckMode::GuardPageAll)
                    m_curBoundsCheck = BoundsCheckMode::GuardPageBack;

                return prepareGPUInputsTyped(problem);
            }

            std::shared_ptr<ManagedInputs> prepareCPUInputsTyped(ContractionProblem const& problem)
            {
                if(!m_cpuInputsPristine)
                    m_cpuInputsPristine = createNewCPUInputs(problem);

                if(m_cpuInputs && m_curBoundsCheck == BoundsCheckMode::Disable
                   && !m_problemDependentData)
                {
                    copyD(m_cpuInputs, m_cpuInputsPristine);
                }
                else
                {
                    if(!m_cpuInputs)
                        m_cpuInputs = allocNewCPUInputs();

                    if(m_problemDependentData)
                        initializeCPUInputs(*m_cpuInputsPristine, problem);

                    if(m_curBoundsCheck == BoundsCheckMode::NaN && !m_cpuBadInputs)
                    {
                        m_cpuBadInputs = createNewCPUBadInputs();
                    }

                    copyInputs(m_cpuInputs, m_cpuInputsPristine, m_cpuBadInputs, problem);
                }

                if(m_convolutionVsContraction)
                {
                    bool allocated = false;
                    if(!m_cpuConvInputs)
                    {
                        allocated       = true;
                        m_cpuConvInputs = allocNewCPUInputs();
                    }

                    if(allocated || m_curBoundsCheck == BoundsCheckMode::NaN)
                        copyInputs(m_cpuConvInputs, m_cpuInputsPristine, m_cpuBadInputs, problem);
                }

                return m_cpuInputs;
            }

            virtual std::shared_ptr<ContractionInputs> cpuConvInputs() const
            {
                return m_cpuConvInputs;
            };

            template <typename T>
            void initGPUBatchedInput(T                          base,
                                     T*                         array,
                                     TensorDescriptor const&    tensor,
                                     const std::vector<size_t>& batchIdx)
            {
                std::vector<size_t> batchSizes;
                std::vector<size_t> batchStrides;
                for(auto& idx : batchIdx)
                {
                    batchSizes.push_back(tensor.sizes().at(idx));
                    batchStrides.push_back(tensor.strides().at(idx));
                }
                std::vector<size_t> coord(batchSizes.size(), 0);

                auto count = CoordCount(batchSizes.begin(), batchSizes.end());

                T* cpuArray = (T*)std::malloc(count * sizeof(T));
                for(size_t idx = 0; idx < count; idx++)
                {
                    CoordNumbered(
                        idx, coord.begin(), coord.end(), batchSizes.begin(), batchSizes.end());
                    cpuArray[idx] = base;
                    for(size_t i = 0; i < batchSizes.size(); i++)
                    {
                        cpuArray[idx] += coord[i] * batchStrides[i];
                    }
                }

                HIP_CHECK_EXC(hipMemcpy(array, cpuArray, count * sizeof(T), hipMemcpyHostToDevice));

                std::free(cpuArray);
            }

            void initializeGPUBatchedInputs(ManagedInputs&            inputs,
                                            ContractionProblem const& problem)
            {
                if(!inputs.gpu)
                    throw std::runtime_error("Initializing GPU batched inputs");

                auto                batchIdxs = problem.batchIndices();
                std::vector<size_t> batchIdxA(batchIdxs.size(), 0);
                std::vector<size_t> batchIdxB(batchIdxs.size(), 0);
                std::vector<size_t> batchIdxC(batchIdxs.size(), 0);
                std::vector<size_t> batchIdxD(batchIdxs.size(), 0);

                ptrdiff_t aPadding = 0;
                ptrdiff_t bPadding = 0;
                ptrdiff_t cPadding = 0;
                ptrdiff_t dPadding = 0;

                for(size_t i = 0; i < batchIdxs.size(); i++)
                {
                    batchIdxA[i] = batchIdxs[i].a;
                    batchIdxB[i] = batchIdxs[i].b;
                    batchIdxC[i] = batchIdxs[i].c;
                    batchIdxD[i] = batchIdxs[i].d;
                }

                if(m_curBoundsCheck == BoundsCheckMode::NaN)
                {
                    aPadding = (inputs.aElements - problem.a().totalAllocatedElements()) / 2;
                    bPadding = (inputs.bElements - problem.b().totalAllocatedElements()) / 2;
                    cPadding = (inputs.cElements - problem.c().totalAllocatedElements()) / 2;
                    dPadding = (inputs.dElements - problem.d().totalAllocatedElements()) / 2;
                }
                else if(m_curBoundsCheck == BoundsCheckMode::GuardPageBack)
                {
                    aPadding = inputs.aElements - problem.a().totalAllocatedElements();
                    bPadding = inputs.bElements - problem.b().totalAllocatedElements();
                    cPadding = inputs.cElements - problem.c().totalAllocatedElements();
                    dPadding = inputs.dElements - problem.d().totalAllocatedElements();
                }

                initGPUBatchedInput(inputs.managedA.get() + aPadding,
                                    inputs.managedBatchA.get(),
                                    problem.a(),
                                    batchIdxA);
                initGPUBatchedInput(inputs.managedB.get() + bPadding,
                                    inputs.managedBatchB.get(),
                                    problem.b(),
                                    batchIdxB);
                initGPUBatchedInput(inputs.managedC.get() + cPadding,
                                    inputs.managedBatchC.get(),
                                    problem.c(),
                                    batchIdxC);
                initGPUBatchedInput(inputs.managedD.get() + dPadding,
                                    inputs.managedBatchD.get(),
                                    problem.d(),
                                    batchIdxD);
            }

            std::shared_ptr<ManagedInputs> prepareGPUInputsTyped(ContractionProblem const& problem)
            {
                std::shared_ptr<ManagedInputs> pristine;
                std::shared_ptr<ManagedInputs> bad;

                if(m_keepPristineCopyOnGPU && !m_problemDependentData)
                {
                    if(!m_gpuInputsPristine)
                        m_gpuInputsPristine = createNewGPUInputs(problem);

                    pristine = m_gpuInputsPristine;

                    if(m_curBoundsCheck == BoundsCheckMode::NaN)
                    {
                        if(!m_gpuBadInputs)
                            m_gpuBadInputs = createNewGPUBadInputs();

                        bad = m_gpuBadInputs;
                    }
                }
                else
                {
                    if(!m_cpuInputsPristine)
                        m_cpuInputsPristine = createNewCPUInputs(problem);

                    pristine = m_cpuInputsPristine;

                    if(m_curBoundsCheck == BoundsCheckMode::NaN)
                    {
                        if(!m_cpuBadInputs)
                            m_cpuBadInputs = createNewCPUBadInputs();

                        bad = m_cpuBadInputs;
                    }
                }

                if(m_gpuInputs && m_curBoundsCheck == BoundsCheckMode::Disable
                   && !m_problemDependentData)
                {
                    if(m_elementsToValidate)
                        copyD(m_gpuInputs, pristine);
                }
                else
                {
                    if(!m_gpuInputs)
                        m_gpuInputs = allocNewGPUInputs(pristine);

                    if(m_problemDependentData)
                        initializeCPUInputs(*m_cpuInputsPristine, problem);

                    copyInputs(m_gpuInputs, pristine, bad, problem);
                }

                initializeGPUBatchedInputs(*m_gpuInputs, problem);

                return m_gpuInputs;
            }

            std::shared_ptr<ManagedInputs> createNewCPUInputs(ContractionProblem const& problem)
            {
                auto rv = allocNewCPUInputs();
                initializeCPUInputs(*rv, problem);

                return rv;
            }

            std::shared_ptr<ManagedInputs> createNewGPUInputs(ContractionProblem const& problem)
            {
                auto rv = allocNewGPUInputs();
                if(!m_cpuInputsPristine)
                    m_cpuInputsPristine = createNewCPUInputs(problem);
                copyInputBuffers(rv, m_cpuInputsPristine);

                return rv;
            }

            std::shared_ptr<ManagedInputs> createNewCPUBadInputs()
            {
                auto rv = allocNewCPUInputs();
                initializeCPUBadInputs(*rv);

                return rv;
            }

            std::shared_ptr<ManagedInputs> createNewGPUBadInputs()
            {
                auto rv = allocNewGPUInputs();
                if(!m_cpuBadInputs)
                    m_cpuBadInputs = createNewCPUBadInputs();
                copyInputBuffers(rv, m_cpuBadInputs);

                return rv;
            }

            std::shared_ptr<ManagedInputs> allocNewCPUInputs(std::shared_ptr<ManagedInputs> pristine
                                                             = nullptr)
            {
                std::shared_ptr<AType> a;
                std::shared_ptr<BType> b;
                std::shared_ptr<CType> c;
                std::shared_ptr<DType> d;

                if(pristine)
                {
                    a = pristine->managedA;
                    b = pristine->managedB;
                }
                else
                {
                    a = std::shared_ptr<AType>(
                        (AType*)std::malloc(TypeInfo<AType>::ElementSize * m_aMaxElements),
                        std::free);
                    if(a == nullptr)
                        throw std::runtime_error("out of host memory allocating a");
                    b = std::shared_ptr<BType>(
                        (BType*)std::malloc(TypeInfo<BType>::ElementSize * m_bMaxElements),
                        std::free);
                    if(a == nullptr)
                        throw std::runtime_error("out of host memory allocating b");
                }

                if(m_cEqualsD || !pristine)
                {
                    c = std::shared_ptr<CType>(
                        (CType*)std::malloc(TypeInfo<CType>::ElementSize * m_cMaxElements),
                        std::free);
                    if(c == nullptr)
                        throw std::runtime_error("out of host memory allocating c");
                }
                else
                {
                    c = pristine->managedC;
                }

                if(m_cEqualsD)
                {
                    d = c;
                }
                else if(pristine)
                {
                    d = pristine->managedD;
                }
                else
                {
                    d = std::shared_ptr<DType>(
                        (DType*)std::malloc(TypeInfo<DType>::ElementSize * m_dMaxElements),
                        std::free);
                    if(d == nullptr)
                        throw std::runtime_error("out of host memory allocating d");
                }

                auto alpha = static_cast<AlphaType>(0);
                auto beta  = static_cast<BetaType>(0);

                auto rv = std::make_shared<ManagedInputs>(a,
                                                          b,
                                                          c,
                                                          d,
                                                          std::shared_ptr<AType*>(),
                                                          std::shared_ptr<BType*>(),
                                                          std::shared_ptr<CType*>(),
                                                          std::shared_ptr<DType*>(),
                                                          m_aMaxElements,
                                                          m_bMaxElements,
                                                          m_cMaxElements,
                                                          m_dMaxElements,
                                                          alpha,
                                                          beta,
                                                          false);

                return rv;
            }

            template <typename T>
            std::shared_ptr<T> allocNewGPUBuffer(const char* title, size_t size)
            {
                static const int sizew = 10;
                T*               ptr   = nullptr;
                HIP_CHECK_EXC(hipMalloc(&ptr, size));
                auto p = std::shared_ptr<T>(ptr, hipFree);
                if(Debug::Instance().printTensorInfo())
                    std::cout << "info: allocate " << title << " " << std::setw(sizew) << size
                              << " bytes at " << static_cast<void*>(ptr) << "\n";
                return p;
            }

            std::shared_ptr<ManagedInputs> allocNewGPUInputs(std::shared_ptr<ManagedInputs> pristine
                                                             = nullptr)
            {
                if(m_curBoundsCheck != BoundsCheckMode::Disable || (pristine && !pristine->gpu))
                    pristine = nullptr;

                std::shared_ptr<AType> a;
                std::shared_ptr<BType> b;
                std::shared_ptr<CType> c;
                std::shared_ptr<DType> d;

                std::shared_ptr<AType*> batch_a;
                std::shared_ptr<BType*> batch_b;
                std::shared_ptr<CType*> batch_c;
                std::shared_ptr<DType*> batch_d;

                std::shared_ptr<void> ws;
                static const int      sizew = 10;

                std::vector<std::shared_ptr<void>> guardPage;
                void*                              guardPagePtr;
                bool enableGuardPage = (m_curBoundsCheck == BoundsCheckMode::GuardPageFront
                                        || m_curBoundsCheck == BoundsCheckMode::GuardPageBack);

                if(pristine)
                {
                    a       = pristine->managedA;
                    b       = pristine->managedB;
                    batch_a = pristine->managedBatchA;
                    batch_b = pristine->managedBatchB;
                }
                else
                {
                    if(enableGuardPage)
                    {
                        HIP_CHECK_EXC(hipMalloc(&guardPagePtr, pageSize));
                        guardPage.push_back(std::shared_ptr<void>(guardPagePtr, hipFree));
                    }

                    a       = allocNewGPUBuffer<AType>("a",
                                                 TypeInfo<AType>::ElementSize * m_aMaxElements);
                    batch_a = allocNewGPUBuffer<AType*>("batchA", sizeof(AType*) * m_maxBatch);

                    if(enableGuardPage)
                    {
                        HIP_CHECK_EXC(hipMalloc(&guardPagePtr, pageSize));
                        guardPage.push_back(std::shared_ptr<void>(guardPagePtr, hipFree));
                    }

                    b       = allocNewGPUBuffer<BType>("b",
                                                 TypeInfo<BType>::ElementSize * m_bMaxElements);
                    batch_b = allocNewGPUBuffer<BType*>("batchB", sizeof(BType*) * m_maxBatch);
                }

                if(m_cEqualsD || !pristine)
                {
                    if(enableGuardPage)
                    {
                        HIP_CHECK_EXC(hipMalloc(&guardPagePtr, pageSize));
                        guardPage.push_back(std::shared_ptr<void>(guardPagePtr, hipFree));
                    }

                    c       = allocNewGPUBuffer<CType>("c",
                                                 TypeInfo<CType>::ElementSize * m_cMaxElements);
                    batch_c = allocNewGPUBuffer<CType*>("batchC", sizeof(CType*) * m_maxBatch);
                }
                else
                {
                    c       = pristine->managedC;
                    batch_c = pristine->managedBatchC;
                }

                if(m_cEqualsD)
                {
                    d       = c;
                    batch_d = batch_c;
                }
                else if(pristine)
                {
                    d       = pristine->managedD;
                    batch_d = pristine->managedBatchD;
                }
                else
                {
                    if(enableGuardPage)
                    {
                        HIP_CHECK_EXC(hipMalloc(&guardPagePtr, pageSize));
                        guardPage.push_back(std::shared_ptr<void>(guardPagePtr, hipFree));
                    }

                    d       = allocNewGPUBuffer<DType>("d",
                                                 TypeInfo<DType>::ElementSize * m_dMaxElements);
                    batch_d = allocNewGPUBuffer<DType*>("batchD", sizeof(DType*) * m_maxBatch);
                }

                if(enableGuardPage)
                {
                    HIP_CHECK_EXC(hipMalloc(&guardPagePtr, pageSize));
                    guardPage.push_back(std::shared_ptr<void>(guardPagePtr, hipFree));
                }

                if(pristine)
                {
                    ws = pristine->managedWS;
                }
                else
                {
                    ws = allocNewGPUBuffer<void>("ws", m_workspaceSize);
                }

                auto alpha = static_cast<AlphaType>(0);
                auto beta  = static_cast<BetaType>(0);

                auto rv = std::make_shared<ManagedInputs>(a,
                                                          b,
                                                          c,
                                                          d,
                                                          batch_a,
                                                          batch_b,
                                                          batch_c,
                                                          batch_d,
                                                          m_aMaxElements,
                                                          m_bMaxElements,
                                                          m_cMaxElements,
                                                          m_dMaxElements,
                                                          alpha,
                                                          beta,
                                                          true,
                                                          ws,
                                                          m_workspaceSize);
                return rv;
            }

            void initializeCPUInputs(ManagedInputs& inputs, ContractionProblem const& problem)
            {
                if(inputs.gpu)
                    throw std::runtime_error("Initializing GPU inputs as CPU.");

                if(m_problem.a() != problem.a() || m_problem.b() != problem.b()
                   || m_problem.c() != problem.c() || (!m_cEqualsD && m_problem.d() != problem.d()))
                {
                    if(m_problemDependentData)
                    {
                        initArray(m_aInit, inputs.managedA.get(), problem.a());
                        initArray(m_bInit, inputs.managedB.get(), problem.b());
                        initArray(m_cInit, inputs.managedC.get(), problem.c());
                        if(!m_cEqualsD)
                            initArray(m_dInit, inputs.managedD.get(), problem.d());
                    }
                    else
                    {
                        initArray(m_aInit, inputs.managedA.get(), m_aMaxElements);
                        initArray(m_bInit, inputs.managedB.get(), m_bMaxElements);
                        initArray(m_cInit, inputs.managedC.get(), m_cMaxElements);
                        if(!m_cEqualsD)
                            initArray(m_dInit, inputs.managedD.get(), m_dMaxElements);
                    }

                    inputs.alpha = getValue<AlphaType>(m_alphaInit);
                    inputs.beta  = getValue<BetaType>(m_betaInit);

                    m_problem = problem;
                }
            }

            void initializeCPUBadInputs(ManagedInputs& inputs)
            {
                if(inputs.gpu)
                    throw std::runtime_error("Initializing GPU inputs as CPU.");

                initArray(InitMode::BadInput, inputs.managedA.get(), m_aMaxElements);
                initArray(InitMode::BadInput, inputs.managedB.get(), m_bMaxElements);

                initArray(InitMode::BadOutput, inputs.managedD.get(), m_dMaxElements);

                if(!m_cEqualsD)
                    initArray(InitMode::BadInput, inputs.managedC.get(), m_cMaxElements);

                inputs.alpha = getValue<AlphaType>(m_alphaInit);
                inputs.beta  = getValue<BetaType>(m_betaInit);
            }

            hipMemcpyKind getCopyKind(std::shared_ptr<ManagedInputs> dst,
                                      std::shared_ptr<ManagedInputs> src)
            {
                if(src->gpu)
                {
                    if(dst->gpu)
                    {
                        return hipMemcpyDeviceToDevice;
                    }
                    else
                    {
                        return hipMemcpyDeviceToHost;
                    }
                }
                else
                {
                    if(dst->gpu)
                    {
                        return hipMemcpyHostToDevice;
                    }
                    else
                    {
                        return hipMemcpyHostToHost;
                    }
                }
            }

            void copyInputBuffers(std::shared_ptr<ManagedInputs> dst,
                                  std::shared_ptr<ManagedInputs> src)
            {
                hipMemcpyKind kind = getCopyKind(dst, src);

                if(dst->managedA != src->managedA)
                    HIP_CHECK_EXC(hipMemcpy(dst->managedA.get(),
                                            src->managedA.get(),
                                            TypeInfo<AType>::ElementSize * m_aMaxElements,
                                            kind));

                if(dst->managedB != src->managedB)
                    HIP_CHECK_EXC(hipMemcpy(dst->managedB.get(),
                                            src->managedB.get(),
                                            TypeInfo<BType>::ElementSize * m_bMaxElements,
                                            kind));

                if(dst->managedC != src->managedC)
                    HIP_CHECK_EXC(hipMemcpy(dst->managedC.get(),
                                            src->managedC.get(),
                                            TypeInfo<CType>::ElementSize * m_cMaxElements,
                                            kind));

                if(!m_cEqualsD && dst->managedD != src->managedD)
                    HIP_CHECK_EXC(hipMemcpy(dst->managedD.get(),
                                            src->managedD.get(),
                                            TypeInfo<DType>::ElementSize * m_dMaxElements,
                                            kind));

                dst->alpha = src->alpha;
                dst->beta  = src->beta;
            }

            void copyInputs(std::shared_ptr<ManagedInputs> dst,
                            std::shared_ptr<ManagedInputs> src,
                            std::shared_ptr<ManagedInputs> bad,
                            ContractionProblem const&      problem)
            {
                hipMemcpyKind kind = getCopyKind(dst, src);

                if(m_curBoundsCheck == BoundsCheckMode::NaN)
                {
                    if(!bad)
                        throw std::runtime_error(
                            "bad inputs must be initialized for bounds check!");
                    if(bad->gpu != src->gpu)
                        throw std::runtime_error(
                            "bad inputs must be in the same location as the source");
                    if(dst->managedA == src->managedA)
                        throw std::runtime_error("A pointers are equal for bounds check!");
                    if(dst->managedB == src->managedB)
                        throw std::runtime_error("B pointers are equal for bounds check!");
                    if(dst->managedC == src->managedC)
                        throw std::runtime_error("C pointers are equal for bounds check!");
                    if(dst->managedD == src->managedD)
                        throw std::runtime_error("D pointers are equal for bounds check!");

                    copyInputBuffers(dst, bad);

                    {
                        ptrdiff_t aPadding = dst->aElements - problem.a().totalAllocatedElements();
                        dst->a             = dst->managedA.get() + aPadding / 2;
                    }

                    {
                        ptrdiff_t bPadding = dst->bElements - problem.b().totalAllocatedElements();
                        dst->b             = dst->managedB.get() + bPadding / 2;
                    }

                    {
                        ptrdiff_t cPadding = dst->cElements - problem.c().totalAllocatedElements();
                        dst->c             = dst->managedC.get() + cPadding / 2;
                    }

                    {
                        ptrdiff_t dPadding = dst->dElements - problem.d().totalAllocatedElements();
                        dst->d             = dst->managedD.get() + dPadding / 2;
                    }

                    Tensile::hip::CopyTensor(const_cast<AType*>(dst->a), src->a, problem.a(), kind);
                    Tensile::hip::CopyTensor(const_cast<BType*>(dst->b), src->b, problem.b(), kind);

                    Tensile::hip::CopyTensor(const_cast<CType*>(dst->c), src->c, problem.c(), kind);

                    if(!m_cEqualsD)
                        Tensile::hip::CopyTensor(dst->d, src->d, problem.d(), kind);

                    dst->alpha = src->alpha;
                    dst->beta  = src->beta;
                }
                else if(m_curBoundsCheck == BoundsCheckMode::GuardPageBack)
                {
                    {
                        ptrdiff_t aPadding = dst->aElements - problem.a().totalAllocatedElements();
                        dst->a             = dst->managedA.get() + aPadding;
                    }

                    {
                        ptrdiff_t bPadding = dst->bElements - problem.b().totalAllocatedElements();
                        dst->b             = dst->managedB.get() + bPadding;
                    }

                    {
                        ptrdiff_t cPadding = dst->cElements - problem.c().totalAllocatedElements();
                        dst->c             = dst->managedC.get() + cPadding;
                    }

                    {
                        ptrdiff_t dPadding = dst->dElements - problem.d().totalAllocatedElements();
                        dst->d             = dst->managedD.get() + dPadding;
                    }
                    HIP_CHECK_EXC(hipMemcpy(const_cast<AType*>(dst->a),
                                            src->a,
                                            TypeInfo<AType>::ElementSize
                                                * problem.a().totalAllocatedElements(),
                                            kind));
                    HIP_CHECK_EXC(hipMemcpy(const_cast<BType*>(dst->b),
                                            src->b,
                                            TypeInfo<BType>::ElementSize
                                                * problem.b().totalAllocatedElements(),
                                            kind));
                    HIP_CHECK_EXC(hipMemcpy(const_cast<CType*>(dst->c),
                                            src->c,
                                            TypeInfo<CType>::ElementSize
                                                * problem.c().totalAllocatedElements(),
                                            kind));

                    if(!m_cEqualsD)
                        HIP_CHECK_EXC(hipMemcpy(const_cast<DType*>(dst->d),
                                                src->d,
                                                TypeInfo<DType>::ElementSize
                                                    * problem.d().totalAllocatedElements(),
                                                kind));

                    dst->alpha = src->alpha;
                    dst->beta  = src->beta;
                }
                else
                {
                    copyInputBuffers(dst, src);
                }
            }

            void copyD(std::shared_ptr<ManagedInputs> dst, std::shared_ptr<ManagedInputs> src)
            {
                hipMemcpyKind kind = getCopyKind(dst, src);

                HIP_CHECK_EXC(hipMemcpy(dst->managedD.get(),
                                        src->managedD.get(),
                                        TypeInfo<DType>::ElementSize * m_dMaxElements,
                                        kind));
            }

        private:
            /**
             * Depending on user configuration, the actual pointers within these inputs structs may not
             * all point to separately allocated buffers.
             */

            std::shared_ptr<ManagedInputs> m_cpuConvInputs;
            std::shared_ptr<ManagedInputs> m_cpuInputsPristine; //< Untouched copies of the inputs
            std::shared_ptr<ManagedInputs>
                m_cpuInputs; //< Inputs used for CPU reference calculations
            std::shared_ptr<ManagedInputs>
                m_cpuBadInputs; //< Inputs containing 'bad' values for bounds checking
            std::shared_ptr<ManagedInputs> m_gpuInputsPristine; //< Untouched copies of the inputs
            std::shared_ptr<ManagedInputs> m_gpuInputs; //< Inputs to be sent in to GPU kernels
            std::shared_ptr<ManagedInputs> m_gpuBadInputs; //< GPU copies of 'bad' values
            ContractionProblem
                m_problem; //< Contraction problem for which current inputs are initialized
        };

        // Commonly used managed contraction input type groupings
        // Naming: _[Ti_To_Tc]_:
        // S=float, D=double, C=complex<float>, Z=complex<double>,
        // H=Half, B=BF16, I8x4=Int8x4, I32=int32_t
        using ManagedContractionInputs_S_S_S = ManagedContractionInputs<float>;
        using ManagedContractionInputs_D_D_D = ManagedContractionInputs<double>;
        using ManagedContractionInputs_C_C_C = ManagedContractionInputs<std::complex<float>>;
        using ManagedContractionInputs_Z_Z_Z = ManagedContractionInputs<std::complex<double>>;
#ifdef TENSILE_USE_HALF
        using ManagedContractionInputs_H_H_H = ManagedContractionInputs<Half>;
        using ManagedContractionInputs_H_H_S
            = ManagedContractionInputs<Half, Half, Half, Half, float, float>;
        using ManagedContractionInputs_H_S_S = ManagedContractionInputs<Half, Half, float, float>;
#endif // TENSILE_USE_HALF
        using ManagedContractionInputs_I8x4_I32_I32
            = ManagedContractionInputs<Int8x4, Int8x4, int32_t, int32_t>;
        using ManagedContractionInputs_I8_I32_I32
            = ManagedContractionInputs<int8_t, int8_t, int32_t, int32_t>;
        using ManagedContractionInputs_I32_I32_I32 = ManagedContractionInputs<int32_t>;
#ifdef TENSILE_USE_BF16
        using ManagedContractionInputs_B_B_S
            = ManagedContractionInputs<BFloat16, BFloat16, BFloat16, BFloat16, float, float>;
        using ManagedContractionInputs_B_S_S
            = ManagedContractionInputs<BFloat16, BFloat16, float, float>;
#endif // TENSILE_USE_BF16
#ifdef TENSILE_USE_FP8_BF8
        using ManagedContractionInputs_F8_F8_S
            = ManagedContractionInputs<Float8, Float8, Float8, Float8, float, float>;
        using ManagedContractionInputs_F8_S_S
            = ManagedContractionInputs<Float8, Float8, float, float>;
        using ManagedContractionInputs_B8_B8_S
            = ManagedContractionInputs<BFloat8, BFloat8, BFloat8, BFloat8, float, float>;
        using ManagedContractionInputs_B8_S_S
            = ManagedContractionInputs<BFloat8, BFloat8, float, float>;
        // hybrid cases: F8B8SS, B8F8SS
        using ManagedContractionInputs_F8B8_S_S
            = ManagedContractionInputs<Float8, BFloat8, float, float>;
        using ManagedContractionInputs_B8F8_S_S
            = ManagedContractionInputs<BFloat8, Float8, float, float>;
        // hybrid cases with To = B8
        using ManagedContractionInputs_F8B8_B8_S
            = ManagedContractionInputs<Float8, BFloat8, BFloat8, BFloat8, float, float>;
        using ManagedContractionInputs_B8F8_B8_S
            = ManagedContractionInputs<BFloat8, Float8, BFloat8, BFloat8, float, float>;
        // cases with To = f16
        using ManagedContractionInputs_F8_H_S
            = ManagedContractionInputs<Float8, Float8, Half, Half, float, float>;
        using ManagedContractionInputs_B8_H_S
            = ManagedContractionInputs<BFloat8, BFloat8, Half, Half, float, float>;
        using ManagedContractionInputs_F8B8_H_S
            = ManagedContractionInputs<Float8, BFloat8, Half, Half, float, float>;
        using ManagedContractionInputs_B8F8_H_S
            = ManagedContractionInputs<BFloat8, Float8, Half, Half, float, float>;
#endif // TENSILE_USE_FP8_BF8

    } // namespace Client
} // namespace Tensile
