/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef GEMM_KERNEL_TEST_IMPL_HPP
#define GEMM_KERNEL_TEST_IMPL_HPP

#include "GEMMKernelTest.hpp"

#include <CL/cl2.hpp>
#include <Tensile/SolutionLibrary.hpp>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/EmbeddedLibrary.hpp>
#include <Tensile/Utils.hpp>

#include <Tensile/AMDGPU_Detail.hpp>
#include <Tensile/ContractionProblem_Detail.hpp>
#include <Tensile/TensorDescriptor_Detail.hpp>
#include <Tensile/TensorOps.hpp>

#include "TestData.hpp"
#include "TestUtils.hpp"

#include <Reference.hpp>

#include <cstddef>
#include <random>
#include <unordered_map>

using namespace Tensile;

namespace std
{
    template <>
    struct hash<std::complex<float>>
    {
        inline size_t operator()(std::complex<float> const& obj) const
        {
            return Tensile::hash_combine(obj.real(), obj.imag());
        }
    };
    template <>
    struct hash<std::complex<double>>
    {
        inline size_t operator()(std::complex<double> const& obj) const
        {
            return Tensile::hash_combine(obj.real(), obj.imag());
        }
    };

    template <>
    struct hash<Tensile::BFloat16>
    {
        inline size_t operator()(Tensile::BFloat16 const& obj) const
        {
            return hash<decltype(obj.data)>()(obj.data);
        }
    };

#ifdef TENSILE_USE_HALF

    template <>
    struct hash<Tensile::Half>
    {
        inline size_t operator()(Tensile::Half const& obj) const
        {
            return hash<float>()(static_cast<float>(obj));
        }
    };

#else
    template <>
    struct hash<Tensile::Half>
    {
        inline size_t operator()(Tensile::Half const& obj) const
        {
            return hash<decltype(obj.value)>()(obj.value);
        }
    };

#endif // TENSILE_USE_HALF

} // namespace std

/* Utils */
template <typename T>
inline void expectEqual(T const& l, T const& r, size_t i, bool& fail)
{
    if(!fail)
    {
        EXPECT_EQ(l, r) << i << ": " << (fail = true);
    }
}

template <>
inline void expectEqual(float const& l, float const& r, size_t i, bool& fail)
{
    if(!fail)
    {
        EXPECT_FLOAT_EQ(l, r) << i << ": " << (fail = true);
    }
}

template <>
inline void expectEqual(double const& l, double const& r, size_t i, bool& fail)
{
    if(!fail)
    {
        EXPECT_DOUBLE_EQ(l, r) << i << ": " << (fail = true);
    }
}

template <typename T>
void expectEqual(std::vector<T> const& l, std::vector<T> const& r)
{
    bool fail = false;

    auto size = std::min(l.size(), r.size());

#pragma omp parallel for
    for(size_t i = 0; i < size; i++)
    {
#pragma omp flush(fail)
        expectEqual(l[i], r[i], i, fail);
    }
    ASSERT_EQ(fail, false);
}

///////////////////////////////////////////////////////////////
// template<typename DeviceBackend>
// class GEMMKernelTest
///////////////////////////////////////////////////////////////

// Static impl
template <typename DeviceBackend>
constexpr typename GEMMKernelTest<DeviceBackend>::ProblemParams
    GEMMKernelTest<DeviceBackend>::RandomGEMMParams;

// Operator impl
template <typename DeviceBackend>
inline std::ostream& operator<<(std::ostream&                                         stream,
                                std::shared_ptr<GEMMKernelTest<DeviceBackend>> const& ptr)
{
    if(ptr)
    {
        return stream << "*" << ptr->ToString();
    }
    else
    {
        return stream << "(nullptr)";
    }
}

///////////////////////////////////////////////////////////////
// template<typename TypedInputs, typename DeviceBackend>
// class TypedGEMMKernelTest
///////////////////////////////////////////////////////////////

// Static impl
template <typename TypedInputs, typename DeviceBackend>
std::unordered_map<size_t, std::vector<typename TypedInputs::DType>>
    TypedGEMMKernelTest<TypedInputs, DeviceBackend>::referenceCache;

// Function impl
template <typename TypedInputs, typename DeviceBackend>
auto TypedGEMMKernelTest<TypedInputs, DeviceBackend>::createProblem(ProblemParams const& props)
    -> ContractionProblemGemm
{
    if(props == Base::RandomGEMMParams)
    {
        return RandomGEMM<AType, BType, CType, DType>();
    }
    else
    {
        bool   transA     = std::get<0>(props);
        bool   transB     = std::get<1>(props);
        size_t m          = std::get<2>(props);
        size_t n          = std::get<3>(props);
        size_t k          = std::get<4>(props);
        size_t lda        = std::get<5>(props);
        size_t ldb        = std::get<6>(props);
        size_t ldc        = std::get<7>(props);
        double beta       = std::get<8>(props);
        size_t batchCount = std::get<9>(props);

        typename ContractionProblemGemm::FreeIndices free(2);
        typename ContractionProblemGemm::BoundIndex  bound;

        free[0].isA = true;
        free[0].i = free[0].c = free[0].d = 0;
        free[1].isA                       = false;
        free[1].i = free[1].c = free[1].d = 1;

        TensorDescriptor a, b, c, d;
        if(transA)
        {
            a         = TensorDescriptor(TypeInfo<AType>::Enum, {k, m}, {1, lda});
            free[0].i = 1;
            bound.a   = 0;
        }
        else
        {
            a         = TensorDescriptor(TypeInfo<AType>::Enum, {m, k}, {1, lda});
            free[0].i = 0;
            bound.a   = 1;
        }

        if(transB)
        {
            b         = TensorDescriptor(TypeInfo<BType>::Enum, {n, k}, {1, ldb});
            free[1].i = 0;
            bound.b   = 1;
        }
        else
        {
            b         = TensorDescriptor(TypeInfo<BType>::Enum, {k, n}, {1, ldb});
            free[1].i = 1;
            bound.b   = 0;
        }

        typename ContractionProblemGemm::FreeIndices  freeIndices{free};
        typename ContractionProblemGemm::BatchIndices batchIndices;
        typename ContractionProblemGemm::BoundIndices boundIndices{bound};

        d = TensorDescriptor(TypeInfo<DType>::Enum, {m, n}, {1, ldc});

        a.appendDim(batchCount);
        b.appendDim(batchCount);
        d.appendDim(batchCount);

        batchIndices.push_back({2, 2, 2, 2});

        c = d;

        TensorOps nop;

        return ContractionProblemGemm(
            a, nop, b, nop, c, nop, d, nop, freeIndices, batchIndices, boundIndices, beta);
    }
}

template <typename TypedInputs, typename DeviceBackend>
void TypedGEMMKernelTest<TypedInputs, DeviceBackend>::SetUp(ProblemParams const&       probParams,
                                                            SolutionParams const&      solParams,
                                                            MemoryPageAlignment const& alignment)
{
    // Extract testing inputs
    problem                                   = createProblem(probParams);
    std::tie(library, adapter, requiredMatch) = solParams;
    memoryAlignment                           = alignment;

    DeviceBackend::setDefaultDevice(0);

    a_h.resize(problem.a().totalAllocatedElements());
    b_h.resize(problem.b().totalAllocatedElements());
    c_h.resize(problem.c().totalAllocatedElements());
    d_h.resize(problem.d().totalAllocatedElements());
    d_in_h.resize(problem.d().totalAllocatedElements());

    std::mt19937 rng;

    InitTensor(a_h.data(), problem.a(), RandomInt<AType>(), rng);
    InitTensor(b_h.data(), problem.b(), RandomAlternatingInt<BType>(), rng);
    InitTensor(c_h.data(), problem.c(), RandomInt<CType>(), rng);
    InitTensor(d_in_h.data(), problem.d(), RandomInt<DType>(), rng);

    d_ref_h = d_h;

    CopyTensor(d_ref_h.data(), c_h.data(), problem.d(), problem.c());

    constexpr size_t pageSize = 4 * 1024 * 1024;

    size_t aSize = RoundUpToMultiple(problem.a().totalAllocatedBytes(), pageSize);
    size_t bSize = RoundUpToMultiple(problem.b().totalAllocatedBytes(), pageSize);
    size_t cSize = RoundUpToMultiple(problem.c().totalAllocatedBytes(), pageSize);
    size_t dSize = RoundUpToMultiple(problem.d().totalAllocatedBytes(), pageSize);

    // Initialize device allocations and data
    DeviceBackend::malloc(a_d_alloc, aSize);
    DeviceBackend::malloc(b_d_alloc, bSize);
    DeviceBackend::malloc(c_d_alloc, cSize);
    DeviceBackend::malloc(d_d_alloc, dSize);

    if(alignment == MemoryPageAlignment::END)
    {
        // Align memory region as close as possible to end
        // of page, while respecting device alignment requirements.
        auto offsetAlignmentMask = ~(DeviceBackend::offsetAlignment() - 1);
        bufferOffsets = {(aSize - problem.a().totalAllocatedBytes()) & offsetAlignmentMask,
                         (bSize - problem.b().totalAllocatedBytes()) & offsetAlignmentMask,
                         (cSize - problem.c().totalAllocatedBytes()) & offsetAlignmentMask,
                         (dSize - problem.d().totalAllocatedBytes()) & offsetAlignmentMask};
    }
    else
    {
        bufferOffsets = {0, 0, 0, 0};
    }

    DeviceBackend::copyHostToDevice(
        a_d_alloc, bufferOffsets[0], a_h.data(), problem.a().totalAllocatedBytes());
    DeviceBackend::copyHostToDevice(
        b_d_alloc, bufferOffsets[1], b_h.data(), problem.b().totalAllocatedBytes());
    DeviceBackend::copyHostToDevice(
        c_d_alloc, bufferOffsets[2], c_h.data(), problem.c().totalAllocatedBytes());
    DeviceBackend::copyHostToDevice(
        d_d_alloc, bufferOffsets[3], d_in_h.data(), problem.d().totalAllocatedBytes());

    // Adjust data pointers to alignment boundaries
    inputs_d.a = DeviceBackend::dataPtr(a_d_alloc, bufferOffsets[0]);
    inputs_d.b = DeviceBackend::dataPtr(b_d_alloc, bufferOffsets[1]);
    inputs_d.c = DeviceBackend::dataPtr(c_d_alloc, bufferOffsets[2]);
    inputs_d.d = DeviceBackend::dataPtr(d_d_alloc, bufferOffsets[3]);

    // Initialize alpha
    inputs_d.alpha = RandomInt<AlphaType>()(rng);

    // Initialize beta
    if(problem.beta() == 1.0)
    {
        inputs_d.beta = static_cast<BetaType>(1.0);
    }
    else if(problem.beta() == 0.0)
    {
        inputs_d.beta = static_cast<BetaType>(0.0);
    }
    else
    {
        inputs_d.beta = RandomInt<BetaType>()(rng);
    }

    // Initialize inputs for CPU reference calcs
    inputs_h.a     = a_h.data();
    inputs_h.b     = b_h.data();
    inputs_h.c     = c_h.data();
    inputs_h.d     = d_ref_h.data();
    inputs_h.alpha = inputs_d.alpha;
    inputs_h.beta  = inputs_d.beta;

    // Initialize hardware
    hardware = DeviceBackend::getCurrentDevice();
    ASSERT_NE(hardware, nullptr);
}

template <typename TypedInputs, typename DeviceBackend>
void TypedGEMMKernelTest<TypedInputs, DeviceBackend>::OverrideAlpha(double val)
{
    inputs_h.alpha = inputs_d.alpha = static_cast<AlphaType>(val);
}

template <typename TypedInputs, typename DeviceBackend>
void TypedGEMMKernelTest<TypedInputs, DeviceBackend>::NullifyAPtr()
{
    inputs_h.a = inputs_d.a = nullptr;
}

template <typename TypedInputs, typename DeviceBackend>
void TypedGEMMKernelTest<TypedInputs, DeviceBackend>::NullifyBPtr()
{
    inputs_h.b = inputs_d.b = nullptr;
}

template <typename TypedInputs, typename DeviceBackend>
void TypedGEMMKernelTest<TypedInputs, DeviceBackend>::calcCPU()
{
    // Run reference CPU calc.
    auto key  = Tensile::hash_combine(problem, inputs_h.alpha, inputs_h.beta);
    auto iter = referenceCache.find(key);
    if(iter == referenceCache.end())
    {
        Client::SolveCPU(problem, inputs_h);
        referenceCache[key] = d_ref_h;
    }
    else
    {
        d_ref_h = iter->second;
    }
}

template <typename TypedInputs, typename DeviceBackend>
void TypedGEMMKernelTest<TypedInputs, DeviceBackend>::calcGPU()
{
    {
        ASSERT_NE(solution->problemPredicate, nullptr);
        EXPECT_EQ((*solution->problemPredicate)(problem), true);

        std::ostringstream msg;
        ASSERT_EQ(solution->problemPredicate->debugEval(problem, msg), true) << msg.str();
    }

    {
        ASSERT_NE(solution->hardwarePredicate, nullptr);
        EXPECT_EQ((*solution->hardwarePredicate)(*hardware), true);

        std::ostringstream msg;
        ASSERT_EQ(solution->hardwarePredicate->debugEval(*hardware, msg), true) << msg.str();
    }

    if(Debug::Instance().printPredicateEvaluation())
    {
        std::cout << "a: " << std::hex << inputs_d.a << ".."
                  << inputs_d.a + problem.a().totalAllocatedElements() << std::endl;
        std::cout << "b: " << std::hex << inputs_d.b << ".."
                  << inputs_d.b + problem.b().totalAllocatedElements() << std::endl;
        std::cout << "c: " << std::hex << inputs_d.c << ".."
                  << inputs_d.c + problem.c().totalAllocatedElements() << std::endl;
        std::cout << "d: " << std::hex << inputs_d.d << ".."
                  << inputs_d.d + problem.d().totalAllocatedElements() << std::endl;
    }

    // Make sure to keep the kernel args log if necessary (E.g. OpenCL requires logs to be kept to
    // is can iterate over them during invocation)
    solution->kernelArgsLog = DeviceBackend::kernelArgsLog();

    std::vector<KernelInvocation> result = solution->solve(problem, inputs_d, *hardware);

    adapter->launchKernels(result);

    DeviceBackend::copyDeviceToHost(
        d_h.data(), d_d_alloc, bufferOffsets[3], problem.d().totalAllocatedBytes());
}

template <typename TypedInputs, typename DeviceBackend>
void TypedGEMMKernelTest<TypedInputs, DeviceBackend>::TestBestSolution()
{
    if(Debug::Instance().printPredicateEvaluation())
    {
        std::cout << problem << std::endl << adapter << std::endl;
    }

    ASSERT_NE(library, nullptr);
    ASSERT_NE(adapter, nullptr);

    solution = library->findBestSolution(problem, *hardware);

    if(requiredMatch)
    {
        ASSERT_NE(solution, nullptr);
    }
    else
    {
        if(!solution)
        {
            return;
        }
    }

    calcCPU();
    calcGPU();
    expectEqual(d_h, d_ref_h);
}

template <typename TypedInputs, typename DeviceBackend>
void TypedGEMMKernelTest<TypedInputs, DeviceBackend>::TestAllSolutions()
{
    if(Debug::Instance().printPredicateEvaluation())
    {
        std::cout << problem << std::endl << adapter << std::endl;
    }

    ASSERT_NE(library, nullptr);
    ASSERT_NE(adapter, nullptr);

    auto solutions = library->findAllSolutions(problem, *hardware);

    if(requiredMatch)
    {
        ASSERT_GT(solutions.size(), 0);
    }

    calcCPU();

    for(auto const& testSolution : solutions)
    {
        solution = testSolution;
        calcGPU();
        expectEqual(d_h, d_ref_h);
    }
}

template <typename TypedInputs, typename DeviceBackend>
void TypedGEMMKernelTest<TypedInputs, DeviceBackend>::TearDown()
{
    DeviceBackend::free(a_d_alloc);
    DeviceBackend::free(b_d_alloc);
    DeviceBackend::free(c_d_alloc);
    DeviceBackend::free(d_d_alloc);

    DeviceBackend::deviceReset();

    a_h.clear();
    b_h.clear();
    c_h.clear();
    d_h.clear();
    d_in_h.clear();
}

template <typename TypedInputs, typename DeviceBackend>
inline std::string TypedGEMMKernelTest<TypedInputs, DeviceBackend>::ToString() const
{
    return std::string("TypedKernelShortCircuitTest<") + TypeInfo<AType>::Name() + ", "
           + TypeInfo<BType>::Name() + ", " + TypeInfo<CType>::Name() + ", "
           + TypeInfo<DType>::Name() + ", " + TypeInfo<AlphaType>::Name() + ", "
           + TypeInfo<BetaType>::Name() + ", " + std::string(">");
}

#endif // GEMM_KERNEL_TEST_IMPL_HPP
