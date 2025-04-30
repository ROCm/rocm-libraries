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
#ifndef GEMM_KERNEL_TEST_HPP
#define GEMM_KERNEL_TEST_HPP

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>

#include "TestData.hpp"
#include "TestUtils.hpp"

#include <iostream>
#include <unordered_map>

using namespace Tensile;

/* Test interface:
 * This is the driving interface
 * for setting up and invoking
 * Tensile GEMM type problems on a
 * particular device backend.
 * E.g. HIP, or OpenCL
 */
template <typename DeviceBackend>
struct GEMMKernelTest
{
    /* Setup test component types that will
     * stay constant for each Device Backend.
     */

    template <typename T>
    using BufferObj = typename DeviceBackend::template BufferObj<T>;

    using ContractionProblemGemm  = ::ContractionProblemGemm;
    using ContractionSolution = typename ContractionProblemGemm::Solution;

    enum class MemoryPageAlignment : int
    {
        BEGIN = 0,
        END   = 1
    };

    using SolutionAdapter = typename DeviceBackend::SolutionAdapter;
    using SolutionLibrary = ::SolutionLibrary<ContractionProblemGemm, ContractionSolution>;

    using ProblemParams = std::tuple<bool, //   transA
                                     bool, //   transB
                                     size_t, // m
                                     size_t, // n
                                     size_t, // k
                                     size_t, // lda
                                     size_t, // ldb
                                     size_t, // ldc
                                     double, // beta
                                     size_t>; // batchCount

    using SolutionParams = std::tuple<std::shared_ptr<SolutionLibrary>,
                                      std::shared_ptr<SolutionAdapter>,
                                      bool>; // is a solution required?

    static constexpr ProblemParams RandomGEMMParams
        = std::make_tuple(false, false, -1, -1, -1, -1, -1, -1, -1.0, -1);

    virtual void SetUp(ProblemParams const&, SolutionParams const&, MemoryPageAlignment const&) = 0;
    virtual void TestBestSolution()                                                             = 0;
    virtual void TestAllSolutions()                                                             = 0;
    virtual void TearDown()                                                                     = 0;
    virtual void OverrideAlpha(double)                                                          = 0;
    virtual void NullifyAPtr()                                                                  = 0;
    virtual void NullifyBPtr()                                                                  = 0;
    virtual std::string ToString() const                                                        = 0;
    virtual ~GEMMKernelTest() {}
};

template <typename DeviceBackend>
inline std::ostream& operator<<(std::ostream&                                         stream,
                                std::shared_ptr<GEMMKernelTest<DeviceBackend>> const& ptr);

/* Typed test implementation:
 * Setup and invocation of Tensile GEMM problems with
 * specific controls for input generation
 * on different device backends.
 */
template <typename TypedInputs, typename DeviceBackend>
struct TypedGEMMKernelTest : public GEMMKernelTest<DeviceBackend>
{
    // Extract base component configuration
    using Base = GEMMKernelTest<DeviceBackend>;
    template <typename T>
    using BufferObj           = typename Base::template BufferObj<T>;
    using ContractionProblemGemm  = typename Base::ContractionProblemGemm;
    using ContractionSolution = typename Base::ContractionSolution;
    using MemoryPageAlignment = typename Base::MemoryPageAlignment;
    using SolutionAdapter     = typename Base::SolutionAdapter;
    using SolutionLibrary     = typename Base::SolutionLibrary;

    // Extract testing params
    using SolutionParams = typename Base::SolutionParams;
    using ProblemParams  = typename Base::ProblemParams;

    // // Extract input types
    // using AType     = typename TypedInputs::AType;
    // using BType     = typename TypedInputs::BType;
    // using CType     = typename TypedInputs::CType;
    // using DType     = typename TypedInputs::DType;
    // using AlphaType = typename TypedInputs::AlphaType;
    // using BetaType  = typename TypedInputs::BetaType;

    // // Host data buffers
    // std::vector<AType> a_h;
    // std::vector<BType> b_h;
    // std::vector<CType> c_h;
    // std::vector<DType> d_h;
    // std::vector<DType> d_in_h;
    // std::vector<DType> d_ref_h;

    // Device data buffers
    // BufferObj<AType> a_d_alloc = nullptr;
    // BufferObj<BType> b_d_alloc = nullptr;
    // BufferObj<CType> c_d_alloc = nullptr;
    // BufferObj<DType> d_d_alloc = nullptr;

    std::array<size_t, 4> bufferOffsets = {0, 0, 0, 0};

    TypedInputs inputs_h;
    TypedInputs inputs_d;

    std::shared_ptr<Hardware> hardware;

    // Testing components
    ContractionProblemGemm                   problem;
    std::shared_ptr<SolutionLibrary>     library;
    std::shared_ptr<SolutionAdapter>     adapter;
    std::shared_ptr<ContractionSolution> solution;
    bool                                 requiredMatch;
    MemoryPageAlignment                  memoryAlignment;

    static std::unordered_map<size_t, std::vector<DType>> referenceCache;

    static ContractionProblemGemm createProblem(ProblemParams const& props);

    void SetUp(ProblemParams const&       probParams,
               SolutionParams const&      solParams,
               MemoryPageAlignment const& alignment) override;

    void OverrideAlpha(double val) override;
    void NullifyAPtr() override;
    void NullifyBPtr() override;

    void calcCPU();
    void calcGPU();

    void TestBestSolution() override;
    void TestAllSolutions() override;

    void TearDown() override;

    inline std::string ToString() const override;
};

#include "GEMMKernelTest_impl.hpp"

#endif // GEMM_KERNEL_TEST_HPP
