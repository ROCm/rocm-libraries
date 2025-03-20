/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#ifndef HIPTENSOR_CONTRACTION_SCALE_COMPLEX_HPP
#define HIPTENSOR_CONTRACTION_SCALE_COMPLEX_HPP

#include "../contraction_pack_util.hpp"
#include "common.hpp"
#include <hip/hip_complex.h>

namespace ck
{
    namespace tensor_operation
    {
        namespace device
        {

            using hiptensor::allocDevice;
            using hiptensor::ceilDiv;
            using hiptensor::DeviceDeleter;
            using hiptensor::elementsFromLengths;

            using Bilinear        = ck::tensor_operation::element_wise::Bilinear;
            using BilinearComplex = ck::tensor_operation::element_wise::BilinearComplex;
            using Scale           = ck::tensor_operation::element_wise::Scale;
            using ScaleComplex    = ck::tensor_operation::element_wise::ScaleComplex;

            // The following is a specialization class for bilinear contractions of complex types.
            // For complex types, the contraction can be decomposed into 4 simple bilinear contractions of
            // the complex element type.
            // The class implements a CK interface to wrap the 4 individual contraction operations and argument
            // handling internally.
            // Note: We are assuming that the data comes in as an Array of Structures (AOS) format in complex pairs.
            // The argument initialization portion decomposes this data into structure of arrays (SOA) where the
            // real and complex elements can be operated on separately.

            // Tensor Contraction:
            //   input : A
            //   input : B
            //   input : D0, D1, ...
            //   output : E
            //   C = a_op(A) * b_op(B)
            //   E = cde_op(C, D0, D1, ...)
            // Assume:
            //   A[M0, M1, M2, ..., K0, K1, K2, ...]
            //   B[N0, N1, N2, ..., K0, K1, K2, ...]
            //   D[M0, M1, M2, ..., N0, N1, N2, ...]
            //   E[M0, M1, M2, ..., N0, N1, N2, ...]
            template <index_t NumDimM,
                      index_t NumDimN,
                      index_t NumDimK,
                      typename ADataType,
                      typename BDataType,
                      typename AccDataType,
                      typename CShuffleDataType,
                      typename EDataType,
                      typename AElementwiseOperation,
                      typename BElementwiseOperation,
                      GemmSpecialization GemmSpec,
                      index_t            NumGemmKPrefetchStage,
                      index_t            BlockSize,
                      index_t            MPerBlock,
                      index_t            NPerBlock,
                      index_t            KPerBlock,
                      index_t            AK1,
                      index_t            BK1,
                      index_t            MPerXDL,
                      index_t            NPerXDL,
                      index_t            MXdlPerWave,
                      index_t            NXdlPerWave,
                      typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
                      typename ABlockTransferThreadClusterArrangeOrder,
                      typename ABlockTransferSrcAccessOrder,
                      index_t ABlockTransferSrcVectorDim,
                      index_t ABlockTransferSrcScalarPerVector,
                      index_t ABlockTransferDstScalarPerVector_AK1,
                      bool    ABlockLdsExtraM,
                      typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
                      typename BBlockTransferThreadClusterArrangeOrder,
                      typename BBlockTransferSrcAccessOrder,
                      index_t BBlockTransferSrcVectorDim,
                      index_t BBlockTransferSrcScalarPerVector,
                      index_t BBlockTransferDstScalarPerVector_BK1,
                      bool    BBlockLdsExtraN,
                      index_t CShuffleMXdlPerWavePerShuffle,
                      index_t CShuffleNXdlPerWavePerShuffle,
                      typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                      index_t CDEBlockTransferScalarPerVector_NPerBlock,
                      typename ComputeDataType,
                      LoopScheduler LoopSched>
            struct DeviceContractionMultipleD_Xdl_CShuffle<
                NumDimM,
                NumDimN,
                NumDimK,
                HIP_vector_type<ADataType, 2>,
                HIP_vector_type<BDataType, 2>,
                AccDataType,
                CShuffleDataType,
                ck::Tuple<>,
                HIP_vector_type<EDataType, 2>,
                AElementwiseOperation,
                BElementwiseOperation,
                ScaleComplex,
                GemmSpec,
                NumGemmKPrefetchStage,
                BlockSize,
                MPerBlock,
                NPerBlock,
                KPerBlock,
                AK1,
                BK1,
                MPerXDL,
                NPerXDL,
                MXdlPerWave,
                NXdlPerWave,
                ABlockTransferThreadClusterLengths_AK0_M_AK1,
                ABlockTransferThreadClusterArrangeOrder,
                ABlockTransferSrcAccessOrder,
                ABlockTransferSrcVectorDim,
                ABlockTransferSrcScalarPerVector,
                ABlockTransferDstScalarPerVector_AK1,
                ABlockLdsExtraM,
                BBlockTransferThreadClusterLengths_BK0_N_BK1,
                BBlockTransferThreadClusterArrangeOrder,
                BBlockTransferSrcAccessOrder,
                BBlockTransferSrcVectorDim,
                BBlockTransferSrcScalarPerVector,
                BBlockTransferDstScalarPerVector_BK1,
                BBlockLdsExtraN,
                CShuffleMXdlPerWavePerShuffle,
                CShuffleNXdlPerWavePerShuffle,
                CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                CDEBlockTransferScalarPerVector_NPerBlock,
                HIP_vector_type<ComputeDataType, 2>,
                LoopSched>

                : public DeviceContractionMultipleD<NumDimM,
                                                    NumDimN,
                                                    NumDimK,
                                                    HIP_vector_type<ADataType, 2>,
                                                    HIP_vector_type<BDataType, 2>,
                                                    ck::Tuple<>,
                                                    HIP_vector_type<EDataType, 2>,
                                                    AElementwiseOperation,
                                                    BElementwiseOperation,
                                                    ScaleComplex,
                                                    HIP_vector_type<ComputeDataType, 2>>
            {
                // Complex device Op
                using DeviceOp = DeviceContractionMultipleD_Xdl_CShuffle;

                // CDE Operations
                using ScaleCDEElementwiseOperation          = ScaleComplex;
                using DecompScaleCDEElementwiseOperation    = Scale;
                using BilinearCDEElementwiseOperation       = BilinearComplex;
                using DecompBilinearCDEElementwiseOperation = Bilinear;

                // Complex types given through the interface
                using ComplexA       = HIP_vector_type<ADataType, 2>;
                using ComplexB       = HIP_vector_type<BDataType, 2>;
                using ComplexDs      = HIP_vector_type<EDataType, 2>;
                using ComplexE       = HIP_vector_type<EDataType, 2>;
                using ComplexCompute = HIP_vector_type<ComputeDataType, 2>;

                // Internal functional types we will use to
                // decompose the complex types and operate on.
                using DecompA       = ADataType;
                using DecompB       = BDataType;
                using DecompDs      = EDataType;
                using DecompE       = EDataType;
                using DecompCompute = ComputeDataType;

                // For complex types, we need to make sure that all of the types are the same
                static_assert(std::is_same_v<DecompA, DecompB> && std::is_same_v<DecompB, DecompE>
                                  && std::is_same_v<DecompE, CShuffleDataType>
                                  && std::is_same_v<DecompE, DecompCompute>,
                              "Complex operations must have the same data type");

                static_assert(std::is_same_v<DecompA, float> || std::is_same_v<DecompA, double>,
                              "Complex operations only supported with single or double precision");

                static constexpr index_t NumDTensor = 0;

                // The internal operation that we will decompose the complex operations with.
                // For complex will be either float or double
                using ScaleDecompOp = DeviceContractionMultipleD_Xdl_CShuffle<
                    NumDimM,
                    NumDimN,
                    NumDimK,
                    DecompA,
                    DecompB,
                    AccDataType,
                    CShuffleDataType,
                    ck::Tuple<>,
                    DecompE,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    DecompScaleCDEElementwiseOperation,
                    GemmSpec,
                    NumGemmKPrefetchStage,
                    BlockSize,
                    MPerBlock,
                    NPerBlock,
                    KPerBlock,
                    AK1,
                    BK1,
                    MPerXDL,
                    NPerXDL,
                    MXdlPerWave,
                    NXdlPerWave,
                    ABlockTransferThreadClusterLengths_AK0_M_AK1,
                    ABlockTransferThreadClusterArrangeOrder,
                    ABlockTransferSrcAccessOrder,
                    ABlockTransferSrcVectorDim,
                    ABlockTransferSrcScalarPerVector,
                    ABlockTransferDstScalarPerVector_AK1,
                    ABlockLdsExtraM,
                    BBlockTransferThreadClusterLengths_BK0_N_BK1,
                    BBlockTransferThreadClusterArrangeOrder,
                    BBlockTransferSrcAccessOrder,
                    BBlockTransferSrcVectorDim,
                    BBlockTransferSrcScalarPerVector,
                    BBlockTransferDstScalarPerVector_BK1,
                    BBlockLdsExtraN,
                    CShuffleMXdlPerWavePerShuffle,
                    CShuffleNXdlPerWavePerShuffle,
                    CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                    CDEBlockTransferScalarPerVector_NPerBlock,
                    DecompCompute,
                    LoopSched>;

                // The internal operation that we will decompose the complex operations with.
                // For complex will be either float or double
                using BilinearDecompOp = DeviceContractionMultipleD_Xdl_CShuffle<
                    NumDimM,
                    NumDimN,
                    NumDimK,
                    DecompA,
                    DecompB,
                    AccDataType,
                    CShuffleDataType,
                    ck::Tuple<DecompDs>,
                    DecompE,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    DecompBilinearCDEElementwiseOperation,
                    GemmSpec,
                    NumGemmKPrefetchStage,
                    BlockSize,
                    MPerBlock,
                    NPerBlock,
                    KPerBlock,
                    AK1,
                    BK1,
                    MPerXDL,
                    NPerXDL,
                    MXdlPerWave,
                    NXdlPerWave,
                    ABlockTransferThreadClusterLengths_AK0_M_AK1,
                    ABlockTransferThreadClusterArrangeOrder,
                    ABlockTransferSrcAccessOrder,
                    ABlockTransferSrcVectorDim,
                    ABlockTransferSrcScalarPerVector,
                    ABlockTransferDstScalarPerVector_AK1,
                    ABlockLdsExtraM,
                    BBlockTransferThreadClusterLengths_BK0_N_BK1,
                    BBlockTransferThreadClusterArrangeOrder,
                    BBlockTransferSrcAccessOrder,
                    BBlockTransferSrcVectorDim,
                    BBlockTransferSrcScalarPerVector,
                    BBlockTransferDstScalarPerVector_BK1,
                    BBlockLdsExtraN,
                    CShuffleMXdlPerWavePerShuffle,
                    CShuffleNXdlPerWavePerShuffle,
                    CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                    CDEBlockTransferScalarPerVector_NPerBlock,
                    DecompCompute,
                    LoopSched>;

                // Argument
                struct Argument : public BaseArgument
                {
                    using ScaleDecompArgument    = typename ScaleDecompOp::Argument;
                    using BilinearDecompArgument = typename BilinearDecompOp::Argument;

                    Argument(Argument&& other)
                        : mScaleArgs(
                              {std::move(other.mScaleArgs[0]), std::move(other.mScaleArgs[1])})
                        , mBilinearArgs({std::move(other.mBilinearArgs[0]),
                                         std::move(other.mBilinearArgs[1])})
                    {
                    }

                    Argument& operator=(Argument&& other)
                    {
                        if(this != &other)
                        {
                            mScaleArgs[0]    = std::move(other.mScaleArgs[0]);
                            mScaleArgs[1]    = std::move(other.mScaleArgs[1]);
                            mBilinearArgs[0] = std::move(other.mBilinearArgs[0]);
                            mBilinearArgs[1] = std::move(other.mBilinearArgs[1]);
                        }
                        return *this;
                    }

                    Argument(const void*                                         p_a_grid,
                             const void*                                         p_b_grid,
                             std::array<const void*, NumDTensor>                 p_ds_grid,
                             void*                                               p_e_grid,
                             const std::vector<index_t>&                         a_ms_ks_lengths,
                             const std::vector<index_t>&                         a_ms_ks_strides,
                             const std::vector<index_t>&                         b_ns_ks_lengths,
                             const std::vector<index_t>&                         b_ns_ks_strides,
                             const std::array<std::vector<index_t>, NumDTensor>& ds_ms_ns_lengths,
                             const std::array<std::vector<index_t>, NumDTensor>& ds_ms_ns_strides,
                             const std::vector<index_t>&                         e_ms_ns_lengths,
                             const std::vector<index_t>&                         e_ms_ns_strides,
                             AElementwiseOperation                               a_element_op,
                             BElementwiseOperation                               b_element_op,
                             ScaleCDEElementwiseOperation                        cde_element_op)
                        : element_op(cde_element_op)
                    {
                        // Take the incoming arguments, treat them as complex.

                        // Allocate Real and Imaginary inputs
                        auto elementsA = elementsFromLengths(a_ms_ks_lengths);
                        auto elementsB = elementsFromLengths(b_ns_ks_lengths);
                        elementsE      = elementsFromLengths(e_ms_ns_lengths);

                        mA_real.reset(nullptr);
                        mA_imag.reset(nullptr);
                        mB_real.reset(nullptr);
                        mB_imag.reset(nullptr);
                        mE_real.reset(nullptr);
                        mE_imag.reset(nullptr);

                        mE_grid       = p_e_grid;
                        auto blockDim = dim3(1024);

                        auto decompGrid = [blockDim](auto&       out_r,
                                                     auto&       out_i,
                                                     auto const* input_grid,
                                                     uint32_t    elementCount) {
                            using DecompT = typename std::decay_t<decltype(out_r)>::element_type;
                            static_assert(std::is_same_v<
                                              DecompT,
                                              typename std::decay_t<decltype(out_i)>::element_type>,
                                          "r and i buffers must be same type");

                            if(input_grid != nullptr)
                            {
                                out_r = std::move(allocDevice<DecompT>(elementCount));
                                out_i = std::move(allocDevice<DecompT>(elementCount));

                                auto gridDim = dim3(ceilDiv(elementCount, blockDim.x));
                                hiptensor::unpack<<<gridDim, blockDim, 0>>>(
                                    input_grid, out_r.get(), out_i.get(), elementCount);
                            }
                        };

                        // Decompose the incoming data from AOS->SOA
                        decompGrid(mA_real, mA_imag, (const ComplexA*)p_a_grid, elementsA);
                        decompGrid(mB_real, mB_imag, (const ComplexB*)p_b_grid, elementsB);
                        decompGrid(mE_real, mE_imag, (const ComplexE*)p_e_grid, elementsE);

                        auto allocScaleArgs = [a_ms_ks_lengths,
                                               a_ms_ks_strides,
                                               b_ns_ks_lengths,
                                               b_ns_ks_strides,
                                               e_ms_ns_lengths,
                                               e_ms_ns_strides,
                                               a_element_op,
                                               b_element_op](auto&       out_e,
                                                             auto const& in_a,
                                                             auto const& in_b,
                                                             auto const& cde_element_op) {
                            return std::make_unique<ScaleDecompArgument>(
                                in_a.get(),
                                in_b.get(),
                                std::array<void const*, 0>{},
                                out_e.get(),
                                a_ms_ks_lengths,
                                a_ms_ks_strides,
                                b_ns_ks_lengths,
                                b_ns_ks_strides,
                                std::array<std::vector<index_t>, 0>{},
                                std::array<std::vector<index_t>, 0>{},
                                e_ms_ns_lengths,
                                e_ms_ns_strides,
                                a_element_op,
                                b_element_op,
                                cde_element_op);
                        };

                        auto allocBilinearArgs = [a_ms_ks_lengths,
                                                  a_ms_ks_strides,
                                                  b_ns_ks_lengths,
                                                  b_ns_ks_strides,
                                                  e_ms_ns_lengths,
                                                  e_ms_ns_strides,
                                                  a_element_op,
                                                  b_element_op](auto&       out_e,
                                                                auto const& in_a,
                                                                auto const& in_b,
                                                                auto const& in_d,
                                                                auto const& cde_element_op) {
                            return std::make_unique<BilinearDecompArgument>(
                                in_a.get(),
                                in_b.get(),
                                std::array<void const*, 1>{in_d.get()},
                                out_e.get(),
                                a_ms_ks_lengths,
                                a_ms_ks_strides,
                                b_ns_ks_lengths,
                                b_ns_ks_strides,
                                std::array<std::vector<index_t>, 1>{e_ms_ns_lengths},
                                std::array<std::vector<index_t>, 1>{e_ms_ns_strides},
                                e_ms_ns_lengths,
                                e_ms_ns_strides,
                                a_element_op,
                                b_element_op,
                                cde_element_op);
                        };

                        mScaleArgs[0] = allocScaleArgs(
                            mE_real, mA_real, mB_real, DecompScaleCDEElementwiseOperation{1.0f});
                        mBilinearArgs[0]
                            = allocBilinearArgs(mE_real,
                                                mA_imag,
                                                mB_imag,
                                                mE_real,
                                                DecompBilinearCDEElementwiseOperation{-1.0f, 1.0f});

                        mScaleArgs[1] = allocScaleArgs(
                            mE_imag, mA_real, mB_imag, DecompScaleCDEElementwiseOperation{1.0f});
                        mBilinearArgs[1]
                            = allocBilinearArgs(mE_imag,
                                                mA_imag,
                                                mB_real,
                                                mE_imag,
                                                DecompBilinearCDEElementwiseOperation{1.0f, 1.0f});
                    }

                    void Print() const
                    {
                        std::cout << "ScaleArgs0:" << std::endl;
                        mScaleArgs[0]->Print();
                        std::cout << "ScaleArgs1:" << std::endl;
                        mScaleArgs[1]->Print();
                        std::cout << "BilinearArgs0:" << std::endl;
                        mBilinearArgs[0]->Print();
                        std::cout << "BilinearArgs1:" << std::endl;
                        mBilinearArgs[1]->Print();
                    }

                    //  private:
                    // Each argument set for complex:
                    std::unique_ptr<ScaleDecompArgument>    mScaleArgs[2];
                    std::unique_ptr<BilinearDecompArgument> mBilinearArgs[2];

                    template <typename DataT>
                    using DeviceArray = std::unique_ptr<DataT, DeviceDeleter>;

                    // Manage extra memory for AOS->SOA
                    DeviceArray<DecompA> mA_real;
                    DeviceArray<DecompA> mA_imag;
                    DeviceArray<DecompB> mB_real;
                    DeviceArray<DecompB> mB_imag;
                    DeviceArray<DecompE> mE_real;
                    DeviceArray<DecompE> mE_imag;

                    ScaleCDEElementwiseOperation element_op;
                    void*                        mE_grid;
                    index_t                      elementsE;
                };

                // Invoker
                struct Invoker : public BaseInvoker
                {
                    using Argument = typename DeviceOp::Argument;

                    Invoker()
                        : mScaleInvoker(std::make_unique<typename ScaleDecompOp::Invoker>())
                        , mBilinearInvoker(std::make_unique<typename BilinearDecompOp::Invoker>())
                    {
                    }

                    Invoker(Invoker&& other)
                        : mScaleInvoker(std::move(other.mScaleInvoker))
                        , mBilinearInvoker(std::move(other.mBilinearInvoker))
                    {
                    }

                    Invoker& operator=(Invoker&& other)
                    {
                        if(this != &other)
                        {
                            mScaleInvoker    = std::move(other.mScaleInvoker);
                            mBilinearInvoker = std::move(other.mBilinearInvoker);
                        }
                        return *this;
                    }

                    float Run(const Argument&     arg,
                              const StreamConfig& stream_config = StreamConfig{})
                    {
                        auto r0 = mScaleInvoker->Run(arg.mScaleArgs[0].get(), stream_config);
                        auto r1 = mScaleInvoker->Run(arg.mScaleArgs[1].get(), stream_config);
                        auto r2 = mBilinearInvoker->Run(arg.mBilinearArgs[0].get(), stream_config);
                        auto r3 = mBilinearInvoker->Run(arg.mBilinearArgs[1].get(), stream_config);

                        if(arg.mE_grid != nullptr)
                        {
                            auto blockDim = dim3(1024);
                            auto gridDim  = dim3(ceilDiv(arg.elementsE, blockDim.x));

                            hiptensor::multiply<<<gridDim, blockDim, 0>>>(arg.mE_real.get(),
                                                                          arg.mE_imag.get(),
                                                                          ((ComplexE*)arg.mE_grid),
                                                                          arg.element_op.scale_,
                                                                          arg.elementsE);
                        }

                        return r0 + r1 + r2 + r3;
                    }

                    // polymorphic
                    float Run(const BaseArgument* p_arg,
                              const StreamConfig& stream_config = StreamConfig{}) override
                    {
                        return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
                    }

                    std::unique_ptr<typename ScaleDecompOp::Invoker>    mScaleInvoker;
                    std::unique_ptr<typename BilinearDecompOp::Invoker> mBilinearInvoker;
                };

                static bool IsSupportedArgument(const Argument& arg)
                {
                    return ScaleDecompOp::IsSupportedArgument(*(arg.mScaleArgs[0].get()))
                           && ScaleDecompOp::IsSupportedArgument(*(arg.mScaleArgs[1].get()))
                           && BilinearDecompOp::IsSupportedArgument(*(arg.mBilinearArgs[0].get()))
                           && BilinearDecompOp::IsSupportedArgument(*(arg.mBilinearArgs[1].get()));
                }

                // polymorphic
                bool IsSupportedArgument(const BaseArgument* p_arg) override
                {
                    return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
                }

                // polymorphic
                virtual void SetWorkSpacePointer(BaseArgument*       p_arg,
                                                 void*               p_workspace,
                                                 StreamConfig const& s
                                                 = StreamConfig{}) const override
                {
                    // Call the base, then fwd to each arg.
                    this->BaseOperator::SetWorkSpacePointer(p_arg, p_workspace, s);
                    auto* arg = dynamic_cast<Argument*>(p_arg);
                    this->BaseOperator::SetWorkSpacePointer(
                        arg->mScaleArgs[0].get(), p_workspace, s);
                    this->BaseOperator::SetWorkSpacePointer(
                        arg->mScaleArgs[1].get(), p_workspace, s);
                    this->BaseOperator::SetWorkSpacePointer(
                        arg->mBilinearArgs[0].get(), p_workspace, s);
                    this->BaseOperator::SetWorkSpacePointer(
                        arg->mBilinearArgs[1].get(), p_workspace, s);
                }

                static auto MakeArgument(
                    const void*                                         p_a,
                    const void*                                         p_b,
                    std::array<const void*, NumDTensor>                 p_ds,
                    void*                                               p_e,
                    const std::vector<index_t>&                         a_ms_ks_lengths,
                    const std::vector<index_t>&                         a_ms_ks_strides,
                    const std::vector<index_t>&                         b_ns_ks_lengths,
                    const std::vector<index_t>&                         b_ns_ks_strides,
                    const std::array<std::vector<index_t>, NumDTensor>& ds_ms_ns_lengths,
                    const std::array<std::vector<index_t>, NumDTensor>& ds_ms_ns_strides,
                    const std::vector<index_t>&                         e_ms_ns_lengths,
                    const std::vector<index_t>&                         e_ms_ns_strides,
                    AElementwiseOperation                               a_element_op,
                    BElementwiseOperation                               b_element_op,
                    ScaleCDEElementwiseOperation                        cde_element_op)
                {
                    return Argument{p_a,
                                    p_b,
                                    p_ds,
                                    p_e,
                                    a_ms_ks_lengths,
                                    a_ms_ks_strides,
                                    b_ns_ks_lengths,
                                    b_ns_ks_strides,
                                    ds_ms_ns_lengths,
                                    ds_ms_ns_strides,
                                    e_ms_ns_lengths,
                                    e_ms_ns_strides,
                                    a_element_op,
                                    b_element_op,
                                    cde_element_op};
                }

                static auto MakeInvoker()
                {
                    return Invoker{};
                }

                // polymorphic
                std::unique_ptr<BaseArgument> MakeArgumentPointer(
                    const void*                                         p_a,
                    const void*                                         p_b,
                    std::array<const void*, NumDTensor>                 p_ds,
                    void*                                               p_e,
                    const std::vector<index_t>&                         a_ms_ks_lengths,
                    const std::vector<index_t>&                         a_ms_ks_strides,
                    const std::vector<index_t>&                         b_ns_ks_lengths,
                    const std::vector<index_t>&                         b_ns_ks_strides,
                    const std::array<std::vector<index_t>, NumDTensor>& ds_ms_ns_lengths,
                    const std::array<std::vector<index_t>, NumDTensor>& ds_ms_ns_strides,
                    const std::vector<index_t>&                         e_ms_ns_lengths,
                    const std::vector<index_t>&                         e_ms_ns_strides,
                    AElementwiseOperation                               a_element_op,
                    BElementwiseOperation                               b_element_op,
                    ScaleCDEElementwiseOperation                        cde_element_op) override
                {
                    return std::make_unique<Argument>(p_a,
                                                      p_b,
                                                      p_ds,
                                                      p_e,
                                                      a_ms_ks_lengths,
                                                      a_ms_ks_strides,
                                                      b_ns_ks_lengths,
                                                      b_ns_ks_strides,
                                                      ds_ms_ns_lengths,
                                                      ds_ms_ns_strides,
                                                      e_ms_ns_lengths,
                                                      e_ms_ns_strides,
                                                      a_element_op,
                                                      b_element_op,
                                                      cde_element_op);
                }

                // polymorphic
                std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
                {
                    return std::make_unique<Invoker>(Invoker{});
                }

                // polymorphic
                std::string GetTypeString() const override
                {
                    auto str = std::stringstream();

                    // clang-format off
        str << "DeviceContractionMultipleD_Xdl_CShuffle"
            << "<"
            << NumDimM << ", "
            << NumDimN << ", "
            << NumDimK << ", "
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << ABlockTransferSrcVectorDim << ", "
            << BBlockTransferSrcVectorDim
            << ">";
                    // clang-format on

                    return str.str();
                }
            };

        } // namespace device
    } // namespace tensor_operation
} // namespace ck

#endif // HIPTENSOR_CONTRACTION_SCALE_COMPLEX_HPP
