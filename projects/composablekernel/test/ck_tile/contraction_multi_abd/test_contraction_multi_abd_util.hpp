// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <sstream>
#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/batched_contraction.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/host/reference/reference_batched_contraction.hpp"

using AddScale          = ck_tile::element_wise::AddScale;
using ElementWiseAddAdd = ck_tile::element_wise::MultiDAdd;
using MultiplyMultiply  = ck_tile::element_wise::MultiDMultiply;
using PassThrough       = ck_tile::element_wise::PassThrough;

struct AddDs
{
    template <typename E, typename C, typename... Ds>
    CK_TILE_HOST_DEVICE auto operator()(E& e, const C& c, const Ds&... ds) const -> void
    {
        const float x0_f =
            ck_tile::type_convert<float>(c) + (ck_tile::type_convert<float>(ds) + ...);
        e = ck_tile::type_convert<E>(x0_f);
    }
};

template <typename A0DataType_,
          typename B0DataType_,
          typename AccDataType_,
          typename EDataType_,
          typename D0DataType_>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeTypeAB =
        std::conditional_t<sizeof(A0DataType_) < sizeof(B0DataType_), A0DataType_, B0DataType_>;
    using ComputeType =
        std::conditional_t<sizeof(ComputeTypeAB) < sizeof(D0DataType_), ComputeTypeAB, D0DataType_>;

    const auto rtol = ck_tile::get_relative_threshold<ComputeType, EDataType_, AccDataType_>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, EDataType_, AccDataType_>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));

    const auto rtol_split_k =
        ck_tile::get_relative_threshold<EDataType_, EDataType_, EDataType_>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<EDataType_, EDataType_, EDataType_>(
        max_accumulated_value, kbatch);

    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

// Test fixture parameterized by a tuple of types
template <typename Tuple>
class TestCkTileContractionMultiABD : public ::testing::Test
{
    protected:
    using A0DataType        = std::tuple_element_t<0, Tuple>;
    using A1DataType        = std::tuple_element_t<1, Tuple>;
    using B0DataType        = std::tuple_element_t<2, Tuple>;
    using D0DataType        = std::tuple_element_t<3, Tuple>;
    using AccDataType       = std::tuple_element_t<4, Tuple>;
    using EDataType         = std::tuple_element_t<5, Tuple>;
    using AElementWiseFn    = std::tuple_element_t<6, Tuple>;
    using BElementWiseFn    = std::tuple_element_t<7, Tuple>;
    using CDEElementWiseFn  = std::tuple_element_t<8, Tuple>;
    using UseCshuffleEpilog = std::tuple_element_t<9, Tuple>;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    using AsDataType = ck_tile::tuple<A0DataType, A1DataType>;
    using BsDataType = ck_tile::tuple<B0DataType>;
    using DsDataType = ck_tile::tuple<D0DataType>;

    using AsLayout = ck_tile::tuple<Row, Row>;
    using BsLayout = ck_tile::tuple<Col>;
    using DsLayout = ck_tile::tuple<Row>;
    using ELayout  = Row;

    static constexpr ck_tile::index_t NumATensor = AsDataType::size();
    static constexpr ck_tile::index_t NumBTensor = BsDataType::size();
    static constexpr ck_tile::index_t NumDTensor = DsDataType::size();

    template <ck_tile::index_t NumDimG,
              ck_tile::index_t NumDimM,
              ck_tile::index_t NumDimN,
              ck_tile::index_t NumDimK>
    void invoke_kernel(
        const ck_tile::BatchedContractionMultiABDHostArgs<NumATensor, NumBTensor, NumDTensor>& args,
        const ck_tile::stream_config& s)
    {
        constexpr ck_tile::index_t M_Tile = 128;
        constexpr ck_tile::index_t N_Tile = 128;
        constexpr ck_tile::index_t K_Tile = 64;

        constexpr ck_tile::index_t M_Warp = 2;
        constexpr ck_tile::index_t N_Warp = 2;
        constexpr ck_tile::index_t K_Warp = 1;

#if CK_TILE_USE_WMMA
        constexpr ck_tile::index_t M_Warp_Tile = 16;
        constexpr ck_tile::index_t N_Warp_Tile = 16;
        constexpr ck_tile::index_t K_Warp_Tile = 16;
#else
        constexpr ck_tile::index_t M_Warp_Tile = 32;
        constexpr ck_tile::index_t N_Warp_Tile = 32;
        constexpr ck_tile::index_t K_Warp_Tile = 16;
#endif

        constexpr bool DoubleSmemBuffer = false;
        constexpr bool kPadM            = false;
        constexpr bool kPadN            = false;
        constexpr bool kPadK            = false;
        constexpr bool TransposeC       = false;

        constexpr int kBlockPerCu                         = 1;
        constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        constexpr ck_tile::index_t TileParitionerM01      = 4;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;
        using TilePartitioner = ck_tile::
            GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<kPadM,
                                                                     kPadN,
                                                                     kPadK,
                                                                     DoubleSmemBuffer,
                                                                     AsLayout,
                                                                     BsLayout,
                                                                     ELayout,
                                                                     TransposeC>;

        using Problem = ck_tile::BatchedContractionMultiABDProblem<AsDataType,
                                                                   BsDataType,
                                                                   DsDataType,
                                                                   EDataType,
                                                                   NumDimG,
                                                                   NumDimM,
                                                                   NumDimN,
                                                                   NumDimK>;

        constexpr auto scheduler = ck_tile::GemmPipelineScheduler::Intrawave;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<AsDataType,
                                                                           BsDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler,
                                                                           AElementWiseFn,
                                                                           BElementWiseFn>;

        using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

        using CShuffleGemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<AsDataType,
                                             BsDataType,
                                             DsDataType,
                                             AccDataType,
                                             EDataType,
                                             DsLayout,
                                             ELayout,
                                             CDEElementWiseFn,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC>>;

        using DefaultGemmEpilogue = ck_tile::DefaultGemm2DEpilogue<
            ck_tile::DefaultGemm2DEpilogueProblem<AsDataType,
                                                  BsDataType,
                                                  DsDataType,
                                                  AccDataType,
                                                  EDataType,
                                                  DsLayout,
                                                  ELayout,
                                                  CDEElementWiseFn,
                                                  TilePartitioner::MPerBlock,
                                                  TilePartitioner::NPerBlock,
                                                  kPadM,
                                                  kPadN,
                                                  M_Warp_Tile,
                                                  N_Warp_Tile,
                                                  K_Warp_Tile,
                                                  UniversalGemmProblem::TransposeC,
                                                  true>>;

        using GemmEpilogue =
            std::conditional_t<UseCshuffleEpilog::value, CShuffleGemmEpilogue, DefaultGemmEpilogue>;

        using Kernel = ck_tile::
            BatchedContractionMultiABDKernel<Problem, TilePartitioner, GemmPipeline, GemmEpilogue>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Kernel::GridSize(kargs);
        const dim3 blocks = Kernel::GetBlockSize();

        if(!Kernel::IsSupportedArguments(kargs))
        {
            throw std::runtime_error("Arguments not supported!");
        }

        ck_tile::ignore = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
    }

    public:
    template <ck_tile::index_t NumDimG,
              ck_tile::index_t NumDimM,
              ck_tile::index_t NumDimN,
              ck_tile::index_t NumDimK>
    bool Run(const std::vector<ck_tile::index_t>& G_dims,
             const std::vector<ck_tile::index_t>& M_dims,
             const std::vector<ck_tile::index_t>& N_dims,
             const std::vector<ck_tile::index_t>& K_dims)
    {
        auto calc_total = [](const std::vector<ck_tile::index_t>& dims) {
            ck_tile::index_t t = 1;
            for(auto d : dims)
                t *= d;
            return t;
        };

        auto concat = [](const std::vector<std::vector<ck_tile::index_t>>& vecs) {
            std::vector<ck_tile::index_t> r;
            for(const auto& v : vecs)
                r.insert(r.end(), v.begin(), v.end());
            return r;
        };

        ck_tile::index_t G_total = calc_total(G_dims);
        ck_tile::index_t M_total = calc_total(M_dims);
        ck_tile::index_t N_total = calc_total(N_dims);
        ck_tile::index_t K_total = calc_total(K_dims);

        std::vector<ck_tile::index_t> A_dims = concat({G_dims, M_dims, K_dims});
        std::vector<ck_tile::index_t> B_dims = concat({G_dims, N_dims, K_dims});
        std::vector<ck_tile::index_t> E_dims = concat({G_dims, M_dims, N_dims});

        const auto a_desc = ck_tile::HostTensorDescriptor(A_dims);
        const auto b_desc = ck_tile::HostTensorDescriptor(B_dims);
        const auto e_desc = ck_tile::HostTensorDescriptor(E_dims);

        ck_tile::HostTensor<A0DataType> a0_host(a_desc);
        ck_tile::HostTensor<A1DataType> a1_host(a_desc);
        ck_tile::HostTensor<B0DataType> b0_host(b_desc);
        ck_tile::HostTensor<D0DataType> d0_host(e_desc);
        ck_tile::HostTensor<EDataType> e_gpu_host(e_desc);

        ck_tile::FillUniformDistribution<A0DataType>{-1.f, 1.f}(a0_host);
        ck_tile::FillUniformDistribution<A1DataType>{-1.f, 1.f}(a1_host);
        ck_tile::FillUniformDistribution<B0DataType>{-1.f, 1.f}(b0_host);
        ck_tile::FillUniformDistribution<D0DataType>{-1.f, 1.f}(d0_host);

        ck_tile::DeviceMem a0_dev(a0_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem a1_dev(a1_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b0_dev(b0_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d0_dev(d0_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem e_dev(e_gpu_host.get_element_space_size_in_bytes());

        a0_dev.ToDevice(a0_host.data());
        a1_dev.ToDevice(a1_host.data());
        b0_dev.ToDevice(b0_host.data());
        d0_dev.ToDevice(d0_host.data());
        e_dev.SetZero();
        e_gpu_host.SetZero();

        auto convert_strides = [](const std::vector<std::size_t>& strides) {
            std::vector<ck_tile::index_t> converted(strides.size());
            std::copy(strides.begin(), strides.end(), converted.begin());
            return converted;
        };

        std::vector<ck_tile::index_t> A0_strides = convert_strides(a0_host.get_strides());
        std::vector<ck_tile::index_t> A1_strides = convert_strides(a1_host.get_strides());
        std::vector<ck_tile::index_t> B0_strides = convert_strides(b0_host.get_strides());
        std::vector<ck_tile::index_t> D0_strides = convert_strides(d0_host.get_strides());
        std::vector<ck_tile::index_t> E_strides  = convert_strides(e_gpu_host.get_strides());

        using HostArgs =
            ck_tile::BatchedContractionMultiABDHostArgs<NumATensor, NumBTensor, NumDTensor>;

        HostArgs host_args;
        host_args.as_ptr     = {a0_dev.GetDeviceBuffer(), a1_dev.GetDeviceBuffer()};
        host_args.bs_ptr     = {b0_dev.GetDeviceBuffer()};
        host_args.ds_ptr     = {d0_dev.GetDeviceBuffer()};
        host_args.e_ptr      = e_dev.GetDeviceBuffer();
        host_args.k_batch    = 1;
        host_args.As_dims    = {A_dims, A_dims};
        host_args.Bs_dims    = {B_dims};
        host_args.Ds_dims    = {E_dims};
        host_args.E_dims     = E_dims;
        host_args.As_strides = {A0_strides, A1_strides};
        host_args.Bs_strides = {B0_strides};
        host_args.Ds_strides = {D0_strides};
        host_args.E_strides  = E_strides;

        invoke_kernel<NumDimG, NumDimM, NumDimN, NumDimK>(host_args,
                                                          ck_tile::stream_config{nullptr, false});

        e_dev.FromDevice(e_gpu_host.data());

        ck_tile::HostTensor<EDataType> e_ref_host(e_desc);
        e_ref_host.SetZero();

        ck_tile::compute_reference_batched_contraction_multi_abd<AsDataType,
                                                                 BsDataType,
                                                                 DsDataType,
                                                                 AccDataType,
                                                                 EDataType,
                                                                 AElementWiseFn,
                                                                 BElementWiseFn,
                                                                 CDEElementWiseFn>(
            {a0_host, a1_host},
            {b0_host},
            {d0_host},
            e_ref_host,
            G_total,
            M_total,
            N_total,
            K_total,
            AElementWiseFn{},
            BElementWiseFn{},
            CDEElementWiseFn{},
            G_dims,
            M_dims,
            N_dims,
            K_dims);

        const float max_accumulated_value =
            *std::max_element(e_ref_host.mData.begin(), e_ref_host.mData.end());
        const auto rtol_atol =
            calculate_rtol_atol<A0DataType, B0DataType, AccDataType, EDataType, D0DataType>(
                K_total, 1, max_accumulated_value);

        bool pass = ck_tile::check_err(e_gpu_host,
                                       e_ref_host,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "G=" << G_total << " M=" << M_total << " N=" << N_total << " K=" << K_total
                  << " rtol=" << rtol_atol.at(ck_tile::number<0>{})
                  << " atol=" << rtol_atol.at(ck_tile::number<1>{}) << " "
                  << (pass ? "PASS" : "FAIL") << std::endl;

        return pass;
    }
};
