#pragma once

#include "ck_tile/core/numeric/numeric.hpp"
#include "ck_tile/host/arg_parser.hpp"
#include "ck_tile/host/check_err.hpp"
#include "ck_tile/host/fill.hpp"
#include "ck_tile/host/reference/reference_swiglu.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

#include "invoker.hpp"
#include "config.hpp"

#include <cstdlib>
#include <stdexcept>
#include <variant>
#include <format>

namespace ck_tile::swiglu_example {
using Row           = ck_tile::tensor_layout::gemm::RowMajor;
using Col           = ck_tile::tensor_layout::gemm::ColumnMajor;
using LayoutVariant = std::variant<Row, Col>;

inline auto string_to_layout(const std::string& layout) -> LayoutVariant
{
    if(layout == "R")
        return Row{};
    if(layout == "C")
        return Col{};
    throw std::runtime_error("Unsupported layout: " + layout);
};

template <typename CDataType>
bool verify_outputs(const ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
                    const ck_tile::HostTensor<CDataType>& c_m_n_ref,
                    const ck_tile::tuple<double, double>& rtol_atol,
                    const std::string& variant)
{
    bool pass = ck_tile::check_err(c_m_n_dev_result,
                                   c_m_n_ref,
                                   "Error: Incorrect results!",
                                   rtol_atol.at(ck_tile::number<0>{}),
                                   rtol_atol.at(ck_tile::number<1>{}));

    std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
              << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{}) << std::endl;
    std::cout << "The " << variant << " verification result is:" << (pass ? "correct" : "fail")
              << std::endl;
    return pass;
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

template <typename GemmConfig,
          typename Invoker,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float invoke_swiglu(ck_tile::DeviceMem& a_m_k_dev_buf,
                    ck_tile::DeviceMem& b0_k_n_dev_buf,
                    ck_tile::DeviceMem& b1_k_n_dev_buf,
                    ck_tile::DeviceMem& c_m_n_dev_buf,
                    ck_tile::index_t M,
                    ck_tile::index_t N,
                    ck_tile::index_t K,
                    ck_tile::index_t stride_A,
                    ck_tile::index_t stride_B,
                    ck_tile::index_t stride_C,
                    ck_tile::index_t kbatch,
                    int n_warmup,
                    int n_repeat,
                    bool persistent,
                    bool flush_cache,
                    int rotating_count)
{
    ck_tile::SwiGLUHostArgs args = {
        a_m_k_dev_buf.GetDeviceBuffer(),
        b0_k_n_dev_buf.GetDeviceBuffer(),
        b1_k_n_dev_buf.GetDeviceBuffer(),
        c_m_n_dev_buf.GetDeviceBuffer(),
        M,
        N,
        K,
        stride_A,
        stride_B,
        stride_B,
        stride_C,
        kbatch,
    };

    float ave_time = 0;
    ck_tile::stream_config stream_config{
        nullptr,
        true,
        1,
        n_warmup,
        n_repeat,
        true,
        flush_cache,
        rotating_count,
    };
    if(persistent)
    {
        ave_time = Invoker::template swiglu<GemmConfig,
                                            ADataType,
                                            BDataType,
                                            DsDataType,
                                            AccDataType,
                                            CDataType,
                                            ALayout,
                                            BLayout,
                                            DsLayout,
                                            CLayout,
                                            true,
                                            CDEElementWise>(args, stream_config);
    }
    else
    {
        ave_time = Invoker::template swiglu<GemmConfig,
                                            ADataType,
                                            BDataType,
                                            DsDataType,
                                            AccDataType,
                                            CDataType,
                                            ALayout,
                                            BLayout,
                                            DsLayout,
                                            CLayout,
                                            false,
                                            CDEElementWise>(args, stream_config);
    }

    return ave_time;
}

// ADataType_ and BDataType_ are original types (e.g., tf32_t for TF32 mode)
// They are passed through invoke_gemm to invoker for tf32 auto-detection
template <typename GemmConfig,
          typename Invoker,
          typename ADataType_,
          typename BDataType_ = ADataType_,
          typename CDataType_ = ADataType_,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_gemm_example_with_layouts(ck_tile::ArgParser& arg_parser,
                                  const ALayout a_layout                  = ALayout{},
                                  const BLayout b_layout                  = BLayout{},
                                  [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    // ADataTypeCompute: compute type (tf32_t for TF32 mode, used for warp gemm selection)
    // ADataTypeBuf: buffer/storage type (fp32 when tf32, from TypeConfig)
    using ADataTypeCompute = ADataType_;
    using BDataTypeCompute = BDataType_;

    // Use GemmTypeConfig to get actual data types for tensor operations
    // This handles tf32 -> float mapping for host tensors and device buffers
    using TypeConfig   = SwiGLUTypeConfig<ADataType_, BDataType_, CDataType_>;
    using ADataTypeBuf = typename TypeConfig::ADataType;
    using BDataTypeBuf = typename TypeConfig::BDataType;
    using CDataType    = typename TypeConfig::CDataType;
    using AccDataType  = typename TypeConfig::AccDataType;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");

    ck_tile::index_t kbatch      = arg_parser.get_int("split_k");
    int n_warmup                 = arg_parser.get_int("warmup");
    int n_repeat                 = arg_parser.get_int("repeat");
    ck_tile::index_t init_method = arg_parser.get_int("init");
    bool persistent              = arg_parser.get_int("persistent");
    bool flush_cache             = arg_parser.get_bool("flush_cache");
    int rotating_count           = arg_parser.get_int("rotating_count");

    const bool preshuffle = GemmConfig::Preshuffle;

    stride_A = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(M, N, stride_C, is_row_major(CLayout{}));

    auto a_desc = ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout));
    auto b_desc = ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout));
    auto c_desc = ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{}));
    ck_tile::HostTensor<ADataTypeBuf> a_m_k(a_desc);
    ck_tile::HostTensor<BDataTypeBuf> b0_k_n(b_desc);
    ck_tile::HostTensor<BDataTypeBuf> b1_k_n(b_desc);
    ck_tile::HostTensor<CDataType> c_m_n_dev_result(c_desc);

    if(init_method == 0)
    {
        ck_tile::FillUniformDistribution<ADataTypeBuf>{-2.f, 2.f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataTypeBuf>{-2.f, 2.f}(b0_k_n);
        ck_tile::FillUniformDistribution<BDataTypeBuf>{-2.f, 2.f}(b1_k_n);
    }
    else if(init_method == 1)
    {
        ck_tile::FillMonotonicSeq<ADataTypeBuf>{}(a_m_k);
        ck_tile::FillMonotonicSeq<BDataTypeBuf>{}(b0_k_n);
        ck_tile::FillMonotonicSeq<BDataTypeBuf>{}(b1_k_n);
    }
    else if(init_method == 2)
    {
        ck_tile::FillUniformDistribution<ADataTypeBuf>{1.f, 1.f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataTypeBuf>{1.f, 1.f}(b0_k_n);
        ck_tile::FillUniformDistribution<BDataTypeBuf>{1.f, 1.f}(b1_k_n);
    }
    else
    {
        a_m_k.SetZero();
        b0_k_n.SetZero();
        b1_k_n.SetZero();
    }

    if(!preshuffle && GemmConfig::UseStructuredSparsity)
    {
        if constexpr(GemmConfig::UseStructuredSparsity)
        {
            ck_tile::AdjustToStructuredSparsity<ADataTypeBuf>{}(a_m_k);
        }
    }

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b0_k_n_dev_buf(b0_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b1_k_n_dev_buf(b1_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

    static_assert(!GemmConfig::PermuteA, "Not implemented");

    if constexpr(preshuffle)
    {
        auto make_b_shuffle_host = [&](const auto& b_k_n) {
            if constexpr(GemmConfig::TiledMMAPermuteN)
            {
                std::cout << "Run with PermuteN" << std::endl;
                return ck_tile::shuffle_b_permuteN<GemmConfig>(b_k_n);
            }
            else
            {
                std::cout << "Run without PermuteN" << std::endl;
                return ck_tile::shuffle_b<GemmConfig>(b_k_n);
            }
        };
        ck_tile::HostTensor<BDataTypeBuf> b0_shuffle_host = make_b_shuffle_host(b0_k_n);
        ck_tile::HostTensor<BDataTypeBuf> b1_shuffle_host = make_b_shuffle_host(b1_k_n);

        // shuffled buffer B for device implementation
        if constexpr(std::is_same_v<BDataTypeBuf, ck_tile::pk_int4_t>)
        {
            ck_tile::permute_vectors_i4x4_b(b0_shuffle_host);
            ck_tile::permute_vectors_i4x4_b(b1_shuffle_host);
        }
        b0_k_n_dev_buf.ToDevice(b0_shuffle_host.data());
        b1_k_n_dev_buf.ToDevice(b1_shuffle_host.data());
    }
    else
    {
        if constexpr(std::is_same_v<BDataTypeBuf, ck_tile::pk_int4_t>)
        {
            // Permute vector pk_i4x4 data for device implementation
            ck_tile::HostTensor<BDataTypeBuf> b0_k_n_dev = b0_k_n;
            ck_tile::HostTensor<BDataTypeBuf> b1_k_n_dev = b1_k_n;
            if constexpr(GemmConfig::PermuteB)
            {
                permute_tensor_b<GemmConfig,
                                 decltype(b0_k_n_dev),
                                 ADataTypeBuf,
                                 BDataTypeBuf,
                                 AccDataType,
                                 CDataType,
                                 ALayout,
                                 BLayout,
                                 CLayout>(b0_k_n_dev);
                permute_tensor_b<GemmConfig,
                                 decltype(b1_k_n_dev),
                                 ADataTypeBuf,
                                 BDataTypeBuf,
                                 AccDataType,
                                 CDataType,
                                 ALayout,
                                 BLayout,
                                 CLayout>(b1_k_n_dev);
            }
            ck_tile::permute_vectors_i4x4_b(b0_k_n_dev);
            ck_tile::permute_vectors_i4x4_b(b1_k_n_dev);
            b0_k_n_dev_buf.ToDevice(b0_k_n_dev.data());
            b1_k_n_dev_buf.ToDevice(b1_k_n_dev.data());
        }
        else
        {
            if constexpr(GemmConfig::PermuteB)
            {
                std::cout << "Permute for this DataType is not implemented." << std::endl;
                return false;
            }
            b0_k_n_dev_buf.ToDevice(b0_k_n.data());
            b1_k_n_dev_buf.ToDevice(b1_k_n.data());
        }
    }

    a_m_k_dev_buf.ToDevice(a_m_k.data());
    c_m_n_dev_buf.SetZero();
    c_m_n_dev_result.SetZero();

    float ave_time = invoke_swiglu<GemmConfig,
                                   Invoker,
                                   ADataTypeCompute,
                                   BDataTypeCompute,
                                   ck_tile::tuple<>,
                                   AccDataType,
                                   CDataType,
                                   ALayout,
                                   BLayout,
                                   ck_tile::tuple<>,
                                   CLayout>(a_m_k_dev_buf,
                                            b0_k_n_dev_buf,
                                            b1_k_n_dev_buf,
                                            c_m_n_dev_buf,
                                            M,
                                            N,
                                            K,
                                            stride_A,
                                            stride_B,
                                            stride_C,
                                            kbatch,
                                            n_warmup,
                                            n_repeat,
                                            persistent,
                                            flush_cache,
                                            rotating_count);

    c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_byte =
        sizeof(ADataTypeBuf) * M * K / ck_tile::numeric_traits<ADataTypeBuf>::PackedSize +
        sizeof(BDataTypeBuf) * N * K / ck_tile::numeric_traits<BDataTypeBuf>::PackedSize +
        sizeof(CDataType) * M * N;
    flop *= 2;
    num_byte *= 2;

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run Gemm kernel with M=" << M << " N=" << N << " K=" << K
              << " StrideA=" << stride_A << " StrideB=" << stride_B << " StrideC=" << stride_C
              << " A_Layout=" << ALayout::name << " B_Layout =" << BLayout::name
              << " C_Layout=" << CLayout::name
              << " A_Type=" << ck_tile::DataTypeTraits<ADataTypeBuf>::name
              << " B_Type=" << ck_tile::DataTypeTraits<BDataTypeBuf>::name
              << " C_Type=" << ck_tile::DataTypeTraits<CDataType>::name
              << " StructuredSparsity=" << (GemmConfig::UseStructuredSparsity ? "on" : "off")
              << " Persistent=" << (persistent ? "on" : "off") << " : " << ave_time << " ms, "
              << tflops << " TFlops, " << gb_per_sec << " GB/s, " << std::endl;

    bool pass = true;

    // memory on host to store gpu reference result
    ck_tile::HostTensor<CDataType> c_m_n_ref(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
    c_m_n_ref.SetZero();

    if(arg_parser.get_int("v") == 1)
    {

        double max_accumulated_value{};
        ck_tile::swiglu_reference<AccDataType>(
            a_m_k, b0_k_n, b1_k_n, c_m_n_ref, {SwishMul{}}, &max_accumulated_value);

        const auto rtol_atol =
            calculate_rtol_atol<ADataTypeCompute, BDataTypeCompute, AccDataType, CDataType>(
                K, kbatch, max_accumulated_value);
        pass = verify_outputs(c_m_n_dev_result, c_m_n_ref, rtol_atol, "CPU");
    }
    return pass;
}

template <typename GemmConfig,
          typename ADataType,
          typename BDataType = ADataType,
          typename CDataType = ADataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_gemm_example_with_layouts_universal(ck_tile::ArgParser& arg_parser,
                                            const ALayout a_layout = ALayout{},
                                            const BLayout b_layout = BLayout{},
                                            const CLayout c_layout = CLayout{})
{
    using Invoker = UniversalInvoker;
    // using AccDataType = typename SwiGLUTypeConfig<ADataType, BDataType,
    // CDataType > ::AccDataType;

    // Normal path - delegate to shared implementation
    return run_gemm_example_with_layouts<GemmConfig, Invoker, ADataType, BDataType, CDataType>(
        arg_parser, a_layout, b_layout, c_layout);
}

// Universal GEMM-specific prec_type dispatcher that uses the wrapper
template <typename GemmConfig,
          typename APrecType,
          typename BPrecType = APrecType,
          typename CPrecType = APrecType>
int run_gemm_example_prec_type_universal(std::string a_layout,
                                         std::string b_layout,
                                         ck_tile::ArgParser& arg_parser)
{
    auto a_layout_variant = string_to_layout(a_layout);
    auto b_layout_variant = string_to_layout(b_layout);

    return std::visit(
        [&](auto a_layout_type, auto b_layout_type) -> int {
            return run_gemm_example_with_layouts_universal<GemmConfig,
                                                           APrecType,
                                                           BPrecType,
                                                           CPrecType>(
                arg_parser, a_layout_type, b_layout_type, Row{});
        },
        a_layout_variant,
        b_layout_variant);
}

template <template <class> typename Pipeline>
auto run_swiglu_example_pipeline(ck_tile::ArgParser& arg_parser) -> int
{
    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");

    if(data_type == "fp16")
        return run_gemm_example_prec_type_universal<Pipeline<ck_tile::fp16_t>, ck_tile::fp16_t>(
            a_layout, b_layout, arg_parser);

    // if(data_type == "bf16")
    //     return run_gemm_example_prec_type_universal<Pipeline<ck_tile::bf16_t>, ck_tile::bf16_t>(
    //         a_layout, b_layout, arg_parser);

    // if(data_type == "fp8")
    //     return run_gemm_example_prec_type_universal<Pipeline<ck_tile::fp8_t>, ck_tile::fp8_t>(
    //         a_layout, b_layout, arg_parser);

    // if(data_type == "pk_int4")
    //     return run_gemm_example_prec_type_universal<Pipeline<ck_tile::pk_int4_t>,
    //                                                 ck_tile::pk_int4_t>(
    //         a_layout, b_layout, arg_parser);
    // if(data_type == "tf32")
    //     return run_gemm_example_prec_type_universal<Pipeline<ck_tile::tf32_t>, ck_tile::tf32_t>(
    //         a_layout, b_layout, arg_parser);

    throw std::runtime_error(std::format("Invalid datatype! \n\t{}", data_type));
}

inline auto run_swiglu_example(ck_tile::ArgParser& arg_parser) -> int
{
    std::string pipeline = arg_parser.get_str("pipeline");

    if(pipeline == "v3")
        return run_swiglu_example_pipeline<SwiGLUConfigComputeV3>(arg_parser);
    if(pipeline == "v4")
        return run_swiglu_example_pipeline<SwiGLUConfigComputeV4>(arg_parser);
    // if(pipeline == "v6")
    //     return run_swiglu_example_pipeline<SwiGLUConfigComputeV6>(arg_parser);

    throw std::runtime_error(std::format("Invalid pipeline! \n\t{}", pipeline));
}
}; // namespace ck_tile::swiglu_example
