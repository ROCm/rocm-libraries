// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/reference/reference_warp_decode.hpp"
#include "ck_tile/ops/warp_decode.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ck_tile;

namespace {

struct WarpDecodeShape
{
    index_t B      = 2;
    index_t HIDDEN = 128;
    index_t INTER  = 256;
    index_t TOP_K  = 2;
    index_t E      = 8;
};

struct KernelPerfMetrics
{
    std::string name;
    double time_ms  = 0.0;
    double tflops   = 0.0;
    double gb_per_s = 0.0;
};

struct WarpDecodePerfMetrics
{
    KernelPerfMetrics gate_up;
    KernelPerfMetrics down_reduce;
    KernelPerfMetrics total;
};

bool IsVerbosePerfEnabled()
{
    if(const char* env = std::getenv("CK_WARP_DECODE_VERBOSE"))
    {
        const std::string value = env;
        return value == "1" || value == "true" || value == "TRUE" || value == "on" ||
               value == "ON";
    }

    return false;
}

template <typename DataType>
constexpr double GetElementBytes()
{
    return static_cast<double>(sizeof(DataType)) /
           static_cast<double>(numeric_traits<DataType>::PackedSize);
}

template <typename XDataType, typename WDataType, typename IntermediateDataType>
double CalculateGateUpFlops(const WarpDecodeShape& shape)
{
    constexpr double kActivationOps = 5.0;

    return static_cast<double>(shape.B) * static_cast<double>(shape.TOP_K) *
           static_cast<double>(shape.INTER) *
           (4.0 * static_cast<double>(shape.HIDDEN) + kActivationOps);
}

template <typename XDataType, typename WDataType, typename IntermediateDataType>
double CalculateGateUpBytes(const WarpDecodeShape& shape)
{
    const double num_outputs = static_cast<double>(shape.B) * static_cast<double>(shape.TOP_K) *
                               static_cast<double>(shape.INTER);
    const double x_bytes = num_outputs * static_cast<double>(shape.HIDDEN) *
                           GetElementBytes<XDataType>();
    const double w_bytes = num_outputs * static_cast<double>(shape.HIDDEN) *
                           GetElementBytes<WDataType>() * 2.0;
    const double router_bytes = num_outputs * static_cast<double>(sizeof(int32_t));
    const double intermediate_bytes =
        num_outputs * GetElementBytes<IntermediateDataType>();

    return x_bytes + w_bytes + router_bytes + intermediate_bytes;
}

template <typename IntermediateDataType, typename WDataType, typename YDataType>
double CalculateDownReduceFlops(const WarpDecodeShape& shape)
{
    const double num_outputs =
        static_cast<double>(shape.B) * static_cast<double>(shape.HIDDEN);
    const double work_per_output = static_cast<double>(shape.TOP_K) *
                                   static_cast<double>(shape.INTER) * 3.0;

    return num_outputs * work_per_output;
}

template <typename IntermediateDataType, typename WDataType, typename YDataType>
double CalculateDownReduceBytes(const WarpDecodeShape& shape)
{
    const double num_outputs =
        static_cast<double>(shape.B) * static_cast<double>(shape.HIDDEN);
    const double inter_bytes = num_outputs * static_cast<double>(shape.TOP_K) *
                               static_cast<double>(shape.INTER) *
                               GetElementBytes<IntermediateDataType>();
    const double weight_bytes = num_outputs * static_cast<double>(shape.TOP_K) *
                                static_cast<double>(shape.INTER) *
                                GetElementBytes<WDataType>();
    const double router_bytes =
        num_outputs * static_cast<double>(shape.TOP_K) * static_cast<double>(sizeof(int32_t));
    const double router_weight_bytes =
        num_outputs * static_cast<double>(shape.TOP_K) * static_cast<double>(sizeof(float));
    const double output_bytes = num_outputs * GetElementBytes<YDataType>();

    return inter_bytes + weight_bytes + router_bytes + router_weight_bytes + output_bytes;
}

KernelPerfMetrics MakeKernelPerfMetrics(const std::string& name,
                                        double time_ms,
                                        double flops,
                                        double bytes)
{
    KernelPerfMetrics metrics;
    metrics.name    = name;
    metrics.time_ms = time_ms;

    if(time_ms > 0.0)
    {
        metrics.tflops   = flops / (time_ms * 1.0e9);
        metrics.gb_per_s = bytes / (time_ms * 1.0e6);
    }

    return metrics;
}

void ReportPerfMetrics(const std::string& test_name, const WarpDecodePerfMetrics& metrics)
{
    auto old_flags     = std::cout.flags();
    auto old_precision = std::cout.precision();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[ PERF ] " << test_name << '\n';
    std::cout << "  " << metrics.gate_up.name << ": " << metrics.gate_up.time_ms
              << " ms, " << metrics.gate_up.tflops << " TFLOP/s, "
              << metrics.gate_up.gb_per_s << " GB/s\n";
    std::cout << "  " << metrics.down_reduce.name << ": " << metrics.down_reduce.time_ms
              << " ms, " << metrics.down_reduce.tflops << " TFLOP/s, "
              << metrics.down_reduce.gb_per_s << " GB/s\n";
    std::cout << "  " << metrics.total.name << ": " << metrics.total.time_ms << " ms, "
              << metrics.total.tflops << " TFLOP/s, " << metrics.total.gb_per_s << " GB/s\n";

    std::cout.flags(old_flags);
    std::cout.precision(old_precision);
}

template <typename DataType>
void FillRandom(HostTensor<DataType>& tensor, float min_val, float max_val, unsigned seed = 42)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);

    for(index_t i = 0; i < static_cast<index_t>(tensor.get_element_space_size()); ++i)
    {
        tensor.mData[i] = type_convert<DataType>(dist(gen));
    }
}

template <>
void FillRandom<pk_fp4_t>(HostTensor<pk_fp4_t>& tensor, float, float, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 15);

    for(index_t i = 0; i < static_cast<index_t>(tensor.get_element_space_size()); ++i)
    {
        uint8_t lo = static_cast<uint8_t>(dist(gen));
        uint8_t hi = static_cast<uint8_t>(dist(gen));
        tensor.mData[i] = pk_fp4_t(static_cast<uint8_t>((hi << 4) | (lo & 0x0F)));
    }
}

template <typename ScaleDataType>
void FillScaleRandom(std::vector<ScaleDataType>& buf, std::size_t count, unsigned seed = 123)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.5f, 2.0f);
    buf.resize(count);
    for(std::size_t i = 0; i < count; ++i)
    {
        buf[i] = type_convert<ScaleDataType>(dist(gen));
    }
}

template <>
void FillScaleRandom<e8m0_t>(std::vector<e8m0_t>& buf, std::size_t count, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(124, 130);
    buf.resize(count);
    for(std::size_t i = 0; i < count; ++i)
    {
        buf[i] = e8m0_t(static_cast<uint8_t>(dist(gen)));
    }
}

void FillRouterData(HostTensor<int32_t>& router_ids,
                    HostTensor<float>& router_wts,
                    index_t B,
                    index_t TOP_K,
                    index_t E)
{
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> int_dist(0, E - 1);
    std::uniform_real_distribution<float> float_dist(0.1f, 1.0f);

    for(index_t b = 0; b < B; ++b)
    {
        float sum = 0.0f;
        for(index_t k = 0; k < TOP_K; ++k)
        {
            router_ids(b, k) = int_dist(gen);
            const float w    = float_dist(gen);
            router_wts(b, k) = w;
            sum += w;
        }

        for(index_t k = 0; k < TOP_K; ++k)
        {
            router_wts(b, k) /= sum;
        }
    }
}

template <typename XDataType,
          typename WDataType,
          typename IntermediateDataType,
          typename YDataType,
          typename ComputeDataType,
          typename XScaleDataType,
          typename WScaleDataType,
          typename XScaleLayout,
          typename WScaleLayout,
          index_t kVector_ = 1>
::testing::AssertionResult RunPositiveCase(const std::string& test_name,
                                           float atol,
                                           const WarpDecodeShape& shape = {})
{
    constexpr bool is_fp8_x   = std::is_same_v<XDataType, fp8_t>;
    constexpr bool is_fp8_w   = std::is_same_v<WDataType, fp8_t>;
    constexpr bool is_mxfp4_w = std::is_same_v<WDataType, pk_fp4_t>;
    constexpr index_t kVector = is_mxfp4_w ? (kVector_ == 1 ? 2 : kVector_) : kVector_;

    const float x_range = is_fp8_x ? 0.5f : 1.0f;
    const float w_range = is_fp8_w ? 0.25f : 1.0f;

    HostTensor<XDataType> x({shape.B, shape.HIDDEN});
    HostTensor<WDataType> w_gate({shape.E, shape.INTER, shape.HIDDEN});
    HostTensor<WDataType> w_up({shape.E, shape.INTER, shape.HIDDEN});
    HostTensor<WDataType> w_down({shape.E, shape.HIDDEN, shape.INTER});
    HostTensor<int32_t> router_ids({shape.B, shape.TOP_K});
    HostTensor<float> router_wts({shape.B, shape.TOP_K});

    HostTensor<IntermediateDataType> intermediate_host({shape.B, shape.TOP_K, shape.INTER});
    HostTensor<IntermediateDataType> intermediate_dev({shape.B, shape.TOP_K, shape.INTER});
    HostTensor<YDataType> y_host({shape.B, shape.HIDDEN});
    HostTensor<YDataType> y_dev({shape.B, shape.HIDDEN});

    FillRandom(x, -x_range, x_range, 42);
    FillRandom(w_gate, -w_range, w_range, 43);
    FillRandom(w_up, -w_range, w_range, 44);
    FillRandom(w_down, -w_range, w_range, 45);
    FillRouterData(router_ids, router_wts, shape.B, shape.TOP_K, shape.E);

    using reference::RefScaleMode;
    constexpr bool x_per_tensor = std::is_same_v<XScaleLayout, WarpDecodeScaleLayout::PerTensor>;
    constexpr bool x_per_token  = std::is_same_v<XScaleLayout, WarpDecodeScaleLayout::PerToken>;
    constexpr bool w_per_tensor = std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerTensor>;
    constexpr bool w_per_token  = std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerToken>;
    constexpr bool w_block2d    = ScaleLayoutTraits<WScaleLayout>::is_block2d;

    constexpr index_t w_block_n = [] {
        if constexpr(w_block2d)
        {
            return ScaleLayoutTraits<WScaleLayout>::block_n;
        }
        else
        {
            return index_t{1};
        }
    }();

    constexpr index_t w_block_k = [] {
        if constexpr(w_block2d)
        {
            return ScaleLayoutTraits<WScaleLayout>::block_k;
        }
        else
        {
            return index_t{1};
        }
    }();

    constexpr RefScaleMode ref_x_mode =
        x_per_token ? RefScaleMode::PerToken :
        x_per_tensor ? RefScaleMode::PerTensor : RefScaleMode::None;
    constexpr RefScaleMode ref_w_mode =
        w_block2d ? RefScaleMode::Block2D :
        w_per_token ? RefScaleMode::PerToken :
        w_per_tensor ? RefScaleMode::PerTensor : RefScaleMode::None;

    std::vector<XScaleDataType> x_scale_buf;
    const XScaleDataType* p_x_scale = nullptr;
    if constexpr(x_per_tensor)
    {
        FillScaleRandom(x_scale_buf, 1, 100);
        p_x_scale = x_scale_buf.data();
    }
    else if constexpr(x_per_token)
    {
        FillScaleRandom(x_scale_buf, shape.B, 101);
        p_x_scale = x_scale_buf.data();
    }

    std::vector<WScaleDataType> w_gate_scale_buf;
    std::vector<WScaleDataType> w_up_scale_buf;
    std::vector<WScaleDataType> w_down_scale_buf;
    const WScaleDataType* p_w_gate_scale = nullptr;
    const WScaleDataType* p_w_up_scale   = nullptr;
    const WScaleDataType* p_w_down_scale = nullptr;

    if constexpr(w_per_tensor)
    {
        FillScaleRandom(w_gate_scale_buf, 1, 200);
        FillScaleRandom(w_up_scale_buf, 1, 201);
        FillScaleRandom(w_down_scale_buf, 1, 202);
        p_w_gate_scale = w_gate_scale_buf.data();
        p_w_up_scale   = w_up_scale_buf.data();
        p_w_down_scale = w_down_scale_buf.data();
    }
    else if constexpr(w_per_token)
    {
        FillScaleRandom(w_gate_scale_buf, shape.E * shape.INTER, 200);
        FillScaleRandom(w_up_scale_buf, shape.E * shape.INTER, 201);
        FillScaleRandom(w_down_scale_buf, shape.E * shape.HIDDEN, 202);
        p_w_gate_scale = w_gate_scale_buf.data();
        p_w_up_scale   = w_up_scale_buf.data();
        p_w_down_scale = w_down_scale_buf.data();
    }
    else if constexpr(w_block2d)
    {
        const std::size_t gate_up_scale_count =
            (shape.E * shape.INTER / w_block_n) * (shape.HIDDEN / w_block_k);
        const std::size_t down_scale_count =
            (shape.E * shape.HIDDEN / w_block_n) * (shape.INTER / w_block_k);
        FillScaleRandom(w_gate_scale_buf, gate_up_scale_count, 200);
        FillScaleRandom(w_up_scale_buf, gate_up_scale_count, 201);
        FillScaleRandom(w_down_scale_buf, down_scale_count, 202);
        p_w_gate_scale = w_gate_scale_buf.data();
        p_w_up_scale   = w_up_scale_buf.data();
        p_w_down_scale = w_down_scale_buf.data();
    }

    reference::reference_warp_decode_gate_up<XDataType,
                                             WDataType,
                                             ComputeDataType,
                                             IntermediateDataType,
                                             XScaleDataType,
                                             WScaleDataType>(x,
                                                             w_gate,
                                                             w_up,
                                                             router_ids,
                                                             intermediate_host,
                                                             p_x_scale,
                                                             p_w_gate_scale,
                                                             p_w_up_scale,
                                                             ref_x_mode,
                                                             ref_w_mode,
                                                             w_block_n,
                                                             w_block_k);

    reference::reference_warp_decode_down_reduce<IntermediateDataType,
                                                 WDataType,
                                                 ComputeDataType,
                                                 YDataType,
                                                 WScaleDataType>(intermediate_host,
                                                                 w_down,
                                                                 router_ids,
                                                                 router_wts,
                                                                 y_host,
                                                                 p_w_down_scale,
                                                                 ref_w_mode,
                                                                 w_block_n,
                                                                 w_block_k);

    DeviceMem x_buf(x.get_element_space_size_in_bytes());
    DeviceMem w_gate_buf(w_gate.get_element_space_size_in_bytes());
    DeviceMem w_up_buf(w_up.get_element_space_size_in_bytes());
    DeviceMem w_down_buf(w_down.get_element_space_size_in_bytes());
    DeviceMem router_ids_buf(router_ids.get_element_space_size_in_bytes());
    DeviceMem router_wts_buf(router_wts.get_element_space_size_in_bytes());
    DeviceMem inter_buf(intermediate_dev.get_element_space_size_in_bytes());
    DeviceMem y_buf(y_dev.get_element_space_size_in_bytes());

    x_buf.ToDevice(x.mData.data());
    w_gate_buf.ToDevice(w_gate.mData.data());
    w_up_buf.ToDevice(w_up.mData.data());
    w_down_buf.ToDevice(w_down.mData.data());
    router_ids_buf.ToDevice(router_ids.mData.data());
    router_wts_buf.ToDevice(router_wts.mData.data());

    DeviceMem x_scale_dbuf(x_scale_buf.size() * sizeof(XScaleDataType));
    DeviceMem w_gate_scale_dbuf(w_gate_scale_buf.size() * sizeof(WScaleDataType));
    DeviceMem w_up_scale_dbuf(w_up_scale_buf.size() * sizeof(WScaleDataType));
    DeviceMem w_down_scale_dbuf(w_down_scale_buf.size() * sizeof(WScaleDataType));

    void* p_x_scale_dev      = nullptr;
    void* p_w_gate_scale_dev = nullptr;
    void* p_w_up_scale_dev   = nullptr;
    void* p_w_down_scale_dev = nullptr;

    if(!x_scale_buf.empty())
    {
        x_scale_dbuf.ToDevice(x_scale_buf.data());
        p_x_scale_dev = x_scale_dbuf.GetDeviceBuffer();
    }
    if(!w_gate_scale_buf.empty())
    {
        w_gate_scale_dbuf.ToDevice(w_gate_scale_buf.data());
        p_w_gate_scale_dev = w_gate_scale_dbuf.GetDeviceBuffer();
    }
    if(!w_up_scale_buf.empty())
    {
        w_up_scale_dbuf.ToDevice(w_up_scale_buf.data());
        p_w_up_scale_dev = w_up_scale_dbuf.GetDeviceBuffer();
    }
    if(!w_down_scale_buf.empty())
    {
        w_down_scale_dbuf.ToDevice(w_down_scale_buf.data());
        p_w_down_scale_dev = w_down_scale_dbuf.GetDeviceBuffer();
    }

    using GateUpProblem = WarpDecodeGateUpProblem<XDataType,
                                                  WDataType,
                                                  ComputeDataType,
                                                  IntermediateDataType,
                                                  XScaleDataType,
                                                  WScaleDataType,
                                                  XScaleLayout,
                                                  WScaleLayout,
                                                  ck_tile::element_wise::Silu,
                                                  kVector>;
    using Policy       = WarpDecodePolicy;
    using GateUpKernel = WarpDecodeGateUpKernel<GateUpProblem, Policy>;

    typename GateUpKernel::Kargs gate_up_args{x_buf.GetDeviceBuffer(),
                                              p_x_scale_dev,
                                              w_gate_buf.GetDeviceBuffer(),
                                              p_w_gate_scale_dev,
                                              w_up_buf.GetDeviceBuffer(),
                                              p_w_up_scale_dev,
                                              static_cast<int32_t*>(router_ids_buf.GetDeviceBuffer()),
                                              inter_buf.GetDeviceBuffer(),
                                              shape.B,
                                              shape.HIDDEN,
                                              shape.INTER,
                                              shape.TOP_K,
                                              shape.E,
                                              shape.HIDDEN,
                                              shape.HIDDEN,
                                              shape.HIDDEN,
                                              shape.INTER};

    if(!GateUpKernel::IsSupportedArgument(gate_up_args))
    {
        return ::testing::AssertionFailure()
               << test_name << ": gate/up Kargs unexpectedly rejected by IsSupportedArgument().";
    }

    using DownProblem = WarpDecodeDownReduceProblem<IntermediateDataType,
                                                    WDataType,
                                                    ComputeDataType,
                                                    YDataType,
                                                    WScaleDataType,
                                                    WScaleLayout,
                                                    kVector>;
    using DownKernel = WarpDecodeDownReduceKernel<DownProblem, Policy>;

    typename DownKernel::Kargs down_args{inter_buf.GetDeviceBuffer(),
                                         w_down_buf.GetDeviceBuffer(),
                                         p_w_down_scale_dev,
                                         static_cast<int32_t*>(router_ids_buf.GetDeviceBuffer()),
                                         static_cast<float*>(router_wts_buf.GetDeviceBuffer()),
                                         y_buf.GetDeviceBuffer(),
                                         shape.B,
                                         shape.HIDDEN,
                                         shape.INTER,
                                         shape.TOP_K,
                                         shape.E,
                                         shape.INTER,
                                         shape.INTER,
                                         shape.HIDDEN};

    if(!DownKernel::IsSupportedArgument(down_args))
    {
        return ::testing::AssertionFailure()
               << test_name << ": down/reduce Kargs unexpectedly rejected by IsSupportedArgument().";
    }

    const bool verbose_perf = IsVerbosePerfEnabled();
    const auto s            = stream_config{nullptr, verbose_perf};
    const float gate_up_ms  = launch_warp_decode_gate_up<GateUpKernel>(gate_up_args, s);
    const float down_ms     = launch_warp_decode_down_reduce<DownKernel>(down_args, s);

    inter_buf.FromDevice(intermediate_dev.mData.data());
    y_buf.FromDevice(y_dev.mData.data());

    float inter_max_diff = 0.0f;
    for(index_t i = 0; i < static_cast<index_t>(intermediate_host.get_element_space_size()); ++i)
    {
        const float host_val = type_convert<float>(intermediate_host.mData[i]);
        const float dev_val  = type_convert<float>(intermediate_dev.mData[i]);
        const float diff     = std::abs(host_val - dev_val);
        inter_max_diff       = std::max(inter_max_diff, diff);
        if(diff > atol)
        {
            return ::testing::AssertionFailure()
                   << test_name << ": intermediate mismatch at " << i << " host=" << host_val
                   << " dev=" << dev_val << " diff=" << diff << " atol=" << atol;
        }
    }

    float y_max_diff = 0.0f;
    for(index_t i = 0; i < static_cast<index_t>(y_host.get_element_space_size()); ++i)
    {
        const float host_val = type_convert<float>(y_host.mData[i]);
        const float dev_val  = type_convert<float>(y_dev.mData[i]);
        const float diff     = std::abs(host_val - dev_val);
        y_max_diff           = std::max(y_max_diff, diff);
        if(diff > atol)
        {
            return ::testing::AssertionFailure()
                   << test_name << ": output mismatch at " << i << " host=" << host_val
                   << " dev=" << dev_val << " diff=" << diff << " atol=" << atol;
        }
    }

    std::ostringstream oss;
    oss << test_name << " passed with inter_diff=" << inter_max_diff << " y_diff=" << y_max_diff
        << " atol=" << atol;

    if(verbose_perf)
    {
        WarpDecodePerfMetrics perf_metrics;
        const double gate_up_flops =
            CalculateGateUpFlops<XDataType, WDataType, IntermediateDataType>(shape);
        const double gate_up_bytes =
            CalculateGateUpBytes<XDataType, WDataType, IntermediateDataType>(shape);
        const double down_flops =
            CalculateDownReduceFlops<IntermediateDataType, WDataType, YDataType>(shape);
        const double down_bytes =
            CalculateDownReduceBytes<IntermediateDataType, WDataType, YDataType>(shape);

        perf_metrics.gate_up =
            MakeKernelPerfMetrics("gate_up", gate_up_ms, gate_up_flops, gate_up_bytes);
        perf_metrics.down_reduce =
            MakeKernelPerfMetrics("down_reduce", down_ms, down_flops, down_bytes);
        perf_metrics.total = MakeKernelPerfMetrics(
            "total", gate_up_ms + down_ms, gate_up_flops + down_flops, gate_up_bytes + down_bytes);

        ReportPerfMetrics(test_name, perf_metrics);
    }

    return ::testing::AssertionSuccess() << oss.str();
}

template <typename GateUpKernel>
typename GateUpKernel::Kargs MakeValidGateUpArgs()
{
    static int x_storage            = 0;
    static int x_scale_storage      = 0;
    static int w_gate_storage       = 0;
    static int w_gate_scale_storage = 0;
    static int w_up_storage         = 0;
    static int w_up_scale_storage   = 0;
    static int32_t router_ids[4]    = {0, 1, 1, 0};
    static int intermediate_storage = 0;

    return typename GateUpKernel::Kargs{&x_storage,
                                        &x_scale_storage,
                                        &w_gate_storage,
                                        &w_gate_scale_storage,
                                        &w_up_storage,
                                        &w_up_scale_storage,
                                        router_ids,
                                        &intermediate_storage,
                                        2,
                                        128,
                                        256,
                                        2,
                                        8,
                                        128,
                                        128,
                                        128,
                                        256};
}

template <typename DownKernel>
typename DownKernel::Kargs MakeValidDownArgs()
{
    static int intermediate_storage = 0;
    static int w_down_storage       = 0;
    static int w_down_scale_storage = 0;
    static int32_t router_ids[4]    = {0, 1, 1, 0};
    static float router_wts[4]      = {0.5f, 0.5f, 0.25f, 0.75f};
    static int y_storage            = 0;

    return typename DownKernel::Kargs{&intermediate_storage,
                                      &w_down_storage,
                                      &w_down_scale_storage,
                                      router_ids,
                                      router_wts,
                                      &y_storage,
                                      2,
                                      128,
                                      256,
                                      2,
                                      8,
                                      256,
                                      256,
                                      128};
}

using ValidationGateUpProblem =
    WarpDecodeGateUpProblem<bf16_t, bf16_t, float, bf16_t, float, float>;
using ValidationGateUpKernel = WarpDecodeGateUpKernel<ValidationGateUpProblem, WarpDecodePolicy>;

using ValidationDownProblem =
    WarpDecodeDownReduceProblem<bf16_t, bf16_t, float, bf16_t, float>;
using ValidationDownKernel = WarpDecodeDownReduceKernel<ValidationDownProblem, WarpDecodePolicy>;

using Block2DGateUpProblem = WarpDecodeGateUpProblem<bf16_t,
                                                     fp8_t,
                                                     float,
                                                     float,
                                                     float,
                                                     float,
                                                     WarpDecodeScaleLayout::PerTensor,
                                                     WarpDecodeScaleLayout::Block2D<3, 128>>;
using Block2DGateUpKernel = WarpDecodeGateUpKernel<Block2DGateUpProblem, WarpDecodePolicy>;

using Block2DDownProblem =
    WarpDecodeDownReduceProblem<float, fp8_t, float, bf16_t, float, WarpDecodeScaleLayout::Block2D<3, 128>>;
using Block2DDownKernel = WarpDecodeDownReduceKernel<Block2DDownProblem, WarpDecodePolicy>;

} // namespace

TEST(WarpDecodePositive, Bf16Bf16PerTensorFloatScale)
{
    EXPECT_TRUE((RunPositiveCase<bf16_t,
                                 bf16_t,
                                 bf16_t,
                                 bf16_t,
                                 float,
                                 float,
                                 float,
                                 WarpDecodeScaleLayout::PerTensor,
                                 WarpDecodeScaleLayout::PerTensor>(
        "BF16xBF16 per-tensor float scale", 0.05f)));
}

TEST(WarpDecodePositive, Fp8Fp8PerTokenPerTensorFloatScale)
{
    EXPECT_TRUE((RunPositiveCase<fp8_t,
                                 fp8_t,
                                 float,
                                 bf16_t,
                                 float,
                                 float,
                                 float,
                                 WarpDecodeScaleLayout::PerToken,
                                 WarpDecodeScaleLayout::PerTensor>(
        "FP8xFP8 per-token/per-tensor float scale", 0.5f)));
}

TEST(WarpDecodePositive, Fp8Fp8PerTokenBlock2DOneBy128FloatScale)
{
    EXPECT_TRUE((RunPositiveCase<fp8_t,
                                 fp8_t,
                                 float,
                                 bf16_t,
                                 float,
                                 float,
                                 float,
                                 WarpDecodeScaleLayout::PerToken,
                                 WarpDecodeScaleLayout::Block2D<1, 128>>(
        "FP8xFP8 per-token/block2d<1,128> float scale", 0.5f)));
}

TEST(WarpDecodePositive, Bf16Fp8PerTensorBlock2DOneBy128FloatScale)
{
    EXPECT_TRUE((RunPositiveCase<bf16_t,
                                 fp8_t,
                                 float,
                                 bf16_t,
                                 float,
                                 float,
                                 float,
                                 WarpDecodeScaleLayout::PerTensor,
                                 WarpDecodeScaleLayout::Block2D<1, 128>>(
        "BF16xFP8 per-tensor/block2d<1,128> float scale", 0.2f)));
}

TEST(WarpDecodePositive, Fp8Fp8PerTokenBlock2DOneBy32E8m0Scale)
{
    EXPECT_TRUE((RunPositiveCase<fp8_t,
                                 fp8_t,
                                 float,
                                 bf16_t,
                                 float,
                                 float,
                                 e8m0_t,
                                 WarpDecodeScaleLayout::PerToken,
                                 WarpDecodeScaleLayout::Block2D<1, 32>>(
        "FP8xFP8 MXFP8-style per-token/block2d<1,32> e8m0 scale", 1.0f)));
}

TEST(WarpDecodePositive, Bf16Fp8PerTokenBlock2DOneBy32E8m0Scale)
{
    EXPECT_TRUE((RunPositiveCase<bf16_t,
                                 fp8_t,
                                 float,
                                 bf16_t,
                                 float,
                                 float,
                                 e8m0_t,
                                 WarpDecodeScaleLayout::PerToken,
                                 WarpDecodeScaleLayout::Block2D<1, 32>>(
        "BF16xFP8 per-token/block2d<1,32> e8m0 scale", 0.5f)));
}

TEST(WarpDecodePositive, Bf16Mxfp4PerTensorFloatScale)
{
    EXPECT_TRUE((RunPositiveCase<bf16_t,
                                 pk_fp4_t,
                                 float,
                                 bf16_t,
                                 float,
                                 float,
                                 float,
                                 WarpDecodeScaleLayout::PerTensor,
                                 WarpDecodeScaleLayout::PerTensor>(
        "BF16xMXFP4 per-tensor float scale", 1.0f)));
}

TEST(WarpDecodePositive, Bf16Mxfp4PerTensorBlock2DOneBy32E8m0Scale)
{
    EXPECT_TRUE((RunPositiveCase<bf16_t,
                                 pk_fp4_t,
                                 float,
                                 bf16_t,
                                 float,
                                 float,
                                 e8m0_t,
                                 WarpDecodeScaleLayout::PerTensor,
                                 WarpDecodeScaleLayout::Block2D<1, 32>>(
        "BF16xMXFP4 per-tensor/block2d<1,32> e8m0 scale", 1.0f)));
}

TEST(WarpDecodePositive, Bf16Mxfp4PerTokenBlock2DOneBy32E8m0Scale)
{
    EXPECT_TRUE((RunPositiveCase<bf16_t,
                                 pk_fp4_t,
                                 float,
                                 bf16_t,
                                 float,
                                 float,
                                 e8m0_t,
                                 WarpDecodeScaleLayout::PerToken,
                                 WarpDecodeScaleLayout::Block2D<1, 32>>(
        "BF16xMXFP4 per-token/block2d<1,32> e8m0 scale", 1.0f)));
}

TEST(WarpDecodePositive, Fp8Mxfp4PerTokenBlock2DOneBy32E8m0Scale)
{
    EXPECT_TRUE((RunPositiveCase<fp8_t,
                                 pk_fp4_t,
                                 float,
                                 bf16_t,
                                 float,
                                 float,
                                 e8m0_t,
                                 WarpDecodeScaleLayout::PerToken,
                                 WarpDecodeScaleLayout::Block2D<1, 32>>(
        "FP8xMXFP4 per-token/block2d<1,32> e8m0 scale", 1.0f)));
}

TEST(WarpDecodePositive, Bf16Fp8PerTensorBlock2DFourBy32FloatScale)
{
    EXPECT_TRUE((RunPositiveCase<bf16_t,
                                 fp8_t,
                                 float,
                                 bf16_t,
                                 float,
                                 float,
                                 float,
                                 WarpDecodeScaleLayout::PerTensor,
                                 WarpDecodeScaleLayout::Block2D<4, 32>>(
        "BF16xFP8 per-tensor/block2d<4,32> float scale", 0.3f)));
}

TEST(WarpDecodePositive, DeepSeekV3LikeDecodeShape)
{
    // DeepSeek-V3 uses hidden=7168, routed-intermediate=2048, topk=8.
    // Keep E bounded in the unit test so the expert weight tensors stay manageable.
    const WarpDecodeShape shape{1, 7168, 2048, 8, 8};

    EXPECT_TRUE((RunPositiveCase<bf16_t,
                                 fp8_t,
                                 float,
                                 bf16_t,
                                 float,
                                 float,
                                 float,
                                 WarpDecodeScaleLayout::PerTensor,
                                 WarpDecodeScaleLayout::Block2D<128, 128>,
                                 8>(
        "DeepSeekV3-like decode shape", 5.0f, shape)));
}

TEST(WarpDecodePositive, MiniMaxLikeDecodeShape)
{
    // DeepSeek-V3 uses hidden=7168, routed-intermediate=2048, topk=8.
    // Keep E bounded in the unit test so the expert weight tensors stay manageable.
    const WarpDecodeShape shape{1, 3072, 1536, 8, 8};

    EXPECT_TRUE((RunPositiveCase<bf16_t,
                                 fp8_t,
                                 float,
                                 bf16_t,
                                 float,
                                 float,
                                 float,
                                 WarpDecodeScaleLayout::PerTensor,
                                 WarpDecodeScaleLayout::Block2D<128, 128>,
                                 8>(
        "MiniMax-like decode shape", 5.0f, shape)));
}

TEST(WarpDecodeValidation, GateUpRejectsNonDivisibleHidden)
{
    auto args = MakeValidGateUpArgs<ValidationGateUpKernel>();
    args.hidden += 1;
    args.stride_x = args.hidden;
    args.stride_w_gate = args.hidden;
    args.stride_w_up = args.hidden;

    EXPECT_FALSE(ValidationGateUpKernel::IsSupportedArgument(args));
    EXPECT_THROW((launch_warp_decode_gate_up<ValidationGateUpKernel>(args, stream_config{})),
                 std::invalid_argument);
}

TEST(WarpDecodeValidation, GateUpRejectsNullPointer)
{
    auto args  = MakeValidGateUpArgs<ValidationGateUpKernel>();
    args.p_x = nullptr;

    EXPECT_FALSE(ValidationGateUpKernel::IsSupportedArgument(args));
    EXPECT_THROW((launch_warp_decode_gate_up<ValidationGateUpKernel>(args, stream_config{})),
                 std::invalid_argument);
}

TEST(WarpDecodeValidation, GateUpRejectsInvalidStride)
{
    auto args        = MakeValidGateUpArgs<ValidationGateUpKernel>();
    args.stride_x = args.hidden - 1;

    EXPECT_FALSE(ValidationGateUpKernel::IsSupportedArgument(args));
}

TEST(WarpDecodeValidation, GateUpRejectsInvalidBlock2DLayout)
{
    auto args = MakeValidGateUpArgs<Block2DGateUpKernel>();

    EXPECT_FALSE(Block2DGateUpKernel::IsSupportedArgument(args));
    EXPECT_THROW((launch_warp_decode_gate_up<Block2DGateUpKernel>(args, stream_config{})),
                 std::invalid_argument);
}

TEST(WarpDecodeValidation, DownReduceRejectsNonDivisibleInter)
{
    auto args = MakeValidDownArgs<ValidationDownKernel>();
    args.inter += 1;
    args.stride_intermediate = args.inter;
    args.stride_w_down = args.inter;

    EXPECT_FALSE(ValidationDownKernel::IsSupportedArgument(args));
    EXPECT_THROW((launch_warp_decode_down_reduce<ValidationDownKernel>(args, stream_config{})),
                 std::invalid_argument);
}

TEST(WarpDecodeValidation, DownReduceRejectsNullPointer)
{
    auto args = MakeValidDownArgs<ValidationDownKernel>();
    args.p_y  = nullptr;

    EXPECT_FALSE(ValidationDownKernel::IsSupportedArgument(args));
    EXPECT_THROW((launch_warp_decode_down_reduce<ValidationDownKernel>(args, stream_config{})),
                 std::invalid_argument);
}

TEST(WarpDecodeValidation, DownReduceRejectsInvalidStride)
{
    auto args      = MakeValidDownArgs<ValidationDownKernel>();
    args.stride_y = args.hidden - 1;

    EXPECT_FALSE(ValidationDownKernel::IsSupportedArgument(args));
}

TEST(WarpDecodeValidation, DownReduceRejectsInvalidBlock2DLayout)
{
    auto args = MakeValidDownArgs<Block2DDownKernel>();

    EXPECT_FALSE(Block2DDownKernel::IsSupportedArgument(args));
    EXPECT_THROW((launch_warp_decode_down_reduce<Block2DDownKernel>(args, stream_config{})),
                 std::invalid_argument);
}
