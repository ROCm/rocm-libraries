// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/warp_decode.hpp"
#include "ck_tile/host/reference/reference_warp_decode.hpp"

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <string>

using namespace ck_tile;

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
        buf[i] = type_convert<ScaleDataType>(dist(gen));
}

template <>
void FillScaleRandom<e8m0_t>(std::vector<e8m0_t>& buf, std::size_t count, unsigned seed)
{
    std::mt19937 gen(seed);
    // e8m0 represents powers of 2 via biased exponent. bias=127, so exponent 127 -> 2^0 = 1.0
    // We pick exponents in [124..130] -> scales in [2^-3 .. 2^3] = [0.125 .. 8]
    std::uniform_int_distribution<int> dist(124, 130);
    buf.resize(count);
    for(std::size_t i = 0; i < count; ++i)
        buf[i] = e8m0_t(static_cast<uint8_t>(dist(gen)));
}

void FillRouterData(HostTensor<int32_t>& router_ids,
                    HostTensor<float>& router_wts,
                    index_t B, index_t TOP_K, index_t E)
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
            float w = float_dist(gen);
            router_wts(b, k) = w;
            sum += w;
        }
        for(index_t k = 0; k < TOP_K; ++k)
            router_wts(b, k) /= sum;
    }
}

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------
// Template params:
//   XDataType, WDataType        - activation / weight element types
//   IntermediateDataType         - gate_up output type (stored in compute precision)
//   YDataType                    - final output type
//   ComputeDataType              - accumulation type (float)
//   XScaleDataType, WScaleDataType - scale element types
//   XScaleLayout, WScaleLayout   - WarpDecodeScaleLayout:: types
template <typename XDataType,
          typename WDataType,
          typename IntermediateDataType,
          typename YDataType,
          typename ComputeDataType,
          typename XScaleDataType,
          typename WScaleDataType,
          typename XScaleLayout,
          typename WScaleLayout>
bool RunTest(const std::string& test_name, float atol = 0.02f)
{
    constexpr index_t B      = 2;
    constexpr index_t HIDDEN = 128;
    constexpr index_t INTER  = 256;
    constexpr index_t TOP_K  = 2;
    constexpr index_t E      = 8;

    constexpr bool is_fp8_x  = std::is_same_v<XDataType, fp8_t>;
    constexpr bool is_fp8_w  = std::is_same_v<WDataType, fp8_t>;
    constexpr bool is_mxfp4  = std::is_same_v<WDataType, pk_fp4_t>;
    constexpr index_t WPACK  = is_mxfp4 ? 2 : 1;

    const float x_range = is_fp8_x ? 0.5f : 1.0f;
    const float w_range = is_fp8_w ? 0.25f : 1.0f;

    HostTensor<XDataType> x({B, HIDDEN});
    HostTensor<WDataType> w_gate({E, INTER, HIDDEN});
    HostTensor<WDataType> w_up({E, INTER, HIDDEN});
    HostTensor<WDataType> w_down({E, HIDDEN, INTER});
    HostTensor<int32_t> router_ids({B, TOP_K});
    HostTensor<float>   router_wts({B, TOP_K});

    HostTensor<IntermediateDataType> intermediate_host({B, TOP_K, INTER});
    HostTensor<IntermediateDataType> intermediate_dev({B, TOP_K, INTER});
    HostTensor<YDataType> y_host({B, HIDDEN});
    HostTensor<YDataType> y_dev({B, HIDDEN});

    FillRandom(x, -x_range, x_range, 42);
    FillRandom(w_gate, -w_range, w_range, 43);
    FillRandom(w_up, -w_range, w_range, 44);
    FillRandom(w_down, -w_range, w_range, 45);
    FillRouterData(router_ids, router_wts, B, TOP_K, E);

    // --- Scale setup ---
    using reference::RefScaleMode;
    constexpr bool x_per_tensor = std::is_same_v<XScaleLayout, WarpDecodeScaleLayout::PerTensor>;
    constexpr bool x_per_token  = std::is_same_v<XScaleLayout, WarpDecodeScaleLayout::PerToken>;
    constexpr bool w_per_tensor = std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerTensor>;
    constexpr bool w_per_token  = std::is_same_v<WScaleLayout, WarpDecodeScaleLayout::PerToken>;
    constexpr bool w_block2d    = ScaleLayoutTraits<WScaleLayout>::is_block2d;

    constexpr index_t w_block_n = []() -> index_t {
        if constexpr(w_block2d) return ScaleLayoutTraits<WScaleLayout>::block_n;
        else return 1;
    }();
    constexpr index_t w_block_k = []() -> index_t {
        if constexpr(w_block2d) return ScaleLayoutTraits<WScaleLayout>::block_k;
        else return 1;
    }();

    // Determine reference scale mode
    constexpr RefScaleMode ref_x_mode =
        x_per_token ? RefScaleMode::PerToken :
        x_per_tensor ? RefScaleMode::PerTensor : RefScaleMode::None;
    constexpr RefScaleMode ref_w_mode =
        w_block2d   ? RefScaleMode::Block2D :
        w_per_token ? RefScaleMode::PerToken :
        w_per_tensor ? RefScaleMode::PerTensor : RefScaleMode::None;

    // Activation scale buffer
    std::vector<XScaleDataType> x_scale_buf;
    const XScaleDataType* p_x_scale = nullptr;
    if constexpr(x_per_tensor)
    {
        FillScaleRandom(x_scale_buf, 1, 100);
        p_x_scale = x_scale_buf.data();
    }
    else if constexpr(x_per_token)
    {
        FillScaleRandom(x_scale_buf, B, 101);
        p_x_scale = x_scale_buf.data();
    }

    // Gate weight scale buffer
    std::vector<WScaleDataType> w_gate_scale_buf;
    const WScaleDataType* p_w_gate_scale = nullptr;
    // Up weight scale buffer
    std::vector<WScaleDataType> w_up_scale_buf;
    const WScaleDataType* p_w_up_scale = nullptr;
    // Down weight scale buffer
    std::vector<WScaleDataType> w_down_scale_buf;
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
        FillScaleRandom(w_gate_scale_buf, E * INTER, 200);
        FillScaleRandom(w_up_scale_buf, E * INTER, 201);
        FillScaleRandom(w_down_scale_buf, E * HIDDEN, 202);
        p_w_gate_scale = w_gate_scale_buf.data();
        p_w_up_scale   = w_up_scale_buf.data();
        p_w_down_scale = w_down_scale_buf.data();
    }
    else if constexpr(w_block2d)
    {
        std::size_t gate_up_scale_count = (E * INTER / w_block_n) * (HIDDEN / w_block_k);
        std::size_t down_scale_count    = (E * HIDDEN / w_block_n) * (INTER / w_block_k);
        FillScaleRandom(w_gate_scale_buf, gate_up_scale_count, 200);
        FillScaleRandom(w_up_scale_buf, gate_up_scale_count, 201);
        FillScaleRandom(w_down_scale_buf, down_scale_count, 202);
        p_w_gate_scale = w_gate_scale_buf.data();
        p_w_up_scale   = w_up_scale_buf.data();
        p_w_down_scale = w_down_scale_buf.data();
    }

    // --- CPU reference ---
    reference::reference_warp_decode_gate_up<XDataType, WDataType, ComputeDataType,
                                             IntermediateDataType, XScaleDataType, WScaleDataType>(
        x, w_gate, w_up, router_ids, intermediate_host,
        p_x_scale, p_w_gate_scale, p_w_up_scale,
        ref_x_mode, ref_w_mode, w_block_n, w_block_k);

    reference::reference_warp_decode_down_reduce<IntermediateDataType, WDataType, ComputeDataType,
                                                  YDataType, WScaleDataType>(
        intermediate_host, w_down, router_ids, router_wts, y_host,
        p_w_down_scale, ref_w_mode, w_block_n, w_block_k);

    // --- Device ---
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

    // Scale device buffers (only allocate if needed)
    DeviceMem x_scale_dbuf(x_scale_buf.size() * sizeof(XScaleDataType));
    DeviceMem w_gate_scale_dbuf(w_gate_scale_buf.size() * sizeof(WScaleDataType));
    DeviceMem w_up_scale_dbuf(w_up_scale_buf.size() * sizeof(WScaleDataType));
    DeviceMem w_down_scale_dbuf(w_down_scale_buf.size() * sizeof(WScaleDataType));

    void* p_x_scale_dev = nullptr;
    void* p_w_gate_scale_dev = nullptr;
    void* p_w_up_scale_dev = nullptr;
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

    // --- Launch gate_up kernel ---
    using GateUpProblem = WarpDecodeGateUpProblem<
        XDataType, WDataType, ComputeDataType, IntermediateDataType,
        XScaleDataType, WScaleDataType, XScaleLayout, WScaleLayout>;
    using Policy = WarpDecodePolicy;
    using GateUpKernel = WarpDecodeGateUpKernel<GateUpProblem, Policy>;

    typename GateUpKernel::Kargs args1{
        x_buf.GetDeviceBuffer(),
        p_x_scale_dev,
        w_gate_buf.GetDeviceBuffer(),
        p_w_gate_scale_dev,
        w_up_buf.GetDeviceBuffer(),
        p_w_up_scale_dev,
        static_cast<int32_t*>(router_ids_buf.GetDeviceBuffer()),
        inter_buf.GetDeviceBuffer(),
        B, HIDDEN, INTER, TOP_K, E,
        HIDDEN, HIDDEN / WPACK, HIDDEN / WPACK, INTER
    };

    auto s = stream_config{};
    launch_warp_decode_gate_up<GateUpKernel>(args1, s);

    // --- Launch down_reduce kernel ---
    using DownProblem = WarpDecodeDownReduceProblem<
        IntermediateDataType, WDataType, ComputeDataType, YDataType,
        WScaleDataType, WScaleLayout>;
    using DownKernel = WarpDecodeDownReduceKernel<DownProblem, Policy>;

    typename DownKernel::Kargs args2{
        inter_buf.GetDeviceBuffer(),
        w_down_buf.GetDeviceBuffer(),
        p_w_down_scale_dev,
        static_cast<int32_t*>(router_ids_buf.GetDeviceBuffer()),
        static_cast<float*>(router_wts_buf.GetDeviceBuffer()),
        y_buf.GetDeviceBuffer(),
        B, HIDDEN, INTER, TOP_K, E,
        INTER, INTER / WPACK, HIDDEN
    };

    launch_warp_decode_down_reduce<DownKernel>(args2, s);

    // --- Verify intermediate (gate_up output) ---
    inter_buf.FromDevice(intermediate_dev.mData.data());

    bool pass = true;
    float inter_max_diff = 0.0f;
    for(index_t i = 0; i < static_cast<index_t>(intermediate_host.get_element_space_size()); ++i)
    {
        float host_val = type_convert<float>(intermediate_host.mData[i]);
        float dev_val  = type_convert<float>(intermediate_dev.mData[i]);
        float diff     = std::abs(host_val - dev_val);
        inter_max_diff = std::max(inter_max_diff, diff);
        if(diff > atol)
        {
            if(pass)
                std::cout << "  Intermediate mismatch at " << i
                          << ": host=" << host_val << " dev=" << dev_val
                          << " diff=" << diff << "\n";
            pass = false;
        }
    }

    // --- Verify final output ---
    y_buf.FromDevice(y_dev.mData.data());

    float y_max_diff = 0.0f;
    for(index_t i = 0; i < static_cast<index_t>(y_host.get_element_space_size()); ++i)
    {
        float host_val = type_convert<float>(y_host.mData[i]);
        float dev_val  = type_convert<float>(y_dev.mData[i]);
        float diff     = std::abs(host_val - dev_val);
        y_max_diff = std::max(y_max_diff, diff);
        if(diff > atol)
        {
            if(pass)
                std::cout << "  Output mismatch at " << i
                          << ": host=" << host_val << " dev=" << dev_val
                          << " diff=" << diff << "\n";
            pass = false;
        }
    }

    std::cout << "[" << (pass ? "PASS" : "FAIL") << "] " << test_name
              << "  (inter_diff=" << inter_max_diff
              << ", y_diff=" << y_max_diff << ", atol=" << atol << ")\n";
    return pass;
}

int main()
{
    int num_pass = 0;
    int num_fail = 0;

    auto run = [&](auto fn) {
        bool ok = fn();
        if(ok) ++num_pass; else ++num_fail;
    };

    // 1. BF16 x BF16, per-tensor float scales
    run([&]{ return RunTest<bf16_t, bf16_t, bf16_t, bf16_t, float,
                            float, float,
                            WarpDecodeScaleLayout::PerTensor,
                            WarpDecodeScaleLayout::PerTensor>(
        "BF16xBF16 per-tensor float scale", 0.05f); });

    // 2. FP8 x FP8, per-token x-scale, per-tensor w-scale (float scales)
    run([&]{ return RunTest<fp8_t, fp8_t, float, bf16_t, float,
                            float, float,
                            WarpDecodeScaleLayout::PerToken,
                            WarpDecodeScaleLayout::PerTensor>(
        "FP8xFP8 per-token/per-tensor float scale", 0.5f); });

    // 3. FP8 x FP8, per-token x-scale, block2d w-scale 1x128 (float scales)
    run([&]{ return RunTest<fp8_t, fp8_t, float, bf16_t, float,
                            float, float,
                            WarpDecodeScaleLayout::PerToken,
                            WarpDecodeScaleLayout::Block2D<1, 128>>(
        "FP8xFP8 per-token/block2d<1,128> float scale", 0.5f); });

    // 4. BF16 x FP8, per-tensor x-scale, block2d w-scale 1x128 (float scales)
    run([&]{ return RunTest<bf16_t, fp8_t, float, bf16_t, float,
                            float, float,
                            WarpDecodeScaleLayout::PerTensor,
                            WarpDecodeScaleLayout::Block2D<1, 128>>(
        "BF16xFP8 per-tensor/block2d<1,128> float scale", 0.2f); });

    // 5. FP8 x FP8, per-token x-scale (float), block2d w-scale 1x32 with e8m0_t (MXFP8-style)
    run([&]{ return RunTest<fp8_t, fp8_t, float, bf16_t, float,
                            float, e8m0_t,
                            WarpDecodeScaleLayout::PerToken,
                            WarpDecodeScaleLayout::Block2D<1, 32>>(
        "FP8xFP8 MXFP8-style per-token/block2d<1,32> e8m0 scale", 1.0f); });

    // 6. BF16 x FP8, per-token x-scale (float), block2d w-scale 1x32 with e8m0_t
    run([&]{ return RunTest<bf16_t, fp8_t, float, bf16_t, float,
                            float, e8m0_t,
                            WarpDecodeScaleLayout::PerToken,
                            WarpDecodeScaleLayout::Block2D<1, 32>>(
        "BF16xFP8 per-token/block2d<1,32> e8m0 scale", 0.5f); });

    // 6b. BF16 x MXFP4, per-tensor float scales (no block2d, no e8m0 — diagnostic)
    run([&]{ return RunTest<bf16_t, pk_fp4_t, float, bf16_t, float,
                            float, float,
                            WarpDecodeScaleLayout::PerTensor,
                            WarpDecodeScaleLayout::PerTensor>(
        "BF16xMXFP4 per-tensor float scale (diag)", 1.0f); });

    // 7. BF16 x MXFP4, per-tensor x-scale, block2d w-scale 1x32 with e8m0_t
    run([&]{ return RunTest<bf16_t, pk_fp4_t, float, bf16_t, float,
                            float, e8m0_t,
                            WarpDecodeScaleLayout::PerTensor,
                            WarpDecodeScaleLayout::Block2D<1, 32>>(
        "BF16xMXFP4 per-tensor/block2d<1,32> e8m0 scale", 1.0f); });

    // 8. BF16 x MXFP4, per-token x-scale, block2d w-scale 1x32 with e8m0_t
    run([&]{ return RunTest<bf16_t, pk_fp4_t, float, bf16_t, float,
                            float, e8m0_t,
                            WarpDecodeScaleLayout::PerToken,
                            WarpDecodeScaleLayout::Block2D<1, 32>>(
        "BF16xMXFP4 per-token/block2d<1,32> e8m0 scale", 1.0f); });

    // 9. FP8 x MXFP4, per-token x-scale, block2d w-scale 1x32 with e8m0_t
    run([&]{ return RunTest<fp8_t, pk_fp4_t, float, bf16_t, float,
                            float, e8m0_t,
                            WarpDecodeScaleLayout::PerToken,
                            WarpDecodeScaleLayout::Block2D<1, 32>>(
        "FP8xMXFP4 per-token/block2d<1,32> e8m0 scale", 1.0f); });

    std::cout << "\n=== Summary: " << num_pass << " passed, " << num_fail << " failed ===\n";
    return num_fail > 0 ? 1 : 0;
}
