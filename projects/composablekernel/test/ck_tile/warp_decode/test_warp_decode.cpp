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

using namespace ck_tile;

template <typename DataType>
void FillRandom(HostTensor<DataType>& tensor, float min_val = -1.0f, float max_val = 1.0f) {
    std::mt19937 generator(42); // fixed seed for reproducibility
    std::uniform_real_distribution<float> distribution(min_val, max_val);

    for(index_t i = 0; i < static_cast<index_t>(tensor.get_element_space_size()); ++i) {
        tensor.mData[i] = type_convert<DataType>(distribution(generator));
    }
}

void FillRouterData(HostTensor<int32_t>& router_ids, HostTensor<float>& router_wts, index_t B, index_t TOP_K, index_t E) {
    std::mt19937 generator(42);
    std::uniform_int_distribution<int32_t> int_dist(0, E - 1);
    std::uniform_real_distribution<float> float_dist(0.1f, 1.0f);

    for(index_t b = 0; b < B; ++b) {
        float sum = 0.0f;
        for(index_t k = 0; k < TOP_K; ++k) {
            router_ids(b, k) = int_dist(generator);
            float w = float_dist(generator);
            router_wts(b, k) = w;
            sum += w;
        }
        // Normalize weights
        for(index_t k = 0; k < TOP_K; ++k) {
            router_wts(b, k) /= sum;
        }
    }
}

template <typename XDataType, typename WDataType, typename IntermediateDataType, typename YDataType, typename ComputeDataType>
bool TestWarpDecode() {
    index_t B = 2;
    index_t HIDDEN = 128;
    index_t INTER = 256;
    index_t TOP_K = 2;
    index_t E = 8;

    HostTensor<XDataType> x({B, HIDDEN});
    HostTensor<WDataType> w_gate({E, INTER, HIDDEN});
    HostTensor<WDataType> w_up({E, INTER, HIDDEN});
    HostTensor<WDataType> w_down({E, HIDDEN, INTER});
    HostTensor<int32_t> router_ids({B, TOP_K});
    HostTensor<float> router_wts({B, TOP_K});

    HostTensor<IntermediateDataType> intermediate_host({B, TOP_K, INTER});
    HostTensor<IntermediateDataType> intermediate_dev({B, TOP_K, INTER});
    
    HostTensor<YDataType> y_host({B, HIDDEN});
    HostTensor<YDataType> y_dev({B, HIDDEN});

    FillRandom(x);
    FillRandom(w_gate);
    FillRandom(w_up);
    FillRandom(w_down);
    FillRouterData(router_ids, router_wts, B, TOP_K, E);

    // Run reference
    reference::reference_warp_decode_gate_up<XDataType, WDataType, ComputeDataType, IntermediateDataType>(
        x, w_gate, w_up, router_ids, intermediate_host);
        
    reference::reference_warp_decode_down_reduce<IntermediateDataType, WDataType, ComputeDataType, YDataType>(
        intermediate_host, w_down, router_ids, router_wts, y_host);

    // Setup device buffers
    DeviceMem x_buf(x.get_element_space_size_in_bytes());
    DeviceMem w_gate_buf(w_gate.get_element_space_size_in_bytes());
    DeviceMem w_up_buf(w_up.get_element_space_size_in_bytes());
    DeviceMem w_down_buf(w_down.get_element_space_size_in_bytes());
    DeviceMem router_ids_buf(router_ids.get_element_space_size_in_bytes());
    DeviceMem router_wts_buf(router_wts.get_element_space_size_in_bytes());
    DeviceMem intermediate_buf(intermediate_dev.get_element_space_size_in_bytes());
    DeviceMem y_buf(y_dev.get_element_space_size_in_bytes());

    x_buf.ToDevice(x.mData.data());
    w_gate_buf.ToDevice(w_gate.mData.data());
    w_up_buf.ToDevice(w_up.mData.data());
    w_down_buf.ToDevice(w_down.mData.data());
    router_ids_buf.ToDevice(router_ids.mData.data());
    router_wts_buf.ToDevice(router_wts.mData.data());

    // Launch Kernels
    using Problem1 = WarpDecodeGateUpProblem<XDataType, WDataType, ComputeDataType, IntermediateDataType>;
    using Policy = WarpDecodePolicy;
    using GateUpKernel = WarpDecodeGateUpKernel<Problem1, Policy>;

    typename GateUpKernel::Kargs args1{
        x_buf.GetDeviceBuffer(), nullptr,
        w_gate_buf.GetDeviceBuffer(), nullptr,
        w_up_buf.GetDeviceBuffer(), nullptr,
        static_cast<int32_t*>(router_ids_buf.GetDeviceBuffer()),
        intermediate_buf.GetDeviceBuffer(),
        B, HIDDEN, INTER, TOP_K, E,
        HIDDEN, HIDDEN, HIDDEN, INTER // strides
    };
    
    auto s = stream_config{};
    launch_warp_decode_gate_up<GateUpKernel>(args1, s);

    using Problem2 = WarpDecodeDownReduceProblem<IntermediateDataType, WDataType, ComputeDataType, YDataType>;
    using DownReduceKernel = WarpDecodeDownReduceKernel<Problem2, Policy>;
    
    typename DownReduceKernel::Kargs args2{
        intermediate_buf.GetDeviceBuffer(),
        w_down_buf.GetDeviceBuffer(), nullptr,
        static_cast<int32_t*>(router_ids_buf.GetDeviceBuffer()),
        static_cast<float*>(router_wts_buf.GetDeviceBuffer()),
        y_buf.GetDeviceBuffer(),
        B, HIDDEN, INTER, TOP_K, E,
        INTER, INTER, HIDDEN // strides
    };
    
    launch_warp_decode_down_reduce<DownReduceKernel>(args2, s);

    // Compare
    y_buf.FromDevice(y_dev.mData.data());

    bool pass = true;
    for (index_t i = 0; i < static_cast<index_t>(y_host.get_element_space_size()); ++i) {
        float host_val = type_convert<float>(y_host.mData[i]);
        float dev_val = type_convert<float>(y_dev.mData[i]);
        if (std::abs(host_val - dev_val) > 1e-2) {
            std::cout << "Mismatch at " << i << ": host=" << host_val << " dev=" << dev_val << "\n";
            pass = false;
            break;
        }
    }
    
    std::cout << "Test " << (pass ? "PASSED" : "FAILED") << std::endl;
    return pass;
}

int main() {
    bool pass = true;
    // We can use bf16, but currently the framework supports various datatypes. 
    // Testing with bf16 for all to keep it simple first
    pass &= TestWarpDecode<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t, float>();
    
    // Future support for MXFP4 etc can be tested here, right now just ensure the basic loop compiles and runs
    return pass ? 0 : 1;
}
