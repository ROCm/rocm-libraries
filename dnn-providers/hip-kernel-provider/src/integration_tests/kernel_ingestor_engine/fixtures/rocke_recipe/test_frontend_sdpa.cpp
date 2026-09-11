// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <hip/hip_runtime_api.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>

static unsigned launch_count = 0;
using Buffers = std::map<int64_t, void*>;
static void require(bool condition, const std::string& message)
{
    if(!condition)
        throw std::runtime_error(message);
}
static void hip_check(hipError_t status)
{
    require(status == hipSuccess, hipGetErrorString(status));
}
static void hip_cleanup(hipError_t status)
{
    if(status != hipSuccess)
        std::terminate();
}
static void fe_check(const hipdnn_frontend::Error& error)
{
    require(error.code == hipdnn_frontend::ErrorCode::OK, error.err_msg);
}
static void dnn_check(hipdnnStatus_t status)
{
    require(status == HIPDNN_STATUS_SUCCESS, "hipDNN call failed: " + std::to_string(status));
}

// Declared before Graph so descriptors/plans are destroyed before their handle,
// including when graph construction fails.
struct FrontendHandle
{
    hipdnnHandle_t value{};
    FrontendHandle()
    {
        dnn_check(hipdnnCreate(&value));
    }
    ~FrontendHandle()
    {
        hipdnnDestroy(value);
    }
    FrontendHandle(const FrontendHandle&) = delete;
    FrontendHandle& operator=(const FrontendHandle&) = delete;
};

struct PreparedFrontend
{
    FrontendHandle handle;
    hipdnn_frontend::graph::Graph graph;
    int64_t uidOffset;
    explicit PreparedFrontend(int sequence,
                              int64_t offset = 0,
                              bool causal = true,
                              const std::string& mode = "normal")
        : uidOffset(offset)
    {
        using namespace hipdnn_frontend;
        graph.set_io_data_type(DataType::BFLOAT16)
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT);
        const auto tensor = [sequence, offset, &mode](int64_t uid) {
            auto result = std::make_shared<graph::TensorAttributes>();
            result->set_uid(uid + offset)
                .set_dim({1, 4, sequence, 128})
                .set_stride({sequence * 4 * 128, 128, 4 * 128, 1})
                .set_data_type(DataType::BFLOAT16);
            if(mode == "dtype")
                result->set_data_type(DataType::HALF);
            if(mode == "layout")
                result->set_stride({sequence * 4 * 128, sequence * 128, 128, 1});
            if(mode == "head")
                result->set_dim({1, 4, sequence, 64});
            return result;
        };
        auto q = tensor(17), k = tensor(42), v = tensor(63);
        graph::SdpaAttributes attributes;
        attributes.set_generate_stats(false).set_causal_mask(causal).set_attn_scale(
            1.0f / std::sqrt(128.0f));
        auto [o, stats] = graph.sdpa(q, k, v, attributes);
        o->set_uid(91 + offset)
            .set_output(true)
            .set_dim({1, 4, sequence, 128})
            .set_stride({sequence * 4 * 128, 128, 4 * 128, 1});
        const auto engine
            = hipdnn_data_sdk::utilities::engineNameToId("hipkernel:RockeSdpaExample");
        graph.set_preferred_engine_id_ext(engine);
        fe_check(graph.build_operation_graph(handle.value));
        fe_check(graph.create_execution_plan_ext(engine, {}));
        fe_check(graph.check_support());
        fe_check(graph.build_plans());
        int64_t selected = 0;
        fe_check(graph.get_execution_plan_engine_id(selected));
        require(selected == engine, "unexpected selected engine");
        int64_t workspace = -1;
        fe_check(graph.get_workspace_size(workspace));
        require(workspace == 0, "unexpected SDPA workspace");
        std::cout << "{\"selected_engine\":\"hipkernel:RockeSdpaExample\",\"S\":" << sequence
                  << "}\n";
    }
    void execute(const Buffers& buffers, float, hipStream_t stream) const
    {
        dnn_check(hipdnnSetStream(handle.value, stream));
        std::unordered_map<int64_t, void*> variants;
        for(const auto& [uid, pointer] : buffers)
            variants.emplace(uid + uidOffset, pointer);
        fe_check(graph.execute(handle.value, variants, nullptr));
    }
};
struct DeviceMemory
{
    void* ptr = nullptr;
    explicit DeviceMemory(size_t size)
    {
        hip_check(hipMalloc(&ptr, size));
    }
    ~DeviceMemory()
    {
        if(ptr)
            hip_cleanup(hipFree(ptr));
    }
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
};
struct Stream
{
    hipStream_t value{};
    Stream()
    {
        hip_check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking));
    }
    ~Stream()
    {
        hip_cleanup(hipStreamDestroy(value));
    }
};

static uint16_t bf16(float value)
{
    uint32_t bits = std::bit_cast<uint32_t>(value);
    return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
static float fp32(uint16_t value)
{
    return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
}

static void numeric(const PreparedFrontend& prepared, int sequence, int seed)
{
    constexpr int heads = 4;
    constexpr int dimension = 128;
    const size_t count = static_cast<size_t>(sequence) * heads * dimension;
    const auto index = [](int token, int head, int column) {
        return (static_cast<size_t>(token) * heads + head) * dimension + column;
    };
    std::vector<uint16_t> q(count), k(count), v(count), out(count + 16, bf16(-99));
    uint32_t rng = 12345 + seed;
    const auto sample = [&rng]() {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return bf16((static_cast<int>(rng % 2049) - 1024) / 1024.0f);
    };
    for(size_t i = 0; i < count; ++i)
    {
        q[i] = sample();
        k[i] = sample();
        v[i] = sample();
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(dimension));
    Stream stream;
    DeviceMemory dq(count * 2), dk(count * 2), dv(count * 2), dout(out.size() * 2);
    hip_check(hipMemcpyAsync(dq.ptr, q.data(), count * 2, hipMemcpyHostToDevice, stream.value));
    hip_check(hipMemcpyAsync(dk.ptr, k.data(), count * 2, hipMemcpyHostToDevice, stream.value));
    hip_check(hipMemcpyAsync(dv.ptr, v.data(), count * 2, hipMemcpyHostToDevice, stream.value));
    hip_check(
        hipMemcpyAsync(dout.ptr, out.data(), out.size() * 2, hipMemcpyHostToDevice, stream.value));
    bool rejected = false;
    try
    {
        prepared.execute({{17, dq.ptr}}, scale, stream.value);
    }
    catch(const std::runtime_error& error)
    {
        rejected = true;
    }
    require(rejected, "missing UID not rejected");
    if(seed == 0)
    {
        for(bool overlap : {false, true})
        {
            bool invalidRejected = false;
            auto invalidQ = overlap ? dk.ptr : static_cast<void*>(static_cast<char*>(dq.ptr) + 2);
            try
            {
                prepared.execute({{17, invalidQ}, {42, dk.ptr}, {63, dv.ptr}, {91, dout.ptr}},
                                 scale,
                                 stream.value);
            }
            catch(const std::runtime_error&)
            {
                invalidRejected = true;
            }
            require(invalidRejected, "invalid buffer was accepted");
        }
    }
    prepared.execute(
        {{17, dq.ptr}, {42, dk.ptr}, {63, dv.ptr}, {91, dout.ptr}}, scale, stream.value);
    hip_check(
        hipMemcpyAsync(out.data(), dout.ptr, out.size() * 2, hipMemcpyDeviceToHost, stream.value));
    hip_check(hipStreamSynchronize(stream.value));
    double max_error = 0, squared_error = 0;
    std::vector<double> scores(sequence);
    for(int h = 0; h < heads; ++h)
    {
        for(int row = 0; row < sequence; ++row)
        {
            double maximum = -std::numeric_limits<double>::infinity();
            for(int col = 0; col <= row; ++col)
            {
                double score = 0;
                for(int d = 0; d < dimension; ++d)
                    score += static_cast<double>(fp32(q[index(row, h, d)]))
                             * fp32(k[index(col, h, d)]);
                scores[col] = score * scale;
                maximum = std::max(maximum, scores[col]);
            }
            double sum = 0;
            for(int col = 0; col <= row; ++col)
            {
                scores[col] = std::exp(scores[col] - maximum);
                sum += scores[col];
            }
            for(int d = 0; d < dimension; ++d)
            {
                double reference = 0;
                for(int col = 0; col <= row; ++col)
                    reference += scores[col] * fp32(v[index(col, h, d)]);
                reference /= sum;
                const double actual = fp32(out[index(row, h, d)]);
                require(std::isfinite(actual), "nonfinite SDPA output");
                const double error = std::abs(actual - reference);
                max_error = std::max(max_error, error);
                squared_error += error * error;
            }
        }
    }
    for(size_t i = count; i < out.size(); ++i)
        require(out[i] == bf16(-99), "output guard overwritten");
    const double rms = std::sqrt(squared_error / count);
    std::cout << "{\"S\":" << sequence << ",\"seed\":" << seed << ",\"max_abs_error\":" << max_error
              << ",\"rms_error\":" << rms << "}\n";
    require(max_error < 0.04 && rms < 0.01, "SDPA numerical mismatch");
    ++launch_count;
}

int main(int argc, char** argv)
{
    try
    {
        require(argc == 2, "usage: test_frontend_sdpa PLUGIN_DIRECTORY");
        const char* paths[] = {argv[1]};
        dnn_check(hipdnnSetEnginePluginPaths_ext(1, paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE));
        hip_check(hipSetDevice(0));
        hipDeviceProp_t props{};
        hip_check(hipGetDeviceProperties(&props, 0));
        require(std::string(props.gcnArchName).starts_with("gfx950"), "test requires gfx950");
        for(int sequence : {512, 768, 1024})
        {
            PreparedFrontend prepared(sequence, sequence * 10);
            numeric(prepared, sequence, 0);
            numeric(prepared, sequence, 1);
        }
        for(int sequence : {513, 1280})
        {
            bool rejected = false;
            try
            {
                PreparedFrontend unsupported(sequence);
            }
            catch(const std::runtime_error&)
            {
                rejected = true;
            }
            require(rejected, "unsupported sequence accepted");
        }
        for(const auto* mode : {"dtype", "layout", "head"})
        {
            bool invalidRejected = false;
            try
            {
                PreparedFrontend invalid(512, 0, true, mode);
            }
            catch(const std::runtime_error&)
            {
                invalidRejected = true;
            }
            require(invalidRejected, std::string("unsupported graph accepted: ") + mode);
        }
        bool rejected = false;
        try
        {
            PreparedFrontend noncausal(512, 0, false);
        }
        catch(const std::runtime_error&)
        {
            rejected = true;
        }
        require(rejected, "noncausal graph accepted");
#ifdef __linux__
        std::ifstream maps("/proc/self/maps");
        const std::string loaded{std::istreambuf_iterator<char>(maps), {}};
        require(loaded.find("libpython") == std::string::npos, "Python loaded in frontend process");
#endif
        std::cout << "{\"frontend_result\":\"PASS\",\"launches\":" << launch_count << "}\n";
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
