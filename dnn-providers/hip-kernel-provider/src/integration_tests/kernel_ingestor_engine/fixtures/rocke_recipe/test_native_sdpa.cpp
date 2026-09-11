// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Standalone consumer of the proposed rocKE public contract. No builder API.
#include <amd_comgr.h>
#include <hip/hip_runtime_api.h>
#include <rocke/abi.h>
#include <rocke/online.h>
#include <rocke/recipe_guard.h>
#include <rocke/recipe_launch.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Bytes = std::vector<unsigned char>;
namespace fs = std::filesystem;
static unsigned compile_count = 0;
static unsigned launch_count = 0; // Updated by the main thread after joins.

static void require(bool ok, const std::string& message)
{
    if(!ok)
        throw std::runtime_error(message);
}
static void hip_check(hipError_t status)
{
    if(status != hipSuccess)
        throw std::runtime_error(hipGetErrorString(status));
}
static void hip_cleanup(hipError_t status) noexcept
{
    if(status != hipSuccess)
    {
        std::cerr << "HIP cleanup failed: " << hipGetErrorString(status) << '\n';
        std::terminate();
    }
}
static void comgr_check(amd_comgr_status_t status)
{
    if(status != AMD_COMGR_STATUS_SUCCESS)
    {
        const char* text = nullptr;
        amd_comgr_status_string(status, &text);
        throw std::runtime_error(std::string("COMGR: ") + (text ? text : "unknown"));
    }
}
static std::string read_text(const fs::path& path)
{
    std::ifstream file(path, std::ios::binary);
    require(file.good(), "cannot read " + path.string());
    return {std::istreambuf_iterator<char>(file), {}};
}
static Bytes read_bytes(const fs::path& path)
{
    const auto text = read_text(path);
    return {text.begin(), text.end()};
}
static void write_bytes(const fs::path& path, const char* data, size_t size)
{
    std::ofstream file(path, std::ios::binary);
    file.write(data, static_cast<std::streamsize>(size));
    require(file.good(), "cannot write " + path.string());
}
struct DataSet
{
    amd_comgr_data_set_t handle{};
    DataSet()
    {
        comgr_check(amd_comgr_create_data_set(&handle));
    }
    ~DataSet()
    {
        amd_comgr_destroy_data_set(handle);
    }
    DataSet(const DataSet&) = delete;
    DataSet& operator=(const DataSet&) = delete;
};
struct ActionInfo
{
    amd_comgr_action_info_t handle{};
    ActionInfo()
    {
        comgr_check(amd_comgr_create_action_info(&handle));
    }
    ~ActionInfo()
    {
        amd_comgr_destroy_action_info(handle);
    }
};
struct Data
{
    amd_comgr_data_t handle{};
    ~Data()
    {
        if(handle.handle)
            amd_comgr_release_data(handle);
    }
};
static void action(amd_comgr_action_kind_t kind, ActionInfo& info, DataSet& input, DataSet& output)
{
    auto status = amd_comgr_do_action(kind, info.handle, input.handle, output.handle);
    if(status != AMD_COMGR_STATUS_SUCCESS)
    {
        size_t count = 0;
        amd_comgr_action_data_count(output.handle, AMD_COMGR_DATA_KIND_LOG, &count);
        for(size_t i = 0; i < count; ++i)
        {
            Data log;
            size_t size = 0;
            if(amd_comgr_action_data_get_data(
                   output.handle, AMD_COMGR_DATA_KIND_LOG, i, &log.handle)
               != AMD_COMGR_STATUS_SUCCESS)
                continue;
            amd_comgr_get_data(log.handle, &size, nullptr);
            std::string text(size, '\0');
            amd_comgr_get_data(log.handle, &size, text.data());
            std::cerr << text << '\n';
        }
    }
    comgr_check(status);
}
static Bytes compile(const std::string& llvm, const std::string& target)
{
    ++compile_count;
    DataSet input, bc, reloc, executable;
    ActionInfo info;
    Data source;
    comgr_check(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &source.handle));
    comgr_check(amd_comgr_set_data(source.handle, llvm.size(), llvm.data()));
    comgr_check(amd_comgr_set_data_name(source.handle, "kernel.ll"));
    comgr_check(amd_comgr_data_set_add(input.handle, source.handle));
    comgr_check(
        amd_comgr_action_info_set_isa_name(info.handle, ("amdgcn-amd-amdhsa--" + target).c_str()));
    comgr_check(amd_comgr_action_info_set_language(info.handle, AMD_COMGR_LANGUAGE_LLVM_IR));
    comgr_check(amd_comgr_action_info_set_logging(info.handle, true));
    const char* options[] = {"-O3"};
    comgr_check(amd_comgr_action_info_set_option_list(info.handle, options, 1));
    action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, info, input, bc);
    action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, info, bc, reloc);
    action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, info, reloc, executable);
    size_t count = 0;
    comgr_check(
        amd_comgr_action_data_count(executable.handle, AMD_COMGR_DATA_KIND_EXECUTABLE, &count));
    require(count == 1, "expected one executable");
    Data code;
    comgr_check(amd_comgr_action_data_get_data(
        executable.handle, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &code.handle));
    size_t size = 0;
    comgr_check(amd_comgr_get_data(code.handle, &size, nullptr));
    Bytes bytes(size);
    comgr_check(amd_comgr_get_data(code.handle, &size, reinterpret_cast<char*>(bytes.data())));
    return bytes;
}

struct Request
{
    long dimension;
    std::string target = "gfx950";
};
using LaunchPlan = std::unique_ptr<rocke_launch_plan_t, decltype(&rocke_launch_plan_free)>;
struct Specialization
{
    LaunchPlan plan{nullptr, rocke_launch_plan_free};
    rocke_launch_dims_t grid{}, block{};
    unsigned lds = 0;
    std::string llvm;
    Bytes code;
};
static Specialization specialize(const Bytes& bundle, const Request& request)
{
    rocke_recipe_spec_int_t ints[] = {{"S", request.dimension}};
    char error[1024]{};
    rocke_guard_verdict_t verdict = ROCKE_GUARD_ABSENT;
    auto status = rocke_bundle_check_guard_cbor(bundle.data(),
                                                bundle.size(),
                                                "sdpa_dense_bf16_d128_causal",
                                                request.target.c_str(),
                                                ints,
                                                1,
                                                nullptr,
                                                0,
                                                0,
                                                &verdict,
                                                error,
                                                sizeof(error));
    require(status == ROCKE_OK, std::string("admission error: ") + error);
    require(verdict == ROCKE_GUARD_ADMITTED, std::string("configuration refused: ") + error);
    Specialization out;
    rocke_launch_plan_t* raw_plan = nullptr;
    status = rocke_bundle_plan_launch_cbor(bundle.data(),
                                           bundle.size(),
                                           "sdpa_dense_bf16_d128_causal",
                                           request.target.c_str(),
                                           ints,
                                           1,
                                           nullptr,
                                           0,
                                           &raw_plan,
                                           error,
                                           sizeof(error));
    out.plan.reset(raw_plan);
    require(status == ROCKE_OK && raw_plan, std::string("planning: ") + error);
    require(rocke_launch_plan_geometry(raw_plan, &out.grid, &out.block, &out.lds),
            "missing geometry");
    char* raw_llvm = nullptr;
    status = rocke_online_bundle_cbor_to_llvm(bundle.data(),
                                              bundle.size(),
                                              "sdpa_dense_bf16_d128_causal",
                                              request.target.c_str(),
                                              ints,
                                              1,
                                              nullptr,
                                              0,
                                              &raw_llvm,
                                              nullptr,
                                              nullptr,
                                              error,
                                              sizeof(error));
    std::unique_ptr<char, decltype(&rocke_online_free)> owned_llvm(raw_llvm, rocke_online_free);
    require(status == ROCKE_OK && raw_llvm, std::string("lowering: ") + error);
    out.llvm = raw_llvm;
    out.code = compile(out.llvm, request.target);
    return out;
}

// Example provider policy: semantic operand names map to tensor UIDs. These
// bindings are caller-owned policy, not information inferred from the recipe.
using Buffers = std::map<int64_t, void*>;
static const std::map<std::string, int64_t> operand_uids{
    {"q_ptr", 17}, {"k_ptr", 42}, {"v_ptr", 63}, {"o_ptr", 91}};
struct Prepared
{
    Specialization spec;
    hipModule_t module{};
    hipFunction_t function{};
    explicit Prepared(Specialization value)
        : spec(std::move(value))
    {
        hip_check(hipModuleLoadData(&module, spec.code.data()));
        auto status = hipModuleGetFunction(
            &function, module, rocke_launch_plan_kernel_name(spec.plan.get()));
        if(status != hipSuccess)
        {
            hip_cleanup(hipModuleUnload(module));
            module = nullptr;
            hip_check(status);
        }
    }
    ~Prepared()
    {
        if(module)
            hip_cleanup(hipModuleUnload(module));
    }
    Prepared(const Prepared&) = delete;
    Prepared& operator=(const Prepared&) = delete;
    void execute(const Buffers& buffers, float scale, hipStream_t stream) const
    {
        size_t size = rocke_launch_plan_kernarg_size(spec.plan.get());
        Bytes args(size, 0);
        for(int i = 0; i < rocke_launch_plan_num_args(spec.plan.get()); ++i)
        {
            const auto* arg = rocke_launch_plan_arg(spec.plan.get(), i);
            require(arg && arg->name && arg->offset <= size && arg->size <= size - arg->offset,
                    "invalid argument description");
            if(arg->kind == ROCKE_ARG_POINTER)
            {
                auto uid = operand_uids.find(arg->name);
                require(uid != operand_uids.end(), "unbound pointer argument");
                auto buffer = buffers.find(uid->second);
                require(buffer != buffers.end(), "missing tensor UID");
                require(arg->size == sizeof(void*), "unexpected pointer width");
                std::memcpy(args.data() + arg->offset, &buffer->second, arg->size);
            }
            else
            {
                require(std::string(arg->name) == "scale" && arg->kind == ROCKE_ARG_F32
                            && arg->size == sizeof(scale),
                        "unbound scalar argument");
                std::memcpy(args.data() + arg->offset, &scale, arg->size);
            }
        }
        void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                         args.data(),
                         HIP_LAUNCH_PARAM_BUFFER_SIZE,
                         &size,
                         HIP_LAUNCH_PARAM_END};
        hip_check(hipModuleLaunchKernel(function,
                                        spec.grid.x,
                                        spec.grid.y,
                                        spec.grid.z,
                                        spec.block.x,
                                        spec.block.y,
                                        spec.block.z,
                                        spec.lds,
                                        stream,
                                        nullptr,
                                        extra));
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

static void numeric(const Prepared& prepared, int sequence, int seed)
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
        rejected = std::string(error.what()) == "missing tensor UID";
    }
    require(rejected, "missing UID not rejected");
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
        require(argc == 3, "usage: test_native_sdpa ARTIFACT_DIR --compile-only|--gpu");
        const bool gpu = std::string(argv[2]) == "--gpu";
        require(gpu || std::string(argv[2]) == "--compile-only", "unknown mode");
        const fs::path root(argv[1]);
        auto bundle = read_bytes(root / "sdpa_dense.cbor");
        if(gpu)
        {
            hip_check(hipSetDevice(0));
            hipDeviceProp_t props{};
            hip_check(hipGetDeviceProperties(&props, 0));
            require(std::string(props.gcnArchName).starts_with("gfx950"), "requires gfx950");
            std::cout << "{\"device_arch\":\"" << props.gcnArchName << "\"}\n";
        }
        for(int refused : {0, 513, 1280})
        {
            bool rejected = false;
            try
            {
                specialize(bundle, Request{refused});
            }
            catch(const std::runtime_error& error)
            {
                rejected = std::string(error.what()).starts_with("configuration refused:");
            }
            require(rejected && compile_count == 0, "guard refusal did not precede compilation");
        }
        for(int sequence : {512, 768, 1024})
        {
            auto spec = specialize(bundle, Request{sequence});
            write_bytes(root / ("native-" + std::to_string(sequence) + ".ll"),
                        spec.llvm.data(),
                        spec.llvm.size());
            require(spec.llvm
                        == read_text(root / ("reference-" + std::to_string(sequence) + ".ll")),
                    "LLVM byte mismatch at S=" + std::to_string(sequence));
            require(spec.grid.x == static_cast<unsigned>(sequence / 256) && spec.grid.y == 4
                        && spec.grid.z == 1 && spec.block.x == 512 && spec.block.y == 1
                        && spec.block.z == 1,
                    "unexpected launch geometry");
            write_bytes(root / ("native-" + std::to_string(sequence) + ".hsaco"),
                        reinterpret_cast<const char*>(spec.code.data()),
                        spec.code.size());
            std::cout << "{\"S\":" << sequence
                      << ",\"llvm_identity\":\"PASS\",\"hsaco_bytes\":" << spec.code.size()
                      << "}\n";
            if(gpu)
            {
                Prepared prepared(std::move(spec));
                numeric(prepared, sequence, 0);
                numeric(prepared, sequence, 1);
            }
        }
        require(compile_count == 3, "unexpected compilation count");
#ifdef __linux__
        require(read_text("/proc/self/maps").find("libpython") == std::string::npos,
                "Python loaded in native consumer");
#endif
        std::cout << "{\"result\":\"PASS\",\"compilations\":" << compile_count
                  << ",\"gpu_launches\":" << launch_count << "}\n";
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
