// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/BindingPublication.hpp>
#include <hipdnn_plugin_sdk/ingestor/CompiledLaunch.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <nlohmann/json.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvFwdTestGraphs.hpp"

/**
 * @file TestDeclarativeLaunchHandler.cpp
 * @brief The declarative launch handler launches exactly the grid, block, dynamic LDS and
 *        argument list its launch resolved to.
 *
 * The kernel under test records what it receives: gridDim and blockDim on all three axes,
 * a word read back through dynamic LDS, the device address of each pointer argument and
 * the bytes of each scalar. The launch gives every axis a distinct extent, a nonzero
 * shared_mem_bytes, a workspace argument, and one scalar of each type, so a handler that
 * swaps two axes, drops the LDS size or passes another address writes another record.
 *
 * The case reaches the handler through restoreLaunch(), the one entry that takes code
 * object bytes rather than a packed archive, so the kernel is compiled here with HIPRTC
 * and no descriptor set is packed for it. prepare() and restoreLaunch() apply the
 * resolved geometry through the same code, and launch() is the same for both.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace packs = hip_kernel_provider::kernel_ingestor_engine::testing;

constexpr int64_t INPUT_UID = 11;
constexpr int64_t RECORD_UID = 13;
constexpr const char* SYMBOL = "DeclarativeLaunchRecord";

/// Written to the last word of the dynamic LDS the launch asks for and read back. LDS
/// bounds-checks every access against the workgroup's allocation: a launch with no
/// dynamic LDS drops the write and reads back zero.
constexpr uint32_t LDS_SENTINEL = 0x5A17C0DEU;

/// The slots of the record the kernel writes, one 64-bit word each.
enum RecordSlot : size_t
{
    GRID_X,
    GRID_Y,
    GRID_Z,
    BLOCK_X,
    BLOCK_Y,
    BLOCK_Z,
    LDS_READBACK,
    INPUT_ADDRESS,
    RECORD_ADDRESS,
    WORKSPACE_ADDRESS,
    COUNT_I32,
    FLAGS_U32,
    STRIDE_I64,
    SCALE_F32_BITS,
    RECORD_SLOTS
};

// Only the first thread of the first workgroup writes, so the record holds one launch's
// view. The LDS probe is volatile so the compiler keeps the round trip through LDS.
constexpr const char* RECORD_KERNEL_SOURCE = R"(
extern "C" __global__ void DeclarativeLaunchRecord(const void* input,
                                                   unsigned long long* record,
                                                   void* workspace,
                                                   int count,
                                                   unsigned int flags,
                                                   long long stride,
                                                   float scale)
{
    extern __shared__ unsigned int dynamicLds[];
    if(threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0 || blockIdx.x != 0
       || blockIdx.y != 0 || blockIdx.z != 0)
    {
        return;
    }
    volatile unsigned int* lds = dynamicLds;
    lds[LDS_PROBE_WORD] = LDS_SENTINEL;

    record[0] = gridDim.x;
    record[1] = gridDim.y;
    record[2] = gridDim.z;
    record[3] = blockDim.x;
    record[4] = blockDim.y;
    record[5] = blockDim.z;
    record[6] = lds[LDS_PROBE_WORD];
    record[7] = reinterpret_cast<unsigned long long>(input);
    record[8] = reinterpret_cast<unsigned long long>(record);
    record[9] = reinterpret_cast<unsigned long long>(workspace);
    record[10] = static_cast<unsigned long long>(static_cast<long long>(count));
    record[11] = flags;
    record[12] = static_cast<unsigned long long>(stride);
    union
    {
        float value;
        unsigned int bits;
    } scaleBits;
    scaleBits.value = scale;
    record[13] = scaleBits.bits;
}
)";

// The launch. Each extent is distinct from every other: grid 3 x 5 x 7 and block 8 x 4 x 2,
// so any two swapped axes differ. LDS is 8 * 4 * 64 = 2048 bytes. The u32 scalar is above
// INT32_MAX and the i64 stride above UINT32_MAX, so a narrowed or sign-changed value
// differs too.
constexpr const char* RECORD_UDD = R"({"version": "1.0",
    "id": "00000000-0000-0000-0000-00000000d0d0", "name": "declarative.record",
    "grid": {"x": {"ceil_div": ["$x.dims[3]", "$kernel.tile[0]"]},
             "y": "$x.dims[1]", "z": "$x.dims[2]"},
    "block": {"x": "$kernel.tile[0]", "y": "$kernel.tile[1]", "z": 2},
    "shared_mem_bytes": {"*": ["$kernel.tile[0]", "$kernel.tile[1]", 64]},
    "workspace_bytes": {"*": ["$x.dims[0]", 256]},
    "args_signature": [
        {"name": "input", "kind": "pointer", "source": {"from": "tensor", "ref": "$x"}},
        {"name": "record", "kind": "pointer", "source": {"from": "tensor", "ref": "$y"}},
        {"name": "workspace", "kind": "workspace"},
        {"name": "count", "kind": "scalar", "type": "i32",
         "source": {"from": "dim", "ref": "$x", "axis": 0}},
        {"name": "flags", "kind": "scalar", "type": "u32",
         "source": {"from": "expr", "expr": {"+": [3000000000, "$x.dims[1]"]}}},
        {"name": "stride", "kind": "scalar", "type": "i64",
         "source": {"from": "stride", "ref": "$x", "axis": 1}},
        {"name": "scale", "kind": "scalar", "type": "f32",
         "source": {"from": "expr", "expr": {"/": ["$x.dims[1]", 2]}}}]})";

constexpr std::array<uint32_t, 3> EXPECTED_GRID{3, 5, 7};
constexpr std::array<uint32_t, 3> EXPECTED_BLOCK{8, 4, 2};
constexpr uint32_t EXPECTED_LDS_BYTES = 2048;
constexpr size_t EXPECTED_WORKSPACE_BYTES = 512;
constexpr int64_t X_STRIDE_C = 5000000000LL;

/// Publishes a tensor the way publishTensor() names its tokens.
void publish(BoundTokens& bound,
             const std::string& root,
             int64_t uid,
             const std::vector<int64_t>& dims,
             const std::vector<int64_t>& strides)
{
    bound[root] = uid;
    bound[tensorField(root, "uid")] = uid;
    for(size_t axis = 0; axis < dims.size(); ++axis)
    {
        bound[tensorElement(root, "dims", axis)] = dims[axis];
        bound[tensorElement(root, "strides", axis)] = strides[axis];
    }
}

/// x is 2 x 5 x 7 x 20. Only the stride the launch reads is unusual.
BoundTokens recordBinding()
{
    BoundTokens bound;
    publish(bound, "x", INPUT_UID, {2, 5, 7, 20}, {700, X_STRIDE_C, 20, 1});
    publish(bound, "y", RECORD_UID, {static_cast<int64_t>(RECORD_SLOTS)}, {1});
    return bound;
}

KernelDefinition recordKernel()
{
    KernelDefinition kernel;
    kernel.name = "declarative.record";
    kernel.metadata = {{"tile", std::vector<int64_t>{8, 4}}};
    return kernel;
}

/// The argument list the kernel is compiled with, as a packager records it.
std::vector<KernelArgument> recordSignature()
{
    return {KernelArgument{"global_buffer", 8, 0, ""},
            KernelArgument{"global_buffer", 8, 8, ""},
            KernelArgument{"global_buffer", 8, 16, ""},
            KernelArgument{"by_value", 4, 24, ""},
            KernelArgument{"by_value", 4, 28, ""},
            KernelArgument{"by_value", 8, 32, ""},
            KernelArgument{"by_value", 4, 40, ""}};
}

// Compiles RECORD_KERNEL_SOURCE for `arch`, the decorated gcnArchName.
//
// Uses fatal assertions: call through ASSERT_NO_FATAL_FAILURE.
void compileRecordKernel(const std::string& arch, std::vector<uint8_t>& codeObject)
{
    hiprtcProgram program = nullptr;
    ASSERT_EQ(
        HIPRTC_SUCCESS,
        hiprtcCreateProgram(
            &program, RECORD_KERNEL_SOURCE, "DeclarativeLaunchRecord.cpp", 0, nullptr, nullptr));
    const std::vector<std::string> options{
        "--offload-arch=" + arch,
        "-DLDS_PROBE_WORD=" + std::to_string((EXPECTED_LDS_BYTES / sizeof(uint32_t)) - 1),
        "-DLDS_SENTINEL=" + std::to_string(LDS_SENTINEL) + "u"};
    std::vector<const char*> optionPointers;
    optionPointers.reserve(options.size());
    for(const auto& option : options)
    {
        optionPointers.push_back(option.c_str());
    }

    const hiprtcResult compiled = hiprtcCompileProgram(
        program, static_cast<int>(optionPointers.size()), optionPointers.data());
    std::string log;
    size_t logSize = 0;
    if(hiprtcGetProgramLogSize(program, &logSize) == HIPRTC_SUCCESS && logSize > 1)
    {
        log.resize(logSize);
        static_cast<void>(hiprtcGetProgramLog(program, log.data()));
    }
    size_t codeSize = 0;
    if(compiled == HIPRTC_SUCCESS && hiprtcGetCodeSize(program, &codeSize) == HIPRTC_SUCCESS)
    {
        codeObject.resize(codeSize);
        if(hiprtcGetCode(program, reinterpret_cast<char*>(codeObject.data())) != HIPRTC_SUCCESS)
        {
            codeObject.clear();
        }
    }
    static_cast<void>(hiprtcDestroyProgram(&program));
    ASSERT_EQ(compiled, HIPRTC_SUCCESS) << log;
    ASSERT_FALSE(codeObject.empty()) << "HIPRTC returned no code object";
}

/// A device allocation of @p bytes, zeroed, freed on scope exit.
class DeviceBytes
{
public:
    explicit DeviceBytes(size_t bytes)
    {
        EXPECT_EQ(hipSuccess, hipMalloc(&_buffer, bytes));
        EXPECT_EQ(hipSuccess, hipMemset(_buffer, 0, bytes));
    }

    ~DeviceBytes()
    {
        static_cast<void>(hipFree(_buffer));
    }

    DeviceBytes(const DeviceBytes&) = delete;
    DeviceBytes& operator=(const DeviceBytes&) = delete;

    void* get() const
    {
        return _buffer;
    }

private:
    void* _buffer = nullptr;
};

template <typename T>
uint64_t bitsOf(T value)
{
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(value));
    return bits;
}

uint64_t addressOf(const void* pointer)
{
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pointer));
}

TEST(TestDeclarativeLaunchHandler, LaunchesTheResolvedGeometryLdsAndArguments)
{
    SKIP_IF_NO_DEVICES();

    int deviceOrdinal = 0;
    ASSERT_EQ(hipSuccess, hipGetDevice(&deviceOrdinal));
    const auto properties = packs::currentDeviceProperties();
    ASSERT_FALSE(properties.gcnArchName.empty());

    // Resolved over a real binding, then saved, as a plan built from a graph saves it.
    const auto compiled
        = compileLaunch(nlohmann::json::parse(RECORD_UDD), "declarative.record", "test UDD");
    const packs::GraphFixture fixture(packs::buildConvFwdGraph(), properties);
    const auto resolved = compiled->resolve(fixture.context(), recordBinding(), recordKernel());
    ASSERT_EQ(resolved.grid, EXPECTED_GRID);
    ASSERT_EQ(resolved.block, EXPECTED_BLOCK);
    ASSERT_EQ(resolved.sharedMemBytes, EXPECTED_LDS_BYTES);
    ASSERT_EQ(resolved.workspaceBytes, EXPECTED_WORKSPACE_BYTES);

    SavedKernelCode code;
    code.sourceKind = KernelSourceKind::KPACK;
    code.symbol = SYMBOL;
    code.target = std::string(hipdnn_plugin_sdk::stripArchFeatures(properties.gcnArchName));
    code.recordedSignature = recordSignature();
    ASSERT_NO_FATAL_FAILURE(compileRecordKernel(properties.gcnArchName, code.codeObject));

    registerNativeIngestorSymbols();
    const auto& handler
        = *DispatchRegistry<Handle>::resolve(std::string(DECLARATIVE_LAUNCH_SYMBOL));
    const SavedLaunchInputs inputs{std::string(DECLARATIVE_LAUNCH_SYMBOL_V1),
                                   saveResolvedLaunch(resolved)};
    const auto prepared = handler.restoreLaunch(inputs, std::move(code), deviceOrdinal);
    ASSERT_NE(prepared, nullptr);

    const DeviceBytes input(64);
    const DeviceBytes record(RECORD_SLOTS * sizeof(uint64_t));
    const DeviceBytes workspace(EXPECTED_WORKSPACE_BYTES);
    const std::array<hipdnnPluginDeviceBuffer_t, 2> buffers{
        hipdnnPluginDeviceBuffer_t{INPUT_UID, input.get()},
        hipdnnPluginDeviceBuffer_t{RECORD_UID, record.get()}};

    const Handle handle;
    ASSERT_NO_THROW(handler.launch(
        handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace.get()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(handle.getStream()));

    std::array<uint64_t, RECORD_SLOTS> seen{};
    ASSERT_EQ(hipSuccess,
              hipMemcpy(seen.data(), record.get(), sizeof(seen), hipMemcpyDeviceToHost));

    EXPECT_EQ(seen[GRID_X], EXPECTED_GRID[0]);
    EXPECT_EQ(seen[GRID_Y], EXPECTED_GRID[1]);
    EXPECT_EQ(seen[GRID_Z], EXPECTED_GRID[2]);
    EXPECT_EQ(seen[BLOCK_X], EXPECTED_BLOCK[0]);
    EXPECT_EQ(seen[BLOCK_Y], EXPECTED_BLOCK[1]);
    EXPECT_EQ(seen[BLOCK_Z], EXPECTED_BLOCK[2]);
    EXPECT_EQ(seen[LDS_READBACK], LDS_SENTINEL)
        << "the last word of the requested dynamic LDS did not hold what was written";
    EXPECT_EQ(seen[INPUT_ADDRESS], addressOf(input.get()));
    EXPECT_EQ(seen[RECORD_ADDRESS], addressOf(record.get()));
    EXPECT_EQ(seen[WORKSPACE_ADDRESS], addressOf(workspace.get()));
    EXPECT_EQ(seen[COUNT_I32], bitsOf(int64_t{2}));
    EXPECT_EQ(seen[FLAGS_U32], uint64_t{3000000005});
    EXPECT_EQ(seen[STRIDE_I64], bitsOf(X_STRIDE_C));
    EXPECT_EQ(seen[SCALE_F32_BITS], bitsOf(2.5F));
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
