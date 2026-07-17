// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "plans/RockeClientPlan.hpp"
#include "dispatcher/KpackModuleLoader.hpp"
#include "dispatcher/SdpaGraphAdapter.hpp"
#include "plans/LaunchAbi.hpp"
#include "plans/PluginError.hpp"

#include <rocm_kpack/kpack.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocke_client
{
namespace
{

namespace fb = hipdnn_flatbuffers_sdk::flatbuffer_utilities;

void checkHip(hipError_t status, const char* call)
{
    if(status != hipSuccess)
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         std::string(call) + " failed: " + hipGetErrorString(status));
    }
}

void checkKpack(kpack_error_t status, const char* call)
{
    if(status != KPACK_SUCCESS)
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         std::string(call) + " failed with kpack_error_t "
                             + std::to_string(static_cast<int>(status)));
    }
}

// Makes the given device current for its lifetime and restores the previous
// device on scope exit. A HIP module (and the hipFunction_t derived from it) is
// bound to the device that was current when it was loaded, so both the load and
// every launch must run with the handle stream's device current -- which the
// dispatcher already proves need not be the thread-current device.
class ScopedDevice
{
public:
    explicit ScopedDevice(int device)
    {
        checkHip(hipGetDevice(&_previous), "hipGetDevice");
        if(device != _previous)
        {
            checkHip(hipSetDevice(device), "hipSetDevice");
            _restore = true;
        }
    }

    ~ScopedDevice()
    {
        if(_restore)
        {
            // Best-effort restore; a destructor cannot surface a plugin status.
            static_cast<void>(hipSetDevice(_previous));
        }
    }

    ScopedDevice(const ScopedDevice&) = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;
    ScopedDevice(ScopedDevice&&) = delete;
    ScopedDevice& operator=(ScopedDevice&&) = delete;

private:
    int _previous = 0;
    bool _restore = false;
};

std::unordered_map<std::int64_t, void*>
    makeDeviceBufferMap(const hipdnnPluginDeviceBuffer_t* deviceBuffers, uint32_t numDeviceBuffers)
{
    std::unordered_map<std::int64_t, void*> ptrs;
    ptrs.reserve(numDeviceBuffers);
    for(uint32_t index = 0; index < numDeviceBuffers; ++index)
    {
        ptrs[deviceBuffers[index].uid] = deviceBuffers[index].ptr;
    }
    return ptrs;
}

dispatcher::SdpaLaunchInputs buildLaunchInputsOrThrow(const fb::IGraph& graph)
{
    auto inputs = dispatcher::buildSdpaLaunchInputs(graph);
    if(!inputs.has_value())
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                         "rocke-client plan could not build launch inputs from the graph");
    }
    return *inputs;
}

} // namespace

void HipModule::reset(hipModule_t module) noexcept
{
    if(_module != nullptr)
    {
        const auto status = hipModuleUnload(_module);
        if(status != hipSuccess)
        {
            // Destructors cannot surface plugin status; preserve diagnostics on stderr.
            std::cerr << "rocke-client hipModuleUnload failed: " << hipGetErrorString(status)
                      << '\n';
        }
    }
    _module = module;
}

void HipDeviceBuffer::reset(void* ptr) noexcept
{
    if(_ptr != nullptr)
    {
        const auto status = hipFree(_ptr);
        if(status != hipSuccess)
        {
            // Destructors cannot surface plugin status; preserve diagnostics on stderr.
            std::cerr << "rocke-client hipFree failed: " << hipGetErrorString(status) << '\n';
        }
    }
    _ptr = ptr;
}

RockeClientPlan::RockeClientPlan(dispatcher::AotInstance instance,
                                 const fb::IGraph& graph,
                                 const RockeClientHandle& handle)
    : _instance(std::move(instance))
{
    // Decode the graph into op-agnostic launch bindings; the SDPA specifics stay
    // in the adapter and the plan holds only generic per-launch data. Grid symbols
    // are sourced from the selected instance's compile spec plus the runtime batch.
    auto inputs = buildLaunchInputsOrThrow(graph);
    _bindings = std::move(inputs.bindings);
    _gridSymbols = dispatcher::sdpaGridSymbols(_instance.compileSpec, inputs.batch);

    checkHip(hipStreamGetDevice(handle.getStream(), &_deviceId), "hipStreamGetDevice");
    const ScopedDevice deviceGuard(_deviceId);

    // Delegate archive open, HSACO extraction, module load and function lookup
    // to the shared kpack loader so kpack_* has a single reference site.
    const auto loaded = dispatcher::loadKernelFromKpack(_instance.runtime.kpackPath,
                                                        _instance.runtime.tocKey,
                                                        _instance.arch,
                                                        _instance.runtime.symbol);
    checkKpack(loaded.kpackError, "loadKernelFromKpack");
    checkHip(loaded.hipError, "loadKernelFromKpack");
    _module.reset(loaded.module);
    _function = loaded.fn;

    // Synthesize the paged-KV index buffers for this dense problem while the
    // handle stream's device is current (they are allocated on it and freed with
    // the plan). Their addresses and block_table_stride complete _bindings.
    buildPagedKvBuffers(inputs.batch);
}

RockeClientPlan::~RockeClientPlan() = default;

void RockeClientPlan::buildPagedKvBuffers(std::int64_t batch)
{
    const dispatcher::CompileSpec& spec = _instance.compileSpec;
    if(spec.blockSize <= 0)
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         "rocke-client compile_spec.block_size must be positive");
    }

    // The unified attention kernel is inherently causal, but hipDNN requested a
    // dense non-causal (mask_mode=none) problem. It is realized by presenting each
    // of the batch*seqlen_q query tokens as its own length-1 pseudo-sequence whose
    // KV context is the full seqlen_k keys of its parent batch: with context_len =
    // seqlen_k - 1 the single query sits at absolute position seqlen_k-1 and its
    // causal window [0, seqlen_k-1] covers every key (see SdpaGraphAdapter). The
    // synthesized index buffers below therefore describe `numSeqs` such sequences.
    const std::int64_t numSeqs = batch * spec.seqlenQ;

    // Number of paged blocks per sequence: ceil(seqlen_k / block_size). For the
    // shipped instances seqlen_k is a multiple of block_size, so the identity
    // paging below reproduces contiguous BSHD KV addressing exactly.
    const std::int64_t btStride = (spec.seqlenK + spec.blockSize - 1) / spec.blockSize;

    // seq_lens[p] = seqlen_k: every pseudo-sequence sees the full KV context.
    std::vector<std::int32_t> seqLens(static_cast<std::size_t>(numSeqs),
                                      static_cast<std::int32_t>(spec.seqlenK));

    // query_start_len[p] = cumulative query tokens (cu_seqlens_q). One query token
    // per pseudo-sequence gives the identity prefix sum [0, 1, 2, ..., numSeqs].
    std::vector<std::int32_t> queryStartLen(static_cast<std::size_t>(numSeqs) + 1);
    for(std::int64_t p = 0; p <= numSeqs; ++p)
    {
        queryStartLen[static_cast<std::size_t>(p)] = static_cast<std::int32_t>(p);
    }

    // block_tables[p, tile] maps pseudo-sequence p (query token p, belonging to
    // batch p/seqlen_q) onto its parent batch's contiguous KV blocks. Batch b owns
    // physical blocks [b*btStride, (b+1)*btStride); block b*btStride+tile holds KV
    // tokens [tile*block_size, (tile+1)*block_size) of a dense [B, S, Hkv, D] cache.
    std::vector<std::int32_t> blockTables(static_cast<std::size_t>(numSeqs * btStride));
    for(std::int64_t p = 0; p < numSeqs; ++p)
    {
        const std::int64_t batchIdx = p / spec.seqlenQ;
        for(std::int64_t tile = 0; tile < btStride; ++tile)
        {
            blockTables[static_cast<std::size_t>(p * btStride + tile)]
                = static_cast<std::int32_t>(batchIdx * btStride + tile);
        }
    }

    const auto upload
        = [this](const std::vector<std::int32_t>& host, const char* what) -> std::uint64_t {
        void* device = nullptr;
        const std::size_t bytes = host.size() * sizeof(std::int32_t);
        checkHip(hipMalloc(&device, bytes), what);
        _pagedBuffers.emplace_back(device);
        checkHip(hipMemcpy(device, host.data(), bytes, hipMemcpyHostToDevice), what);
        return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(device));
    };

    _bindings.pointerValues["block_tables_ptr"] = upload(blockTables, "hipMalloc block_tables");
    _bindings.pointerValues["seq_lens_ptr"] = upload(seqLens, "hipMalloc seq_lens");
    _bindings.pointerValues["query_start_len_ptr"]
        = upload(queryStartLen, "hipMalloc query_start_len");
    _bindings.scalars["block_table_stride"] = static_cast<std::int64_t>(btStride);
}

size_t RockeClientPlan::getWorkspaceSize(const RockeClientHandle& /*handle*/) const
{
    return 0;
}

void RockeClientPlan::execute(const RockeClientHandle& handle,
                              const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                              uint32_t numDeviceBuffers,
                              void* /*workspace*/) const
{
    if(deviceBuffers == nullptr)
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                         "rocke-client execute received null buffers");
    }

    const ScopedDevice deviceGuard(_deviceId);

    const auto ptrs = makeDeviceBufferMap(deviceBuffers, numDeviceBuffers);
    const auto argValues
        = launch::bindArgs(_instance.runtime.launch.argsSignature, _bindings, ptrs);
    auto packed = launch::packArgs(_instance.runtime.launch.argsSignature, argValues);
    auto argSize = packed.size();
    std::array<void*, 5> config = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                   packed.data(),
                                   HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                   &argSize,
                                   HIP_LAUNCH_PARAM_END};
    const auto grid = launch::evalGrid(_instance.runtime.launch.grid, _gridSymbols);
    const auto& block = _instance.runtime.launch.block;

    checkHip(
        hipModuleLaunchKernel(_function,
                              grid[0],
                              grid[1],
                              grid[2],
                              block[0],
                              block[1],
                              block[2],
                              static_cast<unsigned int>(_instance.runtime.launch.sharedMemBytes),
                              handle.getStream(),
                              nullptr,
                              config.data()),
        "hipModuleLaunchKernel");
}

} // namespace rocke_client
