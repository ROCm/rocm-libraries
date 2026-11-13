// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaTestGraphs.hpp"

/**
 * @file FlydslSdpaObjectTestSupport.hpp
 * @brief The SDPA device harnesses, reached with a named object, the split workspace both
 *        harnesses launch with, and what the per-object host walks read of a loaded object.
 *
 * The RDNA and CDNA dispatch suites each own a launch-and-compare harness that picks the
 * object to run itself. The per-object suite (TestFlydslSdpaObjects.cpp) names the object
 * instead, so these entry points run a given staged object through the same harness: the
 * graph @p spec describes is matched, the object's own kernel matcher must admit it, and
 * every output element (and the LSE when @p spec asks for one) is checked against the
 * harness's double-precision reference. Declared here, defined beside each harness.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// A split launch's workspace as a caller allocates it: exactly the bytes the engine's
/// workspaceBytes() reports, then a guard band holding a canary. The workspace comes from
/// the caller, so a launch that writes past the size the engine reported writes into the
/// caller's memory; here it changes the guard, which overwrittenGuardBytes() counts.
class GuardedWorkspace
{
public:
    /// Far past any shortfall a case's shapes could have (a stat per split row).
    static constexpr size_t GUARD_BYTES = size_t{1} << 20;
    static constexpr unsigned char CANARY = 0xA5;

    explicit GuardedWorkspace(size_t bytes)
        : _bytes(bytes)
    {
        if(bytes == 0)
        {
            return;
        }
        EXPECT_EQ(hipSuccess, hipMalloc(&_base, bytes + GUARD_BYTES));
        EXPECT_EQ(hipSuccess, hipMemset(static_cast<char*>(_base) + bytes, CANARY, GUARD_BYTES));
        EXPECT_EQ(hipSuccess, hipDeviceSynchronize());
    }
    ~GuardedWorkspace()
    {
        static_cast<void>(hipFree(_base));
    }
    GuardedWorkspace(const GuardedWorkspace&) = delete;
    GuardedWorkspace& operator=(const GuardedWorkspace&) = delete;
    GuardedWorkspace(GuardedWorkspace&&) = delete;
    GuardedWorkspace& operator=(GuardedWorkspace&&) = delete;

    /// What the launch is given: null when the engine reported no workspace.
    void* get() const
    {
        return _base;
    }

    /// The guard bytes no longer holding the canary. Call after the launch has finished.
    size_t overwrittenGuardBytes() const
    {
        if(_base == nullptr)
        {
            return 0;
        }
        std::vector<unsigned char> guard(GUARD_BYTES);
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(guard.data(),
                            static_cast<const char*>(_base) + _bytes,
                            GUARD_BYTES,
                            hipMemcpyDeviceToHost));
        size_t overwritten = 0;
        for(const auto byte : guard)
        {
            overwritten += byte == CANARY ? 0 : 1;
        }
        return overwritten;
    }

    /// When @p zeroBias, fills the workspace itself with the canary, so the zero bias the
    /// launch must write at its end cannot read as written by chance. Call before the launch.
    void poisonForTheZeroBias(bool zeroBias) const
    {
        if(zeroBias && _base != nullptr)
        {
            EXPECT_EQ(hipSuccess, hipMemset(_base, CANARY, _bytes));
            EXPECT_EQ(hipSuccess, hipDeviceSynchronize());
        }
    }

    /// Fills the workspace itself (not the guard) with @p byte, so a launch that reads a
    /// workspace byte it did not write first gives an answer that depends on the fill. Call
    /// before the launch.
    void fill(unsigned char byte) const
    {
        if(_base != nullptr)
        {
            EXPECT_EQ(hipSuccess, hipMemset(_base, byte, _bytes));
            EXPECT_EQ(hipSuccess, hipDeviceSynchronize());
        }
    }

    /// When @p zeroBias, that the launch zeroed the workspace's last @p tailBytes: the zero
    /// bias an engine stages at the workspace's end (FlydslSdpaShared.hpp zeroBiasBytes),
    /// after what it writes itself. Call after the launch has finished.
    void expectTheZeroBiasAtTheEnd(bool zeroBias, size_t tailBytes) const
    {
        if(zeroBias)
        {
            EXPECT_EQ(nonZeroTailBytes(tailBytes), 0U)
                << "the zero bias is not at the end of the " << _bytes << "-byte workspace";
        }
    }

    /// The bytes of the workspace's last @p tailBytes that are not zero. Call after the launch
    /// has finished.
    size_t nonZeroTailBytes(size_t tailBytes) const
    {
        if(_base == nullptr || tailBytes > _bytes)
        {
            return tailBytes;
        }
        std::vector<unsigned char> tail(tailBytes);
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(tail.data(),
                            static_cast<const char*>(_base) + (_bytes - tailBytes),
                            tailBytes,
                            hipMemcpyDeviceToHost));
        size_t nonZero = 0;
        for(const auto byte : tail)
        {
            nonZero += byte == 0 ? 0 : 1;
        }
        return nonZero;
    }

private:
    size_t _bytes = 0;
    void* _base = nullptr;
};

/// A device buffer of exactly the bytes asked for that ends where its mapping ends: the
/// following page is reserved and never mapped, so a kernel that reads or writes one element
/// past the buffer takes a page fault (which aborts the process) instead of reading a
/// neighbour's memory. Built with HIP's virtual memory management; on a device without it the
/// buffer is a plain hipMalloc and guarded() is false.
class EndGuardedAllocation
{
public:
    explicit EndGuardedAllocation(size_t bytes)
    {
        int device = 0;
        int supported = 0;
        EXPECT_EQ(hipSuccess, hipGetDevice(&device));
        static_cast<void>(hipDeviceGetAttribute(
            &supported, hipDeviceAttributeVirtualMemoryManagementSupported, device));
        if(supported == 0 || bytes == 0)
        {
            EXPECT_EQ(hipSuccess, hipMalloc(&_ptr, std::max<size_t>(bytes, 1)));
            return;
        }
        hipMemAllocationProp prop{};
        prop.type = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id = device;
        size_t granularity = 0;
        EXPECT_EQ(hipSuccess,
                  hipMemGetAllocationGranularity(
                      &granularity, &prop, hipMemAllocationGranularityMinimum));
        granularity = std::max<size_t>(granularity, 1);
        _mapped = (bytes + granularity - 1) / granularity * granularity;
        _reserved = _mapped + granularity;
        EXPECT_EQ(hipSuccess, hipMemAddressReserve(&_base, _reserved, 0, nullptr, 0));
        EXPECT_EQ(hipSuccess, hipMemCreate(&_handle, _mapped, &prop, 0));
        EXPECT_EQ(hipSuccess, hipMemMap(_base, _mapped, 0, _handle, 0));
        hipMemAccessDesc access{};
        access.location = prop.location;
        access.flags = hipMemAccessFlagsProtReadWrite;
        EXPECT_EQ(hipSuccess, hipMemSetAccess(_base, _mapped, &access, 1));
        _ptr = static_cast<char*>(_base) + (_mapped - bytes);
    }
    ~EndGuardedAllocation()
    {
        if(_base == nullptr)
        {
            static_cast<void>(hipFree(_ptr));
            return;
        }
        static_cast<void>(hipDeviceSynchronize());
        static_cast<void>(hipMemUnmap(_base, _mapped));
        static_cast<void>(hipMemRelease(_handle));
        static_cast<void>(hipMemAddressFree(_base, _reserved));
    }
    EndGuardedAllocation(const EndGuardedAllocation&) = delete;
    EndGuardedAllocation& operator=(const EndGuardedAllocation&) = delete;
    EndGuardedAllocation(EndGuardedAllocation&&) = delete;
    EndGuardedAllocation& operator=(EndGuardedAllocation&&) = delete;

    void* get() const
    {
        return _ptr;
    }

    /// Whether the page past the buffer is unmapped (the device supports VMM).
    bool guarded() const
    {
        return _base != nullptr;
    }

private:
    void* _ptr = nullptr;
    void* _base = nullptr;
    size_t _mapped = 0;
    size_t _reserved = 0;
    hipMemGenericAllocationHandle_t _handle{};
};

/// Whether an EndGuardedAllocation on the current device is guarded(): false with no device or
/// without virtual memory management, where a test of what a kernel reads past a buffer has no
/// unmapped page to fault on and skips (as TestGpuFpReferenceSdpa.cpp's guard tests do).
inline bool endGuardedAllocationsAreGuarded()
{
    int count = 0;
    if(hipGetDeviceCount(&count) != hipSuccess || count == 0)
    {
        return false;
    }
    const EndGuardedAllocation probe(1);
    return probe.guarded();
}

/// The output and workspace fill of a repeated launch (a determinism check): NaN, zero and a
/// large finite value in every element width, so a launch that reads an element it did not
/// write gives an answer that depends on which launch it is.
inline unsigned char repeatLaunchFill(int launch)
{
    static constexpr std::array<unsigned char, 3> FILLS{0xFF, 0x00, 0x7F};
    return FILLS[static_cast<size_t>(launch) % FILLS.size()];
}

/// How many elements of @p again differ, bit for bit, from @p first's, and the first of them.
struct BitwiseDifference
{
    size_t count = 0;
    size_t first = 0;
};

template <typename T>
BitwiseDifference bitwiseDifference(const std::vector<T>& first, const std::vector<T>& again)
{
    BitwiseDifference difference;
    for(size_t i = 0; i < first.size() && i < again.size(); ++i)
    {
        // Compared as bytes copied out, not through memcmp on T: a float's object
        // representation is not unique (bugprone-suspicious-memory-comparison), and the bits
        // are what must match.
        std::array<unsigned char, sizeof(T)> firstBytes{};
        std::array<unsigned char, sizeof(T)> againBytes{};
        std::memcpy(firstBytes.data(), &first[i], sizeof(T));
        std::memcpy(againBytes.data(), &again[i], sizeof(T));
        if(firstBytes != againBytes)
        {
            difference.first = difference.count == 0 ? i : difference.first;
            ++difference.count;
        }
    }
    return difference;
}

/// The kernel @p kernel of @p pack as the engine's matchers read it, promoted from the loaded
/// descriptors. A kernel may narrow the arch list its pack claims but never reach outside it,
/// so an empty list on the kernel means "the pack's".
template <typename Pack, typename Kernel>
hipdnn_plugin_sdk::ingestor::KernelDefinition definitionOf(const Pack& pack, const Kernel& kernel)
{
    return hipdnn_plugin_sdk::ingestor::KernelDefinition{kernel.id,
                                                         pack.id,
                                                         pack.dispatchId,
                                                         kernel.source,
                                                         kernel.metadata,
                                                         kernel.priority,
                                                         kernel.arch.empty() ? pack.arch
                                                                             : kernel.arch,
                                                         kernel.originDirectory,
                                                         kernel.name,
                                                         kernel.treeRoot,
                                                         kernel.sources,
                                                         nullptr,
                                                         nullptr};
}

/// A metadata field of type @p T, or nullptr when absent or of another type.
template <typename T>
const T* metadataValue(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                       const char* field)
{
    const auto it = kernel.metadata.find(std::string(field));
    return it == kernel.metadata.end() ? nullptr : std::get_if<T>(&it->second);
}

/// A metadata integer, or @p absent when the object's KMD has no such field (each family
/// leaves out the fields its kernel fixes).
inline int64_t intField(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                        const char* field,
                        int64_t absent = 0)
{
    const auto* value = metadataValue<int64_t>(kernel, field);
    return value == nullptr ? absent : *value;
}

inline std::string stringField(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                               const char* field)
{
    const auto* value = metadataValue<std::string>(kernel, field);
    return value == nullptr ? std::string() : *value;
}

/// The device a host check stands in for when it matches an object of another arch.
inline hipdnn_plugin_sdk::ingestor::DeviceProperties devicePropertiesFor(const std::string& arch)
{
    hipdnn_plugin_sdk::ingestor::DeviceProperties properties;
    properties.gcnArchName = arch;
    properties.warpSize = arch.rfind("gfx9", 0) == 0 ? 64 : 32;
    properties.multiProcessorCount = 64;
    return properties;
}

/// The RDNA engine a forced object belongs to.
enum class RdnaObjectEngine
{
    PREFILL,
    DECODE,
    AIW,
};

/// Runs @p kernel, an object of @p engine, on the graph @p spec describes and checks it
/// against the RDNA harness's reference at the dispatch suite's tolerance for the dtype.
/// Defined in TestFlydslSdpaRdnaEngine.cpp.
void expectRdnaObjectMatchesReference(const SdpaGraphSpec& spec,
                                      RdnaObjectEngine engine,
                                      const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

/// Runs @p kernel, an object of any of the three CDNA engines, on the graph @p spec
/// describes and checks it against the CDNA harness's reference. Defined in
/// TestFlydslSdpaCdnaEngine.cpp.
void expectCdnaObjectMatchesReference(const SdpaGraphSpec& spec,
                                      const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
