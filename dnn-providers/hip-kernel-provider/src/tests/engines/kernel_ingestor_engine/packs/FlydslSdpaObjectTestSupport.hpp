// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaTestGraphs.hpp"

/**
 * @file FlydslSdpaObjectTestSupport.hpp
 * @brief The SDPA device harnesses, reached with a named object, and the split workspace
 *        both harnesses launch with.
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
