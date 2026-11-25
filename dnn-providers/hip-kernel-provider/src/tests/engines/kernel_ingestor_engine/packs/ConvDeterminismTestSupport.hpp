// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

/**
 * @file ConvDeterminismTestSupport.hpp
 * @brief The bitwise-repeat check behind the conv engines' determinism claim.
 *
 * Every hipkernel:CONV_* engine's UED authors a `numerical_notes` list without
 * `nondeterministic` (RFC 0025), so hipDNN reports the engine as bitwise reproducible. RFC 0025
 * admits that claim only when a test proves it: each family's device test calls
 * relaunchReproducesOutput() for every config it runs.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_determinism_test
{

/// Launches a plan a second time and checks that its output comes out bitwise equal to the first
/// run's. Call it after the first launch has completed and before the output is read back. The
/// workspace, if any, is filled with a byte pattern unlike its first-run contents before the
/// second launch, so a result that depends on stale workspace bytes shows as a difference.
inline ::testing::AssertionResult
    relaunchReproducesOutput(const std::function<void()>& launch,
                             hipStream_t stream,
                             hipdnn_data_sdk::utilities::ITensor& output,
                             void* workspace,
                             size_t workspaceBytes)
{
    const size_t bytes = output.elementSpace() * output.elementSize();
    const void* device = output.rawDeviceData();
    std::vector<unsigned char> first(bytes);
    if(hipMemcpy(first.data(), device, bytes, hipMemcpyDeviceToHost) != hipSuccess)
    {
        return ::testing::AssertionFailure() << "copying the first run's output failed";
    }
    if(workspace != nullptr && workspaceBytes > 0
       && (hipMemset(workspace, 0x5a, workspaceBytes) != hipSuccess
           || hipDeviceSynchronize() != hipSuccess))
    {
        return ::testing::AssertionFailure() << "re-filling the workspace failed";
    }
    launch();
    if(hipStreamSynchronize(stream) != hipSuccess)
    {
        return ::testing::AssertionFailure() << "the second launch failed";
    }
    std::vector<unsigned char> second(bytes);
    if(hipMemcpy(second.data(), device, bytes, hipMemcpyDeviceToHost) != hipSuccess)
    {
        return ::testing::AssertionFailure() << "copying the second run's output failed";
    }
    for(size_t i = 0; i < bytes; ++i)
    {
        if(first[i] != second[i])
        {
            return ::testing::AssertionFailure()
                   << "a second launch changed output byte " << i << " of " << bytes;
        }
    }
    return ::testing::AssertionSuccess();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_determinism_test
