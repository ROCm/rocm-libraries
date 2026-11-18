// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Production-side workarounds for known MIOpen-provider issues. Each macro is
// keyed by its upstream issue so it is easy to grep and remove once the
// underlying problem is fixed. Test-side counterparts live in
// `tests/common/TestWorkarounds.hpp` (kept separate so production TUs never
// pull in gtest).
//
// ----------------------------------------------------------------------------
// ROCm/rocm-libraries#5409 — MIOpen Conv-Bias-Activation (CBA) fusion produces
// incorrect/unsupported results on gfx90a (also covers 2-node CB and CA paths
// because all conv-fusion variants route through MiopenConvFwdBiasActivPlan-
// Builder). Until upstream fix lands, we early-return `false` from the plan
// builder's isApplicable() so engine selection skips the MIOpen CBA path on
// gfx90a.
//
// REJECT_IF_WORKAROUND_ISSUE_5409(handle) must only be invoked from a function
// whose return type is `bool` (or convertible from `false`) since it contains
// a `return`. CBA-specific test TUs observe the resulting "no engine applicable"
// outcome via the test-side issue 6979 helpers (see TestWorkarounds.hpp).
//
// Issue 5409 (this file) is gfx90a-specific. Issue 6979 (test-side) tracks the
// broader CBA capability gap across architectures — gfx90a (covered by 5409)
// plus gfx1151 / RDNA archs that lack CK fusion kernels. The two are decoupled:
// fixing 5409 only removes the gfx90a REJECT here; the 6979 test-side skip
// stays in place until CK fusion kernels exist for the remaining archs.
//
// To remove after the 5409 fix: delete this section and its macro, drop the
// include from MiopenConvFwdBiasActivPlanBuilder.cpp, and remove the
// REJECT_IF_WORKAROUND_ISSUE_5409 call sites; delete this file only once no
// other workaround remains. The test-side 6979 helpers stay until that issue is
// independently resolved.
// `git grep WORKAROUND_ISSUE_5409` finds them all.
// ----------------------------------------------------------------------------
// ROCm/rocm-libraries#16024 — MIOpen's forward activation reads a pair of
// tensors that is not both packed through its strided addressing only when the
// N stride exceeds C*H*W. Its network config for a non-packed problem carries
// neither the strides nor the solver id, and a cached invoker is reused for a
// later problem without an applicability check, so one problem can run another
// problem's kernel. The unary activation applicability check therefore
// declines every pair that is not both packed. Applies on all arches (no
// gating).
//
// REJECT_IF_WORKAROUND_ISSUE_16024(inputTensor, outputTensor) takes the two
// flatbuffer TensorAttributes of the activation and must only be invoked from
// a function whose return type is `bool` since it contains a `return`. It runs
// after the generic tensor checks, which already guarantee equal dims, and
// equal strides on non-unit axes for a packed pair; so "both packed" is
// exactly "packed with one layout".
//
// To remove after the 16024 fix: delete the REJECT_IF_WORKAROUND_ISSUE_16024
// call site and this block. The two-dimensional-shortcut and rank-1 blocks in
// MiopenUnaryActivationChecks.cpp remain the guards for ActivFwdSolver0.
// `git grep WORKAROUND_ISSUE_16024` finds them all.
// ----------------------------------------------------------------------------

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/DeviceQuery.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include <cstdint>
#include <exception>
#include <limits>
#include <vector>

#define REJECT_IF_WORKAROUND_ISSUE_5409(handle)                                            \
    do                                                                                     \
    {                                                                                      \
        try                                                                                \
        {                                                                                  \
            if(::hipdnn_plugin_sdk::archMatches(                                           \
                   ::hipdnn_plugin_sdk::getDeviceArch((handle).getStream()),               \
                   "gfx90a",                                                               \
                   ::hipdnn_plugin_sdk::ArchMatchMode::PREFIX))                            \
            {                                                                              \
                HIPDNN_PLUGIN_LOG_INFO("Plan builder disabled on gfx90a (issue 5409)");    \
                return false;                                                              \
            }                                                                              \
        }                                                                                  \
        catch(const std::exception& issue_5409_e)                                          \
        {                                                                                  \
            HIPDNN_PLUGIN_LOG_INFO(                                                        \
                "Arch query failed; treating as not-applicable: " << issue_5409_e.what()); \
            return false;                                                                  \
        }                                                                                  \
    } while(0)

#define REJECT_IF_WORKAROUND_ISSUE_16024(inputTensor, outputTensor)                          \
    do                                                                                       \
    {                                                                                        \
        const std::vector<int64_t> issue_16024_inDims((inputTensor).dims()->begin(),         \
                                                      (inputTensor).dims()->end());          \
        const std::vector<int64_t> issue_16024_inStrides((inputTensor).strides()->begin(),   \
                                                         (inputTensor).strides()->end());    \
        const std::vector<int64_t> issue_16024_outDims((outputTensor).dims()->begin(),       \
                                                       (outputTensor).dims()->end());        \
        const std::vector<int64_t> issue_16024_outStrides((outputTensor).strides()->begin(), \
                                                          (outputTensor).strides()->end());  \
        if(!::hipdnn_data_sdk::utilities::isTensorPacked(issue_16024_inDims,                 \
                                                         issue_16024_inStrides)              \
           || !::hipdnn_data_sdk::utilities::isTensorPacked(issue_16024_outDims,             \
                                                            issue_16024_outStrides))         \
        {                                                                                    \
            HIPDNN_PLUGIN_LOG_INFO(                                                          \
                "Unary activation plan builder declines a pair that is not both packed "     \
                "(issue 16024)");                                                            \
            return false;                                                                    \
        }                                                                                    \
    } while(0)

// ----------------------------------------------------------------------------
// ALMIOPEN-3295 — MIOpen's tiled pooling forward kernels (MIOpenPooling.cpp,
// MIOpenPoolingND.cpp) pad BFLOAT16 max pooling with 0xFF7F, which is above the
// -FLT_MAX their accumulator starts at. A window whose in-bounds inputs are all
// -inf, NaN or the lowest BFLOAT16 value therefore selects a padded position,
// and the index names it: a negative offset or one that aliases another element.
// Pooling backward would read that index.
//
// WORKAROUND_ALMIOPEN_3295_DECLINES_INDEX(dataType, hasPadding) is true for the
// graphs the resample forward checks decline until the fix lands: BFLOAT16 max
// pooling that requests an index with any non-zero padding. Without padding, or
// without an index, no index can name a padded position.
//
// To remove after the fix: delete this section and its macro, its call site in
// engines/plans/MiopenResampleFwdChecks.cpp (and that file's
// `#include "Workarounds.hpp"` once no workaround there remains), and the decline
// test tagged ALMIOPEN-3295 in tests/engines/plans/TestMiopenResampleFwdChecks.cpp;
// drop the BFLOAT16 index rule from the Resample Forward notes in
// docs/OperationSupport.md and
// projects/hipdnn/docs/user-guides/reference/miopen-plugin.rst. `git grep
// ALMIOPEN.3295` finds the code sites.
// ----------------------------------------------------------------------------

#define WORKAROUND_ALMIOPEN_3295_DECLINES_INDEX(dataType, hasPadding) \
    ((dataType) == ::hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16 && (hasPadding))

// ----------------------------------------------------------------------------
// ALMIOPEN-3373 — MIOpen's pooling forward kernels (MIOpenPoolingForwardNaive.cpp,
// and the tiled MIOpenPooling.cpp and MIOpenPoolingND.cpp) compute each window's
// end, start + window, in signed 32-bit arithmetic. The last window of a spatial
// dimension ends at input extent + padding; above INT_MAX the end wraps negative,
// the clamp to the input is lost and the kernel reads past X: a memory fault, or
// a wrong average.
//
// WORKAROUND_ALMIOPEN_3373_DECLINES_EXTENT(inputExtent, padding) is true for the
// spatial dimensions the resample forward checks decline until the fix lands:
// input extent + padding above INT_MAX. Every window end then fits in int.
//
// To remove after the fix: delete this section and its macro, its call site in
// engines/plans/MiopenResampleFwdChecks.cpp (and that file's
// `#include "Workarounds.hpp"` once no workaround there remains), and the tests
// tagged ALMIOPEN-3373 in tests/engines/plans/TestMiopenResampleFwdChecks.cpp;
// drop the extent rule from the Resample Forward notes in
// docs/OperationSupport.md and
// projects/hipdnn/docs/user-guides/reference/miopen-plugin.rst. `git grep
// ALMIOPEN.3373` finds the code sites.
// ----------------------------------------------------------------------------

#define WORKAROUND_ALMIOPEN_3373_DECLINES_EXTENT(inputExtent, padding) \
    (static_cast<int64_t>(inputExtent) + static_cast<int64_t>(padding) \
     > static_cast<int64_t>(::std::numeric_limits<int>::max()))

// ----------------------------------------------------------------------------
// ALMIOPEN-3375 — for average pooling that includes padding, MIOpen's naive
// pooling forward kernel (MIOpenPoolingForwardNaive.cpp) divides by the window
// volume computed as an unsigned 32-bit product. A volume of 2^32 or more wraps,
// so the divisor is 0 (Y is inf) or a small residue (Y is too large). Windows
// that large fit only the naive kernel, so MIOpen selects it.
//
// WORKAROUND_ALMIOPEN_3375_DECLINES_WINDOW(mode, window) is true for the graphs
// the resample forward checks decline until the fix lands: AVGPOOL_INCLUDE_PADDING
// with a window volume above UINT32_MAX. Max pooling and average pooling that
// excludes padding do not use that product.
//
// To remove after the fix: delete this section, its helper and its macro, its
// call site in engines/plans/MiopenResampleFwdChecks.cpp (and that file's
// `#include "Workarounds.hpp"` once no workaround there remains), and the tests
// tagged ALMIOPEN-3375 in tests/engines/plans/TestMiopenResampleFwdChecks.cpp;
// drop the window-volume rule from the Resample Forward notes in
// docs/OperationSupport.md and
// projects/hipdnn/docs/user-guides/reference/miopen-plugin.rst. `git grep
// ALMIOPEN.3375` finds the code sites.
// ----------------------------------------------------------------------------

namespace miopen_plugin::workaround_almiopen_3375
{

// True when the product of the window values exceeds UINT32_MAX. Every value is in [1, INT_MAX],
// so the running product stays below 2^63 until it passes the limit.
inline bool windowVolumeExceedsUint32(const std::vector<int>& window)
{
    uint64_t volume = 1;
    for(const int value : window)
    {
        volume *= static_cast<uint64_t>(value);
        if(volume > ::std::numeric_limits<uint32_t>::max())
        {
            return true;
        }
    }
    return false;
}

} // namespace miopen_plugin::workaround_almiopen_3375

#define WORKAROUND_ALMIOPEN_3375_DECLINES_WINDOW(mode, window)                               \
    ((mode) == ::hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_INCLUDE_PADDING \
     && ::miopen_plugin::workaround_almiopen_3375::windowVolumeExceedsUint32(window))
