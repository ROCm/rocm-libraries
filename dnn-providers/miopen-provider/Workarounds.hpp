// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Production-side workarounds for known MIOpen-provider issues. Each macro is
// keyed by the upstream issue number so it is easy to grep and remove once the
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
// To remove after the 5409 fix: delete its block and the REJECT_IF_WORKAROUND_ISSUE_5409
// call sites; delete this file only once no other workaround remains. The test-side
// 6979 helpers stay until that issue is independently resolved.
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
