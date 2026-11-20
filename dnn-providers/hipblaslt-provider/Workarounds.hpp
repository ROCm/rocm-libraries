// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Production-side workarounds for known hipBLASLt-provider issues. Each macro is
// keyed by the upstream issue number so it is easy to grep and remove once the
// underlying problem is fixed. Test-side counterparts live in
// `tests/TestWorkarounds.hpp` (kept separate so production TUs never pull in
// gtest).
//
// ----------------------------------------------------------------------------
// ROCm/rocm-libraries#9962 — hipBLASLt crashes (0xc0000005 access violation)
// while building a GEMM plan for some problem configs on the gfx115x on Windows.
// It faults inside hipBLASLt's heuristic instead of returning an error, so
// probing support (isApplicable, which constructs a plan) crashes the caller.
// Until the upstream fix lands we early-return `false` from the matmul plan
// builders' isApplicable() so engine selection skips the hipBLASLt matmul path.
//
// REJECT_IF_WORKAROUND_ISSUE_9962(handle) must only be invoked from a function
// whose return type is `bool` (it contains a `return`). The fault is only
// observed on Windows, so it is compile-time gated to Windows builds; elsewhere
// it expands to a no-op. An arch-query failure is fail-closed: declining reports
// the graph unsupported, guessing "not affected" on a real gfx115x crashes.
//
// To remove after the fix: `git grep WORKAROUND_ISSUE_9962` finds the macro, its
// test-side counterpart, and both call sites.
// ----------------------------------------------------------------------------
// ROCm/rocm-libraries#10811 — on gfx950 a block-scaled MX GEMM with FP8 OCP
// (E4M3 or E5M2) on graph operand A and FP6 (E2M3 or E3M2) on graph operand B
// returns incorrect results. Narrowed down with hipblaslt-bench (VEC32 scales,
// every heuristic solution run): hipBLASLt A = FP6 x B = FP8 with transB = T is
// wrong on the rocRoller kernels rr_32x32x64 and rr_16x16x128 (one of them is the
// heuristic's first choice); transB = N is correct. Graph A reaches hipBLASLt as
// B (row-major swap), with transB = T exactly when graph A is column-major, so
// only that layout is declined; a K-contiguous (row-major) A is unaffected.
//
// REJECT_IF_WORKAROUND_ISSUE_10811 must only be invoked from a function with
// return type `bool` (it contains a `return`).
//
// To remove after the fix: `git grep WORKAROUND_ISSUE_10811` finds the macro and its
// single call site. Also delete the test
// `TestGpuHipblasltMxMatmulPlanBuilder.IsApplicableRejectsFp8AWithFp6B` and add the
// four `MxMixedConfig<FP8_*, FP6_*>` pairings to `MxMixedConfigs` in that same file.
// ----------------------------------------------------------------------------
// ROCm/rocm-libraries#14286 — on gfx1250, hipBLASLt selects MX kernels without
// matching their scale layout to the scale mode. For VEC32_UE8M0 (documented as
// natural scales) it selects InMemorySwizzle Tensile kernels, whose layout no scale
// mode documents, and it would select a NoSwizzle kernel just the same. A kernel
// silently returns wrong results unless the scales match its layout. (On gfx950
// hipBLASLt refuses mismatched kernels since ROCm/rocm-libraries#14625.)
//
// WORKAROUND_ISSUE_14286 enables a guard in the MX plan: it reads the selected
// kernel's layout from its name (the `_MXSF` token, or rocRoller), declines when it
// does not match the scales the plan prepares, and feeds natural scales to a gfx1250
// NoSwizzle kernel.
//
// To remove once hipBLASLt matches the kernel's scale layout to the scale mode on
// gfx1250 too: `git grep WORKAROUND_ISSUE_14286` finds the define and its `#if` sites
// in HipblasltMxMatmulPlan.{hpp,cpp} and TestHipblasltMxMatmulPlan.cpp. Delete each
// `#if` block (keep the `#else` branch where there is one) and the define. That also
// deletes the `hipblaslt/hipblaslt-ext.hpp` include, the plugin's only import of
// hipblaslt_ext (getIndexFromAlgo, getKernelNameFromAlgo).
// ----------------------------------------------------------------------------
// ROCm/rocm-libraries#14290 — hipBLASLt ignores the leading dimensions of an MX
// GEMM: it reads A and B and writes C as packed matrices, so a padded one silently
// gives wrong results.
//
// REJECT_IF_WORKAROUND_ISSUE_14290(tensor, name) throws HipdnnPluginException
// (BAD_PARAM) when the row- or column-major matrix `tensor` has padded rows or
// columns. The MX builder applies it to A, B and C after checking that each is a
// row- or column-major matrix.
//
// To remove after the fix: `git grep WORKAROUND_ISSUE_14290` finds the macro and its
// three call sites. Padded A and B are then still declined in the plan by
// packedMxOrder (makeScaleOperand, HipblasltMxMatmulPlan.cpp), which also gives
// each operand's order; take the order from the contiguous axis there instead. The
// padded cases of IsApplicableRejectsNonPackedTensors,
// IsApplicableRejectsNonPackedTensorsNt, DeclinesOutputThatIsNotPackedRowMajor and
// DeclinesMatricesThatAreNotPacked then become accepted cases; keep a decline for a
// C row stride below N. Drop the padded-leading-dimension sentence from
// docs/OperationSupport.md and projects/hipdnn/docs/user-guides/reference/hipblaslt.rst.
// ----------------------------------------------------------------------------

#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/DeviceQuery.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "HipblasltUtils.hpp"

#include <exception>
#include <string>

#ifdef _WIN32
#define REJECT_IF_WORKAROUND_ISSUE_9962(handle)                                                    \
    do                                                                                             \
    {                                                                                              \
        try                                                                                        \
        {                                                                                          \
            if(::hipdnn_plugin_sdk::archMatches(                                                   \
                   ::hipdnn_plugin_sdk::getDeviceArch((handle).getStream()),                       \
                   "gfx115",                                                                       \
                   ::hipdnn_plugin_sdk::ArchMatchMode::SUBSTRING))                                 \
            {                                                                                      \
                HIPDNN_PLUGIN_LOG_INFO(                                                            \
                    "[#9962] hipBLASLt matmul not applicable: GEMM crashes on gfx115x (Windows)"); \
                return false;                                                                      \
            }                                                                                      \
        }                                                                                          \
        catch(const std::exception& workaround_9962_e)                                             \
        {                                                                                          \
            HIPDNN_PLUGIN_LOG_INFO("[#9962] arch query failed; treating as not-applicable: "       \
                                   << workaround_9962_e.what());                                   \
            return false;                                                                          \
        }                                                                                          \
    } while(0)
#else
#define REJECT_IF_WORKAROUND_ISSUE_9962(handle) \
    do                                          \
    {                                           \
    } while(0)
#endif

#define REJECT_IF_WORKAROUND_ISSUE_10811(deqAttrA, deqAttrB, tensorMap)             \
    do                                                                              \
    {                                                                               \
        const auto tXA = ::hipblaslt_plugin::hipblaslt_utils::findTensorAttributes( \
            (tensorMap), (deqAttrA).x_tensor_uid());                                \
        const auto tXB = ::hipblaslt_plugin::hipblaslt_utils::findTensorAttributes( \
            (tensorMap), (deqAttrB).x_tensor_uid());                                \
        const auto& stridesXA = tXA.strides();                                      \
        if(::hipblaslt_plugin::hipblaslt_utils::isTypeFp8Ocp(tXA.dataType())        \
           && ::hipblaslt_plugin::hipblaslt_utils::isTypeFp6Ocp(tXB.dataType())     \
           && (stridesXA.empty() || stridesXA.back() != 1))                         \
        {                                                                           \
            HIPDNN_PLUGIN_LOG_INFO(                                                 \
                "[#10811] MX matmul not applicable: FP8 OCP A (column-major) with " \
                "FP6 B returns incorrect results");                                 \
            return false;                                                           \
        }                                                                           \
    } while(0)

#define REJECT_IF_WORKAROUND_ISSUE_14290(tensor, name)                              \
    do                                                                              \
    {                                                                               \
        const auto& dims14290 = (tensor).dims();                                    \
        const auto& strides14290 = (tensor).strides();                              \
        const auto rank14290 = dims14290.size();                                    \
        const int64_t outer14290 = dims14290[rank14290 - 2];                        \
        const int64_t inner14290 = dims14290[rank14290 - 1];                        \
        const bool packed14290                                                      \
            = strides14290[rank14290 - 1] == 1                                      \
                  ? (outer14290 == 1 || strides14290[rank14290 - 2] == inner14290)  \
                  : (inner14290 == 1 || strides14290[rank14290 - 1] == outer14290); \
        if(!packed14290)                                                            \
        {                                                                           \
            throw ::hipdnn_plugin_sdk::HipdnnPluginException(                       \
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,                                     \
                std::string("[#14290] MX matmul: ") + (name)                        \
                    + " has a padded leading dimension, which hipBLASLt ignores");  \
        }                                                                           \
    } while(0)

// NOLINTNEXTLINE(modernize-macro-to-enum)
#define WORKAROUND_ISSUE_14286 1
