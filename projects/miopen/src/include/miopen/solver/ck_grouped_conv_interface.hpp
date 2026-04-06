// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/// \file ck_grouped_conv_interface.hpp
/// extern "C" interface between MIOpen core and per-architecture CK grouped
/// convolution implementation libraries (libMIOpenCKGroupedConv_<arch>.so).
///
/// This header compiles without any CK includes. Both the MIOpen core
/// (loader side) and the impl library (CK side) include it.

#include <cstddef>         // size_t
#include <miopen/miopen.h> // miopenDataType_t

// Forward declarations — defined in MIOpen headers shared by both sides.
namespace miopen {
struct ExecutionContext;
namespace conv {
struct ProblemDescription;
} // namespace conv
namespace solver {
struct ConvSolution;
} // namespace solver
} // namespace miopen

/// API version constant. Bump when the set of extern "C" symbols or their
/// semantics change.  The loader checks this at dlopen time.
#define CK_GROUPED_CONV_API_VERSION 1

/// Opaque handle wrapping a list of valid kernel ID strings.
/// Allocated by the impl library, freed by the caller via
/// ckgrpconv_kernel_list_free().
struct CKKernelListHandle;

// ---------------------------------------------------------------------------
// DLL export macro for Windows
// ---------------------------------------------------------------------------
#if defined(_WIN32) && defined(CK_GROUPED_CONV_BUILDING_DLL)
#define CK_GROUPED_CONV_API __declspec(dllexport)
#else
#define CK_GROUPED_CONV_API
#endif

// ---------------------------------------------------------------------------
// extern "C" interface
// ---------------------------------------------------------------------------
extern "C" {

// -- API version ------------------------------------------------------------

CK_GROUPED_CONV_API int ckgrpconv_get_api_version();

// -- Kernel list accessors --------------------------------------------------

/// Number of kernel IDs in the list.
CK_GROUPED_CONV_API size_t ckgrpconv_kernel_list_size(const CKKernelListHandle* handle);

/// Pointer to the idx-th kernel ID string (null-terminated, valid for the
/// lifetime of the handle).  Returns nullptr on out-of-range.
CK_GROUPED_CONV_API const char* ckgrpconv_kernel_list_get(const CKKernelListHandle* handle,
                                                          size_t idx);

/// Free a kernel list handle returned by any fill_valid_kernels function.
CK_GROUPED_CONV_API void ckgrpconv_kernel_list_free(CKKernelListHandle* handle);

// -- Free a ConvSolution returned by get_solution ---------------------------

/// Free a heap-allocated ConvSolution returned by any get_solution function.
CK_GROUPED_CONV_API void ckgrpconv_solution_free(miopen::solver::ConvSolution* solution);

// -- FWD direction ----------------------------------------------------------

CK_GROUPED_CONV_API CKKernelListHandle* ckgrpconv_fwd_fill_valid_kernels(
    const miopen::conv::ProblemDescription* problem, miopenDataType_t data_type, bool use_tf32);

CK_GROUPED_CONV_API bool ckgrpconv_fwd_is_applicable(
    const miopen::conv::ProblemDescription* problem, miopenDataType_t data_type, bool use_tf32);

CK_GROUPED_CONV_API bool
ckgrpconv_fwd_is_args_supported(const miopen::conv::ProblemDescription* problem,
                                const char* kernel_id,
                                miopenDataType_t data_type,
                                bool use_tf32);

CK_GROUPED_CONV_API size_t ckgrpconv_fwd_get_workspace_size(
    const miopen::conv::ProblemDescription* problem, miopenDataType_t data_type, bool use_tf32);

CK_GROUPED_CONV_API miopen::solver::ConvSolution*
ckgrpconv_fwd_get_solution(const miopen::ExecutionContext* ctx,
                           const miopen::conv::ProblemDescription* problem,
                           const char* kernel_id,
                           bool use_tf32);

// -- BWD direction ----------------------------------------------------------

CK_GROUPED_CONV_API CKKernelListHandle* ckgrpconv_bwd_fill_valid_kernels(
    const miopen::conv::ProblemDescription* problem, miopenDataType_t data_type, bool use_tf32);

CK_GROUPED_CONV_API bool ckgrpconv_bwd_is_applicable(
    const miopen::conv::ProblemDescription* problem, miopenDataType_t data_type, bool use_tf32);

CK_GROUPED_CONV_API bool
ckgrpconv_bwd_is_args_supported(const miopen::conv::ProblemDescription* problem,
                                const char* kernel_id,
                                miopenDataType_t data_type,
                                bool use_tf32);

CK_GROUPED_CONV_API size_t ckgrpconv_bwd_get_workspace_size(
    const miopen::conv::ProblemDescription* problem, miopenDataType_t data_type, bool use_tf32);

CK_GROUPED_CONV_API miopen::solver::ConvSolution*
ckgrpconv_bwd_get_solution(const miopen::ExecutionContext* ctx,
                           const miopen::conv::ProblemDescription* problem,
                           const char* kernel_id,
                           bool use_tf32);

// -- WRW direction ----------------------------------------------------------

CK_GROUPED_CONV_API CKKernelListHandle* ckgrpconv_wrw_fill_valid_kernels(
    const miopen::conv::ProblemDescription* problem, miopenDataType_t data_type, bool use_tf32);

CK_GROUPED_CONV_API bool ckgrpconv_wrw_is_applicable(
    const miopen::conv::ProblemDescription* problem, miopenDataType_t data_type, bool use_tf32);

CK_GROUPED_CONV_API bool
ckgrpconv_wrw_is_args_supported(const miopen::conv::ProblemDescription* problem,
                                const char* kernel_id,
                                miopenDataType_t data_type,
                                bool use_tf32);

CK_GROUPED_CONV_API size_t ckgrpconv_wrw_get_workspace_size(
    const miopen::conv::ProblemDescription* problem, miopenDataType_t data_type, bool use_tf32);

CK_GROUPED_CONV_API miopen::solver::ConvSolution*
ckgrpconv_wrw_get_solution(const miopen::ExecutionContext* ctx,
                           const miopen::conv::ProblemDescription* problem,
                           const char* kernel_id,
                           bool use_tf32);

} // extern "C"
