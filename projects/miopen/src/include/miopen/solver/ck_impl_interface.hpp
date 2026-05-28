// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/// \file ck_impl_interface.hpp
/// extern "C" interface between MIOpen core and per-architecture CK
/// implementation libraries (libMIOpenCKGroupedConv_<arch>.so).
///
/// This header compiles without any CK includes. Both the MIOpen core
/// (loader side) and the impl library (CK side) include it.

#include <cstddef>         // size_t
#include <miopen/miopen.h> // miopenDataType_t

#include "ck_impl_status.h"

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
#define CK_IMPL_API_VERSION 5

/// Opaque handle wrapping a list of valid kernel ID strings.
/// Allocated by the impl library, freed by the caller via
/// ck_impl_kernel_list_free().
struct CKKernelListHandle;

// ---------------------------------------------------------------------------
// DLL/SO export macro
// ---------------------------------------------------------------------------
#if defined(_WIN32) && defined(CK_IMPL_BUILDING_DLL)
#define CK_IMPL_API __declspec(dllexport)
#elif defined(CK_IMPL_BUILDING_DLL)
#define CK_IMPL_API __attribute__((visibility("default")))
#else
#define CK_IMPL_API
#endif

// ---------------------------------------------------------------------------
// extern "C" interface
// ---------------------------------------------------------------------------
extern "C" {

// -- API version ------------------------------------------------------------

CK_IMPL_API int ck_impl_get_api_version();

// -- Error reporting --------------------------------------------------------

CK_IMPL_API void ck_impl_get_last_error_string(const char** error_str);

// -- Kernel list accessors --------------------------------------------------

/// Number of kernel IDs in the list.  On success, writes the count to
/// *out_size and returns ck_impl_status_success.
CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_kernel_list_size(const CKKernelListHandle* handle, size_t* out_size);

/// Pointer to the idx-th kernel ID string (null-terminated, valid for the
/// lifetime of the handle).  On success, writes the pointer to *out_str and
/// returns ck_impl_status_success.  Returns an error on out-of-range.
CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_kernel_list_get(const CKKernelListHandle* handle, size_t idx, const char** out_str);

/// Free a kernel list handle returned by any fill_valid_kernels function.
CK_IMPL_API void ck_impl_kernel_list_free(CKKernelListHandle* handle);

// -- Free a ConvSolution returned by get_solution ---------------------------

/// Free a heap-allocated ConvSolution returned by any get_solution function.
CK_IMPL_API void ck_impl_solution_free(miopen::solver::ConvSolution* solution);

// -- FWD direction ----------------------------------------------------------

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fwd_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
                               miopenDataType_t data_type,
                               bool use_tf32,
                               CKKernelListHandle** out_handle);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fwd_is_applicable(const miopen::conv::ProblemDescription* problem,
                          miopenDataType_t data_type,
                          bool use_tf32,
                          bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fwd_is_args_supported(const miopen::conv::ProblemDescription* problem,
                              const char* kernel_id,
                              miopenDataType_t data_type,
                              bool use_tf32,
                              bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fwd_get_workspace_size(const miopen::conv::ProblemDescription* problem,
                               miopenDataType_t data_type,
                               bool use_tf32,
                               size_t* out_size);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fwd_get_solution(const miopen::ExecutionContext* ctx,
                         const miopen::conv::ProblemDescription* problem,
                         const char* kernel_id,
                         bool use_tf32,
                         miopen::solver::ConvSolution** out_solution);

// -- BWD direction ----------------------------------------------------------

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_bwd_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
                               miopenDataType_t data_type,
                               bool use_tf32,
                               CKKernelListHandle** out_handle);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_bwd_is_applicable(const miopen::conv::ProblemDescription* problem,
                          miopenDataType_t data_type,
                          bool use_tf32,
                          bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_bwd_is_args_supported(const miopen::conv::ProblemDescription* problem,
                              const char* kernel_id,
                              miopenDataType_t data_type,
                              bool use_tf32,
                              bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_bwd_get_workspace_size(const miopen::conv::ProblemDescription* problem,
                               miopenDataType_t data_type,
                               bool use_tf32,
                               size_t* out_size);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_bwd_get_solution(const miopen::ExecutionContext* ctx,
                         const miopen::conv::ProblemDescription* problem,
                         const char* kernel_id,
                         bool use_tf32,
                         miopen::solver::ConvSolution** out_solution);

// -- WRW direction ----------------------------------------------------------

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_wrw_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
                               miopenDataType_t data_type,
                               bool use_tf32,
                               CKKernelListHandle** out_handle);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_wrw_is_applicable(const miopen::conv::ProblemDescription* problem,
                          miopenDataType_t data_type,
                          bool use_tf32,
                          bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_wrw_is_args_supported(const miopen::conv::ProblemDescription* problem,
                              const char* kernel_id,
                              miopenDataType_t data_type,
                              bool use_tf32,
                              bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_wrw_get_workspace_size(const miopen::conv::ProblemDescription* problem,
                               miopenDataType_t data_type,
                               bool use_tf32,
                               size_t* out_size);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_wrw_get_solution(const miopen::ExecutionContext* ctx,
                         const miopen::conv::ProblemDescription* problem,
                         const char* kernel_id,
                         bool use_tf32,
                         miopen::solver::ConvSolution** out_solution);

// -- 3D grouped FWD (S6) ---------------------------------------------------

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_3d_fwd_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
                                  miopenDataType_t data_type,
                                  bool use_tf32,
                                  CKKernelListHandle** out_handle);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_3d_fwd_is_applicable(const miopen::conv::ProblemDescription* problem,
                             miopenDataType_t data_type,
                             bool use_tf32,
                             bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_3d_fwd_is_args_supported(const miopen::conv::ProblemDescription* problem,
                                 const char* kernel_id,
                                 miopenDataType_t data_type,
                                 bool use_tf32,
                                 bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_3d_fwd_get_workspace_size(const miopen::conv::ProblemDescription* problem,
                                  miopenDataType_t data_type,
                                  bool use_tf32,
                                  size_t* out_size);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_3d_fwd_get_solution(const miopen::ExecutionContext* ctx,
                            const miopen::conv::ProblemDescription* problem,
                            const char* kernel_id,
                            bool use_tf32,
                            miopen::solver::ConvSolution** out_solution);

// -- 3D grouped BWD (S7) ---------------------------------------------------

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_3d_bwd_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
                                  miopenDataType_t data_type,
                                  bool use_tf32,
                                  CKKernelListHandle** out_handle);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_3d_bwd_is_applicable(const miopen::conv::ProblemDescription* problem,
                             miopenDataType_t data_type,
                             bool use_tf32,
                             bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_3d_bwd_is_args_supported(const miopen::conv::ProblemDescription* problem,
                                 const char* kernel_id,
                                 miopenDataType_t data_type,
                                 bool use_tf32,
                                 bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_3d_bwd_get_workspace_size(const miopen::conv::ProblemDescription* problem,
                                  miopenDataType_t data_type,
                                  bool use_tf32,
                                  size_t* out_size);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_3d_bwd_get_solution(const miopen::ExecutionContext* ctx,
                            const miopen::conv::ProblemDescription* problem,
                            const char* kernel_id,
                            bool use_tf32,
                            miopen::solver::ConvSolution** out_solution);

// -- 3D grouped WRW (S8) ---------------------------------------------------

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_3d_wrw_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
                                  miopenDataType_t data_type,
                                  bool use_tf32,
                                  CKKernelListHandle** out_handle);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_3d_wrw_is_applicable(const miopen::conv::ProblemDescription* problem,
                             miopenDataType_t data_type,
                             bool use_tf32,
                             bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_3d_wrw_is_args_supported(const miopen::conv::ProblemDescription* problem,
                                 const char* kernel_id,
                                 miopenDataType_t data_type,
                                 bool use_tf32,
                                 bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_3d_wrw_get_workspace_size(const miopen::conv::ProblemDescription* problem,
                                  miopenDataType_t data_type,
                                  bool use_tf32,
                                  size_t* out_size);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_3d_wrw_get_solution(const miopen::ExecutionContext* ctx,
                            const miopen::conv::ProblemDescription* problem,
                            const char* kernel_id,
                            bool use_tf32,
                            miopen::solver::ConvSolution** out_solution);

// -- Fused Conv+Bias+ReLU (non-grouped, FP16) ---------------------------------

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fused_bias_activ_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
                                            miopenDataType_t data_type,
                                            bool use_tf32,
                                            CKKernelListHandle** out_handle);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fused_bias_activ_is_applicable(const miopen::conv::ProblemDescription* problem,
                                       miopenDataType_t data_type,
                                       bool use_tf32,
                                       bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fused_bias_activ_is_args_supported(const miopen::conv::ProblemDescription* problem,
                                           const char* kernel_id,
                                           miopenDataType_t data_type,
                                           bool use_tf32,
                                           bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fused_bias_activ_get_workspace_size(const miopen::conv::ProblemDescription* problem,
                                            miopenDataType_t data_type,
                                            bool use_tf32,
                                            size_t* out_size);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fused_bias_activ_get_solution(const miopen::ExecutionContext* ctx,
                                      const miopen::conv::ProblemDescription* problem,
                                      const char* kernel_id,
                                      bool use_tf32,
                                      miopen::solver::ConvSolution** out_solution);

// -- Fused Conv+ScaleAdd+Bias+ReLU --------------------------------------------

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fused_bias_res_add_activ_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
                                                    miopenDataType_t data_type,
                                                    bool use_tf32,
                                                    CKKernelListHandle** out_handle);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fused_bias_res_add_activ_is_applicable(const miopen::conv::ProblemDescription* problem,
                                               miopenDataType_t data_type,
                                               bool use_tf32,
                                               bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fused_bias_res_add_activ_is_args_supported(const miopen::conv::ProblemDescription* problem,
                                                   const char* kernel_id,
                                                   miopenDataType_t data_type,
                                                   bool use_tf32,
                                                   bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fused_bias_res_add_activ_get_workspace_size(const miopen::conv::ProblemDescription* problem,
                                                    miopenDataType_t data_type,
                                                    bool use_tf32,
                                                    size_t* out_size);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fused_bias_res_add_activ_get_solution(const miopen::ExecutionContext* ctx,
                                              const miopen::conv::ProblemDescription* problem,
                                              const char* kernel_id,
                                              bool use_tf32,
                                              miopen::solver::ConvSolution** out_solution);

// -- Fused Grouped Conv+Activation(Clamp) -------------------------------------

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fused_grp_activ_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
                                           miopenDataType_t data_type,
                                           bool use_tf32,
                                           CKKernelListHandle** out_handle);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fused_grp_activ_is_applicable(const miopen::conv::ProblemDescription* problem,
                                      miopenDataType_t data_type,
                                      bool use_tf32,
                                      bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fused_grp_activ_is_args_supported(const miopen::conv::ProblemDescription* problem,
                                          const char* kernel_id,
                                          miopenDataType_t data_type,
                                          bool use_tf32,
                                          bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fused_grp_activ_get_workspace_size(const miopen::conv::ProblemDescription* problem,
                                           miopenDataType_t data_type,
                                           bool use_tf32,
                                           size_t* out_size);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fused_grp_activ_get_solution(const miopen::ExecutionContext* ctx,
                                     const miopen::conv::ProblemDescription* problem,
                                     const char* kernel_id,
                                     bool use_tf32,
                                     miopen::solver::ConvSolution** out_solution);

// -- Fused Grouped Conv+Bias+Activation(AddClamp) -----------------------------

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fused_grp_bias_activ_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
                                                miopenDataType_t data_type,
                                                bool use_tf32,
                                                CKKernelListHandle** out_handle);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fused_grp_bias_activ_is_applicable(const miopen::conv::ProblemDescription* problem,
                                           miopenDataType_t data_type,
                                           bool use_tf32,
                                           bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fused_grp_bias_activ_is_args_supported(const miopen::conv::ProblemDescription* problem,
                                               const char* kernel_id,
                                               miopenDataType_t data_type,
                                               bool use_tf32,
                                               bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fused_grp_bias_activ_get_workspace_size(const miopen::conv::ProblemDescription* problem,
                                                miopenDataType_t data_type,
                                                bool use_tf32,
                                                size_t* out_size);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fused_grp_bias_activ_get_solution(const miopen::ExecutionContext* ctx,
                                          const miopen::conv::ProblemDescription* problem,
                                          const char* kernel_id,
                                          bool use_tf32,
                                          miopen::solver::ConvSolution** out_solution);

// -- Depthwise Conv FWD (custom CK kernel, FP16) ---------------------------------

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_depthwise_fwd_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
                                         miopenDataType_t data_type,
                                         bool use_tf32,
                                         CKKernelListHandle** out_handle);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_depthwise_fwd_is_applicable(const miopen::conv::ProblemDescription* problem,
                                    miopenDataType_t data_type,
                                    bool use_tf32,
                                    bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_depthwise_fwd_is_args_supported(const miopen::conv::ProblemDescription* problem,
                                        const char* kernel_id,
                                        miopenDataType_t data_type,
                                        bool use_tf32,
                                        bool* out_result);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_depthwise_fwd_get_workspace_size(const miopen::conv::ProblemDescription* problem,
                                         miopenDataType_t data_type,
                                         bool use_tf32,
                                         size_t* out_size);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_depthwise_fwd_get_solution(const miopen::ExecutionContext* ctx,
                                   const miopen::conv::ProblemDescription* problem,
                                   const char* kernel_id,
                                   bool use_tf32,
                                   miopen::solver::ConvSolution** out_solution);

// -- Get all kernel type strings (for test/metadata validation) -----------------

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_fwd_get_all_kernel_type_strings(CKKernelListHandle** out_handle);
CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_bwd_get_all_kernel_type_strings(CKKernelListHandle** out_handle);
CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_wrw_get_all_kernel_type_strings(CKKernelListHandle** out_handle);

CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_3d_fwd_get_all_kernel_type_strings(CKKernelListHandle** out_handle);
CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_3d_bwd_get_all_kernel_type_strings(CKKernelListHandle** out_handle);
CK_IMPL_NODISCARD CK_IMPL_API ck_impl_status_t
ck_impl_3d_wrw_get_all_kernel_type_strings(CKKernelListHandle** out_handle);

} // extern "C"
