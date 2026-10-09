// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*! \file
 *  \brief Experimental runtime user kernel registration.
 *
 *  Available only when hipBLASLt is built with HIPBLASLT_ENABLE_USER_KERNELS=ON.
 *  The interface may change without notice.
 *
 *  Registrations and mappings last for the process. Nothing is written to
 *  disk.
 */

#pragma once

#include <hipblaslt/hipblaslt.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup library_module
 *  \brief Register compiled kernels with the running process.
 *
 *  \details
 *  A payload is a library shard (msgpack) and the code object it describes, in
 *  the same format the shipped library loads lazily.
 *
 *  Each solution in the payload receives an index usable with the index-based
 *  algorithm path of \ref hipblasLtMatmul, on any device of the same
 *  architecture. Registration does not change heuristic selection; use
 *  \ref hipblasLtUserKernelSetExactMatch for that.
 *
 *  Indices are valid only in the current process. Registering the same
 *  payload twice registers its kernels twice.
 *
 *  @param[in]
 *  handle          Pointer to the allocated hipBLASLt handle.
 *  @param[in]
 *  libraryPath     Path to the payload's ``.dat`` or ``.dat.zlib`` shard. For
 *                  a ``.dat`` path, a ``.dat.zlib`` file next to it is used
 *                  if present, as for the shipped library.
 *  @param[in]
 *  codeObjectPath  Path to the payload's ``.co`` code object. It must stay in
 *                  place for the life of the process: other devices load it
 *                  when they first run one of its kernels.
 *  @param[out]
 *  kernelIndices   Receives the assigned indices, up to ``maxIndices``.
 *  @param[in]
 *  maxIndices      Capacity of ``kernelIndices``.
 *  @param[out]
 *  numIndices      Number of solutions registered, which may exceed
 *                  ``maxIndices``; in that case only the first ``maxIndices``
 *                  are written.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS          The payload was registered.
 *  \retval HIPBLAS_STATUS_INVALID_VALUE    A file could not be read, held no
 *                                          solutions, or the code object could
 *                                          not be loaded on this device.
 */
HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtUserKernelRegister(hipblasLtHandle_t handle,
                                            const char*       libraryPath,
                                            const char*       codeObjectPath,
                                            int*              kernelIndices,
                                            int               maxIndices,
                                            int*              numIndices);

/*! \ingroup library_module
 *  \brief Make a registered kernel selectable for one exact problem shape.
 *
 *  \details
 *  The mapping is for this shape and the kernel's own transposes and data
 *  types. Heuristic selection for such a problem returns the kernel first,
 *  followed by the shipped library's choices, including for problems already
 *  queried. Other shapes and types are unaffected, so kernels for different
 *  data types or transposes can each be mapped to the same shape. Among
 *  kernels mapped under the same shape and types, the one mapped last that can
 *  serve the problem wins.
 *
 *  @param[in]
 *  handle       Pointer to the allocated hipBLASLt handle.
 *  @param[in]
 *  kernelIndex  An index returned by \ref hipblasLtUserKernelRegister.
 *  @param[in]
 *  m, n, batch, k  The problem shape to map, matched exactly.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS          The mapping was published.
 *  \retval HIPBLAS_STATUS_INVALID_VALUE    The index is not a registered user
 *                                          kernel.
 */
HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtUserKernelSetExactMatch(
    hipblasLtHandle_t handle, int kernelIndex, size_t m, size_t n, size_t batch, size_t k);

/*! \ingroup library_module
 *  \brief Report how many kernels are registered and how many mappings exist.
 *
 *  @param[in]
 *  handle      Pointer to the allocated hipBLASLt handle.
 *  @param[out]
 *  registered  Number of registered kernels, or NULL if not wanted.
 *  @param[out]
 *  selectable  Number of distinct (shape, transposes, data types) with at
 *              least one kernel mapped, or NULL if not wanted.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS  The counts were written.
 */
HIPBLASLT_EXPORT
hipblasStatus_t
    hipblasLtUserKernelGetCounts(hipblasLtHandle_t handle, size_t* registered, size_t* selectable);

/*! \ingroup library_module
 *  \brief Whether an algorithm refers to a registered user kernel.
 *
 *  @param[in]
 *  algo          The algorithm to inspect.
 *  @param[out]
 *  isUserKernel  Set to a nonzero value if the algorithm is a registered kernel.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS          The result was written.
 *  \retval HIPBLAS_STATUS_INVALID_VALUE    A pointer argument was NULL.
 */
HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtMatmulAlgoIsUserKernel(const hipblasLtMatmulAlgo_t* algo,
                                                int*                         isUserKernel);

#ifdef __cplusplus
}
#endif
