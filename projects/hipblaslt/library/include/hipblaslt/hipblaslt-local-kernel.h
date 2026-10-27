// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*! \file
 *  \brief Experimental runtime local kernel registration.
 *
 *  Available only when hipBLASLt is built with HIPBLASLT_ENABLE_LOCAL_KERNELS=ON.
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
 *  algorithm path of \ref hipblasLtMatmul. Registration is per architecture:
 *  the kernels serve devices of the architecture of the handle's device. On a
 *  device of any other architecture they are never selected, and their indices
 *  are treated as unknown by every call that takes an index, including
 *  \ref hipblasLtMatmul. Registration does not change heuristic selection; use
 *  \ref hipblasLtLocalKernelSetExactMatch for that.
 *
 *  The code object must define every kernel the shard names. Kernels run only
 *  from the code object they were registered with, even if a shipped or
 *  earlier registered kernel has the same name.
 *
 *  The code object is loaded on the handle's device, which is made current
 *  for the duration of the call; the caller's current device is restored.
 *
 *  Indices are valid only in the current process. Registering the same
 *  payload twice registers its kernels twice. A device loads a code object
 *  path at most once and keeps it loaded for the life of the process, so a
 *  file replaced at the same path after it was loaded is not reloaded, and a
 *  code object loaded by a registration that then failed stays loaded.
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
 *  kernelIndices   Receives the assigned indices, up to ``maxIndices``, or
 *                  NULL if not wanted.
 *  @param[in]
 *  maxIndices      Capacity of ``kernelIndices``. No indices are written if
 *                  this is nonpositive.
 *  @param[out]
 *  numIndices      Number of solutions registered, which may exceed
 *                  ``maxIndices``; in that case only the first ``maxIndices``
 *                  are written. May be NULL if not wanted. A call with no
 *                  output buffer still registers the payload.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS          The payload was registered.
 *  \retval HIPBLAS_STATUS_INVALID_VALUE    A file could not be read, the shard
 *                                          held no solutions or was built for
 *                                          another device, or the code object
 *                                          could not be loaded on this device
 *                                          or lacks one of the shard's kernels.
 *                                          Nothing is registered.
 */
HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtLocalKernelRegister(hipblasLtHandle_t handle,
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
 *  A usable tuning-file override, including one supplied through
 *  HIPBLASLT_TUNING_OVERRIDE_FILE, takes priority over this mapping. An
 *  algorithm passed explicitly to hipblasLtMatmul also keeps its selection.
 *
 *  @param[in]
 *  handle       Pointer to the allocated hipBLASLt handle.
 *  @param[in]
 *  kernelIndex  An index returned by \ref hipblasLtLocalKernelRegister.
 *  @param[in]
 *  m, n, batch, k  The problem shape to map, matched exactly.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS          The mapping was published.
 *  \retval HIPBLAS_STATUS_INVALID_VALUE    The index is not a registered local
 *                                          kernel.
 */
HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtLocalKernelSetExactMatch(
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
    hipblasLtLocalKernelGetCounts(hipblasLtHandle_t handle, size_t* registered, size_t* selectable);

/*! \ingroup library_module
 *  \brief Whether an algorithm's index is in the local kernel range.
 *
 *  \details This checks the index range alone. It does not verify that the
 *  index is registered or that it can run on the current device.
 *
 *  @param[in]
 *  algo          The algorithm to inspect.
 *  @param[out]
 *  isLocalKernel  Set to a nonzero value if the index is in the local kernel range.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS          The result was written.
 *  \retval HIPBLAS_STATUS_INVALID_VALUE    A pointer argument was NULL.
 */
HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtMatmulAlgoIsLocalKernel(const hipblasLtMatmulAlgo_t* algo,
                                                 int*                         isLocalKernel);

#ifdef __cplusplus
}
#endif
