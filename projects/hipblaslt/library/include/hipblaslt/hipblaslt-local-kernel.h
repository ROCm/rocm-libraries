// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*! \file
 *  \brief Experimental runtime local kernel registration.
 *
 *  Available only when hipBLASLt is built with HIPBLASLT_ENABLE_LOCAL_KERNELS=ON.
 *  The interface may change without notice.
 *
 *  By default, registrations and mappings last for the process and nothing is
 *  written to disk.
 *
 *  When the environment variable ``HIPBLASLT_LOCAL_KERNEL_LIBRARY`` names an
 *  absolute directory, it is a local kernel library shared by every process
 *  that names it:
 *
 *  - Kernels and mappings are saved for the kind of device they were made
 *    on, and for the build of hipBLASLt that made them. A kind of device is
 *    an architecture as the runtime reports it (with gfx1250 A0 apart from
 *    later gfx1250 revisions, as for the shipped kernels), a compute unit
 *    count and, on architectures that report one, a PCI chip ID. Those saved
 *    for a device are loaded when this build of hipBLASLt first initializes
 *    it, before any heuristic query. A saved kernel is loaded only if its
 *    files still match what was saved and it can run on the device;
 *    otherwise it is skipped with a warning. One skipped for its files is
 *    tried again by \ref hipblasLtLocalKernelRefresh once they change. Its
 *    code object is loaded when one of its kernels first runs.
 *  - \ref hipblasLtLocalKernelRegister and \ref hipblasLtLocalKernelSetExactMatch
 *    save to it before they return, and fail if they cannot.
 *  - \ref hipblasLtLocalKernelRefresh loads what other processes have saved
 *    since.
 *
 *  The directory is created if absent. It is not used, and a warning is
 *  printed, if it or a directory in it is owned by another user or is group-
 *  or world-writable, or if the process is running with elevated privileges
 *  (set-user-ID, set-group-ID or file capabilities).
 *
 *  As for the shipped kernels, hipBLASLt describes all devices of one
 *  architecture name in a process by one of them, and heuristic queries on
 *  any of them see that one's kind. Local kernels are therefore not
 *  supported on a device of another kind than that one, as in a process whose
 *  devices of one architecture name differ in compute unit count, chip ID or
 *  gfx1250 revision: \ref hipblasLtLocalKernelRegister,
 *  \ref hipblasLtLocalKernelSetExactMatch and \ref hipblasLtLocalKernelRefresh
 *  fail on it with HIPBLAS_STATUS_ARCH_MISMATCH.
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
 *  Indices are valid only in the current process. A payload is identified by
 *  the contents of both files: registering one already registered in this
 *  process, or loaded from the local kernel library, returns the indices it
 *  already has. That includes one registered through another kind of device
 *  (see above), if every kernel can also run on the handle's device;
 *  otherwise the call fails as for any payload not built for the device.
 *
 *  With a local kernel library, both files are copied into it and the copies
 *  are what is loaded, so the originals may be removed afterwards. Files
 *  already saved there with the same contents are not copied again.
 *
 *  A device loads a code object path at most once and keeps it loaded for the
 *  life of the process, including after a registration that then failed.
 *  Without a local kernel library, a file replaced at the same path after it
 *  was loaded is therefore not reloaded.
 *
 *  @param[in]
 *  handle          Pointer to the allocated hipBLASLt handle.
 *  @param[in]
 *  libraryPath     Path to the payload's ``.dat`` or ``.dat.zlib`` shard. For
 *                  a ``.dat`` path, a ``.dat.zlib`` file next to it is used
 *                  if present, as for the shipped library.
 *  @param[in]
 *  codeObjectPath  Path to the payload's ``.co`` code object. Without a local
 *                  kernel library it must stay in place for the life of the
 *                  process: other devices load it when they first run one of
 *                  its kernels.
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
 *  \retval HIPBLAS_STATUS_NOT_INITIALIZED  \p handle is NULL.
 *  \retval HIPBLAS_STATUS_INVALID_VALUE    A path was NULL, a file could not be
 *                                          read, the shard held no solutions or
 *                                          was built for another device, or the
 *                                          code object could not be loaded on
 *                                          this device or lacks one of the
 *                                          shard's kernels. Nothing is
 *                                          registered.
 *  \retval HIPBLAS_STATUS_ARCH_MISMATCH    Local kernels are not supported on
 *                                          the handle's device (see above).
 *                                          Nothing is registered.
 *  \retval HIPBLAS_STATUS_INTERNAL_ERROR   The handle's device could not be made
 *                                          current or initialized, the local
 *                                          kernel index range is exhausted, or
 *                                          the payload could not be saved to the
 *                                          local kernel library. Nothing is
 *                                          registered.
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
 *  types, on devices of the same kind as the handle's (see above); other
 *  devices in the process do not take it, even where the kernel can run on
 *  them. On devices of the handle's kind, heuristic selection for such a
 *  problem returns the kernel first, followed by the shipped library's
 *  choices, including for problems already queried. Other shapes and types
 *  are unaffected, so kernels for different data types or transposes can each
 *  be mapped to the same shape. Among kernels mapped under the same shape and
 *  types, the one mapped last that can serve the problem wins.
 *
 *  A usable tuning-file override, including one supplied through
 *  HIPBLASLT_TUNING_OVERRIDE_FILE, takes priority over this mapping. An
 *  algorithm passed explicitly to hipblasLtMatmul also keeps its selection.
 *
 *  With a local kernel library, "mapped last" is across all processes and
 *  threads: the mapping is saved, then applied together with what other
 *  processes saved before it, as \ref hipblasLtLocalKernelRefresh would. If
 *  another process or thread mapped the same shape and types in between, its
 *  kernel is the one selected, here and in every later process. The mapping
 *  is saved for the same kind of device it applies to.
 *
 *  @param[in]
 *  handle       Pointer to the allocated hipBLASLt handle.
 *  @param[in]
 *  kernelIndex  An index returned by \ref hipblasLtLocalKernelRegister.
 *  @param[in]
 *  m, n, batch, k  The problem shape to map, matched exactly.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS          The mapping was published.
 *  \retval HIPBLAS_STATUS_NOT_INITIALIZED  \p handle is NULL.
 *  \retval HIPBLAS_STATUS_INVALID_VALUE    The index is not a registered local
 *                                          kernel or cannot run on the handle's
 *                                          device.
 *  \retval HIPBLAS_STATUS_ARCH_MISMATCH    Local kernels are not supported on
 *                                          the handle's device (see above).
 *  \retval HIPBLAS_STATUS_INTERNAL_ERROR   The handle's device could not be made
 *                                          current or initialized, or the
 *                                          mapping could not be saved to the
 *                                          local kernel library; it was not
 *                                          published.
 */
HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtLocalKernelSetExactMatch(
    hipblasLtHandle_t handle, int kernelIndex, size_t m, size_t n, size_t batch, size_t k);

/*! \ingroup library_module
 *  \brief Load kernels and mappings other processes have saved since.
 *
 *  \details
 *  Reads what has been added to the local kernel library for the handle's
 *  device since this process last read it. Mappings are applied in the order
 *  they were made, across all processes. Does nothing without a local kernel
 *  library.
 *
 *  @param[in]
 *  handle      Pointer to the allocated hipBLASLt handle.
 *  @param[out]
 *  registered  Number of kernels newly registered, or NULL if not wanted.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS          The library was read. Records that
 *                                          cannot be loaded are skipped with a
 *                                          warning.
 *  \retval HIPBLAS_STATUS_NOT_INITIALIZED  \p handle is NULL.
 *  \retval HIPBLAS_STATUS_ARCH_MISMATCH    Local kernels are not supported on
 *                                          the handle's device (see above);
 *                                          nothing was loaded.
 *  \retval HIPBLAS_STATUS_INTERNAL_ERROR   The handle's device could not be made
 *                                          current or initialized, or the
 *                                          local kernel library could not be
 *                                          read; nothing was loaded.
 */
HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtLocalKernelRefresh(hipblasLtHandle_t handle, size_t* registered);

/*! \ingroup library_module
 *  \brief Report how many kernels are registered and how many mappings exist.
 *
 *  \details Both counts are for the whole process: the handle selects only
 *  the device that is initialized if it is not yet.
 *
 *  @param[in]
 *  handle      Pointer to the allocated hipBLASLt handle.
 *  @param[out]
 *  registered  Number of registered kernels, on every device, or NULL if not
 *              wanted.
 *  @param[out]
 *  selectable  Number of distinct (shape, transposes, data types, kind of
 *              device) with at least one kernel mapped, over every kind of
 *              device in the process, or NULL if not wanted.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS          The counts were written.
 *  \retval HIPBLAS_STATUS_NOT_INITIALIZED  \p handle is NULL.
 *  \retval HIPBLAS_STATUS_INTERNAL_ERROR   The handle's device could not be made
 *                                          current or initialized.
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
