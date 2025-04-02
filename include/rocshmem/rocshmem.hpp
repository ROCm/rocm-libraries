/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_INCLUDE_ROCSHMEM_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_HPP

#include <hip/hip_runtime.h>
#include <mpi.h>

#include "rocshmem_config.h"
#include "rocshmem_common.hpp"
#include "rocshmem_RMA.hpp"
#include "rocshmem_AMO.hpp"
#include "rocshmem_SIG_OP.hpp"
#include "rocshmem_COLL.hpp"
#include "rocshmem_P2P_SYNC.hpp"
#include "rocshmem_RMA_X.hpp"
/**
 * @file rocshmem.hpp
 * @brief Public header for rocSHMEM device and host libraries.
 *
 * This file contains all the callable functions and data structures for both
 * the device-side runtime and host-side runtime.
 *
 * The comments on these functions are sparse, but the semantics are the same
 * as those implemented in OpenSHMEM unless otherwise documented. Please see
 * the OpenSHMEM 1.4 standards documentation for more details:
 *
 * http://openshmem.org/site/sites/default/site_files/OpenSHMEM-1.4.pdf
 */

namespace rocshmem {

/******************************************************************************
 **************************** HOST INTERFACE **********************************
 *****************************************************************************/
/**
 * @brief Initialize the rocSHMEM runtime and underlying transport layer.
 *
 * @param[in] comm      (Optional) MPI Communicator that rocSHMEM will be using
 *                      If MPI_COMM_NULL, rocSHMEM will be using MPI_COMM_WORLD
 */
__host__ void rocshmem_init(MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Initialize the rocSHMEM runtime and underlying transport layer
 *        with an attempt to enable the requested thread support.
 *
 * @param[in] requested Requested thread mode (from rocshmem_thread_ops)
 *                      for host-facing functions.
 * @param[out] provided Thread mode selected by the runtime. May not be equal
 *                      to requested thread mode.
 * @param[in] comm      (Optional) MPI Communicator that rocSHMEM will be using
 *                      If MPI_COMM_NULL, rocSHMEM will be using MPI_COMM_WORLD
 *
 * @return int          returns 0 upon success; otherwise, it returns a nonzero
 *                      value
 */
__host__ int rocshmem_init_thread(int requested, int *provided,
                                  MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Initialize the rocSHMEM runtime and underlying transport layer
 *        using the provided mode and attributes
 *
 * @param[in] flags   initialization method to be used.
 *                    Valid values are ROCSHMEM_INIT_WITH_UNIQUEID and
 *                    ROCSHMEM_INIT_WITH_MPI_COMM
 * @param[in] attr    attribute structure specifying input characteristics
 *
 * @return int        returns 0 upon success; otherwise, it returns a nonzero
 *                    value
 */
__host__ int rocshmem_init_attr(unsigned int flags, rocshmem_init_attr_t *attr);

/**
 * @brief Return a uniqueID
 *
 * @return int        returns 0 upon success; otherwise, it returns a nonzero
 *                    value
 */
__host__ int rocshmem_get_uniqueid(rocshmem_uniqueid_t *uid);

/**
 * @brief Query the thread mode used by the runtime.
 *
 * @param[in] rank     rank of the calling process
 * @param[in] nranks   number of pes
 * @param[in] uid      unique ID used to identify the group processes.
 *                     All processes that
 * @param[out] attr    attribute structure to be passed to rocshmem_init_attr
 *
 * @return int         returns 0 upon success; otherwise, it returns a nonzero
 *                     value
 */
__host__ int rocshmem_set_attr_uniqueid_args(int rank, int nranks,
                                             rocshmem_uniqueid_t *uid,
                                             rocshmem_init_attr_t *attr);
/**
 * @brief Query the thread mode used by the runtime.
 *
 * @param[out] provided Thread mode the runtime is operating in.
 *
 * @return void.
 */
__host__ void rocshmem_query_thread(int *provided);

/**
 * @brief Function that dumps internal stats to stdout.
 */
__host__ void rocshmem_dump_stats();

/**
 * @brief Reset all internal stats.
 */
__host__ void rocshmem_reset_stats();

/**
 * @brief Finalize the rocSHMEM runtime.
 */
__host__ void rocshmem_finalize();

/**
 * @brief Allocate memory of \p size bytes from the symmetric heap.
 * This is a collective operation and must be called by all PEs.
 *
 * @param[in] size Memory allocation size in bytes.
 *
 * @return A pointer to the allocated memory on the symmetric heap.
 *
 * @todo Return error code instead of ptr.
 */
__host__ void *rocshmem_malloc(size_t size);

/**
 * @brief Free a memory allocation from the symmetric heap.
 * This is a collective operation and must be called by all PEs.
 *
 * @param[in] ptr Pointer to previously allocated memory on the symmetric heap.
 */
__host__ void rocshmem_free(void *ptr);

/**
 * @brief Query for the number of PEs.
 *
 * @return Number of PEs.
 */
__host__ int rocshmem_n_pes();

/**
 * @brief Query the PE ID of the caller.
 *
 * @return PE ID of the caller.
 */
__host__ int rocshmem_my_pe();

/**
 * @brief Creates an OpenSHMEM context.
 *
 * @param[in] options Options for context creation. Ignored in current design.
 * @param[out] ctx    Context handle.
 *
 * @return Zero on success and nonzero otherwise.
 */
__host__ int rocshmem_ctx_create(int64_t options, rocshmem_ctx_t *ctx);

/**
 * @brief Destroys an OpenSHMEM context.
 *
 * @param[out] ctx    Context handle.
 *
 * @return void.
 */
__host__ void rocshmem_ctx_destroy(rocshmem_ctx_t ctx);

/**
 * @brief Translate the PE in src_team to that in dest_team.
 *
 * @param[in] src_team  Handle of the team from which to translate
 * @param[in] src_pe    PE-of-interest's index in src_team
 * @param[in] dest_team Handle of the team to which to translate
 *
 * @return PE of src_pe in dest_team. If any input is invalid
 *         or if src_pe is not in both source and destination
 *         teams, a value of -1 is returned.
 */
__host__ int rocshmem_team_translate_pe(rocshmem_team_t src_team, int src_pe,
                                         rocshmem_team_t dest_team);

/**
 * @brief Query the number of PEs in a team.
 *
 * @param[in] team The team to query PE ID in.
 *
 * @return Number of PEs in the provided team.
 */
__host__ int rocshmem_team_n_pes(rocshmem_team_t team);

/**
 * @brief Query the PE ID of the caller in a team.
 *
 * @param[in] team The team to query PE ID in.
 *
 * @return PE ID of the caller in the provided team.
 */
__host__ int rocshmem_team_my_pe(rocshmem_team_t team);

/**
 * @brief Create a new a team of PEs. Must be called by all PEs
 * in the parent team.
 *
 * @param[in] parent_team The team to split from.
 * @param[in] start       The lowest PE number of the subset of the PEs
 *                        from the parent team that will form the new
 *                        team.
 * @param[in] stide       The stride between team PE members in the
 *                        parent team that comprise the subset of PEs
 *                        that will form the new team.
 * @param[in] size        The number of PEs in the new team.
 * @param[in] config      Pointer to the config parameters for the new
 *                        team.
 * @param[in] config_mask Bitwise mask representing parameters to use
 *                        from config
 * @param[out] new_team   Pointer to the newly created team. If an error
 *                        occurs during team creation, or if the PE in
 *                        the parent team is not in the new team, the
 *                        value will be ROCSHMEM_TEAM_INVALID.
 *
 * @return Zero upon successful team creation; non-zero if erroneous.
 */
__host__ int rocshmem_team_split_strided(rocshmem_team_t parent_team,
                                          int start, int stride, int size,
                                          const rocshmem_team_config_t *config,
                                          long config_mask,
                                          rocshmem_team_t *new_team);

/**
 * @brief Destroy a team. Must be called by all PEs in the team.
 * The user must destroy all private contexts created in the
 * team before destroying this team. Otherwise, the behavior
 * is undefined. This call will destroy only the shareable contexts
 * created from the referenced team.
 *
 * @param[in] team The team to destroy. The behavior is undefined if
 *                 the input team is ROCSHMEM_TEAM_WORLD or any other
 *                 invalid team. If the input is ROCSHMEM_TEAM_INVALID,
 *                 this function will not perform any operation.
 *
 * @return None.
 */
__host__ void rocshmem_team_destroy(rocshmem_team_t team);

/**
 * @brief Guarantees order between messages in this context in accordance with
 * OpenSHMEM semantics.
 *
 * @param[in] ctx     Context with which to perform this operation.
 *
 * @return void.
 */
__host__ void rocshmem_ctx_fence(rocshmem_ctx_t ctx);

__host__ void rocshmem_fence();

/**
 * @brief Completes all previous operations posted on the host.
 *
 * @param[in] ctx     Context with which to perform this operation.
 *
 * @return void.
 */
__host__ void rocshmem_ctx_quiet(rocshmem_ctx_t ctx);

__host__ void rocshmem_quiet();

/**
 * @brief perform a collective barrier between all PEs in the system.
 * The caller is blocked until the barrier is resolved.
 *
 * @return void
 */
__host__ void rocshmem_barrier_all();

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_sync_all only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * @return void
 */
__host__ void rocshmem_sync_all();

/**
 * @brief allows any PE to force the termination of an entire program.
 *
 * @param[in] status    The exit status from the main program.
 *
 * @return void
 */
__host__ void rocshmem_global_exit(int status);

/******************************************************************************
 **************************** DEVICE INTERFACE ********************************
 *****************************************************************************/

/**
 * @brief Initializes device-side rocSHMEM resources. Must be called before
 * any threads in this work-group invoke other rocSHMEM functions.
 *
 * Must be called collectively by all threads in the work-group.
 *
 * @return void.
 */
__device__ void rocshmem_wg_init();

/**
 * @brief Finalizes device-side rocSHMEM resources. Must be called before
 * work-group completion if the work-group also called rocshmem_wg_init().
 *
 * Must be called collectively by all threads in the work-group.
 *
 * @return void.
 */
__device__ void rocshmem_wg_finalize();

/**
 * @brief Initializes device-side rocSHMEM resources. Must be called before
 * any threads in this work-group invoke other rocSHMEM functions. This is
 * a variant of rocshmem_wg_init that allows the caller to request a
 * threading mode.
 *
 * @param[in] requested Requested thread mode from rocshmem_thread_ops.
 * @param[out] provided Thread mode selected by the runtime. May not be equal
 *                      to requested thread mode.
 *
 * Must be called collectively by all threads in the work-group.
 *
 * @return void.
 */
__device__ void rocshmem_wg_init_thread(int requested, int *provided);

/**
 * @brief Query the thread mode used by the runtime.
 *
 * @param[out] provided Thread mode the runtime is operating in.
 *
 * @return void.
 */
__device__ void rocshmem_query_thread(int *provided);

/**
 * @brief Creates an OpenSHMEM context. By design, the context is private
 * to the calling work-group.
 *
 * Must be called collectively by all threads in the work-group.
 *
 * @param[in] options Options for context creation. Ignored in current design.
 * @param[out] ctx    Context handle.
 *
 * @return All threads returns 0 if the context was created successfully. If any
 * thread returns non-zero value, the operation failed and a higher number of
 * `ROCSHMEM_MAX_NUM_CONTEXTS` is required.
 */
__device__ ATTR_NO_INLINE int rocshmem_wg_ctx_create(int64_t options,
                                                      rocshmem_ctx_t *ctx);

__device__ ATTR_NO_INLINE int rocshmem_wg_team_create_ctx(
    rocshmem_team_t team, long options, rocshmem_ctx_t *ctx);

/**
 * @brief Destroys an OpenSHMEM context.
 *
 * Must be called collectively by all threads in the work-group.
 *
 * @param[in] The context to destroy.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_wg_ctx_destroy(rocshmem_ctx_t *ctx);

/**
 * @brief Guarantees order between messages in this context in accordance with
 * OpenSHMEM semantics.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * rocSHMEM function.
 *
 * @param[in] ctx Context with which to perform this operation.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_fence(rocshmem_ctx_t ctx);

__device__ ATTR_NO_INLINE void rocshmem_fence();

/**
 * @brief Guarantees order between messages in this context in accordance with
 * OpenSHMEM semantics.
 *
 * This function  is an extension as it is per PE. has same semantics as default
 * API but it is per PE
 *
 * @param[in] ctx Context with which to perform this operation.
 * @param[in] pe destination pe.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_fence(rocshmem_ctx_t ctx, int pe);

__device__ ATTR_NO_INLINE void rocshmem_fence(int pe);

/**
 * @brief Completes all previous operations posted to this context.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * rocSHMEM function.
 *
 * @param[in] ctx Context with which to perform this operation.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_quiet(rocshmem_ctx_t ctx);

__device__ ATTR_NO_INLINE void rocshmem_quiet();

/**
 * @brief Query the total number of PEs.
 *
 * Can be called per thread with no performance penalty.
 *
 * @param[in] ctx GPU side handle.
 *
 * @return Total number of PEs.
 */
__device__ int rocshmem_ctx_n_pes(rocshmem_ctx_t ctx);

__device__ int rocshmem_n_pes();

/**
 * @brief Query the PE ID of the caller.
 *
 * Can be called per thread with no performance penalty.
 *
 * @param[in] ctx GPU side handle
 *
 * @return PE ID of the caller.
 */
__device__ int rocshmem_ctx_my_pe(rocshmem_ctx_t ctx);

__device__ int rocshmem_my_pe();

/**
 * @brief Translate the PE in src_team to that in dest_team.
 *
 * @param[in] src_team  Handle of the team from which to translate
 * @param[in] src_pe    PE-of-interest's index in src_team
 * @param[in] dest_team Handle of the team to which to translate
 *
 * @return PE of src_pe in dest_team. If any input is invalid
 *         or if src_pe is not in both source and destination
 *         teams, a value of -1 is returned.
 */
__device__ int rocshmem_team_translate_pe(rocshmem_team_t src_team,
                                           int src_pe,
                                           rocshmem_team_t dest_team);

/**
 * @brief perform a collective barrier between all PEs in the system.
 * The caller is blocked until the barrier is resolved.
 *
 * This function must be invoked by a single thread within the PE.
 *
 * @param[in] handle GPU side handle.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_barrier_all(
    rocshmem_ctx_t ctx);

__device__ ATTR_NO_INLINE void rocshmem_barrier_all();

/**
 * @brief perform a collective barrier between all PEs in the system.
 * The caller is blocked until the barrier is resolved.
 *
 * This function must be called as a wave-front collective.
 *
 * @param[in] handle GPU side handle.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_wave_barrier_all(
    rocshmem_ctx_t ctx);

__device__ ATTR_NO_INLINE void rocshmem_wave_barrier_all();

/**
 * @brief perform a collective barrier between all PEs in the system.
 * The caller is blocked until the barrier is resolved.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] handle GPU side handle.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_wg_barrier_all(
    rocshmem_ctx_t ctx);

__device__ ATTR_NO_INLINE void rocshmem_wg_barrier_all();

/**
 * @brief perform a collective barrier between all PEs in the team.
 * The caller is blocked until the barrier is resolved.
 *
 * @param[in] team    The team on which to perform barrier synchronization
 *
 * @return void
 */
__device__ void rocshmem_barrier(rocshmem_team_t);

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_sync_all only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * This function must be invoked by a single thread within the PE.
 *
 * @param[in] handle GPU side handle.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_sync_all(rocshmem_ctx_t ctx);

__device__ ATTR_NO_INLINE void rocshmem_sync_all();

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_sync_all only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * This function must be called as a wave-front collective.
 *
 * @param[in] handle GPU side handle.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_wave_sync_all(rocshmem_ctx_t ctx);

__device__ ATTR_NO_INLINE void rocshmem_wave_sync_all();

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_sync_all only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] handle GPU side handle.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_wg_sync_all(rocshmem_ctx_t ctx);

__device__ ATTR_NO_INLINE void rocshmem_wg_sync_all();

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_team_sync only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] handle GPU side handle.
 * @param[in] team  Handle of the team being synchronized
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_wg_team_sync(
    rocshmem_ctx_t ctx, rocshmem_team_t team);

__device__ ATTR_NO_INLINE void rocshmem_wg_team_sync(rocshmem_team_t team);

/**
 * @brief Query a local pointer to a symmetric data object on the
 * specified \pe . Returns an address that may be used to directly reference
 * dest on the specified \pe. This address can be accesses with LD/ST ops.
 *
 * Can be called per thread with no performance penalty.
 */
__device__ ATTR_NO_INLINE void *rocshmem_ptr(const void *dest, int pe);

/**
 * @brief Make all uncacheable GPU data visible to other agents in the sytem.
 *
 * This only works for data that was explicitly allocated uncacheable on the
 * GPU!
 *
 * Can be called per thread with no performance penalty.
 *
 * @param[in] GPU-side handle.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_threadfence_system(
    rocshmem_ctx_t ctx);

__device__ ATTR_NO_INLINE void rocshmem_threadfence_system();

}  // namespace rocshmem

#endif  // LIBRARY_INCLUDE_ROCSHMEM_HPP
