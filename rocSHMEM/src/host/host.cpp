/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "host.hpp"

#include <mpi.h>

#include "rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "host_helpers.hpp"
#include "../memory/window_info.hpp"
#include "../util.hpp"

namespace rocshmem {

__host__ HostContextWindowInfo::HostContextWindowInfo(MPI_Comm comm_world,
                                                      SymmetricHeap* heap) {
  window_info_ =
      new WindowInfo(comm_world, heap->get_local_heap_base(), heap->get_size());
}

__host__ HostContextWindowInfo::~HostContextWindowInfo() {
  delete window_info_;
}

WindowInfo* HostInterface::acquire_window_context() {
  auto index{find_avail_pool_entry()};

  HostContextWindowInfo* acquired_win_info = host_window_context_pool_[index];

  acquired_win_info->mark_unavail();

  return acquired_win_info->get();
}

__host__ void HostInterface::release_window_context(WindowInfo* window_info) {
  auto index{find_win_info_in_pool(window_info)};

  host_window_context_pool_[index]->mark_avail();
}

int HostInterface::find_avail_pool_entry() {
  for (int i{0}; i < max_num_ctxs_; i++) {
    if (host_window_context_pool_[i]->is_avail()) {
      return i;
    }
  }
  /* Entry should have been available; consider this as an error. */
  assert(false);
  return -1;
}

int HostInterface::find_win_info_in_pool(WindowInfo* window_info) {
  for (int i{0}; i < max_num_ctxs_; i++) {
    if (host_window_context_pool_[i]->is_avail()) {
      continue;
    }
    if (window_info == host_window_context_pool_[i]->get()) {
      return i;
    }
  }
  /* Entry should have been present; consider this as an error. */
  assert(false);
  return -1;
}

__host__ HostInterface::HostInterface(HdpPolicy* hdp_policy,
                                      MPI_Comm rocshmem_comm,
                                      SymmetricHeap* heap) {
  /*
   * Duplicate a communicator from roc_shem's comm
   * world for the host interface
   */
  MPI_Comm_dup(rocshmem_comm, &host_comm_world_);
  MPI_Comm_rank(host_comm_world_, &my_pe_);
  MPI_Comm_rank(host_comm_world_, &num_pes_);

  /*
   * Create an MPI window on the HDP so that it can be flushed
   * by remote PEs for host-facing functions
   */
  hdp_policy_ = hdp_policy;

  /*
   * Allocate and initialize pool of windows for contexts
   */
  char* value{nullptr};
  if ((value = getenv("ROCSHMEM_MAX_NUM_HOST_CONTEXTS"))) {
    max_num_ctxs_ = atoi(value);
  }

  size_t pool_size = max_num_ctxs_ * sizeof(HostContextWindowInfo*);
  host_window_context_pool_ =
      reinterpret_cast<HostContextWindowInfo**>(malloc(pool_size));

  for (int ctx_i = 0; ctx_i < max_num_ctxs_; ctx_i++) {
    host_window_context_pool_[ctx_i] =
        new HostContextWindowInfo(host_comm_world_, heap);
  }

#if !defined(USE_COHERENT_HEAP) && !defined(USE_SINGLE_NODE)
  // The single node implementation needs a different path since
  // the HDP flush pointers are allocated on the symmetric heap
  // and we need to wait for other initialization to happen before
  // calling `get_hdp_flush_ptr`.
  create_hdp_window();
#endif  // defined(USE_COHERENT_HEAP) && !defined(USE_SINGLE_NODE)
}

#ifndef USE_COHERENT_HEAP
__host__ void HostInterface::create_hdp_window() {
  MPI_Win_create(hdp_policy_->get_hdp_flush_ptr(),
                 sizeof(unsigned int), /* size of window */
                 sizeof(unsigned int), /* displacement */
                 MPI_INFO_NULL, host_comm_world_, &hdp_win);

  /*
   * Start a shared access epoch on windows of all ranks,
   * and let the library there is no need to check for
   * lock exclusivity during operations on this window
   * (MPI_MODE_NOCHECK).
   */
  MPI_Win_lock_all(MPI_MODE_NOCHECK, hdp_win);
}
#endif  // USE_COHERENT_HEAP

__host__ HostInterface::~HostInterface() {
#ifndef USE_COHERENT_HEAP
  MPI_Win_unlock_all(hdp_win);

  MPI_Win_free(&hdp_win);
#endif  // USE_COHERENT_HEAP

  /* Detroy the pool of contexts */
  for (int ctx_i = 0; ctx_i < max_num_ctxs_; ctx_i++) {
    delete host_window_context_pool_[ctx_i];
  }

  free(host_window_context_pool_);

  MPI_Comm_free(&host_comm_world_);
}

__host__ void HostInterface::putmem_nbi(void* dest, const void* source,
                                        size_t nelems, int pe,
                                        WindowInfo* window_info) {
  initiate_put(dest, source, nelems, pe, window_info);
}

__host__ void HostInterface::getmem_nbi(void* dest, const void* source,
                                        size_t nelems, int pe,
                                        WindowInfo* window_info) {
  initiate_get(dest, source, nelems, pe, window_info);
}

__host__ void HostInterface::putmem(void* dest, const void* source,
                                    size_t nelems, int pe,
                                    WindowInfo* window_info) {
  initiate_put(dest, source, nelems, pe, window_info);

  MPI_Win_flush_local(pe, window_info->get_win());
}

__host__ void HostInterface::getmem(void* dest, const void* source,
                                    size_t nelems, int pe,
                                    WindowInfo* window_info) {
  initiate_get(dest, source, nelems, pe, window_info);

  MPI_Win_flush_local(pe, window_info->get_win());

  /*
   * Flush local HDP to ensure that the NIC's write
   * of the fetched data is visible in device memory
   */
  hdp_policy_->hdp_flush();
}

__host__ void HostInterface::fence(WindowInfo* window_info) {
  complete_all(window_info->get_win());

  /*
   * Flush my HDP and the HDPs of remote GPUs.
   * The HDP is a write-combining (WC) write-through
   * cache. But, even after the WC buffer is full and
   * the data is passed to the Data Fabric (DF), DF
   * can still reorder the writes. A flush ensures
   * that writes after the flush are written only
   * after those before the flush.
   */
  hdp_policy_->hdp_flush();
  flush_remote_hdps();

  return;
}

__host__ void HostInterface::quiet(WindowInfo* window_info) {
  complete_all(window_info->get_win());

  /* Same explanation as in fence */
  hdp_policy_->hdp_flush();
  flush_remote_hdps();

  return;
}

__host__ void HostInterface::sync_all(WindowInfo* window_info) {
  MPI_Win_sync(window_info->get_win());

  hdp_policy_->hdp_flush();
  /*
   * No need to flush remote
   * HDPs here since all PEs are
   * participating.
   */

  MPI_Barrier(host_comm_world_);

  return;
}

__host__ void HostInterface::barrier_all(WindowInfo* window_info) {
  complete_all(window_info->get_win());

  /*
   * Flush my HDP cache so remote NICs will
   * see the latest values in device memory
   */
  hdp_policy_->hdp_flush();

  MPI_Barrier(host_comm_world_);
}

__host__ void HostInterface::barrier_for_sync() {
  MPI_Barrier(host_comm_world_);
}

}  // namespace rocshmem
