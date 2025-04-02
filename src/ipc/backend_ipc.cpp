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

#include "backend_ipc.hpp"
#include "ipc_team.hpp"

namespace rocshmem {

#define NET_CHECK(cmd)                                       \
  {                                                          \
    if (cmd != MPI_SUCCESS) {                                \
      fprintf(stderr, "Unrecoverable error: MPI Failure\n"); \
      abort() ;                                              \
    }                                                        \
  }

extern rocshmem_ctx_t ROCSHMEM_HOST_CTX_DEFAULT;

rocshmem_team_t get_external_team(IPCTeam *team) {
  return reinterpret_cast<rocshmem_team_t>(team);
}

int get_ls_non_zero_bit(char *bitmask, int mask_length) {
  int position = -1;

  for (int bit_i = 0; bit_i < mask_length; bit_i++) {
    int byte_i = bit_i / CHAR_BIT;
    if (bitmask[byte_i] & (1 << (bit_i % CHAR_BIT))) {
      position = bit_i;
      break;
    }
  }

  return position;
}

IPCBackend::IPCBackend(MPI_Comm comm)
    :  Backend() {
  type = BackendType::IPC_BACKEND;

  if (auto maximum_num_contexts_str = getenv("ROCSHMEM_MAX_NUM_CONTEXTS")) {
    std::stringstream sstream(maximum_num_contexts_str);
    sstream >> maximum_num_contexts_;
  }

  init_mpi_once(comm);

  initIPC();

  /**
   * Check if num_pes == ipcImpl.shm_size)
   * All the PEs must be with in a node for IPC conduit
   */
  assert(num_pes == ipcImpl.shm_size);

  /* Initialize the host interface */
  host_interface = std::make_shared<HostInterface>(hdp_proxy_.get(),
                                                 thread_comm,
                                                 &heap);

  default_host_ctx = std::make_unique<IPCHostContext>(this, 0);

  ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque = default_host_ctx.get();

  setup_team_world();

  init_wrk_sync_buffer();

  rocshmem_collective_init();

  setup_fence_buffer();

  teams_init();

  TeamInfo *tinfo = team_tracker.get_team_world()->tinfo_wrt_world;

  default_context_proxy_ = IPCDefaultContextProxyT(this, tinfo);

  setup_ctxs();
}

IPCBackend::~IPCBackend() {
  /**
   * Destroy teams infrastructure
   * and team world
   */
  teams_destroy();
  cleanup_wrk_sync_buffer();
  auto *team_world{team_tracker.get_team_world()};
  team_world->~Team();
  CHECK_HIP(hipFree(team_world));

  CHECK_HIP(hipFree(ctx_array));
}

void IPCBackend::setup_ctxs() {
  CHECK_HIP(hipMalloc(&ctx_array, sizeof(IPCContext) * maximum_num_contexts_));
  // 0th context is default context
  for (size_t i = 0; i < maximum_num_contexts_; i++) {
    new (&ctx_array[i]) IPCContext(this, i + 1);
    ctx_free_list.get()->push_back(ctx_array + i);
  }
}

__device__ bool IPCBackend::create_ctx(int64_t options, rocshmem_ctx_t *ctx) {
  IPCContext *ctx_{nullptr};

  auto pop_result = ctx_free_list.get()->pop_front();
  if (!pop_result.success) {
    return false;
  }
  ctx_ = pop_result.value;

  ctx->ctx_opaque = ctx_;

  ctx_->tinfo = reinterpret_cast<TeamInfo *>(ctx->team_opaque);
  return true;
}

__device__ void IPCBackend::destroy_ctx(rocshmem_ctx_t *ctx) {
  ctx_free_list.get()->push_back(static_cast<IPCContext *>(ctx->ctx_opaque));
}

void IPCBackend::setup_team_world() {
  TeamInfo *team_info_wrt_parent, *team_info_wrt_world;

  /**
   * Allocate device-side memory for team_world and construct a
   * IPC team in it.
   */
  CHECK_HIP(hipMalloc(&team_info_wrt_parent, sizeof(TeamInfo)));
  CHECK_HIP(hipMalloc(&team_info_wrt_world, sizeof(TeamInfo)));

  new (team_info_wrt_parent) TeamInfo(nullptr, 0, 1, num_pes);
  new (team_info_wrt_world) TeamInfo(nullptr, 0, 1, num_pes);

  IPCTeam *team_world{nullptr};
  CHECK_HIP(hipMalloc(&team_world, sizeof(IPCTeam)));
  new (team_world) IPCTeam(this, team_info_wrt_parent, team_info_wrt_world,
                             num_pes, my_pe, thread_comm, 0);
  team_tracker.set_team_world(team_world);

  /**
   * Copy the address to ROCSHMEM_TEAM_WORLD.
   */
  ROCSHMEM_TEAM_WORLD = reinterpret_cast<rocshmem_team_t>(team_world);
}

void IPCBackend::init_mpi_once(MPI_Comm comm) {
  int init_done{};
  NET_CHECK(MPI_Initialized(&init_done));

  int provided{};
  if (!init_done) {
    NET_CHECK(MPI_Init_thread(0, 0, MPI_THREAD_MULTIPLE, &provided));
    if (provided != MPI_THREAD_MULTIPLE) {
      std::cerr << "MPI_THREAD_MULTIPLE support disabled.\n";
    }
  }
  if (comm == MPI_COMM_NULL) comm = MPI_COMM_WORLD;

  NET_CHECK(MPI_Comm_dup(comm, &thread_comm));
  NET_CHECK(MPI_Comm_size(thread_comm, &num_pes));
  NET_CHECK(MPI_Comm_rank(thread_comm, &my_pe));
}

void IPCBackend::team_destroy(rocshmem_team_t team) {
  IPCTeam *team_obj = get_internal_ipc_team(team);

  /* Mark the pool as available */
  int bit = team_obj->pool_index_;
  int byte_i = bit / CHAR_BIT;
  pool_bitmask_[byte_i] |= 1 << (bit % CHAR_BIT);

  team_obj->~IPCTeam();
  CHECK_HIP(hipFree(team_obj));
}

void IPCBackend::create_new_team([[maybe_unused]] Team *parent_team,
                                TeamInfo *team_info_wrt_parent,
                                TeamInfo *team_info_wrt_world, int num_pes,
                                int my_pe_in_new_team, MPI_Comm team_comm,
                                rocshmem_team_t *new_team) {
  /**
   * Read the bit mask and find out a common index into
   * the pool of available work arrays.
   */
  NET_CHECK(MPI_Allreduce(pool_bitmask_, reduced_bitmask_, bitmask_size_,
                          MPI_CHAR, MPI_BAND, team_comm));

  /* Pick the least significant non-zero bit (logical layout) in the reduced
   * bitmask */
  auto max_num_teams{team_tracker.get_max_num_teams()};
  int common_index = get_ls_non_zero_bit(reduced_bitmask_, max_num_teams);
  if (common_index < 0) {
    /* No team available */
    abort();
  }

  /* Mark the team as taken (by unsetting the bit in the pool bitmask) */
  int byte = common_index / CHAR_BIT;
  pool_bitmask_[byte] &= ~(1 << (common_index % CHAR_BIT));

  /**
   * Allocate device-side memory for team_world and
   * construct a IPC team in it
   */
  IPCTeam *new_team_obj;
  CHECK_HIP(hipMalloc(&new_team_obj, sizeof(IPCTeam)));
  new (new_team_obj)
      IPCTeam(this, team_info_wrt_parent, team_info_wrt_world, num_pes,
                my_pe_in_new_team, team_comm, common_index);

  *new_team = get_external_team(new_team_obj);
}

void IPCBackend::ctx_create(int64_t options, void **ctx) {
  IPCHostContext *new_ctx{nullptr};
  new_ctx = new IPCHostContext(this, options);
  *ctx = new_ctx;
}

IPCHostContext *get_internal_ipc_net_ctx(Context *ctx) {
  return reinterpret_cast<IPCHostContext *>(ctx);
}

void IPCBackend::ctx_destroy(Context *ctx) {
  IPCHostContext *ro_net_host_ctx{get_internal_ipc_net_ctx(ctx)};
  delete ro_net_host_ctx;
}

void IPCBackend::reset_backend_stats() {
  assert(false);
}

void IPCBackend::dump_backend_stats() {
  assert(false);
}

void IPCBackend::initIPC() {
  const auto &heap_bases{heap.get_heap_bases()};

  ipcImpl.ipcHostInit(my_pe, heap_bases,
                      thread_comm);
}

void IPCBackend::global_exit(int status) {
  MPI_Abort(MPI_COMM_WORLD, status);
}

void IPCBackend::teams_destroy() {
  free(pool_bitmask_);
  free(reduced_bitmask_);
}

void IPCBackend::init_wrk_sync_buffer() {
  /**
   * calcualte work/sync buffer size
   */
  auto max_num_teams{team_tracker.get_max_num_teams()};

  /**
   * size of barrier sync for all the contexts
   */
  Wrk_Sync_buffer_size_ += sizeof(*barrier_sync) * ROCSHMEM_BARRIER_SYNC_SIZE *
                           (maximum_num_contexts_ + 1);

  /**
   * Size of sync arrays for the teams
  */
  Wrk_Sync_buffer_size_ += sizeof(long) * max_num_teams *
                           (ROCSHMEM_BARRIER_SYNC_SIZE +
                            ROCSHMEM_REDUCE_SYNC_SIZE +
                            ROCSHMEM_BCAST_SYNC_SIZE +
                            ROCSHMEM_ALLTOALL_SYNC_SIZE);

  /**
   * Size of work arrays for the teams
   * Accommodate largest possible data type for pWrk
  */
  Wrk_Sync_buffer_size_ += sizeof(double) * max_num_teams *
                           (ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE +
                            ROCSHMEM_ATA_MAX_WRKDATA_SIZE);

  /**
   * Size of fence array
  */
  Wrk_Sync_buffer_size_ += sizeof(int) * num_pes;

  /**
   * Allocate a buffer of size Wrk_Sync_buffer_size_, using fine-grained
   * memory allocator
  */
  fine_grained_allocator_.allocate((void**)&Wrk_Sync_buffer_ptr_,
                                   Wrk_Sync_buffer_size_);
  assert(Wrk_Sync_buffer_ptr_);
  temp_Wrk_Sync_buff_ptr_ = Wrk_Sync_buffer_ptr_;

  /*
   * Allocate a c-array to hold the IPC handles
   */
  hipIpcMemHandle_t *ipc_handle = reinterpret_cast<hipIpcMemHandle_t*>(
            malloc(num_pes * sizeof(hipIpcMemHandle_t)));

  /*
   * Call into the hip runtime to get an IPC handle for the allocated
   * Wrk_Sync_buffer_ and store that IPC handle
   */
  CHECK_HIP(hipIpcGetMemHandle(&ipc_handle[my_pe], Wrk_Sync_buffer_ptr_));

  /*
   * all-to-all exchange with each PE to share the IPC handles.
   */
  MPI_Allgather(MPI_IN_PLACE, sizeof(hipIpcMemHandle_t), MPI_CHAR,
                ipc_handle, sizeof(hipIpcMemHandle_t), MPI_CHAR, thread_comm);

  /*
   * Allocate device-side fine grained memory to hold IPC addresses of
   * work/sync buffers
   */
  fine_grained_allocator_.allocate(
    reinterpret_cast<void**>(&Wrk_Sync_buffer_bases_),
    num_pes * sizeof(char*));
  assert(Wrk_Sync_buffer_bases_);

  /*
   * For all local processing elements, initialize the device-side array
   * with the IPC work/sync buffer addresses.
   */
  for (int i = 0; i < num_pes; i++) {
    if (i != my_pe) {
      CHECK_HIP(hipIpcOpenMemHandle(
          reinterpret_cast<void**>(&Wrk_Sync_buffer_bases_[i]),
          ipc_handle[i],
          hipIpcMemLazyEnablePeerAccess));
    } else {
      Wrk_Sync_buffer_bases_[i] = Wrk_Sync_buffer_ptr_;
    }
  }
}

void IPCBackend::cleanup_wrk_sync_buffer() {
  for (int i = 0; i < num_pes; i++) {
    if (i != my_pe) {
      CHECK_HIP(hipIpcCloseMemHandle(Wrk_Sync_buffer_bases_[i]));
    }
  }
  fine_grained_allocator_.deallocate(Wrk_Sync_buffer_bases_);
  fine_grained_allocator_.deallocate(Wrk_Sync_buffer_ptr_);
}

void IPCBackend::setup_fence_buffer() {
  /*
  * Allocate memory for fence
  */
  fence_pool = reinterpret_cast<int *>(temp_Wrk_Sync_buff_ptr_);
  temp_Wrk_Sync_buff_ptr_ += sizeof(int) * num_pes;
}

void IPCBackend::rocshmem_collective_init() {
  /*
   * Allocate heap space for barrier_sync
   */
  size_t one_sync_size_bytes {sizeof(*barrier_sync)};
  size_t total_sync_elems {
    ROCSHMEM_BARRIER_SYNC_SIZE * (maximum_num_contexts_ + 1)};
  size_t sync_size_bytes {one_sync_size_bytes * total_sync_elems};

  barrier_sync = reinterpret_cast<int64_t*>(temp_Wrk_Sync_buff_ptr_);
  temp_Wrk_Sync_buff_ptr_ += sync_size_bytes;

  /*
   * Initialize the barrier synchronization array with default values.
   */
  for (int i = 0; i < total_sync_elems; i++) {
    barrier_sync[i] = ROCSHMEM_SYNC_VALUE;
  }

  /*
   * Make sure that all processing elements have done this before
   * continuing.
   */
  NET_CHECK(MPI_Barrier(thread_comm));
}

void IPCBackend::teams_init() {
  /**
   * Allocate pools for the teams sync and work arrary from the SHEAP.
   */
  auto max_num_teams{team_tracker.get_max_num_teams()};

  barrier_pSync_pool = reinterpret_cast<long *>(temp_Wrk_Sync_buff_ptr_);
  temp_Wrk_Sync_buff_ptr_ += sizeof(long) * ROCSHMEM_BARRIER_SYNC_SIZE
                            * max_num_teams;

  reduce_pSync_pool = reinterpret_cast<long *>(temp_Wrk_Sync_buff_ptr_);
  temp_Wrk_Sync_buff_ptr_ += sizeof(long) * ROCSHMEM_REDUCE_SYNC_SIZE
                            * max_num_teams;

  bcast_pSync_pool = reinterpret_cast<long *>(temp_Wrk_Sync_buff_ptr_);
  temp_Wrk_Sync_buff_ptr_ += sizeof(long) * ROCSHMEM_BCAST_SYNC_SIZE
                            * max_num_teams;

  alltoall_pSync_pool = reinterpret_cast<long *>(temp_Wrk_Sync_buff_ptr_);
  temp_Wrk_Sync_buff_ptr_ += sizeof(long) * ROCSHMEM_BCAST_SYNC_SIZE
                            * max_num_teams;

  /* Accommodating for largest possible data type for pWrk */
  pWrk_pool = reinterpret_cast<void *>(temp_Wrk_Sync_buff_ptr_);
  temp_Wrk_Sync_buff_ptr_ += sizeof(double) * ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE
                            * max_num_teams;


  pAta_pool = reinterpret_cast<void *>(temp_Wrk_Sync_buff_ptr_);
  temp_Wrk_Sync_buff_ptr_ += sizeof(double) * ROCSHMEM_ATA_MAX_WRKDATA_SIZE
                            * max_num_teams;

  /**
   * Initialize the sync arrays in the pool with default values.
   */
  long *barrier_pSync, *reduce_pSync, *bcast_pSync, *alltoall_pSync;
  for (int team_i = 0; team_i < max_num_teams; team_i++) {
    barrier_pSync = reinterpret_cast<long *>(
        &barrier_pSync_pool[team_i * ROCSHMEM_BARRIER_SYNC_SIZE]);
    reduce_pSync = reinterpret_cast<long *>(
        &reduce_pSync_pool[team_i * ROCSHMEM_REDUCE_SYNC_SIZE]);
    bcast_pSync = reinterpret_cast<long *>(
        &bcast_pSync_pool[team_i * ROCSHMEM_BCAST_SYNC_SIZE]);
    alltoall_pSync = reinterpret_cast<long *>(
        &alltoall_pSync_pool[team_i * ROCSHMEM_ALLTOALL_SYNC_SIZE]);

    for (size_t i = 0; i < ROCSHMEM_BARRIER_SYNC_SIZE; i++) {
      barrier_pSync[i] = ROCSHMEM_SYNC_VALUE;
    }
    for (size_t i = 0; i < ROCSHMEM_REDUCE_SYNC_SIZE; i++) {
      reduce_pSync[i] = ROCSHMEM_SYNC_VALUE;
    }
    for (size_t i = 0; i < ROCSHMEM_BCAST_SYNC_SIZE; i++) {
      bcast_pSync[i] = ROCSHMEM_SYNC_VALUE;
    }
    for (size_t i = 0; i < ROCSHMEM_ALLTOALL_SYNC_SIZE; i++) {
      alltoall_pSync[i] = ROCSHMEM_SYNC_VALUE;
    }
  }

  /**
   * Initialize bit mask
   *
   * Logical:
   * MSB..........................................................................LSB
   * Physical: MSB...1st least significant 8 bits...LSB  MSB...2nd least
   * signifant 8 bits...LSB
   *
   * Description shows only a 2-byte long mask but idea extends to any
   * arbitrary size.
   */
  bitmask_size_ = (max_num_teams % CHAR_BIT) ? (max_num_teams / CHAR_BIT + 1)
                                             : (max_num_teams / CHAR_BIT);
  pool_bitmask_ = reinterpret_cast<char *>(malloc(bitmask_size_));
  reduced_bitmask_ = reinterpret_cast<char *>(malloc(bitmask_size_));

  memset(pool_bitmask_, 0, bitmask_size_);
  memset(reduced_bitmask_, 0, bitmask_size_);
  /* Set all to available except the 0th one (reserved for TEAM_WORLD) */
  for (int bit_i = 1; bit_i < max_num_teams; bit_i++) {
    int byte_i = bit_i / CHAR_BIT;

    pool_bitmask_[byte_i] |= 1 << (bit_i % CHAR_BIT);
  }

  /**
   * Make sure that all processing elements have done this before
   * continuing.
   */
  NET_CHECK(MPI_Barrier(thread_comm));
}

}  // namespace rocshmem
