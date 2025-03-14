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

#include "mpi_transport.hpp"

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "../host/host.hpp"
#include "backend_ro.hpp"
#include "ro_net_team.hpp"
#include "../util.hpp"

namespace rocshmem {

#define NET_CHECK(cmd)                                       \
  {                                                          \
    if (cmd != MPI_SUCCESS) {                                \
      fprintf(stderr, "Unrecoverable error: MPI Failure\n"); \
      abort() ;                                              \
    }                                                        \
  }

MPITransport::MPITransport(MPI_Comm comm, Queue* q)
  : queue{q}, Transport{} {
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

  NET_CHECK(MPI_Comm_dup(comm, &ro_net_comm_world));
  NET_CHECK(MPI_Comm_size(ro_net_comm_world, &num_pes));
  NET_CHECK(MPI_Comm_rank(ro_net_comm_world, &my_pe));
}

MPITransport::~MPITransport() {}

void MPITransport::threadProgressEngine() {
  auto *bp{backend_proxy->get()};

  transport_up = true;
  while (!(bp->worker_thread_exit)) {
    submitRequestsToMPI();
    progress();
  }
  transport_up = false;
}

void MPITransport::insertRequest(const queue_element_t *element, int queue_id) {
  std::unique_lock<std::mutex> mlock(queue_mutex);
  q.push(*element);
  q_wgid.push(queue_id);
}

void MPITransport::submitRequestsToMPI() {
  if (q.empty()) return;

  std::unique_lock<std::mutex> mlock(queue_mutex);
  queue_element_t next_element{q.front()};
  int queue_idx{q_wgid.front()};
  q.pop();
  q_wgid.pop();
  mlock.unlock();

  switch (next_element.type) {
    case RO_NET_PUT:
      putMem(next_element.dst, next_element.src, next_element.ol1.size,
             next_element.PE, next_element.ro_net_win_id, queue_idx,
             next_element.status, true);
      DPRINTF("Received PUT dst %p src %p size %lu pe %d win_id %d\n",
              next_element.dst, next_element.src, next_element.ol1.size,
              next_element.PE, next_element.ro_net_win_id);
      break;
    case RO_NET_P: {
      // No equivalent inline OP for MPI.
      // Allocate a temp buffer for value.
      // TODO(bpotter) this is a memory leak - fix it
      void *source_buffer{malloc(next_element.ol1.size)};

      ::memcpy(source_buffer, &next_element.src, next_element.ol1.size);

      putMem(next_element.dst, source_buffer, next_element.ol1.size,
             next_element.PE, next_element.ro_net_win_id, queue_idx,
             next_element.status, true, true);
      DPRINTF("Received P dst %p value %p pe %d\n", next_element.dst,
              next_element.src, next_element.PE);
      break;
    }
    case RO_NET_GET:
      getMem(next_element.dst, next_element.src, next_element.ol1.size,
             next_element.PE, next_element.ro_net_win_id, queue_idx,
             next_element.status, true);
      DPRINTF("Received GET dst %p src %p size %lu pe %d\n", next_element.dst,
              next_element.src, next_element.ol1.size, next_element.PE);
      break;
    case RO_NET_PUT_NBI:
      putMem(next_element.dst, next_element.src, next_element.ol1.size,
             next_element.PE, next_element.ro_net_win_id, queue_idx,
             next_element.status, false);
      DPRINTF("Received PUT NBI dst %p src %p size %lu pe %d\n",
              next_element.dst, next_element.src, next_element.ol1.size,
              next_element.PE);
      break;
    case RO_NET_GET_NBI:
      getMem(next_element.dst, next_element.src, next_element.ol1.size,
             next_element.PE, next_element.ro_net_win_id, queue_idx,
             next_element.status, false);
      DPRINTF("Received GET NBI dst %p src %p size %lu pe %d\n",
              next_element.dst, next_element.src, next_element.ol1.size,
              next_element.PE);
      break;
    case RO_NET_AMO_FOP:
      amoFOP(next_element.dst, next_element.src,
             const_cast<unsigned long long *>(&next_element.ol1.atomic_value),
             next_element.PE, next_element.ro_net_win_id, queue_idx,
             next_element.status, true,
             static_cast<ROCSHMEM_OP>(next_element.op),
             static_cast<ro_net_types>(next_element.datatype));
      DPRINTF("Received AMO dst %p src %p Val %llu pe %d\n", next_element.dst,
              next_element.src, next_element.ol1.atomic_value, next_element.PE);
      break;
    case RO_NET_AMO_FCAS:
      amoFCAS(next_element.dst, next_element.src,
              const_cast<unsigned long long *>(&next_element.ol1.atomic_value),
              next_element.PE, next_element.ro_net_win_id, queue_idx,
              next_element.status, true,
              const_cast<void **>(&next_element.ol2.pWrk),
              static_cast<ro_net_types>(next_element.datatype));
      DPRINTF("Received F_CSWAP dst %p src %p Val %llu pe %d cond %ld\n",
              next_element.dst, next_element.src, next_element.ol1.atomic_value,
              next_element.PE,
              reinterpret_cast<int64_t>(next_element.ol2.pWrk));
      break;
    case RO_NET_TEAM_REDUCE:
      team_reduction(next_element.dst, next_element.src, next_element.ol1.size,
                     next_element.ro_net_win_id, queue_idx,
                     next_element.team_comm,
                     static_cast<ROCSHMEM_OP>(next_element.op),
                     static_cast<ro_net_types>(next_element.datatype),
                     next_element.status, true);
      DPRINTF("Received FLOAT_SUM_TEAM_REDUCE dst %p src %p size %lu team %d\n",
              next_element.dst, next_element.src, next_element.ol1.size,
              next_element.team_comm);
      break;
    case RO_NET_TO_ALL:
      reduction(next_element.dst, next_element.src, next_element.ol1.size,
                next_element.PE, next_element.ro_net_win_id, queue_idx,
                next_element.PE, next_element.logPE_stride,
                next_element.PE_size, next_element.ol2.pWrk, next_element.pSync,
                static_cast<ROCSHMEM_OP>(next_element.op),
                static_cast<ro_net_types>(next_element.datatype),
                next_element.status, true);
      DPRINTF(
          "Received FLOAT_SUM_TO_ALL dst %p src %p size %lu "
          "PE_start %d, logPE_stride %d, PE_size %d, pWrk %p, pSync %p\n",
          next_element.dst, next_element.src, next_element.ol1.size,
          next_element.PE, next_element.logPE_stride, next_element.PE_size,
          next_element.ol2.pWrk, next_element.pSync);
      break;
    case RO_NET_TEAM_BROADCAST:
      team_broadcast(next_element.dst, next_element.src, next_element.ol1.size,
                     next_element.ro_net_win_id, queue_idx,
                     next_element.team_comm, next_element.PE_root,
                     static_cast<ro_net_types>(next_element.datatype),
                     next_element.status, true);
      DPRINTF(
          "Received TEAM_BROADCAST dst %p src %p size %lu "
          "team %d, PE_root %d \n",
          next_element.dst, next_element.src, next_element.ol1.size,
          next_element.team_comm, next_element.PE_root);
      break;
    case RO_NET_BROADCAST:
      broadcast(next_element.dst, next_element.src, next_element.ol1.size,
                next_element.ro_net_win_id, next_element.PE, queue_idx,
                next_element.PE, next_element.logPE_stride,
                next_element.PE_size, next_element.PE_root, next_element.pSync,
                static_cast<ro_net_types>(next_element.datatype),
                next_element.status, true);
      DPRINTF(
          "Received BROADCAST dst %p src %p size %lu PE_start %d, "
          "logPE_stride %d, PE_size %d, PE_root %d, pSync %p\n",
          next_element.dst, next_element.src, next_element.ol1.size,
          next_element.PE, next_element.logPE_stride, next_element.PE_size,
          next_element.PE_root, next_element.pSync);
      break;
    case RO_NET_ALLTOALL:
      alltoall(next_element.dst, next_element.src, next_element.ol1.size,
               next_element.ro_net_win_id, queue_idx, next_element.team_comm,
               next_element.ol2.pWrk,
               static_cast<ro_net_types>(next_element.datatype),
               next_element.status, true);
      DPRINTF("Received ALLTOALL  dst %p src %p size %lu team %d\n",
              next_element.dst, next_element.src, next_element.ol1.size,
              next_element.team_comm);
      break;
    case RO_NET_FCOLLECT:
      fcollect(next_element.dst, next_element.src, next_element.ol1.size,
               next_element.ro_net_win_id, queue_idx, next_element.team_comm,
               next_element.ol2.pWrk,
               static_cast<ro_net_types>(next_element.datatype),
               next_element.status, true);
      DPRINTF("Received FCOLLECT  dst %p src %p size %lu team %d\n",
              next_element.dst, next_element.src, next_element.ol1.size,
              next_element.team_comm);
      break;
    case RO_NET_BARRIER_ALL:
      barrier(queue_idx, next_element.status, true, ro_net_comm_world);
      DPRINTF("Received Barrier_all\n");
      break;
    case RO_NET_SYNC:
      barrier(queue_idx, next_element.status, true, next_element.team_comm);
      DPRINTF("Received Sync\n");
      break;
    case RO_NET_FENCE:
    case RO_NET_QUIET:
      quiet(queue_idx, next_element.status);
      DPRINTF("Received FENCE/QUIET\n");
      break;
    case RO_NET_FINALIZE:
      quiet(queue_idx, next_element.status);
      DPRINTF("Received Finalize\n");
      break;
    default:
      fprintf(stderr, "Invalid GPU Packet received, exiting....\n");
      abort();
      break;
  }
}

void MPITransport::initTransport(int num_queues, BackendProxyT *proxy) {
  waiting_quiet.resize(num_queues, std::vector<volatile char *>());
  outstanding.resize(num_queues, 0);
  transport_up = false;

  backend_proxy = proxy;
  auto *bp{backend_proxy->get()};

  host_interface =
      new HostInterface(bp->hdp_policy, ro_net_comm_world, bp->heap_ptr);
  progress_thread = std::thread(&MPITransport::threadProgressEngine, this);
  while (!transport_up) {
  }
}

void MPITransport::finalizeTransport() {
  progress_thread.join();
  delete host_interface;
}

rocshmem_team_t get_external_team(ROTeam *team) {
  return reinterpret_cast<rocshmem_team_t>(team);
}

void MPITransport::createNewTeam(ROBackend *backend, Team *parent_team,
                                   TeamInfo *team_info_wrt_parent,
                                   TeamInfo *team_info_wrt_world, int num_pes,
                                   int my_pe_in_new_team, MPI_Comm team_comm,
                                   rocshmem_team_t *new_team) {
  ROTeam *new_team_obj{nullptr};

  CHECK_HIP(hipMalloc(&new_team_obj, sizeof(ROTeam)));

  new (new_team_obj) ROTeam(backend, team_info_wrt_parent, team_info_wrt_world,
                            num_pes, my_pe_in_new_team, team_comm);

  *new_team = get_external_team(new_team_obj);
}

MPI_Comm MPITransport::createComm(int start, int stride, int size) {
  CommKey key(start, stride, size);
  auto it{comm_map.find(key)};
  if (it != comm_map.end()) {
    DPRINTF("Using cached communicator\n");
    return it->second;
  }

  int world_size{};
  NET_CHECK(MPI_Comm_size(ro_net_comm_world, &world_size));

  MPI_Comm comm{};
  if (start == 0 && stride == 1 && size == world_size) {
    NET_CHECK(MPI_Comm_dup(ro_net_comm_world, &comm));
  } else {
    MPI_Group world_group{};
    NET_CHECK(MPI_Comm_group(ro_net_comm_world, &world_group));

    std::vector<int> group_ranks(size);
    group_ranks[0] = start;
    for (int i{1}; i < size; i++) {
      group_ranks[i] = group_ranks[i - 1] + stride;
    }

    MPI_Group new_group{};
    NET_CHECK(MPI_Group_incl(world_group, size, group_ranks.data(), &new_group));
    NET_CHECK(MPI_Comm_create_group(ro_net_comm_world, new_group, 0, &comm));
  }

  comm_map.insert(std::pair<CommKey, MPI_Comm>(key, comm));
  DPRINTF("Creating new communicator\n");

  return comm;
}

void MPITransport::global_exit(int status) {
  MPI_Abort(ro_net_comm_world, status);
}

void MPITransport::barrier(int contextId, volatile char *status, bool blocking,
                           MPI_Comm team) {
  MPI_Request request{};
  NET_CHECK(MPI_Ibarrier(team, &request));

  requests.push_back({request, {status, contextId, blocking}});
  outstanding[contextId]++;
}

MPI_Op MPITransport::get_mpi_op(ROCSHMEM_OP op) {
  switch (op) {
    case ROCSHMEM_SUM:
      return MPI_SUM;
    case ROCSHMEM_MAX:
      return MPI_MAX;
    case ROCSHMEM_MIN:
      return MPI_MIN;
    case ROCSHMEM_PROD:
      return MPI_PROD;
    case ROCSHMEM_AND:
      return MPI_BAND;
    case ROCSHMEM_OR:
      return MPI_BOR;
    case ROCSHMEM_XOR:
      return MPI_BXOR;
    case ROCSHMEM_REPLACE:
      return MPI_REPLACE;
    default:
      fprintf(stderr, "Unknown rocSHMEM op MPI conversion %d\n", op);
      abort();
  }
}

static MPI_Datatype convertType(ro_net_types type) {
  switch (type) {
    case RO_NET_FLOAT:
      return MPI_FLOAT;
    case RO_NET_DOUBLE:
      return MPI_DOUBLE;
    case RO_NET_INT:
      return MPI_INT;
    case RO_NET_LONG:
      return MPI_LONG;
    case RO_NET_UNSIGNED_LONG:
      return MPI_UNSIGNED_LONG;
    case RO_NET_LONG_LONG:
      return MPI_LONG_LONG;
    case RO_NET_SHORT:
      return MPI_SHORT;
    case RO_NET_LONG_DOUBLE:
      return MPI_LONG_DOUBLE;
    case RO_NET_CHAR:
      return MPI_CHAR;
    case RO_NET_SIGNED_CHAR:
      return MPI_SIGNED_CHAR;
    case RO_NET_UNSIGNED_CHAR:
      return MPI_UNSIGNED_CHAR;
    default:
      fprintf(stderr, "Unknown rocSHMEM type MPI conversion %d\n", type);
      abort();
  }
}

void MPITransport::reduction(void *dst, void *src, int size, int pe,
                             int win_id, int contextId, int start,
                             int logPstride, int sizePE, void *pWrk,
                             long *pSync, ROCSHMEM_OP op, ro_net_types type,
                             volatile char *status, bool blocking) {
  MPI_Request request{};
  MPI_Op mpi_op{get_mpi_op(op)};
  MPI_Datatype mpi_type{convertType(type)};
  MPI_Comm comm{createComm(start, 1 << logPstride, sizePE)};

  if (dst == src) {
    NET_CHECK(MPI_Iallreduce(MPI_IN_PLACE, dst, size, mpi_type, mpi_op, comm,
                             &request));
  } else {
    NET_CHECK(MPI_Iallreduce(src, dst, size, mpi_type, mpi_op, comm, &request));
  }

  requests.push_back({request, {status, contextId, blocking}});
  outstanding[contextId]++;
}

void MPITransport::broadcast(void *dst, void *src, int size, int pe,
                             int win_id, int contextId, int start,
                             int logPstride, int sizePE, int root, long *pSync,
                             ro_net_types type, volatile char *status,
                             bool blocking) {
  MPI_Comm comm{createComm(start, 1 << logPstride, sizePE)};

  int new_rank{};
  MPI_Comm_rank(comm, &new_rank);

  void *data{nullptr};
  if (new_rank == root) {
    data = src;
  } else {
    data = dst;
  }

  MPI_Request request{};
  MPI_Datatype mpi_type{convertType(type)};
  NET_CHECK(MPI_Ibcast(data, size, mpi_type, root, comm, &request));

  requests.push_back({request, {status, contextId, blocking}});

  outstanding[contextId]++;
}

void MPITransport::team_reduction(void *dst, void *src, int size, int win_id,
                                  int contextId, MPI_Comm team, ROCSHMEM_OP op,
                                  ro_net_types type, volatile char* status,
                                  bool blocking) {
  MPI_Request request{};

  MPI_Op mpi_op{get_mpi_op(op)};
  MPI_Datatype mpi_type{convertType(type)};
  MPI_Comm comm{team};

  if (dst == src) {
    NET_CHECK(MPI_Iallreduce(MPI_IN_PLACE, dst, size, mpi_type, mpi_op, comm,
                             &request));
  } else {
    NET_CHECK(MPI_Iallreduce(src, dst, size, mpi_type, mpi_op, comm, &request));
  }

  requests.push_back({request, {status, contextId, blocking}});

  outstanding[contextId]++;
}

void MPITransport::team_broadcast(void *dst, void *src, int size, int win_id,
                                  int contextId, MPI_Comm team, int root,
                                  ro_net_types type, volatile char *status,
                                  bool blocking) {
  MPI_Comm comm{team};
  int new_rank{};
  MPI_Comm_rank(comm, &new_rank);
  void *data{nullptr};
  if (new_rank == root) {
    data = src;
  } else {
    data = dst;
  }

  MPI_Datatype mpi_type{convertType(type)};
  MPI_Request request{};
  NET_CHECK(MPI_Ibcast(data, size, mpi_type, root, comm, &request));

  requests.push_back({request, {status, contextId, blocking}});

  outstanding[contextId]++;
}

void MPITransport::alltoall(void *dst, void *src, int size, int win_id,
                            int contextId, MPI_Comm team, void *ata_buffptr,
                            ro_net_types type, volatile char *status,
                            bool blocking) {
  int pe_size{};
  NET_CHECK(MPI_Comm_size(team, &pe_size));

  int type_size{};
  NET_CHECK(MPI_Type_size(convertType(type), &type_size));

  int num_clust = sqrt(pe_size);
  int clust_size{(pe_size + num_clust - 1) / num_clust};

#ifdef A2A_HEURISTICS
  if ((pe_size >= 8 || type_size * size < 2048) &&
      num_clust * clust_size == pe_size) {
    return alltoall_gcen(dst, src, size, win_id, contextId, team, ata_buffptr,
                         type, status, blocking);
  } else if (size <= 512) {
#endif // A2A_HEURISTICS
    return alltoall_mpi(dst, src, size, contextId, team, ata_buffptr, type,
                        status, blocking);
#ifdef A2A_HEURISTICS
  } else {
    return alltoall_broadcast(dst, src, size, win_id, contextId, team,
                              ata_buffptr, type, status, blocking);
  }
#endif // A2A_HEURISTICS
}

void MPITransport::alltoall_broadcast(void *dst, void *src, int size,
                                      int win_id, int contextId, MPI_Comm team,
                                      void *ata_buffptr, ro_net_types type,
                                      volatile char *status, bool blocking) {
  auto *bp{backend_proxy->get()};

  MPI_Comm comm{team};
  int new_rank{};
  NET_CHECK(MPI_Comm_rank(comm, &new_rank));
  int pe_size{};
  NET_CHECK(MPI_Comm_size(comm, &pe_size));

  MPI_Group grp{};
  NET_CHECK(MPI_Comm_group(comm, &grp));
  MPI_Group world_grp{};
  NET_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_grp));

  int grp_size{};

  NET_CHECK(MPI_Group_size(grp, &grp_size));

  std::vector<int> ranks(grp_size);
  std::vector<int> world_ranks(grp_size);

  for (int i{0}; i < grp_size; i++) ranks[i] = i;

  NET_CHECK(
      MPI_Group_translate_ranks(grp, grp_size, ranks.data(), world_grp, world_ranks.data()));

  int type_size{};
  MPI_Datatype mpi_type{convertType(type)};
  NET_CHECK(MPI_Type_size(mpi_type, &type_size));
  std::vector<MPI_Request> pe_req(pe_size);

  for (int i{0}; i < pe_size; ++i) {
    int src_offset{i * type_size * size};
    int dst_offset{new_rank * type_size * size};
    NET_CHECK(MPI_Rput(reinterpret_cast<char *>(src) + src_offset, size,
                       mpi_type, world_ranks[i],
                       bp->heap_window_info[win_id]->get_offset(
                           reinterpret_cast<char *>(dst) + dst_offset),
                       size, mpi_type, bp->heap_window_info[win_id]->get_win(),
                       &pe_req[i]));
  }
  NET_CHECK(MPI_Waitall(pe_size, pe_req.data(), MPI_STATUSES_IGNORE));
  NET_CHECK(MPI_Win_flush_all(bp->heap_window_info[win_id]->get_win()));

  barrier(contextId, status, blocking, comm);
}

void MPITransport::alltoall_mpi(void *dst, void *src, int size, int contextId,
                                MPI_Comm team, void *ata_buffptr,
                                ro_net_types type, volatile char *status,
                                bool blocking) {
  int new_rank{};
  NET_CHECK(MPI_Comm_rank(team, &new_rank));
  int pe_size{};
  NET_CHECK(MPI_Comm_size(team, &pe_size));
  MPI_Datatype mpi_type{convertType(type)};
  NET_CHECK(MPI_Alltoall(src, size, mpi_type, dst, size, mpi_type, team));
  quiet(contextId, status);
}

void MPITransport::alltoall_gcen(void *dst, void *src, int size, int win_id,
                                 int contextId, MPI_Comm team,
                                 void *ata_buffptr, ro_net_types type,
                                 volatile char *status, bool blocking) {
  auto *bp{backend_proxy->get()};

  int new_rank{};
  NET_CHECK(MPI_Comm_rank(team, &new_rank));
  int pe_size{};
  NET_CHECK(MPI_Comm_size(team, &pe_size));

  MPI_Group grp{};
  NET_CHECK(MPI_Comm_group(team, &grp));
  MPI_Group world_grp{};
  NET_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_grp));

  int grp_size{};
  NET_CHECK(MPI_Group_size(grp, &grp_size));

  std::vector<int> ranks(grp_size);
  std::vector<int> world_ranks(grp_size);

  for (int i{0}; i < grp_size; i++) ranks[i] = i;
  NET_CHECK(
      MPI_Group_translate_ranks(grp, grp_size, ranks.data(), world_grp, world_ranks.data()));

  int type_size{};
  MPI_Datatype mpi_type{convertType(type)};
  NET_CHECK(MPI_Type_size(mpi_type, &type_size));

  int num_clust = sqrt(pe_size);
  int clust_size{(pe_size + num_clust - 1) / num_clust};
  assert(num_clust * clust_size == pe_size);
  int clust_id{new_rank / clust_size};

  if (MAX_ATA_BUFF_SIZE < type_size * size * pe_size) {
    fprintf(stderr, "Alltoall size %d exceeds max MAX_ATA_BUFF_SIZE %d\n",
            type_size * size * pe_size, MAX_ATA_BUFF_SIZE);
    abort();
  }

  std::vector<MPI_Request> clust_req(pe_size);

  // Step 1: Send data to PEs in cluster
  for (int i{0}; i < pe_size; ++i) {
    int src_offset{(new_rank % clust_size + (i / clust_size) * clust_size) *
                    type_size * size};
    int dst_offset{i * type_size * size};
    NET_CHECK(MPI_Rget(
        reinterpret_cast<void *>(
            (reinterpret_cast<char *>(ata_buffptr) + dst_offset)),
        size, mpi_type, world_ranks[clust_id * clust_size + (i % clust_size)],
        bp->heap_window_info[win_id]->get_offset(reinterpret_cast<char *>(src) +
                                                 src_offset),
        size, mpi_type, bp->heap_window_info[win_id]->get_win(),
        &clust_req[i]));
  }

  NET_CHECK(MPI_Waitall(pe_size, clust_req.data(), MPI_STATUSES_IGNORE));

  // Step 2: Send final data to PEs outside cluster
  for (int i{0}; i < num_clust; ++i) {
    int src_offset{i * type_size * size * clust_size};
    int dst_offset{clust_id * type_size * size * clust_size};
    NET_CHECK(MPI_Put(
        reinterpret_cast<void *>(
            (reinterpret_cast<char *>(ata_buffptr) + src_offset)),
        size * clust_size, mpi_type,
        world_ranks[(new_rank % clust_size) + i * clust_size],
        bp->heap_window_info[win_id]->get_offset(dst) + dst_offset,
        size * clust_size, mpi_type, bp->heap_window_info[win_id]->get_win()));

    // Since MPI makes puts as complete as soon as the local buffer is free,
    // we need a flush to satisfy quiet.
    NET_CHECK(
        MPI_Win_flush(world_ranks[(new_rank % clust_size) + i * clust_size],
                      bp->heap_window_info[win_id]->get_win()));
  }

  int stride{world_ranks[1] - world_ranks[0]};
  MPI_Comm comm_cluster{
      createComm(world_ranks[clust_id * clust_size], stride, clust_size)};
  MPI_Comm comm_ring{createComm(world_ranks[new_rank % clust_size],
                                stride * clust_size, num_clust)};

  barrier(contextId, status, false, comm_cluster);
  barrier(contextId, status, blocking, comm_ring);
}

void MPITransport::alltoall_gcen2(void *dst, void *src, int size, int win_id,
                                  int contextId, MPI_Comm team,
                                  void *ata_buffptr, ro_net_types type,
                                  volatile char *status, bool blocking) {
  // GPU-centric alltoall with in-place blocking synchronization
  auto *bp{backend_proxy->get()};
  int new_rank, pe_size;

  MPI_Datatype mpi_type = convertType(type);
  MPI_Comm comm = team;
  NET_CHECK(MPI_Comm_rank(comm, &new_rank));
  NET_CHECK(MPI_Comm_size(comm, &pe_size));

  MPI_Group grp, world_grp;
  NET_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_grp));
  NET_CHECK(MPI_Comm_group(comm, &grp));

  int grp_size;
  NET_CHECK(MPI_Group_size(grp, &grp_size));

  std::vector<int> ranks(grp_size);
  std::vector<int> world_ranks(grp_size);

  for (int i = 0; i < grp_size; i++) ranks[i] = i;
  // Convert comm ranks to global ranks for rput
  NET_CHECK(
      MPI_Group_translate_ranks(grp, grp_size, ranks.data(), world_grp, world_ranks.data()));

  int type_size;
  NET_CHECK(MPI_Type_size(mpi_type, &type_size));

  // Works when number of PEs divisible by root(PE_size)
  int num_clust = sqrt(pe_size);
  int clust_size = (pe_size + num_clust - 1) / num_clust;
  // TODO(bpotter) Allow any size of cluster
  assert(num_clust * clust_size == pe_size);
  int clust_id = new_rank / clust_size;

  if (MAX_ATA_BUFF_SIZE < type_size * size * pe_size) {
    fprintf(stderr, "Alltoall size %d exceeds max MAX_ATA_BUFF_SIZE %d\n",
            type_size * size * pe_size, MAX_ATA_BUFF_SIZE);
    abort();
  }

  std::vector<MPI_Request> clust_req(pe_size);

  // Step 1: Send data to PEs in cluster
  for (int i = 0; i < pe_size; ++i) {
    int src_offset = (new_rank % clust_size + (i / clust_size) * clust_size) *
                     type_size * size;
    int dst_offset = i * type_size * size;
    NET_CHECK(MPI_Rget(reinterpret_cast<void *>(
                           reinterpret_cast<char *>(ata_buffptr) + dst_offset),
                       size, mpi_type,
                       world_ranks[clust_id * clust_size + (i % clust_size)],
                       bp->heap_window_info[win_id]->get_offset(
                           reinterpret_cast<char *>(src) + src_offset),
                       size, mpi_type, bp->heap_window_info[win_id]->get_win(),
                       &clust_req[i]));
  }

  NET_CHECK(MPI_Waitall(pe_size, clust_req.data(), MPI_STATUSES_IGNORE));

  // Now wait
  int stride = world_ranks[1] - world_ranks[0];
  MPI_Comm comm_cluster =
      createComm(world_ranks[clust_id * clust_size], stride, clust_size);
  MPI_Barrier(comm_cluster);

  // Step 2: Send final data to PEs outside cluster
  for (int i = 0; i < num_clust; ++i) {
    int src_offset = i * type_size * size * clust_size;
    int dst_offset = clust_id * type_size * size * clust_size;
    NET_CHECK(MPI_Put(
        reinterpret_cast<void *>(reinterpret_cast<char *>(ata_buffptr) +
                                 src_offset),
        size * clust_size, mpi_type,
        world_ranks[(new_rank % clust_size) + i * clust_size],
        bp->heap_window_info[win_id]->get_offset(dst) + dst_offset,
        size * clust_size, mpi_type, bp->heap_window_info[win_id]->get_win()));

    // Since MPI makes puts as complete as soon as the local buffer is free,
    // we need a flush to satisfy quiet.
    NET_CHECK(
        MPI_Win_flush(world_ranks[(new_rank % clust_size) + i * clust_size],
                      bp->heap_window_info[win_id]->get_win()));
  }

  MPI_Comm comm_ring = createComm(world_ranks[new_rank % clust_size],
                                  stride * clust_size, num_clust);
  // Now wait for completion
  barrier(contextId, status, blocking, comm_ring);
}

void MPITransport::fcollect(void *dst, void *src, int size, int win_id,
                            int contextId, MPI_Comm team, void *ata_buffptr,
                            ro_net_types type, volatile char *status,
                            bool blocking) {
  int pe_size, type_size;
  MPI_Comm comm = team;
  NET_CHECK(MPI_Comm_size(comm, &pe_size));

  MPI_Datatype mpi_type = convertType(type);
  NET_CHECK(MPI_Type_size(mpi_type, &type_size));

  // Currently GPU-centric algo only supports multiples of square root
  // TODO(bpotter) Allow any size of cluster
  int num_clust = sqrt(pe_size);
  int clust_size = (pe_size + num_clust - 1) / num_clust;

  // In most cases the MPI implementation is optimal
  // But it crashes for > 512 messages
  if (size <= 512) {
    fcollect_mpi(dst, src, size, contextId, team, ata_buffptr, type,
                 status, blocking);
  } else if (num_clust * clust_size == pe_size) {
    fcollect_gcen(dst, src, size, win_id, contextId, team, ata_buffptr, type,
                  status, blocking);
  } else {
    fcollect_broadcast(dst, src, size, win_id, contextId, team, ata_buffptr,
                       type, status, blocking);
  }
}

void MPITransport::fcollect_broadcast(void *dst, void *src, int size,
                                      int win_id, int contextId, MPI_Comm team,
                                      void *ata_buffptr, ro_net_types type,
                                      volatile char *status, bool blocking) {
  // Broadcast implementation of fcollect
  auto *bp{backend_proxy->get()};
  int new_rank, pe_size;

  MPI_Datatype mpi_type = convertType(type);
  MPI_Comm comm = team;
  NET_CHECK(MPI_Comm_rank(comm, &new_rank));
  NET_CHECK(MPI_Comm_size(comm, &pe_size));

  MPI_Group grp, world_grp;
  NET_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_grp));
  NET_CHECK(MPI_Comm_group(comm, &grp));

  int grp_size;
  NET_CHECK(MPI_Group_size(grp, &grp_size));

  std::vector<int> ranks(grp_size);
  std::vector<int> world_ranks(grp_size);

  for (int i = 0; i < grp_size; i++) ranks[i] = i;
  // Convert comm ranks to global ranks for rput
  NET_CHECK(
      MPI_Group_translate_ranks(grp, grp_size, ranks.data(), world_grp, world_ranks.data()));

  int type_size;
  NET_CHECK(MPI_Type_size(mpi_type, &type_size));

  std::vector<MPI_Request> pe_req(pe_size);

  // Put data to all PEs
  for (int i = 0; i < pe_size; ++i) {
    int dst_offset = new_rank * type_size * size;
    NET_CHECK(MPI_Rput(
        reinterpret_cast<char *>(src), size, mpi_type, world_ranks[i],
        bp->heap_window_info[win_id]->get_offset(reinterpret_cast<char *>(dst) +
                                                 dst_offset),
        size, mpi_type, bp->heap_window_info[win_id]->get_win(), &pe_req[i]));
  }
  NET_CHECK(MPI_Waitall(pe_size, pe_req.data(), MPI_STATUSES_IGNORE));
  NET_CHECK(MPI_Win_flush_all(bp->heap_window_info[win_id]->get_win()));

  // Now wait for completion
  barrier(contextId, status, blocking, comm);
}

void MPITransport::fcollect_mpi(void *dst, void *src, int size, int contextId,
                                MPI_Comm team, void *ata_buffptr,
                                ro_net_types type, volatile char *status,
                                bool blocking) {
  // MPI's implementation of fcollect
  int new_rank, pe_size;

  MPI_Datatype mpi_type = convertType(type);
  MPI_Comm comm = team;
  NET_CHECK(MPI_Comm_rank(comm, &new_rank));
  NET_CHECK(MPI_Comm_size(comm, &pe_size));
  NET_CHECK(MPI_Allgather(src, size, mpi_type, dst, size, mpi_type, comm));
  quiet(contextId, status);
}

void MPITransport::fcollect_gcen(void *dst, void *src, int size, int win_id,
                                 int contextId, MPI_Comm team,
                                 void *ata_buffptr, ro_net_types type,
                                 volatile char *status, bool blocking) {
  // GPU-centric implementation of fcollect
  auto *bp{backend_proxy->get()};
  int new_rank, pe_size;

  MPI_Datatype mpi_type = convertType(type);
  MPI_Comm comm = team;
  NET_CHECK(MPI_Comm_rank(comm, &new_rank));
  NET_CHECK(MPI_Comm_size(comm, &pe_size));

  MPI_Group grp, world_grp;
  NET_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_grp));
  NET_CHECK(MPI_Comm_group(comm, &grp));

  int grp_size;
  NET_CHECK(MPI_Group_size(grp, &grp_size));

  std::vector<int> ranks(grp_size);
  std::vector<int> world_ranks(grp_size);

  for (int i = 0; i < grp_size; i++) ranks[i] = i;
  // Convert comm ranks to global ranks for rput
  NET_CHECK(
      MPI_Group_translate_ranks(grp, grp_size, ranks.data(), world_grp, world_ranks.data()));

  int type_size;
  NET_CHECK(MPI_Type_size(mpi_type, &type_size));

  // Works when number of PEs divisible by root(PE_size)
  int num_clust = sqrt(pe_size);
  int clust_size = (pe_size + num_clust - 1) / num_clust;
  // TODO(bpotter) Allow any size of cluster
  assert(num_clust * clust_size == pe_size);
  int clust_id = new_rank / clust_size;

  if (MAX_ATA_BUFF_SIZE < type_size * size * pe_size) {
    fprintf(stderr, "Fcollect size %d exceeds max MAX_ATA_BUFF_SIZE %d\n",
            type_size * size * pe_size, MAX_ATA_BUFF_SIZE);
    abort();
  }

  std::vector<MPI_Request> clust_req(pe_size);

  // Step 1: Send data to PEs in cluster
  for (int i = 0; i < clust_size; ++i) {
    int dst_offset = i * type_size * size;
    NET_CHECK(MPI_Rget(
        reinterpret_cast<void *>(reinterpret_cast<char *>(ata_buffptr) +
                                 dst_offset),
        size, mpi_type, world_ranks[clust_id * clust_size + (i % clust_size)],
        bp->heap_window_info[win_id]->get_offset(src), size, mpi_type,
        bp->heap_window_info[win_id]->get_win(), &clust_req[i]));
  }

  NET_CHECK(MPI_Waitall(clust_size, clust_req.data(), MPI_STATUSES_IGNORE));

  // Step 2: Send final data to PEs outside cluster
  for (int i = 0; i < num_clust; ++i) {
    int src_offset = i * type_size * size * clust_size;
    int dst_offset = clust_id * type_size * size * clust_size;
    NET_CHECK(MPI_Put(ata_buffptr, size * clust_size, mpi_type,
                      world_ranks[(new_rank % clust_size) + i * clust_size],
                      bp->heap_window_info[win_id]->get_offset(
                          reinterpret_cast<char *>(dst) + dst_offset),
                      size * clust_size, mpi_type,
                      bp->heap_window_info[win_id]->get_win()));

    // Since MPI makes puts as complete as soon as the local buffer is free,
    // we need a flush to satisfy quiet.
    NET_CHECK(
        MPI_Win_flush(world_ranks[(new_rank % clust_size) + i * clust_size],
                      bp->heap_window_info[win_id]->get_win()));
  }

  int stride = world_ranks[1] - world_ranks[0];
  MPI_Comm comm_cluster =
      createComm(world_ranks[clust_id * clust_size], stride, clust_size);
  MPI_Comm comm_ring = createComm(world_ranks[new_rank % clust_size],
                                  stride * clust_size, num_clust);
  // Now wait for completion
  barrier(contextId, status, false, comm_cluster);
  barrier(contextId, status, blocking, comm_ring);
}

void MPITransport::fcollect_gcen2(void *dst, void *src, int size, int win_id,
                                  int contextId, MPI_Comm team,
                                  void *ata_buffptr, ro_net_types type,
                                  volatile char *status, bool blocking) {
  // GPU-centric implementation with in-place, blocking synchronization
  auto *bp{backend_proxy->get()};
  int new_rank, pe_size;

  MPI_Datatype mpi_type = convertType(type);
  MPI_Comm comm = team;
  NET_CHECK(MPI_Comm_rank(comm, &new_rank));
  NET_CHECK(MPI_Comm_size(comm, &pe_size));

  MPI_Group grp, world_grp;
  NET_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_grp));
  NET_CHECK(MPI_Comm_group(comm, &grp));

  int grp_size;
  NET_CHECK(MPI_Group_size(grp, &grp_size));

  std::vector<int> ranks(grp_size);
  std::vector<int> world_ranks(grp_size);

  for (int i = 0; i < grp_size; i++) ranks[i] = i;
  // Convert comm ranks to global ranks for rput
  NET_CHECK(
      MPI_Group_translate_ranks(grp, grp_size, ranks.data(), world_grp, world_ranks.data()));

  int type_size;
  NET_CHECK(MPI_Type_size(mpi_type, &type_size));

  // Works when number of PEs divisible by root(PE_size)
  int num_clust = sqrt(pe_size);
  int clust_size = (pe_size + num_clust - 1) / num_clust;
  // TODO(bpotter) Allow any size of cluster
  assert(num_clust * clust_size == pe_size);
  int clust_id = new_rank / clust_size;

  if (MAX_ATA_BUFF_SIZE < type_size * size * pe_size) {
    fprintf(stderr, "Fcollect size %d exceeds max MAX_ATA_BUFF_SIZE %d\n",
            type_size * size * pe_size, MAX_ATA_BUFF_SIZE);
    abort();
  }

  std::vector<MPI_Request> clust_req(pe_size);

  // Step 1: Send data to PEs in cluster
  for (int i = 0; i < clust_size; ++i) {
    int dst_offset = i * type_size * size;
    NET_CHECK(MPI_Rget(
        reinterpret_cast<void *>(reinterpret_cast<char *>(ata_buffptr) +
                                 dst_offset),
        size, mpi_type, world_ranks[clust_id * clust_size + (i % clust_size)],
        bp->heap_window_info[win_id]->get_offset(src), size, mpi_type,
        bp->heap_window_info[win_id]->get_win(), &clust_req[i]));
  }

  NET_CHECK(MPI_Waitall(clust_size, clust_req.data(), MPI_STATUSES_IGNORE));

  int stride = world_ranks[1] - world_ranks[0];
  MPI_Comm comm_cluster =
      createComm(world_ranks[clust_id * clust_size], stride, clust_size);
  MPI_Barrier(comm_cluster);

  // Step 2: Send final data to PEs outside cluster
  for (int i = 0; i < num_clust; ++i) {
    int src_offset = i * type_size * size * clust_size;
    int dst_offset = clust_id * type_size * size * clust_size;
    NET_CHECK(MPI_Put(ata_buffptr, size * clust_size, mpi_type,
                      world_ranks[(new_rank % clust_size) + i * clust_size],
                      bp->heap_window_info[win_id]->get_offset(
                          reinterpret_cast<char *>(dst) + dst_offset),
                      size * clust_size, mpi_type,
                      bp->heap_window_info[win_id]->get_win()));

    // Since MPI makes puts as complete as soon as the local buffer is free,
    // we need a flush to satisfy quiet.
    NET_CHECK(
        MPI_Win_flush(world_ranks[(new_rank % clust_size) + i * clust_size],
                      bp->heap_window_info[win_id]->get_win()));
  }

  MPI_Comm comm_ring = createComm(world_ranks[new_rank % clust_size],
                                  stride * clust_size, num_clust);
  // Now wait for completion
  barrier(contextId, status, blocking, comm_ring);
}

void MPITransport::putMem(void *dst, void *src, int size, int pe, int win_id,
                          int contextId, volatile char *status, bool blocking,
                          bool inline_data) {
  queue->flush_hdp();

  auto *bp{backend_proxy->get()};
  MPI_Request request{};

  NET_CHECK(MPI_Rput(
      src, size, MPI_CHAR, pe, bp->heap_window_info[win_id]->get_offset(dst),
      size, MPI_CHAR, bp->heap_window_info[win_id]->get_win(), &request));

  // Since MPI makes puts as complete as soon as the local buffer is free,
  // we need a flush to satisfy quiet.  Put it here as a hack for now even
  // though it should be in the progress loop.
  NET_CHECK(MPI_Win_flush_all(bp->heap_window_info[win_id]->get_win()));

  requests.push_back({request, {status, contextId, blocking}});

  outstanding[contextId]++;
}

void MPITransport::amoFOP(void *dst, void *src, void *val, int pe, int win_id,
                          int contextId, volatile char *status, bool blocking,
                          ROCSHMEM_OP op, ro_net_types type) {
  queue->flush_hdp();

  auto *bp{backend_proxy->get()};
  MPI_Datatype mpi_type{convertType(type)};
  NET_CHECK(MPI_Fetch_and_op(reinterpret_cast<void *>(val), src, mpi_type, pe,
                             bp->heap_window_info[win_id]->get_offset(dst),
                             get_mpi_op(op),
                             bp->heap_window_info[win_id]->get_win()));

  // Since MPI makes puts as complete as soon as the local buffer is free,
  // we need a flush to satisfy quiet.  Put it here as a hack for now even
  // though it should be in the progress loop.
  NET_CHECK(MPI_Win_flush_local(pe, bp->heap_window_info[win_id]->get_win()));

  queue->notify(status);

  queue->sfence_flush_hdp();
}

void MPITransport::amoFCAS(void *dst, void *src, void *val, int pe,
                           int win_id, int contextId, volatile char *status,
                           bool blocking, void *cond, ro_net_types type) {
  queue->flush_hdp();

  auto *bp{backend_proxy->get()};
  MPI_Datatype mpi_type{convertType(type)};
  NET_CHECK(MPI_Compare_and_swap((const void *)val, (const void *)cond, src,
                                 mpi_type, pe,
                                 bp->heap_window_info[win_id]->get_offset(dst),
                                 bp->heap_window_info[win_id]->get_win()));

  // Since MPI makes puts as complete as soon as the local buffer is free,
  // we need a flush to satisfy quiet.  Put it here as a hack for now even
  // though it should be in the progress loop.
  NET_CHECK(MPI_Win_flush_local(pe, bp->heap_window_info[win_id]->get_win()));

  queue->notify(status);

  queue->sfence_flush_hdp();
}

void MPITransport::getMem(void *dst, void *src, int size, int pe, int win_id,
                          int contextId, volatile char *status,
                          bool blocking) {
  outstanding[contextId]++;

  auto *bp{backend_proxy->get()};
  MPI_Request request{};
  NET_CHECK(MPI_Rget(
      dst, size, MPI_CHAR, pe, bp->heap_window_info[win_id]->get_offset(src),
      size, MPI_CHAR, bp->heap_window_info[win_id]->get_win(), &request));

  requests.push_back({request, {status, contextId, blocking}});
}

std::unique_ptr<MPI_Request[]> MPITransport::raw_requests() {
  auto uptr_arr = std::make_unique<MPI_Request[]>(requests.size());
  for (size_t i{0}; i < requests.size(); i++) {
    uptr_arr[i] = requests[i].request;
  }
  return uptr_arr;
}

void MPITransport::progress() {
  if (requests.size() == 0) {
    const int tag{1000};
    int flag{0};
    MPI_Status status{};
    NET_CHECK(MPI_Iprobe(MPI_ANY_SOURCE, tag, ro_net_comm_world, &flag, &status));
  } else {
    DPRINTF("Testing all outstanding requests (%zu)\n", requests.size());

    int incount = (requests.size() < testsome_indices.size())
                      ? requests.size()
                      : testsome_indices.size();
    int outcount{};

    auto uptr_req_arr {raw_requests()};

    NET_CHECK(MPI_Testsome(incount, uptr_req_arr.get(), &outcount,
                           testsome_indices.data(), MPI_STATUSES_IGNORE));

    auto *bp{backend_proxy->get()};
    for (int i{0}; i < outcount; i++) {
      int index{testsome_indices[i]};
      int contextId{requests[index].properties.contextId};
      volatile char *status{requests[index].properties.status};

      if (contextId != -1) {
        outstanding[contextId]--;
        DPRINTF(
            "Finished op for contextId %d at status addr %p "
            "(%d requests outstanding)\n",
            contextId, status, outstanding[contextId]);
      }

      if (requests[index].properties.blocking) {
        if (contextId != -1) {
          queue->notify(status);
        }
        queue->sfence_flush_hdp();
      }

      if (requests[index].properties.inline_data) {
        free(requests[index].properties.src);
      }

      // If the GPU has requested a quiet, notify it of completion when
      // all outstanding requests are complete.
      if (!outstanding[contextId] && !waiting_quiet[contextId].empty()) {
        for (const auto status : waiting_quiet[contextId]) {
          DPRINTF("Finished Quiet for contextId %d at status addr %p\n", contextId,
                  status);
          queue->notify(status);
        }

        waiting_quiet[contextId].clear();

        queue->sfence_flush_hdp();
      }
    }

    sort(testsome_indices.data(), testsome_indices.data() + outcount,
         std::greater<int>());
    for (int i{0}; i < outcount; i++) {
      int index{testsome_indices[i]};
      requests.erase(requests.begin() + index);
    }
  }
}

void MPITransport::quiet(int contextId, volatile char *status) {
  auto *bp{backend_proxy->get()};

  if (!outstanding[contextId]) {
    DPRINTF("Finished Quiet immediately for contextId %d at status addr %p\n",
            contextId, status);
    queue->notify(status);
  } else {
    waiting_quiet[contextId].emplace_back(status);
  }
}

int MPITransport::numOutstandingRequests() { return requests.size() + q.size(); }

}  // namespace rocshmem
