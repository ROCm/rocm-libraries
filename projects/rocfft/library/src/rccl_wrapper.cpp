// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifdef ROCFFT_RCCL_ENABLE

#include "rccl_wrapper.h"
#include "../../shared/array_predicate.h"
#include "../../shared/hip_object_wrapper.h"
#include "../../shared/precision_type.h"
#include "../../shared/rocfft_hip.h"
#include "logging.h"
#include "rocfft_current_function.h"
#include <algorithm>
#include <exception>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

static ncclDataType_t get_nccl_dtype(rocfft_precision precision)
{
    switch(real_type_size(precision))
    {
    case 2:
        return ncclFloat16;
    case 4:
        return ncclFloat32;
    case 8:
        return ncclFloat64;
    default:
        throw std::runtime_error("unsupported rocfft_precision in RCCL datatype mapping");
    }
}

struct rocfft_rccl_comm_t::Impl
{
    struct device_state_t
    {
        // local resources only; ncclCommInitRank is deferred to init() so
        // local failures can be agreed on before any rank blocks in it
        explicit device_state_t(int device)
            : device_id(device)
        {
            rocfft_scoped_device dev(device_id);
            stream.alloc();
        }

        void init(int nccl_rank, int num_nccl_ranks, const ncclUniqueId& unique_id)
        {
            if(nccl_rank < 0 || nccl_rank >= num_nccl_ranks)
                throw std::out_of_range(ROCFFT_CURRENT_FUNCTION + ": rank is out of range");
            rocfft_scoped_device dev(device_id);
            auto nccl_ret = ncclCommInitRank(&comm, num_nccl_ranks, unique_id, nccl_rank);
            if(nccl_ret != ncclSuccess)
                throw rocfft_rccl_exception_t(
                    "ncclCommInitRank failed in " + ROCFFT_CURRENT_FUNCTION, nccl_ret);
        }

        ~device_state_t()
        {
            try
            {
                if(comm)
                {
                    rocfft_scoped_device dev(device_id);
                    auto                 nccl_ret = ncclCommFinalize(comm);
                    if(nccl_ret != ncclSuccess)
                        throw rocfft_rccl_exception_t("ncclCommFinalize failed in destructor",
                                                      nccl_ret);
                    nccl_ret = ncclCommDestroy(comm);
                    if(nccl_ret != ncclSuccess)
                        throw rocfft_rccl_exception_t("ncclCommDestroy failed in destructor",
                                                      nccl_ret);
                }
            }
            catch(const std::exception& e)
            {
                log_trace(__func__, "Failure in device_state_t destructor", e.what());
            }
            catch(...)
            {
                log_trace(__func__,
                          "Failure in device_state_t destructor with unexpected exception");
            }
        }

        ncclComm_t get_comm() const
        {
            if(!comm)
                throw std::runtime_error(ROCFFT_CURRENT_FUNCTION + ": comm is null");
            return comm;
        }
        hipStream_t get_stream() const
        {
            if(!stream)
                throw std::runtime_error(ROCFFT_CURRENT_FUNCTION + ": stream is null");
            return stream;
        }

        int device_id = -1;

    private:
        hipStream_wrapper_t stream;
        ncclComm_t          comm{};
    };

    std::vector<rocfft_location_t>   nccl_rank_to_world_loc;
    std::map<rocfft_location_t, int> world_loc_to_nccl_rank;
    std::map<int, device_state_t>    device_to_state;
    int                              local_world_rank = 0;
    ncclUniqueId                     uniqueId{};
};

std::map<std::set<rocfft_location_t>, rocfft_rccl_comm_t> rocfft_rccl_comm_t::comm_cache;
std::mutex                                                rocfft_rccl_comm_t::comm_cache_mutex;

rocfft_rccl_comm_t rocfft_rccl_comm_t::create(const std::set<int>& devices)
{
    std::set<rocfft_location_t> locs;
    for(int d : devices)
        locs.emplace(0, d);
#ifdef ROCFFT_MPI_ENABLE
    return create_from_locs(locs, 0, MPI_COMM_NULL);
#else
    return create_from_locs(locs, 0);
#endif
}

#ifdef ROCFFT_MPI_ENABLE
rocfft_rccl_comm_t rocfft_rccl_comm_t::create(MPI_Comm                           mpi_comm,
                                              int                                local_comm_rank,
                                              const std::set<rocfft_location_t>& comm_locations)
{
    return create_from_locs(comm_locations, local_comm_rank, mpi_comm);
}
#endif

rocfft_location_t rocfft_rccl_comm_t::to_world(const rocfft_location_t& comm_location) const
{
    if(world_ranks)
        return {world_ranks->at(static_cast<size_t>(comm_location.comm_rank)),
                comm_location.device};
    return comm_location;
}

rocfft_location_t rocfft_rccl_comm_t::to_comm(const rocfft_location_t& world_location) const
{
    if(world_ranks)
    {
        auto it = std::find(world_ranks->begin(), world_ranks->end(), world_location.comm_rank);
        if(it == world_ranks->end())
            throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION + ": MPI_COMM_WORLD rank "
                                        + std::to_string(world_location.comm_rank)
                                        + " is not in the plan communicator");
        return {static_cast<int>(it - world_ranks->begin()), world_location.device};
    }
    return world_location;
}

rocfft_rccl_comm_t rocfft_rccl_comm_t::create_from_locs(const std::set<rocfft_location_t>& locs,
                                                        int local_comm_rank
#ifdef ROCFFT_MPI_ENABLE
                                                        ,
                                                        MPI_Comm mpi_comm
#endif
)
{
    if(locs.size() < 2)
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION + ": need at least 2 locations");

    // the cache and NCCL ranks use MPI_COMM_WORLD ranks, so communicators
    // over different processes never share an RCCL comm
    rocfft_rccl_comm_t new_comm;
#ifdef ROCFFT_MPI_ENABLE
    if(mpi_comm != MPI_COMM_NULL)
    {
        int world_rank = -1;
        int comm_size  = 0;
        if(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank) != MPI_SUCCESS
           || MPI_Comm_size(mpi_comm, &comm_size) != MPI_SUCCESS)
            throw std::runtime_error(ROCFFT_CURRENT_FUNCTION + ": MPI_Comm_rank/size failed");
        auto& world_ranks = new_comm.world_ranks.emplace(static_cast<size_t>(comm_size));
        if(MPI_Allgather(&world_rank, 1, MPI_INT, world_ranks.data(), 1, MPI_INT, mpi_comm)
           != MPI_SUCCESS)
            throw std::runtime_error(ROCFFT_CURRENT_FUNCTION + ": MPI_Allgather failed");
        // identical on all ranks, so every rank throws or none does
        if(std::set<int>(world_ranks.begin(), world_ranks.end()).size() != world_ranks.size())
            throw std::runtime_error(
                ROCFFT_CURRENT_FUNCTION
                + ": plan communicator has processes sharing an MPI_COMM_WORLD rank (e.g. "
                  "from MPI_Comm_spawn), so they cannot be told apart");
    }
#endif
    // new_comm.world_ranks is now set (or nullopt for single-process), so to_world is usable
    std::set<rocfft_location_t> world_locs;
    for(const auto& loc : locs)
        world_locs.insert(new_comm.to_world(loc));

    std::lock_guard<std::mutex> lock(comm_cache_mutex);

    // reuse only the shared state; world_ranks belongs to this plan's communicator
    auto it = comm_cache.find(world_locs);
    if(it != comm_cache.end())
    {
        new_comm.pimpl = it->second.pimpl;
        return new_comm;
    }

    new_comm.pimpl                   = std::make_shared<Impl>();
    new_comm.pimpl->local_world_rank = new_comm.to_world({local_comm_rank, 0}).comm_rank;
    new_comm.pimpl->nccl_rank_to_world_loc.assign(world_locs.begin(), world_locs.end());
    for(size_t r = 0; r < new_comm.pimpl->nccl_rank_to_world_loc.size(); ++r)
        new_comm.pimpl->world_loc_to_nccl_rank[new_comm.pimpl->nccl_rank_to_world_loc[r]]
            = static_cast<int>(r);

#ifdef ROCFFT_MPI_ENABLE
    const bool multi_process = (mpi_comm != MPI_COMM_NULL);
    if(multi_process)
    {
        int mpi_rank = 0;
        int mpi_size = 0;
        if(MPI_Comm_rank(mpi_comm, &mpi_rank) != MPI_SUCCESS
           || MPI_Comm_size(mpi_comm, &mpi_size) != MPI_SUCCESS)
            throw std::runtime_error(ROCFFT_CURRENT_FUNCTION + ": MPI_Comm_rank/size failed");
        if(mpi_rank != local_comm_rank)
            throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION + ": local_comm_rank ("
                                        + std::to_string(local_comm_rank) + ") != MPI rank ("
                                        + std::to_string(mpi_rank) + ")");

        int id_ok = 1;
        if(mpi_rank == 0)
        {
            auto result = ncclGetUniqueId(&new_comm.pimpl->uniqueId);
            if(result != ncclSuccess)
                id_ok = 0;
        }
        if(MPI_Bcast(&id_ok, 1, MPI_INT, 0, mpi_comm) != MPI_SUCCESS)
            throw std::runtime_error(ROCFFT_CURRENT_FUNCTION + ": MPI_Bcast of id_ok failed");
        if(!id_ok)
            throw std::runtime_error("ncclGetUniqueId failed in " + ROCFFT_CURRENT_FUNCTION);
        if(MPI_Bcast(&new_comm.pimpl->uniqueId,
                     static_cast<int>(sizeof(ncclUniqueId)),
                     MPI_BYTE,
                     0,
                     mpi_comm)
           != MPI_SUCCESS)
            throw std::runtime_error(ROCFFT_CURRENT_FUNCTION
                                     + ": MPI_Bcast of ncclUniqueId failed");
    }
    else
#endif
    {
        auto result = ncclGetUniqueId(&new_comm.pimpl->uniqueId);
        if(result != ncclSuccess)
            throw rocfft_rccl_exception_t("ncclGetUniqueId failed in " + ROCFFT_CURRENT_FUNCTION,
                                          result);
    }

    // runs a local step; if it fails on any rank, every rank throws
    const auto current_function = ROCFFT_CURRENT_FUNCTION;
    auto       global_check     = [&](const auto& local_step) {
        std::exception_ptr local_error;
        try
        {
            local_step();
        }
        catch(...)
        {
            local_error = std::current_exception();
        }
        int ok = local_error ? 0 : 1;
#ifdef ROCFFT_MPI_ENABLE
        if(multi_process
           && MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, mpi_comm) != MPI_SUCCESS)
            throw std::runtime_error(current_function + ": MPI_Allreduce failed");
#endif
        if(local_error)
            std::rethrow_exception(local_error);
        if(!ok)
            throw std::runtime_error(current_function + ": failed on another rank");
    };

    // create device states for local devices
    global_check([&] {
        for(const auto& loc : new_comm.pimpl->nccl_rank_to_world_loc)
        {
            if(loc.comm_rank == new_comm.pimpl->local_world_rank)
                new_comm.pimpl->device_to_state.try_emplace(loc.device, loc.device);
        }
        if(new_comm.pimpl->device_to_state.empty())
            throw std::runtime_error(current_function + ": no local locations to initialize");
    });

    const int num_nccl_ranks = static_cast<int>(new_comm.pimpl->nccl_rank_to_world_loc.size());
    // initialize nccl comm(s)
    global_check([&] {
        rocfft_rccl_group_t group;
        for(size_t r = 0; r < new_comm.pimpl->nccl_rank_to_world_loc.size(); ++r)
        {
            const auto& loc = new_comm.pimpl->nccl_rank_to_world_loc[r];
            if(loc.comm_rank != new_comm.pimpl->local_world_rank)
                continue;
            new_comm.pimpl->device_to_state.at(loc.device)
                .init(static_cast<int>(r), num_nccl_ranks, new_comm.pimpl->uniqueId);
        }
        group.end();
    });

    comm_cache[world_locs] = new_comm;
    return new_comm;
}

void rocfft_rccl_comm_t::reset_all()
{
    std::lock_guard<std::mutex> lock(comm_cache_mutex);
    comm_cache.clear();
}

ncclComm_t rocfft_rccl_comm_t::get_comm(int device_id) const
{
    auto it = pimpl->device_to_state.find(device_id);
    if(it == pimpl->device_to_state.end())
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION + ": device_id "
                                    + std::to_string(device_id)
                                    + " is not a local participant of this communicator");
    return it->second.get_comm();
}

hipStream_t rocfft_rccl_comm_t::get_stream(int device_id) const
{
    auto it = pimpl->device_to_state.find(device_id);
    if(it == pimpl->device_to_state.end())
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION + ": device_id "
                                    + std::to_string(device_id)
                                    + " is not a local participant of this communicator");
    return it->second.get_stream();
}

size_t rocfft_rccl_comm_t::num_ranks() const
{
    return pimpl->nccl_rank_to_world_loc.size();
}

int rocfft_rccl_comm_t::get_rank(const rocfft_location_t& location) const
{
    auto it = pimpl->world_loc_to_nccl_rank.find(to_world(location));
    if(it == pimpl->world_loc_to_nccl_rank.end())
        throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION + ": location " + location.str()
                                    + " is not in this communicator");
    return it->second;
}

int rocfft_rccl_comm_t::get_rank(int device_id) const
{
    return get_rank(to_comm(rocfft_location_t{pimpl->local_world_rank, device_id}));
}

std::vector<rocfft_location_t> rocfft_rccl_comm_t::get_locations() const
{
    std::vector<rocfft_location_t> locations;
    for(const auto& loc : pimpl->nccl_rank_to_world_loc)
        locations.push_back(to_comm(loc));
    return locations;
}

std::vector<rocfft_location_t> rocfft_rccl_comm_t::get_local_locations() const
{
    std::vector<rocfft_location_t> local;
    for(const auto& loc : pimpl->nccl_rank_to_world_loc)
    {
        if(loc.comm_rank == pimpl->local_world_rank)
            local.push_back(to_comm(loc));
    }
    return local;
}

std::vector<int> rocfft_rccl_comm_t::get_devices() const
{
    std::vector<int> devices;
    for(const auto& loc : get_local_locations())
        devices.push_back(loc.device);
    return devices;
}

rocfft_rccl_group_t::rocfft_rccl_group_t()
{
    ncclResult_t result = ncclGroupStart();
    if(result != ncclSuccess)
        throw rocfft_rccl_exception_t("ncclGroupStart failed", result);
    needs_ending = true;
}

void rocfft_rccl_group_t::end()
{
    if(!needs_ending)
        return;

    ncclResult_t result = ncclGroupEnd();
    needs_ending        = false;
    if(result != ncclSuccess)
        throw rocfft_rccl_exception_t("ncclGroupEnd failed", result);
}

rocfft_rccl_group_t::~rocfft_rccl_group_t() noexcept
{
    try
    {
        end();
    }
    catch(const rocfft_rccl_exception_t& e)
    {
        log_trace(__func__, "ncclGroupEnd failed in destructor", e.what());
    }
    catch(...)
    {
        log_trace(__func__, "ncclGroupEnd failed in destructor with unexpected exception");
    }
}

void rocfft_rccl_comm_t::alltoall(const std::vector<const void*>& sendbufs,
                                  const std::vector<void*>&       recvbufs,
                                  size_t                          count,
                                  rocfft_precision                precision,
                                  rocfft_array_type               array_type) const
{
    const auto nranks = num_ranks();
    if(sendbufs.size() != nranks || recvbufs.size() != nranks)
        throw std::invalid_argument(
            ROCFFT_CURRENT_FUNCTION + ": sendbufs/recvbufs must each have size num_ranks() ("
            + std::to_string(nranks) + "); got sendbufs=" + std::to_string(sendbufs.size())
            + ", recvbufs=" + std::to_string(recvbufs.size()));

    const auto nccl_count = count * (array_type_is_interleaved(array_type) ? 2 : 1);
    const auto dtype      = get_nccl_dtype(precision);

    rocfft_rccl_group_t group;
    for(const auto& loc : get_local_locations())
    {
        const int r = get_rank(loc);
        if(!sendbufs[r] || !recvbufs[r])
            throw std::invalid_argument(ROCFFT_CURRENT_FUNCTION + ": local rank "
                                        + std::to_string(r) + " has a null send or recv buffer");

        rocfft_scoped_device dev(loc.device);
        ncclResult_t         result = ncclAllToAll(sendbufs[r],
                                           recvbufs[r],
                                           nccl_count,
                                           dtype,
                                           get_comm(loc.device),
                                           get_stream(loc.device));
        if(result != ncclSuccess)
        {
            throw rocfft_rccl_exception_t(
                "ncclAllToAll failed on device " + std::to_string(loc.device), result);
        }
    }
    group.end();
}

void rocfft_rccl_comm_t::send(const void*              sendbuf,
                              size_t                   count,
                              const rocfft_location_t& peer,
                              int                      device_id,
                              rocfft_precision         precision,
                              rocfft_array_type        array_type) const
{
    const int    peer_rank = get_rank(peer);
    ncclResult_t result    = ncclSend(sendbuf,
                                   count * (array_type_is_interleaved(array_type) ? 2 : 1),
                                   get_nccl_dtype(precision),
                                   peer_rank,
                                   get_comm(device_id),
                                   get_stream(device_id));
    if(result != ncclSuccess)
    {
        throw rocfft_rccl_exception_t(
            "ncclSend failed on device " + std::to_string(device_id) + " to " + peer.str(), result);
    }
}

void rocfft_rccl_comm_t::send(const void*       sendbuf,
                              size_t            count,
                              int               peer_device_id,
                              int               device_id,
                              rocfft_precision  precision,
                              rocfft_array_type array_type) const
{
    send(sendbuf,
         count,
         to_comm(rocfft_location_t{pimpl->local_world_rank, peer_device_id}),
         device_id,
         precision,
         array_type);
}

void rocfft_rccl_comm_t::recv(void*                    recvbuf,
                              size_t                   count,
                              const rocfft_location_t& peer,
                              int                      device_id,
                              rocfft_precision         precision,
                              rocfft_array_type        array_type) const
{
    const int    peer_rank = get_rank(peer);
    ncclResult_t result    = ncclRecv(recvbuf,
                                   count * (array_type_is_interleaved(array_type) ? 2 : 1),
                                   get_nccl_dtype(precision),
                                   peer_rank,
                                   get_comm(device_id),
                                   get_stream(device_id));
    if(result != ncclSuccess)
    {
        throw rocfft_rccl_exception_t("ncclRecv failed on device " + std::to_string(device_id)
                                          + " from " + peer.str(),
                                      result);
    }
}

void rocfft_rccl_comm_t::recv(void*             recvbuf,
                              size_t            count,
                              int               peer_device_id,
                              int               device_id,
                              rocfft_precision  precision,
                              rocfft_array_type array_type) const
{
    recv(recvbuf,
         count,
         to_comm(rocfft_location_t{pimpl->local_world_rank, peer_device_id}),
         device_id,
         precision,
         array_type);
}

#endif // ROCFFT_RCCL_ENABLE
