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
        device_state_t(int device, int nccl_rank, int nworld, const ncclUniqueId& unique_id)
            : device_id(device)
        {
            if(nccl_rank < 0 || nccl_rank >= nworld)
                throw std::out_of_range("device_state_t constructor: rank is out of range");
            rocfft_scoped_device dev(device_id);
            stream.alloc();
            auto nccl_ret = ncclCommInitRank(&comm, nworld, unique_id, nccl_rank);
            if(nccl_ret != ncclSuccess)
            {
                throw rocfft_rccl_exception_t(
                    "ncclCommInitRank failed in device_state_t constructor", nccl_ret);
            }
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
                throw std::runtime_error("device_state_t::get_comm: comm is null");
            return comm;
        }
        hipStream_t get_stream() const
        {
            if(!stream)
                throw std::runtime_error("device_state_t::get_stream: stream is null");
            return stream;
        }

        int device_id = -1;

    private:
        hipStream_wrapper_t stream;
        ncclComm_t          comm{};
    };

    std::vector<rocfft_location_t>   world_locations;
    std::map<rocfft_location_t, int> loc_to_rank;
    std::map<int, device_state_t>    device_to_state;
    int                              local_comm_rank = 0;
    ncclUniqueId                     uniqueId{};
};

std::map<std::set<rocfft_location_t>, rocfft_rccl_comm_t> rocfft_rccl_comm_t::comm_cache;
std::mutex                                                rocfft_rccl_comm_t::comm_cache_mutex;

rocfft_rccl_comm_t rocfft_rccl_comm_t::create(const std::set<int>& devices)
{
    std::set<rocfft_location_t> world;
    for(int d : devices)
        world.emplace(0, d);
#ifdef ROCFFT_MPI_ENABLE
    return create_from_world(world, 0, MPI_COMM_NULL);
#else
    return create_from_world(world, 0);
#endif
}

#ifdef ROCFFT_MPI_ENABLE
rocfft_rccl_comm_t rocfft_rccl_comm_t::create(MPI_Comm                           mpi_comm,
                                              int                                local_comm_rank,
                                              const std::set<rocfft_location_t>& world)
{
    return create_from_world(world, local_comm_rank, mpi_comm);
}
#endif

rocfft_rccl_comm_t rocfft_rccl_comm_t::create_from_world(const std::set<rocfft_location_t>& world,
                                                         int local_comm_rank
#ifdef ROCFFT_MPI_ENABLE
                                                         ,
                                                         MPI_Comm mpi_comm
#endif
)
{
    if(world.size() < 2)
        throw std::invalid_argument("rocfft_rccl_comm_t::create: need at least 2 world locations");

    std::lock_guard<std::mutex> lock(comm_cache_mutex);

    auto it = comm_cache.find(world);
    if(it != comm_cache.end())
        return it->second;

    rocfft_rccl_comm_t new_comm;
    new_comm.pimpl                  = std::make_shared<Impl>();
    new_comm.pimpl->local_comm_rank = local_comm_rank;
    new_comm.pimpl->world_locations.assign(world.begin(), world.end());
    for(size_t r = 0; r < new_comm.pimpl->world_locations.size(); ++r)
        new_comm.pimpl->loc_to_rank[new_comm.pimpl->world_locations[r]] = static_cast<int>(r);

#ifdef ROCFFT_MPI_ENABLE
    const bool multi_process = (mpi_comm != MPI_COMM_NULL);
    if(multi_process)
    {
        int mpi_rank = 0;
        int mpi_size = 0;
        if(MPI_Comm_rank(mpi_comm, &mpi_rank) != MPI_SUCCESS
           || MPI_Comm_size(mpi_comm, &mpi_size) != MPI_SUCCESS)
            throw std::runtime_error("rocfft_rccl_comm_t::create: MPI_Comm_rank/size failed");
        if(mpi_rank != local_comm_rank)
            throw std::invalid_argument("rocfft_rccl_comm_t::create: local_comm_rank ("
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
            throw std::runtime_error("rocfft_rccl_comm_t::create: MPI_Bcast of id_ok failed");
        if(!id_ok)
            throw std::runtime_error("ncclGetUniqueId failed in rocfft_rccl_comm_t::create");
        if(MPI_Bcast(&new_comm.pimpl->uniqueId,
                     static_cast<int>(sizeof(ncclUniqueId)),
                     MPI_BYTE,
                     0,
                     mpi_comm)
           != MPI_SUCCESS)
            throw std::runtime_error(
                "rocfft_rccl_comm_t::create: MPI_Bcast of ncclUniqueId failed");
    }
    else
#endif
    {
        auto result = ncclGetUniqueId(&new_comm.pimpl->uniqueId);
        if(result != ncclSuccess)
            throw rocfft_rccl_exception_t("ncclGetUniqueId failed in rocfft_rccl_comm_t::create",
                                          result);
    }

    const int           nworld = static_cast<int>(new_comm.pimpl->world_locations.size());
    rocfft_rccl_group_t group;
    for(size_t r = 0; r < new_comm.pimpl->world_locations.size(); ++r)
    {
        const auto& loc = new_comm.pimpl->world_locations[r];
        if(loc.comm_rank != local_comm_rank)
            continue;
        new_comm.pimpl->device_to_state.try_emplace(
            loc.device, loc.device, static_cast<int>(r), nworld, new_comm.pimpl->uniqueId);
    }
    group.end();

    if(new_comm.pimpl->device_to_state.empty())
        throw std::runtime_error("rocfft_rccl_comm_t::create: no local locations to initialize");

    comm_cache[world] = new_comm;
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
        throw std::invalid_argument("rocfft_rccl_comm_t::get_comm: device_id "
                                    + std::to_string(device_id)
                                    + " is not a local participant of this communicator");
    return it->second.get_comm();
}

hipStream_t rocfft_rccl_comm_t::get_stream(int device_id) const
{
    auto it = pimpl->device_to_state.find(device_id);
    if(it == pimpl->device_to_state.end())
        throw std::invalid_argument("rocfft_rccl_comm_t::get_stream: device_id "
                                    + std::to_string(device_id)
                                    + " is not a local participant of this communicator");
    return it->second.get_stream();
}

size_t rocfft_rccl_comm_t::num_ranks() const
{
    return pimpl->world_locations.size();
}

int rocfft_rccl_comm_t::get_rank(const rocfft_location_t& location) const
{
    auto it = pimpl->loc_to_rank.find(location);
    if(it == pimpl->loc_to_rank.end())
        throw std::invalid_argument("rocfft_rccl_comm_t::get_rank: location " + location.str()
                                    + " is not in this communicator");
    return it->second;
}

int rocfft_rccl_comm_t::get_rank(int device_id) const
{
    return get_rank(rocfft_location_t{pimpl->local_comm_rank, device_id});
}

std::vector<rocfft_location_t> rocfft_rccl_comm_t::get_locations() const
{
    return pimpl->world_locations;
}

std::vector<rocfft_location_t> rocfft_rccl_comm_t::get_local_locations() const
{
    std::vector<rocfft_location_t> local;
    for(const auto& loc : pimpl->world_locations)
    {
        if(loc.comm_rank == pimpl->local_comm_rank)
            local.push_back(loc);
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
            "rocfft_rccl_comm_t::alltoall: sendbufs/recvbufs must each have size "
            "num_ranks() ("
            + std::to_string(nranks) + "); got sendbufs=" + std::to_string(sendbufs.size())
            + ", recvbufs=" + std::to_string(recvbufs.size()));

    const auto nccl_count = count * (array_type_is_interleaved(array_type) ? 2 : 1);
    const auto dtype      = get_nccl_dtype(precision);

    rocfft_rccl_group_t group;
    for(const auto& loc : get_local_locations())
    {
        const int r = get_rank(loc);
        if(!sendbufs[r] || !recvbufs[r])
            throw std::invalid_argument("rocfft_rccl_comm_t::alltoall: local rank "
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
         rocfft_location_t{pimpl->local_comm_rank, peer_device_id},
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
         rocfft_location_t{pimpl->local_comm_rank, peer_device_id},
         device_id,
         precision,
         array_type);
}

#endif // ROCFFT_RCCL_ENABLE
