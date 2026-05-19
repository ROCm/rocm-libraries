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

// entire translation unit is only compiled when RCCL support is enabled.
// see rccl_wrapper.h for the matching header-level guard.
#ifdef ROCFFT_RCCL_ENABLE

#include "rccl_wrapper.h"
#include "logging.h"
#include <map>
#include <mutex>
#include <stdexcept>

// process-wide cache of communicators keyed by device set.
// different plans may use different GPU subsets, so each distinct
// set gets its own RCCL communicator.
static std::map<std::set<int>, rocfft_rccl_comm_t> comm_cache;
static std::mutex                                  comm_cache_mutex;

struct NcclTypeInfo
{
    ncclDataType_t dtype;
    size_t         count_multiplier; // 1 for real, 2 for complex
};

// map (base_type_size, is_complex) to the NCCL datatype and a count
// multiplier.  base_type_size is the size of one real component
// (2 for half, 4 for float, 8 for double).  complex data is
// transferred as 2x the real component type.
static NcclTypeInfo get_nccl_type_info(size_t base_type_size, bool is_complex)
{
    ncclDataType_t dtype;
    switch(base_type_size)
    {
    case 2:
        dtype = ncclFloat16;
        break;
    case 4:
        dtype = ncclFloat32;
        break;
    case 8:
        dtype = ncclFloat64;
        break;
    default:
        // rocFFT only produces half (2), float (4), or double (8).
        // any other size indicates a bug in the caller.  there is
        // no safe fallback since the count_multiplier assumes a
        // floating-point element width.
        throw std::runtime_error("unexpected base_type_size " + std::to_string(base_type_size)
                                 + " in RCCL datatype mapping (expected 2, 4, or 8)");
    }
    return {dtype, is_complex ? size_t{2} : size_t{1}};
}

// implementation details shared by all copies of a handle via shared_ptr
struct rocfft_rccl_comm_t::Impl
{
    // each comm is for one rank, and each rank has exactly one
    // device.  comms is indexed by rank.
    std::vector<ncclComm_t> comms;

    // unique id used to bootstrap this communicator group.
    // stored so it can be broadcast via MPI for multi-node in the future.
    ncclUniqueId uniqueId{};

    // device_id -> NCCL rank mapping
    std::map<int, int> device_to_rank;
    int                nranks = 0;

    // runs exactly once — when the last shared_ptr<Impl> referring to
    // this object is released.  copies of rocfft_rccl_comm_t only bump
    // the refcount and never duplicate these ncclComm_t handles.
    ~Impl()
    {
        for(auto comm : comms)
        {
            if(comm)
            {
                ncclCommFinalize(comm);
                ncclCommDestroy(comm);
            }
        }
    }
};

rocfft_rccl_comm_t rocfft_rccl_comm_t::create(const std::set<int>& devices)
{
    // check if RCCL is disabled via environment variable
    const char* disable_rccl = std::getenv("ROCFFT_DISABLE_RCCL");
    if(disable_rccl && std::string(disable_rccl) == "1")
    {
        return {};
    }

    // need at least 2 devices for a meaningful communicator
    if(devices.size() < 2)
    {
        return {};
    }

    // look up or create a communicator for this exact device set.
    // guard with a mutex so concurrent plan creation from
    // multiple threads does not race on the cache.
    std::lock_guard<std::mutex> lock(comm_cache_mutex);

    auto it = comm_cache.find(devices);
    if(it != comm_cache.end())
    {
        return it->second;
    }

    {
        const int ndevices = static_cast<int>(devices.size());

        rocfft_rccl_comm_t new_comm;
        new_comm.pimpl = std::make_shared<Impl>();

        // generate unique id for this communicator group.
        // for single-node this stays local; for multi-node the root
        // rank would broadcast this via MPI_Bcast.
        ncclUniqueId id;
        ncclResult_t result = ncclGetUniqueId(&id);
        if(result != ncclSuccess)
        {
            return {};
        }
        new_comm.pimpl->uniqueId = id;
        new_comm.pimpl->nranks   = ndevices;

        // save and restore the caller's active device
        int original_device = 0;
        if(hipGetDevice(&original_device) != hipSuccess)
            throw std::runtime_error("hipGetDevice failed during RCCL communicator init");

        // comms is indexed by rank (not device_id); device_to_rank
        // maps device_id -> rank for lookup in get_comm().
        new_comm.pimpl->comms.resize(ndevices, nullptr);

        // init one communicator per device using ncclCommInitRank,
        // batched inside a group call for single-process efficiency.
        // use try/catch to guarantee ncclGroupEnd is called even if
        // hipSetDevice throws between ncclGroupStart and ncclGroupEnd.
        // ranks are assigned in sorted device-id order (std::set).
        ncclGroupStart();
        try
        {
            int rank = 0;
            for(int dev : devices)
            {
                if(hipSetDevice(dev) != hipSuccess)
                    throw std::runtime_error("hipSetDevice failed for device "
                                             + std::to_string(dev));
                new_comm.pimpl->device_to_rank[dev] = rank;
                result = ncclCommInitRank(&new_comm.pimpl->comms[rank], ndevices, id, rank);
                if(result != ncclSuccess)
                {
                    ncclGroupEnd();
                    if(hipSetDevice(original_device) != hipSuccess)
                        throw std::runtime_error("hipSetDevice failed restoring device "
                                                 + std::to_string(original_device));
                    return {};
                }
                ++rank;
            }
        }
        catch(...)
        {
            ncclGroupEnd();
            throw;
        }
        result = ncclGroupEnd();

        if(hipSetDevice(original_device) != hipSuccess)
            throw std::runtime_error("hipSetDevice failed restoring device "
                                     + std::to_string(original_device));

        if(result != ncclSuccess)
        {
            return {};
        }
        comm_cache[devices] = std::move(new_comm);
    }

    return comm_cache[devices];
}

void rocfft_rccl_comm_t::reset_all()
{
    std::lock_guard<std::mutex> lock(comm_cache_mutex);
    comm_cache.clear();
}

void* rocfft_rccl_comm_t::get_comm(int device_id) const
{
    // look up the NCCL rank for this device and index into
    // comms by rank (not device_id) so this stays correct
    // even if device ids are non-contiguous or reordered.
    auto it = pimpl->device_to_rank.find(device_id);
    if(it == pimpl->device_to_rank.end())
        return nullptr;
    return &pimpl->comms[it->second];
}

int rocfft_rccl_comm_t::num_ranks() const
{
    return pimpl->nranks;
}

int rocfft_rccl_comm_t::get_rank(int device_id) const
{
    auto it = pimpl->device_to_rank.find(device_id);
    if(it == pimpl->device_to_rank.end())
        return -1;
    return it->second;
}

// RAII group wrapper
rocfft_rccl_group_t::rocfft_rccl_group_t()
{
    ncclGroupStart();
}

rocfft_rccl_group_t::~rocfft_rccl_group_t()
{
    ncclGroupEnd();
}

bool rocfft_rccl_comm_t::alltoall(const void* sendbuf,
                                  void*       recvbuf,
                                  size_t      count,
                                  int         device_id,
                                  hipStream_t stream,
                                  size_t      base_type_size,
                                  bool        is_complex)
{
    ncclComm_t* comm_ptr = static_cast<ncclComm_t*>(get_comm(device_id));
    if(!comm_ptr)
        return false;

    auto [dtype, multiplier] = get_nccl_type_info(base_type_size, is_complex);

    ncclResult_t result
        = ncclAllToAll(sendbuf, recvbuf, count * multiplier, dtype, *comm_ptr, stream);

    if(result != ncclSuccess)
    {
        log_trace(__func__, "ncclAllToAll failed", result);
        return false;
    }

    return true;
}

bool rocfft_rccl_comm_t::send(const void* sendbuf,
                              size_t      count,
                              int         peer_rank,
                              int         device_id,
                              hipStream_t stream,
                              size_t      base_type_size,
                              bool        is_complex)
{
    ncclComm_t* comm_ptr = static_cast<ncclComm_t*>(get_comm(device_id));
    if(!comm_ptr)
        return false;

    auto [dtype, multiplier] = get_nccl_type_info(base_type_size, is_complex);

    ncclResult_t result
        = ncclSend(sendbuf, count * multiplier, dtype, peer_rank, *comm_ptr, stream);

    if(result != ncclSuccess)
    {
        log_trace(__func__, "ncclSend failed", result);
        return false;
    }

    return true;
}

bool rocfft_rccl_comm_t::recv(void*       recvbuf,
                              size_t      count,
                              int         peer_rank,
                              int         device_id,
                              hipStream_t stream,
                              size_t      base_type_size,
                              bool        is_complex)
{
    ncclComm_t* comm_ptr = static_cast<ncclComm_t*>(get_comm(device_id));
    if(!comm_ptr)
        return false;

    auto [dtype, multiplier] = get_nccl_type_info(base_type_size, is_complex);

    ncclResult_t result
        = ncclRecv(recvbuf, count * multiplier, dtype, peer_rank, *comm_ptr, stream);

    if(result != ncclSuccess)
    {
        log_trace(__func__, "ncclRecv failed", result);
        return false;
    }

    return true;
}

#endif // ROCFFT_RCCL_ENABLE
