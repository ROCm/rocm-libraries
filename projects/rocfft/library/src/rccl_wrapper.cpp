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
#include "../../shared/array_predicate.h"
#include "../../shared/precision_type.h"
#include "../../shared/rocfft_hip.h"
#include "logging.h"
#include <map>
#include <mutex>
#include <stdexcept>

// process-wide cache of communicators keyed by device set.
// different plans may use different GPU subsets, so each distinct
// set gets its own RCCL communicator.
static std::map<std::set<int>, rocfft_rccl_comm_t> comm_cache;
static std::mutex                                  comm_cache_mutex;

// map a rocFFT precision to the corresponding NCCL datatype.
// rocFFT half/float/double map to ncclFloat16/32/64.
//
// note: NCCL has no native complex datatype, so callers transferring
// complex layouts must double the element count via
// array_type_is_complex(array_type) ? 2 : 1.
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
    }
    // rocFFT only produces half (2), float (4), or double (8); any
    // other size indicates a bug in the caller.
    throw std::runtime_error("unsupported rocfft_precision in RCCL datatype mapping");
}

// implementation details shared by all copies of a handle via shared_ptr
struct rocfft_rccl_comm_t::Impl
{
    // one communicator per device, keyed by device_id.
    std::map<int, ncclComm_t> device_to_comm;

    // unique id used to bootstrap this communicator group.
    // stored so it can be broadcast via MPI for multi-node in the future.
    ncclUniqueId uniqueId{};

    ~Impl()
    {
        for(auto& [dev, comm] : device_to_comm)
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
        ncclResult_t result = ncclGetUniqueId(&new_comm.pimpl->uniqueId);
        if(result != ncclSuccess)
        {
            return {};
        }

        // init one communicator per device using ncclCommInitRank,
        // batched inside a group call for single-process efficiency.
        // ranks are assigned in sorted device-id order (std::set).
        {
            rocfft_rccl_group_t group;
            int                 rank = 0;
            for(int dev : devices)
            {
                rocfft_scoped_device set_dev(dev);
                ncclComm_t           comm = nullptr;
                result = ncclCommInitRank(&comm, ndevices, new_comm.pimpl->uniqueId, rank);
                if(result != ncclSuccess)
                    return {};
                new_comm.pimpl->device_to_comm[dev] = comm;
                ++rank;
            }
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

ncclComm_t rocfft_rccl_comm_t::get_comm(int device_id) const
{
    auto it = pimpl->device_to_comm.find(device_id);
    if(it == pimpl->device_to_comm.end())
        throw std::invalid_argument("rocfft_rccl_comm_t::get_comm: device_id "
                                    + std::to_string(device_id)
                                    + " is not part of this communicator");
    return it->second;
}

size_t rocfft_rccl_comm_t::num_ranks() const
{
    return pimpl->device_to_comm.size();
}

int rocfft_rccl_comm_t::get_rank(int device_id) const
{
    auto it = pimpl->device_to_comm.find(device_id);
    if(it == pimpl->device_to_comm.end())
        throw std::invalid_argument("rocfft_rccl_comm_t::get_rank: device_id "
                                    + std::to_string(device_id)
                                    + " is not part of this communicator");
    int rank = -1;
    if(ncclCommUserRank(it->second, &rank) != ncclSuccess)
        throw std::runtime_error("rocfft_rccl_comm_t::get_rank: ncclCommUserRank failed");
    return rank;
}

std::vector<int> rocfft_rccl_comm_t::get_devices() const
{
    // ranks are assigned in sorted device-id order in create(), so
    // std::map's natural ordering already gives us devices in rank order.
    std::vector<int> devices;
    devices.reserve(pimpl->device_to_comm.size());
    for(const auto& [dev, comm] : pimpl->device_to_comm)
        devices.push_back(dev);
    return devices;
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

void rocfft_rccl_comm_t::alltoall(const void*       sendbuf,
                                  void*             recvbuf,
                                  size_t            count,
                                  int               device_id,
                                  hipStream_t       stream,
                                  rocfft_precision  precision,
                                  rocfft_array_type array_type) const
{
    ncclComm_t comm = get_comm(device_id);

    ncclResult_t result = ncclAllToAll(sendbuf,
                                       recvbuf,
                                       count * (array_type_is_complex(array_type) ? 2 : 1),
                                       get_nccl_dtype(precision),
                                       comm,
                                       stream);

    if(result != ncclSuccess)
    {
        log_trace(__func__, "ncclAllToAll failed", result);
        throw rocfft_rccl_exception_t("ncclAllToAll failed on device " + std::to_string(device_id)
                                      + ": " + ncclGetErrorString(result));
    }
}

void rocfft_rccl_comm_t::send(const void*       sendbuf,
                              size_t            count,
                              int               peer_rank,
                              int               device_id,
                              hipStream_t       stream,
                              rocfft_precision  precision,
                              rocfft_array_type array_type) const
{
    ncclComm_t comm = get_comm(device_id);

    ncclResult_t result = ncclSend(sendbuf,
                                   count * (array_type_is_complex(array_type) ? 2 : 1),
                                   get_nccl_dtype(precision),
                                   peer_rank,
                                   comm,
                                   stream);

    if(result != ncclSuccess)
    {
        log_trace(__func__, "ncclSend failed", result);
        throw rocfft_rccl_exception_t("ncclSend failed on device " + std::to_string(device_id)
                                      + " to peer " + std::to_string(peer_rank) + ": "
                                      + ncclGetErrorString(result));
    }
}

void rocfft_rccl_comm_t::recv(void*             recvbuf,
                              size_t            count,
                              int               peer_rank,
                              int               device_id,
                              hipStream_t       stream,
                              rocfft_precision  precision,
                              rocfft_array_type array_type) const
{
    ncclComm_t comm = get_comm(device_id);

    ncclResult_t result = ncclRecv(recvbuf,
                                   count * (array_type_is_complex(array_type) ? 2 : 1),
                                   get_nccl_dtype(precision),
                                   peer_rank,
                                   comm,
                                   stream);

    if(result != ncclSuccess)
    {
        log_trace(__func__, "ncclRecv failed", result);
        throw rocfft_rccl_exception_t("ncclRecv failed on device " + std::to_string(device_id)
                                      + " from peer " + std::to_string(peer_rank) + ": "
                                      + ncclGetErrorString(result));
    }
}

#endif // ROCFFT_RCCL_ENABLE
