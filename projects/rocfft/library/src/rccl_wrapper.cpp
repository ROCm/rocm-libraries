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
#include "../../shared/hip_object_wrapper.h"
#include "../../shared/precision_type.h"
#include "../../shared/rocfft_hip.h"
#include "logging.h"
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

// map a rocFFT precision to the corresponding NCCL datatype.
// rocFFT half/float/double map to ncclFloat16/32/64.
//
// note: NCCL has no complex datatype, interleaved complex doubles the
// count (array_type_is_interleaved ? 2 : 1), planar does not.
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
        // rocFFT only produces half (2), float (4), or double (8); any
        // other size indicates a bug in the caller.
        throw std::runtime_error("unsupported rocfft_precision in RCCL datatype mapping");
    }
}

// implementation details shared by all copies of a handle via shared_ptr
struct rocfft_rccl_comm_t::Impl
{
    // per-device state: comm + the stream it launches on. RCCL requires a
    // comm to always use the same stream (NCCL "CUDA Stream Semantics"), so
    // the stream is owned here, not borrowed from a plan. Completion events
    // stay plan-side and are recorded onto this stream.
    struct device_state_t
    {
        device_state_t(const std::set<int>& devices, size_t rank, const ncclUniqueId& unique_id)
        {
            if(rank >= devices.size())
                throw std::out_of_range("device_state_t constructor: rank is out of range");
            device_id = *std::next(devices.begin(), rank);
            rocfft_scoped_device dev(device_id);
            stream.alloc();
            auto nccl_ret = ncclCommInitRank(
                &comm, static_cast<int>(devices.size()), unique_id, static_cast<int>(rank));
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

    private:
        int                 device_id;
        hipStream_wrapper_t stream;
        ncclComm_t          comm;
    };

    // keyed by device_id.
    std::map<int, device_state_t> device_to_state;

    // unique id used to bootstrap this communicator group.
    // stored so it can be broadcast via MPI for multi-node in the future.
    ncclUniqueId uniqueId{};

    // no explicit destructor: each device_state_t RAII-cleans its comm then
    // stream when device_to_state is destroyed
};

// static cache definitions; placed after Impl so shared_ptr<Impl> is complete
std::map<std::set<int>, rocfft_rccl_comm_t> rocfft_rccl_comm_t::comm_cache;
std::mutex                                  rocfft_rccl_comm_t::comm_cache_mutex;

rocfft_rccl_comm_t rocfft_rccl_comm_t::create(const std::set<int>& devices)
{
    // need at least 2 devices for a meaningful communicator
    if(devices.size() < 2)
        throw std::invalid_argument("rocfft_rccl_comm_t::create: need at least 2 devices");

    // look up or create a communicator for this exact device set.
    // guard with a mutex so concurrent plan creation from
    // multiple threads does not race on the cache.
    std::lock_guard<std::mutex> lock(comm_cache_mutex);

    auto it = comm_cache.find(devices);
    if(it != comm_cache.end())
    {
        // reuse is safe: the comm owns its stream, so the comm/stream
        // pairing holds across sequential and overlapping plans.
        return it->second;
    }

    rocfft_rccl_comm_t new_comm;
    new_comm.pimpl = std::make_shared<Impl>();

    // generate unique id for this communicator group,
    // for single-node this stays local, for multi-node the root
    // rank would broadcast this via MPI_Bcast
    ncclResult_t result = ncclGetUniqueId(&new_comm.pimpl->uniqueId);
    if(result != ncclSuccess)
        throw rocfft_rccl_exception_t("ncclGetUniqueId failed in rocfft_rccl_comm_t::create",
                                      result);

    // init one communicator per device using ncclCommInitRank,
    // batched inside a group call for single-process efficiency.
    // ranks are assigned in sorted device-id order
    rocfft_rccl_group_t group;
    for(size_t rank = 0; rank < devices.size(); rank++)
    {
        new_comm.pimpl->device_to_state.try_emplace(
            *std::next(devices.begin(), rank), devices, rank, new_comm.pimpl->uniqueId);
    }
    group.end();

    // owning ref: comm (and its streams) persists for reuse; freed at reset_all()
    comm_cache[devices] = new_comm;

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
                                    + " is not part of this communicator");
    return it->second.get_comm();
}

hipStream_t rocfft_rccl_comm_t::get_stream(int device_id) const
{
    auto it = pimpl->device_to_state.find(device_id);
    if(it == pimpl->device_to_state.end())
        throw std::invalid_argument("rocfft_rccl_comm_t::get_stream: device_id "
                                    + std::to_string(device_id)
                                    + " is not part of this communicator");
    return it->second.get_stream();
}

size_t rocfft_rccl_comm_t::num_ranks() const
{
    return pimpl->device_to_state.size();
}

int rocfft_rccl_comm_t::get_rank(int device_id) const
{
    auto it = pimpl->device_to_state.find(device_id);
    if(it == pimpl->device_to_state.end())
        throw std::invalid_argument("rocfft_rccl_comm_t::get_rank: device_id "
                                    + std::to_string(device_id)
                                    + " is not part of this communicator");
    int          rank   = -1;
    ncclResult_t result = ncclCommUserRank(it->second.get_comm(), &rank);
    if(result != ncclSuccess)
    {
        // logged by the general rocfft_handle_exception handler when it propagates
        throw rocfft_rccl_exception_t("rocfft_rccl_comm_t::get_rank: ncclCommUserRank failed",
                                      result);
    }
    return rank;
}

std::vector<int> rocfft_rccl_comm_t::get_devices() const
{
    // ranks are assigned in sorted device-id order in create(), so
    // std::map's natural ordering already gives us devices in rank order.
    std::vector<int> devices;
    devices.reserve(pimpl->device_to_state.size());
    for(const auto& [dev, state] : pimpl->device_to_state)
        devices.push_back(dev);
    return devices;
}

// RAII group wrapper
rocfft_rccl_group_t::rocfft_rccl_group_t()
{
    ncclResult_t result = ncclGroupStart();
    if(result != ncclSuccess)
    {
        // not logged here to avoid duplicate traces; the exception carries
        // the code and is logged where it is caught (or by the general handler)
        throw rocfft_rccl_exception_t("ncclGroupStart failed", result);
    }
    needs_ending = true;
}

void rocfft_rccl_group_t::end()
{
    if(!needs_ending)
        return;

    ncclResult_t result = ncclGroupEnd();
    // clear before checking the result so a throw here does not make
    // the destructor retry ncclGroupEnd on the same group
    needs_ending = false;
    if(result != ncclSuccess)
    {
        // not logged here to avoid duplicate traces; logged where caught
        throw rocfft_rccl_exception_t("ncclGroupEnd failed", result);
    }
}

rocfft_rccl_group_t::~rocfft_rccl_group_t() noexcept
{
    // safety net for early returns / stack unwinding where end() was
    // not called explicitly
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

    // resolve precision/complex/device mapping once outside the loop
    const auto devices = get_devices();
    // interleaved complex = 2 real scalars per element; planar/real = 1
    const auto nccl_count = count * (array_type_is_interleaved(array_type) ? 2 : 1);
    const auto dtype      = get_nccl_dtype(precision);

    // batch per-device calls in one RCCL group (required for single-process
    // multi-GPU); each device launches on its comm-owned stream.
    rocfft_rccl_group_t group;

    for(size_t r = 0; r < nranks; ++r)
    {
        rocfft_scoped_device dev(devices[r]);

        ncclResult_t result = ncclAllToAll(sendbufs[r],
                                           recvbufs[r],
                                           nccl_count,
                                           dtype,
                                           get_comm(devices[r]),
                                           get_stream(devices[r]));

        if(result != ncclSuccess)
        {
            // logged by the general rocfft_handle_exception handler when it propagates
            throw rocfft_rccl_exception_t(
                "ncclAllToAll failed on device " + std::to_string(devices[r]), result);
        }
    }

    group.end();
}

void rocfft_rccl_comm_t::send(const void*       sendbuf,
                              size_t            count,
                              int               peer_device_id,
                              int               device_id,
                              rocfft_precision  precision,
                              rocfft_array_type array_type) const
{
    ncclComm_t comm      = get_comm(device_id);
    const int  peer_rank = get_rank(peer_device_id);

    ncclResult_t result = ncclSend(sendbuf,
                                   count * (array_type_is_interleaved(array_type) ? 2 : 1),
                                   get_nccl_dtype(precision),
                                   peer_rank,
                                   comm,
                                   get_stream(device_id));

    if(result != ncclSuccess)
    {
        // logged by the general rocfft_handle_exception handler when it propagates
        throw rocfft_rccl_exception_t("ncclSend failed on device " + std::to_string(device_id)
                                          + " to peer device " + std::to_string(peer_device_id),
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
    ncclComm_t comm      = get_comm(device_id);
    const int  peer_rank = get_rank(peer_device_id);

    ncclResult_t result = ncclRecv(recvbuf,
                                   count * (array_type_is_interleaved(array_type) ? 2 : 1),
                                   get_nccl_dtype(precision),
                                   peer_rank,
                                   comm,
                                   get_stream(device_id));

    if(result != ncclSuccess)
    {
        // logged by the general rocfft_handle_exception handler when it propagates
        throw rocfft_rccl_exception_t("ncclRecv failed on device " + std::to_string(device_id)
                                          + " from peer device " + std::to_string(peer_device_id),
                                      result);
    }
}

#endif // ROCFFT_RCCL_ENABLE
