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

#ifndef ROCFFT_RCCL_WRAPPER_H
#define ROCFFT_RCCL_WRAPPER_H

// this header is only meaningful when rocFFT is built with RCCL support.
// callers must guard their inclusion with ROCFFT_RCCL_ENABLE as well; with
// the macro undefined the file expands to nothing.
#ifdef ROCFFT_RCCL_ENABLE

#include <cstddef>
#include <hip/hip_runtime.h>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <vector>

#include <rccl/rccl.h>
#include <rocfft/rocfft.h>

// thrown by rocfft_rccl_comm_t communication primitives when the
// underlying RCCL call fails. The distinct type lets callers
// recognize and handle RCCL failures specifically while still
// being catchable via std::runtime_error / std::exception. Carries
// the originating ncclResult_t and appends its string form to what()
struct rocfft_rccl_exception_t : std::runtime_error
{
    rocfft_rccl_exception_t(std::string message, ncclResult_t code)
        : std::runtime_error(message)
        , error(code)
    {
        what_message = std::move(message) + " (" + ncclGetErrorString(error) + ")";
    }

    const char* what() const noexcept override
    {
        return what_message.c_str();
    }

private:
    const ncclResult_t error;
    std::string        what_message;
};

// value-semantic handle to an RCCL communicator set for single-process
// multi-GPU transfers.
//
// Thread safety: create()/reset_all() are internally synchronized. A given
// comm is NOT safe for concurrent use (per NCCL: only one thread may
// operate a comm at a time), so plans sharing a comm (same device set) must
// be executed serially; concurrent use needs caller-side serialization.
class rocfft_rccl_comm_t
{
public:
    // default-constructs an empty (unpopulated) handle.
    rocfft_rccl_comm_t()  = default;
    ~rocfft_rccl_comm_t() = default;

    // copy/move share the underlying Impl via shared_ptr; no duplication
    // of ncclComm_t handles occurs.
    rocfft_rccl_comm_t(const rocfft_rccl_comm_t&) = default;
    rocfft_rccl_comm_t& operator=(const rocfft_rccl_comm_t&) = default;
    rocfft_rccl_comm_t(rocfft_rccl_comm_t&&)                 = default;
    rocfft_rccl_comm_t& operator=(rocfft_rccl_comm_t&&) = default;

    // true if this handle refers to an initialized RCCL communicator
    explicit operator bool() const
    {
        return static_cast<bool>(pimpl);
    }

    // return a populated handle for the specified devices, or an empty
    // handle if RCCL is disabled, fewer than two devices were given, or
    // initialization failed. Communicators are cached per device-set
    // so different plans can use different GPU subsets concurrently.
    static rocfft_rccl_comm_t create(const std::set<int>& devices);

    // release all cached communicators (called at rocfft_cleanup()).
    static void reset_all();

    // return the RCCL communicator for a specific device. Throws
    // std::invalid_argument if device_id is not part of this
    // communicator set.
    ncclComm_t get_comm(int device_id) const;

    // communicator-owned stream for a device. RCCL requires a comm to
    // always use the same stream, so the stream lives/dies with the comm;
    // callers record their own event on it to sync.
    hipStream_t get_stream(int device_id) const;

    // total number of ranks in this communicator
    size_t num_ranks() const;

    // NCCL rank assigned to the given device
    int get_rank(int device_id) const;

    // device IDs in RCCL rank order (rank 0 first, ..., rank num_ranks()-1 last).
    // useful for callers that need to iterate over the communicator's devices
    // in a well-defined order matching the NCCL rank numbering.
    std::vector<int> get_devices() const;

    // all-to-all with uniform counts. Sendbufs/recvbufs are sized
    // num_ranks() and indexed by RCCL rank; the wrapper owns the
    // group scope, per-call hipSetDevice, and launches on each comm's
    // own stream. Count is in logical (precision, array_type) elements
    // (mapped to ncclDataType_t / adjusted for complex/planar inside).
    // Throws std::invalid_argument on size mismatch, rocfft_rccl_exception_t
    // on RCCL failure.
    void alltoall(const std::vector<const void*>& sendbufs,
                  const std::vector<void*>&       recvbufs,
                  size_t                          count,
                  rocfft_precision                precision,
                  rocfft_array_type               array_type) const;

    // point-to-point send: endpoints are device ids; runs on the comm's
    // own stream for device_id.
    void send(const void*       sendbuf,
              size_t            count,
              int               peer_device_id,
              int               device_id,
              rocfft_precision  precision,
              rocfft_array_type array_type) const;

    // point-to-point receive: endpoints are device ids; runs on the comm's
    // own stream for device_id. Throws rocfft_rccl_exception_t on failure.
    void recv(void*             recvbuf,
              size_t            count,
              int               peer_device_id,
              int               device_id,
              rocfft_precision  precision,
              rocfft_array_type array_type) const;

private:
    struct Impl;

    // owning cache keyed by device set: the comm (and its streams) persists
    // for reuse by later plans, amortizing ncclCommInitRank; freed at
    // reset_all(). Reuse is safe because the comm owns its stream, keeping
    // RCCL's fixed comm/stream pairing across sequential and overlapping plans.
    // The mutex only makes looking up / creating a cached comm thread-safe;
    // it does NOT make using a comm's collectives thread-safe (a comm still
    // must be used by one thread at a time)
    static std::map<std::set<int>, rocfft_rccl_comm_t> comm_cache;
    static std::mutex                                  comm_cache_mutex;

    // shared so copies of the handle refer to the same RCCL state; the
    // Impl destructor (running exactly once when the last handle dies)
    // calls ncclCommFinalize/Destroy on the owned communicators.
    std::shared_ptr<Impl> pimpl;
};

// RAII wrapper for RCCL group operations
class rocfft_rccl_group_t
{
public:
    // opens an RCCL group, throws rocfft_rccl_exception_t if
    // ncclGroupStart fails
    rocfft_rccl_group_t();
    ~rocfft_rccl_group_t() noexcept;

    // throws rocfft_rccl_exception_t on ncclGroupEnd failure
    void end();

    // non-copyable, non-movable
    rocfft_rccl_group_t(const rocfft_rccl_group_t&) = delete;
    rocfft_rccl_group_t& operator=(const rocfft_rccl_group_t&) = delete;
    rocfft_rccl_group_t(rocfft_rccl_group_t&&)                 = delete;
    rocfft_rccl_group_t& operator=(rocfft_rccl_group_t&&) = delete;

private:
    // true between a successful ncclGroupStart and the matching ncclGroupEnd
    bool needs_ending = false;
};

#endif // ROCFFT_RCCL_ENABLE

#endif // ROCFFT_RCCL_WRAPPER_H
