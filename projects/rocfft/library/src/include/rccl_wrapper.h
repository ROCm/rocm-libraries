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

#include "rocfft_location.h"
#ifdef ROCFFT_MPI_ENABLE
#include "rocfft_mpi.h"
#endif

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

// value-semantic handle to an RCCL communicator whose ranks are
// sorted rocfft_location_t values (comm_rank, then device).
//
// Single-process create(devices) is the special case where every
// location has comm_rank 0. Multi-process create(mpi_comm, ...)
// initializes only the caller's local locations; NCCL rank is the
// index in the sorted world set (with one GPU per MPI rank that
// equals the MPI rank).
//
// Thread safety: create()/reset_all() are internally synchronized. A given
// comm is NOT safe for concurrent use (per NCCL: only one thread may
// operate a comm at a time), so plans sharing a comm must be executed
// serially; concurrent use needs caller-side serialization.
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

    // single-process communicator spanning the given local devices
    // (NCCL ranks = sorted device ids on comm_rank 0). Need >= 2 devices.
    // Communicators are cached per world location set.
    static rocfft_rccl_comm_t create(const std::set<int>& devices);

#ifdef ROCFFT_MPI_ENABLE
    // multi-process communicator. world must contain every participating
    // (mpi_rank, device); this rank initializes only locations whose
    // comm_rank == local_comm_rank. Collective on mpi_comm: unique id is
    // broadcast from rank 0, then every rank calls ncclCommInitRank.
    static rocfft_rccl_comm_t
        create(MPI_Comm mpi_comm, int local_comm_rank, const std::set<rocfft_location_t>& world);
#endif

    // release all cached communicators (called at rocfft_cleanup()).
    static void reset_all();

    // return the RCCL communicator for a local device. Throws
    // std::invalid_argument if device_id is not a local participant.
    ncclComm_t get_comm(int device_id) const;

    // communicator-owned stream for a local device. RCCL requires a comm to
    // always use the same stream, so the stream lives/dies with the comm;
    // callers record their own event on it to sync.
    hipStream_t get_stream(int device_id) const;

    // total number of NCCL ranks (world size), not the local GPU count
    size_t num_ranks() const;

    // NCCL rank of a world location. Throws if the location is not in the world.
    int get_rank(const rocfft_location_t& location) const;

    // single-process helper: NCCL rank of device_id on comm_rank 0
    int get_rank(int device_id) const;

    // world locations in NCCL rank order
    std::vector<rocfft_location_t> get_locations() const;

    // local (this process) locations in NCCL rank order
    std::vector<rocfft_location_t> get_local_locations() const;

    // local device IDs in NCCL rank order among local participants.
    // for a single-process comm this is the full device list.
    std::vector<int> get_devices() const;

    // all-to-all with uniform counts. Sendbufs/recvbufs are sized
    // num_ranks() and indexed by NCCL rank. Non-local slots may be
    // nullptr; only local participants are launched. Count is in
    // logical (precision, array_type) elements.
    void alltoall(const std::vector<const void*>& sendbufs,
                  const std::vector<void*>&       recvbufs,
                  size_t                          count,
                  rocfft_precision                precision,
                  rocfft_array_type               array_type) const;

    // point-to-point send from a local device to a peer location
    void send(const void*              sendbuf,
              size_t                   count,
              const rocfft_location_t& peer,
              int                      device_id,
              rocfft_precision         precision,
              rocfft_array_type        array_type) const;

    // single-process helper: peer identified by device id on comm_rank 0
    void send(const void*       sendbuf,
              size_t            count,
              int               peer_device_id,
              int               device_id,
              rocfft_precision  precision,
              rocfft_array_type array_type) const;

    // point-to-point receive on a local device from a peer location
    void recv(void*                    recvbuf,
              size_t                   count,
              const rocfft_location_t& peer,
              int                      device_id,
              rocfft_precision         precision,
              rocfft_array_type        array_type) const;

    // single-process helper: peer identified by device id on comm_rank 0
    void recv(void*             recvbuf,
              size_t            count,
              int               peer_device_id,
              int               device_id,
              rocfft_precision  precision,
              rocfft_array_type array_type) const;

private:
    struct Impl;

    static rocfft_rccl_comm_t create_from_world(const std::set<rocfft_location_t>& world,
                                                int                                local_comm_rank
#ifdef ROCFFT_MPI_ENABLE
                                                ,
                                                MPI_Comm mpi_comm
#endif
    );

    // owning cache keyed by world location set
    static std::map<std::set<rocfft_location_t>, rocfft_rccl_comm_t> comm_cache;
    static std::mutex                                                comm_cache_mutex;

    std::shared_ptr<Impl> pimpl;
};

// RAII wrapper for RCCL group operations
class rocfft_rccl_group_t
{
public:
    rocfft_rccl_group_t();
    ~rocfft_rccl_group_t() noexcept;

    void end();

    rocfft_rccl_group_t(const rocfft_rccl_group_t&) = delete;
    rocfft_rccl_group_t& operator=(const rocfft_rccl_group_t&) = delete;
    rocfft_rccl_group_t(rocfft_rccl_group_t&&)                 = delete;
    rocfft_rccl_group_t& operator=(rocfft_rccl_group_t&&) = delete;

private:
    bool needs_ending = false;
};

#endif // ROCFFT_RCCL_ENABLE

#endif // ROCFFT_RCCL_WRAPPER_H
