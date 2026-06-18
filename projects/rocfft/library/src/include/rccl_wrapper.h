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
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

#include <rccl/rccl.h>
#include <rocfft/rocfft.h>

// thrown by rocfft_rccl_comm_t communication primitives when the
// underlying RCCL call fails.  the distinct type lets callers
// recognize and handle RCCL failures specifically while still
// being catchable via std::runtime_error / std::exception.  carries
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

    // true iff this handle refers to an initialized RCCL communicator.
    explicit operator bool() const
    {
        return static_cast<bool>(pimpl);
    }

    // return a populated handle for the specified devices, or an empty
    // handle if RCCL is disabled, fewer than two devices were given, or
    // initialization failed.  Communicators are cached per device-set
    // so different plans can use different GPU subsets concurrently.
    static rocfft_rccl_comm_t create(const std::set<int>& devices);

    // release all cached communicators (called at rocfft_cleanup()).
    static void reset_all();

    // return the RCCL communicator for a specific device.  throws
    // std::invalid_argument if device_id is not part of this
    // communicator set.
    ncclComm_t get_comm(int device_id) const;

    // total number of ranks in this communicator
    size_t num_ranks() const;

    // NCCL rank assigned to the given device
    int get_rank(int device_id) const;

    // device IDs in RCCL rank order (rank 0 first, ..., rank num_ranks()-1 last).
    // useful for callers that need to iterate over the communicator's devices
    // in a well-defined order matching the NCCL rank numbering.
    std::vector<int> get_devices() const;

    // all-to-all with uniform counts across every rank.  the three
    // per-rank vectors (sendbufs / recvbufs / streams) must each
    // have size num_ranks() and be indexed by RCCL rank; the
    // wrapper owns the ncclGroupStart/End scope and sets the
    // current device per call internally, so callers cannot
    // accidentally launch a partial collective.
    //
    // count is in elements of the logical rocFFT type described by
    // (precision, array_type); the wrapper internally maps this to
    // the matching ncclDataType_t and adjusts the element count for
    // complex/planar layouts.
    //
    // throws std::invalid_argument if any vector size mismatches
    // num_ranks(); throws rocfft_rccl_exception_t on RCCL failure.
    void alltoall(const std::vector<const void*>& sendbufs,
                  const std::vector<void*>&       recvbufs,
                  const std::vector<hipStream_t>& streams,
                  size_t                          count,
                  rocfft_precision                precision,
                  rocfft_array_type               array_type) const;

    // point-to-point send: endpoints are device ids (peer / local),
    // throws rocfft_rccl_exception_t on RCCL failure.
    void send(const void*       sendbuf,
              size_t            count,
              int               peer_device_id,
              int               device_id,
              hipStream_t       stream,
              rocfft_precision  precision,
              rocfft_array_type array_type) const;

    // point-to-point receive: endpoints are device ids (peer / local),
    // throws rocfft_rccl_exception_t on RCCL failure.
    void recv(void*             recvbuf,
              size_t            count,
              int               peer_device_id,
              int               device_id,
              hipStream_t       stream,
              rocfft_precision  precision,
              rocfft_array_type array_type) const;

private:
    struct Impl;
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
