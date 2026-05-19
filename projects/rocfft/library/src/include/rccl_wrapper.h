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
#include <vector>

#include <rccl/rccl.h>
#include <rocfft/rocfft.h>

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

    // get the RCCL communicator for a specific device
    void* get_comm(int device_id) const;

    // total number of ranks in this communicator
    int num_ranks() const;

    // NCCL rank assigned to the given device, or -1 if not found
    int get_rank(int device_id) const;

    // all-to-all with uniform counts.  count is in elements of the
    // logical rocFFT type described by (precision, array_type); the
    // wrapper internally maps this to the matching ncclDataType_t and
    // adjusts the element count for complex/planar layouts.
    bool alltoall(const void*       sendbuf,
                  void*             recvbuf,
                  size_t            count,
                  int               device_id,
                  hipStream_t       stream,
                  rocfft_precision  precision,
                  rocfft_array_type array_type);

    // point-to-point send
    bool send(const void*       sendbuf,
              size_t            count,
              int               peer_rank,
              int               device_id,
              hipStream_t       stream,
              rocfft_precision  precision,
              rocfft_array_type array_type);

    // point-to-point receive
    bool recv(void*             recvbuf,
              size_t            count,
              int               peer_rank,
              int               device_id,
              hipStream_t       stream,
              rocfft_precision  precision,
              rocfft_array_type array_type);

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
    rocfft_rccl_group_t();
    ~rocfft_rccl_group_t();

    // non-copyable, non-movable
    rocfft_rccl_group_t(const rocfft_rccl_group_t&) = delete;
    rocfft_rccl_group_t& operator=(const rocfft_rccl_group_t&) = delete;
    rocfft_rccl_group_t(rocfft_rccl_group_t&&)                 = delete;
    rocfft_rccl_group_t& operator=(rocfft_rccl_group_t&&) = delete;
};

#endif // ROCFFT_RCCL_ENABLE

#endif // ROCFFT_RCCL_WRAPPER_H
