/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef RPP_TEST_BACKEND_MEMORY_H
#define RPP_TEST_BACKEND_MEMORY_H

#include <rpp/rpp.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "framework/backend_param.hpp"

#if defined(RPP_TEST_HAVE_HIP) && RPP_TEST_HAVE_HIP
#include <hip/hip_runtime.h>
#define RPP_TEST_CHECK_HIP(cmd)                                                          \
    do {                                                                                 \
        hipError_t rpp_test_hip_err = (cmd);                                             \
        if (rpp_test_hip_err != hipSuccess)                                              \
            throw std::runtime_error(std::string("HIP error: ") +                        \
                                     hipGetErrorString(rpp_test_hip_err));               \
    } while (0)
#endif

namespace rpptest {

// RAII wrapper around an rpp handle for the given backend.
class RppHandle {
   public:
    RppHandle(RppBackend backend, std::size_t batchSize) : backend_(backend) {
        if (rppCreate(&handle_, batchSize, 0, nullptr, backend_) != rppStatusSuccess)
            throw std::runtime_error("rppCreate failed");
    }
    ~RppHandle() {
        if (handle_) rppDestroy(handle_, backend_);
    }
    RppHandle(const RppHandle&) = delete;
    RppHandle& operator=(const RppHandle&) = delete;

    rppHandle_t get() const { return handle_; }

    // Blocks the host until all work on this handle's accelerator stream completes; no-op for
    // HOST. Mirrors what a downstream user does before reading results back: synchronize the
    // op's own stream (not the whole device) after launching an op.
    void sync() const {
        if (backend_ == RPP_HIP_BACKEND) {
#if defined(RPP_TEST_HAVE_HIP) && RPP_TEST_HAVE_HIP
            rppAcceleratorQueue_t stream = nullptr;
            if (rppGetStream(handle_, &stream) != rppStatusSuccess)
                throw std::runtime_error("rppGetStream failed");
            RPP_TEST_CHECK_HIP(hipStreamSynchronize(stream));
#endif
        }
    }

   private:
    RppBackend backend_;
    rppHandle_t handle_ = nullptr;
};

// Host-accessible parameter array (alpha/beta/ROI tensors). Plain host memory for the
// HOST backend; pinned (device-accessible) host memory for HIP. Written/read directly.
template <typename T>
class PinnedArray {
   public:
    PinnedArray(RppBackend backend, std::size_t count) : backend_(backend), count_(count) {
        if (backend_ == RPP_HIP_BACKEND) {
#if defined(RPP_TEST_HAVE_HIP) && RPP_TEST_HAVE_HIP
            RPP_TEST_CHECK_HIP(hipHostMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)));
#endif
        } else {
            data_ = new T[count_];
        }
    }
    ~PinnedArray() {
        if (backend_ == RPP_HIP_BACKEND) {
#if defined(RPP_TEST_HAVE_HIP) && RPP_TEST_HAVE_HIP
            (void)hipHostFree(data_);
#endif
        } else {
            delete[] data_;
        }
    }
    PinnedArray(const PinnedArray&) = delete;
    PinnedArray& operator=(const PinnedArray&) = delete;

    T* data() { return data_; }
    std::size_t size() const { return count_; }
    T& operator[](std::size_t i) { return data_[i]; }

   private:
    RppBackend backend_;
    std::size_t count_;
    T* data_ = nullptr;
};

// IO tensor storage passed to RPP: host memory for HOST, device memory for HIP.
// write()/read() move bytes between a host buffer and this storage (memcpy on HOST,
// hipMemcpy on HIP). read() does not synchronize: the caller must first drain the op's
// stream with RppHandle::sync() so the kernel has finished before the copy reads its output.
class DeviceTensor {
   public:
    DeviceTensor(RppBackend backend, std::size_t bytes) : backend_(backend), bytes_(bytes) {
        if (backend_ == RPP_HIP_BACKEND) {
#if defined(RPP_TEST_HAVE_HIP) && RPP_TEST_HAVE_HIP
            RPP_TEST_CHECK_HIP(hipMalloc(&data_, bytes_));
#endif
        } else {
            data_ = std::malloc(bytes_);
        }
    }
    ~DeviceTensor() {
        if (backend_ == RPP_HIP_BACKEND) {
#if defined(RPP_TEST_HAVE_HIP) && RPP_TEST_HAVE_HIP
            (void)hipFree(data_);
#endif
        } else {
            std::free(data_);
        }
    }
    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    void* ptr() const { return data_; }

    void write(const void* host, std::size_t bytes) {
        if (backend_ == RPP_HIP_BACKEND) {
#if defined(RPP_TEST_HAVE_HIP) && RPP_TEST_HAVE_HIP
            RPP_TEST_CHECK_HIP(hipMemcpy(data_, host, bytes, hipMemcpyHostToDevice));
#endif
        } else {
            std::memcpy(data_, host, bytes);
        }
    }

    void read(void* host, std::size_t bytes) const {
        if (backend_ == RPP_HIP_BACKEND) {
#if defined(RPP_TEST_HAVE_HIP) && RPP_TEST_HAVE_HIP
            RPP_TEST_CHECK_HIP(hipMemcpy(host, data_, bytes, hipMemcpyDeviceToHost));
#endif
        } else {
            std::memcpy(host, data_, bytes);
        }
    }

   private:
    RppBackend backend_;
    std::size_t bytes_;
    void* data_ = nullptr;
};

}  // namespace rpptest

#endif  // RPP_TEST_BACKEND_MEMORY_H
