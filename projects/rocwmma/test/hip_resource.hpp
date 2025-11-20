/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef ROCWMMA_HIP_RESOURCE_HPP
#define ROCWMMA_HIP_RESOURCE_HPP

#include <memory>
#include <rocwmma/internal/types.hpp>

// The HipResource class is intended as a wrapper for allocation, deletion and copying
// between host and device resources using the HIP backend.
// Memory is treated as a 1D array, and is managed through the std::unique_ptr class.

namespace rocwmma
{

    struct HipResource
    {
    protected:
        HipResource() = default;

    private: // No Copy
        HipResource(HipResource&&)                 = delete;
        HipResource(const HipResource&)            = delete;
        HipResource& operator=(const HipResource&) = delete;

    public:
        virtual ~HipResource() = default;

        // Types
        template <typename DataT>
        using DevicePtrT = std::unique_ptr<DataT, void (*)(DataT*)>;

        template <typename DataT>
        using HostPtrT = std::unique_ptr<DataT[]>;

        // Alloc
        template <typename DataT>
        static inline DevicePtrT<DataT> allocDevice(int64_t numElements);

        template <typename DataT>
        static inline void reallocDevice(DevicePtrT<DataT>& devicePtr, int64_t numElements);

        template <typename DataT>
        static inline HostPtrT<DataT> allocHost(int64_t numElements);

        template <typename DataT>
        static inline void reallocHost(HostPtrT<DataT>& hostPtr, int64_t numElements);

        template <typename DataT>
        static inline void reallocDeviceHostPair(DevicePtrT<DataT>& devicePtr,
                                                 HostPtrT<DataT>&   hostPtr,
                                                 int64_t            numElements);

        // Transfer wrappers
        template <typename DataT>
        static void
            copyData(HostPtrT<DataT>& dst, DevicePtrT<DataT> const& src, int64_t numElements);
        template <typename DataT>
        static void
            copyData(DevicePtrT<DataT>& dst, HostPtrT<DataT> const& src, int64_t numElements);
        template <typename DataT>
        static void copyData(HostPtrT<DataT>& dst, HostPtrT<DataT> const& src, int64_t numElements);
        template <typename DataT>
        static void
            copyData(DevicePtrT<DataT>& dst, DevicePtrT<DataT> const& src, int64_t numElements);

        virtual void reset() = 0;
    };

} // namespace rocwmma

#include "hip_resource_impl.hpp"

#endif // ROCWMMA_HIP_RESOURCE_HPP
