/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

// Small RAII helpers for tests that drive the public API directly and check the results against
// their own loop references (not the library's CPU reference).

#ifndef HIPTENSOR_TEST_API_TEST_HELPERS_HPP
#define HIPTENSOR_TEST_API_TEST_HELPERS_HPP

#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>
#include <hiptensor/internal/hiptensor_utility.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace hiptensor_test
{
    // Sets (or, for nullptr, removes) an environment variable for the life of the object.
    class ScopedEnvironmentVariable
    {
    public:
        ScopedEnvironmentVariable(char const* name, char const* value)
            : mName(name)
        {
            if(char const* previous = std::getenv(name))
            {
                mPrevious    = previous;
                mHadPrevious = true;
            }
            set(value);
        }

        ~ScopedEnvironmentVariable()
        {
            set(mHadPrevious ? mPrevious.c_str() : nullptr);
        }

        ScopedEnvironmentVariable(ScopedEnvironmentVariable const&)            = delete;
        ScopedEnvironmentVariable& operator=(ScopedEnvironmentVariable const&) = delete;

    private:
        void set(char const* value)
        {
#ifdef _WIN32
            _putenv_s(mName.c_str(), value != nullptr ? value : "");
#else
            if(value != nullptr)
            {
                setenv(mName.c_str(), value, 1);
            }
            else
            {
                unsetenv(mName.c_str());
            }
#endif
        }

        std::string mName;
        std::string mPrevious;
        bool        mHadPrevious = false;
    };

    // Makes the plans created while it lives use one backend ("ck" or "rocke"), whatever
    // HIPTENSOR_BACKEND and HIPTENSOR_BACKEND_AB select for the rest of the process. For tests of
    // one backend's own behaviour (the Composable Kernel backend's tensor size limits, for
    // example), which the other backend does not share.
    class ScopedBackend
    {
    public:
        explicit ScopedBackend(char const* name)
            : mBackend("HIPTENSOR_BACKEND", name)
            , mAb("HIPTENSOR_BACKEND_AB", nullptr)
        {
        }

    private:
        ScopedEnvironmentVariable mBackend;
        ScopedEnvironmentVariable mAb;
    };

    template <typename T>
    struct DeviceBuffer
    {
        explicit DeviceBuffer(std::vector<T> const& host)
            : count(host.size())
        {
            CHECK_HIP_ERROR(hipMalloc(&ptr, count * sizeof(T)));
            write(host);
        }
        ~DeviceBuffer()
        {
            if(ptr)
            {
                static_cast<void>(hipFree(ptr));
            }
        }
        DeviceBuffer(DeviceBuffer const&)            = delete;
        DeviceBuffer& operator=(DeviceBuffer const&) = delete;

        void write(std::vector<T> const& host)
        {
            CHECK_HIP_ERROR(hipMemcpy(ptr, host.data(), count * sizeof(T), hipMemcpyHostToDevice));
        }
        std::vector<T> read() const
        {
            std::vector<T> host(count);
            CHECK_HIP_ERROR(hipMemcpy(host.data(), ptr, count * sizeof(T), hipMemcpyDeviceToHost));
            return host;
        }

        T*     ptr = nullptr;
        size_t count;
    };

    // Owns a handle and every descriptor, operation descriptor and plan created through it.
    struct ApiObjects
    {
        hiptensorHandle_t                           handle = nullptr;
        std::vector<hiptensorTensorDescriptor_t>    tensors;
        std::vector<hiptensorOperationDescriptor_t> operations;
        std::vector<hiptensorPlan_t>                plans;

        ApiObjects()
        {
            CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));
        }
        ~ApiObjects()
        {
            for(auto plan : plans)
            {
                static_cast<void>(hiptensorDestroyPlan(plan));
            }
            for(auto op : operations)
            {
                static_cast<void>(hiptensorDestroyOperationDescriptor(op));
            }
            for(auto t : tensors)
            {
                static_cast<void>(hiptensorDestroyTensorDescriptor(t));
            }
            static_cast<void>(hiptensorDestroy(handle));
        }
        ApiObjects(ApiObjects const&)            = delete;
        ApiObjects& operator=(ApiObjects const&) = delete;

        // A tensor descriptor; strides empty means packed (the library default).
        hiptensorTensorDescriptor_t tensor(std::vector<int64_t> const& lens,
                                           std::vector<int64_t> const& strides,
                                           hiptensorDataType_t         type)
        {
            hiptensorTensorDescriptor_t desc = nullptr;
            CHECK_HIPTENSOR_ERROR(
                hiptensorCreateTensorDescriptor(handle,
                                                &desc,
                                                lens.size(),
                                                lens.data(),
                                                strides.empty() ? nullptr : strides.data(),
                                                type,
                                                0));
            tensors.push_back(desc);
            return desc;
        }

        // Whether the device takes tensors of this type (f64 needs hardware support).
        bool supports(hiptensorDataType_t type)
        {
            int64_t const               len  = 1;
            hiptensorTensorDescriptor_t desc = nullptr;
            auto status = hiptensorCreateTensorDescriptor(handle, &desc, 1, &len, nullptr, type, 0);
            if(desc != nullptr)
            {
                hiptensorDestroyTensorDescriptor(desc);
            }
            return status == HIPTENSOR_STATUS_SUCCESS;
        }

        // Records an operation descriptor created by the caller (nullptr is ignored).
        hiptensorOperationDescriptor_t keep(hiptensorOperationDescriptor_t op)
        {
            if(op != nullptr)
            {
                operations.push_back(op);
            }
            return op;
        }

        // hiptensorCreatePlan with the default preference; the plan is owned when created.
        hiptensorStatus_t plan(hiptensorOperationDescriptor_t op,
                               hiptensorPlan_t&               plan,
                               uint64_t                       workspaceLimit = 0)
        {
            plan        = nullptr;
            auto status = hiptensorCreatePlan(handle, &plan, op, nullptr, workspaceLimit);
            if(plan != nullptr)
            {
                plans.push_back(plan);
            }
            return status;
        }
    };

    // Packed column-major strides (the first mode varies fastest).
    inline std::vector<int64_t> packedStrides(std::vector<int64_t> const& lens)
    {
        std::vector<int64_t> strides;
        int64_t              stride = 1;
        for(auto len : lens)
        {
            strides.push_back(stride);
            stride *= len;
        }
        return strides;
    }

    inline int64_t elementCount(std::vector<int64_t> const& lens)
    {
        int64_t count = 1;
        for(auto len : lens)
        {
            count *= len;
        }
        return count;
    }
} // namespace hiptensor_test

#endif // HIPTENSOR_TEST_API_TEST_HELPERS_HPP
