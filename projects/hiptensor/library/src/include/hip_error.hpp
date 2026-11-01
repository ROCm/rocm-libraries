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

#pragma once

#include <cstdio>
#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime_api.h>

#include <hiptensor/hiptensor_types.h>

#include "logger.hpp"

// A HIP failure inside the library must reach the caller as a hiptensorStatus_t. It must not
// terminate the host process, and no exception may cross the C API.
//  - Code that returns a hiptensorStatus_t checks HIP calls with HIPTENSOR_RETURN_IF_HIP_ERROR.
//  - Code that cannot return a status (e.g. the constructor of a kernel argument) throws
//    hiptensor::HipError.
//  - API entry points that can reach throwing code (ours or Composable Kernel's) catch every
//    exception and convert it with hiptensor::exceptionToStatus().
namespace hiptensor
{
    class HipError : public std::runtime_error
    {
    public:
        explicit HipError(hipError_t status)
            : std::runtime_error(std::string("HIP runtime error: ") + hipGetErrorString(status))
            , mStatus(status)
        {
        }

        hipError_t status() const noexcept
        {
            return mStatus;
        }

    private:
        hipError_t mStatus;
    };

    inline hiptensorStatus_t hipErrorToStatus(hipError_t status)
    {
        switch(status)
        {
        case hipSuccess:
            return HIPTENSOR_STATUS_SUCCESS;
        case hipErrorOutOfMemory:
            return HIPTENSOR_STATUS_ALLOC_FAILED;
        default:
            return HIPTENSOR_STATUS_HIP_ERROR;
        }
    }

    // Logs a failed HIP call and returns the matching status. The failed call has also set
    // HIP's last error; it is cleared here because the failure has been handled, and a stale
    // last error would otherwise be reported by the next kernel launch check.
    inline hiptensorStatus_t
        handleHipError(hipError_t status, char const* expression, char const* file, int line)
    {
        (void)hipGetLastError();

        char msg[512];
        snprintf(msg,
                 sizeof(msg),
                 "%s failed with '%s' (%d) at %s:%d",
                 expression,
                 hipGetErrorString(status),
                 static_cast<int>(status),
                 file,
                 line);
        Logger::instance()->logError("hipTensor", msg);

        return hipErrorToStatus(status);
    }

    // Converts the exception currently being handled into a status. Call only from a catch
    // block.
    inline hiptensorStatus_t exceptionToStatus(char const* apiFuncName) noexcept
    {
        auto        status = HIPTENSOR_STATUS_INTERNAL_ERROR;
        std::string what   = "unknown exception";
        try
        {
            throw;
        }
        catch(HipError const& e)
        {
            status = hipErrorToStatus(e.status());
            what   = e.what();
        }
        catch(std::bad_alloc const& e)
        {
            status = HIPTENSOR_STATUS_ALLOC_FAILED;
            what   = e.what();
        }
        catch(std::exception const& e)
        {
            // Composable Kernel reports a failed HIP call as a std::runtime_error whose message
            // starts with one of these prefixes and carries hipGetErrorString() of the failure.
            what = e.what();
            if(what.find(hipGetErrorString(hipErrorOutOfMemory)) != std::string::npos)
            {
                status = HIPTENSOR_STATUS_ALLOC_FAILED;
            }
            else if(what.find("HIP runtime error") != std::string::npos
                    || what.find("HIP Function Failed") != std::string::npos)
            {
                status = HIPTENSOR_STATUS_HIP_ERROR;
            }
        }
        catch(...)
        {
        }

        char msg[512];
        snprintf(msg, sizeof(msg), "%s (status %d)", what.c_str(), static_cast<int>(status));
        Logger::instance()->logError(apiFuncName, msg);

        return status;
    }

    // Owns a hipEvent_t and destroys it when it goes out of scope, so that an early return or
    // an exception does not leak it.
    struct HipEventGuard
    {
        hipEvent_t event = nullptr;

        HipEventGuard() = default;
        ~HipEventGuard()
        {
            if(event != nullptr)
            {
                (void)hipEventDestroy(event);
            }
        }

        HipEventGuard(HipEventGuard const&)            = delete;
        HipEventGuard& operator=(HipEventGuard const&) = delete;
    };

    // Owns device memory from hipMalloc and frees it when it goes out of scope, so that an
    // early return or an exception does not leak it.
    struct HipMemoryGuard
    {
        void* ptr = nullptr;

        HipMemoryGuard() = default;
        ~HipMemoryGuard()
        {
            if(ptr != nullptr)
            {
                (void)hipFree(ptr);
            }
        }

        HipMemoryGuard(HipMemoryGuard const&)            = delete;
        HipMemoryGuard& operator=(HipMemoryGuard const&) = delete;
    };
} // namespace hiptensor

// Returns the hiptensorStatus_t matching a failed HIP call from the enclosing function, which
// must return hiptensorStatus_t.
#define HIPTENSOR_RETURN_IF_HIP_ERROR(expression)                                          \
    do                                                                                     \
    {                                                                                      \
        hipError_t hipStatus_ = (expression);                                              \
        if(hipStatus_ != hipSuccess)                                                       \
        {                                                                                  \
            return hiptensor::handleHipError(hipStatus_, #expression, __FILE__, __LINE__); \
        }                                                                                  \
    } while(0)

// Returns a failed hiptensorStatus_t from the enclosing function.
#define HIPTENSOR_RETURN_IF_ERROR(expression)              \
    do                                                     \
    {                                                      \
        hiptensorStatus_t hiptensorStatus_ = (expression); \
        if(hiptensorStatus_ != HIPTENSOR_STATUS_SUCCESS)   \
        {                                                  \
            return hiptensorStatus_;                       \
        }                                                  \
    } while(0)
