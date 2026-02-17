// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>

namespace hipdnn_frontend
{

struct HipdnnHandleDeleter
{
    void operator()(hipdnnHandle_t* handlePtr) const
    {
        if(handlePtr == nullptr)
        {
            return;
        }

        if(*handlePtr != nullptr)
        {
            auto status = detail::hipdnnBackend()->destroy(*handlePtr);
            if(status != HIPDNN_STATUS_SUCCESS)
            {
                HIPDNN_FE_LOG_ERROR(
                    "Failed to destroy hipdnn handle: " << static_cast<int>(status));
            }
        }

        delete handlePtr;
    }
};

// Double indirection: unique_ptr holds pointer to hipdnnHandle_t
using HipdnnHandlePtr = std::unique_ptr<hipdnnHandle_t, HipdnnHandleDeleter>;

// Output-param factory

inline Error createHipdnnHandle(HipdnnHandlePtr& handle, hipStream_t stream = nullptr)
{
    auto* handlePtr = new hipdnnHandle_t{nullptr};
    auto status = detail::hipdnnBackend()->create(handlePtr);
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        delete handlePtr;
        HIPDNN_RETURN_ON_BACKEND_FAILURE(status, "Failed to create hipdnn handle");
    }
    handle = HipdnnHandlePtr(handlePtr);

    if(stream != nullptr)
    {
        status = detail::hipdnnBackend()->setStream(*handle, stream);
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            handle.reset(); // Clear the handle on failure
            HIPDNN_RETURN_ON_BACKEND_FAILURE(status, "Failed to set stream on hipdnn handle");
        }
    }
    return {};
}

// Pair-return factory

inline std::pair<HipdnnHandlePtr, Error> createHipdnnHandle(hipStream_t stream = nullptr)
{
    HipdnnHandlePtr handle;
    auto error = createHipdnnHandle(handle, stream);
    return {std::move(handle), std::move(error)};
}

// Stream helpers

inline Error setHipdnnHandleStream(const HipdnnHandlePtr& handle, hipStream_t stream)
{
    if(!handle)
    {
        return {ErrorCode::INVALID_VALUE, "Cannot set stream on null handle"};
    }
    auto status = detail::hipdnnBackend()->setStream(*handle, stream);
    HIPDNN_RETURN_ON_BACKEND_FAILURE(status, "Failed to set stream on hipdnn handle");
    return {};
}

inline Error getHipdnnHandleStream(const HipdnnHandlePtr& handle, hipStream_t* stream)
{
    if(!handle)
    {
        return {ErrorCode::INVALID_VALUE, "Cannot get stream from null handle"};
    }
    if(stream == nullptr)
    {
        return {ErrorCode::INVALID_VALUE, "Stream output pointer is null"};
    }
    auto status = detail::hipdnnBackend()->getStream(*handle, stream);
    HIPDNN_RETURN_ON_BACKEND_FAILURE(status, "Failed to get stream from hipdnn handle");
    return {};
}

// snake_case aliases
using hipdnn_handle_deleter = HipdnnHandleDeleter;
using hipdnn_handle_ptr = HipdnnHandlePtr;

inline auto create_hipdnn_handle(hipStream_t stream // NOLINT(readability-identifier-naming)
                                 = nullptr)
{
    return createHipdnnHandle(stream);
}
inline Error create_hipdnn_handle(HipdnnHandlePtr& handle, // NOLINT(readability-identifier-naming)
                                  hipStream_t stream = nullptr)
{
    return createHipdnnHandle(handle, stream);
}
inline Error
    set_hipdnn_handle_stream(const HipdnnHandlePtr& h, // NOLINT(readability-identifier-naming)
                             hipStream_t s)
{
    return setHipdnnHandleStream(h, s);
}
inline Error
    get_hipdnn_handle_stream(const HipdnnHandlePtr& h, // NOLINT(readability-identifier-naming)
                             hipStream_t* s)
{
    return getHipdnnHandleStream(h, s);
}

} // namespace hipdnn_frontend
