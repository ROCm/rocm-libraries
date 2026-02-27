// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file Handle.hpp
 * @brief RAII handle management for hipDNN backend
 *
 * Provides smart-pointer wrappers and factory functions for creating and
 * managing hipDNN backend handles with automatic resource cleanup.
 */

#pragma once

#include <memory>

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>

namespace hipdnn_frontend
{

/**
 * @struct HipdnnHandleDeleter
 * @brief Custom deleter for RAII management of hipDNN handles
 *
 * Destroys the backend handle and frees the pointer when the owning
 * unique_ptr goes out of scope.
 */
struct HipdnnHandleDeleter
{
    /// @brief Destroys the hipDNN handle and deletes the pointer
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

/// @brief RAII smart pointer to a hipDNN handle with automatic cleanup
using HipdnnHandlePtr = std::unique_ptr<hipdnnHandle_t, HipdnnHandleDeleter>;

/**
 * @brief Create a hipDNN handle via output parameter
 * @param handle Output smart pointer that will own the created handle
 * @param stream Optional HIP stream to associate with the handle
 * @return Error indicating success or failure
 *
 * @code{.cpp}
 * HipdnnHandlePtr handle;
 * auto err = createHipdnnHandle(handle);
 * @endcode
 */
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

/**
 * @brief Create a hipDNN handle, returning a (handle, error) pair
 * @param stream Optional HIP stream to associate with the handle
 * @return Pair of (handle, error); handle is null on failure
 *
 * @code{.cpp}
 * auto [handle, err] = createHipdnnHandle();
 * @endcode
 */
inline std::pair<HipdnnHandlePtr, Error> createHipdnnHandle(hipStream_t stream = nullptr)
{
    HipdnnHandlePtr handle;
    auto error = createHipdnnHandle(handle, stream);
    return {std::move(handle), std::move(error)};
}

/**
 * @brief Set the HIP stream on a hipDNN handle
 * @param handle The handle to configure
 * @param stream The HIP stream to associate
 * @return Error indicating success or failure
 */
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

/**
 * @brief Get the HIP stream associated with a hipDNN handle
 * @param handle The handle to query
 * @param stream Output pointer to receive the associated stream
 * @return Error indicating success or failure
 */
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

/// @brief snake_case alias for HipdnnHandleDeleter
using hipdnn_handle_deleter = HipdnnHandleDeleter;
/// @brief snake_case alias for HipdnnHandlePtr
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
