// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file Handle.hpp
 * @brief RAII handle management for hipDNN backend
 *
 * A hipDNN handle is a GPU context that holds the state needed to run
 * operations on a particular device.
 * Every Graph::build() and Graph::execute() call requires a handle.
 *
 * This file provides RAII wrappers so the handle is automatically destroyed
 * when it goes out of scope (no manual cleanup needed).
 *
 * @code{.cpp}
 * // Minimal usage
 * auto [handle, err] = hipdnn_frontend::createHipdnnHandle();
 * graph.build(*handle);
 * graph.execute(*handle, variantPack, workspace);
 * // handle is destroyed automatically at end of scope
 * @endcode
 *
 * By default the handle uses the default HIP stream. To enqueue work
 * on a stream you created with `hipStreamCreate()`, pass the stream to
 * createHipdnnHandle() or call setHipdnnHandleStream() afterwards.
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

/// @brief RAII smart pointer to a hipDNN handle; automatically calls destroy on scope exit
using HipdnnHandlePtr = std::unique_ptr<hipdnnHandle_t, HipdnnHandleDeleter>;

/**
 * @brief Create a hipDNN handle (output-parameter style)
 *
 * Initializes the backend and populates @p handle with a newly created
 * hipDNN handle. On failure, @p handle is unchanged if backend
 * creation fails, or reset to empty if stream binding fails.
 *
 * @param handle Output smart pointer that receives the created handle
 * @param stream HIP stream to bind (nullptr = default stream)
 * @return Error indicating success or failure
 *
 * @code{.cpp}
 * HipdnnHandlePtr handle;
 * auto err = createHipdnnHandle(handle);
 * if (err.is_bad()) {
 *     // handle error
 * }
 * @endcode
 *
 * @see createHipdnnHandle(hipStream_t) for structured-binding style
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
 * @brief Create a hipDNN handle (structured-binding style)
 *
 * Same as the output-parameter overload, but returns a pair so you can
 * use C++17 structured bindings.
 *
 * @param stream HIP stream to bind (nullptr = default stream)
 * @return Pair of (handle, error); handle is null on failure
 *
 * @code{.cpp}
 * auto [handle, err] = createHipdnnHandle();
 * if (err.is_bad()) {
 *     // handle error
 * }
 * @endcode
 */
inline std::pair<HipdnnHandlePtr, Error> createHipdnnHandle(hipStream_t stream = nullptr)
{
    HipdnnHandlePtr handle;
    auto error = createHipdnnHandle(handle, stream);
    return {std::move(handle), std::move(error)};
}

/**
 * @brief Bind a HIP stream to an existing handle
 *
 * All subsequent operations using this handle will be enqueued on
 * the given stream.
 *
 * @param handle The handle to reconfigure
 * @param stream The HIP stream to bind
 * @return Error indicating success or failure
 */
inline Error setHipdnnHandleStream(const HipdnnHandlePtr& handle, hipStream_t stream)
{
    if(!handle)
    {
        return {ErrorCode::INVALID_VALUE, "Cannot set stream on null handle"};
    }
    const auto status = detail::hipdnnBackend()->setStream(*handle, stream);
    HIPDNN_RETURN_ON_BACKEND_FAILURE(status, "Failed to set stream on hipdnn handle");
    return {};
}

/**
 * @brief Query which HIP stream a handle is currently bound to
 * @param handle The handle to query
 * @param stream Output pointer that receives the bound stream
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
    const auto status = detail::hipdnnBackend()->getStream(*handle, stream);
    HIPDNN_RETURN_ON_BACKEND_FAILURE(status, "Failed to get stream from hipdnn handle");
    return {};
}

/// @brief snake_case alias for HipdnnHandleDeleter
using hipdnn_handle_deleter = HipdnnHandleDeleter;
/// @brief snake_case alias for HipdnnHandlePtr
using hipdnn_handle_ptr = HipdnnHandlePtr;

/// @brief snake_case alias for createHipdnnHandle() (structured-binding style)
inline auto create_hipdnn_handle(hipStream_t stream // NOLINT(readability-identifier-naming)
                                 = nullptr)
{
    return createHipdnnHandle(stream);
}
/// @brief snake_case alias for createHipdnnHandle() (output-parameter style)
inline Error create_hipdnn_handle(HipdnnHandlePtr& handle, // NOLINT(readability-identifier-naming)
                                  hipStream_t stream = nullptr)
{
    return createHipdnnHandle(handle, stream);
}
/// @brief snake_case alias for setHipdnnHandleStream()
inline Error
    set_hipdnn_handle_stream(const HipdnnHandlePtr& h, // NOLINT(readability-identifier-naming)
                             hipStream_t s)
{
    return setHipdnnHandleStream(h, s);
}
/// @brief snake_case alias for getHipdnnHandleStream()
inline Error
    get_hipdnn_handle_stream(const HipdnnHandlePtr& h, // NOLINT(readability-identifier-naming)
                             hipStream_t* s)
{
    return getHipdnnHandleStream(h, s);
}

} // namespace hipdnn_frontend
