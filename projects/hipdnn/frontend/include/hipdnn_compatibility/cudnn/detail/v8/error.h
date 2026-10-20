// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>

#include <hipdnn_backend.h>

#include <hipdnn_compatibility/cudnn/cudnn_status.h>
#include <hipdnn_compatibility/cudnn/detail/status_translation.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
{

// Shim-originated failure. Carries the cuDNN status directly so the entry-point
// catch ladder never has to translate it.
class ShimError : public std::runtime_error
{
public:
    ShimError(cudnnStatus_t status, const std::string& message)
        : std::runtime_error(message)
        , _status(status)
    {
    }

    cudnnStatus_t status() const noexcept
    {
        return _status;
    }

private:
    cudnnStatus_t _status;
};

// Reads and clears the hipDNN backend's last error message.
inline std::string takeHipdnnMessage()
{
    std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> message{};
    hipdnn_frontend::detail::hipdnnBackend()->getLastErrorString(message.data(), message.size());
    return message.data();
}

// Converts a failed hipDNN call into a ShimError. The backend message is drained
// immediately because reading it clears it.
// `context` is a view so that a literal costs nothing on the success path.
inline void checkHipdnn(hipdnnStatus_t status, std::string_view context)
{
    if(status == HIPDNN_STATUS_SUCCESS)
    {
        return;
    }
    std::string message(context);
    message += ": ";
    message += takeHipdnnMessage();
    throw ShimError(toCudnnStatus(status), message);
}

inline void checkFrontend(const hipdnn_frontend::Error& error, const std::string& context)
{
    if(error.is_good())
    {
        return;
    }
    throw ShimError(toCudnnStatus(error.get_code()), context + ": " + error.get_message());
}

inline hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor
    createHipdnnDescriptor(hipdnnBackendDescriptorType_t type, const std::string& context)
{
    hipdnnBackendDescriptor_t raw = nullptr;
    checkHipdnn(hipdnn_frontend::detail::hipdnnBackend()->backendCreateDescriptor(type, &raw),
                context);
    return {raw};
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
