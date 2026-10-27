// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <exception>
#include <new>
#include <string>
#include <utility>

#include <hipdnn_compatibility/cudnn/cudnn_status.h>
#include <hipdnn_compatibility/cudnn/detail/v8/error.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
{

// Result of a core call made from a public builder, getter or filter.
struct Outcome
{
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    std::string message;

    explicit operator bool() const noexcept
    {
        return status == CUDNN_STATUS_SUCCESS;
    }
};

// The only place the shim catches core exceptions. Builders and getters report the outcome through
// set_error_and_throw_exception, so callers see cudnnException (or the object status under
// NV_CUDNN_DISABLE_EXCEPTION) and never another exception type; filters answer false. Setters run
// outside it and reject bad counts the same way; only their copies of caller arrays can still throw
// (std::bad_alloc).
template <typename Fn>
Outcome guarded(Fn&& fn) noexcept
{
    try
    {
        std::forward<Fn>(fn)();
        return {};
    }
    catch(const ShimError& error)
    {
        return {error.status(), error.what()};
    }
    catch(const std::bad_alloc&)
    {
        return {CUDNN_STATUS_ALLOC_FAILED, "out of memory"};
    }
    catch(const std::exception& error)
    {
        return {CUDNN_STATUS_INTERNAL_ERROR, error.what()};
    }
    catch(...)
    {
        return {CUDNN_STATUS_INTERNAL_ERROR, "unknown exception"};
    }
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
