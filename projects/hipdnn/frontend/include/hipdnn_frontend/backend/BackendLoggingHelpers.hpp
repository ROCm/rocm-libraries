// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <ostream>

#include <HipdnnStatus.h>
#include <hipdnn_sdk/logging/Logger.hpp>

inline const char* toString(hipdnnStatus_t status)
{
    switch(status)
    {
    case HIPDNN_STATUS_SUCCESS:
        return "HIPDNN_STATUS_SUCCESS";
    case HIPDNN_STATUS_NOT_INITIALIZED:
        return "HIPDNN_STATUS_NOT_INITIALIZED";
    case HIPDNN_STATUS_BAD_PARAM:
        return "HIPDNN_STATUS_BAD_PARAM";
    case HIPDNN_STATUS_BAD_PARAM_NULL_POINTER:
        return "HIPDNN_STATUS_BAD_PARAM_NULL_POINTER";
    case HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED:
        return "HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED";
    case HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND:
        return "HIPDNN_STATUS_BAD_PARAM_OUT_OF_BOUND";
    case HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT:
        return "HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT";
    case HIPDNN_STATUS_BAD_PARAM_STREAM_MISMATCH:
        return "HIPDNN_STATUS_BAD_PARAM_STREAM_MISMATCH";
    case HIPDNN_STATUS_NOT_SUPPORTED:
        return "HIPDNN_STATUS_NOT_SUPPORTED";
    case HIPDNN_STATUS_INTERNAL_ERROR:
        return "HIPDNN_STATUS_INTERNAL_ERROR";
    case HIPDNN_STATUS_ALLOC_FAILED:
        return "HIPDNN_STATUS_ALLOC_FAILED";
    case HIPDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED:
        return "HIPDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED";
    case HIPDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED:
        return "HIPDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED";
    case HIPDNN_STATUS_EXECUTION_FAILED:
        return "HIPDNN_STATUS_EXECUTION_FAILED";
    case HIPDNN_STATUS_PLUGIN_ERROR:
        return "HIPDNN_STATUS_PLUGIN_ERROR";
    default:
        return "HIPDNN_STATUS_UNKNOWN";
    }
}

inline std::ostream& operator<<(std::ostream& os, hipdnnStatus_t status)
{
    return os << toString(status);
}

template <>
struct fmt::formatter<hipdnnStatus_t> : fmt::formatter<const char*>
{
    template <typename FormatContext>
    auto format(hipdnnStatus_t status, FormatContext& ctx) const
    {
        return fmt::formatter<const char*>::format(toString(status), ctx);
    }
};
