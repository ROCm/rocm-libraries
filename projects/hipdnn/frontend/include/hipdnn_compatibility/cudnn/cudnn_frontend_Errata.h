// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_Errata.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_Errata.h
 * @brief cuDNN v8 errata filters (`load_from_config`, `check_errata`) over the shim's backend C API.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_Logging.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_OperationGraph.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_Errata.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-use-nullptr,readability-container-size-empty,misc-const-correctness,bugprone-multi-level-implicit-pointer-conversion,performance-unnecessary-copy-initialization):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

// The CUDNN_ERRATA_JSON_FILE environment variable takes precedence over errata_json.
#if !defined(CUDNN_FRONTEND_SKIP_JSON_LIB) && !defined(HIPDNN_FRONTEND_SKIP_JSON_LIB)
[[maybe_unused]] inline bool load_from_config(json& json_handle, const std::string& errata_json)
{
    const char* err_json = get_environment("CUDNN_ERRATA_JSON_FILE");
    if(err_json == NULL && errata_json == "")
    {
        return false;
    }
    if(err_json == NULL)
    {
        err_json = errata_json.c_str();
    }
    std::ifstream ifs(err_json, std::ifstream::in);
    if(!ifs.is_open() || !ifs.good())
    {
        return false;
    }
    ifs >> json_handle;
    return true;
}
#endif

// Returns true when the tensor_attr tensor of op matches blocked_shape, given in shape_format
// ("NCHW" or "NHWC") order; a -1 entry is a wildcard.
inline bool check_shape(cudnnBackendDescriptor_t& op,
                        const std::string& shape_format,
                        cudnnBackendAttributeName_t tensor_attr,
                        const std::vector<int64_t>& blocked_shape)
{
    ManagedOpaqueDescriptor tensor = make_shared_backend_pointer(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    cudnnBackendDescriptor_t tensor_ = tensor->get_backend_descriptor();
    int64_t count = 0;
    cudnnStatus_t status = detail::get_attribute(
        op, tensor_attr, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, &tensor_);
    if(status != CUDNN_STATUS_SUCCESS)
    {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        throw cudnnException(
            std::string("Error getting attribute. cudnn_status: " + to_string(status)).c_str(),
            status);
#endif
    }

    std::array<int64_t, 5> tensor_dims{};
    status = detail::get_attribute(
        tensor_, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 5, &count, tensor_dims.data());
    if(status != CUDNN_STATUS_SUCCESS)
    {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        throw cudnnException(
            std::string("Error getting attribute. cudnn_status: " + to_string(status)).c_str(),
            status);
#endif
    }
    int64_t first_dim = tensor_dims[1];
    int64_t blocked_first_dim = blocked_shape[0];

    bool blocked = (blocked_first_dim != -1) ? (first_dim == blocked_first_dim) : true;

    int64_t channels = (shape_format == "NCHW") ? tensor_dims[2] : tensor_dims[4];
    int64_t blocked_channels = (shape_format == "NCHW") ? blocked_shape[1] : blocked_shape[3];
    blocked = (blocked_channels != -1) ? (blocked && channels == blocked_channels) : true;

    int64_t height = (shape_format == "NCHW") ? tensor_dims[3] : tensor_dims[2];
    int64_t blocked_height = (shape_format == "NCHW") ? blocked_shape[2] : blocked_shape[1];
    blocked = (blocked_height != -1) ? (blocked && height == blocked_height) : true;

    int64_t width = (shape_format == "NCHW") ? tensor_dims[4] : tensor_dims[3];
    int64_t blocked_width = (shape_format == "NCHW") ? blocked_shape[3] : blocked_shape[2];
    blocked = (blocked_width != -1) ? (blocked && width == blocked_width) : true;

    return blocked;
}

#if !defined(CUDNN_FRONTEND_SKIP_JSON_LIB) && !defined(HIPDNN_FRONTEND_SKIP_JSON_LIB)
template <typename T>
inline bool check_rule(const json& json_handle,
                       const std::string& executionPlanTag,
                       cudnnHandle_t handle,
                       T fn)
{
    std::string operation = json_handle["operation"];
    int64_t engine = json_handle["engine"];
    uint64_t cudnn_start = 0;
    uint64_t cudnn_end = std::numeric_limits<uint64_t>::max();
    if(json_handle.contains("cudnn_version_start"))
    {
        cudnn_start = json_handle["cudnn_version_start"];
    }
    if(json_handle.contains("cudnn_version_end"))
    {
        cudnn_end = json_handle["cudnn_version_end"];
    }
    std::string tag_prefix = operation + "_eng" + std::to_string(engine) + "_";
    std::string mod_tag = executionPlanTag + "_";
    bool blocked = tag_prefix.size() <= mod_tag.size()
                   && std::equal(tag_prefix.begin(), tag_prefix.end(), mod_tag.begin())
                   && CUDNN_VERSION >= cudnn_start && CUDNN_VERSION < cudnn_end;

    if(blocked && json_handle.contains("knob"))
    {
        for(auto& kv : json_handle["knob"])
        {
            blocked = blocked && (executionPlanTag.find(kv) != std::string::npos);
        }
    }
    blocked = blocked && fn();
    return blocked;

    CUDNN_FRONTEND_UNUSED(handle);
}

template <typename T>
inline bool check_rule(const json& json_handle,
                       const std::string& executionPlanTag,
                       cudnnHandle_t handle,
                       T fn,
                       const OperationGraph& opGraph)
{
    std::string operation = json_handle["operation"];
    int64_t engine = json_handle["engine"];
    uint64_t cudnn_start = 0;
    uint64_t cudnn_end = std::numeric_limits<uint64_t>::max();
    if(json_handle.contains("cudnn_version_start"))
    {
        cudnn_start = json_handle["cudnn_version_start"];
    }
    if(json_handle.contains("cudnn_version_end"))
    {
        cudnn_end = json_handle["cudnn_version_end"];
    }
    std::string tag_prefix = operation + "_eng" + std::to_string(engine) + "_";
    std::string mod_tag = executionPlanTag + "_";
    bool blocked = tag_prefix.size() <= mod_tag.size()
                   && std::equal(tag_prefix.begin(), tag_prefix.end(), mod_tag.begin())
                   && CUDNN_VERSION >= cudnn_start && CUDNN_VERSION < cudnn_end;

    if(blocked && json_handle.contains("knob"))
    {
        for(auto& kv : json_handle["knob"])
        {
            blocked = blocked && (executionPlanTag.find(kv) != std::string::npos);
        }
    }

    if(blocked && json_handle.contains("input_shape"))
    {
        if(!json_handle.contains("shape_format"))
        {
            std::string message = "ERROR: Please set a shape format (e.g. shape_format: \"NCWH\") "
                                  "for errata filters using input/kernel "
                                  "shape";
#ifndef NV_CUDNN_DISABLE_EXCEPTION
            throw cudnnException(message.c_str(), CUDNN_STATUS_BAD_PARAM);
#else
            CUDNN_FE_LOG(message);
            return blocked;
#endif
        }

        std::array<ManagedOpaqueDescriptor, MAX_OPGRAPH_OPS> ops = opGraph.getOps();
        std::array<cudnnBackendDescriptor_t, MAX_OPGRAPH_OPS> ops_;
        for(unsigned int i = 0; i < opGraph.getOpCount(); i++)
        {
            ops_[i] = ops[i]->get_backend_descriptor();
        }

        std::string shape_format = json_handle["shape_format"];
        std::vector<int64_t> blocked_shape = json_handle["input_shape"];

        if(operation == "ConvFwd")
        {
            blocked = blocked
                      && check_shape(ops_[0],
                                     shape_format,
                                     CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                     blocked_shape);
        }
        else if(operation == "ConvBwdFilter")
        {
            blocked = blocked
                      && check_shape(ops_[0],
                                     shape_format,
                                     CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                                     blocked_shape);
        }
        else if(operation == "ConvBwdData")
        {
            blocked = blocked
                      && check_shape(ops_[0],
                                     shape_format,
                                     CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                                     blocked_shape);
        }
    }

    if(blocked && json_handle.contains("filter_shape"))
    {
        std::array<ManagedOpaqueDescriptor, MAX_OPGRAPH_OPS> ops = opGraph.getOps();
        std::array<cudnnBackendDescriptor_t, MAX_OPGRAPH_OPS> ops_;
        for(unsigned int i = 0; i < opGraph.getOpCount(); i++)
        {
            ops_[i] = ops[i]->get_backend_descriptor();
        }

        std::vector<int64_t> blocked_shape = json_handle["filter_shape"];

        // Filters are always [K, C, H, W], hence the hard-coded "NCHW".
        if(operation == "ConvFwd")
        {
            blocked
                = blocked
                  && check_shape(
                      ops_[0], "NCHW", CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, blocked_shape);
        }
        else if(operation == "ConvBwdFilter")
        {
            blocked = blocked
                      && check_shape(ops_[0],
                                     "NCHW",
                                     CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                                     blocked_shape);
        }
        else if(operation == "ConvBwdData")
        {
            blocked
                = blocked
                  && check_shape(
                      ops_[0], "NCHW", CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, blocked_shape);
        }
    }
    blocked = blocked && fn();
    return blocked;

    CUDNN_FRONTEND_UNUSED(handle);
}

// Returns true when executionPlanTag matches a rule of json_handle, i.e. the plan is faulty.
template <typename T>
inline bool check_errata(const json& json_handle,
                         const std::string& executionPlanTag,
                         cudnnHandle_t handle,
                         T fn)
{
    CUDNN_FE_LOG_LABEL("Verifying " << executionPlanTag);
    for(const auto& rule : json_handle["rules"])
    {
        if(check_rule<T>(rule, executionPlanTag, handle, fn))
        {
            CUDNN_FE_LOG(". Blocking.");
            return true;
        }
    }

    CUDNN_FE_LOG(". Passed.");
    return false;
}

// As above, also applying the rules' input_shape / filter_shape filters to opGraph.
template <typename T>
inline bool check_errata(const json& json_handle,
                         const std::string& executionPlanTag,
                         cudnnHandle_t handle,
                         const OperationGraph& opGraph,
                         T fn)
{
    CUDNN_FE_LOG_LABEL("Verifying " << executionPlanTag);
    for(const auto& rule : json_handle["rules"])
    {
        if(check_rule<T>(rule, executionPlanTag, handle, fn, opGraph))
        {
            CUDNN_FE_LOG(". Blocking.");
            return true;
        }
    }

    CUDNN_FE_LOG(". Passed.");
    return false;
}
#endif

// NOLINTEND(readability-identifier-naming,modernize-use-nullptr,readability-container-size-empty,misc-const-correctness,bugprone-multi-level-implicit-pointer-conversion,performance-unnecessary-copy-initialization)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
