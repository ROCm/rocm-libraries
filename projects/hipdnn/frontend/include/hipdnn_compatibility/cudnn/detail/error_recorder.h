// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>
#include <string>
#include <utility>

#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_helpers.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
{

class Graph; // reads the recorded error at node-build time; see Graph::sdpa

// Deferred-error latch: a setter or node the shim cannot honor records the error,
// which surfaces at validate()/build. First bad error wins; good errors are ignored.
// CRTP so recordError() returns the derived type for chaining.
template <typename Derived>
class ErrorRecorder
{
    Derived& recordError(error_t err)
    {
        if(err.is_bad())
        {
            if(!_recordedError.has_value())
            {
                CUDNN_FE_LOG_LABEL("ERROR: recording deferred error: " << err.get_message());
                _recordedError = std::move(err);
            }
            else
            {
                CUDNN_FE_LOG_LABEL("ERROR: skipping later deferred error (first error wins): "
                                   << err.get_message());
            }
        }
        return static_cast<Derived&>(*this);
    }

    Derived& recordError(error_code_t code, std::string message)
    {
        return recordError(error_t{code, std::move(message)});
    }

    bool hasRecordedError() const
    {
        return _recordedError.has_value();
    }

    // Returns the recorded error, or a default (good) error_t when none was
    // recorded. Pair with hasRecordedError() to distinguish the two.
    error_t getRecordedError() const
    {
        return _recordedError.value_or(error_t{});
    }

    std::optional<error_t> _recordedError;

private:
    friend Derived;
    friend class Graph;

    ErrorRecorder() = default;
};

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
