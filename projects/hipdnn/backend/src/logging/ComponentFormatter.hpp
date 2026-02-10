// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "LoggingUtils.hpp"
#include <spdlog/details/log_msg.h>
#include <spdlog/pattern_formatter.h>
#include <string>
#include <vector>

namespace hipdnn::backend::logging
{

class ComponentFormatter final : public spdlog::formatter
{
public:
    // NOLINTNEXTLINE(modernize-use-equals-default) - not trivial, has member initializer
    ComponentFormatter()
        : _callbackReceiverFormatter{
              std::make_unique<spdlog::pattern_formatter>(generateCallbackReceiverPatternString())}
    {
    }

    // NOLINTNEXTLINE(readability-convert-member-functions-to-static) - virtual override
    void format(const spdlog::details::log_msg& msg, spdlog::memory_buf_t& dest) override
    {
        // The logger "hipdnn_callback_receiver" receives pre-formatted strings from a callback sink.
        // The component name is already prepended to the message, so we use a pattern without [component].
        if(msg.logger_name == "hipdnn_callback_receiver")
        {
            _callbackReceiverFormatter->format(msg, dest);
        }
        else
        {
            auto standardFormatter = std::make_unique<spdlog::pattern_formatter>(
                generatePatternString(std::string(msg.logger_name.data(), msg.logger_name.size())));
            standardFormatter->format(msg, dest);
        }
    }

    // NOLINTNEXTLINE(readability-convert-member-functions-to-static) - virtual override
    std::unique_ptr<spdlog::formatter> clone() const override
    {
        return std::make_unique<ComponentFormatter>();
    }

private:
    std::unique_ptr<spdlog::pattern_formatter> _callbackReceiverFormatter;
};

} // namespace hipdnn::backend::logging
