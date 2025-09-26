// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "LoggingUtils.hpp"
#include <spdlog/details/log_msg.h>
#include <spdlog/pattern_formatter.h>
#include <string>
#include <vector>

namespace hipdnn_sdk::logging
{

class ComponentFormatter final : public spdlog::formatter
{
public:
    ComponentFormatter()
        : _passThroughFormatter{std::make_unique<spdlog::pattern_formatter>("%v")}
    {
    }

    void format(const spdlog::details::log_msg& msg, spdlog::memory_buf_t& dest) override
    {
        // The logger "hipdnn_callback_receiver" receives pre-formatted strings from a callback sink
        if(msg.logger_name == "hipdnn_callback_receiver")
        {
            _passThroughFormatter->format(msg, dest);
        }
        else
        {
            auto standardFormatter = std::make_unique<spdlog::pattern_formatter>(
                generatePatternString(std::string(msg.logger_name.data(), msg.logger_name.size())));
            standardFormatter->format(msg, dest);
        }
    }

    std::unique_ptr<spdlog::formatter> clone() const override
    {
        return std::make_unique<ComponentFormatter>();
    }

private:
    std::unique_ptr<spdlog::pattern_formatter> _passThroughFormatter;
};

} // namespace hipdnn_sdk::logging
