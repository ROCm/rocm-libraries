// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>
#include <iostream>
#include <limits>
#include <string>

namespace
{
using hipdnn_plugin_sdk::uhd::FeatureExtractionContext;
using hipdnn_plugin_sdk::uhd::FeatureExtractor;
using hipdnn_plugin_sdk::uhd::JsonLogicError;
using hipdnn_plugin_sdk::uhd::expression::CategoricalEncoding;

CategoricalEncoding readEncoding(const nlohmann::json& request)
{
    CategoricalEncoding encoding;
    if(!request.contains("categorical_encoding"))
    {
        return encoding;
    }
    const auto& source = request.at("categorical_encoding");
    if(!source.is_object())
    {
        throw JsonLogicError("categorical_encoding must be an object");
    }
    for(const auto& field : source.items())
    {
        if(!field.value().is_object() || field.value().empty())
        {
            throw JsonLogicError("Categorical vocabulary must be a nonempty object");
        }
        for(const auto& item : field.value().items())
        {
            const auto& code = item.value();
            if(!code.is_number_integer()
               || (code.is_number_unsigned()
                   && code.get<uint64_t>()
                          > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
               || code.get<int64_t>() < std::numeric_limits<int32_t>::min()
               || code.get<int64_t>() > std::numeric_limits<int32_t>::max())
            {
                throw JsonLogicError("Categorical codes must be signed 32-bit integers");
            }
            encoding[field.key()][item.key()] = code.get<int32_t>();
        }
    }
    return encoding;
}

void bindRow(FeatureExtractionContext& context, const nlohmann::json& row)
{
    if(!row.is_object())
    {
        throw JsonLogicError("Every row must be an object of canonical bindings");
    }
    for(const auto& item : row.items())
    {
        const auto& value = item.value();
        if(value.is_null())
        {
            continue;
        } // An absent optional binding, not a numeric zero.
        if(value.is_boolean())
        {
            context.bind(item.key(), value.get<bool>());
        }
        else if(value.is_number_integer())
        {
            if(value.is_number_unsigned()
               && value.get<uint64_t>()
                      > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            {
                throw JsonLogicError("Binding exceeds signed 64-bit range: " + item.key());
            }
            context.bind(item.key(), value.get<int64_t>());
        }
        else if(value.is_number_float())
        {
            context.bind(item.key(), value.get<double>());
        }
        else if(value.is_array())
        {
            for(size_t i = 0; i < value.size(); ++i)
            {
                const auto& element = value[i];
                if(!element.is_number_integer()
                   || (element.is_number_unsigned()
                       && element.get<uint64_t>()
                              > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())))
                {
                    throw JsonLogicError("Array bindings require signed 64-bit integers: "
                                         + item.key());
                }
                context.bind(item.key() + "[" + std::to_string(i) + "]", element.get<int64_t>());
            }
        }
        else if(value.is_string())
        {
            context.bind(item.key(), value.get<std::string>());
        }
        else
        {
            throw JsonLogicError("Binding must be a scalar: " + item.key());
        }
    }
}
}

int main(int argc, char**)
{
    try
    {
        if(argc != 1)
        {
            throw JsonLogicError("Usage: hipdnn_uhd_features < request.json");
        }
        constexpr size_t MAX_REQUEST_BYTES = 256 * 1024 * 1024;
        std::string input;
        std::array<char, 65536> buffer{};
        while(std::cin.read(buffer.data(), static_cast<std::streamsize>(buffer.size()))
              || std::cin.gcount())
        {
            if(input.size() + static_cast<size_t>(std::cin.gcount()) > MAX_REQUEST_BYTES)
            {
                throw JsonLogicError("Feature request exceeds 256 MiB");
            }
            input.append(buffer.data(), static_cast<size_t>(std::cin.gcount()));
        }
        if(!std::cin.eof())
        {
            throw JsonLogicError("Failed to read feature request");
        }
        const auto request = nlohmann::json::parse(
            input, [](int depth, nlohmann::json::parse_event_t, nlohmann::json&) {
                if(depth > 140)
                {
                    throw JsonLogicError("Feature request exceeds nesting bound");
                }
                return true;
            });
        if(!request.is_object() || !request.contains("signature")
           || !request.at("signature").is_array() || !request.contains("rows")
           || !request.at("rows").is_array())
        {
            throw JsonLogicError("Request requires signature and rows arrays");
        }
        const auto signature = request.at("signature").get<std::vector<nlohmann::json>>();
        const auto encoding = readEncoding(request);
        const FeatureExtractor extractor(signature, encoding);
        nlohmann::json values = nlohmann::json::array();
        for(const auto& row : request.at("rows"))
        {
            FeatureExtractionContext context;
            bindRow(context, row);
            values.push_back(extractor.extract(context));
        }
        std::cout << nlohmann::json{{"features_hash", extractor.getSignatureHash()},
                                    {"values", std::move(values)}}
                         .dump()
                  << '\n';
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "hipdnn_uhd_features: " << error.what() << '\n';
        return 1;
    }
}
