// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace hipblaslt_jit::json
{
    // Ordered member names and JSON-literal values.
    using Members = std::vector<std::pair<std::string, std::string>>;

    inline std::string quote(const std::string& value)
    {
        std::ostringstream out;
        out << '"';
        for(unsigned char c : value)
        {
            if(c == '"' || c == '\\')
                out << '\\' << c;
            else if(c < 0x20)
                out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << int(c);
            else
                out << c;
        }
        out << '"';
        return out.str();
    }

    // Generated requests are compared byte for byte, so every number and boolean
    // goes through the same stream settings.
    template <typename T>
    std::string literal(const T& value)
    {
        std::ostringstream out;
        out << std::setprecision(17) << std::boolalpha << value;
        return out.str();
    }

    template <typename Values>
    std::string array(const Values& values)
    {
        std::string out = "[";
        for(const auto& value : values)
        {
            if(out.size() > 1)
                out += ',';
            out += literal(value);
        }
        return out + ']';
    }

    inline std::string object(const Members& members)
    {
        std::string out = "{";
        for(const auto& [name, value] : members)
        {
            if(out.size() > 1)
                out += ',';
            out += quote(name) + ':' + value;
        }
        return out + '}';
    }
}
