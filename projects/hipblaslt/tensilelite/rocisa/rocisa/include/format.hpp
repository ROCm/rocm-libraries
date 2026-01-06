/*******************************************************************************
 *
 * Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

#pragma once
#include <iomanip>
#include <sstream>
#include <string>

namespace rocisa
{
    // Text format functions
    inline std::string slash(const std::string& comment)
    {
        return "// " + comment + "\n";
    }

    inline std::string slash50(const std::string& comment)
    {
        std::ostringstream oss;
        oss << std::setw(50) << ""
            << " // " << comment << "\n";
        return oss.str();
    }

    inline std::string block(const std::string& comment)
    {
        return "/* " + comment + " */\n";
    }

    inline std::string blockNewLine(const std::string& comment)
    {
        return "\n/* " + comment + " */\n";
    }

    inline std::string block3Line(const std::string& comment)
    {
        std::ostringstream oss;
        oss << "\n/******************************************/\n";
        std::istringstream iss(comment);
        std::string        line;
        while(std::getline(iss, line))
        {
            oss << "/* " << std::setw(38) << std::left << line << " */\n";
        }
        oss << "/******************************************/\n";
        return oss.str();
    }

    // Format string with comment function
    inline std::string formatStr(bool               outputInlineAsm,
                                 const std::string& instStr,
                                 const std::string& comment,
                                 bool               noComment)
    {
        std::string formattedStr = instStr;
        if(outputInlineAsm)
        {
            formattedStr = "\"" + formattedStr + "\\n\\t\"";
        }
        if(!comment.empty() && !noComment)
        {
            std::string buffer = formattedStr
                                 + std::string(std::max(0, 50 - int(formattedStr.length())), ' ')
                                 + " // " + comment + "\n";
            return buffer;
        }
        else
        {
            return formattedStr + "\n";
        }
    }
} // namespace rocisa
