/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "TimingInstrumentation.hpp"

namespace TensileLite
{
    namespace Client
    {
        inline constexpr char DiagnosticTag[] = "[tensilelite:diag]";

        inline thread_local std::string g_diagConfig = "(none)";
        inline thread_local std::string g_diagArch   = "(unknown)";

        class Diagnostic
        {
        public:
            enum class Severity
            {
                Fatal,
                Error,
                Warning
            };

            Diagnostic(Severity severity, std::string category)
                : m_severity(severity)
                , m_category(std::move(category))
            {
                field("config", g_diagConfig);
                field("gpu", g_diagArch);
                field("phase", g_activePhase);
            }

            template <typename T>
            Diagnostic& field(std::string key, T const& value)
            {
                std::ostringstream ss;
                ss << value;
                m_fields.emplace_back(std::move(key), ss.str());
                return *this;
            }

            Diagnostic& next(std::string const& advice)
            {
                return field("next", advice);
            }

            std::string oneLine() const
            {
                std::ostringstream ss;
                ss << DiagnosticTag << " level=" << severityName()
                   << " cat=" << logfmtValue(m_category);
                for(auto const& kv : m_fields)
                    ss << ' ' << kv.first << '=' << logfmtValue(kv.second);
                return ss.str();
            }

            std::string banner() const
            {
                const std::string bar(72, '*');
                std::size_t       keyWidth = 0;
                for(auto const& kv : m_fields)
                    keyWidth = std::max(keyWidth, kv.first.size());

                std::ostringstream ss;
                ss << bar << '\n';
                ss << "* TENSILELITE DIAGNOSTIC - " << m_category << "  [" << severityName()
                   << "]\n";
                for(auto const& kv : m_fields)
                {
                    ss << "* " << kv.first;
                    for(std::size_t i = kv.first.size(); i < keyWidth; ++i)
                        ss << ' ';
                    ss << " : " << kv.second << '\n';
                }
                ss << bar << '\n';
                return ss.str();
            }

            void emit() const
            {
                std::cerr << oneLine() << '\n' << banner() << std::flush;
            }

        private:
            const char* severityName() const
            {
                switch(m_severity)
                {
                case Severity::Fatal:
                    return "FATAL";
                case Severity::Error:
                    return "ERROR";
                case Severity::Warning:
                    return "WARNING";
                }
                return "ERROR";
            }

            static std::string logfmtValue(std::string const& value)
            {
                bool needsQuote = value.empty();
                for(char c : value)
                {
                    if(c == ' ' || c == '"' || c == '=' || c == '\n' || c == '\t' || c == '\r')
                    {
                        needsQuote = true;
                        break;
                    }
                }
                if(!needsQuote)
                    return value;

                std::ostringstream ss;
                ss << '"';
                for(char c : value)
                {
                    switch(c)
                    {
                    case '"':
                        ss << "\\\"";
                        break;
                    case '\\':
                        ss << "\\\\";
                        break;
                    case '\n':
                        ss << "\\n";
                        break;
                    case '\t':
                        ss << "\\t";
                        break;
                    case '\r':
                        ss << "\\r";
                        break;
                    default:
                        ss << c;
                        break;
                    }
                }
                ss << '"';
                return ss.str();
            }

            Severity                                         m_severity;
            std::string                                      m_category;
            std::vector<std::pair<std::string, std::string>> m_fields;
        };
    } // namespace Client
} // namespace TensileLite
