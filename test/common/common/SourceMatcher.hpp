/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#ifdef ROCROLLER_TESTS_USE_YAML_CPP
#include <yaml-cpp/yaml.h>
#endif

#include <rocRoller/Utilities/Generator.hpp>

#include <string>
#include <string_view>

#include "Utilities.hpp"

using namespace rocRoller;

/**
 * Returns true if `std::string(begin, end)` starts with `fragment`.
 */
bool startsWith(std::string const& fragment, auto begin, auto end)
{
    if((end < begin) || ((size_t)(end - begin) < fragment.size()))
        return false;

    auto fiter = fragment.begin();

    for(; (begin < end) && (fiter < fragment.end()); begin++, fiter++)
    {
        auto fragC = *fiter;
        auto testC = *begin;
        if(fragC != testC)
            return false;
    }

    return true;
}

#ifdef ROCROLLER_TESTS_USE_YAML_CPP
inline std::vector<std::string> SortedKeys(YAML::Node node)
{
    std::vector<std::string> keys;
    for(auto it = node.begin(); it != node.end(); it++)
        keys.push_back(it->first.as<std::string>());
    std::sort(keys.begin(), keys.end());

    return keys;
}

inline YAML::Node NormalizedYAMLNode(YAML::Node node)
{
    switch(node.Type())
    {
    case YAML::NodeType::Scalar:
    {
        YAML::Node rv(node.as<std::string>());
        rv.SetStyle(YAML::EmitterStyle::Block);
        return rv;
    }
    case YAML::NodeType::Map:
    {
        YAML::Node rv(YAML::NodeType::Map);
        rv.SetStyle(YAML::EmitterStyle::Block);
        std::vector<std::string> keys = SortedKeys(node);

        for(auto const& k : keys)
            rv[k] = NormalizedYAMLNode(node[k]);
        return rv;
    }
    case YAML::NodeType::Sequence:
    {
        YAML::Node rv(YAML::NodeType::Sequence);
        rv.SetStyle(YAML::EmitterStyle::Block);
        for(auto it = node.begin(); it != node.end(); it++)
        {
            rv.push_back(NormalizedYAMLNode(*it));
        }
        return rv;
    }
    default:
        break;
    }
    return YAML::Node();
}
#endif

inline std::string NormalizedYAML(std::string doc)
{
#ifdef ROCROLLER_TESTS_USE_YAML_CPP
    auto tmp  = YAML::Load(doc);
    auto node = NormalizedYAMLNode(tmp);

    YAML::Emitter emitter;
    emitter << YAML::BeginDoc;
    emitter << node;
    emitter << YAML::EndDoc;
    if(!emitter.good())
        throw std::runtime_error(emitter.GetLastError());
    return emitter.c_str();
#else
    // Normalization not implemented
    return doc;
#endif
}

/**
 * Yields lines from `input` in a way that attempts to normalize spacing
 * and optionally removes comments.  Over time this should be improved to match
 * the assembler's rules.
 *
 * Currently:
 *
 * - Removes white space at the beginning or end of a line
 * - Multiple consecutive spaces are reduced to a single space (but not
 *  within string constants).
 * - Blank lines are removed.
 * - Comments are removed if `includeComments` is false, otherwise they are
 *   subject to these same normalization rules.
 * - Embedded YAML documents are converted entirely to block syntax.
 *
 * TODO:
 *  - Maybe have a separate function that ONLY returns comments?
 *
 */
inline Generator<std::string> NormalizedSourceLines(std::string input, bool includeComments)
{
    std::string curPart;
    bool        anyContent = false;
    bool        anySpace   = false;

    for(auto pos = input.begin(); pos != input.end(); pos++)
    {
        auto ch = *pos;

        if(ch == '\n')
        {
            if(anyContent)
                co_yield curPart;

            curPart    = "";
            anyContent = false;
            anySpace   = false;

            continue;
        }

        if(std::isspace(ch))
        {
            if(anyContent)
                anySpace = true;

            continue;
        }

        if(ch == '"' || ch == '\'')
        {
            if(anySpace)
                curPart += ' ';
            auto curQuote = ch;
            curPart += ch;
            pos++;
            bool escape = false;

            // Copy the string constant verbatim
            for(; pos != input.end(); pos++)
            {
                ch = *pos;
                if(ch == '\\')
                    escape = true;
                else if(!escape && ch == curQuote)
                {
                    curPart += ch;
                    break;
                }
                else
                {
                    escape = false;
                }

                curPart += ch;
            }

            anyContent = true;
            anySpace   = false;

            if(pos == input.end())
                break;

            continue;
        }

        if(!includeComments && startsWith("//", pos, input.end()))
        {
            // Consume until the end of the line.
            // Note that we want to keep the newline so that this line is
            // processed correctly.
            for(; pos + 1 != input.end() && *(pos + 1) != '\n'; pos++)
            {
                // Empty loop
            }
            continue;
        }

        if(!includeComments && startsWith("/*", pos, input.end()))
        {
            // Consume until we see '*/'.
            pos += 2;
            for(; pos != input.end() && !startsWith("*/", pos, input.end()); pos++)
            {
                //Empty loop
            }
            // If this is not checked, we could double increment and go past the end.
            if(pos == input.end())
                break;

            pos += 2;
            continue;
        }

        // YAML document.
        if(!anySpace && !anyContent && startsWith("---", pos, input.end()))
        {
            std::string document;
            std::string terminator = "\n...";
            for(; pos != input.end() && !startsWith(terminator, pos, input.end()); pos++)
            {
                document += *pos;
            }

            if(pos != input.end())
            {
                document += terminator;
                pos += terminator.size();
            }

            co_yield_(NormalizedYAML(document));
            curPart    = "";
            anyContent = false;
            anySpace   = false;
            if(pos == input.end())
                break;

            continue;
        }

        if(anySpace)
            curPart += ' ';

        curPart += ch;
        anySpace   = false;
        anyContent = true;
    }

    if(anyContent)
        co_yield curPart;

    co_return;
}

inline std::string NormalizedSource(std::string const& input, bool includeComments = false)
{
    std::string rv;
    for(auto const& line : NormalizedSourceLines(input, includeComments))
    {
        rv += line;
        rv += "\n";
    }

    return rv;
}
