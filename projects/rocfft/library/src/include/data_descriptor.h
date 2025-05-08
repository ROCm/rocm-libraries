/******************************************************************************
* Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#ifndef DATA_DESCRIPTOR_H
#define DATA_DESCRIPTOR_H

#include <regex>
#include <sstream>

namespace DescriptorFormatVersion
{
    static int UsingVersion = 0;
};

static inline std::string quote_str(const std::string& s)
{
    return "\"" + s + "\"";
};

template <typename T>
struct ToString;

template <typename T>
struct VectorToString;

template <typename T>
struct FieldDescriptor;

template <typename T>
struct VectorFieldDescriptor;

template <typename T>
struct FromString;

template <typename T>
struct StringToVector;

template <typename T>
struct FieldParser;

template <typename T>
struct VectorFieldParser;

template <typename T>
struct ToString
{
    std::string print(const T& value) const
    {
        return std::to_string(value);
    }
};

template <>
struct ToString<bool>
{
    std::string print(const bool& value) const
    {
        return value ? std::string("true") : std::string("false");
    }
};

template <>
struct ToString<std::string>
{
    std::string print(const std::string& value) const
    {
        return quote_str(value);
    }
};

template <typename T>
struct VectorToString
{
    std::string print(const std::vector<T>& vec,
                      bool                  elem_newline = false,
                      const std::string&    indent       = "") const
    {
        const char* COMMA      = ",";
        const char* LIST_DELIM = "";
        std::string list_str   = "[ ";
        for(auto i : vec)
        {
            list_str += LIST_DELIM;
            list_str += ToString<T>().print(i);
            LIST_DELIM = COMMA;

            if(elem_newline)
                list_str += "\n";
            list_str += indent;
        }
        // a trick by adding space before the ']'
        // by doing so, the ']' followed by a comma or '}' would be a individual
        // token as a hint-key of the end of a vector.
        list_str += " ]";
        return list_str;
    }
};

template <typename T>
struct FieldDescriptor
{
    std::string describe(const std::string& key, const T& value) const
    {
        return quote_str(key) + ":" + ToString<T>().print(value);
    }
};

template <typename T>
struct VectorFieldDescriptor
{
    std::string describe(const std::string&    key,
                         const std::vector<T>& vec,
                         bool                  elem_newline = false,
                         const std::string&    indent       = "") const
    {
        return quote_str(key) + ":" + VectorToString<T>().print(vec, elem_newline, indent);
    }
};

template <>
struct FromString<size_t>
{
    void Get(size_t& ret, std::sregex_token_iterator& current) const
    {
        ret = std::stoull(current->str());
    }
};

template <>
struct FromString<int>
{
    void Get(int& ret, std::sregex_token_iterator& current) const
    {
        ret = std::stoi(current->str());
    }
};

template <>
struct FromString<bool>
{
    void Get(bool& ret, std::sregex_token_iterator& current) const
    {
        ret = (current->str() == "true");
    }
};

template <>
struct FromString<std::string>
{
    void Get(std::string& ret, std::sregex_token_iterator& current) const
    {
        ret = current->str();
    }
};

template <typename T>
struct StringToVector
{
    void Get(std::vector<T>& ret, std::sregex_token_iterator& current) const
    {
        static const std::string hintKey("]");
        while(current->str() != hintKey)
        {
            T elem;
            FromString<T>().Get(elem, current);
            ret.push_back(elem);
            ++current;
        }
    }
};

template <typename T>
struct FieldParser
{
    void parse(const std::string& expectedKey, T& value, std::sregex_token_iterator& current) const
    {
        while(current->str() != expectedKey)
            ++current;

        ++current;
        FromString<T>().Get(value, current);
    }
};

template <typename T>
struct VectorFieldParser
{
    void parse(const std::string&          expectedKey,
               std::vector<T>&             vec,
               std::sregex_token_iterator& current) const
    {
        while(current->str() != expectedKey)
            ++current;

        ++current;
        vec.clear();
        StringToVector<T>().Get(vec, current);
    }
};

#endif // DATA_DESCRIPTOR_H
