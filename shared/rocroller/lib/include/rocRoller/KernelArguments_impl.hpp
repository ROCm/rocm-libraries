/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2025 AMD ROCm(TM) Software
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

#include <rocRoller/KernelArguments.hpp>

#include <rocRoller/DataTypes/DataTypes_Utils.hpp>
#include <rocRoller/Operations/CommandArgument_fwd.hpp>

#include <concepts>
#include <cstring>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <rocRoller/Utilities/Logging.hpp>

namespace rocRoller
{

    struct AppendKernelArgumentsVisitor
    {
        KernelArguments*   args;
        std::string const& argName;

        template <CCommandArgumentValue T>
        void operator()(T const& value)
        {
            args->append(argName, value);
        }

        void call(CommandArgumentValue const& value)
        {
            std::visit(*this, value);
        }
    };

    template <typename T>
    inline void KernelArguments::append(std::string const& argName, T value)
    {
        if constexpr(std::is_same_v<T, CommandArgumentValue>)
        {
            AppendKernelArgumentsVisitor visitor{this, argName};
            visitor.call(value);
        }
        else
        {
            append(argName, value, true);
        }
    }

    template <typename T>
    inline void KernelArguments::appendUnbound(std::string const& argName)
    {
        append(argName, static_cast<T>(0), false);
    }

    template <typename T>
    inline void KernelArguments::bind(std::string const& argName, T value)
    {
        if(!m_log)
        {
            throw std::runtime_error("Binding is not supported without logging.");
        }

        auto it = m_argRecords.find(argName);
        if(it == m_argRecords.end())
        {
            throw std::runtime_error("Attempt to bind unknown argument " + argName);
        }

        auto& record = it->second;

        if(std::get<ArgBound>(record))
        {
            throw std::runtime_error("Attempt to bind already bound argument " + argName);
        }

        if(sizeof(T) != std::get<ArgSize>(record))
        {
            throw std::runtime_error("Size mismatch in binding argument " + argName);
        }

        size_t offset = std::get<ArgOffset>(record);

        if(offset % alignof(T) != 0)
        {
            throw std::runtime_error("Alignment error in argument " + argName + ": type mismatch?");
        }

        writeValue(offset, value);

        std::get<ArgString>(record) = stringForValue(value, true);
        std::get<ArgBound>(record)  = true;
    }

    template <typename T>
    inline std::string KernelArguments::stringForValue(T value, bool bound) const
    {
        if(!m_log)
            return "";

        if(!bound)
            return "<unbound>";

        std::ostringstream msg;

        // Some pointer types (char, int8_t, uint8_t, etc) are interpreted
        // as a string, we just want the pointer value.
        if constexpr(std::is_pointer_v<T>)
        {
            msg << static_cast<void const*>(value);
        }
        else
        {
            msg << value;
        }
        return msg.str();
    }

    template <CScaleType T>
    inline void KernelArguments::append(std::string const& argName, T value, bool bound)
    {
        auto constexpr PACKED_SCALE_ALIGN = alignof(typename PackedTypeOf<T>::type);
        auto constexpr PACKED_SCALE_SIZE  = sizeof(typename PackedTypeOf<T>::type);

        alignTo(PACKED_SCALE_ALIGN);
        size_t offset  = m_data.size();
        size_t argSize = PACKED_SCALE_SIZE;

        if(m_log)
        {
            std::string valueString = stringForValue(value, bound);
            appendRecord(argName, Arg(offset, argSize, bound, valueString));
        }

        m_data.insert(m_data.end(), argSize, 0);
        writeValue(offset, value);
    }

    template <CScaleType T>
    inline void KernelArguments::writeValue(size_t offset, T value)
    {
        auto constexpr PACKED_SCALE_SIZE = sizeof(typename PackedTypeOf<T>::type);

        size_t argSize = PACKED_SCALE_SIZE;
        if(offset + argSize > m_data.size())
        {
            throw std::runtime_error("Value exceeds allocated bounds.");
        }

        std::memset(&m_data[offset], value.scale, argSize);
    }

    template <typename T>
    inline void KernelArguments::append(std::string const& argName, T value, bool bound)
    {
        alignTo(alignof(T));

        size_t offset  = m_data.size();
        size_t argSize = sizeof(T);

        if(m_log)
        {
            std::string valueString = stringForValue(value, bound);
            appendRecord(argName, Arg(offset, argSize, bound, valueString));
        }

        m_data.insert(m_data.end(), argSize, 0);
        writeValue(offset, value);
    }

    template <typename T>
    inline void KernelArguments::writeValue(size_t offset, T value)
    {
        if(offset + sizeof(T) > m_data.size())
        {
            throw std::runtime_error("Value exceeds allocated bounds.");
        }

        std::memcpy(&m_data[offset], &value, sizeof(T));
    }

    inline void KernelArguments::alignTo(size_t alignment)
    {
        size_t extraElements = m_data.size() % alignment;
        size_t padding       = (alignment - extraElements) % alignment;

        m_data.insert(m_data.end(), padding, 0);
    }

    inline void KernelArguments::appendRecord(std::string const&   argName,
                                              KernelArguments::Arg record)
    {
        auto it = m_argRecords.find(argName);
        if(it != m_argRecords.end())
        {
            throw std::runtime_error("Duplicate argument name: " + argName);
        }

        m_argRecords[argName] = record;
        m_names.push_back(argName);
    }

    template <typename T>
    KernelArguments::const_iterator::operator T() const
    {
        if(sizeof(T) != m_value.second)
        {
            throw std::bad_cast();
        }
        return *reinterpret_cast<T*>(const_cast<void*>(m_value.first));
    }

    inline bool KernelArguments::log() const
    {
        return m_log;
    }

    inline std::vector<uint8_t> const& KernelArguments::dataVector() const
    {
        return m_data;
    }

    inline RuntimeArguments KernelArguments::runtimeArguments() const
    {
        return RuntimeArguments(m_data.data(), m_data.size());
    }
}
