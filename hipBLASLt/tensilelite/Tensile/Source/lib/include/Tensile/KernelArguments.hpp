/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstring>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <Tensile/DataTypes.hpp>
#include <Tensile/Macros.hpp>

namespace TensileLite
{
    template <typename T>
    class KernelArgumentsContainer
    {
    public:
        void setPointer(void* pointer, size_t size)
        {
            m_data     = (T*)pointer;
            m_dataSize = size;
        }

        void reserve(size_t maxSize)
        {
            m_maxSize = maxSize;
            if(!m_data)
            {
                m_vec_data.reserve(maxSize);
            }
        }

        void insert(size_t startPos, size_t size, T value)
        {
            if(!m_data)
            {
                m_vec_data.insert(m_vec_data.end(), size, value);
                m_currentLocation = m_vec_data.size();
                return;
            }
            else if(startPos + size < m_dataSize)
            {
                // We don't insert 0 here because we'll copy data later.
                // Adding this API is to compatible with vector insert.
                // for(size_t i = startPos; i < startPos + size; i++)
                // {
                //     m_data[i] = value;
                // }
                m_currentLocation += size;
            }
        }

        size_t size() const
        {
            return m_currentLocation;
        }

        size_t end() const
        {
            return m_currentLocation;
        }

        const uint8_t* data() const
        {
            if(!m_data)
            {
                return m_vec_data.data();
            }
            return (const uint8_t*)m_data;
        }

        uint8_t* rawdata()
        {
            if(!m_data)
            {
                T* ptr = m_vec_data.data();
                return ptr;
            }
            return (uint8_t*)m_data;
        }

        const T& operator[](unsigned int i) const
        {
            if(!m_data)
            {
                return m_vec_data[i];
            }
            return m_data[i];
        }

        T& operator[](unsigned int i)
        {
            if(!m_data)
            {
                return m_vec_data[i];
            }
            return m_data[i];
        }

    private:
        size_t         m_maxSize         = 0;
        size_t         m_currentLocation = 0;
        T*             m_data            = nullptr;
        size_t         m_dataSize;
        std::vector<T> m_vec_data;
    };

    class TENSILE_API KernelArguments
    {
    public:
        KernelArguments(bool log = true);
        virtual ~KernelArguments();

        void reserve(size_t bytes, size_t count);

        void append(std::string const& name, ConstantVariant const& value, DataType type);

        void append(std::string const& name, float const value, DataType type);

        template <typename T>
        void append(std::string const& name, T value);

        template <typename T>
        void appendAligned(std::string const& name, T value);

        template <typename T>
        void appendUnbound(std::string const& name);

        template <typename T>
        void bind(std::string const& name, T value);

        bool isFullyBound() const;

        void const* data() const;
        uint8_t*    rawdata();
        size_t      size() const;

        friend std::ostream& operator<<(std::ostream& stream, const KernelArguments& t);
        friend class const_iterator;

        using ArgPair = std::pair<void const*, size_t>;
        class const_iterator
        {
        public:
            // iterator traits
            using iterator_category = std::forward_iterator_tag;
            using difference_type   = ArgPair;
            using value_type        = ArgPair;
            using pointer           = const ArgPair*;
            using reference         = const ArgPair&;

            const_iterator(KernelArguments const& args);
            const_iterator(KernelArguments const& args, std::string const& name);
            const_iterator(const const_iterator& other) = default;
            const_iterator& operator++();
            const_iterator  operator++(int);
            bool            operator==(const const_iterator& rhs) const;
            bool            operator!=(const const_iterator& rhs) const;
            ArgPair const&  operator*() const;
            ArgPair const*  operator->() const;
            void            reset();
            template <typename T>
            operator T() const;

        private:
            void assignCurrentArg();

            std::vector<std::string>::const_iterator m_currentArg;
            KernelArguments const&                   m_args;
            ArgPair                                  m_value;
        };

        const_iterator begin() const;
        const_iterator end() const;

        void useExternalPointer(void* pointer, size_t size);

    private:
        enum
        {
            ArgOffset,
            ArgSize,
            ArgBound,
            ArgString,
            NumArgFields
        };
        using Arg = std::tuple<size_t, size_t, bool, std::string>;
        static_assert(std::tuple_size<Arg>::value == NumArgFields,
                      "Enum for fields of Arg tuple doesn't match size of tuple.");

        void alignTo(size_t alignment);

        template <typename T>
        void append(std::string const& name, T value, bool bound);

        template <typename T>
        std::string stringForValue(T value, bool bound);

        void appendRecord(std::string const& name, Arg info);

        template <typename T>
        void writeValue(size_t offset, T value);

        KernelArgumentsContainer<uint8_t> m_data;

        std::vector<std::string>             m_names;
        std::unordered_map<std::string, Arg> m_argRecords;
        std::unordered_map<std::string, int> m_argNameCounter;

        bool m_log;
    };

    TENSILE_API KernelArguments::const_iterator begin(KernelArguments const&);
    TENSILE_API KernelArguments::const_iterator end(KernelArguments const&);

    inline void KernelArguments::append(std::string const&     name,
                                        ConstantVariant const& value,
                                        DataType               type)
    {
        switch(type)
        {
        case DataType::Float:
            return append<float>(name, (*std::get_if<float>(&value)), true);
        case DataType::Double:
            return append<double>(name, (*std::get_if<double>(&value)), true);
        case DataType::Half:
            return append<Half>(name, (*std::get_if<Half>(&value)), true);
        case DataType::Int32:
            return append<int32_t>(name, (*std::get_if<int32_t>(&value)), true);
        case DataType::BFloat16:
            return append<BFloat16>(name, (*std::get_if<BFloat16>(&value)), true);
        case DataType::Int8:
            return append<int8_t>(name, (*std::get_if<int8_t>(&value)), true);
        default:
            throw std::runtime_error("Unsupported ConstantVariant append type.");
        }
    }

    inline void KernelArguments::append(std::string const& name, float const value, DataType type)
    {
        switch(type)
        {
        case DataType::Float:
            return append<float>(name, value, true);
        case DataType::Double:
            return append<double>(name, (double const)value, true);
        case DataType::Half:
            return append<Half>(name, (Half const)value, true);
        case DataType::Int32:
            return append<int32_t>(name, (int32_t const)value, true);
        case DataType::BFloat16:
            return append<BFloat16>(name, (BFloat16 const)value, true);
        case DataType::Int8:
            return append<int8_t>(name, (int8_t const)value, true);
        default:
            throw std::runtime_error("Unsupported ConstantVariant append type.");
        }
    }

    template <typename T>
    inline void KernelArguments::append(std::string const& name, T value)
    {
        append(name, value, true);
    }

    template <typename T>
    inline void KernelArguments::appendUnbound(std::string const& name)
    {
        append(name, static_cast<T>(0), false);
    }

    template <typename T>
    inline void KernelArguments::bind(std::string const& name, T value)
    {
        if(!m_log)
        {
            throw std::runtime_error("Binding is not supported without logging.");
        }

        auto it = m_argRecords.find(name);
        if(it == m_argRecords.end())
        {
            throw std::runtime_error("Attempt to bind unknown argument " + name);
        }

        auto& record = it->second;

        if(std::get<ArgBound>(record))
        {
            throw std::runtime_error("Attempt to bind already bound argument " + name);
        }

        if(sizeof(T) != std::get<ArgSize>(record))
        {
            throw std::runtime_error("Size mismatch in binding argument " + name);
        }

        size_t offset = std::get<ArgOffset>(record);

        if(offset % alignof(T) != 0)
        {
            throw std::runtime_error("Alignment error in argument " + name + ": type mismatch?");
        }

        writeValue(offset, value);

        std::get<ArgString>(record) = stringForValue(value, true);
        std::get<ArgBound>(record)  = true;
    }

    template <typename T>
    inline std::string KernelArguments::stringForValue(T value, bool bound)
    {
        if(!m_log)
            return "";

        if(!bound)
            return "<unbound>";

        using castType = std::conditional_t<std::is_pointer<T>::value, void const*, T>;

        std::ostringstream msg;
        msg << static_cast<castType>(value);
        return msg.str();
    }

    template <typename T>
    void KernelArguments::appendAligned(std::string const& name, T value)
    {
        alignTo(alignof(T));
        append(name, value, true);
    }

    template <typename T>
    inline void KernelArguments::append(std::string const& name, T value, bool bound)
    {
        size_t offset = m_data.size();
        size_t size   = sizeof(T);

        if(m_log)
        {
            std::string valueString = stringForValue(value, bound);
            appendRecord(name, Arg(offset, size, bound, valueString));
        }

        m_data.insert(m_data.end(), sizeof(value), 0);
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

    inline void KernelArguments::appendRecord(std::string const& name, KernelArguments::Arg record)
    {
        auto it = m_argRecords.find(name);
        if(it != m_argRecords.end())
        {
            std::string name2   = name + "_" + std::to_string(m_argNameCounter[name]);
            m_argRecords[name2] = record;
            m_names.push_back(name2);
            m_argNameCounter[name]++;
            return;
        }
        m_argNameCounter[name] = 1;
        m_argRecords[name]     = record;
        m_names.push_back(name);
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

    class KernelArgumentsCounter
    {
    public:
        KernelArgumentsCounter() {}
        ~KernelArgumentsCounter() {}

        // Dummy function
        void reserve(size_t bytes, size_t count) {}

        inline void append(std::string const& name, ConstantVariant const& value, DataType type)
        {
            switch(type)
            {
            case DataType::Float:
                return append<float>(name, (*std::get_if<float>(&value)));
            case DataType::Double:
                return append<double>(name, (*std::get_if<double>(&value)));
            case DataType::Half:
                return append<Half>(name, (*std::get_if<Half>(&value)));
            case DataType::Int32:
                return append<int32_t>(name, (*std::get_if<int32_t>(&value)));
            case DataType::BFloat16:
                return append<BFloat16>(name, (*std::get_if<BFloat16>(&value)));
            case DataType::Int8:
                return append<int8_t>(name, (*std::get_if<int8_t>(&value)));
            default:
                throw std::runtime_error("Unsupported ConstantVariant append type.");
            }
        }

        inline void append(std::string const& name, float const value, DataType type)
        {
            switch(type)
            {
            case DataType::Float:
                return append<float>(name, value);
            case DataType::Double:
                return append<double>(name, (double const)value);
            case DataType::Half:
                return append<Half>(name, (Half const)value);
            case DataType::Int32:
                return append<int32_t>(name, (int32_t const)value);
            case DataType::BFloat16:
                return append<BFloat16>(name, (BFloat16 const)value);
            case DataType::Int8:
                return append<int8_t>(name, (int8_t const)value);
            default:
                throw std::runtime_error("Unsupported ConstantVariant append type.");
            }
        }

        template <typename T>
        inline void append(std::string const& name, T value)
        {
            counter += sizeof(value);
        }

        template <typename T>
        inline void appendUnbound(std::string const& name)
        {
            append(name, static_cast<T>(0));
        }

        const size_t size() const
        {
            return counter;
        }

    private:
        size_t counter = 0;
    };
} // namespace TensileLite
