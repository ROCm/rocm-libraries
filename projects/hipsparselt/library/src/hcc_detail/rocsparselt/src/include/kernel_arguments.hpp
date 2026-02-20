/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include "hipsparselt_ostream.hpp"
#include <cstring>
#include <hip/hip_runtime_api.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
class KernelArguments
{
public:
    KernelArguments(bool log = true);
    virtual ~KernelArguments();

    void reserve(size_t bytes, size_t count);

    template <typename T>
    void append(std::string const& name, T value);

    template <typename T>
    void appendUnbound(std::string const& name);

    template <typename T>
    void bind(std::string const& name, T value);

    bool isFullyBound() const;

    void const* data() const;
    size_t      size() const;

    friend std::ostream& operator<<(std::ostream& stream, const KernelArguments& t);
    friend class const_iterator;

    using ArgPair = std::pair<void const*, size_t>;
    class const_iterator : public std::iterator<std::input_iterator_tag, ArgPair>
    {
    public:
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

    std::vector<uint8_t> m_data;

    std::vector<std::string>             m_names;
    std::unordered_map<std::string, Arg> m_argRecords;

    bool m_log;
};

KernelArguments::const_iterator begin(KernelArguments const&);
KernelArguments::const_iterator end(KernelArguments const&);

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

    hipsparselt_internal_ostream msg;
    msg << static_cast<castType>(value);
    return msg.str();
}

template <typename T>
inline void KernelArguments::append(std::string const& name, T value, bool bound)
{
    alignTo(alignof(T));

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
        throw std::runtime_error("Duplicate argument name: " + name);
    }

    m_argRecords[name] = record;
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

/**
 * \ingroup Launching
 * Describes a single kernel invocation including kernel name, launch
 * bounds, and arguments.
 */
struct KernelInvocation
{
public:
    std::string kernelName;

    dim3   workGroupSize;
    dim3   numWorkGroups;
    dim3   numWorkItems;
    size_t sharedMemBytes = 0;

    KernelArguments args;
};

struct KernelParams
{
    char         SolutionNameMin[256];
    int          DataType;
    int          DestDataType;
    int          ComputeDataType;
    bool         TransposeA;
    bool         TransposeB;
    unsigned int WorkGroup[3];
    unsigned int ThreadTile[3];
    unsigned int MacroTile[3];
    size_t       StaggerU;
    size_t       DepthU;
    size_t       GlobalSplitU;
    size_t       StaggerStrideShift;
    int          WorkGroupMapping;
    size_t       PackBatchDims;
    bool         UseInitialStridesAB;
    bool         UseInitialStridesCD;
    bool         ActivationFused;
    int          GlobalAccumulation;
    bool         Activation;
    bool         ActivationHPA;
    char         ActivationType[32];
};
