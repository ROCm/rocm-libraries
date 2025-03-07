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

#include <rocRoller/KernelArguments.hpp>

#include <cstddef>
#include <iomanip>

namespace rocRoller
{
    std::ostream& operator<<(std::ostream& stream, const KernelArguments& t)
    {
        size_t prevOffset = 0;
        for(auto const& argName : t.m_names)
        {
            auto const& iter = t.m_argRecords.find(argName);

            if(iter == t.m_argRecords.end())
                throw std::runtime_error("Argument " + argName + " not found in record.");

            auto const& record = iter->second;

            size_t offset = std::get<KernelArguments::ArgOffset>(record);
            size_t size   = std::get<KernelArguments::ArgSize>(record);

            if(prevOffset != offset)
            {
                stream << "[" << prevOffset << ".." << offset - 1 << "] <padding>" << std::endl;
            }

            stream << "[" << offset << ".." << offset + size - 1 << "] " << argName << ":";

            if(std::get<KernelArguments::ArgBound>(record))
            {
                auto oldFill  = stream.fill();
                auto oldWidth = stream.width();
                stream << std::hex;
                for(size_t i = offset; i < offset + size; i++)
                    stream << " " << std::setfill('0') << std::setw(2)
                           << static_cast<uint32_t>(t.m_data[i]);
                stream << std::dec;
                stream.fill(oldFill);
                stream.width(oldWidth);

                if(t.m_log)
                {
                    stream << " (" << std::get<KernelArguments::ArgString>(record) << ")";
                }
            }
            else
            {
                stream << " <unbound>";
            }

            stream << std::endl;

            prevOffset = offset + size;
        }

        return stream;
    }

    std::ostream& operator<<(std::ostream& stream, const KernelArguments::const_iterator& iter)
    {
        if(iter.isEnd())
        {
            return stream << "[end]";
        }
        else
        {
            return stream << "&(" << iter->first << ": " << iter->second << ")";
        }
    }

    KernelArguments::KernelArguments(bool log, size_t bytes)
        : m_log(log)
    {
        m_data.insert(m_data.begin(), bytes, 0);
    }

    KernelArguments::~KernelArguments() {}

    void KernelArguments::reserve(size_t bytes, size_t count)
    {
        m_data.reserve(bytes);
        m_names.reserve(count);
        if(m_log)
            m_argRecords.reserve(count);
    }

    bool KernelArguments::isFullyBound() const
    {
        if(!m_log)
            return true;

        for(auto const& record : m_argRecords)
        {
            if(!std::get<ArgBound>(record.second))
                return false;
        }

        return true;
    }

    void const* KernelArguments::data() const
    {
        if(!isFullyBound())
            throw std::runtime_error("Arguments not fully bound.");

        return reinterpret_cast<void const*>(m_data.data());
    }

    size_t KernelArguments::size() const
    {
        return m_data.size();
    }

    KernelArguments::const_iterator::const_iterator(KernelArguments const& args)
        : m_args(args)
        , m_currentArg(args.m_names.begin())
    {
        if(!args.m_log)
        {
            throw std::runtime_error("KernelArguments::const_iterator requires m_log=true");
        }
        assignCurrentArg();
    }

    KernelArguments::const_iterator::const_iterator(KernelArguments const& args,
                                                    std::string const&     argName)
        : m_args(args)
        , m_currentArg(args.m_names.begin())
    {
        if(!args.m_log)
        {
            throw std::runtime_error("KernelArguments::const_iterator requires m_log=true");
        }

        if(argName.empty())
        {
            m_currentArg = m_args.m_names.end();
        }
        else
        {
            while(m_currentArg != args.m_names.end() && *m_currentArg != argName)
            {
                ++m_currentArg;
            }
        }
        assignCurrentArg();
    }

    auto KernelArguments::const_iterator::operator++() -> const_iterator&
    {
        if(m_currentArg != m_args.m_names.end())
        {
            ++m_currentArg;
            assignCurrentArg();
        }
        return *this;
    }

    auto KernelArguments::const_iterator::operator++(int) -> const_iterator
    {
        if(m_currentArg != m_args.m_names.end())
        {
            auto ret = *this;
            ++m_currentArg;
            assignCurrentArg();
            return ret;
        }
        return *this;
    }

    bool KernelArguments::const_iterator::isEnd() const
    {
        return (*this) == m_args.end();
    }

    bool KernelArguments::const_iterator::operator==(const const_iterator& rhs) const
    {
        return m_value == rhs.m_value;
    }

    bool KernelArguments::const_iterator::operator!=(const const_iterator& rhs) const
    {
        return m_value != rhs.m_value;
    }

    auto KernelArguments::const_iterator::operator*() const
        -> decltype(const_iterator::m_value) const&
    {
        return m_value;
    }

    auto KernelArguments::const_iterator::operator->() const
        -> decltype(const_iterator::m_value) const*
    {
        return &m_value;
    }

    void KernelArguments::const_iterator::reset()
    {
        m_currentArg = m_args.m_names.begin();
        assignCurrentArg();
    }

    void KernelArguments::const_iterator::assignCurrentArg()
    {
        if(m_currentArg != m_args.m_names.end())
        {
            auto const& iter = m_args.m_argRecords.find(*m_currentArg);

            if(iter == m_args.m_argRecords.end())
            {
                throw std::runtime_error("Argument " + *m_currentArg + " not found in record.");
            }

            auto const& record = iter->second;

            if(!m_args.isFullyBound())
            {
                throw std::runtime_error("Arguments not fully bound.");
            }

            m_value = std::make_pair(
                static_cast<void const*>(m_args.m_data.data()
                                         + std::get<KernelArguments::ArgOffset>(record)),
                (size_t)std::get<KernelArguments::ArgSize>(record));
        }
        else
        {
            m_value = std::make_pair((void const*)nullptr, (size_t)0);
        }
    }

    auto KernelArguments::begin() const -> const_iterator
    {
        return const_iterator(*this);
    }

    auto KernelArguments::end() const -> const_iterator
    {
        return const_iterator(*this, "");
    }

    std::string KernelArguments::toString() const
    {
        std::ostringstream msg;
        msg << *this;
        return msg.str();
    }

}
