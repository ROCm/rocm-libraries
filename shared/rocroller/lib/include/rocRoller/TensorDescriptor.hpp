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

#include <numeric>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Operations/CommandArguments.hpp>

namespace rocRoller
{
    /*
     * Describes a tensor including dimensions, memory layout, and data type.
     * Decoupled from any particular pointer value or memory location.
     */
    class TensorDescriptor
    {
    public:
        TensorDescriptor()
        {
            this->calculate();
        }

        template <typename IterA, typename IterB>
        TensorDescriptor(DataType t,
                         IterA    sizesBegin,
                         IterA    sizesEnd,
                         IterB    stridesBegin,
                         IterB    stridesEnd,
                         size_t   offset = 0)
            : m_sizes(sizesBegin, sizesEnd)
            , m_strides(stridesBegin, stridesEnd)
            , m_dataType(t)
            , m_offset(offset)
        {
            this->calculate();
        }

        template <typename Iter>
        TensorDescriptor(DataType t, Iter sizesBegin, Iter sizesEnd, size_t offset = 0)
            : m_sizes(sizesBegin, sizesEnd)
            , m_dataType(t)
            , m_offset(offset)
        {
            this->calculate();
        }

        /*
        *  Allow directly specifying total number of elements instead of sizes
        */
        TensorDescriptor(DataType                      t,
                         size_t                        totalLogicalElements,
                         std::initializer_list<size_t> strides,
                         size_t                        offset = 0)
            : m_totalLogicalElements(totalLogicalElements)
            , m_strides(strides)
            , m_dataType(t)
            , m_offset(offset)
        {
            this->calculate();
        }

        TensorDescriptor(DataType t, std::initializer_list<size_t> sizes, size_t offset = 0)
            : m_sizes(sizes)
            , m_dataType(t)
            , m_offset(offset)

        {
            this->calculate();
        }

        TensorDescriptor(DataType                      t,
                         std::initializer_list<size_t> sizes,
                         std::initializer_list<size_t> strides,
                         size_t                        offset = 0)
            : m_sizes(sizes)
            , m_strides(strides)
            , m_dataType(t)
            , m_offset(offset)
        {
            this->calculate();
        }

        // Specialized constructor for 2-D tensor (i.e., matrix)
        TensorDescriptor(DataType              t,
                         std::array<size_t, 2> sizes,
                         std::string const&    transpose,
                         size_t                offset = 0)
            : m_sizes(sizes.begin(), sizes.end())
            , m_dataType(t)
            , m_offset(offset)
        {
            if(transpose == "T")
            {
                m_strides = {m_sizes[1], 1u};
            }
            else
            {
                m_strides = {1u, m_sizes[0]};
            }
            this->calculate();
        }

        void calculate()
        {
            if(m_strides.size() < m_sizes.size())
            {
                m_strides.resize(m_sizes.size(), UseDefaultStride);
                if(m_strides[0] == UseDefaultStride)
                {
                    m_strides[0] = 1;
                }
            }

            // Calculate total number of logical elements and update strides
            if(not m_sizes.empty())
            {
                m_totalLogicalElements = m_sizes[0];
            }
            for(int i = 1; i < m_sizes.size(); i++)
            {
                m_totalLogicalElements *= m_sizes[i];
                if(m_strides[i] == UseDefaultStride)
                {
                    m_strides[i] = m_strides[i - 1] * m_sizes[i - 1];
                }
            }

            // Calculate total number of allocated elements
            if(not m_sizes.empty())
            {
                m_totalAllocatedElements = 1;
                for(size_t i = 0; i < m_sizes.size(); i++)
                    m_totalAllocatedElements += m_strides[i] * (m_sizes[i] - 1);
            }
            else
            {
                m_totalAllocatedElements = m_totalLogicalElements;
            }
            m_totalAllocatedElements += m_offset;
        }

        const size_t size(size_t index) const
        {
            return m_sizes[index];
        }
        const std::vector<size_t>& sizes() const
        {
            return m_sizes;
        }
        const size_t stride(size_t index) const
        {
            return m_strides[index];
        }
        const std::vector<size_t>& strides() const
        {
            return m_strides;
        }

        size_t offset() const
        {
            return m_offset;
        }

        size_t dimensions() const
        {
            return m_sizes.size();
        }
        size_t totalLogicalElements() const
        {
            return m_totalLogicalElements;
        }
        size_t totalAllocatedElements() const
        {
            return m_totalAllocatedElements;
        }
        size_t elementBytes() const
        {
            return DataTypeInfo::Get(m_dataType).elementBytes;
        }

        DataType dataType() const
        {
            return m_dataType;
        }

        bool operator==(const TensorDescriptor& rhs) const
        {
            return m_dataType == rhs.m_dataType && m_sizes == rhs.m_sizes
                   && m_strides == rhs.m_strides && m_offset == rhs.m_offset;
        }

        bool operator!=(const TensorDescriptor& rhs) const
        {
            return !(*this == rhs);
        }

        std::string toString() const
        {
            std::ostringstream result;

            auto join = [&](std::vector<size_t> const& items) {
                if(items.empty())
                    return;

                result << "(";
                auto last_item = std::prev(items.end());
                for(auto iter = items.begin(); iter != last_item; iter++)
                    result << *iter << ",";
                result << *last_item << ")";
            };

            result << dimensions() << "-tensor<" << dataType() << "> ";
            join(m_sizes);
            join(m_strides);
            result << " offset: " << m_offset;
            return result.str();
        }

        friend std::ostream& operator<<(std::ostream& stream, const TensorDescriptor& t);

    private:
        static inline const size_t UseDefaultStride = -1;

        std::vector<size_t> m_sizes;
        std::vector<size_t> m_strides;
        size_t              m_offset = 0;

        size_t m_totalLogicalElements   = 0;
        size_t m_totalAllocatedElements = 0;

        DataType m_dataType = DataType::Float;
    };

    template <CCommandArgumentValue T>
    inline void setCommandTensorArg(rocRoller::CommandArguments&               commandArgs,
                                    rocRoller::Operations::OperationTag const& tag,
                                    TensorDescriptor&                          desc,
                                    T                                          value)
    {
        commandArgs.setArgument(tag, ArgumentType::Value, value);
        commandArgs.setArgument(tag, ArgumentType::Limit, desc.totalLogicalElements());

        auto const& sizes = desc.sizes();
        for(size_t i = 0; i < sizes.size(); i++)
            commandArgs.setArgument(tag, ArgumentType::Size, i, sizes[i]);

        auto const& strides = desc.strides();
        for(size_t i = 0; i < strides.size(); i++)
            commandArgs.setArgument(tag, ArgumentType::Stride, i, (size_t)strides[i]);
    }
}
