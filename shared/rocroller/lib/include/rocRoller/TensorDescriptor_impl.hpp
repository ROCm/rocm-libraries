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

#include <rocRoller/TensorDescriptor.hpp>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Operations/CommandArguments.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    template <typename SizeIter>
    inline size_t CoordCount(SizeIter sizeBegin, SizeIter sizeEnd)
    {
        size_t rv = 1;

        while(sizeBegin != sizeEnd)
        {
            rv *= *sizeBegin;
            sizeBegin++;
        }

        return rv;
    }

    template <typename CoordIter, typename SizeIter>
    inline void CoordNumbered(
        size_t num, CoordIter coordBegin, CoordIter coordEnd, SizeIter sizeBegin, SizeIter sizeEnd)
    {
        auto coord = coordBegin;
        auto size  = sizeBegin;

        while(coord != coordEnd && size != sizeEnd)
        {
            *coord = num % *size;
            num /= *size;

            coord++;
            size++;
        }

        if(coord != coordEnd || size != sizeEnd)
            throw std::runtime_error("Inconsistent size of coordinates.");
    }

    template <typename CoordIter, typename SizeIter>
    inline bool IncrementCoord(CoordIter coordBegin,
                               CoordIter coordEnd,
                               SizeIter  sizeBegin,
                               SizeIter  sizeEnd)
    {
        auto coord = coordBegin;
        auto size  = sizeBegin;

        while(coord != coordEnd)
        {
            (*coord)++;
            if(*coord < *size)
                return true;

            *coord = 0;

            coord++;
            size++;
        }

        return false;
    }

    inline TensorDescriptor::TensorDescriptor()
    {
        this->calculate();
    }

    template <typename IterA, typename IterB>
    TensorDescriptor::TensorDescriptor(DataType t,
                                       IterA    sizesBegin,
                                       IterA    sizesEnd,
                                       IterB    stridesBegin,
                                       IterB    stridesEnd,
                                       size_t   offset)
        : m_sizes(sizesBegin, sizesEnd)
        , m_strides(stridesBegin, stridesEnd)
        , m_dataType(t)
        , m_offset(offset)
    {
        this->calculate();
    }

    template <typename Iter>
    TensorDescriptor::TensorDescriptor(DataType t, Iter sizesBegin, Iter sizesEnd, size_t offset)
        : m_sizes(sizesBegin, sizesEnd)
        , m_dataType(t)
        , m_offset(offset)
    {
        this->calculate();
    }

    inline TensorDescriptor::TensorDescriptor(DataType                      t,
                                              size_t                        totalLogicalElements,
                                              std::initializer_list<size_t> strides,
                                              size_t                        offset)
        : m_totalLogicalElements(totalLogicalElements)
        , m_strides(strides)
        , m_dataType(t)
        , m_offset(offset)
    {
        this->calculate();
    }

    inline TensorDescriptor::TensorDescriptor(DataType                      t,
                                              std::initializer_list<size_t> sizes,
                                              size_t                        offset)
        : m_sizes(sizes)
        , m_dataType(t)
        , m_offset(offset)

    {
        this->calculate();
    }

    inline TensorDescriptor::TensorDescriptor(DataType t, std::vector<size_t> sizes, size_t offset)
        : m_sizes(std::move(sizes))
        , m_dataType(t)
        , m_offset(offset)
    {
        this->calculate();
    }

    inline TensorDescriptor::TensorDescriptor(DataType                      t,
                                              std::initializer_list<size_t> sizes,
                                              std::initializer_list<size_t> strides,
                                              size_t                        offset)
        : m_sizes(sizes)
        , m_strides(strides)
        , m_dataType(t)
        , m_offset(offset)
    {
        this->calculate();
    }

    inline TensorDescriptor::TensorDescriptor(DataType            t,
                                              std::vector<size_t> sizes,
                                              std::vector<size_t> strides,
                                              size_t              offset)
        : m_sizes(std::move(sizes))
        , m_strides(std::move(strides))
        , m_dataType(t)
        , m_offset(offset)
    {
        this->calculate();
    }

    // Specialized constructor for 2-D tensor (i.e., matrix)
    inline TensorDescriptor::TensorDescriptor(DataType              t,
                                              std::array<size_t, 2> sizes,
                                              std::string const&    transpose,
                                              size_t                offset)
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

    inline TensorDescriptor TensorDescriptor::ShuffledNoPadding(DataType            t,
                                                                std::vector<size_t> sizes,
                                                                std::vector<size_t> dimOrder)
    {
        AssertFatal(sizes.size() == dimOrder.size(), ShowValue(sizes), ShowValue(dimOrder));

        std::vector<size_t> strides(sizes.size(), 0);

        size_t stride = 1;
        for(auto idx : dimOrder)
        {
            strides.at(idx) = stride;
            stride *= sizes.at(idx);
        }

        return TensorDescriptor(t, std::move(sizes), std::move(strides));
    }

    inline TensorDescriptor TensorDescriptor::ShuffledNoPadding(DataType                      t,
                                                                std::initializer_list<size_t> sizes,
                                                                std::vector<size_t> dimOrder)
    {
        std::vector<size_t> theSizes(std::move(sizes));
        return ShuffledNoPadding(t, std::move(theSizes), std::move(dimOrder));
    }

    inline TensorDescriptor TensorDescriptor::ShuffledNoPadding(
        DataType t, std::vector<size_t> sizes, std::initializer_list<size_t> dimOrder)
    {
        std::vector<size_t> theDimOrder(std::move(dimOrder));
        return ShuffledNoPadding(t, std::move(sizes), std::move(theDimOrder));
    }

    inline TensorDescriptor TensorDescriptor::ShuffledNoPadding(
        DataType t, std::initializer_list<size_t> sizes, std::initializer_list<size_t> dimOrder)
    {
        std::vector<size_t> theSizes(std::move(sizes));
        std::vector<size_t> theDimOrder(std::move(dimOrder));

        return ShuffledNoPadding(t, std::move(theSizes), std::move(theDimOrder));
    }

    inline void TensorDescriptor::calculate()
    {
        if(m_sizes.empty())
        {
            m_strides                = m_sizes;
            m_totalLogicalElements   = 0;
            m_totalAllocatedElements = 0;
            return;
        }

        m_strides.resize(m_sizes.size(), UseDefaultStride);
        if(m_strides[0] == UseDefaultStride)
        {
            m_strides[0] = 1;
        }
        m_totalLogicalElements = m_sizes[0];

        for(int i = 1; i < m_sizes.size(); i++)
        {
            m_totalLogicalElements *= m_sizes[i];

            if(m_strides[i] == UseDefaultStride)
            {
                m_strides[i] = m_strides[i - 1] * m_sizes[i - 1];
            }
        }

        m_totalAllocatedElements = 1;
        for(int i = 0; i < m_sizes.size(); i++)
            m_totalAllocatedElements += m_strides[i] * (m_sizes[i] - 1);

        m_totalAllocatedElements += m_offset;
    }

    inline const size_t TensorDescriptor::size(size_t index) const
    {
        return m_sizes[index];
    }
    inline const std::vector<size_t>& TensorDescriptor::sizes() const
    {
        return m_sizes;
    }
    inline const size_t TensorDescriptor::stride(size_t index) const
    {
        return m_strides[index];
    }
    inline const std::vector<size_t>& TensorDescriptor::strides() const
    {
        return m_strides;
    }

    inline size_t TensorDescriptor::offset() const
    {
        return m_offset;
    }

    inline size_t TensorDescriptor::dimensions() const
    {
        return m_sizes.size();
    }
    inline size_t TensorDescriptor::totalLogicalElements() const
    {
        return m_totalLogicalElements;
    }
    inline size_t TensorDescriptor::totalAllocatedElements() const
    {
        return m_totalAllocatedElements;
    }
    inline size_t TensorDescriptor::totalAllocatedBytes() const
    {
        return totalAllocatedElements() * elementBytes();
    }
    inline size_t TensorDescriptor::elementBytes() const
    {
        return DataTypeInfo::Get(m_dataType).elementBytes;
    }

    inline DataType TensorDescriptor::dataType() const
    {
        return m_dataType;
    }

    inline bool TensorDescriptor::operator==(const TensorDescriptor& rhs) const
    {
        return m_dataType == rhs.m_dataType && m_sizes == rhs.m_sizes && m_strides == rhs.m_strides
               && m_offset == rhs.m_offset;
    }

    inline bool TensorDescriptor::operator!=(const TensorDescriptor& rhs) const
    {
        return !(*this == rhs);
    }

    inline std::string TensorDescriptor::toString() const
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

    template <typename Container>
    inline size_t TensorDescriptor::index(Container const& indices) const
    {
        if(indices.size() != dimensions())
            throw std::runtime_error("Incorrect number of indices.");

        for(int i = 0; i < indices.size(); i++)
            if(indices[i] >= m_sizes[i])
                throw std::runtime_error("Index out of bounds.");

        return std::inner_product(indices.begin(), indices.end(), m_strides.begin(), m_offset);
    }

    template <typename T>
    inline size_t TensorDescriptor::index(std::initializer_list<T> indices) const
    {
        if(indices.size() != dimensions())
            throw std::runtime_error("Incorrect number of indices.");

        for(auto i = std::make_pair(indices.begin(), m_sizes.begin()); i.first != indices.end();
            i.first++, i.second++)
            if(*i.first >= *i.second)
                throw std::runtime_error("Index out of bounds.");

        return std::inner_product(indices.begin(), indices.end(), m_strides.begin(), m_offset);
    }

    template <std::integral... Ts>
    inline size_t TensorDescriptor::index(Ts... is) const
    {
        return this->index({is...});
    }

    inline bool TensorDescriptor::incrementCoord(std::vector<size_t>& coord,
                                                 size_t               firstDimension) const
    {
        if(coord.size() != dimensions())
            throw std::runtime_error(concatenate(
                "Invalid coordinate size ", coord.size(), " for ", dimensions(), "-tensor"));

        if(firstDimension >= dimensions())
            return false;

        return IncrementCoord(
            coord.begin() + firstDimension, coord.end(), m_sizes.begin(), m_sizes.end());
    }

    inline TensorDescriptor TensorDescriptor::withNormalizedDimensions() const
    {
        auto dims = iota<size_t>(0, dimensions()).to<std::vector>();

        std::ranges::sort(dims, [this](size_t a, size_t b) { return m_strides[a] < m_strides[b]; });

        std::vector<size_t> sizes, strides;
        sizes.reserve(dimensions());
        strides.reserve(dimensions());

        for(auto dim : dims)
        {
            sizes.push_back(m_sizes[dim]);
            strides.push_back(m_strides[dim]);
        }

        return TensorDescriptor(m_dataType, std::move(sizes), std::move(strides));
    }

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

    template <typename T>
    std::string writeTensor(std::vector<T> const& data, TensorDescriptor desc)
    {
        std::string rv = desc.toString() + "\n";

        auto const& sizes = desc.sizes();
        auto        count = CoordCount(sizes.begin(), std::prev(sizes.end()));

        std::vector<size_t> prevCoord(desc.dimensions(), 0);
        for(size_t coordNum = 0; coordNum < count; coordNum++)
        {
            std::vector<size_t> coord(desc.dimensions(), 0);
            CoordNumbered(coordNum,
                          coord.begin(),
                          std::prev(coord.end()),
                          sizes.begin(),
                          std::prev(sizes.end()));

            for(coord.back() = 0; coord.back() < sizes.back(); coord.back()++)
            {
                auto idx = desc.index(coord);

                if(coord.back() > 0)
                    rv += " ";
                else
                    rv += fmt::format("{: >8}| ", idx);

                rv += fmt::format("{: >8}", data.at(idx));
            }

            rv += "\n";

            bool newDim = false;
            for(int idx = 0; idx < desc.dimensions(); idx++)
            {
                if(coord[idx] < prevCoord[idx])
                {
                    newDim = true;
                    break;
                }
            }

            if(newDim)
            {
                rv += "\n";
                bool first = true;
                for(auto dim : coord)
                {
                    if(!first)
                        rv += ", ";
                    rv += std::to_string(dim);
                    first = false;
                }
            }

            prevCoord = std::move(coord);
        }

        return rv;
    }

    template <typename T>
    inline std::vector<T> shuffleDims(std::vector<T> const&   input,
                                      TensorDescriptor const& dst,
                                      TensorDescriptor const& src)
    {
        AssertFatal(dst.dimensions() > 1, ShowValue(dst.dimensions()));
        AssertFatal(dst.sizes() == src.sizes(), ShowValue(dst.sizes()), ShowValue(src.sizes()));
        AssertFatal(dst.dataType() == src.dataType());

        auto const& sizes = dst.sizes();

        std::vector<T> rv(input.size());

        auto count = CoordCount(sizes.begin(), std::prev(sizes.end()));
#pragma omp parallel for
        for(size_t coordNum = 0; coordNum < count; coordNum++)
        {
            std::vector<size_t> coord(dst.dimensions(), 0);
            CoordNumbered(coordNum,
                          coord.begin(),
                          std::prev(coord.end()),
                          sizes.begin(),
                          std::prev(sizes.end()));

            for(coord.back() = 0; coord.back() < sizes.back(); coord.back()++)
            {
                auto dstIdx = dst.index(coord);
                auto srcIdx = src.index(coord);

                rv.at(dstIdx) = input.at(srcIdx);
            }
        }

        return rv;
    }
}
