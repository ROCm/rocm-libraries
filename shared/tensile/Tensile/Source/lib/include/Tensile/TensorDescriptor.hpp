/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cassert>

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include <Tensile/DataTypes.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/Macros.hpp>
#include <Tensile/Utils.hpp>

namespace Tensile
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

    /**
 * Describes a tensor including dimensions, memory layout, and data type.
 * Decoupled from any particular pointer value or memory location.
 *
 * Provides functions for indexing and otherwise iterating through a tensor.
 */
    class TENSILE_API TensorDescriptor
    {
    public:
        static const size_t UseDefaultStride;

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

        inline void calculate()
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

            if(Debug::Instance().printTensorInfo())
            {
                std::cout << "TensorDescriptor:calculate  " << *this
                          << "totalLogicalElements=" << m_totalLogicalElements
                          << " totalAllocatedElem=" << m_totalAllocatedElements << "\n";
            }
        }

        const std::vector<size_t>& sizes() const
        {
            return m_sizes;
        }
        const std::vector<size_t>& strides() const
        {
            return m_strides;
        }

        size_t offset() const
        {
            return m_offset;
        }

        bool empty() const
        {
            return m_sizes.empty();
        }

        void appendDim(size_t logicalCount);
        void appendDim(size_t logicalCount, size_t allocatedCount);

        /**
   * Returns the number of elements of padding in the given dimension (0 if
   * unpadded). May be negative if stride is less than size
   */
        int64_t dimensionPadding(size_t dim) const;

        /**
   * Collapses dimensions in the interval [begin, end).
   *
   * preconditions:
   * - end >= begin
   * - begin < dimensions()
   * - end <= dimensions()
   * - dimensions in the interval [begin, end-1) are not padded.
   *
   * postconditions:
   * - dimensions() is diminished by end-begin
   * - total elements (allocated and logical) remain the same
   * - dimension 'begin' is the product of all the dimensions in the interval
   * [begin, end).
   */
        void collapseDims(size_t begin, size_t end);

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
        size_t totalAllocatedBytes() const
        {
            return totalAllocatedElements() * elementBytes();
        }

        size_t elementBytes() const
        {
            return DataTypeInfo::Get(m_dataType).elementSize;
        }

        DataType dataType() const
        {
            return m_dataType;
        }

        template <typename Container>
        inline size_t index(Container const& indices) const
        {
            if(indices.size() != dimensions())
                throw std::runtime_error("Incorrect number of indices.");

            for(int i = 0; i < indices.size(); i++)
                if(indices[i] >= m_sizes[i])
                    throw std::runtime_error("Index out of bounds.");

            return std::inner_product(indices.begin(), indices.end(), m_strides.begin(), m_offset);
        }

        template <typename T>
        inline size_t index(std::initializer_list<T> indices) const
        {
            if(indices.size() != dimensions())
                throw std::runtime_error("Incorrect number of indices.");

            for(auto i = std::make_pair(indices.begin(), m_sizes.begin()); i.first != indices.end();
                i.first++, i.second++)
                if(*i.first >= *i.second)
                    throw std::runtime_error("Index out of bounds.");

            return std::inner_product(indices.begin(), indices.end(), m_strides.begin(), m_offset);
        }

        template <class... Ts,
                  typename = typename std::enable_if<
                      std::is_integral<typename std::common_type<Ts...>::type>::value>::type>
        inline size_t index(Ts... is) const
        {
            return this->index({is...});
        }

        inline bool incrementCoord(std::vector<size_t>& coord, size_t firstDimension = 0) const
        {
            if(coord.size() != dimensions())
                throw std::runtime_error(concatenate(
                    "Invalid coordinate size ", coord.size(), " for ", dimensions(), "-tensor"));

            if(firstDimension >= dimensions())
                return false;

            return IncrementCoord(
                coord.begin() + firstDimension, coord.end(), m_sizes.begin(), m_sizes.end());
        }

        bool operator==(const TensorDescriptor& rhs) const;
        bool operator!=(const TensorDescriptor& rhs) const;

        std::string ToString() const;

        friend std::ostream& operator<<(std::ostream& stream, const TensorDescriptor& t);

    private:
        std::vector<size_t> m_sizes;
        std::vector<size_t> m_strides;
        size_t              m_offset = 0;

        size_t m_totalLogicalElements   = 0;
        size_t m_totalAllocatedElements = 0;

        DataType m_dataType = DataType::Float;
    };

    std::ostream& operator<<(std::ostream& stream, const TensorDescriptor& t);

    template <typename T>
    void WriteTensor1D(std::ostream&           stream,
                       T*                      data,
                       TensorDescriptor const& desc,
                       bool                    decorated = true)
    {
        if(desc.dimensions() != 1)
            throw std::runtime_error(
                "WriteTensor1D is only compatible with 1-dimensional tensors.");

        if(decorated)
            stream << "[";

        if(desc.sizes()[0] > 0)
            stream << data[0];

        for(size_t i = 1; i < desc.sizes()[0]; i++)
            stream << " " << data[i];

        if(decorated)
            stream << "]" << std::endl;
    }

    /**
 *  @brief Writes a tensor to an output stream.
 *
 * \param stream The stream to write to
 * \param data Pointer to the tensor data
 * \param desc Tensor descriptor
 * \param ptrValue Pointer value to print to describe the location of the data.
 * \param decorated Print brackets [] to indicate start/end of tensor dims
 */
    template <typename T>
    void WriteTensor(std::ostream&           stream,
                     T const*                data,
                     TensorDescriptor const& desc,
                     T const*                ptrValue  = nullptr,
                     bool                    decorated = true)
    {
        stream << "Tensor(";
        streamJoin(stream, desc.sizes(), ", ");
        stream << ", data_ptr: " << data << ")" << std::endl;

        if(desc.dimensions() == 0)
            return;

        if(desc.dimensions() == 1)
        {
            WriteTensor1D(stream, data, desc, decorated);
            return;
        }

        auto const&         sizes = desc.sizes();
        std::vector<size_t> coord(desc.dimensions(), 0);
        const auto          stride0 = desc.strides()[0];

        auto upperDimCount = CoordCount(sizes.begin() + 2, sizes.end());

        for(size_t idx = 0; idx < upperDimCount; idx++)
        {
            CoordNumbered(idx, coord.begin() + 2, coord.end(), sizes.begin() + 2, sizes.end());

            coord[0] = 0;
            coord[1] = 0;

            if(decorated)
            {
                stream << "(";
                streamJoin(stream, coord, ", ");
                stream << ")" << std::endl << "[" << std::endl;
            }

            for(coord[1] = 0; coord[1] < sizes[1]; coord[1]++)
            {
                coord[0] = 0;

                auto const* localPtr = data + desc.index(coord);

                if(sizes[0] > 0)
                    stream << localPtr[0];

                for(coord[0] = 1; coord[0] < sizes[0]; coord[0]++)
                {
                    stream << " " << localPtr[coord[0] * stride0];
                }

                stream << std::endl;
            }

            if(decorated)
            {
                stream << std::endl << "]" << std::endl;
            }
        }
    }

} // namespace Tensile
