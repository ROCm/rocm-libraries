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
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    /**
     * [sizeBegin,sizeEnd): A range of values representing sizes of dimensions.
     * Returns the number of coordinates within that space,
     * i.e. the product of those values.
     */
    template <typename SizeIter>
    size_t CoordCount(SizeIter sizeBegin, SizeIter sizeEnd);

    /**
     * [sizeBegin,sizeEnd): A range of values representing sizes of dimensions.
     * Writes into [coordBegin:coordEnd) the coordinates numbered `num` within
     * a linearization of all coordinates with earlier values being
     * faster-moving.
     */
    template <typename CoordIter, typename SizeIter>
    void CoordNumbered(
        size_t num, CoordIter coordBegin, CoordIter coordEnd, SizeIter sizeBegin, SizeIter sizeEnd);

    /**
     * If [coordBegin, coordEnd) represents coordinate x within the
     * linearization, updates it to represent coordinate x+1.
     * 
     * If [coordBegin, coordEnd) represents the last coordinate within the
     * linearization, it will be reset to all 0s and false will be returned.
     */
    template <typename CoordIter, typename SizeIter>
    bool IncrementCoord(CoordIter coordBegin,
                        CoordIter coordEnd,
                        SizeIter  sizeBegin,
                        SizeIter  sizeEnd);

    /*
     * Describes a tensor including dimensions, memory layout, and data type.
     * Decoupled from any particular pointer value or memory location.
     */
    class TensorDescriptor
    {
    public:
        TensorDescriptor();

        template <typename IterA, typename IterB>
        TensorDescriptor(DataType t,
                         IterA    sizesBegin,
                         IterA    sizesEnd,
                         IterB    stridesBegin,
                         IterB    stridesEnd,
                         size_t   offset = 0);

        template <typename Iter>
        TensorDescriptor(DataType t, Iter sizesBegin, Iter sizesEnd, size_t offset = 0);

        /**
         *  Allow directly specifying total number of elements instead of sizes
         */
        TensorDescriptor(DataType                      t,
                         size_t                        totalLogicalElements,
                         std::initializer_list<size_t> strides,
                         size_t                        offset = 0);

        TensorDescriptor(DataType t, std::initializer_list<size_t> sizes, size_t offset = 0);

        TensorDescriptor(DataType t, std::vector<size_t> sizes, size_t offset = 0);

        TensorDescriptor(DataType                      t,
                         std::initializer_list<size_t> sizes,
                         std::initializer_list<size_t> strides,
                         size_t                        offset = 0);

        TensorDescriptor(DataType            t,
                         std::vector<size_t> sizes,
                         std::vector<size_t> strides,
                         size_t              offset = 0);

        /**
         * Specialized constructor for 2-D tensor (i.e., matrix)
         */
        TensorDescriptor(DataType              t,
                         std::array<size_t, 2> sizes,
                         std::string const&    transpose,
                         size_t                offset = 0);

        static TensorDescriptor
            ShuffledNoPadding(DataType t, std::vector<size_t> sizes, std::vector<size_t> dimOrder);

        static TensorDescriptor ShuffledNoPadding(DataType                      t,
                                                  std::initializer_list<size_t> sizes,
                                                  std::vector<size_t>           dimOrder);

        static TensorDescriptor ShuffledNoPadding(DataType                      t,
                                                  std::vector<size_t>           sizes,
                                                  std::initializer_list<size_t> dimOrder);

        static TensorDescriptor ShuffledNoPadding(DataType                      t,
                                                  std::initializer_list<size_t> sizes,
                                                  std::initializer_list<size_t> dimOrder);

        inline void calculate();

        const size_t size(size_t index) const;

        const std::vector<size_t>& sizes() const;

        const size_t stride(size_t index) const;

        const std::vector<size_t>& strides() const;

        size_t offset() const;

        size_t dimensions() const;

        size_t totalLogicalElements() const;

        size_t totalAllocatedElements() const;

        size_t totalAllocatedBytes() const;

        size_t elementBytes() const;

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

        DataType dataType() const;

        bool operator==(const TensorDescriptor& rhs) const;

        bool operator!=(const TensorDescriptor& rhs) const;

        std::string toString() const;

        template <typename Container>
        inline size_t index(Container const& indices) const;

        template <typename T>
        inline size_t index(std::initializer_list<T> indices) const;

        template <std::integral... Ts>
        inline size_t index(Ts... is) const;

        inline bool incrementCoord(std::vector<size_t>& coord, size_t firstDimension = 0) const;

        TensorDescriptor withNormalizedDimensions() const;

        friend std::ostream& operator<<(std::ostream& stream, const TensorDescriptor& t);

        static inline const size_t UseDefaultStride = -1;

    private:
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
                                    T                                          value);

    template <typename T>
    std::string writeTensor(std::vector<T> const& data, TensorDescriptor desc);

    /**
     * `dst` and `src` must be two TensorDescriptors with the same data type
     * and dimension sizes (i.e. dst.sizes() == src.sizes()). They should have
     * different strides (or this function is a no-op).
     * 
     * `input` must contain data arranged according to `src`.
     * 
     * Returns `input` rearranged according to the strides in `dst`.
     */
    template <typename T>
    inline std::vector<T> shuffleDims(std::vector<T> const&   input,
                                      TensorDescriptor const& dst,
                                      TensorDescriptor const& src);
}

#include "TensorDescriptor_impl.hpp"
