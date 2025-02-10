/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef ROCWMMA_IO_BEARER_HPP
#define ROCWMMA_IO_BEARER_HPP

#include "layout/layout.hpp"
#include "layout/layout_traits.hpp"
#include "tuple.hpp"
#include "utility/vector.hpp"

namespace rocwmma
{
    template <class DataLayout,
              class MatrixLayout,
              template <typename, uint32_t>
              class BearerPolicy>
    struct IOBearer
    {
    protected:
        using MatrixLayoutTraits = layout_traits<MatrixLayout>;
        using DataLayoutTraits   = layout_traits<DataLayout>;
        using DataT              = typename MatrixLayoutTraits::DataT;

        // Iterative policy traits
        using Bearer                        = BearerPolicy<DataT, MatrixLayoutTraits::VectorWidth>;
        static constexpr uint32_t ChunkSize = Bearer::size();

        // Vector type of the bearer
        template <typename DataT, uint32_t VecSize>
        using VecT =
            typename VecTraits<decay_t<typename Bearer::BufferT>>::template VecT<DataT, VecSize>;

    public:
        // Full buffer traits
        static constexpr uint32_t BufferSize
            = ChunkSize * reduce_mult(MatrixLayout::strideCounts());
        using BufferT = VecT<DataT, BufferSize>;

    protected:
        // Outer loop = index 0,
        // Inner loop = index N-1
        // Notes:
        // - Assumption: MatrixLayout provides constexpr strideCounts and strides.
        //   We can then use static unroll to eliminate looping.
        // - Unroll model will use vector_mutate_for_each because the current bearers are implemented
        //   as visitor patterns. They either write to, or read from an input vector object.
        //   The vector needs to be mutable for loads, but the stores will cast as const reference.
        //   vector_mutate_for_each provides a reference to raw vector data it is operating on, which
        //   will satisfy both loading and storing visitation needs.
        template <size_t Depth = 0, typename BuffT, typename ExternDataT>
        ROCWMMA_DEVICE static inline auto
            unroll_impl(BuffT&& buff, ExternDataT* dataPtr, uint32_t ldm)
        {
            // Get the layout strides for the current depth
            constexpr auto StrideSpace = pop_front<Depth>(MatrixLayout::strideCounts());
            constexpr auto Strides     = pop_front<Depth>(MatrixLayout::strides());

            // Ensure the buffer size is appropriate
            using BufferTraits = VecTraits<decay_t<BuffT>>;
            static_assert(BufferTraits::size() == reduce_mult(StrideSpace) * ChunkSize,
                          "Invalid buffer size");

            constexpr auto CurrentStride     = get_first(Strides);
            auto           currentDataStride = DataLayout::fromMatrixCoord(CurrentStride, ldm);

            // Last depth layer will invoke the chunk transfer
            if constexpr((VecTraits<decay_t<decltype(StrideSpace)>>::size()) == 1u)
            {
                vector_mutate_for_each<ChunkSize>(
                    forward<BuffT>(buff),
                    [](auto&& v, auto idx, auto* dataPtr, auto dataStride) {
                        uint32_t dataOffset = decay_t<decltype(idx)>::value * dataStride;
                        Bearer::exec(v, dataPtr + dataOffset);
                    },
                    dataPtr,
                    currentDataStride);
            }
            // Recurse to the next nested layer
            else
            {
                constexpr auto NextStrideSpace  = pop_front(StrideSpace);
                constexpr auto NextBufferStride = reduce_mult(NextStrideSpace) * ChunkSize;

                vector_mutate_for_each<NextBufferStride>(
                    forward<BuffT>(buff),
                    [](auto&& v, auto idx, auto* dataPtr, auto ldm, auto dataStride) {
                        uint32_t dataOffset = decay_t<decltype(idx)>::value * dataStride;
                        unroll_impl<Depth + 1u>(v, dataPtr + dataOffset, ldm);
                    },
                    dataPtr,
                    ldm,
                    currentDataStride);
            }
        }

    public:
        template <typename BuffT, typename ExternDataT>
        ROCWMMA_DEVICE static void exec(BuffT&& buff, ExternDataT* dataPtr, uint32_t ldm)
        {
            // Arrange wave threads to starting matrix layout offsets.
            auto baseOffset = MatrixLayout::baseOffset();

            // Unroll transfer in each strided dimension
            unroll_impl(
                forward<BuffT>(buff), dataPtr + DataLayout::fromMatrixCoord(baseOffset, ldm), ldm);
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_IO_BEARER_HPP
