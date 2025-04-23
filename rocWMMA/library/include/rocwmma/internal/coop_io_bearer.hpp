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
#ifndef ROCWMMA_COOP_IO_BEARER_HPP
#define ROCWMMA_COOP_IO_BEARER_HPP

#include "io_bearer.hpp"
#include "layout/matrix_coop_layout.hpp"

namespace rocwmma
{
    template <class DataLayout,
              class CoopMatrixLayout,
              template <typename, uint32_t>
              class BearerPolicy>
    struct CoopIOBearer : public IOBearer<DataLayout, CoopMatrixLayout, BearerPolicy>
    {
    private:
        // Traits
        using Base               = IOBearer<DataLayout, CoopMatrixLayout, BearerPolicy>;
        using MatrixLayoutTraits = layout_traits<CoopMatrixLayout>;
        using DataLayoutTraits   = layout_traits<DataLayout>;
        using DataT              = typename MatrixLayoutTraits::DataT;

        // Iterative chunk buffer for unroll decomposition
        using Bearer                        = BearerPolicy<DataT, MatrixLayoutTraits::VectorWidth>;
        static constexpr uint32_t ChunkSize = Base::ChunkSize;
        static constexpr uint32_t WaveCount = MatrixLayoutTraits::WaveCount;

        // Outer loop = index 0,
        // Inner loop = index N-1
        template <size_t Depth = 0,
                  typename Iterator,
                  typename ExternDataT,
                  typename StrideSpace,
                  typename Strides2d>
        ROCWMMA_DEVICE static inline auto unroll_impl(Iterator&     it,
                                                      ExternDataT*  dataPtr,
                                                      uint32_t      ldm,
                                                      StrideSpace&& strideCounts,
                                                      Strides2d&&   strides2d)
        {
            static_assert(VecTraits<decay_t<StrideSpace>>::size()
                              == VecTraits<decay_t<Strides2d>>::size(),
                          "Mismatched size");
            auto strideOffset = DataLayout::fromMatrixCoord(get<Depth>(strides2d), ldm);
            auto strideCount  = get<Depth>(strideCounts);

            // Last depth layer will invoke the load
            if constexpr(Depth == (VecTraits<decay_t<StrideSpace>>::size() - 1u))
            {
                for(uint32_t i = 0u; i < strideCount; i++)
                {
                    Bearer::exec(*it, dataPtr);
                    dataPtr += strideOffset;
                    it++;
                }
            }
            // Recurse to the next nested layer
            else
            {
                for(uint32_t i = 0u; i < strideCount; i++)
                {
                    unroll_impl<Depth + 1>(forward<Iterator&>(it),
                                           dataPtr,
                                           ldm,
                                           forward<StrideSpace>(strideCounts),
                                           forward<Strides2d>(strides2d));

                    dataPtr += strideOffset;
                }
            }
        }

    public:
        // Forwards to base class static unroll using static WaveCount
        template <typename BufferT, typename ExternDataT>
        ROCWMMA_DEVICE static void
            exec(BufferT&& buffer, ExternDataT* dataPtr, uint32_t ldm, uint32_t waveIndex)
        {
            // Filter out waves we don't want participating
            if(!CoopMatrixLayout::waveEnabler(waveIndex, WaveCount))
            {
                return;
            }

            // The base offset will include the wave offset
            auto baseOffset = CoopMatrixLayout::baseOffset(waveIndex, WaveCount);

            // We need to shrink the buffer for the unroll
            constexpr auto BuffSize
                = reduce_mult(CoopMatrixLayout::strideCounts(WaveCount)) * ChunkSize;
            using ReducedBufferT = conditional_t<is_const_v<BufferT>,
                                                 VecT<DataT, BuffSize> const,
                                                 VecT<DataT, BuffSize>>;

            // Base can unroll statically
            Base::unroll_impl((ReducedBufferT&)(buffer),
                              dataPtr + DataLayout::fromMatrixCoord(baseOffset, ldm),
                              ldm);
        }

        // Forwards to iterative unroll because waveCount override may not be known at compile time.
        template <typename BufferT, typename ExternDataT>
        ROCWMMA_DEVICE static void exec(BufferT&&    buffer,
                                        ExternDataT* dataPtr,
                                        uint32_t     ldm,
                                        uint32_t     waveIndex,
                                        uint32_t     waveCount)
        {
            // Filter out waves we don't want participating
            if(!CoopMatrixLayout::waveEnabler(waveIndex, waveCount))
            {
                return;
            }

            // The base offset will include the wave offset
            auto baseOffset = CoopMatrixLayout::baseOffset(waveIndex, waveCount);

            // Each basic transaction will be of ChunkSize
            auto it = makeVectorIterator<ChunkSize>(buffer).begin();

            // Alternative iterative unroll
            unroll_impl(it,
                        dataPtr + DataLayout::fromMatrixCoord(baseOffset, ldm),
                        ldm,
                        CoopMatrixLayout::strideCounts(waveCount),
                        CoopMatrixLayout::strides());
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_COOP_IO_BEARER_HPP
