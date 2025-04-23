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
#ifndef ROCWMMA_ACCESSORS_IMPL_HPP
#define ROCWMMA_ACCESSORS_IMPL_HPP

#include "accessors.hpp"
#include "coop_io_config.hpp"
#include "io_config.hpp"
#include "io_shape.hpp"

// Fwd decl
namespace rocwmma
{
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    class __align__(4) fragment;
}

namespace rocwmma
{
    ///
    /// DataType access
    ///

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    struct GetDataType<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>>
    {
        using type = DataT;
    };

    ///
    /// IOConfig access
    ///

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    struct GetIOConfig<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>>
    {
        using type = IOConfig<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>;
    };

    ///
    /// CoopConfig access
    ///

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t WaveCount>
    struct GetCoopIOConfig<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>, WaveCount>
    {
        using type = CoopIOConfig<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT, WaveCount>;
    };

    ///
    /// IOShape access
    ///

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    struct GetIOShape<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>>
    {
        using type = IOShape<MatrixT, BlockM, BlockN, BlockK>;
    };

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    struct GetDataLayout<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>>
    {
        using type = DataLayout::template Array1d<DataLayoutT>;
    };

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    struct GetMappingUtil<fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>>
    {
        using IOShapeT = IOShape<MatrixT, BlockM, BlockN, BlockK>;
        using type = MappingUtil<IOShapeT::BlockHeight, IOShapeT::BlockWidth, DataT, DataLayoutT>;
    };

} // namespace rocwmma

#endif // ROCWMMA_ACCESSORS_IMPL_HPP
