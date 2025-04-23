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
#ifndef ROCWMMA_COOP_LOAD_HPP
#define ROCWMMA_COOP_LOAD_HPP

#include "coop_io_bearer.hpp"
#include "layout/matrix_coop_layout_impl.hpp"
#include "opaque_load.hpp"

namespace rocwmma
{
    using MatrixLayout::MatrixCoopLayout;

    // This class wraps an incoming MatrixLayout into a cooperative one
    template <class DataLayout, class MatrixLayout, uint32_t WaveCount>
    struct CooperativeLoad : public CoopIOBearer<DataLayout,
                                                 MatrixCoopLayout<MatrixLayout, WaveCount>,
                                                 detail::OpaqueLoadBearer>
    {
    };

} // namespace rocwmma

#endif // ROCWMMA_COOP_LOAD_HPP
