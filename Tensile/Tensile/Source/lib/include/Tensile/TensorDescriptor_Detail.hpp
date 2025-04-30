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

#include <Tensile/Comparison.hpp>
#include <Tensile/TensorDescriptor.hpp>

namespace Tensile
{
    template <>
    struct Comparison<TensorDescriptor>
    {
        enum
        {
            implemented = true
        };

        static int compare(TensorDescriptor const& lhs, TensorDescriptor const& rhs)
        {
            return LexicographicCompare(lhs.dataType(),
                                        rhs.dataType(),
                                        lhs.sizes(),
                                        rhs.sizes(),
                                        lhs.strides(),
                                        rhs.strides(),
                                        lhs.offset(),
                                        rhs.offset());
        }
    };

} // namespace Tensile

namespace std
{

    template <>
    struct hash<Tensile::TensorDescriptor>
    {
        inline size_t operator()(Tensile::TensorDescriptor const& tensor) const
        {
            return Tensile::combine_hashes(
                std::hash<size_t>()((size_t)tensor.dataType()),
                Tensile::hash_combine_iter(tensor.sizes().begin(), tensor.sizes().end()),
                Tensile::hash_combine_iter(tensor.strides().begin(), tensor.strides().end()),
                std::hash<size_t>()((size_t)tensor.offset()));
        }
    };
} // namespace std
