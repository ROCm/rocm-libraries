/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#include <vector>
#include <thread>

#include <miopen/par_walk.hpp>

namespace miopen {

template<class InputIt1, class InputIt2, class OutputIt, class BinaryOp>
OutputIt par_transform(InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt output_first, BinaryOp op)
{
    par_walk(first1, last1, first2, output_first, [&op](auto first, auto last, auto first2, auto output_begin) {
        std::transform(first, last, first2, output_begin, op);
    });

    return output_first + (last1 - first1);
}

template<class InputIt, class OutputIt, class UnaryOp>
OutputIt par_transform(InputIt first1, InputIt last1, OutputIt d_first, UnaryOp unary_op)
{
    par_walk(first1, last1, first1, d_first, [&unary_op](auto first, auto last, auto first2, auto output_begin) {
        std::transform(first, last, output_begin, unary_op);
    });

    return first1 + (last1 - first1);
}
} // namespace miopen
