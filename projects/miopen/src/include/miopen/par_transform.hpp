// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <vector>
#include <thread>

#include <miopen/par_walk.hpp>

namespace miopen {

template <class InputIt1, class InputIt2, class OutputIt, class BinaryOp>
OutputIt
par_transform(InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt output_first, BinaryOp op)
{
    par_walk(first1,
             last1,
             first2,
             output_first,
             [&op](auto first, auto last, auto first2, auto output_begin) {
                 std::transform(first, last, first2, output_begin, op);
             });

    return output_first + (last1 - first1);
}

template <class InputIt, class OutputIt, class UnaryOp>
OutputIt par_transform(InputIt first1, InputIt last1, OutputIt d_first, UnaryOp unary_op)
{
    par_walk(first1,
             last1,
             first1,
             d_first,
             [&unary_op](auto first, auto last, auto first2, auto output_begin) {
                 std::transform(first, last, output_begin, unary_op);
             });

    return first1 + (last1 - first1);
}
} // namespace miopen
