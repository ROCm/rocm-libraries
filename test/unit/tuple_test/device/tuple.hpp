/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc.
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

#ifndef ROCWMMA_DEVICE_TUPLE_TEST_HPP
#define ROCWMMA_DEVICE_TUPLE_TEST_HPP

#include <rocwmma/internal/io_traits.hpp>
#include <rocwmma/internal/tuple.hpp>
#include <rocwmma/internal/vector.hpp>
#include <rocwmma/rocwmma.hpp>

namespace rocwmma
{
    __device__ static inline bool operatorMultTest()
    {
        bool err = false;

        auto const srcTuple          = make_vector(1, 2, 3);
        auto       expectHeadElement = make_vector(2);
        auto       resultHeadElement = detail::mult_poly_vec_impl(
            make_vector(2, 2, 2), srcTuple, detail::index_sequence<0>{});
        err |= vector_reduce_and(expectHeadElement != resultHeadElement);

        auto expectTailElement = make_vector(4, 6);
        auto resultTailElement = detail::mult_poly_vec_impl(
            make_vector(2, 2, 2), srcTuple, detail::index_sequence<1, 2>{});
        err |= vector_reduce_and(expectTailElement != resultTailElement);

        auto expectAllElement = make_vector(2, 4, 6);
        auto resultAllElement = 2 * srcTuple;
        err |= vector_reduce_and(expectAllElement != resultAllElement);

        return err;
    }

    __device__ static inline bool copyTest()
    {
        bool err = false;

        auto const srcTuple = make_vector(1, 2, 3);

        auto expect = make_vector(1, 3);
        auto result = detail::copy_impl(srcTuple, detail::index_sequence<0, 2>{});
        err |= vector_reduce_and(expect != result);

        return err;
    }

    __device__ static inline bool popRightTest()
    {
        bool err = false;

        auto srcTuple = make_vector(1, 2, 3);

        auto expect = make_vector(1, 2);
        auto result = pop_back(srcTuple);
        err |= vector_reduce_and(expect != result);

        return err;
    }

    __device__ static inline bool popLeftTest()
    {
        bool err = false;

        auto srcTuple = make_vector(1, 2, 3);

        auto expect = make_vector(2, 3);
        auto result = pop_front(srcTuple);
        err |= vector_reduce_and(expect != result);

        return err;
    }

    __device__ static inline bool getFirstTest()
    {
        bool err = false;

        auto srcTuple = make_vector(1, 2, 3);

        auto expect = 1;
        auto result = get_first(srcTuple);
        err |= (expect != result);

        return err;
    }

    __device__ static inline bool getLastTest()
    {
        bool err = false;

        auto srcTuple = make_vector(1, 2, 3);

        auto expect = 3u;
        auto result = get_last(srcTuple);
        err |= (expect != result);

        return err;
    }

    __device__ static inline bool reverseTest()
    {
        bool err = false;

        auto srcTuple = make_vector(1, 2, 3);

        auto expect = make_vector(3, 2, 1);
        auto result = reverse(srcTuple);
        err |= vector_reduce_and(expect != result);

        return err;
    }

    /**
     * More Details about flatten and inflate
     * https://coderwall.com/p/fzni3g/bidirectional-translation-between-1d-and-3d-arrays
     */
    __device__ static inline bool flattenCoordRightTest()
    {
        bool err = false;

        auto srcCoord = make_vector(2, 3, 5, 7);
        auto srcDims  = make_vector(3, 5, 7, 11);

        /**
         * | c      | 2 | 3  | 5  | 7   |
         * | d      | 3 | 5  | 7  | 11  |
         * | mul    | 1 | 3  | 15 | 105 |
         * | result | 2 | 11 | 86 | 821 |
         */
        auto expect = 821;
        auto result = flatten_coord_right(srcCoord, srcDims);
        err |= (expect != result);

        return err;
    }

    __device__ static inline bool flattenCoordRightWith1DimTest()
    {
        bool err = false;

        auto srcCoord = make_vector(2);
        auto srcDims  = make_vector(3);

        /**
         * | c      | 2 |
         * | d      | 3 |
         * | mul    | 1 |
         * | result | 2 |
         */
        auto expect = 2;
        auto result = flatten_coord_right(srcCoord, srcDims);
        err |= (expect != result);

        return err;
    }

    __device__ static inline bool flattenCoordLeftTest()
    {
        bool err = false;

        auto srcCoord = make_vector(2, 3, 5, 7);
        auto srcDims  = make_vector(3, 5, 7, 11);

        /**
         * | c      | 7  | 5  | 3   | 2    |
         * | d      | 11 | 7  | 5   | 3    |
         * | mul    | 1  | 11 | 77  | 385  |
         * | result | 7  | 62 | 293 | 1063 |
         */
        auto expect = 1063;
        auto result = flatten_coord_left(srcCoord, srcDims);
        err |= (expect != result);

        return err;
    }

    __device__ static inline bool flattenCoordLeftWith1DimTest()
    {
        bool err = false;

        auto srcCoord = make_vector(7);
        auto srcDims  = make_vector(11);

        /**
         * | c      | 7  |
         * | d      | 11 |
         * | mul    | 1  |
         * | result | 7  |
         */
        auto expect = 7;
        auto result = flatten_coord_left(srcCoord, srcDims);
        err |= (expect != result);

        return err;
    }

    __device__ static inline bool inflateCoordRightTest()
    {
        bool err = false;

        auto srcFlatCoord = 821;
        auto srcDims      = make_vector(3, 5, 7, 11);

        /**
         * | c      | 821 | 821 | 821 | 821 |
         * | d      | 3   | 5   | 7   | 11  |
         * | div    | 1   | 3   | 15  | 105 |
         * | result | 2   | 3   | 5   | 7   |
         */
        auto expect = make_vector(2, 3, 5, 7);
        auto result = inflate_coord_right(srcFlatCoord, srcDims);
        err |= vector_reduce_and(expect != result);

        return err;
    }

    __device__ static inline bool inflateCoordRightWith1DimTest()
    {
        bool err = false;

        auto srcFlatCoord = 2;
        auto srcDims      = make_vector(3);

        /**
         * | c      | 2   |
         * | d      | 3   |
         * | div    | 1   |
         * | result | 2   |
         */
        auto expect = make_vector(2);
        auto result = inflate_coord_right(srcFlatCoord, srcDims);
        err |= vector_reduce_and(expect != result);

        return err;
    }

    __device__ static inline bool inflateCoordLeftTest()
    {
        bool err = false;

        auto srcFlatCoord = 1063;
        auto srcDims      = make_vector(3, 5, 7, 11);

        /**
         * | c               | 1063 | 1063 | 1063 | 1063 |
         * | d               | 11   | 7    | 5    | 3    |
         * | div             | 1    | 11   | 77   | 385  |
         * | result          | 7    | 5    | 3    | 2    |
         * | reversed result | 2    | 3    | 5    | 7    |
         */
        auto expect = make_vector(2, 3, 5, 7);
        auto result = inflate_coord_left(srcFlatCoord, srcDims);
        err |= vector_reduce_and(expect != result);

        return err;
    }

    __device__ static inline bool inflateCoordLeftWith1DimTest()
    {
        bool err = false;

        auto srcFlatCoord = 7;
        auto srcDims      = make_vector(11);

        /**
         * | c               | 7    |
         * | d               | 11   |
         * | div             | 1    |
         * | result          | 7    |
         * | reversed result | 2    |
         */
        auto expect = make_vector(7);
        auto result = inflate_coord_left(srcFlatCoord, srcDims);
        err |= vector_reduce_and(expect != result);

        return err;
    }

    __device__ static inline bool toMatrixSpaceTest()
    {
        bool err = false;

        auto srcStrides      = make_vector(2, 3, 5, 7);
        auto srcStrideCounts = make_vector(3, 5, 7, 11);

        /**
         * | stride | 2 | 3  | 5  | 7   |
         * | count  | 3 | 5  | 7  | 11  |
         * | result | 6 | 21 | 56 | 133 |
         */
        auto expect = 133;
        auto result = to_matrix_space(srcStrides, srcStrideCounts);
        err |= (expect != result);

        return err;
    }

    template <typename DataT, uint32_t VecSize>
    __global__ void tupleTest(uint32_t     m,
                              uint32_t     n,
                              DataT const* in,
                              DataT*       out,
                              uint32_t     ld,
                              DataT        param1,
                              DataT        param2)
    {
        __shared__ int32_t result;
        result = 0;
        synchronize_workgroup();

        bool err = false;

        err = err ? err : operatorMultTest();
        err = err ? err : copyTest();
        err = err ? err : popLeftTest();
        err = err ? err : popRightTest();
        err = err ? err : getFirstTest();
        err = err ? err : getLastTest();
        err = err ? err : reverseTest();
        err = err ? err : flattenCoordRightTest();
        err = err ? err : flattenCoordRightWith1DimTest();
        err = err ? err : flattenCoordLeftTest();
        err = err ? err : flattenCoordLeftWith1DimTest();
        err = err ? err : inflateCoordRightTest();
        err = err ? err : inflateCoordRightWith1DimTest();
        err = err ? err : inflateCoordLeftTest();
        err = err ? err : inflateCoordLeftWith1DimTest();
        err = err ? err : toMatrixSpaceTest();

        // Reduce error count
        atomicAdd(&result, (int32_t)err);

        // Wait for all threads
        synchronize_workgroup();

        // Just need one thread to update output
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0
           && blockIdx.y == 0 && blockIdx.z == 0)
        {
            out[0] = static_cast<DataT>(result == 0 ? SUCCESS_VALUE : ERROR_VALUE);
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_TUPLE_TEST_HPP
