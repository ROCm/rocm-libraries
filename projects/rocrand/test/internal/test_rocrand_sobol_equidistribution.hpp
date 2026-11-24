// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCRAND_TEST_INTERNAL_TEST_ROCRAND_SOBOL_EQUIDISTRIBUTION_HPP_
#define ROCRAND_TEST_INTERNAL_TEST_ROCRAND_SOBOL_EQUIDISTRIBUTION_HPP_

// Equidistribution checks for the Sobol generators.
//
// A digital sequence in base 2 emits x(i) = C * gray(i), where gray(i) = i ^ (i >> 1) and C is
// the dimension's generator matrix over GF(2). Column j of C is the direction vector v[j], and
// row r of C selects output bit (bits - 1 - r), so the most significant output bit is row 0.
// The scrambled generators additionally XOR a per-dimension constant K (a digital shift), giving
// x(i) = C * gray(i) ^ K.
//
// A dimension is equidistributed -- it is a (0, 1)-sequence, so the first 2^m values cover every
// m-bit prefix exactly once, for every m -- precisely when every leading m x m submatrix of C is
// nonsingular. That is the property these checks target, because it is the property a scramble
// has to preserve. A Matousek linear matrix scramble C -> L * C with L unit lower triangular
// preserves it, since (L * C)[0:m, 0:m] = L[0:m, 0:m] * C[0:m, 0:m].
//
// Everything here is derived from values the generator actually produces, not from the
// precomputed tables, so the whole path (table, engine, kernel) is under test.

#include "test_common.hpp"
#include "test_rocrand_common.hpp"

#include <rocrand/rocrand.h>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <vector>

namespace rocrand_test
{
namespace sobol
{

/// A bits x bits matrix over GF(2), held as one bitmask per row. Entry (r, j) is bit
/// (bits - 1 - j) of row r, so that rows and direction vectors share the same
/// most-significant-bit-first convention.
template<class T>
struct gf2_matrix
{
    static constexpr unsigned int bits = sizeof(T) * 8;

    std::array<T, bits> rows{};

    static constexpr T bit(const unsigned int index)
    {
        return static_cast<T>(T{1} << (bits - 1 - index));
    }
};

/// Transposes one dimension's `bits` direction vectors into a generator matrix.
template<class T>
gf2_matrix<T> generator_matrix(const T* direction_vectors)
{
    gf2_matrix<T> matrix;
    for(unsigned int j = 0; j < gf2_matrix<T>::bits; ++j)
    {
        const T column = gf2_matrix<T>::bit(j);
        // Walk the direction vector from its most significant bit down; everything below its
        // lowest set bit is zero, so the loop can stop as soon as the remainder is empty.
        T value = direction_vectors[j];
        for(unsigned int r = 0; value != 0; ++r, value = static_cast<T>(value << 1))
        {
            if((value >> (gf2_matrix<T>::bits - 1)) & 1)
            {
                matrix.rows[r] |= column;
            }
        }
    }
    return matrix;
}

/// Runs Gaussian elimination without row swaps, which succeeds exactly when every leading
/// principal minor is nonsingular. Returns 0 when they all are, otherwise the smallest m whose
/// leading m x m submatrix is singular.
template<class T>
unsigned int first_singular_leading_minor(gf2_matrix<T> matrix)
{
    for(unsigned int p = 0; p < gf2_matrix<T>::bits; ++p)
    {
        const T pivot = gf2_matrix<T>::bit(p);
        if((matrix.rows[p] & pivot) == 0)
        {
            return p + 1;
        }
        for(unsigned int r = p + 1; r < gf2_matrix<T>::bits; ++r)
        {
            if(matrix.rows[r] & pivot)
            {
                matrix.rows[r] ^= matrix.rows[p];
            }
        }
    }
    return 0;
}

template<class T>
gf2_matrix<T> multiply(const gf2_matrix<T>& a, const gf2_matrix<T>& b)
{
    gf2_matrix<T> result;
    for(unsigned int r = 0; r < gf2_matrix<T>::bits; ++r)
    {
        T accumulator{};
        for(unsigned int k = 0; k < gf2_matrix<T>::bits; ++k)
        {
            if(a.rows[r] & gf2_matrix<T>::bit(k))
            {
                accumulator ^= b.rows[k];
            }
        }
        result.rows[r] = accumulator;
    }
    return result;
}

/// Gauss-Jordan inversion. Returns false when the matrix is singular.
template<class T>
bool invert(gf2_matrix<T> matrix, gf2_matrix<T>& inverse)
{
    gf2_matrix<T> identity;
    for(unsigned int r = 0; r < gf2_matrix<T>::bits; ++r)
    {
        identity.rows[r] = gf2_matrix<T>::bit(r);
    }

    for(unsigned int p = 0; p < gf2_matrix<T>::bits; ++p)
    {
        const T pivot = gf2_matrix<T>::bit(p);

        unsigned int pivot_row = p;
        while(pivot_row < gf2_matrix<T>::bits && (matrix.rows[pivot_row] & pivot) == 0)
        {
            ++pivot_row;
        }
        if(pivot_row == gf2_matrix<T>::bits)
        {
            return false;
        }
        std::swap(matrix.rows[p], matrix.rows[pivot_row]);
        std::swap(identity.rows[p], identity.rows[pivot_row]);

        for(unsigned int r = 0; r < gf2_matrix<T>::bits; ++r)
        {
            if(r != p && (matrix.rows[r] & pivot))
            {
                matrix.rows[r] ^= matrix.rows[p];
                identity.rows[r] ^= identity.rows[p];
            }
        }
    }

    inverse = identity;
    return true;
}

/// True when the matrix has ones on the diagonal and zeros strictly above it.
template<class T>
bool is_unit_lower_triangular(const gf2_matrix<T>& matrix)
{
    for(unsigned int r = 0; r < gf2_matrix<T>::bits; ++r)
    {
        const T diagonal = gf2_matrix<T>::bit(r);
        if((matrix.rows[r] & diagonal) == 0)
        {
            return false;
        }
        // Columns to the right of the diagonal occupy the bits below it.
        if(matrix.rows[r] & static_cast<T>(diagonal - 1))
        {
            return false;
        }
    }
    return true;
}

/// Generates the single value that each of `dimensions` dimensions produces at sequence index
/// `offset`. A fresh generator is used so that the offset is applied from a clean state.
template<class Generator, class T>
void generate_at_offset(const unsigned int       dimensions,
                        const unsigned long long offset,
                        std::vector<T>&          values)
{
    T* data{};
    HIP_CHECK(hipMallocHelper(&data, sizeof(T) * dimensions));

    Generator generator;
    ROCRAND_CHECK(generator.set_dimensions(dimensions));
    ROCRAND_CHECK(generator.set_offset(offset));
    ROCRAND_CHECK(generator.generate(data, dimensions));

    values.resize(dimensions);
    HIP_CHECK(hipMemcpy(values.data(), data, sizeof(T) * dimensions, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(data));
}

/// Recovers every dimension's direction vectors, and the scrambling constant applied on top of
/// them, from generated values alone.
///
/// With gray(i) = i ^ (i >> 1):
///   x(0)   = K                    because gray(0) = 0
///   x(1)   = v[0] ^ K             because gray(1) = 1
///   x(2^j) = v[j] ^ v[j-1] ^ K    because gray(2^j) sets bits j and j-1, for j >= 1
/// so v[0] = x(1) ^ x(0) and v[j] = x(2^j) ^ x(0) ^ v[j-1]. Unscrambled generators are the
/// special case K = 0.
///
/// `direction_vectors` comes back laid out as `bits` consecutive vectors per dimension, matching
/// the precomputed tables.
template<class Generator, class T>
void recover_direction_vectors(const unsigned int dimensions,
                               std::vector<T>&    direction_vectors,
                               std::vector<T>&    scramble_constants)
{
    constexpr unsigned int bits = gf2_matrix<T>::bits;

    ASSERT_NO_FATAL_FAILURE((generate_at_offset<Generator, T>(dimensions, 0, scramble_constants)));

    direction_vectors.assign(static_cast<size_t>(dimensions) * bits, T{});

    std::vector<T> values;
    for(unsigned int j = 0; j < bits; ++j)
    {
        SCOPED_TRACE(testing::Message() << "while recovering direction vector " << j);
        ASSERT_NO_FATAL_FAILURE((generate_at_offset<Generator, T>(dimensions, 1ULL << j, values)));

        for(unsigned int d = 0; d < dimensions; ++d)
        {
            T* v = direction_vectors.data() + static_cast<size_t>(d) * bits;
            v[j] = static_cast<T>(values[d] ^ scramble_constants[d] ^ (j == 0 ? T{} : v[j - 1]));
        }
    }
}

/// Fetches the unscrambled Joe-Kuo direction vectors that the scrambled tables are derived from.
template<class T>
void get_unscrambled_direction_vectors(const T** direction_vectors)
{
    if constexpr(std::is_same_v<T, unsigned int>)
    {
        ROCRAND_CHECK(rocrand_get_direction_vectors32(direction_vectors,
                                                      ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
    }
    else
    {
        ROCRAND_CHECK(rocrand_get_direction_vectors64(direction_vectors,
                                                      ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
    }
}

} // namespace sobol
} // namespace rocrand_test

#endif // ROCRAND_TEST_INTERNAL_TEST_ROCRAND_SOBOL_EQUIDISTRIBUTION_HPP_
