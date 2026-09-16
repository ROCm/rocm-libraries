/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_MATRIX_FORMATS_HPP_
#define ROCALUTION_MATRIX_FORMATS_HPP_

#include <cstdint>
#include <string>

namespace rocalution
{

    // Matrix Names
    const std::string _matrix_format_names[8]
        = {"DENSE", "CSR", "MCSR", "BCSR", "COO", "DIA", "ELL", "HYB"};

    // Matrix Enumeration
    enum _matrix_format
    {
        DENSE = 0,
        CSR   = 1,
        MCSR  = 2,
        BCSR  = 3,
        COO   = 4,
        DIA   = 5,
        ELL   = 6,
        HYB   = 7
    };

    // Sparse Matrix - Sparse Compressed Row Format CSR
    template <typename ValueType, typename IndexType, typename PointerType>
    struct MatrixCSR
    {
        // Row offsets (row ptr)
        PointerType* row_offset;

        // Column index
        IndexType* col;

        // Values
        ValueType* val;
    };

    // Sparse Matrix - Modified Sparse Compressed Row Format MCSR
    template <typename ValueType, typename IndexType>
    struct MatrixMCSR
    {
        // Row offsets (row ptr)
        IndexType* row_offset;

        // Column index
        IndexType* col;

        // Values
        ValueType* val;
    };

    template <typename ValueType, typename IndexType, typename Index = IndexType>
    struct MatrixBCSR
    {
        // Number of block rows
        Index nrowb;
        // Number of block columns
        Index ncolb;
        // Number of block nnz
        int64_t nnzb;

        // Block dimension
        Index blockdim;

        // Row offsets (row ptr)
        IndexType* row_offset;

        // Column index
        IndexType* col;

        // Values
        ValueType* val;
    };

    // Sparse Matrix - Coordinate Format COO
    template <typename ValueType, typename IndexType>
    struct MatrixCOO
    {
        // Row index
        IndexType* row;

        // Column index
        IndexType* col;

        // Values
        ValueType* val;
    };

    // Sparse Matrix - Diagonal Format DIA (see DIA_IND for indexing)
    template <typename ValueType, typename IndexType, typename Index = IndexType>
    struct MatrixDIA
    {
        // Number of diagonal
        Index num_diag;

        // Offset with respect to the main diagonal
        IndexType* offset;

        // Values
        ValueType* val;
    };

    // Sparse Matrix - ELL Format (see ELL_IND for indexing)
    template <typename ValueType, typename IndexType, typename Index = IndexType>
    struct MatrixELL
    {
        // Maximal elements per row
        Index max_row;

        // Column index
        IndexType* col;

        // Values
        ValueType* val;
    };

    // Sparse Matrix - Hybrid Format HYB (Contains ELL and COO Matrices)
    template <typename ValueType, typename IndexType, typename Index = IndexType>
    struct MatrixHYB
    {
        MatrixELL<ValueType, IndexType, Index> ELL;
        MatrixCOO<ValueType, IndexType>        COO;
    };

    // Dense Matrix (see DENSE_IND for indexing)
    template <typename ValueType>
    struct MatrixDENSE
    {
        // Values
        ValueType* val;
    };

} // namespace rocalution

#endif // ROCALUTION_MATRIX_FORMATS_HPP_
