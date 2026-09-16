/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_MATRIX_FORMATS_IND_HPP_
#define ROCALUTION_MATRIX_FORMATS_IND_HPP_

// Matrix indexing

// DENSE indexing
#define DENSE_IND(ai, aj, nrow, ncol) (ai) + (aj) * (nrow)
//#define DENSE_IND(ai, aj, nrow, ncol) (aj) + (ai) * (ncol)

// DENSE_IND_BASE == 0 - column-major
// DENSE_IND_BASE == 1 - row-major
#define DENSE_IND_BASE (DENSE_IND(2, 2, 10, 0) == 22 ? 0 : 1)

// ELL indexing
#define ELL_IND_ROW(row, el, nrow, max_row) (el) * (nrow) + (row)
#define ELL_IND_EL(row, el, nrow, max_row) (el) + (max_row) * (row)
#define ELL_IND(row, el, nrow, max_row) ELL_IND_ROW(row, el, nrow, max_row)

// DIA indexing
#define DIA_IND_ROW(row, el, nrow, ndiag) (el) * (nrow) + (row)
#define DIA_IND_EL(row, el, nrow, ndiag) (el) + (ndiag) * (row)
#define DIA_IND(row, el, nrow, ndiag) DIA_IND_ROW(row, el, nrow, ndiag)

// BCSR indexing
#define BCSR_IND(j, bi, bj, dim) (((j) * (dim) * (dim)) + (bi) + (bj) * (dim))
//#define BCSR_IND(j, bi, bj, dim) (((j)*(dim)*(dim)) + (bj) + (bi) * (dim))

// BCSR_IND_BASE == 0 - column-major
// BCSR_IND_BASE == 1 - row-major
#define BCSR_IND_BASE (BCSR_IND(0, 1, 2, 10) == 21 ? 0 : 1)

#endif // ROCALUTION_MATRIX_FORMATS_IND_HPP_
