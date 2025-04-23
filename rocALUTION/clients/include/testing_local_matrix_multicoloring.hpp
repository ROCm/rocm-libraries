/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once

#include "utility.hpp"
#include "validate.hpp"

#include <gtest/gtest.h>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

template <typename T>
bool testing_local_matrix_multicoloring(Arguments argus)
{
    const int          size        = argus.size;
    const std::string  matrix_type = argus.matrix_type;
    const unsigned int format      = argus.format;

    // Initialize rocALUTION
    set_device_rocalution(device);
    init_rocalution();

    // Generate A
    int* csr_ptr = NULL;
    int* csr_col = NULL;
    T*   csr_val = NULL;

    int nrow = 0;
    int ncol = 0;
    if(matrix_type == "Laplacian2D")
    {
        nrow = gen_2d_laplacian(size, &csr_ptr, &csr_col, &csr_val);
        ncol = nrow;
    }
    else if(matrix_type == "PermutedIdentity")
    {
        nrow = gen_permuted_identity(size, &csr_ptr, &csr_col, &csr_val);
        ncol = nrow;
    }
    else if(matrix_type == "Random")
    {
        nrow = gen_random(100 * size, 100 * size, 6, &csr_ptr, &csr_col, &csr_val);
        ncol = 100 * size;
    }
    else
    {
        return false;
    }

    int nnz = csr_ptr[nrow];

    assert(csr_ptr != NULL);
    assert(csr_col != NULL);
    assert(csr_val != NULL);

    LocalMatrix<T> A;

    A.MoveToHost();
    A.AllocateCSR("A", nnz, nrow, ncol);
    A.CopyFromCSR(csr_ptr, csr_col, csr_val);

    // Matrix format
    A.ConvertTo(format, format == BCSR ? argus.blockdim : 1);

    // coloring data
    LocalVector<int> coloring;

    int  num_colors;
    int* size_colors = NULL;
    int* perm        = new int[nrow];

    bool success = true;

    // Check host multicoloring
    A.MultiColoring(num_colors, &size_colors, &coloring);
    coloring.CopyToHostData(perm);
    success &= valid_permutation(nrow, perm);
    success &= valid_coloring(nrow, csr_ptr, csr_col, num_colors, size_colors, perm);

    // Reset
    coloring.Clear();
    delete[] size_colors;
    size_colors = NULL;

    // Check accelerator multicoloring
    A.MoveToAccelerator();
    coloring.MoveToAccelerator();

    A.MultiColoring(num_colors, &size_colors, &coloring);
    coloring.CopyToHostData(perm);
    success &= valid_permutation(nrow, perm);
    success &= valid_coloring(nrow, csr_ptr, csr_col, num_colors, size_colors, perm);

    // Clean up
    delete[] csr_ptr;
    delete[] csr_col;
    delete[] csr_val;

    delete[] size_colors;
    delete[] perm;

    // Stop rocALUTION platform
    stop_rocalution();

    return success;
}
