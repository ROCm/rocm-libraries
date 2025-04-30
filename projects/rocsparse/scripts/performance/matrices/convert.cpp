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

#include <algorithm>
#include <fstream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

int read_mtx_matrix(const char*          filename,
                    int&                 nrow,
                    int&                 ncol,
                    int&                 nnz,
                    std::vector<int>&    row,
                    std::vector<int>&    col,
                    std::vector<double>& val)
{
    FILE* f = fopen(filename, "r");
    if(!f)
    {
        return -1;
    }

    char line[1024];

    // Check for banner
    if(!fgets(line, 1024, f))
    {
        return -1;
    }

    char banner[16];
    char array[16];
    char coord[16];
    char data[16];
    char type[16];

    // Extract banner
    if(sscanf(line, "%s %s %s %s %s", banner, array, coord, data, type) != 5)
    {
        return -1;
    }

    // Convert to lower case
    for(char* p = array; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = coord; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = data; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = type; *p != '\0'; *p = tolower(*p), p++)
        ;

    // Check banner
    if(strncmp(line, "%%MatrixMarket", 14) != 0)
    {
        return -1;
    }

    // Check array type
    if(strcmp(array, "matrix") != 0)
    {
        return -1;
    }

    // Check coord
    if(strcmp(coord, "coordinate") != 0)
    {
        return -1;
    }

    // Check data
    if(strcmp(data, "real") != 0 && strcmp(data, "integer") != 0 && strcmp(data, "pattern") != 0)
    {
        return -1;
    }

    // Check type
    if(strcmp(type, "general") != 0 && strcmp(type, "symmetric") != 0)
    {
        return -1;
    }

    // Symmetric flag
    int symm = !strcmp(type, "symmetric");

    // Skip comments
    while(fgets(line, 1024, f))
    {
        if(line[0] != '%')
        {
            break;
        }
    }

    // Read dimensions
    int snnz;

    sscanf(line, "%d %d %d", &nrow, &ncol, &snnz);
    nnz = symm ? 2 * snnz : snnz;

    std::vector<int>    unsorted_row(nnz);
    std::vector<int>    unsorted_col(nnz);
    std::vector<double> unsorted_val(nnz);

    // Read entries
    int idx = 0;
    while(fgets(line, 1024, f))
    {
        if(idx >= nnz)
        {
            return -1;
        }

        int    irow;
        int    icol;
        double ival;

        if(!strcmp(data, "pattern"))
        {
            sscanf(line, "%d %d", &irow, &icol);
            ival = 1.0;
        }
        else
        {
            sscanf(line, "%d %d %lg", &irow, &icol, &ival);
        }

        --irow;
        --icol;

        unsorted_row[idx] = irow;
        unsorted_col[idx] = icol;
        unsorted_val[idx] = ival;

        ++idx;

        if(symm && irow != icol)
        {
            if(idx >= nnz)
            {
                return -1;
            }

            unsorted_row[idx] = icol;
            unsorted_col[idx] = irow;
            unsorted_val[idx] = ival;
            ++idx;
        }
    }
    fclose(f);

    nnz = idx;

    row.resize(nnz);
    col.resize(nnz);
    val.resize(nnz);

    // Sort by row and column index
    std::vector<int> perm(nnz);
    for(int i = 0; i < nnz; ++i)
    {
        perm[i] = i;
    }

    std::sort(perm.begin(), perm.end(), [&](const int& a, const int& b) {
        if(unsorted_row[a] < unsorted_row[b])
        {
            return true;
        }
        else if(unsorted_row[a] == unsorted_row[b])
        {
            return (unsorted_col[a] < unsorted_col[b]);
        }
        else
        {
            return false;
        }
    });

    for(int i = 0; i < nnz; ++i)
    {
        row[i] = unsorted_row[perm[i]];
        col[i] = unsorted_col[perm[i]];
        val[i] = unsorted_val[perm[i]];
    }

    return 0;
}

int write_bin_matrix(
    const char* filename, int m, int n, int nnz, const int* ptr, const int* col, const double* val)
{
    std::ofstream out(filename, std::ios::out | std::ios::binary);

    if(!out.is_open())
    {
        return -1;
    }

    out << "#rocALUTION binary csr file" << std::endl;

    int version = 10301;
    out.write((char*)&version, sizeof(int));
    out.write((char*)&m, sizeof(int));
    out.write((char*)&n, sizeof(int));
    out.write((char*)&nnz, sizeof(int));
    out.write((char*)ptr, sizeof(int) * (m + 1));
    out.write((char*)col, sizeof(int) * nnz);
    out.write((char*)val, sizeof(double) * nnz);

    out.close();

    return 0;
}

int coo_to_csr(int m, int nnz, const int* src_row, std::vector<int>& dst_ptr)
{
    dst_ptr.resize(m + 1, 0);

    // Compute nnz entries per row
    for(int i = 0; i < nnz; ++i)
    {
        ++dst_ptr[src_row[i] + 1];
    }

    // Exclusive scan
    for(int i = 0; i < m; ++i)
    {
        dst_ptr[i + 1] += dst_ptr[i];
    }

    return 0;
}

int main(int argc, char* argv[])
{
    int m;
    int n;
    int nnz;

    std::vector<int>    ptr;
    std::vector<int>    row;
    std::vector<int>    col;
    std::vector<double> val;

    if(read_mtx_matrix(argv[1], m, n, nnz, row, col, val) != 0)
    {
        fprintf(stderr, "Cannot open [read] %s.\n", argv[1]);
        return -1;
    }

    if(coo_to_csr(m, nnz, row.data(), ptr) != 0)
    {
        fprintf(stderr, "Cannot convert %s from COO to CSR.\n", argv[1]);
        return -1;
    }

    if(write_bin_matrix(argv[2], m, n, nnz, ptr.data(), col.data(), val.data()) != 0)
    {
        fprintf(stderr, "Cannot open [write] %s.\n", argv[2]);
        return -1;
    }

    return 0;
}
