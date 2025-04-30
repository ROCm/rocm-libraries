/* ************************************************************************
 * Copyright (C) 2018-2021 Advanced Micro Devices, Inc. All rights Reserved.
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
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

struct mtx_header
{
    char banner[16];
    char array[16];
    char coord[16];
    char data[16];
    char type[16];
    int  symmetric;
};

bool read_mtx_header(FILE* f, int& nrow, int& ncol, int& nnz, mtx_header& header)
{
    char line[1024];

    // Check for banner
    if(!fgets(line, 1024, f))
    {
        return false;
    }

    // Extract banner
    if(sscanf(line,
              "%s %s %s %s %s",
              header.banner,
              header.array,
              header.coord,
              header.data,
              header.type)
       != 5)
    {
        return false;
    }

    // Convert to lower case
    for(char* p = header.array; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = header.coord; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = header.data; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = header.type; *p != '\0'; *p = tolower(*p), p++)
        ;

    // Check banner
    if(strncmp(line, "%%MatrixMarket", 14))
    {
        return false;
    }

    // Check array type
    if(strcmp(header.array, "matrix"))
    {
        return false;
    }

    // Check coord
    if(strcmp(header.coord, "coordinate"))
    {
        return false;
    }

    // Check data
    if(strcmp(header.data, "real") && strcmp(header.data, "complex")
       && strcmp(header.data, "integer") && strcmp(header.data, "pattern"))
    {
        return false;
    }

    // Check type
    if(strcmp(header.type, "general") && strcmp(header.type, "symmetric")
       && strcmp(header.type, "hermitian"))
    {
        return false;
    }

    // Symmetric flag
    header.symmetric = !strcmp(header.type, "symmetric") || !strcmp(header.type, "hermitian");

    // Skip comments
    while(fgets(line, 1024, f))
    {
        if(line[0] != '%')
        {
            break;
        }
    }

    // Read dimensions
    sscanf(line, "%d %d %d", &nrow, &ncol, &nnz);

    return true;
}

void set_value(double& dst, double rsrc, double isrc)
{
    dst = rsrc;
}

void set_value(std::complex<double>& dst, double rsrc, double isrc)
{
    dst = std::complex<double>(rsrc, isrc);
}

template <typename T>
bool read_mtx_matrix(FILE*             f,
                     const mtx_header& header,
                     int               nrow,
                     int               ncol,
                     int&              nnz,
                     std::vector<int>& row_ind,
                     std::vector<int>& col_ind,
                     std::vector<T>&   val)
{
    // Cache for line
    char line[1024];

    // Read unsorted data
    std::vector<int> unsorted_row(header.symmetric ? 2 * nnz : nnz);
    std::vector<int> unsorted_col(header.symmetric ? 2 * nnz : nnz);
    std::vector<T>   unsorted_val(header.symmetric ? 2 * nnz : nnz);

    // Read entries
    int idx = 0;
    while(fgets(line, 1024, f))
    {
        if(idx >= (header.symmetric ? 2 * nnz : nnz))
        {
            return false;
        }

        int    irow;
        int    icol;
        double rval;
        double ival;

        if(!strcmp(header.data, "pattern"))
        {
            sscanf(line, "%d %d", &irow, &icol);
            rval = 1.0;
        }
        else
        {
            if(!strcmp(header.data, "complex"))
            {
                sscanf(line, "%d %d %lg %lg", &irow, &icol, &rval, &ival);
            }
            else
            {
                sscanf(line, "%d %d %lg", &irow, &icol, &rval);
            }
        }

        --irow;
        --icol;

        unsorted_row[idx] = irow;
        unsorted_col[idx] = icol;
        set_value(unsorted_val[idx], rval, ival);

        ++idx;

        if(header.symmetric && irow != icol)
        {
            if(idx >= (header.symmetric ? 2 * nnz : nnz))
            {
                return false;
            }

            unsorted_row[idx] = icol;
            unsorted_col[idx] = irow;
            set_value(unsorted_val[idx], rval, ival);
            ++idx;
        }
    }

    // Store "real" number of non-zero entries
    nnz = idx;

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

    // Resize arrays
    row_ind.resize(nnz);
    col_ind.resize(nnz);
    val.resize(nnz);

    for(int i = 0; i < nnz; ++i)
    {
        row_ind[i] = unsorted_row[perm[i]];
        col_ind[i] = unsorted_col[perm[i]];
        val[i]     = unsorted_val[perm[i]];
    }

    return true;
}

template <typename T>
bool write_bin_matrix(
    const char* filename, int m, int n, int nnz, const int* ptr, const int* col, const T* val)
{
    std::ofstream out(filename, std::ios::out | std::ios::binary);

    if(!out.is_open())
    {
        return false;
    }

    // Header
    out << "#rocALUTION binary csr file" << std::endl;

    // rocALUTION version
    int version = 10602;
    out.write((char*)&version, sizeof(int));

    // Data
    out.write((char*)&m, sizeof(int));
    out.write((char*)&n, sizeof(int));
    out.write((char*)&nnz, sizeof(int));
    out.write((char*)ptr, (m + 1) * sizeof(int));
    out.write((char*)col, nnz * sizeof(int));
    out.write((char*)val, nnz * sizeof(T));

    out.close();

    return true;
}

bool coo_to_csr(int m, int nnz, const int* src_row, std::vector<int>& dst_ptr)
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

    return true;
}

int main(int argc, char* argv[])
{
    if(argc < 3)
    {
        std::cerr << argv[0] << " <matrix.mtx> <matrix.csr>" << std::endl;
        return -1;
    }

    // Matrix dimensions
    int m;
    int n;
    int nnz;

    // Matrix mtx header
    mtx_header header;

    // Open file for reading
    FILE* f = fopen(argv[1], "r");
    if(!f)
    {
        std::cerr << "Cannot open [read] .mtx file " << argv[1] << std::endl;
        return -1;
    }

    if(!read_mtx_header(f, m, n, nnz, header))
    {
        std::cerr << "Cannot read .mtx header from " << argv[1] << std::endl;
        return -1;
    }

    std::vector<int>                  row_ptr;
    std::vector<int>                  row_ind;
    std::vector<int>                  col_ind;
    std::vector<double>               rval;
    std::vector<std::complex<double>> cval;

    bool status;
    if(!strcmp(header.data, "complex"))
    {
        status = read_mtx_matrix(f, header, m, n, nnz, row_ind, col_ind, cval);
    }
    else
    {
        status = read_mtx_matrix(f, header, m, n, nnz, row_ind, col_ind, rval);
    }

    if(!status)
    {
        std::cerr << "Cannot read .mtx data from " << argv[1] << std::endl;
        return -1;
    }

    // Close file
    fclose(f);

    if(!coo_to_csr(m, nnz, row_ind.data(), row_ptr))
    {
        std::cerr << "Cannot convert " << argv[1] << " from COO to CSR." << std::endl;
        return -1;
    }

    if(!strcmp(header.data, "complex"))
    {
        status = write_bin_matrix(argv[2], m, n, nnz, row_ptr.data(), col_ind.data(), cval.data());
    }
    else
    {
        status = write_bin_matrix(argv[2], m, n, nnz, row_ptr.data(), col_ind.data(), rval.data());
    }

    if(!status)
    {
        std::cerr << "Cannot open [write] " << argv[2] << std::endl;
        return -1;
    }

    return 0;
}
