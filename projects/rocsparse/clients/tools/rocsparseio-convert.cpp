/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "app.hpp"
#include "rocsparseio.h"
#include <algorithm>
#include <chrono>
#include <complex>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>

namespace rocsparseio
{
    ///
    ///
    ///
    ///
    typedef rocsparseio_enum_t rocsparseio_file_format_t;

#define ROCSPARSEIO_FILE_FORMAT_MTX 0
#define ROCSPARSEIO_FILE_FORMAT_CSR 1
#define ROCSPARSEIO_FILE_FORMAT_ROCSPARSEIO 2
#define ROCSPARSEIO_FILE_FORMAT_ASCII 3

    typedef enum rocsparseio_file_format_
    {
        rocsparseio_file_format_mtx         = ROCSPARSEIO_FILE_FORMAT_MTX,
        rocsparseio_file_format_csr         = ROCSPARSEIO_FILE_FORMAT_CSR,
        rocsparseio_file_format_rocsparseio = ROCSPARSEIO_FILE_FORMAT_ROCSPARSEIO,
        rocsparseio_file_format_ascii       = ROCSPARSEIO_FILE_FORMAT_ASCII
    } rocsparseio_file_format;

    ///
    /// @brief c++11 struct for format enum.
    ///
    struct file_format_t
    {
        typedef enum value_type_ : rocsparseio_file_format_t
        {
            mtx         = rocsparseio_file_format_mtx,
            csr         = rocsparseio_file_format_csr,
            rocsparseio = rocsparseio_file_format_rocsparseio,
            ascii       = rocsparseio_file_format_ascii
        } value_type;

        value_type       value{};
        inline constexpr operator value_type() const
        {
            return this->value;
        }
        inline constexpr file_format_t() {}
        inline constexpr file_format_t(rocsparseio_file_format_t ival)
            : value((value_type)ival)
        {
        }

        inline constexpr file_format_t(rocsparseio_file_format ival)
            : value((value_type)ival)
        {
        }

        static constexpr value_type all[4] = {file_format_t::mtx,
                                              file_format_t::csr,
                                              file_format_t::rocsparseio,
                                              file_format_t::ascii};

        inline explicit constexpr operator rocsparseio_file_format() const
        {
            return (rocsparseio_file_format)this->value;
        };

        inline bool is_invalid() const
        {
            switch(this->value)
            {
            case mtx:
            case csr:
            case ascii:
            case rocsparseio:
            {
                return false;
            }
            }
            return true;
        };

        inline file_format_t(const char* name)
        {
            int j = 0;
            for(int i = 0; name[i] != '\0'; ++i)
            {
                if(name[i] == '.')
                {
                    j = i;
                }
            }

            if(!strcmp(name + j, ".mtx"))
            {
                value = mtx;
            }
            else if(!strcmp(name + j, ".csr"))
            {
                value = csr;
            }
            else if(!strcmp(name + j, ".rocsparseio"))
            {
                value = rocsparseio;
            }
            else if(!strcmp(name + j, ".txt"))
            {
                value = ascii;
            }
            else
            {
                value = (value_type)-1;
            }
        };

        inline const char* to_string() const
        {
            switch(this->value)
            {
#define CASE(case_name)    \
    case case_name:        \
    {                      \
        return #case_name; \
    }
                CASE(mtx);
                CASE(csr);
                CASE(rocsparseio);
                CASE(ascii);
#undef CASE
            }
            return "unknown";
        }
    };

    constexpr file_format_t::value_type file_format_t::all[4];

} // namespace rocsparseio

inline std::ostream& operator<<(std::ostream& os, const rocsparseio::file_format_t& that_)
{
    os << that_.to_string();
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const _Float16& that_)
{
    os << (float)that_;
    return os;
}

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
    std::vector<int> unsorted_row(header.symmetric ? nnz * 2 : nnz);
    std::vector<int> unsorted_col(header.symmetric ? nnz * 2 : nnz);
    std::vector<T>   unsorted_val(header.symmetric ? nnz * 2 : nnz);

    // Read entries
    int idx = 0;
    while(fgets(line, 1024, f))
    {
        if(idx >= (header.symmetric ? nnz * 2 : nnz))
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
bool write_csr_matrix(
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

int mtx2csr(const char* ifilename, const char* ofilename)
{
    // Matrix dimensions
    int m;
    int n;
    int nnz;

    // Matrix mtx header
    mtx_header header;

    // Open file for reading
    FILE* f = fopen(ifilename, "r");
    if(!f)
    {
        std::cerr << "Cannot open [read] .mtx file " << ifilename << std::endl;
        return -1;
    }

    if(!read_mtx_header(f, m, n, nnz, header))
    {
        std::cerr << "Cannot read .mtx header from " << ifilename << std::endl;
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
        std::cerr << "Cannot read .mtx data from " << ifilename << std::endl;
        return -1;
    }

    // Close file
    fclose(f);

    if(!coo_to_csr(m, nnz, row_ind.data(), row_ptr))
    {
        std::cerr << "Cannot convert " << ifilename << " from COO to CSR." << std::endl;
        return -1;
    }

    if(!strcmp(header.data, "complex"))
    {
        status
            = write_csr_matrix(ofilename, m, n, nnz, row_ptr.data(), col_ind.data(), cval.data());
    }
    else
    {
        status
            = write_csr_matrix(ofilename, m, n, nnz, row_ptr.data(), col_ind.data(), rval.data());
    }

    if(!status)
    {
        std::cerr << "Cannot open [write] " << ofilename << std::endl;
        return -1;
    }

    return 0;
}

int csr2rocsparseio(const char* ifilename, const char* ofilename)
{
    // Read matrix
    std::ifstream fin(ifilename, std::ios::in | std::ios::binary);
    if(!fin.is_open())
    {
        fprintf(stderr, "Cannot open file %s\n", ifilename);
        exit(1);
    }

    std::string header;
    std::getline(fin, header);

    if(header != "#rocALUTION binary csr file")
    {
        fprintf(stderr, "Cannot open file %s\n", ifilename);
        exit(1);
    }

    int version;
    fin.read((char*)&version, sizeof(int));

    int m;
    int n;
    int nnz;

    fin.read((char*)&m, sizeof(int));
    fin.read((char*)&n, sizeof(int));
    fin.read((char*)&nnz, sizeof(int));

    printf("Reading %d x %d matrix with %d nnz.\n", m, n, nnz);

    std::vector<int>    hcsr_row_ptr(m + 1);
    std::vector<int>    hcsr_col_ind(nnz);
    std::vector<double> hcsr_val(nnz);

    fin.read((char*)hcsr_row_ptr.data(), sizeof(int) * (m + 1));
    fin.read((char*)hcsr_col_ind.data(), sizeof(int) * nnz);
    fin.read((char*)hcsr_val.data(), sizeof(double) * nnz);

    fin.close();

    {
        rocsparseio_status status;
        rocsparseio_handle handle;
        status = rocsparseio_open(&handle, rocsparseio_rwmode_write, ofilename);

        status = rocsparseio_write_sparse_csx(handle,
                                              rocsparseio_direction_row,
                                              m,
                                              n,
                                              nnz,
                                              rocsparseio_type_int32,
                                              hcsr_row_ptr.data(),
                                              rocsparseio_type_int32,
                                              hcsr_col_ind.data(),
                                              rocsparseio_type_float64,
                                              hcsr_val.data(),
                                              rocsparseio_index_base_zero,
                                              nullptr);
        //        // ROCSPARSEIO_CHECK(status);

        status = rocsparseio_close(handle);
        //        // ROCSPARSEIO_CHECK(status);
        return status;
    }
}

int csr2ascii(const char* ifilename, const char* ofilename)
{

    // Read matrix
    std::ifstream fin(ifilename, std::ios::in | std::ios::binary);
    if(!fin.is_open())
    {
        fprintf(stderr, "Cannot open file %s\n", ifilename);
        exit(1);
    }

    std::string header;
    std::getline(fin, header);

    if(header != "#rocALUTION binary csr file")
    {
        fprintf(stderr, "Cannot open file %s\n", ifilename);
        exit(1);
    }

    int version;
    fin.read((char*)&version, sizeof(int));

    int m;
    int n;
    int nnz;

    fin.read((char*)&m, sizeof(int));
    fin.read((char*)&n, sizeof(int));
    fin.read((char*)&nnz, sizeof(int));

    std::vector<int>    hcsr_row_ptr(m + 1);
    std::vector<int>    hcsr_col_ind(nnz);
    std::vector<double> hcsr_val(nnz);

    fin.read((char*)hcsr_row_ptr.data(), sizeof(int) * (m + 1));
    fin.read((char*)hcsr_col_ind.data(), sizeof(int) * nnz);
    fin.read((char*)hcsr_val.data(), sizeof(double) * nnz);

    fin.close();

    std::ofstream out(ofilename);
    out << "m: " << m << std::endl;
    out << "n: " << n << std::endl;
    out << "nnz: " << nnz << std::endl;
    out << "base: " << 0 << std::endl;
    for(int i = 0; i < m; ++i)
    {
        out << "line: " << i << std::endl;
        for(int k = hcsr_row_ptr[i]; k < hcsr_row_ptr[i + 1]; ++k)
        {
            out << " col = " << hcsr_col_ind[k] << ", val =  " << hcsr_val[k] << std::endl;
        }
    }
    out.close();

    return rocsparseio_status_success;
}

template <typename I0, typename I1, typename T>
rocsparseio_status write_ascii_sparse_coo_template(const char*            ofilename,
                                                   uint64_t               m,
                                                   uint64_t               n,
                                                   uint64_t               nnz,
                                                   const void*            row_ind_,
                                                   const void*            col_ind_,
                                                   const void*            val_,
                                                   rocsparseio_index_base index_base)
{
    const I0*     row_ind = (const I0*)row_ind_;
    const I1*     col_ind = (const I1*)col_ind_;
    const T*      val     = (const T*)val_;
    std::ofstream out(ofilename);
    out << "m: " << m << std::endl;
    out << "n: " << n << std::endl;
    out << "nnz: " << nnz << std::endl;
    out << "base: " << index_base << std::endl;
    for(uint64_t i = 0; i < nnz; ++i)
    {
        out << "row = " << row_ind[i] << ", col = " << col_ind[i] << ", val =  " << val[i]
            << std::endl;
    }

    out.close();
    return rocsparseio_status_success;
}

template <typename T>
rocsparseio_status
    write_ascii_dense_vector_template(const char* ofilename, uint64_t m, const void* val_)
{
    const T*      val = (const T*)val_;
    std::ofstream out(ofilename);
    out << "m: " << m << std::endl;
    for(uint64_t i = 0; i < m; ++i)
    {
        out << "val[" << i << "] = " << val[i] << std::endl;
    }

    out.close();
    return rocsparseio_status_success;
}

template <typename I, typename J, typename... P>
rocsparseio_status write_ascii_sparse_coo_dispatch_t(rocsparseio_type t, P... params)
{
    switch(t)
    {
    case rocsparseio_type_float16:
    {
        return write_ascii_sparse_coo_template<I, J, _Float16>(params...);
    }
    case rocsparseio_type_float32:
    {
        return write_ascii_sparse_coo_template<I, J, float>(params...);
    }
    case rocsparseio_type_float64:
    {
        return write_ascii_sparse_coo_template<I, J, double>(params...);
    }
    case rocsparseio_type_complex32:
    {
        return write_ascii_sparse_coo_template<I, J, std::complex<float>>(params...);
    }
    case rocsparseio_type_complex64:
    {
        return write_ascii_sparse_coo_template<I, J, std::complex<double>>(params...);
    }
    case rocsparseio_type_int8:
    {
        return write_ascii_sparse_coo_template<I, J, int8_t>(params...);
    }
    case rocsparseio_type_int32:
    case rocsparseio_type_int64:
    case rocsparseio_type_bfloat16:
    {
        return rocsparseio_status_invalid_value;
    }
    }
}

template <typename I, typename... P>
rocsparseio_status
    write_ascii_sparse_coo_dispatch_jt(rocsparseio_type j, rocsparseio_type t, P... params)
{
    switch(j)
    {
    case rocsparseio_type_int32:
    {
        return write_ascii_sparse_coo_dispatch_t<I, int32_t, P...>(t, params...);
    }
    case rocsparseio_type_int64:
    {
        return write_ascii_sparse_coo_dispatch_t<I, int64_t, P...>(t, params...);
    }
    case rocsparseio_type_int8:
    case rocsparseio_type_float16:
    case rocsparseio_type_bfloat16:
    case rocsparseio_type_float32:
    case rocsparseio_type_float64:
    case rocsparseio_type_complex32:
    case rocsparseio_type_complex64:
    {
        return rocsparseio_status_invalid_value;
    }
    }
}

template <typename... P>
rocsparseio_status
    write_ascii_sparse_coo(rocsparseio_type i, rocsparseio_type j, rocsparseio_type t, P... params)
{
    switch(i)
    {
    case rocsparseio_type_int32:
    {
        return write_ascii_sparse_coo_dispatch_jt<int32_t, P...>(j, t, params...);
    }
    case rocsparseio_type_int64:
    {
        return write_ascii_sparse_coo_dispatch_jt<int64_t, P...>(j, t, params...);
    }
    case rocsparseio_type_int8:
    case rocsparseio_type_float16:
    case rocsparseio_type_bfloat16:
    case rocsparseio_type_float32:
    case rocsparseio_type_float64:
    case rocsparseio_type_complex32:
    case rocsparseio_type_complex64:
    {
        return rocsparseio_status_invalid_value;
    }
    }
}

template <typename... P>
rocsparseio_status write_ascii_dense_vector(rocsparseio_type t, P... params)
{
    switch(t)
    {
    case rocsparseio_type_int32:
    {
        return write_ascii_dense_vector_template<int32_t>(params...);
    }
    case rocsparseio_type_int64:
    {
        return write_ascii_dense_vector_template<int64_t>(params...);
    }
    case rocsparseio_type_int8:
    {
        return write_ascii_dense_vector_template<int8_t>(params...);
    }
    case rocsparseio_type_float16:
    {
        return write_ascii_dense_vector_template<_Float16>(params...);
    }
    case rocsparseio_type_float32:
    {
        return write_ascii_dense_vector_template<float>(params...);
    }
    case rocsparseio_type_float64:
    {
        return write_ascii_dense_vector_template<double>(params...);
    }
    case rocsparseio_type_complex32:
    {
        return write_ascii_dense_vector_template<std::complex<float>>(params...);
    }
    case rocsparseio_type_complex64:
    {
        return write_ascii_dense_vector_template<std::complex<double>>(params...);
    }
    case rocsparseio_type_bfloat16:
    {
        return rocsparseio_status_invalid_value;
    }
    }
}

rocsparseio_status rocsparseio2ascii(const char* ifilename, const char* ofilename)
{
    rocsparseio_status status;
    rocsparseio_handle handle;
    status = rocsparseio_open(&handle, rocsparseio_rwmode_read, ifilename);
    // ROCSPARSEIO_CHECK(status);

    rocsparseio_format format;
    status = rocsparseio_read_format(handle, &format);
    // ROCSPARSEIO_CHECK(status);
    switch(format)
    {
    case rocsparseio_format_dense_vector:
    {
        rocsparseio_type data_type;
        uint64_t         m;
        void*            data;
        status = rocsparseiox_read_metadata_dense_vector(handle, &data_type, &m);
        // ROCSPARSEIO_CHECK(status);
        uint64_t data_type_size;
        status = rocsparseio_type_get_size(data_type, &data_type_size);
        data   = (m > 0) ? malloc(data_type_size * m) : nullptr;
        status = rocsparseiox_read_dense_vector(handle, data, (uint64_t)1);
        // ROCSPARSEIO_CHECK(status);
        write_ascii_dense_vector(data_type, ofilename, m, data);
        break;
    }
    case rocsparseio_format_sparse_dia:
    case rocsparseio_format_sparse_hyb:
    case rocsparseio_format_sparse_ell:
    case rocsparseio_format_sparse_mcsx:
    {
        break;
    }

    case rocsparseio_format_dense_matrix:
    {
#if 0
	rocsparseio_order_t order_type;
	rocsparseio_type data_type;
	uint64_t m, n;
	void  *data;
	status = rocsparseiox_read_metadata_sparse_csx(handle,
						       &order_type,
						       &m,
						       &n,
						       &data_type);
	// ROCSPARSEIO_CHECK(status);
	uint64_t data_type_size;
	status =  rocsparseio_type_get_size(data_type, &data_type_size);
	data = (m * n > 0) ? malloc(data_type_size * m*n) : nullptr;
	status = rocsparseiox_read_sparse_csx(handle, data);
	// ROCSPARSEIO_CHECK(status);
#endif
        break;
    }
    case rocsparseio_format_sparse_csx:
    {
        rocsparseio_type       ptr_type, ind_type, data_type;
        rocsparseio_index_base base;
        uint64_t               m, n, nnz;
        rocsparseio_direction  dir;
        void *                 ptr, *ind, *data;
        status = rocsparseiox_read_metadata_sparse_csx(
            handle, &dir, &m, &n, &nnz, &ptr_type, &ind_type, &data_type, &base);
        // ROCSPARSEIO_CHECK(status);

        {
            uint64_t size;
            status = rocsparseio_type_get_size(ptr_type, &size);
            ptr    = (m > 0) ? malloc(size * (m + 1)) : nullptr;
        }

        {
            uint64_t size;
            status = rocsparseio_type_get_size(ind_type, &size);
            ind    = (nnz > 0) ? malloc(size * nnz) : nullptr;
        }
        {
            uint64_t size;
            status = rocsparseio_type_get_size(data_type, &size);
            data   = (nnz > 0) ? malloc(size * nnz) : nullptr;
        }
        status = rocsparseiox_read_sparse_csx(handle, ptr, ind, data);
        // ROCSPARSEIO_CHECK(status);
        break;
    }
    case rocsparseio_format_sparse_gebsx:
    {
#if 0
	rocsparseio_type ptr_type, ind_type, data_type;
	rocsparseio_index_base base;
	uint64_t mb, nb, nnzb,bm,bn;
	rocsparseio_direction dir, dirb;
	void *ptr, *ind, *data;
	status = rocsparseiox_read_metadata_sparse_gebsx(handle,
							 &dir,
							 &dirb,
							 &mb,
							 &nb,
							 &nnzb,
							 &bm,
							 &bn,
							 &ptr_type,
							 &ind_type,
							 &data_type,
							 &base);
	// ROCSPARSEIO_CHECK(status);
	uint64_t ob = dir == rocsparseio_direction_row ? mb : nb;

	{ uint64_t size;
	  status =  rocsparseio_type_get_size(ptr_type, &size);
	  ptr = (ob > 0) ? malloc(size * (ob + 1)) : nullptr;
	}

	{ uint64_t size;
	  status =  rocsparseio_type_get_size(ind_type, &size);
	  ind = (nnzb > 0) ? malloc(size * nnzb) : nullptr;
	}

	{ uint64_t size;
	  status =  rocsparseio_type_get_size(data_type, &size);
	  data = (nnzb > 0) ? malloc(size * nnzb * bm * bn) : nullptr;
	}
	status = rocsparseiox_read_sparse_gebsx(handle, ptr, ind, data);
	// ROCSPARSEIO_CHECK(status);
#endif
        break;
    }

    case rocsparseio_format_sparse_coo:
    {
        rocsparseio_type       row_ind_type, col_ind_type, data_type;
        rocsparseio_index_base base;
        uint64_t               m, n, nnz;
        rocsparseio_direction  dir;
        void *                 row_ind, *col_ind, *data;
        status = rocsparseiox_read_metadata_sparse_coo(
            handle, &m, &n, &nnz, &row_ind_type, &col_ind_type, &data_type, &base);
        // ROCSPARSEIO_CHECK(status);

        {
            uint64_t size;
            status  = rocsparseio_type_get_size(row_ind_type, &size);
            row_ind = (nnz > 0) ? malloc(size * nnz) : nullptr;
        }

        {
            uint64_t size;
            status  = rocsparseio_type_get_size(col_ind_type, &size);
            col_ind = (nnz > 0) ? malloc(size * nnz) : nullptr;
        }

        {
            uint64_t size;
            status = rocsparseio_type_get_size(data_type, &size);
            data   = (nnz > 0) ? malloc(size * nnz) : nullptr;
        }

        status = rocsparseiox_read_sparse_coo(handle, row_ind, col_ind, data);
        // ROCSPARSEIO_CHECK(status);

        write_ascii_sparse_coo(row_ind_type,
                               col_ind_type,
                               data_type,
                               ofilename,
                               m,
                               n,
                               nnz,
                               row_ind,
                               col_ind,
                               data,
                               base);

        break;
    }
    }
    status = rocsparseio_close(handle);
    // ROCSPARSEIO_CHECK(status);
    return rocsparseio_status_success;
}

rocsparseio_status rocsparseio2csr(const char* ifilename, const char* ofilename)
{

    rocsparseio_status status;
    rocsparseio_handle handle;
    status = rocsparseio_open(&handle, rocsparseio_rwmode_read, ifilename);
    // ROCSPARSEIO_CHECK(status);

    rocsparseio_type       ptr_type, ind_type, data_type;
    rocsparseio_index_base base;
    uint64_t               m, n, nnz;
    rocsparseio_direction  dir;
    void *                 ptr, *ind, *data;
    status = rocsparseiox_read_metadata_sparse_csx(
        handle, &dir, &m, &n, &nnz, &ptr_type, &ind_type, &data_type, &base);
    // ROCSPARSEIO_CHECK(status);

    int im   = (std::numeric_limits<int>::max() < m) ? -1 : m;
    int in   = (std::numeric_limits<int>::max() < n) ? -1 : n;
    int innz = (std::numeric_limits<int>::max() < nnz) ? -1 : nnz;
    if(im < 0)
    {
        std::cerr << "out of bounds, m = " << m << " is not assignable to an int" << std::endl;
        return rocsparseio_status_invalid_value;
    }
    if(in < 0)
    {
        std::cerr << "out of bounds, n = " << n << " is not assignable to an int" << std::endl;
        return rocsparseio_status_invalid_value;
    }
    if(innz < 0)
    {
        std::cerr << "out of bounds, nnz = " << nnz << " is not assignable to an int" << std::endl;
        return rocsparseio_status_invalid_value;
    }

    {
        uint64_t size;
        status = rocsparseio_type_get_size(ptr_type, &size);
        ptr    = (m > 0) ? malloc(size * (m + 1)) : nullptr;
    }

    {
        uint64_t size;
        status = rocsparseio_type_get_size(ind_type, &size);
        ind    = (nnz > 0) ? malloc(size * nnz) : nullptr;
    }

    {
        uint64_t size;
        status = rocsparseio_type_get_size(data_type, &size);
        data   = (nnz > 0) ? malloc(size * nnz) : nullptr;
    }

    status = rocsparseiox_read_sparse_csx(handle, ptr, ind, data);
    // ROCSPARSEIO_CHECK(status);
    status = rocsparseio_close(handle);
    // ROCSPARSEIO_CHECK(status);

    int32_t* iptr  = nullptr;
    int32_t* iind  = nullptr;
    double*  idata = nullptr;
    if(m > 0)
    {
        switch(ptr_type)
        {
        case rocsparseio_type_int32:
        {
            iptr = (int32_t*)ptr;
            break;
        }
        case rocsparseio_type_int64:
        {
            iptr       = (int32_t*)malloc(sizeof(int32_t) * (m + 1));
            int64_t* p = (int64_t*)ptr;
            for(uint64_t i = 0; i < m + 1; ++i)
            {
                iptr[i] = p[i];
            }
            break;
        }
        case rocsparseio_type_int8:
        case rocsparseio_type_float16:
        case rocsparseio_type_bfloat16:
        case rocsparseio_type_float32:
        case rocsparseio_type_float64:
        case rocsparseio_type_complex32:
        case rocsparseio_type_complex64:
        {
            break;
        }
        }
    }
    if(nnz > 0)
    {
        switch(ind_type)
        {
        case rocsparseio_type_int32:
        {
            iind = (int32_t*)ind;
            break;
        }
        case rocsparseio_type_int64:
        {
            iind       = (int32_t*)malloc(sizeof(int32_t) * nnz);
            int64_t* p = (int64_t*)ind;
            for(uint64_t i = 0; i < nnz; ++i)
            {
                iind[i] = p[i];
            }
            break;
        }
        case rocsparseio_type_int8:
        case rocsparseio_type_float16:
        case rocsparseio_type_bfloat16:
        case rocsparseio_type_float32:
        case rocsparseio_type_float64:
        case rocsparseio_type_complex32:
        case rocsparseio_type_complex64:
        {
            break;
        }
        }
    }
    if(nnz > 0)
    {
        switch(data_type)
        {
        case rocsparseio_type_float64:
        {
            idata = (double*)data;
            break;
        }
        case rocsparseio_type_float32:
        {
            idata    = (double*)malloc(sizeof(double) * (nnz));
            float* p = (float*)data;
            for(uint64_t i = 0; i < nnz; ++i)
            {
                idata[i] = p[i];
            }
            break;
        }
        case rocsparseio_type_float16:
        {
            idata       = (double*)malloc(sizeof(double) * (nnz));
            _Float16* p = (_Float16*)data;
            for(uint64_t i = 0; i < nnz; ++i)
            {
                idata[i] = p[i];
            }
            break;
        }
        case rocsparseio_type_int8:
        case rocsparseio_type_int32:
        case rocsparseio_type_int64:
        case rocsparseio_type_complex32:
        case rocsparseio_type_complex64:
        case rocsparseio_type_bfloat16:
        {
            break;
        }
        }
    }

    write_csr_matrix<double>(ofilename, im, in, innz, iptr, iind, idata);

    if(idata != data)
    {
        free(idata);
    }
    if(iind != ind)
    {
        free(iind);
    }
    if(idata != data)
    {
        free(idata);
    }

    return status;
}

int mtx2rocsparseio(const char* ifilename, const char* ofilename)
{
    // Matrix dimensions
    int m;
    int n;
    int nnz;

    // Matrix mtx header
    mtx_header header;

    // Open file for reading
    FILE* f = fopen(ifilename, "r");
    if(!f)
    {
        std::cerr << "Cannot open [read] .mtx file " << ifilename << std::endl;
        return -1;
    }

    if(!read_mtx_header(f, m, n, nnz, header))
    {
        std::cerr << "Cannot read .mtx header from " << ifilename << std::endl;
        return -1;
    }

    std::vector<int>                  row_ptr;
    std::vector<int>                  row_ind;
    std::vector<int>                  col_ind;
    std::vector<double>               rval;
    std::vector<std::complex<double>> cval;

    bool status_mtx;
    if(!strcmp(header.data, "complex"))
    {
        status_mtx = read_mtx_matrix(f, header, m, n, nnz, row_ind, col_ind, cval);
    }
    else
    {
        status_mtx = read_mtx_matrix(f, header, m, n, nnz, row_ind, col_ind, rval);
    }

    if(!status_mtx)
    {
        std::cerr << "Cannot read .mtx data from " << ifilename << std::endl;
        return -1;
    }

    // Close file
    fclose(f);

    if(!coo_to_csr(m, nnz, row_ind.data(), row_ptr))
    {
        std::cerr << "Cannot convert " << ifilename << " from COO to CSR." << std::endl;
        return -1;
    }

    bool is_complex = (!strcmp(header.data, "complex"));

    {
        rocsparseio_status status;
        rocsparseio_handle handle;
        status = rocsparseio_open(&handle, rocsparseio_rwmode_write, ofilename);

        const char* name = nullptr;
        if(is_complex)
        {
            status = rocsparseio_write_sparse_csx(handle,
                                                  rocsparseio_direction_row,
                                                  m,
                                                  n,
                                                  nnz,
                                                  rocsparseio_type_int32,
                                                  row_ptr.data(),
                                                  rocsparseio_type_int32,
                                                  col_ind.data(),
                                                  rocsparseio_type_complex64,
                                                  cval.data(),
                                                  rocsparseio_index_base_zero,
                                                  name);
        }
        else
        {
            status = rocsparseio_write_sparse_csx(handle,
                                                  rocsparseio_direction_row,
                                                  m,
                                                  n,
                                                  nnz,
                                                  rocsparseio_type_int32,
                                                  row_ptr.data(),
                                                  rocsparseio_type_int32,
                                                  col_ind.data(),
                                                  rocsparseio_type_float64,
                                                  rval.data(),
                                                  rocsparseio_index_base_zero,
                                                  name);
        }
        // ROCSPARSEIO_CHECK(status);

        status = rocsparseio_close(handle);
        // ROCSPARSEIO_CHECK(status);
    }
    return 0;
}

//
//
//
void usage(const char* appname_)
{
    fprintf(stderr, "NAME\n");
    fprintf(stderr, "       %s -- Convert rocSPARSE files\n", appname_);
    fprintf(stderr, "SYNOPSIS\n");
    fprintf(stderr, "       %s [OPTION]... -o <output file> \n", appname_);
    fprintf(stderr, "DESCRIPTION\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "       Convert files.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "OPTIONS\n");
    fprintf(stderr, "       -v, --verbose\n");
    fprintf(stderr, "              use verbose of information.\n");
    fprintf(stderr, "       -h, --help\n");
    fprintf(stderr, "              produces this help and exit.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "NOTES\n");
    fprintf(stderr, "\n");
}

const char* get_file_extension(const char* filename)
{
    const char* ext = filename;
    while(*ext != '\0')
        ++ext;
    while(ext != filename && *ext != '.')
        --ext;
    if(ext == filename)
        ext = nullptr;
    return ext;
}

//
// Main.
//
int main(int argc, char** argv)
{
    rocsparseio_cmdline_t cmd(argc, argv);

    if(cmd.option("-h") || cmd.option("--help"))
    {
        usage(argv[0]);
        return ROCSPARSEIO_STATUS_SUCCESS;
    }

    const bool verbose = cmd.option("-v");

    char ofilename[512];
    if(false == cmd.option("-o", ofilename))
    {
        std::cerr << "missing '-o <filename>'" << std::endl;
        return 1;
    }

    if(cmd.isempty())
    {
        std::cerr << "missing input file" << std::endl;
        return 1;
    }

    const char* ifilename = cmd.get_arg(1);
    if(verbose)
    {
        std::cout << "ifilename: '" << ifilename << "' " << std::endl;
        std::cout << "ofilename: '" << ofilename << "' " << std::endl;
    }

    //
    // Get extension.
    //
    const char* iext = get_file_extension(ifilename);

    const char* oext = get_file_extension(ofilename);

    if(verbose)
    {
        std::cout << "ifilename extension: '" << (iext ? iext : "<none>") << "' " << std::endl;
        std::cout << "ofilename extension: '" << (oext ? oext : "<none>") << "' " << std::endl;
    }

    if(!iext)
    {
        std::cerr << "missing file extension in '" << ifilename << "'" << std::endl;
    }
    if(!oext)
    {
        std::cerr << "missing file extension in '" << ofilename << "'" << std::endl;
    }

    rocsparseio::file_format_t input_file_format(iext);
    rocsparseio::file_format_t output_file_format(oext);

    if(input_file_format.is_invalid())
    {
        std::cerr << "file format in relation with file extension '" << iext
                  << "' is not recognized" << std::endl;
        std::cerr << "list of known file formats are:" << std::endl;
        for(auto file_format : rocsparseio::file_format_t::all)
        {
            std::cerr << " " << rocsparseio::file_format_t(file_format) << std::endl;
        }
        return rocsparseio_status_invalid_value;
    }

    if(output_file_format.is_invalid())
    {
        std::cerr << "file format in relation with file extension '" << oext
                  << "' is not recoginized" << std::endl;
        std::cerr << "list of known file formats are:" << std::endl;
        for(auto file_format : rocsparseio::file_format_t::all)
        {
            std::cerr << " ." << rocsparseio::file_format_t(file_format) << std::endl;
        }
        return rocsparseio_status_invalid_value;
    }

    switch(input_file_format)
    {
    case rocsparseio::file_format_t::ascii:
    {
        std::cerr << "not implemented LINE " << __LINE__ << std::endl;
        return rocsparseio_status_invalid_value;
    }

    case rocsparseio::file_format_t::csr:
    {
        switch(output_file_format)
        {
        case rocsparseio::file_format_t::csr:
        {
            //
            // copy file.
            //
            std::cerr << "not implemented LINE " << __LINE__ << std::endl;
            return rocsparseio_status_invalid_value;
        }

        case rocsparseio::file_format_t::mtx:
        {
            //
            // convert csr 2 mtx
            //
            std::cerr << "not implemented LINE " << __LINE__ << std::endl;
            return rocsparseio_status_invalid_value;
        }

        case rocsparseio::file_format_t::ascii:
        {
            return csr2ascii(ifilename, ofilename);
        }

        case rocsparseio::file_format_t::rocsparseio:
        {
            //
            // convert csr 2 rocsparseio
            //
            return csr2rocsparseio(ifilename, ofilename);
        }
        }
        break;
    }

    case rocsparseio::file_format_t::mtx:
    {
        switch(output_file_format)
        {
        case rocsparseio::file_format_t::csr:
        {
            return mtx2csr(ifilename, ofilename);
        }
        case rocsparseio::file_format_t::mtx:
        {
            //
            // copy file.
            //
            std::cerr << "not implemented LINE " << __LINE__ << std::endl;
            return rocsparseio_status_invalid_value;
        }
        case rocsparseio::file_format_t::ascii:
        {
            std::cerr << "not implemented LINE " << __LINE__ << std::endl;
            return rocsparseio_status_invalid_value;
        }
        case rocsparseio::file_format_t::rocsparseio:
        {
            return mtx2rocsparseio(ifilename, ofilename);
        }
        }
    }
    case rocsparseio::file_format_t::rocsparseio:
    {
        switch(output_file_format)
        {
        case rocsparseio::file_format_t::csr:
        {
            return rocsparseio2csr(ifilename, ofilename);
        }
        case rocsparseio::file_format_t::mtx:
        {
            std::cerr << "not implemented LINE " << __LINE__ << std::endl;
            return rocsparseio_status_invalid_value;
        }
        case rocsparseio::file_format_t::ascii:
        {
            return rocsparseio2ascii(ifilename, ofilename);
        }
        case rocsparseio::file_format_t::rocsparseio:
        {
            //
            // copy file.
            //
            std::cerr << "not implemented LINE " << __LINE__ << std::endl;
            return rocsparseio_status_invalid_value;
        }
        }
    }
    }
    std::cerr << "not implemented LINE " << __LINE__ << std::endl;
    return rocsparseio_status_invalid_format;
}
