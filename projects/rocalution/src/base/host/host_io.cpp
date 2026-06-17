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

#include "host_io.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../../utils/rocsparseio.h"
#include "rocalution/version.hpp"

#include <cinttypes>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace rocalution
{

    struct mm_banner
    {
        char array_type[64];
        char matrix_type[64];
        char storage_type[64];
    };

    bool mm_read_banner(FILE* fin, mm_banner& b)
    {
        char line[1025];

        // Read banner line
        if(!fgets(line, 1025, fin))
        {
            return false;
        }

        char banner[64];
        char mtx[64];

        // Read 5 tokens from banner
        if(sscanf(line, "%s %s %s %s %s", banner, mtx, b.array_type, b.matrix_type, b.storage_type)
           != 5)
        {
            return false;
        }

        // clang-format off
        // Convert to lower case
        for(char *p = mtx;            *p != '\0'; *p = tolower(*p), ++p);
        for(char *p = b.array_type;   *p != '\0'; *p = tolower(*p), ++p);
        for(char *p = b.matrix_type;  *p != '\0'; *p = tolower(*p), ++p);
        for(char *p = b.storage_type; *p != '\0'; *p = tolower(*p), ++p);
        // clang-format on

        // Check banner
        if(strncmp(banner, "%%MatrixMarket", 14))
        {
            return false;
        }

        // Check array type
        if(strncmp(mtx, "matrix", 6))
        {
            return false;
        }

        // Check array type
        if(strncmp(b.array_type, "coordinate", 10))
        {
            return false;
        }

        // Check matrix type
        if(strncmp(b.matrix_type, "real", 4) && strncmp(b.matrix_type, "complex", 7)
           && strncmp(b.matrix_type, "integer", 7) && strncmp(b.matrix_type, "pattern", 7))
        {
            return false;
        }

        // Check storage type
        if(strncmp(b.storage_type, "general", 7) && strncmp(b.storage_type, "symmetric", 9)
           && strncmp(b.storage_type, "hermitian", 9))
        {
            return false;
        }

        return true;
    }

    template <typename ValueType,
              typename std::enable_if<std::is_same<ValueType, float>::value
                                          || std::is_same<ValueType, double>::value,
                                      int>::type
              = 0>
    static ValueType read_complex(double real, double imag)
    {
        return static_cast<ValueType>(real);
    }

    template <typename ValueType,
              typename std::enable_if<std::is_same<ValueType, std::complex<float>>::value
                                          || std::is_same<ValueType, std::complex<double>>::value,
                                      int>::type
              = 0>
    static ValueType read_complex(double real, double imag)
    {
        return ValueType(real, imag);
    }

    template <typename ValueType>
    bool mm_read_coordinate(FILE*       fin,
                            mm_banner&  b,
                            int&        nrow,
                            int&        ncol,
                            int64_t&    nnz,
                            int**       row,
                            int**       col,
                            ValueType** val)
    {
        char line[1025];

        // Skip banner and comments
        do
        {
            // Check for EOF
            if(!fgets(line, 1025, fin))
            {
                return false;
            }
        } while(line[0] == '%');

        // Read m, n, nnz
        while(sscanf(line, "%d %d %" SCNd64, &nrow, &ncol, &nnz) != 3)
        {
            // Check for EOF and loop until line with 3 integer entries found
            if(!fgets(line, 1025, fin))
            {
                return false;
            }
        }

        // Allocate arrays
        allocate_host(nnz, row);
        allocate_host(nnz, col);
        allocate_host(nnz, val);

        // Read data
        if(!strncmp(b.matrix_type, "complex", 7))
        {
            double real, imag;
            for(int64_t i = 0; i < nnz; ++i)
            {
                if(fscanf(fin, "%d %d %lg %lg", (*row) + i, (*col) + i, &real, &imag) != 4)
                {
                    return false;
                }
                --(*row)[i];
                --(*col)[i];
                (*val)[i] = read_complex<ValueType>(real, imag);
            }
        }
        else if(!strncmp(b.matrix_type, "real", 4) || !strncmp(b.matrix_type, "integer", 7))
        {
            double tmp;
            for(int64_t i = 0; i < nnz; ++i)
            {
                if(fscanf(fin, "%d %d %lg\n", (*row) + i, (*col) + i, &tmp) != 3)
                {
                    return false;
                }
                --(*row)[i];
                --(*col)[i];
                (*val)[i] = read_complex<ValueType>(tmp, tmp);
            }
        }
        else if(!strncmp(b.matrix_type, "pattern", 7))
        {
            for(int64_t i = 0; i < nnz; ++i)
            {
                if(fscanf(fin, "%d %d\n", (*row) + i, (*col) + i) != 2)
                {
                    return false;
                }
                --(*row)[i];
                --(*col)[i];
                (*val)[i] = static_cast<ValueType>(1);
            }
        }
        else
        {
            return false;
        }

        // Expand symmetric matrix
        if(strncmp(b.storage_type, "general", 7))
        {
            // Count diagonal entries
            int ndiag = 0;
            for(int64_t i = 0; i < nnz; ++i)
            {
                if((*row)[i] == (*col)[i])
                {
                    ++ndiag;
                }
            }

            int64_t tot_nnz = (nnz - ndiag) * 2 + ndiag;

            // Allocate memory
            int*       sym_row = *row;
            int*       sym_col = *col;
            ValueType* sym_val = *val;

            *row = NULL;
            *col = NULL;
            *val = NULL;

            allocate_host(tot_nnz, row);
            allocate_host(tot_nnz, col);
            allocate_host(tot_nnz, val);

            int64_t idx = 0;
            for(int64_t i = 0; i < nnz; ++i)
            {
                (*row)[idx] = sym_row[i];
                (*col)[idx] = sym_col[i];
                (*val)[idx] = sym_val[i];
                ++idx;

                // Do not write diagonal again
                if(sym_row[i] != sym_col[i])
                {
                    (*row)[idx] = sym_col[i];
                    (*col)[idx] = sym_row[i];
                    (*val)[idx] = sym_val[i];
                    ++idx;
                }
            }

            if(idx != tot_nnz)
            {
                return false;
            }

            nnz = tot_nnz;

            free_host(&sym_row);
            free_host(&sym_col);
            free_host(&sym_val);
        }

        return true;
    }

    template <typename ValueType>
    bool read_matrix_mtx(int&        nrow,
                         int&        ncol,
                         int64_t&    nnz,
                         int**       row,
                         int**       col,
                         ValueType** val,
                         const char* filename)
    {
        FILE* file = fopen(filename, "r");

        if(!file)
        {
            LOG_INFO("ReadFileMTX: cannot open file " << filename);
            return false;
        }

        // read banner
        mm_banner banner;
        if(mm_read_banner(file, banner) != true)
        {
            LOG_INFO("ReadFileMTX: invalid matrix market banner");
            return false;
        }

        if(strncmp(banner.array_type, "coordinate", 10))
        {
            return false;
        }
        else
        {
            if(mm_read_coordinate(file, banner, nrow, ncol, nnz, row, col, val) != true)
            {
                LOG_INFO("ReadFileMTX: invalid matrix data");
                return false;
            }
        }

        fclose(file);

        return true;
    }

    template <typename ValueType,
              typename std::enable_if<std::is_same<ValueType, float>::value
                                          || std::is_same<ValueType, double>::value,
                                      int>::type
              = 0>
    void write_banner(FILE* file)
    {
        char sign[3];
        strcpy(sign, "%%");

        fprintf(file, "%sMatrixMarket matrix coordinate real general\n", sign);
    }

    template <typename ValueType,
              typename std::enable_if<std::is_same<ValueType, std::complex<float>>::value
                                          || std::is_same<ValueType, std::complex<double>>::value,
                                      int>::type
              = 0>
    void write_banner(FILE* file)
    {
        char sign[3];
        strcpy(sign, "%%");

        fprintf(file, "%sMatrixMarket matrix coordinate complex general\n", sign);
    }

    template <typename ValueType,
              typename std::enable_if<std::is_same<ValueType, float>::value, int>::type = 0>
    void write_value(FILE* file, ValueType val)
    {
        fprintf(file, "%0.12g\n", val);
    }

    template <typename ValueType,
              typename std::enable_if<std::is_same<ValueType, double>::value, int>::type = 0>
    void write_value(FILE* file, ValueType val)
    {
        fprintf(file, "%0.12lg\n", val);
    }

    template <
        typename ValueType,
        typename std::enable_if<std::is_same<ValueType, std::complex<float>>::value, int>::type = 0>
    void write_value(FILE* file, ValueType val)
    {
        fprintf(file, "%0.12g %0.12g\n", val.real(), val.imag());
    }

    template <
        typename ValueType,
        typename std::enable_if<std::is_same<ValueType, std::complex<double>>::value, int>::type
        = 0>
    void write_value(FILE* file, ValueType val)
    {
        fprintf(file, "%0.12lg %0.12lg\n", val.real(), val.imag());
    }

    template <typename ValueType>
    bool write_matrix_mtx(int              nrow,
                          int              ncol,
                          int64_t          nnz,
                          const int*       row,
                          const int*       col,
                          const ValueType* val,
                          const char*      filename)
    {
        FILE* file = fopen(filename, "w");

        if(!file)
        {
            LOG_INFO("WriteFileMTX: cannot open file " << filename);
            return false;
        }

        // Write MTX banner
        write_banner<ValueType>(file);

        // Write matrix sizes
        fprintf(file, "%d %d %" PRId64 "\n", nrow, ncol, nnz);

        for(int64_t i = 0; i < nnz; ++i)
        {
            fprintf(file, "%d %d ", row[i] + 1, col[i] + 1);
            write_value(file, val[i]);
        }

        fclose(file);

        return true;
    }

    static inline void read_csr_values(std::ifstream& in, int64_t nnz, float* val)
    {
        // Temporary array to convert from double to float
        std::vector<double> tmp(nnz);

        // Read double values
        in.read((char*)tmp.data(), sizeof(double) * nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int64_t i = 0; i < nnz; ++i)
        {
            val[i] = static_cast<float>(tmp[i]);
        }
    }

    static inline void read_csr_values(std::ifstream& in, int64_t nnz, double* val)
    {
        // Read double values
        in.read((char*)val, sizeof(double) * nnz);
    }

    static inline void read_csr_values(std::ifstream& in, int64_t nnz, std::complex<float>* val)
    {
        // Temporary array to convert from complex double to complex float
        std::vector<std::complex<double>> tmp(nnz);

        // Read in double complex values
        in.read((char*)tmp.data(), sizeof(std::complex<double>) * nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int64_t i = 0; i < nnz; ++i)
        {
            val[i] = std::complex<float>(static_cast<float>(tmp[i].real()),
                                         static_cast<float>(tmp[i].imag()));
        }
    }

    static inline void read_csr_values(std::ifstream& in, int64_t nnz, std::complex<double>* val)
    {
        // Read in double complex values
        in.read((char*)val, sizeof(std::complex<double>) * nnz);
    }

    static inline void read_csr_row_ptr_32(std::ifstream& in, int64_t nrow, int* ptr)
    {
        // We can directly read into array
        in.read((char*)ptr, sizeof(int) * (nrow + 1));
    }

    static inline void read_csr_row_ptr_32(std::ifstream& in, int64_t nrow, int64_t* ptr)
    {
        // Temporary array to convert from 32 to 64 bit
        std::vector<int> tmp(nrow + 1);

        // Read 32 bit integers
        in.read((char*)tmp.data(), sizeof(int) * (nrow + 1));

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int64_t i = 0; i < nrow + 1; ++i)
        {
            ptr[i] = static_cast<int64_t>(tmp[i]);
        }
    }

    static inline void read_csr_row_ptr_64(std::ifstream& in, int64_t nrow, int* ptr)
    {
        // We cannot read without overflow, skip and throw warning
        LOG_INFO("ReadFileCSR: cannot read 64 bit sparsity pattern into 32 bit structure");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    static inline void read_csr_row_ptr_64(std::ifstream& in, int64_t nrow, int64_t* ptr)
    {
        // We can directly read into array
        in.read((char*)ptr, sizeof(int64_t) * (nrow + 1));
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool read_matrix_csr(int64_t&      nrow,
                         int64_t&      ncol,
                         int64_t&      nnz,
                         PointerType** ptr,
                         IndexType**   col,
                         ValueType**   val,
                         const char*   filename)
    {
        std::ifstream in(filename, std::ios::in | std::ios::binary);

        if(!in.is_open())
        {
            LOG_INFO("ReadFileCSR: cannot open file " << filename);
            return false;
        }

        // Header
        std::string header;
        std::getline(in, header);

        if(header != "#rocALUTION binary csr file")
        {
            LOG_INFO("ReadFileCSR: invalid rocALUTION matrix header");
            return false;
        }

        // rocALUTION version
        int version;
        in.read((char*)&version, sizeof(int));

        // Read sizes

        // We need backward compatibility
        if(version < 30000)
        {
            // 32 bit backward compatibility sizes
            int nrow32;
            int ncol32;
            int nnz32;

            // Read sizes
            in.read((char*)&nrow32, sizeof(int));
            in.read((char*)&ncol32, sizeof(int));
            in.read((char*)&nnz32, sizeof(int));

            // Convert to 64 bit
            nrow = static_cast<int64_t>(nrow32);
            ncol = static_cast<int64_t>(ncol32);
            nnz  = static_cast<int64_t>(nnz32);

            // Always read 32 bit row pointers
            int* ptr32 = NULL;

            // Allocate for row pointers
            allocate_host(nrow32 + 1, &ptr32);
            allocate_host(nrow + 1, ptr);

            // Read row pointers in 32bit
            in.read((char*)ptr32, (nrow32 + 1) * sizeof(int));

            // Convert to actual precision
            for(int i = 0; i < nrow32 + 1; ++i)
            {
                (*ptr)[i] = ptr32[i];
            }

            free_host(&ptr32);
        }
        else
        {
            // Read sizes
            in.read((char*)&nrow, sizeof(int64_t));
            in.read((char*)&ncol, sizeof(int64_t));
            in.read((char*)&nnz, sizeof(int64_t));

            // Allocate row pointer array
            allocate_host(nrow + 1, ptr);

            // Read data - precision is determined by nnz
            if(nnz < std::numeric_limits<int>::max())
            {
                read_csr_row_ptr_32(in, nrow, *ptr);
            }
            else
            {
                read_csr_row_ptr_64(in, nrow, *ptr);
            }
        }

        //        if(version != __ROCALUTION_VER)
        //        {
        //            LOG_INFO("ReadFileCSR: file version mismatch");
        //            return false;
        //        }

        // Allocate arrays
        allocate_host(nnz, col);
        allocate_host(nnz, val);

        // Read in columns and values
        in.read((char*)*col, nnz * sizeof(int));
        read_csr_values(in, nnz, *val);

        // Check ifstream status
        if(!in)
        {
            LOG_INFO("ReadFileCSR: invalid matrix data");
            return false;
        }

        in.close();

        return true;
    }

    template <typename T>
    rocsparseio_type type2rocsparseio_type();

    template <>
    rocsparseio_type type2rocsparseio_type<std::complex<float>>()
    {
        return rocsparseio_type_complex32;
    }

    template <>
    rocsparseio_type type2rocsparseio_type<std::complex<double>>()
    {
        return rocsparseio_type_complex64;
    }

    template <>
    rocsparseio_type type2rocsparseio_type<float>()
    {
        return rocsparseio_type_float32;
    }

    template <>
    rocsparseio_type type2rocsparseio_type<double>()
    {
        return rocsparseio_type_float64;
    }

    template <>
    rocsparseio_type type2rocsparseio_type<int32_t>()
    {
        return rocsparseio_type_int32;
    }

    template <>
    rocsparseio_type type2rocsparseio_type<int8_t>()
    {
        return rocsparseio_type_int8;
    }

    template <>
    rocsparseio_type type2rocsparseio_type<int64_t>()
    {
        return rocsparseio_type_int64;
    }

    template <typename X, typename Y>
    inline void copy_mixed_arrays(size_t size, X* __restrict__ x, const Y* __restrict__ y)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < size; ++i)
        {
            x[i] = static_cast<X>(y[i]);
        }
    }

    template <>
    inline void copy_mixed_arrays(size_t size,
                                  std::complex<float>* __restrict__ x,
                                  const std::complex<double>* __restrict__ y)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < size; ++i)
        {
            x[i] = std::complex<float>(static_cast<float>(std::real(y[i])),
                                       static_cast<float>(std::imag(y[i])));
        }
    }

    template <>
    inline void copy_mixed_arrays(size_t size,
                                  std::complex<double>* __restrict__ x,
                                  const std::complex<float>* __restrict__ y)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < size; ++i)
        {
            x[i] = std::complex<double>(static_cast<double>(std::real(y[i])),
                                        static_cast<double>(std::imag(y[i])));
        }
    }

    template <>
    inline void copy_mixed_arrays(size_t size,
                                  int8_t* __restrict__ x,
                                  const std::complex<float>* __restrict__ y)
    {
        throw 1;
    }

    template <>
    inline void copy_mixed_arrays(size_t size,
                                  int8_t* __restrict__ x,
                                  const std::complex<double>* __restrict__ y)
    {
        throw 1;
    }

    template <>
    inline void copy_mixed_arrays(size_t size,
                                  float* __restrict__ x,
                                  const std::complex<float>* __restrict__ y)
    {
        throw 1;
    }

    template <>
    inline void copy_mixed_arrays(size_t size,
                                  float* __restrict__ x,
                                  const std::complex<double>* __restrict__ y)
    {
        throw 1;
    }

    template <>
    inline void copy_mixed_arrays(size_t size,
                                  double* __restrict__ x,
                                  const std::complex<float>* __restrict__ y)
    {
        throw 1;
    }

    template <>
    inline void copy_mixed_arrays(size_t size,
                                  double* __restrict__ x,
                                  const std::complex<double>* __restrict__ y)
    {
        throw 1;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool read_matrix_csr_rocsparseio(int64_t&      nrow,
                                     int64_t&      ncol,
                                     int64_t&      nnz,
                                     PointerType** ptr,
                                     IndexType**   col,
                                     ValueType**   val,
                                     const char*   filename)
    {
        rocsparseio_handle handle;
        rocsparseio_status status;

        //
        // Open handle.
        //
        status = rocsparseio_open(&handle, rocsparseio_rwmode_read, filename);
        if(status != rocsparseio_status_success)
        {
            LOG_INFO("ReadFileRSIO: cannot open file " << filename);
            return false;
        }

        rocsparseio_direction  file_direction;
        rocsparseio_index_base file_base;
        uint64_t               file_nrow;
        uint64_t               file_ncol;
        uint64_t               file_nnz;
        rocsparseio_type       file_ptr_type;
        rocsparseio_type       file_ind_type;
        rocsparseio_type       file_val_type;

        //
        // Read meta-data from file.
        //
        status = rocsparseiox_read_metadata_sparse_csx(handle,
                                                       &file_direction,
                                                       &file_nrow,
                                                       &file_ncol,
                                                       &file_nnz,
                                                       &file_ptr_type,
                                                       &file_ind_type,
                                                       &file_val_type,
                                                       &file_base);
        if(status != rocsparseio_status_success)
        {
            LOG_INFO("ReadFileRSIO: rocsparseiox_read_metadata_sparse_csx failed");
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check we have a CSR matrix in hand.
        //
        if(file_direction != rocsparseio_direction_row)
        {
            LOG_INFO("ReadFileRSIO: the matrix is stored with a CSC format.");
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check file_nrow fits with int64_t.
        //
        if(file_nrow > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: nrow from file exceeds int64_t limit, nrow = " << file_nrow);
            rocsparseio_close(handle);
            return false;
        }
        nrow = file_nrow;

        //
        // Check file_ncol fits with int64_t.
        //
        if(file_ncol > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: ncol from file exceeds int64_t limit, ncol = " << file_ncol);
            rocsparseio_close(handle);
            return false;
        }
        ncol = file_ncol;

        //
        // Check file_nnz fits with int64_t.
        //
        if(file_nnz > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: nnz from file exceeds int64_t limit, nnz = " << file_nnz);
            rocsparseio_close(handle);
            return false;
        }
        nnz = file_nnz;

        //
        // Check nnz fits with PointerType.
        //
        if(nnz > std::numeric_limits<PointerType>::max())
        {
            LOG_INFO("ReadFileRSIO: nnz exceeds PointerType limit, nnz = " << nnz);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check ncol fits with IndexType.
        //
        if(ncol > std::numeric_limits<IndexType>::max())
        {
            LOG_INFO("ReadFileRSIO: ncol exceeds PointerType limit, ncol = " << ncol);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check nrow fits with IndexType.
        //
        if(nrow > std::numeric_limits<IndexType>::max())
        {
            LOG_INFO("ReadFileRSIO: nrow exceeds PointerType limit, nrow = " << nrow);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Allocate.
        //
        allocate_host(nrow + 1, ptr);
        allocate_host(nnz, col);
        allocate_host(nnz, val);

        const rocsparseio_type required_ptr_type = type2rocsparseio_type<PointerType>();
        const rocsparseio_type required_ind_type = type2rocsparseio_type<IndexType>();
        const rocsparseio_type required_val_type = type2rocsparseio_type<ValueType>();

        const bool same_ptr_type = (file_ptr_type == required_ptr_type);
        const bool same_ind_type = (file_ind_type == required_ind_type);
        const bool same_val_type = (file_val_type == required_val_type);
        const bool is_consistent = same_ptr_type && same_ind_type && same_val_type;

        if(is_consistent)
        {
            status = rocsparseiox_read_sparse_csx(handle, *ptr, *col, *val);
            if(status != rocsparseio_status_success)
            {
                LOG_INFO("ReadFileRSIO: rocsparseiox_read_sparse_csx failed");
                free_host(ptr);
                free_host(col);
                free_host(val);
                rocsparseio_close(handle);
                return false;
            }
        }
        else
        {
            void* tmp_ptr = (void*)ptr[0];
            void* tmp_ind = (void*)col[0];
            void* tmp_val = (void*)val[0];
            if(!same_ptr_type)
            {
                size_t sizeof_ptr_type;
                status  = rocsparseio_type_get_size(file_ptr_type, &sizeof_ptr_type);
                tmp_ptr = malloc((nrow + 1) * sizeof_ptr_type);
            }

            if(!same_ind_type)
            {
                size_t sizeof_ind_type;
                status  = rocsparseio_type_get_size(file_ind_type, &sizeof_ind_type);
                tmp_ind = malloc(nnz * sizeof_ind_type);
            }

            if(!same_val_type)
            {
                size_t sizeof_val_type;
                status  = rocsparseio_type_get_size(file_val_type, &sizeof_val_type);
                tmp_val = malloc(nnz * sizeof_val_type);
            }

            status = rocsparseiox_read_sparse_csx(handle, tmp_ptr, tmp_ind, tmp_val);
            if(status != rocsparseio_status_success)
            {
                LOG_INFO("ReadFileRSIO: rocsparseiox_read_sparse_csx failed");
                free_host(ptr);
                free_host(col);
                free_host(val);
                rocsparseio_close(handle);
                return false;
            }

            //
            // Copy ptr type.
            //
            if(!same_ptr_type)
            {
                switch(file_ptr_type)
                {
                case rocsparseio_type_int32:
                {
                    copy_mixed_arrays(nrow + 1, ptr[0], (const int32_t*)tmp_ptr);
                    break;
                }

                case rocsparseio_type_int64:
                {
                    copy_mixed_arrays(nrow + 1, ptr[0], (const int64_t*)tmp_ptr);
                    break;
                }

                case rocsparseio_type_int8:
                case rocsparseio_type_float32:
                case rocsparseio_type_float64:
                case rocsparseio_type_complex32:
                case rocsparseio_type_complex64:
                {
                    break;
                }
                }
            }

            //
            // Copy ind type.
            //
            if(!same_ind_type)
            {
                switch(file_ind_type)
                {
                case rocsparseio_type_int32:
                {
                    copy_mixed_arrays(nnz, col[0], (const int32_t*)tmp_ind);
                    break;
                }
                case rocsparseio_type_int64:
                {
                    copy_mixed_arrays(nnz, col[0], (const int64_t*)tmp_ind);
                    break;
                }
                case rocsparseio_type_int8:
                case rocsparseio_type_float32:
                case rocsparseio_type_float64:
                case rocsparseio_type_complex32:
                case rocsparseio_type_complex64:
                {
                    break;
                }
                }
            }

            if(!same_val_type)
            {
                switch(file_val_type)
                {
                case rocsparseio_type_int32:
                case rocsparseio_type_int64:
                {
                    break;
                }

                case rocsparseio_type_int8:
                {
                    copy_mixed_arrays(nnz, val[0], (const int8_t*)tmp_val);
                    break;
                }

                case rocsparseio_type_float32:
                {
                    copy_mixed_arrays(nnz, val[0], (const float*)tmp_val);
                    break;
                }

                case rocsparseio_type_float64:
                {
                    copy_mixed_arrays(nnz, val[0], (const double*)tmp_val);
                    break;
                }

                case rocsparseio_type_complex32:
                {
                    copy_mixed_arrays(nnz, val[0], (const std::complex<float>*)tmp_val);
                }

                case rocsparseio_type_complex64:
                {
                    copy_mixed_arrays(nnz, val[0], (const std::complex<double>*)tmp_val);
                    break;
                }
                }
            }

            if(!same_ptr_type)
            {
                free(tmp_ptr);
            }

            if(!same_ind_type)
            {
                free(tmp_ind);
            }

            if(!same_val_type)
            {
                free(tmp_val);
            }
        }

        status = rocsparseio_close(handle);
        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool read_matrix_mcsr_rocsparseio(int64_t&      nrow,
                                      int64_t&      ncol,
                                      int64_t&      nnz,
                                      PointerType** ptr,
                                      IndexType**   col,
                                      ValueType**   val,
                                      const char*   filename)
    {
        rocsparseio_handle handle;
        rocsparseio_status status;

        //
        // Open handle.
        //
        status = rocsparseio_open(&handle, rocsparseio_rwmode_read, filename);
        if(status != rocsparseio_status_success)
        {
            LOG_INFO("ReadFileRSIO: cannot open file " << filename);
            return false;
        }

        rocsparseio_direction  file_direction;
        rocsparseio_index_base file_base;
        uint64_t               file_nrow;
        uint64_t               file_ncol;
        uint64_t               file_nnz;
        rocsparseio_type       file_ptr_type;
        rocsparseio_type       file_ind_type;
        rocsparseio_type       file_val_type;

        //
        // Read meta-data from file.
        //
        status = rocsparseiox_read_metadata_sparse_mcsx(handle,
                                                        &file_direction,
                                                        &file_nrow,
                                                        &file_ncol,
                                                        &file_nnz,
                                                        &file_ptr_type,
                                                        &file_ind_type,
                                                        &file_val_type,
                                                        &file_base);
        if(status != rocsparseio_status_success)
        {
            LOG_INFO("ReadFileRSIO: rocsparseiox_read_metadata_sparse_mcsx failed");
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check we have a CSR matrix in hand.
        //
        if(file_direction != rocsparseio_direction_row)
        {
            LOG_INFO("ReadFileRSIO: the matrix is stored with a CSC format.");
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check file_nrow fits with int64_t.
        //
        if(file_nrow > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: nrow from file exceeds int64_t limit, nrow = " << file_nrow);
            rocsparseio_close(handle);
            return false;
        }
        nrow = file_nrow;

        //
        // Check file_ncol fits with int64_t.
        //
        if(file_ncol > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: ncol from file exceeds int64_t limit, ncol = " << file_ncol);
            rocsparseio_close(handle);
            return false;
        }
        ncol = file_ncol;

        //
        // Check file_nnz fits with int64_t.
        //
        if(file_nnz > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: nnz from file exceeds int64_t limit, nnz = " << file_nnz);
            rocsparseio_close(handle);
            return false;
        }
        nnz = file_nnz;

        //
        // Check nnz fits with PointerType.
        //
        if(nnz > std::numeric_limits<PointerType>::max())
        {
            LOG_INFO("ReadFileRSIO: nnz exceeds PointerType limit, nnz = " << nnz);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check ncol fits with IndexType.
        //
        if(ncol > std::numeric_limits<IndexType>::max())
        {
            LOG_INFO("ReadFileRSIO: ncol exceeds PointerType limit, ncol = " << ncol);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check nrow fits with IndexType.
        //
        if(nrow > std::numeric_limits<IndexType>::max())
        {
            LOG_INFO("ReadFileRSIO: nrow exceeds PointerType limit, nrow = " << nrow);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Allocate.
        //
        allocate_host(nrow + 1, ptr);
        allocate_host(nnz, col);
        allocate_host(nnz, val);

        const rocsparseio_type required_ptr_type = type2rocsparseio_type<PointerType>();
        const rocsparseio_type required_ind_type = type2rocsparseio_type<IndexType>();
        const rocsparseio_type required_val_type = type2rocsparseio_type<ValueType>();

        const bool same_ptr_type = (file_ptr_type == required_ptr_type);
        const bool same_ind_type = (file_ind_type == required_ind_type);
        const bool same_val_type = (file_val_type == required_val_type);
        const bool is_consistent = same_ptr_type && same_ind_type && same_val_type;

        if(is_consistent)
        {
            status = rocsparseiox_read_sparse_mcsx(handle, *ptr, *col, *val);
            if(status != rocsparseio_status_success)
            {
                LOG_INFO("ReadFileRSIO: rocsparseiox_read_sparse_mcsx failed");
                free_host(ptr);
                free_host(col);
                free_host(val);
                rocsparseio_close(handle);
                return false;
            }
        }
        else
        {
            void* tmp_ptr = (void*)ptr[0];
            void* tmp_ind = (void*)col[0];
            void* tmp_val = (void*)val[0];
            if(!same_ptr_type)
            {
                size_t sizeof_ptr_type;
                status  = rocsparseio_type_get_size(file_ptr_type, &sizeof_ptr_type);
                tmp_ptr = malloc((nrow + 1) * sizeof_ptr_type);
            }

            if(!same_ind_type)
            {
                size_t sizeof_ind_type;
                status  = rocsparseio_type_get_size(file_ind_type, &sizeof_ind_type);
                tmp_ind = malloc(nnz * sizeof_ind_type);
            }

            if(!same_val_type)
            {
                size_t sizeof_val_type;
                status  = rocsparseio_type_get_size(file_val_type, &sizeof_val_type);
                tmp_val = malloc(nnz * sizeof_val_type);
            }

            status = rocsparseiox_read_sparse_mcsx(handle, tmp_ptr, tmp_ind, tmp_val);
            if(status != rocsparseio_status_success)
            {
                LOG_INFO("ReadFileRSIO: rocsparseiox_read_sparse_mcsx failed");
                free_host(ptr);
                free_host(col);
                free_host(val);
                rocsparseio_close(handle);
                return false;
            }

            //
            // Copy ptr type.
            //
            if(!same_ptr_type)
            {
                switch(file_ptr_type)
                {
                case rocsparseio_type_int32:
                {
                    copy_mixed_arrays(nrow + 1, ptr[0], (const int32_t*)tmp_ptr);
                    break;
                }

                case rocsparseio_type_int64:
                {
                    copy_mixed_arrays(nrow + 1, ptr[0], (const int64_t*)tmp_ptr);
                    break;
                }

                case rocsparseio_type_int8:
                case rocsparseio_type_float32:
                case rocsparseio_type_float64:
                case rocsparseio_type_complex32:
                case rocsparseio_type_complex64:
                {
                    break;
                }
                }
            }

            //
            // Copy ind type.
            //
            if(!same_ind_type)
            {
                switch(file_ind_type)
                {
                case rocsparseio_type_int32:
                {
                    copy_mixed_arrays(nnz, col[0], (const int32_t*)tmp_ind);
                    break;
                }
                case rocsparseio_type_int64:
                {
                    copy_mixed_arrays(nnz, col[0], (const int64_t*)tmp_ind);
                    break;
                }
                case rocsparseio_type_int8:
                case rocsparseio_type_float32:
                case rocsparseio_type_float64:
                case rocsparseio_type_complex32:
                case rocsparseio_type_complex64:
                {
                    break;
                }
                }
            }

            if(!same_val_type)
            {
                switch(file_val_type)
                {
                case rocsparseio_type_int32:
                case rocsparseio_type_int64:
                {
                    break;
                }

                case rocsparseio_type_int8:
                {
                    copy_mixed_arrays(nnz, val[0], (const int8_t*)tmp_val);
                    break;
                }

                case rocsparseio_type_float32:
                {
                    copy_mixed_arrays(nnz, val[0], (const float*)tmp_val);
                    break;
                }

                case rocsparseio_type_float64:
                {
                    copy_mixed_arrays(nnz, val[0], (const double*)tmp_val);
                    break;
                }

                case rocsparseio_type_complex32:
                {
                    copy_mixed_arrays(nnz, val[0], (const std::complex<float>*)tmp_val);
                }

                case rocsparseio_type_complex64:
                {
                    copy_mixed_arrays(nnz, val[0], (const std::complex<double>*)tmp_val);
                    break;
                }
                }
            }

            if(!same_ptr_type)
            {
                free(tmp_ptr);
            }

            if(!same_ind_type)
            {
                free(tmp_ind);
            }

            if(!same_val_type)
            {
                free(tmp_val);
            }
        }

        status = rocsparseio_close(handle);
        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool read_matrix_bcsr_rocsparseio(int64_t&      nrowb,
                                      int64_t&      ncolb,
                                      int64_t&      nnzb,
                                      int64_t&      block_dim,
                                      PointerType** ptr,
                                      IndexType**   col,
                                      ValueType**   val,
                                      const char*   filename)
    {
        rocsparseio_handle handle;
        rocsparseio_status status;

        //
        // Open handle.
        //
        status = rocsparseio_open(&handle, rocsparseio_rwmode_read, filename);
        if(status != rocsparseio_status_success)
        {
            LOG_INFO("ReadFileRSIO: cannot open file " << filename);
            return false;
        }

        rocsparseio_direction  file_direction;
        rocsparseio_direction  file_directionb;
        rocsparseio_index_base file_base;
        uint64_t               file_nrowb;
        uint64_t               file_ncolb;
        uint64_t               file_nnzb;
        uint64_t               file_row_block_dim;
        uint64_t               file_col_block_dim;
        rocsparseio_type       file_ptr_type;
        rocsparseio_type       file_ind_type;
        rocsparseio_type       file_val_type;

        //
        // Read meta-data from file.
        //
        status = rocsparseiox_read_metadata_sparse_gebsx(handle,
                                                         &file_direction,
                                                         &file_directionb,
                                                         &file_nrowb,
                                                         &file_ncolb,
                                                         &file_nnzb,
                                                         &file_row_block_dim,
                                                         &file_col_block_dim,
                                                         &file_ptr_type,
                                                         &file_ind_type,
                                                         &file_val_type,
                                                         &file_base);
        if(status != rocsparseio_status_success)
        {
            LOG_INFO("ReadFileRSIO: rocsparseiox_read_metadata_sparse_gebsx failed");
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check we have a GEBSR matrix in hand.
        //
        if(file_direction != rocsparseio_direction_row
           || file_directionb != rocsparseio_direction_row)
        {
            LOG_INFO("ReadFileRSIO: the matrix is stored with a GEBSC format.");
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check we have a BSR matrix in hand
        //
        if(file_row_block_dim != file_col_block_dim)
        {
            LOG_INFO("ReadFileRSIO: the matrix BSR blocks are not squared.");
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check file_row_block_dim fits with int64_t
        //
        if(file_row_block_dim > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: row_block_dim from file exceeds int limit, row_block_dim = "
                     << file_row_block_dim);
            rocsparseio_close(handle);
            return false;
        }
        block_dim = file_row_block_dim;

        //
        // Check file_nrowb fits with int64_t.
        //
        if(file_nrowb > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: nrowb from file exceeds int64_t limit, nrowb = " << file_nrowb);
            rocsparseio_close(handle);
            return false;
        }
        nrowb = file_nrowb;

        //
        // Check file_ncolb fits with int64_t.
        //
        if(file_ncolb > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: ncolb from file exceeds int64_t limit, ncolb = " << file_ncolb);
            rocsparseio_close(handle);
            return false;
        }
        ncolb = file_ncolb;

        //
        // Check file_nnz fits with int64_t.
        //
        if(file_nnzb > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: nnzb from file exceeds int64_t limit, nnzb = " << file_nnzb);
            rocsparseio_close(handle);
            return false;
        }
        nnzb = file_nnzb;

        //
        // Check nnzb fits with PointerType.
        //
        if(nnzb > std::numeric_limits<PointerType>::max())
        {
            LOG_INFO("ReadFileRSIO: nnzb exceeds PointerType limit, nnzb = " << nnzb);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check ncolb fits with IndexType.
        //
        if(ncolb > std::numeric_limits<IndexType>::max())
        {
            LOG_INFO("ReadFileRSIO: ncolb exceeds PointerType limit, ncolb = " << ncolb);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check nrowb fits with IndexType.
        //
        if(nrowb > std::numeric_limits<IndexType>::max())
        {
            LOG_INFO("ReadFileRSIO: nrowb exceeds PointerType limit, nrowb = " << nrowb);
            rocsparseio_close(handle);
            return false;
        }

        // Full BCSR nnz
        int64_t nnz_per_block = block_dim * block_dim;
        int64_t nnz           = nnzb * nnz_per_block;

        // Check if total nnz fits with int64_t
        if((block_dim != 0 && nnz_per_block / block_dim != block_dim)
           || (nnzb != 0 && nnz / nnz_per_block != nnzb))
        {
            LOG_INFO("ReadFileRSIO: nnz exceeds int64_t limits, nnz = " << nnz);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Allocate.
        //
        allocate_host(nrowb + 1, ptr);
        allocate_host(nnzb, col);
        allocate_host(nnz, val);

        const rocsparseio_type required_ptr_type = type2rocsparseio_type<PointerType>();
        const rocsparseio_type required_ind_type = type2rocsparseio_type<IndexType>();
        const rocsparseio_type required_val_type = type2rocsparseio_type<ValueType>();

        const bool same_ptr_type = (file_ptr_type == required_ptr_type);
        const bool same_ind_type = (file_ind_type == required_ind_type);
        const bool same_val_type = (file_val_type == required_val_type);
        const bool is_consistent = same_ptr_type && same_ind_type && same_val_type;

        if(is_consistent)
        {
            status = rocsparseiox_read_sparse_gebsx(handle, *ptr, *col, *val);
            if(status != rocsparseio_status_success)
            {
                LOG_INFO("ReadFileRSIO: rocsparseiox_read_sparse_gebsx failed");
                free_host(ptr);
                free_host(col);
                free_host(val);
                rocsparseio_close(handle);
                return false;
            }
        }
        else
        {
            void* tmp_ptr = (void*)ptr[0];
            void* tmp_ind = (void*)col[0];
            void* tmp_val = (void*)val[0];
            if(!same_ptr_type)
            {
                size_t sizeof_ptr_type;
                status  = rocsparseio_type_get_size(file_ptr_type, &sizeof_ptr_type);
                tmp_ptr = malloc((nrowb + 1) * sizeof_ptr_type);
            }

            if(!same_ind_type)
            {
                size_t sizeof_ind_type;
                status  = rocsparseio_type_get_size(file_ind_type, &sizeof_ind_type);
                tmp_ind = malloc(nnzb * sizeof_ind_type);
            }

            if(!same_val_type)
            {
                size_t sizeof_val_type;
                status  = rocsparseio_type_get_size(file_val_type, &sizeof_val_type);
                tmp_val = malloc(nnz * sizeof_val_type);
            }

            status = rocsparseiox_read_sparse_gebsx(handle, tmp_ptr, tmp_ind, tmp_val);
            if(status != rocsparseio_status_success)
            {
                LOG_INFO("ReadFileRSIO: rocsparseiox_read_sparse_gebsx failed");
                free_host(ptr);
                free_host(col);
                free_host(val);
                rocsparseio_close(handle);
                return false;
            }

            //
            // Copy ptr type.
            //
            if(!same_ptr_type)
            {
                switch(file_ptr_type)
                {
                case rocsparseio_type_int32:
                {
                    copy_mixed_arrays(nrowb + 1, ptr[0], (const int32_t*)tmp_ptr);
                    break;
                }

                case rocsparseio_type_int64:
                {
                    copy_mixed_arrays(nrowb + 1, ptr[0], (const int64_t*)tmp_ptr);
                    break;
                }

                case rocsparseio_type_int8:
                case rocsparseio_type_float32:
                case rocsparseio_type_float64:
                case rocsparseio_type_complex32:
                case rocsparseio_type_complex64:
                {
                    break;
                }
                }
            }

            //
            // Copy ind type.
            //
            if(!same_ind_type)
            {
                switch(file_ind_type)
                {
                case rocsparseio_type_int32:
                {
                    copy_mixed_arrays(nnzb, col[0], (const int32_t*)tmp_ind);
                    break;
                }
                case rocsparseio_type_int64:
                {
                    copy_mixed_arrays(nnzb, col[0], (const int64_t*)tmp_ind);
                    break;
                }
                case rocsparseio_type_int8:
                case rocsparseio_type_float32:
                case rocsparseio_type_float64:
                case rocsparseio_type_complex32:
                case rocsparseio_type_complex64:
                {
                    break;
                }
                }
            }

            if(!same_val_type)
            {
                switch(file_val_type)
                {
                case rocsparseio_type_int32:
                case rocsparseio_type_int64:
                {
                    break;
                }

                case rocsparseio_type_int8:
                {
                    copy_mixed_arrays(nnz, val[0], (const int8_t*)tmp_val);
                    break;
                }

                case rocsparseio_type_float32:
                {
                    copy_mixed_arrays(nnz, val[0], (const float*)tmp_val);
                    break;
                }

                case rocsparseio_type_float64:
                {
                    copy_mixed_arrays(nnz, val[0], (const double*)tmp_val);
                    break;
                }

                case rocsparseio_type_complex32:
                {
                    copy_mixed_arrays(nnz, val[0], (const std::complex<float>*)tmp_val);
                }

                case rocsparseio_type_complex64:
                {
                    copy_mixed_arrays(nnz, val[0], (const std::complex<double>*)tmp_val);
                    break;
                }
                }
            }

            if(!same_ptr_type)
            {
                free(tmp_ptr);
            }

            if(!same_ind_type)
            {
                free(tmp_ind);
            }

            if(!same_val_type)
            {
                free(tmp_val);
            }
        }

        status = rocsparseio_close(handle);
        return true;
    }

    template <typename ValueType, typename IndexType>
    bool read_matrix_coo_rocsparseio(int64_t&    nrow,
                                     int64_t&    ncol,
                                     int64_t&    nnz,
                                     IndexType** row,
                                     IndexType** col,
                                     ValueType** val,
                                     const char* filename)
    {
        rocsparseio_handle handle;
        rocsparseio_status status;

        //
        // Open handle.
        //
        status = rocsparseio_open(&handle, rocsparseio_rwmode_read, filename);
        if(status != rocsparseio_status_success)
        {
            LOG_INFO("ReadFileRSIO: cannot open file " << filename);
            return false;
        }

        rocsparseio_index_base file_base;
        uint64_t               file_nrow;
        uint64_t               file_ncol;
        uint64_t               file_nnz;
        rocsparseio_type       file_row_type;
        rocsparseio_type       file_ind_type;
        rocsparseio_type       file_val_type;

        //
        // Read meta-data from file.
        //
        status = rocsparseiox_read_metadata_sparse_coo(handle,
                                                       &file_nrow,
                                                       &file_ncol,
                                                       &file_nnz,
                                                       &file_row_type,
                                                       &file_ind_type,
                                                       &file_val_type,
                                                       &file_base);
        if(status != rocsparseio_status_success)
        {
            LOG_INFO("ReadFileRSIO: rocsparseiox_read_metadata_sparse_coo failed");
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check file_nrow fits with int64_t.
        //
        if(file_nrow > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: nrow from file exceeds int64_t limit, nrow = " << file_nrow);
            rocsparseio_close(handle);
            return false;
        }
        nrow = file_nrow;

        //
        // Check file_ncol fits with int64_t.
        //
        if(file_ncol > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: ncol from file exceeds int64_t limit, ncol = " << file_ncol);
            rocsparseio_close(handle);
            return false;
        }
        ncol = file_ncol;

        //
        // Check file_nnz fits with int64_t.
        //
        if(file_nnz > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: nnz from file exceeds int64_t limit, nnz = " << file_nnz);
            rocsparseio_close(handle);
            return false;
        }
        nnz = file_nnz;

        //
        // Check ncol fits with IndexType.
        //
        if(ncol > std::numeric_limits<IndexType>::max())
        {
            LOG_INFO("ReadFileRSIO: ncol exceeds PointerType limit, ncol = " << ncol);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check nrow fits with IndexType.
        //
        if(nrow > std::numeric_limits<IndexType>::max())
        {
            LOG_INFO("ReadFileRSIO: nrow exceeds PointerType limit, nrow = " << nrow);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Allocate.
        //
        allocate_host(nnz, row);
        allocate_host(nnz, col);
        allocate_host(nnz, val);

        const rocsparseio_type required_row_type = type2rocsparseio_type<IndexType>();
        const rocsparseio_type required_ind_type = type2rocsparseio_type<IndexType>();
        const rocsparseio_type required_val_type = type2rocsparseio_type<ValueType>();

        const bool same_row_type = (file_row_type == required_row_type);
        const bool same_ind_type = (file_ind_type == required_ind_type);
        const bool same_val_type = (file_val_type == required_val_type);
        const bool is_consistent = same_row_type && same_ind_type && same_val_type;

        if(is_consistent)
        {
            status = rocsparseiox_read_sparse_coo(handle, *row, *col, *val);
            if(status != rocsparseio_status_success)
            {
                LOG_INFO("ReadFileRSIO: rocsparseiox_read_sparse_coo failed");
                free_host(row);
                free_host(col);
                free_host(val);
                rocsparseio_close(handle);
                return false;
            }
        }
        else
        {
            void* tmp_row = (void*)row[0];
            void* tmp_ind = (void*)col[0];
            void* tmp_val = (void*)val[0];
            if(!same_row_type)
            {
                size_t sizeof_row_type;
                status  = rocsparseio_type_get_size(file_row_type, &sizeof_row_type);
                tmp_row = malloc(nnz * sizeof_row_type);
            }

            if(!same_ind_type)
            {
                size_t sizeof_ind_type;
                status  = rocsparseio_type_get_size(file_ind_type, &sizeof_ind_type);
                tmp_ind = malloc(nnz * sizeof_ind_type);
            }

            if(!same_val_type)
            {
                size_t sizeof_val_type;
                status  = rocsparseio_type_get_size(file_val_type, &sizeof_val_type);
                tmp_val = malloc(nnz * sizeof_val_type);
            }

            status = rocsparseiox_read_sparse_coo(handle, tmp_row, tmp_ind, tmp_val);
            if(status != rocsparseio_status_success)
            {
                LOG_INFO("ReadFileRSIO: rocsparseiox_read_sparse_coo failed");
                free_host(row);
                free_host(col);
                free_host(val);
                rocsparseio_close(handle);
                return false;
            }

            //
            // Copy row type.
            //
            if(!same_row_type)
            {
                switch(file_row_type)
                {
                case rocsparseio_type_int32:
                {
                    copy_mixed_arrays(nnz, row[0], (const int32_t*)tmp_row);
                    break;
                }

                case rocsparseio_type_int64:
                {
                    copy_mixed_arrays(nnz, row[0], (const int64_t*)tmp_row);
                    break;
                }

                case rocsparseio_type_int8:
                case rocsparseio_type_float32:
                case rocsparseio_type_float64:
                case rocsparseio_type_complex32:
                case rocsparseio_type_complex64:
                {
                    break;
                }
                }
            }

            //
            // Copy ind type.
            //
            if(!same_ind_type)
            {
                switch(file_ind_type)
                {
                case rocsparseio_type_int32:
                {
                    copy_mixed_arrays(nnz, col[0], (const int32_t*)tmp_ind);
                    break;
                }
                case rocsparseio_type_int64:
                {
                    copy_mixed_arrays(nnz, col[0], (const int64_t*)tmp_ind);
                    break;
                }
                case rocsparseio_type_int8:
                case rocsparseio_type_float32:
                case rocsparseio_type_float64:
                case rocsparseio_type_complex32:
                case rocsparseio_type_complex64:
                {
                    break;
                }
                }
            }

            if(!same_val_type)
            {
                switch(file_val_type)
                {
                case rocsparseio_type_int32:
                case rocsparseio_type_int64:
                {
                    break;
                }

                case rocsparseio_type_int8:
                {
                    copy_mixed_arrays(nnz, val[0], (const int8_t*)tmp_val);
                    break;
                }

                case rocsparseio_type_float32:
                {
                    copy_mixed_arrays(nnz, val[0], (const float*)tmp_val);
                    break;
                }

                case rocsparseio_type_float64:
                {
                    copy_mixed_arrays(nnz, val[0], (const double*)tmp_val);
                    break;
                }

                case rocsparseio_type_complex32:
                {
                    copy_mixed_arrays(nnz, val[0], (const std::complex<float>*)tmp_val);
                }

                case rocsparseio_type_complex64:
                {
                    copy_mixed_arrays(nnz, val[0], (const std::complex<double>*)tmp_val);
                    break;
                }
                }
            }

            if(!same_row_type)
            {
                free(tmp_row);
            }

            if(!same_ind_type)
            {
                free(tmp_ind);
            }

            if(!same_val_type)
            {
                free(tmp_val);
            }
        }

        status = rocsparseio_close(handle);
        return true;
    }

    template <typename ValueType, typename IndexType>
    bool read_matrix_dia_rocsparseio(int64_t&    nrow,
                                     int64_t&    ncol,
                                     int64_t&    nnz,
                                     int64_t&    ndiag,
                                     IndexType** offset,
                                     ValueType** val,
                                     const char* filename)
    {
        rocsparseio_handle handle;
        rocsparseio_status status;

        //
        // Open handle.
        //
        status = rocsparseio_open(&handle, rocsparseio_rwmode_read, filename);
        if(status != rocsparseio_status_success)
        {
            LOG_INFO("ReadFileRSIO: cannot open file " << filename);
            return false;
        }

        rocsparseio_index_base file_base;
        uint64_t               file_nrow;
        uint64_t               file_ncol;
        uint64_t               file_ndiag;
        rocsparseio_type       file_ind_type;
        rocsparseio_type       file_val_type;

        //
        // Read meta-data from file.
        //
        status = rocsparseiox_read_metadata_sparse_dia(handle,
                                                       &file_nrow,
                                                       &file_ncol,
                                                       &file_ndiag,
                                                       &file_ind_type,
                                                       &file_val_type,
                                                       &file_base);
        if(status != rocsparseio_status_success)
        {
            LOG_INFO("ReadFileRSIO: rocsparseiox_read_metadata_sparse_dia failed");
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check file_nrow fits with int64_t.
        //
        if(file_nrow > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: nrow from file exceeds int64_t limit, nrow = " << file_nrow);
            rocsparseio_close(handle);
            return false;
        }
        nrow = file_nrow;

        //
        // Check file_ncol fits with int64_t.
        //
        if(file_ncol > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: ncol from file exceeds int64_t limit, ncol = " << file_ncol);
            rocsparseio_close(handle);
            return false;
        }
        ncol = file_ncol;

        //
        // Check file_ndiag fits with int64_t
        //
        if(file_ndiag > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: nnz from file exceeds int64_t limit, nnz = " << file_ndiag);
            rocsparseio_close(handle);
            return false;
        }
        ndiag = file_ndiag;

        //
        // Check ncol fits with IndexType.
        //
        if(ncol > std::numeric_limits<IndexType>::max())
        {
            LOG_INFO("ReadFileRSIO: ncol exceeds IndexType limit, ncol = " << ncol);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check nrow fits with IndexType.
        //
        if(nrow > std::numeric_limits<IndexType>::max())
        {
            LOG_INFO("ReadFileRSIO: nrow exceeds IndexType limit, nrow = " << nrow);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check ndiag fits with IndexType
        //
        if(ndiag > std::numeric_limits<IndexType>::max())
        {
            LOG_INFO("ReadFileRSIO: ndiag exceeds IndexType limit, nrow = " << ndiag);
            rocsparseio_close(handle);
            return false;
        }

        int64_t size = std::min(nrow, ncol);

        // Full DIA nnz
        nnz = size * ndiag;

        // Check DIA nnz fits with int64_t
        if(size != 0 && nnz / size != ndiag)
        {
            LOG_INFO("ReadFileRSIO: nnz exceeds int64_t limits, nnz = " << nnz);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Allocate.
        //
        allocate_host(ndiag, offset);
        allocate_host(nnz, val);

        const rocsparseio_type required_ind_type = type2rocsparseio_type<IndexType>();
        const rocsparseio_type required_val_type = type2rocsparseio_type<ValueType>();

        const bool same_ind_type = (file_ind_type == required_ind_type);
        const bool same_val_type = (file_val_type == required_val_type);
        const bool is_consistent = same_ind_type && same_val_type;

        if(is_consistent)
        {
            status = rocsparseiox_read_sparse_dia(handle, *offset, *val);
            if(status != rocsparseio_status_success)
            {
                LOG_INFO("ReadFileRSIO: rocsparseiox_read_sparse_dia failed");
                free_host(offset);
                free_host(val);
                rocsparseio_close(handle);
                return false;
            }
        }
        else
        {
            void* tmp_ind = (void*)offset[0];
            void* tmp_val = (void*)val[0];
            if(!same_ind_type)
            {
                size_t sizeof_ind_type;
                status  = rocsparseio_type_get_size(file_ind_type, &sizeof_ind_type);
                tmp_ind = malloc(ndiag * sizeof_ind_type);
            }

            if(!same_val_type)
            {
                size_t sizeof_val_type;
                status  = rocsparseio_type_get_size(file_val_type, &sizeof_val_type);
                tmp_val = malloc(nnz * sizeof_val_type);
            }

            status = rocsparseiox_read_sparse_dia(handle, tmp_ind, tmp_val);
            if(status != rocsparseio_status_success)
            {
                LOG_INFO("ReadFileRSIO: rocsparseiox_read_sparse_dia failed");
                free_host(offset);
                free_host(val);
                rocsparseio_close(handle);
                return false;
            }

            //
            // Copy ind type.
            //
            if(!same_ind_type)
            {
                switch(file_ind_type)
                {
                case rocsparseio_type_int32:
                {
                    copy_mixed_arrays(ndiag, offset[0], (const int32_t*)tmp_ind);
                    break;
                }
                case rocsparseio_type_int64:
                {
                    copy_mixed_arrays(ndiag, offset[0], (const int64_t*)tmp_ind);
                    break;
                }
                case rocsparseio_type_int8:
                case rocsparseio_type_float32:
                case rocsparseio_type_float64:
                case rocsparseio_type_complex32:
                case rocsparseio_type_complex64:
                {
                    break;
                }
                }
            }

            if(!same_val_type)
            {
                switch(file_val_type)
                {
                case rocsparseio_type_int32:
                case rocsparseio_type_int64:
                {
                    break;
                }

                case rocsparseio_type_int8:
                {
                    copy_mixed_arrays(nnz, val[0], (const int8_t*)tmp_val);
                    break;
                }

                case rocsparseio_type_float32:
                {
                    copy_mixed_arrays(nnz, val[0], (const float*)tmp_val);
                    break;
                }

                case rocsparseio_type_float64:
                {
                    copy_mixed_arrays(nnz, val[0], (const double*)tmp_val);
                    break;
                }

                case rocsparseio_type_complex32:
                {
                    copy_mixed_arrays(nnz, val[0], (const std::complex<float>*)tmp_val);
                }

                case rocsparseio_type_complex64:
                {
                    copy_mixed_arrays(nnz, val[0], (const std::complex<double>*)tmp_val);
                    break;
                }
                }
            }

            if(!same_ind_type)
            {
                free(tmp_ind);
            }

            if(!same_val_type)
            {
                free(tmp_val);
            }
        }

        status = rocsparseio_close(handle);
        return true;
    }

    template <typename ValueType, typename IndexType>
    bool read_matrix_ell_rocsparseio(int64_t&    nrow,
                                     int64_t&    ncol,
                                     int64_t&    nnz,
                                     int64_t&    width,
                                     IndexType** col,
                                     ValueType** val,
                                     const char* filename)
    {
        rocsparseio_handle handle;
        rocsparseio_status status;

        //
        // Open handle.
        //
        status = rocsparseio_open(&handle, rocsparseio_rwmode_read, filename);
        if(status != rocsparseio_status_success)
        {
            LOG_INFO("ReadFileRSIO: cannot open file " << filename);
            return false;
        }

        rocsparseio_index_base file_base;
        uint64_t               file_nrow;
        uint64_t               file_ncol;
        uint64_t               file_width;
        rocsparseio_type       file_ind_type;
        rocsparseio_type       file_val_type;

        //
        // Read meta-data from file.
        //
        status = rocsparseiox_read_metadata_sparse_ell(handle,
                                                       &file_nrow,
                                                       &file_ncol,
                                                       &file_width,
                                                       &file_ind_type,
                                                       &file_val_type,
                                                       &file_base);
        if(status != rocsparseio_status_success)
        {
            LOG_INFO("ReadFileRSIO: rocsparseiox_read_metadata_sparse_csx failed");
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check file_nrow fits with int64_t.
        //
        if(file_nrow > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: nrow from file exceeds int64_t limit, nrow = " << file_nrow);
            rocsparseio_close(handle);
            return false;
        }
        nrow = file_nrow;

        //
        // Check file_ncol fits with int64_t.
        //
        if(file_ncol > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: ncol from file exceeds int64_t limit, ncol = " << file_ncol);
            rocsparseio_close(handle);
            return false;
        }
        ncol = file_ncol;

        //
        // Check file_width fits with int64_t.
        //
        if(file_width > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: width from file exceeds int64_t limit, width = " << file_width);
            rocsparseio_close(handle);
            return false;
        }
        width = file_width;

        //
        // Check width fits with IndexType.
        //
        if(width > std::numeric_limits<IndexType>::max())
        {
            LOG_INFO("ReadFileRSIO: nnz exceeds IndexType limit, width = " << width);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check ncol fits with IndexType.
        //
        if(ncol > std::numeric_limits<IndexType>::max())
        {
            LOG_INFO("ReadFileRSIO: ncol exceeds IndexType limit, ncol = " << ncol);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check nrow fits with IndexType.
        //
        if(nrow > std::numeric_limits<IndexType>::max())
        {
            LOG_INFO("ReadFileRSIO: nrow exceeds IndexType limit, nrow = " << nrow);
            rocsparseio_close(handle);
            return false;
        }

        // Full ELL nnz
        nnz = nrow * width;

        // Check if total nnz fits with int64_t
        if(nrow != 0 && nnz / nrow != width)
        {
            LOG_INFO("ReadFileRSIO: nnz exceeds int64_t limits, nnz = " << nnz);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Allocate.
        //
        allocate_host(nnz, col);
        allocate_host(nnz, val);

        const rocsparseio_type required_ind_type = type2rocsparseio_type<IndexType>();
        const rocsparseio_type required_val_type = type2rocsparseio_type<ValueType>();

        const bool same_ind_type = (file_ind_type == required_ind_type);
        const bool same_val_type = (file_val_type == required_val_type);
        const bool is_consistent = same_ind_type && same_val_type;

        if(is_consistent)
        {
            status = rocsparseiox_read_sparse_ell(handle, *col, *val);
            if(status != rocsparseio_status_success)
            {
                LOG_INFO("ReadFileRSIO: rocsparseiox_read_sparse_csx failed");
                free_host(col);
                free_host(val);
                rocsparseio_close(handle);
                return false;
            }
        }
        else
        {
            void* tmp_ind = (void*)col[0];
            void* tmp_val = (void*)val[0];
            if(!same_ind_type)
            {
                size_t sizeof_ind_type;
                status  = rocsparseio_type_get_size(file_ind_type, &sizeof_ind_type);
                tmp_ind = malloc(nnz * sizeof_ind_type);
            }

            if(!same_val_type)
            {
                size_t sizeof_val_type;
                status  = rocsparseio_type_get_size(file_val_type, &sizeof_val_type);
                tmp_val = malloc(nnz * sizeof_val_type);
            }

            status = rocsparseiox_read_sparse_ell(handle, tmp_ind, tmp_val);
            if(status != rocsparseio_status_success)
            {
                LOG_INFO("ReadFileRSIO: rocsparseiox_read_sparse_ell failed");
                free_host(col);
                free_host(val);
                rocsparseio_close(handle);
                return false;
            }

            //
            // Copy ind type.
            //
            if(!same_ind_type)
            {
                switch(file_ind_type)
                {
                case rocsparseio_type_int32:
                {
                    copy_mixed_arrays(nnz, col[0], (const int32_t*)tmp_ind);
                    break;
                }
                case rocsparseio_type_int64:
                {
                    copy_mixed_arrays(nnz, col[0], (const int64_t*)tmp_ind);
                    break;
                }
                case rocsparseio_type_int8:
                case rocsparseio_type_float32:
                case rocsparseio_type_float64:
                case rocsparseio_type_complex32:
                case rocsparseio_type_complex64:
                {
                    break;
                }
                }
            }

            if(!same_val_type)
            {
                switch(file_val_type)
                {
                case rocsparseio_type_int32:
                case rocsparseio_type_int64:
                {
                    break;
                }

                case rocsparseio_type_int8:
                {
                    copy_mixed_arrays(nnz, val[0], (const int8_t*)tmp_val);
                    break;
                }

                case rocsparseio_type_float32:
                {
                    copy_mixed_arrays(nnz, val[0], (const float*)tmp_val);
                    break;
                }

                case rocsparseio_type_float64:
                {
                    copy_mixed_arrays(nnz, val[0], (const double*)tmp_val);
                    break;
                }

                case rocsparseio_type_complex32:
                {
                    copy_mixed_arrays(nnz, val[0], (const std::complex<float>*)tmp_val);
                }

                case rocsparseio_type_complex64:
                {
                    copy_mixed_arrays(nnz, val[0], (const std::complex<double>*)tmp_val);
                    break;
                }
                }
            }

            if(!same_ind_type)
            {
                free(tmp_ind);
            }

            if(!same_val_type)
            {
                free(tmp_val);
            }
        }

        status = rocsparseio_close(handle);
        return true;
    }

    template <typename ValueType, typename IndexType>
    bool read_matrix_hyb_rocsparseio(int64_t&    nrow,
                                     int64_t&    ncol,
                                     int64_t&    nnz,
                                     int64_t&    coo_nnz,
                                     IndexType** coo_row,
                                     IndexType** coo_col,
                                     ValueType** coo_val,
                                     int64_t&    ell_nnz,
                                     int64_t&    ell_width,
                                     IndexType** ell_col,
                                     ValueType** ell_val,
                                     const char* filename)
    {
        rocsparseio_handle handle;
        rocsparseio_status status;

        //
        // Open handle.
        //
        status = rocsparseio_open(&handle, rocsparseio_rwmode_read, filename);
        if(status != rocsparseio_status_success)
        {
            LOG_INFO("ReadFileRSIO: cannot open file " << filename);
            return false;
        }

        rocsparseio_index_base file_coo_base;
        rocsparseio_index_base file_ell_base;
        uint64_t               file_nrow;
        uint64_t               file_ncol;
        uint64_t               file_coo_nnz;
        uint64_t               file_ell_width;
        rocsparseio_type       file_coo_row_type;
        rocsparseio_type       file_coo_col_type;
        rocsparseio_type       file_coo_val_type;
        rocsparseio_type       file_ell_col_type;
        rocsparseio_type       file_ell_val_type;

        //
        // Read meta-data from file.
        //
        status = rocsparseiox_read_metadata_sparse_hyb(handle,
                                                       &file_nrow,
                                                       &file_ncol,
                                                       &file_coo_nnz,
                                                       &file_coo_row_type,
                                                       &file_coo_col_type,
                                                       &file_coo_val_type,
                                                       &file_coo_base,
                                                       &file_ell_width,
                                                       &file_ell_col_type,
                                                       &file_ell_val_type,
                                                       &file_ell_base);
        if(status != rocsparseio_status_success)
        {
            LOG_INFO("ReadFileRSIO: rocsparseiox_read_metadata_sparse_hyb failed");
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check file_nrow fits with int64_t.
        //
        if(file_nrow > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: nrow from file exceeds int64_t limit, nrow = " << file_nrow);
            rocsparseio_close(handle);
            return false;
        }
        nrow = file_nrow;

        //
        // Check file_ncol fits with int64_t.
        //
        if(file_ncol > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: ncol from file exceeds int64_t limit, ncol = " << file_ncol);
            rocsparseio_close(handle);
            return false;
        }
        ncol = file_ncol;

        //
        // Check file_coo_nnz fits with int64_t.
        //
        if(file_coo_nnz > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: coo_nnz from file exceeds int64_t limit, coo_nnz = "
                     << file_coo_nnz);
            rocsparseio_close(handle);
            return false;
        }
        coo_nnz = file_coo_nnz;

        //
        // Check file_ell_width fits with int64_t.
        //
        if(file_ell_width > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: ell_width from file exceeds int64_t limit, ell_width = "
                     << file_ell_width);
            rocsparseio_close(handle);
            return false;
        }
        ell_width = file_ell_width;

        //
        // Check ncol fits with IndexType.
        //
        if(ncol > std::numeric_limits<IndexType>::max())
        {
            LOG_INFO("ReadFileRSIO: ncol exceeds IndexType limit, ncol = " << ncol);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check nrow fits with IndexType.
        //
        if(nrow > std::numeric_limits<IndexType>::max())
        {
            LOG_INFO("ReadFileRSIO: nrow exceeds IndexType limit, nrow = " << nrow);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check ell_width fits with IndexType.
        //
        if(ell_width > std::numeric_limits<IndexType>::max())
        {
            LOG_INFO("ReadFileRSIO: ell_width exceeds IndexType limit, nrow = " << ell_width);
            rocsparseio_close(handle);
            return false;
        }

        // Full ELL nnz
        ell_nnz = nrow * ell_width;

        // Check if ELL nnz fits with int64_t
        if(nrow != 0 && ell_nnz / nrow != ell_width)
        {
            LOG_INFO("ReadFileRSIO: ell_nnz exceeds int64_t limits, nnz = " << ell_nnz);
            rocsparseio_close(handle);
            return false;
        }

        // Full HYB nnz
        nnz = coo_nnz + ell_nnz;

        // Check if HYB nnz fits with int64_t
        if(coo_nnz >= 0 && ell_nnz >= 0 && nnz < 0)
        {
            LOG_INFO("ReadFileRSIO: nnz exceeds int64_t limits, nnz = " << nnz);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Allocate.
        //
        allocate_host(coo_nnz, coo_row);
        allocate_host(coo_nnz, coo_col);
        allocate_host(coo_nnz, coo_val);
        allocate_host(ell_nnz, ell_col);
        allocate_host(ell_nnz, ell_val);

        const rocsparseio_type required_ind_type = type2rocsparseio_type<IndexType>();
        const rocsparseio_type required_val_type = type2rocsparseio_type<ValueType>();

        const bool same_coo_row_type = (file_coo_row_type == required_ind_type);
        const bool same_coo_col_type = (file_coo_col_type == required_ind_type);
        const bool same_coo_val_type = (file_coo_val_type == required_val_type);
        const bool same_ell_col_type = (file_ell_col_type == required_ind_type);
        const bool same_ell_val_type = (file_ell_val_type == required_val_type);

        const bool is_consistent_coo = same_coo_row_type && same_coo_col_type && same_coo_val_type;
        const bool is_consistent_ell = same_ell_col_type && same_ell_val_type;

        const bool is_consistent = is_consistent_coo && is_consistent_ell;

        if(is_consistent)
        {
            status = rocsparseiox_read_sparse_hyb(
                handle, *coo_row, *coo_col, *coo_val, *ell_col, *ell_val);
            if(status != rocsparseio_status_success)
            {
                LOG_INFO("ReadFileRSIO: rocsparseiox_read_sparse_hyb failed");
                free_host(coo_row);
                free_host(coo_col);
                free_host(coo_val);
                free_host(ell_col);
                free_host(ell_val);
                rocsparseio_close(handle);
                return false;
            }
        }
        else
        {
            void* tmp_coo_row = (void*)coo_row[0];
            void* tmp_coo_col = (void*)coo_col[0];
            void* tmp_coo_val = (void*)coo_val[0];
            void* tmp_ell_col = (void*)ell_col[0];
            void* tmp_ell_val = (void*)ell_val[0];

            if(!same_coo_row_type)
            {
                size_t sizeof_coo_row_type;
                status      = rocsparseio_type_get_size(file_coo_row_type, &sizeof_coo_row_type);
                tmp_coo_row = malloc(coo_nnz * sizeof_coo_row_type);
            }

            if(!same_coo_col_type)
            {
                size_t sizeof_coo_col_type;
                status      = rocsparseio_type_get_size(file_coo_col_type, &sizeof_coo_col_type);
                tmp_coo_col = malloc(coo_nnz * sizeof_coo_col_type);
            }

            if(!same_coo_val_type)
            {
                size_t sizeof_coo_val_type;
                status      = rocsparseio_type_get_size(file_coo_val_type, &sizeof_coo_val_type);
                tmp_coo_val = malloc(coo_nnz * sizeof_coo_val_type);
            }

            if(!same_ell_col_type)
            {
                size_t sizeof_ell_col_type;
                status      = rocsparseio_type_get_size(file_ell_col_type, &sizeof_ell_col_type);
                tmp_ell_col = malloc(ell_nnz * sizeof_ell_col_type);
            }

            if(!same_ell_val_type)
            {
                size_t sizeof_ell_val_type;
                status      = rocsparseio_type_get_size(file_ell_val_type, &sizeof_ell_val_type);
                tmp_ell_val = malloc(ell_nnz * sizeof_ell_val_type);
            }

            status = rocsparseiox_read_sparse_hyb(
                handle, tmp_coo_row, tmp_coo_col, tmp_coo_val, tmp_ell_col, tmp_ell_val);
            if(status != rocsparseio_status_success)
            {
                LOG_INFO("ReadFileRSIO: rocsparseiox_read_sparse_hyb failed");
                free_host(coo_row);
                free_host(coo_col);
                free_host(coo_val);
                free_host(ell_col);
                free_host(ell_val);
                rocsparseio_close(handle);
                return false;
            }

            //
            // Copy coo row type.
            //
            if(!same_coo_row_type)
            {
                switch(file_coo_row_type)
                {
                case rocsparseio_type_int32:
                {
                    copy_mixed_arrays(coo_nnz, coo_row[0], (const int32_t*)tmp_coo_row);
                    break;
                }
                case rocsparseio_type_int64:
                {
                    copy_mixed_arrays(coo_nnz, coo_row[0], (const int64_t*)tmp_coo_row);
                    break;
                }
                case rocsparseio_type_int8:
                case rocsparseio_type_float32:
                case rocsparseio_type_float64:
                case rocsparseio_type_complex32:
                case rocsparseio_type_complex64:
                {
                    break;
                }
                }
            }

            //
            // Copy coo col type.
            //
            if(!same_coo_col_type)
            {
                switch(file_coo_col_type)
                {
                case rocsparseio_type_int32:
                {
                    copy_mixed_arrays(coo_nnz, coo_col[0], (const int32_t*)tmp_coo_col);
                    break;
                }
                case rocsparseio_type_int64:
                {
                    copy_mixed_arrays(coo_nnz, coo_col[0], (const int64_t*)tmp_coo_col);
                    break;
                }
                case rocsparseio_type_int8:
                case rocsparseio_type_float32:
                case rocsparseio_type_float64:
                case rocsparseio_type_complex32:
                case rocsparseio_type_complex64:
                {
                    break;
                }
                }
            }

            //
            // Copy coo val type.
            //
            if(!same_coo_val_type)
            {
                switch(file_coo_val_type)
                {
                case rocsparseio_type_int32:
                case rocsparseio_type_int64:
                {
                    break;
                }

                case rocsparseio_type_int8:
                {
                    copy_mixed_arrays(coo_nnz, coo_val[0], (const int8_t*)tmp_coo_val);
                    break;
                }

                case rocsparseio_type_float32:
                {
                    copy_mixed_arrays(coo_nnz, coo_val[0], (const float*)tmp_coo_val);
                    break;
                }

                case rocsparseio_type_float64:
                {
                    copy_mixed_arrays(coo_nnz, coo_val[0], (const double*)tmp_coo_val);
                    break;
                }

                case rocsparseio_type_complex32:
                {
                    copy_mixed_arrays(coo_nnz, coo_val[0], (const std::complex<float>*)tmp_coo_val);
                }

                case rocsparseio_type_complex64:
                {
                    copy_mixed_arrays(
                        coo_nnz, coo_val[0], (const std::complex<double>*)tmp_coo_val);
                    break;
                }
                }
            }

            //
            // Copy ell col type.
            //
            if(!same_ell_col_type)
            {
                switch(file_ell_col_type)
                {
                case rocsparseio_type_int32:
                {
                    copy_mixed_arrays(ell_nnz, ell_col[0], (const int32_t*)tmp_ell_col);
                    break;
                }
                case rocsparseio_type_int64:
                {
                    copy_mixed_arrays(ell_nnz, ell_col[0], (const int64_t*)tmp_ell_col);
                    break;
                }
                case rocsparseio_type_int8:
                case rocsparseio_type_float32:
                case rocsparseio_type_float64:
                case rocsparseio_type_complex32:
                case rocsparseio_type_complex64:
                {
                    break;
                }
                }
            }

            //
            // Copy ell val type.
            //
            if(!same_ell_val_type)
            {
                switch(file_ell_val_type)
                {
                case rocsparseio_type_int32:
                case rocsparseio_type_int64:
                {
                    break;
                }

                case rocsparseio_type_int8:
                {
                    copy_mixed_arrays(ell_nnz, ell_val[0], (const int8_t*)tmp_ell_val);
                    break;
                }

                case rocsparseio_type_float32:
                {
                    copy_mixed_arrays(ell_nnz, ell_val[0], (const float*)tmp_ell_val);
                    break;
                }

                case rocsparseio_type_float64:
                {
                    copy_mixed_arrays(ell_nnz, ell_val[0], (const double*)tmp_ell_val);
                    break;
                }

                case rocsparseio_type_complex32:
                {
                    copy_mixed_arrays(ell_nnz, ell_val[0], (const std::complex<float>*)tmp_ell_val);
                }

                case rocsparseio_type_complex64:
                {
                    copy_mixed_arrays(
                        ell_nnz, ell_val[0], (const std::complex<double>*)tmp_ell_val);
                    break;
                }
                }
            }

            if(!same_coo_row_type)
            {
                free(tmp_coo_row);
            }

            if(!same_coo_col_type)
            {
                free(tmp_coo_col);
            }

            if(!same_coo_val_type)
            {
                free(tmp_coo_val);
            }

            if(!same_ell_col_type)
            {
                free(tmp_ell_col);
            }

            if(!same_ell_val_type)
            {
                free(tmp_ell_val);
            }
        }

        status = rocsparseio_close(handle);
        return true;
    }

    template <typename ValueType>
    bool read_matrix_dense_rocsparseio(int64_t&    nrow,
                                       int64_t&    ncol,
                                       ValueType** val,
                                       const char* filename)
    {
        rocsparseio_handle handle;
        rocsparseio_status status;

        //
        // Open handle.
        //
        status = rocsparseio_open(&handle, rocsparseio_rwmode_read, filename);
        if(status != rocsparseio_status_success)
        {
            LOG_INFO("ReadFileRSIO: cannot open file " << filename);
            return false;
        }

        rocsparseio_index_base file_base;
        uint64_t               file_nrow;
        uint64_t               file_ncol;
        rocsparseio_type       file_row_type;
        rocsparseio_type       file_ind_type;
        rocsparseio_type       file_val_type;
        rocsparseio_order      file_order;

        //
        // Read meta-data from file.
        //
        status = rocsparseiox_read_metadata_dense_matrix(
            handle, &file_order, &file_nrow, &file_ncol, &file_val_type);
        if(status != rocsparseio_status_success)
        {
            LOG_INFO("ReadFileRSIO: rocsparseiox_read_metadata_dense_matrix failed");
            rocsparseio_close(handle);
            return false;
        }

        //
        // Check file_nrow fits with int64_t.
        //
        if(file_nrow > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: nrow from file exceeds int64_t limit, nrow = " << file_nrow);
            rocsparseio_close(handle);
            return false;
        }
        nrow = file_nrow;

        //
        // Check file_ncol fits with int64_t.
        //
        if(file_ncol > std::numeric_limits<int64_t>::max())
        {
            LOG_INFO("ReadFileRSIO: ncol from file exceeds int64_t limit, ncol = " << file_ncol);
            rocsparseio_close(handle);
            return false;
        }
        ncol = file_ncol;

        //
        // Check file_order is row major
        if(file_order != rocsparseio_order_row)
        {
            LOG_INFO("ReadFileRSIO: order from file is not row-major");
            rocsparseio_close(handle);
            return false;
        }

        // Full dense nnz
        int64_t nnz = nrow * ncol;

        // Check if total nnz fits with int64_t
        if(nrow != 0 && nnz / nrow != ncol)
        {
            LOG_INFO("ReadFileRSIO: nnz exceeds int64_t limits, nnz = " << nnz);
            rocsparseio_close(handle);
            return false;
        }

        //
        // Allocate.
        //
        allocate_host(nrow * ncol, val);

        const rocsparseio_type required_val_type = type2rocsparseio_type<ValueType>();

        const bool same_val_type = (file_val_type == required_val_type);

        if(same_val_type)
        {
            status = rocsparseiox_read_dense_matrix(handle, *val, nrow);
            if(status != rocsparseio_status_success)
            {
                LOG_INFO("ReadFileRSIO: rocsparseiox_read_dense_matrix failed");
                free_host(val);
                rocsparseio_close(handle);
                return false;
            }
        }
        else
        {
            void* tmp_val = (void*)val[0];
            if(!same_val_type)
            {
                size_t sizeof_val_type;
                status  = rocsparseio_type_get_size(file_val_type, &sizeof_val_type);
                tmp_val = malloc(nrow * ncol * sizeof_val_type);
            }

            status = rocsparseiox_read_dense_matrix(handle, tmp_val, nrow);
            if(status != rocsparseio_status_success)
            {
                LOG_INFO("ReadFileRSIO: rocsparseiox_read_dense_matrix failed");
                free_host(val);
                rocsparseio_close(handle);
                return false;
            }

            if(!same_val_type)
            {
                switch(file_val_type)
                {
                case rocsparseio_type_int32:
                case rocsparseio_type_int64:
                {
                    break;
                }

                case rocsparseio_type_int8:
                {
                    copy_mixed_arrays(nrow * ncol, val[0], (const int8_t*)tmp_val);
                    break;
                }

                case rocsparseio_type_float32:
                {
                    copy_mixed_arrays(nrow * ncol, val[0], (const float*)tmp_val);
                    break;
                }

                case rocsparseio_type_float64:
                {
                    copy_mixed_arrays(nrow * ncol, val[0], (const double*)tmp_val);
                    break;
                }

                case rocsparseio_type_complex32:
                {
                    copy_mixed_arrays(nrow * ncol, val[0], (const std::complex<float>*)tmp_val);
                }

                case rocsparseio_type_complex64:
                {
                    copy_mixed_arrays(nrow * ncol, val[0], (const std::complex<double>*)tmp_val);
                    break;
                }
                }
            }

            if(!same_val_type)
            {
                free(tmp_val);
            }
        }

        status = rocsparseio_close(handle);
        return true;
    }

    static inline void write_csr_values(std::ofstream& out, int64_t nnz, const float* val)
    {
        // Temporary array to convert from float to double
        std::vector<double> tmp(nnz);

        // Convert values
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int64_t i = 0; i < nnz; ++i)
        {
            tmp[i] = static_cast<double>(val[i]);
        }

        // Write double values
        out.write((char*)tmp.data(), sizeof(double) * nnz);
    }

    static inline void write_csr_values(std::ofstream& out, int64_t nnz, const double* val)
    {
        // Write double values
        out.write((char*)val, sizeof(double) * nnz);
    }

    static inline void
        write_csr_values(std::ofstream& out, int64_t nnz, const std::complex<float>* val)
    {
        // Temporary array to convert from complex float to complex double
        std::vector<std::complex<double>> tmp(nnz);

        // Convert values
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int64_t i = 0; i < nnz; ++i)
        {
            tmp[i] = std::complex<double>(static_cast<double>(val[i].real()),
                                          static_cast<double>(val[i].imag()));
        }

        // Write double complex values
        out.write((char*)tmp.data(), sizeof(std::complex<double>) * nnz);
    }

    static inline void
        write_csr_values(std::ofstream& out, int64_t nnz, const std::complex<double>* val)
    {
        // Write double complex values
        out.write((char*)val, sizeof(std::complex<double>) * nnz);
    }

    static inline void write_csr_row_ptr_32(std::ofstream& out, int64_t nrow, const int* ptr)
    {
        // We can directly write into array
        out.write((char*)ptr, (nrow + 1) * sizeof(int));
    }

    static inline void write_csr_row_ptr_32(std::ofstream& out, int64_t nrow, const int64_t* ptr)
    {
        // We have previously checked, that no entry of ptr exceeds 32 bits
        // Thus, its safe to convert to 32 bits
        std::vector<int> tmp(nrow + 1);

        // Convert integers
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int64_t i = 0; i < nrow + 1; ++i)
        {
            tmp[i] = static_cast<int>(ptr[i]);
        }

        // Write 32 bit integers
        out.write((char*)tmp.data(), sizeof(int) * (nrow + 1));
    }

    static inline void write_csr_row_ptr_64(std::ofstream& out, int64_t nrow, const int* ptr)
    {
        // Writing 32 bit data into 64 bit file makes not much sense, this should never be called
        LOG_INFO("This function should never be called");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    static inline void write_csr_row_ptr_64(std::ofstream& out, int64_t nrow, const int64_t* ptr)
    {
        // We can directly write into array
        out.write((char*)ptr, (nrow + 1) * sizeof(int64_t));
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool write_matrix_csr(int64_t            nrow,
                          int64_t            ncol,
                          int64_t            nnz,
                          const PointerType* ptr,
                          const IndexType*   col,
                          const ValueType*   val,
                          const char*        filename)
    {
        std::ofstream out(filename, std::ios::out | std::ios::binary);

        if(!out.is_open())
        {
            LOG_INFO("WriteFileCSR: cannot open file " << filename);
            return false;
        }

        // Write header
        out << "#rocALUTION binary csr file" << std::endl;

        // rocALUTION version
        int version = __ROCALUTION_VER;
        out.write((char*)&version, sizeof(int));

        // Write matrix sizes
        out.write((char*)&nrow, sizeof(int64_t));
        out.write((char*)&ncol, sizeof(int64_t));
        out.write((char*)&nnz, sizeof(int64_t));

        // Write row pointer data, depending on nnz
        if(nnz <= std::numeric_limits<int>::max())
        {
            write_csr_row_ptr_32(out, nrow, ptr);
        }
        else
        {
            write_csr_row_ptr_64(out, nrow, ptr);
        }

        // Write matrix column and value data
        out.write((char*)col, nnz * sizeof(int));

        write_csr_values(out, nnz, val);

        // Check ofstream status
        if(!out)
        {
            LOG_INFO("WriteFileCSR: filename=" << filename << "; could not write to file");
            return false;
        }

        out.close();

        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool write_matrix_csr_rocsparseio(int64_t            nrow,
                                      int64_t            ncol,
                                      int64_t            nnz,
                                      const PointerType* ptr,
                                      const IndexType*   col,
                                      const ValueType*   val,
                                      const char*        filename)
    {
        rocsparseio_handle handle;
        rocsparseio_status io_status;
        io_status = rocsparseio_open(&handle, rocsparseio_rwmode_write, filename);
        if(io_status != rocsparseio_status_success)
        {
            LOG_INFO("write_matrix_csr_rocsparseio: cannot open file " << filename);
            return false;
        }

        const rocsparseio_type ptr_type = type2rocsparseio_type<PointerType>();
        const rocsparseio_type col_type = type2rocsparseio_type<IndexType>();
        const rocsparseio_type val_type = type2rocsparseio_type<ValueType>();

        static constexpr rocsparseio_direction  dir  = rocsparseio_direction_row;
        static constexpr rocsparseio_index_base base = rocsparseio_index_base_zero;

        const uint64_t save_m   = nrow;
        const uint64_t save_n   = ncol;
        const uint64_t save_nnz = nnz;

        io_status = rocsparseio_write_sparse_csx(handle,
                                                 dir,
                                                 save_m,
                                                 save_n,
                                                 save_nnz,
                                                 ptr_type,
                                                 ptr,
                                                 col_type,
                                                 col,
                                                 val_type,
                                                 val,
                                                 base,
                                                 filename);
        if(io_status != rocsparseio_status_success)
        {
            LOG_INFO("write_matrix_csr_rocsparseio: cannot write file " << filename);
            rocsparseio_close(handle);
            return false;
        }

        rocsparseio_close(handle);
        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool write_matrix_mcsr_rocsparseio(int64_t            nrow,
                                       int64_t            ncol,
                                       int64_t            nnz,
                                       const PointerType* ptr,
                                       const IndexType*   col,
                                       const ValueType*   val,
                                       const char*        filename)
    {
        rocsparseio_handle handle;
        rocsparseio_status io_status;
        io_status = rocsparseio_open(&handle, rocsparseio_rwmode_write, filename);
        if(io_status != rocsparseio_status_success)
        {
            LOG_INFO("write_matrix_mcsr_rocsparseio: cannot open file " << filename);
            return false;
        }

        const rocsparseio_type ptr_type = type2rocsparseio_type<PointerType>();
        const rocsparseio_type col_type = type2rocsparseio_type<IndexType>();
        const rocsparseio_type val_type = type2rocsparseio_type<ValueType>();

        static constexpr rocsparseio_direction  dir  = rocsparseio_direction_row;
        static constexpr rocsparseio_index_base base = rocsparseio_index_base_zero;

        const uint64_t save_m   = nrow;
        const uint64_t save_n   = ncol;
        const uint64_t save_nnz = nnz;

        io_status = rocsparseio_write_sparse_mcsx(handle,
                                                  dir,
                                                  save_m,
                                                  save_n,
                                                  save_nnz,
                                                  ptr_type,
                                                  ptr,
                                                  col_type,
                                                  col,
                                                  val_type,
                                                  val,
                                                  base,
                                                  filename);
        if(io_status != rocsparseio_status_success)
        {
            LOG_INFO("write_matrix_mcsr_rocsparseio: cannot write file " << filename);
            rocsparseio_close(handle);
            return false;
        }

        rocsparseio_close(handle);
        return true;
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool write_matrix_bcsr_rocsparseio(int64_t            nrowb,
                                       int64_t            ncolb,
                                       int64_t            nnzb,
                                       int64_t            block_dim,
                                       const PointerType* ptr,
                                       const IndexType*   col,
                                       const ValueType*   val,
                                       const char*        filename)
    {
        rocsparseio_handle handle;
        rocsparseio_status io_status;
        io_status = rocsparseio_open(&handle, rocsparseio_rwmode_write, filename);
        if(io_status != rocsparseio_status_success)
        {
            LOG_INFO("write_matrix_bcsr_rocsparseio: cannot open file " << filename);
            return false;
        }

        const rocsparseio_type ptr_type = type2rocsparseio_type<PointerType>();
        const rocsparseio_type col_type = type2rocsparseio_type<IndexType>();
        const rocsparseio_type val_type = type2rocsparseio_type<ValueType>();

        static constexpr rocsparseio_direction  dir  = rocsparseio_direction_row;
        static constexpr rocsparseio_index_base base = rocsparseio_index_base_zero;

        io_status = rocsparseio_write_sparse_gebsx(handle,
                                                   dir,
                                                   dir,
                                                   nrowb,
                                                   ncolb,
                                                   nnzb,
                                                   block_dim,
                                                   block_dim,
                                                   ptr_type,
                                                   ptr,
                                                   col_type,
                                                   col,
                                                   val_type,
                                                   val,
                                                   base,
                                                   filename);
        if(io_status != rocsparseio_status_success)
        {
            LOG_INFO("write_matrix_bcsr_rocsparseio: cannot write file " << filename);
            rocsparseio_close(handle);
            return false;
        }

        rocsparseio_close(handle);
        return true;
    }

    template <typename ValueType, typename IndexType>
    bool write_matrix_coo_rocsparseio(int64_t          nrow,
                                      int64_t          ncol,
                                      int64_t          nnz,
                                      const IndexType* row,
                                      const IndexType* col,
                                      const ValueType* val,
                                      const char*      filename)
    {
        rocsparseio_handle handle;
        rocsparseio_status io_status;
        io_status = rocsparseio_open(&handle, rocsparseio_rwmode_write, filename);
        if(io_status != rocsparseio_status_success)
        {
            LOG_INFO("write_matrix_coo_rocsparseio: cannot open file " << filename);
            return false;
        }

        const rocsparseio_type row_type = type2rocsparseio_type<IndexType>();
        const rocsparseio_type col_type = type2rocsparseio_type<IndexType>();
        const rocsparseio_type val_type = type2rocsparseio_type<ValueType>();

        static constexpr rocsparseio_index_base base = rocsparseio_index_base_zero;

        io_status = rocsparseio_write_sparse_coo(
            handle, nrow, ncol, nnz, row_type, row, col_type, col, val_type, val, base, filename);
        if(io_status != rocsparseio_status_success)
        {
            LOG_INFO("write_matrix_coo_rocsparseio: cannot write file " << filename);
            rocsparseio_close(handle);
            return false;
        }

        rocsparseio_close(handle);
        return true;
    }

    template <typename ValueType, typename IndexType>
    bool write_matrix_dia_rocsparseio(int64_t          nrow,
                                      int64_t          ncol,
                                      int64_t          ndiag,
                                      const IndexType* offset,
                                      const ValueType* val,
                                      const char*      filename)
    {
        rocsparseio_handle handle;
        rocsparseio_status io_status;
        io_status = rocsparseio_open(&handle, rocsparseio_rwmode_write, filename);
        if(io_status != rocsparseio_status_success)
        {
            LOG_INFO("write_matrix_dia_rocsparseio: cannot open file " << filename);
            return false;
        }

        const rocsparseio_type offset_type = type2rocsparseio_type<IndexType>();
        const rocsparseio_type val_type    = type2rocsparseio_type<ValueType>();

        static constexpr rocsparseio_index_base base = rocsparseio_index_base_zero;

        io_status = rocsparseio_write_sparse_dia(
            handle, nrow, ncol, ndiag, offset_type, offset, val_type, val, base, filename);
        if(io_status != rocsparseio_status_success)
        {
            LOG_INFO("write_matrix_dia_rocsparseio: cannot write file " << filename);
            rocsparseio_close(handle);
            return false;
        }

        rocsparseio_close(handle);
        return true;
    }

    template <typename ValueType, typename IndexType>
    bool write_matrix_ell_rocsparseio(int64_t          nrow,
                                      int64_t          ncol,
                                      int64_t          width,
                                      const IndexType* col,
                                      const ValueType* val,
                                      const char*      filename)
    {
        rocsparseio_handle handle;
        rocsparseio_status io_status;
        io_status = rocsparseio_open(&handle, rocsparseio_rwmode_write, filename);
        if(io_status != rocsparseio_status_success)
        {
            LOG_INFO("write_matrix_ell_rocsparseio: cannot open file " << filename);
            return false;
        }

        const rocsparseio_type col_type = type2rocsparseio_type<IndexType>();
        const rocsparseio_type val_type = type2rocsparseio_type<ValueType>();

        static constexpr rocsparseio_index_base base = rocsparseio_index_base_zero;

        io_status = rocsparseio_write_sparse_ell(
            handle, nrow, ncol, width, col_type, col, val_type, val, base, filename);
        if(io_status != rocsparseio_status_success)
        {
            LOG_INFO("write_matrix_ell_rocsparseio: cannot write file " << filename);
            rocsparseio_close(handle);
            return false;
        }

        rocsparseio_close(handle);
        return true;
    }

    template <typename ValueType, typename IndexType>
    bool write_matrix_hyb_rocsparseio(int64_t          nrow,
                                      int64_t          ncol,
                                      int64_t          coo_nnz,
                                      const IndexType* coo_row,
                                      const IndexType* coo_col,
                                      const ValueType* coo_val,
                                      int64_t          ell_width,
                                      const IndexType* ell_col,
                                      const ValueType* ell_val,
                                      const char*      filename)
    {
        rocsparseio_handle handle;
        rocsparseio_status io_status;
        io_status = rocsparseio_open(&handle, rocsparseio_rwmode_write, filename);
        if(io_status != rocsparseio_status_success)
        {
            LOG_INFO("write_matrix_hyb_rocsparseio: cannot open file " << filename);
            return false;
        }

        const rocsparseio_type coo_row_type = type2rocsparseio_type<IndexType>();
        const rocsparseio_type coo_col_type = type2rocsparseio_type<IndexType>();
        const rocsparseio_type coo_val_type = type2rocsparseio_type<ValueType>();
        const rocsparseio_type ell_col_type = type2rocsparseio_type<IndexType>();
        const rocsparseio_type ell_val_type = type2rocsparseio_type<ValueType>();

        static constexpr rocsparseio_index_base base = rocsparseio_index_base_zero;

        io_status = rocsparseio_write_sparse_hyb(handle,
                                                 nrow,
                                                 ncol,
                                                 coo_nnz,
                                                 coo_row_type,
                                                 coo_row,
                                                 coo_col_type,
                                                 coo_col,
                                                 coo_val_type,
                                                 coo_val,
                                                 base,
                                                 ell_width,
                                                 ell_col_type,
                                                 ell_col,
                                                 ell_val_type,
                                                 ell_val,
                                                 base,
                                                 filename);
        if(io_status != rocsparseio_status_success)
        {
            LOG_INFO("write_matrix_hyb_rocsparseio: cannot write file " << filename);
            rocsparseio_close(handle);
            return false;
        }

        rocsparseio_close(handle);
        return true;
    }

    template <typename ValueType>
    bool write_matrix_dense_rocsparseio(int64_t          nrow,
                                        int64_t          ncol,
                                        const ValueType* val,
                                        const char*      filename)
    {
        rocsparseio_handle handle;
        rocsparseio_status io_status;
        io_status = rocsparseio_open(&handle, rocsparseio_rwmode_write, filename);
        if(io_status != rocsparseio_status_success)
        {
            LOG_INFO("write_matrix_dense_rocsparseio: cannot open file " << filename);
            return false;
        }

        const rocsparseio_type val_type = type2rocsparseio_type<ValueType>();

        io_status = rocsparseio_write_dense_matrix(
            handle, rocsparseio_order_row, nrow, ncol, val_type, val, nrow, filename);
        if(io_status != rocsparseio_status_success)
        {
            LOG_INFO("write_matrix_dense_rocsparseio: cannot write file " << filename);
            rocsparseio_close(handle);
            return false;
        }

        rocsparseio_close(handle);
        return true;
    }

    template bool read_matrix_mtx(int&        nrow,
                                  int&        ncol,
                                  int64_t&    nnz,
                                  int**       row,
                                  int**       col,
                                  float**     val,
                                  const char* filename);
    template bool read_matrix_mtx(int&        nrow,
                                  int&        ncol,
                                  int64_t&    nnz,
                                  int**       row,
                                  int**       col,
                                  double**    val,
                                  const char* filename);
#ifdef SUPPORT_COMPLEX
    template bool read_matrix_mtx(int&                  nrow,
                                  int&                  ncol,
                                  int64_t&              nnz,
                                  int**                 row,
                                  int**                 col,
                                  std::complex<float>** val,
                                  const char*           filename);
    template bool read_matrix_mtx(int&                   nrow,
                                  int&                   ncol,
                                  int64_t&               nnz,
                                  int**                  row,
                                  int**                  col,
                                  std::complex<double>** val,
                                  const char*            filename);
#endif

    template bool write_matrix_mtx(int          nrow,
                                   int          ncol,
                                   int64_t      nnz,
                                   const int*   row,
                                   const int*   col,
                                   const float* val,
                                   const char*  filename);
    template bool write_matrix_mtx(int           nrow,
                                   int           ncol,
                                   int64_t       nnz,
                                   const int*    row,
                                   const int*    col,
                                   const double* val,
                                   const char*   filename);
#ifdef SUPPORT_COMPLEX
    template bool write_matrix_mtx(int                        nrow,
                                   int                        ncol,
                                   int64_t                    nnz,
                                   const int*                 row,
                                   const int*                 col,
                                   const std::complex<float>* val,
                                   const char*                filename);
    template bool write_matrix_mtx(int                         nrow,
                                   int                         ncol,
                                   int64_t                     nnz,
                                   const int*                  row,
                                   const int*                  col,
                                   const std::complex<double>* val,
                                   const char*                 filename);
#endif

    template bool read_matrix_csr(int64_t&    nrow,
                                  int64_t&    ncol,
                                  int64_t&    nnz,
                                  int**       ptr,
                                  int**       col,
                                  float**     val,
                                  const char* filename);
    template bool read_matrix_csr(int64_t&    nrow,
                                  int64_t&    ncol,
                                  int64_t&    nnz,
                                  int**       ptr,
                                  int**       col,
                                  double**    val,
                                  const char* filename);
#ifdef SUPPORT_COMPLEX
    template bool read_matrix_csr(int64_t&              nrow,
                                  int64_t&              ncol,
                                  int64_t&              nnz,
                                  int**                 ptr,
                                  int**                 col,
                                  std::complex<float>** val,
                                  const char*           filename);
    template bool read_matrix_csr(int64_t&               nrow,
                                  int64_t&               ncol,
                                  int64_t&               nnz,
                                  int**                  ptr,
                                  int**                  col,
                                  std::complex<double>** val,
                                  const char*            filename);
#endif

    template bool read_matrix_csr(int64_t&    nrow,
                                  int64_t&    ncol,
                                  int64_t&    nnz,
                                  int64_t**   ptr,
                                  int**       col,
                                  float**     val,
                                  const char* filename);
    template bool read_matrix_csr(int64_t&    nrow,
                                  int64_t&    ncol,
                                  int64_t&    nnz,
                                  int64_t**   ptr,
                                  int**       col,
                                  double**    val,
                                  const char* filename);
#ifdef SUPPORT_COMPLEX
    template bool read_matrix_csr(int64_t&              nrow,
                                  int64_t&              ncol,
                                  int64_t&              nnz,
                                  int64_t**             ptr,
                                  int**                 col,
                                  std::complex<float>** val,
                                  const char*           filename);
    template bool read_matrix_csr(int64_t&               nrow,
                                  int64_t&               ncol,
                                  int64_t&               nnz,
                                  int64_t**              ptr,
                                  int**                  col,
                                  std::complex<double>** val,
                                  const char*            filename);
#endif

    template bool write_matrix_csr(int64_t      nrow,
                                   int64_t      ncol,
                                   int64_t      nnz,
                                   const int*   ptr,
                                   const int*   col,
                                   const float* val,
                                   const char*  filename);
    template bool write_matrix_csr(int64_t       nrow,
                                   int64_t       ncol,
                                   int64_t       nnz,
                                   const int*    ptr,
                                   const int*    col,
                                   const double* val,
                                   const char*   filename);
#ifdef SUPPORT_COMPLEX
    template bool write_matrix_csr(int64_t                    nrow,
                                   int64_t                    ncol,
                                   int64_t                    nnz,
                                   const int*                 ptr,
                                   const int*                 col,
                                   const std::complex<float>* val,
                                   const char*                filename);
    template bool write_matrix_csr(int64_t                     nrow,
                                   int64_t                     ncol,
                                   int64_t                     nnz,
                                   const int*                  ptr,
                                   const int*                  col,
                                   const std::complex<double>* val,
                                   const char*                 filename);
#endif

    template bool write_matrix_csr(int64_t        nrow,
                                   int64_t        ncol,
                                   int64_t        nnz,
                                   const int64_t* ptr,
                                   const int*     col,
                                   const float*   val,
                                   const char*    filename);
    template bool write_matrix_csr(int64_t        nrow,
                                   int64_t        ncol,
                                   int64_t        nnz,
                                   const int64_t* ptr,
                                   const int*     col,
                                   const double*  val,
                                   const char*    filename);
#ifdef SUPPORT_COMPLEX
    template bool write_matrix_csr(int64_t                    nrow,
                                   int64_t                    ncol,
                                   int64_t                    nnz,
                                   const int64_t*             ptr,
                                   const int*                 col,
                                   const std::complex<float>* val,
                                   const char*                filename);
    template bool write_matrix_csr(int64_t                     nrow,
                                   int64_t                     ncol,
                                   int64_t                     nnz,
                                   const int64_t*              ptr,
                                   const int*                  col,
                                   const std::complex<double>* val,
                                   const char*                 filename);
#endif

    template bool read_matrix_csr_rocsparseio(int64_t&    nrow,
                                              int64_t&    ncol,
                                              int64_t&    nnz,
                                              int**       ptr,
                                              int**       col,
                                              float**     val,
                                              const char* filename);
    template bool read_matrix_csr_rocsparseio(int64_t&    nrow,
                                              int64_t&    ncol,
                                              int64_t&    nnz,
                                              int**       ptr,
                                              int**       col,
                                              double**    val,
                                              const char* filename);
#ifdef SUPPORT_COMPLEX
    template bool read_matrix_csr_rocsparseio(int64_t&              nrow,
                                              int64_t&              ncol,
                                              int64_t&              nnz,
                                              int**                 ptr,
                                              int**                 col,
                                              std::complex<float>** val,
                                              const char*           filename);
    template bool read_matrix_csr_rocsparseio(int64_t&               nrow,
                                              int64_t&               ncol,
                                              int64_t&               nnz,
                                              int**                  ptr,
                                              int**                  col,
                                              std::complex<double>** val,
                                              const char*            filename);
#endif

    template bool read_matrix_csr_rocsparseio(int64_t&    nrow,
                                              int64_t&    ncol,
                                              int64_t&    nnz,
                                              int64_t**   ptr,
                                              int**       col,
                                              float**     val,
                                              const char* filename);
    template bool read_matrix_csr_rocsparseio(int64_t&    nrow,
                                              int64_t&    ncol,
                                              int64_t&    nnz,
                                              int64_t**   ptr,
                                              int**       col,
                                              double**    val,
                                              const char* filename);
#ifdef SUPPORT_COMPLEX
    template bool read_matrix_csr_rocsparseio(int64_t&              nrow,
                                              int64_t&              ncol,
                                              int64_t&              nnz,
                                              int64_t**             ptr,
                                              int**                 col,
                                              std::complex<float>** val,
                                              const char*           filename);
    template bool read_matrix_csr_rocsparseio(int64_t&               nrow,
                                              int64_t&               ncol,
                                              int64_t&               nnz,
                                              int64_t**              ptr,
                                              int**                  col,
                                              std::complex<double>** val,
                                              const char*            filename);
#endif

    template bool write_matrix_csr_rocsparseio(int64_t      nrow,
                                               int64_t      ncol,
                                               int64_t      nnz,
                                               const int*   ptr,
                                               const int*   col,
                                               const float* val,
                                               const char*  filename);
    template bool write_matrix_csr_rocsparseio(int64_t       nrow,
                                               int64_t       ncol,
                                               int64_t       nnz,
                                               const int*    ptr,
                                               const int*    col,
                                               const double* val,
                                               const char*   filename);
#ifdef SUPPORT_COMPLEX
    template bool write_matrix_csr_rocsparseio(int64_t                    nrow,
                                               int64_t                    ncol,
                                               int64_t                    nnz,
                                               const int*                 ptr,
                                               const int*                 col,
                                               const std::complex<float>* val,
                                               const char*                filename);
    template bool write_matrix_csr_rocsparseio(int64_t                     nrow,
                                               int64_t                     ncol,
                                               int64_t                     nnz,
                                               const int*                  ptr,
                                               const int*                  col,
                                               const std::complex<double>* val,
                                               const char*                 filename);
#endif

    template bool write_matrix_csr_rocsparseio(int64_t        nrow,
                                               int64_t        ncol,
                                               int64_t        nnz,
                                               const int64_t* ptr,
                                               const int*     col,
                                               const float*   val,
                                               const char*    filename);
    template bool write_matrix_csr_rocsparseio(int64_t        nrow,
                                               int64_t        ncol,
                                               int64_t        nnz,
                                               const int64_t* ptr,
                                               const int*     col,
                                               const double*  val,
                                               const char*    filename);
#ifdef SUPPORT_COMPLEX
    template bool write_matrix_csr_rocsparseio(int64_t                    nrow,
                                               int64_t                    ncol,
                                               int64_t                    nnz,
                                               const int64_t*             ptr,
                                               const int*                 col,
                                               const std::complex<float>* val,
                                               const char*                filename);
    template bool write_matrix_csr_rocsparseio(int64_t                     nrow,
                                               int64_t                     ncol,
                                               int64_t                     nnz,
                                               const int64_t*              ptr,
                                               const int*                  col,
                                               const std::complex<double>* val,
                                               const char*                 filename);
#endif

    template bool read_matrix_mcsr_rocsparseio(int64_t&    nrow,
                                               int64_t&    ncol,
                                               int64_t&    nnz,
                                               int**       ptr,
                                               int**       col,
                                               float**     val,
                                               const char* filename);
    template bool read_matrix_mcsr_rocsparseio(int64_t&    nrow,
                                               int64_t&    ncol,
                                               int64_t&    nnz,
                                               int**       ptr,
                                               int**       col,
                                               double**    val,
                                               const char* filename);
#ifdef SUPPORT_COMPLEX
    template bool read_matrix_mcsr_rocsparseio(int64_t&              nrow,
                                               int64_t&              ncol,
                                               int64_t&              nnz,
                                               int**                 ptr,
                                               int**                 col,
                                               std::complex<float>** val,
                                               const char*           filename);
    template bool read_matrix_mcsr_rocsparseio(int64_t&               nrow,
                                               int64_t&               ncol,
                                               int64_t&               nnz,
                                               int**                  ptr,
                                               int**                  col,
                                               std::complex<double>** val,
                                               const char*            filename);
#endif

    template bool read_matrix_mcsr_rocsparseio(int64_t&    nrow,
                                               int64_t&    ncol,
                                               int64_t&    nnz,
                                               int64_t**   ptr,
                                               int**       col,
                                               float**     val,
                                               const char* filename);
    template bool read_matrix_mcsr_rocsparseio(int64_t&    nrow,
                                               int64_t&    ncol,
                                               int64_t&    nnz,
                                               int64_t**   ptr,
                                               int**       col,
                                               double**    val,
                                               const char* filename);
#ifdef SUPPORT_COMPLEX
    template bool read_matrix_mcsr_rocsparseio(int64_t&              nrow,
                                               int64_t&              ncol,
                                               int64_t&              nnz,
                                               int64_t**             ptr,
                                               int**                 col,
                                               std::complex<float>** val,
                                               const char*           filename);
    template bool read_matrix_mcsr_rocsparseio(int64_t&               nrow,
                                               int64_t&               ncol,
                                               int64_t&               nnz,
                                               int64_t**              ptr,
                                               int**                  col,
                                               std::complex<double>** val,
                                               const char*            filename);
#endif

    template bool write_matrix_mcsr_rocsparseio(int64_t      nrow,
                                                int64_t      ncol,
                                                int64_t      nnz,
                                                const int*   ptr,
                                                const int*   col,
                                                const float* val,
                                                const char*  filename);
    template bool write_matrix_mcsr_rocsparseio(int64_t       nrow,
                                                int64_t       ncol,
                                                int64_t       nnz,
                                                const int*    ptr,
                                                const int*    col,
                                                const double* val,
                                                const char*   filename);
#ifdef SUPPORT_COMPLEX
    template bool write_matrix_mcsr_rocsparseio(int64_t                    nrow,
                                                int64_t                    ncol,
                                                int64_t                    nnz,
                                                const int*                 ptr,
                                                const int*                 col,
                                                const std::complex<float>* val,
                                                const char*                filename);
    template bool write_matrix_mcsr_rocsparseio(int64_t                     nrow,
                                                int64_t                     ncol,
                                                int64_t                     nnz,
                                                const int*                  ptr,
                                                const int*                  col,
                                                const std::complex<double>* val,
                                                const char*                 filename);
#endif

    template bool write_matrix_mcsr_rocsparseio(int64_t        nrow,
                                                int64_t        ncol,
                                                int64_t        nnz,
                                                const int64_t* ptr,
                                                const int*     col,
                                                const float*   val,
                                                const char*    filename);
    template bool write_matrix_mcsr_rocsparseio(int64_t        nrow,
                                                int64_t        ncol,
                                                int64_t        nnz,
                                                const int64_t* ptr,
                                                const int*     col,
                                                const double*  val,
                                                const char*    filename);
#ifdef SUPPORT_COMPLEX
    template bool write_matrix_mcsr_rocsparseio(int64_t                    nrow,
                                                int64_t                    ncol,
                                                int64_t                    nnz,
                                                const int64_t*             ptr,
                                                const int*                 col,
                                                const std::complex<float>* val,
                                                const char*                filename);
    template bool write_matrix_mcsr_rocsparseio(int64_t                     nrow,
                                                int64_t                     ncol,
                                                int64_t                     nnz,
                                                const int64_t*              ptr,
                                                const int*                  col,
                                                const std::complex<double>* val,
                                                const char*                 filename);
#endif

    template bool read_matrix_bcsr_rocsparseio(int64_t&    nrowb,
                                               int64_t&    ncolb,
                                               int64_t&    nnzb,
                                               int64_t&    block_dim,
                                               int**       ptr,
                                               int**       col,
                                               float**     val,
                                               const char* filename);
    template bool read_matrix_bcsr_rocsparseio(int64_t&    nrowb,
                                               int64_t&    ncolb,
                                               int64_t&    nnzb,
                                               int64_t&    block_dim,
                                               int**       ptr,
                                               int**       col,
                                               double**    val,
                                               const char* filename);
#ifdef SUPPORT_COMPLEX
    template bool read_matrix_bcsr_rocsparseio(int64_t&              nrowb,
                                               int64_t&              ncolb,
                                               int64_t&              nnzb,
                                               int64_t&              block_dim,
                                               int**                 ptr,
                                               int**                 col,
                                               std::complex<float>** val,
                                               const char*           filename);
    template bool read_matrix_bcsr_rocsparseio(int64_t&               nrowb,
                                               int64_t&               ncolb,
                                               int64_t&               nnzb,
                                               int64_t&               block_dim,
                                               int**                  ptr,
                                               int**                  col,
                                               std::complex<double>** val,
                                               const char*            filename);
#endif

    template bool read_matrix_bcsr_rocsparseio(int64_t&    nrowb,
                                               int64_t&    ncolb,
                                               int64_t&    nnzb,
                                               int64_t&    block_dim,
                                               int64_t**   ptr,
                                               int**       col,
                                               float**     val,
                                               const char* filename);
    template bool read_matrix_bcsr_rocsparseio(int64_t&    nrowb,
                                               int64_t&    ncolb,
                                               int64_t&    nnzb,
                                               int64_t&    block_dim,
                                               int64_t**   ptr,
                                               int**       col,
                                               double**    val,
                                               const char* filename);
#ifdef SUPPORT_COMPLEX
    template bool read_matrix_bcsr_rocsparseio(int64_t&              nrowb,
                                               int64_t&              ncolb,
                                               int64_t&              nnzb,
                                               int64_t&              block_dim,
                                               int64_t**             ptr,
                                               int**                 col,
                                               std::complex<float>** val,
                                               const char*           filename);
    template bool read_matrix_bcsr_rocsparseio(int64_t&               nrowb,
                                               int64_t&               ncolb,
                                               int64_t&               nnzb,
                                               int64_t&               block_dim,
                                               int64_t**              ptr,
                                               int**                  col,
                                               std::complex<double>** val,
                                               const char*            filename);
#endif

    template bool write_matrix_bcsr_rocsparseio(int64_t      nrowb,
                                                int64_t      ncolb,
                                                int64_t      nnzb,
                                                int64_t      block_dim,
                                                const int*   ptr,
                                                const int*   col,
                                                const float* val,
                                                const char*  filename);
    template bool write_matrix_bcsr_rocsparseio(int64_t       nrowb,
                                                int64_t       ncolb,
                                                int64_t       nnzb,
                                                int64_t       block_dim,
                                                const int*    ptr,
                                                const int*    col,
                                                const double* val,
                                                const char*   filename);
#ifdef SUPPORT_COMPLEX
    template bool write_matrix_bcsr_rocsparseio(int64_t                    nrowb,
                                                int64_t                    ncolb,
                                                int64_t                    nnzb,
                                                int64_t                    block_dim,
                                                const int*                 ptr,
                                                const int*                 col,
                                                const std::complex<float>* val,
                                                const char*                filename);
    template bool write_matrix_bcsr_rocsparseio(int64_t                     nrowb,
                                                int64_t                     ncolb,
                                                int64_t                     nnzb,
                                                int64_t                     block_dim,
                                                const int*                  ptr,
                                                const int*                  col,
                                                const std::complex<double>* val,
                                                const char*                 filename);
#endif

    template bool write_matrix_bcsr_rocsparseio(int64_t        nrowb,
                                                int64_t        ncolb,
                                                int64_t        nnzb,
                                                int64_t        block_dim,
                                                const int64_t* ptr,
                                                const int*     col,
                                                const float*   val,
                                                const char*    filename);
    template bool write_matrix_bcsr_rocsparseio(int64_t        nrowb,
                                                int64_t        ncolb,
                                                int64_t        nnzb,
                                                int64_t        block_dim,
                                                const int64_t* ptr,
                                                const int*     col,
                                                const double*  val,
                                                const char*    filename);
#ifdef SUPPORT_COMPLEX
    template bool write_matrix_bcsr_rocsparseio(int64_t                    nrowb,
                                                int64_t                    ncolb,
                                                int64_t                    nnzb,
                                                int64_t                    block_dim,
                                                const int64_t*             ptr,
                                                const int*                 col,
                                                const std::complex<float>* val,
                                                const char*                filename);
    template bool write_matrix_bcsr_rocsparseio(int64_t                     nrowb,
                                                int64_t                     ncolb,
                                                int64_t                     nnzb,
                                                int64_t                     block_dim,
                                                const int64_t*              ptr,
                                                const int*                  col,
                                                const std::complex<double>* val,
                                                const char*                 filename);
#endif

    template bool read_matrix_coo_rocsparseio(int64_t&    nrow,
                                              int64_t&    ncol,
                                              int64_t&    nnz,
                                              int**       row,
                                              int**       col,
                                              float**     val,
                                              const char* filename);
    template bool read_matrix_coo_rocsparseio(int64_t&    nrow,
                                              int64_t&    ncol,
                                              int64_t&    nnz,
                                              int**       row,
                                              int**       col,
                                              double**    val,
                                              const char* filename);
#ifdef SUPPORT_COMPLEX
    template bool read_matrix_coo_rocsparseio(int64_t&              nrow,
                                              int64_t&              ncol,
                                              int64_t&              nnz,
                                              int**                 row,
                                              int**                 col,
                                              std::complex<float>** val,
                                              const char*           filename);
    template bool read_matrix_coo_rocsparseio(int64_t&               nrow,
                                              int64_t&               ncol,
                                              int64_t&               nnz,
                                              int**                  row,
                                              int**                  col,
                                              std::complex<double>** val,
                                              const char*            filename);
#endif

    template bool write_matrix_coo_rocsparseio(int64_t      nrow,
                                               int64_t      ncol,
                                               int64_t      nnz,
                                               const int*   row,
                                               const int*   col,
                                               const float* val,
                                               const char*  filename);
    template bool write_matrix_coo_rocsparseio(int64_t       nrow,
                                               int64_t       ncol,
                                               int64_t       nnz,
                                               const int*    row,
                                               const int*    col,
                                               const double* val,
                                               const char*   filename);
#ifdef SUPPORT_COMPLEX
    template bool write_matrix_coo_rocsparseio(int64_t                    nrow,
                                               int64_t                    ncol,
                                               int64_t                    nnz,
                                               const int*                 row,
                                               const int*                 col,
                                               const std::complex<float>* val,
                                               const char*                filename);
    template bool write_matrix_coo_rocsparseio(int64_t                     nrow,
                                               int64_t                     ncol,
                                               int64_t                     nnz,
                                               const int*                  row,
                                               const int*                  col,
                                               const std::complex<double>* val,
                                               const char*                 filename);
#endif

    template bool read_matrix_dia_rocsparseio(int64_t&    nrow,
                                              int64_t&    ncol,
                                              int64_t&    nnz,
                                              int64_t&    ndiag,
                                              int**       offset,
                                              float**     val,
                                              const char* filename);
    template bool read_matrix_dia_rocsparseio(int64_t&    nrow,
                                              int64_t&    ncol,
                                              int64_t&    nnz,
                                              int64_t&    ndiag,
                                              int**       offset,
                                              double**    val,
                                              const char* filename);
#ifdef SUPPORT_COMPLEX
    template bool read_matrix_dia_rocsparseio(int64_t&              nrow,
                                              int64_t&              ncol,
                                              int64_t&              nnz,
                                              int64_t&              ndiag,
                                              int**                 offset,
                                              std::complex<float>** val,
                                              const char*           filename);
    template bool read_matrix_dia_rocsparseio(int64_t&               nrow,
                                              int64_t&               ncol,
                                              int64_t&               nnz,
                                              int64_t&               ndiag,
                                              int**                  offset,
                                              std::complex<double>** val,
                                              const char*            filename);
#endif

    template bool write_matrix_dia_rocsparseio(int64_t      nrow,
                                               int64_t      ncol,
                                               int64_t      ndiag,
                                               const int*   offset,
                                               const float* val,
                                               const char*  filename);
    template bool write_matrix_dia_rocsparseio(int64_t       nrow,
                                               int64_t       ncol,
                                               int64_t       ndiag,
                                               const int*    offset,
                                               const double* val,
                                               const char*   filename);
#ifdef SUPPORT_COMPLEX
    template bool write_matrix_dia_rocsparseio(int64_t                    nrow,
                                               int64_t                    ncol,
                                               int64_t                    ndiag,
                                               const int*                 offset,
                                               const std::complex<float>* val,
                                               const char*                filename);
    template bool write_matrix_dia_rocsparseio(int64_t                     nrow,
                                               int64_t                     ncol,
                                               int64_t                     ndiag,
                                               const int*                  offset,
                                               const std::complex<double>* val,
                                               const char*                 filename);
#endif

    template bool read_matrix_ell_rocsparseio(int64_t&    nrow,
                                              int64_t&    ncol,
                                              int64_t&    nnz,
                                              int64_t&    width,
                                              int**       col,
                                              float**     val,
                                              const char* filename);
    template bool read_matrix_ell_rocsparseio(int64_t&    nrow,
                                              int64_t&    ncol,
                                              int64_t&    nnz,
                                              int64_t&    width,
                                              int**       col,
                                              double**    val,
                                              const char* filename);
#ifdef SUPPORT_COMPLEX
    template bool read_matrix_ell_rocsparseio(int64_t&              nrow,
                                              int64_t&              ncol,
                                              int64_t&              nnz,
                                              int64_t&              width,
                                              int**                 col,
                                              std::complex<float>** val,
                                              const char*           filename);
    template bool read_matrix_ell_rocsparseio(int64_t&               nrow,
                                              int64_t&               ncol,
                                              int64_t&               nnz,
                                              int64_t&               width,
                                              int**                  col,
                                              std::complex<double>** val,
                                              const char*            filename);
#endif

    template bool write_matrix_ell_rocsparseio(int64_t      nrow,
                                               int64_t      ncol,
                                               int64_t      width,
                                               const int*   col,
                                               const float* val,
                                               const char*  filename);
    template bool write_matrix_ell_rocsparseio(int64_t       nrow,
                                               int64_t       ncol,
                                               int64_t       width,
                                               const int*    col,
                                               const double* val,
                                               const char*   filename);
#ifdef SUPPORT_COMPLEX
    template bool write_matrix_ell_rocsparseio(int64_t                    nrow,
                                               int64_t                    ncol,
                                               int64_t                    width,
                                               const int*                 col,
                                               const std::complex<float>* val,
                                               const char*                filename);
    template bool write_matrix_ell_rocsparseio(int64_t                     nrow,
                                               int64_t                     ncol,
                                               int64_t                     width,
                                               const int*                  col,
                                               const std::complex<double>* val,
                                               const char*                 filename);
#endif

    template bool read_matrix_hyb_rocsparseio(int64_t&    nrow,
                                              int64_t&    ncol,
                                              int64_t&    nnz,
                                              int64_t&    coo_nnz,
                                              int**       coo_row,
                                              int**       coo_col,
                                              float**     coo_val,
                                              int64_t&    ell_nnz,
                                              int64_t&    ell_width,
                                              int**       ell_col,
                                              float**     ell_val,
                                              const char* filename);
    template bool read_matrix_hyb_rocsparseio(int64_t&    nrow,
                                              int64_t&    ncol,
                                              int64_t&    nnz,
                                              int64_t&    coo_nnz,
                                              int**       coo_row,
                                              int**       coo_col,
                                              double**    coo_val,
                                              int64_t&    ell_nnz,
                                              int64_t&    ell_width,
                                              int**       ell_col,
                                              double**    ell_val,
                                              const char* filename);
#ifdef SUPPORT_COMPLEX
    template bool read_matrix_hyb_rocsparseio(int64_t&              nrow,
                                              int64_t&              ncol,
                                              int64_t&              nnz,
                                              int64_t&              coo_nnz,
                                              int**                 coo_row,
                                              int**                 coo_col,
                                              std::complex<float>** coo_val,
                                              int64_t&              ell_nnz,
                                              int64_t&              ell_width,
                                              int**                 ell_col,
                                              std::complex<float>** ell_val,
                                              const char*           filename);
    template bool read_matrix_hyb_rocsparseio(int64_t&               nrow,
                                              int64_t&               ncol,
                                              int64_t&               nnz,
                                              int64_t&               coo_nnz,
                                              int**                  coo_row,
                                              int**                  coo_col,
                                              std::complex<double>** coo_val,
                                              int64_t&               ell_nnz,
                                              int64_t&               ell_width,
                                              int**                  ell_col,
                                              std::complex<double>** ell_val,
                                              const char*            filename);
#endif

    template bool write_matrix_hyb_rocsparseio(int64_t      nrow,
                                               int64_t      ncol,
                                               int64_t      nnz_coo,
                                               const int*   coo_row,
                                               const int*   coo_col,
                                               const float* coo_val,
                                               int64_t      ell_width,
                                               const int*   ell_col,
                                               const float* ell_val,
                                               const char*  filename);
    template bool write_matrix_hyb_rocsparseio(int64_t       nrow,
                                               int64_t       ncol,
                                               int64_t       nnz_coo,
                                               const int*    coo_row,
                                               const int*    coo_col,
                                               const double* coo_val,
                                               int64_t       ell_width,
                                               const int*    ell_col,
                                               const double* ell_val,
                                               const char*   filename);
#ifdef SUPPORT_COMPLEX
    template bool write_matrix_hyb_rocsparseio(int64_t                    nrow,
                                               int64_t                    ncol,
                                               int64_t                    nnz_coo,
                                               const int*                 coo_row,
                                               const int*                 coo_col,
                                               const std::complex<float>* coo_val,
                                               int64_t                    ell_width,
                                               const int*                 ell_col,
                                               const std::complex<float>* ell_val,
                                               const char*                filename);
    template bool write_matrix_hyb_rocsparseio(int64_t                     nrow,
                                               int64_t                     ncol,
                                               int64_t                     nnz_coo,
                                               const int*                  coo_row,
                                               const int*                  coo_col,
                                               const std::complex<double>* coo_val,
                                               int64_t                     ell_width,
                                               const int*                  ell_col,
                                               const std::complex<double>* ell_val,
                                               const char*                 filename);
#endif

    template bool read_matrix_dense_rocsparseio(int64_t&    nrow,
                                                int64_t&    ncol,
                                                float**     val,
                                                const char* filename);
    template bool read_matrix_dense_rocsparseio(int64_t&    nrow,
                                                int64_t&    ncol,
                                                double**    val,
                                                const char* filename);
#ifdef SUPPORT_COMPLEX
    template bool read_matrix_dense_rocsparseio(int64_t&              nrow,
                                                int64_t&              ncol,
                                                std::complex<float>** val,
                                                const char*           filename);
    template bool read_matrix_dense_rocsparseio(int64_t&               nrow,
                                                int64_t&               ncol,
                                                std::complex<double>** val,
                                                const char*            filename);
#endif

    template bool write_matrix_dense_rocsparseio(int64_t      nrow,
                                                 int64_t      ncol,
                                                 const float* val,
                                                 const char*  filename);
    template bool write_matrix_dense_rocsparseio(int64_t       nrow,
                                                 int64_t       ncol,
                                                 const double* val,
                                                 const char*   filename);
#ifdef SUPPORT_COMPLEX
    template bool write_matrix_dense_rocsparseio(int64_t                    nrow,
                                                 int64_t                    ncol,
                                                 const std::complex<float>* val,
                                                 const char*                filename);
    template bool write_matrix_dense_rocsparseio(int64_t                     nrow,
                                                 int64_t                     ncol,
                                                 const std::complex<double>* val,
                                                 const char*                 filename);
#endif

} // namespace rocalution
