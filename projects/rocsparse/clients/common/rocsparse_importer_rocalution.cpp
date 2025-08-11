/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_importer_rocalution.hpp"

template <typename T>
void read_csr_values(std::ifstream& in, int64_t nnz, T* csr_val)
{
    // Temporary array to convert from double to T
    std::vector<double> tmp(nnz);

    // Read in double values
    in.read((char*)tmp.data(), sizeof(double) * nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int64_t i = 0; i < nnz; ++i)
    {
        csr_val[i] = static_cast<T>(tmp[i]);
    }
}

template <>
void read_csr_values(std::ifstream& in, int64_t nnz, double* csr_val)
{
    in.read((char*)csr_val, sizeof(double) * nnz);
}

template <>
void read_csr_values(std::ifstream& in, int64_t nnz, rocsparse_float_complex* csr_val)
{
    // Temporary array to convert from double to float complex
    std::vector<rocsparse_double_complex> tmp(nnz);

    // Read in double complex values
    in.read((char*)tmp.data(), sizeof(rocsparse_double_complex) * nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int64_t i = 0; i < nnz; ++i)
    {
        csr_val[i] = rocsparse_float_complex(static_cast<float>(std::real(tmp[i])),
                                             static_cast<float>(std::imag(tmp[i])));
    }
}

template <>
void read_csr_values(std::ifstream& in, int64_t nnz, rocsparse_double_complex* csr_val)
{
    in.read((char*)csr_val, sizeof(rocsparse_double_complex) * nnz);
}

rocsparse_importer_rocalution::rocsparse_importer_rocalution(const std::string& filename_)
    : m_filename(filename_)
{
}

template <typename I, typename J>
rocsparse_status rocsparse_importer_rocalution::import_sparse_gebsx(rocsparse_direction* dir,
                                                                    rocsparse_direction* dirb,
                                                                    J*                   mb,
                                                                    J*                   nb,
                                                                    I*                   nnzb,
                                                                    J* block_dim_row,
                                                                    J* block_dim_column,
                                                                    rocsparse_index_base* base)
{
    return rocsparse_status_not_implemented;
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_importer_rocalution::import_sparse_gebsx(I* ptr, J* ind, T* val)
{
    return rocsparse_status_not_implemented;
}

template <typename I>
rocsparse_status rocsparse_importer_rocalution::import_sparse_coo(I*                    m,
                                                                  I*                    n,
                                                                  int64_t*              nnz,
                                                                  rocsparse_index_base* base)
{
    return rocsparse_status_not_implemented;
}

template <typename T, typename I>
rocsparse_status rocsparse_importer_rocalution::import_sparse_coo(I* row_ind, I* col_ind, T* val)
{
    return rocsparse_status_not_implemented;
}

template <typename I, typename J>
rocsparse_status rocsparse_importer_rocalution::import_sparse_csx(
    rocsparse_direction* dir, J* m, J* n, I* nnz, rocsparse_index_base* base)
{

    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        std::cout << "Opening file '" << this->m_filename << "' ... " << std::endl;
    }

    this->m_info_csx.in = new std::ifstream(this->m_filename, std::ios::in | std::ios::binary);
    if(!this->m_info_csx.in->is_open())
    {
        missing_file_error_message(this->m_filename.c_str());
        return rocsparse_status_internal_error;
    }
    std::string header;
    std::getline(this->m_info_csx.in[0], header);
    if(header != "#rocALUTION binary csr file")
    {
        return rocsparse_status_internal_error;
    }
    int version;
    this->m_info_csx.in->read((char*)&version, sizeof(int));
    int iM;
    int iN;
    int innz;
    this->m_info_csx.in->read((char*)&iM, sizeof(int));
    this->m_info_csx.in->read((char*)&iN, sizeof(int));
    this->m_info_csx.in->read((char*)&innz, sizeof(int));

    rocsparse_status status;
    status = rocsparse_type_conversion(iM, m[0]);
    if(status != rocsparse_status_success)
        return status;

    status = rocsparse_type_conversion(iN, n[0]);
    if(status != rocsparse_status_success)
        return status;

    status = rocsparse_type_conversion(innz, nnz[0]);
    if(status != rocsparse_status_success)
        return status;

    dir[0]               = rocsparse_direction_row;
    base[0]              = rocsparse_index_base_zero;
    this->m_info_csx.m   = iM;
    this->m_info_csx.nnz = innz;

    return rocsparse_status_success;
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_importer_rocalution::import_sparse_csx(I* ptr, J* ind, T* val)
{
    const size_t M   = this->m_info_csx.m;
    const size_t nnz = this->m_info_csx.nnz;

    const bool same_ptr_type = std::is_same<I, int>();
    const bool same_ind_type = std::is_same<J, int>();
    const bool same_val_type
        = std::is_same<T, double>() || std::is_same<T, rocsparse_double_complex>();
    const bool is_consistent = same_ptr_type && same_ind_type && same_val_type;
    if(is_consistent)
    {
        this->m_info_csx.in->read((char*)ptr, sizeof(int) * (M + 1));
        this->m_info_csx.in->read((char*)ind, sizeof(int) * nnz);
        this->m_info_csx.in->read((char*)val, sizeof(T) * nnz);
        this->m_info_csx.in->close();
        delete this->m_info_csx.in;
        this->m_info_csx.in = nullptr;
        {
            const char* env = getenv("GTEST_LISTENER");
            if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
            {
                std::cout << "Import done." << std::endl;
            }
        }
    }
    else
    {
        void* tmp_ptr = (void*)ptr;
        void* tmp_ind = (void*)ind;
        void* tmp_val = (void*)val;

        host_dense_vector<int> tmp_ptrv;
        host_dense_vector<int> tmp_indv;
        host_dense_vector<T>   tmp_valv;

        if(!same_ptr_type)
        {
            tmp_ptrv.resize((M + 1));
            tmp_ptr = tmp_ptrv;
        }
        else
        {
            tmp_ptr = ptr;
        }

        if(!same_ind_type)
        {
            tmp_indv.resize(nnz);
            tmp_ind = tmp_indv;
        }
        else
        {
            tmp_ind = ind;
        }

        if(!same_val_type)
        {
            tmp_valv.resize(nnz);
            tmp_val = tmp_valv;
        }
        else
        {
            tmp_val = val;
        }
        this->m_info_csx.in->read((char*)tmp_ptr, sizeof(int) * (M + 1));
        this->m_info_csx.in->read((char*)tmp_ind, sizeof(int) * nnz);
        read_csr_values(this->m_info_csx.in[0], (int64_t)nnz, (T*)tmp_val);
        //  this->m_info_csx.in->read((char*)tmp_val, sizeof(double) * nnz);
        this->m_info_csx.in->close();
        delete this->m_info_csx.in;
        this->m_info_csx.in = nullptr;
        {
            const char* env = getenv("GTEST_LISTENER");
            if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
            {
                std::cout << "Import done." << std::endl;
            }
        }
        //
        // Copy back.
        //
        if(!same_ptr_type)
        {

            rocsparse_importer_copy_mixed_arrays(M + 1, ptr, (int*)tmp_ptr);
        }
        if(!same_ind_type)
        {
            rocsparse_importer_copy_mixed_arrays(nnz, ind, (int*)tmp_ind);
        }
        if(!same_val_type)
        {
            rocsparse_importer_copy_mixed_arrays(nnz, val, (T*)tmp_val);
        }
    }

    return rocsparse_status_success;
}

#define INSTANTIATE_TIJ(T, I, J)                                                            \
    template rocsparse_status rocsparse_importer_rocalution::import_sparse_csx(I*, J*, T*); \
    template rocsparse_status rocsparse_importer_rocalution::import_sparse_gebsx(I*, J*, T*)

#define INSTANTIATE_TI(T, I)                                                    \
    template rocsparse_status rocsparse_importer_rocalution::import_sparse_coo( \
        I* row_ind, I* col_ind, T* val)

#define INSTANTIATE_I(I)                                                        \
    template rocsparse_status rocsparse_importer_rocalution::import_sparse_coo( \
        I* m, I* n, int64_t* nnz, rocsparse_index_base* base)

#define INSTANTIATE_IJ(I, J)                                                      \
    template rocsparse_status rocsparse_importer_rocalution::import_sparse_csx(   \
        rocsparse_direction*, J*, J*, I*, rocsparse_index_base*);                 \
    template rocsparse_status rocsparse_importer_rocalution::import_sparse_gebsx( \
        rocsparse_direction*, rocsparse_direction*, J*, J*, I*, J*, J*, rocsparse_index_base*)

INSTANTIATE_I(int32_t);
INSTANTIATE_I(int64_t);

INSTANTIATE_IJ(int32_t, int32_t);
INSTANTIATE_IJ(int64_t, int32_t);
INSTANTIATE_IJ(int64_t, int64_t);

INSTANTIATE_TIJ(int8_t, int32_t, int32_t);
INSTANTIATE_TIJ(int8_t, int64_t, int32_t);
INSTANTIATE_TIJ(int8_t, int64_t, int64_t);

INSTANTIATE_TIJ(_Float16, int32_t, int32_t);
INSTANTIATE_TIJ(_Float16, int64_t, int32_t);
INSTANTIATE_TIJ(_Float16, int64_t, int64_t);

INSTANTIATE_TIJ(rocsparse_bfloat16, int32_t, int32_t);
INSTANTIATE_TIJ(rocsparse_bfloat16, int64_t, int32_t);
INSTANTIATE_TIJ(rocsparse_bfloat16, int64_t, int64_t);

INSTANTIATE_TIJ(float, int32_t, int32_t);
INSTANTIATE_TIJ(float, int64_t, int32_t);
INSTANTIATE_TIJ(float, int64_t, int64_t);

INSTANTIATE_TIJ(double, int32_t, int32_t);
INSTANTIATE_TIJ(double, int64_t, int32_t);
INSTANTIATE_TIJ(double, int64_t, int64_t);

INSTANTIATE_TIJ(rocsparse_float_complex, int32_t, int32_t);
INSTANTIATE_TIJ(rocsparse_float_complex, int64_t, int32_t);
INSTANTIATE_TIJ(rocsparse_float_complex, int64_t, int64_t);

INSTANTIATE_TIJ(rocsparse_double_complex, int32_t, int32_t);
INSTANTIATE_TIJ(rocsparse_double_complex, int64_t, int32_t);
INSTANTIATE_TIJ(rocsparse_double_complex, int64_t, int64_t);

INSTANTIATE_TI(int8_t, int32_t);
INSTANTIATE_TI(int8_t, int64_t);

INSTANTIATE_TI(_Float16, int32_t);
INSTANTIATE_TI(_Float16, int64_t);

INSTANTIATE_TI(rocsparse_bfloat16, int32_t);
INSTANTIATE_TI(rocsparse_bfloat16, int64_t);

INSTANTIATE_TI(float, int32_t);
INSTANTIATE_TI(float, int64_t);

INSTANTIATE_TI(double, int32_t);
INSTANTIATE_TI(double, int64_t);

INSTANTIATE_TI(rocsparse_float_complex, int32_t);
INSTANTIATE_TI(rocsparse_float_complex, int64_t);

INSTANTIATE_TI(rocsparse_double_complex, int32_t);
INSTANTIATE_TI(rocsparse_double_complex, int64_t);
