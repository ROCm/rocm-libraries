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

#pragma once
#ifndef TESTING_UTILITY_HPP
#define TESTING_UTILITY_HPP

#include <algorithm>
#include <string>

#include "random.hpp"

template <std::size_t N, typename T>
static constexpr std::size_t countof(T (&)[N])
{
    return N;
}

#define rocalution_bench_errmsg std::cerr << "// rocalution_bench_error msg: "

extern int device;

/* ============================================================================================ */
/*! \brief  Generate 2D laplacian on unit square in CSR format */
template <typename T>
int gen_2d_laplacian(int ndim, int** rowptr, int** col, T** val)
{
    if(ndim == 0)
    {
        return 0;
    }

    int n       = ndim * ndim;
    int nnz_mat = n * 5 - ndim * 4;

    *rowptr = new int[n + 1];
    *col    = new int[nnz_mat];
    *val    = new T[nnz_mat];

    int nnz = 0;

    // Fill local arrays
    for(int i = 0; i < ndim; ++i)
    {
        for(int j = 0; j < ndim; ++j)
        {
            int idx        = i * ndim + j;
            (*rowptr)[idx] = nnz;
            // if no upper boundary element, connect with upper neighbor
            if(i != 0)
            {
                (*col)[nnz] = idx - ndim;
                (*val)[nnz] = static_cast<T>(-1);
                ++nnz;
            }
            // if no left boundary element, connect with left neighbor
            if(j != 0)
            {
                (*col)[nnz] = idx - 1;
                (*val)[nnz] = static_cast<T>(-1);
                ++nnz;
            }
            // element itself
            (*col)[nnz] = idx;
            (*val)[nnz] = static_cast<T>(4);
            ++nnz;
            // if no right boundary element, connect with right neighbor
            if(j != ndim - 1)
            {
                (*col)[nnz] = idx + 1;
                (*val)[nnz] = static_cast<T>(-1);
                ++nnz;
            }
            // if no lower boundary element, connect with lower neighbor
            if(i != ndim - 1)
            {
                (*col)[nnz] = idx + ndim;
                (*val)[nnz] = static_cast<T>(-1);
                ++nnz;
            }
        }
    }
    (*rowptr)[n] = nnz;

    return n;
}

/* ============================================================================================ */
/*! \brief  Generate 3D laplacian on unit square in CSR format */
template <typename T>
int gen_3d_laplacian(int ndim, int** row_ptr, int** col_ind, T** val)
{
    // Do nothing
    if(ndim == 0)
    {
        return 0;
    }

    int n = ndim * ndim * ndim;

    // Approximate 27pt stencil
    int nnz_mat = 27 * n;

    *row_ptr = new int[n + 1];
    *col_ind = new int[nnz_mat];
    *val     = new T[nnz_mat];

    int nnz       = 0;
    (*row_ptr)[0] = 0;

    // Fill local arrays
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int32_t iz = 0; iz < ndim; ++iz)
    {
        for(int32_t iy = 0; iy < ndim; ++iy)
        {
            for(int32_t ix = 0; ix < ndim; ++ix)
            {
                int row = iz * ndim * ndim + iy * ndim + ix;

                for(int32_t sz = -1; sz <= 1; ++sz)
                {
                    if(iz + sz > -1 && iz + sz < ndim)
                    {
                        for(int32_t sy = -1; sy <= 1; ++sy)
                        {
                            if(iy + sy > -1 && iy + sy < ndim)
                            {
                                for(int32_t sx = -1; sx <= 1; ++sx)
                                {
                                    if(ix + sx > -1 && ix + sx < ndim)
                                    {
                                        int col = row + sz * ndim * ndim + sy * ndim + sx;

                                        (*col_ind)[nnz - 0] = col + 0;
                                        (*val)[nnz - 0]     = (col == row) ? 26.0 : -1.0;

                                        ++nnz;
                                    }
                                }
                            }
                        }
                    }
                }

                (*row_ptr)[row + 1] = nnz;
            }
        }
    }

    // Adjust nnz by index base
    nnz -= 0;

    return n;
}

/* ============================================================================================ */
/*! \brief  Generate full rank identity matrix where the row order has been permuted */
template <typename T>
int gen_permuted_identity(int ndim, int** rowptr, int** col, T** val)
{
    if(ndim == 0)
    {
        return 0;
    }

    int n       = ndim * ndim;
    int nnz_mat = n;

    *rowptr = new int[n + 1];
    *col    = new int[nnz_mat];
    *val    = new T[nnz_mat];

    for(int i = 0; i < n + 1; i++)
    {
        (*rowptr)[i] = i;
    }

    srand(12345ULL);
    int r = (int)(static_cast<T>(rand()) / RAND_MAX * (n - 0));

    for(int i = 0; i < n; i++)
    {
        int index = r + i;
        if(r + i >= n)
        {
            index = r + i - n;
        }

        (*col)[i] = index;
        (*val)[i] = static_cast<T>(1);
    }

    return n;
}

/* ============================================================================================ */
/*! \brief  Generate random sparse matrix */
template <typename T>
int gen_random(int m, int n, int max_nnz_per_row, int** rowptr, int** col, T** val)
{
    if(m == 0 || n == 0)
    {
        return 0;
    }

    int  nnz         = 0;
    int* nnz_per_row = new int[m];
    for(int i = 0; i < m; i++)
    {
        nnz_per_row[i] = random_generator_exact<int>(0, max_nnz_per_row);
        nnz += nnz_per_row[i];
    }

    *rowptr = new int[m + 1];
    *col    = new int[nnz];
    *val    = new T[nnz];

    (*rowptr)[0] = 0;
    for(int i = 1; i < m + 1; i++)
    {
        (*rowptr)[i] = (*rowptr)[i - 1] + nnz_per_row[i - 1];
    }

    std::vector<int> candidates(n);
    for(int i = 0; i < n; ++i)
    {
        candidates[i] = i;
    }

    std::random_shuffle(candidates.begin(), candidates.end());

    for(int i = 0; i < m; i++)
    {
        int row_start = (*rowptr)[i];
        int row_end   = (*rowptr)[i + 1];

        // Randomly select candidate entry point
        int idx = random_generator<int>(0, n - row_end + row_start);

        for(int j = row_start; j < row_end; j++)
        {
            (*col)[j] = candidates[idx++];
            (*val)[j] = random_generator<T>();
        }
    }

    for(int i = 0; i < m; ++i)
    {
        std::sort(&(*col)[(*rowptr)[i]], &(*col)[(*rowptr)[i + 1]]);
    }

    delete[] nnz_per_row;

    return m;
}

/* ============================================================================================ */
/*! \brief  Check for environment variable to be set */
static bool is_env_var_set(const char* var, const std::string& value = "1")
{
    const char* env_var = std::getenv(var);

    if(env_var && env_var == value)
    {
        return true;
    }

    return false;
}

/* ============================================================================================ */
/*! \brief  Check for environment variables to be set */
static bool is_any_env_var_set(const std::initializer_list<const char*>& vars,
                               const std::string&                        value = "1")
{
    for(const char* var : vars)
    {
        if(is_env_var_set(var, value))
        {
            return true;
        }
    }

    return false;
}

/* ============================================================================================ */

/*! \brief Class used to parse command arguments in both client & gtest   */

// has to compile with option "-std=c++11", and this rocsparse library uses c++11 everywhere
// c++11 allows intilization of member of a struct

class Arguments
{
public:
    // MPI variables
    int rank         = 0;
    int dev_per_node = 1;

    // OpenMP variables
    int omp_nthreads  = 8;
    int omp_affinity  = true;
    int omp_threshold = 50000;

    // Accelerator variables
    int dev     = 0;
    int use_acc = true;

    // Structure variables
    int size       = 100;
    int index      = 50;
    int chunk_size = 20;
    int blockdim   = 4;

    // Computation variables
    double alpha = 1.0;
    double beta  = 0.0;
    double gamma = 0.0;

    // Solver variables
    std::string solver              = "";
    std::string precond             = "";
    std::string smoother            = "";
    std::string matrix              = "";
    std::string coarsening_strategy = "";
    std::string matrix_type         = "";

    int pre_smooth     = 2;
    int post_smooth    = 2;
    int ordering       = 1;
    int cycle          = 0;
    int rebuildnumeric = 0;

    unsigned int format;

    bool unit_diag = false;

    Arguments& operator=(const Arguments& rhs)
    {
        this->rank         = rhs.rank;
        this->dev_per_node = rhs.dev_per_node;

        this->omp_nthreads  = rhs.omp_nthreads;
        this->omp_affinity  = rhs.omp_affinity;
        this->omp_threshold = rhs.omp_threshold;

        this->dev     = rhs.dev;
        this->use_acc = rhs.use_acc;

        this->size       = rhs.size;
        this->index      = rhs.index;
        this->chunk_size = rhs.chunk_size;
        this->blockdim   = rhs.blockdim;

        this->alpha = rhs.alpha;
        this->beta  = rhs.beta;
        this->gamma = rhs.gamma;

        this->solver      = rhs.solver;
        this->precond     = rhs.precond;
        this->smoother    = rhs.smoother;
        this->matrix      = rhs.matrix;
        this->matrix_type = rhs.matrix_type;

        this->pre_smooth     = rhs.pre_smooth;
        this->post_smooth    = rhs.post_smooth;
        this->ordering       = rhs.ordering;
        this->cycle          = rhs.cycle;
        this->rebuildnumeric = rhs.rebuildnumeric;

        this->coarsening_strategy = rhs.coarsening_strategy;

        this->format = rhs.format;

        this->unit_diag = rhs.unit_diag;

        return *this;
    }
};

#endif // TESTING_UTILITY_HPP
