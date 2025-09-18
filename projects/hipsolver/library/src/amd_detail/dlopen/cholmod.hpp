/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#include "lib_macros.hpp"

#ifdef HAVE_ROCSPARSE
#include <cholmod.h>
#else

// constants
#define CHOLMOD_INT 0
#define CHOLMOD_INTLONG 1
#define CHOLMOD_LONG 2
#define CHOLMOD_DOUBLE 0
#define CHOLMOD_SINGLE 1

#define CHOLMOD_PATTERN 0
#define CHOLMOD_REAL 1
#define CHOLMOD_COMPLEX 2
#define CHOLMOD_ZOMPLEX 3

#define CHOLMOD_NATURAL 0
#define CHOLMOD_GIVEN 1
#define CHOLMOD_AMD 2
#define CHOLMOD_METIS 3
#define CHOLMOD_NESDIS 4
#define CHOLMOD_COLAMD 5

#define CHOLMOD_A 0
#define CHOLMOD_LDLt 1
#define CHOLMOD_LD 2
#define CHOLMOD_DLt 3
#define CHOLMOD_L 4
#define CHOLMOD_Lt 5
#define CHOLMOD_D 6
#define CHOLMOD_P 7
#define CHOLMOD_Pt 8

#define CHOLMOD_HOST_SUPERNODE_BUFFERS 8
#define CHOLMOD_MAXMETHODS 9

#define CHOLMOD_OK 0
#define CHOLMOD_NOT_POSDEF 1

// type definitions
typedef struct _cholmod_common
{
    double dbound;
    double grow0;
    double grow1;
    size_t grow2;
    size_t maxrank;
    double supernodal_switch;
    int    supernodal;
    int    final_asis;
    int    final_super;
    int    final_ll;
    int    final_pack;
    int    final_monotonic;
    int    final_resymbol;
    double zrelax[3];
    size_t nrelax[3];
    int    prefer_zomplex;
    int    prefer_upper;
    int    quick_return_if_not_posdef;
    int    prefer_binary;
    int    print;
    int    precise;
    int    try_catch;

    void (*error_handler)(int status, const char* file, int line, const char* message);

    int nmethods;
    int current;
    int selected;

    struct _cholmod_method
    {
        double lnz;
        double fl;
        double prune_dense;
        double prune_dense2;
        double nd_oksep;
        double other_1[4];
        size_t nd_small;
        size_t other_2[4];
        int    aggressive;
        int    order_for_lu;
        int    nd_compress;
        int    nd_camd;
        int    nd_components;
        int    ordering;
        size_t other_3[4];
    } method[CHOLMOD_MAXMETHODS + 1];

    int     postorder;
    int     default_nesdis;
    double  metis_memory;
    double  metis_dswitch;
    size_t  metis_nswitch;
    size_t  nrow;
    int64_t mark;
    size_t  iworksize;
    size_t  xworksize;
    void*   Flag;
    void*   Head;
    void*   Xwork;
    void*   Iwork;
    int     itype;
    int     dtype;
    int     no_workspace_reallocate;
    int     status;
    double  fl;
    double  lnz;
    double  anz;
    double  modfl;
    size_t  malloc_count;
    size_t  memory_usage;
    size_t  memory_inuse;
    double  nrealloc_col;
    double  nrealloc_factor;
    double  ndbounds_hit;
    double  rowfacfl;
    double  aatfl;
    int     called_nd;
    int     blas_ok;
    double  SPQR_grain;
    double  SPQR_small;
    int     SPQR_shrink;
    int     SPQR_nthreads;
    double  SPQR_flopcount;
    double  SPQR_analyze_time;
    double  SPQR_factorize_time;
    double  SPQR_solve_time;
    double  SPQR_flopcount_bound;
    double  SPQR_tol_used;
    double  SPQR_norm_E_fro;
    int64_t SPQR_istat[10];
    int     useGPU;
    size_t  maxGpuMemBytes;
    double  maxGpuMemFraction;
    size_t  gpuMemorySize;
    double  gpuKernelTime;
    int64_t gpuFlops;
    int     gpuNumKernelLaunches;
    void*   cublasHandle;
    void*   gpuStream[CHOLMOD_HOST_SUPERNODE_BUFFERS];
    void*   cublasEventPotrf[3];
    void*   updateCKernelsComplete;
    void*   updateCBuffersFree[CHOLMOD_HOST_SUPERNODE_BUFFERS];
    void*   dev_mempool;
    size_t  dev_mempool_size;
    void*   host_pinned_mempool;
    size_t  host_pinned_mempool_size;
    size_t  devBuffSize;
    int     ibuffer;
    double  syrkStart;
    double  cholmod_cpu_gemm_time;
    double  cholmod_cpu_syrk_time;
    double  cholmod_cpu_trsm_time;
    double  cholmod_cpu_potrf_time;
    double  cholmod_gpu_gemm_time;
    double  cholmod_gpu_syrk_time;
    double  cholmod_gpu_trsm_time;
    double  cholmod_gpu_potrf_time;
    double  cholmod_assemble_time;
    double  cholmod_assemble_time2;
    size_t  cholmod_cpu_gemm_calls;
    size_t  cholmod_cpu_syrk_calls;
    size_t  cholmod_cpu_trsm_calls;
    size_t  cholmod_cpu_potrf_calls;
    size_t  cholmod_gpu_gemm_calls;
    size_t  cholmod_gpu_syrk_calls;
    size_t  cholmod_gpu_trsm_calls;
    size_t  cholmod_gpu_potrf_calls;
    double  chunk;
    int     nthreads_max;
} cholmod_common;

typedef struct _cholmod_sparse
{
    size_t nrow;
    size_t ncol;
    size_t nzmax;
    void*  p;
    void*  i;
    void*  nz;
    void*  x;
    void*  z;
    int    stype;
    int    itype;
    int    xtype;
    int    dtype;
    int    sorted;
    int    packed;
} cholmod_sparse;

typedef struct _cholmod_dense
{
    size_t nrow;
    size_t ncol;
    size_t nzmax;
    size_t d;
    void*  x;
    void*  z;
    int    xtype;
    int    dtype;
} cholmod_dense;

typedef struct _cholmod_factor
{
    size_t n;
    size_t minor;
    void*  Perm;
    void*  ColCount;
    void*  IPerm;
    size_t nzmax;
    void*  p;
    void*  i;
    void*  x;
    void*  z;
    void*  nz;
    void*  next;
    void*  prev;
    size_t nsuper;
    size_t ssize;
    size_t xsize;
    size_t maxcsize;
    size_t maxesize;
    void*  super;
    void*  pi;
    void*  px;
    void*  s;
    int    ordering;
    int    is_ll;
    int    is_super;
    int    is_monotonic;
    int    itype;
    int    xtype;
    int    dtype;
    int    useGPU;
} cholmod_factor;

HIPSOLVER_BEGIN_NAMESPACE

// function declarations
typedef int (*fp_cholmod_start)(cholmod_common* common);
extern fp_cholmod_start g_cholmod_start;
#define cholmod_start ::hipsolver::g_cholmod_start

typedef int (*fp_cholmod_finish)(cholmod_common* common);
extern fp_cholmod_finish g_cholmod_finish;
#define cholmod_finish ::hipsolver::g_cholmod_finish

typedef cholmod_sparse* (*fp_cholmod_allocate_sparse)(size_t          nrow,
                                                      size_t          ncol,
                                                      size_t          nzmax,
                                                      int             sorted,
                                                      int             packed,
                                                      int             stype,
                                                      int             xtype,
                                                      cholmod_common* common);
extern fp_cholmod_allocate_sparse g_cholmod_allocate_sparse;
#define cholmod_allocate_sparse ::hipsolver::g_cholmod_allocate_sparse

typedef int (*fp_cholmod_free_sparse)(cholmod_sparse** A, cholmod_common* common);
extern fp_cholmod_free_sparse g_cholmod_free_sparse;
#define cholmod_free_sparse ::hipsolver::g_cholmod_free_sparse

typedef cholmod_dense* (*fp_cholmod_allocate_dense)(
    size_t nrow, size_t ncol, size_t d, int xtype, cholmod_common* common);
extern fp_cholmod_allocate_dense g_cholmod_allocate_dense;
#define cholmod_allocate_dense ::hipsolver::g_cholmod_allocate_dense

typedef int (*fp_cholmod_free_dense)(cholmod_dense** A, cholmod_common* common);
extern fp_cholmod_free_dense g_cholmod_free_dense;
#define cholmod_free_dense ::hipsolver::g_cholmod_free_dense

typedef int (*fp_cholmod_free_factor)(cholmod_factor** L, cholmod_common* common);
extern fp_cholmod_free_factor g_cholmod_free_factor;
#define cholmod_free_factor ::hipsolver::g_cholmod_free_factor

typedef int (*fp_cholmod_drop)(double tol, cholmod_sparse* A, cholmod_common* common);
extern fp_cholmod_drop g_cholmod_drop;
#define cholmod_drop ::hipsolver::g_cholmod_drop

typedef cholmod_factor* (*fp_cholmod_analyze)(cholmod_sparse* A, cholmod_common* common);
extern fp_cholmod_analyze g_cholmod_analyze;
#define cholmod_analyze ::hipsolver::g_cholmod_analyze

typedef int (*fp_cholmod_analyze_ordering)(cholmod_sparse* A,
                                           int             ordering,
                                           int32_t*        perm,
                                           int32_t*        fset,
                                           size_t          fsize,
                                           int32_t*        parent,
                                           int32_t*        post,
                                           int32_t*        col_count,
                                           int32_t*        first,
                                           int32_t*        level,
                                           cholmod_common* common);
extern fp_cholmod_analyze_ordering g_cholmod_analyze_ordering;
#define cholmod_analyze_ordering ::hipsolver::g_cholmod_analyze_ordering

typedef int (*fp_cholmod_factorize)(cholmod_sparse* A, cholmod_factor* L, cholmod_common* common);
extern fp_cholmod_factorize g_cholmod_factorize;
#define cholmod_factorize ::hipsolver::g_cholmod_factorize

typedef cholmod_dense* (*fp_cholmod_solve)(int             sys,
                                           cholmod_factor* L,
                                           cholmod_dense*  B,
                                           cholmod_common* common);
extern fp_cholmod_solve g_cholmod_solve;
#define cholmod_solve ::hipsolver::g_cholmod_solve

HIPSOLVER_END_NAMESPACE

#endif // HAVE_ROCSPARSE

#undef TRUE
#define TRUE 1

HIPSOLVER_BEGIN_NAMESPACE

// load methods
bool try_load_cholmod();

HIPSOLVER_END_NAMESPACE
