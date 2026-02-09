/* **************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

/*! \file
    \brief ideal_sizes.hpp gathers all constants that can be tuned for performance.
 *********************************************************************************/

#define BS1 256 // generic 1 dimensional thread-block size used to call common kernels
#define BS2 32 // generic 2 dimensional thread-block size used to call common kernels

/******************************* larf ****************************************
*******************************************************************************/
#ifndef LARF_SSKER_THREADS
#define LARF_SSKER_THREADS 256 // must be 64, 128, 256, 512, or 1024
#endif

#ifndef LARF_SSKER_BLOCKS
#define LARF_SSKER_BLOCKS 64
#endif

#ifndef LARF_SSKER_MAX_DIM
#define LARF_SSKER_MAX_DIM 2048 // should be >= LARF_SSKER_THREADS
#endif

#ifndef LARF_SSKER_MIN_DIM
#define LARF_SSKER_MIN_DIM 64 // should be >= LARF_SSKER_BLOCKS
#endif

/******************************* larfg ****************************************
*******************************************************************************/
#ifndef LARFG_SSKER_THREADS
#define LARFG_SSKER_THREADS 256 // must be 64, 128, 256, 512, or 1024
#endif

#ifndef LARFG_SSKER_MAX_N
#define LARFG_SSKER_MAX_N 2048
#endif

/******************************* larft ****************************************
*******************************************************************************/
#ifndef LARFT_SWITCHSIZE
#define LARFT_SWITCHSIZE 64
#endif

/***************** geqr2/geqrf and geql2/geqlf ********************************
*******************************************************************************/
/*! \brief Determines the size of the block column factorized at each step
    in the blocked QR or QL algorithm (GEQRF or GEQLF). It also applies to the
    corresponding batched and strided-batched routines. */
#ifndef GEQxF_BLOCKSIZE
#define GEQxF_BLOCKSIZE 64
#endif

/*! \brief Determines the size at which rocSOLVER switches from
    the unblocked to the blocked algorithm when executing GEQRF or GEQLF. It also applies to the
    corresponding batched and strided-batched routines.

    \details GEQRF or GEQLF will factorize blocks of GEQxF_BLOCKSIZE columns at a time until
    the rest of the matrix has no more than GEQxF_GEQx2_SWITCHSIZE rows or columns; at this point the last block,
    if any, will be factorized with the unblocked algorithm (GEQR2 or GEQL2).*/
#ifndef GEQxF_GEQx2_SWITCHSIZE
#define GEQxF_GEQx2_SWITCHSIZE 128
#endif

/***************** gerq2/gerqf and gelq2/gelqf ********************************
*******************************************************************************/
/*! \brief Determines the size of the block row factorized at each step
    in the blocked RQ or LQ algorithm (GERQF or GELQF). It also applies to the
    corresponding batched and strided-batched routines. */
#ifndef GExQF_BLOCKSIZE
#define GExQF_BLOCKSIZE 64
#endif

/*! \brief Determines the size at which rocSOLVER switches from
    the unblocked to the blocked algorithm when executing GERQF or GELQF. It also applies to the
    corresponding batched and strided-batched routines.

    \details GERQF or GELQF will factorize blocks of GExQF_BLOCKSIZE rows at a time until
    the rest of the matrix has no more than GExQF_GExQ2_SWITCHSIZE rows or columns; at this point the last block,
    if any, will be factorized with the unblocked algorithm (GERQ2 or GELQ2).*/
#ifndef GExQF_GExQ2_SWITCHSIZE
#define GExQF_GExQ2_SWITCHSIZE 128
#endif

/******** org2r/orgqr, org2l/orgql, ung2r/ungqr and ung2l/ungql ***************
*******************************************************************************/
/*! \brief Determines the size of the block reflector that is applied at each step when
    generating a matrix Q with orthonormal columns with the blocked algorithm (ORGQR/UNGQR or ORGQL/UNGQL). */
#ifndef xxGQx_BLOCKSIZE
#define xxGQx_BLOCKSIZE 64
#endif

/*! \brief Determines the size at which rocSOLVER switches from
    the unblocked to the blocked algorithm when executing ORGQR/UNGQR or ORGQL/UNGQL.

    \details ORGQR/UNGQR or ORGQL/UNGQL will accumulate xxGQx_BLOCKSIZE reflectors at a time until
    there are no more than xxGQx_xxGQx2_SWITCHSIZE reflectors left; the remaining reflectors, if any,
    are applied one by one using the unblocked algorithm (ORG2R/UNG2R or ORG2L/UNG2L).*/
#ifndef xxGQx_xxGQx2_SWITCHSIZE
#define xxGQx_xxGQx2_SWITCHSIZE 128
#endif

/******** orgr2/orgrq, orgl2/orglq, ungr2/ungrq and ungl2/unglq **************
*******************************************************************************/
/*! \brief Determines the size of the block reflector that is applied at each step when
    generating a matrix Q with orthonormal rows with the blocked algorithm (ORGRQ/UNGRQ or ORGLQ/UNGLQ). */
#ifndef xxGxQ_BLOCKSIZE
#define xxGxQ_BLOCKSIZE 64
#endif

/*! \brief Determines the size at which rocSOLVER switches from
    the unblocked to the blocked algorithm when executing ORGRQ/UNGRQ or ORGLQ/UNGLQ.

    \details ORGRQ/UNGRQ or ORGLQ/UNGLQ will accumulate xxGxQ_BLOCKSIZE reflectors at a time until
    there are no more than xxGxQ_xxGxQ2_SWITCHSIZE reflectors left; the remaining reflectors, if any,
    are applied one by one using the unblocked algorithm (ORGR2/UNGR2 or ORGL2/UNGL2).*/
#ifndef xxGxQ_xxGxQ2_SWITCHSIZE
#define xxGxQ_xxGxQ2_SWITCHSIZE 128
#endif

/********* orm2r/ormqr, orm2l/ormql, unm2r/unmqr and unm2l/unmql **************
*******************************************************************************/
/*! \brief Determines the size of the block reflector that multiplies the matrix C at each
    step with the blocked algorithm (ORMQR/UNMQR or ORMQL/UNMQL).

    \details xxMQx_BLOCKSIZE also acts as a switch size; if the total number of reflectors is not greater than xxMQx_BLOCKSIZE (k <= xxMQx_BLOCKSIZE),
    ORMQR/UNMQR or ORMQL/UNMQL will directly call the unblocked routines (ORM2R/UNM2R or ORM2L/UNM2L). However, when k is not a multiple of xxMQx_BLOCKSIZE,
    the last block that updates C in the blocked process is allowed to be smaller than xxMQx_BLOCKSIZE.*/
#ifndef xxMQx_BLOCKSIZE
#define xxMQx_BLOCKSIZE 64
#endif

/********* ormr2/ormrq, orml2/ormlq, unmr2/unmrq and unml2/unmlq ***************
*******************************************************************************/
/*! \brief Determines the size of the block reflector that multiplies the matrix C at each
    step with the blocked algorithm (ORMRQ/UNMRQ or ORMLQ/UNMLQ).

    \details xxMxQ_BLOCKSIZE also acts as a switch size; if the total number of reflectors is not greater than xxMxQ_BLOCKSIZE (k <= xxMxQ_BLOCKSIZE),
    ORMRQ/UNMRQ or ORMLQ/UNMLQ will directly call the unblocked routines (ORMR2/UNMR2 or ORML2/UNML2). However, when k is not a multiple of xxMxQ_BLOCKSIZE,
    the last block that updates C in the blocked process is allowed to be smaller than xxMxQ_BLOCKSIZE.*/
#ifndef xxMxQ_BLOCKSIZE
#define xxMxQ_BLOCKSIZE 64
#endif

/**************************** gebd2/gebrd *************************************
*******************************************************************************/
/*! \brief Determines the size of the leading block that is reduced to bidiagonal form at each step
    when using the blocked algorithm (GEBRD). It also applies to the
    corresponding batched and strided-batched routines.*/
#ifndef GEBRD_BLOCKSIZE
#define GEBRD_BLOCKSIZE 32
#endif

/*! \brief Determines the size at which rocSOLVER switches from
    the unblocked to the blocked algorithm when executing GEBRD. It also applies to the
    corresponding batched and strided-batched routines.

    \details GEBRD will use LABRD to reduce blocks of GEBRD_BLOCKSIZE rows and columns at a time until
    the trailing submatrix has no more than GEBRD_GEBD2_SWITCHSIZE rows or columns; at this point the last block,
    if any, will be reduced with the unblocked algorithm (GEBD2).*/
#ifndef GEBRD_GEBD2_SWITCHSIZE
#define GEBRD_GEBD2_SWITCHSIZE 64
#endif

/******************************* bdsqr ****************************************
*******************************************************************************/
/*! \brief Determines the size at which rocSOLVER switches from updating singular vectors
    in a single thread group, to using multiple thread groups. It also applies to the
    corresponding batched and strided-batched routines.

    \details When nv, nu, and nc are less than or equal to BDSQR_SWITCH_SIZE, BDSQR will update
    the singular vectors in a single thread group. Otherwise, BDSQR will launch a dedicated kernel
    with multiple thread groups.*/
#ifndef BDSQR_SWITCH_SIZE
#define BDSQR_SWITCH_SIZE 512
#endif

/*! \brief Determines the number of iterations that BDSQR will execute between device synchronizations
    in the multi-kernel algorithm.

    \details BDSQR will run an inner loop BDSQR_ITERS_PER_SYNC at a time, before synchronizing with the
    device to check if the stopping criterion has been met. */
#ifndef BDSQR_ITERS_PER_SYNC
#define BDSQR_ITERS_PER_SYNC 10
#endif

/******************************* gesvd ****************************************
*******************************************************************************/
/*! \brief Determines the factor by which one dimension of a matrix should exceed
    the other dimension for the thin SVD to be computed when executing GESVD. It also applies to the
    corresponding batched and strided-batched routines.

    \details When a m-by-n matrix A is passed to GESVD, if m >= THIN_SVD_SWITCH*n or
    n >= THIN_SVD_SWITCH*m, then the thin SVD is computed.*/
#ifndef THIN_SVD_SWITCH
#define THIN_SVD_SWITCH 1.6
#endif

/******************* sytd2/sytrd and hetd2/hetrd *******************************
*******************************************************************************/
/*! \brief Determines the size of the leading block that is reduced to tridiagonal form at each step
    when using the blocked algorithm (SYTRD/HETRD). It also applies to the
    corresponding batched and strided-batched routines.*/
#ifndef xxTRD_BLOCKSIZE
#define xxTRD_BLOCKSIZE 64
#endif

/*! \brief Determines the size at which rocSOLVER switches from
    the unblocked to the blocked algorithm when executing SYTRD/HETRD. It also applies to the
    corresponding batched and strided-batched routines.

    \details SYTRD/HETRD will use LATRD to reduce blocks of xxTRD_BLOCKSIZE rows and columns at a time until
    the rest of the matrix has no more than xxTRD_xxTD2_SWITCHSIZE rows or columns; at this point the last block,
    if any, will be reduced with the unblocked algorithm (SYTD2/HETD2).*/
#ifndef xxTRD_xxTD2_SWITCHSIZE
#define xxTRD_xxTD2_SWITCHSIZE 256
#endif

#ifndef xxTD2_SSKER_MAX_N
#define xxTD2_SSKER_MAX_N 192
#endif

/***************** sygs2/sygst and hegs2/hegst ********************************
*******************************************************************************/
/*! \brief Determines the size of the leading block that is reduced to standard form at each step
    when using the blocked algorithm (SYGST/HEGST). It also applies to the
    corresponding batched and strided-batched routines.

    \details xxGST_BLOCKSIZE also acts as a switch size; if the original size of the problem is not larger than xxGST_BLOCKSIZE (n <= xxGST_BLOCKSIZE),
    SYGST/HEGST will directly call the unblocked routines (SYGS2/HEGS2). However, when n is not a
    multiple of xxGST_BLOCKSIZE, the last block reduced in the blocked process is allowed to be smaller than xxGST_BLOCKSIZE.*/
#ifndef xxGST_BLOCKSIZE
#define xxGST_BLOCKSIZE 64
#endif

/****************************** stedc ******************************************
*******************************************************************************/
/*! \brief Determines the minimum size required for the eigenvectors of an independent block of
    a tridiagonal matrix to be computed using the divide-and-conquer algorithm (STEDC).

    \details If the size of the block is smaller than STEDC_MIN_DC_SIZE (bs < STEDC_MIN_DC_SIZE),
    the eigenvectors are computed with the normal QR algorithm. */
#ifndef STEDC_MIN_DC_SIZE
#define STEDC_MIN_DC_SIZE 16
#endif

/*! \brief Determines the number of split blocks (independent blocks) of a tridiagonal matrix that
    are analyzed in parallel with the divide & conquer method. */
#ifndef STEDC_NUM_SPLIT_BLKS
#define STEDC_NUM_SPLIT_BLKS 8
#endif

/************************** potf2/potrf ***************************************
*******************************************************************************/
/*! \brief Determines the size of the leading block that is factorized at each step
    when using the blocked algorithm (POTRF). It also applies to the
    corresponding batched and strided-batched routines.*/
#ifndef POTRF_BLOCKSIZE
#define POTRF_BLOCKSIZE(T) ((sizeof(T) == 4) ? 180 : (sizeof(T) == 8) ? 127 : 90)
#endif

/*! \brief Determines the size at which rocSOLVER switches from
    the unblocked to the blocked algorithm when executing POTRF. It also applies to the
    corresponding batched and strided-batched routines.

    \details POTRF will factorize blocks of POTRF_BLOCKSIZE columns at a time until
    the rest of the matrix has no more than POTRF_POTF2_SWITCHSIZE columns; at this point the last block,
    if any, will be factorized with the unblocked algorithm (POTF2).*/
#ifndef POTRF_POTF2_SWITCHSIZE
#define POTRF_POTF2_SWITCHSIZE(T) POTRF_BLOCKSIZE(T)
#endif

/*! \brief Determines the maximum size at which rocSOLVER can use POTF2
    \details
    POTF2 will attempt to factorize a small symmetric matrix that can fit entirely
    within the LDS share memory using compact storage.
    The amount of LDS shared memory is assumed to be at least (64 * 1024) bytes. */
#ifndef POTF2_MAX_SMALL_SIZE
#define POTF2_MAX_SMALL_SIZE(T) ((sizeof(T) == 4) ? 180 : (sizeof(T) == 8) ? 127 : 90)
#endif

/************************** syevj/heevj ***************************************
*******************************************************************************/
/*! \brief Determines the size at which rocSOLVER switches from
    the small-size kernel to the blocked algorithm when executing SYEVJ. It also applies to the
    corresponding batched and strided-batched routines. Must be <= 64.

    \details If the size of the matrix is not greater than SYEVJ_BLOCKED_SWITCH, the eigenvalues
    and eigenvectors will be computed with a single kernel call. */
#ifndef SYEVJ_BLOCKED_SWITCH
#define SYEVJ_BLOCKED_SWITCH 58
#endif

/*************************** sytf2/sytrf **************************************
*******************************************************************************/
/*! \brief Determines the maximum size of the partial factorization executed at each step
    when using the blocked algorithm (SYTRF). It also applies to the
    corresponding batched and strided-batched routines.*/
#ifndef SYTRF_BLOCKSIZE
#define SYTRF_BLOCKSIZE 64
#endif

/*! \brief Determines the size at which rocSOLVER switches from
    the unblocked to the blocked algorithm when executing SYTRF. It also applies to the
    corresponding batched and strided-batched routines.

    \details SYTRF will use LASYF to factorize a submatrix of at most SYTRF_BLOCKSIZE columns at a time until
    the rest of the matrix has no more than SYTRF_SYTF2_SWITCHSIZE columns; at this point the last block,
    if any, will be factorized with the unblocked algorithm (SYTF2).*/
#ifndef SYTRF_SYTF2_SWITCHSIZE
#define SYTRF_SYTF2_SWITCHSIZE 128
#endif

/****************************** syevdj ******************************************
*******************************************************************************/
/*! \brief Determines the minimum size required for the Jacobi divide and conquer to be used.

    \details If the size of the block is smaller than SYEVDJ_MIN_DC_SIZE,
    the eigenvectors are computed with the normal Jacobi algorithm. */
#ifndef SYEVDJ_MIN_DC_SIZE
#define SYEVDJ_MIN_DC_SIZE 16
#endif

/****************************** syevdx ******************************************
*******************************************************************************/
/*! \brief Determines the minimum size required for the Bisection divide and conquer to be used.

    \details If the size of the block is smaller than SYEVDX_MIN_DC_SIZE,
    the eigenvectors are computed with the normal inverse iteration algorithm. */
#ifndef SYEVDX_MIN_DC_SIZE
#define SYEVDX_MIN_DC_SIZE 16
#endif

/**************************** getf2/getfr *************************************
*******************************************************************************/
#ifndef GETF2_SPKER_MAX_M
#define GETF2_SPKER_MAX_M 1024 //always <= 1024
#endif
#ifndef GETF2_SPKER_MAX_N
#define GETF2_SPKER_MAX_N 256 //always <= 256
#endif
#ifndef GETF2_SSKER_MAX_M
#define GETF2_SSKER_MAX_M 512 //always <= 512 and <= GETF2_SPKER_MAX_M
#endif
#ifndef GETF2_SSKER_MAX_N
#define GETF2_SSKER_MAX_N 64 //always <= wavefront and <= GETF2_SPKER_MAX_N
#endif
#ifndef GETF2_OPTIM_NGRP
#define GETF2_OPTIM_NGRP \
    16, 15, 8, 8, 8, 8, 8, 8, 6, 6, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
#endif
#ifndef GETRF_NUM_INTERVALS_REAL
#define GETRF_NUM_INTERVALS_REAL 4
#endif
#ifndef GETRF_INTERVALS_REAL
#define GETRF_INTERVALS_REAL 64, 512, 1856, 2944
#endif
#ifndef GETRF_BLKSIZES_REAL
#define GETRF_BLKSIZES_REAL 0, 1, 32, 256, 512
#endif
#ifndef GETRF_BATCH_NUM_INTERVALS_REAL
#define GETRF_BATCH_NUM_INTERVALS_REAL 9
#endif
#ifndef GETRF_BATCH_INTERVALS_REAL
#define GETRF_BATCH_INTERVALS_REAL 40, 42, 46, 49, 52, 58, 112, 800, 1024
#endif
#ifndef GETRF_BATCH_BLKSIZES_REAL
#define GETRF_BATCH_BLKSIZES_REAL 0, 32, 0, 16, 0, 32, 1, 32, 64, 160
#endif
#ifndef GETRF_NPVT_NUM_INTERVALS_REAL
#define GETRF_NPVT_NUM_INTERVALS_REAL 2
#endif
#ifndef GETRF_NPVT_INTERVALS_REAL
#define GETRF_NPVT_INTERVALS_REAL 64, 512
#endif
#ifndef GETRF_NPVT_BLKSIZES_REAL
#define GETRF_NPVT_BLKSIZES_REAL 0, -1, 512
#endif
#ifndef GETRF_NPVT_BATCH_NUM_INTERVALS_REAL
#define GETRF_NPVT_BATCH_NUM_INTERVALS_REAL 6
#endif
#ifndef GETRF_NPVT_BATCH_INTERVALS_REAL
#define GETRF_NPVT_BATCH_INTERVALS_REAL 40, 168, 448, 512, 896, 1408
#endif
#ifndef GETRF_NPVT_BATCH_BLKSIZES_REAL
#define GETRF_NPVT_BATCH_BLKSIZES_REAL 0, -24, -32, -64, 32, 96, 512
#endif

#ifndef GETRF_NUM_INTERVALS_COMPLEX
#define GETRF_NUM_INTERVALS_COMPLEX 4
#endif
#ifndef GETRF_INTERVALS_COMPLEX
#define GETRF_INTERVALS_COMPLEX 64, 512, 1024, 2944
#endif
#ifndef GETRF_BLKSIZES_COMPLEX
#define GETRF_BLKSIZES_COMPLEX 0, 1, 32, 96, 512
#endif
#ifndef GETRF_BATCH_NUM_INTERVALS_COMPLEX
#define GETRF_BATCH_NUM_INTERVALS_COMPLEX 10
#endif
#ifndef GETRF_BATCH_INTERVALS_COMPLEX
#define GETRF_BATCH_INTERVALS_COMPLEX 23, 28, 30, 32, 40, 48, 56, 64, 768, 1024
#endif
#ifndef GETRF_BATCH_BLKSIZES_COMPLEX
#define GETRF_BATCH_BLKSIZES_COMPLEX 0, 16, 0, 1, 24, 16, 24, 16, 48, 64, 160
#endif
#ifndef GETRF_NPVT_NUM_INTERVALS_COMPLEX
#define GETRF_NPVT_NUM_INTERVALS_COMPLEX 2
#endif
#ifndef GETRF_NPVT_INTERVALS_COMPLEX
#define GETRF_NPVT_INTERVALS_COMPLEX 64, 512
#endif
#ifndef GETRF_NPVT_BLKSIZES_COMPLEX
#define GETRF_NPVT_BLKSIZES_COMPLEX 0, -1, 512
#endif
#ifndef GETRF_NPVT_BATCH_NUM_INTERVALS_COMPLEX
#define GETRF_NPVT_BATCH_NUM_INTERVALS_COMPLEX 5
#endif
#ifndef GETRF_NPVT_BATCH_INTERVALS_COMPLEX
#define GETRF_NPVT_BATCH_INTERVALS_COMPLEX 20, 32, 42, 512, 1408
#endif
#ifndef GETRF_NPVT_BATCH_BLKSIZES_COMPLEX
#define GETRF_NPVT_BATCH_BLKSIZES_COMPLEX 0, -16, -32, -48, 64, 128
#endif

/****************************** getri *****************************************
*******************************************************************************/
#ifndef GETRI_MAX_COLS
#define GETRI_MAX_COLS 64 //always <= wavefront size
#endif
#ifndef GETRI_TINY_SIZE
#define GETRI_TINY_SIZE 43
#endif
#ifndef GETRI_NUM_INTERVALS
#define GETRI_NUM_INTERVALS 1
#endif
#ifndef GETRI_INTERVALS
#define GETRI_INTERVALS 1185
#endif
#ifndef GETRI_BLKSIZES
#define GETRI_BLKSIZES 0, 256
#endif
#ifndef GETRI_BATCH_TINY_SIZE
#define GETRI_BATCH_TINY_SIZE 35
#endif
#ifndef GETRI_BATCH_NUM_INTERVALS
#define GETRI_BATCH_NUM_INTERVALS 2
#endif
#ifndef GETRI_BATCH_INTERVALS
#define GETRI_BATCH_INTERVALS 505, 2049
#endif
#ifndef GETRI_BATCH_BLKSIZES
#define GETRI_BATCH_BLKSIZES 32, 0, 256
#endif

/***************************** trtri ******************************************
*******************************************************************************/
#ifndef TRTRI_MAX_COLS
#define TRTRI_MAX_COLS 64 //always <= wavefront size
#endif
#ifndef TRTRI_NUM_INTERVALS
#define TRTRI_NUM_INTERVALS 1
#endif
#ifndef TRTRI_INTERVALS
#define TRTRI_INTERVALS 0
#endif
#ifndef TRTRI_BLKSIZES
#define TRTRI_BLKSIZES 0, 0
#endif
#ifndef TRTRI_BATCH_NUM_INTERVALS
#define TRTRI_BATCH_NUM_INTERVALS 3
#endif
#ifndef TRTRI_BATCH_INTERVALS
#define TRTRI_BATCH_INTERVALS 32, 245, 1009
#endif
#ifndef TRTRI_BATCH_BLKSIZES
#define TRTRI_BATCH_BLKSIZES 0, 16, 32, 0
#endif

/************************** splitlu ***************************************
*******************************************************************************/
/*! \brief Determines the size at which rocSOLVER switches from
    the kernel using a single thread block or using faster algorithm using rocPRIM */

#ifndef SPLITLU_SWITCH_SIZE
#define SPLITLU_SWITCH_SIZE 64
#endif
