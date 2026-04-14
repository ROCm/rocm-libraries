/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#pragma once
/*
 * ============================================================================
 *     Redirect Fortran API to C API
 * ============================================================================
 */
/* ==========
 *   LAPACK
 * ========== */
// orgbr/ungbr
#define hipsolverSorgbr_bufferSizeFortran hipsolverSorgbr_bufferSize
#define hipsolverDorgbr_bufferSizeFortran hipsolverDorgbr_bufferSize
#define hipsolverCungbr_bufferSizeFortran hipsolverCungbr_bufferSize
#define hipsolverZungbr_bufferSizeFortran hipsolverZungbr_bufferSize
#define hipsolverSorgbrFortran hipsolverSorgbr
#define hipsolverDorgbrFortran hipsolverDorgbr
#define hipsolverCungbrFortran hipsolverCungbr
#define hipsolverZungbrFortran hipsolverZungbr
// orgqr/ungqr
#define hipsolverSorgqr_bufferSizeFortran hipsolverSorgqr_bufferSize
#define hipsolverDorgqr_bufferSizeFortran hipsolverDorgqr_bufferSize
#define hipsolverCungqr_bufferSizeFortran hipsolverCungqr_bufferSize
#define hipsolverZungqr_bufferSizeFortran hipsolverZungqr_bufferSize
#define hipsolverSorgqrFortran hipsolverSorgqr
#define hipsolverDorgqrFortran hipsolverDorgqr
#define hipsolverCungqrFortran hipsolverCungqr
#define hipsolverZungqrFortran hipsolverZungqr
// orgtr/ungtr
#define hipsolverSorgtr_bufferSizeFortran hipsolverSorgtr_bufferSize
#define hipsolverDorgtr_bufferSizeFortran hipsolverDorgtr_bufferSize
#define hipsolverCungtr_bufferSizeFortran hipsolverCungtr_bufferSize
#define hipsolverZungtr_bufferSizeFortran hipsolverZungtr_bufferSize
#define hipsolverSorgtrFortran hipsolverSorgtr
#define hipsolverDorgtrFortran hipsolverDorgtr
#define hipsolverCungtrFortran hipsolverCungtr
#define hipsolverZungtrFortran hipsolverZungtr
// ormqr/unmqr
#define hipsolverSormqr_bufferSizeFortran hipsolverSormqr_bufferSize
#define hipsolverDormqr_bufferSizeFortran hipsolverDormqr_bufferSize
#define hipsolverCunmqr_bufferSizeFortran hipsolverCunmqr_bufferSize
#define hipsolverZunmqr_bufferSizeFortran hipsolverZunmqr_bufferSize
#define hipsolverSormqrFortran hipsolverSormqr
#define hipsolverDormqrFortran hipsolverDormqr
#define hipsolverCunmqrFortran hipsolverCunmqr
#define hipsolverZunmqrFortran hipsolverZunmqr
// ormtr/unmtr
#define hipsolverSormtr_bufferSizeFortran hipsolverSormtr_bufferSize
#define hipsolverDormtr_bufferSizeFortran hipsolverDormtr_bufferSize
#define hipsolverCunmtr_bufferSizeFortran hipsolverCunmtr_bufferSize
#define hipsolverZunmtr_bufferSizeFortran hipsolverZunmtr_bufferSize
#define hipsolverSormtrFortran hipsolverSormtr
#define hipsolverDormtrFortran hipsolverDormtr
#define hipsolverCunmtrFortran hipsolverCunmtr
#define hipsolverZunmtrFortran hipsolverZunmtr
// gebrd
#define hipsolverSgebrd_bufferSizeFortran hipsolverSgebrd_bufferSize
#define hipsolverDgebrd_bufferSizeFortran hipsolverDgebrd_bufferSize
#define hipsolverCgebrd_bufferSizeFortran hipsolverCgebrd_bufferSize
#define hipsolverZgebrd_bufferSizeFortran hipsolverZgebrd_bufferSize
#define hipsolverSgebrdFortran hipsolverSgebrd
#define hipsolverDgebrdFortran hipsolverDgebrd
#define hipsolverCgebrdFortran hipsolverCgebrd
#define hipsolverZgebrdFortran hipsolverZgebrd
// gels
#define hipsolverSSgels_bufferSizeFortran hipsolverSSgels_bufferSize
#define hipsolverDDgels_bufferSizeFortran hipsolverDDgels_bufferSize
#define hipsolverCCgels_bufferSizeFortran hipsolverCCgels_bufferSize
#define hipsolverZZgels_bufferSizeFortran hipsolverZZgels_bufferSize
#define hipsolverSSgelsFortran hipsolverSSgels
#define hipsolverDDgelsFortran hipsolverDDgels
#define hipsolverCCgelsFortran hipsolverCCgels
#define hipsolverZZgelsFortran hipsolverZZgels
// geqrf
#define hipsolverSgeqrf_bufferSizeFortran hipsolverSgeqrf_bufferSize
#define hipsolverDgeqrf_bufferSizeFortran hipsolverDgeqrf_bufferSize
#define hipsolverCgeqrf_bufferSizeFortran hipsolverCgeqrf_bufferSize
#define hipsolverZgeqrf_bufferSizeFortran hipsolverZgeqrf_bufferSize
#define hipsolverSgeqrfFortran hipsolverSgeqrf
#define hipsolverDgeqrfFortran hipsolverDgeqrf
#define hipsolverCgeqrfFortran hipsolverCgeqrf
#define hipsolverZgeqrfFortran hipsolverZgeqrf
// gesv
#define hipsolverSSgesv_bufferSizeFortran hipsolverSSgesv_bufferSize
#define hipsolverDDgesv_bufferSizeFortran hipsolverDDgesv_bufferSize
#define hipsolverCCgesv_bufferSizeFortran hipsolverCCgesv_bufferSize
#define hipsolverZZgesv_bufferSizeFortran hipsolverZZgesv_bufferSize
#define hipsolverSSgesvFortran hipsolverSSgesv
#define hipsolverDDgesvFortran hipsolverDDgesv
#define hipsolverCCgesvFortran hipsolverCCgesv
#define hipsolverZZgesvFortran hipsolverZZgesv
// gesvd
#define hipsolverSgesvd_bufferSizeFortran hipsolverSgesvd_bufferSize
#define hipsolverDgesvd_bufferSizeFortran hipsolverDgesvd_bufferSize
#define hipsolverCgesvd_bufferSizeFortran hipsolverCgesvd_bufferSize
#define hipsolverZgesvd_bufferSizeFortran hipsolverZgesvd_bufferSize
#define hipsolverSgesvdFortran hipsolverSgesvd
#define hipsolverDgesvdFortran hipsolverDgesvd
#define hipsolverCgesvdFortran hipsolverCgesvd
#define hipsolverZgesvdFortran hipsolverZgesvd
// gesvdj
#define hipsolverSgesvdj_bufferSizeFortran hipsolverSgesvdj_bufferSize
#define hipsolverDgesvdj_bufferSizeFortran hipsolverDgesvdj_bufferSize
#define hipsolverCgesvdj_bufferSizeFortran hipsolverCgesvdj_bufferSize
#define hipsolverZgesvdj_bufferSizeFortran hipsolverZgesvdj_bufferSize
#define hipsolverSgesvdjFortran hipsolverSgesvdj
#define hipsolverDgesvdjFortran hipsolverDgesvdj
#define hipsolverCgesvdjFortran hipsolverCgesvdj
#define hipsolverZgesvdjFortran hipsolverZgesvdj
// gesvdj_batched
#define hipsolverSgesvdjBatched_bufferSizeFortran hipsolverSgesvdjBatched_bufferSize
#define hipsolverDgesvdjBatched_bufferSizeFortran hipsolverDgesvdjBatched_bufferSize
#define hipsolverCgesvdjBatched_bufferSizeFortran hipsolverCgesvdjBatched_bufferSize
#define hipsolverZgesvdjBatched_bufferSizeFortran hipsolverZgesvdjBatched_bufferSize
#define hipsolverSgesvdjBatchedFortran hipsolverSgesvdjBatched
#define hipsolverDgesvdjBatchedFortran hipsolverDgesvdjBatched
#define hipsolverCgesvdjBatchedFortran hipsolverCgesvdjBatched
#define hipsolverZgesvdjBatchedFortran hipsolverZgesvdjBatched
// getrf
#define hipsolverSgetrf_bufferSizeFortran hipsolverSgetrf_bufferSize
#define hipsolverDgetrf_bufferSizeFortran hipsolverDgetrf_bufferSize
#define hipsolverCgetrf_bufferSizeFortran hipsolverCgetrf_bufferSize
#define hipsolverZgetrf_bufferSizeFortran hipsolverZgetrf_bufferSize
#define hipsolverSgetrfFortran hipsolverSgetrf
#define hipsolverDgetrfFortran hipsolverDgetrf
#define hipsolverCgetrfFortran hipsolverCgetrf
#define hipsolverZgetrfFortran hipsolverZgetrf
// getrs
#define hipsolverSgetrs_bufferSizeFortran hipsolverSgetrs_bufferSize
#define hipsolverDgetrs_bufferSizeFortran hipsolverDgetrs_bufferSize
#define hipsolverCgetrs_bufferSizeFortran hipsolverCgetrs_bufferSize
#define hipsolverZgetrs_bufferSizeFortran hipsolverZgetrs_bufferSize
#define hipsolverSgetrsFortran hipsolverSgetrs
#define hipsolverDgetrsFortran hipsolverDgetrs
#define hipsolverCgetrsFortran hipsolverCgetrs
#define hipsolverZgetrsFortran hipsolverZgetrs
// potrf
#define hipsolverSpotrf_bufferSizeFortran hipsolverSpotrf_bufferSize
#define hipsolverDpotrf_bufferSizeFortran hipsolverDpotrf_bufferSize
#define hipsolverCpotrf_bufferSizeFortran hipsolverCpotrf_bufferSize
#define hipsolverZpotrf_bufferSizeFortran hipsolverZpotrf_bufferSize
#define hipsolverSpotrfFortran hipsolverSpotrf
#define hipsolverDpotrfFortran hipsolverDpotrf
#define hipsolverCpotrfFortran hipsolverCpotrf
#define hipsolverZpotrfFortran hipsolverZpotrf
// potrf_batched
#define hipsolverSpotrfBatched_bufferSizeFortran hipsolverSpotrfBatched_bufferSize
#define hipsolverDpotrfBatched_bufferSizeFortran hipsolverDpotrfBatched_bufferSize
#define hipsolverCpotrfBatched_bufferSizeFortran hipsolverCpotrfBatched_bufferSize
#define hipsolverZpotrfBatched_bufferSizeFortran hipsolverZpotrfBatched_bufferSize
#define hipsolverSpotrfBatchedFortran hipsolverSpotrfBatched
#define hipsolverDpotrfBatchedFortran hipsolverDpotrfBatched
#define hipsolverCpotrfBatchedFortran hipsolverCpotrfBatched
#define hipsolverZpotrfBatchedFortran hipsolverZpotrfBatched
// potri
#define hipsolverSpotri_bufferSizeFortran hipsolverSpotri_bufferSize
#define hipsolverDpotri_bufferSizeFortran hipsolverDpotri_bufferSize
#define hipsolverCpotri_bufferSizeFortran hipsolverCpotri_bufferSize
#define hipsolverZpotri_bufferSizeFortran hipsolverZpotri_bufferSize
#define hipsolverSpotriFortran hipsolverSpotri
#define hipsolverDpotriFortran hipsolverDpotri
#define hipsolverCpotriFortran hipsolverCpotri
#define hipsolverZpotriFortran hipsolverZpotri
// potrs
#define hipsolverSpotrs_bufferSizeFortran hipsolverSpotrs_bufferSize
#define hipsolverDpotrs_bufferSizeFortran hipsolverDpotrs_bufferSize
#define hipsolverCpotrs_bufferSizeFortran hipsolverCpotrs_bufferSize
#define hipsolverZpotrs_bufferSizeFortran hipsolverZpotrs_bufferSize
#define hipsolverSpotrsFortran hipsolverSpotrs
#define hipsolverDpotrsFortran hipsolverDpotrs
#define hipsolverCpotrsFortran hipsolverCpotrs
#define hipsolverZpotrsFortran hipsolverZpotrs
// potrs_batched
#define hipsolverSpotrsBatched_bufferSizeFortran hipsolverSpotrsBatched_bufferSize
#define hipsolverDpotrsBatched_bufferSizeFortran hipsolverDpotrsBatched_bufferSize
#define hipsolverCpotrsBatched_bufferSizeFortran hipsolverCpotrsBatched_bufferSize
#define hipsolverZpotrsBatched_bufferSizeFortran hipsolverZpotrsBatched_bufferSize
#define hipsolverSpotrsBatchedFortran hipsolverSpotrsBatched
#define hipsolverDpotrsBatchedFortran hipsolverDpotrsBatched
#define hipsolverCpotrsBatchedFortran hipsolverCpotrsBatched
#define hipsolverZpotrsBatchedFortran hipsolverZpotrsBatched
// syevd/heevd
#define hipsolverSsyevd_bufferSizeFortran hipsolverSsyevd_bufferSize
#define hipsolverDsyevd_bufferSizeFortran hipsolverDsyevd_bufferSize
#define hipsolverCheevd_bufferSizeFortran hipsolverCheevd_bufferSize
#define hipsolverZheevd_bufferSizeFortran hipsolverZheevd_bufferSize
#define hipsolverSsyevdFortran hipsolverSsyevd
#define hipsolverDsyevdFortran hipsolverDsyevd
#define hipsolverCheevdFortran hipsolverCheevd
#define hipsolverZheevdFortran hipsolverZheevd
// syevj/heevj
#define hipsolverSsyevj_bufferSizeFortran hipsolverSsyevj_bufferSize
#define hipsolverDsyevj_bufferSizeFortran hipsolverDsyevj_bufferSize
#define hipsolverCheevj_bufferSizeFortran hipsolverCheevj_bufferSize
#define hipsolverZheevj_bufferSizeFortran hipsolverZheevj_bufferSize
#define hipsolverSsyevjFortran hipsolverSsyevj
#define hipsolverDsyevjFortran hipsolverDsyevj
#define hipsolverCheevjFortran hipsolverCheevj
#define hipsolverZheevjFortran hipsolverZheevj
// syevj/heevj
#define hipsolverSsyevjBatched_bufferSizeFortran hipsolverSsyevjBatched_bufferSize
#define hipsolverDsyevjBatched_bufferSizeFortran hipsolverDsyevjBatched_bufferSize
#define hipsolverCheevjBatched_bufferSizeFortran hipsolverCheevjBatched_bufferSize
#define hipsolverZheevjBatched_bufferSizeFortran hipsolverZheevjBatched_bufferSize
#define hipsolverSsyevjBatchedFortran hipsolverSsyevjBatched
#define hipsolverDsyevjBatchedFortran hipsolverDsyevjBatched
#define hipsolverCheevjBatchedFortran hipsolverCheevjBatched
#define hipsolverZheevjBatchedFortran hipsolverZheevjBatched
// sygvd/hegvd
#define hipsolverSsygvd_bufferSizeFortran hipsolverSsygvd_bufferSize
#define hipsolverDsygvd_bufferSizeFortran hipsolverDsygvd_bufferSize
#define hipsolverChegvd_bufferSizeFortran hipsolverChegvd_bufferSize
#define hipsolverZhegvd_bufferSizeFortran hipsolverZhegvd_bufferSize
#define hipsolverSsygvdFortran hipsolverSsygvd
#define hipsolverDsygvdFortran hipsolverDsygvd
#define hipsolverChegvdFortran hipsolverChegvd
#define hipsolverZhegvdFortran hipsolverZhegvd
// sygvj/hegvj
#define hipsolverSsygvj_bufferSizeFortran hipsolverSsygvj_bufferSize
#define hipsolverDsygvj_bufferSizeFortran hipsolverDsygvj_bufferSize
#define hipsolverChegvj_bufferSizeFortran hipsolverChegvj_bufferSize
#define hipsolverZhegvj_bufferSizeFortran hipsolverZhegvj_bufferSize
#define hipsolverSsygvjFortran hipsolverSsygvj
#define hipsolverDsygvjFortran hipsolverDsygvj
#define hipsolverChegvjFortran hipsolverChegvj
#define hipsolverZhegvjFortran hipsolverZhegvj
// sytrd/hetrd
#define hipsolverSsytrd_bufferSizeFortran hipsolverSsytrd_bufferSize
#define hipsolverDsytrd_bufferSizeFortran hipsolverDsytrd_bufferSize
#define hipsolverChetrd_bufferSizeFortran hipsolverChetrd_bufferSize
#define hipsolverZhetrd_bufferSizeFortran hipsolverZhetrd_bufferSize
#define hipsolverSsytrdFortran hipsolverSsytrd
#define hipsolverDsytrdFortran hipsolverDsytrd
#define hipsolverChetrdFortran hipsolverChetrd
#define hipsolverZhetrdFortran hipsolverZhetrd
// sytrf
#define hipsolverSsytrf_bufferSizeFortran hipsolverSsytrf_bufferSize
#define hipsolverDsytrf_bufferSizeFortran hipsolverDsytrf_bufferSize
#define hipsolverCsytrf_bufferSizeFortran hipsolverCsytrf_bufferSize
#define hipsolverZsytrf_bufferSizeFortran hipsolverZsytrf_bufferSize
#define hipsolverSsytrfFortran hipsolverSsytrf
#define hipsolverDsytrfFortran hipsolverDsytrf
#define hipsolverCsytrfFortran hipsolverCsytrf
#define hipsolverZsytrfFortran hipsolverZsytrf
