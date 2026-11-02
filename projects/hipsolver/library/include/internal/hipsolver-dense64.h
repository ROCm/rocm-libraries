/* ************************************************************************
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef HIPSOLVER_DENSE64_H
#define HIPSOLVER_DENSE64_H

#include "hipsolver-dense.h"
#include "hipsolver-types.h"

typedef void* hipsolverDnParams_t;

typedef enum
{
    HIPSOLVER_ALG_0 = 231,
    HIPSOLVER_ALG_1 = 232,
} hipsolverAlgMode_t;

typedef enum
{
    HIPSOLVERDN_GETRF = 0,
} hipsolverDnFunction_t;

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Creates a hipsolverDnParams_t object holding advanced options for the
    64-bit generic API.

    @param[out]
    params          pointer to hipsolverDnParams_t on the host.
                    Returns the created object. It must be released with #hipsolverDnDestroyParams.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCreateParams(hipsolverDnParams_t* params);

/*! \brief Destroys a hipsolverDnParams_t object created with #hipsolverDnCreateParams.

    @param[in]
    params          hipsolverDnParams_t.
                    The object to destroy.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDestroyParams(hipsolverDnParams_t params);

/*! \brief Selects the algorithm used by a function of the 64-bit generic API.

    \details
    Not supported by the rocSOLVER backend, which returns HIPSOLVER_STATUS_NOT_SUPPORTED.

    @param[in]
    params          hipsolverDnParams_t.
                    The object to configure.
    @param[in]
    func            hipsolverDnFunction_t.
                    The function the option applies to.
    @param[in]
    alg             hipsolverAlgMode_t.
                    The algorithm to use.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSetAdvOptions(hipsolverDnParams_t   params,
                                                            hipsolverDnFunction_t func,
                                                            hipsolverAlgMode_t    alg);

// geev
/*! \brief Computes the workspace sizes required by #hipsolverDnXgeev.

    \details
    With the rocSOLVER backend, the eigenproblem is solved on the host, so
    lworkOnDevice is 0 and lworkOnHost is nonzero.

    @param[in]
    handle          hipsolverDnHandle_t.
    @param[in]
    params          hipsolverDnParams_t.
                    Advanced options, created with #hipsolverDnCreateParams.
    @param[in]
    jobvl           hipsolverEigMode_t.
                    Specifies whether the left eigenvectors are computed.
                    HIPSOLVER_EIG_MODE_VECTOR is not supported by the rocSOLVER backend.
    @param[in]
    jobvr           hipsolverEigMode_t.
                    Specifies whether the right eigenvectors are computed.
    @param[in]
    n               int64_t. n >= 0.
                    The number of rows and columns of the matrix A.
    @param[in]
    dataTypeA       hipDataType.
                    Data type of the elements of A.
    @param[in]
    A               pointer to dataTypeA. Array on the GPU of dimension lda*n.
                    The matrix that will be passed to #hipsolverDnXgeev.
    @param[in]
    lda             int64_t. lda >= n.
                    Specifies the leading dimension of A.
    @param[in]
    dataTypeW       hipDataType.
                    Data type of the elements of W.
    @param[in]
    W               pointer to dataTypeW.
                    Array on the GPU of dimension n if dataTypeW is complex, or 2*n if
                    dataTypeW is real (real parts followed by imaginary parts).
    @param[in]
    dataTypeVL      hipDataType.
                    Data type of the elements of VL.
    @param[in]
    VL              pointer to dataTypeVL. Array on the GPU of dimension ldvl*n.
    @param[in]
    ldvl            int64_t. ldvl >= 1, and ldvl >= n if jobvl = HIPSOLVER_EIG_MODE_VECTOR.
                    Specifies the leading dimension of VL.
    @param[in]
    dataTypeVR      hipDataType.
                    Data type of the elements of VR.
    @param[in]
    VR              pointer to dataTypeVR. Array on the GPU of dimension ldvr*n.
    @param[in]
    ldvr            int64_t. ldvr >= 1, and ldvr >= n if jobvr = HIPSOLVER_EIG_MODE_VECTOR.
                    Specifies the leading dimension of VR.
    @param[in]
    computeType     hipDataType.
                    Data type used in the computation. Must match dataTypeA.
    @param[out]
    lworkOnDevice   pointer to size_t on the host.
                    Returns the size in bytes of the device workspace workOnDevice
                    required by #hipsolverDnXgeev.
    @param[out]
    lworkOnHost     pointer to size_t on the host.
                    Returns the size in bytes of the host workspace workOnHost
                    required by #hipsolverDnXgeev.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXgeev_bufferSize(hipsolverDnHandle_t handle,
                                                               hipsolverDnParams_t params,
                                                               hipsolverEigMode_t  jobvl,
                                                               hipsolverEigMode_t  jobvr,
                                                               int64_t             n,
                                                               hipDataType         dataTypeA,
                                                               const void*         A,
                                                               int64_t             lda,
                                                               hipDataType         dataTypeW,
                                                               const void*         W,
                                                               hipDataType         dataTypeVL,
                                                               const void*         VL,
                                                               int64_t             ldvl,
                                                               hipDataType         dataTypeVR,
                                                               const void*         VR,
                                                               int64_t             ldvr,
                                                               hipDataType         computeType,
                                                               size_t*             lworkOnDevice,
                                                               size_t*             lworkOnHost);

/*! \brief Computes the eigenvalues and, optionally, the left and/or right
    eigenvectors of a general n-by-n matrix A.

    \details
    With the rocSOLVER backend, the eigenproblem is solved on the host:
    workOnHost is required and workOnDevice is not referenced.

    @param[in]
    handle          hipsolverDnHandle_t.
    @param[in]
    params          hipsolverDnParams_t.
                    Advanced options, created with #hipsolverDnCreateParams.
    @param[in]
    jobvl           hipsolverEigMode_t.
                    Specifies whether the left eigenvectors are computed.
                    HIPSOLVER_EIG_MODE_VECTOR is not supported by the rocSOLVER backend.
    @param[in]
    jobvr           hipsolverEigMode_t.
                    Specifies whether the right eigenvectors are computed.
    @param[in]
    n               int64_t. n >= 0.
                    The number of rows and columns of the matrix A.
    @param[in]
    dataTypeA       hipDataType.
                    Data type of the elements of A.
    @param[inout]
    A               pointer to dataTypeA. Array on the GPU of dimension lda*n.
                    On entry, the matrix A. On exit, the contents of A are destroyed.
    @param[in]
    lda             int64_t. lda >= n.
                    Specifies the leading dimension of A.
    @param[in]
    dataTypeW       hipDataType.
                    Data type of the elements of W.
    @param[out]
    W               pointer to dataTypeW.
                    Array on the GPU of dimension n if dataTypeW is complex, or 2*n if
                    dataTypeW is real (real parts followed by imaginary parts).
                    The computed eigenvalues.
    @param[in]
    dataTypeVL      hipDataType.
                    Data type of the elements of VL.
    @param[out]
    VL              pointer to dataTypeVL. Array on the GPU of dimension ldvl*n.
                    The left eigenvectors, if jobvl = HIPSOLVER_EIG_MODE_VECTOR.
                    Not referenced otherwise.
    @param[in]
    ldvl            int64_t. ldvl >= 1, and ldvl >= n if jobvl = HIPSOLVER_EIG_MODE_VECTOR.
                    Specifies the leading dimension of VL.
    @param[in]
    dataTypeVR      hipDataType.
                    Data type of the elements of VR.
    @param[out]
    VR              pointer to dataTypeVR. Array on the GPU of dimension ldvr*n.
                    The right eigenvectors, if jobvr = HIPSOLVER_EIG_MODE_VECTOR.
                    Not referenced otherwise.
    @param[in]
    ldvr            int64_t. ldvr >= 1, and ldvr >= n if jobvr = HIPSOLVER_EIG_MODE_VECTOR.
                    Specifies the leading dimension of VR.
    @param[in]
    computeType     hipDataType.
                    Data type used in the computation. Must match dataTypeA.
    @param[inout]
    workOnDevice    pointer to void. Workspace array on the GPU of lworkOnDevice bytes.
                    Not referenced by the rocSOLVER backend.
    @param[in]
    lworkOnDevice   size_t.
                    Size in bytes of workOnDevice, as returned by
                    #hipsolverDnXgeev_bufferSize.
    @param[inout]
    workOnHost      pointer to void. Workspace array on the host of lworkOnHost bytes.
    @param[in]
    lworkOnHost     size_t.
                    Size in bytes of workOnHost, as returned by
                    #hipsolverDnXgeev_bufferSize.
    @param[out]
    devInfo         pointer to an int on the GPU.
                    If devInfo = 0, successful exit.
                    If devInfo = i > 0, the QR algorithm failed to compute all the eigenvalues;
                    elements i+1:n of W contain the eigenvalues that converged.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXgeev(hipsolverDnHandle_t handle,
                                                    hipsolverDnParams_t params,
                                                    hipsolverEigMode_t  jobvl,
                                                    hipsolverEigMode_t  jobvr,
                                                    int64_t             n,
                                                    hipDataType         dataTypeA,
                                                    void*               A,
                                                    int64_t             lda,
                                                    hipDataType         dataTypeW,
                                                    void*               W,
                                                    hipDataType         dataTypeVL,
                                                    void*               VL,
                                                    int64_t             ldvl,
                                                    hipDataType         dataTypeVR,
                                                    void*               VR,
                                                    int64_t             ldvr,
                                                    hipDataType         computeType,
                                                    void*               workOnDevice,
                                                    size_t              lworkOnDevice,
                                                    void*               workOnHost,
                                                    size_t              lworkOnHost,
                                                    int*                devInfo);

// geqrf
/*! \brief Computes the workspace sizes required by #hipsolverDnXgeqrf.

    @param[in]
    handle          hipsolverDnHandle_t.
    @param[in]
    params          hipsolverDnParams_t.
                    Advanced options, created with #hipsolverDnCreateParams.
    @param[in]
    m               int64_t. m >= 0.
                    The number of rows of the matrix A.
    @param[in]
    n               int64_t. n >= 0.
                    The number of columns of the matrix A.
    @param[in]
    dataTypeA       hipDataType.
                    Data type of the elements of A.
    @param[in]
    A               pointer to dataTypeA. Array on the GPU of dimension lda*n.
                    The matrix that will be passed to #hipsolverDnXgeqrf.
    @param[in]
    lda             int64_t. lda >= m.
                    Specifies the leading dimension of A.
    @param[in]
    dataTypeTau     hipDataType.
                    Data type of the elements of Tau.
    @param[in]
    tau             pointer to dataTypeTau. Array on the GPU of dimension min(m,n).
    @param[in]
    computeType     hipDataType.
                    Data type used in the computation. Must match dataTypeA.
    @param[out]
    lworkOnDevice   pointer to size_t on the host.
                    Returns the size in bytes of the device workspace workOnDevice
                    required by #hipsolverDnXgeqrf.
    @param[out]
    lworkOnHost     pointer to size_t on the host.
                    Returns the size in bytes of the host workspace workOnHost
                    required by #hipsolverDnXgeqrf.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXgeqrf_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverDnParams_t params,
                                                                int64_t             m,
                                                                int64_t             n,
                                                                hipDataType         dataTypeA,
                                                                const void*         A,
                                                                int64_t             lda,
                                                                hipDataType         dataTypeTau,
                                                                const void*         tau,
                                                                hipDataType         computeType,
                                                                size_t*             lworkOnDevice,
                                                                size_t*             lworkOnHost);

/*! \brief Computes a QR factorization of a general m-by-n matrix A.

    @param[in]
    handle          hipsolverDnHandle_t.
    @param[in]
    params          hipsolverDnParams_t.
                    Advanced options, created with #hipsolverDnCreateParams.
    @param[in]
    m               int64_t. m >= 0.
                    The number of rows of the matrix A.
    @param[in]
    n               int64_t. n >= 0.
                    The number of columns of the matrix A.
    @param[in]
    dataTypeA       hipDataType.
                    Data type of the elements of A.
    @param[inout]
    A               pointer to dataTypeA. Array on the GPU of dimension lda*n.
                    On entry, the m-by-n matrix A to be factored.
                    On exit, the elements on and above the diagonal contain the factor R; the
                    elements below the diagonal are the last m - i elements of Householder
                    vector v_i.
    @param[in]
    lda             int64_t. lda >= m.
                    Specifies the leading dimension of A.
    @param[in]
    dataTypeTau     hipDataType.
                    Data type of the elements of Tau.
    @param[out]
    tau             pointer to dataTypeTau. Array on the GPU of dimension min(m,n).
                    The Householder scalars.
    @param[in]
    computeType     hipDataType.
                    Data type used in the computation. Must match dataTypeA.
    @param[inout]
    workOnDevice    pointer to void. Workspace array on the GPU of lworkOnDevice bytes.
                    If workOnDevice is null or lworkOnDevice is 0, the rocSOLVER backend
                    uses the device workspace managed by the handle instead.
    @param[in]
    lworkOnDevice   size_t.
                    Size in bytes of workOnDevice, as returned by
                    #hipsolverDnXgeqrf_bufferSize.
    @param[inout]
    workOnHost      pointer to void. Workspace array on the host of lworkOnHost bytes.
                    Not referenced by the rocSOLVER backend.
    @param[in]
    lworkOnHost     size_t.
                    Size in bytes of workOnHost, as returned by
                    #hipsolverDnXgeqrf_bufferSize.
    @param[out]
    devInfo         pointer to an int on the GPU.
                    If devInfo = 0, successful exit.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXgeqrf(hipsolverDnHandle_t handle,
                                                     hipsolverDnParams_t params,
                                                     int64_t             m,
                                                     int64_t             n,
                                                     hipDataType         dataTypeA,
                                                     void*               A,
                                                     int64_t             lda,
                                                     hipDataType         dataTypeTau,
                                                     void*               tau,
                                                     hipDataType         computeType,
                                                     void*               workOnDevice,
                                                     size_t              lworkOnDevice,
                                                     void*               workOnHost,
                                                     size_t              lworkOnHost,
                                                     int*                devInfo);

// getrf
/*! \brief Computes the workspace sizes required by #hipsolverDnXgetrf.

    @param[in]
    handle          hipsolverDnHandle_t.
    @param[in]
    params          hipsolverDnParams_t.
                    Advanced options, created with #hipsolverDnCreateParams.
    @param[in]
    m               int64_t. m >= 0.
                    The number of rows of the matrix A.
    @param[in]
    n               int64_t. n >= 0.
                    The number of columns of the matrix A.
    @param[in]
    dataTypeA       hipDataType.
                    Data type of the elements of A.
    @param[in]
    A               pointer to dataTypeA. Array on the GPU of dimension lda*n.
                    The matrix that will be passed to #hipsolverDnXgetrf.
    @param[in]
    lda             int64_t. lda >= m.
                    Specifies the leading dimension of A.
    @param[in]
    computeType     hipDataType.
                    Data type used in the computation. Must match dataTypeA.
    @param[out]
    lworkOnDevice   pointer to size_t on the host.
                    Returns the size in bytes of the device workspace workOnDevice
                    required by #hipsolverDnXgetrf.
    @param[out]
    lworkOnHost     pointer to size_t on the host.
                    Returns the size in bytes of the host workspace workOnHost
                    required by #hipsolverDnXgetrf.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXgetrf_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverDnParams_t params,
                                                                int64_t             m,
                                                                int64_t             n,
                                                                hipDataType         dataTypeA,
                                                                const void*         A,
                                                                int64_t             lda,
                                                                hipDataType         computeType,
                                                                size_t*             lworkOnDevice,
                                                                size_t*             lworkOnHost);

/*! \brief Computes the LU factorization of a general m-by-n matrix A using
    partial pivoting with row interchanges.

    @param[in]
    handle          hipsolverDnHandle_t.
    @param[in]
    params          hipsolverDnParams_t.
                    Advanced options, created with #hipsolverDnCreateParams.
    @param[in]
    m               int64_t. m >= 0.
                    The number of rows of the matrix A.
    @param[in]
    n               int64_t. n >= 0.
                    The number of columns of the matrix A.
    @param[in]
    dataTypeA       hipDataType.
                    Data type of the elements of A.
    @param[inout]
    A               pointer to dataTypeA. Array on the GPU of dimension lda*n.
                    On entry, the m-by-n matrix A to be factored.
                    On exit, the factors L and U from the factorization.
                    The unit diagonal elements of L are not stored.
    @param[in]
    lda             int64_t. lda >= m.
                    Specifies the leading dimension of A.
    @param[out]
    devIpiv         pointer to int64_t. Array on the GPU of dimension min(m,n).
                    The vector of pivot indices. Elements of devIpiv are 1-based indices.
                    If devIpiv is null, the factorization is computed without pivoting.
    @param[in]
    computeType     hipDataType.
                    Data type used in the computation. Must match dataTypeA.
    @param[inout]
    workOnDevice    pointer to void. Workspace array on the GPU of lworkOnDevice bytes.
                    If workOnDevice is null or lworkOnDevice is 0, the rocSOLVER backend
                    uses the device workspace managed by the handle instead.
    @param[in]
    lworkOnDevice   size_t.
                    Size in bytes of workOnDevice, as returned by
                    #hipsolverDnXgetrf_bufferSize.
    @param[inout]
    workOnHost      pointer to void. Workspace array on the host of lworkOnHost bytes.
                    Not referenced by the rocSOLVER backend.
    @param[in]
    lworkOnHost     size_t.
                    Size in bytes of workOnHost, as returned by
                    #hipsolverDnXgetrf_bufferSize.
    @param[out]
    devInfo         pointer to an int on the GPU.
                    If devInfo = 0, successful exit.
                    If devInfo = i > 0, U is singular. U[i,i] is the first zero pivot.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXgetrf(hipsolverDnHandle_t handle,
                                                     hipsolverDnParams_t params,
                                                     int64_t             m,
                                                     int64_t             n,
                                                     hipDataType         dataTypeA,
                                                     void*               A,
                                                     int64_t             lda,
                                                     int64_t*            devIpiv,
                                                     hipDataType         computeType,
                                                     void*               workOnDevice,
                                                     size_t              lworkOnDevice,
                                                     void*               workOnHost,
                                                     size_t              lworkOnHost,
                                                     int*                devInfo);

// getrs
/*! \brief Solves a system of n linear equations A * X = B, A^T * X = B or
    A^H * X = B using the LU factorization computed by #hipsolverDnXgetrf.

    @param[in]
    handle          hipsolverDnHandle_t.
    @param[in]
    params          hipsolverDnParams_t.
                    Advanced options, created with #hipsolverDnCreateParams.
    @param[in]
    trans           hipsolverOperation_t.
                    Specifies the form of the system of equations.
    @param[in]
    n               int64_t. n >= 0.
                    The order of the matrix A.
    @param[in]
    nrhs            int64_t. nrhs >= 0.
                    The number of columns of the matrix B.
    @param[in]
    dataTypeA       hipDataType.
                    Data type of the elements of A.
    @param[in]
    A               pointer to dataTypeA. Array on the GPU of dimension lda*n.
                    The factors L and U of the factorization A = P*L*U returned by
                    #hipsolverDnXgetrf.
    @param[in]
    lda             int64_t. lda >= n.
                    Specifies the leading dimension of A.
    @param[in]
    devIpiv         pointer to int64_t. Array on the GPU of dimension n.
                    The pivot indices returned by #hipsolverDnXgetrf.
                    With the rocSOLVER backend, if devIpiv is null, no row interchanges are applied.
    @param[in]
    dataTypeB       hipDataType.
                    Data type of the elements of B.
    @param[inout]
    B               pointer to dataTypeB. Array on the GPU of dimension ldb*nrhs.
                    On entry, the right hand side matrix B.
                    On exit, the solution matrix X.
    @param[in]
    ldb             int64_t. ldb >= n.
                    Specifies the leading dimension of B.
    @param[out]
    devInfo         pointer to an int on the GPU.
                    If devInfo = 0, successful exit.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXgetrs(hipsolverDnHandle_t  handle,
                                                     hipsolverDnParams_t  params,
                                                     hipsolverOperation_t trans,
                                                     int64_t              n,
                                                     int64_t              nrhs,
                                                     hipDataType          dataTypeA,
                                                     const void*          A,
                                                     int64_t              lda,
                                                     const int64_t*       devIpiv,
                                                     hipDataType          dataTypeB,
                                                     void*                B,
                                                     int64_t              ldb,
                                                     int*                 devInfo);
                                                     
// larft
/*! \brief Computes the workspace sizes required by #hipsolverDnXlarft.

    @param[in]
    handle          hipsolverDnHandle_t.
    @param[in]
    params          hipsolverDnParams_t.
                    Advanced options, created with #hipsolverDnCreateParams.
    @param[in]
    direct          hipsolverDirectMode_t.
                    Specifies the order in which the elementary reflectors are applied.
    @param[in]
    storev          hipsolverStorevMode_t.
                    Specifies how the Householder vectors are stored in V.
    @param[in]
    n               int64_t. n >= 0.
                    The order (size) of the block reflector.
    @param[in]
    k               int64_t. k >= 1.
                    The number of Householder matrices forming H.
    @param[in]
    dataTypeV       hipDataType.
                    Data type of the elements of V.
    @param[in]
    V               pointer to dataTypeV. Array on the GPU of dimension ldv*k if storev is
                    column-wise or ldv*n if row-wise.
                    The matrix of Householder vectors.
    @param[in]
    ldv             int64_t. ldv >= n if storev is column-wise, ldv >= k if row-wise.
                    Specifies the leading dimension of V.
    @param[in]
    dataTypeTau     hipDataType.
                    Data type of the elements of Tau.
    @param[in]
    tau             pointer to dataTypeTau. Array on the GPU of dimension k.
                    The Householder scalars.
    @param[in]
    dataTypeT       hipDataType.
                    Data type of the elements of T.
    @param[in]
    T               pointer to dataTypeT. Array on the GPU of dimension ldt*k.
                    The matrix that will be passed to #hipsolverDnXlarft.
    @param[in]
    ldt             int64_t. ldt >= k.
                    Specifies the leading dimension of T.
    @param[in]
    computeType     hipDataType.
                    Data type used in the computation. Must match dataTypeV.
    @param[out]
    lworkOnDevice   pointer to size_t on the host.
                    Returns the size in bytes of the device workspace workOnDevice
                    required by #hipsolverDnXlarft.
    @param[out]
    lworkOnHost     pointer to size_t on the host.
                    Returns the size in bytes of the host workspace workOnHost
                    required by #hipsolverDnXlarft.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXlarft_bufferSize(hipsolverDnHandle_t   handle,
                                                                hipsolverDnParams_t   params,
                                                                hipsolverDirectMode_t direct,
                                                                hipsolverStorevMode_t storev,
                                                                int64_t               n,
                                                                int64_t               k,
                                                                hipDataType           dataTypeV,
                                                                const void*           V,
                                                                int64_t               ldv,
                                                                hipDataType           dataTypeTau,
                                                                const void*           tau,
                                                                hipDataType           dataTypeT,
                                                                void*                 T,
                                                                int64_t               ldt,
                                                                hipDataType           computeType,
                                                                size_t*               lworkOnDevice,
                                                                size_t*               lworkOnHost);

/*! \brief Generates the triangular factor T of a block reflector H of order n,
    defined as the product of k Householder matrices.

    @param[in]
    handle          hipsolverDnHandle_t.
    @param[in]
    params          hipsolverDnParams_t.
                    Advanced options, created with #hipsolverDnCreateParams.
    @param[in]
    direct          hipsolverDirectMode_t.
                    Specifies the order in which the elementary reflectors are applied.
    @param[in]
    storev          hipsolverStorevMode_t.
                    Specifies how the Householder vectors are stored in V.
    @param[in]
    n               int64_t. n >= 0.
                    The order (size) of the block reflector.
    @param[in]
    k               int64_t. k >= 1.
                    The number of Householder matrices forming H.
    @param[in]
    dataTypeV       hipDataType.
                    Data type of the elements of V.
    @param[in]
    V               pointer to dataTypeV. Array on the GPU of dimension ldv*k if storev is
                    column-wise or ldv*n if row-wise.
                    The matrix of Householder vectors.
    @param[in]
    ldv             int64_t. ldv >= n if storev is column-wise, ldv >= k if row-wise.
                    Specifies the leading dimension of V.
    @param[in]
    dataTypeTau     hipDataType.
                    Data type of the elements of Tau.
    @param[in]
    tau             pointer to dataTypeTau. Array on the GPU of dimension k.
                    The Householder scalars.
    @param[in]
    dataTypeT       hipDataType.
                    Data type of the elements of T.
    @param[out]
    T               pointer to dataTypeT. Array on the GPU of dimension ldt*k.
                    The triangular factor of the block reflector.
    @param[in]
    ldt             int64_t. ldt >= k.
                    Specifies the leading dimension of T.
    @param[in]
    computeType     hipDataType.
                    Data type used in the computation. Must match dataTypeV.
    @param[inout]
    workOnDevice    pointer to void. Workspace array on the GPU of lworkOnDevice bytes.
                    If workOnDevice is null or lworkOnDevice is 0, the rocSOLVER backend
                    uses the device workspace managed by the handle instead.
    @param[in]
    lworkOnDevice   size_t.
                    Size in bytes of workOnDevice, as returned by
                    #hipsolverDnXlarft_bufferSize.
    @param[inout]
    workOnHost      pointer to void. Workspace array on the host of lworkOnHost bytes.
                    Not referenced by the rocSOLVER backend.
    @param[in]
    lworkOnHost     size_t.
                    Size in bytes of workOnHost, as returned by
                    #hipsolverDnXlarft_bufferSize.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXlarft(hipsolverDnHandle_t   handle,
                                                     hipsolverDnParams_t   params,
                                                     hipsolverDirectMode_t direct,
                                                     hipsolverStorevMode_t storev,
                                                     int64_t               n,
                                                     int64_t               k,
                                                     hipDataType           dataTypeV,
                                                     void*                 V,
                                                     int64_t               ldv,
                                                     hipDataType           dataTypeTau,
                                                     void*                 tau,
                                                     hipDataType           dataTypeT,
                                                     void*                 T,
                                                     int64_t               ldt,
                                                     hipDataType           computeType,
                                                     void*                 workOnDevice,
                                                     size_t                lworkOnDevice,
                                                     void*                 workOnHost,
                                                     size_t                lworkOnHost);

// potrf
/*! \brief Computes the workspace sizes required by #hipsolverDnXpotrf.

    @param[in]
    handle          hipsolverDnHandle_t.
    @param[in]
    params          hipsolverDnParams_t.
                    Advanced options, created with #hipsolverDnCreateParams.
    @param[in]
    uplo            hipsolverFillMode_t.
                    Specifies whether the upper or lower triangle of A is used.
    @param[in]
    n               int64_t. n >= 0.
                    The number of rows and columns of the matrix A.
    @param[in]
    dataTypeA       hipDataType.
                    Data type of the elements of A.
    @param[in]
    A               pointer to dataTypeA. Array on the GPU of dimension lda*n.
                    The matrix that will be passed to #hipsolverDnXpotrf.
    @param[in]
    lda             int64_t. lda >= n.
                    Specifies the leading dimension of A.
    @param[in]
    computeType     hipDataType.
                    Data type used in the computation. Must match dataTypeA.
    @param[out]
    lworkOnDevice   pointer to size_t on the host.
                    Returns the size in bytes of the device workspace workOnDevice
                    required by #hipsolverDnXpotrf.
    @param[out]
    lworkOnHost     pointer to size_t on the host.
                    Returns the size in bytes of the host workspace workOnHost
                    required by #hipsolverDnXpotrf.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXpotrf_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverDnParams_t params,
                                                                hipsolverFillMode_t uplo,
                                                                int64_t             n,
                                                                hipDataType         dataTypeA,
                                                                const void*         A,
                                                                int64_t             lda,
                                                                hipDataType         computeType,
                                                                size_t*             lworkOnDevice,
                                                                size_t*             lworkOnHost);

/*! \brief Computes the Cholesky factorization of a real symmetric (complex
    Hermitian) positive definite matrix A.

    @param[in]
    handle          hipsolverDnHandle_t.
    @param[in]
    params          hipsolverDnParams_t.
                    Advanced options, created with #hipsolverDnCreateParams.
    @param[in]
    uplo            hipsolverFillMode_t.
                    Specifies whether the upper or lower triangle of A is used.
    @param[in]
    n               int64_t. n >= 0.
                    The number of rows and columns of the matrix A.
    @param[in]
    dataTypeA       hipDataType.
                    Data type of the elements of A.
    @param[inout]
    A               pointer to dataTypeA. Array on the GPU of dimension lda*n.
                    On entry, the matrix A to be factored.
                    On exit, the factor U or L of the Cholesky factorization.
    @param[in]
    lda             int64_t. lda >= n.
                    Specifies the leading dimension of A.
    @param[in]
    computeType     hipDataType.
                    Data type used in the computation. Must match dataTypeA.
    @param[inout]
    workOnDevice    pointer to void. Workspace array on the GPU of lworkOnDevice bytes.
                    If workOnDevice is null or lworkOnDevice is 0, the rocSOLVER backend
                    uses the device workspace managed by the handle instead.
    @param[in]
    lworkOnDevice   size_t.
                    Size in bytes of workOnDevice, as returned by
                    #hipsolverDnXpotrf_bufferSize.
    @param[inout]
    workOnHost      pointer to void. Workspace array on the host of lworkOnHost bytes.
                    Not referenced by the rocSOLVER backend.
    @param[in]
    lworkOnHost     size_t.
                    Size in bytes of workOnHost, as returned by
                    #hipsolverDnXpotrf_bufferSize.
    @param[out]
    info            pointer to an int on the GPU.
                    If info = 0, successful exit.
                    If info = i > 0, the leading minor of order i of A is not positive definite.
                    The factorization stopped at this point.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXpotrf(hipsolverDnHandle_t handle,
                                                     hipsolverDnParams_t params,
                                                     hipsolverFillMode_t uplo,
                                                     int64_t             n,
                                                     hipDataType         dataTypeA,
                                                     void*               A,
                                                     int64_t             lda,
                                                     hipDataType         computeType,
                                                     void*               workOnDevice,
                                                     size_t              lworkOnDevice,
                                                     void*               workOnHost,
                                                     size_t              lworkOnHost,
                                                     int*                info);

// potrs
/*! \brief Solves a system of n linear equations A * X = B using the Cholesky
    factorization computed by #hipsolverDnXpotrf.

    @param[in]
    handle          hipsolverDnHandle_t.
    @param[in]
    params          hipsolverDnParams_t.
                    Advanced options, created with #hipsolverDnCreateParams.
    @param[in]
    uplo            hipsolverFillMode_t.
                    Specifies whether the upper or lower triangle of A is used.
    @param[in]
    n               int64_t. n >= 0.
                    The order of the matrix A.
    @param[in]
    nrhs            int64_t. nrhs >= 0.
                    The number of columns of the matrix B.
    @param[in]
    dataTypeA       hipDataType.
                    Data type of the elements of A.
    @param[in]
    A               pointer to dataTypeA. Array on the GPU of dimension lda*n.
                    The factor U or L of the Cholesky factorization returned by #hipsolverDnXpotrf.
    @param[in]
    lda             int64_t. lda >= n.
                    Specifies the leading dimension of A.
    @param[in]
    dataTypeB       hipDataType.
                    Data type of the elements of B.
    @param[inout]
    B               pointer to dataTypeB. Array on the GPU of dimension ldb*nrhs.
                    On entry, the right hand side matrix B.
                    On exit, the solution matrix X.
    @param[in]
    ldb             int64_t. ldb >= n.
                    Specifies the leading dimension of B.
    @param[out]
    info            pointer to an int on the GPU.
                    If info = 0, successful exit.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXpotrs(hipsolverDnHandle_t handle,
                                                     hipsolverDnParams_t params,
                                                     hipsolverFillMode_t uplo,
                                                     int64_t             n,
                                                     int64_t             nrhs,
                                                     hipDataType         dataTypeA,
                                                     const void*         A,
                                                     int64_t             lda,
                                                     hipDataType         dataTypeB,
                                                     void*               B,
                                                     int64_t             ldb,
                                                     int*                info);

// syevd
/*! \brief Computes the workspace sizes required by #hipsolverDnXsyevd.

    @param[in]
    handle          hipsolverDnHandle_t.
    @param[in]
    params          hipsolverDnParams_t.
                    Advanced options, created with #hipsolverDnCreateParams.
    @param[in]
    jobz            hipsolverEigMode_t.
                    Specifies whether the eigenvectors are computed.
    @param[in]
    uplo            hipsolverFillMode_t.
                    Specifies whether the upper or lower triangle of A is used.
    @param[in]
    n               int64_t. n >= 0.
                    The number of rows and columns of the matrix A.
    @param[in]
    dataTypeA       hipDataType.
                    Data type of the elements of A.
    @param[in]
    A               pointer to dataTypeA. Array on the GPU of dimension lda*n.
                    The matrix that will be passed to #hipsolverDnXsyevd.
    @param[in]
    lda             int64_t. lda >= n.
                    Specifies the leading dimension of A.
    @param[in]
    dataTypeW       hipDataType.
                    Data type of the elements of W.
    @param[in]
    W               pointer to dataTypeW. Array on the GPU of dimension n.
    @param[in]
    computeType     hipDataType.
                    Data type used in the computation. Must match dataTypeA.
    @param[out]
    lworkOnDevice   pointer to size_t on the host.
                    Returns the size in bytes of the device workspace workOnDevice
                    required by #hipsolverDnXsyevd.
    @param[out]
    lworkOnHost     pointer to size_t on the host.
                    Returns the size in bytes of the host workspace workOnHost
                    required by #hipsolverDnXsyevd.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsyevd_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverDnParams_t params,
                                                                hipsolverEigMode_t  jobz,
                                                                hipsolverFillMode_t uplo,
                                                                int64_t             n,
                                                                hipDataType         dataTypeA,
                                                                const void*         A,
                                                                int64_t             lda,
                                                                hipDataType         dataTypeW,
                                                                const void*         W,
                                                                hipDataType         computeType,
                                                                size_t*             lworkOnDevice,
                                                                size_t*             lworkOnHost);

/*! \brief Computes the eigenvalues and, optionally, the eigenvectors of a real
    symmetric (complex Hermitian) matrix A using the divide and conquer method.

    @param[in]
    handle          hipsolverDnHandle_t.
    @param[in]
    params          hipsolverDnParams_t.
                    Advanced options, created with #hipsolverDnCreateParams.
    @param[in]
    jobz            hipsolverEigMode_t.
                    Specifies whether the eigenvectors are computed.
    @param[in]
    uplo            hipsolverFillMode_t.
                    Specifies whether the upper or lower triangle of A is used.
    @param[in]
    n               int64_t. n >= 0.
                    The number of rows and columns of the matrix A.
    @param[in]
    dataTypeA       hipDataType.
                    Data type of the elements of A.
    @param[inout]
    A               pointer to dataTypeA. Array on the GPU of dimension lda*n.
                    On entry, the matrix A.
                    On exit, the eigenvectors of A if jobz = HIPSOLVER_EIG_MODE_VECTOR.
                    Otherwise, the contents of A are destroyed.
    @param[in]
    lda             int64_t. lda >= n.
                    Specifies the leading dimension of A.
    @param[in]
    dataTypeW       hipDataType.
                    Data type of the elements of W.
    @param[out]
    W               pointer to dataTypeW. Array on the GPU of dimension n.
                    The eigenvalues of A in ascending order.
    @param[in]
    computeType     hipDataType.
                    Data type used in the computation. Must match dataTypeA.
    @param[inout]
    workOnDevice    pointer to void. Workspace array on the GPU of lworkOnDevice bytes.
                    If workOnDevice is null or lworkOnDevice is 0, the rocSOLVER backend
                    uses the device workspace managed by the handle instead.
    @param[in]
    lworkOnDevice   size_t.
                    Size in bytes of workOnDevice, as returned by
                    #hipsolverDnXsyevd_bufferSize.
    @param[inout]
    workOnHost      pointer to void. Workspace array on the host of lworkOnHost bytes.
                    Not referenced by the rocSOLVER backend.
    @param[in]
    lworkOnHost     size_t.
                    Size in bytes of workOnHost, as returned by
                    #hipsolverDnXsyevd_bufferSize.
    @param[out]
    devInfo         pointer to an int on the GPU.
                    If devInfo = 0, successful exit.
                    If devInfo = i > 0, the algorithm did not converge.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsyevd(hipsolverDnHandle_t handle,
                                                     hipsolverDnParams_t params,
                                                     hipsolverEigMode_t  jobz,
                                                     hipsolverFillMode_t uplo,
                                                     int64_t             n,
                                                     hipDataType         dataTypeA,
                                                     void*               A,
                                                     int64_t             lda,
                                                     hipDataType         dataTypeW,
                                                     void*               W,
                                                     hipDataType         computeType,
                                                     void*               workOnDevice,
                                                     size_t              lworkOnDevice,
                                                     void*               workOnHost,
                                                     size_t              lworkOnHost,
                                                     int*                devInfo);

// syevBatched
/*! \brief Computes the workspace sizes required by #hipsolverDnXsyevBatched.

    @param[in]
    handle          hipsolverDnHandle_t.
    @param[in]
    params          hipsolverDnParams_t.
                    Advanced options, created with #hipsolverDnCreateParams.
    @param[in]
    jobz            hipsolverEigMode_t.
                    Specifies whether the eigenvectors are computed.
    @param[in]
    uplo            hipsolverFillMode_t.
                    Specifies whether the upper or lower triangle of A is used.
    @param[in]
    n               int64_t. n >= 0.
                    The number of rows and columns of each matrix A_j.
    @param[in]
    dataTypeA       hipDataType.
                    Data type of the elements of A.
    @param[in]
    A               pointer to dataTypeA. Array on the GPU of dimension lda*n*batchSize.
                    The matrices that will be passed to #hipsolverDnXsyevBatched.
    @param[in]
    lda             int64_t. lda >= n.
                    Specifies the leading dimension of each A_j.
    @param[in]
    dataTypeW       hipDataType.
                    Data type of the elements of W.
    @param[in]
    W               pointer to dataTypeW. Array on the GPU of dimension n*batchSize.
    @param[in]
    computeType     hipDataType.
                    Data type used in the computation. Must match dataTypeA.
    @param[out]
    lworkOnDevice   pointer to size_t on the host.
                    Returns the size in bytes of the device workspace workOnDevice
                    required by #hipsolverDnXsyevBatched.
    @param[out]
    lworkOnHost     pointer to size_t on the host.
                    Returns the size in bytes of the host workspace workOnHost
                    required by #hipsolverDnXsyevBatched.
    @param[in]
    batchSize       int64_t. batchSize >= 0.
                    The number of matrices in the batch.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsyevBatched_bufferSize(hipsolverDnHandle_t handle,
                                                                      hipsolverDnParams_t params,
                                                                      hipsolverEigMode_t  jobz,
                                                                      hipsolverFillMode_t uplo,
                                                                      int64_t             n,
                                                                      hipDataType         dataTypeA,
                                                                      const void*         A,
                                                                      int64_t             lda,
                                                                      hipDataType         dataTypeW,
                                                                      const void*         W,
                                                                      hipDataType computeType,
                                                                      size_t*     lworkOnDevice,
                                                                      size_t*     lworkOnHost,
                                                                      int64_t     batchSize);

/*! \brief Computes the eigenvalues and, optionally, the eigenvectors of a batch
    of real symmetric (complex Hermitian) matrices A_j.

    @param[in]
    handle          hipsolverDnHandle_t.
    @param[in]
    params          hipsolverDnParams_t.
                    Advanced options, created with #hipsolverDnCreateParams.
    @param[in]
    jobz            hipsolverEigMode_t.
                    Specifies whether the eigenvectors are computed.
    @param[in]
    uplo            hipsolverFillMode_t.
                    Specifies whether the upper or lower triangle of A is used.
    @param[in]
    n               int64_t. n >= 0.
                    The number of rows and columns of each matrix A_j.
    @param[in]
    dataTypeA       hipDataType.
                    Data type of the elements of A.
    @param[inout]
    A               pointer to dataTypeA. Array on the GPU of dimension lda*n*batchSize.
                    On entry, the matrices A_j, stored contiguously with stride lda*n.
                    On exit, the eigenvectors of A_j if jobz = HIPSOLVER_EIG_MODE_VECTOR.
                    Otherwise, the contents of A_j are destroyed.
    @param[in]
    lda             int64_t. lda >= n.
                    Specifies the leading dimension of each A_j.
    @param[in]
    dataTypeW       hipDataType.
                    Data type of the elements of W.
    @param[out]
    W               pointer to dataTypeW. Array on the GPU of dimension n*batchSize.
                    The eigenvalues of A_j in ascending order, stored contiguously with stride n.
    @param[in]
    computeType     hipDataType.
                    Data type used in the computation. Must match dataTypeA.
    @param[inout]
    workOnDevice    pointer to void. Workspace array on the GPU of lworkOnDevice bytes.
                    If workOnDevice is null or lworkOnDevice is 0, the rocSOLVER backend
                    uses the device workspace managed by the handle instead.
    @param[in]
    lworkOnDevice   size_t.
                    Size in bytes of workOnDevice, as returned by
                    #hipsolverDnXsyevBatched_bufferSize.
    @param[inout]
    workOnHost      pointer to void. Workspace array on the host of lworkOnHost bytes.
                    Not referenced by the rocSOLVER backend.
    @param[in]
    lworkOnHost     size_t.
                    Size in bytes of workOnHost, as returned by
                    #hipsolverDnXsyevBatched_bufferSize.
    @param[out]
    devInfo         pointer to int. Array of batchSize integers on the GPU.
                    If devInfo[j] = 0, successful exit for matrix A_j.
                    If devInfo[j] = i > 0, the algorithm did not converge for matrix A_j.
    @param[in]
    batchSize       int64_t. batchSize >= 0.
                    The number of matrices in the batch.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsyevBatched(hipsolverDnHandle_t handle,
                                                           hipsolverDnParams_t params,
                                                           hipsolverEigMode_t  jobz,
                                                           hipsolverFillMode_t uplo,
                                                           int64_t             n,
                                                           hipDataType         dataTypeA,
                                                           void*               A,
                                                           int64_t             lda,
                                                           hipDataType         dataTypeW,
                                                           void*               W,
                                                           hipDataType         computeType,
                                                           void*               workOnDevice,
                                                           size_t              lworkOnDevice,
                                                           void*               workOnHost,
                                                           size_t              lworkOnHost,
                                                           int*                devInfo,
                                                           int64_t             batchSize);

// sytrs
/*! \brief Computes the workspace sizes required by #hipsolverDnXsytrs.

    @param[in]
    handle          hipsolverDnHandle_t.
    @param[in]
    uplo            hipsolverFillMode_t.
                    Specifies whether the factorization in A is upper or lower triangular.
    @param[in]
    n               int64_t. n >= 0.
                    The order of the matrix A.
    @param[in]
    nrhs            int64_t. nrhs >= 0.
                    The number of columns of the matrix B.
    @param[in]
    dataTypeA       hipDataType.
                    Data type of the elements of A.
    @param[in]
    A               pointer to dataTypeA. Array on the GPU of dimension lda*n.
                    The block diagonal matrix D and the multipliers from the U*D*U^T or L*D*L^T
                    factorization of A.
    @param[in]
    lda             int64_t. lda >= n.
                    Specifies the leading dimension of A.
    @param[in]
    devIpiv         pointer to int64_t. Array on the GPU of dimension n.
                    The details of the interchanges and the block structure of D, as returned
                    by the sytrf factorization.
    @param[in]
    dataTypeB       hipDataType.
                    Data type of the elements of B.
    @param[in]
    B               pointer to dataTypeB. Array on the GPU of dimension ldb*nrhs.
                    The matrix that will be passed to #hipsolverDnXsytrs.
    @param[in]
    ldb             int64_t. ldb >= n.
                    Specifies the leading dimension of B.
    @param[out]
    lworkOnDevice   pointer to size_t on the host.
                    Returns the size in bytes of the device workspace workOnDevice
                    required by #hipsolverDnXsytrs.
    @param[out]
    lworkOnHost     pointer to size_t on the host.
                    Returns the size in bytes of the host workspace workOnHost
                    required by #hipsolverDnXsytrs.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsytrs_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverFillMode_t uplo,
                                                                int64_t             n,
                                                                int64_t             nrhs,
                                                                hipDataType         dataTypeA,
                                                                const void*         A,
                                                                int64_t             lda,
                                                                const int64_t*      devIpiv,
                                                                hipDataType         dataTypeB,
                                                                void*               B,
                                                                int64_t             ldb,
                                                                size_t*             lworkOnDevice,
                                                                size_t*             lworkOnHost);

/*! \brief Solves a system of n linear equations A * X = B using the U*D*U^T or
    L*D*L^T factorization of a symmetric matrix A computed by sytrf.

    @param[in]
    handle          hipsolverDnHandle_t.
    @param[in]
    uplo            hipsolverFillMode_t.
                    Specifies whether the factorization in A is upper or lower triangular.
    @param[in]
    n               int64_t. n >= 0.
                    The order of the matrix A.
    @param[in]
    nrhs            int64_t. nrhs >= 0.
                    The number of columns of the matrix B.
    @param[in]
    dataTypeA       hipDataType.
                    Data type of the elements of A.
    @param[in]
    A               pointer to dataTypeA. Array on the GPU of dimension lda*n.
                    The block diagonal matrix D and the multipliers from the U*D*U^T or L*D*L^T
                    factorization of A.
    @param[in]
    lda             int64_t. lda >= n.
                    Specifies the leading dimension of A.
    @param[in]
    devIpiv         pointer to int64_t. Array on the GPU of dimension n.
                    The details of the interchanges and the block structure of D, as returned
                    by the sytrf factorization.
    @param[in]
    dataTypeB       hipDataType.
                    Data type of the elements of B.
    @param[inout]
    B               pointer to dataTypeB. Array on the GPU of dimension ldb*nrhs.
                    On entry, the right hand side matrix B.
                    On exit, the solution matrix X.
    @param[in]
    ldb             int64_t. ldb >= n.
                    Specifies the leading dimension of B.
    @param[inout]
    workOnDevice    pointer to void. Workspace array on the GPU of lworkOnDevice bytes.
                    If workOnDevice is null or lworkOnDevice is 0, the rocSOLVER backend
                    uses the device workspace managed by the handle instead.
    @param[in]
    lworkOnDevice   size_t.
                    Size in bytes of workOnDevice, as returned by
                    #hipsolverDnXsytrs_bufferSize.
    @param[inout]
    workOnHost      pointer to void. Workspace array on the host of lworkOnHost bytes.
                    Not referenced by the rocSOLVER backend.
    @param[in]
    lworkOnHost     size_t.
                    Size in bytes of workOnHost, as returned by
                    #hipsolverDnXsytrs_bufferSize.
    @param[out]
    devInfo         pointer to an int on the GPU.
                    If devInfo = 0, successful exit.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsytrs(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int64_t             n,
                                                     int64_t             nrhs,
                                                     hipDataType         dataTypeA,
                                                     const void*         A,
                                                     int64_t             lda,
                                                     const int64_t*      devIpiv,
                                                     hipDataType         dataTypeB,
                                                     void*               B,
                                                     int64_t             ldb,
                                                     void*               workOnDevice,
                                                     size_t              lworkOnDevice,
                                                     void*               workOnHost,
                                                     size_t              lworkOnHost,
                                                     int*                devInfo);

// trtri
/*! \brief Computes the workspace sizes required by #hipsolverDnXtrtri.

    @param[in]
    handle          hipsolverDnHandle_t.
    @param[in]
    uplo            hipsolverFillMode_t.
                    Specifies whether A is upper or lower triangular.
    @param[in]
    diag            hipsolverDiagType_t.
                    Specifies whether A is unit triangular.
    @param[in]
    n               int64_t. n >= 0.
                    The number of rows and columns of the matrix A.
    @param[in]
    dataTypeA       hipDataType.
                    Data type of the elements of A.
    @param[in]
    A               pointer to dataTypeA. Array on the GPU of dimension lda*n.
                    The matrix that will be passed to #hipsolverDnXtrtri.
    @param[in]
    lda             int64_t. lda >= n.
                    Specifies the leading dimension of A.
    @param[out]
    lworkOnDevice   pointer to size_t on the host.
                    Returns the size in bytes of the device workspace workOnDevice
                    required by #hipsolverDnXtrtri.
    @param[out]
    lworkOnHost     pointer to size_t on the host.
                    Returns the size in bytes of the host workspace workOnHost
                    required by #hipsolverDnXtrtri.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXtrtri_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverFillMode_t uplo,
                                                                hipsolverDiagType_t diag,
                                                                int64_t             n,
                                                                hipDataType         dataTypeA,
                                                                void*               A,
                                                                int64_t             lda,
                                                                size_t*             lworkOnDevice,
                                                                size_t*             lworkOnHost);

/*! \brief Computes the inverse of a triangular matrix A.

    @param[in]
    handle          hipsolverDnHandle_t.
    @param[in]
    uplo            hipsolverFillMode_t.
                    Specifies whether A is upper or lower triangular.
    @param[in]
    diag            hipsolverDiagType_t.
                    Specifies whether A is unit triangular.
    @param[in]
    n               int64_t. n >= 0.
                    The number of rows and columns of the matrix A.
    @param[in]
    dataTypeA       hipDataType.
                    Data type of the elements of A.
    @param[inout]
    A               pointer to dataTypeA. Array on the GPU of dimension lda*n.
                    On entry, the triangular matrix A.
                    On exit, the inverse of A.
    @param[in]
    lda             int64_t. lda >= n.
                    Specifies the leading dimension of A.
    @param[inout]
    workOnDevice    pointer to void. Workspace array on the GPU of lworkOnDevice bytes.
                    If workOnDevice is null or lworkOnDevice is 0, the rocSOLVER backend
                    uses the device workspace managed by the handle instead.
    @param[in]
    lworkOnDevice   size_t.
                    Size in bytes of workOnDevice, as returned by
                    #hipsolverDnXtrtri_bufferSize.
    @param[inout]
    workOnHost      pointer to void. Workspace array on the host of lworkOnHost bytes.
                    Not referenced by the rocSOLVER backend.
    @param[in]
    lworkOnHost     size_t.
                    Size in bytes of workOnHost, as returned by
                    #hipsolverDnXtrtri_bufferSize.
    @param[out]
    devInfo         pointer to an int on the GPU.
                    If devInfo = 0, successful exit.
                    If devInfo = i > 0, A is singular. A[i,i] is the first zero diagonal element.
    ********************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXtrtri(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     hipsolverDiagType_t diag,
                                                     int64_t             n,
                                                     hipDataType         dataTypeA,
                                                     void*               A,
                                                     int64_t             lda,
                                                     void*               workOnDevice,
                                                     size_t              lworkOnDevice,
                                                     void*               workOnHost,
                                                     size_t              lworkOnHost,
                                                     int*                devInfo);


#ifdef __cplusplus
}
#endif

#endif // HIPSOLVER_DENSE64_H
