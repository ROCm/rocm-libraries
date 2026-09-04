// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <rocblas/rocblas.h>

#include <algorithm>
#include <new>

struct _rocblas_handle {
    hipStream_t stream = nullptr;
    rocblas_pointer_mode pointer_mode = rocblas_pointer_mode_host;
};

extern "C" rocblas_status rocblas_create_handle(rocblas_handle* result) {
    if (!result) return rocblas_status_invalid_pointer;
    *result = new (std::nothrow) _rocblas_handle;
    return *result ? rocblas_status_success : rocblas_status_memory_error;
}

extern "C" rocblas_status rocblas_destroy_handle(rocblas_handle handle) {
    if (!handle) return rocblas_status_invalid_handle;
    delete handle;
    return rocblas_status_success;
}

extern "C" rocblas_status rocblas_set_stream(rocblas_handle handle, hipStream_t stream) {
    if (!handle) return rocblas_status_invalid_handle;
    handle->stream = stream;
    return rocblas_status_success;
}

extern "C" rocblas_status rocblas_get_stream(rocblas_handle handle, hipStream_t* stream) {
    if (!handle) return rocblas_status_invalid_handle;
    if (!stream) return rocblas_status_invalid_pointer;
    *stream = handle->stream;
    return rocblas_status_success;
}

extern "C" rocblas_status rocblas_set_pointer_mode(rocblas_handle handle,
                                                   rocblas_pointer_mode mode) {
    if (!handle) return rocblas_status_invalid_handle;
    handle->pointer_mode = mode;
    return rocblas_status_success;
}

extern "C" rocblas_status rocblas_get_pointer_mode(rocblas_handle handle,
                                                   rocblas_pointer_mode* mode) {
    if (!handle) return rocblas_status_invalid_handle;
    if (!mode) return rocblas_status_invalid_pointer;
    *mode = handle->pointer_mode;
    return rocblas_status_success;
}

template <typename Index>
rocblas_status axpy(rocblas_handle handle, Index n, const float* alpha, const float* x, Index incx,
                    float* y, Index incy) {
    if (!handle) return rocblas_status_invalid_handle;
    if (!alpha || !x || !y) return rocblas_status_invalid_pointer;
    if (n < 0) return rocblas_status_invalid_size;
    for (Index i = 0; i < n; ++i) y[i * incy] += *alpha * x[i * incx];
    return rocblas_status_success;
}

template <typename Index>
rocblas_status copy(rocblas_handle handle, Index n, const float* x, Index incx, float* y,
                    Index incy) {
    if (!handle) return rocblas_status_invalid_handle;
    if (!x || !y) return rocblas_status_invalid_pointer;
    if (n < 0) return rocblas_status_invalid_size;
    for (Index i = 0; i < n; ++i) y[i * incy] = x[i * incx];
    return rocblas_status_success;
}

template <typename Index>
rocblas_status scal(rocblas_handle handle, Index n, const float* alpha, float* x, Index incx) {
    if (!handle) return rocblas_status_invalid_handle;
    if (!alpha || !x) return rocblas_status_invalid_pointer;
    if (n < 0) return rocblas_status_invalid_size;
    for (Index i = 0; i < n; ++i) x[i * incx] *= *alpha;
    return rocblas_status_success;
}

template <typename Index>
rocblas_status swap(rocblas_handle handle, Index n, float* x, Index incx, float* y, Index incy) {
    if (!handle) return rocblas_status_invalid_handle;
    if (!x || !y) return rocblas_status_invalid_pointer;
    if (n < 0) return rocblas_status_invalid_size;
    for (Index i = 0; i < n; ++i) std::swap(x[i * incx], y[i * incy]);
    return rocblas_status_success;
}

extern "C" rocblas_status rocblas_saxpy(rocblas_handle h, rocblas_int n, const float* a,
                                        const float* x, rocblas_int ix, float* y, rocblas_int iy) {
    return axpy(h, n, a, x, ix, y, iy);
}
extern "C" rocblas_status rocblas_saxpy_64(rocblas_handle h, int64_t n, const float* a,
                                           const float* x, int64_t ix, float* y, int64_t iy) {
    return axpy(h, n, a, x, ix, y, iy);
}
extern "C" rocblas_status rocblas_scopy(rocblas_handle h, rocblas_int n, const float* x,
                                        rocblas_int ix, float* y, rocblas_int iy) {
    return copy(h, n, x, ix, y, iy);
}
extern "C" rocblas_status rocblas_scopy_64(rocblas_handle h, int64_t n, const float* x, int64_t ix,
                                           float* y, int64_t iy) {
    return copy(h, n, x, ix, y, iy);
}
extern "C" rocblas_status rocblas_sscal(rocblas_handle h, rocblas_int n, const float* a, float* x,
                                        rocblas_int ix) {
    return scal(h, n, a, x, ix);
}
extern "C" rocblas_status rocblas_sscal_64(rocblas_handle h, int64_t n, const float* a, float* x,
                                           int64_t ix) {
    return scal(h, n, a, x, ix);
}
extern "C" rocblas_status rocblas_sswap(rocblas_handle h, rocblas_int n, float* x, rocblas_int ix,
                                        float* y, rocblas_int iy) {
    return swap(h, n, x, ix, y, iy);
}
extern "C" rocblas_status rocblas_sswap_64(rocblas_handle h, int64_t n, float* x, int64_t ix,
                                           float* y, int64_t iy) {
    return swap(h, n, x, ix, y, iy);
}
