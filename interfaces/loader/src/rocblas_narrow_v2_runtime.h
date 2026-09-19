// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#ifndef ROCM_INTERFACES_ROCBLAS_NARROW_V2_RUNTIME_H_
#define ROCM_INTERFACES_ROCBLAS_NARROW_V2_RUNTIME_H_

#include <type_traits>

#include "rocm/interfaces/experimental/blas_narrow_v2.h"

namespace rocm::interfaces {

rocm_interfaces_abi_header narrow_v2_header(size_t size) noexcept;
rocm_blas_v2_execution narrow_v2_execution(rocblas_handle, rocm_blas_v2_index_width,
                                           rocm_blas_v2_batch_kind, int64_t) noexcept;
rocblas_pointer_mode narrow_v2_pointer_mode(rocblas_handle) noexcept;

template <class Pointer>
rocblas_datatype narrow_v2_pointer_type(Pointer) noexcept {
    using Element0 = std::remove_pointer_t<Pointer>;
    using Element1 =
        std::conditional_t<std::is_pointer_v<std::remove_cv_t<Element0>>,
                           std::remove_pointer_t<std::remove_cv_t<Element0>>, Element0>;
    using T = std::remove_cv_t<Element1>;
    if constexpr (std::is_same_v<T, float>) return rocblas_datatype_f32_r;
    if constexpr (std::is_same_v<T, double>) return rocblas_datatype_f64_r;
    if constexpr (std::is_same_v<T, rocblas_half>) return rocblas_datatype_f16_r;
    if constexpr (std::is_same_v<T, rocblas_bfloat16>) return rocblas_datatype_bf16_r;
    if constexpr (std::is_same_v<T, rocblas_float_complex>) return rocblas_datatype_f32_c;
    if constexpr (std::is_same_v<T, rocblas_double_complex>) return rocblas_datatype_f64_c;
    if constexpr (std::is_same_v<T, rocblas_int>) return rocblas_datatype_i32_r;
    return rocblas_datatype_f32_r;  // Explicit *_type overrides this for void storage.
}

template <class Pointer>
rocm_blas_v2_memory narrow_v2_memory(Pointer pointer, int64_t stride = 0) noexcept {
    rocm_blas_v2_memory result{};
    result.header = narrow_v2_header(sizeof(result));
    using Pointee = std::remove_cv_t<std::remove_pointer_t<Pointer>>;
    if constexpr (std::is_same_v<Pointer, std::nullptr_t>) {
        result.base = nullptr;
    } else if constexpr (std::is_pointer_v<Pointee>) {
        result.pointer_array = reinterpret_cast<const void* const*>(pointer);
    } else {
        result.base = const_cast<void*>(reinterpret_cast<const void*>(pointer));
    }
    result.batch_stride = stride;
    return result;
}

template <class Pointer>
rocm_blas_v2_scalar narrow_v2_scalar(rocblas_handle handle, Pointer pointer,
                                     rocblas_datatype type) noexcept {
    return {narrow_v2_header(sizeof(rocm_blas_v2_scalar)), type, narrow_v2_pointer_mode(handle),
            pointer};
}

template <class Pointer>
rocm_blas_v2_vector narrow_v2_vector(Pointer pointer, rocblas_datatype type, int64_t length,
                                     int64_t increment, int64_t stride = 0) noexcept {
    return {narrow_v2_header(sizeof(rocm_blas_v2_vector)), narrow_v2_memory(pointer, stride), type,
            length, increment};
}

template <class Pointer>
rocm_blas_v2_matrix narrow_v2_matrix(Pointer pointer, rocblas_datatype type, int64_t rows,
                                     int64_t columns, int64_t ld, int64_t stride = 0) noexcept {
    rocm_blas_v2_matrix result{};
    result.header = narrow_v2_header(sizeof(result));
    result.memory = narrow_v2_memory(pointer, stride);
    result.data_type = type;
    result.storage = ROCM_BLAS_V2_STORAGE_DENSE;
    result.kind = ROCM_BLAS_V2_MATRIX_GENERAL;
    result.fill = rocblas_fill_full;
    result.diagonal = rocblas_diagonal_non_unit;
    result.rows = rows;
    result.columns = columns;
    result.leading_dimension = ld;
    return result;
}

rocblas_status narrow_v2_dispatch(rocblas_handle,
                                  const rocm_blas_v2_vector_transform_request*) noexcept;
rocblas_status narrow_v2_dispatch(rocblas_handle,
                                  const rocm_blas_v2_vector_reduce_request*) noexcept;
rocblas_status narrow_v2_dispatch(rocblas_handle, const rocm_blas_v2_rotation_request*) noexcept;
rocblas_status narrow_v2_dispatch(rocblas_handle,
                                  const rocm_blas_v2_matrix_vector_request*) noexcept;
rocblas_status narrow_v2_dispatch(rocblas_handle, const rocm_blas_v2_rank_update_request*) noexcept;
rocblas_status narrow_v2_dispatch(rocblas_handle, const rocm_blas_v2_matmul_request*) noexcept;
rocblas_status narrow_v2_dispatch(rocblas_handle,
                                  const rocm_blas_v2_structured_matrix_request*) noexcept;
rocblas_status narrow_v2_dispatch(rocblas_handle,
                                  const rocm_blas_v2_triangular_matrix_request*) noexcept;
rocblas_status narrow_v2_dispatch(rocblas_handle,
                                  const rocm_blas_v2_matrix_transform_request*) noexcept;

}  // namespace rocm::interfaces
#endif
