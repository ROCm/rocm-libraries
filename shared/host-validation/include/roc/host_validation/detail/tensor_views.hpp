// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/host_validation/tensor.hpp>

#include <cassert>
#include <cstddef>
#include <stdexcept>

namespace roc::host_validation {
template <typename T>
class ConstMatrixView {
   public:
    ConstMatrixView(const T* data, size_t rows, size_t columns, ptrdiff_t rowStride,
                    ptrdiff_t columnStride)
        : m_data(data),
          m_rows(rows),
          m_columns(columns),
          m_rowStride(rowStride),
          m_columnStride(columnStride) {
        if (m_data == nullptr && m_rows != 0 && m_columns != 0)
            throw std::invalid_argument("ConstMatrixView has null data.");
    }

    const T& operator()(size_t row, size_t column) const {
        assert(row < m_rows);
        assert(column < m_columns);
        return m_data[static_cast<ptrdiff_t>(row) * m_rowStride +
                      static_cast<ptrdiff_t>(column) * m_columnStride];
    }

    size_t rows() const {
        return m_rows;
    }

    size_t columns() const {
        return m_columns;
    }

   private:
    const T* m_data = nullptr;
    size_t m_rows = 0;
    size_t m_columns = 0;
    ptrdiff_t m_rowStride = 0;
    ptrdiff_t m_columnStride = 0;
};

template <typename T>
class MatrixView {
   public:
    MatrixView(T* data, size_t rows, size_t columns, ptrdiff_t rowStride, ptrdiff_t columnStride)
        : m_data(data),
          m_rows(rows),
          m_columns(columns),
          m_rowStride(rowStride),
          m_columnStride(columnStride) {
        if (m_data == nullptr && m_rows != 0 && m_columns != 0)
            throw std::invalid_argument("MatrixView has null data.");
    }

    T& operator()(size_t row, size_t column) const {
        assert(row < m_rows);
        assert(column < m_columns);
        return m_data[static_cast<ptrdiff_t>(row) * m_rowStride +
                      static_cast<ptrdiff_t>(column) * m_columnStride];
    }

    size_t rows() const {
        return m_rows;
    }

    size_t columns() const {
        return m_columns;
    }

    ConstMatrixView<T> asConst() const {
        return ConstMatrixView<T>(
            m_data, m_rows, m_columns, m_rowStride, m_columnStride);
    }

   private:
    T* m_data = nullptr;
    size_t m_rows = 0;
    size_t m_columns = 0;
    ptrdiff_t m_rowStride = 0;
    ptrdiff_t m_columnStride = 0;
};

template <typename T>
class ConstVectorView {
   public:
    ConstVectorView(const T* data, size_t size, ptrdiff_t stride = 1)
        : m_data(data), m_size(size), m_stride(stride) {
        if (m_data == nullptr && m_size != 0)
            throw std::invalid_argument("ConstVectorView has null data.");
    }

    const T& operator[](size_t index) const {
        assert(index < m_size);
        return m_data[static_cast<ptrdiff_t>(index) * m_stride];
    }

    size_t size() const {
        return m_size;
    }

   private:
    const T* m_data = nullptr;
    size_t m_size = 0;
    ptrdiff_t m_stride = 1;
};
}  // namespace roc::host_validation
