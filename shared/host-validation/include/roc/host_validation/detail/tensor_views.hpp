// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <roc/host_validation/tensor.hpp>
#include <utility>
#include <vector>

namespace roc::host_validation {
template <typename T>
class ConstMatrixView {
   public:
    ConstMatrixView(const T* data, size_t rows, size_t columns, ptrdiff_t rowStride,
                    ptrdiff_t columnStride)
        : m_view(data,
                 Layout(Shape{rows, columns}, std::vector<ptrdiff_t>{rowStride, columnStride})) {}

    const T& operator()(size_t row, size_t column) const {
        return m_view.at({row, column});
    }

    size_t rows() const {
        return m_view.shape()[0];
    }

    size_t columns() const {
        return m_view.shape()[1];
    }

    const TensorView<T>& tensor() const {
        return m_view;
    }

   private:
    TensorView<T> m_view;
};

template <typename T>
class MatrixView {
   public:
    MatrixView(T* data, size_t rows, size_t columns, ptrdiff_t rowStride, ptrdiff_t columnStride)
        : m_view(data,
                 Layout(Shape{rows, columns}, std::vector<ptrdiff_t>{rowStride, columnStride})) {}

    T& operator()(size_t row, size_t column) const {
        return m_view.at({row, column});
    }

    size_t rows() const {
        return m_view.shape()[0];
    }

    size_t columns() const {
        return m_view.shape()[1];
    }

    ConstMatrixView<T> asConst() const {
        const auto strides = m_view.layout().strides();
        return ConstMatrixView<T>(m_view.data(), rows(), columns(), strides[0], strides[1]);
    }

    const MutableTensorView<T>& tensor() const {
        return m_view;
    }

   private:
    MutableTensorView<T> m_view;
};

template <typename T>
class ConstVectorView {
   public:
    ConstVectorView(const T* data, size_t size, ptrdiff_t stride = 1)
        : m_view(data, Layout(Shape{size}, std::vector<ptrdiff_t>{stride})) {}

    const T& operator[](size_t index) const {
        return m_view.at({index});
    }

    size_t size() const {
        return m_view.shape()[0];
    }

    const TensorView<T>& tensor() const {
        return m_view;
    }

   private:
    TensorView<T> m_view;
};
}  // namespace roc::host_validation
