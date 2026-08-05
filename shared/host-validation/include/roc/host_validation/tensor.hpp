// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace roc::host_validation {
class Shape {
   public:
    Shape() = default;

    explicit Shape(std::vector<size_t> dimensions) : m_dimensions(std::move(dimensions)) {}

    Shape(std::initializer_list<size_t> dimensions) : m_dimensions(dimensions) {}

    explicit Shape(std::span<const size_t> dimensions)
        : m_dimensions(dimensions.begin(), dimensions.end()) {}

    size_t rank() const {
        return m_dimensions.size();
    }

    bool empty() const {
        return elementCount() == 0;
    }

    size_t operator[](size_t dimension) const {
        return m_dimensions.at(dimension);
    }

    std::span<const size_t> dimensions() const {
        return m_dimensions;
    }

    size_t elementCount() const {
        size_t count = 1;
        for (const size_t dimension : m_dimensions) {
            if (dimension == 0) return 0;
            if (count > std::numeric_limits<size_t>::max() / dimension)
                throw std::overflow_error("Tensor shape element count overflow.");
            count *= dimension;
        }
        return count;
    }

    friend bool operator==(const Shape&, const Shape&) = default;

   private:
    std::vector<size_t> m_dimensions;
};

class Layout {
   public:
    Layout() = default;

    Layout(Shape shape, std::vector<ptrdiff_t> strides, ptrdiff_t offset = 0)
        : m_shape(std::move(shape)), m_strides(std::move(strides)), m_offset(offset) {
        if (m_shape.rank() != m_strides.size())
            throw std::invalid_argument("Tensor layout rank and stride count differ.");
    }

    static Layout contiguous(const Shape& shape) {
        std::vector<ptrdiff_t> strides(shape.rank(), 1);
        ptrdiff_t stride = 1;
        for (size_t dimension = shape.rank(); dimension > 0; --dimension) {
            const size_t index = dimension - 1;
            strides[index] = stride;
            const size_t extent = shape[index];
            if (extent > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
                throw std::overflow_error("Tensor extent exceeds ptrdiff_t.");
            const ptrdiff_t signedExtent = static_cast<ptrdiff_t>(extent);
            if (signedExtent != 0 && stride > std::numeric_limits<ptrdiff_t>::max() / signedExtent)
                throw std::overflow_error("Tensor contiguous stride overflow.");
            stride *= signedExtent;
        }
        return Layout(shape, std::move(strides));
    }

    const Shape& shape() const {
        return m_shape;
    }

    std::span<const ptrdiff_t> strides() const {
        return m_strides;
    }

    ptrdiff_t offset() const {
        return m_offset;
    }

    ptrdiff_t elementOffset(std::span<const size_t> indices) const {
        if (indices.size() != m_shape.rank())
            throw std::invalid_argument("Tensor index rank does not match layout rank.");

        ptrdiff_t result = m_offset;
        for (size_t dimension = 0; dimension < indices.size(); ++dimension) {
            if (indices[dimension] >= m_shape[dimension])
                throw std::out_of_range("Tensor index exceeds shape.");
            result += static_cast<ptrdiff_t>(indices[dimension]) * m_strides[dimension];
        }
        return result;
    }

    friend bool operator==(const Layout&, const Layout&) = default;

   private:
    Shape m_shape;
    std::vector<ptrdiff_t> m_strides;
    ptrdiff_t m_offset = 0;
};

template <typename T>
class TensorView {
   public:
    TensorView(const T* data, Layout layout) : m_data(data), m_layout(std::move(layout)) {
        if (m_data == nullptr && m_layout.shape().elementCount() != 0)
            throw std::invalid_argument("TensorView has null data for a non-empty tensor.");
    }

    const T& at(std::span<const size_t> indices) const {
        return m_data[m_layout.elementOffset(indices)];
    }

    const T& at(std::initializer_list<size_t> indices) const {
        return at(std::span<const size_t>(indices.begin(), indices.size()));
    }

    const Shape& shape() const {
        return m_layout.shape();
    }

    const Layout& layout() const {
        return m_layout;
    }

    const T* data() const {
        return m_data;
    }

   private:
    const T* m_data = nullptr;
    Layout m_layout;
};

template <typename T>
class MutableTensorView {
   public:
    MutableTensorView(T* data, Layout layout) : m_data(data), m_layout(std::move(layout)) {
        if (m_data == nullptr && m_layout.shape().elementCount() != 0)
            throw std::invalid_argument("MutableTensorView has null data for a non-empty tensor.");
    }

    T& at(std::span<const size_t> indices) const {
        return m_data[m_layout.elementOffset(indices)];
    }

    T& at(std::initializer_list<size_t> indices) const {
        return at(std::span<const size_t>(indices.begin(), indices.size()));
    }

    const Shape& shape() const {
        return m_layout.shape();
    }

    const Layout& layout() const {
        return m_layout;
    }

    T* data() const {
        return m_data;
    }

    TensorView<T> asConst() const {
        return TensorView<T>(m_data, m_layout);
    }

   private:
    T* m_data = nullptr;
    Layout m_layout;
};

template <typename T>
class Tensor {
   public:
    static_assert(!std::is_same_v<T, bool>, "Tensor<bool> is not supported.");

    explicit Tensor(Shape shape)
        : m_layout(Layout::contiguous(shape)), m_values(shape.elementCount()) {}

    Tensor(Shape shape, const T& value)
        : m_layout(Layout::contiguous(shape)), m_values(shape.elementCount(), value) {}

    Tensor(Shape shape, std::vector<T> values)
        : m_layout(Layout::contiguous(shape)), m_values(std::move(values)) {
        if (m_values.size() != m_layout.shape().elementCount())
            throw std::invalid_argument("Tensor value count does not match shape.");
    }

    const Shape& shape() const {
        return m_layout.shape();
    }

    const Layout& layout() const {
        return m_layout;
    }

    size_t size() const {
        return m_values.size();
    }

    std::span<const T> values() const {
        return m_values;
    }

    std::span<T> values() {
        return m_values;
    }

    TensorView<T> view() const {
        return TensorView<T>(m_values.data(), m_layout);
    }

    MutableTensorView<T> mutableView() {
        return MutableTensorView<T>(m_values.data(), m_layout);
    }

   private:
    Layout m_layout;
    std::vector<T> m_values;
};
}  // namespace roc::host_validation
