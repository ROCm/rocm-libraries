// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/utilities/Tensor.hpp>

namespace hipdnn_sdk::utilities
{

// Forward declaration
template <typename T>
class TensorSpan;

// The iterator that wraps ITensorIterator
template <typename T, bool IsConst = false>
class TensorSpanIterator
{
public:
    // Iterator traits for STL compatibility
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using reference = std::conditional_t<IsConst, const T&, T&>;
    using pointer = std::conditional_t<IsConst, const T*, T*>;

    // Constructor
    explicit TensorSpanIterator(ITensorIterator iter)
        : _iter(std::move(iter))
    {
    }

    // Dereference returns typed reference (not void*)
    reference operator*() const
    {
        void* ptr = *_iter;
        return *static_cast<pointer>(ptr);
    }

    pointer operator->() const
    {
        void* ptr = *_iter;
        return static_cast<pointer>(ptr);
    }

    // Prefix increment
    TensorSpanIterator& operator++()
    {
        ++_iter;
        return *this;
    }

    // Postfix increment
    TensorSpanIterator operator++(int)
    {
        TensorSpanIterator temp = *this;
        ++_iter;
        return temp;
    }

    // Comparison operators
    bool operator==(const TensorSpanIterator& other) const
    {
        return _iter == other._iter;
    }

    bool operator!=(const TensorSpanIterator& other) const
    {
        return _iter != other._iter;
    }

private:
    ITensorIterator _iter; // Wraps the type-erased iterator
};

template <typename T>
class TensorSpan
{
public:
    using iterator = TensorSpanIterator<T, false>;
    using const_iterator = TensorSpanIterator<T, true>;

    // Constructor takes a reference to ITensor
    explicit TensorSpan(ITensor& tensor)
        : _tensor(tensor)
    {
    }

    // Constructor for const ITensor
    explicit TensorSpan(const ITensor& tensor)
        : _tensor(const_cast<ITensor&>(tensor))
    {
    }

    // Provide typed iterator access
    iterator begin()
    {
        return iterator(_tensor.begin());
    }

    iterator end()
    {
        return iterator(_tensor.end());
    }

    const_iterator begin() const
    {
        return const_iterator(_tensor.begin());
    }

    const_iterator end() const
    {
        return const_iterator(_tensor.end());
    }

private:
    ITensor& _tensor;
};

} // namespace hipdnn_sdk::utilities
