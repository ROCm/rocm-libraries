// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/utilities/Tensor.hpp>

namespace hipdnn_sdk::utilities
{

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
    explicit TensorSpanIterator(ITensorIterator<IsConst> iter)
        : _iter(std::move(iter))
    {
    }

    // Dereference returns typed reference (not void*)
    reference operator*() const
    {
        return *static_cast<pointer>(*_iter);
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
    ITensorIterator<IsConst> _iter; // Wraps the type-erased iterator
};

template <typename T, bool IsConst = false>
class TensorSpan
{
public:
    using iterator = TensorSpanIterator<T, IsConst>;
    using const_iterator = TensorSpanIterator<T, true>;
    using tensor_reference = std::conditional_t<IsConst, const ITensor&, ITensor&>;

    // Constructor takes a reference to ITensor
    explicit TensorSpan(tensor_reference tensor)
        : _tensor(tensor)
    {
    }

    // Provide typed iterator access
    iterator begin()
    {
        if constexpr(IsConst)
        {
            return iterator(_tensor.cbegin());
        }
        else
        {
            return iterator(_tensor.begin());
        }
    }

    iterator end()
    {
        if constexpr(IsConst)
        {
            return iterator(_tensor.cend());
        }
        else
        {
            return iterator(_tensor.end());
        }
    }

    const_iterator cbegin() const
    {
        return const_iterator(_tensor.cbegin());
    }

    const_iterator cend() const
    {
        return const_iterator(_tensor.cend());
    }

private:
    tensor_reference _tensor;
};

template <typename T>
using ConstTensorSpan = TensorSpan<T, true>;

} // namespace hipdnn_sdk::utilities
