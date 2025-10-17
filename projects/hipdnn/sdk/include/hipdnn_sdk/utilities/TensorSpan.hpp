// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/utilities/Tensor.hpp>

namespace hipdnn_sdk::utilities
{

template <typename T, bool IsConst = false>
class TensorSpanIterator
{
public:
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

    reference operator*() const
    {
        return *static_cast<pointer>(*_iter);
    }

    pointer operator->() const
    {
        void* ptr = *_iter;
        return static_cast<pointer>(ptr);
    }

    TensorSpanIterator& operator++()
    {
        ++_iter;
        return *this;
    }

    TensorSpanIterator operator++(int)
    {
        TensorSpanIterator temp = *this;
        ++_iter;
        return temp;
    }

    bool operator==(const TensorSpanIterator& other) const
    {
        return _iter == other._iter;
    }

    bool operator!=(const TensorSpanIterator& other) const
    {
        return _iter != other._iter;
    }

private:
    ITensorIterator<IsConst> _iter;
};

template <typename T, bool IsConst = false>
class TensorSpan
{
public:
    using iterator = TensorSpanIterator<T, IsConst>;
    using const_iterator = TensorSpanIterator<T, true>;
    using tensor_reference = std::conditional_t<IsConst, const ITensor&, ITensor&>;

    explicit TensorSpan(tensor_reference tensor)
        : _tensor(tensor)
    {
    }

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

}
