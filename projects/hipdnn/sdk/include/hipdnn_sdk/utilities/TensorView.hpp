// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/utilities/Tensor.hpp>

namespace hipdnn_sdk::utilities
{

template <typename T, bool IsConst = false>
class TensorViewIterator
{
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using reference = std::conditional_t<IsConst, const T&, T&>;
    using pointer = std::conditional_t<IsConst, const T*, T*>;

    // Constructor
    explicit TensorViewIterator(ITensorIterator<IsConst> iter)
        : _iter(std::move(iter))
    {
    }

    reference operator*()
    {
        return *static_cast<pointer>(*_iter);
    }

    pointer operator->()
    {
        void* ptr = *_iter;
        return static_cast<pointer>(ptr);
    }

    TensorViewIterator& operator++()
    {
        ++_iter;
        return *this;
    }

    TensorViewIterator operator++(int)
    {
        TensorViewIterator temp = *this;
        ++_iter;
        return temp;
    }

    bool operator==(const TensorViewIterator& other) const
    {
        return _iter == other._iter;
    }

    bool operator!=(const TensorViewIterator& other) const
    {
        return _iter != other._iter;
    }

private:
    ITensorIterator<IsConst> _iter;
};

template <typename T, bool IsConst = false>
class TensorView
{
public:
    using iterator = TensorViewIterator<T, IsConst>;
    using const_iterator = TensorViewIterator<T, true>;
    using tensor_reference = std::conditional_t<IsConst, const ITensor&, ITensor&>;
    using value_reference = std::conditional_t<IsConst, const T&, T&>;

    explicit TensorView(tensor_reference tensor)
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

    value_reference getHostValue(const std::vector<int64_t>& indices)
    {
        return *static_cast<T*>(_tensor.hostDataOffsetFromIndex(_tensor.getIndex(indices)));
    }

    const T& getHostValue(const std::vector<int64_t>& indices) const
    {
        return *static_cast<const T*>(_tensor.hostDataOffsetFromIndex(_tensor.getIndex(indices)));
    }

private:
    tensor_reference _tensor;
};

template <typename T>
using ConstTensorView = TensorView<T, true>;

}
