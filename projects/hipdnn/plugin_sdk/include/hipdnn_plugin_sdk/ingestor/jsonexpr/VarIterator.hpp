// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// VarIterator.hpp - a lazy pre-order range over a compiled tree's variables.
//
// References point into the live VarNode::path, so nothing is copied and the
// range must not outlive the Expression.

#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Node.hpp>

#include <cstddef>
#include <iterator>
#include <string>
#include <vector>

namespace hipdnn_plugin_sdk::ingestor::jsonexpr::detail
{
// ---- variable iteration ---------------------------------------------------
// Lazily yields, in pre-order, a reference to every variable path in a
// compiled node tree. The references point at the live VarNode::path, so no
// strings are copied. Duplicates are yielded as they occur; build a std::set
// from the range if you need the unique, sorted set.
class VarIterator
{
public:
    using value_type = std::string;
    using reference = const std::string&;
    using pointer = const std::string*;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::input_iterator_tag;

    VarIterator() = default; // end
    explicit VarIterator(const Node* root)
    {
        if(root != nullptr)
        {
            _stack.push_back(root);
        }
        advance();
    }

    reference operator*() const
    {
        return *_cur;
    }
    pointer operator->() const
    {
        return _cur;
    }

    VarIterator& operator++()
    {
        advance();
        return *this;
    }
    VarIterator operator++(int)
    {
        VarIterator tmp = *this;
        advance();
        return tmp;
    }

    /// Two iterators are equal when they sit on the same variable. An iterator
    /// therefore equals itself, and two end iterators (_cur == nullptr) are
    /// equal. Comparing only against end would break any algorithm that
    /// compares two positions.
    bool operator==(const VarIterator& o) const
    {
        return _cur == o._cur;
    }
    bool operator!=(const VarIterator& o) const
    {
        return !(*this == o);
    }

private:
    void advance()
    {
        while(!_stack.empty())
        {
            const Node* n = _stack.back();
            _stack.pop_back();
            n->pushChildren(_stack);
            if(const std::string* p = n->variable())
            {
                _cur = p;
                return;
            }
        }
        _cur = nullptr;
    }

    std::vector<const Node*> _stack;
    const std::string* _cur = nullptr;
};

class VarRange
{
public:
    explicit VarRange(const Node* root)
        : _begin(root)
    {
    }
    /// Returned by value, not by reference. Expression::variables() returns a
    /// VarRange by value, so in `expr.variables().begin()` the range is a
    /// temporary that dies at the end of the statement; a reference into its
    /// members would dangle.
    VarIterator begin() const
    {
        return _begin;
    }
    VarIterator end() const
    {
        return _end;
    }

private:
    VarIterator _begin;
    VarIterator _end;
};
} // namespace hipdnn_plugin_sdk::ingestor::jsonexpr::detail

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
