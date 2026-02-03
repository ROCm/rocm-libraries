/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include <cstddef>
#include <iterator>

// This is a lightweight intrusive list implementation inspired by LLVM's design.
// Key advantages:
// 1. No memory allocation overhead--nodes manage their own linkage.
// 2. O(1) insertion and removal via direct pointer manipulation.
// 3. Efficient move operations (moveBefore/moveAfter).
// 4. Familiar LLVM-style API for compiler and systems developers.
//
namespace stinkytofu
{
    // Forward declarations
    template <typename T>
    class IntrusiveList;

    template <typename T>
    class IntrusiveListIterator;

    // Base class for nodes that can be inserted into an intrusive list
    template <typename T>
    class IntrusiveListNode
    {
        friend class IntrusiveList<T>;
        friend class IntrusiveListIterator<T>;
        friend class IntrusiveListIterator<const T>;

    private:
        T*                prev_;
        T*                next_;
        IntrusiveList<T>* parent_list_;

    public:
        IntrusiveListNode()
            : prev_(nullptr)
            , next_(nullptr)
            , parent_list_(nullptr)
        {
        }

        // Disable copy constructor and assignment
        IntrusiveListNode(const IntrusiveListNode&)            = delete;
        IntrusiveListNode& operator=(const IntrusiveListNode&) = delete;

        ~IntrusiveListNode() = default;

        // Check if this node is currently in a list
        bool isInList() const
        {
            return parent_list_ != nullptr;
        }

        // Get the next/previous nodes (returns nullptr if at end/beginning)
        T* getNext() const
        {
            return next_;
        }
        T* getPrev() const
        {
            return prev_;
        }

        // Remove this node from whatever list it's currently in
        void removeFromList()
        {
            if(!parent_list_)
                return;

            if(prev_)
                prev_->next_ = next_;
            else
                parent_list_->head_ = next_;

            if(next_)
                next_->prev_ = prev_;
            else
                parent_list_->tail_ = prev_;

            --parent_list_->size_;
            prev_ = next_ = nullptr;
            parent_list_  = nullptr;
        }
    };

    // Iterator for the intrusive list
    template <typename T>
    class IntrusiveListIterator
    {
    private:
        T* node_;

    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using pointer           = T*;
        using reference         = T&;

        explicit IntrusiveListIterator(T* node = nullptr)
            : node_(node)
        {
        }

        // Conversion constructor from non-const to const iterator
        template <typename U = T>
        IntrusiveListIterator(
            const IntrusiveListIterator<U>& other,
            typename std::enable_if<std::is_same<U, typename std::remove_const<T>::type>::value
                                    && std::is_const<T>::value>::type* = nullptr)
            : node_(other.getNodePtr())
        {
        }

        T& operator*() const
        {
            return *node_;
        }
        T* operator->() const
        {
            return node_;
        }

        IntrusiveListIterator& operator++()
        {
            if(node_)
                node_ = node_->next_;
            return *this;
        }

        IntrusiveListIterator operator++(int)
        {
            IntrusiveListIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        IntrusiveListIterator& operator--()
        {
            if(node_)
                node_ = node_->prev_;
            return *this;
        }

        IntrusiveListIterator operator--(int)
        {
            IntrusiveListIterator tmp = *this;
            --(*this);
            return tmp;
        }

        IntrusiveListIterator operator+(size_t n) const
        {
            IntrusiveListIterator tmp = *this;
            for(size_t i = 0; i < n; ++i)
                ++tmp;
            return tmp;
        }

        bool operator==(const IntrusiveListIterator& other) const
        {
            return node_ == other.node_;
        }

        bool operator!=(const IntrusiveListIterator& other) const
        {
            return node_ != other.node_;
        }

        // Get the underlying node pointer
        T* getNodePtr() const
        {
            return node_;
        }
    };

    // The intrusive list container
    template <typename T>
    class IntrusiveList
    {
        friend class IntrusiveListNode<T>;

    private:
        T*     head_;
        T*     tail_;
        size_t size_;

    public:
        using iterator       = IntrusiveListIterator<T>;
        using const_iterator = IntrusiveListIterator<const T>;

        IntrusiveList()
            : head_(nullptr)
            , tail_(nullptr)
            , size_(0)
        {
        }

        // Disable copy constructor and assignment for now
        IntrusiveList(const IntrusiveList&)            = delete;
        IntrusiveList& operator=(const IntrusiveList&) = delete;

        ~IntrusiveList()
        {
            clear();
        }

        // Iterator support
        iterator begin()
        {
            return iterator(head_);
        }
        iterator end()
        {
            return iterator(nullptr);
        }
        const_iterator begin() const
        {
            return const_iterator(head_);
        }
        const_iterator end() const
        {
            return const_iterator(nullptr);
        }

        // Reverse iterator support
        class reverse_iterator
        {
        private:
            T*                node_;
            IntrusiveList<T>* list_;

        public:
            reverse_iterator(T* node, IntrusiveList<T>* list)
                : node_(node)
                , list_(list)
            {
            }

            T& operator*() const
            {
                return *node_;
            }

            T* operator->() const
            {
                return node_;
            }

            reverse_iterator& operator++()
            {
                if(node_)
                    node_ = node_->prev_;
                return *this;
            }

            reverse_iterator operator++(int)
            {
                reverse_iterator tmp = *this;
                ++(*this);
                return tmp;
            }

            reverse_iterator& operator--()
            {
                if(node_)
                    node_ = node_->next_;
                else if(list_)
                    node_ = list_->tail_; // From rend() to last element
                return *this;
            }

            reverse_iterator operator--(int)
            {
                reverse_iterator tmp = *this;
                --(*this);
                return tmp;
            }

            bool operator==(const reverse_iterator& other) const
            {
                return node_ == other.node_;
            }

            bool operator!=(const reverse_iterator& other) const
            {
                return node_ != other.node_;
            }

            // Get the underlying node pointer
            T* getNodePtr() const
            {
                return node_;
            }
        };

        using const_reverse_iterator = reverse_iterator;

        reverse_iterator rbegin()
        {
            return reverse_iterator(tail_, this);
        }

        reverse_iterator rend()
        {
            return reverse_iterator(nullptr, this);
        }

        const_reverse_iterator rbegin() const
        {
            return const_reverse_iterator(tail_, const_cast<IntrusiveList<T>*>(this));
        }

        const_reverse_iterator rend() const
        {
            return const_reverse_iterator(nullptr, const_cast<IntrusiveList<T>*>(this));
        }

        // Size and empty check
        size_t size() const
        {
            return size_;
        }
        bool empty() const
        {
            return size_ == 0;
        }

        // Access front and back
        T& front()
        {
            return *head_;
        }
        T& back()
        {
            return *tail_;
        }
        const T& front() const
        {
            return *head_;
        }
        const T& back() const
        {
            return *tail_;
        }

        // Add to front or back
        void push_front(T* node)
        {
            if(!node)
                return;
            node->removeFromList();

            node->next_        = head_;
            node->prev_        = nullptr;
            node->parent_list_ = this;

            if(head_)
                head_->prev_ = node;
            else
                tail_ = node;

            head_ = node;
            ++size_;
        }

        void push_back(T* node)
        {
            if(!node)
                return;
            node->removeFromList();

            node->prev_        = tail_;
            node->next_        = nullptr;
            node->parent_list_ = this;

            if(tail_)
                tail_->next_ = node;
            else
                head_ = node;

            tail_ = node;
            ++size_;
        }

        // Insert before iterator position
        iterator insert(iterator pos, T* node)
        {
            if(!node)
                return pos;

            if(pos == end())
            {
                push_back(node);
                return iterator(node);
            }

            T* insertPos = pos.getNodePtr();
            node->removeFromList();

            node->next_        = insertPos;
            node->prev_        = insertPos->prev_;
            node->parent_list_ = this;

            if(insertPos->prev_)
                insertPos->prev_->next_ = node;
            else
                head_ = node;

            insertPos->prev_ = node;
            ++size_;

            return iterator(node);
        }

        // Remove node from list
        iterator erase(iterator pos)
        {
            if(pos == end())
                return end();

            T* node = pos.getNodePtr();
            T* next = node->next_;
            node->removeFromList();

            return iterator(next);
        }

        // Remove specific node (if it's in this list)
        void remove(T* node)
        {
            if(!node || node->parent_list_ != this)
                return;
            node->removeFromList();
        }

        // Clear all nodes
        void clear()
        {
            while(!empty())
            {
                head_->removeFromList();
            }
        }

        // Move operations
        void moveBefore(iterator what, iterator where)
        {
            if(what == where || what == end())
                return;

            T* node = what.getNodePtr();
            if(node->parent_list_ != this)
                return;

            // Remove from current position (but keep in same list)
            if(node->prev_)
                node->prev_->next_ = node->next_;
            else
                head_ = node->next_;

            if(node->next_)
                node->next_->prev_ = node->prev_;
            else
                tail_ = node->prev_;

            // Insert before where
            if(where == end())
            {
                // Insert at end
                node->prev_ = tail_;
                node->next_ = nullptr;
                if(tail_)
                    tail_->next_ = node;
                else
                    head_ = node;
                tail_ = node;
            }
            else
            {
                T* insertPos = where.getNodePtr();
                node->next_  = insertPos;
                node->prev_  = insertPos->prev_;

                if(insertPos->prev_)
                    insertPos->prev_->next_ = node;
                else
                    head_ = node;

                insertPos->prev_ = node;
            }
        }

        void moveAfter(iterator what, iterator where)
        {
            if(what == where || what == end() || where == end())
                return;

            iterator nextPos = where;
            ++nextPos;
            moveBefore(what, nextPos);
        }

        // Move to front or back
        void moveToFront(iterator what)
        {
            moveBefore(what, begin());
        }

        void moveToBack(iterator what)
        {
            moveBefore(what, end());
        }

        bool isInList(const T* node) const
        {
            return node && node->parent_list_ == this;
        }
    };
}
