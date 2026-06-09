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
#include <type_traits>

// This is a lightweight intrusive list implementation inspired by LLVM's design.
// Key advantages:
// 1. No memory allocation overhead--nodes manage their own linkage.
// 2. O(1) insertion and removal via direct pointer manipulation.
// 3. Efficient move operations (moveBefore/moveAfter).
// 4. Familiar LLVM-style API for compiler and systems developers.
//
// When Parent is not void, the list stores a pointer to its parent container and
// nodes can query getParent() to obtain it (e.g. IRBase in a BasicBlock's IR list).
namespace stinkytofu {
// Forward declarations (Traits default defined after IntrusiveListTraits)
template <typename T, typename Parent>
class IntrusiveListBase;
template <typename T, typename Parent, typename Traits>
class IntrusiveList;

template <typename T>
class IntrusiveListIterator;

//----------------------------------------------------------------------
// Intrusive list traits (LLVM-style customization points)
//----------------------------------------------------------------------

/// Use delete by default for ownership semantics.
///
/// Specialize this to get different behaviour for ownership-related API.
/// \see IntrusiveListNoAllocTraits
template <typename NodeTy>
struct IntrusiveListAllocTraits {
    static void deleteNode(NodeTy* V) {
        delete V;
    }
};

/// Custom traits to do nothing on deletion.
///
/// Specialize IntrusiveListAllocTraits to inherit from this to disable
/// non-intrusive deletion (e.g. when nodes are not owned by the list).
///
/// \code
/// template <>
/// struct IntrusiveListAllocTraits<MyType> : IntrusiveListNoAllocTraits<MyType> {};
/// \endcode
template <typename NodeTy>
struct IntrusiveListNoAllocTraits {
    static void deleteNode(NodeTy*) {}
};

/// Callbacks do nothing by default.
///
/// Specialize this to use callbacks when nodes change their list membership.
template <typename NodeTy>
struct IntrusiveListCallbackTraits {
    static void addNodeToList(NodeTy*) {}
    static void removeNodeFromList(NodeTy*) {}

    /// Callback before transferring nodes to this list (e.g. splice).
    /// The nodes may already be in this same list.
    template <typename Iterator>
    static void transferNodesFromList(Iterator /*first*/, Iterator /*last*/) {}
};

/// Template traits for intrusive list.
/// Customize callbacks and allocation semantics.
template <typename NodeTy>
struct IntrusiveListTraits : IntrusiveListAllocTraits<NodeTy>,
                             IntrusiveListCallbackTraits<NodeTy> {};

/// Const traits should never be instantiated.
template <typename Ty>
struct IntrusiveListTraits<const Ty> {};

// Helper: store parent pointer only when Parent is not void
template <typename Parent>
struct IntrusiveListParentStorage {
    Parent* parent_ = nullptr;
    Parent* getParent() const {
        return parent_;
    }
    void setParent(Parent* p) {
        parent_ = p;
    }
};
template <>
struct IntrusiveListParentStorage<void> {
    void* getParent() const {  // NOLINT(readability-convert-member-functions-to-static)
        return nullptr;
    }
    void setParent(void*) {}
};

// Base class for nodes that can be inserted into an intrusive list.
// When Parent is not void, getParent() returns the list's parent (e.g. BasicBlock*).
template <typename T, typename Parent = void>
class IntrusiveListNode {
    template <typename U, typename P, typename Tr>
    friend class IntrusiveList;
    friend class IntrusiveListBase<T, Parent>;
    friend class IntrusiveListIterator<T>;
    friend class IntrusiveListIterator<const T>;

   private:
    T* prev_;
    T* next_;

    /// The list that this node is in. nullptr if not in any list.
    IntrusiveListBase<T, Parent>* parent_list_;

   public:
    // NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
    IntrusiveListNode() : prev_(nullptr), next_(nullptr), parent_list_(nullptr) {}

    // NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
    IntrusiveListNode(const IntrusiveListNode&) = delete;
    IntrusiveListNode& operator=(const IntrusiveListNode&) = delete;

    ~IntrusiveListNode() = default;

    // Check if this node is currently in a list
    bool isInList() const {
        return parent_list_ != nullptr;
    }

    // Get the next/previous nodes (returns nullptr if at end/beginning)
    T* getNext() const {
        return next_;
    }
    T* getPrev() const {
        return prev_;
    }

    /// When Parent is not void, return the list's parent (e.g. the BasicBlock owning this node).
    /// Otherwise not available.
    template <typename P = Parent>
    typename std::enable_if<!std::is_void<P>::value, P*>::type getParent() {
        return parent_list_ ? parent_list_->getParent() : nullptr;
    }
    template <typename P = Parent>
    typename std::enable_if<!std::is_void<P>::value, const P*>::type getParent() const {
        return parent_list_ ? parent_list_->getParent() : nullptr;
    }

    // Remove this node from whatever list it's currently in
    void removeFromList() {
        if (!parent_list_) return;

        if (prev_)
            prev_->next_ = next_;
        else
            parent_list_->head_ = next_;

        if (next_)
            next_->prev_ = prev_;
        else
            parent_list_->tail_ = prev_;

        --parent_list_->size_;
        prev_ = next_ = nullptr;
        parent_list_ = nullptr;
    }
};

// Iterator for the intrusive list
template <typename T>
class IntrusiveListIterator {
   private:
    T* node_;

   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;

    explicit IntrusiveListIterator(T* node = nullptr) : node_(node) {}

    // Conversion constructor from non-const to const iterator
    template <typename U = T>
    IntrusiveListIterator(
        const IntrusiveListIterator<U>& other,
        typename std::enable_if<std::is_same<U, typename std::remove_const<T>::type>::value &&
                                std::is_const<T>::value>::type* = nullptr)
        : node_(other.getNodePtr()) {}

    T& operator*() const {
        return *node_;
    }
    T* operator->() const {
        return node_;
    }

    IntrusiveListIterator& operator++() {
        if (node_) node_ = node_->next_;
        return *this;
    }

    IntrusiveListIterator operator++(int) {
        IntrusiveListIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    IntrusiveListIterator& operator--() {
        if (node_) node_ = node_->prev_;
        return *this;
    }

    IntrusiveListIterator operator--(int) {
        IntrusiveListIterator tmp = *this;
        --(*this);
        return tmp;
    }

    IntrusiveListIterator operator+(size_t n) const {
        IntrusiveListIterator tmp = *this;
        for (size_t i = 0; i < n; ++i) ++tmp;
        return tmp;
    }

    bool operator==(const IntrusiveListIterator& other) const {
        return node_ == other.node_;
    }

    bool operator!=(const IntrusiveListIterator& other) const {
        return node_ != other.node_;
    }

    // Get the underlying node pointer
    T* getNodePtr() const {
        return node_;
    }
};

// Base for intrusive list: holds storage and link fields so the node can store a list pointer
// that works with any Traits. IntrusiveList<T, Parent, Traits> inherits this.
template <typename T, typename Parent>
class IntrusiveListBase {
    friend class IntrusiveListNode<T, Parent>;

   protected:
    IntrusiveListParentStorage<Parent> parent_storage_;

    T* head_;
    T* tail_;
    size_t size_;

   public:
    IntrusiveListBase() : head_(nullptr), tail_(nullptr), size_(0) {}

    /// When Parent is not void, return the parent container.
    template <typename P = Parent>
    typename std::enable_if<!std::is_void<P>::value, P*>::type getParent() {
        return static_cast<P*>(parent_storage_.getParent());
    }
    template <typename P = Parent>
    typename std::enable_if<!std::is_void<P>::value, const P*>::type getParent() const {
        return static_cast<const P*>(parent_storage_.getParent());
    }
};

// The intrusive list container. When Parent is not void, the list stores the parent and nodes can
// query it via getParent(). By default uses IntrusiveListTraits<T> (delete on erase/clear; no-op
// callbacks).
template <typename T, typename Parent = void, typename Traits = IntrusiveListTraits<T>>
class IntrusiveList : public IntrusiveListBase<T, Parent> {
    friend class IntrusiveListNode<T, Parent>;

   public:
    using iterator = IntrusiveListIterator<T>;
    using const_iterator = IntrusiveListIterator<const T>;

    IntrusiveList() = default;

    /// When Parent is not void, construct the list with its parent (e.g. BasicBlock* for IRList).
    template <typename P = Parent>
    explicit IntrusiveList(typename std::enable_if<!std::is_void<P>::value, P*>::type p) {
        this->parent_storage_.setParent(p);
    }

    /// When Parent is not void, return the parent container (e.g. the BasicBlock owning this list).
    using IntrusiveListBase<T, Parent>::getParent;

    // Disable copy constructor and assignment for now
    IntrusiveList(const IntrusiveList&) = delete;
    IntrusiveList& operator=(const IntrusiveList&) = delete;

    ~IntrusiveList() {
        clear();
    }

    // Iterator support
    iterator begin() {
        return iterator(this->head_);
    }
    iterator end() {
        return iterator(nullptr);
    }
    const_iterator begin() const {
        return const_iterator(this->head_);
    }
    const_iterator end() const {
        return const_iterator(nullptr);
    }

    // Reverse iterator support
    class reverse_iterator {
       private:
        T* node_;
        IntrusiveList* list_;

       public:
        reverse_iterator(T* node, IntrusiveList* list) : node_(node), list_(list) {}

        T& operator*() const {
            return *node_;
        }

        T* operator->() const {
            return node_;
        }

        reverse_iterator& operator++() {
            if (node_) node_ = node_->prev_;
            return *this;
        }

        reverse_iterator operator++(int) {
            reverse_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        reverse_iterator& operator--() {
            if (node_)
                node_ = node_->next_;
            else if (list_)
                node_ = list_->tail_;  // From rend() to last element (base member)
            return *this;
        }

        reverse_iterator operator--(int) {
            reverse_iterator tmp = *this;
            --(*this);
            return tmp;
        }

        bool operator==(const reverse_iterator& other) const {
            return node_ == other.node_;
        }

        bool operator!=(const reverse_iterator& other) const {
            return node_ != other.node_;
        }

        // Get the underlying node pointer
        T* getNodePtr() const {
            return node_;
        }
    };

    using const_reverse_iterator = reverse_iterator;

    reverse_iterator rbegin() {
        return reverse_iterator(this->tail_, this);
    }

    reverse_iterator rend() {
        return reverse_iterator(nullptr, this);
    }

    const_reverse_iterator rbegin() const {
        return const_reverse_iterator(this->tail_, const_cast<IntrusiveList*>(this));
    }

    const_reverse_iterator rend() const {
        return const_reverse_iterator(nullptr, const_cast<IntrusiveList*>(this));
    }

    // Size and empty check
    size_t size() const {
        return this->size_;
    }
    bool empty() const {
        return this->size_ == 0;
    }

    // Access front and back
    T& front() {
        return *this->head_;
    }
    T& back() {
        return *this->tail_;
    }
    const T& front() const {
        return *this->head_;
    }
    const T& back() const {
        return *this->tail_;
    }

    // Add to front or back
    void push_front(T* node) {
        if (!node) return;
        node->removeFromList();

        node->next_ = this->head_;
        node->prev_ = nullptr;
        node->parent_list_ = this;

        if (this->head_)
            this->head_->prev_ = node;
        else
            this->tail_ = node;

        this->head_ = node;
        ++this->size_;
        Traits::addNodeToList(node);
    }

    void push_back(T* node) {
        if (!node) return;
        node->removeFromList();

        node->prev_ = this->tail_;
        node->next_ = nullptr;
        node->parent_list_ = this;

        if (this->tail_)
            this->tail_->next_ = node;
        else
            this->head_ = node;

        this->tail_ = node;
        ++this->size_;
        Traits::addNodeToList(node);
    }

    // Insert before iterator position
    iterator insert(iterator pos, T* node) {
        if (!node) return pos;

        if (pos == end()) {
            push_back(node);
            return iterator(node);
        }

        T* insertPos = pos.getNodePtr();
        node->removeFromList();

        node->next_ = insertPos;
        node->prev_ = insertPos->prev_;
        node->parent_list_ = this;

        if (insertPos->prev_)
            insertPos->prev_->next_ = node;
        else
            this->head_ = node;

        insertPos->prev_ = node;
        ++this->size_;
        Traits::addNodeToList(node);
        return iterator(node);
    }

    // Remove node from list (and delete via traits when using default alloc traits)
    iterator erase(iterator pos) {
        if (pos == end()) return end();

        T* node = pos.getNodePtr();
        T* next = node->next_;
        node->removeFromList();
        Traits::removeNodeFromList(node);
        Traits::deleteNode(node);
        return iterator(next);
    }

    // Remove specific node (if it's in this list). Does not delete.
    void remove(T* node) {
        if (!node || node->parent_list_ != this) return;
        node->removeFromList();
        Traits::removeNodeFromList(node);
    }

    // Clear all nodes (deletes via traits when using default alloc traits)
    void clear() {
        while (!empty()) {
            T* node = this->head_;
            node->removeFromList();
            Traits::removeNodeFromList(node);
            Traits::deleteNode(node);
        }
    }

    // Move operations
    void moveBefore(iterator what, iterator where) {
        if (what == where || what == end()) return;

        T* node = what.getNodePtr();
        if (node->parent_list_ != this) return;

        // Remove from current position (but keep in same list)
        if (node->prev_)
            node->prev_->next_ = node->next_;
        else
            this->head_ = node->next_;

        if (node->next_)
            node->next_->prev_ = node->prev_;
        else
            this->tail_ = node->prev_;

        // Insert before where
        if (where == end()) {
            // Insert at end
            node->prev_ = this->tail_;
            node->next_ = nullptr;
            if (this->tail_)
                this->tail_->next_ = node;
            else
                this->head_ = node;
            this->tail_ = node;
        } else {
            T* insertPos = where.getNodePtr();
            node->next_ = insertPos;
            node->prev_ = insertPos->prev_;

            if (insertPos->prev_)
                insertPos->prev_->next_ = node;
            else
                this->head_ = node;

            insertPos->prev_ = node;
        }
    }

    void moveAfter(iterator what, iterator where) {
        if (what == where || what == end() || where == end()) return;

        iterator nextPos = where;
        ++nextPos;
        moveBefore(what, nextPos);
    }

    // Move to front or back
    void moveToFront(iterator what) {
        moveBefore(what, begin());
    }

    void moveToBack(iterator what) {
        moveBefore(what, end());
    }

    bool isInList(const T* node) const {
        return node && node->parent_list_ == this;
    }
};
}  // namespace stinkytofu
