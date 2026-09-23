//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// Example move-only deleter

#ifndef SUPPORT_DELETER_TYPES_H
#define SUPPORT_DELETER_TYPES_H

#include <type_traits>
#include <utility>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

#if TEST_STD_VER >= 11

// These two should really go in unique_ptr_test_helper.h, but we need them here so the host deleters can decrement
// count.
// Trivially Moveable, but not copy constructible (even non-trivially)
struct A_h {
  static int count;
  TEST_CONSTEXPR_CXX23 A_h() {
    if (!TEST_IS_CONSTANT_EVALUATED)
      ++count;
  }
  A_h(A_h&&) = default;
};
int A_h::count = 0;

// Trivially Moveable, but not copy constructible (even non-trivially)
struct B_h : public A_h {
  static int count;
  TEST_CONSTEXPR_CXX23 B_h() {
    if (!TEST_IS_CONSTANT_EVALUATED)
      ++count;
  }
  B_h(B_h&&) = default;
};
int B_h::count = 0;


template <class T>
class Deleter {
  int state_;

  __host__ __device__ Deleter(const Deleter&);
  __host__ __device__ Deleter& operator=(const Deleter&);

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 Deleter(Deleter&& r) : state_(r.state_) { r.state_ = 0; }
  __host__ __device__ TEST_CONSTEXPR_CXX23 Deleter& operator=(Deleter&& r) {
    state_   = r.state_;
    r.state_ = 0;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 Deleter() : state_(0) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit Deleter(int s) : state_(s) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~Deleter() {
    assert(state_ >= 0);
    state_ = -1;
  }

  template <class U>
  __host__ __device__ TEST_CONSTEXPR_CXX23 Deleter(Deleter<U>&& d, typename ::std::enable_if<!::std::is_same<U, T>::value>::type* = 0)
      : state_(d.state()) {
    d.set_state(0);
  }

private:
  template <class U>
  __host__ __device__ Deleter(const Deleter<U>& d, typename ::std::enable_if<!::std::is_same<U, T>::value>::type* = 0);

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const { return state_; }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void set_state(int i) { state_ = i; }

  __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p) { delete p; }
  __host__ TEST_CONSTEXPR_CXX23 void operator()(T *p) {
    assert(hipFree(p) == hipSuccess);
    if (::std::is_same<T, B_h>::value) {
        --B_h::count;
        --A_h::count;
    }
    if (::std::is_same<T, A_h>::value) {
        --A_h::count;
    }
  }
};

template <class T>
class Deleter<T[]> {
  int state_;

  __host__ __device__ Deleter(const Deleter&);
  __host__ __device__ Deleter& operator=(const Deleter&);

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 Deleter(Deleter&& r) : state_(r.state_) { r.state_ = 0; }
  __host__ __device__ TEST_CONSTEXPR_CXX23 Deleter& operator=(Deleter&& r) {
    state_   = r.state_;
    r.state_ = 0;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 Deleter() : state_(0) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit Deleter(int s) : state_(s) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~Deleter() {
    assert(state_ >= 0);
    state_ = -1;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const { return state_; }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void set_state(int i) { state_ = i; }

  __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p) { delete[] p; }
  __host__ TEST_CONSTEXPR_CXX23 void operator()(T* p) { assert(hipFree(p) == hipSuccess); }
};

#else // TEST_STD_VER < 11

template <class T>
class Deleter {
  mutable int state_;

public:
  __host__ __device__ Deleter() : state_(0) {}
  __host__ __device__ explicit Deleter(int s) : state_(s) {}

  __host__ __device__ Deleter(Deleter const& other) : state_(other.state_) { other.state_ = 0; }
  __host__ __device__ Deleter& operator=(Deleter const& other) {
    state_       = other.state_;
    other.state_ = 0;
    return *this;
  }

  __host__ __device__ ~Deleter() {
    assert(state_ >= 0);
    state_ = -1;
  }

  template <class U>
  __host__ __device__ Deleter(Deleter<U> d, typename ::std::enable_if<!::std::is_same<U, T>::value>::type* = 0) : state_(d.state()) {}

public:
  __host__ __device__ int state() const { return state_; }
  __host__ __device__ void set_state(int i) { state_ = i; }

  __device__ void operator()(T* p) { delete p; }
  __host__ void operator()(T *p) {
    assert(hipFree(p) == hipSuccess);
    if (::std::is_same<T, B_h>::value) {
        --B_h::count;
        --A_h::count;
    }
    if (::std::is_same<T, A_h>::value) {
        --A_h::count;
    }
  }
};

template <class T>
class Deleter<T[]> {
  mutable int state_;

public:
  __host__ __device__ Deleter(Deleter const& other) : state_(other.state_) { other.state_ = 0; }
  __host__ __device__ Deleter& operator=(Deleter const& other) {
    state_       = other.state_;
    other.state_ = 0;
    return *this;
  }

  __host__ __device__ Deleter() : state_(0) {}
  __host__ __device__ explicit Deleter(int s) : state_(s) {}
  __host__ __device__ ~Deleter() {
    assert(state_ >= 0);
    state_ = -1;
  }

  __host__ __device__ int state() const { return state_; }
  __host__ __device__ void set_state(int i) { state_ = i; }

  __device__ void operator()(T* p) { delete[] p; }
  __host__ void operator()(T* p) { assert(hipFree(p) == hipSuccess); }
};

#endif

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX23 void swap(Deleter<T>& x, Deleter<T>& y) {
  Deleter<T> t(::std::move(x));
  x = ::std::move(y);
  y = ::std::move(t);
}

template <class T>
class CDeleter {
  int state_;

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 CDeleter() : state_(0) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit CDeleter(int s) : state_(s) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 CDeleter(const CDeleter&) = default;
  __host__ __device__ TEST_CONSTEXPR_CXX23 CDeleter& operator=(const CDeleter&) = default;
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~CDeleter() {
    assert(state_ >= 0);
    state_ = -1;
  }

  template <class U>
  __host__ __device__ TEST_CONSTEXPR_CXX23 CDeleter(const CDeleter<U>& d) : state_(d.state()) {}

  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const { return state_; }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void set_state(int i) { state_ = i; }

  __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p) { delete p; }
  __host__ TEST_CONSTEXPR_CXX23 void operator()(T *p) {
    assert(hipFree(p) == hipSuccess);
    if (::std::is_same<T, B_h>::value) {
        --B_h::count;
        --A_h::count;
    }
    if (::std::is_same<T, A_h>::value) {
        --A_h::count;
    }
  }
};

template <class T>
class CDeleter<T[]> {
  int state_;

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 CDeleter() : state_(0) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit CDeleter(int s) : state_(s) {}
  template <class U>
  __host__ __device__ TEST_CONSTEXPR_CXX23 CDeleter(const CDeleter<U>& d) : state_(d.state()) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 CDeleter(const CDeleter&) = default;
  __host__ __device__ TEST_CONSTEXPR_CXX23 CDeleter& operator=(const CDeleter&) = default;
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~CDeleter() {
    assert(state_ >= 0);
    state_ = -1;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const { return state_; }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void set_state(int i) { state_ = i; }

  __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p) { delete[] p; }
  __host__ TEST_CONSTEXPR_CXX23 void operator()(T* p) { assert(hipFree(p) == hipSuccess); }
};

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX23 void swap(CDeleter<T>& x, CDeleter<T>& y) {
  CDeleter<T> t(::std::move(x));
  x = ::std::move(y);
  y = ::std::move(t);
}

// Non-copyable deleter
template <class T>
class NCDeleter {
  int state_;
  __host__ __device__ NCDeleter(NCDeleter const&);
  __host__ __device__ NCDeleter& operator=(NCDeleter const&);

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 NCDeleter() : state_(0) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit NCDeleter(int s) : state_(s) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~NCDeleter() {
    assert(state_ >= 0);
    state_ = -1;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const { return state_; }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void set_state(int i) { state_ = i; }

  __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p) { delete p; }
  __host__ TEST_CONSTEXPR_CXX23 void operator()(T *p) {
    assert(hipFree(p) == hipSuccess);
    if (::std::is_same<T, B_h>::value) {
        --B_h::count;
        --A_h::count;
    }
    else if (::std::is_same<T, A_h>::value) {
        --A_h::count;
    }
  }
};

template <class T>
class NCDeleter<T[]> {
  int state_;
  __host__ __device__ NCDeleter(NCDeleter const&);
  __host__ __device__ NCDeleter& operator=(NCDeleter const&);

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 NCDeleter() : state_(0) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit NCDeleter(int s) : state_(s) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~NCDeleter() {
    assert(state_ >= 0);
    state_ = -1;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const { return state_; }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void set_state(int i) { state_ = i; }

  __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p) { delete[] p; }
  __host__ TEST_CONSTEXPR_CXX23 void operator()(T* p) { assert(hipFree(p) == hipSuccess); }
};

// Non-copyable deleter
template <class T>
class NCConstDeleter {
  int state_;
  __host__ __device__ NCConstDeleter(NCConstDeleter const&);
  __host__ __device__ NCConstDeleter& operator=(NCConstDeleter const&);

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 NCConstDeleter() : state_(0) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit NCConstDeleter(int s) : state_(s) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~NCConstDeleter() {
    assert(state_ >= 0);
    state_ = -1;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const { return state_; }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void set_state(int i) { state_ = i; }

  __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p) const { delete p; }
  __host__ TEST_CONSTEXPR_CXX23 void operator()(T *p) const {
    assert(hipFree(p) == hipSuccess);
    if (::std::is_same<T, B_h>::value) {
        --B_h::count;
        --A_h::count;
    }
    if (::std::is_same<T, A_h>::value) {
        --A_h::count;
    }
  }
};

template <class T>
class NCConstDeleter<T[]> {
  int state_;
  __host__ __device__ NCConstDeleter(NCConstDeleter const&);
  __host__ __device__ NCConstDeleter& operator=(NCConstDeleter const&);

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 NCConstDeleter() : state_(0) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit NCConstDeleter(int s) : state_(s) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~NCConstDeleter() {
    assert(state_ >= 0);
    state_ = -1;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const { return state_; }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void set_state(int i) { state_ = i; }

  __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p) const { delete[] p; }
  __host__ TEST_CONSTEXPR_CXX23 void operator()(T* p) const { assert(hipFree(p) == hipSuccess); }
};

// Non-copyable deleter
template <class T>
class CopyDeleter {
  int state_;

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 CopyDeleter() : state_(0) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit CopyDeleter(int s) : state_(s) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~CopyDeleter() {
    assert(state_ >= 0);
    state_ = -1;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 CopyDeleter(CopyDeleter const& other) : state_(other.state_) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 CopyDeleter& operator=(CopyDeleter const& other) {
    state_ = other.state_;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const { return state_; }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void set_state(int i) { state_ = i; }

  __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p) { delete p; }
  __host__ TEST_CONSTEXPR_CXX23 void operator()(T *p) {
    assert(hipFree(p) == hipSuccess);
    if (::std::is_same<T, B_h>::value) {
        --B_h::count;
        --A_h::count;
    }
    if (::std::is_same<T, A_h>::value) {
        --A_h::count;
    }
  }
};

template <class T>
class CopyDeleter<T[]> {
  int state_;

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 CopyDeleter() : state_(0) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit CopyDeleter(int s) : state_(s) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 ~CopyDeleter() {
    assert(state_ >= 0);
    state_ = -1;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 CopyDeleter(CopyDeleter const& other) : state_(other.state_) {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 CopyDeleter& operator=(CopyDeleter const& other) {
    state_ = other.state_;
    return *this;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const { return state_; }
  __host__ __device__ TEST_CONSTEXPR_CXX23 void set_state(int i) { state_ = i; }

  __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p) { delete[] p; }
  __host__ TEST_CONSTEXPR_CXX23 void operator()(T* p) { assert(hipFree(p) == hipSuccess); }
};

struct test_deleter_base {
  static int count;
  static int dealloc_count;
};

int test_deleter_base::count         = 0;
int test_deleter_base::dealloc_count = 0;

template <class T>
class test_deleter : public test_deleter_base {
  int state_;

public:
  __host__ __device__ test_deleter() : state_(0) { ++count; }
  __host__ __device__ explicit test_deleter(int s) : state_(s) { ++count; }
  __host__ __device__ test_deleter(const test_deleter& d) : state_(d.state_) { ++count; }
  __host__ __device__ ~test_deleter() {
    assert(state_ >= 0);
    --count;
    state_ = -1;
  }

  __host__ __device__ int state() const { return state_; }
  __host__ __device__ void set_state(int i) { state_ = i; }

  __device__ void operator()(T* p) {
    assert(state_ >= 0);
    ++dealloc_count;
    delete p;
  }
  __host__ TEST_CONSTEXPR_CXX23 void operator()(T *p) {
    assert(state_ >= 0);
    ++dealloc_count;
    assert(hipFree(p) == hipSuccess);
    if (::std::is_same<T, B_h>::value) {
        --B_h::count;
        --A_h::count;
    }
    if (::std::is_same<T, A_h>::value) {
        --A_h::count;
    }
  }
#if TEST_STD_VER >= 11
  __host__ __device__ test_deleter* operator&() const = delete;
#else

private:
  __host__ __device__ test_deleter* operator&() const;
#endif
};

template <class T>
__host__ __device__ void swap(test_deleter<T>& x, test_deleter<T>& y) {
  test_deleter<T> t(::std::move(x));
  x = ::std::move(y);
  y = ::std::move(t);
}

#if TEST_STD_VER >= 11

template <class T, size_t ID = 0>
class PointerDeleter {
  __host__ __device__ PointerDeleter(const PointerDeleter&);
  __host__ __device__ PointerDeleter& operator=(const PointerDeleter&);

public:
  typedef min_pointer<T, ::std::integral_constant<size_t, ID>> pointer;

  __host__ __device__ TEST_CONSTEXPR_CXX23 PointerDeleter()                            = default;
  __host__ __device__ TEST_CONSTEXPR_CXX23 PointerDeleter(PointerDeleter&&)            = default;
  __host__ __device__ TEST_CONSTEXPR_CXX23 PointerDeleter& operator=(PointerDeleter&&) = default;
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit PointerDeleter(int) {}

  template <class U>
  __host__ __device__ TEST_CONSTEXPR_CXX23
  PointerDeleter(PointerDeleter<U, ID>&&, typename ::std::enable_if<!::std::is_same<U, T>::value>::type* = 0) {}

  __device__ TEST_CONSTEXPR_CXX23 void operator()(pointer p) {
    if (p) {
      delete ::std::addressof(*p);
    }
  }
  __host__ TEST_CONSTEXPR_CXX23 void operator()(pointer p) {
    if (p) {
      assert(hipFree(::std::addressof(*p)) == hipSuccess);
      if (::std::is_same<T, B_h>::value) {
          --B_h::count;
          --A_h::count;
      }
      if (::std::is_same<T, A_h>::value) {
          --A_h::count;
      }
    }
  }

private:
  template <class U>
  __host__ __device__ PointerDeleter(const PointerDeleter<U, ID>&, typename ::std::enable_if<!::std::is_same<U, T>::value>::type* = 0);
};

template <class T, size_t ID>
class PointerDeleter<T[], ID> {
  __host__ __device__ PointerDeleter(const PointerDeleter&);
  __host__ __device__ PointerDeleter& operator=(const PointerDeleter&);

public:
  typedef min_pointer<T, ::std::integral_constant<size_t, ID> > pointer;

  __host__ __device__ TEST_CONSTEXPR_CXX23 PointerDeleter()                            = default;
  __host__ __device__ TEST_CONSTEXPR_CXX23 PointerDeleter(PointerDeleter&&)            = default;
  __host__ __device__ TEST_CONSTEXPR_CXX23 PointerDeleter& operator=(PointerDeleter&&) = default;
  __host__ __device__ TEST_CONSTEXPR_CXX23 explicit PointerDeleter(int) {}

  template <class U>
  __host__ __device__ TEST_CONSTEXPR_CXX23
  PointerDeleter(PointerDeleter<U, ID>&&, typename ::std::enable_if<!::std::is_same<U, T>::value>::type* = 0) {}

  __device__ TEST_CONSTEXPR_CXX23 void operator()(pointer p) {
    if (p) {
      delete[] ::std::addressof(*p);
    }
  }
  __host__ TEST_CONSTEXPR_CXX23 void operator()(pointer p) {
    if (p) {
      assert(hipFree(::std::addressof(*p)) == hipSuccess);
    }
  }

private:
  template <class U>
  __host__ __device__ PointerDeleter(const PointerDeleter<U, ID>&, typename ::std::enable_if<!::std::is_same<U, T>::value>::type* = 0);
};

#endif // TEST_STD_VER >= 11

template <class T>
class DefaultCtorDeleter {
  int state_;

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const { return state_; }
  __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p) { delete p; }
  __host__ TEST_CONSTEXPR_CXX23 void operator()(T *p) {
    assert(hipFree(p) == hipSuccess);
    if (::std::is_same<T, B_h>::value) {
      --B_h::count;
      --A_h::count;
    }
    if (::std::is_same<T, A_h>::value) {
      --A_h::count;
    }
  }
};

template <class T>
class DefaultCtorDeleter<T[]> {
  int state_;

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const { return state_; }
  __device__ TEST_CONSTEXPR_CXX23 void operator()(T* p) { delete[] p; }
  __host__ TEST_CONSTEXPR_CXX23 void operator()(T* p) { assert(hipFree(p) == hipSuccess); }
};

#endif // SUPPORT_DELETER_TYPES_H
