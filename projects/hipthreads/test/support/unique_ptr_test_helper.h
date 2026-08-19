//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_UNIQUE_PTR_TEST_HELPER_H
#define TEST_SUPPORT_UNIQUE_PTR_TEST_HELPER_H

#include <memory>
#include <type_traits>

#include "test_macros.h"
#include "deleter_types.h"

struct A {
  static __device__ int count;
  __device__ TEST_CONSTEXPR_CXX23 A() {
    if (!TEST_IS_CONSTANT_EVALUATED)
      ++count;
  }
  __device__ TEST_CONSTEXPR_CXX23 A(const A&) {
    if (!TEST_IS_CONSTANT_EVALUATED)
      ++count;
  }
  __device__ TEST_CONSTEXPR_CXX23 virtual ~A() {
    if (!TEST_IS_CONSTANT_EVALUATED)
      --count;
  }
};

__device__ int A::count = 0;

struct B : public A {
  static __device__ int count;
  __device__ TEST_CONSTEXPR_CXX23 B() {
    if (!TEST_IS_CONSTANT_EVALUATED)
      ++count;
  }
  __device__ TEST_CONSTEXPR_CXX23 B(const B& other) : A(other) {
    if (!TEST_IS_CONSTANT_EVALUATED)
      ++count;
  }
  __device__ TEST_CONSTEXPR_CXX23 virtual ~B() {
    if (!TEST_IS_CONSTANT_EVALUATED)
      --count;
  }
};

__device__ int B::count = 0;

template <class T>
__device__ TEST_CONSTEXPR_CXX23 typename ::std::enable_if<!::std::is_array<T>::value, T*>::type newValue(int num_elements) {
  assert(num_elements == 1);
  return new T;
}

template <class T>
__device__ TEST_CONSTEXPR_CXX23 typename ::std::enable_if<::std::is_array<T>::value, typename ::std::remove_all_extents<T>::type*>::type
newValue(int num_elements) {
  typedef typename ::std::remove_all_extents<T>::type VT;
  assert(num_elements >= 1);
  return new VT[num_elements];
}

template <class T>
__host__ TEST_CONSTEXPR_CXX23 typename ::std::enable_if<!::std::is_array<T>::value, T*>::type newValue(int num_elements) {
  assert(num_elements == 1);
  T *__buf;
  assert(hipMalloc(&__buf, sizeof(T) == 0 ? 1 : sizeof(T)) == hipSuccess);
  return __buf;
}

template <class T>
__host__ TEST_CONSTEXPR_CXX23 typename ::std::enable_if<::std::is_array<T>::value, typename ::std::remove_all_extents<T>::type*>::type
newValue(int num_elements) {
  typedef typename ::std::remove_all_extents<T>::type VT;
  assert(num_elements >= 1);
  VT *__buf;
  assert(hipMalloc(&__buf, sizeof(VT) == 0 ? 1 : sizeof(VT) * num_elements) == hipSuccess);
  return __buf;
}

template <>
__host__ TEST_CONSTEXPR_CXX23 const A_h* newValue<const A_h>(int num_elements) {
  assert(num_elements == 1);
  A_h *__buf;
  assert(hipMalloc(&__buf, sizeof(A_h) == 0 ? 1 : sizeof(A_h)) == hipSuccess);
  ++A_h::count;
  return __buf;
}
template <>
__host__ TEST_CONSTEXPR_CXX23 A_h *newValue<A_h>(int num_elements) {
  return const_cast<A_h *>(newValue<const A_h>(num_elements));
}

template <>
__host__ TEST_CONSTEXPR_CXX23 const A_h* newValue<const A_h[]>(int num_elements) {
  assert(num_elements >= 1);
  A_h *__buf;
  assert(hipMalloc(&__buf, sizeof(A_h) == 0 ? 1 : sizeof(A_h) * num_elements) == hipSuccess);
  A_h::count += num_elements;
  return __buf;
}
template <>
__host__ TEST_CONSTEXPR_CXX23 A_h *newValue<A_h[]>(int num_elements) {
  return const_cast<A_h *>(newValue<const A_h[]>(num_elements));
}

template <>
__host__ TEST_CONSTEXPR_CXX23 const B_h* newValue<const B_h>(int num_elements) {
  assert(num_elements == 1);
  B_h *__buf;
  assert(hipMalloc(&__buf, sizeof(B_h) == 0 ? 1 : sizeof(B_h)) == hipSuccess);
  ++A_h::count;
  ++B_h::count;
  return __buf;
}
template <>
__host__ TEST_CONSTEXPR_CXX23 B_h *newValue<B_h>(int num_elements) {
  return const_cast<B_h *>(newValue<const B_h>(num_elements));
}

template <>
__host__ TEST_CONSTEXPR_CXX23 const B_h* newValue<const B_h[]>(int num_elements) {
  assert(num_elements >= 1);
  B_h *__buf;
  assert(hipMalloc(&__buf, sizeof(B_h) == 0 ? 1 : sizeof(B_h) * num_elements) == hipSuccess);
  A_h::count += num_elements;
  B_h::count += num_elements;
  return __buf;
}
template <>
__host__ TEST_CONSTEXPR_CXX23 B_h *newValue<B_h[]>(int num_elements) {
  return const_cast<B_h *>(newValue<const B_h[]>(num_elements));
}

struct IncompleteType;

__device__ void checkNumIncompleteTypeAlive(int i);
__device__ int getNumIncompleteTypeAlive();
__device__ IncompleteType* getNewIncomplete();
__device__ IncompleteType* getNewIncompleteArray(int size);

#if TEST_STD_VER >= 11
template <class ThisT, class... Args>
struct args_is_this_type : ::std::false_type {};

template <class ThisT, class A1>
struct args_is_this_type<ThisT, A1> : ::std::is_same<ThisT, typename ::std::decay<A1>::type> {};
#endif

template <class IncompleteT = IncompleteType, class Del = hip::default_delete<IncompleteT> >
struct StoresIncomplete {
  static_assert(
      (::std::is_same<IncompleteT, IncompleteType>::value || ::std::is_same<IncompleteT, IncompleteType[]>::value), "");

  hip::unique_ptr<IncompleteT, Del> m_ptr;

#if TEST_STD_VER >= 11
  __device__ StoresIncomplete(StoresIncomplete const&) = delete;
  __device__ StoresIncomplete(StoresIncomplete&&)      = default;

  template <class... Args>
  __device__ StoresIncomplete(Args&&... args) : m_ptr(::std::forward<Args>(args)...) {
    static_assert(!args_is_this_type<StoresIncomplete, Args...>::value, "");
  }
#else

private:
  __device__ StoresIncomplete();
  __device__ StoresIncomplete(StoresIncomplete const&);

public:
#endif

  __device__ ~StoresIncomplete();

  __device__ IncompleteType* get() const { return m_ptr.get(); }
  __device__ Del& get_deleter() { return m_ptr.get_deleter(); }
};

#if TEST_STD_VER >= 11
template <class IncompleteT = IncompleteType, class Del = hip::default_delete<IncompleteT>, class... Args>
__device__ void doIncompleteTypeTest(int expect_alive, Args&&... ctor_args) {
  checkNumIncompleteTypeAlive(expect_alive);
  {
    StoresIncomplete<IncompleteT, Del> sptr(::std::forward<Args>(ctor_args)...);
    checkNumIncompleteTypeAlive(expect_alive);
    if (expect_alive == 0)
      assert(sptr.get() == nullptr);
    else
      assert(sptr.get() != nullptr);
  }
  checkNumIncompleteTypeAlive(0);
}
#endif

#define INCOMPLETE_TEST_EPILOGUE()                                                                                     \
  int is_incomplete_test_anchor = [](){ hipLaunchKernelGGL(is_incomplete_test, dim3(1), dim3(1), 0, nullptr); return 0; }(); \
                                                                                                                       \
  struct IncompleteType {                                                                                              \
    static __device__ int count;                                                                                       \
    __device__ IncompleteType() { ++count; }                                                                           \
    __device__ ~IncompleteType() { --count; }                                                                          \
  };                                                                                                                   \
                                                                                                                       \
  int IncompleteType::count = 0;                                                                                       \
                                                                                                                       \
  __device__ void checkNumIncompleteTypeAlive(int i) { assert(IncompleteType::count == i); }                           \
  __device__ int getNumIncompleteTypeAlive() { return IncompleteType::count; }                                         \
  __device__ IncompleteType* getNewIncomplete() { return new IncompleteType; }                                         \
  __device__ IncompleteType* getNewIncompleteArray(int size) { return new IncompleteType[size]; }                      \
                                                                                                                       \
  template <class IncompleteT, class Del>                                                                              \
  __device__ StoresIncomplete<IncompleteT, Del>::~StoresIncomplete() {}
#

#if TEST_STD_VER >= 11
#  define DEFINE_AND_RUN_IS_INCOMPLETE_TEST(...)                                                                       \
    __global__ static void is_incomplete_test() { __VA_ARGS__ return; }                                                \
    INCOMPLETE_TEST_EPILOGUE()
#else
#  define DEFINE_AND_RUN_IS_INCOMPLETE_TEST(...)                                                                       \
    __global__ static void is_incomplete_test() { return; }                                                            \
    INCOMPLETE_TEST_EPILOGUE()
#endif

#endif // TEST_SUPPORT_UNIQUE_PTR_TEST_HELPER_H
