//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: hipthreads-no-stop-token
// XFAIL: availability-synchronization_library-missing

// template<class F, class... Args>
// explicit jthread(F&& f, Args&&... args);

#include <cassert>
#include <stop_token>
#include <hip/thread>
#include <type_traits>
#include <utility>

#include "force_include_hip.h"
#include "make_test_thread.h"
#include "test_macros.h"

template <class... Args>
struct Func {
  void operator()(Args...) const;
};

// Constraints: remove_cvref_t<F> is not the same type as jthread.
static_assert(::std::is_constructible_v<hip::jthread, Func<>>);
static_assert(::std::is_constructible_v<hip::jthread, Func<int>, int>);
static_assert(!::std::is_constructible_v<hip::jthread, hip::jthread const&>);

// explicit
template <class T>
void conversion_test(T);

template <class T, class... Args>
concept ImplicitlyConstructible = requires(Args&&... args) { conversion_test<T>({::std::forward<Args>(args)...}); };

static_assert(!ImplicitlyConstructible<hip::jthread, Func<>>);
static_assert(!ImplicitlyConstructible<hip::jthread, Func<int>, int>);

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  // Effects: Initializes ssource
  // Postconditions: get_id() != id() is true and ssource.stop_possible() is true
  // and *this represents the newly started thread.
  {
    hip::jthread jt = support::make_test_jthread([] __device__ () {});
    assert(jt.get_stop_source().stop_possible());
    assert(jt.get_id() != hip::jthread::id());
  }

  // The new thread of execution executes
  // invoke(auto(::std::forward<F>(f)), get_stop_token(), auto(::std::forward<Args>(args))...)
  // if that expression is well-formed,
  {
    int result = 0;
    hip::jthread jt{[&result] __device__ (::std::stop_token st, int i) {
                      assert(st.stop_possible());
                      assert(!st.stop_requested());
                      result += i;
                    },
                    5};
    jt.join();
    assert(result == 5);
  }

  // otherwise
  // invoke(auto(::std::forward<F>(f)), auto(::std::forward<Args>(args))...)
  {
    int result = 0;
    hip::jthread jt{[&result] __device__ (int i) { result += i; }, 5};
    jt.join();
    assert(result == 5);
  }

  // with the values produced by auto being materialized ([conv.rval]) in the constructing thread.
  /*
  {
    struct TrackThread {
      ::std::jthread::id threadId;
      bool copyConstructed = false;
      bool moveConstructed = false;

      TrackThread() : threadId(hip::this_thread::get_id()) {}
      TrackThread(const TrackThread&) : threadId(hip::this_thread::get_id()), copyConstructed(true) {}
      TrackThread(TrackThread&&) : threadId(hip::this_thread::get_id()), moveConstructed(true) {}
    };

    auto mainThread = hip::this_thread::get_id();

    TrackThread arg1;
    ::std::jthread jt1{[mainThread](const TrackThread& arg) {
                       assert(arg.threadId == mainThread);
                       assert(arg.threadId != hip::this_thread::get_id());
                       assert(arg.copyConstructed);
                     },
                     arg1};

    TrackThread arg2;
    ::std::jthread jt2{[mainThread](const TrackThread& arg) {
                       assert(arg.threadId == mainThread);
                       assert(arg.threadId != hip::this_thread::get_id());
                       assert(arg.moveConstructed);
                     },
                     ::std::move(arg2)};
  }
  */
  // TrackThread copy/move-construction tracking block not portable to GPU:
  // hip::jthread serialises args by raw copy (TriviallyCopyable required),
  // so observing copy- vs move-ctor side effects across the host/device boundary
  // does not match libcxx semantics.

#if !defined(TEST_HAS_NO_EXCEPTIONS)
  // [Note 1: This implies that any exceptions not thrown from the invocation of the copy
  // of f will be thrown in the constructing thread, not the new thread. - end note]
  /*
  {
    struct Exception {
      ::std::jthread::id threadId;
    };
    struct ThrowOnCopyFunc {
      ThrowOnCopyFunc() = default;
      ThrowOnCopyFunc(const ThrowOnCopyFunc&) { throw Exception{hip::this_thread::get_id()}; }
      void operator()() const {}
    };
    ThrowOnCopyFunc f1;
    try {
      ::std::jthread jt{f1};
      assert(false);
    } catch (const Exception& e) {
      assert(e.threadId == hip::this_thread::get_id());
    }
  }
  */
  // ThrowOnCopyFunc exception-propagation block not portable to GPU:
  // device code has no C++ exceptions.
#endif // !defined(TEST_HAS_NO_EXCEPTIONS)

  // Synchronization: The completion of the invocation of the constructor
  // synchronizes with the beginning of the invocation of the copy of f.
  /*
  {
    int flag = 0;
    struct Arg {
      int& flag_;
      Arg(int& f) : flag_(f) {}

      Arg(const Arg& other) : flag_(other.flag_) { flag_ = 5; }
    };

    Arg arg(flag);
    ::std::jthread jt(
        [&flag](const auto&) {
          assert(flag == 5); // happens-after the copy-construction of arg
        },
        arg);
  }
  */
  // int& flag synchronization block not portable to GPU:
  // capturing a reference to a host-stack int into a __device__ lambda is
  // invalid; host stack is not visible to the GPU.

  // Per https://eel.is/c++draft/thread.jthread.class#thread.jthread.cons-8:
  //
  // Throws: system_error if unable to start the new thread.
  // Error conditions:
  // resource_unavailable_try_again - the system lacked the necessary resources to create another thread,
  // or the system-imposed limit on the number of threads in a process would be exceeded.
  //
  // Unfortunately, this is extremely hard to test portably so we don't have a test for this error condition right now.
#endif
  return 0;
}
