/*
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file thrust/system/hip/interpose_allocations.hpp
 *  \brief Interposed allocations/deallocations implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)
#if defined(__HIPSTDPAR_INTERPOSE_ALLOC_V1__)
#include <hip/hip_runtime.h>

#if __has_include(<pthread.h>)
    #include <pthread.h>
    #define __HIPSTDPAR_INTERPOSE_ALLOC_HAS_STACK_ACCESS__
#endif
#if __has_include(<sys/mman.h>)
    #include <sys/mman.h>
#endif
#if __has_include(<sys/unistd.h>)
    #include <sys/unistd.h>
#endif

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>

extern "C" {
    __attribute__((weak)) void __hipstdpar_hidden_free(void*);
    __attribute__((weak)) void* __hipstdpar_hidden_memalign(::std::size_t,
                                                            ::std::size_t);
    #if defined(_POSIX_MAPPED_FILES)
        #define __HIPSTDPAR_INTERPOSE_ALLOC_CAN_MMAP__
        __attribute__((weak))
        void* __hipstdpar_hidden_mmap(
            void*, ::std::size_t, int, int, int, ::off_t) noexcept;
        __attribute__((weak))
        int __hipstdpar_hidden_munmap(void*, ::std::size_t) noexcept;
    #endif // _POSIX_MAPPED_FILES
}

namespace hipstd
{
inline static const bool __initialised{hipInit(0) == hipSuccess};

#if defined(__HIPSTDPAR_INTERPOSE_ALLOC_HAS_STACK_ACCESS__)
    class Stack_accessor final {
        // DATA
        ::std::uint64_t* ps_{};
        ::std::size_t n_{};
        ::std::int32_t d_{};

        // IMPLEMENTATION - ACCESSORS
        bool touch_stack_() const
        {   // Due to how the kernel manages memory, we have to pre-access.
            ::std::uint64_t r{1};
            for (auto i = 0u; i != n_ / sizeof(*ps_); ++i) r += ps_[i];
            return r;
        }
    public:
        // CREATORS
        Stack_accessor()
        {
            pthread_attr_t t{};
            if (pthread_getattr_np(pthread_self(), &t)) {
                throw ::std::runtime_error("Failed to get thread attributes.");
            }
            if (pthread_attr_getstack(&t, reinterpret_cast<void**>(&ps_), &n_)) {
                throw ::std::runtime_error(
                    "Failed to get thread stack attributes.");
            }
            if (!ps_ || n_ == 0)
                return;
            if (hipGetDevice(&d_) != hipSuccess) {
                throw ::std::runtime_error(
                    "Failed to retrieve accelerator for HIPSTDPAR");
            }
            if (touch_stack_() &&
                hipMemAdvise(ps_, n_, hipMemAdviseSetAccessedBy, d_) != hipSuccess) {
                throw ::std::runtime_error(
                    "Failed to make thread stack accessible.");
            }
        }
        ~Stack_accessor()
        {
            if (!ps_ || n_ == 0) return;
            if (hipMemAdvise(ps_, n_, hipMemAdviseUnsetAccessedBy, d_) != hipSuccess) {
                ::std::cerr << "Failed to unset thread stack accessibility." <<
                    ::std::endl;
            }
        }
    };
    inline Stack_accessor __main_stack_accessor{};
    inline thread_local Stack_accessor __thread_stack_accessor{};
#endif // __HIPSTDPAR_INTERPOSE_ALLOC_HAS_STACK_ACCESS__
} // Namespace hipstd.

extern "C" {
    inline __attribute__((used)) void* __hipstdpar_aligned_alloc(std::size_t a,
                                                                 std::size_t n)
    {
        auto r = __hipstdpar_hidden_memalign(a, n);

        if (!hipstd::__initialised) return r;

        hipDevice_t d{};
        hipGetDevice(&d);

        if (hipMemAdvise(r, n, hipMemAdviseSetAccessedBy, d) != hipSuccess)
            return nullptr;

        return r;
    }

    inline __attribute__((used)) void* __hipstdpar_malloc(std::size_t n)
    {
        return __hipstdpar_aligned_alloc(alignof(std::max_align_t), n);
    }

    inline __attribute__((used)) void* __hipstdpar_calloc(std::size_t n,
                                                          std::size_t sz)
    {
        return ::std::memset(__hipstdpar_malloc(n * sz), 0, n * sz);
    }

    inline __attribute__((used))
    int __hipstdpar_posix_aligned_alloc(void** p, std::size_t a, std::size_t n)
    {   // TODO: check invariants on alignment
        if (!p || n == 0) return 0;

        *p = __hipstdpar_aligned_alloc(a, n);

        return 1;
    }

    inline __attribute__((used)) void __hipstdpar_free(void* p)
    {
        if (hipstd::__initialised) {
            hipDevice_t d{};
            hipGetDevice(&d);

            // Even if this fails there isn't much to do.
            hipMemAdvise(p, UINT64_MAX, hipMemAdviseUnsetAccessedBy, d);
        }
        return __hipstdpar_hidden_free(p);
    }

    inline __attribute__((used)) void* __hipstdpar_realloc(void* p,
                                                           std::size_t n)
    {
        auto q = std::memcpy(__hipstdpar_malloc(n), p, n);
        __hipstdpar_free(p);

        return q;
    }

    inline __attribute__((used))
    void* __hipstdpar_realloc_array(void* p, std::size_t n, std::size_t sz)
    {   // TODO: handle overflow in n * sz gracefully, as per spec.
        return __hipstdpar_realloc(p, n * sz);
    }

    inline __attribute__((used))
    void* __hipstdpar_operator_new_aligned(std::size_t n, std::size_t a)
    {
        if (auto p = __hipstdpar_aligned_alloc(a, n)) return p;

        throw std::runtime_error{"Failed __hipstdpar_operator_new_aligned"};
    }

    inline __attribute__((used)) void* __hipstdpar_operator_new(std::size_t n)
    {   // TODO: consider adding the special handling for operator new
        return __hipstdpar_operator_new_aligned(n, alignof(std::max_align_t));
    }

    inline __attribute__((used)) void* __hipstdpar_operator_new_nothrow(
        std::size_t n, std::nothrow_t) noexcept
    {
        try {
            return __hipstdpar_operator_new(n);
        }
        catch (...) {
            // TODO: handle the potential exception
        }
    }

    inline __attribute__((used)) void* __hipstdpar_operator_new_aligned_nothrow(
        std::size_t n, std::size_t a, std::nothrow_t) noexcept
    {   // TODO: consider adding the special handling for operator new
        try {
            return __hipstdpar_operator_new_aligned(n, a);
        }
        catch (...) {
            // TODO: handle the potential exception.
        }
    }

    inline __attribute__((used)) void __hipstdpar_operator_delete_aligned_sized(
        void* p, std::size_t, std::size_t) noexcept
    {
        return __hipstdpar_free(p);
    }

    inline __attribute__((used))
    void __hipstdpar_operator_delete(void* p) noexcept
    {
        return __hipstdpar_free(p);
    }

    inline __attribute__((used))
    void __hipstdpar_operator_delete_aligned(void* p, std::size_t) noexcept
    {
        return __hipstdpar_free(p);
    }

    inline __attribute__((used))
    void __hipstdpar_operator_delete_sized(void* p, std::size_t n) noexcept
    {
        return __hipstdpar_operator_delete_aligned_sized(
            p, n, alignof(std::max_align_t));
    }

    #if defined(__HIPSTDPAR_INTERPOSE_ALLOC_CAN_MMAP__)
        inline __attribute__((used))
        void* __hipstdpar_mmap(void* p, std::size_t n, int prot, int f, int fd,
                               off_t dx) noexcept
        {
            if (auto r = __hipstdpar_hidden_mmap(p, n, prot, f, fd, dx)) {
                if (!hipstd::__initialised) return r;

                hipDevice_t d{};
                hipGetDevice(&d);

                if (hipMemAdvise(r, n, hipMemAdviseSetAccessedBy, d) != hipSuccess)
                    return nullptr;

                return r;
            }
            return nullptr;
        }

        inline __attribute__((used))
        int __hipstdpar_munmap(void* p, std::size_t n) noexcept
        {
            if (hipstd::__initialised) {
                hipDevice_t d{};
                hipGetDevice(&d);

                if (hipMemAdvise(p, n, hipMemAdviseUnsetAccessedBy, d) != hipSuccess)
                    return -1;
            }
            return __hipstdpar_hidden_munmap(p, n);
        }
    #endif // __HIPSTDPAR_INTERPOSE_ALLOC_CAN_MMAP__
} // extern "C"
#  else // __HIPSTDPAR_INTERPOSE_ALLOC_V1__
#    error "__HIPSTDPAR_INTERPOSE_ALLOC_V1__ should be defined. Please use the '--hipstdpar-interpose-alloc' compile option."
#  endif // __HIPSTDPAR_INTERPOSE_ALLOC_V1__

#else // __HIPSTDPAR__
#    error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
