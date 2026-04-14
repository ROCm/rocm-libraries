// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef MIOPEN_GUARD_MLOPEN_OP_KERNEL_ARGS_HPP
#define MIOPEN_GUARD_MLOPEN_OP_KERNEL_ARGS_HPP

#include <cstdint>
#include <type_traits>
#include <vector>

#include <half/half.hpp>
#include <miopen/bfloat16.hpp>

struct OpKernelArg
{

    OpKernelArg(char val, size_t sz) : buffer(sz) { std::fill(buffer.begin(), buffer.end(), val); }

    template <typename T>
    OpKernelArg(T arg) : buffer(sizeof(T))
    {
        static_assert(std::is_trivial<T>{} || std::is_same<T, half_float::half>{} ||
                          std::is_same<T, bfloat16>{},
                      "Only for trivial types");
        *(reinterpret_cast<T*>(buffer.data())) = arg;
    }

    template <typename T>
    OpKernelArg(T* arg) // NOLINT
        : buffer(sizeof(T*))
    {
        *(reinterpret_cast<T**>(buffer.data())) = arg;
        is_ptr                                  = true;
    }

    std::size_t size() const { return buffer.size(); };
    std::vector<char> buffer{8};
    bool is_ptr = false;
};

#endif
