// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: types — TensorName, PhysicalTensor. consteval, no runtime, no CK deps.
//
// Physical tensor types for kernel descriptors.
//
// A PhysicalTensor maps a named tensor from the signature graph to a slot
// in the generic Args struct. Intermediate tensors (register-only, never
// in device memory) are NOT in the physical tensor table.
//
// TensorName is a fixed-capacity structural string (char[16]) that works
// as an NTTP member. All comparisons happen at compile time — zero runtime cost.

#pragma once

#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/layout.hpp>

#include <array>
#include <string_view>

namespace rocm_ck {

/// Maximum number of physical tensors in a kernel descriptor.
/// 8 covers GEMM (A, B, output, D0, D1) and FMHA (Q, K, V, O, S, mask, dropout).
inline constexpr int kMaxPhysicalTensors = 8;

/// Fixed-capacity string for tensor names in structural (NTTP) types.
/// 16 chars covers all practical names ("A", "B", "bias", "output", etc.).
struct TensorName
{
    char data[16]{};
    int len = 0;

    consteval TensorName() = default;

    consteval TensorName(std::string_view sv) : len(static_cast<int>(sv.size()))
    {
        if(sv.size() > 15)
            throw "tensor name too long (max 15 chars)";
        for(int i = 0; i < len; ++i)
            data[i] = sv[i];
    }

    consteval bool operator==(std::string_view sv) const
    {
        if(len != static_cast<int>(sv.size()))
            return false;
        for(int i = 0; i < len; ++i)
            if(data[i] != sv[i])
                return false;
        return true;
    }

    consteval auto operator<=>(const TensorName&) const = default;
};

/// A physical tensor slot in the kernel's Args layout.
/// Maps a user-facing name (from the Signature graph) to an Args slot index.
struct PhysicalTensor
{
    TensorName name;
    DataType dtype = DataType::FP32;
    Layout layout  = Layout::Row;
    int args_slot  = 0;
};

} // namespace rocm_ck
