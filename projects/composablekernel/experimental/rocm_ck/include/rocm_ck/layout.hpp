// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: types — Layout enum, constexpr/consteval helpers. No runtime, no CK deps.
//
// Memory layout enum for matrix operations.

#pragma once

namespace rocm_ck {

/// Memory layout for tensor operands.
/// Contiguous = rank-1: single dimension, always stride 1.
/// Row = rank-2: row-major (last dim contiguous, strides = [K, 1]).
/// Col = rank-2: column-major (first dim contiguous, strides = [1, M]).
/// Auto = inherit from operator slot or explicit tensor entry (resolve-time only).
/// Future: NHWC, NCHW for rank-4 conv/FMHA tensors.
enum class Layout
{
    Contiguous,
    Row,
    Col,
    Auto
};

/// Returns a short string name for the layout (e.g. "Row").
constexpr const char* layout_name(Layout ly)
{
    switch(ly)
    {
    case Layout::Contiguous: return "Contiguous";
    case Layout::Row: return "Row";
    case Layout::Col: return "Col";
    case Layout::Auto: return "Auto";
    }
    return "???";
}

/// Check if a layout is valid for a given tensor rank.
/// Contiguous is for rank-1, Row/Col are for rank-2.
consteval bool is_valid_layout_for_rank(Layout layout, int rank)
{
    switch(layout)
    {
    case Layout::Contiguous: return rank == 1;
    case Layout::Row: return rank == 2;
    case Layout::Col: return rank == 2;
    case Layout::Auto: return false; // Auto is unresolved, never valid for concrete tensors
    }
    return false;
}

// --- is_valid_layout_for_rank compile-time tests ---
// clang-format off
static_assert( is_valid_layout_for_rank(Layout::Contiguous, 1));
static_assert(!is_valid_layout_for_rank(Layout::Contiguous, 2));
static_assert(!is_valid_layout_for_rank(Layout::Row, 1));
static_assert( is_valid_layout_for_rank(Layout::Row, 2));
static_assert(!is_valid_layout_for_rank(Layout::Col, 1));
static_assert( is_valid_layout_for_rank(Layout::Col, 2));
static_assert(!is_valid_layout_for_rank(Layout::Auto, 1));
static_assert(!is_valid_layout_for_rank(Layout::Auto, 2));
// clang-format on

} // namespace rocm_ck
