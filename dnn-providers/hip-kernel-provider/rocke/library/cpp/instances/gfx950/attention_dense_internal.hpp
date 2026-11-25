// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ROCKE_GFX950_DENSE_BUILD_INTERNAL_HPP
#define ROCKE_GFX950_DENSE_BUILD_INTERNAL_HPP

#include "rocke/instance_gfx950_attention_dense.h"
#include <array>
#include <span>
#include <tuple>
#include <utility>

namespace rocke_dense_gfx950
{
using V = rocke_value_t*;
using Builder = rocke_ir_builder_t;
using Spec = rocke_gfx950_attention_dense_spec_t;
using Type = const rocke_type_t*;
inline constexpr double log2e = 1.4426950408889634;

inline V pointer_param(Builder* b, const char* name, Type type, bool writing, int align)
{
    rocke_param_opts_t opts{};
    opts.noalias = opts.noalias_set = true;
    if(writing)
        opts.writeonly = opts.writeonly_set = true;
    else
        opts.readonly = opts.readonly_set = true;
    opts.align_set = true;
    opts.align = align;
    return rocke_b_param(b, name, rocke_ptr_type(b, type, "global"), &opts);
}

// A braced tuple evaluates builder operands from left to right, like Python.
// Nested ordinary function arguments have unspecified evaluation order in C++.
#define DENSE_OP(op, ...) std::apply(rocke_b_##op, std::tuple{b __VA_OPT__(, ) __VA_ARGS__})

// Unset extents are emitted as fresh constants at each use, as in DecodeShape.
struct FastDivPair
{
    V magic = nullptr, shift = nullptr;
};
struct DecodeRuntime
{
    V batch = nullptr, seqlen_q = nullptr, seqlen_kv = nullptr;
    V num_query_heads = nullptr, num_kv_heads = nullptr, gqa = nullptr;
    V nqb = nullptr, w = nullptr;
    FastDivPair batch_div, heads_div, gqa_div, nqb_div;
};
struct Decoded
{
    V qb, hq, bt, hkv;
};
Decoded decode_grid(Builder* b, const Spec& s, const DecodeRuntime& rt, V bx, V by, V bz);
Decoded decode_persistent(Builder* b, const Spec& s, const DecodeRuntime& rt, V wi);
V decode_hkv(Builder* b, const Spec& s, const DecodeRuntime& rt, V hq);

// Shared output sequences in kernels/gfx950/attention_dense.py.
void dense_lse_output(Builder* b,
                      V dst,
                      V index,
                      V m,
                      V l,
                      V row_exists,
                      V row_has_key,
                      V lane_half,
                      V sinks,
                      V head,
                      Type dtype);
void emit_o_store(Builder* b,
                  V o,
                  V q_row_byte,
                  V d_half,
                  std::span<V const> o_acc,
                  V rcp_l,
                  Type dtype,
                  int width,
                  V row_has_key);
std::array<V, 3> segment_params(Builder* b);
std::pair<V, V> segment_tile_range(Builder* b, V seg, V lo, V hi, int segments);
void emit_segment_store(Builder* b,
                        const std::array<V, 3>& params,
                        V token,
                        V head,
                        V num_heads,
                        V seg,
                        int segments,
                        int head_size,
                        std::span<V const> o_acc,
                        V m,
                        V denom,
                        V lane_h);

// Called only after concrete admission, on a builder initialized to the spec name.
void build_grid(Builder* b, const Spec& s);
void build_persistent(Builder* b, const Spec& s);
} // namespace rocke_dense_gfx950
#endif
