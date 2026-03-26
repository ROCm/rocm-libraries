// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: meta — operator structs, Op variant. No runtime, no CK deps.
//
// Operator structs for the rocm_ck signature compute graph.
//
// Each operator is a typed struct with named tensor slots. Operators
// form the edges of the directed compute graph; tensors are nodes.
// The Op variant holds any operator type; std::monostate marks empty slots.

#pragma once

#include <rocm_ck/datatype_utils.hpp>

#include <string_view>
#include <variant>

namespace rocm_ck {

/// Matrix multiplication: out = lhs x rhs.
/// All tensor labels must be explicit — no implicit defaults.
/// acc_dtype defaults to FP32 (universal accumulation convention).
struct GemmOp
{
    std::string_view lhs;
    std::string_view rhs;
    std::string_view out;
    DataType acc_dtype = DataType::FP32;
};

/// Element-wise addition: out = lhs + rhs.
struct AddOp
{
    std::string_view lhs;
    std::string_view rhs;
    std::string_view out;
};

/// Element-wise multiplication: out = lhs * rhs.
struct MulOp
{
    std::string_view lhs;
    std::string_view rhs;
    std::string_view out;
};

/// ReLU activation: out = max(0, in).
struct ReluOp
{
    std::string_view in;
    std::string_view out;
};

/// Approximate GELU: out = in * sigmoid(1.702 * in).
struct FastGeluOp
{
    std::string_view in;
    std::string_view out;
};

/// Exact GELU: out = 0.5 * in * (1 + erf(in / sqrt(2))).
struct GeluOp
{
    std::string_view in;
    std::string_view out;
};

/// SiLU (Swish): out = in * sigmoid(in).
struct SiluOp
{
    std::string_view in;
    std::string_view out;
};

/// Sigmoid: out = 1 / (1 + exp(-in)).
struct SigmoidOp
{
    std::string_view in;
    std::string_view out;
};

/// Softmax along the last dimension (CK Tile convention).
/// For rank-2 input [M, N], reduces over N. This matches the standard
/// attention pattern where S[batch, seq] is softmax'd over seq_len.
struct SoftmaxOp
{
    std::string_view in;
    std::string_view out;
};

/// Scalar scaling: out = scale * in.
/// The 'scale' field names a Scalar parameter in the Signature.
struct ScaleOp
{
    std::string_view in;
    std::string_view out;
    std::string_view scale;
};

/// FMHA backward dQ/dK/dV: fused 5-GEMM backward attention kernel.
///
/// Names input/output tensors without decomposing the internal compute
/// graph. Feature flags (mask, dropout, bias, deterministic) belong in
/// the Algorithm, not in the op.
struct FmhaBwdOp
{
    // Inputs
    std::string_view q;   // query           [seqlen_q, hdim_q]
    std::string_view k;   // key             [seqlen_k, hdim_q]
    std::string_view v;   // value           [seqlen_k, hdim_v]
    std::string_view lse; // log-sum-exp     [seqlen_q] (from forward)
    std::string_view do_; // output gradient [seqlen_q, hdim_v]
    std::string_view d;   // OGradDotO       [seqlen_q]

    // Outputs
    std::string_view dq; // query gradient  [seqlen_q, hdim_q]
    std::string_view dk; // key gradient    [seqlen_k, hdim_q]
    std::string_view dv; // value gradient  [seqlen_k, hdim_v]

    DataType acc_dtype = DataType::FP32;
};

/// Variant of all operator types. std::monostate marks empty slots.
using Op = std::variant<std::monostate,
                        GemmOp,
                        AddOp,
                        MulOp,
                        ReluOp,
                        FastGeluOp,
                        GeluOp,
                        SiluOp,
                        SigmoidOp,
                        SoftmaxOp,
                        ScaleOp,
                        FmhaBwdOp>;

} // namespace rocm_ck
