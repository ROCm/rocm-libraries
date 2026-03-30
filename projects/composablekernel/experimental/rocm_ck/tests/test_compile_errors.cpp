// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Expected compilation failure cases for rocm_ck schema types.
//
// These cases are documented here for reference. Each is also tested as an
// actual compile-error test in compile_fail/*.cpp — verified via CMake
// WILL_FAIL (ctest -R compile_fail).
//
// See tests/README.md for the testing strategy.

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

// ============================================================================
// resolve() — expected failures
// ============================================================================

// No dtype set (neither on Signature nor individual tensors):
//
//   constexpr auto bad = resolve(Signature{
//       .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});
//
// Expected error: "tensor dtype unresolvable: set tensor dtype or signature dtype"

// SSA violation — two ops output the same tensor name:
//
//   constexpr auto bad = resolve(Signature{
//       .dtype = DataType::FP16,
//       .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
//               AddOp{.lhs = "X", .rhs = "Y", .out = "C"}}});
//
// Expected error: "SSA violation: tensor produced by multiple operators"

// Empty tensor name in operator slot:
//
//   constexpr auto bad = resolve(Signature{
//       .dtype = DataType::FP16,
//       .ops = {GemmOp{.lhs = "", .rhs = "B", .out = "C"}}});
//
// Expected error: "operator slot has empty tensor name"

// Tensor entry with metadata but no name:
//
//   constexpr auto bad = resolve(Signature{
//       .dtype = DataType::FP16,
//       .tensors = {Tensor{.dtype = DataType::FP32}},
//       .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});
//
// Expected error: "Tensor entry has metadata but no name"

// Duplicate scalar names:
//
//   constexpr auto bad = resolve(Signature{
//       .dtype = DataType::FP16,
//       .scalars = {Scalar{.name = "alpha"}, Scalar{.name = "alpha"}},
//       .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});
//
// Expected error: "duplicate scalar name in Signature"

// ScaleOp references undeclared scalar:
//
//   constexpr auto bad = resolve(Signature{
//       .dtype = DataType::FP16,
//       .ops = {ScaleOp{.in = "X", .out = "Y", .scale = "missing"}}});
//
// Expected error: "ScaleOp.scale references undeclared Scalar"

// ============================================================================
// make_spec() — expected failures
// ============================================================================

// First op is not GemmOp:
//
//   constexpr auto bad = make_spec(
//       Signature{.dtype = DataType::FP16,
//                 .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
//       GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});
//
// Expected error: "GEMM make_spec requires GemmOp as first operator"

// Invalid MFMA tile for dtype:
//
//   constexpr auto bad = make_spec(
//       Signature{.dtype = DataType::FP32,
//                 .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
//       GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {32, 32, 16}});
//
// Expected error: "mfma_tile is not a valid MFMA instruction shape for this dtype"
// (FP32 32x32 only supports k=4 or k=8, not k=16)

// block_waves.k != 1:
//
//   constexpr auto bad = make_spec(
//       Signature{.dtype = DataType::FP16,
//                 .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
//       GemmAlgorithm{{128, 128, 32}, {2, 2, 2}, {16, 16, 16}});
//
// Expected error: "block_waves.k must be 1 (CShuffleEpilogue constraint)"

// Block tile not divisible by block_waves * mfma_tile:
//
//   constexpr auto bad = make_spec(
//       Signature{.dtype = DataType::FP16,
//                 .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
//       GemmAlgorithm{{100, 128, 32}, {2, 2, 1}, {16, 16, 16}});
//
// Expected error: "block_tile.m must be divisible by (block_waves.m * mfma_tile.m)"

// ============================================================================
// TensorName — expected failures
// ============================================================================

// Name too long (> 15 chars):
//
//   constexpr TensorName bad("1234567890123456");
//
// Expected error: "tensor name too long (max 15 chars)"
