// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_TESTS_CODEGEN_ARITHDEVICETEST_HPP
#define ROCM_TENSILELITE_TESTS_CODEGEN_ARITHDEVICETEST_HPP

#include <tensilelite/codegen/IRBuilder.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace rocm::tensilelite::codegen::test {

/// Where an input of an elementwise device test lives.
enum class InputKind {
  /// Loaded per lane into a VGPR from in[globalId].
  lane,
  /// Loaded per workgroup into an SGPR from in[workgroupId] with s_load.
  uniform,
};

/// Emits the code under test: computes dst from inputs, given in the order and
/// kinds of ArithKernelSpec::inputs.
using ArithEmitter =
    std::function<void(IRBuilder &builder, RegisterRef dst,
                       const std::vector<RegisterRef> &inputs)>;

struct ArithKernelSpec {
  std::vector<InputKind> inputs;
  /// vgpr: every lane stores its result to out[globalId]. sgpr: the result is
  /// uniform and lane 0 stores it to out[workgroupId].
  RegisterKind result;
  ArithEmitter emit;
};

/// Lanes per workgroup: one wave64.
constexpr int arithWorkgroupSize = 64;

/// Returns the kernel's assembly.
std::string buildArithAssembly(const ArithKernelSpec &spec);

/// Assembles and launches workgroupCount workgroups and compares out exactly
/// with expected. A lane input holds workgroupCount * arithWorkgroupSize words,
/// a uniform input workgroupCount words; expected has one word per lane (vgpr
/// result) or per workgroup (sgpr result). Reports up to 8 mismatches with
/// their inputs, prints the assembly on any failure, and fails fatally on HIP
/// errors.
void expectArithResults(const ArithKernelSpec &spec, int workgroupCount,
                        const std::vector<std::vector<uint32_t>> &inputs,
                        const std::vector<uint32_t> &expected);

/// Edge values: 0, 1, -1, 2, -2, 64, 65, -16, -17, INT32_MIN, INT32_MIN + 1,
/// INT32_MAX, INT32_MAX - 1, 0xFFFF, 0x10000, 0x12345678, 0x9E3779B9 (products
/// and sums of these wrap).
std::vector<uint32_t> arithEdgeValues();

struct WordPairs {
  std::vector<uint32_t> a;
  std::vector<uint32_t> b;
};

/// The cross product of the edge values followed by randomCount pairs from
/// std::mt19937 with a fixed seed (raw engine output, which is portable; not a
/// distribution), padded with more random pairs to a multiple of
/// arithWorkgroupSize.
WordPairs makeArithPairs(int randomCount);

} // namespace rocm::tensilelite::codegen::test

#endif // ROCM_TENSILELITE_TESTS_CODEGEN_ARITHDEVICETEST_HPP
