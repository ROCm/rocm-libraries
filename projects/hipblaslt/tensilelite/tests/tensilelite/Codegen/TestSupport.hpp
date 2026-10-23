// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_TESTS_CODEGEN_TESTSUPPORT_HPP
#define ROCM_TENSILELITE_TESTS_CODEGEN_TESTSUPPORT_HPP

#include <tensilelite/Codegen/IRBuilder.hpp>
#include <tensilelite/Codegen/Kernel.hpp>

#include "code.hpp"
#include "container.hpp"

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#define ASSERT_HIP_SUCCESS(expr)                                               \
  do {                                                                         \
    hipError_t status = (expr);                                                \
    ASSERT_EQ(status, hipSuccess)                                              \
        << #expr << ": " << hipGetErrorString(status);                         \
  } while (0)

namespace rocm::tensilelite::codegen::test {

// Text helpers.

inline std::string text(RegisterRef ref) { return ref.container()->toString(); }

inline std::string text(const RegisterRange &range) {
  return text(range.ref());
}

/// Builds "mnemonic dst, src" as rocisa prints it.
inline std::string line(std::string_view mnemonic, const std::string &dst,
                        const std::string &src) {
  return std::string(mnemonic) + " " + dst + ", " + src;
}

/// Builds "mnemonic dst, src0, src1" as rocisa prints it.
inline std::string line(std::string_view mnemonic, const std::string &dst,
                        const std::string &src0, const std::string &src1) {
  return line(mnemonic, dst, src0) + ", " + src1;
}

inline int32_t countLinesContaining(const std::string &text,
                                    std::string_view needle) {
  int32_t count = 0;
  std::istringstream stream(text);
  for (std::string current; std::getline(stream, current);)
    if (current.find(needle) != std::string::npos)
      ++count;
  return count;
}

/// Checks each needle against the lines of text in order, so tests do not
/// depend on rocisa's padding and comment formatting.
void expectLinesInOrder(const std::string &text,
                        std::vector<std::string_view> needles);

// Host text tests.

using BuilderOp = void (IRBuilder::*)(RegisterRef, Operand, Operand);

/// A target, allocator and builder over a body module, for host text tests.
class BuilderTest : public testing::Test {
protected:
  // Ranges a test creates are destroyed before the builder and the allocator.
  Target target = Target::get("gfx950");
  RegisterAllocator allocator{target};
  rocisa::Module body{"body"};
  IRBuilder builder{target, allocator, body};

  /// Runs emit on a builder over a fresh module, so each call is checked in
  /// isolation, and returns the module's text.
  template <typename Emit>
  std::string emitted(Emit emit) {
    rocisa::Module module{"module"};
    IRBuilder local{target, allocator, module};
    emit(local);
    return module.toString();
  }

  /// Emits op(dst, a, b) through emitted.
  std::string emitOp(BuilderOp op, RegisterRef dst, Operand a, Operand b) {
    return emitted([&](IRBuilder &local) { (local.*op)(dst, a, b); });
  }

  /// Asserts that op throws std::invalid_argument and that nothing was
  /// emitted through the fixture's builder.
  void expectRejected(const std::function<void()> &op) {
    EXPECT_THROW(op(), std::invalid_argument);
    EXPECT_EQ(body.toString(), "");
  }
};

// HIP helpers.

// Deleters cannot assert, so they ignore errors.
struct ModuleUnload {
  void operator()(hipModule_t module) const { (void)hipModuleUnload(module); }
};

using ModulePtr =
    std::unique_ptr<std::remove_pointer_t<hipModule_t>, ModuleUnload>;

struct DeviceFree {
  void operator()(void *ptr) const { (void)hipFree(ptr); }
};

using DevicePtr = std::unique_ptr<void, DeviceFree>;

inline bool hasHipDevice() {
  int32_t deviceCount = 0;
  return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
}

/// Skips the test when no HIP device is present.
class DeviceTest : public testing::Test {
protected:
  void SetUp() override {
    if (!hasHipDevice())
      GTEST_SKIP() << "no HIP device";
  }
};

// The helpers below fail fatally on HIP errors, so callers check
// testing::Test::HasFatalFailure() after them.

/// Loads hsaco and looks up the kernel name.
void loadKernel(const std::vector<uint8_t> &hsaco, const char *name,
                ModulePtr &module, hipFunction_t &function);

/// Allocates bytes on the device and fills them with a copy of host, or with
/// 0xff bytes when host is null.
void uploadBytes(const void *host, size_t bytes, DevicePtr &buffer);

/// Copies bytes from the device buffer to host.
void downloadBytes(const DevicePtr &buffer, void *host, size_t bytes);

template <typename T>
void upload(const std::vector<T> &host, DevicePtr &buffer) {
  uploadBytes(host.data(), host.size() * sizeof(T), buffer);
}

template <typename T>
void download(const DevicePtr &buffer, std::vector<T> &host) {
  downloadBytes(buffer, host.data(), host.size() * sizeof(T));
}

/// Launches function with a packed kernarg buffer and waits for it.
void launchWithKernarg(hipFunction_t function, dim3 grid, dim3 block,
                       std::vector<std::byte> &kernarg);

/// Launches function with kernel parameters and waits for it, so the runtime
/// fills the hidden arguments.
void launchWithParams(hipFunction_t function, dim3 grid, dim3 block,
                      void **params);

/// Stores value at offset in the kernarg buffer.
template <typename T>
void writeArgument(std::vector<std::byte> &kernarg, int32_t offset, T value) {
  std::memcpy(kernarg.data() + offset, &value, sizeof(T));
}

/// Computes workgroupId.x * workgroupSize + workitemId.x into a new VGPR.
RegisterRange computeGlobalId(Kernel &kernel);

// Elementwise device tests.

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
constexpr int32_t arithWorkgroupSize = 64;

/// Assembles and launches workgroupCount workgroups and compares out exactly
/// with expected. A lane input holds workgroupCount * arithWorkgroupSize words,
/// a uniform input workgroupCount words; expected has one word per lane (vgpr
/// result) or per workgroup (sgpr result). Reports up to 8 mismatches with
/// their inputs, prints the assembly on any failure, and fails fatally on HIP
/// errors.
void expectArithResults(const ArithKernelSpec &spec, int32_t workgroupCount,
                        const std::vector<std::vector<uint32_t>> &inputs,
                        const std::vector<uint32_t> &expected);

using Words = std::vector<uint32_t>;
using Values = std::vector<uint64_t>;

inline Values widen(const Words &words) {
  return Values(words.begin(), words.end());
}

inline Values join(const Words &low, const Words &high) {
  Values values;
  for (int64_t i = 0; i < std::ssize(low); ++i)
    values.push_back(low[i] | uint64_t{high[i]} << 32);
  return values;
}

inline Words lowWords(const Values &values) {
  Words words;
  for (uint64_t value : values)
    words.push_back(static_cast<uint32_t>(value));
  return words;
}

inline Words highWords(const Values &values) {
  Words words;
  for (uint64_t value : values)
    words.push_back(static_cast<uint32_t>(value >> 32));
  return words;
}

/// Repeats each workgroup's value for every lane of the workgroup.
template <typename T>
std::vector<T> perLane(const std::vector<T> &perGroup) {
  std::vector<T> lanes;
  for (const T &value : perGroup)
    lanes.insert(lanes.end(), arithWorkgroupSize, value);
  return lanes;
}

/// The number of workgroups that cover laneWords.
inline int32_t laneGroupCount(const Words &laneWords) {
  return static_cast<int32_t>(laneWords.size()) / arithWorkgroupSize;
}

/// Edge values: 0, 1, -1, 2, -2, 64, 65, -16, -17, INT32_MIN, INT32_MIN + 1,
/// INT32_MAX, INT32_MAX - 1, 0xFFFF, 0x10000, 0x12345678, 0x9E3779B9 (products
/// and sums of these wrap).
Words arithEdgeValues();

struct WordPairs {
  Words a;
  Words b;
};

/// The cross product of the edge values followed by randomCount pairs from
/// std::mt19937 with a fixed seed (raw engine output, which is portable; not a
/// distribution), padded with more random pairs to a multiple of
/// arithWorkgroupSize.
WordPairs makeArithPairs(int32_t randomCount);

/// The 64-bit operand pairs a and b as their low and high 32-bit words.
struct WordQuads {
  Words aLo, aHi, bLo, bHi;
};

/// The cross product of 64-bit edge values (0, 1, -1, 2, INT64_MIN, INT64_MAX,
/// values around the 32-bit boundaries and values that share a high word but
/// differ in the sign bit of the low word) followed by randomCount pairs from
/// std::mt19937 with a fixed seed (raw engine output), padded with more random
/// pairs to a multiple of arithWorkgroupSize.
WordQuads makeArithPairs64(int32_t randomCount);

/// A register pair under test and the registers that keep it allocated.
struct PairRegisters {
  RegisterRange range;
  RegisterRef pair;
};

/// Allocates registers of lo's kind and copies lo and hi into a pair at an
/// even index, or at an odd index (three registers are allocated and the pair
/// sits at offset 1) when misaligned. The pair is valid while range lives.
PairRegisters makePair(IRBuilder &builder, RegisterRef lo, RegisterRef hi,
                       bool misaligned = false);

} // namespace rocm::tensilelite::codegen::test

#endif // ROCM_TENSILELITE_TESTS_CODEGEN_TESTSUPPORT_HPP
