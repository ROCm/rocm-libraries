#include "race-emulator/Emulator.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef TEST_KERNEL_DIR
#define TEST_KERNEL_DIR "undefined"
#endif

namespace {

using namespace raceemulator;
namespace fs = std::filesystem;

// Helper utilities

std::string loadKernelFile(const std::string &filename) {
  fs::path filepath = fs::path(TEST_KERNEL_DIR) / filename;
  std::ifstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open kernel file: " +
                             filepath.string());
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

struct GemmDims {
  int M, N, K;
  int BatchCount = 1;
};

struct TensileKernelArgs {
  std::vector<uint32_t> preamble;
  std::vector<uint32_t> metadata;
  float alpha;
  float beta;
};

// BF16 conversion utilities

namespace bf16Conversion {

// Union for bit manipulation without strict aliasing violations
union FloatBits {
  float f;
  uint32_t u;
};

// Convert float -> bf16 (truncate)
// Representation: bf16 is just the upper 16 bits of an f32
uint16_t floatToBf16(float value) {
  FloatBits bits;
  bits.f = value;
  // Simple truncation is the standard approach for raw bitwise conversion tests
  // (Round-to-nearest-even is used in HW, but truncation is fine for setting up
  // inputs)
  return static_cast<uint16_t>(bits.u >> 16);
}

// Convert bf16 -> float
float bf16ToFloat(uint16_t value) {
  FloatBits bits;
  bits.u = static_cast<uint32_t>(value) << 16;
  return bits.f;
}

} // namespace bf16Conversion

// CPU reference implementation

// Always calculates in float (F32)
void cpuGemmReferenceF32(const GemmDims &dims, const std::vector<float> &h_a,
                         const std::vector<float> &h_b,
                         const std::vector<float> &h_c,
                         std::vector<float> &h_d_expected, float alpha,
                         float beta) {

  // Resize output to match dimensions
  h_d_expected = h_c; // Initialize with C for beta accumulation

  int strideA = dims.M * dims.K;
  int strideB = dims.K * dims.N;
  int strideC = dims.M * dims.N;

  for (int b = 0; b < dims.BatchCount; ++b) {
    int batchOffsetA = b * strideA;
    int batchOffsetB = b * strideB;
    int batchOffsetC = b * strideC;

    for (int n = 0; n < dims.N; ++n) {
      for (int m = 0; m < dims.M; ++m) {
        float sum = 0.0f;
        for (int k = 0; k < dims.K; ++k) {
          // Column Major: A[m, k], B[k, n]
          float valA = h_a[batchOffsetA + m + dims.M * k];
          float valB = h_b[batchOffsetB + k + dims.K * n];
          sum += valA * valB;
        }

        int idxC = batchOffsetC + m + dims.M * n;
        float valC = h_c[idxC];
        h_d_expected[idxC] = sum * alpha + valC * beta;
      }
    }
  }
}

// Test runner class

// KernelType: The data type used by the GPU (e.g., float or uint16_t for bf16)
template <typename KernelType> class TensileGemmRunner {
public:
  TensileGemmRunner(const std::string &kernel_file)
      : assembly_(loadKernelFile(kernel_file)) {}

  std::optional<std::string> run(const GemmDims &dims,
                                 const TensileKernelArgs &kArgs,
                                 int nWavesPerWorkgroup = 1,
                                 int nWorkgroups = 1) {
    // 1. Calculate Sizes
    size_t sizeA = dims.M * dims.K * dims.BatchCount;
    size_t sizeB = dims.N * dims.K * dims.BatchCount;
    size_t sizeC = dims.M * dims.N * dims.BatchCount;

    // 2. Setup Host Data (F32) - "Golden" Source
    std::vector<float> aF32(sizeA);
    std::vector<float> bF32(sizeB);
    std::vector<float> cF32(sizeC);

    initializeDataF32(aF32);
    initializeDataF32(bF32);
    std::fill(cF32.begin(), cF32.end(), 0.0f); // Zero C for simplicity

    // 3. Convert Host Data to Device Data (KernelType)
    // Your template parameter KernelType ensures aCpu/bCpu are converted
    // correctly here
    std::vector<KernelType> aGpu = convertToKernel(aF32);
    std::vector<KernelType> bGpu = convertToKernel(bF32);
    std::vector<KernelType> cGpu = convertToKernel(cF32);
    std::vector<KernelType> dGpu(sizeC, static_cast<KernelType>(0));

    // 4. Setup Emulator
    Emulator emulator = Emulator::createGfx942(assembly_);
    emulator.enableRaceChecks(false);

    int argIdx = 0;

    // A. Preamble (uint32 args)
    preambleStorage_ = kArgs.preamble;
    for (uint64_t i = 0; i < preambleStorage_.size(); ++i) {
      emulator.addKernarg(argIdx++, &preambleStorage_[i]);
    }

    // B. Pointers (to GPU typed data)
    KernelType *dPtr = dGpu.data();
    KernelType *cPtr = cGpu.data();
    KernelType *aPtr = aGpu.data();
    KernelType *bPtr = bGpu.data();

    // Debug Print
    printTensorDebug("d", dPtr, dGpu.size());
    printTensorDebug("c", cPtr, cGpu.size());
    printTensorDebug("a", aPtr, aGpu.size());
    printTensorDebug("b", bPtr, bGpu.size());

    emulator.addKernarg(argIdx++, &dPtr);
    emulator.addKernarg(argIdx++, &cPtr);
    emulator.addKernarg(argIdx++, &aPtr);
    emulator.addKernarg(argIdx++, &bPtr);

    // C. Metadata
    metadataStorage_ = kArgs.metadata;
    for (uint64_t i = 0; i < metadataStorage_.size(); ++i) {
      emulator.addKernarg(argIdx++, &metadataStorage_[i]);
    }

    // D. Scalars
    alphaStorage_ = kArgs.alpha;
    betaStorage_ = kArgs.beta;
    emulator.addKernarg(argIdx++, &alphaStorage_);
    emulator.addKernarg(argIdx++, &betaStorage_);

    // 5. Run Emulator
    for (int i = 0; i < nWorkgroups; ++i) {

      Dim3d wgId(i, 0, 0);
      Dim3d blockDim(nWavesPerWorkgroup * 64, 1, 1); // nWaves * 64 threads/wave
      emulator.run(wgId, blockDim);
    }

    // 6. Verify Results
    // A. Run Reference (F32 -> F32)
    std::vector<float> dRefF32;
    cpuGemmReferenceF32(dims, aF32, bF32, cF32, dRefF32, kArgs.alpha,
                        kArgs.beta);

    // B. Convert GPU output back to F32 for comparison
    std::vector<float> dGpuAsF32 = convertToHost(dGpu);
    return verifyResults(dGpuAsF32, dRefF32, dims);
  }

private:
  std::string assembly_;
  // Storage to keep arg pointers valid
  std::vector<uint32_t> preambleStorage_;
  std::vector<uint32_t> metadataStorage_;
  float alphaStorage_;
  float betaStorage_;

  void initializeDataF32(std::vector<float> &data) {
    static std::mt19937 rng(1013);
    std::uniform_int_distribution<int> dist(0, 1);
    std::vector<float> choices = {
        1.0f, -1.0f}; // Simple integer-ish floats to avoid rounding noise
    for (auto &val : data)
      val = choices[dist(rng)];
  }

  // Helper: Convert F32 vector to KernelType vector
  std::vector<KernelType> convertToKernel(const std::vector<float> &src) {
    std::vector<KernelType> dst(src.size());
    if constexpr (std::is_same_v<KernelType, float>) {
      dst = src;
    } else if constexpr (std::is_same_v<KernelType, uint16_t>) {
      // Assume uint16_t implies BF16 for this test context
      for (size_t i = 0; i < src.size(); ++i)
        dst[i] = bf16Conversion::floatToBf16(src[i]);
    } else {
      throw std::runtime_error("Unsupported KernelType");
    }
    return dst;
  }

  // Helper: Convert KernelType vector back to F32
  std::vector<float> convertToHost(const std::vector<KernelType> &src) {
    std::vector<float> dst(src.size());
    if constexpr (std::is_same_v<KernelType, float>) {
      dst = src;
    } else if constexpr (std::is_same_v<KernelType, uint16_t>) {
      for (size_t i = 0; i < src.size(); ++i)
        dst[i] = bf16Conversion::bf16ToFloat(src[i]);
    }
    return dst;
  }

  void printTensorDebug(const std::string &name, void *ptr, size_t count) {
    size_t bytes = count * sizeof(KernelType);
    std::cout << "Tensor " << name << " : [" << ptr << ", "
              << static_cast<void *>(reinterpret_cast<char *>(ptr) + bytes)
              << ")\n";
  }

  std::optional<std::string> verifyResults(const std::vector<float> &actual,
                                           const std::vector<float> &expected,
                                           const GemmDims &dims) {
    std::vector<std::pair<int, int>> mismatches;
    int limit = dims.M * dims.N * dims.BatchCount;

    // Set tolerance (Exact for F32->F32 with simple ints, looser for BF16)
    float tolerance = 0.0f;
    if constexpr (std::is_same_v<KernelType, uint16_t>) {
      tolerance = 1e-2f; // BF16 has less precision
    }

    for (int i = 0; i < limit; ++i) {
      // First check that actual is not NaN etc:
      if (std::isnan(actual[i]) || std::isinf(actual[i])) {
        mismatches.emplace_back(i, 0);
        continue;
      }

      float diff = std::abs(actual[i] - expected[i]);
      if (diff > tolerance) {
        mismatches.emplace_back(i, 0);
      }
    }

    if (!mismatches.empty()) {
      std::ostringstream oss;
      oss << "Mismatch in " << mismatches.size() << " elements.\n";
      size_t maxPrint = 10;
      for (size_t k = 0; k < mismatches.size() && k < maxPrint; ++k) {
        int idx = mismatches[k].first;
        oss << " At index " << idx << ": expected " << expected[idx] << ", got "
            << actual[idx] << "\n";
      }
      return oss.str();
    }

    return std::nullopt;
  }
};

} // namespace

// --- Tests ---

// clang-format off
//
//      OperationType: GEMM
//      DataType: s
//      DestDataType: s
//      ComputeDataType: s
//      TransposeA: False
//      TransposeB: False
//      UseBeta: True
//      Batched: True
//      F32XdlMathOp: x
//    - # BenchmarkProblemSizeGroup
//      InitialSolutionParameters:
//      BenchmarkCommonParameters:
//        - KernelLanguage: ["Assembly"]
//      ForkParameters:
//        - TransposeLDS: [0]
//        - MatrixInstruction:
//            - [16, 16, 8, 1, 1, 1, 1, 1, 1]
//        - ThreadTile:
//          - [ 4, 4 ]
//        - WorkGroup:
//          - [8, 8, 1 ]
//        - DepthU: [32]
//        - PrefetchLocalRead: [1]
//        - PrefetchGlobalRead: [True]
//        - WorkGroupMapping: [8]
//        - GlobalSplitU: [1]
//        - InnerUnroll: [2]
//        # - AssertSummationElementMultiple: [1, 2]
//      BenchmarkForkParameters:
//      JoinParameters:
//      BenchmarkJoinParameters:
//      BenchmarkFinalParameters:
//        - ProblemSizes:
//           - Exact: [16, 16, 1, 8]
// Appending argument 'gemm_count' of size 4 and value: 1 at offset 0 (bound=1)
// Appending argument 'internalArgs' of size 4 and value: 18874369 at offset 4 (bound=1)
// Appending argument 'internalArgs1' of size 4 and value: 1275133960 at offset 8 (bound=1)
// Appending argument 'numWorkGroups' of size 4 and value: 1 at offset 12 (bound=1)
// Appending argument '' of size 4 and value: 16 at offset 16 (bound=1)
// Appending argument '' of size 4 and value: 16 at offset 20 (bound=1)
// Appending argument '' of size 4 and value: 1 at offset 24 (bound=1)
// Appending argument '' of size 4 and value: 8 at offset 28 (bound=1)
// Appending argument 'd' of size 8 and value: 0x7cd488c0c000 at offset 32 (bound=1)
// Appending argument 'c' of size 8 and value: 0x7cd488c08000 at offset 40 (bound=1)
// Appending argument 'a' of size 8 and value: 0x7cd488c00000 at offset 48 (bound=1)
// Appending argument 'b' of size 8 and value: 0x7cd488c04000 at offset 56 (bound=1)
// Appending argument '' of size 4 and value: 16 at offset 64 (bound=1)
// Appending argument '' of size 4 and value: 256 at offset 68 (bound=1)
// Appending argument '' of size 4 and value: 16 at offset 72 (bound=1)
// Appending argument '' of size 4 and value: 256 at offset 76 (bound=1)
// Appending argument '' of size 4 and value: 16 at offset 80 (bound=1)
// Appending argument '' of size 4 and value: 128 at offset 84 (bound=1)
// Appending argument '' of size 4 and value: 8 at offset 88 (bound=1)
// Appending argument '' of size 4 and value: 128 at offset 92 (bound=1)
// Appending argument 'alpha' of size 4 and value: 2 at offset 96 (bound=1)
// Appending argument 'beta' of size 4 and value: 2 at offset 100 (bound=1)
//
// clang-format on

TEST(GpuEmulatorTestSuiteZero, MatMul_TensileLite_F32) {
  GemmDims dims{16, 16, 8, 1}; // M, N, K, Batch

  TensileKernelArgs args;
  args.preamble = {1, 18874369, 1275133960, 1, 16, 16, 1, 8};
  args.metadata = {16, 256, 16, 256, 16, 128, 8, 128};
  args.alpha = 2.0f;
  args.beta = 2.0f;

  TensileGemmRunner<float> runner("tensilelite_mm_f32_mi300x.s");
  auto optString = runner.run(dims, args, 1);
  if (optString) {
    FAIL() << *optString;
  }
}

//
// clang-format off
//
//       OperationType: GEMM
//       DataType: b
//       DestDataType: b
//       ComputeDataType: s
//       HighPrecisionAccumulate: True
//       TransposeA: False
//       TransposeB: False
//       # UseBeta: False
//       Batched: True
//     - # BenchmarkProblemSizeGroup
//       InitialSolutionParameters:
//       BenchmarkCommonParameters:
//         - KernelLanguage: ["Assembly"]
//       ForkParameters:
//         - TransposeLDS: [0]
//         - MatrixInstruction:
//             - [16, 16, 16, 1, 1, 1, 1, 1, 1]
//         - WorkGroup:
//           - [8, 8, 1 ]
//         - DepthU: [32]
//         - PrefetchLocalRead: [1]
//         - PrefetchGlobalRead: [True]
//         - WorkGroupMapping: [8]
//         - GlobalSplitU: [1]
//         - InnerUnroll: [2]
//       BenchmarkForkParameters:
//       JoinParameters:
//       BenchmarkJoinParameters:
//       BenchmarkFinalParameters:
//         - ProblemSizes:
//            - Exact: [16, 32, 2, 128]
// Appending argument 'gemm_count' of size 4 and value: 1 at offset 0 (bound=1)
// Appending argument 'internalArgs' of size 4 and value: 35651585 at offset 4 (bound=1)
// Appending argument 'internalArgs1' of size 4 and value: 1275133960 at offset 8 (bound=1)
// Appending argument 'numWorkGroups' of size 4 and value: 4 at offset 12 (bound=1)
// Appending argument '' of size 4 and value: 16 at offset 16 (bound=1)
// Appending argument '' of size 4 and value: 32 at offset 20 (bound=1)
// Appending argument '' of size 4 and value: 2 at offset 24 (bound=1)
// Appending argument '' of size 4 and value: 128 at offset 28 (bound=1)
// Appending argument 'd' of size 8 and value: 0x727181a18000 at offset 32 (bound=1)
// Appending argument 'c' of size 8 and value: 0x727181a14000 at offset 40 (bound=1)
// Appending argument 'a' of size 8 and value: 0x727181a00000 at offset 48 (bound=1)
// Appending argument 'b' of size 8 and value: 0x727181a07000 at offset 56 (bound=1)
// Appending argument '' of size 4 and value: 16 at offset 64 (bound=1)
// Appending argument '' of size 4 and value: 512 at offset 68 (bound=1)
// Appending argument '' of size 4 and value: 16 at offset 72 (bound=1)
// Appending argument '' of size 4 and value: 512 at offset 76 (bound=1)
// Appending argument '' of size 4 and value: 16 at offset 80 (bound=1)
// Appending argument '' of size 4 and value: 2048 at offset 84 (bound=1)
// Appending argument '' of size 4 and value: 128 at offset 88 (bound=1)
// Appending argument '' of size 4 and value: 4096 at offset 92 (bound=1)
// Appending argument 'alpha' of size 4 and value: 2 at offset 96 (bound=1)
// Appending argument 'beta' of size 4 and value: 2 at offset 100 (bound=1)
//
// clang-format on

TEST(GpuEmulatorTestSuiteZero, MatMul_TensileLite_BF16) {
  // From log: Exact: [16, 32, 2, 128] -> [M, N, Batch, K]
  GemmDims dims{16, 32, 128, 2};

  TensileKernelArgs args;
  // Based on logs provided:
  // Preamble args before pointers
  args.preamble = {
      1,                     // gemm_count
      35651585,              // internalArgs
      1275133960,            // internalArgs1
      4,                     // numWorkGroups
      16,         32, 2, 128 // sizes
  };

  // Metadata args after pointers
  args.metadata = {
      16,  512,  // Stride set A
      16,  512,  // Stride set B
      16,  2048, // Stride set C
      128, 4096  // Stride set D
  };

  args.alpha = 2.0f;
  args.beta = 2.0f;

  // Uses uint16_t for KernelType (BF16 storage)
  TensileGemmRunner<uint16_t> runner("tensilelite_mm_bf16_mi300x.s");

  // numWorkGroups = 4 (from log)
  auto optString = runner.run(dims, args, 1, 4);
  if (optString) {
    FAIL() << *optString;
  }
}

// --- Tests ---
