// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Arch.h"
#include "race-emulator/Emulator.h"
#include "race-emulator/FloatTypes.h"
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
  // Additional args after alpha/beta (e.g. scaleAlphaVec ptr, bias ptr,
  // bias_type, strideBias for UserArgs kernels). Pointers are stored as
  // two uint32_t values (low, high).
  std::vector<uint32_t> postScalars;
};

// CPU reference implementation

// Always calculates in float (F32)
void cpuGemmReferenceF32(const GemmDims &dims, const std::vector<float> &h_a,
                         const std::vector<float> &h_b,
                         const std::vector<float> &h_c,
                         std::vector<float> &h_d_expected, float alpha,
                         float beta, bool transposeA = false) {

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
          // Column Major A (NN): A[m + M*k]
          // Row Major A (TN):    A[k + K*m]
          float valA = transposeA ? h_a[batchOffsetA + k + dims.K * m]
                                  : h_a[batchOffsetA + m + dims.M * k];
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
  // useF16: when KernelType is uint16_t, use f16 conversion instead of bf16.
  TensileGemmRunner(const std::string &kernel_file,
                    std::shared_ptr<Architecture> arch, WaveSize waveSize,
                    bool useF16 = false, bool transposeA = false)
      : assembly_(loadKernelFile(kernel_file)), arch_(std::move(arch)),
        waveSize_(waveSize), useF16_(useF16), transposeA_(transposeA) {}

  void enableProfiling(bool enable) { profiling_ = enable; }

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
    Emulator emulator(assembly_, arch_);

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

    // E. Post-scalars (additional args after alpha/beta)
    postScalarsStorage_ = kArgs.postScalars;
    for (uint64_t i = 0; i < postScalarsStorage_.size(); ++i) {
      emulator.addKernarg(argIdx++, &postScalarsStorage_[i]);
    }

    // 5. Run Emulator
    RunConfig config{.raceChecks = true, .profiling = profiling_};
    for (int i = 0; i < nWorkgroups; ++i) {

      Dim3d wgId(i, 0, 0);
      Dim3d blockDim(nWavesPerWorkgroup * waveSize_, 1, 1);
      emulator.run(wgId, blockDim, config);
    }

    if (profiling_) {
      std::cerr << emulator.getProfileReport() << "\n";
    }

    // 6. Verify Results
    // A. Run Reference (F32 -> F32)
    std::vector<float> dRefF32;
    cpuGemmReferenceF32(dims, aF32, bF32, cF32, dRefF32, kArgs.alpha,
                        kArgs.beta, transposeA_);

    // B. Convert GPU output back to F32 for comparison
    std::vector<float> dGpuAsF32 = convertToHost(dGpu);
    return verifyResults(dGpuAsF32, dRefF32, dims);
  }

private:
  std::string assembly_;
  std::shared_ptr<Architecture> arch_;
  WaveSize waveSize_;
  bool useF16_;
  bool transposeA_;
  bool profiling_ = false;
  // Storage to keep arg pointers valid
  std::vector<uint32_t> preambleStorage_;
  std::vector<uint32_t> metadataStorage_;
  std::vector<uint32_t> postScalarsStorage_;
  float alphaStorage_;
  float betaStorage_;

  void initializeDataF32(std::vector<float> &data) {
    static std::mt19937 rng(1013);
    std::uniform_int_distribution<int> dist(0, 1);
    std::vector<float> choices = {
        1.0f, -1.0f}; // Simple integer-ish floats to avoid rounding noise
    for (auto &val : data) {
      val = choices[dist(rng)];
    }
  }

  // Helper: Convert F32 vector to KernelType vector
  std::vector<KernelType> convertToKernel(const std::vector<float> &src) {
    std::vector<KernelType> dst(src.size());
    if constexpr (std::is_same_v<KernelType, float>) {
      dst = src;
    } else if constexpr (std::is_same_v<KernelType, uint16_t>) {
      for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = useF16_ ? raceemulator::floatToF16(src[i])
                         : raceemulator::floatToBf16(src[i]);
      }
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
      for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = useF16_ ? raceemulator::f16ToFloat(src[i])
                         : raceemulator::bf16ToFloat(src[i]);
      }
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
//
// === How to create / reproduce a TensileLite e2e test ===
//
// Each gfx1151 test has a corresponding YAML file in tests/asm/gfx1151/.
// No GPU hardware is required for any of the following steps.
//
// Step 1: Generate assembly from the YAML.
//
//   source <venv>/bin/activate
//   Tensile.sh <yaml> <outputdir> --build-only
//       --prebuilt-client=<builddir>/tensilelite/client/tensilelite-client
//
//   The generated .s file is in <outputdir>/.../SOURCE/assembly/*.s.
//   Copy it to tests/asm/gfx1151/ with a descriptive name.
//
// Step 2: Get the kernel arguments.
//
//   Tensile.sh <yaml> <outputdir> --print-kernel-args
//       --prebuilt-client=<builddir>/tensilelite/client/tensilelite-client
//
//   This prints every kernel argument with its offset, size, and value.
//   Copy the values into the test as TensileKernelArgs: preamble (args before
//   pointers), metadata (strides after pointers), alpha, and beta.
//   Pointer values (d, c, a, b) are placeholders and are set by the test
//   runner to point to host-allocated buffers.
//
// Step 3: Determine dispatch parameters from the assembly metadata.
//
//   .max_flat_workgroup_size / .wavefront_size = waves per workgroup.
//   numWorkGroups (from --print-kernel-args) = total workgroups.
//   Pass these to runner.run(dims, args, wavesPerWG, numWorkgroups).
//
// === Limitations ===
//
// internalArgs1 encodes WGMXCCG (bits [31:22]), which depends on the GPU's
// CU count. Without a GPU, --print-kernel-args uses an approximate value.
// This affects performance tuning on real hardware but does not affect
// numerical correctness. The emulator does not model CU scheduling, so
// approximate values work fine for these tests.
//
// internalArgs encodes StaggerU settings. The StaggerUStrideShift value may
// vary between TensileLite versions. This is also a performance tuning
// parameter and does not affect numerical correctness.
//
// === gfx942 tests ===
//
// The gfx942 tests below were created before the YAML+print-kernel-args
// workflow was established. Their kernel args were obtained by instrumenting
// the TensileLite client on a machine with a gfx942 GPU. Both tests now
// have corresponding YAML files.
//
// YAML: tests/sources/tensilelite/gfx942/tensilelite_mm_f32_mi300x.yaml
TEST(Gfx942, MatMul_TensileLite_F32) {
  GemmDims dims{16, 16, 8, 1}; // M, N, K, Batch

  TensileKernelArgs args;
  args.preamble = {1, 18874369, 1275133960, 1, 16, 16, 1, 8};
  args.metadata = {16, 256, 16, 256, 16, 128, 8, 128};
  args.alpha = 2.0f;
  args.beta = 2.0f;

  TensileGemmRunner<float> runner("gfx942/tensilelite_mm_f32_mi300x.s",
                                  std::make_shared<Gfx942>(), WaveSize{64});
  auto optString = runner.run(dims, args, 1);
  if (optString) {
    FAIL() << *optString;
  }
}

// YAML: tests/sources/tensilelite/gfx942/tensilelite_mm_bf16_mi300x.yaml

TEST(Gfx942, MatMul_TensileLite_BF16) {
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
  TensileGemmRunner<uint16_t> runner("gfx942/tensilelite_mm_bf16_mi300x.s",
                                     std::make_shared<Gfx942>(), WaveSize{64});

  // numWorkGroups = 4 (from log)
  auto optString = runner.run(dims, args, 1, 4);
  if (optString) {
    FAIL() << *optString;
  }
}

// YAML: builds/tensilelite/bf16_128x128x1024.yaml
// Assembly: tests/asm/gfx942/tensilelite_mm_bf16_128x128x1024.s
// Large single-workgroup bf16 GEMM for race-detection benchmarking.
// MT128x128, DepthU=32, 16 waves per WG (1024 threads), 1 workgroup.
// MI=[16,16,16], waveGroup=[2,2], waveTile=[4,4] (MIWT2_2).
TEST(Gfx942, MatMul_TensileLite_BF16_128x128x1024) {
  GemmDims dims{128, 128, 1024, 1}; // M, N, K, Batch

  TensileKernelArgs args;
  args.preamble = {
      1,                       // gemm_count
      35651585,                // internalArgs
      1275133960,              // internalArgs1
      1,                       // numWorkGroups
      128,        128, 1, 1024 // sizes
  };

  args.metadata = {
      128,  16384,  // strideD1, strideD2
      128,  16384,  // strideC1, strideC2
      128,  131072, // strideA1, strideA2
      1024, 131072  // strideB1, strideB2
  };

  args.alpha = 1.0f;
  args.beta = 0.0f;

  TensileGemmRunner<uint16_t> runner(
      "gfx942/tensilelite_mm_bf16_128x128x1024.s", std::make_shared<Gfx942>(),
      WaveSize{64});
  runner.enableProfiling(true);

  // 16 waves per WG (1024 threads / 64), 1 workgroup
  auto optString = runner.run(dims, args, 16, 1);
  if (optString) {
    FAIL() << *optString;
  }
}

// YAML: tests/sources/tensilelite/gfx1151/tensilelite_mm_f32_mac.yaml
// Assembly: tests/asm/gfx1151/tensilelite_mm_f32_mac.s
// MAC-based f32 GEMM (no MFMA/WMMA matrix instructions).
// ThreadTile [2,2], WorkGroup [8,8,1], DepthU 8.
// 2 waves per workgroup, 1 workgroup.
TEST(Gfx1151, MatMul_TensileLite_F32_MAC) {
  GemmDims dims{16, 16, 16, 1}; // M, N, K, Batch

  TensileKernelArgs args;
  // gemm_count, internalArgs, internalArgs1, numWG, M, N, Batch, K
  args.preamble = {1, 52428801, 83951617, 1, 16, 16, 1, 16};
  // strideD1, strideD2, strideC1, strideC2,
  // strideA1, strideA2, strideB1, strideB2
  args.metadata = {16, 256, 16, 256, 16, 256, 16, 256};
  args.alpha = 1.0f;
  args.beta = 0.0f;

  TensileGemmRunner<float> runner("gfx1151/tensilelite_mm_f32_mac.s",
                                  std::make_shared<Gfx1151>(), WaveSize{32});
  auto optString = runner.run(dims, args, 2);
  if (optString) {
    FAIL() << *optString;
  }
}

// YAML: tests/sources/tensilelite/gfx1151/tensilelite_mm_f32_mac_large.yaml
// Assembly: tests/asm/gfx1151/tensilelite_mm_f32_mac_large.s
// MAC-based f32 GEMM, larger problem: M=128, N=256, K=96.
// ThreadTile [4,4], WorkGroup [8,8,1], DepthU 16.
// 2 waves per workgroup, 32 workgroups.
TEST(Gfx1151, MatMul_TensileLite_F32_MAC_Large) {
  GemmDims dims{128, 256, 96, 1};

  TensileKernelArgs args;
  args.preamble = {1, 35651585, 83951624, 32, 128, 256, 1, 96};
  args.metadata = {128, 32768, 128, 32768, 128, 12288, 96, 24576};
  args.alpha = 1.0f;
  args.beta = 0.0f;

  TensileGemmRunner<float> runner("gfx1151/tensilelite_mm_f32_mac_large.s",
                                  std::make_shared<Gfx1151>(), WaveSize{32});
  runner.enableProfiling(true);
  auto optString = runner.run(dims, args, 2, 32);
  if (optString) {
    FAIL() << *optString;
  }
}

// YAML: tests/sources/tensilelite/gfx1151/tensilelite_mm_bf16_wmma_small.yaml
// Assembly: tests/asm/gfx1151/tensilelite_mm_bf16_wmma_small.s
// WMMA bf16->f32 GEMM: M=N=16, K=128, single workgroup.
// Uses v_wmma_f32_16x16x16_bf16 matrix instructions.
// DepthU 16, InnerUnroll 1, no prefetch. 1 wave.
TEST(Gfx1151, MatMul_TensileLite_BF16_WMMA) {
  GemmDims dims{16, 16, 128, 1};

  TensileKernelArgs args;
  args.preamble = {1, 35651585, 83951624, 1, 16, 16, 1, 128};
  args.metadata = {16, 256, 16, 256, 16, 2048, 128, 2048};
  args.alpha = 1.0f;
  args.beta = 0.0f;

  TensileGemmRunner<uint16_t> runner("gfx1151/tensilelite_mm_bf16_wmma_small.s",
                                     std::make_shared<Gfx1151>(), WaveSize{32});
  auto optString = runner.run(dims, args, 1);
  if (optString) {
    FAIL() << *optString;
  }
}

// YAML:
// tests/sources/tensilelite/gfx1151/tensilelite_mm_f16_wmma_nonsquare.yaml
// Assembly: tests/asm/gfx1151/tensilelite_mm_f16_wmma_nonsquare.s
// WMMA f16->f32 GEMM: M=32, N=16, K=128. Non-square output tile with f16
// inputs. Exercises v_wmma_f32_16x16x16_f16. 2 workgroups (1 wave each). HW
// validated: PASSED.
TEST(Gfx1151, MatMul_TensileLite_F16_WMMA_M32N16) {
  GemmDims dims{32, 16, 128, 1};

  TensileKernelArgs args;
  args.preamble = {1, 52428801, 83951624, 2, 32, 16, 1, 128};
  args.metadata = {32, 512, 32, 512, 32, 4096, 128, 2048};
  args.alpha = 1.0f;
  args.beta = 0.0f;

  TensileGemmRunner<uint16_t> runner(
      "gfx1151/tensilelite_mm_f16_wmma_nonsquare.s",
      std::make_shared<Gfx1151>(), WaveSize{32},
      /*useF16=*/true);
  auto optString = runner.run(dims, args, 1, 2);
  if (optString) {
    FAIL() << *optString;
  }
}

// YAML:
// tests/sources/tensilelite/gfx1151/tensilelite_mm_bf16_wmma_2wave_large.yaml
// Assembly: tests/asm/gfx1151/tensilelite_mm_bf16_wmma_2wave_large.s
// WMMA bf16->f32 GEMM: M=16, N=32, K=512. 2 waves per workgroup (MIBlock_N=2),
// 1 workgroup. Kernel tile MT16x32x16 with 32 inner-loop iterations (K/DepthU).
// Exercises inter-wave LDS cooperation and looped inner body. HW validated:
// PASSED.
TEST(Gfx1151, MatMul_TensileLite_BF16_WMMA_2Wave_M16N32K512) {
  GemmDims dims{16, 32, 512, 1};

  TensileKernelArgs args;
  args.preamble = {1, 52428801, 83951624, 1, 16, 32, 1, 512};
  args.metadata = {16, 512, 16, 512, 16, 8192, 512, 16384};
  args.alpha = 1.0f;
  args.beta = 0.0f;

  TensileGemmRunner<uint16_t> runner(
      "gfx1151/tensilelite_mm_bf16_wmma_2wave_large.s",
      std::make_shared<Gfx1151>(), WaveSize{32});
  auto optString = runner.run(dims, args, 2, 1);
  if (optString) {
    FAIL() << *optString;
  }
}

// YAML: tests/sources/tensilelite/gfx1151/tensilelite_mm_f16_wmma_tn_small.yaml
// Assembly: tests/asm/gfx1151/tensilelite_mm_f16_wmma_tn_small.s
// First TN (TransposeA) f16 WMMA test. Exercises v_wmma_f32_16x16x16_f16 with
// row-major A layout (A[k + K*m]). Single WMMA tile, 1 wave, 1 workgroup.
TEST(Gfx1151, MatMul_TensileLite_F16_WMMA_TN_Small) {
  GemmDims dims{16, 16, 16, 1};

  TensileKernelArgs args;
  args.preamble = {1, 52428801, 83951624, 1, 16, 16, 1, 16};
  args.metadata = {16, 256, 16, 256, 16, 256, 16, 256};
  args.alpha = 1.0f;
  args.beta = 0.0f;

  TensileGemmRunner<uint16_t> runner(
      "gfx1151/tensilelite_mm_f16_wmma_tn_small.s", std::make_shared<Gfx1151>(),
      WaveSize{32},
      /*useF16=*/true, /*transposeA=*/true);
  auto optString = runner.run(dims, args, 1, 1);
  if (optString) {
    FAIL() << *optString;
  }
}

// YAML:
// tests/sources/tensilelite/gfx1151/tensilelite_mm_f16_wmma_tn_n64_4wg.yaml
// Assembly: tests/asm/gfx1151/tensilelite_mm_f16_wmma_tn_small.s (reused)
// First multi-workgroup TN test. 4 WGs with WorkGroupMapping=8 remapping.
// Same kernel as TN_Small, different problem size and dispatch.
TEST(Gfx1151, MatMul_TensileLite_F16_WMMA_TN_N64_4WG) {
  GemmDims dims{16, 64, 16, 1};

  TensileKernelArgs args;
  args.preamble = {1, 52428801, 83951624, 4, 16, 64, 1, 16};
  args.metadata = {16, 1024, 16, 1024, 16, 256, 16, 1024};
  args.alpha = 1.0f;
  args.beta = 0.0f;

  TensileGemmRunner<uint16_t> runner(
      "gfx1151/tensilelite_mm_f16_wmma_tn_small.s", std::make_shared<Gfx1151>(),
      WaveSize{32},
      /*useF16=*/true, /*transposeA=*/true);
  auto optString = runner.run(dims, args, 1, 4);
  if (optString) {
    FAIL() << *optString;
  }
}

// YAML:
// tests/sources/tensilelite/gfx1151/tensilelite_mm_f16_wmma_tn_n64_2wg2w.yaml
// Assembly: tests/asm/gfx1151/tensilelite_mm_f16_wmma_tn_n64_2wg2w.s
// First multi-wave TN test. 2 waves per WG (MIBlock_N=2, waves in N direction),
// 2 workgroups. Distinct kernel from TN_Small (MT16x32 vs MT16x16).
TEST(Gfx1151, MatMul_TensileLite_F16_WMMA_TN_N64_2WG2W) {
  GemmDims dims{16, 64, 16, 1};

  TensileKernelArgs args;
  args.preamble = {1, 52428801, 83951624, 2, 16, 64, 1, 16};
  args.metadata = {16, 1024, 16, 1024, 16, 256, 16, 1024};
  args.alpha = 1.0f;
  args.beta = 0.0f;

  TensileGemmRunner<uint16_t> runner(
      "gfx1151/tensilelite_mm_f16_wmma_tn_n64_2wg2w.s",
      std::make_shared<Gfx1151>(), WaveSize{32},
      /*useF16=*/true, /*transposeA=*/true);
  auto optString = runner.run(dims, args, 2, 2);
  if (optString) {
    FAIL() << *optString;
  }
}

// YAML: tests/sources/tensilelite/gfx1151/tensilelite_mm_f16_wmma_tn_1024.yaml
// Assembly: tests/asm/gfx1151/tensilelite_mm_f16_wmma_tn_1024.s
// Large K dimension: K=1024 with DepthU=16 gives 64 inner-loop iterations.
// Stresses the WMMA accumulation loop. Distinct kernel from TN_Small.
TEST(Gfx1151, MatMul_TensileLite_F16_WMMA_TN_1024) {
  GemmDims dims{16, 16, 1024, 1};

  TensileKernelArgs args;
  args.preamble = {1, 52428801, 83951624, 1, 16, 16, 1, 1024};
  args.metadata = {16, 256, 16, 256, 1024, 16384, 1024, 16384};
  args.alpha = 1.0f;
  args.beta = 0.0f;

  TensileGemmRunner<uint16_t> runner(
      "gfx1151/tensilelite_mm_f16_wmma_tn_1024.s", std::make_shared<Gfx1151>(),
      WaveSize{32},
      /*useF16=*/true, /*transposeA=*/true);
  auto optString = runner.run(dims, args, 1, 1);
  if (optString) {
    FAIL() << *optString;
  }
}

// YAML: tests/sources/tensilelite/gfx1151/tensilelite_mm_f16_wmma_tn_4wave.yaml
// Assembly: tests/asm/gfx1151/tensilelite_mm_f16_wmma_tn_4wave.s
// Most complex TN test. 4 waves/WG, 2 WGs, large K=4096 (256 iterations),
// and edge handling (MT_M=32 > M=16). Distinct kernel (MT32x64x16).
TEST(Gfx1151, MatMul_TensileLite_F16_WMMA_TN_4Wave) {
  GemmDims dims{16, 128, 4096, 1};

  TensileKernelArgs args;
  args.preamble = {1, 52428801, 83951624, 2, 16, 128, 1, 4096};
  args.metadata = {16, 2048, 16, 2048, 4096, 65536, 4096, 524288};
  args.alpha = 1.0f;
  args.beta = 0.0f;

  TensileGemmRunner<uint16_t> runner(
      "gfx1151/tensilelite_mm_f16_wmma_tn_4wave.s", std::make_shared<Gfx1151>(),
      WaveSize{32},
      /*useF16=*/true, /*transposeA=*/true);
  auto optString = runner.run(dims, args, 4, 2);
  if (optString) {
    FAIL() << *optString;
  }
}

// Custom kernel: TN f16 WMMA, M=16, N=128, K=4096, GSU=1.
// Hand-optimized kernel from PR #4576, running with GSU=1 for single-kernel
// emulation. MT16x64x64, wave32, 4 waves/WG, 2 workgroups.
// UserArgs format (KernArgsVersion 2) with bias/scaleAlphaVec/activation args.
// Uses addAllKernargs with a raw byte buffer because the arg layout differs
// from the standard TensileGemmRunner pattern (extra trailing pointer and
// scalar args after alpha/beta).
TEST(Gfx1151, MatMul_TensileLite_F16_WMMA_TN_CustomKernel) {
  constexpr int M = 16, N = 128, K = 4096, Batch = 1;
  constexpr int sizeA = M * K * Batch;
  constexpr int sizeB = N * K * Batch;
  constexpr int sizeC = M * N * Batch;

  // 1. Initialize host data
  std::mt19937 rng(1013);
  std::uniform_int_distribution<int> dist(0, 1);
  std::vector<float> choices = {1.0f, -1.0f};

  std::vector<float> aF32(sizeA), bF32(sizeB), cF32(sizeC, 0.0f);
  for (auto &v : aF32) {
    v = choices[dist(rng)];
  }
  for (auto &v : bF32) {
    v = choices[dist(rng)];
  }

  // Convert to f16
  std::vector<uint16_t> aGpu(sizeA), bGpu(sizeB), cGpu(sizeC, 0),
      dGpu(sizeC, 0);
  for (int i = 0; i < sizeA; ++i) {
    aGpu[i] = floatToF16(aF32[i]);
  }
  for (int i = 0; i < sizeB; ++i) {
    bGpu[i] = floatToF16(bF32[i]);
  }

  // ScaleAlphaVec: f32 array, one per M element. All 1.0f (identity).
  std::vector<float> scaleAlphaVec(M, 1.0f);
  // Bias: f16 array, one per M element. All 0 (no bias).
  std::vector<uint16_t> biasGpu(M, 0);

  // 2. Build raw kernarg buffer (140 bytes)
  // Layout from --print-kernel-args (GSU=1):
  //   [0..3]   gemm_count=1
  //   [4..7]   internalArgs=18874369
  //   [8..11]  internalArgs1=83951624
  //   [12..15] numWorkGroups=2
  //   [16..19] M=16, [20..23] N=128, [24..27] Batch=1, [28..31] K=4096
  //   [32..39] d ptr, [40..47] c ptr, [48..55] a ptr, [56..63] b ptr
  //   [64..67] strideD1=16, [68..71] strideD2=2048
  //   [72..75] strideC1=16, [76..79] strideC2=2048
  //   [80..83] strideA1=4096, [84..87] strideA2=65536
  //   [88..91] strideB1=4096, [92..95] strideB2=524288
  //   [96..99] alpha=1.0f, [100..103] beta=0.0f
  //   [104..111] scaleAlphaVec=null, [112..119] bias=null
  //   [120..123] biasType=4, [124..127] strideBias=0
  //   [128..131] activationAlpha=0.0f, [132..135] activationBeta=0.0f
  //   [136..139] activationType=0
  //   [140..143] padding (kernarg_segment_size=144, 8-byte aligned)
  std::vector<uint8_t> argBuf(144, 0);
  auto put32 = [&](int off, uint32_t v) { std::memcpy(&argBuf[off], &v, 4); };
  auto put64 = [&](int off, uint64_t v) { std::memcpy(&argBuf[off], &v, 8); };
  auto putF = [&](int off, float v) { std::memcpy(&argBuf[off], &v, 4); };

  put32(0, 1);        // gemm_count
  put32(4, 18874369); // internalArgs (GSU=1 encoded)
  put32(8, 83951624); // internalArgs1
  put32(12, 2);       // numWorkGroups
  put32(16, M);
  put32(20, N);
  put32(24, Batch);
  put32(28, K);
  put64(32, reinterpret_cast<uint64_t>(dGpu.data()));
  put64(40, reinterpret_cast<uint64_t>(cGpu.data()));
  put64(48, reinterpret_cast<uint64_t>(aGpu.data()));
  put64(56, reinterpret_cast<uint64_t>(bGpu.data()));
  put32(64, M);     // strideD1
  put32(68, M * N); // strideD2
  put32(72, M);     // strideC1
  put32(76, M * N); // strideC2
  put32(80, K);     // strideA1 (TN: A is K x M, stride along M = K)
  put32(84, M * K); // strideA2
  put32(88, K);     // strideB1
  put32(92, N * K); // strideB2
  putF(96, 1.0f);   // alpha
  putF(100, 0.0f);  // beta
  put64(104, reinterpret_cast<uint64_t>(scaleAlphaVec.data())); // scaleAlphaVec
  put64(112, reinterpret_cast<uint64_t>(biasGpu.data()));       // bias
  put32(120, 4); // biasType = Half
  // [124..139] strideBias=0, activationAlpha=0, activationBeta=0,
  // activationType=0

  // 3. Create emulator and load args
  std::string assembly =
      loadKernelFile("gfx1151/tensilelite_custom_f16_wmma_tn_m16n128k4096.s");
  Emulator emulator(assembly, std::make_shared<Gfx1151>());
  emulator.addAllKernargs(argBuf.data());

  // 4. Run: 4 waves/WG (128 threads, wave32), 2 workgroups
  constexpr int wavesPerWG = 4;
  constexpr WaveSize waveSize{32};
  for (int wg = 0; wg < 2; ++wg) {
    emulator.run({wg, 0, 0}, {wavesPerWG * waveSize, 1, 1},
                 {.raceChecks = true});
  }

  // 5. Verify against CPU reference
  std::vector<float> dRefF32;
  cpuGemmReferenceF32({M, N, K, Batch}, aF32, bF32, cF32, dRefF32, 1.0f, 0.0f,
                      /*transposeA=*/true);

  std::vector<float> dGpuF32(sizeC);
  for (int i = 0; i < sizeC; ++i) {
    dGpuF32[i] = f16ToFloat(dGpu[i]);
  }

  int mismatches = 0;
  for (int i = 0; i < sizeC; ++i) {
    if (std::abs(dGpuF32[i] - dRefF32[i]) > 1e-2f) {
      if (mismatches < 5) {
        std::cout << "  idx " << i << ": expected " << dRefF32[i] << ", got "
                  << dGpuF32[i] << "\n";
      }
      ++mismatches;
    }
  }
  EXPECT_EQ(mismatches, 0) << mismatches << "/" << sizeC << " mismatches";
}

// YAML:
// tests/sources/tensilelite/gfx1151/tensilelite_mm_f16_wmma_tn_128x128.yaml
// Assembly: tests/asm/gfx1151/tensilelite_mm_f16_wmma_tn_128x128.s
//
// Large single-workgroup f16 TN WMMA GEMM for race-detection benchmarking.
// The kernel tile is MT128x128 (the largest square tile feasible on gfx1151,
// limited by 256 VGPRs and 64KB LDS). K=8192 is a runtime parameter — the
// kernel is K-agnostic and loops K/DepthU=512 times through the inner body.
//
// Kernel properties:
//   MT128x128x16, DepthU=16, 4 waves/WG, wave32, 1 workgroup.
//   MIWaveGroup=[2,2], MIWaveTile=[4,4], 256 VGPRs, 10240 LDS bytes.
//   9860 assembly lines. HW validated: PASSED (611 GFLOPS at K=4096).
//
// Race emulation profile (release, K=8192):
//   ~55% of 237ms runtime is race detection (removeEvents, DsRead validation,
//   retireEventRegisters, registerLds/VgprEvents).
//
// Skipped in debug builds (~12x slower). RELEASE_BUILD is defined by CMake
// for Release and RelWithDebInfo configurations (see tests/CMakeLists.txt).
//
// YAML: tests/sources/tensilelite/gfx950/tensilelite_dtl_f16_mfma_tn.yaml
// Assembly: tests/asm/gfx950/tensilelite_dtl_f16_mfma_tn_m256n256k128.s
// Direct-to-LDS (DTL) f16 TN MFMA GEMM on gfx950 (MI350X).
// Uses buffer_load ... lds to load A and B directly into LDS.
// MT32x32x64, wave64, 4 waves/WG, 64 workgroups.
TEST(Gfx950, MatMul_TensileLite_F16_MFMA_TN_DTL_256x256x128) {
  GemmDims dims{256, 256, 128, 1};

  TensileKernelArgs args;
  args.preamble = {1, 18874369, 1275133960, 64, 256, 256, 1, 128};
  args.metadata = {256, 65536, 256, 65536, 128, 32768, 128, 32768};
  args.alpha = 1.0f;
  args.beta = 0.0f;

  TensileGemmRunner<uint16_t> runner(
      "gfx950/tensilelite_dtl_f16_mfma_tn_m256n256k128.s",
      std::make_shared<Gfx950>(), WaveSize{64},
      /*useF16=*/true, /*transposeA=*/true);
  runner.enableProfiling(true);

  // 4 waves per WG (256 threads / 64), 64 workgroups
  auto optString = runner.run(dims, args, 4, 64);
  if (optString) {
    FAIL() << *optString;
  }
}

// MXFP4 subtile kernel on gfx950: 32x32x256 GEMM with FP4 inputs, F32 output.
TEST(Gfx950, MatMul_TensileLite_MXFP4_TN_32x32x256) {
  constexpr int M = 32, N = 32, K = 256, Batch = 1;
  constexpr int MXBlock = 32;
  constexpr int numScalesPerRow = K / MXBlock; // 8

  // FP4: 2 elements per byte. Initialize all nibbles to 2 (=1.0f).
  std::vector<uint8_t> aFp4(M * K / 2, 0x22);
  std::vector<uint8_t> bFp4(N * K / 2, 0x22);
  // MX scale factors: E8M0 exponent 127 = scale 1.0.
  std::vector<uint8_t> mxScaleA(M * numScalesPerRow, 127);
  std::vector<uint8_t> mxScaleB(N * numScalesPerRow, 127);
  // F32 output and C (zeroed for beta=0).
  std::vector<float> cF32(M * N, 0.0f);
  std::vector<float> dF32(M * N, 0.0f);

  // Build kernarg buffer (136 bytes, KernArgsVersion 2).
  std::vector<uint8_t> argBuf(136, 0);
  auto put32 = [&](int off, uint32_t v) { std::memcpy(&argBuf[off], &v, 4); };
  auto put64 = [&](int off, uint64_t v) { std::memcpy(&argBuf[off], &v, 8); };
  auto putF = [&](int off, float v) { std::memcpy(&argBuf[off], &v, 4); };
  auto putPtr = [&](int off, const void *p) {
    uint64_t v = reinterpret_cast<uint64_t>(p);
    put64(off, v);
  };

  // Preamble.
  put32(0, 1);        // Gemm info (gemm_count = 1)
  put32(4, 18874369); // kernel info0 (standard GSU=1)
  put32(8, 0);        // kernel info1
  put32(12, 1);       // numWG
  put32(16, M);       // SizesFree0
  put32(20, N);       // SizesFree1
  put32(24, Batch);   // SizesFree2
  put32(28, K);       // SizesSum0

  // Pointers.
  putPtr(32, dF32.data());
  putPtr(40, cF32.data());
  putPtr(48, aFp4.data());
  putPtr(56, mxScaleA.data());
  putPtr(64, bFp4.data());
  putPtr(72, mxScaleB.data());

  // Strides.
  put32(80, 1);              // strideD0
  put32(84, M);              // strideD1
  put32(88, 1);              // strideC0
  put32(92, M);              // strideC1
  put32(96, 1);              // strideA0
  put32(100, K);             // strideA1 (TN layout)
  put32(104, 1);             // strideMXSA0
  put32(108, numScalesPerRow); // strideMXSA1
  put32(112, 1);             // strideB0
  put32(116, K);             // strideB1
  put32(120, 1);             // strideMXSB0
  put32(124, numScalesPerRow); // strideMXSB1

  // Scalars.
  putF(128, 1.0f); // alpha
  putF(132, 0.0f); // beta

  std::string assembly =
      loadKernelFile("gfx950/tensilelite_mxfp4_subtile_m32n32k256.s");
  Emulator emulator(assembly, std::make_shared<Gfx950>());
  emulator.addAllKernargs(argBuf.data());

  // WG 32x8x1 = 256 threads, wave64 = 4 waves, 1 workgroup.
  emulator.run({0, 0, 0}, {256, 1, 1},
               {.raceChecks = true, .completeEmulation = true});

  // Print output for debugging.
  std::cerr << "D output (first 8 rows):\n";
  for (int i = 0; i < 8; ++i) {
    std::cerr << "  row " << i << ":";
    for (int j = 0; j < N; ++j) {
      std::cerr << " " << dF32[i * N + j];
    }
    std::cerr << "\n";
  }
  // Count non-zero elements.
  int nonZero = 0;
  for (int i = 0; i < M * N; ++i) {
    if (dF32[i] != 0.0f) {
      nonZero++;
    }
  }
  std::cerr << "Non-zero elements: " << nonZero << " / " << M * N << "\n";
}

TEST(Gfx1151, MatMul_TensileLite_F16_WMMA_TN_128x128x8192) {
#ifndef RELEASE_BUILD
  GTEST_SKIP() << "Large kernel test, release builds only";
#endif
  GemmDims dims{128, 128, 8192, 1};

  TensileKernelArgs args;
  args.preamble = {1, 52428801, 83951624, 1, 128, 128, 1, 8192};
  args.metadata = {128, 16384, 128, 16384, 8192, 1048576, 8192, 1048576};
  args.alpha = 1.0f;
  args.beta = 0.0f;

  TensileGemmRunner<uint16_t> runner(
      "gfx1151/tensilelite_mm_f16_wmma_tn_128x128.s",
      std::make_shared<Gfx1151>(), WaveSize{32},
      /*useF16=*/true, /*transposeA=*/true);
  runner.enableProfiling(true);

  // 4 waves per WG (128 threads / 32), 1 workgroup
  auto optString = runner.run(dims, args, 4, 1);
  if (optString) {
    FAIL() << *optString;
  }
}
