// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "custom_kernels.hpp"

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <iostream>

std::shared_ptr<GemmKernel> createCustomGemmKernel(const std::string&           customKernelName,
                                                   const KernelType&            kernelType,
                                                   const WorkGroupTileSize&     wgt,
                                                   const std::filesystem::path& path)
{
    std::shared_ptr<GemmKernel> gemmKernel;
    gemmKernel = std::make_shared<AssemblyStoreRowOrderGemm>(customKernelName, path.string());
    gemmKernel->params                = std::make_shared<SolutionParameters>();
    gemmKernel->params->kernelType    = kernelType;
    gemmKernel->params->workgroupTile = wgt;
    return gemmKernel;
}

std::shared_ptr<GemmKernel> createWaveGemmKernel(const std::string&           customKernelName,
                                                   const KernelType&            kernelType,
                                                   const WorkGroupTileSize&     wgt,
                                                   const dim3&    blockSize,
                                                   const ShapeCondition&        condition,
                                                   const std::filesystem::path& path)
{
    std::shared_ptr<GemmKernel> gemmKernel = std::make_shared<WaveKernel>(customKernelName, path.string(), blockSize);
    gemmKernel->params                = std::make_shared<SolutionParameters>();
    gemmKernel->params->kernelType    = kernelType;
    gemmKernel->params->workgroupTile = wgt;
    gemmKernel->shapeCondition = condition;
    return gemmKernel;
}

std::filesystem::path getCoPath()
{
    std::filesystem::path libraryPath;
    bool staticLib = false;

#ifdef HIPBLASLT_STATIC_LIB
    staticLib = true;
#endif

    const char* env = getenv("HIPBLASLT_TENSILE_LIBPATH");
    if(env)
    {
        libraryPath = env;
    }
    else
    {
        // Find the location of librocblaslt.so
        // Fall back on hard-coded path if static library or not found
        std::optional<std::filesystem::path> default_lib_path;
        if(staticLib)
        {
            // Assume library files are in "/opt/rocm"
            default_lib_path = "/opt/rocm/lib";
        }

        if(auto maybe_path = rocblaslt_find_library_relative_path(
               /*relpath=*/std::nullopt, default_lib_path))
        {
            // Worst case use "./"
            libraryPath = maybe_path.value_or(".");
        }
    }

    return libraryPath;
}

// Helper to get kernel name with optional _ntA or _ntB suffix
// Note: Currently only supports either _ntA or _ntB, not both simultaneously
std::string getKernelName(const std::string& baseName, bool nonTemporalA, bool nonTemporalB)
{
    if (nonTemporalA)
        return baseName + "_ntA";
    if (nonTemporalB)
        return baseName + "_ntB";
    return baseName;
}

// Add all custom kernels to the SolutionCache
// Need to specify the KernelType and SolutionIndexParameters
void preloadCustomKernels(SolutionCache& cache)
{
    KernelType mxfp4Kernel;
    mxfp4Kernel.typeA                     = rocRoller::DataType::FP4;
    mxfp4Kernel.typeB                     = rocRoller::DataType::FP4;
    mxfp4Kernel.typeC                     = rocRoller::DataType::BFloat16;
    mxfp4Kernel.typeD                     = rocRoller::DataType::BFloat16;
    mxfp4Kernel.transA                    = true;
    mxfp4Kernel.transB                    = false;
    mxfp4Kernel.scaleTypeA.mode           = rocRoller::Operations::ScaleMode::Separate;
    mxfp4Kernel.scaleTypeA.blockRowSize   = 32;
    mxfp4Kernel.scaleTypeA.blockColSize   = 1;
    mxfp4Kernel.scaleTypeA.preSwizzleTile = {32, 8, 4};
    mxfp4Kernel.scaleTypeA.preTile        = {32, 8};
    mxfp4Kernel.scaleTypeB.mode           = rocRoller::Operations::ScaleMode::Separate;
    mxfp4Kernel.scaleTypeB.blockRowSize   = 1;
    mxfp4Kernel.scaleTypeB.blockColSize   = 32;
    mxfp4Kernel.scaleTypeB.preSwizzleTile = {32, 8, 4};
    mxfp4Kernel.scaleTypeB.preTile        = {8, 32};

    SolutionIndexParameters params;
   
    for (bool workgroupMapping : {false, true})
    {
        // Iterate over nonTemporal combinations: (false,false), (true,false), (false,true)
        // Note: (true,true) would need _ntAB kernels which we don't have
        for (auto [nonTemporalA, nonTemporalB] : std::initializer_list<std::pair<bool,bool>>{{false,false}, {true,false}, {false,true}})
        {
            params.streamK = false;
            params.tailLoops = true;
            params.workgroupMapping = workgroupMapping;
            params.nonTemporalA = nonTemporalA;
            params.nonTemporalB = nonTemporalB;

            mxfp4Kernel.swizzleA = true;

            // 32xN kernels
            params.workgroupTile    = {32, 128, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {32, 256, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_32x256", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {32, 384, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_32x384", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {32, 512, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_32x512", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {32, 640, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_32x640", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {32, 768, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_32x768", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {32, 896, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_32x896", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {32, 1024, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_32x1024", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            // 64xN kernels
            params.workgroupTile    = {64, 128, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {64, 256, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_64x256", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));


            params.workgroupTile    = {64, 384, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_64x384", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {64, 512, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_64x512", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {64, 640, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_64x640", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {64, 768, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_64x768", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {64, 896, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_64x896", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {64, 1024, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            // 96xN kernels
            params.workgroupTile    = {96, 128, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_96x128", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {96, 256, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_96x256", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {96, 384, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_96x384", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {96, 512, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_96x512", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {96, 640, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_96x640", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            // 128xN kernels
            params.workgroupTile    = {128, 128, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_128x128", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));            

            /*params.workgroupTile    = {128, 256, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));*/

            params.workgroupTile    = {128, 384, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_128x384", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {128, 512, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            // 160xN kernels
            params.workgroupTile    = {160, 128, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_160x128", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {160, 256, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_160x256", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {160, 384, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_160x384", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            // 192xN kernels
            params.workgroupTile    = {192, 128, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_192x128", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {192, 256, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_192x256", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            // 224xN kernels
            params.workgroupTile    = {224, 128, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_224x128", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {224, 256, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_224x256", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            // 256xN kernels
            params.workgroupTile    = {256, 128, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_256x128", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile    = {256, 256, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel(
                    getKernelName("f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256", nonTemporalA, nonTemporalB),
                    mxfp4Kernel,
                    params.workgroupTile,
                    getCoPath() / "rr_custom_kernels.co"));

            // No B pre-shuffle variant (only for non-ntA)
            if (!nonTemporalA && !nonTemporalB)
            {
                mxfp4Kernel.swizzleA    = false;
                params.workgroupTile    = {256, 256, 256};
                cache.addKernel(
                    mxfp4Kernel,
                    params,
                    createCustomGemmKernel(
                        "f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256",
                        mxfp4Kernel,
                        params.workgroupTile,
                        getCoPath() / "rr_custom_kernels.co"));
                mxfp4Kernel.swizzleA = true; // Reset for next iteration
            }

            // Wave 192x256x256 kernel
            params.workgroupTile    = {192, 256, 256};
            {
                ShapeCondition wave192Condition;
                wave192Condition.customMatcher = [](size_t m, size_t n, size_t k) {
                    size_t tilesN = n / 256;
                    size_t tilesM = m / 192;
                    bool   pow2N  = tilesN > 0 && (tilesN & (tilesN - 1)) == 0;
                    return pow2N && tilesN >= 16 && tilesM * tilesN >= 256;
                };
                cache.addKernel(
                    mxfp4Kernel,
                    params,
                    createWaveGemmKernel("wave_mxfp4_dynamic_gemm_256x192x256",
                                           mxfp4Kernel,
                                           params.workgroupTile,
                                           {128, 2, 1},
                                           wave192Condition,
                                           getCoPath() / "rr_custom_kernels.co"));
            }
        }
    }
}

// Wave GEMM kernel ABI (104 bytes).
// Kernel computes C[M,N] = A[M,K] @ B[N,K]^T (scaled, preshuffle-B).
// hipBLASLt stores col-major: A as K*M, B as K*N, D as N*M.
// We swap A<->B and M<->N so Wave's row-major view matches hipBLASLt's col-major storage.
// FP4 data is 2 elements per byte, so element strides are halved to get byte strides.
struct __attribute__((packed)) WaveGemmKernelArgs
{
    const void* ptr_a;             //  0: A data
    const void* ptr_a_scale;       //  8: A scale
    const void* ptr_b;             // 16: B data
    const void* ptr_b_scale;       // 24: B scale
    void*       ptr_c;             // 32: C output
    uint64_t    m;                 // 40
    uint64_t    n;                 // 48
    uint64_t    k;                 // 56
    uint64_t    stride_a_dim0;     // 64: byte stride
    uint64_t    stride_a_scale_dim0; // 72
    uint64_t    stride_b_dim0;     // 80: byte stride
    uint64_t    stride_b_scale_dim0; // 88
    uint64_t    stride_c_dim0;     // 96
};
static_assert(sizeof(WaveGemmKernelArgs) == 104, "Wave kernel kernarg must be 104 bytes");

inline WaveGemmKernelArgs makeWaveGemmKernelArgs(const RocblasltContractionProblem& prob)
{
    WaveGemmKernelArgs w  = {};
    w.ptr_a               = prob.B;  // swap
    w.ptr_a_scale         = prob.scaleB;  // swap
    w.ptr_b               = prob.A;  // swap
    w.ptr_b_scale         = prob.scaleA;  // swap
    w.ptr_c               = prob.D;
    w.m                   = prob.n;  // swap
    w.n                   = prob.m;  // swap
    w.k                   = prob.k;
    w.stride_a_dim0       = prob.col_stride_b / 2;  // swap; FP4 byte stride
    w.stride_a_scale_dim0 = prob.k / 32;
    w.stride_b_dim0       = prob.col_stride_a / 2;  // swap; FP4 byte stride
    w.stride_b_scale_dim0 = prob.k / 32;
    w.stride_c_dim0       = prob.col_stride_c;
    return w;
}

// F4 GEMM Kernel Args

struct __attribute__((packed)) p3
{
    uint32_t _p0 = 0;
    uint32_t _p1 = 0;
    uint32_t _p2 = 0;
};
struct __attribute__((packed)) p2
{
    uint32_t _p0 = 0;
    uint32_t _p1 = 0;
};
struct __attribute__((packed)) F4GemmKernelArgs
{
    void*       ptr_D;
    p2          _p0;
    const void* ptr_C;
    p2          _p1;
    const void* ptr_A;
    p2          _p2;
    const void* ptr_B;
    p2          _p3;
    float       alpha;
    p3          _p4;
    float       beta;
    p3          _p5;
    uint32_t    stride_D0;
    p3          _p6;
    uint32_t    stride_D1;
    p3          _p7;
    uint32_t    stride_C0;
    p3          _p8;
    uint32_t    stride_C1;
    p3          _p9;
    uint32_t    stride_A0;
    p3          _p10;
    uint32_t    stride_A1;
    p3          _p11;
    uint32_t    stride_B0;
    p3          _p12;
    uint32_t    stride_B1;
    p3          _p13;
    uint32_t    M;
    p3          _p14;
    uint32_t    N;
    p3          _p15;
    uint32_t    K;
    p3          _p16;
    const void* ptr_ScaleA;
    p2          _p17;
    const void* ptr_ScaleB;
    p2          _p18;
    uint32_t    stride_ScaleA0;
    p3          _p19;
    uint32_t    stride_ScaleA1;
    p3          _p20;
    uint32_t    stride_ScaleB0;
    p3          _p21;
    uint32_t    stride_ScaleB1;
    p3          _p22;
    int         log2_k_split;

    // AITER kernel computes D[N,M] = B^T * A instead of C[M,N] = A^T * B
    // So we swap A<->B pointers/scales and M<->N dimensions
    F4GemmKernelArgs(const RocblasltContractionProblem& prob)
        : ptr_D(prob.D)
        , ptr_C(nullptr)
        , ptr_A(const_cast<void*>(prob.B)) // Swapped: kernel's A = hipBLASLt's B
        , ptr_B(const_cast<void*>(prob.A)) // Swapped: kernel's B = hipBLASLt's A
        , alpha(*static_cast<const float*>(prob.alpha))
        , beta(*static_cast<const float*>(prob.beta))
        , stride_D0(0)
        , stride_D1(0)
        , stride_C0(static_cast<uint32_t>(prob.col_stride_c))
        , stride_C1(0)
        , stride_A0(static_cast<uint32_t>(prob.col_stride_b)) // Swapped
        , stride_A1(0)
        , stride_B0(static_cast<uint32_t>(prob.col_stride_a)) // Swapped
        , stride_B1(0)
        , M(static_cast<uint32_t>(prob.n)) // Swapped: kernel's M = hipBLASLt's N
        , N(static_cast<uint32_t>(prob.m)) // Swapped: kernel's N = hipBLASLt's M
        , K(static_cast<uint32_t>(prob.k))
        , ptr_ScaleA(prob.scaleB) // Swapped
        , ptr_ScaleB(prob.scaleA) // Swapped
        , stride_ScaleA0(static_cast<uint32_t>(prob.k / 32))
        , stride_ScaleA1(0)
        , stride_ScaleB0(static_cast<uint32_t>(prob.k / 32))
        , stride_ScaleB1(0)
        , log2_k_split(0)
    {
    }
};

// WaveKernel method implementations

size_t WaveKernel::workspaceRequired(const RocblasltContractionProblem& prob)
{
    return 0;
}

bool WaveKernel::isSupportedProblem(const RocblasltContractionProblem& prob)
{
    if(shapeCondition.has_value()
           && !shapeCondition->matches(prob.m, prob.n, prob.k))
            return false;

    const auto& wgt = params->workgroupTile;
    return (prob.m % wgt.m == 0 && prob.n % wgt.n == 0 && prob.k % wgt.k == 0);
}

rocblaslt_status WaveKernel::run(const RocblasltContractionProblem& prob)
{
    if(prob.beta && *static_cast<const float*>(prob.beta) != 0)
    {
        std::cerr << "Kernel only supports when beta is 0" << std::endl;
        return rocblaslt_status_invalid_value;
    }

    WaveGemmKernelArgs waveArgs = makeWaveGemmKernelArgs(prob);
    void*              argsPtr  = &waveArgs;
    size_t             argsSize = sizeof(WaveGemmKernelArgs);

    const uint32_t tileM = params->workgroupTile.m;
    const uint32_t tileN = params->workgroupTile.n;

    uint32_t tilesM = (static_cast<uint32_t>(prob.n) + tileN - 1) / tileN;
    uint32_t tilesN = (static_cast<uint32_t>(prob.m) + tileM - 1) / tileM;
    dim3     grid(tilesM, tilesN, 1);

    void* hipLaunchParams[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                               argsPtr,
                               HIP_LAUNCH_PARAM_BUFFER_SIZE,
                               &argsSize,
                               HIP_LAUNCH_PARAM_END};

    hipFunction_t function;
    if(hipError_t error = module.getHipFunction(function))
    {
        std::cerr << "GemmHipModuleWrapper::getHipFunction failed: " << std::endl
                  << " error: " << hipGetErrorString(error) << std::endl;
        return rocblaslt_status_internal_error;
    }

    const std::string& kernelName = module.getKernelName();
    if(hipError_t error = hipModuleLaunchKernel(function,
                                                grid.x,
                                                grid.y,
                                                grid.z,
                                                blockSize.x,
                                                blockSize.y,
                                                blockSize.z,
                                                0,
                                                prob.stream,
                                                nullptr,
                                                (void**)&hipLaunchParams))
    {
        std::cerr << "hipModuleLaunchKernel failed: " << kernelName << std::endl
                  << " error: " << hipGetErrorString(error) << std::endl;
        return rocblaslt_status_internal_error;
    }

    return rocblaslt_status_success;
}

// AssemblyStoreRowOrderGemm method implementations

size_t AssemblyStoreRowOrderGemm::workspaceRequired(const RocblasltContractionProblem& prob)
{
    return 0;
}

bool AssemblyStoreRowOrderGemm::isSupportedProblem(const RocblasltContractionProblem& prob)
{
    const auto& wgt = params->workgroupTile;

    if ((wgt.m == 192 && wgt.n == 256) || (wgt.m == 224 && wgt.n == 256) || (wgt.m == 128 && wgt.n == 512))
    {
        return (prob.m % wgt.m == 0 && prob.n % wgt.n == 0 && prob.k % wgt.k == 0);
    }
    else
    {
        return (prob.m % 32 == 0 && prob.n % 32 == 0 && prob.k % wgt.k == 0);
    }
}

rocblaslt_status AssemblyStoreRowOrderGemm::run(const RocblasltContractionProblem& prob)
{
    if(prob.beta && *static_cast<const float*>(prob.beta) != 0)
    {
        std::cerr << "Kernel only supports when beta is 0" << std::endl;
        return rocblaslt_status_invalid_value;
    }

    F4GemmKernelArgs args(prob);
    void*            argsPtr  = &args;
    size_t           argsSize = sizeof(F4GemmKernelArgs);

    const uint32_t tileM = params->workgroupTile.m;
    const uint32_t tileN = params->workgroupTile.n;

    const uint32_t blockSize = 256;
    dim3           block(blockSize, 1, 1);

    uint32_t tilesM = (static_cast<uint32_t>(prob.n) + tileM - 1) / tileM;
    uint32_t tilesN = (static_cast<uint32_t>(prob.m) + tileN - 1) / tileN;

    dim3 grid(tilesN * blockSize, tilesM, 1);

    void* hipLaunchParams[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                               argsPtr,
                               HIP_LAUNCH_PARAM_BUFFER_SIZE,
                               &argsSize,
                               HIP_LAUNCH_PARAM_END};

    hipFunction_t function;
    if(hipError_t error = module.getHipFunction(function))
    {
        std::cerr << "GemmHipModuleWrapper::getHipFunction failed: " << std::endl
                  << " error: " << hipGetErrorString(error) << std::endl;
        return rocblaslt_status_internal_error;
    }

    if(hipError_t error = hipExtModuleLaunchKernel(function,
                                                     grid.x,
                                                     grid.y,
                                                     grid.z,
                                                     block.x,
                                                     block.y,
                                                     block.z,
                                                     0,
                                                     prob.stream,
                                                     nullptr,
                                                     (void**)&hipLaunchParams,
                                                     nullptr,
                                                     nullptr))
    {
        std::cerr << "hipExtModuleLaunchKernel failed: " << module.getKernelName() << std::endl
                  << " error: " << hipGetErrorString(error) << std::endl;
        return rocblaslt_status_internal_error;
    }

    return rocblaslt_status_success;
}
