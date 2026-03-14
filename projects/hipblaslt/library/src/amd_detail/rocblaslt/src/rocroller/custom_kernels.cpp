// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "custom_kernels.hpp"

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>

std::shared_ptr<GemmKernel> createCustomGemmKernel(const std::string&           customKernelName,
                                                   const KernelType&            kernelType,
                                                   const WorkGroupTileSize&     wgt,
                                                   const std::filesystem::path& path)
{
    auto gemmKernel = std::make_shared<GemmKernel>();

    gemmKernel->params                = std::make_shared<SolutionParameters>();
    gemmKernel->params->kernelType    = kernelType;
    gemmKernel->params->workgroupTile = wgt;

    gemmKernel->module = GemmHipModuleWrapper(customKernelName, path);

    return gemmKernel;
}

std::shared_ptr<GemmKernel> createCustomGemmKernel(const std::string&           customKernelName,
                                                   const KernelType&            kernelType,
                                                   const WorkGroupTileSize&     wgt,
                                                   const std::array<int, 3>&    blockSize,
                                                   const std::filesystem::path& path)
{
    auto gemmKernel             = createCustomGemmKernel(customKernelName, kernelType, wgt, path);
    gemmKernel->customBlockSize = blockSize;
    return gemmKernel;
}

std::shared_ptr<GemmKernel> createCustomGemmKernel(const std::string&           customKernelName,
                                                   const KernelType&            kernelType,
                                                   const WorkGroupTileSize&     wgt,
                                                   const std::array<int, 3>&    blockSize,
                                                   const ShapeCondition&        condition,
                                                   const std::filesystem::path& path)
{
    auto gemmKernel = createCustomGemmKernel(customKernelName, kernelType, wgt, blockSize, path);
    gemmKernel->shapeCondition = condition;
    return gemmKernel;
}

std::shared_ptr<GemmKernel> createCustomGemmKernel(const std::string&           customKernelName,
                                                   const KernelType&            kernelType,
                                                   const WorkGroupTileSize&     wgt,
                                                   const std::array<int, 3>&    blockSize,
                                                   const StaticShape&           shape,
                                                   const std::filesystem::path& path)
{
    auto gemmKernel = createCustomGemmKernel(customKernelName, kernelType, wgt, blockSize, path);
    gemmKernel->staticShape = shape;
    return gemmKernel;
}

std::filesystem::path getCoPath()
{
    std::filesystem::path libraryPath;
    bool                  staticLib = false;

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

    for(bool streamK : {false, true})
    {
        for(bool workgroupMapping : {false, true})
        {
            params.streamK          = streamK;
            params.tailLoops        = true;
            params.workgroupMapping = workgroupMapping;

            mxfp4Kernel.swizzleA = true;

            // 32xN kernels
            params.workgroupTile = {32, 128, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {32, 256, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x256E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {32, 384, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x384E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {32, 512, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x512E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {32, 640, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x640E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {32, 768, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x768E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {32, 896, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x896E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {32, 1024, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_32x1024E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {64, 256, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x256E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {64, 384, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x384E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {64, 512, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x512E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {64, 640, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x640E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {64, 768, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x768E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {64, 896, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x896E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {64, 1024, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            // 96xN kernels
            params.workgroupTile = {96, 128, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_96x128E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {96, 256, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_96x256E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {96, 384, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_96x384E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {96, 512, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_96x512E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {96, 640, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_96x640E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            // 128xN kernels
            params.workgroupTile = {128, 128, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x128E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {128, 256, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {128, 384, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x384E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {128, 512, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            // 160xN kernels
            params.workgroupTile = {160, 128, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_160x128E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {160, 256, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_160x256E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {160, 384, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_160x384E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            // 192xN kernels
            params.workgroupTile = {192, 128, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_192x128E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {192, 256, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_192x256E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            // 224xN kernels
            params.workgroupTile = {224, 128, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_224x128E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {224, 256, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_224x256E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            // 256xN kernels
            params.workgroupTile = {256, 128, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x128E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {256, 256, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            // No B pre-shuffle
            mxfp4Kernel.swizzleA = false;
            params.workgroupTile = {256, 256, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("_ZN5aiter44f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256E",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       getCoPath() / "rr_custom_kernels.co"));

            // Wave kernels (block size {64, 4, 1} from wave_shape (1, 4))

            mxfp4Kernel.swizzleA = true;

            // params.workgroupTile = {64, 64, 256};
            // cache.addKernel(mxfp4Kernel,
            //                 params,
            //                 createCustomGemmKernel("wave_mxfp4_dynamic_gemm_64x64x256",
            //                                        mxfp4Kernel,
            //                                        params.workgroupTile,
            //                                        {256, 2, 1},
            //                                        getCoPath() / "rr_custom_kernels.co"));

            // params.workgroupTile = {128, 32, 256};
            // cache.addKernel(mxfp4Kernel,
            //                 params,
            //                 createCustomGemmKernel("wave_mxfp4_dynamic_gemm_128x32x256",
            //                                        mxfp4Kernel,
            //                                        params.workgroupTile,
            //                                        {256, 2, 1},
            //                                        getCoPath() / "rr_custom_kernels.co"));

            // params.workgroupTile = {224, 160, 256};
            // cache.addKernel(mxfp4Kernel,
            //                 params,
            //                 createCustomGemmKernel("wave_mxfp4_dynamic_gemm_224x160x256",
            //                                        mxfp4Kernel,
            //                                        params.workgroupTile,
            //                                        {64, 4, 1},
            //                                        getCoPath() / "rr_custom_kernels.co"));

            // params.workgroupTile = {256, 160, 256};
            // cache.addKernel(mxfp4Kernel,
            //                 params,
            //                 createCustomGemmKernel("wave_mxfp4_dynamic_gemm_256x160x256",
            //                                        mxfp4Kernel,
            //                                        params.workgroupTile,
            //                                        {256, 2, 1},
            //                                        getCoPath() / "rr_custom_kernels.co"));

            // params.workgroupTile = {256, 192, 256};
            // cache.addKernel(mxfp4Kernel,
            //                 params,
            //                 createCustomGemmKernel("wave_mxfp4_dynamic_gemm_256x192x256",
            //                                        mxfp4Kernel,
            //                                        params.workgroupTile,
            //                                        {64, 4, 1},
            //                                        getCoPath() / "rr_custom_kernels.co"));

            // params.workgroupTile = {256, 224, 256};
            // cache.addKernel(mxfp4Kernel,
            //                 params,
            //                 createCustomGemmKernel("wave_mxfp4_dynamic_gemm_256x224x256",
            //                                        mxfp4Kernel,
            //                                        params.workgroupTile,
            //                                        {64, 4, 1},
            //                                        getCoPath() / "rr_custom_kernels.co"));

            // Static wave kernels (block size {256, 2, 1})
            // Filename shape is Wave M×N×K; StaticShape uses hipBLASLt convention (N,M,K).

            // 64x64x256 macrotile
            params.workgroupTile = {64, 64, 256};
            cache.addKernel(mxfp4Kernel,
                            params,
                            createCustomGemmKernel("wave_mxfp4_static_gemm_64x64x256_256x128x256",
                                                   mxfp4Kernel,
                                                   params.workgroupTile,
                                                   {256, 2, 1},
                                                   StaticShape{128, 256, 256},
                                                   getCoPath() / "rr_custom_kernels.co"));

            // 256x192x256 macrotile
            params.workgroupTile = {256, 192, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("wave_mxfp4_static_gemm_256x192x256_14976x51712x7680",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       {256, 2, 1},
                                       StaticShape{51712, 14976, 7680},
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {256, 192, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("wave_mxfp4_static_gemm_256x192x256_31104x16896x7168",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       {256, 2, 1},
                                       StaticShape{16896, 31104, 7168},
                                       getCoPath() / "rr_custom_kernels.co"));

            params.workgroupTile = {256, 192, 256};
            cache.addKernel(
                mxfp4Kernel,
                params,
                createCustomGemmKernel("wave_mxfp4_static_gemm_256x192x256_43392x13056x1792",
                                       mxfp4Kernel,
                                       params.workgroupTile,
                                       {256, 2, 1},
                                       StaticShape{13056, 43392, 1792},
                                       getCoPath() / "rr_custom_kernels.co"));
        }
    }
}

// Wave GEMM kernel ABI (104 bytes).
// Kernel computes C[M,N] = A[M,K] @ B[N,K]^T (scaled, preshuffle-B).
// hipBLASLt stores col-major: A as K×M, B as K×N, D as N×M.
// We swap A↔B and M↔N so Wave's row-major view matches hipBLASLt's col-major storage.
// FP4 data is 2 elements per byte, so element strides are halved to get byte strides.
struct __attribute__((packed)) WaveGemmKernelArgs
{
    const void* ptr_a; //  0: A data
    const void* ptr_a_scale; //  8: A scale
    const void* ptr_b; // 16: B data
    const void* ptr_b_scale; // 24: B scale
    void*       ptr_c; // 32: C output
    uint64_t    m; // 40
    uint64_t    n; // 48
    uint64_t    k; // 56
    uint64_t    stride_a_dim0; // 64: byte stride
    uint64_t    stride_a_scale_dim0; // 72
    uint64_t    stride_b_dim0; // 80: byte stride
    uint64_t    stride_b_scale_dim0; // 88
    uint64_t    stride_c_dim0; // 96
};
static_assert(sizeof(WaveGemmKernelArgs) == 104, "Wave kernel kernarg must be 104 bytes");

// Static-shape variant: M, N, K are compiled into the assembly as constants,
// so the kernarg segment only contains pointers and strides.
struct __attribute__((packed)) StaticWaveGemmKernelArgs
{
    const void* ptr_a; //  0: A data
    const void* ptr_a_scale; //  8: A scale
    const void* ptr_b; // 16: B data
    const void* ptr_b_scale; // 24: B scale
    void*       ptr_c; // 32: C output
    uint64_t    stride_a_dim0; // 40: byte stride
    uint64_t    stride_a_scale_dim0; // 48
    uint64_t    stride_b_dim0; // 56: byte stride
    uint64_t    stride_b_scale_dim0; // 64
    uint64_t    stride_c_dim0; // 72
};
static_assert(sizeof(StaticWaveGemmKernelArgs) == 80,
              "Static wave kernel kernarg must be 80 bytes");

inline WaveGemmKernelArgs makeWaveGemmKernelArgs(const RocblasltContractionProblem& prob)
{
    WaveGemmKernelArgs w  = {};
    w.ptr_a               = prob.B; // swap
    w.ptr_a_scale         = prob.scaleB; // swap
    w.ptr_b               = prob.A; // swap
    w.ptr_b_scale         = prob.scaleA; // swap
    w.ptr_c               = prob.D;
    w.m                   = prob.n; // swap
    w.n                   = prob.m; // swap
    w.k                   = prob.k;
    w.stride_a_dim0       = prob.col_stride_b / 2; // swap; FP4 byte stride
    w.stride_a_scale_dim0 = prob.k / 32;
    w.stride_b_dim0       = prob.col_stride_a / 2; // swap; FP4 byte stride
    w.stride_b_scale_dim0 = prob.k / 32;
    w.stride_c_dim0       = prob.col_stride_c;
    return w;
}

inline StaticWaveGemmKernelArgs
    makeStaticWaveGemmKernelArgs(const RocblasltContractionProblem& prob)
{
    StaticWaveGemmKernelArgs w = {};
    w.ptr_a                    = prob.B; // swap
    w.ptr_a_scale              = prob.scaleB; // swap
    w.ptr_b                    = prob.A; // swap
    w.ptr_b_scale              = prob.scaleA; // swap
    w.ptr_c                    = prob.D;
    w.stride_a_dim0            = prob.col_stride_b / 2; // swap; FP4 byte stride
    w.stride_a_scale_dim0      = prob.k / 32;
    w.stride_b_dim0            = prob.col_stride_a / 2; // swap; FP4 byte stride
    w.stride_b_scale_dim0      = prob.k / 32;
    w.stride_c_dim0            = prob.col_stride_c;
    return w;
}

// F4 GEMM Kernel Args (AITER)

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
};

// AITER kernel computes D[N,M] = B^T * A instead of C[M,N] = A^T * B
// So we swap A<->B pointers/scales and M<->N dimensions
inline F4GemmKernelArgs makeF4GemmKernelArgs(const RocblasltContractionProblem& prob)
{
    F4GemmKernelArgs a = {};
    a.ptr_D            = prob.D;
    a.ptr_C            = nullptr;
    a.ptr_A            = const_cast<void*>(prob.B); // Swapped: kernel's A = hipBLASLt's B
    a.ptr_B            = const_cast<void*>(prob.A); // Swapped: kernel's B = hipBLASLt's A
    a.alpha            = *static_cast<const float*>(prob.alpha);
    a.beta             = *static_cast<const float*>(prob.beta);
    a.stride_D0        = 0;
    a.stride_D1        = 0;
    a.stride_C0        = static_cast<uint32_t>(prob.col_stride_c);
    a.stride_C1        = 0;
    a.stride_A0        = static_cast<uint32_t>(prob.col_stride_b); // Swapped
    a.stride_A1        = 0;
    a.stride_B0        = static_cast<uint32_t>(prob.col_stride_a); // Swapped
    a.stride_B1        = 0;
    a.M                = static_cast<uint32_t>(prob.n); // Swapped: kernel's M = hipBLASLt's N
    a.N                = static_cast<uint32_t>(prob.m); // Swapped: kernel's N = hipBLASLt's M
    a.K                = static_cast<uint32_t>(prob.k);
    a.ptr_ScaleA       = prob.scaleB; // Swapped
    a.ptr_ScaleB       = prob.scaleA; // Swapped
    a.stride_ScaleA0   = static_cast<uint32_t>(prob.k / 32);
    a.stride_ScaleA1   = 0;
    a.stride_ScaleB0   = static_cast<uint32_t>(prob.k / 32);
    a.stride_ScaleB1   = 0;
    a.log2_k_split     = 0;
    return a;
}

rocblaslt_status runCustomKernel(std::shared_ptr<GemmKernel>        gemm,
                                 const RocblasltContractionProblem& prob)
{
    if(!gemm->module.has_value())
    {
        std::cerr << "runCustomKernel failed: Module not loadable" << std::endl;
        return rocblaslt_status_internal_error;
    }

    if(prob.beta && *static_cast<const float*>(prob.beta) != 0)
    {
        std::cerr << "Kernel only supports when beta is 0" << std::endl;
        return rocblaslt_status_invalid_value;
    }

    const std::string& kernelName   = gemm->module->getKernelName();
    const bool         isWaveKernel = kernelName.rfind("wave", 0) == 0;

    // Build kernel arguments
    static WaveGemmKernelArgs       waveArgsStorage;
    static StaticWaveGemmKernelArgs staticWaveArgsStorage;
    static F4GemmKernelArgs         aiterArgsStorage;

    void*  argsPtr;
    size_t argsSize;

    if(isWaveKernel && gemm->staticShape.has_value())
    {
        staticWaveArgsStorage = makeStaticWaveGemmKernelArgs(prob);
        argsPtr               = &staticWaveArgsStorage;
        argsSize              = sizeof(staticWaveArgsStorage);
    }
    else if(isWaveKernel)
    {
        waveArgsStorage = makeWaveGemmKernelArgs(prob);
        argsPtr         = &waveArgsStorage;
        argsSize        = sizeof(waveArgsStorage);
    }
    else
    {
        aiterArgsStorage = makeF4GemmKernelArgs(prob);
        argsPtr          = &aiterArgsStorage;
        argsSize         = sizeof(aiterArgsStorage);
    }

    // Tile counts (Wave M = hipBLASLt N, Wave N = hipBLASLt M)
    const uint32_t tileM  = gemm->params->workgroupTile.m;
    const uint32_t tileN  = gemm->params->workgroupTile.n;
    const uint32_t tilesM = (static_cast<uint32_t>(prob.n) + tileM - 1) / tileM;
    const uint32_t tilesN = (static_cast<uint32_t>(prob.m) + tileN - 1) / tileN;

    dim3 grid;
    dim3 block;

    if(isWaveKernel)
    {
        if(!gemm->customBlockSize.has_value())
        {
            std::cerr << "runCustomKernel: wave kernel missing customBlockSize" << std::endl;
            return rocblaslt_status_internal_error;
        }
        const auto& bs = *gemm->customBlockSize;
        block.x        = bs[0];
        block.y        = bs[1];
        block.z        = bs[2];

        // hipModuleLaunchKernel: grid = number of blocks
        grid.x = tilesM;
        grid.y = tilesN;
        grid.z = 1;
    }
    else
    {
        const uint32_t blockSize = 256;
        block                    = {blockSize, 1, 1};

        // hipExtModuleLaunchKernel: grid = total threads
        grid.x = tilesN * blockSize;
        grid.y = tilesM;
        grid.z = 1;
    }

    void* hipLaunchParams[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                               argsPtr,
                               HIP_LAUNCH_PARAM_BUFFER_SIZE,
                               &argsSize,
                               HIP_LAUNCH_PARAM_END};

    hipFunction_t function;
    if(hipError_t error = gemm->module->getHipFunction(function))
    {
        std::cerr << "GemmHipModuleWrapper::getHipFunction failed: " << std::endl
                  << " error: " << hipGetErrorString(error) << std::endl;
        return rocblaslt_status_internal_error;
    }

    if(isWaveKernel)
    {
        if(hipError_t error = hipModuleLaunchKernel(function,
                                                    grid.x,
                                                    grid.y,
                                                    grid.z,
                                                    block.x,
                                                    block.y,
                                                    block.z,
                                                    0, // sharedMem
                                                    prob.stream, // stream
                                                    nullptr,
                                                    (void**)&hipLaunchParams))
        {
            std::cerr << "hipModuleLaunchKernel in runCustomKernel failed: " << kernelName
                      << std::endl
                      << " error: " << hipGetErrorString(error) << std::endl;
            return rocblaslt_status_internal_error;
        }
    }
    else
    {
        if(hipError_t error = hipExtModuleLaunchKernel(function,
                                                       grid.x,
                                                       grid.y,
                                                       grid.z,
                                                       block.x,
                                                       block.y,
                                                       block.z,
                                                       0, // sharedMem
                                                       prob.stream, // stream
                                                       nullptr,
                                                       (void**)&hipLaunchParams,
                                                       nullptr, // event
                                                       nullptr // event
                                                       ))
        {
            std::cerr << "hipExtModuleLaunchKernel in runCustomKernel failed: " << kernelName
                      << std::endl
                      << " error: " << hipGetErrorString(error) << std::endl;
            return rocblaslt_status_internal_error;
        }
    }

    return rocblaslt_status_success;
}
