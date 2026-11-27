/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "handle.h"
#include "matrix-transform/gfx950/mxfp6_page24.h"
#include "matrix-transform/matrix_transform.h"
#include "rocblaslt-auxiliary.h"
#include "rocblaslt-types.h"
#include "rocblaslt.h"
#include "rocblaslt_mat_utils.hpp"
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>
#include <functional>
#include <hipblaslt/hipblaslt-export.h>
#include <hipblaslt/hipblaslt-types.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace
{
    // The current device's architecture without target features, for example "gfx950".
    std::optional<std::string> deviceArchName()
    {
        int             deviceId{};
        hipDeviceProp_t props{};
        if(hipGetDevice(&deviceId) != hipSuccess
           || hipGetDeviceProperties(&props, deviceId) != hipSuccess)
        {
            rocblaslt_log_error("deviceArchName",
                                "hipGetDevice/hipGetDeviceProperties",
                                "failed to query device for matrix-transform arch lookup");
            return std::nullopt;
        }

        std::string archName = props.gcnArchName;
        auto        colonPos = archName.find(':');
        if(colonPos != std::string::npos)
            archName = archName.substr(0, colonPos);
        return archName;
    }

    std::optional<std::filesystem::path> transformCodeObjectPath()
    {
        auto archName = deviceArchName();
        if(!archName)
            return std::nullopt;

        auto relpath
            = std::filesystem::path(*archName) / ("hipblasltTransform_" + *archName + ".hsaco");
        auto perArchPath = rocblaslt_find_library_relative_path(relpath);
        if(!perArchPath)
        {
            rocblaslt_log_error("transformCodeObjectPath",
                                "rocblaslt_find_library_relative_path",
                                relpath.string().c_str());
        }
        return perArchPath;
    }

    const std::string& transformCodeObjectFileName()
    {
        static const std::string name = [] {
            auto path = transformCodeObjectPath();
            return path ? path->filename().string() : std::string{};
        }();
        return name;
    }

    TensileLite::hip::SolutionAdapter& transformAdapter()
    {
        using AdapterPtr     = std::unique_ptr<TensileLite::hip::SolutionAdapter>;
        static auto& adapter = []() -> std::vector<AdapterPtr>& {
            static std::vector<AdapterPtr> adapters;
            int                            numDevices{};
            HIP_CHECK_EXC(hipGetDeviceCount(&numDevices));
            for(int i = 0; i < numDevices; ++i)
            {
                adapters.emplace_back(new TensileLite::hip::SolutionAdapter);
            }
            auto coPath = transformCodeObjectPath();
            if(!coPath)
            {
                rocblaslt_log_error("transformAdapter",
                                    "transformCodeObjectPath",
                                    "matrix-transform code object not found under per-base layout");
                return adapters;
            }
            const std::string coFolder = coPath->parent_path().string();
            try
            {
                for(auto& adp : adapters)
                {
                    adp->codeObjectDir(coFolder);
                }
            }
            catch(const std::runtime_error& e)
            {
                rocblaslt_log_error(
                    "transformAdapter", "codeObjectDir", coFolder.c_str());
            }
            return adapters;
        }();

        int deviceId{};
        HIP_CHECK_EXC(hipGetDevice(&deviceId));
        return *adapter.at(deviceId);
    }

    rocblaslt_matrix_layout dummyMatrixLayout()
    {
        static _rocblaslt_matrix_layout layout;
        return &layout;
    }

    template <typename T>
    const T* dummyScalarPtr()
    {
        static T scalar{};
        return &scalar;
    }

    // rocKE MXFP6 packers for gfx950, assembled from device-library/matrix-transform/gfx950. They
    // are built for one shape and bounds-check nothing, so findMxfp6Packer must match that shape
    // exactly before one is launched.
    struct Mxfp6Packer
    {
        const char* kernelName;
        uint32_t    numWorkGroups;
    };

    constexpr char     mxfp6PackerCodeObject[] = "hipblasltTransformPack_gfx950.co";
    constexpr uint64_t mxfp6PackerRows         = 4096; // M or N indices of the operand
    constexpr uint64_t mxfp6PackerK            = 65536;
    constexpr uint64_t mxfp6PackerScaleBlock   = 32;

    bool isMxfp6Pack(const rocblaslt_matrix_transform_desc* desc,
                     rocblaslt_matrix_layout                layoutA,
                     rocblaslt_matrix_layout                layoutB,
                     rocblaslt_matrix_layout                layoutC)
    {
        return desc->scaleModeC != HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F
               || isFp6TiledOrder(layoutA->order) || isFp6TiledOrder(layoutB->order)
               || isFp6TiledOrder(layoutC->order);
    }

    // The packer that writes layoutC from layoutA, or nullptr if there is none.
    const Mxfp6Packer* findMxfp6Packer(const rocblaslt_matrix_transform_desc* desc,
                                       rocblaslt_matrix_layout                layoutA,
                                       rocblaslt_matrix_layout                layoutC)
    {
        static constexpr Mxfp6Packer data{
            "rocke_mxfp6_pack_op_4096x65536_r256_w128t4_xcd_c0c1p_nosc", 4096};
        static constexpr Mxfp6Packer scales{
            "rocke_mxfp6_pack_scales_4096_0x65536_16x16x128_xcd_nts", 512};

        const bool sameShape = layoutA->type == layoutC->type && layoutA->m == layoutC->m
                               && layoutA->n == layoutC->n && layoutA->ld == layoutC->ld
                               && layoutA->batch_count == 1 && layoutC->batch_count == 1;
        const bool packed = static_cast<uint64_t>(layoutA->ld) == layoutA->m;
        if(!sameShape || !packed || layoutA->order != HIPBLASLT_ORDER_COL
           || layoutA->n != mxfp6PackerRows || desc->opA != HIPBLAS_OP_N)
        {
            return nullptr;
        }

        if(desc->scaleModeC == HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F)
        {
            if(isFp6Datatype(layoutA->type) && layoutC->order == HIPBLASLT_ORDER_COL256_R128_P16_EXT
               && layoutA->m == mxfp6PackerK)
            {
                return &data;
            }
        }
        else if(desc->scaleModeC == HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT)
        {
            if(layoutA->type == HIP_R_8F_UE8M0 && layoutC->order == HIPBLASLT_ORDER_COL
               && layoutA->m == mxfp6PackerK / mxfp6PackerScaleBlock)
            {
                return &scales;
            }
        }
        return nullptr;
    }

    // The near-die placement kernels of device-library/matrix-transform/gfx950/mxfp6_page24.cpp.
    // Like the packers, they bounds-check nothing.
    constexpr char     page24CodeObject[] = "hipblasltTransformPage24_gfx950.hsaco";
    constexpr uint32_t page24KTiles       = mxfp6PackerK / 128;
    constexpr uint32_t page24Tiles        = mxfp6PackerRows / 256;

    enum class Page24Write
    {
        None,
        SplitData, // B: HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_SPLIT_EXT
        DuplicatedData, // A: HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_EXT
        DuplicatedScales, // A: HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_PAGE24_EXT
    };

    // What the near-die kernels write to layoutC from layoutA: 48-byte tiles (order 105) to a
    // near-die order, or 128x4 scales (scale mode 1018, as a COL UE8M0 matrix) to near-die scales.
    Page24Write findPage24Write(const rocblaslt_matrix_transform_desc* desc,
                                rocblaslt_matrix_layout                layoutA,
                                rocblaslt_matrix_layout                layoutC)
    {
        const bool sameShape = layoutA->type == layoutC->type && layoutA->m == layoutC->m
                               && layoutA->n == layoutC->n && layoutA->ld == layoutC->ld
                               && layoutA->batch_count == 1 && layoutC->batch_count == 1;
        const bool packed = static_cast<uint64_t>(layoutA->ld) == layoutA->m;
        if(!sameShape || !packed || layoutA->n != mxfp6PackerRows || desc->opA != HIPBLAS_OP_N)
        {
            return Page24Write::None;
        }

        if(desc->scaleModeC == HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F)
        {
            if(isFp6Datatype(layoutA->type) && layoutA->order == HIPBLASLT_ORDER_COL256_R128_P48_EXT
               && layoutA->m == mxfp6PackerK)
            {
                if(layoutC->order == HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_SPLIT_EXT)
                    return Page24Write::SplitData;
                if(layoutC->order == HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_EXT)
                    return Page24Write::DuplicatedData;
            }
        }
        else if(desc->scaleModeC == HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_PAGE24_EXT)
        {
            if(layoutA->type == HIP_R_8F_UE8M0 && layoutA->order == HIPBLASLT_ORDER_COL
               && layoutC->order == HIPBLASLT_ORDER_COL
               && layoutA->m == mxfp6PackerK / mxfp6PackerScaleBlock)
            {
                return Page24Write::DuplicatedScales;
            }
        }
        return Page24Write::None;
    }

    // Set by hipblaslt_page24_driver_for_test: replaces the contents of
    // /sys/module/amdgpu/version.
    std::mutex                 page24DriverMutex;
    std::optional<std::string> page24DriverOverride;

    std::string amdgpuVersion()
    {
        {
            std::lock_guard<std::mutex> lock(page24DriverMutex);
            if(page24DriverOverride)
                return *page24DriverOverride;
        }
        std::ifstream file("/sys/module/amdgpu/version");
        std::string   version;
        file >> version;
        return version;
    }

    rocblaslt_status page24Place(Page24Write write, const void* A, void* C, hipStream_t stream)
    {
        if(deviceArchName() != "gfx950")
        {
            return rocblaslt_status_arch_mismatch;
        }
        const std::string version = amdgpuVersion();
        if(!TensileLite::hip::page24RuleHolds(version))
        {
            log_error(__func__,
                      "the near-die placement needs amdgpu 6.19 or newer; got amdgpu version '",
                      version,
                      "'");
            return rocblaslt_status_not_supported;
        }

        const bool                   scales = write == Page24Write::DuplicatedScales;
        TensileLite::KernelArguments kArgs(false);
        kArgs.appendAligned("dst", static_cast<const void*>(C));
        kArgs.appendAligned("src", A);
        if(!scales)
        {
            kArgs.appendAligned("kTiles", page24KTiles);
            kArgs.appendAligned("split", uint32_t(write == Page24Write::SplitData));
        }

        constexpr uint32_t NUM_WORKITEMS{mxfp6Page24Threads};
        const uint32_t     numWorkGroups = scales ? page24KTiles : page24Tiles * page24KTiles;
        TensileLite::KernelInvocation invocation{scales ? "hipblaslt_mxfp6_page24_scales"
                                                        : "hipblaslt_mxfp6_page24_data",
                                                 page24CodeObject,
                                                 false,
                                                 {1, 1, 1},
                                                 {NUM_WORKITEMS, 1, 1},
                                                 {numWorkGroups, 1, 1},
                                                 {numWorkGroups * NUM_WORKITEMS, 1, 1},
                                                 0,
                                                 kArgs};
        const auto err = transformAdapter().launchKernel(invocation, stream, nullptr, nullptr);
        return (err == hipSuccess) ? rocblaslt_status_success : rocblaslt_status_internal_error;
    }

    rocblaslt_status mxfp6Pack(const rocblaslt_matrix_transform_desc* desc,
                               const void*                            A,
                               rocblaslt_matrix_layout                layoutA,
                               const void*                            B,
                               void*                                  C,
                               rocblaslt_matrix_layout                layoutC,
                               hipStream_t                            stream)
    {
        const Page24Write  page24 = findPage24Write(desc, layoutA, layoutC);
        const Mxfp6Packer* packer = findMxfp6Packer(desc, layoutA, layoutC);
        if((!packer && page24 == Page24Write::None) || !A || B)
        {
            return rocblaslt_status_invalid_value;
        }

        if(page24 != Page24Write::None)
        {
            return page24Place(page24, A, C, stream);
        }

        if(deviceArchName() != "gfx950")
        {
            return rocblaslt_status_arch_mismatch;
        }

        // Each packer takes a second operand's source and destination, which it never reads.
        const void* unused = nullptr;

        TensileLite::KernelArguments kArgs(false);
        kArgs.appendAligned("src", A);
        kArgs.appendAligned("otherSrc", unused);
        kArgs.appendAligned("dst", static_cast<const void*>(C));
        kArgs.appendAligned("otherDst", unused);

        constexpr uint32_t            NUM_WORKITEMS{256};
        TensileLite::KernelInvocation invocation{packer->kernelName,
                                                 mxfp6PackerCodeObject,
                                                 false,
                                                 {1, 1, 1},
                                                 {NUM_WORKITEMS, 1, 1},
                                                 {packer->numWorkGroups, 1, 1},
                                                 {packer->numWorkGroups * NUM_WORKITEMS, 1, 1},
                                                 0,
                                                 kArgs};
        const auto err = transformAdapter().launchKernel(invocation, stream, nullptr, nullptr);
        return (err == hipSuccess) ? rocblaslt_status_success : rocblaslt_status_internal_error;
    }

    template <typename DType,
              typename ScaleType,
              bool     RowMajA,
              bool     RowMajB,
              bool     RowMajC,
              uint32_t NumThreadsM,
              uint32_t NumThreadsN,
              uint32_t VectorWidth>
    hipError_t launchTransformKernel(DType*             c,
                                     const DType*       a,
                                     const DType*       b,
                                     const ScaleType*   alphaPtr,
                                     const ScaleType*   betaPtr,
                                     bool               scalarInDevice,
                                     uint32_t           m,
                                     uint32_t           n,
                                     uint32_t           ldA,
                                     uint32_t           ldB,
                                     uint32_t           ldC,
                                     uint32_t           batchSize,
                                     uint32_t           batchStride,
                                     bool               transA,
                                     bool               transB,
                                     hipStream_t        stream,
                                     const std::string& kernelName)
    {
        constexpr auto TileM        = RowMajC ? NumThreadsM : NumThreadsM * VectorWidth;
        constexpr auto TileN        = RowMajC ? NumThreadsN * VectorWidth : NumThreadsN;
        const auto     numWg        = (m / TileM + !!(m % TileM)) * (n / TileN + !!(n % TileN));
        constexpr auto numWorkitems = NumThreadsM * NumThreadsN;
        TensileLite::KernelArguments kArgs(false);

        if(scalarInDevice)
        {
            kArgs.appendAligned("c", c);
            kArgs.appendAligned("a", a);
            kArgs.appendAligned("b", b);
            kArgs.appendAligned("alpha", ScaleType(1));
            kArgs.appendAligned("alphaPtr", alphaPtr);
            kArgs.appendAligned("beta", ScaleType(1));
            kArgs.appendAligned("betaPtr", betaPtr);
            kArgs.appendAligned("m", m);
            kArgs.appendAligned("n", n);
            kArgs.appendAligned("ldA", ldA);
            kArgs.appendAligned("ldB", ldB);
            kArgs.appendAligned("ldC", ldC);
            kArgs.appendAligned("batchStride", batchStride);
            kArgs.appendAligned("transA", transA);
            kArgs.appendAligned("transB", transB);
        }
        else
        {
            if(!alphaPtr)
            {
                alphaPtr = dummyScalarPtr<ScaleType>();
            }

            if(!betaPtr)
            {
                betaPtr = dummyScalarPtr<ScaleType>();
            }

            const ScaleType* nullScalePtr = nullptr;

            kArgs.appendAligned("c", c);
            kArgs.appendAligned("a", a);
            kArgs.appendAligned("b", b);
            kArgs.appendAligned("alpha", *alphaPtr);
            kArgs.appendAligned("alphaPtr", nullScalePtr);
            kArgs.appendAligned("beta", *betaPtr);
            kArgs.appendAligned("betaPtr", nullScalePtr);
            kArgs.appendAligned("m", m);
            kArgs.appendAligned("n", n);
            kArgs.appendAligned("ldA", ldA);
            kArgs.appendAligned("ldB", ldB);
            kArgs.appendAligned("ldC", ldC);
            kArgs.appendAligned("batchStride", batchStride);
            kArgs.appendAligned("transA", transA);
            kArgs.appendAligned("transB", transB);
        }

        constexpr auto                NUM_WORKITEMS{NumThreadsM * NumThreadsN};
        TensileLite::KernelInvocation invocation{kernelName,
                                                 transformCodeObjectFileName(),
                                                 false,
                                                 {1, 1, 1},
                                                 {NUM_WORKITEMS, 1, 1},
                                                 {numWg, 1, batchSize},
                                                 {numWg * NUM_WORKITEMS, 1, batchSize},
                                                 0,
                                                 kArgs};
        auto&                         adapter = transformAdapter();
        return adapter.launchKernel(invocation, stream, nullptr, nullptr);
    }

// Generate combination of MEMORY ORDER and RowMaj{A/B/C} = true/false
#define GEN_COMBINATION(                                                                        \
    _DTYPE, _SCALETYPE, _DType, _ScaleType, _NumThreadsM, _NumThreadsN, _VectorWidth)           \
    {std::make_tuple(_DTYPE,                                                                    \
                     _SCALETYPE,                                                                \
                     HIPBLASLT_ORDER_COL,                                                       \
                     HIPBLASLT_ORDER_COL,                                                       \
                     HIPBLASLT_ORDER_COL,                                                       \
                     _VectorWidth),                                                             \
     [](void*              c,                                                                   \
        const void*        a,                                                                   \
        const void*        b,                                                                   \
        const void*        alpha,                                                               \
        const void*        beta,                                                                \
        bool               scalarInDevice,                                                      \
        uint32_t           m,                                                                   \
        uint32_t           n,                                                                   \
        uint32_t           ldA,                                                                 \
        uint32_t           ldB,                                                                 \
        uint32_t           ldC,                                                                 \
        uint32_t           batchSize,                                                           \
        uint32_t           batchStride,                                                         \
        bool               opA,                                                                 \
        bool               opB,                                                                 \
        hipStream_t        stream,                                                              \
        const std::string& kernelName) {                                                        \
         return launchTransformKernel<_DType,                                                   \
                                      _ScaleType,                                               \
                                      false,                                                    \
                                      false,                                                    \
                                      false,                                                    \
                                      _NumThreadsM,                                             \
                                      _NumThreadsN,                                             \
                                      _VectorWidth>(static_cast<_DType*>(c),                    \
                                                    static_cast<const _DType*>(a),              \
                                                    static_cast<const _DType*>(b),              \
                                                    reinterpret_cast<const _ScaleType*>(alpha), \
                                                    reinterpret_cast<const _ScaleType*>(beta),  \
                                                    scalarInDevice,                             \
                                                    m,                                          \
                                                    n,                                          \
                                                    ldA,                                        \
                                                    ldB,                                        \
                                                    ldC,                                        \
                                                    batchSize,                                  \
                                                    batchStride,                                \
                                                    opA,                                        \
                                                    opB,                                        \
                                                    stream,                                     \
                                                    kernelName);                                \
     }},                                                                                        \
        {std::make_tuple(_DTYPE,                                                                \
                         _SCALETYPE,                                                            \
                         HIPBLASLT_ORDER_COL,                                                   \
                         HIPBLASLT_ORDER_COL,                                                   \
                         HIPBLASLT_ORDER_ROW,                                                   \
                         _VectorWidth),                                                         \
         [](void*              c,                                                               \
            const void*        a,                                                               \
            const void*        b,                                                               \
            const void*        alpha,                                                           \
            const void*        beta,                                                            \
            bool               scalarInDevice,                                                  \
            uint32_t           m,                                                               \
            uint32_t           n,                                                               \
            uint32_t           ldA,                                                             \
            uint32_t           ldB,                                                             \
            uint32_t           ldC,                                                             \
            uint32_t           batchSize,                                                       \
            uint32_t           batchStride,                                                     \
            bool               opA,                                                             \
            bool               opB,                                                             \
            hipStream_t        stream,                                                          \
            const std::string& kernelName) {                                                    \
             return launchTransformKernel<_DType,                                               \
                                          _ScaleType,                                           \
                                          false,                                                \
                                          false,                                                \
                                          true,                                                 \
                                          _NumThreadsM,                                         \
                                          _NumThreadsN,                                         \
                                          _VectorWidth>(                                        \
                 static_cast<_DType*>(c),                                                       \
                 static_cast<const _DType*>(a),                                                 \
                 static_cast<const _DType*>(b),                                                 \
                 reinterpret_cast<const _ScaleType*>(alpha),                                    \
                 reinterpret_cast<const _ScaleType*>(beta),                                     \
                 scalarInDevice,                                                                \
                 m,                                                                             \
                 n,                                                                             \
                 ldA,                                                                           \
                 ldB,                                                                           \
                 ldC,                                                                           \
                 batchSize,                                                                     \
                 batchStride,                                                                   \
                 opA,                                                                           \
                 opB,                                                                           \
                 stream,                                                                        \
                 kernelName);                                                                   \
         }},                                                                                    \
        {std::make_tuple(_DTYPE,                                                                \
                         _SCALETYPE,                                                            \
                         HIPBLASLT_ORDER_COL,                                                   \
                         HIPBLASLT_ORDER_ROW,                                                   \
                         HIPBLASLT_ORDER_COL,                                                   \
                         _VectorWidth),                                                         \
         [](void*              c,                                                               \
            const void*        a,                                                               \
            const void*        b,                                                               \
            const void*        alpha,                                                           \
            const void*        beta,                                                            \
            bool               scalarInDevice,                                                  \
            uint32_t           m,                                                               \
            uint32_t           n,                                                               \
            uint32_t           ldA,                                                             \
            uint32_t           ldB,                                                             \
            uint32_t           ldC,                                                             \
            uint32_t           batchSize,                                                       \
            uint32_t           batchStride,                                                     \
            bool               opA,                                                             \
            bool               opB,                                                             \
            hipStream_t        stream,                                                          \
            const std::string& kernelName) {                                                    \
             return launchTransformKernel<_DType,                                               \
                                          _ScaleType,                                           \
                                          false,                                                \
                                          true,                                                 \
                                          false,                                                \
                                          _NumThreadsM,                                         \
                                          _NumThreadsN,                                         \
                                          _VectorWidth>(                                        \
                 static_cast<_DType*>(c),                                                       \
                 static_cast<const _DType*>(a),                                                 \
                 static_cast<const _DType*>(b),                                                 \
                 reinterpret_cast<const _ScaleType*>(alpha),                                    \
                 reinterpret_cast<const _ScaleType*>(beta),                                     \
                 scalarInDevice,                                                                \
                 m,                                                                             \
                 n,                                                                             \
                 ldA,                                                                           \
                 ldB,                                                                           \
                 ldC,                                                                           \
                 batchSize,                                                                     \
                 batchStride,                                                                   \
                 opA,                                                                           \
                 opB,                                                                           \
                 stream,                                                                        \
                 kernelName);                                                                   \
         }},                                                                                    \
        {std::make_tuple(_DTYPE,                                                                \
                         _SCALETYPE,                                                            \
                         HIPBLASLT_ORDER_COL,                                                   \
                         HIPBLASLT_ORDER_ROW,                                                   \
                         HIPBLASLT_ORDER_ROW,                                                   \
                         _VectorWidth),                                                         \
         [](void*              c,                                                               \
            const void*        a,                                                               \
            const void*        b,                                                               \
            const void*        alpha,                                                           \
            const void*        beta,                                                            \
            bool               scalarInDevice,                                                  \
            uint32_t           m,                                                               \
            uint32_t           n,                                                               \
            uint32_t           ldA,                                                             \
            uint32_t           ldB,                                                             \
            uint32_t           ldC,                                                             \
            uint32_t           batchSize,                                                       \
            uint32_t           batchStride,                                                     \
            bool               opA,                                                             \
            bool               opB,                                                             \
            hipStream_t        stream,                                                          \
            const std::string& kernelName) {                                                    \
             return launchTransformKernel<_DType,                                               \
                                          _ScaleType,                                           \
                                          false,                                                \
                                          true,                                                 \
                                          true,                                                 \
                                          _NumThreadsM,                                         \
                                          _NumThreadsN,                                         \
                                          _VectorWidth>(                                        \
                 static_cast<_DType*>(c),                                                       \
                 static_cast<const _DType*>(a),                                                 \
                 static_cast<const _DType*>(b),                                                 \
                 reinterpret_cast<const _ScaleType*>(alpha),                                    \
                 reinterpret_cast<const _ScaleType*>(beta),                                     \
                 scalarInDevice,                                                                \
                 m,                                                                             \
                 n,                                                                             \
                 ldA,                                                                           \
                 ldB,                                                                           \
                 ldC,                                                                           \
                 batchSize,                                                                     \
                 batchStride,                                                                   \
                 opA,                                                                           \
                 opB,                                                                           \
                 stream,                                                                        \
                 kernelName);                                                                   \
         }},                                                                                    \
        {std::make_tuple(_DTYPE,                                                                \
                         _SCALETYPE,                                                            \
                         HIPBLASLT_ORDER_ROW,                                                   \
                         HIPBLASLT_ORDER_COL,                                                   \
                         HIPBLASLT_ORDER_COL,                                                   \
                         _VectorWidth),                                                         \
         [](void*              c,                                                               \
            const void*        a,                                                               \
            const void*        b,                                                               \
            const void*        alpha,                                                           \
            const void*        beta,                                                            \
            bool               scalarInDevice,                                                  \
            uint32_t           m,                                                               \
            uint32_t           n,                                                               \
            uint32_t           ldA,                                                             \
            uint32_t           ldB,                                                             \
            uint32_t           ldC,                                                             \
            uint32_t           batchSize,                                                       \
            uint32_t           batchStride,                                                     \
            bool               opA,                                                             \
            bool               opB,                                                             \
            hipStream_t        stream,                                                          \
            const std::string& kernelName) {                                                    \
             return launchTransformKernel<_DType,                                               \
                                          _ScaleType,                                           \
                                          true,                                                 \
                                          false,                                                \
                                          false,                                                \
                                          _NumThreadsM,                                         \
                                          _NumThreadsN,                                         \
                                          _VectorWidth>(                                        \
                 static_cast<_DType*>(c),                                                       \
                 static_cast<const _DType*>(a),                                                 \
                 static_cast<const _DType*>(b),                                                 \
                 reinterpret_cast<const _ScaleType*>(alpha),                                    \
                 reinterpret_cast<const _ScaleType*>(beta),                                     \
                 scalarInDevice,                                                                \
                 m,                                                                             \
                 n,                                                                             \
                 ldA,                                                                           \
                 ldB,                                                                           \
                 ldC,                                                                           \
                 batchSize,                                                                     \
                 batchStride,                                                                   \
                 opA,                                                                           \
                 opB,                                                                           \
                 stream,                                                                        \
                 kernelName);                                                                   \
         }},                                                                                    \
        {std::make_tuple(_DTYPE,                                                                \
                         _SCALETYPE,                                                            \
                         HIPBLASLT_ORDER_ROW,                                                   \
                         HIPBLASLT_ORDER_COL,                                                   \
                         HIPBLASLT_ORDER_ROW,                                                   \
                         _VectorWidth),                                                         \
         [](void*              c,                                                               \
            const void*        a,                                                               \
            const void*        b,                                                               \
            const void*        alpha,                                                           \
            const void*        beta,                                                            \
            bool               scalarInDevice,                                                  \
            uint32_t           m,                                                               \
            uint32_t           n,                                                               \
            uint32_t           ldA,                                                             \
            uint32_t           ldB,                                                             \
            uint32_t           ldC,                                                             \
            uint32_t           batchSize,                                                       \
            uint32_t           batchStride,                                                     \
            bool               opA,                                                             \
            bool               opB,                                                             \
            hipStream_t        stream,                                                          \
            const std::string& kernelName) {                                                    \
             return launchTransformKernel<_DType,                                               \
                                          _ScaleType,                                           \
                                          true,                                                 \
                                          false,                                                \
                                          true,                                                 \
                                          _NumThreadsM,                                         \
                                          _NumThreadsN,                                         \
                                          _VectorWidth>(                                        \
                 static_cast<_DType*>(c),                                                       \
                 static_cast<const _DType*>(a),                                                 \
                 static_cast<const _DType*>(b),                                                 \
                 reinterpret_cast<const _ScaleType*>(alpha),                                    \
                 reinterpret_cast<const _ScaleType*>(beta),                                     \
                 scalarInDevice,                                                                \
                 m,                                                                             \
                 n,                                                                             \
                 ldA,                                                                           \
                 ldB,                                                                           \
                 ldC,                                                                           \
                 batchSize,                                                                     \
                 batchStride,                                                                   \
                 opA,                                                                           \
                 opB,                                                                           \
                 stream,                                                                        \
                 kernelName);                                                                   \
         }},                                                                                    \
        {std::make_tuple(_DTYPE,                                                                \
                         _SCALETYPE,                                                            \
                         HIPBLASLT_ORDER_ROW,                                                   \
                         HIPBLASLT_ORDER_ROW,                                                   \
                         HIPBLASLT_ORDER_COL,                                                   \
                         _VectorWidth),                                                         \
         [](void*              c,                                                               \
            const void*        a,                                                               \
            const void*        b,                                                               \
            const void*        alpha,                                                           \
            const void*        beta,                                                            \
            bool               scalarInDevice,                                                  \
            uint32_t           m,                                                               \
            uint32_t           n,                                                               \
            uint32_t           ldA,                                                             \
            uint32_t           ldB,                                                             \
            uint32_t           ldC,                                                             \
            uint32_t           batchSize,                                                       \
            uint32_t           batchStride,                                                     \
            bool               opA,                                                             \
            bool               opB,                                                             \
            hipStream_t        stream,                                                          \
            const std::string& kernelName) {                                                    \
             return launchTransformKernel<_DType,                                               \
                                          _ScaleType,                                           \
                                          true,                                                 \
                                          true,                                                 \
                                          false,                                                \
                                          _NumThreadsM,                                         \
                                          _NumThreadsN,                                         \
                                          _VectorWidth>(                                        \
                 static_cast<_DType*>(c),                                                       \
                 static_cast<const _DType*>(a),                                                 \
                 static_cast<const _DType*>(b),                                                 \
                 reinterpret_cast<const _ScaleType*>(alpha),                                    \
                 reinterpret_cast<const _ScaleType*>(beta),                                     \
                 scalarInDevice,                                                                \
                 m,                                                                             \
                 n,                                                                             \
                 ldA,                                                                           \
                 ldB,                                                                           \
                 ldC,                                                                           \
                 batchSize,                                                                     \
                 batchStride,                                                                   \
                 opA,                                                                           \
                 opB,                                                                           \
                 stream,                                                                        \
                 kernelName);                                                                   \
         }},                                                                                    \
    {                                                                                           \
        std::make_tuple(_DTYPE,                                                                 \
                        _SCALETYPE,                                                             \
                        HIPBLASLT_ORDER_ROW,                                                    \
                        HIPBLASLT_ORDER_ROW,                                                    \
                        HIPBLASLT_ORDER_ROW,                                                    \
                        _VectorWidth),                                                          \
            [](void*              c,                                                            \
               const void*        a,                                                            \
               const void*        b,                                                            \
               const void*        alpha,                                                        \
               const void*        beta,                                                         \
               bool               scalarInDevice,                                               \
               uint32_t           m,                                                            \
               uint32_t           n,                                                            \
               uint32_t           ldA,                                                          \
               uint32_t           ldB,                                                          \
               uint32_t           ldC,                                                          \
               uint32_t           batchSize,                                                    \
               uint32_t           batchStride,                                                  \
               bool               opA,                                                          \
               bool               opB,                                                          \
               hipStream_t        stream,                                                       \
               const std::string& kernelName) {                                                 \
                return launchTransformKernel<_DType,                                            \
                                             _ScaleType,                                        \
                                             true,                                              \
                                             true,                                              \
                                             true,                                              \
                                             _NumThreadsM,                                      \
                                             _NumThreadsN,                                      \
                                             _VectorWidth>(                                     \
                    static_cast<_DType*>(c),                                                    \
                    static_cast<const _DType*>(a),                                              \
                    static_cast<const _DType*>(b),                                              \
                    reinterpret_cast<const _ScaleType*>(alpha),                                 \
                    reinterpret_cast<const _ScaleType*>(beta),                                  \
                    scalarInDevice,                                                             \
                    m,                                                                          \
                    n,                                                                          \
                    ldA,                                                                        \
                    ldB,                                                                        \
                    ldC,                                                                        \
                    batchSize,                                                                  \
                    batchStride,                                                                \
                    opA,                                                                        \
                    opB,                                                                        \
                    stream,                                                                     \
                    kernelName);                                                                \
            }                                                                                   \
    }

    using MatrixTransformKernelKey    = std::tuple<hipDataType,
                                                   hipDataType,
                                                   hipblasLtOrder_t,
                                                   hipblasLtOrder_t,
                                                   hipblasLtOrder_t,
                                                   size_t>;
    using MatrixTransformFunction     = std::function<hipError_t(void*,
                                                             const void*,
                                                             const void*,
                                                             const void*,
                                                             const void*,
                                                             bool,
                                                             uint32_t,
                                                             uint32_t,
                                                             uint32_t,
                                                             uint32_t,
                                                             uint32_t,
                                                             uint32_t,
                                                             uint32_t,
                                                             bool,
                                                             bool,
                                                             hipStream_t,
                                                             const std::string&)>;
    using MatrixTransformFunctionName = std::string;
    // disable clang-format of compact view.
    // clang-format off
    std::map<MatrixTransformKernelKey, MatrixTransformFunctionName> transformKernelNames{
        {std::make_tuple(HIP_R_32F, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(S, S, 1, 1, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_32F, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(S, S, 1, 1, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_32F, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(S, S, 1, 1, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_32F, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(S, S, 1, 1, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_32F, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(S, S, 1, 0, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_32F, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(S, S, 1, 0, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_32F, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(S, S, 1, 0, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_32F, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(S, S, 1, 0, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_32F, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(S, S, 0, 1, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_32F, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(S, S, 0, 1, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_32F, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(S, S, 0, 1, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_32F, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(S, S, 0, 1, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_32F, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(S, S, 0, 0, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_32F, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(S, S, 0, 0, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_32F, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(S, S, 0, 0, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_32F, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(S, S, 0, 0, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_16F, HIP_R_16F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(H, H, 1, 1, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_16F, HIP_R_16F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(H, H, 1, 1, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_16F, HIP_R_16F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(H, H, 1, 1, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_16F, HIP_R_16F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(H, H, 1, 1, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_16F, HIP_R_16F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(H, H, 1, 0, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_16F, HIP_R_16F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(H, H, 1, 0, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_16F, HIP_R_16F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(H, H, 1, 0, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_16F, HIP_R_16F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(H, H, 1, 0, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_16F, HIP_R_16F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(H, H, 0, 1, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_16F, HIP_R_16F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(H, H, 0, 1, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_16F, HIP_R_16F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(H, H, 0, 1, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_16F, HIP_R_16F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(H, H, 0, 1, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_16F, HIP_R_16F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(H, H, 0, 0, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_16F, HIP_R_16F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(H, H, 0, 0, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_16F, HIP_R_16F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(H, H, 0, 0, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_16F, HIP_R_16F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(H, H, 0, 0, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_16F, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(H, S, 1, 1, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_16F, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(H, S, 1, 1, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_16F, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(H, S, 1, 1, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_16F, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(H, S, 1, 1, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_16F, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(H, S, 1, 0, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_16F, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(H, S, 1, 0, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_16F, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(H, S, 1, 0, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_16F, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(H, S, 1, 0, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_16F, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(H, S, 0, 1, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_16F, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(H, S, 0, 1, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_16F, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(H, S, 0, 1, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_16F, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(H, S, 0, 1, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_16F, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(H, S, 0, 0, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_16F, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(H, S, 0, 0, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_16F, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(H, S, 0, 0, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_16F, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(H, S, 0, 0, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_16BF, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(BF16, S, 1, 1, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_16BF, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(BF16, S, 1, 1, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_16BF, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(BF16, S, 1, 1, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_16BF, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(BF16, S, 1, 1, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_16BF, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(BF16, S, 1, 0, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_16BF, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(BF16, S, 1, 0, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_16BF, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(BF16, S, 1, 0, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_16BF, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(BF16, S, 1, 0, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_16BF, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(BF16, S, 0, 1, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_16BF, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(BF16, S, 0, 1, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_16BF, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(BF16, S, 0, 1, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_16BF, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(BF16, S, 0, 1, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_16BF, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(BF16, S, 0, 0, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_16BF, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(BF16, S, 0, 0, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_16BF, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(BF16, S, 0, 0, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_16BF, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(BF16, S, 0, 0, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_8I, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(I8, S, 1, 1, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_8I, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(I8, S, 1, 1, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_8I, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(I8, S, 1, 1, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_8I, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(I8, S, 1, 1, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_8I, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(I8, S, 1, 0, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_8I, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(I8, S, 1, 0, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_8I, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(I8, S, 1, 0, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_8I, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(I8, S, 1, 0, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_8I, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(I8, S, 0, 1, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_8I, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(I8, S, 0, 1, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_8I, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(I8, S, 0, 1, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_8I, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(I8, S, 0, 1, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_8I, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(I8, S, 0, 0, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_8I, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(I8, S, 0, 0, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_8I, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(I8, S, 0, 0, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_8I, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(I8, S, 0, 0, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_32I, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(I32, S, 1, 1, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_32I, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(I32, S, 1, 1, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_32I, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(I32, S, 1, 1, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_32I, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(I32, S, 1, 1, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_32I, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(I32, S, 1, 0, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_32I, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(I32, S, 1, 0, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_32I, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(I32, S, 1, 0, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_32I, HIP_R_32F, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(I32, S, 1, 0, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_32I, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(I32, S, 0, 1, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_32I, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(I32, S, 0, 1, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_32I, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(I32, S, 0, 1, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_32I, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(I32, S, 0, 1, 0, 16, 16, 4))},
        {std::make_tuple(HIP_R_32I, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 1), TO_STRING(TRANSFORM_FUNC_NAME(I32, S, 0, 0, 1, 16, 16, 1))},
        {std::make_tuple(HIP_R_32I, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_ROW, 4), TO_STRING(TRANSFORM_FUNC_NAME(I32, S, 0, 0, 1, 16, 16, 4))},
        {std::make_tuple(HIP_R_32I, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 1), TO_STRING(TRANSFORM_FUNC_NAME(I32, S, 0, 0, 0, 16, 16, 1))},
        {std::make_tuple(HIP_R_32I, HIP_R_32F, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, HIPBLASLT_ORDER_COL, 4), TO_STRING(TRANSFORM_FUNC_NAME(I32, S, 0, 0, 0, 16, 16, 4))}
    };
    // clang-format on

    std::map<MatrixTransformKernelKey, MatrixTransformFunction> transformKernels{
        {// vector width = 4
         GEN_COMBINATION(HIP_R_32F, HIP_R_32F, hipblasLtFloat, hipblasLtFloat, 16, 16, 4),
         GEN_COMBINATION(HIP_R_16F, HIP_R_16F, hipblasLtHalf, hipblasLtHalf, 16, 16, 4),
         GEN_COMBINATION(HIP_R_16F, HIP_R_32F, hipblasLtHalf, hipblasLtFloat, 16, 16, 4),
         GEN_COMBINATION(HIP_R_16BF, HIP_R_32F, hipblasLtBfloat16, hipblasLtFloat, 16, 16, 4),
         GEN_COMBINATION(HIP_R_8I, HIP_R_32F, hipblasLtInt8, hipblasLtFloat, 16, 16, 4),
         GEN_COMBINATION(HIP_R_32I, HIP_R_32F, hipblasLtInt32, hipblasLtFloat, 16, 16, 4),

         // vector width = 1
         GEN_COMBINATION(HIP_R_32F, HIP_R_32F, hipblasLtFloat, hipblasLtFloat, 16, 16, 1),
         GEN_COMBINATION(HIP_R_16F, HIP_R_16F, hipblasLtHalf, hipblasLtHalf, 16, 16, 1),
         GEN_COMBINATION(HIP_R_16F, HIP_R_32F, hipblasLtHalf, hipblasLtFloat, 16, 16, 1),
         GEN_COMBINATION(HIP_R_16BF, HIP_R_32F, hipblasLtBfloat16, hipblasLtFloat, 16, 16, 1),
         GEN_COMBINATION(HIP_R_8I, HIP_R_32F, hipblasLtInt8, hipblasLtFloat, 16, 16, 1),
         GEN_COMBINATION(HIP_R_32I, HIP_R_32F, hipblasLtInt32, hipblasLtFloat, 16, 16, 1)}};
}

rocblaslt_status rocblaslt_matrix_transform(rocblaslt_handle                 handle,
                                            rocblaslt_matrix_transform_desc* desc,
                                            const void* alpha, /* host or device pointer */
                                            const void* A,
                                            rocblaslt_matrix_layout layoutA,
                                            const void* beta, /* host or device pointer */
                                            const void* B,
                                            rocblaslt_matrix_layout layoutB,
                                            void*                   C,
                                            rocblaslt_matrix_layout layoutC,
                                            hipStream_t             stream)
{
    if(!handle)
    {
        return rocblaslt_status_invalid_handle;
    }

    if((A && !layoutA) || (B && !layoutB) || (!layoutA && !layoutB) || !C || !layoutC)
    {
        return rocblaslt_status_invalid_value;
    }

    if(!A && !layoutA)
    {
        layoutA = dummyMatrixLayout();
    }

    if(!B && !layoutB)
    {
        layoutB = dummyMatrixLayout();
    }

    if(isMxfp6Pack(desc, layoutA, layoutB, layoutC))
    {
        return mxfp6Pack(desc, A, layoutA, B, C, layoutC, stream);
    }

    size_t vw  = layoutC->m < 4 || layoutC->n < 4 ? 1 : 4;
    auto   key = std::make_tuple(
        layoutA->type, desc->scaleType, layoutA->order, layoutB->order, layoutC->order, vw);

    if(!transformKernels.count(key))
    {
        return rocblaslt_status_internal_error;
    }

    bool       transA         = desc->opA != HIPBLAS_OP_N;
    bool       transB         = desc->opB != HIPBLAS_OP_N;
    bool       scalarInDevice = desc->pointerMode == HIPBLASLT_POINTER_MODE_DEVICE;
    const auto kernelName     = transformKernelNames.at(key);

    const auto err = transformKernels.at(key)(C,
                                              A,
                                              B,
                                              alpha,
                                              beta,
                                              scalarInDevice,
                                              layoutC->m,
                                              layoutC->n,
                                              layoutA->ld,
                                              layoutB->ld,
                                              layoutC->ld,
                                              layoutA->batch_count,
                                              layoutA->batch_stride,
                                              transA,
                                              transB,
                                              stream,
                                              kernelName);

    return (err == hipSuccess) ? rocblaslt_status_success : rocblaslt_status_internal_error;
}

// Test support: the near-die transforms take `version` as the contents of
// /sys/module/amdgpu/version, so a test can drive the driver rule on any driver. NULL restores
// the file. Not part of any supported interface.
extern "C" HIPBLASLT_EXPORT void hipblaslt_page24_driver_for_test(const char* version)
{
    std::lock_guard<std::mutex> lock(page24DriverMutex);
    if(version)
        page24DriverOverride = version;
    else
        page24DriverOverride.reset();
}
