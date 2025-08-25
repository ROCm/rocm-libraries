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
#include "matrix-transform/matrix_transform.h"
#include "rocblaslt-auxiliary.h"
#include "rocblaslt-types.h"
#include "rocblaslt.h"
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>
#include <functional>
#include <hipblaslt/hipblaslt-types.h>

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace
{
    std::filesystem::path transformCodeObjectPath()
    {
#ifdef _WIN32
        constexpr char DEFAULT_CO_PATH[]
            = "C:\\opt\\rocm\\bin\\hipblaslt\\library\\hipblasltTransform.hsaco";
#else
        constexpr char DEFAULT_CO_PATH[]
            = "/opt/rocm/lib/hipblaslt/library/hipblasltTransform.hsaco";
#endif

        auto path = rocblaslt_find_library_relative_path(
            std::filesystem::path("hipblasltTransform.hsaco"));
        if(path)
            return *path;
        return std::filesystem::path(DEFAULT_CO_PATH);
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
            auto              coPath   = transformCodeObjectPath();
            const std::string coFolder = coPath.parent_path().string();
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
                    "transformCodeObject", "TransformCodeObjectPath", coFolder.c_str());
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
                                                 "hipblasltTransform.hsaco",
                                                 false,
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
