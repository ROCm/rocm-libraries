// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include <cassert>

#include "ck_grouped_conv_common.hpp"
#include <miopen/solver/ck_impl_interface.hpp>
#include <miopen/solver/ck_impl_error.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/handle.hpp>
#include <miopen/hipoc_kernel.hpp>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "miopen/conv/device_grouped_conv_fwd.hpp"

// ---------------------------------------------------------------------------
// CK type aliases and kernel factory for depthwise conv forward (FP16)
// ---------------------------------------------------------------------------

namespace {

using ProblemDescription = miopen::conv::ProblemDescription;
using miopen::solver::ProblemInterpreter;

template <ck::index_t... Is>
using S                           = ck::Sequence<Is...>;
using InElementOp                 = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp                = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp                = ck::tensor_operation::element_wise::PassThrough;
using InType                      = ck::half_t;
using WeiType                     = ck::half_t;
using AccType                     = float;
using OutType                     = ck::half_t;
constexpr ck::index_t NDimSpatial = 2;
constexpr ck::index_t BlockSize   = 64;
constexpr bool RequirePadding     = false;

// Tuple of potential device CK kernels, parameterized on element type (fp16/bf16).
// Shapes taken to target fp16 Pytorch EfficientNet B0 model:
// https://docs.pytorch.org/vision/main/models/efficientnet.html
template <typename DType>
using DeviceConvFwdFactoryT64 = std::tuple<
    ck::tensor_operation::device::DeviceGroupedConvFwd<
        NDimSpatial,
        BlockSize,
        DType,
        DType,
        AccType,
        DType,
        S<7, 7>,                              // BlockTileSize
        5,                                    // FilterSize
        ck::Tuple<S<1, 1>, S<1, 1>, S<2, 2>>, // FilterParam(dilation, stride, padding)
        InElementOp,
        WeiElementOp,
        OutElementOp,
        32, // NBatch
        4,  // SubTileH
        4,  // SubTileW
        1,  // InScalarPerVector
        1,  // OutScalarPerVector
        RequirePadding>,
    ck::tensor_operation::device::DeviceGroupedConvFwd<
        NDimSpatial,
        BlockSize,
        DType,
        DType,
        AccType,
        DType,
        S<14, 14>,                            // BlockTileSize
        5,                                    // FilterSize
        ck::Tuple<S<1, 1>, S<1, 1>, S<2, 2>>, // FilterParam(dilation, stride, padding)
        InElementOp,
        WeiElementOp,
        OutElementOp,
        32, // NBatch
        4,  // SubTileH
        4,  // SubTileW
        2,  // InScalarPerVector
        2,  // OutScalarPerVector
        RequirePadding>,
    ck::tensor_operation::device::DeviceGroupedConvFwd<
        NDimSpatial,
        BlockSize,
        DType,
        DType,
        AccType,
        DType,
        S<28, 28>,                            // BlockTileSize
        5,                                    // FilterSize
        ck::Tuple<S<1, 1>, S<1, 1>, S<2, 2>>, // FilterParam(dilation, stride, padding)
        InElementOp,
        WeiElementOp,
        OutElementOp,
        32, // NBatch
        4,  // SubTileH
        4,  // SubTileW
        4,  // InScalarPerVector
        4,  // OutScalarPerVector
        RequirePadding>,
    ck::tensor_operation::device::DeviceGroupedConvFwd<
        NDimSpatial,
        BlockSize,
        DType,
        DType,
        AccType,
        DType,
        S<14, 14>,                            // BlockTileSize
        5,                                    // FilterSize
        ck::Tuple<S<1, 1>, S<2, 2>, S<2, 2>>, // FilterParam(dilation, stride, padding)
        InElementOp,
        WeiElementOp,
        OutElementOp,
        32, // NBatch
        4,  // SubTileH
        4,  // SubTileW
        2,  // InScalarPerVector
        1,  // OutScalarPerVector
        RequirePadding>,
    ck::tensor_operation::device::DeviceGroupedConvFwd<
        NDimSpatial,
        BlockSize,
        DType,
        DType,
        AccType,
        DType,
        S<28, 28>,                            // BlockTileSize
        5,                                    // FilterSize
        ck::Tuple<S<1, 1>, S<2, 2>, S<2, 2>>, // FilterParam(dilation, stride, padding)
        InElementOp,
        WeiElementOp,
        OutElementOp,
        32, // NBatch
        4,  // SubTileH
        4,  // SubTileW
        4,  // InScalarPerVector
        2,  // OutScalarPerVector
        RequirePadding>,
    ck::tensor_operation::device::DeviceGroupedConvFwd<
        NDimSpatial,
        BlockSize,
        DType,
        DType,
        AccType,
        DType,
        S<56, 56>,                            // BlockTileSize
        5,                                    // FilterSize
        ck::Tuple<S<1, 1>, S<2, 2>, S<2, 2>>, // FilterParam(dilation, stride, padding)
        InElementOp,
        WeiElementOp,
        OutElementOp,
        8, // NBatch
        4, // SubTileH
        4, // SubTileW
        8, // InScalarPerVector
        4, // OutScalarPerVector
        RequirePadding>

    ,
    ck::tensor_operation::device::DeviceGroupedConvFwd<
        NDimSpatial,
        BlockSize,
        DType,
        DType,
        AccType,
        DType,
        S<7, 7>,                              // BlockTileSize
        3,                                    // FilterSize
        ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>, // FilterParam(dilation, stride, padding)
        InElementOp,
        WeiElementOp,
        OutElementOp,
        32, // NBatch
        4,  // SubTileH
        4,  // SubTileW
        1,  // InScalarPerVector
        1,  // OutScalarPerVector
        RequirePadding>,
    ck::tensor_operation::device::DeviceGroupedConvFwd<
        NDimSpatial,
        BlockSize,
        DType,
        DType,
        AccType,
        DType,
        S<14, 14>,                            // BlockTileSize
        3,                                    // FilterSize
        ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>, // FilterParam(dilation, stride, padding)
        InElementOp,
        WeiElementOp,
        OutElementOp,
        32, // NBatch
        4,  // SubTileH
        4,  // SubTileW
        2,  // InScalarPerVector
        2,  // OutScalarPerVector
        RequirePadding>,
    ck::tensor_operation::device::DeviceGroupedConvFwd<
        NDimSpatial,
        BlockSize,
        DType,
        DType,
        AccType,
        DType,
        S<56, 56>,                            // BlockTileSize
        3,                                    // FilterSize
        ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>, // FilterParam(dilation, stride, padding)
        InElementOp,
        WeiElementOp,
        OutElementOp,
        8, // NBatch
        7, // SubTileH
        8, // SubTileW
        8, // InScalarPerVector
        8, // OutScalarPerVector
        RequirePadding>,
    ck::tensor_operation::device::DeviceGroupedConvFwd<
        NDimSpatial,
        BlockSize,
        DType,
        DType,
        AccType,
        DType,
        S<112, 112>,                          // BlockTileSize
        3,                                    // FilterSize
        ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>, // FilterParam(dilation, stride, padding)
        InElementOp,
        WeiElementOp,
        OutElementOp,
        2,  // NBatch
        14, // SubTileH
        16, // SubTileW
        8,  // InScalarPerVector
        8,  // OutScalarPerVector
        RequirePadding>

    ,
    ck::tensor_operation::device::DeviceGroupedConvFwd<
        NDimSpatial,
        BlockSize,
        DType,
        DType,
        AccType,
        DType,
        S<28, 28>,                            // BlockTileSize
        3,                                    // FilterSize
        ck::Tuple<S<1, 1>, S<2, 2>, S<1, 1>>, // FilterParam(dilation, stride, padding)
        InElementOp,
        WeiElementOp,
        OutElementOp,
        32, // NBatch
        4,  // SubTileH
        4,  // SubTileW
        4,  // InScalarPerVector
        2,  // OutScalarPerVector
        RequirePadding>,
    ck::tensor_operation::device::DeviceGroupedConvFwd<
        NDimSpatial,
        BlockSize,
        DType,
        DType,
        AccType,
        DType,
        S<112, 112>,                          // BlockTileSize
        3,                                    // FilterSize
        ck::Tuple<S<1, 1>, S<2, 2>, S<1, 1>>, // FilterParam(dilation, stride, padding)
        InElementOp,
        WeiElementOp,
        OutElementOp,
        8, // NBatch
        7, // SubTileH
        8, // SubTileW
        8, // InScalarPerVector
        8, // OutScalarPerVector
        RequirePadding>

    // convnext 7x7 depthwise (FilterSize=7, stride 1, pad 3) fp16 instances.
    // BlockTileSize matches the depthwise spatial stages (56/28/14/7);
    // InScalarPerVector is chosen so it divides the tile width.
    ,
    ck::tensor_operation::device::DeviceGroupedConvFwd<
        NDimSpatial,
        BlockSize,
        DType,
        DType,
        AccType,
        DType,
        S<7, 7>,                              // BlockTileSize
        7,                                    // FilterSize
        ck::Tuple<S<1, 1>, S<1, 1>, S<3, 3>>, // FilterParam(dilation, stride, padding)
        InElementOp,
        WeiElementOp,
        OutElementOp,
        32, // NBatch
        4,  // SubTileH
        4,  // SubTileW
        1,  // InScalarPerVector
        1,  // OutScalarPerVector
        RequirePadding>,
    ck::tensor_operation::device::DeviceGroupedConvFwd<
        NDimSpatial,
        BlockSize,
        DType,
        DType,
        AccType,
        DType,
        S<14, 14>,                            // BlockTileSize
        7,                                    // FilterSize
        ck::Tuple<S<1, 1>, S<1, 1>, S<3, 3>>, // FilterParam(dilation, stride, padding)
        InElementOp,
        WeiElementOp,
        OutElementOp,
        32, // NBatch
        4,  // SubTileH
        4,  // SubTileW
        2,  // InScalarPerVector
        2,  // OutScalarPerVector
        RequirePadding>,
    ck::tensor_operation::device::DeviceGroupedConvFwd<
        NDimSpatial,
        BlockSize,
        DType,
        DType,
        AccType,
        DType,
        S<28, 28>,                            // BlockTileSize
        7,                                    // FilterSize
        ck::Tuple<S<1, 1>, S<1, 1>, S<3, 3>>, // FilterParam(dilation, stride, padding)
        InElementOp,
        WeiElementOp,
        OutElementOp,
        32, // NBatch
        4,  // SubTileH
        4,  // SubTileW
        4,  // InScalarPerVector
        4,  // OutScalarPerVector
        RequirePadding>,
    ck::tensor_operation::device::DeviceGroupedConvFwd<
        NDimSpatial,
        BlockSize,
        DType,
        DType,
        AccType,
        DType,
        S<56, 56>,                            // BlockTileSize
        7,                                    // FilterSize
        ck::Tuple<S<1, 1>, S<1, 1>, S<3, 3>>, // FilterParam(dilation, stride, padding)
        InElementOp,
        WeiElementOp,
        OutElementOp,
        8, // NBatch
        7, // SubTileH
        8, // SubTileW
        8, // InScalarPerVector
        8, // OutScalarPerVector
        RequirePadding>

    // Extra efficientnet/mobilenet depthwise fwd coverage: 3x3 stride-1 at
    // the 112 tile (verified). (Small stride-2 tiles need dedicated tuning and
    // are left out rather than shipped unverified.)
    ,
    ck::tensor_operation::device::DeviceGroupedConvFwd<NDimSpatial,
                                                       BlockSize,
                                                       DType,
                                                       DType,
                                                       AccType,
                                                       DType,
                                                       S<112, 112>, // BlockTileSize
                                                       3, // FilterSize (3x3 stride-1 pad-1)
                                                       ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>,
                                                       InElementOp,
                                                       WeiElementOp,
                                                       OutElementOp,
                                                       2,
                                                       14,
                                                       16,
                                                       8,
                                                       8,
                                                       RequirePadding>>;

// Wave32 (RDNA gfx10/gfx11/gfx12) counterparts of the instances above: same shapes with
// BlockSize=32. The 28x28/56x56/112x112 output tiles use larger subtiles so num_subtiles
// fits in a 32-lane wavefront.
template <typename DType>
using DeviceConvFwdFactoryT32 =
    std::tuple<ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   32, // BlockSize (wave32)
                   DType,
                   DType,
                   AccType,
                   DType,
                   S<7, 7>,                              // BlockTileSize
                   5,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<2, 2>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   1,  // InScalarPerVector
                   1,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   32, // BlockSize (wave32)
                   DType,
                   DType,
                   AccType,
                   DType,
                   S<14, 14>,                            // BlockTileSize
                   5,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<2, 2>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   2,  // InScalarPerVector
                   2,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   32, // BlockSize (wave32)
                   DType,
                   DType,
                   AccType,
                   DType,
                   S<28, 28>,                            // BlockTileSize
                   5,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<2, 2>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   3,  // SubTileH
                   12, // SubTileW
                   4,  // InScalarPerVector
                   4,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   32, // BlockSize (wave32)
                   DType,
                   DType,
                   AccType,
                   DType,
                   S<14, 14>,                            // BlockTileSize
                   5,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<2, 2>, S<2, 2>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   2,  // InScalarPerVector
                   1,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   32, // BlockSize (wave32)
                   DType,
                   DType,
                   AccType,
                   DType,
                   S<28, 28>,                            // BlockTileSize
                   5,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<2, 2>, S<2, 2>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   4,  // InScalarPerVector
                   2,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   32, // BlockSize (wave32)
                   DType,
                   DType,
                   AccType,
                   DType,
                   S<56, 56>,                            // BlockTileSize
                   5,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<2, 2>, S<2, 2>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   8,  // NBatch
                   3,  // SubTileH
                   12, // SubTileW
                   8,  // InScalarPerVector
                   4,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   32, // BlockSize (wave32)
                   DType,
                   DType,
                   AccType,
                   DType,
                   S<7, 7>,                              // BlockTileSize
                   3,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   1,  // InScalarPerVector
                   1,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   32, // BlockSize (wave32)
                   DType,
                   DType,
                   AccType,
                   DType,
                   S<14, 14>,                            // BlockTileSize
                   3,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   2,  // InScalarPerVector
                   2,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   32, // BlockSize (wave32)
                   DType,
                   DType,
                   AccType,
                   DType,
                   S<56, 56>,                            // BlockTileSize
                   3,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   8,  // NBatch
                   7,  // SubTileH
                   16, // SubTileW
                   8,  // InScalarPerVector
                   8,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   32, // BlockSize (wave32)
                   DType,
                   DType,
                   AccType,
                   DType,
                   S<112, 112>,                          // BlockTileSize
                   3,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   2,  // NBatch
                   7,  // SubTileH
                   56, // SubTileW
                   8,  // InScalarPerVector
                   8,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   32, // BlockSize (wave32)
                   DType,
                   DType,
                   AccType,
                   DType,
                   S<28, 28>,                            // BlockTileSize
                   3,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<2, 2>, S<1, 1>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   4,  // InScalarPerVector
                   2,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   32, // BlockSize (wave32)
                   DType,
                   DType,
                   AccType,
                   DType,
                   S<112, 112>,                          // BlockTileSize
                   3,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<2, 2>, S<1, 1>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   8,  // NBatch
                   7,  // SubTileH
                   16, // SubTileW
                   8,  // InScalarPerVector
                   8,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   32, // BlockSize (wave32)
                   DType,
                   DType,
                   AccType,
                   DType,
                   S<7, 7>,                              // BlockTileSize
                   7,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<3, 3>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   1,  // InScalarPerVector
                   1,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   32, // BlockSize (wave32)
                   DType,
                   DType,
                   AccType,
                   DType,
                   S<14, 14>,                            // BlockTileSize
                   7,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<3, 3>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   4,  // SubTileH
                   4,  // SubTileW
                   2,  // InScalarPerVector
                   2,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   32, // BlockSize (wave32)
                   DType,
                   DType,
                   AccType,
                   DType,
                   S<28, 28>,                            // BlockTileSize
                   7,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<3, 3>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   32, // NBatch
                   3,  // SubTileH
                   12, // SubTileW
                   4,  // InScalarPerVector
                   4,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   32, // BlockSize (wave32)
                   DType,
                   DType,
                   AccType,
                   DType,
                   S<56, 56>,                            // BlockTileSize
                   7,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<3, 3>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   8,  // NBatch
                   7,  // SubTileH
                   16, // SubTileW
                   8,  // InScalarPerVector
                   8,  // OutScalarPerVector
                   RequirePadding>,
               ck::tensor_operation::device::DeviceGroupedConvFwd<
                   NDimSpatial,
                   32, // BlockSize (wave32)
                   DType,
                   DType,
                   AccType,
                   DType,
                   S<112, 112>,                          // BlockTileSize
                   3,                                    // FilterSize
                   ck::Tuple<S<1, 1>, S<1, 1>, S<1, 1>>, // FilterParam(dilation, stride, padding)
                   InElementOp,
                   WeiElementOp,
                   OutElementOp,
                   2,  // NBatch
                   7,  // SubTileH
                   56, // SubTileW
                   8,  // InScalarPerVector
                   8,  // OutScalarPerVector
                   RequirePadding>>;

#ifndef MIOPEN_CK_DEPTHWISE_WAVE_SIZE
// Fallback when the per-arch value is not provided by the build (wave64 default).
// src/ck_impl/CMakeLists.txt sets this per target GPU.
#define MIOPEN_CK_DEPTHWISE_WAVE_SIZE 64
#endif

#if MIOPEN_CK_DEPTHWISE_WAVE_SIZE == 32
template <typename DType>
using DeviceConvFwdFactoryT = DeviceConvFwdFactoryT32<DType>;
#else
template <typename DType>
using DeviceConvFwdFactoryT = DeviceConvFwdFactoryT64<DType>;
#endif

// Concrete factories per element type.
using DeviceConvFwdFactory     = DeviceConvFwdFactoryT<ck::half_t>;
using DeviceConvFwdFactoryBf16 = DeviceConvFwdFactoryT<ck::bhalf_t>;

// ---------------------------------------------------------------------------
// CKArgs -- extracts convolution dimensions from ProblemDescription for the
// custom depthwise kernel API.
// ---------------------------------------------------------------------------

struct CKArgs
{
    explicit CKArgs(const ProblemDescription& problem)
    {
        G  = ProblemInterpreter::GetGroupCountG(problem);
        N  = ProblemInterpreter::GetBatchN(problem);
        K1 = ProblemInterpreter::GetOutputChannelK(problem);
        C1 = ProblemInterpreter::GetInputChannelC(problem);
        C  = C1 / G; // Number of input channels per group
        K  = K1 / G; // Number of output channels per group
        Hi = ProblemInterpreter::GetInputHeightHi(problem);
        Wi = ProblemInterpreter::GetInputWidthWi(problem);
        Ho = ProblemInterpreter::GetOutputHeightHo(problem);
        Wo = ProblemInterpreter::GetOutputWidthWo(problem);
        Y  = ProblemInterpreter::GetFilterHeightY(problem);
        X  = ProblemInterpreter::GetFilterWidthX(problem);

        input_lengths = {G, N, C, Hi, Wi}; // input
        out_lens      = {G, N, K, Ho, Wo}; // output
        wei_lens      = {G, K, C, Y, X};   // filter = wei
        in_strides    = {Hi * Wi * C, G * Hi * Wi * C, 1, Wi * C, C};
        out_strides   = {Ho * Wo * K, G * Ho * Wo * K, 1, Wo * K, K};
        wei_strides   = {Y * X * C, G * Y * X * C, 1, X * C, C};

        filter_stride   = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                           ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        filter_dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                           ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding        = {ProblemInterpreter::GetInputLeftPadH(problem),
                           ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding        = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                           ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    CKArgs(const CKArgs&)            = default;
    CKArgs& operator=(const CKArgs&) = default;

    int G;
    int N;
    int K;
    int C;
    int C1;
    int K1;
    int Hi;
    int Wi;
    int Ho;
    int Wo;
    int Y;
    int X;
    std::array<ck::index_t, 5> input_lengths;
    std::array<ck::index_t, 5> in_strides;
    std::array<ck::index_t, 5> out_lens;
    std::array<ck::index_t, 5> out_strides;
    std::array<ck::index_t, 5> wei_lens;
    std::array<ck::index_t, 5> wei_strides;
    std::array<ck::index_t, 2> filter_stride;
    std::array<ck::index_t, 2> filter_dilation;
    std::array<ck::index_t, 2> lPadding;
    std::array<ck::index_t, 2> rPadding;
};

// ---------------------------------------------------------------------------
// Helpers: enumerate valid kernels, check applicability, check arg support
// ---------------------------------------------------------------------------

// Each depthwise CK instance in DeviceConvFwdFactory is hard-specialized at compile time for a
// specific filter size, stride, padding and dilation. CK's IsSupportedArgument only validates
// tiling/divisibility -- it does NOT reject a shape-mismatched instance -- so without this check
// the tuner could select e.g. a FilterSize=5 instance for a 3x3 problem and record it as the
// winning perf config. That config is invalid for the problem and later fails to build. Require
// the instance's compile-time shape to match the problem exactly before treating it as applicable.
template <typename DeviceOp>
bool InstanceShapeMatchesProblem(const CKArgs& a)
{
    // Match the instance's compile-time filter/stride/pad/dilation to the problem. Only the left
    // pad is compared: stride>1 convs legitimately have an adjusted (asymmetric) right pad, and the
    // forward kernel handles that, so requiring lPadding==rPadding here would wrongly reject valid
    // stride-2 instances. (Symmetric padding is only required for the backward-data flip, which is
    // enforced in ck_depthwise_bwd_data_impl.cpp.)
    using G = typename DeviceOp::GridwiseConvFwd;
    return G::Filter_Y == a.Y && G::Filter_X == a.X &&                               //
           G::Stride_H == a.filter_stride[0] && G::Stride_W == a.filter_stride[1] && //
           G::Pad_H == a.lPadding[0] && G::Pad_W == a.lPadding[1] &&                 //
           G::Dilation_Y == a.filter_dilation[0] && G::Dilation_X == a.filter_dilation[1];
}

template <typename Factory>
std::vector<std::string> FillValidKernels(const ProblemDescription& problem)
{
    const auto ck_args             = CKArgs{problem};
    constexpr uint32_t kernelCount = std::tuple_size_v<Factory>;
    std::vector<std::string> valid_kernels;

    ck::static_for<0, kernelCount, 1>{}([&](auto i) -> void {
        auto conv_ptr  = std::get<i>(Factory{});
        using DeviceOp = ck::remove_cvref_t<decltype(conv_ptr)>;
        if(!InstanceShapeMatchesProblem<DeviceOp>(ck_args))
            return; // instance is specialized for a different filter/stride/pad/dilation
        auto argument_ptr =
            conv_ptr.MakeArgumentPointer(nullptr,
                                         nullptr,
                                         std::array<const void*, 0>{},
                                         nullptr,
                                         ck_args.input_lengths,
                                         ck_args.in_strides,
                                         ck_args.wei_lens,
                                         ck_args.wei_strides,
                                         std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                         std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                         ck_args.out_lens,
                                         ck_args.out_strides,
                                         ck_args.filter_stride,
                                         ck_args.filter_dilation,
                                         ck_args.lPadding,
                                         ck_args.rPadding,
                                         InElementOp{},
                                         WeiElementOp{},
                                         OutElementOp{});
        if(conv_ptr.IsSupportedArgument(argument_ptr.get()))
        {
            valid_kernels.push_back(conv_ptr.GetTypeString());
        }
    });
    return valid_kernels;
}

template <typename Factory>
bool CheckCKApplicability(const ProblemDescription& problem)
{
    const auto ck_args             = CKArgs{problem};
    constexpr uint32_t kernelCount = std::tuple_size_v<Factory>;
    bool found                     = false;

    ck::static_for<0, kernelCount, 1>{}([&](auto i) -> void {
        if(found)
            return;
        auto conv_ptr  = std::get<i>(Factory{});
        using DeviceOp = ck::remove_cvref_t<decltype(conv_ptr)>;
        if(!InstanceShapeMatchesProblem<DeviceOp>(ck_args))
            return; // instance is specialized for a different filter/stride/pad/dilation
        auto argument_ptr =
            conv_ptr.MakeArgumentPointer(nullptr,
                                         nullptr,
                                         std::array<const void*, 0>{},
                                         nullptr,
                                         ck_args.input_lengths,
                                         ck_args.in_strides,
                                         ck_args.wei_lens,
                                         ck_args.wei_strides,
                                         std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                         std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                         ck_args.out_lens,
                                         ck_args.out_strides,
                                         ck_args.filter_stride,
                                         ck_args.filter_dilation,
                                         ck_args.lPadding,
                                         ck_args.rPadding,
                                         InElementOp{},
                                         WeiElementOp{},
                                         OutElementOp{});
        if(conv_ptr.IsSupportedArgument(argument_ptr.get()))
        {
            found = true;
        }
    });
    return found;
}

template <typename Factory>
bool CheckIsArgSupported(const ProblemDescription& problem, const std::string& kernel_id)
{
    const auto ck_args             = CKArgs{problem};
    constexpr uint32_t kernelCount = std::tuple_size_v<Factory>;
    bool supported                 = false;

    ck::static_for<0, kernelCount, 1>{}([&](auto i) -> void {
        auto conv_ptr  = std::get<i>(Factory{});
        using DeviceOp = ck::remove_cvref_t<decltype(conv_ptr)>;
        if(conv_ptr.GetTypeString() == kernel_id && InstanceShapeMatchesProblem<DeviceOp>(ck_args))
        {
            auto argument_ptr = conv_ptr.MakeArgumentPointer(
                nullptr,
                nullptr,
                std::array<const void*, 0>{},
                nullptr,
                ck_args.input_lengths,
                ck_args.in_strides,
                ck_args.wei_lens,
                ck_args.wei_strides,
                std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                ck_args.out_lens,
                ck_args.out_strides,
                ck_args.filter_stride,
                ck_args.filter_dilation,
                ck_args.lPadding,
                ck_args.rPadding,
                InElementOp{},
                WeiElementOp{},
                OutElementOp{});
            supported = conv_ptr.IsSupportedArgument(argument_ptr.get());
        }
    });
    return supported;
}

template <typename Factory>
bool BuildDepthwiseSolution(const ProblemDescription& problem,
                            const std::string& kid,
                            miopen::solver::ConvSolution& solution)
{
    constexpr uint32_t kernelCount = std::tuple_size_v<Factory>;
    bool found                     = false;
    ck::static_for<0, kernelCount, 1>{}([&](auto i) -> void {
        if(found)
            return;
        const auto device_conv_fwd_instance = std::get<i>(Factory{});
        using DeviceConvFwdInstance = ck::remove_cvref_t<decltype(device_conv_fwd_instance)>;
        if(device_conv_fwd_instance.GetTypeString() != kid)
            return;
        found              = true;
        auto conv_instance = std::make_shared<DeviceConvFwdInstance>();
        solution.invoker_factory =
            [conv_ptr_ = std::move(conv_instance),
             ck_args   = CKArgs{problem}](const std::vector<miopen::Kernel>&) mutable {
                return [conv_ptr = std::move(conv_ptr_),
                        ck_args](const miopen::Handle& handle,
                                 const miopen::AnyInvokeParams& primitive_params) {
                    const auto& fwd_ctx = primitive_params.CastTo<miopen::conv::DataInvokeParams>();
                    auto argument_ptr   = conv_ptr->MakeArgumentPointer(
                        fwd_ctx.tensors.in,
                        fwd_ctx.tensors.w,
                        std::array<const void*, 0>{},
                        fwd_ctx.tensors.out,
                        ck_args.input_lengths,
                        ck_args.in_strides,
                        ck_args.wei_lens,
                        ck_args.wei_strides,
                        std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                        std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                        ck_args.out_lens,
                        ck_args.out_strides,
                        ck_args.filter_stride,
                        ck_args.filter_dilation,
                        ck_args.lPadding,
                        ck_args.rPadding,
                        InElementOp{},
                        WeiElementOp{},
                        OutElementOp{});
                    auto invoker_ptr = conv_ptr->MakeInvokerPointer();
                    {
                        miopen::HipEventProfiler prf(handle);
                        invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), false});
                    }
                    if(handle.IsProfilingEnabled())
                    {
                        float avg_time = handle.GetKernelTime();
                        handle.ResetKernelTime();
                        handle.AccumKernelTime(avg_time);
                    }
                };
            };
    });
    return found;
}

} // anonymous namespace

// ===========================================================================
// Depthwise Conv FWD extern "C" functions
// ===========================================================================

extern "C" ck_impl_status_t
ck_impl_depthwise_fwd_fill_valid_kernels(const miopen::conv::ProblemDescription* problem,
                                         miopenDataType_t data_type,
                                         bool /*use_tf32*/,
                                         CKKernelListHandle** out_handle)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_handle, CK_IMPL_STATUS_BAD_PARAM, "Null out_handle");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        auto result = std::make_unique<CKKernelListHandle>();
        if(data_type == miopenHalf)
            result->kernels = FillValidKernels<DeviceConvFwdFactory>(*problem);
        else if(data_type == miopenBFloat16)
            result->kernels = FillValidKernels<DeviceConvFwdFactoryBf16>(*problem);
        *out_handle = result.release();
    });
}

extern "C" ck_impl_status_t
ck_impl_depthwise_fwd_is_applicable(const miopen::conv::ProblemDescription* problem,
                                    miopenDataType_t data_type,
                                    bool /*use_tf32*/,
                                    bool* out_result)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_result, CK_IMPL_STATUS_BAD_PARAM, "Null out_result");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        if(data_type == miopenHalf)
            *out_result = CheckCKApplicability<DeviceConvFwdFactory>(*problem);
        else if(data_type == miopenBFloat16)
            *out_result = CheckCKApplicability<DeviceConvFwdFactoryBf16>(*problem);
        else
            *out_result = false;
    });
}

extern "C" ck_impl_status_t
ck_impl_depthwise_fwd_is_args_supported(const miopen::conv::ProblemDescription* problem,
                                        const char* kernel_id,
                                        miopenDataType_t data_type,
                                        bool /*use_tf32*/,
                                        bool* out_result)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_result, CK_IMPL_STATUS_BAD_PARAM, "Null out_result");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        CK_IMPL_THROW_IF_NULL(kernel_id, CK_IMPL_STATUS_BAD_PARAM, "Null kernel_id");
        std::string kid(kernel_id);
        if(data_type == miopenHalf)
            *out_result = CheckIsArgSupported<DeviceConvFwdFactory>(*problem, kid);
        else if(data_type == miopenBFloat16)
            *out_result = CheckIsArgSupported<DeviceConvFwdFactoryBf16>(*problem, kid);
        else
            *out_result = false;
    });
}

extern "C" ck_impl_status_t
ck_impl_depthwise_fwd_get_workspace_size(const miopen::conv::ProblemDescription* problem,
                                         miopenDataType_t data_type,
                                         bool /*use_tf32*/,
                                         size_t* out_size)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_size, CK_IMPL_STATUS_BAD_PARAM, "Null out_size");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        if(data_type != miopenHalf)
        {
            *out_size = 0;
            return;
        }
        *out_size = 0;
    });
}

extern "C" ck_impl_status_t
ck_impl_depthwise_fwd_get_solution(const miopen::ExecutionContext* ctx,
                                   const miopen::conv::ProblemDescription* problem,
                                   const char* kernel_id,
                                   bool /*use_tf32*/,
                                   miopen::solver::ConvSolution** out_solution)
{
    return ck_impl_try_catch([&]() {
        CK_IMPL_THROW_IF_NULL(out_solution, CK_IMPL_STATUS_BAD_PARAM, "Null out_solution");
        CK_IMPL_THROW_IF_NULL(ctx, CK_IMPL_STATUS_BAD_PARAM, "Null ctx");
        CK_IMPL_THROW_IF_NULL(problem, CK_IMPL_STATUS_BAD_PARAM, "Null problem");
        CK_IMPL_THROW_IF_NULL(kernel_id, CK_IMPL_STATUS_BAD_PARAM, "Null kernel_id");

        std::string kid(kernel_id);
        miopen::solver::ConvSolution solution;
        bool found = BuildDepthwiseSolution<DeviceConvFwdFactory>(*problem, kid, solution) ||
                     BuildDepthwiseSolution<DeviceConvFwdFactoryBf16>(*problem, kid, solution);

        CK_IMPL_THROW_IF_FALSE(
            found, CK_IMPL_STATUS_INVALID_VALUE, "No matching kernel found for kernel_id");

        *out_solution = new miopen::solver::ConvSolution(std::move(solution));
    });
}
