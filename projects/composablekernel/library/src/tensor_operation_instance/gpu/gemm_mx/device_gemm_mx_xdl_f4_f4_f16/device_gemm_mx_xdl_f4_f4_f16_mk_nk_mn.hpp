// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_mx.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F4     = f4x2_pk_t;
using F16    = half_t;
using F32    = float;
using E8M0   = ck::e8m0_bexp_t;
using E8M0PK = int32_t;

using Row = tensor_layout::gemm::RowMajor;
using Col = tensor_layout::gemm::ColumnMajor;

template <index_t... Is>
using S = Sequence<Is...>;

using PassThrough = element_wise::PassThrough;

static constexpr auto GemmDefault    = GemmSpecialization::Default;
static constexpr auto GemmKPadding   = GemmSpecialization::KPadding;
static constexpr auto GemmMPadding   = GemmSpecialization::MPadding;
static constexpr auto GemmMNPadding  = GemmSpecialization::MNPadding;
static constexpr auto GemmMNKPadding = GemmSpecialization::MNKPadding;

static constexpr auto Intrawave = BlockGemmPipelineScheduler::Intrawave;
static constexpr auto Interwave = BlockGemmPipelineScheduler::Interwave;

static constexpr auto ScaleBlockSize = 32;

template <BlockGemmPipelineScheduler BlkGemmPipeSched, GemmSpecialization GemmSpec>
using device_gemm_mx_xdl_f4_f4_f16_mk_nk_mn_instances = std::tuple<
// clang-format off
    //#############################| ALayout| BLayout| CLayout|AData| AScale|BData| BScale| CData| AccData| Cshuffle|           A|           B|           C|          GEMM|    Scale Block| Block|  MPer|  NPer|  KPer| AK1| BK1|MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|    Block-wiseGemm|               Block-wiseGemm|
    //#############################|        |        |        | Type|   Data| Type|   Data|  Type|    Type|     Type| Elementwise| Elementwise| Elementwise|Specialization|           Size|  Size| Block| Block| Block|    |    | XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|          Pipeline|                     Pipeline|
    //#############################|        |        |        |     |   Type|     |   Type|      |        |         |   Operation|   Operation|   Operation|              |               |      |      |      |      |    |    |    |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|         Scheduler|                     Verision|
    //#############################|        |        |        |     |       |     |       |      |        |         |            |            |            |              |               |      |      |      |      |    |    |    |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                 |                |                  |                             |
#if !defined(__gfx125__) && !defined(CK_USE_GFX1250)
      DeviceGemmMX_Xdl_CShuffleV3<       Row,     Col,     Row,   F4, E8M0PK,   F4, E8M0PK,  F16,     F32,      F16, PassThrough, PassThrough, PassThrough,      GemmSpec, ScaleBlockSize,   256,    32,   128,   128,  16,  16,  16,   16,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,     true,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,     true,           2,           2,                   S<1, 32, 1, 8>,               8,  BlkGemmPipeSched,  BlockGemmPipelineVersion::v1>,
      DeviceGemmMX_Xdl_CShuffleV3<       Row,     Col,     Row,   F4, E8M0PK,   F4, E8M0PK,  F16,     F32,      F16, PassThrough, PassThrough, PassThrough,      GemmSpec, ScaleBlockSize,   256,    64,   128,   128,  16,  16,  16,   16,    4,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,     true,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,     true,           2,           2,                   S<1, 32, 1, 8>,               8,  BlkGemmPipeSched,  BlockGemmPipelineVersion::v1>,
      DeviceGemmMX_Xdl_CShuffleV3<       Row,     Col,     Row,   F4, E8M0PK,   F4, E8M0PK,  F16,     F32,      F16, PassThrough, PassThrough, PassThrough,      GemmSpec, ScaleBlockSize,   256,    96,   128,   128,  16,  16,  16,   16,    6,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,     true,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,     true,           2,           2,                   S<1, 32, 1, 8>,               8,  BlkGemmPipeSched,  BlockGemmPipelineVersion::v1>,
       // FIXME: This 96x256x128 instance is disabled on gfx1250 due to illegal memory access for the 5120x5120x4096 problem.
       // T2-04 (GFX1250_CONV_OPTIMIZATION_ROADMAP.md): root cause confirmed by hardware repro
       // (temporarily re-enabling just this instance and running test_gemm_mx's
       // TestGemmMX_MK_NK/2.Large case reproduces the exact "illegal memory access" on real
       // gfx1250 silicon). Mechanism: gridwise_gemm_xdl_cshuffle_v3_mx.hpp's
       // `Padded_Scale_M = ceil(M, ScaleBlockSize) * ScaleBlockSize` (ScaleBlockSize=32) pads the
       // A-scale tensor's logical M extent only to the fixed 32-row scale granularity, not to a
       // multiple of MPerBlock. For M=5120, MPerBlock=96: ceil(5120,32)*32 == 5120 (M is already
       // 32-aligned, so zero extra headroom is added), but the grid dispatches
       // ceil(5120/96)=54 M-blocks, i.e. addresses a *virtual* M range of 54*96=5184 -- 64 rows
       // past the scale tensor's actual (M-sized) extent on the ragged last block. The identical
       // MPerBlock=96/M=5120 geometry is NOT excluded at NPerBlock=128 (line 51, kept) because
       // GetWarpTileConfig derives MPerXdl/MXdlPack from *both* MPerBlock and NPerBlock
       // (device_gemm_xdl_cshuffle_v3_mx.hpp:172-192), which changes the per-block scale-offset
       // divisor enough to move the OOB access outside the actually-allocated buffer at one
       // NPerBlock and inside it at another -- i.e. this is exactly the FlyDSL #679
       // (tile-shape-coupled scale addressing) and #705 (ragged-M-tail 32-row-only padding)
       // patterns operating together, not two independent bugs. NOT fixed here: a correct fix
       // needs the scale tensor's *allocated* buffer (not just this descriptor's logical extent)
       // sized to cover the grid's full virtual M range (ceil(M/MPerBlock)*MPerBlock, not just
       // ceil(M/ScaleBlockSize)*ScaleBlockSize) -- that touches the device-op workspace/buffer-size
       // calculation and possibly host-side scale-tensor allocation as well as this gridwise
       // descriptor, which is out of scope for a single-pass fix without a broader audit of every
       // other excluded MX-GEMM config sharing this same addressing code (bf6/f6/f8_f8_bf16/
       // f8_f8_f16 mk_nk_mn and the mk_mfma_mn B-preshuffle variants -- see this item's
       // investigation notes for the full list). Left disabled.
      DeviceGemmMX_Xdl_CShuffleV3<       Row,     Col,     Row,   F4, E8M0PK,   F4, E8M0PK,  F16,     F32,      F16, PassThrough, PassThrough, PassThrough,      GemmSpec, ScaleBlockSize,   256,    96,   256,   128,  16,  16,  16,   16,    6,    4,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,     true,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,     true,           2,           2,                   S<1, 32, 1, 8>,               8,  BlkGemmPipeSched,  BlockGemmPipelineVersion::v1>,
#endif
      DeviceGemmMX_Xdl_CShuffleV3<       Row,     Col,     Row,   F4, E8M0PK,   F4, E8M0PK,  F16,     F32,      F16, PassThrough, PassThrough, PassThrough,      GemmSpec, ScaleBlockSize,   256,    32,   256,   128,  16,  16,  16,   16,    2,    4,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,     true,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,     true,           2,           2,                   S<1, 32, 1, 8>,               8,  BlkGemmPipeSched,  BlockGemmPipelineVersion::v1>,
      DeviceGemmMX_Xdl_CShuffleV3<       Row,     Col,     Row,   F4, E8M0PK,   F4, E8M0PK,  F16,     F32,      F16, PassThrough, PassThrough, PassThrough,      GemmSpec, ScaleBlockSize,   256,    64,   256,   128,  16,  16,  16,   16,    4,    4,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,     true,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,     true,           2,           2,                   S<1, 32, 1, 8>,               8,  BlkGemmPipeSched,  BlockGemmPipelineVersion::v1>,

      DeviceGemmMX_Xdl_CShuffleV3<       Row,     Col,     Row,   F4, E8M0PK,   F4, E8M0PK,  F16,     F32,      F16, PassThrough, PassThrough, PassThrough,      GemmSpec, ScaleBlockSize,   256,   256,   256,   128,  16,  16,  16,   16,    8,    8,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,     true,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,     true,           2,           2,                   S<1, 32, 1, 8>,               8,  BlkGemmPipeSched,  BlockGemmPipelineVersion::v3>,
      DeviceGemmMX_Xdl_CShuffleV3<       Row,     Col,     Row,   F4, E8M0PK,   F4, E8M0PK,  F16,     F32,      F16, PassThrough, PassThrough, PassThrough,      GemmSpec, ScaleBlockSize,   256,   128,   256,   128,  16,  16,  16,   16,    4,    8,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,     true,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,     true,           2,           2,                   S<1, 32, 1, 8>,               8,  BlkGemmPipeSched,  BlockGemmPipelineVersion::v3>,
      DeviceGemmMX_Xdl_CShuffleV3<       Row,     Col,     Row,   F4, E8M0PK,   F4, E8M0PK,  F16,     F32,      F16, PassThrough, PassThrough, PassThrough,      GemmSpec, ScaleBlockSize,   256,   256,   128,   128,  16,  16,  16,   16,    8,    4,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,     true,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,     true,           2,           2,                   S<1, 32, 1, 8>,               8,  BlkGemmPipeSched,  BlockGemmPipelineVersion::v3>,
      DeviceGemmMX_Xdl_CShuffleV3<       Row,     Col,     Row,   F4, E8M0PK,   F4, E8M0PK,  F16,     F32,      F16, PassThrough, PassThrough, PassThrough,      GemmSpec, ScaleBlockSize,   256,   128,   128,   128,  16,  16,  16,   16,    4,    4,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,     true,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,     true,           2,           2,                   S<1, 32, 1, 8>,               8,  BlkGemmPipeSched,  BlockGemmPipelineVersion::v3>,
#if !defined(__gfx125__) && !defined(CK_USE_GFX1250)
      DeviceGemmMX_Xdl_CShuffleV3<       Row,     Col,     Row,   F4, E8M0PK,   F4, E8M0PK,  F16,     F32,      F16, PassThrough, PassThrough, PassThrough,      GemmSpec, ScaleBlockSize,    64,    32,    32,   128,  16,  16,  16,   16,    2,    2,     S<8,  8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,     true,     S<8,  8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,     true,           2,           2,                   S<1, 16, 1, 4>,               8,  BlkGemmPipeSched,  BlockGemmPipelineVersion::v3>,
#endif
      std::nullptr_t
    // clang-format on
    >;
} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
