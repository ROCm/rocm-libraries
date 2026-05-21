# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Lower a ck_dsl *instance spec* to CK Tile-style C++ source.

Distinct from :mod:`ck_dsl.core.lower_hip`, which lowers the **post-IR**
``KernelDef`` to a flat ``__global__`` body that mirrors the SSA one-to-one
(useful for IR inspection and quick hipcc validation). The CK Tile lowering
operates **before** the IR exists -- it walks the high-level Python
dataclass spec (e.g. :class:`UniversalGemmSpec`) and emits a C++ source
that composes the same CK Tile templates a hand-written CK Tile kernel
would.

In other words:

  * ``lower_kernel_to_hip(kernel)``        -> raw, IR-shaped C++
  * ``lower_spec_to_cktile(spec)``         -> CK Tile-shaped C++

CK Tile-shaped output is what AMD's CK Tile examples (e.g.
``example/ck_tile/03_gemm/universal_gemm_invoker.hpp``) compile to, and is
the natural artefact to diff against when bringing up new kernels or
comparing tile-distribution choices between ck_dsl and CK Tile.

Supported specs (v1):

  * :class:`UniversalGemmSpec` -> CK Tile ``GemmKernel<TilePartitioner,
    GemmPipeline, GemmEpilogue>`` composition with the same
    ``TileGemmShape`` / pipeline / scheduler / epilogue choices the
    Python builder used.
  * :class:`ImplicitGemmConvSpec` -> CK Tile
    ``GroupedConvolutionForwardKernel`` composition mirroring
    ``example/ck_tile/20_grouped_convolution/grouped_convolution_forward_invoker.hpp``.

For unsupported specs we raise :class:`NotImplementedError` with a list of
the dispatchable spec types so callers can add a handler.
"""

from __future__ import annotations

from typing import Any, Tuple


__all__ = [
    "lower_spec_to_cktile",
    "lower_universal_gemm_to_cktile",
    "lower_implicit_gemm_conv_to_cktile",
]


# ---------------------------------------------------------------------
# Spec-field translations
# ---------------------------------------------------------------------
#
# ck_dsl uses short string names for pipeline / scheduler / dtype /
# layout choices; CK Tile uses fully-qualified enum constants in C++.
# Keep the maps explicit -- adding a new pipeline option is a two-line
# patch here plus a one-line addition to the emitted PipelineTypeTraits
# specialisations below.

_PIPELINE_MAP = {
    "mem": "ck_tile::GemmPipeline::MEMORY",
    "compv3": "ck_tile::GemmPipeline::COMPUTE_V3",
    "compv4": "ck_tile::GemmPipeline::COMPUTE_V4",
}

_SCHEDULER_MAP = {
    "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
    "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
}

_DTYPE_MAP = {
    "fp16": "ck_tile::half_t",
    "f16": "ck_tile::half_t",
    "bf16": "ck_tile::bf16_t",
    "bfloat16": "ck_tile::bf16_t",
    "fp32": "float",
    "f32": "float",
    "fp8e4m3": "ck_tile::fp8_t",
}


# CK Tile uses one-letter axis abbreviations: R=Row, C=Column.
# ck_dsl's :class:`DataSpec.layout` is the three-letter shorthand
# (``"RCR"`` -> A row-major, B col-major, C row-major).
def _layout_3(layout_str: str) -> Tuple[str, str, str]:
    if len(layout_str) != 3 or any(c not in "RC" for c in layout_str):
        raise ValueError(
            f"unsupported gemm layout {layout_str!r}; expected three "
            "letters from {{'R', 'C'}}, e.g. 'RCR'"
        )

    def _to(c: str) -> str:
        return (
            "ck_tile::tensor_layout::gemm::RowMajor"
            if c == "R"
            else "ck_tile::tensor_layout::gemm::ColumnMajor"
        )

    return tuple(_to(c) for c in layout_str)  # type: ignore[return-value]


def _is_fp16_path(spec: Any) -> bool:
    """v1 emitter targets the fp16 RCR fast path only; bf16 / fp8 fall
    through to a NotImplementedError with a clear message so the caller
    can extend the maps above.
    """
    d = spec.data
    return all(
        x in ("fp16", "f16") for x in (d.dtype_a, d.dtype_b, d.dtype_c)
    ) and d.dtype_acc in ("fp32", "f32")


# ---------------------------------------------------------------------
# Pipeline type traits emitted inline
# ---------------------------------------------------------------------
#
# CK Tile's example files (e.g. gemm_utils.hpp) carry a
# ``PipelineTypeTraits`` switch that maps the ``GemmPipeline`` enum to a
# concrete pipeline class template. We inline the same switch here so the
# emitted file is self-contained and doesn't need to ``#include`` the
# example utilities (which aren't part of the CK Tile public surface).

_PIPELINE_TYPE_TRAITS = """\
// Inlined from example/ck_tile/03_gemm/gemm_utils.hpp so the emitted
// source is self-contained. Maps the ``GemmPipeline`` enum to the
// pipeline template instantiation -- one specialisation per supported
// pipeline variant. Class names match CK Tile's public surface in
// ck_tile/ops/gemm/pipeline/* (``AgBgCr`` = ``AGmemBGmemCReg``).
template <ck_tile::GemmPipeline>
struct PipelineTypeTraits;

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::MEMORY>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrMem<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V3>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V4>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV4<PipelineProblem>;
};
"""


# ---------------------------------------------------------------------
# UniversalGemmSpec -> CK Tile gemm source
# ---------------------------------------------------------------------


def lower_universal_gemm_to_cktile(spec: Any, *, kernel_name: str | None = None) -> str:
    """Emit a runnable CK Tile-style C++ source for a :class:`UniversalGemmSpec`.

    The emitted file:

      1. Includes the CK Tile public headers.
      2. Declares a ``GemmConfig`` struct mirroring
         ``example/ck_tile/03_gemm/gemm_utils.hpp::GemmConfigBase`` with the
         spec's tile / warp / MFMA / pipeline / scheduler / pad / persistent
         choices baked in as ``static constexpr``.
      3. Inlines ``PipelineTypeTraits`` so the file doesn't depend on the
         example utilities.
      4. Composes ``GemmShape`` / ``TilePartitioner`` / ``GemmUniversalTraits``
         / ``UniversalGemmProblem`` / ``GemmPipeline`` / ``GemmEpilogue`` /
         ``Kernel`` exactly as :file:`universal_gemm_invoker.hpp` does.
      5. Exposes ``extern "C" float launch_<kernel_name>(a, b, c, M, N, K,
         stride_a, stride_b, stride_c, k_batch, stream)`` so the caller can
         compile + link + run the result.

    Use ``kernel_name`` to override the default mangled name from
    ``UniversalGemmSpec.kernel_name()`` (e.g. when the caller wants the
    emitted launcher to match an existing manifest.json).
    """
    # Lazy-import to keep core/lower_cktile.py importable in environments
    # without the instances subpackage compiled (it pulls in the helpers
    # surface which needs torch in some flows).
    from ..instances.gemm_universal import UniversalGemmSpec

    if not isinstance(spec, UniversalGemmSpec):
        raise TypeError(
            f"lower_universal_gemm_to_cktile expects UniversalGemmSpec, got {type(spec).__name__}"
        )

    if not _is_fp16_path(spec):
        raise NotImplementedError(
            f"CK Tile lowering currently supports the fp16 dtype path only "
            f"(spec.data={spec.data!r}); extend _DTYPE_MAP and the dtype "
            "aliases below to add bf16 / fp8 support"
        )

    t = spec.tile
    trait = spec.trait
    if trait.pipeline not in _PIPELINE_MAP:
        raise NotImplementedError(
            f"unsupported pipeline {trait.pipeline!r}; supported: "
            f"{sorted(_PIPELINE_MAP)}"
        )
    if trait.scheduler not in _SCHEDULER_MAP:
        raise NotImplementedError(
            f"unsupported scheduler {trait.scheduler!r}; supported: "
            f"{sorted(_SCHEDULER_MAP)}"
        )

    a_layout, b_layout, c_layout = _layout_3(spec.data.layout)
    name = kernel_name or spec.kernel_name()

    # compv4 doubles the LDS buffer (the "double-buffer A and B" knob
    # CK Tile gates via DoubleSmemBuffer). Other pipelines stay single-
    # buffered.
    double_smem_buffer = "true" if trait.pipeline == "compv4" else "false"
    persistent = "true" if trait.persistent else "false"

    # The TilePartitioner has two knobs in CK Tile's GemmSpatiallyLocalTilePartitioner;
    # 8 / 4 is the default the example uses and is what we'd get without an
    # explicit override.
    tp_group_num = 8
    tp_m01 = 4
    num_wave_groups = 1
    block_per_cu = 1

    parts = [
        "// =========================================================",
        "// Auto-generated by ck_dsl.core.lower_cktile from",
        "//   UniversalGemmSpec(",
        f"//     name={spec.name!r},",
        f"//     tile_m={t.tile_m}, tile_n={t.tile_n}, tile_k={t.tile_k},",
        f"//     warp_m={t.warp_m}, warp_n={t.warp_n}, warp_k={t.warp_k},",
        f"//     warp_tile_m={t.warp_tile_m}, warp_tile_n={t.warp_tile_n},",
        f"//     warp_tile_k={t.warp_tile_k},",
        f"//     pipeline={trait.pipeline!r}, scheduler={trait.scheduler!r},",
        f"//     epilogue={trait.epilogue!r}, pad=({trait.pad_m},{trait.pad_n},{trait.pad_k}),",
        f"//     persistent={trait.persistent},",
        f"//     layout={spec.data.layout!r})",
        "//",
        "// Mirrors example/ck_tile/03_gemm/universal_gemm_invoker.hpp.",
        "// =========================================================",
        "",
        "#include <ck_tile/core.hpp>",
        "#include <ck_tile/host.hpp>",
        "#include <ck_tile/ops/gemm.hpp>",
        "#include <ck_tile/ops/epilogue.hpp>",
        "",
        _PIPELINE_TYPE_TRAITS,
        "",
        f"namespace ck_dsl_emit_{name} {{",
        "",
        "// -- per-spec GemmConfig: a single struct of static constexpr,",
        "// matching GemmConfigBase + per-config overrides in",
        "// example/ck_tile/03_gemm/gemm_utils.hpp.",
        "struct GemmConfig {",
        f"    static constexpr ck_tile::index_t M_Tile = {t.tile_m};",
        f"    static constexpr ck_tile::index_t N_Tile = {t.tile_n};",
        f"    static constexpr ck_tile::index_t K_Tile = {t.tile_k};",
        f"    static constexpr ck_tile::index_t M_Warp = {t.warp_m};",
        f"    static constexpr ck_tile::index_t N_Warp = {t.warp_n};",
        f"    static constexpr ck_tile::index_t K_Warp = {t.warp_k};",
        f"    static constexpr ck_tile::index_t M_Warp_Tile = {t.warp_tile_m};",
        f"    static constexpr ck_tile::index_t N_Warp_Tile = {t.warp_tile_n};",
        f"    static constexpr ck_tile::index_t K_Warp_Tile = {t.warp_tile_k};",
        "",
        f"    static constexpr bool kPadM = {'true' if trait.pad_m else 'false'};",
        f"    static constexpr bool kPadN = {'true' if trait.pad_n else 'false'};",
        f"    static constexpr bool kPadK = {'true' if trait.pad_k else 'false'};",
        "",
        f"    static constexpr bool DoubleSmemBuffer = {double_smem_buffer};",
        "    static constexpr bool TransposeC = false;",
        "    static constexpr bool UseStructuredSparsity = false;",
        "    static constexpr bool Preshuffle = false;",
        "    static constexpr bool PermuteA = false;",
        "    static constexpr bool PermuteB = false;",
        "",
        f"    static constexpr ck_tile::index_t NumWaveGroups = {num_wave_groups};",
        f"    static constexpr int kBlockPerCu = {block_per_cu};",
        f"    static constexpr ck_tile::index_t TileParitionerGroupNum = {tp_group_num};",
        f"    static constexpr ck_tile::index_t TileParitionerM01 = {tp_m01};",
        "",
        f"    static constexpr auto Scheduler = {_SCHEDULER_MAP[trait.scheduler]};",
        f"    static constexpr ck_tile::GemmPipeline Pipeline = {_PIPELINE_MAP[trait.pipeline]};",
        "};",
        "",
        "// -- spec.data.{dtype_a,b,c,acc,layout} -> CK Tile aliases.",
        f"using ADataType = {_DTYPE_MAP[spec.data.dtype_a]};",
        f"using BDataType = {_DTYPE_MAP[spec.data.dtype_b]};",
        f"using AccDataType = {_DTYPE_MAP[spec.data.dtype_acc]};",
        f"using CDataType = {_DTYPE_MAP[spec.data.dtype_c]};",
        "using DsDataType = ck_tile::tuple<>;",
        f"using ALayout = {a_layout};",
        f"using BLayout = {b_layout};",
        "using DsLayout = ck_tile::tuple<>;",
        f"using ELayout = {c_layout};",
        "using CDEElementWise = ck_tile::element_wise::PassThrough;",
        "",
        "// -- Kernel composition (one-to-one with universal_gemm_invoker.hpp).",
        "using GemmShape = ck_tile::TileGemmShape<",
        "    ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,",
        "    ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,",
        "    ck_tile::sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>,",
        "    GemmConfig::PermuteA,",
        "    GemmConfig::PermuteB>;",
        "",
        "using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<",
        "    GemmShape,",
        "    GemmConfig::TileParitionerGroupNum,",
        "    GemmConfig::TileParitionerM01>;",
        "",
        "using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<",
        "    GemmConfig::kPadM, GemmConfig::kPadN, GemmConfig::kPadK,",
        "    GemmConfig::DoubleSmemBuffer,",
        "    ALayout, BLayout, ELayout,",
        "    GemmConfig::TransposeC,",
        "    GemmConfig::UseStructuredSparsity,",
        f"    /*Persistent=*/{persistent},",
        "    GemmConfig::NumWaveGroups,",
        "    GemmConfig::Preshuffle>;",
        "",
        "using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<",
        "    ADataType, BDataType, AccDataType,",
        "    GemmShape, GemmUniversalTraits, GemmConfig::Scheduler>;",
        "",
        "using GemmPipeline =",
        "    typename ::PipelineTypeTraits<GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;",
        "",
        "using GemmEpilogue = ck_tile::CShuffleEpilogue<",
        "    ck_tile::CShuffleEpilogueProblem<",
        "        ADataType, BDataType, DsDataType, AccDataType, CDataType,",
        "        DsLayout, ELayout, CDEElementWise,",
        "        TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,",
        "        GemmConfig::M_Warp, GemmConfig::N_Warp,",
        "        GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile,",
        "        UniversalGemmProblem::TransposeC,",
        "        GemmConfig::NumWaveGroups,",
        "        /*FixedVectorSize=*/false,",
        "        /*VectorSizeC=*/1,",
        "        /*BlockedXDLN_PerWarp=*/1,",
        "        GemmConfig::DoubleSmemBuffer>>;",
        "",
        "using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;",
        "",
        f"}} // namespace ck_dsl_emit_{name}",
        "",
        "// -- Host launcher. ABI: void*-friendly so a C / Python ctypes shim",
        "// can call this directly. Returns the elapsed kernel time in ms,",
        "// or -1.0f if the kernel rejects the arguments (mirrors how",
        "// universal_gemm_invoker.hpp's std::runtime_error path would surface",
        "// via a non-throwing return).",
        f'extern "C" float launch_{name}(',
        "    const void* a_ptr,",
        "    const void* b_ptr,",
        "    void* c_ptr,",
        "    ck_tile::index_t M,",
        "    ck_tile::index_t N,",
        "    ck_tile::index_t K,",
        "    ck_tile::index_t stride_a,",
        "    ck_tile::index_t stride_b,",
        "    ck_tile::index_t stride_c,",
        "    ck_tile::index_t k_batch,",
        "    hipStream_t stream)",
        "{",
        f"    using namespace ck_dsl_emit_{name};",
        "    ck_tile::GemmHostArgs args;",
        "    args.a_ptr = a_ptr;",
        "    args.b_ptr = b_ptr;",
        "    args.e_ptr = c_ptr;",
        "    args.M = M;",
        "    args.N = N;",
        "    args.K = K;",
        "    args.stride_A = stride_a;",
        "    args.stride_B = stride_b;",
        "    args.stride_E = stride_c;",
        "    args.k_batch = k_batch;",
        "",
        "    ck_tile::stream_config s;",
        "    s.stream_id_ = stream;",
        "    s.time_kernel_ = false;",
        "",
        "    auto kargs = Kernel::MakeKernelArgs(args);",
        "    const dim3 grids = Kernel::GridSize(args.M, args.N, args.k_batch);",
        "    const dim3 blocks = Kernel::BlockSize();",
        "    if (!Kernel::IsSupportedArgument(kargs)) { return -1.0f; }",
        "",
        "    return ck_tile::launch_kernel(",
        "        s,",
        "        ck_tile::make_kernel<GemmConfig::kBlockPerCu>(",
        "            Kernel{}, grids, blocks, 0, kargs));",
        "}",
        "",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------
# ImplicitGemmConvSpec -> CK Tile grouped-convolution-forward source
# ---------------------------------------------------------------------


def lower_implicit_gemm_conv_to_cktile(
    spec: Any, *, kernel_name: str | None = None
) -> str:
    """Emit a CK Tile grouped-convolution-forward source for a
    :class:`ImplicitGemmConvSpec`.

    Composes the same templates as
    ``example/ck_tile/20_grouped_convolution/grouped_convolution_forward_invoker.hpp``,
    parameterised by the spec's tile / warp / MFMA / pipeline / scheduler /
    pad choices. Like the GEMM emitter, this is a source-level analog of
    the runtime path -- compilable HIP C++ that links against the CK Tile
    headers and reaches the same kernel composition.

    v1 supports the NHWC × KRSC -> NHWK implicit-GEMM family (single
    spatial group, fp16 in/out, fp32 acc). The conv-specific spatial
    parameters (R, S, stride, pad, dilation) live in :class:`ConvProblem`
    and are passed to the launcher at runtime via
    ``ck_tile::GroupedConvFwdHostArgs``.
    """
    from ..instances.conv_implicit_gemm import ImplicitGemmConvSpec

    if not isinstance(spec, ImplicitGemmConvSpec):
        raise TypeError(
            f"lower_implicit_gemm_conv_to_cktile expects ImplicitGemmConvSpec, "
            f"got {type(spec).__name__}"
        )

    # The conv pipeline / epilogue knobs map straight onto the GEMM ones
    # because CK Tile's conv forward is implicit-GEMM internally.
    pipeline = spec.pipeline
    epilogue = spec.epilogue
    if pipeline not in _PIPELINE_MAP:
        raise NotImplementedError(
            f"conv pipeline {pipeline!r}; supported {sorted(_PIPELINE_MAP)}"
        )
    if epilogue not in ("cshuffle", "default"):
        raise NotImplementedError(
            f"conv epilogue {epilogue!r}; supported 'cshuffle' / 'default'"
        )

    name = kernel_name or spec.kernel_name()

    # Conv input is f16 NHWC; CK Tile's ImageLayout/WeightLayout encode
    # this via ``GNHWC`` (input) / ``GKYXC`` (weights) / ``GNHWK`` (output).
    in_dtype = "ck_tile::half_t"
    wei_dtype = "ck_tile::half_t"
    out_dtype = "ck_tile::half_t"
    acc_dtype = "float"

    block_per_cu = 1
    num_wave_groups = 1
    num_groups_to_merge = 1
    vector_size_a = 8
    vector_size_b = 8
    vector_size_c = 8
    double_smem_buffer = "true" if pipeline == "compv4" else "false"

    parts = [
        "// =========================================================",
        "// Auto-generated by ck_dsl.core.lower_cktile from",
        "//   ImplicitGemmConvSpec(",
        f"//     name={spec.name!r}, problem={spec.problem!r},",
        f"//     tile_m={spec.tile_m}, tile_n={spec.tile_n}, tile_k={spec.tile_k},",
        f"//     warp_m={spec.warp_m}, warp_n={spec.warp_n},",
        f"//     warp_tile=({spec.warp_tile_m},{spec.warp_tile_n},{spec.warp_tile_k}),",
        f"//     pipeline={pipeline!r}, epilogue={epilogue!r})",
        "//",
        "// Mirrors example/ck_tile/20_grouped_convolution/"
        "grouped_convolution_forward_invoker.hpp.",
        "// =========================================================",
        "",
        "#include <ck_tile/core.hpp>",
        "#include <ck_tile/host.hpp>",
        "#include <ck_tile/ops/gemm.hpp>",
        "#include <ck_tile/ops/grouped_convolution.hpp>",
        "#include <ck_tile/ops/epilogue.hpp>",
        "",
        _PIPELINE_TYPE_TRAITS,
        "",
        f"namespace ck_dsl_emit_{name} {{",
        "",
        "struct ConvConfig {",
        f"    static constexpr ck_tile::index_t M_Tile = {spec.tile_m};",
        f"    static constexpr ck_tile::index_t N_Tile = {spec.tile_n};",
        f"    static constexpr ck_tile::index_t K_Tile = {spec.tile_k};",
        f"    static constexpr ck_tile::index_t M_Warp = {spec.warp_m};",
        f"    static constexpr ck_tile::index_t N_Warp = {spec.warp_n};",
        "    static constexpr ck_tile::index_t K_Warp = 1;",
        f"    static constexpr ck_tile::index_t M_Warp_Tile = {spec.warp_tile_m};",
        f"    static constexpr ck_tile::index_t N_Warp_Tile = {spec.warp_tile_n};",
        f"    static constexpr ck_tile::index_t K_Warp_Tile = {spec.warp_tile_k};",
        "",
        f"    static constexpr bool DoubleSmemBuffer = {double_smem_buffer};",
        f"    static constexpr ck_tile::index_t NumWaveGroups = {num_wave_groups};",
        f"    static constexpr ck_tile::index_t NumGroupsToMerge = {num_groups_to_merge};",
        f"    static constexpr int kBlockPerCu = {block_per_cu};",
        f"    static constexpr ck_tile::index_t VectorSizeA = {vector_size_a};",
        f"    static constexpr ck_tile::index_t VectorSizeB = {vector_size_b};",
        f"    static constexpr ck_tile::index_t VectorSizeC = {vector_size_c};",
        "",
        "    // Scheduler/Pipeline mirror the GEMM lowering's choices.",
        "    static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Intrawave;",
        f"    static constexpr ck_tile::GemmPipeline Pipeline = {_PIPELINE_MAP[pipeline]};",
        "};",
        "",
        "// Conv spatial layouts: NHWGC input, GKYXC weights, NHWGK output, 2D spatial.",
        "// These are the only layout triples ``GroupedConvolutionForwardKernel``'s",
        "// internal ``MakeADescriptor_M_K`` / ``MakeBDescriptor_N_K`` /",
        "// ``MakeCDescriptor_M_N`` overloads accept for NDimSpatial==2 (see",
        "// ck_tile/ops/grouped_convolution/utils/transform_conv_fwd_to_gemm.hpp).",
        "static constexpr ck_tile::index_t NDimSpatial = 2;",
        f"using InDataType = {in_dtype};",
        f"using WeiDataType = {wei_dtype};",
        f"using AccDataType = {acc_dtype};",
        f"using OutDataType = {out_dtype};",
        "using InLayout = ck_tile::tensor_layout::convolution::NHWGC;",
        "using WeiLayout = ck_tile::tensor_layout::convolution::GKYXC;",
        "using OutLayout = ck_tile::tensor_layout::convolution::NHWGK;",
        "using DsDataType = ck_tile::tuple<>;",
        "using DsLayout = ck_tile::tuple<>;",
        "using CDElementWise = ck_tile::element_wise::PassThrough;",
        "",
        "// Implicit-GEMM tile shape -- same TileGemmShape as a pure GEMM, but",
        "// the M / N / K axes are the implicit-GEMM packings of the conv.",
        "using GemmShape = ck_tile::TileGemmShape<",
        "    ck_tile::sequence<ConvConfig::M_Tile, ConvConfig::N_Tile, ConvConfig::K_Tile>,",
        "    ck_tile::sequence<ConvConfig::M_Warp, ConvConfig::N_Warp, ConvConfig::K_Warp>,",
        "    ck_tile::sequence<ConvConfig::M_Warp_Tile, ConvConfig::N_Warp_Tile, ConvConfig::K_Warp_Tile>>;",
        "",
        "static constexpr auto ConvSpec = ck_tile::ConvolutionSpecialization::Default;",
        "using GroupedConvTraitsType = ck_tile::GroupedConvTraits<",
        "    NDimSpatial,",
        "    ConvSpec,",
        "    InLayout, WeiLayout, DsLayout, OutLayout,",
        "    ConvConfig::VectorSizeA, ConvConfig::VectorSizeB, ConvConfig::VectorSizeC,",
        "    ConvConfig::NumGroupsToMerge>;",
        "",
        "using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<",
        "    GemmShape,",
        "    GroupedConvTraitsType::FixedGemmParams::TilePartitionerGroupNum,",
        "    GroupedConvTraitsType::FixedGemmParams::TilePartitionerM01>;",
        "",
        "using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<",
        "    GroupedConvTraitsType::FixedGemmParams::kPadM,",
        "    GroupedConvTraitsType::FixedGemmParams::kPadN,",
        "    GroupedConvTraitsType::FixedGemmParams::kPadK,",
        "    ConvConfig::DoubleSmemBuffer,",
        "    typename GroupedConvTraitsType::AsLayoutFwd,",
        "    typename GroupedConvTraitsType::BsLayoutFwd,",
        "    typename GroupedConvTraitsType::CLayoutFwd,",
        "    GroupedConvTraitsType::FixedGemmParams::TransposeC,",
        "    GroupedConvTraitsType::FixedGemmParams::UseStructuredSparsity,",
        "    GroupedConvTraitsType::FixedGemmParams::Persistent,",
        "    ConvConfig::NumWaveGroups>;",
        "",
        "using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<",
        "    InDataType, WeiDataType, AccDataType,",
        "    GemmShape, GemmUniversalTraits, ConvConfig::Scheduler,",
        "    ck_tile::element_wise::PassThrough,",
        "    ck_tile::element_wise::PassThrough,",
        "    OutDataType,",
        "    GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,",
        "    GroupedConvTraitsType::VectorSizeA,",
        "    GroupedConvTraitsType::VectorSizeB>;",
        "",
        "using GemmPipeline =",
        "    typename ::PipelineTypeTraits<ConvConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;",
        "",
        "using ConvEpilogue = ck_tile::CShuffleEpilogue<",
        "    ck_tile::CShuffleEpilogueProblem<",
        "        InDataType, WeiDataType, DsDataType, AccDataType, OutDataType,",
        "        typename GroupedConvTraitsType::ImplicitGemmDsLayout,",
        "        typename GroupedConvTraitsType::FixedGemmParams::ELayout,",
        "        CDElementWise,",
        "        TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,",
        "        ConvConfig::M_Warp, ConvConfig::N_Warp,",
        "        ConvConfig::M_Warp_Tile, ConvConfig::N_Warp_Tile, ConvConfig::K_Warp_Tile,",
        "        GroupedConvTraitsType::FixedGemmParams::TransposeC,",
        "        ConvConfig::NumWaveGroups,",
        "        GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,",
        "        GroupedConvTraitsType::VectorSizeC>>;",
        "",
        "// GroupedConvolutionForwardKernel takes 4 type parameters; the",
        "// NDimSpatial value lives inside GroupedConvTraitsType already.",
        "// Wiring exactly matches example/ck_tile/20_grouped_convolution/",
        "// grouped_convolution_forward_invoker.hpp.",
        "using Kernel = ck_tile::GroupedConvolutionForwardKernel<",
        "    GroupedConvTraitsType,",
        "    TilePartitioner,",
        "    GemmPipeline,",
        "    ConvEpilogue>;",
        "",
        f"}} // namespace ck_dsl_emit_{name}",
        "",
        "// Host launcher: takes a ``GroupedConvFwdHostArgs<PassThrough>``",
        "// (the public host-side args struct in CK Tile) and forwards to",
        "// ``Kernel::MakeKernelArgs``, exactly as",
        "// ``example/ck_tile/20_grouped_convolution/grouped_convolution_forward_invoker.hpp`` does.",
        f'extern "C" float launch_{name}(',
        "    const ck_tile::GroupedConvFwdHostArgs<ck_tile::element_wise::PassThrough>& args,",
        "    hipStream_t stream)",
        "{",
        f"    using namespace ck_dsl_emit_{name};",
        "    ck_tile::stream_config s;",
        "    s.stream_id_ = stream;",
        "    s.time_kernel_ = false;",
        "",
        "    auto kargs = Kernel::MakeKernelArgs(args);",
        "    const dim3 grids = Kernel::GridSize(kargs);",
        "    const dim3 blocks = Kernel::BlockSize();",
        "    if (!Kernel::IsSupportedArgument(kargs)) { return -1.0f; }",
        "",
        "    return ck_tile::launch_kernel(",
        "        s,",
        "        ck_tile::make_kernel<ConvConfig::kBlockPerCu>(",
        "            Kernel{}, grids, blocks, 0, kargs));",
        "}",
        "",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------


def lower_spec_to_cktile(spec: Any, *, kernel_name: str | None = None) -> str:
    """Dispatch to the right CK Tile emitter for ``spec``.

    Supported spec types are listed in the module docstring.
    """
    # Lazy-imports keep this entry point usable when only one of the
    # instance subpackages is wired up.
    from ..instances.gemm_universal import UniversalGemmSpec
    from ..instances.conv_implicit_gemm import ImplicitGemmConvSpec

    if isinstance(spec, UniversalGemmSpec):
        return lower_universal_gemm_to_cktile(spec, kernel_name=kernel_name)
    if isinstance(spec, ImplicitGemmConvSpec):
        return lower_implicit_gemm_conv_to_cktile(spec, kernel_name=kernel_name)
    raise NotImplementedError(
        f"no CK Tile lowering for {type(spec).__name__}; supported: "
        "UniversalGemmSpec, ImplicitGemmConvSpec"
    )
