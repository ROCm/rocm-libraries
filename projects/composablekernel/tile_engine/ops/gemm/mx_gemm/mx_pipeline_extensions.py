# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Per-pipeline extensions of the MX GEMM tile_engine builder, dispatcher codegen and sweep.

Most MX GEMM pipelines are described by the shared tile and trait configs alone. A pipeline that
needs more (its own tuning, validity rules, kernel name suffix, C++ instance, compile flags, or
shape traits fitted to each problem) implements MxPipelineExtension and registers it in
EXTENSIONS. The shared code (gemm_validation_utils.mx_pipeline_extension, the instance builder,
the codegen, mx_gemm_utils and the sweep driver) looks the pipeline up and calls its hooks, and
never names the pipeline. Pipelines without an extension take no tuning.

In the hooks, tile_config is the tile_engine dict (tile_m ... warp_tile_k), dtypes are the
tile_engine names (fp4, fp8; fp16 for C) and gpu_target is the base architecture (gfx1250).
"""

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _sibling(name: str):
    """A module of this directory, loaded once by path under its own name."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name(f"{name}.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


mx_tdm_scheduled = _sibling("mx_tdm_scheduled")


class MxPipelineExtension:
    """The hooks of one pipeline beyond the shared tile and trait configs."""

    pipeline: str = ""
    #: The epilogues the pipeline takes; the first is the default.
    epilogues: Tuple[str, ...] = ()
    #: hipcc flags added to the shared MX GEMM flags.
    compile_flags: Tuple[str, ...] = ()
    #: The keys of its tuning (pipeline_tuning_config section, MxGemmKernelConfig.tuning).
    tuning_keys: Tuple[str, ...] = ()
    #: Built only through the dispatcher, not the tile_engine CMake build.
    dispatcher_only: bool = False
    #: fit_to_problem changes the tuning per problem (the sweep fits each config to it).
    shape_traits: bool = False

    def normalize_tuning(self, tuning: Optional[Dict]) -> Dict[str, object]:
        """The full tuning (defaults filled in); unknown keys raise ValueError."""
        raise NotImplementedError

    def expand_tuning(self, section: Optional[Dict]) -> List[Dict[str, object]]:
        """The tunings of its pipeline_tuning_config section ({"key": {"values": [...]}})."""
        raise NotImplementedError

    def trait_reject_reason(
        self, epilogue: str, persistent: bool, pads: Sequence[Optional[bool]]
    ) -> str:
        """Why the trait combination is invalid ("" if valid; a pad of None is not checked)."""
        raise NotImplementedError

    def reject_reason(
        self,
        tile_config: Dict,
        a_dtype: str,
        b_dtype: str,
        c_dtype: str,
        gpu_target: str,
        tuning: Optional[Dict] = None,
    ) -> str:
        """Why the instance is invalid ("" if valid). With tuning None, why the tile is invalid
        under every tuning."""
        raise NotImplementedError

    def name_suffix(
        self, tile_config: Dict, tuning: Dict, a_dtype: str, b_dtype: str, c_dtype: str
    ) -> str:
        """The kernel name's suffix for the tuning (distinct tunings, distinct names)."""
        raise NotImplementedError

    def instance_code(
        self, builder, kernel_name: str, tile_config: Dict, tuning: Dict
    ) -> str:
        """The C++ instance header (SelectedKernel and KERNEL_NAME) for the builder."""
        raise NotImplementedError

    def fit_to_problem(
        self,
        tile_config: Dict,
        tuning: Optional[Dict],
        a_dtype: str,
        M: int,
        N: int,
        K: int,
    ) -> Optional[Dict]:
        """The tuning that runs an M x N x K problem, or None if the instance cannot run it."""
        return tuning


class TdmScheduledExtension(MxPipelineExtension):
    """comp_tdm_scheduled (MxGemmPipelineAgBgCrCompTDMScheduled): K-major scales, the TDM
    epilogue (MxTdmStoreEpilogue: TdmEpilogue, E through LDS), the cluster and policy knobs as
    a tuning, and shape traits (MxTdmScheduledGemmTraits) fixed at compile time. The rules are
    mx_tdm_scheduled.py's, which mirrors the C++ GetUnsupportedReason."""

    pipeline = "comp_tdm_scheduled"
    epilogues = ("tdm",)
    # As its CMake targets (example 56, tests, tile_engine).
    compile_flags = (
        "-mllvm",
        "-amdgpu-expert-scheduling-mode",
        "-mllvm",
        "-enable-post-misched=false",
        "-mllvm",
        "-greedy-regclass-priority-trumps-globalness",
    )
    tuning_keys = (*mx_tdm_scheduled.TUNING_DEFAULTS, *mx_tdm_scheduled.SHAPE_DEFAULTS)
    dispatcher_only = True
    shape_traits = True

    def normalize_tuning(self, tuning):
        return mx_tdm_scheduled.normalize_tuning(tuning)

    def expand_tuning(self, section):
        return mx_tdm_scheduled.expand_tuning_config(section)

    def trait_reject_reason(self, epilogue, persistent, pads):
        if epilogue not in self.epilogues:
            return f"{self.pipeline} needs an epilogue of {self.epilogues}"
        if persistent:
            return f"{self.pipeline} takes no persistence"
        if any(pads):
            return f"{self.pipeline} takes no padding (edge_m/edge_n)"
        return ""

    def _spec(self, tile_config, tuning, a_dtype, b_dtype, c_dtype, **problem):
        out = mx_tdm_scheduled.TE_OUT_DTYPES[c_dtype]
        return mx_tdm_scheduled.spec_from_tile(
            tile_config, tuning, a_dtype, b_dtype, out, **problem
        )

    def reject_reason(
        self, tile_config, a_dtype, b_dtype, c_dtype, gpu_target, tuning=None
    ):
        if gpu_target != "gfx1250":
            return f"{self.pipeline} requires gfx1250"
        if (
            c_dtype not in mx_tdm_scheduled.TE_OUT_DTYPES
            or a_dtype not in mx_tdm_scheduled.MX_DTYPES
            or b_dtype not in mx_tdm_scheduled.MX_DTYPES
        ):
            return f"{self.pipeline} does not support {a_dtype}x{b_dtype}->{c_dtype}"
        candidates = (
            [tuning] if tuning is not None else mx_tdm_scheduled.schedule_tunings()
        )
        first = None
        for candidate in candidates:
            try:
                spec = self._spec(tile_config, candidate, a_dtype, b_dtype, c_dtype)
                reason = mx_tdm_scheduled.validate_instance(spec, candidate)
            except ValueError as e:
                return str(e)
            if reason == "ok":
                return ""
            first = first or reason
        return f"{self.pipeline}: {first}"

    def name_suffix(self, tile_config, tuning, a_dtype, b_dtype, c_dtype):
        spec = self._spec(tile_config, tuning, a_dtype, b_dtype, c_dtype)
        return mx_tdm_scheduled.instance_name_suffix(spec, tuning)

    def fit_to_problem(self, tile_config, tuning, a_dtype, M, N, K):
        """None if tile_k > K, else the tuning with the problem's shape traits
        (mx_tdm_scheduled.shape_tuning). A tile the spec rejects, which the build reports,
        keeps its tuning."""
        try:
            spec = mx_tdm_scheduled.spec_from_tile(
                tile_config, tuning, a_dtype, M=M, N=N, K=K
            )
        except ValueError:
            return tuning
        return mx_tdm_scheduled.shape_tuning(spec, tuning)

    def instance_code(self, builder, kernel_name, tile_config, tuning):
        t = tile_config
        spec = self._spec(
            tile_config,
            tuning,
            builder.datatype,
            builder.datatype,
            builder._c_datatype(),
        )
        return f"""// Generated kernel instance for {kernel_name}
#pragma once

#include <stdexcept>
#include <string>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
{builder.populate_kernel_dtype_layout()}
// Kernel name for display
constexpr const char* KERNEL_NAME = "{kernel_name}";

// comp_tdm_scheduled (MxGemmPipelineAgBgCrCompTDMScheduled): K-major scales, E = A * B^T.
struct SelectedKernel {{
    static constexpr ck_tile::index_t BlockSize = 128;
    static constexpr ck_tile::index_t TileM = {t["tile_m"]};
    static constexpr ck_tile::index_t TileN = {t["tile_n"]};
    static constexpr ck_tile::index_t TileK = {t["tile_k"]};
    static constexpr ck_tile::index_t WarpPerBlock_M = {t["warp_m"]};
    static constexpr ck_tile::index_t WarpPerBlock_N = {t["warp_n"]};
    static constexpr ck_tile::index_t WarpPerBlock_K = 1;
    static constexpr ck_tile::index_t WarpTileM = {t["warp_tile_m"]};
    static constexpr ck_tile::index_t WarpTileN = 16;
    static constexpr ck_tile::index_t WarpTileK = 128;
    static constexpr bool Preshuffle = false;
    static constexpr bool KMajorScales = true;
    // The B reshuffle interface the profiler names (unused: B is not preshuffled).
    static constexpr ck_tile::index_t N_Warp_Tile = WarpTileN;
    static constexpr ck_tile::index_t K_Warp_Tile = WarpTileK;
    static constexpr ck_tile::index_t BContiguousItemsPerAccess =
        std::is_same_v<BDataType, ck_tile::pk_fp4_t> ? 32 : 16;
    static constexpr bool BPreshuffleLaneInterleavedK = true;

    using Cluster = {mx_tdm_scheduled.cpp_cluster_sequence(tuning)};
    using Policy = {mx_tdm_scheduled.cpp_policy_type(tuning)};
    using Traits = {mx_tdm_scheduled.cpp_kernel_traits(spec, tuning)};
    using Pipeline = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduled<
        ck_tile::MxGemmTdmScheduledProblem<
            ADataType,
            BDataType,
            CDataType,
            ck_tile::ClusterTileGemmShape<Cluster,
                                          ck_tile::sequence<TileM, TileN, TileK>,
                                          ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, 1>,
                                          ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>,
            Traits>,
        Policy>;
    // One kernel per TailNumber of the pipeline (Pipeline::WithTailNum).
    template <typename P>
    using KernelOf = ck_tile::MxGemmKernel<
        ck_tile::GemmTile2DNMajorPartitioner<typename P::BlockGemmShape>,
        P,
        ck_tile::MxTdmStoreEpilogue<P>>;

    // The pipeline's TailHandler picks the kernel of the loop's tile count on the host.
    static float launch(const MxGemmHostArgs& args, const ck_tile::stream_config& stream)
    {{
        const ck_tile::index_t full_tiles = Pipeline::GetFullTiles(args.K);
        return Pipeline::TailHandler(
            [&](auto, auto tail_num) {{
                return launch_kernel<
                    KernelOf<typename Pipeline::template WithTailNum<decltype(tail_num)::value>>>(
                    args, stream);
            }},
            Pipeline::BlockHasHotloop(full_tiles),
            Pipeline::GetBlockLoopTailNum(full_tiles));
    }}

    template <typename Kernel>
    static float launch_kernel(const MxGemmHostArgs& args, const ck_tile::stream_config& stream)
    {{
        const auto kargs = Kernel::MakeKernelArgs(args);
        if(!Kernel::IsSupportedArgument(kargs))
        {{
            const char* why = Kernel::GetUnsupportedReason(kargs);
            throw std::runtime_error(std::string("Arguments not supported: ") +
                                     (why ? why : "see CK_TILE_LOGGING"));
        }}
        const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
        const dim3 blocks = Kernel::BlockSize();
        if constexpr(Kernel::ClusterLaunch)
        {{
#if CK_TILE_ENABLE_CLUSTER_LAUNCH
            return ck_tile::launch_kernel(
                stream,
                ck_tile::make_kernel<1>(Kernel{{}}, Kernel::ClusterSize(), grids, blocks, 0, kargs));
#else
            throw std::runtime_error(Kernel::GetName() + ": this build has no cluster launch");
#endif
        }}
        else
        {{
            return ck_tile::launch_kernel(
                stream, ck_tile::make_kernel<1>(Kernel{{}}, grids, blocks, 0, kargs));
        }}
    }}
}};
"""


#: The registered extensions by pipeline.
EXTENSIONS: Dict[str, MxPipelineExtension] = {
    ext.pipeline: ext for ext in (TdmScheduledExtension(),)
}


def extension(pipeline: Optional[str]) -> Optional[MxPipelineExtension]:
    """The extension of a pipeline, or None for a pipeline the shared configs describe."""
    return EXTENSIONS.get(pipeline)
