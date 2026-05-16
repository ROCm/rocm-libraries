# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Lowering registry for scheduled fusion regions.

This is the layer that turns abstract :class:`FusionRegion` decisions
from the scheduler into concrete CK DSL kernels (and launchers). Each
:class:`RegionLowerer` is a small object answering four questions:

* ``can_lower(region)``      - is this region within my coverage?
* ``candidates(region)``     - what spec configurations should we try?
* ``build(config)``          - compile + load one candidate.
* ``launch_args(region, …)`` - turn user tensors into a kernel arg dict.

The default registry covers the three families the plan calls out:

* :class:`GemmEpilogueLowerer` - GEMM (with optional fused epilogue
  chain) backed by :func:`build_universal_gemm`.
* :class:`ElementwiseLowerer`  - generic pointwise vector kernels
  backed by :func:`build_elementwise`.
* :class:`ReductionLowerer`    - row reductions backed by
  :func:`build_reduce2d`.

Concrete lowerers are intentionally small and explainable: each one
maps a ``FusionRegion`` to a single existing instance builder plus a
short sweep of autotune candidates. Adding a new family (e.g. paged
attention) means writing one more :class:`RegionLowerer` subclass and
registering it; everything else (graph capture, legality, scheduling,
workspace planning, autotuning) stays the same.

Keep :class:`ExplainOnlyLowerer` for tests and incremental bring-up:
it participates in legality/explanation without committing to a
build. The fusion scaffolding in :mod:`fusion_ir`,
:mod:`fusion_scheduler`, and :mod:`fusion_memory` runs the
explainable plan through it without ever touching the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple

from .autotune import AutotuneConfig
from .fusion_ir import FusionGraph, FusionRegion
from .fusion_legalize import LegalResult


__all__ = [
    "BuiltRegion",
    "ElementwiseLowerer",
    "ExplainOnlyLowerer",
    "GemmEpilogueLowerer",
    "LoweringRegistry",
    "ReductionLowerer",
    "RegionLowerer",
    "default_lowering_registry",
]


@dataclass
class BuiltRegion:
    """The artefact produced by :meth:`RegionLowerer.build`.

    ``launcher`` is a :class:`KernelLauncher` (or pipeline). ``spec`` is
    whatever spec dataclass the lowerer chose (helpful for debugging).
    ``block_size`` is the block dim X; the lowerer's ``launch_args``
    callback uses it to construct the launch grid. ``extra`` carries
    lowerer-specific metadata (e.g. ``vec`` for elementwise).
    """

    launcher: Any
    spec: Any
    block_size: int
    extra: Dict[str, Any] = field(default_factory=dict)


class RegionLowerer(Protocol):
    name: str

    def can_lower(self, graph: FusionGraph, region: FusionRegion) -> LegalResult: ...

    def candidates(
        self, graph: FusionGraph, region: FusionRegion
    ) -> List[AutotuneConfig]: ...

    def build(self, config: AutotuneConfig) -> BuiltRegion: ...

    def launch_args(
        self,
        graph: FusionGraph,
        region: FusionRegion,
        runtime_args: Dict[str, Any],
        built: BuiltRegion,
    ) -> Tuple[Dict[str, Any], Tuple[int, int, int]]: ...


@dataclass
class LoweringRegistry:
    """Map scheduled region kinds to lowerers."""

    _lowerers: Dict[str, RegionLowerer]

    def __init__(self) -> None:
        self._lowerers = {}

    def register(self, lowerer: RegionLowerer) -> None:
        self._lowerers[lowerer.name] = lowerer

    def get(self, name: Optional[str]) -> Optional[RegionLowerer]:
        if name is None:
            return None
        return self._lowerers.get(name)

    def require(self, name: str) -> RegionLowerer:
        lowerer = self.get(name)
        if lowerer is None:
            raise KeyError(f"no lowerer registered for {name!r}")
        return lowerer


class ExplainOnlyLowerer:
    """A stub lowerer useful while bootstrapping or for tests.

    It participates in legality and plan explanation but intentionally
    does not build kernels. Concrete GEMM/elementwise/reduce lowerers
    replace it once they are wired through to actual codegen.
    """

    def __init__(self, name: str, supported_kinds: set) -> None:
        self.name = name
        self.supported_kinds = set(supported_kinds)

    def can_lower(self, graph: FusionGraph, region: FusionRegion) -> LegalResult:
        if region.kind in self.supported_kinds:
            return LegalResult.success()
        return LegalResult.failure(
            f"{self.name}: unsupported region kind {region.kind!r}"
        )

    def candidates(
        self, graph: FusionGraph, region: FusionRegion
    ) -> List[AutotuneConfig]:
        return []

    def build(self, config: AutotuneConfig) -> BuiltRegion:
        raise NotImplementedError(f"{self.name} is explain-only")

    def launch_args(
        self,
        graph: FusionGraph,
        region: FusionRegion,
        runtime_args: Dict[str, Any],
        built: BuiltRegion,
    ) -> Tuple[Dict[str, Any], Tuple[int, int, int]]:
        return dict(runtime_args), (1, 1, 1)


# ---------------------------------------------------------------------
# Concrete lowerers
# ---------------------------------------------------------------------


def _dtype_str_for_tensor(graph: FusionGraph, name: str) -> str:
    """Canonicalize a graph tensor's dtype string ("fp16" vs "f16")."""
    t = graph.tensors[name]
    s = t.dtype.lower()
    if s in ("f16", "float16"):
        return "fp16"
    if s in ("bf16", "bfloat16"):
        return "bf16"
    if s in ("f32", "float32"):
        return "fp32"
    return s


def _region_inputs_outputs(
    graph: FusionGraph, region: FusionRegion
) -> Tuple[List[str], List[str]]:
    return list(region.inputs), list(region.outputs)


class GemmEpilogueLowerer:
    """Lower a ``gemm`` / ``gemm_epilogue`` region to ``UniversalGemmSpec``.

    Coverage
    --------
    * ``gemm``           : single matmul region, no fused pointwise.
    * ``gemm_epilogue``  : matmul + 1..N fused pointwise ops; each op
                          must be representable as an
                          :class:`EpilogueOp` subclass (relu/gelu/
                          silu/scale/bias/clamp/residual).

    The lowerer reads the region's first op as the matmul (the
    scheduler guarantees this for gemm regions) and the trailing ops
    as the epilogue chain, walking them in topological order.
    """

    name = "gemm_epilogue"
    supported_kinds = {"gemm", "gemm_epilogue"}

    # Default sweep tile choices for fp16. bf16 currently only supports
    # 16x16x{16,32} atoms on gfx950 so we fall back to a smaller set.
    _FP16_TILES: Tuple[Tuple[str, int, int, int, int, int, int, int, int], ...] = (
        ("t128x128x32", 128, 128, 32, 2, 2, 32, 32, 16),
        ("t128x128x64", 128, 128, 64, 2, 2, 32, 32, 16),
        ("t256x128x32", 256, 128, 32, 4, 2, 32, 32, 16),
    )
    _BF16_TILES: Tuple[Tuple[str, int, int, int, int, int, int, int, int], ...] = (
        ("t64x64x32", 64, 64, 32, 2, 2, 16, 16, 32),
        ("t128x64x32", 128, 64, 32, 4, 2, 16, 16, 32),
        ("t64x128x32", 64, 128, 32, 2, 4, 16, 16, 32),
    )

    def can_lower(self, graph: FusionGraph, region: FusionRegion) -> LegalResult:
        if region.kind not in self.supported_kinds:
            return LegalResult.failure(f"{self.name}: unsupported kind {region.kind!r}")
        if not region.op_names:
            return LegalResult.failure(f"{self.name}: empty region")
        head = graph.ops[region.op_names[0]]
        if head.kind != "matmul":
            return LegalResult.failure(
                f"{self.name}: head op must be matmul, got {head.kind!r}"
            )
        unsupported = []
        for op_name in region.op_names[1:]:
            kind = graph.ops[op_name].kind
            if kind not in {"add", "mul", "relu", "gelu", "silu", "clamp"}:
                unsupported.append(kind)
        if unsupported:
            return LegalResult.failure(
                f"{self.name}: unsupported epilogue ops {sorted(set(unsupported))}"
            )
        return LegalResult.success()

    def _epilogue_for_region(
        self, graph: FusionGraph, region: FusionRegion
    ) -> Tuple[Any, Optional[str], Tuple[str, ...]]:
        """Build a :class:`FusedEpilogue` reflecting the region's ops.

        Also returns ``(bias_name, residual_names)`` so the runtime
        argument plumbing knows which tensors to pass through.
        """
        from .fuse import (
            BiasAdd,
            Clamp,
            FusedEpilogue,
            GELU,
            ReLU,
            ResidualAdd,
            ResidualMul,
            Scale,
            SiLU,
        )

        ops_list: List[Any] = []
        bias_name: Optional[str] = None
        residual_names: List[str] = []
        # Skip the leading matmul.
        for op_name in region.op_names[1:]:
            op = graph.ops[op_name]
            if op.kind == "relu":
                ops_list.append(ReLU())
            elif op.kind == "gelu":
                ops_list.append(GELU())
            elif op.kind == "silu":
                ops_list.append(SiLU())
            elif op.kind == "clamp":
                lo = float(op.attrs.get("min", float("-inf")))
                hi = float(op.attrs.get("max", float("inf")))
                ops_list.append(Clamp(lo=lo, hi=hi))
            elif op.kind == "mul" and "scalar" in op.attrs:
                ops_list.append(Scale(float(op.attrs["scalar"])))
            elif op.kind == "add":
                # 1D placeholder -> bias; 2D placeholder -> residual.
                other = next(i for i in op.inputs if graph.tensors[i].is_input)
                if len(graph.tensors[other].shape) == 1 and bias_name is None:
                    bias_name = other
                    ops_list.append(BiasAdd())
                else:
                    residual_names.append(other)
                    ops_list.append(
                        ResidualAdd(param_name=f"residual_{len(residual_names) - 1}")
                    )
            elif op.kind == "mul":
                # ``mul`` with a 2D placeholder -> ResidualMul.
                other = next(i for i in op.inputs if graph.tensors[i].is_input)
                residual_names.append(other)
                ops_list.append(
                    ResidualMul(param_name=f"residual_mul_{len(residual_names) - 1}")
                )
            else:
                raise ValueError(
                    f"{self.name}: unsupported epilogue op kind {op.kind!r}"
                )
        return FusedEpilogue(ops=tuple(ops_list)), bias_name, tuple(residual_names)

    def _tile_choices(
        self, dtype: str
    ) -> Tuple[Tuple[str, int, int, int, int, int, int, int, int], ...]:
        if dtype in ("bf16", "bfloat16"):
            return self._BF16_TILES
        return self._FP16_TILES

    def candidates(
        self, graph: FusionGraph, region: FusionRegion
    ) -> List[AutotuneConfig]:
        from ..instances.gemm_universal import (
            DataSpec,
            TileSpec,
            TraitSpec,
            UniversalGemmSpec,
        )
        from .fuse import dtype_to_ir

        head = graph.ops[region.op_names[0]]
        a_name, b_name = head.inputs
        dtype = _dtype_str_for_tensor(graph, a_name)
        ir_dtype = dtype_to_ir(dtype)
        epilogue, bias_name, residual_names = self._epilogue_for_region(graph, region)
        epilogue = epilogue.with_dtype(ir_dtype)
        data = DataSpec(dtype_a=dtype, dtype_b=dtype, dtype_c=dtype, dtype_acc="fp32")
        configs: List[AutotuneConfig] = []
        for tile_name, tm, tn, tk, wm, wn, wtm, wtn, wtk in self._tile_choices(dtype):
            for wgm in (4, 8):
                spec = UniversalGemmSpec(
                    name=f"fused_{epilogue.kernel_name_suffix()}_{tile_name}_wgm{wgm}",
                    tile=TileSpec(
                        tile_m=tm,
                        tile_n=tn,
                        tile_k=tk,
                        warp_m=wm,
                        warp_n=wn,
                        warp_k=1,
                        warp_tile_m=wtm,
                        warp_tile_n=wtn,
                        warp_tile_k=wtk,
                    ),
                    trait=TraitSpec(
                        pipeline="compv4",
                        epilogue="cshuffle",
                        chiplet_swizzle=wgm > 0,
                        chiplet_wgm=wgm,
                    ),
                    data=data,
                )
                object.__setattr__(spec, "_fused_epilogue", epilogue)
                object.__setattr__(spec, "_bias_arg", bias_name)
                object.__setattr__(spec, "_residual_args", residual_names)
                configs.append(
                    AutotuneConfig(
                        spec=spec,
                        name=f"{tile_name}_wgm{wgm}_{dtype}",
                        extra={"bias": bias_name, "residuals": residual_names},
                    )
                )
        return configs

    def build(self, config: AutotuneConfig) -> BuiltRegion:
        from ..core.lower_llvm import lower_kernel_to_llvm
        from ..instances.gemm_universal import build_universal_gemm
        from ..runtime.comgr import build_hsaco_from_llvm_ir
        from ..runtime.launcher import KernelLauncher
        from .fuse import dtype_to_ir

        spec = config.spec
        epilogue = getattr(spec, "_fused_epilogue", None)
        bias_name = getattr(spec, "_bias_arg", None)
        residual_names = getattr(spec, "_residual_args", ())
        ir_dtype = dtype_to_ir(spec.data.dtype_a)
        kernel = build_universal_gemm(spec)
        ir = lower_kernel_to_llvm(kernel)
        hsaco, _ = build_hsaco_from_llvm_ir(ir)
        ptr_ty = f"ptr<{ir_dtype.name}, global>"
        sig: List[Mapping[str, Any]] = [
            {"name": "A", "type": ptr_ty, "size_bytes": 8},
            {"name": "B", "type": ptr_ty, "size_bytes": 8},
            {"name": "C", "type": ptr_ty, "size_bytes": 8},
            {"name": "M", "type": "i32", "size_bytes": 4},
            {"name": "N", "type": "i32", "size_bytes": 4},
            {"name": "K", "type": "i32", "size_bytes": 4},
        ]
        # Order must match the order ops appended their params in
        # ``FusedEpilogue.declare_params``. BiasAdd is appended first
        # (matches the legacy ``compile_fn`` ABI); residuals follow.
        if epilogue is not None:
            for op in epilogue.ops:
                pname = getattr(op, "param_name", None)
                if pname:
                    sig.append({"name": pname, "type": ptr_ty, "size_bytes": 8})
        launcher = KernelLauncher(hsaco=hsaco, kernel_name=kernel.name, signature=sig)
        block_size = (
            spec.tile.warp_m * spec.tile.warp_n * spec.tile.warp_k * spec.wave_size
        )
        return BuiltRegion(
            launcher=launcher,
            spec=spec,
            block_size=block_size,
            extra={"bias": bias_name, "residuals": residual_names},
        )

    def launch_args(
        self,
        graph: FusionGraph,
        region: FusionRegion,
        runtime_args: Dict[str, Any],
        built: BuiltRegion,
    ) -> Tuple[Dict[str, Any], Tuple[int, int, int]]:
        spec = built.spec
        head = graph.ops[region.op_names[0]]
        a_name, b_name = head.inputs
        out_name = region.outputs[0]
        A = runtime_args[a_name]
        B = runtime_args[b_name]
        C = runtime_args[out_name]
        M = int(runtime_args.get("M", A.shape[0]))
        N = int(runtime_args.get("N", C.shape[1]))
        K = int(runtime_args.get("K", A.shape[1]))
        args: Dict[str, Any] = {"A": A, "B": B, "C": C, "M": M, "N": N, "K": K}
        # bias + residual plumbing reads names from BuiltRegion.extra.
        bias_name = built.extra.get("bias")
        residual_names = built.extra.get("residuals") or ()
        # Pass bias if declared by the epilogue.
        epilogue = getattr(spec, "_fused_epilogue", None)
        if epilogue is not None:
            for op in epilogue.ops:
                pname = getattr(op, "param_name", None)
                if not pname:
                    continue
                if pname == "bias" and bias_name is not None:
                    args["bias"] = runtime_args[bias_name]
                elif pname.startswith("residual_") and not pname.startswith(
                    "residual_mul_"
                ):
                    idx = int(pname.split("_")[-1])
                    args[pname] = runtime_args[residual_names[idx]]
                elif pname.startswith("residual_mul_"):
                    idx = int(pname.split("_")[-1])
                    args[pname] = runtime_args[residual_names[idx]]
        grid = (
            (N + spec.tile.tile_n - 1) // spec.tile.tile_n,
            (M + spec.tile.tile_m - 1) // spec.tile.tile_m,
            1,
        )
        return args, grid


class ElementwiseLowerer:
    """Lower an ``elementwise`` region to ``build_elementwise``.

    Currently maps a single :class:`FusionOp` of kind
    ``{add, sub, mul, relu, gelu, silu, neg, abs}`` to one
    elementwise kernel. Multi-op chains can be expressed by a future
    extension that synthesises a small staged pipeline; for now the
    scheduler emits one region per op, so each call here builds one
    pointwise kernel.
    """

    name = "elementwise"
    supported_kinds = {"elementwise"}

    _UNARY = {"relu", "gelu", "silu", "neg", "abs"}
    _BINARY = {"add", "sub", "mul"}

    def can_lower(self, graph: FusionGraph, region: FusionRegion) -> LegalResult:
        if region.kind not in self.supported_kinds:
            return LegalResult.failure(f"{self.name}: unsupported kind {region.kind!r}")
        if len(region.op_names) != 1:
            return LegalResult.failure(
                f"{self.name}: only single-op regions supported (got {len(region.op_names)} ops)"
            )
        op = graph.ops[region.op_names[0]]
        if op.kind not in (self._UNARY | self._BINARY):
            return LegalResult.failure(f"{self.name}: unsupported op {op.kind!r}")
        return LegalResult.success()

    def candidates(
        self, graph: FusionGraph, region: FusionRegion
    ) -> List[AutotuneConfig]:
        from ..instances.elementwise import ElementwiseSpec, is_valid_spec

        op = graph.ops[region.op_names[0]]
        dtype = _dtype_str_for_tensor(graph, op.inputs[0])
        ew_dtype = "f16" if dtype in ("fp16", "f16", "float16") else "bf16"
        # The elementwise instance maps gelu via ``gelu_tanh``.
        op_kind = "gelu_tanh" if op.kind == "gelu" else op.kind
        cfgs: List[AutotuneConfig] = []
        for block_size in (256, 512):
            for vec in (8, 4, 2):
                spec = ElementwiseSpec(
                    op=op_kind, dtype=ew_dtype, block_size=block_size, vec=vec
                )
                ok, _ = is_valid_spec(spec)
                if not ok:
                    continue
                cfgs.append(
                    AutotuneConfig(
                        spec=spec,
                        name=f"ew_{op_kind}_{ew_dtype}_b{block_size}_v{vec}",
                    )
                )
        return cfgs

    def build(self, config: AutotuneConfig) -> BuiltRegion:
        from ..core.lower_llvm import lower_kernel_to_llvm
        from ..instances.elementwise import (
            build_elementwise,
            elementwise_signature,
        )
        from ..runtime.comgr import build_hsaco_from_llvm_ir
        from ..runtime.launcher import KernelLauncher

        spec = config.spec
        kernel = build_elementwise(spec)
        ir = lower_kernel_to_llvm(kernel)
        hsaco, _ = build_hsaco_from_llvm_ir(ir)
        sig = elementwise_signature(spec)
        launcher = KernelLauncher(hsaco=hsaco, kernel_name=kernel.name, signature=sig)
        return BuiltRegion(launcher=launcher, spec=spec, block_size=spec.block_size)

    def launch_args(
        self,
        graph: FusionGraph,
        region: FusionRegion,
        runtime_args: Dict[str, Any],
        built: BuiltRegion,
    ) -> Tuple[Dict[str, Any], Tuple[int, int, int]]:
        spec = built.spec
        op = graph.ops[region.op_names[0]]
        out_name = region.outputs[0]
        args: Dict[str, Any] = {}
        if op.kind in self._BINARY:
            args["A"] = runtime_args[op.inputs[0]]
            args["B"] = runtime_args[op.inputs[1]]
            args["C"] = runtime_args[out_name]
            numel = (
                int(args["C"].numel())
                if hasattr(args["C"], "numel")
                else int(runtime_args["N"])
            )
        else:
            args["A"] = runtime_args[op.inputs[0]]
            args["C"] = runtime_args[out_name]
            numel = (
                int(args["C"].numel())
                if hasattr(args["C"], "numel")
                else int(runtime_args["N"])
            )
        args["N"] = numel
        chunk = spec.elems_per_block()
        grid_x = (numel + chunk - 1) // chunk
        return args, (grid_x, 1, 1)


class ReductionLowerer:
    """Lower a ``rowwise_reduction`` region to ``build_reduce2d``.

    The region's single op must be one of ``{sum, mean, max}`` over
    the last dim of a 2D input. The lowerer chooses ``Reduce2DSpec``
    parameters that match the input shape and emits one CTA per row.
    """

    name = "rowwise_reduction"
    supported_kinds = {"rowwise_reduction"}

    def can_lower(self, graph: FusionGraph, region: FusionRegion) -> LegalResult:
        if region.kind not in self.supported_kinds:
            return LegalResult.failure(f"{self.name}: unsupported kind {region.kind!r}")
        if len(region.op_names) != 1:
            return LegalResult.failure(f"{self.name}: only single-op regions supported")
        op = graph.ops[region.op_names[0]]
        if op.kind not in {"sum", "mean", "max"}:
            return LegalResult.failure(
                f"{self.name}: unsupported reduce op {op.kind!r}"
            )
        in_t = graph.tensors[op.inputs[0]]
        if len(in_t.shape) != 2:
            return LegalResult.failure(f"{self.name}: input must be 2D")
        return LegalResult.success()

    def candidates(
        self, graph: FusionGraph, region: FusionRegion
    ) -> List[AutotuneConfig]:
        from ..instances.reduce import Reduce2DSpec, is_valid_spec

        op = graph.ops[region.op_names[0]]
        in_t = graph.tensors[op.inputs[0]]
        dtype = _dtype_str_for_tensor(graph, op.inputs[0])
        red_dtype = "f16" if dtype in ("fp16", "f16", "float16") else "bf16"
        n_per_block = int(in_t.shape[-1] or 0)
        if n_per_block <= 0:
            return []
        cfgs: List[AutotuneConfig] = []
        # Only emit configs that satisfy ``Reduce2DSpec``'s divisibility
        # constraint (n_per_block % (block_size * vec) == 0). For row
        # extents that don't divide cleanly into any of the canonical
        # (block_size, vec) combos, we still try the smallest combo so
        # the caller gets at least one config to inspect.
        for block_size in (256, 512, 128):
            for vec in (8, 4, 2):
                spec = Reduce2DSpec(
                    n_per_block=n_per_block,
                    op=op.kind,
                    block_size=block_size,
                    vec=vec,
                    dtype=red_dtype,
                )
                ok, _ = is_valid_spec(spec)
                if not ok:
                    continue
                cfgs.append(
                    AutotuneConfig(
                        spec=spec,
                        name=f"red_{op.kind}_{red_dtype}_b{block_size}_v{vec}",
                    )
                )
        return cfgs

    def build(self, config: AutotuneConfig) -> BuiltRegion:
        from ..core.lower_llvm import lower_kernel_to_llvm
        from ..instances.reduce import build_reduce2d, reduce2d_signature
        from ..runtime.comgr import build_hsaco_from_llvm_ir
        from ..runtime.launcher import KernelLauncher

        spec = config.spec
        kernel = build_reduce2d(spec)
        ir = lower_kernel_to_llvm(kernel)
        hsaco, _ = build_hsaco_from_llvm_ir(ir)
        sig = reduce2d_signature(spec)
        launcher = KernelLauncher(hsaco=hsaco, kernel_name=kernel.name, signature=sig)
        return BuiltRegion(launcher=launcher, spec=spec, block_size=spec.block_size)

    def launch_args(
        self,
        graph: FusionGraph,
        region: FusionRegion,
        runtime_args: Dict[str, Any],
        built: BuiltRegion,
    ) -> Tuple[Dict[str, Any], Tuple[int, int, int]]:
        op = graph.ops[region.op_names[0]]
        out_name = region.outputs[0]
        X = runtime_args[op.inputs[0]]
        Y = runtime_args[out_name]
        M = int(X.shape[0]) if hasattr(X, "shape") else int(runtime_args.get("M", 0))
        N = (
            int(X.shape[1])
            if hasattr(X, "shape") and len(X.shape) > 1
            else int(runtime_args.get("N", 0))
        )
        return {"X": X, "Y": Y, "M": M, "N": N}, (M, 1, 1)


def default_lowering_registry() -> LoweringRegistry:
    """A small default registry exposing the in-tree concrete lowerers.

    Use this when bootstrapping an end-to-end test; production code
    should typically build a registry tailored to the workload (e.g.
    by registering only the families it knows about).
    """

    reg = LoweringRegistry()
    reg.register(GemmEpilogueLowerer())
    # The greedy scheduler emits ``gemm`` for matmul-only regions and
    # ``gemm_epilogue`` for matmul + pointwise. Both share the lowerer
    # but the registry keys distinct names for explainability.
    reg._lowerers["gemm"] = reg._lowerers["gemm_epilogue"]
    reg.register(ElementwiseLowerer())
    reg.register(ReductionLowerer())
    return reg
