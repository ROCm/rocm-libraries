# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Lower a rocKE ``KernelDef`` as an LLVM *device function* (Python engine only).

The hybrid path of the Qwen3.8 roadmap links rocKE-authored stage bodies
into the FlyDSL mono layer kernel: FlyDSL declares an external symbol
(``flydsl.expr.extern.ffi``), calls it from the kernel body, and links the
module this file emits (``flydsl.compiler.extern_link.link_extern``, which
hands the path to MLIR's ROCDL attach-target ``l=`` option; MLIR loads it
with ``llvm::getLazyIRFileModule``, so a ``.ll`` text file and a ``.bc``
bitcode file both work, links it with ``LinkOnlyNeeded``, internalizes the
linked symbols and only then runs the O3 pipeline, so the call is normally
inlined).

What differs from :func:`rocke.core.lower_llvm.lower_kernel_to_llvm`:

* ``define void @<symbol>(...)`` instead of ``define amdgpu_kernel``, with
  external linkage and the C calling convention.
* A stable C ABI: every pointer parameter becomes an ``i64`` address (the
  body sees ``inttoptr`` of it in the parameter's own address space); integer
  and float parameters keep their type. FlyDSL's ``ffi`` only takes
  ``int32`` / ``int64`` / ``float32`` / ``float64`` arguments, and the mono
  kernel holds every buffer as an int64 address anyway. Pointer parameter
  attributes (``noalias``, ``readonly``, ``align``...) are dropped: a device
  function's argument attributes would be promises about the caller.
* No kernel attributes: ``uniform-work-group-size``, ``amdgpu-waves-per-eu``,
  ``amdgpu-agpr-alloc``, ``amdgpu-sched-strategy`` and cluster dims belong to
  the caller kernel. ``amdgpu-flat-work-group-size`` is kept (knob) as the
  workgroup-size contract of the body; the inline knob adds ``alwaysinline``
  or ``noinline``.
* LDS (``DeviceFunctionKnobs.lds``):

  - ``"global"``: the body's smem pool stays a module-level
    ``internal addrspace(3)`` global; the AMDGPU LDS lowering adds it to the
    caller kernel's LDS (allocation = caller's + :attr:`DeviceFunction.lds_bytes`).
  - ``"arg"``: the pool global is removed and the function takes one more
    ``i32`` parameter, ``lds_base``, the byte address in LDS of a region of at
    least ``lds_bytes`` bytes, 16-byte aligned, owned by the caller (e.g. a
    view into the mono kernel's arena: ``ptrtoint`` of an addrspace(3)
    pointer). Every pool reference becomes ``inttoptr i32 %lds_base``.

* Barriers, wave intrinsics, inline asm, ``workitem.id`` / ``workgroup.id``
  are unchanged: inside a function they read the caller's values.
* ``ret`` (``b.ret()`` / ``cf.return``) returns to the caller: a stage body
  that returns early ends the stage, not the kernel.

The body is the Python lowerer's output byte for byte (``_Lowerer``
lowering the same ops); only the header, the parameter prologue, the
attribute group and, with ``lds="arg"``, the pool references change. The
kernel path is not touched, so every existing kernel's IR is unchanged.

Compliance: a lowering mode of the Python engine with no C++ counterpart,
like the family's other Python-only pieces (README, "cpp-parity
exemption"). The ``KernelDef`` itself may still use only mirrored ops.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field

from rocke.core.ir import F32, F64, I32, I64, KernelDef, PtrType
from rocke.core.lower_llvm import _Lowerer, _llvm_type, _param_llvm_type

__all__ = [
    "DeviceFunction",
    "DeviceFunctionKnobs",
    "LDS_MODES",
    "INLINE_MODES",
    "find_llvm_as",
    "lower_device_function",
    "write_device_function",
]

LDS_MODES = ("global", "arg")
INLINE_MODES = ("default", "always", "never")
LDS_BASE_PARAM = "lds_base"
_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
# FlyDSL ``ffi`` type names of the ABI's LLVM types.
_FFI_TYPES = {"i32": "int32", "i64": "int64", "float": "float32", "double": "float64"}


@dataclass(frozen=True)
class DeviceFunctionKnobs:
    """How :func:`lower_device_function` emits the function.

    lds: ``"global"`` | ``"arg"`` (module docstring).
    inline: ``"default"`` (the caller's inliner decides; MLIR internalizes the
      linked function, so O3 normally inlines a single call site) |
      ``"always"`` (``alwaysinline``) | ``"never"`` (``noinline``: a real call,
      for measuring call overhead or keeping register pressure separate).
    flat_wg_attr: keep ``amdgpu-flat-work-group-size`` on the function.
    convergent: mark the function ``convergent`` (default). A stage body holds
      barriers and cross-lane ops; without the attribute the caller's call is
      not convergent before inlining, and LLVM may thread or duplicate it
      under a divergent branch (e.g. between the mono's stamped
      ``if tid == 0`` blocks), splitting a wave across two copies of the
      body's ``s_barrier``s: the layer hangs. ``False`` only for A/B.
    """

    lds: str = "global"
    inline: str = "default"
    flat_wg_attr: bool = True
    convergent: bool = True

    def __post_init__(self) -> None:
        if self.lds not in LDS_MODES:
            raise ValueError(f"lds must be one of {LDS_MODES}, got {self.lds!r}")
        if self.inline not in INLINE_MODES:
            raise ValueError(
                f"inline must be one of {INLINE_MODES}, got {self.inline!r}"
            )

    def suffix(self) -> str:
        """Name suffix of the knobs that deviate from the default."""
        d = DeviceFunctionKnobs()
        parts = []
        if self.lds != d.lds:
            parts.append(f"lds{self.lds}")
        if self.inline != d.inline:
            parts.append(f"inl{self.inline}")
        if self.flat_wg_attr != d.flat_wg_attr:
            parts.append("nowg")
        if self.convergent != d.convergent:
            parts.append("nocv")
        return "_".join(parts)


@dataclass(frozen=True)
class DeviceFunction:
    """An emitted device function.

    ``abi``: ``(name, llvm type)`` per parameter in call order (``lds_base``
    last with ``lds="arg"``). ``lds_bytes``: the body's smem pool size (added
    to the caller's LDS with ``lds="global"``; the region ``lds_base`` must
    provide with ``lds="arg"``).
    """

    symbol: str
    ir: str
    abi: tuple[tuple[str, str], ...]
    lds_bytes: int
    arch: str
    flavor: str
    knobs: DeviceFunctionKnobs = field(default_factory=DeviceFunctionKnobs)
    max_workgroup_size: int = 256

    def ffi_types(self) -> list[str]:
        """The FlyDSL ``ffi`` argument type names, in call order."""
        return [_FFI_TYPES[t] for _, t in self.abi]

    def sha256(self) -> str:
        return hashlib.sha256(self.ir.encode()).hexdigest()

    def meta(self) -> dict:
        """JSON-able description (the ``.json`` sidecar)."""
        return {
            "symbol": self.symbol,
            "abi": [list(a) for a in self.abi],
            "ffi_types": self.ffi_types(),
            "lds_bytes": self.lds_bytes,
            "arch": self.arch,
            "flavor": self.flavor,
            "knobs": asdict(self.knobs),
            "max_workgroup_size": self.max_workgroup_size,
            "ir_sha256": self.sha256(),
        }


def _abi_type(p) -> str:
    """C-ABI LLVM type of a kernel parameter."""
    if isinstance(p.type, PtrType):
        return "i64"
    if p.type in (I32, I64, F32, F64):
        return _llvm_type(p.type)
    raise TypeError(
        f"device-function parameter {p.name!r}: type {p.type.name} has no C-ABI "
        "form here (pointers, i32, i64, f32, f64 only)"
    )


def _pool_ref(name: str) -> re.Pattern:
    # ``@smem_pool.<kernel>`` not followed by another identifier character
    return re.compile(re.escape(name) + r"(?![-A-Za-z0-9$._])")


class _DeviceFunctionLowerer(_Lowerer):
    def __init__(self, kernel, *, symbol, knobs, llvm_flavor, arch):
        super().__init__(kernel, llvm_flavor=llvm_flavor, arch=arch)
        self._symbol = symbol
        self._knobs = knobs
        self.abi: list[tuple[str, str]] = []
        # Own copy: the per-flavor table may be shared with kernel lowering.
        self._decls = dict(self._decls)
        self._decls["sqrt.f64"] = "declare double @llvm.sqrt.f64(double)"

    def _op_math_sqrt(self, op) -> None:
        """``math.sqrt`` also on f64 (``llvm.sqrt.f64``; the kernel lowering
        takes f32 only). The FlyDSL mono's ``fmath.sqrt`` of an f64 lowers to
        the same intrinsic (MLIR MathToLLVM), so a stage function that needs
        a bit-identical f64 square root emits it here."""
        (v,) = op.operands
        if v.type.name != "f64":
            return super()._op_math_sqrt(op)
        self._need("sqrt.f64")
        self._current().emit(
            f"  {op.result.name} = call double @llvm.sqrt.f64(double {self._operand(v)})"
        )

    def finalize(self) -> str:
        knobs = self._knobs
        prologue = []
        for p in self.kernel.params:
            t = _abi_type(p)
            name = p.name.lstrip("%")
            self.abi.append((name, t))
            if isinstance(p.type, PtrType):
                prologue.append(
                    f"  %{name} = inttoptr i64 %{name}.abi to {_param_llvm_type(p)}"
                )
        pool = self._smem_pool_name if self._smem_globals else None
        lds_arg = knobs.lds == "arg"
        if lds_arg:
            self.abi.append((LDS_BASE_PARAM, "i32"))
            if pool is not None:
                prologue.append(
                    f"  %lds.pool = inttoptr i32 %{LDS_BASE_PARAM} to ptr addrspace(3)"
                )
                ref = _pool_ref(pool)
                for blk in self._blocks:
                    blk.lines = [ref.sub("%lds.pool", ln) for ln in blk.lines]
        self._blocks[0].lines[:0] = prologue

        text = super().finalize()
        lines = text.split("\n")
        head = f"define amdgpu_kernel void @{self.kernel.name}("
        hits = [i for i, ln in enumerate(lines) if ln.startswith(head)]
        if len(hits) != 1:
            raise RuntimeError("device function: kernel header not found once")
        i = hits[0]
        tail = lines[i][lines[i].rindex(")") + 1 :]  # " #0{ !dbg !N} {"
        params = []
        for name, t in self.abi:
            is_ptr = any(
                p.name.lstrip("%") == name and isinstance(p.type, PtrType)
                for p in self.kernel.params
            )
            params.append(f"{t} %{name}.abi" if is_ptr else f"{t} %{name}")
        lines[i] = f"define void @{self._symbol}({', '.join(params)}){tail}"

        attrs = [j for j, ln in enumerate(lines) if ln.startswith("attributes #0 = {")]
        if len(attrs) != 1:
            raise RuntimeError("device function: attribute group #0 not found once")
        parts = []
        if knobs.flat_wg_attr:
            max_wg = self.kernel.max_workgroup_size
            min_wg = min(self._backend.arch.wave_size, max_wg)
            parts.append(f'"amdgpu-flat-work-group-size"="{min_wg},{max_wg}"')
        if knobs.inline == "always":
            parts.append("alwaysinline")
        elif knobs.inline == "never":
            parts.append("noinline")
        if knobs.convergent:
            parts.append("convergent")
        parts += ["norecurse", "nounwind"]
        lines[attrs[0]] = "attributes #0 = { " + " ".join(parts) + " }"

        if lds_arg and pool is not None:
            decl = [j for j, ln in enumerate(lines) if ln.startswith(f"{pool} = ")]
            if len(decl) != 1:
                raise RuntimeError("device function: smem pool global not found once")
            del lines[decl[0]]
            leftover = [ln for ln in lines if _pool_ref(pool).search(ln)]
            if leftover:
                raise RuntimeError(
                    f"device function: smem pool still referenced: {leftover[:2]}"
                )
        return "\n".join(lines)


def lower_device_function(
    kernel: KernelDef,
    *,
    arch: str,
    symbol: str | None = None,
    llvm_flavor: str | None = None,
    knobs: DeviceFunctionKnobs = DeviceFunctionKnobs(),
) -> DeviceFunction:
    """Lower ``kernel`` as device function ``symbol`` (default: the kernel's
    name). Always the Python engine (no C++ counterpart)."""
    symbol = symbol or kernel.name
    if not _SYMBOL_RE.match(symbol):
        raise ValueError(f"bad device-function symbol {symbol!r}")
    if kernel.attrs.get("cluster_dims") is not None:
        raise ValueError("cluster_dims is a kernel attribute; not for a device function")
    low = _DeviceFunctionLowerer(
        kernel, symbol=symbol, knobs=knobs, llvm_flavor=llvm_flavor, arch=arch
    )
    low._collect_smem(kernel.body)
    low._compute_smem_layout()
    low.lower_region(kernel.body)
    text = low.finalize()
    return DeviceFunction(
        symbol=symbol,
        ir=text,
        abi=tuple(low.abi),
        lds_bytes=int(low._smem_pool_size if low._smem_globals else 0),
        arch=arch,
        flavor=low._flavor,
        knobs=knobs,
        max_workgroup_size=kernel.max_workgroup_size,
    )


def find_llvm_as() -> str | None:
    """An ``llvm-as``: ``$ROCKE_LLVM_AS``, ``$ROCM_PATH/{llvm,lib/llvm}/bin``,
    ``PATH``, ``/opt/rocm/llvm/bin``; ``None`` when there is none. Its LLVM
    must not be newer than the linking compiler's (bitcode is read by equal
    or newer LLVMs only); the ``.ll`` text needs no assembler at all."""
    env = os.environ.get("ROCKE_LLVM_AS")
    if env:
        return env
    roots = [os.environ.get("ROCM_PATH"), os.environ.get("HIP_PATH"), "/opt/rocm"]
    for root in filter(None, roots):
        for sub in ("llvm/bin", "lib/llvm/bin", "bin"):
            cand = os.path.join(root, sub, "llvm-as")
            if os.path.isfile(cand):
                return cand
    return shutil.which("llvm-as")


def write_device_function(
    fn: DeviceFunction,
    out_dir: str,
    *,
    stem: str | None = None,
    bitcode: bool | str = "auto",
    llvm_as: str | None = None,
) -> dict:
    """Write ``<stem>.ll``, ``<stem>.json`` and (``bitcode`` True, or
    ``"auto"`` with an assembler found) ``<stem>.bc``; returns their paths
    (``"bc"`` is ``None`` when not written). ``bitcode=True`` without an
    assembler raises."""
    stem = stem or fn.symbol
    os.makedirs(out_dir, exist_ok=True)
    ll = os.path.join(out_dir, stem + ".ll")
    with open(ll, "w", newline="\n") as f:
        f.write(fn.ir if fn.ir.endswith("\n") else fn.ir + "\n")
    meta = fn.meta()
    bc = None
    if bitcode:
        tool = llvm_as or find_llvm_as()
        if tool is None:
            if bitcode is True:
                raise FileNotFoundError("no llvm-as (set ROCKE_LLVM_AS)")
        else:
            bc = os.path.join(out_dir, stem + ".bc")
            subprocess.run([tool, ll, "-o", bc], check=True)
            meta["llvm_as"] = tool
    meta["files"] = {"ll": os.path.basename(ll), "bc": bc and os.path.basename(bc)}
    js = os.path.join(out_dir, stem + ".json")
    with open(js, "w", newline="\n") as f:
        json.dump(meta, f, indent=1)
        f.write("\n")
    return {"ll": ll, "bc": bc, "json": js}
