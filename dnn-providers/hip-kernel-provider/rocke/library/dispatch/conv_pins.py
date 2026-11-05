# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pinned conv launches as a dispatch heuristic.

A *pins* file (``benchmarks/common/derive_conv_pins.py`` writes one from
benchmark runs; ``benchmark_implicit_gemm_conv --pins`` and
``conv_step_replay.py --pins`` read it) names, per benchmarked case, the
selected kernel and its launch parameters::

    {"cases": {"fwd_s3": {"cmd": "MIOpenDriver convbfp16 ... -F 1 ...",
                          "kernel": "<identity label> @split_k=2/full1824",
                          "calls": 2, "ms": <median>}, ...}}

This module lets dispatch use it the way a tuned table is used:

* :func:`load_conv_pins` reads the file and keys every case by its problem
  (direction, dtype and every dimension / stride / pad / dilation), parsed from
  the MIOpenDriver command -- no GPU, no benchmark import.
* :meth:`ConvPins.lookup` answers a :class:`ConvGroupedRequest` (direction
  ``fwd`` / ``dgrad`` / ``wgrad``) with its :class:`ConvPin`: the instance name
  (the identity label, :func:`kernels.conv.kernel_label.strip_launch_suffix`)
  and the launch parameters the label carries (``split_k``, a tail split's
  ``full`` tile count).
* :meth:`ConvPins.ranker` is a :data:`rocke.dispatch.core.Ranker` for
  :func:`dispatch.grouped_convolution.dispatch_conv_grouped`: the pinned
  instance first when it is among the admitted candidates, the registry's own
  order otherwise (a pin never makes dispatch refuse a request).

Nothing reads a pins file unless a caller passes one; registry order stays the
default policy.

Pin tables (several modes in one file)
--------------------------------------
A *pin table* holds one pins set per selection mode -- e.g. ``deterministic``
(only bitwise run-to-run reproducible launches) and ``fastest`` -- for one
arch::

    {"arch": "gfx942", "llvm_flavor": "llvm22",
     "modes": {"deterministic": {"cases": {...}}, "fastest": {"cases": {...}}}}

:func:`load_conv_pins` reads one mode of it (``mode=``). The shipped tables are
under ``conv_pin_tables/`` (:func:`shipped_conv_pin_tables`).

Consuming a table the way dispatch consumes the opt-in Winograd forward
(``dispatch.grouped_convolution``: a candidate with ``opt_in=True``, never
picked under ``algorithm="auto"``, reached by pinning its ``algorithm``):

* :func:`pinned_conv_candidate` turns one mode of a table into an opt-in
  ``KernelCandidate`` per direction (``algorithm="pinned"``,
  ``spec_id="conv_pinned_<mode>"``). It admits a request only when the table
  pins its exact problem and the pinned binary's own admission rule
  (:func:`kernels.conv.kernel_identity.identity_supports_problem`) accepts it;
  its spec (:class:`ConvPinnedSpec`) is the instance's AOT identity plus the
  launch parameters it was benchmarked with; ``build`` is the IR of the cached
  binary (the job ``--compile-all`` builds the identity from).
* :func:`dispatch_conv_pinned` answers a ``ConvGroupedRequest`` of any
  direction (fwd, dgrad, wgrad) from a table: a ``DispatchResult`` whose
  ``spec.launch_params()`` carries ``split_k`` / ``tail_full``.
* :func:`register_conv_pin_table` adds the fwd / dgrad candidates to the
  grouped-convolution registries, so ``dispatch_conv_grouped`` with
  ``algorithm="pinned"`` returns the pinned instance (wgrad has no
  grouped-convolution registry; :func:`dispatch_conv_pinned` serves it).
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

_SPLIT_RE = re.compile(r"@split_k=(\d+)(?:/full(\d+))?")
_DIRECTIONS = {1: "fwd", 2: "dgrad", 4: "wgrad"}
_DTYPES = {"convbfp16": "bf16", "convfp16": "fp16", "conv": "fp32", "convfp32": "fp32"}

# MIOpenDriver flag -> problem field (long and short forms).
_FLAGS = {
    "-n": "N",
    "--batchsize": "N",
    "-c": "C",
    "--in_channels": "C",
    "-k": "K",
    "--out_channels": "K",
    "--in_d": "Di",
    "-H": "Hi",
    "--in_h": "Hi",
    "-W": "Wi",
    "--in_w": "Wi",
    "--fil_d": "Z",
    "-y": "Y",
    "--fil_h": "Y",
    "-x": "X",
    "--fil_w": "X",
    "--pad_d": "pad_d",
    "-p": "pad_h",
    "--pad_h": "pad_h",
    "-q": "pad_w",
    "--pad_w": "pad_w",
    "--conv_stride_d": "stride_d",
    "-u": "stride_h",
    "--conv_stride_h": "stride_h",
    "-v": "stride_w",
    "--conv_stride_w": "stride_w",
    "--dilation_d": "dilation_d",
    "-l": "dilation_h",
    "--dilation_h": "dilation_h",
    "-j": "dilation_w",
    "--dilation_w": "dilation_w",
    "-g": "G",
    "--group_count": "G",
    "-F": "forw",
    "--forw": "forw",
}
_DEFAULTS = {
    "N": 1,
    "G": 1,
    "stride_h": 1,
    "stride_w": 1,
    "pad_h": 0,
    "pad_w": 0,
    "dilation_h": 1,
    "dilation_w": 1,
}
_KEY_FIELDS = (
    "N",
    "C",
    "K",
    "Hi",
    "Wi",
    "Y",
    "X",
    "G",
    "stride_h",
    "stride_w",
    "pad_h",
    "pad_w",
    "dilation_h",
    "dilation_w",
    "Di",
    "Z",
    "stride_d",
    "pad_d",
    "dilation_d",
)


def _norm_dtype(dtype: str) -> str:
    d = str(dtype).lower()
    return {
        "bfloat16": "bf16",
        "float16": "fp16",
        "half": "fp16",
        "float32": "fp32",
        "float": "fp32",
    }.get(d, d)


def parse_miopen_cmd(cmd: str) -> tuple[str, str, tuple]:
    """``(direction, dtype, dims key)`` of a MIOpenDriver conv command line."""
    toks = shlex.split(cmd)
    if toks and toks[0].endswith("MIOpenDriver"):
        toks = toks[1:]
    if not toks or toks[0] not in _DTYPES:
        raise ValueError(f"not a MIOpenDriver conv command: {cmd!r}")
    dtype = _DTYPES[toks[0]]
    vals: dict[str, int] = dict(_DEFAULTS)
    i = 1
    while i < len(toks):
        name = _FLAGS.get(toks[i])
        if name is not None and i + 1 < len(toks):
            vals[name] = int(toks[i + 1])
            i += 2
        else:
            i += (
                2
                if toks[i].startswith("-")
                and i + 1 < len(toks)
                and not toks[i + 1].startswith("-")
                else 1
            )
    forw = vals.pop("forw", 1)
    if forw not in _DIRECTIONS:
        raise ValueError(
            f"{cmd!r}: -F {forw} is not one direction (1 fwd, 2 dgrad, 4 wgrad)"
        )
    if "Di" in vals:
        for f, d in (("Z", 1), ("stride_d", 1), ("pad_d", 0), ("dilation_d", 1)):
            vals.setdefault(f, d)
    return _DIRECTIONS[forw], dtype, tuple(vals.get(f) for f in _KEY_FIELDS)


def request_key(req) -> tuple[str, str, tuple]:
    """The pins key of a :class:`ConvGroupedRequest`-like object."""
    vals = {f: getattr(req, f, None) for f in _KEY_FIELDS}
    for f, d in _DEFAULTS.items():
        if vals[f] is None:
            vals[f] = d
    vals = {f: (int(v) if v is not None else None) for f, v in vals.items()}
    return (
        str(req.direction).lower(),
        _norm_dtype(req.dtype),
        tuple(vals[f] for f in _KEY_FIELDS),
    )


@dataclass(frozen=True)
class ConvPin:
    """One pinned pick: the instance and the launch it was selected with."""

    case: str
    instance: str
    label: str
    split_k: int | None = None
    tail_full: int | None = None
    ms: float | None = None
    calls: int = 1

    @property
    def launch(self) -> dict:
        d = {}
        if self.split_k is not None:
            d["split_k"] = self.split_k
        if self.tail_full is not None:
            d["tail_full"] = self.tail_full
        return d


@dataclass
class ConvPins:
    """A pins file keyed by problem."""

    source: str = ""
    pins: dict = field(default_factory=dict)  # key -> ConvPin
    arch: str = ""
    mode: str = ""
    llvm_flavor: str = ""

    def __len__(self) -> int:
        return len(self.pins)

    def lookup(self, req) -> ConvPin | None:
        if self.arch and getattr(req, "arch", self.arch) != self.arch:
            return None
        return self.pins.get(request_key(req))

    def ranker(self):
        """A dispatch ranker: the pinned instance first, else registry order."""

        def rank(request, candidates: Sequence):
            pin = self.lookup(request)
            if pin is None:
                return tuple(candidates)
            first = [c for c in candidates if c.name == pin.instance]
            if not first:
                return tuple(candidates)
            return tuple(first) + tuple(c for c in candidates if c.name != pin.instance)

        return rank


def pins_from_mapping(data: Mapping, source: str = "") -> ConvPins:
    """:class:`ConvPins` from a decoded pins document."""
    out = ConvPins(source=source)
    for case, row in (data.get("cases") or {}).items():
        label = str(row["kernel"]).strip()
        instance = label.split()[0] if label else ""
        m = _SPLIT_RE.search(label)
        key = parse_miopen_cmd(row["cmd"])
        if key in out.pins and out.pins[key].instance != instance:
            raise ValueError(
                f"{source}: cases {out.pins[key].case} and {case} pin one problem to two kernels"
            )
        out.pins[key] = ConvPin(
            case=case,
            instance=instance,
            label=label,
            split_k=int(m.group(1)) if m else None,
            tail_full=int(m.group(2)) if m and m.group(2) is not None else None,
            ms=float(row["ms"]) if row.get("ms") is not None else None,
            calls=int(row.get("calls", 1)),
        )
    return out


def table_modes(data: Mapping) -> tuple[str, ...]:
    """The modes of a pin table (``()`` for a plain pins document)."""
    return tuple((data.get("modes") or {}).keys())


def load_conv_pins(path, mode: str | None = None) -> ConvPins:
    """Read a pins JSON file, or one ``mode`` of a pin table.

    A table with one mode reads that mode when ``mode`` is ``None``; a table
    with several needs ``mode``. ``mode`` on a plain pins file is refused.
    """
    p = Path(path)
    data = json.loads(p.read_text())
    modes = table_modes(data)
    if not modes:
        if mode is not None:
            raise ValueError(f"{p}: a plain pins file has no mode {mode!r}")
        out = pins_from_mapping(data, source=str(p))
        out.arch = str(data.get("arch", "") or "")
        return out
    if mode is None:
        if len(modes) != 1:
            raise ValueError(f"{p}: pin table with modes {', '.join(modes)}: name one")
        mode = modes[0]
    if mode not in modes:
        raise ValueError(f"{p}: no mode {mode!r} (modes: {', '.join(modes)})")
    out = pins_from_mapping(data["modes"][mode], source=f"{p}#{mode}")
    out.arch = str(data.get("arch", "") or "")
    out.mode = mode
    out.llvm_flavor = str(data.get("llvm_flavor", "") or "")
    return out


# ---------------------------------------------------------------------------
# Pin tables as an opt-in dispatch candidate
# ---------------------------------------------------------------------------

#: Shipped pin tables (``<arch>_<name>.json``), beside this module.
PIN_TABLE_DIR = Path(__file__).resolve().with_name("conv_pin_tables")

_FAMILY = {"fwd": "conv_fwd", "dgrad": "conv_dgrad", "wgrad": "conv_wgrad"}
PINNED_ALGORITHM = "pinned"
PINNED_ABI_VERSION = "hipkg-conv-grouped/v1+pinned-launch"


def shipped_conv_pin_tables() -> dict[str, Path]:
    """The pin tables shipped beside this module, by file stem."""
    if not PIN_TABLE_DIR.is_dir():
        return {}
    return {f.stem: f for f in sorted(PIN_TABLE_DIR.glob("*.json"))}


@dataclass(frozen=True)
class ConvPinnedSpec:
    """A pinned instance selected for a request.

    ``identity`` is the AOT identity (the cache key of the binary), ``block``
    its workgroup size and ``launch`` the launch parameters the pin was
    benchmarked with (``split_k``; ``tail_full``: a tail split's full-tile count),
    which the launcher applies as the ``--run-from-cache`` path does
    (``benchmark_implicit_gemm_conv._cached_bind``).
    """

    name: str
    identity: object  # kernels.conv.kernel_identity.KernelIdentity
    block: tuple[int, int, int]
    launch: tuple[tuple[str, int], ...] = ()
    mode: str = ""
    case: str = ""

    @property
    def direction(self) -> str:
        return self.identity.direction

    @property
    def dtype(self) -> str:
        return self.identity.dtype_a

    @property
    def algorithm(self) -> str:
        return self.identity.algorithm

    def kernel_name(self) -> str:
        return self.name

    def launch_params(self) -> dict:
        return dict(self.launch)


def pinned_kernel_signature(identity, dtype: str | None = None) -> tuple[dict, ...]:
    """Kernarg signature of a pinned implicit-GEMM binary: the one it was built
    with (the ``--run-from-cache`` launcher's ``_signature_for_identity``)."""
    from kernels.common.conv_abi import conv_args_signature

    dt = dtype or identity.dtype_a
    if identity.direction == "wgrad":
        flags = (
            {"split_flags": True}
            if (getattr(identity, "ws_fuse", 0) or getattr(identity, "ws_tree", 0))
            else {}
        )
        return tuple(
            conv_args_signature(
                dt,
                direction="wgrad",
                is_3d=identity.is_3d,
                two_stage=identity.two_stage,
                streamk=(
                    identity.streamk_reduction if identity.streamk != "off" else None
                ),
                **flags,
            )
        )
    if identity.direction == "dgrad":
        return tuple(conv_args_signature(dt, direction="dgrad", is_3d=identity.is_3d))
    extra = {"split_det": True} if getattr(identity, "split_det", False) else {}
    return tuple(
        conv_args_signature(
            dt,
            is_3d=identity.is_3d,
            two_stage=identity.two_stage,
            split_k=identity.split_k > 1 and not identity.two_stage,
            split_tail=bool(getattr(identity, "split_tail", False)),
            split_flags=bool(
                getattr(identity, "split_flag_store", False)
                or getattr(identity, "split_zero_first", False)
            ),
            **extra,
        )
    )


def _nominal_grid(spec: ConvPinnedSpec, req) -> tuple[int, int, int]:
    """The data-parallel tile grid of the pinned tile on ``req`` (z: groups x
    the pinned split degree). Persistent / stream-K / box-list launches size
    their own grid from the device's CU count at launch time; this is the tile
    grid they cover."""
    from kernels.common.conv_args import ConvArgs

    from .grouped_convolution import _conv_problem

    ident = spec.identity
    p = _conv_problem(req)
    sk = int(spec.launch_params().get("split_k", 1) or 1)
    if ident.direction == "wgrad":
        gx, gy, _ = ConvArgs.from_problem(
            p,
            direction="wgrad",
            tile_m=ident.tile_m,
            tile_n=ident.tile_n,
            tile_k=ident.tile_k,
        ).grid()
        return gx, gy, max(1, p.groups) * sk
    gx, gy, gz = ConvArgs.from_problem(
        p, direction=ident.direction, tile_m=ident.tile_m, tile_n=ident.tile_n
    ).grid()
    return gx, gy, gz * sk


def _pinned_spec(pins: ConvPins, pin: ConvPin) -> ConvPinnedSpec:
    from kernels.conv.kernel_identity import KernelIdentity

    ident = KernelIdentity.from_label(pin.instance)
    return ConvPinnedSpec(
        name=pin.instance,
        identity=ident,
        block=(ident.warp_m * ident.warp_n * ident.wave_size, 1, 1),
        launch=tuple(sorted(pin.launch.items())),
        mode=pins.mode,
        case=pin.case,
    )


def pinned_conv_candidate(pins: ConvPins, direction: str):
    """One mode of a pin table as an opt-in dispatch candidate for ``direction``.

    Mirrors ``grouped_convolution``'s Winograd candidate: ``opt_in=True`` (no
    ``algorithm="auto"`` request ever selects it), reached by
    ``algorithm="pinned"`` (plus ``spec_id="conv_pinned_<mode>"`` to name the
    mode when several tables are registered).
    """
    from rocke.dispatch.core import Capability, KernelCandidate, selector_matches

    from kernels.conv.aot_jobs import job_for_identity, kernel_for_job
    from kernels.conv.kernel_identity import identity_supports_problem

    from .grouped_convolution import _conv_problem

    if direction not in _FAMILY:
        raise ValueError(f"unknown direction {direction!r}")
    mode = pins.mode or "pins"
    name = f"conv_pinned_{mode}"
    specs: dict = {}

    def spec_for(req) -> tuple[ConvPinnedSpec | None, str]:
        if str(req.direction).lower() != direction:
            return None, f"{name} serves {direction}, not {req.direction}"
        pin = pins.lookup(req)
        if pin is None:
            return None, f"{name}: the table pins no kernel for this problem"
        # Keyed by the whole label: one instance may be pinned with different
        # launches on different problems.
        spec = specs.get(pin.label)
        if spec is None:
            spec = specs[pin.label] = _pinned_spec(pins, pin)
        p = _conv_problem(req)
        if p is None:
            return None, "not a valid convolution shape"
        ok, why = identity_supports_problem(
            spec.identity, p, getattr(req, "llvm_flavor", None)
        )
        if not ok:
            return None, f"{name}: pinned {pin.instance} cannot run it: {why}"
        return spec, "ok"

    def support(req) -> tuple[bool, str]:
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        spec, why = spec_for(req)
        return spec is not None, why

    def select(req) -> ConvPinnedSpec:
        spec, why = spec_for(req)
        if spec is None:
            raise ValueError(f"{name} does not support request: {why}")
        return spec

    def build(spec: ConvPinnedSpec, arch: str):
        if arch != spec.identity.arch:
            raise ValueError(
                f"{spec.name} is a {spec.identity.arch} instance, not {arch}"
            )
        return kernel_for_job(
            job_for_identity(spec.identity), arch, spec.identity.dtype_a
        )[0]

    dtypes = tuple(sorted({key[1] for key in pins.pins})) or ("bf16", "fp16")
    candidate = KernelCandidate(
        name=name,
        family=_FAMILY[direction],
        algorithm=PINNED_ALGORITHM,
        spec_id=name,
        abi_version=PINNED_ABI_VERSION,
        # After every listed instance and the Winograd forward; opt-in anyway.
        priority=2_000_000,
        capability=Capability(
            arches=(pins.arch,) if pins.arch else (), dtypes=dtypes, layouts=("NHWC",)
        ),
        _supports=support,
        select_spec=select,
        signature=lambda spec: pinned_kernel_signature(spec.identity),
        grid=_nominal_grid,
        block=lambda spec: spec.block,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=build,
        opt_in=True,
    )
    return candidate


_PIN_REGISTRIES: dict = {}


def _pin_registry(pins: ConvPins, direction: str):
    from rocke.dispatch.core import CandidateRegistry

    key = (id(pins), direction)
    reg = _PIN_REGISTRIES.get(key)
    if reg is None or reg[0] is not pins:
        reg = (pins, CandidateRegistry(_FAMILY[direction]))
        reg[1].register(pinned_conv_candidate(pins, direction))
        _PIN_REGISTRIES[key] = reg
    return reg[1]


def dispatch_conv_pinned(req, pins: ConvPins):
    """The pinned instance of ``req`` (any direction) from one mode of a table.

    ``req`` is a ``ConvGroupedRequest`` (``direction`` fwd / dgrad / wgrad);
    ``algorithm="auto"`` is read as ``"pinned"``. Raises ``ValueError`` when
    the table pins nothing that can run it -- the caller then falls back to
    its default selection.
    """
    import dataclasses

    from rocke.dispatch.core import make_dispatch_result, make_kernel_id

    direction = str(req.direction).lower()
    if direction not in _FAMILY:
        raise ValueError(f"unknown direction {req.direction!r}")
    if str(getattr(req, "algorithm", "auto")).strip().lower() == "auto":
        req = dataclasses.replace(req, algorithm=PINNED_ALGORITHM)
    candidate = _pin_registry(pins, direction).select(req)
    spec = candidate.select_spec(req)
    return make_dispatch_result(
        req,
        candidate,
        spec,
        kernel_id=make_kernel_id(req, candidate, spec, op=_FAMILY[direction]),
        headline=f"pinned {spec.name} on {req.arch} ({pins.mode or 'pins'}: {spec.case})",
    )


def register_conv_pin_table(pins: ConvPins) -> tuple:
    """Add one mode of a table to the grouped-convolution registries (fwd and
    dgrad; opt-in, so ``algorithm="auto"`` routing is unchanged). Returns the
    candidates added (none when a candidate of that name is registered)."""
    from . import grouped_convolution as gc

    done = []
    for direction in ("fwd", "dgrad"):
        cand = pinned_conv_candidate(pins, direction)
        regs = (gc._REGISTRIES[direction], gc._EXECUTION_REGISTRIES[direction])
        if any(c.name == cand.name for c in regs[0].candidates()):
            continue
        for reg in regs:
            reg.register(cand)
        done.append(cand)
    return tuple(done)


def request_from_miopen_cmd(cmd: str, arch: str, **kw):
    """A ``ConvGroupedRequest`` for a MIOpenDriver conv command line."""
    from .grouped_convolution import ConvGroupedRequest

    direction, dtype, key = parse_miopen_cmd(cmd)
    vals = dict(zip(_KEY_FIELDS, key))
    return ConvGroupedRequest(arch=arch, dtype=dtype, direction=direction, **vals, **kw)
