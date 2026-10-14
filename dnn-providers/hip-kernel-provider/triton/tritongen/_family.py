# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Facts shared by every Triton op: layout, pins, compile options, error codes, and the
rules that derive an op's names and its kernarg ABI from its contract (``ops/<op>.py``).

Pure data and pure functions; stdlib only. The native packs restate the derived symbol
and field names; keep them in sync.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

# --- Layout ------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
TRITON_DIR = PACKAGE_DIR.parent
HKP_DIR = TRITON_DIR.parent
REQUIREMENTS = TRITON_DIR / "requirements.txt"
FAMILY_FOLDER = "Triton"
CONTENT_DIR = (
    HKP_DIR
    / "src"
    / "engines"
    / "kernel_ingestor_engine"
    / "descriptors"
    / FAMILY_FOLDER
)

# --- Error codes ---------------------------------------------------------------
ERR_ENV = "TRITONGEN-ENV"
ERR_ARCH = "TRITONGEN-ARCH"
ERR_ABI = "TRITONGEN-ABI"
ERR_SCRATCH = "TRITONGEN-SCRATCH"
ERR_ERRATUM = "TRITONGEN-ERRATUM"
ERR_COMPILE = "TRITONGEN-COMPILE"
ERR_INPUT = "TRITONGEN-INPUT"
ERR_CHECK = "TRITONGEN-CHECK"
ERR_INTERNAL = "TRITONGEN-INTERNAL"


class GeneratorError(RuntimeError):
    """A generator failure; the message starts with its ``TRITONGEN-*`` code."""


# --- Pins ----------------------------------------------------------------------
TRITON_VERSION = "3.8.0"
TRITON_WHEEL_SHA256 = "e91ffa46d095b252248297292dd22bcbacd53a125a0c2eefbbbf74925a320bc3"
WARP_SIZE = 64  # GPUTarget("hip", arch, 64); the wave size on every CDNA arch
GPU_TARGET_BACKEND = "hip"

HIP_OPTIONS_FIXED: Dict[str, object] = {
    "num_stages": 2,
    "waves_per_eu": 0,
    "enable_fp_fusion": True,
    "allow_flush_denorm": False,
    "matrix_instr_nonkdim": 0,
    "kpack": 1,
    "num_ctas": 1,
    "launch_cooperative_grid": False,
}


def hip_options(num_warps: int) -> Dict[str, object]:
    """Full explicit HIPOptions kwargs for one instance."""
    return {"num_warps": num_warps, **HIP_OPTIONS_FIXED}


# --- Files and versions ----------------------------------------------------------
MANIFEST_FILE = "manifest.json"
MANIFEST_VERSION = 1
DESCRIPTOR_VERSION = "1.0"
UKD_SOURCE_KIND = "hsaco"
SOURCE_MD_SUFFIX = ".SOURCE.md"

# --- Name derivations --------------------------------------------------------------
NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://github.com/ROCm/rocm-libraries/dnn-providers/hip-kernel-provider/triton",
)


def make_id(name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE, name)


def stem(op: str) -> str:
    return f"triton_{op}"


def engine_name(pascal_op: str) -> str:
    return f"hipkernel:Triton{pascal_op}"


def symbol(op: str, kind: str) -> str:
    """``hipkernel.<stem>.<kind>`` for kind in graph_match/kernel_match/score/dispatch."""
    return f"hipkernel.{stem(op)}.{kind}"


def op_file(op: str, kind: str) -> str:
    """The op-level descriptor file of ``kind`` (``ued``/``uhd``/``udd``/``kmd``/``umd``)
    or the per-arch ``kdp``."""
    if kind == "umd":
        return f"{stem(op)}_kernel_match.umd.json"
    return f"{stem(op)}.{kind}.json"


def id_name(kind: str, op: str, *parts: str) -> str:
    """UUIDv5 name: ``<kind>/<op>[/<part>...]`` (e.g. ``ukd/rmsnorm/gfx942/<instance>``)."""
    return "/".join((kind, op, *parts))


def ukd_file(instance: str) -> str:
    return f"{instance}.ukd.json"


def hsaco_file(instance: str) -> str:
    return f"{instance}.hsaco"


# --- Kernarg ABI rule (Triton 3.8.0, AMDGPU) ----------------------------------------
# Every non-constexpr parameter in declaration order, at its natural alignment:
# a pointer is an 8-byte global_buffer, an i32/fp32 a 4-byte by_value. Triton then
# appends two 8-byte global_buffer scratch pointers. No hidden arguments.
SCRATCH_ARG_NAMES: Tuple[str, ...] = ("global_scratch", "profile_scratch")
_POINTER = ("global_buffer", 8)
_SCALARS: Dict[str, Tuple[str, int]] = {"i32": ("by_value", 4), "fp32": ("by_value", 4)}


@dataclass(frozen=True)
class KernArg:
    name: str
    kind: str
    size: int
    offset: int


@dataclass(frozen=True)
class Abi:
    """The kernarg layout an op instance's object must have."""

    symbol: str
    args: Tuple[KernArg, ...]
    segment_size: int

    @property
    def signature(self) -> tuple:
        """``(kind, size, offset)`` per argument, the shape ``describe`` reports."""
        return tuple((a.kind, a.size, a.offset) for a in self.args)


def derive_abi(
    symbol_name: str, signature: Mapping[str, str], arg_names: Mapping[str, str]
) -> Abi:
    """The kernarg layout for a Triton compile ``signature``; argument names come from
    ``arg_names`` (parameter -> semantic name), scratch names from the family rule."""
    slots = []
    for param, triton_type in signature.items():
        if triton_type == "constexpr":
            continue
        if triton_type.startswith("*"):
            kind, size = _POINTER
        elif triton_type in _SCALARS:
            kind, size = _SCALARS[triton_type]
        else:
            raise GeneratorError(
                f"{ERR_ABI}: parameter {param!r} has Triton type {triton_type!r}, "
                f"which has no kernarg rule (pointers, {', '.join(sorted(_SCALARS))})"
            )
        slots.append((arg_names[param], kind, size))
    slots += [(name, *_POINTER) for name in SCRATCH_ARG_NAMES]

    args, offset = [], 0
    for name, kind, size in slots:
        offset = (offset + size - 1) // size * size
        args.append(KernArg(name, kind, size, offset))
        offset += size
    return Abi(symbol_name, tuple(args), offset)
