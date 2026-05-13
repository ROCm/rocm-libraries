# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""HSACO / ISA analysis helpers.

The functions here intentionally avoid ROCm-profiler dependencies. They use
`llvm-objdump` and `llvm-readelf` when present, then parse the text output into
small dataclasses that tests and experiment scripts can compare.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import shutil
import subprocess
from typing import Dict, Mapping, Optional


@dataclass(frozen=True)
class IsaStats:
    """Instruction-family counts from final AMDGPU ISA text."""

    mfma: int = 0
    buffer_load: int = 0
    buffer_store: int = 0
    buffer_load_lds: int = 0
    ds_read: int = 0
    ds_write: int = 0
    s_barrier: int = 0
    s_waitcnt: int = 0
    sched_barrier: int = 0
    valu: int = 0
    salu: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "mfma": self.mfma,
            "buffer_load": self.buffer_load,
            "buffer_store": self.buffer_store,
            "buffer_load_lds": self.buffer_load_lds,
            "ds_read": self.ds_read,
            "ds_write": self.ds_write,
            "s_barrier": self.s_barrier,
            "s_waitcnt": self.s_waitcnt,
            "sched_barrier": self.sched_barrier,
            "valu": self.valu,
            "salu": self.salu,
        }


@dataclass(frozen=True)
class ResourceInfo:
    """Resource metadata from AMDHSA code-object notes or disassembly."""

    vgpr_count: Optional[int] = None
    agpr_count: Optional[int] = None
    sgpr_count: Optional[int] = None
    lds_bytes: Optional[int] = None
    scratch_bytes: Optional[int] = None
    wavefront_size: Optional[int] = None
    workgroup_size_hint: Optional[int] = None
    raw: Mapping[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Optional[int]]:
        return {
            "vgpr_count": self.vgpr_count,
            "agpr_count": self.agpr_count,
            "sgpr_count": self.sgpr_count,
            "lds_bytes": self.lds_bytes,
            "scratch_bytes": self.scratch_bytes,
            "wavefront_size": self.wavefront_size,
            "workgroup_size_hint": self.workgroup_size_hint,
        }


@dataclass(frozen=True)
class HsacoAnalysis:
    """Static analysis result for one HSACO file."""

    path: Path
    isa: IsaStats
    resources: ResourceInfo
    objdump_text: str = ""
    readelf_text: str = ""


_AMDHSA_KV_RE = re.compile(r"\.?(amdhsa_[A-Za-z0-9_]+)\s+([0-9]+)")
_YAML_KV_RE = re.compile(r"^\s*\.?([A-Za-z0-9_]+):\s*([0-9]+)\s*$")


def _instruction_lines(text: str):
    """Yield lines likely to contain final ISA instructions."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("."):
            continue
        # llvm-objdump ISA lines usually contain either an address prefix or
        # leading whitespace followed by the mnemonic. Keep this permissive so
        # tests can feed small snippets directly.
        yield stripped


def parse_isa(text: str) -> IsaStats:
    """Parse final ISA text into coarse instruction-family counts."""

    mfma = buffer_load = buffer_store = buffer_load_lds = 0
    ds_read = ds_write = s_barrier = s_waitcnt = sched_barrier = 0
    valu = salu = 0

    for line in _instruction_lines(text):
        # Normalize away address/encoding prefixes. The mnemonic is still
        # present as a token in llvm-objdump output.
        if "v_mfma" in line:
            mfma += 1
        if "buffer_load" in line:
            if "lds" in line:
                buffer_load_lds += 1
            else:
                buffer_load += 1
        if "buffer_store" in line:
            buffer_store += 1
        if "ds_read" in line:
            ds_read += 1
        if "ds_write" in line:
            ds_write += 1
        if "s_barrier" in line:
            s_barrier += 1
        if "s_waitcnt" in line:
            s_waitcnt += 1
        if "sched_barrier" in line:
            sched_barrier += 1

        # Crude scalar/vector ALU buckets. These are intentionally broad
        # enough to spot large address-arithmetic regressions.
        mnemonic = _extract_mnemonic(line)
        if mnemonic.startswith("v_") and not mnemonic.startswith("v_mfma"):
            valu += 1
        if mnemonic.startswith("s_") and mnemonic not in ("s_barrier", "s_waitcnt"):
            salu += 1

    return IsaStats(
        mfma=mfma,
        buffer_load=buffer_load,
        buffer_store=buffer_store,
        buffer_load_lds=buffer_load_lds,
        ds_read=ds_read,
        ds_write=ds_write,
        s_barrier=s_barrier,
        s_waitcnt=s_waitcnt,
        sched_barrier=sched_barrier,
        valu=valu,
        salu=salu,
    )


def _extract_mnemonic(line: str) -> str:
    # Typical forms:
    #   000000000000: d3d94000 v_mfma...
    #   s_waitcnt vmcnt(0)
    tokens = line.replace("\t", " ").split()
    for tok in tokens:
        if tok.startswith(("v_", "s_", "ds_", "buffer_", "flat_", "global_")):
            return tok
    return tokens[0] if tokens else ""


def parse_resources(text: str) -> ResourceInfo:
    """Parse AMDHSA resource metadata from readelf/objdump text.

    Handles both forms commonly seen in ROCm tool output:

    - assembly metadata, e.g. `.amdhsa_next_free_vgpr 76`
    - YAML-ish metadata, e.g. `sgpr_count: 56`
    """

    raw: Dict[str, int] = {}

    for line in text.splitlines():
        m = _AMDHSA_KV_RE.search(line)
        if m:
            raw[m.group(1)] = int(m.group(2))
            continue
        y = _YAML_KV_RE.match(line)
        if y:
            raw[y.group(1)] = int(y.group(2))

    def first(*keys: str) -> Optional[int]:
        for key in keys:
            if key in raw:
                return raw[key]
        return None

    return ResourceInfo(
        vgpr_count=first("amdhsa_next_free_vgpr", "vgpr_count", "next_free_vgpr"),
        agpr_count=first("amdhsa_next_free_agpr", "agpr_count", "next_free_agpr"),
        sgpr_count=first("amdhsa_next_free_sgpr", "sgpr_count", "next_free_sgpr"),
        lds_bytes=first(
            "amdhsa_group_segment_fixed_size",
            "group_segment_fixed_size",
            "lds_bytes",
        ),
        scratch_bytes=first(
            "amdhsa_private_segment_fixed_size",
            "private_segment_fixed_size",
            "scratch_bytes",
        ),
        wavefront_size=first("amdhsa_wavefront_size32", "wavefront_size"),
        workgroup_size_hint=first("amdhsa_workgroup_processor_mode"),
        raw=raw,
    )


def analyze_hsaco(
    hsaco_path: Path,
    *,
    objdump: str = "llvm-objdump",
    readelf: str = "llvm-readelf",
    keep_text: bool = False,
) -> HsacoAnalysis:
    """Disassemble and inspect one HSACO file.

    Raises `FileNotFoundError` if the code object or tools are missing, and
    `RuntimeError` if the tool returns non-zero.
    """

    path = Path(hsaco_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if shutil.which(objdump) is None:
        raise FileNotFoundError(f"missing tool: {objdump}")
    if shutil.which(readelf) is None:
        raise FileNotFoundError(f"missing tool: {readelf}")

    obj = subprocess.run(
        [objdump, "-d", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if obj.returncode != 0:
        raise RuntimeError(f"{objdump} failed: {obj.stderr or obj.stdout}")

    # --notes is where code-object metadata generally lives. Some toolchains
    # expose extra metadata only in --all, so parse the objdump text too below.
    relf = subprocess.run(
        [readelf, "--notes", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if relf.returncode != 0:
        raise RuntimeError(f"{readelf} failed: {relf.stderr or relf.stdout}")

    resource_text = relf.stdout + "\n" + obj.stdout
    return HsacoAnalysis(
        path=path,
        isa=parse_isa(obj.stdout),
        resources=parse_resources(resource_text),
        objdump_text=obj.stdout if keep_text else "",
        readelf_text=relf.stdout if keep_text else "",
    )
