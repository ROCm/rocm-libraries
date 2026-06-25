# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Structural comparison of unified-attention HSACOs (Triton vs CK DSL).

For each input HSACO:
  1. Auto-unbundle if wrapped in __CLANG_OFFLOAD_BUNDLE__ (CK DSL); otherwise
     treat as a raw AMDGPU ELF (Triton emits these directly).
  2. Extract per-kernel metadata (vgpr/sgpr/agpr/lds) from the AMDGPU YAML
     note section via ``llvm-readelf --notes``.
  3. Disassemble with ``llvm-objdump --mcpu=gfx950 -d`` and parse instructions.
  4. Identify the inner K-loop body as the backward-branch loop containing
     the densest MFMA cluster.
  5. Classify instructions into categories (MFMA, LDS r/w, global r/w, AGPR
     shuffle, cross-lane shuffle, VALU, scalar/branch, wait/barrier).
  6. Report metadata, whole-kernel histogram, K-loop histogram (per outer
     iter), and loop-invariant ops.

Optional sidecar: if a ``<hsaco-stem>.json`` exists next to the hsaco
(Triton convention), also report ``shared`` (dynamic LDS) and ``num_warps``
from it.

Usage:
    python src/stage3_extract_isa/compare_ua_hsacos.py \\
        --label triton tests/logs/triton_hsaco_dump/kernel_unified_attention_2d.hsaco \\
        --label rocke_stock tests/logs/regpv_pair/stock__...hsaco \\
        --label rocke_regpv tests/logs/regpv_pair/regpv__...hsaco
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROCM_LLVM = Path("/opt/rocm/llvm/bin")
OBJDUMP = ROCM_LLVM / "llvm-objdump"
READELF = ROCM_LLVM / "llvm-readelf"
BUNDLER = ROCM_LLVM / "clang-offload-bundler"
GFX = "gfx950"


# --- Instruction classification ----------------------------------------------


CATEGORIES: list[tuple[str, re.Pattern]] = [
    ("MFMA", re.compile(r"^v_mfma_")),
    ("LDS_read", re.compile(r"^ds_read_|^ds_load_")),
    ("LDS_write", re.compile(r"^ds_write_|^ds_store_")),
    ("LDS_other", re.compile(r"^ds_(permute|bpermute|swizzle|atomic)")),
    (
        "global_load",
        re.compile(r"^global_load_|^buffer_load_|^flat_load_|^global_load_lds_"),
    ),
    ("global_store", re.compile(r"^global_store_|^buffer_store_|^flat_store_")),
    ("AGPR_move", re.compile(r"^v_accvgpr_")),
    (
        "xlane_shuffle",
        re.compile(
            r"^v_permlane|^v_perm_b32|^v_movrels|^v_movreld|^ds_swizzle_b32|^v_readlane|^v_writelane"
        ),
    ),
    (
        "VALU",
        re.compile(
            r"^v_(add|sub|mul|fma|max|min|cvt|cmp|cndmask|exp|rcp|rsq|log|and|or|xor|lshl|lshr|ashr|mov|pack|sad|med|bfe|bfi|alignbit|readfirstlane|dot|mad)"
        ),
    ),
    ("wait_barrier", re.compile(r"^s_(waitcnt|barrier|nop|sleep|sendmsg|wait)")),
    (
        "scalar",
        re.compile(
            r"^s_(load|store|add|sub|mul|cmp|cmovk|cmov|and|or|xor|lshl|lshr|ashr|mov|cselect|brev|min|max|abs|getpc|bitset|bitcmp|ff1|ff0|bcnt|memrealtime|setpc|setreg|setprio|ttracedata|sext|csel)"
        ),
    ),
    ("branch", re.compile(r"^s_(cbranch|branch|swappc|endpgm|cbr|trap|sethalt)")),
]


def classify(mnemonic: str) -> str:
    for name, pat in CATEGORIES:
        if pat.match(mnemonic):
            return name
    return "other"


# --- HSACO unbundling --------------------------------------------------------


def _is_offload_bundle(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(24).startswith(b"__CLANG_OFFLOAD_BUNDLE__")


def to_amdgpu_elf(hsaco: Path, tmpdir: Path) -> Path:
    if not _is_offload_bundle(hsaco):
        return hsaco
    out = tmpdir / (hsaco.stem + ".gfx950.elf")
    cmd = [
        str(BUNDLER),
        "-unbundle",
        f"--input={hsaco}",
        "--type=o",
        f"--targets=hipv4-amdgcn-amd-amdhsa--{GFX}",
        f"--output={out}",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


# --- AMDGPU metadata extraction (YAML in .note) ------------------------------


def extract_kernel_meta(elf: Path) -> list[dict]:
    """Parse llvm-readelf --notes output (YAML) → list of kernel dicts."""
    out = subprocess.run(
        [str(READELF), "--notes", str(elf)], capture_output=True, text=True
    )
    text = out.stdout
    # Find the YAML block between "---" and "..."
    m = re.search(r"---\n(.*?)\n\.\.\.", text, flags=re.DOTALL)
    if not m:
        return []
    yaml_text = m.group(1)

    # We don't want to add a hard PyYAML dep. Parse the minimal subset we
    # need: per-kernel ``.vgpr_count``, ``.sgpr_count``, ``.agpr_count``,
    # ``.group_segment_fixed_size``, ``.name``, ``.symbol``,
    # ``.private_segment_fixed_size``, ``.max_flat_workgroup_size``.
    kernels: list[dict] = []
    cur: dict | None = None
    in_kernels = False
    for ln in yaml_text.splitlines():
        if ln.startswith("amdhsa.kernels:"):
            in_kernels = True
            continue
        if not in_kernels:
            continue
        stripped = ln.strip()
        if ln.startswith("  - ") or (cur is None and stripped):
            # Start of new kernel record
            if cur is not None:
                kernels.append(cur)
            cur = {}
        if cur is None:
            continue
        m = re.match(r"\s*-?\s*\.(\w+):\s+(.+)$", ln)
        if m:
            key, val = m.group(1), m.group(2).strip()
            # Strip quotes
            if val.startswith("'") and val.endswith("'"):
                val = val[1:-1]
            try:
                cur[key] = int(val)
            except ValueError:
                cur[key] = val
    if cur is not None and cur:
        kernels.append(cur)
    return kernels


# --- Disassembly parsing -----------------------------------------------------


# Function label: "0000000000001a00 <kernel_unified_attention_2d>:"
_LABEL_RE = re.compile(r"^([0-9a-fA-F]+)\s+<(.+)>:\s*$")
# Instruction: leading whitespace, mnemonic, operands, " // ADDR: BYTES"
_INSN_RE = re.compile(r"^\s+(\S+)(?:\s+.*)?//\s+([0-9a-fA-F]+):\s+[0-9a-fA-F ]+\s*$")
# Branch instruction: same plus parse target hex literal in operand
_BRANCH_RE = re.compile(
    r"^\s+(s_cbranch_\w+|s_branch)\s+(-?\d+|0x[0-9a-fA-F]+)\b.*//\s+([0-9a-fA-F]+):"
)


@dataclass
class KernelDisasm:
    name: str
    instructions: list[tuple[int, str]] = field(
        default_factory=list
    )  # (addr_int, mnem)
    branches: list[tuple[int, int]] = field(
        default_factory=list
    )  # (src_addr, tgt_addr)


def parse_disasm(text: str) -> list[KernelDisasm]:
    kernels: list[KernelDisasm] = []
    cur: KernelDisasm | None = None
    for ln in text.splitlines():
        m_lbl = _LABEL_RE.match(ln)
        if m_lbl:
            cur = KernelDisasm(name=m_lbl.group(2).strip())
            kernels.append(cur)
            continue
        if cur is None:
            continue
        m_br = _BRANCH_RE.match(ln)
        if m_br:
            mnem = m_br.group(1)
            raw_off = m_br.group(2)
            src_addr = int(m_br.group(3), 16)
            try:
                # objdump prints branch offsets as int16 (instructions): 4*off.
                # Triton/CK DSL use the decimal form like "s_branch 56" or the
                # unsigned 65514 (=-22 sign-extended) form for backward jumps.
                if raw_off.startswith("0x"):
                    off_insts = int(raw_off, 16)
                else:
                    off_insts = int(raw_off)
                # Sign-extend from int16 (objdump prints SOPP imm16 unsigned).
                if off_insts >= 0x8000:
                    off_insts -= 0x10000
                # On gfx9 SOPP branches: PC' = PC + 4 + 4*off
                tgt_addr = src_addr + 4 + 4 * off_insts
                cur.branches.append((src_addr, tgt_addr))
            except ValueError:
                pass
            cur.instructions.append((src_addr, mnem))
            continue
        m_ins = _INSN_RE.match(ln)
        if m_ins:
            mnem = m_ins.group(1)
            addr = int(m_ins.group(2), 16)
            cur.instructions.append((addr, mnem))
    return kernels


# --- Loop detection ----------------------------------------------------------


def detect_loops(k: KernelDisasm) -> list[tuple[int, int]]:
    """Backward branches → (target_idx, source_idx) tuples."""
    addr_to_idx: dict[int, int] = {
        addr: i for i, (addr, _m) in enumerate(k.instructions)
    }
    loops: list[tuple[int, int]] = []
    for src, tgt in k.branches:
        if tgt >= src:
            continue
        if src not in addr_to_idx or tgt not in addr_to_idx:
            continue
        loops.append((addr_to_idx[tgt], addr_to_idx[src]))
    return sorted(set(loops), key=lambda r: -(r[1] - r[0]))


def pick_k_loop(loops, k: KernelDisasm):
    best, best_mfma = None, -1
    for s, e in loops:
        n_mfma = sum(1 for _a, m in k.instructions[s : e + 1] if classify(m) == "MFMA")
        if n_mfma > best_mfma:
            best, best_mfma = (s, e), n_mfma
    return best if best_mfma > 0 else None


# --- Histogram + reporting ---------------------------------------------------


def histogram(body) -> Counter:
    return Counter(classify(m) for _a, m in body)


def fmt_hist(h: Counter, total: int) -> str:
    order = [n for n, _ in CATEGORIES] + ["other"]
    lines = []
    for cat in order:
        c = h.get(cat, 0)
        if c == 0:
            continue
        pct = 100.0 * c / total if total else 0.0
        lines.append(f"  {cat:>14s}: {c:>6d}  ({pct:5.1f}%)")
    return "\n".join(lines)


def _pick_main_kernel(kernels: list[KernelDisasm]) -> KernelDisasm | None:
    if not kernels:
        return None
    kernels.sort(
        key=lambda k: -sum(1 for _a, m in k.instructions if classify(m) == "MFMA")
    )
    return kernels[0]


def _match_meta(meta_list: list[dict], kname: str) -> dict | None:
    for m in meta_list:
        if m.get("name") == kname or m.get("symbol", "").startswith(kname):
            return m
    return meta_list[0] if meta_list else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--label",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
    )
    p.add_argument("--dump-loops", action="store_true")
    args = p.parse_args()

    tmpdir = Path(tempfile.mkdtemp(prefix="hsaco_compare_"))
    entries = []

    for name, path_str in args.label:
        path = Path(path_str).resolve()
        if not path.exists():
            print(f"[{name}] ERROR: {path} not found", file=sys.stderr)
            return 1
        elf = to_amdgpu_elf(path, tmpdir)
        meta = extract_kernel_meta(elf)
        text = subprocess.run(
            [str(OBJDUMP), f"--mcpu={GFX}", "-d", str(elf)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        kernels = parse_disasm(text)
        k = _pick_main_kernel(kernels)
        loops = detect_loops(k) if k else []
        klp = pick_k_loop(loops, k) if k else None
        kmeta = _match_meta(meta, k.name) if k else None

        # Triton sidecar
        sidecar = path.with_suffix(".json")
        sc = None
        if sidecar.exists():
            try:
                sc = json.loads(sidecar.read_text())
            except Exception:
                pass

        entries.append((name, path, elf, k, kmeta, sc, loops, klp))

    print()
    print("=" * 100)
    print(f"  COMPARISON: {len(entries)} kernel(s) on {GFX}")
    print("=" * 100)

    # --- Per-kernel summary ---
    for name, path, _elf, k, kmeta, sc, loops, klp in entries:
        if k is None:
            print(f"\n--- {name} : <no kernel parsed> ---")
            continue
        n_total = len(k.instructions)
        lds_static = kmeta.get("group_segment_fixed_size", 0) if kmeta else 0
        lds_dyn = sc.get("shared", 0) if sc else 0
        lds_total = lds_static or lds_dyn
        scratch = kmeta.get("private_segment_fixed_size", 0) if kmeta else 0
        kargs = kmeta.get("kernarg_segment_size", 0) if kmeta else 0
        mwgs = kmeta.get("max_flat_workgroup_size", 0) if kmeta else 0
        nw_meta = sc.get("num_warps") if sc else None
        nw_str = f"  num_warps(meta)={nw_meta}" if nw_meta else ""
        print(f"\n--- {name} ---")
        print(f"  hsaco        : {path.name}  ({path.stat().st_size} bytes)")
        print(f"  kernel       : {k.name}")
        print(f"  total insts  : {n_total}")
        print(
            f"  VGPRs        : {kmeta.get('vgpr_count', '?') if kmeta else '?'}"
            f"   SGPRs: {kmeta.get('sgpr_count', '?') if kmeta else '?'}"
            f"   AGPRs: {kmeta.get('agpr_count', '?') if kmeta else '?'}"
        )
        print(
            f"  LDS bytes    : static={lds_static}  dynamic={lds_dyn}  total={lds_total}"
        )
        print(f"  scratch={scratch}  kernarg={kargs}  max_flat_wg={mwgs}{nw_str}")
        if klp:
            sa = k.instructions[klp[0]][0]
            ea = k.instructions[klp[1]][0]
            print(
                f"  K-loop range : insn[{klp[0]}..{klp[1]}] "
                f"({klp[1] - klp[0] + 1} insts)  [0x{sa:x}..0x{ea:x}]"
                f"   ({len(loops)} backward branches total)"
            )
        else:
            print(f"  K-loop       : <none detected>  ({len(loops)} branches)")

    # --- Whole-kernel histogram ---
    print()
    print("=" * 100)
    print("  WHOLE-KERNEL INSTRUCTION MIX")
    print("=" * 100)
    for name, _path, _elf, k, _kmeta, _sc, _loops, _klp in entries:
        if k is None:
            continue
        n = len(k.instructions)
        n_mfma = sum(1 for _a, m in k.instructions if classify(m) == "MFMA")
        print(f"\n--- {name} ({n} insts, {n_mfma} MFMA total) ---")
        print(fmt_hist(histogram(k.instructions), n))

    # --- K-loop histogram (per outer iter) ---
    print()
    print("=" * 100)
    print("  K-LOOP BODY INSTRUCTION MIX (per outer-loop iteration)")
    print("=" * 100)
    for name, _path, _elf, k, _kmeta, _sc, _loops, klp in entries:
        if k is None or klp is None:
            print(f"\n--- {name}: no K-loop ---")
            continue
        body = k.instructions[klp[0] : klp[1] + 1]
        n = len(body)
        n_mfma = sum(1 for _a, m in body if classify(m) == "MFMA")
        print(f"\n--- {name} ({n} insts, {n_mfma} MFMA per iter) ---")
        print(fmt_hist(histogram(body), n))

    # --- Loop-invariant ops ---
    print()
    print("=" * 100)
    print("  LOOP-INVARIANT (out-of-loop) INSTRUCTION MIX")
    print("=" * 100)
    for name, _path, _elf, k, _kmeta, _sc, _loops, klp in entries:
        if k is None or klp is None:
            continue
        outside = k.instructions[: klp[0]] + k.instructions[klp[1] + 1 :]
        n = len(outside)
        print(f"\n--- {name} ({n} insts outside K-loop) ---")
        print(fmt_hist(histogram(outside), n))

    if args.dump_loops:
        print()
        print("=" * 100)
        print("  K-LOOP DISASSEMBLY")
        print("=" * 100)
        for name, _path, _elf, k, _kmeta, _sc, _loops, klp in entries:
            if k is None or klp is None:
                continue
            print(f"\n--- {name} ---")
            for addr, mnem in k.instructions[klp[0] : klp[1] + 1]:
                print(f"    0x{addr:x}: {mnem}")

    shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
