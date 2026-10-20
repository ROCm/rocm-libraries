# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Categorize the AMDGPU ISA for a compiled CK DSL kernel.

Builds each (label, KernelDef) pair for ``mcpu``, runs
``llvm-objdump -d --mcpu=<arch>`` on the produced HSACO, and tallies opcodes by
category:

- ``mfma`` / ``wmma`` — matrix-core ops (``v_mfma_*`` on CDNA, ``v_wmma_*`` /
  ``v_swmmac_*`` on RDNA and gfx12)
- ``ds_read`` / ``ds_write`` — LDS reads/writes (``ds_read_*`` / ``ds_write_*``
  on CDNA, ``ds_load_*`` / ``ds_store_*`` on gfx11+), ``ds_read_tr`` for
  transpose reads
- ``vmem_load`` / ``vmem_store`` (``buffer_*``, ``global_*``)
- ``scratch_load`` / ``scratch_store`` — register spills
- ``waitcnt`` — memory waits (``s_waitcnt*`` and gfx12 ``s_wait_*cnt``), with
  the most-common encoded operands; ``s_wait_alu`` (gfx12 ALU dependency wait)
  separately
- ``barrier`` — barriers (gfx12 ``s_barrier_signal`` + ``s_barrier_wait`` = 1)
- ``valu`` and ``salu`` — VALU/SALU instruction families (with sub-buckets)

The headline counts come from ``rocke.analysis.isa.parse_isa``; this probe
adds VALU/SALU sub-buckets that surface the difference between a
scalar-heavy kernel and a tiled matmul. Categories that are zero are omitted.

Programmatic use (import the framework; drive it from the library-side
attention driver ``dsl_probe_attention_demos.py`` for attention specs):

    from probe_isa_inspect import probe_isa_inspect
    probe_isa_inspect([(label, kdef) for label, kdef in ...])

This script complements ``analyze_lds_conflicts.py`` (which needs
rocprof hardware counters) by giving a static instruction-mix view that
works without a GPU run.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable


def _bootstrap_rocke() -> None:
    try:
        import rocke  # noqa: F401

        return
    except ImportError:
        pass
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "python"
        if (candidate / "rocke" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            return
        candidate = parent / "rocke" / "__init__.py"
        if candidate.exists():
            sys.path.insert(0, str(parent))
            return


_bootstrap_rocke()

from rocke.analysis.isa import parse_isa
from rocke.core.lower_llvm import ArchDomainWarning
from rocke.helpers.compile import compile_kernel  # noqa: E402


_VALU_SUBS = (
    "v_mfma_",
    "v_wmma_",
    "v_swmmac_",
    "v_exp_",
    "v_rcp_",
    "v_cvt_",
    "v_pk_",
    "v_fma_",
    "v_mul_",
    "v_add_",
    "v_sub_",
    "v_mad_",
    "v_and_",
    "v_or_",
    "v_xor_",
    "v_lshl_",
    "v_lshr_",
    "v_ashr_",
    "v_mov_",
    "v_cndmask",
    "v_cmp_",
    "v_max_",
    "v_min_",
    "v_perm_",
    "v_bfe_",
    "v_alignb",
)
_SALU_SUBS = (
    "s_add_",
    "s_sub_",
    "s_mul_",
    "s_and_",
    "s_or_",
    "s_lshl_",
    "s_lshr_",
    "s_ashr_",
    "s_load_",
    "s_mov_",
    "s_cmp_",
    "s_cselect",
    "s_branch",
    "s_cbranch",
    "s_setpc",
    "s_swappc",
    "s_setprio",
)


def _objdump_path() -> str:
    candidates = [
        os.environ.get("LLVM_OBJDUMP"),
        "/opt/rocm/llvm/bin/llvm-objdump",
        "llvm-objdump",
    ]
    for c in candidates:
        if not c:
            continue
        try:
            subprocess.run([c, "--version"], capture_output=True, timeout=5, check=True)
            return c
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ):
            continue
    raise FileNotFoundError(
        "no llvm-objdump found; install /opt/rocm/llvm/bin/llvm-objdump "
        "or set LLVM_OBJDUMP"
    )


def disassemble_hsaco(hsaco_bytes: bytes, mcpu: str = "gfx950") -> str:
    """Return the assembly text emitted by ``llvm-objdump -d``."""
    objdump = _objdump_path()
    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False) as f:
        f.write(hsaco_bytes)
        hsaco_path = f.name
    try:
        proc = subprocess.run(
            [objdump, "-d", f"--mcpu={mcpu}", hsaco_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return proc.stdout
    finally:
        try:
            os.unlink(hsaco_path)
        except OSError:
            pass


_OPCODE_RE = re.compile(
    r"^([sv]_[a-z0-9_]+|ds_[a-z0-9_]+|buffer_[a-z0-9_]+"
    r"|global_[a-z0-9_]+|flat_[a-z0-9_]+|image_[a-z0-9_]+|scratch_[a-z0-9_]+)\b"
)
# Memory-op prefixes that parse_isa already counts as a headline category.
_COUNTED_MEM_PREFIXES = (
    "ds_read",
    "ds_load",
    "ds_write",
    "ds_store",
    "buffer_load",
    "buffer_store",
    "global_load",
    "global_store",
    "scratch_load",
    "scratch_store",
)


def categorize_isa(isa: str) -> tuple[Counter, Counter]:
    """Return ``(category_counts, waitcnt_pattern_counts)``."""
    st = parse_isa(isa)
    cats: Counter = Counter(
        {
            "mfma": st.mfma,
            "wmma": st.wmma,
            "ds_read": st.ds_read,
            "ds_write": st.ds_write,
            "waitcnt": st.s_waitcnt,
            "s_wait_alu": st.s_wait_alu,
            "barrier": st.s_barrier,
            "vmem_load": st.buffer_load
            + st.buffer_load_lds
            + st.global_load
            + st.global_load_lds,
            "vmem_store": st.buffer_store + st.global_store,
            "scratch_load": st.scratch_load,
            "scratch_store": st.scratch_store,
            "valu": st.valu,
            "salu": st.salu,
        }
    )
    waitcnt_patterns: Counter = Counter()
    for raw in isa.split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = _OPCODE_RE.match(line)
        if not m:
            continue
        opcode = m.group(1)

        # Sub-bucket VALU.
        if opcode.startswith("v_"):
            for sub in _VALU_SUBS:
                if opcode.startswith(sub):
                    cats[f"  {sub.rstrip('_')}"] += 1
                    break
            else:
                cats["  v_other"] += 1
        # Sub-bucket SALU.
        elif opcode.startswith("s_"):
            for sub in _SALU_SUBS:
                if opcode.startswith(sub):
                    cats[f"  {sub.rstrip('_')}"] += 1
                    break
            if opcode.startswith("s_waitcnt"):
                mm = re.search(r"s_waitcnt\s+(.+)$", raw)
                if mm:
                    waitcnt_patterns[mm.group(1).strip()[:48]] += 1
            elif opcode.startswith("s_wait_") and opcode.endswith("cnt"):
                # gfx12 split waits: keep the counter name, drop the comment.
                operand = raw.split(opcode, 1)[1].split("//", 1)[0].strip()
                waitcnt_patterns[f"{opcode} {operand}"[:48]] += 1
        elif opcode.startswith(("ds_read", "ds_load")) and "_tr" in opcode:
            cats["ds_read_tr"] += 1
        elif not opcode.startswith(_COUNTED_MEM_PREFIXES):
            cats["other"] += 1
    return +cats, waitcnt_patterns


def probe_isa_inspect(
    entries: Iterable[tuple[str, "object"]],
    *,
    mcpu: str = "gfx950",
    show_waitcnt: int = 6,
) -> list[dict]:
    """Compile each entry, disassemble, and print a category summary."""
    rows: list[dict] = []
    for label, kdef in entries:
        try:
            # A kernel that needs an intrinsic the target provably can't lower
            # would otherwise abort the process inside LLVM; fail just this entry.
            with warnings.catch_warnings():
                warnings.simplefilter("error", ArchDomainWarning)
                artifact = compile_kernel(kdef, arch=mcpu, capture_ir_text=False)
        except Exception as e:  # noqa: BLE001
            print(f"\n=== {label} === BUILD-FAIL: {type(e).__name__}: {e}")
            continue
        try:
            isa = disassemble_hsaco(artifact.hsaco, mcpu=mcpu)
        except Exception as e:  # noqa: BLE001
            print(f"\n=== {label} === OBJDUMP-FAIL: {type(e).__name__}: {e}")
            continue
        cats, waitcnts = categorize_isa(isa)
        rows.append({"label": label, "cats": dict(cats), "waitcnts": dict(waitcnts)})
        print(f"\n=== {label} ===")
        total = isa.count("\n")
        print(f"Total ISA lines: {total}")
        for key, val in sorted(cats.items(), key=lambda kv: -kv[1]):
            print(f"  {key:<22} {val:>6}")
        if "ds_read" in cats:
            print(f"  ds_read non-tr: {cats['ds_read'] - cats.get('ds_read_tr', 0)}")
        if waitcnts and show_waitcnt > 0:
            print(f"  waitcnt patterns (top {show_waitcnt}):")
            for pat, val in waitcnts.most_common(show_waitcnt):
                print(f"    {val:>4}× {pat}")
    return rows


# ---- Demos -------------------------------------------------------------
# The attention-tiled-2D demo has moved to
# library/builders/common/dsl_probe_attention_demos.py.
# Run: python dsl_probe_attention_demos.py --probe isa_inspect


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mcpu", default="gfx950")
    p.parse_args(argv)
    print(
        "probe_isa_inspect: no built-in demo. "
        "For the attention-tiled-2D ISA inspection use:\n"
        "  python library/builders/common/dsl_probe_attention_demos.py "
        "--probe isa_inspect"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
