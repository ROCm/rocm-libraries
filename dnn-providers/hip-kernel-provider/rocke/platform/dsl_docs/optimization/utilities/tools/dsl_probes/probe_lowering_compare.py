# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compare the LLVM-direct and HIP-debug lowering backends.

For each KernelDef this script:

1. compiles via ``rocke.helpers.compile_kernel`` (LLVM-direct → comgr →
   HSACO);
2. lowers via ``rocke.core.lower_hip.lower_kernel_to_hip`` and compiles
   the emitted HIP C++ via ``hipcc --genco`` to a second HSACO;
3. reports the HSACO sizes side-by-side and (optionally) inspects each
   blob with ``llvm-readelf --notes`` so VGPR / LDS deltas are visible
   without launching either kernel.

The HIP-debug backend is **not** the production path. It exists to give
authors a readable HIP C++ source they can compile with ``hipcc`` and to
cross-check that production LLVM lowering and HIP debug lowering agree
on what the kernel should be.

Programmatic use (import the framework; drive it from the library-side
attention driver ``dsl_probe_attention_demos.py`` for attention specs):

    from probe_lowering_compare import probe_lowering_compare
    probe_lowering_compare([("decode", kdef_decode), ("prefill", kdef_prefill)])
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
        candidate = parent / "Python"
        if (candidate / "rocke" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            return
        candidate = parent / "rocke" / "__init__.py"
        if candidate.exists():
            sys.path.insert(0, str(parent))
            return


_bootstrap_rocke()

from rocke.core.lower_hip import lower_kernel_to_hip  # noqa: E402
from rocke.helpers.compile import compile_kernel  # noqa: E402


def _readelf_notes(hsaco_bytes: bytes) -> dict:
    readelf = os.environ.get("LLVM_READELF") or "/opt/rocm/llvm/bin/llvm-readelf"
    if not Path(readelf).exists():
        readelf = shutil.which("llvm-readelf") or "llvm-readelf"
    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False) as f:
        f.write(hsaco_bytes)
        path = f.name
    try:
        proc = subprocess.run(
            [readelf, "--notes", path],
            capture_output=True,
            text=True,
            timeout=20,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    text = proc.stdout
    fields = {}
    for name, pattern in [
        ("vgpr", r"\.vgpr_count:\s+(\d+)"),
        ("agpr", r"\.agpr_count:\s+(\d+)"),
        ("sgpr", r"\.sgpr_count:\s+(\d+)"),
        ("spill", r"\.vgpr_spill_count:\s+(\d+)"),
        ("lds", r"\.group_segment_fixed_size:\s+(\d+)"),
        ("scratch", r"\.private_segment_fixed_size:\s+(\d+)"),
    ]:
        m = re.search(pattern, text)
        if m:
            fields[name] = int(m.group(1))
    return fields


def compile_hip_backend(
    kdef,
    *,
    arch: str = "gfx950",
    out_dir: Path | None = None,
    timeout: int = 240,
) -> tuple[bytes, Path]:
    """Lower kdef to HIP, run ``hipcc --genco``, return ``(hsaco, hip_src)``."""
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="rocke_hip_lower_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = kdef.name.replace(".", "_")[:96]
    hip_src = out_dir / f"{stem}.hip"
    hsaco_out = out_dir / f"{stem}.hsaco"
    hip_src.write_text(lower_kernel_to_hip(kdef))
    proc = subprocess.run(
        [
            "hipcc",
            f"--offload-arch={arch}",
            "--genco",
            "-O3",
            str(hip_src),
            "-o",
            str(hsaco_out),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"hipcc rc={proc.returncode}\n"
            f"stdout:\n{proc.stdout[-2000:]}\n"
            f"stderr:\n{proc.stderr[-2000:]}"
        )
    return hsaco_out.read_bytes(), hip_src


def probe_lowering_compare(
    entries: Iterable[tuple[str, "object"]],
    *,
    arch: str = "gfx950",
    inspect_notes: bool = True,
    out_dir: Path | None = None,
) -> list[dict]:
    """Build both backends for each entry and print a side-by-side table."""
    rows: list[dict] = []
    header_cols = ["label", "llvm B", "hip B", "ratio"]
    if inspect_notes:
        header_cols += ["llvm vgpr", "hip vgpr", "llvm lds", "hip lds"]
    print(" ".join(f"{c:>10}" for c in header_cols))
    print("-" * (11 * len(header_cols)))
    for label, kdef in entries:
        try:
            artifact = compile_kernel(kdef, capture_ir_text=False)
            llvm_bytes = artifact.hsaco
        except Exception as e:  # noqa: BLE001
            print(f"{label:<10} LLVM-FAIL: {type(e).__name__}: {e}")
            continue
        try:
            hip_bytes, _hip_src = compile_hip_backend(kdef, arch=arch, out_dir=out_dir)
        except Exception as e:  # noqa: BLE001
            print(f"{label:<10} HIP-FAIL: {type(e).__name__}: {e}")
            continue
        row = {
            "label": label,
            "llvm_bytes": len(llvm_bytes),
            "hip_bytes": len(hip_bytes),
            "ratio": len(hip_bytes) / max(len(llvm_bytes), 1),
        }
        if inspect_notes:
            llvm_n = _readelf_notes(llvm_bytes)
            hip_n = _readelf_notes(hip_bytes)
            row["llvm_notes"] = llvm_n
            row["hip_notes"] = hip_n
            print(
                f"{label:>10} {row['llvm_bytes']:>10} {row['hip_bytes']:>10} "
                f"{row['ratio']:>9.2f}x "
                f"{llvm_n.get('vgpr', '-'):>10} {hip_n.get('vgpr', '-'):>10} "
                f"{llvm_n.get('lds', '-'):>10} {hip_n.get('lds', '-'):>10}"
            )
        else:
            print(
                f"{label:>10} {row['llvm_bytes']:>10} {row['hip_bytes']:>10} "
                f"{row['ratio']:>9.2f}x"
            )
        rows.append(row)
    return rows


# ---- Demo --------------------------------------------------------------
# The attention-tiled-2D demo has moved to
# library/builders/common/dsl_probe_attention_demos.py.
# Run: python dsl_probe_attention_demos.py --probe lowering_compare


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arch", default="gfx950")
    p.add_argument(
        "--no-notes",
        action="store_true",
        help="skip readelf inspection (only compare HSACO sizes)",
    )
    p.parse_args(argv)
    print(
        "probe_lowering_compare: no built-in demo. "
        "For the attention-tiled-2D lowering comparison use:\n"
        "  python library/builders/common/dsl_probe_attention_demos.py "
        "--probe lowering_compare"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
