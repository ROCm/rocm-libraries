#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Golden IR byte-stability gate for an installed rocKE package.

This is the host-only (no GPU) golden test for the provider CI lane. It rebuilds
the representative IR cases from the installed ``rocke`` package, hashes the
Python-lowered LLVM IR for this host's llvm flavor, and compares the digests to
the committed golden. Any drift means emitted IR changed.

It is intended to run directly from an installed staging prefix, co-located with
the ``rocke`` package, the ``rocke_ir_parity_harness`` module, and the golden
JSON under ``golden/``. It fixes up ``sys.path`` before importing anything, so
RockCI can execute it from a staged test artifact:

  python rocke_installed_golden.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def _add_installed_python_paths() -> None:
    script = Path(__file__).resolve()
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    rel_site_packages = (
        Path("lib") / pyver / "site-packages",
        Path("lib64") / pyver / "site-packages",
        Path("lib") / "python" / "site-packages",
        Path("python"),
    )
    # The script may be installed at <prefix>/bin/ (standalone) or nested under a
    # provider test subdir such as <prefix>/bin/<provider>/. Probe a bounded set
    # of ancestor directories as candidate install prefixes, and always include
    # the script's own directory (covers the co-located rocke package and the
    # rocke_ir_parity_harness module in the provider test bucket).
    candidate_prefixes = [script.parents[i] for i in range(min(4, len(script.parents)))]
    found: list[Path] = []
    for prefix in candidate_prefixes:
        for rel in rel_site_packages:
            path = prefix / rel
            if path.is_dir() and path not in found:
                found.append(path)
    if script.parent not in found:
        found.append(script.parent)
    for path in reversed(found):
        sys.path.insert(0, str(path))


def _find_golden() -> Path:
    script = Path(__file__).resolve()
    name = "rocke_representative_ir_sha256.json"
    # Co-located layouts: <dir>/golden/<name> (installed) or a sibling golden/
    # dir one level up. Fall back to the in-source tests/golden/ when run from a
    # checkout.
    candidates = [
        script.parent / "golden" / name,
        script.parent / name,
        script.parents[1] / "golden" / name,
        script.parents[1] / "tests" / "golden" / name,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise SystemExit(
        "golden not found; looked in:\n  " + "\n  ".join(str(c) for c in candidates)
    )


def main() -> int:
    _add_installed_python_paths()

    from rocke_ir_parity_harness import check_golden, current_flavor

    golden = _find_golden()
    flavor = current_flavor()
    drift = check_golden(golden, flavor)
    if drift:
        print(f"rocKE installed golden gate: FAIL ({flavor})\n  " + "\n  ".join(drift))
        return 1
    print(f"rocKE installed golden gate: PASS ({flavor}, golden={golden.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
