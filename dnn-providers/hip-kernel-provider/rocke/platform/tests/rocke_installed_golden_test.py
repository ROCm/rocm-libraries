#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Golden IR byte-stability gate for an installed rocKE package.

This is the host-only (no GPU) golden test for the provider CI lane. It rebuilds
the representative IR cases from the installed ``rocke`` package, hashes the
Python-lowered LLVM IR for every llvm flavor the golden holds -- the flavor is
an argument to lowering, so the host's own ROCm vintage does not limit what can
be checked -- and compares the digests to the committed golden. Any drift means
emitted IR changed.

It is intended to run directly from an installed staging prefix, co-located with
the ``rocke`` package, the ``rocke_ir_parity_harness`` module, and the golden
shard directory under ``golden/``. It fixes up ``sys.path`` before importing anything, so
RockCI can execute it from a staged test artifact:

  python rocke_installed_golden.py [--rocke-shard-from-gtest-env]

With ``--rocke-shard-from-gtest-env`` it checks only the cases of its CI shard
(``rocke.tests.ci_shard``: ``ROCKE_TEST_SHARD``, else ``GTEST_SHARD_INDEX`` and
``GTEST_TOTAL_SHARDS``); each case falls in exactly one shard, so the shards of
one run together check every case. Without either variable it checks them all.
"""

from __future__ import annotations

import argparse
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
    # An install co-locates the parity harness with this script (covered by
    # script.parent above); a checkout keeps it under tests/instances/.
    harness_dir = script.parent / "instances"
    if (harness_dir / "rocke_ir_parity_harness.py").is_file():
        if harness_dir not in found:
            found.append(harness_dir)
    # Some harness families (attention and KDA) build library kernels, so
    # `kernels` and `builders` must resolve too. They are staged under
    # tests/library/ in an install (the destination TheRock's test-artifact
    # globs capture) and live in the sibling library tree in a checkout.
    lib_roots = [script.parent / "tests" / "library"]
    if len(script.parents) > 2:
        lib_roots.append(script.parents[2] / "library")
    for lib_root in lib_roots:
        if (lib_root / "kernels").is_dir():
            if lib_root not in found:
                found.append(lib_root)
            break
    for path in reversed(found):
        sys.path.insert(0, str(path))


def _find_golden() -> Path:
    script = Path(__file__).resolve()
    # The golden is a directory of per-family JSON shards. Co-located layouts:
    # <dir>/golden/<name> (installed) or a sibling golden/ dir one level up.
    # Fall back to the in-source tests/golden/ when run from a checkout.
    name = "rocke_representative_ir_sha256"
    candidates = [
        script.parent / "golden" / name,
        script.parent / name,
        script.parents[1] / "golden" / name,
        script.parents[1] / "tests" / "golden" / name,
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    raise SystemExit(
        "golden not found; looked in:\n  " + "\n  ".join(str(c) for c in candidates)
    )


def main(argv: list[str] | None = None) -> int:
    _add_installed_python_paths()

    from rocke.core.ir_golden import drift_hint, golden_case_ids, load_golden
    from rocke.tests.ci_shard import (
        add_shard_argument,
        header_line,
        shard_from_arguments,
    )
    from rocke_ir_parity_harness import (
        GOLDEN_FLAVORS,
        GOLDEN_NAME,
        cases,
        check_golden,
    )

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_shard_argument(parser, help="check only this CI shard's cases")
    arguments = parser.parse_args(argv)
    shard = shard_from_arguments(parser, arguments)

    golden = _find_golden()
    flavors = ", ".join(GOLDEN_FLAVORS)
    owns_case = None
    if arguments.shard_from_gtest_env:
        print(header_line(shard))
    if shard is not None:
        # A case only the golden still names is checked (reported missing) by
        # the shard that owns its id, like every other case.
        case_ids = {case["case_id"] for case in cases()}
        case_ids |= golden_case_ids(load_golden(golden))
        print(shard.split_line(case_ids))
        owns_case = shard.owns
    drift = check_golden(golden, owns_case=owns_case)
    if drift:
        print(f"rocKE installed golden gate: FAIL ({flavors})\n  " + "\n  ".join(drift))
        print(drift_hint(GOLDEN_NAME, drift))
        return 1
    print(f"rocKE installed golden gate: PASS " f"({flavors}, golden={golden.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
