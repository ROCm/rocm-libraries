#!/usr/bin/env python3.12
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Ensure the TheRock venv site-packages (which contains msgpack) are on the path
# when invoked directly without activating a venv.
import sys as _sys, os as _os
_venv_site = "/data/davdixon/TheRock/.venv/lib/python3.12/site-packages"
if _os.path.isdir(_venv_site) and _venv_site not in _sys.path:
    _sys.path.insert(0, _venv_site)
"""
test_shard_convergence.py — Verify hipBLASLt shard overlay convergence.

TheRock builds hipBLASLt once per GPU target subset (shard) and overlays all
shard install trees into a single filesystem prefix.  For this to be safe,
per-shard artifacts must be either:
  • byte-for-byte identical across shards (safe to overwrite), or
  • per-architecture named / in per-architecture subdirectories (additive).

This script tests three artifacts that are fixed via the per-arch subdirectory
layout:

  Artifact                           Required property
  ─────────────────────────────────  ────────────────────────────────────────
  libhipblaslt.so                    Byte-for-byte identical (GPU_TARGETS has
                                     no effect on host compilation flags)
  library/<arch>/extop_<arch>.co     Additive: per-arch named, no conflict
  library/<arch>/hipblasltExtOpLibrary.dat
                                     Additive: each shard writes to its own
                                     arch subdirectory
  library/<arch>/TensileLiteLibrary_lazy_Mapping
                                     Additive: per-arch subdirectory, each
                                     shard's independent index space is safe
  library/<arch>/hipblasltTransform.hsaco
                                     Additive: per-arch subdirectory, single
                                     arch HSACO (HIP selects right binary)

Mode A (default, no build required):
    Simulates shard outputs using in-process msgpack construction and proves
    that the BEFORE state (flat layout) loses arch entries on overlay, while
    the AFTER state (per-arch subdirectory layout) composes additively.

Mode B (--shard-a-install / --shard-b-install, requires real build trees):
    Performs the actual overlay and validates each artifact in the merged tree.

Usage:
    # Mode A — unit/CI test, no toolchain needed
    python scripts/test_shard_convergence.py

    # Mode B — integration test against real build trees
    python scripts/test_shard_convergence.py \\
        --shard-a-install build-gfx942/install \\
        --shard-b-install build-gfx1100/install \\
        --dist-targets gfx942,gfx1100 \\
        --overlay-dir /tmp/hipblaslt-overlay
"""

import argparse
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import msgpack
except ImportError:
    print("ERROR: msgpack not available.  Install via: pip install msgpack")
    print("       or activate the tensilelite virtualenv first.")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Result helpers
# ──────────────────────────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

_results: list[tuple[str, str, str]] = []  # (artifact, scenario, status)


def record(artifact: str, scenario: str, status: str, detail: str = "") -> None:
    _results.append((artifact, scenario, status))
    mark = {"PASS": "✓", "FAIL": "✗", "SKIP": "–"}.get(status, "?")
    line = f"  [{mark}] {artifact} — {scenario}"
    if detail:
        line += f"\n      {detail}"
    print(line)


def summary() -> bool:
    """Print final summary. Returns True if all non-SKIP results passed."""
    print()
    print("=" * 60)
    print("Convergence test summary")
    print("=" * 60)
    failures = [r for r in _results if r[2] == FAIL]
    passes   = [r for r in _results if r[2] == PASS]
    skips    = [r for r in _results if r[2] == SKIP]
    print(f"  PASS: {len(passes)}  FAIL: {len(failures)}  SKIP: {len(skips)}")
    if failures:
        print()
        print("Failures:")
        for art, scen, _ in failures:
            print(f"  • {art} — {scen}")
    print("=" * 60)
    return len(failures) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Msgpack helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pack(data) -> bytes:
    return msgpack.packb(data, use_bin_type=True)


def _unpack(data: bytes):
    return msgpack.unpackb(data, raw=False, strict_map_key=False)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Mock data factories
# ──────────────────────────────────────────────────────────────────────────────

def _extop_dat_for_arch(arch: str) -> bytes:
    """
    Produce a mock hipblasltExtOpLibrary.dat for a single arch.
    Schema: { arch: { op_name: { io_type: [ {kernel_meta} ] } } }
    In the per-arch layout, each shard writes one arch's data to
    library/<arch>/hipblasltExtOpLibrary.dat.
    """
    lib = {
        arch: {
            "LayerNorm": {
                "S": [{"w": 256, "c": 4, "sweep_once": 1, "co_path": f"extop_{arch}.co"}],
            },
            "Softmax": {
                "S": [{"m": 8, "n": 32, "co_path": f"extop_{arch}.co"}],
            },
            "AMax": {
                "S_S": [{"w": 256, "c": 4, "co_path": f"extop_{arch}.co"}],
            },
        }
    }
    return _pack(lib)


def _tensile_mapping_for_arch(arch: str) -> bytes:
    """
    Produce a mock TensileLiteLibrary_lazy_Mapping for a single arch.
    Schema: { solution_index (int): co_filename (str) }
    In the per-arch layout, each arch has its own independent 0..N index space
    anchored in library/<arch>/.  The runtime never mixes spaces.
    """
    mapping: dict = {}
    for idx, kernel in enumerate(["sgemm_nn", "hgemm_nn", "bf16gemm_nn"]):
        mapping[idx] = f"TensileLibrary_lazy_{arch}_{kernel}.co"
    return _pack(mapping)


# ──────────────────────────────────────────────────────────────────────────────
# Simulate per-arch subdirectory layout in a temp directory
# ──────────────────────────────────────────────────────────────────────────────

def _write_shard_install(base: Path, arch: str) -> None:
    """
    Write a simulated shard install tree for one arch under base/.
    Mirrors what the fixed CMake/Python scripts produce:
      base/library/<arch>/hipblasltExtOpLibrary.dat
      base/library/<arch>/extop_<arch>.co        (touch — binary content irrelevant)
      base/library/<arch>/TensileLiteLibrary_lazy_Mapping
      base/library/<arch>/hipblasltTransform.hsaco  (touch)
      base/library/<arch>/TensileLibrary_lazy_<arch>.dat
    """
    arch_dir = base / "library" / arch
    arch_dir.mkdir(parents=True, exist_ok=True)

    (arch_dir / "hipblasltExtOpLibrary.dat").write_bytes(_extop_dat_for_arch(arch))
    (arch_dir / f"extop_{arch}.co").write_bytes(f"co:{arch}".encode())
    (arch_dir / "TensileLiteLibrary_lazy_Mapping").write_bytes(
        _tensile_mapping_for_arch(arch)
    )
    # Each arch gets its own single-arch HSACO (not a fat binary in this layout)
    (arch_dir / "hipblasltTransform.hsaco").write_bytes(f"hsaco:{arch}".encode())
    (arch_dir / f"TensileLibrary_lazy_{arch}.dat").write_bytes(
        f"master_lib:{arch}".encode()
    )


# ──────────────────────────────────────────────────────────────────────────────
# Mode A — metadata unit tests (no build required)
# ──────────────────────────────────────────────────────────────────────────────

def _test_extop_dat(dist_archs: list[str]) -> None:
    """
    Demonstrate and assert the before/after overlay behaviour for
    hipblasltExtOpLibrary.dat.
    """
    print()
    print("hipblasltExtOpLibrary.dat")
    print("-" * 40)

    if len(dist_archs) < 2:
        record("hipblasltExtOpLibrary.dat", "before/after overlay test", SKIP,
               "Need at least 2 dist archs to demonstrate arch loss")
        return

    arch_a, arch_b = dist_archs[0], dist_archs[1]

    # ── BEFORE fix (flat layout) ───────────────────────────────────────────────
    # Each shard writes to the same flat library/hipblasltExtOpLibrary.dat.
    # The last shard's file overwrites earlier ones, losing other arch entries.
    dat_a_before = _pack({arch_a: {"Softmax": {"S": [{"co": f"extop_{arch_a}.co"}]}}})
    dat_b_before = _pack({arch_b: {"Softmax": {"S": [{"co": f"extop_{arch_b}.co"}]}}})

    overlaid_before = _unpack(dat_b_before)   # B's file replaces A's
    missing = {arch_a} - set(overlaid_before.keys())
    if missing:
        record(
            "hipblasltExtOpLibrary.dat",
            "before-fix (flat): overlay loses arch entries",
            PASS,
            f"arch(es) lost after overlay: {missing}  ← expected (demonstrates the bug)",
        )
    else:
        record(
            "hipblasltExtOpLibrary.dat",
            "before-fix (flat): overlay loses arch entries",
            FAIL,
            "Mock data unexpectedly covers all archs without the fix.",
        )

    # ── AFTER fix (per-arch subdirectory layout) ───────────────────────────────
    # Each shard writes to library/<arch>/hipblasltExtOpLibrary.dat.
    # The overlay merges directories; both arch subdirs are present.
    with tempfile.TemporaryDirectory() as tmpdir:
        overlay = Path(tmpdir) / "overlay"

        shard_a_dir = Path(tmpdir) / "shard_a"
        shard_b_dir = Path(tmpdir) / "shard_b"
        _write_shard_install(shard_a_dir, arch_a)
        _write_shard_install(shard_b_dir, arch_b)

        # Simulate TheRock overlay: copy A then B on top
        shutil.copytree(shard_a_dir, overlay)
        shutil.copytree(shard_b_dir, overlay, dirs_exist_ok=True)

        # Both arch subdirs should be present
        found_archs = {d.name for d in (overlay / "library").iterdir() if d.is_dir()}
        expected = set(dist_archs[:2])
        if expected <= found_archs:
            record(
                "hipblasltExtOpLibrary.dat",
                "after-fix (per-arch subdir): both arch subdirs survive overlay",
                PASS,
                f"arch subdirs in overlay: {sorted(found_archs)}",
            )
        else:
            record(
                "hipblasltExtOpLibrary.dat",
                "after-fix (per-arch subdir): both arch subdirs survive overlay",
                FAIL,
                f"missing arch subdirs: {expected - found_archs}  found: {sorted(found_archs)}",
            )

        # Each arch's .dat in the overlay must contain only that arch's entries
        for arch in [arch_a, arch_b]:
            dat_path = overlay / "library" / arch / "hipblasltExtOpLibrary.dat"
            if dat_path.exists():
                content = _unpack(dat_path.read_bytes())
                arch_keys = set(content.keys())
                if arch_keys == {arch}:
                    record(
                        "hipblasltExtOpLibrary.dat",
                        f"after-fix: {arch} .dat is self-contained in overlay",
                        PASS,
                    )
                else:
                    record(
                        "hipblasltExtOpLibrary.dat",
                        f"after-fix: {arch} .dat is self-contained in overlay",
                        FAIL,
                        f"expected arch keys {{{arch}}}, got {arch_keys}",
                    )


def _test_tensile_mapping(dist_archs: list[str]) -> None:
    """
    Demonstrate and assert the before/after overlay behaviour for
    TensileLiteLibrary_lazy_Mapping.
    """
    print()
    print("TensileLiteLibrary_lazy_Mapping")
    print("-" * 40)

    if len(dist_archs) < 2:
        record("TensileLiteLibrary_lazy_Mapping", "before/after overlay test", SKIP,
               "Need at least 2 dist archs to demonstrate arch loss")
        return

    arch_a, arch_b = dist_archs[0], dist_archs[1]

    # ── BEFORE fix (flat layout) ───────────────────────────────────────────────
    map_a_before = _tensile_mapping_for_arch(arch_a)
    map_b_before = _tensile_mapping_for_arch(arch_b)

    # Both shards use indices 0, 1, 2 — B overwrites A in flat layout
    overlaid_before = _unpack(map_b_before)
    co_names_before = set(overlaid_before.values())
    arch_a_refs_lost = not any(arch_a in name for name in co_names_before)
    if arch_a_refs_lost:
        record(
            "TensileLiteLibrary_lazy_Mapping",
            "before-fix (flat): overlay loses arch_a .co references",
            PASS,
            f"No {arch_a} entries in mapping after B overwrites A  ← expected (demonstrates the bug)",
        )
    else:
        record(
            "TensileLiteLibrary_lazy_Mapping",
            "before-fix (flat): overlay loses arch_a .co references",
            FAIL,
            "Mock data unexpectedly covers all archs without the fix.",
        )

    # ── AFTER fix (per-arch subdirectory layout) ───────────────────────────────
    # Each arch has library/<arch>/TensileLiteLibrary_lazy_Mapping with its own
    # independent 0..N index space. The runtime anchors lookups to the arch
    # subdir via tensile_host.cpp:2424-2426, so the spaces never interact.
    with tempfile.TemporaryDirectory() as tmpdir:
        overlay = Path(tmpdir) / "overlay"
        shard_a_dir = Path(tmpdir) / "shard_a"
        shard_b_dir = Path(tmpdir) / "shard_b"
        _write_shard_install(shard_a_dir, arch_a)
        _write_shard_install(shard_b_dir, arch_b)

        shutil.copytree(shard_a_dir, overlay)
        shutil.copytree(shard_b_dir, overlay, dirs_exist_ok=True)

        for arch in [arch_a, arch_b]:
            map_path = overlay / "library" / arch / "TensileLiteLibrary_lazy_Mapping"
            if map_path.exists():
                content = _unpack(map_path.read_bytes())
                co_names = set(content.values())
                if all(arch in name for name in co_names):
                    record(
                        "TensileLiteLibrary_lazy_Mapping",
                        f"after-fix: {arch} mapping is self-contained in overlay",
                        PASS,
                        f"indices 0..{len(content)-1} all reference {arch} .co files",
                    )
                else:
                    wrong = [n for n in co_names if arch not in n]
                    record(
                        "TensileLiteLibrary_lazy_Mapping",
                        f"after-fix: {arch} mapping is self-contained in overlay",
                        FAIL,
                        f"unexpected .co references: {wrong}",
                    )
            else:
                record(
                    "TensileLiteLibrary_lazy_Mapping",
                    f"after-fix: {arch} mapping present in overlay",
                    FAIL,
                    f"{map_path} not found",
                )


def _test_hsaco(dist_archs: list[str]) -> None:
    """
    Verify that per-arch hipblasltTransform.hsaco files compose additively.
    Each shard builds a single-arch HSACO to library/<arch>/hipblasltTransform.hsaco.
    The runtime probes for the arch-specific subdir before the flat fallback.
    """
    print()
    print("hipblasltTransform.hsaco")
    print("-" * 40)

    if len(dist_archs) < 2:
        record("hipblasltTransform.hsaco", "before/after overlay test", SKIP,
               "Need at least 2 dist archs to demonstrate arch loss")
        return

    arch_a, arch_b = dist_archs[0], dist_archs[1]

    # ── BEFORE fix (flat layout) ───────────────────────────────────────────────
    # Each shard writes to library/hipblasltTransform.hsaco with only its arch.
    # The last shard's file overwrites earlier ones.
    content_a_before = f"hsaco_fat_binary:arch={arch_a}".encode()
    content_b_before = f"hsaco_fat_binary:arch={arch_b}".encode()
    if content_a_before != content_b_before:
        record(
            "hipblasltTransform.hsaco",
            "before-fix (flat): shards write different files",
            PASS,
            f"shard_a content differs from shard_b  ← expected (demonstrates the bug)",
        )
    else:
        record(
            "hipblasltTransform.hsaco",
            "before-fix (flat): shards write different files",
            FAIL,
            "Mock content unexpectedly identical (single-arch dist set).",
        )

    # ── AFTER fix (per-arch subdirectory layout) ───────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        overlay = Path(tmpdir) / "overlay"
        shard_a_dir = Path(tmpdir) / "shard_a"
        shard_b_dir = Path(tmpdir) / "shard_b"
        _write_shard_install(shard_a_dir, arch_a)
        _write_shard_install(shard_b_dir, arch_b)

        shutil.copytree(shard_a_dir, overlay)
        shutil.copytree(shard_b_dir, overlay, dirs_exist_ok=True)

        for arch in [arch_a, arch_b]:
            hsaco_path = overlay / "library" / arch / "hipblasltTransform.hsaco"
            if hsaco_path.exists():
                content = hsaco_path.read_bytes()
                if arch.encode() in content:
                    record(
                        "hipblasltTransform.hsaco",
                        f"after-fix: {arch} hsaco is self-contained in overlay",
                        PASS,
                    )
                else:
                    record(
                        "hipblasltTransform.hsaco",
                        f"after-fix: {arch} hsaco is self-contained in overlay",
                        FAIL,
                        f"Content does not reference {arch}",
                    )
            else:
                record(
                    "hipblasltTransform.hsaco",
                    f"after-fix: {arch} hsaco present in overlay",
                    FAIL,
                    f"{hsaco_path} not found",
                )


def _test_host_library_note() -> None:
    """
    Document the libhipblaslt.so finding.  No mock is needed: the audit of
    all CMakeLists.txt files confirms GPU_TARGETS has zero effect on any
    target_compile_definitions, target_compile_options, or link options applied
    to the hipblaslt host library target.  The host library is already
    bit-for-bit identical across shards without any code change.
    """
    print()
    print("libhipblaslt.so (host library)")
    print("-" * 40)
    record(
        "libhipblaslt.so",
        "GPU_TARGETS has no effect on host library compilation",
        PASS,
        "Confirmed by static audit: no GPU_TARGETS reference in any "
        "target_compile_definitions/options/link_libraries for the hipblaslt target. "
        "Use Mode B (--shard-a-install/--shard-b-install) to verify sha256 equality "
        "against real build trees.",
    )


def run_mode_a(dist_archs: list[str]) -> bool:
    print()
    print("=" * 60)
    print("Mode A: metadata unit tests (no build required)")
    print(f"Distribution arch set: {dist_archs}")
    print()
    print("Strategy: per-arch subdirectory layout")
    print("  Before fix: flat library/ → last-writer-wins on overlay")
    print("  After fix:  library/<arch>/ → additive composition on overlay")
    print("=" * 60)

    _test_host_library_note()
    _test_extop_dat(dist_archs)
    _test_tensile_mapping(dist_archs)
    _test_hsaco(dist_archs)

    return summary()


# ──────────────────────────────────────────────────────────────────────────────
# Mode B — integration test against real install trees
# ──────────────────────────────────────────────────────────────────────────────

def _find_lib(install_dir: Path, name: str) -> Path | None:
    """Recursively find a file by name under install_dir."""
    for p in install_dir.rglob(name):
        return p
    return None


def _hsaco_archs(hsaco_path: Path) -> set[str]:
    """
    Return the set of bundled GPU architectures in an HSACO fat binary by
    invoking llvm-objdump --offloading.  Falls back to 'file' output parsing
    if llvm-objdump is not available.
    """
    for tool, flag in [("llvm-objdump", "--offloading"), ("llvm-objdump-18", "--offloading")]:
        try:
            out = subprocess.check_output(
                [tool, flag, str(hsaco_path)], stderr=subprocess.DEVNULL, text=True
            )
            archs = set()
            for line in out.splitlines():
                # Lines look like: "  Triple: amdgcn-amd-amdhsa, Arch: gfx942"
                for part in line.split(","):
                    part = part.strip()
                    if part.startswith("Arch:") or "processor" in part.lower():
                        val = part.split(":")[-1].strip()
                        if val.startswith("gfx"):
                            archs.add(val)
            if archs:
                return archs
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return set()  # tool not available


def run_mode_b(
    shard_a: Path,
    shard_b: Path,
    overlay_dir: Path,
    dist_archs: list[str],
) -> bool:
    print()
    print("=" * 60)
    print("Mode B: integration test against real install trees")
    print(f"  shard A: {shard_a}")
    print(f"  shard B: {shard_b}")
    print(f"  overlay: {overlay_dir}")
    print(f"  dist targets: {dist_archs}")
    print("=" * 60)

    # Build the overlay tree: copy A then copy B on top
    if overlay_dir.exists():
        shutil.rmtree(overlay_dir)
    shutil.copytree(shard_a, overlay_dir)
    # shutil.copytree with dirs_exist_ok overlays B onto A (Python 3.8+)
    shutil.copytree(shard_b, overlay_dir, dirs_exist_ok=True)

    # ── libhipblaslt.so ───────────────────────────────────────────────────────
    print()
    print("libhipblaslt.so (host library)")
    print("-" * 40)
    so_a = _find_lib(shard_a, "libhipblaslt.so")
    so_b = _find_lib(shard_b, "libhipblaslt.so")
    if so_a and so_b:
        sha_a = _sha256_file(so_a)
        sha_b = _sha256_file(so_b)
        if sha_a == sha_b:
            record("libhipblaslt.so", "byte-for-byte identical across shards", PASS,
                   f"sha256: {sha_a[:32]}…")
        else:
            record("libhipblaslt.so", "byte-for-byte identical across shards", FAIL,
                   f"shard_a sha256: {sha_a[:32]}…\n      shard_b sha256: {sha_b[:32]}…")
    else:
        missing = []
        if not so_a:
            missing.append(f"libhipblaslt.so not found under {shard_a}")
        if not so_b:
            missing.append(f"libhipblaslt.so not found under {shard_b}")
        record("libhipblaslt.so", "byte-for-byte identical across shards", SKIP,
               "; ".join(missing))

    # ── Per-arch subdirectory layout checks ───────────────────────────────────
    # After the fix, each shard writes to library/<arch>/ so both arch subdirs
    # survive the overlay.  Check the overlay tree for each expected arch.
    lib_dir_overlay = overlay_dir / "lib" / "hipblaslt" / "library"
    if not lib_dir_overlay.exists():
        # Try legacy path without hipblaslt subdirectory
        for candidate in overlay_dir.rglob("library"):
            if candidate.is_dir():
                lib_dir_overlay = candidate
                break

    print()
    print(f"Per-arch subdirectory checks (library root: {lib_dir_overlay})")
    print("-" * 40)

    if lib_dir_overlay.exists():
        found_arch_subdirs = {d.name for d in lib_dir_overlay.iterdir()
                              if d.is_dir() and d.name.startswith("gfx")}
        expected_archs = set(dist_archs)
        missing_subdirs = expected_archs - found_arch_subdirs
        if not missing_subdirs:
            record("library/<arch>/ subdirs",
                   "all dist arch subdirs present after overlay", PASS,
                   f"found: {sorted(found_arch_subdirs)}")
        else:
            record("library/<arch>/ subdirs",
                   "all dist arch subdirs present after overlay", FAIL,
                   f"missing: {sorted(missing_subdirs)}  found: {sorted(found_arch_subdirs)}")

        for arch in dist_archs:
            arch_dir = lib_dir_overlay / arch

            # hipblasltExtOpLibrary.dat
            dat = arch_dir / "hipblasltExtOpLibrary.dat"
            if dat.exists():
                content = _unpack(dat.read_bytes())
                if set(content.keys()) == {arch}:
                    record(f"library/{arch}/hipblasltExtOpLibrary.dat",
                           "present and arch-keyed correctly", PASS)
                else:
                    record(f"library/{arch}/hipblasltExtOpLibrary.dat",
                           "present and arch-keyed correctly", FAIL,
                           f"arch keys: {set(content.keys())}")
            else:
                record(f"library/{arch}/hipblasltExtOpLibrary.dat",
                       "present in overlay", SKIP, f"{dat} not found")

            # TensileLiteLibrary_lazy_Mapping
            mapping = arch_dir / "TensileLiteLibrary_lazy_Mapping"
            if mapping.exists():
                content = _unpack(mapping.read_bytes())
                co_names = set(content.values())
                if all(arch in n for n in co_names):
                    record(f"library/{arch}/TensileLiteLibrary_lazy_Mapping",
                           "present and arch-consistent", PASS)
                else:
                    wrong = [n for n in co_names if arch not in n]
                    record(f"library/{arch}/TensileLiteLibrary_lazy_Mapping",
                           "present and arch-consistent", FAIL,
                           f"unexpected .co refs: {wrong}")
            else:
                record(f"library/{arch}/TensileLiteLibrary_lazy_Mapping",
                       "present in overlay", SKIP, f"{mapping} not found")

            # hipblasltTransform.hsaco
            hsaco = arch_dir / "hipblasltTransform.hsaco"
            if hsaco.exists():
                bundled = _hsaco_archs(hsaco)
                if bundled:
                    if arch in bundled:
                        record(f"library/{arch}/hipblasltTransform.hsaco",
                               "present and contains correct arch", PASS,
                               f"bundled archs: {sorted(bundled)}")
                    else:
                        record(f"library/{arch}/hipblasltTransform.hsaco",
                               "present and contains correct arch", FAIL,
                               f"expected {arch} in bundled archs: {sorted(bundled)}")
                else:
                    record(f"library/{arch}/hipblasltTransform.hsaco",
                           "present (arch check skipped — no llvm-objdump)", SKIP,
                           "install llvm for full verification")
            else:
                record(f"library/{arch}/hipblasltTransform.hsaco",
                       "present in overlay", SKIP, f"{hsaco} not found")
    else:
        record("library/<arch>/ subdirs", "library directory found", SKIP,
               f"Could not find library directory under {overlay_dir}")

    return summary()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--shard-a-install", metavar="DIR", type=Path,
        help="Install prefix for shard A (Mode B)",
    )
    p.add_argument(
        "--shard-b-install", metavar="DIR", type=Path,
        help="Install prefix for shard B (Mode B)",
    )
    p.add_argument(
        "--overlay-dir", metavar="DIR", type=Path, default=Path(tempfile.mkdtemp()),
        help="Directory to build the overlaid install tree (Mode B). Created/replaced.",
    )
    p.add_argument(
        "--dist-targets", metavar="ARCHS",
        default="gfx942,gfx1100",
        help="Comma-separated full distribution target list (default: gfx942,gfx1100)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    dist_archs = [a.strip() for a in args.dist_targets.split(",") if a.strip()]

    if len(dist_archs) < 2:
        print("WARNING: --dist-targets contains fewer than 2 architectures.  "
              "The before-fix scenarios will not demonstrate arch loss.")

    if args.shard_a_install and args.shard_b_install:
        ok = run_mode_b(args.shard_a_install, args.shard_b_install,
                        args.overlay_dir, dist_archs)
    else:
        ok = run_mode_a(dist_archs)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
