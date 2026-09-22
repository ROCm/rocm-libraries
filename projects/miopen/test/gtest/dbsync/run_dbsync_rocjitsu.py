#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# GPU-free MIOpen dbsync (StaticFDBSync) runner, driven under the rocjitsu KMD interposer on a CPU
# runner. This is the entrypoint TheRock's `miopen-dbsync` test component invokes (see
# ROCm/TheRock build_tools/github_actions/fetch_test_configurations.py) -- the test logic lives here
# in rocm-libraries and ships in the MIOpen dist (installed to share/miopen/bin/), and TheRock pulls
# it via the `--miopen` artifact.
#
# dbsync is CPU-only (SKIP_KDB_TESTING, no kernel launch) EXCEPT that CK grouped-conv IsApplicable
# makes a HIP device call -- so rocjitsu must EMULATE THE MATCHING ARCH (a cdna3 config with gfx950
# code objects aborts with "invalid device function"). We fetch the per-family artifact and run
# under the matching rocjitsu config, once per CU count the arch ships a SystemDB for.
#
# Why CU-correction: CK's grouped-wrw auto-split-k reads the device's real CU count via
# hipGetDeviceProperties (bypassing MIOpen's Handle CU-override). rocjitsu's stock cdna configs
# declare a minimal (~4 CU) device, which makes that occupancy calc wrong and false-flags valid perf
# configs. We rewrite the config's device-block CU fields to the arch's real CU count before each
# run (topology untouched). db_sync.cpp's own CU-match skip (GetSysDbSelectionCu) then runs exactly
# the one StaticFDBSync param matching each CU (gfx942 304 -> gfx942130, 228 -> gfx942e4 / MI300A,
# gfx950 256 -> gfx950100) and skips the rest, so no per-run gtest filter is needed.
#
# Env (provided by TheRock's test_component.yml): AMDGPU_FAMILIES (required), THEROCK_BIN_DIR
# (default ./build/bin), OUTPUT_ARTIFACTS_DIR (default ./build), TEST_TYPE (quick/standard/...).

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Pinned rocjitsu source (ROCm/rocm-systems). Bump deliberately.
ROCJITSU_REPO = "https://github.com/ROCm/rocm-systems.git"
ROCJITSU_REF = "481982dfe23e181a302006747c561e7cf43e35ba"  # 2026-06-04

# family -> (arch, rocjitsu KMD config, [CU counts to validate]). The CU list is the arch's real
# per-SKU CU counts; each = a StaticFDBSync param and drives one CU-corrected run. gfx94X-dcgpu
# covers BOTH MI300X (304 -> gfx942130) and MI300A (228 -> gfx942e4). Extend only for arches that
# rocjitsu has a KMD config for.
FAMILY_MAP = {
    "gfx94X-dcgpu": {
        "arch": "gfx942",
        "rj_config": "amdgpu_cdna3_kmd.json",
        "cus": [304, 228],
    },
    "gfx950-dcgpu": {
        "arch": "gfx950",
        "rj_config": "amdgpu_cdna4_kmd.json",
        "cus": [256],
    },
}

# StaticFDBSync's default thread fan-out (min(hw_concurrency, 32)) can deadlock/stall under the
# rocjitsu KMD interposer on some CK builds (observed: gfx942 hangs at ~200 find-db lines, gfx950
# runs pathologically slowly). Cap it via MIOPEN_DBSYNC_MAX_THREADS; 1 = fully serial, which cannot
# hit a concurrency deadlock. The check is CPU-only and completes fine single-threaded.
DBSYNC_MAX_THREADS = 1


def run(cmd, **kw):
    print(f"+ {' '.join(map(str, cmd))}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def cu_correct_config(in_path: Path, cu_count: int, out_path: Path):
    """Rewrite the rocjitsu config's device-block CU fields to cu_count (topology untouched)."""
    config = json.loads(in_path.read_text())
    dev = config["vm"]["gpu"]["device"]
    simd_per_cu = dev.get("simd_per_cu", 4)
    dev["simd_count"] = cu_count * simd_per_cu
    dev["num_shader_engines"] = 1
    dev["num_shader_arrays_per_engine"] = 1
    dev["num_cu_per_sh"] = cu_count
    out_path.write_text(json.dumps(config, indent=2))
    print(
        f"CU-corrected {in_path.name} -> {out_path} "
        f"(cu={cu_count}, simd_count={dev['simd_count']}, num_cu_per_sh={cu_count})",
        flush=True,
    )


def ensure_build_tools():
    """Install cmake/build-essential/libdrm-dev/git if missing (needs root; container runs --user 0)."""
    if (
        all(shutil.which(t) for t in ("cmake", "c++", "git"))
        and Path("/usr/include/libdrm").exists()
    ):
        return
    run(["apt-get", "update"])
    run(
        [
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "cmake",
            "build-essential",
            "libdrm-dev",
            "git",
        ]
    )


def build_rocjitsu(workdir: Path):
    """Sparse-clone rocjitsu @ the pinned ref and build rocjitsu_kmd_shim.

    Returns (kmd_so_path, rocjitsu_source_dir)."""
    src = workdir / "rocm-systems"
    if not src.exists():
        run(["git", "init", str(src)])
        run(["git", "-C", str(src), "remote", "add", "origin", ROCJITSU_REPO])
        run(["git", "-C", str(src), "config", "core.sparseCheckout", "true"])
        (src / ".git" / "info" / "sparse-checkout").write_text("emulation/rocjitsu\n")
        run(["git", "-C", str(src), "fetch", "--depth", "1", "origin", ROCJITSU_REF])
        run(["git", "-C", str(src), "checkout", "FETCH_HEAD"])
    build = workdir / "rocjitsu-build"
    run(
        [
            "cmake",
            "-S",
            str(src / "emulation" / "rocjitsu"),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )
    run(
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "rocjitsu_kmd_shim",
            "-j",
            str(os.cpu_count() or 4),
        ]
    )
    kmd = build / "lib" / "rocjitsu" / "src" / "rocjitsu" / "kmd" / "librocjitsu_kmd.so"
    if not kmd.exists():
        sys.exit(f"::error::rocjitsu KMD shim not found at {kmd} after build")
    return kmd, src


def main():
    # dbsync is a heavy, specialist check (~15 min on gfx942); the `quick` tier is a fast sanity gate
    # for non-component (build/CI) changes and shouldn't pay for it. Component PRs (standard/
    # comprehensive) and nightlies (comprehensive) run it in full. TheRock sets TEST_TYPE on the test
    # step (test_component.yml). Skip == success, so it never blocks.
    if os.environ.get("TEST_TYPE") == "quick":
        print(
            "TEST_TYPE=quick: skipping GPU-free dbsync (covered by component PRs + nightlies).",
            flush=True,
        )
        return 0

    family = os.environ.get("AMDGPU_FAMILIES", "")
    entry = FAMILY_MAP.get(family)
    if entry is None:
        # Not a rocjitsu-supported family. TheRock's `include_family` should keep this from running,
        # but skip cleanly if it ever does (skip is success, so it never blocks).
        print(
            f"AMDGPU_FAMILIES='{family}' has no rocjitsu KMD config -- skipping dbsync.",
            flush=True,
        )
        return 0

    arch = entry["arch"]
    rj_config_name = entry["rj_config"]
    cus = entry["cus"]

    bin_dir = Path(os.environ.get("THEROCK_BIN_DIR", "./build/bin")).resolve()
    dist = Path(os.environ.get("OUTPUT_ARTIFACTS_DIR", "./build")).resolve()
    gtest = bin_dir / "miopen_gtest"
    if not gtest.exists():
        sys.exit(f"::error::miopen_gtest not found at {gtest}")

    # Sanity: the arch-specific CK grouped-conv plugin + this arch's SystemDBs must be in the dist.
    plugins = list(dist.rglob(f"libMIOpenCKGroupedConv_{arch}.so"))
    print(
        f"CK grouped-conv plugins: {[str(p) for p in dist.rglob('libMIOpenCKGroupedConv_*.so')]}",
        flush=True,
    )
    if not plugins:
        sys.exit(
            f"::error::libMIOpenCKGroupedConv_{arch}.so not present in the fetched dist"
        )

    work = Path("rocjitsu-work").resolve()
    work.mkdir(exist_ok=True)
    ensure_build_tools()
    kmd, rj_src = build_rocjitsu(work)
    configs_dir = rj_src / "emulation" / "rocjitsu" / "configs"

    env_base = os.environ.copy()
    env_base["ROCM_PATH"] = str(dist)
    env_base["LD_LIBRARY_PATH"] = (
        f"{dist / 'lib'}:{env_base.get('LD_LIBRARY_PATH', '')}"
    )
    env_base["LD_PRELOAD"] = str(kmd)
    env_base["MIOPEN_DBSYNC_MAX_THREADS"] = str(DBSYNC_MAX_THREADS)

    for cu in cus:
        print(f"::group::StaticFDBSync {arch} @ {cu} CU", flush=True)
        rj_config = work / f"rj_config_{cu}.json"
        cu_correct_config(configs_dir / rj_config_name, cu, rj_config)
        env = env_base.copy()
        env["RJ_CONFIG"] = str(rj_config)
        run([str(gtest), "--gtest_filter=*StaticFDBSync*", "--gtest_color=no"], env=env)
        print("::endgroup::", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
