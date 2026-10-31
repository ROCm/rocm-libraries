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
# Navi (gfx11xx/gfx12xx) differs in two ways. First, those StaticFDBSync params are declared with
# num_cu == 0, so the GetSysDbSelectionCu skip at db_sync.cpp does not apply to them and the DB
# basename comes from Handle::GetDbBasename's gfx11xx/gfx12xx branch, which uses
# GetMaxHardwareComputeUnits() == GetMaxComputeUnits() * 2 (CUs per WGP) in hex. The "cus" values
# below are therefore the arch's PHYSICAL CU count, i.e. the number encoded in the shipped DB
# basename: gfx110060 = 0x60 = 96, gfx110220 = 0x20 = 32, gfx115128 = 0x28 = 40,
# gfx120020 = 0x20 = 32, gfx120140 = 0x40 = 64. A wrong value does not silently validate the wrong
# file: SetupPaths asserts the basename exists.
# Second, a family can span several shipped DBs (gfx110X-all covers gfx1100 and gfx1102;
# gfx120X-all covers gfx1200 and gfx1201), so FAMILY_MAP's value is a LIST of targets and each one
# gets its own rocjitsu run.
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
ROCJITSU_REF = "b2099a6b8703c089134ad526253752d26bbbf43a"  # 2026-09-10

# family -> [ {arch, rj_config, cus, [retarget]} ]. Each entry is one shipped SystemDB arch and
# each CU count in "cus" is one CU-corrected rocjitsu run. gfx94X-dcgpu covers BOTH MI300X
# (304 -> gfx942130) and MI300A (228 -> gfx942e4) with a single arch.
#
# "retarget": the upstream rocjitsu configs dir has no KMD config for gfx1102 or gfx1200, so we
# derive one from the same-generation sibling that does exist (gfx1100_w7900.json for RDNA3,
# gfx1201_r9700.json for RDNA4) by rewriting gfx_target_version. Both pairs share family_id, wave
# size and LDS size, and HIP only needs the target version to pick the right code object -- which
# is the thing that must match, or the run aborts with "invalid device function".
FAMILY_MAP = {
    "gfx94X-dcgpu": [
        {"arch": "gfx942", "rj_config": "gfx942_cdna3_kmd.json", "cus": [304, 228]},
    ],
    "gfx950-dcgpu": [
        {"arch": "gfx950", "rj_config": "gfx950_mi355x_kmd.json", "cus": [256]},
    ],
    "gfx110X-all": [
        # gfx110060, Radeon Pro W7900 / RX 7900 XTX
        {"arch": "gfx1100", "rj_config": "gfx1100_w7900.json", "cus": [96]},
        # gfx110220, RX 7600 XT. No upstream config; derived from the gfx1100 one.
        {
            "arch": "gfx1102",
            "rj_config": "gfx1100_w7900.json",
            "retarget": True,
            "cus": [32],
        },
    ],
    "gfx1151": [
        # gfx115128, Strix Halo 40 CU. The upstream config declares a 32 CU part, so the
        # CU-correction below is load-bearing here, not just an occupancy fix-up.
        {"arch": "gfx1151", "rj_config": "gfx1151.json", "cus": [40]},
    ],
    "gfx120X-all": [
        # gfx120020, RX 9060 XT. No upstream config; derived from the gfx1201 one.
        {
            "arch": "gfx1200",
            "rj_config": "gfx1201_r9700.json",
            "retarget": True,
            "cus": [32],
        },
        # gfx120140, Radeon AI PRO R9700
        {"arch": "gfx1201", "rj_config": "gfx1201_r9700.json", "cus": [64]},
    ],
}

# StaticFDBSync's default thread fan-out (min(hw_concurrency, 32)) can deadlock/stall under the
# rocjitsu KMD interposer on some CK builds (observed: gfx942 hangs at ~200 find-db lines, gfx950
# runs pathologically slowly). Cap it via MIOPEN_DBSYNC_MAX_THREADS; 1 = fully serial, which cannot
# hit a concurrency deadlock. The check is CPU-only and completes fine single-threaded.
DBSYNC_MAX_THREADS = 1


def run(cmd, **kw):
    print(f"+ {' '.join(map(str, cmd))}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def gfx_target_version(arch: str) -> int:
    """'gfx1102' -> 110002. KFD encodes the target as major*10000 + minor*100 + step."""
    digits = arch[len("gfx") :]
    return int(digits[:-2]) * 10000 + int(digits[-2]) * 100 + int(digits[-1])


def write_config(
    in_path: Path, arch: str, cu_count: int, retarget: bool, out_path: Path
):
    """Emit a rocjitsu config for arch at cu_count CUs (topology untouched).

    CU-correction rewrites the device block's CU fields. Retargeting additionally rewrites
    gfx_target_version so HIP loads this arch's code objects; it is only used for arches the
    upstream configs dir has no entry for."""
    config = json.loads(in_path.read_text())
    dev = config["vm"]["gpu"]["device"]
    if retarget:
        dev["gfx_target_version"] = gfx_target_version(arch)
        dev["marketing_name"] = f"rocjitsu {arch} ({cu_count} CU)"
    simd_per_cu = dev.get("simd_per_cu", 4)
    dev["simd_count"] = cu_count * simd_per_cu
    dev["num_shader_engines"] = 1
    dev["num_shader_arrays_per_engine"] = 1
    dev["num_cu_per_sh"] = cu_count
    out_path.write_text(json.dumps(config, indent=2))
    print(
        f"config {in_path.name} -> {out_path} (arch={arch}, "
        f"gfx_target_version={dev['gfx_target_version']}, cu={cu_count}, "
        f"simd_count={dev['simd_count']}, num_cu_per_sh={cu_count})",
        flush=True,
    )


def ensure_build_tools():
    """Install cmake/build-essential/libdrm-dev/git if missing.

    Uses sudo rather than assuming the container runs as root: TheRock's
    no_rocm_image_ubuntu24_04 (and its variants) run as an unprivileged
    `tester` user with passwordless sudo configured, and some CI runner
    pools don't honor a container's --user 0:0 override.
    """
    if (
        all(shutil.which(t) for t in ("cmake", "c++", "git"))
        and Path("/usr/include/libdrm").exists()
    ):
        return
    run(["sudo", "apt-get", "update"])
    run(
        [
            "sudo",
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
    """Sparse-clone rocjitsu @ the pinned ref and build the rocjitsu CLI launcher.

    The launcher (not a raw LD_PRELOAD of librocjitsu.so) is the supported entry point: it stands
    up the simulated KFD and injects the interposer into the child process it execs.

    Returns (rocjitsu_bin_path, rocjitsu_source_dir)."""
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
            "rocjitsu_bin",
            "rocjitsu_shared",
            "-j",
            str(os.cpu_count() or 4),
        ]
    )
    rocjitsu_bin = build / "tools" / "rocjitsu" / "rocjitsu"
    if not rocjitsu_bin.exists():
        sys.exit(
            f"::error::rocjitsu CLI launcher not found at {rocjitsu_bin} after build"
        )
    return rocjitsu_bin, src


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
    targets = FAMILY_MAP.get(family)
    if not targets:
        # Not a rocjitsu-supported family. TheRock's `include_family` should keep this from running,
        # but skip cleanly if it ever does (skip is success, so it never blocks).
        print(
            f"AMDGPU_FAMILIES='{family}' has no rocjitsu KMD config -- skipping dbsync.",
            flush=True,
        )
        return 0

    bin_dir = Path(os.environ.get("THEROCK_BIN_DIR", "./build/bin")).resolve()
    dist = Path(os.environ.get("OUTPUT_ARTIFACTS_DIR", "./build")).resolve()
    gtest = bin_dir / "miopen_gtest"
    if not gtest.exists():
        sys.exit(f"::error::miopen_gtest not found at {gtest}")

    # Sanity: every target arch's CK grouped-conv plugin must be in the dist. A family can ship
    # more than one (gfx110X-all -> gfx1100 and gfx1102), and a missing one would otherwise only
    # surface as a confusing runtime failure inside the interposer.
    print(
        f"CK grouped-conv plugins: {[str(p) for p in dist.rglob('libMIOpenCKGroupedConv_*.so')]}",
        flush=True,
    )
    missing = [
        t["arch"]
        for t in targets
        if not list(dist.rglob(f"libMIOpenCKGroupedConv_{t['arch']}.so"))
    ]
    if missing:
        sys.exit(
            "::error::missing CK grouped-conv plugin(s) in the fetched dist: "
            + ", ".join(f"libMIOpenCKGroupedConv_{a}.so" for a in missing)
        )

    work = Path("rocjitsu-work").resolve()
    work.mkdir(exist_ok=True)
    ensure_build_tools()
    rocjitsu_bin, rj_src = build_rocjitsu(work)
    configs_dir = rj_src / "emulation" / "rocjitsu" / "configs"

    env_base = os.environ.copy()
    env_base["ROCM_PATH"] = str(dist)
    env_base["LD_LIBRARY_PATH"] = (
        f"{dist / 'lib'}:{env_base.get('LD_LIBRARY_PATH', '')}"
    )
    env_base["MIOPEN_DBSYNC_MAX_THREADS"] = str(DBSYNC_MAX_THREADS)

    for target in targets:
        arch = target["arch"]
        for cu in target["cus"]:
            print(f"::group::StaticFDBSync {arch} @ {cu} CU", flush=True)
            rj_config = work / f"rj_config_{arch}_{cu}.json"
            write_config(
                configs_dir / target["rj_config"],
                arch,
                cu,
                target.get("retarget", False),
                rj_config,
            )
            run(
                [
                    str(rocjitsu_bin),
                    "--config",
                    str(rj_config),
                    "--",
                    str(gtest),
                    "--gtest_filter=*StaticFDBSync*",
                    "--gtest_color=no",
                ],
                env=env_base,
            )
            print("::endgroup::", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
