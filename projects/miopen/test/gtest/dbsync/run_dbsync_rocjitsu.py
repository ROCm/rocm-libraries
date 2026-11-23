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
# run (topology untouched). db_sync.cpp's own CU-match skip (GetSysDbSelectionCu) picks the one
# StaticFDBSync param matching each CU (gfx942 304 -> gfx942130, 228 -> gfx942e4 / MI300A,
# gfx950 256 -> gfx950100, gfx90a 104 -> gfx90a68) and skips the rest, so no per-run gtest filter
# is needed.
#
# Navi (gfx11xx/gfx12xx) differs in two ways. First, the DB basename comes from
# Handle::GetDbBasename's gfx11xx/gfx12xx branch, which uses GetMaxHardwareComputeUnits() ==
# GetMaxComputeUnits() * 2 (CUs per WGP) in hex. The "cus" values below are therefore the arch's
# PHYSICAL CU count, i.e. the number encoded in the shipped DB basename: gfx110060 = 0x60 = 96,
# gfx110136 = 0x36 = 54, gfx120020 = 0x20 = 32, gfx120140 = 0x40 = 64.
# Second, a family can span several shipped DBs (gfx120X-all covers gfx1200 and gfx1201), so
# FAMILY_MAP's value is a LIST of targets and each one gets its own rocjitsu run.
#
# A wrong "cus" value or a missing "retarget" can make gtest pass with nothing checked: every
# StaticFDBSync param skips when the device reports another arch, and a param with a nonzero
# num_cu (gfx90a, gfx942, gfx950, gfx1101, gfx1201) skips when the device's CU count selects
# another system DB. Only a num_cu == 0 param that runs against a missing DB file fails on its own
# (SetupPaths). So each run writes gtest's JSON report, and the runner fails unless exactly one
# StaticFDBSync param for the target arch ran and was not skipped.
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
ROCJITSU_REF = "26a2330c02ea5f1191e67d877fe5761c040040fd"  # 2026-10-08

# family -> [ {arch, rj_config, cus, [retarget]} ]. Each entry is one shipped SystemDB arch and
# each CU count in "cus" is one CU-corrected rocjitsu run. gfx94X-dcgpu covers BOTH MI300X
# (304 -> gfx942130) and MI300A (228 -> gfx942e4) with a single arch.
#
# "retarget": the upstream rocjitsu configs dir has no KMD config for gfx1101, gfx1102 or gfx1200,
# so we derive one from the same-generation sibling that does exist (gfx1100_w7900.json for RDNA3,
# gfx1201_r9700.json for RDNA4) by rewriting gfx_target_version. Each pair shares family_id, wave
# size and LDS size, and HIP only needs the target version to pick the right code object. Without
# the rewrite the device reports the sibling arch and every param skips, which
# check_one_param_ran turns into a failure.
FAMILY_MAP = {
    "gfx94X-dcgpu": [
        {"arch": "gfx942", "rj_config": "gfx942_cdna3_kmd.json", "cus": [304, 228]},
    ],
    "gfx950-dcgpu": [
        {"arch": "gfx950", "rj_config": "gfx950_mi355x_kmd.json", "cus": [256]},
    ],
    # TheRock's family string for gfx90a has no -dcgpu suffix. Only the 104-CU MI210 system DB
    # (gfx90a68) is validated here: the 110-CU MI250X GCD one (gfx90a6e) still has wrong workspace
    # sizes in its find-db (known issue in #14519) and fails StaticFDBSync. Add 110 once it is
    # corrected.
    "gfx90a": [
        {"arch": "gfx90a", "rj_config": "gfx90a_mi210_kmd.json", "cus": [104]},
    ],
    "gfx110X-all": [
        # gfx110060, Radeon Pro W7900 / RX 7900 XTX
        {"arch": "gfx1100", "rj_config": "gfx1100_w7900.json", "cus": [96]},
        # gfx110136, Radeon Pro V710. No upstream config, so derived from the gfx1100 one.
        {
            "arch": "gfx1101",
            "rj_config": "gfx1100_w7900.json",
            "retarget": True,
            "cus": [54],
        },
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

# Shipped Navi SystemDBs a family covers that this runner does not validate yet, with the reason
# main() prints for each. Move an arch into FAMILY_MAP once its DB passes StaticFDBSync.
# - gfx1102 (gfx110220, RX 7600 XT): the find-db still has wrong workspace sizes (known issue in
#   #14519, not yet corrected), so StaticFDBSync fails. Entry once corrected (no upstream config,
#   derived from the gfx1100 one): {"arch": "gfx1102", "rj_config": "gfx1100_w7900.json",
#   "retarget": True, "cus": [32]}.
# - gfx1151 (gfx115128, Strix Halo 40 CU): #14519 corrected the find-db, but StaticFDBSync still
#   reports wrong workspace sizes for GemmWrwUniversal and ConvWinoFuryRxS<2-3>. Entry once fixed:
#   "gfx1151": [{"arch": "gfx1151", "rj_config": "gfx1151.json", "cus": [40]}]. The upstream config
#   declares a 32 CU part, so the CU-correction is load-bearing there.
NOT_VALIDATED = {
    "gfx110X-all": {
        "gfx1102": "gfx110220 find-db has wrong workspace sizes (#14519 known issue)",
    },
    "gfx1151": {
        "gfx1151": "gfx115128 find-db still has wrong workspace sizes after #14519",
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
    """Install cmake/build-essential/libdrm-dev/git/ninja-build if missing.

    Uses sudo rather than assuming the container runs as root: TheRock's
    no_rocm_image_ubuntu24_04 (and its variants) run as an unprivileged
    `tester` user with passwordless sudo configured, and some CI runner
    pools don't honor a container's --user 0:0 override.
    """
    if (
        all(shutil.which(t) for t in ("cmake", "c++", "git", "ninja"))
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
            "ninja-build",
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
            "-G",
            "Ninja",
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


def check_one_param_ran(report: Path, arch: str, cu: int):
    """Exit with an error unless exactly one StaticFDBSync param ran unskipped, and it is arch's.

    gtest exits 0 when every selected test skips, so its exit code alone cannot tell a checked
    DB from a run that checked nothing (see the CU-match and arch skips in db_sync.cpp).
    """
    if not report.exists():
        sys.exit(
            f"::error::StaticFDBSync {arch} @ {cu} CU: no gtest report at {report}"
        )
    params = [
        test
        for suite in json.loads(report.read_text())["testsuites"]
        for test in suite["testsuite"]
        if test["name"].startswith("StaticFDBSync/")
    ]
    ran = [t for t in params if t["status"] == "RUN" and t["result"] == "COMPLETED"]
    if len(ran) != 1 or f'"{arch}"' not in ran[0].get("value_param", ""):
        sys.exit(
            f"::error::StaticFDBSync {arch} @ {cu} CU: expected exactly one {arch} param to "
            f"run unskipped, got {[(t['name'], t.get('value_param')) for t in ran]} "
            f"of {len(params)} params"
        )
    print(
        f"StaticFDBSync {arch} @ {cu} CU: checked {ran[0]['name']} {ran[0]['value_param']}",
        flush=True,
    )


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
    for arch, reason in NOT_VALIDATED.get(family, {}).items():
        print(f"Not validating {arch}: {reason}.", flush=True)
    targets = FAMILY_MAP.get(family)
    if not targets:
        # Not a rocjitsu-supported family, or none of its DBs pass yet. TheRock's `include_family`
        # should keep this from running, but skip cleanly if it ever does (skip is success, so it
        # never blocks).
        print(
            f"AMDGPU_FAMILIES='{family}' has no target to validate -- skipping dbsync.",
            flush=True,
        )
        return 0

    bin_dir = Path(os.environ.get("THEROCK_BIN_DIR", "./build/bin")).resolve()
    dist = Path(os.environ.get("OUTPUT_ARTIFACTS_DIR", "./build")).resolve()
    gtest = bin_dir / "miopen_gtest"
    if not gtest.exists():
        sys.exit(f"::error::miopen_gtest not found at {gtest}")

    # Sanity: every target arch's CK grouped-conv plugin must be in the dist. A family can ship
    # more than one (gfx120X-all -> gfx1200 and gfx1201), and a missing one would otherwise only
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
            report = work / f"gtest_{arch}_{cu}.json"
            report.unlink(missing_ok=True)
            run(
                [
                    str(rocjitsu_bin),
                    "--config",
                    str(rj_config),
                    "--",
                    str(gtest),
                    "--gtest_filter=*StaticFDBSync*",
                    "--gtest_color=no",
                    f"--gtest_output=json:{report}",
                ],
                env=env_base,
            )
            check_one_param_ran(report, arch, cu)
            print("::endgroup::", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
