#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Offline measurements and integrity checks for the ASAN transport experiment.

Network requests live in the workflow/composite action. Private curl configs and
response headers stay under work/, outside the uploaded results/ directory.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import time
import zipfile


ARTIFACT_ID = 13437992957
ARTIFACT_SIZE = 1903063180
ARTIFACT_SHA256 = "ff26ec74c7164bb38fea58789df690ea2828493fa4e87b6b43e718d643ab4017"
MEMBERS = ("hipblaslt-dist.tar.gz", "hipblaslt-stage.tar.gz")


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def checked_archive(path: Path, size: int, digest: str):
    if path.stat().st_size != size:
        raise ValueError("Archive length mismatch")
    if sha256(path) != digest:
        raise ValueError("Archive SHA-256 mismatch")


def inventory(root: Path) -> dict:
    entries: dict[str, dict] = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        entry = {"mode": stat.S_IMODE(info.st_mode)}
        if path.is_symlink():
            entry.update(type="symlink", target=os.readlink(path))
        elif path.is_file():
            entry.update(type="file", size=info.st_size, sha256=sha256(path))
        elif path.is_dir():
            entry.update(type="directory")
        else:
            raise ValueError(f"Unexpected file type: {path.relative_to(root)}")
        entries[path.relative_to(root).as_posix()] = entry
    return entries


def compare_inventory(expected: dict, actual: dict):
    differences = sorted(
        k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k)
    )
    if differences:
        raise ValueError(
            f"Tree mismatch ({len(differences)} paths): {differences[:20]}"
        )


def curl_config(path: Path, url: str, token: str | None = None):
    # JSON quoting is also valid for these curl config strings. Never print them.
    if not url.startswith("https://") or any(c in url for c in "\r\n"):
        raise ValueError("Expected an HTTPS URL without line breaks")
    lines = ["url = " + json.dumps(url)]
    if token:
        lines.append("header = " + json.dumps("Authorization: Bearer " + token))
        lines.append('header = "Accept: application/vnd.github+json"')
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o600)


def monitor(work: Path, result: Path):
    # Network/cgroup counters include other job activity; retain raw snapshots.
    paths = [
        "/proc/net/dev",
        "/sys/fs/cgroup/cpu.stat",
        "/sys/fs/cgroup/io.stat",
        "/sys/fs/cgroup/memory.current",
    ]
    with (result / "progress.jsonl").open("w") as out:
        for _ in range(720):
            if (work / "stop-monitor").exists():
                return
            archive = work / "payload.zip"
            sample = {
                "time": time.time(),
                "zip_bytes": archive.stat().st_size if archive.exists() else None,
            }
            for name in paths:
                path = Path(name)
                if path.is_file():
                    sample[name] = path.read_text()
            out.write(json.dumps(sample) + "\n")
            out.flush()
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=[
            "init",
            "begin",
            "monitor",
            "request",
            "redirect",
            "finish",
            "verify",
            "reference",
            "summary",
        ],
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", default="seed", choices=["seed", "A", "B", "C"])
    parser.add_argument("--source", choices=["github", "s3"])
    parser.add_argument("--status", default="success")
    parser.add_argument("--seed", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    work = root / "work" / args.arm
    result = root / "results" / args.arm
    state_path = result / "result.json"

    if args.mode == "reference":
        print("reference_sha256=" + sha256(root / "reference.json"))
        return

    if args.mode == "init":
        root.mkdir(parents=True, exist_ok=False)
        (root / "results").mkdir()
        info = {
            "started_at": time.time(),
            "artifact_id": ARTIFACT_ID,
            "artifact_size": ARTIFACT_SIZE,
            "artifact_sha256": ARTIFACT_SHA256,
            "kernel": platform.platform(),
            "python": sys.version,
        }
        for name in [
            "GITHUB_SHA",
            "GITHUB_RUN_ID",
            "GITHUB_RUN_ATTEMPT",
            "RUNNER_NAME",
            "ORDER",
        ]:
            info[name] = os.environ.get(name)
        for name in [
            "/proc/meminfo",
            "/proc/self/mountinfo",
            "/sys/fs/cgroup/cpu.max",
            "/sys/fs/cgroup/memory.max",
        ]:
            path = Path(name)
            if path.is_file():
                info[name] = path.read_text()
        info["proxy_variables_present"] = [
            k
            for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"]
            if os.environ.get(k)
        ]
        write_json(root / "results" / "environment.json", info)
        return
    if args.mode == "summary":
        states = [
            json.loads(p.read_text())
            for p in sorted((root / "results").glob("*/result.json"))
        ]
        summary = {"finished_at": time.time(), "arms": states}
        write_json(root / "results" / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        if {s["arm"] for s in states} != {"A", "B", "C"} or any(
            s.get("verified") is not True for s in states
        ):
            raise SystemExit(1)
        return
    if args.mode == "begin":
        work.mkdir(parents=True, exist_ok=False)
        result.mkdir(parents=True, exist_ok=False)
        (work / "trees").mkdir()
        write_json(
            state_path,
            {
                "arm": args.arm,
                "start": time.time(),
                "monotonic_start": time.monotonic(),
            },
        )
        subprocess.Popen(
            [
                sys.executable,
                __file__,
                "monitor",
                "--root",
                str(root),
                "--arm",
                args.arm,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return
    if args.mode == "monitor":
        monitor(work, result)
        return
    if args.mode == "request":
        if args.source == "github":
            curl_config(
                work / "api.curl",
                f"https://api.github.com/repos/ROCm/rocm-libraries/actions/artifacts/{ARTIFACT_ID}/zip",
                os.environ["GH_TOKEN"],
            )
        else:
            curl_config(work / "blob.curl", os.environ["S3_URL"] + "/payload.zip")
        return
    if args.mode == "redirect":
        headers = (work / "headers").read_text()
        locations = re.findall(r"^location:\s*(.+)$", headers, flags=re.I | re.M)
        if not locations or not re.search(r"^HTTP/\S+ 302\b", headers, flags=re.M):
            raise ValueError("GitHub did not return an artifact redirect")
        curl_config(work / "blob.curl", locations[-1].strip())
        (work / "api.curl").unlink()
        (work / "headers").unlink()
        return

    state = json.loads(state_path.read_text())
    if args.mode == "finish":
        (work / "stop-monitor").touch()
        state.update(
            download_status=args.status,
            download_seconds=time.monotonic() - state["monotonic_start"],
        )
        write_json(state_path, state)
        return

    # Verification is deliberately separate from transfer. Write timings even on failure.
    timings: dict[str, float] = {}
    phase_start = time.monotonic()

    def phase(name: str):
        nonlocal phase_start
        now = time.monotonic()
        timings[name] = now - phase_start
        phase_start = now

    try:
        if state.get("download_status") != "success":
            raise ValueError("Download failed; partial files are not usable")
        if args.arm != "A":
            checked_archive(work / "payload.zip", ARTIFACT_SIZE, ARTIFACT_SHA256)
            phase("zip_checksum")
            with zipfile.ZipFile(work / "payload.zip") as zf:
                if sorted(zf.namelist()) != sorted(MEMBERS):
                    raise ValueError("Unexpected ZIP members")
                for member in MEMBERS:
                    zf.extract(member, work / "trees")
            phase("zip_extract")
        inner = {name: sha256(work / "trees" / name) for name in MEMBERS}
        if not args.seed:
            if sha256(root / "reference.json") != os.environ["REFERENCE_SHA256"]:
                raise ValueError("Reference inventory SHA-256 mismatch")
            expected = json.loads((root / "reference.json").read_text())
            if inner != expected["inner_sha256"]:
                raise ValueError("Inner tarball SHA-256 mismatch")
        phase("inner_checksums")
        tree = work / "installed"
        tree.mkdir()
        for member in MEMBERS:
            subprocess.run(
                ["tar", "-C", str(tree), "-xzf", str(work / "trees" / member)],
                check=True,
            )
            phase("extract_" + member)
        files = inventory(tree)
        if not args.seed:
            compare_inventory(expected["files"], files)
        for name in ["bin/hipblaslt-test", "lib/llvm/bin/llvm-symbolizer"]:
            if not os.access(tree / name, os.X_OK):
                raise ValueError(f"Missing executable: {name}")
        runtimes = sorted(tree.glob("lib/llvm/lib/clang/**/libclang_rt.asan*.so"))
        if not runtimes:
            raise ValueError("Missing ASAN runtime")
        phase("tree_validation")
        if args.seed:
            write_json(
                root / "reference.json",
                {
                    "artifact_sha256": ARTIFACT_SHA256,
                    "inner_sha256": inner,
                    "files": files,
                },
            )
        else:
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = f"{tree / 'lib'}:{runtimes[0].parent}"
            env["ASAN_SYMBOLIZER_PATH"] = str(tree / "lib/llvm/bin/llvm-symbolizer")
            with (result / "test-discovery.txt").open("w") as output:
                subprocess.run(
                    [str(tree / "bin/hipblaslt-test"), "--gtest_list_tests"],
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=120,
                )
            phase("test_discovery")
        state["verified"] = True
    except Exception as exc:
        state.update(verified=False, error=str(exc))
        raise
    finally:
        state.update(
            timings=timings,
            ready_seconds=time.monotonic() - state["monotonic_start"],
            finished_at=time.time(),
        )
        write_json(state_path, state)


if __name__ == "__main__":
    main()
