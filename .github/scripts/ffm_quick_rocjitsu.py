#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run installed CTest entries under one rocJITsu process tree.

Walks the nightly's CTestTestfile.cmake files, selects tests by CTest label,
and launches each selected command as ``rocjitsu --config <gfx942> --
<command>``. A failing suite is recorded and the walk continues. Each suite
gets one hour. The installed CTest TIMEOUT is the GPU budget and is not used.

By default the label is ``^ffm-quick$`` so the existing hipRAND sweep keeps
working. Pass ``--label '^quick$'`` for the on-demand component workflow, and
``--component <dir>`` to run one installed directory under ``bin/``.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

EXCLUDE_PATTERN = re.compile(r"ex_gpu")
DEFAULT_LABEL = r"^ffm-quick$"
# One hour per suite. The installed CTest TIMEOUT is the GPU budget.
DEFAULT_TIMEOUT = 60 * 60


def tokenize_cmake_args(text: str) -> list[str]:
    """Split one CMake argument list, including quotes and [[brackets]]."""
    args = []
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        if text.startswith("[[", index):
            end = text.find("]]", index + 2)
            if end < 0:
                raise ValueError(f"unclosed bracket argument in: {text}")
            args.append(text[index + 2 : end])
            index = end + 2
            continue
        if text[index] == '"':
            index += 1
            chars = []
            while index < length and text[index] != '"':
                if text[index] == "\\" and index + 1 < length:
                    chars.append(text[index + 1])
                    index += 2
                    continue
                chars.append(text[index])
                index += 1
            index += 1
            args.append("".join(chars))
            continue
        start = index
        while index < length and not text[index].isspace():
            index += 1
        args.append(text[start:index])
    return args


def _call(line: str):
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    for name in ("add_test", "set_tests_properties"):
        prefix = name + "("
        if stripped.startswith(prefix) and stripped.endswith(")"):
            return name, stripped[len(prefix) : -1]
    return None


def parse_ctest_file(text: str) -> dict[str, dict]:
    """Return test name -> command, labels, timeout, skip regex, workdir."""
    tests = {}
    order = []
    for line in text.splitlines():
        parsed = _call(line)
        if parsed is None:
            continue
        name, body = parsed
        args = tokenize_cmake_args(body)
        if name == "add_test":
            if len(args) < 2:
                continue
            if args[0] == "NAME" and "COMMAND" in args:
                test_name = args[1]
                command = args[args.index("COMMAND") + 1 :]
            else:
                test_name = args[0]
                command = args[1:]
            if test_name not in tests:
                order.append(test_name)
                tests[test_name] = {
                    "name": test_name,
                    "command": command,
                    "labels": [],
                    "timeout": None,
                    "skip_regex": "",
                    "working_directory": "",
                }
            else:
                tests[test_name]["command"] = command
            continue
        if "PROPERTIES" not in args:
            continue
        split = args.index("PROPERTIES")
        names = args[:split]
        props = args[split + 1 :]
        updates = {}
        cursor = 0
        while cursor + 1 < len(props):
            key = props[cursor]
            value = props[cursor + 1]
            updates[key] = value
            cursor += 2
        for test_name in names:
            test = tests.setdefault(
                test_name,
                {
                    "name": test_name,
                    "command": [],
                    "labels": [],
                    "timeout": None,
                    "skip_regex": "",
                    "working_directory": "",
                },
            )
            if test_name not in order:
                order.append(test_name)
            if "LABELS" in updates:
                test["labels"] = updates["LABELS"].split(";")
            if "TIMEOUT" in updates:
                test["timeout"] = int(updates["TIMEOUT"])
            if "SKIP_REGULAR_EXPRESSION" in updates:
                test["skip_regex"] = updates["SKIP_REGULAR_EXPRESSION"]
            if "WORKING_DIRECTORY" in updates:
                test["working_directory"] = updates["WORKING_DIRECTORY"]
    return {name: tests[name] for name in order if tests[name]["command"]}


def label_pattern(label: str) -> re.Pattern[str]:
    """Compile a CTest label regex. Bare names become an exact match."""
    text = label.strip()
    if not text:
        text = DEFAULT_LABEL
    if text.startswith("^") or text.endswith("$"):
        return re.compile(text)
    return re.compile(rf"^{re.escape(text)}$")


def selects_label(test: dict, pattern: re.Pattern[str]) -> bool:
    labels = test["labels"]
    if any(EXCLUDE_PATTERN.search(label) for label in labels):
        return False
    return any(pattern.search(label) for label in labels)


def selects_ffm_quick(test: dict) -> bool:
    """Backward-compatible helper used by the existing unit tests."""
    return selects_label(test, label_pattern(DEFAULT_LABEL))


def timeout_seconds(_test: dict) -> int:
    """Return the rocJITsu kill time for one installed test.

    Every suite gets one hour. The parsed CTest TIMEOUT is the GPU budget and
    is ignored.
    """
    return DEFAULT_TIMEOUT


def parse_component_names(text: str) -> list[str]:
    """Split a comma-separated component filter. Empty means no filter."""
    return [part.strip() for part in text.split(",") if part.strip()]


SHARE_COMPONENT_ALIASES = {
    "hsa-amd-aqlprofile": "aqlprofile",
}
JPEG_QUICK_CASES = (
    ("jpeg-decode-fmt-native", []),
    ("jpeg-decode-fmt-yuv-planar", ["-fmt", "yuv_planar"]),
    ("jpeg-decode-fmt-y", ["-fmt", "y"]),
    ("jpeg-decode-fmt-rgb", ["-fmt", "rgb"]),
    ("jpeg-decode-fmt-rgb-planar", ["-fmt", "rgb_planar"]),
)
ROCDECODE_QUICK_CASES = (
    ("video_decodeRaw-HEVC", "AMD_driving_virtual_20-H265.265"),
    ("video_decodeRaw-AVC", "AMD_driving_virtual_20-H264.264"),
    ("video_decodeRaw-AV1", "AMD_driving_virtual_20-AV1.ivf"),
    ("video_decodeRaw-VP9", "AMD_driving_virtual_20-VP9.ivf"),
)


# rocprofiler-systems installs a generated CTest file: multi-line
# set_tests_properties, CMake variables, and setup/cleanup fixtures. It cannot
# be split into per-test commands, so ctest itself runs under rocJITsu. The
# exclusions mirror TheRock's test_rocprofiler_systems.py (AIPROFSYST-441).
CTEST_DRIVEN_COMPONENTS = ("rocprofiler-systems",)
SYSTEMS_EXCLUDED_TESTS = (
    "transferbench-sys-run|fork.*|openmp-target.*|roctx-sampling|"
    "roctx-runtime-instrument|jacobi-usm-sys-run|jacobi-roctx.*|jpeg-decode.*|"
    "matrix-exponential.*|scratch-memory.*|selective-region-region-1-filter.*|"
    "selective-region-region-2-and-3.*|"
    "selective-region-no-marker-region-1-filter.*|shmem-pingpong.*|video-decode.*"
)
SYSTEMS_EXCLUDED_LABELS = (
    "annotate|mpi|julia|attach|lulesh|network|overflow|thread_limit"
)


def component_name_for(rocm_root: Path, ctest_file: Path) -> str | None:
    """Name a CTest directory under bin/, share/ or libexec/.

    rocprofiler-compute installs its CTest payload under libexec/.
    """
    for root_name in ("bin", "share", "libexec"):
        root = rocm_root / root_name
        try:
            relative = ctest_file.parent.relative_to(root).as_posix()
        except ValueError:
            continue
        if root_name == "bin":
            return relative
        top = relative.split("/", 1)[0]
        return SHARE_COMPONENT_ALIASES.get(top, top)
    return None


def _sample_root() -> Path:
    # Absolute: each test runs with cwd set to the nightly source directory.
    return Path(os.environ.get("ROCJITSU_SAMPLE_ROOT", "rj-samples")).resolve()


def _quick_test(name: str, command: list[str]) -> dict:
    return {
        "name": name,
        "command": command,
        "labels": ["quick"],
        "timeout": None,
        "skip_regex": "",
        "working_directory": "",
    }


def source_quick_group(
    rocm_root: Path, name: str, label: str = DEFAULT_LABEL
) -> dict | None:
    """Quick cases whose binaries are built from the nightly sample sources.

    rocjpeg and rocdecode ship CMake sources under share/, not a prebuilt
    CTest binary. The workflow builds the sample before these commands run.
    """
    if name == "rocjpeg":
        source = rocm_root / "share" / "rocjpeg" / "samples" / "jpegDecode"
        images = rocm_root / "share" / "rocjpeg" / "images"
        if not source.is_dir() or not images.is_dir():
            return None
        binary = str(_sample_root() / "jpegDecode" / "jpegdecode")
        tests = [
            _quick_test(test_name, [binary, "-i", str(images), *extra])
            for test_name, extra in JPEG_QUICK_CASES
        ]
        return {"component": name, "directory": source, "tests": tests}
    if name == "aqlprofile":
        # The nightly ships no CTestTestfile for aqlprofile. Run its legacy
        # script directly from the install root, as test_aqlprofile.py does.
        script = rocm_root / "share" / "hsa-amd-aqlprofile" / "run_tests.sh"
        if not script.is_file():
            return None
        tests = [_quick_test("aqlprofile-legacy-tests", [str(script)])]
        return {"component": name, "directory": rocm_root, "tests": tests}
    if name == "rocprofiler-systems":
        # Serial on purpose: the tests share one profiling backend and rely on
        # CTest fixtures, so the whole tree runs through one ctest process.
        source = rocm_root / "share" / "rocprofiler-systems" / "tests"
        if not (source / "CTestTestfile.cmake").is_file():
            return None
        command = [
            "ctest",
            "--test-dir",
            str(source),
            "--output-on-failure",
            "-L",
            label,
            "--exclude-regex",
            SYSTEMS_EXCLUDED_TESTS,
            "--label-exclude",
            SYSTEMS_EXCLUDED_LABELS,
        ]
        tests = [_quick_test("rocprofiler-systems-ctest", command)]
        return {"component": name, "directory": source, "tests": tests}
    if name == "rocprofiler-sdk":
        # The nightly ships the SDK tests as a CMake project, not a prebuilt
        # CTest tree. The workflow configures and builds it into
        # rj-samples/rocprofiler-sdk-tests, then this runs the whole CTest
        # tree as one case, as test_rocprofiler_sdk.py does. The spm label is
        # excluded because those tests need a pinned runner and driver.
        source = rocm_root / "share" / "rocprofiler-sdk" / "tests"
        if not (source / "CMakeLists.txt").is_file():
            return None
        build = _sample_root() / "rocprofiler-sdk-tests"
        command = [
            "ctest",
            "--test-dir",
            str(build),
            "--parallel",
            "4",
            "--output-on-failure",
            "-LE",
            "spm",
        ]
        tests = [_quick_test("rocprofiler-sdk-ctest", command)]
        return {"component": name, "directory": source, "tests": tests}
    if name == "rocdecode":
        source = rocm_root / "share" / "rocdecode" / "samples" / "videoDecodeRaw"
        video = rocm_root / "share" / "rocdecode" / "video"
        if not source.is_dir() or not video.is_dir():
            return None
        binary = str(_sample_root() / "videoDecodeRaw" / "videodecoderaw")
        tests = [
            _quick_test(
                test_name,
                [binary, "-i", str(video / filename)],
            )
            for test_name, filename in ROCDECODE_QUICK_CASES
        ]
        return {"component": name, "directory": source, "tests": tests}
    return None


def discover(
    rocm_root: Path,
    label: str = DEFAULT_LABEL,
    component: str | None = None,
    components: list[str] | None = None,
) -> list[dict]:
    """Installed tests matching ``label``, fewest tests in a directory first.

    ``component`` selects one directory. ``components`` selects a set. When
    both are empty, every matching directory under ``bin/`` and ``share/`` is
    returned. Named rocjpeg and rocdecode requests also include the quick
    sample cases, which are built from share/ sources.
    """
    pattern = label_pattern(label)
    wanted: set[str] | None = None
    if component:
        wanted = {component}
    elif components:
        wanted = set(components)

    groups = []
    seen: set[str] = set()
    roots = [rocm_root / "bin", rocm_root / "share", rocm_root / "libexec"]
    ctest_files = []
    for root in roots:
        if root.is_dir():
            ctest_files.extend(sorted(root.rglob("CTestTestfile.cmake")))
    for ctest_file in ctest_files:
        name = component_name_for(rocm_root, ctest_file)
        if name is None or name in seen:
            continue
        if wanted is not None and name not in wanted:
            continue
        if name in CTEST_DRIVEN_COMPONENTS:
            continue
        parsed = list(
            parse_ctest_file(
                ctest_file.read_text(encoding="utf-8", errors="replace")
            ).values()
        )
        tests = [test for test in parsed if selects_label(test, pattern)]
        under_share = "share" in ctest_file.parts or "libexec" in ctest_file.parts
        if not tests and wanted is not None and under_share:
            tests = [
                test
                for test in parsed
                if not any(EXCLUDE_PATTERN.search(label) for label in test["labels"])
            ]
        if not tests:
            continue
        seen.add(name)
        groups.append(
            {
                "component": name,
                "directory": ctest_file.parent,
                "tests": tests,
            }
        )
    if wanted is not None:
        for name in sorted(wanted):
            if name in seen:
                continue
            group = source_quick_group(rocm_root, name, label)
            if group is None:
                continue
            seen.add(name)
            groups.append(group)
    groups.sort(key=lambda group: (len(group["tests"]), group["component"]))
    return groups


def missing_components(rocm_root: Path, label: str, names: list[str]) -> list[str]:
    """Names from ``names`` that have no installed tests for ``label``."""
    if not names:
        return []
    # Named lookups include the source-built rocjpeg/rocdecode cases.
    found = {
        group["component"]
        for group in discover(rocm_root, label=label, components=names)
    }
    return [name for name in names if name not in found]


def output_skips(text: str, skip_regex: str) -> bool:
    if not skip_regex:
        return False
    for pattern in skip_regex.split(";"):
        if not pattern:
            continue
        try:
            if re.search(pattern, text):
                return True
        except re.error:
            continue
    return False


def build_env(rocm_root: Path) -> dict:
    env = os.environ.copy()
    lib = str(rocm_root / "lib")
    bin_dir = str(rocm_root / "bin")
    env["ROCM_PATH"] = str(rocm_root)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    previous = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = lib if not previous else lib + os.pathsep + previous
    return env


PROFILER_COMPONENTS = ("rocprofiler-compute", "rocprofiler-sdk", "rocprofiler-systems")


def _prepend_path(env: dict, name: str, entries: list[Path]) -> None:
    existing = [str(entry) for entry in entries if entry.is_dir()]
    if not existing:
        return
    previous = env.get(name, "")
    env[name] = os.pathsep.join(existing + ([previous] if previous else []))


def component_env(rocm_root: Path, component: str, base: dict) -> dict:
    """Environment for one component, mirroring TheRock's profiler runners.

    Only the three profiler components get extras, so other components (and
    the hipRAND sweep) keep the plain ROCM_PATH/PATH/LD_LIBRARY_PATH setup.
    """
    if component not in PROFILER_COMPONENTS:
        return base
    env = dict(base)
    env["HIP_PATH"] = str(rocm_root)
    env["HIP_PLATFORM"] = "amd"
    env["ROCPROFILER_METRICS_PATH"] = str(rocm_root / "share" / "rocprofiler-sdk")
    libs = [rocm_root / "lib" / "rocm_sysdeps" / "lib"]
    llvm_lib = rocm_root / "lib" / "llvm" / "lib"
    if llvm_lib.is_dir():
        libs.extend(sorted(llvm_lib.glob("*-linux-gnu")))
    if component == "rocprofiler-systems":
        libs = [
            rocm_root / "lib" / "rocprofiler-systems",
            rocm_root / "share" / "rocprofiler-systems" / "examples" / "lib",
        ] + libs
        env["ROCPROFSYS_INSTALL_DIR"] = str(rocm_root)
        env["ROCPROFSYS_MAX_THREADS"] = "64"
        # Open MPI bakes its build-time prefix into its binaries.
        for name in ("OPAL_PREFIX", "PRTE_PREFIX", "PMIX_PREFIX"):
            env[name] = str(rocm_root)
    _prepend_path(env, "LD_LIBRARY_PATH", libs)
    return env


def run_one(
    rocjitsu: Path,
    config: Path,
    test: dict,
    cwd: Path,
    env: dict,
    log_path: Path,
    rocjitsu_args: list[str] | None = None,
) -> dict:
    timeout = timeout_seconds(test)
    command = [
        str(rocjitsu),
        *list(rocjitsu_args or []),
        "--config",
        str(config),
        "--",
        *test["command"],
    ]
    work = Path(test["working_directory"]) if test["working_directory"] else cwd
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(
        f"START {test['name']} timeout={timeout}s cmd={shlex.join(test['command'])}",
        flush=True,
    )
    try:
        proc = subprocess.run(
            command,
            cwd=work,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        output = proc.stdout or ""
        status = "passed" if proc.returncode == 0 else "failed"
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        status = "timeout"
        returncode = None
    if status != "passed" and output_skips(output, test["skip_regex"]):
        status = "skipped"
    elapsed = time.monotonic() - started
    log_path.write_text(output, encoding="utf-8")
    if status not in ("passed", "skipped"):
        tail = "\n".join(output.splitlines()[-20:])
        if tail:
            print(tail, flush=True)
    print(f"DONE {test['name']} {status} {elapsed:.1f}s", flush=True)
    return {
        "name": test["name"],
        "status": status,
        "seconds": round(elapsed, 1),
        "returncode": returncode,
        "timeout": timeout,
    }


def render_summary(rows: list[dict], label: str) -> str:
    lines = [
        f"# rocJITsu {label}",
        "",
        "| component | tests | passed | failed | timeout | skipped | not started | seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        counts = row["counts"]
        lines.append(
            "| {component} | {total} | {passed} | {failed} | {timeout} | "
            "{skipped} | {not_started} | {seconds:.0f} |".format(
                component=row["component"],
                total=row["total"],
                seconds=row["seconds"],
                **counts,
            )
        )
    lines.append("")
    return "\n".join(lines)


def matrix_payload(groups: list[dict]) -> dict:
    """GitHub Actions matrix JSON: one include entry per component."""
    return {"include": [{"component": group["component"]} for group in groups]}


def attach_option_values(argv: list[str], option: str) -> list[str]:
    """Bind the following token to ``option``, even when that token starts with ``--``."""
    joined = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == option and index + 1 < len(argv):
            joined.append(f"{option}={argv[index + 1]}")
            index += 2
            continue
        joined.append(token)
        index += 1
    return joined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocjitsu")
    parser.add_argument("--config")
    parser.add_argument("--rocm-root", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--deadline-seconds", type=int, default=19800)
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help="CTest label regex. Default ^ffm-quick$. Use ^quick$ for on-demand.",
    )
    parser.add_argument(
        "--component",
        default="",
        help="Run only this installed bin/ directory (e.g. hipRAND).",
    )
    parser.add_argument(
        "--components",
        default="",
        help="Comma-separated bin/ directories. Empty means every matching directory.",
    )
    parser.add_argument(
        "--rocjitsu-arg",
        action="append",
        default=[],
        help="Extra rocJITsu flag before --config. May start with --. Repeatable.",
    )
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument(
        "--list-json",
        action="store_true",
        help="Print a GitHub Actions matrix JSON of selected components and exit.",
    )
    raw = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(attach_option_values(raw, "--rocjitsu-arg"))

    rocm_root = Path(args.rocm_root)
    selected_names = parse_component_names(args.components)
    if args.component:
        selected_names = [args.component]
    missing = missing_components(rocm_root, args.label, selected_names)
    if missing:
        print(
            "unknown components for label " f"{args.label!r}: {', '.join(missing)}",
            file=sys.stderr,
            flush=True,
        )
        return 2

    groups = discover(
        rocm_root,
        label=args.label,
        component=args.component or None,
        components=selected_names or None,
    )
    total = sum(len(group["tests"]) for group in groups)
    # --list-json stdout is the matrix only. The workflow writes that stream
    # into GITHUB_OUTPUT, which rejects a second text line.
    listing = sys.stderr if args.list_json else sys.stdout
    print(
        f"label={args.label} directories: {len(groups)} tests: {total}",
        file=listing,
        flush=True,
    )
    for group in groups:
        print(
            f"  {len(group['tests']):4d}  {group['component']}",
            file=listing,
            flush=True,
        )

    if args.list_json:
        print(json.dumps(matrix_payload(groups), separators=(",", ":")))
        return 0 if groups else 1
    if args.list_only:
        return 0 if groups else 1

    if not args.rocjitsu or not args.config or not args.out_dir:
        parser.error("--rocjitsu, --config, and --out-dir are required to run tests")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_env = build_env(rocm_root)
    rocjitsu = Path(args.rocjitsu).resolve()
    config = Path(args.config).resolve()
    deadline = time.monotonic() + args.deadline_seconds
    rows = []
    any_problem = False
    for group in groups:
        started = time.monotonic()
        results = []
        env = component_env(rocm_root, group["component"], base_env)
        print(f"COMPONENT {group['component']} tests={len(group['tests'])}", flush=True)
        for test in group["tests"]:
            if time.monotonic() >= deadline:
                results.append(
                    {
                        "name": test["name"],
                        "status": "not_started",
                        "seconds": 0,
                        "returncode": None,
                        "timeout": timeout_seconds(test),
                    }
                )
                any_problem = True
                continue
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", test["name"])[:120]
            log_path = out_dir / "logs" / group["component"] / f"{safe}.log"
            result = run_one(
                rocjitsu,
                config,
                test,
                group["directory"],
                env,
                log_path,
                rocjitsu_args=args.rocjitsu_arg,
            )
            results.append(result)
            if result["status"] in ("failed", "timeout"):
                any_problem = True
        counts = {
            status: sum(1 for result in results if result["status"] == status)
            for status in ("passed", "failed", "timeout", "skipped", "not_started")
        }
        row = {
            "component": group["component"],
            "total": len(results),
            "seconds": time.monotonic() - started,
            "counts": counts,
            "tests": results,
        }
        rows.append(row)
        summary = render_summary(rows, args.label)
        (out_dir / "summary.md").write_text(summary, encoding="utf-8")
        (out_dir / "summary.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )
        print(summary, flush=True)
    return 1 if any_problem or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
