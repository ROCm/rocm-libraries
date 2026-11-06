#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Cross-platform (Windows + Linux) CI/parent entrypoint for the rocKE engine.
# One command runs: (1) the relative-path contract guard, (2) the byte-identity
# gate, (3) the pytest suite, (4) the same suite again under ROCKE_BACKEND=both
# when the engine extension is importable, (5) ctest if a build dir exists. All
# paths are derived relative to this file so the rocke/platform/ tree is
# copy-able verbatim.
#
# Usage:
#   python rocke/platform/tests/run_all.py [--no-guard] [--no-gate] [--no-pytest]
#       [--no-ir-validity] [--no-arch-domain] [--no-both] [--only SUBSTR]
#       [--only-ir SUBSTR] [--build-root DIR] [--config CONFIG]
#       [--pytest-target PATH_OR_NODE_ID] [--allow-incomplete]
#
# Every stage is recorded in a `rocke.tests.runner_verdict.RunRecord`, which
# prints the stage table and the final `RESULT: <VERDICT>` line and owns the
# exit code: GREEN/PARTIAL 0 (PARTIAL = narrowed by a filter or `--no-*`), RED
# 1, INCOMPLETE 3 (a stage could not run here or executed no tests; 0 with
# `--allow-incomplete`), usage error 2.
#
# The two filters are separate by *vocabulary*, not by gate. `--only` selects
# *families* for the byte-identity gate; `--only-ir` selects *case ids*
# (`family/arch/variant`) for both corpus gates -- emitted-IR validity and
# arch-domain. The vocabularies overlap without being the same, and a word can
# be a legal family while matching no corpus case; the tools read an empty
# selection oppositely (run_diff reports green, the corpus gates call it FATAL),
# so forwarding one substring to all of them can go red for a filter that was
# valid where the user aimed it. Any new corpus gate belongs on `--only-ir`.
#
# Filtering is separate from skipping: an unfiltered gate still runs in full;
# use `--no-gate` / `--no-ir-validity` / `--no-arch-domain` to opt out of one.

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROCKE = Path(__file__).resolve().parents[1]  # tests -> rocKE
TESTS = ROCKE / "tests"
TOOLS = ROCKE / "tools"

# Same sys.path bootstrap as tests/conftest.py, so the runner can read shared
# constants (e.g. backend.CPP_UNPORTED_ARCHES) out of the package it is testing
# instead of restating them here and letting the two drift.
if str(ROCKE / "python") not in sys.path:
    sys.path.insert(0, str(ROCKE / "python"))
# Source runs keep helpers under tools/. Installed tests stage the same helper
# beside the arch-domain generator instead. Bootstrap the first complete home so
# importing this runner by path never depends on another test's sys.path changes.
for helper_root in (TOOLS, TESTS / "arch_domain"):
    if (helper_root / "_hostcaps.py").is_file():
        if str(helper_root) not in sys.path:
            sys.path.insert(0, str(helper_root))
        break

from _hostcaps import available_cpus
from rocke.tests.runner_verdict import (
    RunRecord,
    junit_counts,
    line_buffered_stdout,
    user_scratch_dir,
)

# Files that may reference an absolute repo path or a path that escapes rocke/platform/
# break the verbatim-copy contract. Enforce on code/build files only (docs are
# exempt). A clean run is required before the tree is dropped into another repo.
_GUARD_SUFFIXES = {".py", ".cmake", ".toml", ".ini", ".sh", ".cfg"}
_GUARD_NAMES = {"CMakeLists.txt"}
_GUARD_SKIP_DIRS = {".git", "__pycache__", "build", "dsl_docs", "examples"}
# Any ".venv*" dir is a local virtual environment (".venv", ".venv-torch", ...);
# keeping side venvs around for CI parity and local dev must not trip the guard.
_GUARD_SKIP_PREFIXES = (".venv",)
_FORBIDDEN = [
    re.compile(r"/workspace\b"),
    re.compile(r"rocm-libraries(?:-[a-z-]+)?/"),
    re.compile(r"projects/composablekernel"),
    re.compile(r"dnn-providers/"),
]


def relative_path_guard() -> int:
    """Fail if any code/build file under rocke/platform/ references an absolute repo path."""
    violations: list[str] = []
    for path in ROCKE.rglob("*"):
        if not path.is_file():
            continue
        if any(
            part in _GUARD_SKIP_DIRS or part.startswith(_GUARD_SKIP_PREFIXES)
            for part in path.relative_to(ROCKE).parts
        ):
            continue
        if path.suffix not in _GUARD_SUFFIXES and path.name not in _GUARD_NAMES:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue  # this guard file defines the patterns
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for pat in _FORBIDDEN:
                if pat.search(line):
                    violations.append(
                        f"{path.relative_to(ROCKE)}:{i}: {line.strip()[:100]}"
                    )
    if violations:
        print(
            "RELATIVE-PATH GUARD: FAIL - absolute/repo paths found under rocke/platform/:"
        )
        for v in violations:
            print(f"  {v}")
        return 1
    print("RELATIVE-PATH GUARD: PASS")
    return 0


def ctest_tests(
    build_root: Path, config: str, pattern: str | None = None
) -> list[dict]:
    command = [
        "ctest",
        "--test-dir",
        str(build_root),
        "-C",
        config,
        "--show-only=json-v1",
    ]
    if pattern:
        command += ["-R", pattern]
    listing = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(listing.stdout)["tests"]


_CTEST_READINESS_EXEMPT = frozenset({"rocke_arch_domain_reproducibility"})


def ctest_unready(build_root: Path, config: str) -> str | None:
    """Why CTest cannot run the registered suite here, or ``None`` when it can."""
    if not (build_root / "CTestTestfile.cmake").is_file():
        return "no configured test build"
    # A configured source-script test is immediately executable, but does not
    # mean the registered native suite was built. Preserve run_all's contract:
    # only launch the entire CTest suite once at least one build product exists.
    commands = [
        test.get("command", [])
        for test in ctest_tests(build_root, config)
        if test.get("name") not in _CTEST_READINESS_EXEMPT
    ]
    if any(command and Path(command[0]).is_file() for command in commands):
        return None
    return f"no built registered tests for {config}"


# Native C++ test binaries that a pytest file drives through an env var. With a
# configured build each one is resolved from CTest, so its pytest cases run
# instead of skipping: (env var, CTest name, label). The installed rocke_pytest
# CTest entry (platform/CMakeLists.txt) sets the same variables to the installed
# binaries; add a new pair in both places.
_ARCH_DOMAIN_COMPLETE_ENV = "ROCKE_RUN_ALL_ARCH_DOMAIN_COMPLETE"


NATIVE_PARITY_TESTS = (
    ("ROCKE_STORAGE_TEST", "rocke_storage", "native storage parity"),
    ("ROCKE_IR_TYPES_TEST", "rocke_ir_types", "native IR type rule parity"),
    (
        "ROCKE_NONTEMPORAL_HIP_TEST",
        "rocke_nontemporal_hip",
        "native nontemporal HIP parity",
    ),
    (
        "ROCKE_UNSIGNED_INT_OPS_CLI",
        "rocke_unsigned_int_ops",
        "native unsigned integer op C API parity",
    ),
    (
        "ROCKE_INTEXPR_CLI",
        "rocke_intexpr",
        "native integer-expression evaluator parity",
    ),
    (
        "ROCKE_PARAM_FACTS_CLI",
        "rocke_param_facts",
        "native parameter-fact print, verify and canonicalize parity",
    ),
    (
        "ROCKE_GPU_CHECKED_VALUE_CLI",
        "rocke_gpu_checked_value",
        "native GPU check builder and range marker verifier parity",
    ),
    (
        "ROCKE_ARCH_TARGET_TEST_EXE",
        "rocke_arch_target_identity",
        "native target identity parity",
    ),
    (
        "ROCKE_BUFFER_RSRC_HIP_TEST",
        "rocke_buffer_rsrc_hip",
        "native buffer descriptor HIP parity",
    ),
)


def _native_executable(
    env: dict[str, str], build_root: Path, config: str, var: str, ctest_name: str
) -> str | None:
    """An explicit env value wins; otherwise ask CTest for the built binary."""
    executable = env.get(var)
    if not executable and (build_root / "CMakeCache.txt").is_file():
        tests = ctest_tests(build_root, config, f"^{ctest_name}$")
        commands = [
            test.get("command", []) for test in tests if test["name"] == ctest_name
        ]
        if len(commands) != 1 or not commands[0]:
            raise ValueError(f"CTest did not resolve the built {ctest_name} executable")
        executable = commands[0][0]
    return executable


def _cmake_home(build_root: Path) -> Path | None:
    cache = build_root / "CMakeCache.txt"
    if not cache.is_file():
        return None
    for line in cache.read_text(errors="ignore").splitlines():
        if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
            return Path(line.split("=", 1)[1]).resolve()
    return None


def configure_pytest_build(build_root: Path, config: str) -> None:
    """Configure run_all's native artifacts independently of earlier gates."""
    home = _cmake_home(build_root)
    if home is not None and home != ROCKE.resolve():
        return
    if home is None and (build_root / "CMakeCache.txt").exists():
        return  # A synthetic/external test build; do not replace its project.
    subprocess.run(
        [
            "cmake",
            "-S",
            str(ROCKE),
            "-B",
            str(build_root),
            f"-DCMAKE_BUILD_TYPE={config}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DROCKE_BUILD_SHARED_ENGINE=ON",
            "-DBUILD_TESTING=ON",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _built_file(
    build_root: Path, config: str, relative_names: tuple[str, ...]
) -> Path | None:
    """Find a single- or multi-config CMake artifact by its relative name."""
    for relative in relative_names:
        for root in (build_root, build_root / config):
            candidate = root / relative
            if candidate.is_file():
                return candidate.resolve()
    return None


def native_pytest_env(build_root: Path, config: str) -> dict[str, str]:
    """Build the configured suite and supply its native artifacts to pytest."""
    env = dict(os.environ)
    # The canonical runner owns these artifacts. Ambient selections must not
    # survive a missing or partial configured build.
    env.pop("ROCKE_ONLINE_LIB", None)
    env.pop("ROCKE_REPLAY_CLI", None)
    env.pop("ROCKE_PROVENANCE_MANIFEST", None)
    # Only this runner may claim that its own complete corpus gate succeeded.
    # An ambient value must not suppress the standalone pytest delegation.
    env.pop(_ARCH_DOMAIN_COMPLETE_ENV, None)
    if (build_root / "CMakeCache.txt").is_file():
        subprocess.run(
            [
                "cmake",
                "--build",
                str(build_root),
                "--config",
                config,
                "--parallel",
                str(available_cpus()),
            ],
            check=True,
        )
    for var, ctest_name, label in NATIVE_PARITY_TESTS:
        executable = _native_executable(env, build_root, config, var, ctest_name)
        if executable:
            path = Path(executable).resolve()
            if not path.is_file():
                raise ValueError(f"{label} test executable does not exist: {path}")
            env[var] = str(path)
            print(f"\n== {label}: {path} ==")
        else:
            print(f"\n== {label}: SKIPPED (no configured build or {var}) ==")

    # Keep every portable-IR binary on this one configured source snapshot.
    # In the ordinary run_all flow check_byte_identity configured the shared
    # target before this function built the full suite. Explicit caller values
    # are intentionally replaced: the canonical runner owns its artifacts.
    shared = _built_file(
        build_root,
        config,
        ("librocke.so", "librocke.dylib", "rocke.dll"),
    )
    if shared:
        env["ROCKE_ONLINE_LIB"] = str(shared)
        print(f"== portable-IR shared engine: {shared} ==")
    elif _cmake_home(build_root) == ROCKE.resolve():
        raise ValueError("configured rocKE build produced no portable-IR shared engine")
    else:
        print("== portable-IR shared engine: not in configured build ==")

    replay_cli = _built_file(
        build_root,
        config,
        (
            "tests/rocke_portable_ir_replay_cli",
            "tests/rocke_portable_ir_replay_cli.exe",
            f"tests/{config}/rocke_portable_ir_replay_cli",
            f"tests/{config}/rocke_portable_ir_replay_cli.exe",
        ),
    )
    if replay_cli:
        env["ROCKE_REPLAY_CLI"] = str(replay_cli)
        print(f"== portable-IR replay CLI: {replay_cli} ==")
    elif _cmake_home(build_root) == ROCKE.resolve():
        raise ValueError("configured rocKE build produced no portable-IR replay CLI")
    else:
        print("== portable-IR replay CLI: not in configured build ==")

    if shared and replay_cli and _cmake_home(build_root) == ROCKE.resolve():
        from rocke.portable_ir.src import online

        try:
            (engine_version, engine_id), _ = online.validate_artifact_bundle(
                str(shared), str(replay_cli)
            )
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        flavor = env.get("ROCKE_LLVM_FLAVOR", "auto")
        print(
            f"== portable-IR provenance: {engine_version} build-id={engine_id} "
            f"LLVM flavor={flavor} =="
        )

    # The byte-identity stage has already built this production archive. The
    # native compiler-detection tests otherwise configure and compile a second
    # copy in pytest's temporary directory, then repeat that work in the
    # differential pytest pass. Reuse is safe because the fixture explicitly
    # asks for the same production archive and only links test wrappers around
    # it; an explicit caller override still wins.
    archive = env.get("ROCKE_TEST_ENGINE_ARCHIVE")
    if not archive:
        # Multi-config generators put archives in the selected configuration.
        for directory in (build_root / config, build_root):
            candidate = directory / "librocke_core.a"
            if candidate.is_file():
                archive = str(candidate.resolve())
                break
    if archive:
        path = Path(archive).resolve()
        if not path.is_file():
            raise ValueError(f"native detector archive does not exist: {path}")
        env["ROCKE_TEST_ENGINE_ARCHIVE"] = str(path)
        print(f"== native detector archive: {path} ==")
    return env


def pytest_targets(selectors: list[str]) -> list[str]:
    """Resolve focused pytest paths/node IDs without allowing tree escapes."""
    if not selectors:
        return [str(TESTS)]

    targets: list[str] = []
    for selector in selectors:
        path_text, separator, node_id = selector.partition("::")
        path = Path(path_text)
        if not path.is_absolute():
            parts = path.parts
            if parts and parts[0] == "tests":
                path = Path(*parts[1:])
            path = TESTS / path
        resolved = path.resolve()
        try:
            resolved.relative_to(TESTS.resolve())
        except ValueError as exc:
            raise ValueError(f"pytest target escapes {TESTS}: {selector}") from exc
        if not resolved.exists():
            raise ValueError(f"pytest target does not exist: {selector}")
        target = str(resolved)
        if separator:
            if not node_id:
                raise ValueError(f"pytest target has an empty node ID: {selector}")
            target += f"::{node_id}"
        if target not in targets:
            targets.append(target)
    return targets


def run_pytest(
    targets: list[str], env: dict[str, str], junit: Path, *extra: str
) -> tuple[int, int | None]:
    """Run one pytest pass; return its exit status and how many tests executed.

    The count comes from the pass's JUnit report and is ``None`` when the report
    is unreadable, so an unknown count is never mistaken for zero.
    """
    code = subprocess.run(
        [sys.executable, "-m", "pytest", *targets, *extra, f"--junitxml={junit}"],
        check=False,
        cwd=str(TESTS),
        env=env,
    ).returncode
    counts = junit_counts(junit)
    return code, None if counts is None else counts[0]


BOTH_STAGE = "pytest (ROCKE_BACKEND=both)"


def differential_pytest_pass(
    record: RunRecord, env: dict[str, str], targets: list[str], junit: Path
) -> None:
    """Re-run pytest with ``ROCKE_BACKEND=both`` (the cross-engine gate).

    The default pass exercises one engine per assertion, so two engines that
    each emit self-consistent but different IR both go green. ``both`` lowers
    through each and raises ``BackendMismatch`` on any byte difference, which
    is the only thing in the suite that can see a divergence the byte-identity
    gate's fixed family list does not cover.

    ``both`` never substitutes the Python result for a kernel the C++ engine
    could not lower, so this pass cannot go green vacuously: a kernel is either
    compared, or it fails, or -- for an arch named in
    ``backend.CPP_UNPORTED_ARCHES`` -- it is reported as a skip by the
    ``BackendCoverageGap`` hook in conftest. The skip count is the size of the
    known gap, and it is printed below so it stays visible.

    Needs the ``rocke_engine`` extension. Without it every kernel would raise
    and the lane would prove nothing, so it is recorded as a skipped stage
    (the run is INCOMPLETE) rather than run.
    """
    probe = subprocess.run(
        [sys.executable, "-c", "import rocke_engine"],
        check=False,
        capture_output=True,
        cwd=str(TESTS),
        env=env,
    )
    if probe.returncode != 0:
        print(
            f"\n== {BOTH_STAGE}: SKIPPED ==\n"
            "   rocke_engine is not importable; build it with "
            "-DROCKE_BUILD_PYBIND=ON and put the build dir on PYTHONPATH."
        )
        record.skipped(BOTH_STAGE, "rocke_engine is not importable")
        return
    from rocke.core.backend import CPP_UNPORTED_ARCHES

    scope = "PARTIAL, " if targets != [str(TESTS)] else ""
    print(f"\n== pytest ({scope}ROCKE_BACKEND=both) ==")
    if CPP_UNPORTED_ARCHES:
        print(
            "   no fallback to Python in this lane; skips are the known C++ "
            f"coverage gap (unported arches: {', '.join(CPP_UNPORTED_ARCHES)})"
        )
    else:
        print(
            "   no fallback to Python in this lane, and no arch is exempt "
            "(backend.CPP_UNPORTED_ARCHES is empty), so every kernel here is "
            "either compared byte-for-byte or fails; remaining skips are "
            "environmental (torch / GPU)"
        )
    code, executed = run_pytest(targets, dict(env, ROCKE_BACKEND="both"), junit, "-rs")
    record.returncode(BOTH_STAGE, code, executed=executed)


RUNNER = "platform tests/run_all.py"


def main() -> int:
    line_buffered_stdout()

    ap = argparse.ArgumentParser(description="rocKE test/validation runner")
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-pytest", action="store_true")
    ap.add_argument(
        "--no-ir-validity",
        action="store_true",
        help="skip the emitted-IR validity gate (compile+link every corpus case)",
    )
    ap.add_argument(
        "--no-arch-domain",
        action="store_true",
        help="skip the arch-domain warning gate (lower every corpus case at "
        "every committed LLVM flavor)",
    )
    ap.add_argument(
        "--no-both",
        action="store_true",
        help="skip the ROCKE_BACKEND=both differential pytest pass",
    )
    ap.add_argument(
        "--only",
        default="",
        help="restrict byte-identity gate to families containing SUBSTR "
        "(comma-separated); does not affect the corpus gates",
    )
    ap.add_argument(
        "--only-ir",
        default="",
        help="restrict the corpus gates (emitted-IR validity, arch-domain) to "
        "case ids containing SUBSTR (comma-separated); a case id is "
        "family/arch/variant",
    )
    ap.add_argument(
        "--pytest-target",
        action="append",
        default=[],
        metavar="PATH_OR_NODE_ID",
        help="run only this pytest path or node ID, relative to tests/; repeatable. "
        "Other gates and CTest remain complete",
    )
    ap.add_argument("--build-root", default=str(user_scratch_dir("rocke_verify")))
    ap.add_argument(
        "--config", default="Release", help="native test build/CTest configuration"
    )
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="exit 0 when the verdict is INCOMPLETE (a stage could not run here "
        "or executed no tests); the verdict line still says INCOMPLETE",
    )
    args = ap.parse_args()
    try:
        selected_pytests = pytest_targets(args.pytest_target)
    except ValueError as exc:
        ap.error(str(exc))

    record = RunRecord(allow_incomplete=args.allow_incomplete)
    for flag, used in (
        ("--no-guard", args.no_guard),
        ("--no-gate", args.no_gate),
        ("--no-ir-validity", args.no_ir_validity),
        ("--no-arch-domain", args.no_arch_domain),
        ("--no-pytest", args.no_pytest),
        ("--no-both", args.no_both),
    ):
        if used:
            record.partial(flag)
    if args.only:
        record.partial(f"--only {args.only}")
    if args.only_ir:
        record.partial(f"--only-ir {args.only_ir}")
    for target in args.pytest_target:
        record.partial(f"--pytest-target {target}")

    arch_domain_complete = False

    if not args.no_guard:
        record.returncode("relative-path guard", relative_path_guard())

    if not args.no_gate:
        print("\n== byte-identity gate ==")
        gate = [
            sys.executable,
            str(TOOLS / "check_byte_identity.py"),
            "--build-root",
            args.build_root,
        ]
        if args.only:
            gate += ["--only", args.only]
        gate_status = subprocess.run(gate, check=False).returncode
        record.returncode("byte-identity gate", gate_status)

    if not args.no_ir_validity:
        # Complements the gate above rather than duplicating it: byte-identity
        # proves the two engines agree, this proves what they agree on is legal
        # IR the AMDGPU toolchain accepts. Self-skips (green, loudly) on a host
        # with no LLVM tools -- pass --strict there to make that a failure.
        print("\n== emitted-IR validity gate ==")
        ir_gate = [sys.executable, str(TOOLS / "check_ir_validity.py")]
        # `--only-ir`, never `--only`: see the module header. An empty selection
        # is FATAL in that tool and that stays correct here, because reaching it
        # now means the user aimed a case-id filter at the case-id gate and hit
        # nothing -- a typo, which is worth a red.
        if args.only_ir:
            ir_gate += ["--only", args.only_ir]
        ir_status = subprocess.run(ir_gate, check=False).returncode
        record.returncode("emitted-IR validity gate", ir_status)

    if not args.no_arch_domain:
        # The gate above's question asked of committed data instead of a
        # compiler: does any kernel demand an intrinsic its target provably
        # cannot lower? Needs no toolchain, so it rules on every LLVM flavor.
        print("\n== arch-domain warning gate ==")
        ad_gate = [sys.executable, str(TOOLS / "check_arch_domain.py")]
        # `--only-ir`, not `--only`: this gate matches case ids, like
        # check_ir_validity. See the note at the top of this file.
        if args.only_ir:
            ad_gate += ["--only", args.only_ir]
        arch_domain_status = subprocess.run(ad_gate, check=False).returncode
        record.returncode("arch-domain gate", arch_domain_status)
        arch_domain_complete = arch_domain_status == 0 and not args.only_ir

    if not args.no_pytest:
        try:
            build_root = Path(args.build_root).resolve()
            configure_pytest_build(build_root, args.config)
            pytest_env = native_pytest_env(build_root, args.config)
            if arch_domain_complete:
                pytest_env[_ARCH_DOMAIN_COMPLETE_ENV] = "1"
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            print(f"native parity test setup failed: {exc}", file=sys.stderr)
            record.failed("native parity setup", str(exc))
            return record.report(RUNNER)
        with tempfile.TemporaryDirectory(prefix="rocke_run_all_") as junit_dir:
            label = "pytest (PARTIAL)" if args.pytest_target else "pytest"
            print(f"\n== {label} ==")
            code, executed = run_pytest(
                selected_pytests, pytest_env, Path(junit_dir) / "pytest.xml"
            )
            record.returncode("pytest", code, executed=executed)
            if not args.no_both:
                differential_pytest_pass(
                    record, pytest_env, selected_pytests, Path(junit_dir) / "both.xml"
                )

    build_root = Path(args.build_root)
    # Run the entire registered suite once any executable is built; partial
    # builds must expose their missing tests rather than silently lose coverage.
    try:
        unready = ctest_unready(build_root, args.config)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"CTest discovery failed: {exc}", file=sys.stderr)
        record.failed("ctest", f"discovery failed: {exc}")
        return record.report(RUNNER)
    if unready is None:
        print("\n== ctest ==")
        ctest = subprocess.run(
            ["ctest", "-C", args.config, "--output-on-failure", "--no-tests=ignore"],
            check=False,
            cwd=str(build_root),
        )
        record.returncode("ctest", ctest.returncode)
    elif args.no_pytest:
        # The pytest stage is what configures and builds the suite, so its
        # absence here follows from the caller's opt-out, not from this host.
        print(f"\n== ctest: SKIPPED ({unready}; --no-pytest) ==")
        record.partial(f"ctest: {unready} (--no-pytest)")
    else:
        print(f"\n== ctest: SKIPPED ({unready}) ==")
        record.skipped("ctest", unready)

    return record.report(RUNNER)


if __name__ == "__main__":
    raise SystemExit(main())
