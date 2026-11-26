#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Cross-platform (Windows + Linux) CI/parent entrypoint for the rocKE engine.
# One command runs: (1) the relative-path contract guard, (2) the byte-identity
# gate, (3) the pytest suite, (4) the same suite again under ROCKE_BACKEND=both
# when the engine extension is importable, (5) the backend access-width gate, a
# compile check kept out of pytest discovery, (6) ctest if a build dir exists.
# All paths are derived relative to this file so the rocke/platform/ tree is
# copy-able verbatim.
#
# Usage:
#   python rocke/platform/tests/run_all.py [--no-guard] [--no-gate] [--no-pytest]
#       [--no-both] [--no-backend-width] [--keep-going] [--only SUBSTR]
#       [--only-ir SUBSTR] [--build-root DIR] [--config CONFIG]
#       [--pytest-target PATH_OR_NODE_ID]
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
# use `--no-gate` / `--no-ir-validity` / `--no-arch-domain` /
# `--no-backend-width` to opt out of one.
#
# The backend access-width gate is resource-bound (it compiles kernels), so it
# does not start once an earlier stage has failed (TESTING.md, section 5.0.1);
# `--keep-going` runs it anyway, to gather more failures in one run. It holds
# that gate only: the ctest stage after it runs whatever came before.

from __future__ import annotations

import argparse
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
# An installed provider tree holds the package beside tests/ rather than under
# python/; add it where it is, so importing this runner by path never depends on
# another test's sys.path changes.
if (ROCKE / "rocke").is_dir() and str(ROCKE) not in sys.path:
    sys.path.insert(0, str(ROCKE))

from rocke.tests.host_cpus import available_cpus

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


_CTEST_READINESS_EXEMPT = frozenset({"rocke_arch_domain_reproducibility"})


def ctest_ready(build_root: Path, config: str) -> bool:
    if not (build_root / "CTestTestfile.cmake").is_file():
        print("\n== ctest: SKIPPED (no configured test build) ==")
        return False
    # Imported here, not at the top: the runner must start where rocke is not
    # importable (an isolated `--help`).
    from rocke.tests.ctest_listing import list_ctest_entries

    # A configured source-script test is immediately executable, but does not
    # mean the registered native suite was built. Preserve run_all's contract:
    # only launch the entire CTest suite once at least one build product exists.
    commands = [
        test.get("command", [])
        for test in list_ctest_entries(build_root, config=config)
        if test.get("name") not in _CTEST_READINESS_EXEMPT
    ]
    if any(command and Path(command[0]).is_file() for command in commands):
        return True
    print(f"\n== ctest: SKIPPED (no built registered tests for {config}) ==")
    return False


# Native C++ test binaries that a pytest file drives through an env var. With a
# configured build each one is resolved from CTest, so its pytest cases run
# instead of skipping: (env var, CTest name, label). The installed rocke_pytest_*
# CTest entries (platform/CMakeLists.txt) set the same variables to the installed
# binaries; add a new pair in both places.
_ARCH_DOMAIN_COMPLETE_ENV = "ROCKE_RUN_ALL_ARCH_DOMAIN_COMPLETE"


NATIVE_PARITY_TESTS = (
    ("ROCKE_STORAGE_TEST", "rocke_storage", "native storage parity"),
    ("ROCKE_IR_TYPES_TEST", "rocke_ir_types", "native IR type rule parity"),
    ("ROCKE_FLOAT_FORMAT_TEST", "rocke_float_format", "native float repr parity"),
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


def _ctest_command_index(
    build_root: Path, config: str, requested: set[str]
) -> dict[str, list[str]]:
    """Read one CTest listing and index the requested test commands by name."""
    # Imported here, not at the top: the runner must start where rocke is not
    # importable (an isolated `--help`).
    from rocke.tests.ctest_listing import list_ctest_entries

    commands: dict[str, list[str]] = {}
    for test in list_ctest_entries(build_root, config=config):
        name = test["name"]
        if name not in requested:
            continue
        if name in commands:
            raise ValueError(f"CTest registered {name} more than once")
        commands[name] = test.get("command", [])
    return commands


def _export_native_executable(
    env: dict[str, str], var: str, executable: str | Path | None, label: str
) -> None:
    """Set ``var`` in ``env`` to ``executable``, resolved, or report the skip."""
    if not executable:
        print(f"\n== {label}: SKIPPED (no configured build or {var}) ==")
        return
    path = Path(executable).resolve()
    if not path.is_file():
        raise ValueError(f"{label} test executable does not exist: {path}")
    env[var] = str(path)
    print(f"\n== {label}: {path} ==")


def _native_executable(
    env: dict[str, str],
    ctest_commands: dict[str, list[str]] | None,
    var: str,
    ctest_name: str,
) -> str | None:
    """An explicit env value wins; otherwise use the configured CTest command."""
    executable = env.get(var)
    if not executable and ctest_commands is not None:
        command = ctest_commands.get(ctest_name)
        if not command:
            raise ValueError(f"CTest did not resolve the built {ctest_name} executable")
        executable = command[0]
    return executable


def _cmake_home(build_root: Path) -> Path | None:
    """The source tree ``build_root`` was configured from, or None when it is
    not a configured tree."""
    if not (build_root / "CMakeCache.txt").is_file():
        return None
    # Imported here, not at the top: the runner must start where rocke is not
    # importable (an isolated `--help`).
    from rocke.tests.platform_build import cmake_cache_entries

    home = cmake_cache_entries(build_root).get("CMAKE_HOME_DIRECTORY")
    return None if home is None else Path(home).resolve()


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


def _built_test_program(build_root: Path, config: str, name: str) -> Path | None:
    """Find the program ``name`` that CMake builds under tests/, with or without
    the executable suffix, in a single- or multi-config build."""
    return _built_file(
        build_root,
        config,
        tuple(
            f"{directory}/{name}{suffix}"
            for directory in ("tests", f"tests/{config}")
            for suffix in ("", ".exe")
        ),
    )


def _recorded_program(build_root: Path, config: str, target: str) -> Path | None:
    """The program ``target`` as the configure of ``build_root`` recorded it
    (rocke_record_program_path in platform/CMakeLists.txt): its path, whatever
    the build mode named it and wherever it put it, or None when the configure
    recorded none."""
    record = _built_file(build_root, config, (f"{target}.path",))
    if record is None:
        return None
    return Path(record.read_text(encoding="utf-8").strip())


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
    configured = (build_root / "CMakeCache.txt").is_file()
    if configured:
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
    unresolved_ctests = {
        ctest_name for var, ctest_name, _ in NATIVE_PARITY_TESTS if not env.get(var)
    }
    ctest_commands = (
        _ctest_command_index(build_root, config, unresolved_ctests)
        if configured and unresolved_ctests
        else None
    )
    for var, ctest_name, label in NATIVE_PARITY_TESTS:
        executable = _native_executable(env, ctest_commands, var, ctest_name)
        _export_native_executable(env, var, executable, label)

    # The C++ HIP lowering program, which is not a CTest: the both-engine HIP
    # checks run it where the binding is not built, and one test holds it to the
    # binding where it is. The installed rocke_hip_source_identity and
    # rocke_hip_lowering_parity CTests supply the same variable.
    from rocke.tests.engines import IR_LOWER_CLI_ENV

    ir_lower_cli = env.get(IR_LOWER_CLI_ENV) or _recorded_program(
        build_root, config, "rocke_ir_lower_cli"
    )
    _export_native_executable(
        env, IR_LOWER_CLI_ENV, ir_lower_cli, "native HIP lowering program"
    )

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

    replay_cli = _built_test_program(build_root, config, "rocke_portable_ir_replay_cli")
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


def differential_pytest_pass(env: dict[str, str], targets: list[str]) -> int:
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
    and the lane would prove nothing, so it is reported as a skipped *pass*
    with a reason rather than run.
    """
    probe = subprocess.run(
        [sys.executable, "-c", "import rocke_engine"],
        capture_output=True,
        cwd=str(TESTS),
        env=env,
    )
    if probe.returncode != 0:
        print(
            "\n== pytest (ROCKE_BACKEND=both): SKIPPED ==\n"
            "   rocke_engine is not importable; build it with "
            "-DROCKE_BUILD_PYBIND=ON and put the build dir on PYTHONPATH."
        )
        return 0
    from rocke.core.backend import CPP_UNPORTED_ARCHES

    scope = "FOCUSED, " if targets != [str(TESTS)] else ""
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
    env = dict(env, ROCKE_BACKEND="both")
    return subprocess.run(
        [sys.executable, "-m", "pytest", *targets, "-rs"], cwd=str(TESTS), env=env
    ).returncode


# A check that compiles kernels is kept out of ordinary pytest discovery
# (TESTING.md, section 5.0.1), so this runner names it as a stage of its own. Its
# installed twin is the CTest entry rocke_tiling_backend_width_pytest, whose
# per-test bound this stage keeps (tests/test_engine_linkage.py checks the two
# agree).
_BACKEND_WIDTH_CHECK = Path("helpers", "tiling", "check_backend_access_width.py")
BACKEND_WIDTH_TEST_TIMEOUT_S = 60


def backend_width_gate() -> int:
    """Run the tiling layer's backend access-width check (rocke.tests.helpers.tiling).

    It compiles a few copy kernels through COMGR and requires every global access
    in their code objects to move the width the tiling layer plans. Without COMGR,
    llvm-objdump or llvm-readelf it skips and the skip is reported; the stage
    inherits the caller's environment, so ROCKE_REQUIRE_COMPILER_TOOLS=1 makes
    those skips failures. The module is
    resolved from the imported ``rocke`` package, so the stage runs the same file
    in a source checkout and in an installed tree.
    """
    from rocke.tests import subprocess_launcher

    check = Path(subprocess_launcher.__file__).resolve().parent / _BACKEND_WIDTH_CHECK
    print("\n== backend access-width gate ==")
    command = subprocess_launcher.rocke_python_command(
        "-m",
        "pytest",
        "-c",
        str(TESTS / "pytest.ini"),
        str(check),
        "-rs",
        f"--timeout={BACKEND_WIDTH_TEST_TIMEOUT_S}",
    )
    return subprocess.run(
        command,
        check=False,
        cwd=str(TESTS),
        env=subprocess_launcher.rocke_subprocess_env(),
    ).returncode


def main() -> int:
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
        "--no-backend-width",
        action="store_true",
        help="skip the backend access-width gate (compile copy kernels through "
        "COMGR and check the width of every global access)",
    )
    ap.add_argument(
        "--keep-going",
        action="store_true",
        help="run the backend access-width gate even after an earlier stage "
        "failed (it holds that gate only; the ctest stage runs either way)",
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
    ap.add_argument(
        "--build-root", default=str(Path(tempfile.gettempdir()) / "rocke_verify")
    )
    ap.add_argument(
        "--config", default="Release", help="native test build/CTest configuration"
    )
    args = ap.parse_args()
    try:
        selected_pytests = pytest_targets(args.pytest_target)
    except ValueError as exc:
        ap.error(str(exc))
    pytest_focused = bool(args.pytest_target)
    run_focused = bool(args.only or args.only_ir or args.pytest_target)

    status = 0
    arch_domain_complete = False

    if not args.no_guard:
        status |= relative_path_guard()

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
        status |= subprocess.run(gate).returncode

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
        status |= subprocess.run(ir_gate).returncode

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
        arch_domain_status = subprocess.run(ad_gate).returncode
        status |= arch_domain_status
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
            return 1
        label = "pytest (FOCUSED)" if pytest_focused else "pytest"
        print(f"\n== {label} ==")
        status |= subprocess.run(
            [sys.executable, "-m", "pytest", *selected_pytests],
            cwd=str(TESTS),
            env=pytest_env,
        ).returncode

    if not args.no_pytest and not args.no_both:
        status |= differential_pytest_pass(pytest_env, selected_pytests)

    if not args.no_backend_width:
        if status and not args.keep_going:
            print(
                "\n== backend access-width gate: NOT RUN (an earlier stage failed; "
                "pass --keep-going to run it) =="
            )
        else:
            status |= backend_width_gate()

    build_root = Path(args.build_root)
    # Run the entire registered suite once any executable is built; partial
    # builds must expose their missing tests rather than silently lose coverage.
    try:
        ready = ctest_ready(build_root, args.config)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"CTest discovery failed: {exc}", file=sys.stderr)
        return 1
    if ready:
        print("\n== ctest ==")
        status |= subprocess.run(
            ["ctest", "-C", args.config, "--output-on-failure", "--no-tests=ignore"],
            cwd=str(build_root),
        ).returncode

    result_scope = "FOCUSED " if run_focused else ""
    print("\nRESULT:", result_scope + ("GREEN" if status == 0 else "RED"))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
