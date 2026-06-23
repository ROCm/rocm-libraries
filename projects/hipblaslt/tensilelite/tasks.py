# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

from invoke.tasks import task
import os
import pathlib
import shlex
import shutil
import subprocess
import sys

_TASKS_DIR = pathlib.Path(__file__).parent.resolve()


def _cmake_bool(value):
    return "ON" if value else "OFF"


def _detect_rocm():
    """Detect ROCm installation path.

    Priority: ROCM_PATH env > rocm-sdk path --root > /opt/rocm.
    """
    env_path = os.environ.get("ROCM_PATH")
    if env_path:
        return env_path

    if shutil.which("rocm-sdk"):
        try:
            result = subprocess.check_output(
                ["rocm-sdk", "path", "--root"], stderr=subprocess.DEVNULL
            ).decode().strip()
            if result:
                return result
        except subprocess.CalledProcessError:
            pass

    return "/opt/rocm"


def detect_gpu_arch():
    try:
        result = subprocess.run(["rocm_agent_enumerator", "-v"], capture_output=True, text=True, timeout=5, check=True)
        if result.returncode == 0:
            target = next((line.strip() for line in result.stdout.splitlines() if line.startswith("gfx") and line.strip() != "gfx000"), None)
            if target:
                return target
    except FileNotFoundError:
        print("Error: 'rocm_agent_enumerator' command not found. Please install ROCm.", file=sys.stderr)

    except subprocess.TimeoutExpired:
        print("Error: GPU detection timed out. Hardware might be unresponsive.", file=sys.stderr)

    except Exception as e:
        print(f"An unexpected error occurred during GPU detection: {e}", file=sys.stderr)

    print(f"Failed to detect a valid GPU architecture (gfx target not found).", file=sys.stderr)
    return None

@task
def get_gpu_arch(c):
    print(detect_gpu_arch())

@task(
    help={
        "rocisa_dir": "Path to the rocisa source directory (default: rocisa/ next to this file).",
        "stinkytofu_prefix": "Install prefix for the stinkytofu build (default: build_tmp/stinkytofu-install).",
    }
)
def rocisa(c, rocisa_dir=None, stinkytofu_prefix=None):
    """Install rocisa as an editable pip package.

    Run once after cloning, or after changes to rocisa's pyproject.toml or
    CMakeLists.txt. C++ source changes are picked up automatically: `invoke
    build-client` re-runs this editable install (an incremental no-op when
    nothing changed), with the staleness check in rocisa/__init__.py as a
    backstop if you skip that.

    Builds and installs stinkytofu locally first so rocisa uses
    find_package(stinkytofu) — mirroring how TheRock wires the two together.
    """
    _pip_install_rocisa(c, rocisa_dir, stinkytofu_prefix)


def _load_stinkytofu_tasks():
    """Import shared/stinkytofu/tasks.py without triggering its venv guard.

    The venv check was moved into build() so this import is side-effect-free.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stinkytofu_tasks",
        _TASKS_DIR.parent.parent.parent / "shared" / "stinkytofu" / "tasks.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_and_install_stinkytofu(c, install_prefix: pathlib.Path, rocm: str) -> None:
    """Build stinkytofu and install it to install_prefix so rocisa can find_package it.

    Build flags come from stinkytofu_tasks.cmake_build_args() — the single source
    of truth — so a new required cmake option only needs to be added there.
    Compiler selection mirrors shared/stinkytofu/tasks.py `invoke build`.
    cmake is incremental, so repeat calls are a fast no-op when nothing changed.
    """
    stinkytofu_src = _TASKS_DIR.parent.parent.parent / "shared" / "stinkytofu"
    build_dir = install_prefix.parent / "stinkytofu-build"
    build_dir.mkdir(parents=True, exist_ok=True)

    rocm_s = rocm if isinstance(rocm, str) else str(rocm)
    _cxx = shutil.which("amdclang++") or f"{rocm_s}/bin/amdclang++"
    _cc = shutil.which("amdclang") or f"{rocm_s}/bin/amdclang"

    st = _load_stinkytofu_tasks()
    cmake_cmd = [
        "cmake",
        "-S", str(stinkytofu_src),
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DROCM_PATH={rocm_s}",
        f"-DCMAKE_CXX_COMPILER={_cxx}",
        f"-DCMAKE_C_COMPILER={_cc}",
        # tests/python OFF for the rocisa integration build; examples ON (default).
        *st.cmake_build_args(install_prefix=install_prefix, tests=False, python=False),
    ]
    if shutil.which("ninja"):
        cmake_cmd.append("-G Ninja")
    if shutil.which("ccache"):
        cmake_cmd += [
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        ]
    c.run(shlex.join(cmake_cmd))
    c.run(shlex.join(["cmake", "--build", str(build_dir), "--parallel"]))
    c.run(shlex.join(["cmake", "--install", str(build_dir)]))


def _pip_install_rocisa(c, rocisa_dir=None, stinkytofu_prefix=None):
    """Editable-install rocisa via scikit-build-core.

    Factored out of the `rocisa` task so `build_client` can reuse it to keep
    the editable install fresh.

    Builds stinkytofu and installs it to stinkytofu_prefix (default:
    build_tmp/stinkytofu-install next to this file) so rocisa's CMake finds it
    via find_package(stinkytofu) — the same path TheRock uses. This exercises
    the installed package layout (stinkytofuConfig.cmake, exported targets) so
    breakage is caught early in the dev/CI workflow.
    """
    src = pathlib.Path(rocisa_dir).resolve() if rocisa_dir else _TASKS_DIR / "rocisa"
    rocm = _detect_rocm()

    prefix = (
        pathlib.Path(stinkytofu_prefix).resolve()
        if stinkytofu_prefix
        else _TASKS_DIR / "build_tmp" / "stinkytofu-install"
    )
    _build_and_install_stinkytofu(c, prefix, rocm)

    cmake_args = (
        f"-DROCM_PATH={rocm}"
        f" -DROCISA_INCLUDE_BUILD_INFO=ON"
        f" -DCMAKE_PREFIX_PATH={prefix}"
    )
    if shutil.which("ccache"):
        cmake_args += " -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    env = dict(os.environ, CMAKE_ARGS=cmake_args)
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 1))
    c.run(f"pip install --no-build-isolation -e {shlex.quote(str(src))}", env=env)


def _rocisa_is_editable():
    """Return True only if rocisa is currently installed as an editable package.

    Reads PEP 610 direct_url.json metadata; it does NOT import rocisa, so the
    import-time staleness check is never triggered here. `invoke rocisa`
    (pip install -e) sets dir_info.editable=true; a plain `pip install rocisa/`
    (e.g. tox) or a wheel does not, so we never rebuild over a deliberate
    non-editable install, and we return False when rocisa is absent.
    """
    import json
    from importlib import metadata

    try:
        raw = metadata.distribution("rocisa").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return False
    if not raw:
        return False
    try:
        return bool(json.loads(raw).get("dir_info", {}).get("editable"))
    except (ValueError, AttributeError):
        return False


def _maybe_rebuild_rocisa(c, rocisa_dir=None):
    """Refresh the editable rocisa so `import rocisa` picks up C++ edits.

    `invoke build-client` builds a *separate* `_rocisa.so` under the client
    build dir; it never touches the editable install that `import rocisa`
    (and `Tensile/bin/Tensile`) actually load. So after editing rocisa C++ and
    running only build-client, those bindings are stale — silently for installs
    without `_build_info.py`, or an ImportError otherwise. Re-running the
    editable install here fixes that.

    This also re-runs the stinkytofu cmake configure+build+install step. cmake is
    incremental, so when stinkytofu sources are unchanged this is a fast no-op and
    does not affect the staleness-check semantics — the stale rocisa detection via
    _build_info.py / direct_url.json is unaffected.

    No-op unless rocisa is installed editable. Degrades to a warning — never a
    hard failure — when the build backend (scikit-build-core / nanobind) is
    unavailable, since the editable install uses --no-build-isolation and needs
    those importable. This keeps build-client working in CI/tox and on fresh
    checkouts that lack the rocisa build toolchain.
    """
    import importlib.util

    if not _rocisa_is_editable():
        return

    missing = [m for m in ("scikit_build_core", "nanobind") if importlib.util.find_spec(m) is None]
    if missing:
        print(
            "warning: editable rocisa is installed but its build backend is "
            f"unavailable ({', '.join(missing)}); skipping rocisa rebuild. If you "
            "changed rocisa C++ sources, run 'invoke rocisa' where the build deps exist.",
            file=sys.stderr,
        )
        return

    try:
        print("Rebuilding editable rocisa to pick up any C++ source changes...")
        _pip_install_rocisa(c, rocisa_dir)
    except Exception as e:
        print(
            f"warning: rocisa rebuild failed ({e}); continuing with the client build. "
            "Run 'invoke rocisa' manually to refresh the bindings.",
            file=sys.stderr,
        )


@task(
    help={
        "clean": "Remove the client build directory before building.",
        "configure": "Run CMake configuration for the client.",
        "build": "Build the tensilelite-client executable.",
        "build_dir": "Path to client build dir.",
        "build_type": "CMake build type (e.g. Release, Debug).",
        "gpu_targets": "Comma-separated list of GPU targets (e.g. gfx90a,gfx1101).",
        "rocm_path": "Path to a ROCm install whose amdclang/amdclang++ should be used.",
        "export_compile_commands": "Enable CMAKE_EXPORT_COMPILE_COMMANDS.",
        "bundle_python_deps": "Enable HIPBLASLT_BUNDLE_PYTHON_DEPS.",
        "enable_rocprof": "Build tensilelite-client with rocprof.",
        "rebuild_rocisa": "Re-install the editable rocisa (if present) so rocisa C++ edits are picked up; pass --no-rebuild-rocisa to skip.",
    }
)
def build_client(
    c,
    clean=False,
    configure=True,
    build=True,
    build_dir="build_tmp",
    build_type="Release",
    gpu_targets=None,
    rocm_path=None,
    export_compile_commands=False,
    bundle_python_deps=False,
    enable_rocprof=False,
    rebuild_rocisa=True,
):
    """Build the tensilelite-client C++ executable.

    To run Tensile after building, use: Tensile/bin/Tensile <args>
    rocisa must first be installed via: invoke rocisa. After that, this task
    keeps the editable rocisa in sync — if it is installed editable, the
    bindings are rebuilt here so rocisa C++ edits are not silently stale
    (disable with --no-rebuild-rocisa).
    """

    if gpu_targets is None:
        gpu_targets = detect_gpu_arch()
        if not gpu_targets:
            print("Error: No GPU detected and no gpu_targets provided. Skipping build.")
            return
        print(f"warning: No GPU targets specified. Detected and using: {gpu_targets}")

    if rocm_path:
        cmake_c_compiler = os.path.join(rocm_path, "bin", "amdclang")
        cmake_cxx_compiler = os.path.join(rocm_path, "bin", "amdclang++")

        for compiler in (cmake_c_compiler, cmake_cxx_compiler):
            try:
                subprocess.run([compiler, "--version"], capture_output=True, timeout=5, check=True)
            except FileNotFoundError:
                print(f"Error: compiler not found at {compiler}", file=sys.stderr)
                return
            except subprocess.SubprocessError as e:
                print(f"Error: compiler check failed for {compiler}: {e}", file=sys.stderr)
                return

    if rebuild_rocisa:
        _maybe_rebuild_rocisa(c)

    if clean and os.path.exists(build_dir):
        c.run(f"rm -rf {shlex.quote(build_dir)}")

    if configure:
        os.makedirs(build_dir, exist_ok=True)

        cmake_cmd = [
            "cmake",
            "--preset",
            "tensilelite",
            "-S", str(_TASKS_DIR.parent),
            "-B", build_dir,
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DGPU_TARGETS={gpu_targets}",
            f"-DTENSILELITE_CLIENT_ENABLE_ROCPROFSDK={_cmake_bool(enable_rocprof)}",
        ]

        if rocm_path:
            cmake_cmd.append(f"-DCMAKE_C_COMPILER={cmake_c_compiler}")
            cmake_cmd.append(f"-DCMAKE_CXX_COMPILER={cmake_cxx_compiler}")
        if shutil.which("ccache"):
            cmake_cmd.append("-DCMAKE_C_COMPILER_LAUNCHER=ccache")
            cmake_cmd.append("-DCMAKE_CXX_COMPILER_LAUNCHER=ccache")
        if export_compile_commands:
            cmake_cmd.append("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
        cmake_cmd.append(f"-DHIPBLASLT_BUNDLE_PYTHON_DEPS={_cmake_bool(bundle_python_deps)}")

        c.run(shlex.join(cmake_cmd))

    if build:
        c.run(shlex.join(["cmake", "--build", build_dir, "--parallel"]))
