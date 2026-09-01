# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

from invoke.exceptions import Exit
from invoke.tasks import task
import os
import pathlib
import shlex
import shutil
import subprocess
import sys

_TASKS_DIR = pathlib.Path(__file__).parent.resolve()

# Ensure the Tensile package (shipped next to this file) is importable when
# invoke runs from the tensilelite root, regardless of cwd/sys.path state.
if str(_TASKS_DIR) not in sys.path:
    sys.path.insert(0, str(_TASKS_DIR))

from Tensile.RocisaStatus import _rocisa_install_status

# gfx1250 v0/v1 ASIC-revision detection lives in the packaged Tensile tree
# (invoke-free) so CI test artifacts can exercise it directly; these @task
# wrappers only expose it on the invoke command line.
from Tensile.GpuRevisionTarget import detect_gpu_arch, detect_gpu_revision_target


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


@task
def get_gpu_arch(c):
    print(detect_gpu_arch())


@task
def get_gpu_revision_target(c):
    """Print the Tensile --gpu-targets value, split by gfx1250 v0/v1 revision."""
    print(detect_gpu_revision_target())

@task(
    help={
        "rocisa_dir": "Path to the rocisa source directory (default: rocisa/ next to this file).",
        "stinkytofu_prefix": "Install prefix for the stinkytofu build (default: build_tmp/stinkytofu-install).",
        "static": "Build stinkytofu static (BUILD_SHARED_LIBS=OFF) instead of the default shared build.",
    }
)
def rocisa(c, rocisa_dir=None, stinkytofu_prefix=None, static=False):
    """Install rocisa as an editable pip package.

    Not required before `invoke build-client` — the client build enables
    HIPBLASLT_BUNDLE_PYTHON_DEPS automatically when rocisa is absent. Run
    this task to make rocisa importable system-wide (outside the build
    directory), or after changes to rocisa's pyproject.toml or CMakeLists.txt.

    Builds and installs stinkytofu locally first so rocisa uses
    find_package(stinkytofu) — mirroring how TheRock wires the two together.

    Pass --static to build stinkytofu static instead of shared — useful for
    exercising the static-plugin path covered by
    rocisa/test/test_pass_plugin.py::TestHelloWorldPassIntegrationStatic.
    """
    _pip_install_rocisa(c, rocisa_dir, stinkytofu_prefix, shared=not static)


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


def _build_and_install_stinkytofu(
    c, install_prefix: pathlib.Path, rocm: str, shared: bool = True
) -> None:
    """Build stinkytofu and install it to install_prefix so rocisa can find_package it.

    Build flags come from stinkytofu_tasks.cmake_build_args() — the single source
    of truth — so a new required cmake option only needs to be added there.
    Compiler selection mirrors shared/stinkytofu/tasks.py `invoke build`.
    cmake is incremental, so repeat calls are a fast no-op when nothing changed.

    shared=False builds stinkytofu static (BUILD_SHARED_LIBS=OFF) instead of
    the default shared library.
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
        # amd_comgr lives in the SDK venv (off the loader path), so bake its dir
        # into the installed libstinkytofu RPATH for this dev/standalone build.
        "-DSTINKYTOFU_INSTALL_RPATH_USE_LINK_PATH=ON",
        # tests/python OFF for the rocisa integration build; examples ON (default).
        *st.cmake_build_args(
            install_prefix=install_prefix, tests=False, python=False, shared=shared
        ),
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


def _pip_install_rocisa(c, rocisa_dir=None, stinkytofu_prefix=None, shared=True):
    """Editable-install rocisa via scikit-build-core.

    Factored out of the `rocisa` task so `build_client` can reuse it to keep
    the editable install fresh.

    Builds stinkytofu and installs it to stinkytofu_prefix (default:
    build_tmp/stinkytofu-install next to this file) so rocisa's CMake finds it
    via find_package(stinkytofu) — the same path TheRock uses. This exercises
    the installed package layout (stinkytofuConfig.cmake, exported targets) so
    breakage is caught early in the dev/CI workflow.

    shared=False builds and links a static stinkytofu instead of the default
    shared library.
    """
    src = pathlib.Path(rocisa_dir).resolve() if rocisa_dir else _TASKS_DIR / "rocisa"
    rocm = _detect_rocm()

    prefix = (
        pathlib.Path(stinkytofu_prefix).resolve()
        if stinkytofu_prefix
        else _TASKS_DIR / "build_tmp" / "stinkytofu-install"
    )
    _build_and_install_stinkytofu(c, prefix, rocm, shared=shared)

    cmake_args = (
        f"-DROCM_PATH={rocm}"
        f" -DROCISA_INCLUDE_BUILD_INFO=ON"
        # Compiles the HelloWorldPass example plugin directly into _rocisa so
        # TestHelloWorldPassIntegrationStatic can exercise the compiled-in
        # plugin path regardless of whether stinkytofu above is shared or static.
        f" -DROCISA_BUILD_HELLOWORLD_STATIC_PLUGIN=ON"
    )
    if shutil.which("ccache"):
        cmake_args += " -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    env = dict(os.environ, CMAKE_ARGS=cmake_args)
    # Append (don't clobber) the stinkytofu install prefix so find_package
    # resolves it, while preserving the CMAKE_PREFIX_PATH that scikit-build-core
    # injects for nanobind. find_package searches the env var and the cache var.
    _existing_prefix = env.get("CMAKE_PREFIX_PATH")
    env["CMAKE_PREFIX_PATH"] = (
        f"{prefix}{os.pathsep}{_existing_prefix}" if _existing_prefix else str(prefix)
    )
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 1))
    c.run(f"pip install --no-build-isolation -e {shlex.quote(str(src))}", env=env)


def _maybe_rebuild_rocisa(c, rocisa_dir=None):
    """Refresh the editable rocisa so `import rocisa` picks up C++ edits.

    Only acts when rocisa is installed editable (pip install -e). When
    absent, rocisa is built by CMake via HIPBLASLT_BUNDLE_PYTHON_DEPS.
    When non-editable (e.g. tox), does nothing.

    Degrades to a warning — never a hard failure — when the build backend
    (scikit-build-core / nanobind) is unavailable.
    """
    import importlib.util

    if _rocisa_install_status() != "editable":
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
        "bundle_python_deps": "Force HIPBLASLT_BUNDLE_PYTHON_DEPS on or off; auto-enabled when rocisa is not pip-installed.",
        "enable_rocprof": "Build tensilelite-client with rocprof.",
        "cxx_flags_release": "Override CMAKE_CXX_FLAGS_RELEASE (for example, -O3 to keep asserts enabled in Release).",
        "rebuild_rocisa": "Re-install the editable rocisa (if present) so rocisa C++ edits are picked up; pass --no-rebuild-rocisa to skip.",
        "enable_asan": "Enable AddressSanitizer.",
        "enable_tsan": "Enable ThreadSanitizer.",
        "enable_sdma": "Build the GPU-initiated SDMA transport path; needs hsakmt and hsa-runtime64.",
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
    cxx_flags_release=None,
    rebuild_rocisa=True,
    enable_asan=False,
    enable_tsan=False,
    enable_sdma=False,
):
    """Build the tensilelite-client C++ executable.

    To run Tensile after building, use: Tensile/bin/Tensile <args>
    When rocisa is not pip-installed, HIPBLASLT_BUNDLE_PYTHON_DEPS is
    enabled automatically so CMake builds it in the client build
    directory. When rocisa is installed editable, the bindings are
    refreshed to pick up C++ edits (disable with --no-rebuild-rocisa).
    """

    if enable_asan and enable_tsan:
        raise Exit("Error: ASAN and TSAN cannot be enabled simultaneously", code=1)

    if gpu_targets is None:
        gpu_targets = detect_gpu_arch()
        if not gpu_targets:
            raise Exit("Error: No GPU detected and no gpu_targets provided", code=1)
        print(f"warning: No GPU targets specified. Detected and using: {gpu_targets}")

    if rocm_path:
        cmake_c_compiler = os.path.join(rocm_path, "bin", "amdclang")
        cmake_cxx_compiler = os.path.join(rocm_path, "bin", "amdclang++")

        for compiler in (cmake_c_compiler, cmake_cxx_compiler):
            try:
                subprocess.run([compiler, "--version"], capture_output=True, timeout=5, check=True)
            except FileNotFoundError:
                raise Exit(f"Error: compiler not found at {compiler}", code=1)
            except subprocess.SubprocessError as e:
                raise Exit(f"Error: compiler check failed for {compiler}: {e}", code=1)

    if rebuild_rocisa:
        _maybe_rebuild_rocisa(c)

    if not bundle_python_deps and _rocisa_install_status() == "absent":
        print("rocisa is not pip-installed; enabling HIPBLASLT_BUNDLE_PYTHON_DEPS=ON "
              "so CMake builds it in the client build directory.")
        bundle_python_deps = True

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

        if cxx_flags_release is not None:
            cmake_cmd.append(f"-DCMAKE_CXX_FLAGS_RELEASE={cxx_flags_release}")
        if rocm_path:
            cmake_cmd.append(f"-DCMAKE_C_COMPILER={cmake_c_compiler}")
            cmake_cmd.append(f"-DCMAKE_CXX_COMPILER={cmake_cxx_compiler}")
        if shutil.which("ccache"):
            cmake_cmd.append("-DCMAKE_C_COMPILER_LAUNCHER=ccache")
            cmake_cmd.append("-DCMAKE_CXX_COMPILER_LAUNCHER=ccache")
        if export_compile_commands:
            cmake_cmd.append("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
        if enable_asan:
            cmake_cmd.append("-DTENSILELITE_ENABLE_HOST_ASAN=ON")
        if enable_tsan:
            cmake_cmd.append("-DTENSILELITE_ENABLE_HOST_TSAN=ON")
        if enable_sdma:
            cmake_cmd.append("-DTENSILELITE_ENABLE_SDMA=ON")
        cmake_cmd.append(f"-DHIPBLASLT_BUNDLE_PYTHON_DEPS={_cmake_bool(bundle_python_deps)}")

        c.run(shlex.join(cmake_cmd))

    if build:
        c.run(shlex.join(["cmake", "--build", build_dir, "--parallel"]))


@task
def precommit_install(c):
    """Install the hipblaslt/TensileLite git pre-commit hook (run once after `uv sync`).

    Clears core.hooksPath only when it points at the default hooks dir; bails if
    it points somewhere custom.
    """
    root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    common = subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"], text=True, cwd=root
    ).strip()
    common_path = pathlib.Path(common)
    if not common_path.is_absolute():
        common_path = (pathlib.Path(root) / common_path).resolve()
    default_hooks = str(common_path / "hooks")
    hooks_path = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"],
        cwd=root, capture_output=True, text=True,
    ).stdout.strip()

    if hooks_path:
        if os.path.realpath(hooks_path) == os.path.realpath(default_hooks):
            print(f"core.hooksPath is set to the default ({hooks_path}); clearing it "
                  "(redundant; git-lfs hooks are unaffected).")
            with c.cd(root):
                c.run("git config --unset-all core.hooksPath")
        else:
            raise Exit(
                f"Refusing to install: core.hooksPath is set to a custom path "
                f"({hooks_path}), not the default ({default_hooks}). Resolve that "
                "first.",
                code=1,
            )

    config = "projects/hipblaslt/.pre-commit-config.yaml"
    with c.cd(root):
        c.run(f"pre-commit install --config {shlex.quote(config)}", pty=True)


@task(
    help={
        "build_dir": "Path to coverage build dir.",
        "gpu_targets": "GPU targets (e.g. gfx90a,gfx942).",
        "rocm_path": "Path to ROCm installation.",
        "clean": "Remove build directory before building.",
    }
)
def build_coverage(
    c,
    build_dir="build_cov",
    gpu_targets=None,
    rocm_path=None,
    clean=False,
):
    """Build TensileLite with code coverage instrumentation.

    Builds rocisa, tensilelite-host, and client with LLVM coverage flags.
    Run tests with tox -e coverage-cpp to generate coverage reports.
    """
    if gpu_targets is None:
        gpu_targets = detect_gpu_arch()
        if not gpu_targets:
            print("Error: No GPU detected and no gpu_targets provided.")
            return

    if clean and os.path.exists(build_dir):
        c.run(f"rm -rf {shlex.quote(build_dir)}")

    os.makedirs(build_dir, exist_ok=True)

    rocm = rocm_path or _detect_rocm()
    cmake_c = os.path.join(rocm, "bin", "amdclang")
    cmake_cxx = os.path.join(rocm, "bin", "amdclang++")

    cmake_cmd = [
        "cmake",
        "--preset", "tensilelite",
        "-S", "../",
        "-B", build_dir,
        "-DCMAKE_BUILD_TYPE=Debug",
        f"-DGPU_TARGETS={gpu_targets}",
        f"-DCMAKE_C_COMPILER={cmake_c}",
        f"-DCMAKE_CXX_COMPILER={cmake_cxx}",
        "-DTENSILELITE_ENABLE_COVERAGE=ON",
        "-DROCISA_ENABLE_COVERAGE=ON",
        "-DTENSILELITE_BUILD_TESTING=ON",
        "-DHIPBLASLT_ENABLE_YAML=OFF",  # Use msgpack, LLVM headers may not be available
    ]

    if shutil.which("ccache"):
        cmake_cmd.extend([
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        ])

    c.run(shlex.join(cmake_cmd))
    c.run(shlex.join(["cmake", "--build", build_dir, "--parallel"]))
