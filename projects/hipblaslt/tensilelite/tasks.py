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

# gfx1250 v0/v1 ASIC-revision detection lives in the packaged tensilelite tree
# (invoke-free) so CI test artifacts can exercise it directly; these @task
# wrappers only expose it on the invoke command line.
from tensilelite.GpuRevisionTarget import detect_gpu_arch, detect_gpu_revision_target


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

def _rocm_base_version(rocm_root: pathlib.Path) -> str:
    version_file = rocm_root / ".info" / "version"
    try:
        version = version_file.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise Exit(
            f"Selected ROCm root has no readable version metadata: {version_file}",
            code=1,
        ) from exc
    if not version:
        raise Exit(f"Selected ROCm version metadata is empty: {version_file}", code=1)
    return version
@task
def get_gpu_arch(c):
    print(detect_gpu_arch())


@task
def get_gpu_revision_target(c):
    """Print the TensileLite --gpu-targets value, split by gfx1250 v0/v1 revision."""
    print(detect_gpu_revision_target())

@task(
    help={
        "rocisa_dir": "Path to the rocisa source directory (default: rocisa/ next to this file).",
        "stinkytofu_prefix": "Install prefix for the stinkytofu build (default: build_tmp/stinkytofu-install).",
        "static": "Build stinkytofu static (BUILD_SHARED_LIBS=OFF) instead of the default shared build.",
        "rocm_path": "ROCm installation used to build rocisa and stinkytofu.",
    }
)
def rocisa(c, rocisa_dir=None, stinkytofu_prefix=None, static=False, rocm_path=None):
    """Install rocisa editably for source development.

    This is a separate rocisa developer workflow. TensileLite packaging and
    ``invoke build-client`` consume an already importable rocisa and do not call
    this task or make decisions about rocisa's release packaging.

    Pass --static to build stinkytofu static instead of shared — useful for
    exercising the static-plugin path covered by
    rocisa/test/test_pass_plugin.py::TestHelloWorldPassIntegrationStatic.
    """
    _pip_install_rocisa(
        c, rocisa_dir, stinkytofu_prefix, shared=not static, rocm_path=rocm_path
    )


def _load_stinkytofu_tasks():
    """Import shared/stinkytofu/tasks.py without triggering its venv guard."""
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
    """Build the rocisa development dependency into a private prefix.

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


def _pip_install_rocisa(
    c, rocisa_dir=None, stinkytofu_prefix=None, shared=True, rocm_path=None
):
    """Build and editable-install rocisa for its standalone developer flow.

    Builds stinkytofu and installs it to stinkytofu_prefix (default:
    build_tmp/stinkytofu-install next to this file) so rocisa's CMake finds it
    via find_package(stinkytofu) — the same path TheRock uses. This exercises
    the installed package layout (stinkytofuConfig.cmake, exported targets) so
    breakage is caught early in the dev/CI workflow.

    shared=False builds and links a static stinkytofu instead of the default
    shared library.
    """
    src = pathlib.Path(rocisa_dir).resolve() if rocisa_dir else _TASKS_DIR / "rocisa"
    rocm = rocm_path or _detect_rocm()

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
        "enable_rocprof": "Build tensilelite-client with rocprof.",
        "cxx_flags_release": "Override CMAKE_CXX_FLAGS_RELEASE (for example, -O3 to keep asserts enabled in Release).",
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
    enable_rocprof=False,
    cxx_flags_release=None,
    enable_asan=False,
    enable_tsan=False,
    enable_sdma=False,
):
    """Build the tensilelite-client C++ executable.

    rocisa must already be importable in the invoking Python environment.
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

        c.run(shlex.join(cmake_cmd))

    if build:
        c.run(shlex.join(["cmake", "--build", build_dir, "--parallel"]))

@task(help={"build_dir": "CMake build directory containing tensilelite-client."})
def configure_client(c, build_dir="build_tmp"):
    """Bind this Python installation to a previously built client."""
    client = _built_client_path(build_dir)
    c.run(
        shlex.join(
            [
                sys.executable,
                "-m",
                "tensilelite_configure_client",
                "--ensure-client",
                str(client),
            ]
        )
    )

@task(
    help={
        "clean": "Remove the client build directory before building.",
        "build_dir": "Path to the client build directory.",
        "build_type": "CMake build type (for example Release or Debug).",
        "gpu_targets": "Comma-separated GPU targets; detected when omitted.",
        "rocm_path": "ROCm installation used to build and version the package.",
        "export_compile_commands": "Enable CMAKE_EXPORT_COMPILE_COMMANDS.",
        "enable_rocprof": "Build tensilelite-client with rocprof.",
        "cxx_flags_release": "Override CMAKE_CXX_FLAGS_RELEASE.",
        "enable_asan": "Enable AddressSanitizer.",
        "enable_tsan": "Enable ThreadSanitizer.",
    }
)
def install(
    c,
    clean=False,
    build_dir="build_tmp",
    build_type="Release",
    gpu_targets=None,
    rocm_path=None,
    export_compile_commands=False,
    enable_rocprof=False,
    cxx_flags_release=None,
    enable_asan=False,
    enable_tsan=False,
):
    """Build rocisa/client and install TensileLite editably into this Python."""
    if sys.platform != "linux":
        raise Exit(
            "invoke install is supported only in the Linux ROCm development environment; "
            "use the manual CMake and pip workflow on other platforms.",
            code=1,
        )

    python = pathlib.Path(sys.executable).absolute()
    selected_rocm = rocm_path or os.environ.get("ROCM_PATH") or "/opt/rocm"
    rocm = pathlib.Path(selected_rocm).expanduser().absolute()
    common_requirements = _TASKS_DIR / "requirements-dev-common.txt"
    if clean and os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    c.run(
        shlex.join(
            [str(python), "-m", "pip", "install", "-r", str(common_requirements)]
        )
    )
    rocisa.body(c, rocm_path=str(rocm))
    build_client.body(
        c,
        clean=False,
        build_dir=build_dir,
        build_type=build_type,
        gpu_targets=gpu_targets,
        rocm_path=str(rocm),
        export_compile_commands=export_compile_commands,
        enable_rocprof=enable_rocprof,
        cxx_flags_release=cxx_flags_release,
        enable_asan=enable_asan,
        enable_tsan=enable_tsan,
    )

    built_client = _built_client_path(build_dir)
    if not built_client.is_file() or not os.access(built_client, os.X_OK):
        raise Exit(f"Built tensilelite-client is missing or not executable: {built_client}", code=1)

    env = dict(
        os.environ,
        ROCM_PATH=str(rocm),
        TENSILELITE_ROCM_VERSION=_rocm_base_version(rocm),
    )
    c.run(
        shlex.join(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "--no-deps",
                "--editable",
                str(_TASKS_DIR),
            ]
        ),
        env=env,
    )
    c.run(
        shlex.join(
            [
                str(python),
                "-m",
                "tensilelite_configure_client",
                "--client",
                str(built_client),
            ]
        ),
        env=env,
    )
    print(f"TensileLite editable install uses: {built_client}")
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
        "-DROCISA_BUILD_PYTHON=ON",
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

def _built_client_path(build_dir):
    executable = "tensilelite-client.exe" if sys.platform == "win32" else "tensilelite-client"
    return pathlib.Path(build_dir).resolve() / "tensilelite" / "client" / executable
