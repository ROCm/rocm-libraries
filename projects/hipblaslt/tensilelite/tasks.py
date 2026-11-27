# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

from invoke.exceptions import Exit
from invoke.tasks import task
import dataclasses
import os
import pathlib
import shlex
import shutil
import subprocess
import sys

_TASKS_DIR = pathlib.Path(__file__).parent.resolve()

# Task discovery and client-path must work before the generator is installed.
# Import GPU helpers only inside the tasks that need the generator.
from _tensilelite_client_binding import built_client_path as _built_client_path


def _cmake_bool(value):
    return "ON" if value else "OFF"


def _cmake_gpu_targets(spec):
    """``GPU_TARGETS`` for cmake: keep steppings, drop colon-delimited features.

    Invoke accepts a comma-separated list. Each entry is passed through
    ``cmake_gpu_target`` so ``gfx942:sramecc+:xnack-`` (what tox forwards from
    an unstripped probe) becomes ``gfx942``.
    """
    from tensilelite.GpuArch import cmake_gpu_target

    return ",".join(cmake_gpu_target(part.strip()) for part in spec.split(",") if part.strip())


def _resolve_gpu_targets(spec, build_env=None):
    if spec is None:
        from tensilelite.GpuArch import detect_gpu_arch

        detected = detect_gpu_arch(rocm_path=build_env.root) if build_env else detect_gpu_arch()
        if not detected:
            return None, None
        return detected, _cmake_gpu_targets(detected)
    return None, _cmake_gpu_targets(spec)


def _detect_rocm():
    """Detect ROCm installation path.

    Priority: ROCM_PATH env > rocm-sdk path --root > /opt/rocm.
    """
    env_path = os.environ.get("ROCM_PATH")
    if env_path:
        return env_path

    if shutil.which("rocm-sdk"):
        try:
            result = (
                subprocess.check_output(["rocm-sdk", "path", "--root"], stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
            if result:
                return result
        except subprocess.CalledProcessError:
            pass

    return "/opt/rocm"


@dataclasses.dataclass(frozen=True)
class RocmBuildEnvironment:
    """One selected ROCm installation for every native configure of a task.

    The selected root supplies the compilers, ``ROCM_PATH`` and the CMake search
    prefix together, so neither an inherited CMake preset (``/opt/rocm``) nor an
    ambient ``ROCM_PATH`` can mix a second installation into the build, and the
    client's recorded build ROCm (``tensilelite-client --build-rocm``) is the ROCm
    rocisa was built against too.
    """

    root: pathlib.Path
    cmake_prefix: pathlib.Path
    c_compiler: pathlib.Path
    cxx_compiler: pathlib.Path
    source: str

    def cmake_args(self, *, prefix: bool = True) -> list:
        args = [f"-DROCM_PATH={self.root}"]
        if prefix:
            args.append(f"-DCMAKE_PREFIX_PATH={self.cmake_prefix}")
        args += [
            f"-DCMAKE_C_COMPILER={self.c_compiler}",
            f"-DCMAKE_CXX_COMPILER={self.cxx_compiler}",
        ]
        return args

    def command_env(self, base=None) -> dict:
        """A subprocess environment naming the selected root; never os.environ itself."""
        env = dict(os.environ if base is None else base)
        env["ROCM_PATH"] = str(self.root)
        return env


def _rocm_tool(root: pathlib.Path, name: str) -> pathlib.Path:
    """A tool from the selected root: ``<root>/bin``, else ``<root>/lib/llvm/bin``."""
    for directory in (root / "bin", root / "lib" / "llvm" / "bin"):
        candidate = directory / name
        if candidate.exists():
            return candidate
    return root / "bin" / name


def _resolve_rocm_build_environment(rocm_path=None) -> RocmBuildEnvironment:
    """Select one ROCm root: ``--rocm-path``, then ``ROCM_PATH``, then a default.

    The default is ``rocm-sdk path --root`` when that command exists, then
    ``/opt/rocm`` -- the existing ``_detect_rocm`` fallback, kept until the Python
    ROCm SDK model replaces it. A set but empty ``ROCM_PATH`` is an error rather
    than a silent fallback.
    """
    if rocm_path:
        root, source = pathlib.Path(rocm_path), "--rocm-path"
    elif "ROCM_PATH" in os.environ:
        value = os.environ["ROCM_PATH"]
        if not value.strip():
            raise Exit(
                "ROCM_PATH is set but empty; set it to a ROCm installation or pass " "--rocm-path.",
                code=1,
            )
        root, source = pathlib.Path(value), "ROCM_PATH"
    else:
        detected = _detect_rocm()
        root = pathlib.Path(detected)
        source = "/opt/rocm" if detected == "/opt/rocm" else "rocm-sdk path --root"
    root = root.expanduser().absolute()
    return RocmBuildEnvironment(
        root=root,
        cmake_prefix=root,
        c_compiler=_rocm_tool(root, "amdclang"),
        cxx_compiler=_rocm_tool(root, "amdclang++"),
        source=source,
    )


def _report_rocm_build_environment(build_env):
    print(f"Using ROCm at {build_env.root} (from {build_env.source})")


@task
def get_gpu_arch(c):
    from tensilelite.GpuArch import cmake_gpu_target, detect_gpu_arch

    detected = detect_gpu_arch()
    if detected:
        print(cmake_gpu_target(detected))


@task(
    help={
        "rocisa_dir": "Path to the rocisa source directory (default: rocisa/ next to this file).",
        "static": "Build rocisa's source StinkyTofu dependency static instead of shared.",
        "rocm_path": "ROCm installation used to build rocisa (default: ROCM_PATH, then rocm-sdk path --root, then /opt/rocm).",
    }
)
def rocisa(c, rocisa_dir=None, static=False, rocm_path=None):
    """Install rocisa editably for source development.

    This is a separate rocisa developer workflow for generator commands and
    compound development tasks. ``invoke build-client`` does not invoke it.
    Rerun it after changing rocisa's C++ sources, pyproject.toml, or
    CMakeLists.txt; importing rocisa never starts a native build.

    Pass --static to build its source StinkyTofu dependency static — useful for
    exercising the static-plugin path covered by
    rocisa/test/test_pass_plugin.py::TestHelloWorldPassIntegrationStatic.
    """
    build_env = _resolve_rocm_build_environment(rocm_path)
    _report_rocm_build_environment(build_env)
    _install_rocisa(c, build_env, rocisa_dir, static)


def _install_rocisa(c, build_env, rocisa_dir=None, static=False):
    _pip_install_rocisa(c, rocisa_dir, shared=not static, build_env=build_env)


def _pip_install_rocisa(c, rocisa_dir=None, shared=True, rocm_path=None, build_env=None):
    """Editable-install rocisa via scikit-build-core.

    The selected root supplies ``ROCM_PATH`` and the compilers. No
    ``CMAKE_PREFIX_PATH`` is passed: scikit-build-core sets that itself (its
    site-packages entry is how nanobind is found), and rocisa's CMakeLists.txt
    appends ``ROCM_PATH`` to it ahead of ``/opt/rocm``.
    """
    src = pathlib.Path(rocisa_dir).resolve() if rocisa_dir else _TASKS_DIR / "rocisa"
    build_env = build_env or _resolve_rocm_build_environment(rocm_path)

    cmake_args = [
        *build_env.cmake_args(prefix=False),
        f"-DBUILD_SHARED_LIBS={_cmake_bool(shared)}",
        "-DROCISA_INCLUDE_BUILD_INFO=ON",
        "-DROCISA_BUILD_HELLOWORLD_STATIC_PLUGIN=ON",
    ]
    if shutil.which("ccache"):
        cmake_args += [
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        ]
    # scikit-build-core splits CMAKE_ARGS with shlex.
    env = build_env.command_env()
    env["CMAKE_ARGS"] = shlex.join(cmake_args)
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 1))
    c.run(
        shlex.join(
            [
                str(pathlib.Path(sys.executable).absolute()),
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "-e",
                str(src),
            ]
        ),
        env=env,
    )



@task(
    help={
        "clean": "Remove the client build directory before building.",
        "configure": "Run CMake configuration for the client.",
        "build": "Build the tensilelite-client executable.",
        "build_dir": "Path to client build dir.",
        "build_type": "CMake build type (e.g. Release, Debug).",
        "gpu_targets": "Comma-separated list of GPU targets (e.g. gfx90a,gfx1101).",
        "rocm_path": "ROCm installation for the compilers, ROCM_PATH and CMake packages (default: ROCM_PATH, then rocm-sdk path --root, then /opt/rocm).",
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

    This task only configures and builds the client. It does not install or
    refresh Python packages: provision rocisa with ``invoke rocisa``.

    To run Tensile after building, use: tensilelite/bin/Tensile <args>
    """
    build_env = _resolve_rocm_build_environment(rocm_path)
    _report_rocm_build_environment(build_env)
    _build_client(
        c,
        build_env,
        clean=clean,
        configure=configure,
        build=build,
        build_dir=build_dir,
        build_type=build_type,
        gpu_targets=gpu_targets,
        export_compile_commands=export_compile_commands,
        enable_rocprof=enable_rocprof,
        cxx_flags_release=cxx_flags_release,
        enable_asan=enable_asan,
        enable_tsan=enable_tsan,
        enable_sdma=enable_sdma,
    )


def _build_client(
    c,
    build_env,
    clean=False,
    configure=True,
    build=True,
    build_dir="build_tmp",
    build_type="Release",
    gpu_targets=None,
    export_compile_commands=False,
    enable_rocprof=False,
    cxx_flags_release=None,
    enable_asan=False,
    enable_tsan=False,
    enable_sdma=False,
):

    if enable_asan and enable_tsan:
        raise Exit("Error: ASAN and TSAN cannot be enabled simultaneously", code=1)

    for compiler in (build_env.c_compiler, build_env.cxx_compiler):
        try:
            subprocess.run(
                [str(compiler), "--version"],
                capture_output=True,
                timeout=5,
                check=True,
                env=build_env.command_env(),
            )
        except FileNotFoundError:
            raise Exit(f"Error: compiler not found at {compiler}", code=1)
        except (OSError, subprocess.SubprocessError) as e:
            raise Exit(f"Error: compiler check failed for {compiler}: {e}", code=1)

    detected, gpu_targets = _resolve_gpu_targets(gpu_targets, build_env)
    if not gpu_targets:
        raise Exit(
            f"Error: No GPU detected using ROCm at {build_env.root} and no gpu_targets provided",
            code=1,
        )
    if detected is not None and detected != gpu_targets:
        print(f"warning: No GPU targets specified. Detected {detected}, using: {gpu_targets}")
    elif detected is not None:
        print(f"warning: No GPU targets specified. Detected and using: {gpu_targets}")

    if clean and os.path.exists(build_dir):
        c.run(f"rm -rf {shlex.quote(build_dir)}")

    if configure:
        os.makedirs(build_dir, exist_ok=True)

        cmake_cmd = [
            "cmake",
            "--preset",
            "tensilelite",
            "-S",
            str(_TASKS_DIR.parent),
            "-B",
            build_dir,
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DGPU_TARGETS={gpu_targets}",
            f"-DTENSILELITE_CLIENT_ENABLE_ROCPROFSDK={_cmake_bool(enable_rocprof)}",
        ]

        if cxx_flags_release is not None:
            cmake_cmd.append(f"-DCMAKE_CXX_FLAGS_RELEASE={cxx_flags_release}")
        cmake_cmd.extend(build_env.cmake_args())
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


@task(help={"build_dir": "TensileLite CMake build directory."})
def client_path(c, build_dir="build_tmp"):
    """Print the absolute client output path without building or installing it."""
    print(_built_client_path(build_dir))


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
        "rocm_path": "ROCm installation used to build native components; runtime selects ROCm independently (default: ROCM_PATH, then rocm-sdk path --root, then /opt/rocm).",
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
    # Resolve once so rocisa and the client use the same native toolchain.
    # Python installation keeps caller env.
    build_env = _resolve_rocm_build_environment(rocm_path)
    _report_rocm_build_environment(build_env)
    ambient = os.environ.get("ROCM_PATH")
    if rocm_path and ambient and ambient.strip():
        runtime_root = pathlib.Path(ambient).expanduser().resolve()
        if runtime_root != build_env.root.resolve():
            print(
                f"warning: Building native components against ROCm at {build_env.root}, "
                f"but ROCM_PATH points to {runtime_root}. Conventional runtime selection "
                f"follows ROCM_PATH; set it to {build_env.root} before running TensileLite "
                "if you intend to use this build installation at runtime."
            )
    common_requirements = _TASKS_DIR / "requirements-dev-common.txt"
    if clean and os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    c.run(shlex.join([str(python), "-m", "pip", "install", "-r", str(common_requirements)]))
    _install_rocisa(c, build_env=build_env)
    _build_client(
        c,
        build_env=build_env,
        clean=False,
        build_dir=build_dir,
        build_type=build_type,
        gpu_targets=gpu_targets,
        export_compile_commands=export_compile_commands,
        enable_rocprof=enable_rocprof,
        cxx_flags_release=cxx_flags_release,
        enable_asan=enable_asan,
        enable_tsan=enable_tsan,
    )

    built_client = _built_client_path(build_dir)
    if not built_client.is_file() or not os.access(built_client, os.X_OK):
        raise Exit(f"Built tensilelite-client is missing or not executable: {built_client}", code=1)

    # The package build reads no ROCm environment; its version is +g<commit>.
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
        )
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
        )
    )
    print(f"TensileLite editable install uses: {built_client}")


@task
def precommit_install(c):
    """Install the hipblaslt/TensileLite git pre-commit hook (run once after `uv sync`).

    Clears core.hooksPath only when it points at the default hooks dir; bails if
    it points somewhere custom.
    """
    root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    common = subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"], text=True, cwd=root
    ).strip()
    common_path = pathlib.Path(common)
    if not common_path.is_absolute():
        common_path = (pathlib.Path(root) / common_path).resolve()
    default_hooks = str(common_path / "hooks")
    hooks_path = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"],
        cwd=root,
        capture_output=True,
        text=True,
    ).stdout.strip()

    if hooks_path:
        if os.path.realpath(hooks_path) == os.path.realpath(default_hooks):
            print(
                f"core.hooksPath is set to the default ({hooks_path}); clearing it "
                "(redundant; git-lfs hooks are unaffected)."
            )
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
    build_env = _resolve_rocm_build_environment(rocm_path)
    _report_rocm_build_environment(build_env)
    _, gpu_targets = _resolve_gpu_targets(gpu_targets, build_env)
    if not gpu_targets:
        print(f"Error: No GPU detected using ROCm at {build_env.root} and no gpu_targets provided.")
        return

    if clean and os.path.exists(build_dir):
        c.run(f"rm -rf {shlex.quote(build_dir)}")

    os.makedirs(build_dir, exist_ok=True)

    cmake_cmd = [
        "cmake",
        "--preset",
        "tensilelite",
        "-S",
        "../",
        "-B",
        build_dir,
        "-DCMAKE_BUILD_TYPE=Debug",
        f"-DGPU_TARGETS={gpu_targets}",
        *build_env.cmake_args(),
        "-DTENSILELITE_ENABLE_COVERAGE=ON",
        "-DROCISA_ENABLE_COVERAGE=ON",
        "-DROCISA_BUILD_PYTHON=ON",
        "-DTENSILELITE_BUILD_TESTING=ON",
        "-DHIPBLASLT_ENABLE_YAML=OFF",  # Use msgpack, LLVM headers may not be available
    ]

    if shutil.which("ccache"):
        cmake_cmd.extend(
            [
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            ]
        )

    c.run(shlex.join(cmake_cmd))
    c.run(shlex.join(["cmake", "--build", build_dir, "--parallel"]))
