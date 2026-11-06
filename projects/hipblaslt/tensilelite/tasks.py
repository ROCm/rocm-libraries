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

# Architecture detection, steppings included, lives in the packaged TensileLite tree
# (invoke-free) so CI test artifacts can exercise it directly; these @task
# wrappers only expose it on the invoke command line.
from tensilelite.GpuArch import cmake_gpu_target, detect_gpu_arch


def _cmake_bool(value):
    return "ON" if value else "OFF"


def _cmake_gpu_targets(spec):
    """``GPU_TARGETS`` for cmake: keep steppings, drop colon-delimited features.

    Invoke accepts a comma-separated list. Each entry is passed through
    ``cmake_gpu_target`` so ``gfx942:sramecc+:xnack-`` (what tox forwards from
    an unstripped probe) becomes ``gfx942``.
    """
    return ",".join(
        cmake_gpu_target(part.strip()) for part in spec.split(",") if part.strip()
    )


def _resolve_gpu_targets(spec):
    if spec is None:
        detected = detect_gpu_arch()
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


@dataclasses.dataclass(frozen=True)
class RocmBuildEnvironment:
    """One selected ROCm installation for every native configure of a task.

    This is the conventional-prefix model of InvokeInstallPythonRocmSupport.md:
    the selected root supplies the compilers, ``ROCM_PATH`` and the CMake search
    prefix together, so neither an inherited CMake preset (``/opt/rocm``) nor an
    ambient ``ROCM_PATH`` can mix a second installation into the build. The
    Python ROCm SDK model described in that document is not implemented yet.
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
                "ROCM_PATH is set but empty; set it to a ROCm installation or pass "
                "--rocm-path.",
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


@task
def get_gpu_arch(c):
    detected = detect_gpu_arch()
    if detected:
        print(cmake_gpu_target(detected))


@task(
    help={
        "rocisa_dir": "Path to the rocisa source directory (default: rocisa/ next to this file).",
        "static": "Build rocisa's source StinkyTofu dependency static instead of shared.",
        "stinkytofu": "Install the standalone stinkytofu Python binding (the ROCISA_BACKEND=stinkytofu / gfx1250 backend). One of auto|on|off (default auto: install when a gfx1250 GPU is detected).",
        "rebuild_stinkytofu": "Force a fresh rebuild+reinstall of the stinkytofu binding even if it is already installed.",
        "rocm_path": "ROCm installation used to build rocisa and stinkytofu (default: ROCM_PATH, then rocm-sdk path --root, then /opt/rocm).",
    }
)
def rocisa(c, rocisa_dir=None, static=False, rocm_path=None,
           stinkytofu="auto", rebuild_stinkytofu=False):
    """Install rocisa editably for source development.

    This is a separate rocisa developer workflow for generator commands and
    compound development tasks. ``invoke build-client`` does not invoke it.
    Rerun it after changing rocisa's C++ sources, pyproject.toml, or
    CMakeLists.txt; importing rocisa never starts a native build.

    Pass --static to build its source StinkyTofu dependency static — useful for
    exercising the static-plugin path covered by
    rocisa/test/test_pass_plugin.py::TestHelloWorldPassIntegrationStatic.

    On gfx1250 this also builds and installs the standalone stinkytofu Python
    binding so `import rocisa` can route through the stinkytofu backend instead
    of silently falling back to native rocisa (control with
    --stinkytofu=auto|on|off).
    """
    build_env = _resolve_rocm_build_environment(rocm_path)
    _pip_install_rocisa(c, rocisa_dir, shared=not static, build_env=build_env)
    if _want_stinkytofu(stinkytofu):
        _pip_install_stinkytofu(c, force=rebuild_stinkytofu, build_env=build_env)


def _load_stinkytofu_tasks():
    """Import shared/stinkytofu/tasks.py without triggering its venv guard.

    The venv check was moved into build() so this import is side-effect-free.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stinkytofu_tasks",
        _TASKS_DIR.parent.parent.parent / "shared" / "stinkytofu" / "tasks.py",
    )


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


def _pip():
    """Invoke pip through the active interpreter."""
    return shlex.join([str(pathlib.Path(sys.executable).absolute()), "-m", "pip"])


def _want_stinkytofu(stinkytofu, targets=None):
    """Resolve the tri-state ``--stinkytofu`` option (auto/on/off) to a bool.

    ``auto`` installs the standalone stinkytofu binding when gfx1250 is present
    -- among ``targets`` when given, otherwise on the detected GPU.
    ``on``/``off`` force the decision.
    """
    val = str(stinkytofu).strip().lower()
    if val in ("on", "true", "yes", "1"):
        return True
    if val in ("off", "false", "no", "0"):
        return False
    if val in ("auto", "none", ""):
        arch = targets if targets is not None else (detect_gpu_arch() or "")
        return "gfx1250" in arch
    raise Exit(f"Invalid stinkytofu mode {stinkytofu!r}; use auto, on, or off", code=1)


def _pip_install_stinkytofu(c, force=False, build_env=None):
    """Build and install the standalone stinkytofu Python binding into the venv.

    On gfx1250 (or with ``ROCISA_BACKEND=stinkytofu``), ``import rocisa`` routes
    through the ``rocisa_stinkytofu_adaptor`` facade, which requires
    ``import stinkytofu`` to resolve the standalone ``_stinkytofu.so`` built from
    ``shared/stinkytofu/python_module``. Neither ``invoke rocisa`` (which only
    builds rocisa's *in-process* libstinkytofu, linked into ``_rocisa.so``) nor
    ``invoke build-client`` produces that standalone binding, so without this
    step ``_stinkytofu_available()`` fails and rocisa silently falls back to the
    native backend at kernel-build time.

    Build isolation is used on purpose: the binding's build backend
    (scikit-build-core / nanobind / cmake / ninja) is often absent from the
    active venv, where ``--no-build-isolation`` would fail.
    """
    repo_root = _TASKS_DIR.parents[2]
    src = repo_root / "shared" / "stinkytofu"
    if not (src / "pyproject.toml").is_file():
        print(
            f"warning: {src} not found; skipping stinkytofu backend install. "
            "rocisa will use the native backend.",
            file=sys.stderr,
        )
        return

    build_env = build_env or _resolve_rocm_build_environment()
    if not (build_env.cxx_compiler.exists() and build_env.c_compiler.exists()):
        print(
            f"warning: amdclang/amdclang++ not found under {build_env.root}; skipping "
            "stinkytofu backend install. rocisa will use the native backend.",
            file=sys.stderr,
        )
        return

    env = build_env.command_env()
    env["CMAKE_ARGS"] = shlex.join(build_env.cmake_args())
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 1))
    # stinkytofu is version-pinned, so pip skips rebuilding an existing install;
    # --force-reinstall keeps the binding matched to the current venv Python and
    # sources. --no-deps: the package declares no runtime dependencies.
    force_flag = "--force-reinstall --no-deps " if force else ""
    print("Building and installing the standalone stinkytofu Python binding "
          "(rocisa stinkytofu backend)...")
    c.run(f"{_pip()} install {force_flag}{shlex.quote(str(src))}", env=env)
    _pip_install_adaptor(c)


def _pip_install_adaptor(c):
    """Install the pure-Python rocisa_stinkytofu_adaptor package into the venv.

    ``import rocisa`` on gfx1250 switches to this package after ``stinkytofu``
    itself is importable. The source tree keeps it one directory deeper than
    the import name, and it is not part of the rocisa wheel, so pip must
    install that inner package explicitly.
    """
    src = _TASKS_DIR / "rocisa_stinkytofu_adaptor"
    if not (src / "pyproject.toml").is_file():
        print(
            f"warning: {src / 'pyproject.toml'} not found; skipping "
            "rocisa_stinkytofu_adaptor install.",
            file=sys.stderr,
        )
        return
    print("Installing rocisa_stinkytofu_adaptor (rocisa stinkytofu backend)...")
    c.run(f"{_pip()} install --no-deps --disable-pip-version-check {shlex.quote(str(src))}")


@task(
    help={
        "mode": "One of auto|on|off (default auto: install when gfx1250 is among --gpu-targets, or is the detected GPU when no targets are given).",
        "gpu_targets": "GPU targets the following client build and tests use; consulted by --mode=auto.",
        "rebuild": "Force a fresh rebuild+reinstall of the binding even if it is already installed.",
    }
)
def install_stinkytofu(c, mode="auto", gpu_targets=None, rebuild=False):
    """Install the standalone stinkytofu Python binding and the rocisa adaptor.

    gfx1250 kernel generation runs on the stinkytofu backend, and
    create_library refuses to emit a kernel when ``ROCISA_BACKEND=stinkytofu``
    is requested (or gfx1250 auto-selects it) without this binding loaded.
    ``invoke build-client`` only builds the C++ client and does not provision
    Python packages, so environments that generate gfx1250 kernels run this
    task before building the client.
    """
    if _want_stinkytofu(mode, targets=gpu_targets):
        _pip_install_stinkytofu(c, force=rebuild)


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
    refresh Python packages: provision rocisa with ``invoke rocisa`` and the
    gfx1250 stinkytofu binding with ``invoke install-stinkytofu``.

    To run Tensile after building, use: tensilelite/bin/Tensile <args>
    """

    if enable_asan and enable_tsan:
        raise Exit("Error: ASAN and TSAN cannot be enabled simultaneously", code=1)

    detected, gpu_targets = _resolve_gpu_targets(gpu_targets)
    if not gpu_targets:
        raise Exit("Error: No GPU detected and no gpu_targets provided", code=1)
    if detected is not None and detected != gpu_targets:
        print(f"warning: No GPU targets specified. Detected {detected}, using: {gpu_targets}")
    elif detected is not None:
        print(f"warning: No GPU targets specified. Detected and using: {gpu_targets}")

    # The selected root overrides the preset's /opt/rocm compilers and
    # CMAKE_PREFIX_PATH, so the client never mixes two installations.
    build_env = _resolve_rocm_build_environment(rocm_path)
    if rocm_path:
        for compiler in (build_env.c_compiler, build_env.cxx_compiler):
            try:
                subprocess.run([str(compiler), "--version"], capture_output=True, timeout=5, check=True)
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
        "stinkytofu": "Install the standalone stinkytofu Python binding (the ROCISA_BACKEND=stinkytofu / gfx1250 backend). One of auto|on|off (default auto: install when a gfx1250 GPU is detected).",
        "rebuild_stinkytofu": "Force a fresh rebuild+reinstall of the stinkytofu binding even if it is already installed.",
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
    stinkytofu="auto",
    rebuild_stinkytofu=False,
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
    # Resolve once; rocisa, the client, the editable install and the client
    # binding all receive this same root.
    build_env = _resolve_rocm_build_environment(rocm_path)
    rocm = build_env.root
    common_requirements = _TASKS_DIR / "requirements-dev-common.txt"
    if clean and os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    c.run(
        shlex.join(
            [str(python), "-m", "pip", "install", "-r", str(common_requirements)]
        )
    )
    rocisa.body(
        c,
        rocm_path=str(rocm),
        stinkytofu=stinkytofu,
        rebuild_stinkytofu=rebuild_stinkytofu,
    )
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

    env = build_env.command_env()
    env["TENSILELITE_ROCM_VERSION"] = _rocm_base_version(rocm)
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
    _, gpu_targets = _resolve_gpu_targets(gpu_targets)
    if not gpu_targets:
        print("Error: No GPU detected and no gpu_targets provided.")
        return

    if clean and os.path.exists(build_dir):
        c.run(f"rm -rf {shlex.quote(build_dir)}")

    os.makedirs(build_dir, exist_ok=True)

    build_env = _resolve_rocm_build_environment(rocm_path)

    cmake_cmd = [
        "cmake",
        "--preset", "tensilelite",
        "-S", "../",
        "-B", build_dir,
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
        cmake_cmd.extend([
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        ])

    c.run(shlex.join(cmake_cmd))
    c.run(shlex.join(["cmake", "--build", build_dir, "--parallel"]))

def _built_client_path(build_dir):
    executable = "tensilelite-client.exe" if sys.platform == "win32" else "tensilelite-client"
    return pathlib.Path(build_dir).resolve() / "tensilelite" / "client" / executable
