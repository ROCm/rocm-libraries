# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
from pathlib import Path
import os
import importlib.util
import shlex
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit
_SOURCE_ROOT = Path(__file__).resolve().parents[4]


def _load_tensilelite_tasks():
    spec = importlib.util.spec_from_file_location("tensilelite_tasks", _SOURCE_ROOT / "tasks.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_hipblaslt_tasks():
    spec = importlib.util.spec_from_file_location(
        "hipblaslt_tasks", _SOURCE_ROOT.parent / "tasks.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tasks = _load_tensilelite_tasks()
hipblaslt_tasks = _load_hipblaslt_tasks()


def test_invoke_install_is_a_discoverable_developer_workflow():
    result = subprocess.run(
        ["invoke", "--help", "install"],
        cwd=_SOURCE_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--build-dir" in result.stdout
    assert "--gpu-targets" in result.stdout
    assert "--rocm-path" in result.stdout


def test_install_binds_the_actual_cmake_client_output(tmp_path):
    executable = "tensilelite-client.exe" if sys.platform == "win32" else "tensilelite-client"
    expected = tmp_path / "build/tensilelite/client" / executable
    assert tasks._built_client_path(tmp_path / "build") == expected


@pytest.mark.parametrize(
    ("platform", "name"),
    [("linux", "tensilelite-client"), ("win32", "tensilelite-client.exe")],
)
def test_client_path_task_reports_build_output(tmp_path, monkeypatch, capsys, platform, name):
    import _tensilelite_client_binding as binding

    monkeypatch.setattr(binding, "sys", SimpleNamespace(platform=platform))
    build_dir = tmp_path / "build with spaces"
    tasks.client_path.body(None, build_dir=str(build_dir))
    assert capsys.readouterr().out == str(build_dir / "tensilelite" / "client" / name) + "\n"


def test_client_path_command_does_not_import_the_generator(tmp_path):
    script = """
import importlib.abc
import sys
from invoke import Program

class NoGenerator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tensilelite" or fullname.startswith("tensilelite."):
            raise ImportError("client-path must work before the generator is installed")

sys.meta_path.insert(0, NoGenerator())
Program().run(argv=["invoke", "client-path", "--build-dir", sys.argv[1]])
"""
    build_dir = tmp_path / "build with spaces"
    result = subprocess.run(
        [sys.executable, "-c", script, str(build_dir)],
        cwd=_SOURCE_ROOT,
        capture_output=True,
        text=True,
    )
    executable = "tensilelite-client.exe" if sys.platform == "win32" else "tensilelite-client"
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(build_dir / "tensilelite" / "client" / executable) + "\n"


def test_rocisa_install_uses_the_invoking_python_and_selected_rocm(tmp_path, monkeypatch):
    commands = []

    class Context:
        def run(self, command, **kwargs):
            commands.append((shlex.split(command), kwargs))

    monkeypatch.setattr(tasks.shutil, "which", lambda name: None)
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "other-rocm"))

    rocm_root = tmp_path / "rocm"
    source = tmp_path / "rocisa"
    source.mkdir()
    tasks._pip_install_rocisa(
        Context(),
        rocisa_dir=source,
        rocm_path=str(rocm_root),
    )

    assert len(commands) == 1
    command, kwargs = commands[0]
    assert command[:4] == [
        str(Path(sys.executable).absolute()),
        "-m",
        "pip",
        "install",
    ]
    assert f"-DROCM_PATH={rocm_root}" in kwargs["env"]["CMAKE_ARGS"].split()


def test_build_client_forwards_the_selected_rocm_root(tmp_path, monkeypatch):
    class RecordingContext:
        def __init__(self):
            self.commands = []

        def run(self, command):
            self.commands.append(command)

    rocm_root = tmp_path / "rocm"
    compiler_dir = rocm_root / "bin"
    compiler_dir.mkdir(parents=True)
    monkeypatch.setattr(tasks.subprocess, "run", lambda *args, **kwargs: None)
    context = RecordingContext()

    tasks.build_client.body(
        context,
        build_dir=str(tmp_path / "build"),
        gpu_targets="gfx942",
        rocm_path=str(rocm_root),
        build=False,
    )

    configure_command = context.commands[0]
    assert f"-DROCM_PATH={rocm_root}" in configure_command
    assert f"-DCMAKE_C_COMPILER={compiler_dir / 'amdclang'}" in configure_command
    assert f"-DCMAKE_CXX_COMPILER={compiler_dir / 'amdclang++'}" in configure_command

def test_build_client_does_not_configure_rocisa_python_extension(tmp_path):
    """The focused client build must not select the code-generation extension."""

    class RecordingContext:
        def __init__(self):
            self.commands = []

        def run(self, command):
            self.commands.append(command)

    context = RecordingContext()
    tasks.build_client.body(
        context,
        build_dir=str(tmp_path / "build"),
        gpu_targets="gfx942",
        build=False,
    )

    assert all(
        "HIPBLASLT_BUNDLE_PYTHON_DEPS" not in command
        and "ROCISA_BUILD_PYTHON" not in command
        for command in context.commands
    )


class TestBuildTaskCommandLine:
    """invoke assigns short flags in signature order, so a new parameter's
    position is part of the interface: placed too early it steals a letter."""

    def _short_flags(self, flag):
        from invoke.parser import Context as ParserContext

        context = ParserContext(
            name="build", args=hipblaslt_tasks.build.get_arguments()
        )
        return context.flags[flag].nicknames

    @pytest.mark.parametrize(
        "flag,short",
        [
            ("--logic-filter", "f"),  # the first casualty if -g is stolen
            ("--gprof", "g"),
            ("--architecture", "a"),
        ],
    )
    def test_existing_short_flags_are_unchanged(self, flag, short):
        assert self._short_flags(flag) == (short,)


@pytest.mark.skipif(sys.platform != "linux", reason="invoke install is Linux-only")
def test_install_uses_one_selected_rocm_root_and_binds_the_built_client(tmp_path, monkeypatch):
    rocm_root = (tmp_path / "rocm").resolve()
    rocm_root.mkdir()
    build_dir = tmp_path / "build"
    client = tasks._built_client_path(build_dir)
    client.parent.mkdir(parents=True)
    client.write_text("client", encoding="utf-8")
    client.chmod(0o755)

    calls = []

    def record_rocisa(context, **kwargs):
        calls.append(("rocisa", kwargs))

    def record_build_client(context, **kwargs):
        calls.append(("build-client", kwargs))

    class Context:
        def run(self, command, **kwargs):
            calls.append(("run", command, kwargs))

    monkeypatch.setenv("ROCM_PATH", str(rocm_root))
    monkeypatch.setattr(tasks, "_install_rocisa", record_rocisa)
    monkeypatch.setattr(tasks, "_build_client", record_build_client)

    tasks.install.body(Context(), build_dir=str(build_dir))

    requirements_command = shlex.split(calls[0][1])
    assert requirements_command == [
        str(Path(sys.executable).absolute()),
        "-m",
        "pip",
        "install",
        "-r",
        str(_SOURCE_ROOT / "requirements-dev-common.txt"),
    ]
    assert calls[0][2] == {}
    assert calls[1] == (
        "rocisa",
        {
            "build_env": tasks._resolve_rocm_build_environment(),
        },
    )
    assert calls[2][0] == "build-client"
    assert calls[2][1]["build_env"] is calls[1][1]["build_env"]
    assert calls[2][1]["build_env"].root == rocm_root
    editable_command = shlex.split(calls[3][1])
    assert editable_command[-4:] == [
        "--no-build-isolation",
        "--no-deps",
        "--editable",
        str(_SOURCE_ROOT),
    ]
    # The package build reads no ROCm environment (its version is +g<commit>),
    # so the editable install and the binding step run in the caller's
    # environment unchanged.
    assert calls[3][2] == {}
    assert calls[4][2] == {}
    assert shlex.split(calls[4][1]) == [
        str(Path(sys.executable).absolute()),
        "-m",
        "tensilelite_configure_client",
        "--client",
        str(client),
    ]


@pytest.mark.skipif(sys.platform != "linux", reason="invoke install is Linux-only")
@pytest.mark.parametrize("create_non_executable", [False, True])
def test_install_rejects_an_invalid_built_client(tmp_path, monkeypatch, create_non_executable):
    rocm_root = (tmp_path / "rocm").resolve()
    rocm_root.mkdir()
    if create_non_executable:
        client = tasks._built_client_path(tmp_path / "build")
        client.parent.mkdir(parents=True)
        client.write_text("client", encoding="utf-8")

    class Context:
        def run(self, command, **kwargs):
            pass

    monkeypatch.setenv("ROCM_PATH", str(rocm_root))
    monkeypatch.setattr(tasks, "_install_rocisa", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks, "_build_client", lambda *args, **kwargs: None)

    with pytest.raises(tasks.Exit, match="Built tensilelite-client is missing or not executable"):
        tasks.install.body(Context(), build_dir=str(tmp_path / "build"))


# ---------------------------------------------------------------------------
# One selected ROCm root for every native configure. These are command-capture
# tests: none of them needs a ROCm installation.
# ---------------------------------------------------------------------------


def _fake_rocm_root(tmp_path, name, *, version="7.2.4"):
    root = (tmp_path / name).resolve()
    (root / "bin").mkdir(parents=True)
    for tool in ("amdclang", "amdclang++"):
        (root / "bin" / tool).write_text("", encoding="utf-8")
    (root / ".info").mkdir()
    (root / ".info" / "version").write_text(f"{version}\n", encoding="utf-8")
    return root


def _selected_root_args(root):
    return [
        f"-DROCM_PATH={root}",
        f"-DCMAKE_PREFIX_PATH={root}",
        f"-DCMAKE_C_COMPILER={root / 'bin' / 'amdclang'}",
        f"-DCMAKE_CXX_COMPILER={root / 'bin' / 'amdclang++'}",
    ]


class _RecordingContext:
    def __init__(self):
        self.commands = []

    def run(self, command, **kwargs):
        self.commands.append((command, kwargs))


def test_rocm_build_environment_prefers_explicit_root_over_ambient_rocm_path(tmp_path, monkeypatch):
    selected = _fake_rocm_root(tmp_path, "selected")
    monkeypatch.setenv("ROCM_PATH", str(_fake_rocm_root(tmp_path, "ambient")))

    build_env = tasks._resolve_rocm_build_environment(str(selected))

    assert build_env.root == selected
    assert build_env.source == "--rocm-path"
    assert build_env.cmake_args() == _selected_root_args(selected)
    assert build_env.command_env({})["ROCM_PATH"] == str(selected)


def test_rocm_build_environment_uses_rocm_path_then_opt_rocm(tmp_path, monkeypatch):
    ambient = _fake_rocm_root(tmp_path, "ambient")
    monkeypatch.setenv("ROCM_PATH", str(ambient))
    assert tasks._resolve_rocm_build_environment().root == ambient

    monkeypatch.delenv("ROCM_PATH")
    monkeypatch.setattr(tasks.shutil, "which", lambda name: None)
    default = tasks._resolve_rocm_build_environment()
    assert default.root == Path("/opt/rocm")
    assert default.source == "/opt/rocm"


def test_rocm_build_environment_rejects_an_empty_rocm_path(monkeypatch):
    monkeypatch.setenv("ROCM_PATH", "")
    with pytest.raises(tasks.Exit, match="ROCM_PATH is set but empty"):
        tasks._resolve_rocm_build_environment()


def test_rocm_build_environment_finds_compilers_under_lib_llvm_bin(tmp_path):
    root = (tmp_path / "rocm").resolve()
    llvm_bin = root / "lib" / "llvm" / "bin"
    llvm_bin.mkdir(parents=True)
    for tool in ("amdclang", "amdclang++"):
        (llvm_bin / tool).write_text("", encoding="utf-8")

    build_env = tasks._resolve_rocm_build_environment(str(root))

    assert build_env.c_compiler == llvm_bin / "amdclang"
    assert build_env.cxx_compiler == llvm_bin / "amdclang++"


def test_rocisa_install_builds_with_the_selected_rocm_compilers(tmp_path, monkeypatch):
    selected = _fake_rocm_root(tmp_path, "selected")
    monkeypatch.setenv("ROCM_PATH", str(_fake_rocm_root(tmp_path, "ambient")))
    monkeypatch.setattr(tasks.shutil, "which", lambda name: None)
    source = tmp_path / "rocisa"
    source.mkdir()
    context = _RecordingContext()

    tasks._pip_install_rocisa(context, rocisa_dir=source, rocm_path=str(selected))

    ((_, kwargs),) = context.commands
    cmake_args = shlex.split(kwargs["env"]["CMAKE_ARGS"])
    root_args = _selected_root_args(selected)
    for arg in (root_args[0], root_args[2], root_args[3]):
        assert arg in cmake_args
    # scikit-build-core owns CMAKE_PREFIX_PATH (it finds nanobind there); rocisa's
    # CMakeLists.txt appends ROCM_PATH to it ahead of /opt/rocm.
    assert not any(arg.startswith("-DCMAKE_PREFIX_PATH=") for arg in cmake_args)
    assert kwargs["env"]["ROCM_PATH"] == str(selected)


def test_build_client_overrides_the_preset_with_the_selected_root(tmp_path, monkeypatch):
    selected = _fake_rocm_root(tmp_path, "selected")
    ambient = _fake_rocm_root(tmp_path, "ambient")
    monkeypatch.setenv("ROCM_PATH", str(ambient))
    monkeypatch.setattr(tasks.subprocess, "run", lambda *args, **kwargs: None)
    context = _RecordingContext()

    tasks.build_client.body(
        context,
        build_dir=str(tmp_path / "build"),
        gpu_targets="gfx942",
        rocm_path=str(selected),
        build=False,
    )

    configure = shlex.split(context.commands[0][0])
    # The tensilelite preset hard-codes /opt/rocm compilers and CMAKE_PREFIX_PATH;
    # the selected root must override all of them, not just the compilers.
    assert "--preset" in configure
    for arg in _selected_root_args(selected):
        assert arg in configure
    assert not any(str(ambient) in arg for arg in configure)


def test_build_client_without_rocm_path_uses_the_rocm_path_environment(tmp_path, monkeypatch):
    ambient = _fake_rocm_root(tmp_path, "ambient")
    monkeypatch.setenv("ROCM_PATH", str(ambient))
    monkeypatch.setattr(tasks.subprocess, "run", lambda *args, **kwargs: None)
    context = _RecordingContext()

    tasks.build_client.body(
        context, build_dir=str(tmp_path / "build"), gpu_targets="gfx942", build=False
    )

    configure = shlex.split(context.commands[0][0])
    for arg in _selected_root_args(ambient):
        assert arg in configure


def test_build_coverage_uses_the_selected_rocm_root(tmp_path, monkeypatch):
    selected = _fake_rocm_root(tmp_path, "selected")
    monkeypatch.setenv("ROCM_PATH", str(_fake_rocm_root(tmp_path, "ambient")))
    context = _RecordingContext()

    tasks.build_coverage.body(
        context,
        build_dir=str(tmp_path / "cov"),
        gpu_targets="gfx942",
        rocm_path=str(selected),
    )

    configure = shlex.split(context.commands[0][0])
    for arg in _selected_root_args(selected):
        assert arg in configure


@pytest.mark.skipif(sys.platform != "linux", reason="invoke install is Linux-only")
def test_install_explicit_rocm_path_wins_over_ambient_rocm_path(tmp_path, monkeypatch):
    selected = _fake_rocm_root(tmp_path, "selected", version="7.2.4")
    monkeypatch.setenv("ROCM_PATH", str(_fake_rocm_root(tmp_path, "ambient", version="9.9.9")))
    build_dir = tmp_path / "build"
    client = tasks._built_client_path(build_dir)
    client.parent.mkdir(parents=True)
    client.write_text("client", encoding="utf-8")
    client.chmod(0o755)
    calls = []

    class Context:
        def run(self, command, **kwargs):
            calls.append(("run", command, kwargs))

    monkeypatch.setattr(
        tasks,
        "_install_rocisa",
        lambda context, **kwargs: calls.append(("rocisa", kwargs)),
    )
    monkeypatch.setattr(
        tasks,
        "_build_client",
        lambda context, **kwargs: calls.append(("build", kwargs)),
    )

    tasks.install.body(Context(), build_dir=str(build_dir), rocm_path=str(selected))

    assert calls[1] == (
        "rocisa",
        {
            "build_env": tasks._resolve_rocm_build_environment(str(selected)),
        },
    )
    assert calls[2][1]["build_env"] is calls[1][1]["build_env"]
    assert calls[2][1]["build_env"].root == selected
    # The editable install and the binding read no ROCm: the package is a dev
    # build (+g<commit>), and invoke passes it no ROCm identity.
    for editable_or_binding in (calls[3], calls[4]):
        assert "env" not in editable_or_binding[2]


@pytest.mark.skipif(sys.platform != "linux", reason="invoke install is Linux-only")
def test_install_passes_no_rocm_identity_to_the_package_build(tmp_path, monkeypatch):
    selected = _fake_rocm_root(tmp_path, "selected", version="7.2.4")
    monkeypatch.delenv("TENSILELITE_ROCM_VERSION", raising=False)
    build_dir = tmp_path / "build"
    client = tasks._built_client_path(build_dir)
    client.parent.mkdir(parents=True)
    client.write_text("client", encoding="utf-8")
    client.chmod(0o755)
    context = _RecordingContext()
    monkeypatch.setattr(tasks, "_install_rocisa", lambda context, **kwargs: None)
    monkeypatch.setattr(tasks, "_build_client", lambda context, **kwargs: None)

    tasks.install.body(context, build_dir=str(build_dir), rocm_path=str(selected))

    for command, kwargs in context.commands:
        assert "TENSILELITE_ROCM_VERSION" not in command
        assert "TENSILELITE_ROCM_VERSION" not in kwargs.get("env", {})


@pytest.mark.parametrize("selection", ["flag", "environment", "sdk", "default"])
def test_build_client_reports_and_checks_the_selected_compilers(
    tmp_path, monkeypatch, capsys, selection
):
    root = _fake_rocm_root(tmp_path, "selected")
    monkeypatch.delenv("ROCM_PATH", raising=False)
    kwargs = {}
    if selection == "flag":
        kwargs["rocm_path"] = str(root)
        source = "--rocm-path"
    elif selection == "environment":
        monkeypatch.setenv("ROCM_PATH", str(root))
        source = "ROCM_PATH"
    else:
        if selection == "default":
            root = Path("/opt/rocm")
        monkeypatch.setattr(tasks, "_detect_rocm", lambda: str(root))
        source = "rocm-sdk path --root" if selection == "sdk" else "/opt/rocm"
    checks = []
    monkeypatch.setattr(tasks.subprocess, "run", lambda *args, **kw: checks.append((args, kw)))
    context = _RecordingContext()
    tasks.build_client.body(
        context, build_dir=str(tmp_path / "build"), gpu_targets="gfx942", build=False, **kwargs
    )
    assert len(checks) == 2
    for (args, options), compiler in zip(checks, ("amdclang", "amdclang++")):
        assert args[0] == [str(tasks._rocm_tool(root, compiler)), "--version"]
        assert options["timeout"] == 5
        assert options["check"] is True
        assert options["env"]["ROCM_PATH"] == str(root)
    assert capsys.readouterr().out == f"Using ROCm at {root} (from {source})\n"
    assert f"-DROCM_PATH={root}" in shlex.split(context.commands[0][0])


@pytest.mark.parametrize("compiler", ["amdclang", "amdclang++"])
@pytest.mark.parametrize("failure", ["missing", "nonzero", "timeout", "permission"])
def test_compiler_failure_stops_before_cleanup_or_cmake(tmp_path, monkeypatch, compiler, failure):
    root = _fake_rocm_root(tmp_path, "selected")
    monkeypatch.setenv("ROCM_PATH", str(root))
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    marker = build_dir / "keep"
    marker.write_text("existing build")
    executable = str(root / "bin" / compiler)
    errors = {
        "missing": FileNotFoundError(executable),
        "nonzero": subprocess.CalledProcessError(1, [executable, "--version"]),
        "timeout": subprocess.TimeoutExpired([executable, "--version"], 5),
        "permission": PermissionError(executable),
    }

    def check(command, **kwargs):
        if command[0] == executable:
            raise errors[failure]

    monkeypatch.setattr(tasks.subprocess, "run", check)
    context = _RecordingContext()
    with pytest.raises(tasks.Exit) as error:
        tasks.build_client.body(
            context, clean=True, build_dir=str(build_dir), gpu_targets="gfx942", build=False
        )
    assert executable in str(error.value)
    expected = "compiler not found" if failure == "missing" else "compiler check failed"
    assert expected in str(error.value)
    assert context.commands == []
    assert marker.read_text() == "existing build"


def _executable_rocm_root(tmp_path, name, arch, stepping=None):
    """Tools check their ROCM_PATH as well as reporting a controlled GPU."""
    root = _fake_rocm_root(tmp_path, name)
    for relative, output in (
        ("bin/amdclang", "compiler"),
        ("bin/amdclang++", "compiler"),
        ("lib/llvm/bin/amdgpu-arch", arch),
        ("bin/rocminfo", "  Name:  " + (stepping or arch)),
    ):
        tool = root / relative
        tool.parent.mkdir(parents=True, exist_ok=True)
        environment_check = (
            f'test "$ROCM_PATH" = {shlex.quote(str(root))} || exit 2\n'
            if relative.endswith(("amdgpu-arch", "rocminfo"))
            else ""
        )
        tool.write_text(
            "#!/bin/sh\n" + environment_check + f"printf '%s\\n' {shlex.quote(output)}\n"
        )
        tool.chmod(0o755)
    return root


@pytest.mark.skipif(sys.platform == "win32", reason="executable shell tool fixtures")
@pytest.mark.parametrize("selection", ["flag", "sdk"])
@pytest.mark.parametrize("workflow", ["client", "coverage", "rocisa"])
def test_native_auto_detection_uses_selected_tools_and_recovers_stepping(
    tmp_path, monkeypatch, selection, workflow
):
    selected = _executable_rocm_root(tmp_path, "selected", "gfx1250", "gfx1250-strict")
    ambient = _executable_rocm_root(tmp_path, "ambient", "gfx942")
    monkeypatch.setenv("ROCM_PATH", str(ambient))
    kwargs = {"rocm_path": str(selected)}
    if selection == "sdk":
        monkeypatch.delenv("ROCM_PATH")
        cli_dir = tmp_path / "sdk-cli"
        cli_dir.mkdir()
        cli = cli_dir / "rocm-sdk"
        cli.write_text(f"#!/bin/sh\nprintf '%s\\n' {shlex.quote(str(selected))}\n")
        cli.chmod(0o755)
        monkeypatch.setenv("PATH", str(cli_dir) + os.pathsep + os.environ["PATH"])
        kwargs = {}
    caller_env = dict(os.environ)
    context = _RecordingContext()
    if workflow == "rocisa":
        tasks.rocisa.body(context, **kwargs)
        assert len(context.commands) == 1  # rocisa
        for _, options in context.commands:
            assert options["env"]["ROCM_PATH"] == str(selected)
    else:
        options = {"build_dir": str(tmp_path / "build"), **kwargs}
        if workflow == "client":
            tasks.build_client.body(context, build=False, **options)
        else:
            tasks.build_coverage.body(context, **options)
        configure = shlex.split(context.commands[0][0])
        assert "-DGPU_TARGETS=gfx1250-strict" in configure
        for arg in _selected_root_args(selected):
            assert arg in configure
    assert dict(os.environ) == caller_env


@pytest.mark.skipif(sys.platform == "win32", reason="executable shell tool fixtures")
@pytest.mark.parametrize("workflow", ["client", "coverage"])
def test_selected_root_without_gpu_tools_does_not_borrow_ambient_tools(
    tmp_path, monkeypatch, capsys, workflow
):
    selected = _executable_rocm_root(tmp_path, "selected root", "gfx1250")
    (selected / "lib/llvm/bin/amdgpu-arch").unlink()
    (selected / "bin/rocminfo").unlink()
    ambient = _executable_rocm_root(tmp_path, "ambient", "gfx942")
    monkeypatch.setenv("ROCM_PATH", str(ambient))
    monkeypatch.setenv("PATH", str(ambient / "lib/llvm/bin") + os.pathsep + os.environ["PATH"])
    context = _RecordingContext()
    kwargs = {"build_dir": str(tmp_path / "build"), "rocm_path": str(selected)}
    if workflow == "client":
        with pytest.raises(tasks.Exit, match="No GPU detected") as error:
            tasks.build_client.body(context, **kwargs)
        assert error.value.code == 1
        diagnostic = str(error.value)
    else:
        tasks.build_coverage.body(context, **kwargs)
        diagnostic = next(
            line for line in capsys.readouterr().out.splitlines() if "No GPU detected" in line
        )
    assert f"No GPU detected using ROCm at {selected}" in diagnostic
    assert context.commands == []


@pytest.mark.skipif(sys.platform == "win32", reason="executable shell tool fixtures")
def test_explicit_targets_skip_gpu_detection(tmp_path, monkeypatch):
    from tensilelite import GpuArch

    selected = _executable_rocm_root(tmp_path, "selected", "gfx1250")

    def forbidden(*args, **kwargs):
        raise AssertionError("automatic GPU detection was used")

    monkeypatch.setattr(GpuArch, "detect_gpu_arch", forbidden)
    context = _RecordingContext()
    tasks.build_client.body(
        context,
        rocm_path=str(selected),
        gpu_targets="gfx950:sramecc+",
        build_dir=str(tmp_path / "build"),
        build=False,
    )
    assert "-DGPU_TARGETS=gfx950" in shlex.split(context.commands[0][0])


@pytest.mark.skipif(sys.platform != "linux", reason="invoke install is Linux-only")
@pytest.mark.parametrize(
    "ambient_spelling", ["different", "same", "symlink", "relative", "unset", "empty"]
)
def test_install_warns_once_only_for_different_explicit_and_exported_roots(
    tmp_path, monkeypatch, capsys, ambient_spelling
):
    selected = _fake_rocm_root(tmp_path, "selected")
    ambient = str(selected)
    if ambient_spelling == "different":
        ambient = str(_fake_rocm_root(tmp_path, "ambient"))
    elif ambient_spelling == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(selected, target_is_directory=True)
        ambient = str(alias)
    elif ambient_spelling == "relative":
        monkeypatch.chdir(tmp_path)
        ambient = "selected"
    elif ambient_spelling == "empty":
        ambient = ""
    if ambient_spelling == "unset":
        monkeypatch.delenv("ROCM_PATH", raising=False)
    else:
        monkeypatch.setenv("ROCM_PATH", ambient)
    caller_env = dict(os.environ)
    build_dir = tmp_path / "build"
    client = tasks._built_client_path(build_dir)
    client.parent.mkdir(parents=True)
    client.write_text("client")
    client.chmod(0o755)
    monkeypatch.setattr(tasks, "_install_rocisa", lambda *args, **kwargs: None)
    monkeypatch.setattr(tasks, "_build_client", lambda *args, **kwargs: None)
    context = _RecordingContext()
    tasks.install.body(context, build_dir=str(build_dir), rocm_path=str(selected))
    output = capsys.readouterr().out
    assert output.count("Using ROCm at") == 1
    assert output.count("warning:") == (1 if ambient_spelling == "different" else 0)
    if ambient_spelling == "different":
        assert str(selected) in output and ambient in output
        assert "Conventional runtime selection follows ROCM_PATH" in output
    assert dict(os.environ) == caller_env
    assert all("env" not in options for _, options in context.commands)


@pytest.mark.skipif(sys.platform != "linux", reason="invoke install is Linux-only")
def test_install_reuses_the_original_environment_for_native_detection(
    tmp_path, monkeypatch, capsys
):
    selected = _executable_rocm_root(tmp_path, "selected", "gfx1250", "gfx1250-strict")
    monkeypatch.setenv("ROCM_PATH", str(selected))
    caller_env = dict(os.environ)
    build_dir = tmp_path / "build"
    client = tasks._built_client_path(build_dir)
    client.parent.mkdir(parents=True)
    client.write_text("client")
    client.chmod(0o755)
    context = _RecordingContext()
    tasks.install.body(context, build_dir=str(build_dir))
    output = capsys.readouterr().out
    assert output.count("Using ROCm at") == 1
    assert f"Using ROCm at {selected} (from ROCM_PATH)" in output
    configure = next(
        shlex.split(command)
        for command, _ in context.commands
        if command.startswith("cmake --preset")
    )
    assert "-DGPU_TARGETS=gfx1250-strict" in configure
    for arg in _selected_root_args(selected):
        assert arg in configure
    assert all("env" not in options for _, options in context.commands[-2:])
    assert dict(os.environ) == caller_env
