# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
from pathlib import Path
from types import SimpleNamespace
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

def test_install_derives_build_version_from_selected_rocm_root(tmp_path):
    version_file = tmp_path / ".info" / "version"
    version_file.parent.mkdir()
    version_file.write_text("7.2.4\n", encoding="utf-8")

    assert tasks._rocm_base_version(tmp_path) == "7.2.4"

def test_install_rejects_rocm_root_without_version_metadata(tmp_path):
    with pytest.raises(tasks.Exit, match="no readable version metadata"):
        tasks._rocm_base_version(tmp_path)
def test_rocisa_install_uses_the_invoking_python_and_selected_rocm(
    tmp_path, monkeypatch
):
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
def test_install_uses_one_selected_rocm_root_and_binds_the_built_client(
    tmp_path, monkeypatch
):
    rocm_root = (tmp_path / "rocm").resolve()
    rocm_root.mkdir()
    (rocm_root / ".info").mkdir()
    (rocm_root / ".info" / "version").write_text("7.2.4\n", encoding="utf-8")
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
    monkeypatch.setattr(tasks, "rocisa", SimpleNamespace(body=record_rocisa))
    monkeypatch.setattr(
        tasks, "build_client", SimpleNamespace(body=record_build_client)
    )

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
            "rocm_path": str(rocm_root),
            "stinkytofu": "auto",
            "rebuild_stinkytofu": False,
        },
    )
    assert calls[2][0] == "build-client"
    assert "stinkytofu" not in calls[2][1]
    assert "rebuild_stinkytofu" not in calls[2][1]
    assert calls[2][1]["rocm_path"] == str(rocm_root)
    editable_command = shlex.split(calls[3][1])
    assert editable_command[-4:] == [
        "--no-build-isolation",
        "--no-deps",
        "--editable",
        str(_SOURCE_ROOT),
    ]
    assert calls[3][2]["env"]["ROCM_PATH"] == str(rocm_root)
    assert calls[3][2]["env"]["TENSILELITE_ROCM_VERSION"] == "7.2.4"
    assert shlex.split(calls[4][1]) == [
        str(Path(sys.executable).absolute()),
        "-m",
        "tensilelite_configure_client",
        "--client",
        str(client),
    ]
    assert calls[4][2]["env"]["TENSILELITE_ROCM_VERSION"] == "7.2.4"


@pytest.mark.skipif(sys.platform != "linux", reason="invoke install is Linux-only")
def test_install_forwards_stinkytofu_options_to_rocisa_only(tmp_path, monkeypatch):
    rocm_root = (tmp_path / "rocm").resolve()
    rocm_root.mkdir()
    (rocm_root / ".info").mkdir()
    (rocm_root / ".info" / "version").write_text("7.2.4\n", encoding="utf-8")
    client = tasks._built_client_path(tmp_path / "build")
    client.parent.mkdir(parents=True)
    client.write_text("client", encoding="utf-8")
    client.chmod(0o755)
    calls = []

    class Context:
        def run(self, command, **kwargs):
            pass

    monkeypatch.setenv("ROCM_PATH", str(rocm_root))
    monkeypatch.setattr(
        tasks,
        "rocisa",
        SimpleNamespace(body=lambda context, **kwargs: calls.append(("rocisa", kwargs))),
    )
    monkeypatch.setattr(
        tasks,
        "build_client",
        SimpleNamespace(body=lambda context, **kwargs: calls.append(("build", kwargs))),
    )

    tasks.install.body(
        Context(),
        build_dir=str(tmp_path / "build"),
        stinkytofu="on",
        rebuild_stinkytofu=True,
    )

    assert calls[0][1]["stinkytofu"] == "on"
    assert calls[0][1]["rebuild_stinkytofu"] is True
    assert "stinkytofu" not in calls[1][1]
    assert "rebuild_stinkytofu" not in calls[1][1]


@pytest.mark.skipif(sys.platform != "linux", reason="invoke install is Linux-only")
@pytest.mark.parametrize("create_non_executable", [False, True])
def test_install_rejects_an_invalid_built_client(
    tmp_path, monkeypatch, create_non_executable
):
    rocm_root = (tmp_path / "rocm").resolve()
    rocm_root.mkdir()
    (rocm_root / ".info").mkdir()
    (rocm_root / ".info" / "version").write_text("7.2.4\n", encoding="utf-8")
    if create_non_executable:
        client = tasks._built_client_path(tmp_path / "build")
        client.parent.mkdir(parents=True)
        client.write_text("client", encoding="utf-8")

    class Context:
        def run(self, command, **kwargs):
            pass

    monkeypatch.setenv("ROCM_PATH", str(rocm_root))
    monkeypatch.setattr(tasks, "rocisa", SimpleNamespace(body=lambda *args, **kwargs: None))
    monkeypatch.setattr(
        tasks, "build_client", SimpleNamespace(body=lambda *args, **kwargs: None)
    )

    with pytest.raises(tasks.Exit, match="Built tensilelite-client is missing or not executable"):
        tasks.install.body(Context(), build_dir=str(tmp_path / "build"))


# ---------------------------------------------------------------------------
# One selected ROCm root for every native configure (conventional-prefix model
# of InvokeInstallPythonRocmSupport.md). These are command-capture tests: none
# of them needs a ROCm installation.
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


def test_rocm_build_environment_prefers_explicit_root_over_ambient_rocm_path(
    tmp_path, monkeypatch
):
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


def test_stinkytofu_binding_builds_with_the_selected_rocm(tmp_path, monkeypatch):
    selected = _fake_rocm_root(tmp_path, "selected")
    monkeypatch.setenv("ROCM_PATH", str(_fake_rocm_root(tmp_path, "ambient")))
    monkeypatch.setattr(tasks, "_pip_install_adaptor", lambda context: None)
    context = _RecordingContext()

    tasks._pip_install_stinkytofu(
        context, build_env=tasks._resolve_rocm_build_environment(str(selected))
    )

    ((_, kwargs),) = context.commands
    assert shlex.split(kwargs["env"]["CMAKE_ARGS"]) == _selected_root_args(selected)
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


def test_build_client_without_rocm_path_uses_the_rocm_path_environment(
    tmp_path, monkeypatch
):
    ambient = _fake_rocm_root(tmp_path, "ambient")
    monkeypatch.setenv("ROCM_PATH", str(ambient))
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
    monkeypatch.setenv(
        "ROCM_PATH", str(_fake_rocm_root(tmp_path, "ambient", version="9.9.9"))
    )
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
        "rocisa",
        SimpleNamespace(body=lambda context, **kwargs: calls.append(("rocisa", kwargs))),
    )
    monkeypatch.setattr(
        tasks,
        "build_client",
        SimpleNamespace(body=lambda context, **kwargs: calls.append(("build", kwargs))),
    )

    tasks.install.body(Context(), build_dir=str(build_dir), rocm_path=str(selected))

    assert calls[1] == (
        "rocisa",
        {"rocm_path": str(selected), "stinkytofu": "auto", "rebuild_stinkytofu": False},
    )
    assert calls[2][1]["rocm_path"] == str(selected)
    for editable_or_binding in (calls[3], calls[4]):
        assert editable_or_binding[2]["env"]["ROCM_PATH"] == str(selected)
        assert editable_or_binding[2]["env"]["TENSILELITE_ROCM_VERSION"] == "7.2.4"
