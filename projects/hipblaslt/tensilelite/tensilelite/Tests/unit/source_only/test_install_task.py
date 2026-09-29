# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import importlib.util
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

import tasks


pytestmark = pytest.mark.unit

_SOURCE_ROOT = Path(__file__).resolve().parents[4]
REVISION_OPT = "-DHIPBLASLT_ASIC_REVISION"


def _load_hipblaslt_tasks():
    spec = importlib.util.spec_from_file_location(
        "hipblaslt_tasks", _SOURCE_ROOT.parent / "tasks.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_rocisa_install_uses_the_invoking_python_and_selected_rocm_compilers(
    tmp_path, monkeypatch
):
    commands = []

    class Context:
        def run(self, command, **kwargs):
            commands.append((shlex.split(command), kwargs))

    stinkytofu_tasks = SimpleNamespace(
        cmake_build_args=lambda **kwargs: [
            f"-DCMAKE_INSTALL_PREFIX={kwargs['install_prefix']}",
            f"-DBUILD_SHARED_LIBS={'ON' if kwargs['shared'] else 'OFF'}",
        ]
    )
    monkeypatch.setattr(tasks, "_load_stinkytofu_tasks", lambda: stinkytofu_tasks)
    monkeypatch.setattr(tasks.shutil, "which", lambda name: None)

    rocm_root = tmp_path / "rocm"
    source = tmp_path / "rocisa"
    source.mkdir()
    prefix = tmp_path / "prefix"
    tasks._pip_install_rocisa(
        Context(),
        rocisa_dir=source,
        stinkytofu_prefix=prefix,
        rocm_path=str(rocm_root),
    )

    configure_command = commands[0][0]
    assert f"-DCMAKE_C_COMPILER={rocm_root / 'bin/amdclang'}" in configure_command
    assert f"-DCMAKE_CXX_COMPILER={rocm_root / 'bin/amdclang++'}" in configure_command
    assert commands[-1][0][:4] == [
        str(Path(sys.executable).absolute()),
        "-m",
        "pip",
        "install",
    ]


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


class TestTargetsIncludeGfx1250:
    """Which --architecture values can produce gfx1250, and so are worth the
    probe's hipcc compile and device open."""

    @pytest.mark.parametrize(
        "architecture",
        [
            "gfx1250",
            "gfx942;gfx1250",
            "gfx1250:xnack-",
            "gfx1250[cu=64]",
            " gfx1250 ",
            # 'all' (the default) and an empty list both expand to
            # BASE_ARCHITECTURES, which contains gfx1250; disagreeing would skip the
            # probe for a build that does produce it.
            "all",
            "gfx942;all",
            "",
            None,
        ],
    )
    def test_targets_that_can_produce_gfx1250(self, architecture):
        assert hipblaslt_tasks._targets_include_gfx1250(architecture)

    @pytest.mark.parametrize(
        "architecture",
        [
            "gfx942",
            "gfx942;gfx950",
            "gfx1200",
            "gfx12501",
            "gfx1250v0",  # substring matches would make these look benign
        ],
    )
    def test_targets_that_cannot(self, architecture):
        assert not hipblaslt_tasks._targets_include_gfx1250(architecture)


class TestAsicRevisionOption:
    """What reaches CMake. A gfx1250 build ships both revisions' trees by
    default and lets the runtime pick by asicRevision, so the build machine no
    longer decides anything and is never probed. The option is emitted even when
    its value is empty: the CMake cache variable is sticky across incremental
    builds, so an unset one would keep a directory once pinned to v0 building
    only v0."""

    def test_the_default_builds_both_trees(self):
        assert hipblaslt_tasks._asic_revision_option(
            "gfx1250", None
        ) == f"{REVISION_OPT}="

    def test_the_default_does_not_look_at_the_local_gpu(self):
        # The build is machine-independent by construction: nothing here may
        # reach for a device, or two CI runners produce different packages.
        assert not hasattr(hipblaslt_tasks, "_detect_asic_revision")

    def test_a_build_without_gfx1250_emits_nothing(self):
        assert hipblaslt_tasks._asic_revision_option("gfx942", None) is None

    @pytest.mark.parametrize("pinned", ["v0", "v1"])
    def test_an_explicit_revision_prunes_to_one_tree(self, pinned):
        assert hipblaslt_tasks._asic_revision_option(
            "gfx1250", pinned
        ) == f"{REVISION_OPT}={pinned}"

    def test_an_explicit_revision_applies_even_without_gfx1250_targets(self):
        # Cross-building: the caller decides, so a target list naming no gfx1250
        # still emits the option (to keep the cache from going stale).
        assert hipblaslt_tasks._asic_revision_option(
            "gfx942", "v0"
        ) == f"{REVISION_OPT}=v0"

    @pytest.mark.parametrize("bogus", ["0", "v2", "V0", "gfx1250v0"])
    def test_an_unrecognized_revision_is_rejected(self, bogus):
        # Anything else reaches CMake as a comparison matching neither branch,
        # which would quietly build the wrong set of trees.
        with pytest.raises(SystemExit) as exit_info:
            hipblaslt_tasks._asic_revision_option("gfx1250", bogus)
        assert exit_info.value.code == 2


class TestTheBuildSaysWhichRevisionItChose:
    """The log line is the only record of which trees an install actually got;
    the failure it guards against is a v0 part finding no library of its own."""

    def _option(self, capsys, architecture, pinned):
        hipblaslt_tasks._asic_revision_option(architecture, pinned)
        return capsys.readouterr().out

    def test_the_default_says_both_and_who_decides(self, capsys):
        out = self._option(capsys, "gfx1250", None)
        assert "gfx1250 ASIC revision: both" in out
        assert "runtime selects by asicRevision" in out

    def test_a_pinned_revision_says_it_was_pinned(self, capsys):
        out = self._option(capsys, "gfx1250", "v1")
        assert "gfx1250 ASIC revision: v1" in out
        assert "pinned by --asic-revision" in out
        assert "no gfx1250" not in out  # that caveat is for gfx1250-free builds

    def test_pinning_v1_does_not_read_as_the_default(self, capsys):
        # Pruning to the v1 tree leaves v0 silicon with no library of its
        # own, so it must not be reported the same way the both-trees default is.
        out = self._option(capsys, "gfx1250", "v1")
        assert "both" not in out
        assert "asicRevision" not in out

    def test_a_pinned_revision_without_gfx1250_targets_says_so(self, capsys):
        out = self._option(capsys, "gfx942", "v0")
        assert "gfx1250 ASIC revision: v0" in out
        assert "these targets contain no gfx1250" in out

    def test_a_build_that_cannot_produce_gfx1250_says_nothing(self, capsys):
        assert "ASIC revision" not in self._option(capsys, "gfx942", None)


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

    def test_the_revision_option_takes_no_letter(self):
        assert self._short_flags("--asic-revision") == ()


@pytest.mark.skipif(sys.platform != "linux", reason="invoke install is Linux-only")
def test_install_uses_one_selected_rocm_root_and_binds_the_built_client(
    tmp_path, monkeypatch
):
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
    assert calls[1] == ("rocisa", {"rocm_path": str(rocm_root)})
    assert calls[2][0] == "build-client"
    assert calls[2][1]["rocm_path"] == str(rocm_root)
    editable_command = shlex.split(calls[3][1])
    assert editable_command[-4:] == [
        "--no-build-isolation",
        "--no-deps",
        "--editable",
        str(_SOURCE_ROOT),
    ]
    assert calls[3][2]["env"]["ROCM_PATH"] == str(rocm_root)
    assert shlex.split(calls[4][1]) == [
        str(Path(sys.executable).absolute()),
        "-m",
        "tensilelite_configure_client",
        "--client",
        str(client),
    ]


@pytest.mark.skipif(sys.platform != "linux", reason="invoke install is Linux-only")
@pytest.mark.parametrize("create_non_executable", [False, True])
def test_install_rejects_an_invalid_built_client(
    tmp_path, monkeypatch, create_non_executable
):
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
    monkeypatch.setattr(tasks, "rocisa", SimpleNamespace(body=lambda *args, **kwargs: None))
    monkeypatch.setattr(
        tasks, "build_client", SimpleNamespace(body=lambda *args, **kwargs: None)
    )

    with pytest.raises(tasks.Exit, match="Built tensilelite-client is missing or not executable"):
        tasks.install.body(Context(), build_dir=str(tmp_path / "build"))
