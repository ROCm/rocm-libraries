# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import warnings

import pytest

import _tensilelite_client_binding as binding
import tensilelite_configure_client as configure_client


pytestmark = pytest.mark.unit

_SOURCE_REVISION = Path(__file__).resolve().parents[2] / "_source_revision.py"
_SHA = "0123456789abcdef0123456789abcdef01234567"
_OTHER_SHA = "89abcdef0123456789abcdef0123456789abcdef"
_DIRTY = f"{_SHA}.dirty.00a1b2c3d4e5"


def _package(tmp_path, revision=_SHA, name="environment"):
    """An installed-looking tensilelite package directory."""
    package_dir = (tmp_path / name / "tensilelite").resolve()
    package_dir.mkdir(parents=True)
    shutil.copyfile(_SOURCE_REVISION, package_dir / "_source_revision.py")
    if revision is not False:
        (package_dir / "_build_info.py").write_text(f"SOURCE_REVISION = {revision!r}\n")
    return package_dir


def _installation(tmp_path, revision=_SHA, editable=False):
    package_dir = tmp_path / "package" / "tensilelite"
    if not package_dir.exists():
        package_dir.mkdir(parents=True)
        shutil.copyfile(_SOURCE_REVISION, package_dir / "_source_revision.py")
    return binding.Installation(package_dir, "id", "5.0.0+g" + _SHA, revision, editable)


class _Distribution:
    version = "5.0.0+g" + _SHA

    def __init__(self, package_dir: Path, *, editable: bool = False):
        self.package_dir = package_dir
        self.files = () if editable else (Path("tensilelite/__init__.py"),)
        self.editable = editable

    def locate_file(self, entry):
        return self.package_dir.parent / entry

    def read_text(self, name):
        if name == "direct_url.json" and self.editable:
            return json.dumps(
                {"url": self.package_dir.parent.as_uri(), "dir_info": {"editable": True}}
            )
        return None


def _identify(monkeypatch, package_dir, editable=False):
    monkeypatch.setattr(binding, "_package_dir", lambda: package_dir)
    monkeypatch.setattr(
        binding.metadata,
        "distributions",
        lambda **unused: [_Distribution(package_dir, editable=editable)],
    )
    return binding.current_installation()


@pytest.mark.parametrize("editable", [False, True])
def test_installation_key_uses_exact_resolved_package_directory(tmp_path, monkeypatch, editable):
    package_dir = _package(tmp_path)
    monkeypatch.setattr(binding, "_installation_source_revision", lambda path, editable: None)

    installation = _identify(monkeypatch, package_dir, editable)

    assert installation.package_dir == package_dir
    assert installation.identifier == hashlib.sha256(os.fsencode(str(package_dir))).hexdigest()
    assert installation.editable is editable


def test_different_worktrees_have_different_installation_keys(tmp_path, monkeypatch):
    identifiers = [
        _identify(monkeypatch, _package(tmp_path, name=name)).identifier for name in ("one", "two")
    ]
    assert identifiers[0] != identifiers[1]


@pytest.mark.parametrize("revision", [_SHA, _DIRTY, None])
def test_installed_package_reads_its_stamped_revision(tmp_path, monkeypatch, revision):
    installation = _identify(monkeypatch, _package(tmp_path, revision=revision))

    assert installation.source_revision == revision
    assert installation.editable is False


def test_installed_package_without_a_stamp_has_no_revision(tmp_path, monkeypatch):
    installation = _identify(monkeypatch, _package(tmp_path, revision=False))

    assert installation.source_revision is None


def test_editable_install_computes_its_revision_from_the_checkout(tmp_path, monkeypatch):
    package_dir = _package(tmp_path, revision=False)
    seen = []

    def fake_resolve(root, environ):
        seen.append((root, environ))
        return _DIRTY

    original = binding._source_revision_module(package_dir)
    monkeypatch.setitem(original, "resolve", fake_resolve)
    monkeypatch.setenv("TENSILELITE_SOURCE_REVISION", _OTHER_SHA)

    installation = _identify(monkeypatch, package_dir, editable=True)

    assert installation.source_revision == _DIRTY
    assert installation.editable is True
    # The checkout's own git is the only source; the build-time override never applies.
    assert seen == [(package_dir.parent, {})]


def test_current_installation_ignores_metadata_for_another_package_tree(tmp_path, monkeypatch):
    package_dir = _package(tmp_path, name="active")
    stale_dir = _package(tmp_path, name="stale")
    distributions = [_Distribution(stale_dir), _Distribution(package_dir)]
    distributions[0].version = "5.0.0+rocm9.9.9"
    distributions[1].version = "5.0.0+rocm7.2.4"
    monkeypatch.setattr(binding, "_package_dir", lambda: package_dir)
    monkeypatch.setattr(binding.metadata, "distributions", lambda **unused: distributions)

    assert binding.current_installation().version == "5.0.0+rocm7.2.4"


def test_binding_root_and_helper_cache_share_tensilelite_home(tmp_path, monkeypatch):
    monkeypatch.setattr(binding.Path, "home", classmethod(lambda cls: tmp_path))
    installation = _installation(tmp_path)
    assert binding.binding_path(installation) == tmp_path / ".tensilelite/bindings/id/client.json"


def test_configure_atomically_replaces_and_reset_removes_one_file(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(binding.Path, "home", classmethod(lambda cls: home))
    installation = _installation(tmp_path)
    client = (tmp_path / "client").absolute()
    client.write_text("client", encoding="utf-8")
    monkeypatch.setattr(configure_client, "current_installation", lambda: installation)
    monkeypatch.setattr(configure_client, "validate_client", lambda path, installation: None)

    configure_client.configure(client)
    path = binding.binding_path(installation)
    assert json.loads(path.read_text(encoding="utf-8")) == str(client)
    assert list(path.parent.iterdir()) == [path]

    configure_client.reset()
    assert not path.exists()


def test_ensure_revalidates_an_unchanged_client(tmp_path, monkeypatch):
    installation = _installation(tmp_path)
    client = (tmp_path / "client").absolute()
    client.write_text("client", encoding="utf-8")
    validated = []
    monkeypatch.setattr(configure_client, "current_installation", lambda: installation)
    monkeypatch.setattr(configure_client, "read_binding", lambda unused: client)
    monkeypatch.setattr(
        configure_client,
        "binding_path",
        lambda unused: pytest.fail("unchanged --ensure-client must not rewrite the binding"),
    )
    monkeypatch.setattr(
        configure_client,
        "validate_client",
        lambda path, installation: validated.append((path, installation)),
    )

    assert configure_client.configure(client, ensure=True) == client
    assert validated == [(client, installation)]


def test_ensure_rejects_a_stale_client_at_the_unchanged_path(tmp_path, monkeypatch):
    installation = _installation(tmp_path)
    client = (tmp_path / "client").absolute()
    client.write_text("client", encoding="utf-8")
    monkeypatch.setattr(configure_client, "current_installation", lambda: installation)
    monkeypatch.setattr(configure_client, "read_binding", lambda unused: client)

    def reject(path, installation):
        raise binding.ClientBindingError(f"stale {installation.source_revision}: {path}")

    monkeypatch.setattr(configure_client, "validate_client", reject)

    with pytest.raises(binding.ClientBindingError, match=f"stale {_SHA}"):
        configure_client.configure(client, ensure=True)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("file:///C:/work/tree", PureWindowsPath("C:/work/tree")),
        ("file://localhost/C:/work/tree", PureWindowsPath("C:/work/tree")),
        ("file://server/share/tree", PureWindowsPath("//server/share/tree")),
        ("file:///C:/work/a%20tree", PureWindowsPath("C:/work/a tree")),
    ],
)
def test_file_url_path_preserves_windows_roots(url, expected):
    assert binding._file_url_path(url, platform="win32") == expected


def test_configured_binding_is_exclusive_even_when_missing(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(binding.Path, "home", classmethod(lambda cls: home))
    installation = _installation(tmp_path)
    configured = (tmp_path / "missing-client").absolute()
    path = binding.binding_path(installation)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(str(configured)), encoding="utf-8")

    def default_client():
        raise AssertionError("configured bindings must not resolve a default client")

    selected = binding.selected_client(default_client, installation)

    assert selected == binding.ClientCandidate(configured, "explicit TensileLite client binding")


@pytest.mark.parametrize(
    ("platform", "name"),
    [("linux", "tensilelite-client"), ("darwin", "tensilelite-client"), ("win32", "tensilelite-client.exe")],
)
def test_client_name_and_build_path_contract(tmp_path, platform, name):
    assert binding.client_executable_name(platform) == name
    expected = (tmp_path / "build").resolve() / "tensilelite" / "client" / name
    assert binding.built_client_path(tmp_path / "build", platform) == expected


def test_default_client_candidate_uses_first_existing_executable_path(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(binding.Path, "home", classmethod(lambda cls: home))
    installation = _installation(tmp_path)
    primary = tmp_path / "primary"
    fallback = tmp_path / "fallback"
    executable = "tensilelite-client.exe" if os.name == "nt" else "tensilelite-client"
    client = fallback / executable
    client.parent.mkdir()
    client.write_text("client", encoding="utf-8")
    client.chmod(0o755)
    identity = binding.ClientIdentity(_SHA, "7.2.4 base")
    monkeypatch.setattr(binding, "validate_client", lambda path, installation: identity)

    def default_client():
        return binding.default_client_candidate(installation, (primary, fallback), "test prefix")

    selected = binding.selected_client(default_client, installation)
    assert selected == binding.ClientCandidate(client, "test prefix", identity)


def _fake_query_client(path, revision):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        f"  --source-revision) echo {revision} ;;\n"
        "  --build-rocm) echo '7.2.4 base' ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX test executable")
def test_default_client_candidate_skips_a_client_from_another_commit(tmp_path):
    installation = _installation(tmp_path)
    primary = tmp_path / "primary" / binding.client_executable_name()
    fallback = tmp_path / "fallback" / binding.client_executable_name()
    _fake_query_client(primary, _OTHER_SHA)
    _fake_query_client(fallback, _SHA)

    candidate = binding.default_client_candidate(
        installation, (primary.parent, fallback.parent), "Python SDK"
    )

    assert candidate == binding.ClientCandidate(
        fallback, "Python SDK", binding.ClientIdentity(_SHA, "7.2.4 base")
    )


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX test executable")
def test_default_client_candidate_reports_every_client_from_another_commit(tmp_path):
    installation = _installation(tmp_path)
    candidates = [tmp_path / name / binding.client_executable_name() for name in ("first", "second")]
    for candidate in candidates:
        _fake_query_client(candidate, _OTHER_SHA)

    with pytest.raises(binding.ClientBindingError, match="built from this package's commit") as exc:
        binding.default_client_candidate(
            installation, tuple(candidate.parent for candidate in candidates), "Python SDK"
        )

    assert all(str(candidate) in str(exc.value) for candidate in candidates)
    assert _OTHER_SHA in str(exc.value)


@pytest.mark.parametrize("editable", [False, True])
def test_missing_default_client_explains_how_to_get_one(tmp_path, editable):
    installation = _installation(tmp_path, editable=editable)

    with pytest.raises(binding.ClientBindingError, match="tensilelite-configure-client"):
        binding.default_client_candidate(installation, (tmp_path / "none",), "test prefix")


def _client(tmp_path):
    client = (tmp_path / "client").absolute()
    client.write_text("client", encoding="utf-8")
    client.chmod(0o755)
    return client


def _fake_client(monkeypatch, revision=_SHA, build_rocm="7.2.4 base", **overrides):
    """Answer the client's identity queries; ``overrides`` replaces one query's result."""
    answers = {
        "--source-revision": subprocess.CompletedProcess([], 0, f"{revision}\n", ""),
        "--build-rocm": subprocess.CompletedProcess([], 0, f"{build_rocm}\n", ""),
    }
    answers.update(overrides)
    queries = []
    real_run = subprocess.run

    def run(command, **kwargs):
        if len(command) != 2 or command[1] not in answers:
            return real_run(command, **kwargs)  # not a client query (e.g. conftest probes)
        assert kwargs["timeout"] == 5
        queries.append(command[1])
        answer = answers[command[1]]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(binding.subprocess, "run", run)
    return queries


@pytest.mark.parametrize("revision", [_SHA, _DIRTY])
def test_installed_package_accepts_the_exact_revision(tmp_path, monkeypatch, revision):
    queries = _fake_client(monkeypatch, revision=revision)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        binding.validate_client(_client(tmp_path), _installation(tmp_path, revision=revision))
    assert "--source-revision" in queries and "--build-rocm" in queries


@pytest.mark.parametrize(
    ("package", "client"),
    [(_SHA, _OTHER_SHA), (_SHA, _DIRTY), (_DIRTY, _SHA), (_DIRTY, f"{_SHA}.dirty.ffffffffffff")],
)
def test_installed_package_rejects_any_other_revision(tmp_path, monkeypatch, package, client):
    _fake_client(monkeypatch, revision=client, build_rocm="7.12.0rc1 full")

    with pytest.raises(binding.ClientBindingError, match="source revision mismatch") as error:
        binding.validate_client(_client(tmp_path), _installation(tmp_path, revision=package))
    message = str(error.value)
    assert package in message and client in message
    assert "client build ROCm: 7.12.0rc1 full" in message
    assert "tensilelite-configure-client" in message


def test_editable_install_rejects_another_commit(tmp_path, monkeypatch):
    _fake_client(monkeypatch, revision=_OTHER_SHA)

    with pytest.raises(binding.ClientBindingError, match="commit mismatch") as error:
        binding.validate_client(_client(tmp_path), _installation(tmp_path, editable=True))
    assert "invoke install" in str(error.value)


@pytest.mark.parametrize(
    ("package", "client"), [(_SHA, _DIRTY), (_DIRTY, _SHA), (_DIRTY, f"{_SHA}.dirty.ffffffffffff")]
)
def test_editable_install_warns_about_uncommitted_changes(tmp_path, monkeypatch, package, client):
    _fake_client(monkeypatch, revision=client)

    with pytest.warns(binding.ClientBindingWarning, match="uncommitted changes"):
        binding.validate_client(
            _client(tmp_path), _installation(tmp_path, revision=package, editable=True)
        )


def test_editable_install_accepts_the_same_clean_commit(tmp_path, monkeypatch):
    _fake_client(monkeypatch)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        binding.validate_client(_client(tmp_path), _installation(tmp_path, editable=True))


@pytest.mark.parametrize("editable", [False, True])
def test_package_without_a_revision_never_pairs(tmp_path, monkeypatch, editable):
    _fake_client(monkeypatch)

    with pytest.raises(binding.ClientBindingError, match="records no source revision") as error:
        binding.validate_client(
            _client(tmp_path), _installation(tmp_path, revision=None, editable=editable)
        )
    assert "TENSILELITE_SOURCE_REVISION" in str(error.value)


def test_client_without_a_revision_never_pairs(tmp_path, monkeypatch):
    _fake_client(monkeypatch, revision="none")

    with pytest.raises(binding.ClientBindingError, match="client records no source revision"):
        binding.validate_client(_client(tmp_path), _installation(tmp_path))


def test_client_revision_must_be_a_commit(tmp_path, monkeypatch):
    _fake_client(monkeypatch, revision="5.0.0+rocm7.2.4")

    with pytest.raises(binding.ClientBindingError, match="not a git commit"):
        binding.validate_client(_client(tmp_path), _installation(tmp_path))


def _result(stdout="x\n", stderr="", returncode=0):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (_result(stderr="loader warning"), "wrote to stderr"),
        (_result(returncode=3), "status 3.*rebuild"),
        (_result(returncode=-9), "signal 9"),
        (_result(stdout=""), "exactly one"),
        (_result(stdout="one\ntwo\n"), "exactly one"),
    ],
)
@pytest.mark.parametrize("query", ["--source-revision", "--build-rocm"])
def test_client_query_diagnostics(tmp_path, monkeypatch, query, result, message):
    _fake_client(monkeypatch, **{query: result})

    with pytest.raises(binding.ClientBindingError, match=message):
        binding.validate_client(_client(tmp_path), _installation(tmp_path))


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (FileNotFoundError("missing loader"), "Could not launch"),
        (OSError("bad image"), "loader failed"),
        (subprocess.TimeoutExpired(["client", "--source-revision"], 5), "timed out"),
    ],
)
def test_client_launch_diagnostics(tmp_path, monkeypatch, failure, message):
    _fake_client(monkeypatch, **{"--source-revision": failure})

    with pytest.raises(binding.ClientBindingError, match=message):
        binding.validate_client(_client(tmp_path), _installation(tmp_path))
