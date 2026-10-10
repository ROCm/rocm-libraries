# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The pin of a run: code, engine module and config it is produced with."""

import json
import subprocess
import sys

from lib import pinning


def _tree(root):
    (root / "lib").mkdir(parents=True)
    (root / "stages").mkdir()
    (root / "tests").mkdir()
    (root / "run_orchestrator.py").write_text("print('orchestrator')\n")
    (root / "lib" / "a.py").write_text("A = 1\n")
    (root / "stages" / "s.py").write_text("S = 1\n")
    (root / "tests" / "t.py").write_text("T = 1\n")
    for rel in pinning.EXCLUDED:
        (root / rel).write_text("V = 1\n")
    return root


def test_the_code_digest_covers_the_files_that_produce_a_run(tmp_path):
    root = _tree(tmp_path / "pipeline")
    first = pinning.code_digest(root)
    (root / "tests" / "t.py").write_text("T = 2\n")
    (root / "lib" / "__pycache__").mkdir()
    (root / "lib" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"x")
    for rel in pinning.EXCLUDED:
        (root / rel).write_text("V = 2\n")
    (root / "lib" / ".#a.py").symlink_to("user@host.1234:1")
    assert pinning.code_digest(root) == first
    (root / "lib" / "a.py").write_text("A = 2\n")
    assert pinning.code_digest(root) != first
    (root / "lib" / "a.py").write_text("A = 1\n")
    (root / "stages" / "new.py").write_text("")
    assert pinning.code_digest(root) != first


def test_the_excluded_stages_are_the_validation_stages(pipeline_dir):
    import run_orchestrator as ro

    scripts = {f"stages/{ro.STAGE_DEFS[k].script}" for k in ("stage07", "stage08")}
    assert set(pinning.EXCLUDED) == scripts
    assert all((pipeline_dir / rel).is_file() for rel in scripts)


def test_a_file_that_cannot_be_read_is_a_change_not_an_error(tmp_path):
    root = _tree(tmp_path / "pipeline")
    before = pinning.code_record(root)
    gone = root / "lib" / "a.py"
    assert pinning._file_digest(tmp_path / "missing.py").startswith(
        "unreadable (FileNotFoundError"
    )
    gone.chmod(0)
    try:
        after = pinning.code_record(root)
    finally:
        gone.chmod(0o644)
    if after["files"]["lib/a.py"] == before["files"]["lib/a.py"]:
        return  # the file stays readable to this user (a privileged one)
    assert after["files"]["lib/a.py"] == "unreadable (PermissionError)"
    (change,) = pinning.differences({"code": before}, {"code": after})
    assert change.endswith("; files: lib/a.py (unreadable (PermissionError))")


def test_the_code_record_names_the_revision_and_uncommitted_changes(tmp_path):
    repo = tmp_path / "repo"
    root = _tree(repo / "training_pipeline")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-qm", "init"], check=True)
    head = subprocess.run(
        git[:3] + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    rec = pinning.code_record(root)
    assert rec["revision"] == head and rec["dirty"] is False
    (root / "lib" / "a.py").write_text("A = 3\n")
    assert pinning.code_record(root)["dirty"] is True
    outside = _tree(tmp_path / "plain")
    rec = pinning.code_record(outside)
    assert rec["digest"] == pinning.code_digest(outside)


def test_the_engine_record_names_the_module_the_stages_import(tmp_path, monkeypatch):
    pkg = tmp_path / "mods" / "tilewright"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        "def feature_catalog_hashes():\n    return ['b', 'a']\n"
    )
    (pkg / "_engine.so").write_bytes(b"\x7fELF one")
    monkeypatch.syspath_prepend(str(tmp_path / "mods"))
    monkeypatch.setitem(sys.modules, "tilewright", None)
    del sys.modules["tilewright"]
    rec = pinning.engine_record()
    assert rec["path"] == str(pkg) and rec["catalogs"] == ["a", "b"]
    pin = {"code": pinning.code_state(tmp_path), "engine": rec}
    assert pinning.drift(pin, tmp_path) == []
    (pkg / "_engine.so").write_bytes(b"\x7fELF two")
    changes = pinning.drift(pin, tmp_path)
    assert len(changes) == 1 and changes[0].startswith("engine module: ")
    assert pinning.drift(pin, tmp_path, ("code", "config")) == []


def test_differences_name_every_changed_part():
    old = {
        "code": {
            "digest": "c1" * 32,
            "files": {"lib/a.py": "1", "lib/b.py": "1"},
            "revision": "r" * 40,
            "dirty": False,
        },
        "engine": {"path": "/e", "sha256": "e1" * 32, "catalogs": ["x"]},
        "config": pinning.config_record({"train.epochs": 200, "seed.seed": 42}),
    }
    assert pinning.differences(old, old) == []
    new = json.loads(json.dumps(old))
    new["code"].update(digest="c2" * 32, dirty=True)
    new["code"]["files"] = {"lib/a.py": "2", "stages/c.py": "1"}
    new["config"] = pinning.config_record({"train.epochs": 100, "seed.seed": 42})
    changes = pinning.differences(old, new)
    assert changes[0].startswith("pipeline code: c1c1")
    assert "with uncommitted changes" in changes[0]
    assert changes[0].endswith("; files: lib/a.py, lib/b.py (gone), stages/c.py (new)")
    assert changes[1:] == ["config train.epochs: 200 -> 100"]
    assert pinning.differences(old, new, ("config",)) == changes[1:]
    assert pinning.differences(old, new, ()) == []
    del new["config"]
    assert len(pinning.differences(old, new)) == 1
    many = {f"lib/m{i}.py": "1" for i in range(12)}
    more = pinning.differences(
        {"code": {"digest": "a", "files": {}}}, {"code": {"digest": "b", "files": many}}
    )
    assert more[0].endswith(" and 4 more")


def test_markers_record_the_pin_without_its_files_and_values():
    pin = {
        "code": {"digest": "c", "files": {"lib/a.py": "1"}, "revision": None},
        "engine": {"sha256": "e"},
        "config": pinning.config_record({"train.epochs": 1}),
    }
    out = pinning.summary(pin)
    assert out["code"] == {"digest": "c", "revision": None}
    assert out["config"] == {"fingerprint": pin["config"]["fingerprint"]}
    assert "files" in pin["code"]


def test_the_run_pin_keeps_the_changes_an_invocation_accepted(tmp_path):
    first = {"code": {"digest": "a"}, "engine": {"sha256": "e"}}
    pinning.write_run_pin(tmp_path, first, first_stage="round_3/stage05")
    rec = pinning.read_run_pin(tmp_path)
    assert rec["pin"] == first and rec["history"] == []
    assert rec["first_stage"] == "round_3/stage05"
    second = {"code": {"digest": "b"}, "engine": {"sha256": "e"}}
    pinning.write_run_pin(tmp_path, second, previous=rec, accepted=["code a -> b"])
    rec = pinning.read_run_pin(tmp_path)
    assert rec["pin"] == second and rec["first_stage"] == "round_3/stage05"
    assert rec["history"][0]["changes"] == ["code a -> b"]
    assert rec["history"][0]["previous"]["code"] == {"digest": "a"}
    assert pinning.read_run_pin(tmp_path / "nothing") is None
