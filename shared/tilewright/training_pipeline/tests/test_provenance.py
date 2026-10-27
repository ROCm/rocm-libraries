# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import datetime
import getpass
import hashlib
import os
import re
import shutil
import socket
import subprocess

import pytest

from lib import features as fs
from lib import hipblaslt_launch as hl
from lib import mlrec
from lib import provenance as prov


def _chunks(folder, contents):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, text in enumerate(contents):
        path = folder / f"chunk_{i:04d}.csv"
        path.write_bytes(text)
        paths.append(path)
    return paths


def test_training_data_hash_covers_contents_and_order_but_no_path(tmp_path):
    a = _chunks(tmp_path / "a", [b"m,n\n1,2\n", b"m,n\n3,4\n"])
    b = _chunks(tmp_path / "elsewhere" / "b", [b"m,n\n1,2\n", b"m,n\n3,4\n"])
    h = prov.training_data_hash(a)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", h)
    assert prov.training_data_hash(b) == h
    assert prov.training_data_hash(a[::-1]) != h
    assert prov.training_data_hash(a[:1]) != h
    joined = _chunks(tmp_path / "c", [b"m,n\n1,2\nm,n\n3,4\n"])
    assert prov.training_data_hash(joined) != h
    expected = hashlib.sha256(b"8\nm,n\n1,2\n8\nm,n\n3,4\n").hexdigest()
    assert h == "sha256:" + expected

    big = _chunks(tmp_path / "big", [os.urandom(3 * (1 << 20) + 5)])
    data = big[0].read_bytes()
    assert (
        prov.training_data_hash(big)
        == "sha256:" + hashlib.sha256(f"{len(data)}\n".encode() + data).hexdigest()
    )


def _git(cwd, *args):
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.com",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.com",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_SYSTEM=os.devnull,
    )
    subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_source_revision(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "sub" / "x.py").write_text("x = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "c")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    head = prov.source_revision(repo / "sub")
    assert re.fullmatch(r"[0-9a-f]{40}", head)
    (repo / "untracked.txt").write_text("u\n")
    assert prov.source_revision(repo / "sub") == head
    (repo / "sub" / "x.py").write_text("x = 2\n")
    assert prov.source_revision(repo / "sub") == head + "-dirty"
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert prov.source_revision(plain) == "unknown"
    monkeypatch.setenv("PATH", str(tmp_path / "no-bin"))
    assert prov.source_revision(repo) == "unknown"


def test_utc_date():
    then = datetime.datetime(
        2000, 1, 1, 23, 30, 5, tzinfo=datetime.timezone(datetime.timedelta(hours=9))
    )
    assert prov.utc_date(then) == "2000-01-01T14:30:05Z"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", prov.utc_date())


def test_written_provenance_holds_no_host_user_or_path(tmp_path):
    chunks = _chunks(tmp_path / "round_0" / "stage04", [b"a\n", b"b\n"])
    v3 = fs.Catalog("v3", (), hl.HipblasltLaunch("gfx950"))
    cfg = {"config_id": "gfx950_bbs_tn", "arch": "gfx950"}
    p = prov.training_provenance(cfg, v3, chunks)
    assert list(p) == list(mlrec.PROVENANCE_KEYS)
    assert mlrec.check_provenance(p) == p
    assert p["config_id"] == "gfx950_bbs_tn"
    assert p["launch_model"] == f"hipblaslt:gfx950:{hl.RULE_VERSION}"
    assert p["source_revision"] == prov.source_revision()
    assert re.fullmatch(r"([0-9a-f]{40,64}(-dirty)?|unknown)", p["source_revision"])

    assert re.fullmatch(r"sha256:[0-9a-f]{64}", p["training_data_hash"])
    host = socket.gethostname().split(".")[0].lower()
    user = getpass.getuser().lower()
    for key, value in p.items():
        if key not in ("date", "source_revision"):
            assert "/" not in value and "\\" not in value, key
        assert str(tmp_path) not in value and os.path.expanduser("~") not in value
        if key in ("training_data_hash", "source_revision"):
            continue  # hex digits only, as checked above
        if len(host) >= 3:
            assert host not in value.lower(), key
        if len(user) >= 3:
            assert user not in value.lower(), key

    assert "launch_model" not in prov.training_provenance(cfg, fs.Catalog(), chunks)
    assert "launch_model" not in prov.training_provenance(cfg, fs.Catalog("v3"), chunks)
    for bad in ("../x", "a/b", "a b", ""):
        q = prov.training_provenance({"config_id": bad}, v3, chunks)
        assert "config_id" not in q
    assert "config_id" not in prov.training_provenance({}, v3, chunks)


def test_a_published_model_records_its_library_and_not_the_run():
    from lib import mlrec

    stem = "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx950"
    full = {
        "config_id": "a_training_config",
        "training_data_hash": "sha256:" + "ab" * 32,
        "source_revision": "0123abc-dirty",
        "date": "2000-01-01T00:00:00Z",
        "launch_model": "hipblaslt:gfx950:1",
    }
    published = prov.published_provenance(full, stem)
    assert published == {
        "config_id": stem,
        "training_data_hash": full["training_data_hash"],
        "launch_model": full["launch_model"],
    }
    assert list(published) == list(prov.PUBLISHED_KEYS)
    assert mlrec.check_provenance(published) == published
    assert prov.published_provenance(None, stem) == {"config_id": stem}
    assert prov.published_provenance(full, "not a stem") == {
        k: full[k] for k in ("training_data_hash", "launch_model")
    }


def test_the_deployed_provenance_follows_deploy_provenance():
    stem = "TensileLibrary_SS_SS_HA_Bias_SAV_MX_UA_Type_SS_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx950"
    full = {
        "config_id": "a_training_config",
        "training_data_hash": "sha256:" + "cd" * 32,
        "date": "2000-01-01",
        "launch_model": "hipblaslt:gfx950:1",
    }
    cfg = {"hipblaslt": {"library_stem": stem}}
    neutral = prov.published_provenance(full, stem)
    assert prov.deployed_provenance(full, cfg) == neutral
    assert prov.deployed_provenance(full, {**cfg, "deploy": {}}) == neutral
    cfg["deploy"] = {"provenance": "neutral"}
    assert prov.deployed_provenance(full, cfg) == neutral
    cfg["deploy"] = {"provenance": "full"}
    assert prov.deployed_provenance(full, cfg) == full
    assert prov.deployed_provenance(None, cfg) == {}
    with pytest.raises(ValueError, match="deploy.provenance"):
        prov.deployed_provenance(full, {"deploy": {"provenance": "some"}})
