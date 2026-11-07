# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""hipdnn_vllm.comgr_cache: where comgr's code-object cache goes (statfs mocked)."""

import os
import stat

import pytest

from hipdnn_vllm import comgr_cache
from hipdnn_vllm.config import ConfigError

NFS = 0x6969
EXT4 = 0xEF53

pytestmark = pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX only")


@pytest.fixture
def fs(monkeypatch, tmp_path):
    """Mock statfs: paths under ``home`` report ``kinds['home']``, everything else ``kinds['tmp']``."""
    home = tmp_path / "home"
    kinds = {"home": NFS, "tmp": EXT4}
    seen = []

    def magic(path):
        seen.append(path)
        return kinds["home"] if str(path).startswith(str(home)) else kinds["tmp"]

    monkeypatch.setattr(comgr_cache, "_fs_magic", magic)
    monkeypatch.setenv("HOME", str(home))
    root = tmp_path / "tmp"
    root.mkdir()
    return {"home": home, "kinds": kinds, "root": str(root), "seen": seen}


def _env(**kw):
    return {k: v for k, v in kw.items()}


def test_nfs_home_moves_cache_to_private_local_dir(fs):
    env = _env()
    path = comgr_cache.configure(env, tmp_root=fs["root"])
    assert path == os.path.join(fs["root"], f"hipdnn_vllm_comgr_{os.getuid()}")
    assert env["AMD_COMGR_CACHE_DIR"] == path
    st = os.lstat(path)
    assert stat.S_ISDIR(st.st_mode) and stat.S_IMODE(st.st_mode) == 0o700 and st.st_uid == os.getuid()
    assert fs["seen"][0] == os.path.join(str(fs["home"]), ".cache", "comgr")


def test_local_home_left_alone(fs):
    fs["kinds"]["home"] = EXT4
    env = _env()
    assert comgr_cache.configure(env, tmp_root=fs["root"]) is None
    assert "AMD_COMGR_CACHE_DIR" not in env


@pytest.mark.parametrize("name,value", [("AMD_COMGR_CACHE", "0"), ("AMD_COMGR_CACHE", "1"), ("AMD_COMGR_CACHE_DIR", "/x")])
def test_user_comgr_setting_wins(fs, name, value):
    env = _env(**{name: value})
    assert comgr_cache.configure(env, tmp_root=fs["root"]) is None
    assert env == {name: value}


def test_keep_opts_out(fs):
    env = _env(HIPDNN_VLLM_COMGR_CACHE="keep")
    assert comgr_cache.configure(env, tmp_root=fs["root"]) is None
    assert "AMD_COMGR_CACHE_DIR" not in env


def test_invalid_mode_rejected(fs):
    with pytest.raises(ConfigError):
        comgr_cache.configure(_env(HIPDNN_VLLM_COMGR_CACHE="off"), tmp_root=fs["root"])


def test_xdg_cache_home_honoured(fs, tmp_path):
    fs["kinds"]["home"] = EXT4
    xdg = fs["home"] / "xdg"
    env = _env(XDG_CACHE_HOME=str(xdg))
    comgr_cache.configure(env, tmp_root=fs["root"])
    assert fs["seen"][0] == os.path.join(str(xdg), "comgr")


def test_network_tmp_left_alone(fs):
    fs["kinds"]["tmp"] = NFS
    env = _env()
    assert comgr_cache.configure(env, tmp_root=fs["root"]) is None
    assert "AMD_COMGR_CACHE_DIR" not in env


def test_shared_dir_rejected(fs):
    path = os.path.join(fs["root"], f"hipdnn_vllm_comgr_{os.getuid()}")
    os.mkdir(path)
    os.chmod(path, 0o755)
    env = _env()
    assert comgr_cache.configure(env, tmp_root=fs["root"]) is None
    assert "AMD_COMGR_CACHE_DIR" not in env


def test_symlink_rejected(fs, tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o700)
    os.symlink(target, os.path.join(fs["root"], f"hipdnn_vllm_comgr_{os.getuid()}"))
    env = _env()
    assert comgr_cache.configure(env, tmp_root=fs["root"]) is None


def test_real_statfs_reads_a_magic():
    if not os.path.exists("/proc/self"):
        pytest.skip("Linux only")
    assert comgr_cache._fs_magic("/proc/self") == 0x9FA0  # PROC_SUPER_MAGIC
