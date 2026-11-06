# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The editor's API and save layout, on the f8 bundle the Python module tests use."""

from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stinkytofu_ir_editor.server import (
    Workspace,
    WorkspaceError,
    _run_isolated,
    make_server,
)

stinkytofu = pytest.importorskip("stinkytofu")
if not hasattr(stinkytofu, "ir"):
    pytest.skip("this stinkytofu build has no stinkytofu.ir", allow_module_level=True)

DATA = (
    Path(__file__).resolve().parents[3]
    / "python_module"
    / "tests"
    / "data"
    / "ir_checkpoint_f8"
)
BUNDLE_ID = "f8__d9d38354c0af2adb"


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    bundle = tmp_path / "dump" / BUNDLE_ID
    bundle.mkdir(parents=True)
    meta = json.loads((DATA / "meta.json").read_text())
    shutil.copyfile(DATA / "region.stir", bundle / meta["stir"])
    shutil.copyfile(DATA / "meta.json", bundle / "meta.json")
    (tmp_path / "dump" / "not_a_bundle").mkdir()
    return Workspace(tmp_path / "dump", tmp_path / "load")


def swap_ids(text: str, a: int, b: int) -> str:
    lines = text.split("\n")
    i = next(k for k, l in enumerate(lines) if f"mod.ckpt = {{ id = {a} }}" in l)
    j = next(k for k, l in enumerate(lines) if f"mod.ckpt = {{ id = {b} }}" in l)
    lines[i], lines[j] = lines[j], lines[i]
    return "\n".join(lines)


def test_lists_only_bundles(workspace: Workspace) -> None:
    (bundle,) = workspace.list_bundles()
    assert bundle["id"] == BUNDLE_ID
    assert (
        bundle["spec"]
        == "loopWithPrefetch+noLoadLoopBody:StinkyBuildImplicitDependencyPass#1"
    )
    assert bundle["instructions"] == 248
    assert not bundle["saved"]


def test_get_bundle(workspace: Workspace) -> None:
    bundle = workspace.get_bundle(BUNDLE_ID)
    assert bundle["meta"]["arch"] == [12, 5, 0]
    assert bundle["stir"].startswith("st.func @temp() {")
    assert bundle["saved_stir"] is None


@pytest.mark.parametrize("bad", ["..", "../dump", "a/b", "missing"])
def test_bad_bundle_ids(workspace: Workspace, bad: str) -> None:
    with pytest.raises(WorkspaceError):
        workspace.get_bundle(bad)


def test_analyze_reports_unparsable_text(workspace: Workspace) -> None:
    with pytest.raises(WorkspaceError, match="does not parse") as error:
        workspace.analyze(
            {"stir": 'st.func @f() {\n  ^entry:\n    v1 = "st.v_mov_b32"(\n}\n'}
        )
    assert error.value.status == 422


def test_save_refuses_a_reversed_edge(workspace: Workspace) -> None:
    stir = workspace.get_bundle(BUNDLE_ID)["stir"]
    # id 23 sets SCC0, id 24 reads it.
    with pytest.raises(WorkspaceError) as error:
        workspace.save({"id": BUNDLE_ID, "stir": swap_ids(stir, 23, 24)})
    assert error.value.status == 409
    assert error.value.details["check"]["reversed"][0]["class"] == "scc"
    assert not (workspace.load_dir / BUNDLE_ID).exists()


def test_save_writes_the_load_layout(workspace: Workspace) -> None:
    bundle = workspace.get_bundle(BUNDLE_ID)
    # Two s_cselects that read SCC0 and write different sgprs.
    edited = swap_ids(bundle["stir"], 24, 25)
    assert workspace.validate({"id": BUNDLE_ID, "stir": edited})["ok"]
    result = workspace.save({"id": BUNDLE_ID, "stir": edited})

    target = workspace.load_dir / BUNDLE_ID
    assert Path(result["saved"]) == target / bundle["meta"]["stir"]
    assert (target / bundle["meta"]["stir"]).read_text() == edited
    assert json.loads((target / "meta.json").read_text()) == bundle["meta"]
    assert workspace.list_bundles()[0]["saved"]
    assert workspace.get_bundle(BUNDLE_ID)["saved_stir"] == edited


def test_load_dir_must_differ(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError, match="must differ"):
        Workspace(tmp_path, tmp_path)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="isolation needs fork")
def test_a_crash_does_not_take_the_server_down() -> None:
    # SIGKILL rather than abort(): StinkyTofu's abort handler prints a backtrace.
    with pytest.raises(WorkspaceError, match="SIGKILL") as error:
        _run_isolated(lambda: os.kill(os.getpid(), signal.SIGKILL))
    assert error.value.status == 422
    assert _run_isolated(lambda x: x + 1, 1) == 2


# The test serves from a thread; the editor itself is single-threaded when it forks.
@pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded:DeprecationWarning"
)
def test_http_api(workspace: Workspace) -> None:
    server = make_server(workspace, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path: str):
        with urllib.request.urlopen(base + path) as response:
            return response.status, response.read()

    def post(path: str, body: dict):
        request = urllib.request.Request(
            base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    try:
        status, page = get("/")
        assert status == 200 and b"<html" in page
        status, _ = get("/app.js")
        assert status == 200
        status, body = get("/api/bundles")
        assert status == 200 and json.loads(body)[0]["id"] == BUNDLE_ID
        stir = json.loads(get(f"/api/bundle/{BUNDLE_ID}")[1])["stir"]

        status, deps = post("/api/analyze", {"id": BUNDLE_ID, "stir": stir})
        assert status == 200 and len(deps["insts"]) == 248
        status, check = post(
            "/api/validate", {"id": BUNDLE_ID, "stir": swap_ids(stir, 23, 24)}
        )
        assert status == 200 and not check["ok"]
        status, refused = post(
            "/api/save", {"id": BUNDLE_ID, "stir": swap_ids(stir, 23, 24)}
        )
        assert status == 409 and refused["check"]["reversed"]
        status, missing = post("/api/validate", {"stir": stir})
        assert status == 400 and "'id'" in missing["error"]

        with pytest.raises(urllib.error.HTTPError):
            get("/../server.py")
    finally:
        server.shutdown()
        server.server_close()
