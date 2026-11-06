# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The editor's JSON API over a dump directory of checkpoint bundles and a load directory.

`Workspace` holds the logic and is what the tests drive; `make_server` puts it behind a
standard-library HTTP server together with the static UI.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import signal
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote

STATIC_DIR = Path(__file__).parent / "static"
META_FILE = "meta.json"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


class WorkspaceError(Exception):
    """A request the workspace refuses; `status` is the HTTP status it maps to."""

    def __init__(
        self, message: str, status: int = HTTPStatus.BAD_REQUEST, **details: Any
    ):
        super().__init__(message)
        self.status = status
        self.details = details


def _run_isolated(fn: Callable[..., Any], *args: Any) -> Any:
    """Run `fn` in a forked child, so an assertion inside StinkyTofu ends the child, not the
    editor. Without fork the call runs in process."""
    if not hasattr(os, "fork"):
        return fn(*args)
    read_end, write_end = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_end)
        try:
            payload = ("ok", fn(*args))
        except ValueError as error:
            payload = ("value_error", str(error))
        except BaseException:  # noqa: BLE001 - reported to the parent, not swallowed
            payload = ("error", traceback.format_exc())
        with os.fdopen(write_end, "wb") as out:
            pickle.dump(payload, out)
        os._exit(0)

    os.close(write_end)
    with os.fdopen(read_end, "rb") as data:
        raw = data.read()
    _, status = os.waitpid(pid, 0)
    if not raw:
        if os.WIFSIGNALED(status):
            name = signal.Signals(os.WTERMSIG(status)).name
            raise WorkspaceError(
                f"StinkyTofu crashed on this IR ({name}); the server log has its message",
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        raise WorkspaceError(
            f"StinkyTofu exited with status {os.WEXITSTATUS(status)} on this IR; "
            "the server log has its message",
            HTTPStatus.UNPROCESSABLE_ENTITY,
        )
    kind, value = pickle.loads(raw)
    if kind == "value_error":
        raise WorkspaceError(value, HTTPStatus.UNPROCESSABLE_ENTITY)
    if kind == "error":
        raise WorkspaceError(value, HTTPStatus.INTERNAL_SERVER_ERROR)
    return value


def _load_ir_module():
    try:
        import stinkytofu
    except ImportError as error:
        raise SystemExit(f"stinkytofu is not importable: {error}") from error
    if not hasattr(stinkytofu, "ir"):
        raise SystemExit(
            f"{stinkytofu.__file__} has no stinkytofu.ir; rebuild the stinkytofu_python target"
        )
    return stinkytofu.ir


class Workspace:
    """Bundles under `bundles_dir`, edits saved to `load_dir` in the layout the loader reads."""

    def __init__(
        self,
        bundles_dir: os.PathLike | str,
        load_dir: os.PathLike | str,
        ir=None,
        isolate: bool = True,
    ):
        self.bundles_dir = Path(bundles_dir).resolve()
        self.load_dir = Path(load_dir).resolve()
        self._ir = ir
        self._isolate = isolate
        if not self.bundles_dir.is_dir():
            raise WorkspaceError(f"{self.bundles_dir} is not a directory")
        if self.load_dir == self.bundles_dir:
            raise WorkspaceError(
                "the load dir must differ from the bundles dir: saving would "
                "overwrite the base a later validation compares against"
            )

    @property
    def ir(self):
        if self._ir is None:
            self._ir = _load_ir_module()
        return self._ir

    def _call(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self._isolate:
            return _run_isolated(fn, *args)
        try:
            return fn(*args)
        except ValueError as error:
            raise WorkspaceError(str(error), HTTPStatus.UNPROCESSABLE_ENTITY) from error

    # ---- bundles ---------------------------------------------------------------------------

    def _bundle_dir(self, bundle_id: str) -> Path:
        if (
            not bundle_id
            or "/" in bundle_id
            or "\\" in bundle_id
            or bundle_id in (".", "..")
        ):
            raise WorkspaceError(f"bad bundle id {bundle_id!r}")
        path = self.bundles_dir / bundle_id
        if not (path / META_FILE).is_file():
            raise WorkspaceError(
                f"no bundle {bundle_id!r} in {self.bundles_dir}", HTTPStatus.NOT_FOUND
            )
        return path

    def _meta(self, bundle_id: str) -> dict:
        return json.loads((self._bundle_dir(bundle_id) / META_FILE).read_text())

    def _base(self, bundle_id: str) -> tuple[dict, str]:
        """meta.json and the dumped .stir."""
        meta = self._meta(bundle_id)
        return meta, (self._bundle_dir(bundle_id) / meta["stir"]).read_text()

    def _saved_path(self, bundle_id: str, meta: dict) -> Path:
        return self.load_dir / bundle_id / meta["stir"]

    def list_bundles(self) -> list[dict]:
        bundles = []
        for path in sorted(self.bundles_dir.iterdir()):
            if not (path / META_FILE).is_file():
                continue
            meta = json.loads((path / META_FILE).read_text())
            bundles.append(
                {
                    "id": path.name,
                    "kernel": meta.get("kernel", ""),
                    "spec": meta.get("spec", ""),
                    "instructions": meta.get("instructions", 0),
                    "saved": self._saved_path(path.name, meta).is_file(),
                }
            )
        return bundles

    def get_bundle(self, bundle_id: str) -> dict:
        meta, stir = self._base(bundle_id)
        saved = self._saved_path(bundle_id, meta)
        return {
            "id": bundle_id,
            "meta": meta,
            "stir": stir,
            "saved_stir": saved.read_text() if saved.is_file() else None,
        }

    # ---- analysis --------------------------------------------------------------------------

    def analyze(self, body: dict) -> dict:
        """The dependency analysis of `body["stir"]`, with the bundle's arch when `body["id"]`
        names one."""
        stir = _require(body, "stir")
        arch = [12, 5, 0]
        if body.get("id"):
            arch = self._meta(body["id"]).get("arch", arch)
        return self._call(self.ir.analyze, stir, list(arch))

    def validate(self, body: dict) -> dict:
        """`check_order` of `body["stir"]` against the bundle's dumped order."""
        bundle_id, stir = _require(body, "id"), _require(body, "stir")
        meta, base = self._base(bundle_id)
        return self._call(
            self.ir.check_order, base, stir, list(meta.get("arch", [12, 5, 0]))
        )

    def save(self, body: dict) -> dict:
        """Validate, then write the edit where `--load-st-ir <load_dir>` finds it."""
        bundle_id, stir = _require(body, "id"), _require(body, "stir")
        check = self.validate(body)
        if not check["ok"]:
            raise WorkspaceError(
                "the edit breaks a dependency; nothing was saved",
                HTTPStatus.CONFLICT,
                check=check,
            )
        meta = self._meta(bundle_id)
        target = self.load_dir / bundle_id
        target.mkdir(parents=True, exist_ok=True)
        (target / meta["stir"]).write_text(stir if stir.endswith("\n") else stir + "\n")
        shutil.copyfile(self._bundle_dir(bundle_id) / META_FILE, target / META_FILE)
        return {"saved": str(target / meta["stir"]), "check": check}


def _require(body: dict, key: str) -> Any:
    if not isinstance(body, dict) or key not in body:
        raise WorkspaceError(f"the request needs {key!r}")
    return body[key]


class _Handler(BaseHTTPRequestHandler):
    workspace: Workspace
    server_version = "stinkytofu-ir-editor"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter than the default
        if os.environ.get("STINKYTOFU_IR_EDITOR_LOG"):
            super().log_message(fmt, *args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _api(self, action: Callable[[], Any]) -> None:
        try:
            self._json(HTTPStatus.OK, action())
        except WorkspaceError as error:
            self._json(error.status, {"error": str(error), **error.details})
        except json.JSONDecodeError as error:
            self._json(
                HTTPStatus.BAD_REQUEST, {"error": f"the body is not JSON: {error}"}
            )

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802 - http.server's naming
        path = unquote(self.path.split("?", 1)[0])
        if path == "/api/bundles":
            return self._api(self.workspace.list_bundles)
        if path.startswith("/api/bundle/"):
            return self._api(
                lambda: self.workspace.get_bundle(path[len("/api/bundle/") :])
            )
        name = "index.html" if path in ("", "/") else path.lstrip("/")
        file = (STATIC_DIR / name).resolve()
        if STATIC_DIR.resolve() not in file.parents or not file.is_file():
            return self._json(HTTPStatus.NOT_FOUND, {"error": f"no such page {path}"})
        self._send(
            HTTPStatus.OK,
            file.read_bytes(),
            _CONTENT_TYPES.get(file.suffix, "application/octet-stream"),
        )

    def do_POST(self) -> None:  # noqa: N802
        routes = {
            "/api/analyze": self.workspace.analyze,
            "/api/validate": self.workspace.validate,
            "/api/save": self.workspace.save,
        }
        route = routes.get(self.path.split("?", 1)[0])
        if route is None:
            return self._json(
                HTTPStatus.NOT_FOUND, {"error": f"no such endpoint {self.path}"}
            )
        self._api(lambda: route(self._body()))


def make_server(
    workspace: Workspace, host: str = "127.0.0.1", port: int = 8765
) -> HTTPServer:
    """A single-threaded server: requests run one at a time, so analyses never overlap."""
    handler = type("Handler", (_Handler,), {"workspace": workspace})
    return HTTPServer((host, port), handler)
