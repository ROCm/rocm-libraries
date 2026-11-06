# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Bounded cold version negotiation; never used for warm kernel compilation."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from ._compiler_query import CompilerInfo, ComgrError, file_signature


def _kill(process):
    if os.name == "nt":
        taskkill = (
            Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/taskkill.exe"
        )
        subprocess.run(
            [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.kill()


def probe_version(
    path: str, *, timeout: float = 30, limit: int = 16384
) -> CompilerInfo:
    path = str(Path(path).absolute())
    before = file_signature(path)
    env = dict(os.environ)
    for key in ("LD_PRELOAD", "LD_AUDIT", "DYLD_INSERT_LIBRARIES"):
        env.pop(key, None)
    flags = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    command = [
        sys.executable,
        "-I",
        "-S",
        str(Path(__file__).with_name("_compiler_query.py")),
        path,
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, **flags
    )
    outputs = [bytearray(), bytearray()]
    overflow = threading.Event()

    def read(stream, output):
        while chunk := stream.read(1024):
            if len(output) + len(chunk) > limit:
                overflow.set()
                process.kill()
                break
            output.extend(chunk)
        stream.close()

    readers = [
        threading.Thread(target=read, args=(stream, output), daemon=True)
        for stream, output in zip((process.stdout, process.stderr), outputs)
    ]
    for thread in readers:
        thread.start()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        _kill(process)
        process.wait()
        raise ComgrError(f"compiler version probe timed out for {path!r}") from error
    finally:
        for thread in readers:
            thread.join(timeout=2)
    if overflow.is_set() or any(thread.is_alive() for thread in readers):
        _kill(process)
        raise ComgrError(f"compiler version probe exceeded output bounds for {path!r}")
    if process.returncode:
        diagnostic = outputs[1].decode("utf-8", "replace")
        raise ComgrError(
            f"compiler version probe failed for {path!r} (exit {process.returncode}): {diagnostic}"
        )
    try:
        document = json.loads(outputs[0])
        signature = document["signature"]
        if (
            type(document["protocol"]) is not int
            or document["protocol"] != 1
            or not isinstance(signature, list)
            or len(signature) != 4
            or any(type(value) is not int for value in signature)
            or tuple(signature) != before
            or file_signature(path) != before
        ):
            raise ValueError("file identity changed or unknown protocol")
        values = document["info"]
        version = values["llvm_version"]
        if (
            not isinstance(version, list)
            or len(version) != 3
            or any(type(value) is not int or value < 0 for value in version)
            or version[0] == 0
        ):
            raise ValueError("invalid compiler version")
        if (
            not isinstance(values["source"], str)
            or not values["source"]
            or not isinstance(values["requested_comgr"], str)
            or not isinstance(values["comgr_path"], str)
            or (
                values["query_library_path"] is not None
                and not isinstance(values["query_library_path"], str)
            )
        ):
            raise ValueError("invalid compiler provenance")
        if values["requested_comgr"] != path or not os.path.samefile(
            path, values["comgr_path"]
        ):
            raise ValueError("probe did not use the requested compiler")
        return CompilerInfo(
            tuple(version),
            values["source"],
            values["requested_comgr"],
            values["comgr_path"],
            values["query_library_path"],
        )
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise ComgrError(
            f"invalid compiler version evidence for {path!r}: {error}"
        ) from error
