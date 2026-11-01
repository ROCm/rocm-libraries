# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Direct compile sessions give the plain direct compiler's results."""

from __future__ import annotations

import threading

import pytest
from rocke.runtime import code_object
from rocke.runtime.code_object import CodeObjectError
from rocke.runtime.direct_session import DirectSession

_PROBE = """ckdsl.ir v1
kernel @session_probe {
  attrs { max_workgroup_size = i:64 }
  params {
    %Y : ptr<f32,global>
    %n : i32
  }
  region @entry {
    %tid : i32 = gpu.thread_id (  ) { axis = s:"x" }
BODY    cf.return (  )
  }
}
"""


def _kernel(lines: int) -> str:
    body = []
    for i in range(lines):
        body.append(
            f'    %k{i} : i32 = arith.constant (  ) {{ ity = s:"i32", value = i:{i % 7} }}\n'
        )
        body.append(f"    %a{i} : i32 = arith.add ( %tid, %k{i} )\n")
        body.append(f"    %e{i} : f32 = arith.bitcast ( %a{i} )\n")
        body.append(
            f"    memref.global_store_typed ( %Y, %a{i}, %e{i} ) {{ align = i:4 }}\n"
        )
    return _PROBE.replace("BODY", "".join(body))


def _require_direct_library() -> None:
    if code_object._direct_library_path() is None:
        pytest.skip("build the Rust cdylib or set ROCKE_CODEOBJ_DIRECT_LIB")


def _plain(ir: str):
    try:
        hsaco, _timings, _compiler, diagnostics = (
            code_object.build_direct_rocke_ir_hsaco(ir)
        )
    except CodeObjectError as error:
        return ("error", str(error))
    return (hsaco, diagnostics.split(" phases_ms=")[0])


def _outcome(call):
    try:
        hsaco, _timings, _compiler, diagnostics = call()
    except CodeObjectError as error:
        return ("error", str(error))
    return (hsaco, diagnostics.split(" phases_ms=")[0])


@pytest.mark.parametrize("lines", [1, 40, 900])
def test_session_compiles_match_the_plain_compiler(lines: int) -> None:
    _require_direct_library()
    ir = _kernel(lines)
    broken = ir.replace("( %tid, %k0 )", "( %nope, %k0 )")
    with DirectSession(worker_threads=2) as session:
        for text in (ir, broken):
            reference = _plain(text)
            assert _outcome(lambda text=text: session.compile(text)) == reference
            for piece in (7, 4096, len(text)):
                for at in range(0, len(text), piece):
                    session.feed(text[at : at + piece])
                assert _outcome(session.finish) == reference
    assert reference[0] == "error"


def test_session_is_shared_by_threads_and_closes() -> None:
    _require_direct_library()
    ir = _kernel(60)
    reference = _plain(ir)
    session = DirectSession()
    results = []

    def run() -> None:
        results.append(_outcome(lambda: session.compile(ir)))

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [reference] * 4
    session.close()
    with pytest.raises(CodeObjectError):
        session.compile(ir)
