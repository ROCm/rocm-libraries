# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Kernel ids and applicability declines parsed from the backend's INFO log, without the library."""

import ctypes
import json
import os
import sys
import threading
import types

import pytest

from hipdnn_sdpa import _kernel_log, _runtime

contract = _runtime.contract

PLAIN = (
    "[hipkernel] ingestor: engine '{engine}' selected kernel {kernel} at rank 0 from 3 "
    "candidate(s) (5 before knob filtering) ranked by metric 'latency'"
)
SLOTTED = (
    "[hipkernel] ingestor: engine '{engine}' slot '{slot}' selected kernel {kernel} at rank 0 "
    "from 2 candidate(s)"
)
KERNEL_A = "0b1c2d3e-0000-4000-8000-00000000000a"
KERNEL_B = "0b1c2d3e-0000-4000-8000-00000000000b"
KERNEL_C = "0b1c2d3e-0000-4000-8000-00000000000c"


def selected(engine, kernel):
    return PLAIN.format(engine=engine, kernel=kernel)


def slot(engine, name, kernel):
    return SLOTTED.format(engine=engine, slot=name, kernel=kernel)


def test_a_build_yields_the_kernel_its_engine_selected():
    with _kernel_log.capture() as built:
        _kernel_log.on_line(selected("flydsl_sdpa_cdna", KERNEL_A))
    assert built.kernel("hipkernel:flydsl_sdpa_cdna") == KERNEL_A


def test_slotted_engine_kernels_join_in_slot_order():
    with _kernel_log.capture() as built:
        _kernel_log.on_line(slot("flydsl_sdpa_bwd_split", "prep", KERNEL_A))
        _kernel_log.on_line(slot("flydsl_sdpa_bwd_split", "dkdv", KERNEL_B))
        _kernel_log.on_line(slot("flydsl_sdpa_bwd_split", "dq", KERNEL_C))
    assert built.kernel("hipkernel:flydsl_sdpa_bwd_split") == (
        f"prep={KERNEL_A}|dkdv={KERNEL_B}|dq={KERNEL_C}"
    )


def test_the_kernel_of_the_served_engine_wins_over_one_that_fell_back():
    with _kernel_log.capture() as built:
        _kernel_log.on_line(selected("flydsl_sdpa_cdna", KERNEL_A))
        _kernel_log.on_line(selected("flydsl_sdpa_cdna_dualwave", KERNEL_B))
    assert built.kernel("hipkernel:flydsl_sdpa_cdna") == KERNEL_A
    assert built.kernel("hipkernel:flydsl_sdpa_cdna_dualwave") == KERNEL_B


def test_a_build_without_a_selection_has_no_kernel():
    with _kernel_log.capture() as built:
        _kernel_log.on_line("[hipkernel] some unrelated INFO line")
    assert built.kernel("ASM_SDPA_ENGINE") == ""


def test_lines_outside_a_build_are_ignored():
    _kernel_log.on_line(selected("flydsl_sdpa_cdna", KERNEL_A))
    with _kernel_log.capture() as built:
        pass
    assert built.kernel("hipkernel:flydsl_sdpa_cdna") == ""


def test_lines_from_other_threads_are_ignored():
    with _kernel_log.capture() as built:
        worker = threading.Thread(
            target=_kernel_log.on_line, args=(selected("flydsl_sdpa_cdna", KERNEL_B),)
        )
        worker.start()
        worker.join()
        _kernel_log.on_line(selected("flydsl_sdpa_cdna", KERNEL_A))
    assert built.kernel("hipkernel:flydsl_sdpa_cdna") == KERNEL_A


def test_a_later_build_does_not_inherit_an_earlier_kernel():
    with _kernel_log.capture():
        _kernel_log.on_line(selected("flydsl_sdpa_cdna", KERNEL_A))
    with _kernel_log.capture() as second:
        pass
    assert second.kernel("hipkernel:flydsl_sdpa_cdna") == ""


# ---------------------------------------------------------------------------- declines

ASM = "[hipkernel] [SdpaFwdPlanBuilder::isApplicable] Device string does not match gfx942 or gfx950 (Actual value: {n})"
INGESTOR = (
    "[hipkernel] ingestor: engine 'flydsl_sdpa_cdna' declined the graph on device 0: "
    "slot 'x' has no applicable kernel ({n})"
)


@pytest.fixture
def declines_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(contract.DECLINES_DIR_ENV, str(tmp_path))
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")
    monkeypatch.setattr(_kernel_log, "_DECLINE_COUNTS", {})
    return tmp_path


def current_nodeid():
    return os.environ["PYTEST_CURRENT_TEST"].rsplit(" (", 1)[0]


def read_declines(directory):
    path = directory / "gw3.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_only_the_first_declines_of_each_engine_are_kept(declines_dir):
    limit = contract.DECLINES_PER_ENGINE
    with _kernel_log.capture():
        for n in range(limit + 5):
            _kernel_log.on_line(ASM.format(n=n))
        _kernel_log.on_line(INGESTOR.format(n=0))
    rows = read_declines(declines_dir)
    asm = [r for r in rows if r["engine"] == "SdpaFwdPlanBuilder"]
    assert len(asm) == limit
    assert "Actual value: 0)" in asm[0]["line"]
    assert "Actual value: %d)" % (limit - 1) in asm[-1]["line"]
    ingestor = [r for r in rows if r["engine"] == "flydsl_sdpa_cdna"]
    assert len(ingestor) == 1 and ingestor[0]["nodeid"] == current_nodeid()


def test_declines_outside_a_build_are_not_kept(declines_dir):
    _kernel_log.on_line(ASM.format(n=0))
    assert read_declines(declines_dir) == []


def test_a_selection_line_is_not_a_decline(declines_dir):
    with _kernel_log.capture():
        _kernel_log.on_line(selected("flydsl_sdpa_cdna", KERNEL_A))
    assert read_declines(declines_dir) == []


def test_declines_are_dropped_when_no_directory_is_configured(monkeypatch, tmp_path):
    monkeypatch.delenv(contract.DECLINES_DIR_ENV, raising=False)
    monkeypatch.setattr(_kernel_log, "_DECLINE_COUNTS", {})
    with _kernel_log.capture():
        _kernel_log.on_line(ASM.format(n=0))
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------- install


class FakeBackend:
    """Stand-in for the ctypes view of ``libhipdnn_backend``; plain functions take ``argtypes``."""

    def __init__(self, status=0):
        self.calls = []

        def set_callback(callback, level, mode, handle):
            self.calls.append(("callback", level, mode, handle))
            self.callback = callback
            return status

        def set_level(level):
            self.calls.append(("level", level))
            return 0

        self.lib = types.SimpleNamespace(
            hipdnnSetUserLogCallback_ext=set_callback,
            hipdnnBackendSetGlobalLogLevel_ext=set_level,
        )


@pytest.fixture
def fresh_install(monkeypatch):
    monkeypatch.setattr(_kernel_log, "_INSTALLED", None)


def test_install_registers_the_sync_callback_before_raising_the_level(fresh_install):
    backend = FakeBackend()
    assert _kernel_log._install(backend.lib) is True
    kinds = [c[0] for c in backend.calls]
    assert kinds == ["callback", "level"]
    _, level, mode, handle = backend.calls[0]
    assert (level, mode) == (0, 0)  # HIPDNN_SEV_INFO, HIPDNN_LOG_CALLBACK_SYNC
    assert handle  # userHandle must be non-null
    assert backend.calls[1] == ("level", 0)


def test_the_registered_callback_decodes_a_c_string_into_the_build(fresh_install):
    backend = FakeBackend()
    _kernel_log._install(backend.lib)
    message = selected("flydsl_sdpa_cdna", KERNEL_A).encode()
    with _kernel_log.capture() as built:
        backend.callback(ctypes.c_void_p(1), 0, message)
    assert built.kernel("hipkernel:flydsl_sdpa_cdna") == KERNEL_A


def test_uninstall_removes_the_callback(fresh_install):
    backend = FakeBackend()
    _kernel_log._install(backend.lib)
    _kernel_log._uninstall()
    assert backend.calls[-1][:2] == ("callback", 4)  # HIPDNN_SEV_OFF removes it


def test_install_reports_false_when_the_callback_is_refused(fresh_install):
    backend = FakeBackend(status=2)
    assert _kernel_log._install(backend.lib) is False
    assert [c[0] for c in backend.calls] == ["callback"]


def test_install_is_a_no_op_reporting_false_without_a_backend(
    fresh_install, monkeypatch
):
    monkeypatch.setattr(_kernel_log, "_mapped_backend", lambda: None)
    monkeypatch.setitem(sys.modules, "hipdnn_frontend", None)
    assert _kernel_log.install_kernel_log() is False


def test_the_backend_is_found_in_the_process_maps(tmp_path):
    maps = tmp_path / "maps"
    maps.write_text(
        "7f00-7f10 r-xp 00000000 08:01 1 /usr/lib/libc.so.6\n"
        "7f20-7f30 r-xp 00000000 08:01 2 /opt/rocm/lib/libhipdnn_backend.so.1\n"
    )
    assert (
        _kernel_log._mapped_backend(str(maps)) == "/opt/rocm/lib/libhipdnn_backend.so.1"
    )
    maps.write_text("7f00-7f10 r-xp 00000000 08:01 1 /usr/lib/libc.so.6\n")
    assert _kernel_log._mapped_backend(str(maps)) is None


@pytest.mark.gpu
def test_registration_on_the_real_backend_captures_a_kernel(cuda_device):
    import torch

    import hipdnn_sdpa

    assert hipdnn_sdpa.install_kernel_log() is True
    q, k, v = (
        torch.randn(1, 2, 128, 128, device=cuda_device, dtype=torch.bfloat16)
        for _ in range(3)
    )
    o = torch.empty_like(q)
    stats = torch.empty(1, 2, 128, 1, device=cuda_device, dtype=torch.float32)
    engine = hipdnn_sdpa.sdpa_fwd(q, k, v, o=o, stats=stats, scale=0.1)
    record = hipdnn_sdpa.last_call("fwd")
    assert record.engine == engine and isinstance(record.kernel, str)
