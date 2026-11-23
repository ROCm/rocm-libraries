# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU numbering and the watch for other processes on a GPU, on fake KFD
trees (tests/gpu_fakes.py) and real processes that use no GPU."""

import json
import os
import pwd
import shutil
import stat
import subprocess
import sys
import threading

import pytest

import gpu_fakes as gf
from lib import kfd


@pytest.fixture
def sysfs(tmp_path, monkeypatch):
    root = gf.write_sysfs(tmp_path / "sys")
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(root))
    return root


def _fake_amd_smi(tmp_path, monkeypatch, entries):
    exe = tmp_path / "fake-amd-smi"
    exe.write_text(f"#!/bin/sh\ncat <<'EOF'\n{json.dumps(entries)}\nEOF\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(kfd.AMD_SMI_VAR, str(exe))


def _smi(g, index):
    return {"gpu": index, "bdf": g.bdf, "kfd_id": g.gpu_id, "node_id": g.node}


def test_gpus_are_numbered_in_node_order_with_their_pci_address(sysfs):
    gpus = kfd.gpus()
    assert [(g.index, g.node, g.gpu_id) for g in gpus] == [
        (0, 1, 1001),
        (1, 2, 1002),
        (2, 3, 1003),
    ]
    assert [g.bdf for g in gpus] == ["0000:0b:00.0", "0000:0a:00.0", "0000:0c:00.0"]
    assert [g.card for g in gpus] == [g.bdf for g in gpus]
    assert gpus[1].uuid == "GPU-0000000000002222"
    assert gpus[0].to_json()["kfd_node"] == 1
    assert kfd.device_dir(gpus[0]) == sysfs / "class" / "drm" / "renderD128" / "device"
    assert kfd.other_partitions(gpus[0]) == []


def test_amd_smi_is_cross_checked_against_the_topology(sysfs, tmp_path, monkeypatch):
    by_bus = sorted(gf.GPUS, key=lambda g: g.location_id)
    _fake_amd_smi(tmp_path, monkeypatch, [_smi(g, i) for i, g in enumerate(by_bus)])
    smi = kfd.amd_smi_list()
    assert kfd.verify_with_amd_smi(kfd.gpus(), smi) == {0: 1, 1: 0, 2: 2}
    wrong = [dict(_smi(by_bus[0], 0), node_id=3)]
    with pytest.raises(kfd.GpuMapError, match="gpu_id 1002 on KFD node 3"):
        kfd.verify_with_amd_smi(kfd.gpus(), wrong)
    assert kfd.verify_with_amd_smi(kfd.gpus(), [{"gpu": 0, "bdf": "x"}]) == {
        0: None,
        1: None,
        2: None,
    }
    monkeypatch.setenv(kfd.AMD_SMI_VAR, "")
    assert kfd.amd_smi_list() is None
    _fake_amd_smi(tmp_path, monkeypatch, {"not": "a list"})
    assert kfd.amd_smi_list() is None


@pytest.mark.parametrize(
    "env, device, physical",
    [
        ({}, 2, 2),
        ({"ROCR_VISIBLE_DEVICES": "2,0"}, 0, 2),
        ({"ROCR_VISIBLE_DEVICES": "2,0", "HIP_VISIBLE_DEVICES": "1"}, 0, 0),
        ({"CUDA_VISIBLE_DEVICES": "1,2"}, 1, 2),
        ({"HIP_VISIBLE_DEVICES": "2", "CUDA_VISIBLE_DEVICES": "1"}, 0, 2),
        ({"ROCR_VISIBLE_DEVICES": "GPU-0000000000003333"}, 0, 2),
        ({"ROCR_VISIBLE_DEVICES": "-1"}, 0, None),
        ({"HIP_VISIBLE_DEVICES": "-1"}, 3, None),
    ],
)
def test_physical_gpu_resolves_the_visible_device_lists(sysfs, env, device, physical):
    assert kfd.physical_gpu(device, env) == physical


@pytest.mark.parametrize(
    "env, device, message",
    [
        ({"ROCR_VISIBLE_DEVICES": "0,3,4"}, 3, "one of the 3 GPU"),
        ({"HIP_VISIBLE_DEVICES": "1,-1,0"}, 1, "does not name a device 1"),
        ({"ROCR_VISIBLE_DEVICES": "GPU-00000000000000ff"}, 0, "no GPU has the id"),
        ({}, -1, "not a device index"),
    ],
)
def test_device_lists_that_do_not_name_the_device_fail(sysfs, env, device, message):
    with pytest.raises(kfd.GpuMapError, match=message):
        kfd.physical_gpu(device, env)


@pytest.mark.parametrize(
    "version, name",
    [(90500, "gfx950"), (90402, "gfx942"), (90010, "gfx90a"), (110501, "gfx1151")],
)
def test_isa_name_spells_the_target_version_as_the_runtime_does(version, name):
    assert kfd.isa_name(version) == name
    assert gf.isa_properties(name, 0).startswith(f"gfx_target_version {version}\n")


def test_gpu_isa_reads_the_target_and_the_asic_revision(tmp_path, monkeypatch):
    root = gf.write_sysfs(tmp_path / "sys", isa=[("gfx950", 0), ("gfx942", 3), None])
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(root))
    assert [kfd.gpu_isa(g) for g in kfd.gpus()] == [
        ("gfx950", 0),
        ("gfx942", 3),
        (None, None),
    ]


def test_benches_use_the_gpus_of_their_devices_only():
    assert kfd.bench_gpus([1], {"ROCR_VISIBLE_DEVICES": "2,0,1"}) == [0]
    assert kfd.bench_gpus([0, 2], {"ROCR_VISIBLE_DEVICES": "2,0,1"}) == [1, 2]
    assert kfd.bench_gpus([0, 1], {"ROCR_VISIBLE_DEVICES": "-1"}) == []


def test_gpu_refs_name_gpus_by_index_pci_address_or_id(sysfs):
    got, unknown = kfd.gpu_refs([0, "0000:0A:00.0", "0c:00.0", "GPU-0000000000001111"])
    assert got == {0, 1, 2} and unknown == []
    got, unknown = kfd.gpu_refs([7, "ff:00.0", "GPU-dead"])
    assert got == {7} and unknown == ["7", "ff:00.0", "GPU-dead"]


def test_a_quoted_index_names_the_gpu_of_that_index(sysfs):
    got, unknown = kfd.gpu_refs(["1", " 2 ", "7"])
    assert got == {1, 2, 7} and unknown == ["7"]
    assert kfd.gpu_refs(["-1"]) == (set(), ["-1"])


def test_without_gpus_drops_the_devices_of_excluded_gpus(sysfs):
    env = {kfd.ROCR_VAR: "2,0,1"}
    assert kfd.without_gpus([0, 1, 2], {0}, env) == ([0, 2], [1])
    assert kfd.without_gpus([0, 5], {2}, env) == ([5], [0])
    assert kfd.without_gpus([0, 1], {0}, {kfd.ROCR_VAR: "-1"}) == ([0, 1], [])


def test_a_bench_launch_sees_its_gpu_alone(sysfs):
    env = {"ROCR_VISIBLE_DEVICES": "2,0", "HIP_VISIBLE_DEVICES": "1", "PATH": "/bin"}
    assert kfd.one_gpu_launch(0, env) == (
        "0",
        {"ROCR_VISIBLE_DEVICES": "0", "PATH": "/bin"},
    )
    uuid = {"ROCR_VISIBLE_DEVICES": "GPU-0000000000003333", "CUDA_VISIBLE_DEVICES": "0"}
    assert kfd.one_gpu_launch(0, uuid) == ("0", {"ROCR_VISIBLE_DEVICES": "2"})
    assert kfd.one_gpu_launch(1, {}) == ("0", {"ROCR_VISIBLE_DEVICES": "1"})
    hidden = {"ROCR_VISIBLE_DEVICES": "-1"}
    assert kfd.one_gpu_launch(3, hidden) == ("3", hidden)
    with pytest.raises(kfd.GpuMapError):
        kfd.one_gpu_launch(2, {"ROCR_VISIBLE_DEVICES": "0,1"})


def test_memory_users_test_the_value_not_the_file(sysfs):
    gf.set_memory(sysfs, 101, {1001: 4096})
    gf.set_memory(sysfs, 102, {})
    assert kfd.memory_users() == {101: {1001: 4096}}


def _gpu(index):
    return kfd.gpus()[index]


def test_the_watch_names_a_process_outside_the_stage(sysfs):
    with gf.outside_process() as pid:
        gf.set_memory(sysfs, pid, {1002: 8192})
        watch = kfd.GpuWatch(_gpu(1))
        assert watch.sample()
        assert watch.rule() == "foreign-process"
        (f,) = watch.foreign.values()
        assert (f.pid, f.comm, f.vram_bytes) == (pid, "sleep", 8192)
        assert f.user == pwd.getpwuid(os.getuid()).pw_name
        assert f.started is not None and f.first_seen
        assert f"pid {pid} ('sleep'" in watch.reason()
        rec = watch.record()
        assert rec["gpu"]["gpu_id"] == 1002 and rec["foreign"][0]["pid"] == pid
        assert not kfd.GpuWatch(_gpu(0)).sample()


def test_the_stage_its_children_and_bench_sessions_are_its_own(sysfs):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with gf.outside_process() as bench:
            gf.set_memory(sysfs, child.pid, {1001: 4096})
            gf.set_memory(sysfs, bench, {1001: 4096})
            watch = kfd.GpuWatch(_gpu(0))
            watch.add_bench(bench)
            assert not watch.sample() and not watch.check_end()
            assert watch.record()["foreign"] == []
            assert [f.pid for f in kfd.foreign_now(_gpu(0))] == [bench]
    finally:
        child.kill()
        child.wait()


def test_exited_and_later_processes_are_not_counted(sysfs):
    done = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    gf.set_memory(sysfs, int(done.stdout), {1001: 4096})
    assert not kfd.GpuWatch(_gpu(0)).check_end()
    with gf.outside_process() as pid:
        gf.set_memory(sysfs, pid, {1001: 4096})
        started = kfd.proc_info(pid).start_s
        assert not kfd.GpuWatch(_gpu(0)).check_end(ended=started - 1.0)
        assert kfd.GpuWatch(_gpu(0)).check_end()


def test_a_bench_with_memory_on_another_gpu_is_a_numbering_error(sysfs):
    with gf.outside_process() as bench:
        gf.set_memory(sysfs, bench, {1002: 4096})
        ok = kfd.GpuWatch(_gpu(1))
        ok.add_bench(bench)
        assert not ok.sample() and ok.mapping_error() is None
        gf.set_memory(sysfs, bench, {1001: 4096, 1002: 4096})
        wrong = kfd.GpuWatch(_gpu(1))
        wrong.add_bench(bench)
        assert wrong.sample()
        assert wrong.rule() == "gpu-numbering"
        assert "gpu_id(s) [1001]" in wrong.mapping_error()


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def quick_confirm(monkeypatch):
    monkeypatch.setattr(kfd, "UNATTRIBUTED_CONFIRM_S", 0.3)
    monkeypatch.setattr(kfd, "UNATTRIBUTED_STEP_S", 0.05)
    monkeypatch.setattr(kfd, "UNATTRIBUTED_SETTLE_S_PER_GIB", 0.0)


def test_vram_no_listed_process_holds_counts_when_it_lasts(sysfs, quick_confirm):
    allowance = kfd.UNATTRIBUTED_ALLOWANCE_BYTES
    gpu = gf.GPUS[1]
    with gf.outside_process() as bench:
        gf.set_memory(sysfs, bench, {1002: 3 << 30})
        gf.set_card_used(sysfs, gpu, (3 << 30) + allowance)
        watch = kfd.GpuWatch(_gpu(1), min_interval_s=0.0)
        watch.add_bench(bench)
        assert not watch.sample()
        gf.set_card_used(sysfs, gpu, (3 << 30) + (5 << 30))
        clear = threading.Timer(0.1, gf.set_card_used, (sysfs, gpu, gf.IDLE_CARD_BYTES))
        clear.start()
        assert not watch.sample()
        clear.join()
        gf.set_card_used(sysfs, gpu, (3 << 30) + (7 << 30))
        assert watch.sample() and watch.rule() == "unattributed-memory"
    rec = watch.record()
    assert rec["foreign"] == [] and rec["unattributed"]["vram_bytes"] == 7 << 30
    assert rec["unattributed"]["card_used_bytes"] == 10 << 30
    assert rec["unattributed"]["attributed_bytes"] == 3 << 30
    assert rec["unattributed"]["allowance_bytes"] == allowance
    assert rec["unattributed"]["settle_s"] == 0.3
    assert "7.00 GiB of VRAM no listed process holds" in rec["reason"]


def test_the_end_check_and_busy_count_lasting_unattributed_use(sysfs, quick_confirm):
    gf.set_card_used(sysfs, gf.GPUS[0], 2 << 30)
    watch = kfd.GpuWatch(_gpu(0))
    assert watch.check_end() and watch.unattributed.vram_bytes == 2 << 30
    (other,) = kfd.foreign_now(_gpu(0))
    assert isinstance(other, kfd.Unattributed)
    assert kfd.foreign_now(_gpu(1)) == []
    gf.set_card_used(sysfs, gf.GPUS[0], gf.IDLE_CARD_BYTES)
    assert not kfd.GpuWatch(_gpu(0)).check_end()


class _Timeline(_Clock):
    """A clock whose sleeps advance it and apply the sysfs changes due by
    then."""

    def __init__(self):
        super().__init__()
        self.due = []

    def at(self, t, change, *args):
        self.due = sorted(self.due + [(t, len(self.due), change, args)])

    def sleep(self, seconds):
        self.t += seconds
        while self.due and self.due[0][0] <= self.t:
            _, _, change, args = self.due.pop(0)
            change(*args)


def test_the_settle_window_grows_with_the_memory_up_to_its_cap():
    gib = 1 << 30
    assert kfd.settle_window_s(0) == kfd.UNATTRIBUTED_CONFIRM_S
    assert kfd.settle_window_s(10 * gib) == pytest.approx(
        kfd.UNATTRIBUTED_CONFIRM_S + 10 * kfd.UNATTRIBUTED_SETTLE_S_PER_GIB
    )
    assert kfd.settle_window_s(1 << 50) == kfd.UNATTRIBUTED_SETTLE_MAX_S


def test_memory_a_large_bench_leaves_on_exit_settles_without_counting(sysfs):
    """A bench that held most of a card exits: its KFD entry goes at once
    and the card shows its memory in use for a few seconds more, longer than
    UNATTRIBUTED_CONFIRM_S but within the window the bench's memory sets."""
    gpu, tl = gf.GPUS[2], _Timeline()
    held = 60 << 30
    with gf.outside_process() as bench:
        gf.set_memory(sysfs, bench, {1003: held})
        gf.set_card_used(sysfs, gpu, gf.IDLE_CARD_BYTES + held)
        watch = kfd.GpuWatch(_gpu(2), clock=tl, sleep=tl.sleep, min_interval_s=0.0)
        watch.add_bench(bench)
        assert not watch.sample() and watch.bench_peak_bytes == held
        shutil.rmtree(gf.kfd_root(sysfs) / "proc" / str(bench))
        gf.set_card_used(sysfs, gpu, gf.IDLE_CARD_BYTES + held // 2)
        tl.at(2.5, gf.set_card_used, sysfs, gpu, gf.IDLE_CARD_BYTES)
        assert not watch.check_end()
    assert tl.t == pytest.approx(2.5)
    assert kfd.settle_window_s(held) > tl.t > kfd.UNATTRIBUTED_CONFIRM_S
    assert watch.unattributed is None and watch.record()["rule"] is None


def test_unattributed_memory_that_outlasts_its_window_counts(sysfs):
    tl = _Timeline()
    gf.set_card_used(sysfs, gf.GPUS[0], 20 << 30)
    watch = kfd.GpuWatch(_gpu(0), clock=tl, sleep=tl.sleep)
    assert watch.check_end() and watch.rule() == "unattributed-memory"
    window = kfd.settle_window_s(20 << 30)
    assert watch.unattributed.settle_s == pytest.approx(window)
    assert window <= tl.t < window + kfd.UNATTRIBUTED_STEP_S
    assert watch.record()["unattributed"]["settle_s"] == pytest.approx(window)
    assert f"for over {window:.1f} s" in watch.reason()


def test_a_benchs_peak_sets_the_window_of_what_lingers_after_it(sysfs):
    tl = _Timeline()
    peak, left = 60 << 30, 2 << 30
    with gf.outside_process() as bench:
        gf.set_memory(sysfs, bench, {1001: peak})
        gf.set_card_used(sysfs, gf.GPUS[0], gf.IDLE_CARD_BYTES + peak)
        watch = kfd.GpuWatch(_gpu(0), clock=tl, sleep=tl.sleep, min_interval_s=0.0)
        watch.add_bench(bench)
        assert not watch.sample()
        shutil.rmtree(gf.kfd_root(sysfs) / "proc" / str(bench))
        gf.set_card_used(sysfs, gf.GPUS[0], left)
        assert watch.check_end()
    assert watch.unattributed.settle_s == pytest.approx(kfd.settle_window_s(peak))
    assert tl.t >= kfd.settle_window_s(peak) > kfd.settle_window_s(left)


def test_a_foreign_process_is_named_at_once_while_memory_settles(sysfs):
    tl = _Timeline()
    with gf.outside_process() as pid:
        gf.set_memory(sysfs, pid, {1001: 8 << 30})
        gf.set_card_used(sysfs, gf.GPUS[0], (8 << 30) + (40 << 30))
        watch = kfd.GpuWatch(_gpu(0), clock=tl, sleep=tl.sleep)
        assert watch.sample() and watch.rule() == "foreign-process"
        assert [f.pid for f in kfd.foreign_now(_gpu(0))] == [pid]
    assert tl.t == 0.0 and watch.unattributed is None


def test_quarantine_moves_the_files_and_writes_the_record(tmp_path):
    log = tmp_path / "logs" / "block.log"
    log.parent.mkdir()
    log.write_text("bench output\n")
    dest = tmp_path / "quarantine" / "block_q1"
    rec = kfd.quarantine(dest, [log, tmp_path / "missing.log", None], {"rule": "x"})
    assert not log.exists() and (dest / "block.log").read_text() == "bench output\n"
    on_disk = json.loads((dest / kfd.QUARANTINE_RECORD).read_text())
    assert on_disk == rec and rec["files"] == ["block.log"] and rec["rule"] == "x"
    assert rec["quarantined"].endswith("Z")


def test_stage_gpus_describe_what_they_cannot_watch(sysfs, monkeypatch):
    for var in kfd.HIP_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2")
    gpus = kfd.StageGpus([0, 2])
    assert gpus.unwatched() == []
    assert gpus.watch(2).gpu.gpu_id == 1003
    assert gpus.retire(2, "busy") and not gpus.retire(2, "again")
    assert gpus.first_active([2, 0]) == 0
    assert not gpus.recheck(2) and not gpus.may_return(2)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "-1")
    hidden = kfd.StageGpus([0])
    assert hidden.watch(0) is None
    assert "hides every GPU" in hidden.unwatched_note()


def test_a_gpu_retired_for_other_use_returns_after_a_clean_spell(sysfs, monkeypatch):
    for var in kfd.HIP_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(kfd, "UNATTRIBUTED_CONFIRM_S", 0.0)
    monkeypatch.setattr(kfd, "UNATTRIBUTED_SETTLE_S_PER_GIB", 0.0)
    monkeypatch.setattr(kfd, "READMIT_CLEAN_S", 100.0)
    clock = _Clock()
    gpus = kfd.StageGpus([0, 1], clock=clock)
    assert gpus.recheck(0) and gpus.all_retired_for() is None
    gf.set_card_used(sysfs, gf.GPUS[1], 4 << 30)
    assert gpus.retire(1, "hidden use", temporary=True)
    assert gpus.retire(0, "numbering")
    clock.t = 10.0
    assert gpus.all_retired_for() == 10.0
    assert not gpus.recheck(1)
    clock.t = 20.0
    gf.set_card_used(sysfs, gf.GPUS[1], gf.IDLE_CARD_BYTES)
    assert not gpus.recheck(1)
    clock.t = 110.0
    assert not gpus.recheck(1)
    clock.t = 120.0
    assert gpus.recheck(1) and gpus.active(1) and gpus.all_retired_for() is None
    (spell,) = gpus.returned
    assert (spell["device"], spell["why"]) == (1, "hidden use")
    assert spell["retired"] <= spell["returned"]
    assert not gpus.recheck(0) and gpus.retired == {0: "numbering"}


# ── compute partitions ───────────────────────────────────────────────────────


@pytest.fixture
def dpx(tmp_path, monkeypatch):
    root = gf.write_dpx_sysfs(tmp_path / "sys")
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(root))
    return root


@pytest.mark.parametrize("other_function", [True, False])
def test_the_partitions_of_a_card_read_the_cards_sysfs(
    tmp_path, monkeypatch, other_function
):
    """KFD gives partition 1 of the DPX card the card's address with 1 as
    the function, a device of another kind or none: both partitions read
    the card's PCI device, and GPU 0, which is no partition and whose PCI
    device amdgpu drives, reads its own as before."""
    root = gf.write_dpx_sysfs(tmp_path / "sys", other_function=other_function)
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(root))
    gf.set_card_used(root, gf.DPX[0], 5 << 30)
    g0, p0, p1 = kfd.gpus()
    assert [g.bdf for g in (g0, p0, p1)] == [
        "0000:0b:00.0",
        "0000:0d:00.0",
        "0000:0d:00.1",
    ]
    assert [g.card for g in (g0, p0, p1)] == [g0.bdf, p0.bdf, p0.bdf]
    assert p1.to_json()["card"] == p0.bdf
    card = gf.pci_dir(root, gf.DPX[0])
    assert kfd.device_dir(p1) == card
    assert kfd.device_dir(p0).resolve() == card.resolve()
    assert kfd.device_dir(g0) == root / "class" / "drm" / "renderD128" / "device"
    assert [kfd.gpu_xcc(g) for g in (g0, p0, p1)] == [
        (3, "SPX"),
        (3, "DPX"),
        (3, "DPX"),
    ]
    assert kfd.card_vram_used(p0) == kfd.card_vram_used(p1) == 5 << 30
    assert kfd.card_vram_used(g0) == gf.IDLE_CARD_BYTES
    assert kfd.other_partitions(p0) == [p1] and kfd.other_partitions(p1) == [p0]
    assert kfd.other_partitions(g0) == []


def test_the_stages_own_memory_on_another_partition_is_attributed(dpx):
    """A process of the stage holds memory on partition 1, and the card's
    use counts it: partition 0's watch names no other process and finds no
    unattributed use, and that memory sets the settle window as the
    benches' own does."""
    tl = _Timeline()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        gf.set_memory(dpx, child.pid, {2002: 6 << 30})
        gf.set_card_used(dpx, gf.DPX[0], gf.IDLE_CARD_BYTES + (6 << 30))
        _, p0, p1 = kfd.gpus()
        assert kfd.vram_balance(p0)[1:] == (gf.IDLE_CARD_BYTES + (6 << 30), 6 << 30)
        watch = kfd.GpuWatch(p0, clock=tl, sleep=tl.sleep, min_interval_s=0.0)
        assert not watch.sample() and not watch.check_end()
        rec = watch.record()
        assert rec["rule"] is None and rec["foreign"] == []
        assert rec["unattributed"] is None and tl.t == 0.0
        assert watch.bench_peak_bytes == 6 << 30
        assert kfd.foreign_now(p0) == [] and kfd.foreign_now(p1) == []
    finally:
        child.kill()
        child.wait()


def test_a_process_outside_the_stage_on_another_partition_is_a_tenant(dpx):
    """A process outside the stage with memory on partition 1 shares the
    card's power with partition 0's benches: partition 0's watch names it
    under the partitions' rule, and as a process on its GPU with what it
    holds there once a later scan sees it there; partition 1's names it as
    a process on its GPU, and the GPU of another card does not see it."""
    with gf.outside_process() as pid:
        gf.set_memory(dpx, pid, {2002: 8192})
        g0, p0, p1 = kfd.gpus()
        watch = kfd.GpuWatch(p0)
        assert watch.sample() and watch.rule() == "sibling-partition"
        (f,) = watch.foreign.values()
        assert (f.pid, f.comm, f.vram_bytes, f.partitions) == (pid, "sleep", 8192, (2,))
        assert watch.reason().startswith(f"pid {pid} ('sleep'")
        assert watch.reason().endswith(
            "on GPU 2 of the same card) beside GPU 1 (gpu_id 2001, 0000:0d:00.0) "
            "while the bench ran"
        )
        assert watch.record()["foreign"][0]["partitions"] == [2]
        assert [o.pid for o in kfd.foreign_now(p0)] == [pid]
        assert kfd.other_use_rule(kfd.foreign_now(p0)) == "sibling-partition"
        own_gpu = kfd.GpuWatch(p1)
        assert own_gpu.sample() and own_gpu.rule() == "foreign-process"
        assert own_gpu.record()["foreign"][0]["partitions"] == []
        assert not kfd.GpuWatch(g0).sample()
        gf.set_memory(dpx, pid, {2001: 4096, 2002: 8192})
        both = kfd.GpuWatch(p0)
        assert both.sample() and both.rule() == "foreign-process"
        assert both.foreign[pid].vram_bytes == 4096
        assert kfd.other_use_rule(kfd.foreign_now(p0)) == "foreign-process"
        assert watch.check_end() and watch.rule() == "foreign-process"
        f = watch.foreign[pid]
        assert (f.partitions, f.vram_bytes) == ((), 4096)
        assert "of the same card" not in watch.reason()
        assert watch.reason().endswith(
            " on GPU 1 (gpu_id 2001, 0000:0d:00.0) while the bench ran"
        )
        gf.set_memory(dpx, pid, {2002: 1 << 30})
        assert watch.check_end() and watch.foreign[pid].vram_bytes == 4096


def test_the_runs_other_bench_stages_on_another_partition_are_not_tenants(dpx):
    """Two bench stages of one run, each a process tree of its own under
    the run's root (this process), bench on the partitions of the DPX card.
    Partition 0's watch takes the other stage's bench on partition 1 for
    the run's (`run_root`, `marker`): it names no tenant and counts that
    memory as its own processes', and without the run's root it names the
    bench under the partitions' rule. A process under the run's root in no
    bench stage's tree is a tenant there, and so is the other stage's bench
    once it holds memory on partition 0."""
    marker, tl = "stage03_bench.py", _Timeline()
    sleep = [sys.executable, "-c", "import time; time.sleep(60)"]
    with gf.bench_stage(marker) as (stage, _), gf.bench_stage(marker) as other:
        bench, work = other[1], subprocess.Popen(sleep)
        try:
            gf.set_memory(dpx, bench, {2002: 6 << 30})
            gf.set_card_used(dpx, gf.DPX[0], gf.IDLE_CARD_BYTES + (6 << 30))
            _, p0, _ = kfd.gpus()
            run = dict(own_root=stage, run_root=os.getpid(), marker=marker)
            watch = kfd.GpuWatch(
                p0, clock=tl, sleep=tl.sleep, min_interval_s=0.0, **run
            )
            assert not watch.sample() and not watch.check_end()
            assert watch.record()["rule"] is None and tl.t == 0.0
            assert watch.bench_peak_bytes == 6 << 30
            assert kfd.foreign_now(p0, **run) == []
            alone = kfd.GpuWatch(p0, own_root=stage)
            assert alone.check_end() and alone.rule() == "sibling-partition"
            assert list(alone.foreign) == [bench]
            gf.set_memory(dpx, work.pid, {2002: 8192})
            assert watch.check_end() and watch.rule() == "sibling-partition"
            assert list(watch.foreign) == [work.pid]
            gf.set_memory(dpx, bench, {2001: 4096, 2002: 6 << 30})
            on_gpu = kfd.GpuWatch(p0, **run)
            assert on_gpu.check_end() and on_gpu.rule() == "foreign-process"
            assert on_gpu.foreign[bench].partitions == ()
        finally:
            work.kill()
            work.wait()


def test_unattributed_memory_on_a_partitioned_card_still_counts(dpx):
    """The card's use beyond what the processes hold on its partitions is
    unattributed for each partition, partition 1 included, whose address
    names no GPU."""
    tl = _Timeline()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        gf.set_memory(dpx, child.pid, {2002: 2 << 30})
        gf.set_card_used(dpx, gf.DPX[0], (2 << 30) + (5 << 30))
        g0, p0, p1 = kfd.gpus()
        for g in (p0, p1):
            watch = kfd.GpuWatch(g, clock=tl, sleep=tl.sleep)
            assert watch.check_end() and watch.rule() == "unattributed-memory"
            u = watch.record()["unattributed"]
            assert (u["vram_bytes"], u["card_used_bytes"], u["attributed_bytes"]) == (
                5 << 30,
                7 << 30,
                2 << 30,
            )
        assert not kfd.GpuWatch(g0, clock=tl, sleep=tl.sleep).check_end()
    finally:
        child.kill()
        child.wait()


def test_other_use_is_beside_the_gpu_while_every_process_is_on_another_partition():
    """The rule of a GPU's other use, as `StageGpus.busy` lists it: the
    partitions' rule while every process of it holds memory on other
    partitions of the card alone, the foreign-process rule once one holds
    memory on the GPU, and unattributed memory only without a process."""
    beside = kfd.Foreign(1, "a", None, None, 8192, "t0", "t0", partitions=(2,))
    on = kfd.Foreign(2, "b", None, None, 4096, "t0", "t0")
    hidden = kfd.Unattributed(1 << 30, 2 << 30, 1 << 30, "t0", "t0")
    assert kfd.other_use_rule([]) is None
    assert kfd.other_use_rule([hidden]) == "unattributed-memory"
    assert kfd.other_use_rule([beside]) == "sibling-partition"
    assert kfd.other_use_rule([beside, hidden]) == "sibling-partition"
    assert kfd.other_use_rule([beside, on]) == "foreign-process"
