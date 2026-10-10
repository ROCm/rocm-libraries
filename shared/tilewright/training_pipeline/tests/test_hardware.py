# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from lib import hardware as hw


def test_gfx950_constants():
    c = hw.arch_constants("gfx950")
    assert c.parallel_mi_cu == 4.0
    assert c.bw == (-0.000013, 0.007070, 0.027355)
    assert c.mi_default == 32.0
    assert len(c.mi_table) == 17
    assert c.mi_table[(32, 32, 16, "bfloat16")] == 32
    assert c.mi_table[(16, 16, 32, "xfloat32")] == 48
    assert c.mi_table[(32, 32, 8, "xfloat32")] == 96
    assert c.mi_table[(16, 16, 128, "float8")] == 16


@pytest.mark.parametrize("arch", ["gfx942", "gfx942:sramecc+"])
def test_an_arch_without_its_own_table_is_an_error(arch):
    assert hw.arch_family(arch) is None
    with pytest.raises(ValueError, match="no model constants"):
        hw.arch_constants(arch)


@pytest.mark.parametrize(
    "arch", ["gfx1250", "gfx1250v0", "gfx1250-strict", "gfx1250v0:xnack+"]
)
def test_gfx1250_family(arch):
    c = hw.arch_constants(arch)
    assert c.parallel_mi_cu == 4.0
    assert c.bw == (0.0, 0.016, 0.0)
    assert len(c.mi_table) == 19
    assert c.mi_table[(16, 16, 128, "float8")] == 8
    assert c.mi_table[(16, 16, 64, "bfloat8float8")] == 4
    assert c.mi_table[(32, 16, 128, "float4")] == 8


def test_unknown_arch_is_an_error():
    with pytest.raises(ValueError):
        hw.arch_constants("gfx90a")
    assert hw.arch_family("gfx1100") is None


def test_constants_are_independent_copies():
    a = hw.arch_constants("gfx950")
    a.mi_table[(16, 16, 32, "bfloat16")] = 1.0
    assert hw.arch_constants("gfx950").mi_table[(16, 16, 32, "bfloat16")] == 16


def test_table_dtypes_have_datatype_values():
    for arch in ("gfx950", "gfx1250"):
        for _m, _n, _k, dt in hw.arch_constants(arch).mi_table:
            assert dt in hw.DATATYPE_VALUE


def test_datatype_values_match_design_and_engine():
    expected = {
        "float": 0,
        "double": 1,
        "complexfloat": 2,
        "complexdouble": 3,
        "half": 4,
        "int32": 6,
        "bfloat16": 7,
        "int8": 8,
        "int4": 9,
        "xfloat32": 11,
        "float8": 16,
        "bfloat8": 17,
        "float8bfloat8": 18,
        "bfloat8float8": 19,
        "float6": 20,
        "bfloat6": 21,
        "float4": 22,
    }
    for name, value in expected.items():
        assert hw.DATATYPE_VALUE[name] == value
    tw = pytest.importorskip("tilewright")
    by_lower = {k.lower(): int(v) for k, v in tw.DataType.__members__.items()}
    for name, value in hw.DATATYPE_VALUE.items():
        assert by_lower[name] == value


@pytest.mark.parametrize(
    "fields", [(0, 1, 1), (1, -1, 1), (1, 1, 0), (True, 1, 1), (1.5, 1, 1)]
)
def test_device_hardware_rejects_bad_values(fields):
    with pytest.raises(ValueError):
        hw.DeviceHardware(*fields)


def _node(root: Path, idx: int, props: dict, caches=()) -> None:
    d = root / str(idx)
    d.mkdir(parents=True)
    (d / "properties").write_text("".join(f"{k} {v}\n" for k, v in props.items()))
    for ci, (level, size_kib) in enumerate(caches):
        cd = d / "caches" / str(ci)
        cd.mkdir(parents=True)
        (cd / "properties").write_text(
            f"processor_id_low 0\nlevel {level}\nsize {size_kib}\n"
        )


@pytest.fixture
def topology(tmp_path, monkeypatch):
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    root = tmp_path / "nodes"
    _node(root, 0, {"cpu_cores_count": 8, "simd_count": 0})
    _node(
        root,
        1,
        {"simd_count": 40, "simd_per_cu": 4, "lds_size_in_kb": 32},
        caches=[(1, 16), (2, 1024), (2, 1024)],
    )
    _node(
        root,
        10,
        {"simd_count": 64, "simd_per_cu": 2, "lds_size_in_kb": 48},
        caches=[(2, 512), (3, 8192)],
    )
    return root


def test_kfd_topology(topology):
    assert hw.kfd_device_hardware(0, topology) == hw.DeviceHardware(
        10, 32 * 1024, 1024 * 1024
    )
    assert hw.kfd_device_hardware(1, topology) == hw.DeviceHardware(
        32, 48 * 1024, 512 * 1024
    )
    with pytest.raises(RuntimeError):
        hw.kfd_device_hardware(2, topology)


def test_kfd_visible_devices(topology, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
    assert hw.kfd_device_hardware(0, topology).n_cu == 32
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "GPU-0123")
    with pytest.raises(ValueError):
        hw.kfd_device_hardware(0, topology)


def test_a_negative_visible_index_hides_the_rest(topology, monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1,-1,0")
    assert hw.kfd_device_hardware(0, topology).n_cu == 32
    with pytest.raises(RuntimeError, match="1 visible with HIP_VISIBLE_DEVICES"):
        hw.kfd_device_hardware(1, topology)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "-1")
    with pytest.raises(RuntimeError, match="0 visible with HIP_VISIBLE_DEVICES=-1"):
        hw.kfd_device_hardware(0, topology)


def test_config_block_wins_and_completes(topology, tmp_path):
    full = {"hardware": {"n_cu": 7, "lds_bytes": 1024, "l2_bytes": 2048}}
    assert hw.device_hardware(full, 0, tmp_path / "absent") == hw.DeviceHardware(
        7, 1024, 2048
    )
    partial = {"hardware": {"n_cu": 7}}
    assert hw.device_hardware(partial, 0, topology) == hw.DeviceHardware(
        7, 32 * 1024, 1024 * 1024
    )
    assert hw.device_hardware({}, 1, topology).n_cu == 32
    with pytest.raises(ValueError):
        hw.device_hardware({"hardware": {"cu_count": 7}}, 0, topology)
    with pytest.raises(RuntimeError):
        hw.device_hardware(partial, 0, tmp_path / "absent")


def test_a_value_the_node_does_not_give_needs_its_key(topology):
    props = topology / "1" / "properties"
    props.write_text(props.read_text().replace("lds_size_in_kb 32\n", ""))
    with pytest.raises(ValueError, match="gives no lds_size_in_kb: set hardware.lds"):
        hw.device_hardware({}, 0, topology)
    pinned = {"hardware": {"lds_bytes": 4096}}
    assert hw.device_hardware(pinned, 0, topology) == hw.DeviceHardware(
        10, 4096, 1024 * 1024
    )


def test_pci_chip_id_from_config_or_kfd(topology, tmp_path):
    (topology / "1" / "properties").write_text(
        (topology / "1" / "properties").read_text() + "device_id 30112\n"
    )
    assert hw.device_pci_chip_id({}, 0, topology) == 0x75A0
    assert hw.device_pci_chip_id({}, 1, topology) is None
    cfg = {"hardware": {"pci_chip_id": 0x75A8}}
    assert hw.device_pci_chip_id(cfg, 1, topology) == 0x75A8
    assert hw.device_pci_chip_id({}, 0, tmp_path / "absent") is None
    full = {"hardware": {"n_cu": 7, "lds_bytes": 1, "l2_bytes": 2, "pci_chip_id": 1}}
    assert hw.device_hardware(full, 0, tmp_path / "absent").n_cu == 7


def test_device_n_cu_reads_the_topology_only_without_a_configured_one(
    topology, tmp_path
):
    assert hw.device_n_cu({"hardware": {"n_cu": 7}}, 0, tmp_path / "absent") == 7
    assert hw.device_n_cu({}, 1, topology) == 32
    with pytest.raises(RuntimeError):
        hw.device_n_cu({"hardware": {"lds_bytes": 1}}, 0, tmp_path / "absent")


def test_device_num_xcd_from_config_or_a_whole_device(tmp_path, monkeypatch):
    import gpu_fakes as gf
    from lib import kfd

    for var in kfd.HIP_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv(kfd.ROCR_VAR, raising=False)
    root = gf.write_sysfs(tmp_path / "sys", xcc=[(3, "SPX"), (1, "CPX"), (5, None)])
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(root))
    assert [hw.device_num_xcd({}, d) for d in (0, 1, 2)] == [3, None, None]
    assert hw.device_num_xcd({"hardware": {"num_xcd": 2}}, 1) == 2
    monkeypatch.setenv(kfd.ROCR_VAR, "2,0")
    assert hw.device_num_xcd({}, 0) is None
    assert hw.device_num_xcd({}, 1) == 3
    assert hw.device_num_xcd({}, 2) is None
    monkeypatch.setenv(kfd.ROCR_VAR, "-1")
    assert hw.device_num_xcd({}, 0) is None
    full = {"hardware": {"n_cu": 7, "lds_bytes": 1, "l2_bytes": 2, "num_xcd": 4}}
    assert hw.device_hardware(full, 0, tmp_path / "absent") == hw.DeviceHardware(
        7, 1, 2, 4
    )
    with pytest.raises(ValueError):
        hw.DeviceHardware(7, 1, 2, 0)


def test_the_xcd_count_comes_from_the_node_of_the_other_values(tmp_path, monkeypatch):
    import shutil

    import gpu_fakes as gf
    from lib import kfd

    for var in (*kfd.HIP_VARS, kfd.ROCR_VAR):
        monkeypatch.delenv(var, raising=False)
    tree = gf.write_sysfs(tmp_path / "tree", xcc=[(3, "SPX"), (1, "CPX"), None])
    other = gf.write_sysfs(tmp_path / "other", xcc=[(6, "SPX")] * 3)
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(other))
    topology = gf.kfd_root(tree) / "topology" / "nodes"
    assert hw.kfd_device_partition(1, topology) == (1, "CPX")
    assert [hw.device_num_xcd({}, d, topology) for d in (0, 1, 2)] == [3, None, None]
    assert hw.kfd_topology() == gf.kfd_root(other) / "topology" / "nodes"
    assert hw.device_num_xcd({}, 0) == 6
    pinned = {"hardware": {"n_cu": 7, "lds_bytes": 1, "l2_bytes": 2}}
    assert hw.device_hardware(pinned, 0, topology).num_xcd == 3
    assert hw.device_hardware(pinned, 0).num_xcd == 6
    loose = tmp_path / "nodes"
    shutil.copytree(topology, loose)
    assert hw.kfd_device_partition(0, loose) == (3, None)
    assert hw.device_num_xcd({}, 0, loose) is None


# ── roofline ─────────────────────────────────────────────────────────────────


def _with_rates(topology, clock=1000, bank=(1024, 500, 1)):
    node = topology / "1"
    props = (node / "properties").read_text()
    if clock is not None:
        props += f"max_engine_clk_fcompute {clock}\nmax_engine_clk_ccompute 3000\n"
    (node / "properties").write_text(props)
    if bank is not None:
        width, mem_clk, heap = bank
        b = node / "mem_banks" / "0"
        b.mkdir(parents=True)
        (b / "properties").write_text(
            f"heap_type {heap}\nsize_in_bytes 1024\nflags 0\nwidth {width}\n"
            f"mem_clk_max {mem_clk}\n"
        )
    return topology


def _cfg(arch="gfx950", a_type="bf16_r", compute_type="c_f32_r"):
    hl = {"a_type": a_type, "b_type": a_type, "c_type": "bf16_r"}
    return {"arch": arch, "hipblaslt": dict(hl, compute_type=compute_type)}


@pytest.mark.parametrize(
    "a_type, compute, expect",
    [
        ("bf16_r", "c_f32_r", "bfloat16"),
        ("f16_r", "c_f32_r", "half"),
        ("f32_r", "c_xf32_r", "xfloat32"),
        ("f32_r", "c_f32_r", "float"),
        ("f8_fnuz_r", "c_f32_r", "float8"),
        ("f8_r", "c_f32_r", "float8"),
        ("f4_r", "c_f32_r", "float4"),
        ("f6_r", "c_f32_r", "float6"),
        ("bf6_r", "c_f32_r", "bfloat6"),
        ("q3_r", "c_f32_r", None),
    ],
)
def test_the_matrix_instruction_type_of_a_config(a_type, compute, expect):
    hl = {"a_type": a_type, "compute_type": compute}
    assert hw.config_mi_dtype(hl) == expect


def test_the_peak_comes_from_the_best_instruction_of_the_type():
    table = hw.arch_constants("gfx950").mi_table
    best = max(
        2 * m * n * k / c for (m, n, k, dt), c in table.items() if dt == "bfloat16"
    )
    assert hw.peak_flops_per_cu_cycle("gfx950", "bfloat16") == best * 4.0
    assert hw.peak_flops_per_cu_cycle("gfx950", "int8") is None
    assert hw.peak_flops_per_cu_cycle("gfx1250", "bfloat16") is not None


def test_roofline_rates_of_a_device(topology):
    _with_rates(topology)
    per_cu = hw.peak_flops_per_cu_cycle("gfx950", "bfloat16")
    rates = hw.roofline_rates(_cfg(), 0, topology)
    assert rates.peak_tflops == pytest.approx(per_cu * 10 * 1000 / 1e6)
    assert (
        rates.peak_note
        == f"bfloat16 at {per_cu:g} FLOP per CU and cycle, 10 CUs, 1000 MHz"
    )
    assert rates.bandwidth_gbs == pytest.approx(1024 / 8 * 500 * 4 / 1e3)
    assert rates.bandwidth_note == "1024-bit bus at 500 MHz, 4 transfers per clock"
    xf32 = hw.roofline_rates(_cfg(a_type="f32_r", compute_type="c_xf32_r"), 0, topology)
    assert (
        xf32.peak_tflops != rates.peak_tflops
        and xf32.bandwidth_gbs == rates.bandwidth_gbs
    )
    seven = hw.roofline_rates(dict(_cfg(), hardware={"n_cu": 7}), 0, topology)
    assert seven.peak_tflops == pytest.approx(per_cu * 7 * 1000 / 1e6)


@pytest.mark.parametrize("a_type, mi_dtype", [("f4_r", "float4"), ("f6_r", "float6")])
def test_four_and_six_bit_types_take_the_peak_of_their_instructions(
    topology, a_type, mi_dtype
):
    _with_rates(topology)
    arch = "gfx1250"
    per_cu = hw.peak_flops_per_cu_cycle(arch, mi_dtype)
    assert per_cu is not None
    rates = hw.roofline_rates(_cfg(arch=arch, a_type=a_type), 0, topology)
    assert rates.peak_tflops == pytest.approx(per_cu * 10 * 1000 / 1e6)
    assert rates.peak_note.startswith(f"{mi_dtype} at {per_cu:g} FLOP per CU")


def test_the_peak_of_bf6_is_unknown_without_its_instructions(topology):
    _with_rates(topology)
    rates = hw.roofline_rates(_cfg(arch="gfx1250", a_type="bf6_r"), 0, topology)
    assert rates.peak_tflops is None
    assert rates.peak_note == "the MI cycle table of gfx1250 has no bfloat6"


def test_rates_a_family_or_device_does_not_give_are_unknown(topology, tmp_path):
    _with_rates(topology, clock=None, bank=(1024, 500, 0))
    rates = hw.roofline_rates(_cfg(), 0, topology)
    assert rates.peak_tflops is None and "no engine clock" in rates.peak_note
    assert rates.bandwidth_gbs is None and "memory bus" in rates.bandwidth_note
    gfx1250 = hw.roofline_rates(_cfg(arch="gfx1250-strict"), 0, topology)
    assert gfx1250.bandwidth_gbs is None
    assert gfx1250.bandwidth_note == "lib/hardware.py knows no memory of gfx1250-strict"
    unknown = hw.roofline_rates(_cfg(a_type="q3_r"), 0, topology)
    assert unknown.peak_tflops is None and "known for q3_r" in unknown.peak_note
    fp4 = hw.roofline_rates(_cfg(a_type="f4_r"), 0, topology)
    assert fp4.peak_note == "the MI cycle table of gfx950 has no float4"
    int8 = hw.roofline_rates(_cfg(a_type="i8_r"), 0, topology)
    assert int8.peak_note == "the MI cycle table of gfx950 has no int8"
    absent = hw.roofline_rates(_cfg(), 0, tmp_path / "absent")
    assert absent.peak_tflops is None and absent.bandwidth_gbs is None
