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
