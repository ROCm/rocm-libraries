# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Hardware inputs of the feature catalog.

Two kinds of values feed the features:

* `ArchConstants`: per-architecture model constants (matrix-instruction
  cycle table, `parallel_mi_cu`, bandwidth-occupancy coefficients). They are
  written into every model file (MLREC_v2) and the engine reads them back, so
  a shipped model always sees the constants it was trained with. Changing a
  value here changes the inputs of every model trained afterwards; it never
  affects models already written.

* `DeviceHardware`: values queried from the device at runtime. They must be
  the numbers hipBLASLt passes to the engine (TensileLite copies them from
  its analytical hardware description):

    n_cu       physical compute-unit count of the device
               (hipDeviceAttributePhysicalMultiProcessorCount; KFD
               topology `simd_count / simd_per_cu`)
    lds_bytes  LDS bytes available to one workgroup
               (hipDeviceProp_t::sharedMemPerBlock; KFD `lds_size_in_kb`)
    l2_bytes   L2 cache bytes (hipDeviceProp_t::l2CacheSize; KFD level-2
               cache `size`)

  `device_hardware()` takes them from the config's `hardware:` block when
  present (keys `n_cu`, `lds_bytes`, `l2_bytes`; any subset overrides the
  probed value) and otherwise reads the KFD sysfs topology. Reading sysfs
  does not open the device.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .arch import canonical_arch

DATATYPE_VALUE: Dict[str, int] = {
    "float": 0,
    "double": 1,
    "complexfloat": 2,
    "complexdouble": 3,
    "half": 4,
    "int8x4": 5,
    "int32": 6,
    "bfloat16": 7,
    "int8": 8,
    "int4": 9,
    "int64": 10,
    "xfloat32": 11,
    "float8_fnuz": 12,
    "bfloat8_fnuz": 13,
    "float8bfloat8_fnuz": 14,
    "bfloat8float8_fnuz": 15,
    "float8": 16,
    "bfloat8": 17,
    "float8bfloat8": 18,
    "bfloat8float8": 19,
    "float6": 20,
    "bfloat6": 21,
    "float4": 22,
}
DATATYPE_NAME: Dict[int, str] = {v: k for k, v in DATATYPE_VALUE.items()}

MI_DEFAULT_CYCLES = 32.0


@dataclass(frozen=True)
class ArchConstants:
    """Model constants of one architecture family. `mi_table` maps
    (mi_m, mi_n, mi_k, dtype_name) to raw instruction cycles; lookups that
    miss use `mi_default`."""

    parallel_mi_cu: float
    bw: Tuple[float, float, float]
    mi_table: Dict[Tuple[int, int, int, str], float] = field(default_factory=dict)
    mi_default: float = MI_DEFAULT_CYCLES


_MI_TABLES: Dict[str, Dict[Tuple[int, int, int, str], float]] = {
    "gfx950": {
        (32, 32, 2, "float"): 64,
        (32, 32, 1, "float"): 64,
        (16, 16, 4, "float"): 32,
        (16, 16, 1, "float"): 32,
        (32, 32, 16, "bfloat16"): 32,
        (16, 16, 32, "bfloat16"): 16,
        (32, 32, 8, "half"): 32,
        (32, 32, 16, "half"): 32,
        (16, 16, 16, "half"): 16,
        (16, 16, 32, "half"): 16,
        (32, 32, 64, "float8"): 32,
        (16, 16, 128, "float8"): 16,
        (32, 32, 64, "bfloat8"): 32,
        (16, 16, 128, "bfloat8"): 16,
        (32, 32, 8, "xfloat32"): 96,
        (16, 16, 32, "xfloat32"): 48,
        (16, 16, 16, "xfloat32"): 48,
    },
    "gfx1250": {
        (16, 16, 4, "float"): 16,
        (16, 16, 4, "double"): 16,
        (16, 16, 4, "complexfloat"): 64,
        (16, 16, 4, "complexdouble"): 64,
        (16, 16, 32, "half"): 8,
        (16, 16, 32, "bfloat16"): 8,
        (16, 16, 64, "float8"): 4,
        (16, 16, 128, "float8"): 8,
        (16, 16, 64, "bfloat8"): 4,
        (16, 16, 128, "bfloat8"): 8,
        (16, 16, 64, "float8bfloat8"): 4,
        (16, 16, 128, "float8bfloat8"): 8,
        (16, 16, 64, "bfloat8float8"): 4,
        (16, 16, 128, "bfloat8float8"): 8,
        (16, 16, 128, "float6"): 8,
        (16, 16, 128, "float4"): 4,
        (32, 16, 128, "float4"): 8,
        (16, 16, 64, "int8"): 8,
        (16, 16, 32, "xfloat32"): 24,
    },
}

_FAMILY_CONSTANTS: Dict[str, Tuple[float, Tuple[float, float, float]]] = {
    "gfx950": (4.0, (-0.000013, 0.007070, 0.027355)),
    "gfx1250": (4.0, (0.0, 0.016, 0.0)),
}

_TARGET_FAMILY = {"gfx950": "gfx950", "gfx1250": "gfx1250"}


def arch_family(arch: str) -> Optional[str]:
    """Constants family of an arch name (any spelling `canonical_arch`
    accepts), or None when unknown."""
    return _TARGET_FAMILY.get(canonical_arch(str(arch or "").strip().lower()))


def arch_constants(arch: str) -> ArchConstants:
    """Model constants for `arch`. Raises ValueError for an architecture
    with no table: guessing would bake wrong constants into a model."""
    family = arch_family(arch)
    if family is None:
        raise ValueError(
            f"no model constants for arch {arch!r}; known families: "
            f"{sorted(_FAMILY_CONSTANTS)} (add one to lib/hardware.py)"
        )
    parallel_mi_cu, bw = _FAMILY_CONSTANTS[family]
    table = {key: float(v) for key, v in _MI_TABLES[family].items()}
    return ArchConstants(parallel_mi_cu=parallel_mi_cu, bw=bw, mi_table=table)


@dataclass(frozen=True)
class DeviceHardware:
    n_cu: int
    lds_bytes: int
    l2_bytes: int

    def __post_init__(self) -> None:
        for name in ("n_cu", "lds_bytes", "l2_bytes"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"DeviceHardware.{name} must be a positive int: {v!r}")


HARDWARE_KEYS = ("n_cu", "lds_bytes", "l2_bytes")
KFD_TOPOLOGY = Path("/sys/class/kfd/kfd/topology/nodes")


def _read_properties(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return out


def _numbered_dirs(path: Path) -> List[Path]:
    if not path.is_dir():
        return []
    return sorted(
        (p for p in path.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)
    )


def _visible_subset(nodes: List[Path], var: str) -> List[Path]:
    """`nodes` filtered by the device list in `var`. A negative index ends
    the list (`-1` hides every device)."""
    spec = os.environ.get(var)
    if spec is None or spec.strip() == "":
        return nodes
    picked: List[Path] = []
    for token in spec.split(","):
        token = token.strip()
        try:
            index = int(token)
        except ValueError:
            raise ValueError(
                f"{var}={spec!r}: only integer device lists can be mapped to "
                f"KFD nodes; set the config `hardware:` block instead"
            ) from None
        if index < 0:
            break
        if index < len(nodes):
            picked.append(nodes[index])
    return picked


def kfd_device_hardware(
    device_index: int = 0, topology: Path = KFD_TOPOLOGY
) -> DeviceHardware:
    """Read `device_index`'s hardware from the KFD sysfs topology. GPU nodes
    are the nodes with SIMDs, in node order, filtered by integer
    ROCR_VISIBLE_DEVICES then HIP_VISIBLE_DEVICES the way HIP numbers
    devices."""
    nodes = []
    for d in _numbered_dirs(Path(topology)):
        props_path = d / "properties"
        if not props_path.is_file():
            continue
        if _read_properties(props_path).get("simd_count", 0) > 0:
            nodes.append(d)
    filters = []
    for var in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        nodes = _visible_subset(nodes, var)
        if os.environ.get(var, "").strip():
            filters.append(f"{var}={os.environ[var]}")
    if not 0 <= int(device_index) < len(nodes):
        raise RuntimeError(
            f"no GPU node {device_index} in {topology} ({len(nodes)} visible"
            + (f" with {', '.join(filters)}" if filters else "")
            + f"); set the config `hardware:` block ({', '.join(HARDWARE_KEYS)})"
        )
    node = nodes[int(device_index)]
    props = _read_properties(node / "properties")
    simd_per_cu = props.get("simd_per_cu", 0)
    if simd_per_cu <= 0:
        raise RuntimeError(f"{node}: simd_per_cu missing")
    l2_kib = 0
    for cache in _numbered_dirs(node / "caches"):
        cp = _read_properties(cache / "properties")
        if cp.get("level") == 2 and cp.get("size", 0) > 0:
            l2_kib = cp["size"]
            break
    return DeviceHardware(
        n_cu=props.get("simd_count", 0) // simd_per_cu,
        lds_bytes=props.get("lds_size_in_kb", 0) * 1024,
        l2_bytes=l2_kib * 1024,
    )


def device_hardware(
    cfg: Optional[Mapping[str, Any]],
    device_index: int = 0,
    topology: Path = KFD_TOPOLOGY,
) -> DeviceHardware:
    """Device values for feature computation: the config's `hardware:` block
    when it sets all of n_cu / lds_bytes / l2_bytes, otherwise the KFD
    topology of `device_index` with any configured key taking precedence."""
    block = dict((cfg or {}).get("hardware") or {})
    unknown = sorted(set(block) - set(HARDWARE_KEYS))
    if unknown:
        raise ValueError(
            f"unknown hardware keys {unknown}; allowed: {list(HARDWARE_KEYS)}"
        )
    values = {k: int(block[k]) for k in HARDWARE_KEYS if block.get(k) is not None}
    if len(values) < len(HARDWARE_KEYS):
        probed = kfd_device_hardware(device_index, topology)
        for k in HARDWARE_KEYS:
            values.setdefault(k, getattr(probed, k))
    return DeviceHardware(**values)
