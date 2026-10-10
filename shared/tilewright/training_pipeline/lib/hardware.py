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
  probed value) and otherwise reads the KFD sysfs topology (`kfd_topology`,
  under TILEWRIGHT_SYSFS_ROOT when it is set, as lib/kfd.py reads it).
  Reading sysfs does not open the device.

  It also gives `num_xcd`, the device's XCD count, which no feature reads:
  hipBLASLt's dynamic-queue Stream-K launches keep one queue per XCD
  (lib/hipblaslt_launch.py). It is the config's `hardware.num_xcd`, else
  the `num_xcc` of the KFD node the other values come from when the node's
  PCI device reports that it runs as one partition, else unknown
  (`device_num_xcd`).

* `RooflineRates`: the device's peak dense matrix rate for a data type and
  its memory bandwidth, the rates no GEMM can beat (lib/bench_yaml.py
  `roofline_us`). The peak is the family's best matrix instruction for the
  type in the MI cycle table, times `parallel_mi_cu`, the CU count and the
  engine clock of the KFD node (`max_engine_clk_fcompute`); the bandwidth
  is the bus width times the clock of the node's frame-buffer memory bank
  times the transfers per clock of the family's memory
  (MEMORY_TRANSFERS_PER_CLOCK). A rate a family or device does not give is
  None (`roofline_rates`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import kfd
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

# Data transfers per memory clock of each family's device memory, with the
# clock as the KFD topology reports it (`mem_clk_max`).
MEMORY_TRANSFERS_PER_CLOCK: Dict[str, int] = {"gfx950": 4}
# KFD memory-bank heap types of the frame buffer (public and private).
_FRAME_BUFFER_HEAPS = (1, 2)


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
    num_xcd: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("n_cu", "lds_bytes", "l2_bytes", "num_xcd"):
            v = getattr(self, name)
            if v is None and name == "num_xcd":
                continue
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"DeviceHardware.{name} must be a positive int: {v!r}")


HARDWARE_KEYS = ("n_cu", "lds_bytes", "l2_bytes")
# Keys of the config's `hardware:` block that are not feature inputs.
DEVICE_ONLY_KEYS = ("pci_chip_id", "num_xcd")
# Where a KFD topology's node directories sit below the sysfs root.
_TOPOLOGY_PARTS = ("class", "kfd", "kfd", "topology", "nodes")


def kfd_topology() -> Path:
    """The KFD topology nodes of this host: /sys/class/kfd/kfd/topology/nodes,
    under TILEWRIGHT_SYSFS_ROOT when it is set (lib/kfd.py)."""
    return kfd.kfd_root() / "topology" / "nodes"


def _topology(topology: Optional[Path]) -> Path:
    return kfd_topology() if topology is None else Path(topology)


def _sysfs_of(topology: Path) -> Optional[Path]:
    """The sysfs root whose KFD topology nodes `topology` is; None for a
    directory elsewhere."""
    if Path(topology).parts[-len(_TOPOLOGY_PARTS) :] != _TOPOLOGY_PARTS:
        return None
    return Path(topology).parents[len(_TOPOLOGY_PARTS) - 1]


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


def kfd_gpu_node(device_index: int = 0, topology: Optional[Path] = None) -> Path:
    """KFD topology node of GPU `device_index`: GPU nodes are the nodes with
    SIMDs, in node order, filtered by integer ROCR_VISIBLE_DEVICES then
    HIP_VISIBLE_DEVICES the way HIP numbers devices."""
    topology = _topology(topology)
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
    return nodes[int(device_index)]


# What a KFD node gives each of HARDWARE_KEYS from.
_KFD_SOURCES = {
    "n_cu": "simd_count / simd_per_cu",
    "lds_bytes": "lds_size_in_kb",
    "l2_bytes": "a level-2 cache",
}


def _kfd_node_values(
    device_index: int = 0, topology: Optional[Path] = None
) -> Tuple[Path, Dict[str, int]]:
    """(`device_index`'s KFD node, its values of HARDWARE_KEYS, 0 for one it
    does not give)."""
    node = kfd_gpu_node(device_index, topology)
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
    return node, {
        "n_cu": props.get("simd_count", 0) // simd_per_cu,
        "lds_bytes": props.get("lds_size_in_kb", 0) * 1024,
        "l2_bytes": l2_kib * 1024,
    }


def kfd_device_hardware(
    device_index: int = 0, topology: Optional[Path] = None
) -> DeviceHardware:
    """Read `device_index`'s hardware from the KFD sysfs topology
    (`kfd_gpu_node`)."""
    return DeviceHardware(**_kfd_node_values(device_index, topology)[1])


def device_hardware(
    cfg: Optional[Mapping[str, Any]],
    device_index: int = 0,
    topology: Optional[Path] = None,
) -> DeviceHardware:
    """Device values for feature computation: the config's `hardware:` block
    when it sets all of n_cu / lds_bytes / l2_bytes, otherwise the KFD
    topology of `device_index` with any configured key taking precedence;
    ValueError naming the key when the node gives no value the block does
    not set. The block's `pci_chip_id` belongs to `device_pci_chip_id`."""
    block = dict((cfg or {}).get("hardware") or {})
    allowed = HARDWARE_KEYS + DEVICE_ONLY_KEYS
    unknown = sorted(set(block) - set(allowed))
    if unknown:
        raise ValueError(f"unknown hardware keys {unknown}; allowed: {list(allowed)}")
    values = {k: int(block[k]) for k in HARDWARE_KEYS if block.get(k) is not None}
    if len(values) < len(HARDWARE_KEYS):
        node, probed = _kfd_node_values(device_index, topology)
        for k in HARDWARE_KEYS:
            if k not in values and probed[k] <= 0:
                raise ValueError(
                    f"the KFD node {node} gives no {_KFD_SOURCES[k]}: set "
                    f"hardware.{k} in the run config"
                )
            values.setdefault(k, probed[k])
    num_xcd = device_num_xcd(cfg, device_index, topology)
    return DeviceHardware(**values, num_xcd=num_xcd)


def kfd_device_partition(
    device_index: int = 0, topology: Optional[Path] = None
) -> Tuple[Optional[int], Optional[str]]:
    """(the XCCs of `device_index`'s KFD node (`kfd_gpu_node`), the compute
    partition mode, such as SPX or CPX, of its PCI device in the sysfs tree
    of `topology`); None for a value the tree does not give."""
    topology = _topology(topology)
    node = kfd_gpu_node(device_index, topology)
    props = _read_properties(node / "properties")
    root = _sysfs_of(topology)
    partition = None if root is None else kfd.compute_partition(props, root)
    return (props.get("num_xcc") or None), partition


def device_num_xcd(
    cfg: Optional[Mapping[str, Any]],
    device_index: int = 0,
    topology: Optional[Path] = None,
) -> Optional[int]:
    """The device's XCD count: the config's `hardware.num_xcd`, else the
    XCCs of its KFD node, the node `device_hardware` reads, when its PCI
    device reports that it runs as one partition (`kfd_device_partition`),
    else None."""
    value = ((cfg or {}).get("hardware") or {}).get("num_xcd")
    if value is not None:
        return int(value)
    try:
        num_xcc, partition = kfd_device_partition(device_index, topology)
    except (RuntimeError, ValueError):
        return None
    return num_xcc if partition == kfd.FULL_DEVICE_PARTITION else None


def device_n_cu(
    cfg: Optional[Mapping[str, Any]],
    device_index: int = 0,
    topology: Optional[Path] = None,
) -> int:
    """The CU count `device_hardware` gives the device, read from the KFD
    topology only when the config's `hardware.n_cu` is unset."""
    value = ((cfg or {}).get("hardware") or {}).get("n_cu")
    if value is not None:
        return int(value)
    return kfd_device_hardware(device_index, topology).n_cu


def device_pci_chip_id(
    cfg: Optional[Mapping[str, Any]],
    device_index: int = 0,
    topology: Optional[Path] = None,
) -> Optional[int]:
    """PCI chip id of the device (hipDeviceProp_t::deviceId): the config's
    `hardware.pci_chip_id`, else the KFD topology `device_id`; None when
    neither gives one."""
    value = ((cfg or {}).get("hardware") or {}).get("pci_chip_id")
    if value is not None:
        return int(value)
    try:
        node = kfd_gpu_node(device_index, topology)
    except RuntimeError:
        return None
    return _read_properties(node / "properties").get("device_id") or None


# ── roofline ─────────────────────────────────────────────────────────────────


def config_mi_dtype(hl: Mapping[str, Any]) -> Optional[str]:
    """The matrix-instruction data type of the config's `hipblaslt`
    section, as the MI cycle tables name it: the A type, xfloat32 for xf32
    compute (lib/features.py `problem_kwargs_from_row`); None for a type
    the catalog does not know."""
    from . import features

    try:
        a = features.dtype_from_hipblaslt(hl.get("a_type"))
    except ValueError:
        return None
    if "xf32" in str(hl.get("compute_type") or "").lower():
        return "xfloat32"
    return a.replace("_fnuz", "")


def peak_flops_per_cu_cycle(arch: str, mi_dtype: str) -> Optional[float]:
    """Dense matrix FLOPs one CU completes per cycle for `mi_dtype` on
    `arch`: `parallel_mi_cu` times the most 2 m n k per cycle of the
    family's MI cycle table entries of that type; None without an entry."""
    c = arch_constants(arch)
    best = max(
        (
            2.0 * m * n * k / cycles
            for (m, n, k, dt), cycles in c.mi_table.items()
            if dt == mi_dtype and cycles > 0
        ),
        default=None,
    )
    return None if best is None else best * c.parallel_mi_cu


def kfd_engine_clock_mhz(
    device_index: int = 0, topology: Optional[Path] = None
) -> Optional[int]:
    """The highest engine clock of `device_index`'s KFD node
    (`max_engine_clk_fcompute`, MHz), or None."""
    node = kfd_gpu_node(device_index, topology)
    return _read_properties(node / "properties").get("max_engine_clk_fcompute") or None


def kfd_memory_bus(
    device_index: int = 0, topology: Optional[Path] = None
) -> Optional[Tuple[int, int]]:
    """(bus width in bits, highest clock in MHz) of the frame-buffer memory
    bank of `device_index`'s KFD node, or None."""
    node = kfd_gpu_node(device_index, topology)
    for bank in _numbered_dirs(node / "mem_banks"):
        props = _read_properties(bank / "properties")
        width, clock = props.get("width", 0), props.get("mem_clk_max", 0)
        if props.get("heap_type") in _FRAME_BUFFER_HEAPS and width > 0 and clock > 0:
            return width, clock
    return None


@dataclass(frozen=True)
class RooflineRates:
    """The peak matrix rate of a data type (TFLOP/s) and the memory
    bandwidth (GB/s) of a device, None when unknown, each with a note
    saying where it comes from or why it is unknown."""

    peak_tflops: Optional[float]
    peak_note: str
    bandwidth_gbs: Optional[float]
    bandwidth_note: str


def _device_peak(
    cfg: Mapping[str, Any], device_index: int, topology: Optional[Path]
) -> Tuple[Optional[float], str]:
    arch = str(cfg.get("arch") or "")
    hl = cfg.get("hipblaslt") or {}
    mi_dtype = config_mi_dtype(hl)
    if mi_dtype is None:
        return None, f"no matrix-instruction type is known for {hl.get('a_type')}"
    per_cu = peak_flops_per_cu_cycle(arch, mi_dtype) if arch_family(arch) else None
    if per_cu is None:
        return None, f"the MI cycle table of {arch} has no {mi_dtype}"
    try:
        n_cu = device_n_cu(cfg, device_index, topology)
        clock = kfd_engine_clock_mhz(device_index, topology)
    except (OSError, RuntimeError, ValueError) as e:
        return None, str(e)
    if not clock:
        return None, "the KFD node gives no engine clock"
    note = f"{mi_dtype} at {per_cu:g} FLOP per CU and cycle, {n_cu} CUs, {clock} MHz"
    return per_cu * n_cu * clock / 1e6, note


def _device_bandwidth(
    cfg: Mapping[str, Any], device_index: int, topology: Optional[Path]
) -> Tuple[Optional[float], str]:
    arch = str(cfg.get("arch") or "")
    transfers = MEMORY_TRANSFERS_PER_CLOCK.get(arch_family(arch) or "")
    if transfers is None:
        return None, f"lib/hardware.py knows no memory of {arch}"
    try:
        bus = kfd_memory_bus(device_index, topology)
    except (OSError, RuntimeError, ValueError) as e:
        return None, str(e)
    if bus is None:
        return None, "the KFD node gives no frame-buffer memory bus"
    width, clock = bus
    note = f"{width}-bit bus at {clock} MHz, {transfers} transfers per clock"
    return width / 8 * clock * transfers / 1e3, note


def roofline_rates(
    cfg: Mapping[str, Any],
    device_index: int = 0,
    topology: Optional[Path] = None,
) -> RooflineRates:
    """The roofline rates of bench device `device_index` for the config's
    data types (see the module docstring)."""
    peak, peak_note = _device_peak(cfg, device_index, topology)
    bandwidth, bandwidth_note = _device_bandwidth(cfg, device_index, topology)
    return RooflineRates(peak, peak_note, bandwidth, bandwidth_note)
