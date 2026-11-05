# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device values as the bench device's KFD node and HIP report them.

The pipeline computes features and the LDS-fit rule with the device values
of lib/hardware.py `device_hardware`: the config's `hardware:` block, else
the KFD topology node of the bench device. hipBLASLt serves a model with the
values its TensileLite integration reads through HIP:
hipDeviceAttributePhysicalMultiProcessorCount for the CU count,
hipDeviceProp_t::sharedMemPerBlock for LDS and l2CacheSize for L2. Values
that differ train a model on inputs it is never served with and move the
LDS-fit rule between training and serving, silently.

`check` reads both on the first device of a run, the one its device values
come from: the KFD node (`kfd_values`: `simd_count / simd_per_cu`,
`lds_size_in_kb`, the first level-2 cache, `wave_front_size`) and HIP
(`hip_values`: tools/device_props.cpp, run on that GPU alone under its
lease). It refuses a device whose HIP values differ from the values the
run's stages use (`used`: lib/hardware.py `device_hardware`, the config's
`hardware:` block over the KFD node) or that the stages cannot read, or
whose KFD node and HIP disagree on the wave size or the architecture (the
topology and HIP would then name different devices), and warns where only
the KFD node differs from a pinned value HIP agrees with. Both sets go into
the run's `platform.json`.

The probe is built from its source with the correctness gate's compiler
(`lib.correctness_gate.default_compiler`, no device code) into
`<cache>/device_props/<digest>/` on first use; TILEWRIGHT_DEVICE_PROBE names
a prebuilt one instead, and set but empty skips the HIP read. With a fake
topology (TILEWRIGHT_SYSFS_ROOT) and no probe named, HIP is not read: it
would describe a real device the fake tree does not.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import kfd

PIPELINE_DIR = Path(__file__).resolve().parent.parent
PROBE_SOURCE = PIPELINE_DIR / "tools" / "device_props.cpp"
PROBE_VAR = "TILEWRIGHT_DEVICE_PROBE"
PROBE_TIMEOUT_S = 300.0
# Seconds a killed probe gets to end: one stuck in the GPU driver does not.
PROBE_REAP_S = 60.0
KEYS = ("n_cu", "lds_bytes", "l2_bytes", "wave_size")
LABELS = {
    "n_cu": "CU count",
    "lds_bytes": "LDS bytes",
    "l2_bytes": "L2 bytes",
    "wave_size": "wave size",
}
# Values a run config can pin (lib/hardware.py HARDWARE_KEYS).
PINNABLE = ("n_cu", "lds_bytes", "l2_bytes")


class ProbeError(RuntimeError):
    """HIP's device values cannot be read."""


class NoAnswer(ProbeError):
    """The probe did not end: the device does not answer."""


def _properties(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return out


def kfd_values(gpu: kfd.Gpu) -> Dict[str, Optional[int]]:
    """The KFD node's values of `gpu`, None for one it does not give, and
    its compiler target (`arch`)."""
    node = kfd.kfd_root() / "topology" / "nodes" / str(gpu.node)
    props = _properties(node / "properties")
    simds, per_cu = props.get("simd_count"), props.get("simd_per_cu")
    l2 = None
    caches = node / "caches"
    try:
        entries = sorted(
            (p for p in caches.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)
        )
    except OSError:
        entries = []
    for c in entries:
        cp = _properties(c / "properties")
        if cp.get("level") == 2 and cp.get("size", 0) > 0:
            l2 = cp["size"] * 1024
            break
    lds = props.get("lds_size_in_kb")
    target, _ = kfd.gpu_isa(gpu)
    return {
        "n_cu": simds // per_cu if simds and per_cu else None,
        "lds_bytes": lds * 1024 if lds else None,
        "l2_bytes": l2,
        "wave_size": props.get("wave_front_size") or None,
        "arch": target,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_command(compiler: str, source: Path, out: Path) -> List[str]:
    """The command that builds the probe: host code calling the HIP runtime
    API only."""
    from .correctness_gate import rocm_path

    rocm = rocm_path()
    rpath = ":".join(str(p) for p in (rocm / "lib", rocm / "lib" / "llvm" / "lib"))
    return [
        compiler,
        "-std=c++17",
        "-O2",
        "-D__HIP_PLATFORM_AMD__",
        f"-I{rocm / 'include'}",
        str(source),
        f"-L{rocm / 'lib'}",
        "-lamdhip64",
        f"-Wl,-rpath,{rpath}",
        "-o",
        str(out),
    ]


def probe_executable(
    cache_dir: Path, compiler: Optional[str] = None, environ: Optional[Mapping] = None
) -> Optional[Path]:
    """The probe: the executable TILEWRIGHT_DEVICE_PROBE names, None when it
    is set but empty or when the KFD topology is a fake tree and it is
    unset, else the one built from PROBE_SOURCE in
    `<cache_dir>/device_props/<digest>/`, built now when missing.
    ProbeError when it cannot be built."""
    from .correctness_gate import default_compiler, rocm_path

    env = os.environ if environ is None else environ
    named = env.get(PROBE_VAR)
    if named is not None:
        if not named.strip():
            return None
        path = Path(named)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ProbeError(f"{PROBE_VAR}={named} is not an executable")
        return path
    if env.get(kfd.SYSFS_ROOT_VAR):
        return None
    compiler = compiler or default_compiler()
    rocm = rocm_path()
    version = rocm / ".info" / "version"
    body = (
        f"{_sha256(PROBE_SOURCE)}\n{compiler}\n"
        f"{version.read_text().strip() if version.is_file() else rocm}\n"
    )
    digest = hashlib.sha256(body.encode()).hexdigest()[:16]
    out = Path(cache_dir) / "device_props" / digest / "device_props"
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".device_props.{os.getpid()}")
    cmd = build_command(compiler, PROBE_SOURCE, tmp)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        raise ProbeError(f"cannot run the compiler {compiler}: {e}") from None
    if proc.returncode != 0 or not tmp.is_file():
        raise ProbeError(
            f"building {PROBE_SOURCE.name} failed (rc={proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()[-600:]}"
        )
    os.replace(tmp, out)
    return out


def _run_probe(probe: Path, gpu: int, env: Mapping[str, str]) -> Tuple[int, str, str]:
    """(exit status, stdout, stderr) of the probe on GPU `gpu`. ProbeError
    when it does not run, NoAnswer when it does not end within
    PROBE_TIMEOUT_S (killed, it gets PROBE_REAP_S more to end)."""
    try:
        proc = subprocess.Popen(
            [str(probe)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            env=dict(env),
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise ProbeError(f"the probe {probe} did not run: {e}") from None
    try:
        out, err = proc.communicate(timeout=PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.communicate(timeout=PROBE_REAP_S)
            after = "it was killed"
        except subprocess.TimeoutExpired:
            after = "it does not end after SIGKILL either"
        raise NoAnswer(
            f"GPU {gpu} does not answer: the probe {probe} did not end in "
            f"{PROBE_TIMEOUT_S:g} s, and {after}"
        ) from None
    return proc.returncode, out, err


def hip_values(probe: Path, gpu: int, environ: Mapping[str, str]) -> Dict[str, Any]:
    """HIP's values of physical GPU `gpu`: the probe run with
    ROCR_VISIBLE_DEVICES naming it alone. ProbeError when it fails."""
    env = {k: v for k, v in environ.items() if k not in kfd.HIP_VARS}
    env[kfd.ROCR_VAR] = str(int(gpu))
    rc, stdout, stderr = _run_probe(probe, gpu, env)
    doc = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                doc = json.loads(line)
            except ValueError:
                doc = None
    if rc != 0 or not isinstance(doc, dict):
        raise ProbeError(
            f"the probe {probe} failed on GPU {gpu} (rc={rc}): "
            f"{(stderr or stdout).strip()[-400:]}"
        )

    def num(key: str) -> Optional[int]:
        v = doc.get(key)
        return (
            int(v) if isinstance(v, int) and not isinstance(v, bool) and v > 0 else None
        )

    return {
        "n_cu": num("physical_multi_processor_count") or num("multi_processor_count"),
        "lds_bytes": num("shared_mem_per_block"),
        "l2_bytes": num("l2_cache_size"),
        "wave_size": num("warp_size"),
        "arch": str(doc.get("arch") or "").split(":", 1)[0] or None,
        "multi_processor_count": num("multi_processor_count"),
    }


@dataclass
class Check:
    """What `check` found of one device: the values of each source, the
    values the run's stages use (`used`; `used_error` when they cannot read
    them) and the keys the config pins, the refusals (`errors`), among them
    a device that does not answer (`no_answer`), and notes (`warnings`)."""

    device: int
    gpu: Optional[int] = None
    bdf: Optional[str] = None
    kfd: Dict[str, Any] = field(default_factory=dict)
    hip: Optional[Dict[str, Any]] = None
    used: Dict[str, Any] = field(default_factory=dict)
    used_error: Optional[str] = None
    pinned: List[str] = field(default_factory=list)
    no_answer: bool = False
    skipped: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def record(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "gpu": self.gpu,
            "bdf": self.bdf,
            "kfd": self.kfd,
            "hip": self.hip,
            "used": self.used,
            "used_error": self.used_error,
            "no_answer": self.no_answer,
            "skipped": self.skipped,
            "problems": self.errors,
            "notes": self.warnings,
        }


def used_values(
    cfg: Mapping[str, Any], device: int, kfd_node: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    """(the values the run's stages use on device `device`, lib/hardware.py
    `device_hardware`'s, with the wave size of the KFD node `kfd_node`; why
    the stages cannot read them, or None)."""
    from .hardware import device_hardware

    wave = {"wave_size": kfd_node.get("wave_size")}
    try:
        hw = device_hardware(cfg, device)
    except (OSError, RuntimeError, ValueError) as e:
        return {**{k: None for k in PINNABLE}, **wave}, str(e)
    return {**{k: getattr(hw, k) for k in PINNABLE}, **wave}, None


def compare(c: Check) -> None:
    """Fill `c.errors` and `c.warnings` from its three sets of values."""
    if c.hip is None:
        return
    where = f"device {c.device} (GPU {c.gpu})"
    k_arch, h_arch = c.kfd.get("arch"), c.hip.get("arch")
    if k_arch and h_arch and k_arch != h_arch.split("-", 1)[0]:
        c.errors.append(
            f"{where}: the KFD node is a {k_arch} and HIP's device 0 a {h_arch}: "
            f"they name different devices"
        )
    served = [k for k in PINNABLE if c.hip.get(k) is not None]
    if c.used_error is not None and served:
        values = ", ".join(f"{LABELS[k]} {c.hip[k]}" for k in served)
        c.errors.append(
            f"{where}: the run's stages cannot read its device values "
            f"({c.used_error}), while hipBLASLt serves models with HIP's "
            f"({values}); set them in the `hardware:` block of the run config "
            f"kept outside the repository"
        )
    for key in KEYS:
        label = LABELS[key]
        k, h, u = c.kfd.get(key), c.hip.get(key), c.used.get(key)
        if key not in PINNABLE:
            if k is not None and h is not None and k != h:
                c.errors.append(
                    f"{where}: the KFD node and HIP disagree on the {label}: KFD "
                    f"{k}, HIP {h}"
                )
            continue
        if u is not None and h is not None and u != h:
            pinned = key in c.pinned
            c.errors.append(
                f"{where}: the run uses {u} as the {label} "
                f"({'hardware.' + key + ' pins it' if pinned else 'from the KFD node'}), "
                f"hipBLASLt serves models with HIP's {h}: features and the LDS fit "
                f"would differ between training and serving; once you know which "
                f"is right, set hardware.{key} in the run config kept outside the "
                f"repository"
            )
        elif k is not None and h is not None and k != h:
            c.warnings.append(
                f"{where}: the KFD node gives {k} as the {label}, HIP {h}; the run "
                f"uses HIP's, which hardware.{key} pins"
            )


def check(
    cfg: Mapping[str, Any],
    probe_cache: Path,
    lease_wait_s: float = 0.0,
    run_dir: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Check:
    """`compare` on the first device of runtime.devices, whose values the
    run reads (lib/hardware.py `device_hardware`); the HIP read runs under
    that GPU's lease. A device the environment hides or the KFD topology
    does not list is skipped, and so is HIP without a probe
    (`probe_executable`)."""
    from . import bench_env as benv
    from . import gpu_lease
    from .hardware import HARDWARE_KEYS

    env = {**(os.environ if environ is None else environ), **benv.runtime_env(cfg)}
    device = int(cfg["runtime"]["devices"][0])
    c = Check(device=device)
    try:
        gpu = kfd.physical_gpu(device, env)
    except kfd.GpuMapError as e:
        c.skipped = str(e)
        return c
    by_index = {g.index: g for g in kfd.gpus()}
    if gpu is None or gpu not in by_index:
        c.skipped = (
            "the environment hides the device or the KFD topology lists no such GPU"
        )
        return c
    g = by_index[gpu]
    c.gpu, c.bdf = gpu, g.bdf
    c.kfd = kfd_values(g)
    pins = dict((cfg.get("hardware") or {}))
    c.pinned = [k for k in PINNABLE if k in HARDWARE_KEYS and pins.get(k) is not None]
    c.used, c.used_error = used_values(cfg, device, c.kfd)
    try:
        probe = probe_executable(probe_cache, environ=env)
    except ProbeError as e:
        c.errors.append(
            f"device {device} (GPU {gpu}): cannot read HIP's device values: {e}; "
            f"name a probe built from tools/{PROBE_SOURCE.name} in {PROBE_VAR}, "
            f"or set {PROBE_VAR} empty to skip this check"
        )
        return c
    if probe is None:
        c.skipped = f"no probe ({PROBE_VAR} empty, or a fake KFD topology)"
        return c
    try:
        with gpu_lease.hold_gpus(
            [gpu], "device values", run_dir, wait_s=lease_wait_s, environ=env
        ):
            c.hip = hip_values(probe, gpu, benv.bench_env_with(benv.runtime_env(cfg)))
    except (ProbeError, gpu_lease.LeaseError) as e:
        c.no_answer = isinstance(e, NoAnswer)
        c.errors.append(
            f"device {device} (GPU {gpu}): cannot read HIP's device values: {e}"
        )
        return c
    compare(c)
    return c


def write_platform(path: Path, c: Check) -> Dict[str, Any]:
    """The platform record `path`: when and on which host the run's device
    values were compared, and `c`'s record."""
    from .fingerprint import host_settings
    from .fs import write_json

    doc = {
        "recorded": kfd.utc_iso(),
        "host": host_settings(),
        "device_values": c.record(),
    }
    write_json(Path(path), doc)
    return doc
