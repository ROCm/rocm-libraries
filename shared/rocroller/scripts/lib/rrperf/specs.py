# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Get host/gpu specs."""

import json
import os
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path as path
from textwrap import dedent

import yaml


@dataclass
class MachineSpecs(yaml.YAMLObject):
    yaml_tag = "!rrperfMachineSpecs"

    hostname: str
    cpu: str
    kernel: str
    ram: str
    distro: str
    rocmversion: str
    vbios: str
    gpuid: str
    deviceinfo: str
    vram: str
    perflevel: str
    mclk: str
    sclk: str

    def __init__(
        self,
        hostname="",
        cpu="",
        kernel="",
        ram="",
        distro="",
        rocmversion="",
        vbios="",
        gpuid="",
        deviceinfo="",
        vram="",
        perflevel="",
        mclk="",
        sclk="",
    ):
        self.hostname = hostname
        self.cpu = cpu
        self.kernel = kernel
        self.ram = ram
        self.distro = distro
        self.rocmversion = rocmversion
        self.vbios = vbios
        self.gpuid = gpuid
        self.deviceinfo = deviceinfo
        self.vram = vram
        self.perflevel = perflevel
        self.mclk = mclk
        self.sclk = sclk

    def __str__(self):
        return yaml.dump(self)

    def __hash__(self):
        return hash(str(self))

    def __lt__(self, other):
        return str(self) < str(other)

    @classmethod
    def from_yaml(cls, loader, node):
        values = loader.construct_mapping(node, deep=True)
        return cls(
            values.get("hostname", ""),
            values.get("cpu", ""),
            values.get("kernel", ""),
            values.get("ram", ""),
            values.get("distro", ""),
            values.get("rocmversion", ""),
            values.get("vbios", ""),
            values.get("gpuid", ""),
            values.get("deviceinfo", ""),
            values.get("vram", ""),
            values.get("perflevel", ""),
            values.get("mclk", ""),
            values.get("sclk", ""),
        )

    def pretty_string(self):
        return dedent(f"""\
        Host info:
            hostname:       {self.hostname}
            cpu info:       {self.cpu}
            ram:            {self.ram}
            distro:         {self.distro}
            kernel version: {self.kernel}
            rocm version:   {self.rocmversion}
        Device info:
            device:            {self.deviceinfo}
            vbios version:     {self.vbios}
            vram:              {self.vram}
            performance level: {self.perflevel}
            system clock:      {self.sclk}
            memory clock:      {self.mclk}
        """)


def search(pattern, string):
    m = re.search(pattern, string, re.MULTILINE)
    if m is not None:
        return m.group(1)
    return None


def _run_amdsmi_json(cmd: list):
    """Run an amd-smi command that emits JSON and return the parsed object."""
    try:
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode != 0:
            return None
        return json.loads(completed.stdout.decode("utf-8", errors="replace"))
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return None


def _gpu_entry(payload, devicenum: int):
    """Return the gpu_data entry for devicenum, if present."""
    if not payload:
        return None
    gpu_data = payload.get("gpu_data")
    if not gpu_data:
        return None
    for entry in gpu_data:
        if entry.get("gpu") == devicenum:
            return entry
    return gpu_data[0]


def _clock_mhz(clock_entry):
    """Extract a clock string from an amd-smi clock block."""
    if not isinstance(clock_entry, dict):
        return None
    clk = clock_entry.get("clk")
    if isinstance(clk, dict):
        value = clk.get("value")
        if value is not None:
            return f"{value}Mhz"
    elif isinstance(clk, (int, float)):
        return f"{clk}Mhz"
    elif isinstance(clk, str) and clk not in ("", "N/A"):
        return clk
    return None


def get_amdsmi_specs(devicenum: int = 0, amd_smi_path: str = "amd-smi") -> dict:
    """
    Collect per-device GPU specs using amd-smi structured JSON output.

    Any field that cannot be resolved is returned as None.
    """
    g = str(devicenum)
    static = _run_amdsmi_json(
        [amd_smi_path, "static", "-g", g, "--asic", "--vbios", "--json"]
    )
    metric = _run_amdsmi_json(
        [
            amd_smi_path,
            "metric",
            "-g",
            g,
            "--mem-usage",
            "--clock",
            "--perf-level",
            "--json",
        ]
    )

    result = {
        "vbios_version": None,
        "gpuid": None,
        "market_name": None,
        "vram": None,
        "performance_level": None,
        "memory_clk": None,
        "system_clk": None,
    }

    if static:
        d = _gpu_entry(static, devicenum)
        if d:
            result["gpuid"] = d.get("asic", {}).get("device_id")
            result["market_name"] = d.get("asic", {}).get("market_name")
            vbios = d.get("ifwi") or d.get("vbios") or {}
            result["vbios_version"] = vbios.get("part_number")

    if metric:
        d = _gpu_entry(metric, devicenum)
        if d:
            total = d.get("mem_usage", {}).get("total_vram", {}).get("value")
            if total is not None:
                result["vram"] = int(total) * 1024 * 1024

            perf = d.get("perf_level")
            if isinstance(perf, str):
                result["performance_level"] = perf.split("_")[-1].lower()

            clock = d.get("clock", {})
            result["system_clk"] = _clock_mhz(clock.get("gfx_0"))
            result["memory_clk"] = _clock_mhz(clock.get("mem_0"))

    return result


def load_machine_specs(path):
    if path.exists():
        contents = path.read_text()
        if contents.startswith(MachineSpecs.yaml_tag):
            return yaml.load(contents, Loader=yaml.Loader)
    return MachineSpecs()


def _gpu_spec_text(value, missing="no amd-smi"):
    return missing if value is None else value


def get_machine_specs(devicenum, amd_smi_path="amd-smi"):
    cpuinfo = path("/proc/cpuinfo").read_text()
    meminfo = path("/proc/meminfo").read_text()
    version = path("/proc/version").read_text()
    os_release = path("/etc/os-release").read_text()
    if os.path.isfile("/opt/rocm/.info/version-utils"):
        rocm_info = path("/opt/rocm/.info/version-utils").read_text()
    elif os.path.isfile("/opt/rocm/.info/version"):
        rocm_info = path("/opt/rocm/.info/version").read_text()
    else:
        rocm_info = "rocm info not available"

    amd_smi_available = shutil.which(amd_smi_path) is not None
    amdsmi = get_amdsmi_specs(devicenum, amd_smi_path) if amd_smi_available else {}

    missing = "no amd-smi"

    # Use the NODE_NAME env var in CI.
    hostname = os.environ.get("NODE_NAME")
    if not hostname:
        hostname = socket.gethostname()
    cpu = search(r"^model name\s*: (.*?)$", cpuinfo)
    kernel = search(r"version (\S*)", version)
    ram = search(r"MemTotal:\s*(\S*)", meminfo)
    distro = search(r'PRETTY_NAME="(.*?)"', os_release)
    rocmversion = rocm_info.strip()
    vbios = _gpu_spec_text(amdsmi.get("vbios_version"), missing) if amd_smi_available else missing
    gpuid = _gpu_spec_text(amdsmi.get("gpuid"), missing) if amd_smi_available else missing
    deviceinfo = _gpu_spec_text(amdsmi.get("market_name"), missing) if amd_smi_available else missing
    vram = amdsmi.get("vram") if amd_smi_available else 0
    perflevel = _gpu_spec_text(amdsmi.get("performance_level"), missing) if amd_smi_available else missing
    mclk = _gpu_spec_text(amdsmi.get("memory_clk"), "") if amd_smi_available else ""
    sclk = _gpu_spec_text(amdsmi.get("system_clk"), "") if amd_smi_available else ""

    if ram is not None:
        ram = "{:.2f} GiB".format(float(ram) / 1024**2)
    if vram is not None:
        vram = "{:.2f} GiB".format(float(vram) / 1024**3)

    return MachineSpecs(
        hostname,
        cpu,
        kernel,
        ram,
        distro,
        rocmversion,
        vbios,
        gpuid,
        deviceinfo,
        vram,
        perflevel,
        mclk,
        sclk,
    )


if __name__ == "__main__":
    print(get_machine_specs(0))
