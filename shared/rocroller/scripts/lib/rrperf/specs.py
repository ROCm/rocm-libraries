################################################################################
#
# MIT License
#
# Copyright 2021-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

"""Get host/gpu specs."""

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
        return dedent(
            f"""\
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
        """
        )


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.stdout.decode("ascii")


def search(pattern, string):
    m = re.search(pattern, string, re.MULTILINE)
    if m is not None:
        return m.group(1)
    return None


def load_machine_specs(path):
    if path.exists():
        contents = path.read_text()
        if contents.startswith(MachineSpecs.yaml_tag):
            return yaml.load(contents, Loader=yaml.Loader)
    return MachineSpecs()


def get_machine_specs(devicenum, rocm_smi_path="rocm-smi"):
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

    rocm_smi_found = shutil.which(rocm_smi_path) is not None
    if rocm_smi_found:
        rocm_smi = run(
            [
                rocm_smi_path,
                "--showvbios",
                "--showid",
                "--showproductname",
                "--showperflevel",
                "--showclocks",
                "--showmeminfo",
                "vram",
            ]
        )
    else:
        rocm_smi = ""

    device = rf"^GPU\[{devicenum}\]\s*: "

    # Use the NODE_NAME env var in CI.
    hostname = os.environ.get("NODE_NAME")
    if not hostname:
        hostname = socket.gethostname()
    cpu = search(r"^model name\s*: (.*?)$", cpuinfo)
    kernel = search(r"version (\S*)", version)
    ram = search(r"MemTotal:\s*(\S*)", meminfo)
    distro = search(r'PRETTY_NAME="(.*?)"', os_release)
    rocmversion = rocm_info.strip()
    vbios = (
        search(device + r"VBIOS version: (.*?)$", rocm_smi)
        if rocm_smi_found
        else "no rocm-smi"
    )
    gpuid = (
        search(device + r"GPU ID: (.*?)$", rocm_smi)
        if rocm_smi_found
        else "no rocm-smi"
    )
    deviceinfo = (
        search(device + r"Card series:\s*(.*?)$", rocm_smi)
        if rocm_smi_found
        else "no rocm-smi"
    )
    vram = (
        search(device + r".... Total Memory .B.: (\d+)$", rocm_smi)
        if rocm_smi_found
        else 0
    )
    perflevel = (
        search(device + r"Performance Level: (.*?)$", rocm_smi)
        if rocm_smi_found
        else "no rocm-smi"
    )
    mclk = search(device + r"mclk.*\((.*?)\)$", rocm_smi) if rocm_smi_found else 0
    sclk = search(device + r"sclk.*\((.*?)\)$", rocm_smi) if rocm_smi_found else 0

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
