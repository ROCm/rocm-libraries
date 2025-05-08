# Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Get host/gpu specs."""

import re
import socket
import subprocess
import os
import shutil

from dataclasses import dataclass
from pathlib import Path as path
from textwrap import dedent


@dataclass
class MachineSpecs:
    hostname: str
    cpu: str
    sbios: str
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
    bandwidth: str

    def __str__(self):
        return dedent(f'''\
        Host info:
            hostname:       {self.hostname}
            cpu info:       {self.cpu}
            sbios info:     {self.sbios}
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
        ''')


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.stdout.decode('ascii')


def search(pattern, string):
    m = re.search(pattern, string, re.MULTILINE)
    if m is not None:
        return m.group(1)
    return None


def get_machine_specs(devicenum, type='default'):

    # host info var
    hostname = None
    cpu = None
    sbios = None
    kernel = None
    ram = None
    distro = None
    rocmversion = None
    # device info var
    vbios = None
    gpuid = None
    deviceinfo = None
    vram = None
    perflevel = None
    mclk = None
    sclk = None
    bandwidth = None

    if type == 'host' or type == 'default':
        hostname = socket.gethostname()

        version = path('/proc/version').read_text()
        kernel = search(r'version (\S*)', version)

        cpuinfo = path('/proc/cpuinfo').read_text()
        cpu = search(r'^model name\s*: (.*?)$', cpuinfo)

        meminfo = path('/proc/meminfo').read_text()
        ram = search(r'MemTotal:\s*(\S*)', meminfo)
        ram = '{:.2f} GiB'.format(float(ram) / 1024**2)

        os_release = path('/etc/os-release').read_text()
        distro = search(r'PRETTY_NAME="(.*?)"', os_release)

        sbios = path('/sys/class/dmi/id/bios_vendor').read_text().strip(
        ) + path('/sys/class/dmi/id/bios_version').read_text().strip()

        # Todo: check the fixed path, maybe set the rocm path as a variable,
        #       the same as rocm-smi.
        if os.path.isfile('/opt/rocm/.info/version-utils'):
            rocm_info = path('/opt/rocm/.info/version-utils').read_text()
        elif os.path.isfile('/opt/rocm/.info/version'):
            rocm_info = path('/opt/rocm/.info/version').read_text()
        else:
            rocm_info = "rocm info not available"
        rocmversion = rocm_info.strip()

    if type == 'device' or type == 'default':
        rocm_smi_found = shutil.which('rocm-smi') != None
        if rocm_smi_found:
            rocm_smi = run([
                'rocm-smi', '--showvbios', '--showid', '--showproductname',
                '--showperflevel', '--showclocks', '--showmeminfo', 'vram'
            ])
        else:
            rocm_smi = ""

        device = rf'^GPU\[{devicenum}\]\s*: '

        vbios = search(device + r'VBIOS version: (.*?)$',
                       rocm_smi) if rocm_smi_found else "no rocm-smi"
        gpuid = search(device + r'GPU ID: (.*?)$',
                       rocm_smi) if rocm_smi_found else "no rocm-smi"
        deviceinfo = search(device + r'Card series:\s*(.*?)$',
                            rocm_smi) if rocm_smi_found else "no rocm-smi"
        vram = search(device + r'.... Total Memory .B.: (\d+)$',
                      rocm_smi) if rocm_smi_found else 0
        perflevel = search(device + r'Performance Level: (.*?)$',
                           rocm_smi) if rocm_smi_found else "no rocm-smi"
        mclk = search(device +
                      r'mclk.*\((.*?)\)$', rocm_smi) if rocm_smi_found else 0
        sclk = search(device +
                      r'sclk.*\((.*?)\)$', rocm_smi) if rocm_smi_found else 0

        vram = '{:.2f} GiB'.format(float(vram) / 1024**3 if vram else 0)

        if gpuid == '0x66af':
            # radeon7: float: 13.8 TFLOPs, double: 3.46 TFLOPs, 1024 GB/s
            bandwidth = (13.8, 3.46, 1024)

    return MachineSpecs(hostname, cpu, sbios, kernel, ram, distro, rocmversion,
                        vbios, gpuid, deviceinfo, vram, perflevel, mclk, sclk,
                        bandwidth)


if __name__ == '__main__':
    print(get_machine_specs(0))
