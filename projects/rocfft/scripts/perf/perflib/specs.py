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

import json
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
    return p.stdout.decode('utf-8', errors='replace')


def parse_amdsmi_json(text):
    # amd-smi may prepend non-JSON warnings (e.g. permission notices) to
    # stdout. Strip anything before the first '{' so json.loads succeeds.
    idx = text.find('{')
    if idx < 0:
        return {}
    return json.loads(text[idx:])


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
        amd_smi_found = shutil.which('amd-smi') is not None
        static_entry = None
        metric_entry = None
        if amd_smi_found:
            try:
                static_data = parse_amdsmi_json(
                    run([
                        'amd-smi', 'static', '--vbios', '--vram', '--board',
                        '--asic', '--json'
                    ]))
                metric_data = parse_amdsmi_json(
                    run([
                        'amd-smi', 'metric', '--clock', '--perf-level',
                        '--json'
                    ]))
                for entry in static_data.get('gpu_data', []):
                    if entry.get('gpu') == devicenum:
                        static_entry = entry
                        break
                for entry in metric_data.get('gpu_data', []):
                    if entry.get('gpu') == devicenum:
                        metric_entry = entry
                        break
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # Conversion factors from amd-smi size units to GiB. Tolerates the
        # MB/MiB (and KB/KiB, GB/GiB) ambiguity in amd-smi output by treating
        # them as equivalent powers of 2.
        unit_to_gib = {
            'KB': 1 / 1024**2,
            'KiB': 1 / 1024**2,
            'MB': 1 / 1024,
            'MiB': 1 / 1024,
            'GB': 1.0,
            'GiB': 1.0,
            'TB': 1024.0,
            'TiB': 1024.0,
        }

        if static_entry is not None:
            vbios = static_entry.get('ifwi', {}).get('part_number',
                                                    'no amd-smi')
            gpuid = static_entry.get('asic', {}).get('device_id', 'no amd-smi')
            deviceinfo = static_entry.get('asic', {}).get(
                'market_name', 'no amd-smi')
            vram_size = static_entry.get('vram', {}).get('size', {})
            vram_value = vram_size.get('value', 0)
            vram_unit = vram_size.get('unit', 'MB')
            vram_gib = float(vram_value) * unit_to_gib.get(vram_unit, 0)
        else:
            vbios = 'no amd-smi'
            gpuid = 'no amd-smi'
            deviceinfo = 'no amd-smi'
            vram_gib = 0

        if metric_entry is not None:
            perflevel = metric_entry.get('perf_level', 'no amd-smi')
            if isinstance(perflevel, str):
                perflevel = perflevel.replace('AMDSMI_DEV_PERF_LEVEL_',
                                              '').lower()
            mem_clk = metric_entry.get('clock',
                                       {}).get('mem_0', {}).get('clk', {})
            mclk = '{}{}'.format(mem_clk.get('value', 0),
                                 mem_clk.get('unit', '')) if mem_clk else 0
            gfx_clk = metric_entry.get('clock',
                                       {}).get('gfx_0', {}).get('clk', {})
            sclk = '{}{}'.format(gfx_clk.get('value', 0),
                                 gfx_clk.get('unit', '')) if gfx_clk else 0
        else:
            perflevel = 'no amd-smi'
            mclk = 0
            sclk = 0

        vram = '{:.2f} GiB'.format(vram_gib)

        if gpuid == '0x66af':
            # radeon7: float: 13.8 TFLOPs, double: 3.46 TFLOPs, 1024 GB/s
            bandwidth = (13.8, 3.46, 1024)

    return MachineSpecs(hostname, cpu, sbios, kernel, ram, distro, rocmversion,
                        vbios, gpuid, deviceinfo, vram, perflevel, mclk, sclk,
                        bandwidth)


if __name__ == '__main__':
    print(get_machine_specs(0))
