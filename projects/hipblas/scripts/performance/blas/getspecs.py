"""Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights reserved.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
"""

import os
from pathlib import Path as path
import csv
from decimal import Decimal

def get_csv_val(datastr, title, gpu=0):
    """
    Parse the csv value for 'title' from 'datastr'. Ensures that the 'datastr' is for the 'gpu' passed.
    """
    reader = csv.reader(datastr.split('\n'))
    try:
        header = next(reader)
        data = next(reader)
        if int(data[header.index('gpu')]) != gpu:
            return "error"
        return data[header.index(title)]
    except:
        return "error"

def _subprocess_helper(cmd, *args, **kwargs):
    import subprocess
    import tempfile
    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")
    cout = ""
    success = False
    try:
        p = subprocess.Popen(cmd, stdout=fout, stderr=ferr, *args, **kwargs)
        p.wait()
        fout.seek(0)
        cout = fout.read()
        success = True
    except subprocess.CalledProcessError:
        pass
    except FileNotFoundError:
        pass
    return success, cout

def get_smi_exec(cuda):
    if cuda:
        return "nvidia-smi"
    else:
        return "/opt/rocm/bin/amd-smi"

_AMD_SMI_CLOCKS = {
    'sclk': {'static': 'SYS', 'metric_cols': ['gfx_0_clk', 'clk', 'cur_clk']},
    'mclk': {'static': 'MEM', 'metric_cols': ['mem_0_clk', 'MEM_clk', 'MEM_cur_clk']},
    'fclk': {'static': 'DF', 'metric_cols': ['fclk_0_clk']},
    'socclk': {'static': 'SOC', 'metric_cols': ['socclk_0_clk']},
    'vclk0': {'static': 'VCLK0', 'metric_cols': ['vclk_0_clk', 'VCLK0_clk', 'VCLK0_cur_clk']},
}

def _metric_clock_val(cout, devicenum, metric_cols):
    for col in metric_cols:
        val = get_csv_val(cout, col, devicenum)
        if val != 'error':
            return val
    return "error"

def getgfx(devicenum, cuda):
    if cuda:
        return "N/A"

    cmd = ["/opt/rocm/bin/rocm_agent_enumerator"]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"

    gfx_archs = [
        line.strip()
        for line in cout.splitlines()
        if line.strip().startswith('gfx')
    ]
    if not gfx_archs:
        return "N/A"

    # Older rocm_agent_enumerator builds prepend a placeholder gfx000 line.
    if gfx_archs[0] == 'gfx000' and len(gfx_archs) > 1:
        index = devicenum + 1
    else:
        index = devicenum

    if index < 0 or index >= len(gfx_archs):
        return "N/A"

    return gfx_archs[index]

# Get the hostname
def gethostname():
    import socket
    hostname = socket.gethostname()
    return hostname

# Get the host cpu information
def getcpu():
    cmd = ["lscpu"]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    cpulist = ""
    searchstr = "Model name:"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            cpulist += line[len(searchstr):].strip()
    return cpulist

# Get the kernel version
def getkernel():
    import subprocess
    cmd = ["uname", "-r"]
    import tempfile
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    return cout.strip()

# Get the host ram size
def getram():
    import re
    cmd = ["lshw", "-class",  "memory"]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    searchstr = "size:"
    for line in cout.split("\n"):
        m = re.search(searchstr, line)
        if not m == None:
            return line.strip()[len(searchstr):].strip()

# Get the Linux distro information
def getdistro():
    cmd = ["lsb_release", "-a"]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    searchstr = "Description:"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            return line[len(searchstr):].strip()

# Get the version number for rocm
def getrocmversion():
    if os.path.isfile("/opt/rocm/.info/version-utils"):
        rocm_info = path("/opt/rocm/.info/version-utils").read_text()
    elif os.path.isfile("/opt/rocm/.info/version"):
        rocm_info = path("/opt/rocm/.info/version").read_text()
    else:
        return "N/A"

    return rocm_info.strip()


# Get the vbios version for the specified device
def getvbios(devicenum, cuda):
    cmd = [get_smi_exec(cuda), "static", "-g", str(devicenum), "-V", "--csv"]
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=vbios_version", "--format=csv,noheader", "-i", str(devicenum)]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    if cuda:
        return cout

    return get_csv_val(cout, 'version', devicenum)

def getgpuid(devicenum, cuda):
    cmd = [get_smi_exec(cuda), "static", "-g", str(devicenum), "-a", "--csv"]
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=name", "--format=csv,noheader", "-i", str(devicenum)]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    if cuda:
        return cout

    return get_csv_val(cout, 'market_name', devicenum)

# Get the name of the device from lshw which has index devicenum
def getdeviceinfo(devicenum, cuda):
    import re
    cmd = ["lshw", "-C", "video"]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    searchstr = "-display"
    indices = []
    name = ""
    for idx, line in enumerate(cout.split("\n")):
        if re.search(searchstr, line) != None:
            indices.append(idx)
    for idx, line in enumerate(cout.split("\n")):
        if idx >= indices[devicenum]:
            searchstr = "product:"
            if re.search(searchstr, line) != None:
                pos = line.find(":")
                name += line[pos+1:].strip()
    name += " " + getgpuid(devicenum, cuda)
    return name

# Get the vram for the specified device
def getvram(devicenum, cuda):
    cmd = [get_smi_exec(cuda), "static", "-g", str(devicenum), "-v", "--csv"]
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=memory.total", "--format=csv,noheader", "-i", str(devicenum)]
    success, cout = _subprocess_helper(cmd)

    if not success:
        return "N/A"
    if cuda:
        return cout

    val = get_csv_val(cout, 'size', devicenum)
    if val == 'error':
        val = get_csv_val(cout, 'vram_size_mb', devicenum)
    return val + " MB"

# Get the performance level for the specified device
def getperflevel(devicenum, cuda):
    cmd = [get_smi_exec(cuda), "metric", "-g", str(devicenum), "-l", "--csv"]
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=pstate", "--format=csv,noheader", "-i", str(devicenum)]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    if cuda:
        return cout

    val = get_csv_val(cout, 'perf_level', devicenum)
    if val == 'error':
        return "N/A"
    return val.replace('AMDSMI_DEV_PERF_LEVEL_', '')

# Get the memory clock for the specified device
def getmclk(devicenum, cuda):
    cmd = [get_smi_exec(cuda), "metric", "-g", str(devicenum), "-c", "--csv"]
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=clocks.mem", "--format=csv,noheader", "-i", str(devicenum)]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    if cuda:
        return cout

    return _metric_clock_val(cout, devicenum, _AMD_SMI_CLOCKS['mclk']['metric_cols'])

# Get the system clock for the specified device
def getsclk(devicenum, cuda):
    cmd = [get_smi_exec(cuda), "metric", "-g", str(devicenum), "-c", "--csv"]
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=clocks.sm", "--format=csv,noheader", "-i", str(devicenum)]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    if cuda:
        return cout

    return _metric_clock_val(cout, devicenum, _AMD_SMI_CLOCKS['sclk']['metric_cols'])

def getbandwidth(devicenum, cuda):
    gpuid = getgpuid(devicenum, cuda)
    if gpuid == "0x66af":
        # radeon7: float: 13.8 TFLOPs, double: 3.46 TFLOPs, 1024 GB/s
        return (13.8, 3.46, 1024)

def listdevices(cuda, smi=None):
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=count", "--format=csv,noheader", "-i", '0']
        success, cout = _subprocess_helper(cmd)
        if not success:
            return []
        return list(range(0, int(cout)))
    else:
        cmd = [get_smi_exec(cuda), "list", "--csv"]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return []
        return list(range(0, cout.count('\n') - 2))

def getbus(devicenum, cuda, smi=None):
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=pci.bus_id", "--format=csv,noheader", "-i", str(devicenum)]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"
        return cout
    else:
        cmd = [get_smi_exec(cuda), "static", "-g", str(devicenum), "-b", "--csv"]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"
        val = get_csv_val(cout, 'bdf', devicenum)
        if val == 'error':
            val = get_csv_val(cout, 'gpu_bdf', devicenum)
        return val

def getprofile(devicenum, cuda):
    if cuda:
        return "N/A"
    else:
        cmd = [get_smi_exec(cuda), "metric", "-g", str(devicenum), "-p", "--csv"]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"

        return get_csv_val(cout, 'power_management', devicenum)

def getfanspeedpercent(devicenum, cuda, smi=None):
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=fan.speed", "--format=csv,noheader", "-i", str(devicenum)]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"
        return str(cout)
    else:
        cmd = [get_smi_exec(cuda), "metric", "-g", str(devicenum), "-f", "--csv"]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"
        return get_csv_val(cout, 'usage', devicenum)

def validclocknames(cuda, smi=None):
    if cuda:
        return ["graphics", "sm", "memory", "video"]
    else:
        return ["sclk", "mclk", "fclk", "socclk", "vclk0"]

def getcurrentclockfreq(devicenum, clock, cuda, smi=None):
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=clocks.current." + clock, "--format=csv,noheader", "-i", str(devicenum)]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"
        return cout
    else:
        clock_info = _AMD_SMI_CLOCKS.get(clock)
        if clock_info is None:
            return "N/A"
        cmd = [get_smi_exec(cuda), "metric", "-g", str(devicenum), "-c", "--csv"]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"
        return _metric_clock_val(cout, devicenum, clock_info['metric_cols'])

def getcurrentclocklevel(devicenum, clock, cuda):
    if cuda:
        return "N/A"
    else:
        clock_info = _AMD_SMI_CLOCKS.get(clock)
        if clock_info is None:
            return "N/A"
        cmd = [get_smi_exec(cuda), "static", "-g", str(devicenum), "-C", clock_info['static'], "--csv"]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"

        val = get_csv_val(cout, 'current_level', devicenum)
        if val != 'error':
            return val
        prefix = clock_info['static'].lower()
        return get_csv_val(cout, prefix + '_current_level', devicenum)

def getmaxlevel(devicenum, clock, cuda):
    if cuda:
        return "N/A"
    else:
        clock_info = _AMD_SMI_CLOCKS.get(clock)
        if clock_info is None:
            return "N/A"
        cmd = [get_smi_exec(cuda), "static", "-g", str(devicenum), "-C", clock_info['static'], "--csv"]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"

        reader = csv.reader(cout.split('\n'))
        try:
            header = next(reader)
        except StopIteration:
            return "N/A"
        level_cols = [
            col for col in header
            if col.startswith('frequency_levels_Level') or col.startswith('Level ')
        ]
        if not level_cols:
            return "N/A"
        return str(len(level_cols) - 1)


def validmemtypes(cuda, smi=None):
    if cuda:
        return ["vram"]
    else:
        return ["vram", "visible_vram", "gtt"]

def getmeminfo(devicenum, mem_type, cuda, smi=None):
    if cuda:
        if mem_type == 'vram':
            cmd = [get_smi_exec(cuda), "--query-gpu=memory.total", "--format=csv,noheader", "-i", str(devicenum)]
            success, cout_total = _subprocess_helper(cmd)
            if not success:
                return "N/A"
            cmd = [get_smi_exec(cuda), "--query-gpu=memory.used", "--format=csv,noheader", "-i", str(devicenum)]
            success, cout_used = _subprocess_helper(cmd)
            if not success:
                return "N/A"
            return cout_used, cout_total
        else:
            return "N/A"
    else:
        cmd = [get_smi_exec(cuda), "metric", "-g", str(devicenum), "-m", "--csv"]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A", "N/A"
        return get_csv_val(cout, "used_" + mem_type, devicenum), get_csv_val(cout, "total_" + mem_type, devicenum)

def validversioncomponents(cuda, smi=None):
    if cuda:
        return ['driver']
    else:
        return ['driver']

def getversion(devicenum, component, cuda, smi=None):
    if cuda:
        if component == 'driver':
            cmd = [get_smi_exec(cuda), "--query-gpu=driver_version", "--format=csv,noheader", "-i", str(devicenum)]
            success, cout = _subprocess_helper(cmd)
            if not success:
                return "N/A"
            return cout
        else:
            return "N/A"
    elif component == 'driver':
        cmd = [get_smi_exec(cuda), "static", "-g", str(devicenum), "-d", "--csv"]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"
        val = get_csv_val(cout, 'version', devicenum)
        if val == 'error':
            val = get_csv_val(cout, 'driver_version', devicenum)
        return val
    else:
        return "N/A"
