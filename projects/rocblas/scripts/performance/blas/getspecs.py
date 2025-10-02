"""Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
   ies of the Software, and to permit persons to whom the Software is furnished
   to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
   PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
   FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
   IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
   CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
from pathlib import Path as path
import csv
from decimal import Decimal

def get_csv_val(datastr, title, gpu=0):
    """
    Parse the csv value for 'title' from 'datastr'. Ensures that the 'datastr' is for the 'gpu' passed.

    :param: datastr: The data as output by amd-smi in csv format, see example:
        gpu,total_vram,used_vram,free_vram,total_visible_vram,used_visible_vram,free_visible_vram,total_gtt,used_gtt,free_gtt
        0,16368,905,15463,256,26,230,32119,755,31364
    :param: title: The title in the datastr to parse for, example for above output: "total_vram", would get "16368" as output.
    :param: gpu: The gpu# we are searching for, it will be validated that 'gpu' is the same as 'gpu' in the datastr, otherwise
                 this function will return "error"
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

def getgfx(devicenum, cuda):
    if cuda:
        return "N/A"
    else:
        cmd = ["/opt/rocm/bin/rocm_agent_enumerator"]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"
        # Add 1 to devicenum since rocm-agent-enum always prints gfx000 first
        return cout.splitlines()[devicenum+1]

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
    import re
    name = ""
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
    import re
    cmd = [get_smi_exec(cuda), "static", "-g", str(devicenum), "-v", "--csv"]
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=memory.total", "--format=csv,noheader", "-i", str(devicenum)]
    success, cout = _subprocess_helper(cmd)

    if not success:
        return "N/A"
    if cuda:
        return cout

    val = get_csv_val(cout, 'size', devicenum)
    if(val == 'error'):
        val = get_csv_val(cout, 'vram_size_mb', devicenum) # ROCm 6.0 output
    return val + " MB"

# Get the memory clock for the specified device
def getmclk(devicenum, cuda):
    import re
    cmd = [get_smi_exec(cuda), "metric", "-g", str(devicenum), "-c", "--csv"]
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=clocks.mem", "--format=csv,noheader", "-i", str(devicenum)]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    if cuda:
        return cout

    val = get_csv_val(cout, 'MEM_clk', devicenum)
    if val == 'error':
        val = get_csv_val(cout, 'MEM_cur_clk', devicenum) # ROCm 6.0 output

# Get the system clock for the specified device
def getsclk(devicenum, cuda):
    import re
    cmd = [get_smi_exec(cuda), "metric", "-g", str(devicenum), "-c", "--csv"]
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=clocks.sm", "--format=csv,noheader", "-i", str(devicenum)]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    if cuda:
        return cout

    val = get_csv_val(cout, 'clk', devicenum)
    if val == 'error':
        val = get_csv_val(cout, 'cur_clk', devicenum) # ROCm 6.0 output
    return val

def listdevices(cuda):
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=count", "--format=csv,noheader", "-i", '0']
        success, cout = _subprocess_helper(cmd)
        if not success:
            return []
        return list(range(0, int(cout)))
    else:
         cmd = [get_smi_exec(cuda), "list", "--csv"]
         success, cout = _subprocess_helper(cmd)
         return list(range(0, cout.count('\n') - 2))

def getbus(devicenum, cuda):
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=pci.bus_id", "--format=csv,noheader", "-i", str(devicenum)]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"
        return cout
    else:
        cmd = [get_smi_exec(cuda), "static", "-g", str(devicenum), "-b", "--csv"]
        success, cout = _subprocess_helper(cmd)
        return get_csv_val(cout, 'bdf', devicenum)

def getprofile(devicenum, cuda):
    import re
    if cuda:
        return "N/A"
    else:
        cmd = [get_smi_exec(cuda), "metric", "-g", str(devicenum), "-p", "--csv"]
        success, cout = _subprocess_helper(cmd)

        return get_csv_val(cout, 'power_management', devicenum)

def getfanspeedpercent(devicenum, cuda):
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=fan.speed", "--format=csv,noheader", "-i", str(devicenum)]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"
        return str(cout)
    else:
        cmd = [get_smi_exec(cuda), "metric", "-g", str(devicenum), "-f", "--csv"]
        success, cout = _subprocess_helper(cmd)
        return get_csv_val(cout, 'usage', devicenum)

def validclocknames(cuda):
    if cuda:
        return ["graphics", "sm", "memory", "video"]
    else:
        host_rocm_ver = Decimal('.'.join(getrocmversion().split('.')[0:2])) # get host's rocm major.minor version
        rocm_6_1_ver = Decimal('6.1')
        if rocm_6_1_ver.compare(host_rocm_ver) == 1:
            return ["cur_clk", "MEM_cur_clk", "VCLK0_cur_clk"] # For versions below ROCm 6.1
        else:
            return ["clk", "MEM_clk", "VCLK0_clk"]

def getcurrentclockfreq(devicenum, clock, cuda):
    import re
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=clocks.current." + clock, "--format=csv,noheader", "-i", str(devicenum)]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"
        return cout
    else:
        cmd = [get_smi_exec(cuda), "metric", "-g", str(devicenum), "-c", "--csv"]
        success, cout = _subprocess_helper(cmd)
        return get_csv_val(cout, clock, devicenum)

def validmemtypes(cuda):
    if cuda:
        return ["vram"]
    else:
        return ["vram", "vram", "gtt"]

def getmeminfo(devicenum, mem_type, cuda):
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
        return get_csv_val(cout, "used_" + mem_type, devicenum), get_csv_val(cout, "total_" + mem_type, devicenum)

def validversioncomponents(cuda):
    # currently only driver according to /opt/rocm/bin/rocm_smi.py
    # driver corresponds to 0 in /opt/rocm/bin/rocm_smi.py
    if cuda:
        return ['driver']
    else:
        # currently only driver according to /opt/rocm/bin/rocm_smi.py
        return ['driver']

def getversion(devicenum, component, cuda):
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
        val = get_csv_val(cout, 'version', devicenum)
        if val == 'error':
            val = get_csv_val(cout, 'driver_version', devicenum) # ROCm 6.0 output
        return val
    else:
        return "N/A"
