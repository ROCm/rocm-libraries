#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# AICK-1303: read-only check of ENABLE_DYNAMIC_VGPR (COMPUTE_PGM_RSRC3 bit 17) for
# every kernel descriptor (.kd) in one or more gfx1250 AMDGPU ELF/code objects.
# Handles both relocatable (.o) and linked (.hsaco) objects (st_value is a section
# offset vs a vaddr). Usage: verify_dvgpr.py OBJ [OBJ ...]
import sys, struct
from pathlib import Path

RSRC3_OFF = 44

def dump(path):
    data = Path(path).read_bytes()
    e_shoff = struct.unpack_from('<Q', data, 0x28)[0]
    e_shentsize = struct.unpack_from('<H', data, 0x3a)[0]
    e_shnum = struct.unpack_from('<H', data, 0x3c)[0]
    secs = [struct.unpack_from('<IIQQQQIIQQ', data, e_shoff + i * e_shentsize) for i in range(e_shnum)]
    sym = next(s for s in secs if s[1] == 2)          # SHT_SYMTAB
    strt = secs[sym[6]]
    n = sym[5] // sym[9]
    print(f"== {path} ==")
    for i in range(n):
        b = sym[4] + i * sym[9]
        st_name, _, _, st_shndx, st_value, _ = struct.unpack_from('<IBBHQQ', data, b)
        end = data.index(b'\0', strt[4] + st_name)
        nm = data[strt[4] + st_name:end].decode()
        if nm.endswith('.kd'):
            sec = secs[st_shndx]
            off = sec[4] + (st_value - sec[3]) + RSRC3_OFF   # sh_off + (val - sh_addr)
            rsrc3 = struct.unpack_from('<I', data, off)[0]
            print(f"   {nm:<34} RSRC3=0x{rsrc3:08x}  dvgpr_en={(rsrc3 >> 17) & 1}")

for p in sys.argv[1:]:
    dump(p)
