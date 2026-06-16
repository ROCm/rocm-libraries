#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# AICK-1303: set ENABLE_DYNAMIC_VGPR (COMPUTE_PGM_RSRC3 bit 17) in every kernel
# descriptor (.kd) of a gfx1250 AMDGPU ELF. Proves the dynamic-VGPR enable can be
# applied as a post-compile patch (the prefetch-PoC pattern). RSRC3 is at byte
# offset 44 of the 64-byte kernel_descriptor_t.
import sys, struct
from pathlib import Path

RSRC3_OFF = 44
DVGPR_BIT = 1 << 17

path = Path(sys.argv[1])
# optional comma-separated substring filter on .kd names (e.g. "fused_conv,solo8")
only = sys.argv[2].split(',') if len(sys.argv) > 2 else None
data = bytearray(path.read_bytes())

def wanted(nm):
    return only is None or any(o in nm for o in only)

# Minimal ELF64 parse: section headers -> find .symtab, .strtab, and section file offsets.
e_shoff = struct.unpack_from('<Q', data, 0x28)[0]
e_shentsize = struct.unpack_from('<H', data, 0x3a)[0]
e_shnum = struct.unpack_from('<H', data, 0x3c)[0]
e_shstrndx = struct.unpack_from('<H', data, 0x3e)[0]

secs = []
for i in range(e_shnum):
    base = e_shoff + i * e_shentsize
    name, stype, flags, addr, off, size, link, info, align, entsize = struct.unpack_from('<IIQQQQIIQQ', data, base)
    secs.append(dict(name=name, type=stype, addr=addr, off=off, size=size, link=link, entsize=entsize))

shstr_off = secs[e_shstrndx]['off']
def secname(s):
    end = data.index(b'\0', shstr_off + s['name'])
    return data[shstr_off + s['name']:end].decode()

sec_by_name = {secname(s): s for s in secs}
symtab = next(s for s in secs if s['type'] == 2)  # SHT_SYMTAB
strtab = secs[symtab['link']]
# file offset of a symbol's bytes: relocatable .o has sh_addr==0 (st_value is the
# section offset); linked .hsaco has sh_addr==vaddr (st_value is a vaddr) -> subtract.
def sym_file_off(shndx, st_value):
    sec = secs[shndx]
    return sec['off'] + (st_value - sec['addr'])

def symname(noff):
    end = data.index(b'\0', strtab['off'] + noff)
    return data[strtab['off'] + noff:end].decode()

patched = []
n = symtab['size'] // symtab['entsize']
for i in range(n):
    base = symtab['off'] + i * symtab['entsize']
    st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack_from('<IBBHQQ', data, base)
    nm = symname(st_name)
    if nm.endswith('.kd') and wanted(nm):
        file_off = sym_file_off(st_shndx, st_value) + RSRC3_OFF
        rsrc3 = struct.unpack_from('<I', data, file_off)[0]
        new = rsrc3 | DVGPR_BIT
        struct.pack_into('<I', data, file_off, new)
        patched.append((nm, rsrc3, new))

path.write_bytes(data)
for nm, old, new in patched:
    print(f"{nm}: RSRC3 0x{old:08x} -> 0x{new:08x}  (dvgpr_en bit17 = {(new>>17)&1})")
if not patched:
    print("no .kd symbols found")
