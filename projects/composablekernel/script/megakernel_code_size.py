#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Per-kernel code size and register/LDS usage of AMDGPU code objects, without running anything.

Accepts any of:
  * a host executable or shared object with an embedded .hip_fatbin (HIP offload bundle),
  * an offload bundle file (__CLANG_OFFLOAD_BUNDLE__, uncompressed or CCOB-compressed),
  * an AMDGPU ELF code object (e.g. from --cuda-device-only -c, or an extracted bundle entry),
  * a device assembly file (.s, from -S / --save-temps), read from its codeLenInByte comments.

For every function symbol it prints the bytes of code (ELF st_size, i.e. the bytes the
instruction cache has to hold), the 128 B lines they span, the kernel descriptor's INST_PREF_SIZE
("ipref": lines the hardware prefetches at wave launch, set by LLVM from the code size), and for kernels the VGPR / SGPR / LDS / scratch and spill counts
from the code object's AMDGPU metadata note (msgpack, code object v5/v6). Also prints the total
size of the executable sections ("code"), an upper bound on what a code-warm routine touches.

Pure Python 3, no dependencies (llvm-cxxfilt is used to demangle if found).

Examples:
  megakernel_code_size.py build/bin/tile_example_megakernel_gemm_chain
  megakernel_code_size.py --arch gfx1250 --filter Megakernel a.out
  megakernel_code_size.py foo.cpp.gfx1250.s
  megakernel_code_size.py --json bin/x > sizes.json
"""

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
CCOB_MAGIC = b"CCOB"
ELF_MAGIC = b"\x7fELF"
EM_AMDGPU = 224
NT_AMDGPU_METADATA = 32
ICACHE_LINE = 128  # gfx1250 instruction cache line size in bytes (LLVM FeatureInstCacheLineSize128)


# ----------------------------------------------------------------------------- msgpack (subset)
class _MsgPack:
    def __init__(self, data):
        self.d, self.i = data, 0

    def _take(self, n):
        b = self.d[self.i : self.i + n]
        self.i += n
        return b

    def _u(self, fmt):
        n = struct.calcsize(fmt)
        return struct.unpack(fmt, self._take(n))[0]

    def read(self):
        t = self._take(1)[0]
        if t <= 0x7F:
            return t
        if t >= 0xE0:
            return t - 0x100
        if 0x80 <= t <= 0x8F:
            return self._map(t & 0x0F)
        if 0x90 <= t <= 0x9F:
            return [self.read() for _ in range(t & 0x0F)]
        if 0xA0 <= t <= 0xBF:
            return self._take(t & 0x1F).decode("utf-8", "replace")
        simple = {
            0xC0: lambda: None,
            0xC2: lambda: False,
            0xC3: lambda: True,
            0xC4: lambda: bytes(self._take(self._u(">B"))),
            0xC5: lambda: bytes(self._take(self._u(">H"))),
            0xC6: lambda: bytes(self._take(self._u(">I"))),
            0xCA: lambda: self._u(">f"),
            0xCB: lambda: self._u(">d"),
            0xCC: lambda: self._u(">B"),
            0xCD: lambda: self._u(">H"),
            0xCE: lambda: self._u(">I"),
            0xCF: lambda: self._u(">Q"),
            0xD0: lambda: self._u(">b"),
            0xD1: lambda: self._u(">h"),
            0xD2: lambda: self._u(">i"),
            0xD3: lambda: self._u(">q"),
            0xD9: lambda: self._take(self._u(">B")).decode("utf-8", "replace"),
            0xDA: lambda: self._take(self._u(">H")).decode("utf-8", "replace"),
            0xDB: lambda: self._take(self._u(">I")).decode("utf-8", "replace"),
            0xDC: lambda: [self.read() for _ in range(self._u(">H"))],
            0xDD: lambda: [self.read() for _ in range(self._u(">I"))],
            0xDE: lambda: self._map(self._u(">H")),
            0xDF: lambda: self._map(self._u(">I")),
        }
        if t in simple:
            return simple[t]()
        raise ValueError("unsupported msgpack type 0x%02x" % t)

    def _map(self, n):
        out = {}
        for _ in range(n):
            k = self.read()
            out[k] = self.read()
        return out


# ----------------------------------------------------------------------------- ELF
class Elf:
    def __init__(self, data):
        if data[:4] != ELF_MAGIC or data[4] != 2 or data[5] != 1:
            raise ValueError("not a little-endian ELF64 file")
        self.data = data
        (self.e_type, self.e_machine, _, _, _, shoff, self.e_flags, _, _, _, shentsize, shnum,
         shstrndx) = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
        self.sections = []
        for k in range(shnum):
            (name, typ, flags, addr, off, size, link, info, align, entsize) = struct.unpack_from(
                "<IIQQQQIIQQ", data, shoff + k * shentsize)
            self.sections.append(dict(name_off=name, type=typ, flags=flags, addr=addr, offset=off,
                                      size=size, link=link, entsize=entsize, index=k))
        strtab = self.sections[shstrndx]
        for s in self.sections:
            s["name"] = self._cstr(strtab["offset"] + s["name_off"])

    def _cstr(self, off):
        end = self.data.index(b"\0", off)
        return self.data[off:end].decode("utf-8", "replace")

    def section(self, name):
        for s in self.sections:
            if s["name"] == name:
                return s
        return None

    def section_bytes(self, s):
        return self.data[s["offset"] : s["offset"] + s["size"]]

    def symbols(self):
        out = []
        for s in self.sections:
            if s["type"] not in (2, 11):  # SHT_SYMTAB, SHT_DYNSYM
                continue
            strs = self.sections[s["link"]]
            for k in range(s["size"] // 24):
                name, info, _, shndx, value, size = struct.unpack_from(
                    "<IBBHQQ", self.data, s["offset"] + k * 24)
                out.append(dict(name=self._cstr(strs["offset"] + name), type=info & 0xF,
                                bind=info >> 4, shndx=shndx, value=value, size=size,
                                table=s["name"]))
        return out

    def amdgpu_metadata(self):
        for s in self.sections:
            if s["type"] != 7:  # SHT_NOTE
                continue
            off, end = s["offset"], s["offset"] + s["size"]
            while off + 12 <= end:
                namesz, descsz, ntype = struct.unpack_from("<III", self.data, off)
                name = self.data[off + 12 : off + 12 + namesz].rstrip(b"\0")
                doff = off + 12 + ((namesz + 3) & ~3)
                if name == b"AMDGPU" and ntype == NT_AMDGPU_METADATA:
                    return _MsgPack(self.data[doff : doff + descsz]).read()
                off = doff + ((descsz + 3) & ~3)
        return {}


# ----------------------------------------------------------------------------- bundles
def _decompress_ccob(blob):
    """Compressed offload bundle (clang -offload-compress). zlib is handled here, zstd is left
    to clang-offload-bundler (which knows every header version)."""
    bundler = shutil.which("clang-offload-bundler") or "/opt/rocm/llvm/bin/clang-offload-bundler"
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.bundle")
        with open(src, "wb") as f:
            f.write(blob)
        try:
            out = subprocess.run([bundler, "--list", "--type=o", "--input=" + src],
                                 capture_output=True, text=True, check=True).stdout.split()
        except (OSError, subprocess.CalledProcessError) as e:
            # Last resort: version-1 header with zlib payload.
            method = struct.unpack_from("<H", blob, 6)[0]
            if method == 0:  # zlib
                return zlib.decompress(blob[24:])
            raise RuntimeError("cannot decompress CCOB bundle: %s" % e)
        entries = {}
        for target in out:
            dst = os.path.join(d, "out%d" % len(entries))
            subprocess.run([bundler, "--unbundle", "--type=o", "--input=" + src,
                            "--targets=" + target, "--output=" + dst], check=True)
            with open(dst, "rb") as f:
                entries[target] = f.read()
        return entries


def split_bundle(blob):
    """Return {target_id: bytes} for an offload bundle blob (possibly several concatenated)."""
    entries = {}
    pos = 0
    while pos < len(blob):
        if blob[pos : pos + 4] == CCOB_MAGIC:
            r = _decompress_ccob(blob[pos:])
            if isinstance(r, dict):
                entries.update(r)
                break
            blob, pos = r, 0
            continue
        if blob[pos : pos + len(BUNDLE_MAGIC)] != BUNDLE_MAGIC:
            nxt = blob.find(BUNDLE_MAGIC, pos + 1)
            nxc = blob.find(CCOB_MAGIC, pos + 1)
            cands = [x for x in (nxt, nxc) if x > 0]
            if not cands:
                break
            pos = min(cands)
            continue
        base = pos
        (n,) = struct.unpack_from("<Q", blob, pos + 24)
        p = pos + 32
        end = base
        for _ in range(n):
            off, size, idlen = struct.unpack_from("<QQQ", blob, p)
            tid = blob[p + 24 : p + 24 + idlen].decode()
            p += 24 + idlen
            if size:
                entries[tid] = blob[base + off : base + off + size]
            end = max(end, base + off + size)
        pos = max(end, p)
        # Bundles in .hip_fatbin are 4 KiB aligned; skip padding.
        nxt = blob.find(BUNDLE_MAGIC, pos)
        nxc = blob.find(CCOB_MAGIC, pos)
        cands = [x for x in (nxt, nxc) if x >= 0]
        if not cands:
            break
        pos = min(cands)
    return entries


def code_objects(path):
    """Yield (label, Elf) for every AMDGPU code object in path."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] == ELF_MAGIC:
        elf = Elf(data)
        if elf.e_machine == EM_AMDGPU:
            yield os.path.basename(path), elf
            return
        found = False
        for name in (".hip_fatbin", "__CLANG_OFFLOAD_BUNDLE__sycl-spir64", ".llvm.offloading"):
            s = elf.section(name)
            if s is None:
                continue
            for tid, blob in split_bundle(elf.section_bytes(s)).items():
                if blob[:4] == ELF_MAGIC and Elf(blob).e_machine == EM_AMDGPU:
                    found = True
                    yield tid, Elf(blob)
        if not found:
            # e.g. a host .o from -fgpu-rdc / a relocatable fat object: scan the whole file.
            for tid, blob in split_bundle(data).items():
                if blob[:4] == ELF_MAGIC and Elf(blob).e_machine == EM_AMDGPU:
                    yield tid, Elf(blob)
        return
    if data[: len(BUNDLE_MAGIC)] == BUNDLE_MAGIC or data[:4] == CCOB_MAGIC:
        for tid, blob in split_bundle(data).items():
            if blob[:4] == ELF_MAGIC and Elf(blob).e_machine == EM_AMDGPU:
                yield tid, Elf(blob)
        return
    raise ValueError("%s: not an ELF, offload bundle or .s file" % path)


# ----------------------------------------------------------------------------- reports
def _demangle(names):
    tool = shutil.which("llvm-cxxfilt") or "/opt/rocm/llvm/bin/llvm-cxxfilt"
    try:
        out = subprocess.run([tool], input="\n".join(names), capture_output=True, text=True,
                             check=True).stdout.splitlines()
        if len(out) == len(names):
            return dict(zip(names, out))
    except (OSError, subprocess.CalledProcessError):
        pass
    return {n: n for n in names}


def _inst_pref_lines(elf, syms, name):
    """INST_PREF_SIZE from the kernel descriptor: COMPUTE_PGM_RSRC3 (offset 44) bits [11:4] on
    gfx12 (gfx11: [9:4]). The hardware prefetches this many 128 B lines of code at wave launch."""
    kd = syms.get(name + ".kd")
    if kd is None or kd["shndx"] == 0 or kd["shndx"] >= len(elf.sections):
        return None
    sec = elf.sections[kd["shndx"]]
    off = sec["offset"] + kd["value"] - sec["addr"]
    if off + 64 > len(elf.data):
        return None
    rsrc3 = struct.unpack_from("<I", elf.data, off + 44)[0]
    return (rsrc3 >> 4) & 0xFF


def report_elf(label, elf):
    # All executable sections: kernels placed in their own section (CK_TILE_ICACHE_SECTION) are
    # outside .text.
    exec_bytes = sum(x["size"] for x in elf.sections if x["flags"] & 0x4 and x["type"] == 1)
    md = elf.amdgpu_metadata()
    kernels = {k.get(".symbol", "").removesuffix(".kd"): k for k in md.get("amdhsa.kernels", [])}
    syms = {s["name"]: s for s in elf.symbols()}
    funcs = []
    for s in elf.symbols():
        if s["type"] == 2 and s["size"] > 0:  # STT_FUNC
            if any(f["name"] == s["name"] for f in funcs):
                continue
            k = kernels.get(s["name"], {})
            funcs.append(dict(
                name=s["name"], bytes=s["size"], addr=s["value"],
                lines=(s["value"] + s["size"] + ICACHE_LINE - 1) // ICACHE_LINE
                - s["value"] // ICACHE_LINE,
                kernel=bool(k),
                vgpr=k.get(".vgpr_count"), sgpr=k.get(".sgpr_count"),
                agpr=k.get(".agpr_count"),
                lds=k.get(".group_segment_fixed_size"),
                scratch=k.get(".private_segment_fixed_size"),
                vgpr_spill=k.get(".vgpr_spill_count"), sgpr_spill=k.get(".sgpr_spill_count"),
                wavefront=k.get(".wavefront_size"),
                inst_pref=_inst_pref_lines(elf, syms, s["name"]) if k else None))
    funcs.sort(key=lambda f: -f["bytes"])
    return dict(code_object=label, target=md.get("amdhsa.target", ""),
                text_bytes=exec_bytes, functions=funcs)


_S_FUNC = re.compile(r"^\s*\.type\s+([^,\s]+),\s*@function")
_S_KEYS = {
    "codeLenInByte": "bytes", "NumVgprs": "vgpr", "NumSgprs": "sgpr", "TotalNumSgprs": "sgpr", "NumAgprs": "agpr",
    "ScratchSize": "scratch", "LDSByteSize": "lds", "Occupancy": "occupancy",
}


def report_asm(path):
    """Read the per-function '; codeLenInByte = N' style comments that llc prints after each
    function body (and the metadata block for spill counts)."""
    funcs, cur = [], None
    with open(path, errors="replace") as f:
        for line in f:
            m = _S_FUNC.match(line)
            if m:
                cur = dict(name=m.group(1), kernel=False)
                funcs.append(cur)
                continue
            if cur is None:
                continue
            m = re.match(r"^\s*;\s*(\w+):?\s*=?\s*(\d+)", line)
            if m and m.group(1) in _S_KEYS:
                cur[_S_KEYS[m.group(1)]] = int(m.group(2))
            if ".amdhsa_kernel " in line and line.split()[-1] == cur["name"]:
                cur["kernel"] = True
            m = re.match(r"^\s*\.amdhsa_kernel\s+(\S+)", line)
            if m:
                for g in funcs:
                    if g["name"] == m.group(1):
                        g["kernel"] = True
    # Spill counts live in the metadata YAML at the end.
    with open(path, errors="replace") as f:
        txt = f.read()
    for g in funcs:
        m = re.search(r"\.name:\s+" + re.escape(g["name"]) + r"\b(.*?)(?=\n  - |\Z)", txt, re.S)
        if m:
            for key, out in ((".vgpr_spill_count", "vgpr_spill"), (".sgpr_spill_count",
                                                                    "sgpr_spill")):
                mm = re.search(re.escape(key) + r":\s+(\d+)", m.group(1))
                if mm:
                    g[out] = int(mm.group(1))
            g["kernel"] = True
    funcs = [g for g in funcs if "bytes" in g]
    for g in funcs:
        g["lines"] = (g["bytes"] + ICACHE_LINE - 1) // ICACHE_LINE
    funcs.sort(key=lambda f: -f["bytes"])
    return dict(code_object=os.path.basename(path), target="(assembly)",
                text_bytes=sum(g["bytes"] for g in funcs), functions=funcs)


def _fmt(v):
    return "-" if v is None else str(v)


def print_report(rep, demangle, width, flt, short):
    funcs = [f for f in rep["functions"] if not flt or re.search(flt, f["name"]) or
             re.search(flt, demangle.get(f["name"], ""))]
    print("== %s  %s  code %d B (%.1f KiB, %d x %d B lines)" % (
        rep["code_object"], rep["target"], rep["text_bytes"], rep["text_bytes"] / 1024.0,
        (rep["text_bytes"] + ICACHE_LINE - 1) // ICACHE_LINE, ICACHE_LINE))
    print("%10s %6s %5s %5s %5s %7s %7s %6s %6s  %s" % (
        "bytes", "lines", "ipref", "vgpr", "sgpr", "lds", "scratch", "vspill", "sspill",
        "function"))
    for f in funcs:
        name = demangle.get(f["name"], f["name"])
        if short:
            name = shorten(name)
        if width and len(name) > width:
            name = name[: width - 3] + "..."
        print("%10d %6s %5s %5s %5s %7s %7s %6s %6s  %s%s" % (
            f["bytes"], _fmt(f.get("lines")), _fmt(f.get("inst_pref")), _fmt(f.get("vgpr")), _fmt(f.get("sgpr")),
            _fmt(f.get("lds")), _fmt(f.get("scratch")), _fmt(f.get("vgpr_spill")),
            _fmt(f.get("sgpr_spill")), "" if f.get("kernel") else "(func) ", name))


def shorten(name):
    """Collapse template argument lists deeper than one level: ck_tile names run to kilobytes."""
    out, depth = [], 0
    for ch in name:
        if ch == "<":
            depth += 1
            if depth == 1:
                out.append("<")
            elif depth == 2:
                out.append("...")
            continue
        if ch == ">":
            depth -= 1
            if depth == 0:
                out.append(">")
            continue
        if depth <= 1:
            out.append(ch)
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="executable / bundle / code object / .s")
    ap.add_argument("--arch", default="", help="only code objects whose target id contains this")
    ap.add_argument("--filter", default="", help="regex on (demangled) function names")
    ap.add_argument("--width", type=int, default=160, help="truncate names (0 = no limit)")
    ap.add_argument("--full-names", action="store_true", help="do not collapse template args")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    ap.add_argument("--extract", metavar="DIR", help="also write each code object to DIR")
    a = ap.parse_args()

    reports = []
    for path in a.inputs:
        if path.endswith(".s") or path.endswith(".S"):
            reports.append(report_asm(path))
            continue
        for label, elf in code_objects(path):
            if a.arch and a.arch not in label:
                continue
            if a.extract:
                os.makedirs(a.extract, exist_ok=True)
                fn = re.sub(r"[^A-Za-z0-9_.+-]", "_", label) + ".co"
                with open(os.path.join(a.extract, fn), "wb") as f:
                    f.write(elf.data)
            reports.append(report_elf(label, elf))
    if a.json:
        json.dump(reports, sys.stdout, indent=1)
        print()
        return 0
    names = sorted({f["name"] for r in reports for f in r["functions"]})
    dm = _demangle(names)
    for r in reports:
        print_report(r, dm, a.width, a.filter, not a.full_names)
    return 0 if reports else 1


if __name__ == "__main__":
    sys.exit(main())
