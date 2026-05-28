#!/usr/bin/env python3
"""okl: query hipBLASLt for the optimal kernel for a given GEMM problem.

Thin wrapper around `hipblaslt-bench --algo_method heuristic --print_kernel_info`.
The heuristic dispatch already encodes the shipped tuning, so we delegate to
hipBLASLt rather than reimplementing its solution-selection logic.

Output: JSON to stdout with the chosen solution's name, index, achieved
gflops, achieved GB/s, the problem echoed back, and the raw bench command
for reproducibility. Non-zero exit on bench failure.

Usage:
    okl.py -m 4096 -n 4096 -k 4096 --transa T --transb N \\
           --a-type bf16_r --b-type bf16_r --c-type bf16_r --d-type bf16_r \\
           --compute-type f32_r
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Byte width per element type (for raw allocation sizing in --package mode).
# Matches hipblaslt-bench's --a_type / --b_type / --c_type / --d_type values.
DTYPE_BYTES = {
    "f16_r": 2, "bf16_r": 2,
    "f32_r": 4, "i32_r": 4, "xf32_r": 4,
    "f64_r": 8,
    "i8_r": 1, "f8_r": 1, "bf8_r": 1,
    "f8_fnuz_r": 1, "bf8_fnuz_r": 1,
}

# clang-offload-bundler: used to read the .co's bundle target tuple at
# packaging time so the runner can preflight arch match against the device.
BUNDLER_CANDIDATES = [
    "/opt/rocm/lib/llvm/bin/clang-offload-bundler",
    "/opt/rocm-7.2.1/lib/llvm/bin/clang-offload-bundler",
    "/opt/rocm-6.4.3/lib/llvm/bin/clang-offload-bundler",
]

# llvm-readobj: used to read the .co's amdhsa.kernels[*].args metadata so we
# learn the kernel's true kernarg layout (offset / size / value_kind /
# value_type / name per slot). This replaces guessing offsets in the C++ runner.
READOBJ_CANDIDATES = [
    "/usr/bin/llvm-readobj",
    "/opt/rocm/lib/llvm/bin/llvm-readobj",
    "/opt/rocm-7.2.1/lib/llvm/bin/llvm-readobj",
    "/opt/rocm-6.4.3/lib/llvm/bin/llvm-readobj",
]


def find_bundler():
    p = shutil.which("clang-offload-bundler")
    if p:
        return p
    for c in BUNDLER_CANDIDATES:
        if Path(c).is_file() and os.access(c, os.X_OK):
            return c
    return None


def find_readobj():
    p = shutil.which("llvm-readobj")
    if p:
        return p
    for c in READOBJ_CANDIDATES:
        if Path(c).is_file() and os.access(c, os.X_OK):
            return c
    return None


def unbundle_co(co_path):
    """Unbundle the amdgcn slice of `co_path` into a temp ELF and return its path.

    Discovers the bundle target via `clang-offload-bundler --list`; returns None
    on any failure. Deterministic /tmp filename so repeated runs reuse it.
    """
    b = find_bundler()
    if b is None:
        return None
    try:
        listing = subprocess.run(
            [b, "--list", "--type=o", "--input", str(co_path)],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError,
            FileNotFoundError):
        return None
    amd = next((t.strip() for t in listing.stdout.splitlines()
                if "amdgcn-amd-amdhsa" in t), "")
    if not amd:
        return None
    elf_path = Path(f"/tmp/okl-unbundle-{co_path.stem}.elf")
    host_path = Path(f"/tmp/okl-unbundle-{co_path.stem}.host.o")
    try:
        subprocess.run(
            [b, "--unbundle", "--type=o", "--input", str(co_path),
             "--targets", f"{amd},host-x86_64-unknown-linux-gnu-",
             "--output", str(elf_path), "--output", str(host_path)],
            capture_output=True, text=True, timeout=20, check=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError,
            FileNotFoundError):
        return None
    return elf_path if elf_path.is_file() else None


def parse_kernel_args(elf_path, kernel_symbol):
    """Parse `amdhsa.kernels[*].args` for the given symbol from an ELF.

    Returns a list of dicts (one per kernarg slot, in declaration order)
    with keys: name, offset, size, value_kind, value_type, address_space.
    Also returns kernarg_segment_size for the kernel.

    Returns (None, None) on any tool / parse failure (caller can fall back).
    The parser is hand-rolled (no PyYAML dep) because llvm-readobj's note
    output is a tightly constrained subset of YAML: each `.foo: value` line
    starts with whitespace + a dot, and structure is by indentation only.
    The `.kd` suffix is stripped from the symbol when matching.
    """
    ro = find_readobj()
    if ro is None:
        return None, None
    try:
        out = subprocess.run(
            [ro, "--notes", str(elf_path)],
            capture_output=True, text=True, timeout=30, check=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError,
            FileNotFoundError):
        return None, None

    # We scan kernel by kernel; a kernel block begins with `- .args:` and ends
    # at the .symbol line which carries the kernel name (with `.kd` suffix).
    # The block contains: args (list of dicts), .kernarg_segment_size, .name,
    # .symbol. We accumulate the current kernel's args and stash them keyed by
    # .name (or .symbol minus .kd) when we hit the trailing fields.
    lines = out.stdout.splitlines()
    kernels = {}  # name -> (args_list, kernarg_size)
    cur_args = []
    cur_arg = None
    cur_kernarg_size = None
    cur_name = None

    def flush_arg():
        nonlocal cur_arg
        if cur_arg is not None:
            cur_args.append(cur_arg)
            cur_arg = None

    arg_field_keys = {"name", "offset", "size", "value_kind",
                      "value_type", "address_space"}
    for raw in lines:
        s = raw.lstrip()
        # An `- .args:` line introduces the args sublist (or marks
        # subsequent kernel-level structure); not the start of an arg.
        if s.startswith("- .args:"):
            flush_arg()
            cur_arg = None
            continue
        # An `- .<key>:` line where <key> is an arg field starts a new arg.
        m_dash = re.match(r"- \.(\w+):", s)
        if m_dash and m_dash.group(1) in arg_field_keys:
            flush_arg()
            cur_arg = {}
            s2 = s[2:]
        else:
            s2 = s
        # Field: ".key: value"
        m = re.match(r"\.(\w+):\s*(.*?)\s*$", s2)
        if m and cur_arg is not None and m.group(1) in arg_field_keys:
            k = m.group(1)
            v = m.group(2)
            if k in ("offset", "size"):
                try:
                    cur_arg[k] = int(v)
                except ValueError:
                    pass
            else:
                cur_arg[k] = v
            continue
        # Kernel-level keys (outside args list).
        m = re.match(r"\s*\.kernarg_segment_size:\s*(\d+)", raw)
        if m:
            flush_arg()
            cur_kernarg_size = int(m.group(1))
            continue
        m = re.match(r"\s*\.name:\s*(\S+)", raw)
        if m and cur_arg is None:
            # This is the kernel's .name (no current arg pending).
            cur_name = m.group(1)
            continue
        m = re.match(r"\s*\.symbol:\s*(\S+)", raw)
        if m:
            flush_arg()
            sym = m.group(1)
            if sym.endswith(".kd"):
                sym = sym[:-3]
            if cur_args:
                kernels[sym] = (cur_args, cur_kernarg_size)
                if cur_name and cur_name != sym:
                    kernels[cur_name] = (cur_args, cur_kernarg_size)
            cur_args = []
            cur_name = None
            cur_kernarg_size = None
            continue

    if kernel_symbol in kernels:
        args, ks = kernels[kernel_symbol]
        return args, ks
    return None, None


BENCH_CANDIDATES = [
    "/home/alvasile/rocm-libraries/projects/hipblaslt/build/release/clients/hipblaslt-bench",
    "/opt/rocm/bin/hipblaslt-bench",
]

LIBPATH_CANDIDATES = [
    "/opt/rocm/lib/hipblaslt/library",
    "/opt/rocm-7.2.1/lib/hipblaslt/library",
    "/home/alvasile/rocm-libraries/projects/hipblaslt/build/release/Tensile/library",
]

SOL_NAME_PREFIX = "--Solution name:"
SOL_IDX_PREFIX = "--Solution index:"
HEADER_MARKER = "hipblaslt-Gflops"


def find_bench(override):
    if override:
        return override
    p = shutil.which("hipblaslt-bench")
    if p:
        return p
    for c in BENCH_CANDIDATES:
        if Path(c).is_file() and os.access(c, os.X_OK):
            return c
    sys.exit("error: hipblaslt-bench not found. Pass --bench /path/to/hipblaslt-bench.")


def find_libpath(override):
    if override:
        return override
    env = os.environ.get("HIPBLASLT_TENSILE_LIBPATH")
    if env:
        return env
    for c in LIBPATH_CANDIDATES:
        if Path(c).is_dir() and any(Path(c).glob("TensileLibrary_lazy_*.dat")):
            return c
    return None  # let bench error out itself


def parse_output(stdout):
    """Pull solution name/index and one timing row out of bench stdout."""
    sol_name, sol_idx = None, None
    header_fields, value_fields = None, None
    lines = stdout.splitlines()
    for i, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith(SOL_NAME_PREFIX):
            sol_name = line[len(SOL_NAME_PREFIX):].strip()
        elif line.startswith(SOL_IDX_PREFIX):
            try:
                sol_idx = int(line[len(SOL_IDX_PREFIX):].strip())
            except ValueError:
                pass
        elif HEADER_MARKER in line and "," in line:
            after_colon = line.split(":", 1)[1] if line.startswith("[") and ":" in line else line
            header_fields = [f.strip() for f in after_colon.split(",")]
            for j in range(i + 1, len(lines)):
                cand = lines[j].strip()
                if cand and "," in cand and not cand.startswith("["):
                    value_fields = [f.strip() for f in cand.split(",")]
                    break

    timing = {}
    if header_fields and value_fields and len(header_fields) == len(value_fields):
        row = dict(zip(header_fields, value_fields))
        for src, dst in (
            ("hipblaslt-Gflops", "gflops"),
            ("hipblaslt-GB/s", "gb_per_s"),
            ("us", "microseconds"),
        ):
            if src in row:
                try:
                    timing[dst] = float(row[src])
                except ValueError:
                    timing[dst] = row[src]
    return sol_name, sol_idx, timing


def parse_dump(stdout):
    """Extract everything okl_run.cpp needs from a TENSILE_DB=0xF0 dump.

    Returns a dict with: co_file, kernel_symbol, internal_args, internal_args1,
    workgroup_size_threads (from `l(X, ...) x g(...)` line), grid_workgroups,
    sizes (M,N,batch,K), strides (8 u32s).
    Returns None if any required field is absent.
    """
    info = {}
    # `loaded code object /path/to/...co` (the actual shard, not the placeholder)
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("loaded code object "):
            path = line[len("loaded code object "):].strip()
            # Skip the very first .hsaco preload; the GEMM shard is loaded later.
            if path.endswith(".co"):
                info["co_file"] = path
        elif line.startswith("Kernel "):
            info["kernel_symbol"] = line[len("Kernel "):].strip()
        elif line.startswith("l(") and "x g(" in line and "=" in line:
            # e.g. "l(256, 1, 1) x g(256, 1, 1) = (65536, 1, 1)"
            m = re.match(
                r"l\(\s*(\d+)\s*,\s*\d+\s*,\s*\d+\s*\)\s*x\s*g\(\s*(\d+)",
                line,
            )
            if m:
                info["workgroup_size_threads"] = int(m.group(1))
                info["grid_workgroups"] = int(m.group(2))
        else:
            # kernarg dump lines like: "[4..7] internalArgs:  01 00 08 20 (537395201)"
            mm = re.match(
                r"\[(\d+)\.\.(\d+)\]\s+(\w+):\s+([0-9a-fA-F\s]+?)\s+\(",
                line,
            )
            if mm:
                start = int(mm.group(1))
                end = int(mm.group(2))
                name = mm.group(3)
                bytes_str = mm.group(4).replace(" ", "")
                raw = bytes.fromhex(bytes_str)
                width = end - start + 1
                if width == 4 and len(raw) == 4:
                    val = int.from_bytes(raw, "little")
                    info.setdefault("kernarg_u32", {})[start] = (name, val)
                elif width == 8 and len(raw) == 8:
                    val = int.from_bytes(raw, "little")
                    info.setdefault("kernarg_u64", {})[start] = (name, val)

    required = ("co_file", "kernel_symbol", "workgroup_size_threads",
                "grid_workgroups", "kernarg_u32")
    if not all(k in info for k in required):
        return None
    info.setdefault("kernarg_u64", {})
    return info


def parse_macro_tile(kernel_symbol):
    """Pull MT0, MT1 out of the kernel name's `_MT<a>x<b>x<c>_` token."""
    m = re.search(r"_MT(\d+)x(\d+)x\d+_", kernel_symbol)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


# All `value_kind: global_buffer` slots the runtime might emit are tensor
# pointers we have to hipMalloc. Map ELF arg name -> logical buffer role.
# The role is used to (a) name the alloc in the conf, (b) decide its size,
# (c) decide its init pattern in the C++ runner.
#
# Anything in this map is treated as a known buffer; anything else with
# value_kind=global_buffer is unknown and the packager fails loud (better than
# silently allocating a zero buffer for an unanticipated slot).
KNOWN_BUFFER_ROLES = {
    "D": "D", "C": "C", "A": "A", "B": "B",
    "MXSA": "MXSA", "MXSB": "MXSB", "MetaData": "MetaData",
    # Stream-K workspace + flags. AddressWS is GSU partial sums, AddressFlags
    # is the per-tile completion bitmap.
    "AddressWS": "WS", "AddressFlags": "Flags",
    # Bias / aux / extras (Tensile names them with the Address prefix in some
    # codepaths and bare in others, depending on Signature.py path).
    "bias": "bias", "AddressBias": "bias",
    "e": "E", "E": "E",
    "AddressScaleA": "scaleA", "AddressScaleB": "scaleB",
    "AddressScaleC": "scaleC", "AddressScaleD": "scaleD",
    "AddressScaleAlphaVec": "scaleAlphaVec",
    # GSU multi-buffer atomic synchronization path.
    "Synchronizer": "Synchronizer", "AmaxSync": "AmaxSync",
    # Alternate D output (gradient path).
    "dstD": "dstD",
}


def buffer_alloc_size(role, args_namespace, sd0, sc0, sa0, sb0,
                      bytes_a, bytes_b, bytes_c, bytes_d):
    """Bytes to hipMalloc for a buffer role. Generous when in doubt.

    For D/C/A/B we use the strides from the dump (they're guaranteed >= what
    the kernel reads). Everything else is bias / scales / stream-k workspace,
    sized from problem dims. Returns 0 for roles we cannot size safely - the
    caller will surface that as an unsupported-feature error.
    """
    M, N, K, B = (args_namespace.m, args_namespace.n,
                  args_namespace.k, args_namespace.batch)
    if role == "A":
        return max(sa0 * M, sa0 * K) * B * bytes_a
    if role == "B":
        return max(sb0 * N, sb0 * K) * B * bytes_b
    if role == "C":
        return sc0 * N * B * bytes_c
    if role == "D":
        return sd0 * N * B * bytes_d
    if role == "bias":
        # Per-row bias of M elements; assume 4 bytes/elem (worst-case f32).
        return max(M, N) * 4
    if role == "dstD":
        # Alternate D buffer (gradient path) - same size as D.
        return sd0 * N * B * bytes_d
    if role in ("scaleA", "scaleB", "scaleC", "scaleD", "scaleAlphaVec"):
        # Per-row/col scale vector; conservatively MxN*4.
        return max(M, N, K) * 4
    if role == "E":
        # Auxiliary output, problem-sized.
        return M * N * B * 4
    if role in ("WS", "Synchronizer", "AmaxSync"):
        # Stream-K / GSU workspace. We can't size this precisely without the
        # solution metadata; allocate a generous slab and let the kernel use
        # what it needs.
        return 64 * 1024 * 1024  # 64 MiB
    if role == "Flags":
        # Per-tile completion flags, one byte per tile.
        return max(1024, (M // 32) * (N // 32) * 4)
    if role in ("MXSA", "MXSB", "MetaData"):
        # MX block scales / sparsity metadata. Not tested; allocate a slab.
        return max(M, N) * K
    return 0


# By-value field name -> source instructions. Source can be:
#   "dump"      - value comes verbatim from the TENSILE_DB dump's u32 at offset
#   "problem.M" - one of {M, N, K, batch} from the okl.py CLI
#   "alpha"/"beta" - from the CLI (typed by the ELF arg's value_type)
# Anything not listed defaults to "dump"; if the dump has no value at that
# offset we fall back to 0 with a warning.
BY_VALUE_FROM_PROBLEM = {
    # Universal-args sizes ordering: SizesFree0..2 = (M, N, batch),
    # SizesSum0 = K. See KernelArguments dump confirming this for our example
    # kernel (size_0..3 print as M, N, batch, K).
    "SizesFree0": "problem.M",
    "SizesFree1": "problem.N",
    "SizesFree2": "problem.batch",
    "SizesSum0":  "problem.K",
    "alpha": "alpha",
    "beta":  "beta",
}


def build_slots(elf_args, dump_info, args_namespace,
                sd0, sd1, sc0, sc1, sa0, sa1, sb0, sb1,
                bytes_a, bytes_b, bytes_c, bytes_d):
    """Walk the ELF args metadata and emit one slot record per kernarg field.

    Each record knows its layout (offset, size) and how to obtain its value
    at run time (source = const|alloc + the typed value). For value_kind =
    global_buffer slots, we also declare the backing buffer (size + init).
    """
    slots = []
    buffers = {}  # role -> bytes
    inits = {}    # role -> "zero" | "poison"
    unknown = []  # list of (name, kind) tuples we couldn't handle

    dump_u32 = dump_info.get("kernarg_u32", {})

    for a in elf_args:
        name = a.get("name", "?")
        off = a.get("offset")
        sz = a.get("size")
        kind = a.get("value_kind", "?")
        vtype = a.get("value_type", "")

        slot = {"name": name, "offset": off, "size": sz}

        if kind == "global_buffer":
            role = KNOWN_BUFFER_ROLES.get(name)
            if role is None:
                unknown.append((name, kind))
                continue
            slot["source"] = "buffer"
            slot["buffer"] = role
            if role not in buffers:
                buffers[role] = buffer_alloc_size(
                    role, args_namespace, sd0, sc0, sa0, sb0,
                    bytes_a, bytes_b, bytes_c, bytes_d)
                # D gets poison init so we can detect kernel didn't write.
                inits[role] = "poison" if role == "D" else "zero"
            slots.append(slot)
            continue

        if kind != "by_value":
            unknown.append((name, kind))
            continue

        # by_value: figure out where the value comes from.
        src = BY_VALUE_FROM_PROBLEM.get(name, "dump")
        slot["ctype"] = vtype or "u32"

        if src.startswith("problem."):
            field = src.split(".", 1)[1]
            val = {"M": args_namespace.m, "N": args_namespace.n,
                   "K": args_namespace.k,
                   "batch": args_namespace.batch}[field]
            slot["source"] = "const"
            slot["value"] = val
        elif src == "alpha":
            slot["source"] = "const"
            slot["value"] = (args_namespace.alpha
                             if args_namespace.alpha is not None else 1.0)
        elif src == "beta":
            slot["source"] = "const"
            slot["value"] = (args_namespace.beta
                             if args_namespace.beta is not None else 0.0)
        else:
            # Dump-derived: pull the raw 4-byte little-endian pattern from
            # the TENSILE_DB dump's u32 reading at this offset. We keep the
            # raw bit pattern (rather than re-encoding via the ctype) so
            # bit-exact replay works even when ctype is f32 / pkf16 and the
            # printed integer is something like 0x7F800000 (+inf as f32).
            if off in dump_u32:
                _name, val = dump_u32[off]
                slot["source"] = "const"
                slot["value"] = val
                # Force raw u32 encoding to preserve the bit pattern.
                slot["raw_u32"] = True
            elif sz and sz <= 4:
                slot["source"] = "const"
                slot["value"] = 0
                slot["dump_missing"] = True
            else:
                unknown.append((name, "by_value:nodump"))
                continue
        slots.append(slot)

    return slots, buffers, inits, unknown


def format_slot_line(s):
    """Render one slot record as a single conf line."""
    fields = [f"offset={s['offset']}", f"size={s['size']}"]
    if "buffer" in s:
        fields.append("kind=buffer")
        fields.append(f"buffer={s['buffer']}")
    else:
        fields.append("kind=value")
        ctype = s.get("ctype", "u32")
        v = s["value"]
        if s.get("raw_u32"):
            # Bit-exact replay of a dump value: always write 4 bytes as u32,
            # regardless of what the kernel reads them back as.
            fields.append("ctype=u32")
            fields.append(f"value=0x{int(v):x}")
        elif ctype in ("f32", "f64"):
            fields.append(f"ctype={ctype}")
            fields.append(f"value={v!r}")
        elif isinstance(v, int):
            fields.append(f"ctype={ctype}")
            fields.append(f"value=0x{v:x}")
        else:
            fields.append(f"ctype={ctype}")
            fields.append(f"value={v}")
    # `name` is for diagnostics + numWG lookup; replace whitespace with `_`
    # so the simple whitespace tokenizer in okl_run.cpp stays simple.
    fields.append("name=" + re.sub(r"\s+", "_", s["name"]))
    if s.get("dump_missing"):
        fields.append("note=dump_missing_defaulted_to_zero")
    return "slot = " + " ".join(fields)


def write_package(out_dir, args, dump_info):
    """Copy the .co and emit kernel.conf for the C++ runner."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    src_co = Path(dump_info["co_file"])
    dst_co = out / "kernel.co"
    shutil.copyfile(src_co, dst_co)

    sym = dump_info["kernel_symbol"]

    # Parse the .co's amdhsa.kernels[*].args metadata for this kernel. This
    # is the ground-truth kernarg layout the kernel reads; everything in the
    # conf is derived from it.
    elf_path = unbundle_co(dst_co)
    elf_args = None
    elf_kernarg_size = None
    if elf_path is not None:
        elf_args, elf_kernarg_size = parse_kernel_args(elf_path, sym)
    if elf_args is None:
        sys.exit(
            f"error: could not parse amdhsa.kernels metadata for symbol\n"
            f"  {sym}\n"
            f"from .co\n"
            f"  {src_co}\n"
            f"Tried bundler={find_bundler()} readobj={find_readobj()}.\n"
            f"Install llvm-readobj or pass a known-good .co.")

    # Strides from the dump (true values the runtime would pass for THIS
    # problem). We use them both for stride slots and for alloc sizing.
    dump_u32 = dump_info["kernarg_u32"]
    def dget(off, default=0):
        return dump_u32[off][1] if off in dump_u32 else default
    sd0, sd1 = dget(64), dget(68)
    sc0, sc1 = dget(72), dget(76)
    sa0, sa1 = dget(80), dget(84)
    sb0, sb1 = dget(88), dget(92)

    bytes_a_elem = DTYPE_BYTES.get(args.a_type, 2)
    bytes_b_elem = DTYPE_BYTES.get(args.b_type, 2)
    bytes_c_elem = DTYPE_BYTES.get(args.c_type, 2)
    bytes_d_elem = DTYPE_BYTES.get(args.d_type, 2)

    slots, buffers, inits, unknown = build_slots(
        elf_args, dump_info, args,
        sd0, sd1, sc0, sc1, sa0, sa1, sb0, sb1,
        bytes_a_elem, bytes_b_elem, bytes_c_elem, bytes_d_elem)

    if unknown:
        msg = "\n  ".join(f"{n} (kind={k})" for n, k in unknown)
        sys.exit(
            "error: kernel uses kernarg slots this packager doesn't know how "
            "to feed:\n  " + msg + "\n"
            "Extend KNOWN_BUFFER_ROLES / BY_VALUE_FROM_PROBLEM in okl.py to "
            "handle them, or pick a kernel that uses only the universal-args "
            "core (D, C, A, B, sizes, strides, alpha, beta).")

    # Sanity: kernarg size from ELF must match what the slot list covers.
    kernarg_size = elf_kernarg_size
    last_byte = max(s["offset"] + s["size"] for s in slots) if slots else 0
    if kernarg_size is None:
        kernarg_size = last_byte
    elif last_byte > kernarg_size:
        sys.exit(f"error: slot layout overruns ELF kernarg_segment_size "
                 f"({last_byte} > {kernarg_size})")

    mt0, mt1 = parse_macro_tile(sym)
    if mt0 is None:
        # Not fatal - numWG already came from the dump; MT is purely
        # informational at this point. But emit zero rather than crashing.
        mt0, mt1 = 0, 0

    # Render the conf.
    lines = [
        "# okl-packaged kernel config",
        "# Generated by okl.py --package for one (solution, problem) pair.",
        "# Heuristic-chosen kernel for the problem below on the recorded gpu/library.",
        "",
        f"co_file                 = kernel.co",
        f"kernel_symbol           = {sym}",
        "",
        "# Kernel layout (ground truth from amdhsa.kernels[*].args in the .co).",
        f"kernarg_size            = {kernarg_size}",
        f"workgroup_size_threads  = {dump_info['workgroup_size_threads']}",
        "",
        "# Problem (echoed for diagnostics; values are baked into the slot list).",
        f"m                       = {args.m}",
        f"n                       = {args.n}",
        f"k                       = {args.k}",
        f"batch                   = {args.batch}",
        f"macro_tile_0            = {mt0}",
        f"macro_tile_1            = {mt1}",
        "",
        "# Buffers to allocate. Each maps to one or more `kind=buffer` slots.",
        "# init: 'zero' fills with 0; 'poison' fills with 0xee so the runner can",
        "# verify the kernel actually wrote to the buffer.",
    ]
    # Deterministic buffer ordering: D, C, A, B first (standard), then alpha.
    role_order = ["D", "C", "A", "B"]
    other_roles = sorted(r for r in buffers if r not in role_order)
    for role in role_order + other_roles:
        if role not in buffers:
            continue
        lines.append(
            f"buffer = name={role} bytes={buffers[role]} init={inits[role]}")
    lines.append("")
    lines.append("# Kernarg slot list (ordered by offset). The runner walks this,")
    lines.append("# writing each slot into the kernarg buffer at its declared offset.")
    for s in sorted(slots, key=lambda x: x["offset"]):
        lines.append(format_slot_line(s))
    lines.append("")
    (out / "kernel.conf").write_text("\n".join(lines))

    # Echo the legacy fields packagers used to grep for, into the returned
    # JSON, even though they no longer drive the runner.
    internal_args  = dget(4)
    internal_args1 = dget(8)
    numwg_val      = dget(12)
    return {
        "package_dir":    str(out.resolve()),
        "kernel_conf":    str((out / "kernel.conf").resolve()),
        "kernel_co":      str(dst_co.resolve()),
        "kernel_co_src":  str(src_co.resolve()),
        "kernel_symbol":  sym,
        "internal_args":  f"0x{internal_args:08x}",
        "internal_args1": f"0x{internal_args1:08x}",
        "numWG":          numwg_val,
        "macro_tile_0":   mt0,
        "macro_tile_1":   mt1,
        "workgroup_size_threads": dump_info["workgroup_size_threads"],
        "grid_workgroups": dump_info["grid_workgroups"],
        "kernarg_size":   kernarg_size,
        "num_slots":      len(slots),
        "num_buffers":    len(buffers),
    }


def build_bench_args(a):
    args = [
        "-m", str(a.m), "-n", str(a.n), "-k", str(a.k),
        "--batch_count", str(a.batch),
        "--transA", a.transa, "--transB", a.transb,
        "--a_type", a.a_type, "--b_type", a.b_type,
        "--c_type", a.c_type, "--d_type", a.d_type,
        "--compute_type", a.compute_type,
        "--algo_method", "heuristic",
        "--requested_solution", "1",
        "--print_kernel_info",
        "--iters", str(a.iters),
        "--cold_iters", str(a.cold_iters),
    ]
    if a.bias_vector:
        args.append("--bias_vector")
    if a.bias_type:
        args += ["--bias_type", a.bias_type]
    if a.activation_type and a.activation_type != "none":
        args += ["--activation_type", a.activation_type]
    if a.alpha is not None:
        args += ["--alpha", str(a.alpha)]
    if a.beta is not None:
        args += ["--beta", str(a.beta)]
    if a.extra:
        args.extend(a.extra)
    return args


def main():
    p = argparse.ArgumentParser(
        description="Query hipBLASLt for the optimal kernel for one GEMM shape.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Anything after `--` is forwarded verbatim to hipblaslt-bench.",
    )
    p.add_argument("-m", type=int, required=True, help="M (rows of op(A) / D)")
    p.add_argument("-n", type=int, required=True, help="N (cols of op(B) / D)")
    p.add_argument("-k", type=int, required=True, help="K (inner dimension)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--transa", default="N", choices=("N", "T"))
    p.add_argument("--transb", default="N", choices=("N", "T"))
    p.add_argument("--a-type", default="f16_r")
    p.add_argument("--b-type", default="f16_r")
    p.add_argument("--c-type", default="f16_r")
    p.add_argument("--d-type", default="f16_r")
    p.add_argument("--compute-type", default="f32_r")
    p.add_argument("--alpha", type=float)
    p.add_argument("--beta", type=float)
    p.add_argument("--bias-vector", action="store_true")
    p.add_argument("--bias-type")
    p.add_argument("--activation-type", default="none")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--cold-iters", type=int, default=2)
    p.add_argument("--bench", help="Path to hipblaslt-bench. Default: $PATH, then known build/install locations.")
    p.add_argument("--libpath", help="HIPBLASLT_TENSILE_LIBPATH override (dir with TensileLibrary_lazy_<arch>.dat).")
    p.add_argument("--timeout", type=int, default=120, help="Seconds before killing bench.")
    p.add_argument("--keep-stdout", action="store_true", help="Include the full bench stdout in the JSON.")
    p.add_argument("--package", metavar="OUT_DIR",
                   help="After finding the winner, re-run bench with TENSILE_DB=0xF0 to capture kernarg + grid; "
                        "copy the .co to OUT_DIR/kernel.co and write OUT_DIR/kernel.conf for okl_run.")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="Pass-through to hipblaslt-bench.")
    a = p.parse_args()

    bench = find_bench(a.bench)
    libpath = find_libpath(a.libpath)
    bench_args = build_bench_args(a)

    env = os.environ.copy()
    if libpath:
        env["HIPBLASLT_TENSILE_LIBPATH"] = libpath

    try:
        proc = subprocess.run(
            [bench, *bench_args],
            env=env,
            capture_output=True,
            text=True,
            timeout=a.timeout,
        )
    except subprocess.TimeoutExpired:
        sys.exit(f"error: hipblaslt-bench timed out after {a.timeout}s")
    except FileNotFoundError as e:
        sys.exit(f"error: cannot execute {bench}: {e}")

    sol_name, sol_idx, timing = parse_output(proc.stdout)

    out = {
        "problem": {
            "m": a.m, "n": a.n, "k": a.k, "batch": a.batch,
            "transA": a.transa, "transB": a.transb,
            "a_type": a.a_type, "b_type": a.b_type,
            "c_type": a.c_type, "d_type": a.d_type,
            "compute_type": a.compute_type,
        },
        "solution_name": sol_name,
        "solution_index": sol_idx,
        "timing": timing,
        "libpath": libpath,
        "bench": bench,
        "bench_args": bench_args,
        "bench_returncode": proc.returncode,
    }
    if proc.returncode != 0 or sol_name is None:
        out["bench_stderr_tail"] = proc.stderr[-2000:]
        out["bench_stdout_tail"] = proc.stdout[-2000:]
    if a.keep_stdout:
        out["bench_stdout"] = proc.stdout

    # --- packaging path: re-run with TENSILE_DB=0xF0 to capture kernarg + grid ---
    if a.package and proc.returncode == 0 and sol_name is not None:
        # Re-run with the SAME problem flags, but limit to 1 iter (we only need
        # the dump). Force --algo_method index --solution_index so we package
        # the kernel that okl.py just identified, not whatever the heuristic
        # picks again (usually the same, but be deterministic).
        pkg_args = list(bench_args)
        # Replace algo_method heuristic -> index, requested_solution -> solution_index
        def replace_flag(args, flag, new_flag, new_val):
            for i, x in enumerate(args):
                if x == flag and i + 1 < len(args):
                    args[i] = new_flag
                    args[i + 1] = new_val
                    return
            args.extend([new_flag, new_val])
        replace_flag(pkg_args, "--algo_method", "--algo_method", "index")
        replace_flag(pkg_args, "--requested_solution",
                     "--solution_index", str(sol_idx))
        # Force a minimal timed run (one iter is enough for the dump).
        for i, x in enumerate(pkg_args):
            if x == "--iters" and i + 1 < len(pkg_args):
                pkg_args[i + 1] = "1"
            elif x == "--cold_iters" and i + 1 < len(pkg_args):
                pkg_args[i + 1] = "0"

        pkg_env = dict(env)
        pkg_env["TENSILE_DB"] = "0xF0"

        try:
            pkg_proc = subprocess.run(
                [bench, *pkg_args],
                env=pkg_env,
                capture_output=True,
                text=True,
                timeout=a.timeout,
            )
        except subprocess.TimeoutExpired:
            sys.exit(f"error: package re-run timed out after {a.timeout}s")

        dump_info = parse_dump(pkg_proc.stdout)
        if dump_info is None:
            out["package"] = {
                "error": "could not parse TENSILE_DB=0xF0 dump",
                "stdout_tail": pkg_proc.stdout[-2000:],
                "stderr_tail": pkg_proc.stderr[-2000:],
            }
        else:
            out["package"] = write_package(a.package, a, dump_info)

    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")

    if proc.returncode != 0 or sol_name is None:
        sys.exit(1)
    if a.package and "package" in out and "error" in out["package"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
