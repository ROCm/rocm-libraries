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
                # Pack little-endian: first byte printed is lowest address
                raw = bytes.fromhex(bytes_str)
                if end - start + 1 == 4 and len(raw) == 4:
                    val = int.from_bytes(raw, "little")
                    info.setdefault("kernarg_u32", {})[start] = (name, val)

    required = ("co_file", "kernel_symbol", "workgroup_size_threads",
                "grid_workgroups", "kernarg_u32")
    if not all(k in info for k in required):
        return None
    return info


def parse_macro_tile(kernel_symbol):
    """Pull MT0, MT1 out of the kernel name's `_MT<a>x<b>x<c>_` token."""
    m = re.search(r"_MT(\d+)x(\d+)x\d+_", kernel_symbol)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def write_package(out_dir, args, dump_info):
    """Copy the .co and emit kernel.conf for the C++ runner."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    src_co = Path(dump_info["co_file"])
    dst_co = out / "kernel.co"
    shutil.copyfile(src_co, dst_co)

    sym = dump_info["kernel_symbol"]
    mt0, mt1 = parse_macro_tile(sym)
    if mt0 is None:
        sys.exit("error: could not parse MT0/MT1 from kernel name: " + sym)

    kernarg = dump_info["kernarg_u32"]
    def kget(offset, expected_name=None):
        if offset not in kernarg:
            sys.exit(f"error: dump missing kernarg field at offset {offset}")
        name, val = kernarg[offset]
        if expected_name and expected_name not in name:
            print(f"warn: kernarg offset {offset} is '{name}', expected '{expected_name}'",
                  file=sys.stderr)
        return val

    internal_args  = kget(4,  "internalArgs")
    internal_args1 = kget(8,  "internalArgs1")
    # Total kernarg size: last printed offset + 4. Default 104 if dump truncated.
    max_off = max(kernarg.keys())
    kernarg_size = max_off + 4

    # Tensor byte sizes. For column-major Tensile dispatch the convention is:
    # T(A) -> KxM with lda=K; N(A) -> MxK with lda=M.
    # T(B) -> KxN with ldb=K; N(B) -> NxK with ldb=N.
    # D, C -> MxN with ld=M.
    bytes_a_elem = DTYPE_BYTES.get(args.a_type, 2)
    bytes_b_elem = DTYPE_BYTES.get(args.b_type, 2)
    bytes_c_elem = DTYPE_BYTES.get(args.c_type, 2)
    bytes_d_elem = DTYPE_BYTES.get(args.d_type, 2)
    lda = args.k if args.transa == "T" else args.m
    ldb = args.k if args.transb == "N" else args.n  # note: NN/NT/TN/TT vary
    # hipblaslt-bench's convention from observed dumps:
    #   TN bf16 512^3: strideA1=K=512, strideB1=K=512, strideD1=M=512.
    # That's what we replay; we don't reinterpret it.
    # Allocate enough bytes for each: max stride * other-dim * elem.
    size_a = lda * args.m * bytes_a_elem if args.transa == "T" else lda * args.k * bytes_a_elem
    size_b = ldb * args.n * bytes_b_elem if args.transb == "N" else ldb * args.k * bytes_b_elem
    # Safer: just take stride * batch * dim from the dump itself.
    sd0 = kget(64, "strideD")
    sc0 = kget(72, "strideC")
    sa0 = kget(80, "strideA")
    sb0 = kget(88, "strideB")
    sd1 = kget(68); sc1 = kget(76); sa1 = kget(84); sb1 = kget(92)
    # Recompute alloc sizes from strides + dims so we always cover what the
    # kernel reads/writes:
    #   bytes(A) = sa0 * (max free dim) * elem; for batched, multiply by batch
    # Conservative: use max of stride*M, stride*K, stride*N.
    size_a = max(sa0 * args.m, sa0 * args.k) * args.batch * bytes_a_elem
    size_b = max(sb0 * args.n, sb0 * args.k) * args.batch * bytes_b_elem
    size_c = sc0 * args.n * args.batch * bytes_c_elem
    size_d = sd0 * args.n * args.batch * bytes_d_elem

    alpha = args.alpha if args.alpha is not None else 1.0
    beta  = args.beta if args.beta is not None else 0.0

    conf = f"""# okl-packaged kernel config
# Generated by okl.py --package for one (solution, problem) pair.
# Heuristic-chosen kernel for the problem below on the recorded gpu/library.

co_file                 = kernel.co
kernel_symbol           = {sym}

# From TENSILE_DB=0x40 dump (bit-packed; treat as opaque)
internal_args           = 0x{internal_args:08x}
internal_args1          = 0x{internal_args1:08x}

# From kernel name `_MT<MT0>x<MT1>x<DepthU>_`
macro_tile_0            = {mt0}
macro_tile_1            = {mt1}

# From kernarg-dump launch-dims line `l(N, 1, 1) x g(...)`
workgroup_size_threads  = {dump_info['workgroup_size_threads']}
kernarg_size            = {kernarg_size}

# Problem
m                       = {args.m}
n                       = {args.n}
k                       = {args.k}
batch                   = {args.batch}

# Allocation sizes (raw bytes)
size_a_bytes            = {size_a}
size_b_bytes            = {size_b}
size_c_bytes            = {size_c}
size_d_bytes            = {size_d}

# Strides (from dump, kernarg offsets [64..95])
stride_d_0              = {sd0}
stride_d_1              = {sd1}
stride_c_0              = {sc0}
stride_c_1              = {sc1}
stride_a_0              = {sa0}
stride_a_1              = {sa1}
stride_b_0              = {sb0}
stride_b_1              = {sb1}

# Scalars (4-byte f32)
alpha                   = {alpha}
beta                    = {beta}
"""
    (out / "kernel.conf").write_text(conf)
    return {
        "package_dir":    str(out.resolve()),
        "kernel_conf":    str((out / "kernel.conf").resolve()),
        "kernel_co":      str(dst_co.resolve()),
        "kernel_co_src":  str(src_co.resolve()),
        "kernel_symbol":  sym,
        "internal_args":  f"0x{internal_args:08x}",
        "internal_args1": f"0x{internal_args1:08x}",
        "macro_tile_0":   mt0,
        "macro_tile_1":   mt1,
        "workgroup_size_threads": dump_info["workgroup_size_threads"],
        "grid_workgroups": dump_info["grid_workgroups"],
        "kernarg_size":   kernarg_size,
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
