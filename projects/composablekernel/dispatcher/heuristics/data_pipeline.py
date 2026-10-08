#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Data pipeline for CK Tile heuristics.

Parses benchmark logs and structured JSON into a canonical parquet dataset.
Supports:
  - Streaming log format (Shape N: headers + inline JSON) from ck_tile profiling runs
  - Structured JSON from generate_benchmark_data.py
  - Direct parquet passthrough
"""

import json
import re
import subprocess
import sys
import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd

# LDS capacity is generated from the real arch specs and is not something
# rocminfo reports, so it has to come from the table rather than the probe.
# The sys.path hop matches feature_engine_vec.py, dispatcher_integration.py and
# ml_heuristic_sweep.py, which reach into sibling dispatcher directories the
# same way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "codegen"))

from arch_specs_generated import LDS_TOTAL_CAPACITY_BY_ARCH  # noqa: E402
from codegen_common import normalize_gfx_arch  # noqa: E402


CANONICAL_COLUMNS = [
    "op_type",
    "dtype",
    "layout",
    "arch",
    "kernel_name",
    "m",
    "n",
    "k",
    "split_k",
    "measured_tflops",
    "latency_ms",
    "bandwidth_gb_s",
    "is_valid",
    "tile_m",
    "tile_n",
    "tile_k",
    "warp_m",
    "warp_n",
    "warp_k",
    "warp_tile_m",
    "warp_tile_n",
    "warp_tile_k",
    "pipeline",
    "scheduler",
    "epilogue",
    "pad_m",
    "pad_n",
    "pad_k",
    "persistent",
    # Fixed A/B/C global vector widths, in elements; 0 means native, i.e. the
    # width the kernel uses when it fixes none -- a property of the KERNEL (tile,
    # waves, warp tile, epilogue, arch) via codegen_common.gemm_native_vector_sizes,
    # NOT the widest width the problem allows. Parsed from the kernel name's optional
    # _vecA_B_C suffix. Only GemmUniversalVecFeatureEngine consumes them; the
    # base engine ignores them, so adding these does not change its schema.
    "vec_a",
    "vec_b",
    "vec_c",
    "run_id",
]


def parse_kernel_name(name: str) -> dict:
    """Extract kernel config fields from a gemm_universal kernel name.

    Name format:
      gemm[_universal]_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}
      _{padM}_{padN}_{padK}_{persistent}_{tileM}x{tileN}x{tileK}
      _{warpM}x{warpN}x{warpK}_{warpTileM}x{warpTileN}x{warpTileK}
      [_vec{A}_{B}_{C}]

    The ``_universal`` segment is optional because two producers emit the same
    field order under different prefixes: the tile-engine benchmark binaries use
    ``gemm_universal_``, while the dispatcher bridge (``GemmKernelConfig.name``)
    uses ``gemm_``. Only the bridge emits the ``_vec`` suffix, so requiring the
    longer prefix made the width columns unreachable from any in-tree producer.

    The optional ``_vec`` suffix carries fixed A/B/C global vector widths. It is
    absent for native-width kernels, which is why ``vec_a/b/c`` come back as 0
    rather than missing: 0 means "native", i.e. the width the kernel uses when it
    fixes none. That is a property of the KERNEL (tile, waves, warp tile,
    epilogue, arch) -- see codegen_common.gemm_native_vector_sizes -- and is NOT
    in general the widest width the problem allows. Kernels differing only in these widths are otherwise identical in
    the canonical schema, so without them a model sees their performance
    difference as noise.
    """
    result = {}
    try:
        # Strip the optional suffix before the positional split, but do not
        # record it until the name is known to be a gemm_universal kernel --
        # an unparseable name must still return {}, not a dict of vec defaults.
        #
        # Not $-anchored: the emitters append the variant suffix AFTER the
        # vector one (unified_gemm_codegen.py adds _preshuffle / _permuteN /
        # _multid / _streamk), so anchoring would read every fixed-width
        # variant kernel as native -- a wrong value, not a missing one.
        vec_match = re.search(r"_vec(\d+)_(\d+)_(\d+)(?=_|$)", name)
        if vec_match:
            vec = tuple(int(g) for g in vec_match.groups())
            name = name[: vec_match.start()] + name[vec_match.end() :]
        elif re.search(r"_vec\d", name):
            # A _vec token that does not parse means the grammar has moved.
            # Returning native here would record a narrowed kernel as the
            # default one, which is a different kernel, not missing data.
            return result
        else:
            vec = (0, 0, 0)
        prefix_match = re.match(
            r"gemm(?:_universal)?_(\w+?)_((?:rcr|rrr|crr|ccr))_(.*)", name
        )
        if not prefix_match:
            return result
        result["dtype"] = prefix_match.group(1)
        result["layout"] = prefix_match.group(2)
        result["vec_a"], result["vec_b"], result["vec_c"] = vec
        remainder = prefix_match.group(3)

        parts = remainder.split("_")
        if len(parts) < 10:
            return result

        result["pipeline"] = parts[0]
        result["epilogue"] = parts[1]
        result["scheduler"] = parts[2]
        result["pad_m"] = parts[3] == "True"
        result["pad_n"] = parts[4] == "True"
        result["pad_k"] = parts[5] == "True"
        result["persistent"] = parts[6] == "True"

        tile_dims = parts[7].split("x")
        warp_dims = parts[8].split("x")
        warp_tile_dims = parts[9].split("x")

        result["tile_m"] = int(tile_dims[0])
        result["tile_n"] = int(tile_dims[1])
        result["tile_k"] = int(tile_dims[2])
        result["warp_m"] = int(warp_dims[0])
        result["warp_n"] = int(warp_dims[1])
        result["warp_k"] = int(warp_dims[2])
        result["warp_tile_m"] = int(warp_tile_dims[0])
        result["warp_tile_n"] = int(warp_tile_dims[1])
        result["warp_tile_k"] = int(warp_tile_dims[2])
    except (IndexError, ValueError):
        pass
    return result


def _layout_from_problem(problem: dict) -> str:
    """Derive layout shorthand (rcr/rrr/etc.) from problem JSON fields."""
    la = problem.get("layout_a", "")
    lb = problem.get("layout_b", "")
    lc = problem.get("layout_c", "")

    def _tag(s):
        s = s.lower()
        if "row" in s:
            return "r"
        if "col" in s:
            return "c"
        return "?"

    return _tag(la) + _tag(lb) + _tag(lc)


def parse_streaming_log(
    path: str | Path,
    arch: str = "unknown",
    run_id: Optional[str] = None,
    op_type: str = "gemm_universal",
) -> pd.DataFrame:
    """Parse a CK Tile streaming benchmark log into a canonical DataFrame.

    The log alternates between shape headers and JSON result blocks:
        Shape N: M=16 N=1536 K=7168 dtype=fp8 layout=rcr
        {
          "name": "gemm_universal_...",
          "problem": { ... },
          "perf_result": { "latency(ms)": ..., "tflops(TFlops)": ..., "bandwidth(GB/s)": ... }
        }
    """
    path = Path(path)
    if run_id is None:
        run_id = hashlib.md5(path.name.encode()).hexdigest()[:12]

    shape_re = re.compile(
        r"Shape\s+\d+:\s+M=(\d+)\s+N=(\d+)\s+K=(\d+)\s+dtype=(\w+)\s+layout=(\w+)"
    )

    rows = []
    current_m, current_n, current_k = 0, 0, 0
    current_dtype, current_layout = "", ""
    json_buf = []
    brace_depth = 0

    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()

            shape_match = shape_re.search(stripped)
            if shape_match:
                current_m = int(shape_match.group(1))
                current_n = int(shape_match.group(2))
                current_k = int(shape_match.group(3))
                current_dtype = shape_match.group(4)
                current_layout = shape_match.group(5)
                continue

            if brace_depth == 0 and stripped.startswith("{"):
                json_buf = [stripped]
                brace_depth = stripped.count("{") - stripped.count("}")
                if brace_depth == 0:
                    raw = "\n".join(json_buf)
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                else:
                    continue
            elif brace_depth > 0:
                json_buf.append(stripped)
                brace_depth += stripped.count("{") - stripped.count("}")
                if brace_depth <= 0:
                    brace_depth = 0
                    raw = "\n".join(json_buf)
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                else:
                    continue
            else:
                continue

            # If we get here, obj was successfully parsed
            kernel_name = obj.get("name", "")
            problem = obj.get("problem", {})
            perf = obj.get("perf_result", {})

            m = problem.get("m", current_m)
            n = problem.get("n", current_n)
            k = problem.get("k", current_k)
            split_k = problem.get("split_k", 1)
            dtype = problem.get("dtype_a", current_dtype)
            layout = (
                _layout_from_problem(problem)
                if problem.get("layout_a")
                else current_layout
            )

            tflops = perf.get("tflops(TFlops)", 0.0)
            latency = perf.get("latency(ms)", 0.0)
            bandwidth = perf.get("bandwidth(GB/s)", 0.0)

            kp = parse_kernel_name(kernel_name)

            row = {
                "op_type": op_type,
                "dtype": dtype,
                "layout": layout,
                "arch": arch,
                "kernel_name": kernel_name,
                "m": m,
                "n": n,
                "k": k,
                "split_k": split_k,
                "measured_tflops": tflops,
                "latency_ms": latency,
                "bandwidth_gb_s": bandwidth,
                "is_valid": tflops > 0 and latency > 0,
                "run_id": run_id,
            }
            row.update(kp)
            rows.append(row)

    df = pd.DataFrame(rows)
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


def check_capture_hw_arch(hw: dict, arch: str) -> None:
    """Refuse a profile that describes a different part than ``--arch``.

    ``--capture_hw`` reads THIS node; the rows are labelled ``--arch``. Those
    differ in the normal gfx1250 workflow, where the log is produced on a bare
    node and converted on a login node of another architecture -- and the
    mismatch used to be harmless only because the profile carried nothing the
    engine would not have defaulted to anyway. It now carries ``lds_capacity``,
    so converting a gfx1250 log on a gfx950 host would stamp half that part's
    LDS onto every row and write it into ``feature_spec.json``, where it
    governs inference.

    Factored out of ``main`` so it can be tested: the check was unreachable
    inside argument parsing, and the test written for it asserted two string
    literals were unequal and passed with the guard deleted.
    """
    local = re.search(r"(gfx[0-9a-f]+)", str(hw.get("gfx_name", "")))
    if not local:
        return
    local_arch = local.group(1)
    if normalize_gfx_arch(str(arch)) != local_arch:
        raise SystemExit(
            f"--capture_hw reads this node's hardware ({local_arch}) but "
            f"--arch says {arch}. Run the conversion on a {arch} node, or drop "
            "--capture_hw and let the engine use its own defaults rather than "
            "another part's constants."
        )


def get_hardware_profile() -> dict:
    """Capture GPU hardware profile from rocminfo."""
    profile = {}
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=30
        )
        output = result.stdout

        gpu_section = False
        for line in output.split("\n"):
            line = line.strip()
            if "Device Type:" in line and "GPU" in line:
                gpu_section = True
                continue
            if gpu_section and "Device Type:" in line and "GPU" not in line:
                break
            if not gpu_section:
                continue

            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()

            mapping = {
                "Name": "gfx_name",
                "Marketing Name": "marketing_name",
                "Compute Unit": "num_cus",
                "SIMDs per CU": "simds_per_cu",
                "Shader Engines": "shader_engines",
                "Shader Arrs. per Eng.": "shader_arrays_per_engine",
                "Max Clock Freq. (MHz)": "max_clock_mhz",
                "Wavefront Size": "wavefront_size",
                "Max Waves Per CU": "max_waves_per_cu",
                "Chip ID": "chip_id",
            }

            if key in mapping:
                raw = val.split("(")[0].strip()
                try:
                    profile[mapping[key]] = int(raw)
                except ValueError:
                    profile[mapping[key]] = raw

        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("L1:") and "num_cus" in profile:
                raw = line.split(":")[1].strip().split("(")[0].strip()
                try:
                    profile["l1_cache_kb"] = int(raw)
                except ValueError:
                    pass
            elif line.startswith("L2:"):
                raw = line.split(":")[1].strip().split("(")[0].strip()
                try:
                    profile["l2_cache_kb"] = int(raw)
                except ValueError:
                    pass
            elif line.startswith("L3:"):
                raw = line.split(":")[1].strip().split("(")[0].strip()
                try:
                    profile["l3_cache_kb"] = int(raw)
                except ValueError:
                    pass

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # rocminfo does not report LDS capacity, so it comes from the generated
    # arch table. Without it the feature engine falls back to its 64 KB
    # constructor default, and lds_usage_ratio -- which divides by the capacity
    # -- saturates at 1.0 on the great majority of rows for any part with more
    # than 64 KB. gfx950 has 160 KB, so that was ~91% of a gfx950 sweep.
    # rocminfo's Name is a full target triple, not a bare target --
    # "amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-" -- so the gfx token has to be
    # pulled out before normalising. Passing the triple straight to
    # normalize_gfx_arch strips the feature suffix and leaves
    # "amdgcn-amd-amdhsa--gfx90a", which matches no table entry and silently
    # restores the 64 KB default: the defect this is here to fix.
    # The regex subsumes normalize_gfx_arch here: it cannot capture a colon,
    # so the token it returns is already bare. Calling the helper as well reads
    # as load-bearing and is not -- removing it changes nothing.
    match = re.search(r"(gfx[0-9a-f]+)", str(profile.get("gfx_name", "")))
    if match:
        gfx = match.group(1)
        if gfx in LDS_TOTAL_CAPACITY_BY_ARCH:
            profile["lds_capacity"] = LDS_TOTAL_CAPACITY_BY_ARCH[gfx]

    return profile


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a canonical parquet dataset."""
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: str | Path):
    """Save a DataFrame in canonical parquet format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")


def build_training_dataset(
    data_dir: str | Path,
    op_type: str = "gemm_universal",
    dtype: str = "fp8",
) -> pd.DataFrame:
    """Load and merge all parquet files matching the given op/dtype from a directory."""
    data_dir = Path(data_dir)
    frames = []
    for f in sorted(data_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        if "op_type" in df.columns:
            df = df[df["op_type"] == op_type]
        if "dtype" in df.columns:
            df = df[df["dtype"] == dtype]
        if len(df) > 0:
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No parquet files with op_type={op_type}, dtype={dtype} in {data_dir}"
        )
    return pd.concat(frames, ignore_index=True)


def main(argv=None):
    """CLI entry point.

    A function rather than a bare ``__main__`` block so the wiring can be
    tested. check_capture_hw_arch was factored out for exactly that
    reason, and its only caller then sat in here where no test could
    reach it -- deleting the call left the whole suite green. Testing a
    guard without testing that it is invoked is the same defect, one
    line further down.
    """
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Parse CK Tile benchmark data")
    parser.add_argument("input", help="Input file (log or parquet)")
    parser.add_argument("--output", "-o", required=True, help="Output parquet path")
    parser.add_argument("--arch", default="gfx950", help="GPU architecture")
    parser.add_argument("--op_type", default="gemm_universal", help="Operation type")
    parser.add_argument(
        "--capture_hw",
        action="store_true",
        help="Capture hardware profile from rocminfo",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)

    print(f"Parsing {input_path}...")
    t0 = time.time()

    if input_path.suffix == ".parquet":
        df = load_parquet(input_path)
    else:
        df = parse_streaming_log(input_path, arch=args.arch, op_type=args.op_type)

    elapsed = time.time() - t0
    print(f"Parsed {len(df)} rows in {elapsed:.1f}s")
    print(f"  Unique shapes: {df.groupby(['m', 'n', 'k']).ngroups}")
    print(f"  Unique kernels: {df['kernel_name'].nunique()}")
    print(f"  Valid rows: {df['is_valid'].sum()} / {len(df)}")

    if df["measured_tflops"].max() > 0:
        print(
            f"  TFLOPS range: {df['measured_tflops'].min():.2f} - {df['measured_tflops'].max():.2f}"
        )

    if args.capture_hw:
        hw = get_hardware_profile()
        # The profile describes THIS node; the rows are labelled --arch. Those
        # differ in the normal gfx1250 workflow, where the log is produced on a
        # bare node and converted on a login node of another architecture --
        # and the mismatch used to be harmless only because the profile carried
        # nothing the engine would not have defaulted to anyway. Now it carries
        # lds_capacity, so converting a gfx1250 log on a gfx950 host would
        # stamp half that part's LDS onto every row and write it into
        # feature_spec.json.
        check_capture_hw_arch(hw, args.arch)
        print(f"  Hardware profile: {hw}")
        for k, v in hw.items():
            df[f"hw_{k}"] = v

    save_parquet(df, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
