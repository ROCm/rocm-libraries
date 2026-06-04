#!/usr/bin/env python3
"""Generate pre-compiled Stockham device function headers for the rocFFT device library.

This script uses the existing rocfft_stockham_source_dump tool to generate
Stockham FFT device functions, then extracts just the device helper functions
(excluding embedded headers and the global kernel) and wraps them as
self-contained header files under stockham/.

Each generated header depends on the foundation headers in detail/ for types,
radix butterflies, and constants.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROCFFT_ROOT = SCRIPT_DIR.parents[3]
DUMP_TOOL = ROCFFT_ROOT / "build" / "library" / "src" / "rocfft_stockham_source_dump"
STOCKHAM_DIR = SCRIPT_DIR / "stockham"

CONFIGURATIONS = [
    # (length, factors, workgroup_size, threads_per_transform, precisions)
    (32,  "4,8",   8,   8,  ["single", "double"]),
    (64,  "2,4,8", 32,  32, ["single", "double"]),
    (128, "16,8",  64,  64, ["single", "double"]),
    (256, "16,16", 128, 128, ["single", "double"]),
    (512, "8,8,8", 256, 256, ["single"]),
    (512, "8,8,8", 128, 128, ["double"]),
]

LICENSE = """\
// Copyright (C) 2016 - 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
"""


def run_dump_tool(
    direction: str, precision: str, factors: str, wgs: int, tpt: int, arch: str = "gfx1201"
) -> str:
    """Run rocfft_stockham_source_dump and return stdout."""
    cmd = [
        str(DUMP_TOOL), direction, precision, factors, str(wgs), str(tpt), "0", "0", arch
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROCFFT_ROOT))
    if result.returncode != 0:
        print(f"ERROR: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout


def _find_function_block(lines: list[str], pattern: re.Pattern, start_from: int = 0) -> tuple[int, int] | None:
    """Find a function block: starts at the template/signature line matching
    pattern, ends at the matching closing brace."""
    start_idx = None
    for i in range(start_from, len(lines)):
        if pattern.search(lines[i]):
            if "template" in lines[i]:
                start_idx = i
            else:
                for j in range(i - 1, max(i - 5, -1), -1):
                    if "template" in lines[j]:
                        start_idx = j
                        break
                if start_idx is None:
                    start_idx = i
            break

    if start_idx is None:
        return None

    # Find the matching closing brace by counting braces.
    brace_depth = 0
    end_idx = start_idx
    for i in range(start_idx, len(lines)):
        brace_depth += lines[i].count("{") - lines[i].count("}")
        if brace_depth <= 0 and "{" in "".join(lines[start_idx:i+1]):
            end_idx = i + 1
            break
    else:
        end_idx = len(lines)

    return start_idx, end_idx


def extract_device_functions(source: str, length: int, direction: str) -> tuple[str, str]:
    """Extract the device helper functions from the dump output.

    Returns (common_functions, pass_function) where:
      - common_functions: lds_to_reg_input + lds_from_reg_output (direction-independent)
      - pass_function: forward_full_pass or inverse_full_pass (direction-specific)
    """
    lines = source.split("\n")

    lds_to_reg_pat = re.compile(rf"lds_to_reg_input_length{length}_device")
    lds_from_reg_pat = re.compile(rf"lds_from_reg_output_length{length}_device")
    pass_pat = re.compile(rf"{direction}_full_pass_length{length}_SBRR_device")

    lds_to_reg = _find_function_block(lines, lds_to_reg_pat)
    lds_from_reg = _find_function_block(lines, lds_from_reg_pat)
    pass_block = _find_function_block(lines, pass_pat)

    if lds_to_reg is None:
        raise RuntimeError(f"Could not find lds_to_reg_input_length{length}_device")
    if lds_from_reg is None:
        raise RuntimeError(f"Could not find lds_from_reg_output_length{length}_device")
    if pass_block is None:
        raise RuntimeError(f"Could not find {direction}_full_pass_length{length}_SBRR_device")

    common = "\n".join(lines[lds_to_reg[0]:lds_to_reg[1]]) + "\n\n" + \
             "\n".join(lines[lds_from_reg[0]:lds_from_reg[1]])
    pass_fn = "\n".join(lines[pass_block[0]:pass_block[1]])

    return common.rstrip(), pass_fn.rstrip()


def factors_to_array_literal(factors_str: str) -> str:
    """Convert '2,4,8' to '{2, 4, 8}'."""
    parts = factors_str.split(",")
    return "{" + ", ".join(parts) + "}"


def compute_twiddle_values(factors_str: str) -> list[tuple[float, float]]:
    """Compute twiddle table values using rocFFT's radix-table layout."""
    factors = [int(f) for f in factors_str.split(",")]
    twiddles: list[tuple[float, float]] = []
    product = 1
    for i, radix in enumerate(factors[:-1]):
        product *= radix
        next_radix = factors[i + 1]
        transform_length = product * next_radix
        for k in range(transform_length // next_radix):
            theta = -2.0 * math.pi * k / transform_length
            for j in range(1, next_radix):
                twiddles.append((math.cos(j * theta), math.sin(j * theta)))
    return twiddles


def _format_literal(value: float, scalar_type: str) -> str:
    """Format a float as a C++ literal with proper suffix."""
    literal = f"{value:.17g}"
    if "e" not in literal and "." not in literal:
        literal += ".0"
    if scalar_type == "float":
        return f"{literal}f"
    return literal


def format_twiddle_array(
    twiddles: list[tuple[float, float]], scalar_type: str, var_name: str
) -> str:
    """Format twiddle values as a __device__ __constant__ array."""
    entries = []
    for real, imag in twiddles:
        r = 0.0 if abs(real) < 1e-15 else real
        i = 0.0 if abs(imag) < 1e-15 else imag
        r_lit = _format_literal(r, scalar_type)
        i_lit = _format_literal(i, scalar_type)
        entries.append(f"    {{{r_lit}, {i_lit}}}")
    body = ",\n".join(entries)
    return (
        f"static __device__ __constant__ scalar_type {var_name}[{len(twiddles)}] = {{\n"
        f"{body}\n"
        f"}};\n"
    )


def generate_header(
    length: int,
    factors: str,
    wgs: int,
    tpt: int,
    precision: str,
    arch: str = "gfx1201",
) -> str:
    """Generate a complete stockham header for one (length, precision) config."""
    ept = length // tpt
    num_factors = len(factors.split(","))
    factors_arr = factors_to_array_literal(factors)
    guard = f"ROCFFT_DEVICE_STOCKHAM_L{length}_WGS{wgs}_{'SP' if precision == 'single' else 'DP'}_HPP"

    fwd_source = run_dump_tool("forward", precision, factors, wgs, tpt, arch)
    inv_source = run_dump_tool("inverse", precision, factors, wgs, tpt, arch)

    fwd_common, fwd_pass = extract_device_functions(fwd_source, length, "forward")
    _, inv_pass = extract_device_functions(inv_source, length, "inverse")

    prec_type = "float" if precision == "single" else "double"
    twiddles = compute_twiddle_values(factors)
    twiddle_array = format_twiddle_array(twiddles, prec_type, "twiddle_storage")

    return f"""{LICENSE}
/*! @file length_{length}_wgs{wgs}_{'sp' if precision == 'single' else 'dp'}.hpp
 *  @brief Pre-generated Stockham FFT device functions for length {length}, {precision} precision.
 *  @details Generated by generate_stockham_headers.py using
 *  rocfft_stockham_source_dump. Do not edit by hand; regenerate instead.
 *
 *  Configuration:
 *  - length                = {length}
 *  - factors               = {factors}
 *  - workgroup_size        = {wgs}
 *  - threads_per_transform = {tpt}
 *  - elements_per_thread   = {ept}
 *  - precision             = {precision}
 */

#pragma once

#ifndef {guard}
#define {guard}

#include "../detail/complex.hpp"
#include "../detail/radix.hpp"

namespace rocfft
{{
namespace device
{{
namespace stockham
{{

namespace detail_l{length}_wgs{wgs}_{precision[0]}p
{{

using scalar_type = rocfft_complex<{prec_type}>;

{twiddle_array}

// ---- LDS load/store helpers (direction-independent) ----

{fwd_common}

// ---- Forward pass ----

{fwd_pass}

// ---- Inverse pass ----

{inv_pass}

}} // namespace detail_l{length}_wgs{wgs}_{precision[0]}p

}} // namespace stockham
}} // namespace device
}} // namespace rocfft

#endif // {guard}
"""


def main():
    if not DUMP_TOOL.is_file():
        print(f"ERROR: Dump tool not found at {DUMP_TOOL}", file=sys.stderr)
        print("Build it first: cd rocm-libraries/projects/rocfft && cmake . -B build && cmake --build build --target rocfft_stockham_source_dump", file=sys.stderr)
        sys.exit(1)

    STOCKHAM_DIR.mkdir(parents=True, exist_ok=True)

    generated_files: list[str] = []

    for length, factors, wgs, tpt, precisions in CONFIGURATIONS:
        for precision in precisions:
            suffix = "sp" if precision == "single" else "dp"
            filename = f"length_{length}_wgs{wgs}_{suffix}.hpp"
            filepath = STOCKHAM_DIR / filename

            print(f"Generating {filename} ... ", end="", flush=True)
            content = generate_header(length, factors, wgs, tpt, precision)
            filepath.write_text(content)
            generated_files.append(filename)
            print("done")

    # Generate the stockham index header that forward-declares StockhamConfig
    # and includes all generated headers.
    index_content = f"""{LICENSE}
/*! @file index.hpp
 *  @brief Aggregating include for all pre-generated Stockham FFT configurations.
 *  @details Auto-generated by generate_stockham_headers.py. Do not edit by hand;
 *  regenerate instead.
 */

#pragma once

#ifndef ROCFFT_DEVICE_STOCKHAM_INDEX_HPP
#define ROCFFT_DEVICE_STOCKHAM_INDEX_HPP

{chr(10).join(f'#include "{f}"' for f in sorted(generated_files))}

#endif // ROCFFT_DEVICE_STOCKHAM_INDEX_HPP
"""
    index_path = STOCKHAM_DIR / "index.hpp"
    index_path.write_text(index_content)
    print(f"Generated {index_path.name}")

    print(f"\nGenerated {len(generated_files)} stockham headers + index.hpp")


if __name__ == "__main__":
    main()
