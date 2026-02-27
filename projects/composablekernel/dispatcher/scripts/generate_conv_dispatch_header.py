#!/usr/bin/env python3
"""Generate the conv_python_dispatch.hpp header for the Python conv library.

Reads the include_all headers to find available kernels and creates dispatch
aliases for 2D/3D x fwd/bwdd/bwdw.
"""
import argparse
import re
from pathlib import Path


def find_3d_launcher(include_all_path: Path, variant_prefix: str) -> str:
    """Find first 3D launcher name from an include_all header."""
    text = include_all_path.read_text()
    pattern = rf'(grouped_conv_{variant_prefix}_\w+_3d_\w+)\.hpp'
    match = re.search(pattern, text)
    if match:
        return match.group(1) + "_Launcher"
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    kdir = Path(args.kernel_dir)

    fwd_3d = find_3d_launcher(kdir / "include_all_grouped_conv_fwd_kernels.hpp", "fwd")
    bwdd_3d = find_3d_launcher(kdir / "include_all_grouped_conv_bwdd_kernels.hpp", "bwdd")
    bwdw_3d = find_3d_launcher(kdir / "include_all_grouped_conv_bwdw_kernels.hpp", "bwdw")

    lines = [
        "// Auto-generated dispatch header for Python conv library",
        "#pragma once",
        "",
        "// Forward kernels",
        '#include "include_all_grouped_conv_fwd_kernels.hpp"',
        "#define CONV_FWD_2D_AVAILABLE 1",
    ]
    if fwd_3d:
        lines += [f"#define CONV_FWD_3D_AVAILABLE 1", f"using ConvFwd3dLauncher = {fwd_3d};"]
    lines += [
        "",
        "// Backward data kernels",
        '#include "include_all_grouped_conv_bwdd_kernels.hpp"',
        "#define CONV_BWDD_2D_AVAILABLE 1",
    ]
    if bwdd_3d:
        lines += [f"#define CONV_BWDD_3D_AVAILABLE 1", f"using ConvBwdData3dLauncher = {bwdd_3d};"]
    lines += [
        "",
        "// Backward weight kernels",
        '#include "include_all_grouped_conv_bwdw_kernels.hpp"',
        "#define CONV_BWDW_2D_AVAILABLE 1",
    ]
    if bwdw_3d:
        lines += [f"#define CONV_BWDW_3D_AVAILABLE 1", f"using ConvBwdWeight3dLauncher = {bwdw_3d};"]

    # Kernel name table for Python introspection
    names = []
    if True:  # fwd 2D always present
        names.append('"fwd_2d"')
    if fwd_3d:
        names.append('"fwd_3d"')
    if True:  # bwdd 2D
        names.append('"bwdd_2d"')
    if bwdd_3d:
        names.append('"bwdd_3d"')
    if True:  # bwdw 2D
        names.append('"bwdw_2d"')
    if bwdw_3d:
        names.append('"bwdw_3d"')

    lines += [
        "",
        "// Kernel inventory for Python",
        f"static const char* CONV_KERNEL_NAMES[] = {{{', '.join(names)}}};",
        f"static const int CONV_KERNEL_COUNT = {len(names)};",
        "",
    ]

    Path(args.output).write_text("\n".join(lines) + "\n")
    print(f"Generated dispatch header: {args.output} ({len(names)} kernels)")


if __name__ == "__main__":
    main()
