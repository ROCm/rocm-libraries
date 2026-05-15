# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Assemble a Python wheel from pre-built artifacts (stdlib only)."""

import argparse
import base64
import csv
import hashlib
import io
import os
import re
import stat
import sys
import zipfile

_PACKAGE_NAME = "hipdnn_frontend"
_DEFAULT_VERSION = "0.1.0"
_REQUIRES_DIST = ["numpy>=1.19.0"]


def _parse_so_tags(so_filename):
    """Extract Python, ABI, and platform tags from a .so filename.

    Example: hipdnn_frontend_python.cpython-312-x86_64-linux-gnu.so
             -> ('cp312', 'cp312', 'linux_x86_64')
    """
    m = re.search(
        r"\.cpython-(\d+)([a-z]*)-(.+)\.so$",
        so_filename,
    )
    if not m:
        sys.exit(f"Cannot parse tags from filename: {so_filename}")

    ver, flags, platform_raw = m.groups()
    py_tag = f"cp{ver}"
    abi_tag = f"cp{ver}{flags}"
    platform_tag = _gnu_triplet_to_wheel_tag(platform_raw)
    return py_tag, abi_tag, platform_tag


def _gnu_triplet_to_wheel_tag(triplet):
    """Convert a GNU triplet (from .so filename) to a PEP 425 platform tag.

    x86_64-linux-gnu   -> linux_x86_64
    aarch64-linux-gnu  -> linux_aarch64
    """
    parts = triplet.split("-")
    if len(parts) >= 2 and parts[1] == "linux":
        return f"linux_{parts[0]}"
    return triplet.replace("-", "_").replace(".", "_")


def _hash_record(data):
    """Return 'sha256=<urlsafe-base64>,<size>' for a RECORD entry."""
    digest = hashlib.sha256(data).digest()
    b64 = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={b64}", len(data)


def _get_init_file(package_dir):
    """Return (arcname, filepath) for __init__.py, or None if missing."""
    init_path = os.path.join(package_dir, "__init__.py")
    if os.path.isfile(init_path):
        return f"{_PACKAGE_NAME}/__init__.py", init_path
    return None


def _build_metadata(version):
    return (
        f"Metadata-Version: 2.1\n"
        f"Name: {_PACKAGE_NAME.replace('_', '-')}\n"
        f"Version: {version}\n"
        f"Summary: Python bindings for the hipDNN frontend library\n"
        f"Author: Advanced Micro Devices, Inc.\n"
        f"License: MIT\n"
        f"Requires-Python: >=3.8\n"
        + "".join(f"Requires-Dist: {dep}\n" for dep in _REQUIRES_DIST)
    )


def _build_wheel_metadata(py_tag, abi_tag, platform_tag):
    return (
        f"Wheel-Version: 1.0\n"
        f"Generator: assemble_wheel.py\n"
        f"Root-Is-Purelib: false\n"
        f"Tag: {py_tag}-{abi_tag}-{platform_tag}\n"
    )


def assemble(so_path, package_dir, output_dir, version):
    so_filename = os.path.basename(so_path)
    py_tag, abi_tag, platform_tag = _parse_so_tags(so_filename)

    dist_info = f"{_PACKAGE_NAME}-{version}.dist-info"
    wheel_name = f"{_PACKAGE_NAME}-{version}-{py_tag}-{abi_tag}-{platform_tag}.whl"
    wheel_path = os.path.join(output_dir, wheel_name)

    records = []

    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as whl:
        # 1. __init__.py
        init_entry = _get_init_file(package_dir)
        if init_entry:
            arcname, filepath = init_entry
            with open(filepath, "rb") as f:
                data = f.read()
            whl.writestr(arcname, data)
            h, sz = _hash_record(data)
            records.append((arcname, h, sz))

        # 2. Compiled extension — use ZIP_STORED for the .so (no compression benefit)
        so_arcname = f"{_PACKAGE_NAME}/{so_filename}"
        with open(so_path, "rb") as f:
            so_data = f.read()
        info = zipfile.ZipInfo(so_arcname)
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = (
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IXGRP
            | stat.S_IROTH
            | stat.S_IXOTH
        ) << 16
        whl.writestr(info, so_data)
        h, sz = _hash_record(so_data)
        records.append((so_arcname, h, sz))

        # 3. dist-info/METADATA
        metadata = _build_metadata(version).encode()
        arcname = f"{dist_info}/METADATA"
        whl.writestr(arcname, metadata)
        h, sz = _hash_record(metadata)
        records.append((arcname, h, sz))

        # 4. dist-info/WHEEL
        wheel_meta = _build_wheel_metadata(py_tag, abi_tag, platform_tag).encode()
        arcname = f"{dist_info}/WHEEL"
        whl.writestr(arcname, wheel_meta)
        h, sz = _hash_record(wheel_meta)
        records.append((arcname, h, sz))

        # 5. dist-info/top_level.txt
        top_level = f"{_PACKAGE_NAME}\n".encode()
        arcname = f"{dist_info}/top_level.txt"
        whl.writestr(arcname, top_level)
        h, sz = _hash_record(top_level)
        records.append((arcname, h, sz))

        # 6. dist-info/RECORD (must be last — its own entry has no hash)
        buf = io.StringIO()
        writer = csv.writer(buf)
        for row in records:
            writer.writerow(row)
        writer.writerow((f"{dist_info}/RECORD", "", ""))
        record_data = buf.getvalue().encode()
        whl.writestr(f"{dist_info}/RECORD", record_data)

    print(f"Wheel written: {wheel_path}")
    return wheel_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--so-path", required=True, help="Path to compiled .so")
    parser.add_argument(
        "--package-dir", required=True, help="Path to pure-Python package dir"
    )
    parser.add_argument("--output-dir", required=True, help="Directory for output .whl")
    parser.add_argument("--version", default=_DEFAULT_VERSION, help="Package version")
    args = parser.parse_args()

    if not os.path.isfile(args.so_path):
        sys.exit(f"Extension not found: {args.so_path}")
    if not os.path.isdir(args.package_dir):
        sys.exit(f"Package directory not found: {args.package_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    assemble(args.so_path, args.package_dir, args.output_dir, args.version)


if __name__ == "__main__":
    main()
