# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Write the hipRTC pre-include header embedded in libhiprtc-builtins to a file.

hipRTC prepends this header to every program it compiles. The builtins library
exports the text as ``__hipRTC_header`` and its length in bytes as
``__hipRTC_header_size``, so loading the library and copying exactly that many
bytes yields the header without relying on object-file tooling.

Usage: dump_hiprtc_header.py <libhiprtc-builtins> <output-header>
"""

import ctypes
import os
import sys

HEADER_SYMBOL = "__hipRTC_header"
SIZE_SYMBOL = "__hipRTC_header_size"


def read_header(library_path):
    library = ctypes.CDLL(library_path)
    try:
        size = ctypes.c_uint.in_dll(library, SIZE_SYMBOL).value
        address = ctypes.addressof(ctypes.c_char.in_dll(library, HEADER_SYMBOL))
    except ValueError as error:
        raise RuntimeError(
            f"{library_path} does not export {HEADER_SYMBOL} and {SIZE_SYMBOL}: {error}"
        ) from error

    header = ctypes.string_at(address, size)
    if not header or b"\0" in header:
        raise RuntimeError(
            f"{library_path} has an invalid {HEADER_SYMBOL} ({size} bytes)"
        )
    return header


def main(argv):
    if len(argv) != 3:
        print(f"usage: {argv[0]} <libhiprtc-builtins> <output-header>", file=sys.stderr)
        return 2

    library_path, output_path = argv[1], argv[2]
    try:
        header = read_header(library_path)
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    # Replace the output atomically so a failed run never leaves a truncated header
    # that looks up to date.
    temporary_path = f"{output_path}.tmp"
    with open(temporary_path, "wb") as output:
        output.write(header)
    os.replace(temporary_path, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
