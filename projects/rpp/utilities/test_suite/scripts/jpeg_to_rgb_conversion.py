#!/usr/bin/env python3
"""
Decode JPEGs with libturbojpeg for offline conversion to the .rgb format consumed by
``read_image_batch_packed`` in ``utilities/test_suite/rpp_test_suite_image.h`` (Tensor-Image ``decoder_type`` 0):

  - tjInitDecompress / tjDestroy once per image
  - Header: prefer tjDecompressHeader3 (width, height, subsamp, colorspace);
    else tjDecompressHeader2 (no colorspace in the C API — only subsamp).
  - Grayscale output if JPEG colorspace is TJCS_GRAY or subsampling is
    TJSAMP_GRAY; else decompress to TJPF_RGB with TJFLAG_ACCURATEDCT.
  - CMYK / YCCK JPEGs → TJPF_CMYK (4 channels).

Constants below must match the installed ``turbojpeg.h`` (libjpeg-turbo).
Wrong TJPF_* values cause buffer overruns and ``munmap_chunk()`` crashes.

Processes one file at a time (sequential loop, no threads).

Output per input ``*.jpg`` / ``*.jpeg``:

  - ``<stem>.rgb`` — 24-byte binary header (magic, version, width, height, channels, reserved)
                     followed by tight row-major pixels (H rows × W×C bytes per row).

    Header format (little-endian):
      magic: 0x52474242 ("RGBB")
      version: 1
      width, height, channels (1=gray, 3=RGB, 4=CMYK): uint32_t each
      reserved1: 0 (for future use)

Requires: libturbojpeg at runtime. No PyPI packages. See ``utilities/test_suite/scripts/README.md`` for install hints.

Usage:
  python3 jpeg_to_rgb_conversion.py /path/to/dir [--out-dir DIR] [--recursive]
"""

from __future__ import annotations

import argparse
import ctypes
import struct
import sys
from pathlib import Path

# --- Match turbojpeg.h (libjpeg-turbo) -----------------------------------------
# Pixel formats (enum TJPF order)
TJPF_RGB = 0
TJPF_GRAY = 6
TJPF_CMYK = 11

TJFLAG_ACCURATEDCT = 4096

# JPEG colorspaces (enum TJCS)
TJCS_GRAY = 2
TJCS_CMYK = 3
TJCS_YCCK = 4

# Chroma subsampling (enum TJSAMP) — TJSAMP_GRAY means grayscale JPEG
TJSAMP_GRAY = 3

_LIB_CANDIDATES = (
    "libturbojpeg.so.0",
    "libturbojpeg.so",
    "libturbojpeg.dylib",
)


def _load_turbojpeg():
    for name in _LIB_CANDIDATES:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    sys.exit(
        "Could not load libturbojpeg. Install libjpeg-turbo / libturbojpeg "
        f"(tried: {', '.join(_LIB_CANDIDATES)})."
    )


def _bind_minimal(lib):
    """Only set restypes (avoids argtype / ABI mismatches on some distros)."""
    lib.tjInitDecompress.restype = ctypes.c_void_p
    lib.tjDestroy.restype = None
    lib.tjDecompressHeader2.restype = ctypes.c_int
    lib.tjDecompress2.restype = ctypes.c_int
    if hasattr(lib, "tjDecompressHeader3"):
        lib.tjDecompressHeader3.restype = ctypes.c_int


def _handle_int(handle) -> int:
    if handle is None:
        return 0
    if isinstance(handle, ctypes.c_void_p):
        val = handle.value
        return 0 if val is None else int(val)
    return int(handle)


def _read_jpeg_header(
    lib, handle: int, buf, nbytes: int, w, h, subsamp, colorspace
) -> int:
    """Return 0 on success. Fills w, h, subsamp, colorspace (colorspace only with Header3)."""
    sz = ctypes.c_ulong(nbytes)
    hv = ctypes.c_void_p(handle)
    if hasattr(lib, "tjDecompressHeader3"):
        return lib.tjDecompressHeader3(
            hv,
            buf,
            sz,
            ctypes.byref(w),
            ctypes.byref(h),
            ctypes.byref(subsamp),
            ctypes.byref(colorspace),
        )
    colorspace.value = -1
    return lib.tjDecompressHeader2(
        hv,
        buf,
        sz,
        ctypes.byref(w),
        ctypes.byref(h),
        ctypes.byref(subsamp),
    )


def decode_to_raw(lib, jpeg_bytes: bytes) -> tuple[bytes, int, int, int, str]:
    """Decode one JPEG: init → header → decompress → destroy."""
    handle_raw = lib.tjInitDecompress()
    handle = _handle_int(handle_raw)
    if not handle:
        raise RuntimeError("tjInitDecompress failed")
    try:
        w = ctypes.c_int()
        h = ctypes.c_int()
        subsamp = ctypes.c_int()
        colorspace = ctypes.c_int()
        nbytes_jpeg = len(jpeg_bytes)
        buf = (ctypes.c_ubyte * nbytes_jpeg).from_buffer_copy(jpeg_bytes)
        rc = _read_jpeg_header(lib, handle, buf, nbytes_jpeg, w, h, subsamp, colorspace)
        if rc != 0:
            raise RuntimeError(f"jpeg header parse failed with code {rc}")

        width, height = w.value, h.value
        if width <= 0 or height <= 0 or width > 65535 or height > 65535:
            raise RuntimeError(f"invalid JPEG dimensions {width}x{height}")

        cs = colorspace.value
        ss = subsamp.value
        is_gray = cs == TJCS_GRAY or (
            cs < 0 and ss == TJSAMP_GRAY
        )  # Header2: no colorspace, use subsamp
        is_cmyk = cs == TJCS_CMYK or cs == TJCS_YCCK

        if is_gray:
            channels = 1
            pf = TJPF_GRAY
            flags = 0
            fmt = "GRAY"
            pix_bytes = 1
        elif is_cmyk:
            channels = 4
            pf = TJPF_CMYK
            flags = 0
            fmt = "CMYK"
            pix_bytes = 4
        else:
            channels = 3
            pf = TJPF_RGB
            flags = TJFLAG_ACCURATEDCT
            fmt = "RGB"
            pix_bytes = 3

        nbytes = width * height * pix_bytes
        dst = (ctypes.c_ubyte * nbytes)()
        # pitch=0 means width * tjPixelSize[pf] per libjpeg-turbo docs
        rc = lib.tjDecompress2(
            ctypes.c_void_p(handle),
            buf,
            ctypes.c_ulong(nbytes_jpeg),
            dst,
            ctypes.c_int(width),
            ctypes.c_int(0),
            ctypes.c_int(height),
            ctypes.c_int(pf),
            ctypes.c_int(flags),
        )
        if rc != 0:
            raise RuntimeError(f"tjDecompress2 failed with code {rc}")

        raw = ctypes.string_at(ctypes.addressof(dst), nbytes)
        return (raw, width, height, channels, fmt)
    finally:
        lib.tjDestroy(ctypes.c_void_p(handle))


def create_rgb_header(width: int, height: int, channels: int) -> bytes:
    """
    Create 24-byte .rgb file header.

    Format (little-endian):
      magic: 0x52474242 ("RGBB")
      version: 1
      width, height, channels: uint32_t
      reserved1: 0
    """
    MAGIC = 0x52474242  # "RGBB"
    VERSION = 1
    return struct.pack("<6I", MAGIC, VERSION, width, height, channels, 0)


def _iter_jpegs(root: Path, recursive: bool):
    patterns = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
    if recursive:
        for pat in patterns:
            yield from root.rglob(pat)
    else:
        for pat in patterns:
            yield from root.glob(pat)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing JPEG files",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write .rgb files here (default: next to each JPEG)",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Also scan subdirectories for JPEGs",
    )
    args = ap.parse_args()

    indir: Path = args.input_dir
    if not indir.is_dir():
        sys.exit(f"Not a directory: {indir}")

    out_root: Path | None = args.out_dir
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)

    lib = _load_turbojpeg()
    _bind_minimal(lib)

    n_ok = 0
    n_fail = 0
    for jpg in sorted(_iter_jpegs(indir, args.recursive), key=lambda p: str(p)):
        if out_root is not None:
            stem = jpg.stem
            rgb_path = out_root / f"{stem}.rgb"
        else:
            rgb_path = jpg.with_suffix(".rgb")

        try:
            data = jpg.read_bytes()
            raw_pixels, w, h, c, fmt = decode_to_raw(lib, data)

            # Create .rgb file with embedded header
            header = create_rgb_header(w, h, c)
            rgb_path.write_bytes(header + raw_pixels)

            print(f"OK {jpg} -> {rgb_path} ({w}x{h}x{c} {fmt})")
            n_ok += 1
        except OSError as e:
            print(f"SKIP read {jpg}: {e}", file=sys.stderr)
            n_fail += 1
        except Exception as e:
            print(f"FAIL {jpg}: {e}", file=sys.stderr)
            n_fail += 1

    print(f"Done: {n_ok} converted, {n_fail} skipped/failed.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
