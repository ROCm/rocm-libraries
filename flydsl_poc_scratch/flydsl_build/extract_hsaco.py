#!/usr/bin/env python3
"""Extract a bare HSACO from a FlyDSL `gpu-module-to-binary` MLIR dump.

FlyDSL's stage-19 dump (`19_gpu_module_to_binary.mlir`) embeds the compiled code
object as a `bin = "<mlir-escaped ELF>"` string attribute inside a `gpu.binary`
op. This pulls that attribute out, un-escapes the MLIR string escapes, and writes
the raw ELF bytes to a `.hsaco` file that `hipModuleLoadData` can load directly.

Throwaway POC tooling — no error handling beyond what's needed to fail loudly.

Usage:
    extract_hsaco.py <19_gpu_module_to_binary.mlir> <out.hsaco>
"""
import sys


def unescape_mlir_string(s: str) -> bytes:
    """Un-escape an MLIR string-attribute body into raw bytes.

    MLIR prints printable chars raw, `\\` as `\\\\`, `"` as `\\"`, and every other
    byte as `\\XX` (two uppercase hex digits). Mirrors jit_function._extract_isa_text
    but yields bytes (each code point is one byte, 0-255).
    """
    out = bytearray()
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if nxt == "\\":
                out.append(0x5C)
                i += 2
                continue
            if nxt == '"':
                out.append(0x22)
                i += 2
                continue
            # \XX hex escape
            hex_str = s[i + 1 : i + 3]
            out.append(int(hex_str, 16))
            i += 3
            continue
        out.append(ord(ch))
        i += 1
    return bytes(out)


def extract_bin_attr(mlir_text: str) -> bytes:
    """Find `bin = "..."` in the gpu.binary op and return its raw bytes."""
    marker = 'bin = "'
    start = mlir_text.index(marker) + len(marker)
    # Walk to the closing unescaped quote.
    i = start
    n = len(mlir_text)
    while i < n:
        if mlir_text[i] == '"' and mlir_text[i - 1] != "\\":
            break
        i += 1
    body = mlir_text[start:i]
    return unescape_mlir_string(body)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    mlir_path, out_path = sys.argv[1], sys.argv[2]
    with open(mlir_path, "r", encoding="utf-8", errors="surrogateescape") as f:
        text = f.read()
    blob = extract_bin_attr(text)
    if blob[:4] != b"\x7fELF":
        raise SystemExit(f"extracted blob is not ELF (starts with {blob[:4]!r})")
    with open(out_path, "wb") as f:
        f.write(blob)
    print(f"wrote {out_path} ({len(blob)} bytes, ELF ok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
