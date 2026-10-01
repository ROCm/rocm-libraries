# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Read packaged Tensile metadata without importing the kernel generator."""

import zlib


def readMessagePack(path):
    """Decode one .dat or .dat.zlib, using the native loader's strict framing."""
    import msgpack

    data = path.read_bytes()
    if path.name.endswith(".zlib"):
        decoder = zlib.decompressobj()
        data = decoder.decompress(data) + decoder.flush()
        if not decoder.eof:
            raise zlib.error("incomplete zlib stream")
        if decoder.unused_data:
            raise zlib.error("trailing bytes after zlib stream")
    return msgpack.unpackb(data, raw=False, strict_map_key=False)
