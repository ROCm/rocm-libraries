# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The descriptor serializer, with standard-library imports only.

Every descriptor is ``json.dumps(indent=2)`` except a KDP's kernel list, which is
written one kernel per line. A catalog holds hundreds of kernels of roughly a
hundred indented lines each; a line per kernel keeps the file and its diffs in
proportion to the number of kernels while the KDP header stays readable.
``tools/splice_additions.py`` imports this module so the files it rewrites keep
the generator's bytes.
"""

import json

#: The KDP key holding the kernel entries.
KERNELS = "kernelDescriptors"

# Stands in for the kernel list while the rest of the KDP is indented.
_ROWS = "\x00kernel rows\x00"


def dump_descriptor(obj: dict) -> str:
    """The text the generator writes for descriptor ``obj``."""
    kernels = obj.get(KERNELS)
    if not isinstance(kernels, list) or not kernels:
        return json.dumps(obj, indent=2, sort_keys=False) + "\n"
    text = json.dumps({**obj, KERNELS: _ROWS}, indent=2, sort_keys=False) + "\n"
    rows = ",\n".join(
        "    " + json.dumps(kernel, sort_keys=False, separators=(",", ":"))
        for kernel in kernels
    )
    return text.replace(json.dumps(_ROWS), "[\n" + rows + "\n  ]", 1)
