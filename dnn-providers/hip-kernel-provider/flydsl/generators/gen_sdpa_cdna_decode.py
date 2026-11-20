# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the checked-in CDNA SDPA code objects of one kernel family: the generic kernel's split-KV decode objects, each carrying its merge kernel.

    python -m generators.gen_sdpa_cdna_decode --arch gfx942

One engine's objects (engine ``hipkernel:flydsl_sdpa_cdna_decode``). The build, the layout checks and the
manifest are shared with the other CDNA SDPA families in ``_sdpa_cdna_build``.
"""

from __future__ import annotations

from ._sdpa_cdna_build import generate as _generate, run

OP = "sdpa_cdna_decode"


def generate(arch, out_root, keep_ir=None):
    return _generate(OP, arch, out_root, keep_ir)


def main(argv: list[str] | None = None) -> int:
    return run(OP, __doc__.splitlines()[0], argv)


if __name__ == "__main__":
    raise SystemExit(main())
