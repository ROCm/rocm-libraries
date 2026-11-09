# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the checked-in CDNA SDPA code objects of one kernel family: gfx950's dual-wave software-pipelined kernel (``flash_attn_gfx950.py``).

    python -m generators.gen_sdpa_cdna_dualwave --arch gfx950

One engine's objects (engine ``hipkernel:flydsl_sdpa_cdna_dualwave``; gfx950 only). The build, the layout checks and the
manifest are shared with the other CDNA SDPA families in ``_sdpa_cdna_build``.
"""

from __future__ import annotations

from ._sdpa_cdna_build import generate as _generate, run

OP = "sdpa_cdna_dualwave"


def generate(arch, out_root, keep_ir=None):
    return _generate(OP, arch, out_root, keep_ir)


def main(argv: list[str] | None = None) -> int:
    return run(OP, __doc__.splitlines()[0], argv)


if __name__ == "__main__":
    raise SystemExit(main())
