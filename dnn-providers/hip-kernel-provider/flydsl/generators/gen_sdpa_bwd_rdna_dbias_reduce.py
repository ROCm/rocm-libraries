# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the RDNA SDPA backward smoke object (gfx11-generic): the dBias reduction of the split backward.

    python -m generators.gen_sdpa_bwd_rdna_dbias_reduce --arch gfx11-generic --out-dir <content root>

One op's objects (``sdpa_bwd_rdna_dbias_reduce``), the backward's landing place on the
generic gfx11 target. The build, the layout checks and the manifest are in
``_sdpa_bwd_rdna_build``; the row is ``_instances``'. The row is built once and recorded
as ``ok``, ``dual_object`` or ``error``. The CDNA gates are gfx950-only and are recorded as ``n/a``.
"""

from __future__ import annotations

from ._sdpa_bwd_rdna_build import generate as _generate, run_op


def generate(arch, out_root, only=None):
    return _generate(arch, out_root, only)


def main(argv: list[str] | None = None) -> int:
    return run_op(__doc__.splitlines()[0], argv)


if __name__ == "__main__":
    raise SystemExit(main())
