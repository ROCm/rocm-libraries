# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the pinned CDNA SDPA backward prep objects (gfx950): the Delta and LSE workspace kernel that runs before dK/dV and dQ; the `sink=1` rows are its attention-sink twin.

    python -m generators.gen_sdpa_bwd_prep --arch gfx950 --out-dir <content root>

One op's objects (``sdpa_bwd_prep``). The build, the layout checks, the gates and the
manifest are shared with the other backward ops in ``_sdpa_bwd_cdna_build``. The rows are
the sweep's pins (``_instances``); each is built once and recorded as ``ok``,
``dual_object`` or ``error`` with its resource row and gate verdicts. Rebuilt objects are
compared with the pins by ``tools/sdpa_bwd_compare.py``.
"""

from __future__ import annotations

from ._sdpa_bwd_cdna_build import generate as _generate, run_op

OP = "sdpa_bwd_prep"


def generate(arch, out_root, only=None):
    return _generate(OP, arch, out_root, only)


def main(argv: list[str] | None = None) -> int:
    return run_op(OP, __doc__.splitlines()[0], argv)


if __name__ == "__main__":
    raise SystemExit(main())
