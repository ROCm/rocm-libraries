# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the pinned SDPA backward prep objects (gfx950, gfx11-generic, gfx942, gfx90a): the Delta and LSE workspace kernel that runs before dK/dV and dQ; the `sink=1` rows are its attention-sink twin.

    python -m generators.gen_sdpa_bwd_prep --arch gfx950 --out-dir <content root> --reference-dir <S1 reference dir>

One op's objects (``sdpa_bwd_prep``). The build, the layout checks, the gates and the
manifest are shared with the other backward ops in ``_sdpa_bwd_cdna_build``. The rows are
the sweep's pins (``_instances``); each is built once and recorded as ``ok``,
``dual_object`` or ``error`` with its resource row and gate verdicts. Rebuilt objects are
compared with the pins by ``tools/sdpa_bwd_compare.py``. ``--arch gfx11-generic`` builds the gfx11 objects
(``_sdpa_bwd_rdna_build``), ``--arch gfx942`` / ``gfx90a`` the 16-bit and fp32 rows of one manifest (``_sdpa_bwd_split_build``); each
must be its S1 pin.
"""

from __future__ import annotations

from . import _sdpa_bwd_rdna_build as _rdna
from . import _sdpa_bwd_split_build as _split
from ._sdpa_bwd_cdna_build import generate as _generate, run_op

OP = "sdpa_bwd_prep"


# The arches the CDNA build does not own alone: the generic ones are built by ``_sdpa_bwd_rdna_build``, gfx942 and gfx90a (rows of
# both families) by ``_sdpa_bwd_split_build``.
_OTHER_ARCHES = {
    **dict.fromkeys(_rdna.GENERIC_ARCHES, _rdna.generate),
    **dict.fromkeys(_split.ARCHES, _split.generate),
}


def generate(arch, out_root, only=None, reference_dir=None):
    if arch in _OTHER_ARCHES:
        return _OTHER_ARCHES[arch](OP, arch, out_root, only, reference_dir)
    return _generate(OP, arch, out_root, only, reference_dir)


def main(argv: list[str] | None = None) -> int:
    return run_op(OP, __doc__.splitlines()[0], argv, _OTHER_ARCHES)


if __name__ == "__main__":
    raise SystemExit(main())
