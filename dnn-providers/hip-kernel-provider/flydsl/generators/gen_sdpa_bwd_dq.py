# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the pinned SDPA backward dQ objects (gfx950, gfx11-generic, gfx942, gfx90a): the dQ (and dBias) kernel.

    python -m generators.gen_sdpa_bwd_dq --arch gfx950 --out-dir <content root> --reference-dir <S1 reference dir>

One op's objects (``sdpa_bwd_dq``). The build, the layout checks, the gates and the
manifest are shared with the other backward ops in ``_sdpa_bwd_cdna_build``. The rows are
the sweep's pins (``_instances``); each is built once and recorded as ``ok``,
``dual_object`` or ``error`` with its resource row and gate verdicts. Rebuilt objects are
compared with the pins by ``tools/sdpa_bwd_compare.py``. ``--arch gfx11-generic`` builds the gfx11 objects
(``_sdpa_bwd_rdna_build``), ``--arch gfx942`` / ``gfx90a`` the 16-bit and fp32 rows of one manifest (``_sdpa_bwd_split_build``); each
must be its S1 pin. The run also builds the tail-zero objects of the arch family
(``gen_sdpa_bwd_tailzero.companion_entries``) into the dQ folder and lists them under ``companions``: no pin, the tail-zero generator is
their provenance. The merged generator lists these same objects, so dQ is generated before merged.
"""

from __future__ import annotations

from . import _sdpa_bwd_rdna_build as _rdna
from . import _sdpa_bwd_split_build as _split
from ._sdpa_bwd_cdna_build import generate as _generate, run_op
from .gen_sdpa_bwd_tailzero import companion_entries as _tailzero

OP = "sdpa_bwd_dq"


# The arches the CDNA build does not own alone: the generic ones are built by ``_sdpa_bwd_rdna_build``, gfx942 and gfx90a (rows of
# both families) by ``_sdpa_bwd_split_build``. Both list the tail-zero companions.
_OTHER_ARCHES = {
    **dict.fromkeys(
        _rdna.GENERIC_ARCHES,
        lambda op, arch, out_root, only, reference_dir, stored_dir=None: _rdna.generate(
            op,
            arch,
            out_root,
            only,
            reference_dir,
            build_tailzero=_tailzero,
            stored_dir=stored_dir,
        ),
    ),
    **dict.fromkeys(
        _split.ARCHES,
        lambda op, arch, out_root, only, reference_dir, stored_dir=None: _split.generate(
            op,
            arch,
            out_root,
            only,
            reference_dir,
            build_tailzero=_tailzero,
            stored_dir=stored_dir,
        ),
    ),
}


def generate(arch, out_root, only=None, reference_dir=None, stored_dir=None):
    if arch in _OTHER_ARCHES:
        return _OTHER_ARCHES[arch](
            OP,
            arch,
            out_root,
            only,
            reference_dir,
            stored_dir,
        )
    return _generate(
        OP,
        arch,
        out_root,
        only,
        reference_dir,
        build_tailzero=_tailzero,
        stored_dir=stored_dir,
    )


def main(argv: list[str] | None = None) -> int:
    return run_op(OP, __doc__.splitlines()[0], argv, _OTHER_ARCHES, _tailzero)


if __name__ == "__main__":
    raise SystemExit(main())
