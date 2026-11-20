# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the pinned CDNA SDPA backward merged objects (gfx950): the merged kernel: dK/dV, dQ and dBias in one launch, with no prep.

    python -m generators.gen_sdpa_bwd_merged --arch gfx950 --out-dir <content root> --reference-dir <pins>

One op's objects (``sdpa_bwd_merged``). The build, the layout checks, the gates and the
manifest are shared with the other backward ops in ``_sdpa_bwd_cdna_build``. The rows are
the sweep's pins (``_instances``); each is built once and recorded with its resource row and
gate verdicts. ``--reference-dir`` holds the pinned objects (``<hash>.hsaco``): a row that never
writes dB ships its pin's bytes and its rebuild only verifies it (a rebuild that differs from the pin
stops the run through ``_stop_reasons``); a dB-storing bias row ships its rebuild, an object with the merged kernel and the dBias reduce
kernel, whose per-kernel gate must be green. The tail-zero companions are the dQ folder's (``sdpa_bwd_dq``, generated first into the same
content root): the manifest lists them by path (``../../sdpa_bwd_dq/<arch>/<name>``) and the run stops if they are missing or altered. A row that does not build or a red gate exits
nonzero and ships nothing.
"""

from __future__ import annotations

from ._sdpa_bwd_cdna_build import dq_tailzero_companions, generate as _generate, run_op

OP = "sdpa_bwd_merged"


def generate(arch, out_root, reference_dir, only=None, stored_dir=None):
    return _generate(
        OP,
        arch,
        out_root,
        only,
        reference_dir,
        build_tailzero=dq_tailzero_companions,
        stored_dir=stored_dir,
    )


def main(argv: list[str] | None = None) -> int:
    return run_op(
        OP, __doc__.splitlines()[0], argv, build_tailzero=dq_tailzero_companions
    )


if __name__ == "__main__":
    raise SystemExit(main())
