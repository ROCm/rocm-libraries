# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Force a pipeline in GPU tests using the real generated selection predicates."""

import argparse
from pathlib import Path
import sys

_CK_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_CK_ROOT / "example/ck_tile/01_fmha"))
from codegen.ops.fmha_fwd import (  # noqa: E402
    FMHA_FWD_API_HEADER,
    FmhaFwdApiPool,
    get_fwd_blobs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {Path(name).name for name in args.sources.read_text().splitlines()}
    pool = FmhaFwdApiPool()
    _, kernels = get_fwd_blobs(
        ["gfx1250"],
        "fmha_fwd_d*_[bf][fp]16_*_nlogits_nbias*_ndropout_nskip_nqscale*",
        0,
        [64, 128, 192],
        "simplified",
    )
    for kernel in kernels:
        if kernel.filename in sources:
            pool.register_traits(kernel.api_trait())
    source = FMHA_FWD_API_HEADER
    # Compare sched and its short-Q/sink QR fallback; exclude the competing
    # pingpong pipeline so its preference cannot bypass sched in these tests.
    for name, tags in (
        ("sched", ("qr_tdm_sched", "qr_tdm")),
        ("cr", ("qr_tdm_cr",)),
    ):
        source += pool.render(
            f"{name}_only", filter_fn=lambda trait: trait.pipeline_tag in tags
        )
        source += f"""
float fmha_fwd_{name}_for_test(fmha_fwd_traits t, fmha_fwd_args a,
                             const ck_tile::stream_config& s) {{
    return {name}_only(t, a, s);
}}
"""
    if not args.output.exists() or args.output.read_text() != source:
        args.output.write_text(source)


if __name__ == "__main__":
    main()
