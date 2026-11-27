#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Print RockeKernelSignaturesTable.inc.

Every row is one kernel argument list a rocKE attention pack launches, keyed
``<arch>.<family>[.<form>]``, as rocKE's own launch-signature functions give
it, in order: the name, the AMDGPU kernarg kind (``global_buffer`` for a
pointer, ``by_value`` for a scalar) and the width in bytes. rocKE's own tests
tie each function to the parameters its builder declares
(``library/tests/test_attention_tiled_launch_signature.py`` and
``library/tests/test_attention_builds.py``), so a row is the list the shipped
kernel takes. RockeKernelSignatures.hpp reads the table; the provider's
signature and dispatch tests check each pack's C++ list against it.

* unified 2D (gfx942, gfx950), without and with LSE:
  ``kernels.common.attention_unified._tiled_2d_signature``;
* unified 3D segment and reduce: ``_tiled_3d_signatures``, strided K/V on
  gfx942 (the only layout that pack launches), paged on gfx950;
* dense: each arch's ``attention_dense_signature`` for the runtime-shape grid
  form, without and with LSE, and on gfx942 for each persistent work decode.

A signature does not depend on the dtype, head size or problem shape, only on
the form, so one representative spec stands for each form.

The build generates the table into the build tree through gen_rocke_table.py
(kernel_ingestor_engine/CMakeLists.txt); it is never committed. To print it by
hand, from the provider root:

    cd rocke/library && PYTHONPATH=.:../platform/python python3 \
        ../../src/tests/engines/kernel_ingestor_engine/packs/gen_rocke_kernel_signatures.py
"""

import dataclasses

from kernels.common.attention_unified import _tiled_2d_signature, _tiled_3d_signatures
from kernels.gfx942.attention_dense import Gfx942AttentionDenseSpec
from kernels.gfx942.attention_dense import (
    attention_dense_signature as gfx942_dense_signature,
)
from kernels.gfx950.attention_dense import Gfx950AttentionDenseSpec
from kernels.gfx950.attention_dense import (
    attention_dense_signature as gfx950_dense_signature,
)

# AMDGPU kernarg kind and width of each scalar type a signature names.
SCALAR_WIDTHS = {"f32": 4, "i32": 4, "i64": 8}
POINTER_WIDTH = 8

# A runtime-shape dense problem. Four 256-row query blocks, so every gfx942
# persistent decode accepts it (hq_minor_swz needs an even count under a causal
# mask).
DENSE_SHAPE = {
    "batch": 2,
    "seqlen_q": 1024,
    "seqlen_kv": 1024,
    "num_query_heads": 8,
    "num_kv_heads": 8,
    "head_size": 128,
}


def kernarg(parameter: dict) -> tuple[str, int, str]:
    """(kind, width, name) of one signature parameter."""
    name, parameter_type = parameter["name"], parameter["type"]
    if parameter_type.startswith("ptr<"):
        return "global_buffer", POINTER_WIDTH, name
    if parameter_type not in SCALAR_WIDTHS:
        raise ValueError(
            f"parameter {name!r} has a type with no kernarg width: {parameter_type}"
        )
    return "by_value", SCALAR_WIDTHS[parameter_type], name


def unified_signatures():
    for arch in ("gfx942", "gfx950"):
        yield f"{arch}.unified_2d", _tiled_2d_signature(arch, "bf16")
        yield f"{arch}.unified_2d.lse", _tiled_2d_signature(arch, "bf16", emit_lse=True)
    for arch, strided_kv in (("gfx942", True), ("gfx950", False)):
        segment, reduce = _tiled_3d_signatures(arch, "bf16", strided_kv=strided_kv)
        yield f"{arch}.unified_3d.segment", segment
        yield f"{arch}.unified_3d.reduce", reduce


def dense_signatures():
    for arch, spec_class, signature in (
        ("gfx942", Gfx942AttentionDenseSpec, gfx942_dense_signature),
        ("gfx950", Gfx950AttentionDenseSpec, gfx950_dense_signature),
    ):
        grid = spec_class(**DENSE_SHAPE)
        assert (
            grid.runtime_shape
        ), f"the {arch} dense grid form is no longer runtime-shape"
        yield f"{arch}.dense", signature(grid)
        yield f"{arch}.dense.lse", signature(dataclasses.replace(grid, emit_lse=True))
    grid = Gfx942AttentionDenseSpec(**DENSE_SHAPE)
    for decode in sorted(
        Gfx942AttentionDenseSpec.supported_persist_decodes() - {"auto"}
    ):
        persistent = dataclasses.replace(grid, persistent=True, persist_decode=decode)
        for suffix, emit_lse in (("", False), (".lse", True)):
            spec = dataclasses.replace(persistent, emit_lse=emit_lse)
            yield f"gfx942.dense.persistent.{decode}{suffix}", gfx942_dense_signature(
                spec
            )


def main():
    for key, signature in (*unified_signatures(), *dense_signatures()):
        arguments = ",\n  ".join(
            f'{{"{kind}", {width}, 0, "{name}"}}'
            for kind, width, name in map(kernarg, signature)
        )
        print(f'{{"{key}",\n {{\n  {arguments}}}}},')


if __name__ == "__main__":
    main()
