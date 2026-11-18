# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate native dilated 2-D Winograd descriptor bundles; no runtime Python dependency."""
import json
import pathlib
import sys
import uuid

NS = uuid.UUID("e2f8ce56-840f-4e9d-9d9c-64055de4d56f")


def generate(out):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)

    def uid(name):
        return str(uuid.uuid5(NS, "winograd_nd." + name))

    def emit(name, data):
        (out / name).write_text(
            json.dumps(dict(version="1.0", **data), indent=2) + "\n"
        )

    fields = [
        "direction",
        "dtype",
        "arch",
        "atom",
        "precision",
        "gemm_bm",
        "gemm_bn",
        "gemm_bk",
        "chunk_tiles",
    ]
    knobs = fields[3:]
    emit(
        "conv_winograd_nd.kmd.json",
        dict(
            id=uid("kmd"),
            name="winograd_nd configuration",
            fields=[
                dict(name=f, type="string" if f in fields[:3] else "int")
                for f in fields
            ],
        ),
    )
    emit(
        "kernel_matches_graph.umd.json",
        dict(
            id=uid("umd"),
            name="winograd_nd applicability",
            scope="kernel",
            match_symbol="hipkernel.conv_winograd_nd.kernel_match",
        ),
    )
    emit(
        "conv_winograd_nd.udd.json",
        dict(
            id=uid("udd"),
            name="winograd_nd dispatch",
            dispatch_symbol="hipkernel.conv_winograd_nd.dispatch",
        ),
    )
    for direction, tag in (("FWD", "fwd"),):
        emit(
            f"conv_winograd_nd_{tag}.ued.json",
            dict(
                id=uid("ued." + tag),
                name="hipkernel:CONV_WINOGRAD_ND_" + direction,
                graph_match=dict(
                    native=f"hipkernel.conv_winograd_nd_{tag}.graph_match"
                ),
                metadata=uid("kmd"),
                knobs=knobs,
                behavior_notes=["runtime_compilation"],
                numerical_notes=["winograd", "winograd_tile_4x4"],
            ),
        )
        kernels = []
        for dtype in ("FLOAT",):
            for atom, arch, bm in ((5, "gfx1250", 64), (5, "gfx1250", 128)):
                for chunk in (256, 4096):
                    name = f"conv_winograd_nd.{tag}.{dtype.lower()}.residue2d.fp32.atom{atom}.m{bm}.chunk{chunk}"
                    metadata = dict(
                        direction=direction,
                        dtype=dtype,
                        arch=arch,
                        atom=atom,
                        precision=0,
                        gemm_bm=bm,
                        gemm_bn=64,
                        gemm_bk=16,
                        chunk_tiles=chunk,
                    )
                    kernels.append(
                        dict(
                            version="1.0",
                            id=uid(name),
                            name=name,
                            kernel_source=dict(
                                kind="embedded_source",
                                source_file="kernels/ConvWinogradNd.cpp",
                                entry_point="ConvWinogradNdGemm",
                            ),
                            metadata=metadata,
                            priority=1000 - len(kernels),
                        )
                    )
        emit(
            f"conv_winograd_nd_{tag}.kdp.json",
            dict(
                id=uid("kdp." + tag),
                name="hipkernel:conv_winograd_nd_" + tag,
                arch=[],
                matchers=[uid("umd")],
                engine=uid("ued." + tag),
                dispatch=uid("udd"),
                kernelDescriptors=kernels,
            ),
        )


if __name__ == "__main__":
    generate(sys.argv[1])
