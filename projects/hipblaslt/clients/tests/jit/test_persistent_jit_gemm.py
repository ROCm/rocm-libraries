# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate and build Origami's Stream-K candidates for an architecture without its GPU.

The knowledge test's --predict mode ranks the catalog for the request's FP16 NN
GEMM on a device of the architecture with the given CU count. The request keeps
the origami.gemm.persistent.v1 candidates in their order, and the first two that
Tensile accepts are generated with the arguments hipBLASLt passes and built with
comgr. Nothing is executed.
"""

import argparse
import json
from pathlib import Path
import subprocess

from test_gfx1250_jit_gemm import generate_and_build, require

PERSISTENT = "origami.gemm.persistent.v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("knowledge_test", type=Path)
    parser.add_argument("code_object_test", type=Path)
    parser.add_argument("request", type=Path, help="a heuristic request of an FP16 NN GEMM")
    parser.add_argument("cxx_compiler")
    parser.add_argument("architecture", choices=("gfx942", "gfx1250"))
    parser.add_argument("cu_count", type=int)
    parser.add_argument("fresh_output", type=Path)
    args = parser.parse_args()
    args.fresh_output.mkdir(parents=True, exist_ok=False)
    output = args.fresh_output.resolve()
    request = json.loads(args.request.read_text())
    problem = request["problem"]
    predicted = subprocess.run(
        [str(args.knowledge_test), "--predict", args.architecture, str(args.cu_count),
         *(str(problem[key]) for key in ("m", "n", "k"))],
        text=True, capture_output=True, timeout=120,
    )
    (output / "predicted.jsonl").write_text(predicted.stdout + predicted.stderr)
    require(predicted.returncode == 0, "No ranking; see predicted.jsonl")
    ranked = [json.loads(line) for line in predicted.stdout.splitlines()]
    persistent = [c for c in ranked if c["modeled"]["contract"] == PERSISTENT]
    require(persistent and len(persistent) < len(ranked),
            "Expected data-parallel and Stream-K candidates; see predicted.jsonl")
    request.update(architecture=args.architecture, candidates=persistent, requested_solutions=2)
    path = output / "request.json"
    path.write_text(json.dumps(request, indent=1))

    manifests = generate_and_build(
        args.code_object_test, path, args.cxx_compiler, output, args.architecture)
    require(len(manifests) == 2, f"{len(manifests)} of 2 Stream-K solutions generated")
    for manifest in manifests:
        prediction = manifest["jit_prediction"]
        resolved = prediction["resolved_parameters"]
        require(prediction["modeled_contract"] == PERSISTENT
                and resolved["TileProcessingStrategy"] == "StreamK"
                and resolved["WorkAssignment"] == "Hybrid"
                and resolved["WorkGroupMapping"] == 0 and resolved["WorkGroupMappingXCC"] == -1,
                f"{manifest['main_kernel']['name']} is not a Hybrid Stream-K kernel"
                " that the runtime maps")
    rejected = len(manifests[-1]["jit_prediction"]["rejections"])
    print(f"PASS jit-gemm-persistent-{args.architecture}: {len(persistent)} of {len(ranked)}"
          f" ranked candidates are Stream-K; two built, {rejected} rejected before them")


if __name__ == "__main__":
    main()
