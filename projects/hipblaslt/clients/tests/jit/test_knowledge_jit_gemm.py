# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate and build tuned knowledge seeds for an architecture without its GPU.

Uses the build's knowledge file for the architecture, or extracts one from the
logic files when the build has none. The knowledge test's --nearest mode runs
the C++ matcher for the request's plain GEMM on a device of the given CU count
without a PCI chip ID. The request's candidates become those seeds under
tensilelite.tuned.v1, and every distinct kernel among them is generated with the
arguments hipBLASLt passes and built with comgr. Nothing is executed.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from test_gfx1250_jit_gemm import generate_and_build, require

LOGIC = {"gfx942": "aquavanjaram", "gfx1250": "gfx1250"}
SIZE_ASSERTS = {
    "AssertFree0ElementMultiple": "m",
    "AssertFree1ElementMultiple": "n",
    "AssertSummationElementMultiple": "k",
}


def knowledge_file(build, logic, architecture, output):
    name = f"hipblaslt-jit-knowledge-{architecture}.dat.zlib"
    built = build / "Tensile/library" / architecture / name
    if built.is_file():
        return built
    extracted = output / name
    with (output / "extract.log").open("w") as log:
        status = subprocess.run(
            [sys.executable, "-m", "Tensile.JitKnowledge", str(logic / LOGIC[architecture]),
             str(extracted), "--architecture", architecture],
            stdout=log, stderr=subprocess.STDOUT, timeout=900,
        ).returncode
    require(status == 0, f"Tensile.JitKnowledge exited with {status}; see {log.name}")
    return extracted


def candidate(identifier, seed, problem):
    parameters = dict(seed["parameters"])
    for name, value in seed["asserts"].items():
        if name in SIZE_ASSERTS and value > 0 and problem[SIZE_ASSERTS[name]] % value == 0:
            parameters[name] = value
    return {
        "id": identifier,
        "predicted_cycles": None,
        "parameters": parameters,
        "modeled": {
            "contract": "tensilelite.tuned.v1",
            "macro_tile": [*seed["macro_tile"], seed["depth_u"]],
            "execution": {"strategy": seed["strategy"], "assignment": seed["assignment"]},
        },
        "knowledge": {key: seed[key] for key in ("branch", "source", "row", "distance")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("knowledge_test", type=Path)
    parser.add_argument("code_object_test", type=Path)
    parser.add_argument("request", type=Path, help="a heuristic request of a plain GEMM")
    parser.add_argument("cxx_compiler")
    parser.add_argument("architecture", choices=sorted(LOGIC))
    parser.add_argument("cu_count", type=int)
    parser.add_argument("build", type=Path)
    parser.add_argument("logic", type=Path, help="the asm_full logic directory")
    parser.add_argument("fresh_output", type=Path)
    parser.add_argument("--transpose-b", action="store_true",
                        help="use the request's GEMM with B transposed")
    args = parser.parse_args()
    args.fresh_output.mkdir(parents=True, exist_ok=False)
    output = args.fresh_output.resolve()
    from Tensile.JitKnowledge import CORE, coreKey

    database = knowledge_file(args.build, args.logic, args.architecture, output)
    decode = subprocess.run([str(args.knowledge_test), "--decode", str(database)],
                            text=True, capture_output=True, timeout=300)
    require(decode.returncode == 0, f"--decode failed: {decode.stdout}{decode.stderr}")
    print(decode.stdout.strip(), flush=True)
    request = json.loads(args.request.read_text())
    problem, problemType = request["problem"], request["problem_type"]
    if args.transpose_b:
        require(not problem["transpose_b"], "B is already transposed")
        problem.update(transpose_b=True, sizes_b=[problem["n"], problem["k"], 1],
                       strides_b=[1, problem["n"], 0])
        problemType["TransposeB"] = True
    core = coreKey({name: problemType[name] for name in CORE})
    nearest = subprocess.run(
        [str(args.knowledge_test), "--nearest", str(database), core,
         *(str(problem[key]) for key in ("m", "n", "batch", "k")), str(args.cu_count)],
        text=True, capture_output=True, timeout=120,
    )
    (output / "seeds.jsonl").write_text(nearest.stdout + nearest.stderr)
    require(nearest.returncode == 0, f"No seeds for {core}; see seeds.jsonl")
    seeds = [json.loads(line) for line in nearest.stdout.splitlines()]
    request.update(
        architecture=args.architecture,
        candidates=[candidate(index, seed, problem) for index, seed in enumerate(seeds)],
        requested_solutions=len(seeds),
    )
    path = output / "request.json"
    path.write_text(json.dumps(request, indent=1))

    manifests = generate_and_build(
        args.code_object_test, path, args.cxx_compiler, output, args.architecture)
    require(manifests, "No tuned seed was generated")
    for manifest in manifests:
        require(manifest["jit_prediction"]["modeled_contract"] == "tensilelite.tuned.v1",
                f"{manifest['main_kernel']['name']} is not a tuned seed")
    # Seeds that differ only in run-time arguments share a kernel.
    rejections = manifests[-1]["jit_prediction"]["rejections"]
    covered = {m["jit_prediction"]["candidate_id"] for m in manifests} | {
        r["candidate_id"] for r in rejections if r["reason"].startswith("Same kernel as")}
    require(covered == set(range(len(seeds))),
            f"Seeds {sorted(set(range(len(seeds))) - covered)} were not generated: {rejections}")
    splitK = sum(seed["gsu"] == -1 for seed in seeds)
    if args.architecture == "gfx942":
        require(2 * splitK > len(seeds), f"Only {splitK} of {len(seeds)} gfx942 seeds have GSU=-1")
    print(f"PASS jit-gemm-knowledge-{args.architecture}: {len(seeds)} seeds of "
          f"{seeds[0]['group']} ({splitK} with GSU=-1) built as {len(manifests)} kernels")


if __name__ == "__main__":
    main()
