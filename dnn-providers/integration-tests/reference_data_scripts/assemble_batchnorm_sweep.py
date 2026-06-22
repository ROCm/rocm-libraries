## Copyright © Advanced Micro Devices, Inc., or its affiliates.
## SPDX-License-Identifier: MIT

import argparse
import json
from pathlib import Path


CASE_SOURCES = {
    "small_fp32_nchw": Path("nchw/fp32/Small/Small"),
    "large_fp32_nchw": Path("nchw/fp32/Large/Large"),
    "miopen_fp32_nchw": Path("nchw/fp32/MIOpen/MIOpen"),
    "small_fp16_nchw": Path("nchw/fp16/Small/Small"),
    "small_bfp16_nchw": Path("nchw/bfp16/Small/Small"),
    "small_fp32_ncdhw": Path("ncdhw/fp32/Small/Small"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        prog="assemble_batchnorm_sweep",
        description=(
            "Build the RFC 0011 BatchnormFwdInference compressed sweep from the "
            "existing single-graph quick-tier bundles."
        ),
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        required=True,
        help=(
            "Path to quick/BatchnormFwdInference. The script reads the existing "
            "single-graph bundles under this root and writes Inference/."
        ),
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def rewrite_dvc_paths(dvc_text: str) -> str:
    rewritten_lines = []
    for line in dvc_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- path: "):
            tensor_suffix = stripped.split(".tensor", 1)[1]
            line = "- path: tensor" + tensor_suffix
        rewritten_lines.append(line)
    return "\n".join(rewritten_lines) + "\n"


def make_template_graph(source_graph):
    template = json.loads(json.dumps(source_graph))
    template["io_data_type"] = "${case.io_data_type}"
    for tensor in template["tensors"]:
        tensor["dims"] = "${case.dims}"
        tensor["strides"] = "${case.strides}"
        tensor["data_type"] = "${case.data_type}"
    return template


def make_case_payload(case_id: str, graph, metadata):
    return {
        "id": case_id,
        "values": {
            "io_data_type": graph["io_data_type"],
            "tensors": [
                {
                    "uid": tensor["uid"],
                    "data_type": tensor["data_type"],
                    "dims": tensor["dims"],
                    "strides": tensor["strides"],
                }
                for tensor in graph["tensors"]
            ],
        },
        "golden": {
            "id": case_id,
            "path": f"golden/{case_id}/tensors.dvc",
        },
        "metadata": metadata,
    }


def main():
    args = parse_args()
    bundle_root = args.bundle_root.resolve()
    output_root = bundle_root / "Inference"

    first_graph = load_json(
        bundle_root / CASE_SOURCES["small_fp32_nchw"].with_suffix(".json")
    )
    write_json(output_root / "graph.template.json", make_template_graph(first_graph))

    sweep = {"version": 1, "cases": []}
    for case_id, relative_base in CASE_SOURCES.items():
        source_base = bundle_root / relative_base
        graph = load_json(source_base.with_suffix(".json"))
        metadata = load_json(source_base.with_suffix(".meta.json"))
        sweep["cases"].append(make_case_payload(case_id, graph, metadata))

        dvc_path = source_base.with_name(source_base.name + ".tensors.dvc")
        dvc_text = dvc_path.read_text(encoding="utf-8")
        golden_dir = output_root / "golden" / case_id
        golden_dir.mkdir(parents=True, exist_ok=True)
        (golden_dir / "tensors.dvc").write_text(
            rewrite_dvc_paths(dvc_text), encoding="utf-8", newline="\n"
        )

    write_json(output_root / "sweep.json", sweep)


if __name__ == "__main__":
    main()
