# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Author a bounded dense SDPA recipe offline; runtime needs only the CBOR."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

# The library authoring package is deliberately absent from deployed runtime inputs.
_rocke = Path(__file__).resolve().parents[5] / "rocke"
sys.path[:0] = [str(_rocke / "platform" / "python"), str(_rocke / "library")]

from kernels.gfx942 import attention_dense as gfx942
from kernels.gfx950 import attention_dense as gfx950
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.portable_ir.src.guard import attach_guard, axis_rules
from rocke.portable_ir.src.launch import attach_launch, plan
from rocke.portable_ir.src.recipe_bundle import build_bundle, cbor_encode
from rocke.portable_ir.src.roll import roll


SAMPLES = [512, 1024]
HOLDOUTS = [768]
KEY = "sdpa_dense_bf16_d128_causal"


# Architecture-specific code stays in the builders. This registry is the supported
# target set for this bounded example, not a promise of arbitrary GPU support.
BUILDERS = {
    "gfx942": (gfx942, gfx942.Gfx942AttentionDenseSpec),
    "gfx950": (gfx950, gfx950.AttentionDenseSpec),
}


def selected_arches(targets: list[str]) -> list[str]:
    arches = sorted({target.split(":", 1)[0] for target in targets})
    if not arches or any(arch not in BUILDERS for arch in arches):
        raise ValueError(
            f"unsupported SDPA targets {targets}; supported: {', '.join(BUILDERS)}"
        )
    return arches


def make_spec(arch: str, sequence: int):
    return BUILDERS[arch][1](
        batch=1,
        seqlen_q=sequence,
        seqlen_kv=sequence,
        num_query_heads=4,
        num_kv_heads=4,
        head_size=128,
        dtype="bf16",
        causal=True,
        block_m=256,
        block_n=64,
        persistent=False,
    )


def produce_target(output: Path, arch: str, llvm_flavor: str):
    builder = BUILDERS[arch][0]

    def build(sequence: int):
        spec = make_spec(arch, sequence)
        supported, reason = builder.supports_attention_dense(spec, arch=arch)
        if not supported:
            raise ValueError(f"{arch}: {reason}")
        return builder.build_attention_dense(spec, arch=arch)

    output.mkdir(parents=True, exist_ok=True)
    first = make_spec(arch, SAMPLES[0])
    result = roll(
        build,
        axis="S",
        sample_points=SAMPLES,
        holdout_points=HOLDOUTS,
        spec_decl=[{"name": "S", "kind": "int"}],
        name_fmt=first.kernel_name().replace(str(SAMPLES[0]), "{S}"),
    )
    if not result.ok:
        (output / "refusal.json").write_text(
            json.dumps({"reason": result.reason}, indent=2) + "\n"
        )
        raise RuntimeError(f"{arch}: {result.reason}")
    guard = {
        "schema": "rocke.guard/v1",
        "free": ["S"],
        "rules": axis_rules("S", sorted(SAMPLES + HOLDOUTS)),
        "verified": [],
    }
    # This formula is specific to the bounded nonpersistent query-block mapping.
    # Check every admitted shape against the builder's public geometry below.
    recipe = attach_launch(
        attach_guard(result.recipe, guard),
        grid=[
            {"div": [{"add": [{"spec": "S"}, first.block_m - 1]}, first.block_m]},
            first.num_query_heads,
            first.batch,
        ],
        block=list(builder.attention_dense_block(first)),
    )
    short_recipe = attach_guard(recipe, {**guard, "rules": axis_rules("S", [512])})
    references = {}
    for sequence in sorted(SAMPLES + HOLDOUTS):
        spec = make_spec(arch, sequence)
        launch = plan(recipe, {"S": sequence})
        expected_grid = builder.attention_dense_grid(spec)
        expected_block = builder.attention_dense_block(spec)
        if (launch["geometry"]["grid"], launch["geometry"]["block"]) != (
            expected_grid,
            expected_block,
        ):
            raise ValueError(f"{arch} S={sequence}: recipe/builder geometry mismatch")
        expected_signature = [
            (arg["name"], "pointer" if arg["type"].startswith("ptr<") else arg["type"])
            for arg in builder.attention_dense_signature(spec)
        ]
        actual_signature = [(arg["name"], arg["kind"]) for arg in launch["args"]]
        if actual_signature != expected_signature:
            raise ValueError(f"{arch} S={sequence}: recipe/builder signature mismatch")
        llvm = lower_kernel_to_llvm(build(sequence), arch=arch, llvm_flavor=llvm_flavor)
        (output / f"reference-{sequence}.ll").write_text(llvm)
        # A simple numeric oracle keeps the standalone native test JSON-free.
        geometry = (*expected_grid, *expected_block, launch["geometry"]["lds_bytes"])
        (output / f"reference-{sequence}.launch").write_text(
            " ".join(map(str, geometry)) + "\n"
        )
        references[str(sequence)] = {
            "llvm_sha256": hashlib.sha256(llvm.encode()).hexdigest(),
            "launch": launch,
        }
    return [
        {"key": KEY, "arch": arch, "recipe": recipe},
        {"key": KEY + "_short", "arch": arch, "recipe": short_recipe},
    ], {
        "fixed_spec_at_first_sample": asdict(first),
        "references": references,
        "guard": guard,
    }


def produce(output: Path, targets: list[str], llvm_flavor: str) -> dict:
    arches = selected_arches(targets)
    entries, manifests = [], {}
    output.mkdir(parents=True, exist_ok=True)
    for arch in arches:
        target_entries, manifests[arch] = produce_target(
            output / arch, arch, llvm_flavor
        )
        entries.extend(target_entries)
    bundle = cbor_encode(build_bundle(entries))
    descriptor = json.loads(
        (Path(__file__).parent / "descriptors" / "sdpa.kdp.json.in").read_text()
    )
    descriptor["arch"] = arches
    (output / "sdpa_dense.cbor").write_bytes(bundle)
    (output / "sdpa.kdp.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    manifest = {
        "samples": SAMPLES,
        "holdouts": HOLDOUTS,
        "key": KEY,
        "targets": manifests,
        "llvm_flavor": llvm_flavor,
        "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arches", nargs="+", required=True)
    parser.add_argument("--llvm-flavor", required=True)
    args = parser.parse_args()
    try:
        manifest = produce(args.output, args.arches, args.llvm_flavor)
    except ValueError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "producer": "PASS",
                "targets": list(manifest["targets"]),
                "bundle_sha256": manifest["bundle_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
