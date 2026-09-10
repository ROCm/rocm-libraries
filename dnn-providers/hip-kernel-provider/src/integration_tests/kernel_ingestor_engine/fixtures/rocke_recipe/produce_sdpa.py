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

from kernels.gfx950.attention_dense import (
    AttentionDenseSpec,
    build_attention_dense,
    supports_attention_dense,
)
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.portable_ir.src.guard import attach_guard, axis_rules
from rocke.portable_ir.src.launch import attach_launch
from rocke.portable_ir.src.recipe_bundle import build_bundle, cbor_encode
from rocke.portable_ir.src.roll import roll


SAMPLES = [512, 1024]
HOLDOUTS = [768]
KEY = "sdpa_dense_bf16_d128_causal"


def make_spec(sequence: int) -> AttentionDenseSpec:
    return AttentionDenseSpec(
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


def build(sequence: int):
    spec = make_spec(sequence)
    supported, reason = supports_attention_dense(spec, arch="gfx950")
    if not supported:
        raise ValueError(reason)
    return build_attention_dense(spec, arch="gfx950")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = roll(
        build,
        axis="S",
        sample_points=SAMPLES,
        holdout_points=HOLDOUTS,
        spec_decl=[{"name": "S", "kind": "int"}],
        name_fmt=make_spec(512).kernel_name().replace("512", "{S}"),
    )
    if not result.ok:
        (args.output / "refusal.json").write_text(
            json.dumps({"reason": result.reason}, indent=2) + "\n"
        )
        raise RuntimeError(result.reason)
    guard = {
        "schema": "rocke.guard/v1",
        "free": ["S"],
        "rules": axis_rules("S", sorted(SAMPLES + HOLDOUTS)),
        "verified": [],
    }
    recipe = attach_launch(
        attach_guard(result.recipe, guard),
        grid=[{"div": [{"spec": "S"}, 256]}, 4, 1],
        block=[512, 1, 1],
    )
    # A preferred candidate with a narrower domain exercises preselection refusal.
    short_guard = {**guard, "rules": axis_rules("S", [512])}
    short_recipe = attach_guard(recipe, short_guard)
    bundle = cbor_encode(
        build_bundle(
            [
                {"key": KEY, "arch": "gfx950", "recipe": recipe},
                {"key": KEY + "_short", "arch": "gfx950", "recipe": short_recipe},
            ]
        )
    )
    (args.output / "sdpa_dense.cbor").write_bytes(bundle)
    references = {}
    for sequence in sorted(SAMPLES + HOLDOUTS):
        llvm = lower_kernel_to_llvm(
            build(sequence), arch="gfx950", llvm_flavor="llvm23"
        )
        (args.output / f"reference-{sequence}.ll").write_text(llvm)
        references[str(sequence)] = hashlib.sha256(llvm.encode()).hexdigest()
    manifest = {
        "samples": SAMPLES,
        "holdouts": HOLDOUTS,
        "fixed_spec_at_first_sample": asdict(make_spec(SAMPLES[0])),
        "key": KEY,
        "target": "gfx950",
        "llvm_flavor": "llvm23",
        "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
        "references": references,
        "guard": guard,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"producer": "PASS", "bundle_sha256": manifest["bundle_sha256"]}))


if __name__ == "__main__":
    main()
