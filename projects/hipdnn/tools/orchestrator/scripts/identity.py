#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""One source of truth for every string a UED engine name expands into.

A single engine name such as `hipkernel:BatchnormRtc` has to turn, consistently,
into roughly eight different spellings scattered across the checkout: a pack
class name, a descriptor directory, a registration function, a native source
file, an engine TOML filename, a CMake external-integration-test target, the
census/matcher gtest suite names the generator emits, and the literal string a
human has to add to the pack table. The kernel-authoring agent, the
kernel-integration agent, the flow YAML that wires their outputs together, and
the `--test-engine` / `--expect-engine` arguments the ingestor and validator
gates pass around, each need a subset of that list. If every consumer derives
its own spelling by hand, the run is one typo away from building
`BatchNormRtc` while testing `Batchnorm_Rtc` -- two names that pass their own
narrow check and validate nothing together.

So this script derives the full set once, from the one name an agent actually
chose, and every other step reads the JSON instead of re-deriving anything.
It also probes the checkout for collisions: a "new" engine whose descriptor
directory, native source, TOML, or pack-table entry already exists is not new,
and a create-path run must fail on that fact before an agent spends a step
writing files nobody asked it to overwrite.

    identity.py --engine-name hipkernel:BatchnormRtc --repo REPO --out identity.json

Exit codes: 0 the report was written (a nonzero `collisions` is data for the
flow to assert on, not a script failure), 2 the engine name fails the
generator's own `NAMESPACE:Local` shape.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: The generator's own rule for a legal engine name: exactly one `:` separating
#: two non-empty, uri-safe segments. An engine name that does not fit this is
#: rejected by the generator itself long before any of the derivations below
#: would matter.
_ENGINE_NAME_RE = re.compile(r"^([A-Za-z0-9_.-]+):([A-Za-z0-9_.-]+)$")

#: The provider's own project() name (`dnn-providers/hip-kernel-provider/CMakeLists.txt`),
#: hyphenated rather than the underscored form its own PLUGIN_TARGET arguments use.
#: `add_external_integration_test_target(TARGET_NAME ${PROJECT_NAME}-...)` therefore
#: expands with hyphens throughout (e.g. `hip-kernel-provider-hipkernel-conv-fwd-
#: external-integration-check`), confirmed against the real ConvFwd/Pointwise
#: invocations in that CMakeLists.txt. A formula built on the underscored spelling
#: would name a target that does not exist.
_PROJECT_NAME = "hip-kernel-provider"


def _snake_case(local: str) -> str:
    """`BatchnormRtc` -> `batchnorm_rtc`. Split before an upper-case letter that
    follows a lower-case letter or digit, then lowercase. This is the same rule
    the generator's own descriptor-directory naming follows (`ConvFwd` ->
    `conv_fwd`, `Pointwise` -> `pointwise`), confirmed against the shipped
    descriptor directories."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", local).lower()


def _pack_class(local: str) -> str:
    """`local` with its first character upper-cased and every non-alphanumeric
    character dropped. Anchoring on `local` (not a re-lowered/re-split form)
    keeps an already-PascalCase name like `BatchnormRtc` untouched."""
    upped = local[0].upper() + local[1:] if local else local
    return "".join(ch for ch in upped if ch.isalnum())


def _derive(engine_name: str, repo: Path) -> dict:
    match = _ENGINE_NAME_RE.match(engine_name)
    if match is None:
        raise ValueError(
            f"engine name {engine_name!r} does not match NAMESPACE:Local "
            f"({_ENGINE_NAME_RE.pattern})"
        )
    namespace, local = match.group(1), match.group(2)

    pack_class = _pack_class(local)
    descriptor_slug = _snake_case(local)
    kebab_local = descriptor_slug.replace("_", "-")

    engines_dir = (
        repo / "dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine"
    )
    native_file = engines_dir / "packs" / f"{pack_class}Native.cpp"
    descriptor_dir = engines_dir / "descriptors" / descriptor_slug
    packs_table = engines_dir / "IngestorPacks.cpp"
    engine_toml = (
        repo / "dnn-providers/hip-kernel-provider/config" / f"{namespace}_{local}.toml"
    )
    generated_tests_dir = (
        repo
        / "dnn-providers/hip-kernel-provider/src/tests/engines/kernel_ingestor_engine/packs"
    )
    # Every generated test source this engine owns. The suites inside them all begin with
    # `Test<PackClass>`, which is what makes one glob a complete filter; the count is what
    # lets a gate notice that one of these files contributed no suite to a census run.
    generated_test_sources = sorted(
        path.name for path in generated_tests_dir.glob(f"Test{pack_class}*.cpp")
    )

    descriptor_dir_exists = int(descriptor_dir.is_dir())
    native_file_exists = int(native_file.is_file())
    engine_toml_exists = int(engine_toml.is_file())
    packs_table_mentions = int(
        packs_table.is_file()
        and f'"{engine_name}"'
        in packs_table.read_text(encoding="utf-8", errors="replace")
    )
    collisions = (
        descriptor_dir_exists
        + native_file_exists
        + engine_toml_exists
        + packs_table_mentions
    )

    feedback_lines: list[str] = []
    if descriptor_dir_exists:
        feedback_lines.append(
            f"descriptor directory already exists: {descriptor_dir.as_posix()}"
        )
    if native_file_exists:
        feedback_lines.append(
            f"native pack source already exists: {native_file.as_posix()}"
        )
    if engine_toml_exists:
        feedback_lines.append(f"engine TOML already exists: {engine_toml.as_posix()}")
    if packs_table_mentions:
        feedback_lines.append(
            f'"{engine_name}" is already registered in {packs_table.as_posix()}'
        )
    feedback = (
        ""
        if collisions == 0
        else "A create-path run must choose a name no installed identity already uses. "
        + "; ".join(feedback_lines)
    )

    return {
        "engine_name": engine_name,
        "namespace": namespace,
        "local": local,
        "pack_class": pack_class,
        "descriptor_slug": descriptor_slug,
        "register_symbol": f"register{pack_class}Symbols",
        "native_file": native_file.as_posix(),
        "descriptor_dir": descriptor_dir.as_posix(),
        "engine_toml": engine_toml.as_posix(),
        "external_test_target": f"{_PROJECT_NAME}-{namespace}-{kebab_local}-external-integration-check",
        # Every suite the engine's generated sources define, not one shard of them. A
        # filter naming a single suite leaves its siblings built, linked and never run:
        # a test file that cannot even start -- gtest rejects a suite mixing TEST with a
        # parameterized fixture at runtime -- then reports nothing to any gate.
        "census_filter": f"Test{pack_class}*.*",
        "generated_test_sources": generated_test_sources,
        "generated_test_source_count": len(generated_test_sources),
        "descriptor_dir_exists": descriptor_dir_exists,
        "native_file_exists": native_file_exists,
        "engine_toml_exists": engine_toml_exists,
        "packs_table_mentions": packs_table_mentions,
        "collisions": collisions,
        "feedback": feedback,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine-name",
        required=True,
        help="UED engine name, e.g. hipkernel:BatchnormRtc",
    )
    parser.add_argument("--repo", required=True, help="checkout root")
    parser.add_argument("--out", required=True, help="where to write the JSON report")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        print(f"error: --repo is not a directory: {repo}", file=sys.stderr)
        return 2

    try:
        report = _derive(args.engine_name, repo)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"{report['engine_name']}: pack_class={report['pack_class']} "
        f"descriptor_slug={report['descriptor_slug']} collisions={report['collisions']} "
        f"-> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
