# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Renders a full descriptor bundle for one ``IngestorConfig``.

Descriptor JSON is built as plain Python dicts and serialized with
``json.dumps`` -- not Jinja2 -- because every key emitted must be on that
type's allow-list (Knowledge/hipdnn/ingestor/02-descriptor-format.md:
"unknown keys are a hard error"), and a dict literal makes "every key this
function writes is a key I chose" a property of the code, not of template
whitespace. Jinja2 (``keep_trailing_newline``/``trim_blocks``/
``lstrip_blocks``, ``undefined=StrictUndefined``) renders the C++ stub/test
files and the CMake/registration text fragments, where hand-tuned
whitespace against ``.clang-format`` actually matters.

UUIDs are minted exactly once per run, in :func:`mint_ids`, and threaded
through every cross-reference from that one dict -- never retyped.
"""

import json
import uuid
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .models import IngestorConfig, PackSpec

#: Two-line AMD copyright + SPDX header every emitted C++/CMake file opens with.
CPP_COPYRIGHT_HEADER = (
    "// Copyright \u00a9 Advanced Micro Devices, Inc., or its affiliates.\n"
    "// SPDX-License-Identifier:  MIT\n"
)
CMAKE_COPYRIGHT_HEADER = (
    "# Copyright \u00a9 Advanced Micro Devices, Inc., or its affiliates.\n"
    "# SPDX-License-Identifier:  MIT\n"
)


def mint_ids(config: IngestorConfig) -> dict:
    """Mint every UUID this bundle needs, once, in one dict.

    Every other function in this module reads ids from this dict rather
    than calling ``uuid.uuid4()`` itself -- the single-mint-point AC #4
    requires.
    """
    ids = {
        "kmd": str(uuid.uuid4()),
        "ued": str(uuid.uuid4()),
        "kernel_match": str(uuid.uuid4()),
    }
    if config.engine.has_heuristic:
        ids["uhd"] = str(uuid.uuid4())
    # One UDD per engine (the design's invariant skeleton: "always ... >=1 UDD").
    ids["udd"] = str(uuid.uuid4())
    for pack in config.packs:
        ids[("pack", pack.name)] = str(uuid.uuid4())
        if config.is_multi_pack:
            ids[("operation_umd", pack.name)] = str(uuid.uuid4())
        for kernel in pack.kernels:
            ids[("kernel", pack.name, kernel.name)] = str(uuid.uuid4())
    return ids


def _dump(obj: dict) -> str:
    return json.dumps(obj, indent=2, sort_keys=False) + "\n"


def build_kmd(config: IngestorConfig, ids: dict) -> dict:
    fields = []
    for f in config.kmd_fields:
        entry = {"name": f.name, "type": f.type}
        if not f.is_mandatory:
            entry["default_value"] = f.default_value
        fields.append(entry)
    return {
        "version": "1.0",
        "id": ids["kmd"],
        "name": f"{config.engine.local_name} variant fields",
        "fields": fields,
    }


def build_uhd(config: IngestorConfig, ids: dict) -> dict | None:
    if not config.engine.has_heuristic:
        return None
    return {
        "version": "1.0",
        "id": ids["uhd"],
        "name": f"{config.engine.local_name} selector",
        "kind": "native",
        "payload": config.score_symbol,
    }


def build_ued(config: IngestorConfig, ids: dict) -> dict:
    ued = {
        "version": "1.0",
        "id": ids["ued"],
        "name": config.engine.name,
        "graph_match": {"native": config.graph_match_symbol},
        "metadata": ids["kmd"],
    }
    if config.engine.has_heuristic:
        ued["heuristic"] = ids["uhd"]
    if config.engine.knobs:
        ued["knobs"] = list(config.engine.knobs)
    if config.engine.behavior_notes:
        ued["behavior_notes"] = list(config.engine.behavior_notes)
    return ued


def build_udd(config: IngestorConfig, ids: dict) -> dict:
    return {
        "version": "1.0",
        "id": ids["udd"],
        "name": f"{config.engine.local_name} dispatch",
        "dispatch_symbol": config.dispatch_symbol,
    }


def build_kernel_match_umd(config: IngestorConfig, ids: dict) -> dict:
    """The shared kernel-scoped dtype matcher -- always emitted, one per
    engine, referenced by every pack's KDP. Not a per-pack narrowing (every
    pack shares it identically), but a genuine per-*kernel* applicability
    check the UED's graph_match cannot express (it has no kernel in scope)."""
    return {
        "version": "1.0",
        "id": ids["kernel_match"],
        "name": "kernel dtype matches the graph's dtype",
        "scope": "kernel",
        "match_symbol": config.kernel_match_symbol,
    }


def build_operation_umd(
    config: IngestorConfig, pack: PackSpec, ids: dict
) -> dict | None:
    """UMD policy: emitted only for genuine per-pack narrowing, i.e. only
    when the engine has more than one pack. A single-pack engine gets zero
    graph-scoped UMDs -- TestConvFwdPack.cpp asserts exactly this."""
    if not config.is_multi_pack:
        return None
    return {
        "version": "1.0",
        "id": ids[("operation_umd", pack.name)],
        "name": f"graph operation is {pack.discriminator}",
        "scope": "graph",
        "match_symbol": config.operation_match_symbol(pack),
    }


def build_kdp(config: IngestorConfig, pack: PackSpec, ids: dict) -> dict:
    matchers = [ids["kernel_match"]]
    if config.is_multi_pack:
        matchers.insert(0, ids[("operation_umd", pack.name)])
    kernel_descriptors = []
    for kernel in pack.kernels:
        entry = {
            "version": "1.0",
            "id": ids[("kernel", pack.name, kernel.name)],
            "name": kernel.name,
            "kernel_source": {
                "kind": kernel.kernel_source.kind,
                "source_file": kernel.kernel_source.source_file,
                "entry_point": kernel.kernel_source.entry_point,
            },
            "metadata": dict(kernel.metadata),
            "priority": kernel.priority,
        }
        if kernel.arch:
            entry["arch"] = list(kernel.arch)
        kernel_descriptors.append(entry)
    kdp = {
        "version": "1.0",
        "id": ids[("pack", pack.name)],
        "name": f"{config.engine.namespace}:{config.kdp_stem(pack)}",
        "matchers": matchers,
        "engine": ids["ued"],
        "dispatch": ids["udd"],
        "kernelDescriptors": kernel_descriptors,
    }
    if pack.arch:
        kdp["arch"] = list(pack.arch)
    return kdp


class IngestorGenerator:
    """Renders every file of one engine's descriptor bundle for a given
    :class:`IngestorConfig`, writing into ``output_dir``."""

    def __init__(self, template_dir: Path):
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
            # The one deliberate deviation from DescriptorGenerator: an
            # unset UUID cross-reference must fail loudly at generation
            # time, not render "" and fail later at the loader with a
            # message about an empty string instead of a missing field.
            undefined=StrictUndefined,
        )

    def preview_files(self, config: IngestorConfig) -> list[str]:
        """The file list :meth:`render` would write, without writing anything."""
        slug = config.engine.slug
        files = [
            f"descriptors/{slug}/{slug}.kmd.json",
            f"descriptors/{slug}/{slug}.ued.json",
            f"descriptors/{slug}/{slug}.udd.json",
        ]
        if config.engine.has_heuristic:
            files.append(f"descriptors/{slug}/{slug}.uhd.json")
        files.append(f"descriptors/{slug}/kernel_dtype_matches_graph.umd.json")
        for pack in config.packs:
            files.append(f"descriptors/{slug}/{config.kdp_stem(pack)}.kdp.json")
            if config.is_multi_pack:
                files.append(
                    f"descriptors/{slug}/operation_is_{pack.discriminator}.umd.json"
                )
        files.append(f"packs/{config.native_class_name}Native.cpp")
        files.append(f"tests/Test{config.engine.pascal_name}Packs.cpp")
        files.append(f"tests/Test{config.engine.pascal_name}Matchers.cpp")
        for fragment in FRAGMENT_FILENAMES:
            files.append(f"fragments/{fragment}")
        return files

    def render(self, config: IngestorConfig, output_dir: Path) -> list[str]:
        """Mint ids, write every descriptor JSON, the native/test C++ stubs,
        and the six CMake/registration fragments. Returns the list of
        relative paths written."""
        ids = mint_ids(config)
        written: list[str] = []
        slug = config.engine.slug
        descriptors_dir = output_dir / "descriptors" / slug
        descriptors_dir.mkdir(parents=True, exist_ok=True)

        def write_json(rel: str, obj: dict) -> None:
            path = output_dir / rel
            path.write_text(_dump(obj))
            written.append(rel)

        write_json(f"descriptors/{slug}/{slug}.kmd.json", build_kmd(config, ids))
        write_json(f"descriptors/{slug}/{slug}.ued.json", build_ued(config, ids))
        write_json(f"descriptors/{slug}/{slug}.udd.json", build_udd(config, ids))
        uhd = build_uhd(config, ids)
        if uhd is not None:
            write_json(f"descriptors/{slug}/{slug}.uhd.json", uhd)
        write_json(
            f"descriptors/{slug}/kernel_dtype_matches_graph.umd.json",
            build_kernel_match_umd(config, ids),
        )
        for pack in config.packs:
            write_json(
                f"descriptors/{slug}/{config.kdp_stem(pack)}.kdp.json",
                build_kdp(config, pack, ids),
            )
            op_umd = build_operation_umd(config, pack, ids)
            if op_umd is not None:
                write_json(
                    f"descriptors/{slug}/operation_is_{pack.discriminator}.umd.json",
                    op_umd,
                )

        # --- C++ stubs/tests ---
        packs_dir = output_dir / "packs"
        packs_dir.mkdir(parents=True, exist_ok=True)
        tests_dir = output_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)

        native_rel = f"packs/{config.native_class_name}Native.cpp"
        (output_dir / native_rel).write_text(
            self._render_template("native.cpp.j2", config, ids=ids)
        )
        written.append(native_rel)

        packs_test_rel = f"tests/Test{config.engine.pascal_name}Packs.cpp"
        (output_dir / packs_test_rel).write_text(
            self._render_template("test_packs.cpp.j2", config, ids=ids)
        )
        written.append(packs_test_rel)

        matchers_test_rel = f"tests/Test{config.engine.pascal_name}Matchers.cpp"
        (output_dir / matchers_test_rel).write_text(
            self._render_template("test_matchers.cpp.j2", config, ids=ids)
        )
        written.append(matchers_test_rel)

        # --- fragments ---
        fragments_dir = output_dir / "fragments"
        fragments_dir.mkdir(parents=True, exist_ok=True)
        for template_name, out_name in FRAGMENT_TEMPLATES:
            content = self._render_template(template_name, config, ids=ids)
            (fragments_dir / out_name).write_text(content)
            written.append(f"fragments/{out_name}")

        return written

    def _render_template(
        self, template_name: str, config: IngestorConfig, **extra
    ) -> str:
        try:
            template = self.env.get_template(template_name)
            return template.render(config=config, **extra)
        except Exception as e:
            raise RuntimeError(
                f"Failed to render template '{template_name}' for engine "
                f"'{config.engine.name}': {e}"
            ) from e


FRAGMENT_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("fragments/cmake_descriptor_files.j2", "cmake_descriptor_files.txt"),
    ("fragments/cmake_ingestor_kernels.j2", "cmake_ingestor_kernels.txt"),
    ("fragments/cmake_target_sources.j2", "cmake_target_sources.txt"),
    ("fragments/cmake_test_sources.j2", "cmake_test_sources.txt"),
    ("fragments/ingestor_packs_hpp.j2", "ingestor_packs.hpp.txt"),
    ("fragments/ingestor_packs_cpp.j2", "ingestor_packs.cpp.txt"),
)
FRAGMENT_FILENAMES: tuple[str, ...] = tuple(name for _, name in FRAGMENT_TEMPLATES)
