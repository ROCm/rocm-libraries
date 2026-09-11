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

from .models import IngestorConfig, KernelSpec, PackSpec

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

    Ids are RANDOM, deliberately. Deriving them from names or metadata was tried and
    reverted: it makes the id only as unique as whatever it is keyed on, and the
    config had no uniqueness guarantee for either. Keying on kernel name gave two
    distinct variants one id (the loader de-duplicates catalog entries by id, so a
    real variant vanished silently); keying on pack name collided a pack's id AND its
    output filename. Uniqueness now comes from uuid4, and the properties the config
    actually needs are enforced by explicit checks in `config_loader` that fail loudly
    and name the offender.

    Indexed by POSITION, not by name, so the lookup stays correct even for a config
    whose names repeat.
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
    for pack_index, pack in enumerate(config.packs):
        ids[("pack", pack_index)] = str(uuid.uuid4())
        if config.is_multi_pack:
            ids[("operation_umd", pack_index)] = str(uuid.uuid4())
        for kernel_index, _kernel in enumerate(pack.kernels):
            ids[("kernel", pack_index, kernel_index)] = str(uuid.uuid4())
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


#: KMD sentinel meaning "not set -- the kernel's own policy decides".
#:
#: It is a CONFIG-AUTHORING convenience and must never reach a descriptor. Every
#: compiled kernel has a definite setting for every knob, so a descriptor claiming
#: "unset" mis-describes its own binary. See `_resolved_metadata`.
UNSET_SENTINEL = -1


def _resolved_metadata(kernel: KernelSpec, config: IngestorConfig) -> dict:
    """Metadata DERIVED from the spec that built the kernel, not carried beside it.

    A tri-state knob lives in three layers that must agree:

      * ``kernel_source.spec[k]`` -- decides the COMPILED BINARY. Absent means the
        kernel's own policy resolves it at build time.
      * ``metadata[k]``           -- what the MATCHER compares; the catalog key.
      * the KMD field's ``default_value`` -- substituted for anything ABSENT at load.

    Letting metadata be authored independently lets it disagree with the binary it
    labels, silently and in both directions. Both directions have shipped: 364
    descriptors advertised "policy decides" while their spec pinned an override, and
    normalising the other way -- dropping the key -- collapsed entries onto the KMD
    default (0) and got the whole engine rejected at load for duplicate tuples.

    So the spec wins. Where the spec pins a knob, that value IS the metadata. The
    authored metadata may only fill knobs the spec does not mention (shape fields
    like batch or seqlen, which the builder takes but does not choose).

    What this deliberately does NOT do is resolve a policy default. That resolution
    lives in the kernel module, is arch- and shape-dependent, and is not importable
    from this tool; guessing it here would recreate the same disagreement one layer
    up. An unresolved tri-state is rejected instead -- see `_check_metadata_resolved`.
    Only knobs the config left UNRESOLVED are derived. Metadata and spec are written
    in two different vocabularies -- metadata carries the hipDNN spelling the matcher
    compares (`"BF16"`), the spec carries the builder's (`"bf16"`) -- so copying the
    spec over an authored value silently breaks matching. Overwriting `dtype` this way
    made every graph decline while the engine still loaded and every count still
    reconciled.
    """
    spec = kernel.kernel_source.spec or {}
    out = dict(kernel.metadata)
    for field_spec in config.kmd_fields:
        name = field_spec.name
        authored = out.get(name, UNSET_SENTINEL)
        if authored != UNSET_SENTINEL:
            # The author stated it, in the vocabulary the matcher reads. Leave it.
            continue
        if name in spec and spec[name] is not None:
            # Unresolved in metadata but pinned in the spec: the binary is definite,
            # so state that rather than ship "undecided".
            value = spec[name]
            out[name] = int(value) if isinstance(value, bool) else value
    return out


def _check_metadata_resolved(
    kernel: KernelSpec, metadata: dict, config: IngestorConfig
) -> None:
    """Refuse to emit a descriptor whose knob does not describe its binary.

    Two ways a knob can fail to describe the binary, and only one of them is
    visible in the emitted document.

    THE STATED SENTINEL. The author wrote ``-1``. A descriptor is a claim about a
    compiled artifact, and every artifact has a definite setting, so shipping the
    sentinel publishes a claim that is not true of anything.

    THE ABSENT KNOB, which is the same defect with nothing to grep for. An optional
    field the author never mentions -- absent from ``metadata`` AND unpinned in
    ``kernel_source.spec`` -- looks like a clean descriptor. It is not: the loader
    substitutes the KMD's ``default_value`` and treats the result as the catalog key,
    while the BINARY was built from the builder dataclass's own default. Those two
    defaults are written by different people for different reasons and are not
    required to agree. When they disagree the descriptor names one kernel and
    advertises another, silently; when they agree, two entries that differ only in
    which layer stated the value collapse onto one tuple and the duplicate takes the
    WHOLE ENGINE down at load.

    Omission is how this shipped, not the sentinel: the sentinel is a value someone
    chose to write and can be searched for, whereas the absent key is indistinguishable
    from a knob nobody needed. Checking only the sentinel therefore catches the
    careful author and misses the hurried one.

    Mandatory fields are not checked here -- the config loader already refuses a
    kernel that omits one, and its diagnostic is better because it names the pack.

    Fail here, where the config author can see which kernel and which knob.
    """
    unresolved = sorted(k for k, v in metadata.items() if v == UNSET_SENTINEL)
    if unresolved:
        raise ValueError(
            f"kernel {kernel.name!r} ships the unset sentinel "
            f"({UNSET_SENTINEL}) for {unresolved}: metadata must state the value the "
            f"kernel was BUILT with. Pin the knob in kernel_source.spec, or write the "
            f"resolved value in metadata -- the descriptor cannot say 'undecided' "
            f"about a binary that already decided."
        )

    spec = kernel.kernel_source.spec or {}
    undeclared = sorted(
        f.name
        for f in config.kmd_fields
        if not f.is_mandatory and f.name not in metadata and spec.get(f.name) is None
    )
    if undeclared:
        raise ValueError(
            f"kernel {kernel.name!r} states {undeclared} in neither its metadata nor "
            f"its kernel_source.spec, so nothing here decides the value. The loader "
            f"will substitute the KMD default_value as the catalog key while the "
            f"kernel is compiled from the builder's own default -- two independent "
            f"defaults that are not required to agree, and whose disagreement is "
            f"silent. Pin the knob in kernel_source.spec if the binary should carry "
            f"it, or write the resolved value in metadata if the builder's default "
            f"is what you mean."
        )


def _dedup_key(metadata: dict) -> str:
    """Identity of a descriptor AS THE MATCHER SEES IT.

    Keyed on the resolved metadata that actually ships, so the key and the artifact
    cannot drift apart. Overlapping generation expressions are expected -- an author
    writes "the model-trace shapes" and "the published-sweep shapes" without
    hand-partitioning them -- and two entries resolving to the same tuple are one
    candidate to the runtime however many expressions produced them.
    """
    return json.dumps(metadata, sort_keys=True)


def _pack_index(config: IngestorConfig, pack: PackSpec) -> int:
    """This pack's position in the config.

    Ids are keyed on position rather than name because nothing guarantees pack names
    are unique, and a name-keyed lookup silently returns another pack's id. Identity
    comparison, not equality: two packs may legitimately compare equal by value.
    """
    for index, candidate in enumerate(config.packs):
        if candidate is pack:
            return index
    raise ValueError(f"pack {pack.name!r} is not part of this config")


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
        "id": ids[("operation_umd", _pack_index(config, pack))],
        "name": f"graph operation is {pack.discriminator}",
        "scope": "graph",
        "match_symbol": config.operation_match_symbol(pack),
    }


def build_kdp(
    config: IngestorConfig,
    pack: PackSpec,
    ids: dict,
    seen_metadata: dict | None = None,
) -> dict:
    """One pack's KDP.

    ``seen_metadata`` carries de-duplication state ACROSS the packs of one engine.
    Pass the same dict to every ``build_kdp`` call for a config; omit it and each
    pack de-duplicates against itself alone, which is the behaviour that ships a
    duplicate.

    Why engine-wide and not per-pack. The loader collects packs into one
    ``DescriptorSet`` BY ENGINE ID and the matcher maps below it are keyed by id, so
    many packs under one engine is the designed shape -- and two entries with
    identical matcher-visible metadata are one candidate to the runtime no matter
    which pack produced them. Per-pack de-duplication cannot see that: it emits both,
    the plan builder benchmarks both, and they can never resolve to different code.
    Worse, identical metadata is a duplicate CATALOG TUPLE, and a duplicate tuple does
    not drop an entry -- it drops the whole engine at load.

    This is why a coverage gap is served by widening the union rather than by shipping
    a second bundle beside the first. A second bundle looks like a packaging choice and
    is really a choice about engine identity: the generator derives native symbol names
    from the engine name, so a new slug mints a new engine that registers, loads, and
    matches nothing. One KDP per pack, one de-duplicated union per engine, is
    duplicate-proof by construction.
    """
    pack_index = _pack_index(config, pack)
    matchers = [ids["kernel_match"]]
    if config.is_multi_pack:
        matchers.insert(0, ids[("operation_umd", pack_index)])
    # Several generation expressions may target one engine and are EXPECTED to
    # overlap -- an author writes "the model-trace shapes" and "the published-sweep
    # shapes" without hand-partitioning them. Two entries with identical
    # matcher-visible metadata are one candidate to the runtime no matter how many
    # expressions produced them, so emitting both costs a compile, catalog space and
    # a benchmark iteration to advertise a choice that does not exist. De-duplicate
    # here, keyed on the metadata the matcher actually reads.
    kernel_descriptors = []
    if seen_metadata is None:
        seen_metadata = {}
    duplicates: list = []
    for index, kernel in enumerate(pack.kernels):
        # Resolve FIRST, then key on the resolved form: the dedup key and the emitted
        # document are the same bytes, so they cannot drift apart.
        metadata = _resolved_metadata(kernel, config)
        _check_metadata_resolved(kernel, metadata, config)
        key = _dedup_key(metadata)
        if key in seen_metadata:
            duplicates.append((kernel.name, seen_metadata[key]))
            continue
        seen_metadata[key] = kernel.name
        entry = {
            "version": "1.0",
            "id": ids[("kernel", pack_index, index)],
            "name": kernel.name,
            # Per-kind keys, never the union: the runtime loader hard-fails an
            # unknown key and hkp_pack validates a closed set per kind.
            "kernel_source": kernel.kernel_source.as_document(),
            "metadata": metadata,
            "priority": kernel.priority,
        }
        if kernel.arch:
            entry["arch"] = list(kernel.arch)
        kernel_descriptors.append(entry)
    if duplicates:
        shown = ", ".join(f"{d} == {k}" for d, k in duplicates[:3])
        more = f" (+{len(duplicates) - 3} more)" if len(duplicates) > 3 else ""
        print(
            f"  pack '{pack.name}': dropped {len(duplicates)} duplicate "
            f"variant(s) with metadata already emitted: {shown}{more}"
        )
    kdp = {
        "version": "1.0",
        "id": ids[("pack", pack_index)],
        "name": f"{config.engine.namespace}:{config.kdp_stem(pack)}",
        "matchers": matchers,
        "engine": ids["ued"],
        "dispatch": ids["udd"],
        "kernelDescriptors": kernel_descriptors,
    }
    if pack.arch:
        kdp["arch"] = list(pack.arch)
    elif config.is_packaged:
        # hkp_pack REQUIRES arch on a KDP (_validate_kdp), unlike the runtime
        # loader, which treats absence as a wildcard. Emitting a packaged KDP
        # without it fails the pack rather than shipping something wrong, but
        # the message would be about a missing key rather than the real cause,
        # so the config loader rejects this earlier with a better one. Reaching
        # here means that check was bypassed; keep the key present and empty so
        # the packager's own diagnostic is the one the author sees.
        kdp["arch"] = []
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
        ddir = config.descriptor_dir
        files = [
            f"{ddir}/{slug}.kmd.json",
            f"{ddir}/{slug}.ued.json",
            f"{ddir}/{slug}.udd.json",
        ]
        if config.engine.has_heuristic:
            files.append(f"{ddir}/{slug}.uhd.json")
        files.append(f"{ddir}/kernel_dtype_matches_graph.umd.json")
        for pack in config.packs:
            files.append(f"{ddir}/{config.kdp_stem(pack)}.kdp.json")
            if config.is_multi_pack:
                files.append(f"{ddir}/operation_is_{pack.discriminator}.umd.json")
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
        ddir = config.descriptor_dir
        (output_dir / ddir).mkdir(parents=True, exist_ok=True)

        def write_json(rel: str, obj: dict) -> None:
            path = output_dir / rel
            path.write_text(_dump(obj))
            written.append(rel)

        write_json(f"{ddir}/{slug}.kmd.json", build_kmd(config, ids))
        write_json(f"{ddir}/{slug}.ued.json", build_ued(config, ids))
        write_json(f"{ddir}/{slug}.udd.json", build_udd(config, ids))
        uhd = build_uhd(config, ids)
        if uhd is not None:
            write_json(f"{ddir}/{slug}.uhd.json", uhd)
        write_json(
            f"{ddir}/kernel_dtype_matches_graph.umd.json",
            build_kernel_match_umd(config, ids),
        )
        # One de-duplication scope for the whole engine, because the loader collects
        # every pack sharing an engine id into ONE catalog. Two packs holding the same
        # matcher-visible metadata are a duplicate candidate the runtime benchmarks
        # twice and a duplicate catalog tuple that drops the engine at load.
        seen_metadata: dict = {}
        for pack in config.packs:
            write_json(
                f"{ddir}/{config.kdp_stem(pack)}.kdp.json",
                build_kdp(config, pack, ids, seen_metadata),
            )
            op_umd = build_operation_umd(config, pack, ids)
            if op_umd is not None:
                write_json(
                    f"{ddir}/operation_is_{pack.discriminator}.umd.json",
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

    #: Emitted files that are splice INSTRUCTIONS, not shipped source. They are
    #: pasted into existing files by hand and never exist as files in the tree,
    #: so they are excluded from the located/missing accounting.
    _NON_SHIPPED_PREFIXES = ("fragments/",)

    @classmethod
    def locate_emitted(
        cls, root: Path, written: list[str]
    ) -> tuple[dict[str, Path], list[str], dict[str, list[Path]]]:
        """``({relative path: real path}, [not found], {relative path: [ambiguous]})``
        for the shippable files in ``written``, searched by BASENAME under ``root``.

        Not ``root / rel``. This tool emits a flat ``packs/`` + ``tests/``
        layout, but the provider splits it: packs land in the engine directory
        and the test stubs under ``src/tests/engines/.../packs/`` -- which this
        generator's own ``cmake_test_sources`` fragment instructs. Resolving
        ``rel`` against one directory therefore found the packs and silently
        missed every test stub, reproducing precisely the ``packs/``-only blind
        spot this scan exists to close.

        Two matches for one basename is an ERROR, not a pick. Keeping the first
        ``rglob`` hit made the answer depend on filesystem order: a stale copy or
        a build tree under ``root`` could bind instead of the real file, and a
        filled decoy would report the gate green while the real file still
        carried its markers. Basenames are not unique here -- 1809 collide
        repo-wide, and ``build/`` already duplicates shipped descriptor names --
        so uniqueness is luck, not a property. "Found something, assumed it was
        the right thing" is the shape this gate exists to reject.
        """
        shippable = [
            rel for rel in written if not rel.startswith(cls._NON_SHIPPED_PREFIXES)
        ]
        wanted = {Path(rel).name: rel for rel in shippable}
        hits: dict[str, list[Path]] = {}
        for path in sorted(root.rglob("*")):
            rel = wanted.get(path.name)
            if rel is not None and path.is_file():
                hits.setdefault(rel, []).append(path)
        found = {rel: paths[0] for rel, paths in hits.items() if len(paths) == 1}
        ambiguous = {rel: paths for rel, paths in hits.items() if len(paths) > 1}
        missing = [rel for rel in shippable if rel not in hits]
        return found, missing, ambiguous

    @classmethod
    def unfilled_placeholders(cls, root: Path, written: list[str]) -> dict[str, int]:
        """``{relative path: placeholder count}`` for every located file that
        still carries an unfilled stub marker, worst first.

        Lives here because this object is the only one that knows the full
        emitted set. The runbook used to carry a hand-written
        ``grep -c "FILL THIS OUT" .../packs/*Native.cpp``, which missed the
        generated ``tests/Test<Name>Matchers.cpp`` entirely -- a transcribed
        glob drifts the moment the emitted set changes, and that one already
        had. Ask the generator instead; it cannot fall behind itself.
        """
        located, _missing, _ambiguous = cls.locate_emitted(root, written)
        counts: dict[str, int] = {}
        for rel, path in located.items():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            n = text.count(PLACEHOLDER_MARKER)
            if n:
                counts[rel] = n
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

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


#: The marker every unfilled stub body carries. Templates emit it; the reader
#: replaces it. One spelling, defined once, so a scan cannot look for a string
#: the templates stopped writing.
PLACEHOLDER_MARKER = "FILL THIS OUT"

FRAGMENT_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("fragments/cmake_descriptor_files.j2", "cmake_descriptor_files.txt"),
    ("fragments/cmake_ingestor_kernels.j2", "cmake_ingestor_kernels.txt"),
    ("fragments/cmake_target_sources.j2", "cmake_target_sources.txt"),
    ("fragments/cmake_test_sources.j2", "cmake_test_sources.txt"),
    ("fragments/ingestor_packs_hpp.j2", "ingestor_packs.hpp.txt"),
    ("fragments/ingestor_packs_cpp.j2", "ingestor_packs.cpp.txt"),
)
FRAGMENT_FILENAMES: tuple[str, ...] = tuple(name for _, name in FRAGMENT_TEMPLATES)
