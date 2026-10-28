# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Generate (or verify) the authored descriptors for one arch's FlyDSL kernels.

Usage:
    python gen_descriptors.py --arch gfx11-generic
    python gen_descriptors.py --arch gfx11-generic --check     # verify, write nothing

**Generated, not hand-written, and checked in.** Every field is derived from
``<content>/<op>/<arch>/manifest.json`` and the code objects it names, so a
descriptor cannot disagree with the object it describes. ``<content>`` is
FlyDSL's producer folder under the provider's production descriptor root
(``generators._flydsl_env.CONTENT_DIR``), the same root rocKE's bundles are
authored in.

**We author the input form; the shared packer produces the shipped form.** Each
UKD is ``kind: "hsaco"`` and names its object, which sits beside it, by file
name. At build time the packer that packs every other bundle in the root
resolves the object, packs it into the per-arch
``hip_kernel_provider_<arch>.kpack``, reads the argument signature out of the
object, and rewrites the UKD to ``kind: "kpack"``. So the archive path, TOC key
and signature are deliberately *not* written here: the packer derives all three
from the object and is the single place they are computed, for FlyDSL as for
every other producer.

**The packer checks neither an object's bytes nor its target processor**; it
packs what a UKD names into every shard the UKD's ``arch`` lists. Both checks
live here instead, and ``--check`` runs before every pack: each object must hash
to its manifest's SHA256, its ``amdhsa.target`` must be the arch its directory
names, and for a generic target its ELF ``e_flags`` must carry that generic
machine.

**Generic targets.** ``<arch>`` may be an LLVM generic target
(``gfx11-generic``): one object set that runs on every member of the family
(``generators/arch_families.json``). Its KDP and UKDs then list every member in
``arch``, so the packer copies the same bytes into each member's own shard and
each device finds them under its own ``gcnArchName`` -- no runtime support for
generic names is needed.

Layout, per op:

* ``<content>/<op>/`` -- the descriptors shared by every arch: metadata schema,
  engine, heuristic, dispatch and the kernel-scoped matcher. One copy, so the
  authored root holds each id once.
* ``<content>/<op>/<arch>/`` -- that arch's pack (KDP), one UKD per object, and
  the objects themselves with ``manifest.json``: generated content, stored with DVC.
  Its readable summary, ``<arch>.SOURCE.md``, sits beside the folder.

``--check`` re-derives the documents and compares them to what is checked in,
then runs the packer's own authoring validator over the whole FlyDSL folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, NamedTuple

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "descriptor-packaging" / "python"))
sys.path.insert(0, str(_HERE))

from hkp_pack.generic_targets import DEFAULT_TABLE_PATH, GenericTargets  # noqa: E402
from hkp_pack.descriptors import (  # noqa: E402
    HkpPackError,
    load_flat_input,
    type_from_filename,
)

from generators import _arch_families as families  # noqa: E402
from generators._codeobject import (  # noqa: E402
    GeneratorError,
    describe,
    verify_arch,
    verify_generic,
)
from generators._flydsl_env import CONTENT_DIR  # noqa: E402
from generators._manifest import inputs_digest  # noqa: E402

# Descriptor schema version, as the loader spells it: UKD_VERSION_MAJOR = 1,
# UKD_VERSION_MINOR = 0 (DescriptorLoader.hpp:288-289).
DESCRIPTOR_VERSION = "1.0"
# Ops whose engines read a runtime pass-by-value scalar (the SDPA softmax scale, the
# RMSNorm epsilon), and the graph schema that introduced such tensors
# (PluginVersionConstants.hpp, K_PASS_BY_VALUE_MIN_API_VERSION).
SDPA_OPS = (
    "sdpa_rdna",
    "sdpa_rdna_decode",
    "sdpa_cdna",
    "sdpa_cdna_dualwave",
    "sdpa_cdna_decode",
    "sdpa_aiw",
)
PASS_BY_VALUE_OPS: tuple[str, ...] = ("rmsnorm",) + SDPA_OPS
PASS_BY_VALUE_SDK_VERSION = "1.2.0"

MANIFEST_NAME = "manifest.json"

# `metadataValueFromJson` (DescriptorLoader.hpp:523) accepts bool/int/float/
# string/int-list and *fails on null*, so "this kernel reads N at runtime"
# cannot be spelled the way the manifest spells it. 0 is the sentinel: no shape
# is ever 0 rows, and the matcher reads it as "generic, accepts any N".
GENERIC_N = 0

# Manifest knobs carried as null when not baked in. Only these may take the
# `GENERIC_N` sentinel; a null anywhere else is a generator bug, not a tier.
SHAPE_KNOBS = {"N", "head_dim", "head_dim_v"}

# Fields the graph fixes rather than the tuner choosing. They belong in the
# metadata schema -- the matcher needs every one of them to pick a variant --
# but not in the engine's `knobs` list, which names what autotune may vary.
# Offering the tuner a dtype or a row length it cannot legally change would
# have it search a space where all but one point fails to match.
#
# Per op: `N` is RMSNorm's row width; `head_dim`, `head_dim_v` and `causal` are what
# an SDPA graph asks for, and on CDNA so is the kernel family, which the target arch
# fixes. `has_bias`, `has_sink` and the bias forms (`bias_f32`, `bias_q`,
# `bias_kstrided`) are fixed by whether and how the graph adds a bias or a sink. The
# tier bounds and capability flags (`n_max`, `head_dim_max`, `any_diagonal`,
# `kv_align`) say which graphs an object can serve: the matcher gates on them, and
# autotune compares the objects they distinguish as separate catalog entries.
GRAPH_DETERMINED_FIELDS = {
    "N",
    "dtype",
    "scale_f32",
    "store_rstd",
    "n_max",
    "reduce_symbol",
    "head_dim",
    "head_dim_v",
    "head_dim_max",
    "qk_mask",
    "causal",
    "any_diagonal",
    "kv_align",
    "has_bias",
    "bias_f32",
    "bias_q",
    "bias_kstrided",
    "has_sink",
    "merge_symbol",
    "kernel",
}

# Fields that say when an object is preferred rather than how it was built: the
# score reads them, and like the graph-determined ones they are no knob for autotune,
# which benchmarks the objects themselves.
SELECTION_FIELDS = {"prefer_feature", "prefer_min", "prefer_max"}

# The schedule and launch geometry an object was built with, which never changes what
# it computes: the only fields an engine exposes as knobs. Every metadata field must be
# in exactly one of these three sets, so a new field cannot become a knob by default.
TUNING_FIELDS = {
    "block_threads",
    "block_m",
    "block_n",
    "row_subtiles",
    "dv_split",
    "hd_pair",
    "k_chunk",
    "waves_per_eu",
    "setprio",
    "kv_gpfetch",
    "k_prefetch_dist",
    "v_prefetch_dist",
    "kv_addr_hoist",
}

# Every engine carries an engine estimate (RFC 0019 §3.1 `predict_engine`, L1): the
# expected tflops of the engine's own pick, which engine selection reads before any
# catalog is enumerated. An engine without one is ordered by static rules whatever the
# graph, and FlyDSL's engines compete for graphs: on gfx1201 sdpa_rdna, sdpa_rdna_decode
# and sdpa_aiw each match nearly every graph of the others' corpora.
#
# The estimate's `default` slot. No scorer is registered under its symbol, so the loader
# disables the estimate and keeps the engine (RFC 0019 §11.2), exactly as an engine with no
# estimate behaves; a trained model installed for an arch (below) is bound under that
# arch's key and takes precedence there.
ENGINE_ESTIMATE_METRIC = "tflops"

# Trained models live beside the op's descriptors where `uhd_gen promote` installs them,
# `heuristics/<ued-id>/<role>/<arch>/<metric>/`, and the UED binds each under its role and
# arch key. The generator owns the UED, so it derives those bindings from what is
# installed: a regeneration keeps every trained model, and `--check` accepts a tree with
# models in it. A model under a gfx key wins over `default` on that arch (RFC 0020 §4.6),
# so the native ranker and the estimate slot stay the fallback everywhere else.
HEURISTICS_DIR = "heuristics"
_SCORING_ROLES = ("sort_kernel_catalog", "predict_engine")
_ARCH_KEY = re.compile(r"default|gfx[0-9a-f]+")


def installed_models(
    op_dir: Path, ued_id: str
) -> dict[str, dict[str, list[tuple[str, str]]]]:
    """Role -> arch key -> [(metric, UHD id)] of the trained models installed for an op.

    Refuses a model filed anywhere but its own engine's, role's, arch's and metric's
    folder, one trained against another engine, and two models for one metric on one
    key -- the loader would disable that metric there (RFC 0019 §3.1).
    """
    root = op_dir / HEURISTICS_DIR
    installed: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for path in sorted(root.rglob("*.uhd.json")) if root.is_dir() else []:
        parts = path.relative_to(root).parts
        if len(parts) != 5 or parts[0] != ued_id:
            raise HkpPackError(
                f"{path}: a trained model is installed at "
                f"{HEURISTICS_DIR}/{ued_id}/<role>/<arch>/<metric>/"
            )
        _, role, arch, metric, _ = parts
        if role not in _SCORING_ROLES or not _ARCH_KEY.fullmatch(arch):
            raise HkpPackError(
                f"{path}: '{role}/{arch}' is not a scoring role and arch key "
                f"({', '.join(_SCORING_ROLES)}; default or gfx<id>)"
            )
        doc = json.loads(path.read_text(encoding="utf-8"))
        if (doc.get("score") or {}).get("metric") != metric:
            raise HkpPackError(
                f"{path}: filed under metric '{metric}' but its score.metric is "
                f"{(doc.get('score') or {}).get('metric')!r}"
            )
        trained = ((doc.get("trained_against") or {}).get("ued") or {}).get("id")
        if trained != ued_id:
            raise HkpPackError(
                f"{path}: trained against UED {trained!r}, not this engine's {ued_id}"
            )
        models = installed.setdefault(role, {}).setdefault(arch, [])
        if any(existing == metric for existing, _ in models):
            raise HkpPackError(
                f"{path}: a second '{metric}' model for {role}/{arch}; the loader would "
                "disable that metric on that key"
            )
        models.append((metric, doc["id"]))
    return installed


# uuid5 from a name, so ids are stable across regenerations and adding a
# thirteenth instance cannot disturb the twelve already shipped. The namespace
# is itself derived from a URL rather than being a magic constant. The names are
# unchanged from when these descriptors were emitted in shipped form, so the ids
# the runtime sees did not move with the switch to authoring them.
_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://github.com/ROCm/rocm-libraries/dnn-providers/hip-kernel-provider/flydsl",
)


def _uuid(name: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, name))


class KernelEntry(NamedTuple):
    """One checked-in code object and its manifest record."""

    op: str
    path: Path
    data: bytes
    record: dict[str, Any]


def collect(content_dir: Path, arch: str) -> list[KernelEntry]:
    """Every object the manifests under ``<content>/*/<arch>/`` name.

    Raises on a manifest that disagrees with its own location, on a named object
    that is absent, on one whose bytes do not hash to the recorded SHA256, and on
    one built for a processor other than the one its directory names.
    """
    manifests = sorted(content_dir.glob(f"*/{arch}/{MANIFEST_NAME}"))
    if not manifests:
        raise FileNotFoundError(
            f"no */{arch}/{MANIFEST_NAME} under {content_dir}; run the generators "
            "for this arch first (see flydsl/REGEN.md)"
        )

    entries: list[KernelEntry] = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        op = manifest["op"]
        # The manifest records the arch it was built for and the op it holds; the
        # directory claims both. A disagreement means an object is about to be
        # described for hardware it was not compiled for.
        if manifest.get("arch") != arch or manifest_path.parent.name != arch:
            raise ValueError(
                f"{manifest_path}: manifest records arch '{manifest.get('arch')}' "
                f"but sits under '{manifest_path.parent.name}/'"
            )
        if manifest_path.parent.parent.name != op:
            raise ValueError(
                f"{manifest_path}: manifest records op '{op}' but sits in "
                f"'{manifest_path.parent.parent.name}/'"
            )
        # The objects must be built from the sources in this tree. A kernel or
        # generator edit without a regeneration would otherwise ship objects that
        # no longer match what a reviewer reads.
        recorded = manifest.get("toolchain", {}).get("inputs_sha256")
        current = inputs_digest(op, arch)
        if recorded != current:
            raise ValueError(
                f"{manifest_path}: the objects were generated from kernel and "
                f"generator sources with digest {recorded}, but the tree's are "
                f"{current}. Regenerate ({op}, {arch}) as flydsl/REGEN.md describes."
            )
        # An object the manifest no longer names is left behind when an instance is
        # renamed or dropped; the generators write objects but never delete them.
        named = {record["file"] for record in manifest["instances"]}
        stray = sorted(
            p.name for p in manifest_path.parent.glob("*.hsaco") if p.name not in named
        )
        if stray:
            raise ValueError(
                f"{manifest_path.parent}: {len(stray)} object(s) the manifest does not "
                f"name, e.g. '{stray[0]}'. Delete them: they are left over from a "
                "previous generation and would ship unreferenced."
            )
        for record in manifest["instances"]:
            path = manifest_path.parent / record["file"]
            if not path.is_file():
                raise FileNotFoundError(
                    f"{manifest_path} names '{record['file']}' but {path} does not exist"
                )
            data = path.read_bytes()
            actual = hashlib.sha256(data).hexdigest()
            if actual != record["sha256"]:
                raise ValueError(
                    f"{path}: SHA256 {actual} does not match the {record['sha256']} "
                    f"recorded in {manifest_path}. Regenerate the object rather than "
                    "editing the manifest."
                )
            # The directory is the claim the descriptors turn into `arch`; the
            # object's own target is the fact. The packer compares neither.
            where = str(path)
            verify_arch(describe(data, where, record["symbol"]), arch, where)
            verify_generic(data, arch, where)
            entries.append(KernelEntry(op, path, data, record))
    return entries


def _metadata_type(value: Any) -> str:
    """The KMD type token for a manifest knob value.

    A null knob means "not baked" and is carried as the `GENERIC_N` int
    sentinel, so it types as an int like its baked siblings -- one field cannot
    change type between instances.
    """
    if value is None or isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    raise HkpPackError(f"knob value {value!r} has no metadata type")


def _metadata_value(key: str, value: Any) -> Any:
    if value is None:
        if key not in SHAPE_KNOBS:
            raise HkpPackError(
                f"knob '{key}' is null but is not a shape knob; only a shape "
                "knob has a 'not baked' sentinel"
            )
        return GENERIC_N
    return value


class OpDescriptors:
    """The one-per-op descriptors: schema, engine, heuristic, dispatch, matcher.

    Shared by every arch's pack of the op and authored once, under ``<op>/``.
    The packer copies them into each shard whose pack reaches them, so the ids
    are deliberately *not* arch-scoped: two shards carrying the same engine with
    the same id and the same bytes is how the catalog settles them into one
    entry rather than two rival engines.
    """

    def __init__(
        self,
        op: str,
        fields: dict[str, str],
        installed: dict[str, dict[str, list[tuple[str, str]]]] | None = None,
    ) -> None:
        self.op = op
        self.stem = f"flydsl_{op}"
        self.symbols = f"hipkernel.{self.stem}"
        self.fields = fields
        self.installed = installed or {}

        self.kmd_id = _uuid(f"kmd/{op}")
        self.uhd_id = _uuid(f"uhd/{op}")
        self.udd_id = _uuid(f"udd/{op}")
        self.ued_id = _uuid(f"ued/{op}")
        self.umd_id = _uuid(f"umd/{op}/kernel")
        self.estimate_uhd_id = _uuid(f"uhd/{op}/predict_engine")

    def documents(self) -> dict[str, dict[str, Any]]:
        ued: dict[str, Any] = {
            "version": DESCRIPTOR_VERSION,
            "id": self.ued_id,
            "name": f"hipkernel:{self.stem}",
        }
        # The graph schema the engine's matcher understands: the ingestor declines any
        # graph whose schema floor is above it. Runtime pass-by-value scalars (1.2.0) are
        # read by the engines whose scale or epsilon may be one; the rest stay at the
        # baseline.
        if self.op in PASS_BY_VALUE_OPS:
            ued["sdk_version"] = PASS_BY_VALUE_SDK_VERSION
        kmd_fields = []
        for name, type_token in self.fields.items():
            field: dict[str, Any] = {"name": name, "type": type_token}
            # Only the tunable field carries a default. The rest are required:
            # `coerceKernelMetadata` (DescriptorLoader.hpp:1291) drops the whole
            # pack when a field with no default is missing, which is the loud
            # failure we want if an instance ever ships without its dtype or N.
            if name == "block_threads":
                field["default_value"] = 256
            kmd_fields.append(field)

        engine: dict[str, Any] = {
            **ued,
            "graph_match": {"native": f"{self.symbols}.graph_match"},
        }
        # Role -> arch key -> [(metric, UHD id)]: the UHD that orders the engine's kernel
        # catalog on every arch, and the engine estimate's slot.
        roles: dict[str, dict[str, list[tuple[str | None, str]]]] = {
            "sort_kernel_catalog": {"default": [(None, self.uhd_id)]},
            "predict_engine": {
                "default": [(ENGINE_ESTIMATE_METRIC, self.estimate_uhd_id)]
            },
        }
        for role, by_arch in self.installed.items():
            for arch, models in by_arch.items():
                slot = roles.setdefault(role, {}).setdefault(arch, [])
                for metric, model_id in models:
                    # A trained model takes its metric's place, as promote's rebinding does.
                    slot[:] = [entry for entry in slot if entry[0] != metric]
                    slot.append((metric, model_id))
        for role, by_arch in roles.items():
            engine[role] = {
                arch: ids[0][1] if len(ids) == 1 else [model for _, model in ids]
                for arch, ids in sorted(
                    by_arch.items(), key=lambda item: (item[0] != "default", item[0])
                )
            }
        estimate = {
            f"{self.stem}_engine_estimate.uhd.json": {
                "version": DESCRIPTOR_VERSION,
                "id": self.estimate_uhd_id,
                "name": f"flydsl {self.op} engine estimate",
                "provenance": (
                    "Default slot for the engine estimate: a trained model installed "
                    "under heuristics/ for an arch takes precedence there. No scorer is "
                    "registered under this symbol, so on any other arch the loader "
                    "disables the estimate and keeps the engine."
                ),
                "adapter": "native",
                "objective": "max",
                "score": {"metric": ENGINE_ESTIMATE_METRIC},
                "native": {"symbol": f"{self.symbols}.engine_estimate"},
            }
        }
        engine["metadata"] = self.kmd_id
        engine["knobs"] = [f for f in self.fields if f in TUNING_FIELDS]

        return {
            f"{self.stem}.kmd.json": {
                "version": DESCRIPTOR_VERSION,
                "id": self.kmd_id,
                "name": f"flydsl {self.op} variant fields",
                "fields": kmd_fields,
            },
            f"{self.stem}.uhd.json": {
                "version": DESCRIPTOR_VERSION,
                "id": self.uhd_id,
                "name": f"flydsl {self.op} selector",
                # The engine's own native scorer, higher first, as the ingestor ranked
                # it before UHD models (RFC 0019): no trained model ships yet.
                "adapter": "native",
                "objective": "max",
                "native": {"symbol": f"{self.symbols}.score"},
            },
            f"{self.stem}.udd.json": {
                "version": DESCRIPTOR_VERSION,
                "id": self.udd_id,
                "name": f"flydsl {self.op} dispatch",
                "dispatch_symbol": f"{self.symbols}.dispatch",
            },
            f"{self.stem}.ued.json": engine,
            **estimate,
            f"{self.stem}_kernel_match.umd.json": {
                "version": DESCRIPTOR_VERSION,
                "id": self.umd_id,
                "name": f"flydsl {self.op}: kernel variant fits the graph",
                "scope": "kernel",
                "match_symbol": f"{self.symbols}.kernel_match",
            },
        }

    def provenance(self) -> dict[str, Any]:
        """What this pack's metadata is used for, stated honestly.

        Nothing is compiled in from a binding: the objects arrive pre-built from
        FlyDSL's own generator, so every field exists only to pick which
        pre-built object fits. `metadata_fields` empty and every field
        matcher-only is the exact shape of that claim, and the only shape the
        packer accepts from a producer that does not compile. Derived from the
        same field list the KMD is built from, so the two cannot drift apart.
        """
        return {
            "specialization_contract": {
                "schema_version": 1,
                "consumers": [
                    {
                        "engine_id": self.ued_id,
                        "kmd_id": self.kmd_id,
                        "metadata_fields": [],
                        "matcher_only_fields": list(self.fields),
                        "bindings": {},
                        "vocabulary": {},
                    }
                ],
            }
        }


def _fields_for_op(entries: list[KernelEntry]) -> dict[str, str]:
    """The metadata schema for an op, derived from its instances' knobs.

    Every instance must declare the same knob set: a schema is one shape, and an
    instance that carries a knob its siblings do not is a generator bug that
    would otherwise surface as a silently unselectable kernel.
    """
    fields: dict[str, str] = {}
    first: dict[str, Any] | None = None
    for entry in entries:
        knobs = entry.record["knobs"]
        if first is None:
            first = knobs
            fields["dtype"] = "string"
            for key, value in knobs.items():
                fields[key] = _metadata_type(value)
        elif set(knobs) != set(first):
            raise HkpPackError(
                f"{entry.record['name']} declares knobs {sorted(knobs)} but its "
                f"siblings declare {sorted(first)}; one op is one schema"
            )
    for name in fields:
        homes = [
            label
            for label, group in (
                ("GRAPH_DETERMINED_FIELDS", GRAPH_DETERMINED_FIELDS),
                ("SELECTION_FIELDS", SELECTION_FIELDS),
                ("TUNING_FIELDS", TUNING_FIELDS),
            )
            if name in group
        ]
        if len(homes) != 1:
            raise HkpPackError(
                f"{entries[0].op}: metadata field '{name}' must be in exactly one of "
                f"GRAPH_DETERMINED_FIELDS, SELECTION_FIELDS and TUNING_FIELDS "
                f"(found in {homes or 'none'}); only TUNING_FIELDS become knobs"
            )
    return fields


def _arch_documents(
    entries: list[KernelEntry], op: OpDescriptors, arch: str
) -> dict[str, dict[str, Any]]:
    """One arch's pack of an op: its KDP, holding one inline ``hsaco`` UKD per object.

    Inline rather than one file per object: no UKD is shared between packs, the
    specialization contract is declared once on the KDP and inherited by every
    kernel, and the packaging probes take only inline UKDs.
    """
    kernels: list[dict[str, Any]] = []
    # A generic target's objects run on every member, so the pack and each UKD
    # list them all and the packer ships the bytes into each member's shard.
    runs_on = list(families.members(arch))

    for entry in entries:
        record = entry.record
        # Named as it was when these were emitted in shipped form, where the
        # string was the archive key, so the id did not change with the form.
        ukd_id = _uuid(f"ukd/{arch}/{entry.op}/{record['file']}")

        metadata = {"dtype": record["dtype"]}
        for key, value in record["knobs"].items():
            metadata[key] = _metadata_value(key, value)

        kernels.append(
            {
                "version": DESCRIPTOR_VERSION,
                "id": ukd_id,
                "name": record["name"],
                "arch": runs_on,
                "kernel_source": {
                    "kind": "hsaco",
                    # Beside the KDP: the packer resolves an inline kernel's file
                    # relative to the folder of the KDP that holds it.
                    "file": record["file"],
                    "symbol": record["symbol"],
                },
                "metadata": metadata,
                "priority": record["priority"],
            }
        )

    kdp = {
        "version": DESCRIPTOR_VERSION,
        "id": _uuid(f"kdp/{op.op}/{arch}"),
        "name": f"flydsl {op.op} kernels ({arch})",
        "arch": runs_on,
        "matchers": [op.umd_id],
        "engine": op.ued_id,
        "dispatch": op.udd_id,
        "provenance": op.provenance(),
        "kernelDescriptors": kernels,
    }
    return {f"{op.stem}.kdp.json": kdp}


def build_documents(
    entries: list[KernelEntry], arch: str, content_dir: Path = CONTENT_DIR
) -> dict[Path, dict[str, dict[str, Any]]]:
    """Every document for `arch`, grouped by the folder (relative to the content
    root) it is written into."""
    by_op: dict[str, list[KernelEntry]] = {}
    for entry in entries:
        by_op.setdefault(entry.op, []).append(entry)

    folders: dict[Path, dict[str, dict[str, Any]]] = {}
    for op_name, op_entries in sorted(by_op.items()):
        installed = installed_models(content_dir / op_name, _uuid(f"ued/{op_name}"))
        op = OpDescriptors(op_name, _fields_for_op(op_entries), installed)
        folders[Path(op_name)] = op.documents()
        folders[Path(op_name) / arch] = _arch_documents(op_entries, op, arch)
    return folders


def _serialize(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2) + "\n"


def _descriptor_files(folder: Path) -> list[Path]:
    """The descriptor files directly in `folder` -- never manifest.json."""
    return sorted(
        path for path in folder.glob("*.json") if type_from_filename(path) is not None
    )


def _write(content_dir: Path, folders: dict[Path, dict[str, dict[str, Any]]]) -> None:
    for rel, documents in sorted(folders.items()):
        folder = content_dir / rel
        folder.mkdir(parents=True, exist_ok=True)
        for name, document in sorted(documents.items()):
            (folder / name).write_text(_serialize(document), encoding="utf-8")

        # A descriptor whose object no longer exists names a file the packer
        # cannot find, so leaving one behind fails the build rather than the
        # regeneration. Regeneration owns the descriptors in these folders --
        # and only those: the manifest and the objects beside them are the
        # generators'.
        for stale in _descriptor_files(folder):
            if stale.name not in documents:
                print(f"  removing stale descriptor: {rel / stale.name}")
                stale.unlink()


def _check(
    content_dir: Path, folders: dict[Path, dict[str, dict[str, Any]]]
) -> list[str]:
    """Compare the checked-in descriptors against what the objects derive."""
    problems: list[str] = []
    for rel, documents in sorted(folders.items()):
        folder = content_dir / rel
        present = {path.name for path in _descriptor_files(folder)}
        for name, document in sorted(documents.items()):
            path = folder / name
            if not path.is_file():
                problems.append(f"{rel / name}: missing")
            elif path.read_text(encoding="utf-8") != _serialize(document):
                problems.append(
                    f"{rel / name}: differs from what the objects derive; regenerate"
                )
        for extra in sorted(present - set(documents)):
            problems.append(f"{rel / extra}: names no object in the manifest")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify the authored descriptors for FlyDSL kernels"
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        help="Content directory arch: an LLVM generic target (gfx11-generic) or a "
        "concrete one",
    )
    parser.add_argument(
        "--content-dir",
        type=Path,
        default=CONTENT_DIR,
        help=f"FlyDSL content root holding <op>/<arch>/ (default: {CONTENT_DIR})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the checked-in descriptors and exit; write nothing",
    )
    parser.add_argument(
        "--generic-targets-json",
        type=Path,
        default=DEFAULT_TABLE_PATH,
        help="The plugin SDK's generic-target table, as the packer reads it (CMake "
        "passes HIPDNN_PLUGIN_SDK_GPU_GENERIC_TARGETS_JSON; default: the copy in "
        "this source tree)",
    )
    args = parser.parse_args()

    try:
        entries = collect(args.content_dir, args.arch)
        folders = build_documents(entries, args.arch, args.content_dir)
    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        HkpPackError,
        GeneratorError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.check:
        count = sum(len(documents) for documents in folders.values())
        print(f"Writing {count} descriptor(s) for {args.arch}...")
        _write(args.content_dir, folders)

    problems = _check(args.content_dir, folders)
    if problems:
        for problem in problems:
            print(f"  DESCRIPTOR FAIL: {problem}", file=sys.stderr)
        print(
            f"Error: {len(problems)} descriptor(s) disagree with the objects under "
            f"{args.content_dir}",
            file=sys.stderr,
        )
        return 1

    # The packer's own authoring validator, over the whole FlyDSL folder: shapes,
    # enums, reserved names, and that every id a KDP references resolves.
    try:
        load_flat_input(
            args.content_dir,
            GenericTargets.load(args.generic_targets_json),
            log=lambda *a, **k: None,
        )
    except HkpPackError as exc:
        print(f"Error: descriptor validation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Validated {len(entries)} kernel descriptor(s) for {args.arch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
