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

**The split engine** (``hipkernel:flydsl_sdpa_bwd_split``) is slotted (RFC 0021). Its three
slots are the ops ``sdpa_bwd_prep``, ``sdpa_bwd_dkdv`` and ``sdpa_bwd_dq`` (``SPLIT_SLOT_OPS``),
each keeping its own manifest and content folder. A slot op's folder holds the slot's KMD,
UDD and kernel matcher (the symbols are ``hipkernel.flydsl_sdpa_bwd_split.<slot>.*``), and
its arch pack is a KDP 1.1 naming the slot. The engine's one UED 1.1 lives in the
descriptor-only folder ``<content>/sdpa_bwd_split/`` (no objects, no manifest): the slots,
each with its KMD and its knobs (the tuning fields plus the op's ``ALTERNATIVE_KNOBS``), the
shared workspace token and the native graph match, and none of the members a slotted UED may not carry. It is derived whenever an
arch holds the content of the slot ops, and then of all three.

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
from dataclasses import replace
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
from generators._sdpa_bwd_spec import U2_SYMBOLS, Spec, knobs_of  # noqa: E402

# Descriptor schema version, as the loader spells it: UKD_VERSION_MAJOR = 1,
# UKD_VERSION_MINOR = 0 (DescriptorLoader.hpp:288-289).
DESCRIPTOR_VERSION = "1.0"
# The graph schema each engine's matcher understands, its UED's `sdk_version`: the ingestor
# declines a graph whose schema floor (hipDNN's PluginVersionConstants.hpp,
# computeMinimumEnginePluginApiVersion) is above it before the matcher runs. An op absent
# here stays at the baseline (1.0.0).
#
# * RMSNorm reads a runtime pass-by-value epsilon: K_PASS_BY_VALUE_MIN_API_VERSION, 1.2.0.
# * The SDPA engines read and gate every field the schema has added through
#   K_RAGGED_OFFSET_MULTIPLIER_MIN_VERSION, 1.4.0, which is K_MAX_SUPPORTED_API_VERSION:
#   overridable shapes (1.1.0, declined), runtime pass-by-value tensors (1.2.0), ragged
#   tensors and declared alignments (1.3.0), ragged offset multipliers (1.4.0). COVERAGE.md's
#   "Graph schema" lists the gate for each; TestFlydslSdpaObjects pins the claim against the
#   SDK's maximum, so an SDK that adds a version fails it until those fields are reviewed.
SDPA_OPS = (
    "sdpa_rdna",
    "sdpa_rdna_decode",
    "sdpa_cdna",
    "sdpa_cdna_dualwave",
    "sdpa_cdna_decode",
    "sdpa_aiw",
)
PASS_BY_VALUE_SDK_VERSION = "1.2.0"
SDPA_SDK_VERSION = "1.4.0"
SDK_VERSION_BY_OP: dict[str, str] = {
    "rmsnorm": PASS_BY_VALUE_SDK_VERSION,
    **{op: SDPA_SDK_VERSION for op in SDPA_OPS},
    # The backward engines read ragged tensors (1.3.0), non-default tensor alignment (1.3.0)
    # and ragged offset multipliers (1.4.0) straight from the graph, so they declare the
    # schema that introduced the last of them.
    "sdpa_bwd_split": SDPA_SDK_VERSION,
    "sdpa_bwd_merged": SDPA_SDK_VERSION,
}
# Ops whose engines carry `supports_graph_capture` (capture-safe with default settings,
# benchmarking off). Capture and graph replay are bit-exact on gfx950 (CDNA ops), on
# gfx1152 (sdpa_rdna, sdpa_rdna_decode) and on gfx1201 (the RDNA ops and sdpa_aiw). The
# note is engine-level: capture legality is a property of the shared host dispatch path
# (GenericPlan -> pack launch()), so it holds for every target an engine's objects ship
# for. A launch() enqueues only kernels and async memsets on the handle's stream -- the
# SDPA head-pad copies, the dropout-mask and RNG-dump kernels, sdpa_aiw's fill kernel and the
# zero-bias and by-value-seed memsets included -- with no allocation, no synchronization
# and no host read of device data. The SDPA backward engines (sdpa_bwd_merged,
# sdpa_bwd_split) enqueue only kernels and async seed-word memsets on the handle's stream;
# the integration graph-capture lanes verify them on gfx950 (merged, split) and gfx1101
# (split).
GRAPH_CAPTURE_OPS: tuple[str, ...] = (
    "rmsnorm",
    "rmsnorm_bwd",
    "sdpa_cdna",
    "sdpa_cdna_dualwave",
    "sdpa_cdna_decode",
    "sdpa_rdna",
    "sdpa_rdna_decode",
    "sdpa_aiw",
    "sdpa_bwd_merged",
    "sdpa_bwd_split",
)

# The slotted descriptor formats (RFC 0021 section 9): UED and KDP are 1.1 only when
# slotted; every other document stays at DESCRIPTOR_VERSION.
SLOTTED_VERSION = "1.1"
# The slotted split engine: its descriptor-only content folder, and the ops that are its
# slots in declaration (launch) order. A slot is named as its op minus the common prefix.
SPLIT_ENGINE = "sdpa_bwd_split"
SPLIT_SLOT_OPS = ("sdpa_bwd_prep", "sdpa_bwd_dkdv", "sdpa_bwd_dq")
SPLIT_SHARED_WORKSPACE = "ws.bytes"
# The ops whose rows may be multi-launch beyond the usual set: a row whose launches are not its op's usual ones is authored in its own
# pack, on its own dispatch (RFC 0021: a pack's launch set is uniform). A dQ or merged row that stores dB through the dBias reduce is the
# ``reduce`` pack of its op, launched as a ``dbias_reduce`` entry of its ``sources`` (a standalone companion object on gfx11, a second
# kernel of the row's own object on gfx950); a dK-only row of the value-head split (``u2=dk``), which launches its dV-only partner beside
# it, is the ``u2`` pack of dkdv; a V-head dQ row (``u2=dq``), whose kernel takes ``num_head_v``, is the ``u2`` pack of dq.
# Per op: the dispatch suffix of each pack (its dispatch symbol ``<symbols>_<suffix>``, its UDD ``<stem>_<suffix>.udd.json``) and the
# stem of its file ``<arch>.<stem>.kdp.json``; ``extra_pack`` names the pack a row belongs to.
EXTRA_PACKS = {
    "sdpa_bwd_dq": {"reduce": "reduce", "u2": "dq_u2"},
    "sdpa_bwd_dkdv": {"u2": "u2"},
    "sdpa_bwd_merged": {"reduce": "reduce"},
}
# The companion roles (the first field of a companion's ``spec``): the standalone dBias reduce of a gfx11 dQ row (the reduce of a
# gfx950 dQ or merged row is a second kernel of the row's own object, so it has no manifest entry), and the padded-row zeroing launched after the gradient launches of every dQ and merged row (``generators/gen_sdpa_bwd_tailzero.py``).
REDUCE_ROLE = "dbias_reduce"
TAILZERO_ROLE = "tailzero"
# The V-head tail-zero object, the ``tailzero`` launch of a dQ row of the value-head split (``u2=dq``); and the dV-only dK/dV object
# (``u2=dv``), the second launch of its dK-only partner. Neither is a KMD kernel of its own.
TAILZERO_HV_ROLE = "tailzero_hv"
DVONLY_ROLE = "dvonly"
# The launches of a row, in launch order, and the ops whose rows are launched through ``sources``: the op's own object first, then
# its dBias reduce when it has one, then the tail-zero companion. A dQ row that stores no dB is ``{dq, tailzero}`` and a merged row
# without a reduce is ``{merged, tailzero}``; with a reduce they are ``{dq, dbias_reduce, tailzero}`` and ``{merged, dbias_reduce, tailzero}``.
MAIN_LAUNCH = {"sdpa_bwd_dq": "dq", "sdpa_bwd_merged": "merged"}
# The same, with the ops whose rows are multi-launch only in their extra pack (a dK-only row launches ``dkdv`` then ``dvonly``).
LAUNCH_NAME = {**MAIN_LAUNCH, "sdpa_bwd_dkdv": "dkdv"}
# The packed-tile bound the engines enforce (``PACK_MAX_BLOCK_Q`` in ``FlydslSdpaBwdNativeShared.hpp``), read from the header at
# generation time. The ops whose ``gqa_pack`` rows carry a resolved ``block_q``, which must be in ``(0, bound]``.
NATIVE_SHARED_HEADER = (
    _HERE.parent
    / "src"
    / "engines"
    / "kernel_ingestor_engine"
    / "packs"
    / "FlydslSdpaBwdNativeShared.hpp"
)
PACKED_TILE_OPS = ("sdpa_bwd_dkdv", "sdpa_bwd_merged")


def pack_max_block_q() -> int:
    """``PACK_MAX_BLOCK_Q`` of the engines' shared header."""
    text = NATIVE_SHARED_HEADER.read_text(encoding="utf-8")
    match = re.search(r"\bPACK_MAX_BLOCK_Q\s*=\s*(\d+)\s*;", text)
    if match is None:
        raise HkpPackError(f"no PACK_MAX_BLOCK_Q definition in {NATIVE_SHARED_HEADER}")
    return int(match.group(1))


def companion_identity(companion: dict[str, Any]) -> tuple[str, str]:
    """``(role, dtype)`` of a manifest ``companions`` entry, from its ``spec`` (``<role>,<dtype>[,...][;arch=<arch>]``)."""
    role, dtype = companion["spec"].split(";")[0].split(",")[:2]
    return role, dtype


def slot_of(op: str) -> str | None:
    """The split engine's slot a content op is, or None for an unslotted op."""
    return op.removeprefix("sdpa_bwd_") if op in SPLIT_SLOT_OPS else None


MANIFEST_NAME = "manifest.json"

# `metadataValueFromJson` (DescriptorLoader.hpp:523) accepts bool/int/float/
# string/int-list and *fails on null*, so "this kernel reads N at runtime"
# cannot be spelled the way the manifest spells it. 0 is the sentinel: no shape
# is ever 0 rows, and the matcher reads it as "generic, accepts any N".
GENERIC_N = 0

# Manifest knobs carried as null when not baked in. Only these may take the
# `GENERIC_N` sentinel; a null anywhere else is a generator bug, not a tier.
SHAPE_KNOBS = {"N", "head_dim", "head_dim_v"}

# **What a knob is.** A knob is a performance alternative with overlapping coverage: two
# kernels that serve the same graph and differ in it are interchangeable, and autotune
# chooses between them. A knob is never a feature field, one that decides whether a kernel
# serves a graph at all (dtype, causal or window, dropout, has_bias, the reduce kernel).
# The runtime also requires the kernels that serve one graph to have distinct knob tuples
# (GenericPlanBuilder's enumeration), so a field that tells apart kernels serving the same
# graph must be a knob, whatever else it is; ALTERNATIVE_KNOBS below lists those per op.
#
# Fields the graph fixes rather than the tuner choosing. They belong in the
# metadata schema -- the matcher needs every one of them to pick a variant --
# but not in the engine's `knobs` list, which names what autotune may vary.
# Offering the tuner a dtype or a row length it cannot legally change would
# have it search a space where all but one point fails to match.
#
# Per op: `N` is RMSNorm's row width; `head_dim`, `head_dim_v` and `causal` are what
# an SDPA graph asks for, and on CDNA so is the kernel family, which the target arch
# fixes. `has_bias`, `has_sink`, `has_dropout` and the bias forms (`bias_f32`, `bias_q`,
# `bias_kstrided`) are fixed by whether and how the graph adds a bias or a sink, or drops
# attention probabilities. The
# tier bounds and capability flags (`n_max`, `head_dim_max`, `any_diagonal`,
# `kv_align`, `qk_mask`, `v_mask`) say which graphs an object can serve: the matcher gates on them, and
# autotune compares the objects they distinguish as separate catalog entries. For the
# SDPA backward, `has_dropout`, `padded`, `floor` and `dform` are fixed by the graph and the
# head-size ladder, and `store_db` and `gqa_pack` say which graphs an object serves (a
# dBias graph needs a dB-storing object; a packed-GQA object serves dense GQA only);
# `sink` selects the prep twin that writes the dSink term, and `bias_stage` is gated on
# the bias layout (an LDS-staged bias needs the 16-byte row contract). `bias_f32` (the bias
# is FLOAT beside 16-bit Q/K/V), `bias_stage_resolved` (how the object loads the bias:
# staged in LDS, directly where a staged row serves the graph too, or directly with no
# staged alternative) and `u2_role` (the value-head split: which of Hk == Hv or Hk != Hv
# graphs a row serves) say which graphs an object serves. The matchers keep
# reading all of them; the ones several kernels of an SDPA backward op share a graph on
# are also knobs of that op (ALTERNATIVE_KNOBS).
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
    "v_mask",
    "causal",
    "any_diagonal",
    "kv_align",
    "has_bias",
    "bias_f32",
    "bias_q",
    "bias_kstrided",
    "has_sink",
    "has_dropout",
    "merge_symbol",
    "fill_symbol",
    "kernel",
    "padded",
    "floor",
    "dform",
    "store_db",
    "gqa_pack",
    "sink",
    "bias_stage",
    "bias_stage_resolved",
    "u2_role",
    "grid",
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
    "block_kv",
    "role_order",
    "hints",
    "num_waves",
    "mfma_rows",
    "dkv_shards",
    "block_q",
    "tight_registers",
    "num_stream_buffers",
}

# Graph-determined fields that are also knobs of one op, because kernels of that op with
# overlapping coverage differ in them. The matcher still reads each as a graph fact, so a
# dBias graph matches only dB-storing rows; the knob only addresses the alternatives among
# the rows that match. Per op and graph, the distinct values are:
#   padded      an exact tile and the padded tile of the same size serve a graph of that size
#               (merged: the tile-32 pair)
#   head_dim    the tile: a graph is served by its own tile and, for padded and `g1` rows, by
#               larger ones. It is the build tile (head_dim_v is not a function of it: a dQ
#               row carries the rung below its tile there)
#   floor       tail and `g1` rows of one tile
#   dform       a baked-dropout and a runtime-dropout row
#   gqa_pack    a packed and an unpacked dK/dV (and merged) row serve a grouped graph
#   store_db    a dB-storing and a dB-less dQ (and merged) row serve a bias graph without dBias
#   bias_stage  a staged and a direct bias row
# `tests/test_sdpa_bwd_enumeration.py` checks both directions over the shipped catalogue.
ALTERNATIVE_KNOBS: dict[str, frozenset[str]] = {
    "sdpa_bwd_prep": frozenset({"head_dim", "padded", "floor"}),
    "sdpa_bwd_dkdv": frozenset({"head_dim", "padded", "floor", "dform", "gqa_pack"}),
    "sdpa_bwd_dq": frozenset(
        {"head_dim", "padded", "floor", "dform", "bias_stage", "store_db"}
    ),
    "sdpa_bwd_merged": frozenset({"padded", "gqa_pack", "store_db"}),
}


def knob_fields(op: str, fields: Any) -> list[str]:
    """The fields of @p fields that @p op's engine exposes as knobs, in schema order."""
    return [
        f
        for f in fields
        if f in TUNING_FIELDS or f in ALTERNATIVE_KNOBS.get(op, frozenset())
    ]


# Every op that owns its engine carries an engine estimate (RFC 0019 §3.1 `predict_engine`,
# L1); the slotted split engine (`sdpa_bwd_split`) writes a UED of its own but has none, and no
# slot is planned for it. The estimate is the expected tflops of the engine's own pick, which engine
# selection reads before any catalog is enumerated. An engine without one is ordered by
# static rules whatever the graph, and FlyDSL's engines compete for graphs: on gfx1201
# sdpa_rdna, sdpa_rdna_decode and sdpa_aiw each match nearly every graph of the others'
# corpora.
#
# The estimate's `default` slot. No scorer is registered under its symbol, so the loader
# disables the estimate and keeps the engine (RFC 0019 §11.2), exactly as an engine with no
# estimate behaves; a trained model installed for an arch (below) is bound under that
# arch's key and takes precedence there.
ENGINE_ESTIMATE_METRIC = "tflops"

# Trained models live beside the op's descriptors where `uhd_gen promote` installs them,
# `heuristics/<ued-id>/<role>/<arch>/<metric>/` (a metric-less model directly under
# `<role>/<arch>/`), and the UED binds each under its role and arch key. The generator owns
# the UED, so it derives those bindings from what is installed: a regeneration keeps every
# trained model, and `--check` accepts a tree with models in it. A model under a gfx key
# wins over `default` on that arch (RFC 0020 §4.6), so the native ranker and the estimate
# slot stay the fallback everywhere else.
#
# `uhd_gen promote` is the reference for every binding: the roles it accepts, where it
# installs, and the form it writes (uhd_gen/promote.py `_rebind`). Every role binds a list per
# arch key, one UHD per ranking metric -- `predict_applicable_kernels`, the candidate
# generators for future JIT spaces, included. The IngestorGenerator emits the same form from
# its `engine.heuristics` config (#15101).
HEURISTICS_DIR = "heuristics"
_ROLES = ("sort_kernel_catalog", "predict_engine", "predict_applicable_kernels")
_ARCH_KEY = re.compile(r"default|gfx[0-9a-f]+")


def installed_models(
    op_dir: Path, ued_id: str
) -> dict[str, dict[str, list[tuple[str | None, str]]]]:
    """Role -> arch key -> [(metric or None, UHD id)] of the trained models installed for an op.

    Refuses a model filed anywhere but its own engine's, role's, arch's and (if it has one)
    metric's folder, one trained against another engine, and two models one key would bind
    for the same metric -- the loader would disable that metric there (RFC 0019 §3.1).
    """
    root = op_dir / HEURISTICS_DIR
    installed: dict[str, dict[str, list[tuple[str | None, str]]]] = {}
    for path in sorted(root.rglob("*.uhd.json")) if root.is_dir() else []:
        parts = path.relative_to(root).parts
        if len(parts) not in (4, 5) or parts[0] != ued_id:
            raise HkpPackError(
                f"{path}: a trained model is installed at "
                f"{HEURISTICS_DIR}/{ued_id}/<role>/<arch>/[<metric>/]"
            )
        role, arch = parts[1], parts[2]
        folder_metric = parts[3] if len(parts) == 5 else None
        if role not in _ROLES or not _ARCH_KEY.fullmatch(arch):
            raise HkpPackError(
                f"{path}: '{role}/{arch}' is not a role and arch key uhd_gen promote "
                f"binds ({', '.join(_ROLES)}; default or gfx<id>)"
            )
        doc = json.loads(path.read_text(encoding="utf-8"))
        metric = (doc.get("score") or {}).get("metric")
        if metric != folder_metric:
            raise HkpPackError(
                f"{path}: filed under metric {folder_metric!r} but its score.metric is "
                f"{metric!r}; promote files a model under its metric, a metric-less one "
                "directly under the arch"
            )
        trained = ((doc.get("trained_against") or {}).get("ued") or {}).get("id")
        if trained != ued_id:
            raise HkpPackError(
                f"{path}: trained against UED {trained!r}, not this engine's {ued_id}"
            )
        models = installed.setdefault(role, {}).setdefault(arch, [])
        if any(existing == metric for existing, _ in models):
            raise HkpPackError(
                f"{path}: a second {metric or '(metric-less)'} model for {role}/{arch}; "
                "the loader would disable that metric on that key"
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
    # The companion objects the row launches beside its own, by role (``REDUCE_ROLE``, ``TAILZERO_ROLE``, ``DVONLY_ROLE``): the
    # manifest ``companions`` entries of the row (the reduce of its bias dtype, the tail-zero of its dtype, which is the V-head object
    # for a ``u2=dq`` row; the dV-only ``instances`` record of a ``u2=dk`` row) plus the path of each object. None for a row with none.
    companions: dict[str, dict[str, Any]] | None = None


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
        companions = manifest.get("companions", [])
        named = {record["file"] for record in manifest["instances"]}
        named |= {companion["file"] for companion in companions}
        stray = sorted(
            p.name for p in manifest_path.parent.glob("*.hsaco") if p.name not in named
        )
        if stray:
            raise ValueError(
                f"{manifest_path.parent}: {len(stray)} object(s) the manifest does not "
                f"name, e.g. '{stray[0]}'. Delete them: they are left over from a "
                "previous generation and would ship unreferenced."
            )
        by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for companion in companions:
            path = manifest_path.parent / companion["file"]
            if not path.is_file():
                raise FileNotFoundError(
                    f"{manifest_path} names companion '{companion['file']}' but {path} does not exist"
                )
            data = path.read_bytes()
            actual = hashlib.sha256(data).hexdigest()
            if actual != companion["sha256"]:
                raise ValueError(
                    f"{path}: SHA256 {actual} does not match the {companion['sha256']} "
                    f"recorded in {manifest_path}. Regenerate the object rather than "
                    "editing the manifest."
                )
            where = str(path)
            verify_arch(describe(data, where, companion["symbol"]), arch, where)
            verify_generic(data, arch, where)
            key = companion_identity(companion)
            if key in by_key:
                raise ValueError(
                    f"{manifest_path}: two companions for role {key[0]}, dtype {key[1]}"
                )
            by_key[key] = {**companion, "path": path}
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
            knobs = record["knobs"]
            # A value-head split object defines the kernel of its own (the dV-only object that of the dK-only one).
            u2 = knobs.get("u2")
            if u2 in ("dk", "dv", "dq"):
                want = U2_SYMBOLS[arch, "dq" if u2 == "dq" else "dkdv"]
                if record["symbol"] != want:
                    raise ValueError(
                        f"{path}: a u2={u2} row of {arch} defines {record['symbol']!r}, not {want!r}"
                    )
            # The dB reduce of a row is keyed by its bias dtype (FLOAT for a ``bias_f32`` row beside 16-bit Q); the tail-zero by its
            # dtype, the V-head object for a row of the value-head split.
            bias_dtype = "fp32" if knobs.get("bias_f32") == 1 else record["dtype"]
            tailzero_role = (
                TAILZERO_HV_ROLE if knobs.get("u2") == "dq" else TAILZERO_ROLE
            )
            row_companions = {
                role: by_key[source_role, dtype]
                for role, source_role, dtype in (
                    (REDUCE_ROLE, REDUCE_ROLE, bias_dtype),
                    (TAILZERO_ROLE, tailzero_role, record["dtype"]),
                )
                if (source_role, dtype) in by_key
            }
            # A dQ or merged row that stores dB through a reduce names its symbol; the reduce is a standalone companion object (gfx11) or a
            # second kernel of the row's own object (gfx950), launched through ``sources`` either way. A row with no ``reduce_symbol``
            # launches none.
            reduce = row_companions.pop(REDUCE_ROLE, None)
            reduce_symbol = knobs.get("reduce_symbol") if op in MAIN_LAUNCH else None
            standalone = any(role == REDUCE_ROLE for role, _ in by_key)
            if reduce_symbol:
                if reduce is None and standalone:
                    raise ValueError(
                        f"{path}: names reduce symbol {reduce_symbol!r}, "
                        f"the manifest has no {bias_dtype} {REDUCE_ROLE} companion"
                    )
                if reduce is None:
                    reduce = {
                        "file": record["file"],
                        "symbol": reduce_symbol,
                        "path": path,
                    }
                elif reduce["symbol"] != reduce_symbol:
                    raise ValueError(
                        f"{path}: names reduce symbol {reduce_symbol!r}, "
                        f"the {bias_dtype} companion is {reduce['symbol']!r}"
                    )
                row_companions[REDUCE_ROLE] = reduce
            if op in MAIN_LAUNCH and TAILZERO_ROLE not in row_companions:
                raise ValueError(
                    f"{path}: {op} rows launch the {tailzero_role} companion, but {manifest_path} has none "
                    f"for dtype {record['dtype']}"
                )
            entries.append(KernelEntry(op, path, data, record, row_companions or None))
    return attach_dvonly(entries)


def attach_dvonly(entries: list[KernelEntry]) -> list[KernelEntry]:
    """@p entries with each dV-only dK/dV row (``u2=dv``) attached to its dK-only row (``u2=dk``) and no longer an entry of its own.

    A dV-only object is no KMD kernel: it is the second launch of the dK-only row whose Spec is equal to its own apart from ``u2``
    (``DVONLY_ROLE`` of that row's ``companions``). The two are launched over one workgroup shape, so a pair whose ``block_threads`` or
    ``block_kv`` differ is refused, as is a dK-only row without a partner and a dV-only row without one.
    """

    def role(entry: KernelEntry) -> str:
        """The ``u2`` knob of a dK/dV row as the manifest spells it (``auto``: none)."""
        return (
            entry.record["knobs"].get("u2", "auto")
            if entry.op == "sdpa_bwd_dkdv"
            else "auto"
        )

    dv_rows: dict[Spec, KernelEntry] = {}
    for entry in entries:
        if role(entry) == "dv":
            dv_rows[replace(Spec.parse(entry.record["spec"]), u2="dk")] = entry
    kept: list[KernelEntry] = []
    paired: set[Spec] = set()
    for entry in entries:
        if role(entry) == "auto":
            kept.append(entry)
        elif role(entry) == "dk":
            spec = Spec.parse(entry.record["spec"])
            partner = dv_rows.get(spec)
            if partner is None:
                raise ValueError(
                    f"{entry.record['name']}: a dK-only row (u2=dk) has no dV-only row (u2=dv) equal to it apart from u2"
                )
            for field in ("block_threads", "block_kv"):
                if entry.record["knobs"].get(field) != partner.record["knobs"].get(
                    field
                ):
                    raise ValueError(
                        f"{entry.record['name']} and {partner.record['name']}: the dK-only and dV-only rows launch over one "
                        f"workgroup shape, but {field} is {entry.record['knobs'].get(field)} and "
                        f"{partner.record['knobs'].get(field)}"
                    )
            paired.add(spec)
            kept.append(
                entry._replace(
                    companions={
                        **(entry.companions or {}),
                        DVONLY_ROLE: {
                            "spec": partner.record["spec"],
                            "file": partner.record["file"],
                            "symbol": partner.record["symbol"],
                            "sha256": partner.record["sha256"],
                            "path": partner.path,
                        },
                    }
                )
            )
    unpaired = sorted(set(dv_rows) - paired)
    if unpaired:
        raise ValueError(
            f"{len(unpaired)} dV-only row(s) (u2=dv) have no dK-only row (u2=dk) equal to them apart from u2, e.g. "
            f"{dv_rows[unpaired[0]].record['name']}"
        )
    return kept


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


def rne_pin_problem(spec: Spec) -> str | None:
    """Why @p spec may not ship, or None: a bf16 dK/dV or dQ pin of an arch with the ``rne`` knob must spell ``rne=1``.

    Without it the kernel narrows P and dS to bf16 by truncation, which biases the gradients; the shipped rows
    round to nearest even. This is a shipping rule, not a Spec rule: a sweep may build and measure ``rne=0``
    objects, but none may be installed (``tools/sdpa_bwd_tables_from_sweep.py``) or described here.

    The rule covers every arch whose knob list (``knobs_of``) has ``rne``: today the generic gfx11 and gfx12 targets.
    There it also covers the dB output: a dQ row with a bias stores its dS packs as dB, so a truncated dS is a
    truncated user-visible dB. gfx950 has no ``rne`` knob and is exempt: its bf16 narrowing is the hardware's
    round-to-nearest-even conversion (``v_cvt_pk_bf16_f32``), the same for every row.
    """
    if spec.kernel not in ("dkdv", "dq") or spec.dtype != "bf16":
        return None
    if "rne" not in knobs_of(spec.kernel, spec.arch, spec.dtype) or spec.rne == 1:
        return None
    return (
        f"{spec.key()}: a bf16 {spec.kernel} row of {spec.arch} must spell rne=1 "
        "(without it P and dS are truncated, not rounded)"
    )


def _spec_keys(entry: KernelEntry) -> list[str]:
    """The Spec keys of the objects of @p entry: its own and, for a dK-only row, its dV-only partner's."""
    dvonly = (entry.companions or {}).get(DVONLY_ROLE)
    return [entry.record["spec"], *([dvonly["spec"]] if dvonly else [])]


def require_rne_pins(entries: list[KernelEntry]) -> None:
    """Raise ValueError naming every shipped SDPA backward dK/dV or dQ row that ``rne_pin_problem`` refuses (a dV-only partner included)."""
    problems = [
        f"{entry.path.name}: {problem}"
        for entry in entries
        if entry.op in ("sdpa_bwd_dkdv", "sdpa_bwd_dq")
        for key in _spec_keys(entry)
        if (problem := rne_pin_problem(Spec.parse(key))) is not None
    ]
    if problems:
        shown = "; ".join(problems[:3]) + (", ..." if len(problems) > 3 else "")
        raise ValueError(f"{len(problems)} shipped row(s) break the rne rule: {shown}")


class OpDescriptors:
    """The one-per-op descriptors: schema, engine, heuristic, dispatch, matcher.

    A slot op of the split engine owns only the slot's share: its KMD, dispatch and
    kernel matcher, on the engine's symbols, ids and shared UED (``SplitEngine``).

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
        installed: dict[str, dict[str, list[tuple[str | None, str]]]] | None = None,
    ) -> None:
        self.op = op
        self.slot = slot_of(op)
        self.stem = f"flydsl_{op}"
        self.fields = fields
        if self.slot is None:
            self.symbols = f"hipkernel.{self.stem}"
            engine = op
        else:
            engine = SPLIT_ENGINE
            self.symbols = f"hipkernel.flydsl_{engine}.{self.slot}"
        self.installed = installed or {}

        self.kmd_id = _uuid(f"kmd/{op}")
        self.uhd_id = _uuid(f"uhd/{op}")
        self.udd_id = _uuid(f"udd/{op}")
        # The dispatches of the op's extra packs (``EXTRA_PACKS``): dispatch suffix -> file stem.
        self.extra_packs = EXTRA_PACKS.get(op, {})
        self.ued_id = _uuid(f"ued/{engine}")
        self.umd_id = _uuid(f"umd/{op}/kernel")
        self.estimate_uhd_id = _uuid(f"uhd/{op}/predict_engine")

    def documents(self) -> dict[str, dict[str, Any]]:
        documents = self._documents()
        if self.slot is None:
            return documents
        # A slot's share: the engine's UED and ranker are the split engine's own.
        return {
            name: document
            for name, document in documents.items()
            if name.endswith((".kmd.json", ".udd.json", ".umd.json"))
        }

    def _documents(self) -> dict[str, dict[str, Any]]:
        ued: dict[str, Any] = {
            "version": DESCRIPTOR_VERSION,
            "id": self.ued_id,
            "name": f"hipkernel:{self.stem}",
        }
        # The graph schema the engine's matcher understands (SDK_VERSION_BY_OP): the
        # ingestor declines any graph whose schema floor is above it.
        if self.op in SDK_VERSION_BY_OP:
            ued["sdk_version"] = SDK_VERSION_BY_OP[self.op]
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

        # The dispatches of the extra packs of an op (``EXTRA_PACKS``): shared documents, authored with
        # every arch's pack of the op whether or not that arch has such rows.
        extra_udd = {
            f"{self.stem}_{suffix}.udd.json": {
                "version": DESCRIPTOR_VERSION,
                "id": _uuid(f"udd/{self.op}/{suffix}"),
                "name": f"flydsl {self.op} {suffix} dispatch",
                "dispatch_symbol": f"{self.symbols}_{suffix}.dispatch",
            }
            for suffix in self.extra_packs
        }
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
        # The UHD pipeline keys every role of an engine one way: once a ranker is bound for
        # an arch, it trains an estimate for that arch only if `predict_engine` binds the
        # arch too. So each such arch gets the estimate's slot, which the loader disables
        # just as it does under `default`. The ranker map is left alone: it is an input of
        # the selector revision every trained estimate is checked against, and this role
        # is not.
        for arch in roles["sort_kernel_catalog"]:
            roles["predict_engine"].setdefault(
                arch, [(ENGINE_ESTIMATE_METRIC, self.estimate_uhd_id)]
            )
        # Each value as `uhd_gen promote` writes it, so promoting into this UED and
        # regenerating bind the same values (promote appends a new arch key; this sorts): a
        # key a trained model is bound under is a list (one UHD per metric), as promote's
        # rebinding leaves it; a key only the generator fills (the native ranker, the
        # estimate's slot) stays the bare id it writes.
        trained = {
            model
            for by_arch in self.installed.values()
            for models in by_arch.values()
            for _, model in models
        }

        def bound(ids: list[tuple[str | None, str]]) -> str | list[str]:
            models = [model for _, model in ids]
            if trained.intersection(models):
                return models
            return models[0] if len(models) == 1 else models

        for role, by_arch in roles.items():
            engine[role] = {
                arch: bound(ids)
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
        engine["knobs"] = knob_fields(self.op, self.fields)
        if self.op in GRAPH_CAPTURE_OPS:
            engine["behavior_notes"] = ["supports_graph_capture"]

        return {
            **extra_udd,
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


def _require_packed_tile(entry: KernelEntry, bound: int) -> None:
    """A packed (``gqa_pack`` 1) dK/dV or merged row carries its resolved ``block_q``, which the engines' packed-span bound covers.

    The engines check the row's ``block_q`` against ``PACK_MAX_BLOCK_Q`` (``FlydslSdpaBwdNativeShared.hpp``) before launching it; a row
    whose tile is missing, not positive or above the bound would pass a graph check that assumes a smaller tile, so it is refused
    here, where the row is made. The value reaches the KMD and the KDP as the row's ``block_q`` knob.
    """
    if entry.op not in PACKED_TILE_OPS or entry.record["knobs"].get("gqa_pack") != 1:
        return
    block_q = entry.record["knobs"].get("block_q")
    if (
        isinstance(block_q, bool)
        or not isinstance(block_q, int)
        or not 0 < block_q <= bound
    ):
        raise HkpPackError(
            f"{entry.record['name']}: a packed row needs its resolved block_q in (0, {bound}] "
            f"(PACK_MAX_BLOCK_Q of {NATIVE_SHARED_HEADER.name}), got {block_q!r}"
        )


def _lds_tile_limits(entries: list[KernelEntry]) -> dict[str, int]:
    """Per dtype, the largest head-dim tile of a dQ row of @p entries that stages its bias in LDS (0 where none does)."""
    limits: dict[str, int] = {}
    for entry in entries:
        knobs = entry.record["knobs"]
        if knobs.get("bias_stage_resolved") == "lds":
            dtype = entry.record["dtype"]
            limits[dtype] = max(limits.get(dtype, 0), knobs["head_dim"])
    return limits


def kmd_knobs(entry: KernelEntry, lds_limits: dict[str, int]) -> dict[str, Any]:
    """The knobs of @p entry as its KMD spells them.

    The manifest spells an unset knob ``-1`` or ``"auto"`` and the build's resolutions as it measured them; the KMD says what a row serves:

    * ``bias_f32``: 1 for a FLOAT bias beside 16-bit Q/K/V, else 0;
    * ``u2`` becomes ``u2_role``: ``none``, ``dk`` (the dK-only object, which launches its dV-only partner) or ``dq`` (the V-head dQ);
    * ``bias_stage_resolved``: the build's ``lds`` or ``direct`` (``na`` without a bias), with ``direct`` split: ``direct`` where a
      staged row serves the graphs of the row too, ``direct_only`` where none does: every row that is not 16-bit, every FLOAT-bias
      row, and every row whose head-dim tile is above the largest tile of its dtype that stages (@p lds_limits).

    A field the manifest does not carry is not added (the schema is whatever the op's rows carry).
    """
    knobs = dict(entry.record["knobs"])
    if "bias_f32" in knobs:
        knobs["bias_f32"] = int(knobs["bias_f32"] == 1)
    if "u2" in knobs:
        role = {"auto": "none", "dk": "dk", "dq": "dq"}.get(knobs.pop("u2"))
        if role is None:
            raise HkpPackError(
                f"{entry.record['name']}: a dV-only row (u2=dv) is no KMD kernel; attach_dvonly pairs it with its dK-only row"
            )
        knobs["u2_role"] = role
    if knobs.get("bias_stage_resolved") == "direct" and (
        knobs.get("bias_f32") == 1
        or knobs["head_dim"] > lds_limits.get(entry.record["dtype"], 0)
    ):
        knobs["bias_stage_resolved"] = "direct_only"
    return knobs


def _fields_for_op(entries: list[KernelEntry]) -> dict[str, str]:
    """The metadata schema for an op, derived from its instances' knobs (``kmd_knobs``).

    Every instance must declare the same knob set: a schema is one shape, and an
    instance that carries a knob its siblings do not is a generator bug that
    would otherwise surface as a silently unselectable kernel.
    """
    fields: dict[str, str] = {}
    first: dict[str, Any] | None = None
    limits = _lds_tile_limits(entries)
    for entry in entries:
        knobs = kmd_knobs(entry, limits)
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
                f"(found in {homes or 'none'}); only TUNING_FIELDS and the op's "
                f"ALTERNATIVE_KNOBS become knobs"
            )
    return fields


def extra_pack(entry: KernelEntry) -> str | None:
    """The dispatch suffix of the extra pack (``EXTRA_PACKS``) @p entry belongs to, None for a row of its op's standard pack.

    A dQ or merged row that stores dB through a reduce is ``reduce``; a V-head dQ row (``u2=dq``) is ``u2``; a dK-only row (it has a
    dV-only partner) is ``u2``.
    """
    companions = entry.companions or {}
    if entry.op == "sdpa_bwd_dq":
        if entry.record["knobs"].get("u2") == "dq":
            return "u2"
        return "reduce" if REDUCE_ROLE in companions else None
    if entry.op == "sdpa_bwd_merged":
        return "reduce" if REDUCE_ROLE in companions else None
    if entry.op == "sdpa_bwd_dkdv":
        return "u2" if DVONLY_ROLE in companions else None
    return None


def _arch_documents(
    entries: list[KernelEntry], op: OpDescriptors, arch: str
) -> dict[str, dict[str, Any]]:
    """One arch's pack of an op: its KDP, holding one inline ``hsaco`` UKD per object.

    Inline rather than one file per object: no UKD is shared between packs, the
    specialization contract is declared once on the KDP and inherited by every
    kernel, and the packaging probes take only inline UKDs.
    """
    kernels: list[dict[str, Any]] = []
    extra_kernels: dict[str, list[dict[str, Any]]] = {}
    limits = _lds_tile_limits(entries)
    # A generic target's objects run on every member, so the pack and each UKD
    # list them all and the packer ships the bytes into each member's shard.
    runs_on = list(families.members(arch))

    bound = pack_max_block_q()
    for entry in entries:
        record = entry.record
        _require_packed_tile(entry, bound)
        # Named as it was when these were emitted in shipped form, where the
        # string was the archive key, so the id did not change with the form.
        ukd_id = _uuid(f"ukd/{arch}/{entry.op}/{record['file']}")

        metadata = {"dtype": record["dtype"]}
        for key, value in kmd_knobs(entry, limits).items():
            metadata[key] = _metadata_value(key, value)

        document = {
            "version": DESCRIPTOR_VERSION,
            "id": ukd_id,
            "name": record["name"],
            "arch": runs_on,
        }
        if entry.companions is None:
            document["kernel_source"] = {
                "kind": "hsaco",
                # Beside the KDP: the packer resolves an inline kernel's file
                # relative to the folder of the KDP that holds it.
                "file": record["file"],
                "symbol": record["symbol"],
            }
        else:
            # A row launched through `sources`: its own object, then its dBias reduce (a dB-storing dQ or merged row: a
            # standalone companion object on gfx11, a second kernel of the row's own object on gfx950), its dV-only partner (a
            # dK-only row) and its tail-zero companion, in launch order.
            reduce = entry.companions.get(REDUCE_ROLE)
            if (
                reduce is not None
                and record["knobs"]["reduce_symbol"] != reduce["symbol"]
            ):
                raise HkpPackError(
                    f"{record['name']}: reduce_symbol {record['knobs']['reduce_symbol']!r} "
                    f"is not the symbol {reduce['symbol']!r} of its dbias_reduce source"
                )
            document["sources"] = {
                LAUNCH_NAME[entry.op]: {
                    "kind": "hsaco",
                    "file": record["file"],
                    "symbol": record["symbol"],
                },
                **{
                    role: {
                        "kind": "hsaco",
                        "file": entry.companions[role]["file"],
                        "symbol": entry.companions[role]["symbol"],
                    }
                    for role in (REDUCE_ROLE, DVONLY_ROLE, TAILZERO_ROLE)
                    if role in entry.companions
                },
            }
        document["metadata"] = metadata
        document["priority"] = record["priority"]
        suffix = extra_pack(entry)
        (kernels if suffix is None else extra_kernels.setdefault(suffix, [])).append(
            document
        )

    def pack(suffix: str, dispatch: str, descriptors: list[dict[str, Any]]):
        kdp = {
            "version": DESCRIPTOR_VERSION if op.slot is None else SLOTTED_VERSION,
            "id": _uuid(f"kdp/{op.op}/{arch}{suffix}"),
            "name": f"flydsl {op.op} kernels ({arch})",
            "arch": runs_on,
            "matchers": [op.umd_id],
            "engine": op.ued_id,
            "dispatch": dispatch,
            "provenance": op.provenance(),
            "kernelDescriptors": descriptors,
        }
        if op.slot is not None:
            kdp["slot"] = op.slot
        return kdp

    documents = {f"{op.stem}.kdp.json": pack("", op.udd_id, kernels)}
    for suffix, rows in sorted(extra_kernels.items()):
        if suffix not in op.extra_packs:
            raise HkpPackError(f"{op.op} has no {suffix} dispatch for its rows")
        documents[f"{arch}.{op.extra_packs[suffix]}.kdp.json"] = pack(
            f"/{suffix}", _uuid(f"udd/{op.op}/{suffix}"), rows
        )
    return documents


def _require_injective_slot(op: OpDescriptors, kdps: list[dict[str, Any]]) -> None:
    """A slot's KMD tuples are unique among its kernels, across the packs of the slot.

    The loader drops a slotted pack whose metadata tuple repeats one in its slot, so a
    generator that emitted two such rows would ship neither silently.
    """
    seen: dict[str, str] = {}
    for kernel in (k for kdp in kdps for k in kdp["kernelDescriptors"]):
        key = json.dumps(kernel["metadata"], sort_keys=True)
        if key in seen:
            raise HkpPackError(
                f"slot '{op.slot}' ({op.op}): {kernel['name']} and {seen[key]} have the "
                f"same metadata {key}; a slot's kernels must differ in their metadata"
            )
        seen[key] = kernel["name"]


def _require_disjoint_ids(slots: dict[str, list[dict[str, Any]]]) -> None:
    """No kernel id (nor pack id) appears in two slots of the engine, or twice in one."""
    owner: dict[str, str] = {}
    for slot, kdps in slots.items():
        ids = [
            identity
            for kdp in kdps
            for identity in (kdp["id"], *(k["id"] for k in kdp["kernelDescriptors"]))
        ]
        for identity in ids:
            if identity in owner:
                raise HkpPackError(
                    f"id {identity} is used by slot '{owner[identity]}' and slot '{slot}'"
                    if owner[identity] != slot
                    else f"id {identity} is used twice in slot '{slot}'"
                )
            owner[identity] = slot


class SplitEngine:
    """The split engine's descriptor-only content: its UED 1.1.

    Each slot names its own KMD and carries only the tuning fields of its own schema as
    knobs. The engine has no ranker (the loader falls back to priority, then id), no
    top-level ``metadata``, ``sort_kernel_catalog``, ``knobs`` or
    ``predict_applicable_kernels``, and does not author the serialization note: a slotted
    engine's plans cannot be saved.
    """

    def __init__(self, slot_ops: dict[str, OpDescriptors]) -> None:
        self.slot_ops = slot_ops
        self.stem = f"flydsl_{SPLIT_ENGINE}"
        self.symbols = f"hipkernel.{self.stem}"

    def documents(self) -> dict[str, dict[str, Any]]:
        ued: dict[str, Any] = {
            "version": SLOTTED_VERSION,
            "id": _uuid(f"ued/{SPLIT_ENGINE}"),
            "name": f"hipkernel:{self.stem}",
            # Pass-by-value scalars (softmax scale, dropout seed and offset), ragged
            # tensors, non-default alignment and ragged offset multipliers.
            "sdk_version": SDK_VERSION_BY_OP[SPLIT_ENGINE],
            "graph_match": {"native": f"{self.symbols}.graph_match"},
            "shared_workspace": SPLIT_SHARED_WORKSPACE,
            "slots": [
                {
                    "name": slot.slot,
                    "metadata": slot.kmd_id,
                    "knobs": knob_fields(slot.op, slot.fields),
                }
                for slot in self.slot_ops.values()
            ],
        }
        if SPLIT_ENGINE in GRAPH_CAPTURE_OPS:
            ued["behavior_notes"] = ["supports_graph_capture"]
        return {f"{self.stem}.ued.json": ued}


def build_documents(
    entries: list[KernelEntry], arch: str, content_dir: Path = CONTENT_DIR
) -> dict[Path, dict[str, dict[str, Any]]]:
    """Every document for `arch`, grouped by the folder (relative to the content
    root) it is written into."""
    by_op: dict[str, list[KernelEntry]] = {}
    for entry in entries:
        by_op.setdefault(entry.op, []).append(entry)

    folders: dict[Path, dict[str, dict[str, Any]]] = {}
    slot_ops: dict[str, OpDescriptors] = {}
    slot_packs: dict[str, list[dict[str, Any]]] = {}
    for op_name, op_entries in sorted(by_op.items()):
        installed = installed_models(content_dir / op_name, _uuid(f"ued/{op_name}"))
        op = OpDescriptors(op_name, _fields_for_op(op_entries), installed)
        folders[Path(op_name)] = op.documents()
        folders[Path(op_name) / arch] = pack = _arch_documents(op_entries, op, arch)
        if op.slot is not None:
            slot_ops[op_name] = op
            slot_packs[op.slot] = [
                document
                for name, document in pack.items()
                if name.endswith(".kdp.json")
            ]
            _require_injective_slot(op, slot_packs[op.slot])

    if slot_ops:
        missing = [op for op in SPLIT_SLOT_OPS if op not in slot_ops]
        if missing:
            raise HkpPackError(
                f"{arch}: the split engine's slots are {list(SPLIT_SLOT_OPS)}; "
                f"no content for {missing}"
            )
        _require_disjoint_ids(slot_packs)
        engine = SplitEngine({op: slot_ops[op] for op in SPLIT_SLOT_OPS})
        folders[Path(SPLIT_ENGINE)] = engine.documents()
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
        require_rne_pins(entries)
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
