# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The generator binds the trained models installed beside an op, as ``uhd_gen promote`` does.

``uhd_gen promote`` files a model at ``heuristics/<ued-id>/<role>/<arch>/<metric>/`` (a
metric-less one directly under ``<role>/<arch>/``) and binds it in the UED as a list per arch
key, for every role. ``gen_descriptors.py`` owns
the UED, so it must bind every such model the same way -- otherwise a regeneration drops the
model, rewrites promote's binding, or ``--check`` refuses a tree that ships one. Host-only, no
``dvc pull``.
"""

from __future__ import annotations

import copy
import json
import re
import sys
import uuid
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
import gen_descriptors  # noqa: E402

OP = "rmsnorm_bwd"
FIELDS = {"dtype": "string", "N": "int"}
UED_ID = gen_descriptors.ued_id(OP, FIELDS)


def _install(
    op_dir: Path,
    role: str,
    arch: str,
    model_id: str,
    metric: str | None = "tflops",
    **overrides,
) -> Path:
    doc = (
        {
            "id": model_id,
            "trained_against": {"ued": {"id": UED_ID}},
        }
        | ({"score": {"metric": metric}} if metric else {})
        | overrides
    )
    folder = op_dir / "heuristics" / UED_ID / role / arch
    if metric:
        folder /= metric
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{model_id}.uhd.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _ued(op_dir: Path) -> dict:
    installed = gen_descriptors.installed_models(op_dir, UED_ID)
    docs = gen_descriptors.OpDescriptors(OP, FIELDS, installed).documents()
    return docs[f"flydsl_{OP}.ued.json"]


def test_an_arch_model_binds_under_its_arch_alone(tmp_path: Path) -> None:
    _install(tmp_path, "predict_engine", "gfx950", "model-950")
    _install(tmp_path, "sort_kernel_catalog", "gfx942", "ranker-942")
    ops = gen_descriptors.OpDescriptors(OP, FIELDS)
    ued = _ued(tmp_path)
    # The estimate binds only the arch with an installed model: no `default`, and no key for
    # gfx942, which has a ranker but no model.
    assert ued["predict_engine"] == {"gfx950": ["model-950"]}
    assert ued["sort_kernel_catalog"] == {
        "default": ops.uhd_id,
        "gfx942": ["ranker-942"],
    }


def test_the_estimate_binds_exactly_the_arches_with_a_model(tmp_path: Path) -> None:
    _install(tmp_path, "predict_engine", "gfx950", "model-950")
    _install(tmp_path, "predict_engine", "gfx1100", "model-1100")
    _install(tmp_path, "predict_engine", "gfx942", "model-942")
    # Sorted with `default` first, one list per key, and no key for an arch without a model.
    assert _ued(tmp_path)["predict_engine"] == {
        "gfx1100": ["model-1100"],
        "gfx942": ["model-942"],
        "gfx950": ["model-950"],
    }


def test_a_default_model_joins_the_native_ranker_only_by_metric(tmp_path: Path) -> None:
    # The native ranker declares no metric, so a tflops ranker joins it; the estimate has no
    # native entry, so its `default` is the model alone.
    _install(tmp_path, "predict_engine", "default", "model")
    _install(tmp_path, "sort_kernel_catalog", "default", "ranker")
    ued = _ued(tmp_path)
    assert ued["predict_engine"] == {"default": ["model"]}
    native = gen_descriptors.OpDescriptors(OP, FIELDS).uhd_id
    assert ued["sort_kernel_catalog"] == {"default": [native, "ranker"]}


def test_no_installed_model_leaves_no_estimate(tmp_path: Path) -> None:
    ops = gen_descriptors.OpDescriptors(OP, FIELDS)
    documents = ops.documents()
    ued = _ued(tmp_path)
    assert ued == documents[f"flydsl_{OP}.ued.json"]
    assert "predict_engine" not in ued
    assert ued["sort_kernel_catalog"] == {"default": ops.uhd_id}
    assert not [name for name in documents if "engine_estimate" in name]


def test_a_ranker_alone_gives_the_estimate_no_key(tmp_path: Path) -> None:
    _install(tmp_path, "sort_kernel_catalog", "gfx942", "ranker-942")
    assert "predict_engine" not in _ued(tmp_path)


def test_a_leftover_engine_estimate_file_is_stale(tmp_path: Path) -> None:
    documents = gen_descriptors.OpDescriptors(OP, FIELDS).documents()
    folders = {Path(OP): documents}
    gen_descriptors._write(tmp_path, folders)
    assert gen_descriptors._check(tmp_path, folders) == []

    leftover = tmp_path / OP / f"flydsl_{OP}_engine_estimate.uhd.json"
    leftover.write_text("{}\n", encoding="utf-8")
    assert gen_descriptors._check(tmp_path, folders) == [
        f"{Path(OP) / leftover.name}: names no object in the manifest"
    ]
    gen_descriptors._write(tmp_path, folders)
    assert not leftover.exists()


def test_a_metric_less_model_binds_from_its_arch_folder(tmp_path: Path) -> None:
    _install(tmp_path, "sort_kernel_catalog", "gfx942", "ranker", metric=None)
    _install(
        tmp_path, "predict_applicable_kernels", "gfx950", "applicable", metric=None
    )
    ued = _ued(tmp_path)
    native = gen_descriptors.OpDescriptors(OP, FIELDS).uhd_id
    assert ued["sort_kernel_catalog"] == {"default": native, "gfx942": ["ranker"]}
    # The candidate generators (future JIT) are a list like every role.
    assert ued["predict_applicable_kernels"] == {"gfx950": ["applicable"]}


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"trained_against": {"ued": {"id": "another-engine"}}}, "trained against UED"),
        ({"score": {"metric": "time"}}, "filed under metric 'tflops'"),
        ({"score": {}}, "filed under metric 'tflops'"),
    ],
)
def test_a_misfiled_model_is_refused(
    tmp_path: Path, overrides: dict, match: str
) -> None:
    _install(tmp_path, "predict_engine", "gfx950", "model", **overrides)
    with pytest.raises(gen_descriptors.HkpPackError, match=match):
        gen_descriptors.installed_models(tmp_path, UED_ID)


def test_a_model_under_a_moved_engine_id_says_the_fields_moved_it(
    tmp_path: Path,
) -> None:
    """A field change moves the UED id; the model left under the old one names the cause."""
    old_id = gen_descriptors.ued_id(OP, {"dtype": "string"})
    path = _install(tmp_path, "predict_engine", "gfx950", "model")
    moved = (
        tmp_path
        / "heuristics"
        / old_id
        / path.relative_to(tmp_path / "heuristics" / UED_ID)
    )
    moved.parent.mkdir(parents=True)
    path.rename(moved)
    with pytest.raises(
        gen_descriptors.HkpPackError,
        match=f"filed under engine id {old_id}.*field change",
    ):
        gen_descriptors.installed_models(tmp_path, UED_ID)


def test_a_metric_model_outside_its_metric_folder_is_refused(tmp_path: Path) -> None:
    path = _install(tmp_path, "predict_engine", "gfx950", "model", metric=None)
    doc = json.loads(path.read_text(encoding="utf-8")) | {"score": {"metric": "tflops"}}
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(gen_descriptors.HkpPackError, match="filed under metric None"):
        gen_descriptors.installed_models(tmp_path, UED_ID)


def test_two_models_for_one_metric_on_one_key_are_refused(tmp_path: Path) -> None:
    _install(tmp_path, "predict_engine", "gfx950", "first")
    _install(tmp_path, "predict_engine", "gfx950", "second")
    with pytest.raises(gen_descriptors.HkpPackError, match="second tflops model"):
        gen_descriptors.installed_models(tmp_path, UED_ID)


@pytest.mark.parametrize(
    "changed",
    [
        {"dtype": "string", "N": "int", "has_dropout": "int"},  # a field added
        {"dtype": "string"},  # a field removed
        {"dtype": "string", "N": "string"},  # a field retyped
        {"N": "int", "dtype": "string"},  # the order its instances declare them in
    ],
)
def test_the_engine_id_moves_exactly_when_its_kernel_fields_do(changed: dict) -> None:
    # A model is trained against one field list, and the UHD pipeline files measurements by
    # engine id: the same fields keep the id (and every model trained against it), other
    # fields get a new id and so new datasets.
    assert gen_descriptors.ued_id(OP, dict(FIELDS)) == UED_ID
    assert gen_descriptors.ued_id(OP, changed) != UED_ID
    assert gen_descriptors.OpDescriptors(OP, FIELDS).ued_id == UED_ID


def test_a_new_field_default_moves_the_engine_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A default is part of the KMD's field record, so giving a field one is a field change.
    monkeypatch.setitem(gen_descriptors.KMD_DEFAULTS, OP, {"N": 0})
    assert gen_descriptors.ued_id(OP, FIELDS) != UED_ID


@pytest.mark.parametrize("slot_op", gen_descriptors.SPLIT_SLOT_OPS)
def test_the_split_engine_id_moves_when_any_slots_fields_do(slot_op: str) -> None:
    # The split engine's slots share one UED: its id covers every slot's fields, so a change
    # in any one slot moves it, and a slot op cannot mint an id of its own.
    slots = {
        op: {"dtype": "string", "block_m": "int"}
        for op in gen_descriptors.SPLIT_SLOT_OPS
    }
    split_id = gen_descriptors.split_ued_id(slots)
    assert (
        gen_descriptors.split_ued_id({op: dict(f) for op, f in slots.items()})
        == split_id
    )
    changed = {
        **slots,
        slot_op: {"dtype": "string", "block_m": "int", "block_n": "int"},
    }
    assert gen_descriptors.split_ued_id(changed) != split_id
    with pytest.raises(ValueError, match="split engine's UED id"):
        gen_descriptors.OpDescriptors(slot_op, slots[slot_op])


def test_two_metric_less_models_on_one_key_are_refused(tmp_path: Path) -> None:
    _install(tmp_path, "predict_applicable_kernels", "gfx950", "first", metric=None)
    _install(tmp_path, "predict_applicable_kernels", "gfx950", "second", metric=None)
    with pytest.raises(
        gen_descriptors.HkpPackError, match=r"second \(metric-less\) model"
    ):
        gen_descriptors.installed_models(tmp_path, UED_ID)


def _promote():
    """uhd_gen's promote module, so a binding is checked against promote rather than a copy
    of its rules. Skips without uhd_gen's requirements, as uhd_gen's suites do, unless the
    build requires them: uhd_gen's own `prerequisites_required()` reads that setting."""
    sys.path.insert(0, str(FLYDSL.parents[2] / "projects" / "hipdnn" / "tools"))
    from uhd_gen.conftest import prerequisites_required

    if prerequisites_required():
        from uhd_gen import promote

        return promote
    return pytest.importorskip("uhd_gen.promote")


@pytest.mark.parametrize(
    ("role", "arch", "metric"),
    [
        ("predict_engine", "gfx950", "tflops"),  # a new role and key
        ("predict_engine", "default", "tflops"),  # a new role, bound under `default`
        ("sort_kernel_catalog", "gfx942", "tflops"),  # a new key
        (
            "sort_kernel_catalog",
            "default",
            "tflops",
        ),  # joins the metric-less native ranker
        ("sort_kernel_catalog", "default", None),  # replaces the native ranker
        ("predict_applicable_kernels", "gfx950", None),  # the candidate generators
    ],
)
def test_a_binding_is_what_promote_writes(
    tmp_path: Path, role: str, arch: str, metric: str | None
) -> None:
    promote = _promote()
    before = _ued(tmp_path)
    installed_ids = _generated_uhds(tmp_path)
    model = str(uuid.uuid4())
    installed_ids[model] = _install(tmp_path, role, arch, model, metric=metric)
    holder = copy.deepcopy(before)
    written, _ = promote._rebind(holder, role, arch, model, metric, installed_ids)
    assert _ued(tmp_path)[role][arch] == written


def _generated_uhds(op_dir: Path) -> dict[str, Path]:
    """The generator's own UHDs written out: promote reads the metric of every UHD a key
    already binds from its installed file."""
    installed_ids = {}
    for name, doc in gen_descriptors.OpDescriptors(OP, FIELDS).documents().items():
        if name.endswith(".uhd.json"):
            path = op_dir / name
            path.write_text(json.dumps(doc), encoding="utf-8")
            installed_ids[doc["id"]] = path
    return installed_ids


@pytest.mark.parametrize(
    ("role", "arch", "first", "second"),
    [
        ("sort_kernel_catalog", "gfx942", "time", "tflops"),
        ("predict_engine", "gfx950", "time", "tflops"),
        ("predict_applicable_kernels", "gfx950", "tflops", None),
    ],
)
def test_two_metrics_on_one_key_bind_as_promote_writes(
    tmp_path: Path, role: str, arch: str, first: str, second: str | None
) -> None:
    """A key gains a second metric: the generator lists both, in the order promote writes
    (metric-less first, then by metric name) whichever was promoted first."""
    promote = _promote()
    installed_ids = _generated_uhds(tmp_path)
    earlier = str(uuid.uuid4())
    installed_ids[earlier] = _install(tmp_path, role, arch, earlier, metric=first)
    holder = copy.deepcopy(_ued(tmp_path))
    later = str(uuid.uuid4())
    installed_ids[later] = _install(tmp_path, role, arch, later, metric=second)
    written, replaced = promote._rebind(
        holder, role, arch, later, second, installed_ids
    )
    assert replaced == []
    assert _ued(tmp_path)[role][arch] == written == [later, earlier]


SHIPPED = FLYDSL.parent / "src/engines/kernel_ingestor_engine/descriptors/FlyDSL"
# The reads that turn a name a graph may leave unbound into a value: the first argument of
# each is read guarded.
GUARDS = ("value_or_default", "present")
# Per family: the folder filter, the names read by every model of the family and set by only
# some of its graphs (unguarded, a graph without them makes the estimate unavailable), and
# the pattern of the names a backward model reads bare on purpose. The backward graph has no
# ``generate_stats`` attribute (it takes the forward pass's stats as an input), so it is not
# listed for it. A backward model reads its required tensors bare as plain features and also
# inside guarded derived features, so a guard there says nothing about whether the name can
# be unbound. ``flops`` is read bare too: a graph whose work is not counted has no tflops to
# estimate, so the estimate is meant to be unavailable for it.
SDPA_FAMILIES = {
    "forward": (
        lambda folder: not folder.startswith("sdpa_bwd"),
        ("$graph.nodes[0].attn_scale_value", "$graph.nodes[0].generate_stats"),
        None,
    ),
    "backward": (
        lambda folder: folder.startswith("sdpa_bwd"),
        ("$graph.nodes[0].attn_scale_value",),
        re.compile(
            r"\$graph\.(flops|nodes\[0\]\.(flops|(q|k|v|o|do|dq|dk|dv|stats)\..+))"
        ),
    ),
}


def _reads(expression, guarded: bool, out: list[tuple[str, bool]]) -> None:
    """Every ``$name`` an expression reads, with whether a guard wraps the read."""
    if isinstance(expression, str):
        if expression.startswith("$"):
            out.append((expression, guarded))
    elif isinstance(expression, list):
        for item in expression:
            _reads(item, guarded, out)
    elif isinstance(expression, dict):
        for op, args in expression.items():
            if op in GUARDS and isinstance(args, list) and args:
                _reads(args[0], True, out)
                _reads(args[1:], guarded, out)
            else:
                _reads(args, guarded, out)


def _shipped_sdpa_reads(in_family) -> dict[str, list[tuple[str, bool]]]:
    """The reads of every installed SDPA model whose folder ``in_family`` accepts, keyed by
    its path in the tree."""
    reads = {}
    for path in sorted(SHIPPED.glob("sdpa_*/heuristics/**/heuristic.uhd.json")):
        if not in_family(path.relative_to(SHIPPED).parts[0]):
            continue
        out: list[tuple[str, bool]] = []
        _reads(
            json.loads(path.read_text(encoding="utf-8"))["features_signature"],
            False,
            out,
        )
        reads[path.relative_to(SHIPPED).as_posix()] = out
    return reads


@pytest.mark.parametrize("family", SDPA_FAMILIES)
def test_every_shipped_sdpa_model_guards_what_any_sdpa_model_guards(
    family: str,
) -> None:
    """A name some graphs of an SDPA family leave unbound must be read through a guard by
    every model of that family.

    One model's guard shows the name can be unbound on a graph of the family; a sibling
    reading it bare reports UNAVAILABLE for every graph that leaves it unbound, which no
    other check sees (the generator, ``--check`` and the census all pass). The forward
    models (every ``sdpa_*`` folder but ``sdpa_bwd_*``) and the backward models (the split
    estimates, the merged estimate and the merged ranker under ``sdpa_bwd_*``) are held
    separately: the two families guard different sets of names, and a backward model reads
    its required tensors (q, k, v, o, do, dq, dk, dv, stats) bare, as every backward graph
    binds them, and ``flops`` bare, because a graph whose work is not counted has no tflops
    to estimate; those are not held to the rule. An engine with a single model is held to
    its siblings.
    """
    in_family, always_guarded, always_bound = SDPA_FAMILIES[family]
    reads = _shipped_sdpa_reads(in_family)
    assert reads, f"no installed SDPA {family} model under {SHIPPED}"
    guarded = {name for model in reads.values() for name, wrapped in model if wrapped}
    guarded.update(always_guarded)
    if always_bound:
        guarded = {name for name in guarded if not always_bound.fullmatch(name)}
    bare = sorted(
        f"{path}: {name}"
        for path, model in reads.items()
        for name, wrapped in model
        if name in guarded and not wrapped
    )
    assert not bare, "read without value_or_default:\n" + "\n".join(bare)
