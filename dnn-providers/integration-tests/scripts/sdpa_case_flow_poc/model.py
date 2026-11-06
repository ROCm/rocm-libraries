# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Load the maintained model (model.yaml, rules.yaml, injections.yaml) and extra rule files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

MODEL_FILES = ("model.yaml", "rules.yaml", "injections.yaml")


class ModelError(Exception):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_yaml(path: Path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Rule:
    id: str
    kind: str
    form: str
    # Each conjunction maps an axis to the forbidden values of that axis.
    conjs: tuple
    source_file: str


@dataclass
class Model:
    axes: list
    values: dict
    parent: dict
    children: dict
    base_cases: dict
    rules: list
    non_exclusion_rules: list
    hints: dict
    input_files: dict = field(default_factory=dict)
    injection_counts: dict = field(default_factory=dict)
    # axis -> {value: tuple of harness needs}; a value that is not listed needs nothing.
    harness_needs: dict = field(default_factory=dict)
    # axis -> {value: {axis: frozenset}}: the need holds only in a row that matches the condition.
    harness_need_when: dict = field(default_factory=dict)
    # need id -> conjunctions ({axis: frozenset}) of the rows that need the capability.
    harness_conditions: dict = field(default_factory=dict)
    # (when, values): in a row that matches `when` ({axis: frozenset}), `values` ({axis: value})
    # count as no change from the shared base.
    plain_when: list = field(default_factory=list)

    def is_active(self, row: dict, axis: str) -> bool:
        if axis not in self.parent:
            return True
        pax, pvals = self.parent[axis]
        return self.is_active(row, pax) and row.get(pax) in pvals

    def violated_rules(self, row: dict) -> list:
        """Return the ids of the exclusion rules that the full row breaks."""
        out = []
        for r in self.rules:
            for c in r.conjs:
                if all(
                    row.get(a) is not None and row.get(a) in vs for a, vs in c.items()
                ):
                    out.append(r.id)
                    break
        return out

    def structure_errors(self, row: dict) -> list:
        errs = []
        for a in self.axes:
            v = row.get(a)
            active = self.is_active(row, a)
            if active and v is None:
                errs.append(f"{a}: active but empty")
            elif not active and v is not None:
                errs.append(f"{a}: inactive but set to {v}")
            elif v is not None and v not in self.values[a]:
                errs.append(f"{a}: unknown value {v}")
        return errs


def _check_axis_values(axes_values: dict, where: str, mapping: dict) -> None:
    for ax, vals in mapping.items():
        if ax not in axes_values:
            raise ModelError(f"{where}: unknown axis {ax}")
        for v in vals if isinstance(vals, (list, tuple, set, frozenset)) else [vals]:
            if v not in axes_values[ax]:
                raise ModelError(f"{where}: unknown value {ax}={v}")


def _parse_rules(doc: dict, source: str, values: dict):
    exclusions, others = [], []
    seen = set()
    for r in doc.get("rules") or []:
        rid = r["id"]
        if rid in seen:
            raise ModelError(f"{source}: duplicate rule id {rid}")
        seen.add(rid)
        for key in ("kind", "reason"):
            if key not in r:
                raise ModelError(f"{source}: rule {rid} has no {key}")
        raw = []
        if "forbid" in r:
            raw.append(r["forbid"])
        raw.extend(r.get("forbid_any") or [])
        if not raw:
            others.append({"id": rid, "kind": r["kind"], "form": r.get("form", "")})
            continue
        conjs = []
        for c in raw:
            _check_axis_values(values, f"{source}: rule {rid}", c)
            conjs.append({ax: frozenset(vs) for ax, vs in c.items()})
        exclusions.append(
            Rule(rid, r["kind"], r.get("form", "exclusion"), tuple(conjs), source)
        )
    return exclusions, others


def load_model(model_dir: Path, extra_rule_files=()) -> Model:
    model_dir = Path(model_dir)
    m = load_yaml(model_dir / "model.yaml")
    axes, values, parent, children = [], {}, {}, {}
    for a in m["axes"]:
        aid = a["id"]
        if aid in values:
            raise ModelError(f"duplicate axis {aid}")
        ids = [v["id"] for v in a["values"]]
        if len(set(ids)) != len(ids):
            raise ModelError(f"duplicate value in axis {aid}")
        axes.append(aid)
        values[aid] = ids
        children[aid] = []
    for a in m["axes"]:
        if a["kind"] == "child":
            ((pax, pvals),) = a["parent"].items()
            _check_axis_values(values, f"axis {a['id']} parent", {pax: pvals})
            parent[a["id"]] = (pax, frozenset(pvals))
            children[pax].append(a["id"])

    bases = {}
    for name, b in (m.get("base_cases") or {}).items():
        vals = b["values"]
        _check_axis_values(
            values, f"base case {name}", {k: [v] for k, v in vals.items()}
        )
        bases[name] = dict(vals)

    rules, others = [], []
    rule_docs = [("rules.yaml", model_dir / "rules.yaml")]
    rule_docs += [(str(Path(p).resolve()), Path(p)) for p in extra_rule_files]
    ids = set()
    for label, path in rule_docs:
        ex, ot = _parse_rules(load_yaml(path), label, values)
        for r in ex:
            if r.id in ids:
                raise ModelError(f"rule id {r.id} appears in two rule files")
            ids.add(r.id)
        rules.extend(ex)
        others.extend(ot)

    inj = load_yaml(model_dir / "injections.yaml")
    inj_counts = _check_injections(inj, values, bases)
    hints = m.get("generator_hints") or {}
    plain_when = _parse_plain_when(hints, values, parent)
    needs, need_when = _parse_harness_needs(m["axes"], hints, values)
    conditions = _parse_harness_conditions(hints, values, needs)
    plain_targets = {a for _, vals in plain_when for a in vals}
    for a, per in need_when.items():
        for v, when in per.items():
            if set(when) & plain_targets:
                raise ModelError(
                    f"axis {a}: the `when` of the harness need of {v} names a plain_when target"
                )

    model = Model(
        axes=axes,
        values=values,
        parent=parent,
        children=children,
        base_cases={},
        rules=rules,
        non_exclusion_rules=others,
        hints=hints,
        injection_counts=inj_counts,
        harness_needs=needs,
        harness_need_when=need_when,
        harness_conditions=conditions,
        plain_when=plain_when,
    )
    for name, vals in bases.items():
        row = {a: vals.get(a) for a in axes}
        errs = model.structure_errors(row)
        if errs:
            raise ModelError(f"base case {name}: {errs}")
        broken = model.violated_rules(row)
        if broken:
            raise ModelError(f"base case {name} breaks rules {broken}")
        model.base_cases[name] = row
    model.input_files = {f"model/{n}": sha256_file(model_dir / n) for n in MODEL_FILES}
    for label, path in rule_docs[1:]:
        model.input_files[label] = sha256_file(path)
    return model


def _parse_harness_needs(axes: list, hints: dict, values: dict):
    """({axis: {value: needs}}, {axis: {value: when}}) from each axis's `harness_needs`.

    A value maps to a list of needs, or to {needs: [...], when: {axis: [values]}} for a need that
    holds only in a row that matches `when`. Needs come from generator_hints.harness_needs.
    """
    known = {n["id"] for n in hints.get("harness_needs") or []}
    out, cond = {}, {}
    for a in axes:
        entry = a.get("harness_needs")
        if entry is None:
            continue
        per, per_when = {}, {}
        for v, spec in entry.items():
            where = f"axis {a['id']}: harness_needs of {v}"
            if v not in values[a["id"]]:
                raise ModelError(
                    f"axis {a['id']}: harness_needs names unknown value {v}"
                )
            needs = spec
            if isinstance(spec, dict):
                if set(spec) != {"needs", "when"} or not spec["when"]:
                    raise ModelError(
                        f"{where}: a mapping needs exactly `needs` and `when`"
                    )
                if any(not isinstance(vs, list) for vs in spec["when"].values()):
                    raise ModelError(
                        f"{where}: each `when` axis needs a list of values"
                    )
                _check_axis_values(values, where, spec["when"])
                per_when[v] = {x: frozenset(vs) for x, vs in spec["when"].items()}
                needs = spec["needs"]
            if not needs or any(n not in known for n in needs):
                raise ModelError(
                    f"{where} must list needs from generator_hints.harness_needs, got {needs}"
                )
            per[v] = tuple(needs)
        out[a["id"]] = per
        if per_when:
            cond[a["id"]] = per_when
    return out, cond


def _parse_harness_conditions(hints: dict, values: dict, needs: dict) -> dict:
    """{need: conjunctions} from generator_hints.harness_needs `forbid_any`. Every value labelled with
    a need must appear in one of that need's conjunctions."""
    out = {}
    for n in hints.get("harness_needs") or []:
        where = f"generator_hints.harness_needs {n.get('id')}"
        if (
            set(n) != {"id", "select_reason", "why", "forbid_any"}
            or not n["forbid_any"]
        ):
            raise ModelError(
                f"{where}: needs exactly `id`, `select_reason`, `why` and `forbid_any`"
            )
        conjs = []
        for c in n["forbid_any"]:
            if not c or any(not isinstance(vs, list) or not vs for vs in c.values()):
                raise ModelError(
                    f"{where}: each forbid_any entry maps axes to value lists"
                )
            _check_axis_values(values, where, c)
            conjs.append({a: frozenset(vs) for a, vs in c.items()})
        out[n["id"]] = tuple(conjs)
    for a, per in needs.items():
        for v, ns in per.items():
            for n in ns:
                if not any(v in c.get(a, ()) for c in out[n]):
                    raise ModelError(
                        f"axis {a}: value {v} needs {n}, but {n}'s forbid_any never names it"
                    )
    return out


def _parse_plain_when(hints: dict, values: dict, parent: dict) -> list:
    """A target axis belongs to one entry only and is no entry's `when` axis, so each target's
    count depends on `when` axes whose own count does not change."""
    out = []
    entries = hints.get("plain_when") or []
    when_axes = {a for e in entries for a in (e.get("when") or {})}
    targets = set()
    for e in entries:
        where = f"generator_hints.plain_when {e.get('id')}"
        if (
            set(e) != {"id", "when", "values", "why"}
            or not e["when"]
            or not e["values"]
        ):
            raise ModelError(f"{where}: needs exactly `id`, `when`, `values` and `why`")
        if any(not isinstance(vs, list) for vs in e["when"].values()):
            raise ModelError(f"{where}: each `when` axis needs a list of values")
        for a in e["values"]:
            if a in when_axes or a in targets:
                raise ModelError(
                    f"{where}: {a} is a `when` axis or a target of another entry"
                )
            targets.add(a)
        _check_axis_values(values, where, e["when"])
        _check_axis_values(values, where, {k: [v] for k, v in e["values"].items()})
        for a in e["values"]:
            if a in parent:
                raise ModelError(f"{where}: {a} must be a top-level axis")
        out.append(
            (
                {a: frozenset(vs) for a, vs in e["when"].items()},
                dict(e["values"]),
            )
        )
    return out


def _check_injections(doc: dict, values: dict, bases: dict) -> dict:
    counts = {}
    for section in (
        "rejection_probes",
        "path_variants",
        "frontend_rejection_controls",
        "conditional_probes",
    ):
        n = 0
        for entry in doc.get(section) or []:
            items = list(entry.get("probes") or entry.get("variants") or [entry])
            if "equivalent_cpp_request" in entry:
                items.append(entry["equivalent_cpp_request"])
            for p in items:
                if not isinstance(p, dict):
                    continue
                start = p.get("start_from", entry.get("start_from"))
                if start is not None and start not in bases:
                    raise ModelError(f"injections {section}: unknown base {start}")
                changes = p.get("changes") or {}
                _check_axis_values(
                    values,
                    f"injections {section} {p.get('id', entry.get('id'))}",
                    {k: [v] for k, v in changes.items()},
                )
                n += 1
        counts[section] = n
    return counts
