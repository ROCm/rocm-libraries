"""Rewrite an enumerated variant-set config into the compact ``variants`` form.

An enumerated set runs to tens of thousands of lines while being the one file
worth reviewing in a descriptor PR. `variants` states the same thing as a
shape list, named knob sets and a name template, which
`codegen/config_loader.py` expands at load time.

For sets already generated and shipped; new sets come out of
`dispatch_parity.py` compact already. The round trip is not optional: this
tool re-expands what it wrote and refuses anything that does not reproduce the
input kernel-for-kernel, key-for-key, in order.

Tri-state knobs: the spec decides the compiled binary (absent = the kernel's
policy decides at build time), the metadata is what the matcher compares, and
the KMD `default_value` fills anything absent at load. "Absent from spec" and
"explicitly false" differ while both reach metadata as 0, so the shape records
the policy's answer under `resolved`.
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from collections import OrderedDict
from pathlib import Path

#: Prefix marking a name-template slot bound to a metadata mirror rather than a
#: spec field. Binding a slot to a field that merely happens to match on this
#: input is how a template silently renders the wrong thing on the next one.
_NAME_BINDABLE_SUFFIX = "md_"


class FactoriseError(RuntimeError):
    """The input could not be factorised without changing what it generates."""


def _load(path: Path) -> dict:
    import yaml

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        return yaml.safe_load(handle)


def _flatten(config: dict) -> list:
    """Every kernel with its pack defaults already merged in, in the loader's
    own merge order so the effective spec's key order is the descriptor's. A
    knob the defaults do not mention stays absent, preserving the tri-state."""
    pack = config["packs"][0]
    # `axes`/`kernel_template` expand to kernels this tool never sees, so it would
    # either crash on a missing `kernels` key or silently drop the expansion.
    unhandled = sorted({"axes", "kernel_template"} & set(pack))
    if unhandled:
        raise FactoriseError(
            f"pack '{pack.get('name', '<unnamed>')}' declares {unhandled}, which "
            f"this tool does not carry into the compact form -- those kernels would "
            f"be silently dropped. Factorise a pack whose kernels are enumerated."
        )
    if "kernels" not in pack:
        raise FactoriseError(
            f"pack '{pack.get('name', '<unnamed>')}' declares no 'kernels' list; "
            f"there is nothing to factorise."
        )
    defaults = pack.get("kernel_defaults") or {}
    default_spec = defaults.get("spec") or {}
    default_source = {k: v for k, v in defaults.items() if k != "spec"}
    flat = []
    for kernel in pack["kernels"]:
        source = kernel["kernel_source"]
        spec = {**default_spec, **(source.get("spec") or {})}
        flat.append(
            {
                "name": kernel["name"],
                "spec": spec,
                "effective_spec": spec,
                "source": {
                    **default_source,
                    **{k: v for k, v in source.items() if k != "spec"},
                },
                "metadata": dict(kernel.get("metadata") or {}),
                "priority": kernel.get("priority"),
            }
        )
    return flat


def _constant_keys(entries: list, getter) -> dict:
    """Keys every entry carries with one identical value."""
    if not entries:
        return {}
    first = getter(entries[0])
    out = {}
    for key, value in first.items():
        if all(key in getter(e) and getter(e)[key] == value for e in entries):
            out[key] = value
    return out


def _shape_blocks(entries: list, knob_fields: list) -> list:
    """Consecutive entries agreeing on every NON-knob spec value -- one shape's arms.

    Consecutive, not globally grouped: the emitted kernel order IS descriptor order,
    and reordering the set changes every id assignment downstream.
    """
    out = []
    for entry in entries:
        key = tuple(
            sorted(
                (k, repr(v)) for k, v in entry["spec"].items() if k not in knob_fields
            )
        )
        if out and out[-1][0] == key:
            out[-1][1].append(entry)
        else:
            out.append((key, [entry]))
    return [block for _, block in out]


def _merge_key_order(orders: list):
    """One key order every input order is a subsequence of, or None. Spec key
    order reaches the emitted descriptor, so two shape-blocks share a group
    only when one `spec_order` renders both."""
    merged: list = []
    for order in orders:
        cursor = 0
        for key in order:
            if key in merged:
                position = merged.index(key)
                if position < cursor:
                    return None
                cursor = position + 1
            else:
                merged.insert(cursor, key)
                cursor += 1
    for order in orders:
        positions = [merged.index(key) for key in order]
        if positions != sorted(positions):
            return None
    return merged


def _name_value(entry: dict, field: str):
    """What `field` renders as inside a name."""
    if field.startswith(_NAME_BINDABLE_SUFFIX):
        value = entry["metadata"].get(field[len(_NAME_BINDABLE_SUFFIX) :])
    else:
        value = entry["effective_spec"].get(field)
    return int(value) if isinstance(value, bool) else value


def _is_abbreviation(prefix: str, bare: str) -> bool:
    """Does `prefix` read as a short form of the field name `bare`? Shipped
    grammars abbreviate by initials, sometimes skipping a word (`hq` <-
    num_query_heads), so a subsequence of the word initials, or a prefix of the
    name or of one word, is accepted."""
    if not prefix:
        return False
    words = [w for w in bare.split("_") if w]
    if bare.startswith(prefix) or any(w.startswith(prefix) for w in words):
        return True
    initials = "".join(w[0] for w in words)
    cursor = 0
    for letter in prefix:
        cursor = initials.find(letter, cursor) + 1
        if cursor == 0:
            return False
    return True


def _binding_rank(prefix: str, field: str) -> tuple:
    """Prefer the field a reader would expect this token to mean: rank by
    `_is_abbreviation` first, then a spec field over its `md_` metadata mirror,
    since the spec decides the binary. Several fields can bind a column
    equally well and render the same names."""
    bare = (
        field[len(_NAME_BINDABLE_SUFFIX) :]
        if field.startswith(_NAME_BINDABLE_SUFFIX)
        else field
    )
    return (
        0 if _is_abbreviation(prefix, bare) else 1,
        1 if field.startswith(_NAME_BINDABLE_SUFFIX) else 0,
        field,
    )


def _bind_column(column: list, group: list, fields: list):
    """`{field}` or `prefix{field}` if one field explains this column, else None.

    A field binds only when its value matches the token for every entry; where
    several survive, `_binding_rank` picks the one the token is named after.
    """
    prefix_match = re.fullmatch(r"([a-zA-Z]+)(\d+)", column[0])
    prefix = prefix_match.group(1) if prefix_match else None
    bare_candidates = []
    prefixed_candidates = []
    for field in fields:
        values = [_name_value(entry, field) for entry in group]
        if any(value is None for value in values):
            continue
        if all(tok == str(val) for tok, val in zip(column, values)):
            bare_candidates.append(field)
        elif prefix and all(
            tok == f"{prefix}{val}" for tok, val in zip(column, values)
        ):
            prefixed_candidates.append(field)
    if bare_candidates:
        best = min(bare_candidates, key=lambda f: _binding_rank("", f))
        return "{" + best + "}"
    if prefixed_candidates:
        best = min(prefixed_candidates, key=lambda f: _binding_rank(prefix, f))
        return prefix + "{" + best + "}"
    return None


def _bind_ordinal(column: list):
    """A zero-padded, per-entry-unique serial -- `full_00020_`. At most one."""
    if not all(re.fullmatch(r"\d+", token) for token in column):
        return None
    if len(set(column)) != len(column):
        return None
    widths = {len(token) for token in column}
    if len(widths) != 1:
        return None
    return [int(token) for token in column], f"{{ordinal:0{widths.pop()}d}}"


def _infer_template(group: list, fields: list):
    """A name template rendering every name in `group`, its ordinals and tags.
    Tokens bind from both ends and whatever refuses to bind in the middle
    becomes a per-arm ``tag``, since shipped grammars carry optional tokens
    (`kpad8`, `persist304`). ``None`` means no template renders the group."""
    rows = [entry["name"].split("_") for entry in group]
    shortest = min(len(row) for row in rows)

    head = []
    ordinals = None
    start = 0
    while start < shortest:
        column = [row[start] for row in rows]
        if len(set(column)) == 1:
            head.append(column[0])
        elif (binding := _bind_column(column, group, fields)) is not None:
            head.append(binding)
        elif ordinals is None and (found := _bind_ordinal(column)) is not None:
            ordinals, placeholder = found
            head.append(placeholder)
        else:
            break
        start += 1

    tail = []
    end = 0
    while end < shortest - start:
        column = [row[len(row) - 1 - end] for row in rows]
        if len(set(column)) == 1:
            tail.append(column[0])
        elif (binding := _bind_column(column, group, fields)) is not None:
            tail.append(binding)
        else:
            break
        end += 1
    tail.reverse()

    tags = ["_".join(row[start : len(row) - end]) for row in rows]
    parts = head + (["{tag}"] if any(tags) else []) + tail
    return "_".join(parts), ordinals, tags


def _bound_slots(template: str) -> int:
    """How many name tokens the template actually renders from a field."""
    return template.count("{") - (1 if "{tag}" in template else 0)


def _group(entries: list, knob_fields: list, fields: list) -> list:
    """Consecutive shape-blocks that one template renders, extended only while
    the wider template still binds at least as many tokens; otherwise the merge
    collapses the set into one group whose `tag` carries the entire name."""
    groups = []
    current = None
    current_template = None

    def attempt(candidate):
        if _merge_key_order([tuple(e["spec"]) for e in candidate]) is None:
            return None
        if len({tuple(e["metadata"]) for e in candidate}) != 1:
            return None
        return _infer_template(candidate, fields)

    for block in _shape_blocks(entries, knob_fields):
        if current is None:
            current = list(block)
            current_template = attempt(current)
            continue
        wider = attempt(current + block)
        # `max(1, ...)` is the floor: otherwise a first block whose names no
        # field explains seeds the run at zero bound slots, `0 >= 0` lets every
        # later block merge, and one odd block costs the whole file.
        if wider and _bound_slots(wider[0]) >= max(
            1, _bound_slots(current_template[0])
        ):
            current = current + block
            current_template = wider
        else:
            groups.append(current)
            current = list(block)
            current_template = attempt(current)
    if current:
        groups.append(current)
    return groups


def _tag_template(tag: str, entry: dict) -> str:
    """A tag written in terms of the metadata field it mirrors, where it does.

    Shipped grammars spell the resolved tri-state value into the name (`_e1`)
    even for policy-decided arms. The substitution is anchored to the letters
    before the value, which must abbreviate the field as `_binding_rank`
    requires; an ambiguous tag stays literal.
    """
    matches = []
    for name, value in entry["metadata"].items():
        rendered = str(int(value) if isinstance(value, bool) else value)
        if not rendered:
            continue
        for found in re.finditer(rf"([a-z]*){re.escape(rendered)}(?![0-9])", tag):
            prefix = found.group(1)
            if not prefix:
                continue
            if _binding_rank(prefix, _NAME_BINDABLE_SUFFIX + name)[0] != 0:
                continue
            matches.append((found, name))
    if len(matches) != 1:
        return tag
    found, name = matches[0]
    replacement = "{" + _NAME_BINDABLE_SUFFIX + name + "}"
    return (
        tag[: found.start(0) + len(found.group(1))] + replacement + tag[found.end(0) :]
    )


def _build_group(group: list, template: str, ordinals, tags: list, knob_fields: list):
    """One group of same-template kernels -> one `variants` entry."""
    spec_order = _merge_key_order([tuple(e["spec"]) for e in group])
    position = {id(entry): index for index, entry in enumerate(group)}
    spec_defaults = {
        key: value
        for key, value in _constant_keys(group, lambda e: e["spec"]).items()
        if key not in knob_fields
    }

    knob_sets = OrderedDict()
    shapes = []
    for block in _shape_blocks(group, knob_fields):
        base = position[id(block[0])]
        # Metadata a shape's arms disagree on while the spec does not mention
        # it: a knob pinned for the matcher but absent from the dispatcher's
        # spec, i.e. same binary, two catalog entries. The shape cannot carry
        # it, so the arm states it.
        per_arm_metadata = {
            field
            for field in group[0]["metadata"]
            if all(field not in entry["spec"] for entry in block)
            and len({entry["metadata"].get(field) for entry in block}) > 1
        }
        arms = []
        for entry in block:
            index = position[id(entry)]
            arm = {f: entry["spec"][f] for f in knob_fields if f in entry["spec"]}
            if per_arm_metadata:
                arm["metadata"] = {
                    field: entry["metadata"][field] for field in per_arm_metadata
                }
            if tags[index]:
                arm["tag"] = _tag_template(tags[index], entry)
            if ordinals is not None:
                arm["ordinal_offset"] = ordinals[index] - ordinals[base]
            arms.append(arm)
        signature = repr(arms)
        if signature not in knob_sets:
            knob_sets[signature] = (f"knobs{len(knob_sets)}", arms)

        shape = {
            key: value
            for key, value in block[0]["spec"].items()
            if key not in knob_fields and key not in spec_defaults
        }
        shape["knobs"] = knob_sets[signature][0]
        # What the kernel's policy resolved to for fields this shape's arms do
        # not pin; otherwise the loader uses the KMD default as the catalog key
        # while the binary was built from the policy's answer. "Does not pin"
        # has two spellings -- key omitted, or present as None -- and testing
        # only for absence drops `resolved` for every builder-dumped spec.
        resolved = {
            field: value
            for entry in block
            for field, value in entry["metadata"].items()
            if entry["spec"].get(field) is None and field not in per_arm_metadata
        }
        if resolved:
            shape["resolved"] = resolved
        if ordinals is not None:
            shape["ordinal"] = ordinals[base]
        shapes.append(shape)

    out = {
        "name": template,
        "metadata": list(group[0]["metadata"]),
        "spec_order": spec_order,
    }
    # Same two spellings as `resolved` above: omitted, or present-and-None.
    # These predicates must agree, since `policy_knobs` is what makes the
    # loader demand a `resolved` entry; a knob listed by one and not the other
    # either asks for a value nothing supplies or supplies one nothing needs.
    policy = sorted(
        f for f in knob_fields if any(e["spec"].get(f) is None for e in group)
    )
    if policy:
        out["policy_knobs"] = policy
    if spec_defaults:
        out["spec_defaults"] = spec_defaults
    out["knob_sets"] = {name: arms for name, arms in knob_sets.values()}
    out["shapes"] = shapes
    return out


def factorise(config: dict, knob_fields: list, vocabulary: dict) -> dict:
    """The compact equivalent of `config`. Round-trip is checked by the caller."""
    if len(config.get("packs") or []) != 1:
        raise FactoriseError(
            "this tool factorises a single-pack variant set; a multi-pack config "
            "would need per-pack grouping, which no shipped set uses."
        )
    entries = _flatten(config)
    if not entries:
        raise FactoriseError("the pack declares no kernels.")

    # Effective priority: an absent key and an explicit 0 both reach the
    # descriptor as 0, and a variants group emits no priority key, so it can
    # only stand for a set uniformly at that default.
    priorities = {e["priority"] or 0 for e in entries}
    if priorities != {0}:
        raise FactoriseError(
            f"kernels carry non-default 'priority' values {sorted(priorities)}; a "
            f"variants group emits no priority key, so the compact form would "
            f"silently reset them to 0."
        )

    name_fields = sorted(
        {f for e in entries for f in e["effective_spec"]}
        | {_NAME_BINDABLE_SUFFIX + f for e in entries for f in e["metadata"]}
    )
    default_source = _constant_keys(entries, lambda e: e["source"])

    groups = []
    for group in _group(entries, knob_fields, name_fields):
        template, ordinals, tags = _infer_template(group, name_fields)
        built = _build_group(group, template, ordinals, tags, knob_fields)
        if vocabulary:
            spellings = {k: v for k, v in vocabulary.items() if k in built["metadata"]}
            if spellings:
                built["vocabulary"] = spellings
        groups.append(built)

    pack = config["packs"][0]
    out = {k: v for k, v in config.items() if k != "packs"}
    compact_pack = {"name": pack["name"]}
    if pack.get("arch"):
        compact_pack["arch"] = list(pack["arch"])
    if pack.get("discriminator"):
        compact_pack["discriminator"] = pack["discriminator"]
    if default_source:
        compact_pack["kernel_defaults"] = default_source
    compact_pack["variants"] = groups
    out["packs"] = [compact_pack]
    return out


class _Row(dict):
    """A mapping YAML renders on one line.

    A shape is a record, not a document section: block style spends fifteen
    lines per shape restating key names.
    """


class _Line(list):
    """A sequence YAML renders on ONE line -- a key-name list, not a document."""


def _represent_row(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True)


def _represent_line(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def _render_group(group: dict) -> dict:
    """One group with its records collapsed to a line each."""
    out = {"name": group["name"], "metadata": _Line(group["metadata"])}
    if group.get("spec_order"):
        out["spec_order"] = _Line(group["spec_order"])
    if group.get("policy_knobs"):
        out["policy_knobs"] = _Line(group["policy_knobs"])
    if group.get("vocabulary"):
        out["vocabulary"] = group["vocabulary"]
    if group.get("spec_defaults"):
        out["spec_defaults"] = _Row(group["spec_defaults"])
    out["knob_sets"] = {
        name: [
            _Row(
                {
                    key: _Row(value) if isinstance(value, dict) else value
                    for key, value in arm.items()
                }
            )
            for arm in arms
        ]
        for name, arms in group["knob_sets"].items()
    }
    out["shapes"] = [
        _Row(
            {
                key: _Row(value) if isinstance(value, dict) else value
                for key, value in shape.items()
            }
        )
        for shape in group["shapes"]
    ]
    return out


def dump(compact: dict) -> str:
    """Render the compact config: shapes and arms one line each, the rest block."""
    import yaml

    class Dumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            # Two groups sharing a vocabulary dict would otherwise emit
            # `&id001`/`*id001`: an anchor saves four lines and costs the
            # reader a cross-reference.
            return True

    Dumper.add_representer(_Row, _represent_row)
    Dumper.add_representer(_Line, _represent_line)

    rendered = {**compact}
    rendered["packs"] = [
        {
            **pack,
            "variants": [_render_group(group) for group in pack["variants"]],
        }
        for pack in compact["packs"]
    ]
    # Wide on purpose: a shape is one record, and wrapping it mid-record puts
    # the reader back to scanning for where an entry ends.
    return yaml.dump(rendered, Dumper=Dumper, sort_keys=False, width=1000)


def _round_trip(original: dict, compact: dict) -> None:
    """Expand the compact form and refuse to emit unless it reproduces the
    input kernel-for-kernel, key-for-key, in order: ids are assigned by
    position and the dedup pass keys on resolved metadata. `specialization` is
    compared separately, since it reaches one UKD field rather than one per
    kernel."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from codegen.config_loader import load_config  # noqa: PLC0415

    import tempfile

    import yaml

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "compact.yaml"
        # The emitted text, not a re-dump of the same object: the flow-style
        # renderer is part of what ships.
        path.write_text(dump(compact))
        reloaded = load_config(path)
        before = Path(tmp) / "original.yaml"
        before.write_text(yaml.safe_dump(original, sort_keys=False))
        expected = load_config(before)

    if reloaded.specialization != expected.specialization:
        raise FactoriseError(
            f"round trip lost or altered the bundle's 'specialization' declaration: "
            f"{reloaded.specialization!r} != {expected.specialization!r}. Without it "
            f"the emitted descriptors carry no specialization_contract, and a "
            f"machine holding the archive has nothing to check the compiled bytes "
            f"against."
        )
    got = reloaded.packs[0].kernels
    want = expected.packs[0].kernels
    if len(got) != len(want):
        raise FactoriseError(
            f"round trip produced {len(got)} kernels, not {len(want)}."
        )
    for index, (a, b) in enumerate(zip(got, want)):
        # Every field that reaches a descriptor. The compact form carries only
        # name, spec and metadata, so a config that varies `arch` or a `hip`
        # kernel's `build` converts lossily; without these rows the check would
        # pass and ship it.
        for label, left, right in (
            ("name", a.name, b.name),
            ("metadata", a.metadata, b.metadata),
            ("metadata key order", list(a.metadata), list(b.metadata)),
            ("spec", a.kernel_source.spec, b.kernel_source.spec),
            ("spec key order", list(a.kernel_source.spec), list(b.kernel_source.spec)),
            ("kind", a.kernel_source.kind, b.kernel_source.kind),
            ("source", a.kernel_source.source, b.kernel_source.source),
            ("builder", a.kernel_source.builder, b.kernel_source.builder),
            ("source_file", a.kernel_source.source_file, b.kernel_source.source_file),
            ("entry_point", a.kernel_source.entry_point, b.kernel_source.entry_point),
            ("entry", a.kernel_source.entry, b.kernel_source.entry),
            ("build", a.kernel_source.build, b.kernel_source.build),
            ("arch", a.arch, b.arch),
            ("priority", a.priority, b.priority),
        ):
            if left != right:
                raise FactoriseError(
                    f"round trip differs at kernel {index} ({b.name}): {label} "
                    f"{left!r} != {right!r}. The compact form carries name, spec and "
                    f"metadata only, so a set that varies anything else per kernel "
                    f"cannot be written in it."
                )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite an enumerated variant-set config into `variants` form."
    )
    parser.add_argument(
        "--config", required=True, help="Enumerated config (.yaml[.gz])."
    )
    parser.add_argument("--out", required=True, help="Where to write the compact form.")
    parser.add_argument(
        "--knobs",
        required=True,
        help="Comma-separated spec fields that vary WITHIN one shape -- the tuning "
        "arms. Everything else identifies the shape. Required rather than inferred: "
        "which fields are knobs is a fact about the kernel, and guessing it from one "
        "input silently reclassifies a field the next set happens not to vary.",
    )
    parser.add_argument(
        "--vocabulary",
        default="",
        help="Metadata spellings the MATCHER compares, as "
        "'field:from=to,from=to'. The spec carries the builder's spelling ('bf16') "
        "and the matcher reads hipDNN's ('BF16'); a descriptor in the wrong one "
        "loads cleanly and matches nothing.",
    )
    args = parser.parse_args(argv)

    vocabulary = {}
    for clause in filter(None, args.vocabulary.split(";")):
        field, _, pairs = clause.partition(":")
        vocabulary[field.strip()] = dict(
            pair.split("=", 1) for pair in pairs.split(",") if pair
        )

    try:
        import yaml

        original = _load(Path(args.config))
        compact = factorise(
            original,
            [k.strip() for k in args.knobs.split(",") if k.strip()],
            vocabulary,
        )
        _round_trip(original, compact)
        text = dump(compact)
        Path(args.out).write_text(text)
    except FactoriseError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    before = len(yaml.safe_dump(original, sort_keys=False).splitlines())
    after = len(text.splitlines())
    kernels = len(original["packs"][0]["kernels"])
    print(
        f"wrote {args.out}: {after} lines for {kernels} kernels "
        f"(was {before}; {100 * (before - after) / before:.1f}% smaller), "
        f"round trip identical"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
