#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for gen_cache_key.py's schema walk and emitters.

Drives the Emitter with hand-built reflection schemas, so no flatc invocation and no
compilation is needed. The generated header's runtime behaviour is covered separately by
TestGraphContentKey.cpp; what is pinned here is the generator's own policy: which fields
participate, that the hash and the comparison always agree on that set, and that a union
is descended into rather than skipped.
"""

import unittest

from gen_cache_key import Emitter, accessor, short_name

NS = "test_ns"


def field(
    name,
    base_type,
    field_id,
    *,
    index=-1,
    element=None,
    ignored=False,
    deprecated=False,
    optional=False,
):
    entry = {
        "name": name,
        "id": field_id,
        "type": {"base_type": base_type, "index": index},
        "attributes": [],
    }
    if element is not None:
        entry["type"]["element"] = element
    if ignored:
        entry["attributes"].append({"key": "cache_ignore"})
    if deprecated:
        entry["deprecated"] = True
    if optional:
        entry["optional"] = True
    return entry


def table(name, fields):
    return {"name": name, "fields": fields}


def schema(objects, enums=None):
    return {"objects": objects, "enums": enums or []}


class TestFieldPolicy(unittest.TestCase):
    """keep()/fields_of(): which fields participate at all."""

    def setUp(self):
        self.emitter = Emitter(schema([table(f"{NS}.Root", [])]), f"{NS}.Root")

    def test_an_unannotated_field_participates(self):
        # Opt-out is the whole point: a field added tomorrow is covered by default.
        self.assertTrue(self.emitter.keep(field("shape", "Int", 0)))

    def test_a_cache_ignore_field_is_dropped(self):
        self.assertFalse(self.emitter.keep(field("id", "Int", 0, ignored=True)))

    def test_a_deprecated_field_is_dropped(self):
        self.assertFalse(self.emitter.keep(field("old", "Int", 0, deprecated=True)))

    def test_fields_are_restored_to_declaration_order(self):
        # The binary schema sorts alphabetically; the emitted stream must follow the
        # .fbs declaration order instead, or the header is unreadable against it.
        root = table(
            f"{NS}.Root",
            [
                field("zulu", "Int", 2),
                field("alpha", "Int", 0),
                field("mike", "Int", 1),
            ],
        )
        emitter = Emitter(schema([root]), f"{NS}.Root")
        self.assertEqual(
            [f["name"] for f in emitter.fields_of(root)], ["alpha", "mike", "zulu"]
        )


def function_bodies(text):
    """Map each emitted definition to its body text.

    The header interleaves a `hashAppend`/`logicallyEqual` pair per type, so the two
    cannot be separated by splitting the file once. Keys are `("hash"|"equal", type)`;
    forward declarations (which end in `;`) are skipped.
    """
    bodies, current, depth = {}, None, 0
    for line in text.splitlines():
        if current is None:
            if not line.startswith("inline ") or line.rstrip().endswith(";"):
                continue
            kind = "hash" if "hashAppend" in line else "equal"
            inside = line[line.index("(") + 1 : line.rindex(")")]
            names = [w.strip("*&,") for w in inside.split() if w[:1].isupper()]
            current, bodies[(kind, names[-1])] = (kind, names[-1]), []
            continue
        bodies[current].append(line)
        depth += line.count("{") - line.count("}")
        if depth == 0 and line.startswith("}"):
            current = None
    return {key: "\n".join(value) for key, value in bodies.items()}


class TestHashAndComparisonAgree(unittest.TestCase):
    """The load-bearing invariant: one traversal feeds both emitters.

    A field hashed but not compared is merely slow; a field compared but not hashed --
    or ignored by one side only -- is a wrong-kernel bug.
    """

    def emit_for(self, fields):
        root = table(f"{NS}.Root", fields)
        return Emitter(schema([root]), f"{NS}.Root").emit()

    def test_a_kept_field_appears_in_both_functions(self):
        bodies = function_bodies(self.emit_for([field("alpha", "Int", 0)]))
        self.assertIn("alpha", bodies[("hash", "Root")])
        self.assertIn("alpha", bodies[("equal", "Root")])

    def test_an_ignored_field_appears_in_neither(self):
        text = self.emit_for(
            [field("alpha", "Int", 0), field("secret", "Int", 1, ignored=True)]
        )
        self.assertNotIn("secret", text)
        self.assertIn("alpha", text)

    def test_ignoring_a_field_is_symmetric(self):
        both = function_bodies(
            self.emit_for([field("alpha", "Int", 0), field("beta", "Int", 1)])
        )
        one = function_bodies(
            self.emit_for(
                [field("alpha", "Int", 0), field("beta", "Int", 1, ignored=True)]
            )
        )
        # beta leaves the hash and the comparison together, never just one of them.
        self.assertIn("beta", both[("hash", "Root")])
        self.assertIn("beta", both[("equal", "Root")])
        self.assertNotIn("beta", one[("hash", "Root")])
        self.assertNotIn("beta", one[("equal", "Root")])


class TestUnionDescent(unittest.TestCase):
    """A skipped union member collapses every variant into one hash bucket."""

    def build(self):
        alpha = table(f"{NS}.AlphaAttr", [field("a", "Int", 0)])
        beta = table(f"{NS}.BetaAttr", [field("b", "Int", 0)])
        union = {
            "name": f"{NS}.Attrs",
            "values": [
                {"name": "NONE"},
                {"name": "AlphaAttr", "union_type": {"base_type": "Obj", "index": 1}},
                {"name": "BetaAttr", "union_type": {"base_type": "Obj", "index": 2}},
            ],
        }
        root = table(f"{NS}.Root", [field("attrs", "Union", 0, index=0)])
        return Emitter(schema([root, alpha, beta], [union]), f"{NS}.Root")

    def test_every_union_member_is_reachable(self):
        emitter = self.build()
        reached = emitter.reachable()
        self.assertIn(f"{NS}.AlphaAttr", reached)
        self.assertIn(f"{NS}.BetaAttr", reached)

    def test_every_union_member_is_emitted_in_both_functions(self):
        bodies = function_bodies(self.build().emit())
        # Each member gets its own pair, and the dispatch switch names them all.
        for member in ("AlphaAttr", "BetaAttr"):
            self.assertIn(("hash", member), bodies)
            self.assertIn(("equal", member), bodies)
            self.assertIn(member, bodies[("hash", "Attrs")])
            self.assertIn(member, bodies[("equal", "Attrs")])


class TestReachability(unittest.TestCase):
    def test_a_type_reached_only_through_an_ignored_field_is_not_emitted(self):
        # Ignoring the only edge to a type removes the type from the walk entirely.
        hidden = table(f"{NS}.Hidden", [field("h", "Int", 0)])
        root = table(f"{NS}.Root", [field("hidden", "Obj", 0, index=1, ignored=True)])
        emitter = Emitter(schema([root, hidden]), f"{NS}.Root")
        self.assertNotIn(f"{NS}.Hidden", emitter.reachable())

    def test_a_cycle_terminates(self):
        # The seen-set guard is what stops a self-referential schema recursing forever.
        root = table(f"{NS}.Root", [field("child", "Obj", 0, index=1)])
        child = table(f"{NS}.Child", [field("parent", "Obj", 0, index=0)])
        emitter = Emitter(schema([root, child]), f"{NS}.Root")
        self.assertEqual(sorted(emitter.reachable()), [f"{NS}.Child", f"{NS}.Root"])

    def test_a_vector_of_tables_is_descended(self):
        item = table(f"{NS}.Item", [field("v", "Int", 0)])
        root = table(
            f"{NS}.Root", [field("items", "Vector", 0, index=1, element="Obj")]
        )
        emitter = Emitter(schema([root, item]), f"{NS}.Root")
        self.assertIn(f"{NS}.Item", emitter.reachable())


class TestAccessorSpelling(unittest.TestCase):
    def test_a_cpp_keyword_field_gets_flatc_s_trailing_underscore(self):
        # flatc escapes keywords in accessor names; mismatching it emits code that
        # does not compile.
        self.assertEqual(accessor("virtual"), "virtual_")

    def test_an_ordinary_field_is_unchanged(self):
        self.assertEqual(accessor("shape"), "shape")

    def test_short_name_strips_the_namespace(self):
        self.assertEqual(short_name("a.b.Graph"), "Graph")


class TestDeterminism(unittest.TestCase):
    def test_two_emissions_of_one_schema_are_identical(self):
        # Any set/dict iteration leaking into the output would trip the drift hook
        # sporadically rather than reproducibly.
        root = table(
            f"{NS}.Root",
            [
                field("alpha", "Int", 0),
                field("beta", "String", 1),
                field("g", "Int", 2),
            ],
        )
        first = Emitter(schema([root]), f"{NS}.Root").emit()
        second = Emitter(schema([root]), f"{NS}.Root").emit()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
