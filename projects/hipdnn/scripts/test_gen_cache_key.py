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
    uid=False,
    uid_key=False,
    uid_domain=False,
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
    if uid:
        entry["attributes"].append({"key": "cache_uid"})
    if uid_key:
        entry["attributes"].append({"key": "cache_uid_key"})
    if uid_domain:
        entry["attributes"].append({"key": "cache_uid_domain"})
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

    def test_a_uid_key_field_is_dropped_from_the_fold(self):
        # It is the identity ordinals resolve against, not content: folding it would
        # only restate the position its element already occupies in the domain vector.
        self.assertFalse(self.emitter.keep(field("uid", "Long", 0, uid_key=True)))

    def test_a_uid_reference_still_participates(self):
        # `cache_uid` changes how a field folds, never whether it folds. A reference
        # that stopped participating would let two different wirings share a key.
        self.assertTrue(self.emitter.keep(field("x_tensor_uid", "Long", 0, uid=True)))


def uid_schema(operand_fields, *, domain_ignored=False):
    """A root with a `Tensor` domain vector plus a node table holding @p operand_fields."""
    tensor = table(
        f"{NS}.Tensor",
        [field("uid", "Long", 0, uid_key=True), field("dtype", "Int", 1)],
    )
    node = table(f"{NS}.Node", operand_fields)
    root = table(
        f"{NS}.Root",
        [
            field(
                "tensors",
                "Vector",
                0,
                index=0,
                element="Obj",
                uid_domain=not domain_ignored,
            ),
            field("nodes", "Vector", 1, index=1, element="Obj"),
        ],
    )
    return schema([tensor, node, root])


class TestUidCanonicalization(unittest.TestCase):
    """`cache_uid`: a uid folds as its ordinal, never as the caller's label.

    The relation this defends: two graphs whose tensors are numbered differently but
    wired identically must key the same, while two graphs wired differently must not.
    """

    def emit(self, operand_fields, **kwargs):
        return Emitter(uid_schema(operand_fields, **kwargs), f"{NS}.Root").emit()

    def test_a_uid_reference_is_folded_through_the_canon(self):
        bodies = function_bodies(
            self.emit([field("x_tensor_uid", "Long", 0, uid=True)])
        )
        self.assertIn("canon(value->x_tensor_uid())", bodies[("hash", "Node")])
        self.assertIn(
            "aCanon(a->x_tensor_uid()) != bCanon(b->x_tensor_uid())",
            bodies[("equal", "Node")],
        )

    def test_an_unannotated_scalar_is_folded_raw(self):
        # The impostor case: `PointwiseAttributes.axis_tensor_uid` is an axis index, not
        # a reference, so a name-driven sweep must not reach it.
        bodies = function_bodies(self.emit([field("axis_tensor_uid", "Long", 0)]))
        self.assertIn("hasher.raw(value->axis_tensor_uid())", bodies[("hash", "Node")])
        self.assertNotIn("canon(", bodies[("hash", "Node")])

    def test_an_optional_uid_keeps_its_presence_tag(self):
        # Presence is content of its own -- a ragged tensor is not a dense one -- so
        # canonicalizing the value must not collapse absent and present.
        operand = field("ragged_uid", "Long", 0, uid=True, optional=True)
        bodies = function_bodies(self.emit([operand]))
        self.assertIn("hasher.tag(optional ? 1 : 0)", bodies[("hash", "Node")])
        self.assertIn("canon(*optional)", bodies[("hash", "Node")])
        self.assertIn("aUid.has_value() != bUid.has_value()", bodies[("equal", "Node")])

    def test_a_uid_vector_is_folded_elementwise_through_the_canon(self):
        operand = field("peer_uids", "Vector", 0, element="Long", uid=True)
        bodies = function_bodies(self.emit([operand]))
        self.assertIn("canon(items->Get(index))", bodies[("hash", "Node")])
        self.assertIn(
            "aCanon(aItems->Get(index)) != bCanon(bItems->Get(index))",
            bodies[("equal", "Node")],
        )

    def test_the_root_entry_points_build_the_canon_from_the_domain(self):
        # Callers pass a graph, never a canon, so it cannot be built against the wrong
        # domain or omitted.
        header = self.emit([field("x_tensor_uid", "Long", 0, uid=True)])
        self.assertIn("UidCanon{value == nullptr ? nullptr : value->tensors()}", header)
        self.assertIn("UidCanon{a == nullptr ? nullptr : a->tensors()}", header)

    def test_a_uid_without_a_domain_is_rejected(self):
        # Nothing to resolve against: every reference would fold to the same sentinel and
        # graphs that differ would match.
        with self.assertRaises(SystemExit):
            self.emit([field("x_tensor_uid", "Long", 0, uid=True)], domain_ignored=True)

    def test_a_domain_without_exactly_one_key_is_rejected(self):
        broken = uid_schema([field("x_tensor_uid", "Long", 0, uid=True)])
        broken["objects"][0]["fields"].append(
            field("other_uid", "Long", 2, uid_key=True)
        )
        with self.assertRaises(SystemExit):
            Emitter(broken, f"{NS}.Root")

    def test_more_than_one_domain_is_rejected(self):
        # Ordinals are positions in one vector; two candidates have no defined answer.
        broken = uid_schema([field("x_tensor_uid", "Long", 0, uid=True)])
        broken["objects"][2]["fields"].append(
            field("more_tensors", "Vector", 2, index=0, element="Obj", uid_domain=True)
        )
        with self.assertRaises(SystemExit):
            Emitter(broken, f"{NS}.Root")

    def test_a_uid_on_a_non_integer_field_is_rejected(self):
        # `UidCanon` compares against an integer key, so a string or table reference
        # cannot resolve.
        with self.assertRaises(SystemExit):
            self.emit([field("x_tensor_uid", "String", 0, uid=True)])

    def test_a_uid_on_an_enum_scalar_is_rejected(self):
        # An enum is an index into the enum table, not a uid.
        with self.assertRaises(SystemExit):
            self.emit([field("x_tensor_uid", "Byte", 0, index=0, uid=True)])

    def test_a_uid_on_a_vector_of_tables_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.emit(
                [field("peer_uids", "Vector", 0, index=0, element="Obj", uid=True)]
            )

    def test_a_uid_key_on_a_non_integer_field_is_rejected(self):
        broken = uid_schema([field("x_tensor_uid", "Long", 0, uid=True)])
        broken["objects"][0]["fields"][0] = field("uid", "String", 0, uid_key=True)
        with self.assertRaises(SystemExit):
            Emitter(broken, f"{NS}.Root")

    def test_an_unresolved_uid_folds_the_original_value_not_a_shared_sentinel(self):
        # A dangling reference folds the caller's uid itself, never a shared sentinel,
        # so two dangling references only alias when they carry the same uid.
        header = self.emit([field("x_tensor_uid", "Long", 0, uid=True)])
        self.assertIn("return Fold{false, uid};", header)

    def test_a_resolved_uid_still_folds_to_exactly_its_ordinal(self):
        # A resolved reference must still fold to its ordinal, the invariant
        # renumbering-equivalence depends on.
        header = self.emit([field("x_tensor_uid", "Long", 0, uid=True)])
        self.assertIn("return Fold{true, static_cast<int64_t>(index)};", header)

    def test_resolved_and_unresolved_folds_cannot_collide_on_the_same_number(self):
        # `resolved` is folded and compared ahead of `value`, so ordinal 3 and a
        # dangling reference to uid 3 diverge on the tag regardless of magnitude.
        header = self.emit([field("x_tensor_uid", "Long", 0, uid=True)])
        self.assertIn("hasher.tag(fold.resolved ? 1 : 0);", header)
        self.assertIn("hasher.raw(fold.value);", header)
        self.assertIn("return a.resolved == b.resolved && a.value == b.value;", header)


class TestHasherConstants(unittest.TestCase):
    """The emitted basis/prime must be the canonical FNV-1a-64 constants.

    Pinned in hex against `StringUtil.hpp` so a reviewer can eyeball the digits
    against the spec instead of counting a 20-digit decimal literal.
    """

    def test_the_offset_basis_matches_the_canonical_fnv_1a_64_constant(self):
        root = table(f"{NS}.Root", [field("alpha", "Int", 0)])
        header = Emitter(schema([root]), f"{NS}.Root").emit()
        self.assertIn(
            "static constexpr uint64_t OFFSET_BASIS = 0xcbf29ce484222325ULL;", header
        )

    def test_the_prime_matches_the_canonical_fnv_1a_64_constant(self):
        root = table(f"{NS}.Root", [field("alpha", "Int", 0)])
        header = Emitter(schema([root]), f"{NS}.Root").emit()
        self.assertIn(
            "static constexpr uint64_t PRIME        = 0x100000001b3ULL;", header
        )


class TestUnhandledBaseType(unittest.TestCase):
    """A base type the generator has no fold rule for must fail loudly.

    `Array` is the concrete case that motivates this -- its accessor returns a
    pointer, which would silently satisfy `Hasher::raw`'s trivially-copyable check
    and hash/compare the address rather than the value.
    """

    def test_an_array_field_raises_naming_the_field_and_its_base_type(self):
        root = table(f"{NS}.Root", [field("id", "Array", 0, element="UByte", index=0)])
        with self.assertRaises(SystemExit) as ctx:
            Emitter(schema([root]), f"{NS}.Root").emit()
        message = str(ctx.exception)
        self.assertIn("Root.id", message)
        self.assertIn("Array", message)

    def test_an_ordinary_scalar_field_does_not_raise(self):
        # The whitelist must not regress the common case it wraps.
        root = table(f"{NS}.Root", [field("alpha", "Int", 0)])
        Emitter(schema([root]), f"{NS}.Root").emit()


def function_bodies(text):
    """Map each emitted definition to its body text.

    The header interleaves a `hashAppend`/`logicallyEqual` pair per type, so the two
    cannot be separated by splitting the file once. Keys are `("hash"|"equal", type)`;
    forward declarations (which end in `;`) are skipped, as are the root's two
    convenience overloads, which take no `UidCanon` and only delegate to the real walk.
    """
    bodies, current, depth = {}, None, 0
    for line in text.splitlines():
        if current is None:
            if not line.startswith("inline ") or line.rstrip().endswith(";"):
                continue
            if "UidCanon&" not in line:
                continue
            kind = "hash" if "hashAppend" in line else "equal"
            inside = line[line.index("(") + 1 : line.rindex(")")]
            names = [
                w.strip("*&,")
                for w in inside.split()
                if w[:1].isupper() and "UidCanon" not in w
            ]
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
