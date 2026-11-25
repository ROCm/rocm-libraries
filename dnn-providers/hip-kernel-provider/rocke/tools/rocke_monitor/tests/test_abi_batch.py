# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for ABI lanes (G9) and merge batches."""

import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import abi, batch  # noqa: E402
from rkmon_lib.store import empty_state  # noqa: E402

HEADER = """/* Binary ABI of the shared library.
 * 2: rocke_memory_caps_t gained mid-struct fields.
 * 3: rocke_mma_op_t gained emulation fields at its end. */
#define ROCKE_ABI_VERSION 3

/* Wire ABI.
 * 2: list-tagged ("l") kernel attrs. */
#define ROCKE_RECIPE_ABI 2
"""


def rec(n, base="develop", head_ref=None, lanes=None, notes=(), families=()):
    head = (str(n) * 40)[
        :40
    ]  # distinct 12-char prefixes, as the pair cache keys on them
    return {
        "n": n,
        "base": base,
        "head_ref": head_ref or f"b{n}",
        "head": head,
        "size": 100 * n,
        "abi_cache": {
            "head": head,
            "lanes": lanes or {},
            "notes": list(notes),
            "families": list(families),
        },
    }


def claim(to, hunk, history=True, dropped=()):
    return {
        "from": to - 1,
        "to": to,
        "history_line": history,
        "dropped_history": list(dropped),
        "hunk": hunk,
    }


class TestParse(unittest.TestCase):
    def test_counters_and_history(self):
        h = abi.parse_header(HEADER)
        self.assertEqual((h["binary"], h["recipe"]), (3, 2))
        self.assertEqual(sorted(h["hist_binary"]), [2, 3])
        self.assertEqual(sorted(h["hist_recipe"]), [2])

    def test_code_lines_ignore_comments_and_versions(self):
        diff = "--- a\n+++ b\n- * 2: x\n+ * 3: y\n-#define ROCKE_RECIPE_ABI 2\n+#define ROCKE_RECIPE_ABI 3\n+int f(void);\n"
        self.assertEqual(abi._code_lines(diff), ["+int f(void);"])


class TestHeaders(unittest.TestCase):
    def test_abi_surface(self):
        k = "rk"
        self.assertTrue(abi.is_abi_header("rk/platform/cpp/include/rocke/ir.h", k))
        for name in (
            "instance_gemm.h",
            "helper_x.h",
            "attention_internal.h",
            "sub/dir.h",
            "error.hpp",
        ):
            self.assertFalse(
                abi.is_abi_header(f"rk/platform/cpp/include/rocke/{name}", k)
            )


class TestStructLayouts(unittest.TestCase):
    """A field inserted mid-struct in spec and problem structs without a bump is an ABI change, never "no ABI
    change" (lessons L-527)."""

    V1 = (
        "/* spec */\ntypedef struct rocke_x_spec\n{\n    int a; /* 1 */\n    bool b; /* False */\n"
        "    struct { int lo, hi; } range;\n} rocke_x_spec_t;\n\nstruct rocke_y { float f; };\n"
    )

    def test_comments_and_spacing_are_not_layout(self):
        v = self.V1.replace("/* 1 */", "/* one */").replace("    int a;", "int   a ;")
        self.assertEqual(abi.struct_layouts(v), abi.struct_layouts(self.V1))
        self.assertEqual(
            sorted(abi.struct_layouts(self.V1)), ["rocke_x_spec_t", "rocke_y"]
        )  # not the member "range"

    def test_a_field_added_anywhere_or_retyped_is_a_change(self):
        """surface_changes counts a struct whose layout differs: each of these moves the size or an offset."""
        was = abi.struct_layouts(self.V1)
        mid = self.V1.replace("    bool b;", "    bool c;\n    bool b;")
        end = self.V1.replace("} range;", "} range;\n    bool emit_lse;")
        retyped = self.V1.replace("int a;", "long a;")
        for v in (mid, end, retyped):
            now = abi.struct_layouts(v)
            self.assertNotEqual(now["rocke_x_spec_t"], was["rocke_x_spec_t"], v)
            self.assertEqual(now["rocke_y"], was["rocke_y"], v)

    def test_a_renamed_field_moves_no_byte(self):
        renamed = self.V1.replace("bool b;", "bool keep_b;").replace(
            "int lo, hi;", "int low, high;"
        )
        self.assertEqual(abi.struct_layouts(renamed), abi.struct_layouts(self.V1))
        self.assertNotEqual(
            abi.struct_layouts(self.V1.replace("int lo, hi;", "int lo, hi, mid;")),
            abi.struct_layouts(self.V1),
        )  # a third declarator is a field

    def test_type_only(self):
        for decl, want in (
            ("const char *name", "const char*"),
            ("int a[4]", "int[4]"),
            ("unsigned x : 3", "unsigned:3"),
            ("void (*fn)(int)", "void(*)(int)"),
            ("rocke_value_t** out", "rocke_value_t**"),
            ("int lo, hi", "int,"),
        ):
            self.assertEqual(abi._type_only(decl), want, decl)

    def test_header_kinds(self):
        p = "rk/platform/cpp/include/rocke/"
        self.assertEqual(abi.header_kind(p + "ir.h", "rk"), "public")
        for name in (
            "helper_helper_rocke.instances.gfx942.attention_tiled_2d.h",
            "instance_gfx942_attention_tiled_3d.h",
        ):
            self.assertEqual(abi.header_kind(p + name, "rk"), "generated")
        for name in (
            "abi.h",
            "attention_internal.h",
            "instance_gfx950_attention_tiled_2d_internal.h",
            "sub/dir.h",
            "x.hpp",
        ):
            self.assertIsNone(abi.header_kind(p + name, "rk"))

    def test_scan_notes_a_public_and_a_generated_struct_unless_bumped(self):
        """A generated instance_*/helper_* header's struct crosses the C API like a public one: abi.h's history bumps
        for them, and Copilot's 8 struct-growth items (#11111, #22222 x2, #33333 x2, #44444, #55555, #66666, #77777,
        #88888) were all fixed by a bump."""
        with tempfile.TemporaryDirectory() as d:

            def g(*a):
                return subprocess.run(
                    ["git", "-C", d, *a], check=True, capture_output=True, text=True
                ).stdout.strip()

            g("init", "-q", "-b", "develop")
            g("config", "user.email", "t@t")
            g("config", "user.name", "t")
            inc = pathlib.Path(d, "rk/platform/cpp/include/rocke")
            inc.mkdir(parents=True)
            (inc / "abi.h").write_text(HEADER)
            (inc / "caps.h").write_text(
                "typedef struct rocke_caps { int a; } rocke_caps_t;\n"
            )
            (inc / "helper_x.h").write_text(
                "typedef struct rocke_x_spec { bool a; int n; } rocke_x_spec_t;\n"
            )
            g("add", "-A")
            g("commit", "-qm", "base")
            mb = g("rev-parse", "HEAD")
            (inc / "caps.h").write_text(
                "typedef struct rocke_caps { int a; int b; } rocke_caps_t;\n"
            )
            (inc / "helper_x.h").write_text(
                "typedef struct rocke_x_spec { bool a; bool bias; int n; } rocke_x_spec_t;\n"
            )
            g("commit", "-qam", "pr")
            c = abi.scan(d, "rk", mb, "HEAD")
            self.assertEqual(c["lanes"], {})
            self.assertTrue(
                any(
                    n.startswith(
                        "C struct layouts change without a binary bump: rocke_caps_t"
                    )
                    for n in c["notes"]
                ),
                c,
            )
            self.assertTrue(
                any(
                    n.startswith(
                        "generated-header struct layouts change without a binary bump: "
                        "rocke_x_spec_t"
                    )
                    for n in c["notes"]
                ),
                c,
            )
            self.assertEqual(c["info"], [])
            (inc / "abi.h").write_text(
                HEADER.replace(
                    " at its end. */\n#define ROCKE_ABI_VERSION 3",
                    " at its end.\n * 4: rocke_caps_t gained b. */\n"
                    "#define ROCKE_ABI_VERSION 4",
                )
            )
            g("commit", "-qam", "bump")
            c = abi.scan(d, "rk", mb, "HEAD")
            self.assertEqual(c["lanes"]["binary"]["to"], 4)
            self.assertEqual((c["notes"], c["info"]), ([], []))

    def test_a_bump_must_move_the_portable_ir_mirror_too(self):
        """A bump of ROCKE_ABI_VERSION that leaves abi.py's BINARY_ABI behind makes online.load() refuse the engine,
        and no CI lane builds the shared engine that would show it."""
        with tempfile.TemporaryDirectory() as d:

            def g(*a):
                return subprocess.run(
                    ["git", "-C", d, *a], check=True, capture_output=True, text=True
                ).stdout.strip()

            g("init", "-q", "-b", "develop")
            g("config", "user.email", "t@t")
            g("config", "user.name", "t")
            inc = pathlib.Path(d, "rk/platform/cpp/include/rocke")
            inc.mkdir(parents=True)
            src = pathlib.Path(d, "rk", abi.ABI_PY)
            src.parent.mkdir(parents=True)
            (inc / "abi.h").write_text(HEADER)
            src.write_text(
                "#: Wire ABI.\nRECIPE_ABI = 2\n\n#: Binary ABI.\nBINARY_ABI = 3\n"
            )
            g("add", "-A")
            g("commit", "-qm", "base")
            mb = g("rev-parse", "HEAD")
            self.assertEqual(
                abi.parse_mirrors(src.read_text()), {"binary": 3, "recipe": 2}
            )
            (inc / "abi.h").write_text(
                HEADER.replace(
                    " at its end. */\n#define ROCKE_ABI_VERSION 3",
                    " at its end.\n * 4: rocke_caps_t gained b. */\n"
                    "#define ROCKE_ABI_VERSION 4",
                )
            )
            g("commit", "-qam", "bump abi.h only")
            c = abi.scan(d, "rk", mb, "HEAD")
            self.assertEqual(c["lanes"]["binary"]["to"], 4)
            self.assertEqual(
                c["notes"],
                [
                    "portable_ir/src/abi.py BINARY_ABI = 3 but abi.h ROCKE_ABI_VERSION = 4: bump "
                    "both together (the shared engine refuses to load)"
                ],
            )
            src.write_text(src.read_text().replace("BINARY_ABI = 3", "BINARY_ABI = 4"))
            g("commit", "-qam", "and its mirror")
            self.assertEqual(abi.scan(d, "rk", mb, "HEAD")["notes"], [])


class TestExportedFunctions(unittest.TestCase):
    """An exported function the PR removes or whose signature it changes breaks a caller built against the old header
    (abi.h: "a function's signature changes" bumps): #99999 removed the binary_search_iters accessors, and #12345
    changed rocke_choose_load_vec's return type, both without a bump."""

    HDR = (
        '#ifdef __cplusplus\nextern "C" {\n#endif\n/* accessors */\n'
        "int rocke_unified_attention_3d_tiled_spec_binary_search_iters(\n"
        "    const rocke_unified_attention_3d_tiled_spec_t* s);\n"
        "int\n    rocke_choose_load_vec(int tile_m, int tile_n, int tile_k, int block_size, int* out_vec);\n"
        "static inline int rocke_local_helper(int x) { return x + 1; }\n"
        "typedef int (*rocke_cb_t)(int);\n"
        "typedef struct rocke_s { int a; } rocke_s_t;\n"
        '#ifdef __cplusplus\n} /* extern "C" */\n#endif\n'
    )

    def test_declarations_with_external_linkage(self):
        self.assertEqual(
            abi.exported_functions(self.HDR),
            {
                "rocke_unified_attention_3d_tiled_spec_binary_search_iters": "int(const rocke_unified_attention_3d_tiled_spec_t*)",
                "rocke_choose_load_vec": "int(int,int,int,int,int*)",
            },
        )
        renamed = self.HDR.replace(
            "int* out_vec", "int *width"
        )  # a parameter's name is no part of it
        self.assertEqual(
            abi.exported_functions(renamed), abi.exported_functions(self.HDR)
        )

    def scan(self, after: str, other: str | None = None) -> dict:
        with tempfile.TemporaryDirectory() as d:

            def g(*a):
                return subprocess.run(
                    ["git", "-C", d, *a], check=True, capture_output=True, text=True
                ).stdout.strip()

            g("init", "-q", "-b", "develop")
            g("config", "user.email", "t@t")
            g("config", "user.name", "t")
            inc = pathlib.Path(d, "rk/platform/cpp/include/rocke")
            inc.mkdir(parents=True)
            (inc / "abi.h").write_text(HEADER)
            (inc / "instance_gfx942_attention_tiled_3d.h").write_text(self.HDR)
            g("add", "-A")
            g("commit", "-qm", "base")
            mb = g("rev-parse", "HEAD")
            (inc / "instance_gfx942_attention_tiled_3d.h").write_text(after)
            if other is not None:
                (inc / "instance_gfx942_attention_tiled_3d_internal.h").write_text(
                    other
                )
            g("add", "-A")
            g("commit", "-qm", "pr")
            return abi.scan(d, "rk", mb, "HEAD")

    def test_a_removed_or_changed_function_is_a_note(self):
        after = self.HDR.replace(
            "int rocke_unified_attention_3d_tiled_spec_binary_search_iters(\n"
            "    const rocke_unified_attention_3d_tiled_spec_t* s);\n",
            "",
        )
        after = after.replace(
            "int\n    rocke_choose_load_vec", "rocke_status_t rocke_choose_load_vec"
        )
        c = self.scan(after)
        self.assertEqual(
            [n for n in c["notes"] if n.startswith("exported C functions")],
            [
                "exported C functions removed: rocke_unified_attention_3d_tiled_spec_binary_search_iters; signature "
                "changed: rocke_choose_load_vec without a binary bump (a caller built against the old header fails to "
                "resolve or misreads them: bump ROCKE_ABI_VERSION and abi.py BINARY_ABI together)"
            ],
        )

    def test_appended_or_moved_functions_and_statics_are_fine(self):
        appended = self.HDR.replace(
            "typedef int", "int rocke_new_accessor(int x);\ntypedef int"
        )
        static = self.HDR.replace("x + 1", "x + 2").replace(
            "rocke_local_helper(int x)", "rocke_local_helper(long x)"
        )
        for after in (appended, static):
            self.assertEqual(self.scan(after)["notes"], [])
        moved = self.HDR.replace(
            "int\n    rocke_choose_load_vec(int tile_m, int tile_n, int tile_k, int block_size, "
            "int* out_vec);\n",
            "",
        )
        self.assertEqual(
            self.scan(
                moved, other="int rocke_choose_load_vec(int, int, int, int, int*);\n"
            )["notes"],
            [],
        )


class TestClaimsAgainstDevelop(unittest.TestCase):
    """A PR whose own ABI numbers develop already used for other changes (#1111 at base 5 with 6, #2222 at base 3
    with 4-6) is not shown as a plain "claims 6", and a PR carrying develop's own bump claims nothing.
    """

    @staticmethod
    def header(entries: dict) -> str:
        top = max(entries)
        lines = "".join(f"\n * {k}: {v}" for k, v in sorted(entries.items()))
        return f"/* Binary ABI.{lines} */\n#define ROCKE_ABI_VERSION {top}\n\n/* Wire ABI. */\n#define ROCKE_RECIPE_ABI 2\n"

    def test_own_entries_and_numbers_taken_on_develop(self):
        base = {2: "caps fields", 3: "mma emulation"}
        dev = {**base, 4: "develop: tensor view buffer", 5: "develop: attention spec"}
        with tempfile.TemporaryDirectory() as d:

            def g(*a):
                return subprocess.run(
                    ["git", "-C", d, *a], check=True, capture_output=True, text=True
                ).stdout.strip()

            g("init", "-q", "-b", "develop")
            g("config", "user.email", "t@t")
            g("config", "user.name", "t")
            inc = pathlib.Path(d, "rk/platform/cpp/include/rocke")
            inc.mkdir(parents=True)
            f = inc / "abi.h"

            def commit(entries, msg):
                f.write_text(self.header(entries))
                g("add", "-A")
                g("commit", "-qm", msg)
                return g("rev-parse", "HEAD")

            mb = commit(base, "base")
            commit(dev, "develop moves to 5")
            cases = {
                "carrier": dev,
                "stale": {**base, 4: "pr: indexed mma"},
                "parallel": {**base, 4: "pr: gett", 5: "pr: gett f64"},
                "ahead": {**dev, 6: "pr: fp8 gemm"},
            }
            got = {}
            for name, entries in cases.items():
                g("checkout", "-q", "-b", name, mb)
                commit(entries, name)
                got[name] = abi.scan(d, "rk", mb, name, "develop")
            self.assertEqual(got["carrier"]["lanes"], {})
            self.assertIn(
                "carries develop's own binary ABI 5", got["carrier"]["info"][0]
            )
            stale, par, ahead = (
                got[k]["lanes"]["binary"] for k in ("stale", "parallel", "ahead")
            )
            self.assertEqual((stale["own"], stale["taken"]), ([4], [4]))
            self.assertEqual((par["own"], par["taken"]), ([4, 5], [4, 5]))
            self.assertEqual(
                (ahead["to"], ahead["own"], ahead["taken"]), (6, [6], [])
            )  # 4-5 are develop's own

    def test_lanes_show_and_gate_on_taken_numbers(self):
        dev = {"binary": 6, "recipe": 3}
        ok = dict(claim(7, "aaa"), own=[7], taken=[])
        stale = dict(claim(6, "bbb"), own=[6], taken=[6])
        par = dict(claim(6, "ccc"), own=[4, 5, 6], taken=[4, 5, 6])
        recs = [
            rec(10, lanes={"binary": ok}),
            rec(11, lanes={"binary": stale}),
            rec(12, lanes={"binary": par}),
        ]
        L = abi.build_lanes(dev, recs, [10, 11, 12], {10})
        texts = [abi.claim_text(g) for g in L["binary"]["groups"]]
        self.assertEqual(
            texts,
            [
                "7",
                "6 (6 taken on develop by other changes: renumber)",
                "6 (4–6 taken on develop by other changes: renumber)",
            ],
        )
        st, note = abi.gate_abi(recs[2], L, set())
        self.assertEqual(st, "fail")
        self.assertIn("4–6 taken on develop", note)
        L = abi.build_lanes(
            dev, recs[1:2], [11], {11}
        )  # the stale claim at the head: renumber, keep develop's line
        st, note = abi.gate_abi(recs[1], L, set())
        self.assertIn("set ROCKE_ABI_VERSION to 7", note)
        self.assertIn("keep develop's history lines [6]", note)


class TestLanes(unittest.TestCase):
    def setUp(self):
        self.dev = {"binary": 3, "recipe": 2}
        # Three different claims on binary 4, a stack (#20 <- #21) sharing one claim, and #30 claiming 5.
        self.recs = [
            rec(10, lanes={"binary": claim(4, "aaa")}),
            rec(11, lanes={"binary": claim(4, "bbb")}),
            rec(20, head_ref="stack20", lanes={"binary": claim(4, "ccc")}),
            rec(21, base="stack20", lanes={"binary": claim(4, "ccc")}),
            rec(30, lanes={"binary": claim(5, "ddd")}),
            rec(40, lanes={"recipe": claim(3, "eee")}),
            rec(41, lanes={"recipe": claim(3, "eee")}),
        ]

    def lanes(self, order, ready):
        return abi.build_lanes(self.dev, self.recs, order, set(ready))

    def test_readiness_order_and_reservations(self):
        L = self.lanes([11, 10, 20, 21, 30, 40, 41], ready=[10, 20])
        b = L["binary"]
        self.assertEqual(
            [g["members"] for g in b["groups"]], [[10], [20, 21], [11], [30]]
        )
        self.assertEqual([g["reserve"] for g in b["groups"]], [4, 5, 6, 7])
        self.assertEqual(b["head"], 0)
        # Identical recipe hunks in two PRs are one claim.
        self.assertEqual(L["recipe"]["groups"][0]["members"], [40, 41])

    def test_gate(self):
        L = self.lanes([10, 11, 20, 21, 30], ready=[10, 20, 30])
        by = {r["n"]: r for r in self.recs}
        self.assertEqual(abi.gate_abi(by[10], L, set())[0], "pass")
        st, note = abi.gate_abi(by[20], L, set())
        self.assertEqual(st, "fail")
        self.assertIn("position 2", note)
        self.assertIn("tentative number 5", note)
        # #30 claims 5 but would be head only after the others: it waits.
        self.assertEqual(abi.gate_abi(by[30], L, set())[0], "fail")

    def test_head_must_claim_develop_plus_one(self):
        self.recs = [rec(30, lanes={"binary": claim(5, "ddd")})]
        L = self.lanes([30], ready=[30])
        st, note = abi.gate_abi(self.recs[0], L, set())
        self.assertEqual(st, "fail")
        self.assertIn("set ROCKE_ABI_VERSION to 4", note)

    def test_history_and_notes(self):
        self.recs = [
            rec(10, lanes={"binary": claim(4, "aaa", history=False, dropped=[2])})
        ]
        st, note = abi.gate_abi(self.recs[0], self.lanes([10], ready=[10]), set())
        self.assertIn("history line", note)
        self.assertIn("restore dropped history lines [2]", note)
        r = rec(50, notes=["abi.h code lines change without a binary bump (2 lines)"])
        L = abi.build_lanes(self.dev, [r], [50], {50})
        self.assertEqual(abi.gate_abi(r, L, set())[0], "pending")
        self.assertEqual(abi.gate_abi(r, L, {(50, r["head"])})[0], "pass")

    def test_lane_claims_are_who_is_in_a_lane_whatever_the_order(self):
        """The merge plan's ABI conflicts read the lanes' claims before the queue
        orders the lanes: build_lanes' claims, the same in any order, and a claim without a recorded hunk is its own.
        """
        claims = abi.lane_claims(self.recs)
        members = lambda cs: sorted(sorted(c["members"]) for c in cs)  # noqa: E731
        self.assertEqual(
            (members(claims["binary"]), members(claims["recipe"])),
            ([[10], [11], [20, 21], [30]], [[40, 41]]),
        )
        self.assertEqual(
            next(c for c in claims["binary"] if c["members"] == [30])["to"], 5
        )
        self.assertEqual(
            members(abi.lane_claims(list(reversed(self.recs)))["binary"]),
            members(claims["binary"]),
        )
        for order in ([11, 10, 20, 21, 30], [30, 21, 20, 11, 10]):
            self.assertEqual(
                members(self.lanes(order, ready=[])["binary"]["groups"]),
                members(claims["binary"]),
            )
        bare = [
            rec(60, lanes={"binary": {"to": 4}}),
            rec(61, lanes={"binary": {"to": 4}}),
        ]
        self.assertEqual(members(abi.lane_claims(bare)["binary"]), [[60], [61]])

    def test_first_in_line_is_pending_not_behind_itself(self):
        """G9 never reads "position 1 (behind #3333)" on #3333 itself while no lane member can land."""
        L = self.lanes([10, 11], ready=[])
        self.assertIsNone(L["binary"]["head"])
        by = {r["n"]: r for r in self.recs}
        self.assertEqual(
            abi.gate_abi(by[10], L, set()),
            (
                "pending",
                "binary ABI (ROCKE_ABI_VERSION) lane position 1: it takes the next number once it can land; "
                "tentative number 4",
            ),
        )
        st, note = abi.gate_abi(by[11], L, set())
        self.assertEqual(st, "fail")
        self.assertIn("lane position 2 (behind #10)", note)
        notes = rec(
            10,
            lanes={"binary": claim(4, "aaa")},
            notes=["abi.h code lines change without a binary bump"],
        )
        st, note = abi.gate_abi(
            notes, L, set()
        )  # first in line, with a note to verify: both said
        self.assertEqual(st, "pending")
        self.assertTrue(
            note.startswith("binary ABI (ROCKE_ABI_VERSION) lane position 1")
            and "verify in review" in note
        )


class TestBatches(unittest.TestCase):
    def test_compatible_members_only(self):
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        recs = {n: dict(rec(n), files_cache={"hot": []}) for n in (1, 2, 3, 4)}
        st["prs"] = {str(n): r for n, r in recs.items()}
        key = lambda a, b: f"{recs[a]['head'][:12]}:{recs[b]['head'][:12]}"
        st["pair_cache"] = {
            key(1, 2): {"conflicts": [], "undecided": False},
            key(1, 3): {"conflicts": ["x.py"], "undecided": False},
            key(2, 3): {"conflicts": [], "undecided": False},
            key(1, 4): {"conflicts": [], "undecided": False},
            key(2, 4): {"conflicts": [], "undecided": False},
            key(3, 4): {"conflicts": [], "undecided": False},
        }
        # #3 conflicts with #1; #4 shares an ABI lane with #1.
        lanes = {
            "binary": {
                "develop": 3,
                "head": 0,
                "groups": [
                    {"members": [1], "root": 1, "to": 4, "reserve": 4, "ready": True},
                    {"members": [4], "root": 4, "to": 4, "reserve": 5, "ready": True},
                ],
            }
        }
        ready = [{"n": n, "size": recs[n]["size"]} for n in (1, 2, 3, 4)]
        bs = batch.build_batches({}, st, ready, lanes)
        self.assertEqual([b["members"] for b in bs], [[1, 2], [3, 4]])
        self.assertTrue(bs[0]["check_prefixes"])  # #1 is in an ABI lane
        self.assertEqual(bs[0]["status"], "candidate")


class TestUserGroups(unittest.TestCase):
    def test_pinned_group_first_and_notes(self):
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        recs = {
            n: dict(rec(n), files_cache={"hot": []}, status="open") for n in (1, 2, 3)
        }
        st["prs"] = {str(n): r for n, r in recs.items()}
        key = lambda a, b: f"{recs[a]['head'][:12]}:{recs[b]['head'][:12]}"
        st["pair_cache"] = {
            key(1, 2): {"conflicts": [], "undecided": False},
            key(1, 3): {"conflicts": [], "undecided": False},
            key(2, 3): {"conflicts": ["y.py"], "undecided": False},
        }
        st["groups"] = [[3, 2]]
        ready = [{"n": n, "size": recs[n]["size"]} for n in (1, 3)]  # #2 is not ready
        bs = batch.build_batches({}, st, ready, {})
        self.assertEqual(bs[0]["members"], [3, 2])
        self.assertEqual(bs[0]["origin"], "user")
        self.assertEqual(bs[0]["status"], "waiting")
        self.assertIn("#2 not ready", bs[0]["notes"])
        self.assertIn("#3 and #2 conflict", bs[0]["notes"])
        self.assertEqual(bs[1]["members"], [1])  # the rest are batched automatically


if __name__ == "__main__":
    unittest.main()
