# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rkmon_lib/peer_checks.py (tools/peer_checks.py): the checks from what peer reviewers catch and our reviews did not
raise first (peer_reviewer_gap.md). Each true positive below is a real line or title from the PR named beside it; each
true negative is a case the check must leave alone. tests/test_peer_checks_pinned.py runs the same checks on the mirror
at the real commits."""

import contextlib
import importlib.util
import io
import json
import pathlib
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rkmon_lib import peer_checks as pc, registry, review  # noqa: E402

K = "prov/rk"
TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "peer_checks.py"


def g(repo, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def diff_of(files: dict[str, str], new: bool = True) -> dict:
    """A parsed diff whose files add `files`' lines (each a new file unless `new` is False)."""
    out = []
    for path, text in files.items():
        lines = text.splitlines()
        out += [f"diff --git a/{path} b/{path}"] + (
            ["new file mode 100644", "--- /dev/null"] if new else [f"--- a/{path}"]
        )
        out += [f"+++ b/{path}", f"@@ -0,0 +1,{len(lines)} @@"] + [
            "+" + ln for ln in lines
        ]
    return pc.parse_diff("\n".join(out) + "\n")


class TestParseDiff(unittest.TestCase):
    def test_new_files_line_numbers_and_renames(self):
        text = (
            "diff --git a/x.py b/x.py\nnew file mode 100644\n--- /dev/null\n+++ b/x.py\n@@ -0,0 +1,2 @@\n+a\n+b\n"
            "diff --git a/y.py b/y.py\n--- a/y.py\n+++ b/y.py\n@@ -3,0 +7,1 @@\n+c\n"
            "diff --git a/old.py b/new.py\nsimilarity index 100%\nrename from old.py\nrename to new.py\n"
            "diff --git a/gone.py b/gone.py\ndeleted file mode 100644\n--- a/gone.py\n+++ /dev/null\n"
            "@@ -1 +0,0 @@\n-z\n"
        )
        d = pc.parse_diff(text)
        self.assertEqual(
            d,
            {
                "x.py": {"new": True, "lines": [(1, "a"), (2, "b")]},
                "y.py": {"new": False, "lines": [(7, "c")]},
            },
        )


class TestTrackers(unittest.TestCase):
    def test_any_prefix_is_a_tracker(self):
        d = diff_of(
            {
                # #11111 (the peer-review rules §E.8, never raised by us): Jira IDs in committed docs.
                "projects/hipdnn/docs/KNOWN_DIVERGENCES.md": "Tracked as TKEY" "-2967.",
                # #22222 at 806727dc (the peer-review rules §E.8).
                f"{K}/library/tests/dispatch/attention/test_review_followups.py": "# ABCD"
                "-1932: k_scale mirror",
                # #33333 at ac3195af (Copilot, fixed): IDs in decline reasons.
                f"{K}/library/dispatch/sdpa/bind_unified.py": 'reason = "fp16 d=256 declines (QRS'
                "-34, UV"
                '-01)"',
                # #44444 at 6b42d6fd (Copilot, fixed): in a source comment.
                "src/tests/TestUnifiedAttentionGeometry.cpp": "// Regression for ABCD"
                "-2544.",
            }
        )
        self.assertEqual(
            sorted((x["at"], x["id"]) for x in pc.trackers(d, K)),
            [
                ("library/dispatch/sdpa/bind_unified.py:1", "QRS" "-34"),
                ("library/dispatch/sdpa/bind_unified.py:1", "UV" "-01"),
                (
                    "library/tests/dispatch/attention/test_review_followups.py:1",
                    "ABCD" "-1932",
                ),
                ("projects/hipdnn/docs/KNOWN_DIVERGENCES.md:1", "TKEY" "-2967"),
                ("src/tests/TestUnifiedAttentionGeometry.cpp:1", "ABCD" "-2544"),
            ],
        )

    def test_standards_registers_and_data_files_are_not(self):
        d = diff_of(
            {
                f"{K}/docs/semantics.md": "SEM"
                "-001 holds; IEEE-754 rounding; SHA-256 key; LLVM-20 lane; MI-300 class; "
                "gfx950-only; a utf-8 file; FP-8 note",
                f"{K}/library/data/cases.json": '{"id": "ABCD' '-1234"}',
                f"{K}/library/x.py": "x = foo-123  # lower case is no tracker; A-12 one letter",
            }
        )
        self.assertEqual(pc.trackers(d, K), [])
        # Without the allowlist: SEM IEEE SHA LLVM MI (FP-8 has one digit, A-12 one letter).
        self.assertEqual(len(pc.trackers(d, K, allow=())), 5)

    def test_the_prefixes_are_the_package_s_unless_the_configuration_names_some(self):
        """The prefixes that never flag are config tracker_allow's, the package's (config.defaults.json) while
        it is empty: the review's checks pass them to the tool, which takes the package's without them.
        """
        package = tuple(registry.default("tracker_allow"))
        self.assertIn("IEEE", package)
        self.assertEqual(pc.tracker_allow({}), package)
        self.assertEqual(pc.tracker_allow({"tracker_allow": []}), package)
        self.assertEqual(pc.tracker_allow({"tracker_allow": ["ABC"]}), ("ABC",))
        self.assertEqual(
            review.peer_args({}, "trackers"),
            " --allow " + shlex.quote(",".join(package)),
        )
        spec = importlib.util.spec_from_file_location("peer_checks_tool", TOOL)
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        seen = []

        def run(check, a):
            seen.append(a.allow)
            return {"check": check, "error": None, "items": []}

        with mock.patch.object(tool, "run", run), mock.patch.object(
            tool, "render", lambda res: []
        ), contextlib.redirect_stdout(io.StringIO()):
            tool.main(["trackers", "--repo", "r", "--head", "h", "--rocke-path", "rk"])
            tool.main(
                ["trackers", "--repo", "r", "--head", "h", "--rocke-path", "rk"]
                + ["--allow", "ABC"]
            )
        self.assertEqual(seen, [package, ("ABC",)])

    def test_render_and_verdict(self):
        res = {
            "check": "trackers",
            "error": None,
            "items": [
                {"at": "a.md:1", "id": "TKEY" "-2967"},
                {"at": "b.md:2", "id": "TKEY" "-2967"},
            ],
        }
        self.assertEqual(
            pc.render(res),
            [
                "FLAG a.md:1: TKEY" "-2967",
                "FLAG b.md:2: TKEY" "-2967",
                "TRACKERS fail 2 tracker IDs in added lines: TKEY" "-2967",
            ],
        )


class TestEnvDocs(unittest.TestCase):
    def test_reads_and_not_reads(self):
        d = diff_of(
            {
                # #55555 (ours): a new driver knob.
                f"{K}/platform/python/rocke/portable_ir/drivers/roll_hsaco_parity.py": 'ARCH = os.environ.get("ROCKE_ROLL_ARCH", "gfx950")',
                # #66666: an aliased os module.
                f"{K}/platform/python/rocke/instances/common/gemm_universal.py": '_agpr = _os_agpr.environ.get("ROCKE_EXP_AGPR_ALLOC")\n'
                'os.environ["ROCKE_SET_ONLY"] = "1"\nenv.pop("ROCKE_POPPED", None)\n'
                'monkeypatch.setenv("ROCKE_SET_TOO", "1")\nv = _env_flag("ROCKE_HELPER_READ")\n'
                '# os.environ.get("ROCKE_IN_A_COMMENT")',
                f"{K}/platform/cpp/core/arch.cpp": 'const char* v = std::getenv("ROCKE_CPP_READ");',
                f"{K}/platform/tests/test_x.py": 'os.environ.get("ROCKE_TEST_ONLY")',
                f"{K}/docs/x.md": 'os.environ.get("ROCKE_IN_DOCS")',
            }
        )
        self.assertEqual(
            [r["name"] for r in pc.env_reads(d, K)],
            [
                "ROCKE_CPP_READ",
                "ROCKE_EXP_AGPR_ALLOC",
                "ROCKE_HELPER_READ",
                "ROCKE_ROLL_ARCH",
            ],
        )

    def test_documented_names_and_families(self):
        names, fams = pc.documented(
            "| `ROCKE_BACKEND` | x |\n| `ROCKE_EXP_*` | experiments |\n"
        )
        self.assertEqual((names, fams), ({"ROCKE_BACKEND"}, ["ROCKE_EXP_"]))

    def test_new_undocumented_reads_against_the_head_and_merge_base(self):
        with tempfile.TemporaryDirectory() as d:
            r = pathlib.Path(d)
            doc = r / K / pc.ENV_FLAGS
            doc.parent.mkdir(parents=True)
            doc.write_text("| `ROCKE_BACKEND` | x |\n| `ROCKE_EXP_*` | experiments |\n")
            src = r / K / "platform/python/rocke/run.py"
            src.parent.mkdir(parents=True)
            src.write_text('import os\nOLD = os.environ.get("ROCKE_OLD_UNDOC")\n')
            g(r, "init", "-q", "-b", "develop")
            g(r, "add", "-A")
            g(r, "commit", "-qm", "base")
            g(r, "checkout", "-qb", "pr")
            src.write_text(
                'import os\nOLD = os.environ.get("ROCKE_OLD_UNDOC")\nB = os.environ.get("ROCKE_BACKEND")\n'
                'E = os.environ.get("ROCKE_EXP_NEW")\nOLD2 = os.getenv("ROCKE_OLD_UNDOC")\n'
                'N = os.environ.get("ROCKE_ROLL_ARCH", "gfx950")\nD = os.environ.get("ROCKE_DOCUMENTED_NOW")\n'
            )
            doc.write_text(
                doc.read_text() + "| `ROCKE_DOCUMENTED_NOW` | the PR's own edit |\n"
            )
            g(r, "commit", "-qam", "pr")
            head = g(r, "rev-parse", "HEAD")
            a = types.SimpleNamespace(
                repo=d,
                head=head,
                base="develop",
                rocke_path=K,
                allow=pc.tracker_allow({}),
            )
            res = pc.run("envdocs", a)
            self.assertIsNone(res["error"])
            self.assertEqual(
                res["items"],
                [{"name": "ROCKE_ROLL_ARCH", "at": "platform/python/rocke/run.py:6"}],
            )
            self.assertEqual(
                pc.render(res)[-1],
                f"ENVDOCS fail 1 ROCKE_* variable read and not in {pc.ENV_FLAGS}: "
                "ROCKE_ROLL_ARCH",
            )


class TestTestLists(unittest.TestCase):
    # develop's platform/CMakeLists.txt collects two library directories whole and lists the rest by file.
    CMAKE = (
        "    set(_ROCKE_LIBRARY_ATTENTION_CPU_TEST_RELPATHS\n"
        '        "tests/library/tests/test_attention_builds.py"\n'
        '        "tests/library/tests/dispatch/test_grouped_conv_dispatch.py")\n'
        '           " \\"${ROCKE_TEST_CTEST_REL}tests/library/tests/dispatch/attention\\""\n'
        '           " \\"${ROCKE_TEST_CTEST_REL}tests/library/tests/sdpa\\""\n'
        'file(GLOB _gdn CONFIGURE_DEPENDS "${_lib}/tests/library/tests/gdn/*.py")\n'
    )

    def test_listed_by_name_or_by_a_directory(self):
        tests = [
            "library/tests/test_attention_combo_sweep_fp8.py",  # #22222 at 806727dc: in no list
            "library/tests/test_attention_builds.py",  # listed by name
            "library/tests/dispatch/attention/test_dense_auto_layout.py",  # its directory is collected
            "library/tests/sdpa/test_dense_sdpa_binding.py",
            "library/tests/gdn/test_gdn_x.py",  # a glob below its directory
            "library/tests/dispatch/test_new_dispatch.py",
        ]  # dispatch/ only prefixes another listed file
        self.assertEqual(
            [x["test"] for x in pc.unlisted(tests, self.CMAKE)],
            [
                "library/tests/test_attention_combo_sweep_fp8.py",
                "library/tests/dispatch/test_new_dispatch.py",
            ],
        )

    def test_only_new_test_files_under_library_tests(self):
        d = diff_of(
            {
                f"{K}/library/tests/test_new.py": "x",
                f"{K}/library/tests/_helper_utils.py": "x",
                f"{K}/library/tests/conftest.py": "x",
                f"{K}/platform/tests/test_platform.py": "x",
                f"{K}/library/kernels/test_like_name.py": "x",
            }
        )
        self.assertEqual(pc.new_library_tests(d, K), ["library/tests/test_new.py"])
        changed = diff_of({f"{K}/library/tests/test_old.py": "x"}, new=False)
        self.assertEqual(pc.new_library_tests(changed, K), [])

    def test_against_a_tree(self):
        with tempfile.TemporaryDirectory() as d:
            r = pathlib.Path(d)
            cm = r / K / "platform/CMakeLists.txt"
            cm.parent.mkdir(parents=True)
            cm.write_text(self.CMAKE)
            g(r, "init", "-q", "-b", "develop")
            g(r, "add", "-A")
            g(r, "commit", "-qm", "base")
            g(r, "checkout", "-qb", "pr")
            for name in (
                "test_attention_combo_sweep_fp8.py",
                "test_attention_builds.py",
            ):
                p = r / K / "library/tests" / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("def test_x():\n    pass\n")
            g(r, "add", "-A")
            g(r, "commit", "-qm", "pr")
            a = types.SimpleNamespace(
                repo=d, head=g(r, "rev-parse", "HEAD"), base="develop", rocke_path=K
            )
            res = pc.run("testlists", a)
            self.assertEqual(
                res["items"],
                [{"test": "library/tests/test_attention_combo_sweep_fp8.py"}],
            )
            self.assertEqual(
                pc.render(res)[-1],
                "TESTLISTS fail 1 new library test in no CMake or CTest list (no CI "
                "lane runs it): test_attention_combo_sweep_fp8.py",
            )
            # Listed by the PR itself: clean.
            cm.write_text(
                self.CMAKE
                + '        "tests/library/tests/test_attention_combo_sweep_fp8.py"\n'
            )
            g(r, "commit", "-qam", "list it")
            a.head = g(r, "rev-parse", "HEAD")
            res = pc.run("testlists", a)
            self.assertEqual((res["items"], pc.verdict(res)[0]), ([], "pass"))


class TestCppEnv(unittest.TestCase):
    def test_getenv_in_added_cpp_lines(self):
        d = diff_of(
            {
                # #77777 / #88888 (a peer reviewer's rule 3; we posted 🟡): selector geometry from the environment.
                f"{K}/platform/cpp/instances/common/attention_unified_selectors.cpp": 'const char* env = getenv(name);\nconst char* env = getenv("HIPDNN_GFX942_K_SLICE_HD");\n'
                '// TODO: consult getenv("HIPDNN_GFX942_FLASH_WIDE")\n * getenv("X") in a comment\n'
                'auto v = std::getenv("ROCKE_X");',
                f"{K}/platform/tests/cpp/test_env.cpp": 'getenv("ROCKE_TEST_ONLY");',
                "projects/hiptensor/library/src/rocke_runtime.cpp": "getenv(kPackPathVariable);",  # outside rocKE
                f"{K}/platform/python/rocke/x.py": 'os.getenv("ROCKE_PY")',
            }
        )  # rule_scan's
        self.assertEqual(
            [(x["at"], x["what"]) for x in pc.cppenv(d, K)],
            [
                (
                    "platform/cpp/instances/common/attention_unified_selectors.cpp:1",
                    "name",
                ),
                (
                    "platform/cpp/instances/common/attention_unified_selectors.cpp:2",
                    '"HIPDNN_GFX942_K_SLICE_HD"',
                ),
                (
                    "platform/cpp/instances/common/attention_unified_selectors.cpp:5",
                    '"ROCKE_X"',
                ),
            ],
        )


class TestTitle(unittest.TestCase):
    def test_the_six_titles_of_the_sweep(self):
        for title, why in (
            (
                " feat(rocke): TF32 compute for the conv kernels (XF32 gfx942, BF16x3 gfx950)",
                "it starts with a space",
            ),
            (
                " test(rocke): split pinned GPU references by operation, architecture, and variant",
                "it starts with a space",
            ),
            ("Enable fp64 GEMM on gfx942", "no `type(scope): ` prefix"),
            (
                "feat(hipTensor/rocKE): Add a new rocKE backend to hipTensor and add optimized kernel support",
                "the scope `(hipTensor/rocKE)` is not lower-case words",
            ),
            (
                "rocKE: check which tile the dense partial-tile refusal names",
                "`rocKE` is not a type",
            ),
        ):
            with self.subTest(title=title):
                self.assertTrue(
                    (pc.title_problem(title) or "").startswith(why),
                    pc.title_problem(title),
                )

    def test_conventional_titles_pass(self):
        for title in (
            "feat(hip-kernel-provider): wire rocKE smoke tests into provider CI",
            "fix: a bug",
            "refactor(rocke/wgrad)!: drop the old path",
            "test(rocke,hipdnn): cover gfx942",
            "feat(rocke): a title well over eighty characters is not a stated rule here, so it passes ok",
        ):
            self.assertIsNone(pc.title_problem(title), title)

    def test_title_from_the_staged_meta(self):
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write(
                "# PR #99999: rocKE: check which tile the dense partial-tile refusal names\n- Author: @x\n"
            )
        self.addCleanup(pathlib.Path(f.name).unlink)
        a = types.SimpleNamespace(title=None, meta=f.name)
        res = pc.run("title", a)
        self.assertEqual(
            pc.render(res)[0],
            "title: rocKE: check which tile the dense partial-tile refusal names",
        )
        self.assertTrue(
            pc.render(res)[-1].startswith(
                "TITLE fail not a Conventional Commits title: `rocKE` is not"
            )
        )


class TestThreads(unittest.TestCase):
    @staticmethod
    def thread(
        first: str,
        *others: str,
        resolved=False,
        outdated=False,
        typename="User",
        path=f"{K}/a.py",
        line=3,
    ):
        nodes = [
            {
                "author": {"login": first, "__typename": typename},
                "createdAt": "2026-11-05T17:11:00Z",
                "url": f"https://example.com/{first}",
            }
        ]
        nodes += [
            {"author": {"login": o}, "createdAt": f"2026-11-06T0{i}:00:00Z"}
            for i, o in enumerate(others)
        ]
        return {
            "isResolved": resolved,
            "isOutdated": outdated,
            "path": path,
            "line": line,
            "originalLine": line,
            "comments": {"nodes": nodes},
        }

    def test_live_threads_no_owner_answered(self):
        data = {
            "author": "author1",
            "commits": [{"login": "author1"}, {"login": "coauthor"}],
            "threads": [
                # #23456's Copilot thread at test_advisory_checks.py:429 (no reply): live and unanswered.
                self.thread(
                    "copilot-pull-request-reviewer",
                    path=f"{K}/library/tests/test_advisory_checks.py",
                    line=429,
                ),
                self.thread(
                    "reviewer1", "reviewer2"
                ),  # another reviewer's reply is no answer
                self.thread("reviewer1", "author1"),  # the author answered
                self.thread("reviewer1", "coauthor"),  # a commit author answered
                self.thread("reviewer1", resolved=True),
                self.thread("reviewer1", outdated=True),
                self.thread("author1"),  # the author's own note
                self.thread("github-actions", typename="Bot"),
            ],
        }  # a bot that is not Copilot
        got = pc.unanswered(data, K)
        self.assertEqual(
            [(t["reviewer"], t["where"], t["replies"]) for t in got],
            [
                (
                    "copilot-pull-request-reviewer",
                    "library/tests/test_advisory_checks.py:429",
                    0,
                ),
                ("reviewer1", "a.py:3", 1),
            ],
        )
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(data, f)
        self.addCleanup(pathlib.Path(f.name).unlink)
        res = pc.run(
            "threads", types.SimpleNamespace(threads_json=f.name, rocke_path=K)
        )
        self.assertEqual(
            pc.render(res)[-1],
            "THREADS fail 2 live threads with no owner reply: "
            "copilot-pull-request-reviewer 1, reviewer1 1",
        )
        data["threads"] = data["threads"][2:]
        pathlib.Path(f.name).write_text(json.dumps(data))
        res = pc.run(
            "threads", types.SimpleNamespace(threads_json=f.name, rocke_path=K)
        )
        self.assertEqual(
            pc.verdict(res),
            ("pass", "no live thread waits for an owner's reply (4 live)"),
        )

    def test_no_thread_data_is_not_run(self):
        res = pc.run(
            "threads",
            types.SimpleNamespace(threads_json=None, slug=None, pr=None, rocke_path=K),
        )
        self.assertEqual(pc.verdict(res)[0], "skip")


class TestLeads(unittest.TestCase):
    def test_leads(self):
        d = diff_of(
            {
                # #45678 (Copilot, confirmed): a request flag forced to bool.
                f"{K}/library/dispatch/attention/bindings.py": "if bool(request.use_attn_bias) != problem.use_additive_bias:\n"
                "batch = int(request.batch)\n"  # a typed field: not a lead
                'or torch.cuda.current_device() != call["q"].device.index',  # #56789 (the peer-review rules §E.2)
                f"{K}/library/tests/test_x.py": "bool(req.flag); torch.cuda.is_available()",  # tests may
                f"{K}/platform/python/rocke/examples/sweep.py": "torch.cuda.is_available()",  # examples may
                # #67890 (two lines, no CONFIGURE_DEPENDS), and the forms that have it.
                f"{K}/library/cpp/CMakeLists.txt": 'file(GLOB _rocke_library_headers RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/include/rocke"\n'
                '     "${CMAKE_CURRENT_SOURCE_DIR}/include/rocke/*")\n'
                'file(GLOB _flydsl_pointers RELATIVE "${FLYDSL_CONTENT_DIR}"\n'
                '     CONFIGURE_DEPENDS "${FLYDSL_CONTENT_DIR}/*/gfx*.dvc")\n'
                'file(GLOB _legacy ${_configure_depends} "${_bundles}/*/*.tar.gz.dvc")',
                "projects/hipdnn/tools/CMakeLists.txt": 'file(GLOB_RECURSE overlayFiles "${D}/*")',
            }
        )
        self.assertEqual(
            sorted((x["kind"], x["at"]) for x in pc.peerleads(d, K)),
            [
                ("coercion", "library/dispatch/attention/bindings.py:1"),
                (
                    "file(GLOB) without CONFIGURE_DEPENDS",
                    "library/cpp/CMakeLists.txt:1",
                ),
                (
                    "file(GLOB) without CONFIGURE_DEPENDS",
                    "projects/hipdnn/tools/CMakeLists.txt:1",
                ),
                ("torch device query", "library/dispatch/attention/bindings.py:3"),
            ],
        )
        res = {"check": "peerleads", "error": None, "items": pc.peerleads(d, K)}
        self.assertEqual(
            pc.render(res)[-1],
            "PEERLEADS fail 4 leads: file(GLOB) without CONFIGURE_DEPENDS 2, "
            "coercion 1, torch device query 1",
        )


def pr_repo(
    d: str,
    base: dict[str, str],
    head: dict[str, str | None],
    moves: dict[str, str] | None = None,
) -> str:
    """A git repo at `d` whose develop holds `base` and whose pr branch moves `moves` (old -> new) then writes `head`
    (None deletes); the pr head's SHA."""
    r = pathlib.Path(d)
    for path, text in base.items():
        (r / path).parent.mkdir(parents=True, exist_ok=True)
        (r / path).write_text(text)
    g(r, "init", "-q", "-b", "develop")
    g(r, "add", "-A")
    g(r, "commit", "-qm", "base", "--allow-empty")
    g(r, "checkout", "-qb", "pr")
    for old, new in (moves or {}).items():
        (r / new).parent.mkdir(parents=True, exist_ok=True)
        g(r, "mv", old, new)
    for path, text in head.items():
        if text is None:
            (r / path).unlink()
            continue
        (r / path).parent.mkdir(parents=True, exist_ok=True)
        (r / path).write_text(text)
    g(r, "add", "-A")
    g(r, "commit", "-qm", "pr")
    return g(r, "rev-parse", "HEAD")


def run_on(check: str, base: dict, head: dict, moves: dict | None = None) -> dict:
    """pc.run(check) on a PR repo built by pr_repo."""
    with tempfile.TemporaryDirectory() as d:
        sha = pr_repo(d, base, head, moves)
        return pc.run(
            check, types.SimpleNamespace(repo=d, head=sha, base="develop", rocke_path=K)
        )


class TestParseDiffRemoved(unittest.TestCase):
    def test_removed_lines_and_old_paths(self):
        text = (
            "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -4,2 +4,1 @@\n-old one\n--- old two\n+new\n"
            "diff --git a/a/old.cpp b/b/new.cpp\nsimilarity index 90%\nrename from a/old.cpp\nrename to b/new.cpp\n"
            "--- a/a/old.cpp\n+++ b/b/new.cpp\n@@ -9,0 +10 @@\n+++i;\n"
            "diff --git a/n.md b/n.md\nnew file mode 100644\n--- /dev/null\n+++ b/n.md\n@@ -0,0 +1 @@\n+# t\n"
        )
        d = pc.parse_diff(text, removed=True)
        # A removed line that starts "-- " and an added one that starts "++" stay lines, not headers.
        self.assertEqual(
            d["x.py"],
            {
                "new": False,
                "lines": [(4, "new")],
                "old": "x.py",
                "removed": [(4, "old one"), (5, "-- old two")],
            },
        )
        self.assertEqual(
            d["b/new.cpp"],
            {"new": False, "lines": [(10, "++i;")], "old": "a/old.cpp", "removed": []},
        )
        self.assertEqual(d["n.md"]["old"], None)
        self.assertEqual(
            pc.parse_diff(text)["x.py"], {"new": False, "lines": [(4, "new")]}
        )


class TestSplitTautology(unittest.TestCase):
    def test_a_literal_that_holds_the_separator(self):
        d = diff_of(
            {
                # #98765 at a1b2c3d4 and #34567 at a2f4d565 (Copilot, fixed): the assertion always passes.
                f"{K}/library/tests/test_attention_tiled_launch_signature.py": 'assert "_dg" not in kernel.name.split("_"), kernel.name\n'
                'assert "dg" not in kernel.name.split("_"), kernel.name\n'  # the fix: a real piece
                'ok = "a b" in line.split()\nok = "ab" in line.split()\n'
                '# assert "_dg" not in name.split("_")'
            }
        )
        got = [(x["at"], x["what"]) for x in pc.splittaut(d, K)]
        self.assertEqual(
            [a for a, _ in got],
            [
                "library/tests/test_attention_tiled_launch_signature.py:1",
                "library/tests/test_attention_tiled_launch_signature.py:3",
            ],
        )
        self.assertIn(
            "is always true: no piece of .split('_') can hold '_dg'", got[0][1]
        )
        self.assertIn("is always false", got[1][1])


class TestSiblingImports(unittest.TestCase):
    BARRIER = "def _spec():\n    return 1\n"

    def test_bare_sibling_imports(self):
        res = run_on(
            "sibimports",
            {
                f"{K}/platform/tests/instances/test_gemm_split_barrier.py": self.BARRIER,
                f"{K}/library/tests/conftest.py": "import sys\nfrom pathlib import Path\n_TESTROOT = Path(__file__).resolve()"
                ".parent\nsys.path.append(str(_TESTROOT))\n",
                f"{K}/library/tests/test_gfx942_gqa_head_fold_numeric.py": self.BARRIER,
            },
            {
                # #87654 at dfd02ac4 (Copilot, fixed by a sys.path insert): no insert, no conftest beside it.
                f"{K}/platform/tests/instances/test_gemm_tdm_early_refill.py": "from test_gemm_split_barrier import _spec\n",
                # The fix: the file inserts its own directory.
                f"{K}/platform/tests/instances/test_gemm_tdm_padding.py": "import os\nimport sys\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
                "from test_gemm_split_barrier import _spec  # noqa: E402\n",
                # #76543: library/tests/conftest.py appends its own directory, so the import resolves.
                f"{K}/library/tests/test_attention_kv_tail_numeric.py": "from test_gfx942_gqa_head_fold_numeric import _spec\n",
                # Not a sibling (nothing of that name beside it).
                f"{K}/platform/tests/instances/test_other.py": "import test_portable_ir\n",
            },
        )
        self.assertEqual(
            [x["at"] for x in res["items"]],
            ["platform/tests/instances/test_gemm_tdm_early_refill.py:1"],
        )
        self.assertEqual(
            pc.render(res)[-1],
            "SIBIMPORTS fail 1 bare sibling test import: "
            "platform/tests/instances/test_gemm_tdm_early_refill.py:1",
        )


class TestWorkload(unittest.TestCase):
    def test_sizes_in_kernel_names(self):
        d = diff_of(
            {
                # #65432 at 94475ba5 (the peer-review rules §C.3, three rounds): golden names carry the test's M, N and K.
                f"{K}/platform/tests/golden/rocke_representative_ir_sha256/gemm.json": '"kernel_name": "irhash_nvfp4_bf16_k128_nvfp4_bf16_block_scaled_fp4_fp4_out_bf16_M32N48K128_bk32",\n'
                '"sha256": "M32N48K128 is not a name line in a golden file either way"',
                f"{K}/platform/python/rocke/instances/common/gemm_block_scaled.py": 'return f"{self.op}_M{self.M}N{self.N}K{self.K}_bk{self.bk}"  # kernel_name\n'
                'shape = "M32N48K128"\n'  # not a name
                'name = f"mfma_{t}_m16n8k16"',
            }
        )  # an instruction shape, not sizes
        self.assertEqual(
            [x["at"] for x in pc.workload(d, K)],
            [
                "platform/python/rocke/instances/common/gemm_block_scaled.py:1",
                "platform/tests/golden/rocke_representative_ir_sha256/gemm.json:1",
                "platform/tests/golden/rocke_representative_ir_sha256/gemm.json:2",
            ],
        )


class TestPrNumbers(unittest.TestCase):
    def test_code_and_docs(self):
        d = diff_of(
            {
                # #54321 (a benchmark comment), #77777 (a test docstring), #43210 at 1be83134 (the peer-review rules §E.8, a C comment).
                f"{K}/library/benchmarks/gfx942/attention/prefill/bench_rocke_7way_specs.py": "# which _use_cfvst resolves ON at BF16 D128 (#1111); dispatch canonicalizes a\n"
                "Primary motivation: PR #2222 changed routing without a test",
                f"{K}/platform/cpp/core/arch/data.cpp": "/* MFMA 16x16x4 fp32 / f64 A operand (#3333): one scalar per lane.",
                # #4444 at 8e02bb69 (a peer reviewer) and #56789 at 13ae4343 (a peer reviewer): docs.
                f"{K}/platform/dsl_docs/development/known_gaps.md": "and the C engine is expected to be deprecated (ROCm/rocm-libraries#56789).",
                f"{K}/library/builders/gfx950/attention/CASE_STUDY_DECODE_HOST_OVERHEAD.md": "PR #5555.\n"
                "see https://github.com/ROCm/example/pull/5432 for the series",
                # Not PR references: a data file, a tracker ID (the trackers check's), a colour, a hex literal, an include.
                f"{K}/library/data/cases.json": '{"pr": "#1111"}',
                f"{K}/platform/python/rocke/x.py": "# ABCD"
                "-1932\nCOLOR = '#1a2b3c'\nMASK = 0x1234\nn = 12345  # issue #12",
                f"{K}/platform/cpp/x.cpp": "#include <array>\n#define N 12345",
            }
        )
        got = [(x["at"], x["what"].split("`")[1]) for x in pc.prnumbers(d, K)]
        self.assertEqual(
            got,
            [
                (
                    "library/benchmarks/gfx942/attention/prefill/bench_rocke_7way_specs.py:1",
                    "#1111",
                ),
                (
                    "library/benchmarks/gfx942/attention/prefill/bench_rocke_7way_specs.py:2",
                    "PR #2222",
                ),
                (
                    "library/builders/gfx950/attention/CASE_STUDY_DECODE_HOST_OVERHEAD.md:1",
                    "PR #5555",
                ),
                (
                    "library/builders/gfx950/attention/CASE_STUDY_DECODE_HOST_OVERHEAD.md:2",
                    "/pull/5432",
                ),
                ("platform/cpp/core/arch/data.cpp:1", "#3333"),
                (
                    "platform/dsl_docs/development/known_gaps.md:1",
                    "rocm-libraries#56789",
                ),
            ],
        )


class TestPerfNotes(unittest.TestCase):
    def test_claims_and_parameters(self):
        for text in (
            "# on Qwen3-8B prefill -- neutral at S=1024, ~3.5% faster than Triton at S=8192,",  # #6666
            "# PATHOLOGICAL at long sequences -- 3-8x slower at L16384 while still fine at",  # #6666
            "It takes ~95 minutes, so run it detached. Drop ``--ctx-sweep`` for the four",  # #6666
            "# (~116 GB"
            "/s vs the ~220 GB"
            "/s ceiling at 256 CTAs), because a single wave",  # #6666
            "# One compile per option; an empty kernel takes well under a second.",  # #7777 (Copilot)
            "The cost is that each shard re-reads the kernel cache: ~18 s for a 447k-entry",  # #8888 (Copilot)
            "At a few hundred microseconds each that is the single largest fixed cost",  # #8888 (Copilot)
            "alone. Costs about 0.1s per call.",
        ):  # #87654 (a peer reviewer)
            with self.subTest(text=text):
                self.assertIsNotNone(pc.perf_claim(text))
        for text in (
            "* **peak**: before every sample the GPU idles ``--gap-ms`` (default 25"
            " ms),",  # parameters (#87654)
            "* **sustained**: 2 s idle, then 20 warmup and 400 timed launches back to back,",
            "# is one test, and the test bounds its emitter run below the 120 s",  # #9999
            "# A timed row's ``us`` is below this (1000 s): anything larger is not a kernel",  # #2345
            "#   = x * sigmoid(2 s (x + c x^3))",  # math (#66666)
            "``4s..4s+3`` of every block's ``k x 4`` operand, block ``b`` at rows",
            "# BEWARE: sustained runs power-throttle this part hard within a few seconds",  # hardware, vague
            "# tiled arch, so each sweep case runs for tens of seconds: it gets its own label",  # a test's runtime
            "# a 2x unroll of the K loop; 16 GB of HBM",
        ):
            with self.subTest(text=text):
                self.assertIsNone(pc.perf_claim(text))

    def test_comments_and_docstrings_only(self):
        res = run_on(
            "perfnotes",
            {},
            {
                f"{K}/library/benchmarks/common/run_conv_shards.py": '"""Shards.\n\nThe cost is that each shard re-reads the kernel cache: ~18 s for a 447k-entry\n'
                'cache, so ~72 s per card over a run.\n"""\n'
                'print(f"  config={config:11s} kernel={kernel}")\n'  # an f-string width is no timing
                'MSG = "takes ~3'
                ' ms"\n',  # a string in code is not a comment
                f"{K}/platform/tools/gen_compiler_options.py": "x = 1  # One compile per option; an empty kernel takes well under a second.\n",
                # #8888 at 45484de0 (Copilot): a claim wrapped over two docstring lines, reported on the second.
                f"{K}/library/benchmarks/common/module_cache.py": '"""One module per binary.\n\ncandidates loaded *C x K* modules of *K* distinct binaries. At a few '
                'hundred\nmicroseconds each that is the single largest fixed cost per candidate in a\n"""\n',
                f"{K}/library/tests/test_speed.py": "# ~3.5% faster than Triton\n",
            },
        )  # tests may
        self.assertEqual(
            [x["at"] for x in res["items"]],
            [
                "library/benchmarks/common/module_cache.py:4",
                "library/benchmarks/common/run_conv_shards.py:3",
                "library/benchmarks/common/run_conv_shards.py:4",
                "platform/tools/gen_compiler_options.py:1",
            ],
        )


class TestBackEdges(unittest.TestCase):
    def test_new_back_edges_against_the_merge_base(self):
        res = run_on(
            "backedges",
            {
                f"{K}/platform/tests/CMakeLists.txt": "project(t)\n",
                f"{K}/platform/tests/cmake/Old.cmake": 'set(X "${CMAKE_CURRENT_SOURCE_DIR}/../../library/tests")\n',
                f"{K}/platform/cpp/bindings/prove_parity_binding.py": "import os\n",
            },
            {
                # #3456 at 6eea863a (a peer reviewer): a platform test reads a library data file.
                f"{K}/platform/tests/CMakeLists.txt": 'project(t)\nset(V "${CMAKE_CURRENT_SOURCE_DIR}/../../library/tests/golden/vectors.json")\n'
                '  "tests/library/tests/test_attention_builds.py"\n'  # the install layout: not a back-edge
                "# see ../../library/README.md\n",  # a comment
                # library/tests was reached at the merge base (develop's edge); library/kernels is new.
                f"{K}/platform/tests/cmake/Old.cmake": 'set(X "${CMAKE_CURRENT_SOURCE_DIR}/../../library/tests")\nset(W "${CMAKE_CURRENT_SOURCE_DIR}/../../'
                'library/tests/more")\nset(Y "${CMAKE_CURRENT_SOURCE_DIR}/../../library/kernels")\n',
                # #4567 at b63ce8e3 (Copilot): a sys.path insert of library/tests through os.path.join.
                f"{K}/platform/cpp/bindings/prove_parity_binding.py": 'import os\nparity = os.path.join(here, "..", "..", "..", "library", "tests", "parity")\n'
                # #67890 (a peer reviewer): the platform binding imports the library's engine module.
                "import rocke_library_engine  # noqa: E402\n",
                f"{K}/library/x.py": "from kernels.common import y\n",
            },
        )  # library may
        self.assertEqual(
            [(x["at"], x["what"].split(" (")[0]) for x in res["items"]],
            [
                (
                    "platform/cpp/bindings/prove_parity_binding.py:2",
                    "reaches library/tests",
                ),
                (
                    "platform/cpp/bindings/prove_parity_binding.py:3",
                    "reaches rocke_library_engine",
                ),
                ("platform/tests/CMakeLists.txt:2", "reaches library/tests"),
                ("platform/tests/cmake/Old.cmake:3", "reaches library/kernels"),
            ],
        )


class TestAbiVersion(unittest.TestCase):
    BASE = (
        "from kernels.common.attention_unified import (\n    _3d_signature,\n)\n\n\ndef make_candidate():\n"
        '    return KernelCandidate(\n        abi_version="rocke-attention-strided-kv/v1",\n'
        "        signature=lambda spec: _3d_signature(spec.kernel_spec.dtype, strided_kv=True),\n    )\n"
    )
    PATH = f"{K}/library/dispatch/attention/strided_decode.py"

    def test_signature_changed_and_string_kept(self):
        # #5678 at 45933cea (Copilot, fixed by a bump).
        head = self.BASE.replace("_3d_signature", "_tiled_3d_signatures").replace(
            "spec.kernel_spec.dtype, strided_kv=True)",
            "spec.arch, spec.kernel_spec.dtype, strided_kv=True)[0]",
        )
        res = run_on("abiversion", {self.PATH: self.BASE}, {self.PATH: head})
        self.assertEqual(
            [x["at"] for x in res["items"]],
            ["library/dispatch/attention/strided_decode.py:9"],
        )
        self.assertIn("and abi_version kept", res["items"][0]["what"])
        # The fix bumps the string: clean.
        res = run_on(
            "abiversion",
            {self.PATH: self.BASE},
            {self.PATH: head.replace("/v1", "/v2")},
        )
        self.assertEqual((res["items"], pc.verdict(res)[0]), ([], "pass"))

    def test_comments_and_docstrings_are_not_kernarg_changes(self):
        # #6789 and #7890: comments that mention kernargs and a helper whose name has "kernarg" in it.
        head = self.BASE.replace(
            "def make_candidate():\n",
            "def make_candidate():\n    # the dgrad-shaped kernargs\n"
            "    d = direct_dgrad_kernarg_direction(a)\n",
        )
        res = run_on("abiversion", {self.PATH: self.BASE}, {self.PATH: head})
        self.assertEqual(res["items"], [])


class TestArrayWrites(unittest.TestCase):
    @staticmethod
    def src(writes: int, slots: int = 12, guard: str = "") -> str:
        body = "".join(
            f'  {guard}flag_names[n_flags] = "f{i}"; flag_on[n_flags++] = on{i};\n'
            for i in range(writes)
        )
        return (
            f"static int describe(const spec_t* s) {{\n  const char* flag_names[{slots}];\n  int flag_on[{slots}];\n"
            f"  int n_flags = 0;\n{body}  return n_flags;\n}}\n"
        )

    def test_more_write_sites_than_slots_against_the_moved_file(self):
        old, new = f"{K}/library/cpp/conv_wgrad.cpp", f"{K}/platform/cpp/conv_wgrad.cpp"
        # #9876 at 9472c456 (Copilot): a 14th flag into the 12-slot arrays (13 sites at the merge base).
        res = run_on(
            "arraywrites", {old: self.src(13)}, {new: self.src(14)}, moves={old: new}
        )
        self.assertEqual(
            [x["what"] for x in res["items"]],
            [
                "flag_names[12] is written at 14 sites through n_flags (13 sites into [12] at the merge base), with no "
                "bound check",
                "flag_on[12] is written at 14 sites through n_flags (13 sites into [12] at the merge "
                "base), with no bound check",
            ],
        )
        # #67890: the move alone (13 sites before and after) adds no write site.
        res = run_on(
            "arraywrites",
            {old: self.src(13)},
            {new: self.src(13) + "\n// moved\n"},
            moves={old: new},
        )
        self.assertEqual(res["items"], [])
        # A bounded index is no overflow.
        res = run_on("arraywrites", {}, {new: self.src(14, guard="if (n_flags < 12) ")})
        self.assertEqual(res["items"], [])


class TestNanBlind(unittest.TestCase):
    def test_comparisons_nan_passes(self):
        res = run_on(
            "nanblind",
            {},
            {
                # #8765 at fa89a824 (Copilot, fixed).
                f"{K}/library/tests/test_direct_conv_correctness.py": "def check(out, ref, tol):\n    rel_err = abs(out - ref).max() / abs(ref).max()\n"
                "    if rel_err >= tol:\n        raise AssertionError(rel_err)\n",
                f"{K}/library/tests/test_safe.py": "def check(out, ref, tol):\n    err = abs(out - ref)\n    if not np.isfinite(err).all():\n"
                "        raise AssertionError\n    if err.max() >= tol:\n        raise AssertionError\n",
            },
        )
        self.assertEqual(
            [x["at"] for x in res["items"]],
            ["library/tests/test_direct_conv_correctness.py:3"],
        )


class TestOptionalImports(unittest.TestCase):
    def test_module_level_imports_without_importorskip(self):
        res = run_on(
            "optimports",
            {},
            {
                # #7654 at 36d3d16e (Copilot, fixed): collection fails in the shared GPU CTest.
                f"{K}/library/tests/test_attn_tiled_2d_gfx950_fp8_k_lds_numeric.py": "import numpy as np\nimport ml_dtypes\n",
                f"{K}/library/tests/test_guarded.py": 'import pytest\nml_dtypes = pytest.importorskip("ml_dtypes")\n'
                "import ml_dtypes  # noqa\n",
                f"{K}/library/tests/test_try.py": "try:\n    import scipy\nexcept ImportError:\n    scipy = None\n",
                f"{K}/library/tests/test_local.py": "def test_x():\n    import scipy\n",
                f"{K}/library/kernels/x.py": "import scipy\n",
            },
        )  # not a test
        self.assertEqual(
            [x["at"] for x in res["items"]],
            ["library/tests/test_attn_tiled_2d_gfx950_fp8_k_lds_numeric.py:2"],
        )


class TestMdHeadings(unittest.TestCase):
    def test_a_repeated_number_the_pr_adds(self):
        base = (
            "# Gaps\n\n## 4 Old\n\n## 5 MLA prefill\n\n### 2.1 Twice\n\n### 2.1 Twice\n"
        )
        # #6543 at 047d7fde (Copilot): a new section reuses §5.
        head = base + "\n## 5 New section\n\n```sh\n# 5 steps in a shell comment\n```\n"
        res = run_on(
            "mdheadings",
            {f"{K}/platform/dsl_docs/development/known_gaps.md": base},
            {f"{K}/platform/dsl_docs/development/known_gaps.md": head},
        )
        self.assertEqual(
            [(x["at"], x["what"]) for x in res["items"]],
            [
                (
                    "platform/dsl_docs/development/known_gaps.md:11",
                    "heading `## 5` appears 2 times (lines 5, 11)",
                )
            ],
        )


class TestNewLeads(unittest.TestCase):
    def test_lowerer_change_with_no_parity_pair(self):
        low = {
            f"{K}/platform/cpp/core/lower_hip/lower_hip_mma.cpp": "x",
            f"{K}/platform/python/rocke/core/lower_hip.py": "x",
        }
        self.assertEqual(
            [x["at"] for x in pc.paritypair(diff_of(low, new=False), K)],
            ["platform/cpp/core/lower_hip/lower_hip_mma.cpp"],
        )
        low[f"{K}/library/tests/parity/barrier_fence_emit.py"] = "x"
        self.assertEqual(pc.paritypair(diff_of(low, new=False), K), [])

    def test_unused_unpack_targets(self):
        # #43210 at 3dcf594e (the peer-review rules §E.6).
        res = run_on(
            "unpack",
            {},
            {
                f"{K}/platform/python/rocke/core/lower_llvm.py": "def _op(op):\n    a, b, c = op.operands\n    _x, y = op.pair\n    return c, y\n"
            },
        )
        self.assertEqual(
            [x["what"] for x in res["items"]], ["a, b never read in _op()"]
        )
        self.assertEqual(pc.verdict(res)[0], "fail")


class TestWiring(unittest.TestCase):
    CFG = {
        "rocke_path": "rk",
        "repo": "o/r",
        "paths": {"working_dir": "/w"},
    }
    PLAN = {
        "families": [],
        "tests": [],
        "build": False,
        "golden": False,
        "engine_wide": False,
        "outside_rocke": [],
        "libparity": False,
        "_dirs": {},
    }

    def test_each_check_is_its_own_section_with_a_status_line(self):
        script = review.checks_script(
            dict(self.CFG, tracker_allow=["SEM", "IEEE"]),
            7,
            "a" * 40,
            "follow-up",
            self.PLAN,
            None,
        )
        for check, (tag, key, title, _lead) in pc.CHECKS.items():
            with self.subTest(check=check):
                sec = script.split(f"sec '{title}'", 1)[1].split("\nsec ", 1)[0]
                self.assertIn(
                    f"tools/peer_checks.py {check} --repo $R0 --head {'a' * 40} --base $BREF --rocke-path $K",
                    sec,
                )
                self.assertIn(f"grep '^{tag} ' \"$OUT/peer_{check}.txt\"", sec)
                self.assertIn(f"st {key} ", sec)
        self.assertIn(
            "peer_checks.py title --repo $R0 --head "
            + "a" * 40
            + " --base $BREF --rocke-path $K --meta "
            "$WK/pr$N/pr/meta.md",
            script,
        )
        self.assertIn("--slug o/r --pr $N", script)
        self.assertIn("--allow SEM,IEEE", script)
        abi = script.split(f"sec '{review.ABI_SECTION}'", 1)[1].split("\nsec ", 1)[0]
        self.assertIn(
            "tools/abi_scan.py $R0 $K "
            + "a" * 40
            + ' $BREF $AM > "$OUT/abi_report.txt"',
            abi,
        )
        self.assertIn("st abi ", abi)
        gate = review.checks_script(
            self.CFG, 7, "a" * 40, "follow-up", self.PLAN, None, mode="gate"
        )
        self.assertNotIn("peer_checks.py", gate)

    def test_flags_and_leads(self):
        for key in ("peerleads", "paritypair", "unpack"):
            self.assertIn(key, review.LEAD_SECTIONS)
        for key in (
            "threads",
            "trackers",
            "envdocs",
            "testlists",
            "cppenv",
            "title",
            "abi",
            "splittaut",
            "sibimports",
            "workload",
            "prnumbers",
            "perfnotes",
            "backedges",
            "abiversion",
            "arraywrites",
            "nanblind",
            "optimports",
            "mdheadings",
        ):
            self.assertNotIn(key, review.LEAD_SECTIONS)

    def test_titles_fit_in_single_quotes(self):
        # The checks script prints each title as `sec '<title>'`.
        for check, (_tag, _key, title, _lead) in pc.CHECKS.items():
            self.assertNotIn("'", title, check)

    def test_the_cli_prints_the_section_then_its_status_line(self):
        out = subprocess.run(
            [
                sys.executable,
                "-I",
                str(TOOL),
                "title",
                "--title",
                "Enable fp64 GEMM on gfx942",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        self.assertEqual(out[0], "title: Enable fp64 GEMM on gfx942")
        self.assertTrue(
            out[-1].startswith(
                "TITLE fail not a Conventional Commits title: no `type(scope): ` prefix"
            )
        )


if __name__ == "__main__":
    unittest.main()
