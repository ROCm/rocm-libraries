# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The admin dashboard's "Not sent" card: the unsent items, newest first, each with its content, escaped;
an unposted review's item links to its file, and the team dashboard's page is one item; while the team dashboard does
not send, its page is linked in either delivery. It is the admin dashboard's only, shown while an item is kept or one
may be. The admin dashboard's service serves an item's file, and that page, with the access token: a review as text,
the page in a sandbox, and no other file."""

import base64
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import post, report, state_store, webdash, webserve  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

PORT = 18613
TOKEN = (
    "invented-" + "access-token-for-the-list"
)  # an invented value, built at run time
LIVE = {"mode": "live"}
T1, T2, T3 = "2026-11-20T07:00:00Z", "2026-11-20T08:00:00Z", "2026-11-20T09:00:00Z"


class Base(unittest.TestCase):
    """A monitor of the test's own: its state folder, its outputs folder (the working folder), its access token."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = pathlib.Path(tmp.name)
        env = mock.patch.dict(os.environ, {"RKMON_HOME": str(self.tmp / "home")})
        env.start()
        self.addCleanup(env.stop)
        self.store = Store(self.tmp / "state")
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        self.store.save(st)
        token = self.tmp / "admin_access_token"
        token.write_text(TOKEN + "\n", encoding="utf-8")
        token.chmod(0o600)
        self.cfg = {
            "repo": "o/r",
            "operator": {"github_login": "op"},
            "codeowner_teams": {"platform": ["plat"]},
            "features": {"review_automation": {"mode": "prototype"}},
            "paths": {
                "working_dir": str(self.tmp),
                "html": str(self.tmp / "x.html"),
                "daily_dir": str(self.tmp),
                "root": str(self.tmp),
            },
            "admin_dashboard": {"port": PORT, "access_token_file": str(token)},
        }

    def keep(self, kind, for_, content="", file="", at="", **kw):
        """An unsent item, kept as the outward gate keeps it (post.unsent_item, then its feature and why)."""
        item = dict(
            post.unsent_item(kind, for_, content, file),
            feature=kw.get("feature", "review_automation"),
            why=kw.get("why", state_store.PROTOTYPE),
            **({"at": at} if at else {}),
        )
        state_store.keep_unsent(self.store.dir, item)

    def review(self, n: int, folder: pathlib.Path | None = None) -> pathlib.Path:
        """A review kept unposted, as save_draft keeps it: a file in the outputs folder (or FOLDER)."""
        folder = folder or post.posted_dir(self.cfg)
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"PR_UNPOSTED_{n}.md"
        p.write_text(f"Automated review of #{n}\n<b>kept</b>\n", encoding="utf-8")
        return p

    def made_page(self) -> pathlib.Path:
        """The team dashboard's page as a tick makes it, the published copy in the state folder; no item kept."""
        pub = self.store.dir / webdash.PUBLISHED_FILE
        pub.write_text("<p>the team's page</p><script>x()</script>", encoding="utf-8")
        return pub

    def page_item(self, at: str = "") -> pathlib.Path:
        """The team dashboard's page, kept as its one item as the page push's gate keeps it (publish.held)."""
        pub = self.made_page()
        self.keep(
            state_store.PAGE,
            "the team dashboard's web host",
            file=str(pub),
            at=at,
            feature="dashboards.team",
        )
        return pub

    def team(self, delivery: str, mode: str = "prototype") -> dict:
        """The configuration with the team dashboard on, in MODE, its delivery DELIVERY ("publish" or "serve")."""
        features = dict(
            self.cfg["features"], dashboards={"team": {"enabled": True, "mode": mode}}
        )
        return dict(self.cfg, features=features, team_dashboard={"mode": delivery})


class TestCard(Base):
    def render(self, variant=webdash.LOCAL, cfg=None) -> str:
        cfg = cfg or self.cfg
        st = self.store.load()
        board = report.board(cfg, st, self.store, [], [])
        return webdash.render(cfg, st, self.store, [], [], board, variant)

    def card(self, page: str) -> str:
        at = page.index("<section id='not-sent'")
        return page[at : page.index("</section>", at)]

    def test_newest_first_each_with_its_content_escaped(self):
        self.keep(
            state_store.COMMENT, "pull request 7", "<script>x()</script> & y", at=T1
        )
        self.keep(state_store.LABEL, "issue 3", "do not merge", at=T2)
        self.keep(
            state_store.ALERT,
            "the Teams channel <ops>",
            "**develop is red**",
            at=T3,
            feature="alerts",
            why=state_store.STANDBY,
        )
        page = self.render()
        card = self.card(page)
        self.assertIn(
            "<h2>Not sent <span class='c'>3 kept · never sent later</span></h2>", card
        )
        self.assertIn("<a href='#not-sent'>Not sent <span class='c'>3</span></a>", page)
        rows = card.split("<tr><td")[1:]
        self.assertEqual(len(rows), 3)
        self.assertIn("the Teams channel &lt;ops&gt;", rows[0])
        # Standby is the whole monitor's, never a feature's mode.
        self.assertIn("<td>the monitor in standby</td>", rows[0])
        self.assertNotIn("alerts in standby", card)
        self.assertIn("<pre class='unsent'>**develop is red**</pre>", rows[0])
        self.assertIn("issue 3", rows[1])
        self.assertIn("pull request 7", rows[2])
        self.assertIn("<td>review automation in prototype</td>", rows[2])
        self.assertIn(
            "<pre class='unsent'>&lt;script&gt;x()&lt;/script&gt; &amp; y</pre>",
            rows[2],
        )
        self.assertNotIn("<script>x()", page)
        self.assertNotIn("<ops>", page)

    def test_an_unposted_reviews_item_links_to_its_file(self):
        self.keep(state_store.REVIEW, "pull request 12", file=str(self.review(12)))
        # A file outside the outputs folder is named, never linked.
        elsewhere = self.review(13, self.tmp / "elsewhere")
        self.keep(state_store.REVIEW, "pull request 13", file=str(elsewhere))
        card = self.card(self.render())
        self.assertIn(
            f"<a href='http://127.0.0.1:{PORT}/unsent?file=PR_UNPOSTED_12.md' target='_blank' "
            "rel='noopener noreferrer'>PR_UNPOSTED_12.md</a>",
            card,
        )
        self.assertIn("PR_UNPOSTED_13.md", card)
        self.assertNotIn("unsent?file=PR_UNPOSTED_13.md", card)
        self.assertNotIn("<pre", card)  # a review's content is its file

    def test_a_link_in_the_outputs_folder_is_named_never_linked(self):
        """An item's file that is a link, even one in the outputs folder, is named and never linked (unsent_file)."""
        link = post.posted_dir(self.cfg) / "PR_UNPOSTED_15.md"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(self.review(16, self.tmp / "elsewhere"))
        self.keep(state_store.REVIEW, "pull request 15", file=str(link))
        (row,) = self.card(self.render()).split("<tr><td")[1:]
        self.assertIn("PR_UNPOSTED_15.md", row)
        self.assertNotIn("<a ", row)

    def test_the_team_dashboards_page_is_one_item(self):
        self.page_item(T1)
        self.page_item(T2)  # the next tick's page replaces it
        card = self.card(self.render())
        rows = card.split("<tr><td")[1:]
        self.assertEqual(len(rows), 1, card)
        self.assertIn("page push", rows[0])
        self.assertIn("the team dashboard&#x27;s web host", rows[0])
        self.assertIn("the team dashboard in prototype", rows[0])
        self.assertIn(f"/unsent?file={webdash.PUBLISHED_FILE}'", rows[0])

    def test_the_team_dashboards_page_is_linked_while_it_does_not_send(self):
        """While the team dashboard does not send (in prototype, or the monitor in standby), its page, made
        after every tick, is shown on the admin dashboard in either delivery: the card links it (team_page), with no
        item kept for it (served here, none is). Live, it is not; nor is a page that is a link.
        """
        pub = self.made_page()
        link = (
            f"<a href='http://127.0.0.1:{PORT}/unsent?file={webdash.PUBLISHED_FILE}' target='_blank' "
            f"rel='noopener noreferrer'>{webdash.PUBLISHED_FILE}</a>"
        )
        for delivery in ("publish", "serve"):
            with self.subTest(delivery=delivery):
                card = self.card(self.render(cfg=self.team(delivery)))
                self.assertIn(
                    "The team dashboard's page, made after every tick (the team dashboard in prototype): "
                    f"{link}.",
                    card,
                )
                self.assertIn("None right now.", card)  # no item kept
                live = self.render(cfg=self.team(delivery, "live"))
                self.assertNotIn(f"file={webdash.PUBLISHED_FILE}", live)
        state_store.set(self.tmp / "home", "standby.on", True)
        card = self.card(self.render(cfg=self.team("serve", "live")))
        self.assertIn(f"made after every tick (the monitor in standby): {link}.", card)
        state_store.set(self.tmp / "home", "standby.on", False)
        pub.unlink()
        pub.symlink_to(self.review(16, self.tmp / "elsewhere"))
        page = self.render(cfg=self.team("serve"))
        self.assertNotIn(f"file={webdash.PUBLISHED_FILE}", page)

    def test_the_admin_dashboards_only(self):
        self.keep(state_store.LABEL, "issue 3", "do not merge")
        self.assertIn("<section id='not-sent'", self.render())
        published = self.render(webdash.PUBLISHED)
        for absent in (
            "not-sent",
            "Not sent",
            "do not merge",
            "unsent-list",
            "/unsent?",
        ):
            self.assertNotIn(absent, published)

    def test_shown_while_an_item_is_kept_or_one_may_be(self):
        live = dict(self.cfg, features={"review_automation": LIVE, "merge_queue": LIVE})
        page = self.render(
            cfg=live
        )  # every feature that is on is live: nothing is kept
        self.assertNotIn("id='not-sent'", page)
        self.assertNotIn("#not-sent", page)
        # A feature in prototype, nothing kept yet.
        self.assertIn("<p class='sub'>None right now.</p>", self.card(self.render()))
        # Standby.
        state_store.set(self.tmp / "home", "standby.on", True)
        self.assertIn("None right now.", self.card(self.render(cfg=live)))
        state_store.set(self.tmp / "home", "standby.on", False)
        # An item kept, which stays after going live.
        self.keep(state_store.LABEL, "issue 3", "do not merge")
        self.assertIn("do not merge", self.card(self.render(cfg=live)))

    def test_the_newest_ones_and_where_the_rest_are(self):
        for i, at in enumerate((T1, T2, T3)):
            self.keep(state_store.LABEL, f"issue {i}", "do not merge", at=at)
        with mock.patch.object(webdash, "UNSENT_SHOWN", 2):
            card = self.card(self.render())
        self.assertIn("<span class='c'>3 kept · never sent later</span>", card)
        self.assertEqual(
            [r.split("issue ")[1][0] for r in card.split("<tr><td")[1:]], ["2", "1"]
        )
        self.assertIn(
            f"The 2 newest of 3; the rest are in {state_store.unsent_path(self.store.dir)}, or its older copy "
            "beside it.",
            card,
        )


class TestRoute(Base):
    """GET /unsent?file=NAME, the card's link: a navigation, with the access token, as test_tickets' full text link."""

    def ask(self, path: str, cfg: dict | None = None, **change) -> tuple:
        h = {
            "Host": f"127.0.0.1:{PORT}",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Dest": "document",
            "Authorization": "Basic "
            + base64.b64encode(f"anyone:{TOKEN}".encode()).decode(),
        }
        h.update(change)
        cfg = cfg or self.cfg
        access = webserve.access_info(cfg, PORT, {})
        return webserve.answer(cfg, self.store, "GET", path, h, b"", PORT, access)

    def test_a_reviews_file_as_text(self):
        p = self.review(12)
        self.keep(state_store.REVIEW, "pull request 12", file=str(p))
        status, h, body = self.ask("/unsent?file=PR_UNPOSTED_12.md")
        self.assertEqual((status, body), (200, p.read_bytes()))
        self.assertEqual(
            (h["Content-Type"], h["X-Content-Type-Options"], h["X-Frame-Options"]),
            ("text/plain; charset=utf-8", "nosniff", "DENY"),
        )

    def test_the_team_dashboards_page_in_a_sandbox(self):
        pub = self.page_item()
        status, h, body = self.ask(f"/unsent?file={webdash.PUBLISHED_FILE}")
        self.assertEqual((status, body), (200, pub.read_bytes()))
        self.assertEqual(h["Content-Type"], "text/html; charset=utf-8")
        policy = h["Content-Security-Policy"]
        self.assertTrue(policy.startswith("sandbox "), policy)
        self.assertIn("default-src 'none'", policy)
        self.assertIn("frame-ancestors 'none'", policy)
        for never in ("allow-scripts", "allow-same-origin", "allow-forms"):
            self.assertNotIn(never, policy)

    def test_the_team_dashboards_page_while_it_does_not_send(self):
        """While the team dashboard does not send, the service serves its page as the last tick made it, in a sandbox,
        in either delivery and with no item kept for it: of the state folder, that file alone. Live, with no item, it
        does not."""
        pub = self.made_page()
        # An item kept, so that the state folder holds its list of unsent items too.
        self.keep(state_store.LABEL, "issue 3", "do not merge")
        page = f"/unsent?file={webdash.PUBLISHED_FILE}"
        for delivery in ("publish", "serve"):
            cfg = self.team(delivery)
            with self.subTest(delivery=delivery):
                status, h, body = self.ask(page, cfg)
                self.assertEqual((status, body), (200, pub.read_bytes()))
                self.assertTrue(h["Content-Security-Policy"].startswith("sandbox "))
                for name in (
                    state_store.UNSENT_FILE,
                    self.store.state_path.name,
                    self.store.journal_path.name,
                    f"../state/{webdash.PUBLISHED_FILE}",
                ):
                    self.assertEqual(
                        self.ask(f"/unsent?file={name}", cfg)[0], 404, name
                    )
                self.assertEqual(self.ask(page, self.team(delivery, "live"))[0], 404)

    def test_no_other_file(self):
        out = post.posted_dir(self.cfg)
        self.keep(
            state_store.REVIEW,
            "pull request 13",
            file=str(self.review(13, self.tmp / "elsewhere")),
        )
        self.review(14)  # in the outputs folder, but no item names it
        link = out / "PR_UNPOSTED_15.md"
        link.symlink_to(self.review(16, self.tmp / "elsewhere"))
        self.keep(state_store.REVIEW, "pull request 15", file=str(link))
        self.keep(
            state_store.COMMENT, "pull request 17", "x", file=str(self.review(17))
        )
        (out / "notes.txt").write_text("x", encoding="utf-8")
        self.keep(state_store.REVIEW, "pull request 18", file=str(out / "notes.txt"))
        gone = self.review(19)
        self.keep(state_store.REVIEW, "pull request 19", file=str(gone))
        gone.unlink()
        for name in (
            "PR_UNPOSTED_13.md",
            "PR_UNPOSTED_14.md",
            "PR_UNPOSTED_15.md",
            "PR_UNPOSTED_17.md",
            "notes.txt",
            "PR_UNPOSTED_19.md",
            "",
            "../state/unsent.jsonl",
            state_store.UNSENT_FILE,
            webdash.PUBLISHED_FILE,  # no page item kept, and the team dashboard off
            "admin_access_token",
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    self.ask(f"/unsent?file={name}")[0::2],
                    (404, {"ok": False, "message": "no such file of an unsent item"}),
                )

    def test_a_file_that_became_a_link_after_the_check_is_never_read(self):
        """A file that becomes a link after the check (webdash.unsent_named) found it is not read: the service never
        opens a file through a link (O_NOFOLLOW)."""
        link = post.posted_dir(self.cfg) / "PR_UNPOSTED_15.md"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(self.review(16, self.tmp / "elsewhere"))
        with mock.patch.object(
            webdash, "unsent_named", return_value=(link, state_store.REVIEW)
        ):
            self.assertEqual(
                self.ask("/unsent?file=PR_UNPOSTED_15.md")[0::2],
                (404, {"ok": False, "message": "no such file of an unsent item"}),
            )

    def test_with_the_access_token_from_this_host_never_in_a_frame(self):
        self.keep(state_store.REVIEW, "pull request 12", file=str(self.review(12)))
        path = "/unsent?file=PR_UNPOSTED_12.md"
        self.assertEqual(self.ask(path, Authorization="")[0], 401)
        self.assertEqual(
            self.ask(path, **{"Sec-Fetch-Dest": "iframe"})[0::2],
            (403, {"ok": False, "message": "never in a frame"}),
        )
        self.assertEqual(self.ask(path, Host=f"rebound.example:{PORT}")[0], 403)
        self.assertEqual(self.ask(path, Origin="http://evil.example")[0], 403)
        self.assertEqual(self.ask(path)[0], 200)


if __name__ == "__main__":
    unittest.main()
