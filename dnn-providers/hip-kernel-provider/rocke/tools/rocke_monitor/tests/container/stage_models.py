# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The local worker's model and the parallel reviews, as stages of the image (tests/container/image.py runs them inside
the image built from docker/Dockerfile, as root, with no network), each on the deployment the image's stages share
(stages.py set_up), the local worker's model a binding of its deploy settings (model_bindings.worker). Every model
server is a fake on a loopback address in the container (tests/fakes/fake_models.py): no real model is ever called.

  python3 -B stages.py reviewer|parallel-reviews|parallel-reviews-local   (only as root inside the image:
  RKMON_CONTAINER=1, RKMON_AS=root)

  reviewer     one binding after another, each confirmed and delivered, then rk provision as each start runs it, whose
               step 7 (the local review loop's server) checks the model from the container, with its key: a fake
               OpenAI-compatible gateway (a route prefix, its own header with the raw key, /v1/models, a LiteLLM-style
               model-info route, which the checks never read, chat completions with tool calls), first
               with the key sent as a bearer token, the binding's default, then in the gateway's own header, the test
               chat answered first with a 429 and a Retry-After; a gateway route that is not /v1-shaped (chat at
               its own /chat/completions, models at the host's own /models with deployment IDs and display names, the
               key wanted in both places); a model of that route that answers in a simplified shape; a fake Ollama (its
               /api/tags, a tool-calling chat); and an Ollama model that only talks. Step 7's row says what it found,
               the internal state records it (the probe, the chat route, the model list's address, the display name,
               each cleared when the next binding's server has none), and rk features shows it. A model whose key the
               gateway refuses, or that fails the tool-call probe, is refused, with the reason: the reviews show as
               incomplete and rk doctor's row fails. Confirm's model probe (rkmon_lib/confirm.py MODEL_PROBE), run in
               the stage, finds each binding's address answering, the gateway that refuses the key too: confirm sends
               no key.
  parallel-reviews
               three parallel reviews against the fake gateway (at 127.0.0.2: a loopback address the monitor does not
               count as this machine, so it stands for a remote gateway; its binding's limits 4 requests and 400000
               tokens a minute), review automation live, three open PRs in the fake GitHub: rk features says why three
               run, the tick starts three drafts at once, rk status shows them, the review driver prints REVIEW_READY
               for each, the shared rate budget holds at the gateway (requests a minute), and a 429's Retry-After makes
               every review wait.
  parallel-reviews-local
               a local Ollama: one draft at a time; its binding's max_parallel_requests 2, delivered, lets two run.
               Its fake is on the loopback address (container_layout.LOOPBACK), the one address the monitor counts
               as this machine (config.is_local_url) and, with no network, the only one the container has: so the
               settings checks warn of it (check 11) and rk doctor's Ollama row fails, as for any Ollama on the
               loopback in the container. The stage reads neither.

Each checks too that no secret the deployment set is in any output, log or file outside the secrets folders.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
TESTS = HERE.parent
sys.path[:0] = [str(TESTS.parent), str(TESTS), str(TESTS / "fakes")]

import fake_models  # noqa: E402
import fakerepo  # noqa: E402
import stages  # noqa: E402
from rkmon_lib import (  # noqa: E402
    config,
    confirm,
    container_layout,
    program,
    runtime_config,
    scaler,
    secrets_registry,
    state_store,
)

GW_KEY = "k-example-" + "gateway-key-909090"  # a planted key, built at run time
HEADER = "X-Example-Key"
PRS = (11, 12, 13)
# the secret the gateway bindings name for their key
KEY = secrets_registry.FIRST_PASS_KEY
SERVER = "local review loop's server"  # rk provision's row of the local review loop's model (its step 7)
REVIEWER = "reviewer (the local review loop)"  # rk doctor's row of it


def make_prs(origin: pathlib.Path, numbers=PRS) -> list[dict]:
    """Open PRs in the fake upstream: refs/pull/N/head, each changing one file under rocKE's folder; the fake GitHub's
    view of each (fakerepo.pull_request)."""
    return [fakerepo.pull_request(origin, n) for n in numbers]


def ollama(server, model: str, **more) -> dict:
    """An Ollama binding of MODEL on the fake SERVER, with MORE (its limits)."""
    return {"provider": "ollama", "model_id": model, "server_url": server.url, **more}


def gateway(server, model: str, sent_as: str = "", **more) -> dict:
    """An OpenAI-compatible binding of MODEL at the fake SERVER, its key the secret KEY, sent as SENT_AS says
    (api_key_sent_as: "header" and "header_and_bearer" in HEADER, raw); blank, as the binding's default sends it (a
    bearer token). With MORE (its limits)."""
    binding = {
        "provider": "openai_compatible",
        "model_id": model,
        "base_url": server.url,
        "api_key_secret_name": KEY,
    }
    if sent_as:
        binding.update(api_key_sent_as=sent_as, api_key_header=HEADER)
    return {**binding, **more}


def delivered(d: stages.Deployment, port: int, binding: dict, over: dict) -> None:
    """BINDING as the local worker's (model_bindings.worker), with OVER's settings, confirmed and delivered."""
    d.confirm(stages.deploy_values(port, {**over, "model_bindings.worker": binding}))
    d.deliver()


def started(d: stages.Deployment, port: int, binding: dict) -> str:
    """BINDING delivered, then rk provision in a one-off container, as each start runs it (rkmon start): its output."""
    delivered(d, port, binding, {})
    return d.one_off("rk", "provision")


def recorded() -> dict:
    """What the internal state records of the local review loop's server (state_store; rk provision's step 7 writes it):
    its probe, chat_path, models_url and model_name, by the configuration's names."""
    return state_store.read(container_layout.MONITOR_HOME).get("worker") or {}


def reach_rows(d: stages.Deployment) -> list:
    """Confirm's rows of the local worker's model (confirm.model_rows) in the settings D confirmed last, its address
    called by confirm's model probe (confirm.MODEL_PROBE) from here, the container the deployment's commands run in, as
    docker/rkmon calls it from a one-off container on the deployment's network."""
    values = runtime_config.from_settings(d.plan.layer["settings"]).values
    used = [
        (name, url)
        for name, provider, url, _passes in confirm.models_in_use(values)
        if provider != "claude" and url
    ]
    urls = list(dict.fromkeys(url for _name, url in used))
    rc, said = stages.run(
        [
            sys.executable,
            "-I",
            "-c",
            confirm.MODEL_PROBE,
            str(confirm.REACH_TIMEOUT_S),
            *urls,
        ],
        dict(os.environ),
        timeout=confirm.REACH_TIMEOUT_S * len(urls) + 60,
    )
    return confirm.model_rows(
        values,
        confirm.model_answers(rc, said, urls),
        first=False,
        names=[name for name, _url in used],
    )


def _refused_key(st: stages.Stage, d: stages.Deployment, gw) -> None:
    """The first start, the gateway's binding sending the key as a bearer token, where the gateway wants it in its own
    header."""
    provision = d.out["rk provision"][1]
    status, row = stages.report_row(provision, SERVER)
    st.check(
        status == "WARN"
        and "HTTP 401 (unauthorized)" in row
        and "(sent a key as `Authorization: Bearer <key>`)" in row,
        "a key the gateway refuses: the first start completes, and its rk provision says the gateway refused the key, "
        "and how it was sent",
        row or provision[-1500:],
    )
    _rc, feats = d.rk("features")
    st.check(
        "| Reviews | on (incomplete) |" in feats
        and "REFUSED for the reviewer role: it failed the test chat: the model list got HTTP 401 (unauthorized)"
        in feats,
        "a key the gateway refuses refuses the model: rk features shows the reviews incomplete, with the reason",
        feats[-3000:],
    )
    _rc, report = d.rk("doctor")
    status, row = stages.report_row(report, REVIEWER)
    st.check(
        status == "FAIL" and f"openai at {gw.url}: no answer (HTTP 401)" in row,
        "rk doctor's row of the reviewer model fails with the 401",
        row or report[-1500:],
    )


def _own_header(st: stages.Stage, d: stages.Deployment, port: int, gw) -> None:
    """The gateway's binding with the key in the gateway's own header; the gateway answers the start's test chat first
    with a 429 and a Retry-After."""
    # the gateway's next chat request: the start's test chat
    gw.handler.fail429 = frozenset({gw.handler.chats + 1})
    text = started(d, port, gateway(gw, fake_models.GOOD, "header"))
    st.check(
        any(
            r["status"] == 429
            for r in fake_models.read_log(d.home / "fakes", "gateway")
        ),
        "the gateway answered the start's test chat with a 429",
    )
    status, row = stages.report_row(text, SERVER)
    st.check(
        status == "PASS"
        and f"{fake_models.GOOD} answers the test chat and the tool-call probe" in row,
        "the key in the gateway's own header, raw: the start's checks pass, the 429's Retry-After waited out",
        row or text[-1500:],
    )
    probe = recorded().get("probe") or {}
    paths = [r["path"] for r in fake_models.read_log(d.home / "fakes", "gateway")]
    st.check(
        not {"max_output", "rpm", "tpm"} & set(probe)
        and not [p for p in paths if "/model/info" in p],
        "the start never reads the gateway's model-info route, and the probe's record has none of its numbers",
        json.dumps(probe)[:800] + " ".join(paths)[-800:],
    )
    _rc, feats = d.rk("features")
    st.check(
        f"| Reviewer model: endpoint (<WORKER_URL>) | {gw.url} |" in feats
        and "max output 8192" not in feats
        and f"`{HEADER}: <key>`" in feats,
        "rk features shows the gateway, its route prefix and how the key is sent, and no model-info number",
        feats[-3000:],
    )


def _route(st: stages.Stage, d: stages.Deployment, port: int, route) -> None:
    """A gateway route that is not /v1-shaped, its model by its deployment ID, the key wanted in both places; then a
    model of the route that answers in a simplified shape."""
    text = started(d, port, gateway(route, fake_models.DEPLOY, "header_and_bearer"))
    status, row = stages.report_row(text, SERVER)
    w = recorded()
    host = route.url[: -len("/api")]
    st.check(
        status == "PASS"
        and (
            w.get("model_name"),
            w.get("chat_path"),
            w.get("models_url"),
            (w.get("probe") or {}).get("tools"),
        )
        == (fake_models.SHOWN, "/chat/completions", host + "/models", "ok"),
        "a route with chat at its own /chat/completions and models at the host's /models: the start finds and records "
        "both, and the name the list shows for the deployment ID the binding names",
        (row or text[-1500:]) + json.dumps(w)[:800],
    )
    _rc, feats = d.rk("features")
    st.check(
        f"{fake_models.DEPLOY} (shown as {fake_models.SHOWN})" in feats
        and f"`{HEADER}: <key>` and `Authorization: Bearer <key>`" in feats
        and "; chat route /chat/completions; models listed at " in feats,
        "rk features shows the route, the model's ID and name, and the key sent twice",
        feats[-3000:],
    )
    text = started(
        d, port, gateway(route, fake_models.UNIFIED.lower(), "header_and_bearer")
    )
    status, row = stages.report_row(text, SERVER)
    _rc, feats = d.rk("features")
    st.check(
        status == "WARN"
        and "but not the tool-call probe: the endpoint answers without tool calls"
        in row
        and "REFUSED for the reviewer role: it failed the tool-call probe: the endpoint answers without tool calls"
        in feats,
        "a model that answers in a simplified shape: the start says so, without a crash, and it is refused, with the "
        "reason",
        (row or text[-1500:]) + feats[-1500:],
    )


def _ollama(st: stages.Stage, d: stages.Deployment, port: int, oll) -> None:
    """An Ollama binding: a model that calls tools, then one that only talks."""
    text = started(d, port, ollama(oll, fake_models.OLLAMA_GOOD))
    status, row = stages.report_row(text, SERVER)
    w = recorded()
    probe = w.get("probe") or {}
    st.check(
        status == "PASS"
        and f"{fake_models.OLLAMA_GOOD} answers the test chat and the tool-call probe"
        in row
        and (probe.get("url"), probe.get("model")) == (oll.url, fake_models.OLLAMA_GOOD)
        and not any(w.get(k) for k in ("chat_path", "models_url", "model_name")),
        "Ollama: the start's checks pass, from the binding alone: what was found for the route is cleared",
        (row or text[-1500:]) + json.dumps(w)[:800],
    )
    _rc, feats = d.rk("features")
    st.check(
        "test chat ok; tool-call probe ok; reported: context 32768" in feats,
        "rk features shows the probe's result",
        feats[-3000:],
    )
    text = started(d, port, ollama(oll, fake_models.OLLAMA_TEXT))
    status, row = stages.report_row(text, SERVER)
    st.check(
        status == "WARN"
        and f"{fake_models.OLLAMA_TEXT} answers the test chat, but not the tool-call probe: it answered in text"
        in row,
        "a model that only talks: the start says it fails the tool-call probe",
        row or text[-1500:],
    )
    _rc, feats = d.rk("features")
    lines = feats.splitlines()
    reviews = next((ln for ln in lines if ln.startswith("| Reviews |")), "")
    refused = next((ln for ln in lines if "REFUSED" in ln), "")
    reason = (fake_models.OLLAMA_TEXT, "tool-call probe", "it answered in text")
    st.check(
        "| on (incomplete) |" in reviews
        and all(p in reviews for p in ("model_bindings.worker.model_id", *reason))
        and all(p in refused for p in ("reviewer role", *reason)),
        "it is refused for the reviewer role, with the reason, and the reviews show as incomplete",
        f"{reviews}\n{refused}" if reviews or refused else feats[-3000:],
    )
    _rc, report = d.rk("doctor")
    status, row = stages.report_row(report, REVIEWER)
    st.check(
        status == "FAIL" and "failed the tool-call probe" in row,
        "rk doctor says why it is refused",
        row or report[-1500:],
    )


@stages.image_stage("reviewer")
def reviewer(st: stages.Stage, home: pathlib.Path, port: int) -> None:
    """The local worker's model, one binding after another, as each start checks it: see the module's docstring."""
    fakes, servers, reach = home / "fakes", [], {}
    with tempfile.TemporaryDirectory() as keys:
        # the gateways' own copy of the key, outside every folder the stage searches for it
        key_file = pathlib.Path(keys) / "gateway_key"
        key_file.write_text(GW_KEY + "\n", encoding="utf-8")
        try:
            gw = fake_models.start(
                fakes, "gateway", "gateway", port=port + 11, key_file=str(key_file)
            )
            servers.append(gw)
            route = fake_models.start(
                fakes,
                "gateway",
                "route",
                port=port + 12,
                prefix="/api",
                key_file=str(key_file),
                bearer_too=True,
                chat_routes=("/chat/completions",),
                models_paths=("/models",),
                deployments=True,
                info=False,
            )
            servers.append(route)
            oll = fake_models.start(fakes, "ollama", "ollama", port=port + 10)
            servers.append(oll)
            first = {"model_bindings.worker": gateway(gw, fake_models.GOOD)}
            secrets = dict(stages.secret_values(), **{KEY: GW_KEY})
            with stages.set_up(home, port, over=first, secrets=secrets) as d:
                reach["the gateway, the key as a bearer token"] = reach_rows(d)
                _refused_key(st, d, gw)
                _own_header(st, d, port, gw)
                reach["the gateway, the key in its own header"] = reach_rows(d)
                _route(st, d, port, route)
                reach["the route"] = reach_rows(d)
                _ollama(st, d, port, oll)
                reach["Ollama"] = reach_rows(d)
                rows = [(where, r) for where, rs in reach.items() for r in rs]
                st.check(
                    len(rows) == len(reach)
                    and all(
                        r.status == "PASS" and r.found == f"answers, {confirm.CALLED}"
                        for _where, r in rows
                    ),
                    "confirm's model probe finds each binding's address answering, the gateway that refuses the key "
                    "too: it sends no key",
                    "; ".join(f"{where}: {r.status} {r.found}" for where, r in rows),
                )
                stages.secrets_absent(st, d)
        finally:
            for s in servers:
                s.stop()


def overlap(convs: dict) -> int:
    """The most review conversations in flight at one moment: each from its first request's start to its last's end."""
    spans = [
        (min(r["t0"] for r in rs), max(r["t1"] for r in rs))
        for rs in convs.values()
        if rs
    ]
    points = sorted([(a, 1) for a, _b in spans] + [(b, -1) for _a, b in spans])
    most = cur = 0
    for _t, d in points:
        cur += d
        most = max(most, cur)
    return most


def most_in_a_minute(times: list[float]) -> int:
    times = sorted(times)
    return max((sum(1 for u in times if t <= u < t + 60.0) for t in times), default=0)


def starts(tick: str) -> list[str]:
    """The drafts a tick's output says it started: its REVIEW_START lines."""
    return [ln for ln in tick.splitlines() if ln.startswith("REVIEW_START #")]


def pipes_of(d: stages.Deployment) -> dict:
    """The parallel reviews of the configuration that the settings D confirmed last make (runtime_config), as the code
    counts and words them, naming the program as each of the deployment's commands is told it (stages.WRAPPER):
    config.pipes's asked, effective and why; `line`, config.pipes_line's, which rk features shows; `cap`, the most the
    reviewer's endpoint allows at once and why (scaler.pipe_cap), which rk status shows. The checks look for these, so
    the stage takes up a new wording of the code (tests/test_models.py pins config.pipes's words).
    """
    values = runtime_config.from_settings(d.plan.layer["settings"]).values
    with program.run_by(stages.WRAPPER):
        return dict(
            config.pipes(values),
            line=config.pipes_line(values),
            cap=scaler.pipe_cap(values),
        )


def _three_at_once(st: stages.Stage, d: stages.Deployment) -> None:
    """Three drafts at once on the gateway, each to REVIEW_READY, under the shared rate budget."""
    provision = d.out["rk provision"][1]
    status, row = stages.report_row(provision, SERVER)
    st.check(
        status == "PASS"
        and f"{fake_models.GOOD} answers the test chat and the tool-call probe" in row,
        "the first start's rk provision: the gateway's model passes its checks from the container",
        row or provision[-1500:],
    )
    p = pipes_of(d)
    _rc, feats = d.rk("features")
    st.check(
        (p["asked"], p["effective"]) == (3, 3) and p["line"] in feats,
        "rk features says: asked 3, running 3, and why",
        f"{feats[-1200:]}\nlooked for: {p['line']}",
    )
    rc, tick = d.rk("tick")
    started_now = starts(tick)
    st.check(
        rc in (0, 3)
        and len(started_now) == 3
        and all("of 3)" in ln for ln in started_now),
        "the tick starts three drafts at once, one parallel review each",
        tick[-2500:],
    )
    _rc, status_text = d.rk("status")
    line = next(
        (
            x.strip()
            for x in status_text.splitlines()
            if x.strip().startswith("work in flight: ")
        ),
        "",
    )
    st.check(
        line.startswith("work in flight: 3 of 3 parallel reviews")
        and all(f"#{n} first-round" in line for n in PRS),
        "rk status shows the three drafts in flight",
        status_text[:1500],
    )
    ready, said, deadline = set(), [], time.time() + 900
    while len(ready) < len(PRS) and time.time() < deadline:
        _rc, text = d.rk("review-driver", timeout=1200)
        said.append(text)
        ready |= {
            int(ln.split("#", 1)[1].split()[0])
            for ln in text.splitlines()
            if ln.startswith("REVIEW_READY #")
        }
        if (
            "REVIEW_FAIL" in text
            or "nothing to start now" in text
            and len(ready) < len(PRS)
        ):
            time.sleep(5)
    st.check(
        ready == set(PRS),
        "the review driver prints REVIEW_READY for each PR",
        "\n".join(said)[-3000:],
    )
    workers = pathlib.Path(container_layout.WORK_DIR) / "workers"
    st.check(
        all(
            (workers / f"pr{n}" / "review_agent.md").is_file()
            and (workers / f"pr{n}" / "review_agent.md").stat().st_size > 0
            for n in PRS
        ),
        "each PR has its draft from the reviewer model",
    )
    log = [
        r
        for r in fake_models.read_log(d.home / "fakes", "gateway")
        if r["method"] == "POST" and r["pr"] in PRS
    ]
    convs = {n: [r for r in log if r["pr"] == n] for n in PRS}
    st.check(
        overlap(convs) == 3,
        "the three drafts ran at once (their conversations overlap at the gateway)",
        json.dumps(
            {
                n: [(r["t0"], r["t1"], r["status"]) for r in rs]
                for n, rs in convs.items()
            }
        )[:1500],
    )
    busiest = most_in_a_minute([r["t0"] for r in log])
    st.check(
        0 < busiest <= 4,
        f"the shared budget held: at most 4 requests in any minute at the gateway (busiest minute: {busiest})",
        json.dumps([(r["t0"], r["pr"], r["status"]) for r in log]),
    )
    r429 = [r for r in log if r["status"] == 429]
    after = [r["t0"] for r in log if r429 and r["t0"] > r429[0]["t1"]]
    st.check(
        len(r429) == 1 and after and min(after) - r429[0]["t1"] >= 4.5,
        "a 429 with Retry-After 5: no review asked again for 5 s",
        json.dumps([(r["t0"], r["t1"], r["pr"], r["status"]) for r in log]),
    )
    budget = stages.read(pathlib.Path(container_layout.WORK_DIR) / "run" / "budget.log")
    st.check(
        "HTTP 429 from" in budget
        and "every review waits 5 s" in budget
        and "budget: waiting" in budget,
        "the shared budget's log shows the 429's wait and the waits for the requests-a-minute limit",
        budget,
    )


@stages.image_stage("parallel-reviews")
def parallel_reviews(st: stages.Stage, home: pathlib.Path, port: int) -> None:
    """Three parallel reviews against a remote gateway: see the module's docstring."""
    with tempfile.TemporaryDirectory() as keys:
        # the gateway's own copy of the key, outside every folder the stage searches for it
        key_file = pathlib.Path(keys) / "gateway_key"
        key_file.write_text(GW_KEY + "\n", encoding="utf-8")
        # A gateway at 127.0.0.2 (a remote one, to the monitor), each chat 1.5 s; the second request of the review
        # conversations gets a 429 with Retry-After 5.
        gw = fake_models.start(
            home / "fakes",
            "gateway",
            "gateway",
            host="127.0.0.2",
            port=port + 14,
            key_file=str(key_file),
            rpm=4,
            delay_s=1.5,
            retry_after="5",
            fail429_review={2},
        )
        try:
            limits = {"requests_per_minute": 4, "tokens_per_minute": 400000}
            over = {
                "features.review_automation.mode": "live",
                "features.review_automation.parallel_reviews": 3,
                "model_bindings.worker": gateway(
                    gw, fake_models.GOOD, "header", limits=limits
                ),
            }
            secrets = dict(stages.secret_values(), **{KEY: GW_KEY})
            with stages.set_up(
                home, port, over=over, secrets=secrets, upstream=make_prs
            ) as d:
                _three_at_once(st, d)
                stages.secrets_absent(st, d)
        finally:
            gw.stop()


@stages.image_stage("parallel-reviews-local")
def parallel_reviews_local(st: stages.Stage, home: pathlib.Path, port: int) -> None:
    """A local Ollama: one draft at a time, two once its binding allows them: see the module's docstring."""
    # On the loopback: an Ollama on this machine, to the monitor (see the module's docstring).
    oll = fake_models.start(
        home / "fakes",
        "ollama",
        "ollama",
        host=container_layout.LOOPBACK,
        port=port + 10,
    )
    try:
        over = {
            "features.review_automation.mode": "live",
            "features.review_automation.parallel_reviews": 3,
        }
        first = {**over, "model_bindings.worker": ollama(oll, fake_models.OLLAMA_GOOD)}
        secrets = stages.secret_values(KEY)  # the binding names no key
        with stages.set_up(
            home, port, over=first, secrets=secrets, upstream=make_prs
        ) as d:
            # the drafts stay in flight; the start's own checks had no wait
            oll.handler.delay_s = 30
            p = pipes_of(d)
            _rc, feats = d.rk("features")
            st.check(
                (p["asked"], p["effective"]) == (3, 1) and p["line"] in feats,
                "a local Ollama: asked 3, running 1, and why",
                f"{feats[-1200:]}\nlooked for: {p['line']}",
            )
            _rc, tick = d.rk("tick")
            started_now = starts(tick)
            st.check(
                len(started_now) == 1 and "(parallel review 1 of 1)" in started_now[0],
                "the tick starts one draft",
                tick[-2000:],
            )
            two = ollama(
                oll, fake_models.OLLAMA_GOOD, limits={"max_parallel_requests": 2}
            )
            delivered(d, port, two, over)
            p = pipes_of(d)
            _rc, feats = d.rk("features")
            st.check(
                (p["asked"], p["effective"]) == (3, 2) and p["line"] in feats,
                "its binding's max_parallel_requests 2, delivered: two at once",
                f"{feats[-1200:]}\nlooked for: {p['line']}",
            )
            _rc, tick = d.rk("tick")
            started_now = starts(tick)
            st.check(
                len(started_now) == 1 and "(parallel review 2 of 2)" in started_now[0],
                "the next tick starts a second draft beside the first",
                tick[-2000:],
            )
            _rc, status_text = d.rk("status")
            cap, why = p["cap"]
            most = f"{cap} run at most: {why}"  # as scaler.status_lines words the cap
            st.check(
                cap == 2 and most in status_text,
                "rk status says 2 run at most, and why",
                f"{status_text[:1200]}\nlooked for: {most}",
            )
            d.rk("kill", "--wait", "1")  # the drafts in flight end with the stage
            stages.secrets_absent(st, d)
    finally:
        oll.stop()


STAGES = {
    "reviewer": reviewer,
    "parallel-reviews": parallel_reviews,
    "parallel-reviews-local": parallel_reviews_local,
}
