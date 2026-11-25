# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The reviewer model's checks, for the doctor and each start: the model the deploy settings file's worker binding
names, on an Ollama server or any OpenAI-compatible endpoint (an on-prem gateway, vLLM, LiteLLM).

An OpenAI-compatible endpoint's model list is at <base>/v1/models or wherever it answers (<base>/models, or the host's
own /v1/models or /models: a gateway may list its models outside its chat route; recorded as `worker.models_url`), each
model by the ID requests send (a gateway's deployment ID) and the display name the list gives besides it
(`worker.model_name`); its chat route is /v1/chat/completions or /chat/completions under the base (recorded as
`worker.chat_path`): `openai_list`.

`check` is what the start runs (rk provision's step 7): the model list, a tiny chat, and the tool-call probe
(worker/reviewer_client.py probe_tools: the review loop's own request shape). Its record is config `worker.probe`; a
model that fails the chat or the probe is refused for the reviewer role (config.probe_failure), and review-drafts stays
incomplete until one passes.

Everything here is read-only: nothing logs in, pulls a model, or loads one beyond the probe's own requests to the
endpoint the operator chose.
"""

from __future__ import annotations

import http.client
import json
import sys

from . import config, container_layout, paths

# Ollama's own port, and this host's Ollama on it: the server of a configuration that names only its model
# (config.worker_settings' older worker_model).
OLLAMA_PORT = 11434
OLLAMA_DEFAULT = f"http://{container_layout.LOOPBACK}:{OLLAMA_PORT}"


def client():
    """worker/reviewer_client.py, the one HTTP client of the reviewer model."""
    sys.path.insert(0, str(paths.PACKAGE / "worker"))
    import reviewer_client

    return reviewer_client


# --- OpenAI-compatible endpoints ----------------------------------------------------------------------------------------


def openai_list(s: dict, timeout: float = 20) -> tuple[list[dict], str, str]:
    """An OpenAI-compatible endpoint's models, [{"name" (the ID requests send: case and dots kept), "display" (the
    name the list shows, else the ID), "context"}], an error ("" when it answered), and the address of the list that
    answered (reviewer_client.list_models: worker.models_url, else <base>/v1/models, <base>/models, the host's own).
    """
    lc = client()
    s = dict(s, backend="openai")
    try:
        status, entries, where = lc.list_models(s, timeout)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        return [], f"no answer from {s['url']}: {type(exc).__name__}: {exc}", ""
    if status != 200:
        return [], lc.explain_status(status, s, "the model list"), ""
    return (
        [
            {"name": m["id"], "display": m["name"], "context": m["context"]}
            for m in entries
        ],
        "",
        where,
    )


# --- the checks -----------------------------------------------------------------------------------------------------------

# Where an OpenAI-compatible endpoint's model list and chat route answered the probe, as its record keeps them.
ROUTES = ("models_url", "chat_path")
# How a probe's record of a model the endpoint's model list does not name begins (its "chat": no test chat sent).
NOT_LISTED = "the endpoint does not list"


def key_sent(s: dict) -> str:
    """How the settings S send their key, as a probe's record keeps it ("key_sent"): a digest of the headers that carry
    it (reviewer_client.api_key, llm_client.auth_headers, then llm_client.key_fingerprint), which tells another key, or
    another header, scheme or way of sending it, from this one, and never shows the key; "none" with no key.
    """
    from . import first_pass

    lc = client()
    headers = lc.auth_headers(s, lc.api_key(s))
    return first_pass.llm_client().key_fingerprint(
        json.dumps(headers, sort_keys=True) if headers else ""
    )


def probe_record(s: dict, p: dict, now=None) -> dict:
    """The record config `worker.probe` keeps of P, reviewer_client.probe's answer for the settings S: {"at", "url",
    "model", "backend", "key_sent" (how it sent the key: `key_sent`), "chat" ("ok", or why not: the endpoint's refusal
    or error, a model it does not list, no HTTP answer (reviewer_client.NO_ANSWER and why), or a rate limit still there
    after its waits, which reads "not checked: ..." and never refuses the model), "tools" ("" here: not reached),
    "context" (the context length the model list or the server reports), "latency_s"}, and for an OpenAI-compatible
    endpoint where its model list and its chat route answered (ROUTES).
    """
    from .store import now as _now

    rec = {
        "at": (now or _now)(),
        "url": s["url"],
        "model": s["model"],
        "backend": s["backend"],
        "key_sent": key_sent(s),
        "chat": "",
        "tools": "",
        "context": p["context"],
        "latency_s": p["latency_s"],
    }
    if s["backend"] == "openai":
        rec.update({key: p.get(key) or "" for key in ROUTES})
    if not p["answers"]:
        rec["chat"] = p["error"] or f"no answer from {config.safe_url(s['url'])}"
    elif not p["model_found"]:
        shown = ", ".join(map(str, p["models"][:8])) or "none"
        rec["chat"] = f"{NOT_LISTED} {s['model']} (it lists: {shown})"
    else:
        rec["chat"] = p.get("rate_limited") or p["error"] or "ok"
    return rec


def check(s: dict, timeout: float = 60, now=None) -> dict:
    """The checks of the reviewer model with the settings S, as config `worker.probe` records them: the probe's record
    (`probe_record`), then, once its test prompt is answered, the tool-call probe ("tools": "ok" or why not, which
    begins with reviewer_client.NO_ANSWER when its request got no HTTP answer). The tool-call probe sends S as it is,
    with only where the probe found the model list and the chat route applied (ROUTES): given the settings the reviews
    send (config.model_for(cfg, "draft"), as rk provision gives them), it sends a review's context and max output. No
    model-info route is read."""
    lc = client()
    rec = probe_record(s, lc.probe(s, timeout), now)
    if rec["chat"] != "ok":
        return rec
    routes = {key: rec[key] for key in ROUTES if rec.get(key)}
    t = lc.probe_tools(dict(s, **routes), timeout=max(timeout, 120))
    rec["tools"] = "ok" if t["ok"] else t["error"]
    return rec


def not_checked(rec: dict) -> str:
    """Why the start's checks of a reviewer model did not finish (the endpoint kept answering HTTP 429), or ""."""
    return (
        next(
            (
                str(rec.get(k))
                for k in ("chat", "tools")
                if str(rec.get(k) or "").startswith(client().NOT_CHECKED)
            ),
            "",
        )
        if isinstance(rec, dict)
        else ""
    )


def apply_found(w: dict, rec: dict) -> dict:
    """The worker settings with what a probe's record (probe_record) holds: the context held to what the endpoint
    reports, and where its model list and chat route answered when they are not the default places. A new dict; `w` is
    not changed."""
    out = dict(w)
    found = rec.get("context")
    if (
        isinstance(found, int)
        and found > 0
        and (not out.get("context") or int(out["context"]) > found)
    ):
        out["context"] = found
    if w.get("backend") == "openai":
        lc = client()
        if rec.get("chat_path") and rec["chat_path"] != lc.CHAT_PATHS[0]:
            out["chat_path"] = rec["chat_path"]
        if (
            rec.get("models_url")
            and rec["models_url"] != lc.models_urls(dict(w, models_url=""))[0]
        ):
            out["models_url"] = rec["models_url"]
    return out


def describe(rec: dict) -> list[str]:
    """A probe's record (probe_record) in words, for the setup summary (rkmon_lib/summary.py); with no record, the start
    that makes one (`<wrapper> stop --drain`, then `<wrapper> start`)."""
    from . import doctor, steps
    from .checks import Steps

    if not rec:
        again = Steps(steps.restart_parts(doctor.wrapper_name()))
        return [f"not checked yet: the next start checks it ({again})"]
    parts = [
        f"checked {rec.get('at') or '?'}: test chat {rec.get('chat') or 'not run'}"
    ]
    if rec.get("chat") == "ok":
        parts.append(f"tool-call probe {rec.get('tools') or 'not run'}")
    if rec.get("context"):
        parts.append(f"reported: context {rec['context']}")
    where = [
        f"chat at {rec['chat_path']}" if rec.get("chat_path") else "",
        (
            f"models listed at {config.safe_url(rec['models_url'])}"
            if rec.get("models_url")
            else ""
        ),
    ]
    if any(where):
        parts.append(", ".join(x for x in where if x))
    return parts
