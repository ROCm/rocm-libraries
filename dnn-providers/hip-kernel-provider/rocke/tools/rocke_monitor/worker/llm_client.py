# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The model client: provider adapters behind one `chat` call, and the shared limiter every model caller uses.

A caller passes a binding's settings (rkmon_lib/bindings.py client_settings): provider, url, chat_path, the key's file
(api_key_file: a key comes from a file alone, never from a variable) and where the key goes (key_headers, else
auth_header, auth_scheme, auth_mode: the same keys as the portable package's worker settings), model, limits and the
client's own settings. A `$NAME` url is read at each request, from the caller's environment (`env_value`). The key and
the headers' values are never printed, logged or written: an error that might quote them has them replaced by `<key>`.

Provider adapters (PROVIDERS) turn one neutral request into the provider's and its answer back:
- "openai": an OpenAI-compatible chat completions route (an API gateway, vLLM, Ollama's /v1): streamed answers (server-
  sent events, with the usage chunk asked for) and whole JSON answers.
- "anthropic": the Anthropic Messages API (/v1/messages; Claude through the Anthropic API or the gateway Claude Code
  uses): the same tool loop, streamed or whole answers (AnthropicMessages).
The neutral answer is {"content", "reasoning", "tool_calls": [{"id", "function": {"name", "arguments": dict}}],
"finish_reason", "usage"} (the anthropic adapter adds the answer's own `blocks`), `usage` normalized by `usage_of`
(prompt, the cached part of it, completion, reasoning, and the fields the provider sent). Tool results go back as
{"role": "tool", "tool_name", "content"}; an answer's reasoning never joins its content (an assistant turn sent back
with its `blocks` keeps them as they came). Errors are RateLimited (HTTP 429, with its wait), GatewayError (with
its status and whether a resend may help) and OverBudget (the caller's own caps). A request ends by its caller's
deadline (`Limiter.deadline_now`): the adapter's socket waits no longer, and a stream still running then is cut.

`Limiter` holds an endpoint's limits for every caller on this machine: one bucket file under the limiter folder per
provider, key and model (`bucket`), so two bindings that share a key and a model share a bucket. It is read and written
under flock, and holds:
- the requests and tokens of the last rolling minute, against the binding's rpm and tpm;
- the requests in flight, against the binding's `concurrency` (0: no cap), so requests the endpoint would queue wait
  here instead, in rank order;
- a time to wait until: an HTTP 429 sets it from its Retry-After, else a back-off that doubles from BACKOFF_S;
- the callers waiting and the callers active (asked in the last CALLER_TTL_S), each with its rank: a priority's work
  first (a develop fix, one of the user's priorities), then the caller closest to done (`progress`). The best-ranked
  waiter goes first, and the last `reserve_pct` of a minute's budget goes only to the best-ranked active caller;
- five minutes of history (tokens, requests, waits, 429s), which `window_stats` sums.
One Limiter is one caller's run: it also holds that run's token budget (`review_token_budget`) and wall-time cap
(`review_wall_min`), and raises OverBudget before a request would pass either. It counts the run's requests, prompt,
cached, completion and reasoning tokens, seconds waited and 429s, per pass. Each answered request is one line of
<limiter folder>/tokens.jsonl, and each wait one line of limiter.log.

A run's threads share its Limiter: each thread's request has its own ticket, tags (the pass and step its token line
names) and progress, and a thread may run a phase with its own wall time (`phase`): its requests end by the sooner of
the phase's deadline and the run's cap, and the run's counts add up every thread's.
"""

from __future__ import annotations

import email.utils
import fcntl
import hashlib
import http.client
import json
import os
import pathlib
import socket
import threading
import re
import time
import urllib.parse

WINDOW_S = 60.0  # the endpoints count per minute
SLACK_S = 1.0  # a request leaves our window a second after its minute: the endpoint counts by its own clock
HISTORY_S = 300.0  # the history window_stats sums
QUEUE_POLL_S = 1.0  # a caller behind better-ranked ones looks again this often
QUEUE_TTL_S = 1800.0  # a waiting caller's place is dropped when its process is gone, or after this long
CALLER_TTL_S = 180.0  # a caller that has not asked for this long is not active
INFLIGHT_TTL_S = 3600.0  # a request in flight is dropped when its process is gone, or after this long
BACKOFF_S, BACKOFF_MAX_S = 5.0, 300.0
MAX_RATE_WAITS = 12  # 429s retried in one chat call before it gives up
CHARS_PER_TOKEN = 3.5  # the estimate a request takes from the window before the endpoint reports its usage
TOKENS_LOG, LIMITER_LOG = "tokens.jsonl", "limiter.log"
EFFORTS = (
    "low",
    "medium",
    "high",
)  # what an OpenAI-compatible request's reasoning_effort takes
CLAUDE_EFFORTS = (
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)  # what a Messages API request's output_config takes


class OverBudget(RuntimeError):
    """The run's token budget or wall-time cap, or a phase's wall time, would be passed: `kind` is "tokens" or "wall",
    `scope` is "run" or "phase"."""

    def __init__(self, kind: str, msg: str, scope: str = "run"):
        super().__init__(msg)
        self.kind, self.scope = kind, scope


class RateLimited(ConnectionError):
    """HTTP 429: wait `wait_s` (its Retry-After), or the back-off when it gave none (0)."""

    def __init__(self, msg: str, wait_s: float = 0.0):
        super().__init__(msg)
        self.wait_s = wait_s


class GatewayError(ConnectionError):
    """An answer that is not a chat completion: `status` (0 for none) and whether a resend may help (`transient`)."""

    def __init__(self, msg: str, status: int = 0, transient: bool = False):
        super().__init__(msg)
        self.status, self.transient = status, transient


# --- the key, the address and the headers: the binding's values, or references to the deployment's ---------------------

ENV_REF = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def env_value(name: str) -> str:
    """A setting by its variable's name, from the environment the caller runs in; "" when it does not set it. Never
    logged."""
    if not name or not ENV_NAME.fullmatch(str(name)):
        return ""
    return os.environ.get(name) or ""


def resolve(value) -> str:
    """A setting that is a `$NAME` (or `${NAME}`) reference, read by env_value; any other value as it is."""
    text = str(value or "").strip()
    m = ENV_REF.fullmatch(text)
    return env_value(m.group(1)) if m else text


def base_url(s: dict) -> str:
    """The binding's base address: `url`, or the value of the variable a `$NAME` url names."""
    return resolve(s.get("url")).rstrip("/")


def api_key(s: dict) -> str:
    """The key, from api_key_file: "" when it names none, or a file that is missing or empty. A key comes from a file
    alone, never from a variable."""
    path = s.get("api_key_file") or ""
    if not path:
        return ""
    try:
        return (
            pathlib.Path(os.path.expanduser(path)).read_text(encoding="utf-8").strip()
        )
    except OSError:
        return ""


def auth_headers(s: dict, key: str) -> dict:
    """Where the key goes: the binding's `key_headers` when it has them ({header: its value, KEY_MARK in the key's
    place}: a claude binding sends the key as the Claude session does, rkmon_lib/bindings.py); else by auth_mode: "" or
    "header", in auth_header after auth_scheme (a blank scheme sends the key alone); "bearer", as
    `Authorization: Bearer <key>` only; "both", in auth_header and as a bearer token too.
    """
    if not key:
        return {}
    if s.get("key_headers"):
        # Key headers are rkmon_lib/bindings.py's alone, so the package's code is there wherever they are; the worker
        # tools installed on their own (rkmon_lib/deploy.py install_worker_tools) never have them.
        from rkmon_lib.custom_headers import KEY_MARK

        return {
            str(name): str(value).replace(KEY_MARK, key)
            for name, value in dict(s["key_headers"]).items()
        }
    mode = (s.get("auth_mode") or "").lower()
    if mode == "bearer":
        return {"Authorization": f"Bearer {key}"}
    header, scheme = s.get("auth_header") or "Authorization", s.get(
        "auth_scheme", "Bearer"
    )
    h = {header: f"{scheme} {key}" if scheme else key}
    if mode == "both" and header.lower() != "authorization":
        h["Authorization"] = f"Bearer {key}"
    return h


def scrub(text: str, key) -> str:
    """`text` with the key (or each of a list of secrets) replaced, for an error message or a log line."""
    for secret in ([key] if isinstance(key, str) else list(key or [])):
        if secret:
            text = text.replace(secret, "<key>")
    return text


def key_fingerprint(key: str) -> str:
    """A short digest that tells keys apart without revealing them ("none" for no key)."""
    return hashlib.sha256(key.encode()).hexdigest()[:16] if key else "none"


def estimate_tokens(*parts) -> int:
    """A request's or a text's size in tokens, roughly (CHARS_PER_TOKEN)."""
    n = sum(
        len(p if isinstance(p, str) else json.dumps(p, ensure_ascii=False))
        for p in parts
        if p
    )
    return int(n / CHARS_PER_TOKEN) + 1


def wait_from_headers(hdrs, now: float | None = None) -> float | None:
    """Seconds an endpoint asks us to wait: Retry-After (seconds or an HTTP date) or retry-after-ms; None for none."""
    h = {k.lower(): v for k, v in (hdrs or [])}
    if h.get("retry-after-ms"):
        try:
            return max(0.0, float(h["retry-after-ms"]) / 1000.0)
        except ValueError:
            pass
    ra = h.get("retry-after")
    if ra:
        try:
            return max(0.0, float(ra))
        except ValueError:
            try:
                return max(
                    0.0,
                    email.utils.parsedate_to_datetime(ra).timestamp()
                    - (now or time.time()),
                )
            except (TypeError, ValueError, IndexError):
                return None
    return None


def _num(v) -> int:
    try:
        return max(0, int(v or 0))
    except (TypeError, ValueError):
        return 0


def usage_of(msg: dict, sent_est: int) -> dict:
    """A request's tokens: what the provider reported (prompt, the cached part of it, completion, reasoning), else
    estimates; `reported` says which, and `usage_keys` names the fields it sent."""
    u = msg.get("usage") if isinstance(msg.get("usage"), dict) else {}
    cdet = (
        u.get("completion_tokens_details")
        if isinstance(u.get("completion_tokens_details"), dict)
        else {}
    )
    pdet = (
        u.get("prompt_tokens_details")
        if isinstance(u.get("prompt_tokens_details"), dict)
        else {}
    )
    reasoning = _num(cdet.get("reasoning_tokens")) or _num(u.get("reasoning_tokens"))
    cached = (
        _num(pdet.get("cached_tokens"))
        or _num(u.get("cached_tokens"))
        or _num(u.get("prompt_cache_hit_tokens"))
        or _num(u.get("cache_read_input_tokens"))
    )
    est_out = estimate_tokens(
        msg.get("content"),
        msg.get("reasoning"),
        [tc.get("function") for tc in msg.get("tool_calls") or []],
    )
    prompt = _num(u.get("prompt_tokens")) or sent_est
    completion = _num(u.get("completion_tokens")) or est_out
    keys = sorted(
        set(u)
        | {f"prompt_tokens_details.{k}" for k in pdet}
        | {f"completion_tokens_details.{k}" for k in cdet}
    )
    return {
        "prompt": prompt,
        "cached": cached,
        "completion": completion,
        "reasoning": reasoning,
        "reported": bool(u),
        "total": prompt + completion,
        "usage_keys": keys,
    }


# --- the provider adapters --------------------------------------------------------------------------------------------


def to_openai(messages: list[dict]) -> list[dict]:
    """The neutral messages in OpenAI's form: tool calls get ids and JSON-string arguments, and each tool result names
    the call it answers, in order."""
    out, pending, k = [], [], 0
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                cid = tc.get("id") or f"call_{k}"
                k += 1
                calls.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": (
                                args
                                if isinstance(args, str)
                                else json.dumps(args or {})
                            ),
                        },
                    }
                )
            pending = [c["id"] for c in calls]
            out.append(
                {
                    "role": "assistant",
                    "content": m.get("content") or "",
                    "tool_calls": calls,
                }
            )
        elif m["role"] == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": pending.pop(0) if pending else f"call_{k}",
                    "content": m.get("content", ""),
                }
            )
        else:
            out.append({"role": m["role"], "content": m.get("content", "")})
    return out


def _args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        out = json.loads(raw) if raw and str(raw).strip() else {}
    except ValueError:
        return {"_unparsed": str(raw)[:2000]}
    return out if isinstance(out, dict) else {"_unparsed": str(raw)[:2000]}


def explain(status: int, s: dict | None = None, sent: bool = True) -> str:
    """An HTTP refusal in words; never the key. A refusal of a request that carried no key (SENT false: its binding
    names none, as an ollama binding, which has no key settings, never does) says so; a refused key is named by the
    settings S, the binding's, take it and its headers from: with the Claude session's key headers (key_headers, which
    rkmon_lib/bindings.py gives a claude binding that has no key of its own), the claude login's key, sent as
    integrations.claude_code's settings say; else the key the binding's own settings name, sent as they say.
    """
    if status in (401, 403) and not sent:
        return "the endpoint refused the request, which carried no key: the binding names none"
    if status in (401, 403):
        whose, how = (
            ("the claude login's", "integrations.claude_code's")
            if (s or {}).get("key_headers")
            else ("the binding's api_key_secret_name", "its")
        )
        return (
            f"the endpoint refused the key ({whose}, sent as {how} api_key_sent_as, api_key_header and "
            "api_key_scheme say)"
        )
    if status == 404:
        return "no such route: check the binding's url and chat_path"
    return (
        "the endpoint failed" if status >= 500 else "the endpoint refused the request"
    )


class OpenAICompatible:
    """An OpenAI-compatible chat completions route."""

    name = "openai"
    default_path = "/chat/completions"

    def route(self, s: dict) -> tuple[str, str, int, str]:
        """(scheme, host, port, path): the base URL's own path (base_url: a `$NAME` url is read here), then
        chat_path."""
        u = urllib.parse.urlsplit(base_url(s))
        path = (
            u.path.rstrip("/")
            + "/"
            + (s.get("chat_path") or self.default_path).lstrip("/")
        )
        return (
            u.scheme,
            u.hostname or "",
            u.port or (443 if u.scheme == "https" else 80),
            path,
        )

    def headers(self, s: dict, key: str) -> dict:
        """Every request's headers but the content type: the key where the binding says."""
        return auth_headers(s, key)

    def models_urls(self, s: dict) -> list[str]:
        """Where the endpoint lists its models, in the order a read-only check tries them."""
        base = base_url(s)
        return [f"{base}/models", f"{base}/v1/models"]

    def recover(self, body: dict, error: str) -> bool:
        """After an HTTP 400: change the request so it may pass once more (True), or leave it (False). An endpoint
        that refuses the usage option streams without it."""
        if "stream_options" in error and "stream_options" in body:
            body.pop("stream_options")
            return True
        return False

    def body(
        self,
        s: dict,
        messages: list[dict],
        tools: list | None,
        effort: str | None,
        max_tokens: int | None,
    ) -> dict:
        body = {
            "model": s["model"],
            "messages": to_openai(messages),
            "temperature": float(s["temperature"]),
            "max_tokens": int(max_tokens or s["max_output_tokens"]),
        }
        if s.get("stream", True):
            body["stream"] = True
            if s.get("stream_usage", True):
                body["stream_options"] = {"include_usage": True}
        if tools:
            body["tools"], body["tool_choice"] = tools, "auto"
        if effort in EFFORTS:
            body["reasoning_effort"] = effort
        return body

    def post(self, s: dict, body: dict, key: str, deadline: float = 0.0) -> dict:
        """One POST; the neutral answer, or RateLimited / GatewayError. `deadline` (a time.time(); 0 for none): the
        socket waits no longer, and a stream still running then is cut (OverBudget)."""
        scheme, host, port, path = self.route(s)
        cls = (
            http.client.HTTPSConnection
            if scheme == "https"
            else http.client.HTTPConnection
        )
        timeout = float(s.get("stall_s") or 300)
        if deadline:
            left = deadline - time.time()
            if left <= 0:
                raise OverBudget(
                    "wall",
                    "the request's deadline passed before it was sent",
                    scope="phase",
                )
            timeout = max(1.0, min(timeout, left))
        if not host:
            raise GatewayError(
                "the binding's url is blank"
                + (
                    f" (the variable {s.get('url')} is not set)"
                    if ENV_REF.fullmatch(str(s.get("url") or ""))
                    else ""
                ),
                0,
            )
        conn = cls(host, port, timeout=timeout)
        hdrs = self.headers(s, key)
        hide = [
            key,
            *(v for k, v in hdrs.items() if k.lower() not in ("anthropic-version",)),
        ]
        try:
            conn.request(
                "POST",
                path,
                body=json.dumps(body).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream, application/json",
                    **hdrs,
                },
            )
            r = conn.getresponse()
            if r.status == 429:
                raise RateLimited(
                    f"HTTP 429: {scrub(r.read(300).decode('utf-8', 'replace'), hide)}",
                    wait_from_headers(r.getheaders()) or 0.0,
                )
            if r.status != 200:
                text = scrub(r.read(400).decode("utf-8", "replace"), hide)
                raise GatewayError(
                    f"HTTP {r.status}: {explain(r.status, s, sent=bool(key))}{': ' + text if text else ''}",
                    r.status,
                    transient=r.status >= 500,
                )
            if "event-stream" in (r.getheader("Content-Type") or "").lower():
                return self.stream_answer(r, deadline)
            data = r.read()
            if data.lstrip().startswith((b"data:", b"event:")):
                return self.stream_answer(data.splitlines())
            return self.whole_answer(data)
        finally:
            conn.close()

    @staticmethod
    def stream_answer(lines, deadline: float = 0.0) -> dict:
        """Server-sent chat completion chunks joined into the neutral answer; OverBudget once `deadline` (a
        time.time()) passes while it streams."""
        msg = {
            "role": "assistant",
            "content": "",
            "reasoning": "",
            "tool_calls": [],
            "finish_reason": "",
            "usage": None,
        }
        calls: dict[int, dict] = {}
        seen = False
        for raw in lines:
            if deadline and time.time() > deadline:
                raise OverBudget(
                    "wall",
                    "the answer was still streaming at its deadline",
                    scope="phase",
                )
            line = (
                raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
            ).strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            if not isinstance(chunk, dict):
                continue
            if chunk.get("error"):
                raise GatewayError(
                    f"the model server: {str(chunk['error'])[:300]}", 0, transient=True
                )
            if isinstance(chunk.get("usage"), dict):
                msg["usage"] = chunk["usage"]
            for choice in chunk.get("choices") or []:
                seen = True
                d = choice.get("delta") or choice.get("message") or {}
                msg["content"] += d.get("content") or ""
                msg["reasoning"] += (
                    d.get("reasoning") or d.get("reasoning_content") or ""
                )
                msg["finish_reason"] = (
                    choice.get("finish_reason") or msg["finish_reason"]
                )
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(
                        tc.get("index", len(calls)),
                        {"id": "", "name": "", "arguments": ""},
                    )
                    c["id"] = tc.get("id") or c["id"]
                    fn = tc.get("function") or {}
                    c["name"] += fn.get("name") or ""
                    args = fn.get("arguments") or ""
                    c["arguments"] += (
                        args if isinstance(args, str) else json.dumps(args)
                    )
        if not seen:
            raise GatewayError(
                "the answer has no choices: not a chat completion", 0, transient=True
            )
        for _i, c in sorted(calls.items()):
            msg["tool_calls"].append(
                {
                    "id": c["id"],
                    "function": {"name": c["name"], "arguments": _args(c["arguments"])},
                }
            )
        return msg

    @staticmethod
    def whole_answer(data: bytes) -> dict:
        """A chat completion sent whole (JSON) in the neutral form."""
        try:
            body = json.loads(data or b"{}")
        except ValueError:
            raise GatewayError(
                "the answer is neither a stream nor JSON", 0, transient=True
            ) from None
        if (
            not isinstance(body, dict)
            or not isinstance(body.get("choices"), list)
            or not body["choices"]
        ):
            if isinstance(body, dict) and body.get("error"):
                raise GatewayError(
                    f"the model server: {str(body['error'])[:300]}", 0, transient=True
                )
            raise GatewayError(
                "the answer has no choices: not a chat completion", 0, transient=True
            )
        choice = body["choices"][0]
        m = choice.get("message") or {}
        calls = []
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            calls.append(
                {
                    "id": tc.get("id") or "",
                    "function": {
                        "name": fn.get("name") or "",
                        "arguments": _args(fn.get("arguments")),
                    },
                }
            )
        return {
            "role": "assistant",
            "content": m.get("content") or "",
            "reasoning": m.get("reasoning") or m.get("reasoning_content") or "",
            "tool_calls": calls,
            "finish_reason": choice.get("finish_reason") or "",
            "usage": body.get("usage") if isinstance(body.get("usage"), dict) else None,
        }


ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_FINISH = {
    "tool_use": "tool_calls",
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
}
_TOOL_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _tool_id(raw, k: int) -> str:
    return str(raw) if raw and _TOOL_ID.fullmatch(str(raw)) else f"toolu_rk{k}"


def to_anthropic_tool(t: dict) -> dict:
    """A tool in the Messages API's form: an OpenAI-style function tool's name, description and parameters (its
    input_schema); a tool already in that form as it is."""
    fn = t.get("function") if isinstance(t.get("function"), dict) else None
    if not fn:
        return t
    return {
        "name": fn.get("name") or "",
        "description": fn.get("description") or "",
        "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
    }


def to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """The neutral messages in the Messages API's form: (the system text, the messages). An assistant message that
    carries the answer's own `blocks` (thinking included) sends them unchanged; else its text and tool calls become
    text and tool_use blocks. Each tool result answers the next open tool_use of the turn before it, in order;
    consecutive messages of one role join into one."""
    system, out, pending, k = [], [], [], 0

    def put(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for m in messages:
        role, text = m.get("role"), m.get("content")
        text = text if isinstance(text, str) else json.dumps(text) if text else ""
        if role == "system":
            system.append(text)
        elif role == "assistant":
            raw = m.get("blocks")
            if isinstance(raw, list) and raw:
                blocks = [dict(b) for b in raw if isinstance(b, dict)]
            else:
                blocks = [{"type": "text", "text": text}] if text.strip() else []
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": _tool_id(tc.get("id"), k),
                            "name": fn.get("name") or "",
                            "input": _args(fn.get("arguments")),
                        }
                    )
                    k += 1
            pending = [
                b["id"] for b in blocks if b.get("type") == "tool_use" and b.get("id")
            ]
            put("assistant", blocks)
        elif role == "tool":
            if pending:
                put(
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": pending.pop(0),
                            "content": text or "(empty)",
                        }
                    ],
                )
            else:
                put("user", [{"type": "text", "text": f"(tool result) {text}"}])
        elif text.strip():
            put("user", [{"type": "text", "text": text}])
    if not out or out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": [{"type": "text", "text": "Begin."}]})
    return "\n\n".join(x for x in system if x.strip()), out


def anthropic_usage(u: dict) -> dict:
    """The Messages API's usage with the keys usage_of reads: the prompt is the input, the cache reads and the cache
    writes together; the cached part is the cache reads."""
    u = u if isinstance(u, dict) else {}
    cr, cw = _num(u.get("cache_read_input_tokens")), _num(
        u.get("cache_creation_input_tokens")
    )
    return dict(
        u,
        prompt_tokens=_num(u.get("input_tokens")) + cr + cw,
        completion_tokens=_num(u.get("output_tokens")),
        prompt_tokens_details={"cached_tokens": cr},
    )


def _neutral(blocks: list[dict], stop: str, usage: dict | None) -> dict:
    calls = [
        {
            "id": b.get("id") or "",
            "function": {
                "name": b.get("name") or "",
                "arguments": _args(b.get("input")),
            },
        }
        for b in blocks
        if b.get("type") == "tool_use"
    ]
    return {
        "role": "assistant",
        "content": "".join(
            b.get("text") or "" for b in blocks if b.get("type") == "text"
        ),
        "reasoning": "".join(
            b.get("thinking") or "" for b in blocks if b.get("type") == "thinking"
        ),
        "tool_calls": calls,
        "finish_reason": ANTHROPIC_FINISH.get(stop or "", stop or ""),
        "usage": anthropic_usage(usage) if usage else None,
        "blocks": blocks,
    }


class AnthropicMessages(OpenAICompatible):
    """The Anthropic Messages API (/v1/messages under the base URL, the route Claude Code calls under
    ANTHROPIC_BASE_URL): the key in x-api-key by default (auth_header Authorization with scheme Bearer for a gateway
    that takes a bearer token), and anthropic-version. Tools go as name, description and input_schema with tool_choice
    auto; the effort as output_config.effort; no temperature and no thinking setting (the model's own defaults: current
    models refuse a sampling setting). The answer's blocks ride on the neutral answer (`blocks`), so a caller that sends
    an assistant turn back with them keeps its thinking blocks unchanged.
    """

    name = "anthropic"
    default_path = "/v1/messages"

    def headers(self, s: dict, key: str) -> dict:
        auth = dict(
            s,
            auth_header=s.get("auth_header") or "x-api-key",
            auth_scheme=s["auth_scheme"] if "auth_scheme" in s else "",
        )
        return {
            "anthropic-version": ANTHROPIC_VERSION,
            **auth_headers(auth, key),
        }

    def models_urls(self, s: dict) -> list[str]:
        return [f"{base_url(s)}/v1/models"]

    def body(
        self,
        s: dict,
        messages: list[dict],
        tools: list | None,
        effort: str | None,
        max_tokens: int | None,
    ) -> dict:
        system, msgs = to_anthropic(messages)
        body = {
            "model": s["model"],
            "max_tokens": int(max_tokens or s["max_output_tokens"]),
            "messages": msgs,
        }
        if system:
            body["system"] = system
        if s.get("stream", True):
            body["stream"] = True
        if tools:
            body["tools"], body["tool_choice"] = [
                to_anthropic_tool(t) for t in tools
            ], {"type": "auto"}
        if effort in CLAUDE_EFFORTS:
            body["output_config"] = {"effort": effort}
        return body

    def recover(self, body: dict, error: str) -> bool:
        """A turn sent back with thinking blocks the endpoint refuses (the history around them was edited, as the
        drafter's kept tool results are): send it once more without them."""
        if "thinking" not in error.lower():
            return False
        changed = False
        for m in body.get("messages") or []:
            if m.get("role") == "assistant" and isinstance(m.get("content"), list):
                kept = [
                    b
                    for b in m["content"]
                    if b.get("type") not in ("thinking", "redacted_thinking")
                ]
                changed |= len(kept) != len(m["content"])
                m["content"] = kept or [{"type": "text", "text": "(no text)"}]
        return changed

    @staticmethod
    def stream_answer(lines, deadline: float = 0.0) -> dict:
        """The Messages API's server-sent events joined into the neutral answer; OverBudget once `deadline` passes."""
        blocks: dict[int, dict] = {}
        usage: dict = {}
        stop, seen = "", False
        for raw in lines:
            if deadline and time.time() > deadline:
                raise OverBudget(
                    "wall",
                    "the answer was still streaming at its deadline",
                    scope="phase",
                )
            line = (
                raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
            ).strip()
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except ValueError:
                continue
            if not isinstance(ev, dict):
                continue
            kind = ev.get("type")
            if kind == "error":
                err = ev.get("error") or {}
                raise GatewayError(
                    f"the model server: {str(err.get('message') or err)[:300]}",
                    0,
                    transient=err.get("type") in ("overloaded_error", "api_error"),
                )
            if kind == "message_start":
                seen = True
                usage.update((ev.get("message") or {}).get("usage") or {})
            elif kind == "content_block_start":
                b = dict(ev.get("content_block") or {})
                if b.get("type") == "tool_use":
                    b["_json"] = ""
                blocks[int(ev.get("index", len(blocks)))] = b
            elif kind == "content_block_delta":
                b = blocks.setdefault(
                    int(ev.get("index", 0)), {"type": "text", "text": ""}
                )
                d = ev.get("delta") or {}
                if d.get("type") == "text_delta":
                    b["text"] = (b.get("text") or "") + (d.get("text") or "")
                elif d.get("type") == "input_json_delta":
                    b["_json"] = (b.get("_json") or "") + (d.get("partial_json") or "")
                elif d.get("type") == "thinking_delta":
                    b["thinking"] = (b.get("thinking") or "") + (
                        d.get("thinking") or ""
                    )
                elif d.get("type") == "signature_delta":
                    b["signature"] = (b.get("signature") or "") + (
                        d.get("signature") or ""
                    )
            elif kind == "message_delta":
                stop = (ev.get("delta") or {}).get("stop_reason") or stop
                usage.update(ev.get("usage") or {})
            elif kind == "message_stop":
                break
        if not seen:
            raise GatewayError(
                "the answer is not a Messages API stream (no message_start)",
                0,
                transient=True,
            )
        out = []
        for _i, b in sorted(blocks.items()):
            if b.get("type") == "tool_use":
                js = b.pop("_json", "")
                b["input"] = _args(js) if js else (b.get("input") or {})
            out.append(b)
        return _neutral(out, stop, usage)

    @staticmethod
    def whole_answer(data: bytes) -> dict:
        try:
            body = json.loads(data or b"{}")
        except ValueError:
            raise GatewayError(
                "the answer is neither a stream nor JSON", 0, transient=True
            ) from None
        if isinstance(body, dict) and body.get("type") == "error":
            err = body.get("error") or {}
            raise GatewayError(
                f"the model server: {str(err.get('message') or err)[:300]}",
                0,
                transient=err.get("type") in ("overloaded_error", "api_error"),
            )
        if not isinstance(body, dict) or not isinstance(body.get("content"), list):
            raise GatewayError(
                "the answer has no content: not a Messages API answer",
                0,
                transient=True,
            )
        return _neutral(
            [b for b in body["content"] if isinstance(b, dict)],
            body.get("stop_reason") or "",
            body.get("usage"),
        )


PROVIDERS = {"openai": OpenAICompatible(), "anthropic": AnthropicMessages()}


def adapter(s: dict):
    """The provider adapter a binding names; GatewayError for an unknown one."""
    name = str(s.get("provider") or s.get("backend") or "openai")
    if name not in PROVIDERS:
        raise GatewayError(
            f"no provider adapter {name!r}; the adapters are {', '.join(PROVIDERS)}", 0
        )
    return PROVIDERS[name]


# --- the limiter ------------------------------------------------------------------------------------------------------


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError, OverflowError, OSError):
        return False
    return True


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _blank() -> dict:
    return {
        "window": [],
        "blocked_until": 0.0,
        "queue": [],
        "callers": {},
        "history": [],
        "r429": [],
        "waits": [],
        "inflight": {},
    }


def _load(raw: bytes) -> dict:
    try:
        st = json.loads(raw) if raw.strip() else {}
    except ValueError:
        st = {}
    out = _blank()
    if isinstance(st, dict):
        for k, v in st.items():
            if k in out and v is not None:
                out[k] = v
    out["blocked_until"] = float(out["blocked_until"] or 0.0)
    return out


def _locked(path: pathlib.Path, fn):
    """fn(state) with the bucket file locked; the state is written back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        st = _load(os.read(fd, 1 << 22))
        out = fn(st)
        data = json.dumps(st).encode()
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, data)
        return out
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def bucket(s: dict, key: str | None = None) -> str:
    """The bucket of a binding: one per provider, key and model."""
    key = api_key(s) if key is None else key
    raw = f"{s.get('provider') or s.get('backend') or 'openai'}|{key_fingerprint(key)}|{s.get('model') or ''}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def bucket_path(s: dict, key: str | None = None) -> pathlib.Path | None:
    d = s.get("limiter_dir")
    return (
        pathlib.Path(os.path.expanduser(d)) / f"bucket-{bucket(s, key)}.json"
        if d
        else None
    )


class Limiter:
    """An endpoint's limits shared through its bucket file, and one run's budget, cap and counts. Thread-safe: a run's
    threads share it, each with its own request ticket, tags, progress and phase."""

    def __init__(
        self,
        s: dict,
        who: str = "",
        budget_tokens: int = 0,
        wall_s: float = 0,
        prio: bool = False,
        clock=time.time,
        sleep=time.sleep,
        log=None,
    ):
        self.rpm, self.tpm = int(s.get("rpm") or 0), int(s.get("tpm") or 0)
        self.inflight_max = int(s.get("concurrency") or 0)
        self.reserve = min(0.9, max(0.0, float(s["reserve_pct"]) / 100.0))
        self.path = bucket_path(s)
        self.dir = self.path.parent if self.path else None
        self.bucket = bucket(s)
        self.model, self.who = str(s.get("model") or ""), who or "-"
        self.clock, self.sleep, self.log = clock, sleep, log
        self.budget = int(budget_tokens or 0)
        self.t0 = clock()
        self.deadline = self.t0 + float(wall_s) if wall_s else 0.0
        self.prio = bool(prio)
        self.key = f"{os.getpid()}.{id(self)}"
        self._mem = _blank()
        self._lock = (
            threading.Lock()
        )  # the run's counts, and the bucket kept in memory when there is no file
        self._tl = threading.local()  # each thread's ticket, tags, progress and phase
        self._count = 0
        self._progress = 0.0
        self.totals = {
            "requests": 0,
            "prompt": 0,
            "cached": 0,
            "completion": 0,
            "reasoning": 0,
            "waited_s": 0.0,
            "rate_limited": 0,
            "estimated": 0,
        }
        self.usage_keys: set[str] = set()
        self.passes: dict[str, dict] = {}

    # --- each thread's own: the tags its token lines carry, its progress (the rank), its phase ------------------------
    @property
    def tags(self) -> dict:
        """The pass and step (and task) this thread's token lines name."""
        return getattr(self._tl, "tags", {})

    @tags.setter
    def tags(self, value: dict) -> None:
        self._tl.tags = dict(value or {})

    @property
    def progress(self) -> float:
        """How close this thread's work is to done (0 to 1): the rank's second key."""
        return getattr(self._tl, "progress", self._progress)

    @progress.setter
    def progress(self, value: float) -> None:
        self._tl.progress = self._progress = float(value)

    def phase(self, name: str = "", wall_s: float = 0.0, tokens: int = 0) -> None:
        """This thread's phase from now on: its requests end within `wall_s` seconds (and within the run's cap), and
        the requests of every thread in a phase of that name stay within `tokens` (0: no cap of its own). No name: the
        run's caps only."""
        self._tl.phase = (
            (
                name,
                self.clock() + float(wall_s) if wall_s else 0.0,
                float(wall_s),
                int(tokens or 0),
            )
            if name
            else None
        )

    def phase_used(self, name: str) -> int:
        """The tokens a phase's requests used so far (the pass its token lines name)."""
        with self._lock:
            p = self.passes.get(name) or {}
            return int(p.get("prompt", 0)) + int(p.get("completion", 0))

    def deadline_now(self) -> float:
        """When this thread's requests must end (0: never): the run's cap or its phase's deadline, the sooner."""
        ph = getattr(self._tl, "phase", None)
        ends = [d for d in (self.deadline, ph[1] if ph else 0.0) if d]
        return min(ends) if ends else 0.0

    def rank(self) -> list:
        """Lower goes first: a priority's work, then the caller closest to done."""
        return [0 if self.prio else 1, -round(min(1.0, max(0.0, self.progress)), 3)]

    def _update(self, fn):
        if self.path:
            return _locked(self.path, fn)
        with self._lock:
            return fn(self._mem)

    def _say(self, text: str) -> None:
        if self.log:
            self.log(text)
        if self.dir:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                with open(self.dir / LIMITER_LOG, "a", encoding="utf-8") as f:
                    f.write(f"{_now_iso()} {self.bucket} {self.who} {text}\n")
            except OSError:
                pass

    def used(self) -> int:
        return self.totals["prompt"] + self.totals["completion"]

    def elapsed(self) -> float:
        return self.clock() - self.t0

    def check_caps(self, tokens: int) -> None:
        """OverBudget when a request of about `tokens` would pass the run's token budget or this thread's phase's, or
        the run's or the phase's time is up."""
        now = self.clock()
        ph = getattr(self._tl, "phase", None)
        if self.deadline and now >= self.deadline:
            raise OverBudget(
                "wall",
                f"the run's wall-time cap ({(self.deadline - self.t0) / 60:.0f} min) is reached",
            )
        if ph and ph[1] and now >= ph[1]:
            span = f"{ph[2] / 60:.0f} min" if ph[2] >= 60 else f"{ph[2]:.0f} s"
            raise OverBudget(
                "wall",
                f"the {ph[0]} phase's wall time ({span}) is spent",
                scope="phase",
            )
        if self.budget and self.used() + tokens > self.budget:
            raise OverBudget(
                "tokens",
                f"the run's token budget ({self.budget}) would be passed: {self.used()} used, "
                f"the next request about {tokens}",
            )
        if ph and ph[3] and self.phase_used(ph[0]) + tokens > ph[3]:
            raise OverBudget(
                "tokens",
                f"the {ph[0]} phase's token budget ({ph[3]}) would be passed: "
                f"{self.phase_used(ph[0])} used, the next request about {tokens}",
                scope="phase",
            )

    def acquire(self, tokens: int) -> float:
        """Wait until a request of about `tokens` fits the shared limits, take its place in the window (and a place in
        flight), and return the seconds waited. The best-ranked waiter goes first (the oldest among equals); the last
        `reserve` of a minute's budget goes only to the best-ranked active caller; a request larger than the whole
        tokens limit goes once the window is empty. Raises OverBudget (the run's caps, this thread's phase).
        """
        with self._lock:
            self._count += 1
            ticket = self._tl.ticket = f"{self.key}.{self._count}"
        pid, since = os.getpid(), self.clock()
        waited, said = 0.0, ""

        def leave(st):
            st["queue"] = [q for q in st["queue"] if q[0] != ticket]

        try:
            while True:
                self.check_caps(tokens)
                now, rank = self.clock(), self.rank()

                def take(st):
                    st["window"] = [
                        w for w in st["window"] if now - w[0] < WINDOW_S + SLACK_S
                    ]
                    st["callers"] = {
                        k: c
                        for k, c in (st["callers"] or {}).items()
                        if now - float(c.get("seen") or 0) < CALLER_TTL_S
                        and _alive(c.get("pid"))
                    }
                    st["callers"][self.key] = {
                        "pid": pid,
                        "rank": rank,
                        "seen": now,
                        "who": self.who,
                    }
                    st["inflight"] = flying = {
                        k: v
                        for k, v in (st.get("inflight") or {}).items()
                        if now - float(v[1]) < INFLIGHT_TTL_S and _alive(v[0])
                    }
                    queue = [
                        q
                        for q in st["queue"]
                        if q[0] != ticket and now - q[1] < QUEUE_TTL_S and _alive(q[2])
                    ]
                    st["queue"] = queue = queue + [[ticket, since, pid, rank]]
                    order = sorted(
                        queue, key=lambda q: (q[3] if len(q) > 3 else [1, 0], q[1])
                    )
                    if st["blocked_until"] > now:
                        return (
                            st["blocked_until"] - now,
                            "the endpoint asked us to wait (HTTP 429)",
                        )
                    free_at = (
                        min((w[0] for w in st["window"]), default=now)
                        + WINDOW_S
                        + SLACK_S
                    )
                    if self.rpm and len(st["window"]) >= self.rpm:
                        return (
                            free_at - now,
                            f"{len(st['window'])} of {self.rpm} requests in the last minute",
                        )
                    if order[0][0] != ticket:
                        ahead = [q[0] for q in order].index(ticket)
                        return QUEUE_POLL_S, f"{ahead} better-ranked caller(s) go first"
                    if self.inflight_max and len(flying) >= self.inflight_max:
                        return (
                            QUEUE_POLL_S,
                            f"{len(flying)} of {self.inflight_max} requests in flight",
                        )
                    used = sum(w[1] for w in st["window"])
                    if self.tpm and st["window"] and used + tokens > self.tpm:
                        return (
                            free_at - now,
                            f"{used} + {tokens} of {self.tpm} tokens in the last minute",
                        )
                    best = min(c["rank"] for c in st["callers"].values())
                    low = (
                        self.tpm and self.tpm - used - tokens < self.reserve * self.tpm
                    ) or (
                        self.rpm
                        and self.rpm - len(st["window"]) - 1 < self.reserve * self.rpm
                    )
                    if low and st["window"] and rank > best:
                        return min(free_at - now, 5.0), (
                            "the minute's last share is kept for the best-ranked caller "
                            "(a priority, or the one closest to done)"
                        )
                    st["window"].append([now, tokens, ticket])
                    flying[ticket] = [pid, now]
                    leave(st)
                    return 0.0, ""

                wait, why = self._update(take)
                if wait <= 0:
                    self._note_wait(waited)
                    return waited
                wait = max(0.2, min(wait, WINDOW_S))
                end = self.deadline_now()
                if end:
                    wait = max(0.2, min(wait, end - self.clock() + 0.01))
                if why != said:
                    self._say(f"waiting {wait:.0f} s: {why}")
                    said = why
                self.sleep(wait)
                waited += wait
        except BaseException:
            self._note_wait(waited)
            self._update(leave)
            raise

    def _note_wait(self, waited: float) -> None:
        with self._lock:
            self.totals["waited_s"] = round(self.totals["waited_s"] + waited, 2)
        if waited > 0:
            now = self.clock()

            def note(st):
                st["waits"] = [w for w in st["waits"] if now - w[0] < HISTORY_S] + [
                    [now, round(waited, 2)]
                ]

            self._update(note)

    def settle(self, tokens: int) -> None:
        """The request this thread just made counts `tokens` in the window and the history (what the endpoint
        reported, prompt and answer; 0 for a request it refused or that was cut), and leaves the requests in flight.
        """
        now = self.clock()
        ticket = getattr(self._tl, "ticket", "")

        def put(st):
            for w in st["window"]:
                if len(w) > 2 and w[2] == ticket:
                    w[1] = int(tokens)
            (st.get("inflight") or {}).pop(ticket, None)
            st["history"] = [h for h in st["history"] if now - h[0] < HISTORY_S] + [
                [now, int(tokens)]
            ]

        self._update(put)

    def block(self, seconds: float, why: str) -> None:
        """Every caller of the bucket waits `seconds` from now (an HTTP 429), and the 429 joins the history."""
        now = self.clock()
        until = now + max(0.0, seconds)

        def set_until(st):
            st["blocked_until"] = max(st["blocked_until"], until)
            st["r429"] = [t for t in st["r429"] if now - t < HISTORY_S] + [now]

        self._update(set_until)
        self._say(f"{why}; every caller waits {seconds:.0f} s")

    def close(self) -> None:
        """This caller is done: it leaves the active callers."""

        def drop(st):
            st["callers"].pop(self.key, None)

        try:
            self._update(drop)
        except OSError:
            pass

    def count(self, use: dict, waited: float, elapsed: float) -> None:
        """One answered request: the run's totals, its pass's, and a line of tokens.jsonl."""
        tags = self.tags
        pass_ = str(tags.get("pass") or "-")
        with self._lock:
            for tot in (
                self.totals,
                self.passes.setdefault(
                    pass_,
                    {
                        "requests": 0,
                        "prompt": 0,
                        "cached": 0,
                        "completion": 0,
                        "reasoning": 0,
                        "waited_s": 0.0,
                    },
                ),
            ):
                tot["requests"] += 1
                for k in ("prompt", "cached", "completion", "reasoning"):
                    tot[k] += int(use.get(k) or 0)
            self.passes[pass_]["waited_s"] = round(
                self.passes[pass_]["waited_s"] + waited, 2
            )
            if not use.get("reported"):
                self.totals["estimated"] += 1
            self.usage_keys |= set(use.get("usage_keys") or [])
        if self.dir:
            rec = {
                "ts": _now_iso(),
                "t": round(self.clock(), 2),
                "bucket": self.bucket,
                "who": self.who,
                "model": self.model,
                **tags,
                "prompt": use["prompt"],
                "cached": use.get("cached", 0),
                "completion": use["completion"],
                "reasoning": use["reasoning"],
                "reported": use["reported"],
                "waited_s": round(waited, 2),
                "elapsed_s": round(elapsed, 2),
            }
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                with self._lock, open(
                    self.dir / TOKENS_LOG, "a", encoding="utf-8"
                ) as f:
                    f.write(json.dumps(rec) + "\n")
            except OSError:
                pass

    def rate_limited(self) -> None:
        """One HTTP 429 in the run's counts."""
        with self._lock:
            self.totals["rate_limited"] += 1

    def summary(self) -> dict:
        """The run's counts: requests, tokens (prompt + completion), prompt, cached, completion and reasoning tokens,
        seconds waited, 429s, requests whose tokens were estimated, wall seconds, the usage fields the provider sent,
        and the same counts per pass."""
        with self._lock:
            t = dict(self.totals)
            passes = {
                k: dict(v, tokens=v["prompt"] + v["completion"])
                for k, v in self.passes.items()
            }
            keys = sorted(self.usage_keys)
        return {
            "requests": t["requests"],
            "tokens": t["prompt"] + t["completion"],
            "prompt_tokens": t["prompt"],
            "cached_tokens": t["cached"],
            "completion_tokens": t["completion"],
            "reasoning_tokens": t["reasoning"],
            "waits_s": t["waited_s"],
            "rate_limited": t["rate_limited"],
            "estimated": t["estimated"],
            "wall_s": round(self.elapsed(), 1),
            "usage_keys": keys,
            "passes": passes,
        }


def limiter_for(
    s: dict, who: str = "", run: bool = True, prio: bool = False, log=None
) -> Limiter:
    """A caller's limiter: a run's (its budget and cap from the settings), or one with no caps."""
    return Limiter(
        s,
        who=who,
        budget_tokens=int(s.get("review_token_budget") or 0) if run else 0,
        wall_s=float(s.get("review_wall_min") or 0) * 60 if run else 0,
        prio=prio,
        log=log,
    )


def window_stats(s: dict, minutes: float = 5.0, now: float | None = None) -> dict:
    """A binding's bucket over the last `minutes` (at most HISTORY_S): the tokens and requests answered, their shares of
    the budget over that time and the larger of the two (`share`), the seconds waited, the HTTP 429s, and the callers
    active and the requests in flight now. All zeros without a bucket file."""
    span = min(float(minutes) * 60.0, HISTORY_S)
    now = time.time() if now is None else now
    out = {
        "minutes": span / 60.0,
        "tokens": 0,
        "requests": 0,
        "share_tokens": 0.0,
        "share_requests": 0.0,
        "share": 0.0,
        "waited_s": 0.0,
        "rate_limited": 0,
        "active": 0,
        "inflight": 0,
    }
    path = bucket_path(s)
    if not path or not path.is_file():
        return out
    try:
        st = _load(path.read_bytes())
    except OSError:
        return out
    hist = [h for h in st["history"] if now - h[0] < span]
    out["tokens"] = sum(int(h[1]) for h in hist)
    out["requests"] = len(hist)
    tpm, rpm = int(s.get("tpm") or 0), int(s.get("rpm") or 0)
    mins = span / 60.0
    out["share_tokens"] = round(out["tokens"] / (tpm * mins), 3) if tpm else 0.0
    out["share_requests"] = round(out["requests"] / (rpm * mins), 3) if rpm else 0.0
    out["share"] = max(out["share_tokens"], out["share_requests"])
    out["waited_s"] = round(
        sum(float(w[1]) for w in st["waits"] if now - w[0] < span), 1
    )
    out["rate_limited"] = sum(1 for t in st["r429"] if now - t < span)
    out["active"] = sum(
        1
        for c in (st["callers"] or {}).values()
        if now - float(c.get("seen") or 0) < CALLER_TTL_S and _alive(c.get("pid"))
    )
    out["inflight"] = sum(
        1
        for v in (st.get("inflight") or {}).values()
        if now - float(v[1]) < INFLIGHT_TTL_S and _alive(v[0])
    )
    return out


# --- one chat request -------------------------------------------------------------------------------------------------


def chat(
    s: dict,
    messages: list[dict],
    tools: list | None = None,
    effort: str | None = None,
    limiter: Limiter | None = None,
    max_tokens: int | None = None,
    post=None,
    sleep=time.sleep,
) -> dict:
    """One chat request through the limiter and the binding's provider adapter: an HTTP 429 blocks every caller of the
    bucket for its Retry-After (else the back-off) and is retried up to MAX_RATE_WAITS times; a failure that a resend
    may cure (5xx, a dropped connection) is resent up to max_resends times, unless the caller's time is up. The adapter's
    request ends by the calling thread's deadline (Limiter.deadline_now). Raises OverBudget, RateLimited or
    GatewayError. The answer's `usage` is the request's tokens (`usage_of`)."""
    limiter = limiter or limiter_for(s, run=False)
    ad = adapter(s)
    key = api_key(s)
    if s.get("api_key_file") and not key:
        raise GatewayError(
            f"no key in the binding's api_key_file ({s.get('api_key_file')})", 0
        )
    body = ad.body(s, messages, tools, effort, max_tokens)
    if post is None:

        def post(s_, body_, key_):
            return ad.post(s_, body_, key_, deadline=limiter.deadline_now())

    fails, limited = 0, 0
    while True:
        sent = estimate_tokens(body["messages"], body.get("tools"), body.get("system"))
        waited = limiter.acquire(sent)
        t0 = time.monotonic()
        try:
            msg = post(s, body, key)
        except RateLimited as exc:
            limiter.settle(0)
            limited += 1
            limiter.rate_limited()
            if limited > MAX_RATE_WAITS:
                raise
            wait = exc.wait_s or min(BACKOFF_MAX_S, BACKOFF_S * 2 ** (limited - 1))
            limiter.block(wait, f"HTTP 429 (retry {limited} of {MAX_RATE_WAITS})")
            continue
        except GatewayError as exc:
            limiter.settle(0)
            if exc.status == 400 and ad.recover(body, str(exc)):
                continue  # the adapter changed the request (OpenAICompatible.recover, AnthropicMessages.recover)
            fails += 1
            if not exc.transient or fails > int(s["max_resends"]):
                raise
            limiter.check_caps(0)
            sleep(
                min(
                    BACKOFF_MAX_S,
                    float(s.get("resend_wait_s", BACKOFF_S)) * 2 ** (fails - 1),
                )
            )
            continue
        except OverBudget:
            limiter.settle(
                0
            )  # cut at the deadline: the run's cap or this thread's phase says which
            limiter.check_caps(0)
            raise
        except (OSError, http.client.HTTPException, socket.timeout) as exc:
            limiter.settle(0)
            limiter.check_caps(
                0
            )  # a socket that waited until the deadline: the time is up, not the endpoint
            fails += 1
            if fails > int(s["max_resends"]):
                raise GatewayError(
                    f"no answer: {type(exc).__name__}: {scrub(str(exc), key)[:200]}",
                    0,
                    transient=True,
                ) from None
            sleep(
                min(
                    BACKOFF_MAX_S,
                    float(s.get("resend_wait_s", BACKOFF_S)) * 2 ** (fails - 1),
                )
            )
            continue
        except BaseException:
            limiter.settle(0)
            raise
        use = usage_of(msg, sent)
        limiter.settle(use["total"])
        limiter.count(use, waited, time.monotonic() - t0)
        msg["usage"] = use
        return msg
