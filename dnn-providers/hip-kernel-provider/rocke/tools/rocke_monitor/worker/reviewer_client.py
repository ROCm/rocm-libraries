# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The reviewer model's client: one `chat` call for two APIs (config worker.backend).

- ollama: Ollama's /api/chat, streamed as JSON lines (ollama_stream.stream_chat, which also notices a wedged runner
  of a local Ollama through its service journal);
- openai: an OpenAI-compatible chat completions route, streamed as server-sent events (vLLM, TGI, LiteLLM, an on-prem
  gateway). The base URL may carry a path prefix (a gateway's route) and may end in /v1; /v1 is never doubled. The chat
  route under it is `chat_path` (/v1/chat/completions by default; some gateways serve /chat/completions, with no /v1),
  and the model list is at `models_url` (<base>/v1/models by default; a gateway may list its models outside its chat
  route). The start's checks find both (`list_models`, `probe`), and the start records them. A list may give each
  model a display name besides the ID requests take (a gateway's deployment ID): requests always send the ID
  (`model_entries`).

Messages and answers use one neutral form, Ollama's: {"role", "content", "thinking", "tool_calls": [{"id"?,
"function": {"name", "arguments": dict}}]}, and tool results {"role": "tool", "tool_name", "content"}; the client turns
them into the backend's form. The settings come from the job's RKMON_WORKER_JSON (rkmon_lib/config.py: worker,
model_for). The API key, when there is one, comes from its file alone (`api_key_file`: the secret the reviewer
model's binding names) and goes in the header `auth_header` after `auth_scheme` (Authorization: Bearer by default; a
blank scheme sends the key alone, as an API-management subscription key wants); no other header is added. The key is
never logged.

Every request first takes its share of the endpoint's rate budget (`Budget`): one file under <RUN_DIR> that the
parallel reviews (the drafts that run at once) share, locked while it is read and written, with the requests and
tokens of the last minute (`rpm`, `tpm`: the limits the deploy settings file gives the reviewer model's binding) and a
time to wait until, set by an HTTP 429, its Retry-After, or rate-limit headers that say the endpoint is spent (an
endpoint that sends none relies on the configured limits and the 429s). The offload passes (draft, quality, regrade)
each have a share of it (offload.budget_shares): a role under its share goes before one that has used it. A 429 is
retried after that wait, with backoff, up to MAX_RATE_WAITS times; each wait is logged. An answer's tokens are what the
endpoint reports in its usage when it does (reasoning tokens included), else an estimate of its text
(`answer_tokens`); each request's prompt and answer tokens go to <RUN_DIR>/tokens.jsonl with the review that asked,
and its role, pass and step when it has them (`Budget.count`), and `TokenLog` adds up a review's per pass, for the
pilot that sizes the parallel reviews. An answer's `reasoning` text never joins its content; fields a server adds
(vLLM's stop_reason, token_ids, routed_experts) are ignored. The model ID goes out exactly as configured (case kept),
whatever case the answer echoes.

`probe` is the start's check of an endpoint (rkmon_lib/models.py check): does it answer, does it have the model, how
long does a tiny prompt take, and what context does it report. `probe_tools` is the start's tool-call probe, sent
through `chat` with the review loop's own request shape. A refusal names the deploy settings file's keys (DEPLOY_KEYS,
URL_KEYS), as the operator sets them.
"""

from __future__ import annotations

import email.utils
import fcntl
import hashlib
import http.client
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.parse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from llm_client import auth_headers  # noqa: E402
from ollama_stream import RunnerHung, journal_progress, stream_chat  # noqa: E402,F401

WINDOW_S = 60.0  # the rate limits count per minute
# A request leaves our window SLACK_S after its minute: the endpoint counts by its own clock, when each request
# arrives, so one sent the moment our minute ends can reach it a few ms inside its own (one request too many in its
# minute).
SLACK_S = 1.0
QUEUE_POLL_S = 1.0  # a review behind others that asked first looks again this often
QUEUE_TTL_S = 1800.0  # a waiting review's place is dropped when its process is gone, or after this long


def _alive(pid) -> bool:
    """Whether the process that queued a place in the shared budget still runs (the parallel reviews share one
    machine)."""
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError, OverflowError, OSError):
        return False
    return True


MAX_RATE_WAITS = 12  # 429s retried in one chat call before it gives up
TOKENS_LOG = "tokens.jsonl"  # in the run folder, beside budget.log: one line per answered request
BACKOFF_S, BACKOFF_MAX_S = 5.0, 300.0


class RateLimited(ConnectionError):
    """The endpoint answered HTTP 429: wait `wait_s` (its Retry-After, or the backoff) before asking again."""

    def __init__(self, msg: str, wait_s: float):
        super().__init__(msg)
        self.wait_s = wait_s


def settings() -> dict:
    """The job's reviewer model settings (RKMON_WORKER_JSON), or `rk env`'s for a run by hand."""
    raw = os.environ.get("RKMON_WORKER_JSON")
    if not raw:
        import rkenv

        raw = rkenv.get("RKMON_WORKER_JSON")
    return json.loads(raw)


def is_local(url: str) -> bool:
    """Whether URL names this machine (rkmon_lib/container_layout.py LOCAL_HOSTS, the one list of its names)."""
    import rkenv

    return (
        urllib.parse.urlsplit(url).hostname or ""
    ) in rkenv.container_layout().LOCAL_HOSTS


def api_key(s: dict) -> str:
    """The endpoint's API key: its file's (`api_key_file`); "" for none, or a file that is missing or empty."""
    if not s.get("api_key_file"):
        return ""
    try:
        return (
            pathlib.Path(os.path.expanduser(s["api_key_file"]))
            .read_text(encoding="utf-8")
            .strip()
        )
    except OSError:
        return ""


def _headers(s: dict) -> dict:
    h = {"Content-Type": "application/json"}
    key = api_key(s)
    if key:
        h.update(auth_headers(s, key))
    return h


def _conn(url: str, timeout: float):
    u = urllib.parse.urlsplit(url)
    cls = (
        http.client.HTTPSConnection
        if u.scheme == "https"
        else http.client.HTTPConnection
    )
    return (
        cls(
            u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=timeout
        ),
        u,
    )


def _path(u, suffix: str) -> str:
    """The request path on the base's own path (a gateway's route prefix kept); a base that already ends in /v1 gets
    no second /v1."""
    base = u.path.rstrip("/")
    if suffix.startswith("/v1/") and base.endswith("/v1"):
        base = base[:-3]
    return (base + suffix) or suffix


# The chat routes an OpenAI-compatible endpoint may serve under its base address, in the order the start tries them.
CHAT_PATHS = ("/v1/chat/completions", "/chat/completions")
# A model list's entry: the ID requests take (a gateway's deployment ID first), and a display name besides it.
ID_KEYS = ("deployment_id", "deploymentId", "deployment", "id", "model")
NAME_KEYS = (
    "display_name",
    "displayName",
    "name",
    "model_name",
    "title",
    "id",
    "model",
)


def chat_path(s: dict) -> str:
    """The chat route under the base address: worker.chat_path, else /v1/chat/completions."""
    return s.get("chat_path") or CHAT_PATHS[0]


def models_urls(s: dict) -> list[str]:
    """Where the model list may be, in the order the start tries them: worker.models_url when it is set; else
    <base>/v1/models and <base>/models, then, for a base with a route prefix, the host's own /v1/models and /models (a
    gateway may list its models outside its chat route)."""
    if s.get("models_url"):
        return [s["models_url"]]
    u = urllib.parse.urlsplit(s["url"])
    host = f"{u.scheme}://{u.netloc}"
    out = [
        host + _path(u, "/v1/models"),
        host + _path(u, "/models"),
        host + "/v1/models",
        host + "/models",
    ]
    return list(dict.fromkeys(out))


def model_entries(body) -> list[dict]:
    """A model list's entries, [{"id", "name", "context"}]: the ID a request sends and the name to show (the ID when
    the list gives none). The list is OpenAI's {"data": [...]}, {"models": [...]}, or a plain JSON list; ids keep their
    case and dots."""
    rows = (
        body
        if isinstance(body, list)
        else (
            (body.get("data") or body.get("models") or [])
            if isinstance(body, dict)
            else []
        )
    )
    out = []
    for m in rows:
        if isinstance(m, str):
            out.append({"id": m, "name": m, "context": None})
            continue
        if not isinstance(m, dict):
            continue
        mid = next(
            (
                str(m[k])
                for k in ID_KEYS
                if isinstance(m.get(k), (str, int)) and str(m[k]).strip()
            ),
            "",
        )
        if not mid:
            continue
        name = next(
            (
                str(m[k])
                for k in NAME_KEYS
                if isinstance(m.get(k), str) and m[k].strip() and m[k] != mid
            ),
            mid,
        )
        out.append(
            {
                "id": mid,
                "name": name,
                "context": m.get("max_model_len")
                or m.get("context_length")
                or m.get("context_window"),
            }
        )
    return out


def list_models(s: dict, timeout: float = 20) -> tuple[int, list[dict], str]:
    """An OpenAI-compatible endpoint's model list: (HTTP status, its entries (`model_entries`), the address that
    answered). Each place of `models_urls` is tried in turn while the one before answers 404 (or a 200 that is no model
    list); any other answer, a 401 among them, ends the search."""
    status, last = 0, ""
    for url in models_urls(s):
        status, body = _request(s, "GET", url, None, timeout)
        last = url
        if status == 200 and (
            isinstance(body, list)
            or isinstance(body, dict)
            and isinstance(body.get("data", body.get("models")), list)
        ):
            return status, model_entries(body), url
        if status not in (200, 404):
            break
    return (404 if status == 200 else status), [], last


# --- the shared rate budget -------------------------------------------------------------------------------------------


def _duration(text) -> float | None:
    """A rate-limit header's duration in seconds: a plain number of seconds, or a Go-style duration such as "6m0s" or
    "1h2m3s" (units ms, s, m and h); None if unknown."""
    t = str(text or "").strip().lower()
    if not t:
        return None
    try:
        return max(0.0, float(t))
    except ValueError:
        pass
    total, seen = 0.0, False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", t):
        total += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
        seen = True
    return total if seen else None


def wait_from_headers(headers, now: float | None = None) -> float | None:
    """Seconds the endpoint asks us to wait: Retry-After (seconds or an HTTP date), retry-after-ms, or the reset of a
    spent requests or tokens window (x-ratelimit-remaining-* of 0). None when the headers say nothing.
    """
    h = {k.lower(): v for k, v in (headers or [])}
    if h.get("retry-after-ms"):
        d = _duration(h["retry-after-ms"])
        if d is not None:
            return d / 1000.0
    if h.get("retry-after"):
        d = _duration(h["retry-after"])
        if d is None:
            try:
                d = max(
                    0.0,
                    email.utils.parsedate_to_datetime(h["retry-after"]).timestamp()
                    - (now or time.time()),
                )
            except (TypeError, ValueError, IndexError):
                d = None
        if d is not None:
            return d
    waits = []
    for kind in ("requests", "tokens"):
        if str(h.get(f"x-ratelimit-remaining-{kind}", "")).strip() == "0":
            d = _duration(h.get(f"x-ratelimit-reset-{kind}"))
            waits.append(d if d is not None else WINDOW_S)
    return max(waits) if waits else None


def prompt_budget(context_tokens) -> tuple[int, int]:
    """The review prompt's size for a model's context (worker.context):
    (the whole prompt, the least the diff gets), in characters. About 4 characters a token, three quarters of the
    context for the prompt (the rest for the tool rounds and the answer), never more than the 650,000 the default
    context was tuned with; the diff's floor is a third of that, at most 150,000."""
    total = min(650_000, 3 * int(context_tokens or 262144))
    return total, min(150_000, total // 3)


def estimate_tokens(*parts) -> int:
    """A request's or an answer's size in tokens, roughly: 4 characters a token."""
    return (
        sum(len(p if isinstance(p, str) else json.dumps(p)) for p in parts if p) // 4
        + 1
    )


def shares_of(raw) -> dict:
    """Per-role shares of the budget ({role: fraction}), positive ones only, scaled to add up to 1; {} for none."""
    out = {}
    for role, v in (raw or {}).items() if isinstance(raw, dict) else ():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > 0:
            out[str(role)] = f
    total = sum(out.values())
    return {r: f / total for r, f in out.items()} if total else {}


class Budget:
    """One endpoint's rate budget, shared by the parallel reviews: <run_dir>/budget-<endpoint>.json, read and written
    under flock. It holds the requests of the last minute and SLACK_S ([time, tokens, ticket, role]), `blocked_until`,
    and the reviews waiting for a share, oldest first ([ticket, since, pid, role]). Without a run folder it lives in
    this process only. Limits of 0 mean none known: only 429s and the headers make it wait. Each answered request's
    tokens go to <run_dir>/tokens.jsonl (`count`, with `tags`, such as an offload pass and its step), and `spent` keeps
    this budget's own total.

    The other model roles on the reviewer's endpoint (the planner, the analyst: rkmon_lib/roles.py) share this budget:
    their settings name the reviewer's budget (`budget_label`), and each keeps within its share of the limits
    (`budget_role`, `budget_share`): a role over its share waits outside the queue, so it never holds the parallel
    reviews back.

    `role` and `shares` (the offload passes, offload.budget_shares: draft, quality, regrade): while requests wait, those
    of a role still under its share of the window (its tokens of `tpm`, else its requests of `rpm`) go before those of
    a role that has used its share, oldest first within each; with no other role waiting, a role may use the whole
    budget. Without shares, or for a role that has none, it is first come, first served.
    """

    def __init__(
        self,
        s: dict,
        run_dir: str | None = None,
        log=None,
        clock=time.time,
        sleep=time.sleep,
        who: str = "",
        role: str = "",
        shares: dict | None = None,
    ):
        self.rpm, self.tpm = int(s.get("rpm") or 0), int(s.get("tpm") or 0)
        self.label = s.get("budget_label") or f"{s.get('url', '')} {s.get('model', '')}"
        self.role = role or str(s.get("budget_role") or "")
        self.share = min(1.0, max(0.01, float(s.get("budget_share") or 1.0)))
        self.shares = shares_of(shares)
        name = hashlib.sha1(self.label.encode()).hexdigest()[:12]
        self.path = pathlib.Path(run_dir) / f"budget-{name}.json" if run_dir else None
        self.log_path = pathlib.Path(run_dir) / "budget.log" if run_dir else None
        self.tokens_path = pathlib.Path(run_dir) / TOKENS_LOG if run_dir else None
        self.log, self.clock, self.sleep, self.who = log or _stderr, clock, sleep, who
        self.model = str(s.get("model") or "")
        self._mem = {"window": [], "blocked_until": 0.0, "queue": []}
        self._ticket, self._count = "", 0
        self.spent = [
            0,
            0,
        ]  # prompt and answer tokens of the requests this budget answered
        self.tags: dict = (
            {}
        )  # added to each request's line of the token log (an offload pass and its step)

    def _spent(self, st: dict, role: str) -> bool:
        """Whether `role` has used its share of the window: its tokens of `tpm`, or its requests of `rpm`."""
        share = self.shares.get(role)
        if not share or not (self.rpm or self.tpm):
            return False
        mine = [w for w in st["window"] if len(w) > 3 and w[3] == role]
        if self.tpm:
            return sum(w[1] for w in mine) >= share * self.tpm
        return len(mine) >= share * self.rpm

    def _update(self, fn):
        """fn(state) -> result, with the state file locked; the state is written back."""
        if not self.path:
            return fn(self._mem)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            raw = os.read(fd, 1 << 20)
            try:
                st = json.loads(raw) if raw.strip() else {}
            except ValueError:
                st = {}
            st = {
                "window": st.get("window") or [],
                "blocked_until": float(st.get("blocked_until") or 0.0),
                "queue": st.get("queue") or [],
            }
            out = fn(st)
            data = json.dumps(st).encode()
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, data)
            return out
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _say(self, text: str) -> None:
        self.log(text)
        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {self.who or '-'} {text}\n"
                    )
            except OSError:
                pass

    def acquire(self, tokens: int) -> float:
        """Wait until a request of about `tokens` fits the shared budget, take its share, and return the seconds
        waited. The reviews that asked first go first: one waiting in the shared queue keeps its place, and a slot
        that frees up goes to the oldest waiter (else a draft's second request can take the slot another draft has
        waited for since the first minute). A request larger than the whole tokens limit goes once the window is empty.
        """
        waited, said = 0.0, ""
        self._count += 1
        ticket = self._ticket = f"{os.getpid()}.{id(self)}.{self._count}"
        pid, since = os.getpid(), self.clock()

        def leave(st):
            st["queue"] = [q for q in st["queue"] if q[0] != ticket]

        try:
            while True:
                now = self.clock()

                def take(st):
                    st["window"] = [
                        w for w in st["window"] if now - w[0] < WINDOW_S + SLACK_S
                    ]
                    capped = self._over_share(st, tokens, now)
                    if capped:  # a role over its share waits outside the queue
                        leave(st)
                        return capped
                    queue = [
                        q
                        for q in st["queue"]
                        if q[0] != ticket and now - q[1] < QUEUE_TTL_S and _alive(q[2])
                    ]
                    st["queue"] = queue = sorted(
                        queue + [[ticket, since, pid, self.role]], key=lambda q: q[1]
                    )
                    # The order the waiters go in: a role under its share before one that has used it, oldest first.
                    spent = {
                        r: self._spent(st, r)
                        for r in {q[3] if len(q) > 3 else "" for q in queue}
                    }
                    order = sorted(
                        queue, key=lambda q: (spent[q[3] if len(q) > 3 else ""], q[1])
                    )
                    ahead = [q[0] for q in order].index(ticket)
                    if st["blocked_until"] > now:
                        return (
                            st["blocked_until"] - now,
                            "the endpoint asked us to wait (429 or its rate-limit headers)",
                        )
                    used = sum(w[1] for w in st["window"])
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
                    if (self.rpm and len(st["window"]) + ahead >= self.rpm) or (
                        self.tpm and ahead
                    ):
                        if spent[self.role] and ahead > queue.index(
                            next(q for q in queue if q[0] == ticket)
                        ):
                            return QUEUE_POLL_S, (
                                f"the {self.role} share is used: {ahead} request(s) of roles under "
                                "their shares go first"
                            )
                        return (
                            QUEUE_POLL_S,
                            f"{ahead} review(s) asked first and go first",
                        )
                    if self.tpm and st["window"] and used + tokens > self.tpm:
                        return (
                            free_at - now,
                            f"{used} + {tokens} of {self.tpm} tokens in the last minute",
                        )
                    st["window"].append([now, tokens, ticket, self.role])
                    leave(st)
                    return 0.0, ""

                wait, why = self._update(take)
                if wait <= 0:
                    return waited
                wait = max(0.5, min(wait, WINDOW_S))
                if why != said:  # once per reason, not once per look
                    self._say(
                        f"budget: waiting {wait:.0f} s on the shared rate budget of {self.label}: {why}"
                    )
                    said = why
                self.sleep(wait)
                waited += wait
        except BaseException:
            self._update(leave)  # a review that stops waiting gives up its place
            raise

    def _over_share(
        self, st: dict, tokens: int, now: float
    ) -> tuple[float, str] | None:
        """(seconds, why) while this role is over its share of the limits in the window; None when it is within it,
        or has the whole budget (the parallel reviews)."""
        if not self.role or self.share >= 1.0:
            return None
        mine = [w for w in st["window"] if len(w) > 3 and w[3] == self.role]
        if not mine:
            return None
        free_at = min(w[0] for w in mine) + WINDOW_S + SLACK_S
        cap_r = max(1, int(self.rpm * self.share)) if self.rpm else 0
        if cap_r and len(mine) >= cap_r:
            return (
                free_at - now,
                f"the {self.role}'s share: {len(mine)} of {cap_r} requests in the last minute",
            )
        cap_t = int(self.tpm * self.share) if self.tpm else 0
        used = sum(w[1] for w in mine)
        if cap_t and used + tokens > cap_t:
            return (
                free_at - now,
                f"the {self.role}'s share: {used} + {tokens} of {cap_t} tokens in the last minute",
            )
        return None

    def record(self, tokens: int) -> None:
        """Add an answer's tokens to the request this review just made (the window counts both ways): its own entry,
        not the last one, which may be another review's."""

        def add(st):
            mine = [w for w in st["window"] if len(w) > 2 and w[2] == self._ticket]
            if mine:
                mine[0][1] += tokens
            elif st["window"]:
                st["window"][-1][1] += tokens

        self._update(add)

    def block(self, seconds: float, why: str) -> None:
        """Every review waits `seconds` from now (a 429, or a spent window by the headers)."""
        until = self.clock() + max(0.0, seconds)

        def set_until(st):
            st["blocked_until"] = max(st["blocked_until"], until)

        self._update(set_until)
        self._say(f"budget: {why}; every review waits {seconds:.0f} s")

    def count(self, prompt: int, answer: int) -> None:
        """One answered request's tokens: `spent`, and a line of <run_dir>/tokens.jsonl (who asked: the review, such as
        pr123, or the tick triage; the model; prompt and answer), which rk usage sums per day and per review.
        """
        self.spent[0] += prompt
        self.spent[1] += answer
        if not self.tokens_path:
            return
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "who": self.who or "-",
            "model": self.model,
            "prompt": prompt,
            "answer": answer,
            **({"role": self.role} if self.role else {}),
            **self.tags,
        }
        try:
            with open(self.tokens_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass


def _stderr(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def run_dir() -> str:
    """<RUN_DIR> (RKMON_RUN, or `rk env`'s for a run by hand); "" without one."""
    run = os.environ.get("RKMON_RUN")
    if not run:
        try:
            import rkenv

            run = rkenv.get("RKMON_RUN", required=False)
        except SystemExit:
            run = ""
    return run or ""


def budget_for(s: dict, log=None, role: str = "", shares: dict | None = None) -> Budget:
    """The shared budget of this job's endpoint (<RUN_DIR> from RKMON_RUN; this process only without it), for a
    request of `role` (the offload passes' shares)."""
    return Budget(
        s,
        run_dir() or None,
        log=log,
        who=os.environ.get("RKMON_PIPE") or "",
        role=role,
        shares=shares,
    )


def _request(s: dict, method: str, suffix: str, body: dict | None, timeout: float):
    """One plain request to a route under the base address, or to a full address (a model list outside the base's
    route); (status, parsed JSON or text)."""
    status, data, _headers_ = _request_with_headers(s, method, suffix, body, timeout)
    return status, data


def _request_with_headers(
    s: dict, method: str, suffix: str, body: dict | None, timeout: float
):
    """`_request`, with the answer's headers: (status, parsed JSON or text, [(name, value)])."""
    if suffix.startswith(("http://", "https://")):
        conn, u = _conn(suffix, timeout)
        target = (u.path or "/") + (f"?{u.query}" if u.query else "")
    else:
        conn, u = _conn(s["url"], timeout)
        target = _path(u, suffix)
    try:
        conn.request(
            method,
            target,
            body=json.dumps(body).encode() if body is not None else None,
            headers=_headers(s),
        )
        r = conn.getresponse()
        data = r.read()
        try:
            return r.status, json.loads(data), r.getheaders()
        except ValueError:
            return r.status, data.decode("utf-8", "replace"), r.getheaders()
    finally:
        conn.close()


# The start's checks and a rate-limited endpoint: an HTTP 429 is waited out (its Retry-After, else the backoff) up to
# PROBE_RATE_WAITS times, each wait at most PROBE_WAIT_MAX_S; past that the check is "not checked", which never refuses
# a model (config.probe_failure): one 429 during a probe must not leave review-drafts incomplete until the next probe.
PROBE_RATE_WAITS, PROBE_WAIT_MAX_S = 2, 30.0
NOT_CHECKED = "not checked"


def _not_checked(what: str, wait: float | None) -> str:
    asked = f", and asked us to wait {wait:.0f} s" if wait else ""
    return (
        f"{NOT_CHECKED}: the endpoint answered {what} with HTTP 429 (rate limited){asked}; the model is not "
        "refused: the next start checks it again"
    )


def chat(
    s: dict,
    messages: list[dict],
    tools: list | None = None,
    think=None,
    budget: Budget | None = None,
    rate_waits: int = MAX_RATE_WAITS,
    on_usage=None,
) -> dict:
    """One streamed chat request; the joined answer in the neutral form. It first takes its share of the shared rate
    budget, and an HTTP 429 is retried after the endpoint's wait (or the backoff: BACKOFF_S, doubling, at most
    BACKOFF_MAX_S) up to `rate_waits` times. `on_usage` gets the request's tokens (`usage_record`). Raises RunnerHung
    (a local Ollama runner that took the prompt and stalled), TimeoutError (nothing for stall_s), RateLimited (429s
    past `rate_waits`), ConnectionError or http.client errors."""
    think = s.get("think") if think is None else think
    budget = budget or budget_for(s)
    waited = 0.0
    for attempt in range(rate_waits + 1):
        sent = estimate_tokens(messages, tools)
        waited += budget.acquire(sent)
        t0 = time.monotonic()
        try:
            msg = _chat_once(s, messages, tools, think, budget)
        except RateLimited as exc:
            if attempt == rate_waits:
                raise
            wait = (
                exc.wait_s
                if exc.wait_s > 0
                else min(BACKOFF_MAX_S, BACKOFF_S * 2**attempt)
            )
            budget.block(
                wait,
                f"HTTP 429 from {budget.label} (retry {attempt + 1} of {rate_waits})",
            )
            continue
        counted = answer_tokens(msg, sent)
        budget.record(counted)  # reasoning tokens count too
        budget.count(*request_tokens(msg, sent))
        if on_usage:
            on_usage(
                usage_record(
                    msg, sent, counted, waited, time.monotonic() - t0, budget.role
                )
            )
        msg.pop("usage", None)
        return msg
    raise RateLimited(f"HTTP 429 from {budget.label}, {rate_waits} times", 0)


def usage_record(
    msg: dict, sent: int, counted: int, waited: float, elapsed: float, role: str = ""
) -> dict:
    """One request's tokens: what the endpoint reported (prompt, completion, reasoning; 0 when it reported none, with
    `reported` false), the estimate the request took from the budget before it went (`sent_est`), what the answer added
    to the budget (`counted`), and the seconds it waited for the budget and took."""
    u = msg.get("usage") if isinstance(msg.get("usage"), dict) else {}

    def num(v) -> int:
        try:
            return max(0, int(v or 0))
        except (TypeError, ValueError):
            return 0

    details = (
        u.get("completion_tokens_details")
        if isinstance(u.get("completion_tokens_details"), dict)
        else {}
    )
    return {
        "role": role,
        "reported": bool(u),
        "prompt_tokens": num(u.get("prompt_tokens")),
        "completion_tokens": num(u.get("completion_tokens")),
        "reasoning_tokens": num(details.get("reasoning_tokens")),
        "sent_est": sent,
        "counted": counted,
        "waited_s": round(waited, 2),
        "elapsed_s": round(elapsed, 2),
    }


class TokenLog:
    """The reviewer model's tokens in one review, per pass, for the pilot that sizes the parallel reviews: each
    request's record (`usage_record`, from chat's on_usage) is kept, and `close` writes the review's totals per pass to
    <ws>/tokens_total.json. Each request also has its line in <RUN_DIR>/tokens.jsonl (Budget.count), with its pass and
    step. Tokens only: no cost."""

    KEYS = (
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "sent_est",
        "counted",
        "waited_s",
        "elapsed_s",
    )

    def __init__(self, ws, pr, **about):
        self.ws, self.pr, self.about = pathlib.Path(ws), str(pr), about
        self.rows: list[dict] = []
        self.t0 = time.time()

    def step(self, rec: dict, pass_: str, step: int) -> None:
        self.rows.append(dict(rec, step=step, **{"pass": pass_}))

    def totals(self) -> dict:
        """{pass: {requests, prompt_tokens, completion_tokens, reasoning_tokens, sent_est, counted, waited_s,
        elapsed_s}}, and "all"."""
        out: dict = {}
        for r in self.rows:
            for key in (r.get("pass") or "?", "all"):
                tot = out.setdefault(key, dict.fromkeys(("requests", *self.KEYS), 0))
                tot["requests"] += 1
                for k in self.KEYS:
                    tot[k] = round(tot[k] + (r.get(k) or 0), 2)
        return out

    def close(self, **extra) -> dict:
        """The review's totals in <ws>/tokens_total.json: this run's passes added to those a run before it in the same
        folder wrote (the worker job's passes, then the finish job's), "all" over them all.
        """
        path = self.ws / "tokens_total.json"
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
        old = old if isinstance(old, dict) else {}
        passes = {k: v for k, v in (old.get("passes") or {}).items() if k != "all"}
        passes.update({k: v for k, v in self.totals().items() if k != "all"})
        every = dict.fromkeys(("requests", *self.KEYS), 0)
        for tot in passes.values():
            for k in every:
                every[k] = round(every[k] + (tot.get(k) or 0), 2)
        line = dict(
            old,
            **self.about,
            pr=self.pr,
            at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            wall_s=round(float(old.get("wall_s") or 0) + time.time() - self.t0, 1),
            passes=dict(passes, all=every),
            **extra,
        )
        try:
            self.ws.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(line, indent=1), encoding="utf-8")
        except OSError:
            pass
        return line


def _chat_once(s: dict, messages: list[dict], tools, think, budget: Budget) -> dict:
    if s["backend"] == "ollama":
        body = {
            "model": s["model"],
            "stream": True,
            "keep_alive": s["keep_alive"],
            "messages": messages,
            "options": {
                "num_ctx": s["context"],
                "num_batch": s["num_batch"],
                "num_predict": s["max_output_tokens"],
                "temperature": s["temperature"],
            },
        }
        # worker.think unset: the request carries no effort, and the model takes its own default
        if think is not None:
            body["think"] = think
        if tools:
            body["tools"] = tools
        progress = (
            journal_progress if is_local(s["url"]) else (lambda since: (None, None))
        )
        try:
            return stream_chat(
                s["url"] + "/api/chat",
                json.dumps(body).encode(),
                s["stall_s"],
                hang_s=s["hang_s"],
                progress=progress,
                headers=_headers(s),
            )
        except ConnectionError as exc:
            if str(exc).startswith("HTTP 429"):
                raise RateLimited(str(exc), 0) from exc
            raise
    return _openai_chat(s, messages, tools, think, budget)


def to_openai(messages: list[dict]) -> list[dict]:
    """The neutral messages in OpenAI's form: tool calls get ids and JSON-string arguments, and each tool result names
    the call it answers (in order)."""
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
                    "content": m.get("content") or None,
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


def _effort(think):
    return think if think in ("high", "medium", "low") else None


def _openai_chat(
    s: dict, messages: list[dict], tools, think, budget: Budget | None = None
) -> dict:
    body = {
        "model": s["model"],
        "stream": True,
        "messages": to_openai(messages),
        "temperature": s["temperature"],
        "max_tokens": s["max_output_tokens"],
    }
    if tools:
        body.update(tools=tools, tool_choice="auto")
    if _effort(think):
        body["reasoning_effort"] = _effort(think)
    conn, u = _conn(s["url"], s["stall_s"])
    try:
        conn.request(
            "POST",
            _path(u, chat_path(s)),
            body=json.dumps(body).encode(),
            headers=_headers(s),
        )
        r = conn.getresponse()
        if r.status == 429:
            raise RateLimited(
                f"HTTP 429: {r.read(300)!r}", wait_from_headers(r.getheaders()) or 0
            )
        if r.status != 200:
            raise ConnectionError(f"HTTP {r.status}: {r.read(300)!r}")
        spent = wait_from_headers(
            [(k, v) for k, v in r.getheaders() if k.lower().startswith("x-ratelimit")]
        )
        if (
            spent and budget is not None
        ):  # the endpoint says a window is spent: every review waits for its reset
            budget.block(spent, f"{budget.label} says its rate window is spent")
        if "event-stream" in (r.getheader("Content-Type") or "").lower():
            return _stream_answer(r)
        data = (
            r.read()
        )  # an endpoint that answered at once: a stream under another type, or one JSON answer
        if data.lstrip().startswith(b"data:"):
            return _stream_answer(data.splitlines())
        return _whole_answer(data)
    finally:
        conn.close()


def _stream_answer(lines) -> dict:
    """Server-sent chat completions chunks joined into the neutral message; NoToolCalls when no chunk is in the chat
    completions shape (a simplified answer with no `choices`)."""
    msg = {"role": "assistant", "content": "", "thinking": "", "tool_calls": []}
    calls: dict[int, dict] = {}
    seen = []
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if isinstance(chunk, dict) and chunk.get("error"):
            raise RuntimeError(f"model server: {chunk['error']}")
        seen.append(chunk)
        if isinstance(chunk, dict) and isinstance(
            chunk.get("usage"), dict
        ):  # a final chunk may carry the usage
            msg["usage"] = chunk["usage"]
        for choice in (chunk.get("choices") if isinstance(chunk, dict) else None) or []:
            d = choice.get("delta") or choice.get("message") or {}
            msg["content"] += d.get("content") or ""
            msg["thinking"] += d.get("reasoning_content") or d.get("reasoning") or ""
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(
                    tc.get("index", len(calls)), {"id": "", "name": "", "arguments": ""}
                )
                c["id"] = tc.get("id") or c["id"]
                fn = tc.get("function") or {}
                c["name"] += fn.get("name") or ""
                args = fn.get("arguments") or ""
                c["arguments"] += args if isinstance(args, str) else json.dumps(args)
    if seen and not any(isinstance(c, dict) and "choices" in c for c in seen):
        raise NoToolCalls(_shape_note(seen[0]))
    for _i, c in sorted(calls.items()):
        try:
            args = json.loads(c["arguments"]) if c["arguments"].strip() else {}
        except ValueError:
            args = {}
        msg["tool_calls"].append(
            {"id": c["id"], "function": {"name": c["name"], "arguments": args}}
        )
    return msg


class NoToolCalls(ValueError):
    """The endpoint answered, but not in the chat completions shape (no `choices`), so it can make no tool call: a
    simplified "unified" answer such as {"model", "deployment", "response": {"role", "content"}}.
    """


def _shape_note(body) -> str:
    fields = ", ".join(sorted(body)) if isinstance(body, dict) else type(body).__name__
    return (
        f"the endpoint answers without tool calls: its answer is not in the chat completions shape (no choices; it "
        f"has {fields or 'nothing'}). Give {URL_KEYS['openai']} the address of the endpoint's OpenAI-compatible API: "
        "the start finds its chat route under it"
    )


def _whole_answer(data: bytes) -> dict:
    """A chat completions answer sent whole (JSON, not a stream): the neutral message, or NoToolCalls when it is not
    in that shape."""
    try:
        body = json.loads(data or b"{}")
    except ValueError:
        raise NoToolCalls(
            "the endpoint answers without tool calls: its answer is neither a stream nor JSON"
        ) from None
    if (
        not isinstance(body, dict)
        or not isinstance(body.get("choices"), list)
        or not body["choices"]
    ):
        if isinstance(body, dict) and body.get("error"):
            raise RuntimeError(f"model server: {body['error']}")
        raise NoToolCalls(_shape_note(body))
    m = body["choices"][0].get("message") or {}
    calls = []
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                args = {}
        calls.append(
            {
                "id": tc.get("id") or "",
                "function": {"name": fn.get("name") or "", "arguments": args or {}},
            }
        )
    out = {
        "role": "assistant",
        "content": m.get("content") or "",
        "thinking": m.get("reasoning_content") or m.get("reasoning") or "",
        "tool_calls": calls,
    }
    if isinstance(body.get("usage"), dict):
        out["usage"] = body["usage"]
    return out


def answer_tokens(msg: dict, prompt_estimate: int = 0) -> int:
    """What an answer adds to the shared token budget: the larger of an estimate of its text (the reasoning included)
    and what the endpoint reports (usage: completion tokens, which count reasoning; reasoning tokens it counted but did
    not send), plus the prompt tokens it reports beyond the estimate the request took before it went.
    """
    calls = [tc.get("function") for tc in msg.get("tool_calls") or []]
    est = estimate_tokens(msg.get("content"), msg.get("thinking"), calls)
    u = msg.get("usage") or {}

    def num(v) -> int:
        try:
            return max(0, int(v or 0))
        except (TypeError, ValueError):
            return 0

    reasoning = num(
        (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        if isinstance(u.get("completion_tokens_details"), dict)
        else 0
    )
    answer = max(
        est,
        num(u.get("completion_tokens")),
        reasoning + estimate_tokens(msg.get("content"), calls),
    )
    return answer + max(0, num(u.get("prompt_tokens")) - prompt_estimate)


def request_tokens(msg: dict, prompt_estimate: int) -> tuple[int, int]:
    """A request's (prompt, answer) tokens for the token log: the prompt tokens the endpoint reports, else the
    request's estimate; the answer as `answer_tokens` counts it."""
    try:
        reported = max(0, int((msg.get("usage") or {}).get("prompt_tokens") or 0))
    except (TypeError, ValueError):
        reported = 0
    return reported or prompt_estimate, answer_tokens(msg, prompt_estimate) - max(
        0, reported - prompt_estimate
    )


def unload(s: dict) -> None:
    """Free a wedged local Ollama runner before a resend (`ollama stop`); nothing for any other endpoint. Never a
    service restart."""
    if s["backend"] == "ollama" and is_local(s["url"]):
        try:
            subprocess.run(
                ["ollama", "stop", s["model"]], capture_output=True, timeout=60
            )
        except (OSError, subprocess.SubprocessError):
            pass


# The deploy settings file's keys that a refusal names, by the reviewer model's own settings: those of the key it
# sends, and its address by its API (config worker.backend). rkmon_lib/deploy_tree.py names each, and this folder
# reads no schema: tests/test_models.py holds the two to agreeing.
DEPLOY_KEYS = {
    "auth_header": "model_bindings.worker.api_key_header",
    "auth_scheme": "model_bindings.worker.api_key_scheme",
    "auth_mode": "model_bindings.worker.api_key_sent_as",
    "api_key_secret": "model_bindings.worker.api_key_secret_name",
}
URL_KEYS = {
    "openai": "model_bindings.worker.base_url",
    "ollama": "model_bindings.worker.server_url",
}


def explain_status(status: int, s: dict, what: str) -> str:
    """An endpoint's refusal in words, naming the deploy settings of the key it sent for 401 and 403 (never the key),
    and of its address for 404."""
    if status in (401, 403):
        how = " and ".join(f"`{k}: {v}`" for k, v in auth_headers(s, "<key>").items())
        have = "a key" if api_key(s) else "no key"
        k = DEPLOY_KEYS
        return (
            f"{what} got HTTP {status} ({'unauthorized' if status == 401 else 'forbidden'}): the endpoint refused "
            f"the credentials (sent {have} as {how}). If it sits behind an API-management layer it may want its "
            f"own header, such as a subscription key: in the deploy settings file, set {k['auth_header']}, and "
            f"{k['auth_scheme']} blank to send the key alone ({k['auth_mode']} header_and_bearer sends it as a "
            f"bearer token too); the key is the secret {k['api_key_secret']} names"
        )
    if status == 404:
        return (
            f"{what} got HTTP 404: check {URL_KEYS.get(s.get('backend'), URL_KEYS['openai'])} in the deploy "
            "settings file (the base address, with any route prefix the endpoint needs"
            + (
                ": the start finds its chat route and its model list under it"
                if s.get("backend") == "openai"
                else ""
            )
            + ")"
        )
    if status == 429:
        return f"{what} got HTTP 429: the endpoint is rate limiting; try again later"
    return f"{what} got HTTP {status}"


# How the probe's error begins when a request of it got no HTTP answer: a connection error, a timeout, a dropped
# connection.
NO_ANSWER = "no answer"


def probe(s: dict, timeout: float = 60, sleep=time.sleep) -> dict:
    """Does the endpoint answer, does it have the model, how long does a tiny prompt take, and what context length does
    it report (None when it reports none). For an OpenAI-compatible endpoint also where its model list and its chat
    route are (`models_url`, `chat_path`): the configured ones, else the first of each place that answers. A test
    prompt answered with HTTP 429 is waited out and asked again (PROBE_RATE_WAITS); still rate limited, `rate_limited`
    says so, with no error: the model is not checked, never refused. A request that gets no HTTP answer makes the
    error begin with NO_ANSWER."""
    out = {
        "answers": False,
        "models": [],
        "model_found": False,
        "context": None,
        "latency_s": None,
        "error": "",
        "models_url": "",
        "chat_path": "",
        "rate_limited": "",
    }
    try:
        if s["backend"] == "ollama":
            status, tags = _request(s, "GET", "/api/tags", None, timeout)
            out["answers"] = status == 200
            out["models"] = (
                [m.get("name") or m.get("model") for m in (tags.get("models") or [])]
                if isinstance(tags, dict)
                else []
            )
        else:
            status, entries, where = list_models(s, timeout)
            out["answers"] = status == 200
            out["models_url"] = where if status == 200 else ""
            out["models"] = [m["id"] for m in entries]
            for m in entries:
                if m["id"] == s["model"]:
                    out["context"] = m["context"]
        out["model_found"] = (
            s["model"] in out["models"] or f"{s['model']}:latest" in out["models"]
        )
        if not out["answers"]:
            out["error"] = explain_status(status, s, "the model list")
            return out
        if not out["model_found"]:
            return out
        if s["backend"] == "ollama":
            status, show = _request(
                s, "POST", "/api/show", {"model": s["model"]}, timeout
            )
            info = (show.get("model_info") or {}) if isinstance(show, dict) else {}
            out["context"] = next(
                (v for k, v in info.items() if k.endswith(".context_length")), None
            )
            chats = [
                (
                    "/api/chat",
                    {
                        "model": s["model"],
                        "stream": False,
                        "think": False,
                        "messages": [{"role": "user", "content": "Reply with OK."}],
                        "options": {"num_predict": 8},
                    },
                )
            ]
        else:
            chats = [
                (
                    path,
                    {
                        "model": s["model"],
                        "messages": [{"role": "user", "content": "Reply with OK."}],
                        "max_tokens": 8,
                    },
                )
                for path in ([s["chat_path"]] if s.get("chat_path") else CHAT_PATHS)
            ]
        t0 = time.monotonic()
        for path, body in chats:
            for attempt in range(PROBE_RATE_WAITS + 1):
                status, _data, headers = _request_with_headers(
                    s, "POST", path, body, timeout
                )
                if status != 429:
                    break
                wait = wait_from_headers(headers) or BACKOFF_S * 2**attempt
                if attempt == PROBE_RATE_WAITS or wait > PROBE_WAIT_MAX_S:
                    out["rate_limited"] = _not_checked(
                        "the test prompt", wait_from_headers(headers)
                    )
                    return out
                sleep(wait)
            if status != 404:
                if s["backend"] != "ollama":
                    out["chat_path"] = path
                break
        out["latency_s"] = round(time.monotonic() - t0, 2)
        if status != 200:
            out["error"] = explain_status(status, s, "the test prompt")
    except (OSError, http.client.HTTPException, ValueError) as exc:
        out["error"] = f"{NO_ANSWER}: {type(exc).__name__}: {exc}"
    return out


# The tool the probe offers: the review loop's read_file, in the loop's own shape (worker/review_agent.py TOOLS).
PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the PR head worktree. Param: path (relative to the worktree "
        "root).",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}


def probe_tools(s: dict, timeout: float = 120, sleep=time.sleep) -> dict:
    """The start's tool-call probe: one request through `chat`, the review loop's own path (streamed, the tools in the
    loop's shape, tool_choice auto, the configured max_output_tokens), asking for one read_file call. {"ok", "error",
    "latency_s"}: ok only when the answer is a read_file call with a path. An HTTP 429 is waited out and asked again
    (PROBE_RATE_WAITS); still rate limited, "ok" is None (not checked, never refused) and "error" says why. A request
    that gets no HTTP answer makes the error begin with NO_ANSWER, as the test chat's does (`probe`).
    """
    t = dict(s, stall_s=min(float(s.get("stall_s") or timeout), timeout))
    msgs = [
        {
            "role": "system",
            "content": "You are checking that tool calls work. Answer only with a tool call.",
        },
        {
            "role": "user",
            "content": "Call the read_file tool with the path README.md. Do not answer in text.",
        },
    ]
    t0 = time.monotonic()
    try:
        for attempt in range(PROBE_RATE_WAITS + 1):
            try:
                msg = chat(
                    t,
                    msgs,
                    tools=[PROBE_TOOL],
                    think=False,
                    budget=Budget(t),
                    rate_waits=0,
                )
                break
            except RateLimited as exc:
                wait = exc.wait_s or BACKOFF_S * 2**attempt
                if attempt == PROBE_RATE_WAITS or wait > PROBE_WAIT_MAX_S:
                    return {
                        "ok": None,
                        "error": _not_checked(
                            "the tool-call probe", exc.wait_s or None
                        ),
                        "latency_s": None,
                    }
                sleep(wait)
    except (
        NoToolCalls
    ) as exc:  # a simplified answer shape: no choices, so no tool call is possible
        return {"ok": False, "error": str(exc), "latency_s": None}
    except (OSError, http.client.HTTPException, RuntimeError, ValueError) as exc:
        m = re.match(r"HTTP (\d{3})", str(exc))
        if m:
            error = explain_status(int(m.group(1)), s, "the request")
        else:
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
            if isinstance(exc, (OSError, http.client.HTTPException)):
                # No HTTP answer: a connection error, a timeout, a dropped connection.
                error = f"{NO_ANSWER}: {error}"
        return {"ok": False, "latency_s": None, "error": error}
    took = round(time.monotonic() - t0, 2)
    calls = msg.get("tool_calls") or []
    fn = (calls[0].get("function") or {}) if calls else {}
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = None
    if (
        fn.get("name") == "read_file"
        and isinstance(args, dict)
        and str(args.get("path") or "").strip()
    ):
        return {"ok": True, "error": "", "latency_s": took}
    if calls:
        return {
            "ok": False,
            "latency_s": took,
            "error": f"it called {fn.get('name')!r} with {json.dumps(args)[:80]}, "
            "not read_file with a path",
        }
    text = " ".join((msg.get("content") or "").split())[:80]
    return {
        "ok": False,
        "latency_s": took,
        "error": "it answered in text, with no tool call (the review loop needs "
        f"function calling): {text!r}",
    }
