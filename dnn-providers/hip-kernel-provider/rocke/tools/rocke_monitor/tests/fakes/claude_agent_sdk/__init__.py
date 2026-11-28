# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A stand-in for the Claude Agent SDK, for the container's deploy stage (tests/container/stage_deploy.py puts this
folder's parent first on the orchestrator's PYTHONPATH). Every prompt gets an assistant message and a result. Each
prompt, and the session's options by their shape (never a value of the environment), go to $FAKE_SDK_LOG. No network
call is made."""

import asyncio
import json
import os

__version__ = "0.0.0-fake"


def _log(rec: dict) -> None:
    path = os.environ.get("FAKE_SDK_LOG", "")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


class ClaudeAgentOptions:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        _log(
            {
                "options": {
                    "permission_mode": kw.get("permission_mode"),
                    "cwd": str(kw.get("cwd")),
                    "resume": kw.get("resume"),
                    "model": kw.get("model"),
                    "effort": kw.get("effort"),
                    "setting_sources": kw.get("setting_sources"),
                    "env_names": sorted((kw.get("env") or {}).keys()),
                    "allowed_tools": len(kw.get("allowed_tools") or []),
                    "hooks": sorted((kw.get("hooks") or {}).keys()),
                }
            }
        )


class HookMatcher:
    def __init__(self, matcher=None, hooks=None, timeout=None):
        self.matcher, self.hooks, self.timeout = matcher, hooks or [], timeout


class AssistantMessage:
    def __init__(self, content=None):
        self.content = content or []


class ResultMessage:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class ClaudeSDKClient:
    def __init__(self, options=None):
        self.options, self.queue = options, None

    async def __aenter__(self):
        self.queue = asyncio.Queue()
        _log({"connect": True})
        return self

    async def __aexit__(self, *exc):
        _log({"disconnect": True})
        return False

    async def query(self, prompt):
        _log({"query": prompt})
        await self.queue.put(AssistantMessage())
        await self.queue.put(
            ResultMessage(
                session_id="fake-session",
                origin=None,
                num_turns=1,
                subtype="success",
                duration_ms=1,
                usage={
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 100,
                    "cache_creation_input_tokens": 0,
                },
            )
        )

    async def receive_messages(self):
        while True:
            yield await self.queue.get()
