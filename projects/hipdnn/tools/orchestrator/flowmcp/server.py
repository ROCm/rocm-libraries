# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""The MCP adapter -- and only an adapter.

The one module that imports the SDK, and the one module the test suite does not
cover. That is the trade: everything with a decision in it lives next door in
`schema.py`, `projections.py`, `resources.py` and `supervisor.py`, which are
stdlib (plus `runner`) and are tested. Nothing here validates anything.

* The flows allow-list is `resources.resolve_flow`.
* `run://` containment is `resources.resolve_in_run`.
* The iteration clamp is `Supervisor._clamp`.
* The subscription set is the supervisor's, not this module's.

Run it with the orchestrator directory as the working directory:

    <orchestrator>/.venv/Scripts/python.exe -m flowmcp.server \\
        --tools configs/tools.local.yaml \\
        --flows-dir configs/flows \\
        --run-root runs \\
        [--profile <name>] [--max-concurrent 2] [--max-iterations-ceiling <n>]

Transport is stdio. The server's own stderr stays free-form diagnostics and
never carries protocol.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError

from . import resources, schema
from .supervisor import ROOT, Supervisor, SupervisorError

SERVER_NAME = "hipdnn-flow-orchestrator"
SERVER_VERSION = "0.1.0"

INSTRUCTIONS = (
    "Launch and monitor hipDNN agent flows. flow_launch returns immediately; "
    "poll flow_status or subscribe to run://<runId>/run.json, which is the "
    "authoritative state. These flows never compile or execute a kernel — a "
    "clean review is not a correctness claim."
)

TEMPLATE = types.ResourceTemplate(
    uriTemplate="run://{runId}/{+path}",
    name="hipdnn-flow-run-artifact",
    title="hipDNN flow run artifact",
    description=(
        "Any file inside a run directory, addressed by its path relative to the "
        "run root. The run manifest is always at run.json; everything else is "
        "whatever the flow and the engine wrote — enumerate it with "
        "resources/list rather than guessing names."
    ),
    mimeType="text/plain",
)

#: JSON-RPC codes the resource surface uses. `RESOURCE_NOT_FOUND` is the MCP
#: code for a URI that names nothing.
RESOURCE_NOT_FOUND = -32002


def _arguments(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Reject argument sets the tool did not declare, before anything acts on them."""
    tool = schema.TOOL_BY_NAME.get(name)
    if tool is None:
        raise McpError(
            types.ErrorData(code=types.INVALID_PARAMS, message=f"unknown tool {name!r}")
        )
    supplied = dict(arguments or {})
    declared = tool["inputSchema"].get("properties", {})
    unknown = sorted(set(supplied) - set(declared))
    missing = [
        key for key in tool["inputSchema"].get("required", []) if key not in supplied
    ]
    if unknown or missing:
        detail = ", ".join(
            [f"unknown argument {key!r}" for key in unknown]
            + [f"missing required argument {key!r}" for key in missing]
        )
        raise McpError(types.ErrorData(code=types.INVALID_PARAMS, message=detail))
    return supplied


def build_server(supervisor: Supervisor) -> Server:
    server: Server = Server(SERVER_NAME, version=SERVER_VERSION)
    state: dict[str, Any] = {"session": None, "loop": None}

    def remember() -> None:
        """Hold on to the live session so background events have somewhere to go.

        Notifications are emitted out of band -- the supervisor observes the
        manifest from a reader thread and a timer, long after the launch call
        returned -- so the session is captured from a request context rather
        than plumbed through one.
        """
        try:
            state["session"] = server.request_context.session
        except LookupError:
            pass

    def dispatch(event: dict[str, Any]) -> None:
        session = state["session"]
        loop = state["loop"]
        if session is None or loop is None:
            return
        kind = event.get("type")
        if kind == "resource_updated":
            coroutine = session.send_resource_updated(types.AnyUrl(event["uri"]))
        elif kind == "resource_list_changed":
            coroutine = session.send_resource_list_changed()
        elif kind == "message":
            coroutine = session.send_log_message(
                level=event.get("level", "info"),
                data=event.get("data", ""),
                logger=event.get("logger"),
            )
        elif kind == "progress":
            coroutine = session.send_progress_notification(
                progress_token=event["token"],
                progress=event["progress"],
                total=event.get("total"),
                message=event.get("message"),
            )
        else:
            return
        asyncio.run_coroutine_threadsafe(coroutine, loop)

    supervisor.set_emitter(dispatch)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        remember()
        return [
            types.Tool(
                name=tool["name"],
                description=tool["description"],
                inputSchema=tool["inputSchema"],
                outputSchema=tool["outputSchema"],
            )
            for tool in schema.TOOLS
        ]

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict[str, Any] | None
    ) -> tuple[list[types.ContentBlock], dict[str, Any]]:
        remember()
        args = _arguments(name, arguments)
        try:
            result = await asyncio.to_thread(_invoke, supervisor, name, args)
        except (SupervisorError, resources.ResourceError) as failure:
            # Operational failures are results, not protocol errors: the caller
            # named a flow that will not load, or hit the concurrency cap.
            raise ValueError(str(failure)) from None
        return [
            types.TextContent(type="text", text=json.dumps(result, indent=2))
        ], result

    @server.list_resources()
    async def list_resources() -> list[types.Resource]:
        remember()
        return [
            types.Resource(
                uri=types.AnyUrl(entry["uri"]),
                name=entry["name"],
                title=entry.get("title"),
                description=entry.get("description"),
                mimeType=entry["mimeType"],
                size=entry["size"],
            )
            for entry in supervisor.list_resources()
        ]

    @server.list_resource_templates()
    async def list_resource_templates() -> list[types.ResourceTemplate]:
        remember()
        return [TEMPLATE]

    @server.read_resource()
    async def read_resource(uri: types.AnyUrl) -> list[ReadResourceContents]:
        remember()
        try:
            content = supervisor.read_resource(str(uri))
        except resources.RunNotFound as failure:
            raise McpError(
                types.ErrorData(code=RESOURCE_NOT_FOUND, message=str(failure))
            ) from None
        except resources.ResourceError as failure:
            raise McpError(
                types.ErrorData(code=types.INVALID_PARAMS, message=str(failure))
            ) from None
        return [
            ReadResourceContents(content=content["text"], mime_type=content["mimeType"])
        ]

    @server.subscribe_resource()
    async def subscribe_resource(uri: types.AnyUrl) -> None:
        remember()
        try:
            supervisor.subscribe(str(uri))
        except SupervisorError as failure:
            raise McpError(
                types.ErrorData(code=types.INVALID_PARAMS, message=str(failure))
            ) from None

    @server.unsubscribe_resource()
    async def unsubscribe_resource(uri: types.AnyUrl) -> None:
        remember()
        supervisor.unsubscribe(str(uri))

    server.flowmcp_state = state  # type: ignore[attr-defined]
    return server


def _invoke(supervisor: Supervisor, name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "flow_list":
        return supervisor.list_flows()
    if name == "flow_inputs":
        return supervisor.flow_inputs(args["flow"])
    if name == "flow_validate":
        return supervisor.validate(
            args["flow"], inputs=args.get("inputs"), profile=args.get("profile")
        )
    if name == "flow_launch":
        return supervisor.launch(
            flow=args["flow"],
            inputs=args["inputs"],
            max_iterations=args.get("maxIterations"),
            profile=args.get("profile"),
            label=args.get("label"),
        )
    if name == "flow_cancel":
        return supervisor.cancel(args["runId"])
    if name == "flow_status":
        return supervisor.status(args["runId"], log_tail=args.get("logTail", 50))
    raise SupervisorError(f"unknown tool {name!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="flowmcp.server",
        description="MCP stdio server for hipDNN agent flows.",
    )
    parser.add_argument(
        "--tools",
        default=str(ROOT / "configs" / "tools.yaml"),
        help="Tool registry. The executables a flow may launch are exactly these.",
    )
    parser.add_argument(
        "--flows-dir",
        default=str(ROOT / "configs" / "flows"),
        help="The only directory a flow argument may resolve inside.",
    )
    parser.add_argument(
        "--run-root",
        default=str(ROOT / "runs"),
        help="Where runs are written, and the only tree served over run://.",
    )
    parser.add_argument("--profile", default=None, help="Tool profile.")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Live runs allowed at once.",
    )
    parser.add_argument(
        "--max-iterations-ceiling",
        type=int,
        default=None,
        help=(
            "Raises the bound a requested maxIterations is clamped to. Without "
            "it a caller can only ever lower what the flow itself declares."
        ),
    )
    return parser.parse_args(argv)


async def serve(args: argparse.Namespace) -> None:
    supervisor = Supervisor(
        tools_path=Path(args.tools),
        flows_dir=Path(args.flows_dir),
        run_root=Path(args.run_root),
        profile=args.profile,
        max_concurrent=args.max_concurrent,
        max_iterations_ceiling=args.max_iterations_ceiling,
    )
    supervisor.start()
    server = build_server(supervisor)
    server.flowmcp_state["loop"] = asyncio.get_running_loop()  # type: ignore[attr-defined]
    capabilities = server.get_capabilities(
        notification_options=NotificationOptions(
            resources_changed=True, tools_changed=False
        ),
        experimental_capabilities={},
    )
    #: The SDK derives capabilities from registered handlers but hardcodes
    #: `subscribe=False`, so a server that implements subscription advertises
    #: that it does not. Run state travels as `resources/updated` on the run
    #: manifest, and a client that believes the advertisement never subscribes
    #: and never hears about a run again -- so the one channel that matters
    #: would be silently dead.
    if capabilities.resources is not None:
        capabilities.resources.subscribe = True
    options = InitializationOptions(
        server_name=SERVER_NAME,
        server_version=SERVER_VERSION,
        capabilities=capabilities,
        instructions=INSTRUCTIONS,
    )
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, options)
    finally:
        supervisor.close()


def main(argv: list[str] | None = None) -> int:
    asyncio.run(serve(parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
