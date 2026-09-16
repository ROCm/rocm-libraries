// The flow bridge: Electron main is the MCP client, the renderer is not.
//
// One long-lived server child speaks JSON-RPC over stdio, connected lazily on
// the first call so a user who never opens the tab never starts a Python. The
// renderer sees none of that -- it gets plain IPC shapes and a `flow:event`
// stream, exactly as the command bridge gets `command:output`.
//
// What a run is, from here: a launch returns as soon as the run has started,
// because a run lasts hours. State afterwards comes from the run's own
// manifest. The server tells us the manifest changed (we subscribe to it on
// launch and unsubscribe when the run reaches a terminal state) and we re-read
// the merged status; a 5 s poll runs alongside so a lost notification costs
// latency rather than correctness.
//
// Events are a scoped union. `status`, `log` and `progress` describe one run
// and carry its `runId`. `disconnected` and `reconnected` describe the server
// and carry none: a panel that filters by `runId` must handle those two
// unconditionally, or losing the server leaves it waiting on a run that can no
// longer report.
//
// The whole lifecycle lives in `createFlowBridge(deps)`: every dep defaults to
// the real implementation, so `registerFlowIpc()` is the production path and
// anything else is a caller supplying a different server, clock or window set.
//
// Nothing here knows the name of a flow, a step, a loop group, an output or a
// file. Flows are enumerated, their inputs are declared, their artifacts are
// discovered, and all three travel through this module untouched.

"use strict";

const { app, BrowserWindow, dialog, ipcMain, shell } = require("electron");
const crypto = require("node:crypto");
const path = require("node:path");

const { Client } = require("@modelcontextprotocol/sdk/client/index.js");
const { StdioClientTransport } = require("@modelcontextprotocol/sdk/client/stdio.js");
const {
  LoggingMessageNotificationSchema,
  ProgressNotificationSchema,
  ResourceListChangedNotificationSchema,
  ResourceUpdatedNotificationSchema,
} = require("@modelcontextprotocol/sdk/types.js");

const flowpaths = require("./flowpaths.cjs");
const graphfile = require("./graphfile.cjs");

const CLIENT_NAME = "hipdnn-graph-studio";

/** Temp-directory scope for the graphs handed to runs; see graphfile.cjs. */
const GRAPH_SCOPE = "implement";
const LAUNCH_DIR_TTL_MS = 7 * 24 * 60 * 60 * 1000;

// Connecting means starting an interpreter; launching means loading the flow,
// hashing it and resolving every tool executable. Both are slower than a call
// that only reads a file.
const CONNECT_TIMEOUT_MS = 30_000;
const CALL_TIMEOUT_MS = 60_000;
const LAUNCH_TIMEOUT_MS = 180_000;
const QUIT_CANCEL_TIMEOUT_MS = 5_000;

/** Loss insurance: correct without a single notification ever arriving. */
const POLL_INTERVAL_MS = 5_000;

const RECONNECT_BACKOFF_MS = [1_000, 2_000, 5_000, 10_000, 30_000];

/** Enough of the server's diagnostics to explain why it would not start. */
const STDERR_KEEP = 4_000;

/** Run states that will not change again, for us.
 *
 * `unknown` belongs here even though the run itself may still be going. It is
 * the server's answer for a run no supervisor is waiting on -- one launched by
 * a session that has since died -- whose manifest can therefore stay `running`
 * for good. Treating it as live kept the poll re-reading a dead run forever,
 * held its subscription open, and made quitting warn about a run that was
 * provably gone.
 */
const TERMINAL_STATES = new Set([
  "ok",
  "failed",
  "cancelled",
  "crashed",
  "unknown",
]);

const RUN_URI = /^run:\/\/([^/]+)\/(.*)$/;

/** The host's own truncation notice, which carries the true size with it. */
const TRUNCATION_NOTICE = /\[truncated: \d+ of (\d+) bytes/;

const errorText = (err) => (err instanceof Error ? err.message : String(err));

const logText = (data) => (typeof data === "string" ? data : JSON.stringify(data));

function broadcast(event) {
  for (const window of BrowserWindow.getAllWindows()) {
    if (window.isDestroyed()) continue;
    const contents = window.webContents;
    if (!contents.isDestroyed()) contents.send("flow:event", event);
  }
}

function runIdOf(uri) {
  const match = RUN_URI.exec(String(uri ?? ""));
  return match ? decodeURIComponent(match[1]) : null;
}

// The logger names the run it belongs to in its last segment.
function loggerRunId(logger) {
  const name = String(logger ?? "");
  const cut = name.lastIndexOf("/");
  return cut < 0 ? name : name.slice(cut + 1);
}

function resultText(result) {
  return (result.content ?? [])
    .filter((block) => block?.type === "text")
    .map((block) => block.text)
    .join("\n")
    .trim();
}

// The size is the file's, not the text's: a truncated read says so, and says
// how big the file really was.
function measureText(text) {
  const lines = text.split("\n");
  const last = lines[lines.length - 1] || lines[lines.length - 2] || "";
  const notice = TRUNCATION_NOTICE.exec(last);
  if (notice) return { truncated: true, size: Number(notice[1]) };
  return { truncated: false, size: Buffer.byteLength(text, "utf8") };
}

/** One run lifecycle, over one server, reporting to one set of windows.
 *
 * `deps` names everything this module cannot reach in a plain process:
 *   `openClient({ attach, recordStderr })` starts a server and returns a
 *     connected client. `attach` installs the close, error and notification
 *     handlers and must run before the transport connects; `recordStderr`
 *     takes the server's diagnostics, which are what explain a failure to
 *     start. `emit(event)` delivers to the renderer, `pollIntervalMs` and
 *     `backoff` are the two clocks, `dialog`, `shell`, `quit` and
 *     `appVersion` are the host.
 */
function createFlowBridge(deps = {}) {
  const emit = deps.emit ?? broadcast;
  const openClient = deps.openClient ?? defaultOpenClient;
  const pollIntervalMs = deps.pollIntervalMs ?? POLL_INTERVAL_MS;
  const backoff = deps.backoff ?? RECONNECT_BACKOFF_MS;
  const dialogApi = deps.dialog ?? dialog;
  const shellApi = deps.shell ?? shell;
  const quitApp = deps.quit ?? (() => app.quit());

  /** Runs this session launched, by id. */
  const runs = new Map();

  let client = null;
  let connecting = null;
  let connected = false;
  let transportError = "";
  let connectError = "";
  let serverStderr = "";
  let reconnectFailures = 0;
  let reconnectAt = 0;
  let pollTimer = null;
  let quitting = false;

  // The server's stderr is free-form diagnostics and never protocol, so it is
  // the only thing that explains a failure to start -- a missing dependency, an
  // import error -- which the protocol itself never gets far enough to report.
  function recordStderr(text) {
    serverStderr = (serverStderr + text).slice(-STDERR_KEEP);
  }

  function withServerStderr(message) {
    const tail = serverStderr.trim();
    return tail === "" ? message : `${message}\n${tail}`;
  }

  // ── Connection ─────────────────────────────────────────────────────────

  async function defaultOpenClient({ attach, recordStderr: record }) {
    const resolved = flowpaths.resolve();
    if (!resolved.ok) throw new Error(resolved.reason);

    const transport = new StdioClientTransport({
      command: resolved.python,
      args: flowpaths.serverArgs(resolved),
      cwd: resolved.orchestratorDir,
      env: { ...process.env, PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8" },
      stderr: "pipe",
    });

    serverStderr = "";
    transport.stderr?.on("data", (chunk) => record(chunk.toString("utf8")));

    const instance = new Client({
      name: CLIENT_NAME,
      version: deps.appVersion ?? app.getVersion(),
    });
    attach(instance);

    try {
      await instance.connect(transport, { timeout: CONNECT_TIMEOUT_MS });
      // Listing the tools caches the output schemas the server publishes, so
      // every structured result below is validated against them before it
      // reaches the renderer.
      await instance.listTools({}, { timeout: CALL_TIMEOUT_MS });
    } catch (err) {
      await instance.close().catch(() => {});
      throw new Error(withServerStderr(errorText(err)));
    }
    return instance;
  }

  function attach(instance) {
    instance.onerror = (err) => {
      transportError = errorText(err);
    };
    instance.onclose = () => handleClosed(instance);
    registerNotifications(instance);
  }

  function ensureClient(auto = false) {
    if (client) return Promise.resolve(client);
    if (auto && Date.now() < reconnectAt) {
      return Promise.reject(new Error(connectError || "The orchestrator is not connected."));
    }
    if (!connecting) {
      connecting = openClient({ attach, recordStderr }).then(
        (instance) => {
          connecting = null;
          client = instance;
          reconnectFailures = 0;
          reconnectAt = 0;
          connectError = "";
          if (connected) void resumeRuns(instance);
          connected = true;
          return instance;
        },
        (err) => {
          connecting = null;
          connectError = errorText(err);
          reconnectFailures += 1;
          reconnectAt =
            Date.now() + backoff[Math.min(reconnectFailures - 1, backoff.length - 1)];
          throw err;
        }
      );
    }
    return connecting;
  }

  // Only the current client's close is news: one we replaced or shut down
  // ourselves has already been accounted for.
  function handleClosed(instance) {
    if (client !== instance) return;
    client = null;
    for (const run of runs.values()) run.subscribed = false;
    const reason = withServerStderr(transportError || "The orchestrator server exited.");
    transportError = "";
    if (quitting) return;
    emit({ kind: "disconnected", reason });
    // The poll drives the reconnect for as long as a run is live; a session
    // with nothing running reconnects on the next call instead.
    reconnectAt = Date.now() + backoff[0];
  }

  async function closeClient() {
    const current = client;
    client = null;
    if (!current) return;
    await current.close().catch(() => {});
  }

  // A new server knows nothing about the runs it did not start, so every live
  // one is re-subscribed and re-read before the renderer is told it is back.
  async function resumeRuns(instance) {
    for (const run of runs.values()) {
      if (!run.terminal) await subscribeRun(run, instance);
    }
    emit({ kind: "reconnected" });
    for (const run of runs.values()) {
      if (!run.terminal) void refreshStatus(run.runId);
    }
  }

  // ── Notifications ──────────────────────────────────────────────────────

  function registerNotifications(instance) {
    instance.setNotificationHandler(ResourceUpdatedNotificationSchema, (message) => {
      const runId = runIdOf(message.params?.uri);
      if (runId && runs.has(runId)) void refreshStatus(runId);
    });

    instance.setNotificationHandler(ProgressNotificationSchema, (message) => {
      const params = message.params ?? {};
      const runId = String(params.progressToken ?? "");
      if (!runs.has(runId)) return;
      const event = { kind: "progress", runId, progress: Number(params.progress) || 0 };
      if (typeof params.total === "number") event.total = params.total;
      if (typeof params.message === "string") event.message = params.message;
      emit(event);
    });

    instance.setNotificationHandler(LoggingMessageNotificationSchema, (message) => {
      const params = message.params ?? {};
      const runId = loggerRunId(params.logger);
      if (!runs.has(runId)) return;
      emit({ kind: "log", runId, text: logText(params.data) });
    });

    // Something appeared or finished; the runs we are tracking may be the ones
    // it happened to.
    instance.setNotificationHandler(ResourceListChangedNotificationSchema, () => {
      for (const run of runs.values()) {
        if (!run.terminal) void refreshStatus(run.runId);
      }
    });
  }

  // ── Tool calls ─────────────────────────────────────────────────────────

  async function callTool(name, args, { timeout = CALL_TIMEOUT_MS, auto = false } = {}) {
    const instance = await ensureClient(auto);
    const result = await instance.callTool({ name, arguments: args }, undefined, { timeout });
    if (result.isError) throw new Error(resultText(result) || `${name} failed.`);
    const payload = result.structuredContent ?? JSON.parse(resultText(result) || "null");
    if (!payload || typeof payload !== "object") {
      throw new Error(`${name} returned no result document.`);
    }
    return payload;
  }

  async function envelope(work) {
    try {
      return { ok: true, ...(await work()) };
    } catch (err) {
      return { ok: false, error: errorText(err) };
    }
  }

  // ── Run tracking ───────────────────────────────────────────────────────

  function trackRun(runId, state, details = {}) {
    const run = runs.get(runId) ?? { runId, subscribed: false };
    if (details.runDir) run.runDir = details.runDir;
    if (details.runJsonUri) run.runJsonUri = details.runJsonUri;
    if (typeof state === "string") {
      run.state = state;
      run.terminal = TERMINAL_STATES.has(state);
    }
    runs.set(runId, run);
    return run;
  }

  const liveRuns = () => [...runs.values()].filter((run) => !run.terminal);

  function updatePollTimer() {
    const live = liveRuns().length > 0;
    if (live && !pollTimer) {
      pollTimer = setInterval(pollTick, pollIntervalMs);
      pollTimer.unref?.();
    } else if (!live && pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function pollTick() {
    for (const run of liveRuns()) void refreshStatus(run.runId);
  }

  async function subscribeRun(run, instance) {
    if (!run.runJsonUri || run.subscribed || run.terminal) return;
    try {
      await instance.subscribeResource({ uri: run.runJsonUri }, { timeout: CALL_TIMEOUT_MS });
      run.subscribed = true;
    } catch {
      // The poll already covers this; a failed subscription costs latency only.
    }
  }

  async function unsubscribeRun(runId) {
    const run = runs.get(runId);
    if (!run?.subscribed) return;
    run.subscribed = false;
    if (!client || !run.runJsonUri) return;
    await client
      .unsubscribeResource({ uri: run.runJsonUri }, { timeout: CALL_TIMEOUT_MS })
      .catch(() => {});
  }

  // Coalesced: a burst of notifications produces one re-read in flight and at
  // most one queued behind it.
  async function refreshStatus(runId) {
    const run = runs.get(runId);
    if (!run) return;
    if (run.refreshing) {
      run.refreshAgain = true;
      return;
    }
    run.refreshing = true;
    try {
      const status = await callTool("flow_status", { runId }, { auto: true });
      trackRun(runId, status.status, status);
      emit({ kind: "status", runId, status });
      if (TERMINAL_STATES.has(status.status)) await unsubscribeRun(runId);
    } catch {
      // A poll that cannot reach the server is not news: the renderer was told
      // the moment the connection dropped.
    } finally {
      run.refreshing = false;
      updatePollTimer();
      if (run.refreshAgain) {
        run.refreshAgain = false;
        void refreshStatus(runId);
      }
    }
  }

  // ── Artifacts ──────────────────────────────────────────────────────────

  async function artifactPath(uri) {
    const match = RUN_URI.exec(String(uri ?? ""));
    if (!match) throw new Error(`${uri} is not a run artifact.`);
    const runId = decodeURIComponent(match[1]);
    const relative = match[2].split("/").map(decodeURIComponent).join(path.sep);

    let runDir = runs.get(runId)?.runDir;
    if (!runDir) {
      const status = await callTool("flow_status", { runId, logTail: 0 });
      runDir = status.runDir;
      trackRun(runId, status.status, status);
    }

    const root = path.resolve(runDir);
    const target = path.resolve(root, relative);
    if (target !== root && !target.startsWith(root + path.sep)) {
      throw new Error(`${uri} resolves outside its run directory.`);
    }
    return target;
  }

  // ── IPC ────────────────────────────────────────────────────────────────

  const handlers = {
    // Whether this machine has an orchestrator at all, and why not when it does
    // not. Every other call reports the same reason as its own error.
    "flow:available": async () => {
      const resolved = flowpaths.resolve();
      return { available: resolved.ok, reason: resolved.ok ? "" : resolved.reason };
    },

    "flow:list": () => envelope(() => callTool("flow_list", {})),

    "flow:inputs": ({ flow }) => envelope(() => callTool("flow_inputs", { flow })),

    // Whether the flow validates is the result's own `ok`, so a flow that does
    // not arrives as a failure carrying its reasons, with the structured fields
    // still attached.
    "flow:validate": async ({ flow, inputs, profile }) => {
      try {
        const args = { flow };
        if (inputs) args.inputs = inputs;
        if (profile) args.profile = profile;
        const result = await callTool("flow_validate", args);
        if (result.ok) return { ...result, ok: true };
        const errors = Array.isArray(result.errors) ? result.errors : [];
        return {
          ...result,
          ok: false,
          error: errors.join("\n") || "The flow did not validate.",
        };
      } catch (err) {
        return { ok: false, error: errorText(err) };
      }
    },

    // The graph is written once, under a directory unique to this launch,
    // because the run holds that path for its whole life and re-reads it. Every
    // name the renderer nominated gets that path; no input name is special here.
    "flow:launch": async (request) => {
      const { flow, inputs, graphInputs, graphJson, graphName, maxIterations, profile, label } =
        request;
      if (typeof flow !== "string" || flow.trim() === "") {
        return { ok: false, error: "No flow was selected." };
      }

      const names = (Array.isArray(graphInputs) ? graphInputs : []).filter(
        (name) => typeof name === "string" && name !== ""
      );

      let graphPath;
      if (names.length > 0) {
        try {
          graphPath = await graphfile.writeGraphFile(
            GRAPH_SCOPE,
            graphName,
            graphJson,
            crypto.randomBytes(4).toString("hex")
          );
        } catch (err) {
          return { ok: false, error: `Failed to write the graph: ${errorText(err)}` };
        }
      }

      const bound = { ...(inputs ?? {}) };
      for (const name of names) bound[name] = graphPath;

      const args = { flow, inputs: bound };
      if (Number.isFinite(maxIterations)) args.maxIterations = maxIterations;
      if (typeof profile === "string" && profile !== "") args.profile = profile;
      if (typeof label === "string" && label !== "") args.label = label;

      try {
        const result = await callTool("flow_launch", args, { timeout: LAUNCH_TIMEOUT_MS });
        const run = trackRun(result.runId, result.status, result);
        if (client) await subscribeRun(run, client);
        updatePollTimer();
        // A first authoritative status now, rather than at the first checkpoint.
        void refreshStatus(result.runId);
        return graphPath ? { ok: true, ...result, graphPath } : { ok: true, ...result };
      } catch (err) {
        return { ok: false, error: errorText(err) };
      }
    },

    "flow:cancel": ({ runId }) =>
      envelope(async () => {
        const result = await callTool("flow_cancel", { runId });
        trackRun(runId, result.state);
        if (TERMINAL_STATES.has(result.state)) await unsubscribeRun(runId);
        void refreshStatus(runId);
        updatePollTimer();
        return result;
      }),

    "flow:status": ({ runId, logTail }) =>
      envelope(async () => {
        const args = { runId };
        if (Number.isFinite(logTail)) args.logTail = logTail;
        const status = await callTool("flow_status", args);
        // Runs this session launched stay tracked; polling one it did not
        // launch leaves no trace behind.
        if (runs.has(runId)) {
          trackRun(runId, status.status, status);
          updatePollTimer();
        }
        return status;
      }),

    "flow:artifact": ({ uri }) =>
      envelope(async () => {
        const instance = await ensureClient();
        const result = await instance.readResource({ uri }, { timeout: CALL_TIMEOUT_MS });
        const contents = (result.contents ?? [])[0];
        if (!contents) throw new Error(`The orchestrator returned no contents for ${uri}.`);
        const text = typeof contents.text === "string" ? contents.text : "";
        return {
          uri: contents.uri ?? uri,
          mimeType: contents.mimeType ?? "text/plain",
          text,
          ...measureText(text),
        };
      }),

    "flow:revealArtifact": ({ uri }) =>
      envelope(async () => {
        shellApi.showItemInFolder(await artifactPath(uri));
        return {};
      }),
  };

  // ── Quit ───────────────────────────────────────────────────────────────

  // Only the server we are already talking to can stop anything. Without a
  // client there is nothing left to cancel: the runs this session launched
  // belonged to a stdio child whose stdin closed when it died, so its workers
  // went with it, and a replacement server would be a fresh supervisor asked
  // about runs it never launched.
  const stoppableRuns = () => (client ? liveRuns() : []);

  async function stopLiveRuns() {
    if (!client) return;
    const cancels = liveRuns().map((run) =>
      callTool("flow_cancel", { runId: run.runId }, { timeout: QUIT_CANCEL_TIMEOUT_MS }).catch(
        () => {}
      )
    );
    const deadline = new Promise((resolve) => setTimeout(resolve, QUIT_CANCEL_TIMEOUT_MS));
    await Promise.race([Promise.all(cancels), deadline]);
  }

  async function shutdown(stopServer) {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    // Leaving the server alone is what lets its runs outlive us: closing the
    // transport signals the child, and the child owns the workers.
    if (stopServer) await closeClient();
  }

  function onBeforeQuit(event) {
    if (quitting) return;
    if (stoppableRuns().length === 0) {
      quitting = true;
      void shutdown(true);
      return;
    }

    event.preventDefault();
    const live = stoppableRuns().length;
    // Quitting ends every run, and the dialog says so because the alternative
    // was measured and does not exist: the server is a stdio child, so when
    // this process goes its stdin closes, the server exits on EOF and takes
    // its workers with it. Leaving the client open changes nothing -- there is
    // no parent left to hold it. Offering to leave runs running would promise
    // something only a detached server, reconnected to by run id, could keep.
    const choice = dialogApi.showMessageBoxSync({
      type: "warning",
      buttons: ["Stop runs and quit", "Cancel"],
      defaultId: 1,
      cancelId: 1,
      noLink: true,
      title: "Runs in progress",
      message: live === 1 ? "One run is still running." : `${live} runs are still running.`,
      detail:
        "Quitting ends each run and every process it started, including any agent " +
        "session it is waiting on. Work already finished stays on disk: each run " +
        "keeps its manifest, logs and artifacts in its own run directory.",
    });
    if (choice === 1) return;

    quitting = true;
    void (async () => {
      await stopLiveRuns();
      await shutdown(true);
      quitApp();
    })();
  }

  return { handlers, onBeforeQuit };
}

/** Register every `flow:*` channel. One call, from main. */
function registerFlowIpc() {
  const bridge = createFlowBridge();
  for (const [channel, handler] of Object.entries(bridge.handlers)) {
    ipcMain.handle(channel, (_event, request) => handler(request ?? {}));
  }
  app.on("before-quit", bridge.onBeforeQuit);
  app.whenReady().then(() => {
    void graphfile.sweepLaunchDirs(GRAPH_SCOPE, LAUNCH_DIR_TTL_MS);
  });
  return bridge;
}

module.exports = { createFlowBridge, registerFlowIpc };
