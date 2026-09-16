// The flow bridge over a scripted server: what the renderer is told, what the
// server is asked, and when each stops.
//
// The bridge is built with its own dependencies replaced, so no Electron, no
// child process and no real clock are involved: `openClient` hands back the
// fake below, `emit` collects the renderer's event stream, and the poll
// interval and reconnect backoff are milliseconds rather than seconds.

import { describe, expect, test } from "bun:test";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";

type TextBlock = { type: "text"; text: string };
type ToolResult = { structuredContent?: unknown; isError?: boolean; content?: TextBlock[] };
type ToolArguments = Record<string, unknown>;
type Reply = ToolResult | ((args: ToolArguments) => ToolResult | Promise<ToolResult>);

type FlowEvent = {
  kind: string;
  runId?: string;
  reason?: string;
  status?: { status?: string };
};

type IpcResult = { ok?: boolean; error?: string };
type MessageBox = { message: string; buttons: string[] };
type QuitEvent = { preventDefault: () => void };

type OpenHooks = { attach: (client: FakeClient) => void; recordStderr: (text: string) => void };

type BridgeDeps = {
  openClient: (hooks: OpenHooks) => Promise<FakeClient>;
  emit: (event: FlowEvent) => void;
  pollIntervalMs: number;
  backoff: number[];
  dialog: { showMessageBoxSync: (request: MessageBox) => number };
  shell: { showItemInFolder: (target: string) => void };
  quit: () => void;
  appVersion: string;
};

type FlowBridge = {
  handlers: Record<string, (request?: Record<string, unknown>) => Promise<IpcResult>>;
  onBeforeQuit: (event: QuitEvent) => void;
};

const require = createRequire(import.meta.url);
// A CommonJS module written for Electron main: no types exist to import.
const flow = require("../electron/flow.cjs") as {
  createFlowBridge: (deps: BridgeDeps) => FlowBridge;
};
const { ResourceUpdatedNotificationSchema } = require("@modelcontextprotocol/sdk/types.js");

/** A result document, as a server that answered returns one. */
const structured = (payload: unknown): ToolResult => ({ structuredContent: payload });

/** A tool that ran and refused, as distinct from a call that never landed. */
const refusal = (text: string): ToolResult => ({ isError: true, content: [{ type: "text", text }] });

/** Replies keyed by tool name: queued ones first, then the standing one.
 *
 * The script outlives any single client, so a test can say what the server
 * answers before the bridge has opened one, and a reconnect keeps answering.
 */
class Script {
  readonly queued = new Map<string, Reply[]>();
  readonly standing = new Map<string, Reply>();

  queue(name: string, reply: Reply) {
    const pending = this.queued.get(name) ?? [];
    pending.push(reply);
    this.queued.set(name, pending);
  }

  always(name: string, reply: Reply) {
    this.standing.set(name, reply);
  }

  take(name: string): Reply {
    const reply = this.queued.get(name)?.shift() ?? this.standing.get(name);
    if (!reply) throw new Error(`no reply scripted for ${name}`);
    return reply;
  }
}

/** Exactly the surface the bridge uses on a connected client. */
class FakeClient {
  onerror: ((err: unknown) => void) | null = null;
  onclose: (() => void) | null = null;
  closed = false;
  readonly calls: { name: string; args: ToolArguments }[] = [];
  readonly subscribed: string[] = [];
  readonly unsubscribed: string[] = [];
  readonly notifications = new Map<unknown, (message: { params?: ToolArguments }) => void>();

  constructor(
    readonly script: Script,
    readonly log: string[]
  ) {}

  async callTool(
    request: { name: string; arguments: ToolArguments },
    _extra: undefined,
    _options: { timeout: number }
  ): Promise<ToolResult> {
    const args = request.arguments ?? {};
    this.calls.push({ name: request.name, args });
    this.log.push(`call:${request.name}`);
    const reply = this.script.take(request.name);
    return typeof reply === "function" ? await reply(args) : reply;
  }

  async subscribeResource({ uri }: { uri: string }, _options: { timeout: number }) {
    this.subscribed.push(uri);
    this.log.push(`subscribe:${uri}`);
    return {};
  }

  async unsubscribeResource({ uri }: { uri: string }, _options: { timeout: number }) {
    this.unsubscribed.push(uri);
    this.log.push(`unsubscribe:${uri}`);
    return {};
  }

  setNotificationHandler(schema: unknown, handler: (message: { params?: ToolArguments }) => void) {
    this.notifications.set(schema, handler);
  }

  /** Deliver a server notification, as the transport would. */
  fire(schema: unknown, message: { params?: ToolArguments }) {
    const handler = this.notifications.get(schema);
    if (!handler) throw new Error("no handler registered for that notification");
    handler(message);
  }

  /** The server went away on its own. */
  simulateClose() {
    this.closed = true;
    this.onclose?.();
  }

  async close() {
    this.closed = true;
    this.onclose?.();
  }

  countOf(name: string) {
    return this.calls.filter((call) => call.name === name).length;
  }
}

const RUN = "tracked-run";
const MANIFEST = `run://${RUN}/manifest`;
const RUN_DIR = join(tmpdir(), "flow bridge", RUN);
const FLOW = "flow-under-test";
const STDERR_TAIL = "the server said why it stopped";

// The subject is a real `setInterval` poll, so proving it stopped means
// letting several of its periods pass; the bridge takes its period as a
// dependency, and these tests set it to 5 ms. Everything a condition can
// express is awaited through `until` instead.
function tick(ms = 0) {
  const { promise, resolve } = Promise.withResolvers<void>();
  setTimeout(resolve, ms);
  return promise;
}

const QUIET = 50;

async function until(what: string, ready: () => boolean, limitMs = 2_000) {
  const deadline = Date.now() + limitMs;
  while (!ready()) {
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await tick(1);
  }
}

function harness(options: { pollIntervalMs?: number; choice?: number } = {}) {
  const script = new Script();
  const log: string[] = [];
  const events: FlowEvent[] = [];
  const clients: FakeClient[] = [];
  const dialogs: MessageBox[] = [];
  const revealed: string[] = [];
  let quits = 0;

  script.always("flow_status", structured({ status: "running" }));
  script.always("flow_list", structured({ flows: [] }));
  script.always("flow_cancel", structured({ state: "cancelled" }));

  const bridge = flow.createFlowBridge({
    openClient: async ({ attach, recordStderr }) => {
      recordStderr(STDERR_TAIL);
      const client = new FakeClient(script, log);
      attach(client);
      clients.push(client);
      return client;
    },
    emit: (event) => {
      events.push(event);
      log.push(`event:${event.kind}`);
    },
    pollIntervalMs: options.pollIntervalMs ?? 60_000,
    backoff: [1],
    dialog: {
      showMessageBoxSync: (request) => {
        dialogs.push(request);
        return options.choice ?? 1;
      },
    },
    shell: { showItemInFolder: (target) => void revealed.push(target) },
    quit: () => void (quits += 1),
    appVersion: "0.0.0-test",
  });

  return {
    bridge,
    handlers: bridge.handlers,
    log,
    events,
    clients,
    dialogs,
    revealed,
    queue: (name: string, reply: Reply) => script.queue(name, reply),
    always: (name: string, reply: Reply) => script.always(name, reply),
    quits: () => quits,
    client: () => clients[clients.length - 1],
    statusCalls: () => log.filter((entry) => entry === "call:flow_status").length,
    /** Start a run and wait for the authoritative status that follows it. */
    launch: async (status = "running") => {
      script.queue(
        "flow_launch",
        structured({ runId: RUN, status, runJsonUri: MANIFEST, runDir: RUN_DIR })
      );
      const result = await bridge.handlers["flow:launch"]({ flow: FLOW });
      expect(result.ok).toBe(true);
      await until("the launch status read", () => log.includes("call:flow_status"));
    },
  };
}

/** A quit request, as Electron delivers it. */
function quitRequest() {
  let prevented = 0;
  return {
    preventDefault: () => void (prevented += 1),
    prevented: () => prevented,
  };
}

describe("run lifecycle", () => {
  test("a run that reaches a terminal state is unsubscribed once and polled no more", async () => {
    const h = harness({ pollIntervalMs: 5 });
    h.queue("flow_status", structured({ status: "running" }));
    h.always("flow_status", structured({ status: "ok" }));

    await h.launch();
    await until("the terminal status", () => h.client().unsubscribed.length > 0);

    expect(h.client().unsubscribed).toEqual([MANIFEST]);
    const settled = h.statusCalls();
    await tick(QUIET);
    expect(h.statusCalls()).toBe(settled);
    expect(h.client().unsubscribed).toEqual([MANIFEST]);
  });

  test("a run the server can no longer report on is retired rather than polled forever", async () => {
    const h = harness({ pollIntervalMs: 5 });
    // `unknown` is the server's answer for a run nothing is supervising: its
    // manifest can say `running` for ever, so waiting on it never ends.
    h.always("flow_status", structured({ status: "unknown" }));

    await h.launch();
    await until("the run to be given up on", () => h.client().unsubscribed.length > 0);

    const settled = h.statusCalls();
    await tick(QUIET);
    expect(h.statusCalls()).toBe(settled);

    const request = quitRequest();
    h.bridge.onBeforeQuit(request);
    expect(h.dialogs).toEqual([]);
    expect(request.prevented()).toBe(0);
  });

  test("the poll re-reads a live run unprompted and never starts for one already finished", async () => {
    const live = harness({ pollIntervalMs: 5 });
    await live.launch();
    await until("the poll to re-read the run", () => live.statusCalls() >= 4);

    const finished = harness({ pollIntervalMs: 5 });
    finished.always("flow_status", structured({ status: "ok" }));
    await finished.launch("ok");
    await tick(QUIET);
    expect(finished.statusCalls()).toBe(1);
    expect(finished.client().subscribed).toEqual([]);
  });

  test("a burst of manifest notifications costs one re-read in flight and one behind it", async () => {
    const h = harness();
    const held = Promise.withResolvers<ToolResult>();
    h.queue("flow_status", () => held.promise);

    await h.launch();
    expect(h.statusCalls()).toBe(1);

    for (let i = 0; i < 3; i += 1) {
      h.client().fire(ResourceUpdatedNotificationSchema, { params: { uri: MANIFEST } });
    }
    expect(h.statusCalls()).toBe(1);

    held.resolve(structured({ status: "running" }));
    await until("the coalesced re-read", () => h.statusCalls() >= 2);
    await tick(20);
    expect(h.statusCalls()).toBe(2);
  });
});

describe("losing the server", () => {
  test("the renderer is told why, and the new server is caught up before it is read", async () => {
    const h = harness();
    await h.launch();
    expect(h.client().subscribed).toEqual([MANIFEST]);

    h.client().simulateClose();
    // Free-form server diagnostics are the only account of why it went.
    const reasons = h.events.filter((event) => event.kind === "disconnected");
    expect(reasons.length).toBe(1);
    expect(reasons[0].reason).toContain(STDERR_TAIL);

    h.log.length = 0;
    await h.handlers["flow:list"]();
    await until("the run to be re-read", () => h.log.includes("call:flow_status"));

    expect(h.clients.length).toBe(2);
    expect(h.clients[1].subscribed).toEqual([MANIFEST]);
    expect(h.log.indexOf("event:reconnected")).toBeGreaterThan(-1);
    expect(h.log.indexOf("event:reconnected")).toBeLessThan(h.log.indexOf("call:flow_status"));
  });
});

describe("quitting", () => {
  test("declining the warning leaves every run alone", async () => {
    const h = harness({ choice: 1 });
    await h.launch();

    const request = quitRequest();
    h.bridge.onBeforeQuit(request);

    expect(request.prevented()).toBe(1);
    expect(h.dialogs.length).toBe(1);
    expect(h.client().countOf("flow_cancel")).toBe(0);
    await tick(20);
    expect(h.client().closed).toBe(false);
  });

  test("accepting it cancels each live run, then closes the transport", async () => {
    const h = harness({ choice: 0 });
    await h.launch();

    h.bridge.onBeforeQuit(quitRequest());
    await until("the transport to close", () => h.client().closed);

    const cancels = h.client().calls.filter((call) => call.name === "flow_cancel");
    expect(cancels.length).toBe(1);
    expect(cancels[0].args.runId).toBe(RUN);
    expect(h.quits()).toBe(1);
  });

  test("a server that already died is not replaced to be asked about its own runs", async () => {
    const h = harness({ choice: 0 });
    await h.launch();
    h.client().simulateClose();
    h.log.length = 0;

    const request = quitRequest();
    h.bridge.onBeforeQuit(request);
    await tick(QUIET);

    expect(h.clients.length).toBe(1);
    expect(h.log).toEqual([]);
    expect(h.events.some((event) => event.kind === "reconnected")).toBe(false);
    // Nothing to warn about: the runs went with the child that owned them.
    expect(h.dialogs).toEqual([]);
    expect(request.prevented()).toBe(0);
  });
});

describe("tool results", () => {
  test("a path that climbs out of its run directory is refused", async () => {
    const h = harness();
    await h.launch();

    const result = await h.handlers["flow:revealArtifact"]({ uri: `run://${RUN}/../../escaped` });

    expect(result.ok).toBe(false);
    expect(result.error).toContain("resolves outside its run directory");
    expect(h.revealed).toEqual([]);
  });

  test("a tool that refuses reports the server's own reason", async () => {
    const h = harness();
    h.queue("flow_list", refusal("the orchestrator declined"));

    const result = await h.handlers["flow:list"]();

    expect(result).toEqual({ ok: false, error: "the orchestrator declined" });
  });
});
