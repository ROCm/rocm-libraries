import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { engine } from "../engine";
import { FLOW_UNAVAILABLE, flowRunner } from "../flow";
import { buildTimeline } from "../flow/timeline";
import { serializeGraph } from "../graph/serialize";
import { platform } from "../platform";
import { FlowTimeline } from "./FlowTimeline";
import type { FlowInputSpec, FlowSummary, LaunchRequest, LaunchResult, RunStatus } from "../flow";
import type { Graph } from "../graph/model";

/**
 * Launches an agent flow against the current graph and monitors it to
 * completion.
 *
 * Everything the panel shows about a flow is read from the flow: the picker
 * lists whatever the host enumerates, the form is generated from the inputs
 * that flow declares, the iteration control appears because the flow declares a
 * loop, and the canvas-graph toggle is offered on inputs the flow declared as
 * paths. No flow, step, loop, input or output name appears in this file.
 *
 * A run lasts minutes to hours and outlives a tab switch, because the tab panel
 * stays mounted while hidden. `active` therefore guards exactly one thing — the
 * first flow listing, which would otherwise start the orchestrator at app
 * launch for every user, including those who never open this tab.
 */

/** A control's value as typed; coerced to the declared type only at launch. */
type InputValue = string | boolean;

const LOG_LIMIT = 500;
const POLL_MS = 3000;
/** Supplementary log lines the host is asked to replay with each status poll. */
const LOG_TAIL = 200;

/**
 * §6.4: derived from the run, not configured. It appears because this run
 * executed nothing but agents, and stops appearing by itself the first time a
 * flow runs a step that builds or executes something.
 */
const CAUTION =
  "Nothing in this run was compiled or executed. Every step was an agent working on text, so this is review evidence, not a correctness claim.";

interface AuthoringPanelProps {
  /** Produces the current graph on demand, as the command panel does. */
  getGraph(): Graph;
  /** True while the Authoring tab is the selected one. Guards the first listing only. */
  active: boolean;
}

export function AuthoringPanel({ getGraph, active }: AuthoringPanelProps) {
  const [flows, setFlows] = useState<readonly FlowSummary[] | null>(null);
  const [listing, setListing] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [selectedName, setSelectedName] = useState("");
  const [values, setValues] = useState<Record<string, InputValue>>({});
  const [graphInputs, setGraphInputs] = useState<readonly string[]>([]);
  const [budget, setBudget] = useState(1);
  const [running, setRunning] = useState(false);
  const [launched, setLaunched] = useState<LaunchResult | null>(null);
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [logLines, setLogLines] = useState<readonly string[]>([]);
  const [progress, setProgress] = useState<string | null>(null);
  const [disconnected, setDisconnected] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const runIdRef = useRef<string | null>(null);
  // Bumped when a run is abandoned and on unmount, so a late reply for a run
  // nobody is watching any more is dropped instead of overwriting the panel.
  const generationRef = useRef(0);
  const listedRef = useRef(false);
  const logRef = useRef<HTMLDivElement>(null);

  const flow = useMemo(
    () => flows?.find((candidate) => candidate.name === selectedName) ?? null,
    [flows, selectedName],
  );

  useEffect(() => () => {
    ++generationRef.current;
  }, []);

  const refresh = useCallback(async () => {
    const generation = generationRef.current;
    setListing(true);
    setListError(null);
    const result = await flowRunner.list();
    setListing(false);
    if (generation !== generationRef.current) return;
    if (!result.ok) {
      setFlows([]);
      setListError(result.error);
      return;
    }
    setFlows(result.flows);
    setSelectedName((previous) =>
      result.flows.some((candidate) => candidate.name === previous)
        ? previous
        : result.flows[0]?.name ?? "",
    );
  }, []);

  // First activation only: listing on mount would start the orchestrator for
  // everyone, including users who never open this tab.
  useEffect(() => {
    if (!active || listedRef.current) return;
    listedRef.current = true;
    void refresh();
  }, [active, refresh]);

  // A newly selected (or re-listed) flow brings its own form: declared
  // defaults, and the canvas pointed at the first path the flow declares.
  useEffect(() => {
    if (!flow) {
      setValues({});
      setGraphInputs([]);
      return;
    }
    const next: Record<string, InputValue> = {};
    for (const input of flow.inputs) {
      if (input.type === "bool") next[input.name] = Boolean(input.default);
      else next[input.name] = input.default === null || input.default === undefined ? "" : String(input.default);
    }
    setValues(next);
    const firstPath = flow.inputs.find((input) => input.type === "path");
    setGraphInputs(firstPath ? [firstPath.name] : []);
    setBudget(1);
  }, [flow]);

  const applyStatus = useCallback(async (runId: string) => {
    const generation = generationRef.current;
    const result = await flowRunner.status(runId, LOG_TAIL);
    if (generation !== generationRef.current || runIdRef.current !== runId) return;
    if (!result.ok) {
      setNotice(result.error);
      return;
    }
    setStatus(result);
    if (result.status !== "running") setRunning(false);
  }, []);

  useEffect(
    () =>
      flowRunner.onEvent((event) => {
        // Server-scoped events carry no run id. Filtering them by one would
        // leave the panel waiting forever on a run that cannot report again.
        if (event.kind === "disconnected") {
          setDisconnected(event.reason);
          setRunning(false);
          return;
        }
        if (event.kind === "reconnected") {
          setDisconnected(null);
          const tracked = runIdRef.current;
          if (tracked) void applyStatus(tracked);
          return;
        }
        if (event.runId !== runIdRef.current) return;
        if (event.kind === "status") {
          setStatus(event.status);
          if (event.status.status !== "running") setRunning(false);
          return;
        }
        if (event.kind === "log") {
          setLogLines((previous) => [...previous, event.text].slice(-LOG_LIMIT));
          return;
        }
        const total = event.total === undefined ? "" : ` of ${event.total}`;
        setProgress(event.message ?? `${event.progress}${total} steps`);
      }),
    [applyStatus],
  );

  // Events are the fast path; the poll is what keeps the panel truthful when
  // none arrive. `launched` is a dependency because the tracked run id lives in
  // a ref the event filter reads synchronously: a new launch re-arms this.
  useEffect(() => {
    const tracked = runIdRef.current;
    if (!running || disconnected !== null || !tracked) return;
    const timer = setInterval(() => void applyStatus(tracked), POLL_MS);
    return () => clearInterval(timer);
  }, [running, disconnected, applyStatus, launched]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logLines]);

  const doRun = useCallback(async () => {
    if (!flow) return;
    const generation = ++generationRef.current;
    runIdRef.current = null;
    setRunning(true);
    setStatus(null);
    setLaunched(null);
    setLogLines([]);
    setProgress(null);
    setNotice(null);

    const inputs: Record<string, unknown> = {};
    for (const spec of flow.inputs) {
      if (graphInputs.includes(spec.name)) continue;
      const raw = values[spec.name];
      if (typeof raw === "boolean") {
        inputs[spec.name] = raw;
        continue;
      }
      const text = (raw ?? "").trim();
      if (text === "") {
        if (spec.required) {
          setNotice(`${spec.name} is required by this flow.`);
          setRunning(false);
          return;
        }
        continue;
      }
      if (spec.type === "int" || spec.type === "float") {
        const parsed = Number(text);
        if (!Number.isFinite(parsed)) {
          setNotice(`${spec.name}: "${text}" is not a ${spec.type}.`);
          setRunning(false);
          return;
        }
        inputs[spec.name] = spec.type === "int" ? Math.trunc(parsed) : parsed;
        continue;
      }
      inputs[spec.name] = text;
    }

    const graph = getGraph();
    let graphJson = "";
    if (graphInputs.length > 0) {
      const serialized = await engine.serializeGraph(serializeGraph(graph));
      if (generation !== generationRef.current) return;
      if (!serialized.ok || !serialized.serializedGraph) {
        const code = serialized.error?.code ?? "UNKNOWN";
        setNotice(`Graph serialization failed [${code}]: ${serialized.error?.message ?? ""}`);
        setRunning(false);
        return;
      }
      graphJson = serialized.serializedGraph;
    }

    const request: LaunchRequest = {
      flow: flow.name,
      inputs,
      graphInputs,
      graphJson,
      graphName: graph.name,
      ...(flow.loops.length > 0 ? { maxIterations: budget } : {}),
    };
    const result = await flowRunner.launch(request);
    if (generation !== generationRef.current) return;
    if (!result.ok) {
      setNotice(result.error);
      setRunning(false);
      return;
    }
    runIdRef.current = result.runId;
    setLaunched(result);
    if (result.warnings.length > 0) setNotice(result.warnings.join(" "));
    void applyStatus(result.runId);
  }, [applyStatus, budget, flow, getGraph, graphInputs, values]);

  const doStop = useCallback(() => {
    const tracked = runIdRef.current;
    if (!tracked) return;
    void flowRunner.cancel(tracked).then((result) => {
      if (runIdRef.current !== tracked) return;
      if (!result.ok) {
        setNotice(result.error);
        return;
      }
      setNotice(result.message);
      if (result.state !== "running") setRunning(false);
      void applyStatus(tracked);
    });
  }, [applyStatus]);

  const browse = useCallback(async (name: string) => {
    const opened = await platform.openTextFile();
    // The Electron platform carries the absolute path as the handle token; a
    // browser has no path to give, which is why this is desktop-only.
    if (!opened || typeof opened.handle.token !== "string") return;
    const path = opened.handle.token;
    setValues((previous) => ({ ...previous, [name]: path }));
  }, []);

  const timeline = useMemo(() => (status ? buildTimeline(status) : null), [status]);

  const budgetCap = useMemo(
    () => flow?.loops.reduce((most, loop) => Math.max(most, loop.maxIterations), 1) ?? 1,
    [flow],
  );

  // Worst case if every loop runs its full budget and no step is skipped.
  const worstCase = useMemo(() => {
    if (!flow) return { steps: 0, seconds: 0 };
    let steps = 0;
    let seconds = 0;
    const passesOf = new Map(
      flow.loops.map((loop) => [loop.id, Math.min(budget, loop.maxIterations)] as const),
    );
    for (const step of flow.steps) {
      const passes = step.group === null ? 1 : passesOf.get(step.group) ?? 1;
      steps += passes;
      seconds += (step.timeout ?? 0) * passes;
    }
    return { steps, seconds };
  }, [budget, flow]);

  const unavailable = !flowRunner.available;
  const controlsDisabled = unavailable || running || disconnected !== null;

  return (
    <div className="authoring">
      <div className="authoring__bar">
        <label className="authoring__label" htmlFor="authoring-flow">
          Flow:
        </label>
        <select
          id="authoring-flow"
          className="authoring__select"
          value={selectedName}
          disabled={controlsDisabled || (flows?.length ?? 0) === 0}
          onChange={(event) => setSelectedName(event.target.value)}
        >
          {(flows ?? []).map((candidate) => (
            <option key={candidate.name} value={candidate.name}>
              {candidate.name}
            </option>
          ))}
          {(flows?.length ?? 0) === 0 && <option value="">No flow available</option>}
        </select>
        <button
          type="button"
          className="authoring__refresh"
          onClick={() => void refresh()}
          disabled={unavailable || listing || running}
        >
          {listing ? "Listing…" : "Refresh"}
        </button>
        {flow && flow.loops.length > 0 && (
          <>
            <label className="authoring__label" htmlFor="authoring-iterations">
              Iterations:
            </label>
            <input
              id="authoring-iterations"
              className="authoring__number"
              type="number"
              min={1}
              max={budgetCap}
              value={budget}
              disabled={controlsDisabled}
              onChange={(event) => {
                const parsed = Number(event.target.value);
                if (Number.isFinite(parsed)) setBudget(Math.min(Math.max(1, Math.trunc(parsed)), budgetCap));
              }}
            />
            <span className="authoring__cap">of {budgetCap} declared</span>
          </>
        )}
        <button
          type="button"
          className="authoring__run"
          data-running={running}
          onClick={() => (running ? doStop() : void doRun())}
          disabled={unavailable || !flow || disconnected !== null}
        >
          {disconnected !== null ? "Server disconnected" : running ? "Stop" : "Run"}
        </button>
      </div>

      {unavailable && <div className="authoring__banner">{FLOW_UNAVAILABLE}</div>}
      {disconnected !== null && (
        <div className="authoring__banner authoring__banner--error">{disconnected}</div>
      )}
      {listError && <div className="authoring__banner authoring__banner--error">{listError}</div>}
      {flow?.loadError && (
        <div className="authoring__banner authoring__banner--error">{flow.loadError}</div>
      )}
      {notice && <div className="authoring__banner">{notice}</div>}

      <div className="authoring__body">
        <div className="authoring__form">
          {flow?.description && <p className="authoring__desc">{flow.description}</p>}
          {flow?.inputs.map((input) => (
            <InputField
              key={input.name}
              input={input}
              value={values[input.name] ?? ""}
              usesGraph={graphInputs.includes(input.name)}
              disabled={controlsDisabled}
              onValue={(next) => setValues((previous) => ({ ...previous, [input.name]: next }))}
              onUseGraph={(use) =>
                setGraphInputs((previous) =>
                  use
                    ? [...previous, input.name]
                    : previous.filter((name) => name !== input.name),
                )
              }
              onBrowse={() => void browse(input.name)}
            />
          ))}
          {flow && flow.inputs.length === 0 && (
            <p className="authoring__desc">This flow declares no inputs.</p>
          )}
          {flow && (
            <p className="authoring__cost">
              Worst case: {worstCase.steps} step execution{worstCase.steps === 1 ? "" : "s"}
              {worstCase.seconds > 0 && `, up to ${Math.round(worstCase.seconds / 60)} minutes of declared timeouts`}
            </p>
          )}
        </div>

        <div className="authoring__run-view">
          {launched && (
            <div className="authoring__runinfo">
              <span className="authoring__runid">{launched.runId}</span>
              <span className="authoring__dim">{launched.runDir}</span>
              {launched.graphPath && <span className="authoring__dim">{launched.graphPath}</span>}
              {progress && <span className="authoring__dim">{progress}</span>}
            </div>
          )}
          {timeline ? (
            <>
              <div className="authoring__terminal" data-status={timeline.terminal.statusKind}>
                <span className="authoring__headline">{timeline.terminal.headline}</span>
                {timeline.durationText && (
                  <span className="authoring__dim">{timeline.durationText}</span>
                )}
                {timeline.terminal.detail.map((line) => (
                  <span className="authoring__detail" key={line}>
                    {line}
                  </span>
                ))}
              </div>
              {timeline.caution && <div className="authoring__caution">{CAUTION}</div>}
              <FlowTimeline timeline={timeline} />
            </>
          ) : (
            <div className="authoring__empty">
              {unavailable ? FLOW_UNAVAILABLE : "No run yet."}
            </div>
          )}
          {logLines.length > 0 && (
            <div className="authoring__log">
              <h4 className="authoring__log-title">Run log (supplementary — the timeline above is the state)</h4>
              <div className="authoring__log-body" ref={logRef}>
                {logLines.map((line, index) => (
                  <span className="authoring__log-line" key={index}>
                    {line}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

interface InputFieldProps {
  input: FlowInputSpec;
  value: InputValue;
  usesGraph: boolean;
  disabled: boolean;
  onValue(next: InputValue): void;
  onUseGraph(use: boolean): void;
  onBrowse(): void;
}

/** One control, chosen from the type the flow declared for the input. */
function InputField({ input, value, usesGraph, disabled, onValue, onUseGraph, onBrowse }: InputFieldProps) {
  const id = `authoring-input-${input.name}`;
  const isPath = input.type === "path";
  const numeric = input.type === "int" || input.type === "float";
  return (
    <div className="authoring__field">
      <label className="authoring__field-label" htmlFor={id}>
        {input.name}
        {input.required && <span className="authoring__required">required</span>}
        <span className="authoring__field-type">{input.type}</span>
      </label>
      {input.type === "bool" ? (
        <input
          id={id}
          type="checkbox"
          checked={value === true}
          disabled={disabled}
          onChange={(event) => onValue(event.target.checked)}
        />
      ) : (
        <div className="authoring__field-row">
          <input
            id={id}
            className="authoring__input"
            type={numeric ? "number" : "text"}
            step={input.type === "float" ? "any" : undefined}
            value={typeof value === "boolean" ? "" : value}
            spellCheck={false}
            disabled={disabled || usesGraph}
            placeholder={usesGraph ? "the current canvas graph" : ""}
            onChange={(event) => onValue(event.target.value)}
          />
          {isPath && (
            <button
              type="button"
              className="authoring__browse"
              onClick={onBrowse}
              disabled={disabled || usesGraph}
            >
              Browse…
            </button>
          )}
        </div>
      )}
      {isPath && (
        <label className="authoring__usegraph">
          <input
            type="checkbox"
            checked={usesGraph}
            disabled={disabled}
            onChange={(event) => onUseGraph(event.target.checked)}
          />
          Use the current canvas graph
        </label>
      )}
      {input.description && <p className="authoring__help">{input.description}</p>}
      {isPath && input.exists && (
        <p className="authoring__help">Checked for existence before the run starts.</p>
      )}
    </div>
  );
}
