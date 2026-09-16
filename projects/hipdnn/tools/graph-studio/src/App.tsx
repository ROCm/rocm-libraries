import {
  addEdge,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Connection,
  type EdgeChange,
  type NodeChange,
  type OnConnect,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ImplementPanel } from "./components/ImplementPanel";
import { TensorView, reportTensorHints, type TensorHint } from "./components/TensorView";
import { VerifyReport, type ShownReport } from "./components/VerifyReport";
import { CommandPanel, type CommandField, type CommandOption } from "./components/CommandPanel";
import { Inspector } from "./components/Inspector";
import { EnginePanel } from "./components/EnginePanel";
import { MainTabs, TabPanel, type TabId } from "./components/MainTabs";
import { DND_MIME, Palette } from "./components/Palette";
import { OpNodeView } from "./components/OpNodeView";
import { Toolbar } from "./components/Toolbar";
import { catalogEntry } from "./graph/catalog";
import {
  createOpNode,
  fromGraph,
  newEdgeId,
  OP_NODE_TYPE,
  toGraph,
  type OpEdge,
  type OpNode,
} from "./graph/flow";
import { emptyGraph } from "./graph/model";
import { parseGraph, serializeGraph } from "./graph/serialize";
import type { ParamValue } from "./graph/model";
import type { DirectoryRef, FileHandleRef, ReadBase } from "./platform/types";
import { platform } from "./platform";
import { fromNativeExecution, type NativeExecutionSnapshot } from "./benchmark/native";
import { parseReport } from "./benchmark/report";
import { GRAPH_PLACEHOLDER, RUN_DIR_PLACEHOLDER } from "./command";

const AUTOSAVE_KEY = "hipdnn.graph.autosave";

// The Verify tab benchmarks the canvas graph. `dnn-benchmark` is installed
// beside Studio and is on the PATH the launcher establishes for child
// processes, so the bare name resolves on both platforms.
//
// `--run-dir` puts the report, the tensor captures, and any profiling output in
// one directory, which is what lets a trace and its tensors load with the
// report instead of needing a second pick.
const VERIFY_COMMAND =
  `dnn-benchmark --graph ${GRAPH_PLACEHOLDER} --run-dir ${RUN_DIR_PLACEHOLDER}`;

const VERIFY_OPTIONS: readonly CommandOption[] = [
  {
    id: "validate",
    label: "Validate against PyTorch",
    args: "--validate pytorch",
    hint: "Compare every engine's outputs against PyTorch, and time PyTorch as a reference row.",
  },
  {
    id: "trace",
    label: "Capture kernel trace",
    args: "--emit-trace pftrace",
    hint: "Re-run under rocprofv3 and record a Perfetto trace, openable from a row's Details. Adds about one extra run.",
  },
  {
    id: "autotune",
    label: "Autotune kernels",
    args: "--autotune",
    hint: "Benchmark every candidate kernel instead of trusting the cold heuristic. Required for a best-vs-best comparison; much slower.",
  },
  {
    id: "perf",
    label: "CPU counters",
    args: "--perf",
    hint: "Re-run under 'perf stat' for cycles, instructions and IPC, shown under a row's Details. Adds about one extra run.",
  },
];

const VERIFY_FIELDS: readonly CommandField[] = [
  {
    id: "iters",
    label: "Iterations",
    flag: "--iters",
    placeholder: "100",
    hint: "Timed iterations per engine. Fewer is faster and noisier.",
  },
  {
    id: "warmup",
    label: "Warmup",
    flag: "--warmup",
    placeholder: "10",
    hint: "Untimed iterations run first, so compilation and clocks settle before measurement.",
  },
];

function Studio() {
  const nodeTypes = useMemo(() => ({ [OP_NODE_TYPE]: OpNodeView }), []);
  const flow = useReactFlow<OpNode, OpEdge>();

  const [nodes, setNodes, onNodesChange] = useNodesState<OpNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<OpEdge>([]);
  const [graphName, setGraphName] = useState("Untitled Graph");
  const [fileHandle, setFileHandle] = useState<FileHandleRef | null>(null);
  const [dirty, setDirty] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  // Bumped whenever the graph is replaced (New/Open) so the engine panel drops
  // any built plan and resets its Build/Execute state.
  const [engineResetKey, setEngineResetKey] = useState(0);
  const [activeTab, setActiveTab] = useState<TabId>("create");
  // The report on screen: the freshest run this session produced — the Create
  // tab's native Execute or the Verify tab's benchmark run — or one opened from
  // disk. Held here so the Tensors tab inspects the same run Verify is showing.
  const [currentReport, setCurrentReport] = useState<ShownReport | null>(null);
  const [currentError, setCurrentError] = useState<string | null>(null);
  // Set when a report row asks for its own captures, which is narrower than the
  // whole report's. Cleared whenever the report changes.
  const [rowHints, setRowHints] = useState<readonly TensorHint[] | null>(null);
  // A folder granted this session. It outlives whichever report is on screen,
  // and both the Verify and Tensors tabs resolve artifacts against it.
  const [granted, setGranted] = useState<DirectoryRef | null>(null);
  const nextNativeId = useRef(0);
  const showReport = useCallback((next: ShownReport | null, error: string | null) => {
    setCurrentReport(next);
    setCurrentError(error);
    setRowHints(null);
  }, []);
  const onExecutionResult = useCallback(
    (snapshot: NativeExecutionSnapshot) =>
      showReport(
        {
          report: fromNativeExecution(snapshot, `native-${++nextNativeId.current}`),
          label: "execution",
          handle: null,
        },
        null,
      ),
    [showReport],
  );
  const onResultsReset = useCallback(() => showReport(null, null), [showReport]);

  const onVerifyResults = useCallback(
    (json: string | null, reportPath: string) => {
      if (json === null) {
        showReport(null, "The run produced no report.");
        return;
      }
      try {
        showReport(
          {
            report: parseReport(json),
            label: "benchmark run",
            // The report is a real file on this machine, so it is its own base:
            // every artifact path it carries is anchored to its directory.
            handle: { name: reportPath.split(/[\\/]/).pop() ?? "results.json", token: reportPath },
          },
          null,
        );
      } catch (failure) {
        showReport(null, `Could not read the run's results: ${(failure as Error).message}`);
      }
    },
    [showReport],
  );

  const requestDirectory = useCallback(async () => {
    const dir = await platform.openDirectory();
    if (dir) setGranted(dir);
  }, []);

  // The Tensors tab follows the report on screen: a row's captures when one was
  // asked for, otherwise everything the report recorded.
  const tensorHints = useMemo(
    () => rowHints ?? (currentReport ? reportTensorHints(currentReport.report) : undefined),
    [currentReport, rowHints],
  );
  const openRowTensors = useCallback((hints: readonly TensorHint[]) => {
    setRowHints(hints);
    setActiveTab("tensors");
  }, []);
  const runBase: ReadBase | null = granted ?? currentReport?.handle ?? null;

  const markDirty = useCallback(() => setDirty(true), []);

  // Restore the last session's graph from the platform store on first mount.
  useEffect(() => {
    let cancelled = false;
    void platform.store.get(AUTOSAVE_KEY).then((saved) => {
      if (cancelled || !saved) return;
      try {
        const graph = parseGraph(saved);
        const restored = fromGraph(graph);
        setNodes(restored.nodes);
        setEdges(restored.edges);
        setGraphName(graph.name);
      } catch {
        // Ignore a corrupt autosave; start fresh.
      }
    });
    return () => {
      cancelled = true;
    };
  }, [setEdges, setNodes]);

  // Debounced autosave whenever the graph changes.
  useEffect(() => {
    const timer = setTimeout(() => {
      const graph = toGraph(graphName, nodes, edges);
      void platform.store.set(AUTOSAVE_KEY, serializeGraph(graph));
    }, 600);
    return () => clearTimeout(timer);
  }, [nodes, edges, graphName]);

  const handleNodesChange = useCallback(
    (changes: NodeChange<OpNode>[]) => {
      onNodesChange(changes);
      if (changes.some((c) => c.type !== "select" && c.type !== "dimensions")) markDirty();
      // A single batch can carry both a deselect (old node) and a select (new
      // node); their order isn't guaranteed. Fold them so a `selected:true`
      // always wins and a deselect only clears the node it names.
      const selectChanges = changes.filter((c) => c.type === "select");
      if (selectChanges.length > 0) {
        setSelectedId((prev) => {
          let next = prev;
          for (const c of selectChanges) {
            if (!("id" in c)) continue;
            if (c.selected) next = c.id;
            else if (next === c.id) next = null;
          }
          return next;
        });
      }
    },
    [onNodesChange, markDirty],
  );

  const handleEdgesChange = useCallback(
    (changes: EdgeChange<OpEdge>[]) => {
      onEdgesChange(changes);
      if (changes.some((c) => c.type !== "select")) markDirty();
    },
    [onEdgesChange, markDirty],
  );

  const onConnect = useCallback<OnConnect>(
    (connection: Connection) => {
      setEdges((current) => {
        // An input port holds one edge: drop any existing edge into the target.
        const filtered = current.filter(
          (e) => !(e.target === connection.target && e.targetHandle === connection.targetHandle),
        );
        return addEdge({ ...connection, id: newEdgeId() }, filtered);
      });
      markDirty();
    },
    [setEdges, markDirty],
  );

  const addNodeAt = useCallback(
    (opType: string, x: number, y: number) => {
      if (!catalogEntry(opType)) return;
      const node = createOpNode(opType, x, y);
      setNodes((current) => [...current, node]);
      markDirty();
    },
    [setNodes, markDirty],
  );

  const onPaletteAdd = useCallback(
    (opType: string) => {
      // Drop click-added nodes near the current viewport centre.
      const pos = flow.screenToFlowPosition({
        x: window.innerWidth / 2,
        y: window.innerHeight / 2,
      });
      addNodeAt(opType, pos.x - 80, pos.y - 30);
    },
    [flow, addNodeAt],
  );

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const opType = event.dataTransfer.getData(DND_MIME);
      if (!opType) return;
      const pos = flow.screenToFlowPosition({ x: event.clientX, y: event.clientY });
      addNodeAt(opType, pos.x - 80, pos.y - 30);
    },
    [flow, addNodeAt],
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const selectedNode = useMemo(
    () => nodes.find((n) => n.id === selectedId) ?? null,
    [nodes, selectedId],
  );

  const getGraph = useCallback(
    () => toGraph(graphName, nodes, edges),
    [graphName, nodes, edges],
  );

  const patchNodeData = useCallback(
    (id: string, patch: (data: OpNode["data"]) => OpNode["data"]) => {
      setNodes((current) =>
        current.map((n) => (n.id === id ? { ...n, data: patch(n.data) } : n)),
      );
      markDirty();
    },
    [setNodes, markDirty],
  );

  const onRename = useCallback(
    (id: string, title: string) => patchNodeData(id, (data) => ({ ...data, title })),
    [patchNodeData],
  );

  const onParamChange = useCallback(
    (id: string, key: string, value: ParamValue) =>
      patchNodeData(id, (data) => ({ ...data, params: { ...data.params, [key]: value } })),
    [patchNodeData],
  );

  const onDeleteNode = useCallback(
    (id: string) => {
      setNodes((current) => current.filter((n) => n.id !== id));
      setEdges((current) => current.filter((e) => e.source !== id && e.target !== id));
      setSelectedId((prev) => (prev === id ? null : prev));
      markDirty();
    },
    [setNodes, setEdges, markDirty],
  );

  const doNew = useCallback(() => {
    if (dirty && !window.confirm("Discard unsaved changes and start a new graph?")) return;
    const graph = emptyGraph();
    setNodes([]);
    setEdges([]);
    setGraphName(graph.name);
    setFileHandle(null);
    setSelectedId(null);
    setDirty(false);
    setEngineResetKey((k) => k + 1);
  }, [dirty, setNodes, setEdges]);

  const doOpen = useCallback(async () => {
    if (dirty && !window.confirm("Discard unsaved changes and open a file?")) return;
    try {
      const result = await platform.openTextFile();
      if (!result) return;
      const graph = parseGraph(result.contents);
      const loaded = fromGraph(graph);
      setNodes(loaded.nodes);
      setEdges(loaded.edges);
      setGraphName(graph.name);
      setFileHandle(result.handle);
      setSelectedId(null);
      setDirty(false);
      setEngineResetKey((k) => k + 1);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to open file.";
      window.alert(message);
    }
  }, [dirty, setNodes, setEdges]);

  const saveWith = useCallback(
    async (handle: FileHandleRef | null) => {
      const graph = toGraph(graphName, nodes, edges);
      const written = await platform.saveTextFile(serializeGraph(graph), {
        suggestedName: `${graphName.replace(/\s+/g, "_") || "graph"}.json`,
        ...(handle ? { handle } : {}),
      });
      if (!written) return;
      setFileHandle(written);
      setDirty(false);
    },
    [graphName, nodes, edges],
  );

  const doSave = useCallback(() => void saveWith(fileHandle), [saveWith, fileHandle]);
  const doSaveAs = useCallback(() => void saveWith(null), [saveWith]);

  // Keyboard: Ctrl/Cmd+S to save.
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        void saveWith(fileHandle);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [saveWith, fileHandle]);

  const emptyHintRef = useRef<HTMLDivElement>(null);

  return (
    <div className="app">
      <Toolbar
        graphName={graphName}
        fileName={fileHandle?.name ?? null}
        dirty={dirty}
        onRenameGraph={(name) => {
          setGraphName(name);
          markDirty();
        }}
        onNew={doNew}
        onOpen={() => void doOpen()}
        onSave={doSave}
        onSaveAs={doSaveAs}
      />
      <MainTabs active={activeTab} onSelect={setActiveTab} />
      <TabPanel id="create" active={activeTab} className="workspace">
        <div className="app__body">
          <Palette onAdd={onPaletteAdd} />
          <div className="canvas" onDrop={onDrop} onDragOver={onDragOver}>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              nodeTypes={nodeTypes}
              onNodesChange={handleNodesChange}
              onEdgesChange={handleEdgesChange}
              onConnect={onConnect}
              onPaneClick={() => setSelectedId(null)}
              onNodesDelete={(deleted) =>
                setSelectedId((prev) => (deleted.some((n) => n.id === prev) ? null : prev))
              }
              deleteKeyCode={["Delete", "Backspace"]}
              fitView
              proOptions={{ hideAttribution: true }}
              defaultEdgeOptions={{ animated: true }}
            >
              <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#2b3350" />
              <MiniMap
                pannable
                zoomable
                className="canvas__minimap"
                style={{ background: "#12182a" }}
                maskColor="rgba(13, 17, 23, 0.7)"
                nodeColor={(node) => catalogEntry((node.data as OpNode["data"]).opType)?.accent ?? "#64748b"}
                nodeStrokeWidth={0}
              />
              <Controls className="canvas__controls" />
            </ReactFlow>
            {nodes.length === 0 && (
              <div className="canvas__hint" ref={emptyHintRef}>
                Drag an operator from the left, or click one to add it.
              </div>
            )}
          </div>
          <div className="rightbar">
            <Inspector
              node={selectedNode}
              onRename={onRename}
              onParamChange={onParamChange}
              onDelete={onDeleteNode}
            />
          </div>
        </div>
        <EnginePanel
          getGraph={getGraph}
          resetKey={engineResetKey}
          onExecutionResult={onExecutionResult}
          onResultsReset={onResultsReset}
        />
      </TabPanel>
      <TabPanel id="implement" active={activeTab}>
        <ImplementPanel getGraph={getGraph} active={activeTab === "implement"} />
      </TabPanel>
      <TabPanel id="verify" active={activeTab}>
        <CommandPanel
          scope="verify"
          getGraph={getGraph}
          command={VERIFY_COMMAND}
          runLabel="Start Benchmarking"
          options={VERIFY_OPTIONS}
          fields={VERIFY_FIELDS}
          onResults={onVerifyResults}
        >
          <VerifyReport
            shown={currentReport}
            shownError={currentError}
            onShow={showReport}
            onOpenTensors={openRowTensors}
            granted={granted}
            onGranted={setGranted}
          />
        </CommandPanel>
      </TabPanel>
      <TabPanel id="tensors" active={activeTab}>
        <TensorView hints={tensorHints} base={runBase} onGrantDirectory={requestDirectory} />
      </TabPanel>
    </div>
  );
}

export function App() {
  return (
    <ReactFlowProvider>
      <Studio />
    </ReactFlowProvider>
  );
}
