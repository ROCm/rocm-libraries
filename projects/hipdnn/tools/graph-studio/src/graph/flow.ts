import type { Edge, Node } from "@xyflow/react";
import { catalogEntry, defaultParams } from "./catalog";
import { GRAPH_VERSION, type Graph, type ParamValue } from "./model";

/**
 * Bridge between the on-disk {@link Graph} model and React Flow's node/edge
 * arrays. React Flow owns live canvas state; this module converts to and from
 * the serialization-friendly domain model at load/save boundaries.
 *
 * Every RF node uses the single custom node type "op"; the concrete operator
 * kind lives in `data.opType` so one renderer covers the whole catalog.
 */

export interface OpNodeData extends Record<string, unknown> {
  opType: string;
  title: string;
  params: Record<string, ParamValue>;
}

export type OpNode = Node<OpNodeData, "op">;
export type OpEdge = Edge;

export const OP_NODE_TYPE = "op" as const;

let idCounter = 0;
function nextId(prefix: string): string {
  idCounter += 1;
  return `${prefix}_${Date.now().toString(36)}_${idCounter}`;
}

export function createOpNode(opType: string, x: number, y: number): OpNode {
  const entry = catalogEntry(opType);
  return {
    id: nextId("n"),
    type: OP_NODE_TYPE,
    position: { x, y },
    data: {
      opType,
      title: entry?.label ?? opType,
      params: defaultParams(opType),
    },
  };
}

export function newEdgeId(): string {
  return nextId("e");
}

/** Convert live React Flow state into the serializable domain graph. */
export function toGraph(name: string, nodes: OpNode[], edges: OpEdge[]): Graph {
  return {
    version: GRAPH_VERSION,
    name,
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.data.opType,
      title: n.data.title,
      x: Math.round(n.position.x),
      y: Math.round(n.position.y),
      params: n.data.params,
    })),
    edges: edges.map((e) => ({
      id: e.id,
      from: { node: e.source, port: e.sourceHandle ?? "out" },
      to: { node: e.target, port: e.targetHandle ?? "in" },
    })),
  };
}

/** Convert a loaded domain graph into React Flow node/edge arrays. */
export function fromGraph(graph: Graph): { nodes: OpNode[]; edges: OpEdge[] } {
  const nodes: OpNode[] = graph.nodes.map((n) => ({
    id: n.id,
    type: OP_NODE_TYPE,
    position: { x: n.x, y: n.y },
    data: {
      opType: n.type,
      title: n.title,
      params: n.params ?? defaultParams(n.type),
    },
  }));
  const edges: OpEdge[] = graph.edges.map((e) => ({
    id: e.id,
    source: e.from.node,
    sourceHandle: e.from.port,
    target: e.to.node,
    targetHandle: e.to.port,
  }));
  return { nodes, edges };
}
