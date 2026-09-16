/**
 * Core domain types for a hipDNN computation graph.
 *
 * A graph is a set of operator nodes connected by edges. Each node has typed
 * input/output ports and a bag of parameters described by its catalog entry.
 * The shape is deliberately serialization-friendly (plain data, no class
 * instances) so a graph round-trips through JSON without custom (de)serializers.
 */

export type ParamType = "int" | "float" | "string" | "bool" | "enum";

export interface ParamSpec {
  readonly key: string;
  readonly label: string;
  readonly type: ParamType;
  readonly default: ParamValue;
  /** Allowed values when type is "enum". */
  readonly options?: readonly string[];
  /** Render this param only while every listed param of the same node holds `equals`. */
  readonly visibleWhen?: readonly { readonly key: string; readonly equals: ParamValue }[];
}

export type ParamValue = number | string | boolean;

export interface PortSpec {
  readonly id: string;
  readonly label: string;
}

/** Static description of an operator kind (Conv2d, ReLU, ...). */
export interface OpCatalogEntry {
  readonly type: string;
  readonly label: string;
  readonly category: string;
  readonly accent: string;
  readonly inputs: readonly PortSpec[];
  readonly outputs: readonly PortSpec[];
  readonly params: readonly ParamSpec[];
}

export interface GraphNode {
  readonly id: string;
  readonly type: string;
  title: string;
  x: number;
  y: number;
  params: Record<string, ParamValue>;
}

export interface GraphEdge {
  readonly id: string;
  readonly from: { node: string; port: string };
  readonly to: { node: string; port: string };
}

export interface Graph {
  readonly version: 1;
  name: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export const GRAPH_VERSION = 1 as const;

export function emptyGraph(name = "Untitled Graph"): Graph {
  return { version: GRAPH_VERSION, name, nodes: [], edges: [] };
}
