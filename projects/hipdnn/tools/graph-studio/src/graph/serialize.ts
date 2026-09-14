import { GRAPH_VERSION, type Graph } from "./model";

/** Serialize a graph to pretty JSON for on-disk storage. */
export function serializeGraph(graph: Graph): string {
  return JSON.stringify(graph, null, 2);
}

export class GraphParseError extends Error {}

/** Parse and validate a graph from JSON text. Throws GraphParseError on bad input. */
export function parseGraph(text: string): Graph {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (cause) {
    throw new GraphParseError("File is not valid JSON.", { cause });
  }

  if (typeof raw !== "object" || raw === null) {
    throw new GraphParseError("Graph root must be an object.");
  }
  const obj = raw as Record<string, unknown>;
  if (obj.version !== GRAPH_VERSION) {
    throw new GraphParseError(`Unsupported graph version: ${String(obj.version)}.`);
  }
  if (!Array.isArray(obj.nodes) || !Array.isArray(obj.edges)) {
    throw new GraphParseError("Graph must have `nodes` and `edges` arrays.");
  }

  return {
    version: GRAPH_VERSION,
    name: typeof obj.name === "string" ? obj.name : "Untitled Graph",
    nodes: obj.nodes as Graph["nodes"],
    edges: obj.edges as Graph["edges"],
  };
}
