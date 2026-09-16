import type { FlowInputSpec } from "./types";

/**
 * Which of a flow's inputs, if any, the canvas may be bound to.
 *
 * One input at most. A pipeline declares several `path` inputs -- a corpus
 * directory, a prior run directory, a graph -- and the canvas can satisfy
 * exactly one of them. Offering it against the others produces a value that is
 * a real file and the wrong thing entirely: a resume flow handed its canvas as
 * `prior_run` looked for an authoring contract underneath a graph file.
 *
 * The input named `graph` wins. A flow that declares no such input falls back
 * to its sole path input, so a single-input flow still works without ceremony;
 * with several unnamed candidates there is nothing to disambiguate on, and the
 * answer is none rather than a guess.
 */
export function canvasInputName(inputs: readonly FlowInputSpec[]): string | null {
  const paths = inputs.filter((input) => input.type === "path");
  const named = paths.find((input) => input.name === "graph");
  if (named) return named.name;
  return paths.length === 1 ? paths[0].name : null;
}
