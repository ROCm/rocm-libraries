// Which flow input the Implement tab may bind the canvas to (src/flow/canvas.ts).
// Bun's built-in test runner; no framework dependency.
//
// This is the seam behind a control the user can tick. The panel offers the
// "use the current canvas graph" toggle on exactly the input this returns, and
// binds that same input on selection, so a wrong answer here is either a
// binding the panel offers and then will not make, or -- the failure that
// actually happened -- a graph file delivered as a prior run directory.
//
// The specs below are shaped like the real pipeline flows rather than minimal:
// several path inputs of which only one is a graph, plus non-path inputs that
// must never be candidates however they are named.
import { expect, test } from "bun:test";

import { canvasInputName } from "../src/flow/canvas";
import type { FlowInputSpec } from "../src/flow/types";

function input(name: string, type: FlowInputSpec["type"]): FlowInputSpec {
  return { name, type, required: false, description: "", default: null, exists: false };
}

test("the input named graph wins over every other path input", () => {
  const inputs = [
    input("engine_name", "string"),
    input("prior_run", "path"),
    input("graph", "path"),
    input("corpus_dir", "path"),
  ];

  expect(canvasInputName(inputs)).toBe("graph");
});

test("declaration order does not decide it", () => {
  const first = [input("graph", "path"), input("prior_run", "path")];
  const last = [input("prior_run", "path"), input("graph", "path")];

  expect(canvasInputName(first)).toBe("graph");
  expect(canvasInputName(last)).toBe("graph");
});

test("a sole path input takes the canvas whatever it is called", () => {
  const inputs = [input("kernel", "path"), input("notes", "string")];

  expect(canvasInputName(inputs)).toBe("kernel");
});

test("several unnamed path inputs disambiguate to none rather than to the first", () => {
  // The regression: this shape used to answer `prior_run`, and the panel then
  // offered to hand a resume flow its canvas as the run directory to adopt.
  const inputs = [input("prior_run", "path"), input("corpus_dir", "path")];

  expect(canvasInputName(inputs)).toBeNull();
});

test("a flow with no path inputs offers the canvas nowhere", () => {
  const inputs = [input("engine_name", "string"), input("tune", "bool")];

  expect(canvasInputName(inputs)).toBeNull();
});

test("an input named graph that is not a path is not a candidate", () => {
  // `graph` as a string is a name, not a file the canvas can be written to.
  const inputs = [input("graph", "string"), input("kernel", "path")];

  expect(canvasInputName(inputs)).toBe("kernel");
});
