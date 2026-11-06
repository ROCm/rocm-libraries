// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// A bundle's text split into what stays put (the st.func line, block headers, Successors
// lines) and the instruction lines the editor reorders. The line classes are the checkpoint
// loader's: an IR line is any line inside a block that is not blank, a comment, or Successors.

const ID_RE = /mod\.ckpt = \{ id = (\d+) \}/;

export function classify(line) {
  const t = line.trim();
  if (t === "}") return "end";
  if (!t || t.startsWith("//") || t.startsWith("#") || t.startsWith("st.func") ||
      t.startsWith("Successors:")) {
    return "other";
  }
  if (t.startsWith("^") && t.endsWith(":")) return "block";
  return "node";
}

/** The checkpoint id a line carries, or null. */
export function idOfLine(line) {
  const match = ID_RE.exec(line);
  return match ? Number(match[1]) : null;
}

/**
 * { prefix, blocks: [{ label, header, lead, nodes, trailer }], suffix }: `nodes` are the IR
 * lines; `lead` and `trailer` the other lines before and after them inside the block.
 */
export function parseBundle(text) {
  const doc = { prefix: [], blocks: [], suffix: [] };
  const lines = text.split("\n");
  if (lines.length && lines[lines.length - 1] === "") lines.pop();
  let block = null;
  let ended = false;
  for (const line of lines) {
    const kind = classify(line);
    if (ended) {
      doc.suffix.push(line);
    } else if (kind === "end") {
      ended = true;
      doc.suffix.push(line);
    } else if (kind === "block") {
      block = { label: line.trim().slice(1, -1), header: line, lead: [], nodes: [], trailer: [] };
      doc.blocks.push(block);
    } else if (!block) {
      doc.prefix.push(line);
    } else if (kind === "node") {
      block.nodes.push(line);
    } else {
      (block.nodes.length ? block.trailer : block.lead).push(line);
    }
  }
  return doc;
}

/** The text of `doc` with each block's IR lines replaced by `blockLines[b]`. */
export function renderBundle(doc, blockLines) {
  const out = [...doc.prefix];
  doc.blocks.forEach((block, b) => {
    out.push(block.header, ...block.lead, ...blockLines[b], ...block.trailer);
  });
  out.push(...doc.suffix);
  return out.join("\n") + "\n";
}

/** The ids of each block's IR lines, in text order; lines without an id are left out. */
export function blockIdOrders(text) {
  return parseBundle(text).blocks.map((block) =>
    block.nodes.map(idOfLine).filter((id) => id !== null));
}
