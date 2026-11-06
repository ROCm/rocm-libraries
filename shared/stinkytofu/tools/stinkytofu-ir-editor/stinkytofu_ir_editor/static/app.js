// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The editor page. The model is one order of ids per block; every row's text is the bundle line
// it came from, so the saved file differs from the dump only in order (and raw-tab edits).

import { blockIdOrders, parseBundle, renderBundle } from "./bundle_text.js";
import {
  blockingConstraint,
  buildIndex,
  currentGap,
  describeConstraint,
  legalRange,
  moveGroup,
  movedIds,
} from "./legal_range.js";

const EDGE_KINDS = ["raw", "war", "waw"];
const $ = (id) => document.getElementById(id);

const state = {
  bundles: [],
  bundle: null, // GET /api/bundle/<id>
  baseOrders: null, // ids per block as dumped; null after a raw edit
  doc: null, // parseBundle of the text the rows came from
  orders: [], // ids per block, current
  lineOf: new Map(), // id -> bundle line
  insts: new Map(), // id -> analyze row
  index: { preds: new Map(), succs: new Map() },
  pinned: new Map(), // id -> reason
  selected: new Set(),
  anchor: null,
  undo: [],
  redo: [],
  drag: null,
  tab: "blocks",
  rawText: null,
};

// ---- server ----------------------------------------------------------------------------------

async function api(path, body) {
  const response = await fetch(path, body === undefined ? undefined : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.error || response.statusText);
    error.data = data;
    throw error;
  }
  return data;
}

function status(message, kind = "") {
  const bar = $("status");
  bar.textContent = message;
  bar.className = kind;
}

// ---- model -----------------------------------------------------------------------------------

function currentText() {
  return renderBundle(state.doc, state.orders.map((order) => order.map((id) => state.lineOf.get(id))));
}

function blockOf(id) {
  return state.insts.get(id).block;
}

function opcodeOf(id) {
  return state.insts.get(id)?.opcode ?? "";
}

/** Analyze `text` and make it the model. Rows map to analyze's insts in text order. */
async function adopt(text, { keepBase }) {
  const deps = await api("/api/analyze", { id: state.bundle.id, stir: text });
  const doc = parseBundle(text);
  const lineOf = new Map();
  let next = 0;
  const orders = doc.blocks.map((block) => block.nodes.map((line) => {
    const inst = deps.insts[next++];
    lineOf.set(inst.id, line);
    return inst.id;
  }));
  if (next !== deps.insts.length) throw new Error("the analysis and the text disagree on the lines");

  Object.assign(state, {
    doc,
    orders,
    lineOf,
    insts: new Map(deps.insts.map((inst) => [inst.id, inst])),
    index: buildIndex(deps.edges),
    pinned: new Map(deps.insts.filter((i) => i.pinned).map((i) => [i.id, i.pin_reason])),
    selected: new Set(),
    anchor: null,
    undo: [],
    redo: [],
    rawText: null,
  });
  state.edgeCount = deps.edges.length;
  if (!keepBase) state.baseOrders = null;
  hideCheck();
  render();
}

async function openBundle(id, { saved = false } = {}) {
  status(`loading ${id} ...`);
  state.bundle = await api(`/api/bundle/${encodeURIComponent(id)}`);
  state.baseOrders = blockIdOrders(state.bundle.stir);
  const text = saved && state.bundle.saved_stir ? state.bundle.saved_stir : state.bundle.stir;
  await adopt(text, { keepBase: true });
  $("open-saved").disabled = !state.bundle.saved_stir;
  status(`${state.bundle.meta.kernel}: ${state.insts.size} instructions, ${state.edgeCount} edges` +
    (saved ? " (the saved edit)" : ""));
}

function snapshot() {
  return state.orders.map((order) => [...order]);
}

function apply(orders, message) {
  state.undo.push(snapshot());
  state.redo = [];
  state.orders = orders;
  render();
  status(message, "ok");
}

function undo(from, to, what) {
  if (!from.length) return;
  to.push(snapshot());
  state.orders = from.pop();
  render();
  status(what);
}

// ---- selection and moves ---------------------------------------------------------------------

function selectedInOrder() {
  if (!state.selected.size) return [];
  const block = blockOf([...state.selected][0]);
  return state.orders[block].filter((id) => state.selected.has(id));
}

function select(id, event) {
  const block = blockOf(id);
  const sameBlock = state.selected.size && blockOf([...state.selected][0]) === block;
  if (event.shiftKey && sameBlock && state.anchor !== null) {
    const order = state.orders[block];
    const [a, b] = [order.indexOf(state.anchor), order.indexOf(id)].sort((x, y) => x - y);
    state.selected = new Set(order.slice(a, b + 1));
  } else if ((event.ctrlKey || event.metaKey) && sameBlock) {
    if (state.selected.has(id)) state.selected.delete(id);
    else state.selected.add(id);
    state.anchor = id;
  } else {
    state.selected = new Set([id]);
    state.anchor = id;
  }
  highlight();
}

function rangeOfSelection() {
  const selected = selectedInOrder();
  const block = blockOf(selected[0]);
  return { block, selected, range: legalRange(state.orders[block], selected, state.index, state.pinned) };
}

/** Move the selection to `gap`, or explain why not. */
function moveTo(block, selected, range, gap) {
  const blocking = blockingConstraint(range, gap);
  if (blocking) {
    status(`Refused: this move breaks ${describeConstraint(blocking, opcodeOf)}`, "error");
    return false;
  }
  if (gap === currentGap(state.orders[block], selected)) return false;
  const orders = snapshot();
  orders[block] = moveGroup(orders[block], selected, gap);
  apply(orders, `moved ${selected.length === 1 ? `id ${selected[0]}` : `${selected.length} instructions`}`);
  return true;
}

function nudge(delta) {
  if (!state.selected.size) return;
  const { block, selected, range } = rangeOfSelection();
  if (!range.ok) return status(`Refused: ${range.reason}`, "error");
  const order = state.orders[block];
  let gap = currentGap(order, selected);
  // A scattered selection gathers where its first member is, then steps.
  if (gap < 0) gap = order.slice(0, order.indexOf(selected[0])).filter((id) => !state.selected.has(id)).length;
  moveTo(block, selected, range, Math.max(0, Math.min(range.rest.length, gap + delta)));
}

// ---- rendering -------------------------------------------------------------------------------

function chip(kind, text, title = "") {
  const span = document.createElement("span");
  span.className = "chip";
  span.style.background = `var(--${kind})`;
  span.textContent = text;
  span.title = title;
  return span;
}

function shortText(text) {
  const cut = text.indexOf(" {");
  return (cut < 0 ? text : text.slice(0, cut)).replace(/"st\.([^"]+)"/, "$1");
}

function render() {
  const moved = new Set();
  if (state.baseOrders) {
    state.orders.forEach((order, b) => {
      for (const id of movedIds(state.baseOrders[b] ?? [], order)) moved.add(id);
    });
  }
  state.moved = moved;
  $("undo").disabled = !state.undo.length;
  $("redo").disabled = !state.redo.length;
  $("summary").textContent = state.baseOrders
    ? `${moved.size} moved from the dumped order`
    : "raw edit: no comparison with the dumped order";
  renderBlocks();
  renderDiff();
  if (state.tab === "raw" && state.rawText === null) $("raw").value = currentText();
}

function renderBlocks() {
  const pane = $("tab-blocks");
  pane.replaceChildren();
  state.doc.blocks.forEach((block, b) => {
    const section = document.createElement("div");
    section.className = "block";
    const title = document.createElement("h2");
    title.textContent = `^${block.label}  (${state.orders[b].length})`;
    section.append(title);
    const table = document.createElement("table");
    table.className = "rows";
    for (const id of state.orders[b]) table.append(row(id));
    section.append(table);
    pane.append(section);
  });
  highlight();
}

function row(id) {
  const inst = state.insts.get(id);
  const tr = document.createElement("tr");
  tr.dataset.id = id;
  tr.draggable = !inst.pinned;
  if (inst.pinned) tr.classList.add("pinned");
  if (state.moved.has(id)) tr.classList.add("moved");
  const cells = [
    ["num", String(id)],
    ["inst", shortText(inst.text)],
    ["cyc", `${inst.issue}/${inst.latency}`],
    ["deps", ""],
  ];
  for (const [cls, text] of cells) {
    const td = document.createElement("td");
    td.className = cls;
    td.textContent = text;
    tr.append(td);
  }
  tr.title = inst.text + (inst.pinned ? `\npinned: ${inst.pin_reason}` : "") +
    (state.moved.has(id) ? "\nmoved from the dumped order" : "");
  return tr;
}

/** Selection, predecessors and successors, in place (re-rendering would cancel a drag). */
function highlight() {
  const relations = new Map();
  for (const id of state.selected) {
    for (const edge of state.index.preds.get(id) ?? []) {
      if (!state.selected.has(edge.src)) push(relations, edge.src, ["pred", edge]);
    }
    for (const edge of state.index.succs.get(id) ?? []) {
      if (!state.selected.has(edge.dst)) push(relations, edge.dst, ["succ", edge]);
    }
  }
  for (const tr of document.querySelectorAll(".rows tr")) {
    const id = Number(tr.dataset.id);
    const related = relations.get(id) ?? [];
    tr.classList.toggle("selected", state.selected.has(id));
    tr.classList.toggle("rel-pred", related.some(([dir]) => dir === "pred"));
    tr.classList.toggle("rel-succ", related.some(([dir]) => dir === "succ"));
    tr.style.borderLeftColor = related.length ? `var(--${related[0][1].kind})` : "";
    const deps = tr.querySelector("td.deps");
    deps.replaceChildren(...related.map(([dir, edge]) => chip(edge.kind,
      `${dir === "pred" ? "\u2191" : "\u2193"} ${edge.kind} ${edge.regs.slice(0, 3).join(",")}` +
        (edge.regs.length > 3 ? "\u2026" : ""),
      `${dir === "pred" ? "must stay before" : "must stay after"} the selection: ` +
        describeConstraint(edge, opcodeOf))));
  }
}

function push(map, key, value) {
  if (!map.has(key)) map.set(key, []);
  map.get(key).push(value);
}

function renderDiff() {
  const pane = $("tab-diff");
  pane.replaceChildren();
  if (!state.baseOrders) {
    pane.textContent = "The raw text was edited, so there is no comparison with the dumped order.";
    return;
  }
  const table = document.createElement("table");
  table.className = "diff";
  let rows = 0;
  state.orders.forEach((order, b) => {
    const base = state.baseOrders[b] ?? [];
    for (const id of order) {
      if (!state.moved.has(id)) continue;
      const tr = document.createElement("tr");
      for (const text of [`^${state.doc.blocks[b].label}`, `id ${id}`, opcodeOf(id),
        `line ${base.indexOf(id)} \u2192 ${order.indexOf(id)}`]) {
        const td = document.createElement("td");
        td.textContent = text;
        tr.append(td);
      }
      table.append(tr);
      ++rows;
    }
  });
  pane.append(rows ? table : document.createTextNode("No instruction moved."));
}

// ---- drag and drop ---------------------------------------------------------------------------

function rowOf(event) {
  const tr = event.target.closest?.(".rows tr");
  return tr ? { tr, id: Number(tr.dataset.id) } : null;
}

function clearDropMarks() {
  for (const tr of document.querySelectorAll(".drop-above, .drop-below, .drop-refused")) {
    tr.classList.remove("drop-above", "drop-below", "drop-refused");
  }
}

function endDrag() {
  clearDropMarks();
  for (const tr of document.querySelectorAll(".out-of-range, .limit")) {
    tr.classList.remove("out-of-range", "limit");
  }
  state.drag = null;
}

function onDragStart(event) {
  const hit = rowOf(event);
  if (!hit) return;
  if (state.pinned.has(hit.id)) {
    event.preventDefault();
    return status(`id ${hit.id} is pinned (${state.pinned.get(hit.id)}); nothing moves it`, "error");
  }
  if (!state.selected.has(hit.id)) select(hit.id, {});
  const drag = rangeOfSelection();
  if (!drag.range.ok) {
    event.preventDefault();
    return status(`Refused: ${drag.range.reason}`, "error");
  }
  state.drag = { ...drag, gap: null, refusal: null };
  event.dataTransfer.effectAllowed = "move";
  event.dataTransfer.setData("text/plain", drag.selected.join(","));

  const { lo, hi, rest } = drag.range;
  rest.forEach((id, i) => {
    const tr = document.querySelector(`.rows tr[data-id="${id}"]`);
    // Row i sits between gaps i and i + 1; it is reachable when either gap is legal.
    if (i + 1 < lo || i > hi) tr.classList.add("out-of-range");
    if (i === lo - 1 || i === hi) tr.classList.add("limit");
  });
  status(`dragging ${drag.selected.length === 1 ? `id ${drag.selected[0]}` : `${drag.selected.length} instructions`}: ` +
    `after ${describeConstraint(drag.range.loBy, opcodeOf)}, before ${describeConstraint(drag.range.hiBy, opcodeOf)}`);
}

function onDragOver(event) {
  const drag = state.drag;
  const hit = rowOf(event);
  if (!drag || !hit) return;
  clearDropMarks();
  if (blockOf(hit.id) !== drag.block) {
    drag.refusal = "moves across blocks are not supported";
    hit.tr.classList.add("drop-refused");
    return status(`Refused: ${drag.refusal}`, "error");
  }
  if (state.selected.has(hit.id)) return;
  const box = hit.tr.getBoundingClientRect();
  const above = event.clientY < box.top + box.height / 2;
  const gap = drag.range.rest.indexOf(hit.id) + (above ? 0 : 1);
  const blocking = blockingConstraint(drag.range, gap);
  if (blocking) {
    drag.refusal = `this move breaks ${describeConstraint(blocking, opcodeOf)}`;
    hit.tr.classList.add("drop-refused");
    return status(`Refused: ${drag.refusal}`, "error");
  }
  event.preventDefault();
  event.dataTransfer.dropEffect = "move";
  drag.gap = gap;
  drag.refusal = null;
  hit.tr.classList.add(above ? "drop-above" : "drop-below");
  status(`drop to move ${above ? "before" : "after"} id ${hit.id}`);
}

function onDrop(event) {
  event.preventDefault();
  const drag = state.drag;
  endDrag();
  if (drag && drag.gap !== null) moveTo(drag.block, drag.selected, drag.range, drag.gap);
}

function onDragEnd() {
  const refusal = state.drag?.refusal;
  endDrag();
  if (refusal) status(`Refused: ${refusal}`, "error");
}

// ---- validate and save -----------------------------------------------------------------------

function hideCheck() {
  $("check").hidden = true;
  $("check").replaceChildren();
}

function showCheck(check, heading) {
  const box = $("check");
  box.replaceChildren();
  const title = document.createElement("strong");
  title.textContent = heading;
  const list = document.createElement("ul");
  for (const edge of check.reversed ?? []) {
    const li = document.createElement("li");
    li.append(chip(edge.kind, edge.kind), ` reversed: ${describeConstraint(edge, opcodeOf)}`);
    list.append(li);
  }
  for (const problem of check.problems ?? []) {
    const li = document.createElement("li");
    li.textContent = problem;
    list.append(li);
  }
  box.append(title, list);
  box.hidden = false;
}

async function validate() {
  const check = await api("/api/validate", { id: state.bundle.id, stir: currentText() });
  if (check.ok) {
    hideCheck();
    status("valid: every dependency of the dumped order holds", "ok");
  } else {
    showCheck(check, `${check.reversed.length} reversed edges, ${check.problems.length} problems`);
    status("not valid; see the list above", "error");
  }
  return check;
}

async function save() {
  const check = await validate();
  if (!check.ok) return status("not saved: the edit breaks a dependency", "error");
  const result = await api("/api/save", { id: state.bundle.id, stir: currentText() });
  state.bundle.saved_stir = currentText();
  $("open-saved").disabled = false;
  status(`saved ${result.saved}; build again with --load-st-ir on the load dir`, "ok");
}

// ---- wiring ----------------------------------------------------------------------------------

function showTab(tab) {
  state.tab = tab;
  for (const button of document.querySelectorAll(".tab")) {
    button.classList.toggle("active", button.dataset.tab === tab);
  }
  for (const name of ["blocks", "diff", "raw"]) $(`tab-${name}`).hidden = name !== tab;
  if (tab === "raw" && state.rawText === null && state.doc) $("raw").value = currentText();
}

function guarded(fn) {
  return async (...args) => {
    try {
      await fn(...args);
    } catch (error) {
      status(error.message, "error");
      if (error.data?.check) showCheck(error.data.check, error.message);
    }
  };
}

async function main() {
  const legend = $("legend");
  for (const kind of EDGE_KINDS) legend.append(chip(kind, kind));

  state.bundles = await api("/api/bundles");
  const picker = $("bundle");
  for (const bundle of state.bundles) {
    const option = document.createElement("option");
    option.value = bundle.id;
    option.textContent = `${bundle.saved ? "\u2713 " : ""}${bundle.kernel} \u2014 ${bundle.spec}`;
    option.title = bundle.id;
    picker.append(option);
  }
  if (!state.bundles.length) return status("no bundles in the dump directory", "error");

  picker.addEventListener("change", guarded(() => openBundle(picker.value)));
  $("open-saved").addEventListener("click", guarded(() => openBundle(picker.value, { saved: true })));
  $("undo").addEventListener("click", () => undo(state.undo, state.redo, "undone"));
  $("redo").addEventListener("click", () => undo(state.redo, state.undo, "redone"));
  $("validate").addEventListener("click", guarded(validate));
  $("save").addEventListener("click", guarded(save));
  for (const button of document.querySelectorAll(".tab")) {
    button.addEventListener("click", () => showTab(button.dataset.tab));
  }
  $("raw").addEventListener("input", () => { state.rawText = $("raw").value; });
  $("raw-apply").addEventListener("click", guarded(async () => {
    await adopt($("raw").value, { keepBase: false });
    status("re-analyzed the raw text", "ok");
  }));
  $("raw-reset").addEventListener("click", () => {
    state.rawText = null;
    $("raw").value = currentText();
  });

  const blocks = $("tab-blocks");
  blocks.addEventListener("click", (event) => {
    const hit = rowOf(event);
    if (hit) select(hit.id, event);
  });
  blocks.addEventListener("dragstart", onDragStart);
  blocks.addEventListener("dragover", onDragOver);
  blocks.addEventListener("drop", onDrop);
  blocks.addEventListener("dragend", onDragEnd);

  document.addEventListener("keydown", (event) => {
    if (event.target.tagName === "TEXTAREA") return;
    const key = event.key.toLowerCase();
    if ((event.ctrlKey || event.metaKey) && key === "z") {
      event.preventDefault();
      if (event.shiftKey) undo(state.redo, state.undo, "redone");
      else undo(state.undo, state.redo, "undone");
    } else if ((event.ctrlKey || event.metaKey) && key === "y") {
      event.preventDefault();
      undo(state.redo, state.undo, "redone");
    } else if (event.altKey && (event.key === "ArrowUp" || event.key === "ArrowDown")) {
      event.preventDefault();
      nudge(event.key === "ArrowUp" ? -1 : 1);
    } else if (event.key === "Escape") {
      state.selected.clear();
      highlight();
    }
  });

  await openBundle(picker.value);
}

guarded(main)();
