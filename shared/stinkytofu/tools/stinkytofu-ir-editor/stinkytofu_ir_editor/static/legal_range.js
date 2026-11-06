// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Where a group of instructions may move inside its block. Pure functions over ids, so the
// tests run them under node; the server's check_order is the authority on save.
//
// A move takes the selected ids out of the block, keeping their order, and inserts them as one
// run at a gap of what is left (`rest`): gap g goes before rest[g], gap rest.length at the end.

/** Edges by endpoint: preds.get(id) are the edges into id, succs.get(id) the edges out. */
export function buildIndex(edges) {
  const preds = new Map();
  const succs = new Map();
  for (const edge of edges) {
    if (!preds.has(edge.dst)) preds.set(edge.dst, []);
    if (!succs.has(edge.src)) succs.set(edge.src, []);
    preds.get(edge.dst).push(edge);
    succs.get(edge.src).push(edge);
  }
  return { preds, succs };
}

/**
 * The gaps of `rest` the selection may land in: every gap in [lo, hi]. `loBy` and `hiBy` are
 * the constraints that set each bound (an edge, or a pinned line), null when the bound is the
 * block's edge. `ok` is false, with a `reason`, when no gap works.
 *
 * @param {number[]} order ids of one block, in the current order
 * @param {number[]} selected ids to move, all in `order`
 * @param {{preds: Map, succs: Map}} index from buildIndex
 * @param {Map<number, string>} pinned pinned id -> why it is pinned
 */
export function legalRange(order, selected, index, pinned) {
  const chosen = new Set(selected);
  const rest = order.filter((id) => !chosen.has(id));
  const result = { ok: true, lo: 0, hi: rest.length, loBy: null, hiBy: null, rest, reason: "" };

  for (const id of selected) {
    if (pinned.has(id)) {
      return { ...result, ok: false, reason: `id ${id} is pinned (${pinned.get(id)})` };
    }
  }

  const restIndex = new Map(rest.map((id, i) => [id, i]));
  for (const id of selected) {
    for (const edge of index.preds.get(id) ?? []) {
      const i = restIndex.get(edge.src);
      if (i !== undefined && i + 1 > result.lo) {
        result.lo = i + 1;
        result.loBy = edge;
      }
    }
    for (const edge of index.succs.get(id) ?? []) {
      const i = restIndex.get(edge.dst);
      if (i !== undefined && i < result.hi) {
        result.hi = i;
        result.hiBy = edge;
      }
    }
  }

  // Nothing crosses a pinned line: it bounds the side the selection is on now.
  const position = new Map(order.map((id, i) => [id, i]));
  const first = Math.min(...selected.map((id) => position.get(id)));
  const last = Math.max(...selected.map((id) => position.get(id)));
  rest.forEach((id, i) => {
    if (!pinned.has(id)) return;
    const at = position.get(id);
    const pin = { kind: "pinned", id, reason: pinned.get(id) };
    if (at < first && i + 1 > result.lo) {
      result.lo = i + 1;
      result.loBy = pin;
    } else if (at > last && i < result.hi) {
      result.hi = i;
      result.hiBy = pin;
    } else if (at > first && at < last) {
      result.ok = false;
      result.reason = `the selection spans id ${id}, which is pinned (${pin.reason})`;
    }
  });

  if (result.ok && result.lo > result.hi) {
    result.ok = false;
    result.reason = "the selection cannot be gathered into one run: " +
      `${describeConstraint(result.loBy)} must stay before it and ` +
      `${describeConstraint(result.hiBy)} after it`;
  }
  return result;
}

/** The constraint a drop at `gap` would break, or null when the gap is legal. */
export function blockingConstraint(range, gap) {
  if (!range.ok) return { kind: "infeasible", reason: range.reason };
  if (gap < range.lo) return range.loBy;
  if (gap > range.hi) return range.hiBy;
  return null;
}

/** `order` with `selected` moved, as one run in their current order, to `gap` of the rest. */
export function moveGroup(order, selected, gap) {
  const chosen = new Set(selected);
  const run = order.filter((id) => chosen.has(id));
  const rest = order.filter((id) => !chosen.has(id));
  return [...rest.slice(0, gap), ...run, ...rest.slice(gap)];
}

/** The gap the selection sits at now, when it is one run; -1 otherwise. */
export function currentGap(order, selected) {
  const chosen = new Set(selected);
  const positions = order.map((id, i) => (chosen.has(id) ? i : -1)).filter((i) => i >= 0);
  if (!positions.length || positions[positions.length - 1] - positions[0] !== positions.length - 1) {
    return -1;
  }
  return positions[0];
}

/** One-line description of an edge or pin, for a refused drop. */
export function describeConstraint(constraint, opcodeOf = () => "") {
  if (!constraint) return "the block boundary";
  if (constraint.kind === "pinned") {
    return `pinned id ${constraint.id}${name(opcodeOf, constraint.id)} (${constraint.reason})`;
  }
  if (constraint.kind === "infeasible") return constraint.reason;
  const named = constraint.regs?.join(", ") ?? "";
  const regs = named && named !== constraint.class ? ` ${named}` : "";
  return `${constraint.kind} ${constraint.class}${regs}: id ${constraint.src}` +
    `${name(opcodeOf, constraint.src)} before id ${constraint.dst}${name(opcodeOf, constraint.dst)}`;
}

function name(opcodeOf, id) {
  const op = opcodeOf(id);
  return op ? ` (${op})` : "";
}

/**
 * Ids of `current` that moved relative to `base`: those outside a longest common subsequence
 * of the two orders. Ids only in one of them are ignored.
 */
export function movedIds(base, current) {
  const known = new Set(base);
  const b = base.filter((id) => current.includes(id));
  const c = current.filter((id) => known.has(id));
  const n = b.length;
  const m = c.length;
  const table = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; --i) {
    for (let j = m - 1; j >= 0; --j) {
      table[i][j] = b[i] === c[j] ? table[i + 1][j + 1] + 1 : Math.max(table[i + 1][j], table[i][j + 1]);
    }
  }
  const kept = new Set();
  for (let i = 0, j = 0; i < n && j < m;) {
    if (b[i] === c[j]) {
      kept.add(b[i]);
      ++i;
      ++j;
    } else if (table[i + 1][j] >= table[i][j + 1]) {
      ++i;
    } else {
      ++j;
    }
  }
  return new Set(c.filter((id) => !kept.has(id)));
}
