// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// node --test tests/legal_range.test.mjs

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  blockingConstraint,
  buildIndex,
  currentGap,
  describeConstraint,
  legalRange,
  moveGroup,
  movedIds,
} from "../stinkytofu_ir_editor/static/legal_range.js";

// 0 -raw v1-> 2 -war v0-> 4, and 6 is pinned (a store).
const ORDER = [0, 1, 2, 3, 4, 5, 6, 7];
const EDGES = [
  { src: 0, dst: 2, kind: "raw", class: "vgpr", regs: ["v1"] },
  { src: 2, dst: 4, kind: "war", class: "vgpr", regs: ["v0"] },
];
const INDEX = buildIndex(EDGES);
const NONE = new Map();
const PINNED = new Map([[6, "side effect"]]);

test("an unconstrained instruction may go anywhere", () => {
  const range = legalRange(ORDER, [3], INDEX, NONE);
  assert.equal(range.ok, true);
  assert.deepEqual([range.lo, range.hi], [0, 7]);
  assert.deepEqual(range.rest, [0, 1, 2, 4, 5, 6, 7]);
});

test("the range runs from the latest predecessor to the earliest successor", () => {
  const range = legalRange(ORDER, [2], INDEX, NONE);
  // rest = [0, 1, 3, 4, ...]: after 0 (gap 1) and before 4 (gap 3).
  assert.deepEqual([range.lo, range.hi], [1, 3]);
  assert.equal(range.loBy, EDGES[0]);
  assert.equal(range.hiBy, EDGES[1]);
  assert.equal(blockingConstraint(range, 0), EDGES[0]);
  assert.equal(blockingConstraint(range, 2), null);
  assert.equal(blockingConstraint(range, 4), EDGES[1]);
});

test("a pinned line bounds the side the selection is on", () => {
  // rest = [0, 1, 2, 3, 4, 6, 7]: the pinned 6 is rest[5].
  const before = legalRange(ORDER, [5], INDEX, PINNED);
  assert.deepEqual([before.lo, before.hi], [0, 5]);
  assert.deepEqual(before.hiBy, { kind: "pinned", id: 6, reason: "side effect" });
  const after = legalRange(ORDER, [7], INDEX, PINNED);
  assert.deepEqual([after.lo, after.hi], [7, 7]);
});

test("a pinned line itself does not move", () => {
  const range = legalRange(ORDER, [6], INDEX, PINNED);
  assert.equal(range.ok, false);
  assert.match(range.reason, /id 6 is pinned/);
  assert.equal(blockingConstraint(range, 0).kind, "infeasible");
});

test("a group takes the tightest bound of its members", () => {
  const range = legalRange(ORDER, [2, 3], INDEX, NONE);
  assert.deepEqual(range.rest, [0, 1, 4, 5, 6, 7]);
  assert.deepEqual([range.lo, range.hi], [1, 2]);
});

test("edges inside the group do not constrain it", () => {
  const range = legalRange(ORDER, [0, 2], INDEX, NONE);
  // Only 2 -> 4 binds: the run must stay before 4 (rest index 2).
  assert.deepEqual([range.lo, range.hi], [0, 2]);
});

test("a group that straddles a pinned line cannot be gathered", () => {
  const range = legalRange(ORDER, [5, 7], INDEX, PINNED);
  assert.equal(range.ok, false);
  assert.match(range.reason, /spans id 6/);
});

test("a group that cannot form one run is refused", () => {
  // 0 and 4 both selected: 2 must follow 0 and precede 4, so it would have to sit inside.
  const range = legalRange(ORDER, [0, 4], INDEX, NONE);
  assert.equal(range.ok, false);
  assert.match(range.reason, /cannot be gathered/);
});

test("moveGroup keeps the run's order and inserts it at the gap", () => {
  assert.deepEqual(moveGroup(ORDER, [3, 1], 0), [1, 3, 0, 2, 4, 5, 6, 7]);
  assert.deepEqual(moveGroup(ORDER, [2], 4), [0, 1, 3, 4, 2, 5, 6, 7]);
  assert.deepEqual(moveGroup(ORDER, [0], 7), [1, 2, 3, 4, 5, 6, 7, 0]);
});

test("currentGap finds a contiguous selection", () => {
  assert.equal(currentGap(ORDER, [2, 3]), 2);
  assert.equal(currentGap(ORDER, [2, 4]), -1);
});

test("every gap the range allows keeps every edge forward", () => {
  for (const id of ORDER) {
    const range = legalRange(ORDER, [id], INDEX, PINNED);
    if (PINNED.has(id)) {
      assert.equal(range.ok, false);
      continue;
    }
    for (let gap = 0; gap <= range.rest.length; ++gap) {
      const legal = range.ok && blockingConstraint(range, gap) === null;
      const moved = moveGroup(ORDER, [id], gap);
      const at = new Map(moved.map((x, i) => [x, i]));
      const forward = EDGES.every((e) => at.get(e.src) < at.get(e.dst));
      const pinsKept = [...PINNED.keys()].every((p) =>
        ORDER.every((x) => x === p || (ORDER.indexOf(x) < ORDER.indexOf(p)) === (at.get(x) < at.get(p))));
      assert.equal(legal, forward && pinsKept, `id ${id} at gap ${gap}`);
    }
  }
});

test("describeConstraint names the edge and its registers", () => {
  const opcode = (id) => ({ 0: "v_mov_b32", 2: "v_add_f32" })[id] ?? "";
  assert.equal(describeConstraint(EDGES[0], opcode),
    "raw vgpr v1: id 0 (v_mov_b32) before id 2 (v_add_f32)");
  assert.equal(describeConstraint({ kind: "pinned", id: 6, reason: "side effect" }),
    "pinned id 6 (side effect)");
  assert.equal(describeConstraint(null), "the block boundary");
});

test("movedIds marks what left the common order", () => {
  assert.deepEqual([...movedIds(ORDER, ORDER)], []);
  assert.deepEqual([...movedIds(ORDER, moveGroup(ORDER, [5], 1))], [5]);
  assert.deepEqual([...movedIds(ORDER, moveGroup(ORDER, [1, 2], 5))].sort(), [1, 2]);
  // A swap moves one of the two; an id only in one order is not "moved".
  const swapped = movedIds([0, 1, 2], [0, 9, 2, 1]);
  assert.equal(swapped.size, 1);
  assert.ok(swapped.has(1) || swapped.has(2));
});
