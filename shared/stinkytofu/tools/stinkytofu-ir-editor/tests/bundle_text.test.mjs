// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// node --test tests/bundle_text.test.mjs

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  blockIdOrders,
  classify,
  idOfLine,
  parseBundle,
  renderBundle,
} from "../stinkytofu_ir_editor/static/bundle_text.js";

const REGION = readFileSync(new URL(
  "../../../python_module/tests/data/ir_checkpoint_f8/region.stir", import.meta.url), "utf8");

const SMALL = `st.func @f() {
  ^entry:
    v1 = "st.v_mov_b32"(v0) { mod.ckpt = { id = 0 } }
    v2 = "st.v_mov_b32"(v0) { mod.ckpt = { id = 1 } }
    Successors: ^next
  ^next:
    "st.s_endpgm"() { mod.ckpt = { id = 2 } }
}
`;

test("line classes match the checkpoint loader", () => {
  assert.equal(classify("  ^entry:"), "block");
  assert.equal(classify("    Successors: ^a, ^b"), "other");
  assert.equal(classify("st.func @temp() {"), "other");
  assert.equal(classify("}"), "end");
  assert.equal(classify('    v1 = "st.v_mov_b32"(v0)'), "node");
  assert.equal(idOfLine('v1 = "st.v_mov_b32"(v0) { mod.ckpt = { id = 12 } }'), 12);
  assert.equal(idOfLine('v1 = "st.v_mov_b32"(v0)'), null);
});

test("parse then render is the identity", () => {
  for (const text of [SMALL, REGION]) {
    const doc = parseBundle(text);
    assert.equal(renderBundle(doc, doc.blocks.map((b) => b.nodes)), text);
  }
});

test("a reorder changes only the order of IR lines", () => {
  const doc = parseBundle(SMALL);
  const lines = doc.blocks.map((b) => [...b.nodes].reverse());
  const text = renderBundle(doc, lines);
  assert.deepEqual(blockIdOrders(text), [[1, 0], [2]]);
  assert.match(text, /\{ id = 0 \} \}\n    Successors: \^next\n/);
});

test("the f8 region has every dumped id once, in order", () => {
  const ids = blockIdOrders(REGION).flat();
  assert.deepEqual(ids, [...ids.keys()]);
  assert.equal(parseBundle(REGION).blocks.length, 19);
});
