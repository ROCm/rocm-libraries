# stinkytofu-ir-editor

A browser editor for [IR checkpoint](../../docs/user/ir-checkpoint.md) bundles. It shows a
dumped region one instruction per row with its dependencies, lets you reorder instructions only
within the range their dependencies allow, and saves a validated edit for `--load-st-ir`.

It needs only the Python standard library and the `stinkytofu` Python module, which supplies the
dependency analysis (`stinkytofu.ir.analyze` and `stinkytofu.ir.check_order`).

## Running it

```bash
cd shared/stinkytofu/tools/stinkytofu-ir-editor
python -m stinkytofu_ir_editor --bundles /tmp/ir/bundles --load-dir /tmp/ir/load \
    --stinkytofu-path <stinkytofu build>/lib
# then open http://127.0.0.1:8765/
```

| Option | Meaning |
| --- | --- |
| `--bundles DIR` | The dump directory (`--dump-st-ir`). It is only read. |
| `--load-dir DIR` | Where Save writes; build with `--load-st-ir DIR`. It must differ from `--bundles`. |
| `--host`, `--port` | Default `127.0.0.1:8765`, this machine only. Through SSH: `ssh -L 8765:127.0.0.1:8765 host`. |
| `--stinkytofu-path DIR` | The directory holding the `stinkytofu` package if it is not importable: `<stinkytofu build>/lib`, or `<tensilelite build>/tensilelite/rocisa` for the module TensileLite builds. |

Set `STINKYTOFU_IR_EDITOR_LOG=1` to log each request.

## Using it

Pick a bundle from the list. **Open saved edit** loads the edit already in the load directory.

**Blocks tab.** Each block lists its rows: the id, the instruction, and issue/latency cycles. A
moved row's id is shown in bold blue; a pinned row is grey, with a lock after its id.

- **Select.** Click a row. Shift-click selects a contiguous range and Ctrl-click (Cmd-click on
  a Mac) adds a row; a selection stays inside one block.
- **Dependencies.** The rows the selection depends on (↑, must stay before it) and the rows
  that depend on it (↓, must stay after it) are highlighted, coloured by edge kind. Their chips
  name the kind and registers, and hovering over a chip says why the edge exists.
- **Drag.** Dragging a selection shades the rows outside its legal range. Dropping outside the
  range is refused, and the status line names the edge that blocks it. A multi-row selection
  moves as one group, in its current order.
- **Alt+Up / Alt+Down** moves the selection one row, past the neighbouring row.
- **Ctrl+Z**, **Ctrl+Shift+Z** or **Ctrl+Y**: undo and redo. **Escape** clears the selection.

**Diff tab.** A line diff of the edit against the dumped order.

**Raw text tab.** The bundle text, for edits the Blocks tab cannot make: changing operands,
adding or deleting lines. **Re-analyze** parses the text and shows parse errors with their line.
A new line needs no id; it gets one above the largest id.

**Validate** runs `check_order` on the whole edit and lists every reversed edge and problem.
**Save** validates first and writes only a valid edit, to `<load dir>/<bundle>/` with a copy of
`meta.json`.

## What is pinned

A pinned row neither moves nor lets anything move across it. These are the scheduler's region
boundaries:

- directives;
- branches and `s_endpgm`;
- instructions with side effects: global stores, waits, barriers, and LDS operations without an
  LDS token;
- instructions that write EXEC, because the VALU instructions that read EXEC do not list it as an
  operand.

The [dependency model](../../docs/user/ir-checkpoint.md#the-dependency-model) lists the edge
kinds.

## HTTP API

All bodies are JSON. Errors return `{"error": "..."}`, with the check result in a 409.

| Request | Body | Reply |
| --- | --- | --- |
| `GET /api/bundles` | | `[{id, kernel, spec, instructions, saved}]` |
| `GET /api/bundle/<id>` | | `{id, meta, stir, saved_stir}` |
| `POST /api/analyze` | `{stir, id?}` | `stinkytofu.ir.analyze` of `stir`, with the bundle's arch when `id` is given |
| `POST /api/validate` | `{id, stir}` | `check_order` of `stir` against the bundle: `{ok, reversed, problems}` |
| `POST /api/save` | `{id, stir}` | `{saved, check}`, or 409 with `check` when it is not valid |

Parse errors return 422. Each analysis runs in a forked child, so an abort inside StinkyTofu
returns an error instead of stopping the server. The server handles one request at a time.

## Layout and tests

| Path | What it is |
| --- | --- |
| `stinkytofu_ir_editor/server.py` | `Workspace` (bundles, analyze, validate, save) and the HTTP handler. |
| `stinkytofu_ir_editor/static/legal_range.js` | Legal range of a selection, the blocking edge, group moves, moved ids. |
| `stinkytofu_ir_editor/static/bundle_text.js` | Splits a bundle into blocks and rows and renders it back. |
| `stinkytofu_ir_editor/static/app.js` | The UI. |

```bash
PYTHONPATH=<stinkytofu build>/lib python -m pytest tests
```

`tests/test_server.py` exercises the workspace and the HTTP API on the f8 region in
`python_module/tests/data/`. `tests/test_legal_range.py` runs the `*.test.mjs` files under Node
and is skipped without `node`.
