# hipDNN Graph Studio

A desktop app for drawing hipDNN computation graphs and running them on a GPU.

![The hipDNN Graph Studio editor: operator palette on the left, a batch-normalization graph on the canvas, and the engine panel along the bottom](images/hipdnn-graph-studio.png)

Drag operators onto a canvas, wire them together, fill in shapes and settings,
then press **Build** to compile the graph with hipDNN and **Execute** to run it.
Errors and engine logs come back in the panel at the bottom, so you can try a
graph out without writing any code.

## What you can do

- Build graphs from a palette of operators: convolution (forward, data and
  weight gradients), matmul, batch/layer/RMS normalization, pointwise math,
  reductions, pooling and resampling, block-scale quantize/dequantize, plus
  input and output tensors.
- Edit each node's shape, data type, and settings in the inspector.
- Let hipDNN infer output shapes, or clear **Use defaults** on an Output node to
  pin its dimensions yourself.
- Save and load graphs as JSON. Your work is also auto-saved between sessions.
- Pick which hipDNN engine to use, or let hipDNN choose the best one.
- Run the graph with randomly filled tensors and see how long it took.
- Import and export hipDNN's own JSON format, so graphs can be shared with
  other hipDNN tools.

## Requirements

To edit graphs:

- [Bun](https://bun.sh) 1.2 or newer.

To also build and run graphs on a GPU, you additionally need:

- An AMD GPU with ROCm installed.
- An in-tree hipDNN build, or an installed hipDNN SDK (headers and libraries).
- Node.js 20+, Python 3.9+, and a C++ compiler — on Windows, Visual Studio 2022
  with the "Desktop development with C++" workload; on Linux, GCC or Clang.

Without the GPU pieces the app still runs: you can draw, save, and export
graphs, and the Build and Execute buttons stay disabled.

## Quick start

Windows, from the project folder:

```
start.bat
```

That installs dependencies on first run and opens the app. Use `start.bat dev`
for a development window that reloads as you edit the source.

Any platform, the same thing by hand:

```
bun install
bun run electron:start     # build and open the app
bun run electron:dev       # development window with live reload
```

You can also run it as a plain web page in the browser, without the GPU parts:

```
bun run dev
```

Saving to a file works best in Chrome or Edge; other browsers fall back to a
normal download.

## Turning on the GPU engine

The GPU support lives in a small native add-on that links hipDNN. The easiest
way to get one is the repository superbuild, which builds hipDNN, every DNN
provider, and the Studio (including this add-on) in one go:

```
cmake --preset hipdnn-graph-studio -B build
cmake --build build
```

Everything it produces stays in the build tree, under `build/graph-studio/`:
`dist/` (web bundle), `native/` (the add-on plus its node-gyp intermediates),
and `build-config.json`, which records the include paths, libraries and plugin
directory of that build. Launch it from there with:

```
build\bin\start-graph-studio.bat        # Linux: build/bin/start-graph-studio.sh
```

Running from this directory works too — `start.bat` and `bun run build:native`
pick up `<repo>/build` automatically, so the app finds hipDNN and the provider
plugins with nothing else to set up. Point `HIPDNN_BUILD_DIR` at a different
build directory to use that one instead.

To build the add-on against an installed SDK rather than a build tree, set
`HIPDNN_SDK` instead:

Windows (PowerShell):

```
cd electron\native
bun install
cd ..\..
$env:HIPDNN_SDK = "C:\path\to\sdk"
bun run build:native
```

Linux:

```
cd electron/native && bun install && cd ../..
HIPDNN_SDK=/path/to/sdk bun run build:native
```

Without a build tree the outputs stay here instead (`dist/` and
`electron/native/build/`); set `GRAPH_STUDIO_OUT_DIR` to move them elsewhere.
Start the app with the same environment variables set, so it resolves the same
hipDNN and the same outputs at runtime. The engine panel shows which device it
found, or explains why it could not load.

If the add-on is missing or fails to load, the app keeps working with Build and
Execute switched off — rebuilding it is the only fix needed.

## Project layout

```
src/            The editor: canvas, palette, inspector, engine panel
src/graph/      Operator catalog and the saved graph format
src/engine/     Talks to the GPU engine, with a no-op version for the browser
src/platform/   File dialogs and settings storage, per platform
electron/       Desktop shell (window, file dialogs, engine bridge)
electron/native/  C++ add-on that calls hipDNN
electron/paths.cjs  Finds hipDNN and the build-output locations
electron/native/build-addon.cjs  Drives node-gyp against those locations
CMakeLists.txt  Superbuild hook (the hipdnn-graph-studio component)
```

## Commands

| Command | What it does |
| --- | --- |
| `bun run dev` | Web version with live reload |
| `bun run build` | Type-check and bundle the web assets |
| `bun run typecheck` | Type-check only |
| `bun run electron:dev` | Desktop app with live reload |
| `bun run electron:start` | Build, then open the desktop app |
| `bun run electron:pack` | Package the desktop app for distribution |
| `bun run build:native` | Build the hipDNN add-on |

## License

MIT. See [LICENSE](LICENSE).
