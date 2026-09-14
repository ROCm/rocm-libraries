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
- A hipDNN SDK (headers and libraries).
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

The GPU support lives in a small native add-on that is built separately. Point
`HIPDNN_SDK` at your SDK folder and build it once:

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

Start the app with the same `HIPDNN_SDK` value set, so it can find the hipDNN
libraries at runtime. The engine panel shows which device it found, or explains
why it could not load.

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
