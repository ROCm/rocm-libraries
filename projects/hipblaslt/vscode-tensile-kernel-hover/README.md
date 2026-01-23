# hipBLASLt Tensile Kernel Hover

This is a lightweight VS Code / Cursor extension for **huge Tensile YAML logic files**.

When you hover on a kernel identifier line (by default: `KernelNameMin`, `SolutionNameMin`, `BaseName`), it shows a hover containing **only that single kernel block** (all parameters for the kernel).

If the kernel block is too large to fit in a hover, it offers a link to **open the full kernel block in an editor tab**.

## How to run (dev)

1. Open this folder in VS Code/Cursor:
   - `projects/hipblaslt/vscode-tensile-kernel-hover`
2. Install dependencies:
   - `npm install`
3. Build:
   - `npm run build`
4. Run extension (debug):
   - Press `F5` (Extension Development Host)
5. Open your Tensile YAML file and hover on `KernelNameMin:` values.

## Settings

- `hipblasltKernelHover.enabled`
- `hipblasltKernelHover.applyOnSave`
- `hipblasltKernelHover.closeEditorOnSave`
- `hipblasltKernelHover.maxHoverChars`
- `hipblasltKernelHover.triggerKeys`

