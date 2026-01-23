import * as vscode from "vscode";

type KernelBlock = {
  startLine: number;
  endLine: number;
  indent: number;
  title?: string;
  text: string;
};

type EditSession = {
  sourceUri: string;
  startLine: number;
  endLine: number;
  title?: string;
  lastAppliedText?: string;
};

function getConfig() {
  const cfg = vscode.workspace.getConfiguration("hipblasltKernelHover");
  return {
    enabled: cfg.get<boolean>("enabled", true),
    applyOnSave: cfg.get<boolean>("applyOnSave", true),
    closeEditorOnSave: cfg.get<boolean>("closeEditorOnSave", true),
    maxHoverChars: cfg.get<number>("maxHoverChars", 60000),
    triggerKeys: cfg.get<string[]>("triggerKeys", ["KernelNameMin", "SolutionNameMin", "BaseName"]),
  };
}

function escapeMarkdownCodeFence(text: string): string {
  // Avoid accidental triple-backtick termination in the content.
  return text.replace(/```/g, "``\\`");
}

function makeCommandLink(command: string, args: unknown, title: string): vscode.MarkdownString {
  const encodedArgs = encodeURIComponent(JSON.stringify(args ?? null));
  const md = new vscode.MarkdownString(`[${title}](command:${command}?${encodedArgs})`);
  md.isTrusted = true;
  return md;
}

function lineIndentAndText(lineText: string): { indent: number; text: string } {
  const m = lineText.match(/^(\s*)(.*)$/);
  return { indent: m?.[1]?.length ?? 0, text: m?.[2] ?? lineText };
}

function kernelStartIndent(lineText: string): number | null {
  // Handles:
  //   "  - 1LDSBuffer: 1"  (inner list item)
  //   "- - 1LDSBuffer: 1"  (first inner item on same line as outer list dash)
  const m1 = lineText.match(/^(\s*)-\s+1LDSBuffer\s*:/);
  if (m1) return m1[1].length;
  const m2 = lineText.match(/^(\s*)-\s+-\s+1LDSBuffer\s*:/);
  if (m2) return m2[1].length + 2; // "- " before the second dash
  return null;
}

function isKernelStartAtIndent(lineText: string, indent: number): boolean {
  // Match either "  - 1LDSBuffer:" or "- - 1LDSBuffer:" but only if the effective dash indent equals `indent`.
  const eff = kernelStartIndent(lineText);
  return eff !== null && eff === indent;
}

function shouldTriggerHoverOnLine(
  lineText: string,
  position: vscode.Position,
  triggerKeys: string[]
): boolean {
  for (const key of triggerKeys) {
    const idx = lineText.indexOf(`${key}:`);
    if (idx === -1) continue;
    const colonIdx = idx + key.length;
    // Only trigger when hovering on the VALUE side of "Key:".
    if (position.character > colonIdx + 1) return true;
  }
  // Also allow hovering on the kernel start line itself.
  if (kernelStartIndent(lineText) !== null) return true;
  return false;
}

function extractKernelBlock(document: vscode.TextDocument, fromLine: number): KernelBlock | null {
  let startLine = -1;
  let indent = 0;

  for (let i = fromLine; i >= 0; i--) {
    const t = document.lineAt(i).text;
    const ind = kernelStartIndent(t);
    if (ind !== null) {
      startLine = i;
      indent = ind;
      break;
    }
  }
  if (startLine === -1) return null;

  let endLine = document.lineCount - 1;
  for (let j = startLine + 1; j < document.lineCount; j++) {
    const t = document.lineAt(j).text;
    if (isKernelStartAtIndent(t, indent)) {
      endLine = j - 1;
      break;
    }
    const { indent: lineInd, text } = lineIndentAndText(t);
    if (text.trim().length > 0 && lineInd < indent) {
      endLine = j - 1;
      break;
    }
  }

  const range = new vscode.Range(startLine, 0, endLine, document.lineAt(endLine).text.length);
  const text = document.getText(range);

  const title =
    text.match(/^\s*KernelNameMin:\s*(.+)\s*$/m)?.[1]?.trim() ??
    text.match(/^\s*SolutionNameMin:\s*(.+)\s*$/m)?.[1]?.trim() ??
    text.match(/^\s*BaseName:\s*(.+)\s*$/m)?.[1]?.trim();

  return { startLine, endLine, indent, title, text };
}

function extractTitleFromKernelText(text: string): string | undefined {
  return (
    text.match(/^\s*KernelNameMin:\s*(.+)\s*$/m)?.[1]?.trim() ??
    text.match(/^\s*SolutionNameMin:\s*(.+)\s*$/m)?.[1]?.trim() ??
    text.match(/^\s*BaseName:\s*(.+)\s*$/m)?.[1]?.trim()
  );
}

function safeFilenamePart(s: string): string {
  return s.replace(/[^\w.-]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 80) || "kernel";
}

export function activate(context: vscode.ExtensionContext) {
  // Edited-buffer URI -> where to apply back in the source YAML
  const editSessions = new Map<string, EditSession>();

  async function openEditBufferFile(title: string, initialText: string): Promise<vscode.TextDocument> {
    const name = safeFilenamePart(title);

    // 1) Prefer extension global storage (does not pollute repo).
    try {
      await vscode.workspace.fs.createDirectory(context.globalStorageUri);
      const fileUri = vscode.Uri.joinPath(context.globalStorageUri, `${name}_${Date.now()}.yaml`);
      await vscode.workspace.fs.writeFile(fileUri, Buffer.from(initialText, "utf8"));
      const doc = await vscode.workspace.openTextDocument(fileUri);
      if (!doc.isUntitled) return doc;
    } catch {
      // fall through
    }

    // 2) Fallback to workspace folder (some remote setups don't allow saving globalStorage cleanly).
    const ws = vscode.workspace.workspaceFolders?.[0]?.uri;
    if (!ws) {
      // Last resort: untitled doc; will require Save As.
      return await vscode.workspace.openTextDocument({ content: initialText, language: "yaml" });
    }

    const dir = vscode.Uri.joinPath(ws, ".hipblasltKernelHover");
    await vscode.workspace.fs.createDirectory(dir);
    const fileUri = vscode.Uri.joinPath(dir, `${name}_${Date.now()}.yaml`);
    await vscode.workspace.fs.writeFile(fileUri, Buffer.from(initialText, "utf8"));
    return await vscode.workspace.openTextDocument(fileUri);
  }

  async function applyTextBackToSource(session: EditSession, newText: string, fromEditorSave: boolean) {
    const sourceDoc = await vscode.workspace.openTextDocument(vscode.Uri.parse(session.sourceUri));
    const edit = new vscode.WorkspaceEdit();
    const endChar = sourceDoc.lineAt(session.endLine).text.length;
    const replaceRange = new vscode.Range(session.startLine, 0, session.endLine, endChar);
    edit.replace(sourceDoc.uri, replaceRange, newText);

    const ok = await vscode.workspace.applyEdit(edit);
    if (!ok) {
      void vscode.window.showErrorMessage("Failed to apply edits back to source YAML.");
      return;
    }

    // Persist to disk so the user doesn't need to save the source manually.
    await sourceDoc.save();

    // Update session range to match edited text line count.
    const lines = newText.split(/\r?\n/).length;
    session.endLine = session.startLine + Math.max(1, lines) - 1;
    session.title = extractTitleFromKernelText(newText) ?? session.title;
    session.lastAppliedText = newText;

    if (!fromEditorSave) {
      void vscode.window.showInformationMessage(
        `Applied kernel edits back to source (lines ${session.startLine + 1}-${session.endLine + 1}).`
      );
    } else {
      void vscode.window.setStatusBarMessage("Kernel edits applied back to source YAML.", 2500);
    }

    const newEndChar = sourceDoc.lineAt(Math.min(session.endLine, sourceDoc.lineCount - 1)).text.length;
    const newRange = new vscode.Range(
      session.startLine,
      0,
      Math.min(session.endLine, sourceDoc.lineCount - 1),
      newEndChar
    );
    return { sourceDoc, range: newRange };
  }

  const openKernelCmd = vscode.commands.registerCommand(
    "hipblasltKernelHover.openKernel",
    async (payload?: { title?: string; text: string; source?: string }) => {
      const title = payload?.title ?? "Tensile kernel parameters";
      const text = payload?.text ?? "";
      const doc = await vscode.workspace.openTextDocument({ content: text, language: "yaml" });
      await vscode.window.showTextDocument(doc, { preview: true });
      void vscode.window.setStatusBarMessage(`Opened: ${title}`, 2500);
    }
  );
  context.subscriptions.push(openKernelCmd);

  const editKernelCmd = vscode.commands.registerCommand(
    "hipblasltKernelHover.editKernel",
    async (payload?: { title?: string; text: string; sourceUri: string; startLine: number; endLine: number }) => {
      const title = payload?.title ?? "Tensile kernel";
      const text = payload?.text ?? "";
      const sourceUri = payload?.sourceUri;
      const startLine = payload?.startLine;
      const endLine = payload?.endLine;
      if (!sourceUri || startLine === undefined || endLine === undefined) return;

      // Open a real file-backed buffer so Ctrl+S doesn't trigger "Save As".
      const doc = await openEditBufferFile(title, text);
      editSessions.set(doc.uri.toString(), { sourceUri, startLine, endLine, title, lastAppliedText: text });
      await vscode.window.showTextDocument(doc, { preview: false });
      void vscode.window.setStatusBarMessage(`Editing kernel buffer (${title}). Ctrl+S will apply back.`, 4000);
    }
  );
  context.subscriptions.push(editKernelCmd);

  const applyKernelEditsCmd = vscode.commands.registerCommand("hipblasltKernelHover.applyKernelEdits", async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    const key = editor.document.uri.toString();
    const session = editSessions.get(key);
    if (!session) {
      void vscode.window.showWarningMessage(
        "Active document is not a kernel edit buffer created by hipBLASLt Kernel Hover."
      );
      return;
    }
    await applyTextBackToSource(session, editor.document.getText(), false);
  });
  context.subscriptions.push(applyKernelEditsCmd);

  const saveListener = vscode.workspace.onDidSaveTextDocument(async (doc) => {
    const cfg = getConfig();
    if (!cfg.applyOnSave) return;
    const session = editSessions.get(doc.uri.toString());
    if (!session) return;

    const newText = doc.getText();
    if (session.lastAppliedText === newText) return; // no-op
    const result = await applyTextBackToSource(session, newText, true);
    if (!result) return;

    // Close the temp editor and return focus to source YAML.
    if (cfg.closeEditorOnSave) {
      // Close the editor showing this temp doc (if any).
      const active = vscode.window.activeTextEditor;
      if (active?.document.uri.toString() === doc.uri.toString()) {
        await vscode.commands.executeCommand("workbench.action.closeActiveEditor");
      } else {
        const tempEditor = vscode.window.visibleTextEditors.find((e) => e.document.uri.toString() === doc.uri.toString());
        if (tempEditor) {
          await vscode.window.showTextDocument(tempEditor.document, { preview: true, preserveFocus: false });
          await vscode.commands.executeCommand("workbench.action.closeActiveEditor");
        }
      }
    }

    const srcEditor = await vscode.window.showTextDocument(result.sourceDoc, { preview: false });
    srcEditor.revealRange(result.range, vscode.TextEditorRevealType.InCenter);
    srcEditor.selection = new vscode.Selection(result.range.start, result.range.start);
  });
  context.subscriptions.push(saveListener);

  const selector: vscode.DocumentSelector = [{ language: "yaml" }];

  const hoverProvider = vscode.languages.registerHoverProvider(selector, {
    provideHover(document, position) {
      const cfg = getConfig();
      if (!cfg.enabled) return;

      const lineText = document.lineAt(position.line).text;
      if (!shouldTriggerHoverOnLine(lineText, position, cfg.triggerKeys)) return;

      const block = extractKernelBlock(document, position.line);
      if (!block) return;

      const header = block.title ? `**Kernel**: \`${block.title}\`\n\n` : "";
      const blockText = escapeMarkdownCodeFence(block.text);
      const md = new vscode.MarkdownString();

      const editLink = makeCommandLink(
        "hipblasltKernelHover.editKernel",
        {
          title: block.title ?? "Tensile kernel",
          text: block.text,
          sourceUri: document.uri.toString(),
          startLine: block.startLine,
          endLine: block.endLine,
        },
        "Edit this kernel (open editable buffer)"
      );
      // Needed for command: links to work in hovers.
      md.isTrusted = true;

      // If it is very large, show a link to open it in editor to avoid hover truncation.
      if (blockText.length > cfg.maxHoverChars) {
        md.appendMarkdown(header);
        md.appendMarkdown(
          `Kernel block is **${blockText.length.toLocaleString()}** characters; hover would be truncated.\n\n`
        );
        md.appendMarkdown(
          makeCommandLink(
            "hipblasltKernelHover.openKernel",
            { title: block.title ?? "Tensile kernel", text: block.text, source: document.uri.toString() },
            "Open full kernel in editor"
          ).value
        );
        md.appendMarkdown("\n\n");
        md.appendMarkdown(editLink.value);
        return new vscode.Hover(md);
      }

      md.appendMarkdown(header);
      md.appendMarkdown(editLink.value);
      md.appendMarkdown("\n\n");
      md.appendMarkdown("```yaml\n");
      md.appendMarkdown(blockText);
      md.appendMarkdown("\n```\n");
      return new vscode.Hover(md);
    },
  });

  context.subscriptions.push(hoverProvider);
}

export function deactivate() {}

