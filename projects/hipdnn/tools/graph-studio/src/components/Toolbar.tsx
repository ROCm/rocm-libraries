interface ToolbarProps {
  graphName: string;
  fileName: string | null;
  dirty: boolean;
  onRenameGraph(name: string): void;
  onNew(): void;
  onOpen(): void;
  onSave(): void;
  onSaveAs(): void;
}

export function Toolbar({
  graphName,
  fileName,
  dirty,
  onRenameGraph,
  onNew,
  onOpen,
  onSave,
  onSaveAs,
}: ToolbarProps) {
  return (
    <header className="toolbar">
      <div className="toolbar__brand">
        <span className="toolbar__logo">◈</span>
        <span className="toolbar__app">hipDNN Graph Studio</span>
      </div>

      <input
        className="toolbar__name"
        value={graphName}
        spellCheck={false}
        onChange={(event) => onRenameGraph(event.target.value)}
        aria-label="Graph name"
      />

      <div className="toolbar__file">
        {fileName ? `${fileName}${dirty ? " •" : ""}` : dirty ? "unsaved •" : "unsaved"}
      </div>

      <div className="toolbar__actions">
        <button type="button" onClick={onNew}>
          New
        </button>
        <button type="button" onClick={onOpen}>
          Open
        </button>
        <button type="button" onClick={onSave}>
          Save
        </button>
        <button type="button" onClick={onSaveAs}>
          Save As
        </button>
      </div>
    </header>
  );
}
