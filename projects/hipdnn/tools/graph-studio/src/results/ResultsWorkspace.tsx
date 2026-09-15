/**
 * Owns imported-report lifecycle (open/drop/remove), the current native
 * execution document, and which document is selected. Presentation lives in
 * `ResultsViewer`; this component never touches graph/canvas/engine state.
 */
import { useEffect, useRef, useState } from "react";
import { platform } from "../platform";
import { parseResults, ResultsParseError } from "./parse";
import { ResultsViewer } from "./ResultsViewer";
import type { ResultDocument } from "./model";

interface ImportError {
  readonly label: string;
  readonly message: string;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function ResultsWorkspace(props: {
  nativeDocument?: ResultDocument | null;
}): JSX.Element {
  const { nativeDocument = null } = props;

  const [imports, setImports] = useState<readonly ResultDocument[]>([]);
  const [errors, setErrors] = useState<readonly ImportError[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  // report-N ids are allocated once here and never reused, even across
  // removals or failed parses that consumed an id.
  const nextReportId = useRef(1);
  // Tracks the previously supplied native document's id so a replacement can
  // tell "the old native doc was active" apart from "the user picked an
  // import" without re-deriving it from render state.
  const previousNativeId = useRef<string | null>(null);

  useEffect(() => {
    const newNativeId = nativeDocument?.id ?? null;
    const oldNativeId = previousNativeId.current;
    setSelectedId((current) => {
      if (newNativeId === null) {
        // Native document cleared: fall back only if it was the active one.
        return current === oldNativeId ? (imports[0]?.id ?? null) : current;
      }
      if (current === null || current === oldNativeId) {
        return newNativeId;
      }
      // A user-selected imported report stays selected.
      return current;
    });
    previousNativeId.current = newNativeId;
    // `imports` intentionally excluded: this effect reacts to native-document
    // identity changes only, not to unrelated import-list edits.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nativeDocument]);

  // Global guard: keeps a stray drop from navigating the page away from the
  // app, without intercepting drags meant for anything else (e.g. the graph
  // canvas's own `application/hipdnn-op` payload, which never carries files).
  useEffect(() => {
    function guard(event: DragEvent) {
      if (
        event.dataTransfer &&
        Array.from(event.dataTransfer.types).includes("Files")
      ) {
        event.preventDefault();
      }
    }
    window.addEventListener("dragover", guard);
    window.addEventListener("drop", guard);
    return () => {
      window.removeEventListener("dragover", guard);
      window.removeEventListener("drop", guard);
    };
  }, []);

  async function importFiles(files: readonly File[]): Promise<void> {
    const parsed: ResultDocument[] = [];
    const failures: ImportError[] = [];
    for (const file of files) {
      let text: string;
      try {
        text = await file.text();
      } catch (error) {
        failures.push({
          label: file.name,
          message: `Could not read ${file.name}: ${errorMessage(error)}`,
        });
        continue;
      }
      const id = `report-${nextReportId.current++}`;
      try {
        parsed.push(parseResults(text, file.name, id));
      } catch (error) {
        const message =
          error instanceof ResultsParseError
            ? error.message
            : errorMessage(error);
        failures.push({ label: file.name, message });
      }
    }
    setErrors(failures);
    if (parsed.length > 0) {
      setImports((prev) => [...prev, ...parsed]);
      setSelectedId(parsed[parsed.length - 1].id);
    }
  }

  function handleDragOver(event: React.DragEvent<HTMLDivElement>): void {
    if (Array.from(event.dataTransfer.types).includes("Files")) {
      event.preventDefault();
      setDragActive(true);
    }
  }

  function handleDragLeave(): void {
    setDragActive(false);
  }

  function handleDrop(event: React.DragEvent<HTMLDivElement>): void {
    event.preventDefault();
    event.stopPropagation();
    setDragActive(false);
    const files = Array.from(event.dataTransfer.files);
    if (files.length === 0) return;
    void importFiles(files);
  }

  async function handleOpenReport(): Promise<void> {
    try {
      const opened = await platform.openTextFile(".json");
      if (!opened) return; // cancelled: last valid document stays visible
      const id = `report-${nextReportId.current++}`;
      const doc = parseResults(opened.contents, opened.handle.name, id);
      setErrors([]);
      setImports((prev) => [...prev, doc]);
      setSelectedId(doc.id);
    } catch (error) {
      const message =
        error instanceof ResultsParseError
          ? error.message
          : errorMessage(error);
      setErrors([
        { label: "Open report", message: `Could not open report: ${message}` },
      ]);
    }
  }

  function handleRemove(): void {
    const index = imports.findIndex((doc) => doc.id === selectedId);
    if (index === -1) return; // only imported documents are removable
    const next = imports.filter((doc) => doc.id !== selectedId);
    setImports(next);
    setSelectedId(next[index]?.id ?? next[index - 1]?.id ?? null);
  }

  const documents: readonly ResultDocument[] = nativeDocument
    ? [nativeDocument, ...imports]
    : imports;
  const selected = documents.find((doc) => doc.id === selectedId) ?? null;
  const isImportedSelected = imports.some((doc) => doc.id === selectedId);

  return (
    <div className="results" data-empty={!selected}>
      <header className="results__source-bar">
        <div className="results__brand">
          <span className="results__brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none">
              <path
                d="M5 18V11M12 18V5M19 18V8"
                stroke="currentColor"
                strokeWidth="3"
                strokeLinecap="round"
              />
            </svg>
          </span>
          <div>
            <strong>Benchmark results</strong>
            <span>hipDNN Graph Studio</span>
          </div>
        </div>
        <div className="results__source-actions">
          <label className="results__source-label">
            <span className="results__sr-only">Report</span>
            <select
              className="results__source-select"
              value={selectedId ?? ""}
              onChange={(event) => setSelectedId(event.target.value || null)}
              disabled={documents.length === 0}
            >
              {documents.length === 0 && <option value="">No reports</option>}
              {documents.map((doc) => (
                <option key={doc.id} value={doc.id}>
                  {doc.kind === "native" ? "Current execution" : doc.label}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className="results__button--primary"
            onClick={() => void handleOpenReport()}
          >
            Open report…
          </button>
          <button
            type="button"
            className="results__remove"
            onClick={handleRemove}
            disabled={!isImportedSelected}
          >
            Remove report
          </button>
        </div>
      </header>

      <div
        className={
          dragActive ? "results__drop results__drop--active" : "results__drop"
        }
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <span>
          {dragActive
            ? "Release to import reports"
            : "Drop benchmark JSON here"}
        </span>
        <span className="results__drop-hint">
          Local files only · Nothing is uploaded
        </span>
      </div>

      {errors.length > 0 && (
        <ul className="results__errors" role="alert">
          {errors.map((error, index) => (
            <li key={`${error.label}-${index}`} className="results__error">
              {error.label}: {error.message}
            </li>
          ))}
        </ul>
      )}

      {selected ? (
        <ResultsViewer key={selected.id} document={selected} />
      ) : (
        <div className="results__empty-workspace">
          <span className="results__eyebrow">Performance, with context</span>
          <h1>Explore your benchmark results.</h1>
          <p>
            Compare engines on the same graph, inspect correctness, and drill
            into timings and profiling traces.
          </p>
          <button
            type="button"
            className="results__button--primary"
            onClick={() => void handleOpenReport()}
          >
            Open report…
          </button>
          <p className="results__note">Drop benchmark JSON or open a report</p>
          <div className="results__empty-formats">
            <span>Suite reports</span>
            <span>Raw timings</span>
            <span>Native executions</span>
          </div>
        </div>
      )}
    </div>
  );
}
