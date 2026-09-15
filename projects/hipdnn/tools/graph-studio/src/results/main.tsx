/**
 * Standalone results-viewer entry (`/results.html`). Deliberately independent
 * of `../App`, React Flow, engine selection, and command bridges: this page
 * only imports local JSON reports and never touches graph/canvas state.
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { ResultsWorkspace } from "./ResultsWorkspace";
import "../styles.css";
import "./results.css";

const container = document.getElementById("root");
if (!container) throw new Error("Root element #root not found");

createRoot(container).render(
  <StrictMode>
    <ResultsWorkspace />
  </StrictMode>,
);
