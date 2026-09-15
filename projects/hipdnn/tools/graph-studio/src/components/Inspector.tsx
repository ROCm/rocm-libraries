import { catalogEntry, visibleParams } from "../graph/catalog";
import type { OpNode } from "../graph/flow";
import type { ParamSpec, ParamValue } from "../graph/model";

/**
 * Properties inspector for the selected node. Renders a typed editor per
 * parameter spec and reports edits upward; the parent applies them to React
 * Flow state.
 */

interface InspectorProps {
  node: OpNode | null;
  onRename(id: string, title: string): void;
  onParamChange(id: string, key: string, value: ParamValue): void;
  onDelete(id: string): void;
}

export function Inspector({ node, onRename, onParamChange, onDelete }: InspectorProps) {
  if (!node) {
    return (
      <aside className="panel inspector">
        <h2 className="panel__title">Properties</h2>
        <p className="inspector__empty">Select a node to edit its properties.</p>
      </aside>
    );
  }

  const entry = catalogEntry(node.data.opType);
  const params = entry ? visibleParams(entry, node.data.params) : [];

  return (
    <aside className="panel inspector">
      <h2 className="panel__title">Properties</h2>
      <div className="inspector__scroll">
        <label className="field">
          <span className="field__label">Title</span>
          <input
            className="field__input"
            value={node.data.title}
            onChange={(event) => onRename(node.id, event.target.value)}
          />
        </label>
        <div className="field">
          <span className="field__label">Operator</span>
          <div className="inspector__optype">{node.data.opType}</div>
        </div>

        {params.length > 0 && <div className="inspector__divider" />}

        {params.map((spec) => (
          <ParamField
            key={spec.key}
            spec={spec}
            value={node.data.params[spec.key]}
            onChange={(value) => onParamChange(node.id, spec.key, value)}
          />
        ))}

        <button type="button" className="inspector__delete" onClick={() => onDelete(node.id)}>
          Delete node
        </button>
      </div>
    </aside>
  );
}

interface ParamFieldProps {
  spec: ParamSpec;
  value: ParamValue | undefined;
  onChange(value: ParamValue): void;
}

function ParamField({ spec, value, onChange }: ParamFieldProps) {
  const current = value ?? spec.default;

  if (spec.type === "bool") {
    return (
      <label className="field field--inline">
        <input
          type="checkbox"
          checked={Boolean(current)}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span className="field__label">{spec.label}</span>
      </label>
    );
  }

  if (spec.type === "enum") {
    return (
      <label className="field">
        <span className="field__label">{spec.label}</span>
        <select
          className="field__input"
          value={String(current)}
          onChange={(event) => onChange(event.target.value)}
        >
          {spec.options?.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </label>
    );
  }

  const numeric = spec.type === "int" || spec.type === "float";
  return (
    <label className="field">
      <span className="field__label">{spec.label}</span>
      <input
        className="field__input"
        type={numeric ? "number" : "text"}
        step={spec.type === "float" ? "any" : undefined}
        value={String(current)}
        onChange={(event) => {
          if (!numeric) {
            onChange(event.target.value);
            return;
          }
          const parsed = spec.type === "int" ? parseInt(event.target.value, 10) : parseFloat(event.target.value);
          onChange(Number.isNaN(parsed) ? 0 : parsed);
        }}
      />
    </label>
  );
}
