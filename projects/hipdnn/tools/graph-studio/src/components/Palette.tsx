import { OP_CATALOG } from "../graph/catalog";
import type { OpCatalogEntry } from "../graph/model";

/**
 * Node palette. Supports two ways to place a node: drag onto the canvas
 * (HTML5 DnD, carrying the op type) or click to drop at a default position.
 */

const DND_MIME = "application/hipdnn-op";
export { DND_MIME };

interface PaletteProps {
  onAdd(opType: string): void;
}

export function Palette({ onAdd }: PaletteProps) {
  const byCategory = new Map<string, OpCatalogEntry[]>();
  for (const entry of OP_CATALOG) {
    const list = byCategory.get(entry.category) ?? [];
    list.push(entry);
    byCategory.set(entry.category, list);
  }

  return (
    <aside className="panel palette">
      <h2 className="panel__title">Operators</h2>
      <div className="palette__scroll">
        {[...byCategory.entries()].map(([category, entries]) => (
          <section className="palette__group" key={category}>
            <h3 className="palette__category">{category}</h3>
            {entries.map((entry) => (
              <button
                type="button"
                key={entry.type}
                className="palette__item"
                draggable
                onDragStart={(event) => {
                  event.dataTransfer.setData(DND_MIME, entry.type);
                  event.dataTransfer.effectAllowed = "move";
                }}
                onClick={() => onAdd(entry.type)}
                title={`Add ${entry.label}`}
              >
                <span className="palette__dot" style={{ background: entry.accent }} />
                <span className="palette__label">{entry.label}</span>
              </button>
            ))}
          </section>
        ))}
      </div>
    </aside>
  );
}
