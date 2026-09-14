import { Handle, Position, type NodeProps } from "@xyflow/react";
import { catalogEntry } from "../graph/catalog";
import type { OpNode } from "../graph/flow";

/**
 * Custom React Flow node renderer for every operator kind. Draws a titled card
 * with one target handle per catalog input and one source handle per output,
 * evenly distributed along the left/right edges. Handle `id`s are the catalog
 * port ids, which is what {@link toGraph} reads back when serializing edges.
 */
export function OpNodeView({ data, selected }: NodeProps<OpNode>) {
  const entry = catalogEntry(data.opType);
  const accent = entry?.accent ?? "#64748b";
  const inputs = entry?.inputs ?? [];
  const outputs = entry?.outputs ?? [];
  const params = entry?.params ?? [];

  return (
    <div className="op-node" data-selected={selected ? "true" : "false"} style={{ borderColor: accent }}>
      <div className="op-node__header" style={{ background: accent }}>
        <span className="op-node__title">{data.title}</span>
        <span className="op-node__type">{data.opType}</span>
      </div>

      {(inputs.length > 0 || outputs.length > 0) && (
        <div className="op-node__ports">
          <div className="op-node__col op-node__col--in">
            {inputs.map((port) => (
              <div className="op-node__port op-node__port--in" key={`in-${port.id}`}>
                <Handle
                  id={port.id}
                  type="target"
                  position={Position.Left}
                  className="op-handle op-handle--in"
                />
                <span className="op-node__portlabel">{port.label}</span>
              </div>
            ))}
          </div>
          <div className="op-node__col op-node__col--out">
            {outputs.map((port) => (
              <div className="op-node__port op-node__port--out" key={`out-${port.id}`}>
                <span className="op-node__portlabel">{port.label}</span>
                <Handle
                  id={port.id}
                  type="source"
                  position={Position.Right}
                  className="op-handle op-handle--out"
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {params.length > 0 && (
        <div className="op-node__body">
          {params.slice(0, 4).map((spec) => (
            <div className="op-node__param" key={spec.key}>
              <span className="op-node__param-key">{spec.label}</span>
              <span className="op-node__param-value">{String(data.params[spec.key])}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
