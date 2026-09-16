import { useMemo, useRef, useState } from "react";
import {
  coordinates,
  compareTensors,
  histogram,
  loadTensorSet,
  MANIFEST_FILENAME,
  TensorArtifactError,
  type Comparison,
  type Histogram,
  type LoadedTensor,
  type TensorSet,
} from "../benchmark/tensors";

/**
 * Tensor artifact inspector: value distribution for one captured tensor, and an
 * element-wise comparison against a second capture (typically an engine's
 * output against the reference provider's).
 *
 * Artifacts are picked as files rather than resolved from the report, because
 * the manifest paths a report carries belong to the machine that ran the
 * benchmark. Select the whole artifact directory contents — `manifest.json`
 * plus its `.bin` files — in one go.
 *
 * ponytail: file-set picker; add a directory picker or Electron path
 * resolution if selecting the files ever becomes the annoying part.
 */

const BUCKET_CHOICES = [16, 32, 64, 128] as const;

/** Where a capture came from, shown so the right directory is easy to find. */
export interface TensorHint {
  readonly label: string;
  readonly path: string;
}

interface TensorViewProps {
  /** Manifest paths recorded by the report, if a report is loaded. */
  hints?: readonly TensorHint[];
  /** Returns to the report view. Omitted when the view stands alone. */
  onBack?: () => void;
}

interface Slot {
  readonly set: TensorSet | null;
  readonly error: string | null;
  readonly busy: boolean;
}

const EMPTY_SLOT: Slot = { set: null, error: null, busy: false };

export function TensorView({ hints, onBack }: TensorViewProps) {
  const [primary, setPrimary] = useState<Slot>(EMPTY_SLOT);
  const [secondary, setSecondary] = useState<Slot>(EMPTY_SLOT);
  const [selectedUid, setSelectedUid] = useState("");
  const [buckets, setBuckets] = useState<number>(32);
  const [rtol, setRtol] = useState("1e-5");
  const [atol, setAtol] = useState("1e-8");

  const tensors = primary.set?.tensors ?? [];
  const selected = tensors.find((t) => t.entry.uid === selectedUid) ?? tensors[0] ?? null;
  const partner = selected
    ? (secondary.set?.tensors.find((t) => t.entry.uid === selected.entry.uid) ?? null)
    : null;

  const valueHistogram = useMemo(
    () => (selected ? histogram(selected.values, buckets) : null),
    [selected, buckets],
  );

  const comparison = useMemo(() => {
    if (!selected || !partner) return null;
    const r = Number(rtol);
    const a = Number(atol);
    if (!Number.isFinite(r) || !Number.isFinite(a) || r < 0 || a < 0) return null;
    if (selected.values.length !== partner.values.length) return null;
    return compareTensors(selected.values, partner.values, r, a);
  }, [selected, partner, rtol, atol]);

  const diffHistogram = useMemo(
    () => (comparison ? histogram(comparison.absDiffs, buckets) : null),
    [comparison, buckets],
  );

  const load = async (files: FileList | null, into: (slot: Slot) => void) => {
    if (!files || files.length === 0) return;
    into({ set: null, error: null, busy: true });
    try {
      const bytes = new Map<string, Uint8Array>();
      let manifestText: string | null = null;
      for (const file of Array.from(files)) {
        if (file.name === MANIFEST_FILENAME) manifestText = await file.text();
        else bytes.set(file.name, new Uint8Array(await file.arrayBuffer()));
      }
      if (manifestText === null) {
        throw new TensorArtifactError(
          `selection has no ${MANIFEST_FILENAME}; pick the whole artifact directory`,
        );
      }
      const label = files[0].webkitRelativePath?.split("/")[0] || MANIFEST_FILENAME;
      const set = await loadTensorSet(manifestText, bytes, label);
      into({ set, error: null, busy: false });
      setSelectedUid(set.tensors[0]?.entry.uid ?? "");
    } catch (error) {
      into({ set: null, error: (error as Error).message, busy: false });
    }
  };

  return (
    <div className="report tensors">
      <header className="report__head">
        <div className="report__ident">
          <div className="report__eyebrow">Tensor artifacts</div>
          <h2 className="report__title">{primary.set?.manifest.graph.name ?? "No capture loaded"}</h2>
          <div className="report__sub">
            {primary.set
              ? describeSet(primary.set)
              : "Select a manifest.json and its .bin files from one artifact directory."}
          </div>
        </div>
        {onBack && (
          <button type="button" className="tensors__back" onClick={onBack}>
            Back to report
          </button>
        )}
      </header>

      {hints && hints.length > 0 && (
        <details className="report__env">
          <summary>Manifest paths from the report</summary>
          <dl className="report__env-grid">
            {hints.map((hint) => (
              <div key={`${hint.label}:${hint.path}`} style={{ display: "contents" }}>
                <dt>{hint.label}</dt>
                <dd>{hint.path}</dd>
              </div>
            ))}
          </dl>
        </details>
      )}

      <div className="tensors__slots">
        <SlotPicker
          title="Capture"
          hint="Engine output, reference output, or graph inputs."
          slot={primary}
          onFiles={(files) => load(files, setPrimary)}
        />
        <SlotPicker
          title="Compare against"
          hint="A second capture of the same tensors, e.g. the reference row."
          slot={secondary}
          onFiles={(files) => load(files, setSecondary)}
        />
      </div>

      {tensors.length > 0 && selected && (
        <>
          <section className="report__panel">
            <div className="report__panel-head">
              <div>
                <h3 className="report__panel-title">Tensors</h3>
                <p className="report__panel-sub">Logical values, with stride padding removed.</p>
              </div>
              <span className="report__count">
                {tensors.length} tensor{tensors.length === 1 ? "" : "s"}
              </span>
            </div>
            <div className="report__tablewrap">
              <table className="report__table">
                <thead>
                  <tr>
                    <th>UID</th>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Shape</th>
                    <th>Graph strides</th>
                    <th className="num">Elements</th>
                    <th className="num">Stored</th>
                    <th className="num">Min</th>
                    <th className="num">Max</th>
                    <th className="num">Mean</th>
                    <th className="num">NaN / Inf</th>
                  </tr>
                </thead>
                <tbody>
                  {tensors.map((tensor) => (
                    <tr
                      key={tensor.entry.uid}
                      className="tensors__row"
                      data-selected={tensor.entry.uid === selected.entry.uid}
                      onClick={() => setSelectedUid(tensor.entry.uid)}
                    >
                      <td>{tensor.entry.uid}</td>
                      <td>{tensor.entry.name}</td>
                      <td>{tensor.entry.encoding}</td>
                      <td>[{tensor.entry.shape.join(", ")}]</td>
                      <td>[{tensor.entry.graph_strides.join(", ")}]</td>
                      <td className="num">{tensor.stats.count}</td>
                      <td className="num">{tensor.entry.storage_elements}</td>
                      <td className="num">{formatValue(tensor.stats.min)}</td>
                      <td className="num">{formatValue(tensor.stats.max)}</td>
                      <td className="num">{formatValue(tensor.stats.mean)}</td>
                      <td className="num">
                        {tensor.stats.nan} / {tensor.stats.infinite}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="report__panel">
            <div className="report__panel-head">
              <div>
                <h3 className="report__panel-title">
                  Value distribution — tensor {selected.entry.uid} ({selected.entry.name})
                </h3>
                <p className="report__panel-sub">
                  {selected.stats.zeros} zero value{selected.stats.zeros === 1 ? "" : "s"} of{" "}
                  {selected.stats.count}
                </p>
              </div>
              <label className="report__field tensors__buckets">
                <span>Buckets</span>
                <select value={buckets} onChange={(e) => setBuckets(Number(e.target.value))}>
                  {BUCKET_CHOICES.map((choice) => (
                    <option key={choice} value={choice}>
                      {choice}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {valueHistogram && <HistogramChart data={valueHistogram} />}
          </section>

          {renderComparison({
            selected,
            partner,
            secondary,
            comparison,
            diffHistogram,
            rtol,
            atol,
            setRtol,
            setAtol,
          })}
        </>
      )}
    </div>
  );
}

interface ComparisonPanelProps {
  selected: LoadedTensor;
  partner: LoadedTensor | null;
  secondary: Slot;
  comparison: Comparison | null;
  diffHistogram: Histogram | null;
  rtol: string;
  atol: string;
  setRtol: (value: string) => void;
  setAtol: (value: string) => void;
}

function renderComparison(props: ComparisonPanelProps) {
  const { selected, partner, secondary, comparison, diffHistogram, rtol, atol } = props;
  if (!secondary.set) return null;

  return (
    <section className="report__panel">
      <div className="report__panel-head">
        <div>
          <h3 className="report__panel-title">Comparison</h3>
          <p className="report__panel-sub">
            Passes when |actual - expected| &lt;= atol + rtol × |expected|, matching the benchmark
            validator.
          </p>
        </div>
        {comparison && (
          <span className="badge" data-state={comparison.failing === 0 ? "passed" : "failed"}>
            {comparison.failing === 0 ? "within tolerance" : `${comparison.failing} failing`}
          </span>
        )}
      </div>

      <div className="report__filters">
        <label className="report__field">
          <span>Relative tolerance</span>
          <input value={rtol} onChange={(e) => props.setRtol(e.target.value)} />
        </label>
        <label className="report__field">
          <span>Absolute tolerance</span>
          <input value={atol} onChange={(e) => props.setAtol(e.target.value)} />
        </label>
      </div>

      {!partner && (
        <div className="report__empty">
          The second capture has no tensor with UID {selected.entry.uid}.
        </div>
      )}
      {partner && !comparison && (
        <div className="report__empty">
          {partner.values.length === selected.values.length
            ? "Tolerances must be non-negative numbers."
            : `Element counts differ: ${selected.values.length} against ${partner.values.length}.`}
        </div>
      )}

      {comparison && (
        <>
          <div className="report__cards">
            <Stat label="Compared" value={`${comparison.compared}`} />
            <Stat
              label="Failing"
              value={`${comparison.failing}`}
              tone={comparison.failing === 0 ? "ok" : "error"}
            />
            <Stat label="Max absolute" value={formatValue(comparison.maxAbs)} />
            <Stat label="Max relative" value={formatValue(comparison.maxRel)} />
          </div>
          {comparison.nonFinite && (
            <p className="report__panel-sub">
              A NaN or infinite value is present, which the benchmark validator treats as a failure
              regardless of tolerance.
            </p>
          )}
          {diffHistogram && (
            <>
              <div className="report__chart-head">
                <strong>Absolute difference</strong>
                <span>every compared element</span>
              </div>
              <HistogramChart data={diffHistogram} />
            </>
          )}
          {comparison.worst.length > 0 && (
            <div className="report__tablewrap">
              <table className="report__table">
                <thead>
                  <tr>
                    <th>Index</th>
                    <th className="num">Actual</th>
                    <th className="num">Expected</th>
                    <th className="num">Absolute</th>
                    <th className="num">Relative</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.worst.map((element) => (
                    <tr key={element.index}>
                      <td>[{coordinates(selected.entry.shape, element.index).join(", ")}]</td>
                      <td className="num">{formatValue(element.actual)}</td>
                      <td className="num">{formatValue(element.expected)}</td>
                      <td className="num">{formatValue(element.abs)}</td>
                      <td className="num">{formatValue(element.rel)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </section>
  );
}

interface SlotPickerProps {
  title: string;
  hint: string;
  slot: Slot;
  onFiles: (files: FileList | null) => void;
}

function SlotPicker({ title, hint, slot, onFiles }: SlotPickerProps) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  return (
    <div
      className="tensors__slot"
      data-over={over}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        onFiles(e.dataTransfer.files);
      }}
    >
      <div className="tensors__slot-head">
        <strong>{title}</strong>
        <button type="button" onClick={() => input.current?.click()}>
          Choose files…
        </button>
      </div>
      <p className="tensors__slot-hint">{slot.busy ? "Reading…" : hint}</p>
      {slot.set && <p className="tensors__slot-state">{describeSet(slot.set)}</p>}
      {slot.error && (
        <p className="tensors__slot-state" data-tone="error">
          {slot.error}
        </p>
      )}
      <input
        ref={input}
        type="file"
        multiple
        accept=".json,.bin,application/json"
        hidden
        onChange={(e) => onFiles(e.target.files)}
      />
    </div>
  );
}

function HistogramChart({ data }: { data: Histogram }) {
  const peak = Math.max(...data.counts, 1);
  return (
    <div className="hist">
      <div className="hist__bars">
        {data.counts.map((value, i) => (
          <div
            key={i}
            className="hist__bar"
            style={{ height: `${(value / peak) * 100}%` }}
            title={`${bucketLabel(data, i)}: ${value}`}
          />
        ))}
      </div>
      <div className="hist__axis">
        <span>{formatValue(data.low)}</span>
        <span>
          peak {peak}
          {data.skipped > 0 ? ` · ${data.skipped} outside range` : ""}
        </span>
        <span>{formatValue(data.high)}</span>
      </div>
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: "ok" | "error" }) {
  return (
    <div className="report__card">
      <span className="report__card-label">{label}</span>
      <span className="report__card-value tensors__stat" data-tone={tone}>
        {value}
      </span>
    </div>
  );
}

function describeSet(set: TensorSet): string {
  const producer = set.manifest.producer;
  const who = producer?.provider ? ` · ${producer.provider}` : "";
  const engine = producer?.engine_id ? ` engine ${producer.engine_id}` : "";
  return `${set.manifest.phase}${who}${engine} · ${set.tensors.length} tensor${
    set.tensors.length === 1 ? "" : "s"
  } · ${set.label}`;
}

function bucketLabel(data: Histogram, index: number): string {
  const width = (data.high - data.low) / data.buckets;
  return `${formatValue(data.low + index * width)} … ${formatValue(data.low + (index + 1) * width)}`;
}

/** Compact fixed/exponential formatting that keeps small differences visible. */
function formatValue(value: number): string {
  if (Number.isNaN(value)) return "NaN";
  if (!Number.isFinite(value)) return value > 0 ? "∞" : "-∞";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1e6 || magnitude < 1e-4) return value.toExponential(3);
  return value.toPrecision(6).replace(/\.?0+$/, "");
}
