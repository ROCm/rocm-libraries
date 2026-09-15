import { useCallback, useState } from "react";
import { flowRunner } from "../flow";
import type {
  Timeline,
  TimelineGroup,
  TimelineIteration,
  TimelineOutput,
  TimelineStep,
} from "../flow/timeline";
import type { FlowArtifact } from "../flow";

/**
 * Draws a `Timeline`. Everything shown is read out of the projection — step
 * ids, tool names, statuses, output keys and their declared types, group
 * budgets and each group's exit condition as the orchestrator measured it.
 * Nothing here knows what any of them mean, and nothing here decides whether a
 * run went well: the run's own status is the outcome.
 *
 * Artifacts are read on demand through the host, by the URI the run reported
 * for them.
 */

/** One artifact's fetched contents, or why they could not be fetched. */
interface ArtifactView {
  readonly loading: boolean;
  readonly text: string;
  readonly error: string | null;
  readonly truncated: boolean;
  readonly size: number;
}

interface ArtifactViews {
  readonly views: Readonly<Record<string, ArtifactView>>;
  onToggle(uri: string): void;
  onReveal(uri: string): void;
}

function OutputValue({ output, artifacts }: { output: TimelineOutput; artifacts: ArtifactViews }) {
  if (output.render === "link" && output.artifactUri) {
    return (
      <ArtifactRow
        label={output.name}
        detail={output.text}
        uri={output.artifactUri}
        artifacts={artifacts}
      />
    );
  }
  if (output.render === "json") {
    return (
      <details className="timeline__json">
        <summary className="timeline__output-name">
          {output.name}
          <span className="timeline__output-type">{output.type}</span>
        </summary>
        <pre className="timeline__json-body">{output.text}</pre>
      </details>
    );
  }
  return (
    <div className="timeline__output" data-render={output.render}>
      <span className="timeline__output-name">
        {output.name}
        <span className="timeline__output-type">{output.type}</span>
      </span>
      <span className="timeline__output-value">{output.text}</span>
    </div>
  );
}

function ArtifactRow({
  label,
  detail,
  uri,
  artifacts,
}: {
  label: string;
  detail: string;
  uri: string;
  artifacts: ArtifactViews;
}) {
  const view = artifacts.views[uri];
  return (
    <div className="timeline__artifact">
      <div className="timeline__artifact-bar">
        <button
          type="button"
          className="timeline__artifact-open"
          onClick={() => artifacts.onToggle(uri)}
          aria-expanded={view !== undefined && !view.loading}
        >
          {label}
        </button>
        <span className="timeline__artifact-path" title={detail}>
          {detail}
        </span>
        <button
          type="button"
          className="timeline__artifact-reveal"
          onClick={() => artifacts.onReveal(uri)}
          disabled={!flowRunner.available}
        >
          Reveal
        </button>
      </div>
      {view && (
        <div className="timeline__artifact-body">
          {view.loading && <span className="timeline__dim">Reading…</span>}
          {view.error && <span className="timeline__error">{view.error}</span>}
          {!view.loading && !view.error && (
            <>
              <pre className="timeline__artifact-text">{view.text}</pre>
              <div className="timeline__dim">
                {view.size} bytes{view.truncated ? ", showing the head of the file" : ""}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function StepCard({ step, artifacts }: { step: TimelineStep; artifacts: ArtifactViews }) {
  return (
    <div className="timeline__step" data-status={step.status} data-current={step.current}>
      <div className="timeline__step-head">
        <span className="timeline__step-id">{step.id}</span>
        {step.tool && <span className="timeline__tool">{step.tool}</span>}
        {step.toolKind && <span className="timeline__kind">{step.toolKind}</span>}
        <span className="timeline__status">{step.status}</span>
        {step.exitCode !== null && <span className="timeline__exit">exit {step.exitCode}</span>}
        {step.timedOut && <span className="timeline__exit">timed out</span>}
        {step.durationText && <span className="timeline__dim">{step.durationText}</span>}
      </div>
      {step.error && <div className="timeline__error">{step.error}</div>}
      {step.dir && <div className="timeline__dim timeline__step-dir">{step.dir}</div>}
      {step.outputs.length > 0 && (
        <div className="timeline__outputs">
          {step.outputs.map((output) => (
            <OutputValue key={output.name} output={output} artifacts={artifacts} />
          ))}
        </div>
      )}
    </div>
  );
}

function IterationRow({
  group,
  row,
  artifacts,
}: {
  group: TimelineGroup;
  row: TimelineIteration;
  artifacts: ArtifactViews;
}) {
  return (
    <div className="timeline__iteration" data-current={row.current}>
      <div className="timeline__iteration-label">
        iteration {row.ordinal}
        {group.budget > 0 ? ` of ${group.budget}` : ""}
      </div>
      <div className="timeline__iteration-steps">
        {row.steps.map((step) => (
          <StepCard key={step.key} step={step} artifacts={artifacts} />
        ))}
      </div>
    </div>
  );
}

function GroupCard({ group, artifacts }: { group: TimelineGroup; artifacts: ArtifactViews }) {
  return (
    <div className="timeline__group" data-current={group.current} data-satisfied={group.satisfied}>
      <div className="timeline__group-head">
        <span className="timeline__group-id">{group.id}</span>
        <span className="timeline__dim">
          {group.iterations} of {group.budget} iterations
          {group.budgetSource ? ` (${group.budgetSource})` : ""}
        </span>
        <span className="timeline__group-satisfied">
          {group.satisfied ? "condition met" : "condition not met"}
        </span>
      </div>
      {group.untilMeasured && (
        <div className="timeline__until" title={group.untilText}>
          {group.untilMeasured}
        </div>
      )}
      {group.iterationRows.length === 0 ? (
        <div className="timeline__dim timeline__iteration">No iteration recorded yet.</div>
      ) : (
        group.iterationRows.map((row) => (
          <IterationRow key={row.index} group={group} row={row} artifacts={artifacts} />
        ))
      )}
    </div>
  );
}

function ArtifactGroups({ timeline, artifacts }: { timeline: Timeline; artifacts: ArtifactViews }) {
  if (timeline.artifactCount === 0) return null;
  return (
    <div className="timeline__artifacts">
      <h4 className="timeline__section">Artifacts ({timeline.artifactCount})</h4>
      {timeline.artifactGroups.map((group) => (
        <div className="timeline__artifact-group" key={group.key}>
          <div className="timeline__artifact-group-head">
            <span className="timeline__group-id">{group.stepId ?? group.source}</span>
            {group.iteration !== null && (
              <span className="timeline__dim">iteration {group.iteration + 1}</span>
            )}
            <span className="timeline__dim">{group.source}</span>
          </div>
          {group.artifacts.map((artifact: FlowArtifact) => (
            <ArtifactRow
              key={artifact.uri}
              label={artifact.label}
              detail={artifact.role ?? artifact.mimeType}
              uri={artifact.uri}
              artifacts={artifacts}
            />
          ))}
        </div>
      ))}
    </div>
  );
}

export function FlowTimeline({ timeline }: { timeline: Timeline }) {
  const [views, setViews] = useState<Record<string, ArtifactView>>({});

  const onToggle = useCallback((uri: string) => {
    let read = false;
    setViews((prev) => {
      if (prev[uri]) {
        const next = { ...prev };
        delete next[uri];
        return next;
      }
      read = true;
      return { ...prev, [uri]: { loading: true, text: "", error: null, truncated: false, size: 0 } };
    });
    if (!read) return;
    void flowRunner.artifact(uri).then((result) => {
      setViews((prev) => {
        // The pane was closed again while the read was in flight.
        if (!prev[uri]) return prev;
        return {
          ...prev,
          [uri]: result.ok
            ? {
                loading: false,
                text: result.text,
                error: null,
                truncated: result.truncated,
                size: result.size,
              }
            : { loading: false, text: "", error: result.error, truncated: false, size: 0 },
        };
      });
    });
  }, []);

  const onReveal = useCallback((uri: string) => {
    void flowRunner.revealArtifact(uri);
  }, []);

  const artifacts: ArtifactViews = { views, onToggle, onReveal };

  return (
    <div className="timeline">
      <div className="timeline__nodes">
        {timeline.nodes.map((node) =>
          node.kind === "group" ? (
            <GroupCard key={node.key} group={node} artifacts={artifacts} />
          ) : (
            <StepCard key={node.key} step={node} artifacts={artifacts} />
          ),
        )}
      </div>
      <ArtifactGroups timeline={timeline} artifacts={artifacts} />
    </div>
  );
}
