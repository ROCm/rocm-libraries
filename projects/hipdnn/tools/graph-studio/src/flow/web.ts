import type {
  ArtifactContents,
  CancelResult,
  FlowAck,
  FlowFailure,
  FlowInputsResult,
  FlowListResult,
  FlowResponse,
  FlowRunner,
  LaunchResult,
  RunStatus,
  ValidateResult,
} from "./types";

/**
 * Web flow runner: a browser can neither spawn the orchestrator nor write the
 * graph to disk, so every call fails with one clear message and the UI keeps
 * its controls disabled. The tab still exists, so both builds share one `TabId`
 * set and one keyboard order.
 */

export const FLOW_UNAVAILABLE =
  "Running agent flows is only available in the Electron desktop build.";

const unavailable: FlowFailure = { ok: false, error: FLOW_UNAVAILABLE };

export const webFlowRunner: FlowRunner = {
  available: false,
  async list(): Promise<FlowResponse<FlowListResult>> {
    return unavailable;
  },
  async inputs(): Promise<FlowResponse<FlowInputsResult>> {
    return unavailable;
  },
  async validate(): Promise<FlowResponse<ValidateResult>> {
    return unavailable;
  },
  async launch(): Promise<FlowResponse<LaunchResult>> {
    return unavailable;
  },
  async cancel(): Promise<FlowResponse<CancelResult>> {
    return unavailable;
  },
  async status(): Promise<FlowResponse<RunStatus>> {
    return unavailable;
  },
  async artifact(): Promise<FlowResponse<ArtifactContents>> {
    return unavailable;
  },
  async revealArtifact(): Promise<FlowAck> {
    return unavailable;
  },
  onEvent(): () => void {
    return () => {};
  },
};
