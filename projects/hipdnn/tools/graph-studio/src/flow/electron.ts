import type {
  ArtifactContents,
  CancelResult,
  FlowAck,
  FlowEvent,
  FlowInputsResult,
  FlowListResult,
  FlowResponse,
  FlowRunner,
  LaunchRequest,
  LaunchResult,
  RunStatus,
  ValidateRequest,
  ValidateResult,
} from "./types";

/**
 * Electron flow runner. Active when the preload has exposed `window.hipdnnFlow`.
 * The shape below is the IPC contract the preload must satisfy; the main
 * process owns the MCP client, the temp graph file and the event pump.
 *
 * `available` is true whenever the bridge object exists, because the tab's
 * controls are usable as soon as there is a host to ask. A host that cannot
 * reach the orchestrator reports that as the `error` of the call that needed
 * it, which the panel shows verbatim.
 */

export interface ElectronFlowApi {
  list(): Promise<FlowResponse<FlowListResult>>;
  inputs(flow: string): Promise<FlowResponse<FlowInputsResult>>;
  validate(request: ValidateRequest): Promise<FlowResponse<ValidateResult>>;
  launch(request: LaunchRequest): Promise<FlowResponse<LaunchResult>>;
  cancel(runId: string): Promise<FlowResponse<CancelResult>>;
  status(runId: string, logTail?: number): Promise<FlowResponse<RunStatus>>;
  artifact(uri: string): Promise<FlowResponse<ArtifactContents>>;
  revealArtifact(uri: string): Promise<FlowAck>;
  onEvent(listener: (event: FlowEvent) => void): () => void;
}

interface FlowWindow {
  hipdnnFlow?: ElectronFlowApi;
}

export function detectElectronFlowRunner(): FlowRunner | null {
  const api = (window as unknown as FlowWindow).hipdnnFlow;
  if (!api) return null;
  return { ...api, available: true };
}
