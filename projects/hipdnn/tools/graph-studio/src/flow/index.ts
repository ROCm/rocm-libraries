import { detectElectronFlowRunner } from "./electron";
import type { FlowRunner } from "./types";
import { webFlowRunner } from "./web";

export { FLOW_UNAVAILABLE } from "./web";
export type {
  ArtifactContents,
  ArtifactSource,
  CancelResult,
  FlowArtifact,
  FlowEvent,
  FlowInputSpec,
  FlowListResult,
  FlowLoopSpec,
  FlowOutputSpec,
  FlowResponse,
  FlowRunner,
  FlowStepSpec,
  FlowSummary,
  LaunchRequest,
  LaunchResult,
  RunLoop,
  RunState,
  RunStatus,
  RunStep,
  StepKind,
  ValidateResult,
} from "./types";

/**
 * The active flow runner. MCP-backed under Electron; an "unavailable" stub in a
 * plain browser. Chosen once at load, mirroring the platform, engine and
 * command selectors.
 */
export const flowRunner: FlowRunner = detectElectronFlowRunner() ?? webFlowRunner;
