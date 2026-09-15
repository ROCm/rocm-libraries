import content from "../../samples/content.json";
import type { BenchmarkReport } from "./types";

/**
 * Bundled example report, used until the Verify tab is wired to a real
 * benchmark run. This module is the single seam to replace at that point.
 *
 * The JSON import widens to structural literal types, so the shape is asserted
 * once here rather than at every use.
 */
export const sampleReport = content as unknown as BenchmarkReport;

export const SAMPLE_LABEL = "content.json";
