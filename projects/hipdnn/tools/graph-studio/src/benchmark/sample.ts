import content from "../../samples/content.json";
import { parseReport } from "./report";

/**
 * Bundled example report, used until the Verify tab is wired to a real
 * benchmark run. This module is the single seam to replace at that point.
 *
 * It goes through the reader rather than being asserted, so the sample gets
 * the same defaults a picked file does and cannot drift from the schema.
 */
export const sampleReport = parseReport(JSON.stringify(content));

export const SAMPLE_LABEL = "content.json";
