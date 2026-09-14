import { detectElectronPlatform } from "./electron";
import type { PlatformBridge } from "./types";
import { webPlatform } from "./web";

export type {
  FileHandleRef,
  KeyValueStore,
  OpenResult,
  PlatformBridge,
  PlatformKind,
  SaveOptions,
} from "./types";

/**
 * The active platform bridge, chosen once at module load. Electron wins when its
 * preload API is present; otherwise the web bridge. Import `platform` anywhere
 * that needs host services — nothing else in the app should know which host it
 * is running on.
 */
export const platform: PlatformBridge = detectElectronPlatform() ?? webPlatform;
