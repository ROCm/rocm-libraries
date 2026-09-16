import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { platform } from "../platform";

/**
 * Draggable divider between two panes, plus the hook that remembers where the
 * user left it. The divider only ever changes one neighbour's size; the other
 * one absorbs the difference.
 */

const STORE_PREFIX = "hipdnn.layout.";
const PERSIST_DELAY_MS = 250;
/** Room the absorbing neighbour always keeps, so a drag can never erase it. */
const NEIGHBOUR_RESERVE_PX = 140;
const KEY_STEP_PX = 16;

/** Pane size in pixels, restored from and written back to platform storage. */
export function useStoredSize(key: string, fallback: number): [number, (next: number) => void] {
  const [size, setSize] = useState(fallback);
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => {
    let cancelled = false;
    void platform.store.get(STORE_PREFIX + key).then((saved) => {
      const restored = Number(saved);
      if (!cancelled && saved !== null && Number.isFinite(restored) && restored > 0) {
        setSize(restored);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [key]);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  const update = useCallback(
    (next: number) => {
      setSize(next);
      // A drag fires per pointer move; only the size it settles on is worth a write.
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => {
        void platform.store.set(STORE_PREFIX + key, String(Math.round(next)));
      }, PERSIST_DELAY_MS);
    },
    [key],
  );

  return [size, update];
}

interface ResizerProps {
  /** "x" drags sideways and sizes widths; "y" drags up/down and sizes heights. */
  axis: "x" | "y";
  /** Which neighbour `size` describes — the one before or after the divider. */
  pane: "before" | "after";
  size: number;
  min: number;
  max: number;
  onResize(size: number): void;
  label: string;
}

export function Resizer({ axis, pane, size, min, max, onResize, label }: ResizerProps) {
  const ref = useRef<HTMLDivElement>(null);
  const drag = useRef<{ origin: number; start: number; ceiling: number } | null>(null);

  // The pane on the far side of the divider pays for every pixel gained, so the
  // upper bound depends on how much room it currently has.
  const ceiling = useCallback(() => {
    const handle = ref.current;
    const payer = pane === "before" ? handle?.nextElementSibling : handle?.previousElementSibling;
    if (!(payer instanceof HTMLElement)) return max;
    const room = axis === "x" ? payer.clientWidth : payer.clientHeight;
    return Math.max(min, Math.min(max, size + room - NEIGHBOUR_RESERVE_PX));
  }, [axis, pane, size, min, max]);

  const apply = useCallback(
    (next: number, limit: number) => onResize(Math.round(Math.min(Math.max(next, min), limit))),
    [onResize, min],
  );

  const onPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) return;
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      drag.current = {
        origin: axis === "x" ? event.clientX : event.clientY,
        start: size,
        ceiling: ceiling(),
      };
    },
    [axis, size, ceiling],
  );

  const onPointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const state = drag.current;
      if (!state) return;
      const delta = (axis === "x" ? event.clientX : event.clientY) - state.origin;
      apply(state.start + (pane === "before" ? delta : -delta), state.ceiling);
    },
    [axis, pane, apply],
  );

  const endDrag = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (!drag.current) return;
    drag.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }, []);

  const onKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      const keys = axis === "x" ? ["ArrowLeft", "ArrowRight"] : ["ArrowUp", "ArrowDown"];
      const index = keys.indexOf(event.key);
      if (index < 0) return;
      event.preventDefault();
      const step = index === 0 ? -KEY_STEP_PX : KEY_STEP_PX;
      apply(size + (pane === "before" ? step : -step), ceiling());
    },
    [axis, pane, size, apply, ceiling],
  );

  return (
    <div
      ref={ref}
      role="separator"
      tabIndex={0}
      aria-label={label}
      aria-orientation={axis === "x" ? "vertical" : "horizontal"}
      aria-valuenow={Math.round(size)}
      aria-valuemin={min}
      aria-valuemax={max}
      className={`resizer resizer--${axis}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onKeyDown={onKeyDown}
    />
  );
}
