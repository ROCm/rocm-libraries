import { useRef } from "react";

export type TabId = "create" | "authoring" | "implement" | "verify";

interface Tab {
  id: TabId;
  label: string;
}

const TABS: readonly Tab[] = [
  { id: "create", label: "Create" },
  { id: "authoring", label: "Authoring" },
  { id: "implement", label: "Implement" },
  { id: "verify", label: "Verify" },
];

export const tabPanelId = (id: TabId) => `tabpanel-${id}`;
const tabId = (id: TabId) => `tab-${id}`;

interface MainTabsProps {
  active: TabId;
  onSelect(id: TabId): void;
}

export function MainTabs({ active, onSelect }: MainTabsProps) {
  const listRef = useRef<HTMLDivElement>(null);

  const move = (delta: number) => {
    const index = TABS.findIndex((t) => t.id === active);
    const next = (index + delta + TABS.length) % TABS.length;
    onSelect(TABS[next].id);
    // Roving tabindex: the newly selected tab is the only focusable one.
    listRef.current?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next]?.focus();
  };

  return (
    <div className="tabs" role="tablist" aria-label="Workspace sections" ref={listRef}>
      {TABS.map((tab) => {
        const selected = tab.id === active;
        return (
          <button
            key={tab.id}
            type="button"
            role="tab"
            id={tabId(tab.id)}
            aria-selected={selected}
            aria-controls={tabPanelId(tab.id)}
            tabIndex={selected ? 0 : -1}
            className={`tabs__tab${selected ? " tabs__tab--active" : ""}`}
            onClick={() => onSelect(tab.id)}
            onKeyDown={(event) => {
              if (event.key === "ArrowRight") {
                event.preventDefault();
                move(1);
              } else if (event.key === "ArrowLeft") {
                event.preventDefault();
                move(-1);
              }
            }}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}

interface TabPanelProps {
  id: TabId;
  active: TabId;
  className?: string;
  children?: React.ReactNode;
}

export function TabPanel({ id, active, className, children }: TabPanelProps) {
  return (
    <div
      id={tabPanelId(id)}
      role="tabpanel"
      aria-labelledby={tabId(id)}
      className={className ?? "tabpanel"}
      hidden={id !== active}
    >
      {children}
    </div>
  );
}
