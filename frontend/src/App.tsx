import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { api, useQuery } from "./api/client";
import Overview from "./views/Overview";
import Models from "./views/Models";
import Slices from "./views/Slices";
import Errors from "./views/Errors";
import Diagnostics from "./views/Diagnostics";
import Quality from "./views/Quality";

const NAV = [
  { to: "/overview", label: "Overview", n: "01" },
  { to: "/models", label: "Model Comparison", n: "02" },
  { to: "/slices", label: "Slice Analysis", n: "03" },
  { to: "/errors", label: "Error Analysis", n: "04" },
  { to: "/diagnostics", label: "Diagnostics", n: "05" },
  { to: "/quality", label: "Audit & Quality", n: "06" },
];

function BackendStatus() {
  const { data, error } = useQuery(api.version, "version");
  const ok = !!data && !error;
  return (
    <div className="flex items-center gap-2 rounded-lg border border-[var(--color-line)] bg-[var(--color-surface)] px-3 py-1.5">
      <span
        className="relative flex h-2 w-2"
        title={ok ? "Backend reachable" : "Backend unreachable"}
      >
        <span
          className={`absolute inline-flex h-full w-full rounded-full opacity-60 ${ok ? "animate-ping bg-[var(--color-good)]" : ""}`}
        />
        <span
          className="relative inline-flex h-2 w-2 rounded-full"
          style={{ background: ok ? "var(--color-good)" : "var(--color-bad)" }}
        />
      </span>
      <span className="text-xs text-[var(--color-ink-2)]">
        {ok ? (
          <>
            API <span className="nums">v{data!.version}</span>
          </>
        ) : (
          "API offline"
        )}
      </span>
    </div>
  );
}

export default function App() {
  return (
    <div className="min-h-screen">
      <div className="mx-auto flex max-w-[1240px] gap-0 px-4 sm:px-6 lg:px-8">
        {/* Sidebar */}
        <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r border-[var(--color-line)] py-8 pr-6 lg:flex">
          <div className="mb-10">
            <div className="font-display text-lg leading-tight text-[var(--color-ink)]">
              Failure
              <br />
              Analysis
            </div>
            <div className="mt-1 text-[11px] uppercase tracking-[0.16em] text-[var(--color-ink-3)]">
              Model Evaluation
            </div>
          </div>
          <nav className="flex flex-col gap-0.5">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `group flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${
                    isActive
                      ? "bg-[var(--color-accent-soft)] font-semibold text-[var(--color-accent-ink)]"
                      : "text-[var(--color-ink-2)] hover:bg-[var(--color-line-2)]"
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    <span
                      className={`nums text-[10px] ${isActive ? "text-[var(--color-accent)]" : "text-[var(--color-ink-3)]"}`}
                    >
                      {item.n}
                    </span>
                    {item.label}
                  </>
                )}
              </NavLink>
            ))}
          </nav>
          <div className="mt-auto space-y-3 pt-8">
            <BackendStatus />
            <p className="px-1 text-[11px] leading-relaxed text-[var(--color-ink-3)]">
              Evaluation is about decisions, not leaderboard scores.
            </p>
          </div>
        </aside>

        {/* Main */}
        <main className="min-w-0 flex-1 py-8 lg:pl-10">
          {/* Mobile nav */}
          <div className="mb-6 flex items-center justify-between lg:hidden">
            <div className="font-display text-lg">Failure Analysis</div>
            <BackendStatus />
          </div>
          <div className="mb-6 flex gap-1.5 overflow-x-auto pb-1 lg:hidden">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `shrink-0 rounded-full px-3 py-1.5 text-xs font-medium ${
                    isActive
                      ? "bg-[var(--color-accent)] text-white"
                      : "bg-[var(--color-surface)] text-[var(--color-ink-2)] border border-[var(--color-line)]"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </div>

          <Routes>
            <Route path="/" element={<Navigate to="/overview" replace />} />
            <Route path="/overview" element={<Overview />} />
            <Route path="/models" element={<Models />} />
            <Route path="/slices" element={<Slices />} />
            <Route path="/errors" element={<Errors />} />
            <Route path="/diagnostics" element={<Diagnostics />} />
            <Route path="/quality" element={<Quality />} />
            <Route path="*" element={<Navigate to="/overview" replace />} />
          </Routes>

          <footer className="mt-16 border-t border-[var(--color-line)] pt-6 text-xs text-[var(--color-ink-3)]">
            ml-failure-analysis-framework · cost-aware, slice-level model evaluation ·
            recommendations are context-dependent by design.
          </footer>
        </main>
      </div>
    </div>
  );
}
