import { NavLink } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  LayoutGrid,
  Sprout,
  CloudSun,
  Leaf,
  IndianRupee,
  Bug,
  Activity,
  ChevronDown,
} from "lucide-react";
import { useAppState } from "../state/AppState";
import { api } from "../lib/api";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutGrid, end: true },
  { to: "/soil", label: "Soil Intelligence", icon: Sprout },
  { to: "/weather", label: "Weather Advisory", icon: CloudSun },
  { to: "/crop-health", label: "Crop Health", icon: Leaf },
  { to: "/market", label: "Market Prices", icon: IndianRupee },
  { to: "/pest-risk", label: "Pest Risk", icon: Bug },
  { to: "/system", label: "System Status", icon: Activity },
];

export default function Sidebar() {
  const { district, season, stage, setDistrict, setSeason, setStage, meta } = useAppState();

  const { data: status } = useQuery({
    queryKey: ["system-status"],
    queryFn: api.systemStatus,
    refetchInterval: 10_000,
  });

  const districts = meta?.districts ?? [district];
  const stages = meta?.growth_stages ?? [stage];

  return (
    <aside className="flex h-screen w-[278px] shrink-0 flex-col overflow-y-auto bg-gradient-to-b from-forest-950 to-forest-900 px-5 py-6">
      {/* Brand */}
      <div className="mb-7 flex items-center gap-3">
        <div className="flex h-11 w-11 items-center justify-center rounded-full bg-forest-700/70 ring-1 ring-forest-400/30">
          <Leaf className="h-5 w-5 text-forest-200" />
        </div>
        <div>
          <div className="font-serif text-[21px] leading-none text-white">KrishiSense</div>
          <div className="mt-1 text-[10px] font-medium uppercase tracking-[0.14em] text-forest-400">
            Paddy Advisory Mesh
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="space-y-1">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-full px-4 py-2.5 text-[14px] transition ${
                isActive
                  ? "bg-forest-700 font-medium text-white shadow-sm"
                  : "text-forest-200/80 hover:bg-white/5 hover:text-white"
              }`
            }
          >
            <Icon className="h-[18px] w-[18px]" />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Selectors */}
      <div className="mt-7 rounded-xl2 bg-black/20 p-4">
        <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-forest-400">
          District
        </div>
        <div className="relative">
          <select value={district} onChange={(e) => setDistrict(e.target.value)} className="field appearance-none pr-8">
            {districts.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
          <ChevronDown className="pointer-events-none absolute right-2.5 top-3 h-4 w-4 text-forest-200/70" />
        </div>

        <div className="mb-1.5 mt-4 text-[10px] font-semibold uppercase tracking-[0.12em] text-forest-400">
          Season
        </div>
        <div className="grid grid-cols-2 gap-2">
          {["Kharif", "Rabi"].map((s) => (
            <button
              key={s}
              onClick={() => setSeason(s)}
              className={`rounded-lg px-3 py-2 text-[13px] font-medium transition ${
                season === s
                  ? "border border-gold/70 bg-gold/10 text-gold-light"
                  : "border border-white/10 bg-white/5 text-forest-200/80 hover:bg-white/10"
              }`}
            >
              {s}
            </button>
          ))}
        </div>

        <div className="mb-1.5 mt-4 text-[10px] font-semibold uppercase tracking-[0.12em] text-forest-400">
          Growth Stage
        </div>
        <div className="relative">
          <select value={stage} onChange={(e) => setStage(e.target.value)} className="field appearance-none pr-8">
            {stages.map((s) => (
              <option key={s} value={s}>
                {s.replace(/_/g, " ")}
              </option>
            ))}
          </select>
          <ChevronDown className="pointer-events-none absolute right-2.5 top-3 h-4 w-4 text-forest-200/70" />
        </div>
      </div>

      {/* Agent mesh */}
      <div className="mt-7">
        <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.12em] text-forest-400">
          Agent Mesh
        </div>
        <div className="space-y-2.5">
          {(status?.active_agents ?? []).map((a) => (
            <div key={a.agent_type} className="flex items-center justify-between text-[13px]">
              <span className="flex items-center gap-2.5 text-forest-200/90">
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    a.status === "ONLINE" ? "live-dot bg-[#4ADE80]" : "bg-forest-400/40"
                  }`}
                />
                {a.agent_name.replace(" Agent", "")}
              </span>
              <span className="text-[12px] text-forest-400/80">
                {a.status === "ONLINE"
                  ? a.last_response_ms < 1
                    ? "<1ms"
                    : `${a.last_response_ms.toFixed(0)}ms`
                  : "offline"}
              </span>
            </div>
          ))}
          {!status && <div className="text-[12px] text-forest-400/70">probing agents…</div>}
        </div>
      </div>

      <div className="mt-auto pt-8 text-[11px] leading-relaxed text-forest-400/60">
        Gateway :8000 bridges the browser to the gRPC mesh. Agents answer directly when running; otherwise the
        gateway computes the same result from the datasets.
      </div>
    </aside>
  );
}
