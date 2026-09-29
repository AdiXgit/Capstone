import { useQuery } from "@tanstack/react-query";
import { Network, ArrowRight, CheckCircle2, XCircle, Cpu } from "lucide-react";
import { api, OrchestratorAgentResult } from "../lib/api";
import { useAppState } from "../state/AppState";
import { Card, ErrorState, InfoBox, SectionTitle, Skeleton, SourceTag } from "../components/ui";

function pct(n: number) {
  return `${Math.round((n ?? 0) * 100)}%`;
}

function AgentCard({ r }: { r: OrchestratorAgentResult }) {
  const ok = r.status === "OK";
  const p = r.payload as Record<string, unknown>;
  const rows: [string, string][] = [];

  const add = (label: string, val: unknown, suffix = "") => {
    if (val !== undefined && val !== null && val !== "") rows.push([label, `${val}${suffix}`]);
  };

  if (r.agent_name.includes("Weather")) {
    add("Condition", p.weather_condition);
    add("Temp", p.temperature_max, "°C");
    add("Rain", p.rainfall_mm, " mm");
    add("Humidity", p.humidity_percent, "%");
  } else if (r.agent_name.includes("Soil")) {
    add("Health", p.soil_health_score, "/10");
    add("N·P·K", `${p.nitrogen}·${p.phosphorus}·${p.potassium}`);
    add("pH", p.ph);
  } else if (r.agent_name.includes("Crop")) {
    add("Status", p.crop_status);
    add("Stress", p.stress_level);
    add("NDVI", p.ndvi);
    add("Disease risk", p.disease_risk);
  }

  return (
    <div className={`rounded-xl2 border p-4 ${ok ? "border-forest-200 bg-forest-100/40" : "border-[#FCA5A5] bg-[#FEF2F2]"}`}>
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-2 text-[14px] font-semibold text-forest-900">
          <Cpu className="h-4 w-4 text-forest-700" />
          {r.agent_name}
        </span>
        {ok ? (
          <CheckCircle2 className="h-4 w-4 text-[#166534]" />
        ) : (
          <XCircle className="h-4 w-4 text-[#B91C1C]" />
        )}
      </div>
      {ok ? (
        <>
          <div className="mt-3 space-y-1.5">
            {rows.map(([k, v]) => (
              <div key={k} className="flex justify-between text-[13px]">
                <span className="text-ink-muted">{k}</span>
                <span className="font-medium text-ink">{v}</span>
              </div>
            ))}
          </div>
          <div className="mt-3 text-[12px] text-ink-muted">confidence {pct(r.confidence)}</div>
        </>
      ) : (
        <div className="mt-2 text-[12px] text-[#7F1D1D]">
          {r.status} — {String((r.payload as Record<string, unknown>)?.error ?? "no response")}
        </div>
      )}
    </div>
  );
}

export default function Orchestrator() {
  const { district, season, stage } = useAppState();
  const q = useQuery({
    queryKey: ["orchestrator", district, season, stage],
    queryFn: () => api.orchestrator(district, season, stage),
  });
  const d = q.data;

  return (
    <div className="space-y-6">
      <SectionTitle
        title="Orchestrator"
        subtitle="One query fanned out across the agent mesh, then synthesized into a single advisory"
        right={<SourceTag meta={d?._meta} />}
      />

      {q.isError && <ErrorState error={q.error} onRetry={() => q.refetch()} />}
      {q.isLoading && <Skeleton className="h-40 w-full" />}

      {d && !d.online && (
        <InfoBox tone="warning" title="Orchestrator offline">
          The cross-agent orchestrator is not running. Start the whole mesh with{" "}
          <code>bash scripts/run_local.sh</code>, or just the orchestrator with{" "}
          <code>python python/orchestrator/orchestrator_server.py</code>. {d._meta?.notes}
        </InfoBox>
      )}

      {d && d.online && (
        <>
          {/* Flow */}
          <Card className="p-7">
            <div className="flex flex-col items-stretch gap-4 lg:flex-row lg:items-center">
              <div className="rounded-xl2 border border-forest-200 bg-white px-5 py-4 text-center">
                <div className="label-cap">Farmer query</div>
                <div className="mt-1 text-[15px] font-semibold text-forest-900">
                  {d.district} · {d.season}
                </div>
                <div className="text-[12px] text-ink-muted">
                  {d.query_type} · DAT {d.days_after_transplant}
                </div>
              </div>

              <ArrowRight className="mx-auto hidden h-5 w-5 shrink-0 text-forest-400 lg:block" />

              <div className="rounded-xl2 bg-forest-900 px-5 py-4 text-center text-white">
                <div className="flex items-center justify-center gap-2">
                  <Network className="h-4 w-4" />
                  <span className="text-[15px] font-semibold">Orchestrator</span>
                </div>
                <div className="mt-1 text-[12px] text-forest-200">
                  fan-out to {d.agents_consulted ?? d.agent_results.length} agents
                </div>
              </div>

              <ArrowRight className="mx-auto hidden h-5 w-5 shrink-0 text-forest-400 lg:block" />

              <div className="grid flex-1 grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {d.agent_results.map((r) => (
                  <AgentCard key={r.agent_name} r={r} />
                ))}
              </div>
            </div>
          </Card>

          {/* Synthesis */}
          <Card className="p-7">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h3 className="text-[22px] text-forest-900">Synthesized advisory</h3>
              <div className="flex items-center gap-2">
                <span className="pill bg-forest-100 text-forest-800">
                  {d.agents_consulted ?? d.agent_results.length} agents consulted
                </span>
                <span className="pill bg-[#F3F4F6] text-ink-muted">
                  {pct(d.overall_confidence)} overall confidence
                </span>
              </div>
            </div>
            <pre className="mt-4 whitespace-pre-wrap rounded-xl2 border border-forest-200/80 bg-forest-100/40 p-5 font-sans text-[14px] leading-relaxed text-ink">
              {d.recommendation}
            </pre>
            <p className="mt-3 text-[12px] text-ink-muted">
              This differs from the Overview page: there the gateway fans in each domain independently. Here a single
              query is routed through the gRPC OrchestratorService, which calls the agents concurrently and merges
              their results — including a cross-agent rule that escalates when signals combine (e.g. rain + disease
              risk, or the flowering spray-lock).
            </p>
          </Card>
        </>
      )}
    </div>
  );
}
