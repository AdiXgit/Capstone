import { useQuery } from "@tanstack/react-query";
import { Server, Radio } from "lucide-react";
import { api } from "../lib/api";
import { Badge, Card, ErrorState, InfoBox, SectionTitle, Skeleton } from "../components/ui";

export default function SystemStatus() {
  const q = useQuery({
    queryKey: ["system-status-page"],
    queryFn: api.systemStatus,
    refetchInterval: 5_000,
  });
  const d = q.data;

  return (
    <div className="space-y-6">
      <SectionTitle
        title="System Status"
        subtitle="Live probe of the orchestrator and all five gRPC agents, refreshed every 5 seconds"
      />

      {q.isError && <ErrorState error={q.error} onRetry={() => q.refetch()} />}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Card className="p-6">
          <div className="label-cap">Orchestrator</div>
          <div className="mt-3 flex items-center gap-2.5">
            <Badge value={d?.orchestrator_status ?? "…"} />
            <span className="text-[13px] text-ink-muted">:{d?.orchestrator_port ?? "50051"}</span>
          </div>
        </Card>
        <Card className="p-6">
          <div className="label-cap">Agents online</div>
          <div className="mt-2 font-serif text-[32px] leading-none text-forest-900">
            {d ? `${d.online_count} / ${d.total_agents}` : "—"}
          </div>
        </Card>
        <Card className="p-6">
          <div className="label-cap">Gateway</div>
          <div className="mt-3 flex items-center gap-2.5">
            <Badge value="ONLINE" />
            <span className="text-[13px] text-ink-muted">:8000 REST bridge</span>
          </div>
        </Card>
      </div>

      <Card className="p-7">
        <h3 className="text-[22px] text-forest-900">Agent mesh</h3>
        <p className="mt-1 text-sm text-ink-muted">Each agent is probed directly over gRPC on its own port</p>

        {q.isLoading && <Skeleton className="mt-5 h-52 w-full" />}
        {d && (
          <div className="mt-5 overflow-x-auto">
            <table className="w-full min-w-[760px] text-[14px]">
              <thead>
                <tr className="bg-forest-900 text-left text-white">
                  <th className="rounded-l-lg px-4 py-3 font-medium">Agent</th>
                  <th className="px-4 py-3 font-medium">Port</th>
                  <th className="px-4 py-3 font-medium">Language</th>
                  <th className="px-4 py-3 font-medium">Status</th>
                  <th className="px-4 py-3 font-medium">Probe</th>
                  <th className="rounded-r-lg px-4 py-3 font-medium">Served by</th>
                </tr>
              </thead>
              <tbody>
                {d.active_agents.map((a, i) => (
                  <tr key={a.agent_type} className={i % 2 ? "bg-forest-100/45" : ""}>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2.5">
                        <Server className="h-4 w-4 text-forest-600" />
                        <div>
                          <div className="font-medium text-forest-900">{a.agent_name}</div>
                          <div className="text-[12px] text-ink-muted">{a.description}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-ink-soft">:{a.port}</td>
                    <td className="px-4 py-3 text-ink-soft">{a.language}</td>
                    <td className="px-4 py-3">
                      <Badge value={a.status} />
                    </td>
                    <td className="px-4 py-3 text-ink-muted">
                      {a.last_response_ms < 1 ? "<1 ms" : `${a.last_response_ms.toFixed(0)} ms`}
                    </td>
                    <td className="px-4 py-3 text-ink-muted">{a.served_by}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <Card className="p-7">
          <h3 className="text-[22px] text-forest-900">How requests flow</h3>
          <div className="mt-5 space-y-3 text-[13.5px] text-ink-soft">
            {[
              ["Browser", "React dashboard on :5173"],
              ["REST gateway", "FastAPI on :8000 — the only thing the browser talks to"],
              ["gRPC mesh", "Orchestrator :50051 and agents :50052–:50056"],
              ["Fallback", "If an agent is down, the gateway computes the same answer from data/"],
            ].map(([title, body], i) => (
              <div key={title} className="flex gap-3">
                <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-forest-900 text-[11px] font-bold text-white">
                  {i + 1}
                </span>
                <div>
                  <div className="font-medium text-forest-900">{title}</div>
                  <div className="text-ink-muted">{body}</div>
                </div>
              </div>
            ))}
          </div>
        </Card>

        <Card className="p-7">
          <h3 className="text-[22px] text-forest-900">Build status</h3>
          <div className="mt-4 space-y-3">
            <InfoBox title="Production agents">
              Crop Health (:50054) is fully implemented — NDVI deviation features, a 4-class Gradient Boosting
              classifier and the KVK treatment matrix, served over gRPC.
            </InfoBox>
            <InfoBox tone="warning" title="In progress">
              Soil and Weather agents exist but are not currently serving; Market Price and Pest Risk agents are
              still being built. Until each one is running, the gateway answers those routes from the datasets in{" "}
              <code className="rounded bg-white/70 px-1 py-0.5">data/</code>, using the same rules the agents will
              serve.
            </InfoBox>
            {d && (
              <div className="flex items-start gap-2.5 rounded-xl2 bg-forest-100/50 p-4 text-[13px] text-ink-soft">
                <Radio className="mt-0.5 h-4 w-4 shrink-0 text-forest-700" />
                {d.gateway_note}
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
