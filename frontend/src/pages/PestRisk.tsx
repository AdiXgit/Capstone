import { useQuery } from "@tanstack/react-query";
import { Bug, CalendarClock, Droplets, ShieldAlert, Thermometer } from "lucide-react";
import { api } from "../lib/api";
import { useAppState } from "../state/AppState";
import { Badge, Card, CardSkeleton, ErrorState, InfoBox, SectionTitle, Skeleton, SourceTag, StatCard } from "../components/ui";

const RISK_ARC: Record<string, { pct: number; color: string }> = {
  LOW: { pct: 28, color: "#40916C" },
  MEDIUM: { pct: 62, color: "#D4A017" },
  HIGH: { pct: 92, color: "#D9534F" },
};

function RiskGauge({ level, score }: { level: string; score: number }) {
  const cfg = RISK_ARC[level] ?? RISK_ARC.LOW;
  const r = 58;
  const circ = Math.PI * r; // half circle
  const filled = (cfg.pct / 100) * circ;
  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 140 78" className="w-[190px]">
        <path d="M12 70 A58 58 0 0 1 128 70" fill="none" stroke="#D8F3DC" strokeWidth={13} strokeLinecap="round" />
        <path
          d="M12 70 A58 58 0 0 1 128 70"
          fill="none"
          stroke={cfg.color}
          strokeWidth={13}
          strokeLinecap="round"
          strokeDasharray={`${filled} ${circ}`}
        />
      </svg>
      <div className="-mt-6 text-center">
        <div className="font-serif text-[30px] leading-none" style={{ color: cfg.color }}>
          {level}
        </div>
        <div className="mt-1.5 text-[12px] text-ink-muted">rule score {score} / 8</div>
      </div>
    </div>
  );
}

export default function PestRisk() {
  const { district, season, stage } = useAppState();
  const q = useQuery({
    queryKey: ["pest-risk", district, season, stage],
    queryFn: () => api.pestRisk(district, season, stage),
  });
  const d = q.data;

  return (
    <div className="space-y-6">
      <SectionTitle
        title="Pest Risk"
        subtitle={`KVK rule engine over season × stage × weather · ${district}`}
        right={<SourceTag meta={d?._meta} />}
      />

      {q.isError && <ErrorState error={q.error} onRetry={() => q.refetch()} />}

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[0.85fr_1.6fr]">
        <Card className="flex flex-col items-center justify-center p-7">
          {q.isLoading ? (
            <Skeleton className="h-40 w-40 rounded-full" />
          ) : (
            d && (
              <>
                <RiskGauge level={d.risk_level} score={d.risk_score} />
                <div className="mt-5 flex items-center gap-2 rounded-full bg-forest-100/70 px-4 py-2 text-[13px] text-forest-800">
                  <CalendarClock className="h-4 w-4" />
                  Re-scout in {d.next_checkin_days} days
                </div>
              </>
            )
          )}
        </Card>

        <div className="space-y-4">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            {q.isLoading && Array.from({ length: 3 }).map((_, i) => <CardSkeleton key={i} />)}
            {d && (
              <>
                <StatCard
                  label="Max temperature"
                  value={`${d.conditions.temperature_max.toFixed(1)}°`}
                  unit="C"
                  accent="red"
                />
                <StatCard label="Humidity" value={`${d.conditions.humidity_percent.toFixed(0)}%`} accent="blue" />
                <StatCard
                  label="Avg sprays / season"
                  value={d.avg_sprays_per_season.toFixed(1)}
                  sub={`Historic pest incidence: ${d.historical_pest_incidence}`}
                  accent="gold"
                />
              </>
            )}
          </div>

          {d && (
            <Card className="p-6">
              <div className="mb-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.1em] text-ink-muted">
                <ShieldAlert className="h-3.5 w-3.5" /> Rule matched
              </div>
              <p className="text-[14px] leading-relaxed text-ink">{d.rule_matched}</p>
              <div className="mt-4">
                <InfoBox tone={d.risk_level === "HIGH" ? "warning" : "default"} title="Preventive action">
                  {d.preventive_action}
                </InfoBox>
              </div>
            </Card>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <Card className="p-7">
          <h3 className="text-[22px] text-forest-900">Likely pests</h3>
          <p className="mt-1 text-sm text-ink-muted">Expected at {stage.replace(/_/g, " ")} stage</p>
          {q.isLoading && <Skeleton className="mt-5 h-32 w-full" />}
          {d && (
            <div className="mt-4 space-y-2.5">
              {d.likely_pests.length === 0 && <p className="text-sm text-ink-muted">No major pest expected at this stage.</p>}
              {d.likely_pests.map((p) => {
                const action = d.pest_actions.find((a) => a.name === p);
                return (
                  <div key={p} className="rounded-xl2 border border-forest-200/70 bg-forest-100/40 p-4">
                    <div className="flex items-center gap-2 text-[14.5px] font-semibold text-forest-900">
                      <Bug className="h-4 w-4 text-forest-700" /> {p}
                    </div>
                    {action && <p className="mt-1.5 text-[13.5px] leading-relaxed text-ink-soft">{action.treatment}</p>}
                  </div>
                );
              })}
            </div>
          )}
        </Card>

        <Card className="p-7">
          <h3 className="text-[22px] text-forest-900">Likely diseases</h3>
          <p className="mt-1 text-sm text-ink-muted">
            Historic disease incidence in {district}: {d?.historical_disease_incidence ?? "—"}
          </p>
          {q.isLoading && <Skeleton className="mt-5 h-32 w-full" />}
          {d && (
            <div className="mt-4 space-y-2.5">
              {d.likely_diseases.map((dis) => {
                const action = d.pest_actions.find((a) => a.name === dis);
                return (
                  <div key={dis} className="rounded-xl2 border border-forest-200/70 bg-forest-100/40 p-4">
                    <div className="flex items-center gap-2 text-[14.5px] font-semibold text-forest-900">
                      <Droplets className="h-4 w-4 text-forest-700" /> {dis}
                    </div>
                    {action && <p className="mt-1.5 text-[13.5px] leading-relaxed text-ink-soft">{action.treatment}</p>}
                  </div>
                );
              })}
            </div>
          )}
        </Card>
      </div>

      {d && (
        <Card className="flex items-start gap-3 p-5">
          <Thermometer className="mt-0.5 h-4 w-4 shrink-0 text-ink-muted" />
          <p className="text-[13px] leading-relaxed text-ink-muted">{d._meta.notes}</p>
        </Card>
      )}
    </div>
  );
}
