import { useQuery } from "@tanstack/react-query";
import { Database } from "lucide-react";
import { api } from "../lib/api";
import { useAppState } from "../state/AppState";
import { Badge, Card, CardSkeleton, ErrorState, Skeleton, StatCard } from "../components/ui";

export default function Overview() {
  const { district, season, stage } = useAppState();
  const q = useQuery({
    queryKey: ["overview", district, season, stage],
    queryFn: () => api.overview(district, season, stage),
  });

  const d = q.data;

  return (
    <div className="space-y-6">
      {/* Hero */}
      <div className="relative overflow-hidden rounded-xl3 bg-gradient-to-br from-forest-900 via-forest-800 to-forest-700 px-10 py-11">
        <div className="pointer-events-none absolute -right-10 -top-24 h-80 w-80 rounded-full bg-[#6E8F4E]/25 blur-3xl" />
        <div className="relative">
          <span className="pill bg-white/10 text-[11px] font-semibold uppercase tracking-[0.13em] text-forest-100">
            <span className="live-dot h-1.5 w-1.5 rounded-full bg-gold-light" />
            Live · Orchestrator :50051
          </span>
          <h1 className="mt-6 max-w-3xl text-[46px] leading-[1.08] text-white">
            Paddy intelligence for {district},<br />
            {season} season
          </h1>
          <p className="mt-5 max-w-2xl text-[15px] leading-relaxed text-forest-200/85">
            Five independent agents — soil, weather, crop health, market price and pest risk — are queried in
            parallel and merged into one advisory for the {stage.replace(/_/g, " ")} stage across Karnataka's
            29 paddy districts.
          </p>
          <div className="mt-7 flex flex-wrap gap-2.5">
            <span className="pill bg-white/10 text-forest-100">
              {d ? `${d.online_count} of ${d.agents.length} agents online` : "probing agents…"}
            </span>
            <span className="pill bg-white/10 text-forest-100">
              {d ? `${d.coverage.length} datasets wired` : "loading datasets…"}
            </span>
          </div>
        </div>
      </div>

      {q.isError && <ErrorState error={q.error} onRetry={() => q.refetch()} />}

      {/* Stat cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-5">
        {q.isLoading &&
          Array.from({ length: 5 }).map((_, i) => <CardSkeleton key={i} />)}

        {d && (
          <>
            <StatCard
              label="Soil health score"
              value={d.cards.soil_health_score.toFixed(1)}
              unit="/ 10"
              sub={d.cards.soil_sub}
              accent="green"
            />
            <StatCard
              label="Today's weather"
              value={`${d.cards.temperature.toFixed(1)}°`}
              unit={`/ ${d.cards.temperature_min.toFixed(1)}°C`}
              sub={d.cards.weather_sub}
              accent="blue"
            />
            <StatCard
              label="Crop stress"
              badge={<Badge value={d.cards.stress_level} className="text-[13px]" />}
              sub={d.cards.crop_sub}
              accent="green"
            />
            <StatCard
              label="Mandi price"
              value={`₹${d.cards.market_price.toLocaleString("en-IN")}`}
              unit="/ qtl"
              sub={d.cards.market_sub}
              accent="gold"
            />
            <StatCard
              label="Pest risk"
              badge={<Badge value={d.cards.pest_risk} className="text-[13px]" />}
              sub={d.cards.pest_sub}
              accent="red"
            />
          </>
        )}
      </div>

      {/* Advisory + coverage */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1.7fr_1fr]">
        <Card className="p-7">
          <h2 className="text-[26px] leading-tight text-forest-900">Today's merged advisory</h2>
          <p className="mt-1 text-sm text-ink-muted">
            Fan-in result for {district} · {(d?.growth_stage ?? stage).replace(/_/g, " ")}
          </p>

          <div className="mt-5 space-y-3.5">
            {q.isLoading &&
              Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-24 w-full" />)}

            {d?.advisory.map((a, i) => {
              const isAlert = a.severity === "HIGH";
              const isWarn = a.severity === "MEDIUM";
              const tone = isAlert
                ? "border-[#F3D0D0] bg-[#FDF3F3]"
                : isWarn
                  ? "border-[#F0DFB8] bg-[#FFFBF0]"
                  : "border-forest-200/70 bg-forest-100/45";
              return (
                <div key={i} className={`rounded-xl2 border p-5 ${tone}`}>
                  <div className="mb-2 flex flex-wrap items-center gap-2.5">
                    {(isAlert || isWarn) && <Badge value={a.severity} />}
                    <span className="text-[11px] font-semibold uppercase tracking-[0.1em] text-ink-muted">
                      {isAlert || isWarn ? a.title : a.agent}
                    </span>
                  </div>
                  {!isAlert && !isWarn && (
                    <div className="mb-1.5 text-[15px] font-semibold text-forest-900">{a.title}</div>
                  )}
                  <p className="text-[14.5px] leading-relaxed text-ink">{a.body}</p>
                  <p className="mt-2 text-[13.5px] leading-relaxed text-ink-muted">{a.action}</p>
                </div>
              );
            })}
          </div>
        </Card>

        <Card className="h-fit p-7">
          <h2 className="text-[26px] leading-tight text-forest-900">Dataset coverage</h2>
          <p className="mt-1 text-sm text-ink-muted">What the agents are trained and served on</p>
          <div className="mt-5 divide-y divide-forest-100">
            {q.isLoading &&
              Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="my-3 h-10 w-full" />)}
            {d?.coverage.map((c) => (
              <div key={c.label} className="flex items-start gap-3 py-3.5">
                <Database className="mt-0.5 h-4 w-4 shrink-0 text-forest-600" />
                <div>
                  <div className="text-[14px] font-medium text-forest-900">{c.label}</div>
                  <div className="text-[13px] text-ink-muted">{c.value}</div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}
