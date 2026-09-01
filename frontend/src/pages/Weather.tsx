import { useQuery } from "@tanstack/react-query";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AlertTriangle, CloudRain, Droplets, Sun, Thermometer, Wind } from "lucide-react";
import { api } from "../lib/api";
import { useAppState } from "../state/AppState";
import { Badge, Card, CardSkeleton, ErrorState, InfoBox, SectionTitle, Skeleton, SourceTag, StatCard } from "../components/ui";

const AXIS = { stroke: "#6B7280", fontSize: 12 };

function dayLabel(iso: string) {
  const d = new Date(iso);
  return d.toLocaleDateString("en-IN", { weekday: "short", day: "numeric" });
}

export default function Weather() {
  const { district, season, stage } = useAppState();
  const q = useQuery({
    queryKey: ["weather", district, season, stage],
    queryFn: () => api.weather(district, season, stage),
  });
  const d = q.data;

  return (
    <div className="space-y-6">
      <SectionTitle
        title="Weather Advisory"
        subtitle={`Current conditions, 7-day outlook and stage-aware alerts for ${district}`}
        right={<SourceTag meta={d?._meta} />}
      />

      {q.isError && <ErrorState error={q.error} onRetry={() => q.refetch()} />}

      {d?.alert.has_alert && (
        <div
          className={`flex items-start gap-3.5 rounded-xl2 border p-5 ${
            d.alert.severity === "HIGH"
              ? "border-[#F3D0D0] bg-[#FDF3F3]"
              : d.alert.severity === "MEDIUM"
                ? "border-[#F0DFB8] bg-[#FFFBF0]"
                : "border-forest-200 bg-forest-100/50"
          }`}
        >
          <AlertTriangle
            className={`mt-0.5 h-5 w-5 shrink-0 ${d.alert.severity === "HIGH" ? "text-[#B91C1C]" : "text-[#92400E]"}`}
          />
          <div>
            <div className="mb-1.5 flex items-center gap-2.5">
              <Badge value={d.alert.severity || "LOW"} />
              <span className="text-[11px] font-semibold uppercase tracking-[0.1em] text-ink-muted">
                {d.alert.alert_type.replace(/_/g, " ")}
              </span>
            </div>
            <p className="text-[15px] font-medium text-ink">{d.alert.message}</p>
            <p className="mt-1 text-[13.5px] text-ink-muted">{d.alert.action_required}</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-5">
        {q.isLoading && Array.from({ length: 5 }).map((_, i) => <CardSkeleton key={i} />)}
        {d && (
          <>
            <StatCard
              label="Temperature"
              value={`${d.temperature_avg.toFixed(1)}°`}
              unit="C"
              sub={`${d.temperature_min.toFixed(1)}° – ${d.temperature_max.toFixed(1)}° today`}
              accent="blue"
            />
            <StatCard label="Rainfall" value={d.rainfall_mm.toFixed(1)} unit="mm" sub={d.weather_condition} accent="blue" />
            <StatCard label="Humidity" value={`${d.humidity_percent.toFixed(0)}%`} sub="Relative humidity" accent="green" />
            <StatCard label="Wind speed" value={d.wind_speed_kmph.toFixed(1)} unit="km/h" accent="slate" />
            <StatCard label="Sunshine" value={d.sunshine_hours.toFixed(1)} unit="hrs" accent="gold" />
          </>
        )}
      </div>

      <Card className="p-7">
        <h3 className="text-[22px] text-forest-900">7-day forecast</h3>
        <p className="mt-1 text-sm text-ink-muted">Daily maximum / minimum temperature and rainfall probability</p>
        {q.isLoading && <Skeleton className="mt-5 h-40 w-full" />}
        {d && (
          <>
            <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">
              {d.forecast.map((f) => (
                <div key={f.date} className="rounded-xl2 border border-forest-200/70 bg-forest-100/40 p-3.5 text-center">
                  <div className="text-[12px] font-semibold text-forest-800">{dayLabel(f.date)}</div>
                  <div className="mt-2 flex items-center justify-center gap-1">
                    {f.rainfall_prob > 40 ? (
                      <CloudRain className="h-5 w-5 text-[#4A90C2]" />
                    ) : (
                      <Sun className="h-5 w-5 text-gold" />
                    )}
                  </div>
                  <div className="mt-2 font-serif text-[19px] text-forest-900">{Math.round(f.temp_max)}°</div>
                  <div className="text-[12px] text-ink-muted">{Math.round(f.temp_min)}° min</div>
                  <div className="mt-1.5 flex items-center justify-center gap-1 text-[12px] text-[#4A90C2]">
                    <Droplets className="h-3 w-3" /> {Math.round(f.rainfall_prob)}%
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-6 h-[210px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={d.forecast.map((f) => ({ ...f, day: dayLabel(f.date) }))} margin={{ top: 5, right: 8, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="tmax" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#40916C" stopOpacity={0.35} />
                      <stop offset="100%" stopColor="#40916C" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="#D8F3DC" vertical={false} />
                  <XAxis dataKey="day" {...AXIS} tickLine={false} axisLine={false} />
                  <YAxis {...AXIS} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #B7E4C7", fontSize: 13 }} />
                  <Area type="monotone" dataKey="temp_max" name="Max °C" stroke="#2D6A4F" strokeWidth={2} fill="url(#tmax)" />
                  <Area type="monotone" dataKey="temp_min" name="Min °C" stroke="#74C69D" strokeWidth={2} fill="none" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
      </Card>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1fr_1.4fr]">
        <Card className="p-7">
          <h3 className="text-[22px] text-forest-900">Field advisory</h3>
          <p className="mt-1 text-sm text-ink-muted">
            {season} · {stage.replace(/_/g, " ")} stage
          </p>
          {q.isLoading && <Skeleton className="mt-5 h-32 w-full" />}
          {d && (
            <div className="mt-5 space-y-3">
              <InfoBox title="Spraying window">{d.farming_advisory}</InfoBox>
              <div className="grid grid-cols-2 gap-3 text-[13px]">
                <div className="flex items-center gap-2 rounded-lg bg-forest-100/60 px-3 py-2.5">
                  <Thermometer className="h-4 w-4 text-forest-700" />
                  <span className="text-ink-soft">Condition: {d.weather_condition}</span>
                </div>
                <div className="flex items-center gap-2 rounded-lg bg-forest-100/60 px-3 py-2.5">
                  <Wind className="h-4 w-4 text-forest-700" />
                  <span className="text-ink-soft">{d.wind_speed_kmph.toFixed(1)} km/h wind</span>
                </div>
              </div>
              {!d.alert.has_alert && <InfoBox title="No active alerts">Conditions are within normal bands for this stage.</InfoBox>}
            </div>
          )}
        </Card>

        <Card className="p-7">
          <h3 className="text-[22px] text-forest-900">Historical baseline</h3>
          <p className="mt-1 text-sm text-ink-muted">Annual rainfall and mean temperature from the district weather record</p>
          {q.isLoading && <Skeleton className="mt-5 h-52 w-full" />}
          {d && d.historical.length > 0 && (
            <div className="mt-5 h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={d.historical} margin={{ top: 5, right: 8, left: -18, bottom: 0 }}>
                  <CartesianGrid stroke="#D8F3DC" vertical={false} />
                  <XAxis dataKey="year" {...AXIS} tickLine={false} axisLine={false} />
                  <YAxis yAxisId="l" {...AXIS} tickLine={false} axisLine={false} />
                  <YAxis yAxisId="r" orientation="right" {...AXIS} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #B7E4C7", fontSize: 13 }} />
                  <Line yAxisId="l" type="monotone" dataKey="rainfall" name="Rainfall (mm)" stroke="#4A90C2" strokeWidth={2} dot={false} />
                  <Line yAxisId="r" type="monotone" dataKey="temperature" name="Mean temp (°C)" stroke="#D4A017" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
