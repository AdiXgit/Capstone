import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { TrendingDown, TrendingUp, Minus } from "lucide-react";
import { api } from "../lib/api";
import { useAppState } from "../state/AppState";
import { Card, CardSkeleton, ErrorState, InfoBox, SectionTitle, Skeleton, SourceTag, StatCard } from "../components/ui";

const AXIS = { stroke: "#6B7280", fontSize: 12 };

function TrendPill({ slope, positiveLabel, negativeLabel }: { slope: number; positiveLabel: string; negativeLabel: string }) {
  if (Math.abs(slope) < 0.001)
    return (
      <span className="inline-flex items-center gap-1 text-[13px] font-semibold text-ink-muted">
        <Minus className="h-3.5 w-3.5" /> Stable
      </span>
    );
  const up = slope > 0;
  return (
    <span className={`inline-flex items-center gap-1 text-[13px] font-semibold ${up ? "text-[#16A34A]" : "text-[#DC2626]"}`}>
      {up ? <TrendingUp className="h-3.5 w-3.5" /> : <TrendingDown className="h-3.5 w-3.5" />}
      {up ? positiveLabel : negativeLabel}
    </span>
  );
}

export default function Soil() {
  const { district, season } = useAppState();
  const q = useQuery({
    queryKey: ["soil", district, season],
    queryFn: () => api.soil(district, season),
  });
  const d = q.data;

  const comparison = d
    ? [...d.district_comparison].sort((a, b) => b.nitrogen - a.nitrogen).slice(0, 12)
    : [];

  return (
    <div className="space-y-6">
      <SectionTitle
        title="Soil Intelligence"
        subtitle={`NPK profile, KVK fertiliser plan and 15-year trend for ${district}`}
        right={<SourceTag meta={d?._meta} />}
      />

      {q.isError && <ErrorState error={q.error} onRetry={() => q.refetch()} />}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-5">
        {q.isLoading && Array.from({ length: 5 }).map((_, i) => <CardSkeleton key={i} />)}
        {d && (
          <>
            <StatCard label="Nitrogen (N)" value={d.nitrogen.toFixed(1)} unit="kg/ha" accent="green" />
            <StatCard label="Phosphorus (P)" value={d.phosphorus.toFixed(1)} unit="kg/ha" accent="green" />
            <StatCard label="Potassium (K)" value={d.potassium.toFixed(1)} unit="kg/ha" accent="green" />
            <StatCard label="Soil pH" value={d.ph.toFixed(2)} sub={`Organic carbon ${d.organic_carbon}%`} accent="blue" />
            <StatCard
              label="Soil health score"
              value={d.soil_health_score.toFixed(1)}
              unit="/ 10"
              sub={`${d.soil_quality} quality · avg yield ${d.avg_yield_kg_per_ha.toLocaleString("en-IN")} kg/ha`}
              accent="gold"
            />
          </>
        )}
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <Card className="p-7">
          <h3 className="text-[22px] text-forest-900">Fertiliser recommendation</h3>
          <p className="mt-1 text-sm text-ink-muted">Karnataka KVK guidelines · {season}</p>
          {q.isLoading && <Skeleton className="mt-5 h-40 w-full" />}
          {d && (
            <>
              <table className="mt-5 w-full text-[14px]">
                <thead>
                  <tr className="bg-forest-900 text-left text-white">
                    <th className="rounded-l-lg px-4 py-2.5 font-medium">Fertiliser</th>
                    <th className="px-4 py-2.5 font-medium">Dose</th>
                    <th className="rounded-r-lg px-4 py-2.5 font-medium">Source</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["Urea", d.urea_kg_per_acre],
                    ["DAP", d.dap_kg_per_acre],
                    ["Potash (MOP)", d.potash_kg_per_acre],
                  ].map(([name, dose], i) => (
                    <tr key={name as string} className={i % 2 ? "bg-forest-100/50" : ""}>
                      <td className="px-4 py-2.5 text-ink-soft">{name}</td>
                      <td className="px-4 py-2.5 font-semibold text-forest-900">{dose} kg/acre</td>
                      <td className="px-4 py-2.5 text-ink-muted">Karnataka KVK</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="mt-4">
                <InfoBox title={`Application timing — ${season}`}>{d.timing_advice}</InfoBox>
              </div>
            </>
          )}
        </Card>

        <Card className="p-7">
          <h3 className="text-[22px] text-forest-900">15-year soil trend</h3>
          <p className="mt-1 text-sm text-ink-muted">Linear regression on district averages</p>
          {q.isLoading && <Skeleton className="mt-5 h-40 w-full" />}
          {d && (
            <>
              <div className="mt-4 grid grid-cols-2 gap-4">
                <div className="rounded-xl2 border border-forest-200/70 bg-forest-100/40 p-4">
                  <div className="label-cap">Soil pH</div>
                  <div className="mt-1 font-serif text-[26px] text-forest-900">{d.ph_last_value}</div>
                  <TrendPill slope={d.ph_slope} positiveLabel="Rising" negativeLabel="Acidifying" />
                  <div className="mt-1 text-[12px] text-ink-muted">{d.ph_slope}/yr</div>
                </div>
                <div className="rounded-xl2 border border-forest-200/70 bg-forest-100/40 p-4">
                  <div className="label-cap">Organic carbon</div>
                  <div className="mt-1 font-serif text-[26px] text-forest-900">{d.oc_last_value}%</div>
                  <TrendPill slope={d.oc_slope} positiveLabel="Improving" negativeLabel="Declining" />
                  <div className="mt-1 text-[12px] text-ink-muted">{d.oc_slope}/yr</div>
                </div>
              </div>
              <div className="mt-5 h-[190px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={d.trend_series} margin={{ top: 5, right: 8, left: -18, bottom: 0 }}>
                    <CartesianGrid stroke="#D8F3DC" vertical={false} />
                    <XAxis dataKey="year" {...AXIS} tickLine={false} axisLine={false} />
                    <YAxis yAxisId="l" {...AXIS} tickLine={false} axisLine={false} domain={["auto", "auto"]} />
                    <YAxis yAxisId="r" orientation="right" {...AXIS} tickLine={false} axisLine={false} />
                    <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #B7E4C7", fontSize: 13 }} />
                    <Line yAxisId="l" type="monotone" dataKey="ph" name="pH" stroke="#2D6A4F" strokeWidth={2} dot={false} />
                    <Line
                      yAxisId="r"
                      type="monotone"
                      dataKey="organic_carbon"
                      name="Organic carbon %"
                      stroke="#D4A017"
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-3">
                <InfoBox>{d.interpretation}</InfoBox>
              </div>
            </>
          )}
        </Card>
      </div>

      <Card className="p-7">
        <h3 className="text-[22px] text-forest-900">District nitrogen comparison</h3>
        <p className="mt-1 text-sm text-ink-muted">Top 12 districts by available soil nitrogen · {district} highlighted</p>
        {q.isLoading && <Skeleton className="mt-5 h-64 w-full" />}
        {d && (
          <div className="mt-5 h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparison} margin={{ top: 5, right: 10, left: -14, bottom: 55 }}>
                <CartesianGrid stroke="#D8F3DC" vertical={false} />
                <XAxis dataKey="district" {...AXIS} interval={0} angle={-38} textAnchor="end" tickLine={false} axisLine={false} />
                <YAxis {...AXIS} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #B7E4C7", fontSize: 13 }} cursor={{ fill: "#D8F3DC66" }} />
                <Bar dataKey="nitrogen" name="N (kg/ha)" radius={[6, 6, 0, 0]}>
                  {comparison.map((c) => (
                    <Cell key={c.district} fill={c.district === district ? "#D4A017" : "#40916C"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card>
    </div>
  );
}
