import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { TrendingDown, TrendingUp, Minus } from "lucide-react";
import { api } from "../lib/api";
import { useAppState } from "../state/AppState";
import { Badge, Card, CardSkeleton, ErrorState, InfoBox, SectionTitle, Skeleton, SourceTag, StatCard } from "../components/ui";

const AXIS = { stroke: "#6B7280", fontSize: 12 };

export default function Market() {
  const { district, season } = useAppState();
  const [variety, setVariety] = useState("");

  const q = useQuery({
    queryKey: ["market", district, season, variety],
    queryFn: () => api.market(district, season, variety || undefined),
  });
  const d = q.data;

  const TrendIcon = d?.trend_direction === "RISING" ? TrendingUp : d?.trend_direction === "FALLING" ? TrendingDown : Minus;

  return (
    <div className="space-y-6">
      <SectionTitle
        title="Market Prices"
        subtitle={`APMC price trend and sell/hold advisory · ${district}`}
        right={<SourceTag meta={d?._meta} />}
      />

      {q.isError && <ErrorState error={q.error} onRetry={() => q.refetch()} />}

      {d && (
        <div className="flex flex-wrap gap-2">
          {d.varieties.map((v) => (
            <button
              key={v}
              onClick={() => setVariety(v)}
              className={`rounded-full px-3.5 py-1.5 text-[13px] font-medium transition ${
                d.crop_variety === v
                  ? "bg-forest-900 text-white"
                  : "border border-forest-200 bg-white text-ink-soft hover:bg-forest-100/70"
              }`}
            >
              {v}
            </button>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {q.isLoading && Array.from({ length: 4 }).map((_, i) => <CardSkeleton key={i} />)}
        {d && (
          <>
            <StatCard
              label="Current price"
              value={`₹${d.current_price_per_quintal.toLocaleString("en-IN")}`}
              unit="/ qtl"
              sub={`${d.price_change_30d >= 0 ? "▲ +" : "▼ −"}₹${Math.abs(d.price_change_30d).toLocaleString("en-IN")} over 30 days`}
              accent="gold"
            />
            <StatCard
              label="MSP 2025–26"
              value={`₹${d.msp_price_per_quintal.toLocaleString("en-IN")}`}
              unit="/ qtl"
              sub={
                d.current_price_per_quintal >= d.msp_price_per_quintal
                  ? "Market is trading above MSP"
                  : "Market is trading below MSP"
              }
              accent="slate"
            />
            <StatCard
              label="30-day trend"
              badge={<Badge value={d.trend_direction} className="text-[13px]" />}
              sub={`Demand ${d.market_demand} · supply ${d.supply_status}`}
              accent="blue"
            />
            <StatCard
              label="Advisory"
              badge={<Badge value={d.sell_or_hold} className="text-[13px]" />}
              sub={`Best historical month: ${d.best_selling_month}`}
              accent={d.sell_or_hold === "SELL" ? "green" : "gold"}
            />
          </>
        )}
      </div>

      <Card className="p-7">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-[22px] text-forest-900">Price trend — {d?.crop_variety ?? "…"}</h3>
            <p className="mt-1 text-sm text-ink-muted">Last 30 recorded market days against the MSP floor</p>
          </div>
          {d && (
            <span className="inline-flex items-center gap-1.5 text-[13px] font-semibold text-ink-soft">
              <TrendIcon className="h-4 w-4" /> {d.trend_direction}
            </span>
          )}
        </div>
        {q.isLoading && <Skeleton className="mt-5 h-64 w-full" />}
        {d && (
          <div className="mt-5 h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={d.price_trend_30d} margin={{ top: 5, right: 10, left: -6, bottom: 0 }}>
                <defs>
                  <linearGradient id="price" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#D4A017" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#D4A017" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#D8F3DC" vertical={false} />
                <XAxis dataKey="date" {...AXIS} tickLine={false} axisLine={false} minTickGap={28} />
                <YAxis domain={["auto", "auto"]} {...AXIS} tickLine={false} axisLine={false} />
                <Tooltip
                  contentStyle={{ borderRadius: 12, border: "1px solid #B7E4C7", fontSize: 13 }}
                  formatter={(v: number) => [`₹${v.toLocaleString("en-IN")}`, "Price"]}
                />
                <ReferenceLine
                  y={d.msp_price_per_quintal}
                  stroke="#2D6A4F"
                  strokeDasharray="5 5"
                  label={{ value: `MSP ₹${d.msp_price_per_quintal}`, position: "insideTopLeft", fontSize: 11, fill: "#2D6A4F" }}
                />
                <Area type="monotone" dataKey="price" stroke="#D4A017" strokeWidth={2.5} fill="url(#price)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1fr_1fr]">
        <Card className="p-7">
          <h3 className="text-[22px] text-forest-900">Why this call</h3>
          {q.isLoading && <Skeleton className="mt-5 h-24 w-full" />}
          {d && (
            <div className="mt-4 space-y-3">
              <InfoBox title={`${d.sell_or_hold} — ${d.crop_variety}`}>{d.reasoning}</InfoBox>
              <InfoBox tone="warning" title="Note">
                {d._meta.notes}
              </InfoBox>
            </div>
          )}
        </Card>

        <Card className="p-7">
          <h3 className="text-[22px] text-forest-900">Variety comparison</h3>
          <p className="mt-1 text-sm text-ink-muted">Latest recorded price per variety</p>
          {q.isLoading && <Skeleton className="mt-5 h-40 w-full" />}
          {d && (
            <table className="mt-5 w-full text-[14px]">
              <thead>
                <tr className="bg-forest-900 text-left text-white">
                  <th className="rounded-l-lg px-4 py-2.5 font-medium">Variety</th>
                  <th className="px-4 py-2.5 font-medium">Price</th>
                  <th className="px-4 py-2.5 font-medium">Demand</th>
                  <th className="rounded-r-lg px-4 py-2.5 font-medium">Supply</th>
                </tr>
              </thead>
              <tbody>
                {d.variety_table.map((v, i) => (
                  <tr key={v.variety} className={`${i % 2 ? "bg-forest-100/50" : ""} ${v.variety === d.crop_variety ? "font-semibold" : ""}`}>
                    <td className="px-4 py-2.5 text-forest-900">{v.variety}</td>
                    <td className="px-4 py-2.5 text-ink-soft">₹{v.price.toLocaleString("en-IN")}</td>
                    <td className="px-4 py-2.5 text-ink-muted">{v.demand}</td>
                    <td className="px-4 py-2.5 text-ink-muted">{v.supply}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
      </div>
    </div>
  );
}
