import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Leaf, RotateCcw } from "lucide-react";
import { api } from "../lib/api";
import { useAppState } from "../state/AppState";
import { Badge, Card, CardSkeleton, ErrorState, InfoBox, SectionTitle, Skeleton, SourceTag } from "../components/ui";

const AXIS = { stroke: "#6B7280", fontSize: 12 };

const STRESS_COPY: Record<string, string> = {
  Healthy: "Canopy vigour is at or above the benchmark for this stage.",
  Mild: "Slightly below the stage benchmark — worth a closer look this week.",
  Moderate: "Measurably below benchmark. Inspect the field for pest or nutrient stress.",
  Severe: "Well below benchmark. Treat this as urgent and confirm with your KVK.",
};

export default function CropHealth() {
  const { district, season, stage, setStage, meta } = useAppState();
  const [ndvi, setNdvi] = useState<number | null>(null);

  // Reset the manual NDVI override whenever the field context changes.
  useEffect(() => {
    setNdvi(null);
  }, [district, season, stage]);

  const q = useQuery({
    queryKey: ["crop-health", district, season, stage, ndvi],
    queryFn: () => api.cropHealth(district, season, stage, ndvi ?? undefined),
  });
  const matrixQ = useQuery({ queryKey: ["treatment-matrix"], queryFn: api.treatmentMatrix });

  const d = q.data;
  const baseline = d?.stage_baseline ?? meta?.stage_baselines?.[stage]?.median ?? 0;
  const sliderValue = ndvi ?? d?.ndvi ?? baseline;
  const deviation = sliderValue - baseline;

  const curve = d?.stage_curve ?? [];
  const currentIndex = curve.findIndex((c) => c.stage_key === (d?.growth_stage ?? stage));

  return (
    <div className="space-y-6">
      <SectionTitle
        title="Crop Health"
        subtitle="NDVI deviation from the stage benchmark, scored by a 4-class Gradient Boosting model"
        right={<SourceTag meta={d?._meta} />}
      />

      {q.isError && <ErrorState error={q.error} onRetry={() => q.refetch()} />}

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1fr_1.45fr]">
        {/* Controls */}
        <Card className="p-7">
          <h3 className="text-[22px] text-forest-900">Field reading</h3>
          <p className="mt-1 text-sm text-ink-muted">Pick a stage, then dial in the NDVI you measured</p>

          <div className="mt-5">
            <div className="label-cap mb-2">Growth stage</div>
            <div className="flex flex-wrap gap-2">
              {(meta?.growth_stages ?? []).map((s) => (
                <button
                  key={s}
                  onClick={() => setStage(s)}
                  className={`rounded-full px-3.5 py-1.5 text-[13px] font-medium transition ${
                    stage === s
                      ? "bg-forest-900 text-white"
                      : "border border-forest-200 bg-white text-ink-soft hover:bg-forest-100/70"
                  }`}
                >
                  {s.replace(/_/g, " ")}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-6">
            <div className="flex items-end justify-between">
              <div className="label-cap">NDVI reading</div>
              <div className="font-serif text-[26px] leading-none text-forest-900">{sliderValue.toFixed(3)}</div>
            </div>
            <input
              type="range"
              min={0}
              max={0.95}
              step={0.005}
              value={sliderValue}
              onChange={(e) => setNdvi(parseFloat(e.target.value))}
              className="mt-3 w-full accent-forest-700"
            />
            <div className="mt-1 flex justify-between text-[11px] text-ink-muted">
              <span>0.00 bare soil</span>
              <span>0.95 dense canopy</span>
            </div>
            {ndvi !== null && (
              <button
                onClick={() => setNdvi(null)}
                className="mt-3 inline-flex items-center gap-1.5 rounded-lg border border-forest-200 px-3 py-1.5 text-[12px] font-medium text-ink-soft transition hover:bg-forest-100/60"
              >
                <RotateCcw className="h-3.5 w-3.5" /> Reset to observed value
              </button>
            )}
          </div>

          <div className="mt-6">
            <InfoBox title={`Stage benchmark — ${(d?.growth_stage ?? stage).replace(/_/g, " ")}`}>
              Expected median NDVI <strong>{baseline.toFixed(3)}</strong>
              <br />
              Your deviation{" "}
              <strong className={deviation < 0 ? "text-[#B45309]" : "text-[#166534]"}>
                {deviation >= 0 ? "+" : ""}
                {deviation.toFixed(3)}
              </strong>
            </InfoBox>
          </div>
        </Card>

        {/* Verdict */}
        <div className="space-y-5">
          {q.isLoading && <CardSkeleton height="h-56" />}
          {d && (
            <Card className="p-7">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <Badge value={d.stress_level} className="px-4 py-1.5 text-[15px]" />
                <div className="flex items-center gap-2">
                  <span className="pill bg-forest-100 text-forest-800">NDVI source · {d.ndvi_source}</span>
                  <span className="pill bg-[#F3F4F6] text-ink-muted">
                    {(d.confidence * 100).toFixed(0)}% confidence
                  </span>
                </div>
              </div>

              <h3 className="mt-4 text-[24px] leading-tight text-forest-900">
                {d.district} · {d.growth_stage.replace(/_/g, " ")} · day {d.days_after_sowing}
              </h3>
              <p className="mt-2 text-[14.5px] leading-relaxed text-ink-soft">{STRESS_COPY[d.stress_level]}</p>

              <div className="mt-5 grid grid-cols-3 gap-3">
                <div className="rounded-xl2 bg-forest-100/60 p-4">
                  <div className="label-cap">NDVI</div>
                  <div className="mt-1 font-serif text-[24px] text-forest-900">{d.ndvi.toFixed(3)}</div>
                </div>
                <div className="rounded-xl2 bg-forest-100/60 p-4">
                  <div className="label-cap">Deviation</div>
                  <div
                    className={`mt-1 font-serif text-[24px] ${
                      d.ndvi_deviation < 0 ? "text-[#B45309]" : "text-forest-900"
                    }`}
                  >
                    {d.ndvi_deviation >= 0 ? "+" : ""}
                    {d.ndvi_deviation.toFixed(3)}
                  </div>
                </div>
                <div className="rounded-xl2 bg-forest-100/60 p-4">
                  <div className="label-cap">Disease risk</div>
                  <div className="mt-1.5">
                    <Badge value={d.disease_risk} />
                  </div>
                </div>
              </div>

              <div className="mt-5 rounded-xl2 border border-forest-200/80 bg-forest-100/40 p-5">
                <div className="mb-1.5 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.1em] text-forest-700">
                  <Leaf className="h-3.5 w-3.5" /> KVK treatment recommendation
                </div>
                <p className="text-[14.5px] leading-relaxed text-ink">{d.treatment}</p>
              </div>
            </Card>
          )}
        </div>
      </div>

      {/* NDVI curve */}
      <Card className="p-7">
        <h3 className="text-[22px] text-forest-900">NDVI growth curve</h3>
        <p className="mt-1 text-sm text-ink-muted">
          Stage baselines (Q25 · median · Q75) from 1,400 field records, with your reading plotted
        </p>
        {q.isLoading && <Skeleton className="mt-5 h-64 w-full" />}
        {d && (
          <div className="mt-5 h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={curve} margin={{ top: 10, right: 16, left: -16, bottom: 5 }}>
                <CartesianGrid stroke="#D8F3DC" vertical={false} />
                <XAxis dataKey="stage" {...AXIS} tickLine={false} axisLine={false} />
                <YAxis domain={[0, 1]} {...AXIS} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #B7E4C7", fontSize: 13 }} />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
                <Line type="monotone" dataKey="q75" name="Q75 (healthy)" stroke="#74C69D" strokeWidth={1.5} strokeDasharray="4 4" dot={false} />
                <Line type="monotone" dataKey="median" name="Median benchmark" stroke="#2D6A4F" strokeWidth={2.5} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="q25" name="Q25 (stressed)" stroke="#D4A017" strokeWidth={1.5} strokeDasharray="4 4" dot={false} />
                {currentIndex >= 0 && (
                  <ReferenceDot
                    x={curve[currentIndex]?.stage}
                    y={d.ndvi}
                    r={7}
                    fill="#D9534F"
                    stroke="#fff"
                    strokeWidth={2}
                    isFront
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card>

      {/* Treatment matrix */}
      <Card className="p-7">
        <h3 className="text-[22px] text-forest-900">KVK treatment protocol</h3>
        <p className="mt-1 text-sm text-ink-muted">
          The full (stress level × growth stage) matrix the agent serves — 7 stages, 4 stress levels
        </p>
        {matrixQ.isLoading && <Skeleton className="mt-5 h-48 w-full" />}
        {matrixQ.data && (
          <div className="mt-5 overflow-x-auto">
            <table className="w-full min-w-[820px] text-[13.5px]">
              <thead>
                <tr className="bg-forest-900 text-left text-white">
                  <th className="rounded-l-lg px-4 py-3 font-medium">Growth stage</th>
                  <th className="px-4 py-3 font-medium">Likely pests</th>
                  <th className="px-4 py-3 font-medium">Likely diseases</th>
                  <th className="rounded-r-lg px-4 py-3 font-medium">Stage advisory</th>
                </tr>
              </thead>
              <tbody>
                {matrixQ.data.stages.map((s, i) => (
                  <tr
                    key={s.stage}
                    className={`${i % 2 ? "bg-forest-100/45" : ""} ${
                      s.stage === (d?.growth_stage ?? stage) ? "ring-1 ring-inset ring-gold/50" : ""
                    }`}
                  >
                    <td className="px-4 py-3 font-medium text-forest-900">{s.label}</td>
                    <td className="px-4 py-3 text-ink-soft">{s.pests.join(", ") || "—"}</td>
                    <td className="px-4 py-3 text-ink-soft">{s.diseases.join(", ") || "—"}</td>
                    <td className="px-4 py-3 text-ink-muted">{s.advisory}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
