import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Target, CheckCircle2, AlertTriangle } from "lucide-react";
import { api } from "../lib/api";
import { Card, ErrorState, InfoBox, SectionTitle, Skeleton } from "../components/ui";

const AXIS = { stroke: "#6B7280", fontSize: 11 };
const PRECISION = "#2D6A4F"; // forest green
const RECALL = "#D4A017"; // gold

// Sequential forest-green ramp for the confusion matrix (light → dark).
function cellStyle(v: number) {
  // v is 0..1 (row-normalised). Interpolate background alpha; flip text for dark cells.
  const bg = `rgba(45, 106, 79, ${0.08 + v * 0.9})`;
  const color = v > 0.55 ? "#F8FAF9" : "#1F2937";
  return { backgroundColor: bg, color };
}

export default function ModelPerformance() {
  const q = useQuery({ queryKey: ["model-metrics"], queryFn: api.modelMetrics, staleTime: 30_000 });
  const histQ = useQuery({ queryKey: ["model-history"], queryFn: api.modelHistory, staleTime: 60_000 });
  const d = q.data;
  const hist = histQ.data;

  return (
    <div className="space-y-6">
      <SectionTitle
        title="Model Performance"
        subtitle="Held-out evaluation of the YOLOv8 paddy disease classifier"
      />

      {q.isError && <ErrorState error={q.error} onRetry={() => q.refetch()} />}
      {q.isLoading && <Skeleton className="h-40 w-full" />}

      {d && !d.trained && (
        <InfoBox tone="warning" title="Model not trained yet">
          {d.note}
          <div className="mt-2 text-[13px]">
            <code>python scripts/download_paddy_dataset.py</code> →{" "}
            <code>python scripts/train_yolo_classifier.py</code> (writes metrics automatically).
          </div>
        </InfoBox>
      )}

      {d && d.trained && (
        <>
          {/* Accuracy hero + target */}
          <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1fr_1.6fr]">
            <Card className="p-7">
              <div className="label-cap flex items-center gap-1.5">
                <Target className="h-3.5 w-3.5" /> Top-1 accuracy
              </div>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="font-serif text-[52px] leading-none text-forest-900">
                  {((d.top1_accuracy ?? 0) * 100).toFixed(1)}
                </span>
                <span className="text-[18px] text-ink-muted">%</span>
              </div>

              {/* progress vs 75% target */}
              <div className="relative mt-5 h-3 w-full rounded-full bg-forest-100">
                <div
                  className={`h-full rounded-full ${d.meets_target ? "bg-forest-600" : "bg-[#D4A017]"}`}
                  style={{ width: `${Math.min(100, (d.top1_accuracy ?? 0) * 100)}%` }}
                />
                <div
                  className="absolute -top-1 h-5 w-0.5 bg-[#D9534F]"
                  style={{ left: `${(d.target_accuracy ?? 0.75) * 100}%` }}
                  title={`Target ${(d.target_accuracy ?? 0.75) * 100}%`}
                />
              </div>
              <div className="mt-2 flex items-center justify-between text-[12px] text-ink-muted">
                <span>0%</span>
                <span>target {(d.target_accuracy ?? 0.75) * 100}%</span>
                <span>100%</span>
              </div>

              <div className="mt-4">
                {d.meets_target ? (
                  <span className="pill bg-[#DCFCE7] text-[#166534]">
                    <CheckCircle2 className="h-3.5 w-3.5" /> Meets the 75% target
                  </span>
                ) : (
                  <span className="pill bg-[#FEF9C3] text-[#854D0E]">
                    <AlertTriangle className="h-3.5 w-3.5" /> Below 75% — train more epochs
                  </span>
                )}
              </div>

              <div className="mt-5 space-y-1 text-[12.5px] text-ink-muted">
                <div>Model: {d.model}</div>
                <div>
                  {d.num_classes} classes · {d.train_images?.toLocaleString()} train /{" "}
                  {d.val_images?.toLocaleString()} val images
                </div>
                {d.evaluated_at && <div>Evaluated {new Date(d.evaluated_at).toLocaleString()}</div>}
              </div>
            </Card>

            {/* Per-class precision / recall */}
            <Card className="p-7">
              <h3 className="text-[20px] text-forest-900">Per-class precision &amp; recall</h3>
              <p className="mt-1 text-sm text-ink-muted">How reliably each disease is identified</p>
              <div className="mt-4 h-[320px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={d.per_class ?? []} margin={{ top: 8, right: 12, left: -18, bottom: 40 }} barGap={2}>
                    <CartesianGrid stroke="#E5E7EB" vertical={false} />
                    <XAxis
                      dataKey="class"
                      {...AXIS}
                      tickLine={false}
                      axisLine={false}
                      angle={-35}
                      textAnchor="end"
                      interval={0}
                      height={60}
                    />
                    <YAxis domain={[0, 1]} {...AXIS} tickLine={false} axisLine={false} />
                    <Tooltip
                      contentStyle={{ borderRadius: 12, border: "1px solid #B7E4C7", fontSize: 13 }}
                      formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
                    />
                    <Legend wrapperStyle={{ fontSize: 12, paddingTop: 6 }} />
                    <Bar dataKey="precision" name="Precision" fill={PRECISION} radius={[3, 3, 0, 0]} />
                    <Bar dataKey="recall" name="Recall" fill={RECALL} radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>

          {/* Confusion matrix */}
          {d.confusion_matrix && d.class_names && (
            <Card className="p-7">
              <h3 className="text-[20px] text-forest-900">Confusion matrix</h3>
              <p className="mt-1 text-sm text-ink-muted">
                Rows = true class, columns = predicted. Shaded by row share; a strong diagonal is good.
              </p>
              <div className="mt-5 overflow-x-auto">
                <table className="border-separate border-spacing-0.5 text-[11px]">
                  <thead>
                    <tr>
                      <th className="p-1"></th>
                      {d.class_names.map((c) => (
                        <th key={c} className="p-1 align-bottom">
                          <div className="mx-auto h-20 w-6 origin-bottom-left -rotate-45 whitespace-nowrap text-left text-ink-soft">
                            {c}
                          </div>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {d.confusion_matrix.map((row, i) => {
                      const rowSum = row.reduce((a, b) => a + b, 0) || 1;
                      return (
                        <tr key={i}>
                          <td className="whitespace-nowrap py-1 pr-2 text-right font-medium text-ink-soft">
                            {d.class_names![i]}
                          </td>
                          {row.map((cell, j) => (
                            <td
                              key={j}
                              className="h-9 w-9 rounded text-center align-middle font-medium tabular-nums"
                              style={cellStyle(cell / rowSum)}
                              title={`true ${d.class_names![i]} → pred ${d.class_names![j]}: ${cell}`}
                            >
                              {cell || ""}
                            </td>
                          ))}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </>
      )}

      {/* Version history + revision log */}
      {hist && (hist.versions?.length > 0 || hist.revisions?.length > 0) && (
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
          {hist.versions?.length > 0 && (
            <Card className="p-7">
              <h3 className="text-[20px] text-forest-900">Model versions</h3>
              <p className="mt-1 text-sm text-ink-muted">How the image model was revised toward an honest metric</p>
              <div className="mt-4 space-y-3">
                {hist.versions.map((v) => (
                  <div key={v.version} className="rounded-xl2 border border-forest-200/70 bg-forest-100/30 p-4">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-forest-900">
                        {v.version} · {(v.accuracy * 100).toFixed(1)}%
                      </span>
                      <span className={`pill ${v.trustworthy ? "bg-[#DCFCE7] text-[#166534]" : "bg-[#FEF9C3] text-[#854D0E]"}`}>
                        {v.trustworthy ? "trustworthy" : "not trustworthy"}
                      </span>
                    </div>
                    <div className="mt-1 text-[12px] text-ink-muted">
                      {v.classes} classes · {v.train_images.toLocaleString()} train / {v.val_images.toLocaleString()} val
                    </div>
                    <p className="mt-1.5 text-[13px] leading-relaxed text-ink-soft">{v.note}</p>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {hist.revisions?.length > 0 && (
            <Card className="p-7">
              <h3 className="text-[20px] text-forest-900">What we revised</h3>
              <p className="mt-1 text-sm text-ink-muted">Flaws we caught and fixed</p>
              <ul className="mt-4 space-y-3">
                {hist.revisions.map((r) => (
                  <li key={r.title} className="flex gap-3">
                    <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-forest-600" />
                    <div>
                      <div className="text-[14px] font-medium text-forest-900">{r.title}</div>
                      <p className="text-[13px] leading-relaxed text-ink-soft">{r.detail}</p>
                    </div>
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
