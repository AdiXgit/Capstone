import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Upload, ScanEye, Loader2, X, Camera, Aperture, Flame } from "lucide-react";
import { api, DetectResponse, DiseasePrediction } from "../lib/api";
import { Badge, Card, InfoBox } from "./ui";

const SEVERITY_BADGE: Record<string, string> = {
  HIGH: "HIGH",
  MEDIUM: "MEDIUM",
  LOW: "LOW",
  NONE: "Healthy",
  UNKNOWN: "INFO",
};

/** Ranked class probabilities. `muted` greys them out when no diagnosis is given. */
function PredictionBars({ predictions, muted }: { predictions: DiseasePrediction[]; muted?: boolean }) {
  return (
    <div>
      <div className="label-cap mb-2">{muted ? "Ranking (not a diagnosis)" : "Top predictions"}</div>
      <div className="space-y-2">
        {predictions.map((p) => (
          <div key={p.raw_label} className="flex items-center gap-3">
            <div className="w-40 shrink-0 truncate text-[13px] text-ink-soft" title={p.disease}>
              {p.disease}
            </div>
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-forest-100">
              <div
                className={`h-full rounded-full ${muted ? "bg-ink-muted/40" : "bg-forest-600"}`}
                style={{ width: `${Math.round(p.confidence * 100)}%` }}
              />
            </div>
            <div className="w-12 text-right text-[12px] text-ink-muted">{(p.confidence * 100).toFixed(0)}%</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Shows the three statistics the confidence gate actually tests, against the
 *  threshold each one has to clear, so the decision is auditable rather than magic. */
function CertaintyReadout({ c }: { c: NonNullable<DetectResponse["certainty"]> }) {
  const rows = [
    { k: "Top-1 probability", v: c.top1, min: c.thresholds.minimum.top1, higher: true },
    { k: "Margin over runner-up", v: c.margin, min: c.thresholds.minimum.margin, higher: true },
    { k: "Spread (entropy)", v: c.entropy, min: c.thresholds.minimum.entropy, higher: false },
  ];
  return (
    <div>
      <div className="label-cap mb-2">Why this confidence level</div>
      <div className="space-y-1">
        {rows.map((r) => {
          const pass = r.higher ? r.v >= r.min : r.v <= r.min;
          return (
            <div key={r.k} className="flex items-center justify-between gap-3 text-[12.5px]">
              <span className="text-ink-soft">{r.k}</span>
              <span className="flex items-center gap-2 tabular-nums">
                <span className="text-ink-muted">
                  {r.v.toFixed(2)} {r.higher ? "≥" : "≤"} {r.min.toFixed(2)}
                </span>
                <span className={pass ? "text-[#166534]" : "text-[#9A3412]"}>{pass ? "✓" : "✗"}</span>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function DiseaseDetect() {
  const inputRef = useRef<HTMLInputElement>(null);
  const captureRef = useRef<HTMLInputElement>(null); // mobile camera fallback
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<DetectResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [camera, setCamera] = useState(false);
  const [overlay, setOverlay] = useState<string | null>(null);
  const [explaining, setExplaining] = useState(false);

  const statusQ = useQuery({ queryKey: ["detect-status"], queryFn: api.detectStatus, staleTime: 60_000 });

  useEffect(() => () => stopCamera(), []); // cleanup on unmount

  async function explain() {
    if (!file) return;
    setExplaining(true);
    try {
      const r = await api.explainDisease(file);
      if (r.overlay_base64) setOverlay(r.overlay_base64);
    } catch {
      /* ignore — explain is best-effort */
    } finally {
      setExplaining(false);
    }
  }

  function pick(f: File | null) {
    setResult(null);
    setErr(null);
    setOverlay(null);
    setFile(f);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(f ? URL.createObjectURL(f) : null);
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCamera(false);
  }

  async function startCamera() {
    setErr(null);
    setResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });
      streamRef.current = stream;
      setCamera(true);
      // wait a tick for the video element to mount
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => {});
        }
      }, 50);
    } catch {
      // Desktop denied or no webcam → fall back to the OS camera picker (mobile).
      captureRef.current?.click();
    }
  }

  function capture() {
    const video = videoRef.current;
    if (!video) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const f = new File([blob], `scan-${Date.now()}.jpg`, { type: "image/jpeg" });
      pick(f);
      stopCamera();
      analyze(f);
    }, "image/jpeg", 0.92);
  }

  async function analyze(f?: File) {
    const target = f ?? file;
    if (!target) return;
    setBusy(true);
    setErr(null);
    try {
      const r = await api.detectDisease(target);
      if (r.error) setErr(r.detail || r.error);
      else setResult(r);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Detection failed");
    } finally {
      setBusy(false);
    }
  }

  const status = statusQ.data;

  return (
    <Card className="p-7">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="flex items-center gap-2 text-[22px] text-forest-900">
            <ScanEye className="h-5 w-5 text-forest-700" /> Scan a leaf (YOLOv8)
          </h3>
          <p className="mt-1 text-sm text-ink-muted">
            Point your camera at a paddy leaf or upload a photo — the model names the disease and its KVK treatment
          </p>
        </div>
        {status && (
          <span
            className={`pill ${status.is_finetuned ? "bg-[#DCFCE7] text-[#166534]" : "bg-[#FEF9C3] text-[#854D0E]"}`}
            title={status.note}
          >
            {status.available
              ? status.is_finetuned
                ? "fine-tuned model"
                : "base model (not fine-tuned)"
              : "vision stack not installed"}
          </span>
        )}
      </div>

      <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-[1fr_1.3fr]">
        {/* Capture / preview */}
        <div>
          <input ref={inputRef} type="file" accept="image/*" className="hidden"
            onChange={(e) => pick(e.target.files?.[0] ?? null)} />
          <input ref={captureRef} type="file" accept="image/*" capture="environment" className="hidden"
            onChange={(e) => { const f = e.target.files?.[0] ?? null; pick(f); if (f) analyze(f); }} />

          {camera ? (
            <div className="relative overflow-hidden rounded-xl2 bg-black">
              <video ref={videoRef} playsInline muted className="h-56 w-full object-cover" />
              <button onClick={stopCamera}
                className="absolute right-2 top-2 rounded-full bg-black/50 p-1.5 text-white transition hover:bg-black/70">
                <X className="h-4 w-4" />
              </button>
              <button onClick={capture}
                className="absolute bottom-3 left-1/2 flex -translate-x-1/2 items-center gap-2 rounded-full bg-white px-5 py-2.5 text-sm font-semibold text-forest-900 shadow-lg transition hover:bg-forest-100">
                <Aperture className="h-4 w-4" /> Capture
              </button>
            </div>
          ) : !preview ? (
            <div className="flex h-56 w-full flex-col items-center justify-center gap-3 rounded-xl2 border-2 border-dashed border-forest-200 bg-forest-100/40">
              <button onClick={startCamera}
                className="flex flex-col items-center gap-1.5 text-forest-800 transition hover:scale-105">
                <span className="flex h-14 w-14 items-center justify-center rounded-full bg-forest-900 text-white shadow-md">
                  <Camera className="h-7 w-7" />
                </span>
                <span className="text-sm font-semibold">Scan plant</span>
              </button>
              <button onClick={() => inputRef.current?.click()}
                className="inline-flex items-center gap-1.5 text-[12px] text-ink-muted transition hover:text-forest-700">
                <Upload className="h-3.5 w-3.5" /> or upload a photo
              </button>
            </div>
          ) : (
            <div className="relative">
              <img src={preview} alt="leaf" className="h-56 w-full rounded-xl2 object-cover" />
              <button onClick={() => pick(null)}
                className="absolute right-2 top-2 rounded-full bg-black/50 p-1.5 text-white transition hover:bg-black/70">
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          {!camera && (
            <div className="mt-4 flex gap-2">
              <button onClick={startCamera}
                className="inline-flex flex-1 items-center justify-center gap-2 rounded-lg border border-forest-200 px-4 py-2.5 text-sm font-semibold text-forest-800 transition hover:bg-forest-100/70">
                <Camera className="h-4 w-4" /> Camera
              </button>
              <button onClick={() => analyze()} disabled={!file || busy}
                className="inline-flex flex-[1.4] items-center justify-center gap-2 rounded-lg bg-forest-900 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-forest-800 disabled:cursor-not-allowed disabled:opacity-50">
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <ScanEye className="h-4 w-4" />}
                {busy ? "Analyzing…" : "Analyze"}
              </button>
            </div>
          )}

          {err && (
            <div className="mt-3">
              <InfoBox tone="alert" title="Detection unavailable">{err}</InfoBox>
            </div>
          )}
        </div>

        {/* Result */}
        <div>
          {!result && !err && (
            <div className="flex h-full min-h-[224px] items-center justify-center rounded-xl2 border border-forest-200/70 bg-forest-100/30 p-6 text-center text-sm text-ink-muted">
              Scan or upload a leaf to see the predicted disease, confidence and treatment.
            </div>
          )}

          {result && result.is_leaf === false && (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-serif text-[22px] leading-none text-[#9A3412]">Not a paddy plant</span>
                <span className="pill bg-[#FFEDD5] text-[#9A3412]">
                  plant score {((result.plant_score ?? 0) * 100).toFixed(0)}%
                </span>
              </div>
              <InfoBox tone="warning" title="Rejected before classification">
                The disease model was <strong>not run</strong> on this image.
              </InfoBox>
              {result.reject_reasons && result.reject_reasons.length > 0 && (
                <div>
                  <div className="label-cap mb-2">Why it was rejected</div>
                  <ul className="space-y-1.5">
                    {result.reject_reasons.map((r) => (
                      <li key={r} className="flex gap-2 text-[13px] leading-relaxed text-ink-soft">
                        <span className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-[#9A3412]" />
                        <span>{r}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              <p className="text-[12.5px] leading-relaxed text-ink-muted">
                A four-cue botanical gate (foliage hue, exposure, colour coherence and leaf texture) runs before the
                classifier, so non-plant images can't be given a false disease label.
              </p>
            </div>
          )}

          {result && result.is_leaf !== false && result.confidence_level === "ABSTAIN" && (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-serif text-[22px] leading-none text-[#854D0E]">Inconclusive</span>
                <span className="pill bg-[#FEF9C3] text-[#854D0E]">no diagnosis given</span>
              </div>
              <InfoBox tone="warning" title="The model isn't confident enough to name a disease">
                {result.confidence_note}
              </InfoBox>
              {result.certainty && <CertaintyReadout c={result.certainty} />}
              {result.predictions && result.predictions.length > 0 && (
                <PredictionBars predictions={result.predictions} muted />
              )}
              <p className="text-[12.5px] leading-relaxed text-ink-muted">
                Ranking shown for transparency only — no treatment is recommended on an uncertain call. Retake with a
                single leaf filling the frame in even daylight.
              </p>
            </div>
          )}

          {result && result.is_leaf !== false && result.confidence_level !== "ABSTAIN" && (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-serif text-[24px] leading-none text-forest-900">{result.predicted_disease}</span>
                <Badge value={SEVERITY_BADGE[result.severity ?? "UNKNOWN"] ?? "INFO"} />
                <span className="pill bg-[#F3F4F6] text-ink-muted">
                  {((result.confidence ?? 0) * 100).toFixed(1)}% confidence
                </span>
                {result.confidence_level === "PROVISIONAL" ? (
                  <span className="pill bg-[#FEF9C3] text-[#854D0E]" title={result.confidence_note}>
                    provisional
                  </span>
                ) : (
                  <span className="pill bg-[#DCFCE7] text-[#166534]" title={result.confidence_note}>
                    confirmed
                  </span>
                )}
                {result.plant_score !== undefined && (
                  <span
                    className={`pill ${result.low_vegetation ? "bg-[#FEF9C3] text-[#854D0E]" : "bg-[#DCFCE7] text-[#166534]"}`}
                    title={result.vegetation_note}
                  >
                    plant {(result.plant_score * 100).toFixed(0)}%
                  </span>
                )}
              </div>

              {result.confidence_level === "PROVISIONAL" && (
                <InfoBox tone="warning" title="Provisional — confirm before spraying">
                  {result.confidence_note}
                  {result.reported_severity && result.reported_severity !== result.severity && (
                    <>
                      {" "}Severity is shown one notch below {result.reported_severity} because the call is not certain.
                    </>
                  )}
                </InfoBox>
              )}

              {result.low_vegetation && (
                <InfoBox tone="warning">
                  Low foliage in frame — the prediction may be unreliable. Retake with the leaf filling the frame in
                  good light.
                </InfoBox>
              )}

              {result.is_paddy_condition === false && (
                <InfoBox tone="warning">
                  This prediction is from the base (non-fine-tuned) model, so the label is generic. Train the model
                  (Model Performance page) for paddy-specific accuracy.
                </InfoBox>
              )}

              <InfoBox title="KVK treatment recommendation">{result.treatment}</InfoBox>

              {result.predictions && result.predictions.length > 1 && (
                <PredictionBars predictions={result.predictions} />
              )}

              {result.certainty && <CertaintyReadout c={result.certainty} />}

              {/* Explainability: occlusion saliency */}
              <div>
                {!overlay ? (
                  <button
                    onClick={explain}
                    disabled={explaining}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-forest-200 px-3 py-1.5 text-[12.5px] font-medium text-forest-800 transition hover:bg-forest-100/70 disabled:opacity-50"
                  >
                    {explaining ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Flame className="h-3.5 w-3.5" />}
                    {explaining ? "Computing heatmap…" : "Explain: where did the model look?"}
                  </button>
                ) : (
                  <div>
                    <div className="label-cap mb-1.5">Saliency — regions driving the prediction</div>
                    <img src={overlay} alt="saliency heatmap" className="w-full max-w-[280px] rounded-xl2" />
                    <p className="mt-1 text-[11.5px] text-ink-muted">
                      Red = covering it most reduces the model's confidence (occlusion saliency).
                    </p>
                  </div>
                )}
              </div>

              {result.model && (
                <div className="text-[12px] text-ink-muted">
                  Model: {result.model.path} · {result.latency_ms?.toFixed(0)}ms · {result.model.note}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}
