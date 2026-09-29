// Resolve the gateway base URL at runtime so the same deployed build can point at
// different backends (e.g. a Cloudflare tunnel for a phone demo):
//   1. ?api=<url> in the URL (also remembered), 2. saved value, 3. build-time env, 4. same origin.
function resolveBase(): string {
  try {
    const q = new URL(window.location.href).searchParams.get("api");
    if (q) {
      localStorage.setItem("krishi_api", q);
      return q.replace(/\/$/, "");
    }
    const saved = localStorage.getItem("krishi_api");
    if (saved) return saved.replace(/\/$/, "");
  } catch {
    /* ignore */
  }
  return (import.meta.env.VITE_GATEWAY_URL ?? "").replace(/\/$/, "");
}

const BASE = resolveBase();

export function getApiBase() {
  return BASE || window.location.origin;
}

export interface Meta {
  source: "GRPC_AGENT" | "GATEWAY_DATASET" | "LIVE_API" | "YOLO_MODEL";
  agent: string;
  latency_ms: number;
  notes: string;
  timestamp: string;
}

export interface MetaResponse {
  districts: string[];
  seasons: string[];
  growth_stages: string[];
  stage_baselines: Record<string, { q25: number; median: number; q75: number; das: number[] }>;
  stage_advisories: Record<string, string>;
}

export interface SoilResponse {
  district: string;
  season: string;
  nitrogen: number;
  phosphorus: number;
  potassium: number;
  ph: number;
  organic_carbon: number;
  soil_health_score: number;
  soil_quality: string;
  urea_kg_per_acre: number;
  dap_kg_per_acre: number;
  potash_kg_per_acre: number;
  timing_advice: string;
  ph_slope: number;
  ph_last_value: number;
  oc_slope: number;
  oc_last_value: number;
  interpretation: string;
  trend_series: { year: number; ph: number; organic_carbon: number }[];
  district_comparison: { district: string; nitrogen: number; phosphorus: number; potassium: number }[];
  avg_yield_kg_per_ha: number;
  _meta: Meta;
}

export interface WeatherAlert {
  has_alert: boolean;
  alert_type: string;
  severity: "HIGH" | "MEDIUM" | "LOW" | "";
  message: string;
  action_required: string;
}

export interface WeatherResponse {
  district: string;
  temperature_avg: number;
  temperature_max: number;
  temperature_min: number;
  rainfall_mm: number;
  humidity_percent: number;
  wind_speed_kmph: number;
  sunshine_hours: number;
  weather_condition: string;
  farming_advisory: string;
  forecast: { date: string; temp_max: number; temp_min: number; rainfall_mm: number; rainfall_prob: number }[];
  historical: { year: number; rainfall: number; temperature: number; humidity: number }[];
  alert: WeatherAlert;
  _meta: Meta;
}

export type StressLevel = "Healthy" | "Mild" | "Moderate" | "Severe";

export interface CropHealthResponse {
  district: string;
  crop_status: string;
  disease_risk: "LOW" | "MEDIUM" | "HIGH";
  recommendation: string;
  confidence: number;
  growth_stage: string;
  days_after_sowing: number;
  ndvi: number;
  ndvi_deviation: number;
  stress_level: StressLevel;
  treatment: string;
  ndvi_source: string;
  stage_baseline: number;
  stage_curve: { stage: string; stage_key: string; q25: number; median: number; q75: number }[];
  _meta: Meta;
}

export interface TreatmentMatrix {
  stages: {
    stage: string;
    label: string;
    pests: string[];
    diseases: string[];
    advisory: string;
    treatments: Record<string, string>;
  }[];
}

export interface MarketResponse {
  district: string;
  crop_variety: string;
  varieties: string[];
  current_price_per_quintal: number;
  msp_price_per_quintal: number;
  price_change_30d: number;
  trend_direction: "RISING" | "FALLING" | "STABLE";
  sell_or_hold: "SELL" | "HOLD";
  best_selling_month: string;
  reasoning: string;
  price_trend_30d: { date: string; price: number; demand: string; supply: string }[];
  variety_table: { variety: string; price: number; demand: string; supply: string }[];
  market_demand: string;
  supply_status: string;
  _meta: Meta;
}

export interface PestResponse {
  district: string;
  season: string;
  growth_stage: string;
  risk_level: "LOW" | "MEDIUM" | "HIGH";
  risk_score: number;
  likely_pests: string[];
  likely_diseases: string[];
  pest_actions: { name: string; treatment: string }[];
  preventive_action: string;
  next_checkin_days: number;
  rule_matched: string;
  historical_pest_incidence: string;
  historical_disease_incidence: string;
  avg_sprays_per_season: number;
  conditions: { temperature_max: number; humidity_percent: number; rainfall_mm: number };
  _meta: Meta;
}

export interface AgentStatus {
  agent_type: string;
  agent_name: string;
  language: string;
  description: string;
  port: number;
  status: "ONLINE" | "OFFLINE";
  last_response_ms: number;
  served_by: string;
}

export interface SystemStatusResponse {
  orchestrator_status: string;
  orchestrator_port: number;
  orchestrator_latency_ms: number;
  active_agents: AgentStatus[];
  online_count: number;
  total_agents: number;
  gateway_note: string;
}

export interface OverviewResponse {
  district: string;
  season: string;
  growth_stage: string;
  cards: {
    soil_health_score: number;
    soil_sub: string;
    temperature: number;
    temperature_min: number;
    weather_sub: string;
    stress_level: StressLevel;
    crop_sub: string;
    market_price: number;
    market_sub: string;
    pest_risk: string;
    pest_sub: string;
  };
  advisory: { agent: string; severity: string; title: string; body: string; action: string }[];
  coverage: { label: string; value: string }[];
  agents: AgentStatus[];
  online_count: number;
  _meta: Meta;
}

// ── Image disease detection (YOLOv8) ──────────────────────────────────────
export interface DiseasePrediction {
  raw_label: string;
  disease: string;
  confidence: number;
  severity: string;
  treatment: string;
  is_paddy_condition: boolean;
}

export interface DetectResponse {
  is_leaf?: boolean;
  vegetation_index?: number;
  vegetation_note?: string;
  low_vegetation?: boolean;
  message?: string;
  predicted_disease?: string | null;
  confidence?: number | null;
  severity?: string;
  treatment?: string;
  is_paddy_condition?: boolean;
  predictions?: DiseasePrediction[];
  model?: { path: string; is_finetuned: boolean; note: string };
  latency_ms?: number;
  filename?: string;
  error?: string;
  detail?: string;
  _meta?: Meta;
}

export interface DetectStatus {
  available: boolean;
  model: string | null;
  is_finetuned: boolean;
  note: string;
}

export interface PerClassMetric {
  class: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ModelMetrics {
  trained: boolean;
  target_accuracy: number;
  note?: string;
  model?: string;
  top1_accuracy?: number;
  meets_target?: boolean;
  num_classes?: number;
  class_names?: string[];
  val_images?: number;
  train_images?: number;
  per_class?: PerClassMetric[];
  confusion_matrix?: number[][];
  eval_seconds?: number;
  evaluated_at?: string;
}

export interface ExplainResponse {
  top_class?: string;
  grid?: number;
  overlay_base64?: string;
  error?: string;
  detail?: string;
}

export interface ModelVersion {
  version: string;
  date: string;
  classes: number;
  train_images: number;
  val_images: number;
  accuracy: number;
  trustworthy: boolean;
  note: string;
}
export interface Revision {
  title: string;
  detail: string;
}
export interface ModelHistory {
  versions: ModelVersion[];
  revisions: Revision[];
  note?: string;
}

// ── Cross-agent orchestrator ───────────────────────────────────────────────
export interface OrchestratorAgentResult {
  agent_name: string;
  status: "OK" | "ERROR" | "UNKNOWN_AGENT";
  confidence: number;
  payload: Record<string, unknown>;
}

export interface OrchestratorResponse {
  online: boolean;
  query_id?: string;
  district?: string;
  season?: string;
  days_after_transplant?: number;
  query_type?: string;
  recommendation: string;
  overall_confidence: number;
  agent_results: OrchestratorAgentResult[];
  agents_consulted?: number;
  timestamp?: string;
  _meta: Meta;
}

async function get<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const url = new URL(`${getApiBase()}/api${path}`);
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== "" && v !== null) url.searchParams.set(k, String(v));
    });
  }
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${path}`);
  return res.json() as Promise<T>;
}

async function postForm<T>(path: string, form: FormData): Promise<T> {
  const res = await fetch(`${getApiBase()}/api${path}`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${path}`);
  return res.json() as Promise<T>;
}

export const api = {
  meta: () => get<MetaResponse>("/meta"),
  detectStatus: () => get<DetectStatus>("/crop-health/detect/status"),
  modelMetrics: () => get<ModelMetrics>("/crop-health/model-metrics"),
  modelHistory: () => get<ModelHistory>("/crop-health/model-history"),
  explainDisease: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return postForm<ExplainResponse>("/crop-health/detect/explain", form);
  },
  detectDisease: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return postForm<DetectResponse>("/crop-health/detect", form);
  },
  orchestrator: (district: string, season: string, stage?: string) =>
    get<OrchestratorResponse>("/orchestrator", { district, season, stage }),
  soil: (district: string, season: string) => get<SoilResponse>("/soil", { district, season }),
  weather: (district: string, season: string, stage: string) =>
    get<WeatherResponse>("/weather", { district, season, stage }),
  cropHealth: (district: string, season: string, stage?: string, ndvi?: number) =>
    get<CropHealthResponse>("/crop-health", { district, season, stage, ndvi }),
  treatmentMatrix: () => get<TreatmentMatrix>("/crop-health/matrix"),
  market: (district: string, season: string, variety?: string) =>
    get<MarketResponse>("/market", { district, season, variety }),
  pestRisk: (district: string, season: string, stage: string) =>
    get<PestResponse>("/pest-risk", { district, season, stage }),
  systemStatus: () => get<SystemStatusResponse>("/system/status"),
  overview: (district: string, season: string, stage?: string) =>
    get<OverviewResponse>("/overview", { district, season, stage }),
};
