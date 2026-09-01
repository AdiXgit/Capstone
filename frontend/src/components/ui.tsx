import { ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { Meta } from "../lib/api";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`card ${className}`}>{children}</div>;
}

export function SectionTitle({
  title,
  subtitle,
  right,
}: {
  title: string;
  subtitle?: string;
  right?: ReactNode;
}) {
  return (
    <div className="mb-5 flex items-end justify-between gap-4">
      <div>
        <h2 className="text-[26px] leading-tight text-forest-900">{title}</h2>
        {subtitle && <p className="mt-1 text-sm text-ink-muted">{subtitle}</p>}
      </div>
      {right}
    </div>
  );
}

const ACCENTS: Record<string, string> = {
  green: "border-l-forest-600",
  blue: "border-l-[#4A90C2]",
  gold: "border-l-gold",
  red: "border-l-[#D9534F]",
  slate: "border-l-ink-muted",
};

export function StatCard({
  label,
  value,
  unit,
  sub,
  accent = "green",
  badge,
}: {
  label: string;
  value?: ReactNode;
  unit?: string;
  sub?: string;
  accent?: keyof typeof ACCENTS | string;
  badge?: ReactNode;
}) {
  return (
    <div className={`card border-l-4 ${ACCENTS[accent] ?? ACCENTS.green} p-5`}>
      <div className="label-cap">{label}</div>
      <div className="mt-2.5 flex items-baseline gap-1.5">
        {badge ?? (
          <>
            <span className="font-serif text-[34px] leading-none text-forest-900">{value}</span>
            {unit && <span className="text-[13px] text-ink-muted">{unit}</span>}
          </>
        )}
      </div>
      {sub && <div className="mt-2.5 text-[13px] leading-snug text-ink-muted">{sub}</div>}
    </div>
  );
}

const BADGES: Record<string, string> = {
  Healthy: "bg-[#DCFCE7] text-[#166534]",
  Mild: "bg-[#FEF9C3] text-[#854D0E]",
  Moderate: "bg-[#FFEDD5] text-[#9A3412]",
  Severe: "bg-[#FEE2E2] text-[#991B1B]",
  LOW: "bg-[#DCFCE7] text-[#166534]",
  MEDIUM: "bg-[#FEF9C3] text-[#854D0E]",
  HIGH: "bg-[#FEE2E2] text-[#991B1B]",
  SELL: "bg-[#DCFCE7] text-[#166534]",
  HOLD: "bg-[#FEF9C3] text-[#854D0E]",
  RISING: "bg-[#DCFCE7] text-[#166534]",
  FALLING: "bg-[#FEE2E2] text-[#991B1B]",
  STABLE: "bg-forest-100 text-forest-800",
  ONLINE: "bg-[#DCFCE7] text-[#166534]",
  OFFLINE: "bg-[#F1F1F1] text-ink-muted",
  INFO: "bg-forest-100 text-forest-800",
};

export function Badge({ value, className = "" }: { value: string; className?: string }) {
  return (
    <span className={`pill ${BADGES[value] ?? "bg-forest-100 text-forest-800"} ${className}`}>{value}</span>
  );
}

export function SourceTag({ meta }: { meta?: Meta }) {
  if (!meta) return null;
  const label =
    meta.source === "GRPC_AGENT"
      ? `${meta.agent.replace("_", " ")} agent · gRPC`
      : meta.source === "LIVE_API"
        ? "Open-Meteo live API"
        : "Gateway · dataset";
  const tone =
    meta.source === "GRPC_AGENT"
      ? "bg-forest-100 text-forest-800"
      : meta.source === "LIVE_API"
        ? "bg-[#DBEAFE] text-[#1E40AF]"
        : "bg-[#F3F4F6] text-ink-muted";
  return (
    <span className={`pill ${tone}`} title={meta.notes || undefined}>
      <span className="h-1.5 w-1.5 rounded-full bg-current opacity-70" />
      {label} · {meta.latency_ms.toFixed(0)}ms
    </span>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton rounded-lg ${className}`} />;
}

export function CardSkeleton({ height = "h-28" }: { height?: string }) {
  return (
    <div className={`card p-5 ${height}`}>
      <Skeleton className="h-3 w-24" />
      <Skeleton className="mt-4 h-8 w-32" />
      <Skeleton className="mt-3 h-3 w-40" />
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const message = error instanceof Error ? error.message : "Something went wrong";
  return (
    <div className="card flex items-start gap-3 border border-[#FCA5A5] bg-[#FEF2F2] p-5">
      <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-[#B91C1C]" />
      <div className="flex-1">
        <div className="text-sm font-semibold text-[#991B1B]">Could not reach the gateway</div>
        <div className="mt-1 text-[13px] text-[#7F1D1D]">{message}</div>
        <div className="mt-2 text-[12px] text-[#7F1D1D]/80">
          Start it with <code className="rounded bg-white/70 px-1.5 py-0.5">python3 gateway/main.py</code> from the
          repo root.
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="mt-3 inline-flex items-center gap-1.5 rounded-lg bg-[#991B1B] px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-[#7F1D1D]"
          >
            <RefreshCw className="h-3.5 w-3.5" /> Retry
          </button>
        )}
      </div>
    </div>
  );
}

export function EmptyState({ message }: { message: string }) {
  return <div className="card p-8 text-center text-sm text-ink-muted">{message}</div>;
}

export function InfoBox({
  title,
  children,
  tone = "default",
}: {
  title?: string;
  children: ReactNode;
  tone?: "default" | "warning" | "alert";
}) {
  const tones = {
    default: "bg-forest-100/70 border-forest-200",
    warning: "bg-[#FFF8E7] border-[#F0C040]",
    alert: "bg-[#FEF2F2] border-[#FCA5A5]",
  };
  return (
    <div className={`rounded-xl2 border p-4 ${tones[tone]}`}>
      {title && <div className="mb-1 text-[13px] font-semibold text-forest-900">{title}</div>}
      <div className="text-[13.5px] leading-relaxed text-ink-soft">{children}</div>
    </div>
  );
}
