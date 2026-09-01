import { createContext, useContext, useMemo, useState, ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, MetaResponse } from "../lib/api";

interface AppState {
  district: string;
  season: string;
  stage: string;
  setDistrict: (d: string) => void;
  setSeason: (s: string) => void;
  setStage: (s: string) => void;
  meta?: MetaResponse;
  metaLoading: boolean;
}

const Ctx = createContext<AppState | null>(null);

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [district, setDistrict] = useState("Mandya");
  const [season, setSeason] = useState("Kharif");
  const [stage, setStage] = useState("Flowering");

  const { data: meta, isLoading: metaLoading } = useQuery({
    queryKey: ["meta"],
    queryFn: api.meta,
    staleTime: Infinity,
  });

  const value = useMemo(
    () => ({ district, season, stage, setDistrict, setSeason, setStage, meta, metaLoading }),
    [district, season, stage, meta, metaLoading]
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAppState() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useAppState must be used inside AppStateProvider");
  return ctx;
}
