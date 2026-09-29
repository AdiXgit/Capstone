import { useState } from "react";
import { Routes, Route } from "react-router-dom";
import { Menu, Leaf } from "lucide-react";
import Sidebar from "./components/Sidebar";
import Overview from "./pages/Overview";
import Orchestrator from "./pages/Orchestrator";
import Soil from "./pages/Soil";
import Weather from "./pages/Weather";
import CropHealth from "./pages/CropHealth";
import ModelPerformance from "./pages/ModelPerformance";
import Market from "./pages/Market";
import PestRisk from "./pages/PestRisk";
import SystemStatus from "./pages/SystemStatus";

export default function App() {
  const [open, setOpen] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden bg-cream">
      {/* Mobile backdrop */}
      {open && (
        <div className="fixed inset-0 z-30 bg-black/40 lg:hidden" onClick={() => setOpen(false)} />
      )}

      <Sidebar open={open} onNavigate={() => setOpen(false)} />

      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Mobile top bar */}
        <header className="flex items-center gap-3 border-b border-forest-200/70 bg-cream px-4 py-3 lg:hidden">
          <button
            onClick={() => setOpen(true)}
            aria-label="Open menu"
            className="rounded-lg p-1.5 text-forest-900 transition hover:bg-forest-100"
          >
            <Menu className="h-6 w-6" />
          </button>
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-full bg-forest-700/80">
              <Leaf className="h-4 w-4 text-forest-100" />
            </span>
            <span className="font-serif text-[18px] leading-none text-forest-900">KrishiSense</span>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto px-4 py-4 sm:px-6 lg:px-8 lg:py-7 xl:px-10">
          <div className="mx-auto max-w-[1400px]">
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/orchestrator" element={<Orchestrator />} />
              <Route path="/soil" element={<Soil />} />
              <Route path="/weather" element={<Weather />} />
              <Route path="/crop-health" element={<CropHealth />} />
              <Route path="/model" element={<ModelPerformance />} />
              <Route path="/market" element={<Market />} />
              <Route path="/pest-risk" element={<PestRisk />} />
              <Route path="/system" element={<SystemStatus />} />
            </Routes>
          </div>
        </main>
      </div>
    </div>
  );
}
