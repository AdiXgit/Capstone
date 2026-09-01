import { Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Overview from "./pages/Overview";
import Soil from "./pages/Soil";
import Weather from "./pages/Weather";
import CropHealth from "./pages/CropHealth";
import Market from "./pages/Market";
import PestRisk from "./pages/PestRisk";
import SystemStatus from "./pages/SystemStatus";

export default function App() {
  return (
    <div className="flex h-screen overflow-hidden bg-cream">
      <Sidebar />
      <main className="flex-1 overflow-y-auto px-8 py-7 xl:px-10">
        <div className="mx-auto max-w-[1400px]">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/soil" element={<Soil />} />
            <Route path="/weather" element={<Weather />} />
            <Route path="/crop-health" element={<CropHealth />} />
            <Route path="/market" element={<Market />} />
            <Route path="/pest-risk" element={<PestRisk />} />
            <Route path="/system" element={<SystemStatus />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}
