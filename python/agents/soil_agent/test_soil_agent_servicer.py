"""
None of test_soil_agent.py's 5 tests call GetSoilHealth() itself -- they test
the helper functions (get_npk_for_district, predict_yield, etc.) directly.
That's why the 'agro' undefined-variable bug shipped with 5/5 green: the one
code path that actually assembles a response was never exercised.
"""
import importlib.util
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../shared"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paddy_agents_pb2 as pb2

# "soil_agent" is ambiguous here: it's both this directory (a package, has
# __init__.py) and soil_agent.py inside it. Load the file directly by path
# so the import can't accidentally resolve to the package instead.
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("soil_agent_module", os.path.join(_here, "soil_agent.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SoilAgentServicer = _mod.SoilAgentServicer


class FakeContext:
    def __init__(self):
        self.code = None
        self.details = None
    def set_code(self, c): self.code = c
    def set_details(self, d): self.details = d


def test_get_soil_health_does_not_error():
    svc = SoilAgentServicer()
    ctx = FakeContext()
    r = svc.GetSoilHealth(pb2.SoilRequest(district="Mandya", season="Kharif"), ctx)
    assert ctx.code is None, f"servicer set an error status: {ctx.details}"
    assert r.predicted_yield_kg_per_ha > 0


def test_fertilizer_dose_reflects_real_soil_quality_not_hardcoded_medium():
    svc = SoilAgentServicer()
    kodagu = svc.GetSoilHealth(pb2.SoilRequest(district="Kodagu", season="Kharif"), FakeContext())
    mandya = svc.GetSoilHealth(pb2.SoilRequest(district="Mandya", season="Kharif"), FakeContext())
    assert kodagu.urea_kg_per_ha != mandya.urea_kg_per_ha


def test_yield_varies_meaningfully_across_districts():
    svc = SoilAgentServicer()
    yields = {}
    for d in ["Mandya", "Kodagu", "Koppal"]:
        r = svc.GetSoilHealth(pb2.SoilRequest(district=d, season="Kharif"), FakeContext())
        yields[d] = r.predicted_yield_kg_per_ha
    spread = max(yields.values()) - min(yields.values())
    assert spread > 500, f"yields too similar across districts: {yields}"