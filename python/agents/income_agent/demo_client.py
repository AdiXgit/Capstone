"""Usage:  python demo_client.py Mandya Kharif BPT-5204 2"""
import os
import sys

import grpc

sys.path.append(os.path.join(os.path.dirname(__file__), "../../shared"))
import paddy_agents_pb2 as pb2
import paddy_agents_pb2_grpc as pb2_grpc

args = sys.argv[1:] + ["Mandya", "Kharif", "BPT-5204", "2"][len(sys.argv) - 1:]
district, season, variety, area = args[:4]

stub = pb2_grpc.IncomeAgentServiceStub(grpc.insecure_channel("localhost:50057"))
r = stub.GetIncomeEstimate(pb2.IncomeRequest(
    district=district, season=season, variety=variety, area_ha=float(area)), timeout=60)

print(f"\n=== FARM INCOME - {r.district}, {r.variety}, {r.area_ha:g} ha ===\n")
print(f"Yield:           {r.yield_kg_per_ha:,.0f} kg/ha  ->  {r.total_yield_quintals:,.1f} quintals")
print(f"Fertilizer cost: Rs {r.fertilizer_cost_total:,.0f}\n")
print(f"{'Scenario':<14}{'Price/q':>10}{'Gross':>14}{'Net':>14}")
for s in r.scenarios:
    print(f"{s.name:<14}{s.price_per_quintal:>10,.0f}{s.gross_income:>14,.0f}{s.net_income:>14,.0f}")
print(f"\nRecommendation: {r.recommendation.upper()}")
print(f"Advisory: {r.advisory}")
for w in r.warnings:
    print(f"Warning: {w}")
