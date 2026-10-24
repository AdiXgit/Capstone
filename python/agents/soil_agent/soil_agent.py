import sys
import os
import time
import grpc
from concurrent import futures

# Add shared proto folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../shared"))
sys.path.insert(0, os.path.dirname(__file__))

import paddy_agents_pb2       as pb2
import paddy_agents_pb2_grpc  as pb2_grpc

from npk_profiles      import get_npk_for_district
from fertilizer_model  import get_fertilizer_recommendation, predict_yield
from soil_trend        import get_soil_trend


class SoilAgentServicer(pb2_grpc.SoilAgentServiceServicer):

    def _meta(self, conf=0.85):
        return pb2.AgentMetadata(
            agent_id="soil-agent-001",
            agent_name="Soil Intelligence Agent",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            confidence=conf,
        )

    # ── Bundled RPC — matches orchestrator's expected call ──
    def GetSoilHealth(self, request, context):
        try:
            district = request.district
            season = request.season or "Kharif"

            npk = get_npk_for_district(district)
            trend = get_soil_trend(district)
            fert = get_fertilizer_recommendation(
                soil_quality="medium",  # can be refined later with real soil_quality input
                season=season,
            )
            yield_result = predict_yield(
                district=district,
                nitrogen=npk["N"],
                phosphorus=npk["P"],
                potassium=npk["K"],
                ph=trend.get("ph_last_value", 6.5),
                organic_carbon=trend.get("oc_last_value", 0.5),
                rainfall=900,       # placeholder average — refine with live weather agent data later
                irrigation=65,
                temperature=25,
                season=season,
            )

            return pb2.SoilResponse(
                metadata=self._meta(),
                district=district,
                nitrogen=npk["N"],
                phosphorus=npk["P"],
                potassium=npk["K"],
                ph=trend.get("ph_last_value", 0),
                organic_carbon=trend.get("oc_last_value", 0),
                urea_kg_per_ha=fert["urea_kg_per_ha"],
                dap_kg_per_ha=fert["dap_kg_per_ha"],
                potash_kg_per_ha=fert["potash_kg_per_ha"],
                timing_advice=fert["timing_advice"],
                predicted_yield_kg_per_ha=yield_result["predicted_yield_kg_per_ha"],
                yield_interpretation=yield_result["interpretation"],
                ph_trend_slope=trend.get("ph_slope", 0),
                oc_trend_slope=trend.get("oc_slope", 0),
                trend_interpretation=trend.get("interpretation", ""),
            )
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.SoilResponse()


def register_with_orchestrator(port, max_retries=10):
    addr = os.environ.get("ORCHESTRATOR_ADDR", "orchestrator:50051")
    for i in range(max_retries):
        try:
            with grpc.insecure_channel(addr) as ch:
                pb2_grpc.OrchestratorServiceStub(ch).RegisterAgent(
                    pb2.AgentRegistration(
                        agent_id="soil-agent-001",
                        agent_name="Soil Intelligence Agent",
                        agent_type="SOIL",
                        host="soil-agent",
                        port=port,
                    ),
                    timeout=5,
                )
                print(f"[SOIL] Registered with orchestrator at {addr}")
                return
        except grpc.RpcError as e:
            print(f"[SOIL] Retry {i+1}/{max_retries}: {e.details()}")
            time.sleep(3)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_SoilAgentServiceServicer_to_server(SoilAgentServicer(), server)
    port = int(os.environ.get("SOIL_AGENT_PORT", "50052"))
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"[SOIL] Soil Agent gRPC server running on port {port}")

    import threading
    threading.Thread(target=register_with_orchestrator, args=(port,), daemon=True).start()

    server.wait_for_termination()


if __name__ == "__main__":
    serve()