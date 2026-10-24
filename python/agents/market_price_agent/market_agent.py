import sys
import os
import time
import grpc
from concurrent import futures

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../shared"))
sys.path.insert(0, os.path.dirname(__file__))

import paddy_agents_pb2      as pb2
import paddy_agents_pb2_grpc as pb2_grpc

from market_analysis import get_market_advisory


class MarketAgentServicer(pb2_grpc.MarketAgentServiceServicer):

    def _meta(self, conf=0.80):
        return pb2.AgentMetadata(
            agent_id="market-agent-001",
            agent_name="Market Price Advisory Agent",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            confidence=conf,
        )

    def GetMarketAdvisory(self, request, context):
        try:
            variety = request.variety or "BPT-5204"
            result = get_market_advisory(variety, district=request.district)

            return pb2.MarketResponse(
                metadata=self._meta(),
                variety=result["variety"],
                current_price=result["current_price"],
                avg_30day=result["avg_30day"],
                avg_90day=result["avg_90day"],
                price_trend=result["price_trend"],
                market_demand=result["market_demand"],
                supply_status=result["supply_status"],
                best_month=result["best_month"],
                advisory=result["advisory"],
            )
        except ValueError as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return pb2.MarketResponse()
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.MarketResponse()


def register_with_orchestrator(port, max_retries=10):
    addr = os.environ.get("ORCHESTRATOR_ADDR", "orchestrator:50051")
    for i in range(max_retries):
        try:
            with grpc.insecure_channel(addr) as ch:
                pb2_grpc.OrchestratorServiceStub(ch).RegisterAgent(
                    pb2.AgentRegistration(
                        agent_id="market-agent-001",
                        agent_name="Market Price Advisory Agent",
                        agent_type="MARKET",
                        host="market-agent",
                        port=port,
                    ),
                    timeout=5,
                )
                print(f"[MARKET] Registered with orchestrator at {addr}")
                return
        except grpc.RpcError as e:
            print(f"[MARKET] Retry {i+1}/{max_retries}: {e.details()}")
            time.sleep(3)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_MarketAgentServiceServicer_to_server(MarketAgentServicer(), server)
    port = int(os.environ.get("MARKET_AGENT_PORT", "50055"))
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"[MARKET] Market Price Agent gRPC server running on port {port}")

    import threading
    threading.Thread(target=register_with_orchestrator, args=(port,), daemon=True).start()

    server.wait_for_termination()


if __name__ == "__main__":
    serve()