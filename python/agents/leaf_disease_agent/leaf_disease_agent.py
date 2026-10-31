"""gRPC server for the leaf disease agent (port 50058).

Sidecar to the Go pest risk agent: it diagnoses a leaf photo with a fine-tuned
ImageNet CNN, then — when weather context is supplied — calls
PestRiskAgentService and fuses the observational and predictive signals.
"""
import os
import sys
import time
import threading
from concurrent import futures

import grpc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../shared"))
sys.path.insert(0, os.path.dirname(__file__))

import paddy_agents_pb2 as pb2
import paddy_agents_pb2_grpc as pb2_grpc

from fusion import fuse
from inference import LeafDiseaseClassifier

AGENT_ID = "leaf-disease-agent-001"
AGENT_NAME = "Leaf Disease Diagnosis Agent"
DEFAULT_PORT = 50058


class LeafDiseaseServicer(pb2_grpc.LeafDiseaseAgentServiceServicer):
    def __init__(self):
        # Loaded once at startup so the first farmer request is not the one
        # that pays for reading a 45MB checkpoint.
        self.classifier = LeafDiseaseClassifier()
        self.pest_addr = os.environ.get("PEST_RISK_AGENT_ADDR", "pest-risk-agent:50056")
        print(f"[LEAF_DISEASE] Model loaded (cv_accuracy={self.classifier.cv_accuracy})")

    def _meta(self, conf):
        return pb2.AgentMetadata(
            agent_id=AGENT_ID,
            agent_name=AGENT_NAME,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            confidence=conf,
            notes="ImageNet-pretrained CNN, fine-tuned on 120 rice leaf images",
        )

    def _fetch_rule_risk(self, request):
        """Ask the Go pest risk agent for weather-driven risk. Never fatal."""
        try:
            with grpc.insecure_channel(self.pest_addr) as ch:
                resp = pb2_grpc.PestRiskAgentServiceStub(ch).GetPestRisk(
                    pb2.PestRiskRequest(
                        district=request.district,
                        season=request.season,
                        growth_stage=request.growth_stage,
                        temperature=request.temperature,
                        humidity=request.humidity,
                        rainfall=request.rainfall,
                    ),
                    timeout=5,
                )
            return resp.overall_risk, [p.name for p in resp.pests], resp.advisory
        except grpc.RpcError as e:
            print(f"[LEAF_DISEASE] Pest risk agent unreachable ({e.code().name}); "
                  f"falling back to image-only diagnosis")
            return None, [], ""

    def DiagnoseLeafImage(self, request, context):
        if not request.image:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("image field is empty")
            return pb2.LeafDiseaseResponse()

        try:
            cnn = self.classifier.predict(request.image)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"classification failed: {e}")
            return pb2.LeafDiseaseResponse()

        # Temperature and humidity both zero means the caller gave no weather
        # context, so there is nothing for the rulebook to reason about.
        has_weather = request.temperature > 0 and request.humidity > 0
        if has_weather and not request.skip_rule_fusion:
            rule_risk, rule_pests, rule_advisory = self._fetch_rule_risk(request)
        else:
            rule_risk, rule_pests, rule_advisory = None, [], ""

        result = fuse(cnn, rule_risk=rule_risk, rule_pests=rule_pests,
                      rule_advisory=rule_advisory)

        print(f"[LEAF_DISEASE] district={request.district} "
              f"detected={result['detected_disease']} risk={result['overall_risk']} "
              f"basis={result['basis']}")

        treatment = result["treatment"] or {}
        return pb2.LeafDiseaseResponse(
            metadata=self._meta(result["confidence"]),
            district=request.district,
            detected_disease=result["detected_disease"] or "",
            detection_confidence=result["detection_confidence"],
            inconclusive=cnn["inconclusive"],
            probabilities=[
                pb2.DiseaseProbability(disease=d, probability=p)
                for d, p in cnn["probabilities"].items()
            ],
            overall_risk=result["overall_risk"],
            basis=result["basis"],
            corroborated=result["corroborated"],
            pathogen=treatment.get("pathogen", ""),
            signs=treatment.get("signs", ""),
            action=treatment.get("action", ""),
            etl_threshold=treatment.get("etl", ""),
            advisory=result["advisory"],
            rule_pests=result["rule_pests"],
        )


def register_with_orchestrator(port, max_retries=10):
    addr = os.environ.get("ORCHESTRATOR_ADDR", "orchestrator:50051")
    for i in range(max_retries):
        try:
            with grpc.insecure_channel(addr) as ch:
                pb2_grpc.OrchestratorServiceStub(ch).RegisterAgent(
                    pb2.AgentRegistration(
                        agent_id=AGENT_ID,
                        agent_name=AGENT_NAME,
                        agent_type="LEAF_DISEASE",
                        host="leaf-disease-agent",
                        port=port,
                    ),
                    timeout=5,
                )
                print(f"[LEAF_DISEASE] Registered with orchestrator at {addr}")
                return
        except grpc.RpcError as e:
            print(f"[LEAF_DISEASE] Retry {i+1}/{max_retries}: {e.details()}")
            time.sleep(3)


def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        # Leaf photos are multi-megabyte JPEGs; the 4MB default rejects them.
        options=[("grpc.max_receive_message_length", 32 * 1024 * 1024)],
    )
    pb2_grpc.add_LeafDiseaseAgentServiceServicer_to_server(LeafDiseaseServicer(), server)
    port = int(os.environ.get("LEAF_DISEASE_AGENT_PORT", str(DEFAULT_PORT)))
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"[LEAF_DISEASE] Leaf Disease Agent gRPC server running on port {port}")

    threading.Thread(target=register_with_orchestrator, args=(port,), daemon=True).start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
