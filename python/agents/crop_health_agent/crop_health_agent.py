import os
import sys
import time
import logging
import threading
from concurrent import futures

import numpy as np
import grpc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../shared"))
sys.path.insert(0, os.path.dirname(__file__))

import paddy_agents_pb2 as pb
import paddy_agents_pb2_grpc as pb_grpc

from stage_profiles import STAGE_PROFILES, das_from_season, das_to_stage, stage_ndvi_median
from data_store import CropHealthDataStore
from ndvi_fallback import NDVIFallbackClient
from treatment_map import get_treatment, crop_status_from_stress

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [CROP_HEALTH] %(levelname)s %(message)s")
log = logging.getLogger("CROP_HEALTH")


class CropHealthAgentServiceServicer(pb_grpc.CropHealthAgentServiceServicer):
    def __init__(self, store: CropHealthDataStore, ndvi_client: NDVIFallbackClient):
        self.store = store
        self.ndvi_client = ndvi_client

    def _meta(self, conf=0.85, notes=""):
        return pb.AgentMetadata(
            agent_id="crop-health-agent-001",
            agent_name="Crop Health Monitoring Agent",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            confidence=conf, notes=notes,
        )

    def GetCropHealthStatus(self, request, context):
        district, season = request.district, request.season or "Kharif"
        log.info("GetCropHealthStatus -> %s %s", district, season)

        das = request.days_after_sowing or das_from_season(season)

        # Tier 1: dataset lookup (nearest growth-stage record for this district/season/DAS)
        rec = self.store.get_closest_growth_record(district, season, das)
        if rec:
            stage = rec["growth_stage"]
            das_used = rec["days_after_sowing"]
            lai, height, tillers = rec["leaf_area_index"], rec["plant_height_cm"], rec["tiller_count"]
            ndvi_result = self.ndvi_client.fetch(district, dataset_value=rec["ndvi"])
            confidence = 0.90
        else:
            # No dataset record — estimate stage from DAS and fall through the
            # NASA POWER -> Open-Meteo -> hardcoded chain for NDVI.
            stage = das_to_stage(das)
            prof = STAGE_PROFILES.get(stage, {})
            lai = float(np.mean(prof.get("lai", (2.0, 4.0))))
            height = float(np.mean(prof.get("height", (50, 90))))
            tillers = int(np.mean(prof.get("tillers", (10, 16))))
            das_used = das
            ndvi_result = self.ndvi_client.fetch(district, dataset_value=None)
            confidence = 0.90 if ndvi_result["source"] == "DATASET" else (
                0.75 if ndvi_result["source"] == "NASA_POWER" else
                0.65 if ndvi_result["source"] == "OPEN_METEO" else 0.50
            )

        ndvi = ndvi_result["ndvi"]
        ndvi_deviation = ndvi - stage_ndvi_median(stage)

        stress_level = self.store.predict_stress(ndvi, lai, das_used, height, tillers, stage)
        treatment = get_treatment(stress_level, stage)
        crop_status = crop_status_from_stress(stress_level)

        pest_risk = self.store.get_pest_risk(district, season)
        disease_risk = str(pest_risk.get("disease", "MEDIUM")).upper()

        return pb.CropHealthResponse(
            metadata=self._meta(confidence, notes=f"ndvi_source={ndvi_result['source']}"),
            district=district,
            crop_status=crop_status,
            disease_risk=disease_risk,
            recommendation=treatment,
            confidence=confidence,
            growth_stage=stage,
            days_after_sowing=int(das_used),
            ndvi=float(ndvi),
            ndvi_deviation=float(ndvi_deviation),
            stress_level=stress_level,
            treatment=treatment,
            ndvi_source=ndvi_result["source"],
        )

    def GetTreatmentRecommendation(self, request, context):
        district, season = request.district, request.season or "Kharif"
        log.info("GetTreatmentRecommendation -> %s stage=%s stress=%s",
                  district, request.growth_stage, request.stress_level)

        stage = request.growth_stage or None
        stress_level = request.stress_level or None

        if stage and stress_level:
            # Direct lookup — caller already knows both.
            confidence = 0.95
        else:
            # Derive whichever is missing the same way GetCropHealthStatus does.
            das = request.days_after_sowing or das_from_season(season)
            rec = self.store.get_closest_growth_record(district, season, das)
            if rec:
                stage = stage or rec["growth_stage"]
                das_used = rec["days_after_sowing"]
                lai, height, tillers = rec["leaf_area_index"], rec["plant_height_cm"], rec["tiller_count"]
            else:
                stage = stage or das_to_stage(das)
                prof = STAGE_PROFILES.get(stage, {})
                lai = float(np.mean(prof.get("lai", (2.0, 4.0))))
                height = float(np.mean(prof.get("height", (50, 90))))
                tillers = int(np.mean(prof.get("tillers", (10, 16))))
                das_used = das

            if not stress_level:
                ndvi_result = self.ndvi_client.fetch(district, dataset_value=rec["ndvi"] if rec else None)
                stress_level = self.store.predict_stress(
                    ndvi_result["ndvi"], lai, das_used, height, tillers, stage)
            confidence = 0.85

        treatment = get_treatment(stress_level, stage)
        return pb.TreatmentResponse(
            metadata=self._meta(confidence),
            growth_stage=stage,
            stress_level=stress_level,
            treatment=treatment,
        )


def register_with_orchestrator(port, max_retries=3):
    """Announce this agent to the orchestrator, if one is running.

    The orchestrator is optional — the agent serves fine standalone (the REST
    gateway and grpcurl both call it directly), so a missing orchestrator is
    logged once as information rather than as a wall of retry warnings.
    """
    # "orchestrator" is the docker-compose service name; on a laptop it is localhost.
    default_addr = "orchestrator:50051" if os.path.exists("/.dockerenv") else "localhost:50051"
    addr = os.environ.get("ORCHESTRATOR_ADDR", default_addr)
    host = os.environ.get("AGENT_ADVERTISE_HOST", "crop-health-agent" if os.path.exists("/.dockerenv") else "localhost")

    for i in range(max_retries):
        try:
            with grpc.insecure_channel(addr) as ch:
                pb_grpc.OrchestratorServiceStub(ch).RegisterAgent(
                    pb.AgentRegistration(
                        agent_id="crop-health-agent-001",
                        agent_name="Crop Health Monitoring Agent",
                        agent_type="CROP_HEALTH", host=host, port=port,
                    ),
                    timeout=3,
                )
                log.info("Registered with orchestrator at %s", addr)
                return
        except grpc.RpcError:
            if i < max_retries - 1:
                time.sleep(2)

    log.info("No orchestrator at %s — serving standalone. This is expected until the "
              "Go orchestrator is running; the REST gateway calls this agent directly.", addr)


def serve():
    base = os.path.dirname(os.path.abspath(__file__))
    default_xlsx = os.path.join(base, "../../../data/Karnataka_Paddy_AI_ML_Dataset.xlsx")
    default_csv = os.path.join(base, "../../../data/Karnataka_Paddy_Main_Dataset.csv")
    xlsx = os.environ.get("AI_ML_XLSX_PATH", default_xlsx)
    csv = os.environ.get("MAIN_CSV_PATH", default_csv)
    port = int(os.environ.get("CROP_HEALTH_AGENT_PORT", "50054"))

    store = CropHealthDataStore(xlsx, csv)
    ndvi_client = NDVIFallbackClient()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb_grpc.add_CropHealthAgentServiceServicer_to_server(
        CropHealthAgentServiceServicer(store, ndvi_client), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("Crop Health Agent running on :%d", port)

    threading.Thread(target=register_with_orchestrator, args=(port,), daemon=True).start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
