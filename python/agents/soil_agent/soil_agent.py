import sys
import os
import grpc
from concurrent import futures

# Add shared proto folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../shared"))
sys.path.insert(0, os.path.dirname(__file__))

import paddy_agents_pb2       as pb2
import paddy_agents_pb2_grpc  as pb2_grpc

from npk_profiles      import get_npk_for_district
from fertilizer_model  import predict_fertilizer
from soil_trend        import get_soil_trend


class SoilAgentServicer(pb2_grpc.SoilAgentServicer):

    def GetNPKProfile(self, request, context):
        try:
            npk = get_npk_for_district(request.district)
            return pb2.NPKResponse(
                district   = request.district,
                nitrogen   = npk["N"],
                phosphorus = npk["P"],
                potassium  = npk["K"],
            )
        except ValueError as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return pb2.NPKResponse()

    def GetFertilizerRec(self, request, context):
        try:
            result = predict_fertilizer(
                district       = request.district,
                nitrogen       = request.nitrogen,
                phosphorus     = request.phosphorus,
                potassium      = request.potassium,
                ph             = request.ph,
                organic_carbon = request.organic_carbon,
                season         = request.season,
            )
            return pb2.FertilizerResponse(
                urea_kg_per_acre   = result["urea_kg_per_acre"],
                dap_kg_per_acre    = result["dap_kg_per_acre"],
                potash_kg_per_acre = result["potash_kg_per_acre"],
                timing_advice      = result["timing_advice"],
                season             = result["season"],
            )
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.FertilizerResponse()

    def GetSoilTrend(self, request, context):
        try:
            trend = get_soil_trend(request.district)
            return pb2.SoilTrendResponse(
                district        = trend["district"],
                ph_slope        = trend["ph_slope"],
                ph_last_value   = trend["ph_last_value"],
                oc_slope        = trend["oc_slope"],
                oc_last_value   = trend["oc_last_value"],
                interpretation  = trend["interpretation"],
            )
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.SoilTrendResponse()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_SoilAgentServicer_to_server(SoilAgentServicer(), server)
    port = "50052"
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Soil Agent gRPC server running on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()