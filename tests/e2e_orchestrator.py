"""End-to-end check against a running stack: python tests/e2e_orchestrator.py [host:port]"""
import json, os, sys, time
import grpc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python/shared"))
import paddy_agents_pb2 as pb
import paddy_agents_pb2_grpc as pb_grpc

ADDR = sys.argv[1] if len(sys.argv) > 1 else "localhost:50051"
EXPECT = {
    "WEATHER": 1, "SOIL": 1, "CROP_HEALTH": 2, "MARKET": 1,
    "PEST_RISK": 2, "INCOME": 1, "FULL": 6,
}


def main():
    stub = pb_grpc.OrchestratorServiceStub(grpc.insecure_channel(ADDR))
    st = stub.GetSystemStatus(pb.StatusRequest(), timeout=10)
    print("registered agents:", sorted(st.active_agents))
    failed = 0
    for qt, n in EXPECT.items():
        t = time.perf_counter()
        r = stub.RouteQuery(pb.FarmerQuery(query_id=f"e2e-{qt}", query_type=qt,
                            district="Mandya", farmer_id="f1", season="Kharif"), timeout=60)
        dt = (time.perf_counter() - t) * 1000
        statuses = {a.agent_name: a.status for a in r.agent_results}
        ok = len(r.agent_results) == n and all(s == "OK" for s in statuses.values())
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {qt:12s} {dt:8.0f} ms  {statuses}")
        if not ok:
            for a in r.agent_results:
                if a.status != "OK":
                    print("   ", a.agent_name, a.result_json[:200])
    sys.exit(1 if failed else 0)


main()
