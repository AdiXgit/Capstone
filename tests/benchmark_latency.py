"""Latency benchmark against a running stack.
Usage: python tests/benchmark_latency.py [host:port] [iterations] [concurrency]
Reports per-query-type wall-clock latency of OrchestratorService.RouteQuery
(sequential), plus FULL throughput under concurrent load.
"""
import os, statistics as st, sys, time
from concurrent.futures import ThreadPoolExecutor
import grpc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python/shared"))
import paddy_agents_pb2 as pb
import paddy_agents_pb2_grpc as pb_grpc

ADDR = sys.argv[1] if len(sys.argv) > 1 else "localhost:50051"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 8
TYPES = ["WEATHER", "SOIL", "MARKET", "PEST_RISK", "INCOME", "CROP_HEALTH", "FULL"]
stub = pb_grpc.OrchestratorServiceStub(grpc.insecure_channel(ADDR))


RPC_ERRORS = []


def call(qt):
    t = time.perf_counter()
    try:
        r = stub.RouteQuery(pb.FarmerQuery(query_id="bench", query_type=qt, district="Mandya",
                                           farmer_id="bench", season="Kharif"), timeout=60)
    except grpc.RpcError as e:
        RPC_ERRORS.append((qt, e.code().name))
        return (time.perf_counter() - t) * 1000, 0, 0
    return (time.perf_counter() - t) * 1000, sum(a.status == "OK" for a in r.agent_results), len(r.agent_results)


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


print(f"target={ADDR} iterations={N} (1 warm-up call per type discarded)")
print(f"{'query':12s} {'ok/total':>9s} {'min':>7s} {'p50':>7s} {'p95':>7s} {'max':>7s} {'mean':>7s}  (ms)")
for qt in TYPES:
    call(qt)
    runs = [call(qt) for _ in range(N)]
    lat = [r[0] for r in runs]
    ok = sum(r[1] for r in runs); tot = sum(r[2] for r in runs)
    print(f"{qt:12s} {ok:>4d}/{tot:<4d} {min(lat):7.1f} {pct(lat,50):7.1f} {pct(lat,95):7.1f} {max(lat):7.1f} {st.mean(lat):7.1f}")

total = N * 2
t0 = time.perf_counter()
with ThreadPoolExecutor(CONC) as ex:
    res = list(ex.map(lambda _: call("FULL"), range(total)))
wall = time.perf_counter() - t0
lat = [r[0] for r in res]
print(f"\nFULL under load: {total} requests, concurrency {CONC}: wall {wall:.2f}s, "
      f"{total/wall:.1f} req/s, p50 {pct(lat,50):.0f} ms, p95 {pct(lat,95):.0f} ms, max {max(lat):.0f} ms")

print(f"RPC-level failures: {len(RPC_ERRORS)} {RPC_ERRORS[:5]}")
