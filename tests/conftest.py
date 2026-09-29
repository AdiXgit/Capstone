import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ["python/shared",
          "python/agents/crop_health_agent",
          "python/agents/market_agent",
          "python/agents/pest_agent"]:
    sys.path.insert(0, os.path.join(BASE, p))

DATA = os.path.join(BASE, "data")
