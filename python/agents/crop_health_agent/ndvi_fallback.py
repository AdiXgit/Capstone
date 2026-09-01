"""NDVI fallback chain: Dataset -> NASA POWER -> Open-Meteo -> hardcoded default.

Each tier is tried in order; the first one to produce a value wins. This keeps
the crop health agent answering even when the dataset has no matching record
and one of the two free satellite/radiation APIs is down or rate-limited.
"""
import logging
from datetime import datetime, timedelta

import numpy as np
import requests

from stage_profiles import DISTRICT_COORDS, KARNATAKA_CENTROID

log = logging.getLogger("CROP_HEALTH.ndvi_fallback")


def _nasa_power(lat, lon):
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    try:
        r = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point", params={
            "parameters": "ALLSKY_SFC_PAR_TOT,T2M", "community": "AG",
            "longitude": lon, "latitude": lat, "start": start, "end": end, "format": "JSON",
        }, timeout=10)
        r.raise_for_status()
        par_vals = [v for v in r.json().get("properties", {}).get("parameter", {})
                    .get("ALLSKY_SFC_PAR_TOT", {}).values() if v > -900]
        if not par_vals:
            return None
        avg = float(np.mean(par_vals))
        ndvi = max(0.0, min(0.95, 0.04 * avg - 0.15))
        return {"ndvi": round(ndvi, 3), "source": "NASA_POWER"}
    except Exception as e:
        log.warning("NASA POWER fallback failed: %s", e)
        return None


def _open_meteo(lat, lon):
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon,
            "daily": "shortwave_radiation_sum",
            "timezone": "Asia/Kolkata", "forecast_days": 7,
        }, timeout=8)
        r.raise_for_status()
        rad = [v for v in r.json().get("daily", {}).get("shortwave_radiation_sum", []) if v]
        if not rad:
            return None
        ndvi = max(0.0, min(0.95, 0.035 * float(np.mean(rad)) - 0.1))
        return {"ndvi": round(ndvi, 3), "source": "OPEN_METEO"}
    except Exception as e:
        log.warning("Open-Meteo fallback failed: %s", e)
        return None


class NDVIFallbackClient:
    """Implements the Dataset -> NASA POWER -> Open-Meteo -> fallback chain."""

    _cache = {}

    def fetch(self, district, dataset_value=None):
        if dataset_value is not None:
            return {"ndvi": round(float(dataset_value), 3), "source": "DATASET"}

        lat, lon = DISTRICT_COORDS.get(district, KARNATAKA_CENTROID)
        cache_key = f"{lat:.2f}_{lon:.2f}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = _nasa_power(lat, lon) or _open_meteo(lat, lon) or {"ndvi": 0.60, "source": "FALLBACK"}
        self._cache[cache_key] = result
        return result
