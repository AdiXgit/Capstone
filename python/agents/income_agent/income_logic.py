"""
Farm Income Agent — calculations (no gRPC here, easy to test).

net income = (yield kg/ha x area / 100) x price  -  fertilizer cost  -  other costs
  yield, fertilizer doses <- Soil Agent
  price                   <- Market Agent (never below MSP)
"""

# ── UPDATE these to current values before the demo ──
FERTILIZER_PRICE_RS_PER_KG = {
    "urea":   266.50 / 45,   # Rs 266.50 per 45 kg bag
    "dap":    1350.0 / 50,   # Rs 1350 per 50 kg bag
    "potash": 1700.0 / 50,   # approx Rs 1700 per 50 kg bag
}
MSP_RS_PER_QUINTAL = 2441.0                 # paddy (common) MSP, KMS 2026-27
HOLDING_COST_RS_PER_QUINTAL_MONTH = 30.0    # storage + weight loss + interest
KG_PER_QUINTAL = 100.0

# Cost of cultivation (used when the farmer doesn't give their own costs).
# MSP is fixed at 1.5 x the A2+FL cost of production (CACP), so
# A2+FL = 2441 / 1.5 = ~Rs 1,627 per quintal. This covers seed, fertilizer,
# labour (hired + family), machinery, irrigation, etc. It is a NATIONAL average -
# Karnataka's own price commission argues real costs are higher.
MSP_MARKUP_OVER_COST = 1.5
COST_OF_PRODUCTION_RS_PER_QUINTAL = round(MSP_RS_PER_QUINTAL / MSP_MARKUP_OVER_COST, 2)


def fertilizer_cost_per_ha(urea_kg, dap_kg, potash_kg, prices=FERTILIZER_PRICE_RS_PER_KG):
    cost = (max(urea_kg, 0) * prices["urea"]
            + max(dap_kg, 0) * prices["dap"]
            + max(potash_kg, 0) * prices["potash"])
    return round(cost, 2)


def default_other_cost_per_ha(yield_kg_per_ha, fert_cost_per_ha,
                              cost_per_quintal=COST_OF_PRODUCTION_RS_PER_QUINTAL):
    """
    Total cost of cultivation per ha = cost per quintal x quintals per ha.
    Fertilizer is already counted separately, so subtract it to avoid double counting.
    """
    total = cost_per_quintal * max(yield_kg_per_ha, 0) / KG_PER_QUINTAL
    return round(max(total - fert_cost_per_ha, 0), 2)


def floor_at_msp(price, msp=MSP_RS_PER_QUINTAL):
    """Farmer can always sell to government procurement at MSP."""
    if price is None or price <= 0:
        return msp
    return max(round(price, 2), msp)


def total_quintals(yield_kg_per_ha, area_ha):
    if area_ha <= 0:
        raise ValueError("area_ha must be positive")
    return round(max(yield_kg_per_ha, 0) * area_ha / KG_PER_QUINTAL, 2)


def income(quintals, price, total_costs):
    gross = quintals * price
    return round(gross, 2), round(gross - total_costs, 2)


def price_scenarios(price_now, price_forecast, msp=MSP_RS_PER_QUINTAL,
                    holding_cost=HOLDING_COST_RS_PER_QUINTAL_MONTH):
    """
    msp          : guaranteed minimum
    sell_now     : today's market price (floored at MSP)
    hold_1_month : next month's forecast minus 1 month of storage cost
    """
    now = floor_at_msp(price_now, msp)
    later = floor_at_msp(price_forecast if price_forecast else price_now, msp) - holding_cost
    return {"msp": msp, "sell_now": now, "hold_1_month": round(later, 2)}


def build_advisory(net_by_scenario, best_month, warnings):
    now = net_by_scenario["sell_now"]
    hold = net_by_scenario["hold_1_month"]
    msp = net_by_scenario["msp"]

    parts = [f"Expected net income if sold now: Rs {now:,.0f}."]
    if now < 0:
        parts.append("Costs are higher than expected income - review input costs.")
    if now <= msp:
        parts.append("Market price is at or below MSP - sell through government procurement.")
    if hold > now:
        parts.append(f"Holding one month could add about Rs {hold - now:,.0f} after storage costs.")
    else:
        parts.append("Selling now is better than storing.")
    if best_month:
        parts.append(f"Historically the best month to sell is {best_month}.")
    if warnings:
        parts.append("Some values used fallbacks (see warnings).")
    return " ".join(parts)