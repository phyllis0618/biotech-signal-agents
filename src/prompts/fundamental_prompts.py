# Structured extraction prompts for biotech fundamentals (8-K / 10-Q / news text).

PROMPT_CLINICAL_RESULTS = """
Task: Extract Phase 1/2/3 trial data from biotech news/8-K.
Output: JSON {
  "primary_endpoint_met": boolean,
  "metrics": {"p_value": float, "hazard_ratio": float, "orr": "string"},
  "safety": ["list of adverse events"],
  "nct_id": "NCT01234567",
  "status": "Positive" | "Negative" | "Mixed"
}
"""

PROMPT_REGULATORY_CATALYST = """
Task: Identify FDA/EMA milestones.
Output: JSON {
  "event_type": "PDUFA" | "AdCom_Vote" | "CRL" | "Priority_Review",
  "milestone_date": "YYYY-MM-DD",
  "adcom_sentiment": "string (e.g. 10-2 favor)",
  "drug_name": "string"
}
"""

PROMPT_CASH_RUNWAY = """
Task: Calculate liquidity risk from 10-Q/10-K filings.
Output: JSON {
  "cash_on_hand": number,
  "quarterly_burn": number,
  "runway_quarters": float,
  "dilution_risk": "High" | "Medium" | "Low" (High if < 3 quarters)
}
"""

PROMPT_DEAL_FLOW = """
Task: Detect M&A rumors or Licensing deals.
Output: JSON {
  "type": "Acquisition" | "Licensing" | "Rumor",
  "big_pharma": "string",
  "upfront_payment": number,
  "total_value": number,
  "asset": "drug_name/platform"
}
"""

PROMPT_MARKET_TAM = """
Task: Estimate market size for specific drug indications.
Output: JSON {
  "target_pop": "string (e.g. 50k US)",
  "annual_price": number,
  "estimated_tam": number,
  "competitors": ["list"],
  "patent_expiry": 20XX
}
"""
