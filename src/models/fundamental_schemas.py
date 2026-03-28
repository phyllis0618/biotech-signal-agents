from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ClinicalMetrics(BaseModel):
    p_value: Optional[float] = None
    hazard_ratio: Optional[float] = None
    orr: str = ""


class ClinicalResultsJson(BaseModel):
    primary_endpoint_met: bool = False
    metrics: ClinicalMetrics = Field(default_factory=ClinicalMetrics)
    safety: List[str] = Field(default_factory=list)
    nct_id: str = ""
    status: str = "Mixed"


class RegulatoryCatalystJson(BaseModel):
    # Keep permissive strings; LLM output may vary slightly from the prompt enum.
    event_type: str = "PDUFA"
    milestone_date: str = ""
    adcom_sentiment: str = ""
    drug_name: str = ""


class CashRunwayJson(BaseModel):
    cash_on_hand: Optional[float] = None
    quarterly_burn: Optional[float] = None
    runway_quarters: Optional[float] = None
    dilution_risk: str = "Medium"


class DealFlowJson(BaseModel):
    type: str = "Licensing"
    big_pharma: str = ""
    upfront_payment: Optional[float] = None
    total_value: Optional[float] = None
    asset: str = ""


class MarketTAMJson(BaseModel):
    target_pop: str = ""
    annual_price: Optional[float] = None
    estimated_tam: Optional[float] = None
    competitors: List[str] = Field(default_factory=list)
    patent_expiry: Optional[int] = None
