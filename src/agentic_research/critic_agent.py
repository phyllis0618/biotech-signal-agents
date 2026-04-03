from __future__ import annotations

from typing import Any, Dict, List


class CriticAgent:
    """
    CriticAgent: Reflection — hunts for bearish / contrarian evidence vs bullish narrative.
    """

    def run(self, research_bundle: Dict[str, Any]) -> Dict[str, Any]:
        ext = research_bundle.get("extraction", {})
        fin = research_bundle.get("financials", {})

        risks: List[str] = []
        bias_flags: List[str] = []

        if ext.get("fda_adverse_event_sample_size", 0) >= 3:
            risks.append("FDA FAERS sample shows multiple events; check causality vs label noise.")
            bias_flags.append("surveillance_bias")

        if ext.get("trial_count", 0) == 0:
            risks.append("No CTGov hits for sponsor string — pipeline visibility may be stale or mis-tagged.")
            bias_flags.append("selection_bias")

        if ext.get("phase2_or_3_active", 0) == 0 and ext.get("active_recruiting_trials", 0) > 3:
            risks.append("Many recruiting trials but none flagged Phase 2/3 — early-stage concentration risk.")

        runway = fin.get("runway_months_est")
        if runway is not None and runway < 9:
            risks.append(f"Estimated runway ~{runway:.1f} mo — financing / dilution risk elevated.")
            bias_flags.append("liquidity_risk")

        filing_chars = fin.get("filing_chars", 0)
        if filing_chars < 2000:
            risks.append("Thin SEC filing context — financial estimates low confidence.")

        reflection_summary = (
            f"Identified {len(risks)} risk themes; "
            f"bias flags: {', '.join(bias_flags) or 'none'}."
        )

        return {
            "contrarian_risks": risks,
            "bias_flags": bias_flags,
            "reflection_summary": reflection_summary,
            "bull_case_undercut_score": min(1.0, 0.15 * len(risks)),
        }

    def counter_thesis(
        self,
        reflection: Dict[str, Any],
        research_bundle: Dict[str, Any],
        memory_hits: List[Dict[str, Any]],
    ) -> str:
        """
        Mandatory counter-thesis for Quant Trading acceptance criteria:
        articulate the bear case before any execution gate.
        """
        risks = reflection.get("contrarian_risks") or []
        risks_txt = "; ".join(risks) if risks else "No explicit structural risks flagged."
        ext = research_bundle.get("extraction") or {}
        fin = research_bundle.get("financials") or {}
        mem_snip = ""
        if memory_hits:
            first = memory_hits[0].get("text", "")[:240]
            mem_snip = f" Historical memory echo: {first!s}"

        return (
            "Counter-thesis: The bullish alpha may be overstated if (1) trial visibility is incomplete "
            f"(trials={ext.get('trial_count', 'n/a')}), (2) "
            f"liquidity is tight (runway est. {fin.get('runway_months_est', 'n/a')} mo), "
            f"(3) regulatory surveillance noise is mistaken for signal. "
            f"Key risks: {risks_txt}.{mem_snip}"
        )
