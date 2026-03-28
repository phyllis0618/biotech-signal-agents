from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List

import pandas as pd
import streamlit as st

from src.analytics.attribution import compute_event_window_attribution
from src.analytics.catalyst import (
    build_catalyst_rows,
    dedupe_fda_calendar_rows,
    split_priority_watchlists,
)
from src.analytics.impact import estimate_signal_price_impact
from src.analytics.institutional_score import compute_institutional_scorecard
from src.analytics.pipeline_snapshot import build_pipeline_table_rows, summarize_pipeline_strength
from src.connectors.market_data import fetch_yahoo_intraday_series, fetch_yahoo_snapshots
from src.pipeline import run_pipeline
from src.universe import load_biotech_universe
from src.utils.history_store import append_signal_history, read_signal_history


st.set_page_config(page_title="Biotech Signal Dashboard", layout="wide")
st.title("Biotech Signal Dashboard")
st.caption("Universe-level biotech signals + catalyst watchlists (no auto page refresh; scroll stays put.)")

with st.sidebar:
    st.header("Settings")
    mode = "Biotech Universe CSV"
    universe_csv = st.text_input("Universe CSV path", "data/biotech_universe_sample.csv")
    cash_runway_months = st.slider("Cash runway (months)", 3, 36, 18)
    single_asset_exposure = st.checkbox("Single-asset exposure", value=False)
    st.markdown("### Basket Risk Controls")
    min_conf_long = st.slider("Min confidence for LONG basket", 0, 100, 60)
    min_conf_short = st.slider("Min confidence for SHORT basket", 0, 100, 55)
    min_financing_quality = st.slider("Min financing quality for LONG", 0, 100, 45)
    max_financing_quality_for_short = st.slider("Max financing quality for SHORT", 0, 100, 60)
    basket_top_n = st.slider("Top N per basket", 3, 25, 10)
    st.caption("Tip: use **Refresh** to reload prices only; **Run Signal Stream** recomputes pipeline.")
    refresh_prices = st.button("Refresh market prices")
    run_btn = st.button("Run Signal Stream")


def _signal_emoji(signal: str) -> str:
    return {"long": "🟢", "short": "🔴", "no_trade": "🟡"}.get(signal, "⚪")


def _build_flow_rows(messages) -> List[Dict]:
    rows = []
    for m in messages:
        rows.append(
            {
                "agent": m.agent,
                "signal_hint": m.signal_hint,
                "confidence": m.confidence,
                "summary": m.summary,
                "evidence_count": len(m.evidence),
            }
        )
    return rows


try:
    universe_rows = load_biotech_universe(universe_csv)
except Exception:
    universe_rows = []
tickers = [r["ticker"] for r in universe_rows]

if refresh_prices:
    st.rerun()

st.subheader("Market Snapshot")
snapshots = fetch_yahoo_snapshots(tickers)
snap_df = pd.DataFrame(snapshots)
if not snap_df.empty:
    st.dataframe(snap_df, use_container_width=True, hide_index=True)

cols = st.columns(max(1, len(tickers)))
for i, ticker in enumerate(tickers):
    row = next((s for s in snapshots if s["ticker"] == ticker), None)
    if not row:
        continue
    with cols[i]:
        st.metric(
            label=ticker,
            value=f'{row["price"]}',
            delta=f'{row["change"]} ({row["change_pct"]}%)',
        )

st.subheader("Intraday Price (K-line Proxy)")
intraday_rows = []
for t in tickers[:6]:
    one = fetch_yahoo_intraday_series(t, interval="5m", range_="1d")
    for row in one:
        intraday_rows.append({"ticker": t, "time": row["time"], "close": row["close"]})
intraday_df = pd.DataFrame(intraday_rows)
if not intraday_df.empty:
    intraday_df["time"] = pd.to_datetime(intraday_df["time"], utc=True)
    pivot_df = intraday_df.pivot_table(index="time", columns="ticker", values="close")
    st.line_chart(pivot_df, use_container_width=True)
else:
    st.warning("No intraday market data available.")

st.subheader("Agent Data Flow")
st.graphviz_chart(
    """
digraph G {
  rankdir=LR;
  ingestion -> fundamental;
  ingestion -> trial_progress;
  ingestion -> regulatory;
  fundamental -> signal;
  trial_progress -> market_impact;
  regulatory -> market_impact;
  market_impact -> signal;
  signal -> coordinator;
}
"""
)

if run_btn:
    report_rows = []
    pipeline_rows = []
    catalyst_rows = []
    scorecard_rows = []
    per_ticker_detail: List[Dict] = []
    run_list = universe_rows

    for item in run_list:
        ticker = item["ticker"]
        company = item["company"]
        messages, report, raw_data = run_pipeline(
            ticker=ticker,
            company=company,
            cash_runway_months=item["cash_runway_months"],
            single_asset_exposure=item["single_asset_exposure"],
        )
        snap_row = next((s for s in snapshots if s["ticker"] == ticker), {"change_pct": 0.0, "price": 0.0})
        per_ticker_detail.append(
            {
                "ticker": ticker,
                "messages": messages,
                "report": report,
                "raw_data": raw_data,
            }
        )

        report_rows.append(
            {
                "ticker": ticker,
                "company": company,
                "final_signal": report.final_signal,
                "confidence": report.confidence,
                "horizon": report.horizon,
                "risk_flags": "; ".join(report.risk_flags),
                "as_of": datetime.utcnow().isoformat(),
                "price": snap_row.get("price", 0.0),
                "change_pct": snap_row.get("change_pct", 0.0),
                "data_quality": "ok" if raw_data else "limited",
            }
        )
        append_signal_history(report_rows[-1])

        ctgov_studies = raw_data.get("ctgov_studies", [])
        pipeline_rows.extend(
            build_pipeline_table_rows(
                ctgov_studies=ctgov_studies,
                ticker=ticker,
                company=company,
                max_rows=12,
            )
        )
        catalyst_rows.extend(
            build_catalyst_rows(
                ctgov_studies=ctgov_studies,
                ticker=ticker,
                company=company,
                fda_calendar_events=raw_data.get("fda_calendar_events", []),
                max_rows=15,
            )
        )
        pipeline_summary = summarize_pipeline_strength(ctgov_studies)
        scorecard = compute_institutional_scorecard(
            pipeline_summary=pipeline_summary,
            report_confidence=report.confidence,
            signal=report.final_signal,
            cash_runway_months=int(item["cash_runway_months"]),
            single_asset_exposure=bool(item["single_asset_exposure"]),
        )
        scorecard_rows.append(
            {
                "ticker": ticker,
                "company": company,
                "final_signal": report.final_signal,
                "confidence": report.confidence,
                "total_trials": pipeline_summary["total_trials"],
                "phase23_trials": pipeline_summary["phase23_trials"],
                "completed_trials": pipeline_summary["completed_trials"],
                **scorecard,
            }
        )

    st.subheader("Cross-Ticker Signal Board")
    board_df = pd.DataFrame(report_rows)
    st.dataframe(board_df, use_container_width=True, hide_index=True)

    catalyst_deduped = dedupe_fda_calendar_rows(catalyst_rows)
    watch = split_priority_watchlists(catalyst_deduped)
    st.subheader("Catalyst Priority Watchlist (direct)")
    col_w1, col_w2, col_w3 = st.columns(3)
    priority_cols = [
        "ticker",
        "company",
        "days_to_target",
        "catalyst_type",
        "target_date",
        "brief_title",
    ]

    def _safe_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
        return [c for c in cols if c in df.columns]

    with col_w1:
        st.markdown("#### Next 7 days")
        dfp = pd.DataFrame(watch["next_7d"])
        if not dfp.empty:
            st.dataframe(dfp[_safe_cols(dfp, priority_cols)], use_container_width=True, hide_index=True)
        else:
            st.info("No dated catalysts in ≤7d.")
    with col_w2:
        st.markdown("#### Next 8–30 days")
        dfp = pd.DataFrame(watch["next_8_30d"])
        if not dfp.empty:
            st.dataframe(dfp[_safe_cols(dfp, priority_cols)], use_container_width=True, hide_index=True)
        else:
            st.info("No dated catalysts in 8–30d.")
    with col_w3:
        st.markdown("#### Next 31–90 days")
        dfp = pd.DataFrame(watch["next_31_90d"])
        if not dfp.empty:
            st.dataframe(dfp[_safe_cols(dfp, priority_cols)], use_container_width=True, hide_index=True)
        else:
            st.info("No dated catalysts in 31–90d.")

    st.subheader("Signal Impact vs Price Move")
    if not board_df.empty:
        board_df["signal_score"] = board_df["final_signal"].map({"short": -1, "no_trade": 0, "long": 1})
        st.scatter_chart(board_df.set_index("ticker")[["signal_score", "change_pct"]], use_container_width=True)

    st.subheader("Institutional Scorecard")
    scorecard_df = pd.DataFrame(scorecard_rows)
    if not scorecard_df.empty:
        scorecard_df = scorecard_df.sort_values(
            by=["composite_score", "confidence"], ascending=[False, False]
        )
        st.dataframe(scorecard_df, use_container_width=True, hide_index=True)

        st.subheader("Top Long / Top Short Baskets (Risk-Managed)")
        long_df = scorecard_df[
            (scorecard_df["final_signal"] == "long")
            & (scorecard_df["confidence"] >= min_conf_long)
            & (scorecard_df["financing_quality"] >= min_financing_quality)
        ].sort_values(by=["composite_score", "confidence"], ascending=[False, False]).head(
            basket_top_n
        )
        short_df = scorecard_df[
            (scorecard_df["final_signal"] == "short")
            & (scorecard_df["confidence"] >= min_conf_short)
            & (scorecard_df["financing_quality"] <= max_financing_quality_for_short)
        ].sort_values(by=["composite_score", "confidence"], ascending=[True, False]).head(
            basket_top_n
        )

        col_long, col_short = st.columns(2)
        with col_long:
            st.markdown("#### Long Basket")
            if not long_df.empty:
                st.dataframe(long_df, use_container_width=True, hide_index=True)
            else:
                st.info("No long names pass current thresholds.")
        with col_short:
            st.markdown("#### Short Basket")
            if not short_df.empty:
                st.dataframe(short_df, use_container_width=True, hide_index=True)
            else:
                st.info("No short names pass current thresholds.")

    st.subheader("Pipeline Monitor (Ticker x Clinical Pipeline)")
    pipeline_df = pd.DataFrame(pipeline_rows)
    if not pipeline_df.empty:
        st.dataframe(pipeline_df.sort_values(by=["ticker", "phase", "status"]), use_container_width=True, hide_index=True)
    else:
        st.info("No clinical pipeline rows found for current run.")

    st.subheader("Catalyst Calendar (full)")
    catalyst_df = pd.DataFrame(catalyst_deduped)
    if not catalyst_df.empty:
        sort_cols = [c for c in ["days_to_target", "eta_bucket", "ticker", "phase"] if c in catalyst_df.columns]
        st.dataframe(
            catalyst_df.sort_values(by=sort_cols or ["ticker"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No catalyst rows inferred from current studies.")

    with st.expander("Per-ticker agent stream & JSON (optional drill-down)", expanded=False):
        for detail in per_ticker_detail:
            t = detail["ticker"]
            report = detail["report"]
            messages = detail["messages"]
            st.markdown(
                f"### {t} {_signal_emoji(report.final_signal)} `{report.final_signal}` "
                f"(confidence={report.confidence}, horizon={report.horizon})"
            )
            flow_df = pd.DataFrame(_build_flow_rows(messages))
            st.dataframe(flow_df, use_container_width=True, hide_index=True)
            st.code(json.dumps(report.model_dump(), indent=2), language="json")
else:
    st.info("Set universe CSV in sidebar, then click **Run Signal Stream**.")

st.subheader("Signal Time Series")
history_rows = read_signal_history(limit=5000)
history_df = pd.DataFrame(history_rows)
if not history_df.empty and {"as_of", "ticker", "confidence"}.issubset(history_df.columns):
    history_df["as_of"] = pd.to_datetime(history_df["as_of"], utc=True, errors="coerce")
    history_df = history_df.dropna(subset=["as_of"]).sort_values("as_of")
    signal_map = {"short": -1, "no_trade": 0, "long": 1}
    history_df["signal_score"] = history_df["final_signal"].map(signal_map).fillna(0)

    hist_global = (
        history_df.groupby("as_of", as_index=False)
        .agg(
            avg_confidence=("confidence", "mean"),
            avg_signal_score=("signal_score", "mean"),
            observations=("ticker", "count"),
        )
        .sort_values("as_of")
    )
    if not hist_global.empty:
        hist_global_idx = hist_global.set_index("as_of")
        st.line_chart(
            hist_global_idx[["avg_confidence", "avg_signal_score"]],
            use_container_width=True,
        )
        st.dataframe(
            history_df.sort_values("as_of", ascending=False).head(200)[
                ["as_of", "ticker", "final_signal", "confidence", "horizon", "risk_flags"]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Estimated Signal -> Price Impact")
        impact_map = estimate_signal_price_impact(history_rows)
        impact_df = pd.DataFrame(
            [{"signal": k, "avg_change_pct": v} for k, v in impact_map.items()]
        ).sort_values("signal")
        st.dataframe(impact_df, use_container_width=True, hide_index=True)

        st.subheader("Event Window Return Attribution")
        attr_df = compute_event_window_attribution(history_rows)
        if not attr_df.empty:
            st.dataframe(attr_df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough history to compute attribution.")
else:
    st.info("No signal history yet. Run the stream to generate time-series records.")
