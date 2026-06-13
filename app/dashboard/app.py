"""World Cup forecast dashboard (Streamlit).

Read-only view over the stored tables — it never recomputes predictions. Every panel
states the model run and timestamps it was built from. See docs/dashboard.md.

Run::

    streamlit run app/dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make src/ importable whether or not the package is installed.
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboards.queries import (
    evaluation_summary,
    latest_model_run,
    reliability_bins,
    upcoming_predictions,
)
from storage.database import get_engine

st.set_page_config(page_title="World Cup Forecast", layout="wide")


@st.cache_resource
def _engine():
    return get_engine()


def _prob_bar(home: float, draw: float, away: float) -> go.Figure:
    fig = go.Figure()
    for label, val, color in [("Home", home, "#1f77b4"), ("Draw", draw, "#7f7f7f"),
                              ("Away", away, "#d62728")]:
        fig.add_trace(go.Bar(y=["1X2"], x=[val], name=label, orientation="h",
                             marker_color=color, text=f"{val:.0%}", textposition="inside"))
    fig.update_layout(barmode="stack", height=80, margin=dict(l=0, r=0, t=0, b=0),
                      showlegend=False, xaxis=dict(range=[0, 1], visible=False),
                      yaxis=dict(visible=False))
    return fig


def _render_reasoning(reasoning_json: str | None) -> None:
    """Show the deterministic 'why' behind a forecast: headline + ranked drivers."""
    if not reasoning_json:
        return
    try:
        bundle = json.loads(reasoning_json)
    except (TypeError, ValueError):
        return
    drivers = bundle.get("drivers", [])
    if bundle.get("headline"):
        st.markdown(f"**Why:** {bundle['headline']}")
    for d in drivers[:3]:
        st.caption(f"• {d['text']}")
    if len(drivers) > 3:
        with st.expander(f"More factors ({len(drivers) - 3})"):
            for d in drivers[3:]:
                st.caption(f"• {d['text']}")


def render() -> None:
    engine = _engine()
    run = latest_model_run(engine)
    st.title("⚽ World Cup Forecast")
    if not run:
        st.warning("No model run found. Run the ETL + training pipeline first "
                   "(see docs/runbooks.md).")
        return

    params = json.loads(run["params_json"]) if run["params_json"] else {}
    st.caption(
        f"Model: **{run['model_name']}** · run #{run['model_run_id']} · "
        f"trained {run['training_window']} · home_adv={params.get('home_adv', '?'):.3f} · "
        f"created {run['created_at_utc'][:19]}Z"
    )
    st.info(
        "Probabilistic forecasts optimized for calibration, not single-match accuracy. "
        "Calibrated Dixon-Coles vs an Elo-only benchmark. Market/Opta columns appear "
        "when those sources are ingested."
    )

    tab_upcoming, tab_perf = st.tabs(["Upcoming matches", "Calibration & performance"])

    # ---- Upcoming matches ----
    with tab_upcoming:
        df = upcoming_predictions(engine)
        if df.empty:
            st.warning("No scheduled fixtures with predictions yet.")
        else:
            st.caption(f"{len(df)} fixtures · predictions generated "
                       f"{df['predicted_at_utc'].iloc[0][:19]}Z (model run "
                       f"#{int(df['model_run_id'].iloc[0])})")
            for _, r in df.iterrows():
                venue = "neutral" if r["neutral"] else "home adv"
                st.markdown(f"#### {r['home']} vs {r['away']}  "
                            f"<span style='color:gray;font-size:0.8em'>"
                            f"{r['kickoff_utc'].strftime('%Y-%m-%d %H:%M')}Z · "
                            f"{r['stage']} · {venue}</span>", unsafe_allow_html=True)
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.plotly_chart(_prob_bar(r["p_home_cal"], r["p_draw_cal"],
                                              r["p_away_cal"]), use_container_width=True,
                                    key=f"bar{r['match_id']}")
                    st.caption(
                        f"Model (calibrated): H **{r['p_home_cal']:.0%}** · "
                        f"D **{r['p_draw_cal']:.0%}** · A **{r['p_away_cal']:.0%}**  |  "
                        f"Elo bench: {r['elo_home']:.0%}/{r['elo_draw']:.0%}/"
                        f"{r['elo_away']:.0%}  |  edge(H) vs Elo: "
                        f"{r['edge_home_vs_elo']:+.0%}"
                    )
                with c2:
                    sl = json.loads(r["scoreline_json"])
                    top = ", ".join(f"{s} ({p:.0%})" for s, p in sl[:3])
                    st.metric("Expected goals",
                              f"{r['exp_goals_home']:.1f} – {r['exp_goals_away']:.1f}")
                    st.caption(f"Most likely: {top}")
                    st.caption(f"BTTS {r['p_btts']:.0%} · Over 2.5 {r['p_over25']:.0%}")
                _render_reasoning(r.get("reasoning_json"))
                st.divider()

    # ---- Calibration & performance ----
    with tab_perf:
        ev = evaluation_summary(engine)
        if ev.empty:
            st.warning("No evaluation metrics yet. Run scripts.evaluation.backtest.")
        else:
            st.subheader("Proper scoring rules (lower is better)")
            st.caption(f"Backtest as of {ev['as_of_utc'].iloc[0][:10]} · "
                       f"{int(ev['n_matches'].iloc[0])} held-out matches")
            show = ev[["label", "log_loss", "brier", "rps", "sharpness", "n_matches"]]
            st.dataframe(show.style.format({
                "log_loss": "{:.4f}", "brier": "{:.4f}", "rps": "{:.4f}",
                "sharpness": "{:.4f}"}), use_container_width=True, hide_index=True)

            st.subheader("Reliability — calibrated Dixon-Coles")
            rb = reliability_bins(engine, "dc_cal")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(dash="dash", color="gray"),
                                     name="perfect"))
            if not rb.empty:
                fig.add_trace(go.Scatter(x=rb["mean_pred"], y=rb["frac_obs"],
                                         mode="markers+lines", name="dc_cal",
                                         marker=dict(size=8)))
            fig.update_layout(height=420, xaxis_title="Mean predicted probability",
                              yaxis_title="Observed frequency",
                              xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True, key="reliability")
            st.caption("Points on the diagonal = well calibrated. One-vs-rest across "
                       "all three 1X2 classes.")


render()
