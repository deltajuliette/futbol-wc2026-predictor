"""Dashboard query layer: predictions + benchmarks + evaluation round-trip."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from clients.types import FixtureRecord, Provenance
from dashboards.queries import (
    evaluation_summary,
    latest_model_run,
    reliability_bins,
    upcoming_predictions,
)
from storage.dao import create_model_run, save_benchmarks, save_predictions, upsert_match
from storage.database import get_engine, init_db


@pytest.fixture()
def engine(tmp_path):
    return init_db(get_engine(f"sqlite:///{tmp_path / 'dash.sqlite'}"))


def _scheduled(engine):
    rec = FixtureRecord(
        competition="world_cup_2026", stage="group",
        kickoff_utc=datetime(2026, 6, 20, 18, tzinfo=UTC),
        home_team="Brazil", away_team="Argentina", neutral=True,
        status="scheduled", provenance=Provenance(source="test"),
    )
    return upsert_match(engine, rec)


def test_upcoming_predictions_joins_model_and_benchmark(engine):
    mid = _scheduled(engine)
    run = create_model_run(engine, "dixon_coles", training_window="2014..2026",
                           params_json=json.dumps({"home_adv": 0.25}))
    save_predictions(engine, run, [{
        "match_id": mid, "p_home_raw": 0.5, "p_draw_raw": 0.25, "p_away_raw": 0.25,
        "p_home_cal": 0.52, "p_draw_cal": 0.24, "p_away_cal": 0.24,
        "exp_goals_home": 1.6, "exp_goals_away": 1.1,
        "scoreline_json": json.dumps([["1-1", 0.12], ["1-0", 0.10]]),
        "p_btts": 0.55, "p_over25": 0.5, "predicted_at_utc": datetime.now(UTC).isoformat(),
    }])
    save_benchmarks(engine, [{
        "match_id": mid, "source": "elo_only", "method": "logistic",
        "p_home": 0.45, "p_draw": 0.27, "p_away": 0.28,
        "captured_at_utc": datetime.now(UTC).isoformat(),
    }])

    df = upcoming_predictions(engine)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["home"] == "Brazil" and row["away"] == "Argentina"
    assert row["elo_home"] == pytest.approx(0.45)
    assert row["edge_home_vs_elo"] == pytest.approx(0.52 - 0.45)


def test_reruns_do_not_multiply_rows(engine):
    # Two model runs + two benchmark snapshots for the same fixture must still yield
    # ONE row (latest run + latest benchmark), not 2x2=4.
    mid = _scheduled(engine)
    for cal in (0.50, 0.55):
        run = create_model_run(engine, "dixon_coles")
        save_predictions(engine, run, [{
            "match_id": mid, "p_home_raw": cal, "p_draw_raw": 0.25, "p_away_raw": 0.75 - cal,
            "p_home_cal": cal, "p_draw_cal": 0.25, "p_away_cal": 0.75 - cal,
            "exp_goals_home": 1.5, "exp_goals_away": 1.0,
            "scoreline_json": json.dumps([["1-0", 0.1]]), "p_btts": 0.5, "p_over25": 0.5,
            "predicted_at_utc": datetime.now(UTC).isoformat(),
        }])
        save_benchmarks(engine, [{
            "match_id": mid, "source": "elo_only", "method": "logistic",
            "p_home": cal - 0.05, "p_draw": 0.27, "p_away": 0.78 - cal,
            "captured_at_utc": datetime.now(UTC).isoformat(),
        }])
    df = upcoming_predictions(engine)
    assert len(df) == 1
    # Reflects the LATEST run/benchmark.
    assert df.iloc[0]["p_home_cal"] == pytest.approx(0.55)
    assert df.iloc[0]["elo_home"] == pytest.approx(0.50)


def test_latest_model_run(engine):
    assert latest_model_run(engine) is None
    create_model_run(engine, "dixon_coles")
    second = create_model_run(engine, "dixon_coles")
    assert latest_model_run(engine)["model_run_id"] == second


def test_evaluation_summary_and_reliability(engine):
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO evaluation_metrics (label, as_of_utc, n_matches, log_loss, "
            "brier, rps, calibration_json, sharpness) VALUES "
            "('dc_cal','2026-06-01T00:00:00Z',800,1.01,0.60,0.20,:c,0.18)"
        ), {"c": json.dumps([{"mean_pred": 0.3, "frac_obs": 0.31, "count": 100}])})
    ev = evaluation_summary(engine)
    assert "dc_cal" in set(ev["label"])
    rb = reliability_bins(engine, "dc_cal")
    assert rb.iloc[0]["mean_pred"] == pytest.approx(0.3)
