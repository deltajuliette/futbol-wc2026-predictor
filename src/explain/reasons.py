"""Deterministic, traceable reasoning for each prediction.

Turns the *stored* model quantities behind a forecast into a short, factual
explanation — never an LLM narrative. Every sentence maps to a number the model
actually produced, so the text is byte-reproducible and traceable (CLAUDE.md:
"every number shown in the dashboard must be traceable to a stored table or model
artifact"; "do not ship dashboard metrics without definitions").

The rigor comes from Dixon-Coles being *log-linear*: a team's log expected-goals
is an exact sum of interpretable terms, so the home side's scoring-rate edge splits
*exactly* into attack, defense, and venue contributions::

    log(lambda_home) - log(lambda_away)
        = (attack_home - attack_away)        # attacking edge
        + (defense_home - defense_away)      # defensive edge  (higher defense = better)
        + home_adv * (venue is not neutral)  # venue

We surface, per fixture, a ranked set of drivers:

* ``strength_gap``  — Elo rating difference (independent cross-check).
* ``goals``         — the exact log-ratio decomposition above + absolute attack/
  defense standing within the field (percentile of the fitted ratings).
* ``shape``         — most likely scoreline, draw likelihood, open vs tight.
* ``edge``          — where the calibrated model disagrees with the Elo benchmark.
* ``calibration``   — how far the honesty layer moved the raw probability.
* ``uncertainty``   — thin-sample caveat when a team has few recent matches.

It references only factors the model uses; it never invents injuries, lineups, or
"momentum". :func:`explain` is pure → unit-testable and reproducible.

Example::

    bundle = explain("Spain", "Cape Verde", neutral=True, model=dc,
                     elo_diff=210.0, p_cal=(0.89, 0.08, 0.03),
                     p_raw=(0.93, 0.05, 0.02), elo_probs=(0.85, 0.10, 0.05),
                     mp=match_probs, recent_counts={"spain": 60, "cape-verde": 22})
    print(bundle.headline)
    print(bundle.to_json())
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from models.dixon_coles import DCModel
from models.scoreline import MatchProbabilities
from utils.naming import team_key

REASONING_VERSION = "deterministic-v1"

# --- Thresholds (defined once; these ARE the definitions behind the words) -------
ELO_GAP_SLIGHT = 60.0      # below this the strength gap is "narrow"
ELO_GAP_CLEAR = 150.0      # clear favorite
ELO_GAP_STRONG = 280.0     # strong / heavy favorite
PCTL_TOP = 0.75            # attack/defense rating in the field's top quartile
PCTL_BOTTOM = 0.25         # ... bottom quartile
PROB_STRONG = 0.65         # calibrated favorite probability bands
PROB_FAVORED = 0.50
PROB_SLIGHT = 0.40
CLOSE_MARGIN = 0.06        # |P(home) - P(away)| below this with no clear side = lineball
DRAW_HIGH = 0.30           # draw probability that is notably high
OVER25_OPEN = 0.55         # leans towards an open (3+ goal) game
OVER25_TIGHT = 0.45        # leans tight
CAL_SHIFT_NOTABLE = 0.03   # calibration moved the favored prob by >= 3 points
EDGE_NOTABLE = 0.05        # model vs benchmark disagreement worth flagging
THIN_SAMPLE_MATCHES = 10   # fewer recent matches than this widens uncertainty
CONF_EDGE_NOTABLE = 0.04   # cross-confederation log-goal-diff worth flagging


@dataclass
class Driver:
    """One factual reason behind a forecast.

    ``magnitude`` is the signed underlying quantity (units depend on ``kind``);
    ``salience`` (0..1) is what the ranking sorts on; ``direction`` is which side
    the driver favors (``home`` / ``away`` / ``neutral``).
    """

    kind: str
    label: str
    magnitude: float
    salience: float
    direction: str
    text: str


@dataclass
class ReasoningBundle:
    """A headline plus drivers ranked most- to least-salient."""

    headline: str
    drivers: list[Driver]
    version: str = REASONING_VERSION

    def top(self, n: int = 3) -> list[Driver]:
        return self.drivers[:n]

    def to_json(self) -> str:
        return json.dumps(
            {"headline": self.headline, "version": self.version,
             "drivers": [asdict(d) for d in self.drivers]},
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, s: str) -> "ReasoningBundle":
        d = json.loads(s)
        return cls(headline=d["headline"],
                   drivers=[Driver(**x) for x in d["drivers"]],
                   version=d.get("version", REASONING_VERSION))


def _percentile(value: float, population: Sequence[float]) -> float:
    """Fraction of the population strictly below ``value`` (0..1)."""
    if not population:
        return 0.5
    below = sum(1 for x in population if x < value)
    return below / len(population)


def _standing(pctl: float) -> str:
    if pctl >= PCTL_TOP:
        return "top-quartile"
    if pctl <= PCTL_BOTTOM:
        return "bottom-quartile"
    return "mid-pack"


def explain(
    home: str,
    away: str,
    *,
    neutral: bool,
    model: DCModel,
    elo_diff: float,
    p_cal: tuple[float, float, float],
    p_raw: tuple[float, float, float],
    elo_probs: tuple[float, float, float] | None,
    mp: MatchProbabilities,
    recent_counts: Mapping[str, int],
    home_conf: str | None = None,
    away_conf: str | None = None,
) -> ReasoningBundle:
    """Build a :class:`ReasoningBundle` from stored model quantities.

    ``elo_diff`` is the pure home-minus-away Elo rating difference (no venue term).
    ``p_cal`` / ``p_raw`` are calibrated / raw (home, draw, away) probabilities.
    ``recent_counts`` maps ``team_key`` -> number of recent matches behind each
    team's strength estimate (drives the thin-sample caveat). ``home_conf`` /
    ``away_conf`` are the sides' confederations, used only for the cross-pool driver.

    Exact decomposition invariant: ``goals`` magnitude + ``continent`` magnitude
    equals ``log(lambda_home) - log(lambda_away)`` to machine precision.
    """
    h, a = team_key(home), team_key(away)
    ph, pd_, pa = p_cal
    drivers: list[Driver] = []

    # --- strength gap (Elo) ------------------------------------------------------
    side = "home" if elo_diff > 0 else "away"
    favored, other = (home, away) if elo_diff > 0 else (away, home)
    g = abs(elo_diff)
    if g >= ELO_GAP_STRONG:
        gap_word = "a heavy favorite"
    elif g >= ELO_GAP_CLEAR:
        gap_word = "a clear favorite"
    elif g >= ELO_GAP_SLIGHT:
        gap_word = "a modest favorite"
    else:
        gap_word = "barely separated"
    if g < ELO_GAP_SLIGHT:
        gap_text = (f"{home} and {away} are barely separated on Elo "
                    f"({g:.0f}-point gap).")
    else:
        gap_text = (f"{favored} enter as {gap_word} — rated ~{g:.0f} Elo points "
                    f"above {other}.")
    drivers.append(Driver(
        kind="strength_gap", label=f"Elo gap {g:.0f}", magnitude=float(elo_diff),
        salience=min(g / 400.0, 1.0), direction=side if g >= ELO_GAP_SLIGHT else "neutral",
        text=gap_text,
    ))

    # --- goals decomposition (exact log-linear split) ---------------------------
    hadv = 0.0 if neutral else model.home_adv
    atk_h, atk_a = model.attack.get(h, 0.0), model.attack.get(a, 0.0)
    def_h, def_a = model.defense.get(h, 0.0), model.defense.get(a, 0.0)
    atk_edge = atk_h - atk_a
    def_edge = def_h - def_a
    conf_edge = model.conf_edge(home_conf, away_conf)   # cross-pool piece (0 if off/intra)
    # Within-pool log-goal-ratio; together with conf_edge == log(lam_h) - log(lam_a).
    log_ratio = atk_edge + def_edge + hadv
    atk_pctl_h = _percentile(atk_h, list(model.attack.values()))
    def_pctl_h = _percentile(def_h, list(model.defense.values()))
    atk_pctl_a = _percentile(atk_a, list(model.attack.values()))
    def_pctl_a = _percentile(def_a, list(model.defense.values()))

    # Which component drives the home side's scoring-rate edge?
    parts = {"attacking": atk_edge, "defensive": def_edge, "home-venue": hadv}
    dom = max(parts, key=lambda k: abs(parts[k]))
    lead, trail = (home, away) if log_ratio >= 0 else (away, home)
    if dom == "home-venue":
        edge_clause = f"playing at a non-neutral venue tilts it toward {home}"
    elif dom == "attacking":
        stronger = home if atk_edge >= 0 else away
        edge_clause = (f"the gap is mostly offensive — {stronger}'s attack "
                       f"({_standing(atk_pctl_h if stronger == home else atk_pctl_a)}) "
                       f"outweighs the opponent's")
    else:
        stronger = home if def_edge >= 0 else away
        edge_clause = (f"the gap is mostly defensive — {stronger} concedes less "
                       f"({_standing(def_pctl_h if stronger == home else def_pctl_a)} "
                       f"defense)")
    goals_text = (f"Projected {mp.exp_goals_home:.1f}–{mp.exp_goals_away:.1f} "
                  f"expected goals; {edge_clause}.")
    drivers.append(Driver(
        kind="goals", label=f"xG {mp.exp_goals_home:.1f}-{mp.exp_goals_away:.1f}",
        magnitude=float(log_ratio), salience=min(abs(log_ratio) / 1.0, 1.0),
        direction="home" if log_ratio > 0 else ("away" if log_ratio < 0 else "neutral"),
        text=goals_text,
    ))

    # --- cross-confederation correction -----------------------------------------
    if abs(conf_edge) >= CONF_EDGE_NOTABLE and home_conf and away_conf:
        favc = home_conf if conf_edge > 0 else away_conf
        othc = away_conf if conf_edge > 0 else home_conf
        fav_name = home if conf_edge > 0 else away
        conf_text = (f"Cross-confederation: {favc} sides have historically out-performed "
                     f"{othc} beyond their ratings, tilting toward {fav_name}.")
        drivers.append(Driver(
            kind="continent", label=f"{home_conf} v {away_conf}",
            magnitude=float(conf_edge), salience=min(abs(conf_edge) / 0.30, 1.0) * 0.7,
            direction="home" if conf_edge > 0 else "away", text=conf_text,
        ))

    # --- match shape -------------------------------------------------------------
    top_score = mp.top_scorelines[0][0] if mp.top_scorelines else "n/a"
    top_p = mp.top_scorelines[0][1] if mp.top_scorelines else 0.0
    if mp.p_over25 >= OVER25_OPEN:
        tempo = "leans open (3+ goals)"
    elif mp.p_over25 <= OVER25_TIGHT:
        tempo = "leans tight (under 2.5)"
    else:
        tempo = "balanced for total goals"
    draw_clause = (f"elevated draw risk ({pd_:.0%})" if pd_ >= DRAW_HIGH
                   else f"low draw chance ({pd_:.0%})")
    shape_text = (f"Most likely {top_score} ({top_p:.0%}); {draw_clause}; {tempo}.")
    # More decisive shapes (very low/high draw, clear tempo lean) are more salient.
    shape_sal = min(abs(pd_ - 0.27) / 0.2 + abs(mp.p_over25 - 0.5) / 0.3, 1.0) * 0.6
    drivers.append(Driver(
        kind="shape", label=f"top {top_score}", magnitude=float(pd_),
        salience=shape_sal, direction="neutral", text=shape_text,
    ))

    # --- edge vs benchmark -------------------------------------------------------
    if elo_probs is not None:
        eh, _ed, ea = elo_probs
        # Compare on the model's favored win side (home vs away, ignoring draw).
        if ph >= pa:
            edge = ph - eh
            edge_side, edge_name = "home", home
        else:
            edge = pa - ea
            edge_side, edge_name = "away", away
        if abs(edge) >= EDGE_NOTABLE:
            verb = "more bullish on" if edge > 0 else "more cautious on"
            edge_text = (f"The calibrated model is {verb} {edge_name} than the Elo "
                         f"benchmark ({edge:+.0%} on the {edge_side} win).")
            drivers.append(Driver(
                kind="edge", label=f"edge {edge:+.0%}", magnitude=float(edge),
                salience=min(abs(edge) / 0.15, 1.0), direction=edge_side,
                text=edge_text,
            ))

    # --- calibration adjustment --------------------------------------------------
    # Compare raw vs calibrated on the favored outcome (argmax of calibrated).
    fav_idx = max(range(3), key=lambda i: p_cal[i])
    shift = p_cal[fav_idx] - p_raw[fav_idx]
    if abs(shift) >= CAL_SHIFT_NOTABLE:
        outcome = ("home win", "draw", "away win")[fav_idx]
        direction_word = "trims" if shift < 0 else "lifts"
        cal_text = (f"Calibration {direction_word} the {outcome} from "
                    f"{p_raw[fav_idx]:.0%} (raw) to {p_cal[fav_idx]:.0%} — correcting "
                    f"the model's historical {'over' if shift < 0 else 'under'}confidence.")
        drivers.append(Driver(
            kind="calibration", label=f"cal {shift:+.0%}", magnitude=float(shift),
            salience=min(abs(shift) / 0.10, 1.0), direction="neutral", text=cal_text,
        ))

    # --- uncertainty / thin sample ----------------------------------------------
    ch = recent_counts.get(h, 0)
    ca = recent_counts.get(a, 0)
    thin = [name for name, c in ((home, ch), (away, ca)) if c < THIN_SAMPLE_MATCHES]
    if thin:
        who = " and ".join(thin)
        unc_text = (f"Thin recent sample for {who} ({min(ch, ca)} matches) — treat the "
                    f"probabilities as wider than they look.")
        drivers.append(Driver(
            kind="uncertainty", label="thin sample", magnitude=float(min(ch, ca)),
            salience=0.9, direction="neutral", text=unc_text,
        ))

    # Rank most- to least-salient (stable on ties for reproducibility).
    drivers.sort(key=lambda d: d.salience, reverse=True)

    headline = _headline(home, away, ph, pd_, pa)
    return ReasoningBundle(headline=headline, drivers=drivers)


def _headline(home: str, away: str, ph: float, pd_: float, pa: float) -> str:
    """One-sentence summary anchored to the calibrated favorite probability."""
    fav_name, fav_p = (home, ph) if ph >= pa else (away, pa)
    if abs(ph - pa) < CLOSE_MARGIN:
        return (f"Lineball: {home} and {away} are closely matched "
                f"({ph:.0%}/{pd_:.0%}/{pa:.0%}).")
    if fav_p >= PROB_STRONG:
        band = "strong favorites"
    elif fav_p >= PROB_FAVORED:
        band = "favored"
    elif fav_p >= PROB_SLIGHT:
        band = "slight favorites"
    else:
        band = "narrow favorites in a wide-open match"
    return f"{fav_name} are {band} at {fav_p:.0%} (H {ph:.0%} / D {pd_:.0%} / A {pa:.0%})."
