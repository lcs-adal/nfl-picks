#!/usr/bin/env python3
"""
NFL Weekly Picks — Differentiation-Optimized (straight-up, no draw)

Same architecture as liga_mx_picks.py, adapted for a binary outcome market
(NFL games don't end in ties for pick'em purposes). Optimizes for
separation in a 50-100 person pool, not just raw accuracy: picks the
favorite ("chalk") by default, but deliberately deviates on a budgeted
number of games where the model disagrees with the implied field
consensus by enough to be worth the contrarian upside.

Data sources (pick one):
    1. Live odds via The Odds API (free tier, 500 req/month):
         https://the-odds-api.com/#get-access
         export ODDS_API_KEY="your_key"
    2. Manual/pasted odds via --input matches.json (see SAMPLE_INPUT below)

Usage:
    python nfl_picks.py                              # live odds, this week
    python nfl_picks.py --input matches.json          # manual odds
    python nfl_picks.py --budget 3 --chalk-bias 0.60  # tune pool config
    python nfl_picks.py --record --log nfl_picks_log.csv
    python nfl_picks.py --analyze-log nfl_picks_log.csv

SAMPLE_INPUT (matches.json):
[
  {
    "home_team": "Kansas City Chiefs",
    "away_team": "Buffalo Bills",
    "ml_home": -145,
    "ml_away": 122,
    "commence_time": "2026-09-14T17:00:00+00:00"
  }
]
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
import statistics

SPORT_KEY = "americanfootball_nfl"
BASE_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"

# Pool sheet abbreviations (match the family pool's xlsx)
TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LVR", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WSH",
}

LOG_FIELDS = [
    "week_date", "home_team", "away_team", "chalk_pick", "final_pick",
    "is_differentiator", "confidence", "actual_result", "field_chalk_pct",
]


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class MatchOdds:
    home_team: str
    away_team: str
    ml_home: int       # moneyline, e.g. -145
    ml_away: int         # e.g. +122
    commence_time: Optional[datetime] = None
    total_line: Optional[float] = None   # median over/under across books


@dataclass
class PoolConfig:
    num_participants: int = 75
    differentiator_budget: int = 3    # NFL cards run ~13-16 games; scale vs Liga MX's 9-game/budget-2
    chalk_bias: float = 0.60          # field's tendency to pick chalk (tune from past pool sheets)
    min_edge_threshold: float = 0.05  # min diff_score required to even qualify as a flip candidate


@dataclass
class PickResult:
    match: MatchOdds
    probs: dict
    chalk: str
    final_pick: str
    is_differentiator: bool
    confidence: float
    diff_score_value: float


# --------------------------------------------------------------------------
# Odds math
# --------------------------------------------------------------------------

def american_to_prob(ml: int) -> float:
    if ml < 0:
        return -ml / (-ml + 100)
    return 100 / (ml + 100)


def devig(probs: dict) -> dict:
    total = sum(probs.values())
    if total == 0:
        return probs
    return {k: v / total for k, v in probs.items()}


def match_probabilities(m: MatchOdds) -> dict:
    raw = {
        "home": american_to_prob(m.ml_home),
        "away": american_to_prob(m.ml_away),
    }
    return devig(raw)


# --------------------------------------------------------------------------
# Pick / differentiation logic
# --------------------------------------------------------------------------

def chalk_pick(probs: dict) -> str:
    return max(probs, key=probs.get)


def field_pick_prob(probs: dict, chalk: str, chalk_bias: float) -> dict:
    """
    Proxy for 'what % of the pool picks each side', since we don't have
    live pool data mid-week. Field leans toward chalk at chalk_bias rate;
    with only two outcomes, the underdog gets the entire remainder
    (1 - chalk_bias) -- no draw-share splitting needed here, unlike Liga MX.
    """
    other = [k for k in probs if k != chalk][0]
    return {chalk: chalk_bias, other: 1 - chalk_bias}


def diff_score(p_correct: float, p_field_also_picks: float) -> float:
    """Expected separation value: P(you're right) * P(field is NOT with you)."""
    return p_correct * (1 - p_field_also_picks)


def confidence_bucket(p: float) -> str:
    if p >= 0.60:
        return "High"
    if p >= 0.50:
        return "Med"
    return "Low"


def build_picks(matches: list, cfg: PoolConfig) -> list:
    candidates = []
    for m in matches:
        probs = match_probabilities(m)
        chalk = chalk_pick(probs)
        field_probs = field_pick_prob(probs, chalk, cfg.chalk_bias)

        underdog = [k for k in probs if k != chalk][0]
        score = diff_score(probs[underdog], field_probs.get(underdog, 0))
        alt = underdog if score >= cfg.min_edge_threshold else None

        candidates.append({
            "match": m, "probs": probs, "chalk": chalk,
            "alt": alt, "alt_score": score if alt else -1.0,
        })

    # rank matches with a qualifying alt by differentiation score; flip the
    # top `differentiator_budget` -- if fewer qualify, don't force weak flips
    ranked = sorted(
        (c for c in candidates if c["alt"] is not None),
        key=lambda c: c["alt_score"],
        reverse=True,
    )
    flip_ids = {id(c["match"]) for c in ranked[: cfg.differentiator_budget]}

    results = []
    for c in candidates:
        is_flip = id(c["match"]) in flip_ids
        final = c["alt"] if is_flip else c["chalk"]
        results.append(PickResult(
            match=c["match"],
            probs=c["probs"],
            chalk=c["chalk"],
            final_pick=final,
            is_differentiator=is_flip,
            confidence=c["probs"][final],
            diff_score_value=c["alt_score"] if is_flip else 0.0,
        ))
    return results


# --------------------------------------------------------------------------
# Data fetching
# --------------------------------------------------------------------------

def fetch_odds_api(api_key: str, region: str, days: int) -> list:
    params = {
        "apiKey": api_key,
        "regions": region,
        "markets": "h2h,totals",   # 2 credits per run (2 markets x 1 region)
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    resp = requests.get(BASE_URL, params=params, timeout=15)
    if resp.status_code == 401:
        sys.exit("Error: invalid or missing API key. Set ODDS_API_KEY or pass --api-key.")
    if resp.status_code == 422:
        sys.exit(f"Error: bad request - {resp.text}")
    resp.raise_for_status()

    remaining = resp.headers.get("x-requests-remaining")
    if remaining is not None:
        print(f"(API requests remaining this month: {remaining})\n")

    games = resp.json()
    cutoff = datetime.now(timezone.utc) + timedelta(days=days)
    matches = []
    for g in games:
        commence = datetime.fromisoformat(g["commence_time"].replace("Z", "+00:00"))
        if commence > cutoff:
            continue
        m = extract_match_odds(g, commence)
        if m:
            matches.append(m)
    return matches


def extract_match_odds(game: dict, commence: datetime) -> Optional[MatchOdds]:
    home, away = game["home_team"], game["away_team"]
    home_prices, away_prices, totals = [], [], []

    for bk in game.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt["key"] == "totals":
                totals += [o["point"] for o in mkt["outcomes"]
                           if o["name"] == "Over" and "point" in o]
                continue
            if mkt["key"] != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in mkt["outcomes"]}
            if home in outcomes and away in outcomes:
                home_prices.append(outcomes[home])
                away_prices.append(outcomes[away])

    if not home_prices:
        return None

    avg = lambda lst: round(sum(lst) / len(lst))
    return MatchOdds(
        home_team=home, away_team=away,
        ml_home=avg(home_prices), ml_away=avg(away_prices),
        commence_time=commence,
        total_line=statistics.median(totals) if totals else None,
    )


def load_manual_matches(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    matches = []
    for row in data:
        commence = None
        if row.get("commence_time"):
            commence = datetime.fromisoformat(row["commence_time"].replace("Z", "+00:00"))
        matches.append(MatchOdds(
            home_team=row["home_team"], away_team=row["away_team"],
            ml_home=int(row["ml_home"]), ml_away=int(row["ml_away"]),
            commence_time=commence,
            total_line=float(row["total"]) if row.get("total") is not None else None,
        ))
    return matches


def trim_to_week(matches: list) -> list:
    """Keep one NFL week: from the earliest game through the following Tuesday 12:00 UTC.

    The --days lookahead alone can reach into next week's Thursday game
    (the 2026-09-30 run returned 17 games, including TB @ DAL from Week 5).
    """
    dated = [m for m in matches if m.commence_time]
    if not dated:
        return matches
    first = min(m.commence_time for m in dated)
    ahead = (1 - first.weekday()) % 7 or 7          # days until Tuesday (weekday 1)
    end = (first + timedelta(days=ahead)).replace(hour=12, minute=0, second=0, microsecond=0)
    return [m for m in matches if m.commence_time is None or m.commence_time < end]


def week_number(matches: list, season_start: str) -> Optional[int]:
    """NFL week from the Tuesday before Week 1 (--season-start)."""
    dated = [m.commence_time for m in matches if m.commence_time]
    if not dated:
        return None
    start = datetime.fromisoformat(season_start).replace(tzinfo=timezone.utc)
    return (min(dated) - start).days // 7 + 1


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def outcome_label(outcome: str, m: MatchOdds) -> str:
    return {"home": m.home_team, "away": m.away_team}[outcome]


def print_picks(results: list, cfg: PoolConfig):
    print(f"NFL Picks — {len(results)} games | pool size ~{cfg.num_participants} | "
          f"differentiator budget {cfg.differentiator_budget} | chalk bias {cfg.chalk_bias:.0%}")
    print("=" * 78)

    for r in results:
        m = r.match
        kickoff = m.commence_time.strftime("%a %b %d, %I:%M %p UTC") if m.commence_time else "TBD"
        flag = "  <-- DIFFERENTIATOR" if r.is_differentiator else ""
        pick_label = outcome_label(r.final_pick, m)
        bucket = confidence_bucket(r.confidence)

        print(f"\n{m.away_team} @ {m.home_team}   ({kickoff})")
        print(f"  Pick: {pick_label}  [{bucket} confidence: {r.confidence:.0%}]{flag}")
        if r.is_differentiator:
            chalk_label = outcome_label(r.chalk, m)
            print(f"  (chalk was {chalk_label}, diff_score={r.diff_score_value:.3f})")
        breakdown = ", ".join(f"{outcome_label(k, m)}: {v:.0%}" for k, v in r.probs.items())
        print(f"  Market breakdown -> {breakdown}")

    n_diff = sum(1 for r in results if r.is_differentiator)
    print("\n" + "=" * 78)
    print(f"{n_diff}/{cfg.differentiator_budget} differentiator slots used "
          f"(no forced flips below min_edge_threshold={cfg.min_edge_threshold}).")
    print("Not betting advice — odds-implied probabilities only.")


def abbr(team: str) -> str:
    return TEAM_ABBR.get(team, team)


def text_block(results: list, week: Optional[int]) -> str:
    """Message for Junior: kickoff order (matches the sheet's game numbers) + tiebreaker.

    Tiebreaker = rounded median over/under of the last game on the slate (MNF).
    """
    lines = [f"Week {week} picks:" if week else "Picks:"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {abbr(outcome_label(r.final_pick, r.match))}")
    last = results[-1].match
    if last.total_line is not None:
        guess = int(last.total_line + 0.5)
        lines.append(f"Tiebreaker ({abbr(last.away_team)} at {abbr(last.home_team)}): {guess}")
    return "\n".join(lines)


def write_summary(results: list, week: Optional[int]):
    block = text_block(results, week)
    print("\n--- Copy to Junior ---\n" + block)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        last = results[-1].match
        with open(path, "a") as f:
            f.write("### Copy to Junior\n```\n" + block + "\n```\n")
            if last.total_line is not None:
                f.write(f"\nTiebreaker from the median over/under of {last.total_line}.\n")
            if len(results) != 16:
                f.write(f"\n**Check the slate:** {len(results)} games (bye weeks vary).\n")


# --------------------------------------------------------------------------
# Logging / backtesting
# --------------------------------------------------------------------------

def record_picks(results: list, path: str):
    week_date = datetime.now().strftime("%Y-%m-%d")
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow({
                "week_date": week_date,
                "home_team": r.match.home_team,
                "away_team": r.match.away_team,
                "chalk_pick": outcome_label(r.chalk, r.match),
                "final_pick": outcome_label(r.final_pick, r.match),
                "is_differentiator": r.is_differentiator,
                "confidence": f"{r.confidence:.4f}",
                "actual_result": "",       # fill in later: home/away
                "field_chalk_pct": "",     # fill in later from reviewing pool sheet, 0-100
            })
    print(f"Logged {len(results)} picks to {path}. "
          f"Fill in actual_result + field_chalk_pct after the week to backtest.")


def analyze_log(path: str):
    if not os.path.exists(path):
        sys.exit(f"No log found at {path}")

    with open(path) as f:
        rows = list(csv.DictReader(f))

    graded = [r for r in rows if r.get("actual_result")]
    chalk_correct = sum(1 for r in graded if r["chalk_pick"].lower() == r["actual_result"].lower())
    final_correct = sum(1 for r in graded if r["final_pick"].lower() == r["actual_result"].lower())
    diffs = [r for r in graded if r["is_differentiator"] == "True"]
    diffs_correct = sum(1 for r in diffs if r["final_pick"].lower() == r["actual_result"].lower())

    field_pcts = [float(r["field_chalk_pct"]) for r in rows if r.get("field_chalk_pct")]

    print(f"Logged weeks: {len(set(r['week_date'] for r in rows))} | Graded games: {len(graded)}")
    if graded:
        print(f"  Chalk pick accuracy:      {chalk_correct}/{len(graded)} ({chalk_correct/len(graded):.0%})")
        print(f"  Your final pick accuracy: {final_correct}/{len(graded)} ({final_correct/len(graded):.0%})")
    if diffs:
        print(f"  Differentiator hit rate:  {diffs_correct}/{len(diffs)} ({diffs_correct/len(diffs):.0%})")
    if field_pcts:
        suggested = sum(field_pcts) / len(field_pcts) / 100
        print(f"  Observed field chalk %:   avg {sum(field_pcts)/len(field_pcts):.1f}% "
              f"across {len(field_pcts)} data points")
        print(f"  Suggested chalk_bias:     {suggested:.2f}  (use --chalk-bias {suggested:.2f} going forward)")
    if not graded and not field_pcts:
        print("  No graded rows yet — fill in actual_result / field_chalk_pct columns to backtest.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="NFL differentiation-optimized weekly picks")
    p.add_argument("--api-key", default=os.environ.get("ODDS_API_KEY"))
    p.add_argument("--region", default="us")
    p.add_argument("--days", type=int, default=10)  # covers Thu-Mon slate even when run as late as Friday
    p.add_argument("--input", help="Path to manual odds JSON file (skips live API)")
    p.add_argument("--season-start", default="2026-09-08",
                   help="Tuesday before Week 1, for week numbering")

    p.add_argument("--participants", type=int, default=75)
    p.add_argument("--budget", type=int, default=3, help="differentiator_budget")
    p.add_argument("--chalk-bias", type=float, default=0.60)
    p.add_argument("--min-edge", type=float, default=0.05, help="min_edge_threshold")

    p.add_argument("--record", action="store_true", help="log this week's picks to --log")
    p.add_argument("--log", default="nfl_picks_log.csv")
    p.add_argument("--analyze-log", metavar="PATH", help="backtest/tune from a logged CSV and exit")
    return p.parse_args()


def main():
    args = get_args()

    if args.analyze_log:
        analyze_log(args.analyze_log)
        return

    if args.input:
        matches = load_manual_matches(args.input)
    else:
        if not args.api_key:
            sys.exit(
                "Error: no API key found and no --input file given. Either get a free "
                "key at https://the-odds-api.com/#get-access and set ODDS_API_KEY, "
                "or pass --input matches.json with manual odds."
            )
        matches = fetch_odds_api(args.api_key, args.region, args.days)

    if not matches:
        print("No matches with usable odds found.")
        return

    matches.sort(key=lambda m: m.commence_time or datetime.max.replace(tzinfo=timezone.utc))
    matches = trim_to_week(matches)

    cfg = PoolConfig(
        num_participants=args.participants,
        differentiator_budget=args.budget,
        chalk_bias=args.chalk_bias,
        min_edge_threshold=args.min_edge,
    )
    results = build_picks(matches, cfg)
    print_picks(results, cfg)
    write_summary(results, week_number(matches, args.season_start))

    if args.record:
        record_picks(results, args.log)


if __name__ == "__main__":
    main()
