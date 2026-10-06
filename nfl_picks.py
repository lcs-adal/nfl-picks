From a06bd5c0a57496fd26f5721e0e17d60b72690ce7 Mon Sep 17 00:00:00 2001
From: Adal Becerra <adal@users.noreply.github.com>
Date: Sun, 4 Oct 2026 22:30:44 +0000
Subject: [PATCH] Trim slate to one NFL week, market tiebreaker, copy-to-Junior
 summary

---
 nfl_picks.py | 88 ++++++++++++++++++++++++++++++++++++++++++++++++++--
 1 file changed, 86 insertions(+), 2 deletions(-)

diff --git a/nfl_picks.py b/nfl_picks.py
index 45ee34c..0f37367 100644
--- a/nfl_picks.py
+++ b/nfl_picks.py
@@ -44,10 +44,26 @@ from datetime import datetime, timedelta, timezone
 from typing import Optional
 
 import requests
+import statistics
 
 SPORT_KEY = "americanfootball_nfl"
 BASE_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
 
+# Pool sheet abbreviations (match the family pool's xlsx)
+TEAM_ABBR = {
+    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
+    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
+    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
+    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
+    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
+    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LVR", "Los Angeles Chargers": "LAC",
+    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
+    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
+    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
+    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
+    "Tennessee Titans": "TEN", "Washington Commanders": "WSH",
+}
+
 LOG_FIELDS = [
     "week_date", "home_team", "away_team", "chalk_pick", "final_pick",
     "is_differentiator", "confidence", "actual_result", "field_chalk_pct",
@@ -65,6 +81,7 @@ class MatchOdds:
     ml_home: int       # moneyline, e.g. -145
     ml_away: int         # e.g. +122
     commence_time: Optional[datetime] = None
+    total_line: Optional[float] = None   # median over/under across books
 
 
 @dataclass
@@ -192,7 +209,7 @@ def fetch_odds_api(api_key: str, region: str, days: int) -> list:
     params = {
         "apiKey": api_key,
         "regions": region,
-        "markets": "h2h",
+        "markets": "h2h,totals",   # 2 credits per run (2 markets x 1 region)
         "oddsFormat": "american",
         "dateFormat": "iso",
     }
@@ -222,10 +239,14 @@ def fetch_odds_api(api_key: str, region: str, days: int) -> list:
 
 def extract_match_odds(game: dict, commence: datetime) -> Optional[MatchOdds]:
     home, away = game["home_team"], game["away_team"]
-    home_prices, away_prices = [], []
+    home_prices, away_prices, totals = [], [], []
 
     for bk in game.get("bookmakers", []):
         for mkt in bk.get("markets", []):
+            if mkt["key"] == "totals":
+                totals += [o["point"] for o in mkt["outcomes"]
+                           if o["name"] == "Over" and "point" in o]
+                continue
             if mkt["key"] != "h2h":
                 continue
             outcomes = {o["name"]: o["price"] for o in mkt["outcomes"]}
@@ -241,6 +262,7 @@ def extract_match_odds(game: dict, commence: datetime) -> Optional[MatchOdds]:
         home_team=home, away_team=away,
         ml_home=avg(home_prices), ml_away=avg(away_prices),
         commence_time=commence,
+        total_line=statistics.median(totals) if totals else None,
     )
 
 
@@ -256,10 +278,35 @@ def load_manual_matches(path: str) -> list:
             home_team=row["home_team"], away_team=row["away_team"],
             ml_home=int(row["ml_home"]), ml_away=int(row["ml_away"]),
             commence_time=commence,
+            total_line=float(row["total"]) if row.get("total") is not None else None,
         ))
     return matches
 
 
+def trim_to_week(matches: list) -> list:
+    """Keep one NFL week: from the earliest game through the following Tuesday 12:00 UTC.
+
+    The --days lookahead alone can reach into next week's Thursday game
+    (the 2026-09-30 run returned 17 games, including TB @ DAL from Week 5).
+    """
+    dated = [m for m in matches if m.commence_time]
+    if not dated:
+        return matches
+    first = min(m.commence_time for m in dated)
+    ahead = (1 - first.weekday()) % 7 or 7          # days until Tuesday (weekday 1)
+    end = (first + timedelta(days=ahead)).replace(hour=12, minute=0, second=0, microsecond=0)
+    return [m for m in matches if m.commence_time is None or m.commence_time < end]
+
+
+def week_number(matches: list, season_start: str) -> Optional[int]:
+    """NFL week from the Tuesday before Week 1 (--season-start)."""
+    dated = [m.commence_time for m in matches if m.commence_time]
+    if not dated:
+        return None
+    start = datetime.fromisoformat(season_start).replace(tzinfo=timezone.utc)
+    return (min(dated) - start).days // 7 + 1
+
+
 # --------------------------------------------------------------------------
 # Output
 # --------------------------------------------------------------------------
@@ -295,6 +342,39 @@ def print_picks(results: list, cfg: PoolConfig):
     print("Not betting advice — odds-implied probabilities only.")
 
 
+def abbr(team: str) -> str:
+    return TEAM_ABBR.get(team, team)
+
+
+def text_block(results: list, week: Optional[int]) -> str:
+    """Message for Junior: kickoff order (matches the sheet's game numbers) + tiebreaker.
+
+    Tiebreaker = rounded median over/under of the last game on the slate (MNF).
+    """
+    lines = [f"Week {week} picks:" if week else "Picks:"]
+    for i, r in enumerate(results, 1):
+        lines.append(f"{i}. {abbr(outcome_label(r.final_pick, r.match))}")
+    last = results[-1].match
+    if last.total_line is not None:
+        guess = int(last.total_line + 0.5)
+        lines.append(f"Tiebreaker ({abbr(last.away_team)} at {abbr(last.home_team)}): {guess}")
+    return "\n".join(lines)
+
+
+def write_summary(results: list, week: Optional[int]):
+    block = text_block(results, week)
+    print("\n--- Copy to Junior ---\n" + block)
+    path = os.environ.get("GITHUB_STEP_SUMMARY")
+    if path:
+        last = results[-1].match
+        with open(path, "a") as f:
+            f.write("### Copy to Junior\n```\n" + block + "\n```\n")
+            if last.total_line is not None:
+                f.write(f"\nTiebreaker from the median over/under of {last.total_line}.\n")
+            if len(results) != 16:
+                f.write(f"\n**Check the slate:** {len(results)} games (bye weeks vary).\n")
+
+
 # --------------------------------------------------------------------------
 # Logging / backtesting
 # --------------------------------------------------------------------------
@@ -362,6 +442,8 @@ def get_args():
     p.add_argument("--region", default="us")
     p.add_argument("--days", type=int, default=10)  # covers Thu-Mon slate even when run as late as Friday
     p.add_argument("--input", help="Path to manual odds JSON file (skips live API)")
+    p.add_argument("--season-start", default="2026-09-08",
+                   help="Tuesday before Week 1, for week numbering")
 
     p.add_argument("--participants", type=int, default=75)
     p.add_argument("--budget", type=int, default=3, help="differentiator_budget")
@@ -397,6 +479,7 @@ def main():
         return
 
     matches.sort(key=lambda m: m.commence_time or datetime.max.replace(tzinfo=timezone.utc))
+    matches = trim_to_week(matches)
 
     cfg = PoolConfig(
         num_participants=args.participants,
@@ -406,6 +489,7 @@ def main():
     )
     results = build_picks(matches, cfg)
     print_picks(results, cfg)
+    write_summary(results, week_number(matches, args.season_start))
 
     if args.record:
         record_picks(results, args.log)
-- 
2.43.0
