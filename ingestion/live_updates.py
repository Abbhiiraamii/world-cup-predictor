"""
ingestion/live_updates.py
==========================
Keeps the predictor current with real match results as they happen, so
predictions improve as the tournament progresses instead of relying only on
pre-tournament data.

Two free, no-paid-plan input paths (use either or both):

1. **Manual CSV** (always available, no signup): drop finished-match rows
   into ``data/raw/live_results_manual.csv`` (see
   ``live_results_manual.csv.example`` for the exact format) and call
   ``refresh_from_manual_csv()``.
2. **football-data.org free tier** (free API key, no cost): covers the World
   Cup competition (code "WC") among others. Set the ``FOOTBALL_DATA_API_KEY``
   environment variable, then call ``refresh_from_api()``.

Both paths do the same three things:
    a) Validate + append new results to ``historical_matches.csv``
    b) Update each affected team's Elo rating in ``teams.csv`` using the
       standard Elo formula (see ``features/elo.py``)
    c) Rebuild engineered features + the "current form" snapshot so the next
       simulation run reflects the new results immediately
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import requests

from config import (
    FOOTBALL_DATA_API_KEY,
    FOOTBALL_DATA_BASE_URL,
    HISTORICAL_MATCHES_CSV,
    MANUAL_LIVE_RESULTS_CSV,
    TEAMS_CSV,
    WC_COMPETITION_CODE,
)
from features.elo import result_to_score, update_elo
from utils.logger import get_logger

logger = get_logger(__name__, log_file="live_updates.log")

REQUIRED_COLUMNS = ["date", "home_team", "away_team", "home_goals", "away_goals"]

# football-data.org uses different display names for some sides than our
# teams.csv. Extend this as needed for teams you track.
NAME_ALIASES = {
    "Korea Republic": "South Korea",
    "USA": "United States",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Cabo Verde": "Cape Verde",
}


@dataclass
class RefreshSummary:
    source: str
    n_fetched: int = 0
    n_applied: int = 0
    n_skipped_unknown_team: int = 0
    elo_changes: dict[str, tuple[float, float]] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    diagnostic: str | None = None  # what actually happened on the wire (API path only)

    def as_markdown(self) -> str:
        lines = [f"**Source:** {self.source}", f"**New results fetched:** {self.n_fetched}",
                 f"**Applied:** {self.n_applied}", f"**Skipped (unrecognized team):** {self.n_skipped_unknown_team}"]
        if self.diagnostic:
            lines.append(f"\n**API status:** {self.diagnostic}")
        if self.elo_changes:
            lines.append("\n**Elo changes:**")
            for team, (old, new) in self.elo_changes.items():
                arrow = "\u2191" if new > old else ("\u2193" if new < old else "\u2192")
                lines.append(f"- {team}: {old:.0f} {arrow} {new:.0f}")
        for m in self.messages:
            lines.append(f"\n_{m}_")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Input sources
# ---------------------------------------------------------------------------
def load_manual_csv() -> pd.DataFrame:
    """Load user-provided finished-match results, if the file exists."""
    if not MANUAL_LIVE_RESULTS_CSV.exists():
        logger.info("No manual live-results file at %s", MANUAL_LIVE_RESULTS_CSV)
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["tournament"])

    df = pd.read_csv(MANUAL_LIVE_RESULTS_CSV)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{MANUAL_LIVE_RESULTS_CSV.name} is missing columns: {missing}")
    if "tournament" not in df.columns:
        df["tournament"] = "World Cup"
    return df


def fetch_football_data_org(
    date_from: str | None = None, date_to: str | None = None,
    competition_code: str = WC_COMPETITION_CODE,
) -> tuple[pd.DataFrame, str]:
    """Fetch FINISHED matches from football-data.org's free tier.

    Returns (DataFrame, diagnostic_message). The DataFrame is empty on any
    error (missing/invalid key, rate limit, network problem) -- that never
    crashes the app -- but the diagnostic message always says exactly what
    happened, so "the request failed" and "the request succeeded but found
    nothing" are never confused with each other in the UI.
    """
    empty = pd.DataFrame(columns=REQUIRED_COLUMNS + ["tournament"])

    if not FOOTBALL_DATA_API_KEY:
        msg = (
            "FOOTBALL_DATA_API_KEY is not set. Get a free key at "
            "https://www.football-data.org/client/register and set it as an "
            "environment variable to enable live API fetching."
        )
        logger.warning(msg)
        return empty, msg

    url = f"{FOOTBALL_DATA_BASE_URL}/competitions/{competition_code}/matches"
    params = {"status": "FINISHED"}
    if date_from:
        params["dateFrom"] = date_from
    if date_to:
        params["dateTo"] = date_to

    try:
        resp = requests.get(
            url, headers={"X-Auth-Token": FOOTBALL_DATA_API_KEY}, params=params, timeout=15
        )
    except requests.RequestException as exc:
        msg = f"Network error contacting football-data.org: {exc}"
        logger.error(msg)
        return empty, msg

    if resp.status_code == 429:
        msg = "football-data.org rate limit hit (free tier: 10 requests/minute). Wait a minute and try again."
        logger.warning(msg)
        return empty, msg
    if resp.status_code == 403:
        msg = (
            "football-data.org rejected the API key (HTTP 403). Double-check "
            "FOOTBALL_DATA_API_KEY is correct, fully activated (check your email "
            "for a confirmation link if you just registered), and restart the app "
            "after setting it."
        )
        logger.error(msg)
        return empty, msg
    if resp.status_code == 404:
        msg = f"football-data.org has no competition with code '{competition_code}' (HTTP 404)."
        logger.error(msg)
        return empty, msg
    if resp.status_code != 200:
        msg = f"football-data.org returned HTTP {resp.status_code}: {resp.text[:200]}"
        logger.error(msg)
        return empty, msg

    payload = resp.json()
    rows = []
    for m in payload.get("matches", []):
        score = m.get("score", {}).get("fullTime", {})
        home_goals, away_goals = score.get("home"), score.get("away")
        if home_goals is None or away_goals is None:
            continue
        home_name = _normalize_team_name(m["homeTeam"]["name"])
        away_name = _normalize_team_name(m["awayTeam"]["name"])
        rows.append({
            "date": m["utcDate"][:10],
            "home_team": home_name,
            "away_team": away_name,
            "home_goals": int(home_goals),
            "away_goals": int(away_goals),
            "tournament": "World Cup",
        })

    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS + ["tournament"])
    n_total_returned = len(payload.get("matches", []))
    if df.empty:
        msg = (
            f"Request succeeded (HTTP 200), but 0 FINISHED matches matched your filters "
            f"(API returned {n_total_returned} match(es) total in other statuses, e.g. "
            f"SCHEDULED/LIVE). Try removing the date filter or widening it."
        )
    else:
        msg = f"Request succeeded: {len(df)} finished match(es) returned."
    logger.info("Fetched %d finished matches from football-data.org (%s)", len(df), competition_code)
    return df, msg


def _normalize_team_name(name: str) -> str:
    return NAME_ALIASES.get(name, name)


def fetch_in_play_matches(competition_code: str = WC_COMPETITION_CODE) -> pd.DataFrame:
    """Fetch matches currently IN_PLAY or PAUSED ('LIVE') from football-data.org.

    Returns columns: home_team, away_team, home_goals, away_goals, minute, status.
    Returns an empty DataFrame (with a logged reason) on any error, same
    fail-safe behaviour as ``fetch_football_data_org``.
    """
    empty = pd.DataFrame(columns=["home_team", "away_team", "home_goals", "away_goals", "minute", "status"])

    if not FOOTBALL_DATA_API_KEY:
        logger.warning("FOOTBALL_DATA_API_KEY is not set -- cannot fetch live matches.")
        return empty

    url = f"{FOOTBALL_DATA_BASE_URL}/competitions/{competition_code}/matches"
    try:
        resp = requests.get(
            url, headers={"X-Auth-Token": FOOTBALL_DATA_API_KEY}, params={"status": "LIVE"}, timeout=15
        )
    except requests.RequestException as exc:
        logger.error("Network error contacting football-data.org: %s", exc)
        return empty

    if resp.status_code != 200:
        logger.error("football-data.org returned HTTP %d for live matches: %s", resp.status_code, resp.text[:200])
        return empty

    rows = []
    for m in resp.json().get("matches", []):
        score = m.get("score", {}).get("fullTime", {})
        rows.append({
            "home_team": _normalize_team_name(m["homeTeam"]["name"]),
            "away_team": _normalize_team_name(m["awayTeam"]["name"]),
            "home_goals": score.get("home") or 0,
            "away_goals": score.get("away") or 0,
            "minute": m.get("minute") or 0,
            "status": m.get("status", ""),
        })
    df = pd.DataFrame(rows, columns=["home_team", "away_team", "home_goals", "away_goals", "minute", "status"])
    logger.info("Fetched %d live match(es) from football-data.org (%s)", len(df), competition_code)
    return df


# ---------------------------------------------------------------------------
# Applying updates
# ---------------------------------------------------------------------------
def _known_teams() -> set[str]:
    return set(pd.read_csv(TEAMS_CSV)["team"].tolist())


def _filter_known_teams(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    known = _known_teams()
    mask = df["home_team"].isin(known) & df["away_team"].isin(known)
    skipped = int((~mask).sum())
    if skipped:
        logger.warning("Skipping %d fetched match(es) with unrecognized team names.", skipped)
    return df[mask].copy(), skipped


def apply_elo_updates(new_matches: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Replay new matches chronologically, updating teams.csv Elo ratings."""
    if new_matches.empty:
        return {}

    teams_df = pd.read_csv(TEAMS_CSV).set_index("team")
    elo_before: dict[str, float] = {}
    changes: dict[str, tuple[float, float]] = {}

    for row in new_matches.sort_values("date").itertuples(index=False):
        home, away = row.home_team, row.away_team
        elo_before.setdefault(home, teams_df.loc[home, "elo_rating"])
        elo_before.setdefault(away, teams_df.loc[away, "elo_rating"])

        elo_h = teams_df.loc[home, "elo_rating"]
        elo_a = teams_df.loc[away, "elo_rating"]
        score_h = result_to_score(row.home_goals, row.away_goals)
        new_elo_h, new_elo_a = update_elo(elo_h, elo_a, score_h)

        teams_df.loc[home, "elo_rating"] = round(new_elo_h, 1)
        teams_df.loc[away, "elo_rating"] = round(new_elo_a, 1)

    for team, before in elo_before.items():
        after = teams_df.loc[team, "elo_rating"]
        if after != before:
            changes[team] = (before, after)

    teams_df.reset_index().to_csv(TEAMS_CSV, index=False)
    logger.info("Updated Elo ratings for %d teams.", len(changes))
    return changes


def _dedupe_against_existing(new_matches: pd.DataFrame) -> pd.DataFrame:
    """Return only rows in ``new_matches`` not already present in
    historical_matches.csv, keyed on (date, home_team, away_team).

    This MUST run before Elo updates are applied -- otherwise re-running a
    refresh with the same input (e.g. re-uploading the same CSV) would
    silently re-apply Elo adjustments for matches already recorded.
    """
    existing = pd.read_csv(HISTORICAL_MATCHES_CSV)[["date", "home_team", "away_team"]].copy()
    existing["date"] = pd.to_datetime(existing["date"]).dt.strftime("%Y-%m-%d")

    candidates = new_matches.copy()
    candidates["date"] = pd.to_datetime(candidates["date"]).dt.strftime("%Y-%m-%d")

    merged = candidates.merge(
        existing, on=["date", "home_team", "away_team"], how="left", indicator=True
    )
    new_only = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    n_dupe = len(candidates) - len(new_only)
    if n_dupe:
        logger.info("Ignoring %d match(es) already present in historical_matches.csv", n_dupe)
    return new_only


def append_to_historical(new_matches: pd.DataFrame) -> int:
    """Append already-deduplicated new results to historical_matches.csv.
    Returns the number of rows added."""
    if new_matches.empty:
        return 0

    existing = pd.read_csv(HISTORICAL_MATCHES_CSV)
    combined = pd.concat([existing, new_matches[REQUIRED_COLUMNS + ["tournament"]]], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date")
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined.to_csv(HISTORICAL_MATCHES_CSV, index=False)
    return len(new_matches)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _finalize(source: str, fetched: pd.DataFrame, diagnostic: str | None = None,
              retrain: bool = False) -> RefreshSummary:
    summary = RefreshSummary(source=source, n_fetched=len(fetched), diagnostic=diagnostic)

    filtered, skipped = _filter_known_teams(fetched)
    summary.n_skipped_unknown_team = skipped

    if filtered.empty:
        summary.messages.append("No new, recognizable match results to apply.")
        return summary

    new_only = _dedupe_against_existing(filtered)
    if new_only.empty:
        summary.messages.append("All fetched results were already applied previously -- nothing to do.")
        return summary

    summary.elo_changes = apply_elo_updates(new_only)
    summary.n_applied = append_to_historical(new_only)

    # Rebuild engineered features + current-form snapshot so simulations
    # immediately reflect the new results.
    from features.build_features import run as build_features
    feature_df, _ = build_features()

    summary.messages.append(
        f"Rebuilt features and team-strength snapshot from {summary.n_applied} new match(es)."
    )

    if retrain:
        from features.match_outcome_model import MatchOutcomeModel
        model = MatchOutcomeModel()
        # tune=False: a full grid search takes ~30-60s, too slow for a
        # button click after every live update. Run `python train_model.py`
        # periodically (uses config.TUNE_HYPERPARAMS) to get the fully-tuned
        # model; this fast path re-fits with the existing hyperparameters.
        result = model.train(feature_df, tune=False)
        summary.messages.append(
            f"Retrained model on {len(feature_df)} total matches (accuracy now "
            f"{result.test_accuracy * 100:.1f}%, backend: {result.backend}). "
            f"Run `python train_model.py` periodically for a full hyperparameter re-tune."
        )

    return summary


def refresh_from_manual_csv(retrain: bool = False) -> RefreshSummary:
    fetched = load_manual_csv()
    return _finalize("manual CSV", fetched, retrain=retrain)


def refresh_from_api(date_from: str | None = None, date_to: str | None = None,
                      retrain: bool = False) -> RefreshSummary:
    fetched, diagnostic = fetch_football_data_org(date_from=date_from, date_to=date_to)
    return _finalize("football-data.org (WC)", fetched, diagnostic=diagnostic, retrain=retrain)


def refresh_from_dataframe(df: pd.DataFrame, retrain: bool = False) -> RefreshSummary:
    """Apply an already-loaded DataFrame of new results (e.g. from a
    Streamlit file-uploader) without touching disk first."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Uploaded results are missing columns: {missing}")
    if "tournament" not in df.columns:
        df = df.copy()
        df["tournament"] = "World Cup"
    return _finalize("uploaded file", df, retrain=retrain)
