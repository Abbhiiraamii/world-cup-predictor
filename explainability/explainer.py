"""
explainability/explainer.py
============================
Turns model feature-importances + a team's raw stats into a short,
human-readable explanation of why the team has the champion probability it
does -- e.g.:

    "Brazil has a 22% probability of winning because:
     - Higher Elo rating (1974 vs. field average 1780)
     - Strong recent form (4W-1D in last 5)
     - High scoring average (2.3 goals/match)
     - Excellent World Cup history (5 titles, 14 appearances)"

This is intentionally rule-based and transparent (not a second black-box
model) so every explanation is directly traceable to a number in the data --
that traceability *is* the "explainable AI" feature from the PRD.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

logger_name = __name__


@dataclass
class TeamExplanation:
    team: str
    champion_probability: float
    bullet_points: list[str]

    def as_markdown(self) -> str:
        header = f"**{self.team}** has a **{self.champion_probability * 100:.1f}%** probability of winning because:"
        bullets = "\n".join(f"- {b}" for b in self.bullet_points)
        return f"{header}\n{bullets}"


class Explainer:
    """Generates rule-based explanations grounded in the team_strength table."""

    def __init__(self, team_strength: pd.DataFrame) -> None:
        self.team_strength = team_strength.set_index("team")
        self.field_avg = {
            "elo_rating": team_strength["elo_rating"].mean(),
            "avg_goals_for": team_strength["avg_goals_for"].mean(),
            "avg_goals_against": team_strength["avg_goals_against"].mean(),
            "win_pct": team_strength["win_pct"].mean(),
            "clean_sheet_pct": team_strength["clean_sheet_pct"].mean(),
            "wc_experience": team_strength["wc_experience"].mean(),
            "form": team_strength["form"].mean(),
        }

    def explain(self, team: str, champion_probability: float, top_n: int = 4) -> TeamExplanation:
        row = self.team_strength.loc[team]
        candidates: list[tuple[float, str]] = []

        elo_gap = row["elo_rating"] - self.field_avg["elo_rating"]
        candidates.append((
            abs(elo_gap),
            f"{'Higher' if elo_gap > 0 else 'Lower'} Elo rating "
            f"({row['elo_rating']:.0f} vs. field average {self.field_avg['elo_rating']:.0f})",
        ))

        form_gap = row["form"] - self.field_avg["form"]
        recent_record = _form_to_record(row["form"])
        candidates.append((
            abs(form_gap) * 3,
            f"{'Strong' if form_gap > 0 else 'Below-average'} recent form ({recent_record} in last 5)",
        ))

        goals_gap = row["avg_goals_for"] - self.field_avg["avg_goals_for"]
        candidates.append((
            abs(goals_gap) * 1.5,
            f"{'High' if goals_gap > 0 else 'Low'} scoring average ({row['avg_goals_for']:.1f} goals/match)",
        ))

        conceded_gap = self.field_avg["avg_goals_against"] - row["avg_goals_against"]
        candidates.append((
            abs(conceded_gap) * 1.5,
            f"{'Solid' if conceded_gap > 0 else 'Shaky'} defense "
            f"({row['avg_goals_against']:.1f} goals conceded/match)",
        ))

        titles = int(row.get("world_cup_titles", 0))
        appearances = int(row.get("world_cup_appearances", 0))
        wc_gap = row["wc_experience"] - self.field_avg["wc_experience"]
        history_label = "Excellent" if titles >= 2 else ("Solid" if appearances >= 10 else "Limited")
        candidates.append((
            abs(wc_gap) * 4,
            f"{history_label} World Cup history ({titles} title(s), {appearances} appearances)",
        ))

        clean_gap = row["clean_sheet_pct"] - self.field_avg["clean_sheet_pct"]
        candidates.append((
            abs(clean_gap) * 2,
            f"{'Frequent' if clean_gap > 0 else 'Infrequent'} clean sheets "
            f"({row['clean_sheet_pct'] * 100:.0f}% of matches)",
        ))

        candidates.sort(key=lambda x: x[0], reverse=True)
        bullets = [text for _, text in candidates[:top_n]]

        return TeamExplanation(team=team, champion_probability=champion_probability, bullet_points=bullets)

    def explain_many(self, champion_prob: pd.Series, top_n: int = 4) -> dict[str, TeamExplanation]:
        return {team: self.explain(team, prob, top_n) for team, prob in champion_prob.items()}


def _form_to_record(form_points_pct: float) -> str:
    """Convert a 0-1 points-percentage back into an illustrative W-D-L-ish label."""
    wins = round(form_points_pct * 5)
    wins = max(0, min(5, wins))
    remaining = 5 - wins
    draws = remaining // 2
    losses = remaining - draws
    return f"{wins}W-{draws}D-{losses}L"
