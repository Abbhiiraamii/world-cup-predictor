"""
utils/ui_theme.py
==================
Design tokens shared across the Streamlit dashboard: a "night-match
scoreboard" theme -- deep pitch green, chalk-line dividers, and a warm
trophy-gold accent, with condensed scoreboard-style numerals for the
probability figures that are the whole point of the product.
"""

from __future__ import annotations

# ---- color tokens ---------------------------------------------------------
BG = "#0E1F17"
BG_CARD = "#14291F"
BG_CARD_ALT = "#1B3327"
GOLD = "#D4A72C"
GOLD_SOFT = "#E8C766"
CHALK = "#F2ECDD"
CHALK_MUTED = "#9FB3A6"
PITCH_LINE = "#2E5741"
WIN_GREEN = "#4C9A6A"
DRAW_GOLD = "#D4A72C"
LOSS_RED = "#C1543A"

CONFEDERATION_COLORS = {
    "UEFA": "#4C9A6A",
    "CONMEBOL": "#D4A72C",
    "CAF": "#C1543A",
    "CONCACAF": "#5B8DB8",
    "AFC": "#9A6AC9",
    "OFC": "#C97FB0",
}

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": BG_CARD,
        "plot_bgcolor": BG_CARD,
        "font": {"color": CHALK, "family": "Inter, sans-serif"},
        "colorway": [GOLD, WIN_GREEN, LOSS_RED, "#5B8DB8", "#9A6AC9", "#C97FB0", GOLD_SOFT, CHALK_MUTED],
        "xaxis": {"gridcolor": PITCH_LINE, "linecolor": PITCH_LINE, "zerolinecolor": PITCH_LINE},
        "yaxis": {"gridcolor": PITCH_LINE, "linecolor": PITCH_LINE, "zerolinecolor": PITCH_LINE},
        "legend": {"bgcolor": "rgba(0,0,0,0)"},
    }
}

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=Inter:wght@400;500;600&family=Roboto+Mono:wght@500;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background:
        repeating-linear-gradient(
            180deg,
            {BG} 0px,
            {BG} 78px,
            {BG_CARD} 78px,
            {BG_CARD} 79px
        ),
        {BG};
}}

h1, h2, h3 {{
    font-family: 'Oswald', sans-serif;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: {CHALK};
}}

h1 {{
    border-bottom: 3px solid {GOLD};
    padding-bottom: 0.3rem;
    display: inline-block;
}}

.wc-hero {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.1rem 1.4rem;
    background: linear-gradient(135deg, {BG_CARD_ALT} 0%, {BG_CARD} 100%);
    border: 1px solid {PITCH_LINE};
    border-left: 5px solid {GOLD};
    border-radius: 6px;
    margin-bottom: 1.2rem;
}}

.wc-hero .trophy {{
    font-size: 2.6rem;
    line-height: 1;
}}

.wc-hero .title {{
    font-family: 'Oswald', sans-serif;
    font-size: 1.6rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: {CHALK};
    margin: 0;
}}

.wc-hero .subtitle {{
    color: {CHALK_MUTED};
    font-size: 0.92rem;
    margin: 0;
}}

.wc-scoreboard-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0.9rem;
    margin-bottom: 0.4rem;
    background: {BG_CARD_ALT};
    border: 1px solid {PITCH_LINE};
    border-radius: 4px;
}}

.wc-scoreboard-row .rank {{
    font-family: 'Roboto Mono', monospace;
    color: {CHALK_MUTED};
    width: 2rem;
}}

.wc-scoreboard-row .team {{
    font-family: 'Oswald', sans-serif;
    font-size: 1.05rem;
    color: {CHALK};
    flex: 1;
}}

.wc-scoreboard-row .prob {{
    font-family: 'Roboto Mono', monospace;
    font-weight: 700;
    font-size: 1.15rem;
    color: {GOLD};
    min-width: 4.5rem;
    text-align: right;
}}

.wc-pill {{
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-family: 'Roboto Mono', monospace;
    border: 1px solid {PITCH_LINE};
    color: {CHALK_MUTED};
    margin-right: 0.3rem;
}}

.wc-divider {{
    border: none;
    border-top: 2px dashed {PITCH_LINE};
    margin: 1.4rem 0;
}}

.wc-explain-card {{
    background: {BG_CARD_ALT};
    border: 1px solid {PITCH_LINE};
    border-left: 4px solid {GOLD};
    border-radius: 6px;
    padding: 1rem 1.2rem;
}}

[data-testid="stMetricValue"] {{
    font-family: 'Roboto Mono', monospace;
    color: {GOLD};
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
}}

.stTabs [data-baseweb="tab"] {{
    background-color: {BG_CARD_ALT};
    border-radius: 4px 4px 0 0;
    font-family: 'Oswald', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}
</style>
"""
