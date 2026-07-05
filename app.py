"""
app.py
======
Streamlit dashboard for the FIFA World Cup Winner Predictor.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    APP_ICON,
    APP_TITLE,
    DEFAULT_N_SIMULATIONS,
    FOOTBALL_DATA_API_KEY,
    MAX_N_SIMULATIONS,
)
from ingestion.generate_group_draw import generate_and_save as generate_group_draw
from ingestion.generate_sample_data import generate_and_save as generate_sample_data
from ingestion import live_updates
from prediction.live_match_predictor import LiveMatchState, explain_live_state
from prediction.predictor_service import (
    artifacts_available,
    explain_team,
    get_group_draw,
    get_model_metadata,
    get_team_table,
    live_match_prediction,
    load_assets,
    run_simulation,
    single_match_prediction,
)
from utils.ui_theme import CONFEDERATION_COLORS, CUSTOM_CSS, DRAW_GOLD, GOLD, LOSS_RED, PLOTLY_TEMPLATE, WIN_GREEN

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# First-run bootstrap: build data + model automatically if missing so the app
# never greets a new user with a crash.
# ----------------------------------------------------------------------------
def _ensure_ready() -> None:
    if artifacts_available():
        return
    with st.spinner("First run detected: generating sample data and training the model..."):
        from features.build_features import run as build_features
        from features.match_outcome_model import MatchOutcomeModel

        generate_sample_data()
        generate_group_draw()
        feature_df, _ = build_features()
        model = MatchOutcomeModel()
        model.train(feature_df)
    st.success("Setup complete.")
    st.rerun()


_ensure_ready()

# ----------------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="wc-hero">
        <div class="trophy">{APP_ICON}</div>
        <div>
            <p class="title">{APP_TITLE}</p>
            <p class="subtitle">Data-driven champion probabilities via Monte Carlo tournament simulation</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metadata = get_model_metadata()
team_table = get_team_table()

with st.container():
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Qualified teams", len(team_table))
    c2.metric("Prediction model", "XGBoost" if metadata["backend"] == "xgboost" else "Gradient Boosting (sklearn)")
    c3.metric("Model accuracy", f"{metadata['test_accuracy'] * 100:.1f}%")
    c4.metric("Baseline (LogReg) accuracy", f"{metadata['baseline_accuracy'] * 100:.1f}%")

st.markdown('<hr class="wc-divider">', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_sim, tab_teams, tab_match, tab_explain, tab_model, tab_live = st.tabs(
    ["\U0001F3C6 Tournament Simulation", "\U0001F4CB Team Database", "\u2694\uFE0F Match Predictor",
     "\U0001F4A1 Explainability", "\U0001F9E0 Model Details", "\U0001F504 Live Updates"]
)

# ============================================================================
# TAB 1 -- Tournament Simulation
# ============================================================================
with tab_sim:
    st.markdown("### Run the Monte Carlo tournament simulation")
    st.caption(
        "Simulates the full 48-team World Cup (group stage \u2192 Round of 32 \u2192 "
        "Round of 16 \u2192 QF \u2192 SF \u2192 Final) thousands of times using the trained "
        "match-outcome model, then tallies how often each team wins it all."
    )

    col_a, col_b = st.columns([3, 1])
    with col_a:
        n_sims = st.slider(
            "Number of simulations", min_value=1_000, max_value=MAX_N_SIMULATIONS,
            value=DEFAULT_N_SIMULATIONS, step=1_000,
        )
    with col_b:
        st.write("")
        st.write("")
        run_clicked = st.button("\u25B6 Run Simulation", type="primary", use_container_width=True)

    if run_clicked or "sim_result" in st.session_state:
        if run_clicked:
            progress_bar = st.progress(0.0, text="Simulating tournaments...")

            def _update(frac: float) -> None:
                progress_bar.progress(min(frac, 1.0), text=f"Simulating tournaments... {frac * 100:.0f}%")

            with st.spinner("Crunching Monte Carlo trials..."):
                result = run_simulation(n_sims, progress_callback=_update)
            progress_bar.empty()
            st.session_state["sim_result"] = result

        result = st.session_state["sim_result"]
        champion_prob = result.champion_prob

        st.markdown(f"##### Champion probabilities across **{result.n_simulations:,}** simulations")

        left, right = st.columns([1.3, 1])

        with left:
            top20 = champion_prob.head(20).sort_values(ascending=True)
            fig = go.Figure(
                go.Bar(
                    x=top20.values * 100,
                    y=top20.index,
                    orientation="h",
                    marker_color=GOLD,
                    text=[f"{v * 100:.1f}%" for v in top20.values],
                    textposition="outside",
                )
            )
            fig.update_layout(
                **PLOTLY_TEMPLATE["layout"],
                height=560,
                margin=dict(l=10, r=40, t=10, b=10),
                xaxis_title="Champion probability (%)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            top6 = champion_prob.head(6)
            other = 1 - top6.sum()
            pie_labels = list(top6.index) + (["Field (all others)"] if other > 0 else [])
            pie_values = list(top6.values) + ([other] if other > 0 else [])
            pie_fig = px.pie(
                names=pie_labels, values=pie_values, hole=0.45,
                color_discrete_sequence=[GOLD, WIN_GREEN, LOSS_RED, "#5B8DB8", "#9A6AC9", "#C97FB0", "#3A4A40"],
            )
            pie_fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=380, margin=dict(l=10, r=10, t=10, b=10))
            pie_fig.update_traces(textinfo="percent+label")
            st.plotly_chart(pie_fig, use_container_width=True)

            st.markdown("###### Scoreboard: Top 5 favorites")
            for i, (team, prob) in enumerate(champion_prob.head(5).items(), start=1):
                st.markdown(
                    f"""<div class="wc-scoreboard-row">
                            <span class="rank">{i:02d}</span>
                            <span class="team">{team}</span>
                            <span class="prob">{prob * 100:.1f}%</span>
                        </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown('<hr class="wc-divider">', unsafe_allow_html=True)
        st.markdown("##### Stage-by-stage advancement probability")
        st.caption("How often each team is projected to reach each stage of the tournament.")

        stage_table = result.stage_reach_prob.head(16) * 100
        heat_fig = go.Figure(
            data=go.Heatmap(
                z=stage_table.values,
                x=stage_table.columns,
                y=stage_table.index,
                colorscale=[[0, "#14291F"], [0.5, "#4C9A6A"], [1, "#D4A72C"]],
                text=[[f"{v:.0f}%" for v in row] for row in stage_table.values],
                texttemplate="%{text}",
                colorbar=dict(title="Prob. %"),
            )
        )
        heat_fig.update_layout(
            **{k: v for k, v in PLOTLY_TEMPLATE["layout"].items() if k != "yaxis"},
            height=520, margin=dict(l=10, r=10, t=10, b=10),
            yaxis={**PLOTLY_TEMPLATE["layout"]["yaxis"], "autorange": "reversed"},
        )
        st.plotly_chart(heat_fig, use_container_width=True)

        st.markdown('<hr class="wc-divider">', unsafe_allow_html=True)
        st.markdown("##### Sample simulated knockout bracket")
        st.caption("One illustrative run from the batch above (results vary simulation to simulation).")
        bracket = result.sample_bracket
        bcols = st.columns(len(bracket))
        for col, (stage, teams) in zip(bcols, bracket.items()):
            with col:
                st.markdown(f"**{stage}**")
                for t in teams:
                    st.markdown(f"<div class='wc-pill'>{t}</div>", unsafe_allow_html=True)
    else:
        st.info("Set the number of simulations and click **Run Simulation** to generate champion probabilities.")

# ============================================================================
# TAB 2 -- Team Database
# ============================================================================
with tab_teams:
    st.markdown("### Qualified teams")
    st.caption(
        "Sample/placeholder ratings shipped with this project -- swap in real Elo/FIFA-ranking "
        "data any time (see `ingestion/DATA_SOURCES.md`)."
    )

    confeds = sorted(team_table["confederation"].unique())
    selected_confeds = st.multiselect("Filter by confederation", confeds, default=confeds)
    filtered = team_table[team_table["confederation"].isin(selected_confeds)].sort_values(
        "elo_rating", ascending=False
    )

    display_cols = [
        "team", "confederation", "fifa_rank", "elo_rating", "coach",
        "avg_squad_age", "squad_value_million_eur", "world_cup_titles",
        "world_cup_appearances", "form", "win_pct",
    ]
    st.dataframe(
        filtered[display_cols].rename(columns={
            "team": "Team", "confederation": "Confederation", "fifa_rank": "FIFA Rank",
            "elo_rating": "Elo Rating", "coach": "Coach", "avg_squad_age": "Avg. Squad Age",
            "squad_value_million_eur": "Squad Value (\u20ac M)", "world_cup_titles": "WC Titles",
            "world_cup_appearances": "WC Appearances", "form": "Form (pts %)", "win_pct": "Win %",
        }),
        use_container_width=True,
        hide_index=True,
        height=460,
    )

    st.markdown('<hr class="wc-divider">', unsafe_allow_html=True)
    st.markdown("##### Elo rating by confederation")
    elo_fig = px.strip(
        filtered, x="confederation", y="elo_rating", color="confederation",
        color_discrete_map=CONFEDERATION_COLORS, hover_name="team",
    )
    elo_fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=420, showlegend=False)
    st.plotly_chart(elo_fig, use_container_width=True)

    st.markdown("##### Group draw")
    group_draw = get_group_draw()
    group_pivot = (
        group_draw.assign(idx=group_draw.groupby("group").cumcount())
        .pivot(index="idx", columns="group", values="team")
    )
    st.dataframe(group_pivot, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 3 -- Match Predictor
# ============================================================================
with tab_match:
    st.markdown("### Head-to-head match predictor")
    st.caption("Predicts a single hypothetical match using each team's current form, Elo, and history.")

    teams_sorted = sorted(team_table["team"].tolist())
    mc1, mc2 = st.columns(2)
    with mc1:
        team_a = st.selectbox("Team A", teams_sorted, index=teams_sorted.index("Brazil") if "Brazil" in teams_sorted else 0)
    with mc2:
        default_b = "Argentina" if "Argentina" in teams_sorted else teams_sorted[1]
        team_b = st.selectbox("Team B", teams_sorted, index=teams_sorted.index(default_b))

    if team_a == team_b:
        st.warning("Choose two different teams.")
    else:
        p_a, p_draw, p_b = single_match_prediction(team_a, team_b)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{team_a} win", f"{p_a * 100:.1f}%")
        m2.metric("Draw", f"{p_draw * 100:.1f}%")
        m3.metric(f"{team_b} win", f"{p_b * 100:.1f}%")

        bar_fig = go.Figure(
            go.Bar(
                x=[f"{team_a} win", "Draw", f"{team_b} win"],
                y=[p_a * 100, p_draw * 100, p_b * 100],
                marker_color=[WIN_GREEN, DRAW_GOLD, LOSS_RED],
                text=[f"{v:.1f}%" for v in [p_a * 100, p_draw * 100, p_b * 100]],
                textposition="outside",
            )
        )
        bar_fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(bar_fig, use_container_width=True)

        st.markdown('<hr class="wc-divider">', unsafe_allow_html=True)
        st.markdown("##### \U0001F534 Live: who wins the CURRENT match?")
        st.caption(
            "If this match has already kicked off, enter the live score and minute to see "
            "the win probability updated for what's actually happened so far. This is a "
            "transparent heuristic (pre-match model + scoreline + time remaining), not a "
            "separately trained in-play model -- see `prediction/live_match_predictor.py`."
        )
        live_on = st.checkbox("This match is in progress", key="live_toggle")
        if live_on:
            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                live_home_goals = st.number_input(f"{team_a} goals", min_value=0, max_value=20, value=0)
            with lc2:
                live_away_goals = st.number_input(f"{team_b} goals", min_value=0, max_value=20, value=0)
            with lc3:
                live_minute = st.slider("Match minute", min_value=0, max_value=90, value=45)

            lp_a, lp_draw, lp_b = live_match_prediction(
                team_a, team_b, int(live_home_goals), int(live_away_goals), int(live_minute)
            )
            state = LiveMatchState(team_a, team_b, int(live_home_goals), int(live_away_goals), int(live_minute))
            st.markdown(
                f"<div class='wc-explain-card'>{explain_live_state(state, lp_a, lp_draw, lp_b)}</div>",
                unsafe_allow_html=True,
            )

            lm1, lm2, lm3 = st.columns(3)
            lm1.metric(f"{team_a} win", f"{lp_a * 100:.1f}%", delta=f"{(lp_a - p_a) * 100:+.1f} pts vs pre-match")
            lm2.metric("Draw", f"{lp_draw * 100:.1f}%", delta=f"{(lp_draw - p_draw) * 100:+.1f} pts vs pre-match")
            lm3.metric(f"{team_b} win", f"{lp_b * 100:.1f}%", delta=f"{(lp_b - p_b) * 100:+.1f} pts vs pre-match")

            live_bar_fig = go.Figure(
                go.Bar(
                    x=[f"{team_a} win", "Draw", f"{team_b} win"],
                    y=[lp_a * 100, lp_draw * 100, lp_b * 100],
                    marker_color=[WIN_GREEN, DRAW_GOLD, LOSS_RED],
                    text=[f"{v:.1f}%" for v in [lp_a * 100, lp_draw * 100, lp_b * 100]],
                    textposition="outside",
                )
            )
            live_bar_fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=340, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(live_bar_fig, use_container_width=True)

            if FOOTBALL_DATA_API_KEY:
                st.caption(
                    "Tip: check the **Live Updates** tab to auto-fetch currently live World Cup "
                    "matches instead of entering the score by hand."
                )

# ============================================================================
# TAB 4 -- Explainability
# ============================================================================
with tab_explain:
    st.markdown("### Why does a team have this probability?")
    st.caption("Rule-based explanations grounded directly in each team's underlying statistics.")

    if "sim_result" in st.session_state:
        champion_prob = st.session_state["sim_result"].champion_prob
        teams_for_explain = champion_prob.index.tolist()
    else:
        st.info("Run a simulation first for champion-probability-based explanations. Showing team-only stats below.")
        champion_prob = pd.Series(dtype=float)
        teams_for_explain = sorted(team_table["team"].tolist())

    selected_team = st.selectbox("Select a team", teams_for_explain)
    prob = float(champion_prob.get(selected_team, 0.0))
    explanation = explain_team(selected_team, prob)

    st.markdown(f"<div class='wc-explain-card'>{explanation.as_markdown()}</div>", unsafe_allow_html=True)

    st.markdown('<hr class="wc-divider">', unsafe_allow_html=True)
    st.markdown("##### Model feature importance")
    st.caption("Which statistical signals the trained model relies on most, across all matches.")
    importance = pd.Series(metadata["feature_importance"]).sort_values(ascending=True)
    imp_fig = go.Figure(go.Bar(x=importance.values, y=importance.index, orientation="h", marker_color=GOLD))
    imp_fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=460, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(imp_fig, use_container_width=True)

# ============================================================================
# TAB 5 -- Model Details
# ============================================================================
with tab_model:
    st.markdown("### Model details & evaluation")
    d1, d2, d3 = st.columns(3)
    d1.metric("Active backend", metadata["backend"])
    d2.metric("Test accuracy", f"{metadata['test_accuracy'] * 100:.1f}%")
    d3.metric("Test log-loss", f"{metadata['test_log_loss']:.3f}")

    if "draw_recall" in metadata and "macro_f1" in metadata:
        d4, d5 = st.columns(2)
        d4.metric(
            "Draw recall", f"{metadata['draw_recall'] * 100:.1f}%",
            help="Of the matches that were actually draws, what fraction did the model correctly call?",
        )
        d5.metric(
            "Macro-F1 (all 3 classes)", f"{metadata['macro_f1']:.3f}",
            help="Unweighted average F1 across Home Win / Draw / Away Win -- rewards getting the rare "
                 "draw class right, not just being right most often.",
        )
        st.caption(
            f"Class-balanced training: **{'on' if metadata.get('class_balanced_training') else 'off'}** "
            "(see `config.BALANCE_CLASSES`). Draws are the hardest, rarest outcome to call -- balancing "
            "trades a couple points of raw accuracy for the model actually predicting draws sometimes, "
            "rather than defaulting to Home/Away Win every time."
        )

    if "blend_weight" in metadata:
        st.markdown('<hr class="wc-divider">', unsafe_allow_html=True)
        st.markdown("##### Ensemble blend (primary model + Logistic Regression baseline)")
        bw = metadata["blend_weight"]
        e1, e2 = st.columns(2)
        e1.metric(
            "Blend weight", f"{bw:.2f}",
            help="1.0 = primary model only, 0.0 = baseline only. Tuned on a held-out validation "
                 "split (never the reported test metrics) by minimizing log-loss.",
        )
        e2.metric(
            "Hyperparameters tuned", "Yes" if metadata.get("hyperparams_tuned") else "No",
            help="Whether this training run searched a small grid of model hyperparameters "
                 "(config.TUNE_HYPERPARAMS) rather than using fixed defaults.",
        )
        if metadata.get("best_hyperparams"):
            st.caption(f"Selected hyperparameters: `{metadata['best_hyperparams']}`")
        st.caption(
            "Blending two differently-biased models and averaging their probabilities is a simple, "
            "well-established way to reduce variance -- see the README's "
            "'Getting more accurate predictions' section for what this measurably changes here."
        )

    st.markdown("##### Features used by the match-outcome model")
    st.write(pd.DataFrame({"feature": metadata["feature_columns"]}))

    st.markdown('<hr class="wc-divider">', unsafe_allow_html=True)
    st.markdown("##### About this project")
    st.markdown(
        """
        This dashboard predicts World Cup outcomes with a fully open-source,
        free-to-run pipeline:

        1. **Data** -- team ratings and historical match results (swap in
           real free datasets any time; see `ingestion/DATA_SOURCES.md`).
        2. **Feature engineering** -- rolling form, goals for/against, win %,
           clean-sheet %, head-to-head strength, World Cup experience,
           Elo/FIFA-rank differentials -- all computed without data leakage.
        3. **Match-outcome model** -- XGBoost (falls back to scikit-learn's
           gradient boosting automatically if `xgboost` isn't installed),
           benchmarked against a Logistic Regression baseline.
        4. **Monte Carlo simulation** -- the full 48-team bracket is played
           out thousands of times to estimate champion / stage-reach
           probabilities.
        5. **Explainability** -- every probability is paired with a
           transparent, rule-based rationale grounded in the same stats the
           model saw.
        """
    )

# ============================================================================
# TAB 6 -- Live Updates
# ============================================================================
with tab_live:
    st.markdown("### Keep predictions current with real results")
    st.caption(
        "As matches are actually played, feed the results back in. This updates each "
        "team's Elo rating, appends the result to match history, rebuilds the "
        "current-form snapshot used by the simulator, and (optionally) retrains the "
        "prediction model itself -- so accuracy keeps improving as the tournament "
        "progresses instead of relying only on pre-tournament data."
    )

    retrain_model = st.checkbox(
        "Also retrain the prediction model on the updated data (recommended)",
        value=True,
        help=(
            "Rebuilding just the team-form snapshot updates simulation inputs "
            "immediately. Retraining the model as well lets it re-learn from the "
            "growing set of real results too -- takes under a second with this "
            "dataset size, so there's little reason to leave it off."
        ),
    )

    def _apply_refresh(summary) -> None:
        if summary.diagnostic:
            lowered = summary.diagnostic.lower()
            if any(k in lowered for k in ("error", "403", "429", "not set", "rejected")):
                st.warning(f"API status: {summary.diagnostic}")
            else:
                st.info(f"API status: {summary.diagnostic}")

        if summary.n_applied > 0:
            st.success(f"Applied {summary.n_applied} new result(s) from {summary.source}.")
        else:
            st.info(f"Nothing new to apply from {summary.source}.")

        st.markdown(summary.as_markdown())
        if summary.n_applied > 0:
            load_assets.cache_clear()
            st.session_state.pop("sim_result", None)
            st.info("Model data refreshed -- re-run the simulation to see updated probabilities.")

    live_col1, live_col2 = st.columns(2)

    with live_col1:
        st.markdown("##### Option A -- Upload finished-match results")
        st.caption(
            "No signup needed. CSV columns: `date, home_team, away_team, home_goals, "
            "away_goals` (optional `tournament`). Team names must match the Team "
            "Database tab exactly."
        )
        uploaded = st.file_uploader("Upload results CSV", type=["csv"])
        if uploaded is not None and st.button("Apply uploaded results"):
            try:
                new_df = pd.read_csv(uploaded)
                summary = live_updates.refresh_from_dataframe(new_df, retrain=retrain_model)
                _apply_refresh(summary)
            except ValueError as exc:
                st.error(str(exc))

        st.markdown("---")
        st.caption(
            "Alternatively, drop rows into `data/raw/live_results_manual.csv` "
            "(see `data/raw/live_results_manual.csv.example`) and click below."
        )
        if st.button("Refresh from data/raw/live_results_manual.csv"):
            summary = live_updates.refresh_from_manual_csv(retrain=retrain_model)
            _apply_refresh(summary)

    with live_col2:
        st.markdown("##### Option B -- Free live-score API (football-data.org)")
        if FOOTBALL_DATA_API_KEY:
            st.success("API key detected -- ready to fetch.")
        else:
            st.warning(
                "No API key set. Get a free key at "
                "[football-data.org/client/register](https://www.football-data.org/client/register) "
                "(free forever for the World Cup competition) and set it as the "
                "`FOOTBALL_DATA_API_KEY` environment variable, then restart the app."
            )
        date_from = st.date_input("Fetch results from date (optional -- leave blank for all-time)", value=None)
        if st.button("Fetch latest World Cup results", disabled=not bool(FOOTBALL_DATA_API_KEY)):
            with st.spinner("Contacting football-data.org..."):
                summary = live_updates.refresh_from_api(
                    date_from=str(date_from) if date_from else None,
                    retrain=retrain_model,
                )
            _apply_refresh(summary)
        st.caption(
            "Tip: if you get 0 results, try clearing the date field first to confirm the "
            "API/competition is returning data at all, then narrow the date range."
        )

    st.markdown('<hr class="wc-divider">', unsafe_allow_html=True)
    st.caption(
        "Both options are safe to re-run: results are matched on (date, home team, away "
        "team), so re-uploading or re-fetching the same match never double-counts it."
    )

    st.markdown('<hr class="wc-divider">', unsafe_allow_html=True)
    st.markdown("##### \U0001F534 Currently live matches")
    if not FOOTBALL_DATA_API_KEY:
        st.info("Set `FOOTBALL_DATA_API_KEY` to auto-list matches in progress right now.")
    else:
        if st.button("Check for live matches"):
            with st.spinner("Checking football-data.org..."):
                live_df = live_updates.fetch_in_play_matches()
            if live_df.empty:
                st.info("No World Cup matches are currently live.")
            else:
                st.dataframe(live_df, use_container_width=True, hide_index=True)
                st.caption(
                    "Copy a team pairing and current score into the **Match Predictor** tab's "
                    "live section to see the in-progress win probability."
                )
