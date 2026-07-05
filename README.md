# \U0001F3C6 FIFA World Cup Winner Predictor

A data-driven, fully open-source web application that predicts each qualified
team's probability of winning the FIFA World Cup using machine learning and
Monte Carlo tournament simulation -- with transparent, explainable output
instead of a single naive pick.

Built entirely with free datasets and open-source libraries. No paid APIs,
no API keys, no proprietary data.

---

## What it does

1. **Team Database** -- browse all 48 qualified teams with FIFA ranking, Elo
   rating, confederation, coach, squad value/age, and recent form.
2. **Match Predictor** -- pick any two teams and see Home Win / Draw / Away
   Win probabilities for a hypothetical match, plus a **live in-play mode**:
   enter the current score and match minute for a match already underway to
   see the win probability adjusted for what's actually happened so far.
3. **Tournament Simulation** -- run 1,000-100,000 Monte Carlo simulations of
   the entire 48-team bracket (group stage through the Final) and see:
   - Champion probability for every team
   - Probability of reaching each stage (R32, R16, QF, SF, Final)
   - A sample simulated knockout bracket
4. **Explainability** -- every probability comes with a rule-based, plainly
   traceable explanation ("Brazil has a 22% probability of winning
   because: higher Elo rating, strong recent form, ...").
5. **Model Details** -- accuracy, log-loss, and feature importance for the
   trained model vs. the Logistic Regression baseline.
6. **Live Updates** -- feed in real match results (manual CSV upload, or a
   free football-data.org API key) and the app updates each team's Elo
   rating, appends the result to match history, and rebuilds the
   current-form snapshot -- so predictions get more accurate as the real
   tournament unfolds, instead of being frozen at pre-tournament data.

---

## Tech stack

| Layer          | Choice                                                  |
|----------------|----------------------------------------------------------|
| Data           | Pandas / NumPy, CSV storage                              |
| ML model       | XGBoost (primary), Logistic Regression (baseline)        |
| ML fallback    | scikit-learn `HistGradientBoostingClassifier` if `xgboost` isn't installed |
| Simulation     | Custom Monte Carlo engine (NumPy)                        |
| Frontend       | Streamlit                                                 |
| Charts         | Plotly                                                    |
| Tests          | pytest                                                    |

---

## Project structure

```
world-cup-predictor/
├── config.py                      # all paths, constants, hyperparameters
├── app.py                         # Streamlit dashboard (entry point)
├── train_model.py                 # CLI: build data -> features -> train
├── requirements.txt
│
├── ingestion/
│   ├── generate_sample_data.py    # synthetic-but-realistic sample data
│   ├── generate_group_draw.py     # seeds 48 teams into 12 groups of 4
│   └── DATA_SOURCES.md            # where to get REAL free data
│
├── features/
│   ├── preprocessing.py           # load/clean/validate raw CSVs
│   ├── feature_engineering.py     # leakage-free rolling stats, H2H, etc.
│   ├── build_features.py          # orchestrates the two modules above
│   └── match_outcome_model.py     # XGBoost + LogReg baseline + fallback
│
├── simulation/
│   └── tournament_simulator.py    # Monte Carlo full-bracket simulator
│
├── explainability/
│   └── explainer.py               # rule-based, traceable explanations
│
├── prediction/
│   ├── predictor_service.py       # thin service layer used by app.py
│   └── live_match_predictor.py    # in-play win probability heuristic
│
├── utils/
│   ├── logger.py                  # shared logging config
│   └── ui_theme.py                # Streamlit CSS + Plotly color theme
│
├── data/
│   ├── raw/                       # teams.csv, historical_matches.csv, tournament_fixtures.csv
│   └── processed/                 # match_features.csv, team_strength.csv
│
├── models/                        # trained model + metadata (generated)
├── tests/                         # pytest unit tests
└── notebooks/                     # exploratory analysis (empty by default)
```

---

## Setup

Requires **Python 3.11+**.

```bash
# 1. Clone / unzip the project, then from the project root:
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (First time only) generate sample data & train the model
python train_model.py

# 4. Launch the dashboard
streamlit run app.py
```

The app also auto-bootstraps on first launch (generates sample data and
trains a model automatically) if you skip step 3 -- but running
`train_model.py` yourself lets you see the evaluation report in your
terminal.

---

## Using real data instead of the sample dataset

This project ships with **synthetic-but-statistically-realistic** sample
data (`data/raw/teams.csv`, `data/raw/historical_matches.csv`) generated by
`ingestion/generate_sample_data.py`, so it runs immediately with zero setup.

To use real match history and ratings:

1. See `ingestion/DATA_SOURCES.md` for a list of free sources (Kaggle
   international-results dataset, clubelo.com, eloratings.net, official FIFA
   rankings, etc.) -- no paid APIs required anywhere.
2. Download and place CSVs in `data/raw/` matching the expected column
   schema (documented in that file and in `features/preprocessing.py`).
3. Re-run `python train_model.py` to rebuild features and retrain on the
   new data.

---

## Running tests

```bash
pytest
```

Tests cover:
- Feature engineering correctness and **no data-leakage guarantee** (a
  match's features can only reflect information available before that
  match's date)
- Model training, prediction, and batch/single prediction consistency
- Monte Carlo simulation invariants (probabilities sum to 1, stage-reach
  probabilities are monotonically non-increasing, results are reproducible
  given a fixed seed, the strongest team is favored)
- Sample data generation and preprocessing/label logic

---

## Design notes

- **No data leakage**: match-outcome features (form, goals, win %, etc.) are
  computed match-by-match in chronological order, using only information
  known strictly before that match. See `features/feature_engineering.py`.
- **Fast simulation**: match probabilities only depend on each team's
  *current* form snapshot, which doesn't change across trials -- so every
  pairwise probability is precomputed once (batch model call) and Monte
  Carlo trials become cheap cached lookups. 10,000 full-tournament
  simulations run in roughly 1-2 seconds, comfortably under the PRD's
  one-minute requirement even at 100,000 simulations.
- **Automatic model fallback**: if `xgboost` fails to install on your
  platform, `features/match_outcome_model.py` transparently swaps in
  scikit-learn's `HistGradientBoostingClassifier` (a similar gradient
  boosted tree algorithm) and records which backend was actually used in
  `models/model_metadata.json`, so the dashboard's "Model Details" tab is
  always accurate about what's running.
- **Explainability is rule-based, not a second model**: every explanation
  bullet is generated directly from a real number in `team_strength.csv`
  compared against the field average, so it's always traceable and never
  hallucinated.

---

## Live in-play win probability ("who wins the current match")

The **Match Predictor** tab has a "This match is in progress" toggle: enter
the current score and match minute, and it shows the win probability
adjusted for what's actually happened so far, plus how much it moved versus
the pre-match number.

**How it works (and its honest limits):** `prediction/live_match_predictor.py`
combines the pre-match model probability with the current scoreline and time
remaining using a transparent, hand-built heuristic -- a lead is weighted
more heavily the less time is left, and the draw probability rises toward a
ceiling as full time approaches if the score is level (or shrinks toward
zero if someone's already ahead). This is **not** a second trained model --
there's no free, ready-made minute-by-minute in-play training dataset to
build one from -- but it's directionally sensible and fully described in the
module's docstring, and it recovers the exact pre-match probability at
minute 0 with a 0-0 scoreline.

If you have a `FOOTBALL_DATA_API_KEY` set, the Live Updates tab can also
list matches currently in progress so you don't have to enter the score by
hand.

---

## Keeping predictions current with live results

Predictions shouldn't freeze at pre-tournament data. As real matches are
played, feed the results back in via the app's **Live Updates** tab (or the
functions in `ingestion/live_updates.py`):

- **Manual CSV (always free, no signup)**: upload a CSV of finished matches
  (`date, home_team, away_team, home_goals, away_goals`) right in the
  dashboard, or drop rows into `data/raw/live_results_manual.csv` (template:
  `data/raw/live_results_manual.csv.example`).
- **football-data.org free tier**: covers the World Cup competition free
  forever (10 req/min). Get a free key at
  [football-data.org/client/register](https://www.football-data.org/client/register),
  set it as the `FOOTBALL_DATA_API_KEY` environment variable, and click
  "Fetch latest World Cup results" in the app.

Either path automatically:
1. Updates both teams' Elo ratings using the standard Elo formula (a win
   updates the winner up and loser down; a draw against a much stronger
   side raises the underdog's rating).
2. Appends the result to `historical_matches.csv`.
3. Rebuilds engineered features and the "current form" snapshot used by the
   simulator.
4. **Optionally retrains the prediction model itself** on the growing
   dataset (checked by default in the Live Updates tab -- takes under a
   second at this dataset size), so accuracy keeps improving as more real
   results come in, not just the simulator's input stats.

Re-uploading or re-fetching the same match is always safe -- results are
matched on (date, home team, away team), so nothing is ever double-counted.
See `ingestion/DATA_SOURCES.md` for full details.

### Getting more accurate predictions

**Context worth having before chasing a number:** for 3-way football outcomes
(Home/Draw/Away), professional bookmakers and published football models
typically land around 50-55% accuracy -- not because the modeling is weak,
but because a single match has very few goal events, so a small amount of
randomness (a deflection, a call, a missed sitter) has an outsized effect on
the result. Random guessing gets ~33%. This ceiling applies to any model,
including this one -- so "50%" is already a real, useful signal, not a sign
something's broken.

With that said, this project measurably improved accuracy through legitimate
means, and reports the before/after honestly rather than just claiming it:

- **Class-balanced training** (`config.BALANCE_CLASSES = True`): draws are
  the rarest, hardest outcome, and an unbalanced model tends to almost never
  predict one. Balancing trades a couple points of raw accuracy for a real
  jump in draw recall and macro-F1.
- **Hyperparameter tuning** (`config.TUNE_HYPERPARAMS = True`): a small
  cross-validated grid search over tree depth, learning rate, and estimator
  count, scored by log-loss. Measured on this project's sample data: primary
  model log-loss improved from 1.024 to ~1.01, and it takes ~30-60 seconds,
  so `train_model.py` runs it by default but live-update retrains use
  `tune=False` to stay fast (see below).
- **Ensemble blending**: the primary model's probabilities are blended with
  the Logistic Regression baseline's, at a weight tuned on a held-out
  validation split (never the reported test metrics, so the numbers stay
  honest). Measured result: log-loss improved from 1.024 (primary model
  alone) to 1.008 (blended), and draw recall rose from 25% to 31%. The
  **Model Details** tab shows the blend weight and the primary-only numbers
  side by side so you can see exactly what blending bought you.
- **Real data beats sample data**: the shipped dataset is
  synthetic-but-realistic, not actual match history -- swapping in real data
  (see above) is the single biggest lever available, because it's the only
  change that adds genuinely new signal rather than squeezing more out of
  the same information.
- **More simulations** reduce Monte Carlo noise in the champion
  probabilities themselves (the model's predictions don't change, but the
  simulation's estimate of them gets more precise) -- push the slider toward
  50,000-100,000 if you want tighter numbers.

All of these are on by default except tuning during live-update retrains
(kept off there for speed -- run `python train_model.py` periodically to get
the fully-tuned model back).

---

## Deployment (free)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) (Streamlit
   Community Cloud, free tier).
3. Point it at `app.py` in this repo.
4. Done -- no servers, no keys, no cost.

---

## Limitations & future enhancements

- The shipped team ratings/history are **sample data** -- treat outputs as
  a demo until you swap in real data (see above).
- Group draw is a simplified seeding (not the exact confederation-constrained
  official draw procedure).
- Tie-breaking in groups uses a simplified goal-difference proxy rather than
  full head-to-head/goals-scored FIFA tie-break rules.
- Planned extensions from the PRD: live in-tournament updates, injury/lineup
  impact, an AI chatbot for explanations, and live free-API data refresh.

---

## License

Built for educational/demo purposes. Uses only free, open-source tooling.
