"""
STAGE 2b: FEATURE ENGINEERING FROM DRIVING HISTORY  (LAB 2)

data/driving_history.csv has one row per policy per month for the 24 months
BEFORE the application. Month 1 is the oldest; month 24 is the month right
before the decision. Columns:

    policy_id, month, miles, trips, hard_brakes, speeding_events,
    night_miles_share, claim_filed

Your job in Lab 2 is to turn 24 rows per driver into ONE row per driver with
features from the families you learned: frequency, recency, velocity,
volatility, extremes, trend, ratio.

THE AS-OF RULE
  Every feature must be computed as of the decision date. In this file that
  means: only rows with month <= DECISION_MONTH may be used. Each function
  receives the history table and must apply that filter itself (use the
  helper as_of()). A test hands your functions a table that ALSO contains
  months 25 to 30 with absurd values; if your output changes, you leaked.

HOW THE STUBS WORK
  A function that returns None is "not implemented" and is skipped, so the
  pipeline runs even if you only finish some. A function that returns a
  pandas Series indexed by policy_id is included under its name.

WHICH ONES YOU MUST IMPLEMENT
  claims_per_month is the guided example (the slides walk you through it).
  Then implement at least FOUR more, from at least FOUR different families,
  including the two families assigned to your last-name group on the exam
  sheet. Record each one in MIDTERM.md under Lab 2: name, family, formula in
  one sentence, and why it should predict claim_free.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = ROOT / "data" / "driving_history.csv"

DECISION_MONTH = 24          # the last month you are allowed to see
RECENT_WINDOW = 6            # "recent" means the last 6 months before the decision

# Feature name -> family. The Lab 6 submission test counts distinct families.
FAMILY = {
    "claims_per_month": "frequency",
    "months_since_last_claim": "recency",
    "speeding_velocity": "velocity",
    "miles_volatility": "volatility",
    "max_hard_brakes_per_1k": "extremes",
    "hard_brake_trend": "trend",
    "night_share_mean": "ratio",
}


def load_history(path=HISTORY_PATH):
    """Reads ONLY the pre-decision file. history_after_decision.csv is never read here."""
    return pd.read_csv(path)


def as_of(history, decision_month=DECISION_MONTH):
    """Keep only what was knowable at the decision: months 1..decision_month."""
    return history[history["month"] <= decision_month]


def recent(history, window=RECENT_WINDOW, decision_month=DECISION_MONTH):
    """The last `window` months before the decision."""
    h = as_of(history, decision_month)
    return h[h["month"] > decision_month - window]


# --------------------------------------------------------------------------
# GUIDED EXAMPLE (Lab 2, part 1)  family: FREQUENCY
# --------------------------------------------------------------------------
def claims_per_month(history):
    """Prior at-fault claims per month over the 24 months before the decision.

    Steps (also on the slides):
      1. h = as_of(history)                       keep months 1..24 only
      2. total = h.groupby("policy_id")["claim_filed"].sum()
      3. return total / DECISION_MONTH            a rate, so it is a FREQUENCY
    """
    h = as_of(history)   # LAB 2: replace this line with the three steps above
    total = h.groupby("policy_id")["claim_filed"].sum()
    return total / DECISION_MONTH

# --------------------------------------------------------------------------
# YOUR FEATURES (Lab 2, part 2). Implement at least four of the six below.
# Each returns a Series indexed by policy_id, or None if not implemented.
# --------------------------------------------------------------------------
def months_since_last_claim(history):
    """RECENCY: months between the most recent prior claim and the decision.
    If a driver never filed a claim in the window, use DECISION_MONTH + 1
    (one more than the longest possible gap). Never use 0 for 'never'."""
    h = as_of(history)
    last = h[h["claim_filed"] == 1].groupby("policy_id")["month"].max()
    last = last.reindex(h["policy_id"].unique())
    return ((DECISION_MONTH + 1) - last).fillna(DECISION_MONTH + 1)


def speeding_velocity(history):
    """VELOCITY: speeding events per 1,000 miles in the recent window, divided
    by speeding events per 1,000 miles over the whole 24 months.
    Add a small constant to the denominator so a driver with zero baseline
    events does not produce infinity."""
    return None


def miles_volatility(history):
    """VOLATILITY: coefficient of variation of monthly miles
    (standard deviation divided by mean) over the 24 months."""
    h = as_of(history)
    g = h.groupby("policy_id")["miles"]
    return g.std() / g.mean()


def max_hard_brakes_per_1k(history):
    """EXTREMES: the single worst month, measured as hard brakes per 1,000
    miles in that month, over the 24 months."""
    h = as_of(history).copy()
    h["rate"] = h["hard_brakes"] / h["miles"] * 1000
    return h.groupby("policy_id")["rate"].max()


def hard_brake_trend(history):
    """TREND: hard brakes per 1,000 miles in months 13..24 minus hard brakes
    per 1,000 miles in months 1..12. Positive means getting worse."""
    return None


def night_share_mean(history):
    """RATIO: average share of miles driven at night over the 24 months."""
    h = as_of(history)
    return h.groupby("policy_id")["night_miles_share"].mean()


# --------------------------------------------------------------------------
# Assembly. Do not edit below this line.
# --------------------------------------------------------------------------
FEATURE_FUNCTIONS = [
    claims_per_month,
    months_since_last_claim,
    speeding_velocity,
    miles_volatility,
    max_hard_brakes_per_1k,
    hard_brake_trend,
    night_share_mean,
]

ENGINEERED = [f.__name__ for f in FEATURE_FUNCTIONS]


def implemented_features(history=None):
    """Names of the feature functions that return something."""
    history = history if history is not None else load_history()
    return [f.__name__ for f in FEATURE_FUNCTIONS if f(history) is not None]


def build_features(history):
    """One row per policy_id with every implemented feature as a column."""
    cols = {}
    for fn in FEATURE_FUNCTIONS:
        out = fn(history)
        if out is None:
            continue
        if not isinstance(out, pd.Series):
            raise TypeError(f"{fn.__name__} must return a pandas Series indexed by policy_id")
        cols[fn.__name__] = out.astype(float)
    if not cols:
        return pd.DataFrame({"policy_id": sorted(history["policy_id"].unique())})
    table = pd.DataFrame(cols)
    table.index.name = "policy_id"
    return table.reset_index()


if __name__ == "__main__":
    h = load_history()
    done = implemented_features(h)
    print(f"implemented: {done}")
    print(f"families: {sorted({FAMILY[d] for d in done})}")
    print(build_features(h).describe().T.round(3))
