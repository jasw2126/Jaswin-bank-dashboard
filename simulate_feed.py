"""Simulated live feed: adds one fictional snapshot every 30 seconds.
Run: python simulate_feed.py   (no labels and no predictions are created here)"""
import io
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE, TARGET = BASE / "data" / "synthetic_customers.csv", BASE / "data" / "live_feed.csv"
INTERVAL = int(os.getenv("FEED_INTERVAL", 30))
COLUMNS = ["request_id", "ingested_at", "customer_id", "snapshot_date", "branch_code",
           "account_type", "avg_balance_inr", "previous_month_balance_inr",
           "monthly_credits_inr", "monthly_debits_inr", "transaction_count_30d",
           "digital_sessions_30d", "days_since_last_transaction",
           "complaint_count_90d", "tenure_months"]
rng = np.random.default_rng()


def load_csv(path, **kwargs):
    """Reads normal CSVs and CSVs that have every line wrapped in quotes."""
    with open(path, encoding="utf-8-sig") as f:
        lines = [line.strip() for line in f]
    lines = [l[1:-1] if l.startswith('"') and l.endswith('"') else l for l in lines]
    return pd.read_csv(io.StringIO("\n".join(lines)), **kwargs)


def next_snapshot(prev):
    """Create next month's fictional snapshot from the customer's latest one."""
    credits = max(0.0, prev["monthly_credits_inr"] * rng.lognormal(0, 0.15))
    debits = max(0.0, prev["monthly_debits_inr"] * rng.lognormal(0, 0.15))
    balance = max(0.0, prev["avg_balance_inr"] + credits - debits)
    if prev["transaction_count_30d"] == 0 and rng.random() < 0.8:
        txns = 0                                            # inactive customers mostly stay inactive
    else:
        txns = max(0, int(prev["transaction_count_30d"] + rng.integers(-3, 4)))
    days = int(rng.integers(30, 91)) if txns == 0 else int(min(round(rng.exponential(30 / txns)), 30))
    return {
        "request_id": uuid.uuid4().hex[:12],
        "ingested_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "customer_id": prev["customer_id"],
        "snapshot_date": prev["snapshot_date"] + pd.offsets.MonthEnd(1),
        "branch_code": prev["branch_code"],
        "account_type": prev["account_type"],
        "avg_balance_inr": round(balance, 2),
        "previous_month_balance_inr": round(float(prev["avg_balance_inr"]), 2),
        "monthly_credits_inr": round(credits, 2),
        "monthly_debits_inr": round(debits, 2),
        "transaction_count_30d": txns,
        "digital_sessions_30d": max(0, int(prev["digital_sessions_30d"] + rng.integers(-2, 3))),
        "days_since_last_transaction": days,
        "complaint_count_90d": max(0, int(prev["complaint_count_90d"] + rng.choice([-1, 0, 0, 0, 1]))),
        "tenure_months": int(prev["tenure_months"]) + 1,
    }


def main():
    if not SOURCE.exists():
        raise FileNotFoundError("data/synthetic_customers.csv not found")
    source = load_csv(SOURCE, parse_dates=["snapshot_date"])
    frames = [source]
    if TARGET.exists():                                     # continue from where the feed stopped
        frames.append(pd.read_csv(TARGET, parse_dates=["snapshot_date"]))
    latest = (pd.concat(frames).sort_values("snapshot_date")
              .groupby("customer_id").tail(1).set_index("customer_id", drop=False))
    state = latest.to_dict("index")

    print(f"JASWIN BANK feed started: one record every {INTERVAL} s. Press Ctrl+C to stop.")
    try:
        while True:
            customer_id = rng.choice(list(state))
            row = next_snapshot(state[customer_id])
            pd.DataFrame([row], columns=COLUMNS).to_csv(
                TARGET, mode="a", header=not TARGET.exists(), index=False)
            state[customer_id] = row
            print(f"{row['ingested_at']}  {row['customer_id']}  balance {row['avg_balance_inr']:,.0f}")
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("Feed stopped.")


if __name__ == "__main__":
    main()
