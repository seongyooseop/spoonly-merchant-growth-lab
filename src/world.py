"""Simulate SPOONLY, a fictional food-delivery app in three Korean service areas (2025, 52 weeks).

Writes data/raw (what an analyst would see) and data/truth (the parameters used to generate it).
Analysis code reads data/raw only; data/truth is read by src/validate.py.

Problems planted on purpose:
  1. Merchants raise their own discounts when demand dips (discounts look weak in pooled data).
  2. Coupon-redeemed orders overstate what a reorder coupon adds (most heavy users order anyway).
  3. First-order coupons bring new customers who reorder less.
  4. A cost-sharing pilot (area W, growth-tier merchants, from week 30) changes merchant discounts.
  5. A year-end surge (weeks 51-52) never appears in the first 39 weeks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, TRUTH = ROOT / "data" / "raw", ROOT / "data" / "truth"
rng = np.random.default_rng(20251113)

WEEKS = 52
START = pd.Timestamp("2025-01-06")
AREAS = {"E": "서울 동부", "W": "서울 서부", "S": "경기 남부"}
# category: share, AOV (won), discount semi-elasticity b (orders x exp(b*d)), base orders/week, repeat share
CATS = {
    "치킨": (0.18, 24000, 2.2, 62, 0.62),
    "한식": (0.20, 14000, 2.6, 46, 0.66),
    "분식": (0.14, 11000, 4.2, 50, 0.55),
    "카페·디저트": (0.14, 9500, 3.8, 36, 0.50),
    "중식": (0.12, 16000, 2.4, 40, 0.60),
    "피자": (0.10, 26000, 2.0, 34, 0.58),
    "일식": (0.12, 21000, 1.3, 30, 0.64),
}
N_MERCHANTS = 2400
TAKE_RATE = 0.10
CAMPAIGN_WEEKS = [6, 7, 18, 19, 31, 32, 44, 45]
HOLIDAY = {4: 0.12, 40: 0.12}          # 설(2025-01-27 week) and 추석(2025-10-06 week)
YEAR_END = {51: 0.18, 52: 0.22}        # not visible before week 40
PILOT_AREA, PILOT_WEEK, PILOT_MATCH = "W", 30, 0.5


def season(w):
    return 1 + 0.06 * np.sin(2 * np.pi * (w - 10) / 52) + np.where(np.isin(w, [27, 28, 29, 30]), 0.05, 0)


def merchants():
    cats = list(CATS)
    cat = rng.choice(cats, N_MERCHANTS, p=[CATS[c][0] for c in cats])
    area = rng.choice(list(AREAS), N_MERCHANTS, p=[0.38, 0.34, 0.28])
    rating = np.clip(rng.normal(4.45, 0.25, N_MERCHANTS), 3.4, 5.0).round(1)
    delivery = np.clip(rng.normal(34, 6, N_MERCHANTS), 18, 58).round()
    join_week = np.where(rng.random(N_MERCHANTS) < 0.18, rng.integers(2, 40, N_MERCHANTS), 1)
    aov = np.array([CATS[c][1] for c in cat]) * np.exp(rng.normal(0, 0.15, N_MERCHANTS))
    b = np.array([CATS[c][2] for c in cat]) * np.exp(rng.normal(0, 0.2, N_MERCHANTS))
    size = np.exp(rng.normal(0, 0.55, N_MERCHANTS))
    # levers: +4% per 0.1 rating above 4.4, -1.5% per minute above 32
    lever = np.exp(0.40 * (rating - 4.4) - 0.015 * np.maximum(delivery - 32, 0))
    base = np.array([CATS[c][3] for c in cat]) * size * lever
    trend = rng.normal(0.004, 0.004, N_MERCHANTS)          # weekly growth
    trend = np.where(join_week > 1, trend + 0.012, trend)   # new merchants ramp up
    disc_prop = np.where(rng.random(N_MERCHANTS) < 0.42, 0, rng.uniform(0.03, 0.12, N_MERCHANTS))
    return pd.DataFrame({
        "merchant_id": [f"M{i:04d}" for i in range(N_MERCHANTS)], "area": area, "category": cat,
        "rating": rating, "delivery_min": delivery, "join_week": join_week, "aov_won": aov.round(-2),
        "_b": b, "_base": base, "_trend": trend, "_disc": disc_prop,
    })


def campaign_cells():
    """Platform coupon rate per (campaign week, area, category): randomised 0 / 5% / 10%."""
    rows = []
    for w in CAMPAIGN_WEEKS:
        for a in AREAS:
            for c in CATS:
                rows.append((w, a, c, rng.choice([0.0, 0.05, 0.10])))
    return pd.DataFrame(rows, columns=["week", "area", "category", "platform_rate"])


def panel(m, cells):
    camp = {(r.week, r.area, r.category): r.platform_rate for r in cells.itertuples()}
    n = len(m)
    u = np.zeros(n)                            # AR(1) demand shock the merchant can see
    disc = m["_disc"].to_numpy().copy()
    growth_tier = np.zeros(n, bool)            # set after week 26 from what merchants look like then
    rows, truth_rows = [], []
    hist_orders = np.zeros((n, WEEKS))
    for w in range(1, WEEKS + 1):
        u = 0.7 * u + rng.normal(0, 0.12, n)
        # endogenous discounting: after a bad week, merchants with a discount habit push it up
        change = rng.random(n) < 0.10
        target = np.clip(m["_disc"].to_numpy() + np.where(u < -0.08, 0.05, 0) + rng.normal(0, 0.01, n), 0, 0.2)
        disc = np.where(change, np.where(m["_disc"].to_numpy() > 0, target, np.where(u < -0.15, 0.05, 0)), disc)
        if w == 27:
            pre = hist_orders[:, 13:26].mean(1)
            potential = (m["rating"].to_numpy() >= 4.4) & (m["delivery_min"].to_numpy() <= 36)
            growth_tier = potential & (pre < np.median(pre))
        pilot = (m["area"].to_numpy() == PILOT_AREA) & growth_tier & (w >= PILOT_WEEK)
        # pilot: platform matches 50% of the merchant discount (capped at 5%); more merchants start discounting
        start = pilot & (disc == 0) & (rng.random(n) < 0.05)
        disc = np.where(start, 0.06, disc)
        plat_match = np.where(pilot, np.minimum(PILOT_MATCH * disc, 0.05), 0)
        plat_camp = np.array([camp.get((w, a, c), 0.0) for a, c in zip(m["area"], m["category"])])
        d = disc + plat_match + plat_camp
        active = w >= m["join_week"].to_numpy()
        age = np.maximum(w - m["join_week"].to_numpy(), 0)
        lam = (m["_base"].to_numpy() * season(np.array(w)) * (1 + HOLIDAY.get(w, 0) + YEAR_END.get(w, 0))
               * np.exp(m["_trend"].to_numpy() * age) * np.exp(u) * np.exp(m["_b"].to_numpy() * d))
        lam = np.where(active, lam, 0)
        orders = rng.poisson(lam)
        hist_orders[:, w - 1] = orders
        rep_base = np.array([CATS[c][4] for c in m["category"]]) + 0.15 * (m["rating"].to_numpy() - 4.4)
        rep_share = np.clip(rep_base - 0.9 * plat_camp - 0.4 * disc, 0.2, 0.9)   # coupons pull in first-timers
        repeat = rng.binomial(orders, rep_share)
        aov = m["aov_won"].to_numpy() * np.exp(rng.normal(0, 0.03, n))
        gmv = orders * aov
        lam0 = lam / np.exp(m["_b"].to_numpy() * d)
        rows.append(pd.DataFrame({
            "merchant_id": m["merchant_id"], "week": w, "orders": orders, "repeat_orders": repeat,
            "gmv_won": gmv.round(-2), "merchant_disc_rate": disc.round(3),
            "platform_disc_rate": (plat_match + plat_camp).round(3),
            "merchant_disc_won": (gmv * disc).round(-1), "platform_disc_won": (gmv * (plat_match + plat_camp)).round(-1),
        })[active])
        truth_rows.append(pd.DataFrame({"merchant_id": m["merchant_id"], "week": w, "lam": lam, "lam_no_discount": lam0,
                                        "pilot": pilot, "growth_tier_true": growth_tier})[active])
    return pd.concat(rows, ignore_index=True), pd.concat(truth_rows, ignore_index=True)


def reorder_coupon_test():
    """Customer holdout, weeks 27-38: 15% of customers never get the 3,000-won reorder coupon.
    Treated customers get it in any week that starts 21+ days after their last order."""
    n = 90000
    seg = rng.choice(["주 1회 이상", "월 2~3회", "월 1회 이하", "휴면(60일+)"], n, p=[0.16, 0.34, 0.32, 0.18])
    p0 = {"주 1회 이상": 0.62, "월 2~3회": 0.22, "월 1회 이하": 0.07, "휴면(60일+)": 0.018}
    lift = {"주 1회 이상": 0.01, "월 2~3회": 0.035, "월 1회 이하": 0.055, "휴면(60일+)": 0.045}
    base = np.array([p0[s] for s in seg]) * np.exp(rng.normal(0, 0.3, n))
    add = np.array([lift[s] for s in seg])
    treated = rng.random(n) >= 0.15
    since = np.where(seg == "휴면(60일+)", 9, rng.integers(0, 5, n))      # weeks since last order
    recs = []
    for w in range(27, 39):
        eligible = since >= 3
        coupon = treated & eligible
        p = np.clip(base + np.where(coupon, add, 0), 0, 0.95)
        order = rng.random(n) < p
        redeemed = order & coupon & (rng.random(n) < 0.85)
        since = np.where(order, 0, since + 1)
        recs.append(pd.DataFrame({"customer_id": np.arange(n), "week": w, "segment": seg, "treated": treated,
                                  "coupon_sent": coupon, "ordered": order, "redeemed": redeemed}))
    df = pd.concat(recs, ignore_index=True)
    truth = {"lift_per_eligible_week": lift, "coupon_won": 3000, "holdout_share": 0.15}
    return df, truth


def first_order_coupon_test():
    """Switchback geo test, weeks 40-47: first-order coupon (5,000 won) on/off by area in 2-week blocks.
    Every new customer is followed 8 weeks for a reorder."""
    sched = {"E": [1, 0, 1, 0], "W": [0, 1, 0, 1], "S": [1, 0, 0, 1]}
    base_new = {"E": 1900, "W": 1650, "S": 1200}
    rows = []
    for a in AREAS:
        for blk in range(4):
            for k in range(2):
                w = 40 + blk * 2 + k
                on = sched[a][blk]
                organic = rng.poisson(base_new[a] * season(np.array(w)) * (1 + HOLIDAY.get(w, 0)))
                extra = rng.poisson(0.32 * base_new[a] * season(np.array(w))) if on else 0
                n_new = organic + extra
                # coupon-acquired customers reorder less; organic ones keep ~38%
                via_coupon = np.zeros(n_new, bool)
                if on:
                    via_coupon[:] = rng.random(n_new) < 0.55   # most new customers use the coupon when it exists
                    marginal = np.zeros(n_new, bool)
                    marginal[rng.choice(n_new, extra, replace=False)] = True
                else:
                    marginal = np.zeros(n_new, bool)
                p_rep = np.where(marginal, 0.22, 0.38)
                rep = rng.random(n_new) < p_rep
                rows.append(pd.DataFrame({"area": a, "week": w, "coupon_on": on, "used_coupon": via_coupon,
                                          "reordered_8w": rep}))
    df = pd.concat(rows, ignore_index=True)
    df.insert(0, "new_customer_id", np.arange(len(df)))
    truth = {"extra_new_share": 0.32, "repeat_organic": 0.38, "repeat_marginal": 0.22, "coupon_won": 5000}
    return df, truth


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    TRUTH.mkdir(parents=True, exist_ok=True)
    m = merchants()
    cells = campaign_cells()
    p, tp = panel(m, cells)
    m[[c for c in m.columns if not c.startswith("_")]].to_csv(RAW / "merchants.csv", index=False)
    p.to_csv(RAW / "merchant_week.csv", index=False)
    cells.to_csv(RAW / "platform_campaigns.csv", index=False)
    pd.DataFrame({"week": range(1, WEEKS + 1), "week_start": [(START + pd.Timedelta(weeks=i)).date() for i in range(WEEKS)],
                  "holiday": [1 if w in HOLIDAY else 0 for w in range(1, WEEKS + 1)]}).to_csv(RAW / "calendar.csv", index=False)
    pd.DataFrame([{"area": PILOT_AREA, "start_week": PILOT_WEEK, "rule": "growth-tier merchants: platform matches 50% of merchant discount, cap 5%p"}]).to_csv(RAW / "pilot_policy.csv", index=False)
    r, rt = reorder_coupon_test()
    r.to_parquet(RAW / "reorder_coupon_test.parquet", index=False)
    f, ft = first_order_coupon_test()
    f.to_csv(RAW / "first_order_coupon_test.csv", index=False)

    m[["merchant_id", "_b", "_base", "_trend"]].rename(columns=lambda c: c.lstrip("_")).to_csv(TRUTH / "merchant_params.csv", index=False)
    tp.to_parquet(TRUTH / "merchant_week_truth.parquet", index=False)
    json.dump({"categories": {c: {"b": v[2]} for c, v in CATS.items()}, "lever_rating_per_0p1": 0.04,
               "lever_delivery_per_min_over_32": -0.015, "take_rate": TAKE_RATE, "pilot": {"area": PILOT_AREA, "week": PILOT_WEEK,
               "match": PILOT_MATCH, "start_prob_per_week": 0.05}, "year_end": YEAR_END, "holiday": HOLIDAY,
               "reorder_test": rt, "first_order_test": ft}, open(TRUTH / "params.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"merchants {len(m):,}  merchant-weeks {len(p):,}  orders {p.orders.sum():,}  GMV {p.gmv_won.sum()/1e8:,.0f}억 원")
    print(f"reorder test rows {len(r):,}  first-order test new customers {len(f):,}")


if __name__ == "__main__":
    main()
