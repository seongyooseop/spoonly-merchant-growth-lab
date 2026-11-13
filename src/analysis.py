"""Chapters 0-5 of the SPOONLY case. Reads data/marts (and data/raw campaign cells); writes outputs/*.json.

0 KPI tree          what moves GMV and platform net revenue
1 Segments          merchant tiers and growth levers (rating, delivery time)
2 Coupons           reorder-coupon holdout (activation) + first-order coupon switchback (acquisition)
3 Discount response platform-randomised campaign cells -> category elasticities -> budget reallocation
4 Cost sharing      pilot DiD (area W, growth tier, week 30+) -> sharing rule by tier
5 Forecast          bottom-up quarterly forecast, Q4 backtest vs naive
6 Impact            the recommendations together, replayed on Q4 2025, with the value-per-order sensitivity
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
MARTS, OUT = ROOT / "data" / "marts", ROOT / "outputs"
TAKE = 0.10
CAMPAIGN_WEEKS = sorted(pd.read_csv(ROOT / "data" / "raw" / "platform_campaigns.csv").week.unique().tolist())
SENS_VALUES = [3000, 4500, 6000, 7500]
VALUE_PER_INCR_ORDER = 4500   # assumption: commission on an order (~1,700 won) + value of the reorders it starts


def load(name):
    return pd.read_parquet(MARTS / f"{name}.parquet")


def dump(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def tiers(h1):
    h = h1.dropna(subset=["orders_wk_q2"]).copy()
    h["potential"] = (h.rating >= 4.4) & (h.delivery_min <= 36)
    h["level_high"] = h.orders_wk_q2 >= h.orders_wk_q2.median()
    h["tier"] = np.select([h.level_high & h.potential, ~h.level_high & h.potential, h.level_high & ~h.potential],
                          ["핵심", "성장", "유지"], "관리")
    return h


# ---------------------------------------------------------------- 0 KPI tree
def ch0(mw, kpi):
    q = kpi.assign(q=(kpi.week - 1) // 13 + 1).groupby("q").agg(
        orders=("orders", "sum"), new=("new_orders", "sum"), gmv=("gmv_won", "sum"),
        pdisc=("platform_disc_won", "sum"), mdisc=("merchant_disc_won", "sum"), net=("platform_net_won", "sum"))
    q["aov"] = q.gmv / q.orders
    q["repeat_share"] = 1 - q.new / q.orders
    q["platform_disc_share"] = q.pdisc / q.gmv
    year = dict(orders=int(kpi.orders.sum()), gmv=float(kpi.gmv_won.sum()), aov=float(kpi.gmv_won.sum() / kpi.orders.sum()),
                repeat_share=float(kpi.repeat_orders.sum() / kpi.orders.sum()), merchants=int(mw.merchant_id.nunique()),
                platform_disc=float(kpi.platform_disc_won.sum()), merchant_disc=float(kpi.merchant_disc_won.sum()),
                net=float(kpi.platform_net_won.sum()))
    # campaign weeks vs neighbours: platform coupons lift GMV but cut net revenue per order
    camp = kpi[kpi.week.isin(CAMPAIGN_WEEKS)]
    near = kpi[kpi.week.isin([w + d for w in CAMPAIGN_WEEKS for d in (-2, 2)]) & ~kpi.week.isin(CAMPAIGN_WEEKS)]
    weekly = kpi[["week", "week_start", "orders", "gmv_won", "platform_net_won", "platform_disc_won"]].copy()
    weekly["week_start"] = weekly.week_start.astype(str)
    dump("ch0_kpi.json", {"year": year, "quarters": q.reset_index().to_dict("records"), "weekly": weekly.to_dict("records"),
                          "campaign": {"net_per_order": float(camp.platform_net_won.sum() / camp.orders.sum()),
                                       "net_per_order_near": float(near.platform_net_won.sum() / near.orders.sum()),
                                       "orders_per_week": float(camp.orders.mean()), "orders_per_week_near": float(near.orders.mean())}})


# ---------------------------------------------------------------- 1 Segments and levers
def ch1(h1, mw):
    h = tiers(h1)
    h2 = mw[mw.week.between(27, 52)].groupby("merchant_id").agg(gmv_h2=("gmv_won", "sum"), orders_h2=("orders", "sum"))
    h = h.join(h2, on="merchant_id")
    g = h.groupby("tier").agg(n=("merchant_id", "size"), orders_wk=("orders_wk_q2", "mean"), gmv_h2=("gmv_h2", "sum"),
                              repeat=("repeat_share", "mean"), rating=("rating", "mean"), delivery=("delivery_min", "mean"),
                              growth=("orders_wk_q1", "mean"))
    g["gmv_share"] = g.gmv_h2 / g.gmv_h2.sum()
    g["growth"] = h.groupby("tier").orders_wk_q2.mean() / g.growth - 1
    # lever model: cross-section of established merchants (joined week 1)
    est = h[h.join_week == 1].copy()
    est["log_orders"] = np.log(est.orders_wk_q2)
    est["over32"] = np.maximum(est.delivery_min - 32, 0)
    fit = smf.ols("log_orders ~ rating + over32 + np.log(aov_won) + C(category) + C(area)", est).fit(cov_type="HC1")
    b_rating, b_del = fit.params["rating"], fit.params["over32"]
    ci_rating, ci_del = fit.conf_int().loc["rating"].tolist(), fit.conf_int().loc["over32"].tolist()
    # opportunity: merchants slower than 36 min brought to 32 min; rated below 4.3 brought to 4.3
    slow = h[h.delivery_min > 36]
    gain_del = (slow.gmv_h2 * (np.exp(-b_del * (slow.delivery_min - 32)) - 1)).sum()
    low = h[h.rating < 4.3]
    gain_rat = (low.gmv_h2 * (np.exp(b_rating * (4.3 - low.rating)) - 1)).sum()
    tiers_out = h[["merchant_id", "tier", "area", "category"]]
    tiers_out.to_csv(OUT / "merchant_tiers.csv", index=False)
    dump("ch1_segments.json", {
        "tiers": g.reset_index().to_dict("records"), "n": int(len(h)),
        "levers": {"rating_per_0p1": float(np.exp(0.1 * b_rating) - 1), "rating_ci": [float(np.exp(0.1 * c) - 1) for c in ci_rating],
                   "delivery_per_min": float(np.exp(b_del) - 1), "delivery_ci": [float(np.exp(c) - 1) for c in ci_del],
                   "n_est": int(len(est)), "r2": float(fit.rsquared)},
        "opportunity": {"slow_n": int(len(slow)), "gain_delivery_h2": float(gain_del), "low_rating_n": int(len(low)),
                        "gain_rating_h2": float(gain_rat)}})
    return h


# ---------------------------------------------------------------- 2 Coupons: activation and acquisition
def ch2():
    c = load("mart_reorder_customer")
    rows = []
    for seg, d in list(c.groupby("segment")) + [("전체", c)]:
        t, k = d[d.treated], d[~d.treated]
        diff = t.orders.mean() - k.orders.mean()
        se = np.sqrt(t.orders.var() / len(t) + k.orders.var() / len(k))
        incr = diff * len(t)
        red = t.redemptions.sum()
        rows.append({"segment": seg, "n_treated": int(len(t)), "n_holdout": int(len(k)),
                     "orders_t": float(t.orders.mean()), "orders_h": float(k.orders.mean()), "diff": float(diff),
                     "ci": [float(diff - 1.96 * se), float(diff + 1.96 * se)], "p": float(2 * stats.norm.sf(abs(diff / se))),
                     "redemptions": int(red), "incremental": float(incr), "incr_share": float(incr / red) if red else None,
                     "cost": float(3000 * red), "cost_per_incr": float(3000 * red / incr) if incr > 0 else None})
    r = pd.DataFrame(rows)
    keep = r[(r.segment != "전체") & (r.cost_per_incr.notna()) & (r.cost_per_incr <= VALUE_PER_INCR_ORDER)]
    seg_rows = r[r.segment != "전체"]
    targeted = {"segments": keep.segment.tolist(), "cost": float(keep.cost.sum()), "incremental": float(keep.incremental.sum()),
                "cost_all": float(seg_rows.cost.sum()), "incr_all": float(seg_rows.incremental.sum())}
    targeted["cost_cut"] = 1 - targeted["cost"] / targeted["cost_all"]
    targeted["incr_kept"] = targeted["incremental"] / targeted["incr_all"]
    # second option: the two segments with the lowest cost per incremental order
    two = seg_rows.dropna(subset=["cost_per_incr"]).nsmallest(2, "cost_per_incr")
    targeted["option2"] = {"segments": two.segment.tolist(), "cost_cut": float(1 - two.cost.sum() / targeted["cost_all"]),
                           "incr_kept": float(two.incremental.sum() / targeted["incr_all"]),
                           "cost_per_incr": float(two.cost.sum() / two.incremental.sum())}

    f = load("mart_first_order")
    fit = smf.ols("new_customers ~ coupon_on + C(area) + C(week)", f).fit(cov_type="HC1")
    extra = fit.params["coupon_on"]
    on, off = f[f.coupon_on == 1], f[f.coupon_on == 0]
    organic_on = on.new_customers.sum() - extra * len(on)
    # reorder rate of the marginal (coupon-induced) customers: on-week reorders minus what organic customers would do
    rate_off = (off.reorder_8w * off.new_customers).sum() / off.new_customers.sum()
    reorders_on = (on.reorder_8w * on.new_customers).sum()
    marginal_rate = (reorders_on - rate_off * organic_on) / (extra * len(on))
    users_on = on.coupon_users.sum()
    # randomisation inference: every schedule with 2 of 4 two-week blocks on in each area (6^3 = 216)
    f["block"] = (f.week - 40) // 2
    combos = list(itertools.combinations(range(4), 2))
    areas = sorted(f.area.unique())
    perm = []
    for ca in itertools.product(combos, repeat=len(areas)):
        if len(set(ca)) == 1:
            continue                                   # same schedule everywhere = collinear with week effects
        on_map = {(a, blk): int(blk in c) for a, c in zip(areas, ca) for blk in range(4)}
        g = f.assign(coupon_on=[on_map[(a, bl)] for a, bl in zip(f.area, f.block)])
        perm.append(smf.ols("new_customers ~ coupon_on + C(area) + C(week)", g).fit().params["coupon_on"])
    perm = np.array(perm)
    p_perm = float((np.abs(perm) >= abs(extra) - 1e-9).mean())
    acq = {"extra_per_area_week": float(extra), "ci": fit.conf_int().loc["coupon_on"].tolist(),
           "base_per_area_week": float(off.new_customers.mean()), "lift": float(extra / off.new_customers.mean()),
           "reorder_organic": float(rate_off), "reorder_marginal": float(marginal_rate),
           "coupon_users_share_on": float(users_on / on.new_customers.sum()),
           "cost_per_incr_new": float(5000 * users_on / (extra * len(on))),
           "cost_per_incr_retained": float(5000 * users_on / (extra * len(on) * marginal_rate)),
           "n_obs": int(len(f)), "p_permutation": p_perm, "n_permutations": int(len(perm))}
    rep4 = load("mart_repeat_4w")
    repeat = [{"segment": (r.segment if isinstance(r.segment, str) else "전체"),
               "arm": ("전체" if pd.isna(r.treated) else ("쿠폰" if r.treated else "홀드아웃")),
               "base_orders": int(r.base_orders), "repeat_4w": float(r.repeat_4w)} for r in rep4.itertuples()]
    breakpoints = {r.segment: float(r.cost_per_incr) for r in seg_rows.dropna(subset=["cost_per_incr"]).itertuples()}
    sens = []
    for v in SENS_VALUES:
        k = seg_rows[seg_rows.cost_per_incr.notna() & (seg_rows.cost_per_incr <= v)]
        sens.append({"value": v, "segments": k.segment.tolist(), "cost_cut": float(1 - k.cost.sum() / targeted["cost_all"]),
                     "incr_kept": float(k.incremental.sum() / targeted["incr_all"])})
    dump("ch2_coupons.json", {"reorder": rows, "value_per_incr_order": VALUE_PER_INCR_ORDER, "targeted": targeted,
                              "first_order": acq, "repeat_4w": repeat, "sensitivity": sens, "breakpoints": breakpoints,
                              "area_weeks": f.drop(columns="block").to_dict("records")})


# ---------------------------------------------------------------- 3 Discount response by category
def ch3(mw):
    d = mw[(mw.orders > 0) & (mw.week >= 2)].copy()
    d["lo"] = np.log(d.orders)
    pilot = pd.read_csv(ROOT / "data" / "raw" / "pilot_policy.csv").iloc[0]
    d = d[~((d.area == pilot.area) & (d.week >= pilot.start_week))]       # keep the pilot's matched discounts out
    cats = sorted(d.category.unique())
    naive = smf.ols("lo ~ total_disc_rate:C(category) + C(category) + C(area)", d).fit()
    # merchant & week fixed effects absorbed by demeaning; identification from randomised campaign cells.
    # Week effects are estimated within area so area-level seasonality does not leak in.
    d["aw"] = d.area + "_" + d.week.astype(str)
    for col in ["lo", "campaign_rate", "merchant_disc_rate"]:
        d[col + "_dm"] = d[col] - d.groupby("merchant_id")[col].transform("mean")
        d[col + "_dm"] = d[col + "_dm"] - d.groupby("aw")[col + "_dm"].transform("mean")
    res = []
    for c in cats:
        x = d[d.category == c]
        fe = smf.ols("lo_dm ~ campaign_rate_dm + merchant_disc_rate_dm - 1", x).fit(cov_type="cluster", cov_kwds={"groups": x.merchant_id})
        aov = x.aov_won.mean()
        cur_rate = x.total_disc_rate.mean()
        b = fe.params["campaign_rate_dm"]
        res.append({"category": c, "b_naive": float(naive.params[f"total_disc_rate:C(category)[{c}]"]),
                    "b_fe_merchant_disc": float(fe.params["merchant_disc_rate_dm"]),
                    "b": float(b), "ci": fe.conf_int().loc["campaign_rate_dm"].tolist(), "aov": float(aov),
                    "disc_rate_now": float(cur_rate), "orders_per_1m_won": float(1e6 * b / (aov * (1 + b * cur_rate)))})
    r = pd.DataFrame(res).sort_values("orders_per_1m_won", ascending=False)
    # reallocation: last campaign (weeks 44-45) spend moved from the bottom 3 to the top 2 categories by orders per won
    camp = mw[mw.week.isin([44, 45])]
    spend = camp.groupby("category").platform_disc_won.sum()
    base_orders = camp.groupby("category").orders.sum()
    top, bottom = r.category.iloc[:2].tolist(), r.category.iloc[-3:].tolist()
    rr = r.set_index("category")
    cells = camp[camp.category.isin(top)]
    rate_before = float(cells.platform_disc_won.sum() / cells.gmv_won.sum())
    cap = spend[top].sum() * (0.10 / rate_before - 1)           # keep the receiving categories' average rate within 0-10%
    wanted = spend[bottom].sum() * 0.5
    moved = min(wanted, cap)
    rate_after = rate_before * (1 + moved / spend[top].sum())
    mid = (rate_before + rate_after) / 2

    def eff(c, d):
        return rr.b[c] / (rr.aov[c] * (1 + rr.b[c] * d))

    give = moved / spend[bottom].sum()
    delta = (-sum(spend[c] * give * eff(c, rr.disc_rate_now[c]) for c in bottom)
             + sum(moved * spend[c] / spend[top].sum() * eff(c, mid) for c in top))
    dump("ch3_elasticity.json", {"categories": r.to_dict("records"), "top": top, "bottom": bottom,
                                 "realloc": {"spend_moved": float(moved), "spend_total": float(spend.sum()),
                                             "orders_delta": float(delta), "orders_base": float(base_orders.sum()),
                                             "rel": float(delta / base_orders.sum()), "rate_top_before": rate_before,
                                             "rate_top_after": float(rate_after), "tested_max": 0.10, "wanted": float(wanted),
                                             "share_of_bottom_moved": float(give),
                                             "within_range": bool(rate_after <= 0.10)}})
    return r


# ---------------------------------------------------------------- 4 Cost-sharing pilot
def ch4(mw, h):
    pilot = pd.read_csv(ROOT / "data" / "raw" / "pilot_policy.csv").iloc[0]
    g = h.loc[h.tier == "성장", ["merchant_id"]]
    d = mw.merge(g, on="merchant_id")
    d = d[(d.week >= 14) & ~d.week.isin(CAMPAIGN_WEEKS)].copy()
    d["treat"] = (d.area == pilot.area).astype(int)
    d["post"] = (d.week >= pilot.start_week).astype(int)
    d["did"] = d.treat * d.post
    d["disc_on"] = (d.merchant_disc_rate > 0).astype(float)
    d["lo"] = np.log(d.orders.clip(lower=1))
    out = {}
    for y in ["disc_on", "lo"]:
        fit = smf.ols(f"{y} ~ did + C(merchant_id) + C(week)", d).fit(cov_type="cluster", cov_kwds={"groups": d.merchant_id})
        out[y] = {"est": float(fit.params["did"]), "ci": fit.conf_int().loc["did"].tolist()}
    # event study in 4-week bins relative to week 30 (bin -1 = weeks 26-29 is the reference)
    d["bin"] = ((d.week - pilot.start_week) // 4).clip(-4, 5)
    es = []
    for b in sorted(d.bin.unique()):
        if b == -1:
            es.append({"bin": int(b), "est": 0.0, "lo": 0.0, "hi": 0.0}); continue
        d[f"e{b}"] = ((d.bin == b) & (d.treat == 1)).astype(int)
    terms = " + ".join(f"Q('e{b}')" for b in sorted(d.bin.unique()) if b != -1)
    fit = smf.ols(f"disc_on ~ {terms} + C(merchant_id) + C(week)", d).fit(cov_type="cluster", cov_kwds={"groups": d.merchant_id})
    for b in sorted(d.bin.unique()):
        if b == -1:
            continue
        k = f"Q('e{b}')"
        es.append({"bin": int(b), "est": float(fit.params[k]), "lo": float(fit.conf_int().loc[k, 0]), "hi": float(fit.conf_int().loc[k, 1])})
    es = sorted(es, key=lambda r: r["bin"])
    post = d[(d.treat == 1) & (d.post == 1)]
    cost = post.platform_disc_won.sum()
    extra_orders = post.orders.sum() * (1 - np.exp(-out["lo"]["est"]))
    dump("ch4_pilot.json", {"n_merchants": int(d.merchant_id.nunique()), "n_treated": int(d[d.treat == 1].merchant_id.nunique()),
                            "did": out, "event_study": es, "platform_cost": float(cost), "extra_orders": float(extra_orders),
                            "cost_per_extra_order": float(cost / extra_orders) if extra_orders > 0 else None,
                            "extra_gmv": float(extra_orders * post.aov_won.mean()),
                            "pays_back": {str(v): bool(cost / extra_orders <= v) for v in SENS_VALUES} if extra_orders > 0 else {}})


# ---------------------------------------------------------------- 5 Forecast
def ch5(mw, elas):
    """Bottom-up forecast: per-merchant log-linear trend on discount-adjusted demand + holiday effect
    + known discount plan x category response. The trend window is chosen on a Q3 backtest
    (data to week 26 -> weeks 27-39), then applied unchanged to Q4 (data to week 39 -> weeks 40-52)."""
    b = elas.set_index("category").b
    cal = pd.read_csv(ROOT / "data" / "raw" / "calendar.csv").set_index("week").holiday
    camp = pd.read_csv(ROOT / "data" / "raw" / "platform_campaigns.csv")
    cr = {(r.week, r.area, r.category): r.platform_rate for r in camp.itertuples()}
    d = mw.copy()
    d["b"] = d.category.map(b)
    d["adj"] = np.log(d.orders.clip(lower=1)) - d.b * d.total_disc_rate          # demand without discounts
    hol = d[d.week.isin([3, 4, 5])].pivot_table(index="merchant_id", columns="week", values="adj")
    hol_effect = float(np.nanmedian(hol[4] - (hol[3] + hol[5]) / 2))
    meta = d.drop_duplicates("merchant_id").set_index("merchant_id")[["area", "category", "b"]]
    resid_var = 0.12 ** 2 / (1 - 0.7 ** 2)

    def run(window, last, weeks):
        x = d[d.week.between(last - window + 1, last) & ~d.week.isin(CAMPAIGN_WEEKS)]
        x = x[x.groupby("merchant_id").week.transform("size") >= 6]
        g = x.groupby("merchant_id")
        wm, am = g.week.transform("mean"), g.adj.transform("mean")
        k = ((x.week - wm) * (x.adj - am)).groupby(x.merchant_id).sum() / ((x.week - wm) ** 2).groupby(x.merchant_id).sum()
        k = k.clip(-0.02, 0.03)
        par = pd.DataFrame({"k": k, "wbar": g.week.mean(), "abar": g.adj.mean(),
                            "disc_last": d[d.week.between(last - 3, last)].groupby("merchant_id").merchant_disc_rate.mean()}).dropna().join(meta)
        out = []
        for w in weeks:
            crate = np.array([cr.get((w, a, c), 0.0) for a, c in zip(par.area, par.category)])
            lvl = par.abar + par.k * (w - par.wbar) + (hol_effect if cal.get(w, 0) else 0)
            out.append(pd.DataFrame({"merchant_id": par.index, "week": w,
                                     "orders_hat": np.exp(lvl + par.b * (par.disc_last + crate) + 0.5 * resid_var)}))
        fc = pd.concat(out)
        act = d[d.week.isin(weeks)].groupby("week").orders.sum()
        newm = d[d.week.isin(weeks) & ~d.merchant_id.isin(par.index)].groupby("week").orders.sum().reindex(act.index, fill_value=0)
        pred = fc.groupby("week").orders_hat.sum() + newm      # merchants too new to model are added at actuals (stated)
        return fc, act, pred

    def mape(p, a):
        return float((abs(p - a) / a).mean())

    windows = [13, 20, 26]
    sel = {w: mape(*reversed(run(w, 26, range(27, 40))[1:])) for w in windows}
    best = min(sel, key=sel.get)
    fc, act, pred = run(best, 39, range(40, 53))
    naive = pd.Series(d[d.week.between(27, 39)].groupby("week").orders.sum().mean(), index=act.index)
    weekly = pd.DataFrame({"week": act.index, "actual": act.values, "bottom_up": pred.values, "naive": naive.values})
    norm = weekly[(weekly.week <= 50) & ~weekly.week.isin(CAMPAIGN_WEEKS + [40])]
    event = weekly[weekly.week.isin([40, 44, 45])]
    t = pd.read_csv(OUT / "merchant_tiers.csv").set_index("merchant_id").tier
    fc["tier"] = fc.merchant_id.map(t)
    at = d[d.week >= 40].assign(tier=lambda x: x.merchant_id.map(t)).groupby("tier").orders.sum()
    seg_err = (fc.groupby("tier").orders_hat.sum() / at - 1).dropna().to_dict()
    # weeks 40-50 only (the year-end surge is a known miss), by tier
    fc50 = fc[fc.week <= 50]
    at50 = d[d.week.between(40, 50)].assign(tier=lambda x: x.merchant_id.map(t)).groupby("tier").orders.sum()
    seg_err50 = (fc50.groupby("tier").orders_hat.sum() / at50 - 1).dropna().to_dict()
    # baseline outlook for 2026 Q1 (weeks 53-65): trend fitted on weeks 38-50 so the year-end surge is not read as growth,
    # 설 (week 59) at the holiday effect, no campaigns, current merchant discounts. Compared with Q4 regular weeks.
    fq, _, _ = run(best, 50, range(53, 66))
    fq.loc[fq.week == 59, "orders_hat"] *= np.exp(hol_effect)
    fq["tier"] = fq.merchant_id.map(t).fillna("신규(등급 없음)")
    reg = [41, 42, 43, 46, 47, 48, 49, 50]
    base_wk = d[d.week.isin(reg)].assign(tier=lambda x: x.merchant_id.map(t).fillna("신규(등급 없음)")).groupby("tier").orders.sum() / len(reg)
    q1_wk = fq[fq.week != 59].groupby("tier").orders_hat.sum() / 12
    outlook = [{"tier": k, "q4_regular_weekly": float(base_wk.get(k, 0)), "q1_weekly": float(q1_wk.get(k, 0)),
                "change": float(q1_wk.get(k, 0) / base_wk[k] - 1) if base_wk.get(k, 0) else None}
               for k in ["핵심", "성장", "유지", "관리", "신규(등급 없음)"]]
    dump("ch5_forecast.json", {
        "weekly": weekly.to_dict("records"), "window_selection_q3_mape": sel, "window": best,
        "mape_bottom_up": mape(pred, act), "mape_naive": mape(naive, act),
        "mape_normal_bu": mape(norm.bottom_up, norm.actual), "mape_normal_naive": mape(norm.naive, norm.actual),
        "mape_event_bu": mape(event.bottom_up, event.actual), "mape_event_naive": mape(event.naive, event.actual),
        "q4_error": float(pred.sum() / act.sum() - 1), "q4_error_naive": float(naive.sum() / act.sum() - 1),
        "q4_error_w40_50": float(pred.loc[40:50].sum() / act.loc[40:50].sum() - 1),
        "holiday_effect": hol_effect, "tier_error": seg_err, "tier_error_w40_50": seg_err50, "outlook_q1": outlook,
        "n_modelled": int(fc.merchant_id.nunique()),
        "year_end_miss": float((weekly.bottom_up / weekly.actual - 1)[weekly.week >= 51].mean())})


# ---------------------------------------------------------------- 6 Impact of the recommendations together
def ch6(mw):
    """What the two measured recommendations are worth per quarter at platform scale, valuing every
    incremental order at VALUE_PER_INCR_ORDER (and re-running the rule for other values):
    (a) roll the reorder coupon out only to segments whose cost per incremental order is within the value,
        compared with rolling it out to every customer the way it was tested;
    (b) move campaign budget toward the categories with the largest response, within the tested range.
    Delivery-time work and the 30% matching cell are upside / to test and are not added in."""
    c2 = json.loads((OUT / "ch2_coupons.json").read_text(encoding="utf-8"))
    c3 = json.loads((OUT / "ch3_elasticity.json").read_text(encoding="utf-8"))
    q4 = mw[mw.week >= 40]
    q_orders, q_gmv = float(q4.orders.sum()), float(q4.gmv_won.sum())
    cust = load("mart_reorder_customer")
    scale = q_orders / (cust.orders.sum() * 13 / 12)          # test customers' share of platform orders, quarter-adjusted
    t = c2["targeted"]
    full_cost = t["cost_all"] * 13 / 12 * scale
    full_incr = t["incr_all"] * 13 / 12 * scale
    ra = c3["realloc"]
    out = []
    for v in SENS_VALUES:
        k = [x for x in c2["sensitivity"] if x["value"] == v][0]
        saved, lost = full_cost * k["cost_cut"], full_incr * (1 - k["incr_kept"])
        out.append({"value": v, "segments": k["segments"],
                    "rows": [{"lever": "재주문 쿠폰: 전 고객 확대 대신 대상 한정", "orders": -lost, "coupon_cost": -saved, "value_won": saved - lost * v},
                             {"lever": "캠페인 예산 카테고리 재배분(실험 범위 안)", "orders": ra["orders_delta"], "coupon_cost": 0.0,
                              "value_won": ra["orders_delta"] * v}]})
        out[-1]["total_value"] = sum(r["value_won"] for r in out[-1]["rows"])
        out[-1]["total_orders"] = sum(r["orders"] for r in out[-1]["rows"])
    base = [x for x in out if x["value"] == VALUE_PER_INCR_ORDER][0]
    seg = {r["segment"]: r for r in c2["reorder"]}

    def sc(x):
        return x * 13 / 12 * scale

    # baseline 2: today (coupon only in the test) -> roll out to the segments the rule keeps
    for x in out:
        kept = [seg[k] for k in x["segments"]]
        o, c = sum(sc(r["incremental"]) for r in kept), sum(sc(r["cost"]) for r in kept)
        x["vs_today"] = {"orders": o + ra["orders_delta"], "coupon_cost": c, "value_won": (o + ra["orders_delta"]) * x["value"] - c}
    # order-recovery cost curve: if an order target requires the full-rollout order level, cheapest measured levers first
    cats = {r["category"]: r for r in c3["categories"]}
    camp = mw[mw.week.isin([44, 45])].groupby("category").agg(spend=("platform_disc_won", "sum"), gmv=("gmv_won", "sum"))
    opts = []
    for k in ["월 1회 이하", "월 2~3회"]:
        r = seg[k]
        opts.append({"lever": f"재주문 쿠폰을 {k} 고객에게도", "cost_per_order": r["cost_per_incr"], "orders": sc(r["incremental"]),
                     "note": "홀드아웃 실험"})
    for c in [x for x in cats if x not in c3["top"] + c3["bottom"]]:
        rate = camp.spend[c] / camp.gmv[c]
        extra = camp.gmv[c] * max(0.10 - rate, 0)
        mid = (rate + 0.10) / 2
        orders = extra * cats[c]["b"] / (cats[c]["aov"] * (1 + cats[c]["b"] * mid))
        if extra > 0:
            opts.append({"lever": f"{c} 캠페인 할인을 10%까지", "cost_per_order": extra / orders, "orders": orders, "note": "캠페인 셀, 분기 1회"})
    fo = c2["first_order"]
    new_q = fo["extra_per_area_week"] * 3 * 13 * 0.5            # the coupon was on half the weeks; switching it on for the rest
    per_new = 1 + fo["reorder_marginal"]                         # first order + at least one reorder within 8 weeks
    opts.append({"lever": "첫 주문 쿠폰을 모든 주에", "cost_per_order": fo["cost_per_incr_new"] / per_new, "orders": new_q * per_new,
                 "note": "지역 스위치백"})
    opts.sort(key=lambda r: r["cost_per_order"])
    gap, cum_o, cum_c = -base["total_orders"], 0.0, 0.0
    for r in opts:
        take = min(r["orders"], max(gap - cum_o, 0))
        r["used_orders"], r["cost"] = take, take * r["cost_per_order"]
        cum_o += take; cum_c += r["cost"]
        r["cum_orders"], r["cum_cost"] = cum_o, cum_c
    recovery = {"gap": gap, "options": opts, "recovered": cum_o, "cost": cum_c,
                "value_won": cum_o * VALUE_PER_INCR_ORDER - cum_c}
    dump("ch6_impact.json", {"q4": {"orders": q_orders, "gmv": q_gmv, "net": float(q4.platform_net_won.sum())}, "scale": float(scale),
                             "full_rollout": {"coupon_cost": full_cost, "incremental": full_incr, "cost_per_incr": full_cost / full_incr},
                             "base": base, "sensitivity": out, "recovery": recovery, "per_campaign": True})


def main():
    mw, kpi, h1 = load("mart_merchant_week"), load("mart_kpi_week"), load("mart_merchant_h1")
    ch0(mw, kpi)
    h = ch1(h1, mw)
    ch2()
    elas = ch3(mw)
    ch4(mw, h)
    ch5(mw, elas)
    ch6(mw)
    print("outputs written:", sorted(p.name for p in OUT.glob("*.json")))


if __name__ == "__main__":
    main()
